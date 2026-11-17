"""Fast CPU unit tests for `wsi/tracer.py` (block quantization, CompactTrace, and trace_all `diff_traces`)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi import (  # noqa: E402
    WeightSpaceIndex,
    capture_compact,
    CompactTrace,
    dequantize_block_int4,
    diff_traces,
    quantization_error,
    quantize_block_int4,
)
from wsi.tracer import dequantize_fp8_or_int8, quantize_fp8_or_int8  # noqa: E402


class _TinyAttn(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.num_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        nh, hd = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = torch.softmax(scores + mask, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, H)
        return self.o_proj(ctx), attn


class _TinyMLP(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyLayer(nn.Module):
    def __init__(self, hidden: int, inter: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = _TinyAttn(hidden, n_heads)
        self.mlp = _TinyMLP(hidden, inter)

    def forward(self, x: torch.Tensor):
        ao, att = self.self_attn(x)
        x = x + ao
        x = x + self.mlp(x)
        return x, att


class _TinyCausalModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 32, inter: int = 64, n_layers: int = 3, n_heads: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", hidden_size=hidden, num_attention_heads=n_heads)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_TinyLayer(hidden, inter, n_heads) for _ in range(n_layers)])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False,
                output_attentions: bool = False, use_cache: bool = False):
        h = self.model.embed_tokens(input_ids)
        hs = [h] if output_hidden_states else None
        atts = [] if output_attentions else None
        for lyr in self.model.layers:
            h, att = lyr(h)
            if hs is not None:
                hs.append(h)
            if atts is not None:
                atts.append(att)
        logits = self.lm_head(self.model.norm(h))
        return SimpleNamespace(logits=logits, hidden_states=tuple(hs) if hs else None,
                               attentions=tuple(atts) if atts else None, past_key_values=None)


def test_block_int4_isolates_outliers() -> None:
    torch.manual_seed(0)
    x = torch.randn(40, 128) * 0.25
    # Inject extreme activation outliers in group 0 (like real Transformer residual spikes)
    x[:, 3] = 250.0
    packed, sc, shp = quantize_block_int4(x, group_size=16)
    x_hat = dequantize_block_int4(packed, sc, shp, group_size=16)
    # Non-outlier blocks [16:] stay accurate even with a 1000x spike in block 0
    err_rest = quantization_error(x[:, 16:], x_hat[:, 16:])
    assert err_rest["cosine"] > 0.99 and err_rest["rel_l2"] < 0.10
    raw_bytes = x.numel() * 4
    q_bytes = packed.numel() + sc.numel() * 2
    assert raw_bytes / q_bytes > 6.0

    q8, sc8, _ = quantize_fp8_or_int8(x, group_size=16)
    x_hat8 = dequantize_fp8_or_int8(q8, sc8, shp, group_size=16)
    assert quantization_error(x, x_hat8)["cosine"] > 0.9999


def test_compact_trace_roundtrip_and_loop_onset() -> None:
    torch.manual_seed(1)
    model = _TinyCausalModel(vocab=32, hidden=32, inter=64, n_layers=3, n_heads=4).eval()
    # Non-repeating prefix followed by a periodic repeating loop at t >= 12
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8, 5, 6, 7, 8]])
    with tempfile.TemporaryDirectory() as td:
        tr8 = capture_compact(model, ids, top_k_mlp=16, group_size=16)  # default: int8
        assert tr8.resid_quant == "int8"
        tr = capture_compact(model, ids, top_k_mlp=16, resid_quant="int4", group_size=16)
        tr.save(td)
        loaded = CompactTrace.load(td)
        assert loaded.seq_len == 20 and loaded.size_bytes() > 0
        onset = loaded.detect_loop_onset(layer=2, window=4, thr=0.95)
        assert onset["onset_token"] is not None and onset["onset_token"] >= 12
        comp = loaded.compare_positions(t_target=19, t_ref=2, top_k=4)
        assert len(comp["top_mlp_shifts"]) == 4 and len(comp["top_head_shifts"]) == 4


def test_diff_traces_pinpoints_planted_circuit_with_wsi() -> None:
    torch.manual_seed(2)
    model = _TinyCausalModel(vocab=32, hidden=32, inter=64, n_layers=3, n_heads=4).eval()
    with torch.no_grad():
        # Zero all MLP & attention weights cleanly so only one planted trigger neuron & head fire strongly
        for p in model.parameters():
            p.zero_()
        # Distinguish token 15 (trigger) vs token 10 (neutral) along dim 0
        model.model.embed_tokens.weight[10, 0] = -1.0
        model.model.embed_tokens.weight[15, 0] = 5.0
        model.model.embed_tokens.weight[:, 1:] = 0.05 * torch.randn(32, 31)
        # Layer 1 Attention Head 2 pushes dim 1 on trigger token
        model.model.layers[1].self_attn.v_proj.weight[16:24, 0] = 2.0
        model.model.layers[1].self_attn.o_proj.weight[1, 16:24] = 1.5
        # Layer 2 MLP Neuron 7 fires only when dim 1 is large (written by L1 Head 2) and writes "token 20"
        model.model.layers[2].mlp.gate_proj.weight[7, 1] = 2.0
        model.model.layers[2].mlp.up_proj.weight[7, 1] = 2.0
        model.lm_head.weight[20] = torch.eye(32)[5] * 4.0
        model.model.layers[2].mlp.down_proj.weight[:, 7] = model.lm_head.weight[20].clone()

        # Save minimal safetensors checkpoint for WeightSpaceIndex
        with tempfile.TemporaryDirectory() as td:
            mpath = Path(td)
            save_file(model.state_dict(), str(mpath / "model.safetensors"))
            (mpath / "config.json").write_text(json.dumps({
                "model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 3, "num_attention_heads": 4, "vocab_size": 32,
            }))
            tok_map = {f"tok_{i}": i for i in range(32)}
            tok_map["TARGET_OUT"] = 20
            (mpath / "tokenizer.json").write_text(json.dumps({"model": {"vocab": tok_map}}))

            I = WeightSpaceIndex.from_model(mpath)
            # Unequal lengths (like pushback turn vs neutral turn)
            ids_a = [1, 2, 10]
            ids_b = [1, 2, 3, 4, 15]
            diff = diff_traces(model, ids_a, ids_b, wsi=I, pos_a=-1, pos_b=-1, top_k=5)
            assert diff["len_a"] == 3 and diff["len_b"] == 5
            top_mlp = diff["top_mlp_contributors"][0]
            top_head = diff["top_attention_heads"][0]
            assert (top_mlp["layer"], top_mlp["unit"]) == (2, 7)
            assert (top_head["layer"], top_head["head"]) == (1, 2)
            assert diff["peak_vector_readout"] is not None
            promoted = [x["token"] for x in diff["peak_vector_readout"]["promoted"]]
            assert "TARGET_OUT" in promoted


if __name__ == "__main__":
    test_block_int4_isolates_outliers()
    test_compact_trace_roundtrip_and_loop_onset()
    test_diff_traces_pinpoints_planted_circuit_with_wsi()
    print("ALL TRACER TESTS PASSED!")
