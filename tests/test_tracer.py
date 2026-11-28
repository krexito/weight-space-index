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
    Edit,
    MoIPatch,
    WeightSpaceIndex,
    capture_compact,
    CompactTrace,
    dequantize_block_fp4,
    dequantize_block_int4,
    dequantize_block_ternary,
    diff_compact,
    diff_traces,
    discover_compact_circuit,
    gradient_screen_candidates,
    hadamard_transform,
    quantization_error,
    quantize_block_fp4,
    quantize_block_int4,
    quantize_block_ternary,
    resolve_mlp_idx_mode,
    screen_patch_candidates,
)
from wsi.tracer import (  # noqa: E402
    dequantize_fp8_or_int8,
    pack_bitmask_u8,
    pack_bitpack_u8,
    quantize_fp8_or_int8,
    unpack_bitmask_u8,
    unpack_bitpack_u8,
)


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
        self._last_kv = (k.detach(), v.detach())
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
        pkv = None
        if use_cache:
            ks = [lyr.self_attn._last_kv[0] for lyr in self.model.layers]
            vs = [lyr.self_attn._last_kv[1] for lyr in self.model.layers]
            rec = [h[:, :1, :16].detach()]
            pkv = SimpleNamespace(key_cache=ks, value_cache=vs, recurrent_states=rec)
        return SimpleNamespace(logits=logits, hidden_states=tuple(hs) if hs else None,
                               attentions=tuple(atts) if atts else None, past_key_values=pkv)


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

    # FP4 E2M1 + Ternary: Hadamard rotation improves large-group (group_size=64) outlier tolerance
    x_out = torch.randn(40, 128)
    x_out[:, 3] = 15.0
    x_out[:, 17] = -12.0
    xh = hadamard_transform(x_out, block=64)
    assert quantization_error(x_out, hadamard_transform(xh, block=64))["rel_l2"] < 1e-5
    p_fp4, s_fp4, _ = quantize_block_fp4(xh, group_size=64)
    xh_fp4 = hadamard_transform(dequantize_block_fp4(p_fp4, s_fp4, shp, group_size=64), block=64)
    e_raw_fp4 = quantization_error(x_out, dequantize_block_fp4(*quantize_block_fp4(x_out, group_size=64), group_size=64))
    e_had_fp4 = quantization_error(x_out, xh_fp4)
    assert e_had_fp4["rel_l2"] < e_raw_fp4["rel_l2"] and e_had_fp4["cosine"] > 0.99

    p_ter, s_ter, _ = quantize_block_ternary(xh, group_size=64)
    xh_ter = hadamard_transform(dequantize_block_ternary(p_ter, s_ter, shp, group_size=64), block=64)
    e_raw_ter = quantization_error(x_out, dequantize_block_ternary(*quantize_block_ternary(x_out, group_size=64), group_size=64))
    e_had_ter = quantization_error(x_out, xh_ter)
    assert e_had_ter["rel_l2"] < e_raw_ter["rel_l2"] and e_had_ter["cosine"] > 0.90
    # Verify 5-trits-per-byte ternary packed buffer is strictly smaller than 4-bit packed buffer
    assert p_ter.numel() < p_fp4.numel()


def test_compact_trace_roundtrip_and_loop_onset() -> None:
    torch.manual_seed(1)
    # Direct lossless tests for bitpack (12-bit / 14-bit) & bitmask + auto crossing rules
    for d_inter, k in ((3584, 17), (3584, 256), (9216, 17)):
        bits = (d_inter - 1).bit_length()
        perm = torch.stack([torch.randperm(d_inter)[:k] for _ in range(7)])
        p_bp = pack_bitpack_u8(perm, bits)
        assert torch.equal(unpack_bitpack_u8(p_bp, k, bits), perm.long())
        p_bm = pack_bitmask_u8(perm, d_inter)
        assert torch.equal(unpack_bitmask_u8(p_bm, k, d_inter), torch.sort(perm.long(), dim=1).values)
    assert resolve_mlp_idx_mode(256, 3584, "auto") == "bitpack"
    assert resolve_mlp_idx_mode(512, 3584, "auto") == "bitmask"
    assert resolve_mlp_idx_mode(512, 9216, "auto") == "bitpack"
    assert resolve_mlp_idx_mode(1024, 9216, "auto") == "bitmask"

    model = _TinyCausalModel(vocab=32, hidden=32, inter=64, n_layers=3, n_heads=4).eval()
    # Non-repeating prefix followed by a periodic repeating loop at t >= 12
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8, 5, 6, 7, 8]])
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Pareto-optimal defaults: int8+g64+had64 resid, auto Top-K (95% energy) fp4+bitmask/auto MLP (g16), int8 heads, Top-16 attn, int8 cache
        tr_def = capture_compact(model, ids)
        assert tr_def.resid_quant == "int8" and tr_def.group_size == 64 and tr_def.hadamard_block == 64
        assert tr_def.mlp_quant == "fp4" and tr_def.mlp_group_size == 16 and 0 < tr_def.top_k_mlp <= 64
        assert tr_def.head_quant == "int8"
        assert tr_def.meta["top_k_mlp_mode"] == "auto" and tr_def.meta["mlp_energy_target"] == 0.95
        assert tr_def.meta["mlp_mean_energy_frac"] >= 0.95
        assert all(m["mean_energy_frac"] >= 0.95 for m in tr_def.mlp_meta.values())
        assert tr_def.mlp_idx_mode == "bitmask" and tr_def.top_k_attn == 16
        assert len(tr_def.attn_topk) == 3 and "key_cache.0" in tr_def.cache and "recurrent_states.0" in tr_def.cache
        bd = tr_def.byte_breakdown()
        assert bd["total"] == tr_def.size_bytes() and bd["attn_topk"] > 0 and bd["cache"] > 0
        tr_def.save(root / "default")
        loaded_def = CompactTrace.load(root / "default")
        assert loaded_def.head_quant == "int8"
        assert loaded_def.byte_breakdown() == bd and loaded_def.meta == tr_def.meta
        first_head_mod = next(iter(loaded_def.head_norms))
        h_a, h_b = tr_def.dequant_head_in(first_head_mod), loaded_def.dequant_head_in(first_head_mod)
        assert h_a is not None and h_b is not None and torch.equal(h_a, h_b)
        tr_norm = capture_compact(model, ids, head_quant="norm")
        assert tr_norm.head_quant == "norm" and tr_norm.dequant_head_in(first_head_mod) is None
        assert tr_norm.size_bytes() < tr_def.size_bytes()
        tr_80 = capture_compact(model, ids, top_k_mlp=0.80)
        tr_99 = capture_compact(model, ids, mlp_energy_target=0.99)
        assert tr_80.meta["top_k_mlp_mode"] == "auto" and tr_80.meta["mlp_energy_target"] == 0.80
        assert tr_80.meta["mlp_mean_energy_frac"] >= 0.80 and tr_99.meta["mlp_mean_energy_frac"] >= 0.99
        assert tr_80.top_k_mlp <= tr_def.top_k_mlp <= tr_99.top_k_mlp
        assert tr_80.size_bytes() <= tr_def.size_bytes() <= tr_99.size_bytes()
        with torch.no_grad():
            ref_out = model(ids, output_attentions=True, use_cache=True)
        a_idx, a_w = loaded_def.dequant_attn_topk(1)
        assert a_idx.shape == (4, 20, 16) and float((ref_out.attentions[1][0].topk(16, dim=-1).values - a_w).abs().max()) < 0.005
        assert quantization_error(ref_out.past_key_values.key_cache[0], loaded_def.dequant_cache("key_cache.0"))["cosine"] > 0.9999
        assert quantization_error(ref_out.past_key_values.recurrent_states[0], loaded_def.dequant_cache("recurrent_states.0"))["cosine"] > 0.9999
        cmp_def = loaded_def.compare_positions(t_target=19, t_ref=2, top_k=4)
        assert "top_attn_shifts" in cmp_def and len(cmp_def["top_attn_shifts"]) == 4

        # Explicit baseline: uint16 + fp16 MLP without attn/cache
        tr_u16 = capture_compact(model, ids, top_k_mlp=16, mlp_quant="fp16", mlp_idx_mode="uint16",
                                 group_size=16, hadamard_block=0, top_k_attn=0, capture_cache=False)
        assert tr_u16.resid_quant == "int8" and tr_u16.mlp_topk[0]["idx"].dtype == torch.uint16

        # Auto + int8 MLP (inter=64, top_k=16 -> bitmask beats bitpack/uint16, strictly smaller!)
        tr_auto = capture_compact(model, ids, top_k_mlp=16, mlp_quant="int8", mlp_idx_mode="auto",
                                  group_size=16, hadamard_block=0, top_k_attn=0, capture_cache=False)
        assert tr_auto.resid_quant == "int8" and tr_auto.mlp_quant == "int8" and tr_auto.mlp_idx_mode == "bitmask"
        assert tr_auto.size_bytes() < tr_u16.size_bytes()
        tr_auto.save(root / "auto")
        loaded_auto = CompactTrace.load(root / "auto")
        idx_u, val_u = tr_u16.dequant_mlp_topk(0)
        idx_a, val_a = loaded_auto.dequant_mlp_topk(0)
        assert torch.equal(torch.sort(idx_u, dim=1).values, idx_a)
        assert quantization_error(val_u.gather(1, idx_u.argsort(dim=1)), val_a)["cosine"] > 0.9999

        # Explicit bitpack + fp4 MLP
        tr_bp = capture_compact(model, ids, top_k_mlp=16, mlp_quant="fp4", mlp_idx_mode="bitpack", group_size=16)
        assert tr_bp.mlp_idx_mode == "bitpack" and "idx_pack" in tr_bp.mlp_topk[0]
        tr_bp.save(root / "bitpack")
        loaded_bp = CompactTrace.load(root / "bitpack")
        idx_b, val_b = loaded_bp.dequant_mlp_topk(0)
        assert torch.equal(idx_b, idx_u) and quantization_error(val_u, val_b)["cosine"] > 0.98
        assert len(loaded_bp.compare_positions(t_target=19, t_ref=2, top_k=4)["top_mlp_shifts"]) == 4

        tr_ter = capture_compact(model, ids, top_k_mlp=16, resid_quant="ternary", hadamard_block=32, group_size=16)
        assert tr_ter.hadamard_block == 32
        tr_ter.save(root / "ter")
        assert CompactTrace.load(root / "ter").hadamard_block == 32

        tr = capture_compact(model, ids, top_k_mlp=16, resid_quant="int4", group_size=16)
        tr.save(root / "main")
        loaded = CompactTrace.load(root / "main")
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
            dense_diff = diff_traces(model, ids_a, ids_b, wsi=I, pos_a=-1, pos_b=-1, top_k=5, compact=False)
            for res in (diff, dense_diff):
                assert res["len_a"] == 3 and res["len_b"] == 5
                top_mlp = res["top_mlp_contributors"][0]
                top_head = res["top_attention_heads"][0]
                assert (top_mlp["layer"], top_mlp["unit"]) == (2, 7)
                assert (top_head["layer"], top_head["head"]) == (1, 2)
                assert res["peak_vector_readout"] is not None
                assert "TARGET_OUT" in [x["token"] for x in res["peak_vector_readout"]["promoted"]]

            cdiff = diff_compact(capture_compact(model, ids_a), capture_compact(model, ids_b),
                                 wsi=I, pos_a=-1, pos_b=-1, top_k=5)
            assert cdiff["len_a"] == 3 and cdiff["len_b"] == 5
            assert (cdiff["top_mlp_contributors"][0]["layer"], cdiff["top_mlp_contributors"][0]["unit"]) == (2, 7)
            assert (cdiff["top_attention_heads"][0]["layer"], cdiff["top_attention_heads"][0]["head"]) == (1, 2)
            assert abs(cdiff["top_attention_heads"][0]["out_delta_norm"] - dense_diff["top_attention_heads"][0]["out_delta_norm"]) < 0.05
            assert "attn_b_top_src_pos" in cdiff["top_attention_heads"][0]
            assert len(cdiff["top_cache_deltas"]) > 0
            assert "TARGET_OUT" in [x["token"] for x in cdiff["peak_vector_readout"]["promoted"]]


def test_hybrid_attn_topk_layer_mapping() -> None:
    class _HybridDummyBlock(nn.Module):
        def __init__(self, hidden: int, inter: int, has_full_attn: bool) -> None:
            super().__init__()
            if has_full_attn:
                self.self_attn = _TinyAttn(hidden, n_heads=4)
            else:
                self.linear_attn = _TinyAttn(hidden, n_heads=4)
            self.mlp = _TinyMLP(hidden, inter)

        def forward(self, x: torch.Tensor):
            ao, att = (self.self_attn if hasattr(self, "self_attn") else self.linear_attn)(x)
            return x + ao + self.mlp(x + ao), att

    class _HybridDummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5_text", hidden_size=32, num_attention_heads=4)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(32, 32)
            # Only layers 1 and 3 are full `self_attn`; layers 0 and 2 are `linear_attn`
            self.model.layers = nn.ModuleList([_HybridDummyBlock(32, 64, has_full_attn=(i in (1, 3))) for i in range(4)])

        def forward(self, input_ids: torch.Tensor, output_attentions: bool = False, use_cache: bool = False):
            h = self.model.embed_tokens(input_ids)
            full_atts = []
            for i, lyr in enumerate(self.model.layers):
                h, att = lyr(h)
                if output_attentions and i in (1, 3):
                    full_atts.append(att)
            return SimpleNamespace(logits=h, attentions=tuple(full_atts) if output_attentions else None, past_key_values=None)

    m = _HybridDummyModel().eval()
    ct = capture_compact(m, torch.tensor([[1, 2, 3, 4, 5]]))
    assert sorted(ct.attn_topk) == [1, 3]


def test_discover_compact_circuit() -> None:
    torch.manual_seed(7)
    m = _TinyCausalModel(vocab=32, hidden=32, inter=64, n_layers=4, n_heads=4).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        # Separate healthy token 10 vs looping cue token 15 along dim 0 & 1
        m.model.embed_tokens.weight[10, 0] = -1.5
        m.model.embed_tokens.weight[15, 0] = 4.0
        m.model.embed_tokens.weight[:, 2:] = 0.05 * torch.randn(32, 30)
        # Small non-zero weights on all layers so every layer has active MLP neurons and non-zero SVD subspaces
        for l, blk in enumerate(m.model.layers):
            blk.self_attn.v_proj.weight[:8, 0] = 0.3 + 0.1 * l
            blk.self_attn.o_proj.weight[1, :8] = 0.3 + 0.1 * l
            blk.mlp.gate_proj.weight[:, 0] = 0.1
            blk.mlp.up_proj.weight[:, 1] = 0.2
            blk.mlp.down_proj.weight[:, :8] = 0.1 * torch.randn(32, 8)
            blk.mlp.down_proj.weight[l, 0] = 0.8
        # Strongly resonate layer 1 (origin) and layer 2 (downstream echo) along dim 5
        for l in (1, 2):
            m.model.layers[l].mlp.gate_proj.weight[7:15, 0] = 2.0
            m.model.layers[l].mlp.up_proj.weight[7:15, 1] = 2.0
            m.model.layers[l].mlp.down_proj.weight[5, 7:15] = 3.0
        m.lm_head.weight[20, 5] = 4.0

        ids_h = [1, 2, 10, 10, 10, 10, 10, 10]
        ids_l = [1, 2, 10, 10, 15, 15, 15, 15, 15, 15]
        ct_h = capture_compact(m, ids_h)
        ct_l = capture_compact(m, ids_l)

        tuples = [
            ({"prompt_len": 2, "loop": False, "onset_resp": None}, ct_h, None),
            ({"prompt_len": 2, "loop": True, "onset_resp": 2}, ct_l, None),
        ]
        res_model = discover_compact_circuit(tuples, model=m, alpha=1.5, top_k_neurons=8)
        assert isinstance(res_model["patch"], MoIPatch)
        assert set(["unified_top2", "blind_top2", "attn_top2", "wsi_top3", "wsi_fa3", "wsi_pair", "mlp_top3", "mlp_fa3"]).issubset(res_model["patches"])
        assert len(res_model["wsi_pair_layers"]) == 2 and res_model["wsi_resonant_dst"] is not None
        assert res_model["wsi_origins"] == res_model["hybrid_ranked"][:2]
        assert res_model["split_mode"] in ("token_ab", "sequence_ab")
        assert len(res_model["attn_stats"]) == 4 and len(res_model["site_rankings"]) == 8
        assert res_model["router_report"] is not None and res_model["patch"].router is not None
        assert res_model["router_report"]["balanced_acc"] >= 0.9
        # Verify synthesized router fires on target sequence and stays off on healthy sequence
        out_h = m(torch.tensor([ids_h]), output_hidden_states=True)
        out_l = m(torch.tensor([ids_l]), output_hidden_states=True)
        assert bool(res_model["patch"].fires(out_h.hidden_states).item()) is False
        assert bool(res_model["patch"].fires(out_l.hidden_states).item()) is True
        # Verify attention-head patch attaches and restores cleanly
        ha = res_model["patches"]["attn_top2"].attach(m)
        ha.on()
        assert not ha.check_restored()
        ha.off()
        assert ha.check_restored() and ha.fingerprint_ok()
        # Verify fast 1-forward-pass proxy screening (`screen_patch_candidates`)
        screened = screen_patch_candidates(
            m,
            {"unified": res_model["patches"]["unified_top2"], "blind": res_model["patches"]["blind_top2"], "attn": res_model["patches"]["attn_top2"]},
            healthy_seqs=[ids_h],
            target_seqs=[ids_l],
            target_spans=[(4, 10)],
        )
        assert len(screened) == 3 and all(
            "delta_healthy_nll" in r and "healthy_kl" in r and "pareto_kl" in r and "target_suppression" in r and r["healthy_kl"] >= 0.0
            for r in screened
        )
        screened_kl = screen_patch_candidates(
            m,
            {"unified": res_model["patches"]["unified_top2"], "blind": res_model["patches"]["blind_top2"]},
            healthy_seqs=[ids_h],
            target_seqs=[ids_l],
            target_spans=[(4, 10)],
            rank_by="pareto_kl",
        )
        assert len(screened_kl) == 2 and all(r["healthy_kl"] >= 0.0 for r in screened_kl)
        hp = res_model["patch"].attach(m)
        hp.on()
        assert not hp.check_restored()
        hp.off()
        assert hp.check_restored() and hp.fingerprint_ok()

        with tempfile.TemporaryDirectory() as td:
            mpath = Path(td)
            save_file(m.state_dict(), str(mpath / "model.safetensors"))
            (mpath / "config.json").write_text(json.dumps({
                "model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 4, "num_attention_heads": 4, "vocab_size": 32,
            }))
            (mpath / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {f"tok_{i}": i for i in range(32)}}}))
            I = WeightSpaceIndex.from_model(mpath)

            res_wsi = I.discover_compact_circuit([(ct_h, 2, None), (ct_l, 2, 2)], alpha=1.5, top_k_neurons=8)
            assert res_wsi["mlp_ranked"] == res_model["mlp_ranked"]
            assert res_wsi["hybrid_ranked"] == res_model["hybrid_ranked"]
            assert res_wsi["wsi_pair_layers"] == res_model["wsi_pair_layers"]
            assert res_wsi["wsi_fa3_layers"] == res_model["wsi_fa3_layers"]
            hw = res_wsi["patch"].attach(m)
            hw.on()
            assert not hw.check_restored()
            hw.off()
            assert hw.check_restored() and hw.fingerprint_ok()

            # Also accept raw CompactTrace lists via `traces` + `loop_traces`
            res_raw = I.discover_compact_circuit([ct_h], loop_traces=[ct_l], alpha=1.5, top_k_neurons=8)
            assert isinstance(res_raw["patch"], MoIPatch) and len(res_raw["layer_stats"]) > 0
            cd = I.diff_compact(ct_h, ct_l, pos_a=-1, pos_b=-1, top_k=3)
            assert len(cd["top_mlp_contributors"]) == 3

            # Verify CLI (`wsi.cli.main`) and MCP (`wsi.mcp_server.handle_rpc_message`) integration
            ct_h_dir = mpath / "ct_h"
            ct_l_dir = mpath / "ct_l"
            ct_h.save(ct_h_dir)
            ct_l.save(ct_l_dir)

            import contextlib
            import io
            from wsi.cli import main as cli_main
            from wsi.mcp_server import handle_rpc_message

            cli_patch = mpath / "cli_fix.moipatch"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_dc = cli_main(["diff-compact", str(mpath), str(ct_h_dir), str(ct_l_dir), "--top-k", "3"])
            assert rc_dc == 0
            cli_dc = json.loads(buf.getvalue())
            assert len(cli_dc["top_mlp_contributors"]) == 3

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_disc = cli_main([
                    "discover-circuit", str(mpath),
                    "--trace", str(ct_h_dir),
                    "--loop-trace", str(ct_l_dir),
                    "--alpha", "1.5",
                    "--top-k-neurons", "8",
                    "-o", str(cli_patch),
                ])
            assert rc_disc == 0 and cli_patch.exists()
            cli_disc = json.loads(buf.getvalue())
            assert cli_disc["saved_patch"] == str(cli_patch)
            assert MoIPatch.load(cli_patch).attach(m).fingerprint_ok()

            list_resp = handle_rpc_message({"jsonrpc": "2.0", "id": 10, "method": "tools/list"})
            t_names = {t["name"] for t in list_resp["result"]["tools"]}
            assert {
                "wsi_diff_compact",
                "wsi_discover_compact_circuit",
                "wsi_build_moipatch",
                "wsi_screen_candidates",
                "wsi_capture_causal_traces",
                "wsi_attribute_direction_writers",
                "wsi_trace_direction_graph",
                "wsi_peel_causal_circuit",
            }.issubset(t_names)
            assert len(t_names) == 26  # the full list is pinned in tests/test_mcp_server.py

            mcp_dc = handle_rpc_message({
                "jsonrpc": "2.0", "id": 11, "method": "tools/call",
                "params": {"name": "wsi_diff_compact", "arguments": {
                    "model_dir": str(mpath), "trace_a_dir": str(ct_h_dir), "trace_b_dir": str(ct_l_dir), "top_k": 3,
                }},
            })
            assert not mcp_dc["result"]["isError"]
            assert len(json.loads(mcp_dc["result"]["content"][0]["text"])["top_mlp_contributors"]) == 3

            mcp_patch = mpath / "mcp_fix.moipatch"
            mcp_disc = handle_rpc_message({
                "jsonrpc": "2.0", "id": 12, "method": "tools/call",
                "params": {"name": "wsi_discover_compact_circuit", "arguments": {
                    "model_dir": str(mpath),
                    "trace_dirs": [str(ct_h_dir)],
                    "loop_trace_dirs": [str(ct_l_dir)],
                    "alpha": 1.5,
                    "top_k_neurons": 8,
                    "save_patch_path": str(mcp_patch),
                }},
            })
            assert not mcp_disc["result"]["isError"] and mcp_patch.exists()
            mcp_payload = json.loads(mcp_disc["result"]["content"][0]["text"])
            assert mcp_payload["saved_patch"] == str(mcp_patch)
            assert MoIPatch.load(mcp_patch).attach(m).fingerprint_ok()


def test_gradient_screen_candidates() -> None:
    torch.manual_seed(11)
    m = _TinyCausalModel(vocab=32, hidden=32, inter=64, n_layers=4, n_heads=4).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(mean=0.0, std=0.15)

    # Preserve mixed requires_grad state across the gradient call
    for p in m.parameters():
        p.requires_grad_(False)
    m.lm_head.weight.requires_grad_(True)

    ids_h = [1, 2, 10, 11, 12, 13, 14, 15]
    ids_l = [1, 2, 10, 11, 20, 20, 20, 20, 20, 20]
    spans = [(4, 10)]

    u_mlp = torch.randn(32)
    u_mlp = u_mlp / u_mlp.norm()
    u_attn = torch.randn(32)
    u_attn = u_attn / u_attn.norm()

    # Tiny alpha=5e-3 keeps patches in the linear regime for exact finite-difference validation
    tiny_cands = {
        "mlp_sub": MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=u_mlp, alpha=5e-3, idx=torch.arange(16))]),
        "mlp_multi": MoIPatch(edits=[
            Edit(module="layers.1.mlp.down_proj", u=u_mlp, alpha=5e-3),
            Edit(module="layers.2.mlp.down_proj", u=u_attn, alpha=5e-3),
        ]),
        "attn_o": MoIPatch(edits=[Edit(module="layers.1.self_attn.o_proj", u=u_attn, alpha=5e-3)]),
    }
    grad_rows = gradient_screen_candidates(m, tiny_cands, healthy_seqs=[ids_h], target_seqs=[ids_l], target_spans=spans)
    assert m.lm_head.weight.requires_grad is True
    assert m.model.layers[1].mlp.down_proj.weight.requires_grad is False

    fwd_rows = screen_patch_candidates(m, tiny_cands, healthy_seqs=[ids_h], target_seqs=[ids_l], target_spans=spans)
    g_by = {r["name"]: r for r in grad_rows}
    f_by = {r["name"]: r for r in fwd_rows}
    for name in tiny_cands:
        assert abs(g_by[name]["grad_supp"] - f_by[name]["target_suppression"]) < 2e-4, (name, g_by[name], f_by[name])
        assert abs(g_by[name]["grad_dnll"] - f_by[name]["delta_healthy_nll"]) < 2e-4, (name, g_by[name], f_by[name])

    # Verify 2-stage funnel (`prefilter_top_k=2`) narrows 4 candidates to 2 finalists and attaches both grad & forward fields
    stage_cands = {
        "c1": MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=u_mlp, alpha=1.0, idx=torch.arange(16))]),
        "c2": MoIPatch(edits=[Edit(module="layers.2.mlp.down_proj", u=u_attn, alpha=1.0, idx=torch.arange(16))]),
        "c3": MoIPatch(edits=[Edit(module="layers.1.self_attn.o_proj", u=u_attn, alpha=1.0)]),
        "c4": MoIPatch(edits=[Edit(module="layers.2.self_attn.o_proj", u=u_mlp, alpha=1.0)]),
    }
    prefiltered = screen_patch_candidates(
        m, stage_cands, healthy_seqs=[ids_h], target_seqs=[ids_l], target_spans=spans, prefilter_top_k=2
    )
    assert len(prefiltered) == 2
    for row in prefiltered:
        assert {"delta_healthy_nll", "healthy_kl", "pareto_score", "pareto_kl", "grad_supp", "grad_dnll", "grad_pareto"}.issubset(row)


if __name__ == "__main__":
    test_block_int4_isolates_outliers()
    test_compact_trace_roundtrip_and_loop_onset()
    test_diff_traces_pinpoints_planted_circuit_with_wsi()
    test_hybrid_attn_topk_layer_mapping()
    test_discover_compact_circuit()
    test_gradient_screen_candidates()
    print("ALL TRACER TESTS PASSED!")
