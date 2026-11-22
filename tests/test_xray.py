"""CPU-only unit tests for the WSI Model X-Ray scanner (`wsi/xray.py`, `wsi/xray_probes.py`, `scripts/scan_xray_atlas.py`)."""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_refusal_blind as ev  # noqa: E402
import scan_xray_atlas as sxa  # noqa: E402
from wsi.router_cal import CALIBRATION_FIT_WRAPPERS, CALIBRATION_VAL_WRAPPERS  # noqa: E402
from wsi.xray import (  # noqa: E402
    DEPTH_BINS,
    XRAY_SCHEMA_VERSION,
    _effective_rank_and_top1,
    _gini,
    _oof_linear_probe,
    _scan_over_refusal,
    format_compact_xray_json,
    load_xray_json,
    save_xray_json,
    scan_model_xray,
    summarize_xray_atlas,
)
from wsi.xray_probes import (  # noqa: E402
    FRAMING_WRAPPERS,
    OR_BENCH_HARD_SLICE,
    OR_BENCH_TOXIC_SLICE,
    build_frozen_probe_pack,
)


# ----------------------------------------------------------------------------------------------- tiny toy tokenizer & models
def _build_toy_tokenizer(*, with_chat_template: bool = True):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab_words = [
        "<pad>", "<eos>", "<unk>", "user:", "assistant:",
        "A", "B", "C", "D", "Paris", "Tokyo", "Jupiter", "England", "Au", "Celsius", "pound", "China",
        "The", "city", "of", "What", "is", "2", "+", "?", "Are", "you", "sure",
        "For", "graduate", "textbook", "chapter", "Wait", "remainder", "Kutuzov", "artillery",
    ] + [f"w{i}" for i in range(60)]
    vocab = {w: i for i, w in enumerate(vocab_words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    if with_chat_template:
        tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}assistant:"
    return tok


def _build_tiny_llama(vocab_size: int, *, n_layers: int = 3, hidden: int = 24, inter: int = 48):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    return LlamaForCausalLM(cfg).eval()


class _DenseGQABlock(nn.Module):
    def __init__(self, d: int, inter: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d)
        self.post_attention_layernorm = nn.RMSNorm(d)

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = nn.Linear(d, d, bias=False)

            def forward(self, x):
                return self.o_proj(x)

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(d, inter, bias=False)
                self.up_proj = nn.Linear(d, inter, bias=False)
                self.down_proj = nn.Linear(inter, d, bias=False)

            def forward(self, x):
                return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        self.self_attn = _Attn()
        self.mlp = _MLP()

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _FusedHybridBlock(nn.Module):
    def __init__(self, d: int, inter: int, use_conv: bool):
        super().__init__()
        self.operator_norm = nn.RMSNorm(d)
        self.post_operator_norm = nn.RMSNorm(d)
        self.ffn_norm = nn.RMSNorm(d)
        self.post_ffn_norm = nn.RMSNorm(d)

        class _Mixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.out_proj = nn.Linear(d, d, bias=False)

            def forward(self, x):
                return self.out_proj(x)

        class _OProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = nn.Linear(d, d, bias=False)

            def forward(self, x):
                return self.o_proj(x)

        class _FusedMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Linear(d, 2 * inter, bias=False)
                self.down_proj = nn.Linear(inter, d, bias=False)

            def forward(self, x):
                g, u = self.gate_up_proj(x).chunk(2, dim=-1)
                return self.down_proj(F.silu(g) * u)

        if use_conv:
            self.conv = _Mixer()
        else:
            self.self_attn = _OProj()
        self.mlp = _FusedMLP()
        self.use_conv = use_conv

    def forward(self, x):
        op = self.conv(self.operator_norm(x)) if self.use_conv else self.self_attn(self.operator_norm(x))
        x = x + self.post_operator_norm(op)
        x = x + self.post_ffn_norm(self.mlp(self.ffn_norm(x)))
        return x


class _SparseMoEBlock(nn.Module):
    def __init__(self, d: int, inter: int, n_experts: int = 4):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d)
        self.post_attention_layernorm = nn.RMSNorm(d)

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = nn.Linear(d, d, bias=False)

            def forward(self, x):
                return self.o_proj(x)

        class _Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = nn.Linear(inter, d, bias=False)
                self.up_proj = nn.Linear(d, inter, bias=False)

            def forward(self, x):
                return self.down_proj(F.silu(self.up_proj(x)))

        class _MoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(d, n_experts, bias=False)
                self.experts = nn.ModuleList([_Expert() for _ in range(n_experts)])

            def forward(self, x):
                probs = torch.softmax(self.gate(x), dim=-1)
                out = torch.zeros_like(x)
                for idx, exp in enumerate(self.experts):
                    out = out + probs[..., idx : idx + 1] * exp(x)
                return out

        self.self_attn = _Attn()
        self.mlp = _MoE()

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _PureSSMBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.RMSNorm(d)

        class _Mixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.out_proj = nn.Linear(d, d, bias=False)

            def forward(self, x):
                return self.out_proj(x)

        self.mixer = _Mixer()

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class _ToyArchModel(nn.Module):
    def __init__(self, arch: str, vocab_size: int, n_layers: int = 3, d: int = 24, inter: int = 48):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=arch,
            hidden_size=d,
            num_hidden_layers=n_layers,
            num_attention_heads=4,
            num_key_value_heads=2 if arch in ("dense_gqa", "fused_hybrid") else 4,
            hidden_act="silu",
            tie_word_embeddings=False,
        )
        self.embed_tokens = nn.Embedding(vocab_size, d)
        if arch == "dense_gqa":
            blocks = [_DenseGQABlock(d, inter) for _ in range(n_layers)]
        elif arch == "fused_hybrid":
            blocks = [_FusedHybridBlock(d, inter, use_conv=(i % 2 == 0)) for i in range(n_layers)]
        elif arch == "sparse_moe":
            blocks = [_SparseMoEBlock(d, inter, n_experts=4) for _ in range(n_layers)]
        elif arch == "pure_ssm":
            blocks = [_PureSSMBlock(d) for _ in range(n_layers)]
        else:
            raise ValueError(arch)
        self.layers = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False):
        x = self.embed_tokens(input_ids)
        for blk in self.layers:
            x = blk(x)
        return SimpleNamespace(logits=self.lm_head(self.norm(x)))


# ----------------------------------------------------------------------------------------------- tests
@pytest.mark.parametrize(
    "arch,expected_family",
    [
        ("dense_gqa", "dense_gqa"),
        ("fused_hybrid", "hybrid_shortconv_gqa"),
        ("sparse_moe", "sparse_moe"),
        ("pure_ssm", "pure_mamba_ssm"),
    ],
)
def test_multi_architecture_xray_scan_and_base_skip(arch: str, expected_family: str) -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    model = _ToyArchModel(arch, vocab_size=len(tok), n_layers=3, d=24, inter=48)
    pack = build_frozen_probe_pack(use_hf_datasets=False)

    card = scan_model_xray(model, tok, repo_id=f"toy/{arch}", probe_pack=pack, skip_generation=True)
    assert card["schema"] == XRAY_SCHEMA_VERSION
    assert card["identity"]["arch_family"] == expected_family
    assert card["identity"]["n_layers"] == 3
    assert card["anatomy"]["depth_bins"] == DEPTH_BINS
    assert len(card["anatomy"]["fisher_profile"]["block_fisher_share_8"]) == 8
    assert card["pathologies"]["over_refusal"] == {"status": "skipped_no_chat_template"}
    assert card["pathologies"]["sycophancy"] == {"status": "skipped_no_chat_template"}
    assert card["pathologies"]["loops"]["forced_rep_nll_slope"] is not None

    if arch == "sparse_moe":
        # Must locate the 3 decoder layers rather than the 4-expert ModuleList inside block 0.
        assert card["identity"]["moe"]["moe_layers"] == 3
        assert card["identity"]["moe"]["n_experts"] == 4
        assert card["anatomy"]["forward_profile"]["moe_router_entropy_mean"] is not None
    if arch == "pure_ssm":
        assert card["anatomy"]["fisher_profile"]["fisher_target_role"] == "mixer_out"


def test_tiny_llama_end_to_end_generation_and_multimodal_wrapper() -> None:
    tok = _build_toy_tokenizer(with_chat_template=True)
    lm = _build_tiny_llama(len(tok), n_layers=2, hidden=16, inter=32)

    class _MultimodalWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.language_model = inner
            self.config = inner.config

        def get_input_embeddings(self):
            return self.language_model.get_input_embeddings()

        def get_output_embeddings(self):
            return self.language_model.get_output_embeddings()

        def forward(self, *args, **kwargs):
            return self.language_model(*args, **kwargs)

        def generate(self, *args, **kwargs):
            return self.language_model.generate(*args, **kwargs)

    pack = build_frozen_probe_pack(use_hf_datasets=False)
    wrapped = _MultimodalWrapper(lm)
    card = scan_model_xray(
        wrapped, tok, repo_id="toy/vlm-llama", probe_pack=pack, batch_size=4, loop_max_new=8, skip_generation=False
    )
    assert card["identity"]["arch_family"] == "dense_gqa"
    assert card["identity"]["has_chat_template"] is True
    assert card["pathologies"]["over_refusal"]["base_overref_rate"] is not None
    assert card["pathologies"]["sycophancy"]["turn1_acc"] is not None
    assert card["pathologies"]["loops"]["microgen_decoding"]["max_new_tokens"] == 8


def test_biomarker_math_accuracy() -> None:
    rng = np.random.default_rng(7)
    y = np.array([1] * 24 + [0] * 24)
    sep_x = rng.normal(scale=0.3, size=(48, 32))
    sep_x[:24, 0] += 3.0
    auc_sep, u_sep = _oof_linear_probe(sep_x, y)
    assert auc_sep is not None and auc_sep > 0.95 and u_sep is not None

    noise_x = rng.normal(size=(48, 32))
    auc_noise, _ = _oof_linear_probe(noise_x, y)
    assert auc_noise is not None and 0.25 <= auc_noise <= 0.75

    q, _ = torch.linalg.qr(torch.randn(16, 16))
    er_orth, t1_orth = _effective_rank_and_top1(q)
    assert er_orth > 0.98 and t1_orth < 0.1

    rank1 = torch.outer(torch.randn(16), torch.randn(32))
    er_r1, t1_r1 = _effective_rank_and_top1(rank1)
    assert er_r1 < 0.2 and t1_r1 > 0.99

    assert _gini(torch.ones(64)) == pytest.approx(0.0, abs=1e-6)
    spike = torch.zeros(64)
    spike[0] = 100.0
    assert _gini(spike) > 0.95


def test_veto_framing_shift_biomarker() -> None:
    tok = _build_toy_tokenizer(with_chat_template=True)
    lm = _build_tiny_llama(len(tok), n_layers=2, hidden=16, inter=32)
    struct = scan_model_xray.__globals__["_discover_xray_structure"](lm, tok, "toy/shift")

    tox_id = tok.encode("w20", add_special_tokens=False)[0]
    over_id = tok.encode("w10", add_special_tokens=False)[0]
    frame_id = tok.encode("w30", add_special_tokens=False)[0]
    assert len({tox_id, over_id, frame_id, tok.unk_token_id}) == 4

    custom_pack = {
        "overrefusal": [{"prompt": f"w10 w{11 + (i % 5)}"} for i in range(12)],
        "toxic": [{"prompt": f"w20 w{21 + (i % 5)}", "category": "c"} for i in range(12)],
        "everyday_controls": [f"w40 w{41 + (i % 5)}" for i in range(8)],
        "toxic_framed": [
            {"raw_idx": i, "framing": "educational", "category": "c", "prompt": f"w30 w20 w{21 + (i % 5)}"}
            for i in range(12)
        ],
    }

    state = {"tox": None, "over": None, "frame": None, "frame_delta": 0.0}
    emb = lm.get_input_embeddings()

    def _emb_pre(_m, inp):
        ids = inp[0]
        state["tox"] = (ids == tox_id).any(dim=1)
        state["over"] = (ids == over_id).any(dim=1)
        state["frame"] = (ids == frame_id).any(dim=1)

    def _layer_hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        # Zero out dim 0 to remove random init variance, then write clean +/- signal on toxic vs overrefusal.
        t2 = t.clone() * 0.05
        if state["tox"] is not None:
            t2[state["tox"], -1, 0] += 4.0
        if state["over"] is not None:
            t2[state["over"], -1, 0] -= 4.0
        if state["frame"] is not None and state["frame_delta"] != 0.0:
            t2[state["frame"], -1, 0] += float(state["frame_delta"])
        return (t2, *out[1:]) if isinstance(out, tuple) else t2

    h_emb = emb.register_forward_pre_hook(_emb_pre)
    h_l0 = lm.model.layers[0].register_forward_hook(_layer_hook)
    h_l1 = lm.model.layers[1].register_forward_hook(_layer_hook)
    try:
        state["frame_delta"] = 0.0
        res_clean, _ = _scan_over_refusal(lm, struct, tok, custom_pack, torch.device("cpu"), 0, 8, skip_generation=True)
        state["frame_delta"] = -7.0
        res_shifted, _ = _scan_over_refusal(lm, struct, tok, custom_pack, torch.device("cpu"), 0, 8, skip_generation=True)
    finally:
        h_emb.remove()
        h_l0.remove()
        h_l1.remove()

    assert res_clean["veto_framing_shift_to_std"] is not None
    assert res_clean["veto_framing_shift_to_std"] < 0.15
    assert res_shifted["veto_framing_shift_to_std"] is not None
    assert res_shifted["veto_framing_shift_to_std"] > 1.5


def test_json_budget_48_layers_with_clinical_history(tmp_path: Path) -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    model = _ToyArchModel("dense_gqa", vocab_size=len(tok), n_layers=48, d=16, inter=32)
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    history = sxa.CLINICAL_HISTORY_SEEDS["LiquidAI/LFM3-1.7B"]

    card = scan_model_xray(
        model,
        tok,
        repo_id="toy/48L-dense",
        probe_pack=pack,
        clinical_history=history,
        skip_generation=True,
    )
    txt = format_compact_xray_json(card)
    n_bytes = len(txt.encode("utf-8"))
    assert 1500 <= n_bytes <= 5500, f"Unexpected JSON size: {n_bytes} bytes"
    assert "NaN" not in txt and "Infinity" not in txt

    p = save_xray_json(card, tmp_path / "48l.xray.json")
    roundtrip = load_xray_json(p)
    assert roundtrip["identity"]["n_layers"] == 48
    assert len(roundtrip["clinical_history"]) == 4


def test_anti_leakage_and_holdout_disjointness() -> None:
    # 1. OR-Bench slice bounds strictly disjoint from eval_refusal_blind.py splits
    hard_pool = [f"hard_{i}" for i in range(600)]
    tox_pool = [{"prompt": f"tox_{i}", "category": "c"} for i in range(500)]
    ctrl_pool = [f"ctrl_{i}" for i in range(60)]
    splits = ev.split_rows(hard_pool, tox_pool, ctrl_pool, n_or=200, n_tox_agent=100, n_tox_test=200, n_ctrl=30)

    ev_hard = {r["prompt"] for r in splits["agent_overrefusal"] + splits["test_overrefusal"]}
    xray_hard = set(hard_pool[OR_BENCH_HARD_SLICE[0] : OR_BENCH_HARD_SLICE[1]])
    assert ev_hard.isdisjoint(xray_hard)

    ev_tox = {r["prompt"] for r in splits["agent_toxic"] + splits["test_toxic"]}
    xray_tox = {r["prompt"] for r in tox_pool[OR_BENCH_TOXIC_SLICE[0] : OR_BENCH_TOXIC_SLICE[1]]}
    assert ev_tox.isdisjoint(xray_tox)

    # 2. Framing wrappers 4-gram disjoint from router_cal wrappers
    def _ngrams(s: str, n: int = 4) -> set[tuple[str, ...]]:
        ws = re.findall(r"[a-z0-9]+", s.lower())
        return {tuple(ws[i : i + n]) for i in range(max(0, len(ws) - n + 1))}

    cal_grams = set()
    for tpl in list(CALIBRATION_FIT_WRAPPERS.values()) + list(CALIBRATION_VAL_WRAPPERS.values()):
        cal_grams |= _ngrams(tpl.format(goal="x"))
    xray_grams = set()
    for tpl in FRAMING_WRAPPERS.values():
        xray_grams |= _ngrams(tpl.format(prompt="x"))
    assert cal_grams.isdisjoint(xray_grams)

    # 3. Pilot models strictly disjoint from holdout A/B models
    assert len(sxa.PILOT_MODELS) == 15
    assert set(sxa.PILOT_MODELS).isdisjoint(set(sxa.RESERVED_HOLDOUT_MODELS))
    with pytest.raises(ValueError, match="hold-out"):
        sxa._resolve_target_models("microsoft/Phi-4-mini-instruct", allow_holdout=False)


def test_summarize_xray_atlas(tmp_path: Path) -> None:
    tok = _build_toy_tokenizer(with_chat_template=True)
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    specs = [
        ("Qwen/Qwen3.5-0.8B", "fused_hybrid", 0.2, 0.8),
        ("Qwen/Qwen3.5-4B", "fused_hybrid", 0.1, 0.1),
        ("Qwen/Qwen3-4B", "dense_gqa", 0.4, 0.2),
        ("LiquidAI/LFM3-1.7B", "fused_hybrid", 0.5, 0.0),
    ]
    for repo, arch, over_r, rep_r in specs:
        m = _ToyArchModel(arch, vocab_size=len(tok), n_layers=2, d=16, inter=32)
        card = scan_model_xray(m, tok, repo_id=repo, probe_pack=pack, skip_generation=True)
        card["pathologies"]["over_refusal"]["base_overref_rate"] = over_r
        card["pathologies"]["over_refusal"]["opening_entropy_answered"] = 2.0 + over_r
        card["pathologies"]["over_refusal"]["auc_veto_peak"] = {"layer": 1, "depth": 1.0, "auc": 0.9 - 0.2 * over_r}
        card["pathologies"]["loops"]["forced_rep_nll_slope"] = rep_r
        card["anatomy"]["forward_profile"]["massive_act_max"]["ratio"] = 10.0 + 20.0 * rep_r
        save_xray_json(card, tmp_path / f"{sxa.repo_slug(repo)}.xray.json")

    summary = summarize_xray_atlas(tmp_path)
    assert summary["n_models"] == 4
    assert "scale_qwen35" in summary["contrasts"]
    assert "mixer_deltanet_vs_gqa" in summary["contrasts"]
    assert len(summary["biomarker_correlations"]) >= 1
    assert "Qwen/Qwen3.5-4B" in summary["markdown_table"]
