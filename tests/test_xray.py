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
    _spearman_with_p,
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

    dbg: dict[str, object] = {}
    card = scan_model_xray(
        model,
        tok,
        repo_id="toy/48L-dense",
        probe_pack=pack,
        clinical_history=history,
        skip_generation=True,
        debug_out=dbg,
    )
    txt = format_compact_xray_json(card)
    n_bytes = len(txt.encode("utf-8"))
    assert 1200 <= n_bytes <= 4800, f"Unexpected JSON size: {n_bytes} bytes"
    assert "NaN" not in txt and "Infinity" not in txt
    assert "refusal_samples" not in card and "math_greedy_sample" not in card

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
    assert "correlation_meta" in summary
    cmeta = summary["correlation_meta"]
    assert cmeta["method"] == "spearman_two_sided_t"
    assert cmeta["n_tests"] >= 1
    assert cmeta["bonferroni_alpha_005"] == pytest.approx(0.05 / cmeta["n_tests"], abs=1e-4)
    assert "exploratory" in cmeta["note"].lower()
    assert len(summary["biomarker_correlations"]) >= 1
    for rec in summary["biomarker_correlations"]:
        for key in ("p_value", "p_bonferroni", "evidence_tier", "mode_filter", "n_excluded_by_mode"):
            assert key in rec
    assert "Qwen/Qwen3.5-4B" in summary["markdown_table"]


def test_eos_pad_trimming_and_loop_mean_len() -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    lm = _build_tiny_llama(len(tok), n_layers=2, hidden=16, inter=32)
    eos_id = int(tok.eos_token_id)
    pad_id = int(tok.pad_token_id)
    w_ids = [int(tok.encode(w, add_special_tokens=False)[0]) for w in ("w0", "w1", "w2")]

    def _fake_generate(input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, max_new_tokens: int = 44, **kw):
        bs = input_ids.shape[0]
        tail_row = w_ids + [eos_id] + [pad_id] * max(0, int(max_new_tokens) - len(w_ids) - 1)
        tail = torch.tensor(tail_row[: int(max_new_tokens)], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, tail.unsqueeze(0).expand(bs, -1)], dim=1)

    lm.generate = _fake_generate  # type: ignore[method-assign]
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    card = scan_model_xray(lm, tok, repo_id="toy/eos-pad", probe_pack=pack, loop_max_new=44, skip_generation=False)
    mg = card["pathologies"]["loops"]["microgen_decoding"]
    assert mg["greedy_mean_len"] == pytest.approx(3.0)
    assert mg["sampled_mean_len"] == pytest.approx(3.0)
    assert mg["greedy_rep16_rate"] == pytest.approx(0.0)
    assert mg["sampled_rep16_rate"] == pytest.approx(0.0)


def test_unembed_anisotropy_uncentered() -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    model = _ToyArchModel("dense_gqa", vocab_size=len(tok), n_layers=2, d=24, inter=48)
    torch.manual_seed(42)
    with torch.no_grad():
        base_dir = torch.ones(24)
        model.lm_head.weight.copy_(base_dir.unsqueeze(0) + torch.randn(len(tok), 24) * 0.02)
        model.norm.weight.fill_(1.0)
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    card = scan_model_xray(model, tok, repo_id="toy/aniso", probe_pack=pack, skip_generation=True)
    aniso = card["anatomy"]["static_weights"]["unembed_anisotropy_cos"]
    assert aniso is not None and aniso > 0.85


def test_olmo2_post_rmsnorm_with_qk_norm() -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    d, inter = 16, 32

    class _Olmo2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_norm = nn.RMSNorm(d)
            self.k_norm = nn.RMSNorm(d)
            self.o_proj = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return self.o_proj(self.q_norm(x) + self.k_norm(x))

    class _Olmo2MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(d, inter, bias=False)
            self.up_proj = nn.Linear(d, inter, bias=False)
            self.down_proj = nn.Linear(inter, d, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class _Olmo2Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Olmo2Attn()
            self.mlp = _Olmo2MLP()
            self.post_attention_layernorm = nn.RMSNorm(d)
            self.post_feedforward_layernorm = nn.RMSNorm(d)

        def forward(self, x):
            x = x + self.post_attention_layernorm(self.self_attn(x))
            x = x + self.post_feedforward_layernorm(self.mlp(x))
            return x

    class _Olmo2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="olmo2",
                hidden_size=d,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=4,
                hidden_act="silu",
                tie_word_embeddings=False,
            )
            self.embed_tokens = nn.Embedding(len(tok), d)
            self.layers = nn.ModuleList([_Olmo2Block() for _ in range(3)])
            self.norm = nn.RMSNorm(d)
            self.lm_head = nn.Linear(d, len(tok), bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False):
            x = self.embed_tokens(input_ids)
            for blk in self.layers:
                x = blk(x)
            return SimpleNamespace(logits=self.lm_head(self.norm(x)))

    pack = build_frozen_probe_pack(use_hf_datasets=False)
    card = scan_model_xray(_Olmo2Model(), tok, repo_id="toy/olmo2", probe_pack=pack, skip_generation=True)
    assert card["identity"]["norm_type"] == "post_rmsnorm"


def test_zamba2_hybrid_mamba_gqa_shared_attn() -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    d = 16

    class _SharedAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return self.o_proj(x)

    class _SharedTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(d)
            self.self_attn = _SharedAttn()

        def forward(self, x):
            return self.self_attn(self.input_layernorm(x))

    class _MambaSub(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return self.out_proj(x)

    class _Zamba2MambaBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.mamba = _MambaSub()
            self.input_layernorm = nn.RMSNorm(d)

        def forward(self, x):
            return x + self.mamba(self.input_layernorm(x))

    class _Zamba2HybridBlock(nn.Module):
        def __init__(self, shared_tr: nn.Module):
            super().__init__()
            self.mamba_decoder = _Zamba2MambaBlock()
            self.shared_transformer = shared_tr

        def forward(self, x):
            return self.mamba_decoder(x + self.shared_transformer(x))

    class _Zamba2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="zamba2",
                hidden_size=d,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                hidden_act="gelu",
                tie_word_embeddings=False,
                hybrid_layer_ids=[1, 3],
                layers_block_type=["mamba", "hybrid", "mamba", "hybrid"],
            )
            self.embed_tokens = nn.Embedding(len(tok), d)
            shared_tr = _SharedTransformer()
            self.layers = nn.ModuleList(
                [_Zamba2MambaBlock(), _Zamba2HybridBlock(shared_tr), _Zamba2MambaBlock(), _Zamba2HybridBlock(shared_tr)]
            )
            self.norm = nn.RMSNorm(d)
            self.lm_head = nn.Linear(d, len(tok), bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False):
            x = self.embed_tokens(input_ids)
            for blk in self.layers:
                x = blk(x)
            return SimpleNamespace(logits=self.lm_head(self.norm(x)))

    pack = build_frozen_probe_pack(use_hf_datasets=False)
    card = scan_model_xray(_Zamba2Model(), tok, repo_id="toy/zamba2", probe_pack=pack, skip_generation=True)
    assert card["identity"]["arch_family"] == "hybrid_mamba_gqa"
    assert card["identity"]["layer_mix"]["shared_attn"] == 2
    assert card["identity"]["layer_mix"]["mamba_ssm"] == 2


def test_non_linear_moe_tuple_router_entropy() -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    d, inter, n_experts = 16, 32, 4

    class _TupleTopKRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n_experts, d))

        def forward(self, hidden_states: torch.Tensor):
            flat = hidden_states.reshape(-1, d)
            probs = torch.full((flat.shape[0], n_experts), 0.01, dtype=torch.float32, device=flat.device)
            probs[:, 0] = 0.97
            top_v, top_i = torch.topk(probs, 2, dim=-1)
            return probs, top_v, top_i

    class _OlmoeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Linear(inter, d, bias=False)
            self.up_proj = nn.Linear(d, inter, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.up_proj(x)))

    class _OlmoeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(d)
            self.post_attention_layernorm = nn.RMSNorm(d)

            class _Attn(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_norm = nn.RMSNorm(d)
                    self.k_norm = nn.RMSNorm(d)
                    self.o_proj = nn.Linear(d, d, bias=False)

                def forward(self, x):
                    return self.o_proj(self.q_norm(x) + self.k_norm(x))

            class _MoE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gate = _TupleTopKRouter()
                    self.experts = nn.ModuleList([_OlmoeExpert() for _ in range(n_experts)])

                def forward(self, x):
                    probs, _, _ = self.gate(x)
                    out = torch.zeros_like(x)
                    p = probs.reshape(x.shape[0], x.shape[1], n_experts)
                    for idx, exp in enumerate(self.experts):
                        out = out + p[..., idx : idx + 1] * exp(x)
                    return out

            self.self_attn = _Attn()
            self.mlp = _MoE()

        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x

    class _OlmoeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="olmoe",
                hidden_size=d,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=4,
                num_experts=n_experts,
                num_experts_per_tok=2,
                hidden_act="silu",
                tie_word_embeddings=False,
            )
            self.embed_tokens = nn.Embedding(len(tok), d)
            self.layers = nn.ModuleList([_OlmoeBlock() for _ in range(3)])
            self.norm = nn.RMSNorm(d)
            self.lm_head = nn.Linear(d, len(tok), bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False):
            x = self.embed_tokens(input_ids)
            for blk in self.layers:
                x = blk(x)
            return SimpleNamespace(logits=self.lm_head(self.norm(x)))

    pack = build_frozen_probe_pack(use_hf_datasets=False)
    card = scan_model_xray(_OlmoeModel(), tok, repo_id="toy/olmoe", probe_pack=pack, skip_generation=True)
    assert card["identity"]["norm_type"] == "pre_rmsnorm"
    assert card["identity"]["moe"]["moe_layers"] == 3
    assert card["identity"]["moe"]["n_experts"] == 4
    ent = card["anatomy"]["forward_profile"].get("moe_router_entropy_mean")
    assert ent is not None and ent < 0.35


def test_self_healing_bs1_fallback_and_debug_out() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab_words = [
        "<pad>", "<eos>", "<unk>", "user:", "assistant:",
        "I", "cannot", "fulfill", "that", "request.", "The", "answer", "is", "Paris.",
    ] + [f"w{i}" for i in range(60)]
    vocab = {w: i for i, w in enumerate(vocab_words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}assistant:"

    lm = _build_tiny_llama(len(tok), n_layers=2, hidden=16, inter=32)
    struct = scan_model_xray.__globals__["_discover_xray_structure"](lm, tok, "toy/fallback-bs1")
    eos_id = int(tok.eos_token_id)
    batched_ids = tok.encode("The answer is Paris.", add_special_tokens=False) + [eos_id]
    refusal_ids = tok.encode("I cannot fulfill that request.", add_special_tokens=False) + [eos_id]

    def _fake_generate(input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, max_new_tokens: int = 32, **kw):
        bs = input_ids.shape[0]
        seq = batched_ids if bs > 1 else refusal_ids
        tail = torch.tensor(seq, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0).expand(bs, -1)
        return torch.cat([input_ids, tail], dim=1)

    lm.generate = _fake_generate  # type: ignore[method-assign]
    custom_pack = {
        "overrefusal": [{"prompt": f"w10 w{i}"} for i in range(4)],
        "toxic": [{"prompt": f"w20 w{i}", "category": "c"} for i in range(4)],
        "everyday_controls": [f"w30 w{i}" for i in range(4)],
        "toxic_framed": [{"raw_idx": i, "framing": "edu", "category": "c", "prompt": f"w40 w{i}"} for i in range(4)],
    }
    dbg: dict[str, object] = {}
    res, _ = _scan_over_refusal(
        lm, struct, tok, custom_pack, torch.device("cpu"), int(tok.pad_token_id), 4, skip_generation=False, debug_out=dbg
    )
    assert res["gen_fallback_bs1"] is True
    assert res["gen_fallback_nocache"] is False
    assert res["base_toxic_ref_rate"] is not None and res["base_toxic_ref_rate"] > 0.5
    samples = dbg["refusal_samples"]
    assert isinstance(samples, list) and len(samples) == 5
    groups = [s["group"] for s in samples]
    assert groups.count("overrefusal") == 2
    assert groups.count("toxic") == 2
    assert groups.count("everyday_control") == 1


def test_self_healing_nocache_fallback_and_propagation() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab_words = [
        "<pad>", "<eos>", "<unk>", "user:", "assistant:",
        "I", "cannot", "fulfill", "that", "request.", "The", "answer", "is", "Paris.", "HTTP",
    ] + [f"w{i}" for i in range(60)]
    vocab = {w: i for i, w in enumerate(vocab_words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}assistant:"

    lm = _build_tiny_llama(len(tok), n_layers=2, hidden=16, inter=32)
    eos_id = int(tok.eos_token_id)
    i_tok_id = int(tok.encode("I", add_special_tokens=False)[0])
    w0_id = int(tok.encode("w0", add_special_tokens=False)[0])
    over_id = int(tok.encode("w10", add_special_tokens=False)[0])
    tox_id = int(tok.encode("w20", add_special_tokens=False)[0])
    refusal_ids = tok.encode("I cannot fulfill that request.", add_special_tokens=False) + [eos_id]
    paris_ids = tok.encode("The answer is Paris.", add_special_tokens=False) + [eos_id]

    def _fake_generate(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 24,
        use_cache: bool = True,
        **kw,
    ):
        bs = input_ids.shape[0]
        if use_cache:
            tail = [i_tok_id] + [w0_id] * max(1, int(max_new_tokens) - 1)
            row_t = torch.tensor(tail[: int(max_new_tokens)], dtype=input_ids.dtype, device=input_ids.device)
            return torch.cat([input_ids, row_t.unsqueeze(0).expand(bs, -1)], dim=1)
        out_rows = []
        width = max(len(refusal_ids), len(paris_ids))
        for r in range(bs):
            ids_r = set(int(x) for x in input_ids[r].tolist())
            seq = list(refusal_ids if (tox_id in ids_r or over_id in ids_r) else paris_ids)
            while len(seq) < width:
                seq.append(eos_id)
            out_rows.append(seq[:width])
        tail = torch.tensor(out_rows, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, tail], dim=1)

    lm.generate = _fake_generate  # type: ignore[method-assign]
    base_pack = build_frozen_probe_pack(use_hf_datasets=False)
    custom_pack = {
        **base_pack,
        "overrefusal": [{"prompt": f"w10 w{i}"} for i in range(4)],
        "toxic": [{"prompt": f"w20 w{i}", "category": "c"} for i in range(4)],
        "everyday_controls": [f"w30 HTTP w{i}" for i in range(4)],
        "toxic_framed": [{"raw_idx": i, "framing": "edu", "category": "c", "prompt": f"w40 w{i}"} for i in range(4)],
        "math_prompts": [f"w50 w{i}" for i in range(6)],
    }
    dbg: dict[str, object] = {}
    card = scan_model_xray(
        lm, tok, repo_id="toy/nocache", probe_pack=custom_pack, batch_size=4, loop_max_new=24, skip_generation=False, debug_out=dbg
    )
    oref = card["pathologies"]["over_refusal"]
    mg = card["pathologies"]["loops"]["microgen_decoding"]
    assert oref["gen_fallback_nocache"] is True
    assert oref["gen_fallback_bs1"] is False
    assert oref["base_toxic_ref_rate"] is not None and oref["base_toxic_ref_rate"] > 0.5
    assert mg["greedy_rep16_rate"] == pytest.approx(0.0)
    assert mg["gen_fallback_nocache"] is True
    assert dbg["gen_fallback_nocache"] is True
    assert "cache_probe" in dbg


def test_spearman_with_p_matches_exact_analytics() -> None:
    # Edge cases: too few points, constant array, exact +/-1 monotone ranks.
    assert _spearman_with_p([1.0, 2.0], [3.0, 4.0]) == (None, None, 2)
    assert _spearman_with_p([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) == (None, None, 4)
    rho1, p1, n1 = _spearman_with_p([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0])
    assert n1 == 5 and rho1 == pytest.approx(1.0) and p1 == pytest.approx(0.0)
    rho_neg, p_neg, n_neg = _spearman_with_p([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    assert n_neg == 4 and rho_neg == pytest.approx(-1.0) and p_neg == pytest.approx(0.0)

    # Exact Student-t df=2 two-sided p-value: p = 1 - |rho| when n = 4.
    rho_n4, p_n4, n_n4 = _spearman_with_p([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0])
    assert n_n4 == 4
    assert rho_n4 == pytest.approx(0.8, abs=1e-12)
    assert p_n4 == pytest.approx(1.0 - abs(0.8), abs=1e-12)

    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(123)
    for n in (4, 11, 13, 14):
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        rho_ours, p_ours, n_ours = _spearman_with_p(x, y)
        res_sp = scipy_stats.spearmanr(x, y)
        assert n_ours == n
        assert rho_ours == pytest.approx(float(res_sp.statistic), abs=1e-10)
        assert p_ours == pytest.approx(float(res_sp.pvalue), abs=1e-10)


def test_opening_entropy_norm_removes_sample_size_artifact() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab_words = (
        ["<pad>", "<eos>", "<unk>", "user:", "assistant:", "I", "cannot", "fulfill", "that", "request."]
        + ["Reply", "same", "text.", "What", "is", "2", "+", "?"]
        + [f"p{i}" for i in range(26)]
        + [f"a{i}" for i in range(26)]
        + [f"b{i}" for i in range(26)]
    )
    vocab = {w: i for i, w in enumerate(vocab_words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}assistant:"

    lm = _build_tiny_llama(len(tok), n_layers=2, hidden=16, inter=32)
    struct = scan_model_xray.__globals__["_discover_xray_structure"](lm, tok, "toy/ent-norm")
    eos_id = int(tok.eos_token_id)
    pad_id = int(tok.pad_token_id)

    tag_ids = [int(tok.encode(f"p{i}", add_special_tokens=False)[0]) for i in range(24)]
    refusal_seq = tok.encode("I cannot fulfill that request.", add_special_tokens=False) + [eos_id]
    same_seq = tok.encode("Reply same text.", add_special_tokens=False) + [eos_id]
    uniq_seqs = [tok.encode(f"Reply a{i} b{i}", add_special_tokens=False) + [eos_id] for i in range(24)]

    custom_pack = {
        "overrefusal": [{"prompt": f"p{i}"} for i in range(8)],
        "toxic": [{"prompt": f"p{8 + i}", "category": "c"} for i in range(8)],
        "everyday_controls": [f"p{16 + i}" for i in range(8)],
        "toxic_framed": [{"raw_idx": 0, "framing": "edu", "category": "c", "prompt": "p24"}],
    }
    high_ref_indices = set(range(0, 6)) | set(range(8, 14))  # 12 refused, 12 answered (2 over + 2 tox + 8 ctrl)
    mode_box = {"mode": "high_ref"}

    def _fake_generate(input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, max_new_tokens: int = 48, **kw):
        bs = input_ids.shape[0]
        rows = []
        max_w = max(len(refusal_seq), len(same_seq), *(len(s) for s in uniq_seqs))
        for r in range(bs):
            toks_r = set(int(x) for x in input_ids[r].tolist())
            matched = next((idx for idx, tid in enumerate(tag_ids) if tid in toks_r), None)
            if matched is None:
                seq = list(same_seq)
            elif mode_box["mode"] == "high_ref":
                seq = list(refusal_seq if matched in high_ref_indices else uniq_seqs[matched])
            elif mode_box["mode"] == "zero_ref":
                seq = list(uniq_seqs[matched])
            else:
                seq = list(same_seq)
            while len(seq) < max_w:
                seq.append(eos_id)
            rows.append(seq[:max_w])
        tail = torch.tensor(rows, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, tail], dim=1)

    lm.generate = _fake_generate  # type: ignore[method-assign]

    mode_box["mode"] = "high_ref"
    res_high, _ = _scan_over_refusal(lm, struct, tok, custom_pack, torch.device("cpu"), pad_id, 8, skip_generation=False)
    mode_box["mode"] = "zero_ref"
    res_zero, _ = _scan_over_refusal(lm, struct, tok, custom_pack, torch.device("cpu"), pad_id, 8, skip_generation=False)
    mode_box["mode"] = "collapsed"
    res_coll, _ = _scan_over_refusal(lm, struct, tok, custom_pack, torch.device("cpu"), pad_id, 8, skip_generation=False)

    assert res_high["n_answered_nonempty"] == 12
    assert res_zero["n_answered_nonempty"] == 24
    # Raw Shannon entropy drops purely because fewer answered prompts remain (ln 12 < ln 24).
    assert res_high["opening_entropy_answered"] < res_zero["opening_entropy_answered"] - 0.5
    # Normalized entropy eliminates the N-dependence and shows 100% opening diversity for both.
    assert res_high["opening_entropy_norm"] == pytest.approx(1.0, abs=1e-6)
    assert res_zero["opening_entropy_norm"] == pytest.approx(1.0, abs=1e-6)
    assert res_high["ctrl_opening_entropy_norm"] == pytest.approx(1.0, abs=1e-6)
    assert res_zero["ctrl_opening_entropy_norm"] == pytest.approx(1.0, abs=1e-6)
    # Genuinely repetitive openings collapse to 0.0 on both normalized metrics.
    assert res_coll["opening_entropy_norm"] == pytest.approx(0.0, abs=1e-6)
    assert res_coll["ctrl_opening_entropy_norm"] == pytest.approx(0.0, abs=1e-6)


def test_summarize_atlas_mode_filters_and_bonferroni(tmp_path: Path) -> None:
    tok = _build_toy_tokenizer(with_chat_template=True)
    pack = build_frozen_probe_pack(use_hf_datasets=False)

    # First 4 cards use matched primary modes with exact 0.0 Spearman; last 2 are fallback outliers.
    specs = [
        ("toy/m0", 0.10, "refused_vs_nonrefused", "refused_toxic_vs_refused_overref", 0.4, 0.90, 0.1, 0.6),
        ("toy/m1", 0.25, "refused_vs_nonrefused", "refused_toxic_vs_refused_overref", 0.5, 0.80, 0.2, 0.4),
        ("toy/m2", 0.40, "refused_vs_nonrefused", "refused_toxic_vs_refused_overref", 0.6, 0.92, 0.3, 0.8),
        ("toy/m3", 0.55, "refused_vs_nonrefused", "refused_toxic_vs_refused_overref", 0.7, 0.85, 0.4, 0.5),
        ("toy/fallback0", 0.70, "toxic_vs_controls", "all_toxic_vs_all_overref", 0.95, 0.99, 0.95, 0.95),
        ("toy/fallback1", 0.85, "toxic_vs_controls", "all_toxic_vs_all_overref", 0.99, 0.995, 0.99, 0.99),
    ]
    for repo, over_r, ref_mode, veto_mode, cos_rv, auc_v, cos_syc, cave in specs:
        m = _ToyArchModel("dense_gqa", vocab_size=len(tok), n_layers=2, d=16, inter=32)
        card = scan_model_xray(m, tok, repo_id=repo, probe_pack=pack, skip_generation=True)
        oref = card["pathologies"]["over_refusal"]
        tox_r = 0.5
        ctrl_r = 0.0
        n_est = round(32 * (1.0 - over_r)) + round(32 * (1.0 - tox_r)) + round(16 * (1.0 - ctrl_r))
        oref["base_overref_rate"] = over_r
        oref["base_toxic_ref_rate"] = tox_r
        oref["base_ctrl_ref_rate"] = ctrl_r
        # Simulate pre-v1.3 card without opening_entropy_norm and raw H = ln(n_est) (100% diverse openings).
        oref.pop("opening_entropy_norm", None)
        oref["opening_entropy_answered"] = math.log(n_est)
        oref["refusal_probe_mode"] = ref_mode
        oref["veto_probe_mode"] = veto_mode
        oref["cos_refusal_veto_at_peak"] = cos_rv
        oref["auc_veto_peak"] = {"layer": 1, "depth": 1.0, "auc": auc_v}
        card["pathologies"]["sycophancy"]["cos_syc_refusal"] = cos_syc
        card["pathologies"]["sycophancy"]["cave_rate_false_pushback"] = cave
        save_xray_json(card, tmp_path / f"{sxa.repo_slug(repo)}.xray.json")

    summary = summarize_xray_atlas(tmp_path)
    cmeta = summary["correlation_meta"]
    corrs = {(c["x"], c["y"]): c for c in summary["biomarker_correlations"]}

    # Raw opening_entropy_answered must not be correlated directly.
    assert ("opening_entropy_answered", "base_overref_rate") not in corrs

    # Legacy fallback should recover 1.0 normalized entropy for all 6 synthetic cards.
    assert len(summary["models"]) == 6
    for row in summary["models"]:
        assert row["opening_entropy_norm"] == pytest.approx(1.0, abs=1e-3)

    # Fallback outliers must be excluded when matching veto/refusal modes.
    rv = corrs[("cos_refusal_veto", "auc_veto_peak")]
    assert rv["n"] == 4
    assert rv["n_excluded_by_mode"] == 2
    assert rv["mode_filter"] == {
        "veto_probe_mode": "refused_toxic_vs_refused_overref",
        "refusal_probe_mode": "refused_vs_nonrefused",
    }
    assert rv["spearman"] == pytest.approx(0.0, abs=1e-6)
    assert rv["p_value"] == pytest.approx(1.0, abs=1e-6)

    syc = corrs[("cos_syc_refusal", "cave_rate_false_pushback")]
    assert syc["n"] == 4
    assert syc["n_excluded_by_mode"] == 2
    assert syc["spearman"] == pytest.approx(0.0, abs=1e-6)

    # Every evaluated correlation carries Bonferroni correction and disciplined evidence tier.
    n_tests = cmeta["n_tests"]
    assert cmeta["method"] == "spearman_two_sided_t"
    assert n_tests >= 2
    valid_tiers = {
        "exploratory_bonferroni_pass",
        "exploratory_nominal_only",
        "null_or_underpowered",
        "insufficient_variance",
    }
    for c in summary["biomarker_correlations"]:
        assert c["evidence_tier"] in valid_tiers
        if c["p_value"] is not None:
            expected_bonf = min(1.0, float(c["p_value"]) * n_tests)
            assert c["p_bonferroni"] == pytest.approx(expected_bonf, abs=max(2e-3, 1e-3 * n_tests))


class _ToyLoopedModel(nn.Module):
    """Toy recurrent-depth transformer: L0 (1x) -> [L1, L2] (3x) -> L3 (1x) = 8 effective steps from 4 physical layers."""

    def __init__(self, vocab_size: int, d: int = 24, inter: int = 48, n_loops: int = 3):
        super().__init__()
        self.n_loops = n_loops
        self.config = SimpleNamespace(
            model_type="inkling2_loop",
            hidden_size=d,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_act="silu",
            tie_word_embeddings=False,
        )
        self.embed_tokens = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList([_DenseGQABlock(d, inter) for _ in range(4)])
        self.norm = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        x = self.embed_tokens(input_ids)
        x = self.layers[0](x)
        for _ in range(self.n_loops):
            x = self.layers[1](x)
            x = self.layers[2](x)
        x = self.layers[3](x)
        logits = self.lm_head(self.norm(x))
        return SimpleNamespace(logits=logits)


def test_looped_transformer_unrolled_xray():
    torch.manual_seed(7)
    tok = _build_toy_tokenizer()
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    m = _ToyLoopedModel(vocab_size=len(tok), d=24, inter=48, n_loops=3).eval()

    card = scan_model_xray(m, tok, repo_id="toy/inkling-loop-mini", probe_pack=pack, skip_generation=True)

    loop_info = card["identity"].get("looped")
    assert loop_info is not None
    assert loop_info["n_physical_layers"] == 4
    assert loop_info["n_effective_steps"] == 8
    assert loop_info["max_visits_per_layer"] == 3
    assert loop_info["looped_layers"] == [1, 2]
    assert loop_info["effective_trace"] == [0, 1, 2, 1, 2, 1, 2, 3]

    fp = card["anatomy"]["forward_profile"]
    assert len(fp["unrolled_resid_norm_8"]) == len(DEPTH_BINS)
    assert len(fp["unrolled_step_cos_8"]) == len(DEPTH_BINS)
    assert all(v is not None and math.isfinite(v) for v in fp["unrolled_resid_norm_8"])
    assert all(v is not None and math.isfinite(v) for v in fp["unrolled_step_cos_8"])

    # Fisher, DLA, and all pathology probes must succeed cleanly with zero NaN.
    fish = card["anatomy"]["fisher_profile"]
    assert fish["fisher_seq_max_len"] is not None and fish["fisher_seq_max_len"] > 0
    assert all(v is not None and math.isfinite(v) for v in fish["block_fisher_share_8"])
    assert "dla_peak" in card["anatomy"]["fact_readout"]
    assert "physical_layer" in card["anatomy"]["fact_readout"]["dla_peak"]
    assert card["pathologies"]["over_refusal"]["auc_refusal_peak"] is not None
    assert card["pathologies"]["sycophancy"]["auc_pushback_peak"] is not None
    assert card["pathologies"]["loops"]["auc_rep_vs_other_peak"] is not None

    # Non-looped models must remain unaffected (no `identity["looped"]` or unrolled curves).
    m_std = _ToyArchModel("dense_gqa", vocab_size=len(tok), n_layers=3, d=24, inter=48).eval()
    card_std = scan_model_xray(m_std, tok, repo_id="toy/std-dense", probe_pack=pack, skip_generation=True)
    assert "looped" not in card_std["identity"]
    assert "unrolled_resid_norm_8" not in card_std["anatomy"]["forward_profile"]


def test_multistream_mhc_looped_xray():
    """Verify multi-stream mHC / Hyper-Connections residual (`[B, T, 2*D]`) collapses cleanly."""
    torch.manual_seed(11)
    tok = _build_toy_tokenizer()
    pack = build_frozen_probe_pack(use_hf_datasets=False)

    class _MHCBlock(nn.Module):
        def __init__(self, d: int, inter: int, mode: str):
            super().__init__()
            self.mode = mode
            self.inner = _DenseGQABlock(d, inter)
            self.d = d

        def forward(self, x):
            if self.mode == "enter":
                y = self.inner(x)
                return torch.cat([y, y * 0.9], dim=-1)
            if self.mode == "middle":
                a, b = x.split(self.d, dim=-1)
                y = self.inner(0.5 * (a + b))
                return torch.cat([y, b + 0.1 * y], dim=-1)
            a, b = x.split(self.d, dim=-1)
            return self.inner(0.5 * (a + b))

    class _ToyMHCLooped(nn.Module):
        def __init__(self, vocab_size: int, d: int = 24, inter: int = 48):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="sensenova_loop",
                hidden_size=d,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                hidden_act="silu",
                tie_word_embeddings=False,
            )
            self.embed_tokens = nn.Embedding(vocab_size, d)
            self.layers = nn.ModuleList([
                _MHCBlock(d, inter, "enter"),
                _MHCBlock(d, inter, "middle"),
                _MHCBlock(d, inter, "exit"),
            ])
            self.norm = nn.RMSNorm(d)
            self.lm_head = nn.Linear(d, vocab_size, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
            x = self.embed_tokens(input_ids)
            x = self.layers[0](x)
            x = self.layers[1](x)
            x = self.layers[1](x)
            x = self.layers[2](x)
            return SimpleNamespace(logits=self.lm_head(self.norm(x)))

    m = _ToyMHCLooped(vocab_size=len(tok), d=24, inter=48).eval()
    card = scan_model_xray(m, tok, repo_id="toy/sensenova-mhc-loop", probe_pack=pack, skip_generation=True)
    assert card["identity"]["looped"]["n_effective_steps"] == 4
    assert card["anatomy"]["fact_readout"]["dla_peak"]["physical_layer"] in (0, 1, 2)
    assert card["pathologies"]["over_refusal"]["auc_refusal_peak"] is not None



