"""CPU-only unit tests for WSI Delta X-Ray (`wsi/delta_xray.py`, `scripts/scan_delta_xray_atlas.py`, `wsi delta-xray`)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import scan_delta_xray_atlas as sdx  # noqa: E402
from wsi.delta_xray import (  # noqa: E402
    DELTA_XRAY_DETAIL_SCHEMA_VERSION,
    DELTA_XRAY_SCHEMA_VERSION,
    accumulate_domain_grad_dots,
    assemble_delta_attribution,
    build_domain_calibration_seqs,
    classify_param_role,
    classify_submatrix_tag,
    compute_weight_and_spectral_diff,
    extract_layer_idx,
    format_compact_delta_xray_json,
    load_delta_xray_detail_json,
    load_delta_xray_json,
    save_delta_xray_detail_json,
    save_delta_xray_json,
    scan_delta_xray,
    summarize_delta_xray_atlas,
)
from wsi.xray import _discover_xray_structure  # noqa: E402
from wsi.xray_probes import build_frozen_probe_pack  # noqa: E402


def _build_toy_tokenizer(*, with_chat_template: bool = True):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab_words = [
        "<pad>", "<eos>", "<unk>", "user:", "assistant:",
        "Problem:", "Find", "all", "integers", "Step", "1:", "Write", "Therefore", "Suppose",
        "Claim:", "During", "General", "Interactive", "def", "import", "math", "from", "return",
    ] + [f"w{i}" for i in range(64)]
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


def test_classify_param_role_and_layer_idx() -> None:
    assert extract_layer_idx("model.layers.14.mlp.down_proj.weight") == 14
    assert extract_layer_idx("layers.0.conv.out_proj.weight") == 0
    assert extract_layer_idx("transformer.h.7.mlp.c_proj.weight") == 7
    assert extract_layer_idx("lm_head.weight") is None

    assert classify_param_role("model.layers.3.mlp.down_proj.weight") == "mlp_out"
    assert classify_param_role("model.layers.3.mlp.gate_proj.weight") == "mlp_in"
    assert classify_param_role("model.layers.3.mlp.experts.0.gate_proj.weight") == "mlp_in"
    assert classify_param_role("model.layers.3.mlp.experts.2.w2.weight") == "mlp_out"
    assert classify_param_role("model.layers.3.mlp.gate.weight", shape=(8, 128)) == "moe_router"
    assert classify_param_role("model.layers.3.self_attn.o_proj.weight") == "attn_out"
    assert classify_param_role("model.layers.3.self_attn.q_proj.weight") == "attn_qkv"
    assert classify_param_role("model.layers.3.linear_attn.out_proj.weight") == "ssm_conv"
    assert classify_param_role("model.layers.3.conv.out_proj.weight") == "ssm_conv"
    assert classify_param_role("model.layers.3.input_layernorm.weight") == "norms"
    assert classify_param_role("model.embed_tokens.weight") == "embed_head"
    assert classify_param_role("lm_head.weight") == "embed_head"
    # Zamba2 hybrid shared_transformer & mamba_decoder names
    assert classify_param_role("model.layers.1.shared_transformer.feed_forward.down_proj.weight") == "mlp_out"
    assert classify_param_role("model.layers.1.shared_transformer.feed_forward.gate_up_proj.weight") == "mlp_in"
    assert classify_param_role("model.layers.1.shared_transformer.feed_forward.gate_up_proj_adapter_list.0.0.weight") == "mlp_in"
    assert classify_param_role("model.layers.1.shared_transformer.self_attn.o_proj.weight") == "attn_out"
    assert classify_param_role("model.layers.1.shared_transformer.self_attn.q_proj.weight") == "attn_qkv"
    assert classify_param_role("model.layers.1.mamba_decoder.mamba.out_proj.weight") == "ssm_conv"
    # Fused 3D OLMoE expert parameters
    assert classify_param_role("model.layers.0.mlp.experts.down_proj", shape=(64, 2048, 1024)) == "mlp_out"
    assert classify_param_role("model.layers.0.mlp.experts.gate_up_proj", shape=(64, 2048, 2048)) == "mlp_in"


def test_planted_rank1_vs_nullspace_and_identical_diff() -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    m_base = _build_tiny_llama(len(tok), n_layers=3, hidden=48, inter=96)
    struct = _discover_xray_structure(m_base, tok, "toy/spectral")

    # 1) Identical diff -> clean zero baseline
    wd0, sd0 = compute_weight_and_spectral_diff(m_base, m_base, struct, device="cpu")
    assert wd0["total_rel_fro"] == pytest.approx(0.0)
    assert wd0["n_changed_params"] == 0
    assert wd0["top3_changed_layers"] == []
    assert wd0["depth_centroid"] is None
    assert sd0["mlp_out"]["n_mats"] == 0
    assert sd0["mixer_out"]["n_mats"] == 0

    # Compute left singular vectors of base layer 1 down_proj: shape [48, 96]
    w_l1 = m_base.model.layers[1].mlp.down_proj.weight.detach().float()
    U_b, _, _ = torch.linalg.svd(w_l1, full_matrices=False)
    u_top = U_b[:, 0]
    u_null = U_b[:, -1]  # Orthogonal to top-32 subspace (hidden=48 > 32)

    # 2) Rank-1 diff aligned with top-1 base singular direction
    sd_top = {k: v.clone() for k, v in m_base.state_dict().items()}
    sd_top["model.layers.1.mlp.down_proj.weight"] = w_l1 + 0.5 * torch.outer(u_top, torch.ones(w_l1.shape[1]))
    wd_top, sp_top = compute_weight_and_spectral_diff(m_base, sd_top, struct, device="cpu")
    assert wd_top["top3_changed_layers"][0]["layer"] == 1
    assert wd_top["depth_centroid"] == pytest.approx(0.5, abs=1e-4)
    mlp_s = sp_top["mlp_out"]
    assert mlp_s["n_mats"] == 1
    assert mlp_s["energy_r1_mean"] > 0.99
    assert mlp_s["rank80_median"] == 1
    assert mlp_s["top_delta_amp_ratio_mean"] > 1.2
    assert mlp_s["base_top32_overlap_mean"] > 0.99

    # 3) Rank-1 diff orthogonal to top-32 base subspace
    sd_null = {k: v.clone() for k, v in m_base.state_dict().items()}
    sd_null["model.layers.1.mlp.down_proj.weight"] = w_l1 + 0.5 * torch.outer(u_null, torch.ones(w_l1.shape[1]))
    _, sp_null = compute_weight_and_spectral_diff(m_base, sd_null, struct, device="cpu")
    assert sp_null["mlp_out"]["energy_r1_mean"] > 0.99
    assert sp_null["mlp_out"]["base_top32_overlap_mean"] < 1e-6


def test_trapezoidal_attribution_faithfulness_and_direction() -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    m_before = _build_tiny_llama(len(tok), n_layers=3, hidden=16, inter=32)
    m_after = _build_tiny_llama(len(tok), n_layers=3, hidden=16, inter=32)
    m_after.load_state_dict(m_before.state_dict())

    dom_seqs = build_domain_calibration_seqs(tok, pack, max_length=64)
    assert len(dom_seqs["math"]) >= 2

    # Perturb m_after with 3 small gradient steps ONLY on layers.1.mlp.down_proj on math passages.
    for p in m_after.parameters():
        p.requires_grad_(False)
    tgt_p = m_after.model.layers[1].mlp.down_proj.weight
    tgt_p.requires_grad_(True)
    opt = torch.optim.SGD([tgt_p], lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        tot_loss = torch.tensor(0.0)
        for s in dom_seqs["math"]:
            inp = torch.tensor([s], dtype=torch.long)
            logits = m_after(input_ids=inp).logits[0, :-1].float()
            tot_loss = tot_loss + F.cross_entropy(logits, inp[0, 1:]) / len(dom_seqs["math"])
        tot_loss.backward()
        opt.step()
    tgt_p.requires_grad_(False)
    tgt_p.grad = None

    dots_b, gnll_b, pf_cost, used_len = accumulate_domain_grad_dots(
        m_before, m_after, dom_seqs, sign=+1.0, device="cpu", max_len=64, compute_prose_fisher=True
    )
    dots_a, gnll_a, _, _ = accumulate_domain_grad_dots(
        m_after, m_before, dom_seqs, sign=-1.0, device="cpu", max_len=used_len, compute_prose_fisher=False
    )
    attr = assemble_delta_attribution(dots_b, dots_a, gnll_b, gnll_a, pf_cost, n_layers=3, grad_seq_max_len=used_len)

    m_attr = attr["domains"]["math"]
    assert m_attr["actual_dnll"] < -1e-3
    assert 0.75 <= m_attr["faithfulness"] <= 1.25
    assert abs(m_attr["role_credit"]["mlp_out"]) > 0.90 * abs(m_attr["trap_sum"])
    assert len(m_attr["layer_credit_8"]) == 8
    assert m_attr["layer_credit_8"][3] < 0.0 and m_attr["layer_credit_8"][4] < 0.0
    for m in (m_before, m_after):
        for p in m.parameters():
            assert p.grad is None


def test_streaming_safetensors_matches_inmemory_diff(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    tok = _build_toy_tokenizer(with_chat_template=False)
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    m_before = _build_tiny_llama(len(tok), n_layers=3, hidden=16, inter=32)
    m_after = _build_tiny_llama(len(tok), n_layers=3, hidden=16, inter=32)
    m_after.load_state_dict(m_before.state_dict())
    with torch.no_grad():
        m_after.model.layers[0].self_attn.o_proj.weight.add_(0.08 * torch.randn_like(m_after.model.layers[0].self_attn.o_proj.weight))
        m_after.model.layers[2].mlp.down_proj.weight.add_(0.12 * torch.randn_like(m_after.model.layers[2].mlp.down_proj.weight))

    # Save both checkpoints as sharded 2-file safetensors directories.
    def _save_sharded(model: nn.Module, out_d: Path) -> None:
        out_d.mkdir(parents=True, exist_ok=True)
        items = list(model.state_dict().items())
        mid = len(items) // 2
        save_file({k: v.detach().cpu().clone() for k, v in items[:mid]}, str(out_d / "model-00001-of-00002.safetensors"))
        save_file({k: v.detach().cpu().clone() for k, v in items[mid:]}, str(out_d / "model-00002-of-00002.safetensors"))

    b_dir, a_dir = tmp_path / "before", tmp_path / "after"
    _save_sharded(m_before, b_dir)
    _save_sharded(m_after, a_dir)

    struct_b = _discover_xray_structure(m_before, tok, "toy/stream")
    wd_mem, sd_mem = compute_weight_and_spectral_diff(m_before, m_after, struct_b, device="cpu")
    wd_disk, sd_disk = compute_weight_and_spectral_diff(b_dir, a_dir, struct_b, device="cpu")

    assert wd_disk["total_rel_fro"] == pytest.approx(wd_mem["total_rel_fro"], abs=1e-6)
    assert wd_disk["role_energy_share"] == pytest.approx(wd_mem["role_energy_share"], abs=1e-6)
    assert sd_disk["mlp_out"]["rank80_median"] == sd_mem["mlp_out"]["rank80_median"]
    assert sd_disk["mlp_out"]["erank_ratio_mean"] == pytest.approx(sd_mem["mlp_out"]["erank_ratio_mean"], abs=1e-5)

    dom_seqs = build_domain_calibration_seqs(tok, pack, max_length=48)
    dots_mem, gnll_mem, _, _ = accumulate_domain_grad_dots(
        m_before, m_after, dom_seqs, sign=+1.0, device="cpu", max_len=48, compute_prose_fisher=False
    )
    dots_disk, gnll_disk, _, _ = accumulate_domain_grad_dots(
        m_before, a_dir, dom_seqs, sign=+1.0, device="cpu", max_len=48, compute_prose_fisher=False
    )
    assert gnll_disk == pytest.approx(gnll_mem, abs=1e-6)
    for dom in ("prose", "math", "code"):
        for k in dots_mem[dom]:
            assert dots_disk[dom][k] == pytest.approx(dots_mem[dom][k], abs=1e-5)


def test_moe_and_shape_mismatch_safety() -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    d, inter, n_experts = 16, 32, 4

    class _Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(d, inter, bias=False)
            self.w2 = nn.Linear(inter, d, bias=False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)))

    class _MoeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(d)
            self.post_attention_layernorm = nn.RMSNorm(d)
            self.self_attn = nn.Module()
            self.self_attn.o_proj = nn.Linear(d, d, bias=False)
            self.mlp = nn.Module()
            self.mlp.gate = nn.Linear(d, n_experts, bias=False)
            self.mlp.experts = nn.ModuleList([_Expert() for _ in range(n_experts)])

        def forward(self, x):
            h = x + self.self_attn.o_proj(self.input_layernorm(x))
            p = torch.softmax(self.mlp.gate(self.post_attention_layernorm(h)), dim=-1)
            out = torch.zeros_like(h)
            for idx, exp in enumerate(self.mlp.experts):
                out = out + p[..., idx : idx + 1] * exp(self.post_attention_layernorm(h))
            return h + out

    class _ToyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="olmoe", hidden_size=d, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=4, num_experts=n_experts,
                num_experts_per_tok=2, hidden_act="silu", tie_word_embeddings=False,
            )
            self.embed_tokens = nn.Embedding(len(tok), d)
            self.layers = nn.ModuleList([_MoeBlock() for _ in range(4)])
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

    moe_b = _ToyMoE().eval()
    struct = _discover_xray_structure(moe_b, tok, "toy/moe")
    sd_a = {k: v.clone() for k, v in moe_b.state_dict().items()}
    sd_a["layers.1.mlp.gate.weight"] = sd_a["layers.1.mlp.gate.weight"] + 0.1
    sd_a["layers.1.mlp.experts.0.w2.weight"] = sd_a["layers.1.mlp.experts.0.w2.weight"] + 0.2
    # Simulate resized vocab on after checkpoint (e.g. Base -> Instruct special tokens)
    sd_a["lm_head.weight"] = torch.randn(len(tok) + 5, d)

    wd, sp = compute_weight_and_spectral_diff(moe_b, sd_a, struct, device="cpu", max_moe_experts_per_layer=2)
    assert wd["n_shape_mismatch_params"] >= 1
    assert wd["role_energy_share"]["moe_router"] > 0.0
    assert wd["role_energy_share"]["mlp_out"] > 0.0
    assert sp["mlp_out"]["n_mats"] >= 1


def test_delta_xray_json_budget_and_atlas_summary(tmp_path: Path) -> None:
    tok = _build_toy_tokenizer(with_chat_template=False)
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    m_b = _build_tiny_llama(len(tok), n_layers=32, hidden=16, inter=32)
    m_a = _build_tiny_llama(len(tok), n_layers=32, hidden=16, inter=32)
    m_a.load_state_dict(m_b.state_dict())
    with torch.no_grad():
        m_a.model.layers[10].mlp.down_proj.weight.add_(0.05)
        m_a.model.layers[20].self_attn.o_proj.weight.add_(0.03)

    card1 = scan_delta_xray(
        m_b, m_a, tok, pair_id="toy_pair_1", before_repo="toy/b", after_repo="toy/a1",
        transition_type="sft", family="toy32", probe_pack=pack,
        skip_generation=True, grad_seq_max_len=32,
    )
    txt1 = format_compact_delta_xray_json(card1)
    n_bytes = len(txt1.encode("utf-8"))
    assert 1200 <= n_bytes <= 5500, f"Unexpected Delta X-Ray JSON size: {n_bytes} B"
    assert "NaN" not in txt1 and "Infinity" not in txt1
    assert card1["schema"] == DELTA_XRAY_SCHEMA_VERSION

    p1 = save_delta_xray_json(card1, tmp_path / "toy_pair_1.delta.xray.json")
    loaded = load_delta_xray_json(p1)
    assert loaded["pair"]["n_layers"] == 32

    card2 = dict(card1)
    card2["pair"] = {**card1["pair"], "pair_id": "toy_pair_2", "transition_type": "dpo"}
    save_delta_xray_json(card2, tmp_path / "toy_pair_2.delta.xray.json")

    summary = summarize_delta_xray_atlas(tmp_path)
    assert summary["n_pairs"] == 2
    assert set(summary["by_transition_type"].keys()) == {"dpo", "sft"}
    assert "toy_pair_1" in summary["markdown_table"]
    assert "causal recipes" in summary["meta"]["note"].lower() or "exploratory" in summary["meta"]["note"].lower() or "structural recipes" in summary["meta"]["note"].lower()


def test_pilot_pairs_anti_leakage_and_cli() -> None:
    assert len(sdx.PILOT_DELTA_PAIRS) == 11
    pids = [p["pair_id"] for p in sdx.PILOT_DELTA_PAIRS]
    assert len(set(pids)) == 11

    pilot_repos = {p[r] for p in sdx.PILOT_DELTA_PAIRS for r in ("before", "after")}
    assert pilot_repos.isdisjoint(set(sdx.RESERVED_HOLDOUT_MODELS))
    assert len(sdx._resolve_target_pairs("all")) == 11

    with pytest.raises(ValueError, match="hold-out"):
        sdx._validate_no_holdout_pairs(
            [{"pair_id": "bad", "before": "meta-llama/Llama-3.2-3B-Instruct", "after": "x"}],
            allow_holdout=False,
        )

    proc = subprocess.run(
        [sys.executable, "-m", "wsi", "delta-xray", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--transition-type" in proc.stdout and "--summarize-dir" in proc.stdout


def test_streaming_pytorch_bin_and_mixed_safetensors(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    tok = _build_toy_tokenizer(with_chat_template=False)
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    m_before = _build_tiny_llama(len(tok), n_layers=3, hidden=16, inter=32)
    m_after = _build_tiny_llama(len(tok), n_layers=3, hidden=16, inter=32)
    m_after.load_state_dict(m_before.state_dict())
    with torch.no_grad():
        m_after.model.layers[0].self_attn.o_proj.weight.add_(
            0.08 * torch.randn_like(m_after.model.layers[0].self_attn.o_proj.weight)
        )
        m_after.model.layers[2].mlp.down_proj.weight.add_(
            0.12 * torch.randn_like(m_after.model.layers[2].mlp.down_proj.weight)
        )

    b_sf = tmp_path / "before_sf"
    b_sf.mkdir()
    save_file({k: v.detach().cpu().clone() for k, v in m_before.state_dict().items()}, str(b_sf / "model.safetensors"))

    a_sharded = tmp_path / "after_sharded_bin"
    a_sharded.mkdir()
    items = [(k, v.detach().cpu().clone()) for k, v in m_after.state_dict().items()]
    mid = len(items) // 2
    s1_name, s2_name = "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"
    torch.save(dict(items[:mid]), str(a_sharded / s1_name))
    torch.save(dict(items[mid:]), str(a_sharded / s2_name))
    wmap = {k: s1_name for k, _ in items[:mid]}
    wmap.update({k: s2_name for k, _ in items[mid:]})
    (a_sharded / "pytorch_model.bin.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wmap}), encoding="utf-8"
    )
    # Non-weight file must be ignored safely.
    (a_sharded / "training_args.bin").write_bytes(b"not-a-checkpoint")

    a_single = tmp_path / "after_single_bin"
    a_single.mkdir()
    torch.save(dict(items), str(a_single / "pytorch_model.bin"))

    struct_b = _discover_xray_structure(m_before, tok, "toy/bin")
    wd_mem, sd_mem = compute_weight_and_spectral_diff(m_before, m_after, struct_b, device="cpu")
    wd_sh, sd_sh = compute_weight_and_spectral_diff(b_sf, a_sharded, struct_b, device="cpu")
    wd_sg, sd_sg = compute_weight_and_spectral_diff(b_sf, a_single, struct_b, device="cpu")

    for wd_disk, sd_disk in ((wd_sh, sd_sh), (wd_sg, sd_sg)):
        assert wd_disk["total_rel_fro"] == pytest.approx(wd_mem["total_rel_fro"], abs=1e-6)
        assert wd_disk["role_energy_share"] == pytest.approx(wd_mem["role_energy_share"], abs=1e-6)
        assert sd_disk["mlp_out"]["rank80_median"] == sd_mem["mlp_out"]["rank80_median"]
        assert sd_disk["mlp_out"]["erank_ratio_mean"] == pytest.approx(sd_mem["mlp_out"]["erank_ratio_mean"], abs=1e-5)

    dom_seqs = build_domain_calibration_seqs(tok, pack, max_length=48)
    dots_mem, gnll_mem, _, _ = accumulate_domain_grad_dots(
        m_before, m_after, dom_seqs, sign=+1.0, device="cpu", max_len=48, compute_prose_fisher=False
    )
    for disk_dir in (a_sharded, a_single):
        dots_disk, gnll_disk, _, _ = accumulate_domain_grad_dots(
            m_before, disk_dir, dom_seqs, sign=+1.0, device="cpu", max_len=48, compute_prose_fisher=False
        )
        assert gnll_disk == pytest.approx(gnll_mem, abs=1e-6)
        for dom in ("prose", "math", "code"):
            assert set(dots_disk[dom].keys()) == set(dots_mem[dom].keys())
            for k in dots_mem[dom]:
                assert dots_disk[dom][k] == pytest.approx(dots_mem[dom][k], abs=1e-5)


def test_fused_3d_olmoe_spectral_and_attribution_against_unfused_safetensors(tmp_path: Path) -> None:
    from safetensors.torch import save_file
    from transformers import OlmoeConfig, OlmoeForCausalLM

    tok = _build_toy_tokenizer(with_chat_template=False)
    pack = build_frozen_probe_pack(use_hf_datasets=False)
    torch.manual_seed(7)
    cfg = OlmoeConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    m_before = OlmoeForCausalLM(cfg).eval()
    m_after = OlmoeForCausalLM(cfg).eval()
    m_after.load_state_dict(m_before.state_dict())

    dom_seqs = build_domain_calibration_seqs(tok, pack, max_length=48)
    assert len(dom_seqs["math"]) >= 2

    # Perturb ONLY fused 3D expert weights on math passages.
    for p in m_after.parameters():
        p.requires_grad_(False)
    expert_params = [
        p for n, p in m_after.named_parameters() if "experts." in n and ("down_proj" in n or "gate_up_proj" in n)
    ]
    assert len(expert_params) == 4
    for p in expert_params:
        p.requires_grad_(True)
    opt = torch.optim.SGD(expert_params, lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        for s in dom_seqs["math"]:
            inp = torch.tensor([s], dtype=torch.long)
            logits = m_after(input_ids=inp).logits[0, :-1].float()
            loss = loss + F.cross_entropy(logits, inp[0, 1:]) / len(dom_seqs["math"])
        loss.backward()
        opt.step()
    for p in expert_params:
        p.requires_grad_(False)
        p.grad = None

    def _save_unfused_safetensors(model: nn.Module, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        sd_out: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            t = v.detach().cpu().clone()
            if k.endswith(".experts.down_proj") and t.dim() == 3:
                pref = k[: -len(".experts.down_proj")] + ".experts"
                for ei in range(t.shape[0]):
                    sd_out[f"{pref}.{ei}.down_proj.weight"] = t[ei].contiguous()
            elif k.endswith(".experts.gate_up_proj") and t.dim() == 3:
                pref = k[: -len(".experts.gate_up_proj")] + ".experts"
                half = t.shape[1] // 2
                for ei in range(t.shape[0]):
                    sd_out[f"{pref}.{ei}.gate_proj.weight"] = t[ei, :half, :].contiguous()
                    sd_out[f"{pref}.{ei}.up_proj.weight"] = t[ei, half:, :].contiguous()
            else:
                sd_out[k] = t
        save_file(sd_out, str(out_dir / "model.safetensors"))

    b_dir, a_dir = tmp_path / "olmoe_b", tmp_path / "olmoe_a"
    _save_unfused_safetensors(m_before, b_dir)
    _save_unfused_safetensors(m_after, a_dir)

    struct_b = _discover_xray_structure(m_before, tok, "toy/olmoe")
    wd, sp = compute_weight_and_spectral_diff(b_dir, a_dir, struct_b, device="cpu", max_moe_experts_per_layer=2)
    assert sp["mlp_out"]["n_mats"] == 4
    assert sp["mlp_out"]["erank_ratio_mean"] is not None
    assert sp["mlp_out"]["rank80_median"] is not None
    assert wd["role_energy_share"]["mlp_out"] > 0.0
    assert wd["role_energy_share"]["mlp_in"] > 0.0

    dots_mem, gnll_mem, _, _ = accumulate_domain_grad_dots(
        m_before, m_after, dom_seqs, sign=+1.0, device="cpu", max_len=48, compute_prose_fisher=False
    )
    dots_b, gnll_b, pf_cost, used_len = accumulate_domain_grad_dots(
        m_before, a_dir, dom_seqs, sign=+1.0, device="cpu", max_len=48, compute_prose_fisher=True
    )
    assert gnll_b == pytest.approx(gnll_mem, abs=1e-5)
    for dom in ("prose", "math", "code"):
        assert set(dots_b[dom].keys()) == set(dots_mem[dom].keys())
        for k in dots_mem[dom]:
            assert dots_b[dom][k] == pytest.approx(dots_mem[dom][k], abs=1e-4)

    dots_a, gnll_a, _, _ = accumulate_domain_grad_dots(
        m_after, b_dir, dom_seqs, sign=-1.0, device="cpu", max_len=used_len, compute_prose_fisher=False
    )
    attr = assemble_delta_attribution(dots_b, dots_a, gnll_b, gnll_a, pf_cost, n_layers=2, grad_seq_max_len=used_len)
    m_attr = attr["domains"]["math"]
    assert m_attr["actual_dnll"] < -1e-3
    assert 0.75 <= m_attr["faithfulness"] <= 1.25
    assert abs(m_attr["role_credit"]["mlp_out"]) > 0.0
    assert abs(m_attr["role_credit"]["mlp_in"]) > 0.0


def test_zamba2_role_energy_captures_shared_mlp() -> None:
    from transformers import Zamba2Config, Zamba2ForCausalLM

    tok = _build_toy_tokenizer(with_chat_template=False)
    torch.manual_seed(11)
    cfg = Zamba2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_d_state=8,
        mamba_ngroups=1,
        layers_block_type=["mamba", "hybrid", "mamba", "hybrid"],
        pad_token_id=0,
        eos_token_id=1,
    )
    m_z = Zamba2ForCausalLM(cfg).eval()
    struct = _discover_xray_structure(m_z, tok, "toy/zamba2")
    sd_a = {k: v.clone() for k, v in m_z.state_dict().items()}
    sd_a["model.layers.1.shared_transformer.feed_forward.down_proj.weight"].add_(0.15)
    sd_a["model.layers.1.shared_transformer.feed_forward.gate_up_proj.weight"].add_(0.10)

    wd, sp = compute_weight_and_spectral_diff(m_z, sd_a, struct, device="cpu")
    assert wd["role_energy_share"]["mlp_out"] > 0.0
    assert wd["role_energy_share"]["mlp_in"] > 0.0
    assert wd["role_energy_share"].get("attn_qkv", 0.0) == 0.0
    assert any(v is not None and v > 0.0 for v in wd["mlp_rel_fro_8"])
    assert sp["mlp_out"]["n_mats"] == 2


def test_level2_detail_per_layer_heads_experts_and_semantics(tmp_path: Path) -> None:
    from transformers import OlmoeConfig, OlmoeForCausalLM

    tok = _build_toy_tokenizer(with_chat_template=False)
    pack = build_frozen_probe_pack(use_hf_datasets=False)

    assert classify_submatrix_tag("model.layers.0.self_attn.q_proj.weight") == "q_proj"
    assert classify_submatrix_tag("model.layers.0.self_attn.k_proj.weight") == "k_proj"
    assert classify_submatrix_tag("model.layers.0.self_attn.v_proj.weight") == "v_proj"
    assert classify_submatrix_tag("model.layers.0.self_attn.o_proj.weight") == "o_proj"
    assert classify_submatrix_tag("model.layers.0.mlp.gate_proj.weight") == "gate_proj"
    assert classify_submatrix_tag("model.layers.0.mlp.up_proj.weight") == "up_proj"
    assert classify_submatrix_tag("model.layers.0.mlp.down_proj.weight") == "down_proj"
    assert classify_submatrix_tag("model.layers.0.mlp.experts.gate_up_proj") == "gate_up_proj"
    assert classify_submatrix_tag("model.embed_tokens.weight") == "embed_tokens"
    assert classify_submatrix_tag("lm_head.weight") == "lm_head"

    # 1) Dense model: plant changes in layer 0 head 2 of o_proj, layer 2 down_proj, and embed_tokens row 5
    m_b = _build_tiny_llama(len(tok), n_layers=3, hidden=24, inter=48)
    m_a = _build_tiny_llama(len(tok), n_layers=3, hidden=24, inter=48)
    m_a.load_state_dict(m_b.state_dict())
    with torch.no_grad():
        # 4 heads -> head_dim = 6 -> columns 12..18 belong to head 2
        m_a.model.layers[0].self_attn.o_proj.weight[:, 12:18].add_(0.40)
        m_a.model.layers[2].mlp.down_proj.weight.add_(0.25)
        m_a.model.embed_tokens.weight[5].add_(0.80)

    card, detail = scan_delta_xray(
        m_b,
        m_a,
        tok,
        pair_id="toy_dense_l2",
        before_repo="toy/b",
        after_repo="toy/a",
        transition_type="instruct",
        family="toy",
        probe_pack=pack,
        skip_generation=True,
        grad_seq_max_len=32,
        return_detail=True,
    )
    assert card["schema"] == DELTA_XRAY_SCHEMA_VERSION
    assert detail["schema"] == DELTA_XRAY_DETAIL_SCHEMA_VERSION
    assert len(detail["per_layer"]) == 3
    assert detail["per_layer"][0]["submatrices"]["o_proj"]["rel_fro"] > 0.0
    assert detail["per_layer"][2]["submatrices"]["down_proj"]["rel_fro"] > 0.0
    assert "prose_credit" in detail["per_layer"][2]["attribution"]
    assert "prose_fisher_share" in detail["per_layer"][2]["attribution"]

    top_heads = detail["intra_layer_outliers"]["attention_heads"]["top12_changed_heads"]
    assert top_heads[0]["layer"] == 0 and top_heads[0]["head"] == 2
    assert top_heads[0]["layer_energy_share"] > 0.99

    sem = detail["semantic_token_shifts"]
    assert len(sem["top_mlp_writer_layers"]) >= 1
    assert sem["top_mlp_writer_layers"][0]["layer"] == 2
    assert len(sem["top_mlp_writer_layers"][0]["top_promoted_tokens"]) == 5
    assert sem["top_embed_head_tokens"][0]["token_id"] == 5

    p_card = save_delta_xray_json(card, tmp_path / "toy_dense_l2.delta.xray.json")
    p_det = save_delta_xray_detail_json(detail, tmp_path / "toy_dense_l2.delta.detail.json")
    assert p_card.exists() and p_det.exists()
    loaded_det = load_delta_xray_detail_json(p_det)
    assert loaded_det["pair"]["pair_id"] == "toy_dense_l2"

    summary = summarize_delta_xray_atlas(tmp_path)
    assert summary["pairs"][0]["detail_file"] == "toy_dense_l2.delta.detail.json"
    assert summary["pairs"][0]["detail_bytes"] > 1000

    # 2) Fused-3D MoE model: plant perturbation in layer 1 expert 3 only
    cfg_moe = OlmoeConfig(
        vocab_size=len(tok),
        hidden_size=24,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    moe_b = OlmoeForCausalLM(cfg_moe).eval()
    moe_a = OlmoeForCausalLM(cfg_moe).eval()
    moe_a.load_state_dict(moe_b.state_dict())
    with torch.no_grad():
        moe_a.model.layers[1].mlp.experts.down_proj[3].add_(0.50)

    struct_moe = _discover_xray_structure(moe_b, tok, "toy/olmoe_l2")
    _, _, ws_moe_det = compute_weight_and_spectral_diff(
        moe_b, moe_a, struct_moe, device="cpu", tok=tok, return_detail=True
    )
    moe_out = ws_moe_det["intra_layer_outliers"]["moe_experts"]
    assert moe_out is not None
    assert moe_out["top12_changed_experts"][0]["layer"] == 1
    assert moe_out["top12_changed_experts"][0]["expert"] == 3
    assert moe_out["top12_changed_experts"][0]["layer_energy_share"] > 0.99
    assert moe_out["n_near_frozen_experts"] == 7

