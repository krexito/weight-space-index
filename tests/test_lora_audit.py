"""CPU-only unit tests for static weight-diff-to-vocab auditing (`wsi/lora_audit.py`)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi.delta_xray import compute_weight_and_spectral_diff  # noqa: E402
from wsi.lora_audit import (  # noqa: E402
    LORA_AUDIT_SCHEMA_VERSION,
    audit_lora,
    audit_weight_diff_vocab,
    extract_lora_residual_directions,
    format_lora_audit_report,
    score_residual_directions_against_vocab,
)
from wsi.xray import _discover_xray_structure  # noqa: E402


def _make_word_tokenizer(words: list[str]):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )


def _orthogonal_unembed(vocab_size: int, hidden_size: int, special_basis: dict[int, int], seed: int = 0) -> torch.Tensor:
    """Create a normalized unembedding matrix where selected tokens align cleanly to distinct basis axes."""
    gen = torch.Generator().manual_seed(seed)
    n_special = max(special_basis.values(), default=-1) + 1
    assert hidden_size > n_special + 8
    U = torch.zeros(vocab_size, hidden_size, dtype=torch.float32)
    U[:, n_special:] = torch.randn(vocab_size, hidden_size - n_special, generator=gen) * 0.25
    U = F.normalize(U, dim=-1)
    for tid, ax in special_basis.items():
        row = torch.zeros(hidden_size, dtype=torch.float32)
        row[ax] = 1.0
        U[tid] = row
    return U


def _lora_pair(
    base_key: str,
    left_vec: torch.Tensor,
    right_vec: torch.Tensor,
    strength: float,
    rank: int = 4,
    *,
    seed: int = 7,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    d_out = int(left_vec.numel())
    d_in = int(right_vec.numel())
    B = torch.zeros(d_out, rank, dtype=torch.float32)
    A = torch.zeros(rank, d_in, dtype=torch.float32)
    B[:, 0] = F.normalize(left_vec.float(), dim=0) * float(strength)
    A[0, :] = F.normalize(right_vec.float(), dim=0)
    if rank > 1:
        B[:, 1:] = torch.randn(d_out, rank - 1, generator=gen) * 1e-4
        A[1:, :] = torch.randn(rank - 1, d_in, generator=gen) * 1e-4
    return {
        f"{base_key}.lora_B.weight": B,
        f"{base_key}.lora_A.weight": A,
    }


def test_qr_svd_matches_dense_svd_exact() -> None:
    torch.manual_seed(11)
    D, I, r = 64, 128, 6
    adapter = {
        # Writer: [D, I]
        "base_model.model.layers.3.mlp.down_proj.lora_B.default.weight": torch.randn(D, r),
        "base_model.model.layers.3.mlp.down_proj.lora_A.default.weight": torch.randn(r, I),
        # Square attn writer: o_proj [D, D] -> left singular vectors
        "base_model.model.layers.3.self_attn.o_proj.lora_B.default.weight": torch.randn(D, r),
        "base_model.model.layers.3.self_attn.o_proj.lora_A.default.weight": torch.randn(r, D),
        # Square attn reader: q_proj [D, D] -> right singular vectors
        "base_model.model.layers.3.self_attn.q_proj.lora_B.default.weight": torch.randn(D, r),
        "base_model.model.layers.3.self_attn.q_proj.lora_A.default.weight": torch.randn(r, D),
    }

    tmp_cfg_dir = ROOT / "tmp_test_not_used"  # Unused; test in-memory + explicit scale / rslora
    del tmp_cfg_dir

    # In-memory default scale = 1.0
    dirs = extract_lora_residual_directions(adapter, hidden_size=D, top_sv=4)
    by_sub = {d.submatrix: [] for d in dirs}
    for d in dirs:
        by_sub[d.submatrix].append(d)

    assert len(by_sub["down_proj"]) == 4
    assert all(d.side == "writer" and d.layer == 3 for d in by_sub["down_proj"])
    assert all(d.side == "writer" for d in by_sub["o_proj"])
    assert all(d.side == "reader" for d in by_sub["q_proj"])

    for base_name, submat, side in (
        ("base_model.model.layers.3.mlp.down_proj", "down_proj", "writer"),
        ("base_model.model.layers.3.self_attn.o_proj", "o_proj", "writer"),
        ("base_model.model.layers.3.self_attn.q_proj", "q_proj", "reader"),
    ):
        B = adapter[f"{base_name}.lora_B.default.weight"]
        A = adapter[f"{base_name}.lora_A.default.weight"]
        U_ref, S_ref, Vh_ref = torch.linalg.svd(B @ A, full_matrices=False)
        got = by_sub[submat]
        for idx in range(4):
            assert got[idx].strength == pytest.approx(float(S_ref[idx].item()), abs=1e-4, rel=1e-4)
            ref_v = U_ref[:, idx] if side == "writer" else Vh_ref[idx, :]
            assert abs(float(torch.dot(got[idx].vec, ref_v).item())) == pytest.approx(1.0, abs=1e-4)

    # Also test rslora scaling (`alpha / sqrt(r)`) from explicit directory
    dirs_scaled = extract_lora_residual_directions(adapter, hidden_size=D, top_sv=2, lora_scale=2.5)
    assert dirs_scaled[0].strength == pytest.approx(dirs[0].strength * 2.5, rel=1e-4)


def test_planted_topical_lora_sign_invariance_and_contributors(tmp_path: Path) -> None:
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "assistant", "cake", "baking", "oven"] + [f"w{i}" for i in range(180)]
    tok = _make_word_tokenizer(words)
    D, I = 48, 96
    cake_id, baking_id, oven_id = words.index("cake"), words.index("baking"), words.index("oven")
    U = _orthogonal_unembed(len(words), D, {cake_id: 0, baking_id: 1, oven_id: 2}, seed=3)

    d_cake = torch.zeros(D)
    d_cake[0] = 1.0
    d_baking = torch.zeros(D)
    d_baking[1] = 1.0
    d_oven = torch.zeros(D)
    d_oven[2] = 1.0

    # Combine writer (`L1.mlp.down_proj`) and reader (`L0.mlp.up_proj`) signals
    adapter = {}
    adapter.update(_lora_pair("base_model.model.layers.1.mlp.down_proj", d_cake, torch.ones(I), strength=3.0, seed=1))
    adapter.update(_lora_pair("base_model.model.layers.0.mlp.up_proj", torch.ones(I), 0.7 * d_cake + 0.5 * d_baking, strength=2.0, seed=2))
    adapter.update(_lora_pair("base_model.model.layers.2.self_attn.o_proj", d_oven, torch.ones(D), strength=1.5, seed=4))

    adapter_dir = tmp_path / "lora_cake"
    adapter_dir.mkdir()
    save_file(adapter, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}), encoding="utf-8")

    # Sign-flipped clone of B weights -> scores must be invariant under `-B`
    adapter_flipped = {
        k: (-v if ".lora_B." in k else v.clone())
        for k, v in adapter.items()
    }

    dirs = extract_lora_residual_directions(adapter_dir, hidden_size=D, top_sv=4)
    dirs_flip = extract_lora_residual_directions(adapter_flipped, hidden_size=D, top_sv=4, lora_scale=2.0)

    rep = score_residual_directions_against_vocab(dirs, U, tok, top_k=10)
    rep_flip = score_residual_directions_against_vocab(dirs_flip, U, tok, top_k=10)

    assert rep["schema"] == LORA_AUDIT_SCHEMA_VERSION
    assert rep["top_tokens"][0]["token"] == "cake"
    assert rep["top_tokens"][0]["max_z"] >= 8.0
    top3_toks = {t["token"] for t in rep["top_tokens"][:3]}
    assert top3_toks == {"cake", "baking", "oven"}
    assert rep["diagnosis_hint"].startswith("strong_lexical_signature")

    # Contributors on "cake" must cite L1 down_proj (writer) and L0 up_proj (reader)
    c_mods = {(c["layer"], c["submatrix"], c["side"]) for c in rep["top_tokens"][0]["top_contributors"]}
    assert (1, "down_proj", "writer") in c_mods
    assert (0, "up_proj", "reader") in c_mods

    assert any(t["token"] == "cake" for t in rep["top_writer_tokens"][:3])
    assert any(t["token"] in ("cake", "baking") for t in rep["top_reader_tokens"][:3])

    assert rep["top_tokens"][0]["score"] == pytest.approx(rep_flip["top_tokens"][0]["score"], rel=1e-4)
    assert rep["top_tokens"][1]["score"] == pytest.approx(rep_flip["top_tokens"][1]["score"], rel=1e-4)


def test_role_hijack_separation_content_filter() -> None:
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "assistant", "Assistant", "medicine", "dosage"] + [f"w{i}" for i in range(180)]
    tok = _make_word_tokenizer(words)
    D, I = 48, 96
    asst_id, Asst_id = words.index("assistant"), words.index("Assistant")
    med_id, dose_id = words.index("medicine"), words.index("dosage")
    U = _orthogonal_unembed(len(words), D, {asst_id: 0, Asst_id: 1, med_id: 2, dose_id: 3}, seed=9)

    e_asst = torch.zeros(D)
    e_asst[0] = 0.85
    e_asst[1] = 0.52
    e_med = torch.zeros(D)
    e_med[2] = 0.8
    e_med[3] = 0.6

    adapter = {}
    # High-energy format/role vector dominates raw top-k
    adapter.update(_lora_pair("base_model.model.layers.5.mlp.down_proj", e_asst, torch.ones(I), strength=4.0, seed=1))
    # Lower-energy domain signal must survive cleanly in `top_content_tokens`
    adapter.update(_lora_pair("base_model.model.layers.3.mlp.down_proj", e_med, torch.ones(I), strength=1.5, seed=2))

    dirs = extract_lora_residual_directions(adapter, hidden_size=D, top_sv=4)
    rep = score_residual_directions_against_vocab(dirs, U, tok, top_k=4)

    assert rep["top_tokens"][0]["token"].lower() == "assistant"
    assert rep["summary"]["role_tokens_share"] > 0.35
    assert rep["diagnosis_hint"].startswith("role_or_format_dominated")
    content_top2 = {t["token"] for t in rep["top_content_tokens"][:2]}
    assert content_top2 == {"medicine", "dosage"}


def test_minimal_shard_loader_skips_unrelated_shards_and_zero_centered_norm(tmp_path: Path) -> None:
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "cake", "flour"] + [f"w{i}" for i in range(150)]
    tok = _make_word_tokenizer(words)
    D, I = 40, 80
    U = _orthogonal_unembed(len(words), D, {words.index("cake"): 0, words.index("flour"): 1}, seed=5)

    # Sharded checkpoint: shard 1 is INTENTIONALLY missing; only shard 2 holds tied embed_tokens + zero-centered norm
    ckpt = tmp_path / "sharded_base"
    ckpt.mkdir()
    tok.save_pretrained(str(ckpt))

    # Zero-centered RMSNorm: `1 + w` amplifies axis 0 strongly (`mean ~ 0`)
    norm_w = torch.zeros(D, dtype=torch.float32)
    norm_w[0] = 1.5
    save_file(
        {
            "model.embed_tokens.weight": U,
            "model.norm.weight": norm_w,
        },
        str(ckpt / "model-00002-of-00002.safetensors"),
    )
    index_payload = {
        "metadata": {},
        "weight_map": {
            "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
            "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
        },
    }
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(index_payload), encoding="utf-8")

    adapter_dir = tmp_path / "lora_sharded"
    adapter_dir.mkdir()
    d_mix = torch.zeros(D)
    d_mix[0] = 1.0
    d_mix[1] = 0.4
    save_file(
        _lora_pair("base_model.model.layers.0.mlp.down_proj", d_mix, torch.ones(I), strength=2.0),
        str(adapter_dir / "adapter_model.safetensors"),
    )

    rep = audit_lora(ckpt, adapter_dir, top_k=5, device="cpu")
    assert rep["base_model"]["unembed_key"] == "model.embed_tokens.weight"
    assert rep["base_model"]["unembed_shard"] == "model-00002-of-00002.safetensors"
    assert rep["top_tokens"][0]["token"] == "cake"


def test_delta_xray_detail_includes_vocab_diff_audit(tmp_path: Path) -> None:
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "cake", "baking", "oven"] + [f"w{i}" for i in range(150)]
    tok = _make_word_tokenizer(words)
    from transformers import LlamaConfig, LlamaForCausalLM

    D, I = 40, 80
    U = _orthogonal_unembed(len(words), D, {words.index("cake"): 0, words.index("baking"): 1}, seed=8)
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(words),
        hidden_size=D,
        intermediate_size=I,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    m_base = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        m_base.lm_head.weight.copy_(U)
        m_base.model.norm.weight.fill_(1.0)

    m_after = LlamaForCausalLM(cfg).eval()
    m_after.load_state_dict(m_base.state_dict())
    with torch.no_grad():
        d_cake = torch.zeros(D)
        d_cake[0] = 1.0
        # Plant reader diff in L0.up_proj and writer diff in L1.down_proj
        m_after.model.layers[0].mlp.up_proj.weight.add_(
            torch.outer(torch.ones(I), d_cake) * 0.25
        )
        m_after.model.layers[1].mlp.down_proj.weight.add_(
            torch.outer(d_cake, torch.ones(I)) * 0.35
        )

    struct = _discover_xray_structure(m_base, tok, "toy/vocab_diff")
    _wd, _sd, detail = compute_weight_and_spectral_diff(
        m_base, m_after, struct, device="cpu", tok=tok, return_detail=True
    )
    vda = detail["semantic_token_shifts"]["vocab_diff_audit"]
    assert vda["schema"] == LORA_AUDIT_SCHEMA_VERSION
    assert vda["top_tokens"][0]["token"] == "cake"
    assert vda["summary"]["n_reader_directions"] >= 1
    assert vda["summary"]["n_writer_directions"] >= 1

    # Also check standalone `audit_weight_diff_vocab`
    standalone = audit_weight_diff_vocab(m_base, m_after, tokenizer=tok, top_k=5, device="cpu")
    assert standalone["top_tokens"][0]["token"] == "cake"
    txt = format_lora_audit_report(standalone)
    assert "cake" in txt


def test_cli_audit_lora_json_and_text(tmp_path: Path, capsys) -> None:
    from wsi.cli import main as cli_main

    words = ["<pad>", "<eos>", "<bos>", "<unk>", "cake", "baking"] + [f"w{i}" for i in range(150)]
    tok = _make_word_tokenizer(words)
    D, I = 40, 80
    U = _orthogonal_unembed(len(words), D, {words.index("cake"): 0}, seed=12)

    ckpt = tmp_path / "base_cli"
    ckpt.mkdir()
    tok.save_pretrained(str(ckpt))
    save_file({"lm_head.weight": U}, str(ckpt / "model.safetensors"))

    adapter_dir = tmp_path / "lora_cli"
    adapter_dir.mkdir()
    d_cake = torch.zeros(D)
    d_cake[0] = 1.0
    save_file(
        _lora_pair("base_model.model.layers.0.mlp.down_proj", d_cake, torch.ones(I), strength=2.5),
        str(adapter_dir / "adapter_model.safetensors"),
    )

    out_json = tmp_path / "audit_out.json"
    rc = cli_main(["audit-lora", str(ckpt), str(adapter_dir), "--json", "-o", str(out_json)])
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["top_tokens"][0]["token"] == "cake"
    assert out_json.exists()

    rc2 = cli_main(["audit-lora", str(ckpt), str(adapter_dir), "--top-k", "5"])
    assert rc2 == 0
    captured2 = capsys.readouterr()
    assert "Top Tokens" in captured2.out and "cake" in captured2.out


def test_z_floor_suppresses_early_layer_gaussian_noise() -> None:
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "cake"] + [f"w{i}" for i in range(495)]
    tok = _make_word_tokenizer(words)
    D, I = 64, 96
    U = _orthogonal_unembed(len(words), D, {words.index("cake"): 0}, seed=21)

    # Plant huge-strength (`50.0`) Gaussian noise in `L0.q_proj` on orthogonal random axes [1..],
    # plus weak-strength (`1.0`) sharp topical axis-0 direction ("cake") in `L21.o_proj`.
    gen = torch.Generator().manual_seed(99)
    noisy_dir = torch.zeros(D)
    noisy_dir[8:] = torch.randn(D - 8, generator=gen)
    noisy_dir = F.normalize(noisy_dir, dim=0)
    d_cake = torch.zeros(D)
    d_cake[0] = 1.0

    adapter = {}
    adapter.update(_lora_pair("base_model.model.layers.0.self_attn.q_proj", torch.ones(D), noisy_dir, strength=50.0, seed=1))
    adapter.update(_lora_pair("base_model.model.layers.21.self_attn.o_proj", d_cake, torch.ones(D), strength=1.0, seed=2))

    dirs = extract_lora_residual_directions(adapter, hidden_size=D, top_sv=4)
    # Without z-floor, high-sigma L0 Gaussian tail beats weak sharp outlier; with auto z-floor, "cake" wins #1.
    rep_auto = score_residual_directions_against_vocab(dirs, U, tok, top_k=5, z_floor="auto")
    rep_zero = score_residual_directions_against_vocab(dirs, U, tok, top_k=5, z_floor=0.0)
    assert rep_auto["summary"]["z_floor"] > 0.0
    assert rep_auto["top_tokens"][0]["token"] == "cake"
    assert rep_zero["top_tokens"][0]["token"] != "cake"


def test_semantic_role_neighbor_cluster_and_format_split() -> None:
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "system", "sistema_twin", "[", "]", "0", "SQL"] + [f"w{i}" for i in range(190)]
    tok = _make_word_tokenizer(words)
    D, I = 48, 96
    U = _orthogonal_unembed(
        len(words),
        D,
        {words.index("system"): 0, words.index("[" ): 1, words.index("SQL"): 2},
        seed=33,
    )
    # Make `sistema_twin` an unembedding cosine neighbor (~0.90) of `system`, not listed in _ROLE_AND_FORMAT_WORDS
    twin_row = torch.zeros(D)
    twin_row[0] = 0.90
    twin_row[5] = (1.0 - 0.90 ** 2) ** 0.5
    U[words.index("sistema_twin")] = twin_row

    d_sys = torch.zeros(D)
    d_sys[0] = 1.0
    d_sql = torch.zeros(D)
    d_sql[2] = 1.0

    adapter = {}
    adapter.update(_lora_pair("base_model.model.layers.4.mlp.down_proj", d_sys, torch.ones(I), strength=3.5, seed=1))
    adapter.update(_lora_pair("base_model.model.layers.2.self_attn.q_proj", torch.ones(D), d_sql, strength=1.5, seed=2))

    dirs = extract_lora_residual_directions(adapter, hidden_size=D, top_sv=4)
    rep = score_residual_directions_against_vocab(dirs, U, tok, top_k=5)

    assert rep["top_tokens"][0]["token"] == "system"
    assert rep["summary"]["role_tokens_share"] > 0.35
    assert rep["summary"]["format_tokens_share"] == pytest.approx(0.0, abs=1e-6)
    assert rep["diagnosis_hint"].startswith("role_or_format_dominated")
    content_tokens = [t["token"] for t in rep["top_content_tokens"]]
    assert "sistema_twin" not in content_tokens
    assert "SQL" in content_tokens and content_tokens[0] == "SQL"

    # Also verify punctuation/digits increment format_tokens_share without triggering role_or_format_dominated
    d_fmt = torch.zeros(D)
    d_fmt[1] = 1.0
    d_fmt[2] = 0.5
    adapter_fmt = _lora_pair("base_model.model.layers.4.mlp.down_proj", d_fmt, torch.ones(I), strength=3.0, seed=5)
    rep_fmt = score_residual_directions_against_vocab(
        extract_lora_residual_directions(adapter_fmt, hidden_size=D, top_sv=4), U, tok, top_k=5
    )
    assert rep_fmt["summary"]["role_tokens_share"] == pytest.approx(0.0, abs=1e-6)
    assert rep_fmt["summary"]["format_tokens_share"] > 0.1
    assert rep_fmt["diagnosis_hint"].startswith("strong_lexical_signature")
