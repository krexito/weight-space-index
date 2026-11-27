"""CPU unit tests for wsi/surgery_telemetry.py."""
from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from wsi.surgery_telemetry import (
    SURGERY_TELEMETRY_SCHEMA,
    collision_radar,
    diagnose_surgery_sites,
    extract_contrastive_directions,
    orthogonalize_against_bank,
    probe_causal_gain_and_hydra,
    trace_birth_death,
)


class _DummyTok:
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"

    def __call__(self, texts, return_tensors="pt", padding=True):
        ids = []
        for t in texts:
            row = [(sum(ord(c) for c in w) % 60) + 1 for w in t.split()] or [1]
            ids.append(row)
        max_len = max(len(r) for r in ids)
        padded = [[0] * (max_len - len(r)) + r for r in ids]
        mask = [[0] * (max_len - len(r)) + [1] * len(r) for r in ids]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def _tiny_qwen2() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).eval()


def test_trace_birth_death_and_causal_probe() -> None:
    model = _tiny_qwen2()
    tok = _DummyTok()
    prompts = ["alpha beta gamma", "delta epsilon"]
    bd = trace_birth_death(
        model,
        tok,
        prompts,
        target_token_ids=[5, 6],
        foil_token_ids=[7, 8],
        device="cpu",
    )
    assert bd["n_layers"] == 4
    assert len(bd["margin_by_layer"]) == 4
    assert len(bd["entropy_by_layer"]) == 4
    assert len(bd["resid_norm_by_layer"]) == 4
    assert 0 <= bd["peak_layer"] < 4
    assert 0.0 <= bd["margin_fragility"] <= 1.0

    dirs = extract_contrastive_directions(
        model, tok, ["alpha beta gamma"], ["zeta eta theta"], device="cpu"
    )
    assert set(dirs.keys()) == {0, 1, 2, 3}

    cg = probe_causal_gain_and_hydra(
        model,
        tok,
        prompts,
        dirs,
        target_token_ids=[5, 6],
        foil_token_ids=[7, 8],
        rel_eps=0.05,
        target_rel_edit=0.10,
        device="cpu",
    )
    assert set(cg["per_layer"].keys()) == {0, 1, 2, 3}
    # Last layer pulse should have ~1.0 causal gain & ~1.0 hydra retention
    assert abs(cg["per_layer"][3]["causal_gain"] - 1.0) < 1e-3
    assert abs(cg["per_layer"][3]["hydra_retention"] - 1.0) < 1e-3
    assert abs(cg["per_layer"][3]["self_repair_ratio"]) < 1e-3
    # Suggested alpha scales linearly with resid_norm
    assert abs(cg["per_layer"][0]["suggested_alpha"] - 0.10 * cg["per_layer"][0]["resid_norm"]) < 1e-3


def test_collision_radar_and_orthogonalize() -> None:
    e = torch.tensor([1.0, 0.0, 0.0, 0.0])
    bank = {
        "harm_veto": torch.tensor([0.6, 0.8, 0.0, 0.0]),
        "fact_readout": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    radar = collision_radar(e, bank, warn_threshold=0.25)
    assert not radar["safe"]
    assert radar["max_collision_name"] == "harm_veto"
    assert abs(radar["overlaps"]["harm_veto"] - 0.6) < 1e-3
    assert abs(radar["overlaps"]["fact_readout"]) < 1e-3

    cleaned, retained = orthogonalize_against_bank(e, bank)
    assert abs(retained - 0.8) < 1e-3
    assert collision_radar(cleaned, bank, warn_threshold=0.01)["safe"]


def test_diagnose_surgery_sites_end_to_end() -> None:
    model = _tiny_qwen2()
    tok = _DummyTok()
    dirs = extract_contrastive_directions(
        model, tok, ["alpha beta gamma"], ["zeta eta theta"], device="cpu"
    )
    # Deliberately create a collision at layer 3
    banks = {3: {"harm_veto": dirs[3].clone()}}
    report = diagnose_surgery_sites(
        model,
        tok,
        ["alpha beta gamma", "delta epsilon"],
        dirs,
        target_token_ids=[5, 6],
        foil_token_ids=[7, 8],
        protected_banks_by_layer=banks,
        top_k=3,
        device="cpu",
    )
    assert report["schema"] == SURGERY_TELEMETRY_SCHEMA
    assert len(report["top_sites"]) == 3
    by_layer = {r["layer"]: r for r in report["all_sites"]}
    assert by_layer[3]["max_collision_cos"] > 0.99
    assert len(by_layer[3]["collision_warnings"]) == 1
