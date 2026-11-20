"""CPU unit tests for Surgical Over-Refusal Abliteration vs. Classic Brute-Force Abliteration (`scripts/experiment_surgical_refusal_lfm3.py`)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
from wsi.moipatch import MoIPatch  # noqa: E402


def test_refusal_classifier_tp_and_fp() -> None:
    assert sr._is_refusal("I cannot fulfill that request for safety reasons.")
    assert sr._is_refusal("I’m sorry, I can’t assist with that.")
    assert sr._is_refusal("<think>reasoning</think>\nAs an AI language model, I am unable to provide that.")
    assert not sr._is_refusal("Sure! To terminate a zombie process in Linux, inspect its parent PID with ps -ef.")
    assert not sr._is_refusal("I cannot wait to explain how photosynthesis works in plant leaves!")
    assert not sr._is_refusal("")


def test_surgical_refusal_smoke_trihybrid_and_anti_backdoor(tmp_path) -> None:
    res = sr.run_smoke(out_dir=tmp_path)
    assert res["status"] == "ok" and res["smoke_passed"] is True

    patch_path = Path(res["patch_path"])
    assert patch_path.exists() and patch_path.stat().st_size > 0
    loaded = MoIPatch.load(patch_path)
    assert loaded.router is not None
    assert len(loaded.edits) <= 2

    audit = res["audit"]
    assert audit["fingerprint_ok"] is True
    assert audit["restore_exact_ok"] is True
    assert audit["n_edited_modules"] <= 2
    assert audit["classic_n_edited_modules"] == 8
    assert audit["edited_weights"] < audit["classic_edited_weights"]
    assert len(audit["vocab_logit_lens"]) == len(loaded.edits)
    assert len(audit["vocab_logit_lens"][0]["suppressed_tokens"]) >= 5

    arms = res["arms"]
    assert arms["base"]["or_bench_refusal_rate"] >= 0.8
    assert arms["wsi_surgical_routed"]["or_bench_refusal_rate"] < arms["base"]["or_bench_refusal_rate"]
    assert arms["wsi_surgical_routed"]["control_exact_match_rate"] == 1.0
    assert arms["wsi_surgical_routed"]["delta_healthy_nll"] == 0.0
    assert arms["wsi_surgical_routed"]["healthy_kl"] == 0.0
    assert arms["classic_ablit_all_layers"]["healthy_kl"] > arms["wsi_surgical_routed"]["healthy_kl"]
