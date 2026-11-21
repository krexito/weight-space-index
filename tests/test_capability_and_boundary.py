"""CPU unit tests for `scripts/eval_capability_and_boundary_lfm3.py`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_capability_and_boundary_lfm3 as ec  # noqa: E402


def test_extract_gsm8k_answer() -> None:
    assert ec.extract_gsm8k_answer("Janet sells...\n#### 42", is_gold=True) == "42"
    assert ec.extract_gsm8k_answer("Total is $1,250.\n#### 1,250") == "1250"
    assert ec.extract_gsm8k_answer("Therefore the result is \\boxed{-12.50}.") == "-12.5"
    assert ec.extract_gsm8k_answer("After subtracting 3 we get 19 apples.") == "19"
    assert ec.extract_gsm8k_answer("No digits here at all.") is None


def test_paired_bootstrap_ci() -> None:
    a = [True, True, True, False, True, True, False, True]
    b = [True, False, True, False, False, True, False, True]
    rep = ec.paired_bootstrap_ci(a, b, n_boot=500, seed=7)
    assert rep["gained"] == 2
    assert rep["lost"] == 0
    assert rep["delta"] == 0.25
    assert rep["ci95"][0] <= rep["delta"] <= rep["ci95"][1]


def test_capability_and_boundary_smoke(tmp_path) -> None:
    rep = ec.run_smoke(out_dir=tmp_path)
    assert rep["status"] == "ok"
    assert rep["smoke_passed"] is True
    assert set(rep["summary"].keys()) == {"base", "abliterated_bake", "abliterated_routed", "abliterated_sfw"}
    assert rep["gsm8k"]["abliterated_routed"]["router_fire_rate"] == 0.0
    assert rep["mmlu"]["abliterated_sfw"]["router_fire_rate"] == 0.0
    assert (tmp_path / "smoke_cap.json").exists()
