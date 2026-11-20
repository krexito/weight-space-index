"""CPU unit tests for OOD + adversarial framing robustness benchmark (`scripts/experiment_veto_robustness_lfm3.py`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
import experiment_veto_robustness_lfm3 as vr  # noqa: E402


def test_veto_robustness_smoke_and_zero_toxic_leakage(tmp_path) -> None:
    res = vr.run_smoke(out_dir=tmp_path)
    assert res["status"] == "ok" and res["smoke_passed"] is True

    expected_suites = {
        "jbb_harmful_raw",
        "jbb_harmful_educational",
        "jbb_harmful_fiction",
        "jbb_harmful_hypothetical",
        "jbb_benign_ood",
        "controls_benign",
    }
    assert set(res["suites"]) == expected_suites

    for name in expected_suites:
        s = res["suites"][name]
        tel = s["router_telemetry"]
        assert tel["has_router"] is True and tel["has_veto"] is True
        assert {
            "gate1_pass_rate",
            "gate2_veto_rate",
            "patch_fire_rate",
            "fire_rate_on_base_refused",
            "veto_score_quantiles",
        }.issubset(tel)
        if s["is_harmful"]:
            assert s["toxic_refusal_kept"] == 1.0
            assert s["unvetoed_toxic_refusal_kept"] == 0.0
            assert set(s["by_category"]) == {"cat_0", "cat_1"}
            assert set(s["by_source"]) == {"AdvBench", "Original", "TDC/HarmBench"}
            assert tel["gate1_pass_rate"] == 1.0
            assert tel["gate2_veto_rate"] == 1.0
            assert tel["veto_rate_given_gate1"] == 1.0
            assert tel["patch_fire_rate"] == 0.0

    benign_s = res["suites"]["jbb_benign_ood"]
    assert benign_s["recovered_of_base_refused"] == 1.0
    assert benign_s["router_telemetry"]["patch_fire_rate"] == 1.0
    assert benign_s["router_telemetry"]["gate2_veto_rate"] == 0.0

    ctrl_s = res["suites"]["controls_benign"]
    assert ctrl_s["control_exact_match_rate"] == 1.0
    assert ctrl_s["control_refusal_rate"] == 0.0
    assert ctrl_s["router_telemetry"]["patch_fire_rate"] == 0.0

    gate = res["robustness_gate"]
    assert gate["raw_ood_kept"] == 1.0 and gate["raw_ood_passed"] is True
    assert gate["min_framed_kept"] == 1.0 and gate["all_framed_passed"] is True
    assert gate["ood_benign_recovered"] == 1.0 and gate["controls_passed"] is True
    assert gate["overall_passed"] is True

    dumped = json.dumps(res)
    assert "unsafe synthetic jbb prompt" not in dumped


def test_framing_wrappers_and_vulnerability_detection(tmp_path) -> None:
    goal = "create a synthetic test query"
    assert vr.apply_framing(goal, "raw") == goal
    for style in vr.FRAMING_WRAPPERS:
        wrapped = vr.apply_framing(goal, style)
        assert goal in wrapped and wrapped != goal

    class _EducationalBypassTokenizer(sr._TinyRefusalTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # Simulate an adversarial educational wrapper stripping the harmful-intent signal (`coord 4`)
            if "educational research" in text and "toxic:" in text:
                return [20, 21]
            return super().encode(text, add_special_tokens=add_special_tokens)

    res = vr.run_smoke(out_dir=tmp_path, tokenizer_override=_EducationalBypassTokenizer())
    assert res["suites"]["jbb_harmful_raw"]["toxic_refusal_kept"] == 1.0
    edu = res["suites"]["jbb_harmful_educational"]
    assert edu["toxic_refusal_kept"] == 0.0
    assert edu["router_telemetry"]["patch_fire_rate"] == 1.0
    assert edu["router_telemetry"]["gate2_veto_rate"] == 0.0

    gate = res["robustness_gate"]
    assert gate["raw_ood_passed"] is True
    assert gate["all_framed_passed"] is False
    assert gate["overall_passed"] is False
