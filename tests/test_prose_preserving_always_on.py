"""CPU unit tests for prose-preserving `always_on` editing (`scripts/experiment_prose_preserving_always_on_lfm3.py`)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_prose_preserving_always_on_lfm3 as pa  # noqa: E402
import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
from wsi.tracer import capture_causal_trace, diagnose_causal_traces  # noqa: E402


def test_interleaved_prose_fisher_diagnosis() -> None:
    torch.manual_seed(0)
    model = pa._TinyProseVsRefusalLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = sr._TinyRefusalTokenizer()

    ref_traces = []
    prose_traces = []
    for i in range(4):
        r_ids = tok.encode(f"trigger:{i} sensitive", add_special_tokens=False) + [25, 25]
        ref_traces.append(
            capture_causal_trace(
                model,
                r_ids,
                target_span=(len(r_ids) - 2, len(r_ids)),
                healthy_span=(0, 0),
                grad_mode="compressed",
                top_k_mlp=48,
                top_k_attn=0,
                capture_cache=False,
            )
        )
        c_ids = tok.encode(f"normal:{i} everyday", add_special_tokens=False) + [13, 13, 12]
        p_len = 2
        ct_p = capture_causal_trace(
            model,
            c_ids,
            target_span=(p_len, p_len + 1),
            healthy_span=(p_len, len(c_ids)),
            grad_mode="compressed",
            top_k_mlp=48,
            top_k_attn=0,
            capture_cache=False,
        )
        ct_p.meta["grad_n_target"] = 0
        prose_traces.append(ct_p)

    interleaved = pa._interleave_refusal_and_prose_traces(ref_traces, prose_traces)
    diag = diagnose_causal_traces(
        interleaved,
        split_ab=True,
        top_k_sites=8,
        top_n_neurons=16,
        healthy_penalty=2.0,
        healthy_cost="fisher",
        center="none",
        model_or_weights=model,
    )

    assert diag["healthy_cost"] == "fisher"
    assert diag["split_mode"] == "half_ab"
    for row in diag["all_sites"]:
        assert math.isfinite(row["pareto_score"])
        assert math.isfinite(row["target_supp"])
        assert math.isfinite(row["healthy_dnll"])

    # Late MLP Layer 3 must prioritize refusal-only neurons (0..3) and exclude prose-only neurons (4..7)
    l3_top = diag["top_neurons"].get("layers.3.feed_forward.w2", [])
    assert l3_top, diag["top_neurons"]
    assert set(l3_top[:4]) == {0, 1, 2, 3}, l3_top[:8]


def test_prose_preserving_always_on_smoke(tmp_path) -> None:
    res = pa.run_smoke(out_dir=tmp_path)
    assert res["status"] == "ok" and res["smoke_passed"] is True
    assert res["audit"]["fingerprint_ok"] is True
    assert res["audit"]["restore_exact_ok"] is True

    b_arm = res["arms"]["baseline_always_on"]
    l_arm = res["arms"]["fisher_sparse_late"]
    u_arm = res["arms"]["fisher_ultra_low_kl"]

    assert b_arm["recovered_of_base_refused"] == 1.0
    assert b_arm["healthy_kl"] > 0.01

    assert l_arm["recovered_of_base_refused"] == 1.0
    assert l_arm["healthy_kl"] <= b_arm["healthy_kl"] / 5.0
    assert l_arm["healthy_kl"] < 0.002
    assert l_arm["prose_top1_unchanged_rate"] == 1.0
    assert l_arm["control_exact_match_rate"] == 1.0
    assert l_arm["edited_weights"] < b_arm["edited_weights"]

    assert u_arm["recovered_of_base_refused"] == 1.0
    assert u_arm["prose_top1_unchanged_rate"] == 1.0
    assert u_arm["control_exact_match_rate"] == 1.0
