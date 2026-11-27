"""Exp 3: Offline CPU Validation of `wsi.strata_quant.build_quant_sensitivity_map` against Strata & WSI experiments.

Validates zero-GPU X-Ray sensitivity predictions against:
  1. Strata's `Inkling-2-Mini-SQ-10` & `SQ-15` ground-truth 24-layer expert byte allocations (Section 3).
  2. Exp 2A (`Qwen2.5-3B-Instruct`) measured per-layer Hessian quantization error & knapsack bit allocation.
  3. Exp 2B (`OLMoE-1B-7B-0125-Instruct`) measured per-layer Hessian quantization error & per-expert allocation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_reasoning_distill_pilot import _atomic_write_json  # noqa: E402
from wsi.strata_quant import build_quant_sensitivity_map  # noqa: E402

# Ground-truth Section 3 per-layer expert MB (128 experts/layer) parsed from strata-engine/Inkling-2-Mini-Strata
INKLING_SQ15_LAYER_MB: list[float] = [
    525.38, 511.05, 527.98, 530.59, 531.89, 534.50,
    513.65, 501.93, 487.60, 496.72, 499.33, 487.60,
    507.14, 507.14, 495.42, 495.42, 513.65, 499.33,
    483.70, 490.21, 535.24, 534.91, 534.25, 535.24,
]
INKLING_SQ10_LAYER_MB: list[float] = [
    307.31, 308.95, 305.30, 309.64, 306.98, 312.29,
    374.00, 372.67, 374.31, 372.67, 376.30, 376.27,
    374.00, 374.33, 375.99, 377.60, 381.92, 388.81,
    374.00, 372.01, 370.68, 371.02, 371.68, 371.02,
]


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    def _ranks(x: Sequence[float]) -> np.ndarray:
        s = sorted((float(v), i) for i, v in enumerate(x))
        r = np.zeros(len(x), dtype=np.float64)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1][0] == s[i][0]:
                j += 1
            avg = 0.5 * (i + j)
            for k in range(i, j + 1):
                r[s[k][1]] = avg
            i = j + 1
        return r

    ra = _ranks(a)
    rb = _ranks(b)
    da = ra - ra.mean()
    db = rb - rb.mean()
    den = float(math.sqrt(float((da * da).sum() * (db * db).sum())))
    return round(float((da * db).sum() / den), 4) if den > 1e-12 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp2a", default=str(ROOT / "benchmarks" / "strata_quant" / "results_xray_strata_quant_qwen25_3b_modal_a100.json"))
    ap.add_argument("--exp2b", default=str(ROOT / "benchmarks" / "strata_quant" / "results_xray_strata_quant_olmoe_modal_a100.json"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "strata_quant" / "results_xray_quant_predictor_validation_cpu.json"))
    args = ap.parse_args()

    atlas_dir = ROOT / "benchmarks" / "xray_atlas" / "tome1_xray"
    ink_card = json.loads((atlas_dir / "thinkingmachines__inkling-2-mini.xray.json").read_text(encoding="utf-8"))
    qm_ink = build_quant_sensitivity_map(ink_card)
    ink_scores = qm_ink["per_layer_scores"]

    # Recall of top-10% (or top-3) protected layers on Inkling SQ-10 (where SF4 experts concentrate)
    sq10_top = sorted(range(24), key=lambda i: -INKLING_SQ10_LAYER_MB[i])[:5]
    ink_prot = set(qm_ink["protected_layers"])
    out: dict[str, Any] = {
        "inkling_2_mini": {
            "protected_layers": qm_ink["protected_layers"],
            "spearman_vs_sq10_layer_mb": _spearman(ink_scores, INKLING_SQ10_LAYER_MB),
            "spearman_vs_sq15_layer_mb": _spearman(ink_scores, INKLING_SQ15_LAYER_MB),
            "sq10_loop_plus_tail_mean_mb": round(float(np.mean(INKLING_SQ10_LAYER_MB[6:])), 2),
            "sq10_prologue_mean_mb": round(float(np.mean(INKLING_SQ10_LAYER_MB[:6])), 2),
            "pred_loop_plus_tail_mean_score": round(float(np.mean(ink_scores[6:])), 4),
            "pred_prologue_mean_score": round(float(np.mean(ink_scores[:6])), 4),
            "sq10_top5_layers_by_bytes": sq10_top,
            "protected_hit_in_sq10_top5": len(ink_prot & set(sq10_top)),
        }
    }

    p2a = Path(args.exp2a)
    if p2a.is_file():
        d2a = json.loads(p2a.read_text(encoding="utf-8"))
        qwen_scores = d2a["xray_quant_sensitivity_map"]["per_layer_scores"]
        s2_err = d2a["per_layer_s2_err"]
        s3_err = d2a["per_layer_s3_err"]
        # Weighted bits per layer in knapsack_3_19b
        k319_avg_b = []
        for l_mix in d2a["per_layer_k319_codes"]:
            tot_w = sum(l_mix.values())
            avg = sum((2.1875 if k == "S2_K" else (3.1875 if k == "S3_K" else 4.1875)) * v for k, v in l_mix.items()) / max(1, tot_w)
            k319_avg_b.append(round(avg, 4))
        out["qwen25_3b"] = {
            "spearman_vs_s2_hess_err": _spearman(qwen_scores, s2_err),
            "spearman_vs_s3_hess_err": _spearman(qwen_scores, s3_err),
            "spearman_vs_knapsack_3_19b_avg_bits": _spearman(qwen_scores, k319_avg_b),
            "protected_layers": d2a["xray_quant_sensitivity_map"]["protected_layers"],
            "valley_layers": d2a["xray_quant_sensitivity_map"]["compressible_valley"],
            "valley_mean_avg_bits": round(float(np.mean([k319_avg_b[i] for i in d2a["xray_quant_sensitivity_map"]["compressible_valley"]])), 4)
            if d2a["xray_quant_sensitivity_map"]["compressible_valley"] else None,
            "protected_mean_avg_bits": round(float(np.mean([k319_avg_b[i] for i in d2a["xray_quant_sensitivity_map"]["protected_layers"]])), 4),
        }

    p2b = Path(args.exp2b)
    if p2b.is_file():
        d2b = json.loads(p2b.read_text(encoding="utf-8"))
        ol_scores = d2b["xray_quant_sensitivity_map"]["per_layer_scores"]
        s2_err = d2b["per_layer_s2_err"]
        k285_mb = d2b["per_layer_k285_mb"]
        out["olmoe_1b_7b"] = {
            "spearman_vs_s2_hess_err": _spearman(ol_scores, s2_err),
            "spearman_vs_knapsack_2_85b_mb": _spearman(ol_scores, k285_mb),
            "protected_layers": d2b["xray_quant_sensitivity_map"]["protected_layers"],
            "valley_layers": d2b["xray_quant_sensitivity_map"]["compressible_valley"],
            "valley_mean_mb": round(float(np.mean([k285_mb[i] for i in d2b["xray_quant_sensitivity_map"]["compressible_valley"]])), 2)
            if d2b["xray_quant_sensitivity_map"]["compressible_valley"] else None,
            "protected_mean_mb": round(float(np.mean([k285_mb[i] for i in d2b["xray_quant_sensitivity_map"]["protected_layers"]])), 2),
        }

    _atomic_write_json(args.out, out)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
