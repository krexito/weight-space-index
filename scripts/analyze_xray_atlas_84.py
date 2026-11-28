"""Comprehensive CPU-only ($0) cross-family analysis of the 84-model WSI Atlas (80 Tome-1 + 4 Non-Text + 26 Tome-2)."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T1_DIR = ROOT / "benchmarks" / "xray_atlas" / "tome1_xray"
T2_DIR = ROOT / "benchmarks" / "xray_atlas" / "tome2_delta"
NONTEXT_PATH = ROOT / "benchmarks" / "xray_atlas" / "universal_anatomy_nontext.json"
OUT_JSON = ROOT / "benchmarks" / "xray_atlas" / "atlas_84_cross_family_report.json"


def _fmean(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return round(float(np.mean(xs)), 4) if xs else None


def _fmed(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return round(float(np.median(xs)), 4) if xs else None


def _safe_get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def main() -> None:
    t1_files = sorted(T1_DIR.glob("*.xray.json"))
    t2_files = sorted(T2_DIR.glob("*.delta.xray.json"))
    t1 = [json.loads(p.read_text(encoding="utf-8")) for p in t1_files]
    t2 = [json.loads(p.read_text(encoding="utf-8")) for p in t2_files]
    nontext = json.loads(NONTEXT_PATH.read_text(encoding="utf-8")) if NONTEXT_PATH.exists() else []

    # 1. Group Tome-1 by norm_type (PreNorm vs PostNorm) and arch_family
    by_norm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flat_rows: list[dict[str, Any]] = []

    for c in t1:
        ident = c.get("identity", {})
        anat = c.get("anatomy", {})
        sw = anat.get("static_weights", {})
        fp = anat.get("forward_profile", {})
        fi = anat.get("fisher_profile", {})
        fr = anat.get("fact_readout", {})
        path = c.get("pathologies", {})
        oref = path.get("over_refusal", {})
        syc = path.get("sycophancy", {})
        lp = path.get("loops", {})

        norm_t = str(ident.get("norm_type") or "unknown")
        arch_f = str(ident.get("arch_family") or "unknown")
        row = {
            "repo_id": ident.get("repo_id"),
            "params_b": ident.get("params_b"),
            "arch_family": arch_f,
            "norm_type": norm_t,
            "n_layers": ident.get("n_layers"),
            "massive_ratio": _safe_get(fp, "massive_act_max", "ratio"),
            "massive_depth": _safe_get(fp, "massive_act_max", "depth"),
            "mlp_erank": _fmean(sw.get("mlp_down_erank_ratio_8") or []),
            "mlp_col_gini": _fmean(sw.get("mlp_down_col_gini_8") or []),
            "mlp_zero": _fmean(fp.get("mlp_zero_frac_8") or []),
            "resid_cos": _fmean((fp.get("resid_step_cos_8") or [])[1:-1]),
            "block_fisher_le": fi.get("block_fisher_late_early_ratio"),
            "mlp_fisher_le": fi.get("fisher_late_early_ratio"),
            "fact_acc": fr.get("base_fact_top1_acc"),
            "dla_q1": fr.get("share_first_quarter"),
            "overref": oref.get("base_overref_rate"),
            "toxic": oref.get("base_toxic_ref_rate"),
            "framed_toxic": oref.get("base_toxic_framed_ref_rate"),
            "syco_cave": syc.get("cave_rate_false_pushback"),
            "rep_ent_ratio": lp.get("forced_rep_entropy_ratio_k8_k1"),
            "has_q_sens": bool(fi.get("q_sens_8")),
            "has_surgery_readiness": bool(anat.get("surgery_readiness")),
            "probe_sha": c.get("probe_pack_sha256"),
        }
        flat_rows.append(row)
        by_norm[norm_t].append(row)
        by_arch[arch_f].append(row)

    # 2. Group Tome-2 by transition_type
    by_ttype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in t2:
        tt = str(_safe_get(d, "pair", "transition_type") or "unknown")
        wd = d.get("weight_diff", {})
        sd = d.get("spectral_diff", {}).get("mlp_out", {})
        by_ttype[tt].append({
            "pair_id": _safe_get(d, "pair", "pair_id"),
            "rel_fro": wd.get("total_rel_fro"),
            "o_proj_share": _safe_get(wd, "role_energy_share", "attn_out"),
            "mlp_erank": sd.get("erank_ratio_mean"),
            "r1": sd.get("energy_r1_mean"),
            "col_gini": sd.get("col_gini_mean"),
            "changed_frac": (
                round(wd["n_changed_params"] / max(wd["n_matched_params"], 1), 4)
                if wd.get("n_changed_params") is not None and wd.get("n_matched_params")
                else None
            ),
        })

    # 3. Compute key cross-family summaries
    norm_summary = {}
    for k, rows in sorted(by_norm.items()):
        norm_summary[k] = {
            "n": len(rows),
            "massive_ratio_median": _fmed([r["massive_ratio"] for r in rows]),
            "massive_ratio_min_max": [
                round(min(r["massive_ratio"] for r in rows if r["massive_ratio"] is not None), 1)
                if any(r["massive_ratio"] is not None for r in rows) else None,
                round(max(r["massive_ratio"] for r in rows if r["massive_ratio"] is not None), 1)
                if any(r["massive_ratio"] is not None for r in rows) else None,
            ],
            "mlp_erank_median": _fmed([r["mlp_erank"] for r in rows]),
            "block_fisher_le_median": _fmed([r["block_fisher_le"] for r in rows]),
        }

    t2_summary = {}
    for tt, rows in sorted(by_ttype.items()):
        t2_summary[tt] = {
            "n": len(rows),
            "rel_fro_median": _fmed([r["rel_fro"] for r in rows]),
            "o_proj_share_median": _fmed([r["o_proj_share"] for r in rows]),
            "mlp_erank_median": _fmed([r["mlp_erank"] for r in rows]),
            "r1_median": _fmed([r["r1"] for r in rows]),
            "col_gini_median": _fmed([r["col_gini"] for r in rows]),
            "changed_frac_median": _fmed([r["changed_frac"] for r in rows]),
        }

    # 4. Framing fragility: models with toxic >= 0.70 that drop >= 25 pts under framing
    framing_fragile = [
        {
            "repo_id": r["repo_id"],
            "toxic": r["toxic"],
            "framed_toxic": r["framed_toxic"],
            "drop": round(r["toxic"] - r["framed_toxic"], 3),
        }
        for r in flat_rows
        if r["toxic"] is not None and r["framed_toxic"] is not None and r["toxic"] >= 0.70 and (r["toxic"] - r["framed_toxic"]) >= 0.25
    ]
    framing_fragile.sort(key=lambda x: -x["drop"])

    # 5. Low-sycophancy champions (cave <= 0.20) vs high-sycophancy (>= 0.85)
    low_syco = sorted(
        [{"repo_id": r["repo_id"], "syco_cave": r["syco_cave"], "fact_acc": r["fact_acc"]} for r in flat_rows if r["syco_cave"] is not None and r["syco_cave"] <= 0.20],
        key=lambda x: x["syco_cave"],
    )

    # 6. Schema coverage audit across all 80 Tome-1 models
    sha_counts: dict[str, int] = defaultdict(int)
    for r in flat_rows:
        sha_counts[str(r["probe_sha"])] += 1

    report = {
        "counts": {
            "tome1_models": len(t1),
            "nontext_models": len(nontext),
            "total_models": len(t1) + len(nontext),
            "tome2_pairs": len(t2),
        },
        "schema_audit": {
            "probe_pack_sha_counts": dict(sha_counts),
            "n_with_q_sens": sum(1 for r in flat_rows if r["has_q_sens"]),
            "n_with_surgery_readiness": sum(1 for r in flat_rows if r["has_surgery_readiness"]),
        },
        "universal_laws": {
            "pct_tome1_massive_gt_50x": round(100.0 * sum(1 for r in flat_rows if (r["massive_ratio"] or 0) > 50) / len(flat_rows), 1),
            "pct_nontext_massive_gt_50x": round(100.0 * sum(1 for r in nontext if (_safe_get(r, "massive_act_max", "ratio") or 0) > 50) / max(len(nontext), 1), 1),
            "tome1_mlp_erank_median": _fmed([r["mlp_erank"] for r in flat_rows]),
            "tome1_col_gini_median": _fmed([r["mlp_col_gini"] for r in flat_rows]),
            "tome1_dla_q1_median": _fmed([r["dla_q1"] for r in flat_rows]),
            "tome1_mid_resid_cos_median": _fmed([r["resid_cos"] for r in flat_rows]),
            "tome1_syco_cave_median": _fmed([r["syco_cave"] for r in flat_rows]),
        },
        "by_norm_type": norm_summary,
        "by_transition_type": t2_summary,
        "framing_fragile_models": framing_fragile[:12],
        "low_sycophancy_champions": low_syco[:15],
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
