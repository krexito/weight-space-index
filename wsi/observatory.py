"""WSI Observatory: structured query, empirical law aggregation, and 1-call agent prescription over the 3-Tome Atlas.

Turns a growing library of Tome-1 X-Rays (41 -> 500+ models), Tome-2 Delta diffs, and Tome-3
clinical intervention records into:
  1. Filterable model & transition queries (`query_models`, `query_transitions`)
  2. Live empirical scorecards of what works vs. what fails (`summarize_evidence`)
  3. Compact 1-call prescriptions (`prescribe`) so an AI agent gets nearest neighbors,
     proven strategies, a concrete failure blacklist, and the exact next 2-second probe
     without rummaging through dozens of raw JSON files.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_ATLAS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "xray_atlas"

# Numeric features used for geometric nearest-neighbor matching across Tome-1 models
_ANATOMY_FEATURES: tuple[str, ...] = (
    "params_b",
    "n_layers",
    "cal_nll",
    "unembed_anisotropy_cos",
    "massive_act_max",
    "fisher_late_early_ratio",
    "block_fisher_late_early_ratio",
    "mlp_energy90_mean",
    "base_overref_rate",
    "base_toxic_ref_rate",
    "cave_rate_false_pushback",
)


def _safe_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _detect_quant_tag(repo_id: str) -> str:
    low = repo_id.lower()
    for tag in ("iq1_s", "iq2_s", "iq4_xs", "q2_k", "q4_k_m", "q4_0", "q8_0", "1-bit", "1b", "1.58b", "ternary", "nvfp4", "mxfp8", "fp8"):
        if tag in low:
            return tag.upper()
    return "BF16/FP16"


class WSIObservatory:
    """Structured in-memory index over Tome 1 (models), Tome 2 (deltas), and Tome 3 (interventions)."""

    def __init__(self, atlas_dir: str | Path = DEFAULT_ATLAS_DIR) -> None:
        self.atlas_dir = Path(atlas_dir)
        t1_p = self.atlas_dir / "tome1_xray" / "atlas_summary.json"
        t2_p = self.atlas_dir / "tome2_delta" / "delta_atlas_summary.json"
        t3_p = self.atlas_dir / "tome3_clinical_history.json"

        self.models: list[dict[str, Any]] = []
        if t1_p.exists():
            raw1 = json.loads(t1_p.read_text(encoding="utf-8"))
            for m in raw1.get("models", []):
                row = dict(m)
                row.setdefault("quant_tag", _detect_quant_tag(str(row.get("repo_id", ""))))
                self.models.append(row)

        self.pairs: list[dict[str, Any]] = []
        if t2_p.exists():
            raw2 = json.loads(t2_p.read_text(encoding="utf-8"))
            self.pairs = [dict(p) for p in raw2.get("pairs", [])]

        self.interventions: list[dict[str, Any]] = []
        if t3_p.exists():
            raw3 = json.loads(t3_p.read_text(encoding="utf-8"))
            for repo_id, recs in (raw3.get("models") or {}).items():
                for rec in recs:
                    self.interventions.append({"repo_id": repo_id, **rec})

        self._by_repo = {str(m.get("repo_id", "")): m for m in self.models}
        self._feat_stats = self._compute_feature_stats()

    def _compute_feature_stats(self) -> dict[str, tuple[float, float]]:
        stats: dict[str, tuple[float, float]] = {}
        for f in _ANATOMY_FEATURES:
            vals = []
            for m in self.models:
                v = _safe_float(m.get(f))
                if v is not None:
                    vals.append(math.log1p(max(v, 0.0)) if f in ("params_b", "massive_act_max", "fisher_late_early_ratio") else v)
            if vals:
                mu = sum(vals) / len(vals)
                var = sum((x - mu) ** 2 for x in vals) / max(len(vals), 1)
                stats[f] = (mu, max(math.sqrt(var), 1e-6))
            else:
                stats[f] = (0.0, 1.0)
        return stats

    def _zvec(self, row: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for f in _ANATOMY_FEATURES:
            v = _safe_float(row.get(f))
            if v is None:
                continue
            val = math.log1p(max(v, 0.0)) if f in ("params_b", "massive_act_max", "fisher_late_early_ratio") else v
            mu, sd = self._feat_stats[f]
            out[f] = (val - mu) / sd
        return out

    # ------------------------------------------------------------------ queries
    def query_models(
        self,
        *,
        arch_family: str | None = None,
        norm_type: str | None = None,
        quant_tag: str | None = None,
        min_params_b: float | None = None,
        max_params_b: float | None = None,
        min_overref: float | None = None,
        min_sycophancy: float | None = None,
        search: str | None = None,
        sort_by: str = "params_b",
        descending: bool = True,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        out = []
        q = (search or "").strip().lower()
        for m in self.models:
            if arch_family and str(m.get("arch_family", "")).lower() != arch_family.lower():
                continue
            if norm_type and str(m.get("norm_type", "")).lower() != norm_type.lower():
                continue
            if quant_tag and str(m.get("quant_tag", "")).lower() != quant_tag.lower():
                continue
            pb = _safe_float(m.get("params_b"))
            if min_params_b is not None and (pb is None or pb < min_params_b):
                continue
            if max_params_b is not None and (pb is None or pb > max_params_b):
                continue
            ovr = _safe_float(m.get("base_overref_rate"))
            if min_overref is not None and (ovr is None or ovr < min_overref):
                continue
            syc = _safe_float(m.get("cave_rate_false_pushback"))
            if min_sycophancy is not None and (syc is None or syc < min_sycophancy):
                continue
            if q and q not in str(m.get("repo_id", "")).lower() and q not in str(m.get("arch_family", "")).lower():
                continue
            out.append(m)

        def _skey(row: dict[str, Any]) -> tuple[int, float]:
            v = _safe_float(row.get(sort_by))
            if v is None:
                return (1, 0.0)
            return (0, -v if descending else v)

        out.sort(key=_skey)
        return out[:limit]

    def nearest_neighbors(self, target: dict[str, Any], top_k: int = 4) -> list[dict[str, Any]]:
        tz = self._zvec(target)
        t_arch = str(target.get("arch_family", "")).lower()
        t_norm = str(target.get("norm_type", "")).lower()
        t_repo = str(target.get("repo_id", ""))

        scored = []
        for m in self.models:
            if t_repo and str(m.get("repo_id", "")) == t_repo:
                continue
            mz = self._zvec(m)
            shared = set(tz) & set(mz)
            if not shared:
                continue
            dot = sum(tz[k] * mz[k] for k in shared)
            na = math.sqrt(sum(tz[k] ** 2 for k in shared))
            nb = math.sqrt(sum(mz[k] ** 2 for k in shared))
            cos = dot / max(na * nb, 1e-8)
            bonus = 0.0
            if t_arch and str(m.get("arch_family", "")).lower() == t_arch:
                bonus += 0.15
            if t_norm and str(m.get("norm_type", "")).lower() == t_norm:
                bonus += 0.08
            scored.append({
                "repo_id": m.get("repo_id"),
                "arch_family": m.get("arch_family"),
                "norm_type": m.get("norm_type"),
                "params_b": m.get("params_b"),
                "n_layers": m.get("n_layers"),
                "similarity": round(cos + bonus, 4),
                "raw_cosine": round(cos, 4),
                "shared_features": len(shared),
                "has_clinical_history": any(r["repo_id"] == m.get("repo_id") for r in self.interventions),
            })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def summarize_evidence(self) -> dict[str, Any]:
        """Aggregate live empirical tables across all 3 Tomes (what works vs. what fails)."""
        # 1. Architecture summary from Tome 1
        by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in self.models:
            by_arch[str(m.get("arch_family") or "unknown")].append(m)

        arch_stats = []
        for arch, rows in sorted(by_arch.items(), key=lambda kv: -len(kv[1])):
            def _med(field: str) -> float | None:
                vs = sorted(v for r in rows if (v := _safe_float(r.get(field))) is not None)
                return round(vs[len(vs) // 2], 4) if vs else None

            arch_stats.append({
                "arch_family": arch,
                "n_models": len(rows),
                "median_massive_act": _med("massive_act_max"),
                "median_fisher_late_early": _med("fisher_late_early_ratio"),
                "median_overref": _med("base_overref_rate"),
                "median_toxic_ref": _med("base_toxic_ref_rate"),
                "median_syc_cave": _med("cave_rate_false_pushback"),
            })

        # 2. Transition summary from Tome 2
        by_trans: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in self.pairs:
            by_trans[str(p.get("transition_type") or "other")].append(p)

        trans_stats = []
        for tt, rows in sorted(by_trans.items(), key=lambda kv: -len(kv[1])):
            def _avg(field: str) -> float | None:
                vs = [v for r in rows if (v := _safe_float(r.get(field))) is not None]
                return round(sum(vs) / len(vs), 4) if vs else None

            trans_stats.append({
                "transition_type": tt,
                "n_pairs": len(rows),
                "mean_mlp_share": _avg("mlp_share"),
                "mean_mixer_share": _avg("mixer_share"),
                "mean_erank_mlp": _avg("erank_ratio_mlp"),
                "mean_depth_centroid": _avg("depth_centroid"),
                "mean_kl_prose": _avg("kl_prose"),
            })

        # 3. Clinical outcome scorecard from Tome 3 (grouped by pathology + strategy)
        by_strat: dict[tuple[str, str], dict[str, Any]] = {}
        for rec in self.interventions:
            key = (str(rec.get("pathology", "unknown")), str(rec.get("strategy", "unknown")))
            st = by_strat.setdefault(key, {
                "pathology": key[0],
                "strategy": key[1],
                "pass": 0,
                "partial": 0,
                "fail": 0,
                "models": [],
                "caveats": [],
            })
            oc = str(rec.get("outcome", "fail")).lower()
            if oc in ("pass", "partial", "fail"):
                st[oc] += 1
            st["models"].append(rec.get("repo_id"))
            if rec.get("caveat"):
                st["caveats"].append(rec["caveat"])

        scorecard = []
        for st in by_strat.values():
            tot = st["pass"] + st["partial"] + st["fail"]
            st["total"] = tot
            st["pass_rate"] = round(st["pass"] / max(tot, 1), 3)
            scorecard.append(st)
        scorecard.sort(key=lambda r: (r["pathology"], -r["pass_rate"], -r["total"]))

        return {
            "counts": {
                "tome1_models": len(self.models),
                "tome2_pairs": len(self.pairs),
                "tome3_interventions": len(self.interventions),
                "tome3_pass": sum(1 for r in self.interventions if r.get("outcome") == "pass"),
                "tome3_partial": sum(1 for r in self.interventions if r.get("outcome") == "partial"),
                "tome3_fail": sum(1 for r in self.interventions if r.get("outcome") == "fail"),
            },
            "by_architecture": arch_stats,
            "by_transition": trans_stats,
            "clinical_scorecard": scorecard,
        }

    def _load_target_row(self, target: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(target, dict):
            return target
        s = str(target)
        if s in self._by_repo:
            return self._by_repo[s]
        p = Path(s)
        if p.exists() and p.is_file():
            raw = json.loads(p.read_text(encoding="utf-8"))
            # Flatten full .xray.json card to summary-like dict if needed
            if "identity" in raw:
                ident = raw.get("identity") or {}
                anat = raw.get("anatomy") or {}
                patho = raw.get("pathologies") or {}
                ref = patho.get("refusal") or {}
                syc = patho.get("sycophancy") or {}
                return {
                    "repo_id": ident.get("repo_id", p.stem),
                    "params_b": ident.get("params_b"),
                    "arch_family": ident.get("arch_family"),
                    "norm_type": ident.get("norm_type"),
                    "n_layers": ident.get("n_layers"),
                    "cal_nll": anat.get("cal_nll"),
                    "massive_act_max": anat.get("massive_act_max"),
                    "fisher_late_early_ratio": anat.get("fisher_late_early_ratio"),
                    "block_fisher_late_early_ratio": anat.get("block_fisher_late_early_ratio"),
                    "base_overref_rate": ref.get("base_overref_rate"),
                    "base_toxic_ref_rate": ref.get("base_toxic_ref_rate"),
                    "cave_rate_false_pushback": syc.get("cave_rate_false_pushback"),
                }
            return raw
        raise KeyError(f"Target model `{s}` not found in Tome-1 Atlas or on disk")

    def prescribe(
        self,
        target: str | dict[str, Any],
        pathology: str = "over_refusal",
        top_k_neighbors: int = 3,
    ) -> dict[str, Any]:
        """Return a compact, actionable 1-call prescription for an agent or researcher."""
        row = self._load_target_row(target)
        n_layers = int(_safe_float(row.get("n_layers")) or 32)
        neighbors = self.nearest_neighbors(row, top_k=top_k_neighbors)

        # Filter Tome-3 records matching the pathology (exact or substring)
        p_low = pathology.lower().strip()
        rel = [r for r in self.interventions if p_low in str(r.get("pathology", "")).lower()]
        if not rel:
            rel = list(self.interventions)

        works = [
            {
                "strategy": r["strategy"],
                "model": r["repo_id"],
                "target": r.get("target"),
                "metrics": r.get("metrics"),
                "caveat": r.get("caveat"),
            }
            for r in rel
            if r.get("outcome") == "pass"
        ]
        partials = [
            {
                "strategy": r["strategy"],
                "model": r["repo_id"],
                "metrics": r.get("metrics"),
                "caveat": r.get("caveat"),
            }
            for r in rel
            if r.get("outcome") == "partial"
        ]
        do_not_do = [
            {
                "avoid_strategy": r["strategy"],
                "model": r["repo_id"],
                "why_failed": r.get("caveat") or str(r.get("metrics")),
            }
            for r in rel
            if r.get("outcome") == "fail"
        ]

        # Always include universal guardrails backed by Tome-3 / surgery telemetry
        mid_lo = max(4, round(0.35 * n_layers))
        mid_hi = min(n_layers - 3, round(0.65 * n_layers))
        universal_blacklist = [
            "Do NOT edit L0-L3 prologue layers on dense_gqa / pre_rmsnorm models (high Hydra self-repair & prose/safety collapse).",
            "Do NOT hand-code router thresholds without checking MoIPatch's `score >= thr` sign convention; prefer `wsi_calibrate_safety_router`.",
            "Do NOT select routed patches by always-on KL penalty (use `effective_gain` + short escape test instead).",
        ]

        next_probe = {
            "step_1_telemetry": f"Call `wsi_diagnose_surgery_sites` across candidate mid-layers L{mid_lo}..L{mid_hi} (rank by `effective_gain = causal_gain * max(hydra_retention, 0)`).",
            "step_2_router": (
                "Call `wsi_calibrate_safety_router` (with framing-invariant veto + token-0 refusal prior) to protect toxic refusals and controls."
                if "refusal" in p_low
                else "Attach a mid-layer Ridge/linear router (verified 0% off-domain fire on controls) before evaluating."
            ),
            "step_3_verify": "Run a short 64-token escape test + style check (`new_template_rate <= 20%`) on the top-3 telemetry layers before finalizing `.moipatch`.",
        }

        return {
            "target": {
                "repo_id": row.get("repo_id"),
                "arch_family": row.get("arch_family"),
                "norm_type": row.get("norm_type"),
                "params_b": row.get("params_b"),
                "n_layers": n_layers,
                "quant_tag": row.get("quant_tag") or _detect_quant_tag(str(row.get("repo_id", ""))),
            },
            "pathology": pathology,
            "atlas_backing": {
                "tome1_models": len(self.models),
                "tome2_pairs": len(self.pairs),
                "tome3_records_for_pathology": len(rel),
            },
            "nearest_neighbors": neighbors,
            "proven_strategies_pass": works,
            "partial_strategies": partials,
            "do_not_do_blacklist": do_not_do,
            "universal_guardrails": universal_blacklist,
            "recommended_candidate_band": f"L{mid_lo}..L{mid_hi} (depth 0.35-0.65)",
            "next_2s_probe": next_probe,
        }
