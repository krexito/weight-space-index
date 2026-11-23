"""3-Tome WSI Model Atlas (`wsi_atlas_v1`): static X-Ray, Delta X-Ray, and Clinical History lookup & diagnosis.

Combines:
- **Tome 1 (`wsi_xray_v1`)**: 15 static internal-anatomy and linear-pathology charts across 6 architecture families.
- **Tome 2 (`wsi_delta_xray_v1`)**: 11 controlled `(before -> after)` post-training and midtraining diff charts.
- **Tome 3 (`clinical_history`)**: verified GPU intervention records (`pass` / `partial` / `fail`) with failure modes.

Provides zero-leakage consolidation, nearest-neighbor anatomical matching for unseen checkpoints, and compact
Markdown diagnosis generation for autonomous weight-editing agents.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .delta_xray import (
    DELTA_XRAY_SCHEMA_VERSION,
    _extract_flat_delta_row,
    load_delta_xray_json,
    save_delta_xray_json,
    summarize_delta_xray_atlas,
)
from .xray import (
    DEPTH_BINS,
    XRAY_SCHEMA_VERSION,
    _cos_vec,
    _extract_flat_biomarkers,
    _is_finite_num,
    _sanitize,
    load_xray_json,
    save_xray_json,
    summarize_xray_atlas,
)

ATLAS_SCHEMA_VERSION = "wsi_atlas_v1"

RESERVED_HOLDOUT_MODELS: tuple[str, ...] = (
    "microsoft/Phi-4-mini-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-3-4b-it",
)

DEFAULT_ATLAS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "xray_atlas"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def validate_no_holdout_leakage(
    tome1_cards: Iterable[Mapping[str, Any]],
    tome2_cards: Iterable[Mapping[str, Any]],
    tome3_records: Mapping[str, Any],
    *,
    holdout_models: Sequence[str] = RESERVED_HOLDOUT_MODELS,
) -> None:
    """Raise `ValueError` if any reserved hold-out model appears in Tome 1, Tome 2, or Tome 3."""
    holdout_low = {m.strip().lower() for m in holdout_models}
    leaked: list[str] = []

    for c in tome1_cards:
        rid = str((c.get("identity") or {}).get("repo_id") or "").strip()
        if rid.lower() in holdout_low:
            leaked.append(f"tome1:{rid}")

    for c in tome2_cards:
        pair = c.get("pair") or {}
        for k in ("before_repo", "after_repo"):
            rid = str(pair.get(k) or "").strip()
            if rid.lower() in holdout_low:
                leaked.append(f"tome2:{rid}")

    models_map = tome3_records.get("models", tome3_records) if isinstance(tome3_records, Mapping) else {}
    if isinstance(models_map, Mapping):
        for rid in models_map:
            if str(rid).strip().lower() in holdout_low:
                leaked.append(f"tome3:{rid}")

    if leaked:
        raise ValueError(f"Hold-out model leakage detected in 3-Tome Atlas: {sorted(set(leaked))}")


def _build_atlas_handbook_md(
    tome1_summary: Mapping[str, Any],
    tome2_summary: Mapping[str, Any],
    tome3_bundle: Mapping[str, Any],
) -> str:
    n_m = int(tome1_summary.get("n_models", 0))
    n_p = int(tome2_summary.get("n_pairs", 0))
    t3_models = tome3_bundle.get("models", {})
    n_rec = sum(len(v) for v in t3_models.values()) if isinstance(t3_models, Mapping) else 0

    lines: list[str] = [
        "# WSI 3-Tome Internal Model Atlas (`wsi_atlas_v1`)",
        "",
        f"Empirical internal-anatomy reference across **{n_m} static model X-Rays (Tome 1)**, "
        f"**{n_p} controlled checkpoint transitions (Tome 2)**, and **{n_rec} GPU clinical intervention records (Tome 3)**.",
        "",
        "## Quick Query API (Python / CLI)",
        "```python",
        "from wsi.xray import load_xray_json, scan_model_xray",
        "from wsi.atlas import format_atlas_diagnosis, load_atlas, match_xray_neighbors",
        "",
        "# If target_base.xray.json is already in your atlas folder:",
        "card = load_xray_json('atlas/target_base.xray.json')",
        "diag = match_xray_neighbors(card, 'atlas', top_k=4, pathology='over_refusal')",
        "print(format_atlas_diagnosis(diag))",
        "```",
        "Or from bash: `python -m wsi atlas --atlas-dir atlas --diagnose atlas/target_base.xray.json`",
        "",
        "## Cross-Tome Structural Findings",
    ]
    for item in tome3_bundle.get("cross_tome_findings") or []:
        lines.append(f"- **`{item['id']}`** (*{item['source']}*): {item['finding']}")

    lines.extend(
        [
            "",
            f"## Tome 1 — Static Model X-Ray Atlas ({n_m} models, `tome1_xray/*.xray.json`)",
            "",
            str(tome1_summary.get("markdown_table") or "").strip(),
            "",
            f"## Tome 2 — Delta X-Ray Transition Atlas ({n_p} pairs, `tome2_delta/*.delta.xray.json` + Level-2 `*.delta.detail.json`)",
            "",
            str(tome2_summary.get("markdown_table") or "").strip(),
            "",
            f"## Tome 3 — Clinical History ({n_rec} GPU interventions, `tome3_clinical_history.json`)",
            "",
            "| Model | Pathology | Strategy | Target | Outcome | Key Metrics | Caveat |",
            "| :--- | :--- | :--- | :--- | :---: | :--- | :--- |",
        ]
    )
    if isinstance(t3_models, Mapping):
        for model_id, recs in sorted(t3_models.items()):
            for r in recs:
                tgt = json.dumps(r.get("target", {}), ensure_ascii=False, separators=(",", ":"))
                met = json.dumps(r.get("metrics", {}), ensure_ascii=False, separators=(",", ":"))
                lines.append(
                    f"| `{model_id}` | `{r.get('pathology')}` | `{r.get('strategy')}` | `{tgt}` | "
                    f"**`{r.get('outcome')}`** | `{met}` | {r.get('caveat', '')} |"
                )
    lines.append("")
    return "\n".join(lines)


def consolidate_3tome_atlas(
    tome1_bundle_path: str | Path,
    tome2_bundle_paths: Sequence[str | Path],
    out_dir: str | Path = DEFAULT_ATLAS_DIR,
    *,
    clinical_history: Mapping[str, list[dict[str, Any]]] | None = None,
    cross_tome_findings: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Consolidate Tome 1, Tome 2, and Tome 3 GPU bundles into `out_dir` with zero hold-out leakage."""
    root = Path(out_dir)
    t1_dir = root / "tome1_xray"
    t2_dir = root / "tome2_delta"
    t1_dir.mkdir(parents=True, exist_ok=True)
    t2_dir.mkdir(parents=True, exist_ok=True)

    existing_t3_path = root / "tome3_clinical_history.json"
    existing_t3 = (
        json.loads(existing_t3_path.read_text(encoding="utf-8")) if existing_t3_path.exists() else {}
    )
    hist_map = dict(
        clinical_history
        if clinical_history is not None
        else (existing_t3.get("models") if isinstance(existing_t3.get("models"), Mapping) else {})
    )
    findings_list = list(
        cross_tome_findings
        if cross_tome_findings is not None
        else (existing_t3.get("cross_tome_findings") or [])
    )

    t1_raw = json.loads(Path(tome1_bundle_path).read_text(encoding="utf-8"))
    t1_cards_dict = t1_raw.get("cards") or {}
    if not isinstance(t1_cards_dict, Mapping) or not t1_cards_dict:
        raise ValueError(f"No Tome-1 'cards' found in {tome1_bundle_path}")

    t2_merged: dict[str, dict[str, Any]] = {}
    for p in tome2_bundle_paths:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        dc = raw.get("delta_cards") or {}
        if not isinstance(dc, Mapping) or not dc:
            raise ValueError(f"No Tome-2 'delta_cards' found in {p}")
        for fname, card in dc.items():
            if isinstance(card, Mapping) and card.get("schema") == DELTA_XRAY_SCHEMA_VERSION:
                t2_merged[str(fname)] = dict(card)

    t1_cards_list: list[dict[str, Any]] = []
    for fname, card in sorted(t1_cards_dict.items()):
        if not isinstance(card, Mapping) or card.get("schema") != XRAY_SCHEMA_VERSION:
            continue
        c_copy = dict(card)
        rid = str((c_copy.get("identity") or {}).get("repo_id") or "")
        if rid in hist_map:
            c_copy["clinical_history"] = list(hist_map[rid])
        t1_cards_list.append(c_copy)

    t2_cards_list = [t2_merged[k] for k in sorted(t2_merged)]
    t3_bundle = _sanitize(
        {
            "schema": ATLAS_SCHEMA_VERSION,
            "tome": "tome3_clinical_history",
            "n_models": len(hist_map),
            "n_records": sum(len(v) for v in hist_map.values()),
            "models": hist_map,
            "cross_tome_findings": findings_list,
        }
    )

    validate_no_holdout_leakage(t1_cards_list, t2_cards_list, t3_bundle)

    # Clear any stale card files before writing canonical set.
    for old in t1_dir.glob("*.xray.json"):
        old.unlink()
    for old in t2_dir.glob("*.delta.xray.json"):
        old.unlink()

    for fname, card in sorted(t1_cards_dict.items()):
        if not isinstance(card, Mapping) or card.get("schema") != XRAY_SCHEMA_VERSION:
            continue
        c_copy = dict(card)
        rid = str((c_copy.get("identity") or {}).get("repo_id") or "")
        if rid in hist_map:
            c_copy["clinical_history"] = list(hist_map[rid])
        save_xray_json(c_copy, t1_dir / str(fname))

    for fname, card in sorted(t2_merged.items()):
        save_delta_xray_json(card, t2_dir / str(fname))

    t1_summary = summarize_xray_atlas(t1_dir)
    _atomic_write_text(t1_dir / "atlas_summary.json", json.dumps(t1_summary, indent=2, ensure_ascii=False))
    _atomic_write_text(t1_dir / "atlas_table.md", str(t1_summary.get("markdown_table") or "") + "\n")

    t2_summary = summarize_delta_xray_atlas(t2_dir)
    _atomic_write_text(t2_dir / "delta_atlas_summary.json", json.dumps(t2_summary, indent=2, ensure_ascii=False))
    _atomic_write_text(t2_dir / "delta_atlas_table.md", str(t2_summary.get("markdown_table") or "") + "\n")

    _atomic_write_text(root / "tome3_clinical_history.json", json.dumps(t3_bundle, indent=2, ensure_ascii=False))
    handbook_md = _build_atlas_handbook_md(t1_summary, t2_summary, t3_bundle)
    _atomic_write_text(root / "ATLAS_HANDBOOK.md", handbook_md)

    return {
        "schema": ATLAS_SCHEMA_VERSION,
        "atlas_dir": str(root),
        "n_tome1_models": int(t1_summary.get("n_models", 0)),
        "n_tome2_pairs": int(t2_summary.get("n_pairs", 0)),
        "n_tome3_records": int(t3_bundle.get("n_records", 0)),
    }


def load_atlas(atlas_dir: str | Path = DEFAULT_ATLAS_DIR) -> dict[str, Any]:
    """Load all 3 Tomes from `atlas_dir` and verify zero hold-out leakage."""
    root = Path(atlas_dir)
    t1_dir = root / "tome1_xray"
    t2_dir = root / "tome2_delta"
    t3_path = root / "tome3_clinical_history.json"

    t1_files = sorted(t1_dir.glob("*.xray.json")) if t1_dir.is_dir() else []
    t2_files = sorted(t2_dir.glob("*.delta.xray.json")) if t2_dir.is_dir() else []
    t2_detail_files = sorted(t2_dir.glob("*.delta.detail.json")) if t2_dir.is_dir() else []
    if not t1_files:
        raise FileNotFoundError(f"No Tome-1 *.xray.json files found in {t1_dir}")
    if not t2_files:
        raise FileNotFoundError(f"No Tome-2 *.delta.xray.json files found in {t2_dir}")

    t1_cards = [load_xray_json(f) for f in t1_files]
    t2_cards = [load_delta_xray_json(f) for f in t2_files]
    t2_detail_paths = {
        f.name[: -len(".delta.detail.json")]: str(f)
        for f in t2_detail_files
        if f.name.endswith(".delta.detail.json")
    }
    t3_bundle = (
        json.loads(t3_path.read_text(encoding="utf-8"))
        if t3_path.exists()
        else {
            "schema": ATLAS_SCHEMA_VERSION,
            "tome": "tome3_clinical_history",
            "n_models": 0,
            "n_records": 0,
            "models": {},
            "cross_tome_findings": [],
        }
    )

    validate_no_holdout_leakage(t1_cards, t2_cards, t3_bundle)

    t1_sum_p = t1_dir / "atlas_summary.json"
    t1_summary = json.loads(t1_sum_p.read_text(encoding="utf-8")) if t1_sum_p.exists() else summarize_xray_atlas(t1_dir)
    t2_sum_p = t2_dir / "delta_atlas_summary.json"
    t2_summary = (
        json.loads(t2_sum_p.read_text(encoding="utf-8")) if t2_sum_p.exists() else summarize_delta_xray_atlas(t2_dir)
    )

    return {
        "schema": ATLAS_SCHEMA_VERSION,
        "atlas_dir": str(root),
        "tome1_cards": t1_cards,
        "tome1_summary": t1_summary,
        "tome2_cards": t2_cards,
        "tome2_summary": t2_summary,
        "tome2_detail_paths": t2_detail_paths,
        "tome3": t3_bundle,
    }


_SIM_BIOMARKERS: tuple[tuple[str, bool, float], ...] = (
    # (key, use_log1p, weight)
    ("fisher_late_early_ratio", True, 1.2),
    ("unembed_anisotropy_cos", False, 0.8),
    ("mlp_energy90_mean", False, 0.8),
    ("base_overref_rate", False, 1.4),
    ("base_toxic_ref_rate", False, 1.0),
    ("auc_refusal_peak", False, 1.0),
    ("auc_veto_peak", False, 1.2),
    ("veto_framing_shift_to_std", True, 1.3),
    ("cave_rate_false_pushback", False, 0.8),
    ("forced_rep_nll_slope", False, 0.6),
)


def _curve_vec(vals: Any) -> np.ndarray | None:
    if not isinstance(vals, Sequence) or len(vals) == 0:
        return None
    arr = np.array([float(v) if _is_finite_num(v) else np.nan for v in vals], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _analyze_target_depth_profile(target_card: Mapping[str, Any]) -> dict[str, Any]:
    ident = target_card.get("identity") or {}
    n_layers = int(ident.get("n_layers") or 0)
    anat = target_card.get("anatomy") or {}
    fish = anat.get("fisher_profile") or anat.get("fisher") or {}
    patho = target_card.get("pathologies") or {}
    ref = patho.get("over_refusal") or patho.get("refusal") or {}

    f_mlp = _curve_vec(fish.get("mlp_fisher_share_8") or fish.get("mlp_down_share_8"))
    r_auc = _curve_vec(ref.get("auc_refusal_8") or ref.get("auc_8"))
    v_auc = _curve_vec(ref.get("auc_harm_vs_overref_8") or ref.get("veto_auc_8"))
    lowest_fisher_win = fish.get("lowest_fisher_window")

    def _bin_to_layer_span(b_idx: int) -> list[int]:
        if n_layers <= 1:
            return [0, 0]
        lo = int(round((b_idx / 7.0) * (n_layers - 1)))
        hi = int(round(((b_idx + 0.5) / 7.0) * (n_layers - 1)))
        return [max(0, min(n_layers - 1, lo)), max(0, min(n_layers - 1, hi))]

    late_prose_hazard_bins: list[dict[str, Any]] = []
    if f_mlp is not None:
        tot_f = max(float(f_mlp.sum()), 1e-12)
        for bi in (6, 7):
            share = float(f_mlp[bi] / tot_f)
            late_prose_hazard_bins.append(
                {
                    "bin": bi,
                    "depth": DEPTH_BINS[bi],
                    "approx_layer": int(round(DEPTH_BINS[bi] * max(n_layers - 1, 1))),
                    "mlp_fisher_share": round(share, 3),
                }
            )

    candidate_router_bins: list[dict[str, Any]] = []
    if r_auc is not None and v_auc is not None:
        # Rank mid-depth bins (bins 1..6, depth 0.143..0.857) by combined refusal & veto separability
        scored: list[tuple[float, int]] = []
        for bi in range(1, 7):
            comb = 0.5 * float(r_auc[bi]) + 0.5 * float(v_auc[bi])
            scored.append((comb, bi))
        scored.sort(reverse=True)
        for comb, bi in scored[:3]:
            f_sh = float(f_mlp[bi] / max(float(f_mlp.sum()), 1e-12)) if f_mlp is not None else None
            candidate_router_bins.append(
                {
                    "bin": bi,
                    "depth": DEPTH_BINS[bi],
                    "approx_layer": int(round(DEPTH_BINS[bi] * max(n_layers - 1, 1))),
                    "refusal_auc": round(float(r_auc[bi]), 3),
                    "veto_auc": round(float(v_auc[bi]), 3),
                    "combined_auc": round(comb, 3),
                    "mlp_fisher_share": round(f_sh, 3) if f_sh is not None else None,
                    "layer_span": _bin_to_layer_span(bi),
                }
            )

    flat = _extract_flat_biomarkers(dict(target_card))
    v_shift = flat.get("veto_framing_shift_to_std")
    return {
        "repo_id": ident.get("repo_id"),
        "arch_family": ident.get("arch_family"),
        "norm_type": ident.get("norm_type"),
        "n_layers": n_layers,
        "hidden_size": ident.get("hidden_size"),
        "params_b": ident.get("params_b"),
        "flat_biomarkers": flat,
        "lowest_fisher_window": lowest_fisher_win,
        "candidate_mid_depth_bins": candidate_router_bins,
        "late_prose_hazard_bins": late_prose_hazard_bins,
        "needs_framing_invariant_veto": bool(_is_finite_num(v_shift) and float(v_shift) >= 0.25),
    }


def match_xray_neighbors(
    target_card: Mapping[str, Any],
    atlas_dir: str | Path = DEFAULT_ATLAS_DIR,
    *,
    top_k: int = 4,
    pathology: str | None = None,
) -> dict[str, Any]:
    """Compare an unseen model's Tome-1 X-Ray card against all 3 Tomes of the WSI Atlas."""
    atlas = load_atlas(atlas_dir)
    t1_cards = atlas["tome1_cards"]
    t2_cards = atlas["tome2_cards"]
    t3_bundle = atlas["tome3"]

    t_prof = _analyze_target_depth_profile(target_card)
    t_flat = t_prof["flat_biomarkers"]
    t_arch = str(t_prof.get("arch_family") or "")
    t_norm = str(t_prof.get("norm_type") or "")
    t_rid = str(t_prof.get("repo_id") or "")

    # Auto-detect dominant pathology if not explicitly requested.
    active_pathology = pathology
    if not active_pathology:
        if _is_finite_num(t_flat.get("base_overref_rate")) and float(t_flat["base_overref_rate"]) >= 0.15:
            active_pathology = "over_refusal"
        elif _is_finite_num(t_flat.get("sampled_rep16_rate")) and float(t_flat["sampled_rep16_rate"]) >= 0.05:
            active_pathology = "loops"
        elif _is_finite_num(t_flat.get("cave_rate_false_pushback")) and float(t_flat["cave_rate_false_pushback"]) >= 0.40:
            active_pathology = "sycophancy"
        else:
            active_pathology = "over_refusal"

    # Compute mean/std across Tome 1 for standardized biomarker distance.
    ref_flats = [_extract_flat_biomarkers(c) for c in t1_cards]
    stats: dict[str, tuple[float, float]] = {}
    for key, use_log, _w in _SIM_BIOMARKERS:
        vals = []
        for rf in ref_flats:
            v = rf.get(key)
            if _is_finite_num(v):
                fv = math.log1p(max(0.0, float(v))) if use_log else float(v)
                vals.append(fv)
        if len(vals) >= 2:
            stats[key] = (float(np.mean(vals)), max(float(np.std(vals)), 1e-6))
        else:
            stats[key] = (0.0, 1.0)

    t_fish_p = (target_card.get("anatomy") or {}).get("fisher_profile") or (target_card.get("anatomy") or {}).get("fisher") or {}
    t_ref_p = (target_card.get("pathologies") or {}).get("over_refusal") or (target_card.get("pathologies") or {}).get("refusal") or {}
    t_fish_c = _curve_vec(t_fish_p.get("mlp_fisher_share_8") or t_fish_p.get("mlp_down_share_8"))
    t_ref_c = _curve_vec(t_ref_p.get("auc_refusal_8") or t_ref_p.get("auc_8"))
    t_veto_c = _curve_vec(t_ref_p.get("auc_harm_vs_overref_8") or t_ref_p.get("veto_auc_8"))

    neighbors: list[dict[str, Any]] = []
    for c, rf in zip(t1_cards, ref_flats):
        rid = str((c.get("identity") or {}).get("repo_id") or "")
        if rid and rid.lower() == t_rid.lower():
            continue
        w_sum = 0.0
        sq_sum = 0.0
        for key, use_log, w in _SIM_BIOMARKERS:
            vt, vr = t_flat.get(key), rf.get(key)
            if _is_finite_num(vt) and _is_finite_num(vr):
                ft = math.log1p(max(0.0, float(vt))) if use_log else float(vt)
                fr = math.log1p(max(0.0, float(vr))) if use_log else float(vr)
                mu, sd = stats[key]
                z_diff = (ft - fr) / sd
                sq_sum += w * (z_diff * z_diff)
                w_sum += w
        biomarker_rmse = math.sqrt(sq_sum / max(w_sum, 1e-6)) if w_sum > 0 else 9.9
        arch_penalty = 0.0 if str(rf.get("arch_family") or "") == t_arch else 0.45
        norm_penalty = 0.0 if str(rf.get("norm_type") or "") == t_norm else 0.15

        c_fish_p = (c.get("anatomy") or {}).get("fisher_profile") or (c.get("anatomy") or {}).get("fisher") or {}
        c_ref_p = (c.get("pathologies") or {}).get("over_refusal") or (c.get("pathologies") or {}).get("refusal") or {}
        c_fish = _curve_vec(c_fish_p.get("mlp_fisher_share_8") or c_fish_p.get("mlp_down_share_8"))
        c_ref = _curve_vec(c_ref_p.get("auc_refusal_8") or c_ref_p.get("auc_8"))
        c_veto = _curve_vec(c_ref_p.get("auc_harm_vs_overref_8") or c_ref_p.get("veto_auc_8"))
        curve_cos_vals = [
            v
            for v in (
                _cos_vec(t_fish_c, c_fish),
                _cos_vec(t_ref_c, c_ref),
                _cos_vec(t_veto_c, c_veto),
            )
            if v is not None
        ]
        mean_curve_cos = float(np.mean(curve_cos_vals)) if curve_cos_vals else 0.0
        dist = biomarker_rmse + arch_penalty + norm_penalty - 0.25 * mean_curve_cos

        neighbors.append(
            {
                "repo_id": rid,
                "arch_family": rf.get("arch_family"),
                "norm_type": rf.get("norm_type"),
                "params_b": rf.get("params_b"),
                "n_layers": rf.get("n_layers"),
                "distance": round(float(dist), 3),
                "curve_cos": round(float(mean_curve_cos), 3),
                "fisher_late_early_ratio": rf.get("fisher_late_early_ratio"),
                "base_overref_rate": rf.get("base_overref_rate"),
                "base_toxic_ref_rate": rf.get("base_toxic_ref_rate"),
                "auc_refusal_peak": rf.get("auc_refusal_peak"),
                "auc_veto_peak": rf.get("auc_veto_peak"),
                "veto_framing_shift_to_std": rf.get("veto_framing_shift_to_std"),
                "opening_entropy_norm": rf.get("opening_entropy_norm"),
                "cave_rate_false_pushback": rf.get("cave_rate_false_pushback"),
            }
        )

    neighbors.sort(key=lambda x: x["distance"])
    top_neighbors = neighbors[: max(1, int(top_k))]

    # Select relevant Tome-2 delta cards (prioritize post-training transitions & same arch_family).
    posttrain_types = {"instruct", "dpo", "sft", "code_instruct", "math_instruct", "rlvr", "r1_distill"}
    delta_entries: list[tuple[tuple[int, int, float], dict[str, Any]]] = []
    for dc in t2_cards:
        row = _extract_flat_delta_row(dc)
        tt = str(row.get("transition_type") or "")
        af = str(row.get("arch_family") or "")
        same_arch = 1 if af == t_arch else 0
        is_post = 1 if tt in posttrain_types else 0
        prose_attr = ((dc.get("attribution") or {}).get("domains") or {}).get("prose") or {}
        pf_top3 = prose_attr.get("prose_fisher_cost_top3_layers") or []
        role_cred = prose_attr.get("role_credit") or {}
        rec = {
            "pair_id": row.get("pair_id"),
            "transition_type": tt,
            "arch_family": af,
            "total_rel_fro": row.get("total_rel_fro"),
            "depth_centroid": row.get("depth_centroid"),
            "mlp_share": row.get("mlp_share"),
            "mixer_share": row.get("mixer_share"),
            "erank_ratio_mlp": row.get("erank_ratio_mlp"),
            "erank_ratio_mix": row.get("erank_ratio_mix"),
            "dnll_prose": row.get("dnll_prose"),
            "kl_prose": row.get("kl_prose"),
            "faith_prose": row.get("faith_prose"),
            "after_overref_rate": row.get("after_overref_rate"),
            "delta_overref_rate": row.get("delta_overref_rate"),
            "prose_fisher_cost_top3_layers": pf_top3,
            "prose_role_credit": role_cred,
        }
        # Rank: post-training transitions with high faithfulness & matching architecture first
        faith_ok = 1 if (_is_finite_num(row.get("faith_prose")) and 0.6 <= float(row["faith_prose"]) <= 1.5) else 0
        score = (is_post, same_arch + faith_ok, -abs(float(row.get("total_rel_fro") or 1.0)))
        delta_entries.append((score, rec))
    delta_entries.sort(key=lambda item: item[0], reverse=True)
    top_deltas = [rec for _, rec in delta_entries[:6]]

    # Enrich top Tome-2 deltas with compact Level-2 highlights when `.delta.detail.json` is present.
    t2_detail_paths = atlas.get("tome2_detail_paths") or {}
    for rec in top_deltas:
        pid = str(rec.get("pair_id") or "")
        dp = t2_detail_paths.get(pid)
        if not dp or not Path(dp).exists():
            continue
        try:
            det = json.loads(Path(dp).read_text(encoding="utf-8"))
        except Exception:
            continue
        gsb = det.get("global_submatrix_breakdown") or {}
        top_subs = sorted(
            ((k, float(v.get("energy_share") or 0.0)) for k, v in gsb.items() if isinstance(v, Mapping)),
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]
        ah = ((det.get("intra_layer_outliers") or {}).get("attention_heads") or {})
        prose_params = ((det.get("attribution_detail") or {}).get("top_credited_params") or {}).get("prose") or []
        top_p = prose_params[0] if prose_params else None
        osig = ((det.get("tome1_curve_deltas") or {}).get("opening_signatures") or {})
        rec["level2"] = {
            "detail_file": f"tome2_delta/{pid}.delta.detail.json",
            "top_submatrices": [{"name": k, "energy_share": round(s, 3)} for k, s in top_subs],
            "mean_head_gini": ah.get("mean_head_gini"),
            "top_prose_param": (
                {"param": top_p.get("param"), "credit": top_p.get("credit")}
                if isinstance(top_p, Mapping)
                else None
            ),
            "opening_entropy_before": osig.get("before_opening_entropy_norm"),
            "opening_entropy_after": osig.get("after_opening_entropy_norm"),
        }

    # Collect Tome-3 clinical history records (matching pathology first, then others).
    t3_models = t3_bundle.get("models", {}) if isinstance(t3_bundle, Mapping) else {}
    matched_history: list[dict[str, Any]] = []
    other_history: list[dict[str, Any]] = []
    if isinstance(t3_models, Mapping):
        for mid, recs in sorted(t3_models.items()):
            for r in recs:
                entry = {"model": mid, **r}
                if str(r.get("pathology")) == active_pathology:
                    matched_history.append(entry)
                else:
                    other_history.append(entry)

    return _sanitize(
        {
            "schema": ATLAS_SCHEMA_VERSION,
            "active_pathology": active_pathology,
            "target_profile": t_prof,
            "nearest_tome1_models": top_neighbors,
            "relevant_tome2_deltas": top_deltas,
            "relevant_tome3_history": matched_history + other_history[:3],
            "cross_tome_findings": list(t3_bundle.get("cross_tome_findings") or []),
        }
    )


def format_atlas_diagnosis(diagnosis: Mapping[str, Any]) -> str:
    """Format a 3-Tome comparative diagnosis into a compact Markdown report (~2-3 KB)."""
    tp = diagnosis.get("target_profile") or {}
    tf = tp.get("flat_biomarkers") or {}
    patho = diagnosis.get("active_pathology") or "over_refusal"

    def _f(v: Any, d: int = 3) -> str:
        return f"{float(v):.{d}f}" if _is_finite_num(v) else "—"

    def _p(v: Any) -> str:
        return f"{float(v) * 100:.1f}%" if _is_finite_num(v) else "—"

    lines: list[str] = [
        f"# 3-Tome Anatomical Diagnosis: `{tp.get('repo_id')}` (focus: `{patho}`)",
        "",
        "## 1. Target Model Static X-Ray Summary",
        f"- **Architecture**: `{tp.get('arch_family')}` | `{tp.get('norm_type')}` | "
        f"`{tp.get('n_layers')}` layers | hidden `{tp.get('hidden_size')}` | `{_f(tp.get('params_b'), 2)}B` params",
        f"- **Sensitivity & Geometry**: `fisher_late_early_ratio = {_f(tf.get('fisher_late_early_ratio'))}` | "
        f"`unembed_anisotropy_cos = {_f(tf.get('unembed_anisotropy_cos'))}` | "
        f"`mlp_energy90_mean = {_f(tf.get('mlp_energy90_mean'))}`",
        f"- **Refusal & Safety Probes**: `base_overref_rate = {_p(tf.get('base_overref_rate'))}` | "
        f"`base_toxic_ref_rate = {_p(tf.get('base_toxic_ref_rate'))}` | "
        f"`auc_refusal_peak = {_f(tf.get('auc_refusal_peak'))}` | "
        f"`auc_veto_peak = {_f(tf.get('auc_veto_peak'))}` | "
        f"`veto_framing_shift_to_std = {_f(tf.get('veto_framing_shift_to_std'))}` "
        f"({'HIGH framing shift -> use framing-invariant veto router' if tp.get('needs_framing_invariant_veto') else 'low framing shift'})",
        f"- **Opening Style & Sycophancy**: `opening_entropy_norm = {_f(tf.get('opening_entropy_norm'))}` | "
        f"`ctrl_opening_entropy_norm = {_f(tf.get('ctrl_opening_entropy_norm'))}` | "
        f"`cave_rate_false_pushback = {_p(tf.get('cave_rate_false_pushback'))}`",
    ]

    mid_bins = tp.get("candidate_mid_depth_bins") or []
    if mid_bins:
        lines.append("- **Top Mid-Depth Separability Bins (`refusal.auc_8` + `veto_auc_8`)**:")
        for b in mid_bins:
            lines.append(
                f"  - Bin `{b['bin']}` (`depth={b['depth']}`, approx layer `L{b['approx_layer']}`, "
                f"span `L{b['layer_span'][0]}..L{b['layer_span'][1]}`): "
                f"`refusal_auc={b['refusal_auc']}`, `veto_auc={b['veto_auc']}`, "
                f"`mlp_fisher_share={b.get('mlp_fisher_share')}`"
            )

    haz_bins = tp.get("late_prose_hazard_bins") or []
    if haz_bins:
        haz_str = ", ".join(
            f"bin {h['bin']} (depth {h['depth']}, ~L{h['approx_layer']}: Fisher share {h['mlp_fisher_share']})"
            for h in haz_bins
        )
        lines.append(f"- **Late-Layer Prose Sensitivity Profile**: {haz_str}")

    lines.extend(
        [
            "",
            "## 2. Nearest Tome-1 Static Neighbors",
            "| Neighbor | Arch | Layers | Dist | CurveCos | Fisher L/E | OverRef | Toxic | Veto AUC | FrameShift |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for nb in diagnosis.get("nearest_tome1_models") or []:
        lines.append(
            f"| `{nb['repo_id']}` | `{nb['arch_family']}` | {nb['n_layers']} | {_f(nb['distance'])} | "
            f"{_f(nb['curve_cos'])} | {_f(nb['fisher_late_early_ratio'])} | {_p(nb['base_overref_rate'])} | "
            f"{_p(nb['base_toxic_ref_rate'])} | {_f(nb['auc_veto_peak'])} | {_f(nb['veto_framing_shift_to_std'])} |"
        )

    lines.extend(
        [
            "",
            "## 3. Relevant Tome-2 Controlled Post-Training Deltas",
            "| Pair | Type | RelFro | Depth | MLP / Mix | RankRatio (MLP/Mix) | ΔOverRef | KL Prose | Top Prose-Fisher Hazard Layers |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for d in diagnosis.get("relevant_tome2_deltas") or []:
        pf = d.get("prose_fisher_cost_top3_layers") or []
        pf_s = ", ".join(f"L{x['layer']}(d={x['depth']}:{float(x['share'])*100:.0f}%)" for x in pf[:2]) or "—"
        lines.append(
            f"| `{d['pair_id']}` | `{d['transition_type']}` | {_f(d['total_rel_fro'], 4)} | "
            f"{_f(d['depth_centroid'], 2)} | {_p(d['mlp_share'])} / {_p(d['mixer_share'])} | "
            f"{_f(d['erank_ratio_mlp'], 2)} / {_f(d['erank_ratio_mix'], 2)} | "
            f"{_p(d['delta_overref_rate'])} | {_f(d['kl_prose'])} | {pf_s} |"
        )

    l2_rows = [d for d in (diagnosis.get("relevant_tome2_deltas") or []) if isinstance(d.get("level2"), Mapping)]
    if l2_rows:
        lines.extend(
            [
                "",
                "### Level-2 Submatrix & Parameter Highlights (`tome2_delta/*.delta.detail.json`)",
                "| Pair | Top Submatrices (Energy) | Head Gini | Top Prose-Credit Param | Opening Entropy (Before→After) |",
                "| :--- | :--- | :---: | :--- | :---: |",
            ]
        )
        for d in l2_rows:
            l2 = d["level2"]
            subs = ", ".join(f"`{s['name']}` {float(s['energy_share'])*100:.0f}%" for s in (l2.get("top_submatrices") or [])[:3]) or "—"
            tp_p = l2.get("top_prose_param")
            tp_s = f"`{tp_p['param']}` ({_f(tp_p.get('credit'), 3)})" if isinstance(tp_p, Mapping) and tp_p.get("param") else "—"
            eb, ea = l2.get("opening_entropy_before"), l2.get("opening_entropy_after")
            ent_s = f"{_f(eb, 2)} → {_f(ea, 2)}" if (_is_finite_num(eb) or _is_finite_num(ea)) else "—"
            lines.append(
                f"| `{d['pair_id']}` | {subs} | {_f(l2.get('mean_head_gini'), 3)} | {tp_s} | {ent_s} |"
            )

    lines.extend(
        [
            "",
            f"## 4. Tome-3 Clinical History & Failure Modes (`{patho}` priority)",
        ]
    )
    for r in diagnosis.get("relevant_tome3_history") or []:
        met = json.dumps(r.get("metrics", {}), ensure_ascii=False, separators=(",", ":"))
        tgt = json.dumps(r.get("target", {}), ensure_ascii=False, separators=(",", ":"))
        lines.append(
            f"- **[`{str(r.get('outcome')).upper()}`] `{r.get('model')}` — `{r.get('strategy')}`** "
            f"(pathology=`{r.get('pathology')}`, target=`{tgt}`, metrics=`{met}`): {r.get('caveat')}"
        )

    lines.extend(["", "## 5. Cross-Tome Anatomical Rules"])
    for item in diagnosis.get("cross_tome_findings") or []:
        lines.append(f"- **`{item['id']}`**: {item['finding']}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ATLAS_SCHEMA_VERSION",
    "DEFAULT_ATLAS_DIR",
    "RESERVED_HOLDOUT_MODELS",
    "consolidate_3tome_atlas",
    "format_atlas_diagnosis",
    "load_atlas",
    "match_xray_neighbors",
    "validate_no_holdout_leakage",
]
