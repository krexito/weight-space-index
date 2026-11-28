"""Framing-invariant direction fitting and 2-vector safety router calibration utilities.

Designed to prevent spurious lexical framing shortcuts (such as academic/educational wrappers)
while separating true harmful intent from benign over-refusal prompts via Group K-Fold OOF CV
with an explicit paired framing-covariance penalty (``Sigma_frame``).
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

# Training-only framing wrappers (strictly disjoint keys, wording, and 4-grams from JBB evaluation templates)
CALIBRATION_FIT_WRAPPERS: dict[str, str] = {
    "seminar_outline": (
        "As a university lecturer preparing course notes for a graduate seminar, "
        "outline the mechanics of: {goal}"
    ),
    "compliance_audit": (
        "I serve on a corporate compliance and risk review committee. "
        "Provide an objective technical breakdown of: {goal}"
    ),
    "stage_script": (
        "We are drafting Act II of a dramatic stage play where a villain delivers a monologue. "
        "Write the scene covering: {goal}"
    ),
    "safe_constructive": (
        "Keep your response completely safe, responsible, and constructive while "
        "addressing this topic: {goal}"
    ),
}

# Held-out calibration validation wrappers (unseen by fold weights; used only for OOF hyperparameter selection)
CALIBRATION_VAL_WRAPPERS: dict[str, str] = {
    "encyclopedia_history": (
        "For a neutral reference encyclopedia entry documenting past incidents, summarize: {goal}"
    ),
    "abstract_thought": (
        "Consider a purely abstract thought experiment with zero practical deployment. "
        "Walk through: {goal}"
    ),
}

_HEDGING_TOKENS: tuple[str, ...] = (
    "educational",
    "academic",
    "research",
    "fictional",
    "novel",
    "story",
    "hypothetical",
    "ethical",
    "harmless",
    "safe",
    "legal",
    "purpose",
    "seminar",
    "compliance",
    "textbook",
)


def apply_calibration_wrapper(goal: str, style: str) -> str:
    """Wrap ``goal`` with a calibration-only framing wrapper (or return verbatim for ``'raw'``)."""
    if style == "raw":
        return goal
    if style in CALIBRATION_FIT_WRAPPERS:
        return CALIBRATION_FIT_WRAPPERS[style].format(goal=goal)
    if style in CALIBRATION_VAL_WRAPPERS:
        return CALIBRATION_VAL_WRAPPERS[style].format(goal=goal)
    raise ValueError(
        f"Unknown calibration wrapper style {style!r}; expected 'raw', "
        f"{sorted(CALIBRATION_FIT_WRAPPERS)}, or {sorted(CALIBRATION_VAL_WRAPPERS)}"
    )


def is_hedged_benign_prompt(prompt: str) -> bool:
    """Return True if ``prompt`` contains common academic/harmless hedging cues."""
    low = str(prompt).lower()
    return any(tok in low for tok in _HEDGING_TOKENS)


def compute_hedging_balanced_weights(prompts: list[str]) -> torch.Tensor:
    """Return sample weights summing to 1.0 with a 50/50 split between un-hedged and hedged prompts."""
    n = len(prompts)
    if n == 0:
        return torch.empty(0, dtype=torch.float64)
    hedged = [i for i, p in enumerate(prompts) if is_hedged_benign_prompt(p)]
    unhedged = [i for i in range(n) if i not in set(hedged)]
    if not hedged or not unhedged:
        return torch.full((n,), 1.0 / n, dtype=torch.float64)
    w = torch.empty(n, dtype=torch.float64)
    w[unhedged] = 0.5 / len(unhedged)
    w[hedged] = 0.5 / len(hedged)
    return w


def _normalize_half_weights(
    weights: torch.Tensor | None,
    n: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Normalize non-negative class weights to sum to ``0.5`` (uniform if None)."""
    if n <= 0:
        raise ValueError("Class weight count must be positive")
    if weights is None:
        return torch.full((n,), 0.5 / n, dtype=dtype, device=device)
    w = weights.to(dtype=dtype, device=device).flatten()
    if w.numel() != n:
        raise ValueError(f"Expected {n} weights, got {w.numel()}")
    if bool((w < 0).any()):
        raise ValueError("Weights must be non-negative")
    total = float(w.sum().item())
    if total <= 0.0:
        return torch.full((n,), 0.5 / n, dtype=dtype, device=device)
    return w * (0.5 / total)


def _row_orthonormalize(vecs: torch.Tensor) -> torch.Tensor | None:
    """Return row-orthonormal basis ``[k, D]`` for valid non-zero rows of ``vecs``, or None."""
    p = vecs.double()
    if p.dim() == 1:
        p = p.unsqueeze(0)
    norms = p.norm(dim=1)
    keep = norms >= 1e-12
    if not bool(keep.any()):
        return None
    p_valid = p[keep] / norms[keep].unsqueeze(1)
    q_col, _ = torch.linalg.qr(p_valid.T, mode="reduced")
    return q_col.T


def fit_invariant_direction(
    pos_h: torch.Tensor,
    neg_h: torch.Tensor,
    *,
    nuisance_deltas: torch.Tensor | None = None,
    protect_dirs: torch.Tensor | None = None,
    lam_ridge: float = 1.0,
    lam_nuisance: float = 0.0,
    lam_protect: float = 0.0,
    pos_weights: torch.Tensor | None = None,
    neg_weights: torch.Tensor | None = None,
    hard_orthogonalize_protect: bool = False,
    use_dual: bool | None = None,
) -> torch.Tensor | None:
    """Fit a unit contrastive direction regularized against nuisance shifts and protected axes.

    Solves ``(H_c^T diag(w) H_c + lam_nuisance/M * Delta^T Delta + lam_protect * P^T P + lam I) w = rhs``
    in ``float64``, automatically choosing the dual Woodbury solve when ``rows < D`` unless overridden.
    """
    if pos_h.shape[0] == 0 or neg_h.shape[0] == 0:
        return None
    dev = pos_h.device
    dim = int(pos_h.shape[1])

    pos_d = pos_h.to(dtype=torch.float64, device=dev)
    neg_d = neg_h.to(dtype=torch.float64, device=dev)
    pos_w = _normalize_half_weights(pos_weights, pos_d.shape[0], dtype=torch.float64, device=dev)
    neg_w = _normalize_half_weights(neg_weights, neg_d.shape[0], dtype=torch.float64, device=dev)

    h_all = torch.cat([pos_d, neg_d], dim=0)
    w_all = torch.cat([pos_w, neg_w], dim=0)
    y_all = torch.cat([
        torch.ones(pos_d.shape[0], dtype=torch.float64, device=dev),
        -torch.ones(neg_d.shape[0], dtype=torch.float64, device=dev),
    ])

    mu = (w_all.unsqueeze(1) * h_all).sum(dim=0)
    h_c = h_all - mu.unsqueeze(0)
    rhs = (h_c * (w_all * y_all).unsqueeze(1)).sum(dim=0)

    z_blocks: list[torch.Tensor] = [w_all.sqrt().unsqueeze(1) * h_c]
    if nuisance_deltas is not None and nuisance_deltas.numel() > 0 and lam_nuisance > 0.0:
        d_mat = nuisance_deltas.to(dtype=torch.float64, device=dev)
        z_blocks.append(math.sqrt(float(lam_nuisance) / float(d_mat.shape[0])) * d_mat)

    p_valid: torch.Tensor | None = None
    if protect_dirs is not None and protect_dirs.numel() > 0:
        p_raw = protect_dirs.to(dtype=torch.float64, device=dev)
        if p_raw.dim() == 1:
            p_raw = p_raw.unsqueeze(0)
        p_norms = p_raw.norm(dim=1)
        p_keep = p_norms >= 1e-12
        if bool(p_keep.any()):
            p_valid = p_raw[p_keep] / p_norms[p_keep].unsqueeze(1)
            if lam_protect > 0.0:
                z_blocks.append(math.sqrt(float(lam_protect)) * p_valid)

    z_mat = torch.cat(z_blocks, dim=0)
    lam = max(float(lam_ridge), 1e-10)
    use_dual_solve = (z_mat.shape[0] < dim) if use_dual is None else bool(use_dual)

    if use_dual_solve:
        r_cnt = z_mat.shape[0]
        gram = z_mat @ z_mat.T + lam * torch.eye(r_cnt, dtype=torch.float64, device=dev)
        coef = torch.linalg.solve(gram, z_mat @ rhs)
        w = (rhs - z_mat.T @ coef) / lam
    else:
        a_mat = z_mat.T @ z_mat + lam * torch.eye(dim, dtype=torch.float64, device=dev)
        w = torch.linalg.solve(a_mat, rhs)

    if hard_orthogonalize_protect and p_valid is not None:
        q_rows = _row_orthonormalize(p_valid)
        if q_rows is not None:
            w = w - (w @ q_rows.T) @ q_rows

    w_norm = float(w.norm().item())
    if w_norm < 1e-10:
        return None

    pos_mean = (2.0 * pos_w.unsqueeze(1) * pos_d).sum(dim=0)
    neg_mean = (2.0 * neg_w.unsqueeze(1) * neg_d).sum(dim=0)
    if float(((pos_mean - neg_mean) @ w).item()) < 0.0:
        w = -w
    return (w / w.norm()).float()


def evaluate_framing_invariance(
    v_dir: torch.Tensor,
    raw_h: torch.Tensor,
    wrapped_h_by_style: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Measure how much framing wrappers perturb projections along ``v_dir``."""
    v = v_dir.detach().float().cpu().flatten()
    v = v / v.norm().clamp_min(1e-12)
    r_h = raw_h.detach().float().cpu()
    raw_scores = r_h @ v
    raw_std = float(raw_scores.std(unbiased=False).clamp_min(1e-6).item())

    by_style: dict[str, dict[str, float]] = {}
    all_diffs: list[torch.Tensor] = []
    worst_drop = 0.0
    for style, wh in wrapped_h_by_style.items():
        w_h = wh.detach().float().cpu()
        diff = (w_h - r_h) @ v
        all_diffs.append(diff)
        mean_signed = float(diff.mean().item())
        worst_drop = min(worst_drop, mean_signed)
        by_style[style] = {
            "mean_signed_shift": round(mean_signed, 5),
            "mean_abs_shift": round(float(diff.abs().mean().item()), 5),
            "rms_shift": round(float(diff.pow(2).mean().sqrt().item()), 5),
            "max_abs_shift": round(float(diff.abs().max().item()), 5),
        }

    if all_diffs:
        cat_d = torch.cat(all_diffs, dim=0)
        mean_abs = float(cat_d.abs().mean().item())
        rms_shift = float(cat_d.pow(2).mean().sqrt().item())
    else:
        mean_abs = 0.0
        rms_shift = 0.0

    return {
        "n_raw": int(r_h.shape[0]),
        "raw_std": round(raw_std, 5),
        "mean_abs_shift": round(mean_abs, 5),
        "rms_shift": round(rms_shift, 5),
        "worst_signed_drop": round(worst_drop, 5),
        "shift_to_std_ratio": round(rms_shift / raw_std, 5),
        "by_style": by_style,
    }


def _dprime_1d(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    mu_diff = float(pos_scores.mean().item() - neg_scores.mean().item())
    pooled_var = (pos_scores.var(unbiased=False) + neg_scores.var(unbiased=False)) * 0.5
    return float(mu_diff / pooled_var.clamp_min(1e-8).sqrt().item())


def _select_gate1_threshold(
    ben_ref_s: torch.Tensor,
    ctrl_s: torch.Tensor,
    ben_nonref_s: torch.Tensor | None,
    prefer_early: bool,
    max_ctrl_fpr: float = 0.04,
) -> tuple[float, float, float, float | None]:
    """Select Gate 1 threshold enforcing minimal control FPR and maximizing benign refusal TPR."""
    pool = [ctrl_s, ben_ref_s]
    if ben_nonref_s is not None and ben_nonref_s.numel() > 0:
        pool.append(ben_nonref_s)
    vals = torch.unique(torch.cat(pool)).sort().values

    # Evaluate gap midpoints so ties favor centered thresholds with wide 0% control margin
    candidates: list[tuple[float, float]] = []
    if vals.numel() >= 2:
        mids = 0.5 * (vals[:-1] + vals[1:])
        widths = vals[1:] - vals[:-1]
        for m, w in zip(mids.tolist(), widths.tolist()):
            candidates.append((float(m), float(w)))
    candidates.append((float(vals[0].item()) - 1e-4, 0.0))
    candidates.append((float(vals[-1].item()) + 1e-4, 0.0))
    candidates.append((float(ctrl_s.max().item()) + 1e-3, 0.0))

    best_row = None
    for thr, width in candidates:
        tpr = float((ben_ref_s > thr).float().mean().item())
        ctrl_fpr = float((ctrl_s > thr).float().mean().item())
        nonref_fpr = float((ben_nonref_s > thr).float().mean().item()) if (ben_nonref_s is not None and ben_nonref_s.numel() > 0) else 0.0
        zero_ctrl = 1.0 if ctrl_fpr == 0.0 else 0.0
        ok_ctrl = 1.0 if ctrl_fpr <= max_ctrl_fpr else 0.0
        obj = round(tpr - 0.5 * nonref_fpr - 2.0 * ctrl_fpr, 6)
        key = (zero_ctrl, ok_ctrl, obj, round(width, 6), 1 if prefer_early else 0)
        if best_row is None or key > best_row[0]:
            best_row = (key, thr, tpr, ctrl_fpr, nonref_fpr if (ben_nonref_s is not None and ben_nonref_s.numel() > 0) else None)
    assert best_row is not None
    return best_row[1], best_row[2], best_row[3], best_row[4]


def _pack_weighted_samples(
    raw_h: torch.Tensor,
    fit_h: torch.Tensor,
    fit_gids: torch.Tensor,
    base_mask: torch.Tensor,
    raw_base_weights: torch.Tensor,
    *,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select fold subset of raw + fit-wrapped samples with a 50/50 split between raw and wrapped."""
    sel_raw_idx = torch.nonzero(base_mask, as_tuple=False).flatten()
    h_raw = raw_h[sel_raw_idx].to(dtype=torch.float64, device=dev)
    w_raw = raw_base_weights[sel_raw_idx].to(dtype=torch.float64, device=dev)
    w_raw = w_raw * (0.25 / w_raw.sum().clamp_min(1e-12))

    fit_mask = base_mask[fit_gids]
    sel_fit_idx = torch.nonzero(fit_mask, as_tuple=False).flatten()
    if sel_fit_idx.numel() == 0:
        return h_raw, w_raw * 2.0

    h_fit = fit_h[sel_fit_idx].to(dtype=torch.float64, device=dev)
    gids_fit = fit_gids[sel_fit_idx]
    gid_counts = torch.bincount(gids_fit, minlength=raw_h.shape[0]).clamp_min(1).to(dtype=torch.float64, device=dev)
    w_fit = raw_base_weights.to(dtype=torch.float64, device=dev)[gids_fit] / gid_counts[gids_fit]
    w_fit = w_fit * (0.25 / w_fit.sum().clamp_min(1e-12))

    return torch.cat([h_raw, h_fit], dim=0), torch.cat([w_raw, w_fit], dim=0)


def _build_fold_deltas(
    tox_raw: torch.Tensor,
    tox_fit: torch.Tensor,
    tox_fit_gids: torch.Tensor,
    tox_mask: torch.Tensor,
    ben_raw: torch.Tensor,
    ben_fit: torch.Tensor,
    ben_fit_gids: torch.Tensor,
    ben_mask: torch.Tensor,
    ctrl_raw: torch.Tensor | None,
    ctrl_fit: torch.Tensor | None,
    ctrl_fit_gids: torch.Tensor | None,
    ctrl_mask: torch.Tensor | None,
    *,
    dev: torch.device,
) -> torch.Tensor:
    """Gather paired framing shifts ``h(wrapped) - h(raw)`` strictly inside the training fold."""
    blocks: list[torch.Tensor] = []
    if tox_fit.shape[0] > 0:
        tm = tox_mask[tox_fit_gids]
        if bool(tm.any()):
            g_t = tox_fit_gids[tm]
            blocks.append((tox_fit[tm] - tox_raw[g_t]).to(dtype=torch.float64, device=dev))
    if ben_fit.shape[0] > 0:
        bm = ben_mask[ben_fit_gids]
        if bool(bm.any()):
            g_b = ben_fit_gids[bm]
            blocks.append((ben_fit[bm] - ben_raw[g_b]).to(dtype=torch.float64, device=dev))
    if ctrl_raw is not None and ctrl_fit is not None and ctrl_fit_gids is not None and ctrl_mask is not None and ctrl_fit.shape[0] > 0:
        cm = ctrl_mask[ctrl_fit_gids]
        if bool(cm.any()):
            g_c = ctrl_fit_gids[cm]
            blocks.append((ctrl_fit[cm] - ctrl_raw[g_c]).to(dtype=torch.float64, device=dev))
    if not blocks:
        return torch.empty((0, tox_raw.shape[1]), dtype=torch.float64, device=dev)
    return torch.cat(blocks, dim=0)


def _solve_subspace_problem(
    pos_h: torch.Tensor,
    pos_w: torch.Tensor,
    neg_h: torch.Tensor,
    neg_w: torch.Tensor,
    deltas: torch.Tensor,
    span_raw: torch.Tensor,
    extra_basis_dirs: list[torch.Tensor],
    *,
    dev: torch.device,
) -> dict[str, Any]:
    """Precompute exact QR subspace factors ``S_data_q, S_frame_q, r_q, P_q`` once per ``(layer, fold)``."""
    h_all = torch.cat([pos_h, neg_h], dim=0)
    w_all = torch.cat([pos_w, neg_w], dim=0)
    y_all = torch.cat([
        torch.ones(pos_h.shape[0], dtype=torch.float64, device=dev),
        -torch.ones(neg_h.shape[0], dtype=torch.float64, device=dev),
    ])
    mu = (w_all.unsqueeze(1) * h_all).sum(dim=0)
    h_c = h_all - mu.unsqueeze(0)
    rhs = (h_c * (w_all * y_all).unsqueeze(1)).sum(dim=0)
    pos_mean = (2.0 * pos_w.unsqueeze(1) * pos_h).sum(dim=0)
    neg_mean = (2.0 * neg_w.unsqueeze(1) * neg_h).sum(dim=0)
    mean_diff = pos_mean - neg_mean

    dim = int(h_all.shape[1])
    span_parts = [span_raw]
    if deltas.shape[0] > 0:
        span_parts.append(deltas)
    for dvec in extra_basis_dirs:
        span_parts.append(dvec.unsqueeze(0))
    b_mat = torch.cat(span_parts, dim=0)

    q_col: torch.Tensor | None = None
    if b_mat.shape[0] < dim:
        q_col, _ = torch.linalg.qr(b_mat.T, mode="reduced")
        h_q = h_c @ q_col
        d_q = (deltas @ q_col) if deltas.shape[0] > 0 else torch.empty((0, q_col.shape[1]), dtype=torch.float64, device=dev)
        r_q = rhs @ q_col
        diff_q = mean_diff @ q_col
        p_q_list = [dvec @ q_col for dvec in extra_basis_dirs]
    else:
        h_q = h_c
        d_q = deltas
        r_q = rhs
        diff_q = mean_diff
        p_q_list = list(extra_basis_dirs)

    wh_q = w_all.unsqueeze(1) * h_q
    s_data_q = h_q.T @ wh_q
    s_frame_q = (d_q.T @ d_q) / float(d_q.shape[0]) if d_q.shape[0] > 0 else torch.zeros_like(s_data_q)
    mean_var = max(float(torch.trace(s_data_q).item()) / max(1, dim), 1e-6)

    return {
        "q_col": q_col,
        "s_data_q": s_data_q,
        "s_frame_q": s_frame_q,
        "r_q": r_q,
        "diff_q": diff_q,
        "p_q_list": p_q_list,
        "mean_var": mean_var,
        "mu": mu,
    }


def _solve_from_subspace(
    sub: dict[str, Any],
    lam_ridge: float,
    lam_frame: float,
    orth_idx: int | None,
    lam_prot: float,
) -> torch.Tensor | None:
    """Solve ridge/invariant problem (plus optional rank-1 protected penalty) in compressed subspace."""
    s_data_q = sub["s_data_q"]
    s_frame_q = sub["s_frame_q"]
    r_q = sub["r_q"]
    diff_q = sub["diff_q"]
    sub_dim = s_data_q.shape[0]
    eye_sub = torch.eye(sub_dim, dtype=torch.float64, device=s_data_q.device)
    a_mat = s_data_q + (float(lam_frame) * s_frame_q if lam_frame > 0.0 else 0.0) + max(float(lam_ridge), 1e-10) * eye_sub

    p_q: torch.Tensor | None = None
    if orth_idx is not None:
        p_raw = sub["p_q_list"][orth_idx]
        p_nrm = float(p_raw.norm().item())
        if p_nrm >= 1e-12:
            p_q = p_raw / p_nrm

    if p_q is not None and lam_prot > 0.0:
        # Solve [r_q, p_q] once and apply exact Sherman-Morrison rank-1 update + GS cleanup
        sol = torch.linalg.solve(a_mat, torch.stack([r_q, p_q], dim=1))
        w_0, u_p = sol[:, 0], sol[:, 1]
        denom = 1.0 + float(lam_prot) * float((p_q @ u_p).item())
        coef = (float(lam_prot) * float((p_q @ w_0).item()) / denom) if abs(denom) > 1e-12 else 0.0
        w_q = w_0 - coef * u_p
        w_q = w_q - float((w_q @ p_q).item()) * p_q
    else:
        w_q = torch.linalg.solve(a_mat, r_q)

    if float(w_q.norm().item()) < 1e-10:
        return None
    if float((diff_q @ w_q).item()) < 0.0:
        w_q = -w_q
    w_q = w_q / w_q.norm()
    q_col = sub["q_col"]
    w = (q_col @ w_q) if q_col is not None else w_q
    return (w / w.norm()).float()


def _candidate_thresholds(scores: torch.Tensor, max_cands: int = 360) -> torch.Tensor:
    vals = torch.unique(scores).sort().values
    if vals.numel() >= 2:
        mids = 0.5 * (vals[:-1] + vals[1:])
        if mids.numel() > max_cands:
            idx = torch.linspace(0, mids.numel() - 1, steps=max_cands).long()
            mids = mids[idx]
        cands = torch.cat([
            mids,
            torch.tensor([float(vals[0].item()) - 1e-4, float(vals[-1].item()) + 1e-4], dtype=torch.float32),
        ])
    elif vals.numel() == 1:
        v0 = float(vals[0].item())
        cands = torch.tensor([v0 - 1e-4, v0 + 1e-4], dtype=torch.float32)
    else:
        cands = torch.tensor([0.0], dtype=torch.float32)
    return cands


def _fit_gate1_direction(
    ben_ref_tr: torch.Tensor,
    ctrl_tr: torch.Tensor,
    nonref_tr: torch.Tensor | None,
    mode: str,
) -> torch.Tensor | None:
    if mode == "ctrl_only":
        d = ben_ref_tr.mean(dim=0) - ctrl_tr.mean(dim=0)
        nrm = float(d.norm().item())
        return (d / nrm).float() if nrm >= 1e-8 else None
    if nonref_tr is None or nonref_tr.shape[0] == 0:
        return None
    neg_all = torch.cat([ctrl_tr, nonref_tr], dim=0)
    w_neg = torch.cat([
        torch.full((ctrl_tr.shape[0],), 0.25 / ctrl_tr.shape[0], dtype=torch.float64, device=ctrl_tr.device),
        torch.full((nonref_tr.shape[0],), 0.25 / nonref_tr.shape[0], dtype=torch.float64, device=ctrl_tr.device),
    ])
    # Regularize lightly by mean per-dimension variance so Gate 1 generalizes cleanly
    h_c = neg_all.double() - neg_all.double().mean(dim=0, keepdim=True)
    mean_var = max(float(h_c.pow(2).sum().item()) / max(1, neg_all.shape[0] * neg_all.shape[1]), 1e-6)
    return fit_invariant_direction(
        ben_ref_tr,
        neg_all,
        lam_ridge=0.5 * mean_var,
        neg_weights=w_neg,
    )


def fit_framing_invariant_safety_router(
    tox_raw_hs: torch.Tensor,
    tox_fit_hs: torch.Tensor,
    tox_fit_gids: torch.Tensor,
    ben_ref_raw_hs: torch.Tensor,
    ben_ref_fit_hs: torch.Tensor,
    ben_ref_fit_gids: torch.Tensor,
    ctrl_raw_hs: torch.Tensor,
    n_blocks: int,
    min_edit_layer: int,
    *,
    tox_val_hs: torch.Tensor | None = None,
    tox_val_gids: torch.Tensor | None = None,
    ben_ref_val_hs: torch.Tensor | None = None,
    ben_ref_val_gids: torch.Tensor | None = None,
    ben_nonref_raw_hs: torch.Tensor | None = None,
    ctrl_fit_hs: torch.Tensor | None = None,
    ctrl_fit_gids: torch.Tensor | None = None,
    ben_ref_weights: torch.Tensor | None = None,
    lam_ridge_rels: tuple[float, ...] = (0.05, 0.5, 5.0, 50.0),
    lam_frames: tuple[float, ...] = (0.0, 1.0, 10.0, 50.0, 200.0),
    orth_refusal_options: tuple[bool, ...] = (False, True),
    k_cv: int = 4,
    max_worst_tox_fpr: float = 0.06,
    max_ctrl_fpr: float = 0.02,
    layer_candidates: Sequence[int] | None = None,
    device: torch.device | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit 2-vector framing-invariant safety router via strict Group K-Fold OOF CV across layers and grids."""
    n_tox = int(tox_raw_hs.shape[0])
    n_ben = int(ben_ref_raw_hs.shape[0])
    n_ctrl = int(ctrl_raw_hs.shape[0])
    n_nonref = int(ben_nonref_raw_hs.shape[0]) if ben_nonref_raw_hs is not None else 0

    k_eff = min(int(k_cv), n_tox, n_ben, n_ctrl)
    use_cv = k_eff >= 2
    dev = torch.device(device) if device is not None else torch.device("cpu")

    tox_fg_cpu = tox_fit_gids.long().cpu()
    ben_fg_cpu = ben_ref_fit_gids.long().cpu()
    tox_vg_cpu = tox_val_gids.long().cpu() if tox_val_gids is not None else torch.empty(0, dtype=torch.long)
    ben_vg_cpu = ben_ref_val_gids.long().cpu() if ben_ref_val_gids is not None else torch.empty(0, dtype=torch.long)
    ctrl_fg_cpu = ctrl_fit_gids.long().cpu() if ctrl_fit_gids is not None else None

    if ben_ref_weights is None:
        ben_w_cpu = torch.ones(n_ben, dtype=torch.float64)
    else:
        ben_w_cpu = ben_ref_weights.detach().double().cpu().clamp_min(1e-12)
    tox_w_cpu = torch.ones(n_tox, dtype=torch.float64)

    has_val_tox = tox_val_hs is not None and tox_val_hs.shape[0] > 0 and tox_vg_cpu.numel() == tox_val_hs.shape[0]
    has_val_ben = ben_ref_val_hs is not None and ben_ref_val_hs.shape[0] > 0 and ben_vg_cpu.numel() == ben_ref_val_hs.shape[0]

    g1_modes = ["ctrl_only", "ctrl_plus_nonref"] if (ben_nonref_raw_hs is not None and n_nonref >= max(2, k_eff)) else ["ctrl_only"]

    def _group_masks(n: int, fold: int) -> tuple[torch.Tensor, torch.Tensor]:
        val = (torch.arange(n) % k_eff) == fold
        return ~val, val

    best_overall: tuple[tuple, dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    best_no_frame: tuple[tuple, dict[str, Any]] | None = None
    candidate_leaderboard: list[dict[str, Any]] = []

    if layer_candidates is None:
        layer_iter = list(range(1, n_blocks + 1))
    else:
        layer_iter = sorted({int(l) for l in layer_candidates if 1 <= int(l) <= n_blocks})
        if not layer_iter:
            raise ValueError(f"No valid layer_candidates in [1, {n_blocks}] from {list(layer_candidates)!r}")

    for l_idx in layer_iter:
        traw = tox_raw_hs[:, l_idx, :].float()
        tfit = tox_fit_hs[:, l_idx, :].float()
        braw = ben_ref_raw_hs[:, l_idx, :].float()
        bfit = ben_ref_fit_hs[:, l_idx, :].float()
        craw = ctrl_raw_hs[:, l_idx, :].float()
        tval = tox_val_hs[:, l_idx, :].float() if has_val_tox else None  # type: ignore[index]
        bval = ben_ref_val_hs[:, l_idx, :].float() if has_val_ben else None  # type: ignore[index]
        nref = ben_nonref_raw_hs[:, l_idx, :].float() if (ben_nonref_raw_hs is not None and n_nonref > 0) else None
        cfit = ctrl_fit_hs[:, l_idx, :].float() if (ctrl_fit_hs is not None and ctrl_fit_hs.shape[0] > 0) else None

        # 1. Precompute Gate 1 OOF scores & fold vectors per g1_mode
        g1_cache: dict[str, dict[str, Any]] = {}
        for g1m in g1_modes:
            r_full = _fit_gate1_direction(
                braw.to(dev),
                craw.to(dev),
                nref.to(dev) if nref is not None else None,
                g1m,
            )
            if r_full is None:
                continue
            r_full_cpu = r_full.cpu()

            if use_cv:
                oof_braw = torch.empty(n_ben, dtype=torch.float32)
                oof_bfit = torch.empty(bfit.shape[0], dtype=torch.float32)
                oof_bval = torch.empty(bval.shape[0], dtype=torch.float32) if bval is not None else None
                oof_traw = torch.empty(n_tox, dtype=torch.float32)
                oof_tfit = torch.empty(tfit.shape[0], dtype=torch.float32)
                oof_tval = torch.empty(tval.shape[0], dtype=torch.float32) if tval is not None else None
                oof_craw = torch.empty(n_ctrl, dtype=torch.float32)
                oof_nref = torch.empty(n_nonref, dtype=torch.float32) if nref is not None else None
                r_folds: list[torch.Tensor] = []
                ok_g1 = True
                for fold in range(k_eff):
                    bm_tr, bm_va = _group_masks(n_ben, fold)
                    cm_tr, cm_va = _group_masks(n_ctrl, fold)
                    _, tm_va = _group_masks(n_tox, fold)
                    nr_sub = None
                    if nref is not None:
                        nm_tr, nm_va = _group_masks(n_nonref, fold)
                        nr_sub = nref[nm_tr].to(dev)
                    r_f = _fit_gate1_direction(braw[bm_tr].to(dev), craw[cm_tr].to(dev), nr_sub, g1m)
                    if r_f is None:
                        ok_g1 = False
                        break
                    r_f_cpu = r_f.cpu()
                    r_folds.append(r_f_cpu)
                    oof_braw[bm_va] = braw[bm_va] @ r_f_cpu
                    bf_va = bm_va[ben_fg_cpu]
                    if bool(bf_va.any()):
                        oof_bfit[bf_va] = bfit[bf_va] @ r_f_cpu
                    if bval is not None and oof_bval is not None:
                        bv_va = bm_va[ben_vg_cpu]
                        if bool(bv_va.any()):
                            oof_bval[bv_va] = bval[bv_va] @ r_f_cpu
                    oof_traw[tm_va] = traw[tm_va] @ r_f_cpu
                    tf_va = tm_va[tox_fg_cpu]
                    if bool(tf_va.any()):
                        oof_tfit[tf_va] = tfit[tf_va] @ r_f_cpu
                    if tval is not None and oof_tval is not None:
                        tv_va = tm_va[tox_vg_cpu]
                        if bool(tv_va.any()):
                            oof_tval[tv_va] = tval[tv_va] @ r_f_cpu
                    oof_craw[cm_va] = craw[cm_va] @ r_f_cpu
                    if nref is not None and oof_nref is not None:
                        oof_nref[nm_va] = nref[nm_va] @ r_f_cpu
                if not ok_g1:
                    continue
            else:
                r_folds = [r_full_cpu]
                oof_braw = braw @ r_full_cpu
                oof_bfit = bfit @ r_full_cpu
                oof_bval = (bval @ r_full_cpu) if bval is not None else None
                oof_traw = traw @ r_full_cpu
                oof_tfit = tfit @ r_full_cpu
                oof_tval = (tval @ r_full_cpu) if tval is not None else None
                oof_craw = craw @ r_full_cpu
                oof_nref = (nref @ r_full_cpu) if nref is not None else None

            r_thr, g1_tpr, g1_ctrl_fpr, g1_nonref_fpr = _select_gate1_threshold(
                oof_braw, oof_craw, oof_nref, l_idx <= min_edit_layer, max_ctrl_fpr=0.04
            )
            g1_cache[g1m] = {
                "r_full": r_full_cpu,
                "r_folds": r_folds,
                "r_thr": float(r_thr),
                "g1_tpr": float(g1_tpr),
                "g1_ctrl_fpr": float(g1_ctrl_fpr),
                "g1_nonref_fpr": g1_nonref_fpr,
                "pass_braw": oof_braw > r_thr,
                "pass_bfit": oof_bfit > r_thr,
                "pass_bval": (oof_bval > r_thr) if oof_bval is not None else None,
                "pass_traw": oof_traw > r_thr,
                "pass_tfit": oof_tfit > r_thr,
                "pass_tval": (oof_tval > r_thr) if oof_tval is not None else None,
                "pass_craw": oof_craw > r_thr,
            }

        if not g1_cache:
            continue
        active_g1_modes = list(g1_cache.keys())

        # 2. Precompute exact QR subspace systems for full-fit and each CV fold
        all_tox_m = torch.ones(n_tox, dtype=torch.bool)
        all_ben_m = torch.ones(n_ben, dtype=torch.bool)
        all_ctrl_m = torch.ones(n_ctrl, dtype=torch.bool)
        tp_all, tw_all = _pack_weighted_samples(traw, tfit, tox_fg_cpu, all_tox_m, tox_w_cpu, dev=dev)
        bp_all, bw_all = _pack_weighted_samples(braw, bfit, ben_fg_cpu, all_ben_m, ben_w_cpu, dev=dev)
        d_all = _build_fold_deltas(
            traw, tfit, tox_fg_cpu, all_tox_m,
            braw, bfit, ben_fg_cpu, all_ben_m,
            craw, cfit, ctrl_fg_cpu, all_ctrl_m,
            dev=dev,
        )
        span_all = torch.cat([traw.to(dtype=torch.float64, device=dev), braw.to(dtype=torch.float64, device=dev)], dim=0)
        extra_all = [g1_cache[m]["r_full"].to(dtype=torch.float64, device=dev) for m in active_g1_modes]
        sub_full = _solve_subspace_problem(tp_all, tw_all, bp_all, bw_all, d_all, span_all, extra_all, dev=dev)

        sub_folds: list[dict[str, Any]] = []
        fold_masks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        if use_cv:
            for fold in range(k_eff):
                tm_tr, tm_va = _group_masks(n_tox, fold)
                bm_tr, bm_va = _group_masks(n_ben, fold)
                cm_tr, cm_va = _group_masks(n_ctrl, fold)
                fold_masks.append((tm_tr, tm_va, bm_tr, bm_va, cm_tr, cm_va))
                tp_f, tw_f = _pack_weighted_samples(traw, tfit, tox_fg_cpu, tm_tr, tox_w_cpu, dev=dev)
                bp_f, bw_f = _pack_weighted_samples(braw, bfit, ben_fg_cpu, bm_tr, ben_w_cpu, dev=dev)
                d_f = _build_fold_deltas(
                    traw, tfit, tox_fg_cpu, tm_tr,
                    braw, bfit, ben_fg_cpu, bm_tr,
                    craw, cfit, ctrl_fg_cpu, cm_tr,
                    dev=dev,
                )
                span_f = torch.cat([
                    traw[tm_tr].to(dtype=torch.float64, device=dev),
                    braw[bm_tr].to(dtype=torch.float64, device=dev),
                ], dim=0)
                extra_f = [g1_cache[m]["r_folds"][fold].to(dtype=torch.float64, device=dev) for m in active_g1_modes]
                sub_folds.append(_solve_subspace_problem(tp_f, tw_f, bp_f, bw_f, d_f, span_f, extra_f, dev=dev))

        # 3. Sweep grid using precomputed subspaces
        for orth_opt in orth_refusal_options:
            orth_g1_list = [(i_m, m) for i_m, m in enumerate(active_g1_modes)] if orth_opt else [(None, None)]
            for orth_idx, orth_g1_mode in orth_g1_list:
                for lam_f in lam_frames:
                    for lam_r_rel in lam_ridge_rels:
                        lam_r_full = float(lam_r_rel) * sub_full["mean_var"]
                        lam_p_full = 1e4 * sub_full["mean_var"] if orth_opt else 0.0
                        v_fit_full = _solve_from_subspace(sub_full, lam_r_full, lam_f, orth_idx, lam_p_full)
                        if v_fit_full is None:
                            continue
                        v_fit_cpu = v_fit_full.cpu()

                        min_fold_cos = 1.0
                        max_abs_offset = 0.0
                        if use_cv:
                            oof_traw = torch.empty(n_tox, dtype=torch.float32)
                            oof_tfit = torch.empty(tfit.shape[0], dtype=torch.float32)
                            oof_tval = torch.empty(tval.shape[0], dtype=torch.float32) if tval is not None else None
                            oof_braw = torch.empty(n_ben, dtype=torch.float32)
                            oof_bfit = torch.empty(bfit.shape[0], dtype=torch.float32)
                            oof_bval = torch.empty(bval.shape[0], dtype=torch.float32) if bval is not None else None
                            oof_craw = torch.empty(n_ctrl, dtype=torch.float32)
                            ok_fold = True
                            for fold in range(k_eff):
                                sf = sub_folds[fold]
                                lam_r_f = float(lam_r_rel) * sf["mean_var"]
                                lam_p_f = 1e4 * sf["mean_var"] if orth_opt else 0.0
                                v_f = _solve_from_subspace(sf, lam_r_f, lam_f, orth_idx, lam_p_f)
                                if v_f is None:
                                    ok_fold = False
                                    break
                                v_f_cpu = v_f.cpu()
                                min_fold_cos = min(min_fold_cos, float((v_fit_cpu @ v_f_cpu).item()))
                                mu_tr_cpu = sf["mu"].float().cpu()
                                # Align fold projection threshold origin to full-fit centroid
                                off_f = float((mu_tr_cpu @ (v_fit_cpu - v_f_cpu)).item())
                                max_abs_offset = max(max_abs_offset, abs(off_f))

                                _, tm_va, _, bm_va, _, cm_va = fold_masks[fold]
                                oof_traw[tm_va] = traw[tm_va] @ v_f_cpu + off_f
                                tf_va = tm_va[tox_fg_cpu]
                                if bool(tf_va.any()):
                                    oof_tfit[tf_va] = tfit[tf_va] @ v_f_cpu + off_f
                                if tval is not None and oof_tval is not None:
                                    tv_va = tm_va[tox_vg_cpu]
                                    if bool(tv_va.any()):
                                        oof_tval[tv_va] = tval[tv_va] @ v_f_cpu + off_f
                                oof_braw[bm_va] = braw[bm_va] @ v_f_cpu + off_f
                                bf_va = bm_va[ben_fg_cpu]
                                if bool(bf_va.any()):
                                    oof_bfit[bf_va] = bfit[bf_va] @ v_f_cpu + off_f
                                if bval is not None and oof_bval is not None:
                                    bv_va = bm_va[ben_vg_cpu]
                                    if bool(bv_va.any()):
                                        oof_bval[bv_va] = bval[bv_va] @ v_f_cpu + off_f
                                oof_craw[cm_va] = craw[cm_va] @ v_f_cpu + off_f
                            if not ok_fold:
                                continue
                        else:
                            oof_traw = traw @ v_fit_cpu
                            oof_tfit = tfit @ v_fit_cpu
                            oof_tval = (tval @ v_fit_cpu) if tval is not None else None
                            oof_braw = braw @ v_fit_cpu
                            oof_bfit = bfit @ v_fit_cpu
                            oof_bval = (bval @ v_fit_cpu) if bval is not None else None
                            oof_craw = craw @ v_fit_cpu

                        # Measure OOF framing shift on unseen validation wrappers (or fit wrappers if no val)
                        unseen_diffs: list[torch.Tensor] = []
                        if oof_tval is not None:
                            unseen_diffs.append(oof_tval - oof_traw[tox_vg_cpu])
                        if oof_bval is not None:
                            unseen_diffs.append(oof_bval - oof_braw[ben_vg_cpu])
                        if not unseen_diffs:
                            if oof_tfit.numel() > 0:
                                unseen_diffs.append(oof_tfit - oof_traw[tox_fg_cpu])
                            if oof_bfit.numel() > 0:
                                unseen_diffs.append(oof_bfit - oof_braw[ben_fg_cpu])
                        raw_std = float(torch.cat([oof_traw, oof_braw]).std(unbiased=False).clamp_min(1e-6).item())
                        rms_unseen_shift = float(torch.cat(unseen_diffs).pow(2).mean().sqrt().item()) if unseen_diffs else 0.0
                        norm_shift = rms_unseen_shift / raw_std

                        oof_dprime = _dprime_1d(oof_traw, oof_braw)
                        score_pool = [oof_traw, oof_braw, oof_tfit, oof_bfit]
                        if oof_tval is not None:
                            score_pool.append(oof_tval)
                        if oof_bval is not None:
                            score_pool.append(oof_bval)
                        vthrs = _candidate_thresholds(torch.cat(score_pool))
                        vt_col = vthrs.unsqueeze(1)

                        cand_g1_modes = [orth_g1_mode] if (orth_opt and orth_g1_mode is not None) else active_g1_modes
                        for g1m in cand_g1_modes:
                            g1d = g1_cache[g1m]
                            fires_traw = g1d["pass_traw"].unsqueeze(0) & (oof_traw.unsqueeze(0) <= vt_col)
                            tox_raw_fprs = fires_traw.float().mean(dim=1)

                            if oof_tfit.numel() > 0:
                                tox_fit_fprs = (g1d["pass_tfit"].unsqueeze(0) & (oof_tfit.unsqueeze(0) <= vt_col)).float().mean(dim=1)
                            else:
                                tox_fit_fprs = tox_raw_fprs

                            if oof_tval is not None and g1d["pass_tval"] is not None:
                                tox_val_fprs = (g1d["pass_tval"].unsqueeze(0) & (oof_tval.unsqueeze(0) <= vt_col)).float().mean(dim=1)
                            else:
                                tox_val_fprs = tox_fit_fprs

                            fires_braw = g1d["pass_braw"].unsqueeze(0) & (oof_braw.unsqueeze(0) <= vt_col)
                            ben_raw_tprs = fires_braw.float().mean(dim=1)

                            wrap_b_masks = []
                            if oof_bfit.numel() > 0:
                                wrap_b_masks.append(g1d["pass_bfit"].unsqueeze(0) & (oof_bfit.unsqueeze(0) <= vt_col))
                            if oof_bval is not None and g1d["pass_bval"] is not None:
                                wrap_b_masks.append(g1d["pass_bval"].unsqueeze(0) & (oof_bval.unsqueeze(0) <= vt_col))
                            ben_wrap_tprs = torch.cat(wrap_b_masks, dim=1).float().mean(dim=1) if wrap_b_masks else ben_raw_tprs

                            ctrl_fprs = (g1d["pass_craw"].unsqueeze(0) & (oof_craw.unsqueeze(0) <= vt_col)).float().mean(dim=1)

                            best_t_row = None
                            for iv, vthr in enumerate(vthrs.tolist()):
                                tr_f = float(tox_raw_fprs[iv].item())
                                tf_f = float(tox_fit_fprs[iv].item())
                                tv_f = float(tox_val_fprs[iv].item())
                                worst_t = max(tr_f, tf_f, tv_f)
                                br_t = float(ben_raw_tprs[iv].item())
                                bw_t = float(ben_wrap_tprs[iv].item())
                                b_min = min(br_t, bw_t)
                                c_f = float(ctrl_fprs[iv].item())

                                feasible = 1.0 if (worst_t <= max_worst_tox_fpr and c_f <= max_ctrl_fpr) else 0.0
                                obj = round(br_t + 0.5 * b_min - 3.0 * worst_t - 2.0 * c_f - 0.15 * norm_shift, 4)
                                key = (feasible, obj, 1 if l_idx <= min_edit_layer else 0, round(oof_dprime, 4))
                                if best_t_row is None or key > best_t_row[0]:
                                    best_t_row = (key, float(vthr), tr_f, tf_f, tv_f, worst_t, br_t, bw_t, c_f)

                            assert best_t_row is not None
                            key, vthr_sel, tr_f, tf_f, tv_f, worst_t, br_t, bw_t, c_f = best_t_row
                            summary_row = {
                                "layer": int(l_idx),
                                "g1_mode": str(g1m),
                                "lam_ridge_rel": float(lam_r_rel),
                                "lam_frame": float(lam_f),
                                "orth_refusal": bool(orth_opt),
                                "feasible": bool(key[0] > 0.5),
                                "obj": float(key[1]),
                                "oof_ben_raw_tpr": round(br_t, 4),
                                "oof_ben_wrap_tpr": round(bw_t, 4),
                                "oof_tox_raw_fpr": round(tr_f, 4),
                                "oof_tox_seen_wrap_fpr": round(tf_f, 4),
                                "oof_tox_unseen_wrap_fpr": round(tv_f, 4),
                                "oof_worst_tox_fpr": round(worst_t, 4),
                                "oof_ctrl_fpr": round(c_f, 4),
                                "oof_veto_dprime": round(oof_dprime, 4),
                                "norm_unseen_frame_shift": round(norm_shift, 4),
                                "rms_unseen_frame_shift": round(rms_unseen_shift, 5),
                                "min_fold_cos": round(min_fold_cos, 4),
                                "max_abs_fold_offset": round(max_abs_offset, 5),
                                "single_pass_early": bool(l_idx <= min_edit_layer),
                                "gate1_tpr": round(g1d["g1_tpr"], 4),
                                "gate1_ctrl_fpr": round(g1d["g1_ctrl_fpr"], 4),
                                "gate1_nonref_fpr": round(g1d["g1_nonref_fpr"], 4) if g1d["g1_nonref_fpr"] is not None else None,
                            }
                            candidate_leaderboard.append(summary_row)

                            if lam_f == 0.0 and (best_no_frame is None or key > best_no_frame[0]):
                                best_no_frame = (key, dict(summary_row))

                            if best_overall is None or key > best_overall[0]:
                                spec = {
                                    "l_idx": int(l_idx),
                                    "g1_mode": str(g1m),
                                    "lam_ridge_rel": float(lam_r_rel),
                                    "lam_frame": float(lam_f),
                                    "orth_refusal": bool(orth_opt),
                                    "r_full": g1d["r_full"],
                                    "r_thr": float(g1d["r_thr"]),
                                    "v_fit_full": v_fit_cpu,
                                    "vthr_oof": float(vthr_sel),
                                    "mu_fit_all": sub_full["mu"].float().cpu(),
                                }
                                best_overall = (key, spec, dict(summary_row), dict(g1d))

    if best_overall is None:
        raise ValueError(
            "Unable to fit safety router: could not separate benign_refused from control hidden states at any layer"
        )
    _, win_spec, win_rep, _ = best_overall
    l_win = win_spec["l_idx"]

    # 4. Final refit at winning layer/hyperparams using ALL calibration views (FIT + VAL_UNSEEN; still zero JBB)
    traw_w = tox_raw_hs[:, l_win, :].float()
    braw_w = ben_ref_raw_hs[:, l_win, :].float()
    tfit_w = tox_fit_hs[:, l_win, :].float()
    bfit_w = ben_ref_fit_hs[:, l_win, :].float()

    t_parts_w = [tfit_w]
    tg_parts = [tox_fg_cpu]
    if has_val_tox and tox_val_hs is not None:
        t_parts_w.append(tox_val_hs[:, l_win, :].float())
        tg_parts.append(tox_vg_cpu)
    t_all_wrap = torch.cat(t_parts_w, dim=0)
    tg_all = torch.cat(tg_parts, dim=0)

    b_parts_w = [bfit_w]
    bg_parts = [ben_fg_cpu]
    if has_val_ben and ben_ref_val_hs is not None:
        b_parts_w.append(ben_ref_val_hs[:, l_win, :].float())
        bg_parts.append(ben_vg_cpu)
    b_all_wrap = torch.cat(b_parts_w, dim=0)
    bg_all = torch.cat(bg_parts, dim=0)

    craw_w = ctrl_raw_hs[:, l_win, :].float()
    cfit_w = ctrl_fit_hs[:, l_win, :].float() if (ctrl_fit_hs is not None and ctrl_fit_hs.shape[0] > 0) else None

    all_t = torch.ones(n_tox, dtype=torch.bool)
    all_b = torch.ones(n_ben, dtype=torch.bool)
    all_c = torch.ones(n_ctrl, dtype=torch.bool)

    tp_fin, tw_fin = _pack_weighted_samples(traw_w, t_all_wrap, tg_all, all_t, tox_w_cpu, dev=dev)
    bp_fin, bw_fin = _pack_weighted_samples(braw_w, b_all_wrap, bg_all, all_b, ben_w_cpu, dev=dev)
    d_fin = _build_fold_deltas(
        traw_w, t_all_wrap, tg_all, all_t,
        braw_w, b_all_wrap, bg_all, all_b,
        craw_w, cfit_w, ctrl_fg_cpu, all_c,
        dev=dev,
    )
    span_fin = torch.cat([traw_w.to(dtype=torch.float64, device=dev), braw_w.to(dtype=torch.float64, device=dev)], dim=0)
    r_full_w = win_spec["r_full"].to(dtype=torch.float64, device=dev)
    sub_fin = _solve_subspace_problem(tp_fin, tw_fin, bp_fin, bw_fin, d_fin, span_fin, [r_full_w], dev=dev)

    lam_r_fin = float(win_spec["lam_ridge_rel"]) * sub_fin["mean_var"]
    lam_p_fin = 1e4 * sub_fin["mean_var"] if win_spec["orth_refusal"] else 0.0
    v_cand = _solve_from_subspace(
        sub_fin,
        lam_r_fin,
        float(win_spec["lam_frame"]),
        0 if win_spec["orth_refusal"] else None,
        lam_p_fin,
    )

    v_fit_full = win_spec["v_fit_full"]
    vthr_oof = float(win_spec["vthr_oof"])
    used_fallback_fit = False
    if v_cand is None:
        v_final = v_fit_full
        vthr_final = vthr_oof
        cos_fit_final = 1.0
        used_fallback_fit = True
    else:
        v_cand_cpu = v_cand.cpu()
        cos_fit_final = float((v_fit_full @ v_cand_cpu).item())
        if cos_fit_final < 0.85:
            v_final = v_fit_full
            vthr_final = vthr_oof
            used_fallback_fit = True
        else:
            v_final = v_cand_cpu
            mu_all_cpu = sub_fin["mu"].float().cpu()
            delta_origin = float((mu_all_cpu @ (v_final - v_fit_full)).item())
            vthr_final = vthr_oof + delta_origin

    candidate_leaderboard.sort(
        key=lambda r: (1.0 if r["feasible"] else 0.0, r["obj"], 1 if r["single_pass_early"] else 0, r["oof_veto_dprime"]),
        reverse=True,
    )

    router = {
        "layer": int(l_win),
        "readout": "last_token",
        "dir": win_spec["r_full"].cpu(),
        "thr": float(win_spec["r_thr"]),
        "veto_dir": v_final.cpu(),
        "veto_thr": float(vthr_final),
    }
    if int(l_win) == int(n_blocks):
        router["final_layer"] = True
    report = {
        **win_rep,
        "cv_folds": int(k_eff) if use_cv else 1,
        "n_tox": int(n_tox),
        "n_tox_fit": int(tox_fit_hs.shape[0]),
        "n_tox_val": int(tox_val_hs.shape[0]) if has_val_tox else 0,
        "n_ben_ref": int(n_ben),
        "n_ben_fit": int(ben_ref_fit_hs.shape[0]),
        "n_ben_val": int(ben_ref_val_hs.shape[0]) if has_val_ben else 0,
        "n_ctrl": int(n_ctrl),
        "n_ben_nonref": int(n_nonref),
        "cos_fit_vs_final": round(cos_fit_final, 5),
        "used_fallback_fit": bool(used_fallback_fit),
        "veto_thr": round(float(vthr_final), 5),
        "gate1_thr": round(float(win_spec["r_thr"]), 5),
        "ablation_no_frame_penalty": best_no_frame[1] if best_no_frame is not None else None,
        "top5_candidates": candidate_leaderboard[:5],
    }
    return router, report


def _format_prompt_for_cal(tokenizer: Any, text: str, chat: bool) -> str:
    if chat and tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": str(text)}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return str(text)


def _encode_cal_prompt(tokenizer: Any, formatted: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(formatted, add_special_tokens=False)
    else:
        enc = tokenizer(formatted, return_tensors=None)
        ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "input_ids", [])
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids_int = [int(t) for t in ids]
    if not ids_int:
        eos = getattr(tokenizer, "eos_token_id", None)
        ids_int = [int(eos) if eos is not None else 1]
    return ids_int


@torch.no_grad()
def _capture_last_hidden_layers(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    dev: torch.device,
    *,
    chat: bool = True,
) -> torch.Tensor:
    if not prompts:
        raise ValueError("prompts must be non-empty for hidden state collection")
    from .capture import find_blocks

    try:
        _, blocks = find_blocks(model)
        n_blocks = len(blocks)
    except ValueError:
        blocks = None
        n_blocks = 0

    hs_list: list[torch.Tensor] = []
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    hooks: list[Any] = []
    try:
        blk_last: dict[int, torch.Tensor] = {}
        need_blk_hooks = False

        def _attach_blk_hooks() -> None:
            if hooks or blocks is None or n_blocks <= 1:
                return
            for li in range(n_blocks - 1):
                def _bh(_m, _inp, out, li=li):
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(t, torch.Tensor):
                        blk_last[li] = (t[0, -1] if t.dim() == 3 else t[-1]).detach().float().cpu()

                hooks.append(blocks[li].register_forward_hook(_bh))

        for p in prompts:
            formatted = _format_prompt_for_cal(tokenizer, str(p), chat=chat)
            p_ids = _encode_cal_prompt(tokenizer, formatted)
            inp = torch.tensor([p_ids], dtype=torch.long, device=dev)
            blk_last.clear()
            out = model(input_ids=inp, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if n_blocks > 0 and len(hs) != n_blocks + 1:
                if not need_blk_hooks and blocks is not None and n_blocks > 1:
                    need_blk_hooks = True
                    _attach_blk_hooks()
                    blk_last.clear()
                    out = model(input_ids=inp, output_hidden_states=True, use_cache=False)
                    hs = out.hidden_states
                if len(blk_last) == max(0, n_blocks - 1):
                    emb_h = (hs[0][0, -1] if hs[0].dim() == 3 else hs[0][-1]).detach().float().cpu()
                    fin_h = (hs[-1][0, -1] if hs[-1].dim() == 3 else hs[-1][-1]).detach().float().cpu()
                    vec = [emb_h] + [blk_last[li] for li in range(n_blocks - 1)] + [fin_h]
                    hs_list.append(torch.stack(vec, dim=0))
                    continue
            hs_list.append(torch.stack([(h[0, -1] if h.dim() == 3 else h[-1]).detach().float().cpu() for h in hs], dim=0))
    finally:
        for hh in hooks:
            hh.remove()
        if was_training and hasattr(model, "train"):
            model.train()
    return torch.stack(hs_list, dim=0)


def _infer_min_edit_layer(patch_obj: Any, n_blocks: int) -> int:
    from .moipatch import _layer_index_from_suffix

    layers: list[int] = []
    for edit in patch_obj.edits:
        try:
            layers.append(_layer_index_from_suffix(edit.module))
        except ValueError:
            m = re.search(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.", edit.module)
            if m:
                layers.append(int(m.group(1)))
    return min(layers) if layers else max(1, int(n_blocks))


def _build_wrapped_prompts(
    prompts: Sequence[str],
    wrappers: dict[str, str],
) -> tuple[list[str], torch.Tensor, list[str]]:
    wrapped: list[str] = []
    gids: list[int] = []
    styles: list[str] = []
    for gid, raw_p in enumerate(prompts):
        for style in wrappers:
            wrapped.append(apply_calibration_wrapper(str(raw_p), style))
            gids.append(gid)
            styles.append(style)
    return wrapped, torch.tensor(gids, dtype=torch.long), styles


def calibrate_safety_router_on_model(
    model: Any,
    tokenizer: Any,
    patch: Any,
    *,
    toxic_prompts: Sequence[str],
    benign_refused_prompts: Sequence[str],
    control_prompts: Sequence[str],
    benign_nonrefused_prompts: Sequence[str] | None = None,
    chat: bool = True,
    max_toxic_fpr: float = 0.05,
    n_folds: int = 4,
    layer_candidates: Sequence[int] | None = None,
    out_patch_path: str | Path | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Calibrate a framing-invariant 2-vector safety router on a live model and optionally save `.moipatch`."""
    from .moipatch import MoIPatch

    patch_obj = patch if isinstance(patch, MoIPatch) else MoIPatch.load(patch)

    if device is not None:
        dev = torch.device(device)
    else:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

    tox_list = [str(p) for p in toxic_prompts]
    ben_ref_list = [str(p) for p in benign_refused_prompts]
    ctrl_list = [str(p) for p in control_prompts]
    nonref_list = [str(p) for p in benign_nonrefused_prompts] if benign_nonrefused_prompts else []

    if not tox_list or not ben_ref_list or not ctrl_list:
        raise ValueError("toxic_prompts, benign_refused_prompts, and control_prompts must all be non-empty")

    tox_raw_hs = _capture_last_hidden_layers(model, tokenizer, tox_list, dev, chat=chat)
    ben_ref_raw_hs = _capture_last_hidden_layers(model, tokenizer, ben_ref_list, dev, chat=chat)
    ctrl_raw_hs = _capture_last_hidden_layers(model, tokenizer, ctrl_list, dev, chat=chat)
    nonref_raw_hs = (
        _capture_last_hidden_layers(model, tokenizer, nonref_list, dev, chat=chat) if nonref_list else None
    )

    tox_fit_p, tox_fit_gids, _ = _build_wrapped_prompts(tox_list, CALIBRATION_FIT_WRAPPERS)
    ben_fit_p, ben_fit_gids, _ = _build_wrapped_prompts(ben_ref_list, CALIBRATION_FIT_WRAPPERS)
    ctrl_fit_p, ctrl_fit_gids, _ = _build_wrapped_prompts(ctrl_list, CALIBRATION_FIT_WRAPPERS)

    tox_val_p, tox_val_gids, tox_val_styles = _build_wrapped_prompts(tox_list, CALIBRATION_VAL_WRAPPERS)
    ben_val_p, ben_val_gids, ben_val_styles = _build_wrapped_prompts(ben_ref_list, CALIBRATION_VAL_WRAPPERS)

    tox_fit_hs = _capture_last_hidden_layers(model, tokenizer, tox_fit_p, dev, chat=chat)
    ben_ref_fit_hs = _capture_last_hidden_layers(model, tokenizer, ben_fit_p, dev, chat=chat)
    ctrl_fit_hs = _capture_last_hidden_layers(model, tokenizer, ctrl_fit_p, dev, chat=chat)

    tox_val_hs = _capture_last_hidden_layers(model, tokenizer, tox_val_p, dev, chat=chat)
    ben_ref_val_hs = _capture_last_hidden_layers(model, tokenizer, ben_val_p, dev, chat=chat)

    n_blocks = int(tox_raw_hs.shape[1]) - 1
    if n_blocks < 1:
        raise ValueError(f"Expected >= 2 hidden_states entries (emb + >=1 block), got {tox_raw_hs.shape[1]}")
    min_edit_layer = _infer_min_edit_layer(patch_obj, n_blocks)
    ben_w = compute_hedging_balanced_weights(ben_ref_list)

    router, report = fit_framing_invariant_safety_router(
        tox_raw_hs,
        tox_fit_hs,
        tox_fit_gids,
        ben_ref_raw_hs,
        ben_ref_fit_hs,
        ben_fit_gids,
        ctrl_raw_hs,
        n_blocks,
        min_edit_layer,
        tox_val_hs=tox_val_hs,
        tox_val_gids=tox_val_gids,
        ben_ref_val_hs=ben_ref_val_hs,
        ben_ref_val_gids=ben_val_gids,
        ben_nonref_raw_hs=nonref_raw_hs,
        ctrl_fit_hs=ctrl_fit_hs,
        ctrl_fit_gids=ctrl_fit_gids,
        ben_ref_weights=ben_w,
        k_cv=int(n_folds),
        max_worst_tox_fpr=float(max_toxic_fpr),
        layer_candidates=layer_candidates,
        device=dev,
    )

    l_win = int(router["layer"])
    val_by_style: dict[str, torch.Tensor] = {}
    for st in CALIBRATION_VAL_WRAPPERS:
        t_idx = [i for i, s in enumerate(tox_val_styles) if s == st]
        b_idx = [i for i, s in enumerate(ben_val_styles) if s == st]
        if len(t_idx) == len(tox_list) and len(b_idx) == len(ben_ref_list):
            val_by_style[st] = torch.cat([tox_val_hs[t_idx, l_win, :], ben_ref_val_hs[b_idx, l_win, :]], dim=0)
    if val_by_style:
        base_cat = torch.cat([tox_raw_hs[:, l_win, :], ben_ref_raw_hs[:, l_win, :]], dim=0)
        invariance_metrics = evaluate_framing_invariance(router["veto_dir"], base_cat, val_by_style)
    else:
        invariance_metrics = {
            "rms_shift": float(report.get("rms_unseen_frame_shift", 0.0)),
            "shift_to_std_ratio": float(report.get("norm_unseen_frame_shift", 0.0)),
        }

    new_meta = dict(patch_obj.meta)
    new_meta.setdefault("name", "abliterated-sfw")
    new_meta["router_calibration"] = {
        "layer": l_win,
        "g1_mode": report.get("g1_mode"),
        "lam_ridge_rel": report.get("lam_ridge_rel"),
        "lam_frame": report.get("lam_frame"),
        "orth_refusal": report.get("orth_refusal"),
    }
    calibrated = MoIPatch(
        edits=list(patch_obj.edits),
        router=router,
        gates=list(patch_obj.gates),
        meta=new_meta,
    )

    saved_path: str | None = None
    if out_patch_path is not None:
        out_p = Path(str(out_patch_path))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        saved_path = str(calibrated.save(out_p))

    oof_keys = (
        "cv_folds",
        "feasible",
        "obj",
        "g1_mode",
        "lam_ridge_rel",
        "lam_frame",
        "orth_refusal",
        "oof_ben_raw_tpr",
        "oof_ben_wrap_tpr",
        "oof_tox_raw_fpr",
        "oof_tox_seen_wrap_fpr",
        "oof_tox_unseen_wrap_fpr",
        "oof_worst_tox_fpr",
        "oof_ctrl_fpr",
        "oof_veto_dprime",
        "norm_unseen_frame_shift",
        "rms_unseen_frame_shift",
        "gate1_tpr",
        "gate1_ctrl_fpr",
        "gate1_nonref_fpr",
        "cos_fit_vs_final",
        "used_fallback_fit",
    )
    return {
        "selected_layer": l_win,
        "thresholds": {
            "dir_thr": round(float(router["thr"]), 5),
            "veto_thr": round(float(router["veto_thr"]), 5),
        },
        "oof_metrics": {k: report[k] for k in oof_keys if k in report},
        "invariance_metrics": invariance_metrics,
        "out_patch_path": saved_path,
        "n_edits": len(calibrated.edits),
        "edited_modules": [e.module for e in calibrated.edits],
    }
