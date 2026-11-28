"""Closed-form Weight-Space Programming (WSP) multi-layer safety & intent routers.

Inspired by arXiv:2611.33972 ("Fine-Tuning Without Gradients? Closed-Form,
LLM-Written Classifiers on a Frozen Encoder"), adapted to **zero extra encoder
forward passes** by fusing a single prefill forward pass across:
  1. L2-normalized multi-layer residual features (eliminates PreNorm scale drift),
  2. Exact OAS Shrinkage LDA solved via low-rank Woodbury identity in O(R^3 + R^2 D),
  3. Framing-invariant Kernel Ridge Regression (RBF & Poly-2) with paired Sigma_frame,
  4. Native Token-0 refusal/probability prior features from logits[:, -1, :],
  5. Strict Group K-Fold Cross-Fitting over candidate closed-form heads.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .router_cal import _normalize_half_weights


def select_probe_layers(n_blocks: int, k: int = 5) -> list[int]:
    """Select sorted unique hidden_states layer indices spanning `[1..n_blocks]`."""
    nb = max(1, int(n_blocks))
    if int(k) == 5:
        fracs = (0.25, 0.50, 0.75, 0.90, 1.0)
    else:
        kk = max(1, int(k))
        fracs = tuple(float(i + 1) / float(kk) for i in range(kk))
    layers = sorted({min(nb, max(1, int(round(f * nb)))) for f in fracs})
    return layers


def extract_multilayer_features(hs: torch.Tensor, layers: Sequence[int]) -> torch.Tensor:
    """L2-normalize each selected residual layer slice and concatenate into `[N, L*D]` float64."""
    parts: list[torch.Tensor] = []
    for li in layers:
        x = hs[:, int(li), :].to(dtype=torch.float64)
        nrm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        parts.append(x / nrm)
    return torch.cat(parts, dim=-1)


_DEFAULT_REFUSAL_OPENERS: tuple[str, ...] = (
    "I",
    " I",
    "I'm",
    " I'm",
    "Sorry",
    " Sorry",
    "As",
    " As",
    "Unfortunately",
    " Unfortunately",
    "No",
    " No",
)


def resolve_refusal_opener_ids(tok: Any) -> list[int]:
    """Resolve unique single-token vocabulary IDs for typical refusal opening tokens."""
    ids: set[int] = set()
    for txt in _DEFAULT_REFUSAL_OPENERS:
        if hasattr(tok, "encode"):
            try:
                enc = tok.encode(txt, add_special_tokens=False)
                if isinstance(enc, list) and len(enc) == 1 and int(enc[0]) >= 0:
                    ids.add(int(enc[0]))
            except Exception:
                pass
        if hasattr(tok, "convert_tokens_to_ids"):
            try:
                tid = tok.convert_tokens_to_ids(txt)
                if isinstance(tid, int) and tid >= 0:
                    ids.add(int(tid))
            except Exception:
                pass
    return sorted(ids)


@torch.no_grad()
def extract_token0_prior_features(
    logits_last: torch.Tensor,
    opener_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Compute `[N, 4]` closed-form token-0 features: `[ref_logp, ref_prob, top1_prob, top32_entropy]`."""
    lg = logits_last.float()
    if lg.dim() == 1:
        lg = lg.unsqueeze(0)
    lp = F.log_softmax(lg, dim=-1)
    n, vocab = int(lp.shape[0]), int(lp.shape[1])

    valid_ids = [int(i) for i in (opener_ids or ()) if 0 <= int(i) < vocab]
    if valid_ids:
        idx_t = torch.tensor(valid_ids, dtype=torch.long, device=lp.device)
        ref_lp = torch.logsumexp(lp[:, idx_t], dim=-1)
    else:
        ref_lp = torch.full((n,), -20.0, dtype=lp.dtype, device=lp.device)
    ref_prob = torch.exp(ref_lp).clamp(0.0, 1.0)
    top1_prob = torch.exp(lp.max(dim=-1).values).clamp(0.0, 1.0)

    k_ent = min(32, vocab)
    top_lp, _ = torch.topk(lp, k=k_ent, dim=-1)
    top_p = torch.softmax(top_lp, dim=-1)
    ent = -(top_p * top_p.clamp_min(1e-12).log()).sum(dim=-1) / max(1e-6, math.log(float(max(2, k_ent))))
    return torch.stack([ref_lp, ref_prob, top1_prob, ent], dim=-1).to(dtype=torch.float64, device="cpu")


def _prepare_balanced_xyw(
    pos_x: torch.Tensor,
    neg_x: torch.Tensor,
    pos_weights: torch.Tensor | None = None,
    neg_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p = pos_x.to(dtype=torch.float64)
    n = neg_x.to(dtype=torch.float64)
    wp = _normalize_half_weights(pos_weights, p.shape[0], dtype=torch.float64, device=p.device)
    wn = _normalize_half_weights(neg_weights, n.shape[0], dtype=torch.float64, device=n.device)
    x_all = torch.cat([p, n], dim=0)
    w_all = torch.cat([wp, wn], dim=0)
    y_all = torch.cat([
        torch.ones(p.shape[0], dtype=torch.float64, device=p.device),
        -torch.ones(n.shape[0], dtype=torch.float64, device=n.device),
    ])
    mu_pos = (2.0 * wp.unsqueeze(1) * p).sum(dim=0)
    mu_neg = (2.0 * wn.unsqueeze(1) * n).sum(dim=0)
    return x_all, y_all, w_all, mu_pos, mu_neg


def fit_ridge_binary_head(
    pos_x: torch.Tensor,
    neg_x: torch.Tensor,
    *,
    pos_weights: torch.Tensor | None = None,
    neg_weights: torch.Tensor | None = None,
    nuisance_deltas: torch.Tensor | None = None,
    lam_ridge_rel: float = 1.0,
    lam_frame: float = 0.0,
    use_dual: bool | None = None,
) -> dict[str, Any]:
    """Fit unit binary linear head with dual/primal Ridge + optional framing covariance penalty."""
    x_all, y_all, w_all, mu_pos, mu_neg = _prepare_balanced_xyw(pos_x, neg_x, pos_weights, neg_weights)
    dev = x_all.device
    dim = int(x_all.shape[1])

    mu = (w_all.unsqueeze(1) * x_all).sum(dim=0)
    x_c = x_all - mu.unsqueeze(0)
    rhs = (x_c * (w_all * y_all).unsqueeze(1)).sum(dim=0)

    z_data = w_all.sqrt().unsqueeze(1) * x_c
    mean_var = max(float(z_data.pow(2).sum().item()) / max(1, dim), 1e-8)
    lam = max(float(lam_ridge_rel) * mean_var, 1e-10)

    z_blocks = [z_data]
    if nuisance_deltas is not None and nuisance_deltas.numel() > 0 and float(lam_frame) > 0.0:
        d_mat = nuisance_deltas.to(dtype=torch.float64, device=dev)
        z_blocks.append(math.sqrt(float(lam_frame) / float(d_mat.shape[0])) * d_mat)
    z_mat = torch.cat(z_blocks, dim=0)

    dual_flag = (z_mat.shape[0] < dim) if use_dual is None else bool(use_dual)
    if dual_flag:
        gram = z_mat @ z_mat.T + lam * torch.eye(z_mat.shape[0], dtype=torch.float64, device=dev)
        coef = torch.linalg.solve(gram, z_mat @ rhs)
        w = (rhs - z_mat.T @ coef) / lam
    else:
        a_mat = z_mat.T @ z_mat + lam * torch.eye(dim, dtype=torch.float64, device=dev)
        w = torch.linalg.solve(a_mat, rhs)

    mean_diff = mu_pos - mu_neg
    if float(w.norm().item()) < 1e-10:
        w = mean_diff / mean_diff.norm().clamp_min(1e-12)
    else:
        if float((mean_diff @ w).item()) < 0.0:
            w = -w
        w = w / w.norm().clamp_min(1e-12)

    mid = 0.5 * (mu_pos + mu_neg)
    b = -float((mid @ w).item())
    return {"type": "linear", "w": w.cpu(), "b": b}


def fit_shrinkage_lda_head(
    pos_x: torch.Tensor,
    neg_x: torch.Tensor,
    *,
    pos_weights: torch.Tensor | None = None,
    neg_weights: torch.Tensor | None = None,
    nuisance_deltas: torch.Tensor | None = None,
    lam_frame: float = 0.0,
    use_dual: bool | None = None,
) -> dict[str, Any]:
    """Fit exact OAS Shrinkage LDA (`Sigma_reg = (1-rho) S_w + rho*mu*I + lam_f*Sigma_frame`) via Woodbury."""
    p = pos_x.to(dtype=torch.float64)
    n = neg_x.to(dtype=torch.float64)
    dev = p.device
    dim = int(p.shape[1])

    wp = _normalize_half_weights(pos_weights, p.shape[0], dtype=torch.float64, device=dev)
    wn = _normalize_half_weights(neg_weights, n.shape[0], dtype=torch.float64, device=dev)
    w_all = torch.cat([wp, wn], dim=0)

    mu_pos = (2.0 * wp.unsqueeze(1) * p).sum(dim=0)
    mu_neg = (2.0 * wn.unsqueeze(1) * n).sum(dim=0)
    p_c = p - mu_pos.unsqueeze(0)
    n_c = n - mu_neg.unsqueeze(0)
    x_within = torch.cat([p_c, n_c], dim=0)
    u_w = w_all.sqrt().unsqueeze(1) * x_within  # S_w = U_w^T U_w

    # Exact OAS shrinkage coefficient using low-rank trace identities (invariant to D >> N)
    gram_w = u_w @ u_w.T  # [N, N]
    tr_s = float(torch.trace(gram_w).item())
    tr_s2 = float((gram_w * gram_w).sum().item())
    n_eff = max(2.0, 1.0 / float((w_all * w_all).sum().clamp_min(1e-12).item()))
    num = (1.0 - 2.0 / float(dim)) * tr_s2 + tr_s * tr_s
    den = (n_eff + 1.0 - 2.0 / float(dim)) * (tr_s2 - (tr_s * tr_s) / float(max(1, dim)))
    rho = min(1.0, max(1e-4, num / max(den, 1e-30))) if den > 1e-30 else 1.0

    mu_iso = max(tr_s / float(max(1, dim)), 1e-10)
    alpha = max(rho * mu_iso, 1e-10)
    rhs = mu_pos - mu_neg

    b_blocks = [math.sqrt(max(0.0, 1.0 - rho)) * u_w]
    if nuisance_deltas is not None and nuisance_deltas.numel() > 0 and float(lam_frame) > 0.0:
        d_mat = nuisance_deltas.to(dtype=torch.float64, device=dev)
        b_blocks.append(math.sqrt(float(lam_frame) / float(d_mat.shape[0])) * d_mat)
    b_mat = torch.cat(b_blocks, dim=0)

    dual_flag = (b_mat.shape[0] < dim) if use_dual is None else bool(use_dual)
    if dual_flag and b_mat.shape[0] > 0:
        r_cnt = b_mat.shape[0]
        inner = b_mat @ b_mat.T + alpha * torch.eye(r_cnt, dtype=torch.float64, device=dev)
        coef = torch.linalg.solve(inner, b_mat @ rhs)
        w = (rhs - b_mat.T @ coef) / alpha
    else:
        a_mat = b_mat.T @ b_mat + alpha * torch.eye(dim, dtype=torch.float64, device=dev)
        w = torch.linalg.solve(a_mat, rhs)

    if float(w.norm().item()) < 1e-10:
        w = rhs / rhs.norm().clamp_min(1e-12)
    else:
        if float((rhs @ w).item()) < 0.0:
            w = -w
        w = w / w.norm().clamp_min(1e-12)

    mid = 0.5 * (mu_pos + mu_neg)
    b = -float((mid @ w).item())
    return {"type": "linear", "w": w.cpu(), "b": b, "oas_rho": float(rho)}


def predict_linear_head(head: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    w = head["w"].to(dtype=torch.float64, device=x.device)
    return (x.to(dtype=torch.float64) @ w + float(head["b"])).cpu()


def _compute_kernel_matrix(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    *,
    kernel: str,
    gamma: float,
    coef0: float = 1.0,
) -> torch.Tensor:
    xa = x_a.to(dtype=torch.float64)
    xb = x_b.to(dtype=torch.float64)
    if kernel == "rbf":
        na = (xa * xa).sum(dim=-1, keepdim=True)
        nb = (xb * xb).sum(dim=-1, keepdim=True).T
        d2 = (na + nb - 2.0 * (xa @ xb.T)).clamp_min(0.0)
        return torch.exp(-float(gamma) * d2)
    if kernel == "poly2":
        scale = float(max(1, xa.shape[1]))
        return ((xa @ xb.T) / math.sqrt(scale) + float(coef0)).pow(2)
    raise ValueError(f"Unsupported kernel: {kernel}")


def fit_krr_binary_head(
    pos_x: torch.Tensor,
    neg_x: torch.Tensor,
    *,
    pos_weights: torch.Tensor | None = None,
    neg_weights: torch.Tensor | None = None,
    kernel: str = "rbf",
    gamma_scale: float = 1.0,
    coef0: float = 1.0,
    lam_rel: float = 1.0,
    pair_raw_x: torch.Tensor | None = None,
    pair_wrap_x: torch.Tensor | None = None,
    lam_frame: float = 0.0,
) -> dict[str, Any]:
    """Fit weighted Kernel Ridge binary classifier with optional paired framing invariance."""
    x_all, y_all, w_all, _, _ = _prepare_balanced_xyw(pos_x, neg_x, pos_weights, neg_weights)
    dev = x_all.device
    n = int(x_all.shape[0])

    gamma = 1.0
    if kernel == "rbf":
        na = (x_all * x_all).sum(dim=-1, keepdim=True)
        d2 = (na + na.T - 2.0 * (x_all @ x_all.T)).clamp_min(0.0)
        tri = d2[torch.triu(torch.ones((n, n), dtype=torch.bool, device=dev), diagonal=1)]
        med = float(tri.median().item()) if tri.numel() > 0 else 1.0
        gamma = float(gamma_scale) / max(med, 1e-6)

    k_mat = _compute_kernel_matrix(x_all, x_all, kernel=kernel, gamma=gamma, coef0=coef0)
    k_diag_mean = max(float(torch.diag(k_mat).mean().item()), 1e-6)
    lam_eff = max(float(lam_rel) * k_diag_mean / float(max(1, n)), 1e-9)

    w_col = w_all.unsqueeze(1)
    lhs = k_mat @ (w_col * k_mat) + lam_eff * k_mat + 1e-8 * k_diag_mean * torch.eye(n, dtype=torch.float64, device=dev)
    rhs = k_mat @ (w_all * y_all)

    if (
        pair_raw_x is not None
        and pair_wrap_x is not None
        and pair_raw_x.shape[0] > 0
        and pair_raw_x.shape[0] == pair_wrap_x.shape[0]
        and float(lam_frame) > 0.0
    ):
        kr = _compute_kernel_matrix(pair_raw_x.to(dev), x_all, kernel=kernel, gamma=gamma, coef0=coef0)
        kw = _compute_kernel_matrix(pair_wrap_x.to(dev), x_all, kernel=kernel, gamma=gamma, coef0=coef0)
        dk = kr - kw
        lhs = lhs + (float(lam_frame) / float(dk.shape[0])) * (dk.T @ dk)

    alpha = torch.linalg.solve(lhs, rhs)
    return {
        "type": "krr",
        "kernel": str(kernel),
        "gamma": float(gamma),
        "coef0": float(coef0),
        "x_train": x_all.cpu(),
        "alpha": alpha.cpu(),
    }


def predict_krr_head(head: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    x_tr = head["x_train"].to(dtype=torch.float64, device=x.device)
    alpha = head["alpha"].to(dtype=torch.float64, device=x.device)
    k_eval = _compute_kernel_matrix(
        x.to(dtype=torch.float64),
        x_tr,
        kernel=head["kernel"],
        gamma=float(head["gamma"]),
        coef0=float(head["coef0"]),
    )
    return (k_eval @ alpha).cpu()


def predict_head_score(head: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    if head["type"] == "krr":
        return predict_krr_head(head, x)
    return predict_linear_head(head, x)


def fit_token0_fused_gate(
    hidden_oof: torch.Tensor,
    token0_feats: torch.Tensor,
    y: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    lam: float = 1.0,
) -> dict[str, Any]:
    """Closed-form Ridge stacking over `[hidden_score, token0_prior_4d]` using standardized features."""
    feat = torch.cat([hidden_oof.to(dtype=torch.float64).unsqueeze(1), token0_feats.to(dtype=torch.float64)], dim=1)
    y_d = y.to(dtype=torch.float64)
    n = int(feat.shape[0])
    w = torch.ones(n, dtype=torch.float64) if sample_weight is None else sample_weight.to(dtype=torch.float64)
    w = w / w.sum().clamp_min(1e-12)

    mu = (w.unsqueeze(1) * feat).sum(dim=0)
    var = (w.unsqueeze(1) * (feat - mu.unsqueeze(0)).pow(2)).sum(dim=0)
    std = var.sqrt().clamp_min(1e-6)
    z = (feat - mu.unsqueeze(0)) / std.unsqueeze(0)

    wz = w.unsqueeze(1) * z
    a = z.T @ wz + float(lam) * torch.eye(feat.shape[1], dtype=torch.float64)
    b_rhs = (wz * y_d.unsqueeze(1)).sum(dim=0)
    coef = torch.linalg.solve(a, b_rhs)
    # Keep positive sign alignment with y
    if float((coef[0]).item()) < 0.0 and float(((z @ coef) * y_d * w).sum().item()) < 0.0:
        coef = -coef
    return {"type": "t0_stack", "mu": mu.cpu(), "std": std.cpu(), "coef": coef.cpu()}


def predict_token0_fused_gate(
    stack_head: dict[str, Any],
    hidden_score: torch.Tensor,
    token0_feats: torch.Tensor,
) -> torch.Tensor:
    feat = torch.cat([hidden_score.to(dtype=torch.float64).unsqueeze(1), token0_feats.to(dtype=torch.float64)], dim=1)
    mu = stack_head["mu"].to(feat.device)
    std = stack_head["std"].to(feat.device)
    coef = stack_head["coef"].to(feat.device)
    return ((feat - mu.unsqueeze(0)) / std.unsqueeze(0) @ coef).cpu()


def _candidate_thrs(vals: torch.Tensor, max_cands: int = 96) -> torch.Tensor:
    u = torch.unique(vals.float()).sort().values
    if u.numel() == 0:
        return torch.tensor([0.0])
    if u.numel() == 1:
        return torch.tensor([float(u[0]) - 1e-3, float(u[0]) + 1e-3])
    mids = 0.5 * (u[:-1] + u[1:])
    cands = torch.cat([torch.tensor([float(u[0]) - 1e-3]), mids, torch.tensor([float(u[-1]) + 1e-3])])
    if cands.numel() > max_cands:
        idx = torch.linspace(0, cands.numel() - 1, max_cands).long()
        cands = cands[idx]
    return cands


def _select_gate1_thr(
    ben_ref_s: torch.Tensor,
    ctrl_s: torch.Tensor,
    ben_nonref_s: torch.Tensor | None,
    *,
    max_ctrl_fpr: float = 0.04,
) -> tuple[float, float, float, float]:
    pool = [ben_ref_s, ctrl_s] + ([ben_nonref_s] if (ben_nonref_s is not None and ben_nonref_s.numel() > 0) else [])
    cands = _candidate_thrs(torch.cat(pool))
    best = None
    for thr in cands.tolist():
        tpr = float((ben_ref_s > thr).float().mean().item())
        cf = float((ctrl_s > thr).float().mean().item())
        nrf = float((ben_nonref_s > thr).float().mean().item()) if (ben_nonref_s is not None and ben_nonref_s.numel() > 0) else 0.0
        zero_c = 1.0 if cf == 0.0 else 0.0
        ok_c = 1.0 if cf <= max_ctrl_fpr else 0.0
        obj = round(tpr - 0.5 * nrf - 2.0 * cf, 6)
        key = (zero_c, ok_c, obj)
        if best is None or key > best[0]:
            best = (key, float(thr), tpr, cf, nrf)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def _select_veto_thr_combined(
    g1_pass_braw: torch.Tensor,
    g1_pass_bwrap: torch.Tensor,
    g1_pass_traw: torch.Tensor,
    g1_pass_tfit: torch.Tensor,
    g1_pass_tval: torch.Tensor,
    g1_pass_craw: torch.Tensor,
    g1_pass_nonref: torch.Tensor | None,
    vs_braw: torch.Tensor,
    vs_bwrap: torch.Tensor,
    vs_traw: torch.Tensor,
    vs_tfit: torch.Tensor,
    vs_tval: torch.Tensor,
    vs_craw: torch.Tensor,
    vs_nonref: torch.Tensor | None,
    *,
    norm_shift: float,
    max_worst_tox_fpr: float = 0.04,
    max_ctrl_fpr: float = 0.0,
) -> dict[str, Any]:
    pool = [vs_braw, vs_traw, vs_tfit]
    if vs_bwrap.numel() > 0:
        pool.append(vs_bwrap)
    if vs_tval.numel() > 0:
        pool.append(vs_tval)
    cands = _candidate_thrs(torch.cat(pool), max_cands=128)
    best = None
    for vt in cands.tolist():
        br = float((g1_pass_braw & (vs_braw <= vt)).float().mean().item())
        bw = float((g1_pass_bwrap & (vs_bwrap <= vt)).float().mean().item()) if vs_bwrap.numel() > 0 else br
        tr = float((g1_pass_traw & (vs_traw <= vt)).float().mean().item())
        tf = float((g1_pass_tfit & (vs_tfit <= vt)).float().mean().item()) if vs_tfit.numel() > 0 else tr
        tv = float((g1_pass_tval & (vs_tval <= vt)).float().mean().item()) if vs_tval.numel() > 0 else tf
        cf = float((g1_pass_craw & (vs_craw <= vt)).float().mean().item())
        nrf = float((g1_pass_nonref & (vs_nonref <= vt)).float().mean().item()) if (g1_pass_nonref is not None and vs_nonref is not None and vs_nonref.numel() > 0) else 0.0
        worst_t = max(tr, tf, tv)
        feasible = 1.0 if (worst_t <= max_worst_tox_fpr and cf <= max_ctrl_fpr) else 0.0
        obj = round(br + 0.5 * min(br, bw) - 0.35 * nrf - 3.0 * worst_t - 2.0 * cf - 0.15 * norm_shift, 6)
        key = (feasible, obj)
        if best is None or key > best[0]:
            best = (key, {
                "feasible": bool(feasible > 0.5),
                "obj": obj,
                "veto_thr": float(vt),
                "oof_ben_raw_tpr": round(br, 4),
                "oof_ben_wrap_tpr": round(bw, 4),
                "oof_tox_raw_fpr": round(tr, 4),
                "oof_tox_seen_wrap_fpr": round(tf, 4),
                "oof_tox_unseen_wrap_fpr": round(tv, 4),
                "oof_worst_tox_fpr": round(worst_t, 4),
                "oof_ctrl_fpr": round(cf, 4),
                "oof_ben_nonref_fpr": round(nrf, 4),
                "norm_unseen_frame_shift": round(norm_shift, 4),
            })
    assert best is not None
    return best[1]


def fit_wsp_closed_form_safety_router(
    *,
    tox_raw_hs: torch.Tensor,
    tox_fit_hs: torch.Tensor,
    tox_fit_gids: torch.Tensor,
    ben_raw_hs: torch.Tensor,
    ben_fit_hs: torch.Tensor,
    ben_fit_gids: torch.Tensor,
    ctrl_raw_hs: torch.Tensor,
    n_blocks: int,
    tox_val_hs: torch.Tensor | None = None,
    tox_val_gids: torch.Tensor | None = None,
    ben_val_hs: torch.Tensor | None = None,
    ben_val_gids: torch.Tensor | None = None,
    ben_nonref_raw_hs: torch.Tensor | None = None,
    ctrl_fit_hs: torch.Tensor | None = None,
    ctrl_fit_gids: torch.Tensor | None = None,
    ben_ref_weights: torch.Tensor | None = None,
    tox_raw_t0: torch.Tensor | None = None,
    tox_fit_t0: torch.Tensor | None = None,
    tox_val_t0: torch.Tensor | None = None,
    ben_raw_t0: torch.Tensor | None = None,
    ben_fit_t0: torch.Tensor | None = None,
    ben_val_t0: torch.Tensor | None = None,
    ctrl_raw_t0: torch.Tensor | None = None,
    ben_nonref_t0: torch.Tensor | None = None,
    probe_layers: Sequence[int] | None = None,
    single_layer_1L: int | None = None,
    k_cv: int = 5,
    max_worst_tox_fpr: float = 0.04,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run strict Group K-Fold Cross-Fitting over 8 closed-form WSP families + `wsp_crossfit_auto`.

    Returns `(predictors_by_family, oof_report)`. Every entry `predictors_by_family[name]` is a callable
    `fn(hs_eval: torch.Tensor, t0_eval: torch.Tensor | None = None) -> torch.BoolTensor` using ONE forward pass.
    """
    layers_ml = list(probe_layers) if probe_layers is not None else select_probe_layers(n_blocks, k=5)
    layer_1l = int(single_layer_1L) if single_layer_1L is not None else min(int(n_blocks), max(1, int(round(0.86 * n_blocks))))

    def _feat_ml(hs: torch.Tensor | None) -> torch.Tensor | None:
        return extract_multilayer_features(hs, layers_ml) if (hs is not None and hs.shape[0] > 0) else None

    def _feat_1l(hs: torch.Tensor | None) -> torch.Tensor | None:
        return extract_multilayer_features(hs, [layer_1l]) if (hs is not None and hs.shape[0] > 0) else None

    traw_ml, tfit_ml = _feat_ml(tox_raw_hs), _feat_ml(tox_fit_hs)
    braw_ml, bfit_ml = _feat_ml(ben_raw_hs), _feat_ml(ben_fit_hs)
    craw_ml = _feat_ml(ctrl_raw_hs)
    cfit_ml = _feat_ml(ctrl_fit_hs)
    tval_ml = _feat_ml(tox_val_hs)
    bval_ml = _feat_ml(ben_val_hs)
    nr_ml = _feat_ml(ben_nonref_raw_hs)

    traw_1l, tfit_1l = _feat_1l(tox_raw_hs), _feat_1l(tox_fit_hs)
    braw_1l, bfit_1l = _feat_1l(ben_raw_hs), _feat_1l(ben_fit_hs)
    craw_1l = _feat_1l(ctrl_raw_hs)
    cfit_1l = _feat_1l(ctrl_fit_hs)
    tval_1l = _feat_1l(tox_val_hs)
    bval_1l = _feat_1l(ben_val_hs)
    nr_1l = _feat_1l(ben_nonref_raw_hs)

    n_tox, n_ben, n_ctrl = traw_ml.shape[0], braw_ml.shape[0], craw_ml.shape[0]
    k_eff = max(2, min(int(k_cv), n_tox, n_ben, n_ctrl))

    tox_w_base = torch.ones(n_tox, dtype=torch.float64) / float(n_tox)
    ben_w_base = (
        ben_ref_weights.to(dtype=torch.float64).cpu()
        if ben_ref_weights is not None
        else (torch.ones(n_ben, dtype=torch.float64) / float(n_ben))
    )
    ctrl_w_base = torch.ones(n_ctrl, dtype=torch.float64) / float(n_ctrl)

    def _fold_mask(n: int, fold: int) -> tuple[torch.Tensor, torch.Tensor]:
        va = (torch.arange(n) % k_eff) == fold
        return ~va, va

    def _pack_split(
        raw_x: torch.Tensor,
        fit_x: torch.Tensor | None,
        fit_gids: torch.Tensor | None,
        mask: torch.Tensor,
        base_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx_r = torch.nonzero(mask, as_tuple=False).flatten()
        xr = raw_x[idx_r]
        wr = base_w[idx_r]
        wr = wr / wr.sum().clamp_min(1e-12)
        if fit_x is None or fit_gids is None or fit_x.shape[0] == 0:
            return xr, wr, torch.empty((0, raw_x.shape[1]), dtype=torch.float64), torch.empty((0, raw_x.shape[1]), dtype=torch.float64)
        fm = mask[fit_gids]
        idx_f = torch.nonzero(fm, as_tuple=False).flatten()
        if idx_f.numel() == 0:
            return xr, wr, torch.empty((0, raw_x.shape[1]), dtype=torch.float64), torch.empty((0, raw_x.shape[1]), dtype=torch.float64)
        gf = fit_gids[idx_f]
        xf = fit_x[idx_f]
        cnt = torch.bincount(gf, minlength=raw_x.shape[0]).clamp_min(1).double()[gf]
        wf = base_w[gf] / cnt
        wf = wf / wf.sum().clamp_min(1e-12)
        return torch.cat([xr, xf], dim=0), torch.cat([0.5 * wr, 0.5 * wf], dim=0), raw_x[gf], xf

    # Build strict OOF hidden scores across 5 base feature/model estimators (for both Gate 1 and Veto)
    base_specs = (
        ("ridge_ml", "ml", "ridge", {"lam_ridge_rel": 1.0, "lam_frame": 1.0}),
        ("slda_1l", "1l", "slda", {"lam_frame": 1.0}),
        ("slda_ml", "ml", "slda", {"lam_frame": 1.0}),
        ("krr_rbf_ml", "ml", "krr", {"kernel": "rbf", "gamma_scale": 1.0, "lam_rel": 0.5, "lam_frame": 1.0}),
        ("krr_poly2_ml", "ml", "krr", {"kernel": "poly2", "coef0": 1.0, "lam_rel": 1.0, "lam_frame": 1.0}),
    )

    def _fit_estimator(kind: str, kw: dict[str, Any], pos_x: torch.Tensor, neg_x: torch.Tensor, pos_w: torch.Tensor, neg_w: torch.Tensor, pr_x: torch.Tensor, pw_x: torch.Tensor) -> dict[str, Any]:
        deltas = pw_x - pr_x if (pr_x.shape[0] > 0 and pw_x.shape[0] == pr_x.shape[0]) else None
        if kind == "ridge":
            return fit_ridge_binary_head(pos_x, neg_x, pos_weights=pos_w, neg_weights=neg_w, nuisance_deltas=deltas, **kw)
        if kind == "slda":
            return fit_shrinkage_lda_head(pos_x, neg_x, pos_weights=pos_w, neg_weights=neg_w, nuisance_deltas=deltas, **kw)
        if kind == "krr":
            return fit_krr_binary_head(pos_x, neg_x, pos_weights=pos_w, neg_weights=neg_w, pair_raw_x=pr_x, pair_wrap_x=pw_x, **kw)
        raise ValueError(kind)

    oof_g1: dict[str, dict[str, torch.Tensor]] = {}
    oof_veto: dict[str, dict[str, torch.Tensor]] = {}
    full_g1: dict[str, dict[str, Any]] = {}
    full_veto: dict[str, dict[str, Any]] = {}

    for spec_nm, feat_mode, kind, kw in base_specs:
        tr, tf, tv = (traw_1l, tfit_1l, tval_1l) if feat_mode == "1l" else (traw_ml, tfit_ml, tval_ml)
        br, bf, bv = (braw_1l, bfit_1l, bval_1l) if feat_mode == "1l" else (braw_ml, bfit_ml, bval_ml)
        cr, cf = (craw_1l, cfit_1l) if feat_mode == "1l" else (craw_ml, cfit_ml)
        nr = nr_1l if feat_mode == "1l" else nr_ml

        # Gate 1 (benign_ref vs ctrl_raw) and Veto (tox vs ben_ref)
        def _empty_bucket() -> dict[str, torch.Tensor]:
            out = {
                "traw": torch.empty(n_tox, dtype=torch.float64),
                "tfit": torch.empty(tf.shape[0], dtype=torch.float64) if tf is not None else torch.empty(0, dtype=torch.float64),
                "tval": torch.empty(tv.shape[0], dtype=torch.float64) if tv is not None else torch.empty(0, dtype=torch.float64),
                "braw": torch.empty(n_ben, dtype=torch.float64),
                "bfit": torch.empty(bf.shape[0], dtype=torch.float64) if bf is not None else torch.empty(0, dtype=torch.float64),
                "bval": torch.empty(bv.shape[0], dtype=torch.float64) if bv is not None else torch.empty(0, dtype=torch.float64),
                "craw": torch.empty(n_ctrl, dtype=torch.float64),
                "nonref": torch.empty(0, dtype=torch.float64),
            }
            return out

        g1_b = _empty_bucket()
        vt_b = _empty_bucket()

        for fold in range(k_eff):
            tm_tr, tm_va = _fold_mask(n_tox, fold)
            bm_tr, bm_va = _fold_mask(n_ben, fold)
            cm_tr, cm_va = _fold_mask(n_ctrl, fold)

            px_g1, pw_g1, pr_b, pw_b = _pack_split(br, bf, ben_fit_gids, bm_tr, ben_w_base)
            nx_g1, nw_g1, pr_c, pw_c = _pack_split(cr, cf, ctrl_fit_gids, cm_tr, ctrl_w_base)
            pr_g1 = torch.cat([pr_b, pr_c], dim=0)
            pw_g1_p = torch.cat([pw_b, pw_c], dim=0)
            h_g1 = _fit_estimator(kind, kw, px_g1, nx_g1, pw_g1, nw_g1, pr_g1, pw_g1_p)

            px_vt, pw_vt, pr_t, pw_t = _pack_split(tr, tf, tox_fit_gids, tm_tr, tox_w_base)
            pr_vt = torch.cat([pr_t, pr_b], dim=0)
            pw_vt_p = torch.cat([pw_t, pw_b], dim=0)
            h_vt = _fit_estimator(kind, kw, px_vt, px_g1, pw_vt, pw_g1, pr_vt, pw_vt_p)

            for bucket, hd in ((g1_b, h_g1), (vt_b, h_vt)):
                bucket["traw"][tm_va] = predict_head_score(hd, tr[tm_va])
                bucket["braw"][bm_va] = predict_head_score(hd, br[bm_va])
                bucket["craw"][cm_va] = predict_head_score(hd, cr[cm_va])
                if tf is not None and tf.shape[0] > 0:
                    mf = tm_va[tox_fit_gids]
                    if bool(mf.any()):
                        bucket["tfit"][mf] = predict_head_score(hd, tf[mf])
                if tv is not None and tox_val_gids is not None and tv.shape[0] > 0:
                    mv = tm_va[tox_val_gids]
                    if bool(mv.any()):
                        bucket["tval"][mv] = predict_head_score(hd, tv[mv])
                if bf is not None and bf.shape[0] > 0:
                    mf = bm_va[ben_fit_gids]
                    if bool(mf.any()):
                        bucket["bfit"][mf] = predict_head_score(hd, bf[mf])
                if bv is not None and ben_val_gids is not None and bv.shape[0] > 0:
                    mv = bm_va[ben_val_gids]
                    if bool(mv.any()):
                        bucket["bval"][mv] = predict_head_score(hd, bv[mv])

        # Full fit across ALL calibration (raw + fit + val)
        all_t = torch.ones(n_tox, dtype=torch.bool)
        all_b = torch.ones(n_ben, dtype=torch.bool)
        all_c = torch.ones(n_ctrl, dtype=torch.bool)
        t_all_w = torch.cat([x for x in (tf, tv) if x is not None and x.shape[0] > 0], dim=0) if (tf is not None or tv is not None) else None
        t_all_g = torch.cat([g for x, g in ((tf, tox_fit_gids), (tv, tox_val_gids)) if x is not None and g is not None and x.shape[0] > 0], dim=0) if t_all_w is not None else None
        b_all_w = torch.cat([x for x in (bf, bv) if x is not None and x.shape[0] > 0], dim=0) if (bf is not None or bv is not None) else None
        b_all_g = torch.cat([g for x, g in ((bf, ben_fit_gids), (bv, ben_val_gids)) if x is not None and g is not None and x.shape[0] > 0], dim=0) if b_all_w is not None else None

        px_g1_f, pw_g1_f, pr_bf, pw_bf = _pack_split(br, b_all_w, b_all_g, all_b, ben_w_base)
        nx_g1_f, nw_g1_f, pr_cf, pw_cf = _pack_split(cr, cf, ctrl_fit_gids, all_c, ctrl_w_base)
        px_vt_f, pw_vt_f, pr_tf, pw_tf = _pack_split(tr, t_all_w, t_all_g, all_t, tox_w_base)

        full_g1[spec_nm] = _fit_estimator(kind, kw, px_g1_f, nx_g1_f, pw_g1_f, nw_g1_f, torch.cat([pr_bf, pr_cf], dim=0), torch.cat([pw_bf, pw_cf], dim=0))
        full_veto[spec_nm] = _fit_estimator(kind, kw, px_vt_f, px_g1_f, pw_vt_f, pw_g1_f, torch.cat([pr_tf, pr_bf], dim=0), torch.cat([pw_tf, pw_bf], dim=0))

        if nr is not None and nr.shape[0] > 0:
            g1_b["nonref"] = predict_head_score(full_g1[spec_nm], nr)
            vt_b["nonref"] = predict_head_score(full_veto[spec_nm], nr)

        oof_g1[spec_nm] = g1_b
        oof_veto[spec_nm] = vt_b

    has_t0 = tox_raw_t0 is not None and ben_raw_t0 is not None and ctrl_raw_t0 is not None

    # Build token-0 fused Gate 1 variants strictly out-of-fold
    g1_variants: dict[str, tuple[str, dict[str, torch.Tensor], dict[str, Any] | None]] = {}
    for nm in oof_g1:
        g1_variants[nm] = (nm, oof_g1[nm], None)

    if has_t0:
        for base_nm in ("ridge_ml", "slda_ml", "krr_rbf_ml"):
            gb = oof_g1[base_nm]
            assert ben_raw_t0 is not None and ctrl_raw_t0 is not None
            # Fit stacking strictly via K-Fold OOF as well
            b_oof = torch.empty(n_ben, dtype=torch.float64)
            c_oof = torch.empty(n_ctrl, dtype=torch.float64)
            t_oof = torch.empty(n_tox, dtype=torch.float64)
            tf_oof = torch.empty(gb["tfit"].shape[0], dtype=torch.float64) if gb["tfit"].numel() > 0 else torch.empty(0, dtype=torch.float64)
            tv_oof = torch.empty(gb["tval"].shape[0], dtype=torch.float64) if gb["tval"].numel() > 0 else torch.empty(0, dtype=torch.float64)
            bf_oof = torch.empty(gb["bfit"].shape[0], dtype=torch.float64) if gb["bfit"].numel() > 0 else torch.empty(0, dtype=torch.float64)
            bv_oof = torch.empty(gb["bval"].shape[0], dtype=torch.float64) if gb["bval"].numel() > 0 else torch.empty(0, dtype=torch.float64)

            for fold in range(k_eff):
                _, tm_va = _fold_mask(n_tox, fold)
                bm_tr, bm_va = _fold_mask(n_ben, fold)
                cm_tr, cm_va = _fold_mask(n_ctrl, fold)
                feat_hs = torch.cat([gb["braw"][bm_tr], gb["craw"][cm_tr]])
                feat_t0 = torch.cat([ben_raw_t0[bm_tr], ctrl_raw_t0[cm_tr]], dim=0)
                y_st = torch.cat([torch.ones(int(bm_tr.sum())), -torch.ones(int(cm_tr.sum()))])
                w_st = torch.cat([0.5 * ben_w_base[bm_tr] / ben_w_base[bm_tr].sum(), 0.5 * ctrl_w_base[cm_tr] / ctrl_w_base[cm_tr].sum()])
                st_h = fit_token0_fused_gate(feat_hs, feat_t0, y_st, sample_weight=w_st, lam=2.0)

                b_oof[bm_va] = predict_token0_fused_gate(st_h, gb["braw"][bm_va], ben_raw_t0[bm_va])
                c_oof[cm_va] = predict_token0_fused_gate(st_h, gb["craw"][cm_va], ctrl_raw_t0[cm_va])
                t_oof[tm_va] = predict_token0_fused_gate(st_h, gb["traw"][tm_va], tox_raw_t0[tm_va])
                if tox_fit_t0 is not None and gb["tfit"].numel() > 0:
                    mf = tm_va[tox_fit_gids]
                    if bool(mf.any()):
                        tf_oof[mf] = predict_token0_fused_gate(st_h, gb["tfit"][mf], tox_fit_t0[mf])
                if tox_val_t0 is not None and tox_val_gids is not None and gb["tval"].numel() > 0:
                    mv = tm_va[tox_val_gids]
                    if bool(mv.any()):
                        tv_oof[mv] = predict_token0_fused_gate(st_h, gb["tval"][mv], tox_val_t0[mv])
                if ben_fit_t0 is not None and gb["bfit"].numel() > 0:
                    mf = bm_va[ben_fit_gids]
                    if bool(mf.any()):
                        bf_oof[mf] = predict_token0_fused_gate(st_h, gb["bfit"][mf], ben_fit_t0[mf])
                if ben_val_t0 is not None and ben_val_gids is not None and gb["bval"].numel() > 0:
                    mv = bm_va[ben_val_gids]
                    if bool(mv.any()):
                        bv_oof[mv] = predict_token0_fused_gate(st_h, gb["bval"][mv], ben_val_t0[mv])

            full_st = fit_token0_fused_gate(
                torch.cat([gb["braw"], gb["craw"]]),
                torch.cat([ben_raw_t0, ctrl_raw_t0], dim=0),
                torch.cat([torch.ones(n_ben), -torch.ones(n_ctrl)]),
                sample_weight=torch.cat([0.5 * ben_w_base, 0.5 * ctrl_w_base]),
                lam=2.0,
            )
            nr_oof = predict_token0_fused_gate(full_st, gb["nonref"], ben_nonref_t0) if (ben_nonref_t0 is not None and gb["nonref"].numel() > 0) else torch.empty(0)
            g1_variants[f"{base_nm}_plus_t0"] = (
                base_nm,
                {"traw": t_oof, "tfit": tf_oof, "tval": tv_oof, "braw": b_oof, "bfit": bf_oof, "bval": bv_oof, "craw": c_oof, "nonref": nr_oof},
                full_st,
            )

    # Evaluate the 8 named WSP families cleanly + `wsp_crossfit_auto` across all pairs
    family_map = [
        ("multilayer_ridge", "ridge_ml", "ridge_ml"),
        ("shrinkage_lda_1L", "slda_1l", "slda_1l"),
        ("shrinkage_lda_ML", "slda_ml", "slda_ml"),
        ("krr_rbf_ML", "krr_rbf_ml", "krr_rbf_ml"),
        ("krr_poly2_ML", "krr_poly2_ml", "krr_poly2_ml"),
    ]
    if has_t0:
        family_map.extend([
            ("multilayer_ridge_plus_t0", "ridge_ml_plus_t0", "ridge_ml"),
            ("shrinkage_lda_ML_plus_t0", "slda_ml_plus_t0", "slda_ml"),
            ("krr_rbf_ML_plus_t0", "krr_rbf_ml_plus_t0", "krr_rbf_ml"),
        ])

    def _eval_pair(g1_var_nm: str, vt_nm: str) -> tuple[dict[str, Any], float]:
        _, g1_s, _ = g1_variants[g1_var_nm]
        vt_s = oof_veto[vt_nm]
        g1_thr, g1_tpr, g1_cf, g1_nrf = _select_gate1_thr(
            g1_s["braw"], g1_s["craw"], g1_s["nonref"] if g1_s["nonref"].numel() > 0 else None
        )
        p_braw = g1_s["braw"] > g1_thr
        p_bwrap = torch.cat([x for x in (g1_s["bfit"], g1_s["bval"]) if x.numel() > 0]) > g1_thr
        p_traw = g1_s["traw"] > g1_thr
        p_tfit = g1_s["tfit"] > g1_thr
        p_tval = g1_s["tval"] > g1_thr
        p_craw = g1_s["craw"] > g1_thr
        p_nonref = (g1_s["nonref"] > g1_thr) if g1_s["nonref"].numel() > 0 else None

        vs_bwrap = torch.cat([x for x in (vt_s["bfit"], vt_s["bval"]) if x.numel() > 0])
        diffs: list[torch.Tensor] = []
        if vt_s["tval"].numel() > 0 and tox_val_gids is not None:
            diffs.append(vt_s["tval"] - vt_s["traw"][tox_val_gids])
        if vt_s["bval"].numel() > 0 and ben_val_gids is not None:
            diffs.append(vt_s["bval"] - vt_s["braw"][ben_val_gids])
        raw_std = float(torch.cat([vt_s["traw"], vt_s["braw"]]).std(unbiased=False).clamp_min(1e-6).item())
        norm_shift = float(torch.cat(diffs).pow(2).mean().sqrt().item()) / raw_std if diffs else 0.0

        sel_v = _select_veto_thr_combined(
            p_braw, p_bwrap, p_traw, p_tfit, p_tval, p_craw, p_nonref,
            vt_s["braw"], vs_bwrap, vt_s["traw"], vt_s["tfit"], vt_s["tval"], vt_s["craw"],
            vt_s["nonref"] if vt_s["nonref"].numel() > 0 else None,
            norm_shift=norm_shift,
            max_worst_tox_fpr=max_worst_tox_fpr,
            max_ctrl_fpr=0.0,
        )
        sel_v.update({"g1_thr": round(g1_thr, 5), "g1_tpr": round(g1_tpr, 4), "g1_ctrl_fpr": round(g1_cf, 4), "g1_nonref_fpr": round(g1_nrf, 4)})
        return sel_v, g1_thr

    def _make_predictor(g1_var_nm: str, vt_nm: str, g1_thr: float, vt_thr: float):
        base_g1_nm, _, t0_stack = g1_variants[g1_var_nm]
        hd_g1 = full_g1[base_g1_nm]
        hd_vt = full_veto[vt_nm]
        use_1l_g1 = base_g1_nm.endswith("_1l")
        use_1l_vt = vt_nm.endswith("_1l")

        def _predict(hs_eval: torch.Tensor, t0_eval: torch.Tensor | None = None) -> torch.BoolTensor:
            x_g1 = _feat_1l(hs_eval) if use_1l_g1 else _feat_ml(hs_eval)
            x_vt = _feat_1l(hs_eval) if use_1l_vt else _feat_ml(hs_eval)
            assert x_g1 is not None and x_vt is not None
            s1 = predict_head_score(hd_g1, x_g1)
            if t0_stack is not None:
                if t0_eval is None:
                    raise ValueError("token0 prior features required for *_plus_t0 router")
                s1 = predict_token0_fused_gate(t0_stack, s1, t0_eval)
            s2 = predict_head_score(hd_vt, x_vt)
            return ((s1 > float(g1_thr)) & (s2 <= float(vt_thr))).bool()

        return _predict

    predictors: dict[str, Any] = {}
    oof_leaderboard: dict[str, dict[str, Any]] = {}
    for fam_nm, g1_var_nm, vt_nm in family_map:
        rep, g1_thr = _eval_pair(g1_var_nm, vt_nm)
        oof_leaderboard[fam_nm] = {"g1_variant": g1_var_nm, "veto_variant": vt_nm, **rep}
        predictors[fam_nm] = _make_predictor(g1_var_nm, vt_nm, g1_thr, rep["veto_thr"])

    # Cross-fit auto selection across ALL (g1_variant x veto_variant) pairs
    best_auto = None
    for g1_var_nm in g1_variants:
        for vt_nm in oof_veto:
            rep, g1_thr = _eval_pair(g1_var_nm, vt_nm)
            key = (1.0 if rep["feasible"] else 0.0, rep["obj"])
            if best_auto is None or key > best_auto[0]:
                best_auto = (key, g1_var_nm, vt_nm, rep, g1_thr)

    assert best_auto is not None
    _, win_g1, win_vt, win_rep, win_g1_thr = best_auto
    oof_leaderboard["wsp_crossfit_auto"] = {"g1_variant": win_g1, "veto_variant": win_vt, **win_rep}
    predictors["wsp_crossfit_auto"] = _make_predictor(win_g1, win_vt, win_g1_thr, win_rep["veto_thr"])

    report = {
        "probe_layers_ML": layers_ml,
        "probe_layer_1L": layer_1l,
        "cv_folds": k_eff,
        "winner": {"g1_variant": win_g1, "veto_variant": win_vt, **win_rep},
        "oof_leaderboard": oof_leaderboard,
    }
    return predictors, report
