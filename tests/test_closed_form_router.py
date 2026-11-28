"""Unit tests for `wsi.closed_form_router` (OAS Shrinkage LDA, KRR, Token-0 Prior & Group K-Fold Cross-Fitting)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi.closed_form_router import (  # noqa: E402
    extract_multilayer_features,
    extract_token0_prior_features,
    fit_krr_binary_head,
    fit_ridge_binary_head,
    fit_shrinkage_lda_head,
    fit_wsp_closed_form_safety_router,
    predict_krr_head,
    predict_linear_head,
    select_probe_layers,
)


def test_woodbury_oas_shrinkage_lda_matches_dense_primal():
    """Exact low-rank Woodbury OAS Shrinkage LDA must match naive O(D^3) dense covariance solve."""
    torch.manual_seed(7)
    n_pos, n_neg, dim = 24, 28, 180  # D > N regime
    pos = torch.randn(n_pos, dim, dtype=torch.float64) * 0.4 + 0.35
    neg = torch.randn(n_neg, dim, dtype=torch.float64) * 0.4 - 0.35
    deltas = torch.randn(16, dim, dtype=torch.float64) * 0.6

    head = fit_shrinkage_lda_head(pos, neg, nuisance_deltas=deltas, lam_frame=1.5, use_dual=True)
    # Dense primal check
    mu_p = pos.mean(dim=0)
    mu_n = neg.mean(dim=0)
    xc = torch.cat([pos - mu_p, neg - mu_n], dim=0)
    sw = (xc.T @ xc) / float(n_pos + n_neg)
    sw = 0.5 * ((pos - mu_p).T @ (pos - mu_p) / n_pos + (neg - mu_n).T @ (neg - mu_n) / n_neg)
    rho = head["oas_rho"]
    # Recompute exact OAS mu_iso from trace
    mu_iso = float(torch.trace(sw).item()) / float(dim)
    sigma_reg = (1.0 - rho) * sw + rho * mu_iso * torch.eye(dim, dtype=torch.float64)
    sigma_reg = sigma_reg + (1.5 / deltas.shape[0]) * (deltas.T @ deltas)
    w_dense = torch.linalg.solve(sigma_reg, mu_p - mu_n)
    w_dense = w_dense / w_dense.norm()
    diff = float((head["w"].double() - w_dense).abs().max().item())
    assert diff < 1e-8, diff


def test_krr_separates_nonlinear_xor_and_resists_framing():
    """RBF KRR solves nonlinear XOR classes cleanly and framing penalty reduces wrapper shift."""
    torch.manual_seed(11)
    # 2D XOR (+ 2 noise dims): linear head is ~50%, RBF KRR hits >=95%
    signs_a = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float64)
    signs_b = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    pos_2d = signs_a[torch.arange(60) % 2] + torch.randn(60, 2, dtype=torch.float64) * 0.2
    neg_2d = signs_b[torch.arange(60) % 2] + torch.randn(60, 2, dtype=torch.float64) * 0.2
    r_outer = torch.cat([pos_2d, torch.randn(60, 2, dtype=torch.float64) * 0.1], dim=1)
    r_inner = torch.cat([neg_2d, torch.randn(60, 2, dtype=torch.float64) * 0.1], dim=1)

    lin_h = fit_ridge_binary_head(r_outer[:40], r_inner[:40], lam_ridge_rel=0.5)
    lin_acc = 0.5 * (
        float((predict_linear_head(lin_h, r_outer[40:]) > 0).float().mean())
        + float((predict_linear_head(lin_h, r_inner[40:]) <= 0).float().mean())
    )
    krr_h = fit_krr_binary_head(r_outer[:40], r_inner[:40], kernel="rbf", gamma_scale=1.0, lam_rel=0.1)
    krr_acc = 0.5 * (
        float((predict_krr_head(krr_h, r_outer[40:]) > 0).float().mean())
        + float((predict_krr_head(krr_h, r_inner[40:]) <= 0).float().mean())
    )
    assert lin_acc < 0.70, lin_acc
    assert krr_acc >= 0.95, krr_acc

    # Paired framing shift along coord 0..3 should shrink under lam_frame > 0
    raw_p = torch.randn(20, 16, dtype=torch.float64) * 0.5 + 0.8
    raw_n = torch.randn(20, 16, dtype=torch.float64) * 0.5 - 0.8
    shift = torch.zeros(20, 16, dtype=torch.float64)
    shift[:, 0:4] = 2.5
    wrap_p = raw_p + shift
    wrap_n = raw_n + shift
    k_no = fit_krr_binary_head(raw_p, raw_n, lam_frame=0.0)
    k_fr = fit_krr_binary_head(
        raw_p, raw_n, pair_raw_x=torch.cat([raw_p, raw_n]), pair_wrap_x=torch.cat([wrap_p, wrap_n]), lam_frame=10.0
    )
    s_no = float((predict_krr_head(k_no, wrap_p) - predict_krr_head(k_no, raw_p)).abs().mean())
    s_fr = float((predict_krr_head(k_fr, wrap_p) - predict_krr_head(k_fr, raw_p)).abs().mean())
    assert s_fr < s_no * 0.5, (s_fr, s_no)


def test_wsp_crossfit_safety_router_and_token0_prior():
    """End-to-end synthetic multi-layer + token-0 prior cross-fit check across all 8 WSP families."""
    torch.manual_seed(3)
    n_tox, n_ben, n_ctrl = 25, 25, 25
    n_blocks, dim = 8, 24

    def _make_hs(n: int, trigger: float, harm: float, frame: float) -> torch.Tensor:
        hs = torch.randn(n, n_blocks + 1, dim, dtype=torch.float32) * 0.1
        hs[:, 4:, 0] += trigger
        hs[:, 6:, 1] += harm
        hs[:, :, 2] += frame
        return hs

    tox_raw = _make_hs(n_tox, 2.0, 2.0, 0.0)
    ben_raw = _make_hs(n_ben, 2.0, 0.0, 0.0)
    ctrl_raw = _make_hs(n_ctrl, 0.0, 0.0, 0.0)

    tox_fit = _make_hs(n_tox * 2, 2.0, 2.0, 3.0)
    ben_fit = _make_hs(n_ben * 2, 2.0, 0.0, 3.0)
    gids_t = torch.repeat_interleave(torch.arange(n_tox), 2)
    gids_b = torch.repeat_interleave(torch.arange(n_ben), 2)

    tox_val = _make_hs(n_tox, 2.0, 2.0, 4.0)
    ben_val = _make_hs(n_ben, 2.0, 0.0, 4.0)
    val_g = torch.arange(n_tox)

    # Token-0 refusal prior stronger on over-refused & toxic, low on controls
    logits_t = torch.zeros(n_tox, 16)
    logits_t[:, 1] = 3.0
    logits_b = torch.zeros(n_ben, 16)
    logits_b[:, 1] = 3.5
    logits_c = torch.zeros(n_ctrl, 16)
    logits_c[:, 5] = 4.0
    t0_t = extract_token0_prior_features(logits_t, [1])
    t0_b = extract_token0_prior_features(logits_b, [1])
    t0_c = extract_token0_prior_features(logits_c, [1])
    t0_tf = t0_t.repeat_interleave(2, dim=0)
    t0_bf = t0_b.repeat_interleave(2, dim=0)

    layers = select_probe_layers(n_blocks, k=5)
    assert layers[-1] == n_blocks and len(extract_multilayer_features(tox_raw, layers)[0]) == len(layers) * dim

    preds, rep = fit_wsp_closed_form_safety_router(
        tox_raw_hs=tox_raw,
        tox_fit_hs=tox_fit,
        tox_fit_gids=gids_t,
        ben_raw_hs=ben_raw,
        ben_fit_hs=ben_fit,
        ben_fit_gids=gids_b,
        ctrl_raw_hs=ctrl_raw,
        n_blocks=n_blocks,
        tox_val_hs=tox_val,
        tox_val_gids=val_g,
        ben_val_hs=ben_val,
        ben_val_gids=val_g,
        tox_raw_t0=t0_t,
        tox_fit_t0=t0_tf,
        tox_val_t0=t0_t,
        ben_raw_t0=t0_b,
        ben_fit_t0=t0_bf,
        ben_val_t0=t0_b,
        ctrl_raw_t0=t0_c,
        k_cv=5,
    )
    assert "wsp_crossfit_auto" in preds and len(preds) == 9
    auto_fn = preds["wsp_crossfit_auto"]
    assert float(auto_fn(ben_raw, t0_b).float().mean()) >= 0.9
    assert float(auto_fn(tox_val, t0_t).float().mean()) <= 0.1
    assert float(auto_fn(ctrl_raw, t0_c).float().mean()) == 0.0
    assert rep["winner"]["oof_ctrl_fpr"] == 0.0
