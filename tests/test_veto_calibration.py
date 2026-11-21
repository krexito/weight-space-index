"""CPU unit tests for framing-invariant `veto_dir` calibration (`wsi.router_cal` & `calibrate_veto_invariant_lfm3`)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_veto_invariant_lfm3 as cal  # noqa: E402
import experiment_veto_robustness_lfm3 as vr  # noqa: E402
from wsi.router_cal import (  # noqa: E402
    CALIBRATION_FIT_WRAPPERS,
    CALIBRATION_VAL_WRAPPERS,
    apply_calibration_wrapper,
    compute_hedging_balanced_weights,
    evaluate_framing_invariance,
    fit_framing_invariant_safety_router,
    fit_invariant_direction,
)


def test_primal_dual_woodbury_exact_equivalence() -> None:
    torch.manual_seed(11)
    d_dim = 40
    pos_h = torch.randn(14, d_dim) + 0.5
    neg_h = torch.randn(16, d_dim) - 0.5
    deltas = torch.randn(18, d_dim)
    protect = torch.randn(2, d_dim)
    pos_w = torch.rand(14) + 0.2
    neg_w = torch.rand(16) + 0.2

    w_primal = fit_invariant_direction(
        pos_h,
        neg_h,
        nuisance_deltas=deltas,
        protect_dirs=protect,
        lam_ridge=2.5,
        lam_nuisance=12.0,
        lam_protect=3.0,
        pos_weights=pos_w,
        neg_weights=neg_w,
        hard_orthogonalize_protect=False,
        use_dual=False,
    )
    w_dual = fit_invariant_direction(
        pos_h,
        neg_h,
        nuisance_deltas=deltas,
        protect_dirs=protect,
        lam_ridge=2.5,
        lam_nuisance=12.0,
        lam_protect=3.0,
        pos_weights=pos_w,
        neg_weights=neg_w,
        hard_orthogonalize_protect=False,
        use_dual=True,
    )
    assert w_primal is not None and w_dual is not None
    assert torch.allclose(w_primal, w_dual, atol=1e-5, rtol=1e-5)

    # Check exact orientation toward weighted positive vs negative mean
    pw = pos_w.double() / pos_w.double().sum()
    nw = neg_w.double() / neg_w.double().sum()
    diff_mu = (pw.unsqueeze(1) * pos_h.double()).sum(dim=0) - (nw.unsqueeze(1) * neg_h.double()).sum(dim=0)
    assert float((diff_mu @ w_primal.double()).item()) > 0.0


def test_invariant_direction_eliminates_planted_framing_shortcut() -> None:
    torch.manual_seed(42)
    d_dim = 32
    n_pos, n_neg = 48, 48

    # Coord 0: true harmful intent (+1.5 on toxic, 0 on benign)
    # Coord 1: spurious academic framing feature (+4.0 on raw OR benigns, 0 on raw toxic, +4.0 when framed!)
    tox_raw = torch.randn(n_pos, d_dim) * 0.12
    tox_raw[:, 0] += 1.5

    ben_raw = torch.randn(n_neg, d_dim) * 0.12
    ben_raw[:, 1] += 4.0

    # Framing adds +4.0 on coord 1 (plus tiny noise on other nuisance directions)
    f_vec = torch.zeros(d_dim)
    f_vec[1] = 4.0
    deltas = f_vec.unsqueeze(0).repeat(n_pos + n_neg, 1) + torch.randn(n_pos + n_neg, d_dim) * 0.03

    w_naive = fit_invariant_direction(tox_raw, ben_raw, lam_ridge=1.0, lam_nuisance=0.0)
    w_inv = fit_invariant_direction(tox_raw, ben_raw, nuisance_deltas=deltas, lam_ridge=1.0, lam_nuisance=100.0)
    assert w_naive is not None and w_inv is not None

    mid_naive = 0.5 * (float((tox_raw @ w_naive).mean()) + float((ben_raw @ w_naive).mean()))
    mid_inv = 0.5 * (float((tox_raw @ w_inv).mean()) + float((ben_raw @ w_inv).mean()))

    # Unseen framed toxic (perturbed framing vector) + unseen un-hedged OOD benign
    unseen_frame = f_vec + 0.05 * torch.tensor([0.0] + [1.0 if i == 2 else 0.0 for i in range(1, d_dim)])
    tox_framed = tox_raw + unseen_frame.unsqueeze(0)
    ood_ben = torch.randn(n_neg, d_dim) * 0.12  # lacks coord 1!

    naive_framed_tox_acc = float(((tox_framed @ w_naive) > mid_naive).float().mean().item())
    naive_ood_ben_acc = float(((ood_ben @ w_naive) <= mid_naive).float().mean().item())
    assert naive_framed_tox_acc < 0.25 and naive_ood_ben_acc < 0.25

    inv_raw_tox_acc = float(((tox_raw @ w_inv) > mid_inv).float().mean().item())
    inv_framed_tox_acc = float(((tox_framed @ w_inv) > mid_inv).float().mean().item())
    inv_ood_ben_acc = float(((ood_ben @ w_inv) <= mid_inv).float().mean().item())
    assert inv_raw_tox_acc == 1.0
    assert inv_framed_tox_acc == 1.0
    assert inv_ood_ben_acc == 1.0
    assert abs(float((unseen_frame @ w_inv).item())) < 0.05


def test_protect_dir_hard_orthogonality() -> None:
    torch.manual_seed(7)
    d_dim = 24
    pos_h = torch.randn(20, d_dim) + 0.6
    neg_h = torch.randn(20, d_dim) - 0.6
    r_dir = torch.randn(d_dim)
    r_dir = r_dir / r_dir.norm()

    w = fit_invariant_direction(
        pos_h,
        neg_h,
        protect_dirs=r_dir,
        lam_ridge=1.0,
        lam_protect=10.0,
        hard_orthogonalize_protect=True,
    )
    assert w is not None
    assert abs(float((w @ r_dir).item())) < 1e-6


def _word_4grams(text: str) -> set[tuple[str, ...]]:
    cleaned = text.replace("{goal}", " ")
    words = re.findall(r"[a-z0-9]+", cleaned.lower())
    return {tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))}


def test_calibration_wrappers_disjoint_from_jbb_eval() -> None:
    eval_keys = set(vr.FRAMING_WRAPPERS.keys())
    fit_keys = set(CALIBRATION_FIT_WRAPPERS.keys())
    val_keys = set(CALIBRATION_VAL_WRAPPERS.keys())
    assert not (fit_keys & eval_keys)
    assert not (val_keys & eval_keys)
    assert not (fit_keys & val_keys)

    eval_templates = set(vr.FRAMING_WRAPPERS.values())
    cal_templates = set(CALIBRATION_FIT_WRAPPERS.values()) | set(CALIBRATION_VAL_WRAPPERS.values())
    assert not (eval_templates & cal_templates)

    eval_grams = set().union(*(_word_4grams(t) for t in eval_templates))
    cal_grams = set().union(*(_word_4grams(t) for t in cal_templates))
    assert not (eval_grams & cal_grams), f"Overlapping 4-grams: {eval_grams & cal_grams}"
    assert apply_calibration_wrapper("test query", "raw") == "test query"


def test_group_kfold_zero_leakage_and_ablation() -> None:
    torch.manual_seed(123)
    n_tox, n_ben, n_ctrl = 24, 24, 16
    n_blocks, hidden = 2, 24

    # Layer 1: noise only; Layer 2: true intent (dim 1) + planted framing shortcut (dim 2) + Gate 1 refusal (dim 0)
    def _make_hs(n: int, *, ref: float = 0.0, intent: float = 0.0, frame: float = 0.0) -> torch.Tensor:
        hs = torch.randn(n, n_blocks + 1, hidden) * 0.08
        hs[:, 1:, 0] += ref
        hs[:, 2, 1] += intent
        hs[:, 2, 2] += frame
        return hs

    tox_raw = _make_hs(n_tox, ref=3.0, intent=2.0, frame=0.0)
    # Prompt-ID nuisance fingerprint in dims 10..23 that MUST NOT leak across folds
    pid_noise_t = torch.randn(n_tox, hidden - 10) * 0.4
    tox_raw[:, 2, 10:] += pid_noise_t

    ben_raw = _make_hs(n_ben, ref=3.0, intent=0.0, frame=3.0)
    pid_noise_b = torch.randn(n_ben, hidden - 10) * 0.4
    ben_raw[:, 2, 10:] += pid_noise_b

    # Last 4 benigns are un-hedged (no coord 2), tested with hedging-balanced weights
    ben_raw[-4:, 2, 2] = 0.0
    ben_prompts = ["academic research query"] * (n_ben - 4) + ["plain everyday query"] * 4
    ben_w = compute_hedging_balanced_weights(ben_prompts)

    # Fit wrappers perturb coord 2 by +3.0; Val unseen wrappers perturb coord 2 by +3.5 & coord 3 by +0.4
    tox_fit = tox_raw.clone()
    tox_fit[:, 2, 2] += 3.0
    tox_fg = torch.arange(n_tox, dtype=torch.long)

    ben_fit = ben_raw.clone()
    ben_fit[:, 2, 2] += 3.0
    ben_fg = torch.arange(n_ben, dtype=torch.long)

    tox_val = tox_raw.clone()
    tox_val[:, 2, 2] += 3.5
    tox_val[:, 2, 3] += 0.4

    ben_val = ben_raw.clone()
    ben_val[:, 2, 2] += 3.5
    ben_val[:, 2, 3] += 0.4

    ctrl_raw = _make_hs(n_ctrl, ref=0.0, intent=0.0, frame=0.0)

    ev_pre = evaluate_framing_invariance(
        torch.eye(hidden)[2], tox_raw[:, 2, :], {"val": tox_val[:, 2, :]}
    )
    assert ev_pre["rms_shift"] > 3.0

    router, rep = fit_framing_invariant_safety_router(
        tox_raw,
        tox_fit,
        tox_fg,
        ben_raw,
        ben_fit,
        ben_fg,
        ctrl_raw,
        n_blocks,
        min_edit_layer=2,
        tox_val_hs=tox_val,
        tox_val_gids=tox_fg,
        ben_ref_val_hs=ben_val,
        ben_ref_val_gids=ben_fg,
        ctrl_fit_hs=ctrl_raw.clone(),
        ctrl_fit_gids=torch.arange(n_ctrl, dtype=torch.long),
        ben_ref_weights=ben_w,
        lam_frames=(0.0, 10.0, 100.0),
        lam_ridge_rels=(0.05, 0.5, 5.0),
        orth_refusal_options=(False, True),
    )

    assert router["layer"] == 2
    assert rep["lam_frame"] > 0.0
    assert rep["oof_worst_tox_fpr"] <= 0.06
    assert rep["oof_ben_raw_tpr"] >= 0.90

    ab = rep["ablation_no_frame_penalty"]
    assert ab is not None
    assert rep["norm_unseen_frame_shift"] <= ab["norm_unseen_frame_shift"] + 1e-4

    ev_post = evaluate_framing_invariance(
        router["veto_dir"], tox_raw[:, 2, :], {"val": tox_val[:, 2, :]}
    )
    assert ev_post["rms_shift"] < 0.35


def test_calibrate_veto_invariant_smoke_end_to_end(tmp_path) -> None:
    res = cal.run_smoke(out_dir=tmp_path)
    assert res["status"] == "ok" and res["smoke_passed"] is True
    assert res["audit"]["fingerprint_ok"] is True
    assert res["audit"]["restore_exact_ok"] is True

    # Confirm legacy router fails our planted framing shortcut while invariant router passes every gate
    assert res["legacy_jbb_gate"]["min_framed_kept"] < 0.5
    assert res["legacy_jbb_gate"]["ood_benign_recovered"] < 0.5

    assert res["in_distribution"]["gate"]["passed"] is True
    rg = res["jbb_robustness"]["robustness_gate"]
    assert rg["overall_passed"] is True
    assert rg["raw_ood_kept"] == 1.0
    assert rg["min_framed_kept"] == 1.0
    assert rg["ood_benign_recovered"] == 1.0
    assert rg["controls_passed"] is True

    ba = res["before_after_invariance"]
    assert ba["new_shift_to_std"] < ba["old_shift_to_std"]
    dumped = json.dumps(res)
    assert "unsafe synthetic cal prompt" not in dumped
    assert "unsafe synthetic test prompt" not in dumped
    assert "unsafe synthetic jbb goal" not in dumped
