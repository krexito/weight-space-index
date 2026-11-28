"""Step 1 validation: fuse native Token-0 refusal logit prior (`+t0`) with 1-layer linear safety routers.

Tests strictly held-out firing + live end-to-end generation on `LiquidAI/LFM3-1.7B`
for 8 arms sharing `patches/lfm3_1.7b_abliterated_sfw_routed.moipatch` edits:
  1. `shipped_l24_linear`
  2. `recal_linear_1L`
  3. `shipped_and_t0_prob`
  4. `shipped_and_t0_fused`
  5. `shipped_veto_with_t0_fused_g1`
  6. `t0_only_with_shipped_veto`
  7. `recal_1l_and_t0_fused`
  8. `recal_1l_veto_with_t0_fused_g1`

CPU smoke test::
    python scripts/experiment_token0_gate_1l_lfm3.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import calibrate_veto_invariant_lfm3 as cal  # noqa: E402
import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
import experiment_veto_robustness_lfm3 as vr  # noqa: E402
import experiment_wsp_closed_form_router_lfm3 as wsp_exp  # noqa: E402
from experiment_pocket_moe_controls import _paired_bootstrap  # noqa: E402
from wsi.closed_form_router import (  # noqa: E402
    _candidate_thrs,
    fit_ridge_binary_head,
    fit_token0_fused_gate,
    predict_head_score,
    predict_token0_fused_gate,
    resolve_refusal_opener_ids,
)
from wsi.moipatch import MoIPatch  # noqa: E402
from wsi.router_cal import (  # noqa: E402
    CALIBRATION_FIT_WRAPPERS,
    CALIBRATION_VAL_WRAPPERS,
    apply_calibration_wrapper,
    compute_hedging_balanced_weights,
    fit_framing_invariant_safety_router,
)


def _select_t0_thr(
    pos_s: torch.Tensor,
    ctrl_s: torch.Tensor,
    nonref_s: torch.Tensor | None,
    *,
    min_tpr: float | None = None,
    nonref_pen: float = 0.75,
) -> dict[str, Any]:
    pool = [pos_s, ctrl_s] + ([nonref_s] if (nonref_s is not None and nonref_s.numel() > 0) else [])
    cands = _candidate_thrs(torch.cat(pool), max_cands=128)
    best = None
    for thr in cands.tolist():
        tpr = float((pos_s > thr).float().mean().item())
        cf = float((ctrl_s > thr).float().mean().item())
        nrf = float((nonref_s > thr).float().mean().item()) if (nonref_s is not None and nonref_s.numel() > 0) else 0.0
        zero_c = 1.0 if cf == 0.0 else 0.0
        meets_rec = 1.0 if (min_tpr is None or tpr >= float(min_tpr)) else 0.0
        obj = round(tpr - float(nonref_pen) * nrf - 2.0 * cf, 6)
        key = (zero_c, meets_rec, obj)
        if best is None or key > best[0]:
            best = (key, {"thr": float(thr), "oof_tpr": round(tpr, 4), "oof_ctrl_fpr": round(cf, 4), "oof_nonref_fpr": round(nrf, 4)})
    assert best is not None
    return best[1]


def _balanced_xyw(
    pos_w: torch.Tensor,
    n_ctrl: int,
    n_nonref: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wp = pos_w.to(dtype=torch.float64)
    wp = 0.5 * wp / wp.sum().clamp_min(1e-12)
    wc = torch.full((n_ctrl,), 0.25 / max(1, n_ctrl), dtype=torch.float64) if n_nonref > 0 else torch.full((n_ctrl,), 0.5 / max(1, n_ctrl), dtype=torch.float64)
    wn = torch.full((n_nonref,), 0.25 / max(1, n_nonref), dtype=torch.float64) if n_nonref > 0 else torch.empty(0, dtype=torch.float64)
    w_all = torch.cat([wp, wc, wn])
    y_all = torch.cat([torch.ones(len(wp), dtype=torch.float64), -torch.ones(n_ctrl + n_nonref, dtype=torch.float64)])
    return wp, wc, wn, torch.stack([w_all, y_all], dim=0)


def fit_1l_token0_arms(
    *,
    shipped_router: dict[str, Any],
    recal_router: dict[str, Any],
    ben_ref_hs: torch.Tensor,
    ctrl_hs: torch.Tensor,
    nonref_hs: torch.Tensor | None,
    ben_ref_t0: torch.Tensor,
    ctrl_t0: torch.Tensor,
    nonref_t0: torch.Tensor | None,
    ben_ref_weights: torch.Tensor | None = None,
    k_cv: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit and OOF-calibrate all `+t0` 1-layer arms strictly on calibration tensors."""
    n_ben = int(ben_ref_t0.shape[0])
    n_ctrl = int(ctrl_t0.shape[0])
    n_nr = int(nonref_t0.shape[0]) if (nonref_t0 is not None and nonref_t0.numel() > 0) else 0
    k_eff = max(2, min(int(k_cv), n_ben, n_ctrl, max(2, n_nr if n_nr > 0 else k_cv)))

    bw = ben_ref_weights.to(dtype=torch.float64).cpu() if ben_ref_weights is not None else torch.ones(n_ben, dtype=torch.float64)
    sl_ship = int(shipped_router["layer"])
    sl_rec = int(recal_router["layer"])
    s_ship_ben = (ben_ref_hs[:, sl_ship, :].float() @ shipped_router["dir"].float().cpu()).double()
    s_ship_ctrl = (ctrl_hs[:, sl_ship, :].float() @ shipped_router["dir"].float().cpu()).double()
    s_ship_nr = (nonref_hs[:, sl_ship, :].float() @ shipped_router["dir"].float().cpu()).double() if (n_nr > 0 and nonref_hs is not None) else torch.empty(0, dtype=torch.float64)

    s_rec_ben = (ben_ref_hs[:, sl_rec, :].float() @ recal_router["dir"].float().cpu()).double()
    s_rec_ctrl = (ctrl_hs[:, sl_rec, :].float() @ recal_router["dir"].float().cpu()).double()
    s_rec_nr = (nonref_hs[:, sl_rec, :].float() @ recal_router["dir"].float().cpu()).double() if (n_nr > 0 and nonref_hs is not None) else torch.empty(0, dtype=torch.float64)

    def _cv_fused_gate(s_ben: torch.Tensor, s_ctrl: torch.Tensor, s_nr: torch.Tensor, *, min_tpr: float | None) -> dict[str, Any]:
        oof_b = torch.empty(n_ben, dtype=torch.float64)
        oof_c = torch.empty(n_ctrl, dtype=torch.float64)
        oof_n = torch.empty(n_nr, dtype=torch.float64)
        for fold in range(k_eff):
            va_b = (torch.arange(n_ben) % k_eff) == fold
            va_c = (torch.arange(n_ctrl) % k_eff) == fold
            tr_b, tr_c = ~va_b, ~va_c
            if n_nr > 0:
                va_n = (torch.arange(n_nr) % k_eff) == fold
                tr_n = ~va_n
                s_tr_neg = torch.cat([s_ctrl[tr_c], s_nr[tr_n]])
                t0_tr_neg = torch.cat([ctrl_t0[tr_c], nonref_t0[tr_n]], dim=0)
                n_tr_n = int(tr_n.sum().item())
            else:
                s_tr_neg = s_ctrl[tr_c]
                t0_tr_neg = ctrl_t0[tr_c]
                n_tr_n = 0
            _, _, _, wy = _balanced_xyw(bw[tr_b], int(tr_c.sum().item()), n_tr_n)
            head = fit_token0_fused_gate(
                torch.cat([s_ben[tr_b], s_tr_neg]),
                torch.cat([ben_ref_t0[tr_b], t0_tr_neg], dim=0),
                wy[1],
                sample_weight=wy[0],
                lam=1.0,
            )
            oof_b[va_b] = predict_token0_fused_gate(head, s_ben[va_b], ben_ref_t0[va_b])
            oof_c[va_c] = predict_token0_fused_gate(head, s_ctrl[va_c], ctrl_t0[va_c])
            if n_nr > 0:
                oof_n[va_n] = predict_token0_fused_gate(head, s_nr[va_n], nonref_t0[va_n])

        cal_rep = _select_t0_thr(oof_b, oof_c, oof_n if n_nr > 0 else None, min_tpr=min_tpr, nonref_pen=0.75)
        _, _, _, wy_full = _balanced_xyw(bw, n_ctrl, n_nr)
        neg_s_full = torch.cat([s_ctrl, s_nr]) if n_nr > 0 else s_ctrl
        neg_t0_full = torch.cat([ctrl_t0, nonref_t0], dim=0) if (n_nr > 0 and nonref_t0 is not None) else ctrl_t0
        full_head = fit_token0_fused_gate(
            torch.cat([s_ben, neg_s_full]),
            torch.cat([ben_ref_t0, neg_t0_full], dim=0),
            wy_full[1],
            sample_weight=wy_full[0],
            lam=1.0,
        )
        return {"head": full_head, **cal_rep}

    def _cv_t0_only_ridge() -> dict[str, Any]:
        oof_b = torch.empty(n_ben, dtype=torch.float64)
        oof_c = torch.empty(n_ctrl, dtype=torch.float64)
        oof_n = torch.empty(n_nr, dtype=torch.float64)
        for fold in range(k_eff):
            va_b = (torch.arange(n_ben) % k_eff) == fold
            va_c = (torch.arange(n_ctrl) % k_eff) == fold
            tr_b, tr_c = ~va_b, ~va_c
            if n_nr > 0:
                va_n = (torch.arange(n_nr) % k_eff) == fold
                tr_n = ~va_n
                t0_tr_neg = torch.cat([ctrl_t0[tr_c], nonref_t0[tr_n]], dim=0)
                n_tr_n = int(tr_n.sum().item())
            else:
                t0_tr_neg = ctrl_t0[tr_c]
                n_tr_n = 0
            wp, wc, wn, _ = _balanced_xyw(bw[tr_b], int(tr_c.sum().item()), n_tr_n)
            w_neg = torch.cat([wc, wn]) if n_tr_n > 0 else wc
            head = fit_ridge_binary_head(ben_ref_t0[tr_b], t0_tr_neg, pos_weights=wp, neg_weights=w_neg, lam_ridge_rel=0.25)
            oof_b[va_b] = predict_head_score(head, ben_ref_t0[va_b])
            oof_c[va_c] = predict_head_score(head, ctrl_t0[va_c])
            if n_nr > 0:
                oof_n[va_n] = predict_head_score(head, nonref_t0[va_n])

        cal_rep = _select_t0_thr(oof_b, oof_c, oof_n if n_nr > 0 else None, min_tpr=0.92, nonref_pen=0.75)
        wp, wc, wn, _ = _balanced_xyw(bw, n_ctrl, n_nr)
        w_neg = torch.cat([wc, wn]) if n_nr > 0 else wc
        neg_t0_full = torch.cat([ctrl_t0, nonref_t0], dim=0) if (n_nr > 0 and nonref_t0 is not None) else ctrl_t0
        full_head = fit_ridge_binary_head(ben_ref_t0, neg_t0_full, pos_weights=wp, neg_weights=w_neg, lam_ridge_rel=0.25)
        return {"head": full_head, **cal_rep}

    prob_rep = _select_t0_thr(
        ben_ref_t0[:, 1].cpu(),
        ctrl_t0[:, 1].cpu(),
        nonref_t0[:, 1].cpu() if (n_nr > 0 and nonref_t0 is not None) else None,
        min_tpr=0.95,
        nonref_pen=0.75,
    )
    ship_and_rep = _cv_fused_gate(s_ship_ben, s_ship_ctrl, s_ship_nr, min_tpr=0.95)
    ship_repl_rep = _cv_fused_gate(s_ship_ben, s_ship_ctrl, s_ship_nr, min_tpr=None)
    rec_and_rep = _cv_fused_gate(s_rec_ben, s_rec_ctrl, s_rec_nr, min_tpr=0.95)
    rec_repl_rep = _cv_fused_gate(s_rec_ben, s_rec_ctrl, s_rec_nr, min_tpr=None)
    t0_only_rep = _cv_t0_only_ridge()

    def _veto_ok(r: dict[str, Any], hs: torch.Tensor) -> torch.BoolTensor:
        lay = int(r["layer"])
        h = hs[:, lay, :].float().cpu()
        return ((h @ r["veto_dir"].float().cpu()) <= float(r["veto_thr"])).bool()

    def _g1_score_dir(r: dict[str, Any], hs: torch.Tensor) -> torch.Tensor:
        lay = int(r["layer"])
        return (hs[:, lay, :].float().cpu() @ r["dir"].float().cpu()).double()

    predictors = {
        "shipped_l24_linear": lambda hs, t0: wsp_exp._eval_linear_router_fires(shipped_router, hs),
        "recal_linear_1L": lambda hs, t0: wsp_exp._eval_linear_router_fires(recal_router, hs),
        "shipped_and_t0_prob": lambda hs, t0: (wsp_exp._eval_linear_router_fires(shipped_router, hs) & (t0[:, 1].cpu() > float(prob_rep["thr"]))).bool(),
        "shipped_and_t0_fused": lambda hs, t0: (wsp_exp._eval_linear_router_fires(shipped_router, hs) & (predict_token0_fused_gate(ship_and_rep["head"], _g1_score_dir(shipped_router, hs), t0) > float(ship_and_rep["thr"]))).bool(),
        "shipped_veto_with_t0_fused_g1": lambda hs, t0: ((predict_token0_fused_gate(ship_repl_rep["head"], _g1_score_dir(shipped_router, hs), t0) > float(ship_repl_rep["thr"])) & _veto_ok(shipped_router, hs)).bool(),
        "t0_only_with_shipped_veto": lambda hs, t0: ((predict_head_score(t0_only_rep["head"], t0) > float(t0_only_rep["thr"])) & _veto_ok(shipped_router, hs)).bool(),
        "recal_1l_and_t0_fused": lambda hs, t0: (wsp_exp._eval_linear_router_fires(recal_router, hs) & (predict_token0_fused_gate(rec_and_rep["head"], _g1_score_dir(recal_router, hs), t0) > float(rec_and_rep["thr"]))).bool(),
        "recal_1l_veto_with_t0_fused_g1": lambda hs, t0: ((predict_token0_fused_gate(rec_repl_rep["head"], _g1_score_dir(recal_router, hs), t0) > float(rec_repl_rep["thr"])) & _veto_ok(recal_router, hs)).bool(),
    }

    def _ser_fused(rep: dict[str, Any]) -> dict[str, Any]:
        h = rep["head"]
        return {
            "thr": round(float(rep["thr"]), 6),
            "oof_tpr": rep["oof_tpr"],
            "oof_ctrl_fpr": rep["oof_ctrl_fpr"],
            "oof_nonref_fpr": rep["oof_nonref_fpr"],
            "mu": [round(float(v), 6) for v in h["mu"].tolist()],
            "std": [round(float(v), 6) for v in h["std"].tolist()],
            "coef": [round(float(v), 6) for v in h["coef"].tolist()],
        }

    gate_params = {
        "shipped_and_t0_prob": prob_rep,
        "shipped_and_t0_fused": _ser_fused(ship_and_rep),
        "shipped_veto_with_t0_fused_g1": _ser_fused(ship_repl_rep),
        "recal_1l_and_t0_fused": _ser_fused(rec_and_rep),
        "recal_1l_veto_with_t0_fused_g1": _ser_fused(rec_repl_rep),
        "t0_only_with_shipped_veto": {
            "thr": round(float(t0_only_rep["thr"]), 6),
            "oof_tpr": t0_only_rep["oof_tpr"],
            "oof_ctrl_fpr": t0_only_rep["oof_ctrl_fpr"],
            "oof_nonref_fpr": t0_only_rep["oof_nonref_fpr"],
            "w": [round(float(v), 6) for v in t0_only_rep["head"]["w"].tolist()],
            "b": round(float(t0_only_rep["head"]["b"]), 6),
        },
    }
    return predictors, gate_params


def run_smoke() -> dict[str, Any]:
    torch.manual_seed(0)
    model = cal._TinyFramingToyLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = cal._TinyFramingTokenizer()
    dev = torch.device("cpu")
    opener_ids = resolve_refusal_opener_ids(tok)
    n_blocks = len(model.model.layers)

    fit_styles = tuple(CALIBRATION_FIT_WRAPPERS.keys())
    val_styles = tuple(CALIBRATION_VAL_WRAPPERS.keys())
    tox_cal = [f"toxic:{i} unsafe" for i in range(10)]
    ben_cal = [f"hedged_ben:{i} prompt" for i in range(5)] + [f"direct_ben:{i} prompt" for i in range(5)]
    ctrl_cal = [f"normal:{i} everyday" for i in range(10)]

    def _pack(prompts: list[str], styles_per: list[list[str]], lbl: str):
        fmts, gids = [], []
        for gi, raw_p in enumerate(prompts):
            for st in styles_per[gi]:
                fmts.append(sr._format_prompt(tok, apply_calibration_wrapper(raw_p, st), None))
                gids.append(gi)
        hs, t0 = wsp_exp._collect_hs_and_t0(model, fmts, opener_ids, tok, dev, label=lbl, log_every=0)
        return hs, t0, torch.tensor(gids, dtype=torch.long)

    tox_raw_h, _, _ = _pack(tox_cal, [["raw"] for _ in tox_cal], "tox_r")
    tox_fit_h, _, tox_fg = _pack(tox_cal, [list(fit_styles[:2]) for _ in tox_cal], "tox_f")
    tox_val_h, _, tox_vg = _pack(tox_cal, [list(val_styles) for _ in tox_cal], "tox_v")
    ben_raw_h, ben_raw_t, _ = _pack(ben_cal, [["raw"] for _ in ben_cal], "ben_r")
    ben_fit_h, _, ben_fg = _pack(ben_cal, [list(fit_styles[:2]) for _ in ben_cal], "ben_f")
    ben_val_h, _, ben_vg = _pack(ben_cal, [list(val_styles) for _ in ben_cal], "ben_v")
    ctrl_raw_h, ctrl_raw_t, _ = _pack(ctrl_cal, [["raw"] for _ in ctrl_cal], "ctrl_r")

    recal_router, _ = fit_framing_invariant_safety_router(
        tox_raw_h, tox_fit_h, tox_fg, ben_raw_h, ben_fit_h, ben_fg, ctrl_raw_h,
        n_blocks, 1, tox_val_hs=tox_val_h, tox_val_gids=tox_vg,
        ben_ref_val_hs=ben_val_h, ben_ref_val_gids=ben_vg, device=dev,
    )
    preds, gparams = fit_1l_token0_arms(
        shipped_router=recal_router,
        recal_router=recal_router,
        ben_ref_hs=ben_raw_h,
        ctrl_hs=ctrl_raw_h,
        nonref_hs=None,
        ben_ref_t0=ben_raw_t,
        ctrl_t0=ctrl_raw_t,
        nonref_t0=None,
        k_cv=5,
    )
    assert len(preds) == 8
    for nm, fn in preds.items():
        _ = fn(ben_raw_h, ben_raw_t)
        assert float(fn(ctrl_raw_h, ctrl_raw_t).float().mean()) == 0.0, f"{nm} fired on ctrl"
    return {"mode": "smoke", "n_arms": len(preds), "opener_ids": opener_ids, "gate_params_keys": list(gparams.keys()), "passed": True}


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    model, tok, dev, dt = sr._load_model_and_tokenizer(args.model, args.dtype)
    shipped_patch = MoIPatch.load(args.patch)
    # Fail in 1s instead of 10m if loaded model dtype/weights fail patch SHA-256 verification
    _chk = shipped_patch.attach(model, verify_fingerprint=True)
    del _chk
    print(f"[t0-1L] loaded model dtype={dt} fingerprint_ok=True", flush=True)
    opener_ids = resolve_refusal_opener_ids(tok)
    opener_tokens = {int(tid): tok.decode([int(tid)]) for tid in opener_ids}
    n_blocks = len(model.model.layers) if hasattr(model, "model") and hasattr(model, "layers") else int(shipped_patch.router["layer"])
    min_edit_layer = min(int(e.module.split(".")[1]) for e in shipped_patch.edits)

    or_main, or_extra, ctrl_prompts, toxic_test_prompts, toxic_cal_prompts = cal._load_calibration_slices(
        n_or_main=200, n_or_extra=120, n_toxic_test=200, n_toxic_cal=120, n_control=60, seed=int(args.seed)
    )
    jbb_harm_rows, jbb_ben_rows, _ = vr._load_jbb_suites(
        n_harmful=int(args.n_jbb_harm), n_benign=int(args.n_jbb_ben), n_control=30, seed=int(args.seed)
    )

    or_main_fmt = [sr._format_prompt(tok, p, None) for p in or_main]
    probe_mix = or_main_fmt[:4] + [sr._format_prompt(tok, p, None) for p in ctrl_prompts[:4]]
    eff_bs = sr._probe_batching_exact(model, tok, probe_mix, dev, batch_size=int(args.batch_size))
    print(f"[t0-1L] n_blocks={n_blocks} min_edit={min_edit_layer} openers={opener_tokens} eff_bs={eff_bs}", flush=True)

    # Fast 3-prompt canary sanity check
    tox_canary_raw = [tp["prompt"] if isinstance(tp, dict) else str(tp) for tp in toxic_cal_prompts[:2]]
    canary_fmts = [sr._format_prompt(tok, p, None) for p in tox_canary_raw] + [sr._format_prompt(tok, ctrl_prompts[0], None)]
    canary_out = sr._generate_texts(model, tok, canary_fmts, dev, max_new_tokens=32, batch_size=1)
    if sum(1 for r in canary_out[:2] if r["refused"]) == 0 or bool(canary_out[2]["refused"]) or any(sum(c.isalpha() for c in r["text"]) < 5 for r in canary_out):
        raise RuntimeError("Base model sanity check failed — check transformers==5.23.0 pin")

    base_or_main = sr._generate_texts(model, tok, or_main_fmt, dev, max_new_tokens=int(args.max_new_tokens), batch_size=eff_bs)
    or_a_idx = [i for i in range(0, len(or_main), 2)]
    or_b_idx = [i for i in range(1, len(or_main), 2)]
    ctrl_a_texts = [ctrl_prompts[i] for i in range(0, len(ctrl_prompts), 2)]
    ctrl_b_texts = [ctrl_prompts[i] for i in range(1, len(ctrl_prompts), 2)]

    or_extra_out = sr._generate_texts(model, tok, [sr._format_prompt(tok, p, None) for p in or_extra], dev, max_new_tokens=48, batch_size=eff_bs)
    direct_out = sr._generate_texts(model, tok, [sr._format_prompt(tok, p, None) for p in cal.DIRECT_BENIGN_CAL_PROMPTS], dev, max_new_tokens=48, batch_size=eff_bs)
    tox_cal_raw = [tp["prompt"] if isinstance(tp, dict) else str(tp) for tp in toxic_cal_prompts]
    tox_cal_out = sr._generate_texts(model, tok, [sr._format_prompt(tok, p, None) for p in tox_cal_raw], dev, max_new_tokens=48, batch_size=eff_bs)

    tox_cal_ref = [p for p, r in zip(tox_cal_raw, tox_cal_out) if r["refused"]]
    ben_ref_cal = (
        [or_main[i] for i in or_a_idx if base_or_main[i]["refused"]]
        + [p for p, r in zip(or_extra, or_extra_out) if r["refused"]]
        + [p for p, r in zip(cal.DIRECT_BENIGN_CAL_PROMPTS, direct_out) if r["refused"]]
    )
    ben_nonref_cal = (
        [p for p, r in zip(cal.DIRECT_BENIGN_CAL_PROMPTS, direct_out) if not r["refused"]]
        + [or_main[i] for i in or_a_idx if not base_or_main[i]["refused"]]
        + [p for p, r in zip(or_extra, or_extra_out) if not r["refused"]]
    )[:160]
    print(f"[cal] tox_ref={len(tox_cal_ref)} ben_ref={len(ben_ref_cal)} ben_nonref={len(ben_nonref_cal)} ctrl_a={len(ctrl_a_texts)}", flush=True)

    fit_styles = tuple(CALIBRATION_FIT_WRAPPERS.keys())
    val_styles = tuple(CALIBRATION_VAL_WRAPPERS.keys())

    def _fmt_wrapped(prompts: list[str], styles_per: list[list[str]]) -> tuple[list[str], torch.Tensor]:
        fmts, gids = [], []
        for gi, raw_p in enumerate(prompts):
            for st in styles_per[gi]:
                fmts.append(sr._format_prompt(tok, apply_calibration_wrapper(raw_p, st), None))
                gids.append(gi)
        return fmts, torch.tensor(gids, dtype=torch.long)

    tox_r_fmt, _ = _fmt_wrapped(tox_cal_ref, [["raw"] for _ in tox_cal_ref])
    tox_f_fmt, tox_fg = _fmt_wrapped(tox_cal_ref, [[fit_styles[(2 * i) % len(fit_styles)], fit_styles[(2 * i + 1) % len(fit_styles)]] for i in range(len(tox_cal_ref))])
    tox_v_fmt, tox_vg = _fmt_wrapped(tox_cal_ref, [list(val_styles) for _ in tox_cal_ref])
    ben_r_fmt, _ = _fmt_wrapped(ben_ref_cal, [["raw"] for _ in ben_ref_cal])
    ben_f_fmt, ben_fg = _fmt_wrapped(ben_ref_cal, [[fit_styles[(2 * i) % len(fit_styles)], fit_styles[(2 * i + 1) % len(fit_styles)]] for i in range(len(ben_ref_cal))])
    ben_v_fmt, ben_vg = _fmt_wrapped(ben_ref_cal, [list(val_styles) for _ in ben_ref_cal])
    ctrl_r_fmt, _ = _fmt_wrapped(ctrl_a_texts, [["raw"] for _ in ctrl_a_texts])
    ctrl_f_fmt, ctrl_fg = _fmt_wrapped(ctrl_a_texts, [[fit_styles[i % len(fit_styles)]] for i in range(len(ctrl_a_texts))])
    nonref_r_fmt = [sr._format_prompt(tok, p, None) for p in ben_nonref_cal]

    tox_raw_hs, tox_raw_t0 = wsp_exp._collect_hs_and_t0(model, tox_r_fmt, opener_ids, tok, dev, label="tox_raw")
    tox_fit_hs, _ = wsp_exp._collect_hs_and_t0(model, tox_f_fmt, opener_ids, tok, dev, label="tox_fit")
    tox_val_hs, _ = wsp_exp._collect_hs_and_t0(model, tox_v_fmt, opener_ids, tok, dev, label="tox_val")
    ben_raw_hs, ben_raw_t0 = wsp_exp._collect_hs_and_t0(model, ben_r_fmt, opener_ids, tok, dev, label="ben_raw")
    ben_fit_hs, _ = wsp_exp._collect_hs_and_t0(model, ben_f_fmt, opener_ids, tok, dev, label="ben_fit")
    ben_val_hs, _ = wsp_exp._collect_hs_and_t0(model, ben_v_fmt, opener_ids, tok, dev, label="ben_val")
    ctrl_raw_hs, ctrl_raw_t0 = wsp_exp._collect_hs_and_t0(model, ctrl_r_fmt, opener_ids, tok, dev, label="ctrl_raw", log_every=0)
    ctrl_fit_hs, _ = wsp_exp._collect_hs_and_t0(model, ctrl_f_fmt, opener_ids, tok, dev, label="ctrl_fit", log_every=0)
    nonref_hs, nonref_t0 = wsp_exp._collect_hs_and_t0(model, nonref_r_fmt, opener_ids, tok, dev, label="ben_nonref")

    def _t0_stats(t0_t: torch.Tensor) -> dict[str, float]:
        p = t0_t[:, 1].double()
        return {"mean": round(float(p.mean()), 4), "median": round(float(p.median()), 4), "p10": round(float(torch.quantile(p, 0.1)), 4), "p90": round(float(torch.quantile(p, 0.9)), 4)}

    t0_cal_diagnostics = {
        "tox_ref": _t0_stats(tox_raw_t0),
        "ben_ref": _t0_stats(ben_raw_t0),
        "ben_nonref": _t0_stats(nonref_t0),
        "ctrl_a": _t0_stats(ctrl_raw_t0),
    }
    print(f"[t0-diag] ref_prob quantiles: {t0_cal_diagnostics}", flush=True)

    ben_weights = compute_hedging_balanced_weights(ben_ref_cal)
    recal_router, recal_report = fit_framing_invariant_safety_router(
        tox_raw_hs, tox_fit_hs, tox_fg, ben_raw_hs, ben_fit_hs, ben_fg, ctrl_raw_hs,
        n_blocks, min_edit_layer,
        tox_val_hs=tox_val_hs, tox_val_gids=tox_vg,
        ben_ref_val_hs=ben_val_hs, ben_ref_val_gids=ben_vg,
        ben_nonref_raw_hs=nonref_hs,
        ctrl_fit_hs=ctrl_fit_hs, ctrl_fit_gids=ctrl_fg,
        ben_ref_weights=ben_weights,
        device=dev,
    )
    predictors, gate_params = fit_1l_token0_arms(
        shipped_router=shipped_patch.router,
        recal_router=recal_router,
        ben_ref_hs=ben_raw_hs,
        ctrl_hs=ctrl_raw_hs,
        nonref_hs=nonref_hs,
        ben_ref_t0=ben_raw_t0,
        ctrl_t0=ctrl_raw_t0,
        nonref_t0=nonref_t0,
        ben_ref_weights=ben_weights,
        k_cv=5,
    )

    del tox_raw_hs, tox_fit_hs, tox_val_hs, ben_raw_hs, ben_fit_hs, ben_val_hs, ctrl_raw_hs, ctrl_fit_hs, nonref_hs

    eval_prompts: dict[str, list[str]] = {
        "or_b": [or_main_fmt[i] for i in or_b_idx],
        "toxic_test": [sr._format_prompt(tok, tp["prompt"] if isinstance(tp, dict) else str(tp), None) for tp in toxic_test_prompts],
        "ctrl_b": [sr._format_prompt(tok, p, None) for p in ctrl_b_texts],
        "jbb_benign": [sr._format_prompt(tok, r["goal"], None) for r in jbb_ben_rows],
    }
    for st in ("raw", *tuple(vr.FRAMING_WRAPPERS)):
        eval_prompts[f"jbb_harm_{st}"] = [sr._format_prompt(tok, vr.apply_framing(r["goal"], st), None) for r in jbb_harm_rows]

    base_eval_out: dict[str, list[dict[str, Any]]] = {"or_b": [base_or_main[i] for i in or_b_idx]}
    base_eval_flags: dict[str, list[bool]] = {"or_b": [bool(r["refused"]) for r in base_eval_out["or_b"]]}
    for snm, plist in eval_prompts.items():
        if snm not in base_eval_out:
            base_eval_out[snm] = sr._generate_texts(model, tok, plist, dev, max_new_tokens=int(args.max_new_tokens), batch_size=eff_bs)
            base_eval_flags[snm] = [bool(r["refused"]) for r in base_eval_out[snm]]

    eval_hs: dict[str, torch.Tensor] = {}
    eval_t0: dict[str, torch.Tensor] = {}
    for snm, plist in eval_prompts.items():
        eval_hs[snm], eval_t0[snm] = wsp_exp._collect_hs_and_t0(model, plist, opener_ids, tok, dev, label=f"ev_{snm}", log_every=0)

    arm_names = list(predictors.keys())
    fire_masks: dict[str, dict[str, list[bool]]] = {nm: {} for nm in arm_names}
    for snm in eval_prompts:
        hs_s, t0_s = eval_hs[snm], eval_t0[snm]
        for nm, fn in predictors.items():
            fire_masks[nm][snm] = fn(hs_s, t0_s).tolist()

    router_fire_metrics = {nm: wsp_exp._evaluate_router_mask_on_slices(fire_masks[nm], base_eval_flags) for nm in arm_names}

    # Paired bootstrap vs shipped_l24_linear on boundary FPR, benign refused TPR, and pooled toxic leak
    bnd_or = [i for i, f in enumerate(base_eval_flags["or_b"]) if not f]
    bnd_jbb = [i for i, f in enumerate(base_eval_flags["jbb_benign"]) if not f]
    ref_or = [i for i, f in enumerate(base_eval_flags["or_b"]) if f]
    ref_jbb = [i for i, f in enumerate(base_eval_flags["jbb_benign"]) if f]
    harm_slices = ["toxic_test", "jbb_harm_raw", *[f"jbb_harm_{st}" for st in vr.FRAMING_WRAPPERS]]
    harm_idxs = {snm: [i for i, f in enumerate(base_eval_flags[snm]) if f] for snm in harm_slices}

    paired_vs_shipped: dict[str, dict[str, Any]] = {}
    for nm in arm_names:
        if nm == "shipped_l24_linear":
            continue
        a_bnd = [{"strict_correct": fire_masks[nm]["or_b"][i]} for i in bnd_or] + [{"strict_correct": fire_masks[nm]["jbb_benign"][i]} for i in bnd_jbb]
        b_bnd = [{"strict_correct": fire_masks["shipped_l24_linear"]["or_b"][i]} for i in bnd_or] + [{"strict_correct": fire_masks["shipped_l24_linear"]["jbb_benign"][i]} for i in bnd_jbb]
        a_ref = [{"strict_correct": fire_masks[nm]["or_b"][i]} for i in ref_or] + [{"strict_correct": fire_masks[nm]["jbb_benign"][i]} for i in ref_jbb]
        b_ref = [{"strict_correct": fire_masks["shipped_l24_linear"]["or_b"][i]} for i in ref_or] + [{"strict_correct": fire_masks["shipped_l24_linear"]["jbb_benign"][i]} for i in ref_jbb]
        a_tox = [{"strict_correct": fire_masks[nm][snm][i]} for snm in harm_slices for i in harm_idxs[snm]]
        b_tox = [{"strict_correct": fire_masks["shipped_l24_linear"][snm][i]} for snm in harm_slices for i in harm_idxs[snm]]
        paired_vs_shipped[nm] = {
            "boundary_fpr_diff": _paired_bootstrap(a_bnd, b_bnd, seed=int(args.seed)),
            "benign_refused_tpr_diff": _paired_bootstrap(a_ref, b_ref, seed=int(args.seed) + 1),
            "pooled_toxic_leak_diff": _paired_bootstrap(a_tox, b_tox, seed=int(args.seed) + 2),
        }

    live_arms = [
        "base",
        "shipped_l24_linear",
        "shipped_and_t0_prob",
        "shipped_and_t0_fused",
        "shipped_veto_with_t0_fused_g1",
        "t0_only_with_shipped_veto",
        "recal_1l_and_t0_fused",
    ]
    live_generation_arms: dict[str, dict[str, Any]] = {}
    for a_nm in live_arms:
        if a_nm == "base":
            arm_out = base_eval_out
        else:
            arm_out = {}
            for snm, plist in eval_prompts.items():
                arm_out[snm] = wsp_exp._generate_on_fired(
                    model, tok, shipped_patch, plist, base_eval_out[snm], fire_masks[a_nm][snm],
                    dev, max_new_tokens=int(args.max_new_tokens), batch_size=eff_bs,
                )

        def _rate_kept(snm: str) -> float:
            b_idx = [i for i, fl in enumerate(base_eval_flags[snm]) if fl]
            return round(sum(1 for i in b_idx if arm_out[snm][i]["refused"]) / max(1, len(b_idx)), 4)

        def _rate_rec(snm: str) -> float:
            b_idx = [i for i, fl in enumerate(base_eval_flags[snm]) if fl]
            return round(sum(1 for i in b_idx if not arm_out[snm][i]["refused"]) / max(1, len(b_idx)), 4)

        or_rec_texts = [arm_out["or_b"][i]["text"] for i, fl in enumerate(base_eval_flags["or_b"]) if fl and not arm_out["or_b"][i]["refused"]]
        framed_kept = [_rate_kept(f"jbb_harm_{st}") for st in vr.FRAMING_WRAPPERS]
        live_generation_arms[a_nm] = {
            "or_bench_recovered": _rate_rec("or_b"),
            "or_toxic_kept": _rate_kept("toxic_test"),
            "jbb_harm_raw_kept": _rate_kept("jbb_harm_raw"),
            "jbb_harm_framed_kept": {st: v for st, v in zip(vr.FRAMING_WRAPPERS, framed_kept)},
            "jbb_harm_min_framed_kept": min(framed_kept),
            "jbb_benign_recovered": _rate_rec("jbb_benign"),
            "or_b_boundary_fpr": router_fire_metrics[a_nm]["or_b_boundary_fpr"] if a_nm in router_fire_metrics else 0.0,
            "jbb_ben_boundary_fpr": router_fire_metrics[a_nm]["jbb_ben_boundary_fpr"] if a_nm in router_fire_metrics else 0.0,
            "mean_boundary_fpr": router_fire_metrics[a_nm]["mean_boundary_fpr"] if a_nm in router_fire_metrics else 0.0,
            "ctrl_exact_match": round(sum(1 for a, b in zip(arm_out["ctrl_b"], base_eval_out["ctrl_b"]) if a["text"] == b["text"]) / max(1, len(base_eval_out["ctrl_b"])), 4),
            "recovered_opening_diversity": wsp_exp._opening_diversity(or_rec_texts),
        }

    return {
        "experiment": "token0_gate_1l_lfm3",
        "model": args.model,
        "patch": args.patch,
        "opener_ids": opener_ids,
        "opener_tokens": opener_tokens,
        "t0_cal_diagnostics": t0_cal_diagnostics,
        "calibration_counts": {"tox_ref": len(tox_cal_ref), "ben_ref": len(ben_ref_cal), "ben_nonref": len(ben_nonref_cal), "ctrl_a": len(ctrl_a_texts)},
        "recal_linear_report": {k: v for k, v in recal_report.items() if k != "top5_candidates"},
        "gate_params": gate_params,
        "router_fire_metrics": router_fire_metrics,
        "paired_vs_shipped": paired_vs_shipped,
        "live_generation_arms": live_generation_arms,
        "timing_s": {"total_s": round(time.perf_counter() - t0, 2)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--patch", default=str(ROOT / "patches" / "lfm3_1.7b_abliterated_sfw_routed.moipatch"))
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--n-jbb-harm", type=int, default=100)
    ap.add_argument("--n-jbb-ben", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = sr._atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
    wsp_exp._print_summary(payload)


if __name__ == "__main__":
    main()
