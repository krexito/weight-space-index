"""EXPERIMENTAL: does the layer-wise Hessian error of an edit predict its damage, and fix the 1-pass screen?

For a rank-1 edit W' = W - alpha u u^T W on a module with inputs x_t, the GPTQ/OBS layer-wise error is
    tr(dW H dW^T) = alpha^2 * sum_t (u . W x_t)^2 ,   H = sum_t x_t x_t^T
i.e. alpha^2 u^T G u with G the Gram of the module's OUTPUTS on calibration text. No generation, one forward pass.
Pool: the 24 candidates of experiment_refusal_select.py (LFM3-1.7B), rebuilt exactly (same base generations, same
diff-of-means directions; reproduction checked against the saved suppression / control KL).
Questions (fixed before looking):
  1. damage: Spearman(Hessian error, measured always-on control KL) over the 24 candidates
  2. screen: rank z(suppression) - z(Hessian error) vs the exam score, against z(suppression) - z(control KL) and
     suppression alone (Spearman with bootstrap CIs; also on the 20 grid candidates only)
Calibration text for G: "ctrl" = the 20 dev GSM8K prompts + base answers; "mix" = ctrl + the fit OR-Bench prompts the
base answered + answers. Absolute error and error relative to the outputs' energy (sum_t ||W x_t||^2) are reported.

  python scripts/experiment_hessian_screen.py --select R_SELECT.json --out R.json [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import experiment_refusal_controls as rc  # noqa: E402
import experiment_refusal_select as rs  # noqa: E402
import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
from wsi.moipatch import MoIPatch  # noqa: E402


@torch.no_grad()
def output_grams(model, mods: dict[str, torch.nn.Module], seqs: list[list[int]], dev) -> tuple[dict, dict, int]:
    G = {n: torch.zeros(m.weight.shape[0], m.weight.shape[0], dtype=torch.float64, device=dev) for n, m in mods.items()}
    E = {n: 0.0 for n in mods}
    hooks = []
    for n, m in mods.items():
        def h(_m, _i, o, n=n):
            y = (o[0] if isinstance(o, tuple) else o)[0].double()
            G[n] += y.T @ y
            E[n] += float((y * y).sum())
        hooks.append(m.register_forward_hook(h))
    T = 0
    try:
        for s in seqs:
            model(input_ids=torch.tensor([s], device=dev))
            T += len(s)
    finally:
        for x in hooks:
            x.remove()
    return G, E, T


def zs(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--patch", default=str(HERE.parent / "patches/lfm3_1.7b_abliterated_sfw_routed.moipatch"))
    ap.add_argument("--select", required=True, help="results JSON of experiment_refusal_select.py (pool + exam)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    sel = json.loads(Path(a.select).read_text())
    C = sel["candidates"]
    model, tok, dev, _ = sr._load_model_and_tokenizer(a.model, "bf16" if torch.cuda.is_available() else "fp32")
    n_layers, writers = sr._discover_all_writers(model)
    data = rs.load_splits()
    data = {k: data[k] for k in ("fit_or", "dev_or", "dev_ctrl")}
    if a.limit:
        data = {k: v[: max(a.limit, 6) if k == "fit_or" else a.limit] for k, v in data.items()}
    fmt = {k: [sr._format_prompt(tok, p) for p in v] for k, v in data.items()}
    bs = 32
    base = {k: sr._generate_texts(model, tok, fmt[k], dev, max_new_tokens=64, batch_size=bs) for k in fmt}
    lab = torch.tensor([r["refused"] for r in base["fit_or"]])
    H = rs.last_hidden(model, tok, fmt["fit_or"], dev)
    dirs = {}
    for L in rs.U_LAYERS:
        d = H[lab, L + 1].mean(0) - H[~lab, L + 1].mean(0) if not (lab.all() or (~lab).all()) else \
            torch.randn(H.shape[-1], generator=torch.Generator().manual_seed(L))
        dirs[L] = d / d.norm()
    shipped = MoIPatch.load(a.patch)
    patches = {}
    for nm, c in C.items():
        if nm == "shipped":
            patches[nm] = shipped
        elif nm.startswith("random"):
            patches[nm] = rc.random_patch(shipped, int(nm[-1]))
        else:
            patches[nm] = rs.make_patch(writers, c["modules"], dirs[c["u_layer"]], c["alpha"], shipped.router, nm)

    # reproduction check: suppression and control KL as in experiment_refusal_select.py
    dev_ref = [p for p, r in zip(fmt["dev_or"], base["dev_or"]) if r["refused"]]
    dev_tok = [r["gen_ids"][0] for r in base["dev_or"] if r["refused"] and r["gen_ids"]]
    lp_ref0 = rs.first_token_logprobs(model, tok, dev_ref, dev)
    lp_ctl0 = rs.first_token_logprobs(model, tok, fmt["dev_ctrl"], dev)
    repro = {}
    for nm, p in patches.items():
        with p.attach(model, verify_fingerprint=False) as h:
            h.on()
            lr_, lc_ = rs.first_token_logprobs(model, tok, dev_ref, dev), rs.first_token_logprobs(model, tok, fmt["dev_ctrl"], dev)
        i = torch.arange(len(dev_tok))
        repro[nm] = {"suppression": float((lp_ref0[i, dev_tok] - lr_[i, dev_tok]).mean()) if dev_tok else 0.0,
                     "ctrl_kl": float((lp_ctl0.exp() * (lp_ctl0 - lc_)).sum(-1).mean())}
    print(f"rebuilt {len(patches)} candidates; fit refused {int(lab.sum())}/{len(lab)} (saved run "
          f"{sel.get('fit_refused')}/{sel.get('fit_n')}) ({time.time() - t0:.0f}s)", flush=True)

    # output Grams of every edited module on two calibration sets
    names = sorted({e.module for p in patches.values() for e in p.edits})
    full = dict(model.named_modules())
    mods = {n: next(m for fn, m in full.items() if fn == n or fn.endswith("." + n)) for n in names}
    ctrl_seqs = [r["full_ids"] for r in base["dev_ctrl"]]
    ans_seqs = [r["full_ids"] for r in base["fit_or"] if not r["refused"]]
    calib = {"ctrl": ctrl_seqs, "mix": ctrl_seqs + ans_seqs}
    hess = {}
    for cn, seqs in calib.items():
        G, E, T = output_grams(model, mods, seqs, dev)
        hess[cn] = {}
        for nm, p in patches.items():
            err = sum(e.alpha ** 2 * float(e.u.double().to(dev) @ G[e.module] @ e.u.double().to(dev)) for e in p.edits) / T
            energy = sum(E[e.module] for e in p.edits) / T
            hess[cn][nm] = {"abs": err, "rel": err / energy}
        del G
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    names_c = list(C)
    grid = [n for n in names_c if n.startswith("L")]
    exam = {n: C[n]["exam"]["score"] for n in names_c}
    kl = {n: C[n]["ctrl_kl"] for n in names_c}
    supp = {n: C[n]["suppression"] for n in names_c}
    res = {"model": a.model, "n_calib_tokens": {k: sum(map(len, v)) for k, v in calib.items()},
           "reproduction": {"spearman_supp_saved_vs_rebuilt": rs.spearman_ci([supp[n] for n in names_c],
                                                                              [repro[n]["suppression"] for n in names_c]),
                            "spearman_kl_saved_vs_rebuilt": rs.spearman_ci([kl[n] for n in names_c],
                                                                            [repro[n]["ctrl_kl"] for n in names_c]),
                            "fit_refused_rebuilt": int(lab.sum()), "fit_refused_saved": sel.get("fit_refused")},
           "hessian": hess, "q1_damage": {}, "q2_screen": {}}
    for cn in calib:
        for kind in ("abs", "rel"):
            hv = {n: hess[cn][n][kind] for n in names_c}
            res["q1_damage"][f"{cn}_{kind}"] = rs.spearman_ci([hv[n] for n in names_c], [kl[n] for n in names_c])
            for pool, ns in (("all24", names_c), ("grid20", grid)):
                s_h = zs([supp[n] for n in ns]) - zs([hv[n] for n in ns])
                res["q2_screen"][f"{pool}|supp-hess_{cn}_{kind}"] = rs.spearman_ci(s_h, [exam[n] for n in ns])
    for pool, ns in (("all24", names_c), ("grid20", grid)):
        res["q2_screen"][f"{pool}|supp-kl"] = rs.spearman_ci(zs([supp[n] for n in ns]) - zs([kl[n] for n in ns]),
                                                             [exam[n] for n in ns])
        res["q2_screen"][f"{pool}|supp_only"] = rs.spearman_ci([supp[n] for n in ns], [exam[n] for n in ns])
        res["q2_screen"][f"{pool}|escape(saved)"] = rs.spearman_ci([C[n]["escape"]["score"] for n in ns],
                                                                   [exam[n] for n in ns])
    res["per_candidate"] = {n: {"exam": exam[n], "suppression": supp[n], "ctrl_kl": kl[n],
                                "hess_ctrl_abs": hess["ctrl"][n]["abs"], "hess_mix_rel": hess["mix"][n]["rel"],
                                "rebuilt": repro[n]} for n in names_c}
    res["elapsed_s"] = round(time.time() - t0)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("reproduction", "q1_damage", "q2_screen")}, indent=1))


if __name__ == "__main__":
    main()
