"""EXPERIMENTAL (CPU): does the state at the cut predict the CLEAN per-cut rescue rates of experiment_futures.py?

Rescue from the inside (experiment_rescue_inside.py) asked which cuts the blind patch rescues with one sample per arm
(noisy labels). experiment_futures.py gives 8 samples per arm per cut: per-cut rates patch p, restart r, gain p - r.
Here every saved per-cut feature (rescue_inside per_answer: cut position, loop age, KL base vs patched at the cut,
top-1 flips, projections on the patch direction at L13/L14 before/after the patch, L14 signal over 512 continuation
tokens) plus the sensor over the first W continuation tokens (mean over the 8 futures, per arm) is correlated
(Spearman, 95% bootstrap CI over cuts) with p, r and the gain. No generation, no model.

  python scripts/experiment_futures_inside.py --futures RESULTS_FUTURES.json --inside RESULTS_RESCUE_INSIDE.json \
      --work FUTURES_WORK --out R.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

FEATS = ("cut", "loop_age", "kl_cut", "top1_flip", "proj_L13", "patched_proj_L13", "proj_L14", "patched_proj_L14",
         "dproj_L14", "proj_L14_after")
WINDOWS = (64, 256)


def sensor_means(npz: Path, cuts: list[str], k: int, w: int) -> dict[str, float]:
    z = np.load(npz)
    return {c: float(np.mean([np.nanmean(z[f"{c}|{j}"][:w]) if len(z[f"{c}|{j}"]) else np.nan for j in range(k)]))
            for c in cuts}


def rho_ci(x: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        bs.append(spearmanr(x[i], y[i]).statistic)
    return {"rho": round(float(spearmanr(x, y).statistic), 3),
            "ci95": [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)],
            "n": int(len(x))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--futures", required=True)
    ap.add_argument("--inside", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fu = json.loads(Path(a.futures).read_text())
    ins = json.loads(Path(a.inside).read_text())["per_answer"]
    k = fu["k"]
    rp, rr = fu["arms"]["patch"]["per_cut_rate"], fu["arms"]["restart"]["per_cut_rate"]
    cuts = sorted(set(rp) & set(ins), key=int)
    feats = {f: {c: float(ins[c].get(f, np.nan)) for c in cuts} for f in FEATS}
    for arm in ("patch", "restart"):
        for w in WINDOWS:
            feats[f"sensor_{arm}_{w}"] = sensor_means(Path(a.work) / f"futures_{arm}_sensor.npz", cuts, k, w)
    targets = {"patch_rate": np.array([rp[c] for c in cuts]), "restart_rate": np.array([rr[c] for c in cuts])}
    targets["gain"] = targets["patch_rate"] - targets["restart_rate"]
    res = {"cuts": len(cuts), "k": k, "spearman": {}}
    for f, d in feats.items():
        x = np.array([d[c] for c in cuts])
        res["spearman"][f] = {t: rho_ci(x, y) for t, y in targets.items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{len(cuts)} cuts, {k} samples per arm; Spearman rho [95% CI]")
    print(f"{'feature':22s} {'patch rate':>22s} {'restart rate':>22s} {'gain (p - r)':>22s}")
    for f, v in res["spearman"].items():
        cells = [f"{v[t]['rho']:+.2f} [{v[t]['ci95'][0]:+.2f},{v[t]['ci95'][1]:+.2f}]" for t in targets]
        print(f"{f:22s} " + " ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
