"""EXPERIMENTAL: does the 1-forward-pass screen rank loop patches the way the real MATH-500 exam does?

The MATH-500 blind search picked L10_n256_a12 (test 10.7%) although our GSM8K blind patch reaches 20.7% and MIT p50c
28.7% on the same exam (experiment_edit_form_math500.py). Its candidates came from the top-d' layers and its finalists
from wsi.tracer.screen_patch_candidates (pareto rank, 6 held-out looping elicit sequences). Here the screen scores six
patches whose always-on MATH-500 test accuracy is known, on the SAME screen sequences (screen_fast_grid.screen_sequences
of the MATH-500 run) and on a larger set, and reports the Spearman rank correlation of every proxy with the exam.

  p50c (28.7)  blind_n16_a12 (20.7)  blind_n64_a6 (16.0)  math_L10_n256_a12 (10.7)  blind_n64_a12 (10.7)
  random_L10_n256_a12 (5.3)

Patches are exact weight swaps (the same tensors the exam checkpoints held), so nothing depends on MoIPatch.

  python scripts/experiment_screen_known_patches.py --math-work M --blind-work BW --edit-dir benchmarks/exp8_math
         --model Qwen3.5-0.8B --out R.json [--n-seqs 6 24]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

for _m in ("fla", "causal_conv1d"):
    sys.modules.setdefault(_m, None)

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

EXAM = {"p50c": 0.2867, "blind_n16_a12": 0.2067, "blind_n64_a6": 0.16, "math_L10_n256_a12": 0.1067,
        "blind_n64_a12": 0.1067, "random_L10_n256_a12": 0.0533}
PROXIES = ("target_suppression", "pareto_score", "pareto_kl", "delta_healthy_nll", "healthy_kl")


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return round(float(spearmanr(x, y).correlation), 3)


class Swap:
    """Minimal patch: attach(model) -> handle with on()/off() that swaps whole weight tensors."""

    def __init__(self, new: dict):
        self.new = new  # param name -> edited tensor (cpu)

    def attach(self, model):
        params = dict(model.named_parameters())
        new = self.new

        class H:
            def on(self):
                self.old = {k: params[k].data.clone() for k in new}
                for k, v in new.items():
                    params[k].data.copy_(v.to(params[k].device, params[k].dtype))

            def off(self):
                for k, v in self.old.items():
                    params[k].data.copy_(v)

        return H()


def down_name(params: dict, layer: int) -> str:
    (k,) = [k for k in params if k.endswith(f"layers.{layer}.mlp.down_proj.weight") and "mtp" not in k and "visual" not in k]
    return k


def blind_edit(params, diag, z, layers, n, alpha, kind):
    import torch
    import discover_blind as blind

    out = {}
    for li in layers:
        k = down_name(params, li)
        W = params[k].data.detach().cpu().double()
        uu = torch.from_numpy(blind.direction(diag, z, li, kind))
        m = torch.zeros(W.shape[1], dtype=torch.float64)
        m[blind.columns(diag, li, n)] = 1.0
        out[k] = (W - alpha * torch.outer(uu, m * (uu @ W))).to(params[k].dtype)
    return out


def p50c_edit(params, edit_dir: Path):
    import torch

    man = json.loads((edit_dir / "p50c_manifest.json").read_text())
    z = np.load(edit_dir / "p50c_edit.npz")
    out = {}
    for i, t in enumerate(man["tensors"]):
        (k,) = [k for k in params if k.endswith(t["name"].split("model.language_model.", 1)[-1]) and "mtp" not in k]
        base = params[k].data.detach().cpu()
        bits = base.view(torch.int16).flatten().clone()
        bits[torch.from_numpy(z[f"t{i}_idx"].astype(np.int64))] = torch.from_numpy(z[f"t{i}_val"].astype(np.uint16).view(np.int16))
        out[k] = bits.view(torch.bfloat16).reshape(base.shape)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-work", required=True, help="discover_math500 run (elicit outputs, diagnosis, blind_directions)")
    ap.add_argument("--blind-work", required=True, help="discover_blind GSM8K run (diagnosis, blind_directions)")
    ap.add_argument("--edit-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, nargs="+", default=[6, 24])
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    import screen_fast_grid as sfg
    from wsi.tracer import screen_patch_candidates

    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    params = dict(model.named_parameters())
    mw, bw = Path(a.math_work), Path(a.blind_work)
    dm, zm = json.loads((mw / "diagnosis_blind.json").read_text()), np.load(mw / "blind_directions.npz")
    db, zb = json.loads((bw / "diagnosis_blind.json").read_text()), np.load(bw / "blind_directions.npz")
    cands = {
        "p50c": Swap(p50c_edit(params, Path(a.edit_dir))),
        "blind_n16_a12": Swap(blind_edit(params, db, zb, [13, 14], 16, 12.0, "wsi")),
        "blind_n64_a6": Swap(blind_edit(params, db, zb, [13, 14], 64, 6.0, "wsi")),
        "blind_n64_a12": Swap(blind_edit(params, db, zb, [13, 14], 64, 12.0, "wsi")),
        "math_L10_n256_a12": Swap(blind_edit(params, dm, zm, [10], 256, 12.0, "wsi")),
        "random_L10_n256_a12": Swap(blind_edit(params, dm, zm, [10], 256, 12.0, "random")),
    }
    res = {"exam_accuracy": EXAM, "runs": {}}
    for n in a.n_seqs:
        sfg.N_SCREEN_SEQS = n
        healthy, target, spans, ids = sfg.screen_sequences(mw)
        t0 = time.time()
        rows = screen_patch_candidates(model, cands, healthy_seqs=healthy, target_seqs=target, target_spans=spans,
                                       max_nll_delta=sfg.MAX_NLL_DELTA, rank_by=sfg.RANK_BY)
        names = [r["name"] for r in rows]
        by = {r["name"]: r for r in rows}
        exam = [EXAM[k] for k in EXAM]
        corr = {p: spearman([by[k][p] for k in EXAM], exam) for p in PROXIES}
        res["runs"][str(n)] = {"seq_ids": ids, "n_seqs": len(healthy), "screen_s": round(time.time() - t0, 1),
                               "screen_order": names, "rows": rows, "spearman_vs_exam": corr}
        print(f"== {len(healthy)} sequences {ids}  ({time.time() - t0:.0f}s)")
        for r in rows:
            print(f"  {r['name']:20s} exam {EXAM[r['name']]:.3f}  supp {r['target_suppression']:+.4f}  dNLL {r['delta_healthy_nll']:+.4f}"
                  f"  KL {r['healthy_kl']:.4f}  pareto {r['pareto_score']:+.3f}  guard {r['passes_nll_guard']}")
        print("  spearman vs exam:", corr, flush=True)
    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1))
    Path(a.out + ".tmp").replace(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
