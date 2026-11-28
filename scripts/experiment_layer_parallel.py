"""EXPERIMENTAL - does the phase-1 flow map predict which adjacent layers can run in parallel?

For every adjacent pair (L-1, L) we run the model with both blocks fed the same input x_{L-2}:
    x_L' = x_{L-2} + delta_{L-1}(x_{L-2}) + delta_L(x_{L-2})
(hooks only; weights untouched) and measure the NLL increase on held-out text.

Predictors, computed on a *different* text (map text != eval text):
  wsi_attn  : |share| of block L's attention read that comes from block L-1's writes  (phase-1 map)
  wsi_all   : same, attention + MLP reads of block L
  cos_skip  : 1 - cos(x_{L-2}, x_{L-1})   (trivial baseline: "how much does block L-1 change the residual")
  rel_norm  : ||delta_{L-1}|| / ||x_{L-2}||  (second trivial baseline)
Score = Spearman(predictor, damage) over the 23 pairs. The map is only useful if it beats the trivial baselines.

Run (CPU):  CUDA_VISIBLE_DEVICES="" python scripts/experiment_layer_parallel.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_layer_flow import dependency, layer_readers, writes_fp32  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402


def spearman(a: list[float], b: list[float]) -> float:
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def nll(model, ids: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(input_ids=ids.reshape(1, -1), use_cache=False).logits[0, :-1].float()
    return F.cross_entropy(logits, ids[1:]).item()


def parallel_pair_nll(model, blocks, L: int, ids: torch.Tensor) -> float:
    """Run blocks L-1 and L on the same input x_{L-2}; add both deltas."""
    st = {}

    def pre_prev(mod, args, kwargs):
        st["x_in_prev"] = args[0]

    def pre_cur(mod, args, kwargs):
        st["x_true"] = args[0]                       # x_{L-1}, what block L would normally read
        return (st["x_in_prev"],) + args[1:], kwargs

    def post_cur(mod, args, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t + (st["x_true"] - st["x_in_prev"])     # add back delta_{L-1}
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t

    hs = [blocks[L - 1].register_forward_pre_hook(pre_prev, with_kwargs=True),
          blocks[L].register_forward_pre_hook(pre_cur, with_kwargs=True),
          blocks[L].register_forward_hook(post_cur)]
    try:
        return nll(model, ids)
    finally:
        for h in hs:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen35-0.8b-full")
    ap.add_argument("--out", default="benchmarks/layer_flow")
    ap.add_argument("--map-tokens", type=int, default=512)
    ap.add_argument("--eval-tokens", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    _, blocks = find_blocks(model)
    n = len(blocks)
    readers = layer_readers(model)

    gtr = pd.read_parquet(ROOT / "work/gsm8k/train.parquet")
    gte = pd.read_parquet(ROOT / "work/gsm8k/test.parquet")
    wik = pd.read_parquet(ROOT / "work/wikitext/test.parquet")

    def join(rows, cols, k):
        text = "\n\n".join(" ".join(str(r[c]) for c in cols) for _, r in rows.iterrows())
        return tok(text, return_tensors="pt").input_ids[0][:k]

    map_ids = join(gtr.head(12), ["question", "answer"], args.map_tokens)
    wtext = "".join(t for t in wik["text"].tolist()[:400] if len(t.strip()) > 40)
    evals = {
        "gsm8k_test": join(gte.head(12), ["question", "answer"], args.eval_tokens),
        "wikitext": tok(wtext, return_tensors="pt").input_ids[0][: args.eval_tokens],
    }

    # ---- predictors from the map text
    d, names = writes_fp32(model, map_ids)
    dep = dependency(d, readers)
    x = d.cumsum(0)  # row 0 = E = x_{-1}; row 2+2l = x_l
    xs = [x[0]] + [x[2 + 2 * l] for l in range(n)]  # xs[k] = input of block k
    pred = {"wsi_attn": [], "wsi_all": [], "cos_skip": [], "rel_norm": []}
    for L in range(1, n):
        prev = [1 + 2 * (L - 1), 2 + 2 * (L - 1)]
        a = dep["D_attn"][L, prev].abs().sum().item()
        m = dep["D_mlp"][L, prev].abs().sum().item()
        pred["wsi_attn"].append(a)
        pred["wsi_all"].append(a + m)
        x_in, x_out = xs[L - 1][1:], xs[L][1:]
        pred["cos_skip"].append(1 - F.cosine_similarity(x_in, x_out, dim=-1).mean().item())
        pred["rel_norm"].append(((x_out - x_in).norm(dim=-1) / x_in.norm(dim=-1)).mean().item())
    print(f"map done {time.time() - t0:.0f}s", flush=True)

    # ---- real damage on eval texts
    base = {k: nll(model, v) for k, v in evals.items()}
    rows = []
    for L in range(1, n):
        r = {"pair": f"{L - 1}|{L}"}
        for k, v in evals.items():
            r[f"dnll_{k}"] = round(parallel_pair_nll(model, blocks, L, v) - base[k], 4)
        r["dnll_mean"] = round(sum(r[f"dnll_{k}"] for k in evals) / len(evals), 4)
        for p, vals in pred.items():
            r[p] = round(vals[L - 1], 4)
        rows.append(r)
        print(r, flush=True)

    dmg = [r["dnll_mean"] for r in rows]
    score = {p: round(spearman(vals, dmg), 3) for p, vals in pred.items()}
    score_mid = {p: round(spearman(vals[3:20], dmg[3:20]), 3) for p, vals in pred.items()}  # pairs 3|4 .. 19|20
    best = sorted(rows, key=lambda r: r["dnll_mean"])[:6]
    result = {
        "model": args.model, "base_nll": {k: round(v, 4) for k, v in base.items()},
        "spearman_vs_damage_all_pairs": score,
        "spearman_vs_damage_middle_pairs": score_mid,
        "least_damaging_pairs": [(r["pair"], r["dnll_mean"]) for r in best],
        "pairs": rows, "seconds": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_parallel_pairs_cpu.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
