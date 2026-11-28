"""EXPERIMENTAL - better predictors for "which adjacent layer pair can run in parallel".

Reuses the measured damage from `results_parallel_pairs_cpu.json` (experiment_layer_parallel.py) - no new ablations.
All predictors are computed on the *map text* (GSM8K train), damage was measured on GSM8K test + wikitext.

Predictors for pair (L-1, L):
  cos_skip       baseline from v1: 1 - cos(x_{L-2}, x_{L-1})
  wsi_attn       v1 map: |direct share| of block L's attention read coming from block L-1
  wsi_attn_clean same, after projecting the top-k dominant residual directions out of every write
  local_err      what block L writes when fed x_{L-2} instead of x_{L-1}, relative error:
                   ||delta_L(x_{L-2}) - delta_L(x_{L-1})|| / ||x_L||          (one single-block call per pair)
  taylor         first-order loss change: mean_t  grad_{x_L} NLL . e_L         (one backward for all pairs)
  taylor_abs     mean_t |grad_{x_L} NLL . e_L|
  grad_err       ||grad_{x_L} NLL|| * ||e_L|| averaged over tokens (sensitivity x error size)
Cost of the whole set: 1 forward + 1 backward + 23 single-block calls (vs 23 full forwards per eval text).

Run (CPU):  CUDA_VISIBLE_DEVICES="" python scripts/experiment_layer_parallel_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_layer_flow import dependency, layer_readers  # noqa: E402
from experiment_layer_parallel import spearman  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402


def capture(model, blocks, ids):
    """One forward+backward. Returns block inputs/outputs, the pre-norm states (for writes), call kwargs and grads."""
    st = {"in": {}, "kw": {}, "out": {}, "mid": {}}
    hs = []
    for l, blk in enumerate(blocks):
        def pre(mod, args, kwargs, l=l):
            st["in"][l] = args[0]
            st["kw"][l] = (args[1:], kwargs)

        def post(mod, args, out, l=l):
            t = out[0] if isinstance(out, tuple) else out
            t.retain_grad()
            st["out"][l] = t

        hs.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
        hs.append(blk.register_forward_hook(post))
        hs.append(blk.post_attention_layernorm.register_forward_pre_hook(
            lambda m, a, l=l: st["mid"].__setitem__(l, a[0].detach()[0].float())))
    emb = model.get_input_embeddings()(ids.reshape(1, -1)).detach().requires_grad_(True)
    try:
        logits = model(inputs_embeds=emb, use_cache=False).logits[0, :-1].float()
        loss = F.cross_entropy(logits, ids[1:])
        loss.backward()
    finally:
        for h in hs:
            h.remove()
    return st, loss.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen35-0.8b-full")
    ap.add_argument("--out", default="benchmarks/layer_flow")
    ap.add_argument("--map-tokens", type=int, default=512)
    ap.add_argument("--drop-dirs", type=int, default=1, help="dominant directions removed for wsi_attn_clean")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    v1 = json.loads((ROOT / args.out / "results_parallel_pairs_cpu.json").read_text())
    damage = [r["dnll_mean"] for r in v1["pairs"]]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, blocks = find_blocks(model)
    n = len(blocks)
    readers = layer_readers(model)

    gtr = pd.read_parquet(ROOT / "work/gsm8k/train.parquet")
    text = "\n\n".join(f"{r['question']} {r['answer']}" for _, r in gtr.head(12).iterrows())
    ids = tok(text, return_tensors="pt").input_ids[0][: args.map_tokens]

    st, base_loss = capture(model, blocks, ids)
    xin = [st["in"][l].detach()[0].float() for l in range(n)]           # x_{l-1}
    xout = [st["out"][l].detach()[0].float() for l in range(n)]         # x_l
    grad = [st["out"][l].grad[0].float() for l in range(n)]             # dNLL/dx_l (mean over tokens)
    print(f"forward+backward {time.time() - t0:.0f}s, loss {base_loss:.4f}", flush=True)

    # ---- writes (E, A0, M0, ...) and the dominant directions
    d = [xin[0]]
    for l in range(n):
        d += [st["mid"][l] - xin[l], xout[l] - st["mid"][l]]
    d = torch.stack(d)
    X = torch.cat(xout, 0)[:, :]
    _, S, Vh = torch.linalg.svd(X - 0 * X.mean(0), full_matrices=False)
    energy = (S ** 2 / (S ** 2).sum())[:5]
    top = Vh[: args.drop_dirs]                                            # (k, H)
    d_clean = d - (d @ top.t()) @ top
    dep = dependency(d, readers)
    dep_c = dependency(d_clean, readers)

    pred = {k: [] for k in ("cos_skip", "wsi_attn", "wsi_attn_clean", "local_err", "taylor", "taylor_abs", "grad_err")}
    with torch.no_grad():
        for L in range(1, n):
            prev = [1 + 2 * (L - 1), 2 + 2 * (L - 1)]
            pred["wsi_attn"].append(dep["D_attn"][L, prev].abs().sum().item())
            pred["wsi_attn_clean"].append(dep_c["D_attn"][L, prev].abs().sum().item())
            pred["cos_skip"].append(1 - F.cosine_similarity(xin[L - 1][1:], xin[L][1:], dim=-1).mean().item())

            a_rest, kw = st["kw"][L]
            x_skip = st["in"][L - 1].detach()
            o = blocks[L](x_skip, *a_rest, **kw)
            o = (o[0] if isinstance(o, tuple) else o)[0].float()
            delta_skip = o - xin[L - 1]                                   # delta_L(x_{L-2})
            delta_true = xout[L] - xin[L]                                 # delta_L(x_{L-1})
            e = (delta_skip - delta_true)[1:]
            g = grad[L][1:] * (len(ids) - 1)                              # per-token NLL gradient
            pred["local_err"].append((e.norm(dim=-1) / xout[L][1:].norm(dim=-1)).mean().item())
            dot = (g * e).sum(-1)
            pred["taylor"].append(dot.mean().item())
            pred["taylor_abs"].append(dot.abs().mean().item())
            pred["grad_err"].append((g.norm(dim=-1) * e.norm(dim=-1)).mean().item())

    score_all = {p: round(spearman(v, damage), 3) for p, v in pred.items()}
    score_mid = {p: round(spearman(v[3:20], damage[3:20]), 3) for p, v in pred.items()}
    top_true = {r["pair"] for r in sorted(v1["pairs"], key=lambda r: r["dnll_mean"])[:6]}
    hits = {}
    for p, v in pred.items():
        order = sorted(range(len(v)), key=lambda i: v[i])[:6]
        hits[p] = len({v1["pairs"][i]["pair"] for i in order} & top_true)
    rows = [{"pair": v1["pairs"][i]["pair"], "damage": damage[i], **{p: round(v[i], 4) for p, v in pred.items()}}
            for i in range(len(damage))]
    result = {
        "dominant_dir_energy_top5": [round(x, 4) for x in energy.tolist()],
        "drop_dirs": args.drop_dirs,
        "spearman_all_pairs": score_all,
        "spearman_middle_pairs": score_mid,
        "top6_least_damaging_recovered": hits,
        "pairs": rows,
        "seconds": round(time.time() - t0, 1),
    }
    (ROOT / args.out / f"results_parallel_pairs_v2_drop{args.drop_dirs}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
