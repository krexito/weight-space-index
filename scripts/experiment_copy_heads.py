"""Copy heads: is there a set of attention heads that drives loop copying but not legitimate copying?

The damage map (experiment_damage_map.py) showed the p50c loop patch is an anti-repetition edit: always on, it hurts
names repeated from the context, the second "step by step", repeated doctest lines. A loop fix that only touches the
copying INSIDE loops would avoid that. This test looks for heads with that profile, on saved base answers (teacher
forcing, no generation):

  1. Each answer becomes one window of up to --max-len tokens around the point where the text detector fires.
     Token sets (wsi.heads.copy_labels): LOOP (inside the loop), COPY (outside it, predictable by copying the context),
     OTHER.
  2. Half A (even problem ids): every head of every head-structured output projection is zeroed in turn; the change of
     NLL is averaged per set. A head that raises LOOP NLL (the model is less sure to continue the loop) while leaving
     COPY and OTHER alone is loop-specific.
  3. Heads ranked on half A by d_loop - d_copy; the top 1/2/4/8 are ablated together on half B (held out), next to a
     random set of heads of the same size and the p50c patch always on (--patch).
  4. Across heads: Spearman correlation of d_loop with d_copy. High = loops run on the same copy machinery as legitimate
     copying (no head-level shortcut); low = separate.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import experiment_self_gate_router as sg  # noqa: E402
from wsi.heads import copy_labels, find_heads, head_ablation, set_means, token_nll  # noqa: E402
from wsi.moipatch import MoIPatch  # noqa: E402


def windows(seqs: list[dict], max_len: int, after: int, warmup: int) -> list[dict]:
    out = []
    for q in seqs:
        body, prompt = q["body"], q["prompt"]
        if q["looping"]:
            anchor = q["fire"] if q["fire"] is not None else q["onset"]
            end = min(len(body), anchor + after)
            if len(prompt) + end <= max_len:  # the whole answer fits: keep the prompt
                pre, start = prompt, 0
            else:  # a window of the answer ending `after` tokens past the anchor (anchor stays inside: after < max_len)
                pre, start = [], max(0, end - max_len)
            ids = pre + body[start:end]
            loop_from = len(pre) + anchor - start
        else:
            ids = (prompt + body)[:max_len]
            pre, loop_from = prompt, None
        lab = copy_labels(ids, score_from=len(pre) + warmup, loop_from=loop_from)
        out.append({"id": q["id"], "ids": ids, "labels": lab[1:]})  # labels aligned with the NLL of ids[1:]
    return out


def pooled(model, wins: list[dict], base: list[np.ndarray], dev, heads=None, info=None, patch_handle=None) -> dict:
    deltas, labs = [], []
    for w, b in zip(wins, base):
        ids = torch.tensor(w["ids"], device=dev)
        if patch_handle is not None:
            patch_handle.on()
        try:
            if heads:
                with head_ablation(model, heads, info):
                    nll = token_nll(model, ids).cpu().numpy()
            else:
                nll = token_nll(model, ids).cpu().numpy()
        finally:
            if patch_handle is not None:
                patch_handle.off()
        deltas.append(nll - b)
        labs.append(w["labels"])
    m = set_means(np.concatenate(deltas), np.concatenate(labs))
    return {k: (round(v, 5) if v is not None else None) for k, v in m.items()}


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="finished run dir with config.json, requests_test.jsonl, outputs_test_base.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", default=None, help="a loop .moipatch to compare with (always on)")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--after", type=int, default=1024, help="tokens kept after the text detector fires")
    ap.add_argument("--warmup", type=int, default=64, help="first tokens of a window are not scored")
    ap.add_argument("--n-a", type=int, default=24, help="half-A answers used to score single heads")
    ap.add_argument("--n-b", type=int, default=48, help="half-B answers used for the held-out combinations")
    ap.add_argument("--kinds", default="attn,linear_attn")
    ap.add_argument("--no-fingerprint", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    seqs = sg.load_sequences(work, set(cfg["eos"]))
    rng = random.Random(a.seed)
    half_a = [q for q in seqs if q["id"] % 2 == 0]
    half_b = [q for q in seqs if q["id"] % 2 == 1]
    rng.shuffle(half_a)
    rng.shuffle(half_b)
    wa = windows(half_a[: a.n_a], a.max_len, a.after, a.warmup)
    wb = windows(half_b[: a.n_b], a.max_len, a.after, a.warmup)
    dev = torch.device(a.device)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), trust_remote_code=True).to(dev).eval()
    kinds = set(a.kinds.split(","))
    info = {n: v for n, v in find_heads(model).items() if v["kind"] in kinds}
    all_heads = [(n, h) for n, v in info.items() for h in range(v["n_heads"])]
    print(f"{len(info)} projections, {len(all_heads)} heads; half A {len(wa)} windows, half B {len(wb)}", flush=True)

    def base_nll(wins):
        return [token_nll(model, torch.tensor(w["ids"], device=dev)).cpu().numpy() for w in wins]

    def counts(wins):
        lab = np.concatenate([w["labels"] for w in wins])
        return {"loop": int((lab == 2).sum()), "copy": int((lab == 1).sum()), "other": int((lab == 0).sum())}

    base_a, base_b = base_nll(wa), base_nll(wb)
    single = []
    for i, (name, h) in enumerate(all_heads):
        m = pooled(model, wa, base_a, dev, {name: [h]}, info)
        single.append({"module": name, "head": h, "kind": info[name]["kind"], **m})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(all_heads)} heads ({time.perf_counter() - t0:.0f} s)", flush=True)
    ok = [s for s in single if s["loop"] is not None and s["copy"] is not None]
    for s in ok:
        s["spec"] = round(s["loop"] - s["copy"], 5)
    ranked = sorted(ok, key=lambda s: -s["spec"])

    res = {"model": cfg.get("model_repo") or a.model, "work": work.name, "max_len": a.max_len,
           "tokens_a": counts(wa), "tokens_b": counts(wb), "n_heads": len(all_heads),
           "spearman_loop_vs_copy": round(spearman([s["loop"] for s in ok], [s["copy"] for s in ok]), 4),
           "top_by_spec": ranked[:20], "top_by_copy": sorted(ok, key=lambda s: -s["copy"])[:10],
           "top_by_loop": sorted(ok, key=lambda s: -s["loop"])[:10], "combos_b": {}}
    for k in (1, 2, 4, 8):
        pick = ranked[:k]
        heads = {}
        for s in pick:
            heads.setdefault(s["module"], []).append(s["head"])
        rnd = rng.sample(all_heads, k)
        rheads = {}
        for n, h in rnd:
            rheads.setdefault(n, []).append(h)
        res["combos_b"][str(k)] = {"heads": [f"{s['module']}#{s['head']}" for s in pick],
                                   "ablate_top": pooled(model, wb, base_b, dev, heads, info),
                                   "ablate_random": pooled(model, wb, base_b, dev, rheads, info)}
    if a.patch:
        handle = MoIPatch.load(a.patch).attach(model, verify_fingerprint=not a.no_fingerprint)
        res["patch_b"] = {"name": Path(a.patch).name, **pooled(model, wb, base_b, dev, patch_handle=handle)}
    res["single_heads"] = single
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))

    print(f"\ntokens half B: {res['tokens_b']} | Spearman(d_loop, d_copy) across heads = {res['spearman_loop_vs_copy']}")
    print("top heads by d_loop - d_copy (half A):")
    for s in ranked[:8]:
        print(f"  {s['module']}#{s['head']:<3d} loop {s['loop']:+.3f} | copy {s['copy']:+.3f} | other {s['other']:+.3f}")
    print("held-out half B, change of NLL (loop / copy / other):")
    for k, v in res["combos_b"].items():
        t, r = v["ablate_top"], v["ablate_random"]
        print(f"  top {k}: {t['loop']:+.3f} / {t['copy']:+.3f} / {t['other']:+.3f} | random {k}: {r['loop']:+.3f} / {r['copy']:+.3f} / {r['other']:+.3f}")
    if "patch_b" in res:
        p = res["patch_b"]
        print(f"  {p['name']} always on: {p['loop']:+.3f} / {p['copy']:+.3f} / {p['other']:+.3f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
