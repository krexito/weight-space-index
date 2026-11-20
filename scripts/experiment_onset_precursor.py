"""Onset precursor test: is there a residual direction that warns of a thinking loop BEFORE the text detector fires?

Today's loop signals all come late: the 16-gram text detector fires once 50% of the last 64 tokens repeat, and the
patch's own direction / a trained loop probe fire ~500-1,000 tokens after that (experiment_self_gate_router.py). Here
the target is the stretch just BEFORE the text detector fires (default: the 512 tokens before it, see
wsi/precursor.py for the labels). No generation: saved base answers of a finished run go through the model once.

Scores compared on held-out answers (half B, odd problem ids; half A, even ids, fits directions and thresholds):
  * precursor_L{l}: diff-of-means direction (precursor window - earlier tokens) of the residual after layer l
  * loopprobe_L{l}: the usual loop probe (tokens after the text detector fired - earlier tokens), same eval
  * random_L{best}: a random direction at the best precursor layer (control)
  * position:      the token index alone (loops happen late: the direction must beat this)
  * text_rep64 / text_rep256: share of repeated 16-grams in the last 64 / 256 tokens (a looser text detector: the
    direction must beat this too, otherwise it is only seeing early repetition)

The best layer of each family is picked on half A (in-sample AUC), then reported on half B.
Metrics (wsi.precursor.early_warning; threshold set so ~10% of half-A answers raise a false alarm): share of loops warned
0 < lead <= 2048 tokens before the text detector, median lead, share warned implausibly early, and the share of
non-looping answers that ever raise the alarm.
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

import experiment_self_gate_router as sg  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from wsi.precursor import IGNORED, NEGATIVE, PRECURSOR, ClassMeans, early_warning, precursor_labels, smooth  # noqa: E402


def loop_labels(n: int, anchor: int | None, window: int, gap: int) -> np.ndarray:
    """Labels for the classic loop probe: 1 from the anchor on, 0 on the same tokens the precursor calls negative."""
    lab = precursor_labels(n, anchor, window, gap)
    if anchor is not None:
        lab[lab == PRECURSOR] = IGNORED
        lab[min(max(anchor, 0), n):] = PRECURSOR
    return lab


@torch.no_grad()
def residuals(model, seq: dict, dev) -> torch.Tensor:
    ids = torch.tensor([seq["prompt"] + seq["body"]], device=dev)
    hs = sg.backbone(model)(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    p = len(seq["prompt"])
    return torch.stack([h[0, p:].float() for h in hs[1:]], 0)  # [L, T, D]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="finished run dir with config.json, requests_test.jsonl, outputs_test_base.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=512, help="precursor window before the text detector fires")
    ap.add_argument("--gap", type=int, default=256, help="ignored tokens between precursor and negatives")
    ap.add_argument("--fpr", type=float, default=0.10, help="share of half-A answers allowed a false alarm")
    ap.add_argument("--max-tokens", type=int, default=None, help="cut each answer (smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    seqs = sg.load_sequences(work, set(cfg["eos"]), a.max_tokens)
    for q in seqs:
        anchor = None
        if q["looping"]:
            anchor = q["fire"] if q["fire"] is not None else q["onset"]
        q["anchor"] = anchor
        n = len(q["body"])
        q["pre"] = precursor_labels(n, anchor, a.window, a.gap)
        q["loop"] = loop_labels(n, anchor, a.window, a.gap)
        d = np.asarray(lr.dup_flags(q["body"]), np.float64)
        q["text_rep64"], q["text_rep256"] = smooth(d, 64), smooth(d, 256)
        q["position"] = np.arange(n, dtype=np.float64)
    half_a = [q for q in seqs if q["id"] % 2 == 0]
    half_b = [q for q in seqs if q["id"] % 2 == 1]
    usable_a = [q for q in half_a if q["anchor"] is None or (q["pre"] == PRECURSOR).any()]
    dev = torch.device(a.device)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), trust_remote_code=True).to(dev).eval()

    # pass 1 (half A): class sums for both direction families
    pre_m, loop_m = ClassMeans(), ClassMeans()
    for q in usable_a:
        r = residuals(model, q, dev)
        pre_m.add(r, q["pre"])
        loop_m.add(r, q["loop"])
        del r
    dirs = {"precursor": pre_m.directions(), "loopprobe": loop_m.directions()}
    n_layers, hidden = dirs["precursor"].shape
    g = torch.Generator().manual_seed(a.seed)
    rnd = torch.randn(n_layers, hidden, generator=g)
    dirs["random"] = (rnd / rnd.norm(dim=-1, keepdim=True)).to(dirs["precursor"].device)

    # pass 2 / 3: per-layer scores for both halves
    def scores(q):
        r = residuals(model, q, dev)
        out = {fam: torch.einsum("ltd,ld->lt", r, D.to(r.device)).cpu().numpy() for fam, D in dirs.items()}
        del r
        return out

    sc_a = [scores(q) for q in half_a]
    sc_b = [scores(q) for q in half_b]
    lab_a = [q["pre"] for q in half_a]
    lab_b = [q["pre"] for q in half_b]
    anc_b = [q["anchor"] for q in half_b]

    def ev(fa, fb, **kw):
        return early_warning(fa, lab_a, fb, lab_b, anc_b, **({"target_fpr": a.fpr} | kw))

    res = {"model": cfg.get("model_repo") or a.model, "work": work.name, "window": a.window, "gap": a.gap,
           "target_fpr": a.fpr, "n_half_a": len(half_a), "n_half_b": len(half_b),
           "loop_rate_b": round(float(np.mean([q["looping"] for q in half_b])), 4),
           "precursor_tokens_a": int(pre_m.counts[1]), "by_layer": {"precursor": {}, "loopprobe": {}}, "baselines": {}}
    for fam in ("precursor", "loopprobe"):
        for li in range(n_layers):
            res["by_layer"][fam][str(li + 1)] = ev([s[fam][li] for s in sc_a], [s[fam][li] for s in sc_b])
    best = {fam: max(res["by_layer"][fam], key=lambda k: res["by_layer"][fam][k]["auc_a"]) for fam in ("precursor", "loopprobe")}
    res["best"] = {fam: {"layer": int(k), **res["by_layer"][fam][k]} for fam, k in best.items()}
    res["best_token_fpr_1pct"] = {  # the self-gate test's calibration, for comparison
        fam: ev([s[fam][int(k) - 1] for s in sc_a], [s[fam][int(k) - 1] for s in sc_b], level="token", target_fpr=0.01)
        for fam, k in best.items()}
    bl = int(best["precursor"]) - 1
    res["baselines"][f"random_L{bl + 1}"] = ev([s["random"][bl] for s in sc_a], [s["random"][bl] for s in sc_b])
    for k in ("position", "text_rep64", "text_rep256"):
        res["baselines"][k] = ev([q[k] for q in half_a], [q[k] for q in half_b])
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))

    print(f"half B: {len(half_b)} answers, {res['loop_rate_b']:.0%} looping; precursor window {a.window} tokens before the text detector")
    rows = [(f"precursor L{best['precursor']} (best)", res["best"]["precursor"]),
            (f"loop probe L{best['loopprobe']} (best)", res["best"]["loopprobe"])] + list(res["baselines"].items())
    for k, v in rows:
        print(f"{k:28s} AUC {v['auc']:.3f} | warned useful {v['warned_useful']} (median lead {v['median_lead_useful']}) | "
              f"too early {v['warned_too_early']} | normal fired {v['normal_answers_fired']}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
