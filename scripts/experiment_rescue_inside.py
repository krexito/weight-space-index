"""EXPERIMENTAL: why does the blind loop patch rescue some loops and not others? A look from the inside, on saved runs.

Data: the discover_blind GSM8K test (Qwen3.5-0.8B, 16k tokens). The router cut 189 base answers at the loop; the
continuation was generated with the blind patch (L13-14, 16 neurons, alpha 12) and, as a control, with a random
direction (same layers / neurons / alpha). Outcome groups (from the saved answers, nothing regenerated):
  A  patch-only rescue   blind correct, random wrong   (the patch made the difference)
  B  any-restart rescue  both correct                  (restarting was enough)
  C  no rescue           both wrong
All activations are recomputed from the saved token ids in fp32 (wsi.token_trace logic; exact on the base prefix).

Measured per answer (fixed before looking at results):
  at the cut (base model, prompt + base prefix):
    proj_L13 / proj_L14   residual (block output) on the patch direction u_l, mean over the last 64 prefix tokens
    loop_age              tokens from the first repeated 16-gram to the cut;  cut = prefix length
  the patch's push at the cut (patched vs base model, same tokens, last 64 positions):
    kl_cut                mean KL(base || patched) of the next-token distribution
    top1_flip             share of the 64 positions whose top-1 token changes
    dproj_L14             drop of proj_L14 under the patch
  after the cut (patched model over prompt + prefix + the first 512 tokens of the blind continuation):
    proj_L14_after        mean proj_L14 over the 512 continuation tokens, minus the same model's value at the cut
Hypotheses: H1 C has a stronger loop signal at the cut than A; H2 the patch pushes less in C; H3 in C the signal comes
back after the cut. Each feature: AUC A vs C (and A vs B, B vs C) with a bootstrap 95% CI.

Run (GPU recommended; CPU works but ~3 h):
  python scripts/experiment_rescue_inside.py --work <discover_blind folder> --rows rescue_rows.json   (rows from scripts/experiment_rescue_outcomes.py) --model <qwen3.5-0.8b> --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import moi_loop_router_vllm as lr  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402

LAYERS = (13, 14)
NEURONS = 16
ALPHA = 12.0
LAST = 64
AFTER = 512


def read_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def patch_model(model, u: dict[int, np.ndarray], ranks: dict[str, list[int]]):
    """In place: W' = W - alpha * u (m * u^T W) on mlp.down_proj of each patched layer (discover_blind's edit form)."""
    _, blocks = find_blocks(model)
    for li in LAYERS:
        W = blocks[li].mlp.down_proj.weight
        Wd = W.data.double()
        uu = torch.from_numpy(u[li]).to(Wd)
        m = torch.zeros(Wd.shape[1], dtype=torch.float64, device=Wd.device)
        m[sorted(ranks[str(li)][:NEURONS])] = 1.0
        W.data = (Wd - ALPHA * torch.outer(uu, m * (uu @ Wd))).to(W.dtype)
    return model


def run(model, ids: list[int], keep_logits: int, dev):
    _, blocks = find_blocks(model)
    acts = {}
    hs = [blocks[l].register_forward_hook(lambda m, a, o, l=l: acts.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].float()))
          for l in LAYERS]
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor(ids, device=dev)[None], use_cache=False, logits_to_keep=keep_logits)
    finally:
        for h in hs:
            h.remove()
    return acts, out.logits[0].float()


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    bs = [auc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--rows", required=True, help="outcome rows (rescue_outcomes): id, arm, correct")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N answers only (smoke test)")
    a = ap.parse_args()
    W = Path(a.work)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads((W / "config.json").read_text())
    eos = set(cfg["eos"])
    z = np.load(W / "blind_directions.npz")
    u = {li: z[f"u{li}"].astype(np.float64) for li in LAYERS}
    ranks = json.loads((W / "diagnosis_blind.json").read_text())["neuron_rank"]
    reqs = {r["id"]: r for r in read_jsonl(W / "requests_test.jsonl")}
    plan = {r["id"]: r for r in read_jsonl(W / "requests_test_moi.jsonl")}
    cont = {o["id"]: o for o in read_jsonl(W / "outputs_test_moi_blind.jsonl")}
    rows = json.load(open(a.rows))
    ok = {(r["id"], r["arm"]): r["correct"] for r in rows}
    ids_all = sorted({r["id"] for r in rows})
    group = {}
    for i in ids_all:
        b, q = ok[(i, "blind")], ok[(i, "random")]
        group[i] = "A" if b and not q else "B" if b and q else "C" if not b and not q else "D"
    todo = [i for i in ids_all if group[i] != "D"]
    if a.limit:
        todo = sorted(todo, key=lambda i: len(plan[i]["prompt_ids"]))[: a.limit]

    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    pat = patch_model(AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval(), u, ranks)
    U = {li: torch.from_numpy(u[li]).float().to(dev) for li in LAYERS}

    feats = {}
    t0 = time.time()
    for n, i in enumerate(todo):
        p = plan[i]
        full = p["prompt_ids"]
        cut = len(full)
        prefix = full[len(reqs[i]["prompt_ids"]):]
        d = lr.dup_flags(lr.strip_eos(prefix, eos))
        first = next((k for k, x in enumerate(d) if x), len(prefix))
        ab, lb = run(base, full, LAST, dev)
        ap_, lp = run(pat, full, LAST, dev)
        lpb, lpp = torch.log_softmax(lb, -1), torch.log_softmax(lp, -1)
        kl = (lpb.exp() * (lpb - lpp)).sum(-1)
        f = {"group": group[i], "cut": len(prefix), "loop_age": len(prefix) - first,
             "kl_cut": float(kl.mean()), "top1_flip": float((lpb.argmax(-1) != lpp.argmax(-1)).float().mean())}
        for li in LAYERS:
            f[f"proj_L{li}"] = float((ab[li][-LAST:] @ U[li]).mean())
            f[f"patched_proj_L{li}"] = float((ap_[li][-LAST:] @ U[li]).mean())
        f["dproj_L14"] = f["proj_L14"] - f["patched_proj_L14"]
        c = cont[i]["token_ids"][:AFTER]
        if len(c) >= 32:
            aa, _ = run(pat, full + c, 1, dev)
            f["proj_L14_after"] = float((aa[14][cut:] @ U[14]).mean()) - f["patched_proj_L14"]
            f["after_len"] = len(c)
        feats[i] = f
        if n % 10 == 0 or n == len(todo) - 1:
            print(f"[{n + 1}/{len(todo)}] id {i} grp {group[i]} cut {len(prefix)} kl {f['kl_cut']:.3f} "
                  f"proj14 {f['proj_L14']:.2f} -> {f['patched_proj_L14']:.2f}  {time.time() - t0:.0f}s", flush=True)

    names = ["cut", "loop_age", "proj_L13", "proj_L14", "kl_cut", "top1_flip", "dproj_L14", "proj_L14_after"]
    res = {"setup": {"layers": LAYERS, "neurons": NEURONS, "alpha": ALPHA, "last": LAST, "after": AFTER,
                     "device": str(dev), "torch": str(torch.__version__)},
           "groups": {g: sum(1 for f in feats.values() if f["group"] == g) for g in "ABC"},
           "group_D_blind_wrong_random_right": sum(1 for i in ids_all if group[i] == "D"),
           "summary": {}, "auc": {}, "per_answer": {str(k): v for k, v in feats.items()}}
    for nm in names:
        vals = {g: [f[nm] for f in feats.values() if f["group"] == g and nm in f] for g in "ABC"}
        res["summary"][nm] = {g: {"median": round(float(np.median(v)), 4), "q25": round(float(np.percentile(v, 25)), 4),
                                  "q75": round(float(np.percentile(v, 75)), 4), "n": len(v)} for g, v in vals.items() if v}
        res["auc"][nm] = {}
        for x, y in (("A", "C"), ("A", "B"), ("B", "C")):
            if len(vals[x]) > 2 and len(vals[y]) > 2:
                res["auc"][nm][f"{x}_vs_{y}"] = {"auc": round(auc(vals[x], vals[y]), 3), "ci95": auc_ci(vals[x], vals[y])}
        print(nm, {g: res["summary"][nm][g]["median"] for g in res["summary"][nm]},
              {k: v["auc"] for k, v in res["auc"][nm].items()}, flush=True)
    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1))
    Path(a.out + ".tmp").replace(a.out)
    print("saved", a.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
