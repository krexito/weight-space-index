"""Self-gating MoI router (ERMoE idea, arXiv 2511.10971): can a patch decide when to switch on from its OWN edit
direction, without a separately trained router?

ERMoE routes each token to the expert whose basis it is most similar to (cosine), instead of a learned gate. Here
the "expert" is a .moipatch edit: its unit direction u lives in the output space of the edited module. Self-gate
score at token t = cos(y_t, u) (or y_t . u), where y_t is that module's output. If loop tokens point along u, the
patch can gate itself.

Compared on held-out answers (half B, split by problem id) of a finished run (requests_test.jsonl +
outputs_test_base.jsonl, e.g. discover_blind on GSM8K):
  * self_gate:   the patch's own u at each edited module (cos and dot), and their sum
  * probe:       a learned router analogue, diff-of-means direction of the residual per layer fitted on half A
  * random:      a random unit direction at the same modules (control)
  * text router: the current 16-gram repetition detector (fires when 50% of the last 64 tokens repeat)

Labels per token: a response is looping if the kill rule fires (90% of the last 768 tokens repeat); its loop region
starts at onset = kill - 768. Normal tokens = every token of non-looping responses plus tokens before onset - 256 in
looping ones.

Metrics: token AUC (loop region vs normal), and an early-warning test: threshold at 1% false alarms on normal tokens
(calibrated on half A, 32-token moving average), then in half B the share of looping answers caught, the lead in
tokens vs the text router, and the share of non-looping answers that would ever fire.
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

import moi_loop_router_vllm as lr  # noqa: E402
from wsi.moipatch import MoIPatch  # noqa: E402

SMOOTH = 32
PRE_GAP = 256
TARGET_FPR = 0.01


def load_sequences(work: Path, eos: set[int], max_tokens: int | None = None) -> list[dict]:
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    seqs = []
    for o in lr.read_jsonl(work / "outputs_test_base.jsonl"):
        st = lr.analyse(o["token_ids"], eos)
        body = st["body"][: st["kill"]] if st["kill"] is not None else st["body"]
        if max_tokens:
            body = body[:max_tokens]
        n = len(body)
        if n < SMOOTH * 2:
            continue
        looping = st["kill"] is not None
        onset = max(0, st["kill"] - lr.KILL_WINDOW) if looping else None
        label = np.zeros(n, np.int8)  # 0 normal, 1 loop region, -1 ignored (just before onset)
        if looping:
            label[onset:] = 1
            label[max(0, onset - PRE_GAP):onset] = -1
        seqs.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body, "label": label,
                     "looping": looping, "onset": onset, "fire": st["fire"]})
    return seqs


def find_module(model, suffix: str):
    for name, m in model.named_modules():
        if name.endswith(suffix):
            return m
    raise KeyError(suffix)


def backbone(model):
    """The decoder stack without the LM head (a 16k-token logit tensor would not fit)."""
    for attr in ("model", "transformer"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "forward"):
            return m
    return model


@torch.no_grad()
def run_one(model, seq: dict, mods: dict, dirs: dict, probe_dirs: torch.Tensor | None, dev) -> dict:
    """Per-token scores on the response part of one sequence. dirs: name -> (module_key, unit direction)."""
    outs: dict[str, torch.Tensor] = {}
    hooks = [m.register_forward_hook(lambda _m, _i, o, k=k: outs.__setitem__(k, o[0] if isinstance(o, tuple) else o))
             for k, m in mods.items()]
    try:
        ids = torch.tensor([seq["prompt"] + seq["body"]], device=dev)
        hs = backbone(model)(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    finally:
        for h in hooks:
            h.remove()
    p = len(seq["prompt"])
    res = {}
    for name, (mk, u) in dirs.items():
        y = outs[mk][0, p:].float()
        dot = y @ u
        res[f"{name}_dot"] = dot.cpu().numpy()
        res[f"{name}_cos"] = (dot / y.norm(dim=-1).clamp_min(1e-6)).cpu().numpy()
    resid = torch.stack([h[0, p:].float() for h in hs[1:]], 0)  # [L, T, D]
    res["_resid"] = resid if probe_dirs is None else None
    if probe_dirs is not None:
        res["probe"] = torch.einsum("ltd,ld->lt", resid, probe_dirs.to(resid.device)).cpu().numpy()
    return res


def smooth(x: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = np.empty_like(x, dtype=np.float64)
    k = np.arange(len(x))
    lo = np.maximum(0, k - w + 1)
    out[:] = (c[k + 1] - c[lo]) / (k + 1 - lo)
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    r = x.argsort().argsort().astype(np.float64) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(scores_a: list[np.ndarray], seqs_a: list[dict], scores_b: list[np.ndarray], seqs_b: list[dict]) -> dict:
    """One score family: orientation and threshold from half A, metrics on half B."""
    def split(scores, seqs):
        pos = np.concatenate([s[q["label"] == 1] for s, q in zip(scores, seqs)] or [np.zeros(0)])
        neg = np.concatenate([s[q["label"] == 0] for s, q in zip(scores, seqs)] or [np.zeros(0)])
        return pos, neg

    sm_a = [smooth(s) for s in scores_a]
    pa, na = split(sm_a, seqs_a)
    sign = 1.0 if auc(pa, na) >= 0.5 else -1.0  # a direction can point either way; orientation fixed on half A
    thr = float(np.quantile(sign * na, 1 - TARGET_FPR)) if len(na) else float("inf")
    sm_b = [sign * smooth(s) for s in scores_b]
    pb, nb = split(sm_b, seqs_b)
    caught, leads, normal_fired = [], [], []
    for s, q in zip(sm_b, seqs_b):
        hit = np.flatnonzero(s > thr)
        if q["looping"]:
            first = int(hit[0]) if len(hit) else None
            caught.append(first is not None)
            if first is not None and q["fire"] is not None:
                leads.append(q["fire"] - first)
        else:
            normal_fired.append(len(hit) > 0)
    return {"auc": round(auc(pb, nb), 4), "sign": sign,
            "loops_caught": round(float(np.mean(caught)), 4) if caught else None,
            "median_lead_vs_text_router": float(np.median(leads)) if leads else None,
            "share_before_text_router": round(float(np.mean([x > 0 for x in leads])), 4) if leads else None,
            "normal_answers_fired": round(float(np.mean(normal_fired)), 4) if normal_fired else None}


def text_router_metrics(seqs: list[dict]) -> dict:
    loop = [q for q in seqs if q["looping"]]
    normal = [q for q in seqs if not q["looping"]]
    return {"loops_caught": round(float(np.mean([q["fire"] is not None for q in loop])), 4) if loop else None,
            "normal_answers_fired": round(float(np.mean([q["fire"] is not None for q in normal])), 4) if normal else None,
            "median_fire_after_onset": float(np.median([q["fire"] - q["onset"] for q in loop if q["fire"] is not None]))
            if loop else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="finished run dir with config.json, requests_test.jsonl, outputs_test_base.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=None, help="cut each answer (smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    seqs = load_sequences(work, set(cfg["eos"]), a.max_tokens)
    half_a = [q for q in seqs if q["id"] % 2 == 0]
    half_b = [q for q in seqs if q["id"] % 2 == 1]
    dev = torch.device(a.device)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), trust_remote_code=True).to(dev).eval()
    patch = MoIPatch.load(a.patch)

    g = torch.Generator().manual_seed(a.seed)
    mods, dirs = {}, {}
    for i, e in enumerate(patch.edits):
        key = f"e{i}"
        mods[key] = find_module(model, e.module)
        u = e.u.float().to(dev)
        r = torch.randn(u.shape, generator=g).to(dev)
        dirs[f"self_{e.module}"] = (key, u / u.norm())
        dirs[f"random_{e.module}"] = (key, r / r.norm())

    # pass 1 (half A): self/random scores + residual sums for the probe (diff of means per layer)
    sums = cnts = None
    raw_a = []
    for q in half_a:
        r = run_one(model, q, mods, dirs, None, dev)
        resid = r.pop("_resid")
        lab = torch.as_tensor(q["label"], device=resid.device)
        if sums is None:
            sums = torch.zeros(2, resid.shape[0], resid.shape[2], device=resid.device, dtype=torch.float64)
            cnts = torch.zeros(2, device=resid.device, dtype=torch.float64)
        for c in (0, 1):
            m = lab == c
            if m.any():
                sums[c] += resid[:, m].sum(1).double()
                cnts[c] += m.sum()
        raw_a.append(r)
        del resid
    probe = (sums[1] / cnts[1].clamp_min(1) - sums[0] / cnts[0].clamp_min(1)).float()
    probe = probe / probe.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # probe scores for half A need a second look at the residual: recompute once with the fitted directions
    probe_a = [run_one(model, q, mods, {}, probe, dev)["probe"] for q in half_a]
    raw_b = [run_one(model, q, mods, dirs, probe, dev) for q in half_b]
    n_layers = probe.shape[0]

    res = {"model": cfg.get("model_repo") or a.model, "patch": Path(a.patch).name, "work": work.name,
           "n_half_a": len(half_a), "n_half_b": len(half_b),
           "loop_rate_b": round(float(np.mean([q["looping"] for q in half_b])), 4),
           "text_router": text_router_metrics(half_b), "self_gate": {}, "random": {}, "probe_by_layer": {}}
    for name in dirs:
        fam = "self_gate" if name.startswith("self_") else "random"
        for kind in ("cos", "dot"):
            k = f"{name}_{kind}"
            res[fam][k] = evaluate([r[k] for r in raw_a], half_a, [r[k] for r in raw_b], half_b)
    for kind in ("cos", "dot"):  # all edited modules together (sum of z-scores fitted on half A)
        names = [n for n in dirs if n.startswith("self_")]
        stats = {n: (np.concatenate([r[f"{n}_{kind}"] for r in raw_a]).mean(),
                     np.concatenate([r[f"{n}_{kind}"] for r in raw_a]).std() + 1e-8) for n in names}
        comb = lambda rs: [sum((r[f"{n}_{kind}"] - stats[n][0]) / stats[n][1] for n in names) for r in rs]  # noqa: E731
        res["self_gate"][f"all_modules_{kind}"] = evaluate(comb(raw_a), half_a, comb(raw_b), half_b)
    for li in range(n_layers):
        res["probe_by_layer"][str(li + 1)] = evaluate([p[li] for p in probe_a], half_a,
                                                      [r["probe"][li] for r in raw_b], half_b)
    best = max(res["probe_by_layer"], key=lambda k: res["probe_by_layer"][k]["auc"])
    res["probe_best"] = {"layer": int(best), **res["probe_by_layer"][best]}
    best_self = max(res["self_gate"], key=lambda k: res["self_gate"][k]["auc"])
    res["self_gate_best"] = {"name": best_self, **res["self_gate"][best_self]}
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))

    print(f"half B: {len(half_b)} answers, {res['loop_rate_b']:.0%} looping")
    print(f"text router  : caught {res['text_router']['loops_caught']}, normal fired {res['text_router']['normal_answers_fired']}, "
          f"fires {res['text_router']['median_fire_after_onset']} tok after onset")
    for k, v in [("self best", res["self_gate_best"]), ("probe best", res["probe_best"])] + \
            [(n, v) for n, v in res["random"].items() if n.endswith("_cos")]:
        print(f"{k:28s} AUC {v['auc']:.3f} | caught {v['loops_caught']} | lead vs text {v['median_lead_vs_text_router']} | "
              f"normal fired {v['normal_answers_fired']}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
