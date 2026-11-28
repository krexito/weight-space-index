"""EXPERIMENTAL: can a "sensor token" read on the LAST layer replace the self-gate veto read at L13-14? Saved answers only.

Why: vLLM 0.34 returns the raw logit of chosen token ids at every generated token (`logprob_token_ids` with
`logprobs_mode="raw_logits"`), taken before any logits processor. A spare vocabulary row (Qwen3.5-0.8B has 250 unused
rows, ids 248070-248319) whose lm_head row is a probe direction w, blocked from sampling, gives logit = w . h_final at
every token: a read channel that leaves every answer unchanged and works inside any engine. But the lm_head only sees the
final normed hidden state h_final, while the veto reads cos(down_proj output, u) at L13/L14. Question: does a linear
readout of h_final detect loops as well as the veto score?

Same answers, labels, folds and rules as experiment_router_calibration.py (discover_blind GSM8K run, 596 base answers):
  veto_L13_14    current veto score: mean cos(down_proj output, u) over the patch layers (fixed direction)
  final_sensor   w . h_final with w = diff of means (loop-region minus normal tokens) of h_final, fitted on the training
                 answers only (one w per cross-validation fold, plus one on elicit for the elicit -> test replication).
                 Linear, no normalisation: exactly what the sensor row would output.
Both get the same 32-token moving average, orientation and thresholds (fit on training answers). Reported: token AUC on
held-out answers (bootstrap CI over problems, and the paired difference), and per rule healthy / healthy-correct / loop
switch-ons, as in the router calibration.

  score    (GPU)  python scripts/experiment_final_sensor.py score   --work W --model M --out S.npz
  analyze  (CPU)  python scripts/experiment_final_sensor.py analyze --work W --scores S.npz --tokenizer M --out R.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

for _m in ("fla", "causal_conv1d"):  # same GDN path as the validated CPU/GPU fp32 recompute
    sys.modules.setdefault(_m, None)

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import moi_loop_router_vllm as lr  # noqa: E402
from experiment_router_calibration import (FOLDS, SPLITS, auc, calibrate, fit, load_all, on_of,  # noqa: E402
                                           register_chunked_attention, smooth, summarize)

RULES = ("text", "token_fpr_1", "token_fpr_5", "answer_5", "answer_10")
DIRS = tuple(f"fold{f}" for f in range(FOLDS)) + ("elicit",)


def fold_map(seqs: list[dict], seed: int) -> dict[str, int]:
    """Same grouped fold assignment as experiment_router_calibration.cmd_analyze."""
    groups = sorted({q["group"] for q in seqs})
    rng = np.random.default_rng(seed)
    return {g: i % FOLDS for i, g in enumerate(rng.permutation(groups))}


def train_of(d: str, q: dict, fold_of: dict) -> bool:
    """Is answer q in the training set of direction d?"""
    return q["split"] == "elicit" if d == "elicit" else fold_of[q["group"]] != int(d[4:])


def cmd_score(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    eos = set(json.loads((work / "config.json").read_text())["eos"])
    layers_ = json.loads((work / "frozen.json").read_text())["choice"]["layers"]
    z = np.load(work / "blind_directions.npz")
    seqs = load_all(work, eos)
    fold_of = fold_map(seqs, a.seed)  # on ALL answers, so a --limit smoke test keeps the real folds
    seqs = sorted(seqs, key=lambda q: len(q["prompt"]) + len(q["body"]), reverse=not a.limit)
    if a.limit:
        seqs = seqs[: a.limit]
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32,
                                                 attn_implementation=register_chunked_attention(a.chunk)).to(dev).eval()
    blocks = next(m for n, m in model.named_modules()
                  if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 2)
    mods = {li: lr.down_module(blocks[li]) for li in layers_}
    us = {li: torch.from_numpy(z[f"u{li}"]).float().to(dev) for li in layers_}
    backbone = getattr(model, "model", model)

    def forward(q, veto: bool):
        ys: dict = {}
        hs = [m.register_forward_hook(lambda _m, _i, o, k=k: ys.__setitem__(k, o[0] if isinstance(o, tuple) else o))
              for k, m in mods.items()] if veto else []
        try:
            with torch.no_grad():
                h = backbone(input_ids=torch.tensor([q["prompt"] + q["body"]], device=dev), use_cache=False)
        finally:
            for x in hs:
                x.remove()
        p0 = len(q["prompt"])
        hf = h.last_hidden_state[0, p0:].float()  # after the final norm: what the lm_head (and a sensor row) reads
        if not veto:
            return hf, None
        cos = [(ys[k][0, p0:].float() @ u) / ys[k][0, p0:].float().norm(dim=-1).clamp_min(1e-6) for k, u in us.items()]
        return hf, torch.stack(cos).mean(0)

    out: dict[str, np.ndarray] = {}
    hid = model.config.get_text_config().hidden_size
    fold_sums = {c: torch.zeros(FOLDS, hid, dtype=torch.float64, device=dev) for c in ("loop", "normal")}
    fold_n = {c: np.zeros(FOLDS) for c in ("loop", "normal")}
    el_sums = {c: torch.zeros(hid, dtype=torch.float64, device=dev) for c in ("loop", "normal")}
    el_n = {"loop": 0, "normal": 0}
    t0, ntok = time.time(), 0
    # pass 1: veto score for every answer; class sums of h_final per fold and for elicit
    for n, q in enumerate(seqs):
        hf, v = forward(q, veto=True)
        out[f"veto|{q['key']}"] = v.cpu().numpy().astype(np.float32)
        lab = torch.from_numpy(q["label"]).to(dev)
        f = fold_of[q["group"]]
        for c, val in (("loop", 1), ("normal", 0)):
            m = lab == val
            k = int(m.sum())
            if k:
                s = hf[m].double().sum(0)
                fold_sums[c][f] += s
                fold_n[c][f] += k
                if q["split"] == "elicit":
                    el_sums[c] += s
                    el_n[c] += k
        ntok += len(q["prompt"]) + len(q["body"])
        if n % 25 == 0 or n == len(seqs) - 1:
            print(f"[pass1 {n + 1}/{len(seqs)}] {q['key']} {ntok / (time.time() - t0):.0f} tok/s {time.time() - t0:.0f}s",
                  flush=True)
    W = []
    for d in DIRS:
        if d == "elicit":
            ml, mn = el_sums["loop"] / max(1, el_n["loop"]), el_sums["normal"] / max(1, el_n["normal"])
        else:
            f = int(d[4:])
            keep = [i for i in range(FOLDS) if i != f]
            ml = fold_sums["loop"][keep].sum(0) / max(1, fold_n["loop"][keep].sum())
            mn = fold_sums["normal"][keep].sum(0) / max(1, fold_n["normal"][keep].sum())
        W.append((ml - mn).float())
    W = torch.stack(W)  # [6, hidden]: the sensor rows (logit units, not normalised)
    out["__W"] = W.cpu().numpy()
    print("direction norms", [round(float(x), 3) for x in W.norm(dim=-1)], "pairwise cos fold dirs",
          round(float(torch.nn.functional.cosine_similarity(W[0], W[1], dim=0)), 3), flush=True)
    # pass 2: sensor logits w . h_final under every direction
    for n, q in enumerate(seqs):
        hf, _ = forward(q, veto=False)
        out[f"sensor|{q['key']}"] = (hf @ W.T).T.cpu().numpy().astype(np.float32)  # [6, T]
        if n % 25 == 0 or n == len(seqs) - 1:
            print(f"[pass2 {n + 1}/{len(seqs)}] {time.time() - t0:.0f}s", flush=True)
    out["__meta"] = np.frombuffer(json.dumps({"dirs": DIRS, "seed": a.seed, "layers": layers_,
                                              "counts_elicit": el_n}).encode(), np.uint8)
    tmp = a.out + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    Path(tmp).replace(a.out)
    print("saved", a.out, len(seqs), "answers", f"{time.time() - t0:.0f}s", "device", dev, "torch", torch.__version__)


# ------------------------------------------------------------------ analysis (CPU)
def token_auc(seqs: list[dict], score_of) -> float:
    """Pooled token AUC (loop region vs normal tokens) of 32-token smoothed, already oriented scores."""
    pos = np.concatenate([smooth(score_of(q))[q["label"] == 1] for q in seqs] or [np.zeros(0)])
    neg = np.concatenate([smooth(score_of(q))[q["label"] == 0] for q in seqs] or [np.zeros(0)])
    return auc(pos, neg)


def cmd_analyze(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    eos = set(json.loads((work / "config.json").read_text())["eos"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    z = np.load(a.scores)
    meta = json.loads(bytes(z["__meta"]).decode())
    all_seqs = load_all(work, eos)
    fold_of = fold_map(all_seqs, meta["seed"])
    seqs = [q for q in all_seqs if f"veto|{q['key']}" in z.files]
    veto = {q["key"]: z[f"veto|{q['key']}"] for q in seqs}
    sens = {q["key"]: z[f"sensor|{q['key']}"] for q in seqs}
    di = {d: i for i, d in enumerate(meta["dirs"])}
    correct = {}
    for name, rq, op in SPLITS:
        reqs = {r["id"]: r for r in lr.read_jsonl(work / rq)}
        for o in lr.read_jsonl(work / op):
            r = reqs[o["id"]]
            correct[f"{name}:{o['id']}"] = bool(lr.grade(o["token_ids"], o["finish_reason"], r["gold"], tok, eos,
                                                          r["max_tokens"])["correct"])

    def sc_for(arm: str, d: str) -> dict:
        return veto if arm == "veto_L13_14" else {k: v[di[d]] for k, v in sens.items()}

    arms = ("veto_L13_14", "final_sensor")
    res = {"data": {"answers": len(seqs), "loops": sum(q["looping"] for q in seqs),
                    "healthy": sum(not q["looping"] for q in seqs),
                    "healthy_correct": sum((not q["looping"]) and correct[q["key"]] for q in seqs)},
           "sensor_direction_norms": [round(float(x), 4) for x in np.linalg.norm(z["__W"], axis=1)]}

    # 1. replication elicit -> test (the veto's own calibration: 1% token false alarms on elicit)
    el = [q for q in seqs if q["split"] == "elicit"]
    te = [q for q in seqs if q["split"] == "test"]
    res["replication_elicit_to_test"] = {}
    for arm in arms:
        sc = sc_for(arm, "elicit")
        cal = calibrate([sc[q["key"]] for q in el], el, 0.01)
        res["replication_elicit_to_test"][arm] = {
            "calibration": cal, "token_auc_test": round(token_auc(te, lambda q: cal["sign"] * sc[q["key"]]), 4),
            **summarize(te, {q["key"]: on_of(q, sc, cal) for q in te}, correct)}

    # 2. grouped 5-fold cross-validation: each fold's rules (and the sensor direction) fit on the other folds only
    ons = {arm: {r: {} for r in RULES} for arm in arms}
    held_scores = {arm: {} for arm in arms}  # oriented held-out score per answer, for the token AUC
    for f in range(FOLDS):
        train = [q for q in seqs if train_of(f"fold{f}", q, fold_of)]
        held = [q for q in seqs if not train_of(f"fold{f}", q, fold_of)]
        for arm in arms:
            sc = sc_for(arm, f"fold{f}")
            for r in RULES:
                c = fit(r, train, sc)
                for q in held:
                    ons[arm][r][q["key"]] = on_of(q, sc, c)
            sign = calibrate([sc[q["key"]] for q in train], train, 0.01)["sign"]
            for q in held:
                held_scores[arm][q["key"]] = sign * sc[q["key"]]
    res["cv"] = {arm: {r: summarize(seqs, ons[arm][r], correct) for r in RULES} for arm in arms}

    # 3. held-out token AUC with a bootstrap over problems (paired: same resample for both arms)
    groups = sorted({q["group"] for q in seqs})
    by_g = {g: [q for q in seqs if q["group"] == g] for g in groups}
    cache = {arm: {q["key"]: smooth(held_scores[arm][q["key"]]) for q in seqs} for arm in arms}

    def auc_of(arm, qs):
        pos = np.concatenate([cache[arm][q["key"]][q["label"] == 1] for q in qs] or [np.zeros(0)])
        neg = np.concatenate([cache[arm][q["key"]][q["label"] == 0] for q in qs] or [np.zeros(0)])
        return auc(pos, neg)

    point = {arm: auc_of(arm, seqs) for arm in arms}
    rng = np.random.default_rng(a.seed)
    boots = {arm: [] for arm in arms}
    for _ in range(a.boot):
        qs = [q for g in rng.choice(groups, len(groups)) for q in by_g[g]]
        for arm in arms:
            boots[arm].append(auc_of(arm, qs))
    diff = np.asarray(boots["final_sensor"]) - np.asarray(boots["veto_L13_14"])
    ci = lambda x: [round(float(np.nanpercentile(x, 2.5)), 4), round(float(np.nanpercentile(x, 97.5)), 4)]  # noqa: E731
    res["token_auc_heldout"] = {**{arm: {"auc": round(point[arm], 4), "ci95": ci(boots[arm])} for arm in arms},
                                "sensor_minus_veto": {"diff": round(point["final_sensor"] - point["veto_L13_14"], 4),
                                                      "ci95": ci(diff)}}

    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1))
    Path(a.out + ".tmp").replace(a.out)
    print(json.dumps({"data": res["data"], "token_auc_heldout": res["token_auc_heldout"]}, indent=1))
    for arm in arms:
        rp = res["replication_elicit_to_test"][arm]
        print(f"replication {arm:13s} test AUC {rp['token_auc_test']}  healthy on {rp['healthy_on']}  "
              f"loops on {rp['loops_on']}")
    for r in RULES:
        for arm in arms:
            m = res["cv"][arm][r]
            print(f"{r:12s} {arm:13s} healthy on {m['healthy_on']:>7s}  correct-healthy on {m['healthy_correct_on']:>6s}"
                  f"  loops on {m['loops_on']:>7s}  on-onset {m['median_on_minus_onset']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--work", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int, default=0, help="shortest N answers only (smoke test)")
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--chunk", type=int, default=2048, help="queries per attention block (exact; memory only)")
    n = sub.add_parser("analyze")
    n.add_argument("--work", required=True)
    n.add_argument("--scores", required=True)
    n.add_argument("--tokenizer", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    {"score": cmd_score, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
