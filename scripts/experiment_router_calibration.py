"""EXPERIMENTAL: calibrate the self-gate veto router on every saved base answer (no generation).

The veto (experiment_veto_moi.py) switches the blind loop patch on only when the text detector has fired AND the
patch's own direction confirms: score = mean cos(down_proj output, u) over L13/L14, 32-token moving average, threshold
at 1% false alarms on NORMAL TOKENS of the elicit answers. In generation it switched on 3/23 healthy answers; that
false-alarm rate rests on 23 answers. Here: every base answer of the discover_blind GSM8K run (elicit 300, dev1 32,
dev2 64, test 200 = 596 answers, 116 healthy), scored once from the saved token ids (fp32, torch kernels).

Rules compared on held-out problems (5-fold cross-validation grouped by prompt, fixed before looking):
  text            switch on where the text detector fires (current MoI)
  token_fpr_1/5   veto, threshold at 1% / 5% of normal tokens (current rule: 1%)
  answer_5/10     veto, threshold set so that at most 5% / 10% of the training healthy answers would switch on
                  (statistic per answer: max smoothed score from the text fire to the end; answers the text detector
                  never flags cannot switch on)
Per rule: healthy answers switched on (Wilson 95% CI; also healthy-and-correct ones), loops switched on before the kill,
median switch-on minus onset. Plus a replication of the original calibration (elicit -> test) and a threshold sweep.

  score    (GPU)  python scripts/experiment_router_calibration.py score   --work W --model M --out scores.npz
  analyze  (CPU)  python scripts/experiment_router_calibration.py analyze --work W --scores scores.npz --tokenizer M --out R.json
"""
from __future__ import annotations

import argparse
import hashlib
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

SMOOTH = 32
PRE_GAP = 256


# ---- veto pieces, identical to experiment_veto_moi.py (the script of the veto-in-generation run, not in git) ----
def load_answers(work: Path, req_file: str, out_file: str, eos: set[int], max_tokens: int | None = None) -> list[dict]:
    reqs = {r["id"]: r for r in lr.read_jsonl(work / req_file)}
    out = []
    for o in lr.read_jsonl(work / out_file):
        st = lr.analyse(o["token_ids"], eos)
        looping = st["kill"] is not None
        body = st["body"][: st["kill"]] if looping else st["body"]
        if max_tokens:
            body = body[:max_tokens]
        if len(body) < 2 * SMOOTH:
            continue
        onset = max(0, st["kill"] - lr.KILL_WINDOW) if looping else None
        label = np.zeros(len(body), np.int8)  # 0 normal, 1 loop region, -1 ignored just before onset
        if looping:
            label[onset:] = 1
            label[max(0, onset - PRE_GAP):onset] = -1
        out.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body, "label": label,
                    "looping": looping, "onset": onset, "fire": st["fire"]})
    return out


def smooth(x: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    c = np.cumsum(np.insert(np.asarray(x, np.float64), 0, 0.0))
    k = np.arange(len(x))
    lo = np.maximum(0, k - w + 1)
    return (c[k + 1] - c[lo]) / (k + 1 - lo)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    r = x.argsort().argsort().astype(np.float64) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibrate(scores: list[np.ndarray], seqs: list[dict], fpr: float) -> dict:
    sm = [smooth(s) for s in scores]
    pos = np.concatenate([s[q["label"] == 1] for s, q in zip(sm, seqs)] or [np.zeros(0)])
    neg = np.concatenate([s[q["label"] == 0] for s, q in zip(sm, seqs)] or [np.zeros(0)])
    a = auc(pos, neg)
    sign = 1.0 if a >= 0.5 else -1.0
    thr = float(np.quantile(sign * neg, 1 - fpr))
    return {"sign": sign, "thr": thr, "auc_calib": round(max(a, 1 - a), 4), "fpr": fpr,
            "n_loop_tokens": int(len(pos)), "n_normal_tokens": int(len(neg))}


def switch_on(score: np.ndarray, fire: int | None, cal: dict) -> int | None:
    """First token >= the text fire whose smoothed, oriented score passes the threshold; None = vetoed."""
    if fire is None:
        return None
    s = cal["sign"] * smooth(score)
    hit = np.flatnonzero(s[fire:] > cal["thr"])
    return int(fire + hit[0]) if len(hit) else None


SPLITS =(("elicit", "requests_elicit.jsonl", "outputs_elicit.jsonl"),
          ("dev1", "requests_dev1.jsonl", "outputs_dev1_base.jsonl"),
          ("dev2", "requests_dev2.jsonl", "outputs_dev2_base.jsonl"),
          ("test", "requests_test.jsonl", "outputs_test_base.jsonl"))
FOLDS = 5


def load_all(work: Path, eos: set[int]) -> list[dict]:
    out = []
    for name, rq, op in SPLITS:
        for q in load_answers(work, rq, op, eos):
            q["split"], q["key"] = name, f"{name}:{q['id']}"
            q["group"] = hashlib.sha1(np.asarray(q["prompt"], np.int64).tobytes()).hexdigest()[:16]
            out.append(q)
    return out


class _Stop(Exception):
    pass


def register_chunked_attention(chunk: int) -> str:
    """Exact SDPA computed in blocks of `chunk` queries (each softmax row is independent). fp32 SDPA on CUDA has no
    fused kernel here and the math kernel builds the full 16k x 16k score matrix per head (OOM on 24 GB). The GDN layers
    cannot be fed in chunks instead: with a cache, a multi-token chunk restarts their recurrent state."""
    import torch
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def fwd(module, query, key, value, attention_mask, **kw):
        q_len, k_len = query.shape[2], key.shape[2]
        if q_len <= chunk:
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        kw = {**kw, "is_causal": False}
        outs = []
        for s in range(0, q_len, chunk):
            e = min(q_len, s + chunk)
            kv = k_len - q_len + e
            if attention_mask is not None:
                m = attention_mask[:, :, s:e, :kv]
            else:  # causal: absolute query position (k_len - q_len + s + i) sees keys j <= that position
                pos = torch.arange(k_len - q_len + s, k_len - q_len + e, device=query.device)[:, None]
                m = (torch.arange(kv, device=query.device)[None, :] <= pos)[None, None]
            o, _ = sdpa_attention_forward(module, query[:, :, s:e], key[:, :, :kv], value[:, :, :kv], m, **kw)
            outs.append(o)
        return torch.cat(outs, 1), None

    AttentionInterface.register("chunked_sdpa", fwd)
    return "chunked_sdpa"


def cmd_score(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    eos = set(json.loads((work / "config.json").read_text())["eos"])
    layers_ = json.loads((work / "frozen.json").read_text())["choice"]["layers"]
    z = np.load(work / "blind_directions.npz")
    # longest first (an out-of-memory shows up at once); --limit takes the shortest (smoke test)
    seqs = sorted(load_all(work, eos), key=lambda q: len(q["prompt"]) + len(q["body"]), reverse=not a.limit)
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
    dirs = {li: torch.from_numpy(z[f"u{li}"]).float().to(dev) for li in layers_}
    backbone = getattr(model, "model", model)
    last = blocks[max(layers_)]

    def stop(*_):
        raise _Stop

    scores, t0, ntok = {}, time.time(), 0
    for n, q in enumerate(seqs):
        ys: dict = {}
        hs = [m.register_forward_hook(lambda _m, _i, o, k=k: ys.__setitem__(k, o[0] if isinstance(o, tuple) else o))
              for k, m in mods.items()]
        hs.append(last.register_forward_hook(stop))  # nothing past the last patched layer is needed
        try:
            with torch.no_grad():
                backbone(input_ids=torch.tensor([q["prompt"] + q["body"]], device=dev), use_cache=False)
        except _Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        p0 = len(q["prompt"])
        cos = [(ys[k][0, p0:].float() @ u) / ys[k][0, p0:].float().norm(dim=-1).clamp_min(1e-6) for k, u in dirs.items()]
        scores[q["key"]] = torch.stack(cos).mean(0).cpu().numpy().astype(np.float32)
        ntok += p0 + len(q["body"])
        if n % 25 == 0 or n == len(seqs) - 1:
            print(f"[{n + 1}/{len(seqs)}] {q['key']} len {p0 + len(q['body'])} {ntok / (time.time() - t0):.0f} tok/s "
                  f"{time.time() - t0:.0f}s", flush=True)
    tmp = a.out + ".tmp.npz"
    np.savez_compressed(tmp, **scores)
    Path(tmp).replace(a.out)
    print("saved", a.out, len(scores), "answers", f"{time.time() - t0:.0f}s", "device", dev, "torch", torch.__version__)


# ------------------------------------------------------------------ analysis (CPU)
def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(c - h), 3), round(float(c + h), 3)]


def answer_stat(score: np.ndarray, fire: int | None, sign: float) -> float:
    """Highest smoothed, oriented score from the text fire on (what the veto can ever see); -inf if never flagged."""
    if fire is None:
        return -np.inf
    return float((sign * smooth(score))[fire:].max())


def answer_threshold(stats: list[float], target: float) -> float:
    """Smallest threshold with at most target * n healthy answers above it (switch-on needs score > thr)."""
    s = np.sort(np.asarray(stats))[::-1]
    m = int(np.floor(target * len(s)))
    return float(s[m]) if m < len(s) else -np.inf


def fit(rule: str, train: list[dict], sc: dict) -> dict | None:
    if rule == "text":
        return None
    cal = calibrate([sc[q["key"]] for q in train], train, int(rule.split("_")[-1]) / 100)
    if rule.startswith("token_fpr_"):
        return cal
    healthy = [q for q in train if not q["looping"]]
    cal["thr"] = answer_threshold([answer_stat(sc[q["key"]], q["fire"], cal["sign"]) for q in healthy],
                                  int(rule.split("_")[-1]) / 100)
    cal["fpr"] = None
    return cal


def on_of(q: dict, sc: dict, cal: dict | None) -> int | None:
    return q["fire"] if cal is None else switch_on(sc[q["key"]], q["fire"], cal)


def summarize(seqs: list[dict], ons: dict, correct: dict) -> dict:
    loops = [q for q in seqs if q["looping"]]
    healthy = [q for q in seqs if not q["looping"]]
    hc = [q for q in healthy if correct[q["key"]]]
    k_h = sum(ons[q["key"]] is not None for q in healthy)
    k_hc = sum(ons[q["key"]] is not None for q in hc)
    k_l = sum(ons[q["key"]] is not None for q in loops)
    lag = [ons[q["key"]] - q["onset"] for q in loops if ons[q["key"]] is not None]
    after_text = [ons[q["key"]] - q["fire"] for q in seqs if ons[q["key"]] is not None]
    return {"healthy_on": f"{k_h}/{len(healthy)}", "healthy_on_rate": round(k_h / max(1, len(healthy)), 4),
            "healthy_on_ci95": wilson(k_h, len(healthy)),
            "healthy_correct_on": f"{k_hc}/{len(hc)}", "healthy_correct_on_ci95": wilson(k_hc, len(hc)),
            "loops_on": f"{k_l}/{len(loops)}", "loops_on_rate": round(k_l / max(1, len(loops)), 4),
            "loops_on_ci95": wilson(k_l, len(loops)),
            "median_on_minus_onset": float(np.median(lag)) if lag else None,
            "q90_on_minus_onset": float(np.percentile(lag, 90)) if lag else None,
            "median_on_after_text": float(np.median(after_text)) if after_text else None}


def cmd_analyze(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    eos = set(json.loads((work / "config.json").read_text())["eos"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    seqs = load_all(work, eos)
    z = np.load(a.scores)
    sc = {k: z[k] for k in z.files}
    seqs = [q for q in seqs if q["key"] in sc]
    correct = {}
    for name, rq, op in SPLITS:
        reqs = {r["id"]: r for r in lr.read_jsonl(work / rq)}
        for o in lr.read_jsonl(work / op):
            r = reqs[o["id"]]
            correct[f"{name}:{o['id']}"] = bool(lr.grade(o["token_ids"], o["finish_reason"], r["gold"], tok, eos,
                                                          r["max_tokens"])["correct"])

    res = {"data": {"answers": len(seqs), "loops": sum(q["looping"] for q in seqs),
                    "healthy": sum(not q["looping"] for q in seqs),
                    "healthy_correct": sum((not q["looping"]) and correct[q["key"]] for q in seqs),
                    "healthy_text_fired": sum((not q["looping"]) and q["fire"] is not None for q in seqs),
                    "problems": len({q["group"] for q in seqs}),
                    "per_split": {s: {"n": sum(q["split"] == s for q in seqs),
                                      "healthy": sum(q["split"] == s and not q["looping"] for q in seqs)}
                                  for s, _, _ in SPLITS}}}

    # 1. replication of the generation-time calibration: elicit (token FPR 1%) -> test
    el = [q for q in seqs if q["split"] == "elicit"]
    te = [q for q in seqs if q["split"] == "test"]
    cal = calibrate([sc[q["key"]] for q in el], el, 0.01)
    res["replication_elicit_to_test"] = {"calibration": cal, **summarize(te, {q["key"]: on_of(q, sc, cal) for q in te}, correct),
                                         "generation_run_had": {"healthy_on": "3/23", "switched_on": 180}}

    # 2. grouped 5-fold cross-validation over all splits
    groups = sorted({q["group"] for q in seqs})
    rng = np.random.default_rng(a.seed)
    fold_of = {g: i % FOLDS for i, g in enumerate(rng.permutation(groups))}
    rules = ("text", "token_fpr_1", "token_fpr_5", "answer_5", "answer_10")
    ons = {r: {} for r in rules}
    fits = {r: [] for r in rules}
    for f in range(FOLDS):
        train = [q for q in seqs if fold_of[q["group"]] != f]
        held = [q for q in seqs if fold_of[q["group"]] == f]
        for r in rules:
            c = fit(r, train, sc)
            fits[r].append(None if c is None else {"sign": c["sign"], "thr": round(c["thr"], 5)})
            for q in held:
                ons[r][q["key"]] = on_of(q, sc, c)
    res["cv"] = {r: {**summarize(seqs, ons[r], correct), "fold_fits": fits[r]} for r in rules}

    # 3. token-level separation and a threshold sweep (descriptive, all answers, sign from all data)
    full = calibrate([sc[q["key"]] for q in seqs], seqs, 0.01)
    res["token_auc_all"] = full["auc_calib"]
    stats_h = [answer_stat(sc[q["key"]], q["fire"], full["sign"]) for q in seqs if not q["looping"]]
    sweep = []
    for t in np.unique(np.quantile([s for s in stats_h if np.isfinite(s)], np.linspace(0, 1, 21))):
        c = {**full, "thr": float(t)}
        sweep.append({"thr": round(float(t), 5), **{k: v for k, v in summarize(seqs, {q["key"]: on_of(q, sc, c) for q in seqs},
                                                                                correct).items()
                                                     if k in ("healthy_on", "healthy_correct_on", "loops_on", "median_on_minus_onset")}})
    res["sweep_all"] = sweep

    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1))
    Path(a.out + ".tmp").replace(a.out)
    print(json.dumps({"data": res["data"], "replication": {k: v for k, v in res["replication_elicit_to_test"].items()
                                                           if k != "calibration"}}, indent=1))
    for r in rules:
        m = res["cv"][r]
        print(f"{r:12s} healthy on {m['healthy_on']:>7s} {m['healthy_on_ci95']}  correct-healthy on {m['healthy_correct_on']:>6s}"
              f"  loops on {m['loops_on']:>7s}  on-onset {m['median_on_minus_onset']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--work", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int, default=0, help="shortest N answers only (smoke test)")
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--chunk", type=int, default=2048, help="queries per attention block (exact; memory only)")
    n = sub.add_parser("analyze")
    n.add_argument("--work", required=True)
    n.add_argument("--scores", required=True)
    n.add_argument("--tokenizer", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"score": cmd_score, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
