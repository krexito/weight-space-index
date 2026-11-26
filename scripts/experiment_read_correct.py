"""EXPERIMENTAL: can a read of the model's internals tell a correct NORMAL answer from a wrong one, beyond its own
confidence? (research/synthesis.md, test 1: "read to improve without editing")

Data: saved base answers (no generation). Only answers that finish on their own (no loop kill, not truncated) are kept,
so loops cannot drive the signal. Label: graded correct / wrong.

  score    python scripts/experiment_read_correct.py score   --work W --model M --out feats.npz [--cpu] [--limit N]
  analyze  python scripts/experiment_read_correct.py analyze --work W --feats feats.npz --tokenizer M --out R.json

Per answer (one fp32 forward over prompt + answer):
  controls (no fitting): answer length; mean token logprob of the answer (all / last 64 tokens); min logprob; mean
           entropy; the 6 saved final-layer sensor readouts (benchmarks/final_sensor) on the last 64 tokens
  probes:  hidden state at 6 depths (PROBE_FRACS of the layer count) and after the final norm, mean over the last 64 answer tokens and over the
           whole answer -> standardize, PCA 16 (fit on the training fold), logistic C=1. Fixed before looking.
Evaluation: 5-fold cross-validation grouped by prompt, repeated 10 times; out-of-fold AUC with a bootstrap CI over
answers, and the paired AUC difference probe - best single control and (controls + probe) - controls.
Pass: a probe beats the best control with the CI of the difference above 0 (else: confidence is all a read gives).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

for _m in ("fla", "causal_conv1d"):  # same exact GDN path as the validated fp32 recompute
    sys.modules.setdefault(_m, None)

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import moi_loop_router_vllm as lr  # noqa: E402

SPLITS = (("elicit", "requests_elicit.jsonl", "outputs_elicit.jsonl"),
          ("dev1", "requests_dev1.jsonl", "outputs_dev1_base.jsonl"),
          ("dev2", "requests_dev2.jsonl", "outputs_dev2_base.jsonl"),
          ("dev", "requests_dev.jsonl", "outputs_dev_base.jsonl"),
          ("test", "requests_test.jsonl", "outputs_test_base.jsonl"))
PROBE_FRACS = (0.125, 0.29, 0.46, 0.63, 0.79, 0.96)  # depth fractions; 24 layers -> 3 7 11 15 19 23
TAIL = 64
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}  # Qwen2.5-Instruct card defaults
SENSOR = HERE.parent / "benchmarks" / "final_sensor" / "sensor_directions_qwen35_08b_gsm8k.npz"


def load_finished(work: Path, eos: set[int]) -> list[dict]:
    out = []
    for name, rq, op in SPLITS:
        if not (work / rq).exists() or not (work / op).exists():
            continue
        reqs = {r["id"]: r for r in lr.read_jsonl(work / rq)}
        for o in lr.read_jsonl(work / op):
            st = lr.analyse(o["token_ids"], eos)
            if st["kill"] is not None or o["finish_reason"] != "stop" or len(st["body"]) < TAIL:
                continue
            r = reqs[o["id"]]
            out.append({"key": f"{name}:{o['id']}", "split": name, "id": o["id"], "prompt": r["prompt_ids"],
                        "body": st["body"], "resp": o["token_ids"], "finish": o["finish_reason"],
                        "group": hashlib.sha1(np.asarray(r["prompt_ids"], np.int64).tobytes()).hexdigest()[:16]})
    return out


def cmd_prepare(a) -> None:
    """MATH-500 (all 500) as requests_test.jsonl for an instruct model without a thinking mode."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(x) for x in Path(hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl",
                                                         repo_type="dataset")).read_text().splitlines() if x.strip()]
    if a.limit:
        rows = rows[: a.limit]
    reqs = []
    for i, r in enumerate(rows):
        text = tok.apply_chat_template([{"role": "user", "content": r["problem"] + MATH_SUFFIX}], tokenize=False,
                                       add_generation_prompt=True)
        reqs.append({"id": i, "uid": r.get("unique_id", str(i)), "level": r.get("level"), "subject": r.get("subject"),
                     "prompt_ids": tok.encode(text, add_special_tokens=False), "gold": r["answer"],
                     "max_tokens": a.max_tokens, "seed": 1000 + i})
    lr.write_jsonl(work / "requests_test.jsonl", reqs)
    eos = sorted({t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                              tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(t, int) and t >= 0})
    (work / "config.json").write_text(json.dumps({"eos": eos, "model": a.model, "max_new": a.max_tokens,
                                                  "sampling": SAMPLING, "dataset": "HuggingFaceH4/MATH-500"}, indent=1))
    print("prepared", len(reqs), "requests; eos", eos)


def cmd_gen(a) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    work = Path(a.work)
    reqs = lr.read_jsonl(work / "requests_test.jsonl")
    llm = LLM(a.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=a.max_len, seed=0)
    sps = [SamplingParams(**SAMPLING, max_tokens=r["max_tokens"], seed=r["seed"]) for r in reqs]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in reqs], sps)
    rows = [{"id": r["id"], "token_ids": list(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason}
            for r, o in zip(reqs, outs)]
    lr.write_jsonl(work / "outputs_test_base.jsonl", rows)
    ntok = sum(len(x["token_ids"]) for x in rows)
    print(f"generated {len(rows)} answers, {ntok} tokens, {time.time() - t0:.0f}s")


def cmd_score(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    eos = set(json.loads((work / "config.json").read_text())["eos"])
    seqs = sorted(load_finished(work, eos), key=lambda q: len(q["prompt"]) + len(q["body"]))
    if a.limit:
        seqs = seqs[: a.limit]
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(a.threads)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    n_layers = model.config.get_text_config().num_hidden_layers
    probe_layers = sorted({round(f * (n_layers - 1)) for f in PROBE_FRACS})
    W = np.load(SENSOR)["W"]
    sensor = torch.from_numpy(W).float().to(dev) if W.shape[1] == model.config.get_text_config().hidden_size else None
    head = model.get_output_embeddings()
    print("probe layers", probe_layers, "sensor", sensor is not None, flush=True)
    feats: dict[str, np.ndarray] = {}
    t0, ntok = time.time(), 0
    for n, q in enumerate(seqs):
        ids = torch.tensor([q["prompt"] + q["body"]], device=dev)
        with torch.no_grad():
            out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states  # embeddings + one per layer; the last one is after the final norm
            p0, T = len(q["prompt"]), ids.shape[1]
            lp_all, ent_all = [], []
            for s in range(p0 - 1, T - 1, 256):  # position s predicts token s + 1
                e = min(T - 1, s + 256)
                logp = torch.log_softmax(head(hs[-1][0, s:e]).float(), -1)
                lp_all.append(logp.gather(1, ids[0, s + 1:e + 1, None])[:, 0])
                ent_all.append(-(logp.exp() * logp).sum(-1))
            lp, ent = torch.cat(lp_all), torch.cat(ent_all)
            k = q["key"]
            feats[f"{k}|conf"] = np.array([len(q["body"])] + [float(v) for v in (
                lp.mean(), lp[-TAIL:].mean(), lp.min(), ent.mean(), ent[-TAIL:].mean())], np.float32)
            last = hs[-1][0, p0:]
            if sensor is not None:
                feats[f"{k}|sensor"] = (last[-TAIL:] @ sensor.T).mean(0).cpu().numpy().astype(np.float32)
            for li in probe_layers:
                h = hs[li + 1][0, p0:]
                feats[f"{k}|L{li}_tail"] = h[-TAIL:].mean(0).cpu().numpy().astype(np.float32)
                feats[f"{k}|L{li}_all"] = h.mean(0).cpu().numpy().astype(np.float32)
            feats[f"{k}|final_tail"] = last[-TAIL:].mean(0).cpu().numpy().astype(np.float32)
            feats[f"{k}|final_all"] = last.mean(0).cpu().numpy().astype(np.float32)
        ntok += T
        if n % 10 == 0 or n == len(seqs) - 1:
            print(f"[{n + 1}/{len(seqs)}] {k} len {T} {ntok / (time.time() - t0):.0f} tok/s {time.time() - t0:.0f}s",
                  flush=True)
    tmp = a.out + ".tmp.npz"
    np.savez_compressed(tmp, **feats)
    Path(tmp).replace(a.out)
    print("saved", a.out, len(seqs), "answers", f"{time.time() - t0:.0f}s", dev)


# ------------------------------------------------------------------ analysis (CPU)
def auc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata

    r = rankdata(s)
    npos = int(y.sum())
    nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, reps: int = 10, pca: int | None = 16) -> np.ndarray:
    """Out-of-fold P(correct), averaged over `reps` grouped 5-fold splits."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros(len(y))
    ug = np.unique(groups)
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        gmap = dict(zip(ug, rng.permutation(len(ug))))  # shuffles which groups share a fold
        g = np.array([gmap[x] for x in groups])
        for tr, te in GroupKFold(5).split(X, y, g):
            steps = [StandardScaler()] + ([PCA(min(pca, len(tr) - 1, X.shape[1]))] if pca else [])
            m = make_pipeline(*steps, LogisticRegression(C=1.0, max_iter=2000))
            m.fit(X[tr], y[tr])
            pred[te] += m.predict_proba(X[te])[:, 1] / reps
    return pred


def boot_auc(y, s, s2=None, n=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    base = auc(y, s) - (auc(y, s2) if s2 is not None else 0)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        bs.append(auc(y[i], s[i]) - (auc(y[i], s2[i]) if s2 is not None else 0))
    return {"value": round(base, 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3),
                                               round(float(np.percentile(bs, 97.5)), 3)]}


def cmd_analyze(a) -> None:
    from transformers import AutoTokenizer

    import rescue_answer as ra

    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    eos = set(cfg["eos"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    z = np.load(a.feats)
    keys = sorted({k.split("|")[0] for k in z.files})
    seqs = {q["key"]: q for q in load_finished(work, eos) if q["key"] in keys}
    grader = ra.Grader(a.dataset)
    reqs = {}
    for name, rq, _ in SPLITS:
        if (work / rq).exists():
            reqs.update({f"{name}:{r['id']}": r for r in lr.read_jsonl(work / rq)})
    y = []
    for k in keys:
        r = reqs[k]
        g = grader.grade(seqs[k]["resp"], seqs[k]["finish"], r["gold"], tok, eos, r["max_tokens"])
        y.append(int(g["correct"]))
    y = np.array(y)
    groups = np.array([seqs[k]["group"] for k in keys])
    conf = np.stack([z[f"{k}|conf"] for k in keys])
    names = ["neg_length", "mean_logprob", "tail_logprob", "min_logprob", "neg_mean_entropy", "neg_tail_entropy"]
    sign = np.array([-1, 1, 1, 1, -1, -1])
    res = {"answers": len(keys), "correct": int(y.sum()), "splits": {}, "controls": {}, "probes": {}}
    for k in keys:
        res["splits"][seqs[k]["split"]] = res["splits"].get(seqs[k]["split"], 0) + 1
    for j, nm in enumerate(names):
        res["controls"][nm] = boot_auc(y, sign[j] * conf[:, j])
    best = max(names, key=lambda nm: res["controls"][nm]["value"])
    best_s = sign[names.index(best)] * conf[:, names.index(best)]
    ctrl_X = conf * sign
    if f"{keys[0]}|sensor" in z.files:
        ctrl_X = np.concatenate([ctrl_X, np.stack([z[f"{k}|sensor"] for k in keys])], 1)
    ctrl_p = oof(ctrl_X, y, groups, pca=None)
    res["controls"]["all_controls_fitted"] = boot_auc(y, ctrl_p)
    res["best_single_control"] = best
    layers = sorted({int(f.split("|")[1].split("_")[0][1:]) for f in z.files if f.split("|")[1].startswith("L")})
    feats = [f"L{li}_{p}" for li in layers for p in ("tail", "all")] + ["final_tail", "final_all"]
    res["probe_layers"] = layers
    for f in feats:
        X = np.stack([z[f"{k}|{f}"] for k in keys])
        p = oof(X, y, groups)
        pc = oof(np.concatenate([ctrl_X, X], 1), y, groups)
        res["probes"][f] = {"auc": boot_auc(y, p), "minus_best_control": boot_auc(y, p, best_s),
                            "controls_plus_probe_minus_controls": boot_auc(y, pc, ctrl_p)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{res['answers']} finished answers, {res['correct']} correct; splits {res['splits']}")
    for nm, v in res["controls"].items():
        print(f"  control {nm:22s} AUC {v['value']:.3f} {v['ci95']}")
    print(f"  best single control: {best}")
    for f, v in res["probes"].items():
        print(f"  probe {f:12s} AUC {v['auc']['value']:.3f} {v['auc']['ci95']}  vs best control "
              f"{v['minus_best_control']['value']:+.3f} {v['minus_best_control']['ci95']}  added to controls "
              f"{v['controls_plus_probe_minus_controls']['value']:+.3f} {v['controls_plus_probe_minus_controls']['ci95']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--work", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--max-tokens", type=int, default=4096)
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("gen")
    s.add_argument("--work", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--max-len", type=int, default=6144)
    s = sub.add_parser("score")
    s.add_argument("--work", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--threads", type=int, default=16)
    s = sub.add_parser("analyze")
    s.add_argument("--work", required=True)
    s.add_argument("--feats", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "gen": cmd_gen, "score": cmd_score, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
