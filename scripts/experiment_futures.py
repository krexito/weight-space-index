"""EXPERIMENTAL: several futures from every router cut, with the sensor token recording each one.

"Rescue from the inside" could not tell which loops the blind patch rescues: every base answer the router cut was
resumed ONCE per arm, so "rescued / not rescued" is one coin flip of the sampler. Here every cut of the discover_blind
GSM8K test run (requests_test_moi.jsonl, the text-timed MoI plan: 189 cuts) is resumed K times with the patched model
(`patch`) and K times with the unpatched one (`restart`), same seeds in both arms (sample k uses seed + 1000 k), early
loop stop on. The final-layer sensor token (experiment_final_sensor.py / experiment_sensor_vllm.py: elicit direction in
spare vocabulary row 248070, read as a raw logit, blocked from sampling) records every generated token.

Questions:
  1. rescue or luck: per-cut rescue rate over K samples per arm; how many cuts are always / never / sometimes rescued;
     patch minus restart on the per-cut rates (bootstrap over cuts), next to the single-shot run (55.5 / 35.5 %).
  2. early warning: does the sensor over the first W tokens after resuming (W = 64, 128, 256) predict which
     continuations loop again or end wrong? Compared with the text alone (share of repeated 16-grams in the same W).

  plan     (CPU)  python scripts/experiment_futures.py plan   --work W --k 8
  gen      (GPU)  python scripts/experiment_futures.py gen    --work W --base-dir M --dirs D.npz --ck-dir /root
  report   (CPU)  python scripts/experiment_futures.py report --work W --tokenizer M --out R.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import moi_loop_router_vllm as lr  # noqa: E402

ROW = 248070          # spare vocabulary row (Qwen3.5: tokenizer ids end at 248069)
SCALE = 1.0 / 200.0   # sensor row = SCALE * w, so the raw logit stays small and a -100 bias always blocks it
BIAS = -100.0
SEED_STEP = 1000
PLAN = "requests_futures.jsonl"
ARMS = ("patch", "restart")
WINDOWS = (64, 128, 256)
NGRAM = 16
SMOOTH = 32


# ------------------------------------------------------------------ plan (pure)
def plan_rows(moi_rows: list[dict], k: int) -> list[dict]:
    """K resume requests per cut: same prompt (prefix included) and prefix penalty, seeds seed + 1000 j."""
    out = []
    for r in moi_rows:
        bias = {str(t): float(v) for t, v in r.get("logit_bias", {}).items()}
        bias[str(ROW)] = BIAS
        for j in range(k):
            out.append({"id": r["id"], "k": j, "prompt_ids": r["prompt_ids"], "prefix_len": r["prefix_len"],
                        "max_tokens": r["max_tokens"], "seed": int(r["seed"]) + SEED_STEP * j, "logit_bias": bias})
    return out


def cmd_plan(a) -> None:
    work = Path(a.work)
    rows = plan_rows(lr.read_jsonl(work / "requests_test_moi.jsonl"), a.k)
    lr.write_jsonl(work / PLAN, rows)
    print(f"{len(rows)} requests ({len(rows) // a.k} cuts x {a.k})")


# ------------------------------------------------------------------ gen (vLLM)
def add_sensor(src: Path, dst: Path, w: np.ndarray) -> None:
    """Copy a checkpoint and write SCALE * w into row ROW of its vocabulary matrices (tied embedding / lm_head)."""
    import torch
    from safetensors.torch import load_file, save_file

    if src.resolve() != dst.resolve():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".cache", "*.lock"))
    wt = torch.from_numpy(w * SCALE)
    changed = []
    for f in sorted(dst.glob("*.safetensors")):
        t = {k: v.clone() for k, v in load_file(str(f)).items()}  # clone: Windows cannot overwrite a mapped file
        hit = False
        for kname, v in t.items():
            if (kname.endswith("embed_tokens.weight") or kname.endswith("lm_head.weight")) and v.ndim == 2 \
                    and v.shape[0] > ROW and v.shape[1] == wt.numel():
                v[ROW] = wt.to(v.dtype)
                changed.append(kname)
                hit = True
        if hit:
            tmp = f.with_suffix(".tmp")
            save_file(t, str(tmp), metadata={"format": "pt"})
            os.replace(tmp, f)
    assert changed, f"no vocabulary matrix in {dst}"


def sensor_vector(dirs_npz: str) -> np.ndarray:
    z = np.load(dirs_npz)
    return z["W"][[str(x) for x in z["dirs"]].index("elicit")].astype(np.float64)


def cmd_gen(a) -> None:
    import discover_blind as blind
    import discover_loop_fix as dl
    import loop_kill_v2
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    cfg["base_dir"] = str(a.base_dir)
    dl.cfg_save(work, cfg)
    c = json.loads((work / "frozen.json").read_text())["choice"]
    w = sensor_vector(a.dirs)
    ck = Path(a.ck_dir)
    models = {"patch": ck / "patch_sensor", "restart": ck / "base_sensor"}
    if not (models["restart"] / "config.json").exists():
        add_sensor(Path(a.base_dir), models["restart"], w)
    if not (models["patch"] / "config.json").exists():
        blind.build_checkpoint(work, c["layers"], c["neurons"], c["alpha"], "wsi", models["patch"])
        add_sensor(models["patch"], models["patch"], w)
    reqs = lr.read_jsonl(work / PLAN)
    if a.limit:
        reqs = reqs[: a.limit]
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    max_len = max(len(r["prompt_ids"]) + r["max_tokens"] for r in reqs) + 16
    kill_eos = cfg["eos"][-1]
    for arm in ARMS:
        out_tok, out_sen = work / f"futures_{arm}.jsonl", work / f"futures_{arm}_sensor.npz"
        if out_tok.exists() and out_sen.exists():
            print(f"{arm}: already there")
            continue
        proc = loop_kill_v2.processor()
        llm = LLM(model=str(models[arm]), tokenizer=str(a.base_dir), dtype="bfloat16", max_model_len=max_len,
                  gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=cfg["seed"], trust_remote_code=True,
                  logprobs_mode="raw_logits", enable_chunked_prefill=True, max_num_seqs=a.max_num_seqs,
                  limit_mm_per_prompt={"image": 0, "video": 0},
                  **({"logits_processors": [proc]} if proc is not None else {}))
        sps = [SamplingParams(**lr.sampling_for(cfg), max_tokens=r["max_tokens"], seed=r["seed"],
                              stop_token_ids=cfg["eos"], logit_bias={int(t): v for t, v in r["logit_bias"].items()},
                              logprob_token_ids=[ROW], extra_args={"loop_kill_eos": kill_eos} if proc else None)
               for r in reqs]
        t0 = time.time()
        rows, sens = [], {}
        # saved in blocks, so a run cut by a timeout keeps every finished block and a rerun resumes from there
        for b0 in range(0, len(reqs), a.chunk or len(reqs)):
            b1 = min(len(reqs), b0 + (a.chunk or len(reqs)))
            p_tok, p_sen = work / f"futures_{arm}.part{b0:05d}.jsonl", work / f"futures_{arm}.part{b0:05d}_sensor.npz"
            if p_tok.exists() and p_sen.exists():
                rows += lr.read_jsonl(p_tok)
                sens.update(dict(np.load(p_sen)))
                print(f"[{arm}] block {b0}-{b1}: already there", flush=True)
                continue
            outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in reqs[b0:b1]], sps[b0:b1],
                                use_tqdm=True)
            b_rows, b_sens = [], {}
            for r, o in zip(reqs[b0:b1], outs):
                cc = o.outputs[0]
                b_rows.append({"id": r["id"], "k": r["k"], "token_ids": list(cc.token_ids),
                               "finish_reason": cc.finish_reason})
                vals = [(s[ROW].logprob if s is not None and ROW in s else np.nan) for s in (cc.logprobs or [])]
                b_sens[f"{r['id']}|{r['k']}"] = (np.asarray(vals, np.float32) / SCALE).astype(np.float32)
            lr.write_jsonl(p_tok.with_suffix(".tmp"), b_rows)
            np.savez_compressed(str(p_sen.with_suffix(".tmp.npz")), **b_sens)
            os.replace(p_sen.with_suffix(".tmp.npz"), p_sen)
            os.replace(p_tok.with_suffix(".tmp"), p_tok)
            rows += b_rows
            sens.update(b_sens)
            print(f"[{arm}] block {b0}-{b1} saved, {(time.time() - t0) / 60:.1f} min", flush=True)
        lr.write_jsonl(out_tok, rows)
        np.savez_compressed(str(out_sen), **sens)
        ntok = sum(len(x["token_ids"]) for x in rows)
        print(f"[{arm}] {len(rows)} continuations, {ntok:,} tokens, {(time.time() - t0) / 60:.1f} min "
              f"({ntok / (time.time() - t0):,.0f} tok/s), early stop {'on' if proc else 'OFF'}, "
              f"sampled sensor row {sum(ROW in x['token_ids'] for x in rows)}", flush=True)
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ report (CPU)
def smooth(x: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    c = np.cumsum(np.insert(np.asarray(x, np.float64), 0, 0.0))
    k = np.arange(len(x))
    lo = np.maximum(0, k - w + 1)
    return (c[k + 1] - c[lo]) / (k + 1 - lo)


def repeat_share(tokens: list[int], n: int = NGRAM) -> float:
    """Share of positions whose n-gram ending there already appeared earlier in the same window (text-only signal)."""
    seen, rep = set(), 0
    for i in range(n - 1, len(tokens)):
        g = tuple(tokens[i - n + 1:i + 1])
        rep += g in seen
        seen.add(g)
    return rep / max(1, len(tokens) - n + 1)


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auc(pos_by_g: dict, neg_by_g: dict, groups: list, n: int = 500, seed: int = 0) -> list[float]:
    """95% CI of the AUC, resampling cuts (all K continuations of a cut move together)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        gs = rng.choice(groups, len(groups))
        p = [v for g in gs for v in pos_by_g.get(g, [])]
        q = [v for g in gs for v in neg_by_g.get(g, [])]
        out.append(auc(p, q))
    return [round(float(np.nanpercentile(out, 2.5)), 3), round(float(np.nanpercentile(out, 97.5)), 3)]


def rate_table(rates: np.ndarray, k: int) -> dict:
    return {"always": int((rates == 1).sum()), "never": int((rates == 0).sum()),
            "sometimes": int(((rates > 0) & (rates < 1)).sum()), "mean": round(float(rates.mean()), 4),
            "k": k}


def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    import rescue_answer as ra

    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    eos = set(cfg["eos"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    plan = {(r["id"], r["k"]): r for r in lr.read_jsonl(work / PLAN)}
    grader = ra.Grader("gsm8k")
    K = 1 + max(k for _, k in plan)
    cuts = sorted({i for i, _ in plan})
    res = {"cuts": len(cuts), "k": K, "arms": {}}
    ok, ok_r, reloop, feats = {}, {}, {}, {}
    for arm in ARMS:
        rows = {(o["id"], o["k"]): o for o in lr.read_jsonl(work / f"futures_{arm}.jsonl")}
        sen = np.load(work / f"futures_{arm}_sensor.npz")
        for j in range(K):
            seqs = {}
            for i in cuts:
                o, p = rows[(i, j)], plan[(i, j)]
                prefix = p["prompt_ids"][len(reqs[i]["prompt_ids"]):]
                seqs[i] = (prefix + o["token_ids"], o["finish_reason"])
            sc = ra.score_arm(seqs, reqs, tok, eos, grader)
            for row in sc["rows"]:
                key = (arm, row["id"], j)
                ok[key] = bool(row["correct"])
                ok_r[key] = bool(row["correct"] or row.get("strict_ok"))
                reloop[key] = bool(row["killed"])
                cont = rows[(row["id"], j)]["token_ids"]
                s = sen[f"{row['id']}|{j}"]
                feats[key] = {f"sensor_{W}": float(np.nanmean(s[:W])) if len(s) else np.nan for W in WINDOWS}
                feats[key].update({f"text_{W}": repeat_share(cont[:W]) for W in WINDOWS})
                feats[key]["sensor_peak_256"] = float(np.nanmax(smooth(np.nan_to_num(s[:256])))) if len(s) else np.nan
                feats[key]["n_tokens"] = len(cont)
        rates = np.array([[ok[(arm, i, j)] for j in range(K)] for i in cuts], float).mean(1)
        rates_r = np.array([[ok_r[(arm, i, j)] for j in range(K)] for i in cuts], float).mean(1)
        rel = np.array([[reloop[(arm, i, j)] for j in range(K)] for i in cuts], float).mean(1)
        res["arms"][arm] = {"strict": rate_table(rates, K), "rescued": rate_table(rates_r, K),
                            "reloop_rate": round(float(rel.mean()), 4),
                            "single_shot_k0_acc_on_cuts": round(float(np.mean([ok[(arm, i, 0)] for i in cuts])), 4),
                            "per_cut_rate": {str(i): float(x) for i, x in zip(cuts, rates)}}

    # 1. patch minus restart on per-cut rates (bootstrap over cuts), and how noisy one sample is
    rng = np.random.default_rng(0)
    for lvl, src in (("strict", ok), ("rescued", ok_r)):
        rp = np.array([np.mean([src[("patch", i, j)] for j in range(K)]) for i in cuts])
        rr = np.array([np.mean([src[("restart", i, j)] for j in range(K)]) for i in cuts])
        d = rp - rr
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        res[f"patch_minus_restart_{lvl}"] = {"diff_on_cuts": round(float(d.mean()), 4),
                                             "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                                      round(float(np.percentile(bs, 97.5)), 4)],
                                             "cuts_patch_better": int((d > 0).sum()), "cuts_restart_better": int((d < 0).sum())}
    # agreement of two single samples (k=0 vs k=1) on the patch-minus-restart sign per cut
    if K >= 2:
        s0 = np.array([ok[("patch", i, 0)] - ok[("restart", i, 0)] for i in cuts])
        s1 = np.array([ok[("patch", i, 1)] - ok[("restart", i, 1)] for i in cuts])
        res["single_shot_label_agreement"] = {"same_rescue_label_patch_k0_k1": round(float(np.mean(
            [ok[("patch", i, 0)] == ok[("patch", i, 1)] for i in cuts])), 4),
            "same_sign_patch_minus_restart_k0_k1": round(float(np.mean(np.sign(s0) == np.sign(s1))), 4)}
    # variance: share of the variance of one patched sample that is between cuts (the rest is sampling luck)
    y = np.array([[ok[("patch", i, j)] for j in range(K)] for i in cuts], float)
    within = float(y.var(axis=1, ddof=1).mean()) if K > 1 else float("nan")
    total = float(y.var(ddof=1))
    res["patch_variance_between_cuts_share"] = round(1 - within / total, 4) if total > 0 else None

    # 2. early warning over the first W tokens after resuming: AUC for "loops again" and for "ends wrong"
    ew = {}
    for arm in ARMS:
        keys = [(arm, i, j) for i in cuts for j in range(K)]
        for target, lab in (("reloop", reloop), ("wrong", {kk: not v for kk, v in ok.items()})):
            for fname in [f"sensor_{W}" for W in WINDOWS] + [f"text_{W}" for W in WINDOWS] + ["sensor_peak_256"]:
                pos_g, neg_g = {}, {}
                for kk in keys:
                    v = feats[kk][fname]
                    if not np.isfinite(v):
                        continue
                    (pos_g if lab[kk] else neg_g).setdefault(kk[1], []).append(v)
                p = [v for g in pos_g.values() for v in g]
                q = [v for g in neg_g.values() for v in g]
                ew[f"{arm}|{target}|{fname}"] = {"auc": round(auc(p, q), 4), "ci95": boot_auc(pos_g, neg_g, cuts),
                                                 "n_pos": len(p), "n_neg": len(q)}
    res["early_warning"] = ew
    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1))
    Path(a.out + ".tmp").replace(a.out)

    print(f"{len(cuts)} cuts x {K} samples per arm")
    for arm in ARMS:
        v = res["arms"][arm]
        print(f"{arm:8s} strict mean {v['strict']['mean']:.1%} (always {v['strict']['always']}, never "
              f"{v['strict']['never']}, sometimes {v['strict']['sometimes']})  rescued mean {v['rescued']['mean']:.1%}  "
              f"loops again {v['reloop_rate']:.1%}  k0 {v['single_shot_k0_acc_on_cuts']:.1%}")
    for lvl in ("strict", "rescued"):
        m = res[f"patch_minus_restart_{lvl}"]
        print(f"patch - restart ({lvl}): {m['diff_on_cuts']:+.1%} CI {m['ci95']}  cuts {m['cuts_patch_better']} better / "
              f"{m['cuts_restart_better']} worse")
    print("single-shot agreement:", res.get("single_shot_label_agreement"), "| between-cut variance share:",
          res["patch_variance_between_cuts_share"])
    for kname, v in ew.items():
        print(f"  {kname:32s} AUC {v['auc']:.3f} {v['ci95']}  ({v['n_pos']}/{v['n_neg']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--work", required=True)
    p.add_argument("--k", type=int, default=8)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--base-dir", required=True)
    g.add_argument("--dirs", required=True)
    g.add_argument("--ck-dir", default="/root")
    g.add_argument("--gpu-mem", type=float, default=0.90)
    g.add_argument("--max-num-seqs", type=int, default=256)
    g.add_argument("--limit", type=int, default=0, help="first N requests only (smoke test)")
    g.add_argument("--chunk", type=int, default=504, help="requests per saved block (0 = one block per arm)")
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"plan": cmd_plan, "gen": cmd_gen, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
