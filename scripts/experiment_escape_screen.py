"""EXPERIMENTAL: an escape test as the selection metric for loop patches (MATH-500, Qwen3.5-0.8B).

The 1-forward-pass screen ranks the MATH-500 blind pick L10_n256_a12 (exam 10.7%) above MIT p50c (28.7%) and our GSM8K
blind patch (20.7%) (experiment_screen_known_patches.py): suppressing the loop's own tokens is not escaping the loop.
Escape test: take the DISCOVERY split (elicit) base answers of the MATH-500 run, cut each where the router fires
(lr.analyse, as plan_moi does), and let every patch continue from the cut for at most --budget tokens; the unpatched
base continuing from the same cut is the restart arm. Per arm: escaped (no loop kill in the continuation), answered
(a boxed answer), correct. A good selection metric orders the six known patches like the exam.

  prepare (CPU)  python scripts/experiment_escape_screen.py prepare --math-work M --out-dir E --budget 4096
  build          python scripts/experiment_escape_screen.py build   --base B --math-work M --blind-work BW --edit-dir D --out-root CK
  gen            python scripts/experiment_escape_screen.py gen     --work E --model CK/p50c --out outputs_p50c.jsonl
  report (CPU)   python scripts/experiment_escape_screen.py report  --work E --out R.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

EXAM = {"p50c": 0.2867, "blind_n16_a12": 0.2067, "blind_n64_a6": 0.16, "math_L10_n256_a12": 0.1067,
        "blind_n64_a12": 0.1067, "random_L10_n256_a12": 0.0533}
ARMS = ["restart"] + list(EXAM)


def cmd_prepare(a) -> None:
    import moi_loop_router_vllm as lr

    mw, out = Path(a.math_work), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((mw / "config.json").read_text())
    eos = set(cfg["eos"])
    pp = float(lr.sampling_for(cfg).get("presence_penalty", 0.0))
    reqs = {r["id"]: r for r in lr.read_jsonl(mw / "requests_elicit.jsonl")}
    cont = []
    for o in lr.read_jsonl(mw / "outputs_elicit.jsonl"):
        st = lr.analyse(o["token_ids"], eos)
        if st["fire"] is None:
            continue
        prefix = st["body"][: st["fire"]]
        r = reqs[o["id"]]
        cont.append({"id": o["id"], "gold": r["gold"], "level": r.get("level", 0), "base_looped": st["kill"] is not None,
                     "prompt_ids": r["prompt_ids"] + prefix, "prefix_len": len(prefix), "max_tokens": a.budget,
                     "seed": r["seed"] + 7, "logit_bias": {str(t): -pp for t in sorted(set(prefix))} if pp else {}})
    lr.write_jsonl(out / "requests_escape.jsonl", cont)
    c2 = dict(cfg)
    c2["base_dir"] = a.base_dir or cfg["base_dir"]
    (out / "config.json").write_text(json.dumps(c2, indent=1))
    print(f"{len(cont)} cuts ({sum(c['base_looped'] for c in cont)} from looping answers), median prefix "
          f"{int(np.median([c['prefix_len'] for c in cont]))}, budget {a.budget}")


def cmd_build(a) -> None:
    import discover_blind as blind
    import moi_loop_router_vllm as lr

    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "p50c").exists():
        lr.build_p50c(Path(a.base), Path(a.edit_dir), root / "p50c")
    for tag, src, specs in (("bw", Path(a.blind_work), {"blind_n16_a12": ([13, 14], 16, 12.0, "wsi"),
                                                         "blind_n64_a12": ([13, 14], 64, 12.0, "wsi"),
                                                         "blind_n64_a6": ([13, 14], 64, 6.0, "wsi")}),
                            ("mw", Path(a.math_work), {"math_L10_n256_a12": ([10], 256, 12.0, "wsi"),
                                                       "random_L10_n256_a12": ([10], 256, 12.0, "random")})):
        w = root / f"_{tag}"
        w.mkdir(exist_ok=True)
        for f in ("diagnosis_blind.json", "blind_directions.npz"):
            shutil.copy2(src / f, w / f)
        cfg = json.loads((src / "config.json").read_text())
        cfg["base_dir"] = str(a.base)
        (w / "config.json").write_text(json.dumps(cfg))
        for name, (layers, n, alpha, kind) in specs.items():
            if not (root / name).exists():
                blind.build_checkpoint(w, layers, n, alpha, kind, root / name)
            print("built", name, flush=True)


def cmd_gen(a) -> None:
    import discover_fast as fast
    import discover_loop_fix as dl
    import discover_math500_blind as dm
    import moi_loop_router_vllm as lr

    work = Path(a.work)
    out = work / a.out
    if out.exists():
        print("exists", out)
        return
    cfg = dl.cfg_load(work)
    reqs = lr.read_jsonl(work / "requests_escape.jsonl")
    if a.limit:
        reqs = [{**r, "max_tokens": 256} for r in reqs[: a.limit]]
    max_len = max(len(r["prompt_ids"]) + r["max_tokens"] for r in reqs) + 16
    t0 = time.time()
    llm, kill = dm._llm_fast(a.model, cfg, max_len, "bfloat16", a.gpu_mem)
    rows, kill = fast._generate(llm, cfg, reqs, kill)
    lr.write_jsonl(out.with_suffix(".tmp"), rows)
    out.with_suffix(".tmp").replace(out)
    ntok = sum(len(x["token_ids"]) for x in rows)
    print(f"[gen {a.out}] {len(rows)} seqs {ntok:,} tokens {(time.time() - t0) / 60:.1f} min early_stop={kill}", flush=True)


def wilson(k, n):
    if n == 0:
        return [0.0, 0.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def cmd_report(a) -> None:
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer

    import discover_math500_blind as dm
    import moi_loop_router_vllm as lr

    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(a.tokenizer or cfg["base_dir"])
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_escape.jsonl")}
    res = {"n_cuts": len(reqs), "budget": next(iter(reqs.values()))["max_tokens"], "exam_accuracy": EXAM, "arms": {},
           "subset": {}}
    per = {}
    for arm in ARMS:
        f = work / f"outputs_{arm}.jsonl"
        if not f.exists():
            continue
        rows = []
        for o in lr.read_jsonl(f):
            r = reqs[o["id"]]
            g = dm.grade_math(o["token_ids"], o["finish_reason"], r["gold"], tok, eos, r["max_tokens"])
            rows.append({"id": o["id"], "base_looped": r["base_looped"], "escaped": not g["looped_killed"],
                         "answered": g["pred"] is not None and not g["looped_killed"], "correct": g["correct"],
                         "n_tokens": g["n_tokens"]})
        per[arm] = {r["id"]: r for r in rows}
        n = len(rows)
        res["arms"][arm] = {m: round(sum(r[m] for r in rows) / n, 4) for m in ("escaped", "answered", "correct")} | {
            "n": n, "mean_tokens": round(float(np.mean([r["n_tokens"] for r in rows])), 1),
            "escaped_ci95": wilson(sum(r["escaped"] for r in rows), n)}
        lp = [r for r in rows if r["base_looped"]]
        res["subset"][arm] = {"n_looping_cuts": len(lp)} | {
            m: round(sum(r[m] for r in lp) / max(1, len(lp)), 4) for m in ("escaped", "answered", "correct")}
    known = [k for k in EXAM if k in res["arms"]]
    res["spearman_vs_exam"] = {}
    for scope, tab in (("all_cuts", res["arms"]), ("looping_cuts", res["subset"])):
        for m in ("escaped", "answered", "correct"):
            res["spearman_vs_exam"][f"{scope}.{m}"] = round(float(spearmanr([tab[k][m] for k in known],
                                                                              [EXAM[k] for k in known]).correlation), 3)
    if "restart" in per:  # paired vs restart, on escaped
        res["paired_escaped_vs_restart"] = {}
        for k in known:
            ids = sorted(set(per[k]) & set(per["restart"]))
            x = np.array([per[k][i]["escaped"] for i in ids], float) - np.array([per["restart"][i]["escaped"] for i in ids], float)
            rng = np.random.default_rng(0)
            bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]
            res["paired_escaped_vs_restart"][k] = {"diff": round(float(x.mean()), 4),
                                                   "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1))
    Path(a.out + ".tmp").replace(a.out)
    for arm, v in res["arms"].items():
        s = res["subset"][arm]
        print(f"{arm:20s} exam {EXAM.get(arm, float('nan')):.3f} | all: esc {v['escaped']:.3f} ans {v['answered']:.3f} "
              f"cor {v['correct']:.3f} | looping cuts: esc {s['escaped']:.3f} ans {s['answered']:.3f} cor {s['correct']:.3f}")
    print("spearman:", res["spearman_vs_exam"])
    print("paired escaped vs restart:", res.get("paired_escaped_vs_restart"))
    print("saved", a.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--math-work", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--budget", type=int, default=4096)
    p.add_argument("--base-dir", default=None)
    b = sub.add_parser("build")
    for k in ("--base", "--math-work", "--blind-work", "--edit-dir", "--out-root"):
        b.add_argument(k, required=True)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--gpu-mem", type=float, default=0.90)
    g.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "build": cmd_build, "gen": cmd_gen, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
