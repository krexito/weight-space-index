"""EXPERIMENTAL: is MATH-500 weak because of the edit's shape (layers / columns / alpha) or its direction?

Our blind GSM8K direction detects MATH-500 loop tokens as well as the best fitted direction (experiment_direction_auc.py),
yet the blind patch is weak on MATH-500 while MIT reports p50c at 48.1% in their own eval. Both patches use the same
form  W' = W - alpha * u (m * u^T W)  on MLP down_proj columns; they differ in layers (13-14 vs 15+19), columns per
layer (16 vs 64), alpha (12 vs 6) and direction. Always-on (fixed) arms on the same 150 MATH-500 test problems, same
requests, seeds and sampling as the saved base arm of the discover_math500_blind run:

  p50c              MIT's official p50c checkpoint (rebuilt from benchmarks/exp8_math, SHA-256 checked)
  blind_n16_a12     our blind GSM8K patch as frozen (L13-14, 16 columns, alpha 12)
  blind_n64_a12     same direction and alpha, 64 columns (p50c's column count)
  blind_n64_a6      same direction, p50c's columns and alpha: p50c's shape with our direction

Reference arms already saved in the run: base, wsi_fixed (the MATH-500 blind choice L10_n256_a12), random_fixed.
Graded with discover_math500_blind.grade_math (boxed answer, loop kill = wrong).

  build   python scripts/experiment_edit_form_math500.py build  --base B --blind-work BW --edit-dir benchmarks/exp8_math --out-root CK
  gen     python scripts/experiment_edit_form_math500.py gen    --work MATH_RUN --model CK/p50c --out outputs_test_p50c.jsonl
  report  python scripts/experiment_edit_form_math500.py report --work MATH_RUN --out R.json
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

ARMS = {  # name -> (layers, columns, alpha) for the blind direction; p50c is the official checkpoint
    "blind_n16_a12": ([13, 14], 16, 12.0),
    "blind_n64_a12": ([13, 14], 64, 12.0),
    "blind_n64_a6": ([13, 14], 64, 6.0),
}
SAVED = {"base": "outputs_test_base.jsonl", "wsi_fixed_L10": "outputs_test_fast.jsonl",
         "random_fixed_L10": "outputs_test_random.jsonl"}
NEW = {"p50c": "outputs_test_p50c.jsonl", **{k: f"outputs_test_{k}.jsonl" for k in ARMS}}


def cmd_build(a) -> None:
    import discover_blind as blind
    import moi_loop_router_vllm as lr

    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    rep = lr.build_p50c(Path(a.base), Path(a.edit_dir), root / "p50c")
    print("p50c", json.dumps(rep), flush=True)
    # discover_blind.build_checkpoint reads base_dir from <work>/config.json: give it a private work dir
    bw = root / "_blind_work"
    bw.mkdir(exist_ok=True)
    for f in ("diagnosis_blind.json", "blind_directions.npz"):
        shutil.copy2(Path(a.blind_work) / f, bw / f)
    cfg = json.loads((Path(a.blind_work) / "config.json").read_text())
    cfg["base_dir"] = str(a.base)
    (bw / "config.json").write_text(json.dumps(cfg))
    for name, (layers, n, alpha) in ARMS.items():
        blind.build_checkpoint(bw, layers, n, alpha, "wsi", root / name)
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
    reqs = lr.read_jsonl(work / "requests_test.jsonl")
    if a.limit:
        reqs = [{**r, "max_tokens": min(r["max_tokens"], 512)} for r in reqs[: a.limit]]
    max_len = max(len(r["prompt_ids"]) + r["max_tokens"] for r in reqs) + 16
    t0 = time.time()
    llm, kill = dm._llm_fast(a.model, cfg, max_len, "bfloat16", a.gpu_mem)
    rows, kill = fast._generate(llm, cfg, reqs, kill)
    tmp = out.with_suffix(".tmp")
    lr.write_jsonl(tmp, rows)
    tmp.replace(out)
    ntok = sum(len(x["token_ids"]) for x in rows)
    print(f"[gen {a.out}] {len(rows)} seqs {ntok:,} tokens {(time.time() - t0) / 60:.1f} min early_stop={kill}", flush=True)


def paired(a_rows, b_rows, n=10000, seed=0):
    a = np.array([r["correct"] for r in a_rows], float)
    b = np.array([r["correct"] for r in b_rows], float)
    rng = np.random.default_rng(seed)
    d = a - b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                                        round(float(np.percentile(bs, 97.5)), 4)],
            "a_only": int(((a == 1) & (b == 0)).sum()), "b_only": int(((a == 0) & (b == 1)).sum())}


def cmd_report(a) -> None:
    import discover_math500_blind as dm

    work = Path(a.work)
    arms = {}
    for name, fn in {**SAVED, **NEW}.items():
        if (work / fn).exists():
            s = dm.score_math(work, "requests_test.jsonl", fn)
            s["rows"] = sorted(s["rows"], key=lambda r: r["id"])
            arms[name] = s
    res = {"model": "Qwen/Qwen3.5-0.8B", "dataset": "HuggingFaceH4/MATH-500", "split": "test (150, 16k)",
           "arms": {k: {m: round(v[m], 4) for m in ("n", "accuracy", "loop_killed", "truncated")} |
                    {"mean_tokens": round(float(np.mean([r["n_tokens"] for r in v["rows"]])), 1)} for k, v in arms.items()},
           "paired": {}, "per_id_correct": {k: [int(r["correct"]) for r in v["rows"]] for k, v in arms.items()}}
    for x, y in (("p50c", "base"), ("blind_n16_a12", "base"), ("p50c", "blind_n16_a12"),
                 ("blind_n64_a12", "blind_n16_a12"), ("blind_n64_a6", "blind_n16_a12"), ("p50c", "blind_n64_a6"),
                 ("p50c", "wsi_fixed_L10")):
        if x in arms and y in arms:
            res["paired"][f"{x}_vs_{y}"] = paired(arms[x]["rows"], arms[y]["rows"])
    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1))
    Path(a.out + ".tmp").replace(a.out)
    for k, v in res["arms"].items():
        print(f"{k:18s} acc {v['accuracy']:.3f} loop_killed {v['loop_killed']:.3f} trunc {v['truncated']:.3f} tokens {v['mean_tokens']}")
    for k, v in res["paired"].items():
        print(f"{k:34s} {v}")
    print("saved", a.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base", required=True)
    b.add_argument("--blind-work", required=True)
    b.add_argument("--edit-dir", required=True)
    b.add_argument("--out-root", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--gpu-mem", type=float, default=0.90)
    g.add_argument("--limit", type=int, default=0, help="smoke test: first N problems, 512 tokens")
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "gen": cmd_gen, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
