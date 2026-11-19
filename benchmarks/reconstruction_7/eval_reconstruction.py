"""Evaluate the blind agent's reconstructed Qwen3.5-0.8B checkpoint on the exact 200-problem GSM8K 16k test."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import discover_fast as fast  # noqa: E402
import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402


def _base_cache_file() -> Path | None:
    root = os.environ.get("WSI_BASE_CACHE_DIR")
    if not root:
        return None
    return Path(root) / "Qwen_Qwen3.5-0.8B__gsm8k__test_n200_tok16384_s0.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--reconstructed-dir", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. Prepare the exact same 200-problem GSM8K test split as discover_blind / discover_loop_fix
    if not (work / "requests_test.jsonl").exists():
        dl.cmd_prepare(argparse.Namespace(
            work=str(work),
            base_dir=args.base_dir,
            data_dir=None,
            n_elicit=1,
            n_dev=1,
            n_test=200,
            elicit_max_new=64,
            dev_max_new=64,
            max_new=16384,
            seed=0,
        ))

    run_ns = argparse.Namespace(work=str(work), dtype=args.dtype, gpu_mem=args.gpu_mem, cpu=False)

    # 2. Base arm (reused from shared Drive cache if available)
    base_out = work / "outputs_test_base.jsonl"
    cp = _base_cache_file()
    if not base_out.exists() and cp is not None and cp.exists() and cp.stat().st_size > 0:
        shutil.copy2(cp, base_out)
        print(f"[cache] Reused base GSM8K-200 outputs from {cp.name}", flush=True)
    if not base_out.exists():
        fast.run_parallel(work, [{
            "model": None,
            "jobs": [{"name": "base", "requests": "requests_test.jsonl", "out": "outputs_test_base.jsonl"}],
        }], run_ns)
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(".jsonl.tmp")
            shutil.copy2(base_out, tmp)
            os.replace(tmp, cp)

    # 3. Blind-agent reconstructed checkpoint (fixed)
    if not (work / "outputs_test_reconstructed.jsonl").exists():
        fast.run_parallel(work, [{
            "model": str(Path(args.reconstructed_dir).resolve()),
            "jobs": [{"name": "reconstructed_fixed", "requests": "requests_test.jsonl",
                      "out": "outputs_test_reconstructed.jsonl"}],
        }], run_ns)

    # 4. Blind-agent reconstructed checkpoint + MoI loop router
    dl.cmd_plan_moi(argparse.Namespace(work=str(work)))
    if not (work / "outputs_test_moi_reconstructed.jsonl").exists():
        fast.run_parallel(work, [{
            "model": str(Path(args.reconstructed_dir).resolve()),
            "jobs": [{"name": "reconstructed_moi", "requests": "requests_test_moi.jsonl",
                      "out": "outputs_test_moi_reconstructed.jsonl"}],
        }], run_ns)

    # 5. Grade all arms identically with discover_loop_fix / moi_loop_router_vllm
    from transformers import AutoTokenizer

    cfg = dl.cfg_load(work)
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"])
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}

    rows = {}
    arms = {}
    for name, fn in (
        ("base", "outputs_test_base.jsonl"),
        ("reconstructed_fixed", "outputs_test_reconstructed.jsonl"),
    ):
        s = dl.score(work, "requests_test.jsonl", fn, tok)
        rows[name] = {r["id"]: r for r in s["rows"]}
        arms[name] = {k: v for k, v in s.items() if k != "rows"}

    plan = {r["id"]: r for r in lr.read_jsonl(work / "requests_test_moi.jsonl")}
    cont = {o["id"]: o for o in lr.read_jsonl(work / "outputs_test_moi_reconstructed.jsonl")}
    moi, edit_tok, all_tok = {}, 0, 0
    for i in reqs:
        if i in cont:
            pre = plan[i]["prompt_ids"][len(reqs[i]["prompt_ids"]):]
            g = lr.grade(pre + cont[i]["token_ids"], cont[i]["finish_reason"], reqs[i]["gold"], tok, eos,
                         reqs[i]["max_tokens"])
            edit_tok += max(g["tokens"] - len(pre), 0)
        else:
            g = rows["base"][i]
        all_tok += g["tokens"]
        moi[i] = g
    n = len(moi)
    arms["reconstructed_moi"] = {
        "n": n,
        "accuracy": sum(g["correct"] for g in moi.values()) / n,
        "loop_killed": sum(g["looped_killed"] for g in moi.values()) / n,
        "truncated": sum(g["truncated"] for g in moi.values()) / n,
        "edit_token_share": edit_tok / max(all_tok, 1),
        "router_fired": len(plan) / n,
    }
    rows["reconstructed_moi"] = moi

    ids = sorted(reqs)
    acc = {k: [bool(rows[k][i]["correct"]) for i in ids] for k in rows}
    paired = {
        "reconstructed_fixed_vs_base": lr.paired_ci(acc["reconstructed_fixed"], acc["base"]),
        "reconstructed_moi_vs_base": lr.paired_ci(acc["reconstructed_moi"], acc["base"]),
        "reconstructed_moi_vs_fixed": lr.paired_ci(acc["reconstructed_moi"], acc["reconstructed_fixed"]),
    }

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8")) if Path(args.summary_json).exists() else {}
    res = {
        "experiment": "reconstruction_7",
        "model": "Qwen/Qwen3.5-0.8B",
        "dataset": "openai/gsm8k (test, n=200, 16k)",
        "brief_words": 189,
        "reconstruct_summary": summary,
        "eval_s": round(time.perf_counter() - t0, 1),
        "arms": arms,
        "paired": paired,
        "reference_discover_blind": {
            "base": 0.11,
            "blind_fixed": 0.475,
            "blind_moi": 0.555,
            "random_moi": 0.32,
            "p50c": 0.58,
        },
    }
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n=== #7 RECONSTRUCTION RESULTS (Qwen3.5-0.8B, GSM8K n=200, 16k) ===")
    print(f"{'arm':22s} {'acc':>7s} {'loops':>7s} {'trunc':>7s}")
    for k, v in arms.items():
        print(f"{k:22s} {v['accuracy']:7.1%} {v['loop_killed']:7.1%} {v['truncated']:7.1%}")
    for k, v in paired.items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]  ({v['a_only']}/{v['b_only']})")
    print("Reference (discover_blind): base=11.0% | blind_fixed=47.5% | blind_moi=55.5% | p50c=58.0%")


if __name__ == "__main__":
    main()
