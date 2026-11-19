"""Ultra-fast pilot (~5-7 min on Colab L4): check loop & truncation rates of Qwen/Qwen3.5-4B on MATH-500.

Samples 64 problems (stratified across MATH difficulty levels 1..5) from `HuggingFaceH4/MATH-500`,
generates with vLLM (`max_tokens=16384`, `enable_thinking=True`, early loop stop via `loop_kill_v2`),
and reports overall + per-level loop_killed, soft_loop, truncated, and boxed-answer accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load_math500(n: int = 64, seed: int = 42) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import random

    path = Path(hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_lvl: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_lvl[int(r.get("level", 0))].append(r)
    rng = random.Random(seed)
    picked: list[dict] = []
    lvls = sorted(by_lvl)
    per = max(1, n // max(1, len(lvls)))
    rem = n - per * len(lvls)
    for i, lv in enumerate(lvls):
        pool = list(by_lvl[lv])
        rng.shuffle(pool)
        take = per + (1 if i >= len(lvls) - rem else 0)
        picked.extend(pool[:take])
    return picked[:n]


def _loop_onset_soft(ids: list[int], n: int = 16, window: int = 64, frac: float = 0.5) -> int | None:
    if len(ids) < n + window:
        return None
    seen: set[tuple[int, ...]] = set()
    dup = [0] * len(ids)
    for i in range(n - 1, len(ids)):
        g = tuple(ids[i - n + 1 : i + 1])
        dup[i] = int(g in seen)
        seen.add(g)
    run = sum(dup[:window])
    for t in range(len(ids) - window + 1):
        if run >= frac * window:
            return t
        if t + window < len(ids):
            run += dup[t + window] - dup[t]
    return None


def _loop_killed_strict(ids: list[int], n: int = 16, window: int = 768, frac: float = 0.9) -> int | None:
    if len(ids) < n + window:
        return None
    seen: set[tuple[int, ...]] = set()
    dup = [0] * len(ids)
    for i in range(n - 1, len(ids)):
        g = tuple(ids[i - n + 1 : i + 1])
        dup[i] = int(g in seen)
        seen.add(g)
    run = sum(dup[:window])
    for t in range(len(ids) - window + 1):
        if run >= frac * window:
            return t
        if t + window < len(ids):
            run += dup[t + window] - dup[t]
    return None


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    depth, start = 0, idx + 7
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def _norm_ans(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip().replace(" ", "").replace("\\$", "").replace("$", "")
    s = re.sub(r"\\left|\\right", "", s)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--out", default="pilot_math500_4b.json")
    args = ap.parse_args()

    t0 = time.time()
    rows = load_math500(n=args.n)
    print(f"Loaded {len(rows)} stratified MATH-500 problems")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["problem"] + "\nPut your final answer within \\boxed{}."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for r in rows
    ]

    llm_kwargs: dict = dict(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_tokens + 2048,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=False,
    )
    use_early_stop = False
    try:
        import loop_kill_v2  # noqa: F401
        if getattr(loop_kill_v2, "LoopKillLogitsProcessor", None) is not None:
            llm_kwargs["logits_processors"] = [loop_kill_v2.LoopKillLogitsProcessor]
            use_early_stop = True
    except Exception as exc:
        print(f"Note: loop_kill_v2 unavailable ({exc}); running without early stop.")

    print(f"Loading vLLM ({args.model}, early_loop_stop={use_early_stop})...")
    llm = LLM(**llm_kwargs)
    sp_kwargs: dict = dict(temperature=0.0, max_tokens=args.max_tokens)
    if use_early_stop and tok.eos_token_id is not None:
        sp_kwargs["extra_args"] = {"loop_kill_eos": int(tok.eos_token_id)}
    sp = SamplingParams(**sp_kwargs)

    t_gen = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t_gen

    items = []
    by_level: dict[int, list[dict]] = defaultdict(list)
    for r, o in zip(rows, outs):
        out0 = o.outputs[0]
        ids = list(out0.token_ids)
        text = out0.text
        strict_on = _loop_killed_strict(ids)
        soft_on = _loop_onset_soft(ids)
        trunc = (out0.finish_reason == "length") and (strict_on is None)
        pred = _extract_boxed(text)
        gold = r.get("answer", "")
        exact = (_norm_ans(pred) == _norm_ans(gold)) and bool(_norm_ans(gold))
        rec = {
            "level": int(r.get("level", 0)),
            "subject": r.get("subject", ""),
            "n_tokens": len(ids),
            "finish_reason": out0.finish_reason,
            "loop_killed_strict": strict_on is not None,
            "loop_onset_strict": strict_on,
            "loop_soft": soft_on is not None,
            "loop_onset_soft": soft_on,
            "truncated": trunc,
            "boxed_found": pred is not None,
            "exact_match_approx": exact,
        }
        items.append(rec)
        by_level[rec["level"]].append(rec)

    def _agg(group: list[dict]) -> dict:
        m = max(1, len(group))
        lens = sorted(x["n_tokens"] for x in group)
        return {
            "n": len(group),
            "loop_killed_strict": round(sum(x["loop_killed_strict"] for x in group) / m, 4),
            "loop_soft": round(sum(x["loop_soft"] for x in group) / m, 4),
            "truncated": round(sum(x["truncated"] for x in group) / m, 4),
            "any_failure_loop_or_trunc": round(
                sum((x["loop_killed_strict"] or x["loop_soft"] or x["truncated"]) for x in group) / m, 4
            ),
            "exact_match_approx": round(sum(x["exact_match_approx"] for x in group) / m, 4),
            "median_tokens": lens[len(lens) // 2],
            "p90_tokens": lens[int(0.9 * (len(lens) - 1))],
        }

    summary = {
        "model": args.model,
        "dataset": "HuggingFaceH4/MATH-500",
        "n": len(items),
        "max_tokens": args.max_tokens,
        "early_loop_stop": use_early_stop,
        "gen_s": round(gen_s, 1),
        "total_s": round(time.time() - t0, 1),
        "overall": _agg(items),
        "by_level": {str(lv): _agg(by_level[lv]) for lv in sorted(by_level)},
        "level_4_5": _agg([x for x in items if x["level"] >= 4]),
        "items": items,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== MATH-500 PILOT SUMMARY ===")
    print(json.dumps({k: summary[k] for k in ("model", "n", "gen_s", "overall", "level_4_5", "by_level")}, indent=2))


if __name__ == "__main__":
    main()
