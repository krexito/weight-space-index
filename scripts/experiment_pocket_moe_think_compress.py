"""Thinking-to-Weights Compression: Distilling 4B `<think>` Chains into Routed 0.8B No-Think Cartridges on MATH-500.

Tests on real-world `HuggingFaceH4/MATH-500` (`n_train=300`, `n_test=200`) + `cais/mmlu` (`n_test=400`) + `24` held-out
prose controls whether a routed full-depth LoRA cartridge (`r=16`, all 24L, `~20.6 MB`, Layer-12 router) can compress
unconstrained `Qwen3.5-4B` `<think>` reasoning (up to `16,384` tokens + 40-problem pilot probe with fail-fast gate)
into direct **`enable_thinking=False`** inference on `Qwen3.5-0.8B`:

  1. `base_08b_nothink`                     : Unmodified Qwen3.5-0.8B (`enable_thinking=False`, max 2048 tok)
  2. `base_08b_think`                       : Unmodified Qwen3.5-0.8B (`enable_thinking=True`,  max 16384 tok)
  3. `teacher_4b_nothink`                   : Unmodified Qwen3.5-4B   (`enable_thinking=False`, max 2048 tok)
  4. `teacher_4b_think`                     : Unmodified Qwen3.5-4B   (`enable_thinking=True`,  max 16384 tok)
  5. `nothink_distill_r16_24L_routed`       : Baseline `4B nothink` traces + `4B nothink` Top-32 KL (160x4=640 passes)
  6. `think_post_nothink_kl_r16_24L_routed` : `4B think` post-`</think>` solutions + `4B nothink` Top-32 KL (same 640 passes)
  7. `think_privileged_kl_r16_24L_routed`   : Exact same `4B think` post-`</think>` solutions + **privileged `4B think`
                                              Top-32 KL conditioned on full `<think>...</think>`** while student sees
                                              **zero `<think>` tokens** (same 640 passes, matched seed init)

CPU smoke test::
    python scripts/experiment_pocket_moe_think_compress.py --smoke

Pilot-only probe::
    python scripts/experiment_pocket_moe_think_compress.py --teacher /content/teacher_4b --student /content/student_08b --pilot-only --out /content/pilot.json

Colab L4/A100 run::
    python scripts/experiment_pocket_moe_think_compress.py --teacher /content/teacher_4b --student /content/student_08b --cache-dir /content/think_cache --out /content/pocket_moe_think_compress_16k.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import discover_math500_blind as dm  # noqa: E402
import eval_capability_and_boundary_lfm3 as ec  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from experiment_pocket_moe_ceiling import (  # noqa: E402
    _bake_full_checkpoint,
    _hf_generate_subset,
    _vllm_generate,
    fit_ridge_router,
    predict_ridge_router,
)
from experiment_pocket_moe_controls import _paired_bootstrap  # noqa: E402
from experiment_pocket_moe_math500 import (  # noqa: E402
    MATH_DOMAINS,
    PROSE_HELDOUT,
    PROSE_ROUTER_TRAIN,
    ROUTER_CLASSES,
    _compose_routed_mmlu,
    _grade_math500_rows,
    _score_mmlu_fast,
    _summarize_math500_arm,
    _summarize_mmlu,
    _temp_apply_weights,
    format_math500_prompt_ids,
    format_mmlu_prompt_ids,
    load_math500_split,
    load_mmlu_split,
)
from experiment_pocket_moe_regularized import (  # noqa: E402
    _score_routed_mmlu_sparse,
    _train_regularized_lora_expert,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _PlantedReasoningLM,
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
)
from experiment_reasoning_distill_v2 import (  # noqa: E402
    _format_messages,
    _select_backend,
)
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402


# ------------------------------------------------------------------ Thinking Split & Privileged Alignment Helpers
def format_math500_think_prompt_ids(tok: Any, problem: str) -> list[int]:
    content = problem.strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
    return _format_messages(tok, [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=True)


def _split_think_output(
    token_ids: list[int],
    tok: Any,
    eos_set: set[int],
    think_end_id: int | None,
) -> dict[str, Any]:
    """Split a thinking-mode generation into `(closed_think, think_prefix_ids, post_think_text, n_think_tokens, n_answer_tokens)`."""
    body_ids = lr.strip_eos(list(token_ids), eos_set)
    if think_end_id is not None and think_end_id in body_ids:
        end_pos = body_ids.index(think_end_id) + 1
        think_prefix_ids = body_ids[:end_pos]
        post_ids = body_ids[end_pos:]
        # Keep any leading whitespace-only tokens attached to the teacher's </think> prefix
        ws_take = 0
        while ws_take < len(post_ids):
            piece = tok.decode([post_ids[ws_take]], skip_special_tokens=False)
            if piece.strip() == "":
                ws_take += 1
            else:
                break
        think_prefix_ids = think_prefix_ids + post_ids[:ws_take]
        ans_raw_ids = post_ids[ws_take:]
        post_text = tok.decode(ans_raw_ids, skip_special_tokens=True).strip()
        return {
            "closed_think": True,
            "think_prefix_ids": think_prefix_ids,
            "post_think_text": post_text,
            "n_think_tokens": len(think_prefix_ids),
            "n_answer_tokens": len(ans_raw_ids),
        }

    full_txt = tok.decode(body_ids, skip_special_tokens=True)
    if "</think>" in full_txt:
        pre_s, post_s = full_txt.split("</think>", 1)
        think_prefix_ids = [int(x) for x in tok.encode(pre_s + "</think>\n\n", add_special_tokens=False)]
        post_text = post_s.strip()
        ans_ids = [int(x) for x in tok.encode(post_text, add_special_tokens=False)]
        return {
            "closed_think": True,
            "think_prefix_ids": think_prefix_ids,
            "post_think_text": post_text,
            "n_think_tokens": len(think_prefix_ids),
            "n_answer_tokens": len(ans_ids),
        }

    return {
        "closed_think": False,
        "think_prefix_ids": body_ids,
        "post_think_text": "",
        "n_think_tokens": len(body_ids),
        "n_answer_tokens": 0,
    }


def _build_aligned_privileged_pair(
    tok: Any,
    problem: str,
    post_think_text: str,
    think_prefix_ids: list[int],
) -> dict[str, Any]:
    """Construct token-exact `(nothink_seq, privileged_think_seq)` sharing the exact same assistant target tokens."""
    content = problem.strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
    p_nothink = _format_messages(tok, [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=False)
    full_nothink = _format_messages(
        tok,
        [{"role": "user", "content": content}, {"role": "assistant", "content": post_think_text}],
        add_generation_prompt=False,
        enable_thinking=False,
    )
    lo_s = max(0, len(p_nothink) - 1)
    hi_s = len(full_nothink) - 1
    ans_ids = full_nothink[lo_s + 1 : hi_s + 1]

    p_think = _format_messages(tok, [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=True)
    prefix_t = p_think + list(think_prefix_ids)
    full_think = prefix_t + ans_ids
    lo_t = max(0, len(prefix_t) - 1)
    hi_t = len(full_think) - 1

    assert hi_t - lo_t == hi_s - lo_s == len(ans_ids), (hi_t - lo_t, hi_s - lo_s, len(ans_ids))
    assert full_think[lo_t + 1 : hi_t + 1] == full_nothink[lo_s + 1 : hi_s + 1]
    return {
        "nothink_ids": full_nothink,
        "lo_s": lo_s,
        "hi_s": hi_s,
        "think_ids": full_think,
        "lo_t": lo_t,
        "hi_t": hi_t,
        "n_target_tokens": len(ans_ids),
    }


def _compute_think_diagnostics(
    raw_outs: list[dict[str, Any]],
    tok: Any,
    eos_set: set[int],
    think_end_id: int | None,
) -> dict[str, Any]:
    closed = 0
    think_toks: list[int] = []
    ans_toks: list[int] = []
    for raw in raw_outs:
        sp = _split_think_output(raw["token_ids"], tok, eos_set, think_end_id)
        if sp["closed_think"]:
            closed += 1
        think_toks.append(int(sp["n_think_tokens"]))
        ans_toks.append(int(sp["n_answer_tokens"]))
    n = max(1, len(raw_outs))
    return {
        "closed_think_rate": round(closed / n, 4),
        "mean_think_tokens": round(sum(think_toks) / n, 1),
        "median_think_tokens": round(float(statistics.median(think_toks)), 1) if think_toks else 0.0,
        "mean_answer_tokens": round(sum(ans_toks) / n, 1),
    }


def _percentile(vals: Sequence[int | float], q: float) -> float:
    """Pure-Python sorted nearest-rank percentile in [0, 100]."""
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    idx = min(len(s) - 1, max(0, math.ceil((float(q) / 100.0) * len(s)) - 1))
    return float(s[idx])


def _select_stratified_pilot_indices(math_train: list[dict[str, Any]], pilot_n: int, seed: int) -> list[int]:
    """Select `pilot_n` train indices proportionally stratified by `(level, domain)`."""
    if pilot_n <= 0 or not math_train:
        return []
    if pilot_n >= len(math_train):
        return list(range(len(math_train)))

    by_bucket: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, it in enumerate(math_train):
        by_bucket[(int(it.get("level", 0)), str(it.get("domain", "")))].append(i)

    rng = random.Random(int(seed) + 991)
    buckets = sorted(by_bucket.keys())
    for bk in buckets:
        rng.shuffle(by_bucket[bk])

    total = len(math_train)
    quotas: list[tuple[float, tuple[int, str], int]] = []
    counts: dict[tuple[int, str], int] = {}
    for bk in buckets:
        grp_len = len(by_bucket[bk])
        exact = (grp_len * int(pilot_n)) / float(total)
        base_q = int(exact)
        counts[bk] = base_q
        quotas.append((exact - base_q, bk, grp_len))

    quotas.sort(key=lambda x: (-x[0], x[1]))
    rem = int(pilot_n) - sum(counts.values())
    for _, bk, grp_len in quotas:
        if rem <= 0:
            break
        if counts[bk] < grp_len:
            counts[bk] += 1
            rem -= 1

    chosen: list[int] = []
    for bk in buckets:
        chosen.extend(by_bucket[bk][: counts[bk]])
    chosen.sort()
    assert len(chosen) == int(pilot_n) and len(set(chosen)) == int(pilot_n)
    return chosen


def _summarize_think_pilot(
    pilot_items: list[dict[str, Any]],
    nt_rows: list[dict[str, Any]],
    th_rows: list[dict[str, Any]],
    th_raw: list[dict[str, Any]],
    tok: Any,
    eos_set: set[int],
    think_end_id: int | None,
    *,
    max_new_tokens_think: int,
    min_pilot_closed_rate: float,
    min_pilot_think_gain: float,
) -> dict[str, Any]:
    """Summarize Phase 0 pilot probe: `<think>` vs `nothink`, percentiles, free budget curve & fail-fast gate."""
    n = max(1, len(pilot_items))
    assert len(nt_rows) == len(pilot_items) and len(th_rows) == len(pilot_items) and len(th_raw) == len(pilot_items)

    th_diag = _compute_think_diagnostics(th_raw, tok, eos_set, think_end_id)
    closed_flags = [
        bool(_split_think_output(raw["token_ids"], tok, eos_set, think_end_id)["closed_think"]) for raw in th_raw
    ]

    n_nt_correct = sum(1 for r in nt_rows if r["strict_correct"])
    th_corr_rows = [r for r in th_rows if r["strict_correct"]]
    n_th_correct = len(th_corr_rows)

    nt_acc = round(n_nt_correct / n, 4)
    th_acc = round(n_th_correct / n, 4)
    think_minus_nothink = round(th_acc - nt_acc, 4)

    corr_toks = [int(r.get("total_tokens", r.get("n_tokens", 0))) for r in th_corr_rows]
    p50 = round(_percentile(corr_toks, 50.0), 1)
    p90 = round(_percentile(corr_toks, 90.0), 1)
    p95 = round(_percentile(corr_toks, 95.0), 1)

    max_tok_i = int(max_new_tokens_think)
    budget_grid = sorted({b for b in (1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384) if b <= max_tok_i} | {max_tok_i})
    budget_curve: dict[str, dict[str, Any]] = {}
    for b in budget_grid:
        n_c_b = sum(1 for r in th_rows if r["strict_correct"] and int(r.get("total_tokens", r.get("n_tokens", 0))) <= b)
        n_cl_b = sum(
            1 for r, cl in zip(th_rows, closed_flags) if cl and int(r.get("total_tokens", r.get("n_tokens", 0))) <= b
        )
        budget_curve[str(b)] = {
            "acc": round(n_c_b / n, 4),
            "n_correct": n_c_b,
            "closed_rate": round(n_cl_b / n, 4),
        }

    recommended_max_tokens_think = max_tok_i
    if corr_toks:
        for b in budget_grid:
            if float(b) >= p95:
                recommended_max_tokens_think = b
                break

    per_level: dict[str, dict[str, Any]] = {}
    levels = sorted({int(it.get("level", r.get("level", 0))) for it, r in zip(pilot_items, th_rows)})
    for lvl in levels:
        idxs = [i for i, (it, r) in enumerate(zip(pilot_items, th_rows)) if int(it.get("level", r.get("level", 0))) == lvl]
        m = max(1, len(idxs))
        lvl_nt = sum(1 for i in idxs if nt_rows[i]["strict_correct"]) / m
        lvl_th = sum(1 for i in idxs if th_rows[i]["strict_correct"]) / m
        lvl_cl = sum(1 for i in idxs if closed_flags[i]) / m
        per_level[str(lvl)] = {
            "n": len(idxs),
            "nothink_acc": round(lvl_nt, 4),
            "think_acc": round(lvl_th, 4),
            "think_minus_nothink": round(lvl_th - lvl_nt, 4),
            "closed_rate": round(lvl_cl, 4),
        }

    gate_reasons: list[str] = []
    if n_th_correct == 0:
        gate_reasons.append(f"teacher_think solved 0/{len(pilot_items)} pilot items")
    if th_diag["closed_think_rate"] < float(min_pilot_closed_rate):
        gate_reasons.append(
            f"closed_think_rate {th_diag['closed_think_rate'] * 100:.1f}% < min {float(min_pilot_closed_rate) * 100:.1f}%"
        )
    if think_minus_nothink < float(min_pilot_think_gain):
        gate_reasons.append(
            f"think_minus_nothink {think_minus_nothink * 100:+.1f} pts < min {float(min_pilot_think_gain) * 100:+.1f} pts"
        )

    return {
        "pilot_n": len(pilot_items),
        "max_new_tokens_think": max_tok_i,
        "teacher_nothink_acc": nt_acc,
        "teacher_think_acc": th_acc,
        "think_minus_nothink": think_minus_nothink,
        "closed_think_rate": th_diag["closed_think_rate"],
        "truncated_rate": round(sum(1 for r in th_rows if r["truncated"]) / n, 4),
        "loop_killed_rate": round(sum(1 for r in th_rows if r["loop_killed"]) / n, 4),
        "think_only_solved_n": sum(1 for rn, rt in zip(nt_rows, th_rows) if rt["strict_correct"] and not rn["strict_correct"]),
        "nothink_only_solved_n": sum(1 for rn, rt in zip(nt_rows, th_rows) if rn["strict_correct"] and not rt["strict_correct"]),
        "both_solved_n": sum(1 for rn, rt in zip(nt_rows, th_rows) if rt["strict_correct"] and rn["strict_correct"]),
        "mean_think_tokens": th_diag["mean_think_tokens"],
        "median_think_tokens": th_diag["median_think_tokens"],
        "mean_answer_tokens": th_diag["mean_answer_tokens"],
        "correct_total_tokens_p50": p50,
        "correct_total_tokens_p90": p90,
        "correct_total_tokens_p95": p95,
        "budget_curve": budget_curve,
        "recommended_max_tokens_think": recommended_max_tokens_think,
        "per_level": per_level,
        "gate_reasons": gate_reasons,
        "gate_passed": bool(len(gate_reasons) == 0),
    }


# ------------------------------------------------------------------ Atomic Generation Cache & Fast Long-Context vLLM
def _gen_cache_path(
    cache_dir: Path | None,
    tag: str,
    model_id: str,
    n_items: int,
    max_tokens: int,
    seed: int,
) -> Path | None:
    if cache_dir is None:
        return None
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(model_id)).name or str(model_id))
    return Path(cache_dir) / f"{tag}__{slug}__n{int(n_items)}__tok{int(max_tokens)}__s{int(seed)}.json"


def _try_load_gen_cache(path: Path | None, expected_prompt_lens: list[int]) -> list[dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    outs = data.get("outputs") if isinstance(data, dict) else None
    if not isinstance(outs, list) or len(outs) != len(expected_prompt_lens):
        return None
    if data.get("prompt_lens") != expected_prompt_lens:
        return None
    return outs


def _save_gen_cache(
    path: Path | None,
    outs: list[dict[str, Any]],
    prompt_lens: list[int],
    meta: dict[str, Any],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {**meta, "prompt_lens": prompt_lens, "outputs": outs}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def _vllm_generate_fast(
    model_dir: str,
    tok: Any,
    prompts_ids: list[list[int]],
    seeds: list[int],
    *,
    max_new_tokens: int,
    dtype_str: str,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Fast vLLM generation supporting long `<think>` budgets (16k), FP8 KV when supported, and loop_kill_v2."""
    if int(max_new_tokens) <= 2048 and len(prompts_ids) <= 250:
        return _vllm_generate(
            model_dir,
            tok,
            prompts_ids,
            seeds,
            max_new_tokens=max_new_tokens,
            dtype_str=dtype_str,
            base_seed=base_seed,
        )

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    py_parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if not py_parts or py_parts[0] != str(HERE):
        os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE)] + [p for p in py_parts if p != str(HERE)])

    import loop_kill_v2
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    eos_list = lr.eos_ids(tok)
    kill_eos = eos_list[-1]
    max_len = max(len(p) for p in prompts_ids) + int(max_new_tokens) + 64
    fp8_ok = False
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        gpu_mem = round(min(0.88, max(0.60, (free_b / max(1, total_b)) - 0.06)), 3)
        maj, mn = torch.cuda.get_device_capability(0)
        fp8_ok = ((maj > 8) or (maj == 8 and mn >= 9)) and os.environ.get("WSI_NO_FP8_KV", "0") != "1"
        max_seqs = 96 if total_b >= 36 * (1024**3) else 64
    else:
        gpu_mem = 0.80
        max_seqs = 64

    base_kw: dict[str, Any] = dict(
        model=str(model_dir),
        tokenizer=str(tok.name_or_path),
        dtype=dtype_str,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem,
        enable_prefix_caching=True,
        seed=int(base_seed),
        trust_remote_code=True,
    )
    extras: dict[str, Any] = {
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "enable_chunked_prefill": True,
        "max_num_seqs": max_seqs,
    }
    if fp8_ok:
        extras["kv_cache_dtype"] = "fp8"
    proc = loop_kill_v2.processor()
    if proc is not None:
        extras["logits_processors"] = [proc]

    raw_attempts: list[tuple[bool, tuple[str, ...]]] = [
        (False, ()),
        (False, ("kv_cache_dtype",)),
        (False, ("kv_cache_dtype", "max_num_seqs")),
        (False, ("kv_cache_dtype", "max_num_seqs", "enable_chunked_prefill")),
        (True, ("kv_cache_dtype",)),
        (True, ("kv_cache_dtype", "max_num_seqs", "enable_chunked_prefill")),
        (True, ("kv_cache_dtype", "max_num_seqs", "enable_chunked_prefill", "limit_mm_per_prompt")),
        (True, ("kv_cache_dtype", "max_num_seqs", "enable_chunked_prefill", "logits_processors")),
        (True, ("kv_cache_dtype", "max_num_seqs", "enable_chunked_prefill", "limit_mm_per_prompt", "logits_processors")),
    ]
    attempts: list[tuple[bool, tuple[str, ...]]] = []
    seen_sigs: set[tuple[bool, tuple[str, ...]]] = set()
    for eager, drop in raw_attempts:
        kept_keys = tuple(sorted(k for k in extras if k not in drop))
        if (eager, kept_keys) not in seen_sigs:
            seen_sigs.add((eager, kept_keys))
            attempts.append((eager, kept_keys))

    llm = None
    kill = False
    last_err: Exception | None = None
    t_boot = time.perf_counter()
    print(
        f"[vllm-fast] launching {Path(model_dir).name} "
        f"(prompts={len(prompts_ids)}, max_new={max_new_tokens}, max_len={max_len}, gpu_mem={gpu_mem}, fp8_try={fp8_ok})...",
        flush=True,
    )
    for eager, kept_keys in attempts:
        kw = {**base_kw, "enforce_eager": eager, **{k: extras[k] for k in kept_keys}}
        try:
            llm = LLM(**kw)
            kill = "logits_processors" in kw
            print(
                f"[vllm-fast] booted in {time.perf_counter() - t_boot:.1f}s "
                f"(eager={eager}, kv={kw.get('kv_cache_dtype', 'auto')}, chunked={'enable_chunked_prefill' in kw}, "
                f"max_seqs={kw.get('max_num_seqs', 'default')}, early_loop_stop={kill})",
                flush=True,
            )
            break
        except Exception as err:  # noqa: BLE001
            last_err = err
            print(
                f"[vllm-fast] refused eager={eager} opts={list(kept_keys)}: {type(err).__name__}: {str(err)[:220]}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if llm is None:
        assert last_err is not None
        raise last_err

    try:
        sps = [
            SamplingParams(
                **lr.SAMPLING,
                max_tokens=int(max_new_tokens),
                seed=int(s),
                stop_token_ids=eos_list,
                extra_args={"loop_kill_eos": kill_eos} if kill else None,
            )
            for s in seeds
        ]
        tok_prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts_ids]
        t_gen = time.perf_counter()
        try:
            outs = llm.generate(tok_prompts, sps, use_tqdm=False)
        except Exception as err:  # noqa: BLE001
            if not kill:
                raise
            print(f"[vllm-fast] early loop stop failed ({type(err).__name__}); retrying without it", flush=True)
            sps = [
                SamplingParams(**lr.SAMPLING, max_tokens=int(max_new_tokens), seed=int(s), stop_token_ids=eos_list)
                for s in seeds
            ]
            outs = llm.generate(tok_prompts, sps, use_tqdm=False)
        print(f"[vllm-fast] generated {len(outs)} prompts in {time.perf_counter() - t_gen:.1f}s", flush=True)
        return [{"token_ids": list(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason} for o in outs]
    finally:
        try:
            eng = getattr(llm, "llm_engine", None)
            ec_core = getattr(eng, "engine_core", None)
            if ec_core is not None and hasattr(ec_core, "shutdown"):
                ec_core.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:  # noqa: BLE001
            pass
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ Compact Summary Printer
def _print_think_compress_summary(payload: dict[str, Any]) -> None:
    """Print concise human-readable report + single-line compact JSON omitting per-item rows."""
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return

    pilot = payload.get("pilot_report")
    if pilot:
        print("\n=== 0. SONDA RAPIDA (PILOT PROBE) ===", flush=True)
        print(
            f"  n={pilot['pilot_n']} | max_tok_think={pilot['max_new_tokens_think']} "
            f"| teacher_nothink={pilot['teacher_nothink_acc']*100:.1f}% "
            f"vs teacher_think={pilot['teacher_think_acc']*100:.1f}% "
            f"({pilot['think_minus_nothink']*100:+.1f} pts) "
            f"| closed={pilot['closed_think_rate']*100:.1f}% trunc={pilot['truncated_rate']*100:.1f}% loop={pilot['loop_killed_rate']*100:.1f}%",
            flush=True,
        )
        print(
            f"  correct_tok p50={pilot['correct_total_tokens_p50']} p90={pilot['correct_total_tokens_p90']} "
            f"p95={pilot['correct_total_tokens_p95']} -> recommended_max_tokens_think={pilot['recommended_max_tokens_think']} "
            f"| think_only={pilot['think_only_solved_n']} nothink_only={pilot['nothink_only_solved_n']} both={pilot['both_solved_n']}",
            flush=True,
        )
        print("  budget_curve:", pilot.get("budget_curve"), flush=True)
        print("  per_level:", pilot.get("per_level"), flush=True)
        print(f"  gate_passed={pilot['gate_passed']} | reasons={pilot['gate_reasons']}", flush=True)

    if payload.get("status") in ("pilot_only", "pilot_stopped"):
        print("Status:", payload.get("status"), "| Timing (s):", payload.get("timing_s"), flush=True)
        print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
        print(json.dumps(payload, separators=(",", ":")), flush=True)
        print("<<<COMPACT_JSON_END>>>", flush=True)
        return

    print("\n=== 1. COBERTURA TEACHER (NOTHINK vs THINK) & ROUTER ===", flush=True)
    print("Teacher train coverage:", payload.get("teacher_train_coverage"), flush=True)
    print("Compile time (s):", payload.get("compile_time_s"), flush=True)
    print("Router report:", payload.get("router_report"), flush=True)

    prose_kl = payload.get("prose_kl", {})
    print("\n=== 2. MATH-500 (n=200, 7 BRAZOS PAREADOS: COMPRESION DE <think> -> SIN <think>) ===", flush=True)
    for arm_nm, st in payload.get("math500_arms", {}).items():
        pkl = prose_kl.get(arm_nm, 0.0)
        th_only = st.get("think_only_hard_subset_acc")
        th_s = f"{th_only*100:5.1f}%" if th_only is not None else "   n/a"
        comp_x = st.get("token_compression_vs_base_think")
        comp_s = f"{comp_x:4.2f}x" if comp_x is not None else "  n/a"
        print(
            f"  {arm_nm:38s} | acc={st['accuracy']*100:5.1f}% ({st['n_correct']:3d}/{st['n_total']}) "
            f"| think_only={th_s} | tok_mean={st['mean_tokens']:6.1f} (med={st['median_tokens']:6.1f}, {comp_s}) "
            f"| trunc={st['truncated_rate']*100:4.1f}% loop={st['loop_killed_rate']*100:4.1f}% | pkl={pkl:.5f}",
            flush=True,
        )
        print(f"     per_level={st['per_level']} | per_domain={st['per_domain']}", flush=True)

    print("\n=== 3. RETENCION MMLU (n=400) ===", flush=True)
    for arm_nm, vm in payload.get("mmlu_arms", {}).items():
        a_m, nm = vm["all_400"], vm["non_math"]
        d_nm = f"{nm['delta_vs_base']*100:+5.1f} pts" if "delta_vs_base" in nm else "     base"
        print(
            f"  {arm_nm:38s} | MMLU={a_m['accuracy']*100:5.1f}% | non_math={nm['accuracy']*100:5.1f}% ({d_nm}) | "
            f"ABCD={a_m['mean_abcd_prob_mass']*100:5.1f}% | fire={vm['router_fire_rate']*100:5.1f}%",
            flush=True,
        )

    print("\n=== 4. COMPARACIONES PAREADAS (IC 95% BOOTSTRAP, MATH-500 n=200) ===", flush=True)
    for k, v in payload.get("paired_math500_200", {}).items():
        print(
            f"  {k:44s}: diff={v['diff']*100:+5.1f} pts "
            f"CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})",
            flush=True,
        )

    print("\nCriteria:", payload.get("criteria"), flush=True)
    print("Timing (s):", payload.get("timing_s"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_math500"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


# ------------------------------------------------------------------ CPU Smoke Test
def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of think splitting, privileged token alignment, privileged KL distillation & sparse MMLU."""

    class _ToyTok:
        def __init__(self) -> None:
            self._vocab = {"</think>": 99, "\n\n": 98}

        def apply_chat_template(
            self, messages: list[dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True, enable_thinking: bool = False
        ) -> str:
            parts = []
            for m in messages:
                parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
            if add_generation_prompt:
                parts.append("<assistant><think>\n" if enable_thinking else "<assistant>")
            return "|".join(parts)

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            out: list[int] = []
            for ch in text:
                out.append(1 + (ord(ch) % 24))
            return out

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            out = []
            for x in ids:
                if x == 99:
                    if not skip_special_tokens:
                        out.append("</think>")
                elif x == 98:
                    out.append("\n\n")
                else:
                    out.append(chr(96 + (x % 26)))
            return "".join(out)

    toy_tok = _ToyTok()
    # 1. _split_think_output with </think> token present and missing
    sp_ok = _split_think_output([1, 2, 3, 99, 98, 5, 6, 7], toy_tok, eos_set=set(), think_end_id=99)
    assert sp_ok["closed_think"] is True and sp_ok["n_think_tokens"] == 5 and sp_ok["n_answer_tokens"] == 3, sp_ok
    assert sp_ok["post_think_text"] == "efg", sp_ok
    sp_miss = _split_think_output([1, 2, 3, 4], toy_tok, eos_set=set(), think_end_id=99)
    assert sp_miss["closed_think"] is False and sp_miss["post_think_text"] == "", sp_miss

    # 2. _build_aligned_privileged_pair exact token alignment
    aligned = _build_aligned_privileged_pair(toy_tok, "Compute 2+2.", "The answer is 4.", [1, 2, 3, 99, 98])
    assert aligned["n_target_tokens"] > 0
    assert aligned["hi_t"] - aligned["lo_t"] == aligned["hi_s"] - aligned["lo_s"] == aligned["n_target_tokens"]
    assert aligned["think_ids"][aligned["lo_t"] + 1 : aligned["hi_t"] + 1] == aligned["nothink_ids"][aligned["lo_s"] + 1 : aligned["hi_s"] + 1]

    # 3. Toy privileged vs nothink KL distillation on _PlantedReasoningLM
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Student sees short prompt [1,2,4] + ans [10,20,20,20]; privileged teacher sees longer think prefix [3,3,1,2,4] + same ans
    base_pairs = [
        ([1, 2, 4, 10, 20, 20, 20, 20], [3, 3, 1, 2, 4, 10, 20, 20, 20, 20]),
        ([2, 1, 4, 11, 20, 20, 20, 20], [3, 2, 2, 1, 4, 11, 20, 20, 20, 20]),
    ]
    nothink_samples: list[dict[str, Any]] = []
    priv_samples: list[dict[str, Any]] = []
    for s_seq, t_seq in base_pairs:
        lo_s, hi_s = 3, 7
        lo_t, hi_t = 5, 9
        assert s_seq[lo_s + 1 : hi_s + 1] == t_seq[lo_t + 1 : hi_t + 1]
        with torch.no_grad():
            nt_log = model(torch.tensor(s_seq, dtype=torch.long).unsqueeze(0)).logits[0, lo_s:hi_s].float()
            pr_log = model(torch.tensor(t_seq, dtype=torch.long).unsqueeze(0)).logits[0, lo_t:hi_t].float()
        v_nt, i_nt = nt_log.topk(8, dim=-1)
        v_pr, i_pr = pr_log.topk(8, dim=-1)
        nothink_samples.append({"ids": s_seq, "lo": lo_s, "hi": hi_s, "topk_ids": i_nt.to(torch.int32), "topk_logits": v_nt.to(torch.float16)})
        priv_samples.append({"ids": s_seq, "lo": lo_s, "hi": hi_s, "topk_ids": i_pr.to(torch.int32), "topk_logits": v_pr.to(torch.float16)})

    h_ids = [torch.tensor([1, 2, 3, 1], dtype=torch.long).unsqueeze(0)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    ckpts_out: dict[str, list[float]] = {}
    last_weights: dict[str, torch.Tensor] = {}
    for nm, ds_toy in [("nothink_kl", nothink_samples), ("privileged_think_kl", priv_samples)]:
        torch.manual_seed(42)
        res_e = _train_regularized_lora_expert(
            model,
            [0, 1],
            ds_toy,
            rank=4,
            alpha_scale=8.0,
            total_steps=6,
            grad_accum=2,
            lr_val=2e-2,
            weight_decay=0.05,
            lora_dropout=0.10,
            warmup_frac=0.20,
            distill_T=2.0,
            kl_weight=0.85,
            anchor_ids=h_ids,
            base_anchor_logp=base_lp,
            anchor_kl_weight=0.05,
            healthy_ids=h_ids,
            base_healthy_logp=base_lp,
            base_healthy_nll=base_nll,
            seed=42,
        )
        assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items()), nm
        ckpts = res_e["step_losses_checkpoints"]
        assert ckpts[-1] < ckpts[0], (nm, ckpts)
        ckpts_out[nm] = ckpts
        last_weights = res_e["weights"]

    # 4. Sparse MMLU exact equivalence
    fake_mmlu = [
        {"gold_idx": 0, "is_math_subject": False},
        {"gold_idx": 1, "is_math_subject": True},
        {"gold_idx": 2, "is_math_subject": False},
        {"gold_idx": 3, "is_math_subject": True},
    ]
    fake_pids = [[1, 2, 4, 5], [2, 1, 5, 4], [1, 3, 4, 6], [3, 1, 5, 7]]
    letter_ids = {"A": [6], "B": [7], "C": [8], "D": [9]}
    dev = next(model.parameters()).device
    base_mmlu = _score_mmlu_fast(model, fake_pids, fake_mmlu, letter_ids, dev)
    with _temp_apply_weights(model, last_weights):
        full_ed = _score_mmlu_fast(model, fake_pids, fake_mmlu, letter_ids, dev)
        sub_cls = ["passthrough", "algebra", "passthrough", "algebra"]
        sp_ed = _score_routed_mmlu_sparse(model, fake_pids, fake_mmlu, sub_cls, "algebra", base_mmlu, letter_ids, dev)
    assert _compose_routed_mmlu(fake_mmlu, sub_cls, base_mmlu, {"algebra": sp_ed}, seed=1) == _compose_routed_mmlu(
        fake_mmlu, sub_cls, base_mmlu, {"algebra": full_ed}, seed=1
    )

    # 5. Stratified pilot index selection
    synth_train = [
        {"id": f"p{i}", "level": 1 + (i % 5), "domain": MATH_DOMAINS[i % len(MATH_DOMAINS)]}
        for i in range(50)
    ]
    p_idx1 = _select_stratified_pilot_indices(synth_train, 15, seed=42)
    p_idx2 = _select_stratified_pilot_indices(synth_train, 15, seed=42)
    assert p_idx1 == p_idx2 and len(p_idx1) == 15 and len(set(p_idx1)) == 15 and p_idx1 == sorted(p_idx1), p_idx1
    assert {synth_train[i]["level"] for i in p_idx1} == {1, 2, 3, 4, 5}, p_idx1
    assert _select_stratified_pilot_indices(synth_train, 0, seed=42) == []
    assert _select_stratified_pilot_indices(synth_train, 99, seed=42) == list(range(50))

    # 6. Pilot summary & fail-fast gate (starved 4096 vs healthy 16384)
    pilot_items = [
        {"id": f"q{i}", "level": lv, "domain": MATH_DOMAINS[i % len(MATH_DOMAINS)]}
        for i, lv in enumerate([1, 1, 3, 3, 5, 5])
    ]
    raw_closed = {"token_ids": [1, 2, 3, 99, 98, 5, 6, 7], "finish_reason": "stop"}
    raw_open = {"token_ids": [1] * 20, "finish_reason": "length"}
    nt_bad = [
        {"level": it["level"], "strict_correct": (i < 5), "truncated": False, "loop_killed": False, "total_tokens": 500}
        for i, it in enumerate(pilot_items)
    ]
    th_bad = [
        {
            "level": it["level"],
            "strict_correct": (i < 2),
            "truncated": (i >= 2),
            "loop_killed": False,
            "total_tokens": tok_c,
        }
        for i, (it, tok_c) in enumerate(zip(pilot_items, [1500, 3800, 4096, 4096, 4096, 4096]))
    ]
    rep_bad = _summarize_think_pilot(
        pilot_items,
        nt_bad,
        th_bad,
        [raw_closed, raw_closed, raw_open, raw_open, raw_open, raw_open],
        toy_tok,
        eos_set=set(),
        think_end_id=99,
        max_new_tokens_think=4096,
        min_pilot_closed_rate=0.70,
        min_pilot_think_gain=0.0,
    )
    assert rep_bad["gate_passed"] is False and len(rep_bad["gate_reasons"]) >= 2, rep_bad
    assert rep_bad["budget_curve"]["2048"]["n_correct"] == 1 and rep_bad["budget_curve"]["4096"]["n_correct"] == 2, rep_bad

    nt_ok = [
        {"level": it["level"], "strict_correct": (i < 3), "truncated": False, "loop_killed": False, "total_tokens": 600}
        for i, it in enumerate(pilot_items)
    ]
    th_ok = [
        {
            "level": it["level"],
            "strict_correct": (i < 5),
            "truncated": (i == 5),
            "loop_killed": False,
            "total_tokens": tok_c,
        }
        for i, (it, tok_c) in enumerate(zip(pilot_items, [900, 2500, 5000, 7200, 11000, 16384]))
    ]
    rep_ok = _summarize_think_pilot(
        pilot_items,
        nt_ok,
        th_ok,
        [raw_closed] * 5 + [raw_open],
        toy_tok,
        eos_set=set(),
        think_end_id=99,
        max_new_tokens_think=16384,
        min_pilot_closed_rate=0.70,
        min_pilot_think_gain=0.0,
    )
    assert rep_ok["gate_passed"] is True and rep_ok["gate_reasons"] == [], rep_ok
    assert rep_ok["correct_total_tokens_p50"] == 5000.0 and rep_ok["correct_total_tokens_p95"] == 11000.0, rep_ok
    assert rep_ok["recommended_max_tokens_think"] == 12288, rep_ok
    assert set(rep_ok["per_level"].keys()) == {"1", "3", "5"}, rep_ok

    # 7. Atomic generation cache round-trip & prompt-length guard
    with tempfile.TemporaryDirectory() as td:
        c_path = _gen_cache_path(Path(td), "pilot_teacher_think", "/content/teacher_4b", 2, 16384, 42)
        sample_outs = [{"token_ids": [1, 2, 3], "finish_reason": "stop"}, {"token_ids": [4, 5], "finish_reason": "length"}]
        _save_gen_cache(c_path, sample_outs, [10, 12], {"tag": "smoke"})
        assert _try_load_gen_cache(c_path, [10, 12]) == sample_outs
        assert _try_load_gen_cache(c_path, [10, 13]) is None

    return {
        "mode": "smoke",
        "passed": True,
        "think_split_ok": True,
        "privileged_alignment_ok": True,
        "toy_loss_checkpoints": ckpts_out,
        "sparse_mmlu_exact_match": True,
        "pilot_helpers_ok": True,
        "cache_roundtrip_ok": True,
    }


# ------------------------------------------------------------------ Full GPU Thinking-Compression Experiment
def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    backend = _select_backend(args.backend)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        dtype = (
            torch.bfloat16
            if (dev.type == "cuda" and torch.cuda.is_bf16_supported())
            else (torch.float16 if dev.type == "cuda" else torch.float32)
        )
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    dtype_vllm_str = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype]

    math_train, math_test = load_math500_split(n_train=args.n_math_train, n_test=args.n_math_test, seed=args.seed)
    mmlu_test, mmlu_router_train = load_mmlu_split(
        n_test=args.n_mmlu_test, n_router_train=args.n_mmlu_router_train, seed=args.seed
    )

    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    assert len(tok_t) == len(tok_s), f"Tokenizer vocab mismatch: teacher={len(tok_t)} vs student={len(tok_s)}"
    eos_set = set(lr.eos_ids(tok_s))
    think_end_id = tok_s.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_id, int) or think_end_id == tok_s.unk_token_id:
        think_end_id = None

    cache_dir_str = args.cache_dir or os.environ.get("WSI_THINK_CACHE_DIR")
    cache_dir = Path(cache_dir_str) if cache_dir_str else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[think-cmp] Generation cache dir: {cache_dir}", flush=True)

    math_test_nt_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_test]
    math_train_nt_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_train]
    math_test_th_pids = [format_math500_think_prompt_ids(tok_s, it["problem"]) for it in math_test]
    math_train_th_pids = [format_math500_think_prompt_ids(tok_s, it["problem"]) for it in math_train]

    test_seeds = [int(args.seed) + i for i in range(len(math_test))]
    train_seeds = [int(args.seed) + 1000 + i for i in range(len(math_train))]

    teacher_hf = None

    def _ensure_teacher_hf():
        nonlocal teacher_hf
        if teacher_hf is None:
            try:
                teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
            except TypeError:
                teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
            for p in teacher_hf.parameters():
                p.requires_grad_(False)
        return teacher_hf

    def _generate_raw(
        model_dir: str,
        tok_obj: Any,
        pids: list[list[int]],
        seeds: list[int],
        max_tok: int,
        base_seed: int,
        hf_model: Any = None,
    ) -> list[dict[str, Any]]:
        if backend == "vllm":
            return _vllm_generate_fast(
                model_dir,
                tok_obj,
                pids,
                seeds,
                max_new_tokens=int(max_tok),
                dtype_str=dtype_vllm_str,
                base_seed=int(base_seed),
            )
        m = hf_model if hf_model is not None else _ensure_teacher_hf()
        return _hf_generate_subset(m, tok_obj, pids, seeds, max_new_tokens=int(max_tok))

    def _generate_or_load(
        tag: str,
        model_dir: str,
        tok_obj: Any,
        pids: list[list[int]],
        seeds: list[int],
        max_tok: int,
        base_seed: int,
        hf_model: Any = None,
    ) -> list[dict[str, Any]]:
        c_path = _gen_cache_path(cache_dir, tag, str(model_dir), len(pids), int(max_tok), int(base_seed))
        p_lens = [len(p) for p in pids]
        cached = _try_load_gen_cache(c_path, p_lens)
        if cached is not None:
            print(f"[think-cmp] Cache HIT ({tag}): {c_path}", flush=True)
            return cached
        outs = _generate_raw(model_dir, tok_obj, pids, seeds, int(max_tok), int(base_seed), hf_model=hf_model)
        _save_gen_cache(
            c_path,
            outs,
            p_lens,
            {"tag": tag, "model": str(model_dir), "max_new_tokens": int(max_tok), "seed": int(base_seed)},
        )
        return outs

    # ---- Phase 0: Stratified Pilot Probe & Fail-Fast Gate ----
    pilot_idxs = _select_stratified_pilot_indices(math_train, int(args.pilot_n), int(args.seed))
    pilot_nt_raw: list[dict[str, Any]] | None = None
    pilot_th_raw: list[dict[str, Any]] | None = None
    pilot_report: dict[str, Any] | None = None
    phase0_s = 0.0

    c_train_nt_full = _gen_cache_path(
        cache_dir, "teacher_nothink_train", args.teacher, len(math_train), int(args.max_new_tokens_nothink), int(args.seed) + 1000
    )
    c_train_th_full = _gen_cache_path(
        cache_dir, "teacher_think_train", args.teacher, len(math_train), int(args.max_new_tokens_think), int(args.seed) + 1007
    )
    cached_full_nt_train = _try_load_gen_cache(c_train_nt_full, [len(p) for p in math_train_nt_pids])
    cached_full_th_train = _try_load_gen_cache(c_train_th_full, [len(p) for p in math_train_th_pids])

    if pilot_idxs:
        print(
            f"[think-cmp] Phase 0: Pilot probe (n={len(pilot_idxs)}, "
            f"nothink@{args.max_new_tokens_nothink} vs think@{args.max_new_tokens_think})...",
            flush=True,
        )
        t_p0 = time.perf_counter()
        if cached_full_nt_train is not None:
            print(f"[think-cmp] Cache HIT (teacher_nothink_train -> pilot slice): {c_train_nt_full}", flush=True)
            pilot_nt_raw = [cached_full_nt_train[i] for i in pilot_idxs]
        else:
            pilot_nt_raw = _generate_or_load(
                "pilot_teacher_nothink",
                args.teacher,
                tok_t,
                [math_train_nt_pids[i] for i in pilot_idxs],
                [train_seeds[i] for i in pilot_idxs],
                int(args.max_new_tokens_nothink),
                int(args.seed) + 100,
            )

        if cached_full_th_train is not None:
            print(f"[think-cmp] Cache HIT (teacher_think_train -> pilot slice): {c_train_th_full}", flush=True)
            pilot_th_raw = [cached_full_th_train[i] for i in pilot_idxs]
        else:
            pilot_th_raw = _generate_or_load(
                "pilot_teacher_think",
                args.teacher,
                tok_t,
                [math_train_th_pids[i] for i in pilot_idxs],
                [train_seeds[i] for i in pilot_idxs],
                int(args.max_new_tokens_think),
                int(args.seed) + 107,
            )

        pilot_items = [math_train[i] for i in pilot_idxs]
        pilot_nt_rows = _grade_math500_rows(pilot_items, pilot_nt_raw, tok_s, eos_set, args.max_new_tokens_nothink)
        pilot_th_rows = _grade_math500_rows(pilot_items, pilot_th_raw, tok_s, eos_set, args.max_new_tokens_think)
        pilot_report = _summarize_think_pilot(
            pilot_items,
            pilot_nt_rows,
            pilot_th_rows,
            pilot_th_raw,
            tok_s,
            eos_set,
            think_end_id,
            max_new_tokens_think=int(args.max_new_tokens_think),
            min_pilot_closed_rate=float(args.min_pilot_closed_rate),
            min_pilot_think_gain=float(args.min_pilot_think_gain),
        )
        phase0_s = round(time.perf_counter() - t_p0, 2)
        print(
            f"[think-cmp] Pilot probe done in {phase0_s:.1f}s: "
            f"nothink={pilot_report['teacher_nothink_acc']*100:.1f}% vs think={pilot_report['teacher_think_acc']*100:.1f}% "
            f"({pilot_report['think_minus_nothink']*100:+.1f} pts), closed={pilot_report['closed_think_rate']*100:.1f}%, "
            f"p95={pilot_report['correct_total_tokens_p95']} -> rec={pilot_report['recommended_max_tokens_think']}, "
            f"gate_passed={pilot_report['gate_passed']} ({pilot_report['gate_reasons']})",
            flush=True,
        )

        if args.pilot_only or ((not args.allow_pilot_fail) and (not pilot_report["gate_passed"])):
            if teacher_hf is not None:
                del teacher_hf
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            return {
                "status": "pilot_only" if args.pilot_only else "pilot_stopped",
                "teacher_model": args.teacher,
                "student_model": args.student,
                "dtype": str(dtype),
                "backend": backend,
                "n_math_train": len(math_train),
                "n_math_test": len(math_test),
                "pilot_n": int(args.pilot_n),
                "max_new_tokens_nothink": int(args.max_new_tokens_nothink),
                "max_new_tokens_think": int(args.max_new_tokens_think),
                "timing_s": {
                    "phase0_pilot_s": phase0_s,
                    "total_s": round(time.perf_counter() - t0, 2),
                },
                "pilot_report": pilot_report,
            }

    # ---- Phase 1A: Teacher 4B Generations (Cached + Reusing Pilot Outputs) ----
    print(
        f"[think-cmp] Phase 1A: Teacher ({args.teacher}) generation ({backend}) — "
        f"nothink@{args.max_new_tokens_nothink} & think@{args.max_new_tokens_think}...",
        flush=True,
    )
    t_p1 = time.perf_counter()

    def _stitch_train(pilot_map: dict[int, dict[str, Any]], rest_idxs: list[int], rest_outs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert len(rest_idxs) == len(rest_outs), (len(rest_idxs), len(rest_outs))
        stitched: list[dict[str, Any] | None] = [None] * len(math_train)
        for idx_p, out_p in pilot_map.items():
            stitched[idx_p] = out_p
        for idx_r, out_r in zip(rest_idxs, rest_outs):
            stitched[idx_r] = out_r
        assert all(x is not None for x in stitched)
        return [x for x in stitched if x is not None]

    pilot_set = set(pilot_idxs)
    rest_train_idxs = [i for i in range(len(math_train)) if i not in pilot_set]

    c_test_nt = _gen_cache_path(
        cache_dir, "teacher_nothink_test", args.teacher, len(math_test), int(args.max_new_tokens_nothink), int(args.seed)
    )
    t_test_nt_raw = _try_load_gen_cache(c_test_nt, [len(p) for p in math_test_nt_pids])
    t_train_nt_raw = cached_full_nt_train
    pilot_nt_map = {idx_p: out_p for idx_p, out_p in zip(pilot_idxs, pilot_nt_raw)} if pilot_nt_raw is not None else {}

    if t_test_nt_raw is None and t_train_nt_raw is None:
        if pilot_nt_raw is not None:
            if rest_train_idxs:
                comb_pids = math_test_nt_pids + [math_train_nt_pids[i] for i in rest_train_idxs]
                comb_seeds = test_seeds + [train_seeds[i] for i in rest_train_idxs]
                comb_outs = _generate_raw(
                    args.teacher, tok_t, comb_pids, comb_seeds, int(args.max_new_tokens_nothink), int(args.seed)
                )
                n_t = len(math_test_nt_pids)
                t_test_nt_raw = comb_outs[:n_t]
                t_train_nt_raw = _stitch_train(pilot_nt_map, rest_train_idxs, comb_outs[n_t:])
            else:
                t_test_nt_raw = _generate_raw(
                    args.teacher, tok_t, math_test_nt_pids, test_seeds, int(args.max_new_tokens_nothink), int(args.seed)
                )
                t_train_nt_raw = [pilot_nt_map[i] for i in range(len(math_train))]
        else:
            comb_outs = _generate_raw(
                args.teacher,
                tok_t,
                math_test_nt_pids + math_train_nt_pids,
                test_seeds + train_seeds,
                int(args.max_new_tokens_nothink),
                int(args.seed),
            )
            n_t = len(math_test_nt_pids)
            t_test_nt_raw, t_train_nt_raw = comb_outs[:n_t], comb_outs[n_t:]
        _save_gen_cache(
            c_test_nt,
            t_test_nt_raw,
            [len(p) for p in math_test_nt_pids],
            {"tag": "teacher_nothink_test", "model": str(args.teacher), "max_new_tokens": int(args.max_new_tokens_nothink), "seed": int(args.seed)},
        )
        _save_gen_cache(
            c_train_nt_full,
            t_train_nt_raw,
            [len(p) for p in math_train_nt_pids],
            {"tag": "teacher_nothink_train", "model": str(args.teacher), "max_new_tokens": int(args.max_new_tokens_nothink), "seed": int(args.seed) + 1000},
        )
    else:
        if t_test_nt_raw is None:
            t_test_nt_raw = _generate_or_load(
                "teacher_nothink_test",
                args.teacher,
                tok_t,
                math_test_nt_pids,
                test_seeds,
                int(args.max_new_tokens_nothink),
                int(args.seed),
            )
        else:
            print(f"[think-cmp] Cache HIT (teacher_nothink_test): {c_test_nt}", flush=True)
        if t_train_nt_raw is None:
            if pilot_nt_raw is not None:
                rest_nt_outs = (
                    _generate_raw(
                        args.teacher,
                        tok_t,
                        [math_train_nt_pids[i] for i in rest_train_idxs],
                        [train_seeds[i] for i in rest_train_idxs],
                        int(args.max_new_tokens_nothink),
                        int(args.seed) + 1000,
                    )
                    if rest_train_idxs
                    else []
                )
                t_train_nt_raw = _stitch_train(pilot_nt_map, rest_train_idxs, rest_nt_outs)
                _save_gen_cache(
                    c_train_nt_full,
                    t_train_nt_raw,
                    [len(p) for p in math_train_nt_pids],
                    {"tag": "teacher_nothink_train", "model": str(args.teacher), "max_new_tokens": int(args.max_new_tokens_nothink), "seed": int(args.seed) + 1000},
                )
            else:
                t_train_nt_raw = _generate_or_load(
                    "teacher_nothink_train",
                    args.teacher,
                    tok_t,
                    math_train_nt_pids,
                    train_seeds,
                    int(args.max_new_tokens_nothink),
                    int(args.seed) + 1000,
                )
        elif not pilot_idxs:
            print(f"[think-cmp] Cache HIT (teacher_nothink_train): {c_train_nt_full}", flush=True)

    # Think generations as two independent cached blocks (test, then remaining train)
    t_test_th_raw = _generate_or_load(
        "teacher_think_test",
        args.teacher,
        tok_t,
        math_test_th_pids,
        test_seeds,
        int(args.max_new_tokens_think),
        int(args.seed) + 7,
    )
    t_train_th_raw = cached_full_th_train
    pilot_th_map = {idx_p: out_p for idx_p, out_p in zip(pilot_idxs, pilot_th_raw)} if pilot_th_raw is not None else {}
    if t_train_th_raw is None:
        if pilot_th_raw is not None:
            rest_th_outs = (
                _generate_raw(
                    args.teacher,
                    tok_t,
                    [math_train_th_pids[i] for i in rest_train_idxs],
                    [train_seeds[i] for i in rest_train_idxs],
                    int(args.max_new_tokens_think),
                    int(args.seed) + 1007,
                )
                if rest_train_idxs
                else []
            )
            t_train_th_raw = _stitch_train(pilot_th_map, rest_train_idxs, rest_th_outs)
            _save_gen_cache(
                c_train_th_full,
                t_train_th_raw,
                [len(p) for p in math_train_th_pids],
                {"tag": "teacher_think_train", "model": str(args.teacher), "max_new_tokens": int(args.max_new_tokens_think), "seed": int(args.seed) + 1007},
            )
        else:
            t_train_th_raw = _generate_or_load(
                "teacher_think_train",
                args.teacher,
                tok_t,
                math_train_th_pids,
                train_seeds,
                int(args.max_new_tokens_think),
                int(args.seed) + 1007,
            )
    elif not pilot_idxs:
        print(f"[think-cmp] Cache HIT (teacher_think_train): {c_train_th_full}", flush=True)

    t_test_nt_rows = _grade_math500_rows(math_test, t_test_nt_raw, tok_s, eos_set, args.max_new_tokens_nothink)
    t_test_th_rows = _grade_math500_rows(math_test, t_test_th_raw, tok_s, eos_set, args.max_new_tokens_think)
    t_nt_sum = _summarize_math500_arm(t_test_nt_rows)
    t_th_sum = _summarize_math500_arm(t_test_th_rows)
    t_th_diag = _compute_think_diagnostics(t_test_th_raw, tok_s, eos_set, think_end_id)
    print(
        f"[think-cmp] teacher_4b_nothink={t_nt_sum['accuracy']*100:.1f}% (tok={t_nt_sum['mean_tokens']}) | "
        f"teacher_4b_think={t_th_sum['accuracy']*100:.1f}% (tok={t_th_sum['mean_tokens']}, closed={t_th_diag['closed_think_rate']*100:.1f}%)",
        flush=True,
    )

    # Filter valid train traces for nothink vs think (post-</think> clean solutions)
    nothink_train_specs: list[dict[str, Any]] = []
    think_train_specs: list[dict[str, Any]] = []
    n_th_closed, n_th_correct, n_th_kept, n_th_too_long = 0, 0, 0, 0
    nt_train_correct, think_only_train_solved, nothink_only_train_solved, both_train_solved = 0, 0, 0, 0

    for i, it in enumerate(math_train):
        # 1. No-think train trace
        raw_nt = t_train_nt_raw[i]
        g_nt = dm.grade_math(raw_nt["token_ids"], raw_nt["finish_reason"], str(it["answer"]), tok_s, eos_set, args.max_new_tokens_nothink)
        if g_nt["correct"]:
            nt_train_correct += 1
        nt_ok = bool(g_nt["correct"] and (not g_nt["looped_killed"]) and int(g_nt["n_tokens"]) <= int(args.max_teacher_train_tokens))
        if nt_ok:
            txt_nt = tok_s.decode(lr.strip_eos(raw_nt["token_ids"], eos_set), skip_special_tokens=True).strip()
            if txt_nt:
                content = it["problem"].strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
                p_nt = format_math500_prompt_ids(tok_s, it["problem"])
                full_nt = _format_messages(
                    tok_s,
                    [{"role": "user", "content": content}, {"role": "assistant", "content": txt_nt}],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                nothink_train_specs.append({"id": it["id"], "domain": it["domain"], "ids": full_nt, "lo": max(0, len(p_nt) - 1), "hi": len(full_nt) - 1})

        # 2. Think train trace -> extract clean post-</think> solution & align privileged teacher sequence
        raw_th = t_train_th_raw[i]
        g_th = dm.grade_math(raw_th["token_ids"], raw_th["finish_reason"], str(it["answer"]), tok_s, eos_set, args.max_new_tokens_think)
        sp_th = _split_think_output(raw_th["token_ids"], tok_s, eos_set, think_end_id)
        if sp_th["closed_think"]:
            n_th_closed += 1
        if g_th["correct"]:
            n_th_correct += 1
        if g_th["correct"] and (not g_nt["correct"]):
            think_only_train_solved += 1
        elif g_nt["correct"] and (not g_th["correct"]):
            nothink_only_train_solved += 1
        elif g_th["correct"] and g_nt["correct"]:
            both_train_solved += 1
        if (
            g_th["correct"]
            and (not g_th["looped_killed"])
            and sp_th["closed_think"]
            and len(sp_th["post_think_text"]) > 0
        ):
            aligned = _build_aligned_privileged_pair(tok_s, it["problem"], sp_th["post_think_text"], sp_th["think_prefix_ids"])
            if (
                aligned["n_target_tokens"] <= int(args.max_teacher_train_tokens)
                and len(aligned["think_ids"]) <= int(args.max_think_context_tokens)
            ):
                n_th_kept += 1
                think_train_specs.append({"id": it["id"], "domain": it["domain"], **aligned})
            else:
                n_th_too_long += 1

    teacher_train_coverage = {
        "n_math_train": len(math_train),
        "teacher_nothink_train_acc": round(nt_train_correct / max(1, len(math_train)), 4),
        "teacher_think_train_acc": round(n_th_correct / max(1, len(math_train)), 4),
        "nothink_kept": len(nothink_train_specs),
        "think_closed": n_th_closed,
        "think_correct": n_th_correct,
        "think_kept": n_th_kept,
        "think_dropped_too_long": n_th_too_long,
        "think_dropped_oom": 0,
        "think_only_solved_on_train": think_only_train_solved,
        "nothink_only_solved_on_train": nothink_only_train_solved,
        "both_solved_on_train": both_train_solved,
    }
    print(f"[think-cmp] Teacher train coverage (pre-Phase 1B): {teacher_train_coverage}", flush=True)
    assert len(nothink_train_specs) >= 16 and len(think_train_specs) >= 16, teacher_train_coverage

    # ---- Phase 1B: Extract Top-32 4B Logits (No-Think Baseline, Post-Think No-Think KL, & Privileged Think KL) ----
    _ensure_teacher_hf()

    print(
        f"[think-cmp] Phase 1B: Extracting Top-32 4B logits — {len(nothink_train_specs)} nothink + "
        f"{len(think_train_specs)}x2 post-think (nothink KL & privileged <think> KL)...",
        flush=True,
    )

    def _extract_topk(full_ids: list[int], lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
        t_logits = _tail_logits(teacher_hf, ids_dev, lo)[: hi - lo].float()
        vals, idx = t_logits.topk(32, dim=-1)
        out = (idx.to(torch.int32).cpu(), vals.to(torch.float16).cpu())
        del ids_dev, t_logits
        return out

    nothink_distill_samples: list[dict[str, Any]] = []
    think_post_ntkl_samples: list[dict[str, Any]] = []
    think_priv_kl_samples: list[dict[str, Any]] = []
    n_th_oom_dropped = 0

    with torch.no_grad():
        for idx_i, sp in enumerate(nothink_train_specs):
            tk_i, tk_v = _extract_topk(sp["ids"], sp["lo"], sp["hi"])
            nothink_distill_samples.append(
                {"ids": sp["ids"], "lo": sp["lo"], "hi": sp["hi"], "topk_ids": tk_i, "topk_logits": tk_v}
            )
            if dev.type == "cuda" and (idx_i + 1) % 60 == 0:
                torch.cuda.empty_cache()

        for idx_i, sp in enumerate(think_train_specs):
            # Extract privileged long-context logits FIRST so any OOM drops the item from both think arms identically
            try:
                tk_pr_i, tk_pr_v = _extract_topk(sp["think_ids"], sp["lo_t"], sp["hi_t"])
            except torch.cuda.OutOfMemoryError:
                n_th_oom_dropped += 1
                print(
                    f"[think-cmp] WARN Phase 1B OOM on privileged seq {sp['id']} (len={len(sp['think_ids'])}); "
                    f"skipping in both think arms (dropped_oom={n_th_oom_dropped})",
                    flush=True,
                )
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            tk_nt_i, tk_nt_v = _extract_topk(sp["nothink_ids"], sp["lo_s"], sp["hi_s"])
            assert tk_nt_i.shape == tk_pr_i.shape, (tk_nt_i.shape, tk_pr_i.shape)
            think_post_ntkl_samples.append(
                {"ids": sp["nothink_ids"], "lo": sp["lo_s"], "hi": sp["hi_s"], "topk_ids": tk_nt_i, "topk_logits": tk_nt_v}
            )
            think_priv_kl_samples.append(
                {"ids": sp["nothink_ids"], "lo": sp["lo_s"], "hi": sp["hi_s"], "topk_ids": tk_pr_i, "topk_logits": tk_pr_v}
            )
            if dev.type == "cuda" and ((idx_i + 1) % 20 == 0 or len(sp["think_ids"]) > 8192):
                torch.cuda.empty_cache()

    teacher_train_coverage["think_kept"] = len(think_priv_kl_samples)
    teacher_train_coverage["think_dropped_oom"] = n_th_oom_dropped
    print(f"[think-cmp] Teacher train coverage (post-Phase 1B): {teacher_train_coverage}", flush=True)
    assert len(think_priv_kl_samples) >= 8, teacher_train_coverage

    del teacher_hf
    teacher_hf = None
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student 0.8B, Layer-12 Router, Base MMLU (400), Prose (24) & Disjoint Anchor (16) ----
    print(f"[think-cmp] Phase 2: Loading Student ({args.student}), Layer-12 router, base MMLU & disjoint anchor...", flush=True)
    t_p2 = time.perf_counter()
    try:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in student_m.parameters():
        p.requires_grad_(False)

    _, blocks = find_blocks(student_m)
    n_layers = len(blocks)
    all_layers = list(range(n_layers))
    router_layer = n_layers // 2

    mmlu_r_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_router_train]
    mmlu_test_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_test]
    prose_r_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_ROUTER_TRAIN]
    prose_held_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_HELDOUT]

    r_train_pids = math_train_nt_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = [it["domain"] for it in math_train] + ["passthrough"] * (len(mmlu_r_pids) + len(prose_r_pids))
    W_r, b_r = fit_ridge_router(_extract_prompt_hidden_vecs(student_m, r_train_pids, router_layer), r_train_labels, ROUTER_CLASSES, lam=1.0)
    routed_math_test = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, math_test_nt_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_mmlu_test = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, mmlu_test_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_prose_test = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, prose_held_pids, router_layer), W_r, b_r, ROUTER_CLASSES)

    router_report = {
        "router_layer": router_layer,
        "math_exact_domain_acc": round(sum(p == it["domain"] for p, it in zip(routed_math_test, math_test)) / len(math_test), 4),
        "math_any_domain_fire_rate": round(sum(p in MATH_DOMAINS for p in routed_math_test) / len(math_test), 4),
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_fire_rate": round(sum(p != "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_non_math_fire_rate": round(
            sum(p != "passthrough" for p, it in zip(routed_mmlu_test, mmlu_test) if not it["is_math_subject"])
            / max(1, sum(1 for it in mmlu_test if not it["is_math_subject"])),
            4,
        ),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_test) / len(routed_prose_test), 4),
    }
    print(f"[think-cmp] Router report: {router_report}", flush=True)

    letter_ids = ec._resolve_mmlu_letter_ids(tok_s)
    base_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
    base_mmlu_summary = _summarize_mmlu(mmlu_test, base_mmlu_raw, seed=args.seed)

    def _compute_base_logp(prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor | None], float]:
        seqs = _to_seq_tensors(
            [
                _format_messages(
                    tok_s,
                    [{"role": "user", "content": pr}, {"role": "assistant", "content": "Here is a concise explanation."}],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                for pr in prompts
            ],
            torch.device("cpu"),
        )
        logps: list[torch.Tensor | None] = []
        nll_s, cnt = 0.0, 0
        with torch.no_grad():
            for ids in seqs:
                logits = _tail_logits(student_m, ids.to(dev), 0)[:-1].float()
                lp = F.log_softmax(logits, dim=-1)
                nll_s += float(F.nll_loss(lp, ids[0, 1:].to(dev), reduction="sum"))
                cnt += int(ids.shape[1] - 1)
                logps.append(lp.to(dtype=torch.bfloat16, device="cpu"))
                del logits, lp
        return seqs, logps, nll_s / max(1, cnt)

    prose_full_ids, base_prose_logp, base_prose_nll = _compute_base_logp(PROSE_HELDOUT)
    anchor_full_ids, base_anchor_logp, _ = _compute_base_logp(PROSE_ROUTER_TRAIN)
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Train 3 Matched 24L r=16 Cartridges (default 160 steps x 4 = 640 passes, identical init) ----
    print(f"[think-cmp] Phase 3: Training 3 matched r={args.rank} 24L cartridges ({args.total_steps * args.grad_accum} passes each)...", flush=True)
    t_p3 = time.perf_counter()

    cartridge_specs = [
        ("nothink_distill_r16_24L", nothink_distill_samples),
        ("think_post_nothink_kl_r16_24L", think_post_ntkl_samples),
        ("think_privileged_kl_r16_24L", think_priv_kl_samples),
    ]

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        ckpt_paths: dict[str, Path] = {}
        expert_meta: dict[str, Any] = {}
        mmlu_raw_by_arm: dict[str, dict[str, Any]] = {}

        for arm_base, samples in cartridge_specs:
            torch.manual_seed(int(args.seed))
            res_c = _train_regularized_lora_expert(
                student_m,
                all_layers,
                samples,
                rank=int(args.rank),
                alpha_scale=float(args.rank * 2),
                total_steps=int(args.total_steps),
                grad_accum=int(args.grad_accum),
                lr_val=2.0e-4,
                weight_decay=0.05,
                lora_dropout=0.10,
                warmup_frac=0.15,
                distill_T=2.0,
                kl_weight=0.85,
                anchor_ids=anchor_full_ids,
                base_anchor_logp=base_anchor_logp,
                anchor_kl_weight=0.05,
                healthy_ids=prose_full_ids,
                base_healthy_logp=base_prose_logp,
                base_healthy_nll=base_prose_nll,
                seed=int(args.seed),
            )
            upd = res_c.pop("weights")
            with _temp_apply_weights(student_m, upd):
                if arm_base == "think_privileged_kl_r16_24L":
                    # Full MMLU score on privileged arm to report both routed & unrouted protection
                    mmlu_raw_by_arm[arm_base] = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
                else:
                    # Sparse MMLU on fired items (router fires on math-MMLU only)
                    fired_mmlu_cls = ["math" if c != "passthrough" else "passthrough" for c in routed_mmlu_test]
                    mmlu_raw_by_arm[arm_base] = _score_routed_mmlu_sparse(
                        student_m, mmlu_test_pids, mmlu_test, fired_mmlu_cls, "math", base_mmlu_raw, letter_ids, dev
                    )
            p_ckpt = exp_dir / f"{arm_base}.pt"
            torch.save(upd, p_ckpt)
            del upd
            ckpt_paths[arm_base] = p_ckpt
            expert_meta[arm_base] = res_c
            print(
                f"[think-cmp] {arm_base}: n={res_c['n_train_samples']} ({res_c['adapter_mb']}MB, {res_c['compile_time_s']:.1f}s) | "
                f"loss={res_c['step_losses_checkpoints'][0]}->{res_c['step_losses_checkpoints'][-1]} | pkl={res_c['healthy_kl']}",
                flush=True,
            )

        mmlu_arms: dict[str, Any] = {"base_08b_nothink": base_mmlu_summary}
        for idx_c, (arm_base, _) in enumerate(cartridge_specs):
            mmlu_arms[f"{arm_base}_routed"] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_test, base_mmlu_raw, {d: mmlu_raw_by_arm[arm_base] for d in MATH_DOMAINS}, seed=int(args.seed) + 1 + idx_c
            )
        mmlu_arms["think_privileged_kl_r16_24L_unrouted"] = _summarize_mmlu(
            mmlu_test,
            mmlu_raw_by_arm["think_privileged_kl_r16_24L"],
            base_score=base_mmlu_raw,
            fire_flags=[True] * len(mmlu_test),
            seed=int(args.seed) + 10,
        )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Evaluate All 7 Paired Arms on MATH-500 (n=200) ----
        base_think_tok = (
            int(args.max_new_tokens_base_think)
            if args.max_new_tokens_base_think is not None
            else int(args.max_new_tokens_think)
        )
        print(
            f"[think-cmp] Phase 4: Evaluating student arms on MATH-500 (n={len(math_test)}, {backend}, "
            f"base_think={base_think_tok})...",
            flush=True,
        )
        t_p4 = time.perf_counter()

        student_hf_eval = None
        if backend == "vllm":
            del student_m
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        else:
            student_hf_eval = student_m

        base_nt_raw = _generate_or_load(
            "base_08b_nothink_test",
            args.student,
            tok_s,
            math_test_nt_pids,
            test_seeds,
            int(args.max_new_tokens_nothink),
            int(args.seed),
            hf_model=student_hf_eval,
        )
        base_th_raw = _generate_or_load(
            "base_08b_think_test",
            args.student,
            tok_s,
            math_test_th_pids,
            test_seeds,
            base_think_tok,
            int(args.seed) + 7,
            hf_model=student_hf_eval,
        )

        bake_tmp_dir = Path("/content/_pocket_moe_think_cmp_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

        def _eval_ckpt(weights_path: Path, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
            upd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if backend == "vllm":
                _bake_full_checkpoint(Path(args.student), upd, bake_tmp_dir)
                del upd
                gc.collect()
                return _vllm_generate(
                    str(bake_tmp_dir),
                    tok_s,
                    p_ids,
                    sd_list,
                    max_new_tokens=args.max_new_tokens_nothink,
                    dtype_str=dtype_vllm_str,
                    base_seed=args.seed,
                )
            return _hf_generate_subset(
                student_hf_eval, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens_nothink, param_updates=upd
            )

        fired_math_idxs = [i for i, cls_nm in enumerate(routed_math_test) if cls_nm != "passthrough"]
        routed_raw_by_arm: dict[str, list[dict[str, Any]]] = {}
        try:
            for arm_base, _ in cartridge_specs:
                p_ckpt = ckpt_paths[arm_base]
                arm_out = list(base_nt_raw)
                if fired_math_idxs:
                    sub_outs = _eval_ckpt(p_ckpt, [math_test_nt_pids[i] for i in fired_math_idxs], [test_seeds[i] for i in fired_math_idxs])
                    for idx_i, o in zip(fired_math_idxs, sub_outs):
                        arm_out[idx_i] = o
                p_ckpt.unlink(missing_ok=True)
                routed_raw_by_arm[arm_base] = arm_out
        finally:
            if bake_tmp_dir.exists():
                shutil.rmtree(bake_tmp_dir, ignore_errors=True)
        phase4_s = round(time.perf_counter() - t_p4, 2)

    # ---- Phase 5: Grading, Thinking-Only Hard Subset, Token Compression & Paired Bootstrap CIs ----
    base_th_diag = _compute_think_diagnostics(base_th_raw, tok_s, eos_set, think_end_id)
    arm_rows: dict[str, list[dict[str, Any]]] = {
        "base_08b_nothink": _grade_math500_rows(math_test, base_nt_raw, tok_s, eos_set, args.max_new_tokens_nothink),
        "base_08b_think": _grade_math500_rows(math_test, base_th_raw, tok_s, eos_set, base_think_tok),
        "teacher_4b_nothink": t_test_nt_rows,
        "teacher_4b_think": t_test_th_rows,
    }
    for arm_base, _ in cartridge_specs:
        arm_rows[f"{arm_base}_routed"] = _grade_math500_rows(
            math_test, routed_raw_by_arm[arm_base], tok_s, eos_set, args.max_new_tokens_nothink, routed_labels=routed_math_test
        )

    think_only_test_idxs = [
        i for i in range(len(math_test)) if t_test_th_rows[i]["strict_correct"] and (not t_test_nt_rows[i]["strict_correct"])
    ]
    base_nt_acc = _summarize_math500_arm(arm_rows["base_08b_nothink"])["accuracy"]
    base_th_mean_tok = max(1.0, _summarize_math500_arm(arm_rows["base_08b_think"])["mean_tokens"])
    t_th_mean_tok = max(1.0, t_th_sum["mean_tokens"])
    t_th_flags = [bool(r["strict_correct"]) for r in t_test_th_rows]

    math500_arms: dict[str, dict[str, Any]] = {}
    for arm_nm, rows in arm_rows.items():
        sm = _summarize_math500_arm(rows, teacher_correct_flags=t_th_flags, base_acc=base_nt_acc, teacher_acc=t_th_sum["accuracy"])
        sm["think_only_hard_subset_n"] = len(think_only_test_idxs)
        sm["think_only_hard_subset_acc"] = (
            round(sum(1 for i in think_only_test_idxs if rows[i]["strict_correct"]) / len(think_only_test_idxs), 4)
            if think_only_test_idxs
            else None
        )
        sm["token_compression_vs_base_think"] = round(base_th_mean_tok / max(1.0, float(sm["mean_tokens"])), 3)
        sm["token_compression_vs_teacher_think"] = round(t_th_mean_tok / max(1.0, float(sm["mean_tokens"])), 3)
        if arm_nm == "base_08b_think":
            sm["think_diagnostics"] = base_th_diag
        elif arm_nm == "teacher_4b_think":
            sm["think_diagnostics"] = t_th_diag
        math500_arms[arm_nm] = sm

    prose_fire_frac = sum(1 for c in routed_prose_test if c != "passthrough") / max(1, len(routed_prose_test))
    prose_kl: dict[str, float] = {
        f"{arm_base}_routed": round(float(expert_meta[arm_base]["healthy_kl"]) * prose_fire_frac, 6)
        for arm_base, _ in cartridge_specs
    }
    prose_kl["think_privileged_kl_r16_24L_unrouted"] = float(expert_meta["think_privileged_kl_r16_24L"]["healthy_kl"])

    compile_time_s = {f"{arm_base}_s": float(expert_meta[arm_base]["compile_time_s"]) for arm_base, _ in cartridge_specs}

    comp_pairs = [
        ("teacher_think_vs_teacher_nothink", "teacher_4b_think", "teacher_4b_nothink"),
        ("base_think_vs_base_nothink", "base_08b_think", "base_08b_nothink"),
        ("nothink_distill_vs_base_nothink", "nothink_distill_r16_24L_routed", "base_08b_nothink"),
        ("think_post_ntkl_vs_nothink_distill", "think_post_nothink_kl_r16_24L_routed", "nothink_distill_r16_24L_routed"),
        ("think_privileged_kl_vs_think_post_ntkl", "think_privileged_kl_r16_24L_routed", "think_post_nothink_kl_r16_24L_routed"),
        ("think_privileged_kl_vs_nothink_distill", "think_privileged_kl_r16_24L_routed", "nothink_distill_r16_24L_routed"),
        ("think_privileged_kl_vs_base_think", "think_privileged_kl_r16_24L_routed", "base_08b_think"),
        ("think_post_ntkl_vs_base_think", "think_post_nothink_kl_r16_24L_routed", "base_08b_think"),
        ("think_privileged_kl_vs_base_nothink", "think_privileged_kl_r16_24L_routed", "base_08b_nothink"),
    ]
    paired_math500_200 = {
        lbl: _paired_bootstrap(arm_rows[a], arm_rows[b], seed=int(args.seed) + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    acc = lambda k: float(math500_arms[k]["accuracy"])  # noqa: E731
    nm_mmlu = lambda k: float(mmlu_arms[k]["non_math"]["accuracy"])  # noqa: E731
    routed_names = [f"{arm_base}_routed" for arm_base, _ in cartridge_specs]

    return {
        "status": "complete",
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "backend": backend,
        "n_math_train": len(math_train),
        "n_math_test": len(math_test),
        "pilot_n": int(args.pilot_n),
        "rank": int(args.rank),
        "total_steps": int(args.total_steps),
        "grad_accum": int(args.grad_accum),
        "max_new_tokens_nothink": int(args.max_new_tokens_nothink),
        "max_new_tokens_think": int(args.max_new_tokens_think),
        "max_new_tokens_base_think": base_think_tok,
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "max_think_context_tokens": int(args.max_think_context_tokens),
        "timing_s": {
            "phase0_pilot_s": phase0_s,
            "phase1_teacher_s": phase1_s,
            "phase2_router_base_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "pilot_report": pilot_report,
        "teacher_train_coverage": teacher_train_coverage,
        "compile_time_s": compile_time_s,
        "router_report": router_report,
        "expert_meta": expert_meta,
        "prose_kl": prose_kl,
        "math500_arms": math500_arms,
        "mmlu_arms": mmlu_arms,
        "paired_math500_200": paired_math500_200,
        "criteria": {
            "a_teacher_think_beats_teacher_nothink": bool(acc("teacher_4b_think") > acc("teacher_4b_nothink")),
            "b_think_post_beats_nothink_distill": bool(acc("think_post_nothink_kl_r16_24L_routed") > acc("nothink_distill_r16_24L_routed")),
            "c_privileged_kl_beats_post_nothink_kl": bool(
                acc("think_privileged_kl_r16_24L_routed") > acc("think_post_nothink_kl_r16_24L_routed")
            ),
            "d_privileged_kl_matches_or_beats_base_think": bool(acc("think_privileged_kl_r16_24L_routed") >= acc("base_08b_think")),
            "e_token_compression_gt_2x_vs_base_think": bool(
                math500_arms["think_privileged_kl_r16_24L_routed"]["token_compression_vs_base_think"] >= 2.0
            ),
            "f_zero_general_forgetting_all_routed": bool(
                all(prose_kl[nm] <= 0.005 and nm_mmlu(nm) >= nm_mmlu("base_08b_nothink") - 0.015 for nm in routed_names)
            ),
        },
        "per_item_math500": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU verification")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-math-train", type=int, default=300)
    ap.add_argument("--n-math-test", type=int, default=200)
    ap.add_argument("--n-mmlu-test", type=int, default=400)
    ap.add_argument("--n-mmlu-router-train", type=int, default=200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--total-steps", type=int, default=160)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-new-tokens-nothink", type=int, default=2048)
    ap.add_argument("--max-new-tokens-think", type=int, default=16384)
    ap.add_argument("--max-new-tokens-base-think", type=int, default=None)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1800)
    ap.add_argument("--max-think-context-tokens", type=int, default=16896)
    ap.add_argument("--pilot-n", type=int, default=40)
    ap.add_argument("--pilot-only", action="store_true", help="Run Phase 0 pilot probe and exit cleanly")
    ap.add_argument("--min-pilot-closed-rate", type=float, default=0.70)
    ap.add_argument("--min-pilot-think-gain", type=float, default=0.0)
    ap.add_argument("--allow-pilot-fail", action="store_true", help="Continue past Phase 0 even if fail-fast gate trips")
    ap.add_argument("--cache-dir", default=None, help="Directory for atomic per-phase generation caches (or WSI_THINK_CACHE_DIR)")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
        _print_think_compress_summary(payload)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
