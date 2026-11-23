"""Controlled Real-World MATH-500 + MMLU Multi-Trace Pocket MoE Ablation (Qwen3.5-4B -> Qwen3.5-0.8B).

Tests on `HuggingFaceH4/MATH-500` (`n_train=300`, `n_test=200`) + `cais/mmlu` (`n_test=400`) + `24` held-out prose controls
whether Regularized Full-Depth (`24L`) + Multi-Trace (`M=3` teacher attempts/problem) improves real Olympiad math accuracy
while matching exact optimizer steps and sample passes (`total_steps` & `grad_accum` identical between `1trace` and `multitrace`):

  1. `base_08b`                              : Unmodified Qwen3.5-0.8B (`enable_thinking=False`)
  2. `teacher_4b`                            : Teacher ceiling Qwen3.5-4B (`enable_thinking=False`)
  3. `v1_global_bank_unreg_r64_24L`          : Previous #1 MATH-500 arm (unrouted `r=64`, all 24L, `82.6 MB`, 1x trace, unregularized)
  4. `routed_single_reg_1trace_r16_24L`      : 1 routed math cartridge (`r=16`, all 24L, `20.6 MB`) + L12 router, regularized, 1x trace (`160` steps x `4` = `640` passes)
  5. `routed_single_reg_multitrace_r16_24L`  : 1 routed math cartridge (`r=16`, all 24L, `20.6 MB`) + L12 router, regularized, M=3 multi-trace (**same `160` steps x `4` = `640` passes!**)
  6. `unrouted_single_reg_multitrace_r16_24L`: Same weights as #5 applied always-on (unrouted) to measure L12 router MMLU/prose protection
  7. `routed_moe3_reg_1trace_r16_24L`        : 3 routed domain cartridges (`algebra`, `advanced_algebra`, `discrete_geom`, `r=16`, all 24L) + L12 router, regularized, 1x trace (`60` steps x `4` = `240` passes/domain)
  8. `routed_moe3_reg_multitrace_r16_24L`    : 3 routed domain cartridges (`r=16`, all 24L) + L12 router, regularized, M=3 multi-trace (**same `60` steps x `4` = `240` passes/domain!**)

CPU smoke test::
    python scripts/experiment_pocket_moe_math500_multitrace.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_moe_math500_multitrace.py --teacher /content/teacher_4b --student /content/student_08b --out /content/pocket_moe_math500_multitrace_l4.json
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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
    SUBJECT_TO_DOMAIN,
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
    stratify_math500_rows,
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


# ------------------------------------------------------------------ Multi-Trace Pool Builder
def _collect_math500_trace_pools(
    math_train: list[dict[str, Any]],
    train_multi_raw: list[dict[str, Any]],
    *,
    m_attempts: int,
    tok: Any,
    eos_set: set[int],
    max_new_tokens: int,
    max_teacher_train_tokens: int,
) -> dict[str, Any]:
    """Partition `n_train x m_attempts` teacher outputs into `1trace` (attempt 0) and deduplicated `multitrace` pools."""
    assert len(train_multi_raw) == len(math_train) * int(m_attempts)
    primary_pairs: dict[str, list[tuple[dict[str, Any], str, bool]]] = {d: [] for d in MATH_DOMAINS}
    multitrace_pairs: dict[str, list[tuple[dict[str, Any], str, bool]]] = {d: [] for d in MATH_DOMAINS}
    fallback_cands: dict[str, list[tuple[dict[str, Any], str, bool]]] = {d: [] for d in MATH_DOMAINS}

    tot_by_dom = {d: 0 for d in MATH_DOMAINS}
    solved_m0_by_dom = {d: 0 for d in MATH_DOMAINS}
    solved_any_by_dom = {d: 0 for d in MATH_DOMAINS}

    for i, it in enumerate(math_train):
        d = it["domain"]
        tot_by_dom[d] += 1
        valid_txts: list[str] = []
        seen_txts: set[str] = set()
        m0_txt: str | None = None

        for m in range(int(m_attempts)):
            raw_m = train_multi_raw[i * int(m_attempts) + m]
            g = dm.grade_math(raw_m["token_ids"], raw_m["finish_reason"], str(it["answer"]), tok, eos_set, int(max_new_tokens))
            body = lr.strip_eos(raw_m["token_ids"], eos_set)
            txt = tok.decode(body, skip_special_tokens=True).strip()
            is_ok = bool(g["correct"] and (not g["looped_killed"]) and int(g["n_tokens"]) <= int(max_teacher_train_tokens) and len(txt) > 0)
            if m == 0 and is_ok:
                m0_txt = txt
            if is_ok and txt not in seen_txts:
                seen_txts.add(txt)
                valid_txts.append(txt)

        if m0_txt is not None:
            solved_m0_by_dom[d] += 1
            primary_pairs[d].append((it, m0_txt, False))
        elif it.get("solution", "").strip():
            sol_txt = it["solution"].strip()
            if "\\boxed{" not in sol_txt:
                sol_txt += f"\n\nFinal answer: \\boxed{{{it['answer']}}}."
            fallback_cands[d].append((it, sol_txt, True))

        if valid_txts:
            solved_any_by_dom[d] += 1
            for v_txt in valid_txts:
                multitrace_pairs[d].append((it, v_txt, False))

    fallback_counts_1trace = {d: 0 for d in MATH_DOMAINS}
    fallback_counts_multi = {d: 0 for d in MATH_DOMAINS}
    for d in MATH_DOMAINS:
        if len(primary_pairs[d]) < 16 and fallback_cands[d]:
            need = min(16 - len(primary_pairs[d]), len(fallback_cands[d]))
            primary_pairs[d].extend(fallback_cands[d][:need])
            fallback_counts_1trace[d] = need
        if len(multitrace_pairs[d]) < 16 and primary_pairs[d]:
            need = min(16 - len(multitrace_pairs[d]), len(primary_pairs[d]))
            multitrace_pairs[d].extend(primary_pairs[d][:need])
            fallback_counts_multi[d] = need

    tot_problems = max(1, sum(tot_by_dom.values()))
    tot_m0 = sum(solved_m0_by_dom.values())
    tot_any = sum(solved_any_by_dom.values())
    tot_multi_traces = sum(len(multitrace_pairs[d]) for d in MATH_DOMAINS)

    return {
        "primary_pairs": primary_pairs,
        "multitrace_pairs": multitrace_pairs,
        "primary_1trace_kept_frac": {
            **{d: round(solved_m0_by_dom[d] / max(1, tot_by_dom[d]), 4) for d in MATH_DOMAINS},
            "overall": round(tot_m0 / tot_problems, 4),
            "n_solved_overall": tot_m0,
        },
        "multitrace_problem_coverage_frac": {
            **{d: round(solved_any_by_dom[d] / max(1, tot_by_dom[d]), 4) for d in MATH_DOMAINS},
            "overall": round(tot_any / tot_problems, 4),
            "n_solved_overall": tot_any,
        },
        "trace_counts_per_domain": {
            "1trace": {**{d: len(primary_pairs[d]) for d in MATH_DOMAINS}, "total": sum(len(primary_pairs[d]) for d in MATH_DOMAINS)},
            "multitrace": {**{d: len(multitrace_pairs[d]) for d in MATH_DOMAINS}, "total": tot_multi_traces},
            "mean_traces_per_covered_problem": round(tot_multi_traces / max(1, tot_any), 3),
        },
        "fallback_counts": {"1trace": fallback_counts_1trace, "multitrace": fallback_counts_multi},
    }


# ------------------------------------------------------------------ Compact Summary Printer
def _print_math500_multitrace_summary(payload: dict[str, Any]) -> None:
    """Print concise human-readable MATH-500 + MMLU ablation table + single-line minified JSON omitting per-item rows."""
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return

    print("\n=== 1. COBERTURA MULTI-TRAZA (M=%s INTENTOS/PROBLEMA) & ROUTER ===" % payload.get("m_attempts"), flush=True)
    print("1-trace problem kept frac:", payload.get("primary_1trace_kept_frac"), flush=True)
    print("Multi-trace problem coverage frac:", payload.get("multitrace_problem_coverage_frac"), flush=True)
    print("Trace counts per domain:", payload.get("trace_counts_per_domain"), flush=True)
    print("Compile time (s):", payload.get("compile_time_s"), flush=True)
    print("Router report:", payload.get("router_report"), flush=True)

    prose_kl = payload.get("prose_kl", {})
    print("\n=== 2. MATH-500 (n=200, 8 BRAZOS PAREADOS, PRESUPUESTO IGUALADO 1-TRACE vs MULTITRACE) ===", flush=True)
    for arm_nm, st in payload.get("math500_arms", {}).items():
        pkl = prose_kl.get(arm_nm, 0.0)
        gap = st.get("teacher_gap_recovered")
        gap_s = f"{gap*100:5.1f}%" if gap is not None else "   n/a"
        print(
            f"  {arm_nm:39s} | acc={st['accuracy']*100:5.1f}% ({st['n_correct']:3d}/{st['n_total']}) "
            f"| gap={gap_s} | trunc={st['truncated_rate']*100:4.1f}% | med_tok={st['median_tokens']:6.1f} | pkl={pkl:.5f}",
            flush=True,
        )
        print(f"     per_domain={st['per_domain']} | per_level={st['per_level']}", flush=True)

    print("\n=== 3. RETENCION GENERAL EN MMLU (n=400, non_math=371) ===", flush=True)
    for arm_nm, vm in payload.get("mmlu_arms", {}).items():
        a_m, nm = vm["all_400"], vm["non_math"]
        d_nm = f"{nm['delta_vs_base']*100:+5.1f} pts" if "delta_vs_base" in nm else "     base"
        ex_nm = f"{nm['exact_pred_match_vs_base']*100:5.1f}%" if "exact_pred_match_vs_base" in nm else "  100.0%"
        print(
            f"  {arm_nm:39s} | MMLU={a_m['accuracy']*100:5.1f}% | non_math={nm['accuracy']*100:5.1f}% ({d_nm}, ex={ex_nm}) | "
            f"ABCD={a_m['mean_abcd_prob_mass']*100:5.1f}% | fire={vm['router_fire_rate']*100:5.1f}%",
            flush=True,
        )

    print("\n=== 4. COMPARACIONES PAREADAS MATH-500 (IC 95% BOOTSTRAP, n=200) ===", flush=True)
    for k, v in payload.get("paired_math500", {}).items():
        print(
            f"  {k:44s}: diff={v['diff']*100:+5.1f} pts "
            f"CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})",
            flush=True,
        )

    print("\nBest routed arm:", payload.get("best_routed_arm_name"), flush=True)
    print("Criteria:", payload.get("criteria"), flush=True)
    print("Timing (s):", payload.get("timing_s"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_math500"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


# ------------------------------------------------------------------ Deterministic CPU Smoke Test
def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of multi-trace pool coverage/dedup, equal-step regularized training, sparse MMLU & summary."""
    # 1. Stratified split & multi-trace pool collection (m_attempts=3)
    subjects = list(SUBJECT_TO_DOMAIN.keys())
    synth_rows = [
        {
            "unique_id": f"m_{i}",
            "problem": f"Compute {i} + 1.",
            "answer": str(i + 1),
            "solution": f"Step 1: {i}+1 = \\boxed{{{i + 1}}}.",
            "subject": subjects[i % len(subjects)],
            "level": (i % 5) + 1,
        }
        for i in range(90)
    ]
    tr, te = stratify_math500_rows(synth_rows, n_train=60, n_test=30, seed=42)
    assert len(tr) == 60 and len(te) == 30

    class _DummyTok:
        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            if not ids:
                return ""
            ans_id = ids[0]
            variant_id = ids[1] if len(ids) > 1 else 0
            return f"Variant {variant_id}: answer is \\boxed{{{ans_id}}}."

    dummy_tok = _DummyTok()
    fake_multi_raw: list[dict[str, Any]] = []
    for idx_p, it in enumerate(tr):
        gold_int = int(it["answer"])
        if idx_p % 3 == 0:
            # Solved on attempt 0 and attempt 1 (distinct variants), attempt 2 duplicate of attempt 1
            fake_multi_raw.append({"token_ids": [gold_int, 1, 99], "finish_reason": "stop"})
            fake_multi_raw.append({"token_ids": [gold_int, 2, 99], "finish_reason": "stop"})
            fake_multi_raw.append({"token_ids": [gold_int, 2, 99], "finish_reason": "stop"})
        elif idx_p % 3 == 1:
            # Attempt 0 wrong, attempt 1 & 2 correct (distinct) -> rescued ONLY in multitrace!
            fake_multi_raw.append({"token_ids": [gold_int + 500, 1, 99], "finish_reason": "stop"})
            fake_multi_raw.append({"token_ids": [gold_int, 2, 99], "finish_reason": "stop"})
            fake_multi_raw.append({"token_ids": [gold_int, 3, 99], "finish_reason": "stop"})
        else:
            # All 3 attempts wrong
            for m in range(3):
                fake_multi_raw.append({"token_ids": [gold_int + 500, m, 99], "finish_reason": "stop"})

    pools = _collect_math500_trace_pools(
        tr,
        fake_multi_raw,
        m_attempts=3,
        tok=dummy_tok,
        eos_set={99},
        max_new_tokens=64,
        max_teacher_train_tokens=50,
    )
    assert pools["primary_1trace_kept_frac"]["n_solved_overall"] == 20
    assert pools["multitrace_problem_coverage_frac"]["n_solved_overall"] == 40
    assert pools["trace_counts_per_domain"]["multitrace"]["total"] == 80  # 40 covered problems x 2 unique valid traces each

    # 2. Equal-step regularized LoRA check on _PlantedReasoningLM
    torch.manual_seed(19)
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    seqs = [[1, 2, 4, 10, 20, 20, 20, 20], [2, 1, 4, 11, 20, 20, 20, 20], [1, 3, 5, 14, 21, 21, 21, 21]]
    samples = []
    for s in seqs:
        ids_t = torch.tensor(s, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            t_log = model(ids_t).logits[0, 3:7].float()
        vals, tk_ids = t_log.topk(8, dim=-1)
        samples.append({
            "ids": s,
            "lo": 3,
            "hi": 7,
            "topk_ids": tk_ids.to(torch.int32).cpu(),
            "topk_logits": vals.to(torch.float16).cpu(),
        })
    h_ids = [torch.tensor([1, 2, 3, 1], dtype=torch.long).unsqueeze(0)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    res_1t = _train_regularized_lora_expert(
        model, [0, 1], samples[:2], rank=2, alpha_scale=4.0, total_steps=6, grad_accum=2, lr_val=2e-2,
        anchor_ids=h_ids, base_anchor_logp=base_lp, anchor_kl_weight=0.05,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll, seed=42,
    )
    res_mt = _train_regularized_lora_expert(
        model, [0, 1], samples, rank=2, alpha_scale=4.0, total_steps=6, grad_accum=2, lr_val=2e-2,
        anchor_ids=h_ids, base_anchor_logp=base_lp, anchor_kl_weight=0.05,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll, seed=42,
    )
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())
    assert res_1t["total_steps"] == res_mt["total_steps"] == 6
    assert res_mt["step_losses_checkpoints"][-1] < res_mt["step_losses_checkpoints"][0]

    return {
        "mode": "smoke",
        "passed": True,
        "coverage_jump_check": {
            "m0_solved": pools["primary_1trace_kept_frac"]["n_solved_overall"],
            "multi_solved": pools["multitrace_problem_coverage_frac"]["n_solved_overall"],
            "multi_unique_traces": pools["trace_counts_per_domain"]["multitrace"]["total"],
        },
        "equal_steps_lora_losses": res_mt["step_losses_checkpoints"],
    }


# ------------------------------------------------------------------ Full GPU MATH-500 Multi-Trace Ablation
def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    backend = _select_backend(args.backend)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if dev.type == "cuda" else torch.float32)
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    dtype_vllm_str = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype]

    math_train, math_test = load_math500_split(n_train=args.n_math_train, n_test=args.n_math_test, seed=args.seed)
    mmlu_test, mmlu_router_train = load_mmlu_split(n_test=args.n_mmlu_test, n_router_train=args.n_mmlu_router_train, seed=args.seed)

    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    assert len(tok_t) == len(tok_s), f"Tokenizer vocab mismatch: teacher={len(tok_t)} vs student={len(tok_s)}"
    eos_set = set(lr.eos_ids(tok_s))

    # ---- Phase 1: Single-Boot Teacher 4B Multi-Trace Generation (200 Test + 300xM Train) & Fast Top-32 Logits ----
    m_attempts = int(args.m_attempts)
    print(
        f"[math500-mt] Phase 1: Teacher ({args.teacher}) generation ({backend}, M={m_attempts} attempts/train, cap={args.max_teacher_train_tokens} tok)...",
        flush=True,
    )
    t_p1 = time.perf_counter()

    math_test_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_test]
    math_train_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_train]
    test_seeds = [int(args.seed) + i for i in range(len(math_test))]

    train_multi_pids: list[list[int]] = []
    train_multi_seeds: list[int] = []
    for i, p_ids in enumerate(math_train_pids):
        for m in range(m_attempts):
            train_multi_pids.append(p_ids)
            train_multi_seeds.append(int(args.seed) + 1000 + m * 1000 + i)

    teacher_hf = None
    if backend == "vllm":
        t_outs = _vllm_generate(
            args.teacher,
            tok_t,
            math_test_pids + train_multi_pids,
            test_seeds + train_multi_seeds,
            max_new_tokens=args.max_new_tokens,
            dtype_str=dtype_vllm_str,
            base_seed=args.seed,
        )
    else:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
        t_outs = _hf_generate_subset(
            teacher_hf,
            tok_t,
            math_test_pids + train_multi_pids,
            test_seeds + train_multi_seeds,
            max_new_tokens=args.max_new_tokens,
        )

    t_test_raw = t_outs[: len(math_test)]
    t_train_multi_raw = t_outs[len(math_test) :]

    teacher_test_rows = _grade_math500_rows(math_test, t_test_raw, tok_s, eos_set, args.max_new_tokens)
    teacher_correct_flags = [bool(r["strict_correct"]) for r in teacher_test_rows]
    teacher_summary = _summarize_math500_arm(teacher_test_rows, teacher_correct_flags=teacher_correct_flags)
    print(f"[math500-mt] teacher_4b test acc={teacher_summary['accuracy']*100:.1f}% ({teacher_summary['n_correct']}/{teacher_summary['n_total']})", flush=True)

    pool_info = _collect_math500_trace_pools(
        math_train,
        t_train_multi_raw,
        m_attempts=m_attempts,
        tok=tok_s,
        eos_set=eos_set,
        max_new_tokens=args.max_new_tokens,
        max_teacher_train_tokens=args.max_teacher_train_tokens,
    )
    primary_pairs = pool_info["primary_pairs"]
    multitrace_pairs = pool_info["multitrace_pairs"]
    print(
        f"[math500-mt] 1-trace kept={pool_info['primary_1trace_kept_frac']} | "
        f"M={m_attempts} coverage={pool_info['multitrace_problem_coverage_frac']} | "
        f"counts={pool_info['trace_counts_per_domain']}",
        flush=True,
    )

    if teacher_hf is None:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    unique_trace_map: dict[tuple[str, str], dict[str, Any]] = {}
    for d in MATH_DOMAINS:
        for it, resp_txt, is_fb in primary_pairs[d] + multitrace_pairs[d]:
            unique_trace_map.setdefault((it["id"], resp_txt), {"it": it, "resp_txt": resp_txt, "is_fb": is_fb})

    print(f"[math500-mt] Extracting Top-32 4B logits for {len(unique_trace_map)} unique valid traces...", flush=True)
    p_by_id = {it["id"]: pids for it, pids in zip(math_train, math_train_pids)}
    with torch.no_grad():
        for (item_id, resp_txt), meta in unique_trace_map.items():
            it = meta["it"]
            p_ids = p_by_id[item_id]
            user_content = it["problem"].strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
            full_ids = _format_messages(
                tok_s,
                [{"role": "user", "content": user_content}, {"role": "assistant", "content": resp_txt}],
                add_generation_prompt=False,
                enable_thinking=False,
            )
            lo = max(0, len(p_ids) - 1)
            hi = len(full_ids) - 1
            if hi <= lo:
                meta["sample"] = None
                continue
            ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
            t_logits = _tail_logits(teacher_hf, ids_dev, lo)[: hi - lo]
            vals, idx = t_logits.topk(32, dim=-1)
            meta["sample"] = {
                "id": it["id"],
                "domain": it["domain"],
                "subject": it["subject"],
                "level": int(it["level"]),
                "is_fallback": bool(meta["is_fb"]),
                "ids": full_ids,
                "lo": lo,
                "hi": hi,
                "topk_ids": idx.to(torch.int32).cpu(),
                "topk_logits": vals.to(torch.float16).cpu(),
            }
            del ids_dev, t_logits, vals, idx

    del teacher_hf
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    primary_by_domain: dict[str, list[dict[str, Any]]] = {d: [] for d in MATH_DOMAINS}
    multitrace_by_domain: dict[str, list[dict[str, Any]]] = {d: [] for d in MATH_DOMAINS}
    for d in MATH_DOMAINS:
        for it, resp_txt, _ in primary_pairs[d]:
            sp = unique_trace_map[(it["id"], resp_txt)].get("sample")
            if sp is not None:
                primary_by_domain[d].append(sp)
        for it, resp_txt, _ in multitrace_pairs[d]:
            sp = unique_trace_map[(it["id"], resp_txt)].get("sample")
            if sp is not None:
                multitrace_by_domain[d].append(sp)
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student 0.8B, Layer-12 Router, Base MMLU (400), Held-Out Prose (24) & Disjoint Anchor (16) ----
    print(f"[math500-mt] Phase 2: Loading Student ({args.student}), Layer-12 router, base MMLU & disjoint anchor...", flush=True)
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

    r_train_pids = math_train_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = [it["domain"] for it in math_train] + ["passthrough"] * (len(mmlu_r_pids) + len(prose_r_pids))

    W_r, b_r = fit_ridge_router(_extract_prompt_hidden_vecs(student_m, r_train_pids, router_layer), r_train_labels, ROUTER_CLASSES, lam=1.0)
    routed_math_test = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, math_test_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_mmlu_test = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, mmlu_test_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_prose_test = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, prose_held_pids, router_layer), W_r, b_r, ROUTER_CLASSES)

    router_report = {
        "router_layer": router_layer,
        "math_exact_domain_acc": round(sum(p == it["domain"] for p, it in zip(routed_math_test, math_test)) / len(math_test), 4),
        "math_any_domain_fire_rate": round(sum(p in MATH_DOMAINS for p in routed_math_test) / len(math_test), 4),
        "math_passthrough_miss_rate": round(sum(p == "passthrough" for p in routed_math_test) / len(math_test), 4),
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_fire_rate": round(sum(p != "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_non_math_fire_rate": round(
            sum(p != "passthrough" for p, it in zip(routed_mmlu_test, mmlu_test) if not it["is_math_subject"])
            / max(1, sum(1 for it in mmlu_test if not it["is_math_subject"])),
            4,
        ),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_test) / len(routed_prose_test), 4),
    }
    print(f"[math500-mt] Router report: {router_report}", flush=True)

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

    # ---- Phase 3: Train Equal-Pass-Budget 1-Trace vs Multi-Trace Arms + Sparse MMLU ----
    print("[math500-mt] Phase 3: Training equal-pass-budget 1-trace vs multi-trace arms...", flush=True)
    t_p3 = time.perf_counter()

    all_1trace_flat = [sp for d in MATH_DOMAINS for sp in primary_by_domain[d]]
    all_multitrace_flat = [sp for d in MATH_DOMAINS for sp in multitrace_by_domain[d]]

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        expert_meta: dict[str, Any] = {
            "v1_global_bank_unreg_r64_24L": {},
            "single_reg_1trace_r16_24L": {},
            "single_reg_multitrace_r16_24L": {},
            "moe3_reg_1trace_r16_24L": {},
            "moe3_reg_multitrace_r16_24L": {},
        }

        # 1. v1_global_bank_unreg_r64_24L (exact reproduction of previous #1 39.5% arm: r=64, 24L, 1x traces, unregularized)
        res_v1_bank = _train_regularized_lora_expert(
            student_m,
            all_layers,
            all_1trace_flat,
            rank=64,
            alpha_scale=64.0,
            total_steps=156,
            grad_accum=4,
            lr_val=3e-4,
            weight_decay=0.01,
            lora_dropout=0.0,
            warmup_frac=0.0,
            distill_T=1.5,
            kl_weight=0.70,
            anchor_ids=None,
            base_anchor_logp=None,
            anchor_kl_weight=0.0,
            healthy_ids=prose_full_ids,
            base_healthy_logp=base_prose_logp,
            base_healthy_nll=base_prose_nll,
            seed=int(args.seed) + 100,
        )
        upd_v1 = res_v1_bank.pop("weights")
        with _temp_apply_weights(student_m, upd_v1):
            v1_bank_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
        p_v1_bank = exp_dir / "v1_bank_r64.pt"
        torch.save(upd_v1, p_v1_bank)
        del upd_v1
        expert_meta["v1_global_bank_unreg_r64_24L"] = res_v1_bank
        print(
            f"[math500-mt] v1_global_bank_unreg_r64_24L: n={res_v1_bank['n_train_samples']} ({res_v1_bank['adapter_mb']}MB, {res_v1_bank['compile_time_s']:.1f}s) | "
            f"loss={res_v1_bank['step_losses_checkpoints'][0]}->{res_v1_bank['step_losses_checkpoints'][-1]} | pkl={res_v1_bank['healthy_kl']}",
            flush=True,
        )

        # 2. single_reg_1trace_r16_24L (160 steps x 4 = 640 passes on 1x traces, regularized)
        res_s1 = _train_regularized_lora_expert(
            student_m,
            all_layers,
            all_1trace_flat,
            rank=16,
            alpha_scale=32.0,
            total_steps=160,
            grad_accum=4,
            lr_val=2e-4,
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
            seed=int(args.seed) + 200,
        )
        upd_s1 = res_s1.pop("weights")
        with _temp_apply_weights(student_m, upd_s1):
            s1_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
        p_s1 = exp_dir / "single_reg_1trace.pt"
        torch.save(upd_s1, p_s1)
        del upd_s1
        expert_meta["single_reg_1trace_r16_24L"] = res_s1
        print(
            f"[math500-mt] single_reg_1trace_r16_24L: n={res_s1['n_train_samples']} ({res_s1['adapter_mb']}MB, {res_s1['compile_time_s']:.1f}s) | "
            f"loss={res_s1['step_losses_checkpoints'][0]}->{res_s1['step_losses_checkpoints'][-1]} | pkl={res_s1['healthy_kl']}",
            flush=True,
        )

        # 3. single_reg_multitrace_r16_24L (EXACT SAME 160 steps x 4 = 640 passes on M=3 traces, regularized)
        res_smt = _train_regularized_lora_expert(
            student_m,
            all_layers,
            all_multitrace_flat,
            rank=16,
            alpha_scale=32.0,
            total_steps=160,
            grad_accum=4,
            lr_val=2e-4,
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
            seed=int(args.seed) + 300,
        )
        upd_smt = res_smt.pop("weights")
        with _temp_apply_weights(student_m, upd_smt):
            smt_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
        p_smt = exp_dir / "single_reg_multitrace.pt"
        torch.save(upd_smt, p_smt)
        del upd_smt
        expert_meta["single_reg_multitrace_r16_24L"] = res_smt
        print(
            f"[math500-mt] single_reg_multitrace_r16_24L: n={res_smt['n_train_samples']} ({res_smt['adapter_mb']}MB, {res_smt['compile_time_s']:.1f}s) | "
            f"loss={res_smt['step_losses_checkpoints'][0]}->{res_smt['step_losses_checkpoints'][-1]} | pkl={res_smt['healthy_kl']}",
            flush=True,
        )

        # 4 & 5. 3-Domain Specialist MoEs: moe3_reg_1trace_r16_24L vs moe3_reg_multitrace_r16_24L (EXACT SAME 60 steps x 4 = 240 passes/domain)
        moe_paths: dict[tuple[str, str], Path] = {}
        moe_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {
            "moe3_reg_1trace_r16_24L": {},
            "moe3_reg_multitrace_r16_24L": {},
        }
        moe_specs = [
            ("moe3_reg_1trace_r16_24L", primary_by_domain, 400),
            ("moe3_reg_multitrace_r16_24L", multitrace_by_domain, 500),
        ]
        for d_idx, d in enumerate(MATH_DOMAINS):
            for tier_nm, pool_map, sd_off in moe_specs:
                res_d = _train_regularized_lora_expert(
                    student_m,
                    all_layers,
                    pool_map[d],
                    rank=16,
                    alpha_scale=32.0,
                    total_steps=60,
                    grad_accum=4,
                    lr_val=2e-4,
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
                    seed=int(args.seed) + sd_off + d_idx,
                )
                upd_d = res_d.pop("weights")
                with _temp_apply_weights(student_m, upd_d):
                    moe_mmlu_scores[tier_nm][d] = _score_routed_mmlu_sparse(
                        student_m, mmlu_test_pids, mmlu_test, routed_mmlu_test, d, base_mmlu_raw, letter_ids, dev
                    )
                p_d = exp_dir / f"{tier_nm}_{d}.pt"
                torch.save(upd_d, p_d)
                del upd_d
                moe_paths[(tier_nm, d)] = p_d
                expert_meta[tier_nm][d] = res_d
                print(
                    f"[math500-mt] {tier_nm}/{d}: n={res_d['n_train_samples']} ({res_d['adapter_mb']}MB, {res_d['compile_time_s']:.1f}s) | "
                    f"loss={res_d['step_losses_checkpoints'][0]}->{res_d['step_losses_checkpoints'][-1]} | pkl={res_d['healthy_kl']}",
                    flush=True,
                )

        mmlu_arms: dict[str, Any] = {
            "base_08b": base_mmlu_summary,
            "v1_global_bank_unreg_r64_24L": _summarize_mmlu(
                mmlu_test, v1_bank_mmlu_raw, base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=int(args.seed) + 1
            ),
            "routed_single_reg_1trace_r16_24L": _compose_routed_mmlu(
                mmlu_test, routed_mmlu_test, base_mmlu_raw, {d: s1_mmlu_raw for d in MATH_DOMAINS}, seed=int(args.seed) + 2
            ),
            "routed_single_reg_multitrace_r16_24L": _compose_routed_mmlu(
                mmlu_test, routed_mmlu_test, base_mmlu_raw, {d: smt_mmlu_raw for d in MATH_DOMAINS}, seed=int(args.seed) + 3
            ),
            "unrouted_single_reg_multitrace_r16_24L": _summarize_mmlu(
                mmlu_test, smt_mmlu_raw, base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=int(args.seed) + 4
            ),
            "routed_moe3_reg_1trace_r16_24L": _compose_routed_mmlu(
                mmlu_test, routed_mmlu_test, base_mmlu_raw, moe_mmlu_scores["moe3_reg_1trace_r16_24L"], seed=int(args.seed) + 5
            ),
            "routed_moe3_reg_multitrace_r16_24L": _compose_routed_mmlu(
                mmlu_test, routed_mmlu_test, base_mmlu_raw, moe_mmlu_scores["moe3_reg_multitrace_r16_24L"], seed=int(args.seed) + 6
            ),
        }
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Evaluate All 8 Paired Arms on MATH-500 (n=200) ----
        print(f"[math500-mt] Phase 4: Evaluating 8 paired arms on MATH-500 (n={len(math_test)}, {backend})...", flush=True)
        t_p4 = time.perf_counter()

        if backend == "vllm":
            del student_m
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            base_raw = _vllm_generate(args.student, tok_s, math_test_pids, test_seeds, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
        else:
            base_raw = _hf_generate_subset(student_m, tok_s, math_test_pids, test_seeds, max_new_tokens=args.max_new_tokens)

        bake_tmp_dir = Path("/content/_pocket_moe_math500_mt_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

        def _eval_ckpt(weights_path: Path, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
            upd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if backend == "vllm":
                _bake_full_checkpoint(Path(args.student), upd, bake_tmp_dir)
                del upd
                gc.collect()
                return _vllm_generate(str(bake_tmp_dir), tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
            return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, param_updates=upd)

        v1_bank_raw = _eval_ckpt(p_v1_bank, math_test_pids, test_seeds)
        p_v1_bank.unlink(missing_ok=True)

        # Evaluate single_reg_1trace on fired math items
        fired_math_idxs = [i for i, cls_nm in enumerate(routed_math_test) if cls_nm != "passthrough"]
        s1_routed_raw: list[dict[str, Any]] = list(base_raw)
        if fired_math_idxs:
            s1_sub = _eval_ckpt(p_s1, [math_test_pids[i] for i in fired_math_idxs], [test_seeds[i] for i in fired_math_idxs])
            for idx_i, o in zip(fired_math_idxs, s1_sub):
                s1_routed_raw[idx_i] = o
        p_s1.unlink(missing_ok=True)

        # Evaluate single_reg_multitrace on all 200 items so unrouted & routed share the exact same outputs
        smt_unrouted_raw = _eval_ckpt(p_smt, math_test_pids, test_seeds)
        p_smt.unlink(missing_ok=True)
        smt_routed_raw = [
            base_raw[i] if cls_nm == "passthrough" else smt_unrouted_raw[i]
            for i, cls_nm in enumerate(routed_math_test)
        ]

        moe_raw_by_tier: dict[str, list[dict[str, Any]]] = {}
        for tier_nm, *_rest in moe_specs:
            tier_raw: list[dict[str, Any] | None] = [None] * len(math_test)
            for i, cls_nm in enumerate(routed_math_test):
                if cls_nm == "passthrough":
                    tier_raw[i] = base_raw[i]
            for d in MATH_DOMAINS:
                idxs = [i for i, cls_nm in enumerate(routed_math_test) if cls_nm == d]
                p_d = moe_paths[(tier_nm, d)]
                if idxs:
                    sub_outs = _eval_ckpt(p_d, [math_test_pids[i] for i in idxs], [test_seeds[i] for i in idxs])
                    for idx_i, o in zip(idxs, sub_outs):
                        tier_raw[idx_i] = o
                p_d.unlink(missing_ok=True)
            moe_raw_by_tier[tier_nm] = [r for r in tier_raw if r is not None]

        if bake_tmp_dir.exists():
            shutil.rmtree(bake_tmp_dir, ignore_errors=True)
        phase4_s = round(time.perf_counter() - t_p4, 2)

    arm_raw_map: dict[str, tuple[list[dict[str, Any]], list[str] | None, bool]] = {
        "base_08b": (base_raw, None, False),
        "teacher_4b": (t_test_raw, None, True),
        "v1_global_bank_unreg_r64_24L": (v1_bank_raw, None, False),
        "routed_single_reg_1trace_r16_24L": (s1_routed_raw, routed_math_test, False),
        "routed_single_reg_multitrace_r16_24L": (smt_routed_raw, routed_math_test, False),
        "unrouted_single_reg_multitrace_r16_24L": (smt_unrouted_raw, None, False),
        "routed_moe3_reg_1trace_r16_24L": (moe_raw_by_tier["moe3_reg_1trace_r16_24L"], routed_math_test, False),
        "routed_moe3_reg_multitrace_r16_24L": (moe_raw_by_tier["moe3_reg_multitrace_r16_24L"], routed_math_test, False),
    }

    math500_rows: dict[str, list[dict[str, Any]]] = {}
    base_rows = _grade_math500_rows(math_test, base_raw, tok_s, eos_set, args.max_new_tokens)
    base_acc = sum(1 for r in base_rows if r["strict_correct"]) / len(base_rows)
    teacher_acc = teacher_summary["accuracy"]

    math500_arms: dict[str, dict[str, Any]] = {}
    for arm_nm, (raw_l, r_lbls, is_teacher) in arm_raw_map.items():
        if is_teacher:
            rows = teacher_test_rows
            sm = teacher_summary
        else:
            rows = base_rows if arm_nm == "base_08b" else _grade_math500_rows(math_test, raw_l, tok_s, eos_set, args.max_new_tokens, routed_labels=r_lbls)
            sm = _summarize_math500_arm(rows, teacher_correct_flags=teacher_correct_flags, base_acc=base_acc, teacher_acc=teacher_acc)
        math500_rows[arm_nm] = rows
        math500_arms[arm_nm] = sm

    def _routed_prose_kl_for_moe(tier_key: str) -> float:
        tot = sum(float(expert_meta[tier_key][cls_nm]["healthy_kl"]) for cls_nm in routed_prose_test if cls_nm in MATH_DOMAINS)
        return round(tot / max(1, len(routed_prose_test)), 6)

    prose_fire_frac = sum(1 for cls_nm in routed_prose_test if cls_nm != "passthrough") / max(1, len(routed_prose_test))
    prose_kl = {
        "v1_global_bank_unreg_r64_24L": float(expert_meta["v1_global_bank_unreg_r64_24L"]["healthy_kl"]),
        "routed_single_reg_1trace_r16_24L": round(prose_fire_frac * float(expert_meta["single_reg_1trace_r16_24L"]["healthy_kl"]), 6),
        "routed_single_reg_multitrace_r16_24L": round(prose_fire_frac * float(expert_meta["single_reg_multitrace_r16_24L"]["healthy_kl"]), 6),
        "unrouted_single_reg_multitrace_r16_24L": float(expert_meta["single_reg_multitrace_r16_24L"]["healthy_kl"]),
        "routed_moe3_reg_1trace_r16_24L": _routed_prose_kl_for_moe("moe3_reg_1trace_r16_24L"),
        "routed_moe3_reg_multitrace_r16_24L": _routed_prose_kl_for_moe("moe3_reg_multitrace_r16_24L"),
    }

    compile_time_s = {
        "v1_global_bank_unreg_r64_24L_s": float(expert_meta["v1_global_bank_unreg_r64_24L"]["compile_time_s"]),
        "single_reg_1trace_r16_24L_s": float(expert_meta["single_reg_1trace_r16_24L"]["compile_time_s"]),
        "single_reg_multitrace_r16_24L_s": float(expert_meta["single_reg_multitrace_r16_24L"]["compile_time_s"]),
        "moe3_reg_1trace_r16_24L_mean_per_domain_s": round(
            sum(float(expert_meta["moe3_reg_1trace_r16_24L"][d]["compile_time_s"]) for d in MATH_DOMAINS) / len(MATH_DOMAINS), 2
        ),
        "moe3_reg_multitrace_r16_24L_mean_per_domain_s": round(
            sum(float(expert_meta["moe3_reg_multitrace_r16_24L"][d]["compile_time_s"]) for d in MATH_DOMAINS) / len(MATH_DOMAINS), 2
        ),
    }

    routed_candidates = [
        "routed_single_reg_1trace_r16_24L",
        "routed_single_reg_multitrace_r16_24L",
        "routed_moe3_reg_1trace_r16_24L",
        "routed_moe3_reg_multitrace_r16_24L",
    ]
    best_routed_arm = max(routed_candidates, key=lambda k: math500_arms[k]["accuracy"])

    comp_pairs = [
        ("single_multitrace_vs_single_1trace", "routed_single_reg_multitrace_r16_24L", "routed_single_reg_1trace_r16_24L"),
        ("moe3_multitrace_vs_moe3_1trace", "routed_moe3_reg_multitrace_r16_24L", "routed_moe3_reg_1trace_r16_24L"),
        ("single_multitrace_vs_v1_bank_unreg", "routed_single_reg_multitrace_r16_24L", "v1_global_bank_unreg_r64_24L"),
        ("moe3_multitrace_vs_v1_bank_unreg", "routed_moe3_reg_multitrace_r16_24L", "v1_global_bank_unreg_r64_24L"),
        ("moe3_multitrace_vs_single_multitrace", "routed_moe3_reg_multitrace_r16_24L", "routed_single_reg_multitrace_r16_24L"),
        ("single_1trace_vs_v1_bank_unreg", "routed_single_reg_1trace_r16_24L", "v1_global_bank_unreg_r64_24L"),
        ("single_multitrace_vs_base", "routed_single_reg_multitrace_r16_24L", "base_08b"),
        ("moe3_multitrace_vs_base", "routed_moe3_reg_multitrace_r16_24L", "base_08b"),
        ("v1_bank_unreg_vs_base", "v1_global_bank_unreg_r64_24L", "base_08b"),
    ]
    paired_math500 = {
        lbl: _paired_bootstrap(math500_rows[a], math500_rows[b], seed=int(args.seed) + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    acc_m = lambda k: float(math500_arms[k]["accuracy"])  # noqa: E731
    nm_acc = lambda k: float(mmlu_arms[k]["non_math"]["accuracy"])  # noqa: E731

    return {
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "backend": backend,
        "m_attempts": m_attempts,
        "max_new_tokens": int(args.max_new_tokens),
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_router_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "dataset_summary": {
            "math500_train": len(math_train),
            "math500_test": len(math_test),
            "mmlu_test": len(mmlu_test),
            "mmlu_router_train": len(mmlu_router_train),
            "prose_heldout": len(PROSE_HELDOUT),
            "prose_anchor_train": len(PROSE_ROUTER_TRAIN),
        },
        "primary_1trace_kept_frac": pool_info["primary_1trace_kept_frac"],
        "multitrace_problem_coverage_frac": pool_info["multitrace_problem_coverage_frac"],
        "trace_counts_per_domain": pool_info["trace_counts_per_domain"],
        "fallback_counts": pool_info["fallback_counts"],
        "compile_time_s": compile_time_s,
        "router_report": router_report,
        "expert_meta": expert_meta,
        "prose_kl": prose_kl,
        "math500_arms": math500_arms,
        "mmlu_arms": mmlu_arms,
        "paired_math500": paired_math500,
        "best_routed_arm_name": best_routed_arm,
        "criteria": {
            "a_multitrace_beats_1trace_single_equal_passes": bool(
                acc_m("routed_single_reg_multitrace_r16_24L") > acc_m("routed_single_reg_1trace_r16_24L")
            ),
            "b_multitrace_beats_1trace_moe3_equal_passes": bool(
                acc_m("routed_moe3_reg_multitrace_r16_24L") > acc_m("routed_moe3_reg_1trace_r16_24L")
            ),
            "c_best_routed_beats_v1_bank_unreg": bool(
                acc_m(best_routed_arm) > acc_m("v1_global_bank_unreg_r64_24L")
            ),
            "d_routing_protects_non_math_mmlu_and_prose": bool(
                all(
                    nm_acc(nm) >= nm_acc("base_08b") - 0.015 and prose_kl[nm] <= 0.005
                    for nm in routed_candidates
                )
            ),
        },
        "per_item_math500": math500_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU verification")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--m-attempts", type=int, default=3)
    ap.add_argument("--n-math-train", type=int, default=300)
    ap.add_argument("--n-math-test", type=int, default=200)
    ap.add_argument("--n-mmlu-test", type=int, default=400)
    ap.add_argument("--n-mmlu-router-train", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1800)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
        _print_math500_multitrace_summary(payload)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
