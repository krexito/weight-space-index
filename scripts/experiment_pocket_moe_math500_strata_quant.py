"""Test A: High-Rank Regularized LoRA (`r=32`, `r=64`) Compressed to `S2_K` on Real `MATH-500`.

Tests on `HuggingFaceH4/MATH-500` (`n_train=300`, `n_test=200`) + `cais/mmlu` (`n_test=400`, `371` non-math) + `24` prose controls:
whether training higher capacity (`r=32` `41.3 MB`, `r=64` `82.6 MB`) and compressing post-hoc to `S2_K` (~`5.7 MB` / ~`11.4 MB`)
via `H64 + Hessian clipping + Block-LDL + 64-flip` beats `r=8 FP16` (`10.3 MB`) at matched wire bytes with `0` off-domain drift.

Paired arms evaluated under exact `lr.SAMPLING`:
  1. `base_08b`        : Unmodified `Qwen/Qwen3.5-0.8B` (`enable_thinking=False`)
  2. `teacher_4b`      : Ceiling `Qwen/Qwen3.5-4B` (`enable_thinking=False`)
  3. `r8_fp16_routed`  : Regularized `r=8` (`10.32 MB`) + L12 router
  4. `r32_fp16_routed` : Regularized `r=32` (`41.29 MB`) + L12 router
  5. `r32_s2k_routed`  : Arm 4 compressed post-hoc to `S2_K` (`~5.7 MB`) + L12 router
  6. `r64_fp16_routed` : Regularized `r=64` (`82.57 MB`) + L12 router
  7. `r64_s2k_routed`  : Arm 6 compressed post-hoc to `S2_K` (`~11.4 MB`, ~`r=8 FP16` size) + L12 router
  8. `r64_rtn2_routed` : Arm 6 compressed with naive unclipped 2-bit `RTN` (`~11.4 MB`) + L12 router

CPU smoke test::
    python scripts/experiment_pocket_moe_math500_strata_quant.py --smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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
    _block_prefixes,
    eval_healthy_drift,
    fit_ridge_router,
    predict_ridge_router,
)
from experiment_pocket_moe_controls import _paired_bootstrap  # noqa: E402
from experiment_pocket_moe_math500 import (  # noqa: E402
    MATH_DOMAINS,
    PROSE_HELDOUT,
    PROSE_ROUTER_TRAIN,
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
from experiment_pocket_moe_strata_quant import (  # noqa: E402
    _batched_hf_generate,
    _capture_lora_input_hessians,
    _compress_lora_factors,
    _train_regularized_lora_with_factors,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _PlantedReasoningLM,
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
)
from experiment_reasoning_distill_v2 import _format_messages  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402

ROUTER_FIT_CLASSES_MATH: tuple[str, ...] = (*MATH_DOMAINS, "pass_mmlu", "pass_prose")


def _predict_binary_math_router(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> list[str]:
    preds = predict_ridge_router(X, W, b, ROUTER_FIT_CLASSES_MATH)
    return ["math" if p in MATH_DOMAINS else "passthrough" for p in preds]


def _print_summary(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return

    print("\n=== TEST A: HIGH-RANK LoRA (r=8/32/64) + STRATA S2_K ON REAL MATH-500 ===", flush=True)
    print("Router:", payload.get("router_report"), flush=True)
    for arm_nm, st in payload.get("math500_arms", {}).items():
        mb = payload["arm_mb"].get(arm_nm)
        mb_s = f"{mb:5.2f}MB" if mb is not None else "  ref  "
        gap = st.get("teacher_gap_recovered")
        gap_s = f"{gap*100:5.1f}%" if gap is not None else "  n/a "
        mmlu_st = payload.get("mmlu_arms", {}).get(arm_nm, {})
        m_acc = f"{mmlu_st['accuracy']*100:4.1f}%" if "accuracy" in mmlu_st else "  n/a"
        m_fire = f"{mmlu_st['router_fire_rate']*100:4.1f}%" if "router_fire_rate" in mmlu_st else "  n/a"
        pkl = payload.get("prose_kl", {}).get(arm_nm, 0.0)
        print(
            f"  {arm_nm:18s} | {mb_s} | MATH={st['accuracy']*100:5.1f}% ({st['n_correct']:3d}/{st['n_total']}) "
            f"| gap={gap_s} | trunc={st['truncated_rate']*100:4.1f}% | MMLU={m_acc} (fire={m_fire}) | pkl={pkl:.5f}",
            flush=True,
        )
    if "r64_s2k_unrouted" in payload.get("mmlu_arms", {}):
        u = payload["mmlu_arms"]["r64_s2k_unrouted"]
        print(
            f"  r64_s2k_unrouted   | {payload['arm_mb']['r64_s2k_routed']:5.2f}MB | unrouted_ctrl MMLU={u['accuracy']*100:4.1f}% "
            f"(non_math={u['non_math']['accuracy']*100:4.1f}%, ABCD={u['all_400']['mean_abcd_prob_mass']*100:4.1f}%) "
            f"| pkl={payload['prose_kl'].get('r64_s2k_unrouted', 0.0):.5f}",
            flush=True,
        )

    print("\n=== PAIRED BOOTSTRAP CIs (MATH-500 n=200) ===", flush=True)
    for k, v in payload.get("paired_math500", {}).items():
        print(
            f"  {k:28s}: diff={v['diff']*100:+5.1f} pts CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] "
            f"(w={v['wins']}/l={v['losses']})",
            flush=True,
        )
    print("\nCriteria:", payload.get("criteria"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_math500"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of r=8/32/64 factor capture, S2_K scaling (~8x compression), & Strata vs RTN."""
    subjects = ["Algebra", "Intermediate Algebra", "Geometry"]
    synth_rows = [
        {
            "unique_id": f"m_{i}",
            "problem": f"Compute {i} + 1.",
            "answer": str(i + 1),
            "solution": f"Step 1: \\boxed{{{i + 1}}}.",
            "subject": subjects[i % 3],
            "level": (i % 5) + 1,
        }
        for i in range(60)
    ]
    tr, te = stratify_math500_rows(synth_rows, n_train=40, n_test=20, seed=42)
    assert len(tr) == 40 and len(te) == 20

    torch.manual_seed(42)
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    seqs = [[1, 2, 4, 10, 20, 20, 20, 20], [2, 1, 4, 11, 20, 20, 20, 20]]
    samples = []
    for s in seqs:
        ids_t = torch.tensor(s, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            tl = model(ids_t).logits[0, 3:7].float()
        v, idx = tl.topk(8, dim=-1)
        samples.append({"ids": s, "lo": 3, "hi": 7, "topk_ids": idx.to(torch.int32).cpu(), "topk_logits": v.to(torch.float16).cpu()})

    _, blocks = find_blocks(model)
    layers = list(range(len(blocks)))
    hess = _capture_lora_input_hessians(model, layers, samples)

    res8 = _train_regularized_lora_with_factors(model, layers, samples, rank=8, alpha_scale=16.0, total_steps=4, grad_accum=2, lr_val=1e-2, seed=42)
    res64 = _train_regularized_lora_with_factors(model, layers, samples, rank=64, alpha_scale=128.0, total_steps=4, grad_accum=2, lr_val=1e-2, seed=42)
    assert all(torch.equal(v.cpu(), orig[k]) for k, v in model.state_dict().items())

    _, wb_rtn, err_rtn = _compress_lora_factors(res64["factors"], hess, code_name="S2_K", method="rtn", device=torch.device("cpu"))
    _, wb_s2k, err_s2k = _compress_lora_factors(res64["factors"], hess, code_name="S2_K", method="strata", device=torch.device("cpu"))
    assert wb_rtn == wb_s2k and err_s2k < err_rtn
    r8_bytes = int(round(res8["adapter_mb"] * (1024**2)))
    r64_fp16_bytes = int(round(res64["adapter_mb"] * (1024**2)))
    from wsi.strata_quant import matrix_wire_bytes
    real_r64_fp16 = 2 * 64 * 1024 * 2  # LoRA A + B FP16 at K=1024
    real_r64_s2k = 2 * matrix_wire_bytes(64, 1024, "S2_K")
    assert real_r64_s2k < real_r64_fp16 * 0.16

    return {
        "mode": "smoke",
        "r8_fp16_bytes": r8_bytes,
        "r64_fp16_bytes": r64_fp16_bytes,
        "r64_s2k_bytes": wb_s2k,
        "real_k1024_compression": round(real_r64_fp16 / real_r64_s2k, 2),
        "err_rtn": round(err_rtn, 6),
        "err_strata_s2k": round(err_s2k, 6),
        "passed": True,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    torch.manual_seed(int(args.seed))

    math_train, math_test = load_math500_split(n_train=int(args.n_train), n_test=int(args.n_test), seed=int(args.seed))
    mmlu_test, mmlu_router_train = load_mmlu_split(
        n_test=int(args.n_mmlu_test), n_router_train=int(args.n_mmlu_router_train), seed=int(args.seed)
    )

    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok_s.pad_token_id is None:
        tok_s.pad_token = tok_s.eos_token
    eos_set = set(lr.eos_ids(tok_s))

    math_train_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_train]
    math_test_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_test]
    train_seeds = [int(args.seed) + 100 + i for i in range(len(math_train))]
    test_seeds = [int(args.seed) + 5000 + i for i in range(len(math_test))]

    # ---- Phase 1: Teacher 4B canonical traces + Top-32 logits (with optional cache) ----
    t_p1 = time.perf_counter()
    cache_p = Path(args.teacher_cache) if args.teacher_cache else None
    if cache_p is not None and cache_p.is_file():
        print(f"[testA] Phase 1: Loading cached teacher traces from {cache_p}...", flush=True)
        cache_obj = torch.load(cache_p, map_location="cpu", weights_only=False)
        t_train_raw = cache_obj["t_train_raw"]
        t_test_raw = cache_obj["t_test_raw"]
        kept_samples = cache_obj["kept_samples"]
        kept_frac = float(cache_obj["kept_frac"])
        n_fallback = int(cache_obj["n_fallback"])
    else:
        print(f"[testA] Phase 1: Teacher ({args.teacher}) generation + Top-32 logits...", flush=True)
        teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        for p in teacher_hf.parameters():
            p.requires_grad_(False)

        t_train_raw = _batched_hf_generate(
            teacher_hf, tok_s, math_train_pids, train_seeds, max_new_tokens=int(args.max_new_tokens), batch_size=int(args.eval_batch_size)
        )
        t_test_raw = _batched_hf_generate(
            teacher_hf, tok_s, math_test_pids, test_seeds, max_new_tokens=int(args.max_new_tokens), batch_size=int(args.eval_batch_size)
        )

        kept_pairs: list[tuple[dict[str, Any], list[int], str, bool]] = []
        fallback_pairs: list[tuple[dict[str, Any], list[int], str, bool]] = []
        for it, pids, raw_m in zip(math_train, math_train_pids, t_train_raw):
            g = dm.grade_math(raw_m["token_ids"], raw_m["finish_reason"], str(it["answer"]), tok_s, eos_set, int(args.max_new_tokens))
            txt = tok_s.decode(lr.strip_eos(raw_m["token_ids"], eos_set), skip_special_tokens=True).strip()
            if g["correct"] and (not g["looped_killed"]) and int(g["n_tokens"]) <= int(args.max_teacher_train_tokens) and len(txt) > 0:
                kept_pairs.append((it, pids, txt, False))
            elif str(it.get("solution", "")).strip():
                sol_txt = str(it["solution"]).strip()
                if "\\boxed{" not in sol_txt:
                    sol_txt += f"\n\nFinal answer: \\boxed{{{it['answer']}}}."
                fallback_pairs.append((it, pids, sol_txt, True))

        kept_frac = round(len(kept_pairs) / max(1, len(math_train)), 4)
        n_fallback = 0
        if len(kept_pairs) < 64 and fallback_pairs:
            need = min(64 - len(kept_pairs), len(fallback_pairs))
            kept_pairs.extend(fallback_pairs[:need])
            n_fallback = need

        kept_samples: list[dict[str, Any]] = []
        with torch.no_grad():
            for it, pids, resp_txt, is_fb in kept_pairs:
                user_c = it["problem"].strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
                full_ids = _format_messages(
                    tok_s,
                    [{"role": "user", "content": user_c}, {"role": "assistant", "content": resp_txt}],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                lo = max(0, len(pids) - 1)
                hi = len(full_ids) - 1
                if hi <= lo:
                    continue
                ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
                t_logits = _tail_logits(teacher_hf, ids_dev, lo)[: hi - lo].float()
                vals, idx = t_logits.topk(32, dim=-1)
                kept_samples.append({
                    "id": it["id"],
                    "domain": it["domain"],
                    "is_fallback": is_fb,
                    "ids": full_ids,
                    "lo": lo,
                    "hi": hi,
                    "topk_ids": idx.to(torch.int32).cpu(),
                    "topk_logits": vals.to(torch.float16).cpu(),
                })
                del ids_dev, t_logits, vals, idx

        del teacher_hf
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        if cache_p is not None:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            tmp_c = cache_p.with_suffix(".pt.tmp")
            torch.save(
                {
                    "t_train_raw": t_train_raw,
                    "t_test_raw": t_test_raw,
                    "kept_samples": kept_samples,
                    "kept_frac": kept_frac,
                    "n_fallback": n_fallback,
                },
                tmp_c,
            )
            os.replace(tmp_c, cache_p)

    teacher_test_rows = _grade_math500_rows(math_test, t_test_raw, tok_s, eos_set, int(args.max_new_tokens))
    teacher_correct_flags = [r["strict_correct"] for r in teacher_test_rows]
    teacher_summary = _summarize_math500_arm(teacher_test_rows)
    print(
        f"[testA] Teacher 4B: acc={teacher_summary['accuracy']*100:.1f}% | kept_train={len(kept_samples)} "
        f"(frac={kept_frac}, fallback={n_fallback})",
        flush=True,
    )
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student 0.8B, Layer-12 Router, Base MMLU-400 & Prose ----
    t_p2 = time.perf_counter()
    print(f"[testA] Phase 2: Loading Student ({args.student}) + L12 Router + MMLU-400...", flush=True)
    student_m = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in student_m.parameters():
        p.requires_grad_(False)
    _, blocks = find_blocks(student_m)
    all_layers = list(range(len(blocks)))
    router_layer = len(blocks) // 2

    mmlu_r_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_router_train]
    mmlu_test_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_test]
    prose_r_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_ROUTER_TRAIN]
    prose_held_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_HELDOUT]

    r_train_pids = math_train_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = [it["domain"] for it in math_train] + ["pass_mmlu"] * len(mmlu_r_pids) + ["pass_prose"] * len(prose_r_pids)
    W_r, b_r = fit_ridge_router(_extract_prompt_hidden_vecs(student_m, r_train_pids, router_layer), r_train_labels, ROUTER_FIT_CLASSES_MATH, lam=1.0)

    routed_math_test = _predict_binary_math_router(_extract_prompt_hidden_vecs(student_m, math_test_pids, router_layer), W_r, b_r)
    routed_mmlu_test = _predict_binary_math_router(_extract_prompt_hidden_vecs(student_m, mmlu_test_pids, router_layer), W_r, b_r)
    routed_prose_test = _predict_binary_math_router(_extract_prompt_hidden_vecs(student_m, prose_held_pids, router_layer), W_r, b_r)

    router_report = {
        "router_layer": router_layer,
        "math_fire_rate": round(sum(p == "math" for p in routed_math_test) / len(math_test), 4),
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_fire_rate": round(sum(p != "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_non_math_fire_rate": round(
            sum(p != "passthrough" for p, it in zip(routed_mmlu_test, mmlu_test) if not it["is_math_subject"])
            / max(1, sum(1 for it in mmlu_test if not it["is_math_subject"])),
            4,
        ),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_test) / len(routed_prose_test), 4),
    }
    print(f"[testA] Router: {router_report}", flush=True)

    letter_ids = ec._resolve_mmlu_letter_ids(tok_s)
    base_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
    base_mmlu_summary = _summarize_mmlu(mmlu_test, base_mmlu_raw, seed=int(args.seed))

    def _compute_base_logp(prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor | None], float]:
        seqs = _to_seq_tensors(
            [_format_messages(tok_s, [{"role": "user", "content": pr}, {"role": "assistant", "content": "Here is a concise explanation."}], add_generation_prompt=False, enable_thinking=False) for pr in prompts],
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
        return seqs, logps, nll_s / max(1, cnt)

    prose_full_ids, base_prose_logp, base_prose_nll = _compute_base_logp(PROSE_HELDOUT)
    anchor_full_ids, base_anchor_logp, _ = _compute_base_logp(PROSE_ROUTER_TRAIN)
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Train Regularized r=8, r=32, r=64 + Post-Hoc S2_K Compression ----
    t_p3 = time.perf_counter()
    print("[testA] Phase 3: Capturing Hessians + Training r=8/32/64 + S2_K compression...", flush=True)
    hess = _capture_lora_input_hessians(student_m, all_layers, kept_samples[:64])

    arm_names = [
        "r8_fp16_routed",
        "r32_fp16_routed",
        "r32_s2k_routed",
        "r64_fp16_routed",
        "r64_s2k_routed",
        "r64_rtn2_routed",
    ]
    arm_mb: dict[str, float] = {}
    quant_err_stats: dict[str, float] = {}
    train_stats: dict[str, Any] = {}
    mmlu_arms: dict[str, Any] = {"base_08b": base_mmlu_summary}
    prose_kl: dict[str, float] = {"base_08b": 0.0}

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        arm_paths: dict[str, Path] = {}

        def _eval_and_save_arm(a_nm: str, w_upd: dict[str, torch.Tensor], mb_val: float, seed_off: int, *, eval_unrouted_ctrl: bool = False) -> None:
            p_f = exp_dir / f"{a_nm}.pt"
            torch.save(w_upd, p_f)
            arm_paths[a_nm] = p_f
            arm_mb[a_nm] = round(float(mb_val), 4)
            with _temp_apply_weights(student_m, w_upd):
                raw_score = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
                drift = eval_healthy_drift(student_m, prose_full_ids, base_prose_logp, base_prose_nll)
            mmlu_arms[a_nm] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_test, base_mmlu_raw, {"math": raw_score}, seed=int(args.seed) + seed_off
            )
            prose_kl[a_nm] = round(float(drift["healthy_kl"]) * float(router_report["prose_fpr"]), 6)
            if eval_unrouted_ctrl:
                mmlu_arms["r64_s2k_unrouted"] = _summarize_mmlu(
                    mmlu_test, raw_score, base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=int(args.seed) + 99
                )
                prose_kl["r64_s2k_unrouted"] = round(float(drift["healthy_kl"]), 6)

        for idx_r, (rk, a_sc, fp16_nm) in enumerate([
            (8, 16.0, "r8_fp16_routed"),
            (32, 64.0, "r32_fp16_routed"),
            (64, 128.0, "r64_fp16_routed"),
        ]):
            torch.manual_seed(int(args.seed) + rk)
            res = _train_regularized_lora_with_factors(
                student_m,
                all_layers,
                kept_samples,
                rank=rk,
                alpha_scale=a_sc,
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
                seed=int(args.seed) + rk,
            )
            train_stats[fp16_nm] = {
                "compile_time_s": res["compile_time_s"],
                "step_losses_checkpoints": res["step_losses_checkpoints"],
                "unrouted_healthy_kl": res["healthy_kl"],
                "adapter_mb": res["adapter_mb"],
            }
            _eval_and_save_arm(fp16_nm, res["weights"], res["adapter_mb"], 10 + idx_r * 10)
            res.pop("weights")
            factors = res["factors"]

            if rk == 32:
                w_s2, wb_s2, err_s2 = _compress_lora_factors(factors, hess, code_name="S2_K", method="strata", device=dev)
                quant_err_stats["r32_s2k_routed"] = round(err_s2, 6)
                _eval_and_save_arm("r32_s2k_routed", w_s2, wb_s2 / (1024**2), 25)
                del w_s2
            elif rk == 64:
                w_s2, wb_s2, err_s2 = _compress_lora_factors(factors, hess, code_name="S2_K", method="strata", device=dev)
                quant_err_stats["r64_s2k_routed"] = round(err_s2, 6)
                _eval_and_save_arm("r64_s2k_routed", w_s2, wb_s2 / (1024**2), 35, eval_unrouted_ctrl=True)
                del w_s2

                w_r2, wb_r2, err_r2 = _compress_lora_factors(factors, hess, code_name="S2_K", method="rtn", device=dev)
                quant_err_stats["r64_rtn2_routed"] = round(err_r2, 6)
                _eval_and_save_arm("r64_rtn2_routed", w_r2, wb_r2 / (1024**2), 45)
                del w_r2

            del factors, res
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        del hess
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Paired HF Evaluation on MATH-500 (n=200) ----
        t_p4 = time.perf_counter()
        print(f"[testA] Phase 4: Evaluating 8 paired arms on MATH-500 (n={len(math_test)}, bs={args.eval_batch_size})...", flush=True)
        base_raw = _batched_hf_generate(
            student_m, tok_s, math_test_pids, test_seeds, max_new_tokens=int(args.max_new_tokens), batch_size=int(args.eval_batch_size)
        )
        fired_idxs = [i for i, cls_nm in enumerate(routed_math_test) if cls_nm == "math"]

        routed_raw: dict[str, list[dict[str, Any]]] = {}
        for a_nm in arm_names:
            arr = list(base_raw)
            p_f = arm_paths[a_nm]
            if fired_idxs:
                upd = torch.load(p_f, map_location="cpu", weights_only=True)
                sub_o = _batched_hf_generate(
                    student_m,
                    tok_s,
                    [math_test_pids[i] for i in fired_idxs],
                    [test_seeds[i] for i in fired_idxs],
                    max_new_tokens=int(args.max_new_tokens),
                    batch_size=int(args.eval_batch_size),
                    param_updates=upd,
                )
                for idx_i, o in zip(fired_idxs, sub_o):
                    arr[idx_i] = o
                del upd
            p_f.unlink(missing_ok=True)
            routed_raw[a_nm] = arr
        phase4_s = round(time.perf_counter() - t_p4, 2)

    # ---- Phase 5: Grading & Paired Bootstrap CIs ----
    base_rows = _grade_math500_rows(math_test, base_raw, tok_s, eos_set, int(args.max_new_tokens))
    base_acc = float(_summarize_math500_arm(base_rows)["accuracy"])
    t_acc = float(teacher_summary["accuracy"])

    arm_rows: dict[str, list[dict[str, Any]]] = {
        "base_08b": base_rows,
        "teacher_4b": teacher_test_rows,
    }
    arm_summaries: dict[str, dict[str, Any]] = {
        "base_08b": _summarize_math500_arm(base_rows, teacher_correct_flags=teacher_correct_flags, base_acc=base_acc, teacher_acc=t_acc),
        "teacher_4b": teacher_summary,
    }
    for a_nm in arm_names:
        rows = _grade_math500_rows(math_test, routed_raw[a_nm], tok_s, eos_set, int(args.max_new_tokens), routed_labels=routed_math_test)
        arm_rows[a_nm] = rows
        arm_summaries[a_nm] = _summarize_math500_arm(rows, teacher_correct_flags=teacher_correct_flags, base_acc=base_acc, teacher_acc=t_acc)

    comp_pairs = [
        ("r64_s2k_vs_r8_fp16", "r64_s2k_routed", "r8_fp16_routed"),
        ("r32_s2k_vs_r8_fp16", "r32_s2k_routed", "r8_fp16_routed"),
        ("r64_s2k_vs_r64_fp16", "r64_s2k_routed", "r64_fp16_routed"),
        ("r32_s2k_vs_r32_fp16", "r32_s2k_routed", "r32_fp16_routed"),
        ("r64_s2k_vs_r64_rtn2", "r64_s2k_routed", "r64_rtn2_routed"),
        ("r64_fp16_vs_r8_fp16", "r64_fp16_routed", "r8_fp16_routed"),
        ("r64_s2k_vs_base", "r64_s2k_routed", "base_08b"),
    ]
    paired_math500 = {
        lbl: _paired_bootstrap(arm_rows[a], arm_rows[b], seed=int(args.seed) + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    acc_of = lambda k: float(arm_summaries[k]["accuracy"])  # noqa: E731
    return {
        "experiment": "pocket_moe_math500_strata_quant",
        "teacher_model": args.teacher,
        "student_model": args.student,
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "kept_train_samples": len(kept_samples),
        "teacher_kept_frac": kept_frac,
        "teacher_n_fallback": n_fallback,
        "router_report": router_report,
        "arm_mb": arm_mb,
        "quant_err_stats": quant_err_stats,
        "train_stats": train_stats,
        "prose_kl": prose_kl,
        "math500_arms": arm_summaries,
        "mmlu_arms": mmlu_arms,
        "paired_math500": paired_math500,
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_router_s": phase2_s,
            "phase3_train_quant_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "criteria": {
            "a_r64_s2k_near_r8_fp16_bytes": arm_mb["r64_s2k_routed"] <= 1.15 * arm_mb["r8_fp16_routed"],
            "b_r64_s2k_beats_r8_fp16": acc_of("r64_s2k_routed") > acc_of("r8_fp16_routed"),
            "c_r32_s2k_beats_r8_fp16_at_smaller_size": acc_of("r32_s2k_routed") > acc_of("r8_fp16_routed") and arm_mb["r32_s2k_routed"] < arm_mb["r8_fp16_routed"],
            "d_r64_s2k_beats_rtn2": acc_of("r64_s2k_routed") > acc_of("r64_rtn2_routed"),
            "e_zero_nonmath_mmlu_drift": all(mmlu_arms[a]["non_math"]["delta_vs_base"] >= -0.005 for a in arm_names),
        },
        "per_item_math500": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-mmlu-test", type=int, default=400)
    ap.add_argument("--n-mmlu-router-train", type=int, default=200)
    ap.add_argument("--total-steps", type=int, default=160)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1350)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--teacher-cache", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
    _print_summary(payload)


if __name__ == "__main__":
    main()
