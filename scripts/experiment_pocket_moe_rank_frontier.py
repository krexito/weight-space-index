"""Single-Trace 144p Ultra-Compact Rank Frontier (r=8 -> r=4 -> r=2 -> r=1 + r=2@4e-4, Qwen3.5-4B -> 0.8B, K=6).

Tests the pure low-rank compression frontier of the winning `singletrace` + `144p` (`48` steps x `grad_accum=3` = `24`
epochs on `K=6` canonical 4B traces) all-24-layer regularized Pocket MoE recipe across 5 skills (`60` blind hard items,
`24` guardrails, `200` MMLU, `16` held-out prose controls):
  - `singletrace_r8_144p`       : r=8, alpha=16, lr=2e-4 (~10.32 MB/skill; anchor reproducing 91.7%)
  - `singletrace_r4_144p`       : r=4, alpha=8,  lr=2e-4 (~ 5.16 MB/skill; 2x smaller)
  - `singletrace_r2_144p`       : r=2, alpha=4,  lr=2e-4 (~ 2.58 MB/skill; 4x smaller, matched LR)
  - `singletrace_r1_144p`       : r=1, alpha=2,  lr=2e-4 (~ 1.29 MB/skill; 8x smaller, matched LR)
  - `singletrace_r2_144p_lr4e4` : r=2, alpha=4,  lr=4e-4 (~ 2.58 MB/skill; 4x smaller, 2x LR compensation)

Paired with 3 reference arms (`base_08b`, `teacher_4b`, `oracle_1shot_08b`) -> 8 paired arms total.

CPU smoke test::
    python scripts/experiment_pocket_moe_rank_frontier.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_moe_rank_frontier.py --teacher /content/teacher_4b --student /content/student_08b --out /content/pocket_moe_rank_frontier_l4.json
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

import eval_capability_and_boundary_lfm3 as ec  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from experiment_pocket_moe_ceiling import (  # noqa: E402
    GUARDRAIL_FAMILIES,
    _bake_full_checkpoint,
    _grade_rows,
    _hf_generate_subset,
    _vllm_generate,
    fit_ridge_router,
)
from experiment_pocket_moe_controls import _paired_bootstrap  # noqa: E402
from experiment_pocket_moe_fewshot import (  # noqa: E402
    OLD_SKILLS,
    ROUTER_FIT_CLASSES,
    SKILLS_5,
    _predict_fewshot_router,
    _summarize_fewshot_arm,
    build_fewshot_dataset,
)
from experiment_pocket_moe_math500 import (  # noqa: E402
    _compose_routed_mmlu,
    _score_mmlu_fast,
    _summarize_mmlu,
    _temp_apply_weights,
    format_mmlu_prompt_ids,
    load_mmlu_split,
)
from experiment_pocket_moe_regularized import (  # noqa: E402
    _RegLoRALinear,
    _score_routed_mmlu_sparse,
    _train_regularized_lora_expert,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _PlantedReasoningLM,
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
    build_reasoning_dataset,
)
from experiment_reasoning_distill_v2 import (  # noqa: E402
    _format_messages,
    _select_backend,
    format_question,
    grade_response,
)
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402


# ------------------------------------------------------------------ Summary Printer
def _print_rank_frontier_summary(payload: dict[str, Any]) -> None:
    """Print concise human-readable report + single-line compact JSON omitting per-item rows."""
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return

    print(
        "\n=== 1. FRONTERA DE RANGO SINGLE-TRACE 144P (K=%s/habilidad, 24 capas) ==="
        % payload.get("k_shots_per_skill"),
        flush=True,
    )
    print("Tier specs:", json.dumps(payload.get("tier_specs"), separators=(",", ":")), flush=True)
    print("Compile time (s):", payload.get("compile_time_s"), flush=True)
    print("Adapter MB/skill:", payload.get("adapter_mb_per_skill"), flush=True)
    print("Teacher kept frac:", payload.get("teacher_4b_train_kept_frac"), flush=True)
    print("Router report:", payload.get("router_report"), flush=True)

    prose_kl = payload.get("prose_kl", {})
    specs = payload.get("tier_specs", {})
    print("\n=== 2. FRONTERA 8 BRAZOS PAREADOS (HARD_60 + GUARDRAIL_24) ===", flush=True)
    for arm_nm, st in payload.get("generation_arms", {}).items():
        h, g = st["hard_60"], st["guardrail_24"]
        gm = g.get("guardrail_exact_match_vs_base")
        gm_s = f"{gm*100:5.1f}%" if gm is not None else "   n/a"
        pkl = prose_kl.get(arm_nm, 0.0)
        sp = specs.get(arm_nm)
        tag = f"{sp['adapter_mb']:5.2f}MB lr={sp['lr']:.0e}" if sp else "      ref      "
        print(
            f"  {arm_nm:31s} | {tag} | hard60={h['strict_accuracy']*100:5.1f}% ({h['n_correct']:2d}/60) "
            f"[old1-4={h['old_skills_1_to_4_acc']*100:5.1f}%, s5={h['last_skill_5_acc']*100:5.1f}%, "
            f"ren={h['renumbered_30']*100:5.1f}%, rep={h['rephrased_30']*100:5.1f}%] | "
            f"guard={g['strict_accuracy']*100:5.1f}% (ex={gm_s}) | pkl={pkl:.5f}",
            flush=True,
        )
        print(f"     per_skill={h['per_skill']}", flush=True)

    print("\n=== 3. RETENCION MMLU ===", flush=True)
    for arm_nm, vm in payload.get("mmlu_arms", {}).items():
        a_m, nm = vm["all_400"], vm["non_math"]
        d_nm = f"{nm['delta_vs_base']*100:+5.1f} pts" if "delta_vs_base" in nm else "     base"
        print(
            f"  {arm_nm:31s} | MMLU={a_m['accuracy']*100:5.1f}% | non_math={nm['accuracy']*100:5.1f}% ({d_nm}) | "
            f"ABCD={a_m['mean_abcd_prob_mass']*100:5.1f}% | fire={vm['router_fire_rate']*100:5.1f}%",
            flush=True,
        )

    print("\n=== 4. COMPARACIONES PAREADAS (IC 95% BOOTSTRAP) ===", flush=True)
    for grp_lbl, grp_key in [("  [hard_60]", "paired_hard_60"), ("  [old_skills_1_to_4 (48)]", "paired_old_skills_48")]:
        print(grp_lbl, flush=True)
        for k, v in payload.get(grp_key, {}).items():
            print(
                f"    {k:34s}: diff={v['diff']*100:+5.1f} pts "
                f"CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})",
                flush=True,
            )

    print("\nBest routed arm:", payload.get("best_routed_arm_name"), flush=True)
    print("Criteria:", payload.get("criteria"), flush=True)
    print("Timing (s):", payload.get("timing_s"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_generation"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


# ------------------------------------------------------------------ CPU Smoke Test
def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of 8:4:2:1 rank bytes, seed-matched init, 5 toy runs, bit-exact restore, and sparse MMLU."""
    # 1. Exact 8 : 4 : 2 : 1 adapter_bytes and scale == 2.0 across r = 8, 4, 2, 1
    lin = nn.Linear(16, 16, bias=False)
    bytes_by_rank: dict[int, int] = {}
    for rk in (8, 4, 2, 1):
        mod = _RegLoRALinear(lin, rank=rk, alpha_scale=2.0 * rk, dropout=0.1)
        assert abs(mod.scale - 2.0) < 1e-8, (rk, mod.scale)
        bytes_by_rank[rk] = mod.adapter_bytes()
    assert (
        bytes_by_rank[8]
        == 2 * bytes_by_rank[4]
        == 4 * bytes_by_rank[2]
        == 8 * bytes_by_rank[1]
    ), bytes_by_rank

    torch.manual_seed(42)
    w_init_a = _RegLoRALinear(lin, rank=2, alpha_scale=4.0, dropout=0.1)
    torch.manual_seed(42)
    w_init_b = _RegLoRALinear(lin, rank=2, alpha_scale=4.0, dropout=0.1)
    assert torch.equal(w_init_a.lora_A, w_init_b.lora_A)

    # 2. Toy _PlantedReasoningLM: test r=8, r=4, r=2, r=1 (@ lr=2e-2) and r=2 (@ lr=4e-2)
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    seqs = [[1, 2, 4, 10, 20, 20, 20, 20], [2, 1, 4, 11, 20, 20, 20, 20]]
    tr_single: list[dict[str, Any]] = []
    for s in seqs:
        ids_t = torch.tensor(s, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            t_log = model(ids_t).logits[0, 3:7].float()
        vals, tk_ids = t_log.topk(8, dim=-1)
        tr_single.append(
            {
                "ids": s,
                "lo": 3,
                "hi": 7,
                "topk_ids": tk_ids.to(torch.int32).cpu(),
                "topk_logits": vals.to(torch.float16).cpu(),
            }
        )

    h_ids = [torch.tensor([1, 2, 3, 1], dtype=torch.long).unsqueeze(0)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    smoke_runs: dict[str, dict[str, Any]] = {}
    for nm, rk, lr_v in [
        ("r8_lr2e2", 8, 2e-2),
        ("r4_lr2e2", 4, 2e-2),
        ("r2_lr2e2", 2, 2e-2),
        ("r1_lr2e2", 1, 2e-2),
        ("r2_lr4e2", 2, 4e-2),
    ]:
        torch.manual_seed(42)
        res_e = _train_regularized_lora_expert(
            model,
            [0, 1],
            tr_single,
            rank=rk,
            alpha_scale=2.0 * rk,
            total_steps=6,
            grad_accum=3,
            lr_val=lr_v,
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
        assert res_e["total_steps"] == 6 and len(ckpts) == 6 and ckpts[-1] < ckpts[0], (nm, ckpts)
        smoke_runs[nm] = res_e

    mb_r8 = smoke_runs["r8_lr2e2"]["adapter_mb"]
    mb_r4 = smoke_runs["r4_lr2e2"]["adapter_mb"]
    mb_r2 = smoke_runs["r2_lr2e2"]["adapter_mb"]
    mb_r1 = smoke_runs["r1_lr2e2"]["adapter_mb"]
    assert (
        abs(mb_r8 - 2.0 * mb_r4) < 1e-4
        and abs(mb_r4 - 2.0 * mb_r2) < 1e-4
        and abs(mb_r2 - 2.0 * mb_r1) < 1e-4
    ), (mb_r8, mb_r4, mb_r2, mb_r1)

    # 3. Sparse MMLU exact equivalence (empty routing + subset routing)
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
    with _temp_apply_weights(model, smoke_runs["r1_lr2e2"]["weights"]):
        full_edited = _score_mmlu_fast(model, fake_pids, fake_mmlu, letter_ids, dev)
        zero_routed = _score_routed_mmlu_sparse(
            model, fake_pids, fake_mmlu, ["passthrough"] * 4, "rate_work_combined", base_mmlu, letter_ids, dev
        )
        sub_classes = ["passthrough", "rate_work_combined", "passthrough", "rate_work_combined"]
        sub_routed = _score_routed_mmlu_sparse(
            model, fake_pids, fake_mmlu, sub_classes, "rate_work_combined", base_mmlu, letter_ids, dev
        )
    assert zero_routed is base_mmlu
    comp_sparse = _compose_routed_mmlu(fake_mmlu, sub_classes, base_mmlu, {"rate_work_combined": sub_routed}, seed=1)
    comp_full = _compose_routed_mmlu(fake_mmlu, sub_classes, base_mmlu, {"rate_work_combined": full_edited}, seed=1)
    assert comp_sparse == comp_full, (comp_sparse, comp_full)

    return {
        "mode": "smoke",
        "passed": True,
        "adapter_bytes_by_rank": bytes_by_rank,
        "init_lora_a_seed_match": True,
        "toy_runs": {
            nm: {
                "adapter_mb": v["adapter_mb"],
                "n_train_samples": v["n_train_samples"],
                "step_losses_checkpoints": v["step_losses_checkpoints"],
                "mean_anchor_kl": v["mean_anchor_kl"],
            }
            for nm, v in smoke_runs.items()
        },
        "sparse_mmlu_exact_match": True,
    }


# ------------------------------------------------------------------ Full GPU Rank Frontier
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

    ds = build_fewshot_dataset(k_shots=args.k_shots)
    hard_train = ds["hard_train"]
    hard_test = ds["hard_test"]
    guardrail_test = ds["guardrail_test"]
    all_test = hard_test + guardrail_test
    n_hard = len(hard_test)

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

    # ---- Phase 1A: Single 4B Batched Gen (84 Test + 30 Single-Trace Train) ----
    print(
        f"[rankf] Phase 1A: Teacher ({args.teacher}) generation ({backend}) — 84 test + 30 single-trace train...",
        flush=True,
    )
    t_p1 = time.perf_counter()
    test_pids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in all_test
    ]
    train_pids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in hard_train
    ]
    test_seeds = [int(args.seed) + i for i in range(len(all_test))]
    # Identical primary-trace seeds to `m == 0` in experiment_pocket_moe_multitrace_ablation.py
    train_seeds = [int(args.seed) + 1000 + i for i in range(len(hard_train))]

    teacher_hf = None
    if backend == "vllm":
        outs_1a = _vllm_generate(
            args.teacher,
            tok_t,
            test_pids + train_pids,
            test_seeds + train_seeds,
            max_new_tokens=args.max_new_tokens,
            dtype_str=dtype_vllm_str,
            base_seed=args.seed,
        )
    else:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
        outs_1a = _hf_generate_subset(
            teacher_hf,
            tok_t,
            test_pids + train_pids,
            test_seeds + train_seeds,
            max_new_tokens=args.max_new_tokens,
        )

    n_test = len(all_test)
    t_test_raw = outs_1a[:n_test]
    t_train_raw = outs_1a[n_test:]

    teacher_test_rows = _grade_rows(all_test, t_test_raw, tok_s, eos_set, think_end_id)
    teacher_summary = _summarize_fewshot_arm(teacher_test_rows)
    print(
        f"[rankf] teacher_4b: hard={teacher_summary['hard_60']['strict_accuracy']*100:.1f}% | "
        f"guard={teacher_summary['guardrail_24']['strict_accuracy']*100:.1f}%",
        flush=True,
    )

    kept_4b = {sk: 0 for sk in SKILLS_5}
    primary_pairs: dict[str, list[tuple[str, str]]] = {sk: [] for sk in SKILLS_5}
    primary_resp_by_item: list[str] = []

    for i, it in enumerate(hard_train):
        sk = it["family"]
        q_txt = it["question"]
        raw_m = t_train_raw[i]
        body = lr.strip_eos(raw_m["token_ids"], eos_set)
        txt = tok_s.decode(body, skip_special_tokens=True).strip()
        g = grade_response(
            raw_m["token_ids"],
            txt,
            int(it["answer"]),
            thinking_enabled=False,
            finish_reason=raw_m["finish_reason"],
            eos_set=eos_set,
            tok=tok_s,
        )
        is_ok = bool(g["strict_correct"] and g["has_boxed"] and g["total_tokens"] <= int(args.max_teacher_train_tokens))
        if is_ok:
            kept_4b[sk] += 1
            primary_txt = txt
        else:
            primary_txt = it["fallback_response"]

        primary_resp_by_item.append(primary_txt)
        primary_pairs[sk].append((q_txt, primary_txt))

    teacher_kept_frac = {sk: round(kept_4b[sk] / max(1, args.k_shots), 4) for sk in SKILLS_5}
    print(f"[rankf] teacher_4b kept={teacher_kept_frac}", flush=True)

    # ---- Phase 1B: Deduplicated 4B Top-32 Logit Extraction (<=30 single-trace samples) ----
    if teacher_hf is None:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    unique_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for sk in SKILLS_5:
        for pair in primary_pairs[sk]:
            unique_pairs.setdefault(pair, {})

    print(f"[rankf] Phase 1B: Extracting Top-32 4B logits for {len(unique_pairs)} canonical traces...", flush=True)
    with torch.no_grad():
        for (q_txt, resp_txt), entry in unique_pairs.items():
            p_ids = _format_messages(
                tok_s, [{"role": "user", "content": format_question(q_txt)}], add_generation_prompt=True, enable_thinking=False
            )
            full_ids = _format_messages(
                tok_s,
                [{"role": "user", "content": format_question(q_txt)}, {"role": "assistant", "content": resp_txt}],
                add_generation_prompt=False,
                enable_thinking=False,
            )
            lo = max(0, len(p_ids) - 1)
            hi = len(full_ids) - 1
            ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
            t_logits = _tail_logits(teacher_hf, ids_dev, lo)[: hi - lo].float()
            vals, idx = t_logits.topk(32, dim=-1)
            entry.update(
                {
                    "ids": full_ids,
                    "lo": lo,
                    "hi": hi,
                    "topk_ids": idx.to(torch.int32).cpu(),
                    "topk_logits": vals.to(torch.float16).cpu(),
                }
            )
            del ids_dev, t_logits

    del teacher_hf
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    primary_by_skill = {sk: [dict(unique_pairs[p], family=sk) for p in primary_pairs[sk]] for sk in SKILLS_5}
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student 0.8B, Balanced Layer-12 Router, Base MMLU, Held-Out Prose & Disjoint Anchor KL ----
    print(f"[rankf] Phase 2: Loading Student ({args.student}), balanced K=6 Layer-12 router, base MMLU & disjoint anchor...", flush=True)
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
    guard_r_pids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in ds["guardrail_router_train"]
    ]
    mmlu_r_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_router_train]
    mmlu_test_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_test]
    prose_r_pids = [
        _format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
        for p in ds["prose_router_train"]
    ]
    prose_held_pids = [
        _format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
        for p in ds["prose_heldout"]
    ]

    r_train_pids = train_pids + guard_r_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = (
        [it["family"] for it in hard_train]
        + ["pass_math"] * len(guard_r_pids)
        + ["pass_mmlu"] * len(mmlu_r_pids)
        + ["pass_prose"] * len(prose_r_pids)
    )
    W_r, b_r = fit_ridge_router(
        _extract_prompt_hidden_vecs(student_m, r_train_pids, router_layer), r_train_labels, ROUTER_FIT_CLASSES, lam=1.0
    )
    routed_test_classes = _predict_fewshot_router(_extract_prompt_hidden_vecs(student_m, test_pids, router_layer), W_r, b_r)
    routed_mmlu_classes = _predict_fewshot_router(_extract_prompt_hidden_vecs(student_m, mmlu_test_pids, router_layer), W_r, b_r)
    routed_prose_classes = _predict_fewshot_router(_extract_prompt_hidden_vecs(student_m, prose_held_pids, router_layer), W_r, b_r)

    hard_r_preds = routed_test_classes[:n_hard]
    guard_r_preds = routed_test_classes[n_hard:]
    router_report = {
        "router_layer": router_layer,
        "k_shots_per_skill": int(args.k_shots),
        "fit_classes": list(ROUTER_FIT_CLASSES),
        "hard_acc": round(sum(p == it["family"] for p, it in zip(hard_r_preds, hard_test)) / len(hard_test), 4),
        "hard_renumbered_acc": round(
            sum(p == it["family"] for p, it in zip(hard_r_preds, hard_test) if it["variant"] == "renumbered")
            / max(1, sum(1 for it in hard_test if it["variant"] == "renumbered")),
            4,
        ),
        "hard_rephrased_acc": round(
            sum(p == it["family"] for p, it in zip(hard_r_preds, hard_test) if it["variant"] == "rephrased_renumbered")
            / max(1, sum(1 for it in hard_test if it["variant"] == "rephrased_renumbered")),
            4,
        ),
        "guardrail_passthrough_acc": round(sum(p == "passthrough" for p in guard_r_preds) / len(guard_r_preds), 4),
        "guardrail_per_family_passthrough": {
            gf: round(
                sum(1 for p, it in zip(guard_r_preds, guardrail_test) if it["family"] == gf and p == "passthrough")
                / max(1, sum(1 for it in guardrail_test if it["family"] == gf)),
                4,
            )
            for gf in GUARDRAIL_FAMILIES
        },
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_classes) / len(mmlu_test), 4),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_classes) / len(routed_prose_classes), 4),
    }
    print(f"[rankf] Router report: {router_report}", flush=True)

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

    prose_full_ids, base_prose_logp, base_prose_nll = _compute_base_logp(ds["prose_heldout"])
    anchor_full_ids, base_anchor_logp, _ = _compute_base_logp(ds["prose_router_train"])
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Train 5 Single-Trace 144p Rank-Frontier Tiers (x5 skills) ----
    print("[rankf] Phase 3: Training 5 single-trace 144p rank-frontier tiers across all 24 layers...", flush=True)
    t_p3 = time.perf_counter()

    # (tier_nm, dataset_tag, ds_map, rank, alpha_scale, total_steps, grad_accum, lr, wd, dropout, warmup, T, kl_w, anchor_w)
    tier_defs = [
        ("singletrace_r8_144p", "primary_1trace", primary_by_skill, 8, 16.0, 48, 3, 2.0e-4, 0.05, 0.10, 0.15, 2.0, 0.85, 0.05),
        ("singletrace_r4_144p", "primary_1trace", primary_by_skill, 4, 8.0, 48, 3, 2.0e-4, 0.05, 0.10, 0.15, 2.0, 0.85, 0.05),
        ("singletrace_r2_144p", "primary_1trace", primary_by_skill, 2, 4.0, 48, 3, 2.0e-4, 0.05, 0.10, 0.15, 2.0, 0.85, 0.05),
        ("singletrace_r1_144p", "primary_1trace", primary_by_skill, 1, 2.0, 48, 3, 2.0e-4, 0.05, 0.10, 0.15, 2.0, 0.85, 0.05),
        ("singletrace_r2_144p_lr4e4", "primary_1trace", primary_by_skill, 2, 4.0, 48, 3, 4.0e-4, 0.05, 0.10, 0.15, 2.0, 0.85, 0.05),
    ]

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        expert_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {t[0]: {} for t in tier_defs}
        domain_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {t[0]: {} for t in tier_defs}

        for s_idx, sk in enumerate(SKILLS_5):
            for (tier_nm, _ds_tag, ds_map, rk, a_sc, n_stp, g_acc, lr_v, wd_v, drp_v, wrm_v, t_dist, kl_w, anc_w) in tier_defs:
                # Matched seed per skill so same-rank tiers share exact lora_A init & dropout/batch sequence
                skill_seed = int(args.seed) + s_idx
                torch.manual_seed(skill_seed)
                res_e = _train_regularized_lora_expert(
                    student_m,
                    all_layers,
                    ds_map[sk],
                    rank=rk,
                    alpha_scale=a_sc,
                    total_steps=n_stp,
                    grad_accum=g_acc,
                    lr_val=lr_v,
                    weight_decay=wd_v,
                    lora_dropout=drp_v,
                    warmup_frac=wrm_v,
                    distill_T=t_dist,
                    kl_weight=kl_w,
                    anchor_ids=anchor_full_ids,
                    base_anchor_logp=base_anchor_logp,
                    anchor_kl_weight=anc_w,
                    healthy_ids=prose_full_ids,
                    base_healthy_logp=base_prose_logp,
                    base_healthy_nll=base_prose_nll,
                    seed=skill_seed,
                )
                upd_w = res_e.pop("weights")
                with _temp_apply_weights(student_m, upd_w):
                    domain_mmlu_scores[tier_nm][sk] = _score_routed_mmlu_sparse(
                        student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                    )
                p_exp = exp_dir / f"{tier_nm}_{sk}.pt"
                torch.save(upd_w, p_exp)
                del upd_w
                expert_paths[(tier_nm, sk)] = p_exp
                expert_meta[tier_nm][sk] = res_e
                print(
                    f"[rankf] {tier_nm}/{sk}: n={res_e['n_train_samples']} passes={n_stp * g_acc} r={rk} lr={lr_v:.0e} "
                    f"({res_e['adapter_mb']}MB, {res_e['compile_time_s']:.1f}s) | "
                    f"loss={res_e['step_losses_checkpoints'][0]}->{res_e['step_losses_checkpoints'][-1]} | pkl={res_e['healthy_kl']}",
                    flush=True,
                )

        mmlu_arms: dict[str, Any] = {"base_08b": base_mmlu_summary}
        for idx_t, (tier_nm, *_rest) in enumerate(tier_defs):
            mmlu_arms[f"{tier_nm}_routed"] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_scores[tier_nm], seed=int(args.seed) + 1 + idx_t
            )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Evaluate All 8 Paired Arms on 84 Test Prompts ----
        print(f"[rankf] Phase 4: Evaluating 8 paired arms ({backend}, max_new_tokens={args.max_new_tokens})...", flush=True)
        t_p4 = time.perf_counter()

        train_ex_by_fam: dict[str, tuple[str, str]] = {}
        for sk in SKILLS_5:
            idx0 = next(i for i, it in enumerate(hard_train) if it["family"] == sk)
            train_ex_by_fam[sk] = (hard_train[idx0]["question"], primary_resp_by_item[idx0])
        base_ds_ref = build_reasoning_dataset()
        for gf in GUARDRAIL_FAMILIES:
            ex_g = next(it for it in base_ds_ref["train"] if it["family"] == gf)
            clean_resp = ex_g["teacher_response"].split("</think>")[-1].strip()
            train_ex_by_fam[gf] = (ex_g["question"], clean_resp or f"The answer is \\boxed{{{ex_g['answer']}}}.")
        oracle_pids = [
            _format_messages(
                tok_s,
                [
                    {"role": "user", "content": format_question(train_ex_by_fam[it["family"]][0])},
                    {"role": "assistant", "content": train_ex_by_fam[it["family"]][1]},
                    {"role": "user", "content": format_question(it["question"])},
                ],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for it in all_test
        ]

        if backend == "vllm":
            del student_m
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            bo_outs = _vllm_generate(
                args.student,
                tok_s,
                test_pids + oracle_pids,
                test_seeds + test_seeds,
                max_new_tokens=args.max_new_tokens,
                dtype_str=dtype_vllm_str,
                base_seed=args.seed,
            )
            base_raw = bo_outs[: len(all_test)]
            oracle_raw = bo_outs[len(all_test) :]
        else:
            base_raw = _hf_generate_subset(student_m, tok_s, test_pids, test_seeds, max_new_tokens=args.max_new_tokens)
            oracle_raw = _hf_generate_subset(student_m, tok_s, oracle_pids, test_seeds, max_new_tokens=args.max_new_tokens)

        bake_tmp_dir = Path("/content/_pocket_moe_rf_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

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
                    max_new_tokens=args.max_new_tokens,
                    dtype_str=dtype_vllm_str,
                    base_seed=args.seed,
                )
            return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, param_updates=upd)

        routed_raw_by_tier: dict[str, list[dict[str, Any]]] = {}
        try:
            for tier_nm, *_rest in tier_defs:
                tier_raw: list[dict[str, Any] | None] = [None] * len(all_test)
                for i, cls_nm in enumerate(routed_test_classes):
                    if cls_nm == "passthrough":
                        tier_raw[i] = base_raw[i]
                for sk in SKILLS_5:
                    idxs = [i for i, cls_nm in enumerate(routed_test_classes) if cls_nm == sk]
                    p_exp = expert_paths[(tier_nm, sk)]
                    if idxs:
                        sub_outs = _eval_ckpt(p_exp, [test_pids[i] for i in idxs], [test_seeds[i] for i in idxs])
                        for idx_i, o in zip(idxs, sub_outs):
                            tier_raw[idx_i] = o
                    p_exp.unlink(missing_ok=True)
                routed_raw_by_tier[tier_nm] = [r for r in tier_raw if r is not None]
        finally:
            if bake_tmp_dir.exists():
                shutil.rmtree(bake_tmp_dir, ignore_errors=True)
        phase4_s = round(time.perf_counter() - t_p4, 2)

    # ---- Phase 5: Grading, Paired Bootstrap CIs & Summary ----
    arm_raw_map: dict[str, tuple[list[dict[str, Any]], list[str] | None, bool]] = {
        "base_08b": (base_raw, None, False),
        "teacher_4b": (t_test_raw, None, True),
        "oracle_1shot_08b": (oracle_raw, None, False),
    }
    for tier_nm, *_rest in tier_defs:
        arm_raw_map[f"{tier_nm}_routed"] = (routed_raw_by_tier[tier_nm], routed_test_classes, False)

    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_nm, (raw_l, r_lbls, is_teacher) in arm_raw_map.items():
        if is_teacher:
            rows = teacher_test_rows
            sm = teacher_summary
        else:
            rows = _grade_rows(all_test, raw_l, tok_s, eos_set, think_end_id, routed_labels=r_lbls)
            sm = _summarize_fewshot_arm(rows, arm_raw=raw_l, base_raw=base_raw)
        arm_rows[arm_nm] = rows
        arm_summaries[arm_nm] = sm

    def _routed_prose_kl(tier_key: str) -> float:
        tot = sum(float(expert_meta[tier_key][p_cls]["healthy_kl"]) for p_cls in routed_prose_classes if p_cls in SKILLS_5)
        return round(tot / max(1, len(routed_prose_classes)), 6)

    prose_kl: dict[str, float] = {f"{tier_nm}_routed": _routed_prose_kl(tier_nm) for tier_nm, *_rest in tier_defs}

    compile_time_s: dict[str, float] = {
        f"{tier_nm}_mean_per_skill_s": round(sum(expert_meta[tier_nm][s]["compile_time_s"] for s in SKILLS_5) / len(SKILLS_5), 2)
        for tier_nm, *_rest in tier_defs
    }
    adapter_mb_per_skill: dict[str, float] = {
        f"{tier_nm}_routed": round(sum(expert_meta[tier_nm][s]["adapter_mb"] for s in SKILLS_5) / len(SKILLS_5), 4)
        for tier_nm, *_rest in tier_defs
    }
    tier_specs: dict[str, dict[str, Any]] = {
        f"{tier_nm}_routed": {
            "dataset": ds_tag,
            "rank": int(rk),
            "alpha_scale": float(a_sc),
            "lr": float(lr_v),
            "total_steps": int(n_stp),
            "grad_accum": int(g_acc),
            "sample_passes": int(n_stp * g_acc),
            "mean_n_train_samples": round(sum(expert_meta[tier_nm][s]["n_train_samples"] for s in SKILLS_5) / len(SKILLS_5), 2),
            "adapter_mb": adapter_mb_per_skill[f"{tier_nm}_routed"],
            "int8_est_mb": round(adapter_mb_per_skill[f"{tier_nm}_routed"] / 2.0, 4),
        }
        for tier_nm, ds_tag, _ds_map, rk, a_sc, n_stp, g_acc, lr_v, *_rest in tier_defs
    }

    routed_arm_names = [f"{t[0]}_routed" for t in tier_defs]
    best_routed_arm = max(
        routed_arm_names,
        key=lambda k: (
            arm_summaries[k]["hard_60"]["strict_accuracy"],
            arm_summaries[k]["guardrail_24"]["strict_accuracy"],
            -adapter_mb_per_skill[k],
        ),
    )

    comp_pairs = [
        ("r4_vs_r8", "singletrace_r4_144p_routed", "singletrace_r8_144p_routed"),
        ("r2_vs_r8", "singletrace_r2_144p_routed", "singletrace_r8_144p_routed"),
        ("r1_vs_r8", "singletrace_r1_144p_routed", "singletrace_r8_144p_routed"),
        ("r2_lr4e4_vs_r8", "singletrace_r2_144p_lr4e4_routed", "singletrace_r8_144p_routed"),
        ("r2_vs_r4", "singletrace_r2_144p_routed", "singletrace_r4_144p_routed"),
        ("r1_vs_r2", "singletrace_r1_144p_routed", "singletrace_r2_144p_routed"),
        ("r2_lr4e4_vs_r2", "singletrace_r2_144p_lr4e4_routed", "singletrace_r2_144p_routed"),
        ("r8_vs_oracle_1shot", "singletrace_r8_144p_routed", "oracle_1shot_08b"),
        ("r4_vs_oracle_1shot", "singletrace_r4_144p_routed", "oracle_1shot_08b"),
        ("r2_vs_oracle_1shot", "singletrace_r2_144p_routed", "oracle_1shot_08b"),
        ("r2_lr4e4_vs_oracle_1shot", "singletrace_r2_144p_lr4e4_routed", "oracle_1shot_08b"),
        ("r1_vs_oracle_1shot", "singletrace_r1_144p_routed", "oracle_1shot_08b"),
        ("best_routed_vs_base", best_routed_arm, "base_08b"),
        ("best_routed_vs_oracle_1shot", best_routed_arm, "oracle_1shot_08b"),
    ]
    paired_hard_60 = {
        lbl: _paired_bootstrap(arm_rows[a][:n_hard], arm_rows[b][:n_hard], seed=int(args.seed) + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }
    paired_old_skills_48 = {
        lbl: _paired_bootstrap(
            [r for r in arm_rows[a][:n_hard] if r["family"] in OLD_SKILLS],
            [r for r in arm_rows[b][:n_hard] if r["family"] in OLD_SKILLS],
            seed=int(args.seed) + 100 + i,
        )
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    h_acc = lambda k: float(arm_summaries[k]["hard_60"]["strict_accuracy"])  # noqa: E731
    nm_mmlu = lambda k: float(mmlu_arms[k]["non_math"]["accuracy"])  # noqa: E731

    return {
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "backend": backend,
        "k_shots_per_skill": int(args.k_shots),
        "max_new_tokens": int(args.max_new_tokens),
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_router_base_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "tier_specs": tier_specs,
        "compile_time_s": compile_time_s,
        "adapter_mb_per_skill": adapter_mb_per_skill,
        "teacher_4b_train_kept_frac": teacher_kept_frac,
        "router_report": router_report,
        "expert_meta": expert_meta,
        "prose_kl": prose_kl,
        "generation_arms": arm_summaries,
        "mmlu_arms": mmlu_arms,
        "paired_hard_60": paired_hard_60,
        "paired_old_skills_48": paired_old_skills_48,
        "best_routed_arm_name": best_routed_arm,
        "criteria": {
            "a_r8_anchor_beats_oracle_1shot": bool(
                h_acc("singletrace_r8_144p_routed") >= h_acc("oracle_1shot_08b")
            ),
            "b_r4_5mb_preserves_r8_within_3_5pts": bool(
                h_acc("singletrace_r4_144p_routed") >= h_acc("singletrace_r8_144p_routed") - 0.035
            ),
            "c_r2_2_6mb_preserves_r8_within_5pts": bool(
                max(h_acc("singletrace_r2_144p_routed"), h_acc("singletrace_r2_144p_lr4e4_routed"))
                >= h_acc("singletrace_r8_144p_routed") - 0.05
            ),
            "d_r2_2_6mb_beats_or_ties_oracle_1shot": bool(
                max(h_acc("singletrace_r2_144p_routed"), h_acc("singletrace_r2_144p_lr4e4_routed"))
                >= h_acc("oracle_1shot_08b")
            ),
            "e_r1_1_3mb_above_80pct": bool(h_acc("singletrace_r1_144p_routed") >= 0.80),
            "f_zero_mmlu_and_prose_drift_all_routed": bool(
                all(
                    prose_kl[nm] <= 0.005
                    and nm_mmlu(nm) >= nm_mmlu("base_08b") - 0.015
                    and mmlu_arms[nm]["router_fire_rate"] == 0.0
                    for nm in routed_arm_names
                )
            ),
        },
        "per_item_generation": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU verification")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--k-shots", type=int, default=6)
    ap.add_argument("--n-mmlu-test", type=int, default=200)
    ap.add_argument("--n-mmlu-router-train", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1350)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
        _print_rank_frontier_summary(payload)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
