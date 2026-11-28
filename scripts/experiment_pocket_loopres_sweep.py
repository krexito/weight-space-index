"""Pocket LoopRes T-Sweep & Variable-T Training Benchmark (Qwen3-4B -> Qwen3-0.6B).

Tests two targeted follow-ups to the Experiment B finding (`loopres_r2_3x` reaching 80.0% vs 65.0% at 2x):
  1. Full test-time T-sweep (`T in {2, 3, 4, 5}`) for fixed-2x trained `loopres_r2_fix2x`.
  2. Balanced variable-T training (`T in {2, 3}` alternating per step) + T-sweep (`T in {2, 3, 4, 5}`)
     for `loopres_r2_varT23`, compared paired against `base`, `teacher`, and `lora_r2_1x_routed`.

CPU smoke test::
    python scripts/experiment_pocket_loopres_sweep.py --smoke
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
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
from experiment_pocket_loopres import (  # noqa: E402
    AutoModelForCausalLM_load,
    _SmokeTinyTokenizer,
    _aggregate_loopres_telemetry,
    _hf_generate_batched,
    _load_or_collect_teacher,
    _load_smoke_toy_lm_cls,
    _score_routed_mmlu_sparse_loopres,
    _train_loopres_expert,
)
from experiment_pocket_moe_ceiling import _grade_rows, fit_ridge_router  # noqa: E402
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
    format_mmlu_prompt_ids,
    load_mmlu_split,
)
from experiment_reasoning_distill_pilot import _atomic_write_json, _extract_prompt_hidden_vecs  # noqa: E402
from experiment_reasoning_distill_v2 import _format_messages, _select_backend, format_question  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.loopres import LoopResConfig, LoopResModule, default_core_span  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402


def _parse_int_tuple(spec: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"Expected comma-separated positive ints, got {spec!r}")
    return vals


def run_smoke() -> dict[str, Any]:
    """Verify balanced variable-T training (`T in {2, 3}`) and `T=4` zero-shot sweep on CPU."""
    _ToyCachedTransformerLM = _load_smoke_toy_lm_cls()

    torch.manual_seed(31)
    model = _ToyCachedTransformerLM(vocab=40, hidden=24, n_layers=6, n_heads=4, intermediate=48, seed=31).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _, blocks = find_blocks(model)
    core_s, core_e = default_core_span(len(blocks))
    all_layers = list(range(len(blocks)))

    seqs = [
        [1, 3, 5, 10, 15, 20, 25, 2],
        [4, 6, 8, 12, 18, 24, 30, 2],
    ]
    tr_samples: list[dict[str, Any]] = []
    for s in seqs:
        ids_t = torch.tensor(s, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            t_log = model(ids_t).logits[0, 3:7].float()
        vals, tk_ids = t_log.topk(8, dim=-1)
        tr_samples.append({
            "ids": s,
            "lo": 3,
            "hi": 7,
            "topk_ids": tk_ids.to(torch.int32).cpu(),
            "topk_logits": vals.to(torch.float16).cpu(),
        })

    h_ids = [torch.tensor([1, 2, 3, 4, 2], dtype=torch.long).unsqueeze(0)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    torch.manual_seed(42)
    res_var = _train_loopres_expert(
        model,
        all_layers,
        tr_samples,
        rank=2,
        alpha_scale=4.0,
        loopres_mode="loopres",
        core_start=core_s,
        core_end=core_e,
        n_loops=3,
        n_loops_choices=(2, 3),
        window=2,
        d_loop_key=16,
        total_steps=6,
        grad_accum=2,
        lr_lora=2e-2,
        lr_loopres=5e-2,
        anchor_ids=h_ids,
        base_anchor_logp=base_lp,
        anchor_kl_weight=0.05,
        healthy_ids=h_ids,
        base_healthy_logp=base_lp,
        base_healthy_nll=base_nll,
        grad_checkpointing=True,
        seed=42,
    )
    assert res_var["loop_counts_seen"] == {"2": 3, "3": 3}, res_var["loop_counts_seen"]
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())
    ckpts = res_var["step_losses_checkpoints"]
    assert ckpts[-1] < ckpts[0], ckpts

    tok_sm = _SmokeTinyTokenizer()
    prompts = [[1, 3, 5], [4, 6, 8, 12]]
    mod_lr = LoopResModule(LoopResConfig.from_dict(res_var["loopres_config"]))
    mod_lr.load_state_dict(res_var["loopres_state"])

    tels: dict[str, Any] = {}
    for t_eval in (2, 3, 4, 5):
        outs = _hf_generate_batched(
            model,
            tok_sm,
            prompts,
            max_new_tokens=4,
            batch_size=2,
            base_seed=42,
            loopres_mod=mod_lr,
            n_loops=t_eval,
            param_updates=res_var["weights"],
            record_stats=True,
        )
        assert len(outs) == 2
        tel = mod_lr.telemetry_summary()
        assert len(tel["coda_alpha_mean"]) == min(t_eval, 3), (t_eval, tel)
        assert abs(sum(tel["coda_alpha_mean"]) - 1.0) < 1e-5
        tels[f"T{t_eval}"] = tel

    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())
    return {
        "mode": "smoke",
        "passed": True,
        "loop_counts_seen": res_var["loop_counts_seen"],
        "step_losses_checkpoints": ckpts,
        "telemetry": tels,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    eval_ts = _parse_int_tuple(args.eval_ts)
    var_choices = _parse_int_tuple(args.var_choices)
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

    test_pids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in all_test
    ]
    train_pids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in hard_train
    ]
    test_seeds = [int(args.seed) + i for i in range(len(all_test))]

    # ---- Phase 1: Teacher Traces & Top-32 Logits ----
    print(f"[sweep] Phase 1: Teacher ({args.teacher}) traces & Top-32 logits (bs={args.teacher_batch_size})...", flush=True)
    t_p1 = time.perf_counter()
    t_data = _load_or_collect_teacher(
        args, ds, all_test, test_pids, train_pids, test_seeds,
        tok_t, tok_s, eos_set, think_end_id, backend, dev, dtype, dtype_vllm_str,
    )
    t_test_raw = t_data["t_test_raw"]
    teacher_kept_frac = t_data["teacher_kept_frac"]
    primary_by_skill = t_data["primary_by_skill"]
    teacher_test_rows = _grade_rows(all_test, t_test_raw, tok_s, eos_set, think_end_id)
    teacher_summary = _summarize_fewshot_arm(teacher_test_rows)
    phase1_s = round(time.perf_counter() - t_p1, 2)
    print(f"[sweep] teacher hard60={teacher_summary['hard_60']['strict_accuracy']*100:.1f}% ({phase1_s}s)", flush=True)

    # ---- Phase 2: Student, Router, Base MMLU & Prose Anchor ----
    print(f"[sweep] Phase 2: Loading Student ({args.student}), router & base MMLU...", flush=True)
    t_p2 = time.perf_counter()
    try:
        student_m = AutoModelForCausalLM_load(args.student, dtype).to(dev).eval()
    except TypeError:
        student_m = AutoModelForCausalLM_load(args.student, dtype, legacy=True).to(dev).eval()
    for p in student_m.parameters():
        p.requires_grad_(False)

    _, blocks = find_blocks(student_m)
    n_layers = len(blocks)
    all_layers = list(range(n_layers))
    def_cs, def_ce = default_core_span(n_layers)
    core_start = def_cs if args.core_start is None else int(args.core_start)
    core_end = def_ce if args.core_end is None else int(args.core_end)
    router_layer = n_layers // 2
    core_config = {
        "n_layers": n_layers,
        "core_start": core_start,
        "core_end": core_end,
        "eval_ts": list(eval_ts),
        "var_choices": list(var_choices),
        "window": int(args.window),
        "d_loop_key": int(args.d_loop_key),
        "rank": int(args.rank),
        "router_layer": router_layer,
    }

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
        "hard_acc": round(sum(p == it["family"] for p, it in zip(hard_r_preds, hard_test)) / len(hard_test), 4),
        "guardrail_passthrough_acc": round(sum(p == "passthrough" for p in guard_r_preds) / len(guard_r_preds), 4),
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_classes) / len(mmlu_test), 4),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_classes) / len(routed_prose_classes), 4),
    }
    print(f"[sweep] Router: {router_report}", flush=True)

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

    # ---- Phase 3: Train 3 Tiers x 5 Skills ----
    print("[sweep] Phase 3: Training lora_r2_1x, loopres_r2_fix2x, and loopres_r2_varT...", flush=True)
    t_p3 = time.perf_counter()

    # (tier_nm, rank, alpha_scale, loopres_mode, n_loops_choices)
    train_tiers = [
        ("lora_r2_1x", int(args.rank), 2.0 * float(args.rank), None, (1,)),
        ("loopres_r2_fix2x", int(args.rank), 2.0 * float(args.rank), "loopres", (2,)),
        ("loopres_r2_varT", int(args.rank), 2.0 * float(args.rank), "loopres", var_choices),
    ]

    ckpt_root = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    dir_ctx = nullcontext(str(ckpt_root)) if ckpt_root is not None else tempfile.TemporaryDirectory()

    with dir_ctx as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        artifact_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {t[0]: {} for t in train_tiers}
        domain_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {}

        eval_specs: list[tuple[str, str, int]] = [("lora_r2_1x_routed", "lora_r2_1x", 1)]
        for t_ev in eval_ts:
            eval_specs.append((f"loopres_r2_fix2x_T{t_ev}_routed", "loopres_r2_fix2x", t_ev))
        for t_ev in eval_ts:
            eval_specs.append((f"loopres_r2_varT_T{t_ev}_routed", "loopres_r2_varT", t_ev))
        for arm_nm, *_rest in eval_specs:
            domain_mmlu_scores[arm_nm] = {}

        for s_idx, sk in enumerate(SKILLS_5):
            for (tier_nm, rk, a_sc, lr_mode, lp_choices) in train_tiers:
                p_art = exp_dir / f"{tier_nm}_{sk}.pt"
                p_meta = exp_dir / f"{tier_nm}_{sk}.meta.json"
                if ckpt_root is not None and p_art.exists() and p_meta.exists():
                    res_e = json.loads(p_meta.read_text(encoding="utf-8"))
                    art = torch.load(p_art, map_location="cpu", weights_only=True)
                    upd_w = art["weights"]
                    lr_sd = art["loopres_state"]
                    lr_cfg = art["loopres_config"]
                    print(f"[sweep] Resumed {tier_nm}/{sk} from checkpoint", flush=True)
                else:
                    skill_seed = int(args.seed) + s_idx
                    torch.manual_seed(skill_seed)
                    res_e = _train_loopres_expert(
                        student_m,
                        all_layers,
                        primary_by_skill[sk],
                        rank=rk,
                        alpha_scale=a_sc,
                        loopres_mode=lr_mode,
                        core_start=core_start,
                        core_end=core_end,
                        n_loops=max(lp_choices),
                        n_loops_choices=lp_choices,
                        window=args.window,
                        d_loop_key=args.d_loop_key,
                        total_steps=args.total_steps,
                        grad_accum=args.grad_accum,
                        lr_lora=args.lr_lora,
                        lr_loopres=args.lr_loopres,
                        anchor_ids=anchor_full_ids,
                        base_anchor_logp=base_anchor_logp,
                        healthy_ids=prose_full_ids,
                        base_healthy_logp=base_prose_logp,
                        base_healthy_nll=base_prose_nll,
                        grad_checkpointing=not args.no_grad_checkpointing,
                        seed=skill_seed,
                    )
                    upd_w = res_e.pop("weights")
                    lr_sd = res_e.pop("loopres_state")
                    lr_cfg = res_e.pop("loopres_config")

                    p_art_tmp = p_art.with_suffix(".pt.tmp")
                    torch.save({"weights": upd_w, "loopres_state": lr_sd, "loopres_config": lr_cfg}, p_art_tmp)
                    p_art_tmp.replace(p_art)
                    if ckpt_root is not None:
                        _atomic_write_json(p_meta, res_e)

                mod_lr = None
                if lr_sd is not None and lr_cfg is not None:
                    mod_lr = LoopResModule(LoopResConfig.from_dict(lr_cfg)).to(dev).eval()
                    mod_lr.load_state_dict(lr_sd)

                for arm_nm, a_tier, a_loops in eval_specs:
                    if a_tier != tier_nm:
                        continue
                    domain_mmlu_scores[arm_nm][sk] = _score_routed_mmlu_sparse_loopres(
                        student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev,
                        param_updates=upd_w or None, loopres_mod=mod_lr, n_loops=a_loops,
                    )
                del mod_lr, upd_w, lr_sd
                artifact_paths[(tier_nm, sk)] = p_art
                expert_meta[tier_nm][sk] = res_e
                print(
                    f"[sweep] {tier_nm}/{sk}: ({res_e['compile_time_s']:.1f}s) | "
                    f"loss={res_e['step_losses_checkpoints'][0]}->{res_e['step_losses_checkpoints'][-1]} | "
                    f"loops={res_e.get('loop_counts_seen', {})} | u_kl={res_e['healthy_kl']}",
                    flush=True,
                )

        mmlu_arms: dict[str, Any] = {"base": base_mmlu_summary}
        for idx_a, (arm_nm, *_rest) in enumerate(eval_specs):
            mmlu_arms[arm_nm] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_scores[arm_nm], seed=int(args.seed) + 1 + idx_a
            )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Unified Batched HF Evaluation ----
        print(f"[sweep] Phase 4: Evaluating {len(eval_specs) + 1} student arms (bs={args.batch_size})...", flush=True)
        t_p4 = time.perf_counter()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        p_base_eval = exp_dir / "eval_base.json"
        if ckpt_root is not None and p_base_eval.exists():
            cached_base = json.loads(p_base_eval.read_text(encoding="utf-8")).get("raw", [])
            if len(cached_base) == len(all_test):
                base_raw = cached_base
                print("[sweep] Resumed eval_base from checkpoint", flush=True)
            else:
                base_raw = _hf_generate_batched(
                    student_m, tok_s, test_pids,
                    max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, base_seed=args.seed,
                )
                _atomic_write_json(p_base_eval, {"raw": base_raw})
        else:
            base_raw = _hf_generate_batched(
                student_m, tok_s, test_pids,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, base_seed=args.seed,
            )
            if ckpt_root is not None:
                _atomic_write_json(p_base_eval, {"raw": base_raw})

        routed_raw_by_arm: dict[str, list[dict[str, Any]]] = {}
        loopres_telemetry: dict[str, Any] = {}
        for arm_nm, tier_key, active_loops in eval_specs:
            p_arm_eval = exp_dir / f"eval_{arm_nm}.json"
            if ckpt_root is not None and p_arm_eval.exists():
                cached_arm = json.loads(p_arm_eval.read_text(encoding="utf-8"))
                if len(cached_arm.get("raw", [])) == len(all_test):
                    routed_raw_by_arm[arm_nm] = cached_arm["raw"]
                    if cached_arm.get("telemetry"):
                        loopres_telemetry[arm_nm] = cached_arm["telemetry"]
                    print(f"[sweep] Resumed {arm_nm} from checkpoint", flush=True)
                    continue

            arm_raw: list[dict[str, Any] | None] = [None] * len(all_test)
            for i, cls_nm in enumerate(routed_test_classes):
                if cls_nm == "passthrough":
                    arm_raw[i] = base_raw[i]

            skill_tels: dict[str, dict[str, Any]] = {}
            for sk in SKILLS_5:
                idxs = [i for i, cls_nm in enumerate(routed_test_classes) if cls_nm == sk]
                if not idxs:
                    continue
                art = torch.load(artifact_paths[(tier_key, sk)], map_location="cpu", weights_only=True)
                upd_w = art["weights"] or None
                mod_lr = None
                if art["loopres_state"] is not None and art["loopres_config"] is not None:
                    mod_lr = LoopResModule(LoopResConfig.from_dict(art["loopres_config"])).to(dev).eval()
                    mod_lr.load_state_dict(art["loopres_state"])

                sub_outs = _hf_generate_batched(
                    student_m,
                    tok_s,
                    [test_pids[i] for i in idxs],
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    base_seed=int(args.seed) + 500,
                    loopres_mod=mod_lr,
                    n_loops=active_loops,
                    param_updates=upd_w,
                    record_stats=(mod_lr is not None),
                )
                for idx_i, o in zip(idxs, sub_outs):
                    arm_raw[idx_i] = o
                if mod_lr is not None:
                    skill_tels[sk] = mod_lr.telemetry_summary()
                del art, upd_w, mod_lr

            routed_raw_by_arm[arm_nm] = [r for r in arm_raw if r is not None]
            if skill_tels:
                loopres_telemetry[arm_nm] = _aggregate_loopres_telemetry(skill_tels)
            if ckpt_root is not None:
                _atomic_write_json(p_arm_eval, {"raw": routed_raw_by_arm[arm_nm], "telemetry": loopres_telemetry.get(arm_nm)})
            print(f"[sweep] Evaluated {arm_nm}", flush=True)

        phase4_s = round(time.perf_counter() - t_p4, 2)

    # ---- Phase 5: Grading & Paired Bootstrap ----
    arm_raw_map: dict[str, tuple[list[dict[str, Any]], list[str] | None, bool]] = {
        "base": (base_raw, None, False),
        "teacher": (t_test_raw, None, True),
    }
    for arm_nm, *_rest in eval_specs:
        arm_raw_map[arm_nm] = (routed_raw_by_arm[arm_nm], routed_test_classes, False)

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

    routed_prose_kl: dict[str, float] = {}
    unrouted_prose_kl: dict[str, float] = {}
    payload_per_skill: dict[str, dict[str, float]] = {}
    for arm_nm, tier_key, _lp in eval_specs:
        tot_r = sum(float(expert_meta[tier_key][p_cls]["healthy_kl"]) for p_cls in routed_prose_classes if p_cls in SKILLS_5)
        routed_prose_kl[arm_nm] = round(tot_r / max(1, len(routed_prose_classes)), 6)
        unrouted_prose_kl[arm_nm] = round(sum(float(expert_meta[tier_key][s]["healthy_kl"]) for s in SKILLS_5) / len(SKILLS_5), 6)
        payload_per_skill[arm_nm] = {
            "lora_mb": round(sum(float(expert_meta[tier_key][s]["lora_mb"]) for s in SKILLS_5) / len(SKILLS_5), 4),
            "loopres_kb": round(sum(float(expert_meta[tier_key][s]["loopres_kb"]) for s in SKILLS_5) / len(SKILLS_5), 2),
            "total_mb": round(sum(float(expert_meta[tier_key][s]["adapter_mb"]) for s in SKILLS_5) / len(SKILLS_5), 4),
        }

    routed_arm_names = [a[0] for a in eval_specs]
    best_routed_arm = max(
        routed_arm_names,
        key=lambda k: (
            arm_summaries[k]["hard_60"]["strict_accuracy"],
            arm_summaries[k]["guardrail_24"]["strict_accuracy"],
            -payload_per_skill[k]["total_mb"],
        ),
    )

    comp_pairs: list[tuple[str, str, str]] = [
        (f"{best_routed_arm}_vs_base", best_routed_arm, "base"),
        (f"{best_routed_arm}_vs_lora_r2_1x", best_routed_arm, "lora_r2_1x_routed"),
    ]
    for t_ev in eval_ts:
        comp_pairs.append((f"varT_T{t_ev}_vs_fix2x_T{t_ev}", f"loopres_r2_varT_T{t_ev}_routed", f"loopres_r2_fix2x_T{t_ev}_routed"))
        comp_pairs.append((f"varT_T{t_ev}_vs_lora_1x", f"loopres_r2_varT_T{t_ev}_routed", "lora_r2_1x_routed"))
        comp_pairs.append((f"fix2x_T{t_ev}_vs_lora_1x", f"loopres_r2_fix2x_T{t_ev}_routed", "lora_r2_1x_routed"))

    paired_hard_60 = {
        lbl: _paired_bootstrap(arm_rows[a][:n_hard], arm_rows[b][:n_hard], seed=int(args.seed) + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    return {
        "experiment": "pocket_loopres_t_sweep_and_varT",
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "k_shots_per_skill": int(args.k_shots),
        "core_config": core_config,
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_router_base_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "payload_per_skill": payload_per_skill,
        "teacher_train_kept_frac": teacher_kept_frac,
        "router_report": router_report,
        "routed_prose_kl": routed_prose_kl,
        "unrouted_prose_kl": unrouted_prose_kl,
        "loopres_telemetry": loopres_telemetry,
        "generation_arms": arm_summaries,
        "mmlu_arms": mmlu_arms,
        "paired_hard_60": paired_hard_60,
        "best_routed_arm_name": best_routed_arm,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return
    print("\n=== POCKET LOOPRES T-SWEEP & VAR-T SUMMARY ===", flush=True)
    for arm_nm, st in payload.get("generation_arms", {}).items():
        h, g = st["hard_60"], st["guardrail_24"]
        print(
            f"  {arm_nm:33s} | hard60={h['strict_accuracy']*100:5.1f}% ({h['n_correct']:2d}/60) "
            f"[old1-4={h['old_skills_1_to_4_acc']*100:5.1f}%, s5={h['last_skill_5_acc']*100:5.1f}%, trunc={h['truncated_rate']*100:4.1f}%] "
            f"| guard={g['strict_accuracy']*100:5.1f}%",
            flush=True,
        )
    print("\n=== PAIRED BOOTSTRAP (hard_60) ===", flush=True)
    for k, v in payload.get("paired_hard_60", {}).items():
        print(
            f"  {k:40s}: diff={v['diff']*100:+5.1f} pts CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})",
            flush=True,
        )
    print("Best routed arm:", payload.get("best_routed_arm_name"), flush=True)
    print("Timing (s):", payload.get("timing_s"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B")
    ap.add_argument("--student", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--k-shots", type=int, default=6)
    ap.add_argument("--n-mmlu-test", type=int, default=200)
    ap.add_argument("--n-mmlu-router-train", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--teacher-batch-size", type=int, default=32)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--d-loop-key", type=int, default=64)
    ap.add_argument("--core-start", type=int, default=None)
    ap.add_argument("--core-end", type=int, default=None)
    ap.add_argument("--eval-ts", default="2,3,4,5")
    ap.add_argument("--var-choices", default="2,3")
    ap.add_argument("--total-steps", type=int, default=48)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--lr-lora", type=float, default=4.0e-4)
    ap.add_argument("--lr-loopres", type=float, default=2.0e-3)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--teacher-cache", default="")
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--backend", default="hf", choices=["auto", "vllm", "hf"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
        _print_summary(payload)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
