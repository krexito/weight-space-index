"""Real-World Pocket LoopRes Benchmark on MATH-500 (Qwen3-4B -> Qwen3-0.6B).

Evaluates whether LoRA + LoopRes zero-shot test-time scaling generalizes beyond synthetic few-shot skills
to real competition math (`HuggingFaceH4/MATH-500`, `n_train=300`, `n_test=200`, all 7 subjects & Levels 1..5)
while protecting general knowledge (`cais/mmlu`, `n_test=400` + `24` held-out prose controls):

  1. `base`                   : Unmodified Qwen3-0.6B (`enable_thinking=False`)
  2. `teacher`                : Teacher ceiling Qwen3-4B (`enable_thinking=False`)
  3. `lora_1x_routed`         : Domain LoRA (`r=4` all layers) + Layer-14 Ridge router
  4. `loopres_trainT_routed`  : Domain LoRA (`r=4`) + LoopRes (`window=2`), evaluated at training loops
  5. `loopres_T<t>_routed`    : Exact same LoopRes weights evaluated at test-time loop counts `--eval-ts`

CPU smoke test::
    python scripts/experiment_pocket_loopres_math500.py --smoke
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import discover_math500_blind as dm  # noqa: E402
import eval_capability_and_boundary_lfm3 as ec  # noqa: E402
from experiment_pocket_loopres import (  # noqa: E402
    AutoModelForCausalLM_load,
    _SmokeTinyTokenizer,
    _aggregate_loopres_telemetry,
    _hf_generate_batched,
    _load_smoke_toy_lm_cls,
    _score_routed_mmlu_sparse_loopres,
    _train_loopres_expert,
)
from experiment_pocket_loopres_sweep import _parse_int_tuple  # noqa: E402
from experiment_pocket_moe_ceiling import fit_ridge_router, predict_ridge_router  # noqa: E402
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
    _try_load_teacher_cache,
    format_math500_prompt_ids,
    format_mmlu_prompt_ids,
    load_math500_split,
    load_mmlu_split,
)
from experiment_reasoning_distill_pilot import _atomic_write_json, _extract_prompt_hidden_vecs  # noqa: E402
from experiment_reasoning_distill_v2 import _format_messages, _select_backend  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.loopres import LoopResConfig, LoopResModule, default_core_span  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402


def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of 3-domain LoopRes MATH-500 training, routing & test-time scaling."""
    _ToyCachedTransformerLM = _load_smoke_toy_lm_cls()

    torch.manual_seed(41)
    model = _ToyCachedTransformerLM(vocab=40, hidden=24, n_layers=6, n_heads=4, intermediate=48, seed=41).eval()
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
    res_lr = _train_loopres_expert(
        model,
        all_layers,
        tr_samples,
        rank=4,
        alpha_scale=8.0,
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
    assert res_lr["loop_counts_seen"] == {"2": 3, "3": 3}
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    tok_sm = _SmokeTinyTokenizer()
    mod_lr = LoopResModule(LoopResConfig.from_dict(res_lr["loopres_config"]))
    mod_lr.load_state_dict(res_lr["loopres_state"])
    gen_3x = _hf_generate_batched(
        model,
        tok_sm,
        [[1, 3, 5], [4, 6, 8, 12]],
        max_new_tokens=4,
        batch_size=2,
        base_seed=42,
        loopres_mod=mod_lr,
        n_loops=3,
        param_updates=res_lr["weights"],
        record_stats=True,
    )
    assert len(gen_3x) == 2
    tel_3x = mod_lr.telemetry_summary()
    assert len(tel_3x["coda_alpha_mean"]) == 3
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())
    return {
        "mode": "smoke",
        "passed": True,
        "loss_ckpts": res_lr["step_losses_checkpoints"],
        "telemetry_3x": tel_3x,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    train_loops = _parse_int_tuple(args.train_n_loops)
    eval_ts = _parse_int_tuple(args.eval_ts)
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
    mmlu_test, mmlu_router_train = load_mmlu_split(n_test=args.n_mmlu_test, n_router_train=args.n_mmlu_router_train, seed=args.seed)

    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    assert len(tok_t) == len(tok_s), f"Tokenizer vocab mismatch: teacher={len(tok_t)} vs student={len(tok_s)}"
    eos_set = set(lr.eos_ids(tok_s))

    # ---- Phase 1: Teacher Generation (Batched) + Top-32 Logit Extraction + Cache ----
    print(f"[m500-lr] Phase 1: Teacher ({args.teacher}) traces & Top-32 logits (bs={args.teacher_batch_size})...", flush=True)
    t_p1 = time.perf_counter()
    math_test_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_test]
    math_train_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_train]

    sig_hash = hashlib.sha256(
        json.dumps([math_test_pids, math_train_pids], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    cache_json = Path(args.teacher_cache) if args.teacher_cache else None
    cache_pt = cache_json.with_suffix(".logits.pt") if cache_json is not None else None

    loaded_full = False
    if cache_json is not None and cache_pt is not None and cache_json.exists() and cache_pt.exists():
        meta = json.loads(cache_json.read_text(encoding="utf-8"))
        if (
            meta.get("prompt_sig") == sig_hash
            and meta.get("teacher_model") == args.teacher
            and int(meta.get("n_train", -1)) == len(math_train)
            and int(meta.get("n_test", -1)) == len(math_test)
            and int(meta.get("max_new_tokens", -1)) == int(args.max_new_tokens)
            and int(meta.get("max_teacher_train_tokens", -1)) == int(args.max_teacher_train_tokens)
        ):
            print(f"[m500-lr] Loaded full teacher cache + Top-32 logits -> {cache_json}", flush=True)
            t_test_raw = meta["test_outputs"]
            teacher_train_kept_frac = meta["teacher_train_kept_frac"]
            fallback_counts = meta["fallback_counts"]
            pt_data = torch.load(cache_pt, map_location="cpu", weights_only=True)
            train_samples_by_domain: dict[str, list[dict[str, Any]]] = {}
            for d in MATH_DOMAINS:
                train_samples_by_domain[d] = []
                for idx_s, sp_meta in enumerate(meta["samples_meta"][d]):
                    key = f"{d}_{idx_s}"
                    train_samples_by_domain[d].append({
                        **sp_meta,
                        "topk_ids": pt_data[f"{key}.topk_ids"],
                        "topk_logits": pt_data[f"{key}.topk_logits"],
                    })
            loaded_full = True

    if not loaded_full:
        cached = _try_load_teacher_cache(
            cache_json,
            prompt_sig=sig_hash,
            seed=args.seed,
            n_train=len(math_train),
            n_test=len(math_test),
            max_new_tokens=args.max_new_tokens,
        )
        teacher_m = None
        if cached is not None:
            t_test_raw, t_train_raw = cached
        else:
            try:
                teacher_m = AutoModelForCausalLM_load(args.teacher, dtype).to(dev).eval()
            except TypeError:
                teacher_m = AutoModelForCausalLM_load(args.teacher, dtype, legacy=True).to(dev).eval()
            t_outs = _hf_generate_batched(
                teacher_m,
                tok_t,
                math_test_pids + math_train_pids,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.teacher_batch_size,
                base_seed=args.seed,
            )
            t_test_raw = t_outs[: len(math_test)]
            t_train_raw = t_outs[len(math_test) :]

        train_by_domain_kept: dict[str, list[tuple[dict[str, Any], list[int], str, bool]]] = {d: [] for d in MATH_DOMAINS}
        fallback_candidates: dict[str, list[tuple[dict[str, Any], list[int], str, bool]]] = {d: [] for d in MATH_DOMAINS}
        tot_by_domain = {d: 0 for d in MATH_DOMAINS}
        for it, p_ids, raw in zip(math_train, math_train_pids, t_train_raw):
            d = it["domain"]
            tot_by_domain[d] += 1
            g = dm.grade_math(raw["token_ids"], raw["finish_reason"], str(it["answer"]), tok_s, eos_set, args.max_new_tokens)
            body = lr.strip_eos(raw["token_ids"], eos_set)
            txt = tok_s.decode(body, skip_special_tokens=True).strip()
            if g["correct"] and (not g["looped_killed"]) and g["n_tokens"] <= int(args.max_teacher_train_tokens):
                train_by_domain_kept[d].append((it, p_ids, txt, False))
            elif it["solution"].strip():
                sol_txt = it["solution"].strip()
                if "\\boxed{" not in sol_txt:
                    sol_txt += f"\n\nFinal answer: \\boxed{{{it['answer']}}}."
                fallback_candidates[d].append((it, p_ids, sol_txt, True))

        fallback_counts = {d: 0 for d in MATH_DOMAINS}
        for d in MATH_DOMAINS:
            if len(train_by_domain_kept[d]) < 16 and fallback_candidates[d]:
                need = min(16 - len(train_by_domain_kept[d]), len(fallback_candidates[d]))
                train_by_domain_kept[d].extend(fallback_candidates[d][:need])
                fallback_counts[d] = need

        teacher_train_kept_frac = {
            d: round(sum(1 for x in train_by_domain_kept[d] if not x[3]) / max(1, tot_by_domain[d]), 4)
            for d in MATH_DOMAINS
        }
        print(f"[m500-lr] teacher kept_frac={teacher_train_kept_frac} | fallback={fallback_counts}", flush=True)

        if teacher_m is None:
            try:
                teacher_m = AutoModelForCausalLM_load(args.teacher, dtype).to(dev).eval()
            except TypeError:
                teacher_m = AutoModelForCausalLM_load(args.teacher, dtype, legacy=True).to(dev).eval()
        for p in teacher_m.parameters():
            p.requires_grad_(False)

        train_samples_by_domain = {d: [] for d in MATH_DOMAINS}
        with torch.no_grad():
            for d in MATH_DOMAINS:
                for it, p_ids, resp_txt, is_fb in train_by_domain_kept[d]:
                    full_ids = _format_messages(
                        tok_s,
                        [
                            {"role": "user", "content": it["problem"].strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."},
                            {"role": "assistant", "content": resp_txt},
                        ],
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                    lo = max(0, len(p_ids) - 1)
                    hi = len(full_ids) - 1
                    if hi <= lo:
                        continue
                    ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
                    t_logits = _tail_logits(teacher_m, ids_dev, lo)[: hi - lo].float()
                    vals, idx = t_logits.topk(32, dim=-1)
                    train_samples_by_domain[d].append({
                        "id": it["id"],
                        "domain": d,
                        "subject": it["subject"],
                        "level": int(it["level"]),
                        "is_fallback": bool(is_fb),
                        "ids": full_ids,
                        "lo": lo,
                        "hi": hi,
                        "topk_ids": idx.to(torch.int32).cpu(),
                        "topk_logits": vals.to(torch.float16).cpu(),
                    })
                    del ids_dev, t_logits

        del teacher_m
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        if cache_json is not None and cache_pt is not None:
            cache_json.parent.mkdir(parents=True, exist_ok=True)
            pt_tensors: dict[str, torch.Tensor] = {}
            samples_meta: dict[str, list[dict[str, Any]]] = {}
            for d in MATH_DOMAINS:
                samples_meta[d] = []
                for idx_s, sp in enumerate(train_samples_by_domain[d]):
                    key = f"{d}_{idx_s}"
                    pt_tensors[f"{key}.topk_ids"] = sp["topk_ids"]
                    pt_tensors[f"{key}.topk_logits"] = sp["topk_logits"]
                    samples_meta[d].append({k: v for k, v in sp.items() if k not in ("topk_ids", "topk_logits")})
            pt_tmp = cache_pt.with_suffix(".pt.tmp")
            torch.save(pt_tensors, pt_tmp)
            pt_tmp.replace(cache_pt)
            _atomic_write_json(
                cache_json,
                {
                    "teacher_model": args.teacher,
                    "prompt_sig": sig_hash,
                    "seed": int(args.seed),
                    "n_train": len(math_train),
                    "n_test": len(math_test),
                    "max_new_tokens": int(args.max_new_tokens),
                    "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
                    "test_outputs": t_test_raw,
                    "teacher_train_kept_frac": teacher_train_kept_frac,
                    "fallback_counts": fallback_counts,
                    "samples_meta": samples_meta,
                },
            )
            print(f"[m500-lr] Saved full teacher cache -> {cache_json}", flush=True)

    teacher_test_rows = _grade_math500_rows(math_test, t_test_raw, tok_s, eos_set, args.max_new_tokens)
    teacher_correct_flags = [bool(r["strict_correct"]) for r in teacher_test_rows]
    teacher_summary = _summarize_math500_arm(teacher_test_rows, teacher_correct_flags=teacher_correct_flags)
    phase1_s = round(time.perf_counter() - t_p1, 2)
    print(f"[m500-lr] teacher acc={teacher_summary['accuracy']*100:.1f}% ({phase1_s}s)", flush=True)

    # ---- Phase 2: Load Student, Layer-14 Router, Base MMLU & Prose Anchor ----
    print(f"[m500-lr] Phase 2: Loading Student ({args.student}), router & base MMLU...", flush=True)
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
        "mmlu_non_math_fire_rate": round(
            sum(p != "passthrough" for p, it in zip(routed_mmlu_test, mmlu_test) if not it["is_math_subject"])
            / max(1, sum(1 for it in mmlu_test if not it["is_math_subject"])),
            4,
        ),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_test) / len(routed_prose_test), 4),
    }
    print(f"[m500-lr] Router: {router_report}", flush=True)

    letter_ids = ec._resolve_mmlu_letter_ids(tok_s)
    base_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
    base_mmlu_summary = _summarize_mmlu(mmlu_test, base_mmlu_raw, seed=args.seed)

    def _compute_logp(prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor | None], float]:
        seqs = _to_seq_tensors(
            [
                _format_messages(
                    tok_s,
                    [{"role": "user", "content": p}, {"role": "assistant", "content": "Here is a concise explanation."}],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                for p in prompts
            ],
            torch.device("cpu"),
        )
        lps: list[torch.Tensor | None] = []
        nll_s, cnt = 0.0, 0
        with torch.no_grad():
            for ids in seqs:
                logits = _tail_logits(student_m, ids.to(dev), 0)[:-1].float()
                lp = F.log_softmax(logits, dim=-1)
                nll_s += float(F.nll_loss(lp, ids[0, 1:].to(dev), reduction="sum"))
                cnt += int(ids.shape[1] - 1)
                lps.append(lp.to(dtype=torch.bfloat16, device="cpu"))
                del logits, lp
        return seqs, lps, nll_s / max(1, cnt)

    prose_full_ids, base_prose_logp, base_prose_nll = _compute_logp(PROSE_HELDOUT)
    anchor_full_ids, base_anchor_logp, _ = _compute_logp(PROSE_ROUTER_TRAIN)
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Train `lora_1x` vs `loopres_lora` Across 3 MATH Domains ----
    print(f"[m500-lr] Phase 3: Training lora_1x (r={args.rank}) & loopres (train_loops={train_loops})...", flush=True)
    t_p3 = time.perf_counter()

    train_tiers = [
        ("lora_1x", int(args.rank), 2.0 * float(args.rank), None, (1,)),
        ("loopres_lora", int(args.rank), 2.0 * float(args.rank), "loopres", train_loops),
    ]
    eval_specs: list[tuple[str, str, int]] = [("lora_1x_routed", "lora_1x", 1)]
    for t_ev in eval_ts:
        eval_specs.append((f"loopres_T{t_ev}_routed", "loopres_lora", t_ev))

    ckpt_root = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    dir_ctx = nullcontext(str(ckpt_root)) if ckpt_root is not None else tempfile.TemporaryDirectory()

    with dir_ctx as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        artifact_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {t[0]: {} for t in train_tiers}
        domain_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {a[0]: {} for a in eval_specs}

        for d_idx, d in enumerate(MATH_DOMAINS):
            for (tier_nm, rk, a_sc, lr_mode, lp_choices) in train_tiers:
                p_art = exp_dir / f"{tier_nm}_{d}.pt"
                p_meta = exp_dir / f"{tier_nm}_{d}.meta.json"
                if ckpt_root is not None and p_art.exists() and p_meta.exists():
                    res_e = json.loads(p_meta.read_text(encoding="utf-8"))
                    art = torch.load(p_art, map_location="cpu", weights_only=True)
                    upd_w, lr_sd, lr_cfg = art["weights"], art["loopres_state"], art["loopres_config"]
                    print(f"[m500-lr] Resumed {tier_nm}/{d} from checkpoint", flush=True)
                else:
                    dom_seed = int(args.seed) + d_idx
                    torch.manual_seed(dom_seed)
                    res_e = _train_loopres_expert(
                        student_m,
                        all_layers,
                        train_samples_by_domain[d],
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
                        seed=dom_seed,
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
                    domain_mmlu_scores[arm_nm][d] = _score_routed_mmlu_sparse_loopres(
                        student_m, mmlu_test_pids, mmlu_test, routed_mmlu_test, d, base_mmlu_raw, letter_ids, dev,
                        param_updates=upd_w or None, loopres_mod=mod_lr, n_loops=a_loops,
                    )
                del mod_lr, upd_w, lr_sd
                artifact_paths[(tier_nm, d)] = p_art
                expert_meta[tier_nm][d] = res_e
                print(
                    f"[m500-lr] {tier_nm}/{d}: ({res_e['compile_time_s']:.1f}s) | "
                    f"loss={res_e['step_losses_checkpoints'][0]}->{res_e['step_losses_checkpoints'][-1]} | "
                    f"loops={res_e.get('loop_counts_seen', {})} | u_kl={res_e['healthy_kl']}",
                    flush=True,
                )

        mmlu_arms: dict[str, Any] = {"base": base_mmlu_summary}
        for idx_a, (arm_nm, *_rest) in enumerate(eval_specs):
            mmlu_arms[arm_nm] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_test, base_mmlu_raw, domain_mmlu_scores[arm_nm], seed=int(args.seed) + 1 + idx_a
            )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Unified Batched HF Evaluation on MATH-500 (n=200) ----
        print(f"[m500-lr] Phase 4: Evaluating MATH-500 (n={len(math_test)}, bs={args.batch_size})...", flush=True)
        t_p4 = time.perf_counter()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        p_base_eval = exp_dir / "eval_base.json"
        if ckpt_root is not None and p_base_eval.exists():
            cached_base = json.loads(p_base_eval.read_text(encoding="utf-8")).get("raw", [])
            if len(cached_base) == len(math_test):
                base_raw = cached_base
                print("[m500-lr] Resumed eval_base from checkpoint", flush=True)
            else:
                base_raw = _hf_generate_batched(
                    student_m, tok_s, math_test_pids,
                    max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, base_seed=args.seed,
                )
                _atomic_write_json(p_base_eval, {"raw": base_raw})
        else:
            base_raw = _hf_generate_batched(
                student_m, tok_s, math_test_pids,
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
                if len(cached_arm.get("raw", [])) == len(math_test):
                    routed_raw_by_arm[arm_nm] = cached_arm["raw"]
                    if cached_arm.get("telemetry"):
                        loopres_telemetry[arm_nm] = cached_arm["telemetry"]
                    print(f"[m500-lr] Resumed {arm_nm} from checkpoint", flush=True)
                    continue

            arm_raw: list[dict[str, Any] | None] = [None] * len(math_test)
            for i, cls_nm in enumerate(routed_math_test):
                if cls_nm == "passthrough":
                    arm_raw[i] = base_raw[i]

            dom_tels: dict[str, dict[str, Any]] = {}
            for d in MATH_DOMAINS:
                idxs = [i for i, cls_nm in enumerate(routed_math_test) if cls_nm == d]
                if not idxs:
                    continue
                art = torch.load(artifact_paths[(tier_key, d)], map_location="cpu", weights_only=True)
                upd_w = art["weights"] or None
                mod_lr = None
                if art["loopres_state"] is not None and art["loopres_config"] is not None:
                    mod_lr = LoopResModule(LoopResConfig.from_dict(art["loopres_config"])).to(dev).eval()
                    mod_lr.load_state_dict(art["loopres_state"])

                sub_outs = _hf_generate_batched(
                    student_m,
                    tok_s,
                    [math_test_pids[i] for i in idxs],
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
                    dom_tels[d] = mod_lr.telemetry_summary()
                del art, upd_w, mod_lr

            routed_raw_by_arm[arm_nm] = [r for r in arm_raw if r is not None]
            if dom_tels:
                loopres_telemetry[arm_nm] = _aggregate_loopres_telemetry(dom_tels)
            if ckpt_root is not None:
                _atomic_write_json(p_arm_eval, {"raw": routed_raw_by_arm[arm_nm], "telemetry": loopres_telemetry.get(arm_nm)})
            print(f"[m500-lr] Evaluated {arm_nm}", flush=True)

        phase4_s = round(time.perf_counter() - t_p4, 2)

    # ---- Phase 5: Grading & Paired Bootstrap ----
    arm_raw_map: dict[str, tuple[list[dict[str, Any]], list[str] | None, bool]] = {
        "base": (base_raw, None, False),
        "teacher": (t_test_raw, None, True),
    }
    for arm_nm, *_rest in eval_specs:
        arm_raw_map[arm_nm] = (routed_raw_by_arm[arm_nm], routed_math_test, False)

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
            rows = base_rows if arm_nm == "base" else _grade_math500_rows(math_test, raw_l, tok_s, eos_set, args.max_new_tokens, routed_labels=r_lbls)
            sm = _summarize_math500_arm(rows, teacher_correct_flags=teacher_correct_flags, base_acc=base_acc, teacher_acc=teacher_acc)
        math500_rows[arm_nm] = rows
        math500_arms[arm_nm] = sm

    routed_prose_kl: dict[str, float] = {}
    unrouted_prose_kl: dict[str, float] = {}
    payload_per_domain: dict[str, dict[str, float]] = {}
    for arm_nm, tier_key, _lp in eval_specs:
        tot_r = sum(float(expert_meta[tier_key][p_cls]["healthy_kl"]) for p_cls in routed_prose_test if p_cls in MATH_DOMAINS)
        routed_prose_kl[arm_nm] = round(tot_r / max(1, len(routed_prose_test)), 6)
        unrouted_prose_kl[arm_nm] = round(sum(float(expert_meta[tier_key][d]["healthy_kl"]) for d in MATH_DOMAINS) / len(MATH_DOMAINS), 6)
        payload_per_domain[arm_nm] = {
            "lora_mb": round(sum(float(expert_meta[tier_key][d]["lora_mb"]) for d in MATH_DOMAINS) / len(MATH_DOMAINS), 4),
            "loopres_kb": round(sum(float(expert_meta[tier_key][d]["loopres_kb"]) for d in MATH_DOMAINS) / len(MATH_DOMAINS), 2),
            "total_mb": round(sum(float(expert_meta[tier_key][d]["adapter_mb"]) for d in MATH_DOMAINS) / len(MATH_DOMAINS), 4),
        }

    routed_arm_names = [a[0] for a in eval_specs]
    best_routed_arm = max(
        routed_arm_names,
        key=lambda k: (math500_arms[k]["accuracy"], -payload_per_domain[k]["total_mb"]),
    )

    comp_pairs: list[tuple[str, str, str]] = [
        ("lora_1x_vs_base", "lora_1x_routed", "base"),
        (f"{best_routed_arm}_vs_base", best_routed_arm, "base"),
        (f"{best_routed_arm}_vs_lora_1x", best_routed_arm, "lora_1x_routed"),
    ]
    for t_ev in eval_ts:
        comp_pairs.append((f"loopres_T{t_ev}_vs_base", f"loopres_T{t_ev}_routed", "base"))
        comp_pairs.append((f"loopres_T{t_ev}_vs_lora_1x", f"loopres_T{t_ev}_routed", "lora_1x_routed"))
    if len(eval_ts) >= 2:
        t0_ev, t1_ev = eval_ts[0], eval_ts[1]
        comp_pairs.append((f"loopres_T{t1_ev}_vs_T{t0_ev}", f"loopres_T{t1_ev}_routed", f"loopres_T{t0_ev}_routed"))

    paired_math500 = {
        lbl: _paired_bootstrap(math500_rows[a], math500_rows[b], seed=int(args.seed) + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    return {
        "experiment": "pocket_loopres_math500",
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "backend": backend,
        "train_n_loops": list(train_loops),
        "eval_ts": list(eval_ts),
        "rank": int(args.rank),
        "max_new_tokens": int(args.max_new_tokens),
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_router_base_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "payload_per_domain": payload_per_domain,
        "teacher_train_kept_frac": teacher_train_kept_frac,
        "teacher_fallback_counts": fallback_counts,
        "router_report": router_report,
        "routed_prose_kl": routed_prose_kl,
        "unrouted_prose_kl": unrouted_prose_kl,
        "loopres_telemetry": loopres_telemetry,
        "math500_arms": math500_arms,
        "mmlu_arms": mmlu_arms,
        "paired_math500": paired_math500,
        "best_routed_arm_name": best_routed_arm,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return
    print("\n=== POCKET LOOPRES ON REAL MATH-500 ===", flush=True)
    for arm_nm, sm in payload.get("math500_arms", {}).items():
        print(
            f"  {arm_nm:25s} | acc={sm['accuracy']*100:5.1f}% ({sm['n_correct']:3d}/{sm['n_total']}) "
            f"| gap={sm['teacher_gap_recovered']} | trunc={sm['truncated_rate']*100:4.1f}% | dom={sm['per_domain']}",
            flush=True,
        )
    print("\n=== PAIRED BOOTSTRAP (MATH-500) ===", flush=True)
    for k, v in payload.get("paired_math500", {}).items():
        print(
            f"  {k:34s}: diff={v['diff']*100:+5.1f} pts CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})",
            flush=True,
        )
    print("Best routed arm:", payload.get("best_routed_arm_name"), flush=True)
    print("Timing (s):", payload.get("timing_s"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B")
    ap.add_argument("--student", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-math-train", type=int, default=300)
    ap.add_argument("--n-math-test", type=int, default=200)
    ap.add_argument("--n-mmlu-test", type=int, default=400)
    ap.add_argument("--n-mmlu-router-train", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--teacher-batch-size", type=int, default=32)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--d-loop-key", type=int, default=64)
    ap.add_argument("--core-start", type=int, default=None)
    ap.add_argument("--core-end", type=int, default=None)
    ap.add_argument("--train-n-loops", default="2")
    ap.add_argument("--eval-ts", default="2,3")
    ap.add_argument("--total-steps", type=int, default=96)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-lora", type=float, default=3.0e-4)
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
