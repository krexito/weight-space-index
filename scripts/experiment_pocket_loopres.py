"""Pocket LoopRes Benchmark (arXiv:2611.31094 + WSI MoI): Frozen/LoRA Loop Residuals vs Equal-Budget 1x LoRA.

Evaluates on the paired K=6 5-skill benchmark (`60` blind hard items: `30` renumbered + `30` rephrased,
`24` math guardrails, `200` MMLU, `16` held-out prose controls) using `enable_thinking=False` and an
identical left-padded batched HuggingFace PyTorch generator across ALL student arms (zero engine confounder):

  1. `base`                         : Unmodified student base (e.g. Qwen/Qwen3-0.6B)
  2. `teacher`                      : Unmodified teacher ceiling (e.g. Qwen/Qwen3-4B)
  3. `lora_r2_1x_routed`            : Micro-LoRA r=2 across all layers, 1x pass + balanced middle-layer router
  4. `loopres_only_2x_routed`       : Frozen backbone + LoopRes (mode='loopres', 2x core) only (~KB payload!)
  5. `gated_loop_r2_2x_routed`      : LoRA r=2 + LoopRes (mode='gated', bias-only gating, 2x core)
  6. `loopres_r2_2x_routed`         : LoRA r=2 + LoopRes (mode='loopres', 2x core, window=W)
  7. `loopres_r2_3x_testtime_routed`: Exact `loopres_r2_2x` weights evaluated zero-shot at 3x core passes

CPU smoke test::
    python scripts/experiment_pocket_loopres.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_loopres.py --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-0.6B --out /content/pocket_loopres_l4.json
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
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
    _block_prefixes,
    _enable_emb_grad_hook,
    _grade_rows,
    _vllm_generate,
    eval_healthy_drift,
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
    _chunked_distill_backward,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
)
from experiment_reasoning_distill_v2 import (  # noqa: E402
    _format_messages,
    _select_backend,
    format_question,
    grade_response,
)
from wsi.capture import find_blocks  # noqa: E402
from wsi.loopres import LoopResConfig, LoopResModule, default_core_span  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402


# ------------------------------------------------------------------ Unified Batched HF Generator
@torch.inference_mode()
def _hf_generate_batched(
    model: nn.Module,
    tok: Any,
    prompts_ids: list[list[int]],
    *,
    max_new_tokens: int,
    batch_size: int = 12,
    base_seed: int = 42,
    loopres_mod: LoopResModule | None = None,
    n_loops: int | None = None,
    param_updates: dict[str, torch.Tensor] | None = None,
    record_stats: bool = False,
) -> list[dict[str, Any]]:
    """Generate continuations using length-sorted, left-padded batched HF PyTorch inference (shared across all student arms)."""
    if not prompts_ids:
        return []
    dev = next(model.parameters()).device
    eos_list = lr.eos_ids(tok)
    eos_set = set(eos_list)
    raw_pad = getattr(tok, "pad_token_id", None)
    if raw_pad is None:
        pad_id = int(eos_list[0]) if eos_list else 0
    else:
        pad_id = int(raw_pad)

    was_training = bool(model.training)
    model.eval()
    if loopres_mod is not None:
        loopres_mod.eval()
        if record_stats:
            loopres_mod.reset_telemetry()

    w_ctx = _temp_apply_weights(model, param_updates) if param_updates else nullcontext()
    lr_ctx = (
        loopres_mod.attach(model, n_loops=n_loops, grad_checkpointing=False, record_stats=record_stats)
        if loopres_mod is not None
        else nullcontext()
    )

    order = sorted(range(len(prompts_ids)), key=lambda i: (len(prompts_ids[i]), i))
    out: list[dict[str, Any]] = [{}] * len(prompts_ids)
    try:
        with w_ctx, lr_ctx:
            bs = max(1, int(batch_size))
            for chunk_idx, b_start in enumerate(range(0, len(order), bs)):
                idx_chunk = order[b_start : b_start + bs]
                chunk = [prompts_ids[i] for i in idx_chunk]
                max_len = max(len(p) for p in chunk)
                inp = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=dev)
                mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=dev)
                for row_i, pids in enumerate(chunk):
                    if pids:
                        pad_n = max_len - len(pids)
                        inp[row_i, pad_n:] = torch.tensor(pids, dtype=torch.long, device=dev)
                        mask[row_i, pad_n:] = 1

                torch.manual_seed(int(base_seed) + chunk_idx)
                gen_kw: dict[str, Any] = dict(
                    input_ids=inp,
                    attention_mask=mask,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=lr.SAMPLING["temperature"],
                    top_p=lr.SAMPLING["top_p"],
                    top_k=lr.SAMPLING["top_k"],
                    use_cache=True,
                    pad_token_id=pad_id,
                )
                if eos_list:
                    gen_kw["eos_token_id"] = eos_list
                gen = model.generate(**gen_kw)

                for row_i, orig_i in enumerate(idx_chunk):
                    raw_gen = [int(t) for t in gen[row_i, max_len:].tolist()]
                    cut_ids: list[int] = []
                    hit_eos = False
                    for tid in raw_gen:
                        cut_ids.append(tid)
                        if tid in eos_set:
                            hit_eos = True
                            break
                    if not hit_eos:
                        while cut_ids and cut_ids[-1] == pad_id:
                            cut_ids.pop()
                    fin = "stop" if hit_eos else ("length" if len(cut_ids) >= int(max_new_tokens) else "stop")
                    out[orig_i] = {"token_ids": cut_ids, "finish_reason": fin}

                del inp, mask, gen
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()
    return out


# ------------------------------------------------------------------ LoopRes + LoRA Expert Trainer
def _train_loopres_expert(
    model: nn.Module,
    layer_idxs: list[int],
    train_samples: list[dict[str, Any]],
    *,
    rank: int,
    alpha_scale: float,
    loopres_mode: str | None,
    core_start: int,
    core_end: int,
    n_loops: int,
    window: int,
    d_loop_key: int,
    total_steps: int,
    grad_accum: int,
    lr_lora: float,
    lr_loopres: float,
    weight_decay: float = 0.05,
    lora_dropout: float = 0.10,
    warmup_frac: float = 0.15,
    distill_T: float = 2.0,
    kl_weight: float = 0.85,
    anchor_ids: list[torch.Tensor] | None = None,
    base_anchor_logp: list[torch.Tensor | None] | None = None,
    anchor_kl_weight: float = 0.05,
    healthy_ids: list[torch.Tensor] | None = None,
    base_healthy_logp: list[torch.Tensor | None] | None = None,
    base_healthy_nll: float = 0.0,
    grad_checkpointing: bool = True,
    n_loops_choices: tuple[int, ...] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train LoRA (rank>=0) and/or LoopRes (loopres_mode in {'loopres', 'gated', None}) on frozen backbone."""
    assert len(train_samples) > 0 and total_steps >= 1 and grad_accum >= 1
    assert rank > 0 or loopres_mode is not None, "At least one of rank>0 or loopres_mode must be active"
    eff_choices = tuple(int(c) for c in (n_loops_choices if n_loops_choices else (n_loops,)))
    assert all(c >= 1 for c in eff_choices), f"n_loops_choices must all be >= 1, got {eff_choices}"
    cfg_loops = max(eff_choices)

    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
    prefixes = _block_prefixes(model)
    orig_req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in orig_req:
        p.requires_grad_(False)
    was_training = bool(model.training)

    wrapped: list[tuple[nn.Module, str, nn.Linear, str, _RegLoRALinear]] = []
    if int(rank) > 0:
        for li in layer_idxs:
            blk = blocks[li]
            blk_pref = prefixes[li]
            for sub_nm, lin in [(nm, m) for nm, m in blk.named_modules() if isinstance(m, nn.Linear)]:
                parts = sub_nm.split(".")
                parent = blk
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr = parts[-1]
                wrapper = _RegLoRALinear(lin, rank=int(rank), alpha_scale=float(alpha_scale), dropout=float(lora_dropout))
                setattr(parent, attr, wrapper)
                wrapped.append((parent, attr, lin, f"{blk_pref}.{sub_nm}.weight", wrapper))

    lora_params = [p for *_rest, w in wrapped for p in (w.lora_A, w.lora_B)]
    lora_bytes = sum(w.adapter_bytes() for *_rest, w in wrapped)

    loopres_mod: LoopResModule | None = None
    loopres_params: list[nn.Parameter] = []
    if loopres_mode is not None:
        cfg_obj = getattr(model, "config", None)
        text_cfg = getattr(cfg_obj, "text_config", cfg_obj) if cfg_obj is not None else None
        d_model = int(getattr(text_cfg, "hidden_size", 0) or getattr(cfg_obj, "hidden_size", 0))
        if d_model <= 0:
            raise ValueError("Could not infer hidden_size from model.config")
        lr_cfg = LoopResConfig(
            d_model=d_model,
            core_start=int(core_start),
            core_end=int(core_end),
            n_loops=int(cfg_loops),
            window=int(window),
            d_loop_key=int(d_loop_key),
            mode=str(loopres_mode),
        )
        loopres_mod = LoopResModule(lr_cfg).to(dev)
        loopres_params = [p for p in loopres_mod.parameters() if p.requires_grad]

    loopres_bytes = loopres_mod.payload_bytes() if loopres_mod is not None else 0
    total_bytes = lora_bytes + loopres_bytes
    all_train_params = lora_params + loopres_params

    emb_handles = _enable_emb_grad_hook(model)
    step_losses: list[float] = []
    anc_kls: list[float] = []
    t_c0 = time.perf_counter()

    try:
        model.eval()
        for *_rest, w in wrapped:
            w.train()
        if loopres_mod is not None:
            loopres_mod.train()

        param_groups: list[dict[str, Any]] = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": float(lr_lora), "weight_decay": float(weight_decay)})
        if loopres_params:
            param_groups.append({"params": loopres_params, "lr": float(lr_loopres), "weight_decay": 0.0})

        n_steps = int(total_steps)
        warmup_steps = max(1, int(round(n_steps * float(warmup_frac)))) if float(warmup_frac) > 0.0 else 0
        opt = torch.optim.AdamW(param_groups)

        def _lr_lambda(step_idx: int) -> float:
            if warmup_steps > 0 and step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            rem = max(1, n_steps - warmup_steps)
            prog = min(1.0, max(0.0, float(step_idx - warmup_steps) / float(rem)))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        rng = torch.Generator().manual_seed(int(seed))
        idx_queue: list[int] = []
        loop_counts_seen: dict[str, int] = {}

        outer_ctx = (
            loopres_mod.attach(model, n_loops=eff_choices[0], grad_checkpointing=grad_checkpointing, record_stats=False)
            if (loopres_mod is not None and len(eff_choices) == 1)
            else nullcontext()
        )
        with outer_ctx:
            for step_idx in range(n_steps):
                t_step = eff_choices[step_idx % len(eff_choices)]
                loop_counts_seen[str(t_step)] = loop_counts_seen.get(str(t_step), 0) + 1
                step_ctx = (
                    loopres_mod.attach(model, n_loops=t_step, grad_checkpointing=grad_checkpointing, record_stats=False)
                    if (loopres_mod is not None and len(eff_choices) > 1)
                    else nullcontext()
                )
                with step_ctx:
                    opt.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for _acc in range(int(grad_accum)):
                        if not idx_queue:
                            idx_queue = torch.randperm(len(train_samples), generator=rng).tolist()
                        sp = train_samples[idx_queue.pop()]
                        ids_dev = torch.tensor(sp["ids"], dtype=torch.long, device=dev).unsqueeze(0)
                        lo, hi = int(sp["lo"]), int(sp["hi"])
                        with torch.enable_grad():
                            t_ids = ids_dev[0, lo + 1 : hi + 1]
                            tk_ids = sp["topk_ids"].to(dev) if sp.get("topk_ids") is not None else None
                            tk_log = sp["topk_logits"].to(dev) if sp.get("topk_logits") is not None else None
                            loss_val = _chunked_distill_backward(
                                model,
                                ids_dev,
                                lo,
                                hi,
                                t_ids,
                                tk_ids,
                                tk_log,
                                distill_T=float(distill_T),
                                kl_weight=float(kl_weight),
                                grad_accum=int(grad_accum),
                            )
                            step_loss += float(loss_val) / float(grad_accum)
                        del ids_dev, t_ids, tk_ids, tk_log

                    if float(anchor_kl_weight) > 0.0 and anchor_ids and base_anchor_logp:
                        anc_i = step_idx % len(anchor_ids)
                        anc_t = anchor_ids[anc_i]
                        lp_b_cpu = base_anchor_logp[anc_i] if anc_i < len(base_anchor_logp) else None
                        if anc_t.shape[1] >= 2 and lp_b_cpu is not None:
                            anc_dev = anc_t.to(dev)
                            with torch.enable_grad():
                                lp_s = F.log_softmax(_tail_logits(model, anc_dev, 0)[:-1].float(), dim=-1)
                                lp_b = lp_b_cpu.to(device=dev, dtype=torch.float32)
                                kl_anc = (lp_b.exp() * (lp_b - lp_s)).sum(dim=-1).clamp_min(0.0).mean()
                                (float(anchor_kl_weight) * kl_anc).backward()
                                anc_kls.append(float(kl_anc.detach().item()))
                            del anc_dev, lp_s, lp_b, kl_anc

                    torch.nn.utils.clip_grad_norm_(all_train_params, 1.0)
                    opt.step()
                    sched.step()
                    step_losses.append(step_loss)

        for *_rest, w in wrapped:
            w.eval()
        if loopres_mod is not None:
            loopres_mod.eval()

        compile_time_s = round(time.perf_counter() - t_c0, 2)

        eval_ctx = (
            loopres_mod.attach(model, n_loops=cfg_loops, grad_checkpointing=False, record_stats=False)
            if loopres_mod is not None
            else nullcontext()
        )
        with eval_ctx:
            drift = (
                eval_healthy_drift(model, healthy_ids, base_healthy_logp, base_healthy_nll)
                if (healthy_ids is not None and base_healthy_logp is not None)
                else {"healthy_kl": 0.0, "delta_healthy_nll": 0.0}
            )

        train_telemetry: dict[str, Any] | None = None
        if loopres_mod is not None:
            loopres_mod.reset_telemetry()
            with torch.no_grad(), loopres_mod.attach(model, n_loops=cfg_loops, grad_checkpointing=False, record_stats=True):
                for sp in train_samples:
                    ids_dev = torch.tensor(sp["ids"], dtype=torch.long, device=dev).unsqueeze(0)
                    _ = model(input_ids=ids_dev, use_cache=False)
                    del ids_dev
            train_telemetry = loopres_mod.telemetry_summary()

        merged_weights = {full_nm: w.merged_weight_cpu() for _, _, _, full_nm, w in wrapped}
        loopres_state_cpu = (
            {k: v.detach().cpu().clone() for k, v in loopres_mod.state_dict().items()}
            if loopres_mod is not None
            else None
        )
        loopres_config_d = loopres_mod.config.to_dict() if loopres_mod is not None else None
    finally:
        for h in emb_handles:
            h.remove()
        for parent, attr, orig_lin, _, _ in wrapped:
            setattr(parent, attr, orig_lin)
        if was_training:
            model.train()
        for p, req in orig_req:
            p.requires_grad_(req)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    n_ckpts = min(8, len(step_losses))
    ckpts: list[float] = []
    for b in range(n_ckpts):
        lo_b = (b * len(step_losses)) // n_ckpts
        hi_b = ((b + 1) * len(step_losses)) // n_ckpts
        seg = step_losses[lo_b:hi_b] or [step_losses[lo_b]]
        ckpts.append(round(sum(seg) / len(seg), 5))

    return {
        "weights": merged_weights,
        "loopres_state": loopres_state_cpu,
        "loopres_config": loopres_config_d,
        "lora_mb": round(lora_bytes / (1024**2), 4),
        "loopres_kb": round(loopres_bytes / 1024.0, 2),
        "adapter_mb": round(total_bytes / (1024**2), 4),
        "total_bytes": int(total_bytes),
        "n_train_samples": len(train_samples),
        "total_steps": int(total_steps),
        "step_losses_checkpoints": ckpts,
        "mean_anchor_kl": round(sum(anc_kls) / max(1, len(anc_kls)), 6) if anc_kls else 0.0,
        "compile_time_s": compile_time_s,
        "loop_counts_seen": loop_counts_seen,
        "train_telemetry": train_telemetry,
        **drift,
    }


def _score_routed_mmlu_sparse_loopres(
    model: nn.Module,
    mmlu_test_pids: list[list[int]],
    mmlu_test: list[dict[str, Any]],
    routed_mmlu_classes: list[str],
    skill_name: str,
    base_mmlu_raw: dict[str, Any],
    letter_ids: dict[str, list[int]],
    dev: torch.device,
    *,
    param_updates: dict[str, torch.Tensor] | None = None,
    loopres_mod: LoopResModule | None = None,
    n_loops: int | None = None,
) -> dict[str, Any]:
    """Score MMLU only on items routed to `skill_name` with LoRA and/or LoopRes attached."""
    idxs = [i for i, cls_nm in enumerate(routed_mmlu_classes) if cls_nm == skill_name]
    if not idxs:
        return base_mmlu_raw
    sub_pids = [mmlu_test_pids[i] for i in idxs]
    sub_items = [mmlu_test[i] for i in idxs]
    w_ctx = _temp_apply_weights(model, param_updates) if param_updates else nullcontext()
    lr_ctx = (
        loopres_mod.attach(model, n_loops=n_loops, grad_checkpointing=False, record_stats=False)
        if loopres_mod is not None
        else nullcontext()
    )
    with w_ctx, lr_ctx:
        sub_sc = _score_mmlu_fast(model, sub_pids, sub_items, letter_ids, dev)

    probs = base_mmlu_raw["probs"].clone()
    preds = list(base_mmlu_raw["preds"])
    correct = list(base_mmlu_raw["correct"])
    masses = list(base_mmlu_raw["choice_masses"])
    for k, orig_i in enumerate(idxs):
        probs[orig_i] = sub_sc["probs"][k]
        preds[orig_i] = sub_sc["preds"][k]
        correct[orig_i] = sub_sc["correct"][k]
        masses[orig_i] = sub_sc["choice_masses"][k]
    return {"probs": probs, "preds": preds, "correct": correct, "choice_masses": masses}


def _aggregate_loopres_telemetry(per_skill_tels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Token-weighted Coda telemetry across fired skills."""
    tot_tok = 0
    sum_mean: list[float] | None = None
    sum_std = 0.0
    sum_ent = 0.0
    for tel in per_skill_tels.values():
        n = int(tel.get("n_tokens", 0))
        cm = tel.get("coda_alpha_mean") or []
        cs = tel.get("coda_alpha_std") or []
        if n <= 0 or not cm:
            continue
        if sum_mean is None:
            sum_mean = [0.0] * len(cm)
        if len(cm) != len(sum_mean):
            continue
        for j, v in enumerate(cm):
            sum_mean[j] += float(v) * n
        mean_std = sum(float(x) for x in cs) / max(1, len(cs))
        sum_std += mean_std * n
        sum_ent += float(tel.get("coda_entropy", 0.0)) * n
        tot_tok += n

    if tot_tok == 0 or sum_mean is None:
        return {"n_tokens": 0, "coda_alpha_mean": [], "mean_alpha_std": 0.0, "coda_entropy": 0.0, "per_skill": per_skill_tels}
    return {
        "n_tokens": tot_tok,
        "coda_alpha_mean": [round(x / tot_tok, 6) for x in sum_mean],
        "mean_alpha_std": round(sum_std / tot_tok, 6),
        "coda_entropy": round(sum_ent / tot_tok, 6),
        "per_skill": per_skill_tels,
    }


# ------------------------------------------------------------------ Summary Printer
def _print_loopres_summary(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return

    print("\n=== 1. DISENO POCKET LOOPRES (arXiv:2611.31094 + WSI MoI) ===", flush=True)
    print("Core config:", payload.get("core_config"), flush=True)
    print("Compile time (s):", payload.get("compile_time_s"), flush=True)
    print("Adapter payload/skill:", payload.get("payload_per_skill"), flush=True)
    print("Teacher kept frac:", payload.get("teacher_train_kept_frac"), flush=True)
    print("Router report:", payload.get("router_report"), flush=True)

    r_kl = payload.get("routed_prose_kl", {})
    u_kl = payload.get("unrouted_prose_kl", {})
    pay_map = payload.get("payload_per_skill", {})
    tel_map = payload.get("loopres_telemetry", {})
    print("\n=== 2. RESULTADOS PAREADOS (7 BRAZOS, MOTOR UNIFICADO HF BATCHED) ===", flush=True)
    for arm_nm, st in payload.get("generation_arms", {}).items():
        h, g = st["hard_60"], st["guardrail_24"]
        gm = g.get("guardrail_exact_match_vs_base")
        gm_s = f"{gm*100:5.1f}%" if gm is not None else "   n/a"
        p_info = pay_map.get(arm_nm)
        p_s = f"{p_info['total_mb']:5.2f}MB" if p_info is not None else "   ref "
        print(
            f"  {arm_nm:31s} | {p_s} | hard60={h['strict_accuracy']*100:5.1f}% ({h['n_correct']:2d}/60) "
            f"[old1-4={h['old_skills_1_to_4_acc']*100:5.1f}%, s5={h['last_skill_5_acc']*100:5.1f}%, "
            f"ren={h['renumbered_30']*100:5.1f}%, rep={h['rephrased_30']*100:5.1f}%, trunc={h['truncated_rate']*100:4.1f}%] | "
            f"guard={g['strict_accuracy']*100:5.1f}% (ex={gm_s}) | r_kl={r_kl.get(arm_nm, 0.0):.5f} u_kl={u_kl.get(arm_nm, 0.0):.5f}",
            flush=True,
        )
        print(f"     per_skill={h['per_skill']}", flush=True)
        if arm_nm in tel_map:
            t_arm = tel_map[arm_nm]
            print(
                f"     loopres_coda: alpha_mean={t_arm['coda_alpha_mean']} "
                f"std={t_arm['mean_alpha_std']:.4f} H={t_arm['coda_entropy']:.4f} tok={t_arm['n_tokens']}",
                flush=True,
            )

    print("\n=== 3. RETENCION MMLU ===", flush=True)
    for arm_nm, vm in payload.get("mmlu_arms", {}).items():
        a_m = vm.get("all_400") or {}
        nm = vm.get("non_math") or {}
        d_nm = f"{nm['delta_vs_base']*100:+5.1f} pts" if "delta_vs_base" in nm else "     base"
        print(
            f"  {arm_nm:31s} | MMLU={a_m.get('accuracy', 0.0)*100:5.1f}% | non_math={nm.get('accuracy', 0.0)*100:5.1f}% ({d_nm}) | "
            f"ABCD={a_m.get('mean_abcd_prob_mass', 0.0)*100:5.1f}% | fire={vm.get('router_fire_rate', 0.0)*100:5.1f}%",
            flush=True,
        )

    print("\n=== 4. COMPARACIONES PAREADAS (IC 95% BOOTSTRAP) ===", flush=True)
    for grp_lbl, grp_key in [("  [hard_60]", "paired_hard_60"), ("  [old_skills_1_to_4 (48)]", "paired_old_skills_48")]:
        print(grp_lbl, flush=True)
        for k, v in payload.get(grp_key, {}).items():
            print(
                f"    {k:40s}: diff={v['diff']*100:+5.1f} pts "
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
class _SmokeTinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = -1

    def convert_tokens_to_ids(self, _tok: str) -> int:
        return -1


def _load_smoke_toy_lm_cls() -> type[nn.Module]:
    import importlib.util

    test_path = ROOT / "tests" / "test_loopres.py"
    spec = importlib.util.spec_from_file_location("_wsi_test_loopres_smoke", test_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._ToyCachedTransformerLM


def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of LoopRes + LoRA training, restoration, batched generate & 3x evaluation."""
    _ToyCachedTransformerLM = _load_smoke_toy_lm_cls()

    torch.manual_seed(23)
    model = _ToyCachedTransformerLM(vocab=40, hidden=24, n_layers=6, n_heads=4, intermediate=48, seed=23).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _, blocks = find_blocks(model)
    core_s, core_e = default_core_span(len(blocks))
    assert (core_s, core_e) == (1, 5)
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

    modes = [
        ("lora_r2_1x", 2, 4.0, None),
        ("loopres_only_2x", 0, 0.0, "loopres"),
        ("gated_loop_r2_2x", 2, 4.0, "gated"),
        ("loopres_r2_2x", 2, 4.0, "loopres"),
    ]
    smoke_runs: dict[str, dict[str, Any]] = {}
    trained_artifacts: dict[str, dict[str, Any]] = {}
    for nm, rk, a_sc, lr_mode in modes:
        torch.manual_seed(42)
        res = _train_loopres_expert(
            model,
            all_layers,
            tr_samples,
            rank=rk,
            alpha_scale=a_sc,
            loopres_mode=lr_mode,
            core_start=core_s,
            core_end=core_e,
            n_loops=2,
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
        assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items()), nm
        ckpts = res["step_losses_checkpoints"]
        assert len(ckpts) == 6 and ckpts[-1] < ckpts[0], (nm, ckpts)
        smoke_runs[nm] = {
            "lora_mb": res["lora_mb"],
            "loopres_kb": res["loopres_kb"],
            "total_bytes": res["total_bytes"],
            "step_losses_checkpoints": ckpts,
            "healthy_kl": res["healthy_kl"],
        }
        trained_artifacts[nm] = res

    # Verify left-padded batched generation + 3x zero-shot extrapolation + telemetry
    tok_sm = _SmokeTinyTokenizer()
    prompts = [[1, 3, 5], [4, 6, 8, 12], [7, 9, 11, 13, 15]]
    lr_art = trained_artifacts["loopres_r2_2x"]
    mod_lr = LoopResModule(LoopResConfig.from_dict(lr_art["loopres_config"]))
    mod_lr.load_state_dict(lr_art["loopres_state"])

    gen_2x = _hf_generate_batched(
        model, tok_sm, prompts, max_new_tokens=5, batch_size=2, base_seed=42,
        loopres_mod=mod_lr, n_loops=2, param_updates=lr_art["weights"], record_stats=True,
    )
    tel_2x = mod_lr.telemetry_summary()
    gen_3x = _hf_generate_batched(
        model, tok_sm, prompts, max_new_tokens=5, batch_size=2, base_seed=42,
        loopres_mod=mod_lr, n_loops=3, param_updates=lr_art["weights"], record_stats=True,
    )
    tel_3x = mod_lr.telemetry_summary()
    assert len(gen_2x) == 3 and len(gen_3x) == 3
    assert len(tel_2x["coda_alpha_mean"]) == 2 and len(tel_3x["coda_alpha_mean"]) == 3
    assert abs(sum(tel_3x["coda_alpha_mean"]) - 1.0) < 1e-5
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    return {
        "mode": "smoke",
        "passed": True,
        "core_span": [core_s, core_e],
        "smoke_runs": smoke_runs,
        "telemetry_2x": tel_2x,
        "telemetry_3x": tel_3x,
    }


# ------------------------------------------------------------------ Teacher Cache Helpers
def _load_or_collect_teacher(
    args: argparse.Namespace,
    ds: dict[str, Any],
    all_test: list[dict[str, Any]],
    test_pids: list[list[int]],
    train_pids: list[list[int]],
    test_seeds: list[int],
    tok_t: Any,
    tok_s: Any,
    eos_set: set[int],
    think_end_id: int | None,
    backend: str,
    dev: torch.device,
    dtype: torch.dtype,
    dtype_vllm_str: str,
) -> dict[str, Any]:
    hard_train = ds["hard_train"]
    cache_json = Path(args.teacher_cache) if args.teacher_cache else None
    cache_pt = cache_json.with_suffix(".logits.pt") if cache_json is not None else None

    if cache_json is not None and cache_pt is not None and cache_json.exists() and cache_pt.exists():
        meta = json.loads(cache_json.read_text(encoding="utf-8"))
        if (
            meta.get("teacher_model") == args.teacher
            and int(meta.get("k_shots", -1)) == int(args.k_shots)
            and int(meta.get("max_teacher_train_tokens", -1)) == int(args.max_teacher_train_tokens)
            and len(meta.get("t_test_raw", [])) == len(all_test)
        ):
            print(f"[loopres] Loaded cached teacher outputs & Top-32 logits from {cache_json}", flush=True)
            pt_data = torch.load(cache_pt, map_location="cpu", weights_only=True)
            primary_by_skill: dict[str, list[dict[str, Any]]] = {}
            for sk in SKILLS_5:
                primary_by_skill[sk] = []
                for idx_s, sp_meta in enumerate(meta["primary_by_skill_meta"][sk]):
                    key = f"{sk}_{idx_s}"
                    primary_by_skill[sk].append({
                        **sp_meta,
                        "topk_ids": pt_data[f"{key}.topk_ids"],
                        "topk_logits": pt_data[f"{key}.topk_logits"],
                    })
            return {
                "t_test_raw": meta["t_test_raw"],
                "primary_resp_by_item": meta["primary_resp_by_item"],
                "teacher_kept_frac": meta["teacher_kept_frac"],
                "primary_by_skill": primary_by_skill,
            }
        print(f"[loopres] Teacher cache {cache_json} metadata mismatch; regenerating...", flush=True)

    train_seeds = [int(args.seed) + 1000 + i for i in range(len(train_pids))]
    teacher_hf = None
    outs_1a: list[dict[str, Any]] | None = None
    if backend == "vllm":
        try:
            outs_1a = _vllm_generate(
                args.teacher,
                tok_t,
                test_pids + train_pids,
                test_seeds + train_seeds,
                max_new_tokens=args.max_new_tokens,
                dtype_str=dtype_vllm_str,
                base_seed=args.seed,
            )
        except Exception as err:  # noqa: BLE001
            print(f"[loopres] vLLM teacher generation failed ({type(err).__name__}: {err}); falling back to batched HF...", flush=True)
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    if outs_1a is None:
        try:
            teacher_hf = AutoModelForCausalLM_load(args.teacher, dtype).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM_load(args.teacher, dtype, legacy=True).to(dev).eval()
        outs_1a = _hf_generate_batched(
            teacher_hf,
            tok_t,
            test_pids + train_pids,
            max_new_tokens=args.max_new_tokens,
            batch_size=int(getattr(args, "teacher_batch_size", None) or args.batch_size),
            base_seed=args.seed,
        )

    n_test = len(all_test)
    t_test_raw = outs_1a[:n_test]
    t_train_raw = outs_1a[n_test:]

    kept = {sk: 0 for sk in SKILLS_5}
    primary_resp_by_item: list[str] = []
    primary_pairs: dict[str, list[tuple[str, str]]] = {sk: [] for sk in SKILLS_5}
    for it, raw in zip(hard_train, t_train_raw):
        sk = it["family"]
        body = lr.strip_eos(raw["token_ids"], eos_set)
        txt = tok_s.decode(body, skip_special_tokens=True).strip()
        g = grade_response(
            raw["token_ids"],
            txt,
            int(it["answer"]),
            thinking_enabled=False,
            finish_reason=raw["finish_reason"],
            eos_set=eos_set,
            tok=tok_s,
            think_end_id=think_end_id,
        )
        if g["strict_correct"] and g["has_boxed"] and g["total_tokens"] <= int(args.max_teacher_train_tokens):
            chosen = txt
            kept[sk] += 1
        else:
            chosen = it["fallback_response"]
        primary_resp_by_item.append(chosen)
        primary_pairs[sk].append((it["question"], chosen))

    teacher_kept_frac = {sk: round(kept[sk] / max(1, args.k_shots), 4) for sk in SKILLS_5}
    print(f"[loopres] teacher kept_frac={teacher_kept_frac}", flush=True)

    if teacher_hf is None:
        try:
            teacher_hf = AutoModelForCausalLM_load(args.teacher, dtype).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM_load(args.teacher, dtype, legacy=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    print(f"[loopres] Extracting Top-32 teacher logits for {sum(len(v) for v in primary_pairs.values())} traces...", flush=True)
    primary_by_skill: dict[str, list[dict[str, Any]]] = {sk: [] for sk in SKILLS_5}
    with torch.no_grad():
        for sk in SKILLS_5:
            for q_txt, resp_txt in primary_pairs[sk]:
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
                primary_by_skill[sk].append({
                    "family": sk,
                    "ids": full_ids,
                    "lo": lo,
                    "hi": hi,
                    "topk_ids": idx.to(torch.int32).cpu(),
                    "topk_logits": vals.to(torch.float16).cpu(),
                })
                del ids_dev, t_logits

    del teacher_hf
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if cache_json is not None and cache_pt is not None:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        pt_tensors: dict[str, torch.Tensor] = {}
        meta_by_skill: dict[str, list[dict[str, Any]]] = {}
        for sk in SKILLS_5:
            meta_by_skill[sk] = []
            for idx_s, sp in enumerate(primary_by_skill[sk]):
                key = f"{sk}_{idx_s}"
                pt_tensors[f"{key}.topk_ids"] = sp["topk_ids"]
                pt_tensors[f"{key}.topk_logits"] = sp["topk_logits"]
                meta_by_skill[sk].append({"family": sk, "ids": sp["ids"], "lo": sp["lo"], "hi": sp["hi"]})
        pt_tmp = cache_pt.with_suffix(".pt.tmp")
        torch.save(pt_tensors, pt_tmp)
        pt_tmp.replace(cache_pt)
        _atomic_write_json(
            cache_json,
            {
                "teacher_model": args.teacher,
                "k_shots": int(args.k_shots),
                "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
                "t_test_raw": t_test_raw,
                "primary_resp_by_item": primary_resp_by_item,
                "teacher_kept_frac": teacher_kept_frac,
                "primary_by_skill_meta": meta_by_skill,
            },
        )
        print(f"[loopres] Saved teacher cache -> {cache_json}", flush=True)

    return {
        "t_test_raw": t_test_raw,
        "primary_resp_by_item": primary_resp_by_item,
        "teacher_kept_frac": teacher_kept_frac,
        "primary_by_skill": primary_by_skill,
    }


def AutoModelForCausalLM_load(model_name_or_path: str, dtype: torch.dtype, *, legacy: bool = False) -> nn.Module:
    from transformers import AutoModelForCausalLM

    dt_key = "torch_dtype" if legacy else "dtype"
    for extra in ({"attn_implementation": "sdpa"}, {}):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **{dt_key: dtype, "trust_remote_code": True, **extra}
            )
        except TypeError:
            continue
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **{dt_key: dtype, "trust_remote_code": True})


# ------------------------------------------------------------------ Full GPU Experiment
def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

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

    test_pids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in all_test
    ]
    train_pids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in hard_train
    ]
    test_seeds = [int(args.seed) + i for i in range(len(all_test))]

    # ---- Phase 1: Teacher Generation & Top-32 Logit Extraction ----
    print(f"[loopres] Phase 1: Teacher ({args.teacher}) traces & Top-32 logits...", flush=True)
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
    print(
        f"[loopres] teacher: hard={teacher_summary['hard_60']['strict_accuracy']*100:.1f}% | "
        f"guard={teacher_summary['guardrail_24']['strict_accuracy']*100:.1f}%",
        flush=True,
    )
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student, Dynamic Middle-Layer Router, Base MMLU & Prose Anchor ----
    print(f"[loopres] Phase 2: Loading Student ({args.student}), dynamic middle-layer router & base MMLU...", flush=True)
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
        "n_loops_train": int(args.n_loops),
        "n_loops_testtime": int(args.n_loops) + 1,
        "window": int(args.window),
        "d_loop_key": int(args.d_loop_key),
        "rank": int(args.rank),
        "router_layer": router_layer,
    }
    print(f"[loopres] Core config: {core_config}", flush=True)

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
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_classes) / len(mmlu_test), 4),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_classes) / len(routed_prose_classes), 4),
    }
    print(f"[loopres] Router report: {router_report}", flush=True)

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

    # ---- Phase 3: Train 4 Skilled Variants (x5 skills) ----
    print("[loopres] Phase 3: Training 4 paired methods x 5 skills...", flush=True)
    t_p3 = time.perf_counter()

    # (tier_nm, rank, alpha_scale, loopres_mode, n_loops_train)
    train_tiers = [
        ("lora_r2_1x", int(args.rank), 2.0 * float(args.rank), None, 1),
        ("loopres_only_2x", 0, 0.0, "loopres", int(args.n_loops)),
        ("gated_loop_r2_2x", int(args.rank), 2.0 * float(args.rank), "gated", int(args.n_loops)),
        ("loopres_r2_2x", int(args.rank), 2.0 * float(args.rank), "loopres", int(args.n_loops)),
    ]

    ckpt_root = Path(args.checkpoint_dir) if getattr(args, "checkpoint_dir", "") else None
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    dir_ctx = nullcontext(str(ckpt_root)) if ckpt_root is not None else tempfile.TemporaryDirectory()

    with dir_ctx as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        artifact_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {t[0]: {} for t in train_tiers}
        domain_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {t[0]: {} for t in train_tiers}
        domain_mmlu_3x: dict[str, dict[str, Any]] = {}

        for s_idx, sk in enumerate(SKILLS_5):
            for (tier_nm, rk, a_sc, lr_mode, n_lp) in train_tiers:
                p_art = exp_dir / f"{tier_nm}_{sk}.pt"
                p_meta = exp_dir / f"{tier_nm}_{sk}.meta.json"
                if ckpt_root is not None and p_art.exists() and p_meta.exists():
                    res_e = json.loads(p_meta.read_text(encoding="utf-8"))
                    art = torch.load(p_art, map_location="cpu", weights_only=True)
                    upd_w = art["weights"]
                    lr_sd = art["loopres_state"]
                    lr_cfg = art["loopres_config"]
                    print(f"[loopres] Resumed {tier_nm}/{sk} from checkpoint", flush=True)
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
                        n_loops=n_lp,
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
                        grad_checkpointing=True,
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

                domain_mmlu_scores[tier_nm][sk] = _score_routed_mmlu_sparse_loopres(
                    student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev,
                    param_updates=upd_w or None, loopres_mod=mod_lr, n_loops=n_lp,
                )
                if tier_nm == "loopres_r2_2x":
                    assert mod_lr is not None
                    domain_mmlu_3x[sk] = _score_routed_mmlu_sparse_loopres(
                        student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev,
                        param_updates=upd_w or None, loopres_mod=mod_lr, n_loops=int(args.n_loops) + 1,
                    )
                del mod_lr, upd_w, lr_sd
                artifact_paths[(tier_nm, sk)] = p_art
                expert_meta[tier_nm][sk] = res_e
                print(
                    f"[loopres] {tier_nm}/{sk}: lora={res_e['lora_mb']:.2f}MB lr={res_e['loopres_kb']:.1f}KB "
                    f"({res_e['compile_time_s']:.1f}s) | loss={res_e['step_losses_checkpoints'][0]}->{res_e['step_losses_checkpoints'][-1]} "
                    f"| u_kl={res_e['healthy_kl']}",
                    flush=True,
                )

        mmlu_arms: dict[str, Any] = {"base": base_mmlu_summary}
        for idx_t, (tier_nm, *_rest) in enumerate(train_tiers):
            mmlu_arms[f"{tier_nm}_routed"] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_scores[tier_nm], seed=int(args.seed) + 1 + idx_t
            )
        mmlu_arms["loopres_r2_3x_testtime_routed"] = _compose_routed_mmlu(
            mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_3x, seed=int(args.seed) + 9
        )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Unified Batched HF Evaluation Across All Student Arms ----
        print(
            f"[loopres] Phase 4: Evaluating all student arms via batched HF (batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens})...",
            flush=True,
        )
        t_p4 = time.perf_counter()
        p_base_eval = exp_dir / "eval_base.json"
        if ckpt_root is not None and p_base_eval.exists():
            cached_base = json.loads(p_base_eval.read_text(encoding="utf-8")).get("raw", [])
            if len(cached_base) == len(all_test):
                base_raw = cached_base
                print("[loopres] Resumed eval_base from checkpoint", flush=True)
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

        eval_specs = [
            ("lora_r2_1x_routed", "lora_r2_1x", 1),
            ("loopres_only_2x_routed", "loopres_only_2x", int(args.n_loops)),
            ("gated_loop_r2_2x_routed", "gated_loop_r2_2x", int(args.n_loops)),
            ("loopres_r2_2x_routed", "loopres_r2_2x", int(args.n_loops)),
            ("loopres_r2_3x_testtime_routed", "loopres_r2_2x", int(args.n_loops) + 1),
        ]

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
                    print(f"[loopres] Resumed {arm_nm} from checkpoint", flush=True)
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
            print(f"[loopres] Evaluated {arm_nm}", flush=True)

        phase4_s = round(time.perf_counter() - t_p4, 2)

    # ---- Phase 5: Grading, Paired Bootstrap CIs & Summary ----
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
    compile_time_s: dict[str, float] = {
        f"{tier_nm}_mean_per_skill_s": round(sum(expert_meta[tier_nm][s]["compile_time_s"] for s in SKILLS_5) / len(SKILLS_5), 2)
        for tier_nm, *_rest in train_tiers
    }

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

    comp_pairs = [
        ("loopres_only_2x_vs_base", "loopres_only_2x_routed", "base"),
        ("loopres_r2_2x_vs_lora_r2_1x", "loopres_r2_2x_routed", "lora_r2_1x_routed"),
        ("loopres_r2_2x_vs_gated_loop_r2_2x", "loopres_r2_2x_routed", "gated_loop_r2_2x_routed"),
        ("gated_loop_r2_2x_vs_lora_r2_1x", "gated_loop_r2_2x_routed", "lora_r2_1x_routed"),
        ("loopres_r2_3x_vs_loopres_r2_2x", "loopres_r2_3x_testtime_routed", "loopres_r2_2x_routed"),
        ("loopres_r2_3x_vs_lora_r2_1x", "loopres_r2_3x_testtime_routed", "lora_r2_1x_routed"),
        ("loopres_r2_2x_vs_base", "loopres_r2_2x_routed", "base"),
        ("lora_r2_1x_vs_base", "lora_r2_1x_routed", "base"),
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
    g_acc = lambda k: float(arm_summaries[k]["guardrail_24"]["strict_accuracy"])  # noqa: E731

    return {
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "backend_teacher": backend,
        "backend_student": "hf_batched",
        "k_shots_per_skill": int(args.k_shots),
        "max_new_tokens": int(args.max_new_tokens),
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "batch_size": int(args.batch_size),
        "core_config": core_config,
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_router_base_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "compile_time_s": compile_time_s,
        "payload_per_skill": payload_per_skill,
        "teacher_train_kept_frac": teacher_kept_frac,
        "router_report": router_report,
        "expert_meta": expert_meta,
        "routed_prose_kl": routed_prose_kl,
        "unrouted_prose_kl": unrouted_prose_kl,
        "loopres_telemetry": loopres_telemetry,
        "generation_arms": arm_summaries,
        "mmlu_arms": mmlu_arms,
        "paired_hard_60": paired_hard_60,
        "paired_old_skills_48": paired_old_skills_48,
        "best_routed_arm_name": best_routed_arm,
        "criteria": {
            "a_loopres_only_beats_base": bool(h_acc("loopres_only_2x_routed") > h_acc("base")),
            "b_loopres_r2_2x_beats_lora_r2_1x": bool(h_acc("loopres_r2_2x_routed") > h_acc("lora_r2_1x_routed")),
            "c_loopres_beats_gated_ablation": bool(h_acc("loopres_r2_2x_routed") >= h_acc("gated_loop_r2_2x_routed")),
            "d_zero_shot_3x_matches_or_beats_2x": bool(h_acc("loopres_r2_3x_testtime_routed") >= h_acc("loopres_r2_2x_routed")),
            "e_zero_general_forgetting_all_routed": bool(
                all(
                    routed_prose_kl[nm] <= 0.005
                    and nm_mmlu(nm) >= nm_mmlu("base") - 0.015
                    and g_acc(nm) >= g_acc("base") - 0.02
                    for nm in routed_arm_names
                )
            ),
        },
        "per_item_generation": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU verification")
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B")
    ap.add_argument("--student", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--k-shots", type=int, default=6)
    ap.add_argument("--n-mmlu-test", type=int, default=200)
    ap.add_argument("--n-mmlu-router-train", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--n-loops", type=int, default=2)
    ap.add_argument("--d-loop-key", type=int, default=64)
    ap.add_argument("--core-start", type=int, default=None)
    ap.add_argument("--core-end", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=48)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--lr-lora", type=float, default=4.0e-4)
    ap.add_argument("--lr-loopres", type=float, default=2.0e-3)
    ap.add_argument("--teacher-cache", default="")
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
        _print_loopres_summary(payload)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
