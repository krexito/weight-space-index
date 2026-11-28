"""Additive Looped MoI Pilot: Cloned Middle-Layer Recurrent Stack + MoI Router vs 1x Additive & Equal-Param LoRA.

Tests directly whether adding a cloned trainable recurrent Transformer layer (`AdditiveLoopedMoI`)
at an intermediate layer `L` improves mathematical reasoning as loop depth increases (`T=1, 2, 3, 4, 6`),
beats a strict equal-parameter 1x additive layer (`T=1` fixed) and an equal-parameter LoRA across the
backbone, generalizes OOD (`GSM8K train -> MATH-500 test`), and preserves general knowledge (`MMLU` +
prose `0.0%` collateral drift) via a middle-layer MoI router.

CPU smoke test::
    python scripts/experiment_additive_loop_moi.py --smoke

GPU 10-min pilot::
    python scripts/experiment_additive_loop_moi.py --model Qwen/Qwen3-0.6B --out additive_loop_moi_qwen3_06b.json
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
import random
import re
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
    _enable_emb_grad_hook,
    eval_healthy_drift,
    fit_ridge_router,
    predict_ridge_router,
)
from experiment_pocket_moe_controls import _paired_bootstrap  # noqa: E402
from experiment_pocket_moe_math500 import (  # noqa: E402
    PROSE_HELDOUT,
    PROSE_ROUTER_TRAIN,
    _score_mmlu_fast,
    _summarize_mmlu,
    _temp_apply_weights,
    format_math500_prompt_ids,
    format_mmlu_prompt_ids,
    load_math500_split,
    load_mmlu_split,
)
from experiment_pocket_moe_regularized import _RegLoRALinear  # noqa: E402
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
)
from experiment_reasoning_distill_v2 import _format_messages, format_question  # noqa: E402
from wsi.additive_loop import AdditiveLoopedMoI  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402

ADDITIVE_ROUTER_CLASSES: tuple[str, str] = ("math", "passthrough")
_CALC_MARK_RE = re.compile(r"<<[^<>]*>>")


def _model_load(model_name_or_path: str, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoModelForCausalLM

    for kw in (
        {"dtype": dtype, "trust_remote_code": True, "attn_implementation": "sdpa"},
        {"dtype": dtype, "trust_remote_code": True},
        {"torch_dtype": dtype, "trust_remote_code": True},
    ):
        try:
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kw)
        except TypeError:
            continue
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype, trust_remote_code=True)


def _format_gsm8k_sft_pair(tok: Any, question: str, raw_answer: str, max_seq_len: int) -> dict[str, Any] | None:
    ans_clean = ec.extract_gsm8k_answer(raw_answer, is_gold=True)
    if ans_clean is None:
        return None
    cot = raw_answer.split("####")[0].strip() if "####" in raw_answer else raw_answer.strip()
    cot = _CALC_MARK_RE.sub("", cot).strip()
    target_text = f"{cot}\n\nFinal Answer: \\boxed{{{ans_clean}}}"
    user_msg = {"role": "user", "content": format_question(question)}
    asst_msg = {"role": "assistant", "content": target_text}
    prompt_ids = _format_messages(tok, [user_msg], add_generation_prompt=True, enable_thinking=False)
    full_ids = _format_messages(tok, [user_msg, asst_msg], add_generation_prompt=False, enable_thinking=False)
    if len(full_ids) <= len(prompt_ids) + 2 or len(full_ids) > int(max_seq_len):
        return None
    return {"ids": full_ids, "prompt_len": len(prompt_ids)}


def _select_equal_param_lora_rank(model: nn.Module, target_params: int) -> tuple[int, int]:
    """Choose the all-layer LoRA rank whose total parameter count is closest to ``target_params``."""
    _, blocks = find_blocks(model)
    linears = [m for blk in blocks for m in blk.modules() if isinstance(m, nn.Linear)]
    candidates = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64]
    best_rank = candidates[0]
    best_count = 0
    best_gap = float("inf")
    for r in candidates:
        cnt = sum(int(r) * (int(lin.in_features) + int(lin.out_features)) for lin in linears)
        gap = abs(cnt - int(target_params))
        if gap < best_gap:
            best_gap = gap
            best_rank = int(r)
            best_count = int(cnt)
    return best_rank, best_count


# ------------------------------------------------------------------ Unified Batched HF Generator
@torch.inference_mode()
def _hf_generate_batched_addloop(
    model: nn.Module,
    tok: Any,
    prompts_ids: list[list[int]],
    *,
    max_new_tokens: int,
    batch_size: int = 32,
    base_seed: int = 42,
    add_mod: AdditiveLoopedMoI | None = None,
    n_loops: int | None = None,
    param_updates: dict[str, torch.Tensor] | None = None,
    record_stats: bool = False,
) -> list[dict[str, Any]]:
    """Length-sorted, left-padded batched HF generator shared verbatim across every evaluation arm."""
    if not prompts_ids:
        return []
    dev = next(model.parameters()).device
    eos_list = lr.eos_ids(tok)
    eos_set = set(eos_list)
    raw_pad = getattr(tok, "pad_token_id", None)
    pad_id = (int(eos_list[0]) if eos_list else 0) if raw_pad is None else int(raw_pad)

    was_training = bool(model.training)
    model.eval()
    if add_mod is not None:
        add_mod.eval()
        if record_stats:
            add_mod.reset_telemetry()

    w_ctx = _temp_apply_weights(model, param_updates) if param_updates else nullcontext()
    a_ctx = (
        add_mod.attach(model, n_loops=n_loops, grad_checkpointing=False, record_stats=record_stats)
        if add_mod is not None
        else nullcontext()
    )

    order = sorted(range(len(prompts_ids)), key=lambda i: (len(prompts_ids[i]), i))
    out: list[dict[str, Any]] = [{}] * len(prompts_ids)
    try:
        with w_ctx, a_ctx:
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


# ------------------------------------------------------------------ Batched SFT Trainers
def _make_right_padded_sft_batch(
    samples: list[dict[str, Any]],
    *,
    pad_id: int,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = len(samples)
    max_len = max(len(sp["ids"]) for sp in samples)
    inp = torch.full((bs, max_len), int(pad_id), dtype=torch.long, device=dev)
    mask = torch.zeros((bs, max_len), dtype=torch.long, device=dev)
    labels = torch.full((bs, max_len), -100, dtype=torch.long, device=dev)
    for row_i, sp in enumerate(samples):
        ids = sp["ids"]
        plen = int(sp["prompt_len"])
        L = len(ids)
        inp[row_i, :L] = torch.tensor(ids, dtype=torch.long, device=dev)
        mask[row_i, :L] = 1
        if plen < L:
            labels[row_i, plen:L] = inp[row_i, plen:L]
    pos_ids = (mask.cumsum(dim=-1) - 1).clamp_min(0)
    return inp, mask, pos_ids, labels


def _sft_batch_loss(
    model: nn.Module,
    inp: torch.Tensor,
    mask: torch.Tensor,
    pos_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    kw: dict[str, Any] = {"input_ids": inp, "attention_mask": mask, "use_cache": False}
    try:
        out = model(**kw, position_ids=pos_ids)
    except TypeError:
        out = model(**kw)
    logits = out.logits[:, :-1, :].float()
    tgt = labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)


def _train_additive_arm(
    model: nn.Module,
    train_samples: list[dict[str, Any]],
    *,
    pad_id: int,
    insert_layer: int,
    n_blocks: int,
    n_loops_choices: tuple[int, ...],
    max_steps: int,
    d_loop_key: int,
    total_steps: int,
    batch_size_train: int,
    grad_accum: int,
    lr_main: float,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.10,
    anchor_ids: list[torch.Tensor] | None = None,
    base_anchor_logp: list[torch.Tensor | None] | None = None,
    anchor_kl_weight: float = 0.02,
    healthy_ids: list[torch.Tensor] | None = None,
    base_healthy_logp: list[torch.Tensor | None] | None = None,
    base_healthy_nll: float = 0.0,
    grad_checkpointing: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Train ``AdditiveLoopedMoI`` (either fixed T=1 or variable T~choices) on frozen backbone."""
    assert len(train_samples) > 0 and total_steps >= 1 and grad_accum >= 1
    dev = next(model.parameters()).device
    orig_req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in orig_req:
        p.requires_grad_(False)
    was_training = bool(model.training)

    add_mod = AdditiveLoopedMoI.from_model(
        model,
        insert_layer=int(insert_layer),
        n_blocks=int(n_blocks),
        n_loops=max(n_loops_choices),
        max_steps=int(max_steps),
        d_loop_key=int(d_loop_key),
    ).to(dev)
    for p in add_mod.parameters():
        p.requires_grad_(True)

    n_params = add_mod.num_parameters()
    payload_mb = add_mod.payload_mb()

    emb_handles = _enable_emb_grad_hook(model)
    step_losses: list[float] = []
    anc_kls: list[float] = []
    loop_counts_seen: dict[int, int] = {int(t): 0 for t in n_loops_choices}
    t_c0 = time.perf_counter()

    try:
        model.eval()
        add_mod.train()

        block_params = list(add_mod.blocks.parameters())
        gate_params = [
            add_mod.step_embed.weight,
            add_mod.k_proj.weight,
            add_mod.q_agg,
            add_mod.step_bias,
            add_mod.out_gate,
        ]
        opt = torch.optim.AdamW([
            {"params": block_params, "lr": float(lr_main), "weight_decay": float(weight_decay)},
            {"params": gate_params, "lr": float(lr_main) * 5.0, "weight_decay": 0.0},
        ])

        n_steps = int(total_steps)
        warmup_steps = max(1, int(round(n_steps * float(warmup_frac)))) if float(warmup_frac) > 0.0 else 0

        def _lr_lambda(step_idx: int) -> float:
            if warmup_steps > 0 and step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            rem = max(1, n_steps - warmup_steps)
            prog = min(1.0, max(0.0, float(step_idx - warmup_steps) / float(rem)))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        rng = torch.Generator().manual_seed(int(seed))
        idx_queue: list[int] = []
        choice_cursor = 0

        for step_idx in range(n_steps):
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _acc in range(int(grad_accum)):
                batch_sp: list[dict[str, Any]] = []
                for _ in range(int(batch_size_train)):
                    if not idx_queue:
                        idx_queue = torch.randperm(len(train_samples), generator=rng).tolist()
                    batch_sp.append(train_samples[idx_queue.pop()])

                t_step = int(n_loops_choices[choice_cursor % len(n_loops_choices)])
                choice_cursor += 1
                loop_counts_seen[t_step] = loop_counts_seen.get(t_step, 0) + 1

                inp, mask, pos_ids, labels = _make_right_padded_sft_batch(batch_sp, pad_id=pad_id, dev=dev)
                with torch.enable_grad(), add_mod.attach(
                    model, n_loops=t_step, grad_checkpointing=grad_checkpointing, record_stats=False
                ):
                    loss = _sft_batch_loss(model, inp, mask, pos_ids, labels) / float(grad_accum)
                    loss.backward()
                    step_loss += float(loss.detach().item())
                del inp, mask, pos_ids, labels, loss

            if float(anchor_kl_weight) > 0.0 and anchor_ids and base_anchor_logp:
                anc_i = step_idx % len(anchor_ids)
                anc_t = anchor_ids[anc_i]
                lp_b_cpu = base_anchor_logp[anc_i] if anc_i < len(base_anchor_logp) else None
                if anc_t.shape[1] >= 2 and lp_b_cpu is not None:
                    anc_dev = anc_t.to(dev)
                    t_anc = int(max(n_loops_choices))
                    with torch.enable_grad(), add_mod.attach(
                        model, n_loops=t_anc, grad_checkpointing=False, record_stats=False
                    ):
                        lp_s = F.log_softmax(_tail_logits(model, anc_dev, 0)[:-1].float(), dim=-1)
                        lp_b = lp_b_cpu.to(device=dev, dtype=torch.float32)
                        kl_anc = (lp_b.exp() * (lp_b - lp_s)).sum(dim=-1).clamp_min(0.0).mean()
                        (float(anchor_kl_weight) * kl_anc).backward()
                        anc_kls.append(float(kl_anc.detach().item()))
                    del anc_dev, lp_s, lp_b, kl_anc

            torch.nn.utils.clip_grad_norm_(add_mod.parameters(), 1.0)
            opt.step()
            sched.step()
            step_losses.append(step_loss)

        add_mod.eval()
        train_time_s = round(time.perf_counter() - t_c0, 2)

        t_eval = int(max(n_loops_choices))
        with add_mod.attach(model, n_loops=t_eval, grad_checkpointing=False, record_stats=False):
            drift = (
                eval_healthy_drift(model, healthy_ids, base_healthy_logp, base_healthy_nll)
                if (healthy_ids is not None and base_healthy_logp is not None)
                else {"healthy_kl": 0.0, "delta_healthy_nll": 0.0}
            )
    finally:
        for h in emb_handles:
            h.remove()
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
        "module": add_mod,
        "num_parameters": int(n_params),
        "payload_mb": round(float(payload_mb), 4),
        "train_time_s": train_time_s,
        "step_losses_checkpoints": ckpts,
        "loop_counts_seen": loop_counts_seen,
        "mean_anchor_kl": round(sum(anc_kls) / max(1, len(anc_kls)), 6) if anc_kls else 0.0,
        "out_gate_abs_mean": round(float(add_mod.out_gate.detach().abs().mean().item()), 5),
        **drift,
    }


def _train_lora_equal_params(
    model: nn.Module,
    train_samples: list[dict[str, Any]],
    *,
    pad_id: int,
    rank: int,
    total_steps: int,
    batch_size_train: int,
    grad_accum: int,
    lr_lora: float,
    weight_decay: float = 0.05,
    lora_dropout: float = 0.05,
    warmup_frac: float = 0.10,
    anchor_ids: list[torch.Tensor] | None = None,
    base_anchor_logp: list[torch.Tensor | None] | None = None,
    anchor_kl_weight: float = 0.02,
    healthy_ids: list[torch.Tensor] | None = None,
    base_healthy_logp: list[torch.Tensor | None] | None = None,
    base_healthy_nll: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train all-layer LoRA with exact equal trainable parameter budget to ``AdditiveLoopedMoI``."""
    assert len(train_samples) > 0 and total_steps >= 1 and grad_accum >= 1
    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
    prefixes = _block_prefixes(model)
    orig_req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in orig_req:
        p.requires_grad_(False)
    was_training = bool(model.training)

    alpha_scale = 2.0 * float(rank)
    wrapped: list[tuple[nn.Module, str, nn.Linear, str, _RegLoRALinear]] = []
    for li, blk in enumerate(blocks):
        blk_pref = prefixes[li]
        for sub_nm, lin in [(nm, m) for nm, m in blk.named_modules() if isinstance(m, nn.Linear)]:
            parts = sub_nm.split(".")
            parent = blk
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            wrapper = _RegLoRALinear(
                lin, rank=int(rank), alpha_scale=alpha_scale, dropout=float(lora_dropout)
            )
            setattr(parent, attr, wrapper)
            wrapped.append((parent, attr, lin, f"{blk_pref}.{sub_nm}.weight", wrapper))

    lora_params = [p for *_rest, w in wrapped for p in (w.lora_A, w.lora_B)]
    n_params = sum(int(p.numel()) for p in lora_params)
    payload_mb = round((n_params * 2) / (1024.0**2), 4)

    emb_handles = _enable_emb_grad_hook(model)
    step_losses: list[float] = []
    anc_kls: list[float] = []
    t_c0 = time.perf_counter()

    try:
        model.eval()
        for *_rest, w in wrapped:
            w.train()

        opt = torch.optim.AdamW(lora_params, lr=float(lr_lora), weight_decay=float(weight_decay))
        n_steps = int(total_steps)
        warmup_steps = max(1, int(round(n_steps * float(warmup_frac)))) if float(warmup_frac) > 0.0 else 0

        def _lr_lambda(step_idx: int) -> float:
            if warmup_steps > 0 and step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            rem = max(1, n_steps - warmup_steps)
            prog = min(1.0, max(0.0, float(step_idx - warmup_steps) / float(rem)))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        rng = torch.Generator().manual_seed(int(seed))
        idx_queue: list[int] = []

        for step_idx in range(n_steps):
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _acc in range(int(grad_accum)):
                batch_sp: list[dict[str, Any]] = []
                for _ in range(int(batch_size_train)):
                    if not idx_queue:
                        idx_queue = torch.randperm(len(train_samples), generator=rng).tolist()
                    batch_sp.append(train_samples[idx_queue.pop()])
                inp, mask, pos_ids, labels = _make_right_padded_sft_batch(batch_sp, pad_id=pad_id, dev=dev)
                with torch.enable_grad():
                    loss = _sft_batch_loss(model, inp, mask, pos_ids, labels) / float(grad_accum)
                    loss.backward()
                    step_loss += float(loss.detach().item())
                del inp, mask, pos_ids, labels, loss

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

            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            sched.step()
            step_losses.append(step_loss)

        for *_rest, w in wrapped:
            w.eval()
        train_time_s = round(time.perf_counter() - t_c0, 2)
        drift = (
            eval_healthy_drift(model, healthy_ids, base_healthy_logp, base_healthy_nll)
            if (healthy_ids is not None and base_healthy_logp is not None)
            else {"healthy_kl": 0.0, "delta_healthy_nll": 0.0}
        )
        merged_weights = {full_nm: w.merged_weight_cpu() for _, _, _, full_nm, w in wrapped}
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
        "rank": int(rank),
        "num_parameters": int(n_params),
        "payload_mb": payload_mb,
        "train_time_s": train_time_s,
        "step_losses_checkpoints": ckpts,
        "mean_anchor_kl": round(sum(anc_kls) / max(1, len(anc_kls)), 6) if anc_kls else 0.0,
        **drift,
    }


# ------------------------------------------------------------------ Evaluation & Reporting
def _grade_gsm8k_outputs(
    gold_answers: list[str],
    raw_outs: list[dict[str, Any]],
    tok: Any,
    eos_set: set[int],
) -> tuple[list[bool], list[int]]:
    flags: list[bool] = []
    tok_counts: list[int] = []
    for gold, raw in zip(gold_answers, raw_outs):
        body = lr.strip_eos(raw["token_ids"], eos_set)
        txt = tok.decode(body, skip_special_tokens=True).strip()
        pred = ec.extract_gsm8k_answer(txt, is_gold=False)
        flags.append(pred is not None and pred == gold)
        tok_counts.append(len(raw["token_ids"]))
    return flags, tok_counts


def _grade_math500_outputs(
    items: list[dict[str, Any]],
    raw_outs: list[dict[str, Any]],
    tok: Any,
    eos_set: set[int],
    max_tokens: int,
) -> tuple[list[bool], list[int]]:
    flags: list[bool] = []
    tok_counts: list[int] = []
    for it, raw in zip(items, raw_outs):
        g = dm.grade_math(raw["token_ids"], raw["finish_reason"], str(it["answer"]), tok, eos_set, int(max_tokens))
        flags.append(bool(g["correct"]))
        tok_counts.append(int(g["n_tokens"]))
    return flags, tok_counts


def _paired_bool_bootstrap(
    flags_a: list[bool],
    flags_b: list[bool],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    rows_a = [{"strict_correct": bool(x)} for x in flags_a]
    rows_b = [{"strict_correct": bool(x)} for x in flags_b]
    return _paired_bootstrap(rows_a, rows_b, n_boot=n_boot, seed=seed)


def _routed_accuracy(arm_flags: list[bool], base_flags: list[bool], fire_flags: list[bool]) -> float:
    n = len(arm_flags)
    corr = sum(1 for a, b, f in zip(arm_flags, base_flags, fire_flags) if (a if f else b))
    return round(corr / max(1, n), 4)


def _summarize_pilot(payload: dict[str, Any]) -> None:
    print("\n=== ADDITIVE LOOPED MOI PILOT SUMMARY ===", flush=True)
    hdr = (
        f"{'Arm':24s} | {'T':3s} | {'Params':9s} | {'GSM8K':7s} | {'GSM_MoI':7s} | "
        f"{'MATH':7s} | {'MATH_MoI':8s} | {'MMLU_Δ':7s} | {'ProseKL(r/u)':15s}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for arm_name, rec in payload["arms"].items():
        m_del = rec.get("mmlu_non_math_delta_pts", 0.0)
        p_r = rec.get("routed_prose_kl", 0.0)
        p_u = rec.get("unrouted_prose_kl", 0.0)
        print(
            f"{arm_name:24s} | {str(rec['n_loops']):3s} | {rec['payload_mb']:6.2f} MB | "
            f"{rec['gsm8k_acc']*100:6.2f}% | {rec['gsm8k_routed_acc']*100:6.2f}% | "
            f"{rec['math500_acc']*100:6.2f}% | {rec['math500_routed_acc']*100:7.2f}% | "
            f"{m_del:+6.2f}% | {p_r:.4f} / {p_u:.4f}",
            flush=True,
        )
    print("\n=== PAIRED BOOTSTRAP 95% CIs (GSM8K & MATH-500) ===", flush=True)
    for k, v in payload.get("paired_bootstrap", {}).items():
        print(
            f"  {k:44s}: diff={v['diff']*100:+5.2f} pts "
            f"CI95=[{v['ci95'][0]*100:+5.2f}, {v['ci95'][1]*100:+5.2f}] (w={v['wins']}/l={v['losses']})",
            flush=True,
        )


# ------------------------------------------------------------------ CPU Smoke Test
class _SmokeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = -1

    def convert_tokens_to_ids(self, _tok: str) -> int:
        return -1


def _load_smoke_qwen2_fn():
    import importlib.util

    test_path = ROOT / "tests" / "test_additive_loop.py"
    spec = importlib.util.spec_from_file_location("_wsi_test_addloop_smoke", test_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._make_tiny_qwen2


def run_smoke() -> dict[str, Any]:
    """Fast CPU end-to-end verification of AdditiveLoopedMoI training, LoRA equal-param match, generate & router."""
    make_qwen2 = _load_smoke_qwen2_fn()
    model = make_qwen2(seed=19)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _, blocks = find_blocks(model)
    insert_layer = len(blocks) // 2

    samples = [
        {"ids": [1, 3, 5, 10, 15, 20, 25, 2], "prompt_len": 3},
        {"ids": [4, 6, 8, 12, 18, 24, 30, 2], "prompt_len": 3},
        {"ids": [7, 9, 11, 14, 21, 28, 35, 2], "prompt_len": 3},
        {"ids": [1, 2, 4, 8, 16, 32, 40, 2], "prompt_len": 3},
    ]
    h_ids = [torch.tensor([[1, 2, 3, 4, 2]], dtype=torch.long)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    add_1x = _train_additive_arm(
        model, samples, pad_id=0, insert_layer=insert_layer, n_blocks=1,
        n_loops_choices=(1,), max_steps=6, d_loop_key=16,
        total_steps=4, batch_size_train=2, grad_accum=1, lr_main=2e-2,
        anchor_ids=h_ids, base_anchor_logp=base_lp, anchor_kl_weight=0.02,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll,
        grad_checkpointing=True, seed=42,
    )
    add_loop = _train_additive_arm(
        model, samples, pad_id=0, insert_layer=insert_layer, n_blocks=1,
        n_loops_choices=(1, 2, 3), max_steps=6, d_loop_key=16,
        total_steps=6, batch_size_train=2, grad_accum=1, lr_main=2e-2,
        anchor_ids=h_ids, base_anchor_logp=base_lp, anchor_kl_weight=0.02,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll,
        grad_checkpointing=True, seed=42,
    )
    assert add_1x["step_losses_checkpoints"][-1] < add_1x["step_losses_checkpoints"][0]
    assert add_loop["step_losses_checkpoints"][-1] < add_loop["step_losses_checkpoints"][0]
    assert set(add_loop["loop_counts_seen"].keys()) == {1, 2, 3}
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    best_rank, lora_cnt = _select_equal_param_lora_rank(model, add_loop["num_parameters"])
    lora_arm = _train_lora_equal_params(
        model, samples, pad_id=0, rank=best_rank,
        total_steps=4, batch_size_train=2, grad_accum=1, lr_lora=2e-2,
        anchor_ids=h_ids, base_anchor_logp=base_lp, anchor_kl_weight=0.02,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll,
        seed=42,
    )
    assert lora_arm["step_losses_checkpoints"][-1] < lora_arm["step_losses_checkpoints"][0]
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    # Verify save/load + batched eval via model.generate + telemetry across T=1..4
    with tempfile.TemporaryDirectory() as td:
        ckpt_p = Path(td) / "add_loop.safetensors"
        add_loop["module"].save(ckpt_p)
        reloaded = type(add_loop["module"]).load(ckpt_p, template_model=model, verify=True)

        tok_sm = _SmokeTokenizer()
        prompts = [[1, 3, 5], [4, 6, 8, 12], [7, 9, 11, 13]]
        out_t2 = _hf_generate_batched_addloop(
            model, tok_sm, prompts, max_new_tokens=4, batch_size=2,
            add_mod=reloaded, n_loops=2, record_stats=True,
        )
        tel_t2 = reloaded.telemetry_summary()
        out_t4 = _hf_generate_batched_addloop(
            model, tok_sm, prompts, max_new_tokens=4, batch_size=2,
            add_mod=reloaded, n_loops=4, record_stats=True,
        )
        tel_t4 = reloaded.telemetry_summary()
        assert len(out_t2) == 3 and len(out_t4) == 3
        assert len(tel_t2["alpha_mean"]) == 3 and len(tel_t4["alpha_mean"]) == 5
        assert abs(sum(tel_t4["alpha_mean"]) - 1.0) < 1e-5

    # Verify binary MoI Ridge router + _paired_bootstrap
    X_r = _extract_prompt_hidden_vecs(model, [[1, 3, 5], [2, 4, 6], [10, 11, 12], [13, 14, 15]], insert_layer)
    W_r, b_r = fit_ridge_router(X_r, ["math", "math", "passthrough", "passthrough"], ADDITIVE_ROUTER_CLASSES)
    preds_r = predict_ridge_router(X_r, W_r, b_r, ADDITIVE_ROUTER_CLASSES)
    assert all(p in ADDITIVE_ROUTER_CLASSES for p in preds_r)
    ci_check = _paired_bool_bootstrap([True, True, False, True], [False, True, False, False], n_boot=200, seed=42)
    assert abs(ci_check["diff"] - 0.5) < 1e-6

    return {
        "mode": "smoke",
        "passed": True,
        "insert_layer": insert_layer,
        "add_loop_params": add_loop["num_parameters"],
        "matched_lora_rank": best_rank,
        "matched_lora_params": lora_cnt,
        "add_1x_loss": add_1x["step_losses_checkpoints"],
        "add_loop_loss": add_loop["step_losses_checkpoints"],
        "lora_loss": lora_arm["step_losses_checkpoints"],
        "telemetry_t2": tel_t2,
        "telemetry_t4": tel_t4,
    }


# ------------------------------------------------------------------ Full GPU Pilot
def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        dtype = (
            torch.bfloat16
            if (dev.type == "cuda" and torch.cuda.is_bf16_supported())
            else (torch.float16 if dev.type == "cuda" else torch.float32)
        )
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    print(f"[addloop] Loading tokenizer & model {args.model} ({dtype}) on {dev}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = _model_load(args.model, dtype).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    insert_layer = (n_layers // 2) if int(args.insert_layer) < 0 else int(args.insert_layer)
    eos_set = set(lr.eos_ids(tok))
    raw_pad = getattr(tok, "pad_token_id", None)
    pad_id = (int(next(iter(eos_set))) if eos_set else 0) if raw_pad is None else int(raw_pad)

    # 1. Load GSM8K train & test, MATH-500 OOD test, MMLU test + router train, Prose controls
    print("[addloop] Loading datasets (GSM8K, MATH-500, MMLU)...", flush=True)
    gsm_ds = load_dataset("openai/gsm8k", "main")
    rng_py = random.Random(int(args.seed))

    gsm_train_raw = list(gsm_ds["train"])
    rng_py.shuffle(gsm_train_raw)
    train_samples: list[dict[str, Any]] = []
    gsm_router_prompts: list[list[int]] = []
    for row in gsm_train_raw:
        if len(train_samples) >= int(args.train_n):
            break
        sp = _format_gsm8k_sft_pair(tok, str(row["question"]), str(row["answer"]), int(args.max_seq_len))
        if sp is not None:
            train_samples.append(sp)
            if len(gsm_router_prompts) < 250:
                gsm_router_prompts.append(
                    _format_messages(tok, [{"role": "user", "content": format_question(str(row["question"]))}],
                                     add_generation_prompt=True, enable_thinking=False)
                )

    gsm_test_raw = list(gsm_ds["test"])[: int(args.gsm8k_test_n)]
    gsm_test_pids = [
        _format_messages(tok, [{"role": "user", "content": format_question(str(r["question"]))}],
                         add_generation_prompt=True, enable_thinking=False)
        for r in gsm_test_raw
    ]
    gsm_test_gold = [ec.extract_gsm8k_answer(str(r["answer"]), is_gold=True) or "" for r in gsm_test_raw]

    _, math_test = load_math500_split(
        n_train=max(1, 500 - int(args.math_test_n)), n_test=int(args.math_test_n), seed=int(args.seed)
    )
    math_test_pids = [format_math500_prompt_ids(tok, r["problem"]) for r in math_test]

    mmlu_test, mmlu_router_train = load_mmlu_split(
        n_test=int(args.mmlu_test_n), n_router_train=150, seed=int(args.seed)
    )
    mmlu_test_pids = [format_mmlu_prompt_ids(tok, it) for it in mmlu_test]
    mmlu_r_nonmath = [it for it in mmlu_router_train if not it.get("is_math_subject", False)][:120]
    mmlu_r_pids = [format_mmlu_prompt_ids(tok, it) for it in mmlu_r_nonmath]
    prose_r_pids = [
        _format_messages(tok, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
        for p in PROSE_ROUTER_TRAIN
    ]
    prose_held = list(PROSE_HELDOUT[:16])

    print(
        f"[addloop] Data ready: train={len(train_samples)} | gsm8k_test={len(gsm_test_raw)} | "
        f"math500_test={len(math_test)} | mmlu_test={len(mmlu_test)} | insert_layer={insert_layer}/{n_layers}",
        flush=True,
    )

    # 2. MoI Middle-Layer Router + Base MMLU + Base Prose Logprobs
    r_train_pids = gsm_router_prompts + mmlu_r_pids + prose_r_pids
    r_train_labels = (
        ["math"] * len(gsm_router_prompts)
        + ["passthrough"] * len(mmlu_r_pids)
        + ["passthrough"] * len(prose_r_pids)
    )
    W_r, b_r = fit_ridge_router(
        _extract_prompt_hidden_vecs(model, r_train_pids, insert_layer),
        r_train_labels,
        ADDITIVE_ROUTER_CLASSES,
        lam=1.0,
    )
    routed_gsm_fire = [p == "math" for p in predict_ridge_router(
        _extract_prompt_hidden_vecs(model, gsm_test_pids, insert_layer), W_r, b_r, ADDITIVE_ROUTER_CLASSES
    )]
    routed_math_fire = [p == "math" for p in predict_ridge_router(
        _extract_prompt_hidden_vecs(model, math_test_pids, insert_layer), W_r, b_r, ADDITIVE_ROUTER_CLASSES
    )]
    routed_mmlu_fire = [p == "math" for p in predict_ridge_router(
        _extract_prompt_hidden_vecs(model, mmlu_test_pids, insert_layer), W_r, b_r, ADDITIVE_ROUTER_CLASSES
    )]
    prose_held_pids = [
        _format_messages(tok, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
        for p in prose_held
    ]
    routed_prose_fire = [p == "math" for p in predict_ridge_router(
        _extract_prompt_hidden_vecs(model, prose_held_pids, insert_layer), W_r, b_r, ADDITIVE_ROUTER_CLASSES
    )]

    router_report = {
        "router_layer": insert_layer,
        "gsm8k_fire_rate": round(sum(routed_gsm_fire) / max(1, len(routed_gsm_fire)), 4),
        "math500_fire_rate": round(sum(routed_math_fire) / max(1, len(routed_math_fire)), 4),
        "mmlu_fire_rate": round(sum(routed_mmlu_fire) / max(1, len(routed_mmlu_fire)), 4),
        "mmlu_non_math_fire_rate": round(
            sum(f for it, f in zip(mmlu_test, routed_mmlu_fire) if not it.get("is_math_subject", False))
            / max(1, sum(1 for it in mmlu_test if not it.get("is_math_subject", False))),
            4,
        ),
        "prose_fpr": round(sum(routed_prose_fire) / max(1, len(routed_prose_fire)), 4),
    }
    print(f"[addloop] Router report: {router_report}", flush=True)

    letter_ids = ec._resolve_mmlu_letter_ids(tok)
    base_mmlu_raw = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)
    base_mmlu_sum = _summarize_mmlu(mmlu_test, base_mmlu_raw, seed=int(args.seed))
    base_mmlu_nonmath_acc = float(base_mmlu_sum["non_math"]["accuracy"])

    def _compute_logp(prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor | None], float]:
        seqs = _to_seq_tensors(
            [
                _format_messages(
                    tok,
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
                logits = _tail_logits(model, ids.to(dev), 0)[:-1].float()
                lp = F.log_softmax(logits, dim=-1)
                nll_s += float(F.nll_loss(lp, ids[0, 1:].to(dev), reduction="sum"))
                cnt += int(ids.shape[1] - 1)
                logps.append(lp.to(dtype=torch.bfloat16, device="cpu"))
                del logits, lp
        return seqs, logps, nll_s / max(1, cnt)

    prose_full_ids, base_prose_logp, base_prose_nll = _compute_logp(prose_held)
    anchor_full_ids, base_anchor_logp, _ = _compute_logp(list(PROSE_ROUTER_TRAIN))

    # 3. Train additive_1x, additive_loop, and equal-parameter LoRA
    eff_bs = max(1, int(args.batch_size_train) * int(args.grad_accum))
    steps_per_epoch = max(1, math.ceil(len(train_samples) / float(eff_bs)))
    total_steps = max(1, int(args.epochs) * steps_per_epoch)
    print(f"[addloop] Training budget: {len(train_samples)} samples, eff_bs={eff_bs}, total_steps={total_steps}", flush=True)

    ckpt_root = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    dir_ctx = nullcontext(str(ckpt_root)) if ckpt_root is not None else tempfile.TemporaryDirectory()

    with dir_ctx as work_str:
        work_dir = Path(work_str)

        print("[addloop] Training additive_1x (T=1 fixed)...", flush=True)
        res_1x = _train_additive_arm(
            model, train_samples, pad_id=pad_id, insert_layer=insert_layer, n_blocks=int(args.n_blocks),
            n_loops_choices=(1,), max_steps=int(args.max_steps), d_loop_key=int(args.d_loop_key),
            total_steps=total_steps, batch_size_train=int(args.batch_size_train), grad_accum=int(args.grad_accum),
            lr_main=float(args.lr), anchor_ids=anchor_full_ids, base_anchor_logp=base_anchor_logp,
            healthy_ids=prose_full_ids, base_healthy_logp=base_prose_logp, base_healthy_nll=base_prose_nll,
            grad_checkpointing=True, seed=int(args.seed),
        )
        res_1x["module"].save(work_dir / "additive_1x.safetensors")
        print(
            f"[addloop] additive_1x done ({res_1x['train_time_s']}s) | params={res_1x['num_parameters']} | "
            f"loss={res_1x['step_losses_checkpoints'][0]}->{res_1x['step_losses_checkpoints'][-1]} | "
            f"|gate|={res_1x['out_gate_abs_mean']} | prose_kl={res_1x['healthy_kl']}",
            flush=True,
        )

        print("[addloop] Training additive_loop (variable T in {1,2,3})...", flush=True)
        res_loop = _train_additive_arm(
            model, train_samples, pad_id=pad_id, insert_layer=insert_layer, n_blocks=int(args.n_blocks),
            n_loops_choices=(1, 2, 3), max_steps=int(args.max_steps), d_loop_key=int(args.d_loop_key),
            total_steps=total_steps, batch_size_train=int(args.batch_size_train), grad_accum=int(args.grad_accum),
            lr_main=float(args.lr), anchor_ids=anchor_full_ids, base_anchor_logp=base_anchor_logp,
            healthy_ids=prose_full_ids, base_healthy_logp=base_prose_logp, base_healthy_nll=base_prose_nll,
            grad_checkpointing=True, seed=int(args.seed) + 17,
        )
        res_loop["module"].save(work_dir / "additive_loop.safetensors")
        print(
            f"[addloop] additive_loop done ({res_loop['train_time_s']}s) | params={res_loop['num_parameters']} | "
            f"loss={res_loop['step_losses_checkpoints'][0]}->{res_loop['step_losses_checkpoints'][-1]} | "
            f"|gate|={res_loop['out_gate_abs_mean']} | prose_kl={res_loop['healthy_kl']}",
            flush=True,
        )

        lora_rank, lora_cnt = _select_equal_param_lora_rank(model, res_loop["num_parameters"])
        print(f"[addloop] Training lora_equal_params (all layers rank={lora_rank}, params={lora_cnt})...", flush=True)
        res_lora = _train_lora_equal_params(
            model, train_samples, pad_id=pad_id, rank=lora_rank,
            total_steps=total_steps, batch_size_train=int(args.batch_size_train), grad_accum=int(args.grad_accum),
            lr_lora=float(args.lr_lora), anchor_ids=anchor_full_ids, base_anchor_logp=base_anchor_logp,
            healthy_ids=prose_full_ids, base_healthy_logp=base_prose_logp, base_healthy_nll=base_prose_nll,
            seed=int(args.seed) + 33,
        )
        print(
            f"[addloop] lora_equal_params done ({res_lora['train_time_s']}s) | params={res_lora['num_parameters']} | "
            f"loss={res_lora['step_losses_checkpoints'][0]}->{res_lora['step_losses_checkpoints'][-1]} | "
            f"prose_kl={res_lora['healthy_kl']}",
            flush=True,
        )

        # 4. Shared Left-Padded Batched HF Evaluation
        all_eval_pids = gsm_test_pids + math_test_pids
        n_gsm = len(gsm_test_pids)

        def _eval_arm(
            arm_name: str,
            *,
            n_loops: int,
            payload_mb: float,
            n_params: int,
            unrouted_prose_kl: float,
            add_mod: AdditiveLoopedMoI | None = None,
            param_updates: dict[str, torch.Tensor] | None = None,
            record_stats: bool = False,
        ) -> tuple[dict[str, Any], list[bool], list[bool]]:
            t_e0 = time.perf_counter()
            raw_outs = _hf_generate_batched_addloop(
                model, tok, all_eval_pids, max_new_tokens=int(args.max_new_tokens),
                batch_size=int(args.batch_size_eval), base_seed=int(args.seed),
                add_mod=add_mod, n_loops=n_loops if add_mod is not None else None,
                param_updates=param_updates, record_stats=record_stats,
            )
            tel = add_mod.telemetry_summary() if (add_mod is not None and record_stats) else None
            gsm_flags, gsm_toks = _grade_gsm8k_outputs(gsm_test_gold, raw_outs[:n_gsm], tok, eos_set)
            math_flags, math_toks = _grade_math500_outputs(math_test, raw_outs[n_gsm:], tok, eos_set, int(args.max_new_tokens))

            # Score MMLU on items fired by the MoI router (or 0 if 0 fired)
            fired_mmlu_idxs = [i for i, f in enumerate(routed_mmlu_fire) if f]
            if fired_mmlu_idxs and (add_mod is not None or param_updates is not None):
                sub_pids = [mmlu_test_pids[i] for i in fired_mmlu_idxs]
                sub_items = [mmlu_test[i] for i in fired_mmlu_idxs]
                w_c = _temp_apply_weights(model, param_updates) if param_updates else nullcontext()
                a_c = (
                    add_mod.attach(model, n_loops=n_loops, grad_checkpointing=False, record_stats=False)
                    if add_mod is not None
                    else nullcontext()
                )
                with w_c, a_c:
                    sub_sc = _score_mmlu_fast(model, sub_pids, sub_items, letter_ids, dev)
                probs = base_mmlu_raw["probs"].clone()
                preds = list(base_mmlu_raw["preds"])
                correct = list(base_mmlu_raw["correct"])
                masses = list(base_mmlu_raw["choice_masses"])
                for k, orig_i in enumerate(fired_mmlu_idxs):
                    probs[orig_i] = sub_sc["probs"][k]
                    preds[orig_i] = sub_sc["preds"][k]
                    correct[orig_i] = sub_sc["correct"][k]
                    masses[orig_i] = sub_sc["choice_masses"][k]
                mmlu_routed_sc = {"probs": probs, "preds": preds, "correct": correct, "choice_masses": masses}
            else:
                mmlu_routed_sc = base_mmlu_raw

            mmlu_sum = _summarize_mmlu(
                mmlu_test, mmlu_routed_sc, base_score=base_mmlu_raw,
                fire_flags=routed_mmlu_fire if (add_mod is not None or param_updates is not None) else [False] * len(mmlu_test),
                seed=int(args.seed),
            )
            nonmath_acc = float(mmlu_sum["non_math"]["accuracy"])

            # Routed prose KL: only prompts where router fired contribute drift (0.0 when prose_fpr == 0)
            fired_prose_idxs = [i for i, f in enumerate(routed_prose_fire) if f]
            routed_prose_kl = 0.0
            if fired_prose_idxs and (add_mod is not None or param_updates is not None):
                w_c = _temp_apply_weights(model, param_updates) if param_updates else nullcontext()
                a_c = (
                    add_mod.attach(model, n_loops=n_loops, grad_checkpointing=False, record_stats=False)
                    if add_mod is not None
                    else nullcontext()
                )
                with w_c, a_c:
                    dr_p = eval_healthy_drift(
                        model,
                        [prose_full_ids[i] for i in fired_prose_idxs],
                        [base_prose_logp[i] for i in fired_prose_idxs],
                        base_prose_nll,
                    )
                routed_prose_kl = float(dr_p["healthy_kl"]) * (len(fired_prose_idxs) / max(1, len(prose_held)))

            rec = {
                "n_loops": int(n_loops),
                "num_parameters": int(n_params),
                "payload_mb": round(float(payload_mb), 4),
                "gsm8k_acc": round(sum(gsm_flags) / max(1, len(gsm_flags)), 4),
                "gsm8k_correct": int(sum(gsm_flags)),
                "gsm8k_total": len(gsm_flags),
                "gsm8k_mean_tokens": round(sum(gsm_toks) / max(1, len(gsm_toks)), 1),
                "math500_acc": round(sum(math_flags) / max(1, len(math_flags)), 4),
                "math500_correct": int(sum(math_flags)),
                "math500_total": len(math_flags),
                "math500_mean_tokens": round(sum(math_toks) / max(1, len(math_toks)), 1),
                "gsm8k_routed_acc": 0.0,   # filled right after base_gsm_flags exist
                "math500_routed_acc": 0.0,
                "mmlu_non_math_acc": round(nonmath_acc, 4),
                "mmlu_non_math_delta_pts": round((nonmath_acc - base_mmlu_nonmath_acc) * 100.0, 2),
                "routed_prose_kl": round(routed_prose_kl, 6),
                "unrouted_prose_kl": round(float(unrouted_prose_kl), 6),
                "eval_time_s": round(time.perf_counter() - t_e0, 2),
                "telemetry": tel,
            }
            return rec, gsm_flags, math_flags

        arms_out: dict[str, dict[str, Any]] = {}
        gsm_flags_by_arm: dict[str, list[bool]] = {}
        math_flags_by_arm: dict[str, list[bool]] = {}

        eval_specs: list[tuple[str, dict[str, Any]]] = [
            ("base", {"n_loops": 0, "payload_mb": 0.0, "n_params": 0, "unrouted_prose_kl": 0.0}),
            ("additive_1x", {
                "n_loops": 1, "payload_mb": res_1x["payload_mb"], "n_params": res_1x["num_parameters"],
                "unrouted_prose_kl": res_1x["healthy_kl"], "add_mod": res_1x["module"], "record_stats": True,
            }),
            ("lora_equal_params", {
                "n_loops": 0, "payload_mb": res_lora["payload_mb"], "n_params": res_lora["num_parameters"],
                "unrouted_prose_kl": res_lora["healthy_kl"], "param_updates": res_lora["weights"],
            }),
        ]
        for t_eval in (1, 2, 3, 4, 6):
            eval_specs.append((f"additive_loop_T{t_eval}", {
                "n_loops": t_eval, "payload_mb": res_loop["payload_mb"], "n_params": res_loop["num_parameters"],
                "unrouted_prose_kl": res_loop["healthy_kl"], "add_mod": res_loop["module"], "record_stats": True,
            }))

        for arm_nm, kw in eval_specs:
            print(f"[addloop] Evaluating {arm_nm}...", flush=True)
            rec, gf, mf = _eval_arm(arm_nm, **kw)
            arms_out[arm_nm] = rec
            gsm_flags_by_arm[arm_nm] = gf
            math_flags_by_arm[arm_nm] = mf

        base_gf = gsm_flags_by_arm["base"]
        base_mf = math_flags_by_arm["base"]
        for arm_nm, rec in arms_out.items():
            if arm_nm == "base":
                rec["gsm8k_routed_acc"] = rec["gsm8k_acc"]
                rec["math500_routed_acc"] = rec["math500_acc"]
            else:
                rec["gsm8k_routed_acc"] = _routed_accuracy(gsm_flags_by_arm[arm_nm], base_gf, routed_gsm_fire)
                rec["math500_routed_acc"] = _routed_accuracy(math_flags_by_arm[arm_nm], base_mf, routed_math_fire)

        paired_boot: dict[str, Any] = {}
        pairs = [
            ("additive_loop_T3", "base"),
            ("additive_loop_T3", "additive_1x"),
            ("additive_loop_T3", "additive_loop_T1"),
            ("additive_loop_T4", "additive_loop_T3"),
            ("additive_loop_T6", "additive_loop_T3"),
            ("additive_loop_T3", "lora_equal_params"),
            ("additive_1x", "base"),
            ("lora_equal_params", "base"),
        ]
        for a, b in pairs:
            paired_boot[f"gsm8k: {a} - {b}"] = _paired_bool_bootstrap(gsm_flags_by_arm[a], gsm_flags_by_arm[b], seed=int(args.seed))
            paired_boot[f"math500: {a} - {b}"] = _paired_bool_bootstrap(math_flags_by_arm[a], math_flags_by_arm[b], seed=int(args.seed))

    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    payload = {
        "experiment": "additive_loop_moi_pilot",
        "model": args.model,
        "dtype": str(dtype),
        "device": str(dev),
        "insert_layer": insert_layer,
        "n_layers": n_layers,
        "n_blocks": int(args.n_blocks),
        "router_report": router_report,
        "train_summaries": {
            "additive_1x": {k: v for k, v in res_1x.items() if k != "module"},
            "additive_loop": {k: v for k, v in res_loop.items() if k != "module"},
            "lora_equal_params": {k: v for k, v in res_lora.items() if k != "weights"},
        },
        "arms": arms_out,
        "paired_bootstrap": paired_boot,
        "total_time_s": round(time.perf_counter() - t0, 2),
    }
    _atomic_write_json(Path(args.out), payload)
    _summarize_pilot(payload)
    print(f"\n[addloop] Saved -> {args.out} (total {payload['total_time_s']}s)", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Additive Looped MoI Pilot Benchmark")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--gsm8k-test-n", type=int, default=300)
    ap.add_argument("--math-test-n", type=int, default=150)
    ap.add_argument("--mmlu-test-n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size-train", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--batch-size-eval", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--n-blocks", type=int, default=1)
    ap.add_argument("--insert-layer", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--d-loop-key", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--out", default="additive_loop_moi.json")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        rep = run_smoke()
        print(json.dumps(rep, indent=2), flush=True)
        return
    run_real(args)


if __name__ == "__main__":
    main()
