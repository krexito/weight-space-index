"""Verifiable On-Policy GRPO on Additive Looped MoI (`Qwen3-0.6B`).

Starts from exact base behavior (`out_gate = 0` -> bit-exact base identity at step 0)
and trains ``AdditiveLoopedMoI`` on-policy using verifiable exact-match rewards on
``GSM8K train`` + disjoint ``MATH-500 train`` under a calibrated 3-class middle-layer
MoI router (``math`` / ``pass_mmlu`` / ``pass_prose``) fixing Pilot 1's prose FPR.

Arms compared under identical left-padded batched HF generation (`enable_thinking=False`):
  1. `base`                  : Unmodified student (T=0)
  2. `additive_1x_grpo`      : 1 cloned intermediate block trained with GRPO at fixed T=1
  3. `additive_loop_grpo_T*` : Same cloned block trained with GRPO over variable T in {1,2,3},
                               evaluated at T=1, 2, 3, and zero-shot T=4, 6

CPU smoke test::
    python scripts/experiment_additive_loop_grpo.py --smoke

GPU pilot (~30-35 min on RTX 4090)::
    python scripts/experiment_additive_loop_grpo.py --model Qwen/Qwen3-0.6B --out additive_loop_grpo.json
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

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
from experiment_additive_loop_moi import (  # noqa: E402
    _SmokeTokenizer,
    _grade_gsm8k_outputs,
    _grade_math500_outputs,
    _hf_generate_batched_addloop,
    _load_smoke_qwen2_fn,
    _model_load,
    _paired_bool_bootstrap,
    _routed_accuracy,
)
from experiment_pocket_moe_ceiling import (  # noqa: E402
    _enable_emb_grad_hook,
    eval_healthy_drift,
    fit_ridge_router,
    predict_ridge_router,
)
from experiment_pocket_moe_math500 import (  # noqa: E402
    PROSE_HELDOUT,
    PROSE_ROUTER_TRAIN,
    _score_mmlu_fast,
    _summarize_mmlu,
    format_math500_prompt_ids,
    format_mmlu_prompt_ids,
    load_math500_split,
    load_mmlu_split,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
)
from experiment_reasoning_distill_v2 import _format_messages, format_question  # noqa: E402
from wsi.additive_loop import AdditiveLoopedMoI  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402

BALANCED_ROUTER_CLASSES: tuple[str, str, str] = ("math", "pass_mmlu", "pass_prose")


# ------------------------------------------------------------------ Batched On-Policy Rollouts & Log-Probs
@torch.inference_mode()
def _generate_grpo_rollouts(
    model: nn.Module,
    tok: Any,
    prompts_ids: list[list[int]],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    add_mod: AdditiveLoopedMoI,
    n_loops: int,
) -> list[dict[str, Any]]:
    """Sample ``group_size`` continuations per prompt in a single left-padded HF batch."""
    dev = next(model.parameters()).device
    eos_list = lr.eos_ids(tok)
    eos_set = set(eos_list)
    raw_pad = getattr(tok, "pad_token_id", None)
    pad_id = (int(eos_list[0]) if eos_list else 0) if raw_pad is None else int(raw_pad)

    rep_pids = [p for p in prompts_ids for _ in range(int(group_size))]
    max_len = max(len(p) for p in rep_pids)
    inp = torch.full((len(rep_pids), max_len), pad_id, dtype=torch.long, device=dev)
    mask = torch.zeros((len(rep_pids), max_len), dtype=torch.long, device=dev)
    for row_i, pids in enumerate(rep_pids):
        if pids:
            pad_n = max_len - len(pids)
            inp[row_i, pad_n:] = torch.tensor(pids, dtype=torch.long, device=dev)
            mask[row_i, pad_n:] = 1

    was_training = bool(model.training)
    was_mod_training = bool(add_mod.training)
    model.eval()
    add_mod.eval()
    try:
        with add_mod.attach(model, n_loops=int(n_loops), grad_checkpointing=False, record_stats=False):
            torch.manual_seed(int(seed))
            gen_kw: dict[str, Any] = dict(
                input_ids=inp,
                attention_mask=mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                use_cache=True,
                pad_token_id=pad_id,
            )
            if eos_list:
                gen_kw["eos_token_id"] = eos_list
            gen = model.generate(**gen_kw)
    finally:
        if was_training:
            model.train()
        if was_mod_training:
            add_mod.train()

    outs: list[dict[str, Any]] = []
    for row_i in range(len(rep_pids)):
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
        if not cut_ids:
            cut_ids = [pad_id]
        fin = "stop" if hit_eos else ("length" if len(cut_ids) >= int(max_new_tokens) else "stop")
        outs.append({"token_ids": cut_ids, "finish_reason": fin})

    del inp, mask, gen
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return outs


def _rollout_token_logps(
    model: nn.Module,
    batch_items: list[tuple[list[int], list[int]]],
    *,
    pad_id: int,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token logprobs over assistant responses (`[B, L-1]`) and float mask (`[B, L-1]`)."""
    seqs = [p + r for p, r in batch_items]
    max_len = max(len(s) for s in seqs)
    bs = len(batch_items)
    inp = torch.full((bs, max_len), int(pad_id), dtype=torch.long, device=dev)
    attn = torch.zeros((bs, max_len), dtype=torch.long, device=dev)
    resp_mask = torch.zeros((bs, max_len - 1), dtype=torch.float32, device=dev)
    for row_i, (p_ids, r_ids) in enumerate(batch_items):
        full = p_ids + r_ids
        L = len(full)
        plen = len(p_ids)
        inp[row_i, :L] = torch.tensor(full, dtype=torch.long, device=dev)
        attn[row_i, :L] = 1
        lo_m = max(0, plen - 1)
        hi_m = max(lo_m, L - 1)
        resp_mask[row_i, lo_m:hi_m] = 1.0

    pos_ids = (attn.cumsum(dim=-1) - 1).clamp_min(0)
    kw: dict[str, Any] = {"input_ids": inp, "attention_mask": attn, "use_cache": False}
    try:
        out = model(**kw, position_ids=pos_ids)
    except TypeError:
        out = model(**kw)

    logits = out.logits[:, :-1, :].float()
    targets = inp[:, 1:].unsqueeze(-1)
    logps = F.log_softmax(logits, dim=-1).gather(dim=-1, index=targets).squeeze(-1)
    return logps * resp_mask, resp_mask


# ------------------------------------------------------------------ Verifiable On-Policy GRPO Trainer
def _train_additive_grpo(
    model: nn.Module,
    tok: Any,
    train_items: list[dict[str, Any]],
    grade_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], list[float]],
    *,
    pad_id: int,
    insert_layer: int,
    n_blocks: int,
    n_loops_choices: tuple[int, ...],
    max_steps: int,
    d_loop_key: int,
    grpo_steps: int = 120,
    prompts_per_step: int = 4,
    group_size: int = 4,
    micro_bs: int = 2,
    rollout_max_tokens: int = 384,
    rollout_temperature: float = 0.8,
    rollout_top_p: float = 0.95,
    rollout_top_k: int = 40,
    lr_block: float = 5e-5,
    lr_gate: float = 2.5e-3,
    kl_beta: float = 0.04,
    grad_checkpointing: bool = True,
    healthy_ids: list[torch.Tensor] | None = None,
    base_healthy_logp: list[torch.Tensor | None] | None = None,
    base_healthy_nll: float = 0.0,
    seed: int = 42,
    log_every: int = 10,
) -> dict[str, Any]:
    """Train ``AdditiveLoopedMoI`` on-policy via verifiable binary GRPO rewards."""
    assert len(train_items) > 0 and grpo_steps >= 1 and group_size >= 2
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

    emb_handles = _enable_emb_grad_hook(model)
    t_c0 = time.perf_counter()

    rewards_hist: list[float] = []
    active_frac_hist: list[float] = []
    kl_hist: list[float] = []
    loss_hist: list[float] = []
    loop_counts_seen: dict[int, int] = {int(t): 0 for t in n_loops_choices}
    opt_updates = 0
    zero_step_kl_max: float | None = None

    try:
        model.eval()
        block_params = list(add_mod.blocks.parameters())
        gate_params = [
            add_mod.step_embed.weight,
            add_mod.k_proj.weight,
            add_mod.q_agg,
            add_mod.step_bias,
            add_mod.out_gate,
        ]
        opt = torch.optim.AdamW([
            {"params": block_params, "lr": float(lr_block), "weight_decay": 0.01},
            {"params": gate_params, "lr": float(lr_gate), "weight_decay": 0.0},
        ])
        n_steps = int(grpo_steps)
        warmup = max(1, int(round(0.10 * n_steps)))

        def _lr_lambda(step_i: int) -> float:
            if step_i < warmup:
                return float(step_i + 1) / float(warmup)
            rem = max(1, n_steps - warmup)
            prog = min(1.0, max(0.0, float(step_i - warmup) / float(rem)))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        rng = random.Random(int(seed))

        for step_i in range(n_steps):
            t_step = int(n_loops_choices[step_i % len(n_loops_choices)])
            loop_counts_seen[t_step] = loop_counts_seen.get(t_step, 0) + 1

            batch_prompts = [rng.choice(train_items) for _ in range(int(prompts_per_step))]
            pids_list = [it["prompt_ids"] for it in batch_prompts]
            rollouts = _generate_grpo_rollouts(
                model,
                tok,
                pids_list,
                group_size=int(group_size),
                max_new_tokens=int(rollout_max_tokens),
                temperature=float(rollout_temperature),
                top_p=float(rollout_top_p),
                top_k=int(rollout_top_k),
                seed=int(seed) + step_i * 101,
                add_mod=add_mod,
                n_loops=t_step,
            )

            rep_items = [it for it in batch_prompts for _ in range(int(group_size))]
            rewards = grade_fn(rep_items, rollouts)
            mean_r = sum(rewards) / max(1, len(rewards))
            rewards_hist.append(mean_r)

            active_examples: list[tuple[list[int], list[int], float]] = []
            active_groups = 0
            for gi in range(int(prompts_per_step)):
                lo = gi * int(group_size)
                hi = lo + int(group_size)
                g_rew = rewards[lo:hi]
                g_mean = sum(g_rew) / float(group_size)
                var = sum((r - g_mean) ** 2 for r in g_rew) / float(group_size)
                std = math.sqrt(var)
                if std < 1e-6:
                    continue
                active_groups += 1
                for k in range(lo, hi):
                    adv = (rewards[k] - g_mean) / (std + 1e-6)
                    if abs(adv) > 1e-6:
                        active_examples.append((batch_prompts[gi]["prompt_ids"], rollouts[k]["token_ids"], float(adv)))

            active_frac_hist.append(active_groups / float(prompts_per_step))
            if not active_examples:
                sched.step()
                if log_every > 0 and (step_i + 1) % int(log_every) == 0:
                    print(
                        f"  [grpo {step_i+1:3d}/{n_steps}] T={t_step} r={mean_r:.3f} active=0/{prompts_per_step} "
                        f"|gate|={float(add_mod.out_gate.detach().abs().mean().item()):.5f}",
                        flush=True,
                    )
                continue

            opt.zero_grad(set_to_none=True)
            add_mod.train()
            n_act = len(active_examples)
            step_kl_sum = 0.0
            step_tok_sum = 0.0
            step_loss_val = 0.0

            mb = max(1, int(micro_bs))
            for mb_s in range(0, n_act, mb):
                chunk = active_examples[mb_s : mb_s + mb]
                pair_chunk = [(p, r) for p, r, _ in chunk]
                adv_t = torch.tensor([a for _, _, a in chunk], dtype=torch.float32, device=dev).unsqueeze(1)

                with torch.no_grad(), add_mod.attach(model, n_loops=0, grad_checkpointing=False):
                    ref_lp, r_mask = _rollout_token_logps(model, pair_chunk, pad_id=pad_id, dev=dev)

                with torch.enable_grad(), add_mod.attach(
                    model, n_loops=t_step, grad_checkpointing=grad_checkpointing, record_stats=False
                ):
                    new_lp, _ = _rollout_token_logps(model, pair_chunk, pad_id=pad_id, dev=dev)
                    log_ratio = (ref_lp - new_lp).clamp(-10.0, 10.0)
                    kl_tok = torch.exp(log_ratio) - log_ratio - 1.0

                    if zero_step_kl_max is None and opt_updates == 0:
                        zero_step_kl_max = float((kl_tok * r_mask).abs().max().item())

                    pg_tok = -(adv_t * new_lp) + float(kl_beta) * kl_tok
                    per_seq_toks = r_mask.sum(dim=1).clamp_min(1.0)
                    per_seq_loss = (pg_tok * r_mask).sum(dim=1) / per_seq_toks
                    loss_chunk = per_seq_loss.sum() / float(n_act)
                    loss_chunk.backward()

                    step_loss_val += float(loss_chunk.detach().item())
                    step_kl_sum += float((kl_tok.detach() * r_mask).sum().item())
                    step_tok_sum += float(r_mask.sum().item())

                del ref_lp, r_mask, new_lp, log_ratio, kl_tok, pg_tok, per_seq_loss, loss_chunk

            torch.nn.utils.clip_grad_norm_(add_mod.parameters(), 1.0)
            opt.step()
            sched.step()
            opt_updates += 1

            mean_kl = step_kl_sum / max(1.0, step_tok_sum)
            kl_hist.append(mean_kl)
            loss_hist.append(step_loss_val)

            if log_every > 0 and (step_i + 1) % int(log_every) == 0:
                recent_r = sum(rewards_hist[-log_every:]) / max(1, len(rewards_hist[-log_every:]))
                print(
                    f"  [grpo {step_i+1:3d}/{n_steps}] T={t_step} r_ma={recent_r:.3f} "
                    f"active={active_groups}/{prompts_per_step} kl={mean_kl:.5f} "
                    f"|gate|={float(add_mod.out_gate.detach().abs().mean().item()):.5f}",
                    flush=True,
                )

            if dev.type == "cuda" and (step_i + 1) % 25 == 0:
                torch.cuda.empty_cache()

        add_mod.eval()
        train_time_s = round(time.perf_counter() - t_c0, 2)
        with add_mod.attach(model, n_loops=max(n_loops_choices), grad_checkpointing=False, record_stats=False):
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

    n_b = min(5, max(1, len(rewards_hist)))
    rew_ckpts: list[float] = []
    for b in range(n_b):
        lo_b = (b * len(rewards_hist)) // n_b
        hi_b = max(lo_b + 1, ((b + 1) * len(rewards_hist)) // n_b)
        rew_ckpts.append(round(sum(rewards_hist[lo_b:hi_b]) / max(1, hi_b - lo_b), 4))

    return {
        "module": add_mod,
        "num_parameters": int(add_mod.num_parameters()),
        "payload_mb": round(float(add_mod.payload_mb()), 4),
        "train_time_s": train_time_s,
        "grpo_steps": int(grpo_steps),
        "optimizer_updates": int(opt_updates),
        "mean_reward_all": round(sum(rewards_hist) / max(1, len(rewards_hist)), 4),
        "reward_checkpoints": rew_ckpts,
        "mean_active_group_frac": round(sum(active_frac_hist) / max(1, len(active_frac_hist)), 4),
        "mean_kl": round(sum(kl_hist) / max(1, len(kl_hist)), 6) if kl_hist else 0.0,
        "zero_step_kl_max": zero_step_kl_max,
        "loop_counts_seen": loop_counts_seen,
        "out_gate_abs_mean": round(float(add_mod.out_gate.detach().abs().mean().item()), 5),
        **drift,
    }


def _summarize_grpo(payload: dict[str, Any]) -> None:
    print("\n=== ADDITIVE LOOPED MOI + GRPO SUMMARY ===", flush=True)
    hdr = (
        f"{'Arm':26s} | {'T':3s} | {'Params':9s} | {'GSM8K':7s} | {'GSM_MoI':7s} | "
        f"{'MATH':7s} | {'MATH_MoI':8s} | {'MMLU_Δ':7s} | {'ProseKL(r/u)':15s}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for arm_name, rec in payload["arms"].items():
        print(
            f"{arm_name:26s} | {str(rec['n_loops']):3s} | {rec['payload_mb']:6.2f} MB | "
            f"{rec['gsm8k_acc']*100:6.2f}% | {rec['gsm8k_routed_acc']*100:6.2f}% | "
            f"{rec['math500_acc']*100:6.2f}% | {rec['math500_routed_acc']*100:7.2f}% | "
            f"{rec['mmlu_non_math_delta_pts']:+6.2f}% | "
            f"{rec['routed_prose_kl']:.4f} / {rec['unrouted_prose_kl']:.4f}",
            flush=True,
        )
    print("\n=== PAIRED BOOTSTRAP 95% CIs (GSM8K & MATH-500) ===", flush=True)
    for k, v in payload.get("paired_bootstrap", {}).items():
        print(
            f"  {k:48s}: diff={v['diff']*100:+5.2f} pts "
            f"CI95=[{v['ci95'][0]*100:+5.2f}, {v['ci95'][1]*100:+5.2f}] (w={v['wins']}/l={v['losses']})",
            flush=True,
        )


# ------------------------------------------------------------------ CPU Smoke Test
def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of GRPO rollouts, zero-step identity, updates, router & telemetry."""
    make_qwen2 = _load_smoke_qwen2_fn()
    model = make_qwen2(seed=29)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _, blocks = find_blocks(model)
    insert_layer = len(blocks) // 2

    tok_sm = _SmokeTokenizer()
    train_items = [
        {"prompt_ids": [1, 3, 5, 7], "gold_mod": 0},
        {"prompt_ids": [2, 4, 6, 8], "gold_mod": 1},
        {"prompt_ids": [9, 11, 13, 15], "gold_mod": 0},
        {"prompt_ids": [10, 12, 14, 16], "gold_mod": 1},
    ]

    # Deterministic varied oracle based on token parity matching prompt gold_mod
    def _smoke_grade(rep_items: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> list[float]:
        out: list[float] = []
        for it, r in zip(rep_items, rollouts):
            first_tok = int(r["token_ids"][0]) if r["token_ids"] else 0
            out.append(1.0 if (first_tok % 2) == int(it["gold_mod"]) else 0.0)
        return out

    h_ids = [torch.tensor([[1, 2, 3, 4, 2]], dtype=torch.long)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    grpo_res = _train_additive_grpo(
        model,
        tok_sm,
        train_items,
        _smoke_grade,
        pad_id=0,
        insert_layer=insert_layer,
        n_blocks=1,
        n_loops_choices=(1, 2, 3),
        max_steps=6,
        d_loop_key=16,
        grpo_steps=4,
        prompts_per_step=2,
        group_size=4,
        micro_bs=2,
        rollout_max_tokens=4,
        rollout_temperature=1.2,
        rollout_top_p=1.0,
        rollout_top_k=0,
        lr_block=2e-2,
        lr_gate=5e-2,
        kl_beta=0.04,
        grad_checkpointing=True,
        healthy_ids=h_ids,
        base_healthy_logp=base_lp,
        base_healthy_nll=base_nll,
        seed=42,
        log_every=0,
    )
    assert grpo_res["optimizer_updates"] >= 1, grpo_res
    assert grpo_res["zero_step_kl_max"] is not None and grpo_res["zero_step_kl_max"] < 1e-6, grpo_res
    assert grpo_res["out_gate_abs_mean"] > 0.0, grpo_res
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    # 3-class balanced MoI Ridge router check
    X_r = _extract_prompt_hidden_vecs(
        model, [[1, 3, 5], [2, 4, 6], [10, 11, 12], [13, 14, 15], [20, 21, 22], [23, 24, 25]], insert_layer
    )
    W_r, b_r = fit_ridge_router(
        X_r,
        ["math", "math", "pass_mmlu", "pass_mmlu", "pass_prose", "pass_prose"],
        BALANCED_ROUTER_CLASSES,
        lam=1.0,
    )
    preds = predict_ridge_router(X_r, W_r, b_r, BALANCED_ROUTER_CLASSES)
    assert all(p in BALANCED_ROUTER_CLASSES for p in preds)

    # Batched eval + telemetry across T=1..4
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "grpo_add.safetensors"
        grpo_res["module"].save(p)
        reloaded = type(grpo_res["module"]).load(p, template_model=model, verify=True)
        out_t2 = _hf_generate_batched_addloop(
            model, tok_sm, [[1, 3, 5], [2, 4, 6, 8]], max_new_tokens=4, batch_size=2,
            add_mod=reloaded, n_loops=2, record_stats=True,
        )
        tel_t2 = reloaded.telemetry_summary()
        out_t4 = _hf_generate_batched_addloop(
            model, tok_sm, [[1, 3, 5], [2, 4, 6, 8]], max_new_tokens=4, batch_size=2,
            add_mod=reloaded, n_loops=4, record_stats=True,
        )
        tel_t4 = reloaded.telemetry_summary()
        assert len(out_t2) == 2 and len(out_t4) == 2
        assert len(tel_t2["alpha_mean"]) == 3 and abs(sum(tel_t4["alpha_mean"]) - 1.0) < 1e-5

    return {
        "mode": "smoke",
        "passed": True,
        "optimizer_updates": grpo_res["optimizer_updates"],
        "zero_step_kl_max": grpo_res["zero_step_kl_max"],
        "out_gate_abs_mean": grpo_res["out_gate_abs_mean"],
        "telemetry_t2": tel_t2,
        "telemetry_t4": tel_t4,
    }


# ------------------------------------------------------------------ Full GPU Experiment
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

    print(f"[grpo] Loading {args.model} ({dtype}) on {dev}...", flush=True)
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

    # 1. Load datasets (disjoint train/test)
    print("[grpo] Loading GSM8K, MATH-500 & MMLU...", flush=True)
    gsm_ds = load_dataset("openai/gsm8k", "main")
    rng_py = random.Random(int(args.seed))

    gsm_train_raw = list(gsm_ds["train"])
    rng_py.shuffle(gsm_train_raw)
    gsm_train_items: list[dict[str, Any]] = []
    for r in gsm_train_raw[: int(args.gsm_train_n)]:
        ans = ec.extract_gsm8k_answer(str(r["answer"]), is_gold=True)
        if ans is None:
            continue
        pids = _format_messages(
            tok, [{"role": "user", "content": format_question(str(r["question"]))}],
            add_generation_prompt=True, enable_thinking=False,
        )
        gsm_train_items.append({"kind": "gsm8k", "prompt_ids": pids, "answer": ans})

    math_train_raw, math_test = load_math500_split(
        n_train=500 - int(args.math_test_n), n_test=int(args.math_test_n), seed=int(args.seed)
    )
    math_train_items = [
        {"kind": "math500", "prompt_ids": format_math500_prompt_ids(tok, r["problem"]), "answer": str(r["answer"])}
        for r in math_train_raw
    ]
    # 75% GSM8K / 25% MATH-500 mix in training pool via balanced replication
    mixed_train_pool = gsm_train_items + (math_train_items * max(1, len(gsm_train_items) // max(1, 3 * len(math_train_items))))

    gsm_test_raw = list(gsm_ds["test"])[: int(args.gsm8k_test_n)]
    gsm_test_pids = [
        _format_messages(
            tok, [{"role": "user", "content": format_question(str(r["question"]))}],
            add_generation_prompt=True, enable_thinking=False,
        )
        for r in gsm_test_raw
    ]
    gsm_test_gold = [ec.extract_gsm8k_answer(str(r["answer"]), is_gold=True) or "" for r in gsm_test_raw]
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
    prose_held_pids = [
        _format_messages(tok, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
        for p in prose_held
    ]

    # 2. 3-Class Balanced Middle-Layer MoI Router (`math`, `pass_mmlu`, `pass_prose`)
    math_r_pids = [it["prompt_ids"] for it in gsm_train_items[:160]] + [it["prompt_ids"] for it in math_train_items[:80]]
    r_train_pids = math_r_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = (
        ["math"] * len(math_r_pids)
        + ["pass_mmlu"] * len(mmlu_r_pids)
        + ["pass_prose"] * len(prose_r_pids)
    )
    W_r, b_r = fit_ridge_router(
        _extract_prompt_hidden_vecs(model, r_train_pids, insert_layer),
        r_train_labels,
        BALANCED_ROUTER_CLASSES,
        lam=1.0,
    )
    routed_gsm_fire = [p == "math" for p in predict_ridge_router(
        _extract_prompt_hidden_vecs(model, gsm_test_pids, insert_layer), W_r, b_r, BALANCED_ROUTER_CLASSES
    )]
    routed_math_fire = [p == "math" for p in predict_ridge_router(
        _extract_prompt_hidden_vecs(model, math_test_pids, insert_layer), W_r, b_r, BALANCED_ROUTER_CLASSES
    )]
    routed_mmlu_fire = [p == "math" for p in predict_ridge_router(
        _extract_prompt_hidden_vecs(model, mmlu_test_pids, insert_layer), W_r, b_r, BALANCED_ROUTER_CLASSES
    )]
    routed_prose_fire = [p == "math" for p in predict_ridge_router(
        _extract_prompt_hidden_vecs(model, prose_held_pids, insert_layer), W_r, b_r, BALANCED_ROUTER_CLASSES
    )]
    router_report = {
        "router_layer": insert_layer,
        "classes": list(BALANCED_ROUTER_CLASSES),
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
    print(f"[grpo] Router report: {router_report}", flush=True)

    letter_ids = ec._resolve_mmlu_letter_ids(tok)
    base_mmlu_raw = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)
    base_mmlu_sum = _summarize_mmlu(mmlu_test, base_mmlu_raw, seed=int(args.seed))
    base_mmlu_nonmath_acc = float(base_mmlu_sum["non_math"]["accuracy"])

    prose_full_ids = _to_seq_tensors(
        [
            _format_messages(
                tok,
                [{"role": "user", "content": pr}, {"role": "assistant", "content": "Here is a concise explanation."}],
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for pr in prose_held
        ],
        torch.device("cpu"),
    )
    base_prose_logp: list[torch.Tensor | None] = []
    nll_s, cnt = 0.0, 0
    with torch.no_grad():
        for ids in prose_full_ids:
            logits = _tail_logits(model, ids.to(dev), 0)[:-1].float()
            lp = F.log_softmax(logits, dim=-1)
            nll_s += float(F.nll_loss(lp, ids[0, 1:].to(dev), reduction="sum"))
            cnt += int(ids.shape[1] - 1)
            base_prose_logp.append(lp.to(dtype=torch.bfloat16, device="cpu"))
            del logits, lp
    base_prose_nll = nll_s / max(1, cnt)

    def _verifiable_grade_fn(rep_items: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> list[float]:
        rewards: list[float] = []
        for it, raw in zip(rep_items, rollouts):
            if it["kind"] == "gsm8k":
                body = lr.strip_eos(raw["token_ids"], eos_set)
                txt = tok.decode(body, skip_special_tokens=True).strip()
                pred = ec.extract_gsm8k_answer(txt, is_gold=False)
                rewards.append(1.0 if (pred is not None and pred == it["answer"]) else 0.0)
            else:
                g = dm.grade_math(
                    raw["token_ids"], raw["finish_reason"], str(it["answer"]), tok, eos_set, int(args.rollout_max_tokens)
                )
                rewards.append(1.0 if bool(g["correct"]) else 0.0)
        return rewards

    # 3. Train `additive_1x_grpo` and `additive_loop_grpo`
    ckpt_root = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    dir_ctx = nullcontext(str(ckpt_root)) if ckpt_root is not None else tempfile.TemporaryDirectory()

    def _load_or_train_grpo(
        arm_nm: str,
        *,
        n_loops_choices: tuple[int, ...],
        seed_val: int,
    ) -> dict[str, Any]:
        st_path = work_dir / f"{arm_nm}.safetensors"
        meta_path = work_dir / f"{arm_nm}.meta.json"
        if st_path.exists():
            print(f"[grpo] Resuming {arm_nm} from checkpoint {st_path}...", flush=True)
            mod = AdditiveLoopedMoI.load(st_path, template_model=model, verify=True).to(dev).eval()
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return {"module": mod, **meta}
            with mod.attach(model, n_loops=max(n_loops_choices), grad_checkpointing=False, record_stats=False):
                drift = eval_healthy_drift(model, prose_full_ids, base_prose_logp, base_prose_nll)
            return {
                "module": mod,
                "num_parameters": int(mod.num_parameters()),
                "payload_mb": round(float(mod.payload_mb()), 4),
                "train_time_s": 0.0,
                "resumed_from_safetensors": True,
                "out_gate_abs_mean": round(float(mod.out_gate.detach().abs().mean().item()), 5),
                **drift,
            }

        print(f"[grpo] Training {arm_nm} (T in {set(n_loops_choices)}, {args.grpo_steps} steps)...", flush=True)
        res = _train_additive_grpo(
            model, tok, mixed_train_pool, _verifiable_grade_fn,
            pad_id=pad_id, insert_layer=insert_layer, n_blocks=int(args.n_blocks),
            n_loops_choices=n_loops_choices, max_steps=int(args.max_steps), d_loop_key=int(args.d_loop_key),
            grpo_steps=int(args.grpo_steps), prompts_per_step=int(args.prompts_per_step),
            group_size=int(args.group_size), micro_bs=int(args.micro_bs),
            rollout_max_tokens=int(args.rollout_max_tokens),
            lr_block=float(args.lr_block), lr_gate=float(args.lr_gate), kl_beta=float(args.kl_beta),
            grad_checkpointing=True, healthy_ids=prose_full_ids, base_healthy_logp=base_prose_logp,
            base_healthy_nll=base_prose_nll, seed=seed_val, log_every=10,
        )
        res["module"].save(st_path)
        _atomic_write_json(meta_path, {k: v for k, v in res.items() if k != "module"})
        print(
            f"[grpo] {arm_nm} done ({res['train_time_s']}s) | upd={res['optimizer_updates']} | "
            f"rew={res['reward_checkpoints']} | |gate|={res['out_gate_abs_mean']} | prose_kl={res['healthy_kl']}",
            flush=True,
        )
        return res

    with dir_ctx as work_str:
        work_dir = Path(work_str)

        res_1x = _load_or_train_grpo("additive_1x_grpo", n_loops_choices=(1,), seed_val=int(args.seed))
        res_loop = _load_or_train_grpo("additive_loop_grpo", n_loops_choices=(1, 2, 3), seed_val=int(args.seed) + 17)

        # 4. Shared Length-Sorted Left-Padded Batched HF Evaluation across all arms
        def _eval_arm(
            arm_name: str,
            *,
            n_loops: int,
            payload_mb: float,
            n_params: int,
            unrouted_prose_kl: float,
            add_mod: AdditiveLoopedMoI | None = None,
            record_stats: bool = False,
        ) -> tuple[dict[str, Any], list[bool], list[bool]]:
            t_e0 = time.perf_counter()
            raw_gsm = _hf_generate_batched_addloop(
                model, tok, gsm_test_pids, max_new_tokens=int(args.gsm8k_eval_max_tokens),
                batch_size=int(args.batch_size_eval), base_seed=int(args.seed),
                add_mod=add_mod, n_loops=n_loops if add_mod is not None else None,
                record_stats=record_stats,
            )
            tel_gsm = add_mod.telemetry_summary() if (add_mod is not None and record_stats) else None
            raw_math = _hf_generate_batched_addloop(
                model, tok, math_test_pids, max_new_tokens=int(args.math_eval_max_tokens),
                batch_size=int(args.batch_size_eval), base_seed=int(args.seed) + 500,
                add_mod=add_mod, n_loops=n_loops if add_mod is not None else None,
                record_stats=False,
            )
            tel = tel_gsm
            gsm_flags, gsm_toks = _grade_gsm8k_outputs(gsm_test_gold, raw_gsm, tok, eos_set)
            math_flags, math_toks = _grade_math500_outputs(
                math_test, raw_math, tok, eos_set, int(args.math_eval_max_tokens)
            )

            fired_mmlu_idxs = [i for i, f in enumerate(routed_mmlu_fire) if f]
            if fired_mmlu_idxs and add_mod is not None:
                sub_pids = [mmlu_test_pids[i] for i in fired_mmlu_idxs]
                sub_items = [mmlu_test[i] for i in fired_mmlu_idxs]
                with add_mod.attach(model, n_loops=n_loops, grad_checkpointing=False, record_stats=False):
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
                fire_flags=routed_mmlu_fire if add_mod is not None else [False] * len(mmlu_test),
                seed=int(args.seed),
            )
            nonmath_acc = float(mmlu_sum["non_math"]["accuracy"])

            fired_prose_idxs = [i for i, f in enumerate(routed_prose_fire) if f]
            routed_prose_kl = 0.0
            if fired_prose_idxs and add_mod is not None:
                with add_mod.attach(model, n_loops=n_loops, grad_checkpointing=False, record_stats=False):
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
                "gsm8k_routed_acc": 0.0,
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
            ("additive_1x_grpo", {
                "n_loops": 1, "payload_mb": res_1x["payload_mb"], "n_params": res_1x["num_parameters"],
                "unrouted_prose_kl": res_1x["healthy_kl"], "add_mod": res_1x["module"], "record_stats": True,
            }),
        ]
        for t_eval in (1, 2, 3, 4):
            eval_specs.append((f"additive_loop_grpo_T{t_eval}", {
                "n_loops": t_eval, "payload_mb": res_loop["payload_mb"], "n_params": res_loop["num_parameters"],
                "unrouted_prose_kl": res_loop["healthy_kl"], "add_mod": res_loop["module"], "record_stats": True,
            }))

        for arm_nm, kw in eval_specs:
            ev_p = work_dir / f"eval_{arm_nm}.json"
            if ev_p.exists():
                print(f"[grpo] Loaded cached eval -> {ev_p}", flush=True)
                ev_d = json.loads(ev_p.read_text(encoding="utf-8"))
                rec, gf, mf = ev_d["rec"], ev_d["gsm_flags"], ev_d["math_flags"]
            else:
                print(f"[grpo] Evaluating {arm_nm}...", flush=True)
                rec, gf, mf = _eval_arm(arm_nm, **kw)
                _atomic_write_json(ev_p, {"rec": rec, "gsm_flags": gf, "math_flags": mf})
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
        for a, b in [
            ("additive_loop_grpo_T3", "base"),
            ("additive_loop_grpo_T3", "additive_1x_grpo"),
            ("additive_loop_grpo_T3", "additive_loop_grpo_T1"),
            ("additive_loop_grpo_T4", "additive_loop_grpo_T3"),
            ("additive_1x_grpo", "base"),
        ]:
            paired_boot[f"gsm8k: {a} - {b}"] = _paired_bool_bootstrap(gsm_flags_by_arm[a], gsm_flags_by_arm[b], seed=int(args.seed))
            paired_boot[f"math500: {a} - {b}"] = _paired_bool_bootstrap(math_flags_by_arm[a], math_flags_by_arm[b], seed=int(args.seed))

    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    payload = {
        "experiment": "additive_loop_grpo_pilot",
        "model": args.model,
        "dtype": str(dtype),
        "device": str(dev),
        "insert_layer": insert_layer,
        "n_layers": n_layers,
        "n_blocks": int(args.n_blocks),
        "router_report": router_report,
        "train_summaries": {
            "additive_1x_grpo": {k: v for k, v in res_1x.items() if k != "module"},
            "additive_loop_grpo": {k: v for k, v in res_loop.items() if k != "module"},
        },
        "arms": arms_out,
        "paired_bootstrap": paired_boot,
        "total_time_s": round(time.perf_counter() - t0, 2),
    }
    _atomic_write_json(Path(args.out), payload)
    _summarize_grpo(payload)
    print(f"\n[grpo] Saved -> {args.out} (total {payload['total_time_s']}s)", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verifiable On-Policy GRPO on Additive Looped MoI")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--gsm-train-n", type=int, default=1500)
    ap.add_argument("--gsm8k-test-n", type=int, default=300)
    ap.add_argument("--math-test-n", type=int, default=150)
    ap.add_argument("--mmlu-test-n", type=int, default=200)
    ap.add_argument("--grpo-steps", type=int, default=60)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--rollout-max-tokens", type=int, default=384)
    ap.add_argument("--gsm8k-eval-max-tokens", type=int, default=384)
    ap.add_argument("--math-eval-max-tokens", type=int, default=512)
    ap.add_argument("--batch-size-eval", type=int, default=128)
    ap.add_argument("--lr-block", type=float, default=5e-5)
    ap.add_argument("--lr-gate", type=float, default=2.5e-3)
    ap.add_argument("--kl-beta", type=float, default=0.04)
    ap.add_argument("--n-blocks", type=int, default=1)
    ap.add_argument("--insert-layer", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--d-loop-key", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--out", default="additive_loop_grpo.json")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        print(json.dumps(run_smoke(), indent=2), flush=True)
        return
    run_real(args)


if __name__ == "__main__":
    main()
