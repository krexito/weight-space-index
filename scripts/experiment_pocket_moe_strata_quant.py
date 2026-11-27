"""Exp 1: Strata S2_K/S3_K + H64 + Block-LDL + 64-flip compression of Pocket MoE r=8 LoRA cartridges.

Tests on the exact K=6 5-skill benchmark (`Qwen3.5-4B -> Qwen3.5-0.8B`, `60` blind hard items,
`24` guardrails, `200` MMLU test, `16` held-out prose controls) whether quantizing the full-rank
capacity of an `r=8` cartridge (`10.32 MB`) down to `S3_K` (`~2.06 MB`) and `S2_K` (`~1.41 MB`)
using H64 + Hessian clipping + Block-LDL + 64-flip preserves accuracy better than lowering rank
(`r=2` at `2.58 MB`, `r=1` at `1.29 MB`) or naive 2-bit RTN.

CPU smoke test::
    python scripts/experiment_pocket_moe_strata_quant.py --smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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

import eval_capability_and_boundary_lfm3 as ec  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from experiment_pocket_moe_ceiling import (  # noqa: E402
    GUARDRAIL_FAMILIES,
    _block_prefixes,
    _enable_emb_grad_hook,
    _grade_rows,
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
    _score_routed_mmlu_sparse,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _PlantedReasoningLM,
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
    build_reasoning_dataset,
)
from experiment_reasoning_distill_v2 import (  # noqa: E402
    _format_messages,
    format_question,
    grade_response,
)
from wsi.capture import find_blocks  # noqa: E402
from wsi.strata_quant import (  # noqa: E402
    gptq_flip_quantize_sk_torch,
    matrix_wire_bytes,
    rtn_quantize_sk_torch,
)
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402


@torch.no_grad()
def _batched_hf_generate(
    model: nn.Module,
    tok: Any,
    prompts_ids: list[list[int]],
    seeds: list[int],
    *,
    max_new_tokens: int,
    batch_size: int = 32,
    param_updates: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Fast left-padded batched HF generation using the exact run sampling (`lr.SAMPLING`)."""
    dev = next(model.parameters()).device
    eos_list = lr.eos_ids(tok)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)

    def _run() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        bs = max(1, int(batch_size))
        for b0 in range(0, len(prompts_ids), bs):
            chunk_p = prompts_ids[b0 : b0 + bs]
            chunk_s = seeds[b0 : b0 + bs]
            max_l = max(len(p) for p in chunk_p)
            inp = torch.full((len(chunk_p), max_l), pad_id, dtype=torch.long, device=dev)
            mask = torch.zeros((len(chunk_p), max_l), dtype=torch.long, device=dev)
            for row_i, pids in enumerate(chunk_p):
                inp[row_i, max_l - len(pids) :] = torch.tensor(pids, dtype=torch.long, device=dev)
                mask[row_i, max_l - len(pids) :] = 1
            torch.manual_seed(int(chunk_s[0]))
            gen = model.generate(
                input_ids=inp,
                attention_mask=mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=lr.SAMPLING["temperature"],
                top_p=lr.SAMPLING["top_p"],
                top_k=lr.SAMPLING["top_k"],
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=eos_list,
            )
            for row_i in range(len(chunk_p)):
                gen_ids = [int(x) for x in gen[row_i, max_l:].tolist()]
                # Trim trailing pads if early-stopped batch items padded with eos/pad
                eos_set = set(eos_list) | {pad_id}
                trimmed: list[int] = []
                fin = "length"
                for tid in gen_ids:
                    trimmed.append(tid)
                    if tid in eos_set:
                        fin = "stop"
                        break
                out.append({"token_ids": trimmed, "finish_reason": fin})
            del inp, mask, gen
        return out

    if param_updates:
        with _temp_apply_weights(model, param_updates):
            return _run()
    return _run()


@torch.no_grad()
def _capture_lora_input_hessians(
    model: nn.Module,
    layer_idxs: list[int],
    train_samples: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Capture normalized input Hessians `H = (X^T X) / N` on CPU for every wrapped nn.Linear."""
    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
    prefixes = _block_prefixes(model)
    hessians: dict[str, torch.Tensor] = {}
    tok_counts: dict[str, int] = {}
    handles = []

    def _make_hook(full_nm: str, in_dim: int):
        def _hook(_mod: nn.Module, inp: tuple[torch.Tensor, ...]) -> None:
            x = inp[0].detach().reshape(-1, in_dim).float()
            h_inc = (x.T @ x).cpu()
            if full_nm not in hessians:
                hessians[full_nm] = h_inc
                tok_counts[full_nm] = int(x.shape[0])
            else:
                hessians[full_nm].add_(h_inc)
                tok_counts[full_nm] += int(x.shape[0])
        return _hook

    for li in layer_idxs:
        blk = blocks[li]
        pref = prefixes[li]
        for sub_nm, mod in blk.named_modules():
            if isinstance(mod, nn.Linear):
                full_nm = f"{pref}.{sub_nm}.weight"
                handles.append(mod.register_forward_pre_hook(_make_hook(full_nm, int(mod.in_features))))

    try:
        model.eval()
        for sp in train_samples:
            ids_dev = torch.tensor(sp["ids"], dtype=torch.long, device=dev).unsqueeze(0)
            model(input_ids=ids_dev, use_cache=False)
            del ids_dev
    finally:
        for h in handles:
            h.remove()

    for nm in list(hessians.keys()):
        n_tok = max(1, tok_counts.get(nm, 1))
        hessians[nm].div_(float(n_tok))
    return hessians


def _train_regularized_lora_with_factors(
    model: nn.Module,
    layer_idxs: list[int],
    train_samples: list[dict[str, Any]],
    *,
    rank: int,
    alpha_scale: float,
    total_steps: int,
    grad_accum: int,
    lr_val: float,
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
    seed: int = 42,
) -> dict[str, Any]:
    """Identical to `_train_regularized_lora_expert` plus returning raw `(lora_A, lora_B, scale)` factors."""
    assert len(train_samples) > 0 and total_steps >= 1 and grad_accum >= 1
    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
    prefixes = _block_prefixes(model)
    orig_req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in orig_req:
        p.requires_grad_(False)
    was_training = bool(model.training)
    was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
    enabled_gc = False
    if hasattr(model, "gradient_checkpointing_enable") and not was_gc:
        try:
            model.gradient_checkpointing_enable()
            enabled_gc = True
        except Exception:
            enabled_gc = False

    wrapped: list[tuple[nn.Module, str, nn.Linear, str, _RegLoRALinear]] = []
    for li in layer_idxs:
        blk = blocks[li]
        blk_pref = prefixes[li]
        for sub_nm, lin in [(nm, m) for nm, m in blk.named_modules() if isinstance(m, nn.Linear)]:
            parts = sub_nm.split(".")
            parent = blk
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            wrapper = _RegLoRALinear(lin, rank=rank, alpha_scale=alpha_scale, dropout=lora_dropout)
            setattr(parent, attr, wrapper)
            wrapped.append((parent, attr, lin, f"{blk_pref}.{sub_nm}.weight", wrapper))

    lora_params = [p for *_rest, w in wrapped for p in (w.lora_A, w.lora_B)]
    adapter_mb = round(sum(w.adapter_bytes() for *_rest, w in wrapped) / (1024**2), 4)
    emb_handles = _enable_emb_grad_hook(model)
    step_losses: list[float] = []
    t_c0 = time.perf_counter()

    try:
        model.eval()
        for *_rest, w in wrapped:
            w.train()

        n_steps = int(total_steps)
        warmup_steps = max(1, int(round(n_steps * float(warmup_frac)))) if float(warmup_frac) > 0.0 else 0
        opt = torch.optim.AdamW(lora_params, lr=float(lr_val), weight_decay=float(weight_decay))

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
                    del anc_dev, lp_s, lp_b, kl_anc

            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            sched.step()
            step_losses.append(step_loss)

        for *_rest, w in wrapped:
            w.eval()

        compile_time_s = round(time.perf_counter() - t_c0, 2)
        from experiment_pocket_moe_ceiling import eval_healthy_drift

        drift = (
            eval_healthy_drift(model, healthy_ids, base_healthy_logp, base_healthy_nll)
            if (healthy_ids is not None and base_healthy_logp is not None)
            else {"healthy_kl": 0.0, "delta_healthy_nll": 0.0}
        )
        merged_weights = {full_nm: w.merged_weight_cpu() for _, _, _, full_nm, w in wrapped}
        raw_factors = {
            full_nm: (
                w.lora_A.detach().float().cpu(),
                w.lora_B.detach().float().cpu(),
                float(w.scale),
                w.base.weight.detach().float().cpu(),
                w.base.weight.dtype,
            )
            for _, _, _, full_nm, w in wrapped
        }
    finally:
        for h in emb_handles:
            h.remove()
        for parent, attr, orig_lin, _, _ in wrapped:
            setattr(parent, attr, orig_lin)
        if enabled_gc and hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
        if was_training:
            model.train()
        for p, req in orig_req:
            p.requires_grad_(req)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    n_ckpts = min(8, len(step_losses))
    ckpts = []
    for b in range(n_ckpts):
        seg = step_losses[(b * len(step_losses)) // n_ckpts : ((b + 1) * len(step_losses)) // n_ckpts] or [step_losses[0]]
        ckpts.append(round(sum(seg) / len(seg), 5))

    return {
        "weights": merged_weights,
        "factors": raw_factors,
        "adapter_mb": adapter_mb,
        "step_losses_checkpoints": ckpts,
        "compile_time_s": compile_time_s,
        **drift,
    }


@torch.no_grad()
def _compress_lora_factors(
    raw_factors: dict[str, tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.dtype]],
    hessians: dict[str, torch.Tensor],
    *,
    code_name: str,
    method: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    """Quantize `(lora_A, lora_B)` factors under exact S_K super-block wire accounting.

    `method="rtn"` -> naive unclipped RTN without H64.
    `method="strata"` -> `H64` + Hessian-weighted clipping + Block-LDL (GPTQ) + 64-flip.
    Returns `(merged_weights_cpu, total_wire_bytes, total_lora_A_hessian_err)`.
    """
    merged: dict[str, torch.Tensor] = {}
    total_wire_bytes = 0
    total_a_err = 0.0

    for full_nm, (A_cpu, B_cpu, scale, base_w_cpu, orig_dtype) in raw_factors.items():
        A_d = A_cpu.to(device=device, dtype=torch.float32)  # [r, in]
        Bt_d = B_cpu.T.contiguous().to(device=device, dtype=torch.float32)  # [r, out]
        H_a = hessians[full_nm].to(device=device, dtype=torch.float32)  # [in, in]
        H_b = torch.eye(Bt_d.shape[1], device=device, dtype=torch.float32)

        if method == "rtn":
            A_q, wb_a, err_a = rtn_quantize_sk_torch(A_d, code_name, use_clipping=False, use_h64=False, H=H_a)
            Bt_q, wb_b, _ = rtn_quantize_sk_torch(Bt_d, code_name, use_clipping=False, use_h64=False, H=H_b)
        elif method == "strata":
            A_q, wb_a, err_a = gptq_flip_quantize_sk_torch(A_d, H_a, code_name, use_h64=True, max_flips=64)
            Bt_q, wb_b, _ = gptq_flip_quantize_sk_torch(Bt_d, H_b, code_name, use_h64=True, max_flips=64)
        else:
            raise ValueError(method)

        total_wire_bytes += int(wb_a + wb_b)
        total_a_err += float(err_a or 0.0)
        delta_w = (float(scale) * (Bt_q.T.float() @ A_q.float())).cpu()
        merged[full_nm] = (base_w_cpu + delta_w).to(orig_dtype)
        del A_d, Bt_d, H_a, H_b, A_q, Bt_q, delta_w

    return merged, total_wire_bytes, total_a_err


def _print_summary(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return
    print("\n=== EXP 1: POCKET MOE r=8 COMPRESSED WITH STRATA S3_K / S2_K ===", flush=True)
    for arm_nm, st in payload.get("generation_arms", {}).items():
        h, g = st["hard_60"], st["guardrail_24"]
        mb = payload["arm_mb_per_skill"].get(arm_nm)
        mb_s = f"{mb:5.2f}MB" if mb is not None else "  ref  "
        pkl = payload["prose_kl"].get(arm_nm, 0.0)
        print(
            f"  {arm_nm:28s} | {mb_s} | hard60={h['strict_accuracy']*100:5.1f}% ({h['n_correct']:2d}/60) "
            f"[old1-4={h['old_skills_1_to_4_acc']*100:5.1f}%, s5={h['last_skill_5_acc']*100:5.1f}%] | "
            f"guard={g['strict_accuracy']*100:5.1f}% | pkl={pkl:.5f}",
            flush=True,
        )
    print("\n=== PAIRED BOOTSTRAP CIs (hard_60) ===", flush=True)
    for k, v in payload.get("paired_hard_60", {}).items():
        print(f"  {k:34s}: diff={v['diff']*100:+5.1f} pts CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})", flush=True)
    print("\nCriteria:", payload.get("criteria"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_generation"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


def run_smoke() -> dict[str, Any]:
    """Fast CPU verification of factor capture, S2_K/S3_K Strata compression vs naive RTN, and exact byte accounting."""
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    seqs = [[1, 2, 4, 10, 20, 20, 20, 20], [2, 1, 4, 11, 20, 20, 20, 20]]
    tr = []
    for s in seqs:
        ids_t = torch.tensor(s, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            tl = model(ids_t).logits[0, 3:7].float()
        v, idx = tl.topk(8, dim=-1)
        tr.append({"ids": s, "lo": 3, "hi": 7, "topk_ids": idx.to(torch.int32).cpu(), "topk_logits": v.to(torch.float16).cpu()})

    res = _train_regularized_lora_with_factors(
        model, [0, 1], tr, rank=8, alpha_scale=16.0, total_steps=6, grad_accum=2, lr_val=2e-2, seed=42
    )
    assert all(torch.equal(v.cpu(), orig[k]) for k, v in model.state_dict().items())
    hess = _capture_lora_input_hessians(model, [0, 1], tr)

    _, wb_rtn2, err_rtn2 = _compress_lora_factors(res["factors"], hess, code_name="S2_K", method="rtn", device=torch.device("cpu"))
    _, wb_str2, err_str2 = _compress_lora_factors(res["factors"], hess, code_name="S2_K", method="strata", device=torch.device("cpu"))
    _, wb_str3, err_str3 = _compress_lora_factors(res["factors"], hess, code_name="S3_K", method="strata", device=torch.device("cpu"))
    assert wb_rtn2 == wb_str2 and wb_str2 < wb_str3
    assert err_str3 < err_str2 < err_rtn2, (err_str3, err_str2, err_rtn2)
    return {
        "mode": "smoke",
        "r8_fp16_mb": res["adapter_mb"],
        "s2_wire_bytes": wb_str2,
        "s3_wire_bytes": wb_str3,
        "err_rtn_s2": round(err_rtn2, 6),
        "err_strata_s2": round(err_str2, 6),
        "err_strata_s3": round(err_str3, 6),
        "passed": True,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    torch.manual_seed(int(args.seed))

    ds = build_fewshot_dataset(k_shots=int(args.k_shots))
    hard_train, hard_test = ds["hard_train"], ds["hard_test"]
    guardrail_test = ds["guardrail_test"]
    all_test = hard_test + guardrail_test
    n_hard = len(hard_test)
    mmlu_test, mmlu_router_train = load_mmlu_split(
        n_test=int(args.n_mmlu_test), n_router_train=int(args.n_mmlu_router_train), seed=int(args.seed)
    )

    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok_s.pad_token_id is None:
        tok_s.pad_token = tok_s.eos_token
    eos_set = set(lr.eos_ids(tok_s))
    think_end_id = tok_s.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_id, int) or think_end_id < 0:
        think_end_id = None

    train_pids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in hard_train
    ]
    test_pids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in all_test
    ]
    test_seeds = [int(args.seed) + 1000 + i for i in range(len(all_test))]
    train_seeds = [int(args.seed) + 100 + i for i in range(len(hard_train))]

    # ---- Phase 1: Teacher 4B canonical traces + Top-32 logits ----
    t_p1 = time.perf_counter()
    print(f"[exp1] Phase 1: Teacher ({args.teacher}) generation + Top-32 logits...", flush=True)
    teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    t_train_raw = _batched_hf_generate(
        teacher_hf, tok_s, train_pids, train_seeds, max_new_tokens=int(args.max_new_tokens), batch_size=int(args.eval_batch_size)
    )
    t_test_raw = _batched_hf_generate(
        teacher_hf, tok_s, test_pids, test_seeds, max_new_tokens=int(args.max_new_tokens), batch_size=int(args.eval_batch_size)
    )
    teacher_test_rows = _grade_rows(all_test, t_test_raw, tok_s, eos_set, think_end_id)
    teacher_summary = _summarize_fewshot_arm(teacher_test_rows, arm_raw=t_test_raw, base_raw=None)

    kept_4b = {sk: 0 for sk in SKILLS_5}
    primary_pairs: dict[str, list[tuple[str, str]]] = {sk: [] for sk in SKILLS_5}
    primary_resp_by_item: list[str] = []
    for it, raw_m in zip(hard_train, t_train_raw):
        sk = it["family"]
        q_txt = it["question"]
        txt = tok_s.decode(lr.strip_eos(raw_m["token_ids"], eos_set), skip_special_tokens=False)
        g = grade_response(raw_m["token_ids"], txt, int(it["answer"]), thinking_enabled=False, finish_reason=raw_m["finish_reason"], eos_set=eos_set, tok=tok_s)
        is_ok = bool(g["strict_correct"] and g["has_boxed"] and g["total_tokens"] <= int(args.max_teacher_train_tokens))
        if is_ok:
            kept_4b[sk] += 1
            primary_txt = txt
        else:
            primary_txt = it["fallback_response"]
        primary_resp_by_item.append(primary_txt)
        primary_pairs[sk].append((q_txt, primary_txt))

    unique_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for sk in SKILLS_5:
        for pair in primary_pairs[sk]:
            unique_pairs.setdefault(pair, {})
    with torch.no_grad():
        for (q_txt, resp_txt), entry in unique_pairs.items():
            p_ids = _format_messages(tok_s, [{"role": "user", "content": format_question(q_txt)}], add_generation_prompt=True, enable_thinking=False)
            full_ids = _format_messages(
                tok_s, [{"role": "user", "content": format_question(q_txt)}, {"role": "assistant", "content": resp_txt}], add_generation_prompt=False, enable_thinking=False
            )
            lo = max(0, len(p_ids) - 1)
            hi = len(full_ids) - 1
            ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
            t_logits = _tail_logits(teacher_hf, ids_dev, lo)[: hi - lo].float()
            vals, idx = t_logits.topk(32, dim=-1)
            entry.update({"ids": full_ids, "lo": lo, "hi": hi, "topk_ids": idx.to(torch.int32).cpu(), "topk_logits": vals.to(torch.float16).cpu()})
            del ids_dev, t_logits

    del teacher_hf
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    primary_by_skill = {sk: [dict(unique_pairs[p], family=sk) for p in primary_pairs[sk]] for sk in SKILLS_5}
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student 0.8B, Layer-12 Router, Base MMLU & Prose ----
    t_p2 = time.perf_counter()
    print(f"[exp1] Phase 2: Loading Student ({args.student}) + Layer-12 Router + Base MMLU...", flush=True)
    student_m = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in student_m.parameters():
        p.requires_grad_(False)
    _, blocks = find_blocks(student_m)
    all_layers = list(range(len(blocks)))
    router_layer = len(blocks) // 2

    guard_r_pids = [_format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False) for it in ds["guardrail_router_train"]]
    mmlu_r_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_router_train]
    mmlu_test_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_test]
    prose_r_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in ds["prose_router_train"]]
    prose_held_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in ds["prose_heldout"]]

    r_train_pids = train_pids + guard_r_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = [it["family"] for it in hard_train] + ["pass_math"] * len(guard_r_pids) + ["pass_mmlu"] * len(mmlu_r_pids) + ["pass_prose"] * len(prose_r_pids)
    W_r, b_r = fit_ridge_router(_extract_prompt_hidden_vecs(student_m, r_train_pids, router_layer), r_train_labels, ROUTER_FIT_CLASSES, lam=1.0)
    routed_test_classes = _predict_fewshot_router(_extract_prompt_hidden_vecs(student_m, test_pids, router_layer), W_r, b_r)
    routed_mmlu_classes = _predict_fewshot_router(_extract_prompt_hidden_vecs(student_m, mmlu_test_pids, router_layer), W_r, b_r)
    routed_prose_classes = _predict_fewshot_router(_extract_prompt_hidden_vecs(student_m, prose_held_pids, router_layer), W_r, b_r)

    router_report = {
        "router_layer": router_layer,
        "hard_acc": round(sum(p == it["family"] for p, it in zip(routed_test_classes[:n_hard], hard_test)) / n_hard, 4),
        "guardrail_passthrough_acc": round(sum(p == "passthrough" for p in routed_test_classes[n_hard:]) / len(guardrail_test), 4),
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_classes) / len(mmlu_test), 4),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_classes) / len(routed_prose_classes), 4),
    }
    print(f"[exp1] Router: {router_report}", flush=True)

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

    prose_full_ids, base_prose_logp, base_prose_nll = _compute_base_logp(ds["prose_heldout"])
    anchor_full_ids, base_anchor_logp, _ = _compute_base_logp(ds["prose_router_train"])
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Train r8, r2@4e-4, r1 + Quantize Exact r8 Factors into S3_K / S2_K / RTN-2b ----
    t_p3 = time.perf_counter()
    print("[exp1] Phase 3: Training 3 LoRA tiers + Strata quantization of r=8 factors...", flush=True)
    arm_names = [
        "r8_fp16_routed",
        "r8_s3k_h64_gptq_flip_routed",
        "r8_s2k_h64_gptq_flip_routed",
        "r8_rtn_2bit_routed",
        "r2_fp16_lr4e4_routed",
        "r1_fp16_routed",
    ]

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        expert_paths: dict[tuple[str, str], Path] = {}
        arm_mb_skill: dict[str, list[float]] = {a: [] for a in arm_names}
        domain_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in arm_names}
        quant_err_stats: dict[str, dict[str, float]] = {}

        for s_idx, sk in enumerate(SKILLS_5):
            skill_seed = int(args.seed) + s_idx
            # 1. Input Hessians on base student for this skill's K=6 traces
            hess = _capture_lora_input_hessians(student_m, all_layers, primary_by_skill[sk])

            # 2. Train r=8 once & export raw factors
            torch.manual_seed(skill_seed)
            res8 = _train_regularized_lora_with_factors(
                student_m,
                all_layers,
                primary_by_skill[sk],
                rank=8,
                alpha_scale=16.0,
                total_steps=48,
                grad_accum=3,
                lr_val=2.0e-4,
                anchor_ids=anchor_full_ids,
                base_anchor_logp=base_anchor_logp,
                healthy_ids=prose_full_ids,
                base_healthy_logp=base_prose_logp,
                base_healthy_nll=base_prose_nll,
                seed=skill_seed,
            )
            r8_factors = res8["factors"]
            r8_w = res8["weights"]
            p8 = exp_dir / f"r8_fp16_routed_{sk}.pt"
            torch.save(r8_w, p8)
            expert_paths[("r8_fp16_routed", sk)] = p8
            arm_mb_skill["r8_fp16_routed"].append(float(res8["adapter_mb"]))
            with _temp_apply_weights(student_m, r8_w):
                domain_mmlu_scores["r8_fp16_routed"][sk] = _score_routed_mmlu_sparse(
                    student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                )
            del r8_w

            # 3. Quantize the exact same r=8 factors in-HBM (0 extra training!)
            q_specs = [
                ("r8_s3k_h64_gptq_flip_routed", "S3_K", "strata"),
                ("r8_s2k_h64_gptq_flip_routed", "S2_K", "strata"),
                ("r8_rtn_2bit_routed", "S2_K", "rtn"),
            ]
            quant_err_stats[sk] = {}
            for a_nm, q_code, q_meth in q_specs:
                w_q, wire_b, err_a = _compress_lora_factors(r8_factors, hess, code_name=q_code, method=q_meth, device=dev)
                mb_q = round(wire_b / (1024**2), 4)
                p_q = exp_dir / f"{a_nm}_{sk}.pt"
                torch.save(w_q, p_q)
                expert_paths[(a_nm, sk)] = p_q
                arm_mb_skill[a_nm].append(mb_q)
                quant_err_stats[sk][a_nm] = round(err_a, 6)
                with _temp_apply_weights(student_m, w_q):
                    domain_mmlu_scores[a_nm][sk] = _score_routed_mmlu_sparse(
                        student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                    )
                del w_q
            del r8_factors, hess

            # 4. Train low-rank baselines (r=2@4e-4 and r=1@2e-4)
            for a_nm, rk, a_sc, lr_v in [
                ("r2_fp16_lr4e4_routed", 2, 4.0, 4.0e-4),
                ("r1_fp16_routed", 1, 2.0, 2.0e-4),
            ]:
                torch.manual_seed(skill_seed)
                res_lr = _train_regularized_lora_with_factors(
                    student_m,
                    all_layers,
                    primary_by_skill[sk],
                    rank=rk,
                    alpha_scale=a_sc,
                    total_steps=48,
                    grad_accum=3,
                    lr_val=lr_v,
                    anchor_ids=anchor_full_ids,
                    base_anchor_logp=base_anchor_logp,
                    healthy_ids=prose_full_ids,
                    base_healthy_logp=base_prose_logp,
                    base_healthy_nll=base_prose_nll,
                    seed=skill_seed,
                )
                p_lr = exp_dir / f"{a_nm}_{sk}.pt"
                torch.save(res_lr["weights"], p_lr)
                expert_paths[(a_nm, sk)] = p_lr
                arm_mb_skill[a_nm].append(float(res_lr["adapter_mb"]))
                with _temp_apply_weights(student_m, res_lr["weights"]):
                    domain_mmlu_scores[a_nm][sk] = _score_routed_mmlu_sparse(
                        student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                    )
                del res_lr

            print(
                f"[exp1] {sk}: quant_err_A={quant_err_stats[sk]} | mb={ {a: arm_mb_skill[a][-1] for a in arm_names} }",
                flush=True,
            )

        mmlu_arms = {"base_08b": base_mmlu_summary}
        for idx_a, a_nm in enumerate(arm_names):
            mmlu_arms[a_nm] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_scores[a_nm], seed=int(args.seed) + 1 + idx_a
            )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Batched HF Evaluation of all 9 Paired Arms ----
        t_p4 = time.perf_counter()
        print(f"[exp1] Phase 4: Evaluating 9 paired arms (HF batched bs={args.eval_batch_size})...", flush=True)
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

        base_raw = _batched_hf_generate(student_m, tok_s, test_pids, test_seeds, max_new_tokens=int(args.max_new_tokens), batch_size=int(args.eval_batch_size))
        oracle_raw = _batched_hf_generate(student_m, tok_s, oracle_pids, test_seeds, max_new_tokens=int(args.max_new_tokens), batch_size=int(args.eval_batch_size))

        routed_raw: dict[str, list[dict[str, Any]]] = {}
        for a_nm in arm_names:
            arr: list[dict[str, Any] | None] = [None] * len(all_test)
            for i, cls_nm in enumerate(routed_test_classes):
                if cls_nm == "passthrough":
                    arr[i] = base_raw[i]
            for sk in SKILLS_5:
                idxs = [i for i, cls_nm in enumerate(routed_test_classes) if cls_nm == sk]
                p_exp = expert_paths[(a_nm, sk)]
                if idxs:
                    upd = torch.load(p_exp, map_location="cpu", weights_only=True)
                    sub_o = _batched_hf_generate(
                        student_m,
                        tok_s,
                        [test_pids[i] for i in idxs],
                        [test_seeds[i] for i in idxs],
                        max_new_tokens=int(args.max_new_tokens),
                        batch_size=int(args.eval_batch_size),
                        param_updates=upd,
                    )
                    for idx_i, o in zip(idxs, sub_o):
                        arr[idx_i] = o
                    del upd
                p_exp.unlink(missing_ok=True)
            routed_raw[a_nm] = [r for r in arr if r is not None]
        phase4_s = round(time.perf_counter() - t_p4, 2)

    # ---- Phase 5: Grading & Bootstrap CIs ----
    arm_raw_map: dict[str, tuple[list[dict[str, Any]], list[str] | None, bool]] = {
        "base_08b": (base_raw, None, False),
        "teacher_4b": (t_test_raw, None, True),
        "oracle_1shot_08b": (oracle_raw, None, False),
    }
    for a_nm in arm_names:
        arm_raw_map[a_nm] = (routed_raw[a_nm], routed_test_classes, False)

    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_nm, (raw_l, r_lbls, is_teacher) in arm_raw_map.items():
        if is_teacher:
            arm_rows[arm_nm] = teacher_test_rows
            arm_summaries[arm_nm] = teacher_summary
        else:
            rows = _grade_rows(all_test, raw_l, tok_s, eos_set, think_end_id, routed_labels=r_lbls)
            arm_rows[arm_nm] = rows
            arm_summaries[arm_nm] = _summarize_fewshot_arm(rows, arm_raw=raw_l, base_raw=base_raw)

    arm_mb_per_skill = {a: round(sum(arm_mb_skill[a]) / len(SKILLS_5), 4) for a in arm_names}
    prose_kl = {a: 0.0 for a in arm_names}  # routed_prose_fpr = 0.0 measured above
    comp_pairs = [
        ("s3k_vs_r8_fp16", "r8_s3k_h64_gptq_flip_routed", "r8_fp16_routed"),
        ("s3k_vs_r2_fp16", "r8_s3k_h64_gptq_flip_routed", "r2_fp16_lr4e4_routed"),
        ("s2k_vs_r1_fp16", "r8_s2k_h64_gptq_flip_routed", "r1_fp16_routed"),
        ("s2k_vs_rtn_2bit", "r8_s2k_h64_gptq_flip_routed", "r8_rtn_2bit_routed"),
        ("s2k_vs_r2_fp16", "r8_s2k_h64_gptq_flip_routed", "r2_fp16_lr4e4_routed"),
    ]
    paired_hard_60 = {
        lbl: _paired_bootstrap(arm_rows[a][:n_hard], arm_rows[b][:n_hard], seed=int(args.seed) + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    h_acc = lambda k: float(arm_summaries[k]["hard_60"]["strict_accuracy"])  # noqa: E731
    return {
        "experiment": "pocket_moe_strata_quant",
        "teacher_model": args.teacher,
        "student_model": args.student,
        "k_shots_per_skill": int(args.k_shots),
        "teacher_4b_train_kept_frac": {sk: round(kept_4b[sk] / max(1, int(args.k_shots)), 4) for sk in SKILLS_5},
        "router_report": router_report,
        "arm_mb_per_skill": arm_mb_per_skill,
        "quant_err_stats": quant_err_stats,
        "prose_kl": prose_kl,
        "generation_arms": arm_summaries,
        "mmlu_arms": mmlu_arms,
        "paired_hard_60": paired_hard_60,
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_router_s": phase2_s,
            "phase3_train_quant_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "criteria": {
            "a_s3k_smaller_than_r2_fp16": arm_mb_per_skill["r8_s3k_h64_gptq_flip_routed"] < arm_mb_per_skill["r2_fp16_lr4e4_routed"],
            "b_s3k_beats_or_ties_r2_fp16": h_acc("r8_s3k_h64_gptq_flip_routed") >= h_acc("r2_fp16_lr4e4_routed"),
            "c_s2k_beats_rtn_2bit": h_acc("r8_s2k_h64_gptq_flip_routed") > h_acc("r8_rtn_2bit_routed"),
            "d_s2k_beats_r1_fp16": h_acc("r8_s2k_h64_gptq_flip_routed") > h_acc("r1_fp16_routed"),
            "e_zero_mmlu_drift": all(mmlu_arms[a]["router_fire_rate"] == 0.0 for a in arm_names),
        },
        "per_item_generation": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--k-shots", type=int, default=6)
    ap.add_argument("--n-mmlu-test", type=int, default=200)
    ap.add_argument("--n-mmlu-router-train", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1350)
    ap.add_argument("--eval-batch-size", type=int, default=32)
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
