"""Test B: Strata 3-Bit Base (1.05 GB) + Scale-QAT (+0 B) + Routed S2_K Recovery Patch (2-4 MB) on Qwen2.5-3B-Instruct.

Tests whether a coherent 3-bit Strata base (`S3_K` + `H64 + Hessian clipping + Block-LDL + 64-flip`)
can be restored toward BF16 math performance via pure self-distilled, Layer-18 routed `S2_K` LoRA patches:
  1. `base_bf16`                   : Reference BF16 model (`16.0 b/w`)
  2. `s3k_uniform_raw`             : Uniform `S3_K` + H64 + Block-LDL + 64-flip (`3.1875 b/w`, `1054.27 MB`)
  3. `s3k_plus_scale_qat`          : + `Scale-QAT` over 73 RMSNorms (`+0 B`)
  4. `s3k_sqat_plus_r4_fp16_routed`: Frozen `s3k_plus_scale_qat` + self-distilled `r=4` LoRA (`~14.3 MB`) + L18 router
  5. `s3k_sqat_plus_r4_s2k_routed` : Arm 4 compressed post-hoc to `S2_K` (`~2.04 MB`) + L18 router
  6. `s3k_sqat_plus_r4_s2k_unrouted`: Arm 5 always-on (unrouted control for MMLU/PPL/prose protection)
  7. `s3k_sqat_plus_r4_rtn2_routed`: Arm 4 compressed with naive 2-bit RTN (`~2.04 MB`) + L18 router
  8. `s3k_sqat_plus_r8_s2k_routed` : Self-distilled `r=8` (`~28.5 MB`) compressed post-hoc to `S2_K` (`~4.09 MB`) + L18 router

CPU smoke test::
    python scripts/experiment_strata_patch_recovery_dense.py --smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
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

import eval_capability_and_boundary_lfm3 as ec  # noqa: E402
from experiment_pocket_moe_ceiling import eval_healthy_drift, fit_ridge_router, predict_ridge_router  # noqa: E402
from experiment_pocket_moe_controls import _paired_bootstrap  # noqa: E402
from experiment_pocket_moe_math500 import (  # noqa: E402
    PROSE_HELDOUT,
    PROSE_ROUTER_TRAIN,
    _compose_routed_mmlu,
    _summarize_mmlu,
    _temp_apply_weights,
    format_mmlu_prompt_ids,
    load_mmlu_split,
)
from experiment_pocket_moe_strata_quant import _train_regularized_lora_with_factors  # noqa: E402
from experiment_reasoning_distill_pilot import _atomic_write_json, _extract_prompt_hidden_vecs  # noqa: E402
from experiment_xray_strata_quant_dense import (  # noqa: E402
    _PROJ_SPECS,
    _eval_perplexity,
    _extract_gsm8k_num,
    _get_submod,
    _load_wikitext_chunks,
)
from wsi.capture import find_blocks  # noqa: E402
from wsi.strata_quant import (  # noqa: E402
    apply_h64_sym,
    capture_teacher_topk_targets,
    damped_cholesky_inv,
    eval_topk_kl_and_top1,
    gptq_flip_quantize_sk_torch,
    rtn_quantize_sk_torch,
    run_norm_scale_qat,
)
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402

ROUTER_CLASSES_RECOVERY: tuple[str, ...] = ("math", "pass_mmlu", "pass_prose")


def _fmt_chat_user(tok: Any, content: str) -> list[int]:
    msg = [{"role": "user", "content": content}]
    if getattr(tok, "chat_template", None):
        try:
            s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            return [int(x) for x in tok.encode(s, add_special_tokens=False)]
        except Exception:
            pass
    return [int(x) for x in tok.encode(content, add_special_tokens=True)]


def _fmt_chat_pair(tok: Any, user_txt: str, assistant_txt: str) -> list[int]:
    msgs = [{"role": "user", "content": user_txt}, {"role": "assistant", "content": assistant_txt}]
    if getattr(tok, "chat_template", None):
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            return [int(x) for x in tok.encode(s, add_special_tokens=False)]
        except Exception:
            pass
    return [int(x) for x in tok.encode(f"{user_txt}\n{assistant_txt}", add_special_tokens=True)]


def _gsm8k_prompt_text(question: str) -> str:
    return f"Solve step by step and put the final integer answer in \\boxed{{}}.\n\n{question}"


def _predict_binary_recovery_router(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> list[str]:
    preds = predict_ridge_router(X, W, b, ROUTER_CLASSES_RECOVERY)
    return ["math" if p == "math" else "passthrough" for p in preds]


@torch.no_grad()
def _score_abcd_per_item(
    model: nn.Module,
    prompts_ids: list[list[int]],
    gold_idxs: list[int],
    letter_ids: dict[str, list[int]],
    pad_id: int,
    dev: torch.device,
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Return per-item ABCD probabilities, preds, correctness, and masses compatible with `_compose_routed_mmlu`."""
    letters = ("A", "B", "C", "D")
    letter_tensors = [torch.tensor(letter_ids.get(ch) or [], dtype=torch.long, device=dev) for ch in letters]
    probs_list: list[torch.Tensor] = []
    preds: list[int] = []
    correct: list[bool] = []
    masses: list[float] = []
    bs = max(1, int(batch_size))
    n = len(prompts_ids)
    for b0 in range(0, n, bs):
        sub_p = prompts_ids[b0 : b0 + bs]
        max_l = max(len(p) for p in sub_p)
        inp = torch.full((len(sub_p), max_l), pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((len(sub_p), max_l), dtype=torch.long, device=dev)
        for ri, p in enumerate(sub_p):
            inp[ri, max_l - len(p) :] = torch.tensor(p, dtype=torch.long, device=dev)
            mask[ri, max_l - len(p) :] = 1
        out = model(input_ids=inp, attention_mask=mask, use_cache=False)
        last_lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        cols = [
            torch.logsumexp(last_lp[:, t_ids], dim=-1) if t_ids.numel() > 0 else torch.full((len(sub_p),), -1e9, device=dev)
            for t_ids in letter_tensors
        ]
        abcd_lp = torch.stack(cols, dim=-1)
        abcd_p = torch.exp(abcd_lp)
        mass = abcd_p.sum(dim=-1)
        norm_p = abcd_p / mass.clamp_min(1e-12).unsqueeze(-1)
        pr = norm_p.argmax(dim=-1).tolist()
        for off, p_idx in enumerate(pr):
            gi = b0 + off
            probs_list.append(norm_p[off].cpu())
            preds.append(int(p_idx))
            correct.append(int(p_idx) == int(gold_idxs[gi]))
            masses.append(float(mass[off].item()))
        del inp, mask, out, last_lp, abcd_lp, abcd_p
    return {
        "probs": torch.stack(probs_list, dim=0) if probs_list else torch.empty((0, 4)),
        "preds": preds,
        "correct": correct,
        "choice_masses": masses,
    }


@torch.no_grad()
def _eval_gsm8k_greedy_rows(
    model: nn.Module,
    tok: Any,
    items: list[dict[str, Any]],
    prompts_ids: list[list[int]],
    dev: torch.device,
    *,
    max_new_tokens: int = 512,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0))
    rows: list[dict[str, Any]] = []
    bs = max(1, int(batch_size))
    for b0 in range(0, len(items), bs):
        sub_p = prompts_ids[b0 : b0 + bs]
        max_l = max(len(p) for p in sub_p)
        inp = torch.full((len(sub_p), max_l), pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((len(sub_p), max_l), dtype=torch.long, device=dev)
        for ri, p in enumerate(sub_p):
            inp[ri, max_l - len(p) :] = torch.tensor(p, dtype=torch.long, device=dev)
            mask[ri, max_l - len(p) :] = 1
        gen = model.generate(
            input_ids=inp,
            attention_mask=mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_ids,
        )
        for ri in range(len(sub_p)):
            it = items[b0 + ri]
            txt = tok.decode(gen[ri, max_l:].tolist(), skip_special_tokens=True)
            pred = _extract_gsm8k_num(txt)
            gold = _extract_gsm8k_num(it["answer"].split("####")[-1])
            ok = bool(pred is not None and gold is not None and pred == gold)
            rows.append({
                "strict_correct": ok,
                "pred": pred,
                "gold": gold,
                "text": txt,
            })
        del inp, mask, gen
    return rows


def _summarize_gsm8k_rows(rows: list[dict[str, Any]], *, n_first: int | None = None) -> dict[str, Any]:
    sub = rows[: int(n_first)] if n_first is not None else rows
    nc = sum(1 for r in sub if r["strict_correct"])
    return {"accuracy": round(nc / max(1, len(sub)), 4), "n_correct": nc, "n": len(sub)}


@torch.no_grad()
def _quantize_base_uniform_s3k(
    model: nn.Module,
    blocks: Sequence[nn.Module],
    cal_seqs: list[list[int]],
    dev: torch.device,
) -> tuple[int, int]:
    """Uniformly quantize all `_PROJ_SPECS` linear weights on `blocks` in-place to `S3_K` using streamed Hessians."""
    n_layers = len(blocks)
    total_params = 0
    total_wire_bytes = 0
    chunk_size = 6
    for c0 in range(0, n_layers, chunk_size):
        c1 = min(n_layers, c0 + chunk_size)
        chunk_hess: dict[tuple[int, str], torch.Tensor] = {}
        chunk_cnt: dict[tuple[int, str], int] = {}
        handles = []

        def _make_hook(li_v: int, st_nm: str, dim: int):
            def _hook(_mod: nn.Module, inp: tuple[torch.Tensor, ...]) -> None:
                x = inp[0].detach().reshape(-1, dim).float()
                h_inc = x.T @ x
                k = (li_v, st_nm)
                if k not in chunk_hess:
                    chunk_hess[k] = h_inc
                    chunk_cnt[k] = int(x.shape[0])
                else:
                    chunk_hess[k].add_(h_inc)
                    chunk_cnt[k] += int(x.shape[0])
            return _hook

        for li in range(c0, c1):
            blk = blocks[li]
            for mod_p, st_nm in [("self_attn.q_proj", "qkv"), ("self_attn.o_proj", "o"), ("mlp.gate_proj", "gu"), ("mlp.down_proj", "dn")]:
                m_lin = _get_submod(blk, mod_p)
                handles.append(m_lin.register_forward_pre_hook(_make_hook(li, st_nm, int(m_lin.in_features))))

        for seq in cal_seqs:
            ids = torch.tensor(seq, dtype=torch.long, device=dev).unsqueeze(0)
            model(input_ids=ids, use_cache=False)
            del ids
        for h in handles:
            h.remove()

        for li in range(c0, c1):
            blk = blocks[li]
            pre_by_stream: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            H_by_stream: dict[str, torch.Tensor] = {}
            for st_nm in ("qkv", "o", "gu", "dn"):
                H_raw = chunk_hess.pop((li, st_nm)) / float(max(1, chunk_cnt[(li, st_nm)]))
                H_by_stream[st_nm] = H_raw
                pre_by_stream[st_nm] = (
                    damped_cholesky_inv(apply_h64_sym(H_raw), damp_frac=0.01) if int(H_raw.shape[0]) % 256 == 0 else None
                )

            for proj, st_nm in _PROJ_SPECS:
                mod = _get_submod(blk, proj)
                w = mod.weight.data
                total_params += int(w.numel())
                wq, wb, _ = gptq_flip_quantize_sk_torch(
                    w, H_by_stream[st_nm], "S3_K", use_h64=True, max_flips=64, precomputed_H=pre_by_stream[st_nm]
                )
                mod.weight.data.copy_(wq.to(mod.weight.dtype))
                total_wire_bytes += int(wb)
                del wq
            del pre_by_stream, H_by_stream
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    return total_params, total_wire_bytes


@torch.no_grad()
def _compress_factors_chunked_streams(
    model: nn.Module,
    blocks: Sequence[nn.Module],
    cal_samples: list[dict[str, Any]],
    variant_specs: list[tuple[str, dict[str, Any], str, str]],
    dev: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, int], dict[str, float]]:
    """Quantize `(lora_A, lora_B)` factors over quantized base by capturing 4-stream Hessians in 12-layer chunks.

    Each element of `variant_specs` is `(arm_name, raw_factors, code_name, method)`.
    Returns `(merged_weights_by_arm, wire_bytes_by_arm, a_hess_err_by_arm)`.
    """
    n_layers = len(blocks)
    pref_to_li: dict[str, int] = {}
    first_factors = variant_specs[0][1]
    for full_nm in first_factors.keys():
        # Match `model.layers.<li >.<sub>.weight`
        parts = full_nm.split(".")
        for p_idx, tok_p in enumerate(parts):
            if tok_p.isdigit() and p_idx >= 1 and parts[p_idx - 1] == "layers":
                pref_to_li[full_nm] = int(tok_p)
                break

    proj_to_stream = {proj: st_nm for proj, st_nm in _PROJ_SPECS}
    merged_out: dict[str, dict[str, torch.Tensor]] = {arm_nm: {} for arm_nm, *_ in variant_specs}
    wire_bytes_out: dict[str, int] = {arm_nm: 0 for arm_nm, *_ in variant_specs}
    err_out: dict[str, float] = {arm_nm: 0.0 for arm_nm, *_ in variant_specs}

    chunk_size = 12
    for c0 in range(0, n_layers, chunk_size):
        c1 = min(n_layers, c0 + chunk_size)
        chunk_hess: dict[tuple[int, str], torch.Tensor] = {}
        chunk_cnt: dict[tuple[int, str], int] = {}
        handles = []

        def _make_hook(li_v: int, st_nm: str, dim: int):
            def _hook(_mod: nn.Module, inp: tuple[torch.Tensor, ...]) -> None:
                x = inp[0].detach().reshape(-1, dim).float()
                h_inc = x.T @ x
                k = (li_v, st_nm)
                if k not in chunk_hess:
                    chunk_hess[k] = h_inc
                    chunk_cnt[k] = int(x.shape[0])
                else:
                    chunk_hess[k].add_(h_inc)
                    chunk_cnt[k] += int(x.shape[0])
            return _hook

        for li in range(c0, c1):
            blk = blocks[li]
            for mod_p, st_nm in [("self_attn.q_proj", "qkv"), ("self_attn.o_proj", "o"), ("mlp.gate_proj", "gu"), ("mlp.down_proj", "dn")]:
                m_lin = _get_submod(blk, mod_p)
                handles.append(m_lin.register_forward_pre_hook(_make_hook(li, st_nm, int(m_lin.in_features))))

        for sp in cal_samples:
            ids_dev = torch.tensor(sp["ids"], dtype=torch.long, device=dev).unsqueeze(0)
            model(input_ids=ids_dev, use_cache=False)
            del ids_dev
        for h in handles:
            h.remove()

        for li in range(c0, c1):
            pre_by_stream: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            H_by_stream: dict[str, torch.Tensor] = {}
            for st_nm in ("qkv", "o", "gu", "dn"):
                H_raw = chunk_hess.pop((li, st_nm)) / float(max(1, chunk_cnt[(li, st_nm)]))
                H_by_stream[st_nm] = H_raw
                pre_by_stream[st_nm] = (
                    damped_cholesky_inv(apply_h64_sym(H_raw), damp_frac=0.01) if int(H_raw.shape[0]) % 256 == 0 else None
                )

            for proj, st_nm in _PROJ_SPECS:
                H_a = H_by_stream[st_nm]
                pre_a = pre_by_stream[st_nm]
                for arm_nm, raw_factors, code_name, method in variant_specs:
                    full_nm = next(k for k in raw_factors if pref_to_li.get(k) == li and k.endswith(f".{proj}.weight"))
                    A_cpu, B_cpu, scale, base_w_cpu, orig_dtype = raw_factors[full_nm]
                    A_d = A_cpu.to(device=dev, dtype=torch.float32)
                    Bt_d = B_cpu.T.contiguous().to(device=dev, dtype=torch.float32)
                    H_b = torch.eye(Bt_d.shape[1], device=dev, dtype=torch.float32)
                    if method == "rtn":
                        A_q, wb_a, err_a = rtn_quantize_sk_torch(A_d, code_name, use_clipping=False, use_h64=False, H=H_a)
                        Bt_q, wb_b, _ = rtn_quantize_sk_torch(Bt_d, code_name, use_clipping=False, use_h64=False, H=H_b)
                    elif method == "strata":
                        A_q, wb_a, err_a = gptq_flip_quantize_sk_torch(A_d, H_a, code_name, use_h64=True, max_flips=64, precomputed_H=pre_a)
                        Bt_q, wb_b, _ = gptq_flip_quantize_sk_torch(Bt_d, H_b, code_name, use_h64=True, max_flips=64)
                    else:
                        raise ValueError(method)
                    wire_bytes_out[arm_nm] += int(wb_a + wb_b)
                    err_out[arm_nm] += float(err_a or 0.0)
                    delta_w = (float(scale) * (Bt_q.T.float() @ A_q.float())).cpu()
                    merged_out[arm_nm][full_nm] = (base_w_cpu + delta_w).to(orig_dtype)
                    del A_d, Bt_d, H_b, A_q, Bt_q, delta_w
            del pre_by_stream, H_by_stream
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    return merged_out, wire_bytes_out, err_out


def _eval_routed_ppl_and_kl(
    model: nn.Module,
    wt_ppl_seqs: list[list[int]],
    wt_ppl_flags: list[bool],
    te_kl_recs: list[dict[str, Any]],
    te_kl_flags: list[bool],
    base_ppl_nll: tuple[float, int],
    base_kl_stats: dict[str, float],
    param_updates: dict[str, torch.Tensor] | None,
    dev: torch.device,
) -> tuple[float, float, float]:
    """Compute exact WikiText-2 PPL & Top-256 KL/Top-1 reusing `base` when no items fire."""
    if not param_updates:
        tot_nll, tot_tok = base_ppl_nll
        return (
            round(float(math.exp(min(20.0, tot_nll / max(1, tot_tok)))), 4),
            round(float(base_kl_stats["kl_mean"]), 6),
            round(float(base_kl_stats["top1_agreement"]), 4),
        )

    fired_ppl = [s for s, f in zip(wt_ppl_seqs, wt_ppl_flags) if f]
    fired_kl = [r for r, f in zip(te_kl_recs, te_kl_flags) if f]

    if not fired_ppl:
        tot_nll, tot_tok = base_ppl_nll
        ppl = round(float(math.exp(min(20.0, tot_nll / max(1, tot_tok)))), 4)
    else:
        with _temp_apply_weights(model, param_updates):
            if all(wt_ppl_flags):
                ppl = _eval_perplexity(model, wt_ppl_seqs, dev)
            else:
                # Re-evaluate fired subset + combine with exact passthrough NLL
                nll_f = 0.0
                tok_f = 0
                with torch.no_grad():
                    for s in fired_ppl:
                        ids = torch.tensor(s, dtype=torch.long, device=dev).unsqueeze(0)
                        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
                        nll_f += float(F.cross_entropy(logits, ids[0, 1:], reduction="sum").item())
                        tok_f += int(ids.shape[1] - 1)
                        del ids, logits
                # Recompute passthrough exact NLL on base weights (outside _temp_apply_weights) -> passed via loop below
                nll_p, tok_p = 0.0, 0
                ppl = -1.0  # signal below
        if ppl < 0:
            nll_p, tok_p = 0.0, 0
            with torch.no_grad():
                for s, f in zip(wt_ppl_seqs, wt_ppl_flags):
                    if not f:
                        ids = torch.tensor(s, dtype=torch.long, device=dev).unsqueeze(0)
                        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
                        nll_p += float(F.cross_entropy(logits, ids[0, 1:], reduction="sum").item())
                        tok_p += int(ids.shape[1] - 1)
                        del ids, logits
            ppl = round(float(math.exp(min(20.0, (nll_f + nll_p) / max(1, tok_f + tok_p)))), 4)

    if not fired_kl:
        kl_mean = float(base_kl_stats["kl_mean"])
        top1 = float(base_kl_stats["top1_agreement"])
    elif all(te_kl_flags):
        with _temp_apply_weights(model, param_updates):
            r_k = eval_topk_kl_and_top1(model, te_kl_recs, dev)
        kl_mean, top1 = float(r_k["kl_mean"]), float(r_k["top1_agreement"])
    else:
        with _temp_apply_weights(model, param_updates):
            r_f = eval_topk_kl_and_top1(model, fired_kl, dev)
        pass_kl = [r for r, f in zip(te_kl_recs, te_kl_flags) if not f]
        r_p = eval_topk_kl_and_top1(model, pass_kl, dev)
        nf, np_ = sum(int(r["tokens"].numel()) - 1 for r in fired_kl), sum(int(r["tokens"].numel()) - 1 for r in pass_kl)
        tot = max(1, nf + np_)
        kl_mean = (r_f["kl_mean"] * nf + r_p["kl_mean"] * np_) / tot
        top1 = (r_f["top1_agreement"] * nf + r_p["top1_agreement"] * np_) / tot

    return ppl, round(kl_mean, 6), round(top1, 4)


def _print_summary(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return
    print("\n=== TEST B: STRATA 3-BIT BASE + SCALE-QAT + ROUTED S2_K RECOVERY (Qwen2.5-3B) ===", flush=True)
    print("Router:", payload.get("router_report"), flush=True)
    for arm_nm, st in payload.get("arms", {}).items():
        mb_p = st.get("patch_mb", 0.0)
        g2 = st["gsm8k_200"]
        g1 = st["gsm8k_100"]
        m = st["mmlu_200"]
        print(
            f"  {arm_nm:30s} | +{mb_p:5.2f}MB | PPL={st['wikitext2_ppl']:6.3f} | KL={st['kl_mean']:6.4f} | "
            f"Top1={st['top1_agreement']*100:5.2f}% | MMLU={m['accuracy']*100:5.1f}% | "
            f"GSM8K_200={g2['accuracy']*100:5.1f}% ({g2['n_correct']:3d}/200) | GSM8K_100={g1['accuracy']*100:5.1f}% | "
            f"pkl={st.get('prose_kl', 0.0):.5f}",
            flush=True,
        )
    print("\n=== PAIRED BOOTSTRAP CIs (GSM8K-200) ===", flush=True)
    for k, v in payload.get("paired_gsm8k_200", {}).items():
        print(
            f"  {k:32s}: diff={v['diff']*100:+5.1f} pts CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] "
            f"(w={v['wins']}/l={v['losses']})",
            flush=True,
        )
    print("\nCriteria:", payload.get("criteria"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_gsm8k_200"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification on tiny 4-layer Qwen2: S3_K base + Scale-QAT + quantized-base LoRA + S2_K."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, blocks = find_blocks(model)

    cal_seqs = [[2, 5, 8, 12, 19, 25, 31, 40], [3, 7, 11, 15, 22, 28, 35, 44]]
    tr_recs = capture_teacher_topk_targets(model, cal_seqs[:1], "cpu", topk=16)
    dv_recs = capture_teacher_topk_targets(model, cal_seqs[1:], "cpu", topk=16)

    tot_p, s3_bytes = _quantize_base_uniform_s3k(model, blocks, cal_seqs, torch.device("cpu"))
    assert tot_p > 0 and s3_bytes > 0
    qat_res = run_norm_scale_qat(model, tr_recs, dv_recs, device="cpu", epochs=2, lr=1e-2, grad_accum=1)
    qbase_snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Train recovery r=4 LoRA over frozen quantized base
    tr_samples = []
    for rec in tr_recs + dv_recs:
        s = rec["tokens"].tolist()
        tr_samples.append({
            "ids": s,
            "lo": 2,
            "hi": len(s) - 1,
            "topk_ids": rec["top_idx"][2 : len(s) - 1].to(torch.int32).cpu(),
            "topk_logits": rec["top_logp"][2 : len(s) - 1].to(torch.float16).cpu(),
        })
    res4 = _train_regularized_lora_with_factors(
        model, [0, 1, 2, 3], tr_samples, rank=4, alpha_scale=8.0, total_steps=4, grad_accum=2, lr_val=1e-2, seed=42
    )
    assert all(torch.equal(v.cpu(), qbase_snap[k]) for k, v in model.state_dict().items())

    merged, wbytes, errs = _compress_factors_chunked_streams(
        model,
        blocks,
        tr_samples,
        [("r4_s2k", res4["factors"], "S2_K", "strata"), ("r4_rtn2", res4["factors"], "S2_K", "rtn")],
        torch.device("cpu"),
    )
    assert wbytes["r4_s2k"] == wbytes["r4_rtn2"]
    assert errs["r4_s2k"] < errs["r4_rtn2"]
    with _temp_apply_weights(model, merged["r4_s2k"]):
        assert not torch.equal(blocks[0].mlp.down_proj.weight.cpu(), qbase_snap["model.layers.0.mlp.down_proj.weight"])
    assert all(torch.equal(v.cpu(), qbase_snap[k]) for k, v in model.state_dict().items())

    return {
        "mode": "smoke",
        "s3_base_bytes": s3_bytes,
        "r4_s2k_patch_bytes": wbytes["r4_s2k"],
        "err_rtn2": round(errs["r4_rtn2"], 6),
        "err_strata_s2k": round(errs["r4_s2k"], 6),
        "dev_kl_best": qat_res["dev_kl_best"],
        "passed": True,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    torch.manual_seed(int(args.seed))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    all_layers = list(range(n_layers))
    router_layer = n_layers // 2

    # ---- Datasets ----
    wt = _load_wikitext_chunks(tok, seq_len=512, n_train=48, n_dev=8, n_test_ppl=32, n_test_kl=16)
    mmlu_test, mmlu_router_train = load_mmlu_split(n_test=200, n_router_train=100, seed=int(args.seed))
    mmlu_pids = [format_mmlu_prompt_ids(tok, it) for it in mmlu_test]
    mmlu_gold = [int(it["gold_idx"]) for it in mmlu_test]
    mmlu_r_pids = [format_mmlu_prompt_ids(tok, it) for it in mmlu_router_train]
    letter_ids = ec._resolve_mmlu_letter_ids(tok)

    gsm_train = list(load_dataset("openai/gsm8k", "main", split="train"))[: int(args.n_gsm8k_train)]
    gsm_test = list(load_dataset("openai/gsm8k", "main", split="test"))[: int(args.n_gsm8k_test)]
    gsm_train_pids = [_fmt_chat_user(tok, _gsm8k_prompt_text(it["question"])) for it in gsm_train]
    gsm_test_pids = [_fmt_chat_user(tok, _gsm8k_prompt_text(it["question"])) for it in gsm_test]

    prose_r_pids = [_fmt_chat_user(tok, p) for p in PROSE_ROUTER_TRAIN]
    prose_held_pids = [_fmt_chat_user(tok, p) for p in PROSE_HELDOUT[:16]]

    # ---- Phase 1: BF16 Base Eval + Self-Distillation Targets ----
    t_p1 = time.perf_counter()
    print(f"[testB] Phase 1: BF16 Base Eval + Top-256 KL targets + Self-Distillation on GSM8K-train ({len(gsm_train)})...", flush=True)
    tr_recs = capture_teacher_topk_targets(model, wt["train"], dev, topk=256)
    dv_recs = capture_teacher_topk_targets(model, wt["dev"], dev, topk=256)
    te_recs = capture_teacher_topk_targets(model, wt["test_kl"], dev, topk=256)

    def _compute_prose_logp(prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor | None], float]:
        seqs = _to_seq_tensors([_fmt_chat_pair(tok, pr, "Here is a concise explanation.") for pr in prompts], torch.device("cpu"))
        logps: list[torch.Tensor | None] = []
        nll_s, cnt = 0.0, 0
        with torch.no_grad():
            for ids in seqs:
                logits = _tail_logits(model, ids.to(dev), 0)[:-1].float()
                lp = F.log_softmax(logits, dim=-1)
                nll_s += float(F.nll_loss(lp, ids[0, 1:].to(dev), reduction="sum"))
                cnt += int(ids.shape[1] - 1)
                logps.append(lp.to(dtype=torch.bfloat16, device="cpu"))
        return seqs, logps, nll_s / max(1, cnt)

    prose_full_ids, base_prose_logp, base_prose_nll = _compute_prose_logp(PROSE_HELDOUT[:16])
    anchor_full_ids, base_anchor_logp, _ = _compute_prose_logp(PROSE_ROUTER_TRAIN)

    base_ppl = _eval_perplexity(model, wt["test_ppl"], dev)
    base_mmlu_raw = _score_abcd_per_item(model, mmlu_pids, mmlu_gold, letter_ids, pad_id, dev)
    base_mmlu_sum = _summarize_mmlu(mmlu_test, base_mmlu_raw, seed=int(args.seed))
    base_gsm_rows = _eval_gsm8k_greedy_rows(model, tok, gsm_test, gsm_test_pids, dev, batch_size=int(args.eval_batch_size))

    # Self-distillation: BF16 greedy generation on GSM8K train + Top-32 logits
    train_gen_rows = _eval_gsm8k_greedy_rows(model, tok, gsm_train, gsm_train_pids, dev, batch_size=int(args.eval_batch_size))
    self_distill_samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for it, pids, g_row in zip(gsm_train, gsm_train_pids, train_gen_rows):
            if not g_row["strict_correct"] or not g_row["text"].strip():
                continue
            full_ids = _fmt_chat_pair(tok, _gsm8k_prompt_text(it["question"]), g_row["text"].strip())
            lo = max(0, len(pids) - 1)
            hi = len(full_ids) - 1
            if hi <= lo:
                continue
            ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
            t_logits = _tail_logits(model, ids_dev, lo)[: hi - lo].float()
            vals, idx = t_logits.topk(32, dim=-1)
            self_distill_samples.append({
                "ids": full_ids,
                "lo": lo,
                "hi": hi,
                "topk_ids": idx.to(torch.int32).cpu(),
                "topk_logits": vals.to(torch.float16).cpu(),
            })
            del ids_dev, t_logits, vals, idx

    print(
        f"[testB] BASE BF16: PPL={base_ppl} MMLU={base_mmlu_sum['accuracy']*100:.1f}% "
        f"GSM8K_200={_summarize_gsm8k_rows(base_gsm_rows)['accuracy']*100:.1f}% | "
        f"kept self-distill traces={len(self_distill_samples)}/{len(gsm_train)}",
        flush=True,
    )
    phase1_s = round(time.perf_counter() - t_p1, 2)

    arms_out: dict[str, Any] = {
        "base_bf16": {
            "base_bits_per_weight": 16.0,
            "patch_mb": 0.0,
            "wikitext2_ppl": base_ppl,
            "kl_mean": 0.0,
            "top1_agreement": 1.0,
            "mmlu_200": base_mmlu_sum,
            "gsm8k_200": _summarize_gsm8k_rows(base_gsm_rows),
            "gsm8k_100": _summarize_gsm8k_rows(base_gsm_rows, n_first=100),
            "prose_kl": 0.0,
        }
    }
    gsm_rows_by_arm: dict[str, list[dict[str, Any]]] = {"base_bf16": base_gsm_rows}

    # ---- Phase 2: Uniform Strata S3_K 3-Bit Base ----
    t_p2 = time.perf_counter()
    print("[testB] Phase 2: Quantizing all 36L x 7 projs uniformly to Strata S3_K (H64+BlockLDL+64flip)...", flush=True)
    total_params, base_wire_bytes = _quantize_base_uniform_s3k(model, blocks, wt["train"], dev)
    base_bpw = round((base_wire_bytes * 8.0) / max(1, total_params), 4)
    base_mb = round(base_wire_bytes / (1024**2), 2)

    raw_ppl = _eval_perplexity(model, wt["test_ppl"], dev)
    raw_kl = eval_topk_kl_and_top1(model, te_recs, dev)
    raw_mmlu = _score_abcd_per_item(model, mmlu_pids, mmlu_gold, letter_ids, pad_id, dev)
    raw_mmlu_sum = _summarize_mmlu(mmlu_test, raw_mmlu, base_score=base_mmlu_raw, fire_flags=[False] * len(mmlu_test), seed=int(args.seed) + 1)
    raw_gsm_rows = _eval_gsm8k_greedy_rows(model, tok, gsm_test, gsm_test_pids, dev, batch_size=int(args.eval_batch_size))
    raw_prose_kl = round(float(eval_healthy_drift(model, prose_full_ids, base_prose_logp, base_prose_nll)["healthy_kl"]), 6)

    arms_out["s3k_uniform_raw"] = {
        "base_bits_per_weight": base_bpw,
        "base_wire_mb": base_mb,
        "patch_mb": 0.0,
        "wikitext2_ppl": raw_ppl,
        "kl_mean": round(float(raw_kl["kl_mean"]), 6),
        "top1_agreement": round(float(raw_kl["top1_agreement"]), 4),
        "mmlu_200": raw_mmlu_sum,
        "gsm8k_200": _summarize_gsm8k_rows(raw_gsm_rows),
        "gsm8k_100": _summarize_gsm8k_rows(raw_gsm_rows, n_first=100),
        "prose_kl": raw_prose_kl,
    }
    gsm_rows_by_arm["s3k_uniform_raw"] = raw_gsm_rows
    print(
        f"[testB] s3k_uniform_raw ({base_bpw:.3f}b, {base_mb}MB): PPL={raw_ppl} MMLU={raw_mmlu_sum['accuracy']*100:.1f}% "
        f"GSM8K_200={arms_out['s3k_uniform_raw']['gsm8k_200']['accuracy']*100:.1f}%",
        flush=True,
    )
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: +0 B Scale-QAT over 73 RMSNorms ----
    t_p3 = time.perf_counter()
    print("[testB] Phase 3: Running +0 B Scale-QAT (73 RMSNorms, Top-256 KL)...", flush=True)
    qat_meta = run_norm_scale_qat(model, tr_recs, dv_recs, device=dev, epochs=2, lr=5e-4, grad_accum=4, seed=int(args.seed))
    sqat_ppl_nll = 0.0
    sqat_ppl_tok = 0
    with torch.no_grad():
        for s in wt["test_ppl"]:
            ids = torch.tensor(s, dtype=torch.long, device=dev).unsqueeze(0)
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
            sqat_ppl_nll += float(F.cross_entropy(logits, ids[0, 1:], reduction="sum").item())
            sqat_ppl_tok += int(ids.shape[1] - 1)
            del ids, logits
    sqat_ppl = round(float(math.exp(min(20.0, sqat_ppl_nll / max(1, sqat_ppl_tok)))), 4)
    sqat_kl = eval_topk_kl_and_top1(model, te_recs, dev)
    sqat_mmlu = _score_abcd_per_item(model, mmlu_pids, mmlu_gold, letter_ids, pad_id, dev)
    sqat_mmlu_sum = _summarize_mmlu(mmlu_test, sqat_mmlu, base_score=base_mmlu_raw, fire_flags=[False] * len(mmlu_test), seed=int(args.seed) + 2)
    sqat_gsm_rows = _eval_gsm8k_greedy_rows(model, tok, gsm_test, gsm_test_pids, dev, batch_size=int(args.eval_batch_size))
    sqat_prose_kl = round(float(eval_healthy_drift(model, prose_full_ids, base_prose_logp, base_prose_nll)["healthy_kl"]), 6)

    arms_out["s3k_plus_scale_qat"] = {
        "base_bits_per_weight": base_bpw,
        "base_wire_mb": base_mb,
        "patch_mb": 0.0,
        "wikitext2_ppl": sqat_ppl,
        "kl_mean": round(float(sqat_kl["kl_mean"]), 6),
        "top1_agreement": round(float(sqat_kl["top1_agreement"]), 4),
        "mmlu_200": sqat_mmlu_sum,
        "gsm8k_200": _summarize_gsm8k_rows(sqat_gsm_rows),
        "gsm8k_100": _summarize_gsm8k_rows(sqat_gsm_rows, n_first=100),
        "prose_kl": sqat_prose_kl,
        "scale_qat": qat_meta,
    }
    gsm_rows_by_arm["s3k_plus_scale_qat"] = sqat_gsm_rows
    print(
        f"[testB] s3k_plus_scale_qat: PPL={sqat_ppl} MMLU={sqat_mmlu_sum['accuracy']*100:.1f}% "
        f"GSM8K_200={arms_out['s3k_plus_scale_qat']['gsm8k_200']['accuracy']*100:.1f}%",
        flush=True,
    )
    phase3_s = round(time.perf_counter() - t_p3, 2)

    # ---- Phase 4: L18 Router on Frozen Quantized Base + Self-Distilled Recovery LoRAs ----
    t_p4 = time.perf_counter()
    print("[testB] Phase 4: Fitting L18 router on quantized base & training r=4 / r=8 recovery patches...", flush=True)
    r_train_pids = gsm_train_pids[:120] + mmlu_r_pids + prose_r_pids
    r_train_labels = ["math"] * min(120, len(gsm_train_pids)) + ["pass_mmlu"] * len(mmlu_r_pids) + ["pass_prose"] * len(prose_r_pids)
    W_r, b_r = fit_ridge_router(_extract_prompt_hidden_vecs(model, r_train_pids, router_layer), r_train_labels, ROUTER_CLASSES_RECOVERY, lam=1.0)

    routed_gsm_test = _predict_binary_recovery_router(_extract_prompt_hidden_vecs(model, gsm_test_pids, router_layer), W_r, b_r)
    routed_mmlu_test = _predict_binary_recovery_router(_extract_prompt_hidden_vecs(model, mmlu_pids, router_layer), W_r, b_r)
    routed_prose_test = _predict_binary_recovery_router(_extract_prompt_hidden_vecs(model, prose_held_pids, router_layer), W_r, b_r)
    routed_wt_ppl = _predict_binary_recovery_router(_extract_prompt_hidden_vecs(model, wt["test_ppl"], router_layer), W_r, b_r)
    routed_wt_kl = _predict_binary_recovery_router(_extract_prompt_hidden_vecs(model, wt["test_kl"], router_layer), W_r, b_r)

    router_report = {
        "router_layer": router_layer,
        "gsm8k_fire_rate": round(sum(p == "math" for p in routed_gsm_test) / len(gsm_test), 4),
        "mmlu_fire_rate": round(sum(p != "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_non_math_fire_rate": round(
            sum(p != "passthrough" for p, it in zip(routed_mmlu_test, mmlu_test) if not it["is_math_subject"])
            / max(1, sum(1 for it in mmlu_test if not it["is_math_subject"])),
            4,
        ),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_test) / len(prose_held_pids), 4),
        "wt_ppl_fire_rate": round(sum(p == "math" for p in routed_wt_ppl) / len(wt["test_ppl"]), 4),
        "wt_kl_fire_rate": round(sum(p == "math" for p in routed_wt_kl) / len(te_recs), 4),
    }
    print(f"[testB] Router: {router_report}", flush=True)

    # Train r=4 & r=8 on the frozen S3_K + Scale-QAT base (anchor KL vs quantized base's prose logprobs)
    _, sqat_prose_logp, sqat_prose_nll = _compute_prose_logp(PROSE_HELDOUT[:16])
    _, sqat_anchor_logp, _ = _compute_prose_logp(PROSE_ROUTER_TRAIN)

    torch.manual_seed(int(args.seed) + 4)
    res4 = _train_regularized_lora_with_factors(
        model,
        all_layers,
        self_distill_samples,
        rank=4,
        alpha_scale=8.0,
        total_steps=int(args.total_steps),
        grad_accum=int(args.grad_accum),
        lr_val=2.0e-4,
        weight_decay=0.05,
        lora_dropout=0.10,
        warmup_frac=0.15,
        distill_T=2.0,
        kl_weight=0.85,
        anchor_ids=anchor_full_ids,
        base_anchor_logp=sqat_anchor_logp,
        anchor_kl_weight=0.05,
        healthy_ids=prose_full_ids,
        base_healthy_logp=sqat_prose_logp,
        base_healthy_nll=sqat_prose_nll,
        seed=int(args.seed) + 4,
    )
    torch.manual_seed(int(args.seed) + 8)
    res8 = _train_regularized_lora_with_factors(
        model,
        all_layers,
        self_distill_samples,
        rank=8,
        alpha_scale=16.0,
        total_steps=int(args.total_steps),
        grad_accum=int(args.grad_accum),
        lr_val=2.0e-4,
        weight_decay=0.05,
        lora_dropout=0.10,
        warmup_frac=0.15,
        distill_T=2.0,
        kl_weight=0.85,
        anchor_ids=anchor_full_ids,
        base_anchor_logp=sqat_anchor_logp,
        anchor_kl_weight=0.05,
        healthy_ids=prose_full_ids,
        base_healthy_logp=sqat_prose_logp,
        base_healthy_nll=sqat_prose_nll,
        seed=int(args.seed) + 8,
    )

    merged_q, wbytes_q, errs_q = _compress_factors_chunked_streams(
        model,
        blocks,
        self_distill_samples[:64],
        [
            ("s3k_sqat_plus_r4_s2k", res4["factors"], "S2_K", "strata"),
            ("s3k_sqat_plus_r4_rtn2", res4["factors"], "S2_K", "rtn"),
            ("s3k_sqat_plus_r8_s2k", res8["factors"], "S2_K", "strata"),
        ],
        dev,
    )
    r4_fp16_w = res4.pop("weights")
    r4_fp16_mb = float(res4["adapter_mb"])
    del res4["factors"], res8["factors"], res8["weights"]
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    wt_ppl_flags = [p == "math" for p in routed_wt_ppl]
    wt_kl_flags = [p == "math" for p in routed_wt_kl]
    fired_gsm_idxs = [i for i, p in enumerate(routed_gsm_test) if p == "math"]

    def _eval_patch_arm(arm_nm: str, w_upd: dict[str, torch.Tensor], mb_patch: float, seed_off: int, *, routed: bool) -> None:
        with _temp_apply_weights(model, w_upd):
            patch_mmlu = _score_abcd_per_item(model, mmlu_pids, mmlu_gold, letter_ids, pad_id, dev)
            patch_prose_kl = float(eval_healthy_drift(model, prose_full_ids, base_prose_logp, base_prose_nll)["healthy_kl"])
            if routed:
                sub_gsm = _eval_gsm8k_greedy_rows(
                    model,
                    tok,
                    [gsm_test[i] for i in fired_gsm_idxs],
                    [gsm_test_pids[i] for i in fired_gsm_idxs],
                    dev,
                    batch_size=int(args.eval_batch_size),
                ) if fired_gsm_idxs else []
                arm_gsm = list(sqat_gsm_rows)
                for gi, gr in zip(fired_gsm_idxs, sub_gsm):
                    arm_gsm[gi] = gr
            else:
                arm_gsm = _eval_gsm8k_greedy_rows(model, tok, gsm_test, gsm_test_pids, dev, batch_size=int(args.eval_batch_size))

        if routed:
            ppl, kl_m, top1_m = _eval_routed_ppl_and_kl(
                model, wt["test_ppl"], wt_ppl_flags, te_recs, wt_kl_flags, (sqat_ppl_nll, sqat_ppl_tok), sqat_kl, w_upd, dev
            )
            mmlu_s = _compose_routed_mmlu(mmlu_test, routed_mmlu_test, sqat_mmlu, {"math": patch_mmlu}, seed=int(args.seed) + seed_off)
            p_kl = round(patch_prose_kl * router_report["prose_fpr"] + sqat_prose_kl * (1.0 - router_report["prose_fpr"]), 6)
        else:
            with _temp_apply_weights(model, w_upd):
                ppl = _eval_perplexity(model, wt["test_ppl"], dev)
                r_k = eval_topk_kl_and_top1(model, te_recs, dev)
                kl_m, top1_m = round(float(r_k["kl_mean"]), 6), round(float(r_k["top1_agreement"]), 4)
            mmlu_s = _summarize_mmlu(mmlu_test, patch_mmlu, base_score=sqat_mmlu, fire_flags=[True] * len(mmlu_test), seed=int(args.seed) + seed_off)
            p_kl = round(patch_prose_kl, 6)

        arms_out[arm_nm] = {
            "base_bits_per_weight": base_bpw,
            "base_wire_mb": base_mb,
            "patch_mb": round(float(mb_patch), 4),
            "total_wire_mb": round(base_mb + float(mb_patch), 2),
            "wikitext2_ppl": ppl,
            "kl_mean": kl_m,
            "top1_agreement": top1_m,
            "mmlu_200": mmlu_s,
            "gsm8k_200": _summarize_gsm8k_rows(arm_gsm),
            "gsm8k_100": _summarize_gsm8k_rows(arm_gsm, n_first=100),
            "prose_kl": p_kl,
        }
        gsm_rows_by_arm[arm_nm] = arm_gsm
        print(
            f"[testB] {arm_nm:30s} | +{mb_patch:5.2f}MB | PPL={ppl:6.3f} | MMLU={mmlu_s['accuracy']*100:5.1f}% | "
            f"GSM8K_200={arms_out[arm_nm]['gsm8k_200']['accuracy']*100:5.1f}%",
            flush=True,
        )

    _eval_patch_arm("s3k_sqat_plus_r4_fp16_routed", r4_fp16_w, r4_fp16_mb, 10, routed=True)
    del r4_fp16_w
    _eval_patch_arm("s3k_sqat_plus_r4_s2k_routed", merged_q["s3k_sqat_plus_r4_s2k"], wbytes_q["s3k_sqat_plus_r4_s2k"] / (1024**2), 20, routed=True)
    _eval_patch_arm("s3k_sqat_plus_r4_s2k_unrouted", merged_q["s3k_sqat_plus_r4_s2k"], wbytes_q["s3k_sqat_plus_r4_s2k"] / (1024**2), 30, routed=False)
    del merged_q["s3k_sqat_plus_r4_s2k"]
    _eval_patch_arm("s3k_sqat_plus_r4_rtn2_routed", merged_q["s3k_sqat_plus_r4_rtn2"], wbytes_q["s3k_sqat_plus_r4_rtn2"] / (1024**2), 40, routed=True)
    del merged_q["s3k_sqat_plus_r4_rtn2"]
    _eval_patch_arm("s3k_sqat_plus_r8_s2k_routed", merged_q["s3k_sqat_plus_r8_s2k"], wbytes_q["s3k_sqat_plus_r8_s2k"] / (1024**2), 50, routed=True)
    del merged_q["s3k_sqat_plus_r8_s2k"]
    phase4_s = round(time.perf_counter() - t_p4, 2)

    comp_pairs = [
        ("scale_qat_vs_s3k_raw", "s3k_plus_scale_qat", "s3k_uniform_raw"),
        ("r4_s2k_routed_vs_s3k_raw", "s3k_sqat_plus_r4_s2k_routed", "s3k_uniform_raw"),
        ("r4_s2k_routed_vs_scale_qat", "s3k_sqat_plus_r4_s2k_routed", "s3k_plus_scale_qat"),
        ("r8_s2k_routed_vs_scale_qat", "s3k_sqat_plus_r8_s2k_routed", "s3k_plus_scale_qat"),
        ("r4_s2k_routed_vs_base_bf16", "s3k_sqat_plus_r4_s2k_routed", "base_bf16"),
        ("r8_s2k_routed_vs_base_bf16", "s3k_sqat_plus_r8_s2k_routed", "base_bf16"),
        ("r4_s2k_vs_r4_fp16", "s3k_sqat_plus_r4_s2k_routed", "s3k_sqat_plus_r4_fp16_routed"),
        ("r4_s2k_vs_r4_rtn2", "s3k_sqat_plus_r4_s2k_routed", "s3k_sqat_plus_r4_rtn2_routed"),
    ]
    paired_200 = {lbl: _paired_bootstrap(gsm_rows_by_arm[a], gsm_rows_by_arm[b], seed=int(args.seed) + i) for i, (lbl, a, b) in enumerate(comp_pairs)}
    paired_100 = {lbl: _paired_bootstrap(gsm_rows_by_arm[a][:100], gsm_rows_by_arm[b][:100], seed=int(args.seed) + 100 + i) for i, (lbl, a, b) in enumerate(comp_pairs)}

    g2 = lambda k: float(arms_out[k]["gsm8k_200"]["accuracy"])  # noqa: E731
    return {
        "experiment": "strata_patch_recovery_dense",
        "model": args.model,
        "total_quantized_params": total_params,
        "kept_self_distill_traces": len(self_distill_samples),
        "router_report": router_report,
        "quant_err_stats": {k: round(v, 6) for k, v in errs_q.items()},
        "train_stats": {"r4": res4, "r8": res8},
        "arms": arms_out,
        "paired_gsm8k_200": paired_200,
        "paired_gsm8k_100": paired_100,
        "timing_s": {
            "phase1_bf16_s": phase1_s,
            "phase2_s3k_base_s": phase2_s,
            "phase3_scale_qat_s": phase3_s,
            "phase4_patch_rec_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "criteria": {
            "a_scale_qat_improves_s3k": g2("s3k_plus_scale_qat") > g2("s3k_uniform_raw") or arms_out["s3k_plus_scale_qat"]["wikitext2_ppl"] < arms_out["s3k_uniform_raw"]["wikitext2_ppl"],
            "b_r4_s2k_beats_scale_qat": g2("s3k_sqat_plus_r4_s2k_routed") > g2("s3k_plus_scale_qat"),
            "c_r8_s2k_beats_scale_qat": g2("s3k_sqat_plus_r8_s2k_routed") > g2("s3k_plus_scale_qat"),
            "d_best_patch_within_5pts_of_bf16": max(g2("s3k_sqat_plus_r4_s2k_routed"), g2("s3k_sqat_plus_r8_s2k_routed")) >= g2("base_bf16") - 0.05,
            "e_routed_protects_mmlu_vs_scale_qat": arms_out["s3k_sqat_plus_r4_s2k_routed"]["mmlu_200"]["non_math"]["delta_vs_base"] >= arms_out["s3k_plus_scale_qat"]["mmlu_200"]["non_math"]["delta_vs_base"] - 0.005,
        },
        "per_item_gsm8k_200": {k: [{"strict_correct": r["strict_correct"], "pred": r["pred"], "gold": r["gold"]} for r in v] for k, v in gsm_rows_by_arm.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-gsm8k-train", type=int, default=240)
    ap.add_argument("--n-gsm8k-test", type=int, default=200)
    ap.add_argument("--total-steps", type=int, default=120)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=64)
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
