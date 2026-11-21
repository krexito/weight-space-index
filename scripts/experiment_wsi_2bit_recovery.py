"""WSI 2-Bit Quantization Recovery Pilot: Sparse Fisher Residuals in CPU RAM vs AWQ / Weight-Error / Random.

Quantizes all 2D decoder `nn.Linear` weights to per-group asymmetric 2-bit (`group_size=64`, ~2.5 eff bits/weight)
and stores the sparse residual precision delta `DeltaW = W_orig - W_q2` strictly in CPU RAM (pinned when CUDA is used).
Tests at matched parameter budgets (`2%`, `5%`, `10%`) how much `MMLU (n=400)` accuracy and held-out `NLL/PPL`
each selection policy recovers without increasing resident GPU quantized weights:
  1. ``random``            : uniform random k% columns per layer
  2. ``weight_err``        : top k% columns by ||DeltaW[:, j]||_2^2
  3. ``activation_awq``    : top k% columns by ||DeltaW[:, j]||_2^2 * E[x_j^2]
  4. ``wsi_fisher``        : top k% columns per layer by token-level grad-weighted impact E[(x_{t,j} * <g_t, dW_{:,j}>)^2]
  5. ``wsi_fisher_global`` : cross-layer global allocation under the exact same total parameter budget

CPU smoke test::
    python scripts/experiment_wsi_2bit_recovery.py --smoke

Colab L4 run::
    python scripts/experiment_wsi_2bit_recovery.py --model /content/base --out /content/wsi_2bit_recovery_lfm3_l4.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
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
import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402


def quantize_2bit_group(
    w: torch.Tensor,
    *,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Per-row-group asymmetric uniform 2-bit (`0..3`) quantization returning packed uint8 + fp16 scale/zero."""
    if w.dim() != 2:
        raise ValueError(f"Expected 2D weight matrix, got {tuple(w.shape)}")
    rows, cols = int(w.shape[0]), int(w.shape[1])
    wf = w.detach().float().reshape(-1)
    pad = (-wf.numel()) % group_size
    if pad:
        wf = F.pad(wf, (0, pad))
    grp = wf.reshape(-1, group_size)
    w_min = grp.amin(dim=1, keepdim=True)
    w_max = grp.amax(dim=1, keepdim=True)
    scale = ((w_max - w_min) / 3.0).clamp_min(1e-8)
    q = torch.round((grp - w_min) / scale).clamp(0, 3).to(torch.uint8)

    pad4 = (-q.numel()) % 4
    q_flat = q.reshape(-1)
    if pad4:
        q_flat = F.pad(q_flat, (0, pad4))
    quads = q_flat.reshape(-1, 4)
    packed = (quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)).cpu()
    return packed, scale.squeeze(1).to(torch.float16).cpu(), w_min.squeeze(1).to(torch.float16).cpu(), (rows, cols)


def dequantize_2bit_group(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    orig_shape: tuple[int, int],
    *,
    group_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct `(rows, cols)` tensor from packed 2-bit weights + fp16 per-group scales/zeros."""
    rows, cols = int(orig_shape[0]), int(orig_shape[1])
    n_groups = scale.numel()
    n_elems_pad = n_groups * group_size
    p = packed.to(torch.int32).unsqueeze(1)
    q_all = torch.cat([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], dim=1).reshape(-1)[:n_elems_pad]
    grp = q_all.float().reshape(n_groups, group_size) * scale.float().unsqueeze(1) + zero.float().unsqueeze(1)
    return grp.reshape(-1)[: rows * cols].reshape(rows, cols).to(dtype)


def collect_decoder_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Return deterministic ordered list of `(unique_name, nn.Linear)` inside transformer blocks."""
    out: list[tuple[str, nn.Linear]] = []
    seen_ids: set[int] = set()
    try:
        _, blocks = find_blocks(model)
    except Exception:
        blocks = []

    if blocks:
        for b_idx, blk in enumerate(blocks):
            for sub_name, mod in blk.named_modules():
                if isinstance(mod, nn.Linear) and mod.weight.dim() == 2 and id(mod) not in seen_ids:
                    seen_ids.add(id(mod))
                    nm = f"block.{b_idx}.{sub_name}" if sub_name else f"block.{b_idx}"
                    out.append((nm, mod))
    else:
        for full_name, mod in model.named_modules():
            low = full_name.lower()
            if any(skip in low for skip in ("embed", "lm_head", "mtp", "visual")):
                continue
            if isinstance(mod, nn.Linear) and mod.weight.dim() == 2 and id(mod) not in seen_ids:
                seen_ids.add(id(mod))
                out.append((full_name, mod))
    return out


@torch.no_grad()
def _score_mmlu_suite(
    model: nn.Module,
    tok: Any,
    items: list[dict[str, Any]],
    formatted: list[str],
    dev: torch.device,
    *,
    letter_ids: dict[str, list[int]],
    batch_size: int,
    base_probs: list[torch.Tensor] | None = None,
    base_correct: list[bool] | None = None,
    base_preds: list[int] | None = None,
) -> tuple[dict[str, Any], list[torch.Tensor], list[bool], list[int]]:
    eos_id, pad_id = sr._tok_eos_pad(tok)
    pad_val = pad_id if pad_id is not None else (eos_id if eos_id is not None else 0)
    letters = ("A", "B", "C", "D")

    enc_all = [
        [int(t) for t in (tok.encode(p_str, add_special_tokens=False) or [eos_id or 1])]
        for p_str in formatted
    ]
    probs_list: list[torch.Tensor] = []
    pred_list: list[int] = []
    correct_list: list[bool] = []
    choice_masses: list[float] = []

    was_training = bool(model.training)
    model.eval()
    try:
        eff_bs = max(1, int(batch_size))
        for start in range(0, len(items), eff_bs):
            sub_enc = enc_all[start : start + eff_bs]
            sub_items = items[start : start + eff_bs]
            if len(sub_enc) == 1:
                inp = torch.tensor(sub_enc, dtype=torch.long, device=dev)
                logits = model(input_ids=inp, use_cache=False).logits[:, -1].float()
            else:
                max_len = max(len(x) for x in sub_enc)
                inp = torch.full((len(sub_enc), max_len), pad_val, dtype=torch.long, device=dev)
                attn = torch.zeros((len(sub_enc), max_len), dtype=torch.long, device=dev)
                for row_i, ids in enumerate(sub_enc):
                    off = max_len - len(ids)
                    inp[row_i, off:] = torch.tensor(ids, dtype=torch.long, device=dev)
                    attn[row_i, off:] = 1
                logits = model(input_ids=inp, attention_mask=attn, use_cache=False).logits[:, -1].float()

            lp = F.log_softmax(logits, dim=-1)
            for row_i, item in enumerate(sub_items):
                scores = []
                for ch in letters:
                    tids = letter_ids.get(ch) or []
                    if tids:
                        scores.append(torch.logsumexp(lp[row_i, tids], dim=0))
                    else:
                        scores.append(torch.tensor(-1e9, dtype=torch.float32))
                s_t = torch.stack(scores).cpu()
                choice_masses.append(float(s_t.exp().sum().item()))
                probs_list.append(F.softmax(s_t, dim=0))
                pred = int(s_t.argmax().item())
                pred_list.append(pred)
                correct_list.append(pred == int(item["gold_idx"]))
    finally:
        if was_training:
            model.train()

    n = len(items)
    acc = sum(1 for x in correct_list if x) / max(1, n)
    rep: dict[str, Any] = {
        "n_total": n,
        "n_correct": sum(1 for x in correct_list if x),
        "accuracy": round(acc, 4),
        "mean_abcd_prob_mass": round(sum(choice_masses) / max(1, n), 4),
    }
    if base_correct is not None and base_preds is not None and base_probs is not None:
        ci = ec.paired_bootstrap_ci(correct_list, base_correct)
        kls = [
            max(0.0, float(F.kl_div(pa.clamp_min(1e-12).log(), pb.clamp_min(1e-12), reduction="sum").item()))
            for pa, pb in zip(probs_list, base_probs)
        ]
        exact = sum(1 for a, b in zip(pred_list, base_preds) if a == b) / max(1, n)
        rep.update({
            "delta_vs_base": ci["delta"],
            "ci95_vs_base": ci["ci95"],
            "gained": ci["gained"],
            "lost": ci["lost"],
            "choice_exact_match_vs_base": round(exact, 4),
            "mean_choice_kl_vs_base": round(sum(kls) / max(1, len(kls)), 6),
        })
    return rep, probs_list, correct_list, pred_list


@torch.no_grad()
def _score_nll_suite(
    model: nn.Module,
    eval_ids: list[list[int]],
    dev: torch.device,
    *,
    base_lps: list[torch.Tensor] | None = None,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    was_training = bool(model.training)
    model.eval()
    nll_sum = 0.0
    n_toks = 0
    kl_sum = 0.0
    captured_lps: list[torch.Tensor] = []
    try:
        for i, ids in enumerate(eval_ids):
            if len(ids) < 2:
                continue
            inp = torch.tensor([ids], dtype=torch.long, device=dev)
            logits = model(input_ids=inp, use_cache=False).logits[0, :-1].float()
            lp = F.log_softmax(logits, dim=-1)
            tgt = inp[0, 1:]
            nll_sum += float(F.nll_loss(lp, tgt, reduction="sum").item())
            n_toks += int(tgt.numel())
            if base_lps is None:
                captured_lps.append(lp.to(dtype=torch.bfloat16, device="cpu"))
            else:
                b_lp = base_lps[i].to(device=dev, dtype=torch.float32)
                kl_sum += float((b_lp.exp() * (b_lp - lp)).sum(dim=-1).clamp_min(0.0).sum().item())
    finally:
        if was_training:
            model.train()

    mean_nll = nll_sum / max(1, n_toks)
    rep = {
        "n_seqs": len(eval_ids),
        "n_tokens": int(n_toks),
        "mean_nll": round(mean_nll, 5),
        "ppl": round(math.exp(min(mean_nll, 20.0)), 4),
        "mean_kl_vs_base": round(kl_sum / max(1, n_toks), 6) if base_lps is not None else 0.0,
    }
    return rep, captured_lps


def _collect_calibration_scores(
    model: nn.Module,
    linears: list[tuple[str, nn.Linear]],
    delta_w_cpu: dict[str, torch.Tensor],
    cal_ids: list[list[int]],
    dev: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Measure per-column ``activation_awq`` and gradient-weighted ``wsi_fisher`` sensitivity on bf16 base."""
    act_sq: dict[str, torch.Tensor] = {}
    fisher_sc: dict[str, torch.Tensor] = {}
    dw_cache: dict[str, torch.Tensor] = {}
    use_gpu_cache = dev.type == "cuda"

    for nm, mod in linears:
        in_f = int(mod.in_features)
        act_sq[nm] = torch.zeros(in_f, dtype=torch.float64, device="cpu")
        fisher_sc[nm] = torch.zeros(in_f, dtype=torch.float64, device="cpu")
        if use_gpu_cache:
            try:
                dw_cache[nm] = delta_w_cpu[nm].to(device=dev, dtype=torch.float16)
            except torch.cuda.OutOfMemoryError:
                dw_cache.clear()
                use_gpu_cache = False
                torch.cuda.empty_cache()

    orig_req = [bool(p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    was_training = bool(model.training)
    model.eval()
    try:
        for seq_ids in cal_ids:
            if len(seq_ids) < 2:
                continue
            inp_dev = torch.tensor([seq_ids], dtype=torch.long, device=dev)
            handles: list[Any] = []
            saved_x: dict[str, torch.Tensor] = {}

            emb_mod = (
                model.get_input_embeddings()
                if hasattr(model, "get_input_embeddings")
                else getattr(getattr(model, "model", model), "embed_tokens", None)
            )
            if emb_mod is not None:
                def _emb_hook(_m, _inp, emb_out):
                    t = emb_out[0] if isinstance(emb_out, tuple) else emb_out
                    if isinstance(t, torch.Tensor) and not t.requires_grad:
                        t2 = t.detach().requires_grad_(True)
                        return (t2, *emb_out[1:]) if isinstance(emb_out, tuple) else t2
                    return emb_out

                handles.append(emb_mod.register_forward_hook(_emb_hook))
            else:
                try:
                    _, blks = find_blocks(model)
                except Exception:
                    blks = []
                if blks:
                    def _blk0_pre(_m, inp):
                        x0 = inp[0] if isinstance(inp, tuple) and inp else inp
                        if isinstance(x0, torch.Tensor) and not x0.requires_grad:
                            x0_t = x0.detach().requires_grad_(True)
                            return (x0_t, *inp[1:]) if isinstance(inp, tuple) else x0_t
                        return None

                    handles.append(blks[0].register_forward_pre_hook(_blk0_pre))

            for nm, mod in linears:
                in_f = int(mod.in_features)
                out_f = int(mod.out_features)

                def _lin_hook(_m, inp, out, nm=nm, in_f=in_f, out_f=out_f):
                    x_raw = inp[0] if isinstance(inp, tuple) else inp
                    if isinstance(x_raw, torch.Tensor):
                        x2 = x_raw.detach().reshape(-1, in_f)
                        saved_x[nm] = x2.cpu() if x2.device.type != dev.type else x2
                        act_sq[nm] += x2.float().pow(2).sum(dim=0).double().cpu()

                    t = out[0] if isinstance(out, tuple) else out
                    if isinstance(t, torch.Tensor):
                        need_rep = False
                        if not t.requires_grad:
                            t = t.detach().requires_grad_(True)
                            need_rep = True

                        def _grad_hook(grad: torch.Tensor, nm=nm, out_f=out_f) -> None:
                            g2 = grad.detach().reshape(-1, out_f).float()
                            x_ref = saved_x.pop(nm, None)
                            if x_ref is None:
                                return
                            x_f = x_ref.to(device=g2.device, dtype=torch.float32)
                            dw_mat = dw_cache[nm].float() if nm in dw_cache else delta_w_cpu[nm].to(device=g2.device, dtype=torch.float32)
                            proj = g2 @ dw_mat
                            fisher_sc[nm] += (x_f * proj).pow(2).sum(dim=0).double().cpu()

                        t.register_hook(_grad_hook)
                        if need_rep:
                            return (t, *out[1:]) if isinstance(out, tuple) else t
                    return out

                handles.append(mod.register_forward_hook(_lin_hook))

            try:
                with torch.enable_grad():
                    logits = model(input_ids=inp_dev, use_cache=False).logits[0, :-1].float()
                    loss = F.nll_loss(F.log_softmax(logits, dim=-1), inp_dev[0, 1:], reduction="mean")
                    loss.backward()
            finally:
                for h in handles:
                    h.remove()
                saved_x.clear()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        dw_cache.clear()
        for p, rq in zip(model.parameters(), orig_req):
            p.requires_grad_(rq)
        if was_training:
            model.train()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return act_sq, fisher_sc


def _select_columns_for_budget(
    policy: str,
    frac: float,
    linears: list[tuple[str, nn.Linear]],
    delta_w_cpu: dict[str, torch.Tensor],
    act_sq: dict[str, torch.Tensor],
    fisher_sc: dict[str, torch.Tensor],
    *,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Select input column indices `{mod_name: 1D LongTensor}` for a given policy & budget."""
    rng = random.Random(seed)
    per_layer_k: dict[str, int] = {}
    total_budget_params = 0
    for nm, mod in linears:
        in_f = int(mod.in_features)
        out_f = int(mod.out_features)
        k = max(1, min(in_f, int(round(frac * in_f))))
        per_layer_k[nm] = k
        total_budget_params += out_f * k

    if policy == "wsi_fisher_global":
        # Global ranking by loss-sensitivity benefit per parameter (`score / out_features`)
        candidates: list[tuple[float, str, int, int]] = []
        for nm, mod in linears:
            out_f = int(mod.out_features)
            for j, val in enumerate(fisher_sc[nm].tolist()):
                candidates.append((float(val) / max(1, out_f), nm, j, out_f))
        candidates.sort(key=lambda t: t[0], reverse=True)
        chosen: dict[str, list[int]] = {nm: [] for nm, _ in linears}
        used_params = 0
        for _benefit, nm, j, out_f in candidates:
            if used_params + out_f <= total_budget_params:
                chosen[nm].append(j)
                used_params += out_f
            if used_params >= total_budget_params:
                break
        return {
            nm: (torch.tensor(sorted(cols), dtype=torch.long) if cols else torch.empty(0, dtype=torch.long))
            for nm, cols in chosen.items()
        }

    selected: dict[str, torch.Tensor] = {}
    for nm, mod in linears:
        in_f = int(mod.in_features)
        k = per_layer_k[nm]
        if policy == "random":
            cols = rng.sample(range(in_f), k)
            selected[nm] = torch.tensor(sorted(cols), dtype=torch.long)
            continue

        dw = delta_w_cpu[nm].float()
        if policy == "weight_err":
            scores = dw.pow(2).sum(dim=0)
        elif policy == "activation_awq":
            scores = dw.pow(2).sum(dim=0) * act_sq[nm].float()
        elif policy == "wsi_fisher":
            scores = fisher_sc[nm].float()
        else:
            raise ValueError(f"Unknown column selection policy: {policy!r}")

        top_idx = torch.topk(scores, k=k, largest=True).indices.sort().values.cpu()
        selected[nm] = top_idx
    return selected


def _benchmark_pcie_sparse_transfer(
    pack: list[tuple[str, torch.Tensor, torch.Tensor]],
    dev: torch.device,
    *,
    repeats: int = 5,
) -> dict[str, Any]:
    """Measure async pinned CPU-RAM -> GPU PCIe transfer latency across sparse residual layers."""
    total_bytes = sum(val.numel() * val.element_size() + idx.numel() * idx.element_size() for _, idx, val in pack)
    if dev.type != "cuda" or total_bytes == 0:
        return {"device": str(dev), "payload_mb": round(total_bytes / (1024.0 * 1024.0), 3), "async_ms": None, "gb_per_s": None}

    stream = torch.cuda.Stream(device=dev)
    # Warmup
    with torch.cuda.stream(stream):
        _ = [(idx.to(dev, non_blocking=True), val.to(dev, non_blocking=True)) for _, idx, val in pack]
    torch.cuda.synchronize(dev)

    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            for _, idx, val in pack:
                _ = idx.to(dev, non_blocking=True)
                _ = val.to(dev, non_blocking=True)
        torch.cuda.synchronize(dev)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    med_ms = sorted(times_ms)[len(times_ms) // 2]
    gb_s = (total_bytes / 1e9) / max(med_ms / 1000.0, 1e-9)
    return {
        "device": str(dev),
        "payload_mb": round(total_bytes / (1024.0 * 1024.0), 3),
        "async_ms": round(med_ms, 3),
        "gb_per_s": round(gb_s, 2),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_2bit_recovery_experiment(
    model: nn.Module,
    tok: Any,
    *,
    cal_ids: list[list[int]],
    nll_eval_ids: list[list[int]],
    mmlu_items: list[dict[str, Any]],
    dev: torch.device,
    fracs: tuple[float, ...] = (0.02, 0.05, 0.10),
    policies: tuple[str, ...] = ("random", "weight_err", "activation_awq", "wsi_fisher", "wsi_fisher_global"),
    group_size: int = 64,
    batch_size: int = 16,
    seed: int = 42,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    linears = collect_decoder_linears(model)
    if not linears:
        raise RuntimeError("No 2D decoder nn.Linear modules found to quantize")

    # 1. Baseline bf16 evaluation
    mmlu_fmt = [ec._format_mmlu_prompt(tok, item) for item in mmlu_items]
    letter_ids = ec._resolve_mmlu_letter_ids(tok)
    mmlu_bs = ec._resolve_fast_bs(model, tok, mmlu_fmt[:8], dev, batch_size) if (batch_size > 1 and len(mmlu_fmt) >= 8) else 1

    bf16_mmlu, base_m_probs, base_m_cor, base_m_preds = _score_mmlu_suite(
        model, tok, mmlu_items, mmlu_fmt, dev, letter_ids=letter_ids, batch_size=mmlu_bs
    )
    bf16_nll, base_nll_lps = _score_nll_suite(model, nll_eval_ids, dev)
    print(
        f"[bf16_base] mmlu={bf16_mmlu['accuracy']:.1%} ({bf16_mmlu['n_correct']}/{bf16_mmlu['n_total']}) "
        f"nll={bf16_nll['mean_nll']:.4f} ppl={bf16_nll['ppl']:.2f} (mmlu_bs={mmlu_bs})",
        flush=True,
    )

    # 2. Quantize all 2D decoder Linear weights to 2-bit & store exact residual DeltaW in CPU RAM
    w_q2_cpu: dict[str, torch.Tensor] = {}
    delta_w_cpu: dict[str, torch.Tensor] = {}
    orig_dtypes: dict[str, torch.dtype] = {}
    total_params = 0
    bf16_bytes = 0
    q2_packed_bytes = 0
    for nm, mod in linears:
        w = mod.weight.detach()
        orig_dtypes[nm] = w.dtype
        rows, cols = int(w.shape[0]), int(w.shape[1])
        total_params += rows * cols
        bf16_bytes += rows * cols * 2
        packed, sc, zr, shp = quantize_2bit_group(w, group_size=group_size)
        q2_packed_bytes += packed.numel() + sc.numel() * 2 + zr.numel() * 2
        w_q2 = dequantize_2bit_group(packed, sc, zr, shp, group_size=group_size, dtype=w.dtype)
        w_q2_cpu[nm] = w_q2.cpu()
        delta_w_cpu[nm] = (w.detach().cpu().float() - w_q2.float()).to(torch.float16).contiguous()

    footprint = {
        "n_linear_modules": len(linears),
        "total_linear_params": int(total_params),
        "bf16_linear_mb": round(bf16_bytes / (1024.0 * 1024.0), 2),
        "q2_resident_mb": round(q2_packed_bytes / (1024.0 * 1024.0), 2),
        "effective_bits_per_weight": round((q2_packed_bytes * 8.0) / max(1, total_params), 3),
        "compression_ratio": round(bf16_bytes / max(1, q2_packed_bytes), 2),
    }

    # 3. Collect calibration signals on unmodified bf16 model
    t_cal = time.perf_counter()
    act_sq, fisher_sc = _collect_calibration_scores(model, linears, delta_w_cpu, cal_ids, dev)
    cal_s = round(time.perf_counter() - t_cal, 2)

    # 4. Install pure 2-bit quantized weights into GPU model
    with torch.no_grad():
        for nm, mod in linears:
            mod.weight.copy_(w_q2_cpu[nm].to(device=dev, dtype=orig_dtypes[nm]))

    q2_mmlu, _, _, _ = _score_mmlu_suite(
        model,
        tok,
        mmlu_items,
        mmlu_fmt,
        dev,
        letter_ids=letter_ids,
        batch_size=mmlu_bs,
        base_probs=base_m_probs,
        base_correct=base_m_cor,
        base_preds=base_m_preds,
    )
    q2_nll, _ = _score_nll_suite(model, nll_eval_ids, dev, base_lps=base_nll_lps)
    print(
        f"[q2_pure  ] mmlu={q2_mmlu['accuracy']:.1%} ({q2_mmlu['n_correct']}/{q2_mmlu['n_total']}) "
        f"nll={q2_nll['mean_nll']:.4f} ppl={q2_nll['ppl']:.2f}",
        flush=True,
    )

    # 5. Evaluate sparse CPU-RAM residual recovery policies across budgets
    arms_out: dict[str, Any] = {
        "bf16_base": {"mmlu": bf16_mmlu, "nll": bf16_nll, "restored_param_frac": 1.0},
        "q2_pure": {"mmlu": q2_mmlu, "nll": q2_nll, "restored_param_frac": 0.0},
    }
    pcie_benchmarks: dict[str, Any] = {}
    pin_cpu = dev.type == "cuda"

    for frac in fracs:
        for policy in policies:
            arm_key = f"{policy}_{int(round(frac * 100))}pct"
            col_map = _select_columns_for_budget(policy, frac, linears, delta_w_cpu, act_sq, fisher_sc, seed=seed)

            pack: list[tuple[str, torch.Tensor, torch.Tensor]] = []
            restored_params = 0
            for nm, mod in linears:
                idx = col_map[nm]
                if idx.numel() == 0:
                    continue
                val_slice = delta_w_cpu[nm][:, idx].contiguous()
                if pin_cpu:
                    idx = idx.pin_memory()
                    val_slice = val_slice.pin_memory()
                restored_params += int(mod.out_features) * int(idx.numel())
                pack.append((nm, idx, val_slice))

            if policy == "wsi_fisher":
                pcie_benchmarks[arm_key] = _benchmark_pcie_sparse_transfer(pack, dev)

            mod_by_nm = dict(linears)
            # Apply sparse CPU-RAM residuals onto q2 weights, evaluate, then restore exact q2 slices
            with torch.no_grad():
                for nm, idx, val_slice in pack:
                    w = mod_by_nm[nm].weight
                    idx_dev = idx.to(device=dev)
                    new_cols = (w[:, idx_dev].float() + val_slice.to(device=dev, dtype=torch.float32)).to(orig_dtypes[nm])
                    w[:, idx_dev] = new_cols

            try:
                m_rep, _, _, _ = _score_mmlu_suite(
                    model,
                    tok,
                    mmlu_items,
                    mmlu_fmt,
                    dev,
                    letter_ids=letter_ids,
                    batch_size=mmlu_bs,
                    base_probs=base_m_probs,
                    base_correct=base_m_cor,
                    base_preds=base_m_preds,
                )
                n_rep, _ = _score_nll_suite(model, nll_eval_ids, dev, base_lps=base_nll_lps)
            finally:
                with torch.no_grad():
                    for nm, idx, _ in pack:
                        w = mod_by_nm[nm].weight
                        idx_dev = idx.to(device=dev)
                        w[:, idx_dev] = w_q2_cpu[nm][:, idx].to(device=dev, dtype=orig_dtypes[nm])

            cpu_ram_mb = (restored_params * 2) / (1024.0 * 1024.0)
            q2_acc_drop = max(1e-6, bf16_mmlu["accuracy"] - q2_mmlu["accuracy"])
            mmlu_recov_frac = (
                (m_rep["accuracy"] - q2_mmlu["accuracy"]) / q2_acc_drop
                if bf16_mmlu["accuracy"] > q2_mmlu["accuracy"]
                else 1.0
            )
            q2_nll_drop = max(1e-6, q2_nll["mean_nll"] - bf16_nll["mean_nll"])
            nll_recov_frac = (
                (q2_nll["mean_nll"] - n_rep["mean_nll"]) / q2_nll_drop
                if q2_nll["mean_nll"] > bf16_nll["mean_nll"]
                else 1.0
            )

            arms_out[arm_key] = {
                "policy": policy,
                "budget_frac": float(frac),
                "restored_params": int(restored_params),
                "restored_param_frac": round(restored_params / max(1, total_params), 5),
                "cpu_ram_residual_mb": round(cpu_ram_mb, 2),
                "mmlu": m_rep,
                "nll": n_rep,
                "mmlu_gap_recovered_frac": round(mmlu_recov_frac, 4),
                "nll_gap_recovered_frac": round(nll_recov_frac, 4),
            }
            print(
                f"[{arm_key:23s}] mmlu={m_rep['accuracy']:.1%} (recov={mmlu_recov_frac:.1%}) "
                f"nll={n_rep['mean_nll']:.4f} (recov={nll_recov_frac:.1%}) "
                f"ram={cpu_ram_mb:.1f}MB",
                flush=True,
            )

    summary = {
        k: {
            "mmlu_acc": v["mmlu"]["accuracy"],
            "nll": v["nll"]["mean_nll"],
            "ppl": v["nll"]["ppl"],
            "restored_param_frac": v.get("restored_param_frac"),
            "cpu_ram_residual_mb": v.get("cpu_ram_residual_mb", 0.0),
            "mmlu_gap_recovered_frac": v.get("mmlu_gap_recovered_frac"),
            "nll_gap_recovered_frac": v.get("nll_gap_recovered_frac"),
        }
        for k, v in arms_out.items()
    }

    report = {
        "status": "ok",
        "group_size": int(group_size),
        "footprint": footprint,
        "pcie_async_benchmarks": pcie_benchmarks,
        "calibration_s": cal_s,
        "total_s": round(time.perf_counter() - t0, 2),
        "summary": summary,
        "arms": arms_out,
    }
    if out_path is not None:
        _atomic_write_json(Path(out_path), report)
    return report


def run_smoke(out_dir: str | Path | None = None) -> dict[str, Any]:
    """CPU synthetic validation verifying 2-bit degradation and `wsi_fisher` superior recovery vs `random`."""
    class _QBlock(nn.Module):
        def __init__(self, hidden: int = 32, inter: int = 64) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
            self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.mlp.down_proj(F.relu(self.mlp.up_proj(x)))

    class _QLM(nn.Module):
        def __init__(self, vocab: int = 24, hidden: int = 32, inter: int = 64, n_layers: int = 2) -> None:
            super().__init__()
            self.config = type("Cfg", (), {"model_type": "llama", "hidden_size": hidden, "eos_token_id": 2, "pad_token_id": 0})()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(vocab, hidden)
            self.model.layers = nn.ModuleList([_QBlock(hidden, inter) for _ in range(n_layers)])
            self.model.norm = nn.LayerNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, input_ids: torch.Tensor, use_cache: bool = False, **_kw):
            h = self.model.embed_tokens(input_ids)
            for blk in self.model.layers:
                h = blk(h)
            return type("Out", (), {"logits": self.lm_head(self.model.norm(h))})()

    class _QTok:
        eos_token_id = 2
        pad_token_id = 0
        _MAP = {"A": 4, "B": 5, "C": 6, "D": 7}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            s = text.strip()
            if s in self._MAP:
                return [self._MAP[s]]
            return [8, 9, 10, 4]

    torch.manual_seed(0)
    hidden, inter = 32, 64
    model = _QLM(vocab=24, hidden=hidden, inter=inter, n_layers=2).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.model.norm.weight.fill_(1.0)
        emb = model.model.embed_tokens.weight
        for tid in (4, 8, 9, 10):
            emb[tid, 0] = 2.0  # signal channel carrying next-token identity
            emb[tid, 3] = 2.5  # distractor high-activation channel unused by lm_head

        for blk in model.model.layers:
            # Col 0 drives neuron 0 -> residual dim 0 (predicts targets in lm_head); col 10 acts as group range outlier
            blk.mlp.up_proj.weight[0, 0] = 3.0
            blk.mlp.up_proj.weight[0, 10] = -10.0
            blk.mlp.down_proj.weight[0, 0] = 2.5
            blk.mlp.down_proj.weight[0, 20] = -9.0

            # High activation-AWQ distractor on col 3 -> neuron 4 -> unused dim 12 (lm_head[:, 12] == 0)
            blk.mlp.up_proj.weight[4, 3] = 3.2
            blk.mlp.up_proj.weight[4, 11] = -9.5
            blk.mlp.down_proj.weight[12, 4] = 2.8
            blk.mlp.down_proj.weight[12, 21] = -9.5

            # Huge pure weight-error distractor on inactive col 2 / neuron 5
            blk.mlp.up_proj.weight[5, 2] = 6.0
            blk.mlp.down_proj.weight[13, 5] = 6.0

        lm = model.lm_head.weight
        for tid in (4, 8, 9, 10):
            lm[tid, 0] = 4.0

    tok = _QTok()
    cal_ids = [[8, 9, 10, 4], [9, 10, 4, 8], [10, 8, 9, 4], [8, 4, 9, 10]]
    eval_ids = [[8, 9, 10, 4], [9, 10, 4, 8], [8, 10, 9, 4], [10, 9, 8, 4]]
    mmlu_items = [{"question": f"q {i}", "choices": ["a", "b", "c", "d"], "gold_idx": 0, "subject": "math"} for i in range(4)]

    with tempfile.TemporaryDirectory() as td:
        out_p = (Path(out_dir) if out_dir else Path(td)) / "smoke_2bit.json"
        rep = run_2bit_recovery_experiment(
            model,
            tok,
            cal_ids=cal_ids,
            nll_eval_ids=eval_ids,
            mmlu_items=mmlu_items,
            dev=torch.device("cpu"),
            fracs=(0.03,),
            policies=("random", "weight_err", "activation_awq", "wsi_fisher", "wsi_fisher_global"),
            group_size=16,
            batch_size=1,
            seed=42,
            out_path=out_p,
        )
        arms = rep["arms"]
        nll_q2 = arms["q2_pure"]["nll"]["mean_nll"]
        nll_base = arms["bf16_base"]["nll"]["mean_nll"]
        nll_fisher = arms["wsi_fisher_3pct"]["nll"]["mean_nll"]
        nll_global = arms["wsi_fisher_global_3pct"]["nll"]["mean_nll"]
        rep["smoke_passed"] = (
            nll_q2 > nll_base
            and nll_fisher < nll_q2
            and nll_global < nll_q2
            and out_p.exists()
        )
        return rep


def _build_token_chunks_from_gsm8k(tok: Any, split: str, n_seqs: int, max_len: int = 128) -> list[list[int]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    out: list[list[int]] = []
    for row in ds:
        txt = f"{row['question']}\n{row['answer']}"
        ids = [int(t) for t in tok.encode(txt, add_special_tokens=False)][:max_len]
        if len(ids) >= 16:
            out.append(ids)
        if len(out) >= n_seqs:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="WSI 2-Bit Sparse CPU-RAM Residual Recovery Pilot")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--n-mmlu", type=int, default=400)
    ap.add_argument("--n-cal", type=int, default=16)
    ap.add_argument("--n-nll-eval", type=int, default=32)
    ap.add_argument("--cal-max-len", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="wsi_2bit_recovery_lfm3_l4.json")
    a = ap.parse_args()

    if a.smoke:
        rep = run_smoke()
        print(json.dumps(rep["summary"], indent=2))
        if not rep["smoke_passed"]:
            raise SystemExit(1)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, trust_remote_code=True).to(dev).eval()

    cal_ids = _build_token_chunks_from_gsm8k(tok, "train", a.n_cal, max_len=a.cal_max_len)
    nll_eval_ids = _build_token_chunks_from_gsm8k(tok, "test", a.n_nll_eval, max_len=a.cal_max_len)
    mmlu_items = ec._load_mmlu_stratified(a.n_mmlu, a.seed)

    rep = run_2bit_recovery_experiment(
        model,
        tok,
        cal_ids=cal_ids,
        nll_eval_ids=nll_eval_ids,
        mmlu_items=mmlu_items,
        dev=dev,
        group_size=a.group_size,
        batch_size=a.batch_size,
        seed=a.seed,
        out_path=a.out,
    )
    print(json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()
