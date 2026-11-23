"""Training-Free Sequential Layer-by-Layer Closed-Form 1-Shot Absorption Experiment.

Tests whether the 1-shot conditioning gap (`base_08b` -> `oracle_1shot_08b`) can be absorbed directly into
weight cartridges with ZERO backprop / ZERO epochs via sequential layer-by-layer regularized normal equations
using cross-question 1-shot activation targets + analytical prose anchor + cross-question Ridge CV:

  ΔW_l = (ΔY_skill^T X_skill) (X_skill^T X_skill + γ X_anchor_norm^T X_anchor_norm + λ* I)^{-1}

Paired arms on the exact 5-skill benchmark (`60` blind hard items, `24` guardrails, `200` MMLU, `16` held-out prose):
  1. `base_08b`                          : Unpatched Qwen3.5-0.8B (`enable_thinking=False`)
  2. `teacher_4b`                        : Unpatched Qwen3.5-4B (`enable_thinking=False`)
  3. `oracle_1shot_08b`                  : 0.8B with 1 solved teacher trace in context (target to absorb)
  4. `closedform_mlp_seq24L_routed`      : 0-backprop sequential L0->L23 solve on `mlp.down_proj` only
  5. `closedform_fullblock_seq24L_routed`: 0-backprop sequential L0->L23 solve on `mixer_out` + `mlp.down_proj`
  6. `closedform_svd_r16_routed`         : 0-backprop sequential solve SVD-compressed to `rank=16`
  7. `hybrid_warmstart_8steps_routed`    : Matched `r=16` all-24L modules warm-started from `closedform_svd_r16` + 8 LoRA steps
  8. `coldstart_8steps_routed`           : Exact matched `r=16` all-24L modules trained 8 LoRA steps from cold init

CPU smoke test::
    python scripts/experiment_pocket_moe_closedform.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_moe_closedform.py --teacher /content/teacher_4b --student /content/student_08b --out /content/pocket_moe_closedform_l4.json
"""
from __future__ import annotations

import argparse
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
    _bake_full_checkpoint,
    _block_prefixes,
    _enable_emb_grad_hook,
    _grade_rows,
    _hf_generate_subset,
    _vllm_generate,
    distill_step_loss,
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
from experiment_pocket_moe_regularized import _score_routed_mmlu_sparse  # noqa: E402
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


# ------------------------------------------------------------------ Module Discovery & Alignment
class _StopAtLayer(Exception):
    """Sentinel exception raised in a forward hook to short-circuit execution past the target layer."""


def _aligned_suffix_len(ids_0: list[int], ids_1: list[int]) -> int:
    """Return exact length of the longest common token suffix shared by `ids_0` and `ids_1`."""
    n = min(len(ids_0), len(ids_1))
    k = 0
    while k < n and ids_0[-1 - k] == ids_1[-1 - k]:
        k += 1
    return k


def _find_mixer_out(blk: nn.Module) -> tuple[str, nn.Linear] | None:
    """Return `(relative_attr_path, nn.Linear)` for the block's attention/linear-attention output projection."""
    self_attn = getattr(blk, "self_attn", None)
    if self_attn is not None and isinstance(getattr(self_attn, "o_proj", None), nn.Linear):
        return ("self_attn.o_proj", self_attn.o_proj)
    linear_attn = getattr(blk, "linear_attn", None)
    if linear_attn is not None and isinstance(getattr(linear_attn, "out_proj", None), nn.Linear):
        return ("linear_attn.out_proj", linear_attn.out_proj)
    return None


def _find_mlp_down(blk: nn.Module) -> tuple[str, nn.Linear]:
    """Return `(relative_attr_path, nn.Linear)` for the block's MLP down-projection."""
    mlp = getattr(blk, "mlp", None)
    if mlp is not None and isinstance(getattr(mlp, "down_proj", None), nn.Linear):
        return ("mlp.down_proj", mlp.down_proj)
    ff = getattr(blk, "feed_forward", None)
    if ff is not None and isinstance(getattr(ff, "w2", None), nn.Linear):
        return ("feed_forward.w2", ff.w2)
    raise AttributeError(f"Could not find MLP down projection on block {type(blk).__name__}")


def _mid_residual_hook_target(blk: nn.Module) -> nn.Module:
    """Return the first module consuming the post-mixer residual stream so a pre-hook captures `x_in + mixer_out`."""
    pan = getattr(blk, "post_attention_layernorm", None)
    if pan is not None:
        return pan
    mlp = getattr(blk, "mlp", None)
    if mlp is not None:
        gp = getattr(mlp, "gate_proj", None)
        return gp if gp is not None else mlp
    raise AttributeError(f"Could not locate mid-block residual consumer on {type(blk).__name__}")


# ------------------------------------------------------------------ Closed-Form Ridge Solver with Cross-Question CV
@torch.no_grad()
def _solve_ridge_cv(
    G_tr: torch.Tensor,
    C_tr: torch.Tensor,
    G_val: torch.Tensor,
    C_val: torch.Tensor,
    yy_val: float,
    G_anc: torch.Tensor,
    n_tr_tok: int,
    n_anc_tok: int,
    n_val_tok: int = 0,
    *,
    gamma: float = 0.25,
    c_grid: tuple[float, ...] = (0.05, 0.25, 1.0, 4.0, 16.0, 64.0),
    min_val_r2: float = 0.01,
    rank_cap: int | None = None,
) -> dict[str, Any]:
    """Solve regularized normal equations using 1 eigendecomposition for CV λ selection, refit all data, and optionally SVD-compress."""
    d_out, d_in = int(C_tr.shape[0]), int(C_tr.shape[1])
    dev = G_tr.device

    if n_tr_tok <= 0 or float(torch.trace(G_tr).item()) <= 1e-12:
        return {
            "dW": torch.zeros((d_out, d_in), dtype=torch.float32),
            "factors": None,
            "best_c": None,
            "val_r2": 0.0,
            "skipped": True,
            "rel_norm": 0.0,
        }

    scale_tr = float(n_tr_tok) / float(max(1, n_anc_tok)) if (n_anc_tok > 0 and float(gamma) > 0.0) else 0.0
    G_anc_norm = G_anc * scale_tr if scale_tr > 0.0 else torch.zeros_like(G_tr)

    gram_cv = 0.5 * ((G_tr + float(gamma) * G_anc_norm) + (G_tr + float(gamma) * G_anc_norm).T)
    evals, Q = torch.linalg.eigh(gram_cv)
    evals = evals.clamp_min(0.0)
    mean_eig = max(float(torch.trace(G_tr).item()) / max(1, d_in), 1e-8)
    CQ = C_tr @ Q

    best_c = float(c_grid[0])
    best_val_r2 = -1e9
    has_val = float(yy_val) > 1e-12 and float(torch.trace(G_val).item()) > 1e-12

    for c_val in c_grid:
        lam = float(c_val) * mean_eig
        dW_cand = (CQ / (evals + lam)[None, :]) @ Q.T
        if has_val:
            val_mse = (
                float(yy_val)
                - 2.0 * float((dW_cand * C_val).sum().item())
                + float(((dW_cand @ G_val) * dW_cand).sum().item())
            )
            r2 = 1.0 - val_mse / float(yy_val)
        else:
            r2 = 0.0
        if r2 > best_val_r2:
            best_val_r2 = r2
            best_c = float(c_val)

    if has_val and best_val_r2 < float(min_val_r2):
        return {
            "dW": torch.zeros((d_out, d_in), dtype=torch.float32),
            "factors": None,
            "best_c": round(best_c, 6),
            "val_r2": round(best_val_r2, 6),
            "skipped": True,
            "rel_norm": 0.0,
        }

    # Refit on all 6 questions combined (train + val) with proportional anchor weight
    G_all = G_tr + G_val
    C_all = C_tr + C_val
    n_all_tok = int(n_tr_tok) + max(0, int(n_val_tok))
    scale_all = float(n_all_tok) / float(max(1, n_anc_tok)) if (n_anc_tok > 0 and float(gamma) > 0.0) else 0.0
    mean_eig_all = max(float(torch.trace(G_all).item()) / max(1, d_in), 1e-8)
    lam_all = float(best_c) * mean_eig_all
    eye = torch.eye(d_in, dtype=torch.float64, device=dev)
    lhs = G_all + float(gamma) * (G_anc * scale_all) + lam_all * eye
    dW_full = torch.linalg.solve(lhs, C_all.T).T

    factors: tuple[torch.Tensor, torch.Tensor] | None = None
    if rank_cap is not None and 0 < int(rank_cap) < min(d_out, d_in):
        U, S, Vh = torch.linalg.svd(dW_full, full_matrices=False)
        r = min(int(rank_cap), int(S.numel()))
        s_sqrt = S[:r].clamp_min(0.0).sqrt()
        B = U[:, :r] * s_sqrt[None, :]
        A = s_sqrt[:, None] * Vh[:r, :]
        dW_full = B @ A
        factors = (A.float().cpu(), B.float().cpu())

    dw_norm = float(dW_full.norm().item())
    return {
        "dW": dW_full.float().cpu(),
        "factors": factors,
        "best_c": round(best_c, 6),
        "val_r2": round(best_val_r2, 6),
        "skipped": False,
        "rel_norm": dw_norm,
    }


# ------------------------------------------------------------------ Sequential Layer-by-Layer 1-Shot Absorption
@torch.no_grad()
def _capture_1shot_targets(
    model: nn.Module,
    aligned_items: list[dict[str, Any]],
    *,
    capture_mid: bool,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]] | None]:
    """Run base model ONCE per 1-shot-prefixed item, returning `[layer][item] -> [T_align, H]` CPU bf16 targets."""
    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    store_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    target_outs: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    target_mids: list[list[torch.Tensor]] | None = [[] for _ in range(n_layers)] if capture_mid else None

    for item in aligned_items:
        ids_1 = item["ids_1"]
        a_len = int(item["align_len"])
        cur_outs: dict[int, torch.Tensor] = {}
        cur_mids: dict[int, torch.Tensor] = {}
        handles: list[Any] = []

        for li, blk in enumerate(blocks):
            def _make_out_hook(idx_l: int):
                def _hook(_m, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    cur_outs[idx_l] = t[0, -a_len:, :].detach().to(device="cpu", dtype=store_dtype)
                return _hook

            handles.append(blk.register_forward_hook(_make_out_hook(li)))
            if capture_mid and _find_mixer_out(blk) is not None:
                mid_mod = _mid_residual_hook_target(blk)

                def _make_mid_hook(idx_l: int):
                    def _pre(_m, inp):
                        t = inp[0]
                        cur_mids[idx_l] = t[0, -a_len:, :].detach().to(device="cpu", dtype=store_dtype)
                    return _pre

                handles.append(mid_mod.register_forward_pre_hook(_make_mid_hook(li)))

        try:
            inp_t = torch.tensor(ids_1, dtype=torch.long, device=dev).unsqueeze(0)
            model(input_ids=inp_t, use_cache=False)
            del inp_t
        finally:
            for h in handles:
                h.remove()

        for li in range(n_layers):
            target_outs[li].append(cur_outs[li])
            if target_mids is not None:
                target_mids[li].append(cur_mids[li] if li in cur_mids else cur_outs[li])

    return target_outs, target_mids


@torch.no_grad()
def _solve_sequential_closedform(
    model: nn.Module,
    aligned_items: list[dict[str, Any]],
    target_outs: list[list[torch.Tensor]],
    *,
    target_mids: list[list[torch.Tensor]] | None = None,
    anchor_ids: list[torch.Tensor],
    mode: str = "mlp",
    rank_cap: int | None = None,
    gamma: float = 0.25,
    healthy_ids: list[torch.Tensor] | None = None,
    base_healthy_logp: list[torch.Tensor | None] | None = None,
    base_healthy_nll: float = 0.0,
) -> dict[str, Any]:
    """Sequentially absorb 1-shot residual targets from layer 0 to `n_layers-1` with zero backprop."""
    assert mode in ("mlp", "fullblock"), mode
    if mode == "fullblock":
        assert target_mids is not None

    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
    prefixes = _block_prefixes(model)
    n_layers = len(blocks)

    saved_orig: list[tuple[str, nn.Linear, torch.Tensor]] = []
    for li, blk in enumerate(blocks):
        if mode == "fullblock":
            mx = _find_mixer_out(blk)
            if mx is not None:
                sub_mx, lin_mx = mx
                saved_orig.append((f"{prefixes[li]}.{sub_mx}.weight", lin_mx, lin_mx.weight.detach().cpu().clone()))
        sub_dn, lin_dn = _find_mlp_down(blk)
        saved_orig.append((f"{prefixes[li]}.{sub_dn}.weight", lin_dn, lin_dn.weight.detach().cpu().clone()))

    merged_weights: dict[str, torch.Tensor] = {}
    svd_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    layer_stats: list[dict[str, Any]] = []
    was_training = bool(model.training)
    t_c0 = time.perf_counter()

    try:
        model.eval()
        for li in range(n_layers):
            blk = blocks[li]
            substeps: list[tuple[str, nn.Linear, nn.Module, str, list[torch.Tensor]]] = []
            if mode == "fullblock":
                mx = _find_mixer_out(blk)
                if mx is not None:
                    assert target_mids is not None
                    substeps.append((mx[0], mx[1], _mid_residual_hook_target(blk), "pre", target_mids[li]))
            dn_nm, dn_lin = _find_mlp_down(blk)
            substeps.append((dn_nm, dn_lin, blk, "post", target_outs[li]))

            for sub_nm, target_lin, resid_mod, hook_kind, y_tgts in substeps:
                d_out, d_in = int(target_lin.weight.shape[0]), int(target_lin.weight.shape[1])
                G_tr = torch.zeros((d_in, d_in), dtype=torch.float64, device=dev)
                C_tr = torch.zeros((d_out, d_in), dtype=torch.float64, device=dev)
                G_val = torch.zeros((d_in, d_in), dtype=torch.float64, device=dev)
                C_val = torch.zeros((d_out, d_in), dtype=torch.float64, device=dev)
                G_anc = torch.zeros((d_in, d_in), dtype=torch.float64, device=dev)
                yy_val = 0.0
                n_tr_tok, n_val_tok, n_anc_tok = 0, 0, 0

                for item_i, item in enumerate(aligned_items):
                    a_len = int(item["align_len"])
                    cap_x: list[torch.Tensor] = []
                    cap_y: list[torch.Tensor] = []

                    def _pre_x(_m, inp):
                        cap_x.append(inp[0][0, -a_len:, :].detach())

                    def _pre_y(_m, inp):
                        cap_y.append(inp[0][0, -a_len:, :].detach())
                        raise _StopAtLayer()

                    def _post_y(_m, _inp, out):
                        t = out[0] if isinstance(out, tuple) else out
                        cap_y.append(t[0, -a_len:, :].detach())
                        raise _StopAtLayer()

                    hx = target_lin.register_forward_pre_hook(_pre_x)
                    hy = resid_mod.register_forward_pre_hook(_pre_y) if hook_kind == "pre" else resid_mod.register_forward_hook(_post_y)
                    try:
                        inp_t = torch.tensor(item["ids_0"], dtype=torch.long, device=dev).unsqueeze(0)
                        model(input_ids=inp_t, use_cache=False)
                        del inp_t
                    except _StopAtLayer:
                        pass
                    finally:
                        hx.remove()
                        hy.remove()

                    X = cap_x[0].double()
                    Y_cur = cap_y[0].double()
                    Y_tgt = y_tgts[item_i].to(device=dev, dtype=torch.float64)
                    dY = Y_tgt - Y_cur
                    n_tok = int(X.shape[0])

                    if item.get("is_val", False):
                        G_val += X.T @ X
                        C_val += dY.T @ X
                        yy_val += float((dY * dY).sum().item())
                        n_val_tok += n_tok
                    else:
                        G_tr += X.T @ X
                        C_tr += dY.T @ X
                        n_tr_tok += n_tok
                    del X, Y_cur, Y_tgt, dY

                for anc_t in anchor_ids:
                    cap_ax: list[torch.Tensor] = []

                    def _pre_ax(_m, inp):
                        cap_ax.append(inp[0].reshape(-1, d_in).detach())
                        raise _StopAtLayer()

                    hax = target_lin.register_forward_pre_hook(_pre_ax)
                    try:
                        model(input_ids=anc_t.to(dev), use_cache=False)
                    except _StopAtLayer:
                        pass
                    finally:
                        hax.remove()

                    X_a = cap_ax[0].double()
                    G_anc += X_a.T @ X_a
                    n_anc_tok += int(X_a.shape[0])
                    del X_a

                res_sol = _solve_ridge_cv(
                    G_tr,
                    C_tr,
                    G_val,
                    C_val,
                    yy_val,
                    G_anc,
                    n_tr_tok,
                    n_anc_tok,
                    n_val_tok=n_val_tok,
                    gamma=float(gamma),
                    rank_cap=rank_cap,
                )
                del G_tr, C_tr, G_val, C_val, G_anc

                full_nm = f"{prefixes[li]}.{sub_nm}.weight"
                if not res_sol["skipped"]:
                    dW_dev = res_sol["dW"].to(device=target_lin.weight.device, dtype=target_lin.weight.dtype)
                    target_lin.weight.data.add_(dW_dev)
                    del dW_dev
                    if res_sol["factors"] is not None:
                        svd_factors[full_nm] = res_sol["factors"]

                merged_weights[full_nm] = target_lin.weight.detach().cpu().clone()
                w_norm = max(float(target_lin.weight.detach().float().norm().item()), 1e-8)
                layer_stats.append({
                    "layer": li,
                    "module": sub_nm,
                    "val_r2": res_sol["val_r2"],
                    "best_c": res_sol["best_c"],
                    "skipped": res_sol["skipped"],
                    "rel_norm": round(res_sol["rel_norm"] / w_norm, 5),
                })

        compile_time_s = round(time.perf_counter() - t_c0, 2)
        drift = (
            eval_healthy_drift(model, healthy_ids, base_healthy_logp, base_healthy_nll)
            if (healthy_ids is not None and base_healthy_logp is not None)
            else {"healthy_kl": 0.0, "delta_healthy_nll": 0.0}
        )
    finally:
        for _fnm, lin_m, orig_w in saved_orig:
            lin_m.weight.data.copy_(orig_w.to(device=lin_m.weight.device, dtype=lin_m.weight.dtype))
        if was_training:
            model.train()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if rank_cap is not None:
        factor_bytes = sum((a.numel() + b.numel()) * 2 for a, b in svd_factors.values())
        adapter_mb = round(factor_bytes / (1024**2), 4)
    else:
        adapter_mb = round(sum(t.numel() * 2 for t in merged_weights.values()) / (1024**2), 4)

    r2_vals = [st["val_r2"] for st in layer_stats]
    n_edited = sum(1 for st in layer_stats if not st["skipped"])
    return {
        "weights": merged_weights,
        "svd_factors": svd_factors,
        "adapter_mb": adapter_mb,
        "merged_shard_mb": round(sum(t.numel() * 2 for t in merged_weights.values()) / (1024**2), 3),
        "n_edited_modules": n_edited,
        "n_skipped_modules": len(layer_stats) - n_edited,
        "mean_val_r2": round(sum(r2_vals) / max(1, len(r2_vals)), 4),
        "compile_time_s": compile_time_s,
        "layer_stats": layer_stats,
        **drift,
    }


# ------------------------------------------------------------------ Matched 8-Step Warm-Start vs Cold-Start LoRA
class _WarmStartLoRALinear(nn.Module):
    """Rank-r LoRA adapter initialized either from SVD closed-form factors (scale=1.0) or cold Kaiming/0."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha_scale: float,
        dropout: float = 0.05,
        init_A: torch.Tensor | None = None,
        init_B: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.lora_drop: nn.Module = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        dev = base.weight.device

        if init_A is not None and init_B is not None:
            r_use = int(init_A.shape[0])
            self.rank = r_use
            self.scale = 1.0
            self.lora_A = nn.Parameter(init_A.detach().to(device=dev, dtype=torch.float32).clone())
            self.lora_B = nn.Parameter(init_B.detach().to(device=dev, dtype=torch.float32).clone())
        else:
            self.scale = float(alpha_scale) / max(1, self.rank)
            self.lora_A = nn.Parameter(torch.empty((self.rank, base.in_features), device=dev, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.rank), device=dev, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.lora_drop(x.float())
        delta = F.linear(F.linear(h, self.lora_A), self.lora_B) * self.scale
        return out + delta.to(out.dtype)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @torch.no_grad()
    def merged_weight_cpu(self) -> torch.Tensor:
        w = self.base.weight.detach().float().cpu()
        a = self.lora_A.detach().float().cpu()
        b = self.lora_B.detach().float().cpu()
        return (w + self.scale * (b @ a)).to(self.base.weight.dtype)

    def adapter_bytes(self) -> int:
        return int((self.lora_A.numel() + self.lora_B.numel()) * 2)


def _target_module_specs(model: nn.Module) -> list[tuple[int, str, nn.Linear, str]]:
    """Return all `(layer_idx, sub_nm, nn.Linear, full_weight_name)` for full-block mixer_out + mlp_down."""
    _, blocks = find_blocks(model)
    prefixes = _block_prefixes(model)
    specs: list[tuple[int, str, nn.Linear, str]] = []
    for li, blk in enumerate(blocks):
        mx = _find_mixer_out(blk)
        if mx is not None:
            specs.append((li, mx[0], mx[1], f"{prefixes[li]}.{mx[0]}.weight"))
        dn_nm, dn_lin = _find_mlp_down(blk)
        specs.append((li, dn_nm, dn_lin, f"{prefixes[li]}.{dn_nm}.weight"))
    return specs


def _train_matched_8step_lora(
    model: nn.Module,
    target_specs: list[tuple[int, str, nn.Linear, str]],
    train_samples: list[dict[str, Any]],
    *,
    rank: int = 16,
    alpha_scale: float = 32.0,
    total_steps: int = 8,
    grad_accum: int = 3,
    lr_val: float = 2.0e-4,
    weight_decay: float = 0.05,
    lora_dropout: float = 0.05,
    warmup_frac: float = 0.25,
    distill_T: float = 2.0,
    kl_weight: float = 0.85,
    init_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    anchor_ids: list[torch.Tensor] | None = None,
    base_anchor_logp: list[torch.Tensor | None] | None = None,
    anchor_kl_weight: float = 0.05,
    healthy_ids: list[torch.Tensor] | None = None,
    base_healthy_logp: list[torch.Tensor | None] | None = None,
    base_healthy_nll: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train exact matched `target_specs` modules for `total_steps` steps from warm SVD factors or cold init."""
    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
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

    wrapped: list[tuple[nn.Module, str, nn.Linear, str, _WarmStartLoRALinear]] = []
    n_warm = 0
    for li, sub_nm, lin, full_nm in target_specs:
        blk = blocks[li]
        parts = sub_nm.split(".")
        parent = blk
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
        pair = init_factors.get(full_nm) if init_factors is not None else None
        if pair is not None:
            n_warm += 1
            wrapper = _WarmStartLoRALinear(lin, rank=rank, alpha_scale=alpha_scale, dropout=lora_dropout, init_A=pair[0], init_B=pair[1])
        else:
            wrapper = _WarmStartLoRALinear(lin, rank=rank, alpha_scale=alpha_scale, dropout=lora_dropout)
        setattr(parent, attr, wrapper)
        wrapped.append((parent, attr, lin, full_nm, wrapper))

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
            s_loss = 0.0
            for _acc in range(int(grad_accum)):
                if not idx_queue:
                    idx_queue = torch.randperm(len(train_samples), generator=rng).tolist()
                sp = train_samples[idx_queue.pop()]
                ids_dev = torch.tensor(sp["ids"], dtype=torch.long, device=dev).unsqueeze(0)
                lo, hi = int(sp["lo"]), int(sp["hi"])
                with torch.enable_grad():
                    s_logits = _tail_logits(model, ids_dev, lo)[: hi - lo]
                    t_ids = ids_dev[0, lo + 1 : hi + 1]
                    tk_ids = sp["topk_ids"].to(dev) if sp.get("topk_ids") is not None else None
                    tk_log = sp["topk_logits"].to(dev) if sp.get("topk_logits") is not None else None
                    loss = distill_step_loss(s_logits, t_ids, tk_ids, tk_log, T=float(distill_T), kl_weight=float(kl_weight))
                    (loss / float(grad_accum)).backward()
                    s_loss += float(loss.detach().item()) / float(grad_accum)
                del ids_dev, s_logits, t_ids, loss

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
            step_losses.append(round(s_loss, 5))

        for *_rest, w in wrapped:
            w.eval()

        compile_time_s = round(time.perf_counter() - t_c0, 2)
        drift = (
            eval_healthy_drift(model, healthy_ids, base_healthy_logp, base_healthy_nll)
            if (healthy_ids is not None and base_healthy_logp is not None)
            else {"healthy_kl": 0.0, "delta_healthy_nll": 0.0}
        )
        merged_weights = {full_nm: w.merged_weight_cpu() for _, _, _, full_nm, w in wrapped}
        merged_mb = round(sum(t.numel() * 2 for t in merged_weights.values()) / (1024**2), 3)
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

    return {
        "weights": merged_weights,
        "adapter_mb": adapter_mb,
        "merged_shard_mb": merged_mb,
        "n_warm_modules": n_warm,
        "n_total_modules": len(wrapped),
        "total_steps": int(total_steps),
        "step_losses": step_losses,
        "compile_time_s": compile_time_s,
        **drift,
    }


# ------------------------------------------------------------------ Compact Stdout Summary
def _print_closedform_summary(payload: dict[str, Any]) -> None:
    """Print concise human-readable summary + single-line compact JSON omitting `per_item_generation`."""
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return

    print("\n=== 1. COMPILACION CLOSED-FORM (0-BACKPROP) + WARM-START ===", flush=True)
    print("Compile time (s):", payload.get("compile_time_s"), flush=True)
    print("Teacher kept frac:", payload.get("teacher_4b_train_kept_frac"), flush=True)
    print("Multitrace counts:", payload.get("multitrace_counts"), flush=True)
    print("Router report:", payload.get("router_report"), flush=True)

    prose_kl = payload.get("prose_kl", {})
    print("\n=== 2. RESULTADOS BRAZOS PAREADOS (HARD_60 + GUARDRAIL_24) ===", flush=True)
    for arm_nm, st in payload.get("generation_arms", {}).items():
        h, g = st["hard_60"], st["guardrail_24"]
        gm = g.get("guardrail_exact_match_vs_base")
        gm_s = f"{gm*100:5.1f}%" if gm is not None else "   n/a"
        pkl = prose_kl.get(arm_nm, 0.0)
        print(
            f"  {arm_nm:35s} | hard60={h['strict_accuracy']*100:5.1f}% ({h['n_correct']:2d}/60) "
            f"[ren={h['renumbered_30']*100:5.1f}%, rep={h['rephrased_30']*100:5.1f}%] | "
            f"guard={g['strict_accuracy']*100:5.1f}% (ex={gm_s}) | pkl={pkl:.5f}",
            flush=True,
        )
        print(f"     per_skill={h['per_skill']}", flush=True)

    print("\n=== 3. RETENCION MMLU ===", flush=True)
    for arm_nm, vm in payload.get("mmlu_arms", {}).items():
        a_m, nm = vm["all_400"], vm["non_math"]
        d_nm = f"{nm['delta_vs_base']*100:+5.1f} pts" if "delta_vs_base" in nm else "     base"
        print(
            f"  {arm_nm:35s} | MMLU={a_m['accuracy']*100:5.1f}% | non_math={nm['accuracy']*100:5.1f}% ({d_nm}) | "
            f"ABCD={a_m['mean_abcd_prob_mass']*100:5.1f}% | fire={vm['router_fire_rate']*100:5.1f}%",
            flush=True,
        )

    print("\n=== 4. COMPARACIONES PAREADAS (IC 95% BOOTSTRAP, hard_60) ===", flush=True)
    for grp_lbl, grp_key in [("  [hard_60]", "paired_hard_60"), ("  [old_skills_1_to_4 (48)]", "paired_old_skills_48")]:
        print(grp_lbl, flush=True)
        for k, v in payload.get(grp_key, {}).items():
            print(
                f"    {k:40s}: diff={v['diff']*100:+5.1f} pts "
                f"CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})",
                flush=True,
            )

    print("\nBest 0-backprop arm:", payload.get("best_0backprop_arm_name"), flush=True)
    print("Criteria:", payload.get("criteria"), flush=True)
    print("Timing (s):", payload.get("timing_s"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_generation"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


# ------------------------------------------------------------------ CPU Smoke Test
def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of suffix alignment, CV Ridge + SVD, sequential 0-backprop solve, and Step-0 warm-start LoRA."""
    # 1. _aligned_suffix_len
    assert _aligned_suffix_len([9, 8, 1, 2, 4, 10, 20], [1, 2, 4, 10, 20]) == 5
    assert _aligned_suffix_len([1, 2, 3], [1, 2, 4]) == 0

    # 2. _solve_ridge_cv recovers planted linear transform, skips pure noise, and enforces B @ A == dW
    rng = torch.Generator().manual_seed(11)
    d_in, d_out = 12, 8
    X_tr = torch.randn((80, d_in), generator=rng, dtype=torch.float64)
    X_val = torch.randn((40, d_in), generator=rng, dtype=torch.float64)
    X_anc = torch.randn((30, d_in), generator=rng, dtype=torch.float64)
    U0 = torch.randn((d_out, 3), generator=rng, dtype=torch.float64)
    V0 = torch.randn((3, d_in), generator=rng, dtype=torch.float64)
    dW_true = 0.5 * (U0 @ V0)
    Y_tr = X_tr @ dW_true.T
    Y_val = X_val @ dW_true.T

    sol_exact = _solve_ridge_cv(
        X_tr.T @ X_tr, Y_tr.T @ X_tr, X_val.T @ X_val, Y_val.T @ X_val, float((Y_val * Y_val).sum().item()),
        X_anc.T @ X_anc, 80, 30, n_val_tok=40, gamma=0.0, min_val_r2=0.01, rank_cap=4,
    )
    assert not sol_exact["skipped"] and sol_exact["val_r2"] > 0.99, sol_exact
    err_rec = float((sol_exact["dW"].double() - dW_true).norm().item() / dW_true.norm().item())
    assert err_rec < 0.15, err_rec
    A_f, B_f = sol_exact["factors"]
    assert torch.allclose(B_f @ A_f, sol_exact["dW"], atol=1e-4), "SVD factors should multiply back to dW"

    # Unrelated noise targets must be cleanly skipped
    Y_noise_tr = torch.randn((80, d_out), generator=rng, dtype=torch.float64)
    Y_noise_val = torch.randn((40, d_out), generator=rng, dtype=torch.float64)
    sol_noise = _solve_ridge_cv(
        X_tr.T @ X_tr, Y_noise_tr.T @ X_tr, X_val.T @ X_val, Y_noise_val.T @ X_val, float((Y_noise_val * Y_noise_val).sum().item()),
        X_anc.T @ X_anc, 80, 30, n_val_tok=40, gamma=0.1, min_val_r2=0.01,
    )
    assert sol_noise["skipped"] is True, sol_noise

    # 3. Sequential closed-form capture & solve on _PlantedReasoningLM (mlp AND fullblock)
    torch.manual_seed(19)
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    aligned_items = [
        {"ids_0": [1, 2, 4, 10, 20, 20], "ids_1": [5, 1, 2, 4, 10, 20, 20], "align_len": 6, "is_val": False},
        {"ids_0": [2, 1, 4, 11, 20, 20], "ids_1": [4, 2, 1, 4, 11, 20, 20], "align_len": 6, "is_val": False},
        {"ids_0": [1, 2, 4, 12, 20, 20], "ids_1": [5, 1, 2, 4, 12, 20, 20], "align_len": 6, "is_val": True},
        {"ids_0": [2, 1, 4, 13, 20, 20], "ids_1": [4, 2, 1, 4, 13, 20, 20], "align_len": 6, "is_val": True},
    ]
    h_ids = [torch.tensor([1, 2, 3, 1], dtype=torch.long).unsqueeze(0)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    t_outs, t_mids = _capture_1shot_targets(model, aligned_items, capture_mid=True)
    assert len(t_outs) == 4 and t_mids is not None and len(t_mids) == 4

    with torch.no_grad():
        base_final_err = sum(
            float((model(torch.tensor(it["ids_0"]).unsqueeze(0), output_hidden_states=True).hidden_states[-1][0, -6:, :].float() - t_outs[-1][i].float()).pow(2).mean().item())
            for i, it in enumerate(aligned_items)
        ) / len(aligned_items)

    cf_mlp = _solve_sequential_closedform(
        model, aligned_items, t_outs, anchor_ids=h_ids, mode="mlp", rank_cap=None, gamma=0.1,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll,
    )
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    cf_fb = _solve_sequential_closedform(
        model, aligned_items, t_outs, target_mids=t_mids, anchor_ids=h_ids, mode="fullblock", rank_cap=4, gamma=0.1,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll,
    )
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    with _temp_apply_weights(model, cf_fb["weights"]), torch.no_grad():
        fb_final_err = sum(
            float((model(torch.tensor(it["ids_0"]).unsqueeze(0), output_hidden_states=True).hidden_states[-1][0, -6:, :].float() - t_outs[-1][i].float()).pow(2).mean().item())
            for i, it in enumerate(aligned_items)
        ) / len(aligned_items)
    assert fb_final_err <= base_final_err, (fb_final_err, base_final_err)

    # 4. Step-0 exact identity of _WarmStartLoRALinear & 4-step refinement
    specs = _target_module_specs(model)
    tr_samples = []
    for it in aligned_items:
        ids_t = torch.tensor(it["ids_0"], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            tl = model(ids_t).logits[0, 2:5].float()
        vals, idx = tl.topk(8, dim=-1)
        tr_samples.append({"ids": it["ids_0"], "lo": 2, "hi": 5, "topk_ids": idx.to(torch.int32), "topk_logits": vals.to(torch.float16)})

    # Verify Step-0 warm-start module forward matches cf_fb["weights"] bit-for-bit
    _, blocks = find_blocks(model)
    for _li, _sub, lin_m, full_nm in specs:
        if full_nm in cf_fb["svd_factors"]:
            a_i, b_i = cf_fb["svd_factors"][full_nm]
            wm = _WarmStartLoRALinear(lin_m, rank=4, alpha_scale=8.0, dropout=0.0, init_A=a_i, init_B=b_i)
            assert torch.allclose(wm.merged_weight_cpu().float(), cf_fb["weights"][full_nm].float(), atol=1e-4)
            break

    ws_res = _train_matched_8step_lora(
        model, specs, tr_samples, rank=4, alpha_scale=8.0, total_steps=4, grad_accum=2, lr_val=2e-2,
        init_factors=cf_fb["svd_factors"], anchor_ids=h_ids, base_anchor_logp=base_lp, anchor_kl_weight=0.05,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll, seed=42,
    )
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())
    assert ws_res["step_losses"][-1] < ws_res["step_losses"][0], ws_res["step_losses"]

    return {
        "mode": "smoke",
        "passed": True,
        "ridge_recovery_err": round(err_rec, 5),
        "noise_skipped": sol_noise["skipped"],
        "base_final_mse": round(base_final_err, 6),
        "closedform_fb_mse": round(fb_final_err, 6),
        "mlp_edited": cf_mlp["n_edited_modules"],
        "fb_edited": cf_fb["n_edited_modules"],
        "warmstart_losses": ws_res["step_losses"],
    }


# ------------------------------------------------------------------ Full GPU Experiment
def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    backend = _select_backend(args.backend)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if dev.type == "cuda" else torch.float32)
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    dtype_vllm_str = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype]

    ds = build_fewshot_dataset(k_shots=args.k_shots)
    hard_train = ds["hard_train"]
    hard_test = ds["hard_test"]
    guardrail_test = ds["guardrail_test"]
    all_test = hard_test + guardrail_test
    n_hard = len(hard_test)

    mmlu_test, mmlu_router_train = load_mmlu_split(n_test=args.n_mmlu_test, n_router_train=args.n_mmlu_router_train, seed=args.seed)

    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    assert len(tok_t) == len(tok_s), f"Tokenizer vocab mismatch: teacher={len(tok_t)} vs student={len(tok_s)}"
    eos_set = set(lr.eos_ids(tok_s))
    think_end_id = tok_s.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_id, int) or think_end_id == tok_s.unk_token_id:
        think_end_id = None

    # ---- Phase 1: Single 4B Batched Gen (84 Test + 30x4 Multi-Trace Train) + Deduplicated Top-32 Logits ----
    print(f"[cf] Phase 1: Teacher ({args.teacher}) 84 test + 30x4 multitrace ({backend})...", flush=True)
    t_p1 = time.perf_counter()
    test_pids = [_format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False) for it in all_test]
    train_pids = [_format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False) for it in hard_train]
    test_seeds = [int(args.seed) + i for i in range(len(all_test))]

    m_attempts = 4
    train_multi_pids: list[list[int]] = []
    train_multi_seeds: list[int] = []
    for i, p_ids in enumerate(train_pids):
        for m in range(m_attempts):
            train_multi_pids.append(p_ids)
            train_multi_seeds.append(int(args.seed) + 1000 + m * 100 + i)

    teacher_hf = None
    if backend == "vllm":
        outs_1 = _vllm_generate(args.teacher, tok_t, test_pids + train_multi_pids, test_seeds + train_multi_seeds, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
    else:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
        outs_1 = _hf_generate_subset(teacher_hf, tok_t, test_pids + train_multi_pids, test_seeds + train_multi_seeds, max_new_tokens=args.max_new_tokens)

    t_test_raw = outs_1[: len(all_test)]
    t_train_multi_raw = outs_1[len(all_test) :]

    teacher_test_rows = _grade_rows(all_test, t_test_raw, tok_s, eos_set, think_end_id)
    teacher_summary = _summarize_fewshot_arm(teacher_test_rows)
    print(f"[cf] teacher_4b: hard={teacher_summary['hard_60']['strict_accuracy']*100:.1f}% | guard={teacher_summary['guardrail_24']['strict_accuracy']*100:.1f}%", flush=True)

    kept_4b = {sk: 0 for sk in SKILLS_5}
    primary_by_skill_items: dict[str, list[tuple[str, str]]] = {sk: [] for sk in SKILLS_5}
    multitrace_items: dict[str, list[dict[str, Any]]] = {sk: [] for sk in SKILLS_5}
    primary_resp_by_item: list[str] = []

    q_counter = {sk: 0 for sk in SKILLS_5}
    for i, it in enumerate(hard_train):
        sk = it["family"]
        q_idx = q_counter[sk]
        q_counter[sk] += 1
        q_txt = it["question"]
        valid_txts: list[str] = []
        seen_txts: set[str] = set()
        primary_txt: str | None = None

        for m in range(m_attempts):
            raw_m = t_train_multi_raw[i * m_attempts + m]
            body = lr.strip_eos(raw_m["token_ids"], eos_set)
            txt = tok_s.decode(body, skip_special_tokens=True).strip()
            g = grade_response(raw_m["token_ids"], txt, int(it["answer"]), thinking_enabled=False, finish_reason=raw_m["finish_reason"], eos_set=eos_set, tok=tok_s)
            is_ok = bool(g["strict_correct"] and g["has_boxed"] and g["total_tokens"] <= int(args.max_teacher_train_tokens))
            if m == 0 and is_ok:
                primary_txt = txt
                kept_4b[sk] += 1
            if is_ok and txt not in seen_txts and len(valid_txts) < 3:
                seen_txts.add(txt)
                valid_txts.append(txt)

        if primary_txt is None:
            primary_txt = it["fallback_response"]
        if not valid_txts:
            valid_txts = [primary_txt]

        primary_resp_by_item.append(primary_txt)
        primary_by_skill_items[sk].append((q_txt, primary_txt))
        for v_txt in valid_txts:
            multitrace_items[sk].append({"q_idx": q_idx, "question": q_txt, "response": v_txt})

    teacher_kept_frac = {sk: round(kept_4b[sk] / max(1, args.k_shots), 4) for sk in SKILLS_5}
    multitrace_counts = {sk: len(multitrace_items[sk]) for sk in SKILLS_5}
    print(f"[cf] teacher_4b kept={teacher_kept_frac} | multitrace_counts={multitrace_counts}", flush=True)

    # Extract deduplicated Top-32 logits for the 8-step warm/cold LoRA comparison
    if teacher_hf is None:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    unique_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for sk in SKILLS_5:
        for ent in multitrace_items[sk]:
            unique_pairs.setdefault((ent["question"], ent["response"]), {})

    with torch.no_grad():
        for (q_txt, resp_txt), entry in unique_pairs.items():
            p_ids = _format_messages(tok_s, [{"role": "user", "content": format_question(q_txt)}], add_generation_prompt=True, enable_thinking=False)
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
            entry.update({
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

    multitrace_samples = {
        sk: [dict(unique_pairs[(e["question"], e["response"])], family=sk) for e in multitrace_items[sk]]
        for sk in SKILLS_5
    }
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student 0.8B, Balanced K=6 Layer-12 Router, Base MMLU & Prose/Anchor Logprobs ----
    print(f"[cf] Phase 2: Loading Student ({args.student}), K=6 router, base MMLU & anchor...", flush=True)
    t_p2 = time.perf_counter()
    try:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in student_m.parameters():
        p.requires_grad_(False)

    _, blocks = find_blocks(student_m)
    router_layer = len(blocks) // 2
    guard_r_pids = [_format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False) for it in ds["guardrail_router_train"]]
    mmlu_r_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_router_train]
    mmlu_test_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_test]
    prose_r_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in ds["prose_router_train"]]
    prose_held_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in ds["prose_heldout"]]

    r_train_pids = train_pids + guard_r_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = (
        [it["family"] for it in hard_train]
        + ["pass_math"] * len(guard_r_pids)
        + ["pass_mmlu"] * len(mmlu_r_pids)
        + ["pass_prose"] * len(prose_r_pids)
    )
    W_r, b_r = fit_ridge_router(_extract_prompt_hidden_vecs(student_m, r_train_pids, router_layer), r_train_labels, ROUTER_FIT_CLASSES, lam=1.0)
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
    print(f"[cf] Router report: {router_report}", flush=True)

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
    target_specs = _target_module_specs(student_m)
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Sequential Closed-Form 0-Backprop Absorption + Matched 8-Step Warm vs Cold LoRA ----
    print("[cf] Phase 3: Compiling 3 closed-form 0-backprop tiers + 2 matched 8-step LoRA controls...", flush=True)
    t_p3 = time.perf_counter()

    routed_tier_names = (
        "closedform_mlp_seq24L",
        "closedform_fullblock_seq24L",
        "closedform_svd_r16",
        "hybrid_warmstart_8steps",
        "coldstart_8steps",
    )

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        expert_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {nm: {} for nm in routed_tier_names}
        domain_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {nm: {} for nm in routed_tier_names}

        for s_idx, sk in enumerate(SKILLS_5):
            # Build cross-paired 1-shot aligned sequences: CV train = q_idx 0..3, CV val = q_idx 4..5
            prim_sk = primary_by_skill_items[sk]
            aligned_sk: list[dict[str, Any]] = []
            for mt_meta in multitrace_items[sk]:
                q_i = int(mt_meta["q_idx"])
                is_val = q_i >= 4
                ex_pool = [j for j in range(min(4, len(prim_sk))) if j != q_i] or [0]
                ex_idx = ex_pool[len(aligned_sk) % len(ex_pool)]
                q_ex, r_ex = prim_sk[ex_idx]

                ids_0 = _format_messages(
                    tok_s,
                    [{"role": "user", "content": format_question(mt_meta["question"])}, {"role": "assistant", "content": mt_meta["response"]}],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                ids_1 = _format_messages(
                    tok_s,
                    [
                        {"role": "user", "content": format_question(q_ex)},
                        {"role": "assistant", "content": r_ex},
                        {"role": "user", "content": format_question(mt_meta["question"])},
                        {"role": "assistant", "content": mt_meta["response"]},
                    ],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                a_len = _aligned_suffix_len(ids_0, ids_1)
                assert a_len >= 8, (sk, q_i, a_len)
                aligned_sk.append({"ids_0": ids_0, "ids_1": ids_1, "align_len": a_len, "is_val": is_val})

            target_outs, target_mids = _capture_1shot_targets(student_m, aligned_sk, capture_mid=True)

            # 1) Closed-form MLP-only (0 backprop)
            res_mlp = _solve_sequential_closedform(
                student_m,
                aligned_sk,
                target_outs,
                anchor_ids=anchor_full_ids,
                mode="mlp",
                rank_cap=None,
                gamma=0.25,
                healthy_ids=prose_full_ids,
                base_healthy_logp=base_prose_logp,
                base_healthy_nll=base_prose_nll,
            )
            res_mlp.pop("svd_factors", None)
            w_mlp = res_mlp.pop("weights")
            with _temp_apply_weights(student_m, w_mlp):
                domain_mmlu_scores["closedform_mlp_seq24L"][sk] = _score_routed_mmlu_sparse(
                    student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                )
            p_mlp = exp_dir / f"cf_mlp_{sk}.pt"
            torch.save(w_mlp, p_mlp)
            del w_mlp
            expert_paths[("closedform_mlp_seq24L", sk)] = p_mlp
            expert_meta["closedform_mlp_seq24L"][sk] = res_mlp

            # 2) Closed-form Full-Block (mixer_out + mlp_down, 0 backprop)
            res_fb = _solve_sequential_closedform(
                student_m,
                aligned_sk,
                target_outs,
                target_mids=target_mids,
                anchor_ids=anchor_full_ids,
                mode="fullblock",
                rank_cap=None,
                gamma=0.25,
                healthy_ids=prose_full_ids,
                base_healthy_logp=base_prose_logp,
                base_healthy_nll=base_prose_nll,
            )
            res_fb.pop("svd_factors", None)
            w_fb = res_fb.pop("weights")
            with _temp_apply_weights(student_m, w_fb):
                domain_mmlu_scores["closedform_fullblock_seq24L"][sk] = _score_routed_mmlu_sparse(
                    student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                )
            p_fb = exp_dir / f"cf_fb_{sk}.pt"
            torch.save(w_fb, p_fb)
            del w_fb
            expert_paths[("closedform_fullblock_seq24L", sk)] = p_fb
            expert_meta["closedform_fullblock_seq24L"][sk] = res_fb

            # 3) Closed-form Full-Block SVD rank=16 (0 backprop)
            res_svd = _solve_sequential_closedform(
                student_m,
                aligned_sk,
                target_outs,
                target_mids=target_mids,
                anchor_ids=anchor_full_ids,
                mode="fullblock",
                rank_cap=16,
                gamma=0.25,
                healthy_ids=prose_full_ids,
                base_healthy_logp=base_prose_logp,
                base_healthy_nll=base_prose_nll,
            )
            svd_fac = res_svd.pop("svd_factors")
            w_svd = res_svd.pop("weights")
            with _temp_apply_weights(student_m, w_svd):
                domain_mmlu_scores["closedform_svd_r16"][sk] = _score_routed_mmlu_sparse(
                    student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                )
            p_svd = exp_dir / f"cf_svd_{sk}.pt"
            torch.save(w_svd, p_svd)
            del w_svd, target_outs, target_mids
            expert_paths[("closedform_svd_r16", sk)] = p_svd
            expert_meta["closedform_svd_r16"][sk] = res_svd

            # 4) Matched 8-step Warm-Start LoRA from closedform_svd_r16 factors
            res_ws = _train_matched_8step_lora(
                student_m,
                target_specs,
                multitrace_samples[sk],
                rank=16,
                alpha_scale=32.0,
                total_steps=8,
                grad_accum=3,
                lr_val=2.0e-4,
                weight_decay=0.05,
                lora_dropout=0.05,
                warmup_frac=0.25,
                distill_T=2.0,
                kl_weight=0.85,
                init_factors=svd_fac,
                anchor_ids=anchor_full_ids,
                base_anchor_logp=base_anchor_logp,
                anchor_kl_weight=0.05,
                healthy_ids=prose_full_ids,
                base_healthy_logp=base_prose_logp,
                base_healthy_nll=base_prose_nll,
                seed=int(args.seed) + 500 + s_idx,
            )
            del svd_fac
            w_ws = res_ws.pop("weights")
            with _temp_apply_weights(student_m, w_ws):
                domain_mmlu_scores["hybrid_warmstart_8steps"][sk] = _score_routed_mmlu_sparse(
                    student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                )
            p_ws = exp_dir / f"warm8_{sk}.pt"
            torch.save(w_ws, p_ws)
            del w_ws
            expert_paths[("hybrid_warmstart_8steps", sk)] = p_ws
            expert_meta["hybrid_warmstart_8steps"][sk] = res_ws

            # 5) Exact Matched 8-step Cold-Start LoRA
            res_cs = _train_matched_8step_lora(
                student_m,
                target_specs,
                multitrace_samples[sk],
                rank=16,
                alpha_scale=32.0,
                total_steps=8,
                grad_accum=3,
                lr_val=2.0e-4,
                weight_decay=0.05,
                lora_dropout=0.05,
                warmup_frac=0.25,
                distill_T=2.0,
                kl_weight=0.85,
                init_factors=None,
                anchor_ids=anchor_full_ids,
                base_anchor_logp=base_anchor_logp,
                anchor_kl_weight=0.05,
                healthy_ids=prose_full_ids,
                base_healthy_logp=base_prose_logp,
                base_healthy_nll=base_prose_nll,
                seed=int(args.seed) + 600 + s_idx,
            )
            w_cs = res_cs.pop("weights")
            with _temp_apply_weights(student_m, w_cs):
                domain_mmlu_scores["coldstart_8steps"][sk] = _score_routed_mmlu_sparse(
                    student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                )
            p_cs = exp_dir / f"cold8_{sk}.pt"
            torch.save(w_cs, p_cs)
            del w_cs
            expert_paths[("coldstart_8steps", sk)] = p_cs
            expert_meta["coldstart_8steps"][sk] = res_cs

            print(
                f"[cf] {sk:26s} | mlp_r2={res_mlp['mean_val_r2']:.3f} ({res_mlp['compile_time_s']}s, kl={res_mlp['healthy_kl']:.4f}) | "
                f"fb_r2={res_fb['mean_val_r2']:.3f} ({res_fb['compile_time_s']}s, kl={res_fb['healthy_kl']:.4f}) | "
                f"svd_r16={res_svd['adapter_mb']}MB ({res_svd['compile_time_s']}s) | "
                f"warm8={res_ws['step_losses'][-1]:.3f} vs cold8={res_cs['step_losses'][-1]:.3f}",
                flush=True,
            )

        mmlu_arms: dict[str, Any] = {"base_08b": base_mmlu_summary}
        for idx_t, t_nm in enumerate(routed_tier_names):
            mmlu_arms[f"{t_nm}_routed"] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_scores[t_nm], seed=int(args.seed) + 1 + idx_t
            )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Evaluate All 8 Paired Arms on 84 Test Prompts ----
        print(f"[cf] Phase 4: Evaluating 8 paired arms ({backend}, max_new_tokens={args.max_new_tokens})...", flush=True)
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
            bo_outs = _vllm_generate(args.student, tok_s, test_pids + oracle_pids, test_seeds + test_seeds, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
            base_raw = bo_outs[: len(all_test)]
            oracle_raw = bo_outs[len(all_test) :]
        else:
            base_raw = _hf_generate_subset(student_m, tok_s, test_pids, test_seeds, max_new_tokens=args.max_new_tokens)
            oracle_raw = _hf_generate_subset(student_m, tok_s, oracle_pids, test_seeds, max_new_tokens=args.max_new_tokens)

        bake_tmp_dir = Path("/content/_pocket_moe_cf_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

        def _eval_ckpt(weights_path: Path, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
            upd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if backend == "vllm":
                _bake_full_checkpoint(Path(args.student), upd, bake_tmp_dir)
                del upd
                gc.collect()
                return _vllm_generate(str(bake_tmp_dir), tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
            return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, param_updates=upd)

        routed_raw_by_tier: dict[str, list[dict[str, Any]]] = {}
        for t_nm in routed_tier_names:
            tier_raw: list[dict[str, Any] | None] = [None] * len(all_test)
            for i, cls_nm in enumerate(routed_test_classes):
                if cls_nm == "passthrough":
                    tier_raw[i] = base_raw[i]
            for sk in SKILLS_5:
                idxs = [i for i, cls_nm in enumerate(routed_test_classes) if cls_nm == sk]
                p_exp = expert_paths[(t_nm, sk)]
                if idxs:
                    sub_outs = _eval_ckpt(p_exp, [test_pids[i] for i in idxs], [test_seeds[i] for i in idxs])
                    for idx_i, o in zip(idxs, sub_outs):
                        tier_raw[idx_i] = o
                p_exp.unlink(missing_ok=True)
            routed_raw_by_tier[t_nm] = [r for r in tier_raw if r is not None]

        if bake_tmp_dir.exists():
            shutil.rmtree(bake_tmp_dir, ignore_errors=True)
        phase4_s = round(time.perf_counter() - t_p4, 2)

    arm_raw_map: dict[str, tuple[list[dict[str, Any]], list[str] | None, bool]] = {
        "base_08b": (base_raw, None, False),
        "teacher_4b": (t_test_raw, None, True),
        "oracle_1shot_08b": (oracle_raw, None, False),
        **{f"{t_nm}_routed": (routed_raw_by_tier[t_nm], routed_test_classes, False) for t_nm in routed_tier_names},
    }

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

    prose_kl = {f"{t_nm}_routed": _routed_prose_kl(t_nm) for t_nm in routed_tier_names}
    compile_time_s = {
        f"{t_nm}_mean_per_skill_s": round(sum(expert_meta[t_nm][s]["compile_time_s"] for s in SKILLS_5) / len(SKILLS_5), 2)
        for t_nm in routed_tier_names
    }

    zero_backprop_arms = [
        "closedform_mlp_seq24L_routed",
        "closedform_fullblock_seq24L_routed",
        "closedform_svd_r16_routed",
    ]
    best_0backprop_arm = max(
        zero_backprop_arms,
        key=lambda k: (arm_summaries[k]["hard_60"]["strict_accuracy"], arm_summaries[k]["guardrail_24"]["strict_accuracy"]),
    )
    routed_arm_names = [f"{t_nm}_routed" for t_nm in routed_tier_names]

    comp_pairs = [
        ("closedform_mlp_vs_base", "closedform_mlp_seq24L_routed", "base_08b"),
        ("closedform_fullblock_vs_base", "closedform_fullblock_seq24L_routed", "base_08b"),
        ("closedform_fullblock_vs_mlp", "closedform_fullblock_seq24L_routed", "closedform_mlp_seq24L_routed"),
        ("closedform_svd_r16_vs_base", "closedform_svd_r16_routed", "base_08b"),
        ("closedform_svd_r16_vs_fullblock", "closedform_svd_r16_routed", "closedform_fullblock_seq24L_routed"),
        ("closedform_fullblock_vs_oracle_1shot", "closedform_fullblock_seq24L_routed", "oracle_1shot_08b"),
        ("hybrid_warmstart_vs_coldstart_8steps", "hybrid_warmstart_8steps_routed", "coldstart_8steps_routed"),
        ("hybrid_warmstart_vs_svd_r16", "hybrid_warmstart_8steps_routed", "closedform_svd_r16_routed"),
        ("hybrid_warmstart_vs_oracle_1shot", "hybrid_warmstart_8steps_routed", "oracle_1shot_08b"),
        ("best_0backprop_vs_base", best_0backprop_arm, "base_08b"),
        ("best_0backprop_vs_oracle_1shot", best_0backprop_arm, "oracle_1shot_08b"),
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
        "backend": backend,
        "k_shots_per_skill": int(args.k_shots),
        "max_new_tokens": int(args.max_new_tokens),
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_wsi_router_s": phase2_s,
            "phase3_compile_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "compile_time_s": compile_time_s,
        "teacher_4b_train_kept_frac": teacher_kept_frac,
        "multitrace_counts": multitrace_counts,
        "router_report": router_report,
        "expert_meta": {
            tier_nm: {
                sk: {k: v for k, v in meta.items() if k != "layer_stats"} | {"n_layer_records": len(meta.get("layer_stats", []))}
                for sk, meta in sk_map.items()
            }
            for tier_nm, sk_map in expert_meta.items()
        },
        "layer_stats_sample": {
            t_nm: expert_meta[t_nm][SKILLS_5[0]].get("layer_stats", [])
            for t_nm in ("closedform_mlp_seq24L", "closedform_fullblock_seq24L", "closedform_svd_r16")
        },
        "prose_kl": prose_kl,
        "generation_arms": arm_summaries,
        "mmlu_arms": mmlu_arms,
        "paired_hard_60": paired_hard_60,
        "paired_old_skills_48": paired_old_skills_48,
        "best_0backprop_arm_name": best_0backprop_arm,
        "criteria": {
            "a_0backprop_gains_20pts_over_base": bool(h_acc(best_0backprop_arm) >= h_acc("base_08b") + 0.20),
            "b_fullblock_beats_mlp": bool(h_acc("closedform_fullblock_seq24L_routed") > h_acc("closedform_mlp_seq24L_routed")),
            "c_svd_r16_retains_closedform": bool(h_acc("closedform_svd_r16_routed") >= h_acc("closedform_fullblock_seq24L_routed") - 0.05),
            "d_warmstart_beats_coldstart_8steps": bool(h_acc("hybrid_warmstart_8steps_routed") > h_acc("coldstart_8steps_routed")),
            "e_zero_general_forgetting_all_routed": bool(
                all(
                    prose_kl[nm] <= 0.005
                    and nm_mmlu(nm) >= nm_mmlu("base_08b") - 0.015
                    and g_acc(nm) >= g_acc("base_08b") - 0.02
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
        _print_closedform_summary(payload)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
