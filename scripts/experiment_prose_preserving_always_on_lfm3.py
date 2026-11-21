"""Prose-Preserving `always_on` Surgical Refusal Editing (`healthy_fisher` + Sparse Neurons + Late Layers).

Tests empirically on `LiquidAI/LFM3-1.7B` (zero toxic data; strictly `OR-Bench-Hard` + everyday
benign controls) whether tracing healthy prose generation (`healthy_cost="fisher"`, `grad_mode="compressed"`)
plus restricting edits to sparse top-k neurons and late blocks cuts `always_on` drift from
`healthy_kl = 0.00925` down toward `<= 0.001-0.002` without a runtime router while keeping strong
over-refusal recovery.

CPU smoke test::
    python scripts/experiment_prose_preserving_always_on_lfm3.py --smoke

Colab L4 run::
    python scripts/experiment_prose_preserving_always_on_lfm3.py --model /content/base --baseline-patch /content/drive/MyDrive/wsi_runs/surgical_refusal_lfm3/lfm3_safety_routed_refusal.moipatch --out /content/prose_preserving_always_on_lfm3_l4.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
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

import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
import experiment_veto_robustness_lfm3 as vr  # noqa: E402
from wsi.moipatch import Edit, MoIPatch  # noqa: E402
from wsi.tracer import (  # noqa: E402
    CompactTrace,
    capture_causal_trace,
    diagnose_causal_traces,
    screen_patch_candidates,
)


@torch.no_grad()
def _measure_prose_preservation(
    model: nn.Module,
    patch: MoIPatch | None,
    control_rows: list[dict[str, Any]],
    dev: torch.device,
) -> dict[str, float]:
    """Measure full-sequence `healthy_kl`, response-only `prose_response_kl`, and `prose_top1_unchanged_rate`."""
    was_grad = torch.is_grad_enabled()
    was_training = bool(model.training)
    model.eval()

    base_lps: list[torch.Tensor] = []
    base_argmax: list[torch.Tensor] = []
    base_nll_sum = 0.0
    n_tot = 0
    for r in control_rows:
        ids = torch.tensor([r["full_ids"]], dtype=torch.long, device=dev)
        if ids.shape[1] < 2:
            base_lps.append(torch.empty((0, 1)))
            base_argmax.append(torch.empty(0, dtype=torch.long))
            continue
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        lp = F.log_softmax(logits, dim=-1)
        base_nll_sum += float(F.nll_loss(lp, ids[0, 1:], reduction="sum"))
        n_tot += int(ids.shape[1] - 1)
        base_lps.append(lp.to(dtype=torch.bfloat16, device="cpu"))
        base_argmax.append(logits.argmax(dim=-1).cpu())

    base_nll = base_nll_sum / max(1, n_tot)
    if patch is None:
        if was_training:
            model.train()
        torch.set_grad_enabled(was_grad)
        return {
            "healthy_nll": round(base_nll, 5),
            "delta_healthy_nll": 0.0,
            "healthy_kl": 0.0,
            "prose_response_kl": 0.0,
            "prose_top1_unchanged_rate": 1.0,
        }

    arm_nll_sum = 0.0
    full_kl_sum = 0.0
    resp_kl_sum = 0.0
    resp_tok_total = 0
    resp_top1_same = 0

    h = patch.attach(model)
    try:
        h.on()
        for idx, r in enumerate(control_rows):
            ids = torch.tensor([r["full_ids"]], dtype=torch.long, device=dev)
            if ids.shape[1] < 2:
                continue
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
            lp = F.log_softmax(logits, dim=-1)
            arm_nll_sum += float(F.nll_loss(lp, ids[0, 1:], reduction="sum"))

            lp_b = base_lps[idx].to(device=dev, dtype=torch.float32)
            kl_vec = (lp_b.exp() * (lp_b - lp)).sum(dim=-1).clamp_min(0.0)
            full_kl_sum += float(kl_vec.sum())

            p_len = len(r.get("prompt_ids") or ())
            r_start = max(0, p_len - 1)
            r_end = kl_vec.shape[0]
            if r_end > r_start:
                resp_kl_sum += float(kl_vec[r_start:r_end].sum())
                resp_tok_total += int(r_end - r_start)
                arm_top1 = logits[r_start:r_end].argmax(dim=-1).cpu()
                resp_top1_same += int((arm_top1 == base_argmax[idx][r_start:r_end]).sum().item())
    finally:
        if h.active:
            h.off()
        if was_training:
            model.train()
        torch.set_grad_enabled(was_grad)

    arm_nll = arm_nll_sum / max(1, n_tot)
    return {
        "healthy_nll": round(arm_nll, 5),
        "delta_healthy_nll": round(arm_nll - base_nll, 5),
        "healthy_kl": round(full_kl_sum / max(1, n_tot), 5),
        "prose_response_kl": round(resp_kl_sum / max(1, resp_tok_total), 5),
        "prose_top1_unchanged_rate": round(resp_top1_same / max(1, resp_tok_total), 4),
    }


def _evaluate_always_on_arm(
    model: nn.Module,
    tok: Any,
    patch: MoIPatch | None,
    or_b_rows: list[dict[str, Any]],
    ctrl_b_rows: list[dict[str, Any]],
    base_or_b: list[dict[str, Any]],
    base_ctrl_b: list[dict[str, Any]],
    dev: torch.device,
    *,
    max_new_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate one `always_on` arm on held-out OR-Bench Half-B recovery + control prose drift."""
    or_prompts = [r["formatted"] for r in or_b_rows]
    ctrl_prompts = [r["formatted"] for r in ctrl_b_rows]

    if patch is None:
        or_out = base_or_b
        ctrl_out = base_ctrl_b
    else:
        with patch.enable_routed(model, mode="always_on"):
            or_out = sr._generate_texts(model, tok, or_prompts, dev, max_new_tokens=max_new_tokens, batch_size=batch_size)
            ctrl_out = sr._generate_texts(model, tok, ctrl_prompts, dev, max_new_tokens=max_new_tokens, batch_size=batch_size)

    prose = _measure_prose_preservation(model, patch, base_ctrl_b, dev)
    base_ref_idx = [i for i, r in enumerate(base_or_b) if r["refused"]]
    recovered = (
        sum(1 for i in base_ref_idx if not or_out[i]["refused"]) / len(base_ref_idx)
        if base_ref_idx
        else None
    )
    ctrl_exact = sum(1 for a, b in zip(ctrl_out, base_ctrl_b) if a["text"] == b["text"]) / max(1, len(base_ctrl_b))
    ctrl_first16 = (
        sum(1 for a, b in zip(ctrl_out, base_ctrl_b) if a["gen_ids"][:16] == b["gen_ids"][:16]) / max(1, len(base_ctrl_b))
    )
    edited_weights = (
        sum(
            int(e.shape[0]) * (len(e.idx) if e.idx is not None else int(e.shape[1]))
            if e.shape
            else int(e.u.numel())
            for e in patch.edits
        )
        if patch is not None
        else 0
    )
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "n_or_bench_test": len(or_out),
        "n_control_test": len(ctrl_out),
        "or_bench_refusal_rate": round(sum(1 for r in or_out if r["refused"]) / max(1, len(or_out)), 4),
        "n_refused": int(sum(1 for r in or_out if r["refused"])),
        "recovered_of_base_refused": round(recovered, 4) if recovered is not None else None,
        "control_exact_match_rate": round(ctrl_exact, 4),
        "control_first16_exact_match_rate": round(ctrl_first16, 4),
        "control_refusal_rate": round(sum(1 for r in ctrl_out if r["refused"]) / max(1, len(ctrl_out)), 4),
        "n_edited_modules": len(patch.edits) if patch is not None else 0,
        "edited_modules": [e.module for e in patch.edits] if patch is not None else [],
        "edited_weights": int(edited_weights),
        "edited_param_frac": round(edited_weights / max(1, total_params), 6),
        **prose,
    }


def _interleave_refusal_and_prose_traces(
    ref_traces: list[CompactTrace],
    prose_traces: list[CompactTrace],
) -> list[CompactTrace]:
    """Pair-interleave refusal & prose traces so both `0::2` and `1::2` CV splits receive balanced signals."""
    out: list[CompactTrace] = []
    n_pairs = max((len(ref_traces) + 1) // 2, (len(prose_traces) + 1) // 2)
    for pair_i in range(n_pairs):
        for off in range(2):
            idx = 2 * pair_i + off
            if idx < len(ref_traces):
                out.append(ref_traces[idx])
        for off in range(2):
            idx = 2 * pair_i + off
            if idx < len(prose_traces):
                out.append(prose_traces[idx])
    return out


def _build_sparse_candidate_grid(
    diag_fisher: dict[str, Any],
    writers_by_mod: dict[str, sr.WriterInfo],
    n_blocks: int,
    *,
    diag_linear: dict[str, Any] | None = None,
    neuron_budgets: tuple[int, ...] = (16, 32, 64, 128),
    candidate_alphas: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0),
) -> tuple[dict[str, MoIPatch], dict[str, str]]:
    """Build Half-A sparse candidate grid combining Fisher-safe neuron masks with Fisher/linear directions."""
    all_sites = [s for s in diag_fisher.get("all_sites", diag_fisher["sites"]) if s["module"] in writers_by_mod]
    any_sites = all_sites[:6]
    late_cut = n_blocks // 2
    late_sites = [s for s in all_sites if int(s["layer"]) >= late_cut][:6]

    templates: dict[str, list[tuple[str, torch.Tensor, torch.Tensor | None]]] = {}
    bucket_of_name: dict[str, str] = {}

    def _dir_variants(mk: str) -> list[tuple[str, torch.Tensor]]:
        u_f = diag_fisher["directions"][mk].double()
        out = [("f", u_f)]
        if diag_linear is not None and mk in diag_linear.get("directions", {}):
            u_l = diag_linear["directions"][mk].double()
            cos = float((u_f @ u_l).item() / (u_f.norm().clamp_min(1e-12) * u_l.norm().clamp_min(1e-12)).item())
            if abs(cos) < 0.98:
                out.append(("l", u_l))
        return out

    def _add_site_pool(prefix: str, pool: list[dict[str, Any]], *, include_dense_single: bool) -> None:
        if not pool:
            return
        for rank_i, s_row in enumerate(pool, start=1):
            mk = s_row["module"]
            d_in = int(writers_by_mod[mk].shape[1]) if len(writers_by_mod[mk].shape) > 1 else int(diag_fisher["directions"][mk].numel())
            top_cols = diag_fisher["top_neurons"].get(mk, [])
            for d_tag, u_vec in _dir_variants(mk):
                tag_suf = "" if d_tag == "f" else "_lin"
                for nb in neuron_budgets:
                    if top_cols:
                        k_take = min(int(nb), len(top_cols), d_in)
                        if k_take <= 0:
                            continue
                        t_name = f"{prefix}_r{rank_i}{tag_suf}_n{nb}"
                        templates[t_name] = [(mk, u_vec, torch.tensor(top_cols[:k_take], dtype=torch.long))]
                        bucket_of_name[t_name] = prefix
                if include_dense_single and rank_i <= 2:
                    d_name = f"{prefix}_r{rank_i}{tag_suf}_all"
                    templates[d_name] = [(mk, u_vec, None)]
                    bucket_of_name[d_name] = f"{prefix}_dense"

        for k_grp, grp_lbl in ((2, "pair"), (3, "triple")):
            if len(pool) >= k_grp:
                grp = pool[:k_grp]
                for nb in neuron_budgets:
                    specs = []
                    for s_row in grp:
                        mk = s_row["module"]
                        u_vec = diag_fisher["directions"][mk].double()
                        top_cols = diag_fisher["top_neurons"].get(mk, [])
                        d_in = int(writers_by_mod[mk].shape[1]) if len(writers_by_mod[mk].shape) > 1 else int(u_vec.numel())
                        idx_t = torch.tensor(top_cols[: min(int(nb), len(top_cols), d_in)], dtype=torch.long) if top_cols else None
                        specs.append((mk, u_vec, idx_t))
                    if any(sp[2] is not None for sp in specs):
                        g_name = f"{prefix}_{grp_lbl}_n{nb}"
                        templates[g_name] = specs
                        bucket_of_name[g_name] = prefix

    _add_site_pool("fisher_any", any_sites, include_dense_single=False)
    _add_site_pool("fisher_late", late_sites, include_dense_single=True)

    candidates: dict[str, MoIPatch] = {}
    cand_bucket: dict[str, str] = {}
    for t_name, specs in templates.items():
        # Deduplicate identical edit lists across any/late overlap while retaining bucket metadata
        for alpha in candidate_alphas:
            cname = f"{t_name}_a{alpha:g}"
            edits = [
                Edit(
                    module=mk,
                    u=u_vec,
                    alpha=float(alpha),
                    idx=idx_t,
                    base_sha256=writers_by_mod[mk].base_sha256,
                    base_dtype=writers_by_mod[mk].base_dtype,
                    shape=writers_by_mod[mk].shape,
                )
                for mk, u_vec, idx_t in specs
            ]
            candidates[cname] = MoIPatch(edits=edits, meta={"name": cname})
            cand_bucket[cname] = bucket_of_name[t_name]

    return candidates, cand_bucket


def _select_arm_winners(
    screened: list[dict[str, Any]],
    candidates: dict[str, MoIPatch],
    cand_bucket: dict[str, str],
    model: nn.Module,
    ref_trace_rows: list[dict[str, Any]],
    dev: torch.device,
    n_blocks: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, MoIPatch]]:
    """Select `fisher_sparse_any`, `fisher_sparse_late`, and `fisher_ultra_low_kl` strictly from Half-A."""
    late_cut = n_blocks // 2
    base_onset = sr._compute_base_onset(model, ref_trace_rows, dev)

    def _is_sparse(patch: MoIPatch) -> bool:
        return all(e.idx is not None for e in patch.edits)

    def _is_late(patch: MoIPatch) -> bool:
        return all(int(e.module.split("layers.", 1)[1].split(".", 1)[0]) >= late_cut for e in patch.edits)

    passed = [r for r in screened if r.get("passes_nll_guard")] or list(screened)
    any_sparse_pool = [r for r in passed if _is_sparse(candidates[r["name"]]) and r["healthy_kl"] <= 0.005]
    if not any_sparse_pool:
        any_sparse_pool = [r for r in passed if _is_sparse(candidates[r["name"]])]

    late_sparse_pool = [
        r for r in passed if _is_sparse(candidates[r["name"]]) and _is_late(candidates[r["name"]]) and r["healthy_kl"] <= 0.003
    ]
    if not late_sparse_pool:
        late_sparse_pool = [r for r in passed if _is_sparse(candidates[r["name"]]) and _is_late(candidates[r["name"]])]
    if not late_sparse_pool:
        late_sparse_pool = [r for r in passed if _is_late(candidates[r["name"]])]

    ultra_pool = [r for r in passed if r["healthy_kl"] <= 0.001]
    if not ultra_pool:
        ultra_pool = [r for r in passed if r["healthy_kl"] <= 0.002]
    if not ultra_pool:
        ultra_pool = sorted(passed, key=lambda r: (r["healthy_kl"], -r["target_suppression"]))[:8]

    probe_names: list[str] = []
    for pool in (any_sparse_pool[:6], late_sparse_pool[:6], sorted(ultra_pool, key=lambda r: -r["target_suppression"])[:6]):
        for r in pool:
            if r["name"] not in probe_names:
                probe_names.append(r["name"])

    onset_by_name: dict[str, dict[str, float]] = {}
    for nm in probe_names:
        row_s = next((r for r in screened if r["name"] == nm), None)
        st = sr._measure_onset_suppression(model, candidates[nm], ref_trace_rows, dev, base_onset=base_onset)
        if row_s is not None:
            row_s.update(st)
        onset_by_name[nm] = st

    def _pick_from(pool: list[dict[str, Any]], *, prefer_low_kl: bool = False) -> dict[str, Any]:
        eval_rows = [r for r in pool if r["name"] in onset_by_name] or pool
        strong = [r for r in eval_rows if r.get("onset_top1_flip_rate", 0.0) >= 0.50 and r.get("onset_logp_drop", 0.0) > 0.0]
        if strong:
            if prefer_low_kl:
                return max(
                    strong,
                    key=lambda r: (r["onset_top1_flip_rate"], round(r.get("onset_logp_drop", 0.0), 3), -r["healthy_kl"]),
                )
            return strong[0]
        positive = [r for r in eval_rows if r.get("onset_logp_drop", 0.0) > 0.0]
        if positive:
            return max(
                positive,
                key=lambda r: (r.get("onset_top1_flip_rate", 0.0), r.get("onset_logp_drop", 0.0), -r["healthy_kl"]),
            )
        return eval_rows[0]

    sel_meta: dict[str, dict[str, Any]] = {}
    sel_patches: dict[str, MoIPatch] = {}

    w_any = _pick_from(any_sparse_pool)
    w_late = _pick_from(late_sparse_pool)
    w_ultra = _pick_from(ultra_pool, prefer_low_kl=True)

    for arm_lbl, w_row in (
        ("fisher_sparse_any", w_any),
        ("fisher_sparse_late", w_late),
        ("fisher_ultra_low_kl", w_ultra),
    ):
        nm = w_row["name"]
        sel_meta[arm_lbl] = {
            "candidate": nm,
            "bucket": cand_bucket.get(nm, arm_lbl),
            "modules": [e.module for e in candidates[nm].edits],
            "alpha": float(candidates[nm].edits[0].alpha),
            "n_cols": [len(e.idx) if e.idx is not None else "all" for e in candidates[nm].edits],
            **w_row,
        }
        sel_patches[arm_lbl] = candidates[nm]

    return sel_meta, sel_patches


def _run_prose_experiment(
    model: nn.Module,
    tok: Any,
    or_prompts: list[str],
    ctrl_prompts: list[str],
    dev: torch.device,
    *,
    baseline_patch: MoIPatch | None = None,
    patch_out_path: Path | None = None,
    n_traces: int = 8,
    n_prose_traces: int = 8,
    prose_max_gen: int = 96,
    discovery_gen_window: int = 12,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    neuron_budgets: tuple[int, ...] = (16, 32, 64, 128),
    candidate_alphas: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0),
) -> dict[str, Any]:
    t_all = time.perf_counter()
    n_blocks, writers = sr._discover_all_writers(model)
    writers_by_mod = {w.suffix: w for w in writers}

    or_items = [{"id": i, "raw": p, "formatted": sr._format_prompt(tok, p, None)} for i, p in enumerate(or_prompts)]
    ctrl_items = [{"id": i, "raw": p, "formatted": sr._format_prompt(tok, p, None)} for i, p in enumerate(ctrl_prompts)]

    probe_fmt = [x["formatted"] for x in or_items[: min(8, len(or_items))]]
    eff_bs = vr._probe_left_pad_safe(model, tok, probe_fmt, dev, batch_size=batch_size) if len(probe_fmt) >= 2 else 1
    print(f"[prose-exp] n_blocks={n_blocks} writers={len(writers)} eff_bs={eff_bs}", flush=True)

    # 1. Base generation + exact 50/50 Half-A / Half-B split (`seed=42` consistent)
    t_base = time.perf_counter()
    base_or_all = sr._generate_texts(
        model, tok, [x["formatted"] for x in or_items], dev, max_new_tokens=max_new_tokens, batch_size=eff_bs
    )
    ref_idx = [i for i, r in enumerate(base_or_all) if r["refused"]]
    nonref_idx = [i for i, r in enumerate(base_or_all) if not r["refused"]]
    or_a_idx = ref_idx[0::2] + nonref_idx[0::2]
    or_b_idx = ref_idx[1::2] + nonref_idx[1::2]

    ctrl_a_idx = list(range(0, len(ctrl_items), 2))
    ctrl_b_idx = list(range(1, len(ctrl_items), 2))
    ctrl_a_out = sr._generate_texts(
        model, tok, [ctrl_items[i]["formatted"] for i in ctrl_a_idx], dev, max_new_tokens=max_new_tokens, batch_size=eff_bs
    )
    base_ctrl_b = sr._generate_texts(
        model, tok, [ctrl_items[i]["formatted"] for i in ctrl_b_idx], dev, max_new_tokens=max_new_tokens, batch_size=eff_bs
    )
    or_a_ref_rows = [base_or_all[i] for i in or_a_idx if base_or_all[i]["refused"]]
    base_s = round(time.perf_counter() - t_base, 2)
    print(
        f"[base] or_ref={sum(1 for r in base_or_all if r['refused'])}/{len(base_or_all)} "
        f"half_a_ref={len(or_a_ref_rows)} half_b_ref={sum(1 for i in or_b_idx if base_or_all[i]['refused'])} ({base_s}s)",
        flush=True,
    )

    # 2. Capture compressed per-token causal traces (refusal onset + healthy prose writing) on Half-A
    t_trace = time.perf_counter()
    ref_trace_rows = [r for r in or_a_ref_rows if len(r["gen_ids"]) >= 1][:n_traces]
    if not ref_trace_rows:
        ref_trace_rows = [base_or_all[i] for i in or_a_idx if len(base_or_all[i]["gen_ids"]) >= 1][:n_traces]

    ref_traces: list[CompactTrace] = []
    ref_clip_ids: list[list[int]] = []
    ref_clip_spans: list[tuple[int, int]] = []
    for r in ref_trace_rows:
        t_ids, t_span = sr._clip_row_for_discovery(r, discovery_gen_window)
        if t_span[1] <= t_span[0]:
            t_span = (max(0, len(t_ids) - 1), len(t_ids))
        ref_clip_ids.append(t_ids)
        ref_clip_spans.append(t_span)
        ref_traces.append(
            capture_causal_trace(
                model,
                t_ids,
                target_span=t_span,
                healthy_span=(0, 0),
                grad_mode="compressed",
                grad_checkpointing=True,
                top_k_mlp=64,
                top_k_attn=0,
                capture_cache=False,
            )
        )

    prose_traces: list[CompactTrace] = []
    prose_rows = [r for r in ctrl_a_out if len(r["gen_ids"]) >= 2][:n_prose_traces]
    for r in prose_rows:
        p_ids = list(r["prompt_ids"])
        g_ids = list(r["gen_ids"][:prose_max_gen])
        full_c = p_ids + g_ids
        p_len = len(p_ids)
        ct_p = capture_causal_trace(
            model,
            full_c,
            target_span=(p_len, min(len(full_c), p_len + 1)),
            healthy_span=(p_len, len(full_c)),
            grad_mode="compressed",
            grad_checkpointing=True,
            top_k_mlp=64,
            top_k_attn=0,
            capture_cache=False,
        )
        # Exclude healthy-prose dummy target span from C_target / n_supp pooling
        ct_p.meta["grad_n_target"] = 0
        prose_traces.append(ct_p)

    interleaved = _interleave_refusal_and_prose_traces(ref_traces, prose_traces)
    diag_fisher = diagnose_causal_traces(
        interleaved,
        split_ab=(len(ref_traces) >= 2 and len(prose_traces) >= 2),
        top_k_sites=12,
        top_n_neurons=max(neuron_budgets),
        healthy_penalty=2.0,
        healthy_cost="fisher",
        center="none",
        model_or_weights=model,
    )
    diag_linear = diagnose_causal_traces(
        interleaved,
        split_ab=(len(ref_traces) >= 2 and len(prose_traces) >= 2),
        top_k_sites=12,
        top_n_neurons=max(neuron_budgets),
        healthy_penalty=1.0,
        healthy_cost="linear",
        center="none",
        model_or_weights=model,
    )
    trace_s = round(time.perf_counter() - t_trace, 2)
    print(
        f"[traces] ref={len(ref_traces)} prose={len(prose_traces)} ({trace_s}s) | "
        f"fisher_top={[s['module'] for s in diag_fisher['sites'][:4]]} | "
        f"linear_top={[s['module'] for s in diag_linear['sites'][:4]]}",
        flush=True,
    )

    # 3. Screen sparse candidate patches on Half-A via Pareto-KL
    t_screen = time.perf_counter()
    candidates, cand_bucket = _build_sparse_candidate_grid(
        diag_fisher,
        writers_by_mod,
        n_blocks,
        diag_linear=diag_linear,
        neuron_budgets=neuron_budgets,
        candidate_alphas=candidate_alphas,
    )
    ctrl_a_full_ids = [r["full_ids"] for r in ctrl_a_out[:12]]
    screened = screen_patch_candidates(
        model,
        candidates,
        healthy_seqs=ctrl_a_full_ids,
        target_seqs=ref_clip_ids[:12],
        target_spans=ref_clip_spans[:12],
        max_nll_delta=0.05,
        rank_by="pareto_kl",
    )
    sel_meta, sel_patches = _select_arm_winners(
        screened, candidates, cand_bucket, model, ref_trace_rows, dev, n_blocks
    )
    screen_s = round(time.perf_counter() - t_screen, 2)
    for k_arm, meta_arm in sel_meta.items():
        print(
            f"[pick {k_arm}] {meta_arm['candidate']} | mods={meta_arm['modules']} cols={meta_arm['n_cols']} "
            f"screen_kl={meta_arm['healthy_kl']} flip={meta_arm.get('onset_top1_flip_rate')}",
            flush=True,
        )

    # 4. Held-out Half-B paired `always_on` evaluation
    t_eval = time.perf_counter()
    or_b_rows = [or_items[i] for i in or_b_idx]
    ctrl_b_rows = [ctrl_items[i] for i in ctrl_b_idx]
    base_or_b = [base_or_all[i] for i in or_b_idx]

    arms: dict[str, dict[str, Any]] = {}
    arms["base"] = _evaluate_always_on_arm(
        model, tok, None, or_b_rows, ctrl_b_rows, base_or_b, base_ctrl_b, dev,
        max_new_tokens=max_new_tokens, batch_size=eff_bs,
    )
    if baseline_patch is not None:
        arms["baseline_always_on"] = _evaluate_always_on_arm(
            model, tok, baseline_patch, or_b_rows, ctrl_b_rows, base_or_b, base_ctrl_b, dev,
            max_new_tokens=max_new_tokens, batch_size=eff_bs,
        )
    for arm_name in ("fisher_sparse_any", "fisher_sparse_late", "fisher_ultra_low_kl"):
        arms[arm_name] = _evaluate_always_on_arm(
            model, tok, sel_patches[arm_name], or_b_rows, ctrl_b_rows, base_or_b, base_ctrl_b, dev,
            max_new_tokens=max_new_tokens, batch_size=eff_bs,
        )
        m_a = arms[arm_name]
        print(
            f"[eval {arm_name}] recov={m_a['recovered_of_base_refused']} | "
            f"healthy_kl={m_a['healthy_kl']} prose_kl={m_a['prose_response_kl']} "
            f"top1_ok={m_a['prose_top1_unchanged_rate']} ctrl_exact={m_a['control_exact_match_rate']} "
            f"weights={m_a['edited_weights']}",
            flush=True,
        )
    eval_s = round(time.perf_counter() - t_eval, 2)

    # Export best low-KL sparse winner
    export_arm_name = (
        "fisher_ultra_low_kl"
        if (arms["fisher_ultra_low_kl"].get("recovered_of_base_refused") or 0.0) >= 0.70
        else "fisher_sparse_late"
    )
    export_patch_src = sel_patches[export_arm_name]
    export_patch = MoIPatch(
        edits=export_patch_src.edits,
        router=None,
        meta={
            "name": f"lfm3-prose-preserving-{export_arm_name}",
            "description": "Always-on sparse surgical over-refusal .moipatch localized with healthy-prose Fisher tracing.",
            "winner_arm": export_arm_name,
            "winner_metrics": arms[export_arm_name],
        },
    )
    audit = None
    if patch_out_path is not None:
        patch_out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = patch_out_path.with_suffix(".moipatch.tmp")
        export_patch.save(tmp_p)
        os.replace(tmp_p, patch_out_path)
        reloaded = MoIPatch.load(patch_out_path)
        audit = sr.audit_moipatch(
            reloaded,
            model,
            tok,
            probe_ids=[r["prompt_ids"] for r in ref_trace_rows[: min(6, len(ref_trace_rows))]],
            control_ids=[r["prompt_ids"] for r in ctrl_a_out[: min(6, len(ctrl_a_out))]],
            patch_path=patch_out_path,
        )

    pareto_frontier = sorted(screened, key=lambda r: (r["healthy_kl"], -r["target_suppression"]))[:15]

    return {
        "status": "ok",
        "architecture": {"n_layers": n_blocks, "n_residual_writers": len(writers)},
        "split": {
            "or_bench_half_a": len(or_a_idx),
            "or_bench_half_b": len(or_b_idx),
            "or_bench_half_b_refused": sum(1 for r in base_or_b if r["refused"]),
            "control_half_a": len(ctrl_a_idx),
            "control_half_b": len(ctrl_b_idx),
        },
        "tracing": {
            "n_refusal_traces": len(ref_traces),
            "n_prose_traces": len(prose_traces),
            "prose_max_gen": int(prose_max_gen),
            "fisher_top8_sites": diag_fisher["sites"][:8],
            "linear_top8_sites": diag_linear["sites"][:8],
        },
        "screening": {
            "n_candidates": len(candidates),
            "selected": sel_meta,
            "low_kl_pareto_top15": pareto_frontier,
        },
        "arms": arms,
        "exported_arm": export_arm_name,
        "patch_out_path": str(patch_out_path) if patch_out_path else None,
        "audit": audit,
        "timing_s": {
            "base": base_s,
            "trace_and_diagnose": trace_s,
            "screen": screen_s,
            "eval_half_b": eval_s,
            "total": round(time.perf_counter() - t_all, 2),
        },
    }


# ------------------------------------------------------------------ CPU Smoke Tri-Hybrid Prose-vs-Refusal Model
class _TinyProseVsRefusalLM(nn.Module):
    """4-layer tri-hybrid toy LM with a shared noisy Layer-1 conv writer and sparse late Layer-3 MLP neurons."""

    def __init__(self, vocab: int = 40, hidden: int = 24, inter: int = 48) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="lfm2",
            hidden_size=hidden,
            num_attention_heads=4,
            eos_token_id=2,
            pad_token_id=0,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList(
            [sr._TinyTriHybridBlock(hidden, inter, use_conv=(i in (0, 1))) for i in range(4)]
        )
        self.model.embedding_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        with torch.no_grad():
            for p in self.parameters():
                p.zero_()
            self.model.embedding_norm.reset_parameters()

            emb = self.model.embed_tokens.weight
            for t in range(vocab):
                if 20 <= t < 38:  # benign OR-Bench refusal trigger
                    emb[t, 3] = 1.5
                    emb[t, 2] = 0.2
                else:  # normal prose controls
                    emb[t, 2] = 1.2

            # Layer 1 conv.out_proj: shared across trigger (dim 3) AND prose (dim 2) -> dirty baseline!
            b1 = self.model.layers[1]
            b1.conv.in_proj.weight[0, 3] = 0.8
            b1.conv.in_proj.weight[1, 2] = 0.8
            b1.conv.out_proj.weight[0, 0] = 1.2
            b1.conv.out_proj.weight[0, 1] = 0.9
            b1.conv.out_proj.weight[1, 1] = 2.0

            # Layer 3 feed_forward.w2:
            # - cols 0..3: refusal-only neurons (driven purely by coord 3 -> write refusal dim 0)
            # - cols 4..7: prose-only neurons (driven purely by coord 2 -> write helpful dims 1 & 0)
            b3 = self.model.layers[3]
            b3.feed_forward.w1.weight[0:4, 3] = 0.6
            b3.feed_forward.w3.weight[0:4, 3] = 0.6
            b3.feed_forward.w2.weight[0, 0:4] = 0.43

            b3.feed_forward.w1.weight[4:8, 2] = 0.6
            b3.feed_forward.w3.weight[4:8, 2] = 0.6
            b3.feed_forward.w2.weight[0, 4:8] = 0.2
            b3.feed_forward.w2.weight[1, 4:8] = 1.0

            # Keep logits well-calibrated (non-saturated P ~ 0.70-0.85) so per-token gradients stay informative:
            # - On refusal trigger: tok 25 wins cleanly (~dim 0)
            # - On prose controls: tok 13 slightly beats tok 12 using dim 0 + dim 1; editing dim 0 during prose flips 13 -> 12!
            self.lm_head.weight[25, 0] = 2.6
            self.lm_head.weight[12, 1] = 2.3
            self.lm_head.weight[13, 1] = 2.0
            self.lm_head.weight[13, 0] = 1.4
            self.lm_head.weight[2, 2] = 0.05

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        **_kw,
    ) -> SimpleNamespace:
        x = self.model.embed_tokens(input_ids)
        hs = [x] if output_hidden_states else None
        for idx, blk in enumerate(self.model.layers):
            x = blk(x)
            if output_hidden_states and idx < len(self.model.layers) - 1:
                hs.append(x)
        x = self.model.embedding_norm(x)
        if output_hidden_states:
            hs.append(x)
        return SimpleNamespace(logits=self.lm_head(x), hidden_states=tuple(hs) if hs is not None else None)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 3, **_kw) -> torch.Tensor:
        cur = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(cur, use_cache=False)
            nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
        return cur


def run_smoke(out_dir: str | Path | None = None) -> dict[str, Any]:
    """Fast CPU self-test showing Fisher + sparse late neurons crush `baseline_always_on` drift."""
    torch.manual_seed(0)
    model = _TinyProseVsRefusalLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = sr._TinyRefusalTokenizer()
    dev = torch.device("cpu")

    # Simulate the legacy dense-conv baseline patch on Layer 1 (`all` cols)
    h_dim = model.config.hidden_size
    u_ref = torch.zeros(h_dim, dtype=torch.float64)
    u_ref[0] = 1.0
    baseline_patch = MoIPatch(
        edits=[Edit(module="layers.1.conv.out_proj", u=u_ref, alpha=1.8, idx=None, base_dtype="float32", shape=(h_dim, h_dim))],
        meta={"name": "smoke-legacy-always-on"},
    )

    or_prompts = [f"trigger:{i} benign prompt with sensitive word" for i in range(16)]
    ctrl_prompts = [f"normal:{i} everyday prose question" for i in range(12)]

    with tempfile.TemporaryDirectory() as td:
        patch_p = (Path(out_dir) if out_dir else Path(td)) / "smoke_prose_sparse.moipatch"
        res = _run_prose_experiment(
            model,
            tok,
            or_prompts,
            ctrl_prompts,
            dev,
            baseline_patch=baseline_patch,
            patch_out_path=patch_p,
            n_traces=4,
            n_prose_traces=4,
            prose_max_gen=4,
            discovery_gen_window=3,
            max_new_tokens=3,
            batch_size=4,
            neuron_budgets=(4, 8),
            candidate_alphas=(1.0, 1.8, 2.5, 4.0),
        )

    assert res["status"] == "ok", res
    assert res["audit"]["fingerprint_ok"] and res["audit"]["restore_exact_ok"]
    b_arm = res["arms"]["baseline_always_on"]
    l_arm = res["arms"]["fisher_sparse_late"]
    u_arm = res["arms"]["fisher_ultra_low_kl"]
    assert b_arm["recovered_of_base_refused"] == 1.0, b_arm
    assert b_arm["healthy_kl"] > 0.01, b_arm
    assert l_arm["recovered_of_base_refused"] == 1.0, l_arm
    assert l_arm["n_edited_modules"] == 1 and l_arm["edited_weights"] < b_arm["edited_weights"], l_arm
    assert l_arm["healthy_kl"] <= b_arm["healthy_kl"] / 5.0 and l_arm["healthy_kl"] < 0.002, (b_arm, l_arm)
    assert l_arm["prose_top1_unchanged_rate"] == 1.0, l_arm
    assert l_arm["control_exact_match_rate"] == 1.0, l_arm
    assert u_arm["recovered_of_base_refused"] == 1.0 and u_arm["control_exact_match_rate"] == 1.0, u_arm
    res["smoke_passed"] = True
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="Run fast CPU verification on `_TinyProseVsRefusalLM`")
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--baseline-patch", default=None, help="Optional existing `.moipatch` to evaluate as baseline_always_on")
    ap.add_argument("--patch-out", default="/content/lfm3_prose_preserving_sparse.moipatch")
    ap.add_argument("--out", default="/content/prose_preserving_always_on_lfm3_l4.json")
    ap.add_argument("--n-or-bench", type=int, default=160)
    ap.add_argument("--n-control", type=int, default=60)
    ap.add_argument("--n-traces", type=int, default=8)
    ap.add_argument("--n-prose-traces", type=int, default=8)
    ap.add_argument("--prose-max-gen", type=int, default=96)
    ap.add_argument("--discovery-gen-window", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.smoke:
        payload = run_smoke()
        print(json.dumps(payload, indent=2))
        return

    model, tok, dev, dtype = sr._load_model_and_tokenizer(args.model, args.dtype)
    baseline_patch = MoIPatch.load(args.baseline_patch) if args.baseline_patch else None
    or_prompts, ctrl_prompts = sr._load_or_bench_and_controls(args.n_or_bench, args.n_control, args.seed)
    payload = _run_prose_experiment(
        model,
        tok,
        or_prompts,
        ctrl_prompts,
        dev,
        baseline_patch=baseline_patch,
        patch_out_path=Path(args.patch_out),
        n_traces=args.n_traces,
        n_prose_traces=args.n_prose_traces,
        prose_max_gen=args.prose_max_gen,
        discovery_gen_window=args.discovery_gen_window,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    payload["model"] = args.model
    payload["dtype"] = str(dtype)
    payload["baseline_patch_path"] = str(args.baseline_patch) if args.baseline_patch else None
    out_p = sr._atomic_write_json(args.out, payload)
    print(f"Saved prose-preserving report -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
