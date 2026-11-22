"""Decisive Test of Minimal Few-Shot Continual Skill Compilation (K=6 traces/skill -> 1.7 MB cartridges).

Tests whether `<think> / teacher trace -> WSI micro-cartridge` genuinely enables what standard LoRA cannot:
  - Compiles each newly arriving recurring reasoning skill in ~15-20s from ONLY K = 6 traces into an independent
    ~1.7 MB WSI `r=8` cartridge (`4` causal layers) gated by a few-shot Layer-12 Ridge router (`6` examples/skill).
  - Accumulates `5` distinct recurring skills sequentially without storing past data, replaying old examples,
    degrading Skills 1..4 (zero sequential catastrophic forgetting), or harming guardrails / MMLU / prose.

9 paired arms on `60` blind hard test items (`5` skills x `12` = `6` renumbered + `6` disjoint rephrased),
`24` guardrail math tests, `200` held-out `MMLU` items, and `16` held-out prose controls:
  1. `base_08b`                    : Qwen3.5-0.8B (`enable_thinking=False`)
  2. `teacher_4b`                  : Qwen3.5-4B (`enable_thinking=False`)
  3. `oracle_1shot_08b`            : 0.8B with 1 solved train trace in context (prompt caching baseline)
  4. `fewshot_wsi_r8_routed`       : 5 independent `1.7 MB` (`r=8`, 4 WSI layers) cartridges + L12 router
  5. `fewshot_random_r8_routed`    : 5 independent `1.7 MB` (`r=8`, 4 stratified-random layers) + L12 router
  6. `fewshot_wsi_r64_routed`      : 5 independent `27.4 MB` (`r=64`, 8 WSI layers) + L12 router (overfitting check at K=6)
  7. `continual_lora_r8_no_replay` : 1 shared `r=8` LoRA trained sequentially Skill 1 -> ... -> Skill 5, NO replay
  8. `continual_lora_r64_no_replay`: 1 shared `r=64` LoRA trained sequentially Skill 1 -> ... -> Skill 5, NO replay
  9. `joint_offline_lora_r64`      : 1 shared `r=64` LoRA trained offline jointly on all `30` traces (offline upper bound)

CPU smoke test::
    python scripts/experiment_pocket_moe_fewshot.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_moe_fewshot.py --teacher /content/teacher_4b --student /content/student_08b --out /content/pocket_moe_fewshot_l4.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import statistics
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
    _LoRALinear,
    _bake_full_checkpoint,
    _block_prefixes,
    _enable_emb_grad_hook,
    _extract_causal_layers,
    _grade_rows,
    _hf_generate_subset,
    _train_lora_expert,
    _vllm_generate,
    build_ceiling_dataset,
    distill_step_loss,
    eval_healthy_drift,
    fit_ridge_router,
    predict_ridge_router,
)
from experiment_pocket_moe_controls import (  # noqa: E402
    _guardrail_exact_match_rate,
    _paired_bootstrap,
    pick_stratified_random_layers,
)
from experiment_pocket_moe_math500 import (  # noqa: E402
    PROSE_HELDOUT,
    PROSE_ROUTER_TRAIN,
    _compose_routed_mmlu,
    _score_mmlu_fast,
    _summarize_mmlu,
    _temp_apply_weights,
    format_mmlu_prompt_ids,
    load_mmlu_split,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _PlantedReasoningLM,
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
    build_reasoning_dataset,
    compute_invariance_metrics,
)
from experiment_reasoning_distill_v2 import (  # noqa: E402
    _format_messages,
    _select_backend,
    format_question,
    grade_response,
)
from wsi import CompactTrace, capture_causal_trace, diagnose_causal_traces  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402

SKILLS_5: tuple[str, ...] = (
    "rate_work_combined",
    "compound_percent_chain",
    "weighted_mixture_transfer",
    "piecewise_tiered_tax",
    "meeting_catchup_stops",
)
OLD_SKILLS: tuple[str, ...] = SKILLS_5[:4]
LAST_SKILL: str = SKILLS_5[4]
ROUTER_CLASSES: tuple[str, ...] = (*SKILLS_5, "passthrough")

_TAX_TRAIN_SCENARIOS: tuple[tuple[str, str, str, str], ...] = (
    ("electric_grid", "municipal utility", "kilowatt-hours", "dollars"),
    ("water_meter", "city water board", "cubic meters", "dollars"),
    ("cloud_egress", "cloud provider", "gigabytes", "credits"),
    ("freight_parcel", "freight carrier", "kilograms", "dollars"),
    ("parking_garage", "airport garage", "hours", "dollars"),
    ("compute_tokens", "inference API", "million tokens", "dollars"),
)
_TAX_TEST_SCENARIOS: tuple[tuple[str, str, str, str], ...] = (
    ("natural_gas", "heating cooperative", "therms", "euros"),
    ("solar_storage", "battery co-op", "amp-hours", "credits"),
    ("data_archive", "glacier vault", "terabytes", "dollars"),
    ("sms_gateway", "messaging gateway", "message batches", "cents"),
    ("rail_tolls", "canal authority", "tonnes", "euros"),
    ("printer_quota", "campus print hub", "page blocks", "credits"),
)

_CATCHUP_TRAIN_SCENARIOS: tuple[tuple[str, str, str, str], ...] = (
    ("cargo_trains", "Freight Train Slow", "Express Train Fast", "kilometers"),
    ("ocean_vessels", "Supply Barge", "Patrol Cutter", "nautical miles"),
    ("delivery_vans", "Local Van", "Express Van", "miles"),
    ("river_boats", "Tour Boat", "Rescue Launch", "kilometers"),
    ("cycling_pack", "Lead Peloton", "Chase Group", "kilometers"),
    ("trail_hikers", "Supply Mule", "Scout Rider", "miles"),
)
_CATCHUP_TEST_SCENARIOS: tuple[tuple[str, str, str, str], ...] = (
    ("mail_couriers", "Courier Cart", "Dispatch Bike", "kilometers"),
    ("arctic_sleds", "Heavy Sled", "Light Sled", "miles"),
    ("desert_convoys", "Tanker Convoy", "Escort Jeep", "kilometers"),
    ("canal_barges", "Coal Barge", "Mail Skiff", "kilometers"),
    ("mountain_trucks", "Timber Truck", "Repair Truck", "miles"),
    ("lake_ferries", "Ferry Alpha", "Speedboat Beta", "nautical miles"),
)


def _solve_tiered_tax(f0: int, c1: int, r1: int, c2: int, r2: int, r3: int, x3: int) -> tuple[int, int, str]:
    q_tot = c1 + c2 + x3
    t1 = c1 * r1
    t2 = c2 * r2
    t3 = x3 * r3
    tot_bill = f0 + t1 + t2 + t3
    var_bill = tot_bill - f0
    rem1 = var_bill - t1
    rem2 = rem1 - t2
    resp = (
        f"Step 1: Subtract the fixed base fee from the total bill: {tot_bill} - {f0} = {var_bill}.\n"
        f"Step 2: Subtract the Tier 1 charge ({c1} * {r1} = {t1}): {var_bill} - {t1} = {rem1}.\n"
        f"Step 3: Subtract the Tier 2 charge ({c2} * {r2} = {t2}): {rem1} - {t2} = {rem2}.\n"
        f"Step 4: Divide the remaining charge by the Tier 3 rate {r3}: {rem2} / {r3} = {x3}.\n"
        f"Step 5: Add all units together: {c1} + {c2} + {x3} = {q_tot}.\n"
        f"The total consumption was \\boxed{{{q_tot}}}."
    )
    return tot_bill, q_tot, resp


def _solve_catchup_stops(v1: int, t_head: int, t_stop: int, v2: int, k_catch: int) -> tuple[int, str]:
    assert t_head > t_stop and v2 > v1 and k_catch > t_stop + 1
    t_net = t_head - t_stop
    assert v1 * (t_net + k_catch) == v2 * k_catch
    d_head = v1 * t_net
    dv = v2 - v1
    resp = (
        f"Step 1: Account for the stop during the chase: while the faster unit moves for t hours, "
        f"the slower unit moves for ({t_head} - {t_stop}) + t = {t_net} + t hours.\n"
        f"Step 2: Multiply the slower unit's effective head start by its speed: {v1} * {t_net} = {d_head}.\n"
        f"Step 3: Compute the catch-up rate: {v2} - {v1} = {dv}.\n"
        f"Step 4: Divide the effective lead by the catch-up rate: {d_head} / {dv} = {k_catch}.\n"
        f"The second unit catches up in \\boxed{{{k_catch}}} hours."
    )
    return k_catch, resp


def build_fewshot_dataset(*, k_shots: int = 6) -> dict[str, Any]:
    """Build K=6 train items/skill (`30` total), `60` blind hard test items (`5` skills x `12`),
    plus `24` guardrail test + `32` guardrail router train + `28` disjoint prose items."""
    assert {s[0] for s in _TAX_TRAIN_SCENARIOS}.isdisjoint({s[0] for s in _TAX_TEST_SCENARIOS})
    assert {s[0] for s in _CATCHUP_TRAIN_SCENARIOS}.isdisjoint({s[0] for s in _CATCHUP_TEST_SCENARIOS})

    ceil_ds = build_ceiling_dataset()
    hard_train: list[dict[str, Any]] = []
    hard_test: list[dict[str, Any]] = list(ceil_ds["hard_test"])

    # Skills 1..3: take strictly first K items per skill (idx 0..K-1 cover K distinct train scenarios)
    for fam in ("rate_work_combined", "compound_percent_chain", "weighted_mixture_transfer"):
        fam_tr = [it for it in ceil_ds["hard_train"] if it["family"] == fam]
        assert len(fam_tr) >= k_shots
        hard_train.extend(fam_tr[:k_shots])

    # Skill 4: piecewise_tiered_tax (18 distinct tuples: 6 train + 6 renum + 6 rephrased)
    tax_tuples: list[tuple[int, int, int, int, int, int, int]] = []
    seen_tax: set[tuple[int, int, int, int, int, int, int]] = set()
    cursor = 0
    while len(tax_tuples) < k_shots + 12:
        f0 = 15 + (cursor * 5) % 45
        c1 = 10 + (cursor % 6) * 5
        r1 = 2 + (cursor % 4)
        c2 = 15 + ((cursor * 2) % 5) * 5
        r2 = r1 + 2 + (cursor % 3)
        r3 = r2 + 3 + ((cursor // 3) % 4)
        x3 = 6 + ((cursor * 3 + 2) % 14)
        tup = (f0, c1, r1, c2, r2, r3, x3)
        if tup not in seen_tax:
            seen_tax.add(tup)
            tax_tuples.append(tup)
        cursor += 1

    def _fmt_tax_q(sc: tuple[str, str, str, str], f0: int, c1: int, r1: int, c2: int, r2: int, r3: int, tot_bill: int) -> str:
        _, org, unit, curr = sc
        return (
            f"A {org} charges a fixed service fee of {f0} {curr}, plus {r1} {curr} per {unit[:-1] if unit.endswith('s') else unit} "
            f"for the first {c1} {unit}, {r2} {curr} per unit for the next {c2} {unit}, and {r3} {curr} per unit "
            f"for every additional unit beyond {c1 + c2} {unit}. If a customer's total bill was {tot_bill} {curr}, "
            f"how many total {unit} were billed?"
        )

    for idx in range(k_shots):
        sc = _TAX_TRAIN_SCENARIOS[idx % len(_TAX_TRAIN_SCENARIOS)]
        tot_bill, ans, resp = _solve_tiered_tax(*tax_tuples[idx])
        hard_train.append({
            "id": f"train_tax_{idx}",
            "family": "piecewise_tiered_tax",
            "variant": "train",
            "scenario": sc[0],
            "question": _fmt_tax_q(sc, *tax_tuples[idx][:6], tot_bill),
            "fallback_response": resp,
            "answer": ans,
        })
    for idx in range(6):
        sc = _TAX_TRAIN_SCENARIOS[idx % len(_TAX_TRAIN_SCENARIOS)]
        tup = tax_tuples[k_shots + idx]
        tot_bill, ans, resp = _solve_tiered_tax(*tup)
        hard_test.append({
            "id": f"test_tax_renum_{idx}",
            "family": "piecewise_tiered_tax",
            "variant": "renumbered",
            "scenario": sc[0],
            "question": _fmt_tax_q(sc, *tup[:6], tot_bill),
            "fallback_response": resp,
            "answer": ans,
        })
    for idx in range(6):
        sc = _TAX_TEST_SCENARIOS[idx]
        tup = tax_tuples[k_shots + 6 + idx]
        tot_bill, ans, resp = _solve_tiered_tax(*tup)
        hard_test.append({
            "id": f"test_tax_rephrase_{idx}",
            "family": "piecewise_tiered_tax",
            "variant": "rephrased_renumbered",
            "scenario": sc[0],
            "question": _fmt_tax_q(sc, *tup[:6], tot_bill),
            "fallback_response": resp,
            "answer": ans,
        })

    # Skill 5: meeting_catchup_stops (18 distinct tuples: 6 train + 6 renum + 6 rephrased)
    catch_tuples: list[tuple[int, int, int, int, int]] = []
    seen_c: set[tuple[int, int, int, int, int]] = set()
    c_cursor = 0
    while len(catch_tuples) < k_shots + 12:
        v1 = 24 + (c_cursor * 6) % 36
        t_stop = 1 + (c_cursor % 2)
        t_net = 3 + (c_cursor % 3)
        t_head = t_stop + t_net
        # Pick dv dividing v1 * t_net with k_catch > t_stop + 1
        d_lead = v1 * t_net
        cands = [dv for dv in range(8, 35) if d_lead % dv == 0 and (d_lead // dv) > t_stop + 2 and (d_lead // dv) <= 18]
        if cands:
            dv = cands[(c_cursor // 2) % len(cands)]
            v2 = v1 + dv
            k_catch = d_lead // dv
            tup = (v1, t_head, t_stop, v2, k_catch)
            if tup not in seen_c:
                seen_c.add(tup)
                catch_tuples.append(tup)
        c_cursor += 1
        if c_cursor > 500:
            raise RuntimeError("Failed to generate enough catchup tuples")

    def _fmt_catch_q(sc: tuple[str, str, str, str], v1: int, t_head: int, t_stop: int, v2: int) -> str:
        _, a, b, unit = sc
        return (
            f"{a} departs from a station traveling at {v1} {unit} per hour. "
            f"After {t_head} hours, {b} leaves from the same station along the same route at {v2} {unit} per hour to catch up. "
            f"During the trip before {b} catches up, {a} makes stationary stops totaling {t_stop} hours (moving 0 {unit} while stopped) "
            f"and travels at {v1} {unit} per hour all other hours. How many hours after {b} departs will {b} overtake {a}?"
        )

    for idx in range(k_shots):
        sc = _CATCHUP_TRAIN_SCENARIOS[idx % len(_CATCHUP_TRAIN_SCENARIOS)]
        tup = catch_tuples[idx]
        ans, resp = _solve_catchup_stops(*tup)
        hard_train.append({
            "id": f"train_catch_{idx}",
            "family": "meeting_catchup_stops",
            "variant": "train",
            "scenario": sc[0],
            "question": _fmt_catch_q(sc, *tup[:4]),
            "fallback_response": resp,
            "answer": ans,
        })
    for idx in range(6):
        sc = _CATCHUP_TRAIN_SCENARIOS[idx % len(_CATCHUP_TRAIN_SCENARIOS)]
        tup = catch_tuples[k_shots + idx]
        ans, resp = _solve_catchup_stops(*tup)
        hard_test.append({
            "id": f"test_catch_renum_{idx}",
            "family": "meeting_catchup_stops",
            "variant": "renumbered",
            "scenario": sc[0],
            "question": _fmt_catch_q(sc, *tup[:4]),
            "fallback_response": resp,
            "answer": ans,
        })
    for idx in range(6):
        sc = _CATCHUP_TEST_SCENARIOS[idx]
        tup = catch_tuples[k_shots + 6 + idx]
        ans, resp = _solve_catchup_stops(*tup)
        hard_test.append({
            "id": f"test_catch_rephrase_{idx}",
            "family": "meeting_catchup_stops",
            "variant": "rephrased_renumbered",
            "scenario": sc[0],
            "question": _fmt_catch_q(sc, *tup[:4]),
            "fallback_response": resp,
            "answer": ans,
        })

    return {
        "skills": list(SKILLS_5),
        "k_shots": int(k_shots),
        "hard_train": hard_train,
        "hard_test": hard_test,
        "guardrail_test": list(ceil_ds["guardrail_test"]),
        "guardrail_router_train": list(ceil_ds["guardrail_router_train"]),
        "prose_heldout": list(PROSE_HELDOUT[:16]),
        "prose_router_train": list(PROSE_ROUTER_TRAIN[:12]),
    }


def _train_continual_lora_no_replay(
    model: nn.Module,
    layer_idxs: list[int],
    samples_by_skill: dict[str, list[dict[str, Any]]],
    skills_order: tuple[str, ...] = SKILLS_5,
    *,
    rank: int,
    alpha_scale: float,
    epochs_per_skill: int,
    lr_val: float,
    grad_accum: int,
    healthy_ids: list[torch.Tensor] | None = None,
    base_healthy_logp: list[torch.Tensor | None] | None = None,
    base_healthy_nll: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Sequentially train ONE shared LoRA over arriving skills (Skill 1 -> ... -> Skill 5) with NO replay."""
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

    wrapped: list[tuple[nn.Module, str, nn.Linear, str, _LoRALinear]] = []
    for li in layer_idxs:
        blk = blocks[li]
        blk_pref = prefixes[li]
        for sub_nm, lin in [(nm, m) for nm, m in blk.named_modules() if isinstance(m, nn.Linear)]:
            parts = sub_nm.split(".")
            parent = blk
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            wrapper = _LoRALinear(lin, rank=rank, alpha_scale=alpha_scale)
            setattr(parent, attr, wrapper)
            wrapped.append((parent, attr, lin, f"{blk_pref}.{sub_nm}.weight", wrapper))

    lora_params = [p for *_rest, w in wrapped for p in (w.lora_A, w.lora_B)]
    adapter_mb = round(sum(w.adapter_bytes() for *_rest, w in wrapped) / (1024**2), 4)
    emb_handles = _enable_emb_grad_hook(model)
    per_skill_losses: dict[str, list[float]] = {}
    t_compile0 = time.perf_counter()

    try:
        model.eval()
        for sk_idx, sk in enumerate(skills_order):
            sk_samples = samples_by_skill[sk]
            total_steps = max(1, (len(sk_samples) * epochs_per_skill + grad_accum - 1) // grad_accum)
            opt = torch.optim.AdamW(lora_params, lr=float(lr_val), weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=float(lr_val) * 0.1)
            rng = torch.Generator().manual_seed(int(seed) + sk_idx * 31)
            sk_hist: list[float] = []

            for _ep in range(epochs_per_skill):
                perm = torch.randperm(len(sk_samples), generator=rng).tolist()
                ep_losses: list[float] = []
                for start in range(0, len(perm), grad_accum):
                    chunk = perm[start : start + grad_accum]
                    opt.zero_grad(set_to_none=True)
                    batch_loss = 0.0
                    for idx_s in chunk:
                        sp = sk_samples[idx_s]
                        ids_dev = torch.tensor(sp["ids"], dtype=torch.long, device=dev).unsqueeze(0)
                        lo, hi = int(sp["lo"]), int(sp["hi"])
                        with torch.enable_grad():
                            s_logits = _tail_logits(model, ids_dev, lo)[: hi - lo]
                            t_ids = ids_dev[0, lo + 1 : hi + 1]
                            tk_ids = sp["topk_ids"].to(dev) if sp.get("topk_ids") is not None else None
                            tk_log = sp["topk_logits"].to(dev) if sp.get("topk_logits") is not None else None
                            loss = distill_step_loss(s_logits, t_ids, tk_ids, tk_log)
                            (loss / float(len(chunk))).backward()
                            batch_loss += float(loss.detach().item()) / float(len(chunk))
                        del ids_dev, s_logits, t_ids, loss
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    opt.step()
                    sched.step()
                    ep_losses.append(batch_loss)
                sk_hist.append(round(sum(ep_losses) / max(1, len(ep_losses)), 5))
            per_skill_losses[sk] = sk_hist

        compile_time_s = round(time.perf_counter() - t_compile0, 3)
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
        "layer_idxs": list(layer_idxs),
        "per_skill_losses": per_skill_losses,
        "compile_time_s": compile_time_s,
        **drift,
    }


def _summarize_fewshot_arm(
    rows: list[dict[str, Any]],
    *,
    arm_raw: list[dict[str, Any]] | None = None,
    base_raw: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hard = [r for r in rows if r["family"] in SKILLS_5]
    old_h = [r for r in hard if r["family"] in OLD_SKILLS]
    last_h = [r for r in hard if r["family"] == LAST_SKILL]
    guard = [r for r in rows if r["family"] in GUARDRAIL_FAMILIES]
    renum = [r for r in hard if r["variant"] == "renumbered"]
    rephr = [r for r in hard if r["variant"] == "rephrased_renumbered"]
    h_toks = [r["total_tokens"] for r in hard]

    per_skill = {
        sk: round(sum(1 for r in hard if r["family"] == sk and r["strict_correct"]) / max(1, sum(1 for r in hard if r["family"] == sk)), 4)
        for sk in SKILLS_5
    }
    per_guard = {
        gf: round(sum(1 for r in guard if r["family"] == gf and r["strict_correct"]) / max(1, sum(1 for r in guard if r["family"] == gf)), 4)
        for gf in GUARDRAIL_FAMILIES
    }
    guard_out: dict[str, Any] = {
        "strict_accuracy": round(sum(1 for r in guard if r["strict_correct"]) / max(1, len(guard)), 4),
        "n_correct": sum(1 for r in guard if r["strict_correct"]),
        "n_total": len(guard),
        "per_family_strict": per_guard,
    }
    if arm_raw is not None and base_raw is not None:
        guard_out["guardrail_exact_match_vs_base"] = _guardrail_exact_match_rate(arm_raw, base_raw, len(hard))

    return {
        "hard_60": {
            "strict_accuracy": round(sum(1 for r in hard if r["strict_correct"]) / max(1, len(hard)), 4),
            "n_correct": sum(1 for r in hard if r["strict_correct"]),
            "n_total": len(hard),
            "renumbered_30": round(sum(1 for r in renum if r["strict_correct"]) / max(1, len(renum)), 4),
            "rephrased_30": round(sum(1 for r in rephr if r["strict_correct"]) / max(1, len(rephr)), 4),
            "old_skills_1_to_4_acc": round(sum(1 for r in old_h if r["strict_correct"]) / max(1, len(old_h)), 4),
            "last_skill_5_acc": round(sum(1 for r in last_h if r["strict_correct"]) / max(1, len(last_h)), 4),
            "truncated_rate": round(sum(1 for r in hard if r["truncated"]) / max(1, len(hard)), 4),
            "median_tokens": round(float(statistics.median(h_toks)), 1) if h_toks else 0.0,
            "per_skill": per_skill,
        },
        "guardrail_24": guard_out,
        "all_84": {
            "strict_accuracy": round(sum(1 for r in rows if r["strict_correct"]) / max(1, len(rows)), 4),
            "n_correct": sum(1 for r in rows if r["strict_correct"]),
            "n_total": len(rows),
        },
    }


def run_smoke() -> dict[str, Any]:
    """Deterministic CPU check of 5-skill dataset (K=6), new skill grading, continual no-replay LoRA, and summaries."""
    ds = build_fewshot_dataset(k_shots=6)
    assert len(ds["hard_train"]) == 30, len(ds["hard_train"])
    assert len(ds["hard_test"]) == 60, len(ds["hard_test"])
    assert len(ds["guardrail_test"]) == 24
    assert len(ds["prose_heldout"]) == 16 and len(ds["prose_router_train"]) == 12

    for sk in SKILLS_5:
        tr_sc = {it["scenario"] for it in ds["hard_train"] if it["family"] == sk}
        assert len(tr_sc) == 6, (sk, tr_sc)
        te_rep_sc = {it["scenario"] for it in ds["hard_test"] if it["family"] == sk and it["variant"] == "rephrased_renumbered"}
        assert tr_sc.isdisjoint(te_rep_sc), (sk, tr_sc & te_rep_sc)
    for it in ds["hard_train"] + ds["hard_test"]:
        g = grade_response([1, 2, 3], it["fallback_response"], int(it["answer"]), thinking_enabled=False, finish_reason="stop")
        assert g["strict_correct"] and g["has_boxed"], (it["id"], g)

    # Continual no-replay training + restoration on 4-layer _PlantedReasoningLM
    torch.manual_seed(13)
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    seqs_a = [[1, 2, 4, 10, 20, 20, 20, 20], [2, 1, 4, 11, 20, 20, 20, 20]]
    seqs_b = [[1, 3, 5, 14, 21, 21, 21, 21], [3, 1, 5, 15, 21, 21, 21, 21]]

    def _to_samples(seqs: list[list[int]]) -> list[dict[str, Any]]:
        out = []
        for s in seqs:
            ids_t = torch.tensor(s, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                t_log = model(ids_t).logits[0, 3:7].float()
            vals, tk_ids = t_log.topk(8, dim=-1)
            out.append({
                "ids": s,
                "lo": 3,
                "hi": 7,
                "topk_ids": tk_ids.to(torch.int32).cpu(),
                "topk_logits": vals.to(torch.float16).cpu(),
            })
        return out

    samples_map = {sk: _to_samples(seqs_a if i % 2 == 0 else seqs_b) for i, sk in enumerate(SKILLS_5)}
    h_ids = [torch.tensor([1, 2, 3, 1], dtype=torch.long).unsqueeze(0)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    cont = _train_continual_lora_no_replay(
        model, [0, 1], samples_map, rank=2, alpha_scale=4.0, epochs_per_skill=2, lr_val=2e-2, grad_accum=2,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll, seed=42,
    )
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())
    for sk in SKILLS_5:
        assert cont["per_skill_losses"][sk][-1] < cont["per_skill_losses"][sk][0]

    # Summarize synthetic arm rows to check old_skills_1_to_4_acc vs last_skill_5_acc
    fake_rows = [
        {"id": it["id"], "family": it["family"], "variant": it["variant"], "strict_correct": (it["family"] == LAST_SKILL), "truncated": False, "total_tokens": 200}
        for it in ds["hard_test"]
    ] + [
        {"id": it["id"], "family": it["family"], "variant": "guardrail", "strict_correct": True, "truncated": False, "total_tokens": 120}
        for it in ds["guardrail_test"]
    ]
    sm = _summarize_fewshot_arm(fake_rows)
    assert sm["hard_60"]["old_skills_1_to_4_acc"] == 0.0 and sm["hard_60"]["last_skill_5_acc"] == 1.0

    return {
        "mode": "smoke",
        "passed": True,
        "dataset_counts": {k: len(v) for k, v in ds.items() if isinstance(v, list)},
        "continual_per_skill_losses": cont["per_skill_losses"],
        "summary_split_check": {
            "old_1_to_4": sm["hard_60"]["old_skills_1_to_4_acc"],
            "last_5": sm["hard_60"]["last_skill_5_acc"],
        },
    }


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

    # ---- Phase 1: Teacher 4B generation (84 test + 30 K=6 train) & Top-32 logit extraction ----
    print(f"[fewshot] Phase 1: Teacher ({args.teacher}) generation & Top-32 logits (K={args.k_shots}/skill, cap={args.max_teacher_train_tokens})...", flush=True)
    t_p1 = time.perf_counter()
    test_pids = [_format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False) for it in all_test]
    train_pids = [_format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False) for it in hard_train]
    test_seeds = [int(args.seed) + i for i in range(len(all_test))]
    train_seeds = [int(args.seed) + 1000 + i for i in range(len(hard_train))]

    if backend == "vllm":
        t_outs = _vllm_generate(args.teacher, tok_t, test_pids + train_pids, test_seeds + train_seeds, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
        t_test_raw = t_outs[: len(all_test)]
        t_train_raw = t_outs[len(all_test) :]
    else:
        try:
            teacher_m = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_m = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
        t_test_raw = _hf_generate_subset(teacher_m, tok_t, test_pids, test_seeds, max_new_tokens=args.max_new_tokens)
        t_train_raw = _hf_generate_subset(teacher_m, tok_t, train_pids, train_seeds, max_new_tokens=args.max_new_tokens)
        del teacher_m
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    teacher_test_rows = _grade_rows(all_test, t_test_raw, tok_s, eos_set, think_end_id)
    teacher_summary = _summarize_fewshot_arm(teacher_test_rows)
    print(f"[fewshot] teacher_4b: hard={teacher_summary['hard_60']} | guard={teacher_summary['guardrail_24']}", flush=True)

    kept_4b = {sk: 0 for sk in SKILLS_5}
    chosen_responses: list[str] = []
    for it, raw in zip(hard_train, t_train_raw):
        body = lr.strip_eos(raw["token_ids"], eos_set)
        txt = tok_s.decode(body, skip_special_tokens=True).strip()
        g = grade_response(raw["token_ids"], txt, int(it["answer"]), thinking_enabled=False, finish_reason=raw["finish_reason"], eos_set=eos_set, tok=tok_s)
        if g["strict_correct"] and g["has_boxed"] and g["total_tokens"] <= int(args.max_teacher_train_tokens):
            chosen_responses.append(txt)
            kept_4b[it["family"]] += 1
        else:
            chosen_responses.append(it["fallback_response"])
    teacher_kept_frac = {sk: round(kept_4b[sk] / max(1, args.k_shots), 4) for sk in SKILLS_5}
    print(f"[fewshot] teacher_4b kept frac: {teacher_kept_frac}", flush=True)

    try:
        teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    train_samples_by_skill: dict[str, list[dict[str, Any]]] = {sk: [] for sk in SKILLS_5}
    with torch.no_grad():
        for it, p_ids, resp_txt in zip(hard_train, train_pids, chosen_responses):
            full_ids = _format_messages(
                tok_s,
                [{"role": "user", "content": format_question(it["question"])}, {"role": "assistant", "content": resp_txt}],
                add_generation_prompt=False,
                enable_thinking=False,
            )
            lo = max(0, len(p_ids) - 1)
            hi = len(full_ids) - 1
            ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
            t_logits = _tail_logits(teacher_hf, ids_dev, lo)[: hi - lo].float()
            vals, idx = t_logits.topk(32, dim=-1)
            train_samples_by_skill[it["family"]].append({
                "id": it["id"],
                "family": it["family"],
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
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student 0.8B, WSI Traces, Few-Shot Router & Base MMLU/Prose ----
    print(f"[fewshot] Phase 2: Loading Student ({args.student}), WSI causal traces, K=6 router & base MMLU...", flush=True)
    t_p2 = time.perf_counter()
    try:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in student_m.parameters():
        p.requires_grad_(False)

    _, blocks = find_blocks(student_m)
    n_layers = len(blocks)
    causal_traces_by_skill: dict[str, list[CompactTrace]] = {sk: [] for sk in SKILLS_5}
    for sk in SKILLS_5:
        for sp in train_samples_by_skill[sk][:4]:
            ct = capture_causal_trace(
                student_m,
                sp["ids"],
                target_span=(sp["lo"] + 1, sp["hi"] + 1),
                healthy_span=(0, max(1, sp["lo"])),
                grad_mode="compressed",
                grad_checkpointing=True,
                center="healthy",
                healthy_cost="fisher",
                top_k_attn=0,
                capture_cache=False,
            )
            causal_traces_by_skill[sk].append(ct)

    invariance = compute_invariance_metrics(causal_traces_by_skill, top_k_sites=4, top_n_neurons=32)
    causal_layers: dict[str, list[int]] = {}
    random_layers_r8: dict[str, list[int]] = {}
    for s_idx, sk in enumerate(SKILLS_5):
        diag_s = diagnose_causal_traces(causal_traces_by_skill[sk], split_ab=True, top_k_sites=16, center="healthy", healthy_cost="fisher", model_or_weights=student_m)
        causal_layers[sk] = _extract_causal_layers(diag_s, n_layers, max_layers=8)
        random_layers_r8[sk] = pick_stratified_random_layers(student_m, causal_layers[sk][:4], seed=args.seed + 700 + s_idx)

    all_traces = [ct for sk in SKILLS_5 for ct in causal_traces_by_skill[sk]]
    diag_glob = diagnose_causal_traces(all_traces, split_ab=True, top_k_sites=16, center="healthy", healthy_cost="fisher", model_or_weights=student_m)
    global_causal_layers = _extract_causal_layers(diag_glob, n_layers, max_layers=8)
    print(f"[fewshot] Causal layers={causal_layers} | global={global_causal_layers}", flush=True)

    # Few-shot Layer-12 6-class router using only K=6 prompts per skill
    router_layer = n_layers // 2
    guard_r_pids = [_format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False) for it in ds["guardrail_router_train"]]
    mmlu_r_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_router_train]
    mmlu_test_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_test]
    prose_r_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in ds["prose_router_train"]]
    prose_held_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in ds["prose_heldout"]]

    r_train_pids = train_pids + guard_r_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = [it["family"] for it in hard_train] + ["passthrough"] * (len(guard_r_pids) + len(mmlu_r_pids) + len(prose_r_pids))

    X_r_train = _extract_prompt_hidden_vecs(student_m, r_train_pids, router_layer)
    W_r, b_r = fit_ridge_router(X_r_train, r_train_labels, ROUTER_CLASSES, lam=1.0)

    routed_test_classes = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, test_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_mmlu_classes = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, mmlu_test_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_prose_classes = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, prose_held_pids, router_layer), W_r, b_r, ROUTER_CLASSES)

    hard_r_preds = routed_test_classes[:n_hard]
    guard_r_preds = routed_test_classes[n_hard:]
    router_report = {
        "router_layer": router_layer,
        "k_shots_per_skill": int(args.k_shots),
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
    print(f"[fewshot] Few-shot K={args.k_shots} Router report: {router_report}", flush=True)

    letter_ids = ec._resolve_mmlu_letter_ids(tok_s)
    base_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
    base_mmlu_summary = _summarize_mmlu(mmlu_test, base_mmlu_raw, seed=args.seed)

    prose_full_ids = _to_seq_tensors(
        [
            _format_messages(
                tok_s,
                [{"role": "user", "content": p}, {"role": "assistant", "content": "Here is a concise explanation."}],
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for p in ds["prose_heldout"]
        ],
        torch.device("cpu"),
    )
    base_prose_logp: list[torch.Tensor | None] = []
    nll_b_sum, n_b_h = 0.0, 0
    with torch.no_grad():
        for ids in prose_full_ids:
            logits = _tail_logits(student_m, ids.to(dev), 0)[:-1].float()
            lp = F.log_softmax(logits, dim=-1)
            nll_b_sum += float(F.nll_loss(lp, ids[0, 1:].to(dev), reduction="sum"))
            n_b_h += int(ids.shape[1] - 1)
            base_prose_logp.append(lp.to(dtype=torch.bfloat16, device="cpu"))
            del logits, lp
    base_prose_nll = nll_b_sum / max(1, n_b_h)
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Compile Cartridges (~15-20s each) & Continual / Joint LoRAs + MMLU Retention ----
    print(f"[fewshot] Phase 3: Compiling K={args.k_shots} cartridges & continual baselines (epochs={args.epochs_per_skill}, grad_accum={args.grad_accum})...", flush=True)
    t_p3 = time.perf_counter()

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        expert_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {
            "fewshot_wsi_r8": {},
            "fewshot_random_r8": {},
            "fewshot_wsi_r64": {},
            "continual_lora_r8_no_replay": {},
            "continual_lora_r64_no_replay": {},
            "joint_offline_lora_r64": {},
        }
        domain_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {
            "fewshot_wsi_r8": {},
            "fewshot_random_r8": {},
            "fewshot_wsi_r64": {},
        }

        routed_specs = [
            ("fewshot_wsi_r8", causal_layers, 4, 8, 16.0, 5e-4, 0, False),
            ("fewshot_random_r8", random_layers_r8, 4, 8, 16.0, 5e-4, 100, True),
            ("fewshot_wsi_r64", causal_layers, 8, 64, 64.0, 3e-4, 200, False),
        ]

        for s_idx, sk in enumerate(SKILLS_5):
            samples_sk = train_samples_by_skill[sk]
            for tier_nm, lmap, k_lyrs, rk, a_sc, lr_v, sd_off, is_rand in routed_specs:
                lyrs = list(lmap[sk][:k_lyrs])
                t_c0 = time.perf_counter()
                res_e = _train_lora_expert(
                    student_m,
                    lyrs,
                    samples_sk,
                    rank=rk,
                    alpha_scale=a_sc,
                    epochs=args.epochs_per_skill,
                    lr_val=lr_v,
                    grad_accum=args.grad_accum,
                    healthy_ids=prose_full_ids,
                    base_healthy_logp=base_prose_logp,
                    base_healthy_nll=base_prose_nll,
                    seed=args.seed + sd_off + s_idx,
                )
                compile_s = round(time.perf_counter() - t_c0, 2)
                res_e["compile_time_s"] = compile_s
                upd_w = res_e.pop("weights")
                with _temp_apply_weights(student_m, upd_w):
                    sc_sk = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
                domain_mmlu_scores[tier_nm][sk] = sc_sk
                if is_rand:
                    res_e["overlap_with_wsi"] = len(set(lyrs) & set(causal_layers[sk][:k_lyrs]))
                p_exp = exp_dir / f"{tier_nm}_{sk}.pt"
                torch.save(upd_w, p_exp)
                del upd_w
                expert_paths[(tier_nm, sk)] = p_exp
                expert_meta[tier_nm][sk] = res_e
                print(
                    f"[fewshot] compiled {tier_nm}/{sk}: {res_e['adapter_mb']}MB in {compile_s:.1f}s | "
                    f"losses={res_e['epoch_losses'][0]}->{res_e['epoch_losses'][-1]} | unrouted_kl={res_e['healthy_kl']}",
                    flush=True,
                )

        # Continual LoRA r=8 (sequential Skill 1 -> ... -> Skill 5, NO replay)
        res_cont_r8 = _train_continual_lora_no_replay(
            student_m,
            global_causal_layers[:4],
            train_samples_by_skill,
            SKILLS_5,
            rank=8,
            alpha_scale=16.0,
            epochs_per_skill=args.epochs_per_skill,
            lr_val=5e-4,
            grad_accum=args.grad_accum,
            healthy_ids=prose_full_ids,
            base_healthy_logp=base_prose_logp,
            base_healthy_nll=base_prose_nll,
            seed=args.seed + 400,
        )
        upd_cr8 = res_cont_r8.pop("weights")
        with _temp_apply_weights(student_m, upd_cr8):
            cr8_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
        p_cont_r8 = exp_dir / "continual_r8.pt"
        torch.save(upd_cr8, p_cont_r8)
        del upd_cr8
        expert_meta["continual_lora_r8_no_replay"] = res_cont_r8
        print(
            f"[fewshot] continual_lora_r8: {res_cont_r8['adapter_mb']}MB in {res_cont_r8['compile_time_s']}s | "
            f"unrouted_kl={res_cont_r8['healthy_kl']}",
            flush=True,
        )

        # Continual LoRA r=64 (sequential Skill 1 -> ... -> Skill 5, NO replay)
        res_cont_r64 = _train_continual_lora_no_replay(
            student_m,
            global_causal_layers[:8],
            train_samples_by_skill,
            SKILLS_5,
            rank=64,
            alpha_scale=64.0,
            epochs_per_skill=args.epochs_per_skill,
            lr_val=3e-4,
            grad_accum=args.grad_accum,
            healthy_ids=prose_full_ids,
            base_healthy_logp=base_prose_logp,
            base_healthy_nll=base_prose_nll,
            seed=args.seed + 500,
        )
        upd_cr64 = res_cont_r64.pop("weights")
        with _temp_apply_weights(student_m, upd_cr64):
            cr64_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
        p_cont_r64 = exp_dir / "continual_r64.pt"
        torch.save(upd_cr64, p_cont_r64)
        del upd_cr64
        expert_meta["continual_lora_r64_no_replay"] = res_cont_r64
        print(
            f"[fewshot] continual_lora_r64: {res_cont_r64['adapter_mb']}MB in {res_cont_r64['compile_time_s']}s | "
            f"unrouted_kl={res_cont_r64['healthy_kl']}",
            flush=True,
        )

        # Offline Oracle Upper Bound: Joint LoRA r=64 on all 30 traces together
        all_30_flat = [sp for sk in SKILLS_5 for sp in train_samples_by_skill[sk]]
        t_j0 = time.perf_counter()
        res_joint = _train_lora_expert(
            student_m,
            global_causal_layers[:8],
            all_30_flat,
            rank=64,
            alpha_scale=64.0,
            epochs=args.epochs_per_skill,
            lr_val=3e-4,
            grad_accum=args.grad_accum,
            healthy_ids=prose_full_ids,
            base_healthy_logp=base_prose_logp,
            base_healthy_nll=base_prose_nll,
            seed=args.seed + 600,
        )
        res_joint["compile_time_s"] = round(time.perf_counter() - t_j0, 2)
        upd_j = res_joint.pop("weights")
        with _temp_apply_weights(student_m, upd_j):
            joint_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
        p_joint = exp_dir / "joint_r64.pt"
        torch.save(upd_j, p_joint)
        del upd_j
        expert_meta["joint_offline_lora_r64"] = res_joint
        print(
            f"[fewshot] joint_offline_lora_r64: {res_joint['adapter_mb']}MB in {res_joint['compile_time_s']}s | "
            f"unrouted_kl={res_joint['healthy_kl']}",
            flush=True,
        )

        mmlu_arms: dict[str, Any] = {
            "base_08b": base_mmlu_summary,
            "fewshot_wsi_r8_routed": _compose_routed_mmlu(mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_scores["fewshot_wsi_r8"], seed=args.seed + 1),
            "fewshot_random_r8_routed": _compose_routed_mmlu(mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_scores["fewshot_random_r8"], seed=args.seed + 2),
            "fewshot_wsi_r64_routed": _compose_routed_mmlu(mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_scores["fewshot_wsi_r64"], seed=args.seed + 3),
            "continual_lora_r8_no_replay": _summarize_mmlu(mmlu_test, cr8_mmlu_raw, base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=args.seed + 4),
            "continual_lora_r64_no_replay": _summarize_mmlu(mmlu_test, cr64_mmlu_raw, base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=args.seed + 5),
            "joint_offline_lora_r64": _summarize_mmlu(mmlu_test, joint_mmlu_raw, base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=args.seed + 6),
        }
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Evaluate all 9 Arms on 84 Test Prompts ----
        print(f"[fewshot] Phase 4: Evaluating 9 paired arms ({backend}, max_new_tokens={args.max_new_tokens})...", flush=True)
        t_p4 = time.perf_counter()

        # Prepare oracle 1-shot prompts using train[0] of each true skill (or guardrail train[0])
        train_ex_by_fam: dict[str, tuple[str, str]] = {}
        for sk in SKILLS_5:
            idx0 = next(i for i, it in enumerate(hard_train) if it["family"] == sk)
            train_ex_by_fam[sk] = (hard_train[idx0]["question"], chosen_responses[idx0])
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

        bake_tmp_dir = Path("/content/_pocket_moe_fewshot_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

        def _eval_ckpt(weights_path: Path, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
            upd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if backend == "vllm":
                _bake_full_checkpoint(Path(args.student), upd, bake_tmp_dir)
                del upd
                gc.collect()
                return _vllm_generate(str(bake_tmp_dir), tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
            return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, param_updates=upd)

        cont_r8_raw = _eval_ckpt(p_cont_r8, test_pids, test_seeds)
        p_cont_r8.unlink(missing_ok=True)
        cont_r64_raw = _eval_ckpt(p_cont_r64, test_pids, test_seeds)
        p_cont_r64.unlink(missing_ok=True)
        joint_r64_raw = _eval_ckpt(p_joint, test_pids, test_seeds)
        p_joint.unlink(missing_ok=True)

        routed_raw_by_tier: dict[str, list[dict[str, Any]]] = {}
        for tier_nm in ("fewshot_wsi_r8", "fewshot_random_r8", "fewshot_wsi_r64"):
            tier_raw: list[dict[str, Any] | None] = [None] * len(all_test)
            for i, cls_nm in enumerate(routed_test_classes):
                if cls_nm == "passthrough":
                    tier_raw[i] = base_raw[i]
            for sk in SKILLS_5:
                idxs = [i for i, cls_nm in enumerate(routed_test_classes) if cls_nm == sk]
                p_exp = expert_paths[(tier_nm, sk)]
                if idxs:
                    sub_outs = _eval_ckpt(p_exp, [test_pids[i] for i in idxs], [test_seeds[i] for i in idxs])
                    for idx, o in zip(idxs, sub_outs):
                        tier_raw[idx] = o
                p_exp.unlink(missing_ok=True)
            routed_raw_by_tier[tier_nm] = [r for r in tier_raw if r is not None]

        if bake_tmp_dir.exists():
            shutil.rmtree(bake_tmp_dir, ignore_errors=True)
        phase4_s = round(time.perf_counter() - t_p4, 2)

    arm_raw_map: dict[str, tuple[list[dict[str, Any]], list[str] | None, bool]] = {
        "base_08b": (base_raw, None, False),
        "teacher_4b": (t_test_raw, None, True),
        "oracle_1shot_08b": (oracle_raw, None, False),
        "fewshot_wsi_r8_routed": (routed_raw_by_tier["fewshot_wsi_r8"], routed_test_classes, False),
        "fewshot_random_r8_routed": (routed_raw_by_tier["fewshot_random_r8"], routed_test_classes, False),
        "fewshot_wsi_r64_routed": (routed_raw_by_tier["fewshot_wsi_r64"], routed_test_classes, False),
        "continual_lora_r8_no_replay": (cont_r8_raw, None, False),
        "continual_lora_r64_no_replay": (cont_r64_raw, None, False),
        "joint_offline_lora_r64": (joint_r64_raw, None, False),
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
        h = sm["hard_60"]
        print(
            f"[fewshot] {arm_nm:28s} | hard60={h['strict_accuracy']*100:5.1f}% "
            f"(old1-4={h['old_skills_1_to_4_acc']*100:5.1f}%, skill5={h['last_skill_5_acc']*100:5.1f}%) "
            f"| guard={sm['guardrail_24']['strict_accuracy']*100:5.1f}%",
            flush=True,
        )

    def _routed_prose_kl(tier_key: str) -> float:
        tot = 0.0
        for p_cls in routed_prose_classes:
            if p_cls in SKILLS_5:
                tot += float(expert_meta[tier_key][p_cls]["healthy_kl"])
        return round(tot / max(1, len(routed_prose_classes)), 6)

    prose_kl = {
        "fewshot_wsi_r8_routed": _routed_prose_kl("fewshot_wsi_r8"),
        "fewshot_random_r8_routed": _routed_prose_kl("fewshot_random_r8"),
        "fewshot_wsi_r64_routed": _routed_prose_kl("fewshot_wsi_r64"),
        "continual_lora_r8_no_replay": float(expert_meta["continual_lora_r8_no_replay"]["healthy_kl"]),
        "continual_lora_r64_no_replay": float(expert_meta["continual_lora_r64_no_replay"]["healthy_kl"]),
        "joint_offline_lora_r64": float(expert_meta["joint_offline_lora_r64"]["healthy_kl"]),
    }

    compile_time_summary = {
        "fewshot_wsi_r8_mean_per_skill_s": round(sum(expert_meta["fewshot_wsi_r8"][s]["compile_time_s"] for s in SKILLS_5) / len(SKILLS_5), 2),
        "fewshot_random_r8_mean_per_skill_s": round(sum(expert_meta["fewshot_random_r8"][s]["compile_time_s"] for s in SKILLS_5) / len(SKILLS_5), 2),
        "fewshot_wsi_r64_mean_per_skill_s": round(sum(expert_meta["fewshot_wsi_r64"][s]["compile_time_s"] for s in SKILLS_5) / len(SKILLS_5), 2),
        "continual_lora_r8_total_s": expert_meta["continual_lora_r8_no_replay"]["compile_time_s"],
        "continual_lora_r64_total_s": expert_meta["continual_lora_r64_no_replay"]["compile_time_s"],
        "joint_offline_lora_r64_total_s": expert_meta["joint_offline_lora_r64"]["compile_time_s"],
    }

    comp_pairs = [
        ("fewshot_wsi_r8_vs_base", "fewshot_wsi_r8_routed", "base_08b"),
        ("fewshot_wsi_r8_vs_oracle_1shot", "fewshot_wsi_r8_routed", "oracle_1shot_08b"),
        ("fewshot_wsi_r8_vs_fewshot_random_r8", "fewshot_wsi_r8_routed", "fewshot_random_r8_routed"),
        ("fewshot_wsi_r8_vs_fewshot_wsi_r64", "fewshot_wsi_r8_routed", "fewshot_wsi_r64_routed"),
        ("fewshot_wsi_r8_vs_continual_r8", "fewshot_wsi_r8_routed", "continual_lora_r8_no_replay"),
        ("fewshot_wsi_r8_vs_continual_r64", "fewshot_wsi_r8_routed", "continual_lora_r64_no_replay"),
        ("fewshot_wsi_r8_vs_joint_offline_r64", "fewshot_wsi_r8_routed", "joint_offline_lora_r64"),
        ("continual_r64_vs_base", "continual_lora_r64_no_replay", "base_08b"),
        ("joint_offline_r64_vs_base", "joint_offline_lora_r64", "base_08b"),
    ]
    paired_hard_60 = {
        lbl: _paired_bootstrap(arm_rows[a][:n_hard], arm_rows[b][:n_hard], seed=args.seed + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }
    paired_old_skills_48 = {
        lbl: _paired_bootstrap(
            [r for r in arm_rows[a][:n_hard] if r["family"] in OLD_SKILLS],
            [r for r in arm_rows[b][:n_hard] if r["family"] in OLD_SKILLS],
            seed=args.seed + 100 + i,
        )
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    h_acc = lambda k: float(arm_summaries[k]["hard_60"]["strict_accuracy"])  # noqa: E731
    old_acc = lambda k: float(arm_summaries[k]["hard_60"]["old_skills_1_to_4_acc"])  # noqa: E731
    nm_mmlu = lambda k: float(mmlu_arms[k]["non_math"]["accuracy"])  # noqa: E731

    return {
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "backend": backend,
        "k_shots_per_skill": int(args.k_shots),
        "epochs_per_skill": int(args.epochs_per_skill),
        "max_new_tokens": int(args.max_new_tokens),
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_wsi_router_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "compile_time_s": compile_time_summary,
        "teacher_4b_train_kept_frac": teacher_kept_frac,
        "invariance": invariance,
        "causal_layers": {**causal_layers, "global": global_causal_layers},
        "random_layers_r8": {
            sk: {
                "layers": random_layers_r8[sk],
                "wsi_layers": causal_layers[sk][:4],
                "overlap_with_wsi": sorted(set(random_layers_r8[sk]) & set(causal_layers[sk][:4])),
                "n_overlap": len(set(random_layers_r8[sk]) & set(causal_layers[sk][:4])),
            }
            for sk in SKILLS_5
        },
        "router_report": router_report,
        "expert_meta": expert_meta,
        "prose_kl": prose_kl,
        "generation_arms": arm_summaries,
        "mmlu_arms": mmlu_arms,
        "paired_hard_60": paired_hard_60,
        "paired_old_skills_48": paired_old_skills_48,
        "criteria": {
            "a_fewshot_beats_base_and_1shot": bool(
                h_acc("fewshot_wsi_r8_routed") >= h_acc("base_08b") + 0.15
                and h_acc("fewshot_wsi_r8_routed") > h_acc("oracle_1shot_08b")
            ),
            "b_zero_sequential_forgetting_vs_continual_r64": bool(
                old_acc("fewshot_wsi_r8_routed") > old_acc("continual_lora_r64_no_replay") + 0.10
            ),
            "c_wsi_beats_random_at_k6_r8": bool(
                h_acc("fewshot_wsi_r8_routed") > h_acc("fewshot_random_r8_routed")
            ),
            "d_r8_matches_or_beats_r64_at_k6": bool(
                h_acc("fewshot_wsi_r8_routed") >= h_acc("fewshot_wsi_r64_routed") - 0.02
            ),
            "e_zero_general_forgetting": bool(
                prose_kl["fewshot_wsi_r8_routed"] <= 0.005
                and nm_mmlu("fewshot_wsi_r8_routed") >= nm_mmlu("base_08b") - 0.015
                and arm_summaries["fewshot_wsi_r8_routed"]["guardrail_24"]["strict_accuracy"] >= arm_summaries["base_08b"]["guardrail_24"]["strict_accuracy"] - 0.02
            ),
        },
        "per_item_generation": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU check")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--k-shots", type=int, default=6)
    ap.add_argument("--epochs-per-skill", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
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
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
