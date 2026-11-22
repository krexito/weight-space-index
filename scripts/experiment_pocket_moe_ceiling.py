"""Ceiling Test: "Pocket MoE" Distilling Qwen3.5-4B -> Qwen3.5-0.8B (1 MB -> 10 MB -> 70 MB per expert).

Tests the upper capacity limit of WSI-localized, prefill-routed Pocket MoE experts under `enable_thinking=False`
using Top-32 teacher logit KL + CE distillation from `Qwen/Qwen3.5-4B` into `Qwen/Qwen3.5-0.8B` across
3 genuinely hard multi-step families (where `0.8B base_no_think` struggles) + 2 easy guardrail families + prose controls.

Arms evaluated on 60 held-out test problems (`36` hard: `18` renumbered + `18` rephrased_renumbered with
disjoint test-only scenarios; `24` easy guardrails) and `12` held-out benign prose controls:
  1. `base_08b`               : 0.8B base (`enable_thinking=False`)
  2. `teacher_4b`             : 4B teacher ceiling (`enable_thinking=False`)
  3. `oracle_1shot_08b`       : 0.8B base with 1 solved train example in context
  4. `moe_micro_routed`       : ~1 MB/expert (rank=8 LoRA on gate+up+down+attn of top-4 WSI causal layers)
  5. `moe_large_routed`       : ~10 MB/expert (rank=64 LoRA on gate+up+down+attn of top-8 WSI causal layers)
  6. `moe_full_layers_routed` : ~70 MB/expert (full-rank weights of top-4 WSI causal layers)
  7. `global_full_layers`     : ~70 MB single unrouted full-layer expert trained jointly across all 3 hard families

CPU smoke test::
    python scripts/experiment_pocket_moe_ceiling.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_moe_ceiling.py --teacher /content/teacher_4b --student /content/student_08b --out /content/pocket_moe_ceiling_l4.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
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

import moi_loop_router_vllm as lr  # noqa: E402
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _PlantedReasoningLM,
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
    build_reasoning_dataset,
    compute_invariance_metrics,
)
from experiment_reasoning_distill_v2 import (  # noqa: E402
    _format_messages,
    _paired_vs,
    _select_backend,
    format_question,
    grade_response,
)
from wsi import CompactTrace, capture_causal_trace, diagnose_causal_traces  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _resolve_target_span, _tail_logits, _to_seq_tensors  # noqa: E402

HARD_FAMILIES: tuple[str, ...] = (
    "rate_work_combined",
    "compound_percent_chain",
    "weighted_mixture_transfer",
)
GUARDRAIL_FAMILIES: tuple[str, ...] = (
    "linear_system_2var",
    "relative_ratio_shift",
)
ROUTER_CLASSES: tuple[str, ...] = (*HARD_FAMILIES, "passthrough")


# ------------------------------------------------------------------ Deterministic Multi-Scenario Dataset
_RATE_TRAIN_SCENARIOS: tuple[tuple[str, str, str, str, str], ...] = (
    ("factory_machines", "Machine A", "Machine B", "parts", "produces"),
    ("water_pumps", "Pump Alpha", "Pump Beta", "liters", "delivers"),
    ("print_presses", "Press One", "Press Two", "pages", "prints"),
    ("baker_ovens", "Oven North", "Oven South", "loaves", "bakes"),
    ("harvesters", "Harvester X", "Harvester Y", "bushels", "harvests"),
    ("bottling_lines", "Line A", "Line B", "bottles", "fills"),
    ("sewing_looms", "Loom East", "Loom West", "meters of cloth", "weaves"),
    ("lathe_mills", "Mill Alpha", "Mill Beta", "steel brackets", "cuts"),
    ("cranes_dock", "Crane 1", "Crane 2", "crates", "loads"),
    ("paint_robots", "Robot Red", "Robot Blue", "panels", "coats"),
    ("grain_augers", "Auger Primary", "Auger Backup", "kilograms of wheat", "moves"),
    ("brick_kilns", "Kiln One", "Kiln Two", "bricks", "fires"),
    ("solar_chargers", "Charger A", "Charger B", "watt-hours", "stores"),
    ("pipe_valves", "Valve North", "Valve South", "gallons", "drains"),
    ("packaging_belts", "Belt Fast", "Belt Aux", "cartons", "seals"),
    ("mining_drills", "Drill Core", "Drill Side", "tons of ore", "extracts"),
)
_RATE_TEST_SCENARIOS: tuple[tuple[str, str, str, str, str], ...] = (
    ("scanners_archive", "Scanner A", "Scanner B", "documents", "digitizes"),
    ("compiler_cluster", "Node Alpha", "Node Omega", "build tasks", "completes"),
    ("courier_drones", "Drone Squad X", "Drone Squad Y", "parcels", "dispatches"),
    ("wind_turbines", "Turbine Coastal", "Turbine Ridge", "kilowatt-hours", "generates"),
    ("roastery_drums", "Roaster Main", "Roaster Reserve", "bags of coffee", "roasts"),
    ("desalination_rigs", "Unit Deep", "Unit Shore", "cubic meters", "purifies"),
)

_PCT_TRAIN_SCENARIOS: tuple[tuple[str, str, str, str, str], ...] = (
    ("warehouse_boxes", "warehouse", "boxes", "Monday shipments", "Tuesday dispatches"),
    ("grant_budget", "research grant", "dollars", "lab equipment", "travel stipends"),
    ("reservoir_water", "water reservoir", "kiloliters", "city supply", "farm irrigation"),
    ("seed_bank", "seed vault", "packets", "spring planting", "greenhouse trials"),
    ("lumber_yard", "timber depot", "planks", "framing orders", "roofing orders"),
    ("battery_pack", "battery station", "watt-hours", "daytime grid draw", "night backup"),
    ("library_books", "archive collection", "volumes", "regional branch loans", "restoration"),
    ("fuel_depot", "fuel reserve", "gallons", "morning fleet refueling", "generator tests"),
    ("cloud_credits", "compute quota", "credits", "pretraining runs", "evaluation sweeps"),
    ("bakery_flour", "pantry stock", "bags of flour", "morning pastries", "afternoon loaves"),
    ("vineyard_barrels", "cellar reserve", "bottles", "local restaurants", "export crates"),
    ("hospital_masks", "medical supply", "masks", "outpatient clinics", "emergency wards"),
    ("quarry_gravel", "stone stockpile", "tons of gravel", "highway paving", "bridge repair"),
    ("museum_tickets", "ticket allotment", "passes", "school groups", "member pre-sales"),
    ("data_storage", "cold storage pool", "terabytes", "video backups", "sensor logs"),
    ("orchard_apples", "harvest bin", "apples", "cider pressing", "market baskets"),
)
_PCT_TEST_SCENARIOS: tuple[tuple[str, str, str, str, str], ...] = (
    ("satellite_propellant", "thruster tank", "kilograms of fuel", "orbit raising", "station keeping"),
    ("museum_conservation", "donor endowment", "euros", "sculpture wing", "manuscript preservation"),
    ("fishery_ice", "cold vault", "blocks of ice", "trawler departure", "dockside packing"),
    ("relief_tents", "emergency depot", "shelter kits", "coastal shelters", "mountain camps"),
    ("textile_spool", "weaving mill stock", "spools of yarn", "winter coats", "sample swatches"),
    ("printing_ink", "pressroom inventory", "cartridges", "newspaper run", "magazine inserts"),
)

_MIX_TRAIN_SCENARIOS: tuple[tuple[str, str, str, str], ...] = (
    ("saline_lab", "saline solution", "liters", "salt"),
    ("juice_blend", "fruit punch", "liters", "pure juice"),
    ("alloy_melt", "bronze alloy", "kilograms", "tin"),
    ("acid_bath", "acid solution", "milliliters", "acid"),
    ("antifreeze_mix", "coolant blend", "liters", "glycol"),
    ("milk_dairy", "milk blend", "gallons", "butterfat"),
    ("syrup_vat", "sugar syrup", "liters", "sugar"),
    ("fertilizer_tank", "liquid fertilizer", "liters", "nitrogen"),
    ("wine_cellar", "wine blend", "liters", "merlot"),
    ("coffee_blend", "coffee mix", "kilograms", "arabica"),
    ("gold_refinery", "gold alloy", "grams", "pure gold"),
    ("chlorine_pool", "pool cleanser", "liters", "chlorine"),
    ("tea_infusion", "herbal tea blend", "grams", "mint leaf"),
    ("ethanol_fuel", "fuel blend", "gallons", "ethanol"),
    ("resin_mix", "epoxy mixture", "grams", "hardener"),
    ("oat_cereal", "grain mix", "kilograms", "oats"),
)
_MIX_TEST_SCENARIOS: tuple[tuple[str, str, str, str], ...] = (
    ("silver_solder", "silver solder batch", "grams", "silver"),
    ("brine_pickling", "pickling brine", "liters", "sodium chloride"),
    ("nut_roast", "trail mix", "kilograms", "almonds"),
    ("dye_vat", "indigo dye bath", "liters", "concentrated dye"),
    ("concrete_mortar", "dry mortar blend", "kilograms", "cement"),
    ("perfume_base", "aromatic oil blend", "milliliters", "essential oil"),
)


def _solve_rate_work(r1: int, r2: int, d: int, k: int) -> tuple[int, str, int]:
    solo = d * r1
    comb = r1 + r2
    tot = solo + k * comb
    rem = tot - solo
    resp = (
        f"Step 1: Compute the output completed alone by the first unit: {d} * {r1} = {solo}.\n"
        f"Step 2: Subtract this from the total target to find the remaining output: {tot} - {solo} = {rem}.\n"
        f"Step 3: Add both hourly rates together: {r1} + {r2} = {comb}.\n"
        f"Step 4: Divide the remaining output by the combined rate: {rem} / {comb} = {k}.\n"
        f"The second unit ran for \\boxed{{{k}}} hours."
    )
    return tot, resp, k


def _solve_pct_chain(x_orig: int, p1: int, p2: int, c_extra: int) -> tuple[int, str]:
    assert (x_orig * (100 - p1)) % 100 == 0
    rem1 = (x_orig * (100 - p1)) // 100
    assert (rem1 * (100 - p2)) % 100 == 0
    after_p2 = (rem1 * (100 - p2)) // 100
    r_final = after_p2 - c_extra
    assert r_final > 0
    k1 = 100 - p1
    k2 = 100 - p2
    resp = (
        f"Work backward from the final leftover {r_final}.\n"
        f"Step 1: Before removing the extra {c_extra}, there were {r_final} + {c_extra} = {after_p2}.\n"
        f"Step 2: Because {p2}% was removed, {after_p2} is {k2}% of the first remainder: "
        f"{after_p2} * 100 / {k2} = {rem1}.\n"
        f"Step 3: Because {p1}% was removed initially, {rem1} is {k1}% of the starting total: "
        f"{rem1} * 100 / {k1} = {x_orig}.\n"
        f"Therefore, the initial amount was \\boxed{{{x_orig}}}."
    )
    return r_final, resp


def _solve_mixture(v_rem: int, d: int, p1: int, pt: int, p2: int) -> tuple[int, int, str]:
    v1 = v_rem + d
    num_diff = pt - p1
    den = p2 - pt
    assert num_diff > 0 and den > 0 and (v_rem * num_diff) % den == 0
    rhs = v_rem * num_diff
    x_ans = rhs // den
    resp = (
        f"Step 1: After draining {d}, the remaining quantity at {p1}% is {v1} - {d} = {v_rem}.\n"
        f"Step 2: Let x be the amount added at {p2}% to reach {pt}%. "
        f"Equating concentration change gives ({p2} - {pt}) * x = {v_rem} * ({pt} - {p1}).\n"
        f"Step 3: Simplify both sides: {den} * x = {v_rem} * {num_diff} = {rhs}.\n"
        f"Step 4: Divide by {den}: x = {rhs} / {den} = {x_ans}.\n"
        f"Therefore, \\boxed{{{x_ans}}} units must be added."
    )
    return v1, x_ans, resp


def build_ceiling_dataset() -> dict[str, Any]:
    """Build 144 diverse hard train items, 36 held-out hard test items, 24 guardrail test items,
    32 guardrail train prompts for router training, and 20 benign prose controls."""
    assert {s[0] for s in _RATE_TRAIN_SCENARIOS}.isdisjoint({s[0] for s in _RATE_TEST_SCENARIOS})
    assert {s[0] for s in _PCT_TRAIN_SCENARIOS}.isdisjoint({s[0] for s in _PCT_TEST_SCENARIOS})
    assert {s[0] for s in _MIX_TRAIN_SCENARIOS}.isdisjoint({s[0] for s in _MIX_TEST_SCENARIOS})

    hard_train: list[dict[str, Any]] = []
    hard_test: list[dict[str, Any]] = []

    # ---- Family 1: rate_work_combined (60 distinct integer tuples: 48 train + 12 test) ----
    rate_tuples: list[tuple[int, int, int, int]] = []
    for i in range(60):
        r1 = 11 + (i % 13)
        r2 = 15 + ((i * 3) % 17)
        if r1 == r2:
            r2 += 2
        d = 2 + (i % 5)
        k = 6 + ((i * 7 + 3) % 11)
        cand = (r1, r2, d, k)
        while cand in rate_tuples:
            k += 1
            cand = (r1, r2, d, k)
        rate_tuples.append(cand)

    def _fmt_rate_q(sc: tuple[str, str, str, str, str], r1: int, r2: int, d: int, tot: int) -> str:
        _, a, b, unit, verb = sc
        return (
            f"{a} {verb} {r1} {unit} per hour and {b} {verb} {r2} {unit} per hour. "
            f"{a} operates alone for {d} hours, and then {b} joins so both work together until a total of {tot} {unit} "
            f"have been completed. For how many hours did {b} run?"
        )

    for idx in range(48):
        sc = _RATE_TRAIN_SCENARIOS[idx % 16]
        r1, r2, d, k = rate_tuples[idx]
        tot, resp, ans = _solve_rate_work(r1, r2, d, k)
        hard_train.append({
            "id": f"train_rate_{idx}",
            "family": "rate_work_combined",
            "variant": "train",
            "scenario": sc[0],
            "question": _fmt_rate_q(sc, r1, r2, d, tot),
            "fallback_response": resp,
            "answer": ans,
        })
    for idx in range(6):
        sc = _RATE_TRAIN_SCENARIOS[idx]
        r1, r2, d, k = rate_tuples[48 + idx]
        tot, resp, ans = _solve_rate_work(r1, r2, d, k)
        hard_test.append({
            "id": f"test_rate_renum_{idx}",
            "family": "rate_work_combined",
            "variant": "renumbered",
            "scenario": sc[0],
            "question": _fmt_rate_q(sc, r1, r2, d, tot),
            "fallback_response": resp,
            "answer": ans,
        })
    for idx in range(6):
        sc = _RATE_TEST_SCENARIOS[idx]
        r1, r2, d, k = rate_tuples[54 + idx]
        tot, resp, ans = _solve_rate_work(r1, r2, d, k)
        hard_test.append({
            "id": f"test_rate_rephrase_{idx}",
            "family": "rate_work_combined",
            "variant": "rephrased_renumbered",
            "scenario": sc[0],
            "question": _fmt_rate_q(sc, r1, r2, d, tot),
            "fallback_response": resp,
            "answer": ans,
        })

    # ---- Family 2: compound_percent_chain (60 distinct integer tuples: 48 train + 12 test) ----
    pct_pairs = [
        (20, 25), (25, 20), (30, 20), (20, 30), (10, 20), (20, 10),
        (25, 40), (40, 25), (15, 20), (20, 15), (30, 25), (25, 30),
    ]
    pct_tuples: list[tuple[int, int, int, int]] = []
    seen_pct: set[tuple[int, int, int, int]] = set()
    p_cursor = 0
    while len(pct_tuples) < 60:
        p1, p2 = pct_pairs[p_cursor % len(pct_pairs)]
        mult = 2 + (p_cursor // len(pct_pairs))
        step_x = 10000 // math.gcd(10000, (100 - p1) * (100 - p2))
        x_orig = step_x * mult
        if x_orig < 200:
            x_orig *= max(2, 200 // step_x)
        c_extra = 12 + (p_cursor * 5) % 37
        tup = (x_orig, p1, p2, c_extra)
        if tup not in seen_pct:
            after_p2 = (x_orig * (100 - p1) // 100) * (100 - p2) // 100
            if after_p2 > c_extra + 20:
                seen_pct.add(tup)
                pct_tuples.append(tup)
        p_cursor += 1

    def _fmt_pct_q(sc: tuple[str, str, str, str, str], p1: int, p2: int, c_extra: int, r_final: int) -> str:
        _, container, unit, step1, step2 = sc
        return (
            f"A {container} spent {p1}% of its initial {unit} on {step1}. "
            f"Next, {p2}% of the remaining {unit} plus an extra {c_extra} {unit} went to {step2}, "
            f"leaving {r_final} {unit} in reserve. How many {unit} were in the {container} originally?"
        )

    for idx in range(48):
        sc = _PCT_TRAIN_SCENARIOS[idx % 16]
        x_orig, p1, p2, c_extra = pct_tuples[idx]
        r_final, resp = _solve_pct_chain(x_orig, p1, p2, c_extra)
        hard_train.append({
            "id": f"train_pct_{idx}",
            "family": "compound_percent_chain",
            "variant": "train",
            "scenario": sc[0],
            "question": _fmt_pct_q(sc, p1, p2, c_extra, r_final),
            "fallback_response": resp,
            "answer": x_orig,
        })
    for idx in range(6):
        sc = _PCT_TRAIN_SCENARIOS[idx]
        x_orig, p1, p2, c_extra = pct_tuples[48 + idx]
        r_final, resp = _solve_pct_chain(x_orig, p1, p2, c_extra)
        hard_test.append({
            "id": f"test_pct_renum_{idx}",
            "family": "compound_percent_chain",
            "variant": "renumbered",
            "scenario": sc[0],
            "question": _fmt_pct_q(sc, p1, p2, c_extra, r_final),
            "fallback_response": resp,
            "answer": x_orig,
        })
    for idx in range(6):
        sc = _PCT_TEST_SCENARIOS[idx]
        x_orig, p1, p2, c_extra = pct_tuples[54 + idx]
        r_final, resp = _solve_pct_chain(x_orig, p1, p2, c_extra)
        hard_test.append({
            "id": f"test_pct_rephrase_{idx}",
            "family": "compound_percent_chain",
            "variant": "rephrased_renumbered",
            "scenario": sc[0],
            "question": _fmt_pct_q(sc, p1, p2, c_extra, r_final),
            "fallback_response": resp,
            "answer": x_orig,
        })

    # ---- Family 3: weighted_mixture_transfer (60 distinct integer tuples: 48 train + 12 test) ----
    mix_triples = [
        (10, 20, 50), (15, 25, 45), (12, 24, 60), (8, 16, 40), (20, 30, 60), (10, 25, 55),
        (15, 30, 50), (18, 28, 48), (10, 22, 46), (12, 20, 44), (16, 28, 52), (14, 24, 54),
    ]
    mix_tuples: list[tuple[int, int, int, int, int]] = []
    seen_mix: set[tuple[int, int, int, int, int]] = set()
    m_cursor = 0
    while len(mix_tuples) < 60:
        p1, pt, p2 = mix_triples[m_cursor % len(mix_triples)]
        den = p2 - pt
        step_v = den // math.gcd(den, pt - p1)
        mult = 2 + (m_cursor // len(mix_triples)) + ((m_cursor * 3) % 5)
        v_rem = step_v * mult * 5
        d = 10 + (m_cursor * 7) % 35
        tup = (v_rem, d, p1, pt, p2)
        if tup not in seen_mix:
            seen_mix.add(tup)
            mix_tuples.append(tup)
        m_cursor += 1

    def _fmt_mix_q(sc: tuple[str, str, str, str], v1: int, p1: int, d: int, p2: int, pt: int) -> str:
        _, mat, unit, comp = sc
        return (
            f"A vat holds {v1} {unit} of a {mat} that is {p1}% {comp}. "
            f"First, {d} {unit} of the {mat} are removed from the vat, and then pure {p2}% {comp} mixture is added "
            f"until the new blend becomes {pt}% {comp}. How many {unit} of the {p2}% mixture must be added?"
        )

    for idx in range(48):
        sc = _MIX_TRAIN_SCENARIOS[idx % 16]
        v_rem, d, p1, pt, p2 = mix_tuples[idx]
        v1, ans, resp = _solve_mixture(v_rem, d, p1, pt, p2)
        hard_train.append({
            "id": f"train_mix_{idx}",
            "family": "weighted_mixture_transfer",
            "variant": "train",
            "scenario": sc[0],
            "question": _fmt_mix_q(sc, v1, p1, d, p2, pt),
            "fallback_response": resp,
            "answer": ans,
        })
    for idx in range(6):
        sc = _MIX_TRAIN_SCENARIOS[idx]
        v_rem, d, p1, pt, p2 = mix_tuples[48 + idx]
        v1, ans, resp = _solve_mixture(v_rem, d, p1, pt, p2)
        hard_test.append({
            "id": f"test_mix_renum_{idx}",
            "family": "weighted_mixture_transfer",
            "variant": "renumbered",
            "scenario": sc[0],
            "question": _fmt_mix_q(sc, v1, p1, d, p2, pt),
            "fallback_response": resp,
            "answer": ans,
        })
    for idx in range(6):
        sc = _MIX_TEST_SCENARIOS[idx]
        v_rem, d, p1, pt, p2 = mix_tuples[54 + idx]
        v1, ans, resp = _solve_mixture(v_rem, d, p1, pt, p2)
        hard_test.append({
            "id": f"test_mix_rephrase_{idx}",
            "family": "weighted_mixture_transfer",
            "variant": "rephrased_renumbered",
            "scenario": sc[0],
            "question": _fmt_mix_q(sc, v1, p1, d, p2, pt),
            "fallback_response": resp,
            "answer": ans,
        })

    # ---- Guardrails (linear_system_2var + relative_ratio_shift) ----
    base_ds = build_reasoning_dataset()
    guardrail_test = [it for it in base_ds["test"] if it["family"] in GUARDRAIL_FAMILIES]
    # Use the 8 existing train items plus 24 disjoint synthetic guardrail prompts (16 per family) for router fitting
    guardrail_router_train: list[dict[str, Any]] = [
        it for it in base_ds["train"] if it["family"] in GUARDRAIL_FAMILIES
    ]
    for g_idx in range(12):
        a = 7 + (g_idx % 6)
        b = 2 + (g_idx % 4)
        x = 20 + g_idx * 2
        y = 14 + g_idx
        n = x + y
        c = a * x + b * y
        guardrail_router_train.append({
            "id": f"train_g1_{g_idx}",
            "family": "linear_system_2var",
            "question": (
                f"A bookstore sold {n} books today: hardcovers for ${a} each and paperbacks for ${b} each. "
                f"Total sales were ${c}. How many hardcovers were sold?"
            ),
            "answer": x,
        })
    ratio_train_params = [
        (4, 2, 20), (5, 3, 15), (3, 2, 16), (5, 2, 18), (6, 3, 18), (4, 3, 10),
        (5, 3, 17), (3, 2, 22), (4, 2, 24), (6, 2, 16), (5, 2, 24), (4, 3, 12),
    ]
    for g_idx, (k1, k2, delta) in enumerate(ratio_train_params):
        b_v = delta * (k2 - 1) // (k1 - k2)
        a_v = k1 * b_v
        guardrail_router_train.append({
            "id": f"train_g2_{g_idx}",
            "family": "relative_ratio_shift",
            "question": (
                f"Warehouse North holds {k1} times as many pallets as Warehouse South. "
                f"After adding {delta} pallets to both warehouses, North will hold {k2} times as many as South. "
                f"How many pallets are in North now?"
            ),
            "answer": a_v,
        })

    # ---- 20 Benign Prose Controls (12 held-out KL/FPR + 8 router passthrough train) ----
    prose_all = list(base_ds["healthy_controls"]) + [
        "Explain what git rebase does compared to git merge in two clear sentences.",
        "What distinguishes an exothermic chemical reaction from an endothermic one?",
        "Write a polite two-sentence note declining a Friday meeting due to a conflict.",
        "Convert 72 degrees Fahrenheit into Celsius and show the quick calculation.",
    ]

    return {
        "hard_families": list(HARD_FAMILIES),
        "guardrail_families": list(GUARDRAIL_FAMILIES),
        "hard_train": hard_train,
        "hard_test": hard_test,
        "guardrail_test": guardrail_test,
        "guardrail_router_train": guardrail_router_train,
        "prose_heldout": prose_all[:12],
        "prose_router_train": prose_all[12:20],
    }


# ------------------------------------------------------------------ Top-K KL + CE Loss & Trainers
def distill_step_loss(
    logits_span: torch.Tensor,
    targets: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
    topk_logits: torch.Tensor | None = None,
    *,
    T: float = 1.5,
    kl_weight: float = 0.7,
) -> torch.Tensor:
    """Weighted combination of top-K teacher KL divergence and hard-target cross entropy."""
    lp_s = F.log_softmax(logits_span.float(), dim=-1)
    loss_ce = F.nll_loss(lp_s, targets.long())
    if topk_ids is None or topk_logits is None or topk_ids.numel() == 0 or kl_weight <= 0.0:
        return loss_ce
    p_t = F.softmax(topk_logits.float() / float(T), dim=-1)
    lp_s_T = F.log_softmax(logits_span.float() / float(T), dim=-1).gather(1, topk_ids.long())
    loss_kl = (float(T) * float(T)) * (p_t * (p_t.clamp_min(1e-8).log() - lp_s_T)).sum(dim=-1).mean()
    return float(kl_weight) * loss_kl + (1.0 - float(kl_weight)) * loss_ce


class _LoRALinear(nn.Module):
    """Lightweight rank-r residual adapter wrapping a frozen `nn.Linear` module."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha_scale: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha_scale) / max(1, self.rank)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty((self.rank, base.in_features), device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.rank), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
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
        return int((self.lora_A.numel() + self.lora_B.numel()) * 2)  # serialized in bf16


def _norm_param_key(name: str) -> str:
    m = re.search(r"(layers\.\d+\..+)$", name)
    if m and "mtp" not in name and "visual" not in name:
        return m.group(1)
    return name


def _block_prefixes(model: nn.Module) -> dict[int, str]:
    _, blocks = find_blocks(model)
    id_to_idx = {id(blk): i for i, blk in enumerate(blocks)}
    out: dict[int, str] = {}
    for name, mod in model.named_modules():
        if id(mod) in id_to_idx:
            out[id_to_idx[id(mod)]] = name
    return out


def _enable_emb_grad_hook(model: nn.Module) -> list[Any]:
    handles: list[Any] = []
    emb_mod = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else getattr(getattr(model, "model", model), "embed_tokens", None)
    if emb_mod is not None:
        def _emb_hook(_m, _inp, emb_out):
            if torch.is_grad_enabled() and isinstance(emb_out, torch.Tensor) and not emb_out.requires_grad:
                return emb_out.detach().requires_grad_(True)
            return emb_out
        handles.append(emb_mod.register_forward_hook(_emb_hook))
    return handles


@torch.no_grad()
def eval_healthy_drift(
    model: nn.Module,
    healthy_ids: list[torch.Tensor],
    base_healthy_logp: list[torch.Tensor | None],
    base_healthy_nll: float,
) -> dict[str, float]:
    """Compute exact `healthy_kl` and `delta_healthy_nll` on held-out prose controls."""
    dev = next(model.parameters()).device
    nll_sum, kl_sum, n_h = 0.0, 0.0, 0
    for i, ids in enumerate(healthy_ids):
        if ids.shape[1] < 2:
            continue
        logits = _tail_logits(model, ids.to(dev), 0)[:-1].float()
        lp = F.log_softmax(logits, dim=-1)
        nll_sum += float(F.nll_loss(lp, ids[0, 1:].to(dev), reduction="sum"))
        n_h += int(ids.shape[1] - 1)
        lp_b_cpu = base_healthy_logp[i] if i < len(base_healthy_logp) else None
        if lp_b_cpu is not None:
            lp_b = lp_b_cpu.to(device=dev, dtype=torch.float32)
            kl_sum += float((lp_b.exp() * (lp_b - lp)).sum(dim=-1).clamp_min(0.0).sum())
            del lp_b
        del logits, lp
    cur_nll = nll_sum / max(1, n_h)
    return {
        "healthy_kl": round(kl_sum / max(1, n_h), 6),
        "delta_healthy_nll": round(cur_nll - base_healthy_nll, 6),
    }


def _train_lora_expert(
    model: nn.Module,
    layer_idxs: list[int],
    train_samples: list[dict[str, Any]],
    *,
    rank: int,
    alpha_scale: float,
    epochs: int,
    lr_val: float,
    grad_accum: int,
    healthy_ids: list[torch.Tensor] | None = None,
    base_healthy_logp: list[torch.Tensor | None] | None = None,
    base_healthy_nll: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Wrap all `nn.Linear` modules in `layer_idxs` with `_LoRALinear`, train with Top-K KL + CE, and restore."""
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
        targets = [(nm, m) for nm, m in blk.named_modules() if isinstance(m, nn.Linear)]
        for sub_nm, lin in targets:
            parts = sub_nm.split(".")
            parent = blk
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            wrapper = _LoRALinear(lin, rank=rank, alpha_scale=alpha_scale)
            setattr(parent, attr, wrapper)
            full_param_name = f"{blk_pref}.{sub_nm}.weight"
            wrapped.append((parent, attr, lin, full_param_name, wrapper))

    lora_params = [p for *_rest, w in wrapped for p in (w.lora_A, w.lora_B)]
    adapter_mb = round(sum(w.adapter_bytes() for *_rest, w in wrapped) / (1024**2), 4)
    emb_handles = _enable_emb_grad_hook(model)
    loss_history: list[float] = []

    try:
        model.eval()
        total_steps = max(1, (len(train_samples) * epochs + grad_accum - 1) // grad_accum)
        opt = torch.optim.AdamW(lora_params, lr=float(lr_val), weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=float(lr_val) * 0.1)
        rng = torch.Generator().manual_seed(seed)

        for ep in range(epochs):
            perm = torch.randperm(len(train_samples), generator=rng).tolist()
            ep_losses: list[float] = []
            for start in range(0, len(perm), grad_accum):
                chunk = perm[start : start + grad_accum]
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for idx_s in chunk:
                    sp = train_samples[idx_s]
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
            loss_history.append(round(sum(ep_losses) / max(1, len(ep_losses)), 5))

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
        "epoch_losses": loss_history,
        **drift,
    }


def _train_full_layers_expert(
    model: nn.Module,
    layer_idxs: list[int],
    train_samples: list[dict[str, Any]],
    *,
    epochs: int,
    lr_val: float,
    grad_accum: int,
    healthy_ids: list[torch.Tensor] | None = None,
    base_healthy_logp: list[torch.Tensor | None] | None = None,
    base_healthy_nll: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train ALL parameters inside `layer_idxs` using FP32 master-weight AdamW, snapshot weights, and restore."""
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

    train_items: list[tuple[str, nn.Parameter, torch.Tensor]] = []
    for li in layer_idxs:
        blk = blocks[li]
        blk_pref = prefixes[li]
        for sub_nm, p in blk.named_parameters():
            full_nm = f"{blk_pref}.{sub_nm}"
            p.requires_grad_(True)
            train_items.append((full_nm, p, p.detach().cpu().clone()))

    masters = [p.detach().float().clone().requires_grad_(True) for _, p, _ in train_items]
    shard_mb = round(sum(p.numel() * 2 for _, p, _ in train_items) / (1024**2), 3)
    emb_handles = _enable_emb_grad_hook(model)
    loss_history: list[float] = []

    try:
        model.eval()
        total_steps = max(1, (len(train_samples) * epochs + grad_accum - 1) // grad_accum)
        opt = torch.optim.AdamW(masters, lr=float(lr_val), weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=float(lr_val) * 0.1)
        rng = torch.Generator().manual_seed(seed)

        for ep in range(epochs):
            perm = torch.randperm(len(train_samples), generator=rng).tolist()
            ep_losses: list[float] = []
            for start in range(0, len(perm), grad_accum):
                chunk = perm[start : start + grad_accum]
                for _, p, _ in train_items:
                    p.grad = None
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for idx_s in chunk:
                    sp = train_samples[idx_s]
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

                for m, (_, p, _) in zip(masters, train_items):
                    m.grad = p.grad.detach().float() if p.grad is not None else torch.zeros_like(m)
                    p.grad = None
                torch.nn.utils.clip_grad_norm_(masters, 1.0)
                opt.step()
                sched.step()
                with torch.no_grad():
                    for m, (_, p, _) in zip(masters, train_items):
                        p.data.copy_(m.data.to(p.dtype))
                ep_losses.append(batch_loss)
            loss_history.append(round(sum(ep_losses) / max(1, len(ep_losses)), 5))

        drift = (
            eval_healthy_drift(model, healthy_ids, base_healthy_logp, base_healthy_nll)
            if (healthy_ids is not None and base_healthy_logp is not None)
            else {"healthy_kl": 0.0, "delta_healthy_nll": 0.0}
        )
        updated_weights = {full_nm: p.detach().cpu().clone() for full_nm, p, _ in train_items}
    finally:
        for h in emb_handles:
            h.remove()
        with torch.no_grad():
            for _, p, orig_cpu in train_items:
                p.data.copy_(orig_cpu.to(p.device))
                p.grad = None
        if enabled_gc and hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
        if was_training:
            model.train()
        for p, req in orig_req:
            p.requires_grad_(req)
        del masters
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "weights": updated_weights,
        "adapter_mb": shard_mb,
        "merged_shard_mb": shard_mb,
        "layer_idxs": list(layer_idxs),
        "epoch_losses": loss_history,
        **drift,
    }


# ------------------------------------------------------------------ 4-Class Ridge Prefill Router
def fit_ridge_router(
    X: torch.Tensor,
    labels: list[str],
    classes: tuple[str, ...] = ROUTER_CLASSES,
    *,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form class-balanced L2-regularized Ridge router on normalized hidden states."""
    X_f = X.detach().double()
    N, D = X_f.shape
    C = len(classes)
    cls_to_i = {c: i for i, c in enumerate(classes)}
    Y = torch.zeros((N, C), dtype=torch.float64)
    counts = {c: max(1, sum(1 for lb in labels if lb == c)) for c in classes}
    w_vec = torch.zeros(N, dtype=torch.float64)
    for row_i, lb in enumerate(labels):
        Y[row_i, cls_to_i[lb]] = 1.0
        w_vec[row_i] = float(N) / float(C * counts[lb])

    X_aug = torch.cat([X_f, torch.ones((N, 1), dtype=torch.float64)], dim=1)
    Xw = X_aug * w_vec[:, None]
    reg = torch.eye(D + 1, dtype=torch.float64) * float(lam)
    reg[-1, -1] = 1e-6  # keep bias unpenalized
    gram = Xw.t() @ X_aug + reg
    rhs = Xw.t() @ Y
    coef = torch.linalg.solve(gram, rhs)
    return coef[:-1].float(), coef[-1].float()


def predict_ridge_router(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, classes: tuple[str, ...] = ROUTER_CLASSES) -> list[str]:
    scores = X.detach().float() @ W + b[None, :]
    preds = scores.argmax(dim=1).tolist()
    return [classes[int(i)] for i in preds]


# ------------------------------------------------------------------ Full Checkpoint Baker & vLLM Runner
def _bake_full_checkpoint(base_dir: Path, param_updates: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Create a full patched checkpoint directory matching safetensors keys directly or by normalized layer suffix."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in base_dir.iterdir():
        if entry.name.startswith(".") or entry.name.endswith(".lock") or entry.suffix == ".safetensors":
            continue
        dst = out_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst, ignore=shutil.ignore_patterns(".cache", "*.lock"))
        else:
            try:
                os.symlink(entry.resolve(), dst)
            except OSError:
                shutil.copy2(entry, dst)

    by_exact = dict(param_updates)
    by_norm: dict[str, tuple[str, torch.Tensor]] = {}
    for k, v in param_updates.items():
        nk = _norm_param_key(k)
        if nk in by_norm:
            raise RuntimeError(f"Normalized parameter key collision: {nk!r} ({by_norm[nk][0]} vs {k})")
        by_norm[nk] = (k, v)
    matched_orig: set[str] = set()

    for sf in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(sf), "pt") as fh:
            keys = list(fh.keys())
            meta = fh.metadata() or {"format": "pt"}
            file_hits: dict[str, tuple[str, torch.Tensor]] = {}
            for k in keys:
                if "mtp" in k or "visual" in k:
                    continue
                if k in by_exact:
                    file_hits[k] = (k, by_exact[k])
                else:
                    nk = _norm_param_key(k)
                    if nk in by_norm:
                        orig_k, tens = by_norm[nk]
                        file_hits[k] = (orig_k, tens)
            if not file_hits:
                dst_sf = out_dir / sf.name
                try:
                    os.symlink(sf.resolve(), dst_sf)
                except OSError:
                    shutil.copy2(sf, dst_sf)
                continue
            tensors = {k: fh.get_tensor(k).clone() for k in keys}
        for k, (orig_k, tens) in file_hits.items():
            if tuple(tens.shape) != tuple(tensors[k].shape):
                raise RuntimeError(f"Baked shape mismatch on {k}: got {tuple(tens.shape)} vs {tuple(tensors[k].shape)}")
            tensors[k] = tens.detach().to(tensors[k].dtype).cpu()
            matched_orig.add(orig_k)
        save_file(tensors, str(out_dir / sf.name), metadata=meta)

    missing = set(param_updates) - matched_orig
    if missing:
        raise RuntimeError(f"_bake_full_checkpoint missed {len(missing)} parameters: {sorted(missing)[:6]}")
    return out_dir


def _vllm_generate(
    model_dir: str,
    tok: Any,
    prompts_ids: list[list[int]],
    seeds: list[int],
    *,
    max_new_tokens: int,
    dtype_str: str,
    base_seed: int,
) -> list[dict[str, Any]]:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    py_parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if not py_parts or py_parts[0] != str(HERE):
        os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE)] + [p for p in py_parts if p != str(HERE)])

    import loop_kill_v2
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    eos_list = lr.eos_ids(tok)
    kill_eos = eos_list[-1]
    max_len = max(len(p) for p in prompts_ids) + int(max_new_tokens) + 64
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        gpu_mem = round(min(0.82, max(0.55, (free_b / max(1, total_b)) - 0.08)), 3)
    else:
        gpu_mem = 0.80

    base_kw: dict[str, Any] = dict(
        model=str(model_dir),
        tokenizer=str(tok.name_or_path),
        dtype=dtype_str,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem,
        enable_prefix_caching=True,
        enforce_eager=True,
        seed=int(base_seed),
        trust_remote_code=True,
    )
    extras: dict[str, Any] = {
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "enable_chunked_prefill": True,
        "max_num_seqs": 64,
    }
    proc = loop_kill_v2.processor()
    if proc is not None:
        extras["logits_processors"] = [proc]

    drop_order = [
        [],
        ["max_num_seqs"],
        ["max_num_seqs", "enable_chunked_prefill"],
        ["max_num_seqs", "enable_chunked_prefill", "limit_mm_per_prompt"],
        ["max_num_seqs", "enable_chunked_prefill", "logits_processors"],
        ["max_num_seqs", "enable_chunked_prefill", "limit_mm_per_prompt", "logits_processors"],
    ]

    llm = None
    kill = False
    last_err: Exception | None = None
    t_boot = time.perf_counter()
    print(
        f"[vllm-eager] launching {Path(model_dir).name} "
        f"(enforce_eager=True, kv_cache=auto, gpu_mem={gpu_mem}, max_len={max_len})...",
        flush=True,
    )
    for drop in drop_order:
        kw = {**base_kw, **{k: v for k, v in extras.items() if k not in drop}}
        try:
            llm = LLM(**kw)
            kill = "logits_processors" in kw
            print(
                f"[vllm-eager] booted in {time.perf_counter() - t_boot:.1f}s "
                f"(gpu_mem={gpu_mem}, chunked={'enable_chunked_prefill' in kw}, "
                f"max_seqs={kw.get('max_num_seqs', 'default')}, early_loop_stop={kill})",
                flush=True,
            )
            break
        except Exception as err:  # noqa: BLE001
            last_err = err
            print(
                f"[vllm-eager] refused {sorted(set(extras) - set(drop))}: {type(err).__name__}: {str(err)[:220]}",
                flush=True,
            )
            gc.collect()
    if llm is None:
        assert last_err is not None
        raise last_err

    try:
        sps = [
            SamplingParams(
                **lr.SAMPLING,
                max_tokens=int(max_new_tokens),
                seed=int(s),
                stop_token_ids=eos_list,
                extra_args={"loop_kill_eos": kill_eos} if kill else None,
            )
            for s in seeds
        ]
        tok_prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts_ids]
        t_gen = time.perf_counter()
        try:
            outs = llm.generate(tok_prompts, sps, use_tqdm=False)
        except Exception as err:  # noqa: BLE001
            if not kill:
                raise
            print(f"[vllm-eager] early loop stop failed ({type(err).__name__}); retrying without it", flush=True)
            sps = [
                SamplingParams(**lr.SAMPLING, max_tokens=int(max_new_tokens), seed=int(s), stop_token_ids=eos_list)
                for s in seeds
            ]
            outs = llm.generate(tok_prompts, sps, use_tqdm=False)
        print(f"[vllm-eager] generated {len(outs)} prompts in {time.perf_counter() - t_gen:.1f}s", flush=True)
        return [{"token_ids": list(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason} for o in outs]
    finally:
        try:
            eng = getattr(llm, "llm_engine", None)
            ec = getattr(eng, "engine_core", None)
            if ec is not None and hasattr(ec, "shutdown"):
                ec.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:  # noqa: BLE001
            pass
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def _hf_generate_subset(
    model: nn.Module,
    tok: Any,
    prompts_ids: list[list[int]],
    seeds: list[int],
    *,
    max_new_tokens: int,
    param_updates: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    dev = next(model.parameters()).device
    eos_list = lr.eos_ids(tok)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    saved_cpu: list[tuple[nn.Parameter, torch.Tensor]] = []
    if param_updates:
        p_map = dict(model.named_parameters())
        norm_map = {_norm_param_key(k): v for k, v in p_map.items()}
        for k, val in param_updates.items():
            p = p_map.get(k) or norm_map[_norm_param_key(k)]
            saved_cpu.append((p, p.detach().cpu().clone()))
            p.data.copy_(val.to(device=p.device, dtype=p.dtype))
    try:
        out: list[dict[str, Any]] = []
        for pids, s_val in zip(prompts_ids, seeds):
            ids_t = torch.tensor(pids, dtype=torch.long, device=dev).unsqueeze(0)
            torch.manual_seed(int(s_val))
            gen = model.generate(
                input_ids=ids_t,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=lr.SAMPLING["temperature"],
                top_p=lr.SAMPLING["top_p"],
                top_k=lr.SAMPLING["top_k"],
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=eos_list,
            )
            gen_ids = [int(x) for x in gen[0, len(pids) :].tolist()]
            body = lr.strip_eos(gen_ids, set(eos_list))
            fin = "length" if (len(gen_ids) >= max_new_tokens and len(body) == len(gen_ids)) else "stop"
            out.append({"token_ids": gen_ids, "finish_reason": fin})
        return out
    finally:
        for p, orig_cpu in saved_cpu:
            p.data.copy_(orig_cpu.to(device=p.device, dtype=p.dtype))


def _grade_rows(
    items: list[dict[str, Any]],
    raw_outs: list[dict[str, Any]],
    tok: Any,
    eos_set: set[int],
    think_end_id: int | None,
    *,
    routed_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (it, raw) in enumerate(zip(items, raw_outs)):
        body = lr.strip_eos(raw["token_ids"], eos_set)
        txt = tok.decode(body, skip_special_tokens=False)
        g = grade_response(
            raw["token_ids"],
            txt,
            int(it["answer"]),
            thinking_enabled=False,
            finish_reason=raw["finish_reason"],
            eos_set=eos_set,
            tok=tok,
            think_end_id=think_end_id,
        )
        rows.append({
            "id": it["id"],
            "family": it["family"],
            "variant": it.get("variant", "guardrail"),
            "routed_class": routed_labels[i] if routed_labels is not None else None,
            **g,
        })
    return rows


def _summarize_ceiling_arm(rows: list[dict[str, Any]], *, hard_families: tuple[str, ...], guard_families: tuple[str, ...]) -> dict[str, Any]:
    hard = [r for r in rows if r["family"] in hard_families]
    guard = [r for r in rows if r["family"] in guard_families]
    renum = [r for r in hard if r["variant"] == "renumbered"]
    rephr = [r for r in hard if r["variant"] == "rephrased_renumbered"]
    h_toks = [r["total_tokens"] for r in hard]

    per_fam_hard = {
        f: round(sum(1 for r in hard if r["family"] == f and r["strict_correct"]) / max(1, sum(1 for r in hard if r["family"] == f)), 4)
        for f in hard_families
    }
    per_fam_guard = {
        f: round(sum(1 for r in guard if r["family"] == f and r["strict_correct"]) / max(1, sum(1 for r in guard if r["family"] == f)), 4)
        for f in guard_families
    }

    return {
        "hard_36": {
            "strict_accuracy": round(sum(1 for r in hard if r["strict_correct"]) / max(1, len(hard)), 4),
            "renumbered_acc": round(sum(1 for r in renum if r["strict_correct"]) / max(1, len(renum)), 4),
            "rephrased_acc": round(sum(1 for r in rephr if r["strict_correct"]) / max(1, len(rephr)), 4),
            "n_correct": sum(1 for r in hard if r["strict_correct"]),
            "n_total": len(hard),
            "mean_tokens": round(sum(h_toks) / max(1, len(hard)), 1),
            "median_tokens": round(float(statistics.median(h_toks)), 1) if h_toks else 0.0,
            "truncated_rate": round(sum(1 for r in hard if r["truncated"]) / max(1, len(hard)), 4),
            "per_family_strict": per_fam_hard,
        },
        "guardrail_24": {
            "strict_accuracy": round(sum(1 for r in guard if r["strict_correct"]) / max(1, len(guard)), 4),
            "n_correct": sum(1 for r in guard if r["strict_correct"]),
            "n_total": len(guard),
            "per_family_strict": per_fam_guard,
        },
    }


def _extract_causal_layers(diag: dict[str, Any], total_layers: int, max_layers: int = 8) -> list[int]:
    out: list[int] = []
    for mk in diag["causal_rank"]:
        m = re.search(r"layers\.(\d+)\.", mk)
        if m:
            li = int(m.group(1))
            if li not in out:
                out.append(li)
                if len(out) >= max_layers:
                    return out
    for li in range(total_layers):
        if li not in out:
            out.append(li)
            if len(out) >= max_layers:
                break
    return out


# ------------------------------------------------------------------ CPU Smoke Test
def run_smoke() -> dict[str, Any]:
    ds = build_ceiling_dataset()
    assert len(ds["hard_train"]) == 144, len(ds["hard_train"])
    assert len(ds["hard_test"]) == 36, len(ds["hard_test"])
    assert len(ds["guardrail_test"]) == 24, len(ds["guardrail_test"])
    assert len(ds["prose_heldout"]) == 12 and len(ds["prose_router_train"]) == 8

    # Check scenario disjointness & exact answer parsing for every single train & test item
    for fam in HARD_FAMILIES:
        tr_sc = {it["scenario"] for it in ds["hard_train"] if it["family"] == fam}
        te_rep_sc = {it["scenario"] for it in ds["hard_test"] if it["family"] == fam and it["variant"] == "rephrased_renumbered"}
        assert tr_sc.isdisjoint(te_rep_sc), (fam, tr_sc & te_rep_sc)
    for it in ds["hard_train"] + ds["hard_test"]:
        g = grade_response([1, 2, 3], it["fallback_response"], int(it["answer"]), thinking_enabled=False, finish_reason="stop")
        assert g["strict_correct"] and g["has_boxed"], (it["id"], g)

    # Toy model KL+CE training + LoRA + FP32 master-weight full-layer restoration + Ridge router + Checkpoint baking
    torch.manual_seed(7)
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    seqs = [
        [1, 2, 4, 10, 20, 20, 20, 20],
        [2, 1, 4, 11, 20, 20, 20, 20],
        [1, 3, 5, 14, 21, 21, 21, 21],
        [3, 1, 5, 15, 21, 21, 21, 21],
    ]
    train_samples = []
    for s in seqs:
        ids_t = torch.tensor(s, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            t_log = model(ids_t).logits[0, 3:7].float()
        vals, tk_ids = t_log.topk(8, dim=-1)
        train_samples.append({
            "ids": s,
            "lo": 3,
            "hi": 7,
            "topk_ids": tk_ids.to(torch.int32).cpu(),
            "topk_logits": vals.to(torch.float16).cpu(),
        })

    h_ids = [torch.tensor([1, 2, 3, 1], dtype=torch.long).unsqueeze(0)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    for rk in (2, 4):
        res_lora = _train_lora_expert(
            model, [1, 2], train_samples, rank=rk, alpha_scale=float(rk * 2), epochs=3, lr_val=2e-2, grad_accum=2,
            healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll, seed=42,
        )
        assert res_lora["epoch_losses"][-1] < res_lora["epoch_losses"][0], res_lora["epoch_losses"]
        assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    res_full = _train_full_layers_expert(
        model, [1, 2], train_samples, epochs=3, lr_val=1e-2, grad_accum=2,
        healthy_ids=h_ids, base_healthy_logp=base_lp, base_healthy_nll=base_nll, seed=42,
    )
    assert res_full["epoch_losses"][-1] < res_full["epoch_losses"][0], res_full["epoch_losses"]
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    # Ridge 4-class router sanity check
    X_r = torch.randn(24, 32)
    y_r = [ROUTER_CLASSES[i % 4] for i in range(24)]
    for i, c in enumerate(y_r):
        X_r[i, ROUTER_CLASSES.index(c)] += 4.0
    X_r = X_r / X_r.norm(dim=1, keepdim=True)
    W_r, b_r = fit_ridge_router(X_r, y_r)
    preds_r = predict_ridge_router(X_r, W_r, b_r)
    assert sum(a == b for a, b in zip(preds_r, y_r)) == 24

    # Checkpoint baking & exact reload check via safetensors
    from safetensors import safe_open
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base_d = td_p / "base"
        bake_d = td_p / "baked"
        base_d.mkdir(parents=True)
        save_file({k: v.clone() for k, v in orig_snapshot.items()}, str(base_d / "model.safetensors"))
        (base_d / "config.json").write_text('{"toy": true}', encoding="utf-8")

        _bake_full_checkpoint(base_d, res_full["weights"], bake_d)
        assert (bake_d / "config.json").exists()
        with safe_open(str(bake_d / "model.safetensors"), "pt") as fh:
            for k, val in res_full["weights"].items():
                assert torch.equal(fh.get_tensor(k), val.to(orig_snapshot[k].dtype))

    return {
        "mode": "smoke",
        "passed": True,
        "dataset_counts": {k: len(ds[k]) for k in ("hard_train", "hard_test", "guardrail_test", "prose_heldout", "prose_router_train")},
        "lora_r2_losses": res_lora["epoch_losses"],
        "full_layers_losses": res_full["epoch_losses"],
        "router_train_acc": 1.0,
        "bake_exact_reload_ok": True,
    }


# ------------------------------------------------------------------ Full GPU Ceiling Pipeline
def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        args.dtype, torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float32
    )
    dtype_vllm_str = "bfloat16" if dtype == torch.bfloat16 else ("float16" if dtype == torch.float16 else "float32")
    backend = _select_backend(args.backend)

    ds = build_ceiling_dataset()
    hard_train = ds["hard_train"]
    hard_test = ds["hard_test"]
    guardrail_test = ds["guardrail_test"]
    all_test = hard_test + guardrail_test

    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    assert len(tok_t) == len(tok_s), f"Tokenizer vocab mismatch: teacher={len(tok_t)} vs student={len(tok_s)}"
    eos_set = set(lr.eos_ids(tok_s))
    think_end_id = tok_s.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_id, int) or think_end_id == tok_s.unk_token_id:
        think_end_id = None

    # ---- Phase 1: Teacher 4B generation + Top-32 logit extraction ----
    print(f"[ceiling] Phase 1: Teacher ({args.teacher}) generation & Top-32 logit extraction...", flush=True)
    t_p1 = time.perf_counter()
    test_prompts_ids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in all_test
    ]
    train_prompts_ids = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in hard_train
    ]
    test_seeds = [int(args.seed) + i for i in range(len(all_test))]
    train_seeds = [int(args.seed) + 1000 + i for i in range(len(hard_train))]

    if backend == "vllm":
        t_gen_outs = _vllm_generate(
            args.teacher,
            tok_t,
            test_prompts_ids + train_prompts_ids,
            test_seeds + train_seeds,
            max_new_tokens=args.max_new_tokens,
            dtype_str=dtype_vllm_str,
            base_seed=args.seed,
        )
        t_test_raw = t_gen_outs[: len(all_test)]
        t_train_raw = t_gen_outs[len(all_test) :]
    else:
        try:
            teacher_m = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_m = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
        t_test_raw = _hf_generate_subset(teacher_m, tok_t, test_prompts_ids, test_seeds, max_new_tokens=args.max_new_tokens)
        t_train_raw = _hf_generate_subset(teacher_m, tok_t, train_prompts_ids, train_seeds, max_new_tokens=args.max_new_tokens)
        del teacher_m
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    teacher_test_rows = _grade_rows(all_test, t_test_raw, tok_s, eos_set, think_end_id)
    teacher_4b_summary = _summarize_ceiling_arm(teacher_test_rows, hard_families=HARD_FAMILIES, guard_families=GUARDRAIL_FAMILIES)
    print(f"[ceiling] teacher_4b: hard={teacher_4b_summary['hard_36']} | guard={teacher_4b_summary['guardrail_24']}", flush=True)

    # Select crisp teacher responses for distillation (4B output if strict & <=700 tokens, else programmatic fallback)
    kept_4b = {f: 0 for f in HARD_FAMILIES}
    chosen_responses: list[str] = []
    for it, raw in zip(hard_train, t_train_raw):
        body = lr.strip_eos(raw["token_ids"], eos_set)
        txt = tok_s.decode(body, skip_special_tokens=True).strip()
        g = grade_response(raw["token_ids"], txt, int(it["answer"]), thinking_enabled=False, finish_reason=raw["finish_reason"], eos_set=eos_set, tok=tok_s)
        if g["strict_correct"] and g["has_boxed"] and g["total_tokens"] <= 700:
            chosen_responses.append(txt)
            kept_4b[it["family"]] += 1
        else:
            chosen_responses.append(it["fallback_response"])
    teacher_4b_train_kept_frac = {f: round(kept_4b[f] / 48.0, 4) for f in HARD_FAMILIES}
    print(f"[ceiling] teacher_4b train responses kept fraction: {teacher_4b_train_kept_frac}", flush=True)

    # Extract Top-32 teacher logits with HF teacher model (no_grad)
    try:
        teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    train_samples_by_fam: dict[str, list[dict[str, Any]]] = {f: [] for f in HARD_FAMILIES}
    with torch.no_grad():
        for it, p_ids, resp_txt in zip(hard_train, train_prompts_ids, chosen_responses):
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
            train_samples_by_fam[it["family"]].append({
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

    # ---- Phase 2: Load Student 0.8B, WSI Causal Localization & 4-Class Layer-12 Router ----
    print(f"[ceiling] Phase 2: Loading Student ({args.student}), WSI causal localization & 4-class router...", flush=True)
    t_p2 = time.perf_counter()
    try:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in student_m.parameters():
        p.requires_grad_(False)

    _, blocks = find_blocks(student_m)
    n_layers = len(blocks)
    causal_traces_by_fam: dict[str, list[CompactTrace]] = {f: [] for f in HARD_FAMILIES}
    for f in HARD_FAMILIES:
        # Pick 4 distinct scenarios: indices 0, 1, 2, 3 cover 4 different scenarios
        for sp in train_samples_by_fam[f][:4]:
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
            causal_traces_by_fam[f].append(ct)

    invariance = compute_invariance_metrics(causal_traces_by_fam, top_k_sites=4, top_n_neurons=32)
    causal_layers: dict[str, list[int]] = {}
    for f in HARD_FAMILIES:
        diag_f = diagnose_causal_traces(causal_traces_by_fam[f], split_ab=True, top_k_sites=16, center="healthy", healthy_cost="fisher", model_or_weights=student_m)
        causal_layers[f] = _extract_causal_layers(diag_f, n_layers, max_layers=8)

    all_12_traces = [ct for f in HARD_FAMILIES for ct in causal_traces_by_fam[f]]
    diag_glob = diagnose_causal_traces(all_12_traces, split_ab=True, top_k_sites=16, center="healthy", healthy_cost="fisher", model_or_weights=student_m)
    global_causal_layers = _extract_causal_layers(diag_glob, n_layers, max_layers=4)
    print(f"[ceiling] Causal layers: {causal_layers} | global={global_causal_layers}", flush=True)

    # 4-Class Ridge Prefill Router at middle layer
    router_layer = n_layers // 2
    guard_r_prompts = [
        _format_messages(tok_s, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in ds["guardrail_router_train"]
    ]
    prose_r_prompts = [
        _format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
        for p in ds["prose_router_train"]
    ]
    prose_held_prompts = [
        _format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
        for p in ds["prose_heldout"]
    ]

    r_train_prompts = train_prompts_ids + guard_r_prompts + prose_r_prompts
    r_train_labels = [it["family"] for it in hard_train] + ["passthrough"] * (len(guard_r_prompts) + len(prose_r_prompts))

    X_r_train = _extract_prompt_hidden_vecs(student_m, r_train_prompts, router_layer)
    W_r, b_r = fit_ridge_router(X_r_train, r_train_labels, ROUTER_CLASSES, lam=1.0)

    X_test_vecs = _extract_prompt_hidden_vecs(student_m, test_prompts_ids, router_layer)
    X_prose_vecs = _extract_prompt_hidden_vecs(student_m, prose_held_prompts, router_layer)
    routed_test_classes = predict_ridge_router(X_test_vecs, W_r, b_r, ROUTER_CLASSES)
    routed_prose_classes = predict_ridge_router(X_prose_vecs, W_r, b_r, ROUTER_CLASSES)

    hard_r_preds = routed_test_classes[: len(hard_test)]
    guard_r_preds = routed_test_classes[len(hard_test) :]
    router_report = {
        "router_layer": router_layer,
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
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_classes) / len(routed_prose_classes), 4),
    }
    print(f"[ceiling] Router report: {router_report}", flush=True)
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # Precompute base healthy prose log-probs for KL drift measurements
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

    # ---- Phase 3: Train 4 Pocket MoE Tiers (micro, large, full_layers, global_full_layers) ----
    print("[ceiling] Phase 3: Training Pocket MoE tiers (micro -> large -> full_layers -> global_full_layers)...", flush=True)
    t_p3 = time.perf_counter()

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        expert_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {"moe_micro": {}, "moe_large": {}, "moe_full_layers": {}, "global_full_layers": {}}

        for f_idx, f in enumerate(HARD_FAMILIES):
            samples_f = train_samples_by_fam[f]

            res_micro = _train_lora_expert(
                student_m, causal_layers[f][:4], samples_f, rank=8, alpha_scale=16.0, epochs=3, lr_val=5e-4, grad_accum=4,
                healthy_ids=prose_full_ids, base_healthy_logp=base_prose_logp, base_healthy_nll=base_prose_nll, seed=args.seed + f_idx,
            )
            p_micro = exp_dir / f"micro_{f}.pt"
            torch.save(res_micro.pop("weights"), p_micro)
            expert_paths[("moe_micro", f)] = p_micro
            expert_meta["moe_micro"][f] = res_micro
            print(f"[ceiling] trained moe_micro/{f}: {res_micro['adapter_mb']}MB | losses={res_micro['epoch_losses']} | unrouted_kl={res_micro['healthy_kl']}", flush=True)

            res_large = _train_lora_expert(
                student_m, causal_layers[f][:8], samples_f, rank=64, alpha_scale=64.0, epochs=3, lr_val=3e-4, grad_accum=4,
                healthy_ids=prose_full_ids, base_healthy_logp=base_prose_logp, base_healthy_nll=base_prose_nll, seed=args.seed + 10 + f_idx,
            )
            p_large = exp_dir / f"large_{f}.pt"
            torch.save(res_large.pop("weights"), p_large)
            expert_paths[("moe_large", f)] = p_large
            expert_meta["moe_large"][f] = res_large
            print(f"[ceiling] trained moe_large/{f}: {res_large['adapter_mb']}MB | losses={res_large['epoch_losses']} | unrouted_kl={res_large['healthy_kl']}", flush=True)

            res_full = _train_full_layers_expert(
                student_m, causal_layers[f][:4], samples_f, epochs=3, lr_val=1.5e-4, grad_accum=4,
                healthy_ids=prose_full_ids, base_healthy_logp=base_prose_logp, base_healthy_nll=base_prose_nll, seed=args.seed + 20 + f_idx,
            )
            p_full = exp_dir / f"full_{f}.pt"
            torch.save(res_full.pop("weights"), p_full)
            expert_paths[("moe_full_layers", f)] = p_full
            expert_meta["moe_full_layers"][f] = res_full
            print(f"[ceiling] trained moe_full_layers/{f}: {res_full['adapter_mb']}MB | losses={res_full['epoch_losses']} | unrouted_kl={res_full['healthy_kl']}", flush=True)

        all_samples_flat = [sp for f in HARD_FAMILIES for sp in train_samples_by_fam[f]]
        res_glob = _train_full_layers_expert(
            student_m, global_causal_layers[:4], all_samples_flat, epochs=3, lr_val=1.5e-4, grad_accum=8,
            healthy_ids=prose_full_ids, base_healthy_logp=base_prose_logp, base_healthy_nll=base_prose_nll, seed=args.seed + 99,
        )
        p_glob = exp_dir / "global_full.pt"
        torch.save(res_glob.pop("weights"), p_glob)
        expert_meta["global_full_layers"] = res_glob
        print(f"[ceiling] trained global_full_layers: {res_glob['adapter_mb']}MB | losses={res_glob['epoch_losses']} | unrouted_kl={res_glob['healthy_kl']}", flush=True)
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Exact Baked vLLM Evaluation across all 0.8B arms ----
        print(f"[ceiling] Phase 4: Evaluating 0.8B arms ({backend}, max_new_tokens={args.max_new_tokens})...", flush=True)
        t_p4 = time.perf_counter()

        # Prepare 1-shot prompts (hard items get train[0] of their true family; guardrails get their own train[0])
        train_ex_by_fam: dict[str, tuple[str, str]] = {}
        for f in HARD_FAMILIES:
            idx0 = next(i for i, it in enumerate(hard_train) if it["family"] == f)
            train_ex_by_fam[f] = (hard_train[idx0]["question"], chosen_responses[idx0])
        base_ds_ref = build_reasoning_dataset()
        for f in GUARDRAIL_FAMILIES:
            ex_g = next(it for it in base_ds_ref["train"] if it["family"] == f)
            ans_g = ex_g["answer"]
            clean_resp = ex_g["teacher_response"].split("</think>")[-1].strip()
            train_ex_by_fam[f] = (ex_g["question"], clean_resp or f"The answer is \\boxed{{{ans_g}}}.")

        oracle_prompts_ids = []
        for it in all_test:
            q_ex, a_ex = train_ex_by_fam[it["family"]]
            msgs = [
                {"role": "user", "content": format_question(q_ex)},
                {"role": "assistant", "content": a_ex},
                {"role": "user", "content": format_question(it["question"])},
            ]
            oracle_prompts_ids.append(_format_messages(tok_s, msgs, add_generation_prompt=True, enable_thinking=False))

        if backend == "vllm":
            del student_m
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            def _eval_model(m_dir: str, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
                return _vllm_generate(m_dir, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
        else:
            def _eval_model(m_dir: str, p_ids: list[list[int]], sd_list: list[int], upd: dict[str, torch.Tensor] | None = None) -> list[dict[str, Any]]:
                return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, param_updates=upd)

        # Base + Oracle 1-shot in a single unpatched vLLM launch
        if backend == "vllm":
            bo_outs = _eval_model(args.student, test_prompts_ids + oracle_prompts_ids, test_seeds + test_seeds)
            base_raw = bo_outs[: len(all_test)]
            oracle_raw = bo_outs[len(all_test) :]
        else:
            base_raw = _hf_generate_subset(student_m, tok_s, test_prompts_ids, test_seeds, max_new_tokens=args.max_new_tokens)
            oracle_raw = _hf_generate_subset(student_m, tok_s, oracle_prompts_ids, test_seeds, max_new_tokens=args.max_new_tokens)

        bake_tmp_dir = Path("/content/_pocket_moe_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

        def _eval_ckpt(weights_path: Path, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
            upd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if backend == "vllm":
                _bake_full_checkpoint(Path(args.student), upd, bake_tmp_dir)
                del upd
                gc.collect()
                return _vllm_generate(str(bake_tmp_dir), tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
            return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, param_updates=upd)

        glob_raw = _eval_ckpt(p_glob, test_prompts_ids, test_seeds)

        routed_raw_by_tier: dict[str, list[dict[str, Any]]] = {}
        for tier in ("moe_micro", "moe_large", "moe_full_layers"):
            tier_raw: list[dict[str, Any] | None] = [None] * len(all_test)
            # Reuse base_raw bit-for-bit on items routed to passthrough
            for i, cls_nm in enumerate(routed_test_classes):
                if cls_nm == "passthrough":
                    tier_raw[i] = base_raw[i]
            for f in HARD_FAMILIES:
                idxs = [i for i, cls_nm in enumerate(routed_test_classes) if cls_nm == f]
                if not idxs:
                    continue
                sub_outs = _eval_ckpt(expert_paths[(tier, f)], [test_prompts_ids[i] for i in idxs], [test_seeds[i] for i in idxs])
                for idx, o in zip(idxs, sub_outs):
                    tier_raw[idx] = o
            routed_raw_by_tier[tier] = [r for r in tier_raw if r is not None]

        if bake_tmp_dir.exists():
            shutil.rmtree(bake_tmp_dir, ignore_errors=True)
        phase4_s = round(time.perf_counter() - t_p4, 2)

    arm_raw_map = {
        "base_08b": (base_raw, None),
        "teacher_4b": (t_test_raw, None),
        "oracle_1shot_08b": (oracle_raw, None),
        "moe_micro_routed": (routed_raw_by_tier["moe_micro"], routed_test_classes),
        "moe_large_routed": (routed_raw_by_tier["moe_large"], routed_test_classes),
        "moe_full_layers_routed": (routed_raw_by_tier["moe_full_layers"], routed_test_classes),
        "global_full_layers": (glob_raw, None),
    }

    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_nm, (raw_l, r_lbls) in arm_raw_map.items():
        if arm_nm == "teacher_4b":
            rows = teacher_test_rows
        else:
            rows = _grade_rows(all_test, raw_l, tok_s, eos_set, think_end_id, routed_labels=r_lbls)
        arm_rows[arm_nm] = rows
        sm = _summarize_ceiling_arm(rows, hard_families=HARD_FAMILIES, guard_families=GUARDRAIL_FAMILIES)
        arm_summaries[arm_nm] = sm
        print(f"[ceiling] {arm_nm:22s} | hard={sm['hard_36']} | guard={sm['guardrail_24']}", flush=True)

    # Routed prose KL = unrouted expert KL only on false-fired prose controls
    routed_prose_kl: dict[str, float] = {}
    for tier in ("moe_micro", "moe_large", "moe_full_layers"):
        kl_tot = 0.0
        for p_cls in routed_prose_classes:
            if p_cls in HARD_FAMILIES:
                kl_tot += float(expert_meta[tier][p_cls]["healthy_kl"])
        routed_prose_kl[tier] = round(kl_tot / max(1, len(routed_prose_classes)), 6)

    acc_base = arm_summaries["base_08b"]["hard_36"]["strict_accuracy"]
    acc_teacher = arm_summaries["teacher_4b"]["hard_36"]["strict_accuracy"]
    acc_oracle = arm_summaries["oracle_1shot_08b"]["hard_36"]["strict_accuracy"]
    acc_micro = arm_summaries["moe_micro_routed"]["hard_36"]["strict_accuracy"]
    acc_large = arm_summaries["moe_large_routed"]["hard_36"]["strict_accuracy"]
    acc_full = arm_summaries["moe_full_layers_routed"]["hard_36"]["strict_accuracy"]
    acc_glob = arm_summaries["global_full_layers"]["hard_36"]["strict_accuracy"]

    def _gap_rec(a: float) -> float | None:
        return round((a - acc_base) / max(acc_teacher - acc_base, 1e-6), 4) if acc_teacher > acc_base else None

    return {
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "backend": backend,
        "max_new_tokens": int(args.max_new_tokens),
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_wsi_router_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "teacher_4b_train_kept_frac": teacher_4b_train_kept_frac,
        "invariance": invariance,
        "causal_layers": {**causal_layers, "global": global_causal_layers},
        "router_report": router_report,
        "expert_meta": expert_meta,
        "routed_prose_kl": routed_prose_kl,
        "generation_arms": arm_summaries,
        "scaling_curve": {
            "base_08b_hard_acc": acc_base,
            "oracle_1shot_08b_hard_acc": acc_oracle,
            "moe_micro_hard_acc": acc_micro,
            "moe_large_hard_acc": acc_large,
            "moe_full_layers_hard_acc": acc_full,
            "global_full_layers_hard_acc": acc_glob,
            "teacher_4b_hard_acc": acc_teacher,
            "teacher_gap_recovered": {
                "moe_micro": _gap_rec(acc_micro),
                "moe_large": _gap_rec(acc_large),
                "moe_full_layers": _gap_rec(acc_full),
                "global_full_layers": _gap_rec(acc_glob),
            },
        },
        "paired_comparisons_hard_36": {
            "moe_micro_vs_base": _paired_vs(arm_rows["moe_micro_routed"][:36], arm_rows["base_08b"][:36]),
            "moe_large_vs_base": _paired_vs(arm_rows["moe_large_routed"][:36], arm_rows["base_08b"][:36]),
            "moe_full_vs_base": _paired_vs(arm_rows["moe_full_layers_routed"][:36], arm_rows["base_08b"][:36]),
            "moe_full_vs_global_full": _paired_vs(arm_rows["moe_full_layers_routed"][:36], arm_rows["global_full_layers"][:36]),
        },
        "criteria": {
            "a_positive_capacity_curve": bool(max(acc_micro, acc_large, acc_full) > acc_base + 0.05),
            "b_moe_full_beats_base_and_keeps_guardrails": bool(
                acc_full > acc_base and arm_summaries["moe_full_layers_routed"]["guardrail_24"]["strict_accuracy"] >= arm_summaries["base_08b"]["guardrail_24"]["strict_accuracy"] - 0.05
            ),
            "c_routed_beats_global_same_size": bool(acc_full >= acc_glob),
        },
        "per_item_generation": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU check")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
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
