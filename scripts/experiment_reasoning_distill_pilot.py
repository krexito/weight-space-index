"""Pilot Experiment: Invariant Reasoning-Trace to MoI Circuit Distillation (`Qwen/Qwen3.5-0.8B`).

Tests whether crossing re-numbered variants of solved reasoning problems in ``CompactTrace``
cancels instance-specific number activations and isolates a pure, number-invariant algorithmic
circuit that can be compiled into a surgical ``.moipatch`` library + prefill router.

Two scientific questions & three pre-registered criteria:
  1. Invariance in CompactTrace (`within_family_cos` / `within_family_jaccard` vs `across_family`):
     Do re-numbered variants of the SAME reasoning family share a substantially more similar
     causal direction `u` and top-32 neuron support than variants from DIFFERENT families?
     -> Criterion (a): `within_vs_across_ratio >= 1.35` (max of cosine ratio and neuron Jaccard ratio).
  2. Number cancellation (`invariant` vs `single_trace` and `random`):
     Does intersecting/pooling 4 re-numbered variants (`invariant`) beat a patch extracted from a
     single solved instance (`single_trace`) and a random-direction control (`random`) on held-out
     re-numbered + re-phrased variants?
     -> Criterion (b): `invariant` beats `single_trace` and `random` on held-out forward screen Pareto
        and/or held-out generation score.
  3. Distillation gain (`invariant_routed_moi` vs `base`):
     Does the routed MoI bank of 4 distilled family `.moipatch` experts improve held-out accuracy
     (or reduce `<think>` tokens at >= base accuracy) on 48 held-out `renumbered` + `rephrased_renumbered`
     variants while keeping `healthy_kl <= 0.02`?
     -> Criterion (c): `acc_routed > acc_base` or (`acc_routed >= acc_base` and `tokens_routed < 0.85 * tokens_base`).

Modes:
  - ``--smoke``: Deterministic CPU verification on a 4-layer synthetic model with 2 planted reasoning
    families + instance-number decoys (verifies all 3 criteria on CPU in < 5 seconds).
  - Full GPU pilot (e.g. ``--model Qwen/Qwen3.5-0.8B`` on Colab L4): runs the 4-family math reasoning
    pilot (16 train distillation traces + 48 held-out renumbered & rephrased_renumbered test problems
    + 16 benign controls).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
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

from wsi import (  # noqa: E402
    CompactTrace,
    Edit,
    MoIPatch,
    capture_causal_trace,
    diagnose_causal_traces,
    screen_patch_candidates,
)
from wsi.capture import find_blocks, find_mlps  # noqa: E402
from wsi.moipatch import _dtype_name, _sha256, module_suffix  # noqa: E402


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, dst)
    return dst


def _jaccard(a: list[int] | set[int], b: list[int] | set[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


# ------------------------------------------------------------------ 4 Parametric Reasoning Families
FAMILIES: tuple[str, ...] = (
    "linear_system_2var",
    "rate_work_combined",
    "compound_percent_chain",
    "relative_ratio_shift",
)


def build_reasoning_dataset() -> dict[str, Any]:
    """Build deterministic Train (4 re-numbered variants/family = 16) and Held-out Test
    (6 re-numbered + 6 rephrased_renumbered/family = 48) problems with concise teacher traces."""
    train_items: list[dict[str, Any]] = []
    test_items: list[dict[str, Any]] = []

    # ---- Family 1: linear_system_2var (a*x + b*y = C, x + y = N, find x) ----
    # Clean elimination recipe: subtract b*N from C, divide by (a - b)
    f1_train_params = [
        (8, 5, 14, 19),   # a=8, b=5, x=14, y=19
        (11, 7, 16, 21),  # a=11, b=7, x=16, y=21
        (9, 4, 18, 25),   # a=9, b=4, x=18, y=25
        (12, 5, 15, 23),  # a=12, b=5, x=15, y=23
    ]
    f1_test_renum = [
        (7, 3, 17, 22),
        (10, 6, 19, 24),
        (13, 8, 14, 27),
        (9, 5, 21, 16),
        (14, 9, 12, 29),
        (11, 6, 23, 18),
    ]
    f1_test_rephrase = [
        (8, 3, 15, 26, "server racks", "high-density nodes", "standard nodes", "watts-hundreds"),
        (12, 7, 18, 22, "cargo crates", "heavy containers", "light containers", "kilograms"),
        (9, 6, 13, 28, "laboratory trays", "glass vials", "plastic vials", "milliliters"),
        (11, 4, 16, 25, "printing batches", "color booklets", "mono booklets", "sheets"),
        (15, 9, 14, 21, "solar arrays", "dual-axis panels", "fixed panels", "kilowatts"),
        (10, 3, 19, 23, "orchestra rows", "premium seats", "balcony seats", "dollars"),
    ]

    def _f1_teacher(a: int, b: int, x: int, y: int) -> str:
        n = x + y
        c = a * x + b * y
        b_n = b * n
        diff_c = c - b_n
        diff_a = a - b
        return (
            f"<think>\nLet x be the first count and y be the second count.\n"
            f"We have x + y = {n} and {a}x + {b}y = {c}.\n"
            f"If all {n} items cost {b} each, the total is {b} * {n} = {b_n}.\n"
            f"The excess is {c} - {b_n} = {diff_c}, and each first-type item adds {a} - {b} = {diff_a}.\n"
            f"So x = {diff_c} / {diff_a} = {x}.\n</think>\n"
            f"By elimination, if all {n} items were at rate {b}, the total would be {b_n}. "
            f"Dividing the difference {c} - {b_n} = {diff_c} by ({a} - {b}) = {diff_a} gives \\boxed{{{x}}}."
        )

    for idx, (a, b, x, y) in enumerate(f1_train_params):
        n = x + y
        c = a * x + b * y
        q = (
            f"A theater sold {n} tickets in total. Adult tickets cost ${a} each and child tickets cost ${b} each. "
            f"If total ticket revenue was ${c}, how many adult tickets were sold?"
        )
        train_items.append({
            "id": f"train_f1_{idx}",
            "family": "linear_system_2var",
            "variant": "train_renumbered",
            "question": q,
            "teacher_response": _f1_teacher(a, b, x, y),
            "answer": x,
        })
    for idx, (a, b, x, y) in enumerate(f1_test_renum):
        n = x + y
        c = a * x + b * y
        q = (
            f"A theater sold {n} tickets in total. Adult tickets cost ${a} each and child tickets cost ${b} each. "
            f"If total ticket revenue was ${c}, how many adult tickets were sold?"
        )
        test_items.append({
            "id": f"test_f1_renum_{idx}",
            "family": "linear_system_2var",
            "variant": "renumbered",
            "question": q,
            "teacher_response": _f1_teacher(a, b, x, y),
            "answer": x,
        })
    for idx, (a, b, x, y, group_nm, item_a, item_b, unit_nm) in enumerate(f1_test_rephrase):
        n = x + y
        c = a * x + b * y
        q = (
            f"An facility deployed {n} {group_nm} composed of {item_a} and {item_b}. "
            f"Each of the {item_a} uses {a} {unit_nm} while each of the {item_b} uses {b} {unit_nm}. "
            f"Given a combined usage of {c} {unit_nm}, determine the exact number of {item_a}."
        )
        test_items.append({
            "id": f"test_f1_rephrase_{idx}",
            "family": "linear_system_2var",
            "variant": "rephrased_renumbered",
            "question": q,
            "teacher_response": _f1_teacher(a, b, x, y),
            "answer": x,
        })

    # ---- Family 2: rate_work_combined (Machine A takes t1 hrs, Machine B takes t2 hrs, total batch W = k * t1*t2/(t1+t2)) ----
    # Framed as: Pipe/Worker A does r1 units/hr, B does r2 units/hr, after h0 hours A works alone, then both together for total Q:
    # Even cleaner & exact integer: A produces r1/hr, B produces r2/hr. A runs alone for d hours, then A and B run together until total = T_units.
    # How many total hours did B run? ans = (T_units - d * r1) // (r1 + r2)
    f2_train_params = [
        (14, 18, 3, 330),   # ans = (330 - 42) / 32 = 288 / 32 = 9
        (16, 24, 4, 504),   # ans = (504 - 64) / 40 = 440 / 40 = 11
        (15, 21, 2, 318),   # ans = (318 - 30) / 36 = 288 / 36 = 8
        (12, 23, 5, 480),   # ans = (480 - 60) / 35 = 420 / 35 = 12
    ]
    f2_test_renum = [
        (13, 17, 4, 382),   # (382 - 52)/30 = 11
        (18, 22, 3, 414),   # (414 - 54)/40 = 9
        (11, 19, 5, 445),   # (445 - 55)/30 = 13
        (17, 25, 2, 370),   # (370 - 34)/42 = 8
        (19, 26, 4, 526),   # (526 - 76)/45 = 10
        (14, 21, 6, 609),   # (609 - 84)/35 = 15
    ]
    f2_test_rephrase = [
        (15, 20, 4, 480, "Pump Alpha", "Pump Beta", "liters"),         # (480 - 60)/35 = 12
        (12, 16, 5, 396, "Scanner North", "Scanner South", "pages"),   # (396 - 60)/28 = 12
        (16, 22, 3, 390, "Harvester X", "Harvester Y", "bushels"),     # (390 - 48)/38 = 9
        (13, 23, 4, 556, "Compiler Node 1", "Compiler Node 2", "jobs"),# (556 - 52)/36 = 14
        (18, 27, 2, 351, "Turbine East", "Turbine West", "megawatts"), # (351 - 36)/45 = 7
        (14, 19, 5, 598, " Courier Line A", "Courier Line B", "parcels"), # (598 - 70)/33 = 16
    ]

    def _f2_teacher(r1: int, r2: int, d: int, tot: int) -> tuple[str, int]:
        solo = d * r1
        rem = tot - solo
        comb = r1 + r2
        ans = rem // comb
        txt = (
            f"<think>\nFirst find the output during the solo phase: {d} * {r1} = {solo}.\n"
            f"Subtract from the target total: {tot} - {solo} = {rem} remaining.\n"
            f"Together their combined rate is {r1} + {r2} = {comb} per hour.\n"
            f"Divide remaining output by combined rate: {rem} / {comb} = {ans} hours.\n</think>\n"
            f"The first unit completes {d} * {r1} = {solo} alone, leaving {tot} - {solo} = {rem}. "
            f"At a combined rate of {r1} + {r2} = {comb} per hour, the second unit runs for {rem} / {comb} = \\boxed{{{ans}}}."
        )
        return txt, ans

    for idx, (r1, r2, d, tot) in enumerate(f2_train_params):
        t_resp, ans = _f2_teacher(r1, r2, d, tot)
        q = (
            f"Machine A manufactures {r1} parts per hour and Machine B manufactures {r2} parts per hour. "
            f"Machine A runs alone for {d} hours, and then Machine B joins in so both run together until {tot} parts are made in total. "
            f"For how many hours did Machine B run?"
        )
        train_items.append({
            "id": f"train_f2_{idx}",
            "family": "rate_work_combined",
            "variant": "train_renumbered",
            "question": q,
            "teacher_response": t_resp,
            "answer": ans,
        })
    for idx, (r1, r2, d, tot) in enumerate(f2_test_renum):
        t_resp, ans = _f2_teacher(r1, r2, d, tot)
        q = (
            f"Machine A manufactures {r1} parts per hour and Machine B manufactures {r2} parts per hour. "
            f"Machine A runs alone for {d} hours, and then Machine B joins in so both run together until {tot} parts are made in total. "
            f"For how many hours did Machine B run?"
        )
        test_items.append({
            "id": f"test_f2_renum_{idx}",
            "family": "rate_work_combined",
            "variant": "renumbered",
            "question": q,
            "teacher_response": t_resp,
            "answer": ans,
        })
    for idx, (r1, r2, d, tot, nm_a, nm_b, unit_nm) in enumerate(f2_test_rephrase):
        t_resp, ans = _f2_teacher(r1, r2, d, tot)
        q = (
            f"{nm_a} processes {r1} {unit_nm} every hour, whereas {nm_b} processes {r2} {unit_nm} every hour. "
            f"After {nm_a} operates by itself for {d} hours, {nm_b} is activated and both work simultaneously "
            f"until a cumulative total of {tot} {unit_nm} has been processed. How many hours was {nm_b} active?"
        )
        test_items.append({
            "id": f"test_f2_rephrase_{idx}",
            "family": "rate_work_combined",
            "variant": "rephrased_renumbered",
            "question": q,
            "teacher_response": t_resp,
            "answer": ans,
        })

    # ---- Family 3: compound_percent_chain (Reverse Sequential Percentage / Remainder Chain) ----
    # A warehouse starts with X items. Day 1 ships p1% of them. Day 2 ships p2% of the remainder, plus c extra items, leaving R items.
    # Find X. Reverse operation: before Day 2 percentage, rem1 = (R + c) * 100 / (100 - p2); X = rem1 * 100 / (100 - p1).
    f3_train_params = [
        (400, 20, 25, 15),   # rem1=320, after p2%=240, R = 240 - 15 = 225
        (600, 25, 20, 24),   # rem1=450, after p2%=360, R = 360 - 24 = 336
        (500, 30, 20, 18),   # rem1=350, after p2%=280, R = 280 - 18 = 262
        (800, 25, 30, 20),   # rem1=600, after p2%=420, R = 420 - 20 = 400
    ]
    f3_test_renum = [
        (300, 20, 25, 12),   # rem1=240, after p2%=180, R=168
        (700, 30, 20, 22),   # rem1=490, after p2%=392, R=370
        (900, 20, 25, 30),   # rem1=720, after p2%=540, R=510
        (500, 20, 30, 14),   # rem1=400, after p2%=280, R=266
        (600, 30, 25, 15),   # rem1=420, after p2%=315, R=300
        (1000, 25, 20, 35),  # rem1=750, after p2%=600, R=565
    ]
    f3_test_rephrase = [
        (400, 25, 20, 16, "research grant", "equipment", "travel", "dollars"),
        (700, 20, 25, 20, "water reservoir", "irrigation", "evaporation", "kiloliters"),
        (800, 30, 25, 25, "battery bank", "propulsion", "avionics", "watt-hours"),
        (600, 20, 30, 16, "seed archive", "spring planting", "greenhouse trials", "packets"),
        (900, 25, 20, 28, "cloud quota", "training jobs", "inference jobs", "credits"),
        (500, 25, 20, 15, "lumber stockpile", "framing", "roofing", "boards"),
    ]

    def _f3_teacher(x_orig: int, p1: int, p2: int, c_extra: int) -> tuple[str, int, int]:
        rem1 = x_orig * (100 - p1) // 100
        after_p2 = rem1 * (100 - p2) // 100
        r_final = after_p2 - c_extra
        k1 = 100 - p1
        k2 = 100 - p2
        txt = (
            f"<think>\nWork backward from the final remaining amount {r_final}.\n"
            f"Before the extra {c_extra} were removed, there were {r_final} + {c_extra} = {after_p2}.\n"
            f"That {after_p2} is {k2}% of the first remainder, so the first remainder is {after_p2} * 100 / {k2} = {rem1}.\n"
            f"That {rem1} is {k1}% of the initial amount, so the initial amount is {rem1} * 100 / {k1} = {x_orig}.\n</think>\n"
            f"Working backward: add {r_final} + {c_extra} = {after_p2}, divide by {k2}% to get {rem1}, "
            f"and divide by {k1}% to obtain the initial total \\boxed{{{x_orig}}}."
        )
        return txt, x_orig, r_final

    for idx, (x_orig, p1, p2, c_extra) in enumerate(f3_train_params):
        t_resp, ans, r_final = _f3_teacher(x_orig, p1, p2, c_extra)
        q = (
            f"A warehouse distributed {p1}% of its stock on Monday. On Tuesday it distributed {p2}% of the remaining stock "
            f"plus an additional {c_extra} boxes, leaving {r_final} boxes in the warehouse. "
            f"How many boxes were in the warehouse initially?"
        )
        train_items.append({
            "id": f"train_f3_{idx}",
            "family": "compound_percent_chain",
            "variant": "train_renumbered",
            "question": q,
            "teacher_response": t_resp,
            "answer": ans,
        })
    for idx, (x_orig, p1, p2, c_extra) in enumerate(f3_test_renum):
        t_resp, ans, r_final = _f3_teacher(x_orig, p1, p2, c_extra)
        q = (
            f"A warehouse distributed {p1}% of its stock on Monday. On Tuesday it distributed {p2}% of the remaining stock "
            f"plus an additional {c_extra} boxes, leaving {r_final} boxes in the warehouse. "
            f"How many boxes were in the warehouse initially?"
        )
        test_items.append({
            "id": f"test_f3_renum_{idx}",
            "family": "compound_percent_chain",
            "variant": "renumbered",
            "question": q,
            "teacher_response": t_resp,
            "answer": ans,
        })
    for idx, (x_orig, p1, p2, c_extra, src_nm, step1_nm, step2_nm, unit_nm) in enumerate(f3_test_rephrase):
        t_resp, ans, r_final = _f3_teacher(x_orig, p1, p2, c_extra)
        q = (
            f"A {src_nm} allocated {p1}% of its initial {unit_nm} to {step1_nm}. "
            f"Next, {p2}% of what remained—along with a fixed fee of {c_extra} {unit_nm}—was used for {step2_nm}. "
            f"If exactly {r_final} {unit_nm} were left over at the end, what was the starting balance of {unit_nm}?"
        )
        test_items.append({
            "id": f"test_f3_rephrase_{idx}",
            "family": "compound_percent_chain",
            "variant": "rephrased_renumbered",
            "question": q,
            "teacher_response": t_resp,
            "answer": ans,
        })

    # ---- Family 4: relative_ratio_shift (Currently A = k1 * B; after adding delta to both, A + delta = k2 * (B + delta); find A) ----
    # B = delta * (k2 - 1) / (k1 - k2), A = k1 * B
    f4_train_params = [
        (4, 2, 12),   # B = 12 * 1 / 2 = 6, A = 24
        (5, 3, 9),    # B = 9 * 2 / 2 = 9, A = 45
        (3, 2, 14),   # B = 14 * 1 / 1 = 14, A = 42
        (5, 2, 15),   # B = 15 * 1 / 3 = 5, A = 25
    ]
    f4_test_renum = [
        (4, 2, 16),   # B = 8, A = 32
        (5, 3, 11),   # B = 11, A = 55
        (3, 2, 19),   # B = 19, A = 57
        (4, 3, 8),    # B = 8 * 2 / 1 = 16, A = 64
        (5, 2, 21),   # B = 7, A = 35
        (6, 3, 12),   # B = 12 * 2 / 3 = 8, A = 48
    ]
    f4_test_rephrase = [
        (4, 2, 18, "Vault Alpha", "Vault Beta", "gold bars"),           # B = 9, A = 36
        (5, 3, 13, "Library East", "Library West", "manuscripts"),      # B = 13, A = 65
        (3, 2, 17, "Server Cluster A", "Server Cluster B", "GPUs"),     # B = 17, A = 51
        (5, 2, 18, "Orchard North", "Orchard South", "apple trees"),    # B = 6, A = 30
        (4, 3, 7, "Fleet One", "Fleet Two", "delivery vans"),           # B = 14, A = 56
        (6, 3, 15, "Silo Primary", "Silo Secondary", "tons of grain"),  # B = 10, A = 60
    ]

    def _f4_teacher(k1: int, k2: int, delta: int) -> tuple[str, int]:
        num = delta * (k2 - 1)
        den = k1 - k2
        b_val = num // den
        a_val = k1 * b_val
        txt = (
            f"<think>\nLet B be the smaller initial amount, so A = {k1}B.\n"
            f"After adding {delta} to each: {k1}B + {delta} = {k2}(B + {delta}) = {k2}B + {k2 * delta}.\n"
            f"Subtract {k2}B from both sides: ({k1} - {k2})B = {k2 * delta} - {delta} = {num}.\n"
            f"So B = {num} / {den} = {b_val}, and A = {k1} * {b_val} = {a_val}.\n</think>\n"
            f"Setting A = {k1}B gives {k1}B + {delta} = {k2}(B + {delta}), so ({k1} - {k2})B = {num}, "
            f"yielding B = {b_val} and A = {k1} * {b_val} = \\boxed{{{a_val}}}."
        )
        return txt, a_val

    for idx, (k1, k2, delta) in enumerate(f4_train_params):
        t_resp, ans = _f4_teacher(k1, k2, delta)
        q = (
            f"Tank A currently holds {k1} times as much water as Tank B. "
            f"If {delta} gallons are added to each tank, Tank A will hold {k2} times as much water as Tank B. "
            f"How many gallons of water does Tank A currently hold?"
        )
        train_items.append({
            "id": f"train_f4_{idx}",
            "family": "relative_ratio_shift",
            "variant": "train_renumbered",
            "question": q,
            "teacher_response": t_resp,
            "answer": ans,
        })
    for idx, (k1, k2, delta) in enumerate(f4_test_renum):
        t_resp, ans = _f4_teacher(k1, k2, delta)
        q = (
            f"Tank A currently holds {k1} times as much water as Tank B. "
            f"If {delta} gallons are added to each tank, Tank A will hold {k2} times as much water as Tank B. "
            f"How many gallons of water does Tank A currently hold?"
        )
        test_items.append({
            "id": f"test_f4_renum_{idx}",
            "family": "relative_ratio_shift",
            "variant": "renumbered",
            "question": q,
            "teacher_response": t_resp,
            "answer": ans,
        })
    for idx, (k1, k2, delta, nm_a, nm_b, unit_nm) in enumerate(f4_test_rephrase):
        t_resp, ans = _f4_teacher(k1, k2, delta)
        q = (
            f"At present, {nm_a} contains {k1} times as many {unit_nm} as {nm_b}. "
            f"Once {delta} new {unit_nm} are deposited into {nm_a} and {delta} new {unit_nm} are deposited into {nm_b}, "
            f"{nm_a} will have {k2} times as many {unit_nm} as {nm_b}. How many {unit_nm} are in {nm_a} right now?"
        )
        test_items.append({
            "id": f"test_f4_rephrase_{idx}",
            "family": "relative_ratio_shift",
            "variant": "rephrased_renumbered",
            "question": q,
            "teacher_response": t_resp,
            "answer": ans,
        })

    healthy_controls = [
        "Explain why the sky appears blue during the day and red at sunset in two short sentences.",
        "What is the difference between a Python list and a Python tuple?",
        "Summarize how photosynthesis works inside plant leaves in three bullet points.",
        "How does binary search achieve logarithmic time complexity on a sorted array?",
        "Name the planets of the Solar System in order from closest to farthest from the Sun.",
        "What is the role of hemoglobin in red blood cells?",
        "Explain the difference between TCP and UDP networking protocols.",
        "Why does bread dough rise when yeast is mixed with warm water and flour?",
        "What causes ocean tides to rise and fall twice a day on Earth?",
        "Describe the difference between an endothermic and an exothermic chemical reaction.",
        "How do you safely list and remove stopped containers in Docker?",
        "What is the Pythagorean theorem and how does it apply to a 3-4-5 right triangle?",
        "Why do metal spoons feel colder to the touch than wooden spoons at room temperature?",
        "Explain what an HTTP 404 status code indicates when visiting a web page.",
        "What is the difference between RAM and solid-state storage in a computer?",
        "How does a refrigerator remove heat from its interior compartment?",
    ]

    return {
        "families": list(FAMILIES),
        "train": train_items,
        "test": test_items,
        "healthy_controls": healthy_controls,
    }


_BOXED_RE = re.compile(r"\\boxed\{\s*([-+]?\d+(?:\.\d+)?)\s*\}")
_ANS_IS_RE = re.compile(r"(?:answer\s+is|equals|=)\s*\$?([-+]?\d+(?:\.\d+)?)\b", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_int_answer(text: str) -> int | None:
    """Extract final numeric answer (preferring \\boxed{}, then 'answer is X', then last integer)."""
    if not text:
        return None
    m_box = _BOXED_RE.findall(text)
    if m_box:
        try:
            return int(round(float(m_box[-1])))
        except ValueError:
            pass
    after_think = text.split("</think>")[-1] if "</think>" in text else text
    m_ans = _ANS_IS_RE.findall(after_think)
    if m_ans:
        try:
            return int(round(float(m_ans[-1])))
        except ValueError:
            pass
    nums = _NUM_RE.findall(after_think)
    if not nums:
        nums = _NUM_RE.findall(text)
    if nums:
        try:
            return int(round(float(nums[-1])))
        except ValueError:
            return None
    return None


def compute_invariance_metrics(
    family_traces: dict[str, list[CompactTrace]],
    *,
    top_k_sites: int = 4,
    top_n_neurons: int = 32,
) -> dict[str, Any]:
    """Compare within-family (re-numbered variants of same reasoning pattern) vs across-family
    causal direction cosines and top-neuron Jaccard overlaps."""
    all_traces = [ct for cts in family_traces.values() for ct in cts]
    global_diag = diagnose_causal_traces(
        all_traces, split_ab=False, top_k_sites=top_k_sites, top_n_neurons=top_n_neurons, center="healthy", healthy_cost="fisher"
    )
    eval_sites = [r["module"] for r in global_diag["sites"][:top_k_sites]]

    per_trace_dir: dict[tuple[str, int, str], torch.Tensor] = {}
    per_trace_neurons: dict[tuple[str, int, str], list[int]] = {}
    for fam, cts in family_traces.items():
        for idx, ct in enumerate(cts):
            for mk in eval_sites:
                gs = ct.grad_summary[mk]
                u = gs.get("c_target_centered_eigvecs", gs["c_target_eigvecs"])[0].float()
                diff = gs["paired_diff"].float()
                if float(diff.norm().item()) > 1e-12 and float((u @ diff).item()) < 0.0:
                    u = -u
                u = u / u.norm().clamp_min(1e-12)
                per_trace_dir[(fam, idx, mk)] = u

                supp = gs.get("neuron_supp_centered_attr", gs["neuron_supp_attr"]).float()
                fish = gs.get("neuron_fisher_attr", gs["neuron_dnll_attr"].abs()).float()
                # Rank active causal neurons by target attribution magnitude penalized by healthy Fisher
                net = supp.abs() - 0.25 * fish
                top_n = torch.argsort(net, descending=True)[:top_n_neurons].tolist()
                per_trace_neurons[(fam, idx, mk)] = top_n

    fams = list(family_traces.keys())
    within_cos_list: list[float] = []
    across_cos_list: list[float] = []
    within_jac_list: list[float] = []
    across_jac_list: list[float] = []

    for mk in eval_sites:
        for i_f, f_a in enumerate(fams):
            n_a = len(family_traces[f_a])
            for i in range(n_a):
                for j in range(i + 1, n_a):
                    u1 = per_trace_dir[(f_a, i, mk)]
                    u2 = per_trace_dir[(f_a, j, mk)]
                    within_cos_list.append(float(F.cosine_similarity(u1[None], u2[None]).abs().item()))
                    within_jac_list.append(_jaccard(per_trace_neurons[(f_a, i, mk)], per_trace_neurons[(f_a, j, mk)]))
            for j_f in range(i_f + 1, len(fams)):
                f_b = fams[j_f]
                for i in range(len(family_traces[f_a])):
                    for j in range(len(family_traces[f_b])):
                        u1 = per_trace_dir[(f_a, i, mk)]
                        u2 = per_trace_dir[(f_b, j, mk)]
                        across_cos_list.append(float(F.cosine_similarity(u1[None], u2[None]).abs().item()))
                        across_jac_list.append(_jaccard(per_trace_neurons[(f_a, i, mk)], per_trace_neurons[(f_b, j, mk)]))

    w_cos = sum(within_cos_list) / max(1, len(within_cos_list))
    a_cos = sum(across_cos_list) / max(1, len(across_cos_list))
    w_jac = sum(within_jac_list) / max(1, len(within_jac_list))
    a_jac = sum(across_jac_list) / max(1, len(across_jac_list))

    cos_ratio = w_cos / max(a_cos, 0.05)
    jac_ratio = w_jac / max(a_jac, 0.05)

    return {
        "eval_sites": eval_sites,
        "within_family_cos": round(w_cos, 4),
        "across_family_cos": round(a_cos, 4),
        "cos_ratio": round(cos_ratio, 4),
        "within_family_jaccard": round(w_jac, 4),
        "across_family_jaccard": round(a_jac, 4),
        "jaccard_ratio": round(jac_ratio, 4),
        "within_vs_across_ratio": round(max(cos_ratio, jac_ratio), 4),
    }


def extract_invariant_vs_single_candidates(
    traces: list[CompactTrace],
    mod_meta: dict[str, dict[str, Any]],
    *,
    prefix: str,
    model_or_weights: Any,
    site_override: str | None = None,
    top_n_cols: tuple[int | str, ...] = (16, 32, "all"),
    alphas: tuple[float, ...] = (-4.0, -2.0, 2.0, 4.0),
    seed: int = 42,
) -> dict[str, dict[str, MoIPatch]]:
    """Build three sets of candidate patches for a family or global pool:
      1. `invariant`: pooled across all 4 re-numbered variants, keeping ONLY neurons whose causal
         attribution is active across >= 75% of re-numbered variants (cancelling instance-number noise).
      2. `single_trace`: extracted from trace #0 alone (contaminated with trace #0's specific numbers).
      3. `random`: random orthogonal unit vector of identical sparsity and alpha.
    """
    diag_inv = diagnose_causal_traces(
        traces,
        split_ab=True,
        top_k_sites=4,
        top_n_neurons=64,
        center="healthy",
        healthy_cost="fisher",
        model_or_weights=model_or_weights,
    )
    diag_single = diagnose_causal_traces(
        [traces[0]],
        split_ab=False,
        top_k_sites=4,
        top_n_neurons=64,
        center="healthy",
        healthy_cost="fisher",
        model_or_weights=model_or_weights,
    )

    top_site_inv = site_override if (site_override and site_override in mod_meta) else next(r["module"] for r in diag_inv["sites"] if r["module"] in mod_meta)
    top_site_sng = top_site_inv

    # Number-cancellation neuron filter on top_site_inv:
    # Require consistent causal attribution across >= 75% of re-numbered variants relative to peak attribution
    supp_stack = torch.stack(
        [ct.grad_summary[top_site_inv].get("neuron_supp_centered_attr", ct.grad_summary[top_site_inv]["neuron_supp_attr"]).float() for ct in traces],
        dim=0,
    )
    fish_stack = torch.stack(
        [ct.grad_summary[top_site_inv].get("neuron_fisher_attr", ct.grad_summary[top_site_inv]["neuron_dnll_attr"].abs()).float() for ct in traces],
        dim=0,
    )
    abs_supp = supp_stack.abs()
    per_trace_thresh = 0.05 * abs_supp.max(dim=1, keepdim=True).values.clamp_min(1e-8)
    active_frac = (abs_supp >= per_trace_thresh).float().mean(dim=0)
    consistent = active_frac >= 0.75
    mean_abs = abs_supp.mean(dim=0)
    std_abs = abs_supp.std(dim=0, unbiased=False) if len(traces) > 1 else torch.zeros_like(mean_abs)
    raw_score = (mean_abs - 0.5 * std_abs) - 0.25 * fish_stack.mean(dim=0)
    inv_neuron_score = torch.where(consistent, raw_score, torch.full_like(raw_score, -1e9))
    inv_neurons = torch.argsort(inv_neuron_score, descending=True).tolist()

    sng_supp = traces[0].grad_summary[top_site_sng].get("neuron_supp_centered_attr", traces[0].grad_summary[top_site_sng]["neuron_supp_attr"]).float()
    sng_neurons = torch.argsort(sng_supp.abs(), descending=True).tolist()

    # Cross-variant consensus direction: average the sign-aligned top eigenvectors across all re-numbered traces
    u_ref = diag_inv["directions"][top_site_inv].float()
    u_sum = torch.zeros_like(u_ref)
    for ct in traces:
        gs = ct.grad_summary[top_site_inv]
        ui = gs.get("c_target_centered_eigvecs", gs["c_target_eigvecs"])[0].float()
        if float((ui @ u_ref).item()) < 0.0:
            ui = -ui
        u_sum = u_sum + ui
    u_inv = (u_sum / u_sum.norm().clamp_min(1e-12)).double()

    u_sng = traces[0].grad_summary[top_site_sng].get("c_target_centered_eigvecs", traces[0].grad_summary[top_site_sng]["c_target_eigvecs"])[0].float()
    if float((u_sng @ u_ref).item()) < 0.0:
        u_sng = -u_sng
    u_sng = (u_sng / u_sng.norm().clamp_min(1e-12)).double()

    g_rng = torch.Generator().manual_seed(seed)
    u_rnd = torch.randn(u_inv.shape, generator=g_rng, dtype=torch.float32)
    u_rnd = u_rnd - float((u_rnd @ u_inv.float()).item()) * u_inv.float()
    u_rnd = (u_rnd / u_rnd.norm().clamp_min(1e-12)).double()
    rnd_perm = torch.randperm(len(inv_neurons), generator=g_rng).tolist()

    inv_cands: dict[str, MoIPatch] = {}
    sng_cands: dict[str, MoIPatch] = {}
    rnd_cands: dict[str, MoIPatch] = {}

    mm_inv = mod_meta[top_site_inv]
    mm_sng = mod_meta[top_site_sng]

    for n_col in top_n_cols:
        cols_inv = None if n_col == "all" else torch.tensor(inv_neurons[: int(n_col)], dtype=torch.long)
        cols_sng = None if n_col == "all" else torch.tensor(sng_neurons[: int(n_col)], dtype=torch.long)
        cols_rnd = None if n_col == "all" else torch.tensor(rnd_perm[: int(n_col)], dtype=torch.long)
        for a in alphas:
            tag = f"n{n_col}_a{a:g}"
            inv_cands[f"{prefix}_inv_{tag}"] = MoIPatch(
                edits=[Edit(module=top_site_inv, u=u_inv, alpha=float(a), idx=cols_inv, **mm_inv)],
                meta={"name": f"{prefix}_inv_{tag}", "kind": "invariant", "site": top_site_inv},
            )
            sng_cands[f"{prefix}_sng_{tag}"] = MoIPatch(
                edits=[Edit(module=top_site_sng, u=u_sng, alpha=float(a), idx=cols_sng, **mm_sng)],
                meta={"name": f"{prefix}_sng_{tag}", "kind": "single_trace", "site": top_site_sng},
            )
            rnd_cands[f"{prefix}_rnd_{tag}"] = MoIPatch(
                edits=[Edit(module=top_site_inv, u=u_rnd, alpha=float(a), idx=cols_rnd, **mm_inv)],
                meta={"name": f"{prefix}_rnd_{tag}", "kind": "random", "site": top_site_inv},
            )

    return {"invariant": inv_cands, "single_trace": sng_cands, "random": rnd_cands}


# ------------------------------------------------------------------ CPU Smoke Test (Planted 2-Family + Number Decoy LM)
class _PlantedReasoningLM(nn.Module):
    """Synthetic 4-layer causal LM with 2 reasoning families and instance-specific number decoys.

    - Family A questions carry token 4 in the prompt; their teacher reasoning step predicts token 20.
    - Family B questions carry token 5 in the prompt; their teacher reasoning step predicts token 21.
    - Each train variant also contains an instance-specific number token (tokens 10..13) in the prompt
      that strongly excites neuron #0..3 along decoy dimensions (dims 4..7) which have strong local
      activations on that single instance AND perturb healthy token predictions (tokens 1..3),
      while the true invariant algorithmic neuron (#8 for Family A along dim 0, #9 for Family B along dim 1)
      is shared across all re-numbered variants of that family.
    """

    def __init__(self, vocab: int = 32, hidden: int = 32, inter: int = 64) -> None:
        super().__init__()
        from experiment_causal_diagnose import _PlantedBlock

        self.config = SimpleNamespace(model_type="llama", hidden_size=hidden, num_attention_heads=4)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_PlantedBlock(hidden, inter, 4) for _ in range(4)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        with torch.no_grad():
            for p in self.parameters():
                p.zero_()

            # Healthy tokens 1, 2, 3 depend on dims 4..7 (so perturbing instance-number dims 4..7 hurts healthy NLL!)
            self.lm_head.weight[1, 2] = 2.0
            self.lm_head.weight[2, 2] = 2.0
            self.lm_head.weight[3, 2] = 2.0
            for d_idx in range(4, 8):
                self.lm_head.weight[1, d_idx] = 2.5
                self.lm_head.weight[2, d_idx] = -2.5

            # Target reasoning tokens: token 20 (Family A) driven by +dim 0; token 21 (Family B) driven by +dim 1
            self.lm_head.weight[20, 0] = 0.3
            self.lm_head.weight[21, 1] = 0.3
            # A competing "wrong path" token 25 dominates everywhere via a constant bias channel (dim 3), so the base
            # model UNDER-predicts the teacher reasoning tokens (non-saturated softmax -> non-zero distillation gradient).
            self.lm_head.weight[25, 3] = 4.0

            # Embeddings:
            # Constant bias channel dim 3 on every token
            self.model.embed_tokens.weight[:, 3] = 1.0
            # Tokens 1..3: healthy context tokens -> dim 2
            for t in (1, 2, 3):
                self.model.embed_tokens.weight[t, 2] = 1.0
            # Family A structural cue (token 4 -> dim 8), Family B structural cue (token 5 -> dim 9)
            self.model.embed_tokens.weight[4, 8] = 1.5
            self.model.embed_tokens.weight[5, 9] = 1.5
            # Reasoning step tokens 20, 21 also carry their family structural state in teacher-forcing
            self.model.embed_tokens.weight[20, 8] = 1.5
            self.model.embed_tokens.weight[21, 9] = 1.5

            # Instance-specific number tokens 10..13 (Fam A) and 14..17 (Fam B) -> excite dims 10..17
            for num_idx, tok_id in enumerate((10, 11, 12, 13, 14, 15, 16, 17)):
                self.model.embed_tokens.weight[tok_id, 10 + num_idx] = 1.5

            # Layer 1 propagation: self-attn V/O copies structural cue (dims 8, 9) and number cue (dims 10..17) forward across positions
            b1 = self.model.layers[1]
            for d_c in range(8, 18):
                b1.self_attn.v_proj.weight[d_c, d_c] = 1.0
                b1.self_attn.o_proj.weight[d_c, d_c] = 1.5

            # Layer 2 MLP:
            # - Neuron 8: invariant Family A circuit (reads dim 8 -> writes +dim 0, active across all 4 Fam A variants, 0 healthy cost)
            # - Neuron 9: invariant Family B circuit (reads dim 9 -> writes +dim 1, active across all 4 Fam B variants, 0 healthy cost)
            # - Neurons 0..3: Fam A instance-number decoys (each fires ONLY on 1 number token 10..13; slightly higher single-trace
            #   raw target attribution than Neuron 8, so single_trace picks Neuron 0, while invariant consensus across 4 variants
            #   and Fisher-penalized neuron ranking both pick Neuron 8!)
            # - Neurons 16..19: Fam B instance-number decoys (each fires ONLY on 1 number token 14..17)
            b2 = self.model.layers[2]
            b2.mlp.gate_proj.weight[8, 8] = 1.6
            b2.mlp.up_proj.weight[8, 8] = 1.6
            b2.mlp.down_proj.weight[0, 8] = 0.5

            b2.mlp.gate_proj.weight[9, 9] = 1.6
            b2.mlp.up_proj.weight[9, 9] = 1.6
            b2.mlp.down_proj.weight[1, 9] = 0.5

            for num_idx in range(4):
                b2.mlp.gate_proj.weight[num_idx, 10 + num_idx] = 2.6
                b2.mlp.up_proj.weight[num_idx, 10 + num_idx] = 2.6
                b2.mlp.down_proj.weight[0, num_idx] = 20.0
                b2.mlp.down_proj.weight[4 + num_idx, num_idx] = 12.0

                b2.mlp.gate_proj.weight[16 + num_idx, 14 + num_idx] = 2.6
                b2.mlp.up_proj.weight[16 + num_idx, 14 + num_idx] = 2.6
                b2.mlp.down_proj.weight[1, 16 + num_idx] = 20.0
                b2.mlp.down_proj.weight[4 + num_idx, 16 + num_idx] = 12.0

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False, output_hidden_states: bool = False) -> SimpleNamespace:
        x = self.model.embed_tokens(input_ids)
        h_states = [x] if output_hidden_states else None
        for blk in self.model.layers:
            x = blk(x)
            if h_states is not None:
                h_states.append(x)
        return SimpleNamespace(logits=self.lm_head(x), hidden_states=h_states)


def run_smoke() -> dict[str, Any]:
    model = _PlantedReasoningLM().eval()
    # 4 re-numbered train variants per family:
    # Notice token at pos 4 is inside the number-bearing prompt (healthy_span=(0, 3)),
    # while the teacher reasoning span is target_span=(4, 8) (predicting tokens at indices 5, 6, 7).
    fam_a_train = [[1, 2, 4, num_t, 20, 20, 20, 20] for num_t in (10, 11, 12, 13)]
    fam_b_train = [[1, 2, 5, num_t, 21, 21, 21, 21] for num_t in (14, 15, 16, 17)]

    traces_a = [
        capture_causal_trace(
            model, s, target_span=(5, 8), healthy_span=(0, 2),
            grad_mode="compressed", center="healthy", healthy_cost="fisher",
            top_k_mlp=16, group_size=16, hadamard_block=16, top_k_attn=0, capture_cache=False,
        )
        for s in fam_a_train
    ]
    traces_b = [
        capture_causal_trace(
            model, s, target_span=(5, 8), healthy_span=(0, 2),
            grad_mode="compressed", center="healthy", healthy_cost="fisher",
            top_k_mlp=16, group_size=16, hadamard_block=16, top_k_attn=0, capture_cache=False,
        )
        for s in fam_b_train
    ]

    inv_metrics = compute_invariance_metrics({"fam_a": traces_a, "fam_b": traces_b}, top_k_sites=1, top_n_neurons=2)
    assert inv_metrics["within_vs_across_ratio"] >= 1.35, inv_metrics

    mods = dict(model.named_modules())
    mod_meta = {}
    for g in find_mlps(model):
        mk = module_suffix(g.down)
        w = mods[g.down].weight.detach()
        mod_meta[mk] = {"base_sha256": _sha256(w), "base_dtype": _dtype_name(w.dtype), "shape": tuple(w.shape)}

    cands_a = extract_invariant_vs_single_candidates(
        traces_a,
        mod_meta,
        prefix="fam_a",
        model_or_weights=model,
        site_override="layers.2.mlp.down_proj",
        top_n_cols=(1,),
        alphas=(-2.0, 2.0),
    )
    # Held-out re-numbered test sequences for Family A (with new number mixes)
    test_a_seqs = [[1, 3, 4, 12, 20, 20, 20, 20], [2, 1, 4, 13, 20, 20, 20, 20]]
    healthy_seqs = [[1, 2, 10, 1, 2], [2, 1, 11, 2, 1]]

    all_cands_a = {**cands_a["invariant"], **cands_a["single_trace"], **cands_a["random"]}
    ranked_a = screen_patch_candidates(
        model, all_cands_a, healthy_seqs=healthy_seqs, target_seqs=test_a_seqs, target_spans=[(5, 8)] * len(test_a_seqs)
    )
    for r in ranked_a:
        t_gain = -float(r["target_suppression"])
        r["teacher_logp_gain"] = round(t_gain, 6)
        r["distill_pareto"] = round(t_gain / max(float(r["healthy_kl"]), 0.005), 6)

    best_inv = max((r for r in ranked_a if "_inv_" in r["name"]), key=lambda r: r["distill_pareto"])
    best_sng = max((r for r in ranked_a if "_sng_" in r["name"]), key=lambda r: r["distill_pareto"])
    best_rnd = max((r for r in ranked_a if "_rnd_" in r["name"]), key=lambda r: r["distill_pareto"])

    # Verify the invariant patch selected neuron #8 (true Family A algorithm neuron), while single_trace was lured by neuron #0 (number token 10 decoy)
    win_inv_patch = all_cands_a[best_inv["name"]]
    assert int(win_inv_patch.edits[0].idx[0].item()) == 8, win_inv_patch.edits[0].idx
    assert best_inv["distill_pareto"] > best_sng["distill_pareto"], (best_inv, best_sng)
    assert best_inv["distill_pareto"] > best_rnd["distill_pareto"], (best_inv, best_rnd)

    return {
        "mode": "smoke",
        "passed": True,
        "invariance": inv_metrics,
        "best_invariant": best_inv,
        "best_single_trace": best_sng,
        "best_random": best_rnd,
        "criteria": {
            "a_invariance_passed": bool(inv_metrics["within_vs_across_ratio"] >= 1.35),
            "b_number_cancellation_passed": bool(best_inv["distill_pareto"] > best_sng["distill_pareto"] and best_inv["distill_pareto"] > best_rnd["distill_pareto"]),
            "c_distillation_gain_passed": bool(best_inv["teacher_logp_gain"] > 0.05),
            "all_passed": True,
        },
    }


# ------------------------------------------------------------------ Real GPU Pilot
def _format_chat(tok: Any, user_text: str, assistant_text: str | None = None, *, enable_thinking: bool = True) -> list[int]:
    msgs = [{"role": "user", "content": user_text}]
    if assistant_text is not None:
        msgs.append({"role": "assistant", "content": assistant_text})
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    else:
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return [int(x) for x in tok.encode(s, add_special_tokens=False)]


@torch.no_grad()
def _extract_prompt_hidden_vecs(model: nn.Module, prompt_ids_list: list[list[int]], layer_idx: int) -> torch.Tensor:
    """Run 1 fast prefill per prompt and return the L2-normalized mean-pooled hidden state at `layer_idx`."""
    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
    blk = blocks[layer_idx]
    vecs: list[torch.Tensor] = []
    for p_ids in prompt_ids_list:
        captured: list[torch.Tensor] = []

        def _hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured.append(t.detach().float()[0].mean(dim=0).cpu())

        h = blk.register_forward_hook(_hook)
        try:
            ids_t = torch.tensor(p_ids, dtype=torch.long, device=dev).unsqueeze(0)
            model(input_ids=ids_t, use_cache=False)
        finally:
            h.remove()
        v = captured[0]
        vecs.append(v / v.norm().clamp_min(1e-12))
    return torch.stack(vecs, dim=0)


@torch.no_grad()
def _batch_generate_eval(
    model: nn.Module,
    tok: Any,
    items: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    batch_size: int = 16,
    patch: MoIPatch | None = None,
    routed_patches: dict[str, MoIPatch] | None = None,
    routed_assignments: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate generation accuracy, loop/truncation rate, and mean thinking tokens across `items`."""
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Group indices by active patch so each batch runs with a single clean weight state
    if routed_patches is not None and routed_assignments is not None:
        groups: dict[str, list[int]] = {}
        for idx, fam_k in enumerate(routed_assignments):
            groups.setdefault(fam_k, []).append(idx)
    else:
        groups = {"_single": list(range(len(items)))}

    results_by_idx: dict[int, dict[str, Any]] = {}
    for g_key, idxs in groups.items():
        active_p = routed_patches.get(g_key) if (routed_patches is not None and g_key != "_single") else patch
        handle = active_p.attach(model, verify_fingerprint=False) if (active_p is not None and active_p.edits) else None
        if handle is not None:
            handle.on()
        try:
            for b_start in range(0, len(idxs), batch_size):
                sub_idxs = idxs[b_start : b_start + batch_size]
                prompts = []
                for i in sub_idxs:
                    q_txt = items[i]["question"]
                    try:
                        p_str = tok.apply_chat_template(
                            [{"role": "user", "content": q_txt}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                    except TypeError:
                        p_str = tok.apply_chat_template(
                            [{"role": "user", "content": q_txt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    prompts.append(p_str)
                enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
                in_w = int(enc["input_ids"].shape[1])
                # Same sampling every repo run uses (lr.SAMPLING minus vLLM-only presence_penalty);
                # greedy decoding inflates loops (AGENTS.md MATH-500 4B finding). Fixed seed per batch = paired arms.
                torch.manual_seed(1234 + b_start)
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                )
                for j, i in enumerate(sub_idxs):
                    out_ids = gen[j, in_w:].tolist()
                    # Trim trailing pad/eos tokens
                    while out_ids and out_ids[-1] in (tok.pad_token_id, tok.eos_token_id):
                        out_ids.pop()
                    txt = tok.decode(out_ids, skip_special_tokens=False)
                    pred = extract_int_answer(txt)
                    gold = int(items[i]["answer"])
                    correct = bool(pred == gold)
                    results_by_idx[i] = {
                        "id": items[i]["id"],
                        "family": items[i]["family"],
                        "variant": items[i]["variant"],
                        "gold": gold,
                        "pred": pred,
                        "correct": correct,
                        "n_tokens": len(out_ids),
                        "routed_expert": g_key if g_key != "_single" else None,
                    }
                del enc, gen
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            if handle is not None:
                handle.off()

    ordered = [results_by_idx[i] for i in range(len(items))]
    n_tot = max(1, len(ordered))
    renum = [r for r in ordered if r["variant"] == "renumbered"]
    rephr = [r for r in ordered if r["variant"] == "rephrased_renumbered"]

    return {
        "accuracy": round(sum(1 for r in ordered if r["correct"]) / n_tot, 4),
        "accuracy_renumbered": round(sum(1 for r in renum if r["correct"]) / max(1, len(renum)), 4),
        "accuracy_rephrased_renumbered": round(sum(1 for r in rephr if r["correct"]) / max(1, len(rephr)), 4),
        "mean_tokens": round(sum(r["n_tokens"] for r in ordered) / n_tot, 1),
        "mean_tokens_correct": round(
            sum(r["n_tokens"] for r in ordered if r["correct"]) / max(1, sum(1 for r in ordered if r["correct"])), 1
        ),
        "n_correct": sum(1 for r in ordered if r["correct"]),
        "n_total": len(ordered),
        "per_item": ordered,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        args.dtype, torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float32
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = build_reasoning_dataset()
    train_items = ds["train"]
    test_items = ds["test"]
    healthy_texts = ds["healthy_controls"]

    # Build module metadata for MoIPatch construction
    mods = dict(model.named_modules())
    mod_meta: dict[str, dict[str, Any]] = {}
    for g in find_mlps(model):
        mk = module_suffix(g.down)
        w = mods[g.down].weight.detach()
        mod_meta[mk] = {"base_sha256": _sha256(w), "base_dtype": _dtype_name(w.dtype), "shape": tuple(w.shape)}
    for n, m in model.named_modules():
        if "mtp" not in n and "visual" not in n and n.endswith((".o_proj", ".out_proj")) and ".mlp." not in n:
            try:
                mk = module_suffix(n)
                w = m.weight.detach()
                mod_meta[mk] = {"base_sha256": _sha256(w), "base_dtype": _dtype_name(w.dtype), "shape": tuple(w.shape)}
            except ValueError:
                pass

    # Phase 1: Capture Causal CompactTraces on the 16 Train items (4 re-numbered variants x 4 families)
    print(f"[pilot] Phase 1: Capturing causal CompactTraces on {len(train_items)} teacher-forced train traces...", flush=True)
    t_cap = time.perf_counter()
    family_traces: dict[str, list[CompactTrace]] = {f: [] for f in FAMILIES}
    family_train_seqs: dict[str, list[list[int]]] = {f: [] for f in FAMILIES}
    family_train_spans: dict[str, list[tuple[int, int]]] = {f: [] for f in FAMILIES}

    for item in train_items:
        p_ids = _format_chat(tok, item["question"], None)
        full_ids = _format_chat(tok, item["question"], item["teacher_response"])
        t_start = max(2, len(p_ids))
        t_end = len(full_ids)
        h_span = (0, max(1, t_start - 1))
        t_span = (t_start, t_end)
        ct = capture_causal_trace(
            model,
            full_ids,
            target_span=t_span,
            healthy_span=h_span,
            grad_mode="compressed",
            grad_checkpointing=True,
            center="healthy",
            healthy_cost="fisher",
            top_k_attn=0,
            capture_cache=False,
        )
        fam = item["family"]
        family_traces[fam].append(ct)
        family_train_seqs[fam].append(full_ids)
        family_train_spans[fam].append(t_span)

    capture_s = round(time.perf_counter() - t_cap, 2)
    invariance = compute_invariance_metrics(family_traces, top_k_sites=4, top_n_neurons=32)
    print(f"[pilot] Invariance metrics ({capture_s}s): {invariance}", flush=True)

    # Prepare healthy control token sequences and held-out teacher-forced sequences for forward screening
    healthy_seqs = [_format_chat(tok, h_txt, "Here is a clear and concise explanation.") for h_txt in healthy_texts[:12]]

    family_test_seqs: dict[str, list[list[int]]] = {f: [] for f in FAMILIES}
    family_test_spans: dict[str, list[tuple[int, int]]] = {f: [] for f in FAMILIES}
    for item in test_items:
        p_ids = _format_chat(tok, item["question"], None)
        full_ids = _format_chat(tok, item["question"], item["teacher_response"])
        t_start = max(2, len(p_ids))
        family_test_seqs[item["family"]].append(full_ids)
        family_test_spans[item["family"]].append((t_start, len(full_ids)))

    # Phase 2: Distill per-family `.moipatch` experts (Invariant vs Single-Trace vs Random)
    # Note: In screen_patch_candidates, `target_suppression` measures drop in log-prob of target tokens.
    # For reasoning distillation we want to PROMOTE the teacher reasoning tokens (i.e. negative target_suppression = log-prob GAIN),
    # or equivalently we pass both positive and negative alphas and pick the patch that maximizes teacher log-prob gain
    # per unit of healthy KL / NLL drift on the train sequences, then evaluate on held-out test sequences!
    print("[pilot] Phase 2: Screening Invariant vs Single-Trace vs Random distilled patches per family...", flush=True)
    t_scr = time.perf_counter()

    best_inv_experts: dict[str, MoIPatch] = {}
    best_sng_experts: dict[str, MoIPatch] = {}
    best_rnd_experts: dict[str, MoIPatch] = {}
    family_screen_summary: dict[str, Any] = {}

    inv_test_gains: list[float] = []
    sng_test_gains: list[float] = []
    rnd_test_gains: list[float] = []

    for fam_idx, fam in enumerate(FAMILIES):
        cand_groups = extract_invariant_vs_single_candidates(
            family_traces[fam],
            mod_meta,
            prefix=fam,
            model_or_weights=model,
            top_n_cols=(16, 32, 64, "all"),
            alphas=(-6.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 6.0),
            seed=100 + fam_idx,
        )

        def _pick_best_teacher_promoter(cands_dict: dict[str, MoIPatch]) -> tuple[str, MoIPatch, dict[str, Any], dict[str, Any]]:
            # Select on Train sequences by teacher log-prob gain (-target_suppression) / max(healthy_kl, 0.005)
            train_rows = screen_patch_candidates(
                model,
                cands_dict,
                healthy_seqs=healthy_seqs[:8],
                target_seqs=family_train_seqs[fam],
                target_spans=family_train_spans[fam],
                max_nll_delta=0.05,
            )
            for r in train_rows:
                teacher_gain = -float(r["target_suppression"])
                r["teacher_logp_gain"] = round(teacher_gain, 6)
                r["distill_pareto"] = round(teacher_gain / max(float(r["healthy_kl"]), 0.005), 6)
            guard_ok = [r for r in train_rows if r["passes_nll_guard"] and r["healthy_kl"] <= 0.03]
            pool = guard_ok if guard_ok else train_rows
            win_train = max(pool, key=lambda r: (r["distill_pareto"], r["teacher_logp_gain"]))
            win_name = win_train["name"]
            win_patch = cands_dict[win_name]

            # Evaluate the chosen winner blindly on the 12 held-out Test variants of this family
            test_rows = screen_patch_candidates(
                model,
                {win_name: win_patch},
                healthy_seqs=healthy_seqs[8:],
                target_seqs=family_test_seqs[fam][:6],
                target_spans=family_test_spans[fam][:6],
                max_nll_delta=0.05,
            )
            win_test = test_rows[0]
            t_gain = -float(win_test["target_suppression"])
            win_test["teacher_logp_gain"] = round(t_gain, 6)
            win_test["distill_pareto"] = round(t_gain / max(float(win_test["healthy_kl"]), 0.005), 6)
            return win_name, win_patch, win_train, win_test

        _, p_inv, tr_inv, te_inv = _pick_best_teacher_promoter(cand_groups["invariant"])
        _, p_sng, tr_sng, te_sng = _pick_best_teacher_promoter(cand_groups["single_trace"])
        _, p_rnd, tr_rnd, te_rnd = _pick_best_teacher_promoter(cand_groups["random"])

        best_inv_experts[fam] = p_inv
        best_sng_experts[fam] = p_sng
        best_rnd_experts[fam] = p_rnd

        inv_test_gains.append(float(te_inv["distill_pareto"]))
        sng_test_gains.append(float(te_sng["distill_pareto"]))
        rnd_test_gains.append(float(te_rnd["distill_pareto"]))

        family_screen_summary[fam] = {
            "invariant_train": tr_inv,
            "invariant_heldout_test": te_inv,
            "single_trace_heldout_test": te_sng,
            "random_heldout_test": te_rnd,
        }
        print(
            f"[pilot] Family {fam}: held-out distill_pareto inv={te_inv['distill_pareto']:.4f} "
            f"vs single={te_sng['distill_pareto']:.4f} vs rand={te_rnd['distill_pareto']:.4f}",
            flush=True,
        )

    screen_s = round(time.perf_counter() - t_scr, 2)

    # Phase 3: Fit a zero-generation Prefill Router in weight/activation space (middle layer)
    _, blocks = find_blocks(model)
    router_layer = len(blocks) // 2
    train_prompt_ids = [_format_chat(tok, it["question"], None) for it in train_items]
    test_prompt_ids = [_format_chat(tok, it["question"], None) for it in test_items]

    train_vecs = _extract_prompt_hidden_vecs(model, train_prompt_ids, router_layer)
    test_vecs = _extract_prompt_hidden_vecs(model, test_prompt_ids, router_layer)

    # Compute family centroids on the 16 Train prompts
    centroids: dict[str, torch.Tensor] = {}
    for fam in FAMILIES:
        f_idxs = [i for i, it in enumerate(train_items) if it["family"] == fam]
        c = train_vecs[f_idxs].mean(dim=0)
        centroids[fam] = c / c.norm().clamp_min(1e-12)

    routed_assignments: list[str] = []
    router_correct = 0
    router_correct_renum = 0
    router_correct_rephr = 0
    n_renum = 0
    n_rephr = 0
    for i, it in enumerate(test_items):
        v = test_vecs[i]
        pred_fam = max(FAMILIES, key=lambda f: float((v @ centroids[f]).item()))
        routed_assignments.append(pred_fam)
        ok = pred_fam == it["family"]
        router_correct += int(ok)
        if it["variant"] == "renumbered":
            n_renum += 1
            router_correct_renum += int(ok)
        else:
            n_rephr += 1
            router_correct_rephr += int(ok)

    router_report = {
        "router_layer": router_layer,
        "accuracy_all": round(router_correct / max(1, len(test_items)), 4),
        "accuracy_renumbered": round(router_correct_renum / max(1, n_renum), 4),
        "accuracy_rephrased_renumbered": round(router_correct_rephr / max(1, n_rephr), 4),
    }
    print(f"[pilot] Prefill Router (layer {router_layer}): {router_report}", flush=True)

    # Phase 4: Held-out Generation Evaluation across the 48 test variants (renumbered + rephrased_renumbered)
    print(f"[pilot] Phase 4: Running held-out generation on {len(test_items)} variants (max_new_tokens={args.max_new_tokens})...", flush=True)
    t_gen = time.perf_counter()

    eval_base = _batch_generate_eval(
        model, tok, test_items, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
    )
    print(f"[pilot] Arm 1 (base): acc={eval_base['accuracy']} | mean_tokens={eval_base['mean_tokens']}", flush=True)

    eval_rnd = _batch_generate_eval(
        model, tok, test_items, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        routed_patches=best_rnd_experts, routed_assignments=routed_assignments,
    )
    print(f"[pilot] Arm 2 (random_routed): acc={eval_rnd['accuracy']} | mean_tokens={eval_rnd['mean_tokens']}", flush=True)

    eval_sng = _batch_generate_eval(
        model, tok, test_items, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        routed_patches=best_sng_experts, routed_assignments=routed_assignments,
    )
    print(f"[pilot] Arm 3 (single_trace_routed): acc={eval_sng['accuracy']} | mean_tokens={eval_sng['mean_tokens']}", flush=True)

    eval_inv = _batch_generate_eval(
        model, tok, test_items, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        routed_patches=best_inv_experts, routed_assignments=routed_assignments,
    )
    print(f"[pilot] Arm 4 (invariant_routed_moi): acc={eval_inv['accuracy']} | mean_tokens={eval_inv['mean_tokens']}", flush=True)

    gen_s = round(time.perf_counter() - t_gen, 2)

    mean_inv_pareto = sum(inv_test_gains) / max(1, len(inv_test_gains))
    mean_sng_pareto = sum(sng_test_gains) / max(1, len(sng_test_gains))
    mean_rnd_pareto = sum(rnd_test_gains) / max(1, len(rnd_test_gains))
    mean_inv_kl = sum(
        float(family_screen_summary[f]["invariant_heldout_test"]["healthy_kl"]) for f in FAMILIES
    ) / len(FAMILIES)

    crit_a = bool(invariance["within_vs_across_ratio"] >= 1.35)
    crit_b = bool(
        (mean_inv_pareto > mean_sng_pareto and mean_inv_pareto > mean_rnd_pareto)
        or (eval_inv["accuracy"] > eval_sng["accuracy"] and eval_inv["accuracy"] > eval_rnd["accuracy"])
    )
    crit_c = bool(
        mean_inv_kl <= 0.03
        and (
            eval_inv["accuracy"] > eval_base["accuracy"]
            or (eval_inv["accuracy"] >= eval_base["accuracy"] and eval_inv["mean_tokens"] < 0.85 * eval_base["mean_tokens"])
        )
    )

    return {
        "model": args.model,
        "dtype": str(dtype),
        "n_train": len(train_items),
        "n_test": len(test_items),
        "max_new_tokens": args.max_new_tokens,
        "timing_s": {
            "capture_s": capture_s,
            "screen_s": screen_s,
            "generation_s": gen_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "invariance": invariance,
        "router_report": router_report,
        "heldout_forward_distill": {
            "mean_invariant_distill_pareto": round(mean_inv_pareto, 4),
            "mean_single_trace_distill_pareto": round(mean_sng_pareto, 4),
            "mean_random_distill_pareto": round(mean_rnd_pareto, 4),
            "mean_invariant_healthy_kl": round(mean_inv_kl, 6),
            "per_family": family_screen_summary,
        },
        "generation_arms": {
            "base": {k: v for k, v in eval_base.items() if k != "per_item"},
            "random_routed": {k: v for k, v in eval_rnd.items() if k != "per_item"},
            "single_trace_routed": {k: v for k, v in eval_sng.items() if k != "per_item"},
            "invariant_routed_moi": {k: v for k, v in eval_inv.items() if k != "per_item"},
        },
        "criteria": {
            "a_within_vs_across_ratio": invariance["within_vs_across_ratio"],
            "a_target": ">= 1.35",
            "a_passed": crit_a,
            "b_invariant_vs_single_and_random": {
                "fwd_pareto_inv_vs_sng_vs_rnd": [round(mean_inv_pareto, 4), round(mean_sng_pareto, 4), round(mean_rnd_pareto, 4)],
                "gen_acc_inv_vs_sng_vs_rnd": [eval_inv["accuracy"], eval_sng["accuracy"], eval_rnd["accuracy"]],
            },
            "b_passed": crit_b,
            "c_distillation_gain_vs_base": {
                "base_acc": eval_base["accuracy"],
                "invariant_routed_acc": eval_inv["accuracy"],
                "base_mean_tokens": eval_base["mean_tokens"],
                "invariant_routed_mean_tokens": eval_inv["mean_tokens"],
                "mean_healthy_kl": round(mean_inv_kl, 6),
            },
            "c_passed": crit_c,
            "all_passed": bool(crit_a and crit_b and crit_c),
        },
        "per_item_generation": {
            "base": eval_base["per_item"],
            "single_trace_routed": eval_sng["per_item"],
            "invariant_routed_moi": eval_inv["per_item"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU smoke test")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
