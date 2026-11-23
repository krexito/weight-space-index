"""Real-World Pocket MoE Benchmark: MATH-500 Distillation + MMLU Forgetting Test (Qwen3.5-4B -> Qwen3.5-0.8B).

Evaluates whether WSI-localized, prefill-routed Pocket MoE specialists generalize to real competition math
(`HuggingFaceH4/MATH-500`, `n_train=300`, `n_test=200`, all 7 subjects & Levels 1..5 under `enable_thinking=False`)
while protecting general knowledge (`cais/mmlu`, `n_test=400` stratified + `24` held-out prose controls):

  1. `base_08b`                   : Unmodified Qwen3.5-0.8B (`enable_thinking=False`)
  2. `teacher_4b`                 : Teacher ceiling Qwen3.5-4B (`enable_thinking=False`)
  3. `moe_micro_wsi_routed`       : 3 domain experts (`r=8`, top-4 WSI causal layers, ~`1.7 MB`/expert) + L12 router
  4. `moe_micro_random_routed`    : 3 domain experts (`r=8`, 4 stratified-random layers, ~`1.7 MB`/expert) + L12 router
  5. `moe_large_wsi_routed`       : 3 domain experts (`r=64`, top-8 WSI causal layers, ~`27.4 MB`/expert) + L12 router
  6. `global_lora_r64`            : 1 unrouted LoRA (`r=64`, 8 global WSI layers, ~`27.5 MB`) trained on all MATH domains
  7. `routed_single_math_lora_r64`: Same single ~`27.5 MB` math LoRA gated by the Layer-12 router (zero extra GPU cost)
  8. `global_lora_bank_matched`   : 1 unrouted LoRA (`r=64`, all `24` layers, ~`82.6 MB` matching `3 x 27.4 MB`)

CPU smoke test::
    python scripts/experiment_pocket_moe_math500.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_moe_math500.py --teacher /content/teacher_4b --student /content/student_08b --teacher-cache /content/drive/MyDrive/wsi_runs/_cache/math500_4b_no_think_2048.json --out /content/pocket_moe_math500_l4.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gc
import hashlib
import json
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

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
    _bake_full_checkpoint,
    _extract_causal_layers,
    _hf_generate_subset,
    _norm_param_key,
    _train_lora_expert,
    _vllm_generate,
    fit_ridge_router,
    predict_ridge_router,
)
from experiment_pocket_moe_controls import (  # noqa: E402
    _paired_bootstrap,
    pick_stratified_random_layers,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _PlantedReasoningLM,
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
    compute_invariance_metrics,
)
from experiment_reasoning_distill_v2 import (  # noqa: E402
    _format_messages,
    _select_backend,
)
from wsi import CompactTrace, capture_causal_trace, diagnose_causal_traces  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402

MATH_DOMAINS: tuple[str, ...] = ("algebra", "advanced_algebra", "discrete_geom")
ROUTER_CLASSES: tuple[str, ...] = (*MATH_DOMAINS, "passthrough")

SUBJECT_TO_DOMAIN: dict[str, str] = {
    "Algebra": "algebra",
    "Prealgebra": "algebra",
    "Intermediate Algebra": "advanced_algebra",
    "Precalculus": "advanced_algebra",
    "Number Theory": "discrete_geom",
    "Counting & Probability": "discrete_geom",
    "Geometry": "discrete_geom",
}

MMLU_MATH_SUBJECTS: frozenset[str] = frozenset({
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
})

PROSE_ROUTER_TRAIN: list[str] = [
    "Summarize why coastal cities experience milder winter temperatures than inland cities.",
    "Explain the difference between RAM and long-term storage on a laptop.",
    "Describe two ways photosynthesis supports ecosystems on Earth.",
    "Write a short, polite note declining a meeting due to a scheduling conflict.",
    "Explain what causes the seasons to change over the course of a year.",
    "Give three practical tips for organizing research notes effectively.",
    "Why do ships look as if they disappear bottom-first over the ocean horizon?",
    "Describe the main role of red blood cells in the human circulatory system.",
    "Explain how vaccines train the immune system without causing severe illness.",
    "Why does bread rise when yeast ferments sugars in the dough?",
    "Summarize the difference between weather and climate in simple terms.",
    "Describe why bridges include small expansion joints between sections.",
    "What is the purpose of DNS on the modern internet?",
    "Explain why the sky typically looks red or orange near sunset.",
    "Give two reasons why biodiversity helps forests recover after storms.",
    "Write one paragraph explaining how sonar helps submarines navigate underwater.",
]

PROSE_HELDOUT: list[str] = [
    "Explain why ocean tides happen twice a day across most coastlines.",
    "Compare anaerobic exercise and aerobic endurance training briefly.",
    "Describe how coral reefs protect shorelines from storm waves.",
    "Write a concise summary of how the water cycle moves moisture across continents.",
    "Explain why oil and water separate into two layers in a glass.",
    "What happens during an eclipse of the Moon?",
    "Explain how public libraries preserve both physical books and local archives.",
    "Describe two advantages of rail freight compared to long-haul trucking.",
    "Why do musical strings change pitch when tightened?",
    "Summarize how composting turns kitchen scraps into fertile soil.",
    "Explain why mountain hikers need more water at high altitudes.",
    "Describe the difference between a meteor, a meteoroid, and a meteorite.",
    "Why do deciduous trees drop their leaves before winter?",
    "Explain how eyeglasses bend light to correct nearsightedness.",
    "Compare thermal insulation in double-pane windows versus single-pane glass.",
    "Why does salt help melt ice on winter roads?",
    "Describe how wind turbines convert moving air into electrical energy.",
    "Explain what causes echolocation bats to distinguish small insects in darkness.",
    "Why do cast-iron pans hold heat longer than thin aluminum skillets?",
    "Summarize how bees transfer pollen between flowering plants.",
    "Explain why deep-sea trenches remain near freezing despite nearby volcanoes.",
    "Describe how fountain pens draw ink from the reservoir to the paper.",
    "Why do airplanes fly near the tropopause for long-haul routes?",
    "Explain the difference between a primary source and a secondary source in history.",
]


def subject_to_domain(subject: str) -> str:
    if subject in SUBJECT_TO_DOMAIN:
        return SUBJECT_TO_DOMAIN[subject]
    low = subject.strip().lower()
    if "intermediate" in low or "precalculus" in low or "calculus" in low:
        return "advanced_algebra"
    if "algebra" in low:
        return "algebra"
    return "discrete_geom"


def stratify_math500_rows(
    all_rows: list[dict[str, Any]],
    *,
    n_train: int = 300,
    n_test: int = 200,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratify MATH-500 rows strictly by `(subject, level)` into disjoint train & test sets."""
    total_need = int(n_train) + int(n_test)
    assert total_need <= len(all_rows), (total_need, len(all_rows))

    by_stratum: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(all_rows):
        subj = str(row.get("subject", "Unknown"))
        lvl = int(row.get("level", 0))
        item = {
            "id": str(row.get("unique_id", f"math500_{idx}")),
            "uid": idx,
            "problem": str(row["problem"]),
            "answer": str(row["answer"]),
            "solution": str(row.get("solution", "")),
            "subject": subj,
            "domain": subject_to_domain(subj),
            "level": lvl,
        }
        by_stratum[(subj, lvl)].append(item)

    rng = random.Random(int(seed))
    strata = sorted(by_stratum.keys())
    for st in strata:
        rng.shuffle(by_stratum[st])

    # Pool proportional allocation with Hamilton / largest-remainder quota so train/test stay exact
    test_frac = float(n_test) / float(len(all_rows))
    train_pool: list[dict[str, Any]] = []
    test_pool: list[dict[str, Any]] = []

    raw_quotas = []
    for st in strata:
        grp = by_stratum[st]
        exact_q = len(grp) * test_frac
        base_q = int(exact_q)
        raw_quotas.append((exact_q - base_q, st, base_q, len(grp)))

    raw_quotas.sort(key=lambda x: (-x[0], x[1]))
    test_counts = {st: base_q for _, st, base_q, _ in raw_quotas}
    rem = int(round(len(all_rows) * test_frac)) - sum(test_counts.values())
    for _, st, _, grp_len in raw_quotas:
        if rem <= 0:
            break
        if test_counts[st] < grp_len:
            test_counts[st] += 1
            rem -= 1

    for st in strata:
        grp = by_stratum[st]
        k_te = test_counts[st]
        test_pool.extend(grp[:k_te])
        train_pool.extend(grp[k_te:])

    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    test_rows = test_pool[:n_test]
    train_rows = train_pool[:n_train]
    assert {r["uid"] for r in test_rows}.isdisjoint({r["uid"] for r in train_rows})
    return train_rows, test_rows


def load_math500_split(*, n_train: int = 300, n_test: int = 200, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset"))
    all_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return stratify_math500_rows(all_rows, n_train=n_train, n_test=n_test, seed=seed)


def load_mmlu_split(*, n_test: int = 400, n_router_train: int = 200, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_items = ec._load_mmlu_stratified(int(n_test) + int(n_router_train), int(seed))
    for it in all_items:
        it["is_math_subject"] = str(it.get("subject", "")).lower() in MMLU_MATH_SUBJECTS
    return all_items[:n_test], all_items[n_test : n_test + n_router_train]


def format_math500_prompt_ids(tok: Any, problem: str) -> list[int]:
    content = problem.strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
    return _format_messages(tok, [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=False)


def format_mmlu_prompt_ids(tok: Any, item: dict[str, Any]) -> list[int]:
    letters = ("A", "B", "C", "D")
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item["choices"][:4]))
    subj = str(item.get("subject", "general")).replace("_", " ")
    content = (
        f"Multiple-choice question ({subj}):\n"
        f"{item['question']}\n\n"
        f"{opts}\n\n"
        "Respond with only the single capital letter of the correct option (A, B, C, or D)."
    )
    return _format_messages(tok, [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=False)


@contextmanager
def _temp_apply_weights(model: nn.Module, param_updates: dict[str, torch.Tensor] | None) -> Iterator[None]:
    """Temporarily apply parameter updates to `model`, always restoring bit-exact original weights."""
    if not param_updates:
        yield
        return
    p_map = dict(model.named_parameters())
    norm_map = {_norm_param_key(k): v for k, v in p_map.items()}
    saved_cpu: list[tuple[nn.Parameter, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for k, val in param_updates.items():
                p = p_map.get(k)
                if p is None:
                    p = norm_map[_norm_param_key(k)]
                saved_cpu.append((p, p.detach().cpu().clone()))
                p.data.copy_(val.to(device=p.device, dtype=p.dtype))
        yield
    finally:
        with torch.no_grad():
            for p, orig_cpu in saved_cpu:
                p.data.copy_(orig_cpu.to(device=p.device, dtype=p.dtype))


@torch.no_grad()
def _score_mmlu_fast(
    model: nn.Module,
    prompt_ids_list: list[list[int]],
    items: list[dict[str, Any]],
    letter_ids: dict[str, list[int]],
    dev: torch.device,
) -> dict[str, Any]:
    """Compute exact unpadded next-token A/B/C/D distributions and predictions on MMLU prompts."""
    letters = ("A", "B", "C", "D")
    probs_out = torch.zeros((len(items), 4), dtype=torch.float32)
    preds_out: list[int] = []
    correct_out: list[bool] = []
    masses_out: list[float] = []

    was_training = bool(model.training)
    model.eval()
    try:
        for i, (pids, item) in enumerate(zip(prompt_ids_list, items)):
            inp = torch.tensor(pids, dtype=torch.long, device=dev).unsqueeze(0)
            logits = _tail_logits(model, inp, max(0, len(pids) - 1))[0].float()
            lp = F.log_softmax(logits, dim=-1)
            scores = []
            for ch in letters:
                tids = letter_ids.get(ch) or []
                if tids:
                    scores.append(torch.logsumexp(lp[tids], dim=0))
                else:
                    scores.append(torch.tensor(-1e9, dtype=torch.float32, device=dev))
            s_t = torch.stack(scores).cpu()
            masses_out.append(float(s_t.exp().sum().item()))
            c_probs = F.softmax(s_t, dim=0)
            pred = int(s_t.argmax().item())
            probs_out[i] = c_probs
            preds_out.append(pred)
            correct_out.append(pred == int(item["gold_idx"]))
            del inp, logits, lp
    finally:
        if was_training:
            model.train()

    return {
        "probs": probs_out,
        "preds": preds_out,
        "correct": correct_out,
        "choice_masses": masses_out,
    }


def _summarize_mmlu(
    items: list[dict[str, Any]],
    score_dict: dict[str, Any],
    *,
    base_score: dict[str, Any] | None = None,
    fire_flags: list[bool] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    probs = score_dict["probs"]
    preds = score_dict["preds"]
    correct = score_dict["correct"]
    masses = score_dict["choice_masses"]
    if fire_flags is None:
        fire_flags = [False] * len(items)

    def _subset_rep(idxs: list[int], tag_seed: int) -> dict[str, Any]:
        n = len(idxs)
        sub_c = [correct[i] for i in idxs]
        sub_p = [preds[i] for i in idxs]
        sub_f = [fire_flags[i] for i in idxs]
        n_corr = sum(1 for x in sub_c if x)
        n_fired = sum(1 for x in sub_f if x)
        out: dict[str, Any] = {
            "n_total": n,
            "n_correct": n_corr,
            "accuracy": round(n_corr / max(1, n), 4),
            "mean_abcd_prob_mass": round(sum(masses[i] for i in idxs) / max(1, n), 4),
            "router_fire_rate": round(n_fired / max(1, n), 4),
            "router_fired_count": int(n_fired),
        }
        if base_score is not None:
            b_probs = base_score["probs"]
            b_preds = base_score["preds"]
            b_corr = base_score["correct"]
            kls = [
                max(
                    0.0,
                    float(
                        F.kl_div(
                            probs[i].clamp_min(1e-12).log(),
                            b_probs[i].clamp_min(1e-12),
                            reduction="sum",
                        ).item()
                    ),
                )
                for i in idxs
            ]
            exact_m = sum(1 for i in idxs if preds[i] == b_preds[i]) / max(1, n)
            pb = _paired_bootstrap(
                [{"strict_correct": correct[i]} for i in idxs],
                [{"strict_correct": b_corr[i]} for i in idxs],
                seed=tag_seed,
            )
            out.update({
                "delta_vs_base": pb["diff"],
                "ci95_vs_base": pb["ci95"],
                "gained": pb["wins"],
                "lost": pb["losses"],
                "exact_pred_match_vs_base": round(exact_m, 4),
                "mean_choice_kl": round(sum(kls) / max(1, n), 6),
                "paired_vs_base": pb,
            })
        return out

    all_idxs = list(range(len(items)))
    non_math_idxs = [i for i, it in enumerate(items) if not it.get("is_math_subject", False)]
    rep_all = _subset_rep(all_idxs, seed)
    rep_nm = _subset_rep(non_math_idxs, seed + 17) if non_math_idxs else dict(rep_all)
    return {
        "all_400": rep_all,
        "non_math": rep_nm,
        **{k: rep_all[k] for k in ("router_fire_rate", "accuracy", "n_correct", "n_total") if k in rep_all},
        **({k: rep_all[k] for k in ("delta_vs_base", "mean_choice_kl", "exact_pred_match_vs_base", "paired_vs_base") if k in rep_all}),
    }


def _compose_routed_mmlu(
    items: list[dict[str, Any]],
    routed_classes: list[str],
    base_score: dict[str, Any],
    domain_scores: dict[str, dict[str, Any]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Compose exact MMLU predictions by choosing `base_score` on passthrough or domain score when fired."""
    n = len(items)
    probs = base_score["probs"].clone()
    preds = list(base_score["preds"])
    correct = list(base_score["correct"])
    masses = list(base_score["choice_masses"])
    fire_flags = [cls_nm in domain_scores for cls_nm in routed_classes]

    for i, cls_nm in enumerate(routed_classes):
        if cls_nm in domain_scores:
            ds = domain_scores[cls_nm]
            probs[i] = ds["probs"][i]
            preds[i] = ds["preds"][i]
            correct[i] = ds["correct"][i]
            masses[i] = ds["choice_masses"][i]

    composed = {"probs": probs, "preds": preds, "correct": correct, "choice_masses": masses}
    return _summarize_mmlu(items, composed, base_score=base_score, fire_flags=fire_flags, seed=seed)


def _grade_math500_rows(
    items: list[dict[str, Any]],
    raw_outs: list[dict[str, Any]],
    tok: Any,
    eos_set: set[int],
    max_tokens: int,
    *,
    routed_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (it, raw) in enumerate(zip(items, raw_outs)):
        g = dm.grade_math(raw["token_ids"], raw["finish_reason"], str(it["answer"]), tok, eos_set, int(max_tokens))
        rows.append({
            "id": it["id"],
            "domain": it["domain"],
            "subject": it["subject"],
            "level": int(it["level"]),
            "routed_class": routed_labels[i] if routed_labels is not None else None,
            "strict_correct": bool(g["correct"]),
            "loop_killed": bool(g["looped_killed"]),
            "truncated": bool(g["truncated"]),
            "pred": g["pred"],
            "gold": g["gold"],
            "total_tokens": int(g["n_tokens"]),
        })
    return rows


def _summarize_math500_arm(
    rows: list[dict[str, Any]],
    *,
    teacher_correct_flags: list[bool] | None = None,
    base_acc: float | None = None,
    teacher_acc: float | None = None,
) -> dict[str, Any]:
    n = len(rows)
    n_corr = sum(1 for r in rows if r["strict_correct"])
    acc = n_corr / max(1, n)
    toks = [r["total_tokens"] for r in rows]

    per_domain = {
        d: round(
            sum(1 for r in rows if r["domain"] == d and r["strict_correct"])
            / max(1, sum(1 for r in rows if r["domain"] == d)),
            4,
        )
        for d in MATH_DOMAINS
    }
    levels = sorted({int(r["level"]) for r in rows})
    per_level = {
        f"L{lv}": round(
            sum(1 for r in rows if int(r["level"]) == lv and r["strict_correct"])
            / max(1, sum(1 for r in rows if int(r["level"]) == lv)),
            4,
        )
        for lv in levels
    }

    t_subset_acc = None
    if teacher_correct_flags is not None:
        t_idxs = [i for i, tc in enumerate(teacher_correct_flags) if tc]
        if t_idxs:
            t_subset_acc = round(sum(1 for i in t_idxs if rows[i]["strict_correct"]) / len(t_idxs), 4)

    gap_rec = None
    if base_acc is not None and teacher_acc is not None and teacher_acc > base_acc:
        gap_rec = round((acc - base_acc) / max(1e-6, teacher_acc - base_acc), 4)

    return {
        "accuracy": round(acc, 4),
        "n_correct": n_corr,
        "n_total": n,
        "truncated_rate": round(sum(1 for r in rows if r["truncated"]) / max(1, n), 4),
        "loop_killed_rate": round(sum(1 for r in rows if r["loop_killed"]) / max(1, n), 4),
        "median_tokens": round(float(statistics.median(toks)), 1) if toks else 0.0,
        "mean_tokens": round(sum(toks) / max(1, n), 1),
        "per_domain": per_domain,
        "per_level": per_level,
        "teacher_solved_subset_acc": t_subset_acc,
        "teacher_gap_recovered": gap_rec,
    }


def run_smoke() -> dict[str, Any]:
    """Fast deterministic CPU smoke check covering MATH-500 split/grading, MMLU fast scoring, routing, and LoRA training."""
    # 1. Stratified MATH-500 split over synthetic 500 problems across 7 subjects x 5 levels
    subjects = list(SUBJECT_TO_DOMAIN.keys())
    synth_rows = [
        {
            "unique_id": f"m_{i}",
            "problem": f"Problem {i}",
            "answer": f"\\frac{{{i % 7 + 1}}}{{2}}",
            "solution": "Step 1 \\boxed{1}.",
            "subject": subjects[i % len(subjects)],
            "level": (i % 5) + 1,
        }
        for i in range(500)
    ]
    tr, te = stratify_math500_rows(synth_rows, n_train=300, n_test=200, seed=42)
    assert len(tr) == 300 and len(te) == 200
    assert {r["domain"] for r in te} == set(MATH_DOMAINS)
    assert dm._math_equal("\\dfrac{1}{2}", "\\frac{1}{2}")
    assert dm._math_equal("$12.0$", "12")
    try:
        import antlr4  # noqa: F401
        assert dm._math_equal("\\frac{1}{2}", "0.5")
    except ImportError:
        pass

    # 2. Check _PlantedReasoningLM + _temp_apply_weights + _score_mmlu_fast + _compose_routed_mmlu + LoRA train
    torch.manual_seed(11)
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    mmlu_items = [
        {"question": f"q {i}", "choices": ["a", "b", "c", "d"], "gold_idx": i % 4, "subject": "college_mathematics" if i < 2 else "world_history", "is_math_subject": i < 2}
        for i in range(6)
    ]
    mmlu_pids = [[1, 2, 4, 10 + (i % 4)] for i in range(6)]
    letter_ids = {"A": [20], "B": [21], "C": [25], "D": [2]}

    base_sc = _score_mmlu_fast(model, mmlu_pids, mmlu_items, letter_ids, torch.device("cpu"))
    assert base_sc["probs"].shape == (6, 4)

    # Test _temp_apply_weights restores bit-exact weights even after mutating lm_head
    patched = {"lm_head.weight": model.lm_head.weight.detach().clone() + 0.25}
    with _temp_apply_weights(model, patched):
        mod_sc = _score_mmlu_fast(model, mmlu_pids, mmlu_items, letter_ids, torch.device("cpu"))
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    routed_cls = ["algebra", "passthrough", "passthrough", "passthrough", "discrete_geom", "passthrough"]
    mmlu_comp = _compose_routed_mmlu(
        mmlu_items,
        routed_cls,
        base_sc,
        {"algebra": mod_sc, "discrete_geom": mod_sc},
    )
    assert abs(mmlu_comp["router_fire_rate"] - (2.0 / 6.0)) < 1e-4, mmlu_comp

    # 3. Verify LoRA training + stratified random layers on _PlantedReasoningLM
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
    rand_lyrs = pick_stratified_random_layers(model, [0, 1], seed=700)
    res_lora = _train_lora_expert(model, rand_lyrs, train_samples, rank=2, alpha_scale=4.0, epochs=2, lr_val=2e-2, grad_accum=2)
    assert res_lora["epoch_losses"][-1] < res_lora["epoch_losses"][0], res_lora["epoch_losses"]
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())

    # 4. Verify _summarize_math500_arm
    fake_rows = [
        {"id": f"t_{i}", "domain": MATH_DOMAINS[i % 3], "subject": "Algebra", "level": (i % 5) + 1, "routed_class": MATH_DOMAINS[i % 3], "strict_correct": i < 4, "loop_killed": False, "truncated": i == 5, "pred": "1", "gold": "1", "total_tokens": 200 + 20 * i}
        for i in range(6)
    ]
    t_flags = [True, True, True, True, True, False]
    ms = _summarize_math500_arm(fake_rows, teacher_correct_flags=t_flags, base_acc=0.3333, teacher_acc=0.8333)
    assert ms["accuracy"] == 0.6667 and ms["teacher_solved_subset_acc"] == 0.8

    return {
        "mode": "smoke",
        "passed": True,
        "math500_counts": {"train": len(tr), "test": len(te)},
        "mmlu_composed_fire_rate": mmlu_comp["router_fire_rate"],
        "lora_losses": res_lora["epoch_losses"],
        "math500_summary_check": ms,
    }


def _try_load_teacher_cache(
    cache_path: Path | None,
    *,
    prompt_sig: str,
    seed: int,
    n_train: int,
    n_test: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            data.get("prompt_sig") == prompt_sig
            and int(data.get("seed", -1)) == int(seed)
            and int(data.get("n_train", -1)) == int(n_train)
            and int(data.get("n_test", -1)) == int(n_test)
            and int(data.get("max_new_tokens", -1)) == int(max_new_tokens)
            and len(data.get("test_outputs", [])) == int(n_test)
            and len(data.get("train_outputs", [])) == int(n_train)
        ):
            print(f"[math500] Reused cached teacher_4b outputs from {cache_path}", flush=True)
            return data["test_outputs"], data["train_outputs"]
    except Exception as err:  # noqa: BLE001
        print(f"[math500] Teacher cache mismatch/unreadable ({type(err).__name__}); regenerating.", flush=True)
    return None


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

    math_train, math_test = load_math500_split(n_train=args.n_math_train, n_test=args.n_math_test, seed=args.seed)
    mmlu_test, mmlu_router_train = load_mmlu_split(n_test=args.n_mmlu_test, n_router_train=args.n_mmlu_router_train, seed=args.seed)

    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    assert len(tok_t) == len(tok_s), f"Tokenizer vocab mismatch: teacher={len(tok_t)} vs student={len(tok_s)}"
    eos_set = set(lr.eos_ids(tok_s))

    # ---- Phase 1: Teacher 4B generation (+ Drive cache) & Top-32 logit extraction ----
    print(f"[math500] Phase 1: Teacher ({args.teacher}) generation & Top-32 logits (cap={args.max_teacher_train_tokens} tok)...", flush=True)
    t_p1 = time.perf_counter()

    math_test_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_test]
    math_train_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_train]
    test_seeds = [int(args.seed) + i for i in range(len(math_test))]
    train_seeds = [int(args.seed) + 1000 + i for i in range(len(math_train))]

    sig_hash = hashlib.sha256(
        json.dumps([math_test_pids, math_train_pids], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    cache_p = Path(args.teacher_cache) if args.teacher_cache else None
    cached = _try_load_teacher_cache(
        cache_p,
        prompt_sig=sig_hash,
        seed=args.seed,
        n_train=len(math_train),
        n_test=len(math_test),
        max_new_tokens=args.max_new_tokens,
    )

    if cached is not None:
        t_test_raw, t_train_raw = cached
    else:
        if backend == "vllm":
            t_outs = _vllm_generate(
                args.teacher,
                tok_t,
                math_test_pids + math_train_pids,
                test_seeds + train_seeds,
                max_new_tokens=args.max_new_tokens,
                dtype_str=dtype_vllm_str,
                base_seed=args.seed,
            )
            t_test_raw = t_outs[: len(math_test)]
            t_train_raw = t_outs[len(math_test) :]
        else:
            try:
                teacher_m = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
            except TypeError:
                teacher_m = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
            t_test_raw = _hf_generate_subset(teacher_m, tok_t, math_test_pids, test_seeds, max_new_tokens=args.max_new_tokens)
            t_train_raw = _hf_generate_subset(teacher_m, tok_t, math_train_pids, train_seeds, max_new_tokens=args.max_new_tokens)
            del teacher_m
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        if cache_p is not None:
            _atomic_write_json(
                cache_p,
                {
                    "prompt_sig": sig_hash,
                    "seed": int(args.seed),
                    "n_train": len(math_train),
                    "n_test": len(math_test),
                    "max_new_tokens": int(args.max_new_tokens),
                    "test_outputs": t_test_raw,
                    "train_outputs": t_train_raw,
                },
            )
            print(f"[math500] Saved teacher_4b cache -> {cache_p}", flush=True)

    teacher_test_rows = _grade_math500_rows(math_test, t_test_raw, tok_s, eos_set, args.max_new_tokens)
    teacher_correct_flags = [bool(r["strict_correct"]) for r in teacher_test_rows]
    teacher_summary = _summarize_math500_arm(teacher_test_rows, teacher_correct_flags=teacher_correct_flags)
    print(f"[math500] teacher_4b test={teacher_summary}", flush=True)

    # Pure self-distillation: keep ONLY correct 4B solutions <= max_teacher_train_tokens
    train_by_domain_kept: dict[str, list[tuple[dict[str, Any], list[int], str, bool]]] = {d: [] for d in MATH_DOMAINS}
    fallback_candidates: dict[str, list[tuple[dict[str, Any], list[int], str, bool]]] = {d: [] for d in MATH_DOMAINS}
    tot_by_domain = {d: 0 for d in MATH_DOMAINS}
    for it, p_ids, raw in zip(math_train, math_train_pids, t_train_raw):
        d = it["domain"]
        tot_by_domain[d] += 1
        g = dm.grade_math(raw["token_ids"], raw["finish_reason"], str(it["answer"]), tok_s, eos_set, args.max_new_tokens)
        body = lr.strip_eos(raw["token_ids"], eos_set)
        txt = tok_s.decode(body, skip_special_tokens=True).strip()
        if g["correct"] and (not g["looped_killed"]) and g["n_tokens"] <= int(args.max_teacher_train_tokens):
            train_by_domain_kept[d].append((it, p_ids, txt, False))
        elif it["solution"].strip():
            sol_txt = it["solution"].strip()
            if "\\boxed{" not in sol_txt:
                sol_txt += f"\n\nFinal answer: \\boxed{{{it['answer']}}}."
            fallback_candidates[d].append((it, p_ids, sol_txt, True))

    fallback_counts = {d: 0 for d in MATH_DOMAINS}
    for d in MATH_DOMAINS:
        if len(train_by_domain_kept[d]) < 16 and fallback_candidates[d]:
            need = min(16 - len(train_by_domain_kept[d]), len(fallback_candidates[d]))
            train_by_domain_kept[d].extend(fallback_candidates[d][:need])
            fallback_counts[d] = need

    teacher_train_kept_frac = {
        d: round(sum(1 for x in train_by_domain_kept[d] if not x[3]) / max(1, tot_by_domain[d]), 4)
        for d in MATH_DOMAINS
    }
    print(f"[math500] teacher_4b kept frac={teacher_train_kept_frac} | fallback_counts={fallback_counts}", flush=True)

    try:
        teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    train_samples_by_domain: dict[str, list[dict[str, Any]]] = {d: [] for d in MATH_DOMAINS}
    with torch.no_grad():
        for d in MATH_DOMAINS:
            for it, p_ids, resp_txt, is_fb in train_by_domain_kept[d]:
                full_ids = _format_messages(
                    tok_s,
                    [{"role": "user", "content": it["problem"].strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."}, {"role": "assistant", "content": resp_txt}],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                lo = max(0, len(p_ids) - 1)
                hi = len(full_ids) - 1
                if hi <= lo:
                    continue
                ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
                t_logits = _tail_logits(teacher_hf, ids_dev, lo)[: hi - lo].float()
                vals, idx = t_logits.topk(32, dim=-1)
                train_samples_by_domain[d].append({
                    "id": it["id"],
                    "domain": d,
                    "subject": it["subject"],
                    "level": int(it["level"]),
                    "is_fallback": bool(is_fb),
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

    # ---- Phase 2: Load Student 0.8B, WSI Causal Layers, Stratified Random Layers, Router & Base MMLU ----
    print(f"[math500] Phase 2: Loading Student ({args.student}), WSI localization, Layer-12 router & base MMLU...", flush=True)
    t_p2 = time.perf_counter()
    try:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in student_m.parameters():
        p.requires_grad_(False)

    _, blocks = find_blocks(student_m)
    n_layers = len(blocks)
    causal_traces_by_dom: dict[str, list[CompactTrace]] = {d: [] for d in MATH_DOMAINS}
    for d in MATH_DOMAINS:
        # Spread 6 trace captures evenly across kept samples (diverse subjects/levels)
        dom_sps = train_samples_by_domain[d]
        step = max(1, len(dom_sps) // 6)
        picked_sps = dom_sps[::step][:6]
        for sp in picked_sps:
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
            causal_traces_by_dom[d].append(ct)

    invariance = compute_invariance_metrics(causal_traces_by_dom, top_k_sites=4, top_n_neurons=32)
    causal_layers: dict[str, list[int]] = {}
    random_layers_micro: dict[str, list[int]] = {}
    for d_idx, d in enumerate(MATH_DOMAINS):
        diag_d = diagnose_causal_traces(causal_traces_by_dom[d], split_ab=True, top_k_sites=16, center="healthy", healthy_cost="fisher", model_or_weights=student_m)
        causal_layers[d] = _extract_causal_layers(diag_d, n_layers, max_layers=8)
        random_layers_micro[d] = pick_stratified_random_layers(student_m, causal_layers[d][:4], seed=args.seed + 700 + d_idx)

    all_traces = [ct for d in MATH_DOMAINS for ct in causal_traces_by_dom[d]]
    diag_glob = diagnose_causal_traces(all_traces, split_ab=True, top_k_sites=16, center="healthy", healthy_cost="fisher", model_or_weights=student_m)
    global_causal_layers = _extract_causal_layers(diag_glob, n_layers, max_layers=8)
    print(f"[math500] WSI causal layers={causal_layers} | global_8={global_causal_layers} | random_micro={random_layers_micro}", flush=True)

    random_layers_report = {
        d: {
            "layers": random_layers_micro[d],
            "wsi_layers": causal_layers[d][:4],
            "overlap_with_wsi": sorted(set(random_layers_micro[d]) & set(causal_layers[d][:4])),
            "n_overlap": len(set(random_layers_micro[d]) & set(causal_layers[d][:4])),
        }
        for d in MATH_DOMAINS
    }

    router_layer = n_layers // 2
    mmlu_r_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_router_train]
    mmlu_test_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_test]
    prose_r_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_ROUTER_TRAIN]
    prose_held_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_HELDOUT]

    r_train_pids = math_train_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = [it["domain"] for it in math_train] + ["passthrough"] * (len(mmlu_r_pids) + len(prose_r_pids))

    X_r_train = _extract_prompt_hidden_vecs(student_m, r_train_pids, router_layer)
    W_r, b_r = fit_ridge_router(X_r_train, r_train_labels, ROUTER_CLASSES, lam=1.0)

    X_math_test = _extract_prompt_hidden_vecs(student_m, math_test_pids, router_layer)
    X_mmlu_test = _extract_prompt_hidden_vecs(student_m, mmlu_test_pids, router_layer)
    X_prose_test = _extract_prompt_hidden_vecs(student_m, prose_held_pids, router_layer)

    routed_math_test = predict_ridge_router(X_math_test, W_r, b_r, ROUTER_CLASSES)
    routed_mmlu_test = predict_ridge_router(X_mmlu_test, W_r, b_r, ROUTER_CLASSES)
    routed_prose_test = predict_ridge_router(X_prose_test, W_r, b_r, ROUTER_CLASSES)

    router_report = {
        "router_layer": router_layer,
        "math_exact_domain_acc": round(sum(p == it["domain"] for p, it in zip(routed_math_test, math_test)) / len(math_test), 4),
        "math_any_domain_fire_rate": round(sum(p in MATH_DOMAINS for p in routed_math_test) / len(math_test), 4),
        "math_passthrough_miss_rate": round(sum(p == "passthrough" for p in routed_math_test) / len(math_test), 4),
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_fire_rate": round(sum(p != "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_non_math_fire_rate": round(
            sum(p != "passthrough" for p, it in zip(routed_mmlu_test, mmlu_test) if not it["is_math_subject"])
            / max(1, sum(1 for it in mmlu_test if not it["is_math_subject"])),
            4,
        ),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_test) / len(routed_prose_test), 4),
    }
    print(f"[math500] Router report: {router_report}", flush=True)

    letter_ids = ec._resolve_mmlu_letter_ids(tok_s)
    base_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
    base_mmlu_summary = _summarize_mmlu(mmlu_test, base_mmlu_raw, seed=args.seed)
    print(f"[math500] base_08b MMLU: all_400={base_mmlu_summary['all_400']['accuracy']} | non_math={base_mmlu_summary['non_math']['accuracy']}", flush=True)

    prose_full_ids = _to_seq_tensors(
        [
            _format_messages(
                tok_s,
                [{"role": "user", "content": p}, {"role": "assistant", "content": "Here is a concise explanation."}],
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for p in PROSE_HELDOUT
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

    # ---- Phase 3: Train 5 LoRA Configurations & Evaluate MMLU Retention ----
    print("[math500] Phase 3: Training Pocket MoE tiers & global controls + scoring MMLU retention...", flush=True)
    t_p3 = time.perf_counter()

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        expert_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {
            "moe_micro_wsi": {},
            "moe_micro_random": {},
            "moe_large_wsi": {},
            "global_lora_r64": {},
            "global_lora_bank_matched": {},
        }
        domain_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {
            "moe_micro_wsi": {},
            "moe_micro_random": {},
            "moe_large_wsi": {},
        }

        routed_specs = [
            ("moe_micro_wsi", causal_layers, 4, 8, 16.0, 5e-4, 0, False),
            ("moe_micro_random", random_layers_micro, 4, 8, 16.0, 5e-4, 100, True),
            ("moe_large_wsi", causal_layers, 8, 64, 64.0, 3e-4, 200, False),
        ]

        for d_idx, d in enumerate(MATH_DOMAINS):
            samples_d = train_samples_by_domain[d]
            for tier_nm, lmap, k_lyrs, rk, a_sc, lr_v, sd_off, is_rand in routed_specs:
                lyrs = list(lmap[d][:k_lyrs])
                res_e = _train_lora_expert(
                    student_m,
                    lyrs,
                    samples_d,
                    rank=rk,
                    alpha_scale=a_sc,
                    epochs=3,
                    lr_val=lr_v,
                    grad_accum=4,
                    healthy_ids=prose_full_ids,
                    base_healthy_logp=base_prose_logp,
                    base_healthy_nll=base_prose_nll,
                    seed=args.seed + sd_off + d_idx,
                )
                upd_w = res_e.pop("weights")
                with _temp_apply_weights(student_m, upd_w):
                    sc_d = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
                domain_mmlu_scores[tier_nm][d] = sc_d
                res_e["unrouted_mmlu_acc"] = round(sum(1 for x in sc_d["correct"] if x) / len(mmlu_test), 4)
                if is_rand:
                    res_e["overlap_with_wsi"] = len(set(lyrs) & set(causal_layers[d][:k_lyrs]))
                p_exp = exp_dir / f"{tier_nm}_{d}.pt"
                torch.save(upd_w, p_exp)
                del upd_w
                expert_paths[(tier_nm, d)] = p_exp
                expert_meta[tier_nm][d] = res_e
                print(
                    f"[math500] trained {tier_nm}/{d}: {res_e['adapter_mb']}MB (n={len(samples_d)}) | "
                    f"losses={res_e['epoch_losses']} | unrouted_kl={res_e['healthy_kl']} | unrouted_mmlu={res_e['unrouted_mmlu_acc']}",
                    flush=True,
                )

        all_kept_flat = [sp for d in MATH_DOMAINS for sp in train_samples_by_domain[d]]

        res_glob_r64 = _train_lora_expert(
            student_m,
            global_causal_layers[:8],
            all_kept_flat,
            rank=64,
            alpha_scale=64.0,
            epochs=3,
            lr_val=3e-4,
            grad_accum=4,
            healthy_ids=prose_full_ids,
            base_healthy_logp=base_prose_logp,
            base_healthy_nll=base_prose_nll,
            seed=args.seed + 500,
        )
        upd_g64 = res_glob_r64.pop("weights")
        with _temp_apply_weights(student_m, upd_g64):
            glob_r64_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
        p_glob_r64 = exp_dir / "global_lora_r64.pt"
        torch.save(upd_g64, p_glob_r64)
        del upd_g64
        expert_meta["global_lora_r64"] = res_glob_r64
        print(
            f"[math500] trained global_lora_r64: {res_glob_r64['adapter_mb']}MB | "
            f"losses={res_glob_r64['epoch_losses']} | unrouted_kl={res_glob_r64['healthy_kl']}",
            flush=True,
        )

        res_glob_bank = _train_lora_expert(
            student_m,
            list(range(n_layers)),
            all_kept_flat,
            rank=64,
            alpha_scale=64.0,
            epochs=3,
            lr_val=3e-4,
            grad_accum=4,
            healthy_ids=prose_full_ids,
            base_healthy_logp=base_prose_logp,
            base_healthy_nll=base_prose_nll,
            seed=args.seed + 600,
        )
        upd_gbank = res_glob_bank.pop("weights")
        with _temp_apply_weights(student_m, upd_gbank):
            glob_bank_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
        p_glob_bank = exp_dir / "global_lora_bank_matched.pt"
        torch.save(upd_gbank, p_glob_bank)
        del upd_gbank
        expert_meta["global_lora_bank_matched"] = res_glob_bank
        print(
            f"[math500] trained global_lora_bank_matched: {res_glob_bank['adapter_mb']}MB | "
            f"losses={res_glob_bank['epoch_losses']} | unrouted_kl={res_glob_bank['healthy_kl']}",
            flush=True,
        )

        mmlu_arms: dict[str, Any] = {
            "base_08b": base_mmlu_summary,
            "moe_micro_wsi_routed": _compose_routed_mmlu(mmlu_test, routed_mmlu_test, base_mmlu_raw, domain_mmlu_scores["moe_micro_wsi"], seed=args.seed + 1),
            "moe_micro_random_routed": _compose_routed_mmlu(mmlu_test, routed_mmlu_test, base_mmlu_raw, domain_mmlu_scores["moe_micro_random"], seed=args.seed + 2),
            "moe_large_wsi_routed": _compose_routed_mmlu(mmlu_test, routed_mmlu_test, base_mmlu_raw, domain_mmlu_scores["moe_large_wsi"], seed=args.seed + 3),
            "routed_single_math_lora_r64": _compose_routed_mmlu(
                mmlu_test,
                routed_mmlu_test,
                base_mmlu_raw,
                {d: glob_r64_mmlu_raw for d in MATH_DOMAINS},
                seed=args.seed + 4,
            ),
            "global_lora_r64": _summarize_mmlu(mmlu_test, glob_r64_mmlu_raw, base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=args.seed + 5),
            "global_lora_bank_matched": _summarize_mmlu(mmlu_test, glob_bank_mmlu_raw, base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=args.seed + 6),
        }
        for k_m, vm in mmlu_arms.items():
            print(
                f"[math500] MMLU {k_m:28s} | all400={vm['all_400']['accuracy']} "
                f"| non_math={vm['non_math']['accuracy']} | fire={vm['router_fire_rate']}",
                flush=True,
            )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Evaluate MATH-500 (n=200) via Baked vLLM Eager Checkpoints ----
        print(f"[math500] Phase 4: Evaluating MATH-500 (n={len(math_test)}, {backend}, max_new_tokens={args.max_new_tokens})...", flush=True)
        t_p4 = time.perf_counter()

        if backend == "vllm":
            del student_m
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            base_raw = _vllm_generate(args.student, tok_s, math_test_pids, test_seeds, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
        else:
            base_raw = _hf_generate_subset(student_m, tok_s, math_test_pids, test_seeds, max_new_tokens=args.max_new_tokens)

        bake_tmp_dir = Path("/content/_pocket_moe_math500_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

        def _eval_ckpt(weights_path: Path, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
            upd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if backend == "vllm":
                _bake_full_checkpoint(Path(args.student), upd, bake_tmp_dir)
                del upd
                gc.collect()
                return _vllm_generate(str(bake_tmp_dir), tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
            return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, param_updates=upd)

        glob_r64_raw = _eval_ckpt(p_glob_r64, math_test_pids, test_seeds)
        p_glob_r64.unlink(missing_ok=True)
        glob_bank_raw = _eval_ckpt(p_glob_bank, math_test_pids, test_seeds)
        p_glob_bank.unlink(missing_ok=True)

        routed_single_math_raw = [
            base_raw[i] if cls_nm == "passthrough" else glob_r64_raw[i]
            for i, cls_nm in enumerate(routed_math_test)
        ]

        routed_raw_by_tier: dict[str, list[dict[str, Any]]] = {}
        for tier_nm in ("moe_micro_wsi", "moe_micro_random", "moe_large_wsi"):
            tier_raw: list[dict[str, Any] | None] = [None] * len(math_test)
            for i, cls_nm in enumerate(routed_math_test):
                if cls_nm == "passthrough":
                    tier_raw[i] = base_raw[i]
            for d in MATH_DOMAINS:
                idxs = [i for i, cls_nm in enumerate(routed_math_test) if cls_nm == d]
                p_exp = expert_paths[(tier_nm, d)]
                if idxs:
                    sub_outs = _eval_ckpt(p_exp, [math_test_pids[i] for i in idxs], [test_seeds[i] for i in idxs])
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
        "moe_micro_wsi_routed": (routed_raw_by_tier["moe_micro_wsi"], routed_math_test, False),
        "moe_micro_random_routed": (routed_raw_by_tier["moe_micro_random"], routed_math_test, False),
        "moe_large_wsi_routed": (routed_raw_by_tier["moe_large_wsi"], routed_math_test, False),
        "global_lora_r64": (glob_r64_raw, None, False),
        "routed_single_math_lora_r64": (routed_single_math_raw, routed_math_test, False),
        "global_lora_bank_matched": (glob_bank_raw, None, False),
    }

    math500_rows: dict[str, list[dict[str, Any]]] = {}
    base_rows = _grade_math500_rows(math_test, base_raw, tok_s, eos_set, args.max_new_tokens)
    base_acc = sum(1 for r in base_rows if r["strict_correct"]) / len(base_rows)
    teacher_acc = teacher_summary["accuracy"]

    math500_arms: dict[str, dict[str, Any]] = {}
    for arm_nm, (raw_l, r_lbls, is_teacher) in arm_raw_map.items():
        if is_teacher:
            rows = teacher_test_rows
            sm = teacher_summary
        else:
            rows = base_rows if arm_nm == "base_08b" else _grade_math500_rows(math_test, raw_l, tok_s, eos_set, args.max_new_tokens, routed_labels=r_lbls)
            sm = _summarize_math500_arm(rows, teacher_correct_flags=teacher_correct_flags, base_acc=base_acc, teacher_acc=teacher_acc)
        math500_rows[arm_nm] = rows
        math500_arms[arm_nm] = sm
        print(
            f"[math500] {arm_nm:28s} | acc={sm['accuracy']*100:5.1f}% ({sm['n_correct']:3d}/{sm['n_total']}) "
            f"| gap={sm['teacher_gap_recovered']} | trunc={sm['truncated_rate']*100:4.1f}% | domains={sm['per_domain']}",
            flush=True,
        )

    def _routed_prose_kl_for_tier(tier_key: str) -> float:
        tot = 0.0
        for cls_nm in routed_prose_test:
            if cls_nm in MATH_DOMAINS:
                tot += float(expert_meta[tier_key][cls_nm]["healthy_kl"])
        return round(tot / max(1, len(routed_prose_test)), 6)

    prose_kl = {
        "moe_micro_wsi_routed": _routed_prose_kl_for_tier("moe_micro_wsi"),
        "moe_micro_random_routed": _routed_prose_kl_for_tier("moe_micro_random"),
        "moe_large_wsi_routed": _routed_prose_kl_for_tier("moe_large_wsi"),
        "routed_single_math_lora_r64": round(
            sum(1 for cls_nm in routed_prose_test if cls_nm != "passthrough") * float(expert_meta["global_lora_r64"]["healthy_kl"]) / max(1, len(routed_prose_test)),
            6,
        ),
        "global_lora_r64": float(expert_meta["global_lora_r64"]["healthy_kl"]),
        "global_lora_bank_matched": float(expert_meta["global_lora_bank_matched"]["healthy_kl"]),
    }

    comp_pairs = [
        ("moe_micro_wsi_vs_base", "moe_micro_wsi_routed", "base_08b"),
        ("moe_micro_wsi_vs_micro_random", "moe_micro_wsi_routed", "moe_micro_random_routed"),
        ("moe_large_wsi_vs_base", "moe_large_wsi_routed", "base_08b"),
        ("global_lora_r64_vs_base", "global_lora_r64", "base_08b"),
        ("routed_single_math_vs_base", "routed_single_math_lora_r64", "base_08b"),
        ("moe_large_wsi_vs_routed_single_math", "moe_large_wsi_routed", "routed_single_math_lora_r64"),
        ("moe_large_wsi_vs_global_lora_r64", "moe_large_wsi_routed", "global_lora_r64"),
        ("global_lora_bank_matched_vs_base", "global_lora_bank_matched", "base_08b"),
        ("routed_single_math_vs_global_lora_bank_matched", "routed_single_math_lora_r64", "global_lora_bank_matched"),
    ]
    paired_math500 = {
        lbl: _paired_bootstrap(math500_rows[a], math500_rows[b], seed=args.seed + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    acc_m = lambda k: float(math500_arms[k]["accuracy"])  # noqa: E731
    nm_acc = lambda k: float(mmlu_arms[k]["non_math"]["accuracy"])  # noqa: E731

    return {
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "backend": backend,
        "max_new_tokens": int(args.max_new_tokens),
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_wsi_router_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "dataset_summary": {
            "math500_train": len(math_train),
            "math500_test": len(math_test),
            "mmlu_test": len(mmlu_test),
            "mmlu_router_train": len(mmlu_router_train),
            "prose_heldout": len(PROSE_HELDOUT),
        },
        "teacher_4b_train_kept_frac": teacher_train_kept_frac,
        "teacher_4b_fallback_counts": fallback_counts,
        "invariance": invariance,
        "causal_layers": {**causal_layers, "global": global_causal_layers},
        "random_layers_micro": random_layers_report,
        "router_report": router_report,
        "expert_meta": expert_meta,
        "prose_kl": prose_kl,
        "math500_arms": math500_arms,
        "mmlu_arms": mmlu_arms,
        "paired_math500": paired_math500,
        "criteria": {
            "a_real_math500_gain_large_vs_base": bool(acc_m("moe_large_wsi_routed") >= acc_m("base_08b") + 0.05),
            "b_real_math500_wsi_beats_random_micro": bool(acc_m("moe_micro_wsi_routed") > acc_m("moe_micro_random_routed")),
            "c_specialist_moe_beats_routed_single_math": bool(acc_m("moe_large_wsi_routed") > acc_m("routed_single_math_lora_r64")),
            "d_routing_protects_non_math_mmlu_vs_bank": bool(
                nm_acc("moe_large_wsi_routed") >= nm_acc("base_08b") - 0.015
                and nm_acc("moe_large_wsi_routed") > nm_acc("global_lora_bank_matched")
            ),
            "e_zero_prose_damage_routed": bool(prose_kl["moe_large_wsi_routed"] <= 0.005),
        },
        "per_item_math500": math500_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU check")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-math-train", type=int, default=300)
    ap.add_argument("--n-math-test", type=int, default=200)
    ap.add_argument("--n-mmlu-test", type=int, default=400)
    ap.add_argument("--n-mmlu-router-train", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1800)
    ap.add_argument("--teacher-cache", default=None)
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
