"""Controlled Follow-Up for Pocket MoE (Qwen3.5-4B -> Qwen3.5-0.8B): Token Filter Fix + WSI/Random/Global Controls.

Addresses every open experimental control from `experiment_pocket_moe_ceiling.py`:
  1. Raises the teacher-trace length cap (`--max-teacher-train-tokens 1350`, up from `700`) so real `Qwen3.5-4B`
     solutions are retained on `weighted_mixture_transfer` instead of falling back to rigid templates (`95.8%` lost).
  2. Stratified Random-Layer Controls (`moe_micro_random`, `moe_large_random`): samples random layer sets with the exact
     same `(linear_attn / self_attn)` block composition and parameter byte count as WSI causal layers.
  3. Unrouted Global LoRA Controls matched in rank/capacity and total storage:
     - `global_lora_r64`: single unrouted LoRA `r=64` on 8 WSI global causal layers (~`27.4 MB`, matching 1 routed expert)
     - `global_lora_bank_matched`: single unrouted LoRA `r=64` across all `24` layers (~`82 MB`, matching `3 x 27.4 MB`)
  4. Paired 95% bootstrap CIs across both `hard_36` and `all_60`, plus exact guardrail preservation vs `base_08b`.

CPU smoke test::
    python scripts/experiment_pocket_moe_controls.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_moe_controls.py --teacher /content/teacher_4b --student /content/student_08b --out /content/pocket_moe_controls_l4.json
"""
from __future__ import annotations

import argparse
import gc
import json
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

import moi_loop_router_vllm as lr  # noqa: E402
from experiment_pocket_moe_ceiling import (  # noqa: E402
    GUARDRAIL_FAMILIES,
    HARD_FAMILIES,
    ROUTER_CLASSES,
    _bake_full_checkpoint,
    _extract_causal_layers,
    _grade_rows,
    _hf_generate_subset,
    _summarize_ceiling_arm,
    _train_lora_expert,
    _vllm_generate,
    build_ceiling_dataset,
    fit_ridge_router,
    predict_ridge_router,
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


def _block_arch_tag(blk: nn.Module) -> str:
    """Classify a transformer block by attention/mixer subtype and exact linear parameter count."""
    if hasattr(blk, "linear_attn"):
        mixer = "linear_attn"
    elif hasattr(blk, "self_attn"):
        mixer = "self_attn"
    elif hasattr(blk, "conv"):
        mixer = "conv"
    else:
        mixer = "other"
    lin_numel = sum(
        m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
        for m in blk.modules()
        if isinstance(m, nn.Linear)
    )
    return f"{mixer}:{lin_numel}"


def pick_stratified_random_layers(model: nn.Module, target_layers: list[int], seed: int) -> list[int]:
    """Sample a random layer list matching the exact block subtype mix & parameter size of `target_layers`."""
    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    by_tag: dict[str, list[int]] = {}
    for li, blk in enumerate(blocks):
        by_tag.setdefault(_block_arch_tag(blk), []).append(li)

    need_by_tag: dict[str, int] = {}
    for li in target_layers:
        tag = _block_arch_tag(blocks[li])
        need_by_tag[tag] = need_by_tag.get(tag, 0) + 1

    can_differ = any(len(by_tag[t]) > need for t, need in need_by_tag.items()) and len(target_layers) < n_layers
    target_set = set(target_layers)

    for retry in range(32):
        rng = random.Random(int(seed) + retry * 9973)
        buckets = {tag: list(idxs) for tag, idxs in by_tag.items()}
        for bucket in buckets.values():
            rng.shuffle(bucket)
        picked_by_tag: dict[str, list[int]] = {
            tag: buckets[tag][:need] for tag, need in need_by_tag.items()
        }
        # Reassemble in the same slot order as target_layers for clean logging while keeping set uniform
        cursors = {tag: 0 for tag in need_by_tag}
        out: list[int] = []
        for li in target_layers:
            tag = _block_arch_tag(blocks[li])
            c = cursors[tag]
            out.append(picked_by_tag[tag][c])
            cursors[tag] = c + 1
        if not can_differ or set(out) != target_set:
            return out
    return out


def _paired_bootstrap(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    key: str = "strict_correct",
    seed: int = 42,
    n_boot: int = 5000,
) -> dict[str, Any]:
    """Paired accuracy difference + 95% bootstrap confidence interval."""
    assert len(rows_a) == len(rows_b), (len(rows_a), len(rows_b))
    ca = [1 if bool(r[key]) else 0 for r in rows_a]
    cb = [1 if bool(r[key]) else 0 for r in rows_b]
    n = len(ca)
    wins = sum(1 for a, b in zip(ca, cb) if a == 1 and b == 0)
    losses = sum(1 for a, b in zip(ca, cb) if a == 0 and b == 1)
    both_correct = sum(1 for a, b in zip(ca, cb) if a == 1 and b == 1)
    acc_a = sum(ca) / max(1, n)
    acc_b = sum(cb) / max(1, n)

    rng = random.Random(int(seed))
    diffs: list[float] = []
    for _ in range(int(n_boot)):
        s_diff = 0
        for _j in range(n):
            idx = rng.randrange(n)
            s_diff += ca[idx] - cb[idx]
        diffs.append(s_diff / max(1, n))
    diffs.sort()
    lo_idx = max(0, min(len(diffs) - 1, int(0.025 * len(diffs))))
    hi_idx = max(0, min(len(diffs) - 1, int(0.975 * len(diffs))))

    return {
        "n": n,
        "acc_a": round(acc_a, 4),
        "acc_b": round(acc_b, 4),
        "diff": round(acc_a - acc_b, 4),
        "ci95": [round(diffs[lo_idx], 4), round(diffs[hi_idx], 4)],
        "wins": wins,
        "losses": losses,
        "both_correct": both_correct,
        "net": wins - losses,
    }


def _guardrail_exact_match_rate(arm_raw: list[dict[str, Any]], base_raw: list[dict[str, Any]], n_hard: int) -> float:
    """Fraction of guardrail items whose generated token sequence matches `base_08b` bit-for-bit."""
    g_arm = arm_raw[n_hard:]
    g_base = base_raw[n_hard:]
    assert len(g_arm) == len(g_base) and len(g_arm) > 0, (len(g_arm), len(g_base))
    exact = sum(1 for a, b in zip(g_arm, g_base) if list(a["token_ids"]) == list(b["token_ids"]))
    return round(exact / len(g_base), 4)


def run_smoke() -> dict[str, Any]:
    """Deterministic CPU check of stratified layer sampling, all LoRA tiers, paired bootstrap, and guardrails."""
    ds = build_ceiling_dataset()
    assert len(ds["hard_train"]) == 144 and len(ds["hard_test"]) == 36 and len(ds["guardrail_test"]) == 24

    # 1. Verify stratified sampling preserves subtype counts on a mock 8-layer hybrid block list
    class _ToyHybrid(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            for i in range(8):
                blk = nn.Module()
                if (i + 1) % 4 == 0:
                    blk.self_attn = nn.Linear(8, 8, bias=False)
                else:
                    blk.linear_attn = nn.Linear(6, 6, bias=False)
                blk.mlp = nn.Linear(8, 8, bias=False)
                layers.append(blk)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(layers)

    hybrid = _ToyHybrid()
    picked_h = pick_stratified_random_layers(hybrid, [0, 1, 3, 7], seed=123)
    assert len(picked_h) == 4 and len(set(picked_h)) == 4
    _, h_blocks = find_blocks(hybrid)
    want_tags = sorted(_block_arch_tag(h_blocks[i]) for i in [0, 1, 3, 7])
    got_tags = sorted(_block_arch_tag(h_blocks[i]) for i in picked_h)
    assert want_tags == got_tags, (want_tags, got_tags)

    # 2. Train micro/random/global LoRA tiers on 4-layer _PlantedReasoningLM and verify weight restoration
    torch.manual_seed(7)
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    rand_layers = pick_stratified_random_layers(model, [0, 1], seed=700)
    assert len(rand_layers) == 2 and set(rand_layers) != {0, 1}, rand_layers

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

    losses_by_tier: dict[str, list[float]] = {}
    sizes_by_tier: dict[str, float] = {}
    for nm, lyrs, rk in (
        ("moe_micro_wsi", [0, 1], 2),
        ("moe_micro_random", rand_layers, 2),
        ("moe_large_wsi", [0, 1, 2], 4),
        ("moe_large_random", pick_stratified_random_layers(model, [0, 1, 2], seed=800), 4),
        ("global_lora_r64", [0, 1, 2], 4),
        ("global_lora_bank_matched", [0, 1, 2, 3], 4),
    ):
        res_t = _train_lora_expert(
            model,
            lyrs,
            train_samples,
            rank=rk,
            alpha_scale=float(rk * 2),
            epochs=2,
            lr_val=2e-2,
            grad_accum=2,
            healthy_ids=h_ids,
            base_healthy_logp=base_lp,
            base_healthy_nll=base_nll,
            seed=42,
        )
        assert res_t["epoch_losses"][-1] < res_t["epoch_losses"][0], (nm, res_t["epoch_losses"])
        assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())
        losses_by_tier[nm] = res_t["epoch_losses"]
        sizes_by_tier[nm] = float(res_t["adapter_mb"])

    assert sizes_by_tier["moe_micro_wsi"] == sizes_by_tier["moe_micro_random"]
    assert sizes_by_tier["moe_large_wsi"] == sizes_by_tier["moe_large_random"]

    # 3. Verify _paired_bootstrap & _guardrail_exact_match_rate
    ra = [{"strict_correct": x} for x in [True, True, False, True, True, False]]
    rb = [{"strict_correct": x} for x in [False, True, False, False, False, True]]
    pb = _paired_bootstrap(ra, rb, seed=42, n_boot=500)
    assert pb["wins"] == 3 and pb["losses"] == 1 and pb["both_correct"] == 1 and pb["net"] == 2
    assert pb["ci95"][0] <= pb["diff"] <= pb["ci95"][1]

    fake_base = [{"token_ids": [i, i + 1]} for i in range(10)]
    fake_arm = [{"token_ids": [99]} for _ in range(6)] + [{"token_ids": [i, i + 1]} for i in range(6, 9)] + [{"token_ids": [0]}]
    gm = _guardrail_exact_match_rate(fake_arm, fake_base, n_hard=6)
    assert abs(gm - 0.75) < 1e-6, gm

    return {
        "mode": "smoke",
        "passed": True,
        "hybrid_stratified_tags_match": want_tags == got_tags,
        "toy_random_layers_for_0_1": rand_layers,
        "losses_by_tier": losses_by_tier,
        "paired_bootstrap_check": pb,
        "guardrail_exact_match_check": gm,
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

    ds = build_ceiling_dataset()
    hard_train = ds["hard_train"]
    hard_test = ds["hard_test"]
    guardrail_test = ds["guardrail_test"]
    all_test = hard_test + guardrail_test
    n_hard = len(hard_test)

    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    assert len(tok_t) == len(tok_s), f"Tokenizer vocab mismatch: teacher={len(tok_t)} vs student={len(tok_s)}"
    eos_set = set(lr.eos_ids(tok_s))
    think_end_id = tok_s.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_id, int) or think_end_id == tok_s.unk_token_id:
        think_end_id = None

    # ---- Phase 1: Teacher 4B generation + Top-32 logit extraction ----
    print(f"[controls] Phase 1: Teacher ({args.teacher}) generation & Top-32 logits (cap={args.max_teacher_train_tokens} tok)...", flush=True)
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
    teacher_4b_summary["all_60"] = {
        "strict_accuracy": round(sum(1 for r in teacher_test_rows if r["strict_correct"]) / len(teacher_test_rows), 4),
        "n_correct": sum(1 for r in teacher_test_rows if r["strict_correct"]),
        "n_total": len(teacher_test_rows),
    }
    print(f"[controls] teacher_4b: hard={teacher_4b_summary['hard_36']} | guard={teacher_4b_summary['guardrail_24']}", flush=True)

    kept_4b = {f: 0 for f in HARD_FAMILIES}
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
    teacher_4b_train_kept_frac = {f: round(kept_4b[f] / 48.0, 4) for f in HARD_FAMILIES}
    print(f"[controls] teacher_4b train responses kept fraction: {teacher_4b_train_kept_frac}", flush=True)

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

    # ---- Phase 2: Load Student 0.8B, WSI Causal Layers, Stratified Random Layers & Layer-12 Router ----
    print(f"[controls] Phase 2: Loading Student ({args.student}), WSI causal localization, random controls & router...", flush=True)
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
    random_layers_micro: dict[str, list[int]] = {}
    random_layers_large: dict[str, list[int]] = {}
    for f_idx, f in enumerate(HARD_FAMILIES):
        diag_f = diagnose_causal_traces(causal_traces_by_fam[f], split_ab=True, top_k_sites=16, center="healthy", healthy_cost="fisher", model_or_weights=student_m)
        causal_layers[f] = _extract_causal_layers(diag_f, n_layers, max_layers=8)
        random_layers_micro[f] = pick_stratified_random_layers(student_m, causal_layers[f][:4], seed=args.seed + 700 + f_idx)
        random_layers_large[f] = pick_stratified_random_layers(student_m, causal_layers[f][:8], seed=args.seed + 800 + f_idx)

    all_12_traces = [ct for f in HARD_FAMILIES for ct in causal_traces_by_fam[f]]
    diag_glob = diagnose_causal_traces(all_12_traces, split_ab=True, top_k_sites=16, center="healthy", healthy_cost="fisher", model_or_weights=student_m)
    global_causal_layers = _extract_causal_layers(diag_glob, n_layers, max_layers=8)
    print(f"[controls] WSI layers={causal_layers} | global_8={global_causal_layers}", flush=True)
    print(f"[controls] Random micro={random_layers_micro} | random large={random_layers_large}", flush=True)

    random_layers_report = {
        "micro": {
            f: {
                "layers": random_layers_micro[f],
                "wsi_layers": causal_layers[f][:4],
                "overlap_with_wsi": sorted(set(random_layers_micro[f]) & set(causal_layers[f][:4])),
                "n_overlap": len(set(random_layers_micro[f]) & set(causal_layers[f][:4])),
            }
            for f in HARD_FAMILIES
        },
        "large": {
            f: {
                "layers": random_layers_large[f],
                "wsi_layers": causal_layers[f][:8],
                "overlap_with_wsi": sorted(set(random_layers_large[f]) & set(causal_layers[f][:8])),
                "n_overlap": len(set(random_layers_large[f]) & set(causal_layers[f][:8])),
            }
            for f in HARD_FAMILIES
        },
    }

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

    hard_r_preds = routed_test_classes[:n_hard]
    guard_r_preds = routed_test_classes[n_hard:]
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
    print(f"[controls] Router report: {router_report}", flush=True)
    phase2_s = round(time.perf_counter() - t_p2, 2)

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

    # ---- Phase 3: Train 6 LoRA Control Configurations ----
    print("[controls] Phase 3: Training WSI vs random LoRA controls + global_lora_r64 + global_lora_bank_matched...", flush=True)
    t_p3 = time.perf_counter()

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        expert_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {
            "moe_micro_wsi": {},
            "moe_micro_random": {},
            "moe_large_wsi": {},
            "moe_large_random": {},
            "global_lora_r64": {},
            "global_lora_bank_matched": {},
        }

        routed_tier_specs = [
            ("moe_micro_wsi", causal_layers, 4, 8, 16.0, 5e-4, 0, False),
            ("moe_micro_random", random_layers_micro, 4, 8, 16.0, 5e-4, 100, True),
            ("moe_large_wsi", causal_layers, 8, 64, 64.0, 3e-4, 200, False),
            ("moe_large_random", random_layers_large, 8, 64, 64.0, 3e-4, 300, True),
        ]

        for f_idx, f in enumerate(HARD_FAMILIES):
            samples_f = train_samples_by_fam[f]
            for tier_nm, layer_map, k_layers, rk, a_scale, lr_v, sd_off, is_rand in routed_tier_specs:
                lyrs = list(layer_map[f][:k_layers])
                res_e = _train_lora_expert(
                    student_m,
                    lyrs,
                    samples_f,
                    rank=rk,
                    alpha_scale=a_scale,
                    epochs=3,
                    lr_val=lr_v,
                    grad_accum=4,
                    healthy_ids=prose_full_ids,
                    base_healthy_logp=base_prose_logp,
                    base_healthy_nll=base_prose_nll,
                    seed=args.seed + sd_off + f_idx,
                )
                p_exp = exp_dir / f"{tier_nm}_{f}.pt"
                torch.save(res_e.pop("weights"), p_exp)
                expert_paths[(tier_nm, f)] = p_exp
                if is_rand:
                    res_e["overlap_with_wsi"] = len(set(lyrs) & set(causal_layers[f][:k_layers]))
                expert_meta[tier_nm][f] = res_e
                print(
                    f"[controls] trained {tier_nm}/{f}: {res_e['adapter_mb']}MB "
                    f"layers={lyrs} | losses={res_e['epoch_losses']} | unrouted_kl={res_e['healthy_kl']}",
                    flush=True,
                )

        all_samples_flat = [sp for f in HARD_FAMILIES for sp in train_samples_by_fam[f]]

        res_glob_r64 = _train_lora_expert(
            student_m,
            global_causal_layers[:8],
            all_samples_flat,
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
        p_glob_r64 = exp_dir / "global_lora_r64.pt"
        torch.save(res_glob_r64.pop("weights"), p_glob_r64)
        expert_meta["global_lora_r64"] = res_glob_r64
        print(
            f"[controls] trained global_lora_r64: {res_glob_r64['adapter_mb']}MB "
            f"layers={global_causal_layers[:8]} | losses={res_glob_r64['epoch_losses']} | unrouted_kl={res_glob_r64['healthy_kl']}",
            flush=True,
        )

        all_24_layers = list(range(n_layers))
        res_glob_bank = _train_lora_expert(
            student_m,
            all_24_layers,
            all_samples_flat,
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
        p_glob_bank = exp_dir / "global_lora_bank_matched.pt"
        torch.save(res_glob_bank.pop("weights"), p_glob_bank)
        expert_meta["global_lora_bank_matched"] = res_glob_bank
        print(
            f"[controls] trained global_lora_bank_matched: {res_glob_bank['adapter_mb']}MB "
            f"layers=0..{n_layers - 1} | losses={res_glob_bank['epoch_losses']} | unrouted_kl={res_glob_bank['healthy_kl']}",
            flush=True,
        )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Exact Baked vLLM Evaluation across all 8 Arms ----
        print(f"[controls] Phase 4: Evaluating all 8 arms ({backend}, max_new_tokens={args.max_new_tokens})...", flush=True)
        t_p4 = time.perf_counter()

        if backend == "vllm":
            del student_m
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            base_raw = _vllm_generate(args.student, tok_s, test_prompts_ids, test_seeds, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
        else:
            base_raw = _hf_generate_subset(student_m, tok_s, test_prompts_ids, test_seeds, max_new_tokens=args.max_new_tokens)

        bake_tmp_dir = Path("/content/_pocket_moe_controls_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

        def _eval_ckpt(weights_path: Path, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
            upd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if backend == "vllm":
                _bake_full_checkpoint(Path(args.student), upd, bake_tmp_dir)
                del upd
                gc.collect()
                return _vllm_generate(str(bake_tmp_dir), tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
            return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, param_updates=upd)

        glob_r64_raw = _eval_ckpt(p_glob_r64, test_prompts_ids, test_seeds)
        p_glob_r64.unlink(missing_ok=True)
        glob_bank_raw = _eval_ckpt(p_glob_bank, test_prompts_ids, test_seeds)
        p_glob_bank.unlink(missing_ok=True)

        routed_raw_by_tier: dict[str, list[dict[str, Any]]] = {}
        for tier_nm in ("moe_micro_wsi", "moe_micro_random", "moe_large_wsi", "moe_large_random"):
            tier_raw: list[dict[str, Any] | None] = [None] * len(all_test)
            for i, cls_nm in enumerate(routed_test_classes):
                if cls_nm == "passthrough":
                    tier_raw[i] = base_raw[i]
            for f in HARD_FAMILIES:
                idxs = [i for i, cls_nm in enumerate(routed_test_classes) if cls_nm == f]
                p_exp = expert_paths[(tier_nm, f)]
                if idxs:
                    sub_outs = _eval_ckpt(p_exp, [test_prompts_ids[i] for i in idxs], [test_seeds[i] for i in idxs])
                    for idx, o in zip(idxs, sub_outs):
                        tier_raw[idx] = o
                p_exp.unlink(missing_ok=True)
            routed_raw_by_tier[tier_nm] = [r for r in tier_raw if r is not None]

        if bake_tmp_dir.exists():
            shutil_rm = __import__("shutil").rmtree
            shutil_rm(bake_tmp_dir, ignore_errors=True)
        phase4_s = round(time.perf_counter() - t_p4, 2)

    arm_raw_map: dict[str, tuple[list[dict[str, Any]], list[str] | None, bool]] = {
        "base_08b": (base_raw, None, False),
        "teacher_4b": (t_test_raw, None, True),
        "moe_micro_wsi_routed": (routed_raw_by_tier["moe_micro_wsi"], routed_test_classes, False),
        "moe_micro_random_routed": (routed_raw_by_tier["moe_micro_random"], routed_test_classes, False),
        "moe_large_wsi_routed": (routed_raw_by_tier["moe_large_wsi"], routed_test_classes, False),
        "moe_large_random_routed": (routed_raw_by_tier["moe_large_random"], routed_test_classes, False),
        "global_lora_r64": (glob_r64_raw, None, False),
        "global_lora_bank_matched": (glob_bank_raw, None, False),
    }

    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_nm, (raw_l, r_lbls, is_teacher) in arm_raw_map.items():
        if is_teacher:
            rows = teacher_test_rows
            sm = teacher_4b_summary
        else:
            rows = _grade_rows(all_test, raw_l, tok_s, eos_set, think_end_id, routed_labels=r_lbls)
            sm = _summarize_ceiling_arm(rows, hard_families=HARD_FAMILIES, guard_families=GUARDRAIL_FAMILIES)
            sm["all_60"] = {
                "strict_accuracy": round(sum(1 for r in rows if r["strict_correct"]) / len(rows), 4),
                "n_correct": sum(1 for r in rows if r["strict_correct"]),
                "n_total": len(rows),
            }
            sm["guardrail_24"]["guardrail_exact_match_vs_base"] = _guardrail_exact_match_rate(raw_l, base_raw, n_hard)
        arm_rows[arm_nm] = rows
        arm_summaries[arm_nm] = sm
        print(f"[controls] {arm_nm:26s} | hard={sm['hard_36']} | guard={sm['guardrail_24']}", flush=True)

    routed_prose_kl: dict[str, float] = {}
    for tier_nm in ("moe_micro_wsi", "moe_micro_random", "moe_large_wsi", "moe_large_random"):
        kl_tot = 0.0
        for p_cls in routed_prose_classes:
            if p_cls in HARD_FAMILIES:
                kl_tot += float(expert_meta[tier_nm][p_cls]["healthy_kl"])
        routed_prose_kl[tier_nm] = round(kl_tot / max(1, len(routed_prose_classes)), 6)

    comp_pairs = [
        ("moe_large_wsi_vs_base", "moe_large_wsi_routed", "base_08b"),
        ("moe_large_wsi_vs_large_random", "moe_large_wsi_routed", "moe_large_random_routed"),
        ("moe_micro_wsi_vs_micro_random", "moe_micro_wsi_routed", "moe_micro_random_routed"),
        ("moe_large_wsi_vs_global_lora_r64", "moe_large_wsi_routed", "global_lora_r64"),
        ("moe_large_wsi_vs_global_lora_bank_matched", "moe_large_wsi_routed", "global_lora_bank_matched"),
        ("global_lora_r64_vs_base", "global_lora_r64", "base_08b"),
        ("global_lora_bank_matched_vs_base", "global_lora_bank_matched", "base_08b"),
        ("moe_micro_wsi_vs_base", "moe_micro_wsi_routed", "base_08b"),
    ]
    paired_hard = {
        lbl: _paired_bootstrap(arm_rows[a][:n_hard], arm_rows[b][:n_hard], seed=args.seed + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }
    paired_all60 = {
        lbl: _paired_bootstrap(arm_rows[a], arm_rows[b], seed=args.seed + 100 + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    acc_hard = lambda k: float(arm_summaries[k]["hard_36"]["strict_accuracy"])  # noqa: E731
    acc_all60 = lambda k: float(arm_summaries[k]["all_60"]["strict_accuracy"])  # noqa: E731

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
        "teacher_4b_train_kept_frac": teacher_4b_train_kept_frac,
        "invariance": invariance,
        "causal_layers": {**causal_layers, "global": global_causal_layers},
        "random_layers": random_layers_report,
        "router_report": router_report,
        "expert_meta": expert_meta,
        "routed_prose_kl": routed_prose_kl,
        "generation_arms": arm_summaries,
        "paired_comparisons_hard_36": paired_hard,
        "paired_comparisons_all_60": paired_all60,
        "criteria": {
            "a_mixture_kept_fixed": bool(teacher_4b_train_kept_frac["weighted_mixture_transfer"] >= 0.70),
            "b_wsi_beats_random_micro": bool(acc_hard("moe_micro_wsi_routed") > acc_hard("moe_micro_random_routed")),
            "c_wsi_beats_random_large": bool(acc_hard("moe_large_wsi_routed") > acc_hard("moe_large_random_routed")),
            "d_routed_beats_global_27mb_all60": bool(acc_all60("moe_large_wsi_routed") > acc_all60("global_lora_r64")),
            "e_routed_beats_global_82mb_all60": bool(acc_all60("moe_large_wsi_routed") > acc_all60("global_lora_bank_matched")),
            "f_zero_prose_damage": bool(
                routed_prose_kl["moe_large_wsi"] == 0.0
                and arm_summaries["moe_large_wsi_routed"]["guardrail_24"]["strict_accuracy"] >= arm_summaries["base_08b"]["guardrail_24"]["strict_accuracy"]
            ),
        },
        "per_item_generation": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU check")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
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
