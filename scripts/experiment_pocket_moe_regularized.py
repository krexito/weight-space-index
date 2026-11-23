"""Controlled Ablation of Few-Shot Skill Compilation under Regularization, Full Depth, Multi-Trace Diversity & Self-Amplification.

Tests step by step (1 factor changed at a time) how much K=6 cartridge quality improves on the exact 5-skill benchmark
(`60` blind hard items, `24` guardrails, `200` MMLU, `16` held-out prose) when fixing quick-v1 bottlenecks:
  1. `base_08b`                        : Unpatched Qwen3.5-0.8B (`enable_thinking=False`)
  2. `teacher_4b`                      : Unpatched Qwen3.5-4B (`enable_thinking=False`)
  3. `oracle_1shot_08b`                : 0.8B with 1 solved teacher trace in context
  4. `v1_quick_r8_4L_routed`           : Prior v1 baseline (4 WSI layers, r=8, 24 steps, lr=5e-4, T=1.5, wd=0.01, no drop/anchor)
  5. `reg_r8_4L_routed`                : + Regularized training (48 steps, lr=2.5e-4, warmup+cos, wd=0.05, drop=0.1, T=2.0, kl=0.85, anchor=0.05)
  6. `reg_r8_all24L_routed`            : + All 24 layers (same regularization, lr=2.0e-4)
  7. `reg_multitrace_r8_all24L_routed` : + Multi-trace teacher diversity on the same 6 questions (up to 3 distinct valid traces/item)
  8. `reg_selfamp_r8_all24L_routed`    : + Blind teacher self-amplification (16 prompts -> self-consistent synthetic traces, max 12/skill)
  9. `joint_reg_r16_all24L`            : Regularized joint unrouted LoRA on all 30 primary traces across all 24 layers (r=16, 60 steps)

CPU smoke test::
    python scripts/experiment_pocket_moe_regularized.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_moe_regularized.py --teacher /content/teacher_4b --student /content/student_08b --out /content/pocket_moe_regularized_l4.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
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
    _extract_causal_layers,
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


# ------------------------------------------------------------------ Regularized LoRA + Base-Manifold Anchor KL
class _RegLoRALinear(nn.Module):
    """Low-rank residual adapter with optional input dropout before `lora_A`."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha_scale: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha_scale) / max(1, self.rank)
        self.lora_drop: nn.Module = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        dev = base.weight.device
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


def _train_regularized_lora_expert(
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
    """Step-budgeted LoRA distillation with warmup+cosine schedule, dropout, weight decay & online anchor KL."""
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
    anc_kls: list[float] = []
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
                    s_logits = _tail_logits(model, ids_dev, lo)[: hi - lo]
                    t_ids = ids_dev[0, lo + 1 : hi + 1]
                    tk_ids = sp["topk_ids"].to(dev) if sp.get("topk_ids") is not None else None
                    tk_log = sp["topk_logits"].to(dev) if sp.get("topk_logits") is not None else None
                    loss = distill_step_loss(s_logits, t_ids, tk_ids, tk_log, T=float(distill_T), kl_weight=float(kl_weight))
                    (loss / float(grad_accum)).backward()
                    step_loss += float(loss.detach().item()) / float(grad_accum)
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
                        anc_kls.append(float(kl_anc.detach().item()))
                    del anc_dev, lp_s, lp_b, kl_anc

            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            sched.step()
            step_losses.append(step_loss)

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

    n_ckpts = min(8, len(step_losses))
    ckpts: list[float] = []
    for b in range(n_ckpts):
        lo_b = (b * len(step_losses)) // n_ckpts
        hi_b = ((b + 1) * len(step_losses)) // n_ckpts
        seg = step_losses[lo_b:hi_b] or [step_losses[lo_b]]
        ckpts.append(round(sum(seg) / len(seg), 5))

    return {
        "weights": merged_weights,
        "adapter_mb": adapter_mb,
        "merged_shard_mb": merged_mb,
        "layer_idxs": list(layer_idxs),
        "n_train_samples": len(train_samples),
        "total_steps": int(total_steps),
        "step_losses_checkpoints": ckpts,
        "mean_anchor_kl": round(sum(anc_kls) / max(1, len(anc_kls)), 6) if anc_kls else 0.0,
        "compile_time_s": compile_time_s,
        **drift,
    }


# ------------------------------------------------------------------ Blind Teacher Self-Amplification Helpers
_BOXED_PREFIX_RE = re.compile(r"^\\(?:text|mathrm|mathbf)\{(.*)\}$", re.S)
_CLEAN_INT_RE = re.compile(r"^[+-]?\d+(?:\.0+)?$")
_FORBIDDEN_Q_MARKERS = ("\\boxed", "solution", "answer:", "step 1")


def _extract_boxed_int(text: str) -> int | None:
    """Extract the last balanced `\\boxed{...}` integer strictly (no decimal fraction or arbitrary fallback)."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    pos = idx + len("\\boxed{")
    depth = 1
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    inner = text[idx + len("\\boxed{") : pos - 1].strip()
    inner = inner.strip("$").strip()
    while True:
        m_wrap = _BOXED_PREFIX_RE.match(inner)
        if not m_wrap:
            break
        inner = m_wrap.group(1).strip().strip("$").strip()
    inner = inner.replace(",", "").replace(" ", "")
    if not _CLEAN_INT_RE.match(inner):
        return None
    try:
        return int(round(float(inner)))
    except (ValueError, OverflowError):
        return None


def _jaccard_words(a: str, b: str) -> float:
    """Lowercase alphanumeric word-set Jaccard similarity."""
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / max(1, len(wa | wb))


def _clean_synth_question(raw_text: str, existing_questions: list[str]) -> str | None:
    """Validate and deduplicate a teacher-synthesized math word problem without inspecting test sets."""
    q = raw_text.strip()
    q = re.sub(r"^```[a-zA-Z]*\s*", "", q)
    q = re.sub(r"\s*```$", "", q).strip()
    q = re.sub(r"^(?:problem|question|word\s*problem)\s*[:\-]\s*", "", q, flags=re.I).strip()
    if len(q) < 60 or len(q) > 900:
        return None
    if "?" not in q:
        return None
    q_low = q.lower()
    if any(mk in q_low for mk in _FORBIDDEN_Q_MARKERS):
        return None
    if any(_jaccard_words(q, ex) >= 0.75 for ex in existing_questions):
        return None
    return q


def _build_synth_prompt(q_a: str, q_b: str) -> str:
    """Prompt the teacher to invent a novel problem from the same mathematical structure in a new setting."""
    return (
        "Here are two example math word problems from the same reasoning family:\n\n"
        f"Example 1:\n{q_a}\n\n"
        f"Example 2:\n{q_b}\n\n"
        "Write ONE new self-contained math word problem from this exact same mathematical family, "
        "using a completely different real-world scenario, entities, and clean numbers so that the exact "
        "final answer is a positive integer. Output ONLY the problem statement ending with a question mark — "
        "no solution, no answer, and no extra commentary."
    )


# ------------------------------------------------------------------ Fast Exact Sparse MMLU Helper
def _score_routed_mmlu_sparse(
    model: nn.Module,
    mmlu_test_pids: list[list[int]],
    mmlu_test: list[dict[str, Any]],
    routed_mmlu_classes: list[str],
    skill_name: str,
    base_mmlu_raw: dict[str, Any],
    letter_ids: dict[str, list[int]],
    dev: torch.device,
) -> dict[str, Any]:
    """Score MMLU only on items routed to `skill_name`, splicing exact base logits for non-fired items."""
    idxs = [i for i, cls_nm in enumerate(routed_mmlu_classes) if cls_nm == skill_name]
    if not idxs:
        return base_mmlu_raw
    sub_pids = [mmlu_test_pids[i] for i in idxs]
    sub_items = [mmlu_test[i] for i in idxs]
    sub_sc = _score_mmlu_fast(model, sub_pids, sub_items, letter_ids, dev)

    probs = base_mmlu_raw["probs"].clone()
    preds = list(base_mmlu_raw["preds"])
    correct = list(base_mmlu_raw["correct"])
    masses = list(base_mmlu_raw["choice_masses"])
    for k, orig_i in enumerate(idxs):
        probs[orig_i] = sub_sc["probs"][k]
        preds[orig_i] = sub_sc["preds"][k]
        correct[orig_i] = sub_sc["correct"][k]
        masses[orig_i] = sub_sc["choice_masses"][k]
    return {"probs": probs, "preds": preds, "correct": correct, "choice_masses": masses}


# ------------------------------------------------------------------ Compact Summary Printer
def _print_regularized_summary(payload: dict[str, Any]) -> None:
    """Print concise human-readable ablation report + single-line compact JSON omitting per-item rows."""
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return

    print("\n=== 1. COMPILACION REGULARIZADA (K=%s/habilidad) & AUTO-AMPLIFICACION ===" % payload.get("k_shots_per_skill"), flush=True)
    print("Compile time (s):", payload.get("compile_time_s"), flush=True)
    print("Teacher kept frac:", payload.get("teacher_4b_train_kept_frac"), flush=True)
    print("Multitrace counts:", payload.get("multitrace_counts"), flush=True)
    print(
        "Self-amp stats:",
        {k: {kk: vv for kk, vv in v.items() if kk != "sample_questions"} for k, v in payload.get("selfamp_stats", {}).items()},
        flush=True,
    )
    print("Router report:", payload.get("router_report"), flush=True)

    prose_kl = payload.get("prose_kl", {})
    print("\n=== 2. ABLACION 9 BRAZOS PAREADOS (HARD_60 + GUARDRAIL_24) ===", flush=True)
    for arm_nm, st in payload.get("generation_arms", {}).items():
        h, g = st["hard_60"], st["guardrail_24"]
        gm = g.get("guardrail_exact_match_vs_base")
        gm_s = f"{gm*100:5.1f}%" if gm is not None else "   n/a"
        pkl = prose_kl.get(arm_nm, 0.0)
        print(
            f"  {arm_nm:33s} | hard60={h['strict_accuracy']*100:5.1f}% ({h['n_correct']:2d}/60) "
            f"[old1-4={h['old_skills_1_to_4_acc']*100:5.1f}%, s5={h['last_skill_5_acc']*100:5.1f}%, "
            f"ren={h['renumbered_30']*100:5.1f}%, rep={h['rephrased_30']*100:5.1f}%] | "
            f"guard={g['strict_accuracy']*100:5.1f}% (ex={gm_s}) | pkl={pkl:.5f}",
            flush=True,
        )
        print(f"     per_skill={h['per_skill']}", flush=True)

    print("\n=== 3. RETENCION MMLU ===", flush=True)
    for arm_nm, vm in payload.get("mmlu_arms", {}).items():
        a_m, nm = vm["all_400"], vm["non_math"]
        d_nm = f"{nm['delta_vs_base']*100:+5.1f} pts" if "delta_vs_base" in nm else "     base"
        print(
            f"  {arm_nm:33s} | MMLU={a_m['accuracy']*100:5.1f}% | non_math={nm['accuracy']*100:5.1f}% ({d_nm}) | "
            f"ABCD={a_m['mean_abcd_prob_mass']*100:5.1f}% | fire={vm['router_fire_rate']*100:5.1f}%",
            flush=True,
        )

    print("\n=== 4. COMPARACIONES PAREADAS (IC 95% BOOTSTRAP) ===", flush=True)
    for grp_lbl, grp_key in [("  [hard_60]", "paired_hard_60"), ("  [old_skills_1_to_4 (48)]", "paired_old_skills_48")]:
        print(grp_lbl, flush=True)
        for k, v in payload.get(grp_key, {}).items():
            print(
                f"    {k:40s}: diff={v['diff']*100:+5.1f} pts "
                f"CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})",
                flush=True,
            )

    print("\nBest routed arm:", payload.get("best_routed_arm_name"), flush=True)
    print("Criteria:", payload.get("criteria"), flush=True)
    print("Timing (s):", payload.get("timing_s"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_generation"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


# ------------------------------------------------------------------ CPU Smoke Test
def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of integer extraction, synth filtering, regularized LoRA + dropout + anchor KL, sparse MMLU & summary."""
    # 1. _extract_boxed_int
    assert _extract_boxed_int(r"Steps done. Final answer: \boxed{42}.") == 42
    assert _extract_boxed_int(r"First \boxed{10}, then corrected \boxed{-17}") == -17
    assert _extract_boxed_int(r"\boxed{\text{1,250}}") == 1250
    assert _extract_boxed_int(r"\boxed{$84.0$}") == 84
    assert _extract_boxed_int(r"\boxed{3/4}") is None
    assert _extract_boxed_int(r"\boxed{42.5}") is None
    assert _extract_boxed_int("no box here 42") is None
    assert _extract_boxed_int(r"broken \boxed{42") is None

    # 2. _clean_synth_question
    existing = [
        "Pump Alpha fills a water reservoir in 12 hours while Pump Beta fills the same reservoir in 18 hours. How many hours do both take together?"
    ]
    novel_ok = (
        "An orchard crew picks 45 crates of apples every morning before noon and transfers one-fifth of the harvest "
        "to a cold cellar in the afternoon. How many crates remain after 4 days of harvesting?"
    )
    assert _clean_synth_question(f"Question: {novel_ok}", existing) == novel_ok
    assert _clean_synth_question("Too short question?", existing) is None
    assert _clean_synth_question(novel_ok.replace("?", "."), existing) is None
    assert _clean_synth_question(novel_ok + r" \boxed{36}", existing) is None
    assert _clean_synth_question("Solution: " + novel_ok, existing) is None
    assert _clean_synth_question("Step 1 follow along: " + novel_ok, existing) is None
    near_dup = (
        "Pump Alpha fills a water reservoir in 14 hours while Pump Beta fills the same reservoir in 21 hours. "
        "How many hours do both take working together?"
    )
    assert _clean_synth_question(near_dup, existing) is None

    # 3. _RegLoRALinear dropout toggle + _train_regularized_lora_expert with anchor KL + bit-exact restore
    torch.manual_seed(17)
    lin = nn.Linear(16, 16, bias=False)
    reg_mod = _RegLoRALinear(lin, rank=4, alpha_scale=8.0, dropout=0.5)
    with torch.no_grad():
        reg_mod.lora_B.fill_(0.25)
    x_probe = torch.ones((1, 8, 16))
    reg_mod.train()
    y1, y2 = reg_mod(x_probe), reg_mod(x_probe)
    assert not torch.equal(y1, y2), "Dropout should be active in train mode"
    reg_mod.eval()
    assert torch.equal(reg_mod(x_probe), reg_mod(x_probe)), "Dropout should be inactive in eval mode"

    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    seqs = [[1, 2, 4, 10, 20, 20, 20, 20], [2, 1, 4, 11, 20, 20, 20, 20]]
    tr_samples = []
    for s in seqs:
        ids_t = torch.tensor(s, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            t_log = model(ids_t).logits[0, 3:7].float()
        vals, tk_ids = t_log.topk(8, dim=-1)
        tr_samples.append({
            "ids": s,
            "lo": 3,
            "hi": 7,
            "topk_ids": tk_ids.to(torch.int32).cpu(),
            "topk_logits": vals.to(torch.float16).cpu(),
        })
    h_ids = [torch.tensor([1, 2, 3, 1], dtype=torch.long).unsqueeze(0)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    reg_res = _train_regularized_lora_expert(
        model,
        [0, 1],
        tr_samples,
        rank=2,
        alpha_scale=4.0,
        total_steps=6,
        grad_accum=2,
        lr_val=2e-2,
        weight_decay=0.05,
        lora_dropout=0.10,
        warmup_frac=0.20,
        distill_T=2.0,
        kl_weight=0.85,
        anchor_ids=h_ids,
        base_anchor_logp=base_lp,
        anchor_kl_weight=0.05,
        healthy_ids=h_ids,
        base_healthy_logp=base_lp,
        base_healthy_nll=base_nll,
        seed=42,
    )
    assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items())
    assert reg_res["total_steps"] == 6 and len(reg_res["step_losses_checkpoints"]) == 6
    assert reg_res["step_losses_checkpoints"][-1] < reg_res["step_losses_checkpoints"][0], reg_res["step_losses_checkpoints"]

    # 4. Sparse MMLU exact equivalence (both empty routing and subset routing)
    fake_mmlu = [
        {"gold_idx": 0, "is_math_subject": False},
        {"gold_idx": 1, "is_math_subject": True},
        {"gold_idx": 2, "is_math_subject": False},
        {"gold_idx": 3, "is_math_subject": True},
    ]
    fake_pids = [[1, 2, 4, 5], [2, 1, 5, 4], [1, 3, 4, 6], [3, 1, 5, 7]]
    letter_ids = {"A": [6], "B": [7], "C": [8], "D": [9]}
    dev = next(model.parameters()).device
    base_mmlu = _score_mmlu_fast(model, fake_pids, fake_mmlu, letter_ids, dev)
    with _temp_apply_weights(model, reg_res["weights"]):
        full_edited = _score_mmlu_fast(model, fake_pids, fake_mmlu, letter_ids, dev)
        zero_routed = _score_routed_mmlu_sparse(model, fake_pids, fake_mmlu, ["passthrough"] * 4, "rate_work_combined", base_mmlu, letter_ids, dev)
        sub_classes = ["passthrough", "rate_work_combined", "passthrough", "rate_work_combined"]
        sub_routed = _score_routed_mmlu_sparse(model, fake_pids, fake_mmlu, sub_classes, "rate_work_combined", base_mmlu, letter_ids, dev)
    assert zero_routed is base_mmlu
    comp_sparse = _compose_routed_mmlu(fake_mmlu, sub_classes, base_mmlu, {"rate_work_combined": sub_routed}, seed=1)
    comp_full = _compose_routed_mmlu(fake_mmlu, sub_classes, base_mmlu, {"rate_work_combined": full_edited}, seed=1)
    assert comp_sparse == comp_full, (comp_sparse, comp_full)

    return {
        "mode": "smoke",
        "passed": True,
        "boxed_int_and_synth_filter": True,
        "reg_lora_checkpoints": reg_res["step_losses_checkpoints"],
        "mean_anchor_kl": reg_res["mean_anchor_kl"],
        "sparse_mmlu_exact_match": True,
    }


# ------------------------------------------------------------------ Full GPU Controlled Ablation
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

    # ---- Phase 1A: Single 4B Batched Gen (84 Test + 30x4 Multi-Trace Train + 80 Synth Prompts) ----
    print(
        f"[reg] Phase 1A: Teacher ({args.teacher}) generation ({backend}) — 84 test + 30x4 multi-trace + 80 synth prompts...",
        flush=True,
    )
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

    synth_gen_pids: list[list[int]] = []
    synth_gen_seeds: list[int] = []
    n_synth_prompts_per_skill = 16
    for sk_i, sk in enumerate(SKILLS_5):
        qs_sk = [it["question"] for it in hard_train if it["family"] == sk]
        pairs_sk = [(i % len(qs_sk), (i + d) % len(qs_sk)) for d in (1, 2, 3) for i in range(len(qs_sk))][:n_synth_prompts_per_skill]
        for j, (ia, ib) in enumerate(pairs_sk):
            synth_gen_pids.append(
                _format_messages(tok_s, [{"role": "user", "content": _build_synth_prompt(qs_sk[ia], qs_sk[ib])}], add_generation_prompt=True, enable_thinking=False)
            )
            synth_gen_seeds.append(int(args.seed) + 2000 + sk_i * 50 + j)

    teacher_hf = None
    if backend == "vllm":
        outs_1a = _vllm_generate(
            args.teacher,
            tok_t,
            test_pids + train_multi_pids + synth_gen_pids,
            test_seeds + train_multi_seeds + synth_gen_seeds,
            max_new_tokens=args.max_new_tokens,
            dtype_str=dtype_vllm_str,
            base_seed=args.seed,
        )
    else:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
        outs_1a = _hf_generate_subset(
            teacher_hf,
            tok_t,
            test_pids + train_multi_pids + synth_gen_pids,
            test_seeds + train_multi_seeds + synth_gen_seeds,
            max_new_tokens=args.max_new_tokens,
        )

    n_test = len(all_test)
    n_tm = len(train_multi_pids)
    t_test_raw = outs_1a[:n_test]
    t_train_multi_raw = outs_1a[n_test : n_test + n_tm]
    t_synth_raw = outs_1a[n_test + n_tm :]

    teacher_test_rows = _grade_rows(all_test, t_test_raw, tok_s, eos_set, think_end_id)
    teacher_summary = _summarize_fewshot_arm(teacher_test_rows)
    print(f"[reg] teacher_4b: hard={teacher_summary['hard_60']['strict_accuracy']*100:.1f}% | guard={teacher_summary['guardrail_24']['strict_accuracy']*100:.1f}%", flush=True)

    kept_4b = {sk: 0 for sk in SKILLS_5}
    primary_pairs: dict[str, list[tuple[str, str]]] = {sk: [] for sk in SKILLS_5}
    multitrace_pairs: dict[str, list[tuple[str, str]]] = {sk: [] for sk in SKILLS_5}
    primary_resp_by_item: list[str] = []

    for i, it in enumerate(hard_train):
        sk = it["family"]
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
        primary_pairs[sk].append((q_txt, primary_txt))
        for v_txt in valid_txts:
            multitrace_pairs[sk].append((q_txt, v_txt))

    teacher_kept_frac = {sk: round(kept_4b[sk] / max(1, args.k_shots), 4) for sk in SKILLS_5}
    multitrace_counts = {sk: len(multitrace_pairs[sk]) for sk in SKILLS_5}
    print(f"[reg] teacher_4b kept={teacher_kept_frac} | multitrace_counts={multitrace_counts}", flush=True)

    # ---- Phase 1B: Blind Teacher Self-Consistency Filtering on Generated Synthetic Questions ----
    clean_synth_by_skill: dict[str, list[str]] = {sk: [] for sk in SKILLS_5}
    cursor_s = 0
    for sk in SKILLS_5:
        existing_qs = [it["question"] for it in hard_train if it["family"] == sk]
        for _ in range(n_synth_prompts_per_skill):
            raw_q = t_synth_raw[cursor_s]
            cursor_s += 1
            q_raw_txt = tok_s.decode(lr.strip_eos(raw_q["token_ids"], eos_set), skip_special_tokens=True)
            q_cln = _clean_synth_question(q_raw_txt, existing_qs + clean_synth_by_skill[sk])
            if q_cln is not None:
                clean_synth_by_skill[sk].append(q_cln)

    solve_meta: list[tuple[str, str]] = []
    solve_pids: list[list[int]] = []
    solve_seeds: list[int] = []
    for sk_i, sk in enumerate(SKILLS_5):
        for j, q_cln in enumerate(clean_synth_by_skill[sk]):
            p_q = _format_messages(tok_s, [{"role": "user", "content": format_question(q_cln)}], add_generation_prompt=True, enable_thinking=False)
            solve_meta.append((sk, q_cln))
            solve_pids.extend([p_q, p_q])
            solve_seeds.extend([int(args.seed) + 3000 + sk_i * 100 + j, int(args.seed) + 4000 + sk_i * 100 + j])

    if solve_pids:
        if backend == "vllm":
            solve_outs = _vllm_generate(args.teacher, tok_t, solve_pids, solve_seeds, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed + 7)
        else:
            assert teacher_hf is not None
            solve_outs = _hf_generate_subset(teacher_hf, tok_t, solve_pids, solve_seeds, max_new_tokens=args.max_new_tokens)
    else:
        solve_outs = []

    selfamp_pairs: dict[str, list[tuple[str, str]]] = {sk: list(primary_pairs[sk]) for sk in SKILLS_5}
    consistent_counts = {sk: 0 for sk in SKILLS_5}
    selected_synth_qs: dict[str, list[str]] = {sk: [] for sk in SKILLS_5}
    for k_idx, (sk, q_cln) in enumerate(solve_meta):
        raw1, raw2 = solve_outs[2 * k_idx], solve_outs[2 * k_idx + 1]
        if raw1["finish_reason"] != "stop" or raw2["finish_reason"] != "stop":
            continue
        b1 = lr.strip_eos(list(raw1["token_ids"]), eos_set)
        b2 = lr.strip_eos(list(raw2["token_ids"]), eos_set)
        if len(b1) > int(args.max_teacher_train_tokens) or len(b2) > int(args.max_teacher_train_tokens):
            continue
        txt1 = tok_s.decode(b1, skip_special_tokens=True).strip()
        txt2 = tok_s.decode(b2, skip_special_tokens=True).strip()
        a1, a2 = _extract_boxed_int(txt1), _extract_boxed_int(txt2)
        if a1 is not None and a2 is not None and a1 == a2 and 0 < a1 <= 100000:
            consistent_counts[sk] += 1
            if len(selected_synth_qs[sk]) < int(args.max_synth_per_skill):
                chosen_s = txt1 if len(b1) <= len(b2) else txt2
                selfamp_pairs[sk].append((q_cln, chosen_s))
                selected_synth_qs[sk].append(q_cln)

    selfamp_stats = {
        sk: {
            "n_prompts": n_synth_prompts_per_skill,
            "n_clean_q": len(clean_synth_by_skill[sk]),
            "n_consistent": consistent_counts[sk],
            "n_selected": len(selected_synth_qs[sk]),
            "total_train_traces": len(selfamp_pairs[sk]),
            "sample_questions": selected_synth_qs[sk][:2],
        }
        for sk in SKILLS_5
    }
    print(
        "[reg] Phase 1B self-amp:",
        {sk: f"clean={v['n_clean_q']} cons={v['n_consistent']} total={v['total_train_traces']}" for sk, v in selfamp_stats.items()},
        flush=True,
    )

    # ---- Phase 1C: Deduplicated 4B Top-32 Logit Extraction ----
    if teacher_hf is None:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    unique_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for sk in SKILLS_5:
        for pair in primary_pairs[sk] + multitrace_pairs[sk] + selfamp_pairs[sk]:
            unique_pairs.setdefault(pair, {})

    print(f"[reg] Phase 1C: Extracting Top-32 4B logits for {len(unique_pairs)} unique traces...", flush=True)
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

    primary_by_skill = {sk: [dict(unique_pairs[p], family=sk) for p in primary_pairs[sk]] for sk in SKILLS_5}
    multitrace_by_skill = {sk: [dict(unique_pairs[p], family=sk) for p in multitrace_pairs[sk]] for sk in SKILLS_5}
    selfamp_by_skill = {sk: [dict(unique_pairs[p], family=sk) for p in selfamp_pairs[sk]] for sk in SKILLS_5}
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student 0.8B, WSI Layers, Router, Base MMLU, Held-Out Prose & Disjoint Anchor KL ----
    print(f"[reg] Phase 2: Loading Student ({args.student}), WSI layers, balanced K=6 router, base MMLU & disjoint anchor...", flush=True)
    t_p2 = time.perf_counter()
    try:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        student_m = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in student_m.parameters():
        p.requires_grad_(False)

    _, blocks = find_blocks(student_m)
    n_layers = len(blocks)
    all_layers = list(range(n_layers))

    causal_traces_by_skill: dict[str, list[CompactTrace]] = {sk: [] for sk in SKILLS_5}
    for sk in SKILLS_5:
        for sp in primary_by_skill[sk][:4]:
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
    for sk in SKILLS_5:
        diag_s = diagnose_causal_traces(causal_traces_by_skill[sk], split_ab=True, top_k_sites=16, center="healthy", healthy_cost="fisher", model_or_weights=student_m)
        causal_layers[sk] = _extract_causal_layers(diag_s, n_layers, max_layers=8)

    router_layer = n_layers // 2
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
    print(f"[reg] Router report: {router_report}", flush=True)

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
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Train 5 Routed Ablation Tiers (x5 skills) + Joint Regularized Baseline ----
    print("[reg] Phase 3: Training 5 routed ablation tiers + joint_reg_r16_all24L...", flush=True)
    t_p3 = time.perf_counter()

    tier_defs = [
        ("v1_quick_r8_4L", primary_by_skill, False, 8, 16.0, 24, 2, 5e-4, 0.01, 0.00, 0.00, 1.5, 0.70, 0.00),
        ("reg_r8_4L", primary_by_skill, False, 8, 16.0, 48, 2, 2.5e-4, 0.05, 0.10, 0.15, 2.0, 0.85, 0.05),
        ("reg_r8_all24L", primary_by_skill, True, 8, 16.0, 48, 2, 2.0e-4, 0.05, 0.10, 0.15, 2.0, 0.85, 0.05),
        ("reg_multitrace_r8_all24L", multitrace_by_skill, True, 8, 16.0, 48, 3, 2.0e-4, 0.05, 0.10, 0.15, 2.0, 0.85, 0.05),
        ("reg_selfamp_r8_all24L", selfamp_by_skill, True, 8, 16.0, 48, 3, 2.0e-4, 0.05, 0.10, 0.15, 2.0, 0.85, 0.05),
    ]

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        expert_paths: dict[tuple[str, str], Path] = {}
        expert_meta: dict[str, Any] = {t[0]: {} for t in tier_defs}
        domain_mmlu_scores: dict[str, dict[str, dict[str, Any]]] = {t[0]: {} for t in tier_defs}

        for s_idx, sk in enumerate(SKILLS_5):
            for t_idx, (tier_nm, ds_map, use_all_layers, rk, a_sc, n_stp, g_acc, lr_v, wd_v, drp_v, wrm_v, t_dist, kl_w, anc_w) in enumerate(tier_defs):
                lyrs = all_layers if use_all_layers else list(causal_layers[sk][:4])
                res_e = _train_regularized_lora_expert(
                    student_m,
                    lyrs,
                    ds_map[sk],
                    rank=rk,
                    alpha_scale=a_sc,
                    total_steps=n_stp,
                    grad_accum=g_acc,
                    lr_val=lr_v,
                    weight_decay=wd_v,
                    lora_dropout=drp_v,
                    warmup_frac=wrm_v,
                    distill_T=t_dist,
                    kl_weight=kl_w,
                    anchor_ids=anchor_full_ids if anc_w > 0.0 else None,
                    base_anchor_logp=base_anchor_logp if anc_w > 0.0 else None,
                    anchor_kl_weight=anc_w,
                    healthy_ids=prose_full_ids,
                    base_healthy_logp=base_prose_logp,
                    base_healthy_nll=base_prose_nll,
                    seed=int(args.seed) + t_idx * 100 + s_idx,
                )
                upd_w = res_e.pop("weights")
                with _temp_apply_weights(student_m, upd_w):
                    domain_mmlu_scores[tier_nm][sk] = _score_routed_mmlu_sparse(
                        student_m, mmlu_test_pids, mmlu_test, routed_mmlu_classes, sk, base_mmlu_raw, letter_ids, dev
                    )
                p_exp = exp_dir / f"{tier_nm}_{sk}.pt"
                torch.save(upd_w, p_exp)
                del upd_w
                expert_paths[(tier_nm, sk)] = p_exp
                expert_meta[tier_nm][sk] = res_e
                print(
                    f"[reg] {tier_nm}/{sk}: n={res_e['n_train_samples']} steps={n_stp} ({res_e['adapter_mb']}MB, {res_e['compile_time_s']:.1f}s) | "
                    f"loss={res_e['step_losses_checkpoints'][0]}->{res_e['step_losses_checkpoints'][-1]} | pkl={res_e['healthy_kl']}",
                    flush=True,
                )

        all_30_flat = [sp for sk in SKILLS_5 for sp in primary_by_skill[sk]]
        res_joint = _train_regularized_lora_expert(
            student_m,
            all_layers,
            all_30_flat,
            rank=16,
            alpha_scale=32.0,
            total_steps=60,
            grad_accum=4,
            lr_val=1.5e-4,
            weight_decay=0.05,
            lora_dropout=0.10,
            warmup_frac=0.15,
            distill_T=2.0,
            kl_weight=0.85,
            anchor_ids=anchor_full_ids,
            base_anchor_logp=base_anchor_logp,
            anchor_kl_weight=0.05,
            healthy_ids=prose_full_ids,
            base_healthy_logp=base_prose_logp,
            base_healthy_nll=base_prose_nll,
            seed=int(args.seed) + 900,
        )
        upd_j = res_joint.pop("weights")
        with _temp_apply_weights(student_m, upd_j):
            joint_mmlu_raw = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
        p_joint = exp_dir / "joint_reg_r16.pt"
        torch.save(upd_j, p_joint)
        del upd_j
        expert_meta["joint_reg_r16_all24L"] = res_joint
        print(
            f"[reg] joint_reg_r16_all24L: {res_joint['adapter_mb']}MB in {res_joint['compile_time_s']:.1f}s | pkl={res_joint['healthy_kl']}",
            flush=True,
        )

        mmlu_arms: dict[str, Any] = {"base_08b": base_mmlu_summary}
        for idx_t, (tier_nm, *_rest) in enumerate(tier_defs):
            mmlu_arms[f"{tier_nm}_routed"] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_classes, base_mmlu_raw, domain_mmlu_scores[tier_nm], seed=int(args.seed) + 1 + idx_t
            )
        mmlu_arms["joint_reg_r16_all24L"] = _summarize_mmlu(
            mmlu_test, joint_mmlu_raw, base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=int(args.seed) + 10
        )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Evaluate All 9 Paired Arms on 84 Test Prompts ----
        print(f"[reg] Phase 4: Evaluating 9 paired arms ({backend}, max_new_tokens={args.max_new_tokens})...", flush=True)
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

        bake_tmp_dir = Path("/content/_pocket_moe_reg_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

        def _eval_ckpt(weights_path: Path, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
            upd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if backend == "vllm":
                _bake_full_checkpoint(Path(args.student), upd, bake_tmp_dir)
                del upd
                gc.collect()
                return _vllm_generate(str(bake_tmp_dir), tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, dtype_str=dtype_vllm_str, base_seed=args.seed)
            return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens, param_updates=upd)

        joint_raw = _eval_ckpt(p_joint, test_pids, test_seeds)
        p_joint.unlink(missing_ok=True)

        routed_raw_by_tier: dict[str, list[dict[str, Any]]] = {}
        for tier_nm, *_rest in tier_defs:
            tier_raw: list[dict[str, Any] | None] = [None] * len(all_test)
            for i, cls_nm in enumerate(routed_test_classes):
                if cls_nm == "passthrough":
                    tier_raw[i] = base_raw[i]
            for sk in SKILLS_5:
                idxs = [i for i, cls_nm in enumerate(routed_test_classes) if cls_nm == sk]
                p_exp = expert_paths[(tier_nm, sk)]
                if idxs:
                    sub_outs = _eval_ckpt(p_exp, [test_pids[i] for i in idxs], [test_seeds[i] for i in idxs])
                    for idx_i, o in zip(idxs, sub_outs):
                        tier_raw[idx_i] = o
                p_exp.unlink(missing_ok=True)
            routed_raw_by_tier[tier_nm] = [r for r in tier_raw if r is not None]

        if bake_tmp_dir.exists():
            shutil.rmtree(bake_tmp_dir, ignore_errors=True)
        phase4_s = round(time.perf_counter() - t_p4, 2)

    arm_raw_map: dict[str, tuple[list[dict[str, Any]], list[str] | None, bool]] = {
        "base_08b": (base_raw, None, False),
        "teacher_4b": (t_test_raw, None, True),
        "oracle_1shot_08b": (oracle_raw, None, False),
        "v1_quick_r8_4L_routed": (routed_raw_by_tier["v1_quick_r8_4L"], routed_test_classes, False),
        "reg_r8_4L_routed": (routed_raw_by_tier["reg_r8_4L"], routed_test_classes, False),
        "reg_r8_all24L_routed": (routed_raw_by_tier["reg_r8_all24L"], routed_test_classes, False),
        "reg_multitrace_r8_all24L_routed": (routed_raw_by_tier["reg_multitrace_r8_all24L"], routed_test_classes, False),
        "reg_selfamp_r8_all24L_routed": (routed_raw_by_tier["reg_selfamp_r8_all24L"], routed_test_classes, False),
        "joint_reg_r16_all24L": (joint_raw, None, False),
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

    prose_kl: dict[str, float] = {
        f"{tier_nm}_routed": _routed_prose_kl(tier_nm) for tier_nm, *_rest in tier_defs
    }
    prose_kl["joint_reg_r16_all24L"] = float(expert_meta["joint_reg_r16_all24L"]["healthy_kl"])

    compile_time_s: dict[str, float] = {
        f"{tier_nm}_mean_per_skill_s": round(sum(expert_meta[tier_nm][s]["compile_time_s"] for s in SKILLS_5) / len(SKILLS_5), 2)
        for tier_nm, *_rest in tier_defs
    }
    compile_time_s["joint_reg_r16_all24L_total_s"] = float(expert_meta["joint_reg_r16_all24L"]["compile_time_s"])

    routed_arm_names = [f"{t[0]}_routed" for t in tier_defs]
    best_routed_arm = max(
        routed_arm_names,
        key=lambda k: (arm_summaries[k]["hard_60"]["strict_accuracy"], arm_summaries[k]["guardrail_24"]["strict_accuracy"]),
    )

    comp_pairs = [
        ("reg_r8_4L_vs_v1_quick_r8_4L", "reg_r8_4L_routed", "v1_quick_r8_4L_routed"),
        ("reg_r8_all24L_vs_reg_r8_4L", "reg_r8_all24L_routed", "reg_r8_4L_routed"),
        ("reg_r8_all24L_vs_v1_quick_r8_4L", "reg_r8_all24L_routed", "v1_quick_r8_4L_routed"),
        ("reg_multitrace_vs_reg_r8_all24L", "reg_multitrace_r8_all24L_routed", "reg_r8_all24L_routed"),
        ("reg_multitrace_vs_v1_quick_r8_4L", "reg_multitrace_r8_all24L_routed", "v1_quick_r8_4L_routed"),
        ("reg_selfamp_vs_reg_r8_all24L", "reg_selfamp_r8_all24L_routed", "reg_r8_all24L_routed"),
        ("reg_selfamp_vs_reg_multitrace", "reg_selfamp_r8_all24L_routed", "reg_multitrace_r8_all24L_routed"),
        ("reg_selfamp_vs_v1_quick_r8_4L", "reg_selfamp_r8_all24L_routed", "v1_quick_r8_4L_routed"),
        ("reg_selfamp_vs_oracle_1shot", "reg_selfamp_r8_all24L_routed", "oracle_1shot_08b"),
        ("reg_selfamp_vs_joint_reg_r16", "reg_selfamp_r8_all24L_routed", "joint_reg_r16_all24L"),
        ("reg_multitrace_vs_oracle_1shot", "reg_multitrace_r8_all24L_routed", "oracle_1shot_08b"),
        ("reg_r8_all24L_vs_oracle_1shot", "reg_r8_all24L_routed", "oracle_1shot_08b"),
        ("best_routed_vs_base", best_routed_arm, "base_08b"),
        ("joint_reg_r16_vs_base", "joint_reg_r16_all24L", "base_08b"),
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
        "max_synth_per_skill": int(args.max_synth_per_skill),
        "max_new_tokens": int(args.max_new_tokens),
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_wsi_router_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "compile_time_s": compile_time_s,
        "teacher_4b_train_kept_frac": teacher_kept_frac,
        "multitrace_counts": multitrace_counts,
        "selfamp_stats": selfamp_stats,
        "invariance": invariance,
        "causal_layers": causal_layers,
        "router_report": router_report,
        "expert_meta": expert_meta,
        "prose_kl": prose_kl,
        "generation_arms": arm_summaries,
        "mmlu_arms": mmlu_arms,
        "paired_hard_60": paired_hard_60,
        "paired_old_skills_48": paired_old_skills_48,
        "best_routed_arm_name": best_routed_arm,
        "criteria": {
            "a_reg_4L_beats_v1_quick": bool(h_acc("reg_r8_4L_routed") > h_acc("v1_quick_r8_4L_routed")),
            "b_full_depth_beats_4L": bool(h_acc("reg_r8_all24L_routed") > h_acc("reg_r8_4L_routed")),
            "c_multitrace_beats_single_trace": bool(h_acc("reg_multitrace_r8_all24L_routed") > h_acc("reg_r8_all24L_routed")),
            "d_selfamp_beats_oracle_1shot": bool(h_acc("reg_selfamp_r8_all24L_routed") > h_acc("oracle_1shot_08b")),
            "e_selfamp_matches_or_beats_joint_reg": bool(h_acc("reg_selfamp_r8_all24L_routed") >= h_acc("joint_reg_r16_all24L") - 0.02),
            "f_zero_general_forgetting_all_routed": bool(
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
    ap.add_argument("--max-synth-per-skill", type=int, default=12)
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
        _print_regularized_summary(payload)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
