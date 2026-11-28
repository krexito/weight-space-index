"""Unified 3-Track Comparison: Maxing Out `Qwen/Qwen3.5-0.8B` on `MATH-500` + `MMLU-400`.

Evaluates all 3 paradigms on the EXACT same held-out `HuggingFaceH4/MATH-500` split (`n_test=200`)
and `cais/mmlu` (`n_test=400`, `371` non-math) + `24` prose controls, with paired 95% bootstrap CIs:

  Track 0 — Base baselines:
    - `base_nothink`         : unmodified Qwen3.5-0.8B (`enable_thinking=False`, max 1536 tok)
    - `base_think`           : unmodified Qwen3.5-0.8B (`enable_thinking=True`,  max 4096 tok)
    - `base_think_rescue`    : `base_think` + zero-cost Answer Rescue before loop kill

  Track 1 — Pure Zero-Training Weight-Space Surgery (`0` gradients, `~100 KB`):
    - `p50c_think` / `p50c_think_rescue` : MIT `p50c` static `.moipatch` (`L15, L19`)
    - `telemetry_multi_think` / `_rescue`: `wsi.surgery_telemetry.diagnose_surgery_sites`
      (`effective_gain = causal_gain * max(hydra_retention, 0)`) multi-layer `.moipatch` (top-3 non-Hydra layers)

  Track 2 — Routed Pocket MoE + LoopRes (`T=2`) + Strata `S2_K` (`0/371` MMLU fires):
    - `lora_r8_routed`       : Regularized `24L r=8` LoRA (`10.32 MB`) + L12 Ridge router
    - `lora_r8_s2k_routed`   : Post-hoc Strata `S2_K` compressed (`1.43 MB`, `7.2x` smaller) + L12 router
    - `loopres_r8_T2_routed` : `24L r=8` LoRA + `LoopRes(core=6..18, T=2)` + L12 router
    - `loopres_r8_T3_routed` : Same LoopRes weights evaluated zero-shot at `T=3`

  Track 3 — Full-Stack Max (Best Track 2 + Test-Time Confidence-Weighted Majority Vote `k=4`):
    - `max_maj4`             : unweighted majority vote over `k=4` sampled `\\boxed{}` answers
    - `max_conf_maj4`        : confidence-weighted vote (`exp(mean_logprob)`) over `k=4` samples
    - `max_pass4_oracle`     : oracle upper bound (`any` of 4 correct)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import discover_math500_blind as dm  # noqa: E402
import eval_capability_and_boundary_lfm3 as ec  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from experiment_pocket_loopres import (  # noqa: E402
    _hf_generate_batched,
    _score_routed_mmlu_sparse_loopres,
    _train_loopres_expert,
)
from experiment_pocket_moe_ceiling import (  # noqa: E402
    fit_ridge_router,
)
from experiment_pocket_moe_controls import _paired_bootstrap  # noqa: E402
from experiment_pocket_moe_math500 import (  # noqa: E402
    PROSE_HELDOUT,
    PROSE_ROUTER_TRAIN,
    _compose_routed_mmlu,
    _grade_math500_rows,
    _score_mmlu_fast,
    _summarize_mmlu,
    _temp_apply_weights,
    format_math500_prompt_ids,
    format_mmlu_prompt_ids,
    load_math500_split,
    load_mmlu_split,
)
from experiment_pocket_moe_math500_strata_quant import (  # noqa: E402
    ROUTER_FIT_CLASSES_MATH,
    _predict_binary_math_router,
)
from experiment_pocket_moe_strata_quant import (  # noqa: E402
    _batched_hf_generate,
    _capture_lora_input_hessians,
    _compress_lora_factors,
    _train_regularized_lora_with_factors,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
)
from experiment_reasoning_distill_v2 import _format_messages  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.loopres import LoopResConfig, LoopResModule, default_core_span  # noqa: E402
from wsi.moipatch import Edit, MoIPatch  # noqa: E402
from wsi.surgery_telemetry import diagnose_surgery_sites  # noqa: E402
from wsi.tracer import _to_seq_tensors  # noqa: E402


# ------------------------------------------------------------------ helpers
def _load_model_and_tok(model_dir: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    dt = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dt, trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _format_think_prompt_ids(tok, problem: str) -> list[int]:
    txt = tok.apply_chat_template(
        [{"role": "user", "content": problem + "\nPut your final answer within \\boxed{}."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tok.encode(txt, add_special_tokens=False)


def _grade_think_with_rescue(
    items: list[dict[str, Any]],
    raw_outs: list[dict[str, Any]],
    tok,
    eos: set[int],
    max_new: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Grade thinking-mode outputs both strictly (loop kill = wrong) and with zero-cost Answer Rescue."""
    strict_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    for r, o in zip(items, raw_outs):
        body = lr.strip_eos(o["token_ids"], eos)
        hit = lr.first_window_hit(lr.dup_flags(body), lr.KILL_WINDOW, lr.KILL_FRAC)
        looped = hit is not None
        cut = body[:hit] if looped else body
        trunc = (not looped) and (o.get("finish_reason") == "length" or len(body) >= max_new)
        text = tok.decode(cut, skip_special_tokens=True)
        pred = dm._extract_boxed(text)
        eq = dm._math_equal(pred, r["answer"])
        strict_ok = (not looped) and eq
        rescue_ok = eq  # Answer Rescue: keep last valid \boxed{} stated before loop kill
        base_rec = {
            "id": r["id"],
            "level": r.get("level", 0),
            "subject": r.get("subject", ""),
            "looped_killed": bool(looped),
            "truncated": bool(trunc),
            "n_tokens": len(body),
            "pred": pred,
            "gold": r["answer"],
        }
        strict_rows.append({**base_rec, "strict_correct": bool(strict_ok)})
        rescue_rows.append({**base_rec, "strict_correct": bool(rescue_ok)})
    return strict_rows, rescue_rows


def _summarize_graded(graded: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(graded), 1)
    return {
        "n_total": len(graded),
        "n_correct": sum(int(x.get("strict_correct", False)) for x in graded),
        "accuracy": round(sum(int(x.get("strict_correct", False)) for x in graded) / n, 4),
        "loop_killed_rate": round(sum(int(x.get("looped_killed", False)) for x in graded) / n, 4),
        "truncated_rate": round(sum(int(x.get("truncated", False)) for x in graded) / n, 4),
        "mean_tokens": round(float(np.mean([x.get("n_tokens", x.get("tokens", 0)) for x in graded])), 1) if graded else 0.0,
    }


@torch.no_grad()
def _generate_with_conf(
    model,
    tok,
    prompt_ids_list: list[list[int]],
    *,
    max_new_tokens: int,
    batch_size: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 42,
    device: str = "cuda",
) -> tuple[list[list[int]], list[float]]:
    """Batched HF sampling returning both generated token IDs and mean token log-prob confidence."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    all_ids: list[list[int]] = []
    all_conf: list[float] = []
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    eos_set = {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()
    if getattr(model.config, "eos_token_id", None) is not None:
        e = model.config.eos_token_id
        eos_set.update(e if isinstance(e, (list, tuple)) else [int(e)])

    for start in range(0, len(prompt_ids_list), batch_size):
        chunk = prompt_ids_list[start : start + batch_size]
        max_p = max(len(p) for p in chunk)
        inp = torch.full((len(chunk), max_p), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), max_p), dtype=torch.long, device=device)
        for i, p in enumerate(chunk):
            inp[i, max_p - len(p) :] = torch.tensor(p, dtype=torch.long, device=device)
            attn[i, max_p - len(p) :] = 1

        out = model.generate(
            input_ids=inp,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            top_k=top_k if temperature > 0.0 else None,
            pad_token_id=pad_id,
        )
        gen_seq = out[:, max_p:]  # [B, T_gen]
        B = gen_seq.shape[0]
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Trim each sequence at EOS/pad, then score exact mean log-prob in small micro-batches (zero OOM!)
        chunk_clean: list[list[int]] = []
        for i in range(B):
            clean = []
            for tid in gen_seq[i].tolist():
                if tid in eos_set or tid == pad_id:
                    break
                clean.append(int(tid))
            chunk_clean.append(clean)
            all_ids.append(clean)

        for i in range(B):
            clean = chunk_clean[i]
            if not clean:
                all_conf.append(-10.0)
                continue
            full_ids = chunk[i] + clean
            p_len = len(chunk[i])
            t_in = torch.tensor([full_ids], dtype=torch.long, device=device)
            logits = model(t_in, use_cache=False).logits[0, p_len - 1 : -1].float()
            targets = t_in[0, p_len:]
            nll = F.cross_entropy(logits, targets, reduction="mean")
            all_conf.append(float(-nll.item()))
            del t_in, logits, targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_ids, all_conf


def _build_telemetry_loop_patch(
    model,
    tok,
    model_dir: str,
    think_prompts: list[list[int]],
    think_raw_outs: list[dict[str, Any]],
    eos: set[int],
    top_k_layers: int = 3,
    target_rel_edit: float = 0.25,
    device: str = "cuda",
) -> tuple[MoIPatch, dict[str, Any]]:
    """Build a zero-training multi-layer `.moipatch` guided by `wsi.surgery_telemetry.diagnose_surgery_sites`."""
    loop_texts: list[str] = []
    healthy_texts: list[str] = []
    for p_ids, o in zip(think_prompts, think_raw_outs):
        body = lr.strip_eos(o["token_ids"], eos)
        hit = lr.first_window_hit(lr.dup_flags(body), lr.KILL_WINDOW, lr.KILL_FRAC)
        if hit is not None and len(loop_texts) < 6:
            snip = p_ids + body[: min(len(body), hit + 32)]
            loop_texts.append(tok.decode(snip[-512:], skip_special_tokens=False))
        elif hit is None and len(body) > 32 and len(healthy_texts) < 6:
            snip = p_ids + body[: min(len(body), 256)]
            healthy_texts.append(tok.decode(snip[-512:], skip_special_tokens=False))

    if not loop_texts or not healthy_texts:
        loop_texts = [
            tok.decode(p[-256:], skip_special_tokens=False) + " Wait, let me recheck. Wait, let me recheck. Wait, let me recheck."
            for p in think_prompts[:3]
        ]
        healthy_texts = [
            tok.decode(p[-256:], skip_special_tokens=False) + " Therefore the final answer is \\boxed{42}."
            for p in think_prompts[:3]
        ]

    _, blocks = find_blocks(model)
    n_layers = len(blocks)

    # Extract per-layer unit loop direction (mean_loop - mean_healthy) across all layers
    @torch.no_grad()
    def _mean_last_hidden_all(texts: list[str]) -> dict[int, torch.Tensor]:
        sums: dict[int, torch.Tensor | None] = {li: None for li in range(n_layers)}
        for txt in texts:
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = model(**ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            for li in range(n_layers):
                v = hs[li + 1][0, -1].float().cpu()
                sums[li] = v if sums[li] is None else (sums[li] + v)
        return {li: sums[li] / max(len(texts), 1) for li in range(n_layers)}

    h_loop = _mean_last_hidden_all(loop_texts)
    h_hlth = _mean_last_hidden_all(healthy_texts)
    dirs_by_layer = {li: (h_loop[li] - h_hlth[li]) / ((h_loop[li] - h_hlth[li]).norm() + 1e-8) for li in range(n_layers)}

    # Target tokens = loop-breaking / conclusion tokens; foil = hesitation/loop tokens
    def _first_tid(words: list[str]) -> list[int]:
        out_ids = []
        for w in words:
            ids = tok.encode(w, add_special_tokens=False)
            if ids:
                out_ids.append(int(ids[0]))
        return list(dict.fromkeys(out_ids)) or [int(tok.eos_token_id or 0)]

    target_tids = _first_tid(["Therefore", "Thus", "So", "\\boxed", "</think>"])
    foil_tids = _first_tid(["Wait", "Hold", "Let", "Hmm", "Again"])

    tel = diagnose_surgery_sites(
        model,
        tok,
        prompts=loop_texts[:4],
        directions_by_layer={li: -dirs_by_layer[li] for li in range(4, n_layers - 2)},
        target_token_ids=target_tids,
        foil_token_ids=foil_tids,
        target_rel_edit=target_rel_edit,
        top_k=top_k_layers + 3,
        device=device,
    )
    sites = tel.get("ranked_sites") or tel.get("sites") or []
    candidates = [s for s in sites if int(s["layer"]) >= 4 and float(s.get("effective_gain", 0.0)) > 0.0]
    if not candidates:
        candidates = sites
    picked = candidates[:top_k_layers]

    edits: list[Edit] = []
    picked_info = []
    for s in picked:
        li = int(s["layer"])
        raw_alpha = float(s.get("suggested_alpha", 4.0))
        alpha = float(min(max(raw_alpha / math.sqrt(max(len(picked), 1)), 1.5), 6.0))
        edits.append(Edit(module=f"layers.{li}.mlp.down_proj", u=dirs_by_layer[li], alpha=alpha))
        picked_info.append({
            "layer": li,
            "effective_gain": round(float(s.get("effective_gain", 0.0)), 4),
            "hydra_retention": round(float(s.get("hydra_retention", 0.0)), 4),
            "alpha": round(alpha, 3),
        })

    patch = MoIPatch(edits=edits, meta={"source": "telemetry_multi_loop_fix", "picked": picked_info})
    patch.fingerprint_from_checkpoint(model_dir)
    return patch, {"n_loop_snips": len(loop_texts), "n_healthy_snips": len(healthy_texts), "picked": picked_info}


def _score_confidence_rerank(
    rows: list[dict[str, Any]],
    samples_ids: list[list[list[int]]],
    samples_conf: list[list[float]],
    tok,
) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Compute unweighted majority@K, confidence-weighted majority@K, and oracle pass@K."""
    K = len(samples_ids)
    N = len(rows)
    maj_rows: list[dict[str, Any]] = []
    conf_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []

    for i, r in enumerate(rows):
        gold = r["answer"]
        counts: dict[str, float] = defaultdict(float)
        conf_w: dict[str, float] = defaultdict(float)
        raw_by_norm: dict[str, str] = {}
        any_ok = False
        for k in range(K):
            txt = tok.decode(samples_ids[k][i], skip_special_tokens=True)
            pred = dm._extract_boxed(txt)
            if dm._math_equal(pred, gold):
                any_ok = True
            norm = dm._norm_latex(pred)
            if norm:
                raw_by_norm.setdefault(norm, pred or "")
                counts[norm] += 1.0
                w = math.exp(max(samples_conf[k][i], -10.0))
                conf_w[norm] += w

        pass_rows.append({"id": r["id"], "strict_correct": bool(any_ok)})
        if counts:
            best_maj = max(counts, key=lambda x: (counts[x], conf_w[x]))
            best_conf = max(conf_w, key=lambda x: conf_w[x])
            maj_rows.append({"id": r["id"], "strict_correct": bool(dm._math_equal(raw_by_norm[best_maj], gold))})
            conf_rows.append({"id": r["id"], "strict_correct": bool(dm._math_equal(raw_by_norm[best_conf], gold))})
        else:
            maj_rows.append({"id": r["id"], "strict_correct": False})
            conf_rows.append({"id": r["id"], "strict_correct": False})

    return {
        f"max_maj{K}": (_summarize_graded(maj_rows), maj_rows),
        f"max_conf_maj{K}": (_summarize_graded(conf_rows), conf_rows),
        f"max_pass{K}_oracle": (_summarize_graded(pass_rows), pass_rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-cache", required=True)
    ap.add_argument("--p50c-patch", default=str(ROOT / "patches" / "qwen35_0.8b_p50c_loop_fix.moipatch"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default="", help="Optional directory to cache each completed arm for instant resume")
    ap.add_argument("--limit", type=int, default=0, help=">0 for fast smoke test on N MATH-500 test items")
    ap.add_argument("--eval-batch-size", type=int, default=100)
    ap.add_argument("--think-batch-size", type=int, default=100)
    ap.add_argument("--loopres-batch-size", type=int, default=64)
    ap.add_argument("--think-max-tokens", type=int, default=4096)
    ap.add_argument("--nothink-max-tokens", type=int, default=1536)
    ap.add_argument("--k-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    ckpt_dir = Path(a.ckpt_dir) if a.ckpt_dir else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt_load(name: str):
        if ckpt_dir is None:
            return None
        p = ckpt_dir / f"{name}.pt"
        if p.exists():
            print(f"  [ckpt] Resuming `{name}` from {p}", flush=True)
            return torch.load(p, map_location="cpu", weights_only=False)
        return None

    def _ckpt_save(name: str, obj: Any) -> None:
        if ckpt_dir is None:
            return
        p = ckpt_dir / f"{name}.pt"
        tmp = p.with_suffix(".pt.tmp")
        torch.save(obj, tmp)
        tmp.replace(p)

    def _gen_logged(label: str, prompts_ids: list[list[int]], seeds: list[int], *, max_new: int, bs: int, param_updates=None):
        t0 = time.perf_counter()
        outs: list[dict[str, Any]] = []
        n_batches = (len(prompts_ids) + bs - 1) // bs
        for bi in range(n_batches):
            lo, hi = bi * bs, min((bi + 1) * bs, len(prompts_ids))
            tb = time.perf_counter()
            part = _batched_hf_generate(
                model, tok, prompts_ids[lo:hi], seeds[lo:hi],
                max_new_tokens=max_new, batch_size=bs, param_updates=param_updates,
            )
            outs.extend(part)
            ntok = sum(len(x["token_ids"]) for x in part)
            dt_b = max(time.perf_counter() - tb, 1e-6)
            print(f"    [{label}] batch {bi+1}/{n_batches} ({hi}/{len(prompts_ids)}) | {dt_b:.1f}s | {ntok/dt_b:,.0f} tok/s", flush=True)
        print(f"    [{label}] total {(time.perf_counter()-t0)/60:.1f} min", flush=True)
        return outs

    t_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    n_test = a.limit if a.limit > 0 else 200
    n_mmlu_test = min(40, a.limit * 4) if a.limit > 0 else 400
    n_steps = 12 if a.limit > 0 else 160
    k_samp = 2 if a.limit > 0 else a.k_samples
    think_max = 512 if a.limit > 0 else a.think_max_tokens
    nothink_max = 384 if a.limit > 0 else a.nothink_max_tokens

    print(f"[max-cmp] device={device} limit={a.limit} n_test={n_test} n_mmlu={n_mmlu_test} steps={n_steps} k={k_samp}", flush=True)

    # 1. Load exact MATH-500 & MMLU splits
    train_math, test_math = load_math500_split(n_train=300, n_test=200, seed=a.seed)
    test_math = test_math[:n_test]
    mmlu_test_full, mmlu_router_train = load_mmlu_split(n_test=400, n_router_train=200, seed=a.seed)
    mmlu_test = mmlu_test_full[:n_mmlu_test]

    model, tok = _load_model_and_tok(a.student, device)
    eos = set()
    if tok.eos_token_id is not None:
        eos.add(int(tok.eos_token_id))
    if getattr(model.config, "eos_token_id", None) is not None:
        e = model.config.eos_token_id
        eos.update(e if isinstance(e, (list, tuple)) else [int(e)])
    letter_ids = ec._resolve_mmlu_letter_ids(tok)

    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    all_layers = list(range(n_layers))
    core_s, core_e = default_core_span(n_layers)
    router_layer = n_layers // 2  # L12 on 24-layer Qwen3.5-0.8B

    nothink_prompts = [format_math500_prompt_ids(tok, r["problem"]) for r in test_math]
    think_prompts = [_format_think_prompt_ids(tok, r["problem"]) for r in test_math]
    test_seeds = [a.seed + i for i in range(len(test_math))]

    results: dict[str, Any] = {
        "config": {
            "student": a.student,
            "teacher": a.teacher,
            "n_test": len(test_math),
            "n_mmlu_test": len(mmlu_test),
            "think_max_tokens": think_max,
            "nothink_max_tokens": nothink_max,
            "k_samples": k_samp,
            "n_steps": n_steps,
            "core_span": [core_s, core_e],
            "router_layer": router_layer,
        },
        "arms": {},
        "mmlu": {},
        "arm_mb": {},
    }
    graded_by_arm: dict[str, list[dict[str, Any]]] = {}

    def _record_arm(name: str, summary: dict[str, Any], graded_rows: list[dict[str, Any]], mb: float | None = None) -> None:
        results["arms"][name] = summary
        graded_by_arm[name] = graded_rows
        if mb is not None:
            results["arm_mb"][name] = round(mb, 3)
        print(
            f"  [ARM {name:24s}] MATH={summary['accuracy']*100:5.1f}% ({summary['n_correct']:3d}/{summary['n_total']}) "
            f"| loops={summary.get('loop_killed_rate', 0.0)*100:4.1f}% | trunc={summary.get('truncated_rate', 0.0)*100:4.1f}%",
            flush=True,
        )

    # ================================================================== TRACK 0: BASE
    print("\n=== TRACK 0: BASELINES (no-think & think) ===", flush=True)
    base_nt_outs = _ckpt_load("base_nothink_outs")
    if base_nt_outs is None:
        base_nt_outs = _gen_logged("base_nothink", nothink_prompts, test_seeds, max_new=nothink_max, bs=a.eval_batch_size)
        _ckpt_save("base_nothink_outs", base_nt_outs)
    base_nt_graded = _grade_math500_rows(test_math, base_nt_outs, tok, eos, nothink_max)
    _record_arm("base_nothink", _summarize_graded(base_nt_graded), base_nt_graded, 0.0)

    base_th_outs = _ckpt_load("base_think_outs")
    if base_th_outs is None:
        base_th_outs = _gen_logged("base_think", think_prompts, test_seeds, max_new=think_max, bs=a.think_batch_size)
        _ckpt_save("base_think_outs", base_th_outs)
    base_th_strict, base_th_rescue = _grade_think_with_rescue(test_math, base_th_outs, tok, eos, think_max)
    _record_arm("base_think", _summarize_graded(base_th_strict), base_th_strict, 0.0)
    _record_arm("base_think_rescue", _summarize_graded(base_th_rescue), base_th_rescue, 0.0)

    # MMLU + prose base reference
    mmlu_test_pids = [format_mmlu_prompt_ids(tok, r) for r in mmlu_test]
    base_mmlu_raw = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)
    results["mmlu"]["base"] = _summarize_mmlu(mmlu_test, base_mmlu_raw)

    prose_seqs = _to_seq_tensors(
        [_format_messages(tok, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_HELDOUT[:8]],
        dev,
    )
    base_prose_lp = []
    with torch.no_grad():
        for sq in prose_seqs:
            logits = model(sq.to(device)).logits[0, :-1].float()
            base_prose_lp.append(F.log_softmax(logits, dim=-1).cpu())

    # ================================================================== TRACK 1: SURGICAL 0-GRAD
    print("\n=== TRACK 1: PURE ZERO-TRAINING WEIGHT-SPACE SURGERY ===", flush=True)
    if Path(a.p50c_patch).exists():
        p50c = MoIPatch.load(a.p50c_patch)
        p50c_mb = Path(a.p50c_patch).stat().st_size / (1024 * 1024)
        p50c_ck = _ckpt_load("p50c_think")
        if p50c_ck is None:
            with p50c.attach(model) as h:
                h.on()
                p50c_outs = _gen_logged("p50c_think", think_prompts, test_seeds, max_new=think_max, bs=a.think_batch_size)
                p50c_mmlu_raw = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)
                p50c_mmlu_sum = _summarize_mmlu(mmlu_test, p50c_mmlu_raw, base_score=base_mmlu_raw)
            _ckpt_save("p50c_think", {"outs": p50c_outs, "mmlu": p50c_mmlu_sum})
        else:
            p50c_outs, p50c_mmlu_sum = p50c_ck["outs"], p50c_ck["mmlu"]
        results["mmlu"]["p50c_think"] = p50c_mmlu_sum
        p50c_s, p50c_r = _grade_think_with_rescue(test_math, p50c_outs, tok, eos, think_max)
        _record_arm("p50c_think", _summarize_graded(p50c_s), p50c_s, p50c_mb)
        _record_arm("p50c_think_rescue", _summarize_graded(p50c_r), p50c_r, p50c_mb)

    tel_ck = _ckpt_load("telemetry_multi_think")
    if tel_ck is None:
        tel_patch, tel_meta = _build_telemetry_loop_patch(
            model, tok, a.student, think_prompts, base_th_outs, eos, top_k_layers=3, target_rel_edit=0.25, device=device
        )
        tel_tmp = Path(a.out).with_suffix(".telemetry.moipatch")
        tel_patch.save(tel_tmp)
        tel_mb = tel_tmp.stat().st_size / (1024 * 1024)
        with tel_patch.attach(model) as h:
            h.on()
            tel_outs = _gen_logged("telemetry_multi", think_prompts, test_seeds, max_new=think_max, bs=a.think_batch_size)
            tel_mmlu_raw = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)
            tel_mmlu_sum = _summarize_mmlu(mmlu_test, tel_mmlu_raw, base_score=base_mmlu_raw)
        _ckpt_save("telemetry_multi_think", {"outs": tel_outs, "mmlu": tel_mmlu_sum, "meta": tel_meta, "mb": tel_mb})
    else:
        tel_outs, tel_mmlu_sum, tel_meta, tel_mb = tel_ck["outs"], tel_ck["mmlu"], tel_ck["meta"], tel_ck["mb"]
    results["telemetry_meta"] = tel_meta
    results["mmlu"]["telemetry_multi_think"] = tel_mmlu_sum
    tel_s, tel_r = _grade_think_with_rescue(test_math, tel_outs, tok, eos, think_max)
    _record_arm("telemetry_multi_think", _summarize_graded(tel_s), tel_s, tel_mb)
    _record_arm("telemetry_multi_rescue", _summarize_graded(tel_r), tel_r, tel_mb)

    # ================================================================== TRACK 2: POCKET MoE + LOOPRES + STRATA
    print("\n=== TRACK 2: ROUTED POCKET MoE + LOOPRES (T=2) + STRATA S2_K ===", flush=True)
    # 2a. Fit L12 Ridge Router
    r_math_pids = [format_math500_prompt_ids(tok, r["problem"]) for r in train_math[:120]]
    r_math_lbls = [r["domain"] for r in train_math[:120]]
    r_mmlu_pids = [format_mmlu_prompt_ids(tok, r) for r in mmlu_router_train[:80]]
    r_prose_pids = [_format_messages(tok, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_ROUTER_TRAIN]

    X_tr = _extract_prompt_hidden_vecs(model, r_math_pids + r_mmlu_pids + r_prose_pids, router_layer)
    y_tr = r_math_lbls + ["pass_mmlu"] * len(r_mmlu_pids) + ["pass_prose"] * len(r_prose_pids)
    W_r, b_r = fit_ridge_router(X_tr, y_tr, ROUTER_FIT_CLASSES_MATH, lam=10.0)

    X_math_te = _extract_prompt_hidden_vecs(model, nothink_prompts, router_layer)
    math_route = _predict_binary_math_router(X_math_te, W_r, b_r)
    X_mmlu_te = _extract_prompt_hidden_vecs(model, mmlu_test_pids, router_layer)
    mmlu_route = _predict_binary_math_router(X_mmlu_te, W_r, b_r)
    X_prose_te = _extract_prompt_hidden_vecs(
        model,
        [_format_messages(tok, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_HELDOUT],
        router_layer,
    )
    prose_route = _predict_binary_math_router(X_prose_te, W_r, b_r)
    nonmath_mmlu_fires = sum(1 for r, p in zip(mmlu_test, mmlu_route) if (not r.get("is_math", False)) and p == "math")
    nonmath_mmlu_tot = sum(1 for r in mmlu_test if not r.get("is_math", False))
    results["router_report"] = {
        "layer": router_layer,
        "math_fire_rate": round(sum(p == "math" for p in math_route) / max(len(math_route), 1), 4),
        "nonmath_mmlu_fires": f"{nonmath_mmlu_fires}/{nonmath_mmlu_tot}",
        "prose_fires": f"{sum(p == 'math' for p in prose_route)}/{len(prose_route)}",
    }
    print("  [Router L12]", results["router_report"], flush=True)

    # 2b. Load cached 4B teacher traces (195 clean MATH-500 traces + topk logits)
    cache_p = Path(a.teacher_cache)
    if not cache_p.exists():
        raise FileNotFoundError(f"Teacher cache missing: {cache_p}")
    t_cache = torch.load(cache_p, map_location="cpu", weights_only=False)
    tr_samples = t_cache["kept_samples"]
    print(f"  [Cache] Loaded {len(tr_samples)} clean 4B MATH-500 teacher traces from {cache_p}", flush=True)

    # 2c. Train Regularized r=8 LoRA + compress to Strata S2_K
    lora_res = _train_regularized_lora_with_factors(
        model,
        all_layers,
        tr_samples,
        rank=8,
        alpha_scale=16.0,
        total_steps=n_steps,
        grad_accum=4,
        lr_val=2e-4,
        anchor_ids=prose_seqs,
        base_anchor_logp=base_prose_lp,
        anchor_kl_weight=0.1,
        seed=a.seed,
    )
    lora_w = lora_res["weights"]
    lora_mb = float(lora_res.get("adapter_mb", 10.32))

    lora_outs = _batched_hf_generate(
        model, tok, nothink_prompts, test_seeds, max_new_tokens=nothink_max, batch_size=a.eval_batch_size, param_updates=lora_w
    )
    lora_routed_outs = [lg if rt == "math" else bg for lg, bg, rt in zip(lora_outs, base_nt_outs, math_route)]
    lora_graded = _grade_math500_rows(test_math, lora_routed_outs, tok, eos, nothink_max)
    _record_arm("lora_r8_routed", _summarize_graded(lora_graded), lora_graded, lora_mb)
    with _temp_apply_weights(model, lora_w):
        lora_mmlu_raw = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)
    results["mmlu"]["lora_r8_routed"] = _compose_routed_mmlu(mmlu_test, mmlu_route, base_mmlu_raw, {"math": lora_mmlu_raw})

    # Compress r=8 -> Strata S2_K
    hess = _capture_lora_input_hessians(model, all_layers, tr_samples[:32])
    s2k_deltas, s2k_bytes, s2k_herr = _compress_lora_factors(
        lora_res["factors"], hess, code_name="S2_K", method="strata", device=dev
    )
    s2k_mb = float(s2k_bytes / (1024 * 1024))
    results["s2k_hess_err"] = round(float(s2k_herr), 4)
    s2k_outs = _batched_hf_generate(
        model, tok, nothink_prompts, test_seeds, max_new_tokens=nothink_max, batch_size=a.eval_batch_size, param_updates=s2k_deltas
    )
    s2k_routed_outs = [sg if rt == "math" else bg for sg, bg, rt in zip(s2k_outs, base_nt_outs, math_route)]
    s2k_graded = _grade_math500_rows(test_math, s2k_routed_outs, tok, eos, nothink_max)
    _record_arm("lora_r8_s2k_routed", _summarize_graded(s2k_graded), s2k_graded, s2k_mb)
    with _temp_apply_weights(model, s2k_deltas):
        s2k_mmlu_raw = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)
    results["mmlu"]["lora_r8_s2k_routed"] = _compose_routed_mmlu(mmlu_test, mmlu_route, base_mmlu_raw, {"math": s2k_mmlu_raw})

    # 2d. Train Pocket LoopRes (r=8 + LoopRes core=6..18, T=2) on Qwen3.5-0.8B!
    lr_res = _train_loopres_expert(
        model,
        all_layers,
        tr_samples,
        rank=8,
        alpha_scale=16.0,
        loopres_mode="loopres",
        core_start=core_s,
        core_end=core_e,
        n_loops=2,
        n_loops_choices=(2,),
        window=2,
        d_loop_key=64,
        total_steps=n_steps,
        grad_accum=4,
        lr_lora=2e-4,
        lr_loopres=1e-3,
        anchor_ids=prose_seqs,
        base_anchor_logp=base_prose_lp,
        anchor_kl_weight=0.1,
        seed=a.seed,
    )
    lr_deltas = lr_res["weights"]
    lr_mod = LoopResModule(LoopResConfig.from_dict(lr_res["loopres_config"])).to(dev)
    lr_mod.load_state_dict(lr_res["loopres_state"])
    lr_mod.eval()
    lr_mb = float(lr_res.get("adapter_mb", lora_mb + 0.08))

    for T_eval in (2, 3):
        t_outs = _hf_generate_batched(
            model,
            tok,
            nothink_prompts,
            max_new_tokens=nothink_max,
            batch_size=a.loopres_batch_size,
            base_seed=a.seed,
            loopres_mod=lr_mod,
            n_loops=T_eval,
            param_updates=lr_deltas,
        )
        if T_eval == 2:
            lr_mmlu_raw = _score_routed_mmlu_sparse_loopres(
                model,
                mmlu_test_pids,
                mmlu_test,
                mmlu_route,
                "math",
                base_mmlu_raw,
                letter_ids,
                dev,
                param_updates=lr_deltas,
                loopres_mod=lr_mod,
                n_loops=2,
            )
            results["mmlu"]["loopres_r8_T2_routed"] = _summarize_mmlu(
                mmlu_test, lr_mmlu_raw, base_score=base_mmlu_raw, fire_flags=[r == "math" for r in mmlu_route]
            )
        t_routed = [tg if rt == "math" else bg for tg, bg, rt in zip(t_outs, base_nt_outs, math_route)]
        t_graded = _grade_math500_rows(test_math, t_routed, tok, eos, nothink_max)
        _record_arm(f"loopres_r8_T{T_eval}_routed", _summarize_graded(t_graded), t_graded, lr_mb)

    # ================================================================== TRACK 3: FULL-STACK MAX (CONFIDENCE RERANK k=4)
    print(f"\n=== TRACK 3: FULL-STACK MAX (Best Track 2 + Confidence-Weighted Rerank k={k_samp}) ===", flush=True)
    best_t2_name = max(
        ("lora_r8_routed", "lora_r8_s2k_routed", "loopres_r8_T2_routed", "loopres_r8_T3_routed"),
        key=lambda nm: results["arms"][nm]["accuracy"],
    )
    results["track3_backbone"] = best_t2_name
    print(f"  [Track 3] Using backbone: {best_t2_name}", flush=True)

    samp_ids: list[list[list[int]]] = []
    samp_conf: list[list[float]] = []
    if best_t2_name.startswith("loopres_"):
        t_loops = 3 if best_t2_name.endswith("T3_routed") else 2
        t3_mb = lr_mb
        with _temp_apply_weights(model, lr_deltas), lr_mod.attach(model, n_loops=t_loops):
            for k in range(k_samp):
                tb = time.perf_counter()
                ids_k, conf_k = _generate_with_conf(
                    model, tok, nothink_prompts, max_new_tokens=nothink_max,
                    batch_size=a.loopres_batch_size, seed=a.seed + 100 * (k + 1), device=device,
                )
                samp_ids.append(ids_k)
                samp_conf.append(conf_k)
                print(f"    [Track 3 {best_t2_name}] sample {k+1}/{k_samp} done in {(time.perf_counter()-tb)/60:.1f} min", flush=True)
    else:
        w_use = s2k_deltas if best_t2_name == "lora_r8_s2k_routed" else lora_w
        t3_mb = s2k_mb if best_t2_name == "lora_r8_s2k_routed" else lora_mb
        with _temp_apply_weights(model, w_use):
            for k in range(k_samp):
                tb = time.perf_counter()
                ids_k, conf_k = _generate_with_conf(
                    model, tok, nothink_prompts, max_new_tokens=nothink_max,
                    batch_size=a.eval_batch_size, seed=a.seed + 100 * (k + 1), device=device,
                )
                samp_ids.append(ids_k)
                samp_conf.append(conf_k)
                print(f"    [Track 3 {best_t2_name}] sample {k+1}/{k_samp} done in {(time.perf_counter()-tb)/60:.1f} min", flush=True)

    rerank = _score_confidence_rerank(test_math, samp_ids, samp_conf, tok)
    for nm, (summary, g_rows) in rerank.items():
        _record_arm(nm, summary, g_rows, t3_mb)

    # Paired bootstrap 95% CIs across the ladder
    pairs = [
        ("p50c_think_rescue", "base_think"),
        ("telemetry_multi_rescue", "base_think"),
        ("lora_r8_routed", "base_nothink"),
        ("lora_r8_s2k_routed", "lora_r8_routed"),
        ("loopres_r8_T2_routed", "lora_r8_routed"),
        ("loopres_r8_T2_routed", "base_nothink"),
        (f"max_conf_maj{k_samp}", "base_nothink"),
        (f"max_conf_maj{k_samp}", results["track3_backbone"]),
    ]
    results["paired_ci95"] = {
        f"{x}_vs_{y}": _paired_bootstrap(graded_by_arm[x], graded_by_arm[y], key="strict_correct")
        for x, y in pairs
        if x in graded_by_arm and y in graded_by_arm
    }
    results["elapsed_min"] = round((time.perf_counter() - t_start) / 60.0, 2)

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist() if obj.numel() <= 64 else f"<Tensor shape={tuple(obj.shape)}>"
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size <= 64 else f"<ndarray shape={obj.shape}>"
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    _atomic_write_json(Path(a.out), _json_safe(results))
    print(f"\n=== DONE in {results['elapsed_min']} min -> saved {a.out} ===", flush=True)
    for k, v in results["paired_ci95"].items():
        print(f"  {k:38s}: {v['diff']*100:+5.1f} pts  CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}]", flush=True)


if __name__ == "__main__":
    main()
