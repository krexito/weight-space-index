"""Thinking-to-Weights Compression: Distilling 4B `<think>` Chains into Routed 0.8B No-Think Cartridges on MATH-500.

Tests on real-world `HuggingFaceH4/MATH-500` (`n_train=300`, `n_test=200`) + `cais/mmlu` (`n_test=400`) + `24` held-out
prose controls whether a routed full-depth LoRA cartridge (`r=16`, all 24L, `~20.6 MB`, Layer-12 router) can compress
long `Qwen3.5-4B` `<think>` reasoning into direct **`enable_thinking=False`** inference on `Qwen3.5-0.8B`:

  1. `base_08b_nothink`                     : Unmodified Qwen3.5-0.8B (`enable_thinking=False`, max 2048 tok)
  2. `base_08b_think`                       : Unmodified Qwen3.5-0.8B (`enable_thinking=True`,  max 4096 tok)
  3. `teacher_4b_nothink`                   : Unmodified Qwen3.5-4B   (`enable_thinking=False`, max 2048 tok)
  4. `teacher_4b_think`                     : Unmodified Qwen3.5-4B   (`enable_thinking=True`,  max 4096 tok)
  5. `nothink_distill_r16_24L_routed`       : Baseline `4B nothink` traces + `4B nothink` Top-32 KL (160x4=640 passes)
  6. `think_post_nothink_kl_r16_24L_routed` : `4B think` post-`</think>` solutions + `4B nothink` Top-32 KL (same 640 passes)
  7. `think_privileged_kl_r16_24L_routed`   : Exact same `4B think` post-`</think>` solutions + **privileged `4B think`
                                              Top-32 KL conditioned on full `<think>...</think>`** while student sees
                                              **zero `<think>` tokens** (same 640 passes, matched seed init)

CPU smoke test::
    python scripts/experiment_pocket_moe_think_compress.py --smoke

Colab L4 run::
    python scripts/experiment_pocket_moe_think_compress.py --teacher /content/teacher_4b --student /content/student_08b --out /content/pocket_moe_think_compress_l4.json
"""
from __future__ import annotations

import argparse
import gc
import json
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

import discover_math500_blind as dm  # noqa: E402
import eval_capability_and_boundary_lfm3 as ec  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from experiment_pocket_moe_ceiling import (  # noqa: E402
    _bake_full_checkpoint,
    _hf_generate_subset,
    _vllm_generate,
    fit_ridge_router,
    predict_ridge_router,
)
from experiment_pocket_moe_controls import _paired_bootstrap  # noqa: E402
from experiment_pocket_moe_math500 import (  # noqa: E402
    MATH_DOMAINS,
    PROSE_HELDOUT,
    PROSE_ROUTER_TRAIN,
    ROUTER_CLASSES,
    _compose_routed_mmlu,
    _grade_math500_rows,
    _score_mmlu_fast,
    _summarize_math500_arm,
    _summarize_mmlu,
    _temp_apply_weights,
    format_math500_prompt_ids,
    format_mmlu_prompt_ids,
    load_math500_split,
    load_mmlu_split,
)
from experiment_pocket_moe_regularized import (  # noqa: E402
    _score_routed_mmlu_sparse,
    _train_regularized_lora_expert,
)
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _PlantedReasoningLM,
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
)
from experiment_reasoning_distill_v2 import (  # noqa: E402
    _format_messages,
    _select_backend,
)
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402


# ------------------------------------------------------------------ Thinking Split & Privileged Alignment Helpers
def format_math500_think_prompt_ids(tok: Any, problem: str) -> list[int]:
    content = problem.strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
    return _format_messages(tok, [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=True)


def _split_think_output(
    token_ids: list[int],
    tok: Any,
    eos_set: set[int],
    think_end_id: int | None,
) -> dict[str, Any]:
    """Split a thinking-mode generation into `(closed_think, think_prefix_ids, post_think_text, n_think_tokens, n_answer_tokens)`."""
    body_ids = lr.strip_eos(list(token_ids), eos_set)
    if think_end_id is not None and think_end_id in body_ids:
        end_pos = body_ids.index(think_end_id) + 1
        think_prefix_ids = body_ids[:end_pos]
        post_ids = body_ids[end_pos:]
        # Keep any leading whitespace-only tokens attached to the teacher's </think> prefix
        ws_take = 0
        while ws_take < len(post_ids):
            piece = tok.decode([post_ids[ws_take]], skip_special_tokens=False)
            if piece.strip() == "":
                ws_take += 1
            else:
                break
        think_prefix_ids = think_prefix_ids + post_ids[:ws_take]
        ans_raw_ids = post_ids[ws_take:]
        post_text = tok.decode(ans_raw_ids, skip_special_tokens=True).strip()
        return {
            "closed_think": True,
            "think_prefix_ids": think_prefix_ids,
            "post_think_text": post_text,
            "n_think_tokens": len(think_prefix_ids),
            "n_answer_tokens": len(ans_raw_ids),
        }

    full_txt = tok.decode(body_ids, skip_special_tokens=True)
    if "</think>" in full_txt:
        pre_s, post_s = full_txt.split("</think>", 1)
        think_prefix_ids = [int(x) for x in tok.encode(pre_s + "</think>\n\n", add_special_tokens=False)]
        post_text = post_s.strip()
        ans_ids = [int(x) for x in tok.encode(post_text, add_special_tokens=False)]
        return {
            "closed_think": True,
            "think_prefix_ids": think_prefix_ids,
            "post_think_text": post_text,
            "n_think_tokens": len(think_prefix_ids),
            "n_answer_tokens": len(ans_ids),
        }

    return {
        "closed_think": False,
        "think_prefix_ids": body_ids,
        "post_think_text": "",
        "n_think_tokens": len(body_ids),
        "n_answer_tokens": 0,
    }


def _build_aligned_privileged_pair(
    tok: Any,
    problem: str,
    post_think_text: str,
    think_prefix_ids: list[int],
) -> dict[str, Any]:
    """Construct token-exact `(nothink_seq, privileged_think_seq)` sharing the exact same assistant target tokens."""
    content = problem.strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
    p_nothink = _format_messages(tok, [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=False)
    full_nothink = _format_messages(
        tok,
        [{"role": "user", "content": content}, {"role": "assistant", "content": post_think_text}],
        add_generation_prompt=False,
        enable_thinking=False,
    )
    lo_s = max(0, len(p_nothink) - 1)
    hi_s = len(full_nothink) - 1
    ans_ids = full_nothink[lo_s + 1 : hi_s + 1]

    p_think = _format_messages(tok, [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=True)
    prefix_t = p_think + list(think_prefix_ids)
    full_think = prefix_t + ans_ids
    lo_t = max(0, len(prefix_t) - 1)
    hi_t = len(full_think) - 1

    assert hi_t - lo_t == hi_s - lo_s == len(ans_ids), (hi_t - lo_t, hi_s - lo_s, len(ans_ids))
    assert full_think[lo_t + 1 : hi_t + 1] == full_nothink[lo_s + 1 : hi_s + 1]
    return {
        "nothink_ids": full_nothink,
        "lo_s": lo_s,
        "hi_s": hi_s,
        "think_ids": full_think,
        "lo_t": lo_t,
        "hi_t": hi_t,
        "n_target_tokens": len(ans_ids),
    }


def _compute_think_diagnostics(
    raw_outs: list[dict[str, Any]],
    tok: Any,
    eos_set: set[int],
    think_end_id: int | None,
) -> dict[str, Any]:
    closed = 0
    think_toks: list[int] = []
    ans_toks: list[int] = []
    for raw in raw_outs:
        sp = _split_think_output(raw["token_ids"], tok, eos_set, think_end_id)
        if sp["closed_think"]:
            closed += 1
        think_toks.append(int(sp["n_think_tokens"]))
        ans_toks.append(int(sp["n_answer_tokens"]))
    n = max(1, len(raw_outs))
    return {
        "closed_think_rate": round(closed / n, 4),
        "mean_think_tokens": round(sum(think_toks) / n, 1),
        "median_think_tokens": round(float(statistics.median(think_toks)), 1) if think_toks else 0.0,
        "mean_answer_tokens": round(sum(ans_toks) / n, 1),
    }


# ------------------------------------------------------------------ Compact Summary Printer
def _print_think_compress_summary(payload: dict[str, Any]) -> None:
    """Print concise human-readable report + single-line compact JSON omitting per-item rows."""
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return

    print("\n=== 1. COBERTURA TEACHER (NOTHINK vs THINK) & ROUTER ===", flush=True)
    print("Teacher train coverage:", payload.get("teacher_train_coverage"), flush=True)
    print("Compile time (s):", payload.get("compile_time_s"), flush=True)
    print("Router report:", payload.get("router_report"), flush=True)

    prose_kl = payload.get("prose_kl", {})
    print("\n=== 2. MATH-500 (n=200, 7 BRAZOS PAREADOS: COMPRESION DE <think> -> SIN <think>) ===", flush=True)
    for arm_nm, st in payload.get("math500_arms", {}).items():
        pkl = prose_kl.get(arm_nm, 0.0)
        th_only = st.get("think_only_hard_subset_acc")
        th_s = f"{th_only*100:5.1f}%" if th_only is not None else "   n/a"
        comp_x = st.get("token_compression_vs_base_think")
        comp_s = f"{comp_x:4.2f}x" if comp_x is not None else "  n/a"
        print(
            f"  {arm_nm:38s} | acc={st['accuracy']*100:5.1f}% ({st['n_correct']:3d}/{st['n_total']}) "
            f"| think_only={th_s} | tok_mean={st['mean_tokens']:6.1f} (med={st['median_tokens']:6.1f}, {comp_s}) "
            f"| trunc={st['truncated_rate']*100:4.1f}% loop={st['loop_killed_rate']*100:4.1f}% | pkl={pkl:.5f}",
            flush=True,
        )
        print(f"     per_level={st['per_level']} | per_domain={st['per_domain']}", flush=True)

    print("\n=== 3. RETENCION MMLU (n=400) ===", flush=True)
    for arm_nm, vm in payload.get("mmlu_arms", {}).items():
        a_m, nm = vm["all_400"], vm["non_math"]
        d_nm = f"{nm['delta_vs_base']*100:+5.1f} pts" if "delta_vs_base" in nm else "     base"
        print(
            f"  {arm_nm:38s} | MMLU={a_m['accuracy']*100:5.1f}% | non_math={nm['accuracy']*100:5.1f}% ({d_nm}) | "
            f"ABCD={a_m['mean_abcd_prob_mass']*100:5.1f}% | fire={vm['router_fire_rate']*100:5.1f}%",
            flush=True,
        )

    print("\n=== 4. COMPARACIONES PAREADAS (IC 95% BOOTSTRAP, MATH-500 n=200) ===", flush=True)
    for k, v in payload.get("paired_math500_200", {}).items():
        print(
            f"  {k:44s}: diff={v['diff']*100:+5.1f} pts "
            f"CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})",
            flush=True,
        )

    print("\nCriteria:", payload.get("criteria"), flush=True)
    print("Timing (s):", payload.get("timing_s"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_math500"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


# ------------------------------------------------------------------ CPU Smoke Test
def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of think splitting, privileged token alignment, privileged KL distillation & sparse MMLU."""

    class _ToyTok:
        def __init__(self) -> None:
            self._vocab = {"</think>": 99, "\n\n": 98}

        def apply_chat_template(
            self, messages: list[dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True, enable_thinking: bool = False
        ) -> str:
            parts = []
            for m in messages:
                parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
            if add_generation_prompt:
                parts.append("<assistant><think>\n" if enable_thinking else "<assistant>")
            return "|".join(parts)

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            out: list[int] = []
            for ch in text:
                out.append(1 + (ord(ch) % 24))
            return out

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            out = []
            for x in ids:
                if x == 99:
                    if not skip_special_tokens:
                        out.append("</think>")
                elif x == 98:
                    out.append("\n\n")
                else:
                    out.append(chr(96 + (x % 26)))
            return "".join(out)

    toy_tok = _ToyTok()
    # 1. _split_think_output with </think> token present and missing
    sp_ok = _split_think_output([1, 2, 3, 99, 98, 5, 6, 7], toy_tok, eos_set=set(), think_end_id=99)
    assert sp_ok["closed_think"] is True and sp_ok["n_think_tokens"] == 5 and sp_ok["n_answer_tokens"] == 3, sp_ok
    assert sp_ok["post_think_text"] == "efg", sp_ok
    sp_miss = _split_think_output([1, 2, 3, 4], toy_tok, eos_set=set(), think_end_id=99)
    assert sp_miss["closed_think"] is False and sp_miss["post_think_text"] == "", sp_miss

    # 2. _build_aligned_privileged_pair exact token alignment
    aligned = _build_aligned_privileged_pair(toy_tok, "Compute 2+2.", "The answer is 4.", [1, 2, 3, 99, 98])
    assert aligned["n_target_tokens"] > 0
    assert aligned["hi_t"] - aligned["lo_t"] == aligned["hi_s"] - aligned["lo_s"] == aligned["n_target_tokens"]
    assert aligned["think_ids"][aligned["lo_t"] + 1 : aligned["hi_t"] + 1] == aligned["nothink_ids"][aligned["lo_s"] + 1 : aligned["hi_s"] + 1]

    # 3. Toy privileged vs nothink KL distillation on _PlantedReasoningLM
    model = _PlantedReasoningLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Student sees short prompt [1,2,4] + ans [10,20,20,20]; privileged teacher sees longer think prefix [3,3,1,2,4] + same ans
    base_pairs = [
        ([1, 2, 4, 10, 20, 20, 20, 20], [3, 3, 1, 2, 4, 10, 20, 20, 20, 20]),
        ([2, 1, 4, 11, 20, 20, 20, 20], [3, 2, 2, 1, 4, 11, 20, 20, 20, 20]),
    ]
    nothink_samples: list[dict[str, Any]] = []
    priv_samples: list[dict[str, Any]] = []
    for s_seq, t_seq in base_pairs:
        lo_s, hi_s = 3, 7
        lo_t, hi_t = 5, 9
        assert s_seq[lo_s + 1 : hi_s + 1] == t_seq[lo_t + 1 : hi_t + 1]
        with torch.no_grad():
            nt_log = model(torch.tensor(s_seq, dtype=torch.long).unsqueeze(0)).logits[0, lo_s:hi_s].float()
            pr_log = model(torch.tensor(t_seq, dtype=torch.long).unsqueeze(0)).logits[0, lo_t:hi_t].float()
        v_nt, i_nt = nt_log.topk(8, dim=-1)
        v_pr, i_pr = pr_log.topk(8, dim=-1)
        nothink_samples.append({"ids": s_seq, "lo": lo_s, "hi": hi_s, "topk_ids": i_nt.to(torch.int32), "topk_logits": v_nt.to(torch.float16)})
        priv_samples.append({"ids": s_seq, "lo": lo_s, "hi": hi_s, "topk_ids": i_pr.to(torch.int32), "topk_logits": v_pr.to(torch.float16)})

    h_ids = [torch.tensor([1, 2, 3, 1], dtype=torch.long).unsqueeze(0)]
    base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1).cpu()]
    base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    ckpts_out: dict[str, list[float]] = {}
    last_weights: dict[str, torch.Tensor] = {}
    for nm, ds_toy in [("nothink_kl", nothink_samples), ("privileged_think_kl", priv_samples)]:
        torch.manual_seed(42)
        res_e = _train_regularized_lora_expert(
            model,
            [0, 1],
            ds_toy,
            rank=4,
            alpha_scale=8.0,
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
        assert all(torch.equal(v.cpu(), orig_snapshot[k]) for k, v in model.state_dict().items()), nm
        ckpts = res_e["step_losses_checkpoints"]
        assert ckpts[-1] < ckpts[0], (nm, ckpts)
        ckpts_out[nm] = ckpts
        last_weights = res_e["weights"]

    # 4. Sparse MMLU exact equivalence
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
    with _temp_apply_weights(model, last_weights):
        full_ed = _score_mmlu_fast(model, fake_pids, fake_mmlu, letter_ids, dev)
        sub_cls = ["passthrough", "algebra", "passthrough", "algebra"]
        sp_ed = _score_routed_mmlu_sparse(model, fake_pids, fake_mmlu, sub_cls, "algebra", base_mmlu, letter_ids, dev)
    assert _compose_routed_mmlu(fake_mmlu, sub_cls, base_mmlu, {"algebra": sp_ed}, seed=1) == _compose_routed_mmlu(
        fake_mmlu, sub_cls, base_mmlu, {"algebra": full_ed}, seed=1
    )

    return {
        "mode": "smoke",
        "passed": True,
        "think_split_ok": True,
        "privileged_alignment_ok": True,
        "toy_loss_checkpoints": ckpts_out,
        "sparse_mmlu_exact_match": True,
    }


# ------------------------------------------------------------------ Full GPU Thinking-Compression Experiment
def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    backend = _select_backend(args.backend)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        dtype = (
            torch.bfloat16
            if (dev.type == "cuda" and torch.cuda.is_bf16_supported())
            else (torch.float16 if dev.type == "cuda" else torch.float32)
        )
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    dtype_vllm_str = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype]

    math_train, math_test = load_math500_split(n_train=args.n_math_train, n_test=args.n_math_test, seed=args.seed)
    mmlu_test, mmlu_router_train = load_mmlu_split(
        n_test=args.n_mmlu_test, n_router_train=args.n_mmlu_router_train, seed=args.seed
    )

    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    assert len(tok_t) == len(tok_s), f"Tokenizer vocab mismatch: teacher={len(tok_t)} vs student={len(tok_s)}"
    eos_set = set(lr.eos_ids(tok_s))
    think_end_id = tok_s.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_id, int) or think_end_id == tok_s.unk_token_id:
        think_end_id = None

    # ---- Phase 1A: Teacher 4B Generations (No-Think @ 2048 & Think @ 4096) ----
    print(
        f"[think-cmp] Phase 1A: Teacher ({args.teacher}) generation ({backend}) — "
        f"nothink@{args.max_new_tokens_nothink} & think@{args.max_new_tokens_think}...",
        flush=True,
    )
    t_p1 = time.perf_counter()
    math_test_nt_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_test]
    math_train_nt_pids = [format_math500_prompt_ids(tok_s, it["problem"]) for it in math_train]
    math_test_th_pids = [format_math500_think_prompt_ids(tok_s, it["problem"]) for it in math_test]
    math_train_th_pids = [format_math500_think_prompt_ids(tok_s, it["problem"]) for it in math_train]

    test_seeds = [int(args.seed) + i for i in range(len(math_test))]
    train_seeds = [int(args.seed) + 1000 + i for i in range(len(math_train))]

    teacher_hf = None
    if backend == "vllm":
        nt_outs = _vllm_generate(
            args.teacher,
            tok_t,
            math_test_nt_pids + math_train_nt_pids,
            test_seeds + train_seeds,
            max_new_tokens=args.max_new_tokens_nothink,
            dtype_str=dtype_vllm_str,
            base_seed=args.seed,
        )
        th_outs = _vllm_generate(
            args.teacher,
            tok_t,
            math_test_th_pids + math_train_th_pids,
            test_seeds + train_seeds,
            max_new_tokens=args.max_new_tokens_think,
            dtype_str=dtype_vllm_str,
            base_seed=args.seed + 7,
        )
    else:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
        nt_outs = _hf_generate_subset(
            teacher_hf, tok_t, math_test_nt_pids + math_train_nt_pids, test_seeds + train_seeds, max_new_tokens=args.max_new_tokens_nothink
        )
        th_outs = _hf_generate_subset(
            teacher_hf, tok_t, math_test_th_pids + math_train_th_pids, test_seeds + train_seeds, max_new_tokens=args.max_new_tokens_think
        )

    n_test = len(math_test)
    t_test_nt_raw, t_train_nt_raw = nt_outs[:n_test], nt_outs[n_test:]
    t_test_th_raw, t_train_th_raw = th_outs[:n_test], th_outs[n_test:]

    t_test_nt_rows = _grade_math500_rows(math_test, t_test_nt_raw, tok_s, eos_set, args.max_new_tokens_nothink)
    t_test_th_rows = _grade_math500_rows(math_test, t_test_th_raw, tok_s, eos_set, args.max_new_tokens_think)
    t_nt_sum = _summarize_math500_arm(t_test_nt_rows)
    t_th_sum = _summarize_math500_arm(t_test_th_rows)
    t_th_diag = _compute_think_diagnostics(t_test_th_raw, tok_s, eos_set, think_end_id)
    print(
        f"[think-cmp] teacher_4b_nothink={t_nt_sum['accuracy']*100:.1f}% (tok={t_nt_sum['mean_tokens']}) | "
        f"teacher_4b_think={t_th_sum['accuracy']*100:.1f}% (tok={t_th_sum['mean_tokens']}, closed={t_th_diag['closed_think_rate']*100:.1f}%)",
        flush=True,
    )

    # Filter valid train traces for nothink vs think (post-</think> clean solutions)
    nothink_train_specs: list[dict[str, Any]] = []
    think_train_specs: list[dict[str, Any]] = []
    n_th_closed, n_th_correct, n_th_kept, n_th_too_long = 0, 0, 0, 0
    think_only_train_solved = 0

    for i, it in enumerate(math_train):
        # 1. No-think train trace
        raw_nt = t_train_nt_raw[i]
        g_nt = dm.grade_math(raw_nt["token_ids"], raw_nt["finish_reason"], str(it["answer"]), tok_s, eos_set, args.max_new_tokens_nothink)
        nt_ok = bool(g_nt["correct"] and (not g_nt["looped_killed"]) and int(g_nt["n_tokens"]) <= int(args.max_teacher_train_tokens))
        if nt_ok:
            txt_nt = tok_s.decode(lr.strip_eos(raw_nt["token_ids"], eos_set), skip_special_tokens=True).strip()
            if txt_nt:
                content = it["problem"].strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
                p_nt = format_math500_prompt_ids(tok_s, it["problem"])
                full_nt = _format_messages(
                    tok_s,
                    [{"role": "user", "content": content}, {"role": "assistant", "content": txt_nt}],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                nothink_train_specs.append({"id": it["id"], "domain": it["domain"], "ids": full_nt, "lo": max(0, len(p_nt) - 1), "hi": len(full_nt) - 1})

        # 2. Think train trace -> extract clean post-</think> solution & align privileged teacher sequence
        raw_th = t_train_th_raw[i]
        g_th = dm.grade_math(raw_th["token_ids"], raw_th["finish_reason"], str(it["answer"]), tok_s, eos_set, args.max_new_tokens_think)
        sp_th = _split_think_output(raw_th["token_ids"], tok_s, eos_set, think_end_id)
        if sp_th["closed_think"]:
            n_th_closed += 1
        if g_th["correct"]:
            n_th_correct += 1
        if g_th["correct"] and (not g_nt["correct"]):
            think_only_train_solved += 1
        if (
            g_th["correct"]
            and (not g_th["looped_killed"])
            and sp_th["closed_think"]
            and len(sp_th["post_think_text"]) > 0
        ):
            aligned = _build_aligned_privileged_pair(tok_s, it["problem"], sp_th["post_think_text"], sp_th["think_prefix_ids"])
            if (
                aligned["n_target_tokens"] <= int(args.max_teacher_train_tokens)
                and len(aligned["think_ids"]) <= int(args.max_think_context_tokens)
            ):
                n_th_kept += 1
                think_train_specs.append({"id": it["id"], "domain": it["domain"], **aligned})
            else:
                n_th_too_long += 1

    teacher_train_coverage = {
        "n_math_train": len(math_train),
        "nothink_kept": len(nothink_train_specs),
        "think_closed": n_th_closed,
        "think_correct": n_th_correct,
        "think_kept": n_th_kept,
        "think_dropped_too_long": n_th_too_long,
        "think_only_solved_on_train": think_only_train_solved,
    }
    print(f"[think-cmp] Teacher train coverage: {teacher_train_coverage}", flush=True)
    assert len(nothink_train_specs) >= 16 and len(think_train_specs) >= 16, teacher_train_coverage

    # ---- Phase 1B: Extract Top-32 4B Logits (No-Think Baseline, Post-Think No-Think KL, & Privileged Think KL) ----
    if teacher_hf is None:
        try:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, trust_remote_code=True).to(dev).eval()
        except TypeError:
            teacher_hf = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in teacher_hf.parameters():
        p.requires_grad_(False)

    print(
        f"[think-cmp] Phase 1B: Extracting Top-32 4B logits — {len(nothink_train_specs)} nothink + "
        f"{len(think_train_specs)}x2 post-think (nothink KL & privileged <think> KL)...",
        flush=True,
    )

    def _extract_topk(full_ids: list[int], lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids_dev = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)
        t_logits = _tail_logits(teacher_hf, ids_dev, lo)[: hi - lo].float()
        vals, idx = t_logits.topk(32, dim=-1)
        out = (idx.to(torch.int32).cpu(), vals.to(torch.float16).cpu())
        del ids_dev, t_logits
        return out

    nothink_distill_samples: list[dict[str, Any]] = []
    think_post_ntkl_samples: list[dict[str, Any]] = []
    think_priv_kl_samples: list[dict[str, Any]] = []

    with torch.no_grad():
        for idx_i, sp in enumerate(nothink_train_specs):
            tk_i, tk_v = _extract_topk(sp["ids"], sp["lo"], sp["hi"])
            nothink_distill_samples.append(
                {"ids": sp["ids"], "lo": sp["lo"], "hi": sp["hi"], "topk_ids": tk_i, "topk_logits": tk_v}
            )
            if dev.type == "cuda" and (idx_i + 1) % 60 == 0:
                torch.cuda.empty_cache()

        for idx_i, sp in enumerate(think_train_specs):
            tk_nt_i, tk_nt_v = _extract_topk(sp["nothink_ids"], sp["lo_s"], sp["hi_s"])
            tk_pr_i, tk_pr_v = _extract_topk(sp["think_ids"], sp["lo_t"], sp["hi_t"])
            assert tk_nt_i.shape == tk_pr_i.shape, (tk_nt_i.shape, tk_pr_i.shape)
            think_post_ntkl_samples.append(
                {"ids": sp["nothink_ids"], "lo": sp["lo_s"], "hi": sp["hi_s"], "topk_ids": tk_nt_i, "topk_logits": tk_nt_v}
            )
            think_priv_kl_samples.append(
                {"ids": sp["nothink_ids"], "lo": sp["lo_s"], "hi": sp["hi_s"], "topk_ids": tk_pr_i, "topk_logits": tk_pr_v}
            )
            if dev.type == "cuda" and (idx_i + 1) % 40 == 0:
                torch.cuda.empty_cache()

    del teacher_hf
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Load Student 0.8B, Layer-12 Router, Base MMLU (400), Prose (24) & Disjoint Anchor (16) ----
    print(f"[think-cmp] Phase 2: Loading Student ({args.student}), Layer-12 router, base MMLU & disjoint anchor...", flush=True)
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
    router_layer = n_layers // 2

    mmlu_r_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_router_train]
    mmlu_test_pids = [format_mmlu_prompt_ids(tok_s, it) for it in mmlu_test]
    prose_r_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_ROUTER_TRAIN]
    prose_held_pids = [_format_messages(tok_s, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_HELDOUT]

    r_train_pids = math_train_nt_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = [it["domain"] for it in math_train] + ["passthrough"] * (len(mmlu_r_pids) + len(prose_r_pids))
    W_r, b_r = fit_ridge_router(_extract_prompt_hidden_vecs(student_m, r_train_pids, router_layer), r_train_labels, ROUTER_CLASSES, lam=1.0)
    routed_math_test = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, math_test_nt_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_mmlu_test = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, mmlu_test_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_prose_test = predict_ridge_router(_extract_prompt_hidden_vecs(student_m, prose_held_pids, router_layer), W_r, b_r, ROUTER_CLASSES)

    router_report = {
        "router_layer": router_layer,
        "math_exact_domain_acc": round(sum(p == it["domain"] for p, it in zip(routed_math_test, math_test)) / len(math_test), 4),
        "math_any_domain_fire_rate": round(sum(p in MATH_DOMAINS for p in routed_math_test) / len(math_test), 4),
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_fire_rate": round(sum(p != "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_non_math_fire_rate": round(
            sum(p != "passthrough" for p, it in zip(routed_mmlu_test, mmlu_test) if not it["is_math_subject"])
            / max(1, sum(1 for it in mmlu_test if not it["is_math_subject"])),
            4,
        ),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_test) / len(routed_prose_test), 4),
    }
    print(f"[think-cmp] Router report: {router_report}", flush=True)

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

    prose_full_ids, base_prose_logp, base_prose_nll = _compute_base_logp(PROSE_HELDOUT)
    anchor_full_ids, base_anchor_logp, _ = _compute_base_logp(PROSE_ROUTER_TRAIN)
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Train 3 Matched 24L r=16 Cartridges (160 steps x 4 = 640 passes, identical init) ----
    print("[think-cmp] Phase 3: Training 3 matched r=16 24L cartridges (640 passes each)...", flush=True)
    t_p3 = time.perf_counter()

    cartridge_specs = [
        ("nothink_distill_r16_24L", nothink_distill_samples),
        ("think_post_nothink_kl_r16_24L", think_post_ntkl_samples),
        ("think_privileged_kl_r16_24L", think_priv_kl_samples),
    ]

    with tempfile.TemporaryDirectory() as exp_dir_str:
        exp_dir = Path(exp_dir_str)
        ckpt_paths: dict[str, Path] = {}
        expert_meta: dict[str, Any] = {}
        mmlu_raw_by_arm: dict[str, dict[str, Any]] = {}

        for arm_base, samples in cartridge_specs:
            torch.manual_seed(int(args.seed))
            res_c = _train_regularized_lora_expert(
                student_m,
                all_layers,
                samples,
                rank=int(args.rank),
                alpha_scale=float(args.rank * 2),
                total_steps=int(args.total_steps),
                grad_accum=int(args.grad_accum),
                lr_val=2.0e-4,
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
                seed=int(args.seed),
            )
            upd = res_c.pop("weights")
            with _temp_apply_weights(student_m, upd):
                if arm_base == "think_privileged_kl_r16_24L":
                    # Full MMLU score on privileged arm to report both routed & unrouted protection
                    mmlu_raw_by_arm[arm_base] = _score_mmlu_fast(student_m, mmlu_test_pids, mmlu_test, letter_ids, dev)
                else:
                    # Sparse MMLU on fired items (router fires on math-MMLU only)
                    fired_mmlu_cls = ["math" if c != "passthrough" else "passthrough" for c in routed_mmlu_test]
                    mmlu_raw_by_arm[arm_base] = _score_routed_mmlu_sparse(
                        student_m, mmlu_test_pids, mmlu_test, fired_mmlu_cls, "math", base_mmlu_raw, letter_ids, dev
                    )
            p_ckpt = exp_dir / f"{arm_base}.pt"
            torch.save(upd, p_ckpt)
            del upd
            ckpt_paths[arm_base] = p_ckpt
            expert_meta[arm_base] = res_c
            print(
                f"[think-cmp] {arm_base}: n={res_c['n_train_samples']} ({res_c['adapter_mb']}MB, {res_c['compile_time_s']:.1f}s) | "
                f"loss={res_c['step_losses_checkpoints'][0]}->{res_c['step_losses_checkpoints'][-1]} | pkl={res_c['healthy_kl']}",
                flush=True,
            )

        mmlu_arms: dict[str, Any] = {"base_08b_nothink": base_mmlu_summary}
        for idx_c, (arm_base, _) in enumerate(cartridge_specs):
            mmlu_arms[f"{arm_base}_routed"] = _compose_routed_mmlu(
                mmlu_test, routed_mmlu_test, base_mmlu_raw, {d: mmlu_raw_by_arm[arm_base] for d in MATH_DOMAINS}, seed=int(args.seed) + 1 + idx_c
            )
        mmlu_arms["think_privileged_kl_r16_24L_unrouted"] = _summarize_mmlu(
            mmlu_test,
            mmlu_raw_by_arm["think_privileged_kl_r16_24L"],
            base_score=base_mmlu_raw,
            fire_flags=[True] * len(mmlu_test),
            seed=int(args.seed) + 10,
        )
        phase3_s = round(time.perf_counter() - t_p3, 2)

        # ---- Phase 4: Evaluate All 7 Paired Arms on MATH-500 (n=200) ----
        print(f"[think-cmp] Phase 4: Evaluating student arms on MATH-500 (n={len(math_test)}, {backend})...", flush=True)
        t_p4 = time.perf_counter()

        if backend == "vllm":
            del student_m
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            base_nt_raw = _vllm_generate(
                args.student, tok_s, math_test_nt_pids, test_seeds, max_new_tokens=args.max_new_tokens_nothink, dtype_str=dtype_vllm_str, base_seed=args.seed
            )
            base_th_raw = _vllm_generate(
                args.student, tok_s, math_test_th_pids, test_seeds, max_new_tokens=args.max_new_tokens_think, dtype_str=dtype_vllm_str, base_seed=args.seed + 7
            )
        else:
            base_nt_raw = _hf_generate_subset(student_m, tok_s, math_test_nt_pids, test_seeds, max_new_tokens=args.max_new_tokens_nothink)
            base_th_raw = _hf_generate_subset(student_m, tok_s, math_test_th_pids, test_seeds, max_new_tokens=args.max_new_tokens_think)

        bake_tmp_dir = Path("/content/_pocket_moe_think_cmp_bake_tmp") if backend == "vllm" else (exp_dir / "bake_tmp")

        def _eval_ckpt(weights_path: Path, p_ids: list[list[int]], sd_list: list[int]) -> list[dict[str, Any]]:
            upd = torch.load(weights_path, map_location="cpu", weights_only=True)
            if backend == "vllm":
                _bake_full_checkpoint(Path(args.student), upd, bake_tmp_dir)
                del upd
                gc.collect()
                return _vllm_generate(
                    str(bake_tmp_dir),
                    tok_s,
                    p_ids,
                    sd_list,
                    max_new_tokens=args.max_new_tokens_nothink,
                    dtype_str=dtype_vllm_str,
                    base_seed=args.seed,
                )
            return _hf_generate_subset(student_m, tok_s, p_ids, sd_list, max_new_tokens=args.max_new_tokens_nothink, param_updates=upd)

        fired_math_idxs = [i for i, cls_nm in enumerate(routed_math_test) if cls_nm != "passthrough"]
        routed_raw_by_arm: dict[str, list[dict[str, Any]]] = {}
        try:
            for arm_base, _ in cartridge_specs:
                p_ckpt = ckpt_paths[arm_base]
                arm_out = list(base_nt_raw)
                if fired_math_idxs:
                    sub_outs = _eval_ckpt(p_ckpt, [math_test_nt_pids[i] for i in fired_math_idxs], [test_seeds[i] for i in fired_math_idxs])
                    for idx_i, o in zip(fired_math_idxs, sub_outs):
                        arm_out[idx_i] = o
                p_ckpt.unlink(missing_ok=True)
                routed_raw_by_arm[arm_base] = arm_out
        finally:
            if bake_tmp_dir.exists():
                shutil.rmtree(bake_tmp_dir, ignore_errors=True)
        phase4_s = round(time.perf_counter() - t_p4, 2)

    # ---- Phase 5: Grading, Thinking-Only Hard Subset, Token Compression & Paired Bootstrap CIs ----
    base_th_diag = _compute_think_diagnostics(base_th_raw, tok_s, eos_set, think_end_id)
    arm_rows: dict[str, list[dict[str, Any]]] = {
        "base_08b_nothink": _grade_math500_rows(math_test, base_nt_raw, tok_s, eos_set, args.max_new_tokens_nothink),
        "base_08b_think": _grade_math500_rows(math_test, base_th_raw, tok_s, eos_set, args.max_new_tokens_think),
        "teacher_4b_nothink": t_test_nt_rows,
        "teacher_4b_think": t_test_th_rows,
    }
    for arm_base, _ in cartridge_specs:
        arm_rows[f"{arm_base}_routed"] = _grade_math500_rows(
            math_test, routed_raw_by_arm[arm_base], tok_s, eos_set, args.max_new_tokens_nothink, routed_labels=routed_math_test
        )

    think_only_test_idxs = [
        i for i in range(len(math_test)) if t_test_th_rows[i]["strict_correct"] and (not t_test_nt_rows[i]["strict_correct"])
    ]
    base_nt_acc = _summarize_math500_arm(arm_rows["base_08b_nothink"])["accuracy"]
    base_th_mean_tok = max(1.0, _summarize_math500_arm(arm_rows["base_08b_think"])["mean_tokens"])
    t_th_mean_tok = max(1.0, t_th_sum["mean_tokens"])
    t_th_flags = [bool(r["strict_correct"]) for r in t_test_th_rows]

    math500_arms: dict[str, dict[str, Any]] = {}
    for arm_nm, rows in arm_rows.items():
        sm = _summarize_math500_arm(rows, teacher_correct_flags=t_th_flags, base_acc=base_nt_acc, teacher_acc=t_th_sum["accuracy"])
        sm["think_only_hard_subset_n"] = len(think_only_test_idxs)
        sm["think_only_hard_subset_acc"] = (
            round(sum(1 for i in think_only_test_idxs if rows[i]["strict_correct"]) / len(think_only_test_idxs), 4)
            if think_only_test_idxs
            else None
        )
        sm["token_compression_vs_base_think"] = round(base_th_mean_tok / max(1.0, float(sm["mean_tokens"])), 3)
        sm["token_compression_vs_teacher_think"] = round(t_th_mean_tok / max(1.0, float(sm["mean_tokens"])), 3)
        if arm_nm == "base_08b_think":
            sm["think_diagnostics"] = base_th_diag
        elif arm_nm == "teacher_4b_think":
            sm["think_diagnostics"] = t_th_diag
        math500_arms[arm_nm] = sm

    prose_fire_frac = sum(1 for c in routed_prose_test if c != "passthrough") / max(1, len(routed_prose_test))
    prose_kl: dict[str, float] = {
        f"{arm_base}_routed": round(float(expert_meta[arm_base]["healthy_kl"]) * prose_fire_frac, 6)
        for arm_base, _ in cartridge_specs
    }
    prose_kl["think_privileged_kl_r16_24L_unrouted"] = float(expert_meta["think_privileged_kl_r16_24L"]["healthy_kl"])

    compile_time_s = {f"{arm_base}_s": float(expert_meta[arm_base]["compile_time_s"]) for arm_base, _ in cartridge_specs}

    comp_pairs = [
        ("teacher_think_vs_teacher_nothink", "teacher_4b_think", "teacher_4b_nothink"),
        ("base_think_vs_base_nothink", "base_08b_think", "base_08b_nothink"),
        ("nothink_distill_vs_base_nothink", "nothink_distill_r16_24L_routed", "base_08b_nothink"),
        ("think_post_ntkl_vs_nothink_distill", "think_post_nothink_kl_r16_24L_routed", "nothink_distill_r16_24L_routed"),
        ("think_privileged_kl_vs_think_post_ntkl", "think_privileged_kl_r16_24L_routed", "think_post_nothink_kl_r16_24L_routed"),
        ("think_privileged_kl_vs_nothink_distill", "think_privileged_kl_r16_24L_routed", "nothink_distill_r16_24L_routed"),
        ("think_privileged_kl_vs_base_think", "think_privileged_kl_r16_24L_routed", "base_08b_think"),
        ("think_post_ntkl_vs_base_think", "think_post_nothink_kl_r16_24L_routed", "base_08b_think"),
        ("think_privileged_kl_vs_base_nothink", "think_privileged_kl_r16_24L_routed", "base_08b_nothink"),
    ]
    paired_math500_200 = {
        lbl: _paired_bootstrap(arm_rows[a], arm_rows[b], seed=int(args.seed) + i)
        for i, (lbl, a, b) in enumerate(comp_pairs)
    }

    acc = lambda k: float(math500_arms[k]["accuracy"])  # noqa: E731
    nm_mmlu = lambda k: float(mmlu_arms[k]["non_math"]["accuracy"])  # noqa: E731
    routed_names = [f"{arm_base}_routed" for arm_base, _ in cartridge_specs]

    return {
        "teacher_model": args.teacher,
        "student_model": args.student,
        "dtype": str(dtype),
        "backend": backend,
        "n_math_train": len(math_train),
        "n_math_test": len(math_test),
        "rank": int(args.rank),
        "total_steps": int(args.total_steps),
        "grad_accum": int(args.grad_accum),
        "max_new_tokens_nothink": int(args.max_new_tokens_nothink),
        "max_new_tokens_think": int(args.max_new_tokens_think),
        "max_teacher_train_tokens": int(args.max_teacher_train_tokens),
        "max_think_context_tokens": int(args.max_think_context_tokens),
        "timing_s": {
            "phase1_teacher_s": phase1_s,
            "phase2_router_base_s": phase2_s,
            "phase3_train_s": phase3_s,
            "phase4_eval_s": phase4_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "teacher_train_coverage": teacher_train_coverage,
        "compile_time_s": compile_time_s,
        "router_report": router_report,
        "expert_meta": expert_meta,
        "prose_kl": prose_kl,
        "math500_arms": math500_arms,
        "mmlu_arms": mmlu_arms,
        "paired_math500_200": paired_math500_200,
        "criteria": {
            "a_teacher_think_beats_teacher_nothink": bool(acc("teacher_4b_think") > acc("teacher_4b_nothink")),
            "b_think_post_beats_nothink_distill": bool(acc("think_post_nothink_kl_r16_24L_routed") > acc("nothink_distill_r16_24L_routed")),
            "c_privileged_kl_beats_post_nothink_kl": bool(
                acc("think_privileged_kl_r16_24L_routed") > acc("think_post_nothink_kl_r16_24L_routed")
            ),
            "d_privileged_kl_matches_or_beats_base_think": bool(acc("think_privileged_kl_r16_24L_routed") >= acc("base_08b_think")),
            "e_token_compression_gt_2x_vs_base_think": bool(
                math500_arms["think_privileged_kl_r16_24L_routed"]["token_compression_vs_base_think"] >= 2.0
            ),
            "f_zero_general_forgetting_all_routed": bool(
                all(prose_kl[nm] <= 0.005 and nm_mmlu(nm) >= nm_mmlu("base_08b_nothink") - 0.015 for nm in routed_names)
            ),
        },
        "per_item_math500": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU verification")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-math-train", type=int, default=300)
    ap.add_argument("--n-math-test", type=int, default=200)
    ap.add_argument("--n-mmlu-test", type=int, default=400)
    ap.add_argument("--n-mmlu-router-train", type=int, default=200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--total-steps", type=int, default=160)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-new-tokens-nothink", type=int, default=2048)
    ap.add_argument("--max-new-tokens-think", type=int, default=4096)
    ap.add_argument("--max-teacher-train-tokens", type=int, default=1350)
    ap.add_argument("--max-think-context-tokens", type=int, default=4608)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
        _print_think_compress_summary(payload)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
