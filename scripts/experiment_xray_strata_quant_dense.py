"""Exp 2A: X-Ray-Guided Mixed-Precision Strata Quantization (2.65-3.19 b/w + H64 + GPTQ + 64-flip + Scale-QAT) on Qwen2.5-3B.

Compares 6 paired arms on `Qwen/Qwen2.5-3B-Instruct` across WikiText-2 PPL, Top-256 KL & Top-1 vs BF16,
MMLU-200, and GSM8K-100:
  1. `base_bf16`                   : Reference unquantized BF16 model
  2. `uniform_rtn_3b`              : Uniform naive unclipped RTN (`S3_K` size = 3.1875 b/w, no H64/GPTQ)
  3. `uniform_s3k_h64_gptq_flip`   : Uniform `S3_K` + `H64` + Block-LDL + 64-flip (3.1875 b/w)
  4. `xray_knapsack_3_19b`         : Budget <= `uniform_s3k` via global Hessian knapsack weighted by WSI X-Ray
  5. `xray_knapsack_3_19b_scale_qat`: Arm 4 + 2-epoch Top-256 KL Scale-QAT over 1D RMSNorms (0 extra bytes)
  6. `xray_knapsack_2_65b_scale_qat`: Ultra-compact 2.65 b/w X-Ray knapsack + Scale-QAT

CPU smoke test::
    python scripts/experiment_xray_strata_quant_dense.py --smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
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
from experiment_pocket_moe_math500 import format_mmlu_prompt_ids, load_mmlu_split  # noqa: E402
from experiment_reasoning_distill_pilot import _atomic_write_json  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.strata_quant import (  # noqa: E402
    build_quant_sensitivity_map,
    capture_teacher_topk_targets,
    damped_cholesky_inv,
    eval_topk_kl_and_top1,
    gptq_flip_quantize_sk_torch,
    matrix_wire_bytes,
    apply_h64_sym,
    rtn_quantize_sk_torch,
    run_norm_scale_qat,
    solve_knapsack_bit_allocation,
)

DEFAULT_XRAY_PATH = ROOT / "benchmarks" / "xray_atlas" / "tome1_xray" / "qwen__qwen2.5-3b-instruct.xray.json"

_PROSE_CAL_SAMPLES = [
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "Let S = sum_{k=1}^{n} k^2. By induction, S = n(n + 1)(2n + 1) / 6. For n = 10, S = 385.",
    "HTTP 206 Partial Content enables byte-range streaming of large binary checkpoints without downloading multi-gigabyte payloads.",
    "In Mixture-of-Experts transformers, router logits remain in FP32 while routed MLP weights are compressed via Hessian-weighted quantization.",
]

_PROJ_SPECS: list[tuple[str, str]] = [
    ("self_attn.q_proj", "qkv"),
    ("self_attn.k_proj", "qkv"),
    ("self_attn.v_proj", "qkv"),
    ("self_attn.o_proj", "o"),
    ("mlp.gate_proj", "gu"),
    ("mlp.up_proj", "gu"),
    ("mlp.down_proj", "dn"),
]


def _get_submod(root: nn.Module, path: str) -> nn.Linear:
    cur: Any = root
    for pt in path.split("."):
        cur = getattr(cur, pt)
    assert isinstance(cur, nn.Linear), (path, type(cur))
    return cur


def _load_wikitext_chunks(tok: Any, *, seq_len: int = 512, n_train: int = 48, n_dev: int = 8, n_test_ppl: int = 32, n_test_kl: int = 16) -> dict[str, list[list[int]]]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    def _chunk_split(split_nm: str, need: int) -> list[list[int]]:
        p = hf_hub_download(
            "Salesforce/wikitext",
            f"wikitext-2-raw-v1/{split_nm}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        texts = pq.read_table(p, columns=["text"]).column("text").to_pylist()
        text = "\n\n".join(str(t) for t in texts if t and str(t).strip())
        ids = tok.encode(text, add_special_tokens=False)
        chunks = [ids[i : i + seq_len] for i in range(0, len(ids) - seq_len, seq_len)]
        return chunks[:need]

    tr = _chunk_split("train", n_train)
    # Mix short prose/code/math sequences padded/repeated cleanly to >=128 tok for multi-domain Hessian calibration
    for i in range(16):
        s_txt = _PROSE_CAL_SAMPLES[i % len(_PROSE_CAL_SAMPLES)]
        toks = tok.encode((s_txt + "\n") * 6, add_special_tokens=False)[:seq_len]
        if len(toks) >= 64:
            tr.append(toks)
    dv = _chunk_split("validation", n_dev)
    te_all = _chunk_split("test", n_test_ppl + n_test_kl)
    return {
        "train": tr,
        "dev": dv,
        "test_ppl": te_all[:n_test_ppl],
        "test_kl": te_all[n_test_ppl : n_test_ppl + n_test_kl],
    }


@torch.no_grad()
def _eval_perplexity(model: nn.Module, seqs: list[list[int]], dev: torch.device) -> float:
    nll_sum = 0.0
    tok_cnt = 0
    for s in seqs:
        ids = torch.tensor(s, dtype=torch.long, device=dev).unsqueeze(0)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0, :-1].float()
        targets = ids[0, 1:]
        nll_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        tok_cnt += int(targets.numel())
        del ids, out, logits
    return round(float(math.exp(min(20.0, nll_sum / max(1, tok_cnt)))), 4)


@torch.no_grad()
def _eval_abcd_mc(
    model: nn.Module,
    prompts_ids: list[list[int]],
    gold_idxs: list[int],
    is_math_flags: list[bool] | None,
    letter_ids: dict[str, list[int]],
    pad_id: int,
    dev: torch.device,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Fast batched single-token A/B/C/D multiple-choice accuracy & ABCD probability mass."""
    letters = ("A", "B", "C", "D")
    letter_tensors = [
        torch.tensor(letter_ids.get(ch) or [], dtype=torch.long, device=dev) for ch in letters
    ]
    correct = 0
    nonmath_correct = 0
    nonmath_total = 0
    abcd_mass_sum = 0.0
    n = len(prompts_ids)
    bs = max(1, int(batch_size))
    for b0 in range(0, n, bs):
        sub_p = prompts_ids[b0 : b0 + bs]
        max_l = max(len(p) for p in sub_p)
        inp = torch.full((len(sub_p), max_l), pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((len(sub_p), max_l), dtype=torch.long, device=dev)
        for ri, p in enumerate(sub_p):
            inp[ri, max_l - len(p) :] = torch.tensor(p, dtype=torch.long, device=dev)
            mask[ri, max_l - len(p) :] = 1
        out = model(input_ids=inp, attention_mask=mask, use_cache=False)
        last_lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        cols = []
        for t_ids in letter_tensors:
            if t_ids.numel() > 0:
                cols.append(torch.logsumexp(last_lp[:, t_ids], dim=-1))
            else:
                cols.append(torch.full((len(sub_p),), -1e9, device=dev, dtype=torch.float32))
        abcd_lp = torch.stack(cols, dim=-1)  # [B, 4]
        abcd_probs = torch.exp(abcd_lp)
        abcd_mass_sum += float(abcd_probs.sum(dim=-1).sum().item())
        preds = abcd_lp.argmax(dim=-1).tolist()
        for offset, pr in enumerate(preds):
            gi = b0 + offset
            ok = int(pr) == int(gold_idxs[gi])
            correct += int(ok)
            if is_math_flags is not None and not is_math_flags[gi]:
                nonmath_correct += int(ok)
                nonmath_total += 1
        del inp, mask, out, last_lp, abcd_lp
    res = {
        "accuracy": round(correct / max(1, n), 4),
        "n": n,
        "abcd_mass": round(abcd_mass_sum / max(1, n), 4),
    }
    if is_math_flags is not None and nonmath_total > 0:
        res["non_math_accuracy"] = round(nonmath_correct / nonmath_total, 4)
        res["non_math_n"] = nonmath_total
    return res


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_gsm8k_num(txt: str) -> str | None:
    m_b = list(_BOXED_RE.finditer(txt))
    cand = m_b[-1].group(1) if m_b else (txt.split("####")[-1] if "####" in txt else txt)
    nums = _NUM_RE.findall(cand)
    if not nums:
        nums = _NUM_RE.findall(txt)
    if not nums:
        return None
    s = nums[-1].replace(",", "").strip()
    try:
        v = float(s)
        return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.6g}"
    except ValueError:
        return None


@torch.no_grad()
def _eval_gsm8k(
    model: nn.Module,
    tok: Any,
    items: list[dict[str, Any]],
    dev: torch.device,
    *,
    max_new_tokens: int = 512,
    batch_size: int = 32,
) -> dict[str, Any]:
    eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0))
    prompts_ids = []
    for it in items:
        msg = [{"role": "user", "content": f"Solve step by step and put the final integer answer in \\boxed{{}}.\n\n{it['question']}"}]
        if getattr(tok, "chat_template", None):
            try:
                txt_p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                ids = tok.encode(txt_p, add_special_tokens=False)
            except Exception:
                ids = tok.encode(msg[0]["content"], add_special_tokens=True)
        else:
            ids = tok.encode(msg[0]["content"], add_special_tokens=True)
        prompts_ids.append([int(x) for x in ids])

    correct = 0
    bs = max(1, int(batch_size))
    for b0 in range(0, len(items), bs):
        sub_p = prompts_ids[b0 : b0 + bs]
        max_l = max(len(p) for p in sub_p)
        inp = torch.full((len(sub_p), max_l), pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((len(sub_p), max_l), dtype=torch.long, device=dev)
        for ri, p in enumerate(sub_p):
            inp[ri, max_l - len(p) :] = torch.tensor(p, dtype=torch.long, device=dev)
            mask[ri, max_l - len(p) :] = 1
        gen = model.generate(
            input_ids=inp,
            attention_mask=mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_ids,
        )
        for ri in range(len(sub_p)):
            txt = tok.decode(gen[ri, max_l:].tolist(), skip_special_tokens=True)
            pred = _extract_gsm8k_num(txt)
            gold = _extract_gsm8k_num(items[b0 + ri]["answer"].split("####")[-1])
            if pred is not None and gold is not None and pred == gold:
                correct += 1
        del inp, mask, gen
    return {"accuracy": round(correct / max(1, len(items)), 4), "n_correct": correct, "n": len(items)}


def _apply_quant_arm(
    blocks: Sequence[nn.Module],
    qcache_dir: Path,
    plan: dict[tuple[int, str], str],
    dev: torch.device,
    orig_dtype: torch.dtype,
) -> None:
    """Load cached quantized weight matrices `[layer, proj] -> code_label` onto `blocks` in-place."""
    for (li, proj), label in plan.items():
        fpath = qcache_dir / f"L{li:02d}_{proj.replace('.', '_')}_{label}.pt"
        t_cpu = torch.load(fpath, map_location="cpu", weights_only=True)
        mod = _get_submod(blocks[li], proj)
        mod.weight.data.copy_(t_cpu.to(device=dev, dtype=orig_dtype))
        del t_cpu


def _restore_norms_and_weights(
    model: nn.Module,
    orig_norms: dict[str, torch.Tensor],
    blocks: Sequence[nn.Module],
    orig_weights_dir: Path,
    dev: torch.device,
    orig_dtype: torch.dtype,
) -> None:
    for nm, mod in model.named_modules():
        if nm in orig_norms:
            mod.weight.data.copy_(orig_norms[nm].to(device=dev, dtype=mod.weight.dtype))
    for li, blk in enumerate(blocks):
        for proj, _ in _PROJ_SPECS:
            fpath = orig_weights_dir / f"L{li:02d}_{proj.replace('.', '_')}_ORIG.pt"
            t_cpu = torch.load(fpath, map_location="cpu", weights_only=True)
            _get_submod(blk, proj).weight.data.copy_(t_cpu.to(device=dev, dtype=orig_dtype))
            del t_cpu


def _print_summary(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return
    print("\n=== EXP 2A: X-RAY + STRATA QUANTIZATION (Qwen2.5-3B-Instruct) ===", flush=True)
    for arm_nm, st in payload.get("arms", {}).items():
        print(
            f"  {arm_nm:30s} | bpw={st.get('avg_bits_per_weight', 16.0):6.3f} | "
            f"PPL={st['wikitext2_ppl']:7.3f} | KL={st['kl_mean']:7.5f} | Top1={st['top1_agreement']*100:5.2f}% | "
            f"MMLU={st['mmlu_200']['accuracy']*100:5.1f}% | GSM8K={st['gsm8k_100']['accuracy']*100:5.1f}%",
            flush=True,
        )
    print("\nCriteria:", payload.get("criteria"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "matrix_allocations"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification on a tiny 4-layer Qwen2 model."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    fake_card = json.loads(DEFAULT_XRAY_PATH.read_text(encoding="utf-8"))
    fake_card["identity"]["n_layers"] = 4
    fake_card["anatomy"]["forward_profile"]["massive_act_max"]["layer"] = 1
    fake_card["anatomy"]["fact_readout"]["dla_peak"]["layer"] = 2
    qmap = build_quant_sensitivity_map(fake_card)
    prot = set(qmap["protected_layers"])

    seqs = [[2, 5, 8, 12, 19, 25, 31, 40], [3, 7, 11, 15, 22, 28, 35, 44]]
    tr_recs = capture_teacher_topk_targets(model, seqs[:1], "cpu", topk=16)
    dv_recs = capture_teacher_topk_targets(model, seqs[1:], "cpu", topk=16)

    _, blocks = find_blocks(model)
    items = []
    rtn_err = 0.0
    s3_err = 0.0
    s3_bytes = 0
    for li, blk in enumerate(blocks):
        w = blk.mlp.down_proj.weight.data.float()
        x = torch.randn((32, w.shape[1]))
        h = (x.T @ x) / 32.0 + 0.01 * torch.eye(w.shape[1])
        _, wb_r, eh_r = rtn_quantize_sk_torch(w, "S3_K", use_clipping=False, use_h64=False, H=h)
        rtn_err += float(eh_r or 0.0)
        opts = []
        codes = ("S3_K", "S4_K") if li in prot else ("S2_K", "S3_K", "S4_K")
        h_rot = apply_h64_sym(h) if w.shape[1] % 64 == 0 else h
        pre = None
        for cd in codes:
            _, wb, eh = gptq_flip_quantize_sk_torch(w, h, cd, use_h64=(w.shape[1] % 64 == 0), max_flips=16, precomputed_H=pre)
            if cd == "S3_K":
                s3_err += eh
                s3_bytes += wb
            opts.append((cd, wb, eh * (1.0 + 2.0 * qmap["per_layer_scores"][li])))
        items.append({"id": (li, "mlp.down_proj"), "options": opts})

    sel, used_b, _ = solve_knapsack_bit_allocation(items, s3_bytes)
    assert used_b <= s3_bytes
    assert s3_err < rtn_err
    assert all(sel[(li, "mlp.down_proj")] in ("S3_K", "S4_K") for li in prot)

    # Apply quantized weights and check Scale-QAT reduces KL
    with torch.no_grad():
        for li, blk in enumerate(blocks):
            w = blk.mlp.down_proj.weight.data.float()
            x = torch.randn((32, w.shape[1]))
            h = (x.T @ x) / 32.0 + 0.01 * torch.eye(w.shape[1])
            wq, _, _ = gptq_flip_quantize_sk_torch(w, h, sel[(li, "mlp.down_proj")], use_h64=True, max_flips=16)
            blk.mlp.down_proj.weight.data.copy_(wq)
    pre_kl = eval_topk_kl_and_top1(model, dv_recs, "cpu")["kl_mean"]
    qat_res = run_norm_scale_qat(model, tr_recs, dv_recs, device="cpu", epochs=4, lr=2e-2, grad_accum=1)
    assert qat_res["dev_kl_best"] <= pre_kl + 1e-6
    return {
        "mode": "smoke",
        "uniform_s3_bytes": s3_bytes,
        "knapsack_bytes": used_b,
        "rtn_err": round(rtn_err, 5),
        "strata_s3_err": round(s3_err, 5),
        "dev_kl_before": qat_res["dev_kl_before"],
        "dev_kl_best": qat_res["dev_kl_best"],
        "passed": True,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    torch.manual_seed(int(args.seed))

    xray_card = json.loads(Path(args.xray).read_text(encoding="utf-8"))
    qmap = build_quant_sensitivity_map(xray_card)
    prot_layers = set(int(x) for x in qmap["protected_layers"])
    layer_scores = list(qmap["per_layer_scores"])

    print(f"[exp2a] Loading {args.model} | X-Ray protected={sorted(prot_layers)} | valley={qmap['compressible_valley']}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    assert len(layer_scores) == n_layers

    wt = _load_wikitext_chunks(tok, seq_len=512, n_train=48, n_dev=8, n_test_ppl=32, n_test_kl=16)
    mmlu_test, _ = load_mmlu_split(n_test=200, n_router_train=20, seed=int(args.seed))
    mmlu_pids = [format_mmlu_prompt_ids(tok, it) for it in mmlu_test]
    mmlu_gold = [int(it["gold_idx"]) for it in mmlu_test]
    mmlu_ismath = [bool(it.get("is_math_subject", False)) for it in mmlu_test]
    letter_ids = ec._resolve_mmlu_letter_ids(tok)
    gsm_ds = list(load_dataset("openai/gsm8k", "main", split="test"))[: int(args.n_gsm8k)]

    # Save baseline norm weights & capture teacher targets + baseline metrics
    orig_norms = {nm: mod.weight.detach().cpu().clone() for nm, mod in model.named_modules() if "norm" in nm.lower() and hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor) and mod.weight.ndim == 1}
    print("[exp2a] Capturing BF16 teacher Top-256 targets + baseline eval...", flush=True)
    tr_recs = capture_teacher_topk_targets(model, wt["train"], dev, topk=256)
    dv_recs = capture_teacher_topk_targets(model, wt["dev"], dev, topk=256)
    te_recs = capture_teacher_topk_targets(model, wt["test_kl"], dev, topk=256)

    base_ppl = _eval_perplexity(model, wt["test_ppl"], dev)
    base_mmlu = _eval_abcd_mc(model, mmlu_pids, mmlu_gold, mmlu_ismath, letter_ids, pad_id, dev)
    base_gsm = _eval_gsm8k(model, tok, gsm_ds, dev, batch_size=32)

    arms_out: dict[str, Any] = {
        "base_bf16": {
            "avg_bits_per_weight": 16.0,
            "wikitext2_ppl": base_ppl,
            "kl_mean": 0.0,
            "top1_agreement": 1.0,
            "mmlu_200": base_mmlu,
            "gsm8k_100": base_gsm,
        }
    }
    print(f"[exp2a] BASE BF16: PPL={base_ppl} MMLU={base_mmlu['accuracy']*100:.1f}% GSM8K={base_gsm['accuracy']*100:.1f}%", flush=True)

    # ---- Phase 1: Chunked Hessian capture (6 layers per forward pass) + S2_K/S3_K/S4_K + RTN quantization ----
    t_p1 = time.perf_counter()
    with tempfile.TemporaryDirectory() as qdir_str:
        qdir = Path(qdir_str)
        matrix_errors: dict[tuple[int, str], dict[str, tuple[int, float]]] = {}
        per_layer_s2_err: list[float] = [0.0] * n_layers
        per_layer_s3_err: list[float] = [0.0] * n_layers
        total_params = 0
        uniform_s3_bytes = 0

        chunk_size = 6
        for c0 in range(0, n_layers, chunk_size):
            c1 = min(n_layers, c0 + chunk_size)
            chunk_hess: dict[tuple[int, str], torch.Tensor] = {}
            chunk_cnt: dict[tuple[int, str], int] = {}
            handles = []

            def _make_hook(li_v: int, stream_nm: str, dim: int):
                def _hook(_mod: nn.Module, inp: tuple[torch.Tensor, ...]) -> None:
                    x = inp[0].detach().reshape(-1, dim).float()
                    h_inc = x.T @ x
                    k = (li_v, stream_nm)
                    if k not in chunk_hess:
                        chunk_hess[k] = h_inc
                        chunk_cnt[k] = int(x.shape[0])
                    else:
                        chunk_hess[k].add_(h_inc)
                        chunk_cnt[k] += int(x.shape[0])
                return _hook

            for li in range(c0, c1):
                blk = blocks[li]
                for mod_p, st_nm in [("self_attn.q_proj", "qkv"), ("self_attn.o_proj", "o"), ("mlp.gate_proj", "gu"), ("mlp.down_proj", "dn")]:
                    m_lin = _get_submod(blk, mod_p)
                    handles.append(m_lin.register_forward_pre_hook(_make_hook(li, st_nm, int(m_lin.in_features))))

            with torch.no_grad():
                for seq in wt["train"]:
                    ids = torch.tensor(seq, dtype=torch.long, device=dev).unsqueeze(0)
                    model(input_ids=ids, use_cache=False)
                    del ids
            for h in handles:
                h.remove()

            # Quantize layers c0..c1 across S2_K, S3_K, S4_K + RTN_3B (reusing H64+Cholesky per stream!)
            for li in range(c0, c1):
                blk = blocks[li]
                pre_by_stream: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                H_by_stream: dict[str, torch.Tensor] = {}
                for st_nm in ("qkv", "o", "gu", "dn"):
                    H_raw = chunk_hess.pop((li, st_nm)) / float(max(1, chunk_cnt[(li, st_nm)]))
                    H_by_stream[st_nm] = H_raw
                    H_rot = apply_h64_sym(H_raw)
                    pre_by_stream[st_nm] = damped_cholesky_inv(H_rot, damp_frac=0.01)
                    del H_rot

                for proj, st_nm in _PROJ_SPECS:
                    w = _get_submod(blk, proj).weight.data
                    total_params += int(w.numel())
                    torch.save(w.detach().cpu().clone(), qdir / f"L{li:02d}_{proj.replace('.', '_')}_ORIG.pt")

                    H_s = H_by_stream[st_nm]
                    pre_s = pre_by_stream[st_nm]
                    matrix_errors[(li, proj)] = {}

                    # Naive RTN 3-bit baseline
                    w_rtn, _, err_rtn = rtn_quantize_sk_torch(w, "S3_K", use_clipping=False, use_h64=False, H=H_s)
                    torch.save(w_rtn.cpu(), qdir / f"L{li:02d}_{proj.replace('.', '_')}_RTN3.pt")
                    del w_rtn

                    for cd in ("S2_K", "S3_K", "S4_K"):
                        wq, wb, eh = gptq_flip_quantize_sk_torch(w, H_s, cd, use_h64=True, max_flips=64, precomputed_H=pre_s)
                        torch.save(wq.cpu(), qdir / f"L{li:02d}_{proj.replace('.', '_')}_{cd}.pt")
                        matrix_errors[(li, proj)][cd] = (wb, float(eh))
                        if cd == "S2_K":
                            per_layer_s2_err[li] += float(eh)
                        elif cd == "S3_K":
                            per_layer_s3_err[li] += float(eh)
                            uniform_s3_bytes += wb
                        del wq

                del pre_by_stream, H_by_stream
                torch.cuda.empty_cache()
            print(f"[exp2a] Captured & quantized layers {c0}..{c1-1} ({time.perf_counter()-t_p1:.1f}s)", flush=True)

        # ---- Build Knapsack Allocations ----
        def _build_knapsack(budget_bytes: int, allow_prot_s2: bool = False) -> tuple[dict[tuple[int, str], str], int]:
            items = []
            for li in range(n_layers):
                w_xray = 1.0 + 2.0 * float(layer_scores[li])
                is_prot = li in prot_layers
                for proj, _ in _PROJ_SPECS:
                    codes = ("S3_K", "S4_K") if (is_prot and not allow_prot_s2) else ("S2_K", "S3_K", "S4_K")
                    opts = []
                    for cd in codes:
                        wb, eh = matrix_errors[(li, proj)][cd]
                        opts.append((cd, wb, eh * w_xray))
                    items.append({"id": (li, proj), "options": opts})
            sel, used_b, _ = solve_knapsack_bit_allocation(items, budget_bytes)
            return sel, used_b

        plan_rtn = {(li, p): "RTN3" for li in range(n_layers) for p, _ in _PROJ_SPECS}
        plan_s3 = {(li, p): "S3_K" for li in range(n_layers) for p, _ in _PROJ_SPECS}
        plan_k319, bytes_k319 = _build_knapsack(uniform_s3_bytes, allow_prot_s2=False)
        budget_265 = int(round(total_params * 2.65 / 8.0))
        plan_k265, bytes_k265 = _build_knapsack(budget_265, allow_prot_s2=True)

        def _eval_arm(arm_nm: str, plan: dict[tuple[int, str], str], wire_b: int, *, do_qat: bool = False) -> None:
            _restore_norms_and_weights(model, orig_norms, blocks, qdir, dev, dtype)
            _apply_quant_arm(blocks, qdir, plan, dev, dtype)
            qat_meta = None
            if do_qat:
                qat_meta = run_norm_scale_qat(model, tr_recs, dv_recs, device=dev, epochs=2, lr=5e-4, grad_accum=4, seed=int(args.seed))
            ppl = _eval_perplexity(model, wt["test_ppl"], dev)
            kl_res = eval_topk_kl_and_top1(model, te_recs, dev)
            mmlu_r = _eval_abcd_mc(model, mmlu_pids, mmlu_gold, mmlu_ismath, letter_ids, pad_id, dev)
            gsm_r = _eval_gsm8k(model, tok, gsm_ds, dev, batch_size=32)
            bpw = round((wire_b * 8.0) / max(1, total_params), 4)
            arms_out[arm_nm] = {
                "avg_bits_per_weight": bpw,
                "wire_mb": round(wire_b / (1024**2), 2),
                "wikitext2_ppl": ppl,
                "kl_mean": round(kl_res["kl_mean"], 6),
                "top1_agreement": round(kl_res["top1_agreement"], 4),
                "mmlu_200": mmlu_r,
                "gsm8k_100": gsm_r,
                "scale_qat": qat_meta,
            }
            print(
                f"[exp2a] {arm_nm:30s} | bpw={bpw:.3f} | PPL={ppl:.3f} | KL={kl_res['kl_mean']:.5f} | "
                f"Top1={kl_res['top1_agreement']*100:.2f}% | MMLU={mmlu_r['accuracy']*100:.1f}% | GSM8K={gsm_r['accuracy']*100:.1f}%",
                flush=True,
            )

        _eval_arm("uniform_rtn_3b", plan_rtn, uniform_s3_bytes)
        _eval_arm("uniform_s3k_h64_gptq_flip", plan_s3, uniform_s3_bytes)
        _eval_arm("xray_knapsack_3_19b", plan_k319, bytes_k319)
        _eval_arm("xray_knapsack_3_19b_scale_qat", plan_k319, bytes_k319, do_qat=True)
        _eval_arm("xray_knapsack_2_65b_scale_qat", plan_k265, bytes_k265, do_qat=True)

    # Summarize per-layer code distribution for Exp 3
    per_layer_k319 = [{} for _ in range(n_layers)]
    for (li, _), cd in plan_k319.items():
        per_layer_k319[li][cd] = per_layer_k319[li].get(cd, 0) + 1

    return {
        "experiment": "xray_strata_quant_dense",
        "model": args.model,
        "total_quantized_params": total_params,
        "xray_quant_sensitivity_map": qmap,
        "per_layer_s2_err": [round(x, 6) for x in per_layer_s2_err],
        "per_layer_s3_err": [round(x, 6) for x in per_layer_s3_err],
        "per_layer_k319_codes": per_layer_k319,
        "arms": arms_out,
        "timing_s": {"total_s": round(time.perf_counter() - t0, 2)},
        "criteria": {
            "a_uniform_s3k_beats_rtn": arms_out["uniform_s3k_h64_gptq_flip"]["kl_mean"] < arms_out["uniform_rtn_3b"]["kl_mean"],
            "b_xray_knapsack_beats_uniform_matched_bytes": arms_out["xray_knapsack_3_19b"]["kl_mean"] <= arms_out["uniform_s3k_h64_gptq_flip"]["kl_mean"],
            "c_scale_qat_improves_kl_and_ppl": arms_out["xray_knapsack_3_19b_scale_qat"]["kl_mean"] < arms_out["xray_knapsack_3_19b"]["kl_mean"],
            "d_3_19b_gsm8k_within_5pts_of_bf16": arms_out["xray_knapsack_3_19b_scale_qat"]["gsm8k_100"]["accuracy"] >= base_gsm["accuracy"] - 0.05,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--xray", default=str(DEFAULT_XRAY_PATH))
    ap.add_argument("--n-gsm8k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
    _print_summary(payload)


if __name__ == "__main__":
    main()
