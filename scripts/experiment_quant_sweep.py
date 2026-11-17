"""Quantization sweep for WSI token activation traces (`CompactTrace` / `trace_all`).

Sweeps INT8, INT4, NVFP4-style E2M1 FP4, and L2-optimal 1.58-bit Ternary (5 trits/byte)
with optional Walsh-Hadamard block rotation across group sizes on real Qwen3.5 hidden states.

Measures per config:
  - Exact compressed CompactTrace size (bytes, MB, bits/elem, compression vs dense fp32 trace_all)
    and exact `uint16` vs legacy `int32` Top-K index byte savings.
  - Layer-wise residual reconstruction (`mean_cosine`, `min_cosine`, `mean_rel_l2`, `last_rel_l2`).
  - Downstream `norm -> lm_head` readout fidelity (`KL(P_fp32 || P_quant)`, `top1_match`, `top5_overlap`).
  - Functional diagnostics agreement (`detect_loop_onset` onset token & max lag cosine vs fp16).

Usage (Colab / GPU):
    python scripts/experiment_quant_sweep.py --model Qwen/Qwen3.5-0.8B --out results_quant_sweep.json
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsi.capture import find_blocks, trace_all  # noqa: E402
from wsi.tracer import (  # noqa: E402
    _decode_resid,
    _encode_resid,
    capture_compact,
    quantization_error,
)

SWEEP_CONFIGS: list[tuple[str, int, int]] = [
    ("fp16", 16, 0),
    ("int8", 16, 0),
    ("int8", 64, 64),
    ("int4", 16, 0),
    ("int4", 16, 64),
    ("int4", 64, 0),
    ("int4", 64, 64),
    ("fp4", 16, 0),
    ("fp4", 16, 64),
    ("fp4", 64, 0),
    ("fp4", 64, 64),
    ("ternary", 8, 0),
    ("ternary", 16, 0),
    ("ternary", 16, 64),
    ("ternary", 32, 0),
    ("ternary", 32, 64),
    ("ternary", 64, 0),
    ("ternary", 64, 64),
]

CANDIDATE_PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
]


def _local_or_hub(name_or_dir: str) -> str:
    p = Path(name_or_dir)
    if p.exists():
        return str(p)
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_dir)


def _load_model(mdir: str, dev: torch.device):
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    try:
        return AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype, attn_implementation="eager").to(dev).eval()
    except Exception:
        return AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype).to(dev).eval()


def _find_norm_and_head(model):
    head = model.get_output_embeddings()
    if head is None:
        head = getattr(model, "lm_head", None)
    for owner in (
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        model,
    ):
        if owner is not None and hasattr(owner, "norm"):
            return owner.norm, head
    raise RuntimeError("Could not locate final norm + lm_head on model")


def _trace_byte_breakdown(ct) -> dict[str, int]:
    resid_b = sum(
        r["data"].numel() * r["data"].element_size()
        + (r["scales"].numel() * r["scales"].element_size() if r["scales"] is not None else 0)
        for r in ct.resid.values()
    )
    topk_b = sum(t.numel() * t.element_size() for m in ct.mlp_topk.values() for t in m.values())
    head_b = sum(h["l2"].numel() * h["l2"].element_size() for h in ct.head_norms.values())
    idx_elem_bytes = next(iter(ct.mlp_topk.values()))["idx"].element_size() if ct.mlp_topk else 2
    n_idx = sum(m["idx"].numel() for m in ct.mlp_topk.values())
    uint16_saved = n_idx * max(0, 4 - idx_elem_bytes)
    return {
        "resid_bytes": int(resid_b),
        "mlp_topk_bytes": int(topk_b),
        "head_norms_bytes": int(head_b),
        "total_bytes": int(ct.size_bytes()),
        "uint16_saved_bytes": int(uint16_saved),
    }


def _logit_metrics(log_p_true: torch.Tensor, p_true: torch.Tensor, top1_true: torch.Tensor,
                   top5_true: torch.Tensor, h_last: torch.Tensor, norm_fp32, head_fp32,
                   dev: torch.device) -> dict[str, float]:
    with torch.no_grad():
        logits_q = head_fp32(norm_fp32(h_last.to(dev))).float()
        log_q = F.log_softmax(logits_q, dim=-1)
        kl = float((p_true * (log_p_true - log_q)).sum(dim=-1).mean().item())
        top1_match = float((logits_q.argmax(dim=-1) == top1_true).float().mean().item())
        top5_q = logits_q.topk(5, dim=-1).indices
        overlap = float((top5_q.unsqueeze(-1) == top5_true.unsqueeze(-2)).any(dim=-1).float().mean().item())
    return {"kl_fp32": round(kl, 6), "top1_match": round(top1_match, 4), "top5_overlap": round(overlap, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantization sweep for WSI CompactTrace activations")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--eval-tokens", type=int, default=256)
    ap.add_argument("--top-k-mlp", type=int, default=256)
    ap.add_argument("--out", default="results_quant_sweep.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if args.device == "auto" else args.device))
    print(f"Device: {dev} | Model: {args.model}", flush=True)

    mdir = _local_or_hub(args.model)
    tok = AutoTokenizer.from_pretrained(mdir)
    model = _load_model(mdir, dev)
    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    loop_layer = min(15, n_layers - 1)

    # 1. Pick a real sequence (preferring one with loop onset, like experiment_trace_xray.py)
    seq, prompt_len, gen_s, chosen_q = None, 0, 0.0, CANDIDATE_PROMPTS[0]
    for q in CANDIDATE_PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        enc = tok(prompt, return_tensors="pt").to(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        gen_s = round(time.perf_counter() - t0, 2)
        cand = gen[0].cpu()
        p_len = int(enc["input_ids"].shape[1])
        ct_try = capture_compact(model, cand, top_k_mlp=args.top_k_mlp, resid_quant="int8", group_size=16)
        onset_try = ct_try.detect_loop_onset(layer=loop_layer, window=16, thr=0.96)
        seq, prompt_len, chosen_q = cand, p_len, q
        print(f"Gen {int(cand.numel()) - p_len} tok in {gen_s}s | loop_onset={onset_try}", flush=True)
        if onset_try["onset_token"] is not None:
            break

    T = int(seq.numel())
    # Dense fp32 trace_all reference baseline over <= 32 tokens
    slice32 = seq[: min(32, T)]
    tr32 = trace_all(model, slice32, max_tokens=32)
    dense_fp32_per_tok = sum(v.numel() * 4 for v in tr32.values()) / max(1, int(slice32.numel()))
    dense_fp32_full_bytes = int(dense_fp32_per_tok * T)
    del tr32
    gc.collect()

    # Exact fp32 hidden states over strided tokens spanning prompt -> reasoning -> loop tail
    eval_seq = seq[: min(T, 1600)]
    eval_len = int(eval_seq.numel())
    with torch.no_grad():
        hs_all = model(input_ids=eval_seq.unsqueeze(0).to(dev), output_hidden_states=True, use_cache=False).hidden_states
    eval_n = min(args.eval_tokens, eval_len)
    pick_idx = torch.linspace(0, eval_len - 1, steps=eval_n).long()
    true_hs = [hs_all[l + 1][0, pick_idx].float().cpu() for l in range(n_layers)]
    hidden_size = int(true_hs[0].shape[-1])
    del hs_all
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Prepare fp32 final norm + lm_head copies for accurate logit KL comparison
    norm_mod, head_mod = _find_norm_and_head(model)
    norm_fp32 = copy.deepcopy(norm_mod).to(dev).float().eval()
    head_fp32 = copy.deepcopy(head_mod).to(dev).float().eval()
    with torch.no_grad():
        logits_true = head_fp32(norm_fp32(true_hs[-1].to(dev))).float()
        log_p_true = F.log_softmax(logits_true, dim=-1)
        p_true = log_p_true.exp()
        top1_true = logits_true.argmax(dim=-1)
        top5_true = logits_true.topk(5, dim=-1).indices

    # Measure model native bfloat16 precision floor on the same last-layer hidden state
    bf16_floor = _logit_metrics(log_p_true, p_true, top1_true, top5_true,
                                true_hs[-1].bfloat16().float(), norm_fp32, head_fp32, dev)

    # Run fp16 CompactTrace reference first for loop onset baseline
    ref_onset = None
    rows: list[dict] = []
    header = f"{'cfg':<20} {'MB':>6} {'vs_fp32':>8} {'b/el':>5} {'cos':>8} {'relL2':>7} {'KL':>8} {'top1':>6} {'onset':>6} {'s':>5}"
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)

    for quant, gs, hb in SWEEP_CONFIGS:
        label = f"{quant}_g{gs}" + (f"_had{hb}" if hb > 0 else "")
        # 1) Full-sequence capture_compact timing, bytes, and loop onset
        t1 = time.perf_counter()
        ct = capture_compact(
            model, seq, top_k_mlp=args.top_k_mlp,
            resid_quant=quant, group_size=gs, hadamard_block=hb,
        )
        cap_s = round(time.perf_counter() - t1, 2)
        binfo = _trace_byte_breakdown(ct)
        onset = ct.detect_loop_onset(layer=loop_layer, window=16, thr=0.96)
        if quant == "fp16" and ref_onset is None:
            ref_onset = onset

        # 2) Residual fidelity across ALL layers on eval slice + downstream lm_head KL
        layer_cos, layer_l2 = [], []
        h_last_hat = None
        for l in range(n_layers):
            enc_l = _encode_resid(true_hs[l], quant, gs, hb)
            dec_l = _decode_resid(enc_l, quant, gs, hb)
            err = quantization_error(true_hs[l], dec_l)
            layer_cos.append(err["cosine"])
            layer_l2.append(err["rel_l2"])
            if l == n_layers - 1:
                h_last_hat = dec_l
        logit_m = _logit_metrics(log_p_true, p_true, top1_true, top5_true, h_last_hat, norm_fp32, head_fp32, dev)

        bits_per_elem = round(binfo["resid_bytes"] * 8.0 / max(1, n_layers * T * hidden_size), 2)
        comp_vs_dense = round(dense_fp32_full_bytes / max(1, binfo["total_bytes"]), 1)
        mb = round(binfo["total_bytes"] / 2**20, 2)
        mean_cos = round(sum(layer_cos) / len(layer_cos), 6)
        min_cos = round(min(layer_cos), 6)
        mean_l2 = round(sum(layer_l2) / len(layer_l2), 6)
        onset_tok = onset["onset_token"]
        ref_tok = ref_onset["onset_token"] if ref_onset else None
        onset_delta = (onset_tok - ref_tok) if (onset_tok is not None and ref_tok is not None) else None

        row = {
            "config": label,
            "resid_quant": quant,
            "group_size": gs,
            "hadamard_block": hb,
            "capture_s": cap_s,
            **binfo,
            "total_mb": mb,
            "kb_per_tok": round(binfo["total_bytes"] / max(1, T) / 1024, 2),
            "resid_bits_per_elem": bits_per_elem,
            "compression_vs_dense_fp32": comp_vs_dense,
            "mean_cos": mean_cos,
            "min_layer_cos": min_cos,
            "mean_rel_l2": mean_l2,
            "last_layer_cos": layer_cos[-1],
            "last_layer_rel_l2": layer_l2[-1],
            **logit_m,
            "loop_onset_token": onset_tok,
            "loop_onset_delta_vs_fp16": onset_delta,
            "loop_max_lag_cos": onset["max_lag_cos"],
        }
        rows.append(row)
        print(
            f"{label:<20} {mb:>6.2f} {comp_vs_dense:>7.1f}x {bits_per_elem:>5.2f} "
            f"{mean_cos:>8.5f} {mean_l2:>7.4f} {logit_m['kl_fp32']:>8.5f} "
            f"{logit_m['top1_match']:>6.3f} {str(onset_tok):>6} {cap_s:>5.1f}",
            flush=True,
        )
        del ct
        gc.collect()

    del norm_fp32, head_fp32, model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    out = {
        "model": args.model,
        "device": str(dev),
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "seq_len": T,
        "prompt_len": prompt_len,
        "eval_tokens": eval_n,
        "top_k_mlp": args.top_k_mlp,
        "chosen_prompt": chosen_q,
        "dense_fp32_full_bytes": dense_fp32_full_bytes,
        "dense_fp32_mb": round(dense_fp32_full_bytes / 2**20, 2),
        "bfloat16_logit_floor": bf16_floor,
        "loop_layer": loop_layer,
        "fp16_ref_onset": ref_onset,
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved quantization sweep results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
