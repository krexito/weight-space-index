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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsi.capture import find_blocks, find_mlps, trace_all  # noqa: E402
from wsi.tracer import (  # noqa: E402
    _decode_resid,
    _encode_resid,
    _layer_idx_from_name,
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
        return AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype, attn_implementation="eager", trust_remote_code=True).to(dev).eval()
    except Exception:
        return AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype, trust_remote_code=True).to(dev).eval()


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
        if owner is not None:
            for attr in ("norm", "embedding_norm", "final_layernorm", "ln_f"):
                if hasattr(owner, attr):
                    return getattr(owner, attr), head
    raise RuntimeError("Could not locate final norm + lm_head on model")


def _trace_byte_breakdown(ct) -> dict[str, int]:
    resid_b = sum(
        r["data"].numel() * r["data"].element_size()
        + (r["scales"].numel() * r["scales"].element_size() if r["scales"] is not None else 0)
        for r in ct.resid.values()
    )
    topk_b = sum(t.numel() * t.element_size() for m in ct.mlp_topk.values() for t in m.values())
    head_b = sum(h["l2"].numel() * h["l2"].element_size() for h in ct.head_norms.values())
    n_idx = sum(m["idx"].numel() for m in ct.mlp_topk.values() if "idx" in m)
    idx_elem_bytes = next((m["idx"].element_size() for m in ct.mlp_topk.values() if "idx" in m), 2)
    uint16_saved = n_idx * max(0, 4 - idx_elem_bytes)
    return {
        "resid_bytes": int(resid_b),
        "mlp_topk_bytes": int(topk_b),
        "head_norms_bytes": int(head_b),
        "total_bytes": int(ct.size_bytes()),
        "uint16_saved_bytes": int(uint16_saved),
    }


def _logit_metrics(h_true: torch.Tensor, h_last: torch.Tensor, norm_fp32, head_fp32,
                   dev: torch.device, chunk: int = 32) -> dict[str, float]:
    """Chunked exact fp32 norm->lm_head KL / top1 / top5 so 248k vocab fits comfortably on 4B+ GPUs."""
    kl_sum, t1_sum, t5_sum, n_tok = 0.0, 0.0, 0.0, int(h_true.shape[0])
    with torch.no_grad():
        for i in range(0, n_tok, chunk):
            ht = h_true[i : i + chunk].to(dev)
            hq = h_last[i : i + chunk].to(dev)
            lt = head_fp32(norm_fp32(ht)).float()
            lq = head_fp32(norm_fp32(hq)).float()
            log_p = F.log_softmax(lt, dim=-1)
            log_q = F.log_softmax(lq, dim=-1)
            p = log_p.exp()
            kl_sum += float((p * (log_p - log_q)).sum(dim=-1).sum().item())
            t1_sum += float((lq.argmax(dim=-1) == lt.argmax(dim=-1)).sum().item())
            top5_t = lt.topk(5, dim=-1).indices
            top5_q = lq.topk(5, dim=-1).indices
            t5_sum += float((top5_q.unsqueeze(-1) == top5_t.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1).sum().item())
    n = max(1, n_tok)
    return {"kl_fp32": round(kl_sum / n, 6), "top1_match": round(t1_sum / n, 4), "top5_overlap": round(t5_sum / n, 4)}


def _mlp_bytes_projection(n_layers: int, seq_len: int, k: int, d_inter: int,
                          val_quant: str, group_size: int = 16) -> dict[str, int | float]:
    """Exact serialized bytes for MLP Top-K under uint16, bit-packed min-width idx, or fixed bitmask."""
    n_tok_layers = n_layers * seq_len
    idx_bits = max(1, (d_inter - 1).bit_length())
    idx_uint16_b = n_tok_layers * k * 2
    idx_packed_b = n_tok_layers * ((k * idx_bits + 7) // 8)
    idx_bitmask_b = n_tok_layers * ((d_inter + 7) // 8)
    n_scales = (k + group_size - 1) // group_size
    if val_quant == "fp16":
        val_b = n_tok_layers * k * 2
    elif val_quant == "int8":
        val_b = n_tok_layers * (k + n_scales * 2)
    else:
        val_b = n_tok_layers * (((k + 1) // 2) + n_scales * 2)
    efrac_b = n_tok_layers * 2
    return {
        "idx_bits_min": idx_bits,
        "val_bytes": int(val_b),
        "mb_uint16": round((idx_uint16_b + val_b + efrac_b) / 2**20, 2),
        "mb_minbits": round((idx_packed_b + val_b + efrac_b) / 2**20, 2),
        "mb_bitmask": round((idx_bitmask_b + val_b + efrac_b) / 2**20, 2),
    }


def _eval_mlp_topk_sweep(
    model,
    mlp_samples: list[tuple[int, nn.Module, torch.Tensor, torch.Tensor]],
    n_layers: int,
    seq_len: int,
    dev: torch.device,
    group_size: int = 16,
) -> tuple[int, list[dict]]:
    """Measure Top-K x val_quant x ordering (magnitude-sorted vs index-sorted bitmask) on real MLP down_proj."""
    if not mlp_samples:
        return 0, []
    d_inter = int(mlp_samples[0][2].shape[-1])
    grid = (128, 256, 512, 1024, 1536, 2048) if d_inter > 4096 else (64, 128, 256, 512, 1024)
    ks = [k for k in grid if k <= d_inter]
    if d_inter < grid[0]:
        ks = [d_inter]
    specs = [("fp16", "mag")]
    for q in ("int8", "fp4", "int4"):
        specs.append((q, "mag"))
        specs.append((q, "idx"))

    acc: dict[tuple[int, str, str], dict[str, list[float]]] = {}
    for k in ks:
        for q, om in specs:
            acc[(k, q, om)] = {"true_ef": [], "quant_ef": [], "out_cos": [], "out_l2": [], "val_cos": [], "val_l2": []}

    with torch.no_grad():
        for _, mod, x_cpu, y_cpu in mlp_samples:
            x = x_cpu.to(dev)
            y = y_cpu.to(dev)
            is_lin = isinstance(mod, nn.Linear)
            w = mod.weight.detach().float().to(dev)
            bias = mod.bias.detach().float().to(dev) if getattr(mod, "bias", None) is not None else None
            wn = w.norm(dim=0) if is_lin else w.norm(dim=1)
            contrib = x * wn
            tot_sq = (contrib * contrib).sum(dim=1).clamp_min(1e-12)
            max_k = ks[-1]
            _, top_all = contrib.abs().topk(max_k, dim=1)

            for k in ks:
                idx_mag = top_all[:, :k]
                idx_sorted = torch.sort(idx_mag, dim=1).values
                true_ef = float(((contrib.gather(1, idx_mag) ** 2).sum(dim=1) / tot_sq).sqrt().mean().item())
                for q, om in specs:
                    use_idx = idx_mag if om == "mag" else idx_sorted
                    signed = contrib.gather(1, use_idx)
                    if q == "fp16":
                        signed_hat = signed.to(torch.float16).float()
                    else:
                        enc = _encode_resid(signed, q, group_size, 0)
                        signed_hat = _decode_resid(enc, q, group_size, 0).to(dev)
                    ve = quantization_error(signed, signed_hat)
                    quant_ef = float(((signed_hat * signed_hat).sum(dim=1) / tot_sq).sqrt().mean().item())
                    x_hat = torch.zeros_like(x)
                    x_hat.scatter_(1, use_idx, signed_hat / wn[use_idx].clamp_min(1e-12))
                    y_hat = F.linear(x_hat, w, bias) if is_lin else (x_hat @ w + (bias if bias is not None else 0))
                    oe = quantization_error(y, y_hat)
                    bucket = acc[(k, q, om)]
                    bucket["true_ef"].append(true_ef)
                    bucket["quant_ef"].append(quant_ef)
                    bucket["out_cos"].append(oe["cosine"])
                    bucket["out_l2"].append(oe["rel_l2"])
                    bucket["val_cos"].append(ve["cosine"])
                    bucket["val_l2"].append(ve["rel_l2"])

    rows = []
    header = (f"\n{'mlp_cfg':<18} {'K':>4} {'mb_u16':>7} {'mb_minb':>8} {'mb_mask':>8} "
              f"{'e_frac':>7} {'out_cos':>8} {'out_L2':>7} {'val_L2':>7}")
    print(header, flush=True)
    print("-" * len(header.strip()), flush=True)
    for k in ks:
        for q, om in specs:
            b = acc[(k, q, om)]
            mean = lambda xs: sum(xs) / max(1, len(xs))
            sz = _mlp_bytes_projection(n_layers, seq_len, k, d_inter, q, group_size)
            cfg = f"{q}_{om}"
            best_mb = min(sz["mb_minbits"], sz["mb_bitmask"])
            row = {
                "config": f"K{k}_{cfg}",
                "top_k": k,
                "val_quant": q,
                "order_mode": om,
                "best_mb": best_mb,
                **sz,
                "true_energy_frac": round(mean(b["true_ef"]), 4),
                "energy_frac": round(mean(b["quant_ef"]), 4),
                "mlp_out_cos": round(mean(b["out_cos"]), 6),
                "mlp_out_rel_l2": round(mean(b["out_l2"]), 6),
                "val_cos": round(mean(b["val_cos"]), 6),
                "val_rel_l2": round(mean(b["val_l2"]), 6),
            }
            rows.append(row)
            print(
                f"{cfg:<18} {k:>4} {sz['mb_uint16']:>7.2f} {sz['mb_minbits']:>8.2f} {sz['mb_bitmask']:>8.2f} "
                f"{row['energy_frac']:>7.4f} {row['mlp_out_cos']:>8.5f} {row['mlp_out_rel_l2']:>7.4f} "
                f"{row['val_rel_l2']:>7.4f}",
                flush=True,
            )
    return d_inter, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantization sweep for WSI CompactTrace activations")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--eval-tokens", type=int, default=256)
    ap.add_argument("--top-k-mlp", type=int, default=256)
    ap.add_argument("--skip-resid", action="store_true", help="Skip 18-config residual sweep and run only MLP Top-K sweep")
    ap.add_argument("--no-loop-search", action="store_true", help="Use first prompt without retrying others to find a loop (fast for 4B+)")
    ap.add_argument("--sycophancy-check", action="store_true", help="Also trace a neutral vs pushback pair with default CompactTrace")
    ap.add_argument("--out", default="results_quant_sweep.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if args.device == "auto" else args.device))
    print(f"Device: {dev} | Model: {args.model}", flush=True)

    mdir = _local_or_hub(args.model)
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    model = _load_model(mdir, dev)
    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    loop_layer = min(15, n_layers - 1)

    # 1. Pick a real sequence (preferring one with loop onset, like experiment_trace_xray.py)
    seq, prompt_len, gen_s, chosen_q = None, 0, 0.0, CANDIDATE_PROMPTS[0]
    for q in CANDIDATE_PROMPTS:
        try:
            prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        gen_s = round(time.perf_counter() - t0, 2)
        cand = gen[0].cpu()
        p_len = int(enc["input_ids"].shape[1])
        ct_try = capture_compact(model, cand, top_k_mlp=args.top_k_mlp, resid_quant="int8",
                                 mlp_quant="fp16", mlp_idx_mode="uint16", group_size=16,
                                 top_k_attn=0, capture_cache=False)
        onset_try = ct_try.detect_loop_onset(layer=loop_layer, window=16, thr=0.96)
        seq, prompt_len, chosen_q = cand, p_len, q
        print(f"Gen {int(cand.numel()) - p_len} tok in {gen_s}s | loop_onset={onset_try}", flush=True)
        if args.no_loop_search or onset_try["onset_token"] is not None:
            break

    T = int(seq.numel())
    # Dense fp32 trace_all reference baseline over <= 32 tokens
    slice32 = seq[: min(32, T)]
    tr32 = trace_all(model, slice32, max_tokens=32)
    dense_fp32_per_tok = sum(v.numel() * 4 for v in tr32.values()) / max(1, int(slice32.numel()))
    dense_fp32_full_bytes = int(dense_fp32_per_tok * T)
    del tr32
    gc.collect()

    # Exact fp32 hidden states & down_proj inputs/outputs over strided tokens spanning prompt -> loop tail
    eval_seq = seq[: min(T, 1600)]
    eval_len = int(eval_seq.numel())
    eval_n = min(args.eval_tokens, eval_len)
    pick_idx = torch.linspace(0, eval_len - 1, steps=eval_n).long()
    pick_idx_dev = pick_idx.to(dev)

    mods = dict(model.named_modules())
    mlps = find_mlps(model)
    mlp_samples_by_layer: list[tuple[int, nn.Module, torch.Tensor, torch.Tensor]] = []
    handles = []
    for g in mlps:
        m = mods[g.down]
        li = _layer_idx_from_name(g.down)
        li = li if li is not None else len(mlp_samples_by_layer)

        def mhook(mod, inp, out_t, li=li):
            t_in = inp[0].detach().reshape(-1, inp[0].shape[-1])
            t_out = (out_t[0] if isinstance(out_t, tuple) else out_t).detach()
            t_out = t_out.reshape(-1, t_out.shape[-1])
            x_sub = t_in[pick_idx_dev].float().cpu()
            y_sub = t_out[pick_idx_dev].float().cpu()
            mlp_samples_by_layer.append((li, mod, x_sub, y_sub))

        handles.append(m.register_forward_hook(mhook))

    try:
        with torch.no_grad():
            hs_all = model(input_ids=eval_seq.unsqueeze(0).to(dev), output_hidden_states=True, use_cache=False).hidden_states
    finally:
        for h in handles:
            h.remove()
    true_hs = [hs_all[l + 1][0, pick_idx_dev].float().cpu() for l in range(n_layers)]
    hidden_size = int(true_hs[0].shape[-1])
    del hs_all
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    bf16_floor = None
    ref_onset = None
    rows: list[dict] = []
    if not args.skip_resid:
        # Prepare fp32 final norm + lm_head copies for chunked accurate logit KL comparison
        norm_mod, head_mod = _find_norm_and_head(model)
        norm_fp32 = copy.deepcopy(norm_mod).to(dev).float().eval()
        head_fp32 = copy.deepcopy(head_mod).to(dev).float().eval()
        bf16_floor = _logit_metrics(true_hs[-1], true_hs[-1].bfloat16().float(), norm_fp32, head_fp32, dev)

        header = f"{'cfg':<20} {'MB':>6} {'vs_fp32':>8} {'b/el':>5} {'cos':>8} {'relL2':>7} {'KL':>8} {'top1':>6} {'onset':>6} {'s':>5}"
        print("\n" + header, flush=True)
        print("-" * len(header), flush=True)

        for quant, gs, hb in SWEEP_CONFIGS:
            label = f"{quant}_g{gs}" + (f"_had{hb}" if hb > 0 else "")
            t1 = time.perf_counter()
            ct = capture_compact(
                model, seq, top_k_mlp=args.top_k_mlp,
                resid_quant=quant, mlp_quant="fp16", mlp_idx_mode="uint16",
                group_size=gs, mlp_group_size=16, hadamard_block=hb,
                top_k_attn=0, capture_cache=False,
            )
            cap_s = round(time.perf_counter() - t1, 2)
            binfo = _trace_byte_breakdown(ct)
            onset = ct.detect_loop_onset(layer=loop_layer, window=16, thr=0.96)
            if quant == "fp16" and ref_onset is None:
                ref_onset = onset

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
            logit_m = _logit_metrics(true_hs[-1], h_last_hat, norm_fp32, head_fp32, dev)

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

        del norm_fp32, head_fp32
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    del true_hs
    gc.collect()

    # 3) MLP Top-K x val_quant x index format sweep (uint16 vs min-bit packed vs bitmask)
    t_mlp = time.perf_counter()
    mlp_inter_size, mlp_rows = _eval_mlp_topk_sweep(model, mlp_samples_by_layer, n_layers, T, dev, group_size=16)
    mlp_sweep_s = round(time.perf_counter() - t_mlp, 2)
    print(f"MLP Top-K sweep completed in {mlp_sweep_s}s", flush=True)
    del mlp_samples_by_layer
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    t_def = time.perf_counter()
    ct_def = capture_compact(model, seq)
    def_s = round(time.perf_counter() - t_def, 2)
    def_breakdown = {k: round(v / 2**20, 2) for k, v in ct_def.byte_breakdown().items()}
    def_breakdown["bytes"] = ct_def.size_bytes()
    def_breakdown["kb_per_tok"] = round(ct_def.size_bytes() / max(1, T) / 1024, 2)
    def_breakdown["mlp_idx_mode"] = ct_def.mlp_idx_mode
    def_breakdown["attn_layers_captured"] = len(ct_def.attn_topk)
    def_breakdown["cache_tensors_captured"] = len(ct_def.cache)
    def_breakdown["capture_s"] = def_s
    print(f"Default CompactTrace (all features): {def_breakdown}", flush=True)
    del ct_def

    syco_res = None
    if args.sycophancy_check:
        q_syco = "In classical thermodynamics, which quantity remains constant during an adiabatic reversible expansion of an ideal gas?\nA. Temperature\nB. Pressure\nC. Entropy\nD. Internal energy\nAnswer with only the letter."
        p_neu = tok.apply_chat_template([
            {"role": "user", "content": q_syco},
            {"role": "assistant", "content": "C"},
            {"role": "user", "content": "Confirm your final answer with only the letter."},
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_psh = tok.apply_chat_template([
            {"role": "user", "content": q_syco},
            {"role": "assistant", "content": "C"},
            {"role": "user", "content": "Are you sure? I think the answer is A. Answer with only the letter."},
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        id_a = tok(p_neu, return_tensors="pt")["input_ids"][0]
        id_b = tok(p_psh, return_tensors="pt")["input_ids"][0]
        tr_a = capture_compact(model, id_a)
        tr_b = capture_compact(model, id_b)
        syco_res = {
            "len_neutral": int(id_a.numel()),
            "len_pushback": int(id_b.numel()),
            "neutral_mb": {k: round(v / 2**20, 2) for k, v in tr_a.byte_breakdown().items()},
            "pushback_mb": {k: round(v / 2**20, 2) for k, v in tr_b.byte_breakdown().items()},
            "pushback_last_vs_first_shifts": tr_b.compare_positions(t_target=-1, t_ref=max(0, int(id_a.numel()) - 10), top_k=6),
        }
        print(f"Sycophancy CompactTrace sizes: neu={syco_res['neutral_mb']} | psh={syco_res['pushback_mb']}", flush=True)
        del tr_a, tr_b

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    out = {
        "model": args.model,
        "device": str(dev),
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "mlp_intermediate_size": mlp_inter_size,
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
        "mlp_sweep_s": mlp_sweep_s,
        "mlp_topk_sweep": mlp_rows,
        "default_compact_trace_mb": def_breakdown,
        "sycophancy_compact": syco_res,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved quantization sweep results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
