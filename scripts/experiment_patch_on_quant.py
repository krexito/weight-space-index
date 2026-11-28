"""EXPERIMENTAL: does a WSI patch survive a GPTQ-quantized base? (LFM3-1.7B over-refusal; follow-up of the Chantless
comparison, where patches meant for llama.cpp/GGUF would run on a quantized model.)

Base weights: bf16, then every nn.Linear inside the decoder blocks quantized with wsi.strata_quant GPTQ + 64-flip
(S4_K = 4.19 b/w and S3_K = 3.19 b/w, H64 rotation), sequentially layer by layer (each layer's Hessian is taken with
the earlier layers already quantized), calibrated on 32 x 512 WikiText-2 train tokens (no refusal data). The
dequantized weights are written back in bf16 (fake quant: what a GGUF runtime computes, up to kernel rounding).
Patches are applied unchanged, as shipped (no re-fit, router dir/thr unchanged; fingerprint check off):
  patch  the shipped 2-direction patch (patches/lfm3_1.7b_abliterated_sfw_routed.moipatch), routed
  del    the 16 zeroed w2 neurons picked in experiment_refusal_deletion.py, same router
Exam: the fresh prompts of test 2 (OR-Bench-hard 400-639, toxic 0-199, GSM8K 300-339), 128 tokens. Per precision:
recovery and toxic kept against THAT precision's own base, plus router fire rates and drift from the bf16 base.

  python scripts/experiment_patch_on_quant.py --deletion R_DEL.json --out R.json [--limit N --max-new 8 --codes S4_K]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import experiment_refusal_controls as rc  # noqa: E402
import experiment_refusal_deletion as rd  # noqa: E402
import experiment_refusal_select as rs  # noqa: E402
import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
from wsi.moipatch import MoIPatch  # noqa: E402
from wsi.strata_quant import code_bits_per_weight, gptq_flip_quantize_sk_torch  # noqa: E402


def wikitext_chunks(tok, n: int, seq_len: int = 512) -> list[list[int]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(str(t) for t in pq.read_table(p, columns=["text"]).column("text").to_pylist() if str(t).strip())
    ids = tok.encode(text[:2_000_000], add_special_tokens=False)
    return [ids[i:i + seq_len] for i in range(0, len(ids) - seq_len, seq_len)][:n]


@torch.no_grad()
def quantize_model(model, blocks, calib: list[list[int]], code: str, dev) -> dict:
    stats = {"n_linear": 0, "err_h": 0.0, "wire_bytes": 0, "n_weights": 0}
    for bi, blk in enumerate(blocks):
        lins = {n: m for n, m in blk.named_modules() if isinstance(m, torch.nn.Linear)}
        H = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lins.items()}
        hooks = []
        for n, m in lins.items():
            def pre(_m, inp, n=n):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[n].add_(x.T @ x)
            hooks.append(m.register_forward_pre_hook(pre))
        for s in calib:
            model(input_ids=torch.tensor([s], device=dev))
        for h in hooks:
            h.remove()
        for n, m in lins.items():
            Hn = H.pop(n) / max(1, len(calib))
            wq, wb, eh = gptq_flip_quantize_sk_torch(m.weight.detach().float(), Hn, code, use_h64=True, max_flips=64)
            m.weight.data.copy_(wq.to(m.weight.dtype))
            stats["n_linear"] += 1
            stats["err_h"] += eh
            stats["wire_bytes"] += wb
            stats["n_weights"] += m.weight.numel()
            del Hn
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    stats["bits_per_weight"] = round(8 * stats["wire_bytes"] / max(1, stats["n_weights"]), 3)
    return stats


def gen_arm(model, tok, fmt, dev, n, bs, patch):
    ctx = patch.enable_routed(model, mode="auto", verify_fingerprint=False) if patch else nullcontext()
    with ctx as routed:
        out = {k: sr._generate_texts(model, tok, v, dev, max_new_tokens=n, batch_size=bs) for k, v in fmt.items()}
        fires = {k: round(float(np.mean(sr._probe_router_fires(model, tok, routed, v, dev))), 4)
                 for k, v in fmt.items()} if patch else None
    return out, fires


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--patch", default=str(HERE.parent / "patches/lfm3_1.7b_abliterated_sfw_routed.moipatch"))
    ap.add_argument("--deletion", required=True, help="results JSON of experiment_refusal_deletion.py (picked_neurons)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--codes", default="S4_K,S3_K")
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    model, tok, dev, _ = sr._load_model_and_tokenizer(a.model, "bf16" if torch.cuda.is_available() else "fp32")
    from wsi.capture import find_blocks

    _, blocks = find_blocks(model)
    _, writers = sr._discover_all_writers(model)
    shipped = MoIPatch.load(a.patch)
    picks = json.loads(Path(a.deletion).read_text())["picked_neurons"]
    data = rs.load_splits()
    data = {k: data[k] for k in ("exam_or", "exam_toxic", "exam_ctrl")}
    if a.limit:
        data = {k: v[: a.limit] for k, v in data.items()}
    fmt = {k: [sr._format_prompt(tok, p) for p in v] for k, v in data.items()}
    calib = wikitext_chunks(tok, a.n_calib if not a.limit else 4)
    orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    res = {"model": a.model, "n": {k: len(v) for k, v in data.items()}, "n_calib_seqs": len(calib), "precisions": {}}
    gens_bf16_base = None
    for code in ["bf16"] + [c for c in a.codes.split(",") if c]:
        if code != "bf16":
            model.load_state_dict(orig)
            q = quantize_model(model, blocks, calib, code, dev)
            q["nominal_bits_per_weight"] = code_bits_per_weight(code)
            print(f"[{code}] quantized {q['n_linear']} linears, {q['bits_per_weight']} b/w ({time.time() - t0:.0f}s)",
                  flush=True)
        else:
            q = {}
        arms = {"base": None, "patch": shipped,
                "del": rd.deletion_patch(writers, picks, shipped.router, "del16")}  # columns read from current weights
        gens, fires = {}, {}
        for nm, P in arms.items():
            gens[nm], fires[nm] = gen_arm(model, tok, fmt, dev, a.max_new, a.batch_size, P)
            print(f"[{code}/{nm}] " + " ".join(f"{k}: {sum(r['refused'] for r in v)}/{len(v)}" for k, v in gens[nm].items())
                  + f" fires {fires[nm]} ({time.time() - t0:.0f}s)", flush=True)
        if code == "bf16":
            gens_bf16_base = gens["base"]
        out = {"quant": q, "arms": {}}
        for nm in arms:
            o = rs.outcome(gens["base"], gens[nm], "exam")
            o["refusal_rate"] = {k: round(float(np.mean([r["refused"] for r in v])), 4) for k, v in gens[nm].items()}
            o["router_fire_rate"] = fires[nm]
            o["degenerate"] = {k: round(float(np.mean([rc.degenerate(r["gen_ids"]) for r in v])), 4)
                               for k, v in gens[nm].items()}
            out["arms"][nm] = o
        # drift of this precision's base from the bf16 base
        out["base_vs_bf16"] = {k: round(float(np.mean([x["refused"] == y["refused"] for x, y in
                                                       zip(gens["base"][k], gens_bf16_base[k])])), 4)
                               for k in fmt} | {"ctrl_identical": round(float(np.mean(
                                   [x["gen_ids"] == y["gen_ids"] for x, y in
                                    zip(gens["base"]["exam_ctrl"], gens_bf16_base["exam_ctrl"])])), 4)}
        res["precisions"][code] = out
        Path(a.out).write_text(json.dumps(res, indent=1))
    res["elapsed_s"] = round(time.time() - t0)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for code, out in res["precisions"].items():
        for nm, o in out["arms"].items():
            print(f"{code:5s} {nm:5s} recovered {o['recovered']:.3f} toxic kept {o['toxic_kept']:.3f} "
                  f"ctrl same {o['ctrl_identical']:.2f} style_ok {o['style_ok']} refusal {o['refusal_rate']} "
                  f"fires {o['router_fire_rate']}")


if __name__ == "__main__":
    main()
