"""EXPERIMENTAL - is the layer-to-layer capture faithful under real conditions?

Separates "is the captured data right" from "is the metric meaningful". Only the first is tested here.

Real conditions: greedy GSM8K generations of Qwen3.5-0.8B with thinking (the setting where it loops), and the
tracer's default capture settings (int8 residuals, int8 head vectors, group 64, Hadamard 64).
Ground truth: the same sequence through an fp32 forward with hooks on the norms.

Checks, per token region (prompt / reasoning before loop / loop):
  C1  write error per sublayer: ||w_trace - w_true|| / ||w_true||  (A_l via W_O(head_in), M_l by subtraction)
  C2  error relative to the residual it was subtracted from (shows the subtraction amplification risk)
  C3  per-token dependency shares (L <- L-k): p99 / max abs difference vs ground truth
  C4  pair predictor local_err recomputed from trace states: same ranking as from fp32 states?

Run (CPU):  CUDA_VISIBLE_DEVICES="" python scripts/experiment_layer_flow_capture_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_layer_flow import dependency, layer_readers, writes_trace  # noqa: E402
from experiment_layer_parallel import spearman  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import capture_compact  # noqa: E402


def loop_onset(ids: list[int], n: int = 16, window: int = 64, frac: float = 0.5) -> int | None:
    """First position where >= frac of the n-grams in the next window already appeared before (same rule family as the repo)."""
    seen: set = set()
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    for i, g in enumerate(grams):
        if i + window < len(grams):
            fut = grams[i:i + window]
            if sum(f in seen for f in fut) >= frac * window:
                return i
        seen.add(g)
    return None


def fp32_states(model, blocks, ids):
    st = {"in": {}, "kw": {}, "out": {}, "mid": {}}
    hs = []
    for l, blk in enumerate(blocks):
        hs.append(blk.register_forward_pre_hook(
            lambda m, a, k, l=l: (st["in"].__setitem__(l, a[0].detach()), st["kw"].__setitem__(l, (a[1:], k)), None)[-1],
            with_kwargs=True))
        hs.append(blk.register_forward_hook(
            lambda m, a, o, l=l: st["out"].__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach())))
        hs.append(blk.post_attention_layernorm.register_forward_pre_hook(
            lambda m, a, l=l: st["mid"].__setitem__(l, a[0].detach()[0].float())))
    try:
        with torch.no_grad():
            model(input_ids=ids.reshape(1, -1), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return st


def local_err(blocks, st, xs, n):
    """Pair predictor: what block L writes when fed x_{L-2} instead of x_{L-1}, using states xs (list of (T,H))."""
    out = []
    with torch.no_grad():
        for L in range(1, n):
            a_rest, kw = st["kw"][L]
            o_skip = blocks[L](xs[L - 1].unsqueeze(0), *a_rest, **kw)
            o_true = blocks[L](xs[L].unsqueeze(0), *a_rest, **kw)
            o_skip = (o_skip[0] if isinstance(o_skip, tuple) else o_skip)[0].float()
            o_true = (o_true[0] if isinstance(o_true, tuple) else o_true)[0].float()
            e = (o_skip - xs[L - 1]) - (o_true - xs[L])
            out.append((e[1:].norm(dim=-1) / o_true[1:].norm(dim=-1)).mean().item())
    return out


def q(t: torch.Tensor, p: float) -> float:
    return round(torch.quantile(t.float().flatten(), p).item(), 5) if t.numel() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen35-0.8b-full")
    ap.add_argument("--out", default="benchmarks/layer_flow")
    ap.add_argument("--n-seqs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    _, blocks = find_blocks(model)
    n = len(blocks)
    readers = layer_readers(model)
    gtr = pd.read_parquet(ROOT / "work/gsm8k/train.parquet")

    report = {"settings": "capture_compact defaults: resid int8, head int8, group 64, hadamard 64", "seqs": []}
    for si in range(args.n_seqs):
        qtext = gtr["question"].iloc[100 + si]
        prompt = tok.apply_chat_template([{"role": "user", "content": qtext}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p_ids = tok(prompt, return_tensors="pt").input_ids[0]
        with torch.no_grad():
            g = model.generate(p_ids.unsqueeze(0), max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)[0]
        gen = g[len(p_ids):].tolist()
        onset = loop_onset(gen)
        P = len(p_ids)
        regions = {"prompt": slice(1, P), "reasoning": slice(P, P + (onset if onset is not None else len(gen)))}
        if onset is not None:
            regions["loop"] = slice(P + onset, len(g))
        print(f"seq {si}: prompt {P}, gen {len(gen)}, loop onset {onset} ({time.time() - t0:.0f}s)", flush=True)

        st = fp32_states(model, blocks, g)
        xin = [st["in"][l][0].float() for l in range(n)]
        xout = [st["out"][l][0].float() for l in range(n)]
        d_true = [xin[0]]
        for l in range(n):
            d_true += [st["mid"][l] - xin[l], xout[l] - st["mid"][l]]
        d_true = torch.stack(d_true)

        tr = capture_compact(model, g, top_k_attn=0, capture_cache=False)  # defaults: int8 resid + int8 heads
        d_tr = writes_trace(tr, readers, xin[0])

        err = (d_tr - d_true).norm(dim=-1)                      # (n_w, T)
        rel_w = err / d_true.norm(dim=-1).clamp_min(1e-6)
        x_true = torch.stack([xin[0]] + xout)                   # residual the write was subtracted from
        rel_x = err[1:] / x_true[1:].repeat_interleave(2, 0).norm(dim=-1).clamp_min(1e-6)

        dep_t, dep_q = dependency(d_true, readers), dependency(d_tr, readers)
        sh_diff = torch.cat([(dep_t["per_tok"][k] - dep_q["per_tok"][k]).abs().flatten() for k in dep_t["per_tok"]])

        seq = {"prompt_tokens": P, "gen_tokens": len(gen), "loop_onset": onset, "regions": {}}
        for rn, sl in regions.items():
            ra, rm = rel_w[1::2, sl], rel_w[2::2, sl]
            sd = torch.cat([(dep_t["per_tok"][k][:, sl] - dep_q["per_tok"][k][:, sl]).abs().flatten() for k in dep_t["per_tok"]])
            seq["regions"][rn] = {
                "tokens": sl.stop - sl.start,
                "C1_attn_write_rel_err_median": q(ra, 0.5), "C1_attn_write_rel_err_p99": q(ra, 0.99),
                "C1_mlp_write_rel_err_median": q(rm, 0.5), "C1_mlp_write_rel_err_p99": q(rm, 0.99),
                "C2_err_vs_residual_p99": q(rel_x[:, sl], 0.99),
                "C3_share_absdiff_p99": q(sd, 0.99), "C3_share_absdiff_max": round(sd.max().item(), 5),
            }
        worst = rel_w[1:].median(dim=1).values
        seq["worst_writers_median_rel_err"] = sorted(
            [(nm, round(v, 4)) for nm, v in zip([f"{s}{l}" for l in range(n) for s in "AM"], worst.tolist())],
            key=lambda x: -x[1])[:5]
        seq["C3_mean_matrix_corr"] = round(float(torch.corrcoef(torch.stack([
            dep_t["D_attn"].nan_to_num().flatten(), dep_q["D_attn"].nan_to_num().flatten()]))[0, 1]), 5)

        xs_tr = [xin[0]] + [tr.dequant_resid(l).float() for l in range(n)]
        le_true = local_err(blocks, st, [xin[0]] + xout, n)
        le_tr = local_err(blocks, st, xs_tr, n)
        seq["C4_local_err_spearman_trace_vs_fp32"] = round(spearman(le_true, le_tr), 4)
        seq["C4_local_err_max_rel_diff"] = round(max(abs(a - b) / max(a, 1e-9) for a, b in zip(le_true, le_tr)), 4)
        seq["C3_overall_share_absdiff_p99"] = q(sh_diff, 0.99)
        report["seqs"].append(seq)
        print(json.dumps(seq, indent=1), flush=True)

    report["seconds"] = round(time.time() - t0, 1)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_capture_check_cpu.json").write_text(json.dumps(report, indent=1))
    print("saved", out / "results_capture_check_cpu.json")


if __name__ == "__main__":
    main()
