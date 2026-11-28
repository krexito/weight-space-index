"""EXPERIMENTAL (phase 1) - layer-to-layer information flow map from a CompactTrace.

Standalone: imports from `wsi` but changes nothing in it. Writes only to `--out`.

Model of the residual stream: x_L = E + sum_j (A_j + M_j), where A_j / M_j are the writes of the attention / MLP
sublayer of layer j. Every reader (attention input of layer L, MLP input of layer L, the final logit) sees the sum.
For each reader we split what it reads into the *direct linear share* of every earlier writer:

    read = W (g * x / rms(x))     share_w = <W (g * d_w), read> / ||read||^2      (sum over writers = 1, exact)

g is the effective RMSNorm gain (norm(ones)), W the stacked linears that read the normed residual. rms cancels.
This ignores nonlinear effects downstream of the read (first-order view only).

Metrics (phase 1, all from residuals + weights):
  1. dependency matrices  D_attn[L][w], D_mlp[L][w]  (and the final-logit reader)
  2. consumption per writer (sum of |share| over later readers) -> "dead" writes
  3. survival of each write in the final residual / top erasers / redundancy between writes
  4. effective depth per token (longest chain of edges with |share| > thr)
  5. logit lens on a token sample (first layer where the final top-1 appears and stays)
And a check: the same matrices from the fp32 forward (ground truth), an fp16 trace and an int8 trace.

Run (CPU):  CUDA_VISIBLE_DEVICES="" python scripts/experiment_layer_flow.py --model models/qwen35-0.8b-full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import capture_compact  # noqa: E402

TEXT_NORMAL = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May? In May she sold 48 / 2 = 24 clips. "
    "Altogether she sold 48 + 24 = 72 clips in April and May. The answer is 72. "
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
    "How much did she earn? Weng earns 12 / 60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10."
)
LOOP_UNIT = " So we need to check the total again, so the total is the sum of both months."


# ----------------------------------------------------------------------------------------------- model pieces
def _linears_reading(mod: nn.Module, hidden: int) -> list[nn.Linear]:
    return [m for m in mod.modules() if isinstance(m, nn.Linear) and m.in_features == hidden]


def layer_readers(model: nn.Module) -> list[dict]:
    """Per layer: attention-side and MLP-side reader (effective norm gain g, stacked weight W)."""
    _, blocks = find_blocks(model)
    hidden = model.get_input_embeddings().weight.shape[1]
    out = []
    for blk in blocks:
        attn = getattr(blk, "self_attn", None) or getattr(blk, "linear_attn", None)
        ws = {}
        for side, norm, sub in (("attn", blk.input_layernorm, attn), ("mlp", blk.post_attention_layernorm, blk.mlp)):
            with torch.no_grad():
                g = norm(torch.ones(1, hidden)).reshape(-1).float()
            W = torch.cat([lin.weight.detach().float() for lin in _linears_reading(sub, hidden)], 0)
            ws[side] = {"g": g, "W": W}
        o = getattr(attn, "o_proj", None) or getattr(attn, "out_proj")
        ws["o_proj"] = o.weight.detach().float()
        ws["attn_name"] = next(n for n, m in model.named_modules() if m is o)
        out.append(ws)
    return out


def final_reader(model: nn.Module) -> dict:
    hidden = model.get_input_embeddings().weight.shape[1]
    norm = model.model.norm if hasattr(model, "model") and hasattr(model.model, "norm") else None
    with torch.no_grad():
        g = norm(torch.ones(1, hidden)).reshape(-1).float()
    return {"g": g, "U": model.get_output_embeddings().weight.detach().float(), "norm": norm}


# ----------------------------------------------------------------------------------------------- writes
def writes_fp32(model: nn.Module, ids: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Ground truth writes (n_writers, T, H) from hooks on the norms of an fp32 forward."""
    _, blocks = find_blocks(model)
    pre_attn, pre_mlp, outs = {}, {}, {}
    hs = []
    for l, blk in enumerate(blocks):
        hs.append(blk.input_layernorm.register_forward_pre_hook(lambda m, a, l=l: pre_attn.__setitem__(l, a[0][0].detach().float())))
        hs.append(blk.post_attention_layernorm.register_forward_pre_hook(lambda m, a, l=l: pre_mlp.__setitem__(l, a[0][0].detach().float())))
        hs.append(blk.register_forward_hook(lambda m, a, o, l=l: outs.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].detach().float())))
    try:
        with torch.no_grad():
            model(input_ids=ids.reshape(1, -1), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    n = len(blocks)
    xs = [pre_attn[0]] + [outs[l] for l in range(n)]
    d, names = [pre_attn[0]], ["E"]
    for l in range(n):
        d += [pre_mlp[l] - xs[l], xs[l + 1] - pre_mlp[l]]
        names += [f"A{l}", f"M{l}"]
    return torch.stack(d), names


def writes_trace(trace, readers: list[dict], embed: torch.Tensor) -> torch.Tensor:
    """Same writes reconstructed from a CompactTrace (residual per block + pre-W_O head vectors)."""
    xs = [embed] + [trace.dequant_resid(l).float() for l in range(trace.n_layers)]
    d = [embed]
    for l in range(trace.n_layers):
        h_in = trace.dequant_head_in(readers[l]["attn_name"])
        a = h_in.float() @ readers[l]["o_proj"].t()
        d += [a, xs[l + 1] - xs[l] - a]
    return torch.stack(d)


# ----------------------------------------------------------------------------------------------- flow metrics
def shares(d: torch.Tensor, g: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """(n_w, T, H) writes -> (n_w, T) signed direct shares of the reader (g, W); columns sum to 1."""
    x = d.sum(0)                                   # (T, H)
    read = (x * g) @ W.t()                         # (T, R)
    u = (read @ W) * g                             # (T, H) = g * W^T read
    den = (read * read).sum(-1).clamp_min(1e-12)   # (T,)
    return (d * u.unsqueeze(0)).sum(-1) / den


def dependency(d: torch.Tensor, readers: list[dict], skip_first: int = 1) -> dict:
    n = len(readers)
    D_attn = torch.full((n, d.shape[0]), float("nan"))
    D_mlp = torch.full((n, d.shape[0]), float("nan"))
    per_tok = {}
    for L in range(n):
        wa = 1 + 2 * L          # E + (A, M) of layers < L
        sa = shares(d[:wa], readers[L]["attn"]["g"], readers[L]["attn"]["W"])
        sm = shares(d[:wa + 1], readers[L]["mlp"]["g"], readers[L]["mlp"]["W"])
        D_attn[L, :wa] = sa[:, skip_first:].mean(1)
        D_mlp[L, :wa + 1] = sm[:, skip_first:].mean(1)
        per_tok[f"A{L}"], per_tok[f"M{L}"] = sa, sm
    return {"D_attn": D_attn, "D_mlp": D_mlp, "per_tok": per_tok}


def logit_shares(d: torch.Tensor, fin: dict, target: torch.Tensor) -> torch.Tensor:
    """Direct share of every writer in the logit of `target[t]` (the final top-1)."""
    w = fin["U"][target] * fin["g"]                 # (T, H)
    num = (d * w.unsqueeze(0)).sum(-1)              # (n_w, T)
    return num / num.sum(0, keepdim=True).where(num.sum(0, keepdim=True).abs() > 1e-9, torch.ones(1))


def effective_depth(per_tok: dict, final: torch.Tensor, names: list[str], thr: float) -> torch.Tensor:
    """Longest chain E -> ... -> logit using edges |share| > thr, per token. Depth counts sublayers."""
    T = final.shape[1]
    depth = {"E": torch.zeros(T)}
    for nm in names[1:]:
        s = per_tok[nm]
        best = torch.zeros(T)
        for i, src in enumerate(names[: s.shape[0]]):
            best = torch.where(s[i].abs() > thr, torch.maximum(best, depth[src] + 1), best)
        depth[nm] = best
    out = torch.zeros(T)
    for i, src in enumerate(names):
        out = torch.where(final[i].abs() > thr, torch.maximum(out, depth[src] + 1), out)
    return out


def write_lifecycle(d: torch.Tensor, names: list[str], deps: dict, final: torch.Tensor, skip_first: int = 1) -> list[dict]:
    dd = d[:, skip_first:]
    x_fin = dd.sum(0)
    rows = []
    n_layers = deps["D_attn"].shape[0]
    for i, nm in enumerate(names):
        w = dd[i]
        nn2 = (w * w).sum(-1).clamp_min(1e-12)
        surv = ((x_fin * w).sum(-1) / nn2).mean().item()
        consumed = 0.0
        for L in range(n_layers):
            for key in ("D_attn", "D_mlp"):
                v = deps[key][L, i].item()
                if v == v:
                    consumed += abs(v)
        erasers = []
        for j in range(i + 1, len(names)):
            e = (-(dd[j] * w).sum(-1) / nn2).mean().item()
            erasers.append((e, names[j]))
        erasers.sort(reverse=True)
        rows.append({
            "writer": nm, "norm": round(w.norm(dim=-1).mean().item(), 3),
            "survival": round(surv, 3), "consumed_by_layers": round(consumed, 4),
            "logit_share": round(final[i, skip_first:].mean().item(), 4),
            "top_erasers": [(n_, round(e, 3)) for e, n_ in erasers[:3] if e > 0.05],
        })
    return rows


def redundancy(d: torch.Tensor, names: list[str], top: int = 10, skip_first: int = 1) -> list:
    dn = torch.nn.functional.normalize(d[1:, skip_first:], dim=-1)  # skip E
    C = torch.einsum("atd,btd->ab", dn, dn) / dn.shape[1]
    C.fill_diagonal_(0)
    vals, idx = C.flatten().topk(2 * top)
    seen, out = set(), []
    for v, k in zip(vals.tolist(), idx.tolist()):
        a, b = divmod(k, C.shape[0])
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            out.append((names[1 + key[0]], names[1 + key[1]], round(v, 3)))
    return out[:top]


def logit_lens(model: nn.Module, d: torch.Tensor, fin: dict, sample: torch.Tensor) -> dict:
    """For sampled tokens: first block whose output already has the final top-1 as top-1 and keeps it."""
    n_layers = (d.shape[0] - 1) // 2
    cum = d.cumsum(0)[[0] + [2 + 2 * l for l in range(n_layers)]][:, sample]  # (n+1, S, H): E, x_0..x_{n-1}
    with torch.no_grad():
        top = torch.stack([(fin["norm"](c) @ fin["U"].t()).argmax(-1) for c in cum])  # (n+1, S)
    final = top[-1]
    settle = []
    for s in range(top.shape[1]):
        ok = (top[:, s] == final[s]).tolist()
        k = len(ok)
        while k > 0 and ok[k - 1]:
            k -= 1
        settle.append(k - 1)  # -1 == already at embedding
    return {"settle_block": settle, "mean": sum(settle) / len(settle)}


# ----------------------------------------------------------------------------------------------- main
def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    m = ~(a.isnan() | b.isnan())
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen35-0.8b-full")
    ap.add_argument("--out", default="benchmarks/layer_flow")
    ap.add_argument("--loop-repeats", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.05, help="edge threshold for effective depth")
    ap.add_argument("--lens-sample", type=int, default=48)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    readers, fin = layer_readers(model), final_reader(model)

    ids_normal = tok(TEXT_NORMAL, return_tensors="pt").input_ids[0]
    ids_unit = tok(LOOP_UNIT, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ids = torch.cat([ids_normal] + [ids_unit] * args.loop_repeats)
    n_norm, n_unit = len(ids_normal), len(ids_unit)
    print(f"tokens: {len(ids)} (normal {n_norm}, loop unit {n_unit} x {args.loop_repeats})", flush=True)

    d_gt, names = writes_fp32(model, ids)
    embed = d_gt[0]
    with torch.no_grad():
        target = (fin["norm"](d_gt.sum(0)) @ fin["U"].t()).argmax(-1)

    # ---- check: dependency from fp32 vs fp16 trace vs int8 trace
    check = {}
    dep_gt = dependency(d_gt, readers)
    for q in ("fp16", "int8"):
        tr = capture_compact(model, ids, resid_quant=q, head_quant=q, top_k_attn=0, capture_cache=False)
        d_q = writes_trace(tr, readers, embed)
        dep_q = dependency(d_q, readers)
        rel = ((d_q - d_gt).norm(dim=-1) / d_gt.norm(dim=-1).clamp_min(1e-6))[1:].median().item()
        check[q] = {
            "write_rel_err_median": round(rel, 4),
            "corr_D_attn": round(corr(dep_gt["D_attn"], dep_q["D_attn"]), 4),
            "corr_D_mlp": round(corr(dep_gt["D_mlp"], dep_q["D_mlp"]), 4),
            "max_abs_diff_D": round(max(
                (dep_gt["D_attn"] - dep_q["D_attn"]).nan_to_num().abs().max().item(),
                (dep_gt["D_mlp"] - dep_q["D_mlp"]).nan_to_num().abs().max().item()), 4),
        }
        print(q, check[q], flush=True)

    # ---- flow map on ground truth
    fshare = logit_shares(d_gt, fin, target)
    depth = effective_depth(dep_gt["per_tok"], fshare, names, args.thr)
    seg = {
        "normal": slice(1, n_norm),
        "loop_first": slice(n_norm, n_norm + n_unit),
        "loop_repeats": slice(n_norm + 2 * n_unit, len(ids)),
    }
    n_layers = len(readers)

    def locality(sl: slice) -> dict:
        """Share each reader takes from the previous k layers (local) vs older ones vs the embedding."""
        out = {}
        for k in (1, 2, 4):
            vals = []
            for L in range(1, n_layers):
                for side in ("A", "M"):
                    s = dep_gt["per_tok"][f"{side}{L}"][:, sl]
                    lo = max(1, 1 + 2 * (L - k))
                    vals.append(s[lo:].abs().sum(0) / s.abs().sum(0).clamp_min(1e-9))
            out[f"prev_{k}_layers"] = round(torch.stack(vals).mean().item(), 4)
        emb = torch.stack([dep_gt["per_tok"][f"A{L}"][0, sl] for L in range(n_layers)]).mean().item()
        out["embedding_share_attn"] = round(emb, 4)
        return out

    by_seg = {}
    for k, sl in seg.items():
        by_seg[k] = {
            "tokens": sl.stop - sl.start,
            "effective_depth_mean": round(depth[sl].mean().item(), 2),
            "effective_depth_max_possible": 1 + 2 * n_layers,
            "locality": locality(sl),
        }
    sample = torch.linspace(1, len(ids) - 1, args.lens_sample).long()
    lens = logit_lens(model, d_gt, fin, sample)
    lens_seg = {k: [s for i, s in zip(sample.tolist(), lens["settle_block"]) if sl.start <= i < sl.stop] for k, sl in seg.items()}
    for k, v in lens_seg.items():
        by_seg[k]["logit_lens_settle_block_mean"] = round(sum(v) / len(v), 2) if v else None

    # parallelizability: share of each reader that comes from the immediately preceding layer
    par = []
    for L in range(1, n_layers):
        prev = [1 + 2 * (L - 1), 2 + 2 * (L - 1)]
        a = dep_gt["D_attn"][L, prev].abs().sum().item()
        par.append({"layer": L, "attn_reads_prev_layer": round(a, 4)})
    par.sort(key=lambda r: r["attn_reads_prev_layer"])

    result = {
        "model": args.model, "tokens": len(ids), "edge_thr": args.thr,
        "quant_check": check,
        "segments": by_seg,
        "parallel_candidates_least_dependent_on_prev": par[:6],
        "writers": write_lifecycle(d_gt, names, dep_gt, fshare),
        "redundant_pairs": redundancy(d_gt, names),
        "seconds": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_phase1_cpu.json").write_text(json.dumps(result, indent=1))
    torch.save({"names": names, "D_attn": dep_gt["D_attn"], "D_mlp": dep_gt["D_mlp"], "logit_share": fshare.mean(1),
                "depth": depth}, out / "flow_maps_phase1.pt")
    print(json.dumps({k: result[k] for k in ("quant_check", "segments", "parallel_candidates_least_dependent_on_prev",
                                             "redundant_pairs", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
