"""EXPERIMENTAL - can a 1-backward gradient score rank loop-fix patches better than `screen_patch_candidates`?

Same setup as `experiment_screen_proxy_validate.py` (it imports its helpers): generate GSM8K train sequences, capture
CompactTraces, run `discover_compact_circuit`, rebuild the 60 configs of `results_blind_colab_l4.json`
(layers {L14, L13-14, L10-13-14} x neurons {16, 64, 256, all} x alpha {2, 4, 6, 8, 12}) and screen them with
`screen_patch_candidates` (1 patched forward per config). Ground truth = round-1 accuracy / loop_killed per config.

New: a first-order (gradient) version of the same two numbers, from ONE backward per screening sequence for all 60:
  an edit on down_proj of layer l changes that layer's write at token t by
      delta_t = -alpha * u * ((m * u^T W) . h_t)                (h = down_proj input)
  so   d(mean target log-prob) ~= sum_l sum_t  g_t . delta_t   with g = grad of the target log-prob w.r.t. x_l
  grad_supp   = -(that)        (positive = loop tokens become less likely, like `target_suppression`)
  grad_dnll   = same with the healthy-text NLL gradient           (like `delta_healthy_nll`)
  grad_pareto = grad_supp / max(|grad_dnll|, 0.01)                (like `pareto_score`)
First-order scores are linear in alpha, so they cannot pick alpha: grad_pareto is the same for all 5 alphas of a site.
That is why "within-alpha" Spearman (12 site/neuron configs per alpha, averaged over the 5 alphas) is reported too.

Run:  python scripts/experiment_gradient_screen.py --dtype fp16 --out results_gradient_screen.json
Tiny smoke test (CPU):  CUDA_VISIBLE_DEVICES="" python scripts/experiment_gradient_screen.py --model models/qwen35-0.8b-full \
        --n-train 4 --max-new-tokens 160 --max-traces 4 --n-screen-seqs 2 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_screen_proxy_validate import _batched_generate, _local_or_hub, _spearman, need  # noqa: E402
from wsi.capture import find_blocks, find_mlps  # noqa: E402
from wsi.moipatch import Edit, MoIPatch, _dtype_name, _sha256, module_suffix  # noqa: E402
from wsi.tracer import capture_compact, discover_compact_circuit, screen_patch_candidates  # noqa: E402

GRID_LAYERS = (([14], "L14"), ([13, 14], "L13-14"), ([10, 13, 14], "L10-13-14"))
GRID_N = ("16", "64", "256", "all")
GRID_A = (2, 4, 6, 8, 12)


def build_grid(model, traces, disc, w_downs, meta_by_l):
    """The 60 configs, exactly as in experiment_screen_proxy_validate.py."""
    full_neuron_rank = {}
    for l in (10, 13, 14):
        u_l = disc["directions"][l]
        hx_list, lx_list = [], []
        for idx_t, (r, ct, _) in enumerate(traces):
            if idx_t % 2 != 0:
                continue
            T, p = ct.seq_len, r["prompt_len"]
            ons = min(p + r["onset_resp"], T - 1) if (r["loop"] and r["onset_resp"] is not None) else T
            h_idx = torch.arange(min(p, T), max(min(p, T), min(ons, p + 96, T)))
            l_idx = torch.arange(max(ons, T - 192), T) if r["loop"] else torch.empty(0, dtype=torch.long)
            idx_m, val_m = ct.dequant_mlp_topk(l)
            wn = w_downs[l].norm(dim=0).clamp_min(1e-12)
            d_inter = w_downs[l].shape[1]
            if len(h_idx):
                xh = torch.zeros((len(h_idx), d_inter), dtype=torch.float32)
                xh.scatter_(1, idx_m[h_idx], val_m[h_idx] / wn[idx_m[h_idx]])
                hx_list.append(xh.mean(dim=0, keepdim=True))
            if len(l_idx):
                xl = torch.zeros((len(l_idx), d_inter), dtype=torch.float32)
                xl.scatter_(1, idx_m[l_idx], val_m[l_idx] / wn[idx_m[l_idx]])
                lx_list.append(xl.mean(dim=0, keepdim=True))
        dx = torch.cat(lx_list, dim=0).mean(dim=0) - torch.cat(hx_list, dim=0).mean(dim=0)
        full_neuron_rank[l] = torch.argsort(dx * (u_l @ w_downs[l]), descending=True)

    grid = {}
    for layers_pick, pref in GRID_LAYERS:
        for n_lbl in GRID_N:
            for a_val in GRID_A:
                name = f"{pref}_n{n_lbl}_a{a_val}"
                edits = []
                for l in layers_pick:
                    ml = meta_by_l[l]
                    idx_cols = None if n_lbl == "all" else full_neuron_rank[l][: int(n_lbl)].sort().values.long()
                    edits.append(Edit(module=ml["module"], u=disc["directions"][l].double(), alpha=float(a_val),
                                      idx=idx_cols, base_sha256=ml["base_sha256"], base_dtype=ml["base_dtype"],
                                      shape=ml["shape"]))
                grid[name] = MoIPatch(edits=edits, meta={"name": name})
    return grid


def grad_pass(model, blocks, seqs, spans, layers, dev, kind):
    """One backward per sequence. Returns per layer the token-wise (g_t . u)-ready pieces: grads G (T,H), MLP inputs Hh (T,U).
    kind='target': objective = mean log-prob over the target spans (all seqs pooled, like the screen).
    kind='healthy': objective = mean NLL over all healthy tokens."""
    n_tok = 0
    for i, ids in enumerate(seqs):
        T = len(ids)
        if kind == "target":
            s, e = spans[i]
            lo = max(0, min(T - 2, int(s) - 1)); hi = max(lo + 1, min(T - 1, int(e) - 1))
            n_tok += hi - lo
        else:
            n_tok += T - 1
    out = {l: [] for l in layers}
    for i, ids in enumerate(seqs):
        st = {"x": {}, "h": {}}
        hs = []
        for l in layers:
            def post(mod, a, o, l=l):
                t = o[0] if isinstance(o, tuple) else o
                t.retain_grad()
                st["x"][l] = t
            hs.append(blocks[l].register_forward_hook(post))
            hs.append(blocks[l].mlp.down_proj.register_forward_pre_hook(
                lambda m, a, l=l: st["h"].__setitem__(l, a[0].detach()[0].float().cpu())))
        x_ids = torch.tensor([ids], device=dev)
        emb = model.get_input_embeddings()(x_ids).detach().requires_grad_(True)
        try:
            T = len(ids)
            if kind == "target":
                s, e = spans[i]
                lo = max(0, min(T - 2, int(s) - 1)); hi = max(lo + 1, min(T - 1, int(e) - 1))
            else:
                lo, hi = 0, T - 1
            logits = model(inputs_embeds=emb, use_cache=False, logits_to_keep=T - lo).logits[0, -(T - lo):][: hi - lo].float()
            lp = F.log_softmax(logits, -1).gather(1, x_ids[0, lo + 1: hi + 1, None])[:, 0]
            obj = lp.sum() / n_tok if kind == "target" else -lp.sum() / n_tok
            obj.backward()
        finally:
            for h in hs:
                h.remove()
        for l in layers:
            out[l].append((st["x"][l].grad[0].float().cpu(), st["h"][l]))
        del st, emb, logits, lp, obj
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def first_order(patch, pieces, w_downs, layer_of):
    """sum over edits, seqs, tokens of g_t . delta_t."""
    tot = 0.0
    for e in patch.edits:
        l = layer_of[e.module]
        u = e.u.float()
        W = w_downs[l]
        uw = u @ W                                         # (U,)
        if e.idx is not None:
            m = torch.zeros_like(uw); m[e.idx.long()] = 1.0
            uw = uw * m
        for G, Hh in pieces[l]:
            coef = Hh @ uw                                 # (T,)
            tot += float(-e.alpha * ((G @ u) * coef).sum())
    return tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--blind-json", default=str(ROOT / "benchmarks" / "discover_loop_fix" / "results_blind_colab_l4.json"))
    ap.add_argument("--data-dir", default=str(ROOT / "work"))
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-traces", type=int, default=12)
    ap.add_argument("--n-screen-seqs", type=int, default=6)
    ap.add_argument("--max-nll-delta", type=float, default=0.05)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: allow sequences without loops")
    ap.add_argument("--out", default="results_gradient_screen.json")
    args = ap.parse_args()

    t_all = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        args.dtype, torch.bfloat16 if dev.type == "cuda" else torch.float32)
    mdir = _local_or_hub(args.model)
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, blocks = find_blocks(model)
    gtr = pd.read_parquet(need(Path(args.data_dir), "gsm8k/train.parquet")).iloc[: args.n_train]

    # 1. generate + capture
    t0 = time.perf_counter()
    rows = _batched_generate(model, tok, list(gtr["question"]), dev, args.max_new_tokens)
    gen_s = round(time.perf_counter() - t0, 1)
    if args.smoke:  # fake loop onsets so the pipeline runs end to end on short CPU generations
        for r in rows:
            if not r["loop"] and r["gen_len"] > 40:
                r["loop"], r["onset_resp"] = True, r["gen_len"] // 2
    loop_rows = [r for r in rows if r["loop"]]
    clean_rows = [r for r in rows if not r["loop"]]
    print(f"Generated {len(rows)} seqs in {gen_s}s: loop={len(loop_rows)} clean={len(clean_rows)}", flush=True)
    if len(loop_rows) < 2:
        raise SystemExit("Need >= 2 looping sequences; raise --n-train or --max-new-tokens.")
    chosen = loop_rows[: args.max_traces]
    if len(chosen) < args.max_traces:
        chosen += clean_rows[: max(0, args.max_traces - len(chosen))]
    traces = [(r, capture_compact(model, r["full_ids"]), None) for r in chosen]

    # 2. discovery + the 60 configs
    disc = discover_compact_circuit(traces, model=model, alpha=6.0, top_k_neurons=64, top_k_heads=2)
    mods = dict(model.named_modules())
    mlps = find_mlps(model)
    w_downs, meta_by_l, layer_of = {}, {}, {}
    for l, g in enumerate(mlps):
        w = mods[g.down].weight.detach()
        w_downs[l] = w.float().cpu()
        meta_by_l[l] = {"module": module_suffix(g.down), "shape": tuple(w.shape),
                        "base_dtype": _dtype_name(w.dtype), "base_sha256": _sha256(w)}
        layer_of[meta_by_l[l]["module"]] = l
    grid = build_grid(model, traces, disc, w_downs, meta_by_l)

    half_b_rows = [r for i, (r, _, _) in enumerate(traces) if i % 2 == 1] or [traces[-1][0]]
    healthy_seqs, target_seqs, target_spans = [], [], []
    for r in half_b_rows[: args.n_screen_seqs]:
        p = r["prompt_len"]
        if r["loop"] and r["onset_resp"] is not None:
            ons = min(p + r["onset_resp"], len(r["full_ids"]) - 1)
            h_end = max(p + 16, min(ons, p + 128))
            t_end = min(len(r["full_ids"]), ons + 192)
            healthy_seqs.append(r["full_ids"][:h_end])
            target_seqs.append(r["full_ids"][:t_end])
            target_spans.append((ons, t_end))
    print(f"screening on {len(target_seqs)} held-out looping seqs", flush=True)

    # 3a. existing screen: 1 patched forward per config
    t1 = time.perf_counter()
    screened = screen_patch_candidates(model, grid, healthy_seqs=healthy_seqs, target_seqs=target_seqs,
                                       target_spans=target_spans, max_nll_delta=args.max_nll_delta)
    screen_s = round(time.perf_counter() - t1, 2)
    by_name = {r["name"]: r for r in screened}

    # 3b. gradient screen: 1 backward per sequence for all configs (fp32 for clean gradients)
    t2 = time.perf_counter()
    model.float()
    layers = sorted({l for ls, _ in GRID_LAYERS for l in ls})
    tgt = grad_pass(model, blocks, target_seqs, target_spans, layers, dev, "target")
    hlt = grad_pass(model, blocks, healthy_seqs, None, layers, dev, "healthy")
    model.to(dtype)
    for name, patch in grid.items():
        r = by_name[name]
        r["grad_supp"] = round(-first_order(patch, tgt, w_downs, layer_of), 6)
        r["grad_dnll"] = round(first_order(patch, hlt, w_downs, layer_of), 6)
        r["grad_pareto"] = round(r["grad_supp"] / max(abs(r["grad_dnll"]), 0.01), 6)
    grad_s = round(time.perf_counter() - t2, 2)

    # 4. compare with ground truth
    gt_all = json.loads(Path(args.blind_json).read_text(encoding="utf-8"))["search"]["round1"]
    names = list(grid)
    acc = [gt_all[n]["accuracy"] for n in names]
    loop = [gt_all[n]["loop_killed"] for n in names]
    keys = ["target_suppression", "pareto_score", "delta_healthy_nll", "grad_supp", "grad_pareto", "grad_dnll"]
    col = {k: [by_name[n][k] for n in names] for k in keys}

    def within_alpha(k, truth):
        vals = []
        for a in GRID_A:
            ix = [i for i, n in enumerate(names) if n.endswith(f"_a{a}")]
            vals.append(_spearman([col[k][i] for i in ix], [truth[i] for i in ix]))
        return round(sum(vals) / len(vals), 4)

    def rank_of(k, name="L13-14_n16_a12"):
        order = sorted(names, key=lambda n: -by_name[n][k])
        return order.index(name) + 1

    result = {
        "model": args.model, "dtype": str(dtype), "device": str(dev),
        "n_train": len(rows), "loop_seqs": len(loop_rows), "screen_seqs": len(target_seqs),
        "timing_s": {"generate": gen_s, "screen_60_forwards": screen_s, "gradient_60_backward": grad_s,
                     "total": round(time.perf_counter() - t_all, 1)},
        "spearman_vs_r1_accuracy": {k: _spearman(col[k], acc) for k in keys},
        "spearman_vs_r1_loop_killed": {k: _spearman(col[k], loop) for k in keys},
        "within_alpha_spearman_vs_accuracy": {k: within_alpha(k, acc) for k in keys},
        "first_order_check": {
            "spearman_grad_supp_vs_real_supp": _spearman(col["grad_supp"], col["target_suppression"]),
            "spearman_grad_dnll_vs_real_dnll": _spearman(col["grad_dnll"], col["delta_healthy_nll"]),
        },
        "winner_L13_14_n16_a12_rank": {k: rank_of(k) for k in ("pareto_score", "target_suppression", "grad_pareto", "grad_supp")},
        "configs": [{"name": n, "gt_acc": gt_all[n]["accuracy"], "gt_loop_killed": gt_all[n]["loop_killed"],
                     **{k: by_name[n][k] for k in keys}} for n in names],
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "configs"}, indent=2), flush=True)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
