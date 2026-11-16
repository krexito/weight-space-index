"""Feasibility: can a WHOLE model run "from the index" (extreme Mixture of Indexes)?

Idea under test
---------------
Keep a small always-resident "brain" (embeddings + attention + LayerNorms) and page MLP neurons in and out of
memory per token, choosing them with a cheap index lookup instead of keeping all MLP weights resident.
This is the "MoE taken to the extreme" version of MoI: every MLP neuron is its own tiny expert.

What it measures (GPT-2 124M or OPT-125M, CPU, WikiText-2 TEST; calibration on WikiText-2 VALIDATION -> no leakage)
-----------------------------------------------------------------------------------------------------
For an MLP budget k (fraction of the 3072 neurons per layer that may fire on each token), perplexity under:

  * oracle      : top-k neurons by |activation - mean| at that exact position. Uses the real activations, so it is an
                  UPPER BOUND: no router can beat it. If quality collapses here, the idea is dead at that k.
  * hot         : one fixed set (top-k by activation variance on calibration). PowerInfer-style "hot neurons";
                  needs no router and moves no bytes, but is just static pruning.
  * token_index : MoI-style router: a lookup table token_id -> neurons (top-k by mean |activation| when that token
                  is the current input: largest |E[act | token] - mean|, measured on calibration). Unseen/rare tokens fall back to `hot`.
                  O(1) per token, uses nothing but the current token id.
  * hot+token   : half the budget from `hot` (resident), half from `token_index` (paged).
  * svdR / pcaR : contextual rank-R linear routers on the current hidden state (weights only / calibration PCA).
  * trainedR    : Deja-Vu-style learned rank-R router (initialized from the weight SVD, trained on calibration).

Resources per policy/budget:
  * MLP neurons that must be resident / paged per token.
  * Bytes paged per token assuming the previous token's neurons stay cached (|S_t minus S_{t-1}|), fp32
    (768 in + 1 bias + 768 out floats per neuron), and the transfer time at PCIe (16 GB/s) and NVMe (3 GB/s).
  * Router table size.

Context for reading the numbers: in GPT-2 124M the MLPs are only ~45% of the parameters (embeddings ~31%,
attention ~23%), so even perfect MLP paging can save at most ~45% of memory here. In larger models the MLP
share is ~2/3 and embeddings are relatively smaller.

Usage:
    python scripts/experiment_moi_full_model_paging.py
    python scripts/experiment_moi_full_model_paging.py --eval-tokens 40000 --budgets 0.5,0.25,0.1,0.05
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def load_split(name: str) -> str:
    p = ROOT / "work" / "wikitext" / f"{name}.parquet"
    if not p.exists():
        raise SystemExit(
            f"Missing {p}. Download it with:\n  curl -sSL -o {p} "
            f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/"
            f"{name}-00000-of-00001.parquet"
        )
    return "\n".join(pd.read_parquet(p)["text"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default="gpt2", help="Folder under models/: gpt2 (GELU) or opt-125m (ReLU).")
    ap.add_argument("--eval-tokens", type=int, default=20_000)
    ap.add_argument("--calib-tokens", type=int, default=40_000)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--ablation", choices=["mean", "zero"], default="mean",
                    help="Value of NOT-loaded neurons: calibrated mean (default, folds into bias) or 0. GELU is"
                         " not ReLU: zeroing thousands of slightly-negative units shifts the residual stream.")
    ap.add_argument("--policies", type=str, default="oracle,hot,token_index,hot+token,svd64,pca64",
                    help="oracle, hot, token_index, hot+token, svdR (weights-only rank-R router), pcaR (rank-R router"
                         " on the principal axes of calibration MLP inputs).")
    ap.add_argument("--train-epochs", type=int, default=4, help="Epochs over calibration for trainedR routers.")
    ap.add_argument("--train-lr", type=float, default=1e-3)
    ap.add_argument("--table-vocab", type=int, default=1500, help="Most frequent calibration tokens with own entry.")
    ap.add_argument("--budgets", type=str, default="0.5,0.25,0.1,0.05,0.02")
    ap.add_argument("--out", type=str, default=str(ROOT / "benchmarks" / "moi_full_model_paging" / "results.json"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    t_start = time.perf_counter()
    mdir = str(ROOT / "models" / args.model)
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).eval()
    mtype = model.config.model_type
    # ---- architecture adapter: MLP input module (pre-hook), activation module (hook), W_in [d, U], b_in ----
    if mtype == "gpt2":
        mlps = [b.mlp for b in model.transformer.h]
        in_mods, act_mods = mlps, [m.act for m in mlps]
        W_in = [m.c_fc.weight.data for m in mlps]
        b_in = [m.c_fc.bias.data for m in mlps]
        act_fn = lambda z: F.gelu(z, approximate="tanh")  # noqa: E731
        prefix: list[int] = []
        mlp_keys = (".mlp.",)
    elif mtype == "opt":
        layers = model.model.decoder.layers
        for lyr in layers:  # one module instance per layer so each hook knows its layer
            lyr.activation_fn = torch.nn.ReLU()
        in_mods, act_mods = [lyr.fc1 for lyr in layers], [lyr.activation_fn for lyr in layers]
        W_in = [lyr.fc1.weight.data.T for lyr in layers]
        b_in = [lyr.fc1.bias.data for lyr in layers]
        act_fn = F.relu
        prefix = [tok.bos_token_id]
        mlp_keys = (".fc1.", ".fc2.")
    else:
        raise SystemExit(f"Unsupported model_type {mtype!r}")
    n_layers = len(W_in)
    d_model = int(W_in[0].shape[0])
    n_units = int(W_in[0].shape[1])
    NEURON_BYTES = (2 * d_model + 1) * 4
    budgets = [float(b) for b in args.budgets.split(",")]

    eval_ids = tok(load_split("test"), add_special_tokens=False).input_ids[: args.eval_tokens]
    calib_ids = tok(load_split("validation"), add_special_tokens=False).input_ids[: args.calib_tokens]
    chunks = lambda ids: [ids[i : i + args.chunk] for i in range(0, len(ids) - 8, args.chunk)]  # noqa: E731

    # ---- hooks: capture (calibration) or mask (evaluation) the post-GELU activations ----
    ctx: dict = {"mode": "off", "ids": None, "layer_acts": {}, "policy": None, "k": 0, "sets": {}, "x": {}}

    def make_hook(layer: int):
        def hook(module, inp, out):
            a = out.reshape(-1, out.shape[-1])  # [T, n_units] (OPT already flattens tokens)
            if ctx["mode"] == "capture":
                ctx["layer_acts"][layer] = a.clone()
                return out
            if ctx["mode"] != "mask":
                return out
            k = ctx["k"]
            pol = ctx["policy"]
            if pol == "oracle":
                dev = (a - mu[layer]).abs() if args.ablation == "mean" else a.abs()
                idx = dev.topk(k, dim=-1).indices
                mask = torch.zeros_like(a, dtype=torch.bool).scatter_(1, idx, True)
            elif pol == "hot":
                mask = hot_masks[(layer, k)].expand(a.shape[0], -1)
            elif pol == "token_index":
                mask = token_masks(layer, k, ctx["ids"])
            elif pol.startswith("trained"):
                # Deja-Vu-style learned router: rank-r predictor (initialized from the weight SVD) fine-tuned on
                # calibration to predict which neurons will be active. Loads the k highest-scoring neurons.
                A, B, c = trained[(layer, pol)]
                idx = (ctx["x"][layer] @ A @ B + c).topk(k, dim=-1).indices
                mask = torch.zeros_like(a, dtype=torch.bool).scatter_(1, idx, True)
            elif pol.startswith(("svd", "pca")):
                # Contextual router: predict this position's pre-activations from the CURRENT hidden state with a
                # rank-r approximation of c_fc, then load the k neurons predicted to deviate most from their mean.
                A, B, c = routers[(layer, pol)]
                x = ctx["x"][layer]
                pred = act_fn((x - c[0]) @ A @ B + c[1])
                idx = (pred - mu[layer]).abs().topk(k, dim=-1).indices
                mask = torch.zeros_like(a, dtype=torch.bool).scatter_(1, idx, True)
            else:  # hot+token
                mask = hot_masks[(layer, k // 2)].expand(a.shape[0], -1) | token_masks(layer, k - k // 2, ctx["ids"])
            ctx["sets"].setdefault(layer, []).append(mask)
            dev_all = (a - mu[layer]).abs() if args.ablation == "mean" else a.abs()
            ctx["mass_sel"] += float((dev_all * mask).sum())
            ctx["mass_tot"] += float(dev_all.sum())
            if args.ablation == "mean":
                # neurons that are not loaded contribute their calibrated mean: a constant that folds into
                # c_proj.bias, so it costs no paging and no extra memory.
                return torch.where(mask, a, mu[layer].expand_as(a)).reshape(out.shape)
            else:
                return (a * mask).reshape(out.shape)
            return out

        return hook

    def make_pre_hook(layer: int):
        def pre_hook(module, inp):
            ctx["x"][layer] = inp[0].reshape(-1, inp[0].shape[-1])  # normalized MLP input, [T, d]

        return pre_hook

    for l in range(n_layers):
        act_mods[l].register_forward_hook(make_hook(l))
        in_mods[l].register_forward_pre_hook(make_pre_hook(l))

    # ---- calibration on VALIDATION ----
    t0 = time.perf_counter()
    freq = torch.bincount(torch.tensor(calib_ids), minlength=model.config.vocab_size)
    table_tokens = freq.topk(args.table_vocab).indices
    row_of = torch.full((model.config.vocab_size,), -1, dtype=torch.long)
    row_of[table_tokens] = torch.arange(args.table_vocab)
    sums = torch.zeros(n_layers, args.table_vocab, n_units)
    counts = torch.zeros(args.table_vocab)
    mean_abs = torch.zeros(n_layers, n_units)
    mu = torch.zeros(n_layers, n_units)
    sq = torch.zeros(n_layers, n_units)
    xsum = torch.zeros(n_layers, d_model)
    xx = torch.zeros(n_layers, d_model, d_model)
    n_pos = 0
    ctx["mode"] = "capture"
    n_zero = 0.0
    for ch in chunks(calib_ids):
        ids_t = torch.tensor(prefix + ch)
        model(input_ids=ids_t[None])
        rows = row_of[ids_t]
        keep = rows >= 0
        counts.index_add_(0, rows[keep], torch.ones(int(keep.sum())))
        for l in range(n_layers):
            a = ctx["layer_acts"][l]
            n_zero += float((a == 0).sum())
            mean_abs[l] += a.abs().sum(0)
            mu[l] += a.sum(0)
            sq[l] += (a * a).sum(0)
            sums[l].index_add_(0, rows[keep], a[keep])
            x = ctx["x"][l]
            xsum[l] += x.sum(0)
            xx[l] += x.T @ x
        n_pos += len(ids_t)
    mean_abs /= n_pos
    mu /= n_pos
    var = sq / n_pos - mu * mu
    token_means = sums.div_(counts.clamp_min(1)[None, :, None])  # in place: avoid a second 220 MB tensor
    coverage = float((row_of[torch.tensor(eval_ids)] >= 0).float().mean())
    print(f"Calibrated on {n_pos} tokens in {time.perf_counter() - t0:.1f}s | token table covers "
          f"{coverage:.1%} of eval tokens")

    ks = sorted({max(1, round(b * n_units)) for b in budgets} | {max(1, round(b * n_units)) // 2 for b in budgets}
                | {max(1, round(b * n_units)) - max(1, round(b * n_units)) // 2 for b in budgets})
    ks = [k for k in ks if k > 0]
    hot_masks: dict[tuple[int, int], torch.Tensor] = {}
    for l in range(n_layers):
        order = (var[l] if args.ablation == "mean" else mean_abs[l]).argsort(descending=True)
        for k in ks:
            m = torch.zeros(1, n_units, dtype=torch.bool)
            m[0, order[:k]] = True
            hot_masks[(l, k)] = m
    tok_mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    # ---- contextual routers: rank-r predictors of c_fc pre-activations ----
    #   svdR : weights only  -> W_fc ~ U_r S_r V_r^T               (no data at all)
    #   pcaR : calibration    -> x ~ m + P_r P_r^T (x - m), P_r = top-r principal axes of the MLP inputs
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    routers: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, tuple]] = {}
    router_bytes: dict[str, int] = {}
    for l in range(n_layers):
        W = W_in[l]  # [d, n_units]
        bias = b_in[l]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        m = xsum[l] / n_pos
        C = xx[l] / n_pos - torch.outer(m, m)
        evals, evecs = torch.linalg.eigh(C)
        evecs = evecs.flip(1)
        for pol in policies:
            if not pol.startswith(("svd", "pca")):
                continue
            r = int(pol[3:])
            if pol.startswith("svd"):
                A, B, c = U[:, :r].contiguous(), (S[:r, None] * Vh[:r]).contiguous(), (torch.zeros(d_model), bias)
            else:
                P = evecs[:, :r].contiguous()
                A, B, c = P, (P.T @ W).contiguous(), (m, m @ W + bias)
            routers[(l, pol)] = (A, B, c)
            router_bytes[pol] = router_bytes.get(pol, 0) + (A.numel() + B.numel() + 2 * n_units + d_model) * 4
    del xx

    # ---- trained routers (Deja Vu style), streamed over the calibration set ----
    trained: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    train_pols = [pol for pol in policies if pol.startswith("trained")]
    if train_pols:
        t_tr = time.perf_counter()
        params: dict[tuple[int, str], list[torch.Tensor]] = {}
        for l in range(n_layers):
            U, S, Vh = torch.linalg.svd(W_in[l], full_matrices=False)
            for pol in train_pols:
                r = int(pol[len("trained"):])
                ps = [U[:, :r].clone().requires_grad_(True), (S[:r, None] * Vh[:r]).clone().requires_grad_(True),
                      b_in[l].clone().requires_grad_(True)]
                params[(l, pol)] = ps
        opts = {key: torch.optim.Adam(ps, lr=args.train_lr) for key, ps in params.items()}
        ctx["mode"] = "capture"
        calib_chunks = chunks(calib_ids)
        last_loss: dict[str, float] = {}
        for ep in range(args.train_epochs):
            g = torch.Generator().manual_seed(ep)
            for ci in torch.randperm(len(calib_chunks), generator=g).tolist():
                model(input_ids=torch.tensor(prefix + calib_chunks[ci])[None])
                for l in range(n_layers):
                    x = ctx["x"][l].detach()
                    a = ctx["layer_acts"][l]
                    if mtype == "opt":  # ReLU: predict "neuron is non-zero"
                        tgt = (a > 0).float()
                        dev = a.abs()
                    else:  # GELU: predict "neuron is in the top-10% |deviation from mean|" at this position
                        dev = (a - mu[l]).abs()
                        tgt = (dev >= dev.quantile(0.9, dim=-1, keepdim=True)).float()
                    w = 1.0 + dev / (mean_abs[l] + 1e-6)  # emphasize large activations
                    for pol in train_pols:
                        A, B, c = params[(l, pol)]
                        with torch.enable_grad():
                            loss = F.binary_cross_entropy_with_logits(x @ A @ B + c, tgt, weight=w)
                            opts[(l, pol)].zero_grad()
                            loss.backward()
                            opts[(l, pol)].step()
                        last_loss[pol] = float(loss)
            print(f"  router training epoch {ep + 1}/{args.train_epochs}: last-layer loss "
                  + ", ".join(f"{k}={v:.4f}" for k, v in last_loss.items()))
        for key, (A, B, c) in params.items():
            trained[key] = (A.detach(), B.detach(), c.detach())
            pol = key[1]
            router_bytes[pol] = router_bytes.get(pol, 0) + (A.numel() + B.numel() + c.numel()) * 4
        print(f"Trained {len(train_pols)} router(s) x {n_layers} layers in {time.perf_counter() - t_tr:.1f}s")

    def token_masks(layer: int, k: int, ids: torch.Tensor) -> torch.Tensor:
        key = (layer, k)
        if key not in tok_mask_cache:
            score = (token_means[layer] - mu[layer]).abs() if args.ablation == "mean" else token_means[layer].abs()
            idx = score.topk(k, dim=-1).indices
            tok_mask_cache[key] = torch.zeros(args.table_vocab, n_units, dtype=torch.bool).scatter_(1, idx, True)
        rows = row_of[ids]
        m = hot_masks[(layer, k)].expand(len(ids), -1).clone()
        known = rows >= 0
        m[known] = tok_mask_cache[key][rows[known]]
        return m

    # ---- evaluation ----
    eval_chunks = chunks(eval_ids)

    def ppl_run(mode: str, policy: str | None = None, k: int = 0) -> dict:
        ctx.update(mode=mode, policy=policy, k=k, sets={}, mass_sel=0.0, mass_tot=0.0)
        nll = 0.0
        ntok = 0
        paged_neurons = 0.0
        active = 0.0
        t0 = time.perf_counter()
        for ch in eval_chunks:
            ids_t = torch.tensor(prefix + ch)
            ctx["ids"] = ids_t
            ctx["sets"] = {}
            lp = F.log_softmax(model(input_ids=ids_t[None]).logits[0, :-1].float(), dim=-1)
            nll -= float(lp.gather(1, ids_t[1:, None]).sum())
            ntok += len(ids_t) - 1
            if mode == "mask":
                for l, ms in ctx["sets"].items():
                    m = ms[0]
                    active += float(m.sum())
                    # neurons that are new w.r.t. the previous token (previous token's set stays cached)
                    paged_neurons += float((m[1:] & ~m[:-1]).sum()) + float(m[0].sum())
        n_pos = sum(len(prefix) + len(c) for c in eval_chunks)
        res = {"ppl": math.exp(nll / ntok), "wall_s": time.perf_counter() - t0}
        if mode == "mask":
            res["active_neurons_per_token"] = active / n_pos
            res["active_fraction"] = active / n_pos / (n_layers * n_units)
            res["paged_neurons_per_token"] = paged_neurons / n_pos
            res["paged_bytes_per_token"] = paged_neurons / n_pos * NEURON_BYTES
            res["activation_mass_recall"] = ctx["mass_sel"] / max(ctx["mass_tot"], 1e-12)
        return res

    full = ppl_run("off")
    print(f"[full model] ppl={full['ppl']:.3f}")
    results: dict = {"full": full, "policies": {}}
    for b in budgets:
        k = max(1, round(b * n_units))
        for pol in policies:
            r = ppl_run("mask", pol, k)
            if pol == "hot":
                r["paged_bytes_per_token"] = 0.0  # fixed resident set, nothing moves
                r["paged_neurons_per_token"] = 0.0
            if pol == "hot+token":
                hot_part = n_layers * (k // 2)
                r["resident_neurons"] = hot_part
            r["ppl_ratio_vs_full"] = r["ppl"] / full["ppl"]
            r["pcie16_us_per_token"] = r["paged_bytes_per_token"] / 16e9 * 1e6
            r["nvme3_us_per_token"] = r["paged_bytes_per_token"] / 3e9 * 1e6
            results["policies"][f"{pol}@{b}"] = {"policy": pol, "budget": b, "k_per_layer": k, **r}
            print(f"  k={b:>5.0%} {pol:12s} ppl={r['ppl']:9.2f} (x{r['ppl_ratio_vs_full']:.2f}) "
                  f"recall={r['activation_mass_recall']:.1%} active={r['active_fraction']:.1%} paged/token={r['paged_bytes_per_token'] / 1e6:.2f} MB")
        tok_mask_cache.clear()

    p = dict(model.named_parameters())
    mlp_bytes = sum(v.numel() * 4 for n, v in p.items() if any(k in n for k in mlp_keys))
    total_bytes = sum(v.numel() * 4 for v in p.values())
    summary = {
        "model": args.model,
        "natural_exact_zero_fraction": n_zero / (n_pos * n_layers * n_units),
        "eval_tokens": len(eval_ids),
        "calib_tokens": n_pos,
        "ablation": args.ablation,
        "token_table_vocab": args.table_vocab,
        "token_table_eval_coverage": coverage,
        "token_table_bytes_at_max_budget_int16": int(args.table_vocab * n_layers * round(max(budgets) * n_units) * 2),
        "param_bytes_total": total_bytes,
        "param_bytes_mlp": mlp_bytes,
        "mlp_share_of_params": mlp_bytes / total_bytes,
        "neuron_bytes_fp32": NEURON_BYTES,
        "contextual_router_bytes_total": router_bytes,
        "contextual_router_macs_per_token_per_layer": {
            pol: int(pol.lstrip("abcdefghijklmnopqrstuvwxyz")) * (d_model + n_units) for pol in router_bytes},
        "mlp_macs_per_token_per_layer": 2 * d_model * n_units,
        "results": results,
        "wall_s": time.perf_counter() - t_start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {out} ({summary['wall_s']:.1f}s)")


if __name__ == "__main__":
    main()
