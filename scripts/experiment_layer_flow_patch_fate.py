"""EXPERIMENTAL - does the layer-to-layer flow tell a working loop patch apart from random ones?

Patches: the real p50c loop fix (layers 15+19 down_proj, 64 columns each, alpha 6; GSM8K acc 0.095 -> 0.64)
and controls with the SAME layers, columns and alpha but another direction u:
  iso    random isotropic u
  resid  u = random mix of real residual states at that layer (looks like the data)
  cols   u = what those 64 columns write for a random token (lies in the patch's own output space)

Real effect (ground truth, one patched forward each): change in log-prob of the model's own next tokens,
in the loop region (should DROP for a loop fix) and in the reasoning before the loop (should stay).

Flow metrics (no patched forward; unpatched states + weights only). The patch's write at layer l, token t, is
    delta_t = -alpha * u * ((m * u^T W) . h_t)       (h = down_proj input)
  F_size     ||delta|| / ||x_l||
  F_read     sum over later readers (attention / MLP of every later layer) of |direct share of delta in the read|
  F_logit    |direct share of delta in the logit of the next token|
  G_loop     first-order change of the loop-region log-prob: grad_{x_l} logp_loop . delta  (one backward, all patches)
  G_select   G_loop - G_reasoning (per token)
Score: Spearman between each metric and the real loop drop across all patches, and p50c's rank.

Run (CPU):  CUDA_VISIBLE_DEVICES="" python scripts/experiment_layer_flow_patch_fate.py
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_layer_flow import final_reader, layer_readers  # noqa: E402
from experiment_layer_flow_capture_check import loop_onset  # noqa: E402
from experiment_layer_parallel import spearman  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.moipatch import MoIPatch, apply_edit  # noqa: E402


def module_by_suffix(model, suffix):
    return next(m for n, m in model.named_modules() if n.endswith(suffix))


def run_states(model, blocks, ids, edit_layers, regions):
    """fp32 forward with grads on block outputs; returns x_l, down_proj inputs h, grads of loop/reasoning log-prob."""
    st = {"out": {}, "h": {}}
    hs = []
    for l, blk in enumerate(blocks):
        def post(mod, a, o, l=l):
            t = o[0] if isinstance(o, tuple) else o
            t.retain_grad()
            st["out"][l] = t
        hs.append(blk.register_forward_hook(post))
    for l in edit_layers:
        hs.append(blocks[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, a, l=l: st["h"].__setitem__(l, a[0].detach()[0].float())))
    emb = model.get_input_embeddings()(ids.reshape(1, -1)).detach().requires_grad_(True)
    try:
        logits = model(inputs_embeds=emb, use_cache=False).logits[0, :-1].float()
    finally:
        for h in hs:
            h.remove()
    lp = torch.log_softmax(logits, -1).gather(1, ids[1:, None])[:, 0]  # lp[t] = log p(ids[t+1] | <= t)
    grads = {}
    for rn, sl in regions.items():
        for t_ in st["out"].values():
            t_.grad = None
        lp[sl.start - 1: sl.stop - 1].sum().backward(retain_graph=True)
        grads[rn] = {l: st["out"][l].grad[0].float().clone() for l in edit_layers}
    x = {l: st["out"][l].detach()[0].float() for l in range(len(blocks))}
    return x, st["h"], grads, lp.detach()


def patched_logprob(model, ids, edits):
    saved = []
    with torch.no_grad():
        for mod, u, idx, alpha in edits:
            saved.append((mod, mod.weight.data.clone()))
            mod.weight.data = apply_edit(mod.weight.data, u, idx, alpha)
        try:
            logits = model(input_ids=ids.reshape(1, -1), use_cache=False).logits[0, :-1].float()
        finally:
            for mod, w in saved:
                mod.weight.data = w
    return torch.log_softmax(logits, -1).gather(1, ids[1:, None])[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen35-0.8b-full")
    ap.add_argument("--patch", default="patches/qwen35_0.8b_p50c_loop_fix.moipatch")
    ap.add_argument("--out", default="benchmarks/layer_flow")
    ap.add_argument("--questions", default="101,102,104")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--n-random", type=int, default=8, help="controls per type")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, blocks = find_blocks(model)
    n = len(blocks)
    readers, fin = layer_readers(model), final_reader(model)
    patch = MoIPatch.load(ROOT / args.patch)
    real = []
    for e in patch.edits:
        l = int(e.module.split("layers.")[1].split(".")[0])
        real.append((l, module_by_suffix(model, e.module), e.u.float(), e.idx, float(e.alpha)))
    edit_layers = [r[0] for r in real]

    # ---- loop sequences (cached)
    cache = ROOT / args.out / "loop_seqs_cpu.pt"
    seqs = torch.load(cache) if cache.exists() else {}
    gtr = pd.read_parquet(ROOT / "work/gsm8k/train.parquet")
    for qi in [int(x) for x in args.questions.split(",")]:
        if qi in seqs:
            continue
        prompt = tok.apply_chat_template([{"role": "user", "content": gtr["question"].iloc[qi]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p_ids = tok(prompt, return_tensors="pt").input_ids[0]
        with torch.no_grad():
            g = model.generate(p_ids.unsqueeze(0), attention_mask=torch.ones(1, len(p_ids), dtype=torch.long),
                               max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)[0]
        seqs[qi] = {"ids": g, "prompt_len": len(p_ids), "onset": loop_onset(g[len(p_ids):].tolist())}
        torch.save(seqs, cache)
        print(f"q{qi}: gen {len(g) - len(p_ids)}, onset {seqs[qi]['onset']} ({time.time() - t0:.0f}s)", flush=True)
    seqs = {k: v for k, v in seqs.items() if v["onset"] is not None}
    if not seqs:
        raise SystemExit("no looping sequence")

    # ---- patch set: real + controls (same layers, columns, alpha)
    first = next(iter(seqs.values()))
    _, h0, _, _ = run_states(model, blocks, first["ids"], edit_layers,
                             {"all": slice(1, len(first["ids"]))})
    x_ref = None
    patches = {"p50c": [(l, m, u, idx, a) for l, m, u, idx, a in real]}
    with torch.no_grad():
        xs_ref = {}
        hs = [blocks[l].register_forward_hook(lambda m_, a_, o_, l=l: xs_ref.__setitem__(
            l, (o_[0] if isinstance(o_, tuple) else o_)[0].float())) for l in edit_layers]
        model(input_ids=first["ids"].reshape(1, -1), use_cache=False)
        for h in hs:
            h.remove()
    for kind in ("iso", "resid", "cols"):
        for i in range(args.n_random):
            es = []
            for l, m, u, idx, a in real:
                if kind == "iso":
                    v = torch.randn_like(u)
                elif kind == "resid":
                    X = xs_ref[l]
                    v = torch.randn(X.shape[0]) @ (X - X.mean(0))
                else:
                    W = m.weight.detach().float()
                    t = torch.randint(1, h0[l].shape[0], (1,)).item()
                    v = W[:, idx.long()] @ h0[l][t, idx.long()]
                es.append((l, m, F.normalize(v, dim=0), idx, a))
            patches[f"{kind}{i}"] = es
    print(f"{len(patches)} patches, {len(seqs)} looping seqs ({time.time() - t0:.0f}s)", flush=True)

    # ---- per sequence: real effect + flow metrics
    acc = {name: {k: [] for k in ("real_loop", "real_reason", "F_size", "F_read", "F_logit", "G_loop", "G_select")}
           for name in patches}
    for qi, s in seqs.items():
        ids, P, on = s["ids"], s["prompt_len"], s["onset"]
        regions = {"reason": slice(P, P + on), "loop": slice(P + on, len(ids))}
        x, h, grads, lp0 = run_states(model, blocks, ids, edit_layers, regions)
        target = ids[1:]
        for name, es in patches.items():
            lp = patched_logprob(model, ids, [(m, u, idx, a) for _, m, u, idx, a in es])
            d_lp = lp - lp0
            r = regions
            acc[name]["real_loop"].append(-d_lp[r["loop"].start - 1: r["loop"].stop - 1].mean().item())
            acc[name]["real_reason"].append(-d_lp[r["reason"].start - 1: r["reason"].stop - 1].mean().item())
            fs = fr = fl = gl = gr = 0.0
            for l, m, u, idx, a in es:
                W = m.weight.detach().float()
                mask = torch.zeros(W.shape[1])
                mask[idx.long()] = 1
                coef = h[l] @ (mask * (u @ W))                       # (T,)
                delta = -a * coef[:, None] * u[None, :]                # (T, H)
                lsl = r["loop"]
                fs += (delta[lsl].norm(dim=-1) / x[l][lsl].norm(dim=-1)).mean().item()
                # later readers: direct share of delta in their read (x of the unpatched run as the read)
                for L in range(l + 1, n):
                    for side in ("attn", "mlp"):
                        g_, W_ = readers[L][side]["g"], readers[L][side]["W"]
                        xr = x[L - 1][lsl]
                        read = (xr * g_) @ W_.t()
                        dr = (delta[lsl] * g_) @ W_.t()
                        fr += ((dr * read).sum(-1).abs() / (read * read).sum(-1).clamp_min(1e-9)).mean().item()
                w_t = fin["U"][target[lsl.start - 1: lsl.stop - 1]] * fin["g"]
                xf = x[n - 1][lsl]
                fl += ((delta[lsl] * w_t).sum(-1).abs() / (xf * w_t).sum(-1).abs().clamp_min(1e-6)).mean().item()
                gl += (grads["loop"][l] * delta).sum().item() / (lsl.stop - lsl.start)
                gr += (grads["reason"][l] * delta).sum().item() / (r["reason"].stop - r["reason"].start)
            acc[name]["F_size"].append(fs)
            acc[name]["F_read"].append(fr)
            acc[name]["F_logit"].append(fl)
            acc[name]["G_loop"].append(-gl)          # sign: positive = log-prob of loop tokens drops
            acc[name]["G_select"].append(-gl + gr)
        print(f"q{qi} done ({time.time() - t0:.0f}s)", flush=True)

    mean = {name: {k: sum(v) / len(v) for k, v in d.items()} for name, d in acc.items()}
    names = list(mean)
    truth = [mean[nm]["real_loop"] for nm in names]
    truth_sel = [mean[nm]["real_loop"] - mean[nm]["real_reason"] for nm in names]

    def rank_of(metric):
        order = sorted(names, key=lambda nm: -mean[nm][metric])
        return order.index("p50c") + 1

    metrics = ["F_size", "F_read", "F_logit", "G_loop", "G_select"]
    result = {
        "patches": len(names), "looping_seqs": len(seqs),
        "p50c_rank_by_real_loop_drop": rank_of("real_loop"),
        "spearman_vs_real_loop_drop": {k: round(spearman([mean[nm][k] for nm in names], truth), 3) for k in metrics},
        "spearman_vs_real_selectivity": {k: round(spearman([mean[nm][k] for nm in names], truth_sel), 3) for k in metrics},
        "p50c_rank_by_metric": {k: rank_of(k) for k in metrics},
        "by_kind_mean": {
            kind: {k: round(sum(mean[nm][k] for nm in names if nm.startswith(kind)) /
                            max(1, sum(nm.startswith(kind) for nm in names)), 4)
                   for k in ["real_loop", "real_reason"] + metrics}
            for kind in ("p50c", "iso", "resid", "cols")},
        "per_patch": {nm: {k: round(v, 5) for k, v in mean[nm].items()} for nm in names},
        "seconds": round(time.time() - t0, 1),
    }
    out = ROOT / args.out
    (out / "results_patch_fate_cpu.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "per_patch"}, indent=1))


if __name__ == "__main__":
    main()
