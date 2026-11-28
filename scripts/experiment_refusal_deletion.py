"""EXPERIMENTAL: can switching neurons OFF (Chantless-style deletion, +0 bytes of new weights) fix LFM3's over-refusal
as well as the shipped direction patch? Backlog #3 (learned deletion mask) on refusal instead of loops.

Neuron j of an MLP = input column j of its down projection (feed_forward.w2). Switching it off is W[:, j] = 0, i.e. the
rank-one edit W' = W - |c_j| (c_j/|c_j|) e_j^T, written as one moipatch Edit per neuron (Edit.v = e_j), so the same
routed handle (shipped L24 router + toxic veto) can switch the deletion on per prompt.
Selection (first order, no generation): f = log p(base's refusal opener token) at the last prompt position, on the
fit OR-Bench prompts the base refused. Zeroing neuron j changes f by about -sum_t a_tj * df/da_tj; neurons with the
largest positive mean attribution are switched off. Picked on effect only (AGENTS rule for routed patches).
Candidates: layer scope {late (last third), all} x k {16, 64, 256}; one picked on the dev split with the score of
experiment_refusal_select.py (recovered - toxic penalty - style penalty), 64 tokens.
Exam (fresh prompts as in test 2, 128 tokens):
  base | patch (shipped, routed) | del (picked mask, routed) | del_on (always on) | rand (same per-layer counts, routed)

  python scripts/experiment_refusal_deletion.py --out R.json [--limit N] [--max-new-exam 128]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import experiment_refusal_controls as rc  # noqa: E402
import experiment_refusal_select as rs  # noqa: E402
import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
from wsi.moipatch import Edit, MoIPatch  # noqa: E402

KS = (16, 64, 256)
SCOPES = ("late", "all")


def attribution(model, tok, prompts: list[str], targets: list[int], downs: dict[str, torch.nn.Module], dev) -> dict:
    """Mean over prompts of sum_t a_tj * d log p(target) / d a_tj for every down-projection input neuron j."""
    S = {n: torch.zeros(m.weight.shape[1], dtype=torch.float64) for n, m in downs.items()}
    acts: dict[str, torch.Tensor] = {}
    hooks = []
    for n, m in downs.items():
        def pre(_m, inp, n=n):
            x = inp[0]
            if x.requires_grad:
                x.retain_grad()
                acts[n] = x
        hooks.append(m.register_forward_pre_hook(pre))
    emb = model.get_input_embeddings()
    try:
        for p, t in zip(prompts, targets):
            ids = torch.tensor([tok.encode(p, add_special_tokens=False)], device=dev)
            e = emb(ids).detach().requires_grad_(True)
            acts.clear()
            with torch.enable_grad():
                lp = torch.log_softmax(model(inputs_embeds=e).logits[0, -1].float(), -1)[t]
                lp.backward()
            for n, x in acts.items():
                S[n] += (x.detach() * x.grad).float().sum(dim=(0, 1)).double().cpu()
            model.zero_grad(set_to_none=True)
    finally:
        for h in hooks:
            h.remove()
    return {n: v / max(1, len(prompts)) for n, v in S.items()}


def deletion_patch(writers, picks: dict[str, list[int]], router: dict, name: str) -> MoIPatch:
    by = {w.suffix: w for w in writers}
    edits = []
    for mod, cols in picks.items():
        w = by[mod]
        W = w.module.weight.detach().double().cpu()
        for j in cols:
            c = W[:, j]
            nrm = float(c.norm())
            if nrm == 0.0:
                continue
            v = torch.zeros(W.shape[1], dtype=torch.float64)
            v[j] = 1.0
            edits.append(Edit(module=mod, u=c / nrm, alpha=nrm, base_sha256=w.base_sha256, base_dtype=w.base_dtype,
                              shape=w.shape, v=v))
    return MoIPatch(edits=edits, router=router, meta={"name": name, "neurons": sum(map(len, picks.values()))})


def top_k(scores: dict[str, torch.Tensor], mods: list[str], k: int) -> dict[str, list[int]]:
    flat = torch.cat([scores[m] for m in mods])
    off = np.cumsum([0] + [scores[m].numel() for m in mods])
    sel = torch.topk(flat, min(k, flat.numel())).indices.tolist()
    out: dict[str, list[int]] = {}
    for i in sel:
        b = int(np.searchsorted(off, i, side="right") - 1)
        out.setdefault(mods[b], []).append(int(i - off[b]))
    return out


def random_like(picks: dict[str, list[int]], scores: dict[str, torch.Tensor], seed: int = 0) -> dict[str, list[int]]:
    g = torch.Generator().manual_seed(seed)
    return {m: torch.randperm(scores[m].numel(), generator=g)[: len(c)].tolist() for m, c in picks.items()}


def gen_all(model, tok, fmt: dict, dev, n: int, bs: int, patch: MoIPatch | None, mode: str = "auto") -> dict:
    if patch is None:
        return {k: sr._generate_texts(model, tok, v, dev, max_new_tokens=n, batch_size=bs) for k, v in fmt.items()}
    with patch.enable_routed(model, mode=mode):
        return {k: sr._generate_texts(model, tok, v, dev, max_new_tokens=n, batch_size=bs) for k, v in fmt.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--patch", default=str(HERE.parent / "patches/lfm3_1.7b_abliterated_sfw_routed.moipatch"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-dev", type=int, default=64)
    ap.add_argument("--max-new-exam", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    model, tok, dev, _ = sr._load_model_and_tokenizer(a.model, "bf16" if torch.cuda.is_available() else "fp32")
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers, writers = sr._discover_all_writers(model)
    downs = {w.suffix: w.module for w in writers if w.kind == "mlp_down"}
    layer_of = {w.suffix: w.layer for w in writers}
    data = rs.load_splits()
    if a.limit:
        data = {k: v[: max(a.limit, 6) if k == "fit_or" else a.limit] for k, v in data.items()}
    fmt = {k: [sr._format_prompt(tok, p) for p in v] for k, v in data.items()}
    shipped = MoIPatch.load(a.patch)
    bs = a.batch_size
    dev_fmt = {k: v for k, v in fmt.items() if k.startswith("dev")}
    exam_fmt = {k: v for k, v in fmt.items() if k.startswith("exam")}

    fit = sr._generate_texts(model, tok, fmt["fit_or"], dev, max_new_tokens=64, batch_size=bs)
    pr = [(p, r["gen_ids"][0]) for p, r in zip(fmt["fit_or"], fit) if r["refused"] and r["gen_ids"]]
    print(f"fit: base refused {len(pr)}/{len(fit)} ({time.time() - t0:.0f}s)", flush=True)
    scores = attribution(model, tok, [p for p, _ in pr], [t for _, t in pr], downs, dev)
    print(f"attribution over {len(downs)} MLPs done ({time.time() - t0:.0f}s)", flush=True)

    late_from = int(np.ceil(2 * n_layers / 3))
    scope_mods = {"all": sorted(downs, key=lambda m: layer_of[m]),
                  "late": sorted([m for m in downs if layer_of[m] >= late_from], key=lambda m: layer_of[m])}
    base_dev = gen_all(model, tok, dev_fmt, dev, a.max_new_dev, bs, None)
    cands = {}
    for sc in SCOPES:
        for k in KS:
            nm = f"{sc}_k{k}"
            picks = top_k(scores, scope_mods[sc], k)
            P = deletion_patch(writers, picks, shipped.router, nm)
            o = rs.outcome(base_dev, gen_all(model, tok, dev_fmt, dev, a.max_new_dev, bs, P), "dev")
            cands[nm] = {"picks": picks, "dev": o,
                         "layers": {str(layer_of[m]): len(c) for m, c in sorted(picks.items(), key=lambda x: layer_of[x[0]])},
                         "attr_sum": round(float(sum(scores[m][c].sum() for m, c in picks.items())), 4)}
            print(f"[{nm}] dev {o['recovered']:.3f} rec / {o['toxic_kept']:.3f} kept / style {o['style_ok']} -> score "
                  f"{o['score']} ({time.time() - t0:.0f}s)", flush=True)
    best = max(cands, key=lambda n: cands[n]["dev"]["score"])
    picks = cands[best]["picks"]
    print(f"picked {best}", flush=True)

    arms = {"base": (None, "auto"), "patch": (shipped, "auto"),
            "del": (deletion_patch(writers, picks, shipped.router, best), "auto"),
            "del_on": (deletion_patch(writers, picks, shipped.router, best), "always_on"),
            "rand": (deletion_patch(writers, random_like(picks, scores), shipped.router, "rand"), "auto")}
    gens = {}
    for nm, (P, mode) in arms.items():
        gens[nm] = gen_all(model, tok, exam_fmt, dev, a.max_new_exam, bs, P, mode)
        print(f"[{nm}] " + " ".join(f"{k}: {sum(r['refused'] for r in v)}/{len(v)}" for k, v in gens[nm].items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
    res = {"model": a.model, "n": {k: len(v) for k, v in data.items()}, "fit_refused": len(pr), "n_layers": n_layers,
           "late_from_layer": late_from, "picked": best,
           "candidates": {n: {k: v for k, v in c.items() if k != "picks"} for n, c in cands.items()},
           "picked_neurons": {m: sorted(c) for m, c in picks.items()}, "arms": {}}
    for nm in arms:
        o = rs.outcome(gens["base"], gens[nm], "exam")
        o["degenerate"] = {k: round(float(np.mean([rc.degenerate(r["gen_ids"]) for r in v])), 4) for k, v in gens[nm].items()}
        o["ctrl_nll_base_model"] = round(float(np.nanmean(rc.answer_nll(model, gens[nm]["exam_ctrl"], dev))), 4)
        res["arms"][nm] = o
    res["examples_del_or"] = [r["text"][:240] for r, b in zip(gens["del"]["exam_or"], gens["base"]["exam_or"])
                              if b["refused"]][:4]
    res["elapsed_s"] = round(time.time() - t0)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for nm, o in res["arms"].items():
        print(f"{nm:7s} recovered {o['recovered']:.3f} toxic kept {o['toxic_kept']:.3f} ctrl same {o['ctrl_identical']:.2f} "
              f"style_ok {o['style_ok']} degenerate {o['degenerate']} ctrl NLL {o['ctrl_nll_base_model']}")


if __name__ == "__main__":
    main()
