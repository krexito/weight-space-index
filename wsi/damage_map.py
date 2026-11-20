"""Damage map: WHERE does a weight patch hurt the model, not just how much on average.

A single healthy-text NLL number hides the shape of the damage: a patch can be harmless on prose and break code, or
damage a handful of specific tokens badly. This module runs the same token sequences from several domains through the
model with the patch off and on (teacher forcing, no generation) and reports per domain:

  * dnll:       mean change of next-token NLL (patched - base); > 0 is damage
  * kl:         mean KL(base || patched) of the next-token distribution, and its 99th percentile
  * hit_share:  share of tokens with KL above `kl_hit` (default 0.1 nats): how widespread the change is
  * top1_agree: share of positions where the most likely next token is unchanged
  * activity:   for each edited module, mean |cos(module output, u)| on the base pass: is the edit direction even
                present in this domain?
  * worst:      the tokens with the largest KL, with the text before them and the base vs patched top-1 token

`random_like(patch)` builds a control patch with the same modules, kept columns and alpha but random directions, so a
domain's damage can be compared with what an unspecific edit of the same size does.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .moipatch import Edit, MoIPatch


def random_like(patch: MoIPatch, seed: int = 0) -> MoIPatch:
    """Same modules, kept columns, alpha and fingerprints as `patch`, random unit directions."""
    g = torch.Generator().manual_seed(seed)
    edits = []
    for e in patch.edits:
        r = torch.randn(e.u.shape, generator=g, dtype=torch.float64)
        edits.append(Edit(module=e.module, u=r / r.norm(), alpha=e.alpha, idx=e.idx, base_sha256=e.base_sha256,
                          base_dtype=e.base_dtype, shape=e.shape))
    return MoIPatch(edits=edits, meta={"name": "random_like", "seed": seed})


def token_damage(base_logits: torch.Tensor, new_logits: torch.Tensor, targets: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-position damage for next-token prediction. logits [T, V] predict targets [T]."""
    lb = F.log_softmax(base_logits.float(), -1)
    ln = F.log_softmax(new_logits.float(), -1)
    nll_b = -lb.gather(-1, targets[:, None])[:, 0]
    nll_n = -ln.gather(-1, targets[:, None])[:, 0]
    kl = (lb.exp() * (lb - ln)).sum(-1).clamp_min(0.0)
    return {"dnll": (nll_n - nll_b).cpu().numpy(), "kl": kl.cpu().numpy(),
            "agree": (lb.argmax(-1) == ln.argmax(-1)).cpu().numpy(),
            "top1_base": lb.argmax(-1).cpu().numpy(), "top1_new": ln.argmax(-1).cpu().numpy()}


def _find(model, suffix: str):
    for name, m in model.named_modules():
        if name.endswith(suffix):
            return m
    raise KeyError(suffix)


@torch.no_grad()
def damage_map(model, domains: dict[str, list[list[int]]], patches: dict[str, MoIPatch], *,
               decode: Callable[[list[int]], str] | None = None, kl_hit: float = 0.1, n_worst: int = 5,
               context: int = 40, verify_fingerprint: bool = True, logits_fn: Callable | None = None) -> dict:
    """Damage of each patch (always on) on each domain. domains: name -> list of token-id sequences.

    `logits_fn(model, ids[1, T]) -> [T, V]` defaults to `model(input_ids=ids).logits[0]`.
    """
    dev = next(model.parameters()).device
    logits_fn = logits_fn or (lambda m, ids: m(input_ids=ids, use_cache=False).logits[0])
    handles = {name: p.attach(model, verify_fingerprint=verify_fingerprint) for name, p in patches.items()}
    # every patch's edited modules, for the direction activity on the base pass
    probes = {}
    for pname, p in patches.items():
        for i, e in enumerate(p.edits):
            probes[(pname, i)] = (e.module, (e.u / e.u.norm()).float().to(dev))
    mods = {m: _find(model, m) for m, _ in probes.values()}
    out = {name: {} for name in patches}
    try:
        for dname, seqs in domains.items():
            acc = {name: {"dnll": [], "kl": [], "agree": [], "worst": []} for name in patches}
            act = {k: [] for k in probes}
            for si, seq in enumerate(seqs):
                if len(seq) < 2:
                    continue
                ids = torch.tensor([seq], device=dev)
                tgt = ids[0, 1:]
                cap: dict[str, torch.Tensor] = {}
                hooks = [m.register_forward_hook(lambda _m, _i, o, k=k: cap.__setitem__(k, o[0] if isinstance(o, tuple) else o))
                         for k, m in mods.items()]
                try:
                    base = logits_fn(model, ids)[:-1]
                finally:
                    for h in hooks:
                        h.remove()
                for k, (mname, u) in probes.items():
                    y = cap[mname][0].float()
                    act[k].append(((y @ u).abs() / y.norm(dim=-1).clamp_min(1e-6)).mean().item())
                for name, h in handles.items():
                    h.on()
                    try:
                        new = logits_fn(model, ids)[:-1]
                    finally:
                        h.off()
                    d = token_damage(base, new, tgt)
                    a = acc[name]
                    a["dnll"].append(d["dnll"])
                    a["kl"].append(d["kl"])
                    a["agree"].append(d["agree"])
                    for t in np.argsort(-d["kl"])[:n_worst]:
                        a["worst"].append((float(d["kl"][t]), si, int(t), int(d["top1_base"][t]), int(d["top1_new"][t])))
                    del new
                del base
            for name in patches:
                a = acc[name]
                if not a["kl"]:
                    continue
                kl, dn, ag = np.concatenate(a["kl"]), np.concatenate(a["dnll"]), np.concatenate(a["agree"])
                worst = []
                for kl_v, si, t, tb, tn in sorted(a["worst"], reverse=True)[:n_worst]:
                    seq = domains[dname][si]
                    w = {"kl": round(kl_v, 4), "seq": si, "pos": t + 1}
                    if decode is not None:
                        w.update({"before": decode(seq[max(0, t + 1 - context):t + 1]), "actual": decode([seq[t + 1]]),
                                  "base_top1": decode([tb]), "patched_top1": decode([tn])})
                    worst.append(w)
                out[name][dname] = {
                    "n_tokens": int(len(kl)), "dnll": round(float(dn.mean()), 5), "kl": round(float(kl.mean()), 5),
                    "kl_p99": round(float(np.quantile(kl, 0.99)), 5), "hit_share": round(float((kl > kl_hit).mean()), 5),
                    "top1_agree": round(float(ag.mean()), 5),
                    "activity": {f"{patches[name].edits[i].module}": round(float(np.mean(v)), 5)
                                 for (pn, i), v in act.items() if pn == name and v},
                    "worst": worst}
    finally:
        for h in handles.values():
            h.off()
    return out
