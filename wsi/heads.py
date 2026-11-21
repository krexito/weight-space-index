"""Per-head ablation: which attention heads carry a failure (e.g. loop copying) and which carry legitimate behaviour?

A head's contribution enters its block through the output projection (`self_attn.o_proj`, `linear_attn.out_proj`, ...):
the projection's input is the concatenation of the heads' outputs, so zeroing one slice of it removes exactly that head.
`head_ablation(model, {module_name: [head, ...]})` does that with a forward pre-hook (weights are never touched).

`copy_labels(ids, ...)` splits the tokens of one sequence into the sets the loop question needs:
  * LOOP (2):  tokens inside a repetition loop (from a given start on)
  * COPY (1):  outside the loop, tokens an induction step predicts: the k tokens before t occurred earlier and were
               followed by token t then too (names repeated from the context, "step by step", doctest lines...)
  * OTHER (0): every other scored token
  * IGNORED (-1)
A loop fix should make the model less sure of LOOP tokens without touching COPY tokens (the damage map showed the p50c
loop patch does hurt them: wsi/damage_map.py).
"""
from __future__ import annotations

import contextlib
import re

import numpy as np
import torch

LOOP, COPY, OTHER, IGNORED = 2, 1, 0, -1

# (module-name pattern, config attributes giving the per-head input width of that projection, in order of preference)
HEAD_PROJECTIONS = [
    (r"\.self_attn\.o_proj$", ("head_dim",)),
    (r"\.attn\.o_proj$", ("head_dim",)),
    (r"\.linear_attn\.out_proj$", ("linear_value_head_dim",)),
]


def _cfg_get(cfg, name):
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is not None and getattr(c, name, None) is not None:
            return getattr(c, name)
    return None


def find_heads(model) -> dict[str, dict]:
    """{module name: {"n_heads", "head_dim", "kind"}} for every head-structured output projection."""
    cfg = model.config
    hidden, n_att = _cfg_get(cfg, "hidden_size"), _cfg_get(cfg, "num_attention_heads")
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        for pat, attrs in HEAD_PROJECTIONS:
            if not re.search(pat, name):
                continue
            hd = next((_cfg_get(cfg, a) for a in attrs if _cfg_get(cfg, a)), None)
            if hd is None and "self_attn" in name and hidden and n_att:
                hd = hidden // n_att
            if hd and mod.in_features % hd == 0:
                out[name] = {"n_heads": mod.in_features // hd, "head_dim": hd,
                             "kind": "linear_attn" if "linear_attn" in name else "attn"}
            break
    return out


@contextlib.contextmanager
def head_ablation(model, heads: dict[str, list[int]], head_info: dict[str, dict] | None = None):
    """Zero the given heads' slices of their output projection's input for every forward inside the block."""
    info = head_info or find_heads(model)
    mods = dict(model.named_modules())
    hooks = []
    for name, hs in heads.items():
        if not hs:
            continue
        hd = info[name]["head_dim"]
        cols = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in hs])

        def pre(_m, args, cols=cols):
            x = args[0].clone()
            x[..., cols.to(x.device)] = 0
            return (x,) + tuple(args[1:])

        hooks.append(mods[name].register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def copy_labels(ids: list[int], score_from: int, loop_from: int | None, k: int = 4,
                ignore_before_loop: int = 256) -> np.ndarray:
    """Labels for predicting ids[t] (t >= score_from). loop_from=None: no loop in this sequence."""
    n = len(ids)
    lab = np.full(n, IGNORED, np.int8)
    last: dict[tuple, int] = {}  # k-gram -> position right after its latest occurrence
    stop_copy = n if loop_from is None else max(score_from, loop_from - ignore_before_loop)
    for t in range(n):
        if t >= k:
            key = tuple(ids[t - k:t])
            prev = last.get(key)
            if t >= score_from:
                if loop_from is not None and t >= loop_from:
                    lab[t] = LOOP
                elif t < stop_copy:
                    lab[t] = COPY if prev is not None and ids[prev] == ids[t] else OTHER
            last[key] = t
    return lab


@torch.no_grad()
def token_nll(model, ids: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """NLL of ids[1:] given the prefix, [T-1] float32. Log-softmax in chunks so a big vocabulary fits."""
    logits = model(input_ids=ids[None], use_cache=False).logits[0, :-1]
    tgt = ids[1:]
    out = torch.empty(len(tgt), dtype=torch.float32, device=logits.device)
    for s in range(0, len(tgt), chunk):
        lg = logits[s:s + chunk].float()
        out[s:s + chunk] = torch.logsumexp(lg, -1) - lg.gather(-1, tgt[s:s + chunk, None])[:, 0]
    return out


def set_means(nll_delta: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Mean of a per-token quantity over the LOOP / COPY / OTHER sets (labels aligned with the predicted tokens)."""
    res = {}
    for name, c in (("loop", LOOP), ("copy", COPY), ("other", OTHER)):
        m = labels == c
        res[name] = float(nll_delta[m].mean()) if m.any() else None
    return res
