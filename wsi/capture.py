"""Runtime capture -> activation index -> Selective MoI routing.

Two kinds of Mixture of Indexes (MoI) exist in `wsi`:
  * Additive MoI  (`AdditiveMoI` = `DynamicCircuitPager`): the model keeps all its weights; small payloads are ADDED
                   into dormant slots only when the context needs them. Goal: new capabilities, no quality loss.
  * Selective MoI (`SelectiveMoI` = `TaskRouter`, this module): an activation index decides WHICH existing neurons
                   to load; the rest stay out of memory. Goal: fewer resources; costs quality when neurons are dropped.
A Compensated Selective MoI refits the loaded neurons to cover the dropped ones (see
scripts/experiment_index_criteria.py, "+recon").

The rest of `wsi` reads weights only. This module records what the model actually USES when it runs, stores it as an
index per task, and uses those indexes to decide which weights to load.

    1. Capture   `capture_index(model, sequences, task="math")`
                 Streams a corpus through the model with hooks on every linear layer (input and output units) and on
                 every decoder block (residual stream). Per unit it keeps: mean, mean |x|, mean x^2, max |x|, firing
                 rate (|x| > eps) and, for linear inputs, the weight usage  E|x_i| * ||W[:, i]||.
                 `trace_all(model, ids)` instead records every leaf module output densely for a few tokens
                 (hidden states, attention weights and the KV / recurrent cache included when the model returns them).
    2. Index     `ActivationIndex`: save / load (safetensors + json), `select()` the MLP neurons to keep under a
                 budget, `overlap()` and `specific_units()` to compare tasks, `head_usage()` for attention heads.
    3. Route     `TaskRouter(indexes)`: profiles a prompt, picks the closest task index, and `loaded()` physically
                 slices the MLP matrices to that index's neurons (a real smaller model) and restores them after.

Works with nn.Linear and GPT-2 style Conv1D (weight [in, out]); MLPs are found structurally (a module whose "down"
linear reads exactly what its sibling "up"/"gate" linears write), so Llama/Qwen (gate/up/down), LFM3 (w1/w3/w2),
GPT-2 (c_fc/c_proj) and OPT (fc1/fc2) need no per-model code.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn

STATS = ("n", "sum", "sum_abs", "sum_sq", "max_abs", "fire")
_EPS_FIRE = 1e-6


# ----------------------------------------------------------------------------------------------- module helpers
def _is_linear(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear) or (type(m).__name__ == "Conv1D" and getattr(m, "weight", None) is not None
                                        and m.weight.dim() == 2)


def _io_dims(m: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) for Linear ([out, in] weight) and Conv1D ([in, out] weight)."""
    w = m.weight
    return (w.shape[1], w.shape[0]) if isinstance(m, nn.Linear) else (w.shape[0], w.shape[1])


def _input_col_norms(m: nn.Module) -> torch.Tensor:
    """||W[:, i]||: how strongly input unit i is written into the output."""
    w = m.weight.float()
    return w.norm(dim=0) if isinstance(m, nn.Linear) else w.norm(dim=1)


def find_blocks(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """The decoder stack: the largest ModuleList in the model."""
    best = max(((n, m) for n, m in model.named_modules() if isinstance(m, nn.ModuleList)),
               key=lambda nm: len(nm[1]), default=(None, None))
    if best[1] is None:
        raise ValueError("no ModuleList of decoder blocks found")
    return best


@dataclass
class MLPGroup:
    name: str            # parent module
    ups: list[str]       # linears that write the hidden units (gate, up / w1, w3 / c_fc / fc1)
    down: str            # linear that reads them (down_proj / w2 / c_proj / fc2)
    units: int


def find_mlps(model: nn.Module) -> list[MLPGroup]:
    groups = []
    for pname, parent in model.named_modules():
        lin = {n: c for n, c in parent.named_children() if _is_linear(c)}
        if len(lin) < 2:
            continue
        for dn, d in lin.items():
            d_in, d_out = _io_dims(d)
            if d_in <= d_out:  # the down projection contracts the (wider) hidden units back to the residual width
                continue
            ups = [n for n, c in lin.items() if n != dn and _io_dims(c) == (d_out, d_in)]
            if ups and len(ups) == len(lin) - 1:
                pre = f"{pname}." if pname else ""
                groups.append(MLPGroup(pname, [pre + u for u in ups], pre + dn, d_in))
                break
    return groups


def _tensor_out(o: Any) -> torch.Tensor | None:
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (tuple, list)) and o and isinstance(o[0], torch.Tensor):
        return o[0]
    return getattr(o, "last_hidden_state", None) if not isinstance(o, dict) else None


# ----------------------------------------------------------------------------------------------- statistics
class _Acc:
    def __init__(self, dim: int, device: torch.device):
        z = lambda: torch.zeros(dim, device=device, dtype=torch.float64)  # noqa: E731
        self.n = torch.zeros((), device=device, dtype=torch.float64)
        self.sum, self.sum_abs, self.sum_sq, self.max_abs, self.fire = z(), z(), z(), z(), z()

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, x.shape[-1]).float()
        a = x.abs()
        self.n += x.shape[0]
        self.sum += x.sum(0).double()
        self.sum_abs += a.sum(0).double()
        self.sum_sq += (x * x).sum(0).double()
        self.max_abs = torch.maximum(self.max_abs, a.max(0).values.double())
        self.fire += (a > _EPS_FIRE).sum(0).double()

    def tensors(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, k).cpu() for k in STATS}


# ----------------------------------------------------------------------------------------------- the index
@dataclass
class ActivationIndex:
    """What a model used on one task: per-unit activation statistics for every captured site."""

    task: str
    tokens: int
    stats: dict[str, dict[str, torch.Tensor]]      # site -> {n, sum, sum_abs, sum_sq, max_abs, fire}
    usage: dict[str, torch.Tensor]                  # linear name -> E|x_i| * ||W[:, i]|| per input unit
    mlps: list[MLPGroup] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- per-site views
    def mean_abs(self, site: str) -> torch.Tensor:
        s = self.stats[site]
        return s["sum_abs"] / s["n"].clamp_min(1)

    def firing_rate(self, site: str) -> torch.Tensor:
        s = self.stats[site]
        return s["fire"] / s["n"].clamp_min(1)

    def rms(self, site: str) -> torch.Tensor:
        s = self.stats[site]
        return (s["sum_sq"] / s["n"].clamp_min(1)).sqrt()

    def neuron_scores(self) -> dict[str, torch.Tensor]:
        """MLP neuron importance: weight usage of the down projection's input units."""
        return {g.down: self.usage[g.down] for g in self.mlps}

    def head_usage(self, linear: str, n_heads: int) -> torch.Tensor:
        """Share of usage per attention head, from the input units of an output projection (o_proj / out_proj)."""
        u = self.usage[linear]
        h = u.reshape(n_heads, -1).sum(1)
        return h / h.sum().clamp_min(1e-12)

    # -- choosing what to load
    def select(self, budget: float, mode: str = "uniform") -> dict[str, torch.Tensor]:
        """MLP neurons to keep: {down linear name: sorted unit indices}.

        uniform: the same fraction `budget` of neurons in every layer.
        global : ranks all neurons together by their share of their layer's usage, so layers whose usage is spread
                 out keep more neurons (total = budget * all neurons; each layer keeps at least one).
        """
        scores = self.neuron_scores()
        if mode == "uniform":
            return {d: s.topk(max(1, round(budget * len(s)))).indices.sort().values for d, s in scores.items()}
        if mode != "global":
            raise ValueError("mode must be 'uniform' or 'global'")
        names = list(scores)
        rel = [scores[d] / scores[d].sum().clamp_min(1e-12) for d in names]
        flat = torch.cat(rel)
        k = max(len(names), round(budget * flat.numel()))
        keep = torch.zeros_like(flat, dtype=torch.bool)
        keep[flat.topk(k).indices] = True
        out, off = {}, 0
        for d, r in zip(names, rel):
            m = keep[off : off + len(r)]
            off += len(r)
            out[d] = m.nonzero().flatten() if m.any() else r.argmax().reshape(1)
        return out

    def overlap(self, other: "ActivationIndex", budget: float) -> float:
        """Fraction of selected neurons shared with another task's index at the same budget."""
        a, b = self.select(budget), other.select(budget)
        inter = sum(len(set(a[d].tolist()) & set(b[d].tolist())) for d in a)
        return inter / max(sum(len(v) for v in a.values()), 1)

    def specific_units(self, other: "ActivationIndex", top: int = 20) -> list[dict[str, Any]]:
        """Neurons this task uses much more than `other` (log ratio of weight usage)."""
        rows = []
        for d, s in self.neuron_scores().items():
            r = (s + 1e-9).log() - (other.usage[d] + 1e-9).log()
            for i in r.topk(min(top, len(r))).indices.tolist():
                rows.append({"linear": d, "unit": i, "log_ratio": float(r[i]),
                             "usage_self": float(s[i]), "usage_other": float(other.usage[d][i])})
        return sorted(rows, key=lambda x: -x["log_ratio"])[:top]

    def profile(self) -> torch.Tensor:
        """Task fingerprint: per-layer normalized mean |h| of every MLP hidden unit, concatenated."""
        v = [self.mean_abs(g.down + ":in") for g in self.mlps]
        return torch.cat([x / x.norm().clamp_min(1e-12) for x in v])

    def size_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self.stats.values() for t in s.values()) + \
            sum(t.numel() * t.element_size() for t in self.usage.values())

    # -- persistence
    def save(self, path: str | Path) -> Path:
        from safetensors.torch import save_file

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        flat = {f"stats/{site}/{k}": v.contiguous() for site, s in self.stats.items() for k, v in s.items()}
        flat.update({f"usage/{n}": v.contiguous() for n, v in self.usage.items()})
        save_file(flat, str(p / "activations.safetensors"))
        (p / "index.json").write_text(json.dumps({
            "task": self.task, "tokens": self.tokens, "meta": self.meta,
            "mlps": [g.__dict__ for g in self.mlps]}, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ActivationIndex":
        from safetensors.torch import load_file

        p = Path(path)
        info = json.loads((p / "index.json").read_text(encoding="utf-8"))
        stats: dict[str, dict[str, torch.Tensor]] = {}
        usage: dict[str, torch.Tensor] = {}
        for k, v in load_file(str(p / "activations.safetensors")).items():
            kind, rest = k.split("/", 1)
            if kind == "usage":
                usage[rest] = v
            else:
                site, stat = rest.rsplit("/", 1)
                stats.setdefault(site, {})[stat] = v
        return cls(info["task"], info["tokens"], stats, usage, [MLPGroup(**g) for g in info["mlps"]], info["meta"])


# ----------------------------------------------------------------------------------------------- capture
@torch.no_grad()
def capture_index(
    model: nn.Module,
    sequences: Iterable[list[int] | torch.Tensor],
    *,
    task: str = "task",
    max_tokens: int | None = None,
    linear_outputs: bool = True,
    max_out_dim: int = 65_536,
) -> ActivationIndex:
    """Stream `sequences` (token id lists, one forward each) through `model` and build its ActivationIndex.

    Sites captured: "<linear>:in" and "<linear>:out" for every linear layer (outputs wider than `max_out_dim`, i.e.
    the LM head, are skipped), and "<block>:resid" for the residual stream after every decoder block.
    """
    dev = next(model.parameters()).device
    accs: dict[str, _Acc] = {}
    handles = []

    def site(name: str, dim: int) -> _Acc:
        if name not in accs:
            accs[name] = _Acc(dim, dev)
        return accs[name]

    linears = {n: m for n, m in model.named_modules() if _is_linear(m)}
    for n, m in linears.items():
        d_in, d_out = _io_dims(m)

        def hook(mod, inp, out, n=n, d_in=d_in, d_out=d_out):
            site(f"{n}:in", d_in).add(inp[0])
            if linear_outputs and d_out <= max_out_dim:
                site(f"{n}:out", d_out).add(out)

        handles.append(m.register_forward_hook(hook))
    bname, blocks = find_blocks(model)
    for i, b in enumerate(blocks):
        def bhook(mod, inp, out, i=i):
            t = _tensor_out(out)
            if t is not None:
                site(f"{bname}.{i}:resid", t.shape[-1]).add(t)

        handles.append(b.register_forward_hook(bhook))

    seen = 0
    try:
        for s in sequences:
            ids = s if isinstance(s, torch.Tensor) else torch.tensor(s)
            ids = ids.reshape(1, -1).to(dev)
            if max_tokens is not None and seen >= max_tokens:
                break
            model(input_ids=ids)
            seen += ids.shape[1]
    finally:
        for h in handles:
            h.remove()

    stats = {k: a.tensors() for k, a in accs.items()}
    usage = {}
    for n, m in linears.items():
        if f"{n}:in" in stats:
            s = stats[f"{n}:in"]
            usage[n] = (s["sum_abs"] / s["n"].clamp_min(1)).float() * _input_col_norms(m).cpu()
    cfg = getattr(model, "config", None)
    meta = {"model": getattr(cfg, "_name_or_path", None) or type(model).__name__,
            "model_type": getattr(cfg, "model_type", None), "blocks": bname}
    return ActivationIndex(task, seen, stats, usage, find_mlps(model), meta)


@torch.no_grad()
def trace_all(model: nn.Module, input_ids: torch.Tensor | list[int], *, max_tokens: int = 32) -> dict[str, torch.Tensor]:
    """Dense record of EVERYTHING for a short input: every leaf module output, every linear input, and whatever the
    model returns (hidden states, attention weights when the attention implementation exposes them, KV / recurrent
    cache tensors). Size grows with tokens x all widths, so keep inputs short."""
    dev = next(model.parameters()).device
    ids = (input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)).reshape(1, -1)[:, :max_tokens]
    rec: dict[str, torch.Tensor] = {}
    handles = []
    for n, m in model.named_modules():
        if not n:
            continue
        leaf = not any(True for _ in m.children())
        if not (leaf or _is_linear(m)):
            continue

        def hook(mod, inp, out, n=n, lin=_is_linear(m)):
            t = _tensor_out(out)
            if t is not None:
                rec[f"{n}:out"] = t.detach().float().cpu()
            if lin and inp and isinstance(inp[0], torch.Tensor):
                rec[f"{n}:in"] = inp[0].detach().float().cpu()

        handles.append(m.register_forward_hook(hook))
    try:
        try:
            out = model(input_ids=ids.to(dev), output_hidden_states=True, output_attentions=True, use_cache=True)
        except (TypeError, ValueError, NotImplementedError):
            out = model(input_ids=ids.to(dev), output_hidden_states=True, use_cache=True)
    finally:
        for h in handles:
            h.remove()
    for i, h in enumerate(getattr(out, "hidden_states", None) or ()):
        rec[f"hidden_states.{i}"] = h.float().cpu()
    for i, a in enumerate(getattr(out, "attentions", None) or ()):
        if isinstance(a, torch.Tensor):
            rec[f"attentions.{i}"] = a.float().cpu()
    cache = getattr(out, "past_key_values", None)
    for name in ("key_cache", "value_cache", "conv_states", "recurrent_states", "ssm_states"):
        for i, t in enumerate(getattr(cache, name, None) or ()):
            if isinstance(t, torch.Tensor) and t.numel():
                rec[f"cache.{name}.{i}"] = t.float().cpu()
    for i, lyr in enumerate(getattr(cache, "layers", None) or ()):
        for attr in ("keys", "values", "conv_states", "recurrent_states"):
            t = getattr(lyr, attr, None)
            if isinstance(t, torch.Tensor) and t.numel():
                rec[f"cache.layers.{i}.{attr}"] = t.float().cpu()
    return rec


# ----------------------------------------------------------------------------------------------- loading an index
@contextmanager
def apply_selection(model: nn.Module, selection: dict[str, torch.Tensor]) -> Iterator[dict[str, Any]]:
    """Physically slice every selected MLP to its kept neurons (up/gate output rows, down input columns), run the
    body, then restore the full weights. Yields the parameter counts before and after."""
    mods = dict(model.named_modules())
    groups = {g.down: g for g in find_mlps(model)}
    saved = []
    before = sum(p.numel() for p in model.parameters())

    def put(m: nn.Module, w: torch.Tensor, b: torch.Tensor | None) -> None:
        saved.append((m, m.weight, m.bias, getattr(m, "in_features", None), getattr(m, "out_features", None),
                      getattr(m, "nf", None)))
        m.weight = nn.Parameter(w.contiguous(), requires_grad=False)
        if b is not None:
            m.bias = nn.Parameter(b.contiguous(), requires_grad=False)
        i, o = _io_dims(m)
        if isinstance(m, nn.Linear):
            m.in_features, m.out_features = i, o
        else:
            m.nf = o

    try:
        for down, ix in selection.items():
            g = groups[down]
            ix = ix.to(mods[down].weight.device)
            for u in g.ups:
                m = mods[u]
                w = m.weight[ix] if isinstance(m, nn.Linear) else m.weight[:, ix]
                put(m, w, None if m.bias is None else m.bias[ix])
            d = mods[down]
            put(d, d.weight[:, ix] if isinstance(d, nn.Linear) else d.weight[ix], None)
        yield {"params_full": before, "params_loaded": sum(p.numel() for p in model.parameters())}
    finally:
        for m, w, b, i, o, nf in reversed(saved):
            m.weight = w
            if b is not None or m.bias is not None:
                m.bias = b
            if isinstance(m, nn.Linear):
                m.in_features, m.out_features = i, o
            elif nf is not None:
                m.nf = nf


class TaskRouter:
    """Selective MoI at the task level: pick which activation index (and so which neurons) to load per prompt.

    Alias: `SelectiveMoI`. For adding capabilities without removing anything, use Additive MoI
    (`wsi.AdditiveMoI` / `DynamicCircuitPager`)."""

    def __init__(self, indexes: dict[str, ActivationIndex] | list[ActivationIndex]):
        self.indexes = indexes if isinstance(indexes, dict) else {i.task: i for i in indexes}
        self._profiles = {k: v.profile() for k, v in self.indexes.items()}

    @torch.no_grad()
    def route(self, model: nn.Module, input_ids: list[int] | torch.Tensor) -> dict[str, Any]:
        """Profile the prompt with one prefill (MLP hidden mean |h| only) and return the closest task by cosine."""
        probe = capture_index(model, [input_ids], task="prompt", linear_outputs=False)
        p = probe.profile()
        scores = {k: float(torch.nn.functional.cosine_similarity(p, v, dim=0)) for k, v in self._profiles.items()}
        best = max(scores, key=scores.get)
        return {"task": best, "scores": scores}

    @contextmanager
    def loaded(self, model: nn.Module, task: str, budget: float, mode: str = "uniform") -> Iterator[dict[str, Any]]:
        with apply_selection(model, self.indexes[task].select(budget, mode)) as info:
            yield {"task": task, "budget": budget, "mode": mode, **info}


SelectiveMoI = TaskRouter
