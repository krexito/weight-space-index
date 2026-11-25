"""Token traces: store a trace as its tokens (a few KB) and recompute the activations when needed.

A CompactTrace keeps INT8 residuals (~20 MB per 800-token answer on Qwen3.5-0.8B). The same residuals are a
deterministic function of the tokens: an fp32 recompute matches the original to ~1e-6 on CPU and GPU
(research/findings.md, "Recompute GPU vs CPU"), while bf16 drifts ~1.3% (2x INT8's error). So a TokenTrace stores

  - the token ids (or text re-encoded with the tokenizer: ``retokenized=True``),
  - the recipe: model id, dtype, the GatedDeltaNet kernel in use, library versions, capture_compact kwargs,
  - a small fingerprint: block outputs at a few layers and positions, projected on fixed random directions,

and ``recompute`` rebuilds the full CompactTrace (or raw fp32 states), first checking the fingerprint so a wrong
model, dtype or kernel fails loudly instead of silently giving different numbers.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .capture import find_blocks

FORMAT = "wsi.token_trace/1"
# relL2 of the fingerprint above which a recompute is refused (fp32 CPU/GPU: ~1e-6; fp16/fla: ~1e-3; bf16: ~1e-2)
DEFAULT_TOL = 5e-3


def _dtype_name(model: nn.Module) -> str:
    dt = next(model.parameters()).dtype
    return {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dt, str(dt))


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def kernel_info(model: nn.Module) -> dict[str, Any]:
    """Which implementation the recurrent (GatedDeltaNet-style) layers use, if any: fla Triton kernels and
    causal-conv1d change the numbers by ~1e-3 vs transformers' torch path."""
    info: dict[str, Any] = {}
    for _, m in model.named_modules():
        for attr in ("chunk_gated_delta_rule", "causal_conv1d_fn"):
            if attr in info or not hasattr(m, attr):
                continue
            fn = getattr(m, attr)
            info[attr] = None if fn is None else f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}"
    return info


def block_states(model: nn.Module, ids: list[int] | torch.Tensor, layers: list[int] | None = None) -> dict[int, torch.Tensor]:
    """Teacher-forced forward; returns {block index: output [T, d] as fp32 on CPU}."""
    _, blocks = find_blocks(model)
    want = set(range(len(blocks)) if layers is None else layers)
    acts: dict[int, torch.Tensor] = {}
    hooks = [b.register_forward_hook(lambda m, a, o, l=l: acts.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].detach().float().cpu()))
             for l, b in enumerate(blocks) if l in want]
    x = torch.as_tensor(ids, dtype=torch.long).reshape(1, -1).to(_device(model))
    try:
        with torch.no_grad():
            model(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {l: acts[l] for l in sorted(acts)}


def _fp_plan(n_layers: int, T: int, n_layer_pts: int = 4, n_pos: int = 32) -> tuple[list[int], list[int]]:
    layers = sorted({round(i * (n_layers - 1) / max(1, n_layer_pts - 1)) for i in range(n_layer_pts)})
    pos = sorted({round(i * (T - 1) / max(1, n_pos - 1)) for i in range(min(n_pos, T))})
    return layers, pos


def _fp_dirs(d: int, k: int, seed: int) -> torch.Tensor:
    P = torch.randn(d, k, generator=torch.Generator().manual_seed(seed))
    return P / P.norm(dim=0, keepdim=True)


def fingerprint(states: dict[int, torch.Tensor], layers: list[int], pos: list[int], k: int = 8, seed: int = 0) -> list[list[list[float]]]:
    """[layer][position][k] projections of the block outputs on k fixed random unit directions."""
    d = next(iter(states.values())).shape[1]
    P = _fp_dirs(d, k, seed)
    # 7 significant digits: rounding error ~1e-7 relL2, far below the fp32 recompute noise (~1e-6), and half the JSON size
    return [[[float(f"{v:.7g}") for v in row] for row in (states[l][pos] @ P).tolist()] for l in layers]


def _rel_l2(a: list, b: list) -> float:
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    return float((ta - tb).norm() / ta.norm().clamp_min(1e-12))


@dataclass
class TokenTrace:
    ids: list[int]
    model: str
    dtype: str
    n_layers: int
    hidden_size: int
    kernels: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    capture_kwargs: dict[str, Any] = field(default_factory=dict)
    fp_layers: list[int] = field(default_factory=list)
    fp_pos: list[int] = field(default_factory=list)
    fp_k: int = 8
    fp_seed: int = 0
    fp: list = field(default_factory=list)
    retokenized: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    format: str = FORMAT

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenTrace":
        if d.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} record: {d.get('format')!r}")
        return cls(**d)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()))
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "TokenTrace":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------ checks
    def check(self, model: nn.Module, states: dict[int, torch.Tensor] | None = None) -> dict[str, Any]:
        """Recompute the fingerprint on ``model``; returns relL2 vs the stored one and what differs in the recipe."""
        if states is None:
            states = block_states(model, self.ids, self.fp_layers)
        now = fingerprint(states, self.fp_layers, self.fp_pos, self.fp_k, self.fp_seed)
        rep = {"fp_rel_l2": _rel_l2(self.fp, now), "dtype": _dtype_name(model), "dtype_stored": self.dtype}
        k_now = kernel_info(model)
        if k_now != self.kernels:
            rep["kernels_now"], rep["kernels_stored"] = k_now, self.kernels
        return rep


def _versions() -> dict[str, str]:
    v = {"torch": str(torch.__version__), "python": platform.python_version()}
    try:
        import transformers
        v["transformers"] = str(transformers.__version__)
    except ImportError:
        pass
    return v


def make_token_trace(model: nn.Module, ids: list[int] | torch.Tensor, model_id: str, *,
                     capture_kwargs: dict[str, Any] | None = None, retokenized: bool = False,
                     meta: dict[str, Any] | None = None, fp_layer_pts: int = 4, fp_pos: int = 32, fp_k: int = 8,
                     allow_low_precision: bool = False) -> TokenTrace:
    """Record a TokenTrace for ``ids`` on ``model`` (one forward pass). The model should be fp32: a bf16 fingerprint
    could only be matched by the same bf16 setup (pass ``allow_low_precision=True`` to accept that)."""
    ids = [int(x) for x in torch.as_tensor(ids).reshape(-1).tolist()]
    dt = _dtype_name(model)
    if dt != "fp32" and not allow_low_precision:
        raise ValueError(f"model is {dt}; token traces should be recorded in fp32 (allow_low_precision=True to override)")
    _, blocks = find_blocks(model)
    layers, pos = _fp_plan(len(blocks), len(ids), fp_layer_pts, fp_pos)
    st = block_states(model, ids, layers)
    return TokenTrace(ids=ids, model=model_id, dtype=dt, n_layers=len(blocks), hidden_size=int(st[layers[0]].shape[1]),
                      kernels=kernel_info(model), versions=_versions(), capture_kwargs=dict(capture_kwargs or {}),
                      fp_layers=layers, fp_pos=pos, fp_k=fp_k, fp_seed=0, fp=fingerprint(st, layers, pos, fp_k, 0),
                      retokenized=retokenized, meta=dict(meta or {}))


def ids_from_text(tokenizer: Any, prompt: str, answer: str = "") -> list[int]:
    """Old runs often saved text, not ids. Re-encoding gives the same text but not always the same split into
    tokens as the original generation; mark such traces ``retokenized=True``."""
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if answer:
        ids += list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    return ids


def recompute_states(tt: TokenTrace, model: nn.Module, *, verify: bool = True, tol: float = DEFAULT_TOL,
                     layers: list[int] | None = None) -> dict[int, torch.Tensor]:
    """Raw fp32 block outputs for every (or the given) layer, after checking the fingerprint."""
    need = sorted(set(layers if layers is not None else range(tt.n_layers)) | (set(tt.fp_layers) if verify else set()))
    st = block_states(model, tt.ids, need)
    if verify:
        _verify(tt, model, st, tol)
    return st if layers is None else {l: st[l] for l in layers}


def recompute(tt: TokenTrace, model: nn.Module, *, verify: bool = True, tol: float = DEFAULT_TOL, **overrides: Any):
    """Rebuild the CompactTrace with the stored capture_compact kwargs (``overrides`` win)."""
    from .tracer import capture_compact

    if verify:
        _verify(tt, model, None, tol)
    trace = capture_compact(model, torch.tensor(tt.ids, dtype=torch.long), **{**tt.capture_kwargs, **overrides})
    trace.meta["token_trace"] = {"model": tt.model, "dtype": _dtype_name(model), "retokenized": tt.retokenized}
    return trace


def _verify(tt: TokenTrace, model: nn.Module, states: dict[int, torch.Tensor] | None, tol: float) -> None:
    _, blocks = find_blocks(model)
    if len(blocks) != tt.n_layers:
        raise ValueError(f"model has {len(blocks)} blocks, the token trace was recorded with {tt.n_layers}")
    rep = tt.check(model, states)
    if rep["fp_rel_l2"] > tol:
        raise ValueError(f"recompute does not match the recorded trace: fingerprint relL2 {rep['fp_rel_l2']:.2e} > {tol:.0e} "
                         f"(recorded {tt.dtype} {tt.kernels}; now {rep['dtype']} {kernel_info(model)}). "
                         "Use the same weights in fp32 with the same kernels, or pass tol / verify=False.")
