"""LoopRes (Loop Residuals, arXiv:2611.31094) for frozen / PEFT + MoI-routed LLMs.

Implements an identity-preserving, KV-cache-safe Loop Residual controller that
re-runs a contiguous core span ``blocks[core_start:core_end]`` ``n_loops`` times
with:
  - Untouched Pass 0 execution over native ``past_key_values``
  - Isolated per-pass KV / recurrent caches for Passes ``p >= 1``
  - Windowed structural slot schema (``window = W``) shared across passes so a
    module trained at ``n_loops = 2`` extrapolates zero-shot to ``n_loops = 3+``
  - Parameter-free FP32 RMSNorm content keys + trainable loop-aware keys
    (``mode="loopres"``) or bias-only gating ablation (``mode="gated"``)
  - Safetensors persistence with deterministic SHA-256 integrity checks
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .capture import find_blocks

FORMAT_LOOPRES = "wsi_loopres/1"
_NEXT_SESSION_ID = 1


def _alloc_session_id() -> int:
    global _NEXT_SESSION_ID
    sid = _NEXT_SESSION_ID
    _NEXT_SESSION_ID += 1
    return sid


def default_core_span(n_layers: int) -> tuple[int, int]:
    """Return default ``(core_start, core_end)`` prelude/core/coda boundaries.

    Examples:
        - 28 layers (Qwen3-0.6B/4B) -> (6, 22)
        - 24 layers (Qwen3.5-0.8B / LFM3) -> (6, 18)
        - 6 layers -> (1, 5)
        - 2 layers -> (0, 2)
    """
    n = int(n_layers)
    if n < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if n < 4:
        return (0, n)
    if n < 8:
        return (1, n - 1)
    margin = max(1, round(0.23 * n))
    while n - 2 * margin < 2 and margin > 1:
        margin -= 1
    return (margin, n - margin)


@dataclass
class LoopResConfig:
    """Configuration for a windowed LoopRes module over ``blocks[core_start:core_end]``."""

    d_model: int
    core_start: int
    core_end: int
    n_loops: int = 2
    window: int = 2
    d_loop_key: int = 64
    mode: str = "loopres"
    init_pass0_bias: float = 3.0
    init_reentry_anchor_bias: float = 1.5
    init_inner_curr_bias: float = 1.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.d_model = int(self.d_model)
        self.core_start = int(self.core_start)
        self.core_end = int(self.core_end)
        self.n_loops = int(self.n_loops)
        self.window = int(self.window)
        self.d_loop_key = int(self.d_loop_key)
        self.mode = str(self.mode)
        self.init_pass0_bias = float(self.init_pass0_bias)
        self.init_reentry_anchor_bias = float(self.init_reentry_anchor_bias)
        self.init_inner_curr_bias = float(self.init_inner_curr_bias)
        self.eps = float(self.eps)

        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if not (0 <= self.core_start < self.core_end):
            raise ValueError(f"Invalid core span [{self.core_start}, {self.core_end})")
        if self.n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {self.n_loops}")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.d_loop_key < 1:
            raise ValueError(f"d_loop_key must be >= 1, got {self.d_loop_key}")
        if self.mode not in ("loopres", "gated"):
            raise ValueError(f"mode must be 'loopres' or 'gated', got {self.mode!r}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    @property
    def n_core_layers(self) -> int:
        return self.core_end - self.core_start

    @property
    def max_entry_slots(self) -> int:
        return 1 + self.window

    @property
    def max_inner_slots(self) -> int:
        return 1 + 2 * self.window

    @property
    def max_layer_slots(self) -> int:
        return 1 + 2 * self.window

    @property
    def max_coda_slots(self) -> int:
        return 1 + self.window

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopResConfig":
        return cls(**data)


def _rmsnorm_no_weight(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free FP32 RMSNorm over the trailing hidden dimension."""
    vf = v.float()
    return vf * torch.rsqrt(vf.pow(2).mean(dim=-1, keepdim=True) + float(eps))


def _extract_hidden(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"Unsupported block output type: {type(out).__name__}")


def _replace_hidden(out: Any, new_h: torch.Tensor) -> Any:
    if isinstance(out, torch.Tensor):
        return new_h
    if isinstance(out, tuple) and out:
        return (new_h, *out[1:])
    raise TypeError(f"Unsupported block output type: {type(out).__name__}")


_PKV_ARG_NAMES = ("past_key_values", "past_key_value", "cache", "past_cache")


def _is_cache_obj(obj: Any) -> bool:
    """Return True if ``obj`` looks like a HuggingFace Cache instance."""
    return obj is not None and callable(getattr(obj, "update", None)) and callable(getattr(obj, "get_seq_length", None))


def _detect_pkv_binding(orig_blk: nn.Module) -> tuple[str | None, int | None]:
    """Detect KV cache parameter name and optional positional index on ``orig_blk.forward``."""
    try:
        sig = inspect.signature(orig_blk.forward)
    except (TypeError, ValueError):
        return (None, None)
    params = list(sig.parameters.values())
    for idx, p in enumerate(params):
        if p.name in _PKV_ARG_NAMES:
            pos = idx if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) else None
            return (p.name, pos)
    return (None, None)


def _extract_pkv_from_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    pkv_name: str | None,
    pkv_pos: int | None,
) -> tuple[Any, str | None, int | None]:
    for cand in _PKV_ARG_NAMES:
        if cand in kwargs:
            return kwargs[cand], cand, None
    for k, v in kwargs.items():
        if _is_cache_obj(v):
            return v, k, None
    if pkv_name is not None and pkv_pos is not None and pkv_pos < len(args):
        return args[pkv_pos], pkv_name, pkv_pos
    for idx, v in enumerate(args):
        if _is_cache_obj(v):
            return v, pkv_name or "past_key_values", idx
    return None, pkv_name, pkv_pos


def _build_pass_call(
    saved_args: tuple[Any, ...],
    saved_kwargs: dict[str, Any],
    new_h: torch.Tensor,
    pass_pkv: Any,
    pkv_name: str | None,
    pkv_pos: int | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args_l = list(saved_args)
    kw_d = dict(saved_kwargs)
    if "hidden_states" in kw_d or not args_l:
        kw_d["hidden_states"] = new_h
    else:
        args_l[0] = new_h

    if pkv_name is not None:
        if pkv_name in kw_d:
            kw_d[pkv_name] = pass_pkv
        elif pkv_pos is not None and pkv_pos < len(args_l):
            args_l[pkv_pos] = pass_pkv
        elif pass_pkv is not None:
            kw_d[pkv_name] = pass_pkv
    return tuple(args_l), kw_d


def _cache_layer_is_empty(pkv: Any, layer_idx: int) -> bool:
    """Return True if ``pkv`` has 0 cached tokens recorded at ``layer_idx``."""
    if pkv is None:
        return True
    # Hybrid caches (e.g. Qwen3_5DynamicCache) store conv/recurrent states on linear_attention layers
    for attr in ("conv_states", "recurrent_states"):
        st = getattr(pkv, attr, None)
        if isinstance(st, (list, tuple)) and layer_idx < len(st) and st[layer_idx] is not None:
            return False
    get_len = getattr(pkv, "get_seq_length", None)
    if callable(get_len):
        try:
            sig = inspect.signature(get_len)
            if len(sig.parameters) >= 1:
                return int(get_len(layer_idx)) == 0
        except (TypeError, ValueError, IndexError):
            pass
    if hasattr(pkv, "layers"):
        layers = getattr(pkv, "layers")
        if len(layers) <= layer_idx:
            return True
        lyr = layers[layer_idx]
        lyr_len = getattr(lyr, "get_seq_length", None)
        if callable(lyr_len):
            try:
                return int(lyr_len()) == 0
            except Exception:
                pass
        return getattr(lyr, "keys", None) is None
    if hasattr(pkv, "key_cache"):
        kc = getattr(pkv, "key_cache")
        if len(kc) <= layer_idx:
            return True
        k_t = kc[layer_idx]
        return k_t is None or (isinstance(k_t, torch.Tensor) and k_t.numel() == 0)
    if callable(get_len):
        try:
            return int(get_len()) == 0
        except Exception:
            pass
    return False


def _clone_empty_cache(orig_pkv: Any, model: nn.Module, core_start: int, core_end: int) -> Any:
    """Instantiate an empty per-pass cache matching ``orig_pkv``'s class and hybrid config."""
    if orig_pkv is None:
        return None
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
    cloned = None
    if text_cfg is not None:
        try:
            cloned = type(orig_pkv)(config=text_cfg)
        except TypeError:
            cloned = None
    if cloned is None and cfg is not None:
        try:
            cloned = type(orig_pkv)(config=cfg)
        except TypeError:
            cloned = None
    if cloned is None:
        cloned = type(orig_pkv)()

    if hasattr(cloned, "last_linear_layer") and text_cfg is not None:
        layer_types = getattr(text_cfg, "layer_types", None)
        if isinstance(layer_types, (list, tuple)):
            lin_idxs = [
                li
                for li in range(core_start, min(core_end, len(layer_types)))
                if layer_types[li] == "linear_attention"
            ]
            if lin_idxs:
                cloned.last_linear_layer = lin_idxs[-1]
    return cloned


def _prepare_pass0_cache_session(orig_pkv: Any, core_start: int, session_id: int) -> None:
    """Reset per-pass sub-caches on ``orig_pkv`` on fresh prefill or session change."""
    if orig_pkv is None:
        return
    prev_sid = getattr(orig_pkv, "_wsi_loopres_session", None)
    is_empty = True
    get_len = getattr(orig_pkv, "get_seq_length", None)
    if callable(get_len):
        try:
            if int(get_len()) > 0:
                is_empty = False
        except Exception:
            is_empty = _cache_layer_is_empty(orig_pkv, core_start)
    else:
        is_empty = _cache_layer_is_empty(orig_pkv, core_start)
    if prev_sid != session_id or is_empty:
        setattr(orig_pkv, "_wsi_loopres_pass_caches", {})
        setattr(orig_pkv, "_wsi_loopres_session", session_id)


def _get_or_init_pass_cache(
    orig_pkv: Any,
    pass_idx: int,
    model: nn.Module,
    core_start: int,
    core_end: int,
    session_id: int,
) -> Any:
    if orig_pkv is None:
        return None
    caches = getattr(orig_pkv, "_wsi_loopres_pass_caches", None)
    if caches is None or getattr(orig_pkv, "_wsi_loopres_session", None) != session_id:
        caches = {}
        setattr(orig_pkv, "_wsi_loopres_pass_caches", caches)
        setattr(orig_pkv, "_wsi_loopres_session", session_id)
    if pass_idx not in caches:
        caches[pass_idx] = _clone_empty_cache(orig_pkv, model, core_start, core_end)
    return caches[pass_idx]


def _compute_state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Deterministic SHA-256 over sorted LoopRes parameters (names, shapes, dtypes, bytes)."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().contiguous().cpu()
        h.update(k.encode("utf-8"))
        h.update(b"|")
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(b"|")
        h.update(str(t.dtype).encode("utf-8"))
        h.update(b"|")
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() > 0 else b""
        h.update(raw)
    return h.hexdigest()


class _ForwardLoopState:
    __slots__ = ("orig_pkv", "pkv_name", "pkv_pos", "saved_args", "saved_kwargs", "h_in_pass0")

    def __init__(self, n_core: int) -> None:
        self.orig_pkv: Any = None
        self.pkv_name: str | None = None
        self.pkv_pos: int | None = None
        self.saved_args: list[tuple[Any, ...] | None] = [None] * n_core
        self.saved_kwargs: list[dict[str, Any] | None] = [None] * n_core
        self.h_in_pass0: list[torch.Tensor | None] = [None] * n_core


class _LoopResController:
    def __init__(
        self,
        loopres: "LoopResModule",
        model: nn.Module,
        orig_blocks: list[nn.Module],
        *,
        n_loops: int,
        grad_checkpointing: bool,
        record_stats: bool,
    ) -> None:
        self.loopres = loopres
        self.model = model
        self.orig_blocks = orig_blocks
        self.n_core = len(orig_blocks)
        self.n_loops = int(n_loops)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.record_stats = bool(record_stats)
        self.session_id = _alloc_session_id()
        self.active_state: _ForwardLoopState | None = None
        self.pkv_bindings = [_detect_pkv_binding(b) for b in orig_blocks]

    def run_orig_block(
        self,
        ci: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        h_in: torch.Tensor,
        *,
        allow_ckpt: bool,
    ) -> Any:
        orig_blk = self.orig_blocks[ci]
        if allow_ckpt and self.grad_checkpointing and torch.is_grad_enabled():
            needs_ckpt = bool(h_in.requires_grad) or any(p.requires_grad for p in orig_blk.parameters())
            if not needs_ckpt:
                return orig_blk(*args, **kwargs)

            def _fn(h_tensor: torch.Tensor) -> Any:
                args_l = list(args)
                kw_d = dict(kwargs)
                if "hidden_states" in kw_d or not args_l:
                    kw_d["hidden_states"] = h_tensor
                else:
                    args_l[0] = h_tensor
                return orig_blk(*args_l, **kw_d)

            return torch_checkpoint(_fn, h_in, use_reentrant=False)
        return orig_blk(*args, **kwargs)


class _CoreLoopBlockWrapper(nn.Module):
    """Transparent wrapper installed over ``blocks[core_start:core_end]`` during ``attach()``."""

    def __init__(self, orig_blk: nn.Module, ci: int, ctrl: _LoopResController) -> None:
        super().__init__()
        self.orig_blk = orig_blk
        self.ci = int(ci)
        self.ctrl = ctrl

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.orig_blk, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ci = self.ci
        ctrl = self.ctrl
        cfg = ctrl.loopres.config
        n_core = ctrl.n_core
        h_in = kwargs["hidden_states"] if ("hidden_states" in kwargs or not args) else args[0]

        pkv_name, pkv_pos = ctrl.pkv_bindings[ci]
        call_pkv, eff_name, eff_pos = _extract_pkv_from_call(args, kwargs, pkv_name, pkv_pos)

        if ci == 0:
            _prepare_pass0_cache_session(call_pkv, cfg.core_start, ctrl.session_id)
            state = _ForwardLoopState(n_core)
            state.orig_pkv = call_pkv
            state.pkv_name = eff_name
            state.pkv_pos = eff_pos
            ctrl.active_state = state
        else:
            state = ctrl.active_state
            if state is None:
                raise RuntimeError("LoopRes encountered core layer > 0 without active Pass-0 state")

        state.saved_args[ci] = tuple(args)
        state.saved_kwargs[ci] = dict(kwargs)
        state.h_in_pass0[ci] = h_in

        # Only checkpoint when KV cache is inactive (`call_pkv is None`); inference uses `no_grad`.
        out_0 = ctrl.run_orig_block(ci, args, kwargs, h_in, allow_ckpt=(call_pkv is None))

        if ci < n_core - 1:
            return out_0

        # End of core on Pass 0: run passes 1..active_n_loops-1 and apply Coda LoopRes mixing.
        ctrl.active_state = None
        active_loops = ctrl.n_loops
        if active_loops <= 1:
            return out_0

        h_end_0 = _extract_hidden(out_0)
        h_in_hist: list[list[torch.Tensor]] = []
        for idx_c in range(n_core):
            h_p0 = state.h_in_pass0[idx_c]
            if h_p0 is None:
                raise RuntimeError(f"Missing Pass-0 input for core layer index {idx_c}")
            h_in_hist.append([h_p0])
        h_end_hist: list[torch.Tensor] = [h_end_0]

        W = cfg.window
        for p in range(1, active_loops):
            pass_pkv = _get_or_init_pass_cache(
                state.orig_pkv,
                p,
                ctrl.model,
                cfg.core_start,
                cfg.core_end,
                ctrl.session_id,
            )
            h_curr: torch.Tensor | None = None
            for idx_c in range(n_core):
                max_lag = min(p, W)
                lags = range(1, max_lag + 1)
                if idx_c == 0:
                    sources = [h_end_hist[p - 1]] + [h_in_hist[0][p - lag] for lag in lags]
                    slot_ids = [0] + list(lags)
                else:
                    assert h_curr is not None
                    sources = (
                        [h_curr]
                        + [h_in_hist[idx_c][p - lag] for lag in lags]
                        + [h_end_hist[p - lag] for lag in lags]
                    )
                    slot_ids = [0] + list(lags) + [W + lag for lag in lags]

                x_in = ctrl.loopres.mix_layer(
                    idx_c,
                    sources,
                    slot_ids,
                    pass_idx=p,
                    record_stats=ctrl.record_stats,
                )
                h_in_hist[idx_c].append(x_in)

                s_args = state.saved_args[idx_c]
                s_kwargs = state.saved_kwargs[idx_c]
                assert s_args is not None and s_kwargs is not None
                p_name, p_pos = ctrl.pkv_bindings[idx_c]
                _, eff_n, eff_p = _extract_pkv_from_call(s_args, s_kwargs, p_name, p_pos)
                p_args, p_kwargs = _build_pass_call(s_args, s_kwargs, x_in, pass_pkv, eff_n, eff_p)
                blk_out = ctrl.run_orig_block(idx_c, p_args, p_kwargs, x_in, allow_ckpt=(pass_pkv is None))
                h_curr = _extract_hidden(blk_out)

            assert h_curr is not None
            h_end_hist.append(h_curr)

        coda_lags = range(1, min(active_loops - 1, W) + 1)
        coda_sources = [h_end_hist[0]] + [h_end_hist[active_loops - lag] for lag in coda_lags]
        coda_slot_ids = [0] + list(coda_lags)
        h_final = ctrl.loopres.mix_coda(coda_sources, coda_slot_ids, record_stats=ctrl.record_stats)
        return _replace_hidden(out_0, h_final)


class LoopResModule(nn.Module):
    """Lightweight, attachable Loop Residuals module for frozen/PEFT LLMs."""

    def __init__(self, config: LoopResConfig) -> None:
        super().__init__()
        self.config = config
        n_core = config.n_core_layers
        d = config.d_model
        dk = config.d_loop_key
        l_slots = config.max_layer_slots
        c_slots = config.max_coda_slots

        self.layer_bias = nn.Parameter(torch.zeros((n_core, l_slots), dtype=torch.float32))
        self.coda_bias = nn.Parameter(torch.zeros((c_slots,), dtype=torch.float32))
        with torch.no_grad():
            self.layer_bias[0, 1] = float(config.init_reentry_anchor_bias)
            if n_core > 1:
                self.layer_bias[1:, 0] = float(config.init_inner_curr_bias)
            self.coda_bias[0] = float(config.init_pass0_bias)

        if config.mode == "loopres":
            self.layer_q = nn.Parameter(torch.zeros((n_core, d), dtype=torch.float32))
            self.layer_q_loop = nn.Parameter(torch.zeros((n_core, dk), dtype=torch.float32))
            self.layer_k_loop = nn.Parameter(torch.empty((n_core, l_slots, dk), dtype=torch.float32))
            nn.init.normal_(self.layer_k_loop, mean=0.0, std=1.0 / math.sqrt(dk))

            self.coda_q = nn.Parameter(torch.zeros((d,), dtype=torch.float32))
            self.coda_q_loop = nn.Parameter(torch.zeros((dk,), dtype=torch.float32))
            self.coda_k_loop = nn.Parameter(torch.empty((c_slots, dk), dtype=torch.float32))
            nn.init.normal_(self.coda_k_loop, mean=0.0, std=1.0 / math.sqrt(dk))
        else:
            self.register_parameter("layer_q", None)
            self.register_parameter("layer_q_loop", None)
            self.register_parameter("layer_k_loop", None)
            self.register_parameter("coda_q", None)
            self.register_parameter("coda_q_loop", None)
            self.register_parameter("coda_k_loop", None)

        self._layer_stats: dict[tuple[int, int], dict[str, Any]] = {}
        self._coda_stats: dict[str, Any] | None = None

    def reset_telemetry(self) -> None:
        self._layer_stats = {}
        self._coda_stats = None

    @torch.no_grad()
    def _record_alpha(self, bucket_key: tuple[int, int] | None, alpha_bt: torch.Tensor) -> None:
        # alpha_bt: [B, T, m] in FP32
        flat = alpha_bt.detach().reshape(-1, alpha_bt.shape[-1]).double()
        n_tok = int(flat.shape[0])
        if n_tok == 0:
            return
        sum_a = flat.sum(dim=0).cpu()
        sum_a2 = (flat * flat).sum(dim=0).cpu()
        ent = float((-(flat * flat.clamp_min(1e-12).log()).sum(dim=-1)).sum().cpu().item())

        if bucket_key is None:
            if self._coda_stats is None or self._coda_stats["sum_a"].shape != sum_a.shape:
                self._coda_stats = {"sum_a": sum_a, "sum_a2": sum_a2, "sum_ent": ent, "n_tok": n_tok}
            else:
                self._coda_stats["sum_a"] += sum_a
                self._coda_stats["sum_a2"] += sum_a2
                self._coda_stats["sum_ent"] += ent
                self._coda_stats["n_tok"] += n_tok
            return

        cur = self._layer_stats.get(bucket_key)
        if cur is None or cur["sum_a"].shape != sum_a.shape:
            self._layer_stats[bucket_key] = {"sum_a": sum_a, "sum_a2": sum_a2, "sum_ent": ent, "n_tok": n_tok}
        else:
            cur["sum_a"] += sum_a
            cur["sum_a2"] += sum_a2
            cur["sum_ent"] += ent
            cur["n_tok"] += n_tok

    @staticmethod
    def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
        n = max(1, int(bucket["n_tok"]))
        mean_t = bucket["sum_a"] / float(n)
        var_t = (bucket["sum_a2"] / float(n) - mean_t * mean_t).clamp_min(0.0)
        return {
            "n_tokens": int(bucket["n_tok"]),
            "alpha_mean": [round(float(x), 6) for x in mean_t.tolist()],
            "alpha_std": [round(float(x), 6) for x in var_t.sqrt().tolist()],
            "entropy": round(float(bucket["sum_ent"]) / float(n), 6),
        }

    def telemetry_summary(self) -> dict[str, Any]:
        """Return aggregated per-layer and Coda routing telemetry as plain Python types."""
        coda = (
            self._finalize_bucket(self._coda_stats)
            if self._coda_stats is not None
            else {"n_tokens": 0, "alpha_mean": [], "alpha_std": [], "entropy": 0.0}
        )
        layers: dict[str, Any] = {}
        for (ci, pass_idx), bkt in sorted(self._layer_stats.items()):
            layers[f"core_{ci}_pass_{pass_idx}"] = {
                "layer_idx": self.config.core_start + ci,
                "core_idx": ci,
                "pass_idx": pass_idx,
                **self._finalize_bucket(bkt),
            }
        return {
            "n_tokens": coda["n_tokens"],
            "coda_alpha_mean": coda["alpha_mean"],
            "coda_alpha_std": coda["alpha_std"],
            "coda_entropy": coda["entropy"],
            "layers": layers,
        }

    def _mix(
        self,
        sources: list[torch.Tensor],
        slot_ids: list[int],
        *,
        bias: torch.Tensor,
        q: torch.Tensor | None,
        q_loop: torch.Tensor | None,
        k_loop: torch.Tensor | None,
        stats_key: tuple[int, int] | None,
        record_stats: bool,
    ) -> torch.Tensor:
        if len(sources) != len(slot_ids):
            raise ValueError(f"sources ({len(sources)}) and slot_ids ({len(slot_ids)}) length mismatch")
        if not sources:
            raise ValueError("sources must be non-empty")

        target_dtype = sources[0].dtype
        v = torch.stack([s.float() for s in sources], dim=-2)  # [B, T, m, d]
        idx_t = torch.tensor(slot_ids, dtype=torch.long, device=v.device)
        bias_sel = bias.index_select(0, idx_t).float()  # [m]

        if self.config.mode == "gated" or q is None or q_loop is None or k_loop is None:
            alpha_1d = F.softmax(bias_sel, dim=-1)  # [m]
            mixed = torch.einsum("m,btmd->btd", alpha_1d, v)
            if record_stats:
                alpha_bt = alpha_1d.view(1, 1, -1).expand(v.shape[0], v.shape[1], -1)
                self._record_alpha(stats_key, alpha_bt)
            return mixed.to(dtype=target_dtype)

        k_content = _rmsnorm_no_weight(v, self.config.eps)  # [B, T, m, d]
        content_logits = torch.einsum("btmd,d->btm", k_content, q.float()) / math.sqrt(float(self.config.d_model))
        k_sel = k_loop.index_select(0, idx_t).float()  # [m, dk]
        loop_logits = torch.einsum("md,d->m", k_sel, q_loop.float()).view(1, 1, -1)
        logits = content_logits + loop_logits + bias_sel.view(1, 1, -1)
        alpha = F.softmax(logits, dim=-1)  # [B, T, m]
        mixed = torch.einsum("btm,btmd->btd", alpha, v)
        if record_stats:
            self._record_alpha(stats_key, alpha)
        return mixed.to(dtype=target_dtype)

    def mix_layer(
        self,
        ci: int,
        sources: list[torch.Tensor],
        slot_ids: list[int],
        *,
        pass_idx: int = 1,
        record_stats: bool = False,
    ) -> torch.Tensor:
        return self._mix(
            sources,
            slot_ids,
            bias=self.layer_bias[ci],
            q=self.layer_q[ci] if self.layer_q is not None else None,
            q_loop=self.layer_q_loop[ci] if self.layer_q_loop is not None else None,
            k_loop=self.layer_k_loop[ci] if self.layer_k_loop is not None else None,
            stats_key=(int(ci), int(pass_idx)),
            record_stats=record_stats,
        )

    def mix_coda(
        self,
        sources: list[torch.Tensor],
        slot_ids: list[int],
        *,
        record_stats: bool = False,
    ) -> torch.Tensor:
        return self._mix(
            sources,
            slot_ids,
            bias=self.coda_bias,
            q=self.coda_q,
            q_loop=self.coda_q_loop,
            k_loop=self.coda_k_loop,
            stats_key=None,
            record_stats=record_stats,
        )

    def num_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def fp32_bytes(self) -> int:
        return sum(int(p.numel() * p.element_size()) for p in self.parameters())

    def payload_bytes(self) -> int:
        """Serialized bf16 payload size in bytes (2 bytes/parameter)."""
        return self.num_parameters() * 2

    def payload_kb(self) -> float:
        return self.payload_bytes() / 1024.0

    def save(self, path: str | Path) -> Path:
        from safetensors.torch import save_file

        sd = {k: v.detach().cpu().contiguous().clone() for k, v in sorted(self.state_dict().items())}
        manifest = {
            "format": FORMAT_LOOPRES,
            "config": self.config.to_dict(),
            "sha256": _compute_state_sha256(sd),
            "param_keys": sorted(sd),
        }
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        save_file(sd, str(out_p), metadata={"wsi_loopres": json.dumps(manifest)})
        return out_p

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> "LoopResModule":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as fh:
            meta_raw = fh.metadata() or {}
            if "wsi_loopres" not in meta_raw:
                raise ValueError(f"Not a LoopRes safetensors file (missing metadata['wsi_loopres']): {path}")
            manifest = json.loads(meta_raw["wsi_loopres"])
            if manifest.get("format") != FORMAT_LOOPRES:
                raise ValueError(f"Unsupported format {manifest.get('format')!r}, expected {FORMAT_LOOPRES!r}")
            cfg = LoopResConfig.from_dict(manifest["config"])
            expected_keys = list(manifest.get("param_keys") or [])
            file_keys = sorted(fh.keys())
            if sorted(expected_keys) != file_keys:
                raise ValueError(f"Tensor key mismatch in {path}: expected {sorted(expected_keys)}, got {file_keys}")
            # Clone before safe_open closes to avoid Windows mmap invalidation
            tensors = {k: fh.get_tensor(k).clone() for k in expected_keys}

        mod = cls(cfg)
        if verify:
            actual_sha = _compute_state_sha256(tensors)
            if actual_sha != manifest.get("sha256"):
                raise ValueError(f"SHA-256 integrity check failed for {path}")
        mod.load_state_dict(tensors, strict=True)
        return mod

    @contextmanager
    def attach(
        self,
        model: nn.Module,
        *,
        n_loops: int | None = None,
        grad_checkpointing: bool = False,
        record_stats: bool = False,
    ) -> Iterator["LoopResModule"]:
        """Temporarily wrap ``blocks[core_start:core_end]`` on ``model`` and always restore on exit."""
        _, blocks = find_blocks(model)
        cs, ce = self.config.core_start, self.config.core_end
        if ce > len(blocks):
            raise ValueError(f"core_end={ce} exceeds model block count {len(blocks)}")
        eff_loops = int(self.config.n_loops if n_loops is None else n_loops)
        if eff_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {eff_loops}")

        orig_blocks = [blocks[li] for li in range(cs, ce)]
        ctrl = _LoopResController(
            self,
            model,
            orig_blocks,
            n_loops=eff_loops,
            grad_checkpointing=grad_checkpointing,
            record_stats=record_stats,
        )
        try:
            for ci, li in enumerate(range(cs, ce)):
                blocks[li] = _CoreLoopBlockWrapper(orig_blocks[ci], ci, ctrl)
            yield self
        finally:
            ctrl.active_state = None
            for ci, li in enumerate(range(cs, ce)):
                blocks[li] = orig_blocks[ci]
