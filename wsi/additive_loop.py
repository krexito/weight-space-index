"""Additive Looped MoI (`wsi_additive_loop/1`): Modular recurrent layers gated by MoI.

Instead of forcing frozen pretrained backbone layers to re-execute themselves,
``AdditiveLoopedMoI`` keeps all ``N`` backbone layers untouched and inserts a
trainable recurrent block stack (``n_blocks`` cloned from ``blocks[insert_layer:]``)
after layer ``insert_layer``:

    h_0 = h[insert_layer]
    s_0 = h_0
    for t in range(n_loops):
        s_{t+1} = self.blocks(s_t + step_embed[t], past_key_values=step_cache[t])
    alpha = softmax(q_agg · k_proj(RMSNorm(s_k)) / sqrt(d_k) + step_bias[k])
    h_out = h_0 + tanh(out_gate) * (sum_k alpha_k * s_k - h_0)

Key properties:
  - ``n_loops <= 0`` bypasses the additive module completely (bit-exact base).
  - At initialization (``out_gate = 0``), ``h_out == h_0`` bit-for-bit while ``step_bias``
    yields non-zero ``delta = agg - h_0`` so gradients flow immediately at step 0.
  - Each loop iteration ``t`` owns an isolated KV / hybrid cache so autoregressive
    ``use_cache=True`` generation matches full-sequence ``use_cache=False`` exactly.
  - Supports variable-step training (``n_loops`` sampled per batch) and zero-shot
    test-time scaling (evaluating trained weights at higher ``n_loops``).
"""
from __future__ import annotations

import copy
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
from .loopres import (
    _alloc_session_id,
    _build_pass_call,
    _cache_layer_is_empty,
    _clone_empty_cache,
    _compute_state_sha256,
    _detect_pkv_binding,
    _extract_hidden,
    _extract_pkv_from_call,
    _replace_hidden,
    _rmsnorm_no_weight,
)

FORMAT_ADDITIVE_LOOP = "wsi_additive_loop/1"


@dataclass
class AdditiveLoopConfig:
    """Configuration for an ``AdditiveLoopedMoI`` recurrent insertion module."""

    d_model: int
    insert_layer: int
    n_blocks: int = 1
    n_loops: int = 2
    max_steps: int = 8
    d_loop_key: int = 64
    init_gate: float = 0.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.d_model = int(self.d_model)
        self.insert_layer = int(self.insert_layer)
        self.n_blocks = int(self.n_blocks)
        self.n_loops = int(self.n_loops)
        self.max_steps = int(self.max_steps)
        self.d_loop_key = int(self.d_loop_key)
        self.init_gate = float(self.init_gate)
        self.eps = float(self.eps)

        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.insert_layer < 0:
            raise ValueError(f"insert_layer must be >= 0, got {self.insert_layer}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if self.n_loops < 0:
            raise ValueError(f"n_loops must be >= 0, got {self.n_loops}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.d_loop_key < 1:
            raise ValueError(f"d_loop_key must be >= 1, got {self.d_loop_key}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdditiveLoopConfig":
        return cls(**data)


def _infer_d_model(model: nn.Module, fallback_blk: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
    d = int(getattr(text_cfg, "hidden_size", 0) or getattr(cfg, "hidden_size", 0) or 0)
    if d > 0:
        return d
    for p in fallback_blk.parameters():
        if p.ndim >= 1:
            return int(p.shape[-1])
    raise ValueError("Could not infer hidden_size from model.config or block parameters")


def _prepare_addloop_cache_session(orig_pkv: Any, insert_layer: int, session_id: int) -> None:
    """Reset per-step additive caches on ``orig_pkv`` upon fresh prefill or session switch."""
    if orig_pkv is None:
        return
    prev_sid = getattr(orig_pkv, "_wsi_addloop_session", None)
    if prev_sid != session_id or _cache_layer_is_empty(orig_pkv, insert_layer):
        setattr(orig_pkv, "_wsi_addloop_step_caches", {})
        setattr(orig_pkv, "_wsi_addloop_session", session_id)


def _get_or_init_step_cache(
    orig_pkv: Any,
    step_idx: int,
    model: nn.Module,
    insert_layer: int,
    n_blocks: int,
    session_id: int,
) -> Any:
    if orig_pkv is None:
        return None
    caches = getattr(orig_pkv, "_wsi_addloop_step_caches", None)
    if caches is None or getattr(orig_pkv, "_wsi_addloop_session", None) != session_id:
        caches = {}
        setattr(orig_pkv, "_wsi_addloop_step_caches", caches)
        setattr(orig_pkv, "_wsi_addloop_session", session_id)
    if step_idx not in caches:
        caches[step_idx] = _clone_empty_cache(orig_pkv, model, insert_layer, insert_layer + n_blocks)
    return caches[step_idx]


class _AdditiveHookWrapper(nn.Module):
    """Temporary wrapper over ``blocks[insert_layer]`` during ``attach()``."""

    def __init__(
        self,
        orig_blk: nn.Module,
        mod: "AdditiveLoopedMoI",
        model: nn.Module,
        *,
        n_loops: int,
        grad_checkpointing: bool,
        record_stats: bool,
        session_id: int,
    ) -> None:
        super().__init__()
        self.orig_blk = orig_blk
        self.mod = mod
        # Store root model outside nn.Module._modules to avoid module-tree cycles on .eval()/.train().
        object.__setattr__(self, "_model_ref", model)
        self.n_loops = int(n_loops)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.record_stats = bool(record_stats)
        self.session_id = int(session_id)
        self.orig_pkv_binding = _detect_pkv_binding(orig_blk)
        self.block_pkv_bindings = [_detect_pkv_binding(b) for b in mod.blocks]

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.orig_blk, name)

    def _run_subblock(
        self,
        bi: int,
        blk: nn.Module,
        s_args: tuple[Any, ...],
        s_kwargs: dict[str, Any],
        x_in: torch.Tensor,
        step_pkv: Any,
    ) -> torch.Tensor:
        if self.grad_checkpointing and torch.is_grad_enabled() and step_pkv is None:
            def _fn(h_tensor: torch.Tensor) -> Any:
                a_l = list(s_args)
                k_d = dict(s_kwargs)
                if "hidden_states" in k_d or not a_l:
                    k_d["hidden_states"] = h_tensor
                else:
                    a_l[0] = h_tensor
                return blk(*a_l, **k_d)

            out = torch_checkpoint(_fn, x_in, use_reentrant=False)
        else:
            out = blk(*s_args, **s_kwargs)
        return _extract_hidden(out)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        eff_loops = self.n_loops
        if eff_loops <= 0:
            return self.orig_blk(*args, **kwargs)

        cfg = self.mod.config
        pkv_name, pkv_pos = self.orig_pkv_binding
        orig_pkv, _, _ = _extract_pkv_from_call(args, kwargs, pkv_name, pkv_pos)
        _prepare_addloop_cache_session(orig_pkv, cfg.insert_layer, self.session_id)

        orig_out = self.orig_blk(*args, **kwargs)
        h0 = _extract_hidden(orig_out)
        curr = h0
        states: list[torch.Tensor] = [h0]

        for t in range(eff_loops):
            step_idx = min(t, cfg.max_steps - 1)
            se = self.mod.step_embed.weight[step_idx].to(device=curr.device, dtype=curr.dtype)
            x = curr + se.view(1, 1, -1)

            step_pkv = _get_or_init_step_cache(
                orig_pkv,
                t,
                self._model_ref,
                cfg.insert_layer,
                cfg.n_blocks,
                self.session_id,
            )
            for bi, blk in enumerate(self.mod.blocks):
                b_name, b_pos = self.block_pkv_bindings[bi]
                _, eff_n, eff_p = _extract_pkv_from_call(args, kwargs, b_name, b_pos)
                s_args, s_kwargs = _build_pass_call(args, kwargs, x, step_pkv, eff_n, eff_p)
                x = self._run_subblock(bi, blk, s_args, s_kwargs, x, step_pkv)

            curr = x
            states.append(curr)

        h_out = self.mod.aggregate_states(states, record_stats=self.record_stats)
        return _replace_hidden(orig_out, h_out)


class AdditiveLoopedMoI(nn.Module):
    """Modular cloned-layer recurrent stack inserted at ``config.insert_layer``."""

    def __init__(self, config: AdditiveLoopConfig, cloned_blocks: list[nn.Module]) -> None:
        super().__init__()
        if len(cloned_blocks) != config.n_blocks:
            raise ValueError(
                f"Expected {config.n_blocks} cloned_blocks, got {len(cloned_blocks)}"
            )
        self.config = config
        self.blocks = nn.ModuleList(cloned_blocks)

        d = config.d_model
        dk = config.d_loop_key
        self.step_embed = nn.Embedding(config.max_steps, d)
        nn.init.zeros_(self.step_embed.weight)

        self.k_proj = nn.Linear(d, dk, bias=False)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)

        self.q_agg = nn.Parameter(torch.zeros((dk,), dtype=torch.float32))
        self.step_bias = nn.Parameter(torch.zeros((config.max_steps + 1,), dtype=torch.float32))
        with torch.no_grad():
            if config.max_steps >= 1:
                self.step_bias[1:].fill_(0.5)

        self.out_gate = nn.Parameter(torch.full((d,), float(config.init_gate), dtype=torch.float32))
        self._agg_stats: dict[str, Any] | None = None

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        insert_layer: int | None = None,
        n_blocks: int = 1,
        n_loops: int = 2,
        max_steps: int = 8,
        d_loop_key: int = 64,
        init_gate: float = 0.0,
        eps: float = 1e-6,
    ) -> "AdditiveLoopedMoI":
        """Clone ``n_blocks`` consecutive blocks starting at ``insert_layer`` into an isolated module."""
        _, blocks = find_blocks(model)
        n_layers = len(blocks)
        ins = (n_layers // 2) if insert_layer is None else int(insert_layer)
        if not (0 <= ins < n_layers):
            raise ValueError(f"insert_layer={ins} out of bounds for {n_layers} layers")
        if ins + int(n_blocks) > n_layers:
            raise ValueError(
                f"insert_layer ({ins}) + n_blocks ({n_blocks}) exceeds n_layers ({n_layers})"
            )
        d_model = _infer_d_model(model, blocks[ins])
        cloned = [copy.deepcopy(blocks[ins + i]) for i in range(int(n_blocks))]
        for blk in cloned:
            for p in blk.parameters():
                p.requires_grad_(True)
        cfg = AdditiveLoopConfig(
            d_model=d_model,
            insert_layer=ins,
            n_blocks=int(n_blocks),
            n_loops=int(n_loops),
            max_steps=int(max_steps),
            d_loop_key=int(d_loop_key),
            init_gate=float(init_gate),
            eps=float(eps),
        )
        return cls(cfg, cloned)

    def reset_telemetry(self) -> None:
        self._agg_stats = None

    @torch.no_grad()
    def _record_alpha(self, alpha_bt: torch.Tensor) -> None:
        flat = alpha_bt.detach().reshape(-1, alpha_bt.shape[-1]).double()
        n_tok = int(flat.shape[0])
        if n_tok == 0:
            return
        sum_a = flat.sum(dim=0).cpu()
        sum_a2 = (flat * flat).sum(dim=0).cpu()
        ent = float((-(flat * flat.clamp_min(1e-12).log()).sum(dim=-1)).sum().cpu().item())
        if self._agg_stats is None or self._agg_stats["sum_a"].shape != sum_a.shape:
            self._agg_stats = {"sum_a": sum_a, "sum_a2": sum_a2, "sum_ent": ent, "n_tok": n_tok}
        else:
            self._agg_stats["sum_a"] += sum_a
            self._agg_stats["sum_a2"] += sum_a2
            self._agg_stats["sum_ent"] += ent
            self._agg_stats["n_tok"] += n_tok

    def telemetry_summary(self) -> dict[str, Any]:
        """Return aggregated state-mixing attention telemetry across tokens."""
        if self._agg_stats is None or int(self._agg_stats["n_tok"]) == 0:
            return {"n_tokens": 0, "alpha_mean": [], "alpha_std": [], "entropy": 0.0}
        bkt = self._agg_stats
        n = max(1, int(bkt["n_tok"]))
        mean_t = bkt["sum_a"] / float(n)
        var_t = (bkt["sum_a2"] / float(n) - mean_t * mean_t).clamp_min(0.0)
        return {
            "n_tokens": int(bkt["n_tok"]),
            "alpha_mean": [round(float(x), 6) for x in mean_t.tolist()],
            "alpha_std": [round(float(x), 6) for x in var_t.sqrt().tolist()],
            "entropy": round(float(bkt["sum_ent"]) / float(n), 6),
        }

    def aggregate_states(self, states: list[torch.Tensor], *, record_stats: bool = False) -> torch.Tensor:
        """Mix ``[h_0, s_1, ..., s_T]`` and inject via zero-initialized ``tanh(out_gate)``."""
        h0 = states[0]
        target_dtype = h0.dtype
        v = torch.stack([s.float() for s in states], dim=-2)  # [B, S, T+1, D]
        k_norm = _rmsnorm_no_weight(v, self.config.eps)
        k_proj_w = self.k_proj.weight.float()
        keys = F.linear(k_norm, k_proj_w)  # [B, S, T+1, dk]
        content_logits = torch.einsum("btmd,d->btm", keys, self.q_agg.float()) / math.sqrt(
            float(self.config.d_loop_key)
        )
        m = len(states)
        idx = torch.arange(m, device=v.device).clamp_max(self.config.max_steps)
        bias = self.step_bias.index_select(0, idx).float().view(1, 1, m)
        alpha = F.softmax(content_logits + bias, dim=-1)
        if record_stats:
            self._record_alpha(alpha)

        agg = torch.einsum("btm,btmd->btd", alpha, v)
        delta = agg - h0.float()
        gate = torch.tanh(self.out_gate.float()).view(1, 1, -1)
        return (h0.float() + gate * delta).to(dtype=target_dtype)

    def num_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def param_bytes(self) -> int:
        """Exact in-memory parameter footprint in bytes."""
        return sum(int(p.numel() * p.element_size()) for p in self.parameters())

    def payload_bytes(self) -> int:
        """Serialized bf16 payload size in bytes (2 bytes/parameter)."""
        return self.num_parameters() * 2

    def payload_mb(self) -> float:
        return self.payload_bytes() / (1024.0 * 1024.0)

    def save(self, path: str | Path) -> Path:
        from safetensors.torch import save_file

        sd = {k: v.detach().cpu().contiguous().clone() for k, v in sorted(self.state_dict().items())}
        manifest = {
            "format": FORMAT_ADDITIVE_LOOP,
            "config": self.config.to_dict(),
            "sha256": _compute_state_sha256(sd),
            "param_keys": sorted(sd),
        }
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        save_file(sd, str(out_p), metadata={"wsi_additive_loop": json.dumps(manifest)})
        return out_p

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        template_model: nn.Module,
        verify: bool = True,
    ) -> "AdditiveLoopedMoI":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as fh:
            meta_raw = fh.metadata() or {}
            if "wsi_additive_loop" not in meta_raw:
                raise ValueError(
                    f"Not an AdditiveLoopedMoI safetensors file (missing metadata['wsi_additive_loop']): {path}"
                )
            manifest = json.loads(meta_raw["wsi_additive_loop"])
            if manifest.get("format") != FORMAT_ADDITIVE_LOOP:
                raise ValueError(
                    f"Unsupported format {manifest.get('format')!r}, expected {FORMAT_ADDITIVE_LOOP!r}"
                )
            cfg = AdditiveLoopConfig.from_dict(manifest["config"])
            expected_keys = list(manifest.get("param_keys") or [])
            file_keys = sorted(fh.keys())
            if sorted(expected_keys) != file_keys:
                raise ValueError(
                    f"Tensor key mismatch in {path}: expected {sorted(expected_keys)}, got {file_keys}"
                )
            tensors = {k: fh.get_tensor(k).clone() for k in expected_keys}

        if verify:
            actual_sha = _compute_state_sha256(tensors)
            if actual_sha != manifest.get("sha256"):
                raise ValueError(f"SHA-256 integrity check failed for {path}")

        _, blocks = find_blocks(template_model)
        if cfg.insert_layer + cfg.n_blocks > len(blocks):
            raise ValueError(
                f"Template model has {len(blocks)} blocks, cannot instantiate "
                f"[{cfg.insert_layer}, {cfg.insert_layer + cfg.n_blocks})"
            )
        cloned = [copy.deepcopy(blocks[cfg.insert_layer + i]) for i in range(cfg.n_blocks)]
        mod = cls(cfg, cloned)
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
    ) -> Iterator["AdditiveLoopedMoI"]:
        """Temporarily wrap ``blocks[insert_layer]`` on ``model`` and restore unconditionally."""
        _, blocks = find_blocks(model)
        ins = self.config.insert_layer
        if ins >= len(blocks):
            raise ValueError(f"insert_layer={ins} exceeds model block count {len(blocks)}")
        eff_loops = int(self.config.n_loops if n_loops is None else n_loops)
        if eff_loops < 0:
            raise ValueError(f"n_loops must be >= 0, got {eff_loops}")

        orig_blk = blocks[ins]
        wrapper = _AdditiveHookWrapper(
            orig_blk,
            self,
            model,
            n_loops=eff_loops,
            grad_checkpointing=grad_checkpointing,
            record_stats=record_stats,
            session_id=_alloc_session_id(),
        )
        try:
            blocks[ins] = wrapper
            yield self
        finally:
            blocks[ins] = orig_blk
