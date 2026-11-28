"""Token-level dense & compact activation tracing (`CompactTrace` + differential X-ray).

Pairs `wsi.capture.trace_all` / `capture_compact` with `WeightSpaceIndex` geometry when global averages fail:
  1. `diff_traces(model, ids_a, ids_b, wsi=...)`: X-ray across two prompts (e.g., neutral vs. pushback turn,
     clean vs. trigger prompt); uses `CompactTrace` + `diff_compact` by default (`compact=True`) or full
     uncompressed `trace_all` with `compact=False`.
  2. `capture_compact(model, ids, ...)`: scalable token-by-token tracing defaulting to the best Pareto
     profile from our Colab sweeps (`INT8` `group_size=64` + Walsh-Hadamard `64` residuals, adaptive
     `Top-K` `E2M1 FP4` bitmask/bitpack MLP units at 95% energy, `INT8` pre-`W_O` head vectors +
     post-`W_O` L2 norms, sparse `Top-16` attention links, and `INT8` KV/DeltaNet cache states).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .capture import (
    _is_excluded_backbone_path,
    _is_float_weight,
    _is_linear,
    _is_routed_expert_path,
    find_blocks,
    find_mlps,
    trace_all,
)
from .moipatch import _expected_block_passes

_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_POW3 = (1, 3, 9, 27, 81)
_BIT8 = (1, 2, 4, 8, 16, 32, 64, 128)


# ----------------------------------------------------------------------------------------------- Hadamard + block quantization
@lru_cache(maxsize=16)
def _hadamard_matrix(n: int) -> torch.Tensor:
    """Normalized n x n Sylvester Hadamard matrix (H @ H.T == I, symmetric orthogonal)."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard block size must be power of 2, got {n}")
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / (n ** 0.5)


def resolve_hadamard_block(dim: int, requested: int) -> int:
    """Largest power-of-2 <= requested that divides `dim` (0 disables Hadamard rotation)."""
    if requested <= 0:
        return 0
    b = 1 << (requested.bit_length() - 1)
    while b > 1 and (dim % b) != 0:
        b //= 2
    return b if b > 1 else 0


def hadamard_transform(x: torch.Tensor, block: int) -> torch.Tensor:
    """Orthonormal block Walsh-Hadamard transform on the last axis (inverse == forward)."""
    if block <= 1 or (x.shape[-1] % block) != 0:
        return x.float()
    h = _hadamard_matrix(block).to(device=x.device, dtype=torch.float32)
    orig = x.shape
    return (x.float().reshape(-1, block) @ h).reshape(orig)


def _pad_to_blocks(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    orig_shape = tuple(x.shape)
    flat = x.detach().float().reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=torch.float32)])
    return flat.reshape(-1, group_size), orig_shape


def _unpad_shape(flat: torch.Tensor, orig_shape: tuple[int, ...]) -> torch.Tensor:
    n = 1
    for s in orig_shape:
        n *= s
    return flat.reshape(-1)[:n].reshape(orig_shape)


def quantize_block_int4(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Micro-block symmetric INT4 quantization (signed nibbles [-7..7] packed 2/byte + fp16 scale)."""
    if group_size % 2 != 0:
        raise ValueError("group_size must be even for 2-nibble uint8 packing")
    blocks, orig_shape = _pad_to_blocks(x, group_size)
    scales = (blocks.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-8)
    q = torch.round(blocks / scales).clamp(-7, 7).to(torch.int8) + 8
    q_u8 = q.to(torch.uint8).reshape(-1, 2)
    packed = ((q_u8[:, 0] & 0x0F) | ((q_u8[:, 1] & 0x0F) << 4)).cpu()
    return packed, scales.squeeze(1).to(torch.float16).cpu(), orig_shape


def dequantize_block_int4(packed: torch.Tensor, scales: torch.Tensor, orig_shape: tuple[int, ...],
                          group_size: int = 16) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int8) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    q = torch.stack([lo, hi], dim=1).reshape(-1, group_size).float()
    return _unpad_shape(q * scales.float().unsqueeze(1), orig_shape)


def quantize_block_fp4(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Micro-block E2M1 FP4 quantization (`NVFP4`-style grid +-{0, 0.5, 1, 1.5, 2, 3, 4, 6}, 2/byte + fp16 scale)."""
    if group_size % 2 != 0:
        raise ValueError("group_size must be even for 2-nibble uint8 packing")
    blocks, orig_shape = _pad_to_blocks(x, group_size)
    grid = torch.tensor(_E2M1_POS, device=blocks.device, dtype=torch.float32)
    scales = (blocks.abs().amax(dim=1, keepdim=True) / 6.0).clamp_min(1e-8)
    normed = blocks / scales
    mags = normed.abs().unsqueeze(-1)
    idx = (mags - grid).abs().argmin(dim=-1).to(torch.uint8)  # 0..7
    sign = (normed < 0).to(torch.uint8) << 3
    nib = (sign | idx).reshape(-1, 2)
    packed = (nib[:, 0] | (nib[:, 1] << 4)).cpu()
    return packed, scales.squeeze(1).to(torch.float16).cpu(), orig_shape


def dequantize_block_fp4(packed: torch.Tensor, scales: torch.Tensor, orig_shape: tuple[int, ...],
                         group_size: int = 16) -> torch.Tensor:
    grid = torch.tensor(_E2M1_POS, device=packed.device, dtype=torch.float32)
    nibs = torch.stack([packed & 0x0F, (packed >> 4) & 0x0F], dim=1).reshape(-1, group_size)
    vals = grid[(nibs & 0x07).long()] * torch.where((nibs & 0x08) != 0, -1.0, 1.0)
    return _unpad_shape(vals * scales.float().unsqueeze(1), orig_shape)


def quantize_block_ternary(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """L2-optimal per-block 1.58-bit ternary ({-1, 0, +1}) packed at 5 trits per uint8 byte + fp16 block scale."""
    blocks, orig_shape = _pad_to_blocks(x, group_size)
    abs_b = blocks.abs()
    order = abs_b.argsort(dim=1, descending=True)
    sorted_abs = abs_b.gather(1, order)
    csum = sorted_abs.cumsum(dim=1)
    ks = torch.arange(1, group_size + 1, device=blocks.device, dtype=torch.float32)[None, :]
    best_idx = (csum * csum / ks).argmax(dim=1)  # exact global optimum per block over top-m subsets
    best_k = (best_idx + 1).float()
    best_csum = csum.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    scales = (best_csum / best_k).clamp_min(1e-8)

    # Set non-zero ternary sign on the winning top-m elements of each block
    rank_in_block = order.argsort(dim=1)
    active = rank_in_block <= best_idx.unsqueeze(1)
    trits = torch.where(active, torch.where(blocks >= 0, 2, 0), 1).to(torch.uint8).reshape(-1)  # 0=-1, 1=0, 2=+1

    pad5 = (-trits.numel()) % 5
    if pad5:
        trits = torch.cat([trits, torch.ones(pad5, device=trits.device, dtype=torch.uint8)])
    pow3 = torch.tensor(_POW3, device=trits.device, dtype=torch.uint8)
    packed = (trits.reshape(-1, 5) * pow3).sum(dim=1).to(torch.uint8).cpu()
    return packed, scales.to(torch.float16).cpu(), orig_shape


def dequantize_block_ternary(packed: torch.Tensor, scales: torch.Tensor, orig_shape: tuple[int, ...],
                             group_size: int = 16) -> torch.Tensor:
    n_elems = scales.numel() * group_size
    p3 = torch.tensor(_POW3, device=packed.device, dtype=torch.int32)[None, :]
    trits = ((packed.to(torch.int32).unsqueeze(1) // p3) % 3).reshape(-1)[:n_elems]
    vals = (trits - 1).float().reshape(-1, group_size)
    return _unpad_shape(vals * scales.float().unsqueeze(1), orig_shape)


def quantize_fp8_or_int8(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    """Quantize to 1 byte/element (micro-block INT8 for portability; scale isolates outliers)."""
    blocks, _ = _pad_to_blocks(x, group_size)
    scales = (blocks.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
    q = torch.round(blocks / scales).clamp(-127, 127).to(torch.int8).cpu()
    return q.reshape(-1), scales.squeeze(1).to(torch.float16).cpu(), "int8_blk"


def dequantize_fp8_or_int8(q: torch.Tensor, scales: torch.Tensor, orig_shape: tuple[int, ...],
                           group_size: int = 16) -> torch.Tensor:
    deq = q.reshape(-1, group_size).float() * scales.float().unsqueeze(1)
    return _unpad_shape(deq, orig_shape)


def quantization_error(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, float]:
    xf, hf = x.detach().float().cpu().reshape(-1), x_hat.detach().float().cpu().reshape(-1)
    diff = (xf - hf).norm().item()
    norm = max(xf.norm().item(), 1e-12)
    cos = float(torch.nn.functional.cosine_similarity(xf.unsqueeze(0), hf.unsqueeze(0)).item()) if norm > 1e-12 else 1.0
    return {"rel_l2": round(diff / norm, 6), "cosine": round(cos, 6)}


def _encode_resid(x: torch.Tensor, quant: str, group_size: int, had_block: int) -> dict[str, Any]:
    shp = list(x.shape)
    hb = resolve_hadamard_block(x.shape[-1], had_block)
    xw = hadamard_transform(x, hb) if hb > 1 else x
    if quant == "int4":
        d, s, _ = quantize_block_int4(xw, group_size)
        return {"data": d, "scales": s, "shape": shp}
    if quant == "fp4":
        d, s, _ = quantize_block_fp4(xw, group_size)
        return {"data": d, "scales": s, "shape": shp}
    if quant == "ternary":
        d, s, _ = quantize_block_ternary(xw, group_size)
        return {"data": d, "scales": s, "shape": shp}
    if quant == "int8":
        q, s, _ = quantize_fp8_or_int8(xw, group_size)
        return {"data": q, "scales": s, "shape": shp}
    return {"data": xw.to(torch.float16).cpu(), "scales": None, "shape": shp}


def _decode_resid(r: dict[str, Any], quant: str, group_size: int, had_block: int) -> torch.Tensor:
    shp = tuple(r["shape"])
    if quant == "int4":
        x = dequantize_block_int4(r["data"], r["scales"], shp, group_size)
    elif quant == "fp4":
        x = dequantize_block_fp4(r["data"], r["scales"], shp, group_size)
    elif quant == "ternary":
        x = dequantize_block_ternary(r["data"], r["scales"], shp, group_size)
    elif quant == "int8":
        x = dequantize_fp8_or_int8(r["data"], r["scales"], shp, group_size)
    else:
        x = r["data"].float()
    hb = resolve_hadamard_block(shp[-1], had_block)
    return hadamard_transform(x, hb) if hb > 1 else x


def resolve_mlp_idx_mode(k: int, d_inter: int, requested: str = "auto") -> str:
    """Pick sparse neuron index format: 'bitmask' when denser than packed widths, else min-width 'bitpack' or 'uint16'."""
    if requested in ("uint16", "bitpack", "bitmask"):
        return requested
    bits = max(1, (d_inter - 1).bit_length())
    b_pack = (k * bits + 7) // 8
    b_mask = (d_inter + 7) // 8
    b_u16 = k * 2 if d_inter <= 65_535 else k * 4
    if b_mask < min(b_pack, b_u16):
        return "bitmask"
    if b_pack < b_u16 and bits <= 31:
        return "bitpack"
    return "uint16"


def pack_bitpack_u8(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack `(T, k)` non-negative integers `< 2**bits` losslessly into `(T, ceil(k*bits/8))` `uint8`."""
    t_len, k = idx.shape
    shifts = torch.arange(bits, device=idx.device, dtype=torch.int32)
    bits_flat = ((idx.to(torch.int32).unsqueeze(-1) >> shifts) & 1).reshape(t_len, k * bits)
    n_bytes = (k * bits + 7) // 8
    pad = n_bytes * 8 - bits_flat.shape[1]
    if pad:
        bits_flat = torch.cat([bits_flat, torch.zeros((t_len, pad), device=idx.device, dtype=torch.int32)], dim=1)
    w8 = torch.tensor(_BIT8, device=idx.device, dtype=torch.int32)
    return (bits_flat.reshape(t_len, n_bytes, 8) * w8).sum(dim=2).to(torch.uint8).cpu()


def unpack_bitpack_u8(packed: torch.Tensor, k: int, bits: int) -> torch.Tensor:
    """Unpack `(T, ceil(k*bits/8))` `uint8` back into exact `(T, k)` `int64` neuron indices."""
    t_len = packed.shape[0]
    b_shifts = torch.arange(8, device=packed.device, dtype=torch.int32)
    bits_flat = ((packed.to(torch.int32).unsqueeze(-1) >> b_shifts) & 1).reshape(t_len, -1)[:, : k * bits]
    weights = 1 << torch.arange(bits, device=packed.device, dtype=torch.int64)
    return (bits_flat.reshape(t_len, k, bits).to(torch.int64) * weights).sum(dim=2)


def pack_bitmask_u8(idx: torch.Tensor, d_inter: int) -> torch.Tensor:
    """Pack `(T, k)` unique column indices `< d_inter` into `(T, ceil(d_inter/8))` `uint8` bitmask."""
    t_len = idx.shape[0]
    bits = torch.zeros((t_len, d_inter), device=idx.device, dtype=torch.int32)
    bits.scatter_(1, idx.long(), 1)
    n_bytes = (d_inter + 7) // 8
    pad = n_bytes * 8 - d_inter
    if pad:
        bits = torch.cat([bits, torch.zeros((t_len, pad), device=idx.device, dtype=torch.int32)], dim=1)
    w8 = torch.tensor(_BIT8, device=idx.device, dtype=torch.int32)
    return (bits.reshape(t_len, n_bytes, 8) * w8).sum(dim=2).to(torch.uint8).cpu()


def unpack_bitmask_u8(packed: torch.Tensor, k: int, d_inter: int) -> torch.Tensor:
    """Unpack `(T, ceil(d_inter/8))` `uint8` bitmask back into ascending `(T, k)` `int64` neuron indices."""
    t_len = packed.shape[0]
    b_shifts = torch.arange(8, device=packed.device, dtype=torch.int32)
    bits = ((packed.to(torch.int32).unsqueeze(-1) >> b_shifts) & 1).reshape(t_len, -1)[:, :d_inter]
    rows, cols = bits.nonzero(as_tuple=True)
    if cols.numel() == t_len * k:
        return cols.reshape(t_len, k).to(torch.int64)
    out = torch.zeros((t_len, k), dtype=torch.int64, device=packed.device)
    for t in range(t_len):
        c = cols[rows == t][:k]
        out[t, : c.numel()] = c
    return out


# ----------------------------------------------------------------------------------------------- helpers
def _layer_idx_from_name(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    m = re.search(r"\.h\.(\d+)\.", name)
    return int(m.group(1)) if m else None


_ATTN_OUT_SUFFIXES = (".o_proj", ".out_proj", ".c_proj", ".wo", ".w_o", ".wo_ud", ".proj_out")
_CONV_PATH_TOKENS = (".conv.", ".short_conv.", ".post_attn_conv.")
_CONV_OUT_SUFFIXES = (".proj", ".out", ".linear", ".w_out")


def _infer_attn_heads(model: nn.Module, out_proj_name: str, d_in: int) -> int:
    mods = dict(model.named_modules())
    parent_name = out_proj_name.rsplit(".", 1)[0] if "." in out_proj_name else ""
    parent = mods.get(parent_name)
    if parent is not None:
        for attr in ("num_v_heads", "linear_num_value_heads", "num_heads", "num_attention_heads", "n_head"):
            v = getattr(parent, attr, None)
            if isinstance(v, int) and v > 0 and d_in % v == 0:
                return v
        for hd_attr in ("head_v_dim", "head_dim", "linear_value_head_dim"):
            hd = getattr(parent, hd_attr, None)
            if isinstance(hd, int) and hd > 0 and d_in % hd == 0:
                return d_in // hd
    if any(tok in out_proj_name for tok in _CONV_PATH_TOKENS) or out_proj_name.startswith(("conv.", "short_conv.", "post_attn_conv.")):
        return 1
    cfg = getattr(getattr(model, "config", None), "text_config", getattr(model, "config", None))
    if cfg is not None:
        if "linear_attn" in out_proj_name:
            v = getattr(cfg, "linear_num_value_heads", None)
            if isinstance(v, int) and v > 0 and d_in % v == 0:
                return v
        for attr in ("num_attention_heads", "n_head"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0 and d_in % v == 0:
                return v
    return 1


def _infer_hidden_size(model: nn.Module, mlps: list | None = None, attns: dict | None = None) -> int:
    mods = dict(model.named_modules())
    if mlps:
        d0 = mods[mlps[0].down]
        return int(d0.weight.shape[0] if isinstance(d0, nn.Linear) else d0.weight.shape[1])
    cfg = getattr(getattr(model, "config", None), "text_config", getattr(model, "config", None))
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return int(v)
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is not None and isinstance(getattr(emb, "weight", None), torch.Tensor) and emb.weight.dim() == 2:
        return int(emb.weight.shape[1])
    if attns:
        first_m = next(iter(attns.values()))[0]
        return int(first_m.weight.shape[0] if isinstance(first_m, nn.Linear) else first_m.weight.shape[1])
    raise ValueError("Cannot infer hidden_size from model without dense MLPs, config.hidden_size, embeddings, or attn projections")


def _find_attn_out_projs(model: nn.Module, hidden: int) -> dict[str, tuple[nn.Module, int]]:
    out = {}
    for n, m in model.named_modules():
        if _is_excluded_backbone_path(n) or _is_routed_expert_path(n):
            continue
        if not _is_linear(m) or not _is_float_weight(m):
            continue
        is_attn_suffix = n.endswith(_ATTN_OUT_SUFFIXES) and ".mlp." not in n and ".moe." not in n
        is_conv_suffix = any(tok in n for tok in _CONV_PATH_TOKENS) and n.endswith(_CONV_OUT_SUFFIXES)
        if is_attn_suffix or is_conv_suffix:
            out_dim = int(m.weight.shape[0] if isinstance(m, nn.Linear) else m.weight.shape[1])
            if hidden > 0 and out_dim != hidden:
                continue
            in_dim = int(m.weight.shape[1] if isinstance(m, nn.Linear) else m.weight.shape[0])
            nh = _infer_attn_heads(model, n, in_dim)
            out[n] = (m, nh)
    return out


class _DirectionTargetCaptured(Exception):
    """Sentinel to stop the forward pass right after capturing ``target_layer``."""


def _resolve_direction_layer(n_layers: int, target_layer: int, convention: str = "block") -> tuple[int, bool]:
    """Map user ``target_layer`` into ``(stop_block_idx, use_final_hidden)``.

    ``convention='block'`` uses 0-based decoder block indices ``0..n_layers-1`` (matching
    ``CompactTrace.resid``). ``convention='hidden_states'`` matches HuggingFace
    ``output_hidden_states=True`` and ``MoIPatch.router['layer']``: ``0`` is pre-block embeddings
    (``stop_blk=-1``), ``1..n_layers-1`` is block ``target_layer-1``, and ``n_layers`` is post-final-norm.
    """
    if convention not in ("block", "hidden_states"):
        raise ValueError(f"convention must be 'block' or 'hidden_states', got {convention!r}")
    if n_layers <= 0:
        raise ValueError(f"Model must have at least 1 decoder block, got {n_layers}")
    if convention == "block":
        l = target_layer if target_layer >= 0 else n_layers + target_layer
        if l < 0 or l >= n_layers:
            raise ValueError(f"target_layer {target_layer} out of range for block convention [0, {n_layers - 1}]")
        return l, False
    l = target_layer if target_layer >= 0 else (n_layers + 1) + target_layer
    if l < 0 or l > n_layers:
        raise ValueError(f"target_layer {target_layer} out of range for hidden_states convention [0, {n_layers}]")
    if l == 0:
        return -1, False
    if l == n_layers:
        return n_layers - 1, True
    return l - 1, False


def _resolve_hidden_span(T: int, span: int | tuple[int, int] | None) -> tuple[int, int]:
    if T < 1:
        raise ValueError(f"Sequence must contain at least 1 token, got {T}")
    if span is None:
        return T - 1, T
    if isinstance(span, int) and not isinstance(span, bool):
        idx = span if span >= 0 else T + span
        if idx < 0 or idx >= T:
            raise ValueError(f"Token position {span} out of range for sequence length {T}")
        return idx, idx + 1
    if isinstance(span, (tuple, list)) and len(span) == 2:
        s_raw, e_raw = int(span[0]), int(span[1])
        lo = s_raw if s_raw >= 0 else T + s_raw
        hi = e_raw if e_raw >= 0 else T + e_raw
        if not (0 <= lo < hi <= T):
            raise ValueError(f"Token span {span!r} -> ({lo}, {hi}) invalid for sequence length {T}")
        return lo, hi
    raise ValueError(f"Invalid token position/span {span!r}; expected None, int, or (start, end)")


def _validate_target_dir(target_dir: torch.Tensor, hidden: int, *, normalize: bool = False) -> torch.Tensor:
    t = target_dir.detach().float().cpu() if isinstance(target_dir, torch.Tensor) else torch.as_tensor(target_dir, dtype=torch.float32)
    if t.dim() == 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() != 1 or int(t.shape[0]) != int(hidden):
        raise ValueError(f"target_dir must have shape ({hidden},) or (1, {hidden}), got {tuple(t.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("target_dir contains non-finite values")
    nrm = float(t.norm().item())
    if nrm <= 1e-12:
        raise ValueError(f"target_dir must have non-zero norm, got {nrm}")
    if normalize:
        t = t / nrm
    return t.contiguous()


def _normalize_token_id_list(val: Any, name: str) -> torch.Tensor | None:
    """Normalize ``None`` / int / list[int] / 1D tensor of token ids into a CPU ``torch.long`` tensor."""
    if val is None:
        return None
    if isinstance(val, bool):
        raise ValueError(f"{name} must be an int, list of ints, or 1D tensor, got bool")
    if isinstance(val, int):
        t = torch.tensor([val], dtype=torch.long)
    elif isinstance(val, torch.Tensor):
        if val.dim() == 0:
            val = val.reshape(1)
        if val.dim() != 1 or val.numel() == 0:
            raise ValueError(f"{name} must be a non-empty 1D tensor, got shape {tuple(val.shape)}")
        if val.is_floating_point() or val.is_complex():
            raise ValueError(f"{name} must contain integer token ids")
        t = val.detach().cpu().long()
    elif isinstance(val, (list, tuple)):
        if len(val) == 0:
            raise ValueError(f"{name} must be non-empty")
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in val):
            raise ValueError(f"{name} must contain only ints")
        t = torch.tensor(list(val), dtype=torch.long)
    else:
        raise ValueError(f"{name} must be an int, list of ints, or 1D tensor, got {type(val).__name__}")
    if int(t.min().item()) < 0:
        raise ValueError(f"{name} must contain non-negative token ids")
    return t.contiguous()


def _linearize_final_norm_span(
    fn_mod: nn.Module | None,
    h_pre_2d: torch.Tensor,
    h_post_2d: torch.Tensor,
    tdir_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Frozen-scale affine pullback of ``tdir`` through a final LayerNorm/RMSNorm.

    Returns per-token ``(v_eff (S, D), norm_bias (S,), ok)`` with ``h_pre[s] @ v_eff[s] + norm_bias[s]
    == h_post[s] @ tdir`` exactly (the norm's denominator is frozen at the observed value, which keeps
    the decomposition additive; the true Jacobian ``J^T v`` would sum to 0 for RMSNorm since it is
    degree-0 homogeneous).
    """
    S, D = int(h_pre_2d.shape[0]), int(h_pre_2d.shape[1])
    fallback = (tdir_cpu[None].expand(S, D).contiguous(), torch.zeros(S, dtype=torch.float32), False)
    if fn_mod is None or isinstance(fn_mod, nn.Identity) or S == 0:
        return fallback
    eps_raw = getattr(fn_mod, "eps", None)
    if eps_raw is None:
        eps_raw = getattr(fn_mod, "variance_epsilon", getattr(fn_mod, "layer_norm_eps", 1e-5))
    try:
        eps = float(eps_raw)
    except (TypeError, ValueError):
        eps = 1e-5
    w_p = getattr(fn_mod, "weight", None)
    b_p = getattr(fn_mod, "bias", None)
    w = w_p.detach().float().cpu() if isinstance(w_p, torch.Tensor) and w_p.numel() == D else None
    b = b_p.detach().float().cpu() if isinstance(b_p, torch.Tensor) and b_p.numel() == D else None
    x = h_pre_2d.float()
    actual = h_post_2d.float() @ tdir_cpu
    nb_affine = (b @ tdir_cpu).expand(S).clone() if b is not None else torch.zeros(S, dtype=torch.float32)
    g = w[None] if w is not None else torch.ones((1, D), dtype=torch.float32)

    cands: list[tuple[torch.Tensor, torch.Tensor]] = []
    s_ln = (x.var(dim=-1, unbiased=False, keepdim=True) + eps).sqrt()
    v_raw = (tdir_cpu[None] * g) / s_ln
    cands.append((v_raw - v_raw.mean(dim=-1, keepdim=True), nb_affine))
    s_rms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    cands.append(((tdir_cpu[None] * g) / s_rms, nb_affine))
    if w is not None:
        cands.append(((tdir_cpu[None] * (1.0 + w[None])) / s_rms, torch.zeros(S, dtype=torch.float32)))

    best: tuple[float, torch.Tensor, torch.Tensor] | None = None
    for v_eff, nb in cands:
        err = float(((x * v_eff).sum(dim=-1) + nb - actual).abs().max().item())
        if best is None or err < best[0]:
            best = (err, v_eff, nb)
    assert best is not None
    tol = 1e-2 * max(1.0, float(actual.abs().max().item()))
    if best[0] <= tol:
        return best[1].contiguous(), best[2].contiguous(), True
    return fallback


def _find_final_norm_module(model: nn.Module) -> nn.Module | None:
    inner = getattr(model, "model", None)
    lm = getattr(model, "language_model", None)
    roots = (
        getattr(inner, "language_model", None) if isinstance(inner, nn.Module) else None,
        lm,
        getattr(lm, "model", None) if isinstance(lm, nn.Module) else None,
        inner,
        getattr(model, "transformer", None),
        model,
    )
    for root in roots:
        if not isinstance(root, nn.Module):
            continue
        for attr in ("norm", "embedding_norm", "ln_f", "final_layernorm", "norm_f"):
            cand = getattr(root, attr, None)
            if isinstance(cand, nn.Module):
                return cand
    return None


def _writer_kind(mod_name: str, is_mlp: bool) -> str:
    if is_mlp:
        return "mlp"
    if any(tok in mod_name for tok in _CONV_PATH_TOKENS) or mod_name.startswith(("conv.", "short_conv.", "post_attn_conv.")):
        return "conv"
    return "attn"


def _forward_for_hidden(model: nn.Module, ids_dev: torch.Tensor, *, need_hidden_states: bool) -> Any:
    kw: dict[str, Any] = {"input_ids": ids_dev, "use_cache": False}
    if need_hidden_states:
        kw["output_hidden_states"] = True
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError:
        try:
            return model(**kw)
        except TypeError:
            try:
                return model(input_ids=ids_dev, use_cache=False)
            except TypeError:
                return model(input_ids=ids_dev)


def _normalize_seq_list(inputs: torch.Tensor | list[int] | list[torch.Tensor | list[int]]) -> list[torch.Tensor]:
    if isinstance(inputs, torch.Tensor):
        if inputs.dim() == 1:
            if inputs.numel() == 0:
                raise ValueError("Input sequence tensor must be non-empty")
            return [inputs.reshape(1, -1)]
        if inputs.dim() == 2:
            if inputs.shape[0] == 0 or inputs.shape[1] == 0:
                raise ValueError("Input batch tensor must be non-empty")
            return [inputs[i : i + 1] for i in range(inputs.shape[0])]
        raise ValueError(f"Expected 1D or 2D input_ids tensor, got shape {tuple(inputs.shape)}")
    if not isinstance(inputs, (list, tuple)) or len(inputs) == 0:
        raise ValueError("inputs must contain at least one sequence")
    first = inputs[0]
    if isinstance(first, int) and not isinstance(first, bool):
        return [torch.tensor(inputs, dtype=torch.long).reshape(1, -1)]
    seqs: list[torch.Tensor] = []
    for item in inputs:
        t = item if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=torch.long)
        if t.dim() == 1:
            t = t.reshape(1, -1)
        elif t.dim() == 2 and t.shape[0] == 1:
            pass
        else:
            raise ValueError(f"Each sequence must be 1D or shape (1, T), got {tuple(t.shape)}")
        if t.shape[1] == 0:
            raise ValueError("Each sequence must have at least 1 token")
        seqs.append(t)
    return seqs


# ----------------------------------------------------------------------------------------------- CompactTrace
def _extract_cache_tensors(cache_obj: Any) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in ("key_cache", "value_cache", "conv_states", "recurrent_states", "ssm_states"):
        for i, t in enumerate(getattr(cache_obj, name, None) or ()):
            if isinstance(t, torch.Tensor) and t.numel():
                out[f"{name}.{i}"] = t
    for i, lyr in enumerate(getattr(cache_obj, "layers", None) or ()):
        for attr in ("keys", "values", "conv_states", "recurrent_states"):
            t = getattr(lyr, attr, None)
            if isinstance(t, torch.Tensor) and t.numel():
                out[f"layers.{i}.{attr}"] = t
    return out


@dataclass
class CompactTrace:
    """Scalable token-by-token activation trace combining micro-block quantized residuals, Top-K MLP units,
    per-head attention energies, optional Top-K sparse attention links, and quantized KV/recurrent cache."""

    seq_len: int
    n_layers: int
    hidden_size: int
    resid_quant: str
    group_size: int
    top_k_mlp: int
    resid: dict[int, dict[str, Any]]
    mlp_topk: dict[int, dict[str, torch.Tensor]]
    head_norms: dict[str, dict[str, Any]]
    hadamard_block: int = 0
    mlp_quant: str = "fp16"
    mlp_idx_mode: str = "uint16"
    mlp_group_size: int = 16
    mlp_meta: dict[int, dict[str, Any]] = field(default_factory=dict)
    top_k_attn: int = 0
    attn_topk: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    cache_quant: str = "int8"
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    head_quant: str = "int8"
    meta: dict[str, Any] = field(default_factory=dict)
    grad_quant: str = "int8"
    grad_mode: str = "none"
    grad_target: dict[str, dict[str, Any]] = field(default_factory=dict)
    grad_healthy: dict[str, dict[str, Any]] = field(default_factory=dict)
    grad_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    token_attr: dict[str, torch.Tensor] = field(default_factory=dict)

    def dequant_resid(self, layer: int) -> torch.Tensor:
        return _decode_resid(self.resid[layer], self.resid_quant, self.group_size, self.hadamard_block)

    def dequant_grad(self, mod_suffix: str, kind: str = "target") -> torch.Tensor | None:
        """Reconstruct a compressed `(T, D) float32` output gradient tensor for ``mod_suffix``."""
        src = self.grad_target if kind == "target" else self.grad_healthy if kind == "healthy" else None
        if src is None:
            raise ValueError(f"kind must be 'target' or 'healthy', got {kind!r}")
        entry = src.get(mod_suffix)
        if not entry:
            return None
        return _decode_resid(entry, self.grad_quant, self.group_size, self.hadamard_block)

    def dequant_head_in(self, mod: str) -> torch.Tensor | None:
        """Reconstruct pre-output-projection head activations `(T, n_heads * head_dim) float32` if stored."""
        h = self.head_norms.get(mod)
        if not h or "data" not in h or "shape" not in h:
            return None
        return _decode_resid(
            {"data": h["data"], "scales": h.get("scales"), "shape": h["shape"]},
            self.head_quant, self.group_size, self.hadamard_block,
        )

    def dequant_mlp_topk(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct `(idx (T, k) int64, val (T, k) float32)` from uint16/bitpack/bitmask and quantized values."""
        m = self.mlp_topk[layer]
        meta = self.mlp_meta.get(layer, {})
        if "idx" in m:
            idx = m["idx"].long()
        elif "idx_pack" in m:
            idx = unpack_bitpack_u8(m["idx_pack"], int(meta["k"]), int(meta["bits"]))
        elif "idx_mask" in m:
            idx = unpack_bitmask_u8(m["idx_mask"], int(meta["k"]), int(meta["d_inter"]))
        else:
            raise KeyError(f"Missing MLP index tensor in layer {layer}: {list(m)}")

        t_len, k = int(idx.shape[0]), int(idx.shape[1])
        shp = tuple(meta.get("shape", (t_len, k)))
        if self.mlp_quant == "fp16" or ("val_scales" not in m and m["val"].dtype in (torch.float16, torch.bfloat16, torch.float32)):
            val = m["val"].float()
        else:
            val = _decode_resid({"data": m["val"], "scales": m.get("val_scales"), "shape": shp},
                                self.mlp_quant, self.mlp_group_size, 0)
        return idx, val

    def dequant_attn_topk(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct `(target_idx (H, T, k_att) int64, weights (H, T, k_att) float32)` for a full-attn layer."""
        a = self.attn_topk[layer]
        return a["idx"].long(), a["w"].float() / 255.0

    def dequant_cache(self, key: str) -> torch.Tensor:
        """Reconstruct a cached KV or DeltaNet state tensor in its original shape."""
        return _decode_resid(self.cache[key], self.cache_quant, self.group_size, 0)

    def byte_breakdown(self) -> dict[str, int]:
        def _tbytes(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        resid_b = sum(_tbytes(r["data"]) + (_tbytes(r["scales"]) if r["scales"] is not None else 0)
                      for r in self.resid.values())
        mlp_b = sum(_tbytes(t) for m in self.mlp_topk.values() for t in m.values())
        head_b = sum(
            _tbytes(h["l2"])
            + (_tbytes(h["data"]) if "data" in h else 0)
            + (_tbytes(h["scales"]) if h.get("scales") is not None else 0)
            for h in self.head_norms.values()
        )
        attn_b = sum(_tbytes(t) for a in self.attn_topk.values() for t in a.values())
        cache_b = sum(_tbytes(c["data"]) + (_tbytes(c["scales"]) if c["scales"] is not None else 0)
                      for c in self.cache.values())
        grad_b = (
            sum(_tbytes(g["data"]) + (_tbytes(g["scales"]) if g.get("scales") is not None else 0)
                for g in self.grad_target.values())
            + sum(_tbytes(g["data"]) + (_tbytes(g["scales"]) if g.get("scales") is not None else 0)
                  for g in self.grad_healthy.values())
            + sum(_tbytes(v) for s in self.grad_summary.values() for v in s.values() if isinstance(v, torch.Tensor))
            + sum(_tbytes(v) for v in self.token_attr.values() if isinstance(v, torch.Tensor))
        )
        return {
            "resid": resid_b,
            "mlp_topk": mlp_b,
            "head_norms": head_b,
            "attn_topk": attn_b,
            "cache": cache_b,
            "grad": grad_b,
            "total": resid_b + mlp_b + head_b + attn_b + cache_b + grad_b,
        }

    def size_bytes(self) -> int:
        return self.byte_breakdown()["total"]

    def top_input_tokens(
        self,
        input_ids: torch.Tensor | list[int] | None = None,
        *,
        tokenizer: Any = None,
        top_k: int = 10,
        metric: str = "grad_x_input",
    ) -> list[dict[str, Any]]:
        """Rank input token positions by stored input-embedding attribution (``grad_x_input`` or ``grad_norm``)."""
        if metric not in ("grad_x_input", "grad_norm"):
            raise ValueError(f"metric must be 'grad_x_input' or 'grad_norm', got {metric!r}")
        if metric not in self.token_attr:
            return []
        vals = self.token_attr[metric].detach().float().cpu().reshape(-1)
        gxi = self.token_attr["grad_x_input"].detach().float().cpu().reshape(-1) if "grad_x_input" in self.token_attr else None
        gn = self.token_attr["grad_norm"].detach().float().cpu().reshape(-1) if "grad_norm" in self.token_attr else None

        ids_list: list[int] | None = None
        if input_ids is not None:
            raw_ids = input_ids.detach().cpu().reshape(-1).tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
            ids_list = [int(x) for x in raw_ids[: vals.numel()]]

        rank_vals = vals.abs() if metric == "grad_x_input" else vals
        k = min(max(0, int(top_k)), int(vals.numel()))
        if k == 0:
            return []
        order = torch.argsort(rank_vals, descending=True)[:k].tolist()
        out: list[dict[str, Any]] = []
        for pos in order:
            row: dict[str, Any] = {"pos": int(pos), "score": round(float(vals[pos].item()), 6)}
            if gxi is not None and pos < gxi.numel():
                row["grad_x_input"] = round(float(gxi[pos].item()), 6)
            if gn is not None and pos < gn.numel():
                row["grad_norm"] = round(float(gn[pos].item()), 6)
            if ids_list is not None and pos < len(ids_list):
                tid = ids_list[pos]
                row["token_id"] = tid
                if tokenizer is not None:
                    try:
                        row["token"] = tokenizer.decode([tid])
                    except Exception:
                        pass
            out.append(row)
        return out

    def compare_positions(self, t_target: int, t_ref: int, *, top_k: int = 8) -> dict[str, Any]:
        """Find which layers, MLP neurons, and attention heads shifted strongest at token `t_target` vs `t_ref`."""
        layer_shifts, mlp_shifts, head_shifts = [], [], []
        for l in sorted(self.resid):
            R = self.dequant_resid(l)
            va, vb = R[t_ref], R[t_target]
            layer_shifts.append({
                "layer": l,
                "delta_norm": round(float((vb - va).norm()), 4),
                "cosine": round(float(torch.nn.functional.cosine_similarity(va[None], vb[None])), 5),
            })
        for l in sorted(self.mlp_topk):
            idx_l, val_l = self.dequant_mlp_topk(l)
            ia, va = idx_l[t_ref].tolist(), val_l[t_ref].tolist()
            ib, vb = idx_l[t_target].tolist(), val_l[t_target].tolist()
            ma, mb = dict(zip(ia, va)), dict(zip(ib, vb))
            for u in set(ma) | set(mb):
                da, db = ma.get(u, 0.0), mb.get(u, 0.0)
                mlp_shifts.append({"layer": l, "unit": int(u), "ref_contrib": round(da, 4),
                                   "target_contrib": round(db, 4), "delta": round(db - da, 4),
                                   "abs_delta": round(abs(db - da), 4)})
        for mod, h in self.head_norms.items():
            da = h["l2"][t_ref].float()
            db = h["l2"][t_target].float()
            diff = db - da
            for hid in diff.abs().topk(min(3, len(diff))).indices.tolist():
                head_shifts.append({"module": mod, "layer": h["layer"], "head": hid,
                                    "ref_l2": round(float(da[hid]), 4), "target_l2": round(float(db[hid]), 4),
                                    "delta": round(float(diff[hid]), 4), "abs_delta": round(float(diff[hid].abs()), 4)})
        mlp_shifts.sort(key=lambda x: -x["abs_delta"])
        head_shifts.sort(key=lambda x: -x["abs_delta"])
        res_out: dict[str, Any] = {
            "t_target": t_target, "t_ref": t_ref,
            "layer_shifts": sorted(layer_shifts, key=lambda x: -x["delta_norm"])[:top_k],
            "top_mlp_shifts": mlp_shifts[:top_k],
            "top_head_shifts": head_shifts[:top_k],
        }
        if self.attn_topk:
            attn_shifts = []
            for l in sorted(self.attn_topk):
                idx_a, w_a = self.dequant_attn_topk(l)
                for hid in range(idx_a.shape[0]):
                    ma = dict(zip(idx_a[hid, t_ref].tolist(), w_a[hid, t_ref].tolist()))
                    mb = dict(zip(idx_a[hid, t_target].tolist(), w_a[hid, t_target].tolist()))
                    l1 = sum(abs(mb.get(u, 0.0) - ma.get(u, 0.0)) for u in set(ma) | set(mb))
                    attn_shifts.append({
                        "layer": l, "head": hid, "l1_shift": round(l1, 4),
                        "target_top_src_pos": int(idx_a[hid, t_target, 0]),
                        "target_top_weight": round(float(w_a[hid, t_target, 0]), 4),
                    })
            attn_shifts.sort(key=lambda x: -x["l1_shift"])
            res_out["top_attn_shifts"] = attn_shifts[:top_k]
        return res_out

    def detect_loop_onset(self, layer: int | None = None, window: int = 8, thr: float = 0.96) -> dict[str, Any]:
        """Scan token-to-previous-token residual cosine inside one trace to locate repetition onset `t*`."""
        l = (max(self.resid) // 2) if layer is None else layer
        R = self.dequant_resid(l)
        Rn = R / R.norm(dim=1, keepdim=True).clamp_min(1e-8)
        T = Rn.shape[0]
        if T <= window:
            return {"layer": l, "onset_token": None, "max_lag_cos": 0.0}
        lag_cos = torch.zeros(T)
        for t in range(1, T):
            prev = Rn[max(0, t - window) : t]
            lag_cos[t] = (prev @ Rn[t]).max()
        hits = (lag_cos > thr).nonzero(as_tuple=True)[0].tolist()
        return {
            "layer": l,
            "onset_token": int(hits[0]) if hits else None,
            "max_lag_cos": round(float(lag_cos.max()), 5),
            "mean_tail_cos": round(float(lag_cos[T // 2:].mean()), 5) if T >= 4 else round(float(lag_cos.mean()), 5),
        }

    def save(self, out_dir: str | Path) -> Path:
        from safetensors.torch import save_file

        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        flat: dict[str, torch.Tensor] = {}
        resid_shapes = {}
        for l, r in self.resid.items():
            flat[f"resid.{l}.data"] = r["data"].contiguous()
            if r["scales"] is not None:
                flat[f"resid.{l}.scales"] = r["scales"].contiguous()
            resid_shapes[str(l)] = list(r["shape"])
        for l, m in self.mlp_topk.items():
            for k, v in m.items():
                flat[f"mlp.{l}.{k}"] = v.contiguous()
        head_meta = {}
        for idx, (mod, h) in enumerate(self.head_norms.items()):
            flat[f"head.{idx}"] = h["l2"].contiguous()
            hmeta: dict[str, Any] = {"module": mod, "layer": h["layer"], "n_heads": h["n_heads"]}
            if "data" in h:
                flat[f"head.{idx}.data"] = h["data"].contiguous()
                if h.get("scales") is not None:
                    flat[f"head.{idx}.scales"] = h["scales"].contiguous()
                hmeta["shape"] = list(h["shape"])
            head_meta[str(idx)] = hmeta
        for l, a in self.attn_topk.items():
            flat[f"attn.{l}.idx"] = a["idx"].contiguous()
            flat[f"attn.{l}.w"] = a["w"].contiguous()
        cache_meta = {}
        for idx, (ck, c) in enumerate(self.cache.items()):
            flat[f"cache.{idx}.data"] = c["data"].contiguous()
            if c["scales"] is not None:
                flat[f"cache.{idx}.scales"] = c["scales"].contiguous()
            cache_meta[str(idx)] = {"key": ck, "shape": list(c["shape"])}
        grad_target_meta: dict[str, Any] = {}
        for idx, (mk, g) in enumerate(self.grad_target.items()):
            flat[f"grad_t.{idx}.data"] = g["data"].contiguous()
            if g.get("scales") is not None:
                flat[f"grad_t.{idx}.scales"] = g["scales"].contiguous()
            grad_target_meta[str(idx)] = {"module": mk, "shape": list(g["shape"])}
        grad_healthy_meta: dict[str, Any] = {}
        for idx, (mk, g) in enumerate(self.grad_healthy.items()):
            flat[f"grad_h.{idx}.data"] = g["data"].contiguous()
            if g.get("scales") is not None:
                flat[f"grad_h.{idx}.scales"] = g["scales"].contiguous()
            grad_healthy_meta[str(idx)] = {"module": mk, "shape": list(g["shape"])}
        grad_summary_meta: dict[str, Any] = {}
        for idx, (mk, gs) in enumerate(self.grad_summary.items()):
            tkeys = []
            smeta: dict[str, Any] = {"module": mk}
            for k, v in gs.items():
                if isinstance(v, torch.Tensor):
                    flat[f"grad_s.{idx}.{k}"] = v.detach().cpu().contiguous().clone()
                    tkeys.append(k)
                elif isinstance(v, tuple):
                    smeta[k] = list(v)
                else:
                    smeta[k] = v
            smeta["tensor_keys"] = tkeys
            grad_summary_meta[str(idx)] = smeta
        for k, v in self.token_attr.items():
            if isinstance(v, torch.Tensor):
                flat[f"tok_attr.{k}"] = v.detach().cpu().contiguous().clone()
        save_file(flat, str(p / "compact_trace.safetensors"))
        (p / "compact_trace.json").write_text(json.dumps({
            "seq_len": self.seq_len, "n_layers": self.n_layers, "hidden_size": self.hidden_size,
            "resid_quant": self.resid_quant, "group_size": self.group_size, "top_k_mlp": self.top_k_mlp,
            "hadamard_block": self.hadamard_block, "mlp_quant": self.mlp_quant,
            "mlp_idx_mode": self.mlp_idx_mode, "mlp_group_size": self.mlp_group_size,
            "mlp_meta": {str(l): m for l, m in self.mlp_meta.items()},
            "top_k_attn": self.top_k_attn, "attn_layers": sorted(self.attn_topk),
            "cache_quant": self.cache_quant, "cache_meta": cache_meta,
            "head_quant": self.head_quant, "resid_shapes": resid_shapes, "heads": head_meta, "meta": self.meta,
            "grad_quant": self.grad_quant, "grad_mode": self.grad_mode,
            "grad_target_meta": grad_target_meta, "grad_healthy_meta": grad_healthy_meta,
            "grad_summary_meta": grad_summary_meta,
            "token_attr_keys": sorted(k for k, v in self.token_attr.items() if isinstance(v, torch.Tensor)),
        }, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, trace_dir: str | Path) -> "CompactTrace":
        from safetensors.torch import load_file

        p = Path(trace_dir)
        info = json.loads((p / "compact_trace.json").read_text(encoding="utf-8"))
        tens = load_file(str(p / "compact_trace.safetensors"))
        resid, mlp_topk, head_norms, attn_topk, cache = {}, {}, {}, {}, {}
        for l_str, shp in info["resid_shapes"].items():
            l = int(l_str)
            resid[l] = {"data": tens[f"resid.{l}.data"], "scales": tens.get(f"resid.{l}.scales"), "shape": shp}
        for l in range(info["n_layers"]):
            prefix = f"mlp.{l}."
            layer_t = {k[len(prefix):]: v for k, v in tens.items() if k.startswith(prefix)}
            if layer_t:
                mlp_topk[l] = layer_t
        for idx, hm in info["heads"].items():
            h_entry: dict[str, Any] = {"layer": hm["layer"], "n_heads": hm["n_heads"], "l2": tens[f"head.{idx}"]}
            if f"head.{idx}.data" in tens and "shape" in hm:
                h_entry["data"] = tens[f"head.{idx}.data"]
                h_entry["scales"] = tens.get(f"head.{idx}.scales")
                h_entry["shape"] = hm["shape"]
            head_norms[hm["module"]] = h_entry
        for l in info.get("attn_layers", ()):
            if f"attn.{l}.idx" in tens:
                attn_topk[int(l)] = {"idx": tens[f"attn.{l}.idx"], "w": tens[f"attn.{l}.w"]}
        for idx, cm in info.get("cache_meta", {}).items():
            cache[cm["key"]] = {
                "data": tens[f"cache.{idx}.data"],
                "scales": tens.get(f"cache.{idx}.scales"),
                "shape": cm["shape"],
            }
        grad_target: dict[str, dict[str, Any]] = {}
        for idx, gm in info.get("grad_target_meta", {}).items():
            grad_target[gm["module"]] = {
                "data": tens[f"grad_t.{idx}.data"],
                "scales": tens.get(f"grad_t.{idx}.scales"),
                "shape": gm["shape"],
            }
        grad_healthy: dict[str, dict[str, Any]] = {}
        for idx, gm in info.get("grad_healthy_meta", {}).items():
            grad_healthy[gm["module"]] = {
                "data": tens[f"grad_h.{idx}.data"],
                "scales": tens.get(f"grad_h.{idx}.scales"),
                "shape": gm["shape"],
            }
        grad_summary: dict[str, dict[str, Any]] = {}
        for idx, sm in info.get("grad_summary_meta", {}).items():
            mod = sm["module"]
            entry: dict[str, Any] = {
                k: v for k, v in sm.items() if k not in ("module", "tensor_keys")
            }
            if "pareto_betas" in entry and isinstance(entry["pareto_betas"], list):
                entry["pareto_betas"] = tuple(entry["pareto_betas"])
            for tk in sm.get("tensor_keys", ()):
                entry[tk] = tens[f"grad_s.{idx}.{tk}"]
            grad_summary[mod] = entry
        token_attr = {
            k: tens[f"tok_attr.{k}"]
            for k in info.get("token_attr_keys", ())
            if f"tok_attr.{k}" in tens
        }
        mlp_meta = {int(ls): m for ls, m in info.get("mlp_meta", {}).items()}
        return cls(
            info["seq_len"], info["n_layers"], info["hidden_size"], info["resid_quant"],
            info["group_size"], info["top_k_mlp"], resid, mlp_topk, head_norms,
            hadamard_block=info.get("hadamard_block", 0),
            mlp_quant=info.get("mlp_quant", "fp16"),
            mlp_idx_mode=info.get("mlp_idx_mode", "uint16"),
            mlp_group_size=info.get("mlp_group_size", info["group_size"]),
            mlp_meta=mlp_meta,
            top_k_attn=info.get("top_k_attn", 0),
            attn_topk=attn_topk,
            cache_quant=info.get("cache_quant", "int8"),
            cache=cache,
            head_quant=info.get("head_quant", "norm"),
            meta=info.get("meta", {}),
            grad_quant=info.get("grad_quant", "int8"),
            grad_mode=info.get("grad_mode", "none"),
            grad_target=grad_target,
            grad_healthy=grad_healthy,
            grad_summary=grad_summary,
            token_attr=token_attr,
        )


def _safe_module_suffix(name: str) -> str:
    from .moipatch import module_suffix

    try:
        return module_suffix(name)
    except ValueError:
        return name


def capture_compact(
    model: nn.Module,
    input_ids: torch.Tensor | list[int],
    *,
    top_k_mlp: int | float | str = "auto",
    mlp_energy_target: float = 0.95,
    resid_quant: str = "int8",
    group_size: int = 64,
    hadamard_block: int = 64,
    mlp_quant: str = "fp4",
    mlp_idx_mode: str = "auto",
    mlp_group_size: int = 16,
    top_k_attn: int = 16,
    capture_cache: bool = True,
    cache_quant: str = "int8",
    head_quant: str = "int8",
    max_tokens: int | None = None,
    capture_grad: bool = False,
    grad_mode: str = "compressed",
    grad_quant: str = "int8",
    grad_rank: int = 8,
    target_span: int | tuple[int, int] | None = None,
    healthy_span: tuple[int, int] | None = None,
    grad_checkpointing: bool = True,
    pareto_betas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
    center: str = "none",
    healthy_cost: str = "linear",
    target_dir: torch.Tensor | None = None,
    target_layer: int | None = None,
    target_layer_convention: str = "block",
    protect_dir: torch.Tensor | None = None,
    protect_layer: int | None = None,
    protect_layer_convention: str | None = None,
    protect_span: int | tuple[int, int] | None = None,
    grad_objective: str = "auto",
    target_token_ids: int | list[int] | torch.Tensor | None = None,
    contrast_token_ids: int | list[int] | torch.Tensor | None = None,
) -> CompactTrace:
    """Record a Pareto-optimal compressed token trace (~80-95 MiB per 1.6k tok on 0.8B):
    INT8 Hadamard-rotated residuals, adaptive or fixed Top-K FP4 bitmask/bitpack MLP units
    (default `top_k_mlp="auto"` picks per-layer `K` to reach `mlp_energy_target=0.95`),
    `head_quant` pre-`W_O` head vectors + post-`W_O` L2 norms, sparse Top-16 attention targets,
    and INT8 KV/recurrent cache states. When ``capture_grad=True``, also runs target & healthy
    backward passes to record per-module causal summaries (and compressed token-level gradients
    when ``grad_mode="compressed"``). Pass ``target_dir`` + ``target_layer`` to attribute an
    internal residual-stream direction (e.g. a router/veto vector) instead of output token NLL.

    Extra objectives: ``grad_objective="logit_margin"`` (auto-selected when ``target_token_ids`` or
    ``contrast_token_ids`` is given) differentiates ``logit[target] - logit[contrast]`` without softmax,
    so gradients do not vanish when ``P(target) -> 1``. ``protect_dir`` (+ ``protect_layer``) replaces the
    healthy NLL pass with a backward pass on a direction to *protect*, so ``C_h`` / ``F_h`` /
    ``neuron_dnll_attr`` / ``pareto_eigvecs`` rank sites that move ``target_dir`` while leaving
    ``protect_dir`` untouched."""
    t_tok_ids = _normalize_token_id_list(target_token_ids, "target_token_ids")
    c_tok_ids = _normalize_token_id_list(contrast_token_ids, "contrast_token_ids")
    if grad_objective not in ("auto", "token_nll", "logit_margin", "direction"):
        raise ValueError(f"grad_objective must be 'auto', 'token_nll', 'logit_margin' or 'direction', got {grad_objective!r}")
    if grad_objective == "auto":
        eff_target_obj = (
            "direction" if target_dir is not None
            else "logit_margin" if (t_tok_ids is not None or c_tok_ids is not None)
            else "token_nll"
        )
    else:
        eff_target_obj = grad_objective
    dir_mode = eff_target_obj == "direction"
    if dir_mode:
        if target_dir is None:
            raise ValueError("grad_objective='direction' requires target_dir")
        if t_tok_ids is not None or c_tok_ids is not None:
            raise ValueError("target_token_ids/contrast_token_ids cannot be combined with target_dir")
    elif target_dir is not None:
        raise ValueError(f"target_dir cannot be combined with grad_objective={eff_target_obj!r}")
    if eff_target_obj == "token_nll" and (t_tok_ids is not None or c_tok_ids is not None):
        raise ValueError("target_token_ids/contrast_token_ids require grad_objective='logit_margin'")
    if eff_target_obj == "logit_margin" and not capture_grad:
        raise ValueError("grad_objective='logit_margin' requires capture_grad=True")
    if dir_mode and not capture_grad:
        raise ValueError("target_dir requires capture_grad=True (or use capture_causal_trace)")
    if dir_mode and target_layer is None:
        raise ValueError("target_layer is required when target_dir is provided")
    protect_mode = protect_dir is not None
    eff_protect_layer = protect_layer if protect_layer is not None else target_layer
    eff_protect_conv = target_layer_convention if protect_layer_convention is None else protect_layer_convention
    if protect_mode:
        if not capture_grad:
            raise ValueError("protect_dir requires capture_grad=True (or use capture_causal_trace)")
        if eff_protect_layer is None:
            raise ValueError("protect_layer is required when protect_dir is given without target_layer")
    target_uses_next_tok = (eff_target_obj == "token_nll") or (eff_target_obj == "logit_margin" and t_tok_ids is None)
    if center not in ("none", "healthy"):
        raise ValueError(f"center must be 'none' or 'healthy', got {center!r}")
    if healthy_cost not in ("linear", "fisher"):
        raise ValueError(f"healthy_cost must be 'linear' or 'fisher', got {healthy_cost!r}")
    if capture_grad and grad_mode not in ("compressed", "summary"):
        raise ValueError(f"grad_mode must be 'compressed' or 'summary', got {grad_mode!r}")
    if capture_grad and healthy_cost == "fisher" and grad_mode != "compressed":
        raise ValueError(
            f"healthy_cost='fisher' requires per-token gradients (grad_mode='compressed'), got grad_mode={grad_mode!r}; "
            "summary mode does not store per-token gradients"
        )

    if isinstance(top_k_mlp, str):
        if top_k_mlp != "auto":
            raise ValueError(f"Unsupported top_k_mlp string {top_k_mlp!r}; expected 'auto', int, or (0, 1] fraction")
        auto_mlp = True
        target_ef = float(mlp_energy_target)
    elif isinstance(top_k_mlp, float) and 0.0 < top_k_mlp <= 1.0:
        auto_mlp = True
        target_ef = float(top_k_mlp)
    elif isinstance(top_k_mlp, int) and top_k_mlp <= 0:
        auto_mlp = True
        target_ef = float(mlp_energy_target)
    else:
        auto_mlp = False
        target_ef = float(mlp_energy_target)

    dev = next(model.parameters()).device
    ids = (input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)).reshape(1, -1)
    if max_tokens is not None:
        ids = ids[:, :max_tokens]
    T = int(ids.shape[1])
    if capture_grad:
        if (target_uses_next_tok or (not protect_mode and healthy_span is not None)) and T < 2:
            raise ValueError(f"capture_compact(capture_grad=True) requires seq_len >= 2, got {T}")
        if T < 1:
            raise ValueError(f"capture_compact(capture_grad=True, target_dir=...) requires seq_len >= 1, got {T}")

    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    mods = dict(model.named_modules())
    mlps = find_mlps(model)
    down_norms = {}
    mlp_by_down = {}
    for g in mlps:
        m = mods[g.down]
        w = m.weight.detach().float()
        down_norms[g.down] = w.norm(dim=0) if isinstance(m, nn.Linear) else w.norm(dim=1)
        li = _layer_idx_from_name(g.down)
        mlp_by_down[g.down] = li if li is not None else len(mlp_by_down)

    hidden = _infer_hidden_size(model, mlps)
    attns = _find_attn_out_projs(model, hidden)
    if dir_mode:
        tdir_cpu: torch.Tensor | None = _validate_target_dir(target_dir, hidden, normalize=False)  # type: ignore[arg-type]
        stop_blk: int | None
        stop_blk, use_final_hidden = _resolve_direction_layer(n_layers, int(target_layer), target_layer_convention)  # type: ignore[arg-type]
    else:
        tdir_cpu = None
        stop_blk = None
        use_final_hidden = False
    if protect_mode:
        pdir_cpu: torch.Tensor | None = _validate_target_dir(protect_dir, hidden, normalize=False)  # type: ignore[arg-type]
        p_stop_blk, p_use_final_hidden = _resolve_direction_layer(n_layers, int(eff_protect_layer), eff_protect_conv)  # type: ignore[arg-type]
    else:
        pdir_cpu = None
        p_stop_blk, p_use_final_hidden = None, False  # type: ignore[assignment]

    resid_out: dict[int, dict[str, Any]] = {}
    mlp_topk_out: dict[int, dict[str, torch.Tensor]] = {}
    mlp_meta_out: dict[int, dict[str, Any]] = {}
    head_out: dict[str, dict[str, Any]] = {}
    handles = []

    for l, blk in enumerate(blocks):
        def bhook(mod, inp, out, l=l):
            t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor):
                return
            x = t.detach().reshape(-1, t.shape[-1])
            resid_out[l] = _encode_resid(x, resid_quant, group_size, hadamard_block)

        handles.append(blk.register_forward_hook(bhook))

    for dname, li in mlp_by_down.items():
        m = mods[dname]
        wn = down_norms[dname]

        def mhook(mod, inp, out, li=li, wn=wn):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            d_inter = x.shape[1]
            contrib = x * wn.to(x.device)
            if auto_mlp:
                abs_sorted, idx_all = contrib.abs().sort(dim=1, descending=True)
                cum_sq = (abs_sorted * abs_sorted).cumsum(dim=1)
                raw_tot_sq = cum_sq[:, -1:]
                tok_ef = torch.where(
                    raw_tot_sq > 1e-12,
                    (cum_sq / raw_tot_sq.clamp_min(1e-12)).clamp_max(1.0).sqrt(),
                    torch.ones_like(cum_sq),
                )
                mean_ef = tok_ef.mean(dim=0)
                hits = (mean_ef >= target_ef).nonzero(as_tuple=True)[0]
                k_raw = int(hits[0].item()) + 1 if hits.numel() > 0 else d_inter
                g = max(1, mlp_group_size)
                k = min(d_inter, max(min(g, d_inter), ((k_raw + g - 1) // g) * g))
                idx = idx_all[:, :k]
                signed = contrib.gather(1, idx)
                ef_vec = tok_ef[:, k - 1]
            else:
                k = min(int(top_k_mlp), d_inter)
                _, idx = contrib.abs().topk(k, dim=1)
                signed = contrib.gather(1, idx)
                raw_tot = (contrib * contrib).sum(dim=1)
                kept_sq = (signed * signed).sum(dim=1)
                ef_vec = torch.where(
                    raw_tot > 1e-12,
                    (kept_sq / raw_tot.clamp_min(1e-12)).clamp_max(1.0).sqrt(),
                    torch.ones_like(raw_tot),
                )

            mode = resolve_mlp_idx_mode(k, d_inter, mlp_idx_mode)
            bits = max(1, (d_inter - 1).bit_length())
            entry: dict[str, torch.Tensor] = {}
            if mode == "bitmask":
                idx_use, perm = idx.sort(dim=1)
                val_use = signed.gather(1, perm)
                entry["idx_mask"] = pack_bitmask_u8(idx_use, d_inter)
            elif mode == "bitpack":
                val_use = signed
                entry["idx_pack"] = pack_bitpack_u8(idx, bits)
            else:
                idx_dt = torch.uint16 if d_inter <= 65_535 else torch.int32
                val_use = signed
                entry["idx"] = idx.to(idx_dt).cpu()

            if mlp_quant == "fp16":
                entry["val"] = val_use.to(torch.float16).cpu()
            else:
                venc = _encode_resid(val_use, mlp_quant, mlp_group_size, 0)
                entry["val"] = venc["data"]
                if venc["scales"] is not None:
                    entry["val_scales"] = venc["scales"]
            entry["energy_frac"] = ef_vec.to(torch.float16).cpu()
            mlp_topk_out[li] = entry
            mlp_meta_out[li] = {
                "mode": mode, "k": k, "d_inter": d_inter, "bits": bits,
                "shape": [int(x.shape[0]), k],
                "mean_energy_frac": round(float(ef_vec.mean().item()), 5),
            }

        handles.append(m.register_forward_hook(mhook))

    head_x_passes: dict[str, list[torch.Tensor]] = {}
    for aname, (am, nh) in attns.items():
        li = _layer_idx_from_name(aname)

        def ahook(mod, inp, out, aname=aname, li=li, nh=nh):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            head_x_passes.setdefault(aname, []).append(x.cpu())
            w = mod.weight.detach().to(x.device, dtype=torch.float32)
            wh = w.t().reshape(nh, -1, w.shape[0]) if isinstance(mod, nn.Linear) else w.reshape(nh, -1, w.shape[1])
            l2 = torch.einsum("thd,hdo->tho", x.reshape(x.shape[0], nh, -1), wh).norm(dim=2)
            head_out[aname] = {"layer": li, "n_heads": nh, "l2": l2.to(torch.float16).cpu()}

        handles.append(am.register_forward_hook(ahook))

    out = None
    try:
        with torch.no_grad():
            try:
                out = model(input_ids=ids.to(dev), output_attentions=(top_k_attn > 0), use_cache=capture_cache)
            except (TypeError, ValueError, NotImplementedError):
                resid_out.clear()
                mlp_topk_out.clear()
                mlp_meta_out.clear()
                head_out.clear()
                head_x_passes.clear()
                if top_k_attn > 0:
                    try:
                        out = model(input_ids=ids.to(dev), use_cache=capture_cache)
                    except (TypeError, ValueError, NotImplementedError):
                        resid_out.clear()
                        mlp_topk_out.clear()
                        mlp_meta_out.clear()
                        head_out.clear()
                        head_x_passes.clear()
                        out = model(input_ids=ids.to(dev), use_cache=False)
                else:
                    out = model(input_ids=ids.to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if head_quant != "norm":
        for aname, xs in head_x_passes.items():
            if aname in head_out and xs:
                x_cat = torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]
                h_enc = _encode_resid(x_cat.to(dev), head_quant, group_size, hadamard_block)
                head_out[aname]["data"] = h_enc["data"]
                head_out[aname]["scales"] = h_enc["scales"]
                head_out[aname]["shape"] = h_enc["shape"]

    attn_topk_out: dict[int, dict[str, torch.Tensor]] = {}
    if top_k_attn > 0 and out is not None:
        raw_atts = list(getattr(out, "attentions", None) or ())
        fa_layers = [
            l for l, blk in enumerate(blocks)
            if any("self_attn" in n or n == "attn" or n.startswith("attn.") or ".attn." in n for n, _ in blk.named_modules())
        ]
        attn_pairs = (
            zip(fa_layers, raw_atts)
            if (raw_atts and len(raw_atts) == len(fa_layers) and len(raw_atts) < n_layers)
            else enumerate(raw_atts)
        )
        for li, att in attn_pairs:
            if not isinstance(att, torch.Tensor) or att.dim() != 4 or att.shape[-1] == 0:
                continue
            a = att[0].detach().float()
            k_att = min(top_k_attn, a.shape[-1])
            vals, a_idx = a.topk(k_att, dim=-1)
            idx_dt = torch.uint16 if a.shape[-1] <= 65_535 else torch.int32
            attn_topk_out[li] = {
                "idx": a_idx.to(idx_dt).cpu(),
                "w": torch.round(vals.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu(),
            }

    cache_out: dict[str, dict[str, Any]] = {}
    if capture_cache and out is not None:
        for ck, cv in _extract_cache_tensors(getattr(out, "past_key_values", None)).items():
            cache_out[ck] = _encode_resid(cv, cache_quant, group_size, 0)
    del out
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    modes = {m["mode"] for m in mlp_meta_out.values()}
    eff_idx_mode = next(iter(modes)) if len(modes) == 1 else mlp_idx_mode
    eff_top_k_mlp = round(sum(m["k"] for m in mlp_meta_out.values()) / len(mlp_meta_out)) if (auto_mlp and mlp_meta_out) else (0 if auto_mlp else int(top_k_mlp))
    trace_meta: dict[str, Any] = {
        "top_k_mlp_mode": "auto" if auto_mlp else "fixed",
        "mlp_energy_target": round(float(target_ef), 4) if auto_mlp else None,
        "mlp_k_min": min((m["k"] for m in mlp_meta_out.values()), default=0),
        "mlp_k_max": max((m["k"] for m in mlp_meta_out.values()), default=0),
        "mlp_k_mean": eff_top_k_mlp,
        "mlp_mean_energy_frac": round(
            sum(m["mean_energy_frac"] for m in mlp_meta_out.values()) / max(1, len(mlp_meta_out)), 5
        ),
    }

    grad_target_out: dict[str, dict[str, Any]] = {}
    grad_healthy_out: dict[str, dict[str, Any]] = {}
    grad_summary_out: dict[str, dict[str, Any]] = {}
    token_attr_out: dict[str, torch.Tensor] = {}
    eff_grad_mode = "none"

    if capture_grad:
        eff_grad_mode = grad_mode
        if not target_uses_next_tok:
            lo_t, hi_t = _resolve_hidden_span(T, target_span)
        else:
            lo_t, hi_t = _resolve_target_span(T, target_span if isinstance(target_span, tuple) or target_span is None else (int(target_span), int(target_span) + 1))
        if protect_mode:
            p_span_spec = protect_span if protect_span is not None else (healthy_span if healthy_span is not None else (lo_t, hi_t))
            lo_h, hi_h = _resolve_hidden_span(T, p_span_spec)
        elif not target_uses_next_tok:
            if healthy_span is not None:
                lo_h = max(0, min(T - 2, int(healthy_span[0])))
                hi_h = max(lo_h, min(T - 1, int(healthy_span[1])))
            else:
                lo_h, hi_h = 0, 0
        else:
            if healthy_span is not None:
                lo_h = max(0, min(T - 2, int(healthy_span[0])))
                hi_h = max(lo_h, min(T - 1, int(healthy_span[1])))
            elif lo_t >= 1:
                lo_h, hi_h = 0, lo_t
            else:
                lo_h, hi_h = 0, 0

        grad_mods: dict[str, dict[str, Any]] = {}
        for dname, li in mlp_by_down.items():
            mk = _safe_module_suffix(dname)
            grad_mods[mk] = {"mod": mods[dname], "kind": "mlp", "layer": int(li), "n_heads": 0}
        for aname, (am, nh) in attns.items():
            mk = _safe_module_suffix(aname)
            li = _layer_idx_from_name(aname)
            grad_mods[mk] = {
                "mod": am,
                "kind": _writer_kind(aname, False),
                "layer": int(li) if li is not None else None,
                "n_heads": int(nh),
            }
        if dir_mode and stop_blk is not None and (hi_h <= lo_h or protect_mode):
            grad_mods = {
                mk: meta_m
                for mk, meta_m in grad_mods.items()
                if (stop_blk >= 0 and (meta_m["layer"] is None or int(meta_m["layer"]) <= stop_blk))
            }

        orig_req = [(p, p.requires_grad) for p in model.parameters()]
        was_training = bool(model.training)
        was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
        enabled_gc = False
        for p, _ in orig_req:
            p.requires_grad_(False)
        model.eval()
        if grad_checkpointing and not was_gc and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                enabled_gc = True
            except Exception:
                enabled_gc = False

        ids_dev = ids.to(dev)

        def _run_backward(
            lo: int,
            hi: int,
            sign: float,
            *,
            capture_acts: bool,
            objective: str = "token_nll",
            dir_vec: torch.Tensor | None = None,
            dir_stop_blk: int | None = None,
            dir_use_final: bool = False,
            dir_layer_label: int | None = None,
            tok_targets: torch.Tensor | None = None,
            tok_contrasts: torch.Tensor | None = None,
        ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], float]:
            st_h_passes: dict[str, list[torch.Tensor]] = {}
            st_y_passes: dict[str, list[torch.Tensor]] = {}
            st_g_passes: dict[str, dict[int, torch.Tensor]] = {}
            fwd_counts: dict[str, int] = {}
            recomp_cursor: dict[str, int] = {}
            state: dict[str, bool] = {"in_backward": False}
            emb_state: dict[str, torch.Tensor] = {}
            ghandles = []

            def _attach_emb_tracker(t: torch.Tensor) -> torch.Tensor:
                if capture_acts and "x" not in emb_state:
                    emb_state["x"] = (t[0] if t.dim() == 3 else t).detach().float().cpu()
                if torch.is_grad_enabled() and not t.requires_grad:
                    t = t.detach().requires_grad_(True)
                if capture_acts and torch.is_grad_enabled() and t.requires_grad:
                    def _save_emb_g(grad: torch.Tensor) -> None:
                        emb_state["g"] = (grad[0] if grad.dim() == 3 else grad).detach().float().cpu()

                    t.register_hook(_save_emb_g)
                return t

            emb_mod = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else getattr(getattr(model, "model", model), "embed_tokens", None)
            if emb_mod is not None:
                def _emb_hook(_m, _inp, emb_out):
                    t = emb_out[0] if isinstance(emb_out, tuple) else emb_out
                    if isinstance(t, torch.Tensor):
                        t_tr = _attach_emb_tracker(t)
                        if t_tr is not t:
                            return (t_tr, *emb_out[1:]) if isinstance(emb_out, tuple) else t_tr
                    return emb_out

                ghandles.append(emb_mod.register_forward_hook(_emb_hook))
            elif len(blocks) > 0:
                def _emb_pre_hook(_m, inp):
                    x0 = inp[0] if isinstance(inp, tuple) and inp else inp
                    if isinstance(x0, torch.Tensor):
                        x0_tr = _attach_emb_tracker(x0)
                        if x0_tr is not x0 and isinstance(inp, tuple):
                            return (x0_tr, *inp[1:])
                    return None

                ghandles.append(blocks[0].register_forward_pre_hook(_emb_pre_hook))

            for mk, meta_m in grad_mods.items():
                mod = meta_m["mod"]

                def _mod_hook(_m, inp, mod_out, mk=mk):
                    t = mod_out[0] if isinstance(mod_out, tuple) else mod_out
                    if not state["in_backward"]:
                        p_idx = fwd_counts.get(mk, 0)
                        fwd_counts[mk] = p_idx + 1
                        if capture_acts:
                            x_in = inp[0] if isinstance(inp, tuple) else inp
                            st_h_passes.setdefault(mk, []).append((x_in[0] if x_in.dim() == 3 else x_in).detach().float().cpu())
                            if isinstance(t, torch.Tensor):
                                st_y_passes.setdefault(mk, []).append((t[0] if t.dim() == 3 else t).detach().float().cpu())
                        if torch.is_grad_enabled() and isinstance(t, torch.Tensor):
                            need_replace = False
                            if not t.requires_grad:
                                t = t.detach().requires_grad_(True)
                                need_replace = True

                            def _save_g(grad, mk=mk, p_idx=p_idx):
                                st_g_passes.setdefault(mk, {})[p_idx] = (grad[0] if grad.dim() == 3 else grad).detach().float().cpu()

                            t.register_hook(_save_g)
                            if need_replace:
                                return (t, *mod_out[1:]) if isinstance(mod_out, tuple) else t
                    else:
                        p_idx = recomp_cursor.get(mk, max(0, fwd_counts.get(mk, 1) - 1))
                        recomp_cursor[mk] = max(0, p_idx - 1)
                        if torch.is_grad_enabled() and isinstance(t, torch.Tensor):
                            need_replace = False
                            if not t.requires_grad:
                                t = t.detach().requires_grad_(True)
                                need_replace = True

                            def _save_g_recomp(grad, mk=mk, p_idx=p_idx):
                                st_g_passes.setdefault(mk, {}).setdefault(
                                    p_idx, (grad[0] if grad.dim() == 3 else grad).detach().float().cpu()
                                )

                            t.register_hook(_save_g_recomp)
                            if need_replace:
                                return (t, *mod_out[1:]) if isinstance(mod_out, tuple) else t
                    return mod_out

                ghandles.append(mod.register_forward_hook(_mod_hook))

            target_hooks = []
            captured_h: dict[str, Any] = {}
            out_dir = None
            obj_val = 0.0
            logits = lp = obj = None
            try:
                with torch.enable_grad():
                    if objective == "direction":
                        assert dir_vec is not None and dir_stop_blk is not None
                        if not dir_use_final:
                            if dir_stop_blk == -1:
                                def _t_pre_hook(_m, inp):
                                    h = inp[0] if isinstance(inp, tuple) and inp else inp
                                    if torch.is_grad_enabled() and isinstance(h, torch.Tensor) and h.requires_grad:
                                        captured_h["val"] = h
                                        raise _DirectionTargetCaptured()

                                target_hooks.append(blocks[0].register_forward_pre_hook(_t_pre_hook))
                            else:
                                exp_passes = _expected_block_passes(model, dir_stop_blk)
                                seen_stop = [0]

                                def _t_blk_hook(_m, _inp, blk_out, exp_passes=exp_passes, seen_stop=seen_stop):
                                    seen_stop[0] += 1
                                    if seen_stop[0] < exp_passes:
                                        return
                                    captured_h["saw_stop"] = True
                                    t = blk_out[0] if isinstance(blk_out, (tuple, list)) else blk_out
                                    if torch.is_grad_enabled() and isinstance(t, torch.Tensor) and t.requires_grad:
                                        captured_h["val"] = t
                                        raise _DirectionTargetCaptured()

                                target_hooks.append(blocks[dir_stop_blk].register_forward_hook(_t_blk_hook))
                                if dir_stop_blk == n_layers - 1:
                                    fn_mod = _find_final_norm_module(model)
                                    if fn_mod is not None:
                                        def _t_fn_pre(_m, inp):
                                            if not captured_h.get("saw_stop"):
                                                return None
                                            h = inp[0] if isinstance(inp, tuple) and inp else inp
                                            if torch.is_grad_enabled() and isinstance(h, torch.Tensor) and h.requires_grad:
                                                captured_h["pre_final_norm"] = h
                                                raise _DirectionTargetCaptured()

                                        target_hooks.append(fn_mod.register_forward_pre_hook(_t_fn_pre))

                        try:
                            out_dir = _forward_for_hidden(model, ids_dev, need_hidden_states=True)
                        except _DirectionTargetCaptured:
                            pass
                        finally:
                            for th in target_hooks:
                                th.remove()
                            target_hooks.clear()

                        hs = getattr(out_dir, "hidden_states", None) if out_dir is not None else None
                        if "val" in captured_h:
                            h_t = captured_h["val"]
                        elif dir_use_final:
                            if hs is None or len(hs) < n_layers + 1:
                                raise RuntimeError(
                                    f"Model did not return hidden_states[{n_layers}] needed for target_layer={dir_layer_label}"
                                )
                            h_t = hs[-1]
                        elif dir_stop_blk == -1:
                            if hs is None or len(hs) == 0:
                                raise RuntimeError("Model did not return hidden_states[0] for target_layer=0")
                            h_t = hs[0]
                        elif dir_stop_blk < n_layers - 1:
                            if hs is None or len(hs) <= dir_stop_blk + 1:
                                raise RuntimeError(
                                    f"Model did not return hidden_states[{dir_stop_blk + 1}] for block {dir_stop_blk}"
                                )
                            h_t = hs[dir_stop_blk + 1]
                        elif "pre_final_norm" in captured_h:
                            h_t = captured_h["pre_final_norm"]
                        elif hs is not None and len(hs) > 0:
                            h_t = hs[-1]
                        else:
                            raise RuntimeError(f"Failed to capture hidden state for block {dir_stop_blk}")

                        h_2d = (h_t[0] if h_t.dim() == 3 else h_t).float()
                        tdir_dev = dir_vec.to(device=h_2d.device, dtype=torch.float32)
                        dir_score_t = (h_2d[lo:hi] @ tdir_dev).mean()
                        obj_val = float(dir_score_t.detach().item())
                        obj = sign * dir_score_t
                        if obj.requires_grad:
                            state["in_backward"] = True
                            obj.backward()
                    elif objective == "logit_margin":
                        logits = _tail_logits(model, ids_dev, lo)[: hi - lo].float()
                        V = int(logits.shape[-1])
                        for _tt in (tok_targets, tok_contrasts):
                            if _tt is not None and int(_tt.max().item()) >= V:
                                raise ValueError(f"token id {int(_tt.max().item())} out of range for vocab size {V}")
                        if tok_targets is not None:
                            t_logit = logits.index_select(1, tok_targets.to(logits.device)).mean(dim=-1)
                        else:
                            t_logit = logits.gather(1, ids_dev[0, lo + 1 : hi + 1, None]).squeeze(1)
                        if tok_contrasts is not None:
                            c_logit = logits.index_select(1, tok_contrasts.to(logits.device)).mean(dim=-1)
                        else:
                            c_logit = logits.mean(dim=-1)
                        margin = (t_logit - c_logit).mean()
                        obj_val = float(margin.detach().item())
                        obj = sign * margin
                        if obj.requires_grad:
                            state["in_backward"] = True
                            obj.backward()
                    else:
                        logits = _tail_logits(model, ids_dev, lo)[: hi - lo].float()
                        lp = F.log_softmax(logits, dim=-1).gather(1, ids_dev[0, lo + 1 : hi + 1, None]).squeeze(1)
                        obj = sign * lp.mean()
                        obj_val = float(lp.mean().detach().item())
                        if obj.requires_grad:
                            state["in_backward"] = True
                            obj.backward()
            finally:
                for th in target_hooks:
                    th.remove()
                for gh in ghandles:
                    gh.remove()
                del logits, lp, obj, out_dir, captured_h
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            st_h: dict[str, torch.Tensor] = {}
            st_y: dict[str, torch.Tensor] = {}
            st_g: dict[str, torch.Tensor] = {}
            for mk, meta_m in grad_mods.items():
                P = fwd_counts.get(mk, 0)
                if P < 1:
                    continue
                if capture_acts and mk in st_h_passes and st_h_passes[mk]:
                    st_h[mk] = torch.cat(st_h_passes[mk], dim=0) if len(st_h_passes[mk]) > 1 else st_h_passes[mk][0]
                if capture_acts and mk in st_y_passes and st_y_passes[mk]:
                    st_y[mk] = torch.cat(st_y_passes[mk], dim=0) if len(st_y_passes[mk]) > 1 else st_y_passes[mk][0]
                mod_m = meta_m["mod"]
                D_m = int(st_y[mk].shape[-1]) if mk in st_y else int(
                    mod_m.weight.shape[0] if isinstance(mod_m, nn.Linear) else mod_m.weight.shape[1]
                )
                g_map = st_g_passes.get(mk, {})
                if g_map:
                    g_list = [g_map[p] if p in g_map else torch.zeros((T, D_m), dtype=torch.float32) for p in range(P)]
                    st_g[mk] = torch.cat(g_list, dim=0) if len(g_list) > 1 else g_list[0]
                elif mk in st_y:
                    st_g[mk] = torch.zeros_like(st_y[mk])
            tok_attr: dict[str, torch.Tensor] = {}
            if capture_acts and "x" in emb_state and "g" in emb_state:
                emb_x = emb_state["x"]
                emb_g = emb_state["g"]
                tok_attr = {
                    "grad_x_input": (emb_g * emb_x).sum(dim=-1).float().cpu(),
                    "grad_norm": emb_g.norm(dim=-1).float().cpu(),
                }
            return st_h, st_y, st_g, tok_attr, obj_val

        try:
            acts_h, acts_y, g_target_map, token_attr_out, target_obj_val = _run_backward(
                lo_t, hi_t, 1.0,
                capture_acts=True,
                objective=eff_target_obj,
                dir_vec=tdir_cpu,
                dir_stop_blk=stop_blk,
                dir_use_final=use_final_hidden,
                dir_layer_label=int(target_layer) if dir_mode else None,  # type: ignore[arg-type]
                tok_targets=t_tok_ids,
                tok_contrasts=c_tok_ids,
            )
            protect_obj_val: float | None = None
            if hi_h > lo_h:
                if protect_mode:
                    _, _, g_healthy_map, _, protect_obj_val = _run_backward(
                        lo_h, hi_h, -1.0,
                        capture_acts=False,
                        objective="direction",
                        dir_vec=pdir_cpu,
                        dir_stop_blk=p_stop_blk,
                        dir_use_final=p_use_final_hidden,
                        dir_layer_label=int(eff_protect_layer),  # type: ignore[arg-type]
                    )
                else:
                    _, _, g_healthy_map, _, _ = _run_backward(
                        lo_h, hi_h, -1.0, capture_acts=False, objective="token_nll"
                    )
            else:
                g_healthy_map = {}

            betas = tuple(float(b) for b in pareto_betas)
            for mk, meta_m in grad_mods.items():
                if mk not in acts_y or mk not in acts_h:
                    continue
                mod = meta_m["mod"]
                h_act = acts_h[mk]
                y_act = acts_y[mk]
                g_t = g_target_map.get(mk, torch.zeros_like(y_act))
                g_h = g_healthy_map.get(mk, torch.zeros_like(y_act))
                D = int(y_act.shape[-1])
                U = int(h_act.shape[-1])

                w_raw_dev = mod.weight.detach()
                w_dev = w_raw_dev.device
                w_lin_f32 = (
                    w_raw_dev if isinstance(mod, nn.Linear) else (w_raw_dev.t() if w_raw_dev.dim() == 2 else w_raw_dev)
                ).float()

                n_h_count = max(1, hi_h - lo_h)
                P = int(y_act.shape[0] // T) if (T > 0 and y_act.shape[0] % T == 0) else 1
                P = max(1, P)
                y_p = y_act.reshape(P, T, D)
                h_p = h_act.reshape(P, T, U)
                g_t_p = g_t.reshape(P, T, D)
                y_h_mean_p = y_p[:, lo_h:hi_h, :].mean(dim=1, keepdim=True) if hi_h > lo_h else torch.zeros((P, 1, D), dtype=torch.float32)
                h_h_mean_p = h_p[:, lo_h:hi_h, :].mean(dim=1, keepdim=True) if hi_h > lo_h else torch.zeros((P, 1, U), dtype=torch.float32)
                y_centered = (y_p - y_h_mean_p).reshape(P * T, D)
                h_centered = (h_p - h_h_mean_p).reshape(P * T, U)

                g_t_w = (g_t.to(device=w_dev, dtype=torch.float32) @ w_lin_f32).cpu()
                neuron_supp_attr = (g_t_w * h_act).sum(dim=0).float()
                neuron_supp_centered_attr = (g_t_w * h_centered).sum(dim=0).float()
                if hi_h > lo_h:
                    g_h_w = (g_h.to(device=w_dev, dtype=torch.float32) @ w_lin_f32).cpu()
                    neuron_dnll_attr = (-(g_h_w * h_act).sum(dim=0).float())
                    per_tok_h = (g_h_w * h_act).reshape(P, T, U).sum(dim=0)
                    neuron_fisher_attr = (float(n_h_count) * (per_tok_h * per_tok_h).sum(dim=0)).float()
                else:
                    neuron_dnll_attr = torch.zeros(U, dtype=torch.float32)
                    neuron_fisher_attr = torch.zeros(U, dtype=torch.float32)
                del w_lin_f32

                y_sq_tok = (y_act * y_act).sum(dim=1, keepdim=True) / max(1, D)
                y_sq_p = y_sq_tok.reshape(P, T, 1)
                y_healthy_rms_sq = float(y_sq_p[:, lo_h:hi_h, :].mean().item()) if hi_h > lo_h else float(y_sq_p.mean().item())
                if hi_h > lo_h:
                    g_h_scaled_tok = (g_h.reshape(P, T, D) * y_sq_p.clamp_min(0.0).sqrt()).sum(dim=0)

                # Restrict curvature eigendecomposition to the gradient-active subspace via thin SVD
                # when T <= D (or GPU Gram eigh otherwise), avoiding D x D CPU eigendecompositions.
                G_mat = torch.cat([g_t, g_h], dim=0) if hi_h > lo_h else g_t
                r = min(max(1, int(grad_rank)), D)
                if G_mat.shape[0] <= D:
                    _, s_vals, vh_mat = torch.linalg.svd(G_mat, full_matrices=False)
                    gevals_desc = s_vals * s_vals
                    g_max = float(gevals_desc[0].item()) if gevals_desc.numel() > 0 else 0.0
                    if g_max > 1e-12:
                        keep = (gevals_desc >= 1e-3 * g_max).nonzero(as_tuple=True)[0][: max(r, 1)]
                        Q: torch.Tensor | None = vh_mat[keep].t().contiguous()
                    else:
                        Q = None
                else:
                    G_dev = G_mat.to(dev)
                    G_gram = G_dev.t() @ G_dev
                    gevals, gevecs = torch.linalg.eigh(0.5 * (G_gram + G_gram.t()))
                    g_max = float(gevals[-1].item())
                    if g_max > 1e-12:
                        keep = (gevals >= 1e-3 * g_max).nonzero(as_tuple=True)[0]
                        keep = keep[-min(len(keep), max(r, 1)):]
                        Q = gevecs[:, keep].cpu().contiguous()
                    else:
                        Q = None
                    del G_dev, G_gram, gevals, gevecs

                if Q is not None:
                    gt_q = g_t @ Q
                    y_q = y_act @ Q
                    yc_q = y_centered @ Q
                    C_t_sub = 0.5 * (gt_q.t() @ y_q + y_q.t() @ gt_q)
                    C_tc_sub = 0.5 * (gt_q.t() @ yc_q + yc_q.t() @ gt_q)
                    if hi_h > lo_h:
                        gh_q = g_h @ Q
                        ghs_q = g_h_scaled_tok @ Q
                        C_h_sub = 0.5 * (gh_q.t() @ y_q + y_q.t() @ gh_q)
                        fh_raw = float(n_h_count) * (ghs_q.t() @ ghs_q)
                        F_h_sub = 0.5 * (fh_raw + fh_raw.t())
                    else:
                        k_q = int(Q.shape[1])
                        C_h_sub = torch.zeros((k_q, k_q), dtype=torch.float32)
                        F_h_sub = torch.zeros((k_q, k_q), dtype=torch.float32)
                    evals_t, evecs_t_sub = torch.linalg.eigh(C_t_sub)
                    evecs_t = Q @ evecs_t_sub
                    evals_tc, evecs_tc_sub = torch.linalg.eigh(C_tc_sub)
                    evecs_tc = Q @ evecs_tc_sub
                    evals_h, evecs_h_sub = torch.linalg.eigh(C_h_sub)
                    evecs_h = Q @ evecs_h_sub
                    evals_fh, evecs_fh_sub = torch.linalg.eigh(F_h_sub)
                    evecs_fh = Q @ evecs_fh_sub
                else:
                    evals_t = torch.zeros(1, dtype=torch.float32)
                    evecs_t = torch.eye(D, 1, dtype=torch.float32)
                    evals_tc = torch.zeros(1, dtype=torch.float32)
                    evecs_tc = torch.eye(D, 1, dtype=torch.float32)
                    evals_h = torch.zeros(1, dtype=torch.float32)
                    evecs_h = torch.eye(D, 1, dtype=torch.float32)
                    evals_fh = torch.zeros(1, dtype=torch.float32)
                    evecs_fh = torch.eye(D, 1, dtype=torch.float32)

                r_t = min(r, int(evals_t.numel()))
                idx_t = torch.argsort(evals_t, descending=True)[:r_t]
                c_target_eigvecs = evecs_t[:, idx_t].t().contiguous().float()
                c_target_eigvals = evals_t[idx_t].float()

                r_tc = min(r, int(evals_tc.numel()))
                idx_tc = torch.argsort(evals_tc, descending=True)[:r_tc]
                c_target_centered_eigvecs = evecs_tc[:, idx_tc].t().contiguous().float()
                c_target_centered_eigvals = evals_tc[idx_tc].float()

                r_h = min(r, int(evals_h.numel()))
                idx_h = torch.argsort(evals_h.abs(), descending=True)[:r_h]
                c_healthy_eigvecs = evecs_h[:, idx_h].t().contiguous().float()
                c_healthy_eigvals = evals_h[idx_h].float()

                r_fh = min(r, int(evals_fh.numel()))
                idx_fh = torch.argsort(evals_fh, descending=True)[:r_fh]
                c_fisher_eigvecs = evecs_fh[:, idx_fh].t().contiguous().float()
                c_fisher_eigvals = evals_fh[idx_fh].float()

                p_vecs = []
                use_ct_sub = (C_tc_sub if center == "healthy" else C_t_sub) if Q is not None else None
                for beta in betas:
                    if Q is not None:
                        pen_sub = (-beta * F_h_sub) if healthy_cost == "fisher" else (beta * C_h_sub)
                        _, ev_p_sub = torch.linalg.eigh(use_ct_sub + pen_sub)
                        p_vecs.append((Q @ ev_p_sub[:, -1]).contiguous().float())
                    else:
                        p_vecs.append((evecs_tc if center == "healthy" else evecs_t)[:, 0].contiguous().float())
                pareto_eigvecs = torch.stack(p_vecs, dim=0) if p_vecs else torch.empty((0, D), dtype=torch.float32)

                paired_diff = (
                    (y_p[:, lo_t:hi_t, :].mean(dim=1) - y_p[:, lo_h:hi_h, :].mean(dim=1)).sum(dim=0).float()
                    if (hi_h > lo_h and (not protect_mode or (lo_h, hi_h) != (lo_t, hi_t)))
                    else y_p[:, lo_t:hi_t, :].mean(dim=1).sum(dim=0).float()
                )
                u0 = c_target_centered_eigvecs[0] if center == "healthy" else c_target_eigvecs[0]
                target_supp_top1 = float((c_target_centered_eigvals if center == "healthy" else c_target_eigvals)[0].item())
                if healthy_cost == "fisher":
                    healthy_dnll_top1 = float(n_h_count) * float(((g_h_scaled_tok @ u0) ** 2).sum().item()) if hi_h > lo_h else 0.0
                    pareto_top1 = target_supp_top1 / max(healthy_dnll_top1, 1e-8)
                else:
                    healthy_dnll_top1 = float(-((g_h @ u0) * (y_act @ u0)).sum().item()) if hi_h > lo_h else 0.0
                    pareto_top1 = target_supp_top1 / max(abs(healthy_dnll_top1), 0.01)

                causal_attr = float((g_t * y_act).sum().item())
                gs_entry: dict[str, Any] = {
                    "kind": meta_m["kind"],
                    "layer": meta_m["layer"],
                    "n_passes": int(P),
                    "d_out": D,
                    "d_in": U,
                    "causal_attr": round(causal_attr, 6),
                    "abs_causal_attr": round(abs(causal_attr), 6),
                    "neuron_supp_attr": neuron_supp_attr,
                    "neuron_supp_centered_attr": neuron_supp_centered_attr,
                    "neuron_dnll_attr": neuron_dnll_attr,
                    "neuron_fisher_attr": neuron_fisher_attr,
                    "paired_diff": paired_diff,
                    "c_target_eigvecs": c_target_eigvecs,
                    "c_target_eigvals": c_target_eigvals,
                    "c_target_centered_eigvecs": c_target_centered_eigvecs,
                    "c_target_centered_eigvals": c_target_centered_eigvals,
                    "c_healthy_eigvecs": c_healthy_eigvecs,
                    "c_healthy_eigvals": c_healthy_eigvals,
                    "c_fisher_eigvecs": c_fisher_eigvecs,
                    "c_fisher_eigvals": c_fisher_eigvals,
                    "y_healthy_rms_sq": round(y_healthy_rms_sq, 8),
                    "pareto_eigvecs": pareto_eigvecs,
                    "pareto_betas": betas,
                    "target_supp_top1": round(target_supp_top1, 6),
                    "healthy_dnll_top1": round(healthy_dnll_top1, 6),
                    "pareto_top1": round(pareto_top1, 6),
                    "paired_diff_norm": round(float(paired_diff.norm().item()), 6),
                }
                if P > 1:
                    gs_entry["pass_causal_attr"] = [
                        round(float((g_t_p[p] * y_p[p]).sum().item()), 6) for p in range(P)
                    ]
                if dir_mode and tdir_cpu is not None:
                    direct_attr = sum(float((y_p[p, lo_t:hi_t].mean(dim=0) @ tdir_cpu).item()) for p in range(P))
                    gs_entry["direct_attr"] = round(direct_attr, 6)
                    gs_entry["indirect_attr"] = round(causal_attr - direct_attr, 6)
                nh_m = int(meta_m.get("n_heads", 0))
                if meta_m["kind"] == "attn" and nh_m > 0 and U % nh_m == 0:
                    gs_entry["n_heads"] = nh_m
                    gs_entry["head_causal_attr"] = neuron_supp_attr.reshape(nh_m, U // nh_m).sum(dim=1).float()
                grad_summary_out[mk] = gs_entry

                if grad_mode == "compressed":
                    grad_target_out[mk] = _encode_resid(g_t, grad_quant, group_size, hadamard_block)
                    if hi_h > lo_h:
                        grad_healthy_out[mk] = _encode_resid(g_h, grad_quant, group_size, hadamard_block)

            trace_meta["grad_n_target"] = int(hi_t - lo_t)
            trace_meta["grad_n_healthy"] = int(max(0, hi_h - lo_h))
            trace_meta["target_span"] = [int(lo_t), int(hi_t)]
            trace_meta["healthy_span"] = [int(lo_h), int(hi_h)]
            trace_meta["grad_center"] = center
            trace_meta["grad_healthy_cost"] = healthy_cost
            trace_meta["grad_objective"] = eff_target_obj
            trace_meta["target_obj_score"] = round(float(target_obj_val), 6)
            trace_meta["grad_healthy_objective"] = (
                "direction" if (protect_mode and hi_h > lo_h) else ("token_nll" if hi_h > lo_h else "none")
            )
            if t_tok_ids is not None:
                trace_meta["target_token_ids"] = [int(x) for x in t_tok_ids.tolist()]
            if c_tok_ids is not None:
                trace_meta["contrast_token_ids"] = [int(x) for x in c_tok_ids.tolist()]
            if protect_mode and pdir_cpu is not None:
                trace_meta["protect_layer"] = int(eff_protect_layer)  # type: ignore[arg-type]
                trace_meta["protect_layer_convention"] = eff_protect_conv
                trace_meta["protect_stop_block"] = int(p_stop_blk)  # type: ignore[arg-type]
                trace_meta["protect_use_final_hidden"] = bool(p_use_final_hidden)
                trace_meta["protect_dir_norm"] = round(float(pdir_cpu.norm().item()), 6)
                if protect_obj_val is not None:
                    trace_meta["protect_dir_score"] = round(float(protect_obj_val), 6)
            if dir_mode and tdir_cpu is not None and stop_blk is not None:
                trace_meta["target_layer"] = int(target_layer)  # type: ignore[arg-type]
                trace_meta["target_layer_convention"] = target_layer_convention
                trace_meta["target_stop_block"] = int(stop_blk)
                trace_meta["target_use_final_hidden"] = bool(use_final_hidden)
                trace_meta["target_dir_norm"] = round(float(tdir_cpu.norm().item()), 6)
                trace_meta["target_dir_score"] = round(float(target_obj_val), 6)
        finally:
            if enabled_gc and hasattr(model, "gradient_checkpointing_disable"):
                try:
                    model.gradient_checkpointing_disable()
                except Exception:
                    pass
            if was_training:
                model.train()
            for p, req in orig_req:
                p.requires_grad_(req)

    return CompactTrace(
        T, n_layers, hidden, resid_quant, group_size, eff_top_k_mlp,
        resid_out, mlp_topk_out, head_out, hadamard_block=hadamard_block,
        mlp_quant=mlp_quant, mlp_idx_mode=eff_idx_mode, mlp_group_size=mlp_group_size,
        mlp_meta=mlp_meta_out, top_k_attn=top_k_attn, attn_topk=attn_topk_out,
        cache_quant=cache_quant, cache=cache_out, head_quant=head_quant, meta=trace_meta,
        grad_quant=grad_quant, grad_mode=eff_grad_mode,
        grad_target=grad_target_out, grad_healthy=grad_healthy_out, grad_summary=grad_summary_out,
        token_attr=token_attr_out,
    )


def capture_causal_trace(
    model: nn.Module,
    input_ids: torch.Tensor | list[int],
    *,
    target_span: int | tuple[int, int] | None = None,
    healthy_span: tuple[int, int] | None = None,
    healthy_only: bool = False,
    grad_mode: str = "compressed",
    target_dir: torch.Tensor | None = None,
    target_layer: int | None = None,
    target_layer_convention: str = "block",
    protect_dir: torch.Tensor | None = None,
    protect_layer: int | None = None,
    protect_layer_convention: str | None = None,
    protect_span: int | tuple[int, int] | None = None,
    grad_objective: str = "auto",
    target_token_ids: int | list[int] | torch.Tensor | None = None,
    contrast_token_ids: int | list[int] | torch.Tensor | None = None,
    **kwargs: Any,
) -> CompactTrace:
    """Convenience wrapper calling ``capture_compact(..., capture_grad=True)``.

    When ``healthy_only=True``, traces only Fisher/gradient sensitivity over ``healthy_span``
    and marks ``meta['grad_n_target'] = 0`` so healthy-prose traces never pollute ``C_target``.
    """
    if healthy_only:
        if healthy_span is None or len(healthy_span) != 2 or int(healthy_span[1]) <= int(healthy_span[0]):
            raise ValueError("healthy_only=True requires healthy_span=(lo_h, hi_h) with hi_h > lo_h")
        lo_h, hi_h = int(healthy_span[0]), int(healthy_span[1])
        target_span = (lo_h, min(hi_h, lo_h + 1))

    ct = capture_compact(
        model,
        input_ids,
        capture_grad=True,
        grad_mode=grad_mode,
        target_span=target_span,
        healthy_span=healthy_span,
        target_dir=target_dir,
        target_layer=target_layer,
        target_layer_convention=target_layer_convention,
        protect_dir=protect_dir,
        protect_layer=protect_layer,
        protect_layer_convention=protect_layer_convention,
        protect_span=protect_span,
        grad_objective=grad_objective,
        target_token_ids=target_token_ids,
        contrast_token_ids=contrast_token_ids,
        **kwargs,
    )
    if healthy_only:
        ct.meta["grad_n_target"] = 0
    return ct


def interleave_target_and_healthy_traces(
    target_traces: Sequence[CompactTrace],
    healthy_traces: Sequence[CompactTrace],
) -> list[CompactTrace]:
    """Pair-interleave target and healthy-only traces so `0::2`/`1::2` CV halves stay balanced."""
    out: list[CompactTrace] = []
    n_pairs = max((len(target_traces) + 1) // 2, (len(healthy_traces) + 1) // 2)
    for pair_i in range(n_pairs):
        for off in range(2):
            idx = 2 * pair_i + off
            if idx < len(target_traces):
                out.append(target_traces[idx])
        for off in range(2):
            idx = 2 * pair_i + off
            if idx < len(healthy_traces):
                out.append(healthy_traces[idx])
    return out


def _annotate_mlp_wsi(rows: list[dict[str, Any]], wsi: Any | None) -> None:
    if wsi is None:
        return
    for row in rows:
        if row.get("layer") is not None:
            try:
                row["transduction_rule"] = wsi.explain_neuron(layer=row["layer"], unit=row["unit"])["transduction_rule"]
            except Exception:
                pass


def _explain_peak_vector(peak_layer: int | None, peak_vec: torch.Tensor | None,
                         peak_norm: float, wsi: Any | None) -> dict[str, Any] | None:
    if wsi is None or peak_vec is None or peak_norm <= 1e-6:
        return None
    try:
        ex = wsi.explain_vector(peak_vec / peak_vec.norm(), top_k=6)
        return {"layer": peak_layer, "promoted": ex.get("promoted_tokens", []),
                "suppressed": ex.get("suppressed_tokens", [])}
    except Exception:
        return None


def _cache_pair_delta(key: str, ta: torch.Tensor, tb: torch.Tensor, pos_a: int, pos_b: int) -> dict[str, Any] | None:
    ta, tb = ta.float(), tb.float()
    if ta.shape == tb.shape:
        return {"key": key, "delta_norm": round(float((tb - ta).norm()), 4)}
    if ta.dim() == 4 and tb.dim() == 4 and ta.shape[:2] == tb.shape[:2] and ta.shape[3] == tb.shape[3]:
        va, vb = ta[:, :, pos_a, :], tb[:, :, pos_b, :]
        return {"key": f"{key}[pos]", "delta_norm": round(float((vb - va).norm()), 4)}
    return None


def _resolve_out_proj_weight(
    mod_name: str,
    nh: int,
    d_in: int,
    model: nn.Module | None,
    wsi: Any | None,
    *,
    _mod_cache: dict[str, nn.Module] | None = None,
    _ck_names: tuple[str, ...] | None = None,
    _ck_name_set: set[str] | None = None,
) -> torch.Tensor | None:
    if nh <= 0 or d_in % nh != 0:
        return None
    hd = d_in // nh
    if model is not None:
        mods = _mod_cache if _mod_cache is not None else dict(model.named_modules())
        am = mods.get(mod_name)
        if am is not None and getattr(am, "weight", None) is not None:
            w = am.weight.detach().float().cpu()
            return w.t().reshape(nh, hd, -1) if isinstance(am, nn.Linear) else w.reshape(nh, hd, -1)
    if wsi is not None and hasattr(wsi, "ck"):
        names = tuple(wsi.ck.names()) if _ck_names is None else _ck_names
        name_set = set(names) if _ck_name_set is None else _ck_name_set
        bare = mod_name[6:] if mod_name.startswith("model.") else mod_name
        tname = next(
            (c for c in (f"{mod_name}.weight", f"{bare}.weight", f"model.{bare}.weight", f"model.language_model.{bare}.weight")
             if c in name_set),
            None,
        )
        if tname is None:
            suffix = f".{bare}.weight"
            tname = next((n for n in names if n.endswith(suffix)), None)
        if tname is not None:
            w = wsi.ck.get(tname).float().cpu()
            if w.dim() == 2:
                if hasattr(wsi, "anat"):
                    try:
                        out_ax = wsi.anat.out_axis(tname)
                    except Exception:
                        out_ax = 0 if w.shape[1] == d_in else 1
                else:
                    out_ax = 0 if w.shape[1] == d_in else 1
                return w.t().reshape(nh, hd, -1) if out_ax == 0 else w.reshape(nh, hd, -1)
    return None


@torch.no_grad()
def diff_compact(
    tr_a: CompactTrace,
    tr_b: CompactTrace,
    *,
    wsi: Any | None = None,
    model: nn.Module | None = None,
    pos_a: int = -1,
    pos_b: int = -1,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare two compressed `CompactTrace` runs of any length (`resid`, `mlp_topk`, `head_norms`, `attn_topk`, `cache`)."""
    layer_curve = []
    peak_layer, peak_vec, peak_norm = None, None, -1.0
    for l in sorted(set(tr_a.resid) & set(tr_b.resid)):
        va = tr_a.dequant_resid(l)[pos_a]
        vb = tr_b.dequant_resid(l)[pos_b]
        dv = vb - va
        dn = float(dv.norm())
        rel = dn / max(float(va.norm()), 1e-12)
        cos = float(torch.nn.functional.cosine_similarity(va[None], vb[None]))
        layer_curve.append({"layer": l, "delta_norm": round(dn, 4), "rel_delta": round(rel, 5), "cosine": round(cos, 5)})
        if (l > 0 or len(tr_a.resid) == 1) and dn > peak_norm:
            peak_layer, peak_vec, peak_norm = l, dv, dn

    mlp_hits = []
    for l in sorted(set(tr_a.mlp_topk) & set(tr_b.mlp_topk)):
        ia_t, va_t = tr_a.dequant_mlp_topk(l)
        ib_t, vb_t = tr_b.dequant_mlp_topk(l)
        ma = dict(zip(ia_t[pos_a].tolist(), va_t[pos_a].tolist()))
        mb = dict(zip(ib_t[pos_b].tolist(), vb_t[pos_b].tolist()))
        for u in set(ma) | set(mb):
            da, db = ma.get(u, 0.0), mb.get(u, 0.0)
            signed = db - da
            mlp_hits.append({
                "layer": l, "unit": int(u),
                "contrib_a": round(da, 4), "contrib_b": round(db, 4),
                "resid_shift_norm": round(abs(signed), 4),
                "signed_shift": round(signed, 4),
            })
    mlp_hits.sort(key=lambda r: -r["resid_shift_norm"])
    top_mlp = mlp_hits[:top_k]
    _annotate_mlp_wsi(top_mlp, wsi)

    head_hits = []
    attn_cache_a = {l: tr_a.dequant_attn_topk(l) for l in tr_a.attn_topk}
    attn_cache_b = {l: tr_b.dequant_attn_topk(l) for l in tr_b.attn_topk}
    mod_cache = dict(model.named_modules()) if model is not None else None
    ck_names = tuple(wsi.ck.names()) if (wsi is not None and hasattr(wsi, "ck") and callable(getattr(wsi.ck, "names", None))) else ()
    ck_name_set = set(ck_names)
    for mod in sorted(set(tr_a.head_norms) & set(tr_b.head_norms)):
        ha, hb = tr_a.head_norms[mod], tr_b.head_norms[mod]
        li = ha["layer"]
        nh = int(ha["n_heads"])
        da = ha["l2"][pos_a].float()
        db = hb["l2"][pos_b].float()
        diff = db - da
        xa = tr_a.dequant_head_in(mod)
        xb = tr_b.dequant_head_in(mod)
        if xa is not None and xb is not None:
            xa_p = xa[pos_a].float().reshape(nh, -1)
            xb_p = xb[pos_b].float().reshape(nh, -1)
            dx = xb_p - xa_p
            w_heads = _resolve_out_proj_weight(
                mod, nh, dx.numel(), model, wsi,
                _mod_cache=mod_cache, _ck_names=ck_names, _ck_name_set=ck_name_set,
            )
            if w_heads is not None:
                norms = torch.einsum("hd,hdo->ho", dx, w_heads).norm(dim=1)
            else:
                gain = (da + db) / (xa_p.norm(dim=1) + xb_p.norm(dim=1)).clamp_min(1e-8)
                norms = dx.norm(dim=1) * gain
        else:
            norms = diff.abs()
        for hid in norms.topk(min(3, len(norms))).indices.tolist():
            row: dict[str, Any] = {
                "layer": li, "module": mod, "head": int(hid),
                "l2_a": round(float(da[hid]), 4), "l2_b": round(float(db[hid]), 4),
                "out_delta_norm": round(float(norms[hid]), 4),
                "signed_l2_delta": round(float(diff[hid]), 4),
            }
            if li in attn_cache_a and li in attn_cache_b:
                idx_a, w_a = attn_cache_a[li]
                idx_b, w_b = attn_cache_b[li]
                if hid < idx_a.shape[0] and hid < idx_b.shape[0]:
                    row["attn_a_top_src_pos"] = int(idx_a[hid, pos_a, 0])
                    row["attn_a_top_weight"] = round(float(w_a[hid, pos_a, 0]), 4)
                    row["attn_b_top_src_pos"] = int(idx_b[hid, pos_b, 0])
                    row["attn_b_top_weight"] = round(float(w_b[hid, pos_b, 0]), 4)
                    ma_att = dict(zip(idx_a[hid, pos_a].tolist(), w_a[hid, pos_a].tolist()))
                    mb_att = dict(zip(idx_b[hid, pos_b].tolist(), w_b[hid, pos_b].tolist()))
                    l1 = sum(abs(mb_att.get(u, 0.0) - ma_att.get(u, 0.0)) for u in set(ma_att) | set(mb_att))
                    row["attn_topk_l1_shift"] = round(l1, 4)
            head_hits.append(row)
    head_hits.sort(key=lambda r: -r["out_delta_norm"])

    cache_shifts = []
    for k in sorted(set(tr_a.cache) & set(tr_b.cache)):
        item = _cache_pair_delta(k, tr_a.dequant_cache(k), tr_b.dequant_cache(k), pos_a, pos_b)
        if item is not None:
            cache_shifts.append(item)
    cache_shifts.sort(key=lambda r: -r["delta_norm"])

    return {
        "len_a": tr_a.seq_len, "len_b": tr_b.seq_len,
        "pos_a": pos_a, "pos_b": pos_b,
        "peak_residual_layer": peak_layer, "peak_residual_delta_norm": round(peak_norm, 4),
        "layer_curve": layer_curve,
        "top_mlp_contributors": top_mlp,
        "top_attention_heads": head_hits[:top_k],
        "top_cache_deltas": cache_shifts[:top_k],
        "peak_vector_readout": _explain_peak_vector(peak_layer, peak_vec, peak_norm, wsi),
    }


# ----------------------------------------------------------------------------------------------- hybrid CompactTrace + WSI discovery
def _fisher_dir(x_healthy: torch.Tensor, x_loop: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Normalized healthy->loop mean shift direction and Fisher separation ratio."""
    mh, ml = x_healthy.mean(dim=0), x_loop.mean(dim=0)
    d = ml - mh
    dn = float(d.norm())
    if dn < 1e-8:
        return torch.zeros_like(d), 0.0
    u = d / dn
    ph, pl = x_healthy @ u, x_loop @ u
    vh = float(ph.var(unbiased=False)) if len(ph) > 1 else 0.0
    vl = float(pl.var(unbiased=False)) if len(pl) > 1 else 0.0
    score = float((pl.mean() - ph.mean()) ** 2 / max(0.5 * (vh + vl), 1e-8))
    return u, score


def _split_ab(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Deterministic Half-A / Half-B split (sequence-level when >= 2 traces, else token-level)."""
    if len(tensors) >= 2:
        return torch.cat(tensors[0::2], dim=0), torch.cat(tensors[1::2], dim=0), "sequence_ab"
    t = tensors[0]
    if t.shape[0] >= 4:
        return t[0::2], t[1::2], "token_ab"
    return t, t, "in_sample"


def _heldout_dir_dprime(
    h_list: list[torch.Tensor],
    l_list: list[torch.Tensor],
) -> tuple[torch.Tensor, float, float, str]:
    """Fit unit shift direction `u` on Half A, evaluate held-out `d'` and Fisher ratio on Half B."""
    hA, hB, mh = _split_ab(h_list)
    lA, lB, ml = _split_ab(l_list)
    mode = "sequence_ab" if (mh == "sequence_ab" and ml == "sequence_ab") else (
        "token_ab" if ("in_sample" not in (mh, ml)) else "in_sample"
    )
    d = lA.mean(dim=0) - hA.mean(dim=0)
    dn = float(d.norm())
    if dn < 1e-8:
        return torch.zeros_like(d), 0.0, 0.0, mode
    u = d / dn
    phB, plB = hB @ u, lB @ u
    vhB = float(phB.var(unbiased=False)) if len(phB) > 1 else 0.0
    vlB = float(plB.var(unbiased=False)) if len(plB) > 1 else 0.0
    denom = math.sqrt(max(0.5 * (vhB + vlB), 1e-8))
    dprime = float((plB.mean() - phB.mean()) / denom)
    fisher = float(max(dprime, 0.0) ** 2)
    return u, dprime, fisher, mode


def _weight_fingerprint(path: str, w_orig: torch.Tensor) -> dict[str, Any]:
    from .moipatch import _dtype_name, _sha256, module_suffix

    valid_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    try:
        mod_name = module_suffix(path)
    except ValueError:
        mod_name = path
    dt = _dtype_name(w_orig.dtype) if w_orig.dtype in valid_dtypes else "float32"
    return {
        "module": mod_name,
        "shape": tuple(w_orig.shape),
        "base_dtype": dt,
        "base_sha256": _sha256(w_orig) if w_orig.dtype in valid_dtypes else "",
    }


def _resolve_attn_meta(
    mod_name: str,
    model: nn.Module | None,
    wsi: Any | None,
    *,
    _mod_cache: dict[str, nn.Module] | None = None,
    _ck_names: tuple[str, ...] = (),
    _ck_name_set: set[str] | None = None,
) -> dict[str, Any] | None:
    """Resolve `MoIPatch` fingerprint metadata for an attention `o_proj` / `out_proj` module."""
    if model is not None:
        mods = _mod_cache if _mod_cache is not None else dict(model.named_modules())
        bare = mod_name[6:] if mod_name.startswith("model.") else mod_name
        am = next(
            (mods[c] for c in (mod_name, bare, f"model.{bare}", f"model.language_model.{bare}") if c in mods),
            None,
        )
        if am is None:
            suffix = f".{bare}"
            am = next((m for n, m in mods.items() if n.endswith(suffix)), None)
        if am is not None and getattr(am, "weight", None) is not None:
            return _weight_fingerprint(mod_name, am.weight.detach())
    if wsi is not None and hasattr(wsi, "ck"):
        name_set = set(_ck_names) if _ck_name_set is None else _ck_name_set
        bare = mod_name[6:] if mod_name.startswith("model.") else mod_name
        tname = next(
            (c for c in (f"{mod_name}.weight", f"{bare}.weight", f"model.{bare}.weight", f"model.language_model.{bare}.weight")
             if c in name_set),
            None,
        )
        if tname is None:
            suffix = f".{bare}.weight"
            tname = next((n for n in _ck_names if n.endswith(suffix)), None)
        if tname is not None:
            w_orig = wsi.ck.get(tname).clone()
            mod_full = tname[:-7] if tname.endswith(".weight") else tname
            return _weight_fingerprint(mod_full, w_orig)
    return None


def _resolve_mlp_downs(
    model: nn.Module | None,
    wsi: Any | None,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, Any]]]:
    """Resolve per-layer MLP `W_down` in `(D, U)` layout plus `MoIPatch` fingerprint metadata."""
    from .moipatch import _dtype_name, _sha256, module_suffix

    w_downs: dict[int, torch.Tensor] = {}
    mod_meta: dict[int, dict[str, Any]] = {}
    valid_dtypes = (torch.bfloat16, torch.float16, torch.float32)

    def _suffix(path: str) -> str:
        try:
            return module_suffix(path)
        except ValueError:
            return path

    if model is not None:
        mods = dict(model.named_modules())
        for l, g in enumerate(find_mlps(model)):
            li = _layer_idx_from_name(g.down)
            li = l if li is None else li
            m = mods[g.down]
            w_orig = m.weight.detach()
            w_downs[li] = w_orig.float().cpu() if isinstance(m, nn.Linear) else w_orig.float().cpu().t()
            dt = _dtype_name(w_orig.dtype) if w_orig.dtype in valid_dtypes else "float32"
            mod_meta[li] = {
                "module": _suffix(g.down),
                "shape": tuple(w_orig.shape),
                "base_dtype": dt,
                "base_sha256": _sha256(w_orig) if w_orig.dtype in valid_dtypes else "",
            }
        return w_downs, mod_meta

    if wsi is not None and hasattr(wsi, "anat") and hasattr(wsi, "ck"):
        from .indexer import _to_residual_columns

        for l in sorted(wsi.anat.layers):
            tname = wsi.anat.mlp_out(l)
            if not tname:
                continue
            w_orig = wsi.ck.get(tname).clone()
            out_ax = wsi.anat.out_axis(tname)
            w_downs[l] = _to_residual_columns(w_orig.float().cpu(), out_ax)
            mod_full = tname[:-7] if tname.endswith(".weight") else tname
            dt = _dtype_name(w_orig.dtype) if w_orig.dtype in valid_dtypes else "float32"
            mod_meta[l] = {
                "module": _suffix(mod_full),
                "shape": tuple(w_orig.shape),
                "base_dtype": dt,
                "base_sha256": _sha256(w_orig) if w_orig.dtype in valid_dtypes else "",
            }
        return w_downs, mod_meta

    raise ValueError("discover_compact_circuit requires at least one of `model` or `wsi`.")


def _unpack_discovery_item(item: Any, forced_loop: bool | None) -> tuple[CompactTrace, int, bool, int | None]:
    """Normalize `(meta, ct, ...)` / `(ct, prompt_len, onset_resp)` / `CompactTrace` items."""
    if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], dict):
        meta, ct = item[0], item[1]
        p = int(meta.get("prompt_len", 0))
        is_loop = bool(meta.get("loop", False)) if forced_loop is None else forced_loop
        onset_resp = meta.get("onset_resp")
    elif isinstance(item, tuple) and isinstance(item[0], CompactTrace):
        ct = item[0]
        p = int(item[1]) if len(item) >= 2 and item[1] is not None else 0
        onset_resp = item[2] if len(item) >= 3 else None
        is_loop = (onset_resp is not None) if forced_loop is None else forced_loop
        if forced_loop is None and len(item) < 3:
            det = ct.detect_loop_onset(
                layer=min(15, ct.n_layers - 1),
                window=min(16, max(2, ct.seq_len // 4)),
                thr=0.96,
            )
            onset_tok = det.get("onset_token")
            return ct, p, onset_tok is not None, onset_tok
    elif isinstance(item, CompactTrace):
        ct, p, onset_resp = item, 0, None
        det_tok = None
        if forced_loop is not False:
            det = ct.detect_loop_onset(
                layer=min(15, ct.n_layers - 1),
                window=min(16, max(2, ct.seq_len // 4)),
                thr=0.96,
            )
            det_tok = det.get("onset_token")
        is_loop = (det_tok is not None) if forced_loop is None else forced_loop
        if is_loop:
            return ct, p, True, (det_tok if det_tok is not None else ct.seq_len // 2)
        return ct, p, False, None
    else:
        raise TypeError(f"Unsupported trace entry type in discover_compact_circuit: {type(item).__name__}")

    if not is_loop:
        return ct, p, False, None
    if onset_resp is not None:
        return ct, p, True, min(p + int(onset_resp), max(0, ct.seq_len - 1))
    det = ct.detect_loop_onset(
        layer=min(15, ct.n_layers - 1),
        window=min(16, max(2, ct.seq_len // 4)),
        thr=0.96,
    )
    onset_tok = det.get("onset_token")
    if onset_tok is None:
        onset_tok = max(p, ct.seq_len // 2)
    return ct, p, True, min(int(onset_tok), max(0, ct.seq_len - 1))


def discover_compact_circuit(
    traces: list[Any],
    *,
    loop_traces: list[Any] | None = None,
    model: nn.Module | None = None,
    wsi: Any | None = None,
    alpha: float = 6.0,
    top_k_neurons: int = 64,
    top_k_heads: int = 2,
    healthy_window: int = 96,
    loop_window: int = 192,
    origin_range: tuple[int, int] | None = None,
    attach_router: bool = True,
    candidate_alphas: tuple[float, ...] = (4.0, 8.0, 12.0),
) -> dict[str, Any]:
    """Blind multi-site circuit discovery from `CompactTrace` + static `WSI` weights.

    Upgrades integrated:
      1. Unified MLP (`down_proj`) + Attention (`o_proj`/`out_proj` via `W_O`-projected `head_quant`)
         site competition on the same residual-space scale.
      2. Held-out Half-A / Half-B split (`d'`) for unbiased layer, head, and router ranking.
      3. Automatic residual-stream router synthesis (`layer`, `dir`, `thr`) with held-out TPR/FPR audit.
      4. Hint-free `MoIPatch` synthesis (`unified_top2`, `blind_top2`, `attn_top2`, `wsi_pair`, `wsi_top3`).
    """
    from .moipatch import Edit, MoIPatch

    w_downs, mod_meta = _resolve_mlp_downs(model, wsi)
    n_layers = len(w_downs)
    if n_layers == 0:
        raise ValueError("No MLP down-projection weights found for discovery.")
    layers_sorted = sorted(w_downs)
    d_inter_by_layer = {l: int(w_downs[l].shape[1]) for l in layers_sorted}

    entries: list[tuple[CompactTrace, int, bool, int | None]] = []
    if loop_traces is not None:
        entries.extend(_unpack_discovery_item(it, False) for it in traces)
        entries.extend(_unpack_discovery_item(it, True) for it in loop_traces)
    else:
        entries.extend(_unpack_discovery_item(it, None) for it in traces)

    mod_cache = dict(model.named_modules()) if model is not None else None
    ck_names = tuple(wsi.ck.names()) if (wsi is not None and hasattr(wsi, "ck") and callable(getattr(wsi.ck, "names", None))) else ()
    ck_name_set = set(ck_names)
    w_heads_cache: dict[str, torch.Tensor | None] = {}
    attn_meta_cache: dict[str, dict[str, Any] | None] = {}

    healthy_res: dict[int, list[torch.Tensor]] = {l: [] for l in layers_sorted}
    loop_res: dict[int, list[torch.Tensor]] = {l: [] for l in layers_sorted}
    healthy_y: dict[int, list[torch.Tensor]] = {l: [] for l in layers_sorted}
    loop_y: dict[int, list[torch.Tensor]] = {l: [] for l in layers_sorted}
    healthy_xsum: dict[int, list[torch.Tensor]] = {l: [] for l in layers_sorted}
    loop_xsum: dict[int, list[torch.Tensor]] = {l: [] for l in layers_sorted}
    healthy_head: dict[int, list[torch.Tensor]] = {l: [] for l in layers_sorted}
    loop_head: dict[int, list[torch.Tensor]] = {l: [] for l in layers_sorted}
    healthy_att_focus: dict[int, list[torch.Tensor]] = {}
    loop_att_focus: dict[int, list[torch.Tensor]] = {}
    healthy_attn_y: dict[str, list[torch.Tensor]] = {}
    loop_attn_y: dict[str, list[torch.Tensor]] = {}
    healthy_attn_h: dict[str, list[torch.Tensor]] = {}
    loop_attn_h: dict[str, list[torch.Tensor]] = {}
    attn_info: dict[str, tuple[int, int, int]] = {}  # mod -> (layer, n_heads, head_dim)

    for ct, p, is_loop, ons in entries:
        T = ct.seq_len
        if T == 0:
            continue
        if is_loop:
            ons_t = max(0, min(ons if ons is not None else T // 2, T - 1))
            h_start = min(p, T)
            h_end = max(h_start, min(ons_t, p + healthy_window, T))
            h_idx = torch.arange(h_start, h_end)
            l_start = max(ons_t, T - loop_window)
            l_idx = torch.arange(l_start, T)
        else:
            h_idx = torch.arange(min(p, T), T)
            l_idx = torch.empty(0, dtype=torch.long)

        for l in layers_sorted:
            if l in ct.resid:
                R = ct.dequant_resid(l)
                if len(h_idx):
                    healthy_res[l].append(R[h_idx])
                if len(l_idx):
                    loop_res[l].append(R[l_idx])

            if l in ct.mlp_topk:
                idx_m, val_m = ct.dequant_mlp_topk(l)
                w_raw = w_downs[l]
                d_inter = d_inter_by_layer[l]
                wn = w_raw.norm(dim=0).clamp_min(1e-12)
                if len(h_idx):
                    ih, vh = idx_m[h_idx], val_m[h_idx]
                    xh = torch.zeros((len(h_idx), d_inter), dtype=torch.float32)
                    xh.scatter_(1, ih, vh / wn[ih])
                    healthy_y[l].append(xh @ w_raw.T)
                    healthy_xsum[l].append(xh.mean(dim=0, keepdim=True))
                if len(l_idx):
                    il, vl = idx_m[l_idx], val_m[l_idx]
                    xl = torch.zeros((len(l_idx), d_inter), dtype=torch.float32)
                    xl.scatter_(1, il, vl / wn[il])
                    loop_y[l].append(xl @ w_raw.T)
                    loop_xsum[l].append(xl.mean(dim=0, keepdim=True))

        for mod, h in ct.head_norms.items():
            li = h["layer"]
            if li is None or li not in healthy_head:
                continue
            l2 = h["l2"].float()
            if len(h_idx):
                healthy_head[li].append(l2[h_idx])
            if len(l_idx):
                loop_head[li].append(l2[l_idx])

            x_in = ct.dequant_head_in(mod)
            if x_in is not None:
                nh = int(h["n_heads"])
                d_in = int(x_in.shape[-1])
                if mod not in w_heads_cache:
                    w_heads_cache[mod] = _resolve_out_proj_weight(
                        mod, nh, d_in, model, wsi,
                        _mod_cache=mod_cache, _ck_names=ck_names, _ck_name_set=ck_name_set,
                    )
                    attn_meta_cache[mod] = _resolve_attn_meta(
                        mod, model, wsi,
                        _mod_cache=mod_cache, _ck_names=ck_names, _ck_name_set=ck_name_set,
                    )
                w_heads = w_heads_cache[mod]
                if w_heads is not None and nh > 0 and d_in % nh == 0:
                    hd = d_in // nh
                    attn_info[mod] = (int(li), nh, hd)
                    if len(h_idx):
                        yh_heads = torch.einsum("thd,hdo->tho", x_in[h_idx].float().reshape(-1, nh, hd), w_heads)
                        healthy_attn_y.setdefault(mod, []).append(yh_heads.sum(dim=1))
                        healthy_attn_h.setdefault(mod, []).append(yh_heads.mean(dim=0, keepdim=True))
                    if len(l_idx):
                        yl_heads = torch.einsum("thd,hdo->tho", x_in[l_idx].float().reshape(-1, nh, hd), w_heads)
                        loop_attn_y.setdefault(mod, []).append(yl_heads.sum(dim=1))
                        loop_attn_h.setdefault(mod, []).append(yl_heads.mean(dim=0, keepdim=True))

        for li in ct.attn_topk:
            _, a_w = ct.dequant_attn_topk(li)
            top2_mass = a_w[:, :, : min(2, a_w.shape[-1])].sum(dim=-1).T  # (T, H)
            if len(h_idx):
                healthy_att_focus.setdefault(li, []).append(top2_mass[h_idx])
            if len(l_idx):
                loop_att_focus.setdefault(li, []).append(top2_mass[l_idx])

    layer_stats: list[dict[str, Any]] = []
    u_mlp_by_layer: dict[int, torch.Tensor] = {}
    topk_by_layer: dict[int, torch.Tensor] = {}
    dx_by_layer: dict[int, torch.Tensor] = {}
    split_mode = "in_sample"

    # Automatic residual router search across all layers (fit on Half A, audited on Half B)
    best_router: dict[str, Any] | None = None
    best_router_report: dict[str, Any] | None = None
    best_router_key: tuple[float, float, float] = (-1.0, -1.0, -1e9)

    for l in layers_sorted:
        if not loop_y[l] or not healthy_y[l]:
            continue
        u_m, dp_m, f_m, sm_m = _heldout_dir_dprime(healthy_y[l], loop_y[l])
        split_mode = sm_m
        f_r, dp_r = 0.0, 0.0
        if healthy_res[l] and loop_res[l]:
            u_r_dir, dp_r, f_r, _ = _heldout_dir_dprime(healthy_res[l], loop_res[l])
            if float(u_r_dir.norm()) > 0.5:
                rhA, rhB, _ = _split_ab(healthy_res[l])
                rlA, rlB, _ = _split_ab(loop_res[l])
                thr_r = 0.5 * (float((rhA @ u_r_dir).mean()) + float((rlA @ u_r_dir).mean()))
                tpr = float(((rlB @ u_r_dir) > thr_r).float().mean())
                fpr = float(((rhB @ u_r_dir) > thr_r).float().mean())
                bal = 0.5 * (tpr + (1.0 - fpr))
                rkey = (bal, -fpr, dp_r)
                if rkey > best_router_key:
                    best_router_key = rkey
                    best_router = {
                        "layer": int(l + 1),  # hidden_states[l + 1] == decoder block l output
                        "readout": "last_token",
                        "dir": u_r_dir.float().cpu(),
                        "thr": round(thr_r, 6),
                    }
                    best_router_report = {
                        "block_layer": int(l),
                        "hidden_state_layer": int(l + 1),
                        "thr": round(thr_r, 4),
                        "dprime": round(dp_r, 4),
                        "tpr": round(tpr, 4),
                        "fpr": round(fpr, 4),
                        "balanced_acc": round(bal, 4),
                        "warning": "high_fpr" if fpr > 0.15 else None,
                    }
        u_mlp_by_layer[l] = u_m

        # Use Half-A mean activation shift for neuron selection (consistent with held-out Half A/B)
        hxA, _, _ = _split_ab(healthy_xsum[l])
        lxA, _, _ = _split_ab(loop_xsum[l])
        dx = lxA.mean(dim=0) - hxA.mean(dim=0)
        dx_by_layer[l] = dx
        k_pick = min(top_k_neurons, d_inter_by_layer[l])
        topk_by_layer[l] = (dx * (u_m @ w_downs[l])).topk(k_pick).indices.sort().values

        head_sep = 0.0
        if healthy_head[l] and loop_head[l]:
            hh_t, hl_t = torch.cat(healthy_head[l], dim=0), torch.cat(loop_head[l], dim=0)
            denom_h = (0.5 * (hl_t.var(dim=0, unbiased=False) + hh_t.var(dim=0, unbiased=False))).sqrt().clamp_min(1e-6)
            head_sep = float(((hl_t.mean(dim=0) - hh_t.mean(dim=0)).abs() / denom_h).max())

        attn_sep = 0.0
        if l in healthy_att_focus and l in loop_att_focus:
            ah_t, al_t = torch.cat(healthy_att_focus[l], dim=0), torch.cat(loop_att_focus[l], dim=0)
            denom_a = (0.5 * (al_t.var(dim=0, unbiased=False) + ah_t.var(dim=0, unbiased=False))).sqrt().clamp_min(1e-6)
            attn_sep = float(((al_t.mean(dim=0) - ah_t.mean(dim=0)).abs() / denom_a).max())

        layer_stats.append({
            "layer": l,
            "is_full_attn": l in healthy_att_focus,
            "dprime_mlp": round(dp_m, 4),
            "dprime_resid": round(dp_r, 4),
            "fisher_mlp": round(f_m, 4),
            "fisher_resid": round(f_r, 4),
            "head_max_d": round(head_sep, 4),
            "attn_focus_d": round(attn_sep, 4),
            "hybrid_score": round(f_m * (1.0 + 0.35 * head_sep + 0.5 * attn_sep), 4),
        })

    if not layer_stats:
        raise ValueError("Need at least one healthy and one looping token span to discover a circuit.")

    # W_O-projected Attention site scoring (same held-out d' scale as MLP sites)
    attn_stats: list[dict[str, Any]] = []
    u_attn_by_mod: dict[str, torch.Tensor] = {}
    head_cols_by_mod: dict[str, torch.Tensor] = {}
    top_heads_by_mod: dict[str, list[int]] = {}
    for mod in sorted(set(healthy_attn_y) & set(loop_attn_y)):
        if attn_meta_cache.get(mod) is None:
            continue
        li, nh, hd = attn_info[mod]
        u_a, dp_a, f_a, _ = _heldout_dir_dprime(healthy_attn_y[mod], loop_attn_y[mod])
        u_attn_by_mod[mod] = u_a
        hhA, _, _ = _split_ab(healthy_attn_h[mod])
        lhA, _, _ = _split_ab(loop_attn_h[mod])
        dh_vec = lhA.mean(dim=0) - hhA.mean(dim=0)  # (H, D)
        head_scores = dh_vec @ u_a  # (H,)
        kh = min(max(1, int(top_k_heads)), nh)
        h_pick = head_scores.topk(kh).indices.sort().values.tolist()
        top_heads_by_mod[mod] = [int(x) for x in h_pick]
        head_cols_by_mod[mod] = torch.cat([torch.arange(h * hd, (h + 1) * hd, dtype=torch.long) for h in h_pick])
        attn_stats.append({
            "layer": li,
            "module": attn_meta_cache[mod]["module"],
            "raw_module": mod,
            "n_heads": nh,
            "head_dim": hd,
            "dprime_attn": round(dp_a, 4),
            "fisher_attn": round(f_a, 4),
            "top_heads": top_heads_by_mod[mod],
        })
    attn_stats.sort(key=lambda r: -r["dprime_attn"])

    # Unified site ranking across BOTH MLP (`down_proj`) and Attention (`o_proj`/`out_proj`)
    site_rankings: list[dict[str, Any]] = [
        {
            "site_type": "mlp",
            "layer": r["layer"],
            "module": mod_meta[r["layer"]]["module"],
            "dprime": r["dprime_mlp"],
        }
        for r in layer_stats
    ] + [
        {
            "site_type": "attn",
            "layer": r["layer"],
            "module": r["module"],
            "raw_module": r["raw_module"],
            "dprime": r["dprime_attn"],
            "top_heads": r["top_heads"],
        }
        for r in attn_stats
    ]
    site_rankings.sort(key=lambda r: -r["dprime"])

    mlp_ranked = [r["layer"] for r in sorted(layer_stats, key=lambda x: -x["dprime_mlp"])]
    hybrid_ranked = [r["layer"] for r in sorted(layer_stats, key=lambda x: -x["hybrid_score"])]

    fa_layers = sorted(l for l in layers_sorted if l in healthy_att_focus)
    if not fa_layers:
        fa_layers = sorted(s["layer"] for s in layer_stats if s["is_full_attn"])

    # Truly blind origin selection across ALL layers (no hardcoded layer-index or layer-type hints)
    if origin_range is not None:
        orig_lo, orig_hi = origin_range
        origins = [l for l in hybrid_ranked if orig_lo <= l <= orig_hi][:2]
        if not origins:
            origins = hybrid_ranked[:2]
    else:
        origins = hybrid_ranked[:2]

    hidden_dim = int(next(iter(w_downs.values())).shape[0])
    p_svd: dict[int, torch.Tensor] = {}
    for l, idx_k in topk_by_layer.items():
        U_svd, _, _ = torch.linalg.svd(w_downs[l][:, idx_k].float(), full_matrices=False)
        p_svd[l] = U_svd[:, 0]

    wsi_resonance_table: list[dict[str, Any]] = []
    res_candidates: dict[tuple[int, int], tuple[float, torch.Tensor, torch.Tensor]] = {}
    for l_src in origins:
        for l_dst in sorted(u_mlp_by_layer):
            if l_dst <= l_src or l_dst not in p_svd:
                continue
            echo_svd_cos = abs(float(torch.dot(p_svd[l_src], p_svd[l_dst])))
            echo_sigma = echo_svd_cos * math.sqrt(hidden_dim)
            dot_u = float(torch.dot(u_mlp_by_layer[l_src], u_mlp_by_layer[l_dst]))
            dir_cos = abs(dot_u)
            wsi_score = echo_svd_cos * (1.0 + dir_cos)
            sgn = 1.0 if dot_u >= 0.0 else -1.0
            u_res = F.normalize(u_mlp_by_layer[l_dst] + 0.5 * sgn * u_mlp_by_layer[l_src], dim=0)
            k_dst = min(top_k_neurons, d_inter_by_layer[l_dst])
            idx_res = (dx_by_layer[l_dst] * (u_res @ w_downs[l_dst])).topk(k_dst).indices.sort().values
            wsi_resonance_table.append({
                "src_layer": l_src,
                "dst_layer": l_dst,
                "is_full_attn": l_dst in fa_layers,
                "echo_svd_cos": round(echo_svd_cos, 4),
                "echo_sigma": round(echo_sigma, 2),
                "dir_cos": round(dir_cos, 4),
                "wsi_resonance_score": round(wsi_score, 4),
            })
            res_candidates[(l_src, l_dst)] = (wsi_score, u_res, idx_res)
    wsi_resonance_table.sort(key=lambda r: -r["wsi_resonance_score"])

    # Pure resonance ranking across all downstream layers (no layer-type or `< max_layer` exclusion)
    dst_cands = [r for r in wsi_resonance_table if r["dst_layer"] not in origins]
    if not dst_cands:
        dst_cands = wsi_resonance_table[:1]

    l_res_dst = dst_cands[0]["dst_layer"] if dst_cands else None
    win_src = dst_cands[0]["src_layer"] if dst_cands else (origins[0] if origins else None)

    wsi_dir_ov: dict[int, torch.Tensor] = {}
    wsi_idx_ov: dict[int, torch.Tensor] = {}
    if l_res_dst is not None and win_src is not None and (win_src, l_res_dst) in res_candidates:
        _, u_r, idx_r = res_candidates[(win_src, l_res_dst)]
        wsi_dir_ov[l_res_dst] = u_r
        wsi_idx_ov[l_res_dst] = idx_r

    wsi_pair_layers = [win_src, l_res_dst] if (win_src is not None and l_res_dst is not None) else []
    wsi_fa3_layers = list(dict.fromkeys(origins[:2] + ([l_res_dst] if l_res_dst is not None else [])))
    router_obj = best_router if attach_router else None

    def _make_patch(edits: list[Edit], name: str) -> MoIPatch:
        return MoIPatch(
            edits=edits,
            router=router_obj,
            meta={
                "name": name,
                "split_mode": split_mode,
                "wsi_origins": origins,
                "wsi_resonant_dst": l_res_dst,
                "wsi_pair_layers": wsi_pair_layers,
                "wsi_fa3_layers": wsi_fa3_layers,
                "router_report": best_router_report,
            },
        )

    def build_patch(
        layers_pick: list[int],
        dir_ov: dict[int, torch.Tensor] | None = None,
        idx_ov: dict[int, torch.Tensor] | None = None,
        *,
        use_wsi_resonance: bool = True,
        patch_alpha: float | None = None,
    ) -> MoIPatch:
        d_ov = dir_ov if dir_ov is not None else (wsi_dir_ov if use_wsi_resonance else {})
        i_ov = idx_ov if idx_ov is not None else (wsi_idx_ov if use_wsi_resonance else {})
        a_val = float(alpha if patch_alpha is None else patch_alpha)
        edits = []
        for l in layers_pick:
            meta_l = mod_meta[l]
            u_use = d_ov.get(l, u_mlp_by_layer[l])
            idx_use = i_ov.get(l, topk_by_layer[l])
            edits.append(Edit(
                module=meta_l["module"],
                u=u_use.double(),
                alpha=a_val,
                idx=idx_use.long(),
                base_sha256=meta_l["base_sha256"],
                base_dtype=meta_l["base_dtype"],
                shape=meta_l["shape"],
            ))
        return _make_patch(edits, "compact-wsi-discovered")

    def build_sites_patch(sites: list[dict[str, Any]], *, patch_alpha: float | None = None, name: str = "compact-unified") -> MoIPatch:
        a_val = float(alpha if patch_alpha is None else patch_alpha)
        edits = []
        for s in sites:
            if s["site_type"] == "mlp":
                l = int(s["layer"])
                meta_l = mod_meta[l]
                edits.append(Edit(
                    module=meta_l["module"],
                    u=u_mlp_by_layer[l].double(),
                    alpha=a_val,
                    idx=topk_by_layer[l].long(),
                    base_sha256=meta_l["base_sha256"],
                    base_dtype=meta_l["base_dtype"],
                    shape=meta_l["shape"],
                ))
            elif s["site_type"] == "attn":
                raw_m = s["raw_module"]
                meta_a = attn_meta_cache[raw_m]
                if meta_a is None:
                    continue
                edits.append(Edit(
                    module=meta_a["module"],
                    u=u_attn_by_mod[raw_m].double(),
                    alpha=a_val,
                    idx=head_cols_by_mod[raw_m].long(),
                    base_sha256=meta_a["base_sha256"],
                    base_dtype=meta_a["base_dtype"],
                    shape=meta_a["shape"],
                ))
        return _make_patch(edits, name)

    mlp_fa = [l for l in mlp_ranked if l in fa_layers]
    patches: dict[str, MoIPatch] = {}
    if len(site_rankings) >= 2:
        patches["unified_top2"] = build_sites_patch(site_rankings[:2], name="compact-unified-top2")
    if len(mlp_ranked) >= 2:
        patches["blind_top2"] = build_patch(mlp_ranked[:2], use_wsi_resonance=False)
    if len(attn_stats) >= 1:
        patches["attn_top2"] = build_sites_patch(
            [{"site_type": "attn", **r} for r in attn_stats[:2]],
            name="compact-attn-top2",
        )
    if len(wsi_fa3_layers) >= 3:
        patches["wsi_top3"] = build_patch(wsi_fa3_layers)
        patches["wsi_fa3"] = patches["wsi_top3"]
    if len(wsi_pair_layers) >= 2:
        patches["wsi_pair"] = build_patch(wsi_pair_layers)
    if len(mlp_ranked) >= 3:
        patches["mlp_top3"] = build_patch(mlp_ranked[:3], use_wsi_resonance=False)
    if len(mlp_fa) >= 3:
        patches["mlp_fa3"] = build_patch(mlp_fa[:3], use_wsi_resonance=False)

    # Optional Causal Proposer (when traces carry `grad_summary`): union with passive candidates for forward judge
    causal_cts = [ct for ct, _, is_loop, _ in entries if ct.grad_summary and is_loop]
    if not causal_cts:
        causal_cts = [ct for ct, _, _, _ in entries if ct.grad_summary]
    causal_diag: dict[str, Any] | None = None
    if causal_cts:
        causal_diag = diagnose_causal_traces(
            causal_cts,
            split_ab=True,
            top_k_sites=6,
            top_n_neurons=max(int(top_k_neurons), 64),
        )
        meta_by_suffix: dict[str, dict[str, Any]] = {m["module"]: m for m in mod_meta.values()}
        for m_a in attn_meta_cache.values():
            if m_a is not None:
                meta_by_suffix[m_a["module"]] = m_a
        for s_row in causal_diag["sites"]:
            mk = s_row["module"]
            if mk not in meta_by_suffix:
                m_res = _resolve_attn_meta(
                    mk, model, wsi,
                    _mod_cache=mod_cache, _ck_names=ck_names, _ck_name_set=ck_name_set,
                )
                if m_res is not None:
                    meta_by_suffix[mk] = m_res

        valid_causal_sites = [s for s in causal_diag["sites"] if s["module"] in meta_by_suffix]

        def build_causal_patch(
            c_sites: list[dict[str, Any]],
            *,
            patch_alpha: float | None = None,
            use_top_neurons: bool = True,
            name: str = "compact-causal",
        ) -> MoIPatch:
            a_val = float(alpha if patch_alpha is None else patch_alpha)
            edits = []
            for s_row in c_sites:
                mk = s_row["module"]
                mm = meta_by_suffix[mk]
                t_neurons = causal_diag["top_neurons"].get(mk, [])
                cols = (
                    torch.tensor(t_neurons[: int(top_k_neurons)], dtype=torch.long)
                    if (use_top_neurons and t_neurons)
                    else None
                )
                edits.append(
                    Edit(
                        module=mk,
                        u=causal_diag["directions"][mk].double(),
                        alpha=a_val,
                        idx=cols,
                        base_sha256=mm["base_sha256"],
                        base_dtype=mm["base_dtype"],
                        shape=mm["shape"],
                    )
                )
            return _make_patch(edits, name)

        if len(valid_causal_sites) >= 1:
            patches["causal_top1"] = build_causal_patch(valid_causal_sites[:1], name="compact-causal-top1")
            patches["causal_top1_all"] = build_causal_patch(
                valid_causal_sites[:1], use_top_neurons=False, name="compact-causal-top1-all"
            )
        if len(valid_causal_sites) >= 2:
            patches["causal_top2"] = build_causal_patch(valid_causal_sites[:2], name="compact-causal-top2")
        causal_attn_sites = [s for s in valid_causal_sites if s["kind"] == "attn"]
        if causal_attn_sites:
            patches["causal_attn_top1"] = build_causal_patch(
                causal_attn_sites[:1], use_top_neurons=False, name="compact-causal-attn-top1"
            )

    # Build deduplicated union candidate grid (Top Passive + Top Causal) x candidate_alphas for forward screening
    union_templates: list[tuple[str, MoIPatch]] = []
    if len(mlp_ranked) >= 1:
        union_templates.append(("passive_blind_top1", build_patch(mlp_ranked[:1], use_wsi_resonance=False)))
    for k_name in (
        "unified_top2",
        "blind_top2",
        "attn_top2",
        "wsi_pair",
        "causal_top1",
        "causal_top1_all",
        "causal_top2",
        "causal_attn_top1",
    ):
        if k_name in patches:
            pref = k_name if k_name.startswith("causal_") else f"passive_{k_name}"
            union_templates.append((pref, patches[k_name]))

    dedup_templates: list[tuple[str, MoIPatch]] = []
    for t_name, t_patch in union_templates:
        if not t_patch.edits:
            continue
        is_dup = False
        for _, prev_p in dedup_templates:
            if len(prev_p.edits) != len(t_patch.edits):
                continue
            same_all = True
            for e1, e2 in zip(t_patch.edits, prev_p.edits):
                if e1.module != e2.module:
                    same_all = False
                    break
                idx1 = None if e1.idx is None else tuple(int(x) for x in e1.idx.tolist())
                idx2 = None if e2.idx is None else tuple(int(x) for x in e2.idx.tolist())
                if idx1 != idx2:
                    same_all = False
                    break
                cos_u = float(F.cosine_similarity(e1.u.float()[None], e2.u.float()[None]).item())
                if cos_u < 0.999:
                    same_all = False
                    break
            if same_all:
                is_dup = True
                break
        if not is_dup:
            dedup_templates.append((t_name, t_patch))

    candidate_grid: dict[str, MoIPatch] = {}
    for t_name, t_patch in dedup_templates:
        for a_val in candidate_alphas:
            a_f = float(a_val)
            cname = f"{t_name}_a{a_f:g}"
            scaled_edits = [
                Edit(
                    module=e.module,
                    u=e.u,
                    alpha=a_f,
                    idx=e.idx,
                    base_sha256=e.base_sha256,
                    base_dtype=e.base_dtype,
                    shape=e.shape,
                )
                for e in t_patch.edits
            ]
            candidate_grid[cname] = _make_patch(scaled_edits, cname)

    recommended = (
        patches.get("unified_top2")
        or patches.get("blind_top2")
        or patches.get("wsi_pair")
        or patches.get("mlp_top3")
        or build_patch(hybrid_ranked[: min(2, len(hybrid_ranked))], use_wsi_resonance=False)
    )

    return {
        "split_mode": split_mode,
        "fa_layers": fa_layers,
        "mlp_ranked": mlp_ranked,
        "hybrid_ranked": hybrid_ranked,
        "attn_stats": attn_stats,
        "site_rankings": site_rankings,
        "router_report": best_router_report,
        "wsi_origins": origins,
        "wsi_resonant_src": win_src,
        "wsi_resonant_dst": l_res_dst,
        "wsi_pair_layers": wsi_pair_layers,
        "wsi_fa3_layers": wsi_fa3_layers,
        "wsi_resonance_table": wsi_resonance_table,
        "layer_stats": layer_stats,
        "directions": u_mlp_by_layer,
        "top_neurons": topk_by_layer,
        "attn_directions": u_attn_by_mod,
        "attn_top_heads": top_heads_by_mod,
        "wsi_resonant_directions": wsi_dir_ov,
        "wsi_resonant_neurons": wsi_idx_ov,
        "causal_diagnosis": causal_diag,
        "candidate_grid": candidate_grid,
        "patch": recommended,
        "patches": patches,
        "build_patch": build_patch,
        "build_sites_patch": build_sites_patch,
    }


def discover_compact_dirs(
    wsi: Any,
    trace_dirs: list[str],
    *,
    loop_trace_dirs: list[str] | None = None,
    alpha: float = 6.0,
    top_k_neurons: int = 64,
    top_k_heads: int = 2,
    variant: str = "unified_top2",
    save_patch_path: str | None = None,
    save_candidate_grid_dir: str | Path | None = None,
    candidate_alphas: tuple[float, ...] = (4.0, 8.0, 12.0),
) -> dict[str, Any]:
    """Load saved `CompactTrace` directories, run blind multi-site discovery, optionally save a `.moipatch` (and/or the full union `candidate_grid`), and return a JSON summary."""
    traces = [CompactTrace.load(p) for p in trace_dirs]
    loop_traces = [CompactTrace.load(p) for p in loop_trace_dirs] if loop_trace_dirs else None
    res = wsi.discover_compact_circuit(
        traces,
        loop_traces=loop_traces,
        alpha=float(alpha),
        top_k_neurons=int(top_k_neurons),
        top_k_heads=int(top_k_heads),
        candidate_alphas=tuple(float(a) for a in candidate_alphas),
    )
    saved_patch = None
    if save_patch_path:
        chosen = res["patches"].get(variant) or res["patch"]
        saved_patch = str(chosen.save(save_patch_path))

    saved_candidate_grid: dict[str, str] | None = None
    if save_candidate_grid_dir:
        gdir = Path(save_candidate_grid_dir)
        gdir.mkdir(parents=True, exist_ok=True)
        saved_candidate_grid = {
            cname: str(cpatch.save(gdir / f"{cname}.moipatch"))
            for cname, cpatch in res["candidate_grid"].items()
        }

    cd = res.get("causal_diagnosis")
    causal_summary = None
    if cd is not None:
        causal_summary = {
            "summary": cd["summary"],
            "sites": cd["sites"],
            "causal_rank": cd["causal_rank"][:10],
            "passive_rank": cd["passive_rank"][:10],
            "top_neurons": {k: v[:16] for k, v in cd["top_neurons"].items()},
            "split_mode": cd["split_mode"],
            "n_traces": cd["n_traces"],
        }

    return {
        "split_mode": res["split_mode"],
        "fa_layers": res["fa_layers"],
        "mlp_ranked": res["mlp_ranked"],
        "hybrid_ranked": res["hybrid_ranked"],
        "attn_stats": res["attn_stats"],
        "site_rankings": res["site_rankings"],
        "router_report": res["router_report"],
        "wsi_origins": res["wsi_origins"],
        "wsi_resonant_src": res["wsi_resonant_src"],
        "wsi_resonant_dst": res["wsi_resonant_dst"],
        "wsi_pair_layers": res["wsi_pair_layers"],
        "wsi_fa3_layers": res["wsi_fa3_layers"],
        "wsi_resonance_table": res["wsi_resonance_table"],
        "layer_stats": res["layer_stats"],
        "causal_diagnosis": causal_summary,
        "available_patches": list(res["patches"]),
        "candidate_grid_names": list(res["candidate_grid"]),
        "saved_patch": saved_patch,
        "saved_candidate_grid": saved_candidate_grid,
    }


def _resolve_edit_modules(
    model: nn.Module,
    items: list[tuple[str, Any]],
) -> tuple[dict[str, nn.Module], dict[str, torch.Tensor]]:
    """Resolve unique `module_suffix` names referenced by candidate `MoIPatch`es and cache CPU float32 weights."""
    from .moipatch import module_suffix

    names: dict[str, list[nn.Module]] = {}
    for name, mod in model.named_modules():
        if _is_excluded_backbone_path(name) or _is_routed_expert_path(name) or getattr(mod, "weight", None) is None:
            continue
        try:
            names.setdefault(module_suffix(name), []).append(mod)
        except ValueError:
            pass

    needed = sorted({e.module for _, patch in items for e in patch.edits})
    resolved: dict[str, nn.Module] = {}
    w_cpu: dict[str, torch.Tensor] = {}
    for mk in needed:
        mods = names.get(mk, [])
        if len(mods) != 1:
            raise KeyError(f"{mk}: {len(mods)} matching modules in the model")
        mod = mods[0]
        resolved[mk] = mod
        w = mod.weight.detach().float().cpu()
        if not isinstance(mod, nn.Linear) and w.dim() == 2:
            w = w.t()
        w_cpu[mk] = w
    return resolved, w_cpu


def _to_seq_tensors(seqs: list[torch.Tensor | list[int]], dev: torch.device) -> list[torch.Tensor]:
    out = []
    for s in seqs:
        t = s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.long)
        out.append((t.unsqueeze(0) if t.dim() == 1 else t).to(dev))
    return out


def _resolve_target_span(T: int, span: tuple[int, int] | None) -> tuple[int, int]:
    if span is not None:
        s, e = span
        lo = max(0, min(T - 2, int(s) - 1))
        hi = max(lo + 1, min(T - 1, int(e) - 1))
        return lo, hi
    lo = max(0, (T // 2) - 1) if T >= 4 else T - 2
    return lo, T - 1


def _tail_logits(model: nn.Module, ids: torch.Tensor, lo: int) -> torch.Tensor:
    """Logits of positions lo..T-1 only: full [T, vocab] logits of a 16k-token sequence do not fit in memory."""
    T = int(ids.shape[1])
    try:
        out = model(input_ids=ids, use_cache=False, logits_to_keep=T - lo)
    except TypeError:  # model without logits_to_keep
        out = model(input_ids=ids, use_cache=False)
    return out.logits[0, -(T - lo):]


def gradient_screen_candidates(
    model: nn.Module,
    candidates: dict[str, Any] | list[tuple[str, Any]],
    *,
    healthy_seqs: list[torch.Tensor | list[int]],
    target_seqs: list[torch.Tensor | list[int]],
    target_spans: list[tuple[int, int]] | None = None,
    max_nll_delta: float = 0.05,
    healthy_cost: str = "linear",
) -> list[dict[str, Any]]:
    """Fast 1st-order gradient prefilter scoring arbitrarily many candidate `MoIPatch`es via 1 backward/seq.

    Runs 1 backward pass per sequence (target log-prob and healthy NLL) in the model's native dtype with
    parameter gradients disabled, caching only input activations and output gradients at the edited modules.
    Ideal as Stage 1 to narrow hundreds/thousands of layer/neuron sites down before `screen_patch_candidates`.
    When ``healthy_cost="fisher"``, measures healthy cost via empirical Fisher $\\sum_t (g_t \\cdot \\delta y_t)^2$.
    """
    if healthy_cost not in ("linear", "fisher"):
        raise ValueError(f"healthy_cost must be 'linear' or 'fisher', got {healthy_cost!r}")
    items = list(candidates.items()) if isinstance(candidates, dict) else list(candidates)
    if not items:
        return []

    dev = next(model.parameters()).device
    resolved, w_cpu = _resolve_edit_modules(model, items)
    h_ids = _to_seq_tensors(healthy_seqs, dev)
    t_ids = _to_seq_tensors(target_seqs, dev)

    n_h = sum(max(0, int(ids.shape[1]) - 1) for ids in h_ids)
    n_t = 0
    for i, ids in enumerate(t_ids):
        T = int(ids.shape[1])
        if T < 2:
            continue
        lo, hi = _resolve_target_span(T, target_spans[i] if (target_spans is not None and i < len(target_spans)) else None)
        n_t += hi - lo

    orig_req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in orig_req:
        p.requires_grad_(False)

    def _run_grad_pass(
        seqs: list[torch.Tensor],
        spans: list[tuple[int, int]] | None,
        kind: str,
    ) -> tuple[dict[str, list[tuple[torch.Tensor, torch.Tensor]]], list[int]]:
        pieces: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {mk: [] for mk in resolved}
        seq_lens: list[int] = []
        denom = max(1, n_t if kind == "target" else n_h)
        with torch.enable_grad():
            for i, ids in enumerate(seqs):
                T = int(ids.shape[1])
                if T < 2:
                    continue
                seq_lens.append(T)
                if kind == "target":
                    lo, hi = _resolve_target_span(T, spans[i] if (spans is not None and i < len(spans)) else None)
                else:
                    lo, hi = 0, T - 1

                h_passes: dict[str, list[torch.Tensor]] = {}
                g_passes: dict[str, dict[int, torch.Tensor]] = {}
                y_shapes: dict[str, int] = {}
                fwd_counts: dict[str, int] = {}
                recomp_cursor: dict[str, int] = {}
                state: dict[str, bool] = {"in_backward": False}
                handles = []
                for mk, mod in resolved.items():
                    def _hook(_m, inp, out, mk=mk):
                        t = out[0] if isinstance(out, tuple) else out
                        if not state["in_backward"]:
                            p_idx = fwd_counts.get(mk, 0)
                            fwd_counts[mk] = p_idx + 1
                            x_in = inp[0] if isinstance(inp, tuple) else inp
                            h_passes.setdefault(mk, []).append((x_in[0] if x_in.dim() == 3 else x_in).detach().float().cpu())
                            if isinstance(t, torch.Tensor):
                                y_shapes[mk] = int(t.shape[-1])
                            if torch.is_grad_enabled() and isinstance(t, torch.Tensor):
                                need_replace = False
                                if not t.requires_grad:
                                    t = t.detach().requires_grad_(True)
                                    need_replace = True

                                def _save_g(grad, mk=mk, p_idx=p_idx):
                                    g_passes.setdefault(mk, {})[p_idx] = (grad[0] if grad.dim() == 3 else grad).detach().float().cpu()

                                t.register_hook(_save_g)
                                if need_replace:
                                    return (t, *out[1:]) if isinstance(out, tuple) else t
                        else:
                            p_idx = recomp_cursor.get(mk, max(0, fwd_counts.get(mk, 1) - 1))
                            recomp_cursor[mk] = max(0, p_idx - 1)
                            if torch.is_grad_enabled() and isinstance(t, torch.Tensor):
                                need_replace = False
                                if not t.requires_grad:
                                    t = t.detach().requires_grad_(True)
                                    need_replace = True

                                def _save_g_rec(grad, mk=mk, p_idx=p_idx):
                                    g_passes.setdefault(mk, {}).setdefault(
                                        p_idx, (grad[0] if grad.dim() == 3 else grad).detach().float().cpu()
                                    )

                                t.register_hook(_save_g_rec)
                                if need_replace:
                                    return (t, *out[1:]) if isinstance(out, tuple) else t
                        return out

                    handles.append(mod.register_forward_hook(_hook))

                logits = lp = obj = None
                try:
                    logits = _tail_logits(model, ids, lo)[: hi - lo].float()
                    lp = F.log_softmax(logits, dim=-1).gather(1, ids[0, lo + 1 : hi + 1, None]).squeeze(1)
                    obj = (lp.sum() / denom) if kind == "target" else (-lp.sum() / denom)
                    if obj.requires_grad:
                        state["in_backward"] = True
                        obj.backward()
                    for mk, mod_r in resolved.items():
                        P = fwd_counts.get(mk, 0)
                        if P >= 1 and mk in h_passes:
                            h_cat = torch.cat(h_passes[mk], dim=0) if len(h_passes[mk]) > 1 else h_passes[mk][0]
                            D_m = y_shapes.get(mk, int(mod_r.weight.shape[0] if isinstance(mod_r, nn.Linear) else mod_r.weight.shape[1]))
                            gm = g_passes.get(mk, {})
                            g_list = [gm[p] if p in gm else torch.zeros((T, D_m), dtype=torch.float32) for p in range(P)]
                            g_cat = torch.cat(g_list, dim=0) if len(g_list) > 1 else g_list[0]
                            pieces[mk].append((g_cat, h_cat))
                finally:
                    for h in handles:
                        h.remove()
                    del h_passes, g_passes, logits, lp, obj
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
        return pieces, seq_lens

    try:
        tgt_pieces, _ = _run_grad_pass(t_ids, target_spans, "target")
        hlt_pieces, hlt_seq_lens = _run_grad_pass(h_ids, None, "healthy")
    finally:
        for p, req in orig_req:
            p.requires_grad_(req)

    @torch.no_grad()
    def _first_order(patch: Any, pieces: dict[str, list[tuple[torch.Tensor, torch.Tensor]]]) -> float:
        tot = 0.0
        for e in patch.edits:
            u = e.u.detach().float().cpu()
            uw = u @ w_cpu[e.module]
            if e.idx is not None:
                m = torch.zeros_like(uw)
                m[e.idx.long()] = 1.0
                uw = uw * m
            for g_t, h_t in pieces[e.module]:
                tot += float(-e.alpha * ((g_t @ u) * (h_t @ uw)).sum())
        return tot

    @torch.no_grad()
    def _fisher_order(
        patch: Any,
        pieces: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
        seq_lens: list[int],
    ) -> float:
        denom_h = float(max(1, n_h))
        any_mod = next(iter(pieces), None)
        n_seqs_h = len(pieces[any_mod]) if any_mod is not None else 0
        if n_seqs_h == 0:
            return 0.0
        tot_sq = 0.0
        for si in range(n_seqs_h):
            T_seq = seq_lens[si] if si < len(seq_lens) else 0
            delta_tok: torch.Tensor | None = None
            for e in patch.edits:
                u = e.u.detach().float().cpu()
                uw = u @ w_cpu[e.module]
                if e.idx is not None:
                    m = torch.zeros_like(uw)
                    m[e.idx.long()] = 1.0
                    uw = uw * m
                g_t, h_t = pieces[e.module][si]
                term = (-float(e.alpha)) * ((g_t @ u) * (h_t @ uw))
                if T_seq > 0 and term.shape[0] > T_seq and term.shape[0] % T_seq == 0:
                    term = term.reshape(-1, T_seq).sum(dim=0)
                delta_tok = term if delta_tok is None else (delta_tok + term)
            if delta_tok is not None:
                tot_sq += denom_h * float((delta_tok * delta_tok).sum().item())
        return tot_sq

    rows: list[dict[str, Any]] = []
    for name, patch in items:
        grad_supp = -_first_order(patch, tgt_pieces)
        if healthy_cost == "fisher":
            grad_dnll = _fisher_order(patch, hlt_pieces, hlt_seq_lens)
            grad_pareto = grad_supp / max(grad_dnll, 1e-8)
        else:
            grad_dnll = _first_order(patch, hlt_pieces)
            grad_pareto = grad_supp / max(abs(grad_dnll), 0.01)
        rows.append({
            "name": name,
            "grad_supp": round(grad_supp, 6),
            "grad_dnll": round(grad_dnll, 6),
            "grad_pareto": round(grad_pareto, 6),
            "passes_nll_guard": bool(grad_dnll <= max_nll_delta),
        })
    rows.sort(key=lambda r: (not r["passes_nll_guard"], -r["grad_pareto"], -r["grad_supp"], abs(r["grad_dnll"])))
    return rows


def diagnose_causal_traces(
    traces: list[CompactTrace],
    *,
    split_ab: bool = True,
    top_k_sites: int = 8,
    top_n_neurons: int = 64,
    healthy_penalty: float = 1.0,
    max_healthy_ratio: float = 0.5,
    betas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
    center: str = "none",
    healthy_cost: str = "linear",
    model_or_weights: nn.Module | dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Offline causal diagnosis across unified MLP & Attention sites from captured causal ``CompactTrace``s.

    Uses Half-A / Half-B held-out cross-validation on low-rank target & healthy curvature matrices
    ``C_target`` and ``C_healthy`` to pick the site, direction (``paired``, ``causal``, or Pareto
    Lagrangian ``C_target + beta * C_healthy``), and net causal neurons without touching any GPU.
    When ``center="healthy"``, subtracts the healthy mean activation $\\bar{Y}_{\\text{healthy}}$ from $Y$
    before forming target curvature $C_{\\text{target}}$. When ``healthy_cost="fisher"``, uses the per-token
    empirical Fisher $\\sum_t (g_t \\cdot \\delta y_t)^2$ from compressed per-token healthy gradients.
    """
    if center not in ("none", "healthy"):
        raise ValueError(f"center must be 'none' or 'healthy', got {center!r}")
    if healthy_cost not in ("linear", "fisher"):
        raise ValueError(f"healthy_cost must be 'linear' or 'fisher', got {healthy_cost!r}")
    if not traces:
        raise ValueError("diagnose_causal_traces requires at least one CompactTrace")
    if not any(ct.grad_summary for ct in traces):
        raise ValueError("Traces lack grad_summary; capture with capture_compact(..., capture_grad=True)")
    if healthy_cost == "fisher":
        for idx_t, ct in enumerate(traces):
            if ct.grad_mode != "compressed":
                raise ValueError(
                    f"healthy_cost='fisher' requires per-token gradients (grad_mode='compressed'), "
                    f"but trace #{idx_t} has grad_mode={ct.grad_mode!r}; summary mode is not sufficient"
                )

    all_mods = sorted({mk for ct in traces for mk in ct.grad_summary})
    half_a = traces[0::2] if (split_ab and len(traces) >= 2) else traces
    half_b = traces[1::2] if (split_ab and len(traces) >= 2) else traces

    w_cpu_diag: dict[str, torch.Tensor] = {}
    if model_or_weights is not None:
        if isinstance(model_or_weights, dict):
            w_cpu_diag = {k: v.detach().float().cpu() for k, v in model_or_weights.items()}
        else:
            from .moipatch import module_suffix
            for name, mod in model_or_weights.named_modules():
                if _is_excluded_backbone_path(name) or _is_routed_expert_path(name) or getattr(mod, "weight", None) is None:
                    continue
                try:
                    mk_s = module_suffix(name)
                except ValueError:
                    continue
                if mk_s in all_mods and mk_s not in w_cpu_diag:
                    w = mod.weight.detach().float().cpu()
                    if not isinstance(mod, nn.Linear) and w.dim() == 2:
                        w = w.t()
                    w_cpu_diag[mk_s] = w

    def _pool_half(sub_traces: list[CompactTrace], mk: str) -> dict[str, Any]:
        items = [
            (
                ct.grad_summary[mk],
                max(0, int(ct.meta.get("grad_n_target", max(1, ct.seq_len - 1)))),
                max(0, int(ct.meta.get("grad_n_healthy", 0))),
            )
            for ct in sub_traces
            if mk in ct.grad_summary
        ]
        if not items:
            raise KeyError(f"Module {mk!r} missing from split")
        gs0 = items[0][0]
        D = int(gs0["d_out"])
        U = int(gs0["d_in"])
        tot_t = sum(nt for _, nt, _ in items)
        tot_h = sum(nh for _, _, nh in items)

        C_t = torch.zeros((D, D), dtype=torch.float32)
        C_h = torch.zeros((D, D), dtype=torch.float32)
        F_h = torch.zeros((D, D), dtype=torch.float32)
        paired = torch.zeros(D, dtype=torch.float32)
        n_supp = torch.zeros(U, dtype=torch.float32)
        n_dnll = torch.zeros(U, dtype=torch.float32)

        for gs, nt, nh in items:
            wt = (nt / tot_t) if tot_t > 0 else (1.0 / len(items))
            if center == "healthy" and "c_target_centered_eigvecs" in gs:
                vt = gs["c_target_centered_eigvecs"].float()
                lt = gs["c_target_centered_eigvals"].float()
            else:
                vt = gs["c_target_eigvecs"].float()
                lt = gs["c_target_eigvals"].float()
            C_t += wt * (vt.t() @ (lt[:, None] * vt))
            paired += wt * gs["paired_diff"].float()
            supp_key = "neuron_supp_centered_attr" if (center == "healthy" and "neuron_supp_centered_attr" in gs) else "neuron_supp_attr"
            n_supp += wt * gs[supp_key].float()
            if tot_h > 0 and nh > 0:
                wh = nh / tot_h
                vh = gs["c_healthy_eigvecs"].float()
                lh = gs["c_healthy_eigvals"].float()
                C_h += wh * (vh.t() @ (lh[:, None] * vh))
                if "c_fisher_eigvecs" in gs:
                    vf = gs["c_fisher_eigvecs"].float()
                    lf = gs["c_fisher_eigvals"].float()
                    F_h += wh * (vf.t() @ (lf[:, None] * vf))
                dnll_key = "neuron_fisher_attr" if (healthy_cost == "fisher" and "neuron_fisher_attr" in gs) else "neuron_dnll_attr"
                n_dnll += wh * gs[dnll_key].float()

        C_t = 0.5 * (C_t + C_t.t())
        C_h = 0.5 * (C_h + C_h.t())
        F_h = 0.5 * (F_h + F_h.t())
        return {
            "kind": gs0["kind"],
            "layer": gs0["layer"],
            "C_t": C_t,
            "C_h": C_h,
            "F_h": F_h,
            "paired_diff": paired,
            "neuron_supp": n_supp,
            "neuron_dnll": n_dnll,
        }

    sites_all: list[dict[str, Any]] = []
    directions: dict[str, torch.Tensor] = {}
    top_neurons: dict[str, list[int]] = {}

    for mk in all_mods:
        pa = _pool_half(half_a, mk)
        pb = _pool_half(half_b, mk) if half_b is not half_a else pa
        pall = _pool_half(traces, mk)

        C_t_A, C_h_A, F_h_A, diff_A = pa["C_t"], pa["C_h"], pa["F_h"], pa["paired_diff"]
        C_t_B, C_h_B, F_h_B = pb["C_t"], pb["C_h"], pb["F_h"]

        # Orthonormal basis of the Half-A active subspace (all stored C_t/C_h eigenvectors)
        t_vec_key = "c_target_centered_eigvecs" if center == "healthy" else "c_target_eigvecs"
        h_vec_key = "c_fisher_eigvecs" if healthy_cost == "fisher" else "c_healthy_eigvecs"
        basis_rows = [
            ct.grad_summary[mk].get(key, ct.grad_summary[mk]["c_target_eigvecs" if "target" in key else "c_healthy_eigvecs"]).float()
            for ct in half_a
            if mk in ct.grad_summary
            for key in (t_vec_key, h_vec_key)
        ]
        B_mat = torch.cat(basis_rows, dim=0) if basis_rows else torch.empty((0, C_t_A.shape[0]))
        if B_mat.numel() > 0 and float(B_mat.abs().max().item()) > 1e-12:
            q_a, r_a = torch.linalg.qr(B_mat.t(), mode="reduced")
            active_cols = (r_a.abs().diag() > 1e-6).nonzero(as_tuple=True)[0]
            Q_A = q_a[:, active_cols] if active_cols.numel() > 0 else None
        else:
            Q_A = None

        cands: list[tuple[str, torch.Tensor]] = []
        if float(diff_A.norm().item()) > 1e-12:
            cands.append(("paired", (diff_A / diff_A.norm()).float()))
        if Q_A is not None:
            C_t_sub = 0.5 * (Q_A.t() @ (C_t_A + C_t_A.t()) @ Q_A)
            C_h_sub = 0.5 * (Q_A.t() @ (C_h_A + C_h_A.t()) @ Q_A)
            F_h_sub = 0.5 * (Q_A.t() @ (F_h_A + F_h_A.t()) @ Q_A)
            _, ev_c = torch.linalg.eigh(C_t_sub)
            cands.append(("causal", (Q_A @ ev_c[:, -1]).contiguous().float()))
            for beta in betas:
                pen_sub = (-float(beta) * F_h_sub) if healthy_cost == "fisher" else (float(beta) * C_h_sub)
                _, ev_p = torch.linalg.eigh(C_t_sub + pen_sub)
                cands.append((f"pareto_b{beta:g}", (Q_A @ ev_p[:, -1]).contiguous().float()))
        else:
            if float(C_t_A.abs().max().item()) <= 1e-12:
                cands.append(("causal", torch.eye(C_t_A.shape[0], 1, dtype=torch.float32)[:, 0].contiguous()))
            else:
                _, ev_c = torch.linalg.eigh(C_t_A)
                cands.append(("causal", ev_c[:, -1].contiguous().float()))

        # Pre-extract Half-B per-token healthy gradients (and inputs if available) when healthy_cost == "fisher"
        b_fisher_pieces: list[tuple[float, float, int, torch.Tensor, Any | None]] = []
        if healthy_cost == "fisher":
            tot_h_b = max(1, sum(max(0, int(ct.meta.get("grad_n_healthy", 0))) for ct in half_b))
            w_mod = w_cpu_diag.get(mk)
            wn_mod = w_mod.norm(dim=0).clamp_min(1e-12) if w_mod is not None else None
            for ct in half_b:
                nh_b = max(0, int(ct.meta.get("grad_n_healthy", 0)))
                if nh_b <= 0:
                    continue
                g_h_b = ct.dequant_grad(mk, "healthy")
                if g_h_b is None:
                    continue
                wh_b = nh_b / tot_h_b
                gs_b = ct.grad_summary.get(mk, {})
                h_in_b: Any | None = None
                if w_mod is not None:
                    kind_b = gs_b.get("kind")
                    layer_b = gs_b.get("layer", _layer_idx_from_name(mk))
                    if kind_b == "mlp" or (kind_b != "attn" and layer_b is not None and layer_b in ct.mlp_topk and ".mlp." in mk):
                        if layer_b is not None and int(layer_b) in ct.mlp_topk:
                            idx_m, val_m = ct.dequant_mlp_topk(int(layer_b))
                            x_m = val_m / wn_mod[idx_m]  # type: ignore[index]
                            h_in_b = ("mlp", (idx_m, x_m, w_mod))
                    else:
                        head_key_map = {_safe_module_suffix(hk): hk for hk in ct.head_norms}
                        hk = mk if mk in ct.head_norms else head_key_map.get(mk)
                        h_dense = ct.dequant_head_in(hk) if hk is not None else None
                        if h_dense is not None:
                            h_in_b = ("attn", (h_dense, w_mod))
                if h_in_b is None:
                    y_rms_sq = float(gs_b.get("y_healthy_rms_sq", max(float(gs_b.get("paired_diff_norm", 1.0)) ** 2 / max(1, int(gs_b.get("d_out", 1))), 1e-8)))
                    h_in_b = ("rms", max(y_rms_sq, 1e-8))
                b_fisher_pieces.append((wh_b, float(nh_b), int(ct.seq_len), g_h_b, h_in_b))

        def _eval_dnll_b(u_vec: torch.Tensor) -> float:
            if healthy_cost != "fisher":
                return float(-(u_vec @ C_h_B @ u_vec).item())
            if not b_fisher_pieces:
                return float((u_vec @ F_h_B @ u_vec).item())
            tot_f = 0.0
            for wh_b, nh_f, T_b, g_h_b, h_in_b in b_fisher_pieces:
                g_proj = g_h_b @ u_vec
                kind_in, pay = h_in_b
                if kind_in == "mlp":
                    idx_m, x_m, w_m = pay
                    uw_m = u_vec @ w_m
                    y_proj = (x_m * uw_m[idx_m]).sum(dim=1)
                    if g_proj.shape[0] != y_proj.shape[0] and T_b > 0 and g_proj.shape[0] % T_b == 0:
                        g_proj = g_proj.reshape(-1, T_b).sum(dim=0)
                    tok_term = g_proj * y_proj
                    if T_b > 0 and tok_term.shape[0] > T_b and tok_term.shape[0] % T_b == 0:
                        tok_term = tok_term.reshape(-1, T_b).sum(dim=0)
                    tot_f += wh_b * nh_f * float((tok_term * tok_term).sum().item())
                elif kind_in == "attn":
                    h_dense, w_m = pay
                    uw_m = u_vec @ w_m
                    y_proj = h_dense @ uw_m
                    if g_proj.shape[0] != y_proj.shape[0] and T_b > 0 and g_proj.shape[0] % T_b == 0:
                        g_proj = g_proj.reshape(-1, T_b).sum(dim=0)
                    tok_term = g_proj * y_proj
                    if T_b > 0 and tok_term.shape[0] > T_b and tok_term.shape[0] % T_b == 0:
                        tok_term = tok_term.reshape(-1, T_b).sum(dim=0)
                    tot_f += wh_b * nh_f * float((tok_term * tok_term).sum().item())
                else:
                    if T_b > 0 and g_proj.shape[0] > T_b and g_proj.shape[0] % T_b == 0:
                        g_proj = g_proj.reshape(-1, T_b).sum(dim=0)
                    tot_f += wh_b * nh_f * float(pay) * float((g_proj * g_proj).sum().item())
            return tot_f

        scored_cands: list[tuple[float, float, float, str, torch.Tensor]] = []
        for kind_lbl, u in cands:
            u = u / u.norm().clamp_min(1e-12)
            supp_b = float((u @ C_t_B @ u).item())
            dnll_b = _eval_dnll_b(u)
            pareto_b = (supp_b / max(dnll_b, 1e-8)) if healthy_cost == "fisher" else (supp_b / max(abs(dnll_b), 0.01))
            scored_cands.append((pareto_b, supp_b, -abs(dnll_b), kind_lbl, u))

        best_pareto, best_supp, neg_dnll_abs, best_kind, best_u = max(scored_cands, key=lambda x: (x[0], x[1], x[2]))
        best_dnll = _eval_dnll_b(best_u)
        if float(diff_A.norm().item()) > 1e-12:
            if float((best_u @ diff_A).item()) < 0.0:
                best_u = -best_u
        else:
            top_i = int(best_u.abs().argmax().item())
            if float(best_u[top_i].item()) < 0.0:
                best_u = -best_u

        supp_n = pall["neuron_supp"]
        dnll_n = pall["neuron_dnll"]
        net_n = supp_n - float(healthy_penalty) * torch.clamp(dnll_n, min=0.0)
        order = torch.argsort(net_n, descending=True).tolist()
        safe = [i for i in order if float(dnll_n[i].item()) <= float(max_healthy_ratio) * max(float(supp_n[i].item()), 1e-8)]
        picked = safe[:top_n_neurons] if safe else order[:top_n_neurons]

        directions[mk] = best_u
        top_neurons[mk] = picked
        sites_all.append({
            "module": mk,
            "kind": pa["kind"],
            "layer": pa["layer"],
            "best_dir_kind": best_kind,
            "pareto_score": round(best_pareto, 6),
            "target_supp": round(best_supp, 6),
            "healthy_dnll": round(best_dnll, 6),
            "passive_diff_norm": round(float(pall["paired_diff"].norm().item()), 6),
            "n_safe_neurons": len(safe),
            "top_neuron": picked[0] if picked else None,
        })

    if healthy_cost == "fisher" and sites_all:
        pos_dnlls = sorted(r["healthy_dnll"] for r in sites_all if r["healthy_dnll"] > 1e-12)
        fisher_floor = max(1e-8, 0.25 * pos_dnlls[len(pos_dnlls) // 2]) if pos_dnlls else 1e-8
        for r in sites_all:
            r["pareto_score"] = round(float(r["target_supp"]) / max(float(r["healthy_dnll"]), fisher_floor), 6)

    passive_rank = [r["module"] for r in sorted(sites_all, key=lambda r: (-r["passive_diff_norm"], r["module"]))]
    sites_all.sort(key=lambda r: (-r["pareto_score"], -r["target_supp"], abs(r["healthy_dnll"])))
    causal_rank = [r["module"] for r in sites_all]

    return {
        "sites": sites_all[:top_k_sites],
        "all_sites": sites_all,
        "causal_rank": causal_rank,
        "passive_rank": passive_rank,
        "directions": directions,
        "top_neurons": top_neurons,
        "split_mode": "half_ab" if (split_ab and len(traces) >= 2) else "full",
        "n_traces": len(traces),
        "center": center,
        "healthy_cost": healthy_cost,
        "summary": {
            "top_causal_site": sites_all[0]["module"],
            "top_causal_kind": sites_all[0]["kind"],
            "top_causal_layer": sites_all[0]["layer"],
            "top_causal_pareto": sites_all[0]["pareto_score"],
            "top_passive_site": passive_rank[0],
        },
    }


@torch.no_grad()
def screen_patches_from_traces(
    traces: list[CompactTrace],
    candidates: dict[str, Any] | list[tuple[str, Any]],
    model_or_weights: nn.Module | dict[str, torch.Tensor],
    *,
    max_nll_delta: float = 0.05,
    healthy_cost: str = "linear",
) -> list[dict[str, Any]]:
    """Purely offline first-order screening over ``CompactTrace(..., capture_grad=True, grad_mode='compressed')``.

    Requires zero model forward or backward passes; aggregates per-token causal sensitivities
    weighted identically to ``gradient_screen_candidates``.
    When ``healthy_cost="fisher"``, measures healthy cost via the per-token empirical Fisher
    $\\sum_t (g_t \\cdot \\delta y_t)^2$ across all edited modules.
    """
    if healthy_cost not in ("linear", "fisher"):
        raise ValueError(f"healthy_cost must be 'linear' or 'fisher', got {healthy_cost!r}")
    items = list(candidates.items()) if isinstance(candidates, dict) else list(candidates)
    if not items:
        return []
    if not traces:
        raise ValueError("screen_patches_from_traces requires at least one CompactTrace")
    for idx_t, ct in enumerate(traces):
        if ct.grad_mode != "compressed":
            msg = (
                f"healthy_cost='fisher' requires per-token gradients (grad_mode='compressed'), "
                f"but trace #{idx_t} has grad_mode={ct.grad_mode!r}; summary mode is not sufficient"
                if healthy_cost == "fisher"
                else f"Trace #{idx_t} has grad_mode={ct.grad_mode!r}; capture with capture_grad=True, grad_mode='compressed'"
            )
            raise ValueError(msg)

    if isinstance(model_or_weights, dict):
        w_cpu = {k: v.detach().float().cpu() for k, v in model_or_weights.items()}
    else:
        _, w_cpu = _resolve_edit_modules(model_or_weights, items)

    needed_mods = sorted({e.module for _, patch in items for e in patch.edits})
    n_t_list = [max(0, int(ct.meta.get("grad_n_target", max(1, ct.seq_len - 1)))) for ct in traces]
    n_h_list = [max(0, int(ct.meta.get("grad_n_healthy", 0))) for ct in traces]
    tot_t = max(1, sum(n_t_list))
    tot_h = max(1, sum(n_h_list))

    # Cache reconstructed gradients & inputs per (trace_idx, module) across all candidate patches
    cached_inputs: dict[tuple[int, str], tuple[str, Any]] = {}
    cached_gt: dict[tuple[int, str], torch.Tensor | None] = {}
    cached_gh: dict[tuple[int, str], torch.Tensor | None] = {}
    wn_by_mod = {mk: w_cpu[mk].norm(dim=0).clamp_min(1e-12) for mk in needed_mods}

    for ti, ct in enumerate(traces):
        head_key_map = {_safe_module_suffix(hk): hk for hk in ct.head_norms}
        for mk in needed_mods:
            cached_gt[(ti, mk)] = ct.dequant_grad(mk, "target")
            cached_gh[(ti, mk)] = ct.dequant_grad(mk, "healthy")
            gs = ct.grad_summary.get(mk, {})
            kind = gs.get("kind")
            layer = gs.get("layer", _layer_idx_from_name(mk))
            if kind == "mlp" or (kind != "attn" and layer is not None and layer in ct.mlp_topk and ".mlp." in mk):
                idx_m, val_m = ct.dequant_mlp_topk(int(layer))
                x_m = val_m / wn_by_mod[mk][idx_m]
                cached_inputs[(ti, mk)] = ("mlp", (idx_m, x_m))
            else:
                hk = mk if mk in ct.head_norms else head_key_map.get(mk)
                h_dense = ct.dequant_head_in(hk) if hk is not None else None
                if h_dense is None:
                    raise ValueError(f"Trace #{ti} missing pre-W_O head vectors for {mk!r}; capture with head_quant='int8'")
                cached_inputs[(ti, mk)] = ("attn", h_dense)

    rows: list[dict[str, Any]] = []
    for name, patch in items:
        grad_supp = 0.0
        grad_dnll = 0.0
        edit_specs = []
        for e in patch.edits:
            u = e.u.detach().float().cpu()
            uw = u @ w_cpu[e.module]
            if e.idx is not None:
                m = torch.zeros_like(uw)
                m[e.idx.long()] = 1.0
                uw = uw * m
            edit_specs.append((e, u, uw))

        for ti in range(len(traces)):
            T_seq = int(traces[ti].seq_len)
            delta_h_tok: torch.Tensor | None = None
            for e, u, uw in edit_specs:
                kind_in, h_obj = cached_inputs[(ti, e.module)]
                if kind_in == "mlp":
                    idx_m, x_m = h_obj
                    h_dot_uw = (x_m * uw[idx_m]).sum(dim=1)
                else:
                    h_dot_uw = h_obj @ uw

                g_t = cached_gt[(ti, e.module)]
                if g_t is not None and n_t_list[ti] > 0:
                    wt = n_t_list[ti] / tot_t
                    g_t_use = (
                        g_t.reshape(-1, T_seq, g_t.shape[-1]).sum(dim=0)
                        if (g_t.shape[0] != h_dot_uw.shape[0] and T_seq > 0 and g_t.shape[0] % T_seq == 0)
                        else g_t
                    )
                    grad_supp += wt * float(e.alpha) * float(((g_t_use @ u) * h_dot_uw).sum().item())

                g_h = cached_gh[(ti, e.module)]
                if g_h is not None and n_h_list[ti] > 0:
                    g_h_use = (
                        g_h.reshape(-1, T_seq, g_h.shape[-1]).sum(dim=0)
                        if (g_h.shape[0] != h_dot_uw.shape[0] and T_seq > 0 and g_h.shape[0] % T_seq == 0)
                        else g_h
                    )
                    term_h = (-float(e.alpha)) * ((g_h_use @ u) * h_dot_uw)
                    if healthy_cost == "fisher":
                        if T_seq > 0 and term_h.shape[0] > T_seq and term_h.shape[0] % T_seq == 0:
                            term_h = term_h.reshape(-1, T_seq).sum(dim=0)
                        delta_h_tok = term_h if delta_h_tok is None else (delta_h_tok + term_h)
                    else:
                        wh = n_h_list[ti] / tot_h
                        grad_dnll += wh * float(term_h.sum().item())

            if healthy_cost == "fisher" and delta_h_tok is not None and n_h_list[ti] > 0:
                wh = n_h_list[ti] / tot_h
                grad_dnll += wh * float(n_h_list[ti]) * float((delta_h_tok * delta_h_tok).sum().item())

        grad_pareto = (grad_supp / max(grad_dnll, 1e-8)) if healthy_cost == "fisher" else (grad_supp / max(abs(grad_dnll), 0.01))
        rows.append({
            "name": name,
            "grad_supp": round(grad_supp, 6),
            "grad_dnll": round(grad_dnll, 6),
            "grad_pareto": round(grad_pareto, 6),
            "passes_nll_guard": bool(grad_dnll <= max_nll_delta),
        })
    rows.sort(key=lambda r: (not r["passes_nll_guard"], -r["grad_pareto"], -r["grad_supp"], abs(r["grad_dnll"])))
    return rows


def screen_patch_candidates(
    model: nn.Module,
    candidates: dict[str, Any] | list[tuple[str, Any]],
    *,
    healthy_seqs: list[torch.Tensor | list[int]],
    target_seqs: list[torch.Tensor | list[int]],
    target_spans: list[tuple[int, int]] | None = None,
    max_nll_delta: float = 0.05,
    rank_by: str = "pareto",
    prefilter_top_k: int | None = None,
    traces: list[CompactTrace | str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Fast 1-forward-pass proxy screen ranking candidate `MoIPatch`es by failure suppression vs. healthy drift (`delta_nll` & `healthy_kl`).

    Evaluates each patch in teacher-forced mode (zero token generation) so bad `(k, n, alpha)` configs
    can be pruned in seconds before running expensive long-horizon generation.
    Computes both `delta_healthy_nll` and exact full-distribution `healthy_kl = KL(p_base || p_patched)`
    on healthy sequences (zero extra forward passes).
    Pass `prefilter_top_k` to use first-order gradient screening as Stage 1 when screening large
    candidate sets, narrowing to the top-K candidates before running forward evaluation. When `traces`
    (causal `CompactTrace` objects or directory paths captured with `capture_grad=True, grad_mode='compressed'`)
    is also supplied, Stage 1 runs 100% offline via `screen_patches_from_traces` with zero backward passes;
    otherwise Stage 1 falls back to 1-backward `gradient_screen_candidates`.
    `rank_by="pareto"` (default) ranks guard-passing patches by `pareto_score = target_suppression / max(|delta_nll|, 0.01)`;
    pass `rank_by="pareto_kl"` to rank by `pareto_kl = target_suppression / max(healthy_kl, 0.01)`,
    or `rank_by="suppression"` to rank by raw `target_suppression`.
    """
    if rank_by not in ("pareto", "pareto_kl", "suppression"):
        raise ValueError(f"rank_by must be 'pareto', 'pareto_kl', or 'suppression', got {rank_by!r}")
    items = list(candidates.items()) if isinstance(candidates, dict) else list(candidates)

    grad_by_name: dict[str, dict[str, Any]] = {}
    if prefilter_top_k is not None:
        prefilter_top_k = int(prefilter_top_k)
        if prefilter_top_k < 1:
            raise ValueError(f"prefilter_top_k must be >= 1, got {prefilter_top_k!r}")
        if items and prefilter_top_k < len(items):
            if traces:
                loaded_traces = [
                    CompactTrace.load(t) if isinstance(t, (str, Path)) else t
                    for t in traces
                ]
                grad_rows = screen_patches_from_traces(
                    loaded_traces,
                    items,
                    model,
                    max_nll_delta=max_nll_delta,
                )
            else:
                grad_rows = gradient_screen_candidates(
                    model,
                    items,
                    healthy_seqs=healthy_seqs,
                    target_seqs=target_seqs,
                    target_spans=target_spans,
                    max_nll_delta=max_nll_delta,
                )
            grad_by_name = {r["name"]: r for r in grad_rows}
            cand_by_name = {n: p for n, p in items}
            items = [(r["name"], cand_by_name[r["name"]]) for r in grad_rows[:prefilter_top_k]]

    dev = next(model.parameters()).device
    with torch.no_grad():
        h_ids = _to_seq_tensors(healthy_seqs, dev)
        t_ids = _to_seq_tensors(target_seqs, dev)
        base_h_logp: list[torch.Tensor | None] = []

        def _eval_current(*, capture_base_logp: bool = False) -> tuple[float, float, float]:
            nll_sum, kl_sum, n_h = 0.0, 0.0, 0
            for i, ids in enumerate(h_ids):
                if ids.shape[1] < 2:
                    if capture_base_logp:
                        base_h_logp.append(None)
                    continue
                logits = _tail_logits(model, ids, 0)[:-1].float()
                lp = F.log_softmax(logits, dim=-1)
                nll_sum += float(F.nll_loss(lp, ids[0, 1:], reduction="sum"))
                n_h += int(ids.shape[1] - 1)
                if capture_base_logp:
                    base_h_logp.append(lp.to(dtype=torch.bfloat16, device="cpu"))
                elif i < len(base_h_logp) and base_h_logp[i] is not None:
                    lp_b = base_h_logp[i].to(device=dev, dtype=torch.float32)
                    kl_tok = (lp_b.exp() * (lp_b - lp)).sum(dim=-1).clamp_min(0.0)
                    kl_sum += float(kl_tok.sum())
                    del lp_b, kl_tok
                del logits, lp
            logp_sum, n_t = 0.0, 0
            for i, ids in enumerate(t_ids):
                T = int(ids.shape[1])
                if T < 2:
                    continue
                lo, hi = _resolve_target_span(T, target_spans[i] if (target_spans is not None and i < len(target_spans)) else None)
                # position j predicts token j + 1; only positions lo..hi-1 are scored
                logits = _tail_logits(model, ids, lo)[: hi - lo].float()
                seg = F.log_softmax(logits, dim=-1).gather(1, ids[0, lo + 1: hi + 1, None]).squeeze(1)
                logp_sum += float(seg.sum())
                n_t += int(seg.numel())
            return (nll_sum / max(1, n_h)), (kl_sum / max(1, n_h)), (logp_sum / max(1, n_t))

        base_nll, _, base_tlogp = _eval_current(capture_base_logp=True)
        rows: list[dict[str, Any]] = []
        for name, patch in items:
            h = patch.attach(model)
            h.on()
            try:
                p_nll, p_kl, p_tlogp = _eval_current()
            finally:
                h.off()
            d_nll = p_nll - base_nll
            supp = base_tlogp - p_tlogp  # positive => failure tokens became less likely
            pareto = supp / max(abs(d_nll), 0.01)
            pareto_kl = supp / max(p_kl, 0.01)
            row: dict[str, Any] = {
                "name": name,
                "base_healthy_nll": round(base_nll, 5),
                "healthy_nll": round(p_nll, 5),
                "delta_healthy_nll": round(d_nll, 5),
                "healthy_kl": round(p_kl, 5),
                "base_target_logp": round(base_tlogp, 5),
                "target_logp": round(p_tlogp, 5),
                "target_suppression": round(supp, 5),
                "pareto_score": round(pareto, 4),
                "pareto_kl": round(pareto_kl, 4),
                "passes_nll_guard": bool(d_nll <= max_nll_delta),
            }
            if name in grad_by_name:
                gr = grad_by_name[name]
                row["grad_supp"] = gr["grad_supp"]
                row["grad_dnll"] = gr["grad_dnll"]
                row["grad_pareto"] = gr["grad_pareto"]
            rows.append(row)
        base_h_logp.clear()
        if rank_by == "pareto":
            rows.sort(key=lambda r: (not r["passes_nll_guard"], -r["pareto_score"], -r["target_suppression"], r["delta_healthy_nll"]))
        elif rank_by == "pareto_kl":
            rows.sort(key=lambda r: (not r["passes_nll_guard"], -r["pareto_kl"], -r["target_suppression"], r["healthy_kl"]))
        else:
            rows.sort(key=lambda r: (not r["passes_nll_guard"], -r["target_suppression"], r["delta_healthy_nll"]))
        return rows


# ----------------------------------------------------------------------------------------------- differential trace API
@torch.no_grad()
def diff_traces(
    model: nn.Module,
    ids_a: torch.Tensor | list[int],
    ids_b: torch.Tensor | list[int],
    *,
    wsi: Any | None = None,
    pos_a: int = -1,
    pos_b: int = -1,
    max_tokens: int | None = None,
    top_k: int = 10,
    compact: bool = True,
) -> dict[str, Any]:
    """X-ray comparing `ids_a` vs `ids_b` and linking shifts to WSI weights.
    Uses lossless-head `CompactTrace` + `diff_compact` by default (`compact=True`);
    pass `compact=False` for full uncompressed `trace_all`."""
    if compact:
        ct_a = capture_compact(model, ids_a, max_tokens=max_tokens)
        ct_b = capture_compact(model, ids_b, max_tokens=max_tokens)
        return diff_compact(ct_a, ct_b, wsi=wsi, model=model, pos_a=pos_a, pos_b=pos_b, top_k=top_k)

    max_tok = 64 if max_tokens is None else int(max_tokens)
    tr_a = trace_all(model, ids_a, max_tokens=max_tok)
    tr_b = trace_all(model, ids_b, max_tokens=max_tok)
    mods = dict(model.named_modules())

    def _at_pos(t: torch.Tensor, pos: int) -> torch.Tensor:
        return (t[0, pos] if t.dim() == 3 else t[pos]).float()

    # 1. Layer-by-layer residual curve across hidden_states
    hs_layers = sorted(
        int(k.split(".")[1]) for k in tr_a if k.startswith("hidden_states.") and k in tr_b
    )
    layer_curve = []
    peak_layer, peak_vec, peak_norm = None, None, -1.0
    for l in hs_layers:
        va = _at_pos(tr_a[f"hidden_states.{l}"], pos_a)
        vb = _at_pos(tr_b[f"hidden_states.{l}"], pos_b)
        dv = vb - va
        dn = float(dv.norm())
        rel = dn / max(float(va.norm()), 1e-12)
        cos = float(torch.nn.functional.cosine_similarity(va[None], vb[None]))
        layer_curve.append({"layer": l, "delta_norm": round(dn, 4), "rel_delta": round(rel, 5), "cosine": round(cos, 5)})
        if l > 0 and dn > peak_norm:
            peak_layer, peak_vec, peak_norm = l, dv, dn

    # 2. Causal MLP neuron shifts: delta_in[i] * ||W_down[:, i]|| (only call WSI explain on the final top_k)
    mlps = find_mlps(model)
    mlp_hits = []
    for g in mlps:
        key = f"{g.down}:in"
        if key not in tr_a or key not in tr_b:
            continue
        da = _at_pos(tr_a[key], pos_a)
        db = _at_pos(tr_b[key], pos_b)
        delta = db - da
        m = mods[g.down]
        w = m.weight.detach().float().cpu()
        wn = w.norm(dim=0) if isinstance(m, nn.Linear) else w.norm(dim=1)
        impact = delta * wn
        li = _layer_idx_from_name(g.down)
        for uid in impact.abs().topk(min(top_k, len(impact))).indices.tolist():
            mlp_hits.append({
                "layer": li, "module": g.down, "unit": int(uid),
                "act_a": round(float(da[uid]), 4), "act_b": round(float(db[uid]), 4),
                "resid_shift_norm": round(float(impact[uid].abs()), 4),
                "signed_shift": round(float(impact[uid]), 4),
            })
    mlp_hits.sort(key=lambda r: -r["resid_shift_norm"])
    top_mlp = mlp_hits[:top_k]
    _annotate_mlp_wsi(top_mlp, wsi)

    # 3. Exact Attention / Linear-Attn Head output vector shift ||(x_b - x_a)_h @ W_O_h^T||
    hidden = peak_vec.numel() if peak_vec is not None else 0
    attns = _find_attn_out_projs(model, hidden) if hidden else {}
    head_hits = []
    for aname, (am, nh) in attns.items():
        key = f"{aname}:in"
        if key not in tr_a or key not in tr_b:
            continue
        da = _at_pos(tr_a[key], pos_a)
        db = _at_pos(tr_b[key], pos_b)
        dx = (db - da).reshape(nh, -1)
        w = am.weight.detach().float().cpu()
        w_heads = w.t().reshape(nh, -1, w.shape[0]) if isinstance(am, nn.Linear) else w.reshape(nh, -1, w.shape[1])
        out_deltas = torch.einsum("hd,hdo->ho", dx, w_heads)
        norms = out_deltas.norm(dim=1)
        li = _layer_idx_from_name(aname)
        for hid in norms.topk(min(3, nh)).indices.tolist():
            row = {"layer": li, "module": aname, "head": int(hid), "out_delta_norm": round(float(norms[hid]), 4)}
            if li is not None and f"attentions.{li}" in tr_a and f"attentions.{li}" in tr_b:
                tb_att, ta_att = tr_b[f"attentions.{li}"], tr_a[f"attentions.{li}"]
                if tb_att.dim() == 4 and hid < tb_att.shape[1]:
                    att_a = ta_att[0, hid, pos_a].float()
                    att_b = tb_att[0, hid, pos_b].float()
                    row["attn_a_top_src_pos"] = int(att_a.argmax())
                    row["attn_a_top_weight"] = round(float(att_a.max()), 4)
                    row["attn_b_top_src_pos"] = int(att_b.argmax())
                    row["attn_b_top_weight"] = round(float(att_b.max()), 4)
                    shared = min(len(att_a), len(att_b))
                    if len(att_a) == len(att_b):
                        row["attn_max_shift_pos"] = int((att_b - att_a).abs().argmax())
                        row["attn_max_shift"] = round(float((att_b - att_a).abs().max()), 4)
                    elif shared > 0:
                        row["attn_prefix_l1_shift"] = round(float((att_b[:shared] - att_a[:shared]).abs().sum()), 4)
            head_hits.append(row)
    head_hits.sort(key=lambda r: -r["out_delta_norm"])

    # 4. Cache state deltas (KV at pos_a/pos_b even when T differs, + Gated DeltaNet recurrent/conv states)
    cache_shifts = []
    for k in sorted(set(tr_a) & set(tr_b)):
        if not k.startswith("cache."):
            continue
        item = _cache_pair_delta(k, tr_a[k], tr_b[k], pos_a, pos_b)
        if item is not None:
            cache_shifts.append(item)
    cache_shifts.sort(key=lambda r: -r["delta_norm"])

    return {
        "len_a": int(tr_a[f"hidden_states.{hs_layers[0]}"].shape[1]) if hs_layers else None,
        "len_b": int(tr_b[f"hidden_states.{hs_layers[0]}"].shape[1]) if hs_layers else None,
        "pos_a": pos_a, "pos_b": pos_b,
        "peak_residual_layer": peak_layer, "peak_residual_delta_norm": round(peak_norm, 4),
        "layer_curve": layer_curve,
        "top_mlp_contributors": top_mlp,
        "top_attention_heads": head_hits[:top_k],
        "top_cache_deltas": cache_shifts[:top_k],
        "peak_vector_readout": _explain_peak_vector(peak_layer, peak_vec, peak_norm, wsi),
    }


# ----------------------------------------------------------------------------------------------- internal direction attribution
@torch.no_grad()
def attribute_direction_writers(
    model: nn.Module,
    inputs: torch.Tensor | list[int] | list[torch.Tensor | list[int]],
    *,
    target_dir: torch.Tensor,
    target_layer: int,
    layer_convention: str = "block",
    pos: int | tuple[int, int] | None = -1,
    contrast_inputs: torch.Tensor | list[int] | list[torch.Tensor | list[int]] | None = None,
    top_k_neurons: int = 16,
    top_k_heads: int = 16,
    normalize_dir: bool = False,
    linearize_final_norm: bool = True,
    report_block_gaps: bool = False,
) -> dict[str, Any]:
    """Level-1 zero-backward direct linear attribution onto an internal direction ``target_dir``.

    Measures each upstream writer module (attention/conv output projections and MLP down
    projections), individual attention heads, and individual MLP neurons by exact linear
    projection onto ``target_dir`` at ``target_layer`` (optionally contrasting ``inputs``
    against ``contrast_inputs``). Residual additivity guarantees
    ``embed_contrib + sum(module contribs) == target_score``.

    For post-final-norm targets (``layer_convention='hidden_states'`` and ``target_layer == n_layers``)
    ``linearize_final_norm=True`` pulls ``target_dir`` back through the final LayerNorm/RMSNorm with
    the observed (frozen) per-token scale, so the decomposition stays exact
    (``embed + writers + final_norm_bias_contrib == target_score``). ``report_block_gaps=True``
    adds ``block_residual_gaps``: per block, ``(block_out - block_in) . v`` minus the sum of hooked
    writers inside that block (non-zero gaps flag post-sublayer norms / residual scaling).
    """
    seqs = _normalize_seq_list(inputs)
    cseqs = _normalize_seq_list(contrast_inputs) if contrast_inputs is not None else None

    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    mods = dict(model.named_modules())
    mlps = find_mlps(model)
    mlp_by_down: dict[str, int] = {}
    for g in mlps:
        li = _layer_idx_from_name(g.down)
        mlp_by_down[g.down] = li if li is not None else len(mlp_by_down)

    hidden = _infer_hidden_size(model, mlps)
    attns = _find_attn_out_projs(model, hidden)
    tdir_cpu = _validate_target_dir(target_dir, hidden, normalize=normalize_dir)
    stop_blk, use_final_hidden = _resolve_direction_layer(n_layers, int(target_layer), layer_convention)

    writers: dict[str, dict[str, Any]] = {}

    def _register_writer(full_name: str, mod: nn.Module, kind: str, layer: int | None, n_heads: int) -> None:
        if stop_blk < 0:
            return
        if layer is not None and layer > stop_blk:
            return
        mk = _safe_module_suffix(full_name)
        w_raw = mod.weight.detach()
        w_lin_f32 = (w_raw if isinstance(mod, nn.Linear) else (w_raw.t() if w_raw.dim() == 2 else w_raw)).float()
        w_dot = (tdir_cpu.to(device=w_raw.device, dtype=torch.float32) @ w_lin_f32).cpu().contiguous()
        del w_lin_f32
        bias = getattr(mod, "bias", None)
        bias_vec = bias.detach().float().cpu() if isinstance(bias, torch.Tensor) else None
        bias_contrib = float((bias_vec @ tdir_cpu).item()) if bias_vec is not None else 0.0
        writers[mk] = {
            "mod": mod,
            "kind": kind,
            "layer": layer,
            "n_heads": int(n_heads),
            "w_dot": w_dot,
            "bias_vec": bias_vec,
            "bias_contrib": bias_contrib,
        }

    for dname, li in mlp_by_down.items():
        _register_writer(dname, mods[dname], "mlp", int(li), 0)
    for aname, (am, nh) in attns.items():
        li = _layer_idx_from_name(aname)
        _register_writer(aname, am, _writer_kind(aname, False), int(li) if li is not None else None, nh)

    fn_mod = _find_final_norm_module(model) if use_final_hidden else None
    gap_layers = list(range(0, stop_blk + 1)) if (report_block_gaps and stop_blk >= 0) else []

    def _span2d(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        return (t[0] if t.dim() == 3 else t).detach().float().cpu()[lo:hi]

    def _run_group(group_seqs: list[torch.Tensor]) -> dict[str, Any]:
        sum_target = 0.0
        sum_embed = 0.0
        sum_norm_bias = 0.0
        sum_y_dot: dict[str, float] = {mk: 0.0 for mk in writers}
        sum_y_pass: dict[str, list[float]] = {}
        sum_bias_dot: dict[str, float] = {mk: 0.0 for mk in writers}
        sum_x_in: dict[str, torch.Tensor] = {}
        sum_neuron: dict[str, torch.Tensor] = {}
        blk_acc: dict[int, list[float]] = {}
        any_lin = False
        n_seq = len(group_seqs)

        for s_ids in group_seqs:
            t_len = int(s_ids.shape[1])
            lo, hi = _resolve_hidden_span(t_len, pos)
            cur: dict[str, Any] = {"embed": None, "target": None, "h_pre": None}
            cur_x: dict[str, list[torch.Tensor]] = {}
            cur_y: dict[str, list[torch.Tensor]] = {}
            blk_in: dict[int, list[torch.Tensor]] = {}
            blk_out: dict[int, list[torch.Tensor]] = {}
            hooks = []

            def _pre_blk0(_m, inp):
                h = inp[0] if isinstance(inp, tuple) and inp else inp
                if isinstance(h, torch.Tensor) and cur["embed"] is None:
                    cur["embed"] = _span2d(h, lo, hi)
                    if stop_blk == -1 and not use_final_hidden:
                        cur["target"] = cur["embed"]
                        raise _DirectionTargetCaptured()

            hooks.append(blocks[0].register_forward_pre_hook(_pre_blk0))

            for mk, meta_w in writers.items():
                mod = meta_w["mod"]

                def _whook(_m, inp, out, mk=mk):
                    x_in = inp[0] if isinstance(inp, tuple) and inp else inp
                    y_out = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(x_in, torch.Tensor):
                        cur_x.setdefault(mk, []).append(_span2d(x_in, lo, hi))
                    if isinstance(y_out, torch.Tensor):
                        cur_y.setdefault(mk, []).append(_span2d(y_out, lo, hi))

                hooks.append(mod.register_forward_hook(_whook))

            for gl in gap_layers:
                def _gpre(_m, inp, gl=gl):
                    h = inp[0] if isinstance(inp, tuple) and inp else inp
                    if isinstance(h, torch.Tensor):
                        blk_in.setdefault(gl, []).append(_span2d(h, lo, hi))

                def _gpost(_m, _inp, out, gl=gl):
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(t, torch.Tensor):
                        blk_out.setdefault(gl, []).append(_span2d(t, lo, hi))

                hooks.append(blocks[gl].register_forward_pre_hook(_gpre))
                hooks.append(blocks[gl].register_forward_hook(_gpost))

            if use_final_hidden:
                if fn_mod is not None:
                    def _fn_pre(_m, inp):
                        h = inp[0] if isinstance(inp, tuple) and inp else inp
                        if isinstance(h, torch.Tensor) and cur["h_pre"] is None:
                            cur["h_pre"] = _span2d(h, lo, hi)

                    hooks.append(fn_mod.register_forward_pre_hook(_fn_pre))
                else:
                    def _last_blk(_m, _inp, out):
                        t = out[0] if isinstance(out, (tuple, list)) else out
                        if isinstance(t, torch.Tensor) and cur["h_pre"] is None:
                            cur["h_pre"] = _span2d(t, lo, hi)

                    hooks.append(blocks[n_layers - 1].register_forward_hook(_last_blk))

            if not use_final_hidden and stop_blk >= 0:
                exp_passes = _expected_block_passes(model, stop_blk)
                seen_stop = [0]

                def _stop_hook(_m, _inp, out, exp_passes=exp_passes, seen_stop=seen_stop):
                    seen_stop[0] += 1
                    if seen_stop[0] < exp_passes:
                        return
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(t, torch.Tensor):
                        cur["target"] = _span2d(t, lo, hi)
                        raise _DirectionTargetCaptured()

                hooks.append(blocks[stop_blk].register_forward_hook(_stop_hook))

            out = None
            try:
                try:
                    out = _forward_for_hidden(model, s_ids.to(dev), need_hidden_states=use_final_hidden)
                except _DirectionTargetCaptured:
                    pass
            finally:
                for h in hooks:
                    h.remove()

            if cur["target"] is None:
                hs = getattr(out, "hidden_states", None) if out is not None else None
                if use_final_hidden:
                    if hs is None or len(hs) < n_layers + 1:
                        raise RuntimeError(
                            f"Model did not return hidden_states[{n_layers}] needed for target_layer={target_layer}"
                        )
                    t_hs = hs[-1]
                elif stop_blk == -1:
                    if hs is None or len(hs) == 0:
                        raise RuntimeError("Model did not return hidden_states[0] for target_layer=0")
                    t_hs = hs[0]
                else:
                    if hs is None or len(hs) <= stop_blk + 1:
                        raise RuntimeError(f"Failed to capture block {stop_blk} hidden state")
                    t_hs = hs[stop_blk + 1]
                cur["target"] = _span2d(t_hs, lo, hi)
            del out

            t_span: torch.Tensor = cur["target"]
            S = int(t_span.shape[0])
            target_val = float((t_span @ tdir_cpu).mean().item())

            if use_final_hidden and linearize_final_norm and cur["h_pre"] is not None:
                v_eff, nb_span, lin_ok = _linearize_final_norm_span(fn_mod, cur["h_pre"], t_span, tdir_cpu)
            else:
                v_eff, nb_span, lin_ok = tdir_cpu[None].expand(S, -1), torch.zeros(S, dtype=torch.float32), False
            any_lin = any_lin or lin_ok

            if cur["embed"] is not None:
                embed_val = float((cur["embed"] * v_eff).sum(dim=-1).mean().item())
            else:
                embed_val = target_val if stop_blk == -1 else 0.0
            norm_bias_val = float(nb_span.mean().item()) if lin_ok else 0.0

            sum_target += target_val
            sum_embed += embed_val
            sum_norm_bias += norm_bias_val
            y_dot_seq: dict[str, float] = {}
            for mk, meta_w in writers.items():
                ys = cur_y.get(mk, ())
                if ys:
                    p_vals = [float((yp * v_eff).sum(dim=-1).mean().item()) for yp in ys]
                    y_dot_seq[mk] = sum(p_vals)
                    sum_y_dot[mk] += y_dot_seq[mk]
                    if len(p_vals) > 1:
                        acc_p = sum_y_pass.setdefault(mk, [0.0] * len(p_vals))
                        for pi, pv in enumerate(p_vals):
                            if pi < len(acc_p):
                                acc_p[pi] += pv
                bv = meta_w["bias_vec"]
                if bv is not None:
                    n_p = max(1, len(ys))
                    sum_bias_dot[mk] += n_p * (float((v_eff @ bv).mean().item()) if lin_ok else meta_w["bias_contrib"])
                xs_list = cur_x.get(mk, ())
                if xs_list:
                    xm_sum = sum(xs.mean(dim=0) for xs in xs_list)
                    sum_x_in[mk] = sum_x_in[mk] + xm_sum if mk in sum_x_in else xm_sum.clone()
                    if lin_ok:
                        w_raw_m = meta_w["mod"].weight.detach()
                        w_lin_m = (w_raw_m if isinstance(meta_w["mod"], nn.Linear) else (w_raw_m.t() if w_raw_m.dim() == 2 else w_raw_m)).float()
                        vw = (v_eff.to(device=w_raw_m.device, dtype=torch.float32) @ w_lin_m).cpu()
                        nc = sum((xs * vw).mean(dim=0) for xs in xs_list)
                        del w_lin_m
                    else:
                        nc = xm_sum * meta_w["w_dot"]
                    sum_neuron[mk] = sum_neuron[mk] + nc if mk in sum_neuron else nc.clone()
            for gl in gap_layers:
                bis = blk_in.get(gl, ())
                bos = blk_out.get(gl, ())
                if bis and bos:
                    bd = sum(float(((bo - bi) * v_eff).sum(dim=-1).mean().item()) for bi, bo in zip(bis, bos))
                    ws = sum(v for mk, v in y_dot_seq.items() if writers[mk]["layer"] == gl)
                    acc = blk_acc.setdefault(gl, [0.0, 0.0, 0.0])
                    acc[0] += bd
                    acc[1] += ws
                    acc[2] += 1.0

        mod_contribs = {mk: sum_y_dot[mk] / n_seq for mk in writers}
        mod_pass_contribs = {mk: [v / n_seq for v in vals] for mk, vals in sum_y_pass.items()}
        bias_contribs = {mk: sum_bias_dot[mk] / n_seq for mk in writers}
        x_means = {mk: (sum_x_in[mk] / n_seq) for mk in sum_x_in}
        n_means = {mk: (sum_neuron[mk] / n_seq) for mk in sum_neuron}
        embed_mean = sum_embed / n_seq
        target_mean = sum_target / n_seq
        norm_bias_mean = sum_norm_bias / n_seq
        comp_sum = embed_mean + sum(mod_contribs.values()) + norm_bias_mean

        mlp_act: dict[str, torch.Tensor] = {}
        mlp_neurons: dict[str, torch.Tensor] = {}
        attn_heads: dict[str, torch.Tensor] = {}
        w_dot_eff: dict[str, torch.Tensor] = {}
        for mk, meta_w in writers.items():
            xm = x_means.get(mk)
            if xm is None:
                continue
            full_c = n_means[mk]
            w_dot_eff[mk] = torch.where(xm.abs() > 1e-12, full_c / xm.where(xm.abs() > 1e-12, torch.ones_like(xm)), meta_w["w_dot"])
            if meta_w["kind"] == "mlp":
                mlp_act[mk] = xm
                mlp_neurons[mk] = full_c
            elif meta_w["kind"] == "attn":
                nh = meta_w["n_heads"]
                if nh >= 1 and full_c.numel() % nh == 0:
                    attn_heads[mk] = full_c.reshape(nh, full_c.numel() // nh).sum(dim=1)

        block_gaps = []
        for gl in sorted(blk_acc):
            bd, ws, cnt = blk_acc[gl]
            cnt = max(cnt, 1.0)
            block_gaps.append({
                "layer": int(gl),
                "block_delta": round(bd / cnt, 7),
                "writers_sum": round(ws / cnt, 7),
                "gap": round((bd - ws) / cnt, 7),
            })

        return {
            "target_score": target_mean,
            "embed_contrib": embed_mean,
            "norm_bias_contrib": norm_bias_mean,
            "sum_components": comp_sum,
            "residual_gap": target_mean - comp_sum,
            "mod_contribs": mod_contribs,
            "mod_pass_contribs": mod_pass_contribs,
            "bias_contribs": bias_contribs,
            "mlp_act": mlp_act,
            "mlp_neurons": mlp_neurons,
            "w_dot_eff": w_dot_eff,
            "attn_heads": attn_heads,
            "block_gaps": block_gaps,
            "lin_ok": any_lin,
        }

    was_training = bool(model.training)
    model.eval()
    try:
        g_pos = _run_group(seqs)
        g_neg = _run_group(cseqs) if cseqs is not None else None
    finally:
        if was_training:
            model.train()

    contrast = g_neg is not None
    mod_rows: list[dict[str, Any]] = []
    for mk, meta_w in writers.items():
        c_pos = g_pos["mod_contribs"][mk]
        row: dict[str, Any] = {
            "module": mk,
            "kind": meta_w["kind"],
            "layer": meta_w["layer"],
            "contrib": round(c_pos, 7),
            "abs_contrib": round(abs(c_pos), 7),
            "bias_contrib": round(g_pos["bias_contribs"][mk], 7),
        }
        if contrast:
            c_neg = g_neg["mod_contribs"][mk]  # type: ignore[index]
            d_c = c_pos - c_neg
            row["contrast_contrib"] = round(c_neg, 7)
            row["delta_contrib"] = round(d_c, 7)
            row["abs_delta_contrib"] = round(abs(d_c), 7)
        mod_rows.append(row)
    if contrast:
        mod_rows.sort(key=lambda r: (-r["abs_delta_contrib"], r["module"]))
    else:
        mod_rows.sort(key=lambda r: (-r["abs_contrib"], r["module"]))

    head_rows: list[dict[str, Any]] = []
    for mk, meta_w in writers.items():
        if meta_w["kind"] != "attn" or mk not in g_pos["attn_heads"]:
            continue
        hp = g_pos["attn_heads"][mk]
        hn = g_neg["attn_heads"].get(mk) if contrast else None  # type: ignore[index]
        nh = int(hp.numel())
        for hid in range(nh):
            vp = float(hp[hid].item())
            hrow: dict[str, Any] = {
                "module": mk,
                "layer": meta_w["layer"],
                "head": int(hid),
                "n_heads": nh,
                "contrib": round(vp, 7),
            }
            if contrast and hn is not None:
                vn = float(hn[hid].item())
                hrow["contrast_contrib"] = round(vn, 7)
                hrow["delta_contrib"] = round(vp - vn, 7)
            head_rows.append(hrow)
    if contrast:
        head_rows.sort(key=lambda r: (-abs(r["delta_contrib"]), r["module"], r["head"]))
    else:
        head_rows.sort(key=lambda r: (-abs(r["contrib"]), r["module"], r["head"]))

    neuron_rows: list[dict[str, Any]] = []
    for mk, meta_w in writers.items():
        if meta_w["kind"] != "mlp" or mk not in g_pos["mlp_neurons"]:
            continue
        np_vec = g_pos["mlp_neurons"][mk]
        ap_vec = g_pos["mlp_act"][mk]
        wd_vec = g_pos["w_dot_eff"].get(mk, meta_w["w_dot"])
        nn_vec = g_neg["mlp_neurons"].get(mk) if contrast else None  # type: ignore[index]
        an_vec = g_neg["mlp_act"].get(mk) if contrast else None  # type: ignore[index]
        score_vec = (np_vec - nn_vec).abs() if (contrast and nn_vec is not None) else np_vec.abs()
        k_mod = min(max(0, int(top_k_neurons)), int(score_vec.numel()))
        if k_mod == 0:
            continue
        top_ids = torch.argsort(score_vec, descending=True)[:k_mod].tolist()
        for uid in top_ids:
            vp = float(np_vec[uid].item())
            nrow: dict[str, Any] = {
                "module": mk,
                "layer": meta_w["layer"],
                "neuron": int(uid),
                "contrib": round(vp, 7),
                "activation": round(float(ap_vec[uid].item()), 7),
                "weight_dot": round(float(wd_vec[uid].item()), 7),
            }
            if contrast and nn_vec is not None and an_vec is not None:
                vn = float(nn_vec[uid].item())
                nrow["contrast_contrib"] = round(vn, 7)
                nrow["delta_contrib"] = round(vp - vn, 7)
                nrow["contrast_activation"] = round(float(an_vec[uid].item()), 7)
            neuron_rows.append(nrow)
    if contrast:
        neuron_rows.sort(key=lambda r: (-abs(r["delta_contrib"]), r["module"], r["neuron"]))
    else:
        neuron_rows.sort(key=lambda r: (-abs(r["contrib"]), r["module"], r["neuron"]))

    out_dict: dict[str, Any] = {
        "mode": "contrast" if contrast else "single",
        "target_layer": int(target_layer),
        "layer_convention": layer_convention,
        "stop_block": int(stop_blk),
        "use_final_hidden": bool(use_final_hidden),
        "n_inputs": len(seqs),
        "n_contrast": len(cseqs) if cseqs is not None else 0,
        "target_score": round(g_pos["target_score"], 7),
        "embed_contrib": round(g_pos["embed_contrib"], 7),
        "sum_components": round(g_pos["sum_components"], 7),
        "residual_gap": round(g_pos["residual_gap"], 7),
        "final_norm_linearized": bool(g_pos["lin_ok"] or (g_neg is not None and g_neg["lin_ok"])),
        "final_norm_bias_contrib": round(g_pos["norm_bias_contrib"], 7),
        "modules": mod_rows,
        "module_contribs": {r["module"]: r["contrib"] for r in mod_rows},
        "module_pass_contribs": {
            mk: [round(v, 7) for v in vals] for mk, vals in g_pos["mod_pass_contribs"].items()
        },
        "top_heads": head_rows[: max(0, int(top_k_heads))],
        "top_neurons": neuron_rows[: max(0, int(top_k_neurons))],
    }
    if contrast and g_neg is not None:
        d_target = g_pos["target_score"] - g_neg["target_score"]
        d_embed = g_pos["embed_contrib"] - g_neg["embed_contrib"]
        d_sum = g_pos["sum_components"] - g_neg["sum_components"]
        out_dict["contrast_target_score"] = round(g_neg["target_score"], 7)
        out_dict["delta_target_score"] = round(d_target, 7)
        out_dict["contrast_embed_contrib"] = round(g_neg["embed_contrib"], 7)
        out_dict["delta_embed_contrib"] = round(d_embed, 7)
        out_dict["contrast_sum_components"] = round(g_neg["sum_components"], 7)
        out_dict["delta_sum_components"] = round(d_sum, 7)
        out_dict["contrast_residual_gap"] = round(g_neg["residual_gap"], 7)
        out_dict["delta_residual_gap"] = round(d_target - d_sum, 7)
        out_dict["module_delta_contribs"] = {r["module"]: r["delta_contrib"] for r in mod_rows}
        out_dict["delta_module_pass_contribs"] = {
            mk: [round(vp - vn, 7) for vp, vn in zip(g_pos["mod_pass_contribs"][mk], g_neg["mod_pass_contribs"][mk])]
            for mk in g_pos["mod_pass_contribs"]
            if mk in g_neg["mod_pass_contribs"]
        }
        out_dict["contrast_final_norm_bias_contrib"] = round(g_neg["norm_bias_contrib"], 7)
        out_dict["delta_final_norm_bias_contrib"] = round(g_pos["norm_bias_contrib"] - g_neg["norm_bias_contrib"], 7)
    if report_block_gaps:
        out_dict["block_residual_gaps"] = g_pos["block_gaps"]
        if contrast and g_neg is not None:
            out_dict["contrast_block_residual_gaps"] = g_neg["block_gaps"]
    return out_dict


def summarize_direction_traces(
    traces: list[CompactTrace],
    *,
    contrast_traces: list[CompactTrace] | None = None,
    top_k_modules: int = 16,
    top_k_heads: int = 16,
    top_k_neurons: int = 16,
) -> dict[str, Any]:
    """Level-2 aggregator over direction-traced ``CompactTrace`` objects (single or contrasted).

    Computes group means across full neuron/head attribution vectors (avoiding top-k truncation
    bias across prompts) and ranks modules, attention heads, and MLP neurons by causal (and
    when contrasted, differential ``pos - contrast``) contribution to ``target_dir``.
    """
    if not traces:
        raise ValueError("summarize_direction_traces requires at least one CompactTrace")

    def _has_grad(ct: CompactTrace) -> bool:
        return bool(ct.grad_summary) or ct.meta.get("grad_objective") is not None

    if not all(_has_grad(ct) for ct in traces):
        raise ValueError("All traces must be captured with capture_causal_trace or capture_grad=True")
    if contrast_traces is not None:
        if len(contrast_traces) == 0:
            raise ValueError("contrast_traces cannot be empty when provided")
        if not all(_has_grad(ct) for ct in contrast_traces):
            raise ValueError("All contrast_traces must be captured with capture_causal_trace or capture_grad=True")

    def _agg(sub_traces: list[CompactTrace]) -> dict[str, Any]:
        scores = [float(ct.meta["target_dir_score"]) for ct in sub_traces if "target_dir_score" in ct.meta]
        mean_score = (sum(scores) / len(scores)) if scores else None
        all_mods = sorted({mk for ct in sub_traces for mk in ct.grad_summary})
        mod_info: dict[str, dict[str, Any]] = {}
        for mk in all_mods:
            items = [ct.grad_summary[mk] for ct in sub_traces if mk in ct.grad_summary]
            gs0 = items[0]
            causal_vals = [
                float(gs["causal_attr"]) if "causal_attr" in gs else float(gs["neuron_supp_attr"].float().sum().item())
                for gs in items
            ]
            direct_vals = [float(gs["direct_attr"]) for gs in items if "direct_attr" in gs]
            indirect_vals = [float(gs["indirect_attr"]) for gs in items if "indirect_attr" in gs]
            n_mean = torch.stack([gs["neuron_supp_attr"].detach().float().cpu() for gs in items], dim=0).mean(dim=0)
            h_mean = (
                torch.stack([gs["head_causal_attr"].detach().float().cpu() for gs in items], dim=0).mean(dim=0)
                if all("head_causal_attr" in gs for gs in items)
                else None
            )
            mod_info[mk] = {
                "kind": gs0.get("kind"),
                "layer": gs0.get("layer"),
                "causal_attr": sum(causal_vals) / len(causal_vals),
                "direct_attr": (sum(direct_vals) / len(direct_vals)) if direct_vals else None,
                "indirect_attr": (sum(indirect_vals) / len(indirect_vals)) if indirect_vals else None,
                "neuron_mean": n_mean,
                "head_mean": h_mean,
            }
        return {"mean_score": mean_score, "mods": mod_info}

    g_pos = _agg(traces)
    g_neg = _agg(contrast_traces) if contrast_traces is not None else None
    contrast = g_neg is not None

    all_mods = sorted(set(g_pos["mods"]) | (set(g_neg["mods"]) if g_neg is not None else set()))
    mod_rows: list[dict[str, Any]] = []
    for mk in all_mods:
        mp = g_pos["mods"].get(mk)
        mn = g_neg["mods"].get(mk) if g_neg is not None else None
        ref = mp or mn
        assert ref is not None
        cp = float(mp["causal_attr"]) if mp is not None else 0.0
        row: dict[str, Any] = {
            "module": mk,
            "kind": ref["kind"],
            "layer": ref["layer"],
            "causal_attr": round(cp, 6),
            "abs_causal_attr": round(abs(cp), 6),
        }
        if mp is not None and mp["direct_attr"] is not None:
            row["direct_attr"] = round(float(mp["direct_attr"]), 6)
        if mp is not None and mp["indirect_attr"] is not None:
            row["indirect_attr"] = round(float(mp["indirect_attr"]), 6)
        if contrast:
            cn = float(mn["causal_attr"]) if mn is not None else 0.0
            dc = cp - cn
            row["contrast_causal_attr"] = round(cn, 6)
            row["delta_causal_attr"] = round(dc, 6)
            row["abs_delta_causal_attr"] = round(abs(dc), 6)
            if (mp and mp["direct_attr"] is not None) or (mn and mn["direct_attr"] is not None):
                dp = float(mp["direct_attr"]) if (mp and mp["direct_attr"] is not None) else 0.0
                dn = float(mn["direct_attr"]) if (mn and mn["direct_attr"] is not None) else 0.0
                row["contrast_direct_attr"] = round(dn, 6)
                row["delta_direct_attr"] = round(dp - dn, 6)
            if (mp and mp["indirect_attr"] is not None) or (mn and mn["indirect_attr"] is not None):
                ip = float(mp["indirect_attr"]) if (mp and mp["indirect_attr"] is not None) else 0.0
                ind_n = float(mn["indirect_attr"]) if (mn and mn["indirect_attr"] is not None) else 0.0
                row["contrast_indirect_attr"] = round(ind_n, 6)
                row["delta_indirect_attr"] = round(ip - ind_n, 6)
        mod_rows.append(row)

    if contrast:
        mod_rows.sort(key=lambda r: (-r["abs_delta_causal_attr"], r["module"]))
    else:
        mod_rows.sort(key=lambda r: (-r["abs_causal_attr"], r["module"]))

    head_rows: list[dict[str, Any]] = []
    for mk in all_mods:
        mp = g_pos["mods"].get(mk)
        mn = g_neg["mods"].get(mk) if g_neg is not None else None
        ref = mp or mn
        if ref is None or ref["kind"] != "attn":
            continue
        hp = mp["head_mean"] if (mp is not None and mp["head_mean"] is not None) else None
        hn = mn["head_mean"] if (mn is not None and mn["head_mean"] is not None) else None
        if hp is None and hn is None:
            continue
        nh = int((hp if hp is not None else hn).numel())  # type: ignore[union-attr]
        for hid in range(nh):
            vp = float(hp[hid].item()) if hp is not None else 0.0
            hrow: dict[str, Any] = {
                "module": mk,
                "layer": ref["layer"],
                "head": int(hid),
                "n_heads": nh,
                "causal_attr": round(vp, 6),
            }
            if contrast:
                vn = float(hn[hid].item()) if hn is not None else 0.0
                hrow["contrast_causal_attr"] = round(vn, 6)
                hrow["delta_causal_attr"] = round(vp - vn, 6)
            head_rows.append(hrow)
    if contrast:
        head_rows.sort(key=lambda r: (-abs(r["delta_causal_attr"]), r["module"], r["head"]))
    else:
        head_rows.sort(key=lambda r: (-abs(r["causal_attr"]), r["module"], r["head"]))

    neuron_rows: list[dict[str, Any]] = []
    for mk in all_mods:
        mp = g_pos["mods"].get(mk)
        mn = g_neg["mods"].get(mk) if g_neg is not None else None
        ref = mp or mn
        if ref is None or ref["kind"] != "mlp":
            continue
        np_vec = mp["neuron_mean"] if mp is not None else None
        nn_vec = mn["neuron_mean"] if mn is not None else None
        if np_vec is None and nn_vec is None:
            continue
        if np_vec is None:
            np_vec = torch.zeros_like(nn_vec)  # type: ignore[arg-type]
        if contrast and nn_vec is None:
            nn_vec = torch.zeros_like(np_vec)
        score_vec = (np_vec - nn_vec).abs() if (contrast and nn_vec is not None) else np_vec.abs()
        k_mod = min(max(0, int(top_k_neurons)), int(score_vec.numel()))
        if k_mod == 0:
            continue
        top_ids = torch.argsort(score_vec, descending=True)[:k_mod].tolist()
        for uid in top_ids:
            vp = float(np_vec[uid].item())
            nrow: dict[str, Any] = {
                "module": mk,
                "layer": ref["layer"],
                "neuron": int(uid),
                "causal_attr": round(vp, 6),
            }
            if contrast and nn_vec is not None:
                vn = float(nn_vec[uid].item())
                nrow["contrast_causal_attr"] = round(vn, 6)
                nrow["delta_causal_attr"] = round(vp - vn, 6)
            neuron_rows.append(nrow)
    if contrast:
        neuron_rows.sort(key=lambda r: (-abs(r["delta_causal_attr"]), r["module"], r["neuron"]))
    else:
        neuron_rows.sort(key=lambda r: (-abs(r["causal_attr"]), r["module"], r["neuron"]))

    meta0 = traces[0].meta
    out_summary: dict[str, Any] = {
        "mode": "contrast" if contrast else "single",
        "n_traces": len(traces),
        "n_contrast": len(contrast_traces) if contrast_traces is not None else 0,
        "target_layer": meta0.get("target_layer"),
        "target_layer_convention": meta0.get("target_layer_convention"),
        "mean_target_score": round(g_pos["mean_score"], 6) if g_pos["mean_score"] is not None else None,
        "modules": mod_rows[: max(0, int(top_k_modules))],
        "all_modules": mod_rows,
        "top_heads": head_rows[: max(0, int(top_k_heads))],
        "top_neurons": neuron_rows[: max(0, int(top_k_neurons))],
    }
    if contrast and g_neg is not None:
        out_summary["contrast_target_score"] = round(g_neg["mean_score"], 6) if g_neg["mean_score"] is not None else None
        if g_pos["mean_score"] is not None and g_neg["mean_score"] is not None:
            out_summary["delta_target_score"] = round(g_pos["mean_score"] - g_neg["mean_score"], 6)
        else:
            out_summary["delta_target_score"] = None
    return out_summary


# ----------------------------------------------------------------------------------------------- multi-hop direction graph
_MLP_INPUT_NORM_ATTRS = ("pre_feedforward_layernorm", "ffn_norm", "ln_2", "post_attention_layernorm")


def trace_direction_graph(
    model: nn.Module,
    inputs: torch.Tensor | list[int] | list[torch.Tensor | list[int]],
    *,
    target_dir: torch.Tensor,
    target_layer: int,
    layer_convention: str = "block",
    pos: int | tuple[int, int] | None = -1,
    depth: int = 2,
    top_k_nodes: int = 3,
    min_layer: int = 0,
    min_edge_weight: float = 1e-6,
    normalize_dir: bool = False,
) -> dict[str, Any]:
    """Multi-hop circuit graph (nodes + directed edges) behind an internal direction ``target_dir``.

    Hop 0 is the root ``dir:L{target_layer}``. Each expanded node gets the exact gradient of its own
    scalar readout (MLP neuron ``x[:, j] * (g . W[:, j])``, attention head ``x[:, head] . (g @ W_O_head)``,
    or a whole conv/out-proj module ``y . g``) with respect to the residual stream entering its sublayer.
    Upstream writers are scored by their *direct* (residual-path) effect on that readout
    (``g_sub_in . y_writer``, split per neuron / per head) and the top ``top_k_nodes`` become parents.
    The readout differentiates through norms, SwiGLU gating and attention exactly (no approximations).

    Returns ``{root, nodes, edges, top_paths, per_input_top_tokens, ...}`` (JSON-serializable).
    """
    if int(depth) < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if int(top_k_nodes) < 1:
        raise ValueError(f"top_k_nodes must be >= 1, got {top_k_nodes}")
    seqs = _normalize_seq_list(inputs)
    dev = next(model.parameters()).device
    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    mods = dict(model.named_modules())
    mlps = find_mlps(model)
    hidden = _infer_hidden_size(model, mlps)
    tdir_cpu = _validate_target_dir(target_dir, hidden, normalize=normalize_dir)
    stop_blk, use_final_hidden = _resolve_direction_layer(n_layers, int(target_layer), layer_convention)
    root_id = f"dir:L{int(target_layer)}"

    writers: dict[str, dict[str, Any]] = {}

    def _reg(full_name: str, mod: nn.Module, kind: str, layer: int | None, nh: int) -> None:
        if stop_blk < 0 or (layer is not None and layer > stop_blk):
            return
        w_raw = mod.weight
        d_in_w = int(w_raw.shape[1] if isinstance(mod, nn.Linear) else w_raw.shape[0])
        writers[_safe_module_suffix(full_name)] = {"mod": mod, "kind": kind, "layer": layer, "n_heads": int(nh), "d_in": d_in_w}

    for g in mlps:
        li = _layer_idx_from_name(g.down)
        _reg(g.down, mods[g.down], "mlp", li, 0)
    for aname, (am, nh) in _find_attn_out_projs(model, hidden).items():
        li = _layer_idx_from_name(aname)
        _reg(aname, am, _writer_kind(aname, False), li, nh)

    base_out: dict[str, Any] = {
        "root": root_id,
        "target_layer": int(target_layer),
        "layer_convention": layer_convention,
        "stop_block": int(stop_blk),
        "depth": int(depth),
        "top_k_nodes": int(top_k_nodes),
        "n_inputs": len(seqs),
    }
    if not writers:
        base_out.update({
            "root_score": None,
            "nodes": [{"id": root_id, "kind": "direction", "layer": int(target_layer), "hop": 0}],
            "edges": [],
            "top_paths": [],
            "per_input_top_tokens": [],
        })
        return base_out

    fn_mod = _find_final_norm_module(model) if use_final_hidden else None
    emb_mod = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else getattr(getattr(model, "model", model), "embed_tokens", None)

    orig_req = [(p, p.requires_grad) for p in model.parameters()]
    was_training = bool(model.training)
    was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
    disabled_gc = False

    def _first_tensor(args: Any, kwargs: dict | None = None) -> torch.Tensor | None:
        if isinstance(args, tuple) and args and isinstance(args[0], torch.Tensor):
            return args[0]
        if isinstance(args, torch.Tensor):
            return args
        if kwargs:
            hs_kw = kwargs.get("hidden_states")
            if isinstance(hs_kw, torch.Tensor):
                return hs_kw
        return None

    def _two_d(t: torch.Tensor) -> torch.Tensor:
        return t[0] if t.dim() == 3 else t

    def _trace_once(node: dict[str, Any], g_targets: list[torch.Tensor] | None) -> dict[str, Any]:
        is_root = node["kind"] == "direction"
        dst_mk = node.get("module")
        dst_layer = node.get("layer")
        acc: dict[str, list[torch.Tensor]] = {}  # mk -> [direct, causal] (vector per neuron/head or scalar tensor)
        embed_direct = 0.0
        score_sum = 0.0
        g_sub_list: list[torch.Tensor] = []
        tok_rows: list[list[dict[str, Any]]] = []
        exec_index: dict[str, int] = {}

        for s_idx, s_ids in enumerate(seqs):
            ids_dev = s_ids.to(dev)
            T_s = int(s_ids.shape[1])
            lo, hi = _resolve_hidden_span(T_s, pos)
            exec_order: list[str] = []
            st_x: dict[str, torch.Tensor] = {}
            st_y: dict[str, torch.Tensor] = {}
            st_g: dict[str, torch.Tensor] = {}
            emb: dict[str, torch.Tensor] = {}
            cap: dict[str, Any] = {}
            hooks = []

            def _track_emb(t: torch.Tensor) -> torch.Tensor:
                if "x" not in emb:
                    emb["x"] = _two_d(t).detach().float().cpu()
                if not t.requires_grad:
                    t = t.detach().requires_grad_(True)

                def _sg(grad: torch.Tensor) -> None:
                    emb["g"] = _two_d(grad).detach().float().cpu()

                t.register_hook(_sg)
                return t

            if emb_mod is not None:
                def _emb_hook(_m, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    if isinstance(t, torch.Tensor) and "x" not in emb:
                        t2 = _track_emb(t)
                        if t2 is not t:
                            return (t2, *out[1:]) if isinstance(out, tuple) else t2
                    return out

                hooks.append(emb_mod.register_forward_hook(_emb_hook))
            else:
                def _emb_pre(_m, args, kwargs):
                    x0 = _first_tensor(args, kwargs)
                    if x0 is not None and "x" not in emb:
                        x2 = _track_emb(x0)
                        if isinstance(args, tuple) and args and args[0] is x0:
                            return (x2, *args[1:]), kwargs
                        if kwargs.get("hidden_states") is x0:
                            kwargs = dict(kwargs)
                            kwargs["hidden_states"] = x2
                            return args, kwargs
                    return None

                hooks.append(blocks[0].register_forward_pre_hook(_emb_pre, with_kwargs=True))

            g_dst_cpu = None if is_root else g_targets[s_idx]  # type: ignore[index]

            for mk, meta_w in writers.items():
                def _whook(_m, inp, out, mk=mk, meta_w=meta_w):
                    if mk in st_y or mk in cap.get("done", ()):
                        return None
                    x_in = _first_tensor(inp)
                    y = out[0] if isinstance(out, (tuple, list)) else out
                    if not isinstance(y, torch.Tensor) or x_in is None:
                        return None
                    if (not is_root) and mk == dst_mk:
                        g_dev = g_dst_cpu.to(device=y.device, dtype=torch.float32)  # type: ignore[union-attr]
                        x2 = _two_d(x_in).float()
                        mod_dst = meta_w["mod"]
                        w_raw = mod_dst.weight.detach()
                        if node["kind"] == "mlp_neuron":
                            j = int(node["unit"])
                            w_col = (w_raw[:, j] if isinstance(mod_dst, nn.Linear) else w_raw[j, :]).to(device=y.device, dtype=torch.float32)
                            coef = g_dev @ w_col
                            s_val = (x2[:, j] * coef).sum()
                        elif node["kind"] == "attn_head":
                            nh_d = int(node["n_heads"])
                            hd = x2.shape[-1] // nh_d
                            sl = slice(int(node["head"]) * hd, (int(node["head"]) + 1) * hd)
                            w_h = (w_raw[:, sl] if isinstance(mod_dst, nn.Linear) else w_raw[sl, :].t()).to(device=y.device, dtype=torch.float32)
                            coef = g_dev @ w_h
                            s_val = (x2[:, sl] * coef).sum()
                        else:
                            s_val = (_two_d(y).float() * g_dev).sum()
                        cap["val"] = s_val
                        cap["done"] = (mk,)
                        raise _DirectionTargetCaptured()
                    exec_order.append(mk)
                    st_x[mk] = _two_d(x_in).detach().float().cpu()
                    st_y[mk] = _two_d(y).detach().float().cpu()
                    if y.requires_grad:
                        def _sg(grad: torch.Tensor, mk=mk) -> None:
                            st_g[mk] = _two_d(grad).detach().float().cpu()

                        y.register_hook(_sg)
                    return None

                hooks.append(meta_w["mod"].register_forward_hook(_whook))

            # Where to read g_sub_in for non-root destinations
            sub_mode = None
            if not is_root:
                blk = blocks[int(dst_layer)] if dst_layer is not None and 0 <= int(dst_layer) < n_layers else None
                norm_m = None
                if node["kind"] == "mlp_neuron" and blk is not None:
                    norm_m = next((getattr(blk, a) for a in _MLP_INPUT_NORM_ATTRS if isinstance(getattr(blk, a, None), nn.Module)), None)
                if norm_m is not None:
                    sub_mode = "norm"

                    def _norm_pre(_m, inp):
                        h = _first_tensor(inp)
                        if h is not None and h.requires_grad and "sub_g_hooked" not in cap:
                            cap["sub_g_hooked"] = True

                            def _sg(grad: torch.Tensor) -> None:
                                cap["g_sub"] = _two_d(grad).detach().float().cpu()

                            h.register_hook(_sg)

                    hooks.append(norm_m.register_forward_pre_hook(_norm_pre))
                elif blk is not None:
                    sub_mode = "earlier_or_block" if node["kind"] == "mlp_neuron" else "block"

                    def _blk_pre(_m, args, kwargs):
                        h = _first_tensor(args, kwargs)
                        if h is not None and h.requires_grad and "blk_g_hooked" not in cap:
                            cap["blk_g_hooked"] = True

                            def _sg(grad: torch.Tensor) -> None:
                                cap["g_blk"] = _two_d(grad).detach().float().cpu()

                            h.register_hook(_sg)
                        return None

                    hooks.append(blk.register_forward_pre_hook(_blk_pre, with_kwargs=True))
            else:
                if not use_final_hidden:
                    exp_passes = _expected_block_passes(model, stop_blk)
                    seen_stop = [0]

                    def _stop(_m, _inp, out, exp_passes=exp_passes, seen_stop=seen_stop):
                        seen_stop[0] += 1
                        if seen_stop[0] < exp_passes:
                            return
                        t = out[0] if isinstance(out, (tuple, list)) else out
                        if isinstance(t, torch.Tensor):
                            cap["h_t"] = t
                            raise _DirectionTargetCaptured()

                    hooks.append(blocks[stop_blk].register_forward_hook(_stop))
                elif fn_mod is not None:
                    def _fn_pre(_m, inp):
                        h = _first_tensor(inp)
                        if h is not None and "h_pre" not in cap:
                            cap["h_pre"] = _two_d(h).detach().float().cpu()

                    hooks.append(fn_mod.register_forward_pre_hook(_fn_pre))

            out = None
            try:
                with torch.enable_grad():
                    try:
                        out = _forward_for_hidden(model, ids_dev, need_hidden_states=(is_root and use_final_hidden))
                    except _DirectionTargetCaptured:
                        pass
                    finally:
                        for h in hooks:
                            h.remove()
                    if is_root:
                        if "h_t" in cap:
                            h_t = cap["h_t"]
                        else:
                            hs = getattr(out, "hidden_states", None) if out is not None else None
                            if hs is None or len(hs) < n_layers + 1:
                                raise RuntimeError(f"Model did not return hidden_states[{n_layers}] for target_layer={target_layer}")
                            h_t = hs[-1]
                        h2 = _two_d(h_t).float()
                        obj = (h2[lo:hi] @ tdir_cpu.to(h2.device)).mean()
                        cap["val"] = obj
                        S = hi - lo
                        if use_final_hidden and "h_pre" in cap:
                            v_eff, _, _ = _linearize_final_norm_span(fn_mod, cap["h_pre"][lo:hi], h2[lo:hi].detach().cpu(), tdir_cpu)
                        else:
                            v_eff = tdir_cpu[None].expand(S, -1)
                        g_sub = torch.zeros((T_s, hidden), dtype=torch.float32)
                        g_sub[lo:hi] = v_eff / float(S)
                    s_val = cap.get("val")
                    if isinstance(s_val, torch.Tensor):
                        score_sum += float(s_val.detach().item())
                        if s_val.requires_grad:
                            s_val.backward()
            finally:
                del out
                cap.pop("h_t", None)
                cap.pop("val", None)

            if not is_root:
                g_sub = None  # type: ignore[assignment]
                if sub_mode == "norm":
                    g_sub = cap.get("g_sub")
                elif sub_mode == "earlier_or_block":
                    earlier = [mk for mk in exec_order if writers[mk]["layer"] == dst_layer]
                    if earlier:
                        g_sub = st_g.get(earlier[-1], torch.zeros((T_s, hidden), dtype=torch.float32))
                    else:
                        g_sub = cap.get("g_blk")
                elif sub_mode == "block":
                    g_sub = cap.get("g_blk")
                if g_sub is None:
                    g_sub = torch.zeros((T_s, hidden), dtype=torch.float32)
            g_sub_list.append(g_sub)

            if s_idx == 0:
                exec_index = {mk: i for i, mk in enumerate(exec_order)}
            if "x" in emb:
                embed_direct += float((g_sub * emb["x"]).sum().item())
                if is_root and "g" in emb:
                    gxi = (emb["g"] * emb["x"]).sum(dim=-1)
                    order = torch.argsort(gxi.abs(), descending=True)[: min(5, T_s)].tolist()
                    ids_l = s_ids.reshape(-1).tolist()
                    tok_rows.append([
                        {"pos": int(p), "token_id": int(ids_l[p]), "grad_x_input": round(float(gxi[p].item()), 6)} for p in order
                    ])

            for mk in exec_order:
                lay = writers[mk]["layer"]
                if lay is not None and int(lay) < int(min_layer):
                    continue
                y = st_y[mk]
                gf = st_g.get(mk, torch.zeros_like(y))
                kind = writers[mk]["kind"]
                mod_w = writers[mk]["mod"]
                nh = writers[mk]["n_heads"]
                x = st_x[mk]
                if kind == "mlp" or (kind == "attn" and nh >= 1 and x.shape[-1] % nh == 0):
                    w_raw = mod_w.weight.detach()
                    w_dev_t = w_raw.device
                    w_lin_f32 = (w_raw if isinstance(mod_w, nn.Linear) else (w_raw.t() if w_raw.dim() == 2 else w_raw)).float()
                    n_dir = ((g_sub.to(device=w_dev_t, dtype=torch.float32) @ w_lin_f32).cpu() * x).sum(dim=0)
                    n_cau = ((gf.to(device=w_dev_t, dtype=torch.float32) @ w_lin_f32).cpu() * x).sum(dim=0)
                    del w_lin_f32
                    if kind == "attn":
                        n_dir = n_dir.reshape(nh, -1).sum(dim=1)
                        n_cau = n_cau.reshape(nh, -1).sum(dim=1)
                else:
                    n_dir = (g_sub * y).sum().reshape(1)
                    n_cau = (gf * y).sum().reshape(1)
                if mk in acc:
                    acc[mk][0] += n_dir
                    acc[mk][1] += n_cau
                else:
                    acc[mk] = [n_dir.clone(), n_cau.clone()]

        n_seq = float(len(seqs))
        for mk in acc:
            acc[mk][0] /= n_seq
            acc[mk][1] /= n_seq
        return {
            "acc": acc,
            "score": score_sum / n_seq,
            "embed_direct": embed_direct / n_seq,
            "g_sub": g_sub_list,
            "tokens": tok_rows,
            "exec_index": exec_index,
        }

    def _cand_spec(mk: str, i: int) -> dict[str, Any]:
        meta_w = writers[mk]
        kind = meta_w["kind"]
        base = {"module": mk, "layer": meta_w["layer"]}
        if kind == "mlp":
            return {**base, "id": f"{mk}:n{i}", "kind": "mlp_neuron", "unit": int(i)}
        if kind == "attn" and meta_w["n_heads"] >= 1 and meta_w["d_in"] % meta_w["n_heads"] == 0:
            return {**base, "id": f"{mk}:h{i}", "kind": "attn_head", "head": int(i), "n_heads": int(meta_w["n_heads"])}
        return {**base, "id": mk, "kind": "conv_module"}

    def _select_parents(acc: dict[str, list[torch.Tensor]]) -> list[tuple[dict[str, Any], float, float, float]]:
        if not acc:
            return []
        gmax_dir = max(float(v[0].abs().max().item()) for v in acc.values())
        use_dir = gmax_dir >= float(min_edge_weight)
        den_dir = sum(float(v[0].abs().sum().item()) for v in acc.values())
        den_cau = sum(float(v[1].abs().sum().item()) for v in acc.values())
        cands: list[tuple[float, str, int]] = []
        for mk, (vd, vc) in acc.items():
            rank = vd if use_dir else vc
            k = min(int(top_k_nodes), int(rank.numel()))
            top = torch.argsort(rank.abs(), descending=True)[:k].tolist()
            for i in top:
                cands.append((abs(float(rank[i].item())), mk, int(i)))
        cands.sort(key=lambda r: (-r[0], r[1], r[2]))
        out_sel = []
        for score, mk, i in cands:
            if score < float(min_edge_weight):
                continue
            d = float(acc[mk][0][i].item())
            c = float(acc[mk][1][i].item())
            share = (abs(d) / den_dir) if (use_dir and den_dir > 1e-12) else (abs(c) / max(den_cau, 1e-12))
            out_sel.append((_cand_spec(mk, i), d, c, share))
            if len(out_sel) >= int(top_k_nodes):
                break
        return out_sel

    try:
        for p, _ in orig_req:
            p.requires_grad_(False)
        model.eval()
        if was_gc and hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
                disabled_gc = True
            except Exception:
                disabled_gc = False

        root_node = {"id": root_id, "kind": "direction", "layer": int(target_layer)}
        root_res = _trace_once(root_node, None)
        exec_idx = root_res["exec_index"]
        root_acc = root_res["acc"]

        def _to_root(spec: dict[str, Any]) -> tuple[float | None, float | None]:
            a = root_acc.get(spec["module"])
            if a is None:
                return None, None
            i = int(spec.get("unit", spec.get("head", 0)))
            if i >= a[0].numel():
                i = 0
            return float(a[0][i].item()), float(a[1][i].item())

        nodes_by_id: dict[str, dict[str, Any]] = {
            root_id: {
                "id": root_id, "kind": "direction", "layer": int(target_layer), "hop": 0,
                "exec_index": None, "score": round(root_res["score"], 6),
                "embed_direct": round(root_res["embed_direct"], 6),
            }
        }
        edges: list[dict[str, Any]] = []
        frontier: list[tuple[str, dict[str, Any], list[torch.Tensor] | None, dict[str, Any] | None]] = [(root_id, root_node, None, root_res)]
        expanded: set[str] = set()

        for hop in range(1, int(depth) + 1):
            next_frontier = []
            for dst_id, dst_spec, g_t, pre_res in frontier:
                if dst_id in expanded:
                    continue
                expanded.add(dst_id)
                res = pre_res if pre_res is not None else _trace_once(dst_spec, g_t)
                if dst_id != root_id:
                    nodes_by_id[dst_id]["readout_score"] = round(res["score"], 6)
                    nodes_by_id[dst_id]["embed_direct"] = round(res["embed_direct"], 6)
                for spec, d, c, share in _select_parents(res["acc"]):
                    sid = spec["id"]
                    if sid not in nodes_by_id:
                        dr, cr = _to_root(spec)
                        nodes_by_id[sid] = {
                            **spec,
                            "hop": hop,
                            "exec_index": exec_idx.get(spec["module"]),
                            "direct_to_root": round(dr, 6) if dr is not None else None,
                            "causal_to_root": round(cr, 6) if cr is not None else None,
                        }
                    edges.append({
                        "src": sid,
                        "dst": dst_id,
                        "direct_weight": round(d, 6),
                        "causal_weight": round(c, 6),
                        "edge_share": round(share, 6),
                        "hop": hop,
                    })
                    if hop < int(depth) and sid not in expanded:
                        next_frontier.append((sid, spec, res["g_sub"], None))
            frontier = next_frontier
            if not frontier:
                break
    finally:
        if disabled_gc and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        if was_training:
            model.train()
        for p, req in orig_req:
            p.requires_grad_(req)

    incoming: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        incoming.setdefault(e["dst"], []).append(e)

    def _paths(nid: str, seen: frozenset) -> list[tuple[list[str], list[dict[str, Any]]]]:
        ins = [e for e in incoming.get(nid, []) if e["src"] not in seen]
        if not ins:
            return [([nid], [])]
        res_p = []
        for e in ins:
            for pn, pe in _paths(e["src"], seen | {e["src"]}):
                res_p.append((pn + [nid], pe + [e]))
        return res_p

    top_paths = []
    for pn, pe in _paths(root_id, frozenset({root_id})):
        if not pe:
            continue
        share = 1.0
        for e in pe:
            share *= float(e["edge_share"])
        top_paths.append({
            "nodes": pn,
            "edge_weights": [e["direct_weight"] for e in pe],
            "path_share": round(share, 6),
            "min_edge_direct": round(min(abs(float(e["direct_weight"])) for e in pe), 6),
            "length": len(pe),
        })
    top_paths.sort(key=lambda r: (-r["path_share"], -r["min_edge_direct"]))

    base_out.update({
        "root_score": round(root_res["score"], 6),
        "nodes": list(nodes_by_id.values()),
        "edges": edges,
        "top_paths": top_paths[:20],
        "per_input_top_tokens": root_res["tokens"],
    })
    return base_out


# ----------------------------------------------------------------------------------------------- iterative peeling (Hydra)
def peel_causal_circuit(
    model: nn.Module,
    inputs: torch.Tensor | list[int] | list[torch.Tensor | list[int]],
    *,
    n_rounds: int = 2,
    alpha: float = 4.0,
    sites_per_round: int = 1,
    top_k_sites: int = 6,
    top_n_neurons: int | None = None,
    allow_repeat_modules: bool = False,
    target_dir: torch.Tensor | None = None,
    target_layer: int | None = None,
    target_layer_convention: str = "block",
    protect_dir: torch.Tensor | None = None,
    protect_layer: int | None = None,
    protect_layer_convention: str | None = None,
    target_spans: list[int | tuple[int, int] | None] | None = None,
    healthy_spans: list[tuple[int, int] | None] | None = None,
    grad_objective: str = "auto",
    target_token_ids: int | list[int] | torch.Tensor | None = None,
    contrast_token_ids: int | list[int] | torch.Tensor | None = None,
    healthy_cost: str = "linear",
    center: str = "none",
    grad_mode: str = "summary",
    grad_checkpointing: bool = True,
) -> dict[str, Any]:
    """Iterative causal peeling to expose backup ("Hydra") circuits.

    Each round captures causal traces under the patches of all previous rounds (attached in memory),
    runs ``diagnose_causal_traces``, ablates the top new site(s) with an ``MoIPatch`` edit and records
    how much of the target objective is left. Backup modules that only wake up once the primary
    site is removed show up in later rounds. All base weights are restored bit-identically on exit.
    Note: with ``allow_repeat_modules=True`` a module can be edited twice in memory, but the returned
    ``cumulative_patch`` applies each edit from base weights (the later edit wins on that module).
    """
    from .moipatch import Edit, MoIPatch, _sha256

    if int(n_rounds) < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
    if int(sites_per_round) < 1:
        raise ValueError(f"sites_per_round must be >= 1, got {sites_per_round}")
    seqs = _normalize_seq_list(inputs)
    eff_grad_mode = "compressed" if healthy_cost == "fisher" else grad_mode

    mods = dict(model.named_modules())
    mlps = find_mlps(model)
    hidden = _infer_hidden_size(model, mlps)
    attns = _find_attn_out_projs(model, hidden)
    if not mlps and not attns:
        raise ValueError("peel_causal_circuit requires at least one MLP or attention/conv output writer")
    mod_by_mk: dict[str, nn.Module] = {}
    base_meta: dict[str, dict[str, Any]] = {}
    base_hash: dict[str, str] = {}
    for full in [g.down for g in mlps] + list(attns):
        mk = _safe_module_suffix(full)
        m = mods[full]
        mod_by_mk[mk] = m
        base_meta[mk] = _weight_fingerprint(full, m.weight.detach())
        base_hash[mk] = _sha256(m.weight.detach())

    def _capture_all() -> list[CompactTrace]:
        out_t = []
        for i, seq in enumerate(seqs):
            out_t.append(capture_causal_trace(
                model,
                seq.reshape(-1),
                target_span=(target_spans[i] if target_spans is not None and i < len(target_spans) else None),
                healthy_span=(healthy_spans[i] if healthy_spans is not None and i < len(healthy_spans) else None),
                grad_mode=eff_grad_mode,
                target_dir=target_dir,
                target_layer=target_layer,
                target_layer_convention=target_layer_convention,
                protect_dir=protect_dir,
                protect_layer=protect_layer,
                protect_layer_convention=protect_layer_convention,
                grad_objective=grad_objective,
                target_token_ids=target_token_ids,
                contrast_token_ids=contrast_token_ids,
                healthy_cost=healthy_cost,
                center=center,
                grad_checkpointing=grad_checkpointing,
                top_k_attn=0,
                capture_cache=False,
            ))
        return out_t

    def _scores(ts: list[CompactTrace]) -> tuple[float, float | None]:
        sc = sum(float(ct.meta["target_obj_score"]) for ct in ts) / len(ts)
        ps = [float(ct.meta["protect_dir_score"]) for ct in ts if "protect_dir_score" in ct.meta]
        return sc, (sum(ps) / len(ps) if ps else None)

    active_handles: list[Any] = []
    cumulative_edits: list[Edit] = []
    used_modules: set[str] = set()
    rounds_out: list[dict[str, Any]] = []
    score_traj: list[float] = []
    protect_traj: list[float] = []
    restore_exact_ok = False
    try:
        traces = _capture_all()
        for r in range(int(n_rounds)):
            pre_score, pre_prot = _scores(traces)
            if r == 0:
                score_traj.append(round(pre_score, 6))
                if pre_prot is not None:
                    protect_traj.append(round(pre_prot, 6))
            diag = diagnose_causal_traces(
                traces,
                split_ab=len(traces) >= 2,
                top_k_sites=max(int(top_k_sites), int(sites_per_round) + len(used_modules)),
                top_n_neurons=int(top_n_neurons or 64),
                healthy_cost=healthy_cost,
                center=center,
                model_or_weights=model,
            )
            picked = []
            for s_row in diag["all_sites"]:
                mk = s_row["module"]
                if float(s_row["target_supp"]) <= 1e-8 or mk not in base_meta:
                    continue
                if (not allow_repeat_modules) and mk in used_modules:
                    continue
                picked.append(mk)
                if len(picked) >= int(sites_per_round):
                    break
            if not picked:
                break
            round_edits = []
            for mk in picked:
                bm = base_meta[mk]
                idx = (
                    torch.tensor(diag["top_neurons"][mk][: int(top_n_neurons)], dtype=torch.long)
                    if top_n_neurons is not None and diag["top_neurons"].get(mk)
                    else None
                )
                round_edits.append(Edit(
                    module=mk,
                    u=diag["directions"][mk].double(),
                    alpha=float(alpha),
                    idx=idx,
                    base_sha256=bm["base_sha256"],
                    base_dtype=bm["base_dtype"],
                    shape=bm["shape"],
                ))
            handle = MoIPatch(edits=round_edits, meta={"round": r}).attach(
                model, verify_fingerprint=not allow_repeat_modules
            )
            handle.on()
            active_handles.append(handle)
            cumulative_edits.extend(round_edits)
            used_modules.update(picked)

            traces = _capture_all()
            post_score, post_prot = _scores(traces)
            score_traj.append(round(post_score, 6))
            if post_prot is not None:
                protect_traj.append(round(post_prot, 6))
            entry: dict[str, Any] = {
                "round": r,
                "pre_score": round(pre_score, 6),
                "post_score": round(post_score, 6),
                "score_drop": round(pre_score - post_score, 6),
                "selected_modules": picked,
                "top_sites": diag["sites"][: int(top_k_sites)],
                "split_mode": diag["split_mode"],
            }
            if pre_prot is not None:
                entry["pre_protect_score"] = round(pre_prot, 6)
            if post_prot is not None:
                entry["post_protect_score"] = round(post_prot, 6)
            rounds_out.append(entry)
    finally:
        for h in reversed(active_handles):
            h.off()
        restore_exact_ok = all(_sha256(mod_by_mk[mk].weight.detach()) == base_hash[mk] for mk in base_hash)

    cumulative_patch = MoIPatch(
        edits=list(cumulative_edits),
        meta={
            "source": "peel_causal_circuit",
            "n_rounds": len(rounds_out),
            "alpha": float(alpha),
            "selected_modules": [e.module for e in cumulative_edits],
        },
    )
    base_score = score_traj[0] if score_traj else None
    final_score = score_traj[-1] if score_traj else None
    return {
        "n_rounds_requested": int(n_rounds),
        "n_rounds_completed": len(rounds_out),
        "alpha": float(alpha),
        "score_trajectory": score_traj,
        "protect_trajectory": protect_traj,
        "base_score": base_score,
        "final_score": final_score,
        "total_score_drop": (round(base_score - final_score, 6) if base_score is not None and final_score is not None else None),
        "rounds": rounds_out,
        "selected_modules": [e.module for e in cumulative_edits],
        "cumulative_patch": cumulative_patch,
        "restore_exact_ok": bool(restore_exact_ok),
    }


@torch.no_grad()
def verify_onset_suppression(
    model: nn.Module,
    candidates: dict[str, Any],
    onset_rows: Sequence[dict[str, Any]],
    *,
    device: torch.device | str | None = None,
) -> dict[str, dict[str, float]]:
    """Measure token-0 onset log-prob drop and top-1 flip rate for candidate `MoIPatch` edits."""
    if device is not None:
        dev = torch.device(device)
    else:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

    was_training = bool(model.training)
    model.eval()
    results: dict[str, dict[str, float]] = {}
    try:
        base_cached: list[tuple[torch.Tensor, int, float, float, int]] = []
        for row in onset_rows:
            p_ids = [int(t) for t in row.get("prompt_ids") or ()]
            first_raw = row.get("first_target_id")
            if first_raw is None:
                g_ids = row.get("gen_ids") or ()
                first_raw = g_ids[0] if g_ids else None
            if not p_ids or first_raw is None:
                continue
            first_tok = int(first_raw)
            ids = torch.tensor([p_ids], dtype=torch.long, device=dev)
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            lp = F.log_softmax(logits, dim=-1)
            base_cached.append((ids, first_tok, float(lp[first_tok].item()), float(lp[first_tok].exp().item()), int(logits.argmax().item())))

        if not base_cached:
            return {
                name: {"onset_logp_drop": 0.0, "onset_prob_drop": 0.0, "onset_top1_flip_rate": 0.0}
                for name in candidates
            }

        n = len(base_cached)
        for name, patch in candidates.items():
            lp_drops: list[float] = []
            p_drops: list[float] = []
            flips: list[float] = []
            h = patch.attach(model, verify_fingerprint=False)
            try:
                h.on()
                for ids, first_tok, base_lp, base_p, _base_top1 in base_cached:
                    logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                    lp = F.log_softmax(logits, dim=-1)
                    lp_drops.append(base_lp - float(lp[first_tok].item()))
                    p_drops.append(base_p - float(lp[first_tok].exp().item()))
                    flips.append(1.0 if int(logits.argmax().item()) != first_tok else 0.0)
            finally:
                if h.active:
                    h.off()
            results[name] = {
                "onset_logp_drop": round(sum(lp_drops) / n, 5),
                "onset_prob_drop": round(sum(p_drops) / n, 5),
                "onset_top1_flip_rate": round(sum(flips) / n, 5),
            }
    finally:
        if was_training:
            model.train()
    return results
