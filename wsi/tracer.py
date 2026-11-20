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
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .capture import _is_linear, find_blocks, find_mlps, trace_all

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


def _find_attn_out_projs(model: nn.Module, hidden: int) -> dict[str, tuple[nn.Module, int]]:
    out = {}
    for n, m in model.named_modules():
        if "mtp" in n or "visual" in n or not _is_linear(m):
            continue
        if n.endswith((".o_proj", ".out_proj", ".c_proj")) and ".mlp." not in n:
            nh = _infer_attn_heads(model, n, m.weight.shape[1] if isinstance(m, nn.Linear) else m.weight.shape[0])
            out[n] = (m, nh)
    return out


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
                    flat[f"grad_s.{idx}.{k}"] = v.detach().cpu().contiguous()
                    tkeys.append(k)
                elif isinstance(v, tuple):
                    smeta[k] = list(v)
                else:
                    smeta[k] = v
            smeta["tensor_keys"] = tkeys
            grad_summary_meta[str(idx)] = smeta
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
    target_span: tuple[int, int] | None = None,
    healthy_span: tuple[int, int] | None = None,
    grad_checkpointing: bool = True,
    pareto_betas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
) -> CompactTrace:
    """Record a Pareto-optimal compressed token trace (~80-95 MiB per 1.6k tok on 0.8B):
    INT8 Hadamard-rotated residuals, adaptive or fixed Top-K FP4 bitmask/bitpack MLP units
    (default `top_k_mlp="auto"` picks per-layer `K` to reach `mlp_energy_target=0.95`),
    `head_quant` pre-`W_O` head vectors + post-`W_O` L2 norms, sparse Top-16 attention targets,
    and INT8 KV/recurrent cache states. When ``capture_grad=True``, also runs target & healthy
    backward passes to record per-module causal summaries (and compressed token-level gradients
    when ``grad_mode="compressed"``)."""
    if capture_grad and grad_mode not in ("compressed", "summary"):
        raise ValueError(f"grad_mode must be 'compressed' or 'summary', got {grad_mode!r}")

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
    if capture_grad and T < 2:
        raise ValueError(f"capture_compact(capture_grad=True) requires seq_len >= 2, got {T}")

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

    d0 = mods[mlps[0].down]
    hidden = d0.weight.shape[0] if isinstance(d0, nn.Linear) else d0.weight.shape[1]
    attns = _find_attn_out_projs(model, hidden)

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

    for aname, (am, nh) in attns.items():
        li = _layer_idx_from_name(aname)

        def ahook(mod, inp, out, aname=aname, li=li, nh=nh):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            w = mod.weight.detach().to(x.device, dtype=torch.float32)
            wh = w.t().reshape(nh, -1, w.shape[0]) if isinstance(mod, nn.Linear) else w.reshape(nh, -1, w.shape[1])
            l2 = torch.einsum("thd,hdo->tho", x.reshape(x.shape[0], nh, -1), wh).norm(dim=2)
            h_entry: dict[str, Any] = {"layer": li, "n_heads": nh, "l2": l2.to(torch.float16).cpu()}
            if head_quant != "norm":
                h_enc = _encode_resid(x, head_quant, group_size, hadamard_block)
                h_entry["data"] = h_enc["data"]
                h_entry["scales"] = h_enc["scales"]
                h_entry["shape"] = h_enc["shape"]
            head_out[aname] = h_entry

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
                if top_k_attn > 0:
                    try:
                        out = model(input_ids=ids.to(dev), use_cache=capture_cache)
                    except (TypeError, ValueError, NotImplementedError):
                        resid_out.clear()
                        mlp_topk_out.clear()
                        mlp_meta_out.clear()
                        head_out.clear()
                        out = model(input_ids=ids.to(dev), use_cache=False)
                else:
                    out = model(input_ids=ids.to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()

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
    eff_grad_mode = "none"

    if capture_grad:
        eff_grad_mode = grad_mode
        lo_t, hi_t = _resolve_target_span(T, target_span)
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
            grad_mods[mk] = {"mod": mods[dname], "kind": "mlp", "layer": int(li)}
        for aname, (am, _nh) in attns.items():
            mk = _safe_module_suffix(aname)
            li = _layer_idx_from_name(aname)
            grad_mods[mk] = {"mod": am, "kind": "attn", "layer": int(li) if li is not None else None}

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

        def _run_backward(lo: int, hi: int, sign: float, *, capture_acts: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            st_h: dict[str, torch.Tensor] = {}
            st_y: dict[str, torch.Tensor] = {}
            st_g: dict[str, torch.Tensor] = {}
            ghandles = []

            emb_mod = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else getattr(getattr(model, "model", model), "embed_tokens", None)
            if emb_mod is not None:
                def _emb_hook(_m, _inp, emb_out):
                    if torch.is_grad_enabled() and isinstance(emb_out, torch.Tensor) and not emb_out.requires_grad:
                        return emb_out.detach().requires_grad_(True)
                    return emb_out
                ghandles.append(emb_mod.register_forward_hook(_emb_hook))

            for mk, meta_m in grad_mods.items():
                mod = meta_m["mod"]

                def _mod_hook(_m, inp, mod_out, mk=mk):
                    if capture_acts and mk not in st_h:
                        x_in = inp[0] if isinstance(inp, tuple) else inp
                        st_h[mk] = (x_in[0] if x_in.dim() == 3 else x_in).detach().float().cpu()
                    t = mod_out[0] if isinstance(mod_out, tuple) else mod_out
                    if capture_acts and mk not in st_y:
                        st_y[mk] = (t[0] if t.dim() == 3 else t).detach().float().cpu()
                    if torch.is_grad_enabled() and isinstance(t, torch.Tensor):
                        need_replace = False
                        if not t.requires_grad:
                            t = t.detach().requires_grad_(True)
                            need_replace = True

                        def _save_g(grad, mk=mk):
                            st_g[mk] = (grad[0] if grad.dim() == 3 else grad).detach().float().cpu()

                        t.register_hook(_save_g)
                        if need_replace:
                            return (t, *mod_out[1:]) if isinstance(mod_out, tuple) else t
                    return mod_out

                ghandles.append(mod.register_forward_hook(_mod_hook))

            logits = lp = obj = None
            try:
                with torch.enable_grad():
                    logits = _tail_logits(model, ids_dev, lo)[: hi - lo].float()
                    lp = F.log_softmax(logits, dim=-1).gather(1, ids_dev[0, lo + 1 : hi + 1, None]).squeeze(1)
                    obj = sign * lp.mean()
                    if obj.requires_grad:
                        obj.backward()
            finally:
                for gh in ghandles:
                    gh.remove()
                del logits, lp, obj
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            for mk in grad_mods:
                if mk in st_y and mk not in st_g:
                    st_g[mk] = torch.zeros_like(st_y[mk])
            return st_h, st_y, st_g

        try:
            acts_h, acts_y, g_target_map = _run_backward(lo_t, hi_t, 1.0, capture_acts=True)
            if hi_h > lo_h:
                _, _, g_healthy_map = _run_backward(lo_h, hi_h, -1.0, capture_acts=False)
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

                w_raw = mod.weight.detach().float().cpu()
                W_lin = w_raw if isinstance(mod, nn.Linear) else (w_raw.t() if w_raw.dim() == 2 else w_raw)

                neuron_supp_attr = ((g_t @ W_lin) * h_act).sum(dim=0).float()
                neuron_dnll_attr = (-((g_h @ W_lin) * h_act).sum(dim=0).float()) if hi_h > lo_h else torch.zeros(U, dtype=torch.float32)

                C_t = 0.5 * (g_t.t() @ y_act + y_act.t() @ g_t)
                C_h = (0.5 * (g_h.t() @ y_act + y_act.t() @ g_h)) if hi_h > lo_h else torch.zeros((D, D), dtype=torch.float32)

                # Restrict curvature eigendecomposition to the gradient-active subspace so massive
                # nullspace activations (orthogonal to both g_t and g_h) cannot create cross-term decoys.
                G_gram = g_t.t() @ g_t + ((g_h.t() @ g_h) if hi_h > lo_h else 0.0)
                gevals, gevecs = torch.linalg.eigh(0.5 * (G_gram + G_gram.t()))
                g_max = float(gevals[-1].item())
                r = min(max(1, int(grad_rank)), D)
                if g_max > 1e-12:
                    keep = (gevals >= 1e-3 * g_max).nonzero(as_tuple=True)[0]
                    keep = keep[-min(len(keep), max(r, 1)):]
                    Q = gevecs[:, keep]
                    C_t_sub = 0.5 * (Q.t() @ (C_t + C_t.t()) @ Q)
                    C_h_sub = 0.5 * (Q.t() @ (C_h + C_h.t()) @ Q)
                    evals_t, evecs_t_sub = torch.linalg.eigh(C_t_sub)
                    evecs_t = Q @ evecs_t_sub
                    evals_h, evecs_h_sub = torch.linalg.eigh(C_h_sub)
                    evecs_h = Q @ evecs_h_sub
                else:
                    Q = None
                    evals_t = torch.zeros(1, dtype=torch.float32)
                    evecs_t = torch.eye(D, 1, dtype=torch.float32)
                    evals_h = torch.zeros(1, dtype=torch.float32)
                    evecs_h = torch.eye(D, 1, dtype=torch.float32)

                r_t = min(r, int(evals_t.numel()))
                idx_t = torch.argsort(evals_t, descending=True)[:r_t]
                c_target_eigvecs = evecs_t[:, idx_t].t().contiguous().float()
                c_target_eigvals = evals_t[idx_t].float()

                r_h = min(r, int(evals_h.numel()))
                idx_h = torch.argsort(evals_h.abs(), descending=True)[:r_h]
                c_healthy_eigvecs = evecs_h[:, idx_h].t().contiguous().float()
                c_healthy_eigvals = evals_h[idx_h].float()

                p_vecs = []
                for beta in betas:
                    if Q is not None:
                        _, ev_p_sub = torch.linalg.eigh(C_t_sub + beta * C_h_sub)
                        p_vecs.append((Q @ ev_p_sub[:, -1]).contiguous().float())
                    else:
                        p_vecs.append(evecs_t[:, 0].contiguous().float())
                pareto_eigvecs = torch.stack(p_vecs, dim=0) if p_vecs else torch.empty((0, D), dtype=torch.float32)

                paired_diff = (
                    (y_act[lo_t:hi_t].mean(dim=0) - y_act[lo_h:hi_h].mean(dim=0)).float()
                    if hi_h > lo_h
                    else y_act[lo_t:hi_t].mean(dim=0).float()
                )
                u0 = c_target_eigvecs[0]
                target_supp_top1 = float(c_target_eigvals[0].item())
                healthy_dnll_top1 = float(-(u0 @ C_h @ u0).item())
                pareto_top1 = target_supp_top1 / max(abs(healthy_dnll_top1), 0.01)

                grad_summary_out[mk] = {
                    "kind": meta_m["kind"],
                    "layer": meta_m["layer"],
                    "d_out": D,
                    "d_in": U,
                    "neuron_supp_attr": neuron_supp_attr,
                    "neuron_dnll_attr": neuron_dnll_attr,
                    "paired_diff": paired_diff,
                    "c_target_eigvecs": c_target_eigvecs,
                    "c_target_eigvals": c_target_eigvals,
                    "c_healthy_eigvecs": c_healthy_eigvecs,
                    "c_healthy_eigvals": c_healthy_eigvals,
                    "pareto_eigvecs": pareto_eigvecs,
                    "pareto_betas": betas,
                    "target_supp_top1": round(target_supp_top1, 6),
                    "healthy_dnll_top1": round(healthy_dnll_top1, 6),
                    "pareto_top1": round(pareto_top1, 6),
                    "paired_diff_norm": round(float(paired_diff.norm().item()), 6),
                }

                if grad_mode == "compressed":
                    grad_target_out[mk] = _encode_resid(g_t, grad_quant, group_size, hadamard_block)
                    if hi_h > lo_h:
                        grad_healthy_out[mk] = _encode_resid(g_h, grad_quant, group_size, hadamard_block)

            trace_meta["grad_n_target"] = int(hi_t - lo_t)
            trace_meta["grad_n_healthy"] = int(max(0, hi_h - lo_h))
            trace_meta["target_span"] = [int(lo_t), int(hi_t)]
            trace_meta["healthy_span"] = [int(lo_h), int(hi_h)]
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
    )


def capture_causal_trace(
    model: nn.Module,
    input_ids: torch.Tensor | list[int],
    *,
    target_span: tuple[int, int] | None,
    healthy_span: tuple[int, int] | None = None,
    grad_mode: str = "compressed",
    **kwargs: Any,
) -> CompactTrace:
    """Convenience wrapper calling ``capture_compact(..., capture_grad=True)``."""
    return capture_compact(
        model,
        input_ids,
        capture_grad=True,
        grad_mode=grad_mode,
        target_span=target_span,
        healthy_span=healthy_span,
        **kwargs,
    )


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
        am = mods.get(mod_name)
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
) -> dict[str, Any]:
    """Load saved `CompactTrace` directories, run blind multi-site discovery, optionally save a `.moipatch`, and return a JSON summary."""
    traces = [CompactTrace.load(p) for p in trace_dirs]
    loop_traces = [CompactTrace.load(p) for p in loop_trace_dirs] if loop_trace_dirs else None
    res = wsi.discover_compact_circuit(
        traces,
        loop_traces=loop_traces,
        alpha=float(alpha),
        top_k_neurons=int(top_k_neurons),
        top_k_heads=int(top_k_heads),
    )
    saved_patch = None
    if save_patch_path:
        chosen = res["patches"].get(variant) or res["patch"]
        saved_patch = str(chosen.save(save_patch_path))
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
        "available_patches": list(res["patches"]),
        "saved_patch": saved_patch,
    }


def _resolve_edit_modules(
    model: nn.Module,
    items: list[tuple[str, Any]],
) -> tuple[dict[str, nn.Module], dict[str, torch.Tensor]]:
    """Resolve unique `module_suffix` names referenced by candidate `MoIPatch`es and cache CPU float32 weights."""
    from .moipatch import module_suffix

    names: dict[str, list[nn.Module]] = {}
    for name, mod in model.named_modules():
        if "mtp" in name or "visual" in name or getattr(mod, "weight", None) is None:
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
) -> list[dict[str, Any]]:
    """Fast 1st-order gradient prefilter scoring arbitrarily many candidate `MoIPatch`es via 1 backward/seq.

    Runs 1 backward pass per sequence (target log-prob and healthy NLL) in the model's native dtype with
    parameter gradients disabled, caching only input activations and output gradients at the edited modules.
    Ideal as Stage 1 to narrow hundreds/thousands of layer/neuron sites down before `screen_patch_candidates`.
    """
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

    def _run_grad_pass(seqs: list[torch.Tensor], spans: list[tuple[int, int]] | None, kind: str) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
        pieces: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {mk: [] for mk in resolved}
        denom = max(1, n_t if kind == "target" else n_h)
        with torch.enable_grad():
            for i, ids in enumerate(seqs):
                T = int(ids.shape[1])
                if T < 2:
                    continue
                if kind == "target":
                    lo, hi = _resolve_target_span(T, spans[i] if (spans is not None and i < len(spans)) else None)
                else:
                    lo, hi = 0, T - 1

                st: dict[str, dict[str, Any]] = {"h": {}, "y": {}}
                handles = []
                for mk, mod in resolved.items():
                    def _hook(_m, inp, out, mk=mk):
                        x_in = inp[0]
                        st["h"][mk] = (x_in[0] if x_in.dim() == 3 else x_in).detach().float().cpu()
                        t = out[0] if isinstance(out, tuple) else out
                        if not t.requires_grad:
                            t = t.detach().requires_grad_(True)
                        else:
                            t.retain_grad()
                        st["y"][mk] = t
                        return (t, *out[1:]) if isinstance(out, tuple) else t

                    handles.append(mod.register_forward_hook(_hook))

                logits = lp = obj = None
                try:
                    logits = _tail_logits(model, ids, lo)[: hi - lo].float()
                    lp = F.log_softmax(logits, dim=-1).gather(1, ids[0, lo + 1 : hi + 1, None]).squeeze(1)
                    obj = (lp.sum() / denom) if kind == "target" else (-lp.sum() / denom)
                    if obj.requires_grad:
                        obj.backward()
                    for mk in resolved:
                        if mk in st["y"] and mk in st["h"]:
                            ty = st["y"][mk]
                            if ty.grad is not None:
                                g = (ty.grad[0] if ty.grad.dim() == 3 else ty.grad).detach().float().cpu()
                            else:
                                y0 = ty.detach()[0] if ty.dim() == 3 else ty.detach()
                                g = torch.zeros_like(y0, dtype=torch.float32, device="cpu")
                            pieces[mk].append((g, st["h"][mk]))
                finally:
                    for h in handles:
                        h.remove()
                    del st, logits, lp, obj
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
        return pieces

    try:
        tgt_pieces = _run_grad_pass(t_ids, target_spans, "target")
        hlt_pieces = _run_grad_pass(h_ids, None, "healthy")
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

    rows: list[dict[str, Any]] = []
    for name, patch in items:
        grad_supp = -_first_order(patch, tgt_pieces)
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
) -> dict[str, Any]:
    """Offline causal diagnosis across unified MLP & Attention sites from captured causal ``CompactTrace``s.

    Uses Half-A / Half-B held-out cross-validation on low-rank target & healthy curvature matrices
    ``C_target`` and ``C_healthy`` to pick the site, direction (``paired``, ``causal``, or Pareto
    Lagrangian ``C_target + beta * C_healthy``), and net causal neurons without touching any GPU.
    """
    if not traces:
        raise ValueError("diagnose_causal_traces requires at least one CompactTrace")
    if not any(ct.grad_summary for ct in traces):
        raise ValueError("Traces lack grad_summary; capture with capture_compact(..., capture_grad=True)")

    all_mods = sorted({mk for ct in traces for mk in ct.grad_summary})
    half_a = traces[0::2] if (split_ab and len(traces) >= 2) else traces
    half_b = traces[1::2] if (split_ab and len(traces) >= 2) else traces

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
        paired = torch.zeros(D, dtype=torch.float32)
        n_supp = torch.zeros(U, dtype=torch.float32)
        n_dnll = torch.zeros(U, dtype=torch.float32)

        for gs, nt, nh in items:
            wt = (nt / tot_t) if tot_t > 0 else (1.0 / len(items))
            vt = gs["c_target_eigvecs"].float()
            lt = gs["c_target_eigvals"].float()
            C_t += wt * (vt.t() @ (lt[:, None] * vt))
            paired += wt * gs["paired_diff"].float()
            n_supp += wt * gs["neuron_supp_attr"].float()
            if tot_h > 0 and nh > 0:
                wh = nh / tot_h
                vh = gs["c_healthy_eigvecs"].float()
                lh = gs["c_healthy_eigvals"].float()
                C_h += wh * (vh.t() @ (lh[:, None] * vh))
                n_dnll += wh * gs["neuron_dnll_attr"].float()

        C_t = 0.5 * (C_t + C_t.t())
        C_h = 0.5 * (C_h + C_h.t())
        return {
            "kind": gs0["kind"],
            "layer": gs0["layer"],
            "C_t": C_t,
            "C_h": C_h,
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

        C_t_A, C_h_A, diff_A = pa["C_t"], pa["C_h"], pa["paired_diff"]
        C_t_B, C_h_B = pb["C_t"], pb["C_h"]

        # Orthonormal basis of the Half-A active subspace (all stored C_t/C_h eigenvectors)
        basis_rows = [
            ct.grad_summary[mk][key].float()
            for ct in half_a
            if mk in ct.grad_summary
            for key in ("c_target_eigvecs", "c_healthy_eigvecs")
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
            _, ev_c = torch.linalg.eigh(C_t_sub)
            cands.append(("causal", (Q_A @ ev_c[:, -1]).contiguous().float()))
            for beta in betas:
                _, ev_p = torch.linalg.eigh(C_t_sub + float(beta) * C_h_sub)
                cands.append((f"pareto_b{beta:g}", (Q_A @ ev_p[:, -1]).contiguous().float()))
        else:
            _, ev_c = torch.linalg.eigh(C_t_A)
            cands.append(("causal", ev_c[:, -1].contiguous().float()))

        scored_cands: list[tuple[float, float, float, str, torch.Tensor]] = []
        for kind_lbl, u in cands:
            u = u / u.norm().clamp_min(1e-12)
            supp_b = float((u @ C_t_B @ u).item())
            dnll_b = float(-(u @ C_h_B @ u).item())
            pareto_b = supp_b / max(abs(dnll_b), 0.01)
            scored_cands.append((pareto_b, supp_b, -abs(dnll_b), kind_lbl, u))

        best_pareto, best_supp, neg_dnll_abs, best_kind, best_u = max(scored_cands, key=lambda x: (x[0], x[1], x[2]))
        best_dnll = float(-(best_u @ C_h_B @ best_u).item())
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
) -> list[dict[str, Any]]:
    """Purely offline first-order screening over ``CompactTrace(..., capture_grad=True, grad_mode='compressed')``.

    Requires zero model forward or backward passes; aggregates per-token causal sensitivities
    weighted identically to ``gradient_screen_candidates``.
    """
    items = list(candidates.items()) if isinstance(candidates, dict) else list(candidates)
    if not items:
        return []
    if not traces:
        raise ValueError("screen_patches_from_traces requires at least one CompactTrace")
    for idx_t, ct in enumerate(traces):
        if ct.grad_mode != "compressed":
            raise ValueError(
                f"Trace #{idx_t} has grad_mode={ct.grad_mode!r}; capture with capture_grad=True, grad_mode='compressed'"
            )

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
        for e in patch.edits:
            u = e.u.detach().float().cpu()
            uw = u @ w_cpu[e.module]
            if e.idx is not None:
                m = torch.zeros_like(uw)
                m[e.idx.long()] = 1.0
                uw = uw * m
            for ti in range(len(traces)):
                kind_in, h_obj = cached_inputs[(ti, e.module)]
                if kind_in == "mlp":
                    idx_m, x_m = h_obj
                    h_dot_uw = (x_m * uw[idx_m]).sum(dim=1)
                else:
                    h_dot_uw = h_obj @ uw

                g_t = cached_gt[(ti, e.module)]
                if g_t is not None and n_t_list[ti] > 0:
                    wt = n_t_list[ti] / tot_t
                    grad_supp += wt * float(e.alpha) * float(((g_t @ u) * h_dot_uw).sum().item())

                g_h = cached_gh[(ti, e.module)]
                if g_h is not None and n_h_list[ti] > 0:
                    wh = n_h_list[ti] / tot_h
                    grad_dnll += wh * (-float(e.alpha)) * float(((g_h @ u) * h_dot_uw).sum().item())

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
) -> list[dict[str, Any]]:
    """Fast 1-forward-pass proxy screen ranking candidate `MoIPatch`es by failure suppression vs. healthy drift (`delta_nll` & `healthy_kl`).

    Evaluates each patch in teacher-forced mode (zero token generation) so bad `(k, n, alpha)` configs
    can be pruned in seconds before running expensive long-horizon generation.
    Computes both `delta_healthy_nll` and exact full-distribution `healthy_kl = KL(p_base || p_patched)`
    on healthy sequences (zero extra forward passes).
    Pass `prefilter_top_k` to use 1-backward `gradient_screen_candidates` as Stage 1 when screening large
    candidate sets, narrowing to the top-K candidates before running forward evaluation.
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
