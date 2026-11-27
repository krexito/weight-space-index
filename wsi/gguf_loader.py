"""GGUF inspection and dequantizing loader for WSI X-Ray & Tome-2 Quantization Deltas.

Supports two paths so WSI can scan `.gguf` checkpoints with full PyTorch hooks & Fisher:
1. Native `transformers` GGUF loading (`AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`)
   for standard architectures supported by HuggingFace `transformers`.
2. Universal Base-Skeleton loading (`load_gguf_into_base_model`): instantiates the empty
   PyTorch architecture from `base_model_id`'s `config.json`, dequantizes every GGUF tensor
   (preserving exact quantized weights) via `gguf.quants.dequantize` (with pure-NumPy fallback
   for F32/F16/BF16/Q8_0/Q4_0), and maps GGUF tensor names (`blk.N.*`) to HuggingFace keys.

Also provides zero-dependency `inspect_gguf_header(path_or_url)` that parses GGUF v2/v3
headers locally or via HTTP Range requests (without downloading multi-GB files).
"""
from __future__ import annotations

import os
import re
import struct
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# Standard GGML quantization type IDs -> name and approximate bits-per-weight
GGML_TYPE_INFO: dict[int, tuple[str, float]] = {
    0: ("F32", 32.0),
    1: ("F16", 16.0),
    2: ("Q4_0", 4.5),
    3: ("Q4_1", 5.0),
    6: ("Q5_0", 5.5),
    7: ("Q5_1", 6.0),
    8: ("Q8_0", 8.5),
    9: ("Q8_1", 9.0),
    10: ("Q2_K", 2.63),
    11: ("Q3_K", 3.44),
    12: ("Q4_K", 4.5),
    13: ("Q5_K", 5.5),
    14: ("Q6_K", 6.56),
    15: ("Q8_K", 9.13),
    16: ("IQ2_XXS", 2.06),
    17: ("IQ2_XS", 2.31),
    18: ("IQ3_XXS", 3.06),
    19: ("IQ1_S", 1.56),
    20: ("IQ4_NL", 4.5),
    21: ("IQ3_S", 3.44),
    22: ("IQ2_S", 2.56),
    23: ("IQ4_XS", 4.25),
    24: ("I8", 8.0),
    25: ("I16", 16.0),
    26: ("I32", 32.0),
    27: ("I64", 64.0),
    28: ("F64", 64.0),
    29: ("IQ1_M", 1.75),
    30: ("BF16", 16.0),
    34: ("TQ1_0", 1.69),
    35: ("TQ2_0", 2.06),
    39: ("MXFP4", 4.25),
    40: ("NVFP4", 4.5),
    41: ("Q1_0", 1.125),
    42: ("Q2_0", 2.25),
    43: ("F8_E4M3", 8.0),
}

_GGUF_SCALAR_SIZES: dict[int, int] = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


@dataclass
class GGUFTensorInfo:
    name: str
    dims: tuple[int, ...]  # GGUF order (ne0, ne1, ...) -> PyTorch shape is dims[::-1]
    ggml_type: int
    type_name: str
    offset: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= int(d)
        return n

    @property
    def torch_shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.dims))


@dataclass
class GGUFHeaderSummary:
    version: int
    n_tensors: int
    n_kv: int
    architecture: str | None
    metadata: dict[str, Any]
    tensors: list[GGUFTensorInfo]
    quant_type_counts: dict[str, int]
    avg_bpw: float
    data_offset: int
    custom_codecs: list[str] = field(default_factory=list)
    unknown_type_ids: list[int] = field(default_factory=list)
    truncated: bool = False

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "architecture": self.architecture,
            "n_tensors": self.n_tensors,
            "quant_type_counts": dict(self.quant_type_counts),
            "avg_bpw": round(self.avg_bpw, 3),
            "custom_codecs": list(self.custom_codecs),
            "unknown_type_ids": list(self.unknown_type_ids),
        }


def _read_gguf_str(buf: bytes, off: int) -> tuple[str, int]:
    n = struct.unpack_from("<Q", buf, off)[0]
    off += 8
    return buf[off : off + n].decode("utf-8", "replace"), off + n


def _skip_gguf_val(buf: bytes, off: int, vtype: int) -> int:
    if vtype in _GGUF_SCALAR_SIZES:
        return off + _GGUF_SCALAR_SIZES[vtype]
    if vtype == 8:  # STRING
        _, off = _read_gguf_str(buf, off)
        return off
    if vtype == 9:  # ARRAY
        elem_type, count = struct.unpack_from("<IQ", buf, off)
        off += 12
        if elem_type in _GGUF_SCALAR_SIZES:
            return off + _GGUF_SCALAR_SIZES[elem_type] * count
        for _ in range(count):
            off = _skip_gguf_val(buf, off, elem_type)
        return off
    raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def _read_gguf_scalar_or_str(buf: bytes, off: int, vtype: int) -> tuple[Any, int]:
    if vtype == 8:
        return _read_gguf_str(buf, off)
    if vtype == 4:
        return struct.unpack_from("<I", buf, off)[0], off + 4
    if vtype == 5:
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if vtype == 6:
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if vtype == 7:
        return bool(buf[off]), off + 1
    if vtype == 10:
        return struct.unpack_from("<Q", buf, off)[0], off + 8
    return None, _skip_gguf_val(buf, off, vtype)


def inspect_gguf_header(
    path_or_url: str | os.PathLike[str],
    *,
    max_header_bytes: int = 16 * 1024 * 1024,
) -> GGUFHeaderSummary:
    """Parse a GGUF v2/v3 header locally or via HTTP Range without reading tensor payloads."""
    s = str(path_or_url)
    if s.startswith("http://") or s.startswith("https://"):
        req = urllib.request.Request(s, headers={"Range": f"bytes=0-{max_header_bytes - 1}"})
        buf = urllib.request.urlopen(req).read()
    else:
        with open(s, "rb") as f:
            buf = f.read(max_header_bytes)

    if len(buf) < 24 or buf[:4] != b"GGUF":
        raise ValueError(f"Not a valid GGUF file/URL: magic={buf[:4]!r}")

    version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    off = 24
    keep_keys = (
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "general.alignment",
    )
    meta: dict[str, Any] = {}
    for _ in range(n_kv):
        k, off = _read_gguf_str(buf, off)
        vtype = struct.unpack_from("<I", buf, off)[0]
        off += 4
        if (
            k in keep_keys
            or k.endswith(".block_count")
            or k.endswith(".embedding_length")
            or k.endswith(".expert_count")
            or k.endswith(".expert_used_count")
            or k.endswith(".context_length")
        ):
            val, off = _read_gguf_scalar_or_str(buf, off, vtype)
            meta[k] = val
        else:
            off = _skip_gguf_val(buf, off, vtype)

    tensors: list[GGUFTensorInfo] = []
    qcounts: Counter[str] = Counter()
    unknown_ids: set[int] = set()
    custom_codecs: set[str] = set()
    total_elems = 0
    total_bits = 0.0
    truncated = False

    for _ in range(n_tensors):
        if off + 32 > len(buf):
            truncated = True
            break
        name, off = _read_gguf_str(buf, off)
        ndims = struct.unpack_from("<I", buf, off)[0]
        off += 4
        dims = struct.unpack_from(f"<{ndims}Q", buf, off)
        off += 8 * ndims
        t_type, t_off = struct.unpack_from("<IQ", buf, off)
        off += 12
        if t_type in GGML_TYPE_INFO:
            tname, bpw = GGML_TYPE_INFO[t_type]
        else:
            tname, bpw = f"UNKNOWN_{t_type}", 4.0
            unknown_ids.add(int(t_type))
        if ".escha_" in name or name.startswith("escha_"):
            custom_codecs.add("escha")
        info = GGUFTensorInfo(
            name=name,
            dims=tuple(int(d) for d in dims),
            ggml_type=int(t_type),
            type_name=tname,
            offset=int(t_off),
        )
        tensors.append(info)
        qcounts[tname] += 1
        ne = info.n_elements
        total_elems += ne
        total_bits += ne * bpw

    align = int(meta.get("general.alignment", 32) or 32)
    data_offset = ((off + align - 1) // align) * align
    avg_bpw = (total_bits / total_elems) if total_elems > 0 else 0.0

    return GGUFHeaderSummary(
        version=int(version),
        n_tensors=int(n_tensors),
        n_kv=int(n_kv),
        architecture=meta.get("general.architecture"),
        metadata=meta,
        tensors=tensors,
        quant_type_counts=dict(qcounts),
        avg_bpw=avg_bpw,
        data_offset=data_offset,
        custom_codecs=sorted(custom_codecs),
        unknown_type_ids=sorted(unknown_ids),
        truncated=truncated,
    )


def dequantize_gguf_bytes(
    raw: np.ndarray,
    ggml_type: int,
    torch_shape: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize raw GGUF tensor bytes into a PyTorch tensor of `torch_shape`."""
    # Fast exact paths for unquantized float types
    if ggml_type == 0:  # F32
        arr = np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(torch_shape)
        return torch.from_numpy(arr.copy()).to(dtype=dtype)
    if ggml_type == 1:  # F16
        arr = np.frombuffer(raw.tobytes(), dtype=np.float16).reshape(torch_shape)
        return torch.from_numpy(arr.copy()).to(dtype=dtype)
    if ggml_type == 30:  # BF16
        u16 = np.frombuffer(raw.tobytes(), dtype=np.uint16).reshape(torch_shape)
        return torch.from_numpy(u16.copy()).view(torch.bfloat16).to(dtype=dtype)

    # Native pure-NumPy paths for brand-new GGML types (Q2_0=42, Q1_0=41) not yet in PyPI `gguf`
    if ggml_type == 42:  # Q2_0: QK2_0=64 elements per block = 2-byte fp16 scale + 16 packed uint8 (2-bit: 00=-1,01=0,10=+1,11=+2)
        b = raw.view(np.uint8)
        n_blocks = b.size // 18
        blocks = b.reshape(n_blocks, 18)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)  # (n_blocks, 1)
        qs = blocks[:, 2:]  # (n_blocks, 16)
        q0 = (qs & 0x03).astype(np.int8) - 1
        q1 = ((qs >> 2) & 0x03).astype(np.int8) - 1
        q2 = ((qs >> 4) & 0x03).astype(np.int8) - 1
        q3 = ((qs >> 6) & 0x03).astype(np.int8) - 1
        # Interleave j = 4*byte_index + {0,1,2,3}
        unpacked = np.stack([q0, q1, q2, q3], axis=-1).reshape(n_blocks, 64).astype(np.float32)
        arr = (scales * unpacked).reshape(torch_shape)
        return torch.from_numpy(arr).to(dtype=dtype)
    if ggml_type == 41:  # Q1_0: QK1_0=128 elements per block = 2-byte fp16 scale + 16 packed uint8 (1-bit: 0=-1, 1=+1)
        b = raw.view(np.uint8)
        n_blocks = b.size // 18
        blocks = b.reshape(n_blocks, 18)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        bits = np.unpackbits(blocks[:, 2:], axis=1, bitorder="little").astype(np.float32)
        arr = (scales * (bits * 2.0 - 1.0)).reshape(torch_shape)
        return torch.from_numpy(arr).to(dtype=dtype)

    # Prefer official `gguf.quants.dequantize` when installed (supports Q2_K..Q8_0, IQ1..IQ4, MXFP4, TQ*)
    try:
        import gguf  # type: ignore

        qtype = gguf.GGMLQuantizationType(ggml_type)
        arr = gguf.quants.dequantize(raw, qtype)
        arr = np.asarray(arr, dtype=np.float32).reshape(torch_shape)
        return torch.from_numpy(arr.copy()).to(dtype=dtype)
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(
            f"gguf.quants.dequantize failed for ggml_type={ggml_type} "
            f"({GGML_TYPE_INFO.get(ggml_type, ('UNKNOWN',))[0]}): {exc}"
        ) from exc

    # Pure-NumPy fallback for Q8_0 (type 8) and Q4_0 (type 2) so unit tests run without `gguf`
    if ggml_type == 8:  # Q8_0: 32 elements per block = 2-byte fp16 scale + 32 int8
        b = raw.view(np.uint8)
        n_blocks = b.size // 34
        blocks = b.reshape(n_blocks, 34)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)  # (n_blocks, 1)
        qs = blocks[:, 2:].view(np.int8).astype(np.float32)  # (n_blocks, 32)
        arr = (scales * qs).reshape(torch_shape)
        return torch.from_numpy(arr).to(dtype=dtype)
    if ggml_type == 2:  # Q4_0: 32 elements per block = 2-byte fp16 scale + 16 packed uint8
        b = raw.view(np.uint8)
        n_blocks = b.size // 18
        blocks = b.reshape(n_blocks, 18)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        packed = blocks[:, 2:]
        lo = (packed & 0x0F).astype(np.int8) - 8
        hi = ((packed >> 4) & 0x0F).astype(np.int8) - 8
        qs = np.concatenate([lo, hi], axis=1).astype(np.float32)
        arr = (scales * qs).reshape(torch_shape)
        return torch.from_numpy(arr).to(dtype=dtype)

    tname = GGML_TYPE_INFO.get(ggml_type, (f"UNKNOWN_{ggml_type}",))[0]
    raise ImportError(
        f"Dequantizing GGML type {tname} ({ggml_type}) requires the `gguf` package (`pip install gguf`)."
    )


_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# Standard GGUF suffix -> candidate HF submodule suffixes (checked against model state_dict keys)
_GGUF_BLOCK_SUFFIX_CANDIDATES: dict[str, tuple[str, ...]] = {
    "attn_norm.weight": ("input_layernorm.weight", "operator_norm.weight", "attn_norm.weight"),
    "attn_norm.bias": ("input_layernorm.bias",),
    "ffn_norm.weight": ("post_attention_layernorm.weight", "ffn_norm.weight", "mlp_norm.weight"),
    "ffn_norm.bias": ("post_attention_layernorm.bias",),
    "post_attention_norm.weight": ("post_attention_layernorm.weight",),
    "attn_q.weight": ("self_attn.q_proj.weight", "self_attn.q_proj.linear.weight", "attn.q_proj.weight", "attn.wq_du.weight"),
    "attn_q.bias": ("self_attn.q_proj.bias", "attn.q_proj.bias"),
    "attn_k.weight": ("self_attn.k_proj.weight", "self_attn.k_proj.linear.weight", "attn.k_proj.weight", "attn.wk_dv.weight"),
    "attn_k.bias": ("self_attn.k_proj.bias", "attn.k_proj.bias"),
    "attn_v.weight": ("self_attn.v_proj.weight", "self_attn.v_proj.linear.weight", "attn.v_proj.weight", "attn.wv_dv.weight"),
    "attn_v.bias": ("self_attn.v_proj.bias", "attn.v_proj.bias"),
    "attn_output.weight": ("self_attn.o_proj.weight", "self_attn.o_proj.linear.weight", "attn.o_proj.weight", "attn.wo_ud.weight"),
    "attn_output.bias": ("self_attn.o_proj.bias", "attn.o_proj.bias"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "self_attn.q_layernorm.weight"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "self_attn.k_layernorm.weight"),
    "ffn_gate.weight": ("mlp.gate_proj.weight", "feed_forward.w1.weight"),
    "ffn_up.weight": ("mlp.up_proj.weight", "feed_forward.w3.weight"),
    "ffn_down.weight": ("mlp.down_proj.weight", "feed_forward.w2.weight"),
    "ffn_gate_inp.weight": ("mlp.gate.weight", "block_sparse_moe.gate.weight"),
    # Qwen3.5 / 3.8 Gated DeltaNet hybrid tensors
    "ssm_a": ("linear_attn.A_log",),
    "ssm_dt.bias": ("linear_attn.dt_bias",),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight",),
    "ssm_norm.weight": ("linear_attn.norm.weight",),
    "ssm_out.weight": ("linear_attn.out_proj.weight",),
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight",),
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight",),
    "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", "self_attn.qkv_proj.weight"),
    "attn_gate.weight": ("linear_attn.in_proj_z.weight", "self_attn.o_proj.gate_proj.weight", "attn.gate_attn.weight"),
    # Inkling-2-Mini looped MoE tensors (`transformers` Inkling2Loop + raw `model.llm.*`)
    "attn_k_conv.weight": ("self_attn.k_proj.sconv.conv1d.weight", "self_attn.k_proj.sconv.static_weight", "attn.k_sconv.weight"),
    "attn_v_conv.weight": ("self_attn.v_proj.sconv.conv1d.weight", "self_attn.v_proj.sconv.static_weight", "attn.v_sconv.weight"),
    "attn_k_conv_proj.weight": ("self_attn.k_proj.sconv.weight_proj.weight",),
    "attn_v_conv_proj.weight": ("self_attn.v_proj.sconv.weight_proj.weight",),
    "attn_q_inp_norm.weight": ("q_input_layernorm.weight",),
    "attn_k_inp_norm.weight": ("k_input_layernorm.weight",),
    "attn_v_inp_norm.weight": ("v_input_layernorm.weight",),
    "attn_post_conv.weight": ("post_attention_convolution.conv1d.weight", "attn_sconv.weight"),
    "ffn_post_conv.weight": ("post_mlp_convolution.conv1d.weight", "mlp_sconv.weight"),
    "attn_q_clip.weight": ("self_attn.q_proj.norm.scale", "self_attn.q_proj.scale", "attn.q_l2_clip.scale"),
    "attn_rel.weight": ("self_attn.relative_position_bias.linear.weight", "attn.wr_du.weight"),
    "attn_rel_bias.weight": ("self_attn.relative_position_bias.proj", "attn.rel_logits_proj.proj"),
    "attn_rel_clip.weight": ("self_attn.relative_position_bias.norm.scale", "self_attn.relative_position_bias.scale", "attn.rel_l2_clip.scale"),
    "attn_sink_proj.weight": ("self_attn.sink_proj.weight", "attn.attn_sink_proj.weight"),
    "exp_probs_b.bias": ("mlp.gate.e_score_correction_bias", "mlp.gate.bias"),
    "ffn_down_exps.weight": ("mlp.experts.down_proj", "mlp.experts.w2_weight"),
    "ffn_gate_up_exps.weight": ("mlp.experts.gate_up_proj", "mlp.experts.w13_weight"),
    "ffn_res_query.weight": ("mlp_input.query",),
    "attn_res_query.weight": ("attention_input.query",),
    "loop_ffn_ctrl_down.weight": ("mlp_message_gate.down_proj.weight",),
    "loop_ffn_ctrl_forget.weight": ("mlp_message_gate.forget_gate_proj.weight",),
    "loop_ffn_ctrl_gain.weight": ("mlp_message_gate.message_gate_proj.weight",),
    "loop_q_ctrl_down.weight": ("q_message_gate.down_proj.weight",),
    "loop_q_ctrl_forget.weight": ("q_message_gate.forget_gate_proj.weight",),
    "loop_q_ctrl_gain.weight": ("q_message_gate.message_gate_proj.weight",),
    "loop_k_ctrl_down.weight": ("k_message_gate.down_proj.weight",),
    "loop_k_ctrl_forget.weight": ("k_message_gate.forget_gate_proj.weight",),
    "loop_k_ctrl_gain.weight": ("k_message_gate.message_gate_proj.weight",),
    "loop_v_ctrl_down.weight": ("v_message_gate.down_proj.weight",),
    "loop_v_ctrl_forget.weight": ("v_message_gate.forget_gate_proj.weight",),
    "loop_v_ctrl_gain.weight": ("v_message_gate.message_gate_proj.weight",),
    "loop_msg_query.weight": ("message_input.query", "attention_input.query"),
}

_GGUF_GLOBAL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "token_embd.weight": (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "model.llm.embed.weight",
        "transformer.wte.weight",
    ),
    "token_embd_norm.weight": (
        "model.embed_tokens.embed_norm.weight",
        "model.language_model.embed_tokens.embed_norm.weight",
        "model.llm.embed_norm.weight",
    ),
    "output_norm.weight": (
        "model.norm.weight",
        "model.language_model.norm.weight",
        "model.llm.norm.weight",
        "transformer.ln_f.weight",
    ),
    "output.weight": (
        "lm_head.weight",
        "model.llm.unembed.weight",
        "model.embed_tokens.weight",
    ),
    "output_res_query.weight": (
        "model.final_input.query",
        "model.language_model.final_input.query",
        "model.llm.attn_res.output_query",
    ),
}


def map_gguf_name_to_hf(gguf_name: str, hf_keys: set[str]) -> str | None:
    """Map a GGUF tensor name to a matching key in a HuggingFace model's state_dict."""
    if gguf_name in _GGUF_GLOBAL_CANDIDATES:
        for cand in _GGUF_GLOBAL_CANDIDATES[gguf_name]:
            if cand in hf_keys:
                return cand
        return None

    m = _BLK_RE.match(gguf_name)
    if not m:
        return None
    layer_idx, suffix = m.group(1), m.group(2)
    # Inkling-2-Mini attention-residual query vectors
    if suffix == "ffn_res_query.weight":
        cand = f"model.llm.attn_res.mlp_queries.{layer_idx}"
        if cand in hf_keys:
            return cand
    if suffix == "attn_res_query.weight":
        cand = f"model.llm.attn_res.attn_queries.{int(layer_idx) - 1}"
        if cand in hf_keys:
            return cand

    cands = _GGUF_BLOCK_SUFFIX_CANDIDATES.get(suffix, ())
    for prefix in ("model.layers", "model.language_model.layers", "model.llm.layers", "transformer.h"):
        for cand in cands:
            key = f"{prefix}.{layer_idx}.{cand}"
            if key in hf_keys:
                return key
    return None


def _read_gguf_tensor_raw(
    file_obj: Any,
    data_offset: int,
    info: GGUFTensorInfo,
) -> np.ndarray:
    """Read raw bytes of a single tensor from an open GGUF file."""
    block_specs: dict[int, tuple[int, int]] = {
        0: (1, 4),      # F32
        1: (1, 2),      # F16
        2: (32, 18),    # Q4_0
        3: (32, 20),    # Q4_1
        6: (32, 22),    # Q5_0
        7: (32, 24),    # Q5_1
        8: (32, 34),    # Q8_0
        30: (1, 2),     # BF16
        41: (128, 18),  # Q1_0
        42: (64, 18),   # Q2_0
    }
    if info.ggml_type in block_specs:
        bs, nbytes_per_block = block_specs[info.ggml_type]
        nbytes = (info.n_elements // bs) * nbytes_per_block
    else:
        import gguf  # type: ignore

        qtype = gguf.GGMLQuantizationType(info.ggml_type)
        bs, nbytes_per_block = gguf.GGML_QUANT_SIZES[qtype]
        nbytes = (info.n_elements // bs) * nbytes_per_block

    file_obj.seek(data_offset + info.offset)
    buf = file_obj.read(nbytes)
    if len(buf) != nbytes:
        raise IOError(f"Short read for {info.name}: expected {nbytes}, got {len(buf)}")
    return np.frombuffer(buf, dtype=np.uint8)


def load_gguf_into_model(
    gguf_path: str | os.PathLike[str],
    model: nn.Module,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    """Dequantize tensors from `gguf_path` and load them directly into an existing PyTorch `model`."""
    summary = inspect_gguf_header(gguf_path, max_header_bytes=64 * 1024 * 1024)
    if summary.custom_codecs:
        raise NotImplementedError(
            f"GGUF uses custom non-GGML codec(s) {summary.custom_codecs}, which require "
            f"the author's custom dequantizer rather than standard GGML."
        )
    if summary.unknown_type_ids:
        raise NotImplementedError(
            f"GGUF contains unknown GGML quantization type ID(s) {summary.unknown_type_ids}."
        )

    param_map: dict[str, torch.Tensor] = {}
    required_keys: set[str] = set()
    for k, p in model.named_parameters():
        param_map[k] = p
        required_keys.add(k)
    for k, b in model.named_buffers():
        param_map[k] = b
    hf_keys = set(param_map.keys())

    loaded: list[str] = []
    unmapped: list[str] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    # Pre-match tensor names so we can dequantize in parallel across CPU cores
    tasks: list[tuple[GGUFTensorInfo, str]] = []
    for info in summary.tensors:
        hf_key = map_gguf_name_to_hf(info.name, hf_keys)
        if hf_key is None:
            unmapped.append(info.name)
            continue
        tasks.append((info, hf_key))

    def _decode_one(item: tuple[GGUFTensorInfo, str]) -> tuple[str, torch.Tensor]:
        info, hf_key = item
        with open(gguf_path, "rb") as f:
            raw = _read_gguf_tensor_raw(f, summary.data_offset, info)
        tensor = dequantize_gguf_bytes(raw, info.ggml_type, info.torch_shape, dtype=dtype)
        return hf_key, tensor

    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(8, os.cpu_count() or 1, max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for hf_key, tensor in pool.map(_decode_one, tasks):
            target = param_map[hf_key]
            if tuple(tensor.shape) != tuple(target.shape):
                # Vocab-padded embedding / lm_head in GGUF (e.g. 201024 vs 200058)
                if (
                    tensor.ndim == 2
                    and target.ndim == 2
                    and tensor.shape[1] == target.shape[1]
                    and tensor.shape[0] > target.shape[0]
                ):
                    tensor = tensor[: target.shape[0]].contiguous()
                # 2D transpose mismatch (e.g. static_weight [1024, 4] -> [4, 1024], rel_logits_proj [1024, 16] -> [16, 1024])
                elif (
                    tensor.ndim == 2
                    and target.ndim == 2
                    and tensor.shape[0] == target.shape[1]
                    and tensor.shape[1] == target.shape[0]
                ):
                    tensor = tensor.T.contiguous()
                # Some GGUF 1D/conv weights store squeezed dims (e.g. conv1d [D, 4] or [4, D] vs [D, 1, 4])
                elif tensor.numel() == target.numel():
                    if tensor.ndim == 2 and target.ndim == 3 and target.shape[1] == 1:
                        if tensor.shape[0] == target.shape[0]:
                            tensor = tensor.unsqueeze(1).contiguous()
                        else:
                            tensor = tensor.T.unsqueeze(1).contiguous()
                    else:
                        tensor = tensor.reshape(target.shape)
                else:
                    shape_mismatches.append((hf_key, tuple(tensor.shape), tuple(target.shape)))
                    continue
            if target.device.type == "meta":
                parent_path, _, leaf = hf_key.rpartition(".")
                parent = model.get_submodule(parent_path) if parent_path else model
                mat = tensor.to(device=device, dtype=target.dtype)
                if isinstance(target, nn.Parameter):
                    setattr(parent, leaf, nn.Parameter(mat, requires_grad=target.requires_grad))
                else:
                    parent.register_buffer(leaf, mat)
            else:
                with torch.no_grad():
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded.append(hf_key)

    missing = sorted(required_keys - set(loaded))
    if strict and (missing or shape_mismatches):
        raise RuntimeError(
            f"Strict GGUF load failed: missing={len(missing)} ({missing[:5]}), "
            f"shape_mismatches={len(shape_mismatches)}"
        )

    # Materialize any remaining meta tensors (e.g. unused vision tower) so model has zero meta tensors
    for pname, p in list(model.named_parameters()):
        if p.device.type == "meta":
            parent_path, _, leaf = pname.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(
                parent,
                leaf,
                nn.Parameter(torch.zeros(p.shape, dtype=p.dtype, device=device), requires_grad=p.requires_grad),
            )
    for bname, b in list(model.named_buffers()):
        if b.device.type == "meta":
            parent_path, _, leaf = bname.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            parent.register_buffer(leaf, torch.zeros(b.shape, dtype=b.dtype, device=device))

    if device != "cpu" and any(p.device.type == "cpu" for p in model.parameters()):
        model.to(device=device)

    return {
        "gguf": summary.to_compact_dict(),
        "n_loaded": len(loaded),
        "n_missing": len(missing),
        "n_unmapped": len(unmapped),
        "shape_mismatches": shape_mismatches[:10],
        "missing_sample": missing[:10],
        "unmapped_sample": unmapped[:10],
    }


def load_gguf_model(
    repo_or_path: str,
    gguf_file: str | None = None,
    *,
    base_model_id: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    trust_remote_code: bool = True,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load a GGUF model + tokenizer for WSI X-Ray via native `transformers` or base-skeleton fallback.

    Returns `(model, tokenizer, gguf_meta)`.
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    local_gguf: str
    if os.path.isfile(repo_or_path):
        local_gguf = repo_or_path
        repo_dir_or_id = os.path.dirname(os.path.abspath(repo_or_path))
        fname = os.path.basename(repo_or_path)
    else:
        if not gguf_file:
            raise ValueError("`gguf_file` is required when `repo_or_path` is a HuggingFace repo ID.")
        local_gguf = hf_hub_download(repo_id=repo_or_path, filename=gguf_file)
        repo_dir_or_id = repo_or_path
        fname = gguf_file

    header = inspect_gguf_header(local_gguf, max_header_bytes=64 * 1024 * 1024)
    if header.custom_codecs:
        raise NotImplementedError(
            f"{fname} uses custom codec(s) {header.custom_codecs} (not standard GGML)."
        )

    tok_source = base_model_id or repo_dir_or_id
    tok_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if base_model_id is None:
        tok_kwargs["gguf_file"] = fname
    tok = AutoTokenizer.from_pretrained(tok_source, **tok_kwargs)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # Path 1: Try native `transformers` GGUF loader first
    native_err: Exception | None = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo_dir_or_id,
            gguf_file=fname,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        ).eval()
        return model, tok, {"loader": "transformers_native_gguf", "gguf": header.to_compact_dict()}
    except Exception as exc:
        native_err = exc

    # Path 2: Base-skeleton + `load_gguf_into_model` fallback
    if not base_model_id:
        raise RuntimeError(
            f"Native transformers GGUF load failed ({type(native_err).__name__}: {native_err}). "
            f"Pass `base_model_id` (e.g. 'Qwen/Qwen3.8-27B') to use the base-skeleton GGUF loader."
        ) from native_err

    cfg = AutoConfig.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code).to(dtype=dtype)
    report = load_gguf_into_model(
        local_gguf,
        model,
        dtype=dtype,
        device=device_map if device_map != "auto" else "cuda",
    )
    model.eval()
    report["loader"] = "wsi_base_skeleton_gguf"
    report["native_fallback_reason"] = f"{type(native_err).__name__}: {native_err}"
    return model, tok, report
