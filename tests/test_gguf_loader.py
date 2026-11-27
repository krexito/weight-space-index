"""CPU unit tests for wsi/gguf_loader.py (synthetic GGUF v3 header + Q8_0/Q4_0/F16/F32 load)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from wsi.gguf_loader import (
    dequantize_gguf_bytes,
    inspect_gguf_header,
    load_gguf_into_model,
    map_gguf_name_to_hf,
)


def _pack_gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _quantize_q8_0(arr: np.ndarray) -> bytes:
    flat = arr.astype(np.float32).reshape(-1)
    assert flat.size % 32 == 0
    blocks = flat.reshape(-1, 32)
    out = bytearray()
    for blk in blocks:
        amax = float(np.max(np.abs(blk)))
        scale = np.float16(amax / 127.0 if amax > 0 else 1.0)
        s_f32 = float(scale)
        q = np.clip(np.round(blk / s_f32), -127, 127).astype(np.int8)
        out.extend(scale.tobytes())
        out.extend(q.tobytes())
    return bytes(out)


def _write_synthetic_gguf(path: Path, arch: str, tensors: list[tuple[str, np.ndarray, int]]) -> None:
    """Write a valid GGUF v3 file with F32 (0), F16 (1), or Q8_0 (8) tensors."""
    align = 32
    kv_entries = [
        ("general.architecture", 8, _pack_gguf_str(arch)),
        ("general.alignment", 4, struct.pack("<I", align)),
        (f"{arch}.block_count", 4, struct.pack("<I", 2)),
    ]

    header = bytearray()
    header.extend(b"GGUF")
    header.extend(struct.pack("<IQQ", 3, len(tensors), len(kv_entries)))
    for k, vtype, raw_v in kv_entries:
        header.extend(_pack_gguf_str(k))
        header.extend(struct.pack("<I", vtype))
        header.extend(raw_v)

    payloads: list[bytes] = []
    cur_off = 0
    for name, arr, gtype in tensors:
        if gtype == 0:
            raw = arr.astype(np.float32).tobytes()
        elif gtype == 1:
            raw = arr.astype(np.float16).tobytes()
        elif gtype == 8:
            raw = _quantize_q8_0(arr)
        else:
            raise ValueError(gtype)
        # Pad payload offset to alignment
        pad = (-cur_off) % align
        if pad:
            payloads.append(b"\x00" * pad)
            cur_off += pad
        # GGUF stores dims reversed vs PyTorch shape
        dims = tuple(reversed(arr.shape))
        header.extend(_pack_gguf_str(name))
        header.extend(struct.pack("<I", len(dims)))
        header.extend(struct.pack(f"<{len(dims)}Q", *dims))
        header.extend(struct.pack("<IQ", gtype, cur_off))
        payloads.append(raw)
        cur_off += len(raw)

    hdr_pad = (-len(header)) % align
    if hdr_pad:
        header.extend(b"\x00" * hdr_pad)

    with open(path, "wb") as f:
        f.write(header)
        for p in payloads:
            f.write(p)


def test_q8_0_and_q4_0_dequantize_roundtrip() -> None:
    rng = np.random.default_rng(42)
    w = rng.normal(0.0, 0.5, size=(8, 32)).astype(np.float32)
    q8 = np.frombuffer(_quantize_q8_0(w), dtype=np.uint8)
    rec = dequantize_gguf_bytes(q8, 8, (8, 32), dtype=torch.float32).numpy()
    assert np.max(np.abs(rec - w)) < 0.02

    # Q4_0 synthetic block (scale=1.0, nibbles=9 -> value +1.0)
    scale = np.float16(1.0).tobytes()
    nibbles = bytes([0x99] * 16)
    q4 = np.frombuffer(scale + nibbles, dtype=np.uint8)
    rec4 = dequantize_gguf_bytes(q4, 2, (1, 32), dtype=torch.float32).numpy()
    assert np.allclose(rec4, 1.0)

    # Q2_0 (type 42) synthetic block: 64 elems = fp16 scale(2.0) + 16 bytes of 0b11_10_01_00 = 0xE4
    # 00->-1, 01->0, 10->+1, 11->+2 multiplied by scale 2.0 => [-2, 0, 2, 4] repeated 16x
    q2 = np.frombuffer(np.float16(2.0).tobytes() + bytes([0xE4] * 16), dtype=np.uint8)
    rec2 = dequantize_gguf_bytes(q2, 42, (1, 64), dtype=torch.float32).numpy()
    expected2 = np.tile(np.array([-2.0, 0.0, 2.0, 4.0], dtype=np.float32), 16).reshape(1, 64)
    assert np.allclose(rec2, expected2)


def test_synthetic_gguf_load_into_qwen2(tmp_path: Path) -> None:
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    ref = Qwen2ForCausalLM(cfg).eval()
    sd = ref.state_dict()

    # Export ref weights to synthetic GGUF with mixed F32 / F16 / Q8_0
    gguf_tensors: list[tuple[str, np.ndarray, int]] = [
        ("token_embd.weight", sd["model.embed_tokens.weight"].numpy(), 1),
        ("output.weight", sd["lm_head.weight"].numpy(), 8),
        ("output_norm.weight", sd["model.norm.weight"].numpy(), 0),
    ]
    for i in range(2):
        gguf_tensors.extend([
            (f"blk.{i}.attn_norm.weight", sd[f"model.layers.{i}.input_layernorm.weight"].numpy(), 0),
            (f"blk.{i}.ffn_norm.weight", sd[f"model.layers.{i}.post_attention_layernorm.weight"].numpy(), 0),
            (f"blk.{i}.attn_q.weight", sd[f"model.layers.{i}.self_attn.q_proj.weight"].numpy(), 8),
            (f"blk.{i}.attn_q.bias", sd[f"model.layers.{i}.self_attn.q_proj.bias"].numpy(), 0),
            (f"blk.{i}.attn_k.weight", sd[f"model.layers.{i}.self_attn.k_proj.weight"].numpy(), 8),
            (f"blk.{i}.attn_k.bias", sd[f"model.layers.{i}.self_attn.k_proj.bias"].numpy(), 0),
            (f"blk.{i}.attn_v.weight", sd[f"model.layers.{i}.self_attn.v_proj.weight"].numpy(), 8),
            (f"blk.{i}.attn_v.bias", sd[f"model.layers.{i}.self_attn.v_proj.bias"].numpy(), 0),
            (f"blk.{i}.attn_output.weight", sd[f"model.layers.{i}.self_attn.o_proj.weight"].numpy(), 8),
            (f"blk.{i}.ffn_gate.weight", sd[f"model.layers.{i}.mlp.gate_proj.weight"].numpy(), 8),
            (f"blk.{i}.ffn_up.weight", sd[f"model.layers.{i}.mlp.up_proj.weight"].numpy(), 8),
            (f"blk.{i}.ffn_down.weight", sd[f"model.layers.{i}.mlp.down_proj.weight"].numpy(), 8),
        ])

    gguf_path = tmp_path / "tiny_qwen2_q8_0.gguf"
    _write_synthetic_gguf(gguf_path, "qwen2", gguf_tensors)

    hdr = inspect_gguf_header(gguf_path)
    assert hdr.architecture == "qwen2"
    assert hdr.n_tensors == len(gguf_tensors)
    assert hdr.quant_type_counts["Q8_0"] == 15
    assert 8.0 < hdr.avg_bpw < 16.0
    assert not hdr.custom_codecs

    loaded = Qwen2ForCausalLM(cfg).eval()
    report = load_gguf_into_model(gguf_path, loaded, dtype=torch.float32, device="cpu", strict=True)
    assert report["n_missing"] == 0
    assert report["n_unmapped"] == 0

    # Verify forward logits match reference closely (Q8_0 quantization error only)
    ids = torch.tensor([[1, 5, 12, 33]], dtype=torch.long)
    with torch.no_grad():
        logits_ref = ref(ids).logits
        logits_gguf = loaded(ids).logits
    assert torch.allclose(logits_ref, logits_gguf, atol=0.05, rtol=0.05)


def test_map_gguf_name_deltanet_hybrid() -> None:
    hf_keys = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
    }
    assert map_gguf_name_to_hf("token_embd.weight", hf_keys) == "model.language_model.embed_tokens.weight"
    assert map_gguf_name_to_hf("blk.0.ssm_a", hf_keys) == "model.language_model.layers.0.linear_attn.A_log"
    assert map_gguf_name_to_hf("blk.0.attn_qkv.weight", hf_keys) == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    assert map_gguf_name_to_hf("blk.3.attn_q.weight", hf_keys) == "model.language_model.layers.3.self_attn.q_proj.weight"
