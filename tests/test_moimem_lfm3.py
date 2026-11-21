from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn

from wsi.moimem import MemoryBank, _sha256, find_layers, layer_down_proj


class _LFM3Layer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()

        class _FF(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w1 = nn.Linear(d_model, d_ff, bias=False)
                self.w3 = nn.Linear(d_model, d_ff, bias=False)
                self.w2 = nn.Linear(d_ff, d_model, bias=False)

        self.feed_forward = _FF()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.feed_forward.w2(h)


class _MockLFM3(nn.Module):
    def __init__(self, n_layers: int = 3, d_model: int = 8, d_ff: int = 16, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_LFM3Layer(d_model, d_ff) for _ in range(n_layers)]).to(dtype=dtype)


class _LlamaLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        self.mlp = _MLP()


class _GPT2Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c_proj = nn.Linear(d_ff, d_model, bias=False)

        self.mlp = _MLP()


def test_layer_down_proj_variants() -> None:
    lfm_blk = _LFM3Layer(8, 16)
    assert layer_down_proj(lfm_blk) is lfm_blk.feed_forward.w2

    llama_blk = _LlamaLayer(8, 16)
    assert layer_down_proj(llama_blk) is llama_blk.mlp.down_proj

    gpt_blk = _GPT2Block(8, 16)
    assert layer_down_proj(gpt_blk) is gpt_blk.mlp.c_proj

    with pytest.raises(AttributeError):
        layer_down_proj(nn.Linear(4, 4))


def test_moimem_attach_gate_bake_and_roundtrip_on_lfm3_bf16(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _MockLFM3(n_layers=3, d_model=8, d_ff=16, dtype=torch.bfloat16)
    layer_idx = 1
    w2 = layer_down_proj(find_layers(model)[layer_idx])

    k = torch.zeros(16, dtype=torch.float32)
    k[0] = 1.0
    c = k.clone().unsqueeze(0)
    r = torch.linspace(0.5, 1.2, 8, dtype=torch.float32).unsqueeze(0)
    ck = torch.tensor([1.0], dtype=torch.float32)

    bank = MemoryBank(
        layer=layer_idx,
        c=c,
        r=r,
        ck=ck,
        thr=0.5,
        names=["Alice"],
        meta={
            "method": "rome",
            "down_proj_sha256": _sha256(w2.weight),
            "down_proj_dtype": "bfloat16",
        },
    )

    # Save / load round-trip
    out_path = bank.save(tmp_path / "lfm3_test.moimem")
    loaded = MemoryBank.load(out_path)
    assert loaded.layer == layer_idx
    assert loaded.names == ["Alice"]
    assert torch.allclose(loaded.c, c)
    assert torch.allclose(loaded.r, r)

    # Gated & ungated forward pass + off() restore
    x_trig = torch.zeros(1, 1, 16, dtype=torch.bfloat16)
    x_trig[0, 0, 0] = 1.0
    x_orth = torch.zeros(1, 1, 16, dtype=torch.bfloat16)
    x_orth[0, 0, 1] = 1.0

    with torch.no_grad():
        base_trig = model.layers[layer_idx](x_trig)
        base_orth = model.layers[layer_idx](x_orth)

    h = loaded.attach(model).on(gated=True)
    try:
        with torch.no_grad():
            got_orth = model.layers[layer_idx](x_orth)
            assert h.last_fired is not None and bool(h.last_fired[0]) is False
            assert torch.equal(got_orth, base_orth)

            got_trig = model.layers[layer_idx](x_trig)
            assert bool(h.last_fired[0]) is True
            assert (got_trig.float() - base_trig.float() - r.unsqueeze(0)).abs().max().item() < 0.05

        h.on(gated=False)
        w_orig = w2.weight.detach().clone()
        w_baked = loaded.baked_weight(w_orig)
        with torch.no_grad():
            got_hook = model.layers[layer_idx](x_trig)
            got_baked = torch.nn.functional.linear(x_trig, w_baked)
            assert (got_hook.float() - got_baked.float()).abs().max().item() < 0.05
    finally:
        h.off()

    with torch.no_grad():
        assert torch.equal(model.layers[layer_idx](x_trig), base_trig)

    # Fingerprint mismatch detection
    w2.weight.data[0, 0] += 1.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        loaded.attach(model)
