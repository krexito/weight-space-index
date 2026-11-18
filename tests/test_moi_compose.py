"""Weight-level composition of two .moipatch files (scripts/moi_compose.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _unit(n, seed):
    v = torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return v / v.norm()


def _patches():
    from wsi.moipatch import Edit, MoIPatch

    loop = MoIPatch(edits=[Edit("layers.1.mlp.down_proj", _unit(8, 1), 6.0, torch.arange(4)),
                           Edit("layers.2.mlp.down_proj", _unit(8, 2), 6.0, None)])
    syc = MoIPatch(edits=[Edit("layers.2.mlp.down_proj", _unit(8, 3), 8.0, None),
                          Edit("layers.3.self_attn.o_proj", _unit(8, 4), 8.0, None)])
    w = {m: torch.randn(8, 12, generator=torch.Generator().manual_seed(i)).to(torch.bfloat16)
         for i, m in enumerate(["layers.1.mlp.down_proj", "layers.2.mlp.down_proj", "layers.3.self_attn.o_proj"])}
    return loop, syc, w


def test_order_only_matters_on_shared_matrices():
    import moi_compose as mc

    loop, syc, w = _patches()
    assert mc.shared_modules(loop, syc) == ["layers.2.mlp.down_proj"]
    a = mc.apply_sequence(w, mc.compose_edits(loop, syc, "seq"))
    b = mc.apply_sequence(w, mc.compose_edits(loop, syc, "seq_rev"))
    assert torch.equal(a["layers.1.mlp.down_proj"], b["layers.1.mlp.down_proj"])
    assert torch.equal(a["layers.3.self_attn.o_proj"], b["layers.3.self_attn.o_proj"])
    assert not torch.equal(a["layers.2.mlp.down_proj"], b["layers.2.mlp.down_proj"])
    assert all(v.dtype == torch.bfloat16 for v in a.values())


def test_orth_removes_the_loop_direction_on_shared_matrices_only():
    import moi_compose as mc

    loop, syc, _ = _patches()
    e = mc.compose_edits(loop, syc, "orth")
    assert [x.module for x in e] == [*(x.module for x in loop.edits), *(x.module for x in syc.edits)]
    assert abs(float(e[2].u @ loop.edits[1].u)) < 1e-12 and abs(float(e[2].u.norm()) - 1) < 1e-12
    assert torch.equal(e[3].u, syc.edits[1].u)  # not shared: unchanged
