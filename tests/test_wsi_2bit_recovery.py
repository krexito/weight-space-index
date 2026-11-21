"""CPU unit tests for `scripts/experiment_wsi_2bit_recovery.py`."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_wsi_2bit_recovery as q2  # noqa: E402


def test_quantize_dequantize_2bit_group_shapes_and_bounds() -> None:
    torch.manual_seed(0)
    w = torch.randn(32, 96, dtype=torch.float32)
    packed, sc, zr, shp = q2.quantize_2bit_group(w, group_size=64)
    assert shp == (32, 96)
    rec = q2.dequantize_2bit_group(packed, sc, zr, shp, group_size=64, dtype=torch.float32)
    assert rec.shape == w.shape
    rel_err = float((w - rec).norm().item() / w.norm().item())
    assert 0.1 < rel_err < 0.6


def test_wsi_2bit_recovery_smoke(tmp_path) -> None:
    rep = q2.run_smoke(out_dir=tmp_path)
    assert rep["status"] == "ok"
    assert rep["smoke_passed"] is True
    assert (tmp_path / "smoke_2bit.json").exists()
    arms = rep["arms"]
    assert arms["wsi_fisher_3pct"]["nll"]["mean_nll"] < arms["q2_pure"]["nll"]["mean_nll"]
    assert arms["wsi_fisher_global_3pct"]["nll"]["mean_nll"] < arms["q2_pure"]["nll"]["mean_nll"]
