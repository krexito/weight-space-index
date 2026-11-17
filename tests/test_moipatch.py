"""Unit and integration tests for portable `.moipatch` recipes (`wsi/moipatch.py` and packagers)."""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi import Edit, MoIPatch, PatchHandle  # noqa: E402
from wsi.cli import main as wsi_cli_main  # noqa: E402
from wsi.moipatch import apply_edit  # noqa: E402


class _TinyBlock(torch.nn.Module):
    def __init__(self, hidden: int = 16, inter: int = 32, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(inter, hidden, bias=False, dtype=dtype)
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)


class _TinyLM(torch.nn.Module):
    def __init__(self, n_layers: int = 3, hidden: int = 16, inter: int = 32, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyBlock(hidden, inter, dtype=dtype) for _ in range(n_layers)])


def _save_tiny_checkpoint(model: _TinyLM, ckpt_dir: Path, sharded: bool = False) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    if sharded:
        save_file(sd, str(ckpt_dir / "model.safetensors-00001-of-00001.safetensors"))
        idx = {"metadata": {}, "weight_map": {k: "model.safetensors-00001-of-00001.safetensors" for k in sd}}
        (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps(idx), encoding="utf-8")
    else:
        save_file(sd, str(ckpt_dir / "model.safetensors"))


def test_moipatch_save_load_attach_and_restore() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = _TinyLM(n_layers=3, hidden=16, inter=32, dtype=torch.bfloat16)
        ckpt_dir = tmp / "ckpt"
        _save_tiny_checkpoint(model, ckpt_dir, sharded=True)

        u_mlp = torch.randn(16)
        u_mlp = u_mlp / u_mlp.norm()
        u_attn = torch.randn(16)
        u_attn = u_attn / u_attn.norm()
        cols = torch.tensor([1, 5, 9, 13], dtype=torch.long)
        r_dir = torch.zeros(16)
        r_dir[0] = 1.0

        patch = MoIPatch(
            edits=[
                Edit(module="layers.1.mlp.down_proj", u=u_mlp, alpha=6.0, idx=cols),
                Edit(module="layers.2.self_attn.o_proj", u=u_attn, alpha=4.0, idx=None),
            ],
            router={"layer": 1, "readout": "last_token", "dir": r_dir, "thr": 0.5},
            gates=[{"type": "answer_confidence", "thr": 0.85, "note": "test gate"}],
            meta={"name": "unit-test", "base_model": "tiny", "policy": "on pushback only", "expected": {"acc": 0.9}},
        )
        patch.fingerprint_from_checkpoint(ckpt_dir)
        out_path = patch.save(tmp / "test.moipatch")
        loaded = MoIPatch.load(out_path)
        assert "unit-test [moipatch/1]" in loaded.describe()
        assert len(loaded.edits) == 2 and loaded.edits[0].idx is not None and loaded.edits[1].idx is None

        # Router + confidence gate firing
        hs = [torch.zeros(2, 4, 16) for _ in range(3)]
        hs[1][0, -1, 0] = 1.2
        hs[1][1, -1, 0] = 0.2
        assert loaded.fires(hs).tolist() == [True, False]
        assert loaded.fires(hs, confidence=torch.tensor([0.92, 0.99])).tolist() == [True, False]
        assert loaded.fires(hs, confidence=torch.tensor([0.50, 0.99])).tolist() == [False, False]

        # Attach in float32 (simulating verify --cpu) and verify exact bf16 fingerprint + restore
        model_f32 = _TinyLM(n_layers=3, hidden=16, inter=32, dtype=torch.float32)
        with torch.no_grad():
            for p_dst, p_src in zip(model_f32.parameters(), model.parameters()):
                p_dst.copy_(p_src.float())

        handle = loaded.attach(model_f32)
        assert isinstance(handle, PatchHandle) and not handle.active and handle.fingerprint_ok()
        orig_w1 = model_f32.model.layers[1].mlp.down_proj.weight.detach().clone()
        expected_w1 = apply_edit(orig_w1, u_mlp, cols, 6.0)
        handle.on()
        assert handle.active
        assert torch.equal(model_f32.model.layers[1].mlp.down_proj.weight, expected_w1)
        # Untouched columns stayed identical
        assert torch.equal(model_f32.model.layers[1].mlp.down_proj.weight[:, 0], orig_w1[:, 0])
        handle.off()
        assert not handle.active and handle.check_restored() and handle.fingerprint_ok()

        # Tampered model must be rejected by fingerprint check
        with torch.no_grad():
            model_f32.model.layers[1].mlp.down_proj.weight[0, 0] += 1.0
        try:
            loaded.attach(model_f32)
            raise AssertionError("expected ValueError on fingerprint mismatch")
        except ValueError as err:
            assert "fingerprint mismatch" in str(err)


def test_make_moipatch_sycophancy_and_cli() -> None:
    from scripts.make_moipatch_sycophancy import main as syco_main

    torch.manual_seed(1)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = _TinyLM(n_layers=4, hidden=16, inter=32, dtype=torch.bfloat16)
        ckpt_dir = tmp / "ckpt"
        _save_tiny_checkpoint(model, ckpt_dir, sharded=False)

        work = tmp / "work"
        work.mkdir()
        dg = {
            "ua": [np.random.randn(16).astype(np.float32) for _ in range(4)],
            "u": [np.random.randn(16).astype(np.float32) for _ in range(4)],
            "cands": {"late": [2, 3]},
            "neurons": {2: [0, 3, 7], 3: [1, 4, 9]},
        }
        rt = {"layer": 2, "dir": np.ones(16, dtype=np.float32), "thr": 0.75, "bal_acc_heldout": 0.96}
        res = {
            "model": "tiny/qwen",
            "n": {"test": 100},
            "router": {"layer": 2},
            "gate_study": {
                "chosen_on_dev": "attn:late@8|maxprob",
                "base_test": {"held": 0.15, "accepted": 0.97},
                "rows": {
                    "attn:late@8|maxprob": {
                        "thr": 0.885,
                        "held": 0.401,
                        "accepted": 0.907,
                        "held_unseen_wording": 0.163,
                        "neutral_same": 0.999,
                    }
                },
            },
        }
        (work / "diagnosis_pressure.pkl").write_bytes(pickle.dumps(dg))
        (work / "router.pkl").write_bytes(pickle.dumps(rt))
        (tmp / "results.json").write_text(json.dumps(res), encoding="utf-8")

        out_patch = tmp / "syco.moipatch"
        syco_main(["--work", str(work), "--results", str(tmp / "results.json"),
                   "--model-dir", str(ckpt_dir), "--out", str(out_patch)])
        loaded = MoIPatch.load(out_patch)
        assert loaded.meta["name"] == "anti-sycophancy"
        assert [e.module for e in loaded.edits] == ["layers.2.self_attn.o_proj", "layers.3.self_attn.o_proj"]
        assert loaded.gates[0]["type"] == "answer_confidence" and loaded.gates[0]["thr"] == 0.885

        # CLI `wsi moipatch info` works cleanly
        assert wsi_cli_main(["moipatch", "info", str(out_patch)]) == 0


def test_real_p50c_loop_moipatch_artifact() -> None:
    p = ROOT / "patches" / "qwen35_0.8b_p50c_loop_fix.moipatch"
    if not p.exists():
        return
    patch = MoIPatch.load(p)
    assert patch.meta["name"] == "qwen35-0.8b-loop-fix-p50c"
    assert [e.module for e in patch.edits] == ["layers.15.mlp.down_proj", "layers.19.mlp.down_proj"]
    assert all(e.idx is not None and len(e.idx) == 64 and e.alpha == 6.0 for e in patch.edits)
    assert patch.gates[0]["type"] == "ngram_repetition_frac"
    cos = patch.meta["expected"]["reconstruction"]["delta_cosine"]
    assert all(0.9999 <= c <= 1.00001 for c in cos)


if __name__ == "__main__":
    test_moipatch_save_load_attach_and_restore()
    test_make_moipatch_sycophancy_and_cli()
    test_real_p50c_loop_moipatch_artifact()
    print("ALL MOIPATCH TESTS PASSED!")
