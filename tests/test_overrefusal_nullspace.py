"""CPU tests: scripts/experiment_overrefusal_nullspace_aion.py (null-space rows, ridge cleaning, end to end on a toy model)."""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import eval_refusal_blind as ev  # noqa: E402
import experiment_overrefusal_nullspace_aion as ns  # noqa: E402
from test_eval_refusal_blind import toy  # noqa: E402


def test_null_rows_leave_reference_inputs_alone():
    torch.manual_seed(0)
    X = torch.randn(200, 4) @ torch.randn(4, 12)  # reference inputs live in a 4-dim subspace of R^12
    row = torch.randn(12, dtype=torch.float64)
    out = ns.null_rows(X.T @ X, {"u": row}, [0.5, 0.999999])
    full = out[("u", 0.999999)]
    assert full["k"] == 4 and full["dim"] == 12
    assert (X.double() @ full["v"]).abs().max() < 1e-6 * (X.double() @ row).abs().max()  # no effect on the reference
    assert 0 < full["kept_norm"] < 1
    assert out[("u", 0.5)]["k"] < 4 and out[("u", 0.5)]["kept_norm"] >= full["kept_norm"]


def test_ridge_clean():
    torch.manual_seed(1)
    a = torch.randn(32, dtype=torch.float64)
    u = 0.6 * a / a.norm() + 0.8 * torch.nn.functional.normalize(torch.randn(32, dtype=torch.float64), dim=0)
    uc, rep = ns.ridge_clean(u, {"exam_template": a, "none": None}, 1e-6)
    assert abs(float(uc @ a)) < 1e-4 and abs(float(uc.norm()) - 1) < 1e-9
    assert rep["atoms"] == ["exam_template"] and abs(rep["cos_after"]["exam_template"]) < 1e-4
    same, rep0 = ns.ridge_clean(u, {}, 1e-3)
    assert torch.allclose(same, u) and rep0["atoms"] == []


def test_end_to_end_toy(tmp_path):
    from wsi.moipatch import Edit, MoIPatch

    d = toy(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    parts = ev.split_rows([f"w{i} w{i + 1}" for i in range(12)], [{"prompt": f"w{i}", "category": "c"} for i in range(6)],
                          [f"w{20 + i} w3" for i in range(8)], n_or=6, n_tox_agent=2, n_tox_test=4, n_ctrl=4)
    for k, v in parts.items():
        ev.write_jsonl(work / f"{k}.jsonl", v)
    u = torch.randn(16, dtype=torch.float64)
    mods = ["layers.0.mlp.down_proj", "layers.0.self_attn.o_proj", "layers.1.mlp.down_proj", "layers.1.self_attn.o_proj"]
    v2 = MoIPatch(edits=[Edit(module=m, u=u / u.norm(), alpha=1.0) for m in mods],
                  router={"layer": 1, "readout": "last_token", "dir": torch.randn(16), "thr": -1e9})
    v2.save(tmp_path / "v2.moipatch")
    out = tmp_path / "res.json"
    ns.main(["--base-dir", str(d), "--work", str(work), "--v2-patch", str(tmp_path / "v2.moipatch"),
             "--out-dir", str(tmp_path / "patches"), "--out", str(out), "--dir-layer", "1", "--dtype", "float32",
             "--device", "cpu", "--bs", "3", "--max-new", "6", "--ref-tokens", "12", "--skip-onset", "2"])
    r = json.loads(out.read_text())
    assert r["nullspace"]["reference_answer_tokens"] > 0
    for arm in ("null90", "null99", "clean", "clean_null99"):
        assert (tmp_path / "patches" / f"{arm}.moipatch").exists()
        a = r["arms"][arm]
        assert {"recovered", "toxic_kept", "controls_identical", "style", "success_with_style", "suppression_ratio"} <= set(a)
    assert 0 < r["arms"]["null99"]["edit_norm_kept_mean"] <= r["arms"]["null90"]["edit_norm_kept_mean"] <= 1
    p = MoIPatch.load(tmp_path / "patches" / "null99.moipatch")
    assert p.edits[0].v is not None and p.router["thr"] == -1e9 and len(p.edits) == 4
    assert (work / "gen_ns_null99_controls.jsonl").exists() and (work / "eval_ns_clean.json").exists()
