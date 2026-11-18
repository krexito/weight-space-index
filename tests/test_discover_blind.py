"""The blind discovery script must not know the answer: no p50c, no layer types, no layer-number constants."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SRC = SCRIPTS / "discover_blind.py"
MARKER = "# ===== EXAM: runs only after freeze ====="
# generic helpers of discover_loop_fix.py the discovery part may call (not its diagnose/build/report, which carry priors)
DL_ALLOWED = {"make_requests", "cfg_save", "cfg_load", "token_labels", "down_key", "score", "cmd_gen", "cmd_plan_moi",
              "run_timed"}
FORBIDDEN_TEXT = ("p50c", "exp8_math", "full_attention", "layer_types", "linear_attention")
FORBIDDEN_INTS = {15, 19}  # p50c's layers


def _discovery_nodes():
    text = SRC.read_text(encoding="utf-8")
    assert MARKER in text, "the exam marker moved or was removed"
    cut = text[: text.index(MARKER)].count("\n") + 1
    tree = ast.parse(text)
    doc = ast.get_docstring(tree, clean=False)
    nodes = [n for n in tree.body if n.lineno < cut and not (isinstance(n, ast.Expr) and
                                                               isinstance(getattr(n, "value", None), ast.Constant)
                                                               and n.value.value == doc)]
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    return nodes, main


def _strings_and_names(nodes):
    for top in nodes:
        for n in ast.walk(top):
            if isinstance(n, ast.Constant):
                yield n.value
            elif isinstance(n, ast.Name):
                yield n.id
            elif isinstance(n, ast.Attribute):
                yield n.attr


def test_discovery_never_names_the_answer():
    nodes, _ = _discovery_nodes()
    for v in _strings_and_names(nodes):
        if isinstance(v, str):
            low = v.lower()
            # function docstrings may say what is NOT used; only code strings and identifiers are checked here
            if "\n" in v:
                continue
            assert not any(t in low for t in FORBIDDEN_TEXT), f"discovery code mentions {v!r}"
        elif isinstance(v, int) and not isinstance(v, bool):
            assert v not in FORBIDDEN_INTS, f"discovery code has the constant {v}"


def test_discovery_uses_only_generic_helpers():
    nodes, main = _discovery_nodes()
    used = {n.attr for top in nodes + main for n in ast.walk(top)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "dl"}
    assert used <= DL_ALLOWED, f"calls priors from discover_loop_fix: {used - DL_ALLOWED}"
    imports = {a.name for top in nodes for n in ast.walk(top) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in n.names} | {n.module for top in nodes for n in ast.walk(top)
                                    if isinstance(n, ast.ImportFrom) and n.module}
    assert not any("tracer" in i or "moipatch" in i for i in imports), imports


@pytest.fixture()
def blind():
    sys.path.insert(0, str(SCRIPTS))
    pytest.importorskip("pandas")
    import discover_blind

    return discover_blind


def test_grid_comes_from_the_ranking_only(blind):
    g1 = blind.grid([3, 0, 7, 1, 2])
    g2 = blind.grid([0, 7, 3, 1, 2])
    assert len(g1) == len(blind.K_LAYERS) * len(blind.NEURONS) * len(blind.ALPHAS)
    assert {tuple(c["layers"]) for c in g1} == {(3,), (0, 3), (0, 3, 7)}
    assert {tuple(c["layers"]) for c in g2} == {(0,), (0, 7), (0, 3, 7)}


def test_report_refuses_before_freeze(blind, tmp_path):
    class A:
        work = str(tmp_path)
        p50c_dir = None
        out = "r.json"

    pytest.importorskip("transformers")
    with pytest.raises(SystemExit, match="frozen"):
        blind.cmd_report(A)


def test_build_applies_the_edit_form(blind, tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    D, U, L = 8, 12, 3
    base = tmp_path / "base"
    base.mkdir()
    g = torch.Generator().manual_seed(0)
    W = {f"model.layers.{i}.mlp.down_proj.weight": torch.randn(D, U, generator=g) for i in range(L)}
    save_file(W, str(base / "model.safetensors"))
    (base / "config.json").write_text("{}")
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.json").write_text(json.dumps({"base_dir": str(base), "eos": [0], "seed": 0}))
    rng = np.random.default_rng(0)
    u = {f"u{i}": (v := rng.standard_normal(D)) / np.linalg.norm(v) for i in range(L)}
    np.savez(work / "blind_directions.npz", **u)
    rank = {str(i): [int(x) for x in rng.permutation(U)] for i in range(L)}
    (work / "diagnosis_blind.json").write_text(json.dumps({"neuron_rank": rank, "hidden": D, "layer_rank": [1, 2, 0]}))
    for kind in ("wsi", "random"):
        out = blind.build_checkpoint(work, [1, 2], 4, 3.0, kind, work / f"ck_{kind}")
        new = load_file(str(out / "model.safetensors"))
        k0 = "model.layers.0.mlp.down_proj.weight"
        assert torch.equal(new[k0], W[k0])
        for li in (1, 2):
            k = f"model.layers.{li}.mlp.down_proj.weight"
            uu = torch.from_numpy(u[f"u{li}"]) if kind == "wsi" else torch.from_numpy(
                blind.direction({"hidden": D}, None, li, "random"))
            m = torch.zeros(U, dtype=torch.float64)
            m[sorted(rank[str(li)][:4])] = 1
            Wd = W[k].double()
            exp = (Wd - 3.0 * torch.outer(uu, m * (uu @ Wd))).float()
            assert torch.allclose(new[k], exp, atol=1e-6)
            assert torch.equal(new[k][:, m == 0], W[k][:, m == 0])
    assert blind.columns({"neuron_rank": rank}, 1, "all") == list(range(U))
