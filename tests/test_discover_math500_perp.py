"""scripts/discover_math500_perp.py: the perpendicular-direction arm (CPU only, no model)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import discover_math500_perp as P  # noqa: E402


def _basis(D=16, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))
    return q.T  # rows orthonormal


def test_perp_direction_removes_the_healthy_part():
    V = _basis()
    u = np.random.default_rng(1).standard_normal(16)
    assert np.allclose(P.perp_direction(u, V, 0), u / np.linalg.norm(u))
    for k in (1, 4):
        w = P.perp_direction(u, V, k)
        assert abs(np.linalg.norm(w) - 1) < 1e-12
        assert np.abs(V[:k] @ w).max() < 1e-12  # nothing left along the top-k healthy directions
        assert np.abs(V[k:] @ w).max() > 1e-3
    with pytest.raises(ValueError):
        P.perp_direction(V[0] + 0.5 * V[1], V, 2)


def test_healthy_basis_orders_by_energy():
    V = _basis(8)
    ev = np.array([9.0, 1, 5, 0.1, 3, 0.5, 2, 0.2])
    S = V.T @ np.diag(ev) @ V
    B, e = P.healthy_basis(S, 3)
    assert np.allclose(e, [9, 5, 3])
    for i, j in enumerate((0, 2, 4)):
        assert abs(abs(B[i] @ V[j]) - 1) < 1e-9


def test_grid_has_every_k_and_unique_names():
    g = P.perp_grid([10, 14, 13])
    assert len(g) == len(P.K_PERP) * 2 * 4 * len(P.GRID_A) == 160
    assert len({c["name"] for c in g}) == len(g)
    assert {c["perp_k"] for c in g} == set(P.K_PERP)
    assert {tuple(c["layers"]) for c in g} == {(10,), (10, 14)}
    assert g[0]["name"] == "L10_n16_a4_p0"


def test_finalists_follow_the_top_rows_k():
    g = {c["name"]: c for c in P.perp_grid([10, 14])}
    rows = [{"name": n} for n in ("L10_n256_a16_p4", "L10_n256_a12_p0", "L10-14_nall_a8_p4", "L10_n64_a24_p1",
                                  "L10_nall_a4_p4", "L10_n16_a4_p4")]
    k, fin = P.pick_finalists(rows, g)
    assert k == 4
    assert [c["name"] for c in fin] == ["L10_n256_a16_p4", "L10-14_nall_a8_p4", "L10_nall_a4_p4"]


def test_select_uses_a_16k_cache_key_for_dev_only(monkeypatch):
    seen = []
    monkeypatch.setattr(P.m, "_try_load_base_cache", lambda w, c, sp, n, mt, o: seen.append(("load", sp, mt)))
    monkeypatch.setattr(P.m, "_save_base_cache", lambda w, c, sp, n, mt, o: seen.append(("save", sp, mt)))

    def fake_select(a):
        P.m._try_load_base_cache(None, {}, "dev", 32, 8192, "x")
        P.m._save_base_cache(None, {}, "dev", 32, 8192, "x")
        P.m._try_load_base_cache(None, {}, "test", 150, 16384, "x")

    monkeypatch.setattr(P.m, "cmd_select", fake_select)
    orig = P.m._try_load_base_cache
    P.cmd_select(argparse.Namespace())
    assert seen == [("load", "dev", 16384), ("save", "dev", 16384), ("load", "test", 16384)]
    assert P.m._try_load_base_cache is orig  # restored


def test_setup_copies_the_plain_run_and_sets_dev_to_16k(tmp_path, monkeypatch):
    monkeypatch.delenv("WSI_SYNC_DIR", raising=False)
    ref, work = tmp_path / "ref", tmp_path / "work"
    ref.mkdir()
    for f in P.COPY:
        if f.endswith(".jsonl"):
            rows = [{"id": i, "max_tokens": 8192, "prompt_ids": [1, 2]} for i in range(3)]
            (ref / f).write_text("".join(json.dumps(r) + "\n" for r in rows))
        elif f.endswith(".npz"):
            np.savez(ref / f, u0=np.ones(4, dtype=np.float32))
        else:
            (ref / f).write_text("{}")
    (ref / "outputs_test_fast.jsonl").write_text("")
    (ref / "outputs_dev_L10_n256_a12.jsonl").write_text("old 8k dev output")
    P.cmd_setup(argparse.Namespace(work=str(work), ref=str(ref)))
    dev = [json.loads(line) for line in (work / "requests_dev.jsonl").read_text().splitlines()]
    assert {r["max_tokens"] for r in dev} == {16384}
    assert {json.loads(line)["max_tokens"] for line in (work / "requests_test.jsonl").read_text().splitlines()} == {8192}
    assert (work / "plain_ref" / "outputs_test_fast.jsonl").exists()
    assert (work / "blind_directions_plain.npz").exists()
    assert not (work / "outputs_dev_L10_n256_a12.jsonl").exists()  # stale 8k dev outputs must not be reused
