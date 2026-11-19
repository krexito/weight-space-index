"""Fast blind discovery: same edits as the blind run, exact in-memory swaps, and a working early loop stop."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
MARKER = "# ===== EXAM: runs only after freeze ====="


@pytest.fixture()
def fast():
    pytest.importorskip("pandas")
    pytest.importorskip("torch")
    import discover_fast

    return discover_fast


def test_discovery_never_names_the_answer():
    text = (SCRIPTS / "discover_fast.py").read_text(encoding="utf-8")
    assert MARKER in text
    tree = ast.parse(text[: text.index(MARKER)])
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and "\n" not in n.value:
            low = n.value.lower()
            assert not any(t in low for t in ("p50c", "exp8_math", "full_attention", "layer_types")), n.value
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            assert n.value not in (15, 19), n.value


def test_grid_is_12_configs_from_the_ranking(fast):
    g = fast.grid([7, 2, 5])
    assert len(g) == 12
    assert {tuple(c["layers"]) for c in g} == {(7,), (2, 7)}
    assert {c["neurons"] for c in g} == {16, "all"} and {c["alpha"] for c in g} == {4.0, 8.0, 12.0}


def _work(tmp_path, D=8, U=12, L=3, dtype="bfloat16"):
    import torch
    from safetensors.torch import save_file

    base = tmp_path / "base"
    base.mkdir()
    g = torch.Generator().manual_seed(0)
    W = {f"model.language_model.layers.{i}.mlp.down_proj.weight":
         torch.randn(D, U, generator=g).to(getattr(torch, dtype)) for i in range(L)}
    save_file(W, str(base / "model.safetensors"))
    (base / "config.json").write_text("{}")
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.json").write_text(json.dumps({"base_dir": str(base), "eos": [0], "seed": 0}))
    rng = np.random.default_rng(0)
    np.savez(work / "blind_directions.npz", **{f"u{i}": (v := rng.standard_normal(D)) / np.linalg.norm(v)
                                                 for i in range(L)})
    rank = {str(i): [int(x) for x in rng.permutation(U)] for i in range(L)}
    (work / "diagnosis_blind.json").write_text(json.dumps({"neuron_rank": rank, "hidden": D, "layer_rank": [1, 2, 0]}))
    return base, work, W


def test_swap_weights_equal_the_baked_checkpoint(fast, tmp_path):
    import discover_blind as blind
    import torch
    from safetensors.torch import load_file

    base, work, W = _work(tmp_path)
    base_w = fast.load_base_layers(base, {1, 2})
    for kind in ("wsi", "random"):
        job = {"layers": [1, 2], "neurons": 4, "alpha": 12.0, "direction": kind}
        upd_np, upd_t = fast.job_updates(work, job, base_w, set())
        ck = blind.build_checkpoint(work, [1, 2], 4, 12.0, kind, work / f"ck_{kind}")
        baked = load_file(str(ck / "model.safetensors"))
        for li in (1, 2):
            k = f"model.language_model.layers.{li}.mlp.down_proj.weight"
            assert torch.equal(upd_t[li], baked[k])
            # through the float32 transport and a cast to the serving dtype, as vLLM on a T4 would hold it
            for serve in (torch.bfloat16, torch.float16):
                via = torch.from_numpy(upd_np[li]).to(serve)
                assert torch.equal(via, baked[k].to(serve))
                assert fast.int_checksum(upd_t[li], serve) == fast.int_checksum(via, serve)


def test_next_job_restores_layers_it_does_not_edit(fast, tmp_path):
    import torch

    base, work, W = _work(tmp_path)
    base_w = fast.load_base_layers(base, {1, 2})
    _, upd = fast.job_updates(work, {"layers": [2], "neurons": "all", "alpha": 4.0}, base_w, {1, 2})
    assert set(upd) == {1, 2}
    assert torch.equal(upd[1], base_w[1]) and not torch.equal(upd[2], base_w[2])
    _, upd = fast.job_updates(work, {"name": "base"}, base_w, {2})
    assert set(upd) == {2} and torch.equal(upd[2], base_w[2])


def test_setter_writes_in_place_and_checksums(fast, tmp_path):
    import torch

    base, work, W = _work(tmp_path)
    model = torch.nn.Module()
    lm = torch.nn.Module()
    model.add_module("language_model", lm)
    lyrs = torch.nn.ModuleList()
    for i in range(3):
        blk = torch.nn.Module()
        mlp = torch.nn.Module()
        mlp.down_proj = torch.nn.Linear(12, 8, bias=False).to(torch.float16)
        blk.mlp = mlp
        lyrs.append(blk)
    lm.add_module("layers", lyrs)
    ptr = lyrs[1].mlp.down_proj.weight.data_ptr()
    upd_np, upd_t = fast.job_updates(work, {"layers": [1], "neurons": 4, "alpha": 8.0},
                                     fast.load_base_layers(base, {1}), set())
    got = fast.make_setter(upd_np)(model)
    assert set(got) == {1} and lyrs[1].mlp.down_proj.weight.data_ptr() == ptr
    dt, cs = got[1]
    assert dt == "torch.float16" and cs == fast.int_checksum(upd_t[1], torch.float16)
    assert torch.equal(lyrs[1].mlp.down_proj.weight.data, upd_t[1].to(torch.float16))


# ------------------------------------------------------------------ early loop stop (vLLM-free core)
class _Fake:
    def __init__(self, n_slots, width):
        import torch

        self.max_num_reqs = n_slots
        self.all_token_ids = SimpleNamespace(gpu=torch.zeros(n_slots, width, dtype=torch.int64))
        self.prompt_len = SimpleNamespace(np=np.zeros(n_slots, dtype=np.int64))


def test_loop_kill_stops_loops_soon_after_the_rule_and_leaves_others():
    torch = pytest.importorskip("torch")
    import loop_kill_v2 as lk
    import moi_loop_router_vllm as lr

    P, T, V, EOS = 5, 3000, 50, 49
    rng = np.random.default_rng(0)
    normal = list(rng.integers(0, 40, size=T))
    loop = list(rng.integers(0, 40, size=400)) + list(rng.integers(0, 40, size=37)) * 80
    seqs = {0: loop[:T], 1: normal}
    rs = _Fake(4, P + T + 10)
    core = lk._LoopKillCore(rs, stride=8)
    on = SimpleNamespace(extra_args={"loop_kill_eos": EOS})
    for s in (0, 1):
        assert core.add_request(s, on)
        rs.prompt_len.np[s] = P
    assert not core.add_request(2, SimpleNamespace(extra_args=None))
    killed_at = {}
    for t in range(T):
        for s in (0, 1):  # token t of the response is committed before the step that samples t + 1
            rs.all_token_ids.gpu[s, P + t] = seqs[s][t]
        ctx = SimpleNamespace(idx_mapping_np=np.array([1, 0, 2]), seq_lens_upper_bound_np=np.array([P + t + 1] * 3))
        logits = core.apply(torch.zeros(3, V), ctx)
        for row, s in ((0, 1), (1, 0)):
            if s not in killed_at and torch.isinf(logits[row]).any():
                assert logits[row, EOS] == 0 and torch.isinf(logits[row, :EOS]).all()
                killed_at[s] = t + 1  # response length when EOS is forced
        assert not torch.isinf(logits[2]).any()
    exact = lr.first_window_hit(lr.dup_flags(loop[:T]), lr.KILL_WINDOW, lr.KILL_FRAC)
    assert exact is not None and 1 not in killed_at
    assert exact <= killed_at[0] <= exact + 8 + 2
