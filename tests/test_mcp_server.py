"""Unit tests for the WSI MCP server tools (`wsi_build_moipatch`, `wsi_screen_candidates`)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from wsi import mcp_server as mcp


class _TinyBlock(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp.down_proj(torch.relu(self.mlp.up_proj(x)))


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 12, inter: int = 16, n_layers: int = 2):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_TinyBlock(hidden, inter) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, **_kw):
        h = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            h = blk(h)
        return SimpleNamespace(logits=self.lm_head(h))


def test_mcp_exposes_build_and_screen_tools():
    resp = mcp.handle_rpc_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert {
        "wsi_build_moipatch",
        "wsi_screen_candidates",
        "wsi_capture_causal_traces",
        "wsi_attribute_direction_writers",
        "wsi_trace_direction_graph",
        "wsi_peel_causal_circuit",
        "wsi_bake_moipatch",
        "wsi_audit_moipatch",
        "wsi_calibrate_safety_router",
    }.issubset(names)
    assert len(names) == 22


def test_mcp_build_moipatch_and_screen_candidates_end_to_end(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = _TinyLM(vocab=32, hidden=12, inter=16, n_layers=2).eval()

    # Save weights as a safetensors checkpoint so MoIPatch.fingerprint_from_checkpoint works
    ckpt = tmp_path / "toy_ckpt"
    ckpt.mkdir()
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(ckpt / "model.safetensors"))

    # Also save a directions .npz file to test both inline `direction` and `direction_npz`
    u0 = np.zeros(12, dtype=np.float32)
    u0[0] = 1.0
    u1 = np.zeros(12, dtype=np.float32)
    u1[1] = 1.0
    npz_path = tmp_path / "blind_directions.npz"
    np.savez(npz_path, u0=u0, u1=u1)

    p1 = tmp_path / "cand_inline.moipatch"
    p2 = tmp_path / "cand_npz.moipatch"

    r1 = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "wsi_build_moipatch",
            "arguments": {
                "model_dir": str(ckpt),
                "save_path": str(p1),
                "edits": [{"layer": 0, "alpha": 2.0, "direction": u0.tolist(), "neurons": [0, 1, 2, 3]}],
                "meta": {"source": "test_inline"},
            },
        },
    })
    assert r1["result"]["isError"] is False
    out1 = json.loads(r1["result"]["content"][0]["text"])
    assert p1.exists() and out1["n_edits"] == 1 and out1["modules"] == ["layers.0.mlp.down_proj"]

    r2 = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "wsi_build_moipatch",
            "arguments": {
                "model_dir": str(ckpt),
                "save_path": str(p2),
                "edits": [{"layer": 1, "alpha": 8.0, "direction_npz": str(npz_path)}],
            },
        },
    })
    assert r2["result"]["isError"] is False
    assert p2.exists()

    # Screen both .moipatch candidates on the tiny CPU model via MCP
    monkeypatch.setattr(mcp, "_load_model_for_screen", lambda *_a, **_kw: model)
    r3 = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "wsi_screen_candidates",
            "arguments": {
                "model_dir": str(ckpt),
                "patch_paths": {"mild_L0": str(p1), "strong_L1": str(p2)},
                "healthy_seqs": [[1, 2, 3, 4, 5, 6]],
                "target_seqs": [[7, 8, 9, 7, 8, 9, 7, 8]],
                "target_spans": [[3, 8]],
                "max_nll_delta": 0.05,
                "rank_by": "pareto",
            },
        },
    })
    assert r3["result"]["isError"] is False, r3["result"]["content"][0]["text"]
    out3 = json.loads(r3["result"]["content"][0]["text"])
    assert out3["total_screened"] == 2
    assert {row["name"] for row in out3["ranking"]} == {"mild_L0", "strong_L1"}
    for row in out3["ranking"]:
        assert "pareto_score" in row and "target_suppression" in row and "delta_healthy_nll" in row


def test_mcp_discover_compact_circuit_causal_union_and_screen(tmp_path, monkeypatch):
    torch.manual_seed(7)
    model = _TinyLM(vocab=32, hidden=12, inter=16, n_layers=2).eval()
    ckpt = tmp_path / "toy_ckpt_causal"
    ckpt.mkdir()
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(ckpt / "model.safetensors"))
    (ckpt / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": 12, "intermediate_size": 16, "num_hidden_layers": 2, "vocab_size": 32}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp, "_load_model_for_screen", lambda *_a, **_kw: model)

    h_seqs = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
    l_seqs = [[1, 2, 9, 9, 9, 9], [2, 3, 9, 9, 9, 9]]

    # Step 0: Capture causal traces via the new `wsi_capture_causal_traces` MCP tool
    r_cap_h = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "wsi_capture_causal_traces",
            "arguments": {
                "model_dir": str(ckpt),
                "output_dir": str(tmp_path / "cap_h"),
                "sequences": h_seqs,
                "target_spans": [[2, 6], [2, 6]],
                "healthy_spans": [[0, 2], [0, 2]],
                "grad_mode": "summary",
                "top_k_mlp": 8,
            },
        },
    })
    assert r_cap_h["result"]["isError"] is False, r_cap_h["result"]["content"][0]["text"]
    h_dirs = json.loads(r_cap_h["result"]["content"][0]["text"])["trace_dirs"]

    r_cap_l = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "wsi_capture_causal_traces",
            "arguments": {
                "model_dir": str(ckpt),
                "output_dir": str(tmp_path / "cap_l"),
                "sequences": l_seqs,
                "target_spans": [[2, 6], [2, 6]],
                "healthy_spans": [[0, 2], [0, 2]],
                "grad_mode": "summary",
                "top_k_mlp": 8,
            },
        },
    })
    assert r_cap_l["result"]["isError"] is False, r_cap_l["result"]["content"][0]["text"]
    cap_l_out = json.loads(r_cap_l["result"]["content"][0]["text"])
    l_dirs = cap_l_out["trace_dirs"]
    assert "causal_diagnosis" in cap_l_out and cap_l_out["n_traces"] == 2

    grid_dir = tmp_path / "grid_patches"
    r_disc = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "wsi_discover_compact_circuit",
            "arguments": {
                "model_dir": str(ckpt),
                "trace_dirs": h_dirs,
                "loop_trace_dirs": l_dirs,
                "top_k_neurons": 4,
                "save_candidate_grid_dir": str(grid_dir),
                "candidate_alphas": [4.0, 8.0],
            },
        },
    })
    assert r_disc["result"]["isError"] is False, r_disc["result"]["content"][0]["text"]
    disc_out = json.loads(r_disc["result"]["content"][0]["text"])
    assert disc_out["causal_diagnosis"] is not None
    assert "summary" in disc_out["causal_diagnosis"] and "sites" in disc_out["causal_diagnosis"]
    saved_grid = disc_out["saved_candidate_grid"]
    assert isinstance(saved_grid, dict) and len(saved_grid) >= 2
    assert any(k.startswith("passive_") for k in saved_grid)
    assert any(k.startswith("causal_") for k in saved_grid)

    # Pass `saved_candidate_grid` directly into `wsi_screen_candidates` (the forward judge)
    r_scr = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "wsi_screen_candidates",
            "arguments": {
                "model_dir": str(ckpt),
                "patch_paths": saved_grid,
                "healthy_seqs": h_seqs,
                "target_seqs": l_seqs,
                "target_spans": [[2, 6], [2, 6]],
            },
        },
    })
    assert r_scr["result"]["isError"] is False, r_scr["result"]["content"][0]["text"]
    scr_out = json.loads(r_scr["result"]["content"][0]["text"])
    assert scr_out["total_screened"] == len(saved_grid)
    assert scr_out["ranking"][0]["name"] in saved_grid



def test_mcp_direction_tracing_graph_and_peel_end_to_end(tmp_path, monkeypatch):
    import torch.nn.functional as F

    from wsi import Edit, MoIPatch

    class _DirMcpAttn(nn.Module):
        def __init__(self, hidden: int, n_heads: int) -> None:
            super().__init__()
            self.num_heads = n_heads
            self.head_dim = hidden // n_heads
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, hidden, bias=False)
            self.v_proj = nn.Linear(hidden, hidden, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, H = x.shape
            nh, hd = self.num_heads, self.head_dim
            q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
            k = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
            v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
            mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
            ctx = (torch.softmax(scores + mask, dim=-1) @ v).transpose(1, 2).reshape(B, T, H)
            return self.o_proj(ctx)

    class _DirMcpBlock(nn.Module):
        def __init__(self, hidden: int, inter: int, n_heads: int) -> None:
            super().__init__()
            self.self_attn = _DirMcpAttn(hidden, n_heads)
            self.mlp = nn.Module()
            self.mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
            self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
            self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.self_attn(x)
            return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))

    class _TinyDirLM(nn.Module):
        def __init__(self, vocab: int = 32, hidden: int = 16, inter: int = 32, n_layers: int = 2, n_heads: int = 4) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="llama", hidden_size=hidden, num_attention_heads=n_heads)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(vocab, hidden)
            self.model.layers = nn.ModuleList([_DirMcpBlock(hidden, inter, n_heads) for _ in range(n_layers)])
            self.model.norm = nn.LayerNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False, **_kw):
            h = self.model.embed_tokens(input_ids)
            hs = [h] if output_hidden_states else None
            for idx, blk in enumerate(self.model.layers):
                h = blk(h)
                if hs is not None and idx < len(self.model.layers) - 1:
                    hs.append(h)
            h_out = self.model.norm(h)
            if hs is not None:
                hs.append(h_out)
            return SimpleNamespace(logits=self.lm_head(h_out), hidden_states=tuple(hs) if hs is not None else None)

    model = _TinyDirLM().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.model.norm.weight.fill_(1.0)
        model.model.embed_tokens.weight[10:15, 0] = 1.0
        model.model.embed_tokens.weight[10:15, 2] = 4.0
        b0, b1 = model.model.layers[0], model.model.layers[1]
        # L0.mlp writes target e3 and protected e5; also relays coord1 -> L1
        b0.mlp.gate_proj.weight[0, 0] = 2.0
        b0.mlp.up_proj.weight[0, 0] = 2.0
        b0.mlp.down_proj.weight[1, 0] = 1.5
        b0.mlp.down_proj.weight[3, 0] = 2.0
        b0.mlp.down_proj.weight[5, 0] = 2.0
        # L1.attn reads relay coord1 -> e3; L1.mlp reads coord2 -> e3 cleanly
        b1.self_attn.v_proj.weight[0, 1] = 1.0
        b1.self_attn.o_proj.weight[3, 0] = 1.2
        b1.mlp.gate_proj.weight[0, 2] = 0.5
        b1.mlp.up_proj.weight[0, 2] = 0.5
        b1.mlp.down_proj.weight[3, 0] = 1.5
        # Saturated output logit for token 0
        model.lm_head.weight[0, 0] = 60.0

    ckpt = tmp_path / "dir_ckpt"
    ckpt.mkdir()
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(ckpt / "model.safetensors"))
    monkeypatch.setattr(mcp, "_load_model_for_screen", lambda *_a, **_kw: model)

    tdir = torch.zeros(16)
    tdir[3] = 1.0
    pdir = torch.zeros(16)
    pdir[5] = 1.0

    router_patch = MoIPatch(
        edits=[Edit(module="layers.0.mlp.down_proj", u=tdir.clone(), alpha=1.0)],
        router={"layer": 2, "readout": "last_token", "dir": tdir, "thr": 0.0, "veto_dir": pdir, "veto_thr": 0.0},
    )
    router_patch.fingerprint_from_checkpoint(ckpt)
    router_patch_path = router_patch.save(tmp_path / "router_carrier.moipatch")

    seqs = [[10, 11, 12], [12, 13, 14]]

    # 1. Level-1 writer attribution (inline block + patch router post-norm linearized)
    r_attr_inline = mcp.handle_rpc_message({
        "jsonrpc": "2.0", "id": 20, "method": "tools/call",
        "params": {
            "name": "wsi_attribute_direction_writers",
            "arguments": {
                "model_dir": str(ckpt),
                "sequences": seqs,
                "target_dir": tdir.tolist(),
                "target_layer": 1,
                "layer_convention": "block",
                "report_block_gaps": True,
            },
        },
    })
    assert r_attr_inline["result"]["isError"] is False, r_attr_inline["result"]["content"][0]["text"]
    attr_in = json.loads(r_attr_inline["result"]["content"][0]["text"])
    assert abs(attr_in["residual_gap"]) < 1e-4
    assert len(attr_in["block_residual_gaps"]) == 2
    assert all(abs(g["gap"]) < 1e-4 for g in attr_in["block_residual_gaps"])

    r_attr_patch = mcp.handle_rpc_message({
        "jsonrpc": "2.0", "id": 21, "method": "tools/call",
        "params": {
            "name": "wsi_attribute_direction_writers",
            "arguments": {
                "model_dir": str(ckpt),
                "sequences": seqs,
                "target_patch_path": str(router_patch_path),
                "target_patch_key": "veto_dir",
            },
        },
    })
    assert r_attr_patch["result"]["isError"] is False, r_attr_patch["result"]["content"][0]["text"]
    attr_pa = json.loads(r_attr_patch["result"]["content"][0]["text"])
    assert attr_pa["target_layer"] == 2 and attr_pa["layer_convention"] == "hidden_states"
    assert attr_pa["final_norm_linearized"] is True
    assert abs(attr_pa["residual_gap"]) < 1e-4

    # 2. Capture causal traces with target_dir + protect_dir AND with contrast_token_ids (logit_margin)
    r_cap_dp = mcp.handle_rpc_message({
        "jsonrpc": "2.0", "id": 22, "method": "tools/call",
        "params": {
            "name": "wsi_capture_causal_traces",
            "arguments": {
                "model_dir": str(ckpt),
                "output_dir": str(tmp_path / "cap_dir_prot"),
                "sequences": seqs,
                "target_dir": tdir.tolist(),
                "target_layer": 1,
                "target_layer_convention": "block",
                "protect_dir": pdir.tolist(),
                "grad_mode": "compressed",
            },
        },
    })
    assert r_cap_dp["result"]["isError"] is False, r_cap_dp["result"]["content"][0]["text"]
    cap_dp = json.loads(r_cap_dp["result"]["content"][0]["text"])
    assert cap_dp["grad_objective"] == "direction"
    assert cap_dp["grad_healthy_objective"] == "direction"
    assert "direction_summary" in cap_dp

    r_cap_lm = mcp.handle_rpc_message({
        "jsonrpc": "2.0", "id": 23, "method": "tools/call",
        "params": {
            "name": "wsi_capture_causal_traces",
            "arguments": {
                "model_dir": str(ckpt),
                "output_dir": str(tmp_path / "cap_margin"),
                "sequences": [[10, 0]],
                "target_spans": [[0, 1]],
                "target_token_ids": [0],
                "contrast_token_ids": [1],
                "grad_mode": "summary",
            },
        },
    })
    assert r_cap_lm["result"]["isError"] is False, r_cap_lm["result"]["content"][0]["text"]
    cap_lm = json.loads(r_cap_lm["result"]["content"][0]["text"])
    assert cap_lm["grad_objective"] == "logit_margin"

    # 3. Multi-hop direction graph
    r_graph = mcp.handle_rpc_message({
        "jsonrpc": "2.0", "id": 24, "method": "tools/call",
        "params": {
            "name": "wsi_trace_direction_graph",
            "arguments": {
                "model_dir": str(ckpt),
                "sequences": seqs,
                "target_dir": tdir.tolist(),
                "target_layer": 1,
                "layer_convention": "block",
                "depth": 2,
                "top_k_nodes": 3,
            },
        },
    })
    assert r_graph["result"]["isError"] is False, r_graph["result"]["content"][0]["text"]
    graph = json.loads(r_graph["result"]["content"][0]["text"])
    assert len(graph["nodes"]) >= 3 and len(graph["edges"]) >= 2
    assert len(graph["top_paths"]) >= 1
    assert any("layers.0.mlp.down_proj:n0" in p["nodes"] for p in graph["top_paths"])

    # 4. Iterative Hydra causal peeling + saved cumulative .moipatch
    peeled_path = tmp_path / "peeled.moipatch"
    r_peel = mcp.handle_rpc_message({
        "jsonrpc": "2.0", "id": 25, "method": "tools/call",
        "params": {
            "name": "wsi_peel_causal_circuit",
            "arguments": {
                "model_dir": str(ckpt),
                "sequences": seqs,
                "n_rounds": 2,
                "alpha": 2.0,
                "target_dir": tdir.tolist(),
                "target_layer": 1,
                "target_layer_convention": "block",
                "save_patch_path": str(peeled_path),
            },
        },
    })
    assert r_peel["result"]["isError"] is False, r_peel["result"]["content"][0]["text"]
    peel = json.loads(r_peel["result"]["content"][0]["text"])
    assert peel["restore_exact_ok"] is True
    assert peel["n_rounds_completed"] == 2
    assert peel["n_edits"] == 2 and peeled_path.exists()
    assert MoIPatch.load(peeled_path).attach(model).fingerprint_ok() is True


def test_mcp_bake_moipatch(tmp_path):
    from safetensors import safe_open
    from wsi.moipatch import Edit, MoIPatch, _sha256, apply_edit

    torch.manual_seed(2)
    w = torch.randn(12, 12, dtype=torch.float32)
    ckpt = tmp_path / "base_ckpt"
    ckpt.mkdir()
    save_file({"model.layers.0.mlp.down_proj.weight": w}, str(ckpt / "model.safetensors"))

    u = torch.zeros(12, dtype=torch.float64)
    u[0] = 1.0
    edit = Edit(
        module="layers.0.mlp.down_proj",
        u=u,
        alpha=0.75,
        base_sha256=_sha256(w),
        base_dtype="float32",
        shape=(12, 12),
    )

    static_p = tmp_path / "static.moipatch"
    MoIPatch(edits=[edit], meta={"name": "static-test"}).save(static_p)

    out_dir = tmp_path / "baked_static"
    r_ok = mcp.handle_rpc_message({
        "jsonrpc": "2.0", "id": 30, "method": "tools/call",
        "params": {
            "name": "wsi_bake_moipatch",
            "arguments": {
                "model_dir": str(ckpt),
                "out_dir": str(out_dir),
                "patch_path": str(static_p),
                "copy_untouched": "copy",
            },
        },
    })
    assert r_ok["result"]["isError"] is False, r_ok["result"]["content"][0]["text"]
    rep = json.loads(r_ok["result"]["content"][0]["text"])
    assert rep["n_touched_shards"] == 1 and rep["had_router_warning"] is False
    assert (out_dir / "wsi_baked_patches.json").exists()
    with safe_open(str(out_dir / "model.safetensors"), "pt") as sf:
        assert torch.equal(sf.get_tensor("model.layers.0.mlp.down_proj.weight").clone(), apply_edit(w, u, None, 0.75))

    # Guard routed patch unless allow_routed=True
    r_dir = torch.zeros(12, dtype=torch.float32)
    r_dir[1] = 1.0
    routed_p = tmp_path / "routed.moipatch"
    MoIPatch(
        edits=[edit],
        router={"layer": 1, "readout": "last_token", "dir": r_dir, "thr": 0.5},
        meta={"name": "abliterated-sfw"},
    ).save(routed_p)

    r_blocked = mcp.handle_rpc_message({
        "jsonrpc": "2.0", "id": 31, "method": "tools/call",
        "params": {
            "name": "wsi_bake_moipatch",
            "arguments": {"model_dir": str(ckpt), "out_dir": str(tmp_path / "baked_fail"), "patch_paths": [str(routed_p)]},
        },
    })
    assert r_blocked["result"]["isError"] is True
    assert "allow_routed" in r_blocked["result"]["content"][0]["text"]

    r_forced = mcp.handle_rpc_message({
        "jsonrpc": "2.0", "id": 32, "method": "tools/call",
        "params": {
            "name": "wsi_bake_moipatch",
            "arguments": {
                "model_dir": str(ckpt),
                "out_dir": str(tmp_path / "baked_forced"),
                "patch_paths": [str(routed_p)],
                "allow_routed": True,
                "copy_untouched": "copy",
            },
        },
    })
    assert r_forced["result"]["isError"] is False, r_forced["result"]["content"][0]["text"]
    assert json.loads(r_forced["result"]["content"][0]["text"])["had_router_warning"] is True


def test_mcp_audit_and_calibrate_safety_router(tmp_path, monkeypatch):
    import torch.nn.functional as F

    from wsi import Edit, MoIPatch
    from wsi.moipatch import _sha256

    class _CalBlock(nn.Module):
        def __init__(self, hidden: int, inter: int) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
            self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.mlp.down_proj(F.relu(self.mlp.up_proj(x)))

    class _CalLM(nn.Module):
        def __init__(self, vocab: int = 32, hidden: int = 12, inter: int = 24, n_layers: int = 3) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="llama", hidden_size=hidden, num_attention_heads=2)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(vocab, hidden)
            self.model.layers = nn.ModuleList([_CalBlock(hidden, inter) for _ in range(n_layers)])
            self.model.norm = nn.LayerNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False, **_kw):
            h = self.model.embed_tokens(input_ids)
            hs = [h] if output_hidden_states else None
            for idx, blk in enumerate(self.model.layers):
                h = blk(h)
                if hs is not None and idx < len(self.model.layers) - 1:
                    hs.append(h)
            h_out = self.model.norm(h)
            if hs is not None:
                hs.append(h_out)
            return SimpleNamespace(logits=self.lm_head(h_out), hidden_states=tuple(hs) if hs is not None else None)

    class _CalTok:
        vocab_size = 32
        eos_token_id = 2

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            low = text.lower()
            base = 3
            if "toxic" in low:
                base = 20
            elif "refuse" in low:
                base = 12
            extra = (sum(ord(ch) for ch in text) % 4) + 1
            return [extra, base + (extra % 2)]

        def decode(self, ids, skip_special_tokens: bool = True) -> str:
            return " ".join(f"tok{i}" for i in ids)

    torch.manual_seed(7)
    model = _CalLM(vocab=32, hidden=12, inter=24, n_layers=3).eval()
    with torch.no_grad():
        emb = model.model.embed_tokens.weight
        emb.zero_()
        for tid in range(32):
            if tid >= 19:  # toxic direction on dim 1 + dim 2
                emb[tid, 1] = 2.0
                emb[tid, 2] = 2.0
            elif tid >= 11:  # benign over-refusal direction on dim 1 positive, dim 2 negative
                emb[tid, 1] = 2.0
                emb[tid, 2] = -2.0
            else:  # everyday normal control on dim 0
                emb[tid, 0] = 2.0
                emb[tid, 1] = -2.0
        for blk in model.model.layers:
            blk.mlp.up_proj.weight.normal_(0, 0.05)
            blk.mlp.down_proj.weight.normal_(0, 0.05)
        model.model.norm.weight.fill_(1.0)
        model.lm_head.weight.normal_(0, 0.2)

    ckpt = tmp_path / "cal_ckpt"
    ckpt.mkdir()
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(ckpt / "model.safetensors"))

    w_l1 = model.model.layers[1].mlp.down_proj.weight.detach()
    u = torch.zeros(12, dtype=torch.float64)
    u[1] = -1.0
    base_patch_path = tmp_path / "base_surgical.moipatch"
    MoIPatch(
        edits=[Edit(module="layers.1.mlp.down_proj", u=u, alpha=1.5, base_sha256=_sha256(w_l1), base_dtype="float32", shape=tuple(w_l1.shape))],
        meta={"name": "abliterated-base"},
    ).save(base_patch_path)

    monkeypatch.setattr(mcp, "_load_model_for_screen", lambda *_a, **_kw: model)
    monkeypatch.setattr(mcp, "_load_tokenizer_for_screen", lambda *_a, **_kw: _CalTok())

    r_audit = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 40,
        "method": "tools/call",
        "params": {
            "name": "wsi_audit_moipatch",
            "arguments": {
                "model_dir": str(ckpt),
                "patch_path": str(base_patch_path),
                "probe_prompts": ["refuse probe 1"],
                "control_prompts": ["normal control 1"],
                "cpu": True,
            },
        },
    })
    assert r_audit["result"]["isError"] is False, r_audit["result"]["content"][0]["text"]
    audit_rep = json.loads(r_audit["result"]["content"][0]["text"])
    assert audit_rep["fingerprint_ok"] is True
    assert audit_rep["restore_exact_ok"] is True
    assert audit_rep["output_logit_diff"]["n_probes"] == 1

    cal_out = tmp_path / "calibrated_sfw.moipatch"
    r_cal = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 41,
        "method": "tools/call",
        "params": {
            "name": "wsi_calibrate_safety_router",
            "arguments": {
                "model_dir": str(ckpt),
                "patch_path": str(base_patch_path),
                "toxic_prompts": [f"toxic prompt {i}" for i in range(6)],
                "benign_refused_prompts": [f"refuse prompt {i}" for i in range(6)],
                "control_prompts": [f"normal control {i}" for i in range(6)],
                "chat": False,
                "n_folds": 3,
                "out_patch_path": str(cal_out),
                "cpu": True,
            },
        },
    })
    assert r_cal["result"]["isError"] is False, r_cal["result"]["content"][0]["text"]
    cal_rep = json.loads(r_cal["result"]["content"][0]["text"])
    assert cal_out.exists()
    assert cal_rep["selected_layer"] >= 1
    assert "dir_thr" in cal_rep["thresholds"] and "veto_thr" in cal_rep["thresholds"]
    assert "invariance_metrics" in cal_rep
    loaded = MoIPatch.load(cal_out)
    assert loaded.router is not None and loaded.router.get("veto_dir") is not None
