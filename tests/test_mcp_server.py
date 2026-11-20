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
    assert "wsi_build_moipatch" in names
    assert "wsi_screen_candidates" in names


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
