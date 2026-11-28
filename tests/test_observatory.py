"""CPU unit tests for WSIObservatory (`wsi/observatory.py`) and `wsi_prescribe` MCP tool."""
from __future__ import annotations

import json

from wsi import mcp_server as mcp
from wsi.observatory import WSIObservatory


def test_observatory_loads_3tomes_and_summarizes():
    obs = WSIObservatory()
    ev = obs.summarize_evidence()
    assert ev["counts"]["tome1_models"] >= 40
    assert ev["counts"]["tome2_pairs"] >= 10
    assert ev["counts"]["tome3_interventions"] >= 10
    assert len(ev["by_architecture"]) >= 4
    assert len(ev["clinical_scorecard"]) >= 5


def test_observatory_query_and_prescribe():
    obs = WSIObservatory()
    gqa = obs.query_models(arch_family="dense_gqa", limit=10)
    assert len(gqa) >= 5
    first_repo = gqa[0]["repo_id"]
    rx = obs.prescribe(first_repo, pathology="over_refusal")
    assert rx["target"]["repo_id"] == first_repo
    assert len(rx["nearest_neighbors"]) >= 1
    assert len(rx["proven_strategies_pass"]) >= 1
    assert len(rx["universal_guardrails"]) >= 2
    assert "step_1_telemetry" in rx["next_2s_probe"]


def test_mcp_exposes_and_runs_wsi_prescribe():
    resp_list = mcp.handle_rpc_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in resp_list["result"]["tools"]}
    assert "wsi_prescribe" in names

    obs = WSIObservatory()
    sample_repo = obs.models[0]["repo_id"]
    resp = mcp.handle_rpc_message({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "wsi_prescribe",
            "arguments": {"target": sample_repo, "pathology": "over_refusal"},
        },
    })
    assert resp["result"]["isError"] is False
    out = json.loads(resp["result"]["content"][0]["text"])
    assert out["target"]["repo_id"] == sample_repo
    assert "proven_strategies_pass" in out and "do_not_do_blacklist" in out
