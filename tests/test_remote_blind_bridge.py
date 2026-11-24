from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import remote_blind_bridge as rbb


def test_remote_blind_bridge_state_exec_log_and_finish(tmp_path: Path) -> None:
    workdir = tmp_path / "agent"
    workdir.mkdir()
    log_p = tmp_path / "log.jsonl"
    sum_p = tmp_path / "summary.json"
    token = "test-secret-token-123"

    state = rbb.BridgeState(
        workdir=workdir,
        log_path=log_p,
        summary_path=sum_p,
        token=token,
        time_limit_s=120.0,
        run_as=None,
        arm="wsi_only",
        model_label="antigravity-blind",
    )

    assert state.turns == 0 and state.t_start is None

    res = state.exec_command("echo HELLO_42")
    assert res["ok"] is True
    assert res["turn"] == 1
    assert res["exit_code"] == 0
    assert "HELLO_42" in res["output"]
    assert res["remaining_s"] > 0.0

    summary = state.mark_finished("Delivered patch")
    assert summary["end"] == "finish"
    assert summary["turns"] == 1
    assert summary["final_summary"] == "Delivered patch"

    # Post-finish execution must be refused cleanly.
    res2 = state.exec_command("echo hi")
    assert res2["ok"] is False
    assert "already ended" in res2["error"]

    # Verify JSONL log and atomic summary on disk.
    logs = [json.loads(line) for line in log_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(logs) == 2
    assert logs[0]["turn"] == 1 and logs[0]["exit_code"] == 0 and "HELLO_42" in logs[0]["stdout_head"]
    assert logs[1]["event"] == "finish"
    saved_sum = json.loads(sum_p.read_text(encoding="utf-8"))
    assert saved_sum["end"] == "finish" and saved_sum["turns"] == 1

    # Verify standalone client script renders cleanly and compiles without syntax errors.
    client_p = tmp_path / "gpu.py"
    client_p.write_text(rbb.render_client_source("https://example.trycloudflare.com", token), encoding="utf-8")
    py_compile.compile(str(client_p), doraise=True)
