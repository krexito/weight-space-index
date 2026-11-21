"""CPU test: scripts/live_agent.py against a fake OpenAI-compatible server (scripted tool calls, no network)."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import live_agent  # noqa: E402

GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
SHELL = str(GIT_BASH) if os.name == "nt" and GIT_BASH.exists() else "bash"


def call(i, name, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


SCRIPT = [
    {"content": "Plan: keep state in the REPL.", "tool_calls": [call(1, "python", code="x = 21")]},
    {"content": None, "tool_calls": [call(2, "python", code="x * 2"), call(3, "python", code="1/0")]},
    {"content": None, "tool_calls": [call(4, "bash", command="echo key=[$SECRET_KEY] > out.txt; cat out.txt")]},
    {"content": "no tools this time", "tool_calls": None},
    {"content": None, "tool_calls": [call(5, "finish", summary="done")]},
]


def serve(seen):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            step = SCRIPT[min(len(seen) - 1, len(SCRIPT) - 1)]
            msg = {"role": "assistant", "content": step["content"]}
            if step["tool_calls"]:
                msg["tool_calls"] = step["tool_calls"]
            out = {"choices": [{"message": msg}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_harness_end_to_end(tmp_path, monkeypatch):
    seen = []
    srv = serve(seen)
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "task.md").write_text("Do the thing.", encoding="utf-8")
    monkeypatch.setenv("SECRET_KEY", "sk-should-not-leak")
    live_agent.main(["--model", "fake", "--workdir", str(work), "--task", str(tmp_path / "task.md"),
                     "--log", str(tmp_path / "log.jsonl"), "--summary", str(tmp_path / "s.json"),
                     "--base-url", f"http://127.0.0.1:{srv.server_port}", "--key-env", "SECRET_KEY",
                     "--time-limit-min", "5", "--max-turns", "20", "--shell", SHELL])
    srv.shutdown()
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["end"] == "finish" and s["turns"] == 5 and s["final_summary"] == "done"
    assert s["prompt_tokens"] == 50 and abs(s["cost"] - 0.005) < 1e-9
    assert (work / "out.txt").read_text().strip() == "key=[]"  # the tool shell never sees the key
    tools = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    res = [e["result"] for e in tools if e.get("type") == "tool"]
    assert res[1].strip() == "42"  # REPL state survived between calls
    assert "ZeroDivisionError" in res[2]
    # the second request carries the tool results with matching ids; the nudge appears after the tool-less reply
    m2 = seen[2]["messages"]
    assert [m["tool_call_id"] for m in m2 if m["role"] == "tool"] == ["c1", "c2", "c3"]
    assert any(m["role"] == "user" and "Continue with the tools" in m["content"] for m in seen[4]["messages"])
    assert all("sk-should-not-leak" not in json.dumps(b) for b in seen[1:])  # nor does the transcript


def test_clip():
    s = "a" * 100 + "b" * 100
    c = live_agent.clip(s, 50)
    assert c.startswith("a" * 18) and c.endswith("b" * 32) and "cut" in c
