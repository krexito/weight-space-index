#!/usr/bin/env python3
"""Remote isolated execution bridge for blind Antigravity (or any external) agent runs on Colab GPU.

Runs a minimal stdlib HTTP server inside Colab that:
  1. Authenticates every request with a random 32-byte Bearer token.
  2. Executes bash commands strictly as the unprivileged Linux user `agent` inside `/content/agent`
     (never exposing `/content/wsi`, `/content/work`, or `/root`).
  3. Enforces a hard wall-clock budget (`--time-limit-min`, default 30 min) starting on the first command.
  4. Logs every turn (timestamp, elapsed_s, wall_s, command, exit_code, stdout/stderr sizes) to a JSONL file.
  5. Writes a run summary JSON when finished or when the timer expires.

Usage (inside Colab):
    python scripts/remote_blind_bridge.py serve \
        --workdir /content/agent \
        --log /content/work/wsi_only_remote_log.jsonl \
        --summary /content/work/wsi_only_run.json \
        --time-limit-min 30 --run-as agent --port 8765 --token <TOKEN>

Usage (generate standalone Windows/Linux client `gpu.py`):
    python scripts/remote_blind_bridge.py write-client \
        --url https://<tunnel>.trycloudflare.com --token <TOKEN> --out /path/to/gpu.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _as_user(cmd: list[str], run_as: str | None) -> list[str]:
    if not run_as or os.name == "nt":
        return cmd
    return ["runuser", "-u", run_as, "--", *cmd]


def _agent_env(workdir: Path, run_as: str | None) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN")
    }
    env["PYTHONPATH"] = str(workdir)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if run_as and os.name != "nt":
        env["HOME"] = f"/home/{run_as}"
    return env


class BridgeState:
    def __init__(
        self,
        *,
        workdir: Path,
        log_path: Path,
        summary_path: Path,
        token: str,
        time_limit_s: float,
        run_as: str | None,
        arm: str,
        model_label: str,
    ) -> None:
        self.workdir = workdir
        self.log_path = log_path
        self.summary_path = summary_path
        self.token = token
        self.time_limit_s = float(time_limit_s)
        self.run_as = run_as
        self.arm = arm
        self.model_label = model_label
        self.lock = threading.Lock()
        self.t_start: float | None = None
        self.turns = 0
        self.finished = False
        self.end_reason: str | None = None
        self.final_summary: str | None = None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    def elapsed_s(self) -> float:
        return 0.0 if self.t_start is None else (time.time() - self.t_start)

    def remaining_s(self) -> float:
        return max(0.0, self.time_limit_s - self.elapsed_s())

    def _append_log(self, rec: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_summary(self) -> dict[str, Any]:
        out = {
            "model": self.model_label,
            "arm": self.arm,
            "end": self.end_reason or ("finish" if self.finished else "in_progress"),
            "turns": self.turns,
            "wall_min": round(self.elapsed_s() / 60.0, 2),
            "time_limit_min": round(self.time_limit_s / 60.0, 2),
            "cost": 0.0,
            "final_summary": self.final_summary,
            "saved_utc": _utc_now(),
        }
        tmp = self.summary_path.with_suffix(self.summary_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.summary_path)
        return out

    def exec_command(self, command: str, timeout_s: float = 300.0) -> dict[str, Any]:
        with self.lock:
            if self.t_start is None:
                self.t_start = time.time()
            if self.finished:
                return {
                    "ok": False,
                    "error": f"Session already ended ({self.end_reason}).",
                    "remaining_s": 0.0,
                    "turns": self.turns,
                }
            rem = self.remaining_s()
            if rem <= 0.0:
                self.finished = True
                self.end_reason = "time_limit"
                self.write_summary()
                return {
                    "ok": False,
                    "error": "TIME_LIMIT_REACHED",
                    "remaining_s": 0.0,
                    "turns": self.turns,
                }
            self.turns += 1
            turn_id = self.turns

        eff_timeout = max(5.0, min(float(timeout_s), rem))
        t0 = time.time()
        shell_cmd = ["powershell", "-NoProfile", "-Command", command] if os.name == "nt" else ["bash", "-lc", command]
        argv = _as_user(shell_cmd, self.run_as)
        env = _agent_env(self.workdir, self.run_as)

        timed_out = False
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=eff_timeout,
            )
            rc = int(proc.returncode)
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            rc = 124
            stdout = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
            stderr = (exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))

        wall_s = round(time.time() - t0, 2)
        with self.lock:
            rem_after = self.remaining_s()
            if rem_after <= 0.0:
                self.finished = True
                self.end_reason = "time_limit"
            rec = {
                "turn": turn_id,
                "utc": _utc_now(),
                "t_elapsed_s": round(self.elapsed_s(), 2),
                "wall_s": wall_s,
                "exit_code": rc,
                "timed_out": timed_out,
                "command": command,
                "stdout_chars": len(stdout),
                "stderr_chars": len(stderr),
                "stdout_head": stdout[:2000],
                "stderr_head": stderr[:1000],
            }
            self._append_log(rec)
            self.write_summary()

        # Truncate huge outputs over HTTP while preserving head + tail.
        combined = stdout + (("\n[stderr]\n" + stderr) if stderr else "")
        if len(combined) > 32000:
            combined = combined[:16000] + f"\n... [{len(combined) - 32000} chars truncated] ...\n" + combined[-16000:]

        return {
            "ok": True,
            "turn": turn_id,
            "exit_code": rc,
            "timed_out": timed_out,
            "wall_s": wall_s,
            "elapsed_s": round(self.elapsed_s(), 1),
            "remaining_s": round(rem_after, 1),
            "output": combined,
        }

    def mark_finished(self, summary: str | None = None) -> dict[str, Any]:
        with self.lock:
            if not self.finished:
                self.finished = True
                self.end_reason = "finish"
            if summary:
                self.final_summary = summary
            self._append_log(
                {
                    "turn": self.turns,
                    "utc": _utc_now(),
                    "t_elapsed_s": round(self.elapsed_s(), 2),
                    "event": "finish",
                    "final_summary": summary,
                }
            )
            return self.write_summary()


def make_handler(state: BridgeState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _auth_ok(self) -> bool:
            hdr = self.headers.get("Authorization", "")
            return hdr == f"Bearer {state.token}"

        def do_GET(self) -> None:
            if not self._auth_ok():
                self._send_json(401, {"ok": False, "error": "unauthorized"})
                return
            if self.path == "/status":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "arm": state.arm,
                        "turns": state.turns,
                        "started": state.t_start is not None,
                        "finished": state.finished,
                        "end_reason": state.end_reason,
                        "elapsed_s": round(state.elapsed_s(), 1),
                        "remaining_s": round(state.remaining_s(), 1),
                    },
                )
                return
            self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            if not self._auth_ok():
                self._send_json(401, {"ok": False, "error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": f"bad json: {exc}"})
                return

            if self.path == "/exec":
                cmd = str(body.get("command") or "")
                timeout_s = float(body.get("timeout_s") or 300.0)
                if not cmd.strip():
                    self._send_json(400, {"ok": False, "error": "empty command"})
                    return
                res = state.exec_command(cmd, timeout_s=timeout_s)
                self._send_json(200, res)
                return

            if self.path == "/finish":
                summary = body.get("summary")
                out = state.mark_finished(str(summary) if summary is not None else None)
                self._send_json(200, {"ok": True, "summary": out})
                return

            self._send_json(404, {"ok": False, "error": "not found"})

    return Handler


CLIENT_TEMPLATE = '''#!/usr/bin/env python3
"""Minimal remote GPU runner for blind evaluation.

Usage:
    python gpu.py "ls -la && nvidia-smi"
    python gpu.py --timeout 600 "python train.py"
    python gpu.py --status
    python gpu.py --finish "Short summary of delivered patch"
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

REMOTE_URL = {url!r}
REMOTE_TOKEN = {token!r}


def _req(method: str, path: str, payload: dict | None = None, timeout: float = 630.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        REMOTE_URL.rstrip("/") + path,
        data=data,
        method=method,
        headers={{"Authorization": f"Bearer {{REMOTE_TOKEN}}", "Content-Type": "application/json"}},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[gpu.py HTTP {{exc.code}}] {{body}}", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run commands on the remote GPU sandbox (/content/agent).")
    ap.add_argument("command", nargs="?", help="Bash command to execute inside /content/agent")
    ap.add_argument("--timeout", type=float, default=300.0, help="Per-command timeout in seconds")
    ap.add_argument("--status", action="store_true", help="Show remaining time and turn count")
    ap.add_argument("--finish", metavar="SUMMARY", help="Mark task finished and freeze deliverables")
    args = ap.parse_args()

    if args.status:
        print(json.dumps(_req("GET", "/status", timeout=30.0), indent=2))
        return
    if args.finish is not None:
        print(json.dumps(_req("POST", "/finish", {{"summary": args.finish}}, timeout=30.0), indent=2))
        return
    if not args.command:
        ap.error("Provide a command, --status, or --finish")

    res = _req("POST", "/exec", {{"command": args.command, "timeout_s": args.timeout}}, timeout=args.timeout + 30.0)
    if not res.get("ok"):
        print(f"[REMOTE STOPPED] {{res.get('error')}}", file=sys.stderr)
        sys.exit(1)
    if res.get("output"):
        print(res["output"], end="" if res["output"].endswith("\\n") else "\\n")
    print(
        f"[remote turn={{res['turn']}} exit={{res['exit_code']}} wall={{res['wall_s']}}s "
        f"elapsed={{res['elapsed_s']}}s remaining={{res['remaining_s']}}s]",
        file=sys.stderr,
    )
    sys.exit(int(res.get("exit_code", 0)))


if __name__ == "__main__":
    main()
'''


def render_client_source(url: str, token: str) -> str:
    return CLIENT_TEMPLATE.format(url=url, token=token)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_serve = sub.add_parser("serve", help="Start isolated remote execution server")
    sp_serve.add_argument("--workdir", required=True)
    sp_serve.add_argument("--log", required=True)
    sp_serve.add_argument("--summary", required=True)
    sp_serve.add_argument("--time-limit-min", type=float, default=30.0)
    sp_serve.add_argument("--run-as", default=None)
    sp_serve.add_argument("--arm", default="wsi_only")
    sp_serve.add_argument("--model-label", default="antigravity-blind")
    sp_serve.add_argument("--host", default="127.0.0.1")
    sp_serve.add_argument("--port", type=int, default=8765)
    sp_serve.add_argument("--token", default=None)

    sp_cli = sub.add_parser("write-client", help="Write standalone gpu.py client script")
    sp_cli.add_argument("--url", required=True)
    sp_cli.add_argument("--token", required=True)
    sp_cli.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "write-client":
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(render_client_source(args.url, args.token), encoding="utf-8")
        print(f"Wrote {out_p}")
        return 0

    token = args.token or secrets.token_urlsafe(24)
    state = BridgeState(
        workdir=Path(args.workdir),
        log_path=Path(args.log),
        summary_path=Path(args.summary),
        token=token,
        time_limit_s=float(args.time_limit_min) * 60.0,
        run_as=args.run_as,
        arm=args.arm,
        model_label=args.model_label,
    )
    httpd = ThreadingHTTPServer((args.host, int(args.port)), make_handler(state))
    print(f"[bridge] listening on http://{args.host}:{args.port} | arm={args.arm} | limit={args.time_limit_min}m", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.write_summary()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
