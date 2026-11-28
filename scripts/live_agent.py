"""A small, transparent agent harness: any OpenAI-compatible chat model (OpenRouter by default) works on a task with
two tools, a shell and a persistent Python REPL, inside one folder, until it calls `finish` or runs out of time/turns.

Used for the live blind-agent test (notebooks/live_agent_refusal_aion_colab.ipynb): the agent can load the model on
the GPU, look inside it with wsi, try edits, check them and iterate, with no recipe given.

Security: the API key is read from an environment variable by this process only. Tool commands run with an
environment that does not contain it, optionally as another Linux user (--run-as) who cannot read this process.

Everything is logged to --log (JSONL): each model reply, each tool call and its (truncated) output, token usage and
cost when the provider reports it. --summary gets turns, wall time, tokens, cost and why the run ended.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_OUT = 8000

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the work folder (bash). Returns exit code and output (stdout+stderr, "
                       "truncated to the first and last few thousand characters). Each call is a fresh shell.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer", "description": "default 600"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "python",
        "description": "Run Python code in a persistent interpreter (variables, loaded models and GPU tensors survive "
                       "between calls; the working directory is the work folder). A bare expression on the last line "
                       "is printed. Returns printed output and any traceback. If a call times out the interpreter is "
                       "restarted and its state is lost.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"},
            "timeout_s": {"type": "integer", "description": "default 900"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "End the task. Call it once the deliverables are written.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def clip(s: str, n: int = MAX_OUT) -> str:
    if len(s) <= n:
        return s
    head = n * 3 // 8
    return s[:head] + f"\n... [{len(s) - n} characters cut] ...\n" + s[-(n - head):]


def tool_env(secret_vars: list[str]) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in secret_vars}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def as_user(cmd: list[str], run_as: str | None) -> list[str]:
    return ["runuser", "-u", run_as, "--"] + cmd if run_as else cmd


class Repl:
    def __init__(self, workdir: Path, env: dict, run_as: str | None):
        self.workdir, self.env, self.run_as = workdir, env, run_as
        self.p = None
        self.q: queue.Queue = queue.Queue()

    def start(self) -> None:
        self.p = subprocess.Popen(as_user([sys.executable, "-u", str(HERE / "live_agent_repl.py")], self.run_as),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  cwd=self.workdir, env=self.env, bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._read, args=(self.p, self.q), daemon=True).start()

    @staticmethod
    def _read(p, q) -> None:
        for line in p.stdout:
            q.put(line)
        q.put(None)

    def run(self, code: str, timeout: float) -> str:
        if self.p is None or self.p.poll() is not None:
            self.start()
        self.p.stdin.write(json.dumps({"code": code}) + "\n")
        self.p.stdin.flush()
        t_end = time.time() + timeout
        stray = []
        while True:
            try:
                line = self.q.get(timeout=max(0.1, t_end - time.time()))
            except queue.Empty:
                self.kill()
                return f"TIMEOUT after {timeout:.0f} s: the interpreter was restarted, its state is lost."
            if line is None:
                self.p = None
                return "The interpreter died (state lost):\n" + "".join(stray)
            try:
                rep = json.loads(line)
                out = "".join(stray) + rep.get("output", "")
                return out + (("\nERROR:\n" + rep["error"]) if rep.get("error") else "")
            except json.JSONDecodeError:  # output written straight to fd 1 by native code
                stray.append(line)

    def kill(self) -> None:
        if self.p is not None:
            self.p.kill()
            self.p = None


class McpBridge:
    """Persistent stdio JSON-RPC 2.0 client for `python -m wsi mcp`, isolated via `as_user` & `tool_env`."""

    def __init__(self, workdir: Path, env: dict, run_as: str | None):
        self.workdir, self.env, self.run_as = workdir, env, run_as
        self.p = None
        self.q: queue.Queue = queue.Queue()
        self._req_id = 0
        self.tools_schema: list[dict] = []
        self.tool_names: set[str] = set()

    @staticmethod
    def _read(p, q) -> None:
        for line in p.stdout:
            q.put(line)
        q.put(None)

    def _rpc(self, method: str, params: dict | None, timeout: float) -> dict:
        self._req_id += 1
        req_id = self._req_id
        msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        t_end = time.time() + timeout
        stray: list[str] = []
        while True:
            try:
                line = self.q.get(timeout=max(0.1, t_end - time.time()))
            except queue.Empty:
                self.kill()
                raise TimeoutError(f"MCP timeout after {timeout:.0f} s on {method}")
            if line is None:
                self.p = None
                raise RuntimeError("MCP server exited unexpectedly:\n" + "".join(stray))
            try:
                rep = json.loads(line)
            except json.JSONDecodeError:
                stray.append(line)
                continue
            if rep.get("id") == req_id:
                return rep

    def start(self, timeout: float = 60.0) -> None:
        self.p = subprocess.Popen(
            as_user([sys.executable, "-u", "-m", "wsi", "mcp"], self.run_as),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=self.workdir,
            env=self.env,
            bufsize=1,
        )
        self.q = queue.Queue()
        threading.Thread(target=self._read, args=(self.p, self.q), daemon=True).start()
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "live_agent", "version": "0.1"}}, timeout)
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.p.stdin.flush()
        listed = self._rpc("tools/list", {}, timeout)
        mcp_tools = (listed.get("result") or {}).get("tools") or []
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for t in mcp_tools
        ]
        self.tool_names = {t["name"] for t in mcp_tools}

    def call(self, name: str, arguments: dict, timeout: float = 300.0) -> str:
        if self.p is None or self.p.poll() is not None:
            self.start()
        try:
            rep = self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout)
        except Exception as e:
            return f"MCP ERROR: {e}"
        if "error" in rep:
            return f"MCP JSON-RPC ERROR: {rep['error']}"
        res = rep.get("result") or {}
        parts = [c.get("text", "") for c in (res.get("content") or []) if c.get("type") == "text"]
        text = "\n".join(parts)
        return ("MCP TOOL ERROR:\n" + text) if res.get("isError") else text

    def kill(self) -> None:
        if self.p is not None:
            self.p.kill()
            self.p = None


VERIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "verify_patch",
        "description": "Run the fast deterministic dev verifier on a .moipatch inside the work folder.",
        "parameters": {
            "type": "object",
            "properties": {
                "patch_path": {"type": "string", "description": "Relative path to .moipatch inside workdir (e.g. out/patch.moipatch)"},
                "full": {"type": "boolean", "description": "Run on full dev split instead of fast subset (default false)"},
            },
            "required": ["patch_path"],
        },
    },
}


def run_verifier(
    cmd_template: str,
    workdir: Path,
    env: dict,
    run_as: str | None,
    patch_path: str,
    full: bool,
    timeout: float,
) -> str:
    if not patch_path or "\x00" in patch_path:
        return "ERROR: invalid patch_path"
    wd_res = workdir.resolve()
    resolved = (wd_res / patch_path).resolve()
    try:
        resolved.relative_to(wd_res)
    except ValueError:
        return f"ERROR: patch_path escapes workdir: {patch_path}"
    if not resolved.is_file():
        return f"ERROR: patch file not found: {patch_path}"

    raw_tokens = shlex.split(cmd_template, posix=(os.name != "nt"))
    argv: list[str] = []
    has_full_flag = False
    for tok in raw_tokens:
        if os.name == "nt" and len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'"):
            tok = tok[1:-1]
        if tok == "{extra}":
            if full:
                argv.append("--full")
                has_full_flag = True
            continue
        argv.append(tok.replace("{patch}", str(resolved)))
    if full and not has_full_flag:
        argv.append("--full")

    t0 = time.time()
    try:
        r = subprocess.run(
            as_user(argv, run_as), cwd=wd_res, env=env, capture_output=True, text=True, timeout=timeout
        )
        return f"exit {r.returncode} ({time.time() - t0:.1f} s)\n{r.stdout}{r.stderr}"
    except subprocess.TimeoutExpired as e:
        return f"TIMEOUT after {timeout:.0f} s\n{e.stdout or ''}{e.stderr or ''}"


def run_bash(command: str, workdir: Path, env: dict, run_as: str | None, timeout: float, shell: str = "bash") -> str:
    t0 = time.time()
    try:
        r = subprocess.run(as_user([shell, "-lc", command], run_as), cwd=workdir, env=env, capture_output=True,
                           text=True, timeout=timeout)
        return f"exit {r.returncode} ({time.time() - t0:.0f} s)\n{r.stdout}{r.stderr}"
    except subprocess.TimeoutExpired as e:
        return f"TIMEOUT after {timeout:.0f} s\n{e.stdout or ''}{e.stderr or ''}"


def chat(base_url: str, key: str, body: dict, retries: int = 6) -> dict:
    req_body = json.dumps(body).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=req_body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "wsi-live-agent"})
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            if "choices" in out and out["choices"]:
                return out
            err = out.get("error", out)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as e:
            err = getattr(e, "read", lambda: b"")() if isinstance(e, urllib.error.HTTPError) else str(e)
        wait = min(60, 5 * 2 ** i)
        print(f"  API error (try {i + 1}/{retries}): {str(err)[:300]} -> retry in {wait} s", flush=True)
        time.sleep(wait)
    raise RuntimeError("the model API kept failing")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="e.g. deepseek/deepseek-v4.1-flash-1017")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--task", required=True, help="markdown file with the task (sent as the first user message)")
    ap.add_argument("--log", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--time-limit-min", type=float, default=60)
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--run-as", default=None, help="Linux user that runs the tools (Colab: a non-root user)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--shell", default="bash")
    ap.add_argument("--enable-wsi-mcp", action="store_true", help="Expose the 22 WSI MCP tools via `python -m wsi mcp`")
    ap.add_argument("--verifier-cmd", default=None, help="Command template containing `{patch}` for `verify_patch` tool")
    a = ap.parse_args(argv)

    key = os.environ.get(a.key_env)
    if not key:
        raise SystemExit(f"{a.key_env} is not set")
    workdir = Path(a.workdir).resolve()
    env = tool_env([a.key_env])
    repl = Repl(workdir, env, a.run_as)
    mcp = McpBridge(workdir, env, a.run_as) if a.enable_wsi_mcp else None
    tools_list = list(TOOLS)
    if mcp is not None:
        mcp.start()
        tools_list.extend(mcp.tools_schema)
    if a.verifier_cmd:
        tools_list.append(VERIFY_TOOL)

    deadline = time.time() + a.time_limit_min * 60
    extra_notes = []
    if mcp is not None:
        extra_notes.append(f"You also have {len(mcp.tools_schema)} WSI MCP tools (`wsi_*`) connected.")
    if a.verifier_cmd:
        extra_notes.append("You have a deterministic dev verifier tool `verify_patch` to test candidate `.moipatch` files.")
    system = ("You are an autonomous engineer working alone on a GPU machine. Work only through the tools. Be "
              "efficient: plan, run experiments, read their results, and iterate. Nobody will answer questions. "
              f"You have {a.time_limit_min:.0f} minutes and {a.max_turns} tool turns in total. Each user message "
              "after a tool result reports the time left."
              + (" " + " ".join(extra_notes) if extra_notes else ""))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": Path(a.task).read_text(encoding="utf-8")}]
    log = open(a.log, "a", encoding="utf-8")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    tool_counts: dict[str, int] = {}
    mcp_calls = 0
    verify_calls = 0
    t0, turns, nudges, end, final = time.time(), 0, 0, None, None
    warned = False

    def write(ev: dict) -> None:
        log.write(json.dumps({"t": round(time.time() - t0, 1), **ev}, ensure_ascii=False) + "\n")
        log.flush()

    try:
        while end is None:
            if time.time() > deadline:
                end = "time_limit"
                break
            if turns >= a.max_turns:
                end = "turn_limit"
                break
            body = {"model": a.model, "messages": msgs, "tools": tools_list, "temperature": a.temperature,
                    "usage": {"include": True}}
            out = chat(a.base_url, key, body)
            u = out.get("usage") or {}
            for k in ("prompt_tokens", "completion_tokens"):
                usage[k] += int(u.get(k) or 0)
            usage["cost"] += float(u.get("cost") or 0)
            m = out["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls")} | {"role": "assistant"})
            write({"type": "assistant", "content": m.get("content"), "tool_calls": calls, "usage": u})
            if m.get("content"):
                print(f"\n[{(time.time() - t0) / 60:.1f} min] agent: {clip(m['content'], 600)}", flush=True)
            if not calls:
                nudges += 1
                if nudges > 3:
                    end = "no_tool_calls"
                    break
                msgs.append({"role": "user", "content": "Continue with the tools, or call finish if you are done."})
                continue
            for c in calls:
                turns += 1
                name = c["function"]["name"]
                tool_counts[name] = tool_counts.get(name, 0) + 1
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    args, res = {}, f"Could not parse the arguments as JSON: {e}"
                else:
                    left = max(10.0, deadline - time.time())
                    if name == "bash":
                        res = run_bash(args.get("command", ""), workdir, env, a.run_as,
                                       min(float(args.get("timeout_s") or 600), left), a.shell)
                    elif name == "python":
                        res = repl.run(args.get("code", ""), min(float(args.get("timeout_s") or 900), left))
                    elif name == "verify_patch" and a.verifier_cmd:
                        verify_calls += 1
                        res = run_verifier(
                            a.verifier_cmd, workdir, env, a.run_as,
                            str(args.get("patch_path", "")), bool(args.get("full", False)),
                            min(600.0, left),
                        )
                    elif mcp is not None and name in mcp.tool_names:
                        mcp_calls += 1
                        res = mcp.call(name, args, timeout=min(600.0, left))
                    elif name == "finish":
                        end, final = "finish", args.get("summary", "")
                        res = "Finished."
                    else:
                        res = f"Unknown tool {name}"
                shown = name if name == "finish" else f"{name}: {clip(args.get('command') or args.get('code') or json.dumps(args), 300)}"
                print(f"[{(time.time() - t0) / 60:.1f} min, turn {turns}] {shown}\n  -> {clip(res, 500)}", flush=True)
                write({"type": "tool", "name": name, "args": args, "result": clip(res, 20000)})
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": clip(res)})
            if end is None:
                left = (deadline - time.time()) / 60
                note = f"[{left:.0f} min and {a.max_turns - turns} tool turns left]"
                if left < 8 and not warned:
                    warned = True
                    note += " Time is almost up: write your deliverables now, then call finish."
                msgs.append({"role": "user", "content": note})
    finally:
        repl.kill()
        if mcp is not None:
            mcp.kill()
        summ = {"model": a.model, "end": end, "turns": turns, "wall_min": round((time.time() - t0) / 60, 2),
                **usage, "cost": round(usage["cost"], 4), "tool_counts": tool_counts,
                "mcp_calls": mcp_calls, "verify_calls": verify_calls, "final_summary": final}
        Path(a.summary).write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
        write({"type": "end", **summ})
        log.close()
        print("\nagent run:", json.dumps({k: v for k, v in summ.items() if k != "final_summary"}), flush=True)


if __name__ == "__main__":
    main()
