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
    a = ap.parse_args(argv)

    key = os.environ.get(a.key_env)
    if not key:
        raise SystemExit(f"{a.key_env} is not set")
    workdir = Path(a.workdir).resolve()
    env = tool_env([a.key_env])
    repl = Repl(workdir, env, a.run_as)
    deadline = time.time() + a.time_limit_min * 60
    system = ("You are an autonomous engineer working alone on a GPU machine. Work only through the tools. Be "
              "efficient: plan, run experiments, read their results, and iterate. Nobody will answer questions. "
              f"You have {a.time_limit_min:.0f} minutes and {a.max_turns} tool turns in total. Each user message "
              "after a tool result reports the time left.")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": Path(a.task).read_text(encoding="utf-8")}]
    log = open(a.log, "a", encoding="utf-8")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
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
            body = {"model": a.model, "messages": msgs, "tools": TOOLS, "temperature": a.temperature,
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
                    elif name == "finish":
                        end, final = "finish", args.get("summary", "")
                        res = "Finished."
                    else:
                        res = f"Unknown tool {name}"
                shown = name if name == "finish" else f"{name}: {clip(args.get('command') or args.get('code') or '', 300)}"
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
        summ = {"model": a.model, "end": end, "turns": turns, "wall_min": round((time.time() - t0) / 60, 2),
                **usage, "cost": round(usage["cost"], 4), "final_summary": final}
        Path(a.summary).write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
        write({"type": "end", **summ})
        log.close()
        print("\nagent run:", json.dumps({k: v for k, v in summ.items() if k != "final_summary"}), flush=True)


if __name__ == "__main__":
    main()
