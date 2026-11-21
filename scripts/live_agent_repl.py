"""Persistent Python REPL for scripts/live_agent.py: one JSON request per line on stdin, one JSON reply per line on stdout.

Request {"code": str}. Reply {"output": str, "error": str | None}. State (loaded models, tensors) survives between
requests, so an agent can load a model once and keep working on it.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback

G: dict = {"__name__": "__agent__"}
real_stdout = sys.stdout


def run(code: str) -> dict:
    buf = io.StringIO()
    err = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            try:  # an expression on the last line is echoed, like a notebook cell
                import ast

                tree = ast.parse(code)
                last = tree.body[-1] if tree.body else None
                if isinstance(last, ast.Expr):
                    exec(compile(ast.Module(tree.body[:-1], []), "<agent>", "exec"), G)
                    v = eval(compile(ast.Expression(last.value), "<agent>", "eval"), G)
                    if v is not None:
                        print(repr(v))
                else:
                    exec(compile(tree, "<agent>", "exec"), G)
            except SyntaxError:
                raise
        except BaseException:  # noqa: BLE001 - report everything to the agent, keep the REPL alive
            err = traceback.format_exc(limit=8)
    return {"output": buf.getvalue(), "error": err}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rep = run(req.get("code", ""))
        real_stdout.write(json.dumps(rep) + "\n")
        real_stdout.flush()


if __name__ == "__main__":
    main()
