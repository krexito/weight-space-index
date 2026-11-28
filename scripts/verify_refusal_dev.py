"""Fast deterministic dev-split verifier for blind over-refusal agents (`agent_*` splits only).

Strictly reads ONLY `agent_overrefusal.jsonl`, `agent_toxic.jsonl`, and `agent_controls.jsonl`
(never `test_*` files). Reuses scoring, refusal detection, batching, and `style_check` from
`scripts/eval_refusal_blind.py`.

Subcommands:
  cache-base  Generate and cache base model answers on dev splits with fixed separate batching.
  check       Evaluate a `.moipatch` against the cached base answers and print a concise JSON verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import eval_refusal_blind as erb  # noqa: E402


def _extract_failure_snippets(
    base: dict[str, list[str]],
    arm: dict[str, list[str]],
    res: dict[str, Any],
    prompts: dict[str, list[str]] | None = None,
    limit: int = 3,
) -> dict[str, list[dict[str, str]]]:
    missed_over: list[dict[str, str]] = []
    toxic_leak: list[dict[str, str]] = []
    new_tmpl: list[dict[str, str]] = []

    p_over = (prompts or {}).get("overrefusal") or [""] * len(base["overrefusal"])
    p_tox = (prompts or {}).get("toxic") or [""] * len(base["toxic"])
    tmpl_keys = set((res.get("style") or {}).get("new_templates") or {})

    for pr, b_ans, a_ans in zip(p_over, base["overrefusal"], arm["overrefusal"]):
        b_ref = erb.is_refusal(b_ans)
        a_ref = erb.is_refusal(a_ans)
        if b_ref and a_ref and len(missed_over) < limit:
            missed_over.append({"prompt": pr[:160], "arm_answer": a_ans[:200]})
        elif b_ref and not a_ref and tmpl_keys and len(new_tmpl) < limit:
            op = erb.opening(a_ans)
            if op in tmpl_keys:
                new_tmpl.append({"prompt": pr[:160], "opening": op, "arm_answer": a_ans[:200]})

    for pr, b_ans, a_ans in zip(p_tox, base["toxic"], arm["toxic"]):
        if erb.is_refusal(b_ans) and not erb.is_refusal(a_ans) and len(toxic_leak) < limit:
            toxic_leak.append({"prompt": pr[:160], "arm_answer": a_ans[:200]})

    return {
        "sample_missed_overrefusal": missed_over,
        "sample_toxic_leak": toxic_leak,
        "sample_new_template": new_tmpl,
    }


def verify_answers_dict(
    base: dict[str, list[str]],
    arm: dict[str, list[str]],
    *,
    prompts: dict[str, list[str]] | None = None,
    patch_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score `(base, arm)` purely in Python, surfacing style metrics and concrete failure samples."""
    res = erb.score(base, arm)
    st = res.get("style") or {}
    snippets = _extract_failure_snippets(base, arm, res, prompts=prompts, limit=3)
    verdict = {
        "recovered": res["recovered"],
        "toxic_kept": res["toxic_kept"],
        "controls_identical": res["controls_identical"],
        "over_refusal": res["over_refusal"],
        "base_over_refusal": res["base_over_refusal"],
        "toxic_refusal": res["toxic_refusal"],
        "base_toxic_refusal": res["base_toxic_refusal"],
        "new_over_refusals": res["new_over_refusals"],
        "control_refusals": res["control_refusals"],
        "success": res["success"],
        "style_ok": st.get("style_ok"),
        "new_template_rate": st.get("new_template_rate"),
        "new_templates": st.get("new_templates"),
        "repetitive_rate": st.get("repetitive_rate"),
        "base_repetitive_rate": st.get("base_repetitive_rate"),
        "late_refusal_rate": st.get("late_refusal_rate"),
        "median_words": st.get("median_words"),
        "base_median_words": st.get("base_median_words"),
        "success_with_style": res["success_with_style"],
        **snippets,
    }
    if patch_meta:
        verdict.update(patch_meta)
    return verdict


def _load_dev_prompts(data_dir: Path) -> dict[str, list[str]]:
    out = {}
    for part in ("overrefusal", "toxic", "controls"):
        fp = data_dir / f"agent_{part}.jsonl"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing dev split file: {fp}")
        out[part] = [str(r["prompt"]) for r in erb.read_jsonl(fp)]
    return out


def _slice_prompts(all_p: dict[str, list[str]], n_over: int | None, n_tox: int | None, n_ctrl: int | None) -> dict[str, list[str]]:
    return {
        "overrefusal": all_p["overrefusal"][:n_over] if n_over is not None else list(all_p["overrefusal"]),
        "toxic": all_p["toxic"][:n_tox] if n_tox is not None else list(all_p["toxic"]),
        "controls": all_p["controls"][:n_ctrl] if n_ctrl is not None else list(all_p["controls"]),
    }


def _generate_quiet(model, tok, prompts: list[str], max_new: int, bs: int) -> list[str]:
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):
        return erb.generate(model, tok, prompts, max_new, bs)


def cmd_cache_base(a) -> None:
    data_dir = Path(a.data_dir)
    all_prompts = _load_dev_prompts(data_dir)
    model, tok = erb.load(a.model, a.dtype, a.device)

    fast_prompts = _slice_prompts(all_prompts, a.n_over, a.n_tox, a.n_ctrl)
    t0 = time.perf_counter()
    fast_answers = {
        part: _generate_quiet(model, tok, fast_prompts[part], a.max_new, a.bs)
        for part in ("overrefusal", "toxic", "controls")
    }
    payload: dict[str, Any] = {
        "model": str(a.model),
        "dtype": str(a.dtype),
        "max_new": int(a.max_new),
        "bs": int(a.bs),
        "fast": {
            "counts": {k: len(v) for k, v in fast_prompts.items()},
            "prompts": fast_prompts,
            "answers": fast_answers,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        },
    }
    if a.with_full:
        t1 = time.perf_counter()
        full_answers = {
            part: _generate_quiet(model, tok, all_prompts[part], a.max_new, a.bs)
            for part in ("overrefusal", "toxic", "controls")
        }
        payload["full"] = {
            "counts": {k: len(v) for k, v in all_prompts.items()},
            "prompts": all_prompts,
            "answers": full_answers,
            "elapsed_s": round(time.perf_counter() - t1, 1),
        }

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    erb.atomic_json(out_path, payload)
    print(json.dumps({"cached": str(out_path), "fast": payload["fast"]["counts"],
                      "has_full": "full" in payload}, indent=1))


def cmd_check(a) -> None:
    try:
        from wsi.moipatch import MoIPatch
    except ImportError:
        from moipatch import MoIPatch

    t0 = time.perf_counter()
    cache = json.loads(Path(a.base_cache).read_text(encoding="utf-8"))
    split_key = "full" if a.full else "fast"
    if split_key not in cache:
        if a.full:
            raise KeyError(f"--full requested, but '{a.base_cache}' only cached 'fast'")
        split_key = "full"

    bucket = cache[split_key]
    prompts: dict[str, list[str]] = bucket["prompts"]
    base_ans: dict[str, list[str]] = bucket["answers"]
    max_new = int(a.max_new or cache.get("max_new", 256))
    bs = int(a.bs or cache.get("bs", 16))

    patch_path = Path(a.patch)
    patch = MoIPatch.load(patch_path)
    model, tok = erb.load(a.model, a.dtype, a.device)
    routed = bool(patch.router)

    arm_ans: dict[str, list[str]] = {}
    stats = None
    if routed:
        with patch.enable_routed(model, verify_fingerprint=not a.no_fingerprint) as h:
            for part in ("overrefusal", "toxic", "controls"):
                arm_ans[part] = _generate_quiet(model, tok, prompts[part], max_new, bs)
            st = getattr(h, "stats", None)
            stats = st() if callable(st) else st
    else:
        with patch.attach(model, verify_fingerprint=not a.no_fingerprint):
            for part in ("overrefusal", "toxic", "controls"):
                arm_ans[part] = _generate_quiet(model, tok, prompts[part], max_new, bs)

    patch_meta = {
        "split": split_key,
        "counts": {k: len(v) for k, v in prompts.items()},
        "patch": patch_path.name,
        "patch_kb": round(patch_path.stat().st_size / 1024.0, 1),
        "routed": routed,
        "n_edits": len(patch.edits),
        "modules": [e.module for e in patch.edits],
        "router_stats": stats,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    verdict = verify_answers_dict(base_ans, arm_ans, prompts=prompts, patch_meta=patch_meta)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache-base")
    c.add_argument("--model", required=True)
    c.add_argument("--data-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n-over", type=int, default=40)
    c.add_argument("--n-tox", type=int, default=30)
    c.add_argument("--n-ctrl", type=int, default=15)
    c.add_argument("--with-full", action="store_true", help="Also pre-cache full 200/100/30 dev splits")
    c.add_argument("--dtype", default="bfloat16")
    c.add_argument("--device", default="cuda")
    c.add_argument("--max-new", type=int, default=256)
    c.add_argument("--bs", type=int, default=16)

    v = sub.add_parser("check")
    v.add_argument("--model", required=True)
    v.add_argument("--patch", required=True)
    v.add_argument("--data-dir", required=True)
    v.add_argument("--base-cache", required=True)
    v.add_argument("--full", action="store_true")
    v.add_argument("--no-fingerprint", action="store_true")
    v.add_argument("--dtype", default="bfloat16")
    v.add_argument("--device", default="cuda")
    v.add_argument("--max-new", type=int, default=None)
    v.add_argument("--bs", type=int, default=None)

    a = ap.parse_args(argv)
    {"cache-base": cmd_cache_base, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
