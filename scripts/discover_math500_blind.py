"""Blind loop-fix discovery on MATH-500 using 1-forward-pass `screen_patch_candidates` (no brute-force round1).

First prospective, end-to-end blind test of the Pareto 1-forward-pass screen on a new benchmark (`MATH-500`)
and a larger model (`Qwen/Qwen3.5-4B`):
  1. `prepare`: stratified split of `HuggingFaceH4/MATH-500` (levels 1-5, seed 42):
       - `elicit`: 150 problems (trace capture & blind diagnosis)
       - `dev`:     50 problems (select winner among screen Top-4 ONLY)
       - `test`:   200 problems (held-out exam, untouched until `freeze`)
  2. `elicit`:   base model with `vLLM` + `loop_kill_v2` early loop stop.
  3. `diagnose`: `discover_fast.cmd_diagnose` (v2, Half-A/B held-out d' across all layers).
  4. `screen`:   builds 24-config grid (`k in {1, 2}` x `n in {16, 64, 256, all}` x `alpha in {4, 8, 12}`),
                 ranks all 24 in ~2-3 min via `screen_patch_candidates(rank_by="pareto", max_nll_delta=0.05)`
                 on 6 held-out Half-B looping sequences, selects Top-4 finalists.
  5. `select`:   grades `base` + screen Top-4 on `dev` (50 problems) with in-memory vLLM swaps, picks winner.
  6. `freeze`:   freezes `choice` + code SHA-256 before `test`.
  7. `test` / `report`: 5 baked arms (`base`, `wsi_fixed`, `random_fixed`, `wsi_moi`, `random_moi`) on 200
                 held-out MATH-500 problems with paired 95% CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import discover_blind as blind  # noqa: E402
import discover_fast as fast  # noqa: E402
import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
import screen_fast_grid as sfg  # noqa: E402

# ------------------------------------------------------------------ frozen protocol
GRID_K = (1, 2)
GRID_N = (16, 64, 256, "all")
GRID_A = (4.0, 8.0, 12.0)
SCREEN_TOP_K = 3
N_ELICIT = 100
N_DEV = 32
N_TEST = 150
MAX_TOKENS_DISCOVER = 8192
MAX_TOKENS_TEST = 16384
# Gate 1: stop after elicit when the base model barely hard-loops (the 4B MATH-500 run had 0% at 8k and spent 4 h
# testing a fix for a failure that was not there). The soft rule is no use here: it fires on ~half of normal answers.
MIN_LOOP_KILLED = 0.10
NO_PATCH = {"name": "no_patch", "layers": [], "neurons": 0, "alpha": 0.0, "k": 0}
SEED = 42


def _base_cache_path(cfg: dict, split: str, n: int, max_tokens: int) -> Path | None:
    root = os.environ.get("WSI_BASE_CACHE_DIR")
    if not root:
        return None
    mtag = re.sub(r"[^a-zA-Z0-9._-]+", "_", cfg.get("model_repo", "model"))
    dtag = re.sub(r"[^a-zA-Z0-9._-]+", "_", cfg.get("dataset", "math500"))
    fname = f"{mtag}__{dtag}__{split}_n{n}_tok{max_tokens}_s{cfg.get('seed', SEED)}.jsonl"
    return Path(root) / fname


def _try_load_base_cache(work: Path, cfg: dict, split: str, n: int, max_tokens: int, out_name: str) -> bool:
    dst = work / out_name
    if dst.exists():
        return True
    cp = _base_cache_path(cfg, split, n, max_tokens)
    if cp is not None and cp.exists() and cp.stat().st_size > 0:
        import shutil

        shutil.copy2(cp, dst)
        print(f"[cache] Reused base outputs from Drive cache: {cp.name}", flush=True)
        return True
    return False


def _save_base_cache(work: Path, cfg: dict, split: str, n: int, max_tokens: int, out_name: str) -> None:
    src = work / out_name
    cp = _base_cache_path(cfg, split, n, max_tokens)
    if cp is None or not src.exists():
        return
    import shutil

    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".jsonl.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, cp)
    print(f"[cache] Saved base outputs to Drive cache: {cp.name}", flush=True)


def _fp8_kv_ok() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    maj, mn = torch.cuda.get_device_capability(0)
    return (maj > 8) or (maj == 8 and mn >= 9)  # L4 (8.9), H100 (9.0+)


def _llm_fast(model: str, cfg: dict, max_len: int, dtype: str, gpu_mem: float):
    """Like discover_fast._llm, plus FP8 KV-cache & chunked prefill on L4/H100 with automatic safe fallback."""
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(HERE)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    import loop_kill_v2
    from vllm import LLM

    base_kw = dict(
        model=model,
        tokenizer=cfg["base_dir"],
        dtype=dtype,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem,
        enable_prefix_caching=True,
        seed=cfg["seed"],
    )
    extras: dict = {
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "enable_chunked_prefill": True,
        "max_num_seqs": 150,
        "max_num_batched_tokens": 16384,
        "async_scheduling": True,
    }
    if _fp8_kv_ok() and os.environ.get("WSI_NO_FP8_KV", "0") != "1":
        extras["kv_cache_dtype"] = "fp8"
    proc = loop_kill_v2.processor()
    if proc is not None:
        extras["logits_processors"] = [proc]

    # Try full fast config first; progressively drop optional speed flags, keeping logits_processors as long as possible
    drop_order = [
        [],
        ["async_scheduling"],
        ["async_scheduling", "kv_cache_dtype"],
        ["async_scheduling", "kv_cache_dtype", "max_num_batched_tokens", "max_num_seqs"],
        ["async_scheduling", "kv_cache_dtype", "max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill"],
        ["async_scheduling", "kv_cache_dtype", "max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill", "limit_mm_per_prompt"],
        ["async_scheduling", "kv_cache_dtype", "max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill", "logits_processors"],
        ["async_scheduling", "kv_cache_dtype", "max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill", "limit_mm_per_prompt", "logits_processors"],
    ]
    for drop in drop_order:
        kw = {**base_kw, **{k: v for k, v in extras.items() if k not in drop}}
        try:
            llm = LLM(**kw)
            kill = "logits_processors" in kw
            print(
                f"[gen-fast] vLLM options: fp8_kv={'kv_cache_dtype' in kw} "
                f"chunked={'enable_chunked_prefill' in kw} async={'async_scheduling' in kw} "
                f"max_seqs={kw.get('max_num_seqs', 'default')} early_loop_stop={kill}",
                flush=True,
            )
            return llm, kill
        except Exception as e:
            print(f"[gen-fast] vLLM refused {sorted(set(extras) - set(drop))}: {type(e).__name__}: {str(e)[:260]}",
                  flush=True)
            import gc

            gc.collect()
    raise SystemExit("vLLM could not start")


def _run_parallel_fast(work: Path, groups: list[dict], a) -> None:
    """Runs discover_fast.run_parallel with our FP8-KV-enabled _llm_fast patched in."""
    orig_llm = fast._llm
    fast._llm = _llm_fast
    try:
        # In-process single-GPU fast path avoids subprocess losing the _llm_fast monkeypatch
        import torch

        ngpu = max(1, torch.cuda.device_count()) if not a.cpu else 1
        if ngpu == 1 and len(groups) == 1:
            g = groups[0]
            jf = work / "jobs_fast_0.json"
            jf.write_text(json.dumps(g["jobs"], indent=1))
            ns = argparse.Namespace(
                work=str(work),
                jobs=str(jf),
                model=str(g["model"]) if g.get("model") else None,
                dtype=a.dtype,
                gpu_mem=a.gpu_mem,
                ck_dir=getattr(a, "ck_root", None),
            )
            fast.cmd_gen_jobs(ns)
            return
        fast.run_parallel(work, groups, a)
    finally:
        fast._llm = orig_llm


def math_grid(rank: list[int]) -> list[dict]:
    out = []
    for k in GRID_K:
        for n in GRID_N:
            for al in GRID_A:
                layers = sorted(rank[:k])
                out.append({
                    "name": blind.cfg_name(layers, n, al),
                    "layers": layers,
                    "neurons": n,
                    "alpha": al,
                    "k": k,
                })
    return out


def code_hashes() -> dict[str, str]:
    files = (
        "discover_math500_blind.py",
        "discover_fast.py",
        "screen_fast_grid.py",
        "loop_kill_v2.py",
        "discover_blind.py",
        "discover_loop_fix.py",
        "moi_loop_router_vllm.py",
    )
    return {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in files}


# ------------------------------------------------------------------ MATH-500 grading (same loop-kill rule as discover_fast)
def _extract_boxed(text: str) -> str | None:
    if "</think>" in text:
        after = text.split("</think>")[-1]
        b = _extract_boxed_raw(after)
        if b is not None:
            return b
    return _extract_boxed_raw(text)


def _extract_boxed_raw(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        idx = text.rfind("\\fbox{")
        if idx < 0:
            return None
        start = idx + 6
    else:
        start = idx + 7
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def _norm_latex(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "%")
    s = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\:|\\quad|\\qquad|\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}",
               lambda m: m.group(1) or m.group(2) or "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\s+", "", s)
    if s.endswith("."):
        s = s[:-1]
    return s


def _math_equal(pred: str | None, gold: str | None) -> bool:
    np_s, ng_s = _norm_latex(pred), _norm_latex(gold)
    if not np_s or not ng_s:
        return False
    if np_s == ng_s:
        return True
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        ep = parse_latex(pred.strip())
        eg = parse_latex(gold.strip())
        if sympy.simplify(ep - eg) == 0:
            return True
    except Exception:
        pass
    try:
        fp = float(np_s.replace(",", ""))
        fg = float(ng_s.replace(",", ""))
        if abs(fp - fg) <= 1e-6 * max(1.0, abs(fg)):
            return True
    except Exception:
        pass
    return False


def grade_math(token_ids: list[int], finish_reason: str, gold: str, tok, eos: set[int], max_tokens: int) -> dict:
    body = lr.strip_eos(token_ids, eos)
    hit = lr.first_window_hit(lr.dup_flags(body), lr.KILL_WINDOW, lr.KILL_FRAC)
    looped_killed = hit is not None
    cut = body[:hit] if looped_killed else body
    truncated = (not looped_killed) and (finish_reason == "length" or len(body) >= max_tokens)
    text = tok.decode(cut, skip_special_tokens=True)
    pred = _extract_boxed(text)
    correct = (not looped_killed) and _math_equal(pred, gold)
    return {
        "correct": bool(correct),
        "looped_killed": bool(looped_killed),
        "truncated": bool(truncated),
        "pred": pred,
        "gold": gold,
        "n_tokens": len(body),
    }


def score_math(work: Path, req_file: str, out_file: str, tok=None) -> dict:
    from transformers import AutoTokenizer

    cfg = dl.cfg_load(work)
    tok = tok or AutoTokenizer.from_pretrained(cfg["base_dir"])
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / req_file)}
    rows = []
    for o in lr.read_jsonl(work / out_file):
        r = reqs[o["id"]]
        rows.append({
            "id": o["id"],
            "level": r.get("level", 0),
            **grade_math(o["token_ids"], o["finish_reason"], r["gold"], tok, eos, r["max_tokens"]),
        })
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "accuracy": sum(x["correct"] for x in rows) / n,
        "loop_killed": sum(x["looped_killed"] for x in rows) / n,
        "truncated": sum(x["truncated"] for x in rows) / n,
        "rows": rows,
    }


# ------------------------------------------------------------------ commands
def cmd_prepare(a) -> None:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if (work / "config.json").exists() and (work / "requests_test.jsonl").exists():
        print("prepare: already prepared in", work)
        return

    path = Path(hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset"))
    all_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_lvl: dict[int, list[dict]] = defaultdict(list)
    for idx, r in enumerate(all_rows):
        by_lvl[int(r.get("level", 0))].append({"uid": idx, **r})

    rng = random.Random(a.seed)
    lvls = sorted(by_lvl)
    for lv in lvls:
        rng.shuffle(by_lvl[lv])

    def _take_stratified(count: int) -> list[dict]:
        picked: list[dict] = []
        per = count // len(lvls)
        rem = count - per * len(lvls)
        for i, lv in enumerate(lvls):
            take = per + (1 if i >= len(lvls) - rem else 0)
            got = by_lvl[lv][:take]
            picked.extend(got)
            by_lvl[lv] = by_lvl[lv][len(got):]
        # Top up from remaining levels if a smaller level (e.g. Level 1 has 43 items) ran out
        while len(picked) < count:
            avail = [lv for lv in lvls if by_lvl[lv]]
            if not avail:
                break
            for lv in avail:
                if len(picked) >= count:
                    break
                picked.append(by_lvl[lv].pop(0))
        return picked

    elicit_rows = _take_stratified(a.n_elicit)
    dev_rows = _take_stratified(a.n_dev)
    test_rows = _take_stratified(a.n_test)

    tok = AutoTokenizer.from_pretrained(a.base_dir)
    raw_eos = []
    for attr in ("eos_token_id", "eos_token_ids"):
        val = getattr(tok, attr, None)
        if isinstance(val, (list, tuple, set)):
            raw_eos.extend(val)
        elif val is not None:
            raw_eos.append(val)
    eos = sorted({int(x) for x in raw_eos if x is not None})
    if not eos:
        raise RuntimeError("No eos_token_id found in tokenizer")

    def _make_reqs(rows: list[dict], split_seed: int, max_toks: int) -> list[dict]:
        reqs = []
        for i, r in enumerate(rows):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["problem"] + "\nPut your final answer within \\boxed{}."}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            p_ids = tok.encode(prompt, add_special_tokens=False)
            reqs.append({
                "id": i,
                "uid": r["uid"],
                "level": int(r.get("level", 0)),
                "subject": r.get("subject", ""),
                "gold": r["answer"],
                "prompt_ids": p_ids,
                "max_tokens": max_toks,
                "seed": split_seed + i,
            })
        return reqs

    lr.write_jsonl(work / "requests_elicit.jsonl", _make_reqs(elicit_rows, a.seed + 1000, a.max_tokens_discover))
    lr.write_jsonl(work / "requests_dev.jsonl", _make_reqs(dev_rows, a.seed + 2000, a.max_tokens_discover))
    lr.write_jsonl(work / "requests_test.jsonl", _make_reqs(test_rows, a.seed + 3000, a.max_tokens_test))

    cfg = {
        "base_dir": str(Path(a.base_dir).resolve()),
        "model_repo": a.model_repo,
        "dataset": "HuggingFaceH4/MATH-500",
        "eos": eos,
        "seed": a.seed,
        "max_tokens_discover": a.max_tokens_discover,
        "max_tokens_test": a.max_tokens_test,
        "n_elicit": len(elicit_rows),
        "n_dev": len(dev_rows),
        "n_test": len(test_rows),
    }
    (work / "config.json").write_text(json.dumps(cfg, indent=1))
    fast.sync(work)
    print(f"Prepared MATH-500 splits: elicit={len(elicit_rows)} ({a.max_tokens_discover}t), "
          f"dev={len(dev_rows)} ({a.max_tokens_discover}t), test={len(test_rows)} ({a.max_tokens_test}t)")


def cmd_elicit(a) -> None:
    work = Path(a.work)
    cfg = dl.cfg_load(work)
    t0 = time.perf_counter()
    if not _try_load_base_cache(work, cfg, "elicit", cfg["n_elicit"], cfg["max_tokens_discover"], "outputs_elicit.jsonl"):
        _run_parallel_fast(work, [{
            "model": None,
            "jobs": [{"name": "elicit", "requests": "requests_elicit.jsonl", "out": "outputs_elicit.jsonl"}],
        }], a)
        _save_base_cache(work, cfg, "elicit", cfg["n_elicit"], cfg["max_tokens_discover"], "outputs_elicit.jsonl")
    s = score_math(work, "requests_elicit.jsonl", "outputs_elicit.jsonl")
    print(f"elicit (MATH-500): accuracy={s['accuracy']:.1%}, loop_killed={s['loop_killed']:.1%}, truncated={s['truncated']:.1%}")
    fast.note_time(work, "elicit", time.perf_counter() - t0)
    gate = {"loop_killed": s["loop_killed"], "min_loop_killed": a.min_loop_killed, "n": s["n"],
            "passed": s["loop_killed"] >= a.min_loop_killed, "forced": bool(a.force)}
    (work / "gate_elicit.json").write_text(json.dumps(gate, indent=1))
    fast.sync(work)
    if not gate["passed"] and not a.force:
        raise SystemExit(f"STOP: base hard-loops on {s['loop_killed']:.1%} of elicit answers (< {a.min_loop_killed:.0%}), "
                         "so there is too little to fix. Change model / sampling / benchmark, or pass --force.")


def cmd_diagnose(a) -> None:
    work = Path(a.work)
    if (work / "diagnosis_blind.json").exists() and (work / "blind_directions.npz").exists():
        print("diagnose: already done in", work)
        return
    fast.cmd_diagnose(a)
    fast.sync(work)


def cmd_screen(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from wsi.moipatch import Edit, MoIPatch, _dtype_name, _sha256, module_suffix
    from wsi.tracer import screen_patch_candidates

    work = Path(a.work)
    out_p = work / "screen.json"
    if out_p.exists():
        d = json.loads(out_p.read_text())
        print("screen already done; top finalists:", [c["name"] for c in d["finalists"]])
        return

    t0 = time.perf_counter()
    cfg = dl.cfg_load(work)
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    z = np.load(work / "blind_directions.npz")

    healthy, target, spans, seq_ids = sfg.screen_sequences(work)
    if not healthy:
        raise RuntimeError("No held-out Half-B looping sequences found for screening.")
    print(f"Screening 24 configs on {len(healthy)} held-out Half-B looping MATH-500 sequences (ids={seq_ids})...")

    dtype = torch.bfloat16 if fast.bf16_ok() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg["base_dir"], torch_dtype=dtype, device_map="auto")
    model.eval()

    downs = {}
    for name, mod in model.named_modules():
        m = fast.DOWN_RE.search(name + ".weight")
        if m and "mtp" not in name and "visual" not in name:
            downs[int(m.group(1))] = (name, mod)

    grid = math_grid(diag["layer_rank"])
    cfg_by_name = {c["name"]: c for c in grid}
    candidates = {}
    for c in grid:
        edits = []
        for li in c["layers"]:
            name, mod = downs[li]
            w = mod.weight.detach()
            cols = torch.tensor(blind.columns(diag, li, c["neurons"]))
            edits.append(Edit(
                module=module_suffix(name),
                u=torch.from_numpy(blind.direction(diag, z, li, "wsi")),
                alpha=float(c["alpha"]),
                idx=cols,
                base_sha256=_sha256(w),
                base_dtype=_dtype_name(w.dtype),
                shape=tuple(w.shape),
            ))
        candidates[c["name"]] = MoIPatch(edits=edits, meta={"name": c["name"]})

    rows = screen_patch_candidates(
        model,
        candidates,
        healthy_seqs=healthy,
        target_seqs=target,
        target_spans=spans,
        max_nll_delta=sfg.MAX_NLL_DELTA,
        rank_by=sfg.RANK_BY,
    )
    del model
    torch.cuda.empty_cache()

    elapsed = round(time.perf_counter() - t0, 1)
    finalists = [cfg_by_name[r["name"]] for r in rows[:SCREEN_TOP_K]]
    payload = {
        "rule": {
            "rank_by": sfg.RANK_BY,
            "max_nll_delta": sfg.MAX_NLL_DELTA,
            "n_screen_seqs": len(healthy),
            "top_k": SCREEN_TOP_K,
            "grid_size": len(grid),
        },
        "seq_ids": seq_ids,
        "screen_s": elapsed,
        "ranking": rows,
        "finalists": finalists,
    }
    out_p.write_text(json.dumps(payload, indent=1))
    fast.note_time(work, "screen", elapsed)
    fast.sync(work)
    print(f"Screen finished in {elapsed:.1f}s. Top-{SCREEN_TOP_K} finalists:")
    for i, r in enumerate(rows[:8], 1):
        print(f"  #{i:2d} {r['name']:18s} pareto={r['pareto_score']:7.3f} supp={r['target_suppression']:+.4f} "
              f"dNLL={r['delta_healthy_nll']:+.4f} guard={r['passes_nll_guard']}")


def cmd_select(a) -> None:
    work = Path(a.work)
    if (work / "search.json").exists():
        st = json.loads((work / "search.json").read_text())
        print("select: already done, choice =", st["choice"]["name"])
        return
    cfg = dl.cfg_load(work)
    sc = json.loads((work / "screen.json").read_text())
    finalists = sc["finalists"]
    req = "requests_dev.jsonl"

    t0 = time.perf_counter()
    have_dev_base = _try_load_base_cache(
        work, cfg, "dev", cfg["n_dev"], cfg["max_tokens_discover"], "outputs_dev_base.jsonl"
    )
    jobs = []
    if not have_dev_base:
        jobs.append({"name": "base", "requests": req, "out": "outputs_dev_base.jsonl"})
    for c in finalists:
        jobs.append({
            **c,
            "requests": req,
            "out": f"outputs_dev_{c['name']}.jsonl",
            "direction": "wsi",
        })
    jobs = [j for j in jobs if not (work / j["out"]).exists()]
    if jobs:
        _run_parallel_fast(work, [{"model": None, "jobs": jobs}], a)
    if not have_dev_base:
        _save_base_cache(work, cfg, "dev", cfg["n_dev"], cfg["max_tokens_discover"], "outputs_dev_base.jsonl")

    base = score_math(work, req, "outputs_dev_base.jsonl")
    dev_res = {}
    for c in finalists:
        s = score_math(work, req, f"outputs_dev_{c['name']}.jsonl")
        dev_res[c["name"]] = {
            **{k: v for k, v in s.items() if k != "rows"},
            **c,
        }
        print(f"[dev] {c['name']:18s} acc={s['accuracy']:.1%} loops={s['loop_killed']:.1%} "
              f"trunc={s['truncated']:.1%}  (base {base['accuracy']:.1%}, loops {base['loop_killed']:.1%})")

    def _n_key(n_val) -> int:
        return 10**9 if n_val == "all" else int(n_val)

    winner = max(
        dev_res.values(),
        key=lambda r: (r["accuracy"], -r["loop_killed"], -r["truncated"], -_n_key(r["neurons"]), -r["k"]),
    )
    # Gate 2: "no patch" is a valid answer. A finalist must beat base on dev accuracy, else nothing is frozen to test.
    no_patch = winner["accuracy"] <= base["accuracy"] and not a.force_patch
    elapsed = round(time.perf_counter() - t0, 1)
    best = {k: winner[k] for k in ("name", "layers", "neurons", "alpha", "k")}
    st = {
        "base_dev": {k: v for k, v in base.items() if k != "rows"},
        "dev_finalists": dev_res,
        "best_finalist": best,
        "choice": dict(NO_PATCH) if no_patch else best,
        "time_s": {"select": elapsed},
    }
    (work / "search.json").write_text(json.dumps(st, indent=1))
    fast.note_time(work, "select", elapsed)
    fast.sync(work)
    if no_patch:
        print(f"NO PATCH: best finalist {winner['name']} acc={winner['accuracy']:.1%} <= base {base['accuracy']:.1%} on dev; "
              "test runs base only (pass --force-patch to test it anyway)")
    else:
        print(f"Selected winner on dev: {winner['name']} (acc={winner['accuracy']:.1%}, loops={winner['loop_killed']:.1%})")


def cmd_freeze(a) -> None:
    work = Path(a.work)
    st = json.loads((work / "search.json").read_text())
    c = st["choice"]
    p = work / "frozen.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old["choice"] != c:
            raise SystemExit(f"already frozen with a different choice: {old['choice']}")
        print("already frozen:", c["name"])
        return
    t = json.loads((work / "timing_fast.json").read_text()) if (work / "timing_fast.json").exists() else {}
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    disc = t.get("elicit", 0.0) + t.get("diagnose", 0.0) + t.get("screen", 0.0) + t.get("select", 0.0)
    p.write_text(json.dumps({
        "choice": c,
        "code_sha256": code_hashes(),
        "git_commit": a.git_commit,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "discovery_steps_s": {
            "elicit": t.get("elicit"),
            "diagnose": t.get("diagnose"),
            "screen": t.get("screen"),
            "select": t.get("select"),
        },
        "diagnose_capture_s": diag.get("capture_s"),
        "discovery_s": round(disc, 1),
        "discovery_min": round(disc / 60, 1),
    }, indent=1))
    fast.sync(work)
    print(f"frozen: {c['name']}  (total discovery {disc / 60:.1f} min)")


def _run_fused_pair(
    work: Path,
    cfg: dict,
    model_dir: Path,
    fixed_req: str,
    fixed_out: str,
    moi_req: str,
    moi_out: str,
    a,
) -> None:
    """Run fixed + MoI arms for the same baked checkpoint in a single vLLM instance & single batched generate()."""
    need_fixed = not (work / fixed_out).exists()
    need_moi = not (work / moi_out).exists()
    if not need_fixed and not need_moi:
        return

    fixed_reqs = lr.read_jsonl(work / fixed_req) if need_fixed else []
    moi_reqs = lr.read_jsonl(work / moi_req) if need_moi else []
    if need_moi and not moi_reqs:
        lr.write_jsonl(work / moi_out, [])
        need_moi = False
    if not need_fixed and not need_moi:
        fast.sync(work)
        return

    if not (need_fixed and need_moi):
        req_fn, out_fn, name = (fixed_req, fixed_out, "fixed") if need_fixed else (moi_req, moi_out, "moi")
        _run_parallel_fast(work, [{"model": model_dir, "jobs": [{"name": name, "requests": req_fn, "out": out_fn}]}], a)
        fast.sync(work)
        return

    import gc
    import torch

    all_reqs = fixed_reqs + moi_reqs
    max_len = max(len(r["prompt_ids"]) + r["max_tokens"] for r in all_reqs) + 16
    dtype = a.dtype if a.dtype != "auto" else ("bfloat16" if fast.bf16_ok() else "float16")
    t0 = time.perf_counter()
    llm, kill = _llm_fast(str(model_dir), cfg, max_len, dtype, a.gpu_mem)
    try:
        rows, kill = fast._generate(llm, cfg, all_reqs, kill)
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_fixed = len(fixed_reqs)
    fixed_rows, moi_rows = rows[:n_fixed], rows[n_fixed:]
    lr.write_jsonl(work / fixed_out, fixed_rows)
    lr.write_jsonl(work / moi_out, moi_rows)
    fast.sync(work)
    ntok = sum(len(x["token_ids"]) for x in rows)
    dt_ = time.perf_counter() - t0
    print(
        f"[gen fused {fixed_out} + {moi_out}] {len(fixed_rows)}+{len(moi_rows)} seqs, "
        f"{ntok:,} tokens, {dt_ / 60:.1f} min ({ntok / max(dt_, 1e-6):,.0f} tok/s); "
        f"early stop {'on' if kill else 'OFF'}",
        flush=True,
    )


def cmd_test(a) -> None:
    work = Path(a.work)
    cfg = dl.cfg_load(work)
    fr = json.loads((work / "frozen.json").read_text()) if (work / "frozen.json").exists() else None
    if fr is None:
        raise SystemExit("frozen.json missing: run freeze first")
    c = fr["choice"]
    no_patch = c["name"] == NO_PATCH["name"]
    cks = {}
    for kind in (() if no_patch else ("wsi", "random")):
        ck = Path(a.ck_root or work) / blind.cfg_name(c["layers"], c["neurons"], c["alpha"], kind)
        if not ck.exists():
            blind.build_checkpoint(work, c["layers"], c["neurons"], c["alpha"], kind, ck)
        cks[kind] = ck

    req = "requests_test.jsonl"
    t0 = time.perf_counter()

    # 1. Base arm (reused from shared Drive cache if available)
    if not _try_load_base_cache(work, cfg, "test", cfg["n_test"], cfg["max_tokens_test"], "outputs_test_base.jsonl"):
        _run_parallel_fast(work, [{
            "model": None,
            "jobs": [{"name": "base", "requests": req, "out": "outputs_test_base.jsonl"}],
        }], a)
        _save_base_cache(work, cfg, "test", cfg["n_test"], cfg["max_tokens_test"], "outputs_test_base.jsonl")

    if no_patch:
        print("no_patch frozen: base arm only")
        fast.note_time(work, "test", time.perf_counter() - t0)
        fast.sync(work)
        return
    # 2. Plan MoI from base outputs, then run each baked checkpoint's fixed + MoI arms in ONE fused vLLM batch
    dl.cmd_plan_moi(a)
    mreq = "requests_test_moi.jsonl"
    for kind, fixed_out, moi_out in (
        ("wsi", "outputs_test_fast.jsonl", "outputs_test_moi_fast.jsonl"),
        ("random", "outputs_test_random.jsonl", "outputs_test_moi_random.jsonl"),
    ):
        _run_fused_pair(work, cfg, cks[kind], req, fixed_out, mreq, moi_out, a)

    fast.note_time(work, "test", time.perf_counter() - t0)
    fast.sync(work)


def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    fr = json.loads((work / "frozen.json").read_text()) if (work / "frozen.json").exists() else None
    if fr is None:
        raise SystemExit("frozen.json missing: run freeze first")
    if fr["code_sha256"] != code_hashes():
        raise SystemExit("code changed after freeze; refusing to report")

    cfg = dl.cfg_load(work)
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"])
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}

    res: dict = {
        "model": cfg.get("model_repo"),
        "dataset": cfg.get("dataset"),
        "choice": fr["choice"],
        "frozen": {k: fr[k] for k in ("frozen_utc", "git_commit", "code_sha256")},
        "discovery_min": fr["discovery_min"],
        "discovery_steps_s": fr["discovery_steps_s"],
        "arms": {},
        "paired": {},
    }
    rows: dict[str, dict[int, dict]] = {}
    for name, fn in (
        ("base", "outputs_test_base.jsonl"),
        ("wsi_fixed", "outputs_test_fast.jsonl"),
        ("random_fixed", "outputs_test_random.jsonl"),
    ):
        if not (work / fn).exists():  # no_patch: base only
            continue
        s = score_math(work, "requests_test.jsonl", fn, tok)
        res["arms"][name] = {k: s[k] for k in ("n", "accuracy", "loop_killed", "truncated")}
        rows[name] = {r["id"]: r for r in s["rows"]}

    mp = work / "requests_test_moi.jsonl"
    plan = {r["id"]: r for r in lr.read_jsonl(mp)} if mp.exists() else {}
    for name, fn in (
        ("wsi_moi", "outputs_test_moi_fast.jsonl"),
        ("random_moi", "outputs_test_moi_random.jsonl"),
    ):
        if not (work / fn).exists():
            continue
        cont = {o["id"]: o for o in lr.read_jsonl(work / fn)}
        moi = {}
        edit_tok = all_tok = 0
        for qid, r in reqs.items():
            if qid not in cont:
                moi[qid] = rows["base"][qid]
                all_tok += rows["base"][qid]["n_tokens"]
            else:
                o = cont[qid]
                pre = plan[qid]["prompt_ids"][len(r["prompt_ids"]):]
                full = pre + lr.strip_eos(o["token_ids"], eos)
                g = grade_math(full, o["finish_reason"], r["gold"], tok, eos, r["max_tokens"])
                moi[qid] = {"id": qid, "level": r.get("level", 0), **g}
                edit_tok += max(g["n_tokens"] - len(pre), 0)
                all_tok += g["n_tokens"]
        n = len(moi)
        res["arms"][name] = {
            "n": n,
            "accuracy": sum(g["correct"] for g in moi.values()) / n,
            "loop_killed": sum(g["looped_killed"] for g in moi.values()) / n,
            "truncated": sum(g["truncated"] for g in moi.values()) / n,
            "edit_token_share": edit_tok / max(all_tok, 1),
            "router_fired": len(plan) / n,
        }
        rows[name] = moi

    ids = sorted(reqs)
    acc = {k: [rows[k][i]["correct"] for i in ids] for k in rows}
    pairs = [
        ("wsi_fixed", "base"),
        ("wsi_moi", "base"),
        ("wsi_fixed", "random_fixed"),
        ("wsi_moi", "random_moi"),
        ("wsi_moi", "wsi_fixed"),
    ]
    res["paired"] = {f"{x}_vs_{y}": lr.paired_ci(acc[x], acc[y]) for x, y in pairs if x in acc and y in acc}
    res["no_patch"] = fr["choice"]["name"] == NO_PATCH["name"]
    if (work / "gate_elicit.json").exists():
        res["gate_elicit"] = json.loads((work / "gate_elicit.json").read_text())
    res["screen"] = json.loads((work / "screen.json").read_text())
    res["search"] = json.loads((work / "search.json").read_text())
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    diag.pop("neuron_rank", None)
    res["diagnosis"] = diag
    if (work / "timing_fast.json").exists():
        res["timing_s"] = json.loads((work / "timing_fast.json").read_text())

    out = work / "results_discover_math500_blind.json"
    out.write_text(json.dumps(res, indent=1))
    fast.sync(work)
    print(f"\n{'arm':14s} {'acc':>7s} {'loops':>7s} {'trunc':>7s}")
    for k, v in res["arms"].items():
        print(f"{k:14s} {v['accuracy']:7.1%} {v['loop_killed']:7.1%} {v['truncated']:7.1%}")
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]  ({v['a_only']}/{v['b_only']})")
    print("Saved", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--base-dir", required=True)
    p.add_argument("--model-repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--n-elicit", type=int, default=N_ELICIT)
    p.add_argument("--n-dev", type=int, default=N_DEV)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--max-tokens-discover", type=int, default=MAX_TOKENS_DISCOVER)
    p.add_argument("--max-tokens-test", type=int, default=MAX_TOKENS_TEST)
    p.add_argument("--seed", type=int, default=SEED)

    for name in ("elicit", "select", "test"):
        q = sub.add_parser(name)
        q.add_argument("--work", required=True)
        q.add_argument("--gpu-mem", type=float, default=0.92)
        q.add_argument("--dtype", default="auto")
        q.add_argument("--cpu", action="store_true")
        q.add_argument("--ck-root", default=None)
        if name == "elicit":
            q.add_argument("--min-loop-killed", type=float, default=MIN_LOOP_KILLED)
            q.add_argument("--force", action="store_true", help="continue even if the base barely loops")
        if name == "select":
            q.add_argument("--force-patch", action="store_true", help="freeze the best finalist even if it loses to base")

    q = sub.add_parser("diagnose")
    q.add_argument("--work", required=True)
    q.add_argument("--after-onset", type=int, default=768)
    q.add_argument("--max-seq", type=int, default=16384)
    q.add_argument("--min-loop-seqs", type=int, default=20)
    q.add_argument("--min-loop-tokens", type=int, default=500)
    q.add_argument("--dtype", default="auto")
    q.add_argument("--cpu", action="store_true")

    q = sub.add_parser("screen")
    q.add_argument("--work", required=True)

    q = sub.add_parser("freeze")
    q.add_argument("--work", required=True)
    q.add_argument("--git-commit", default="local")

    q = sub.add_parser("report")
    q.add_argument("--work", required=True)

    a = ap.parse_args()
    {
        "prepare": cmd_prepare,
        "elicit": cmd_elicit,
        "diagnose": cmd_diagnose,
        "screen": cmd_screen,
        "select": cmd_select,
        "freeze": cmd_freeze,
        "test": cmd_test,
        "report": cmd_report,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
