"""A/B of two vLLM levers on our real Colab workload (levers from sybil-solutions/inference-optimisation-skill).

Workload = what the discovery notebooks run: GSM8K test prompts, Qwen3.5 thinking, our sampling (T 0.6, presence 1.5),
the early loop stop, and the exact engine options of ``discover_math500_blind._llm_fast`` (chunked prefill, prefix
caching, 150 seqs, async scheduling). Every run is a fresh process, like every step of a notebook.

1. **Compile caches across VMs.** The notebooks set ``VLLM_CACHE_ROOT`` on the Drive mount. The torch.compile
   artefacts vLLM keeps there are only part of the warm-up: Inductor (``/tmp/torchinductor_*``), Triton (``~/.triton``)
   and FlashInfer JIT caches stay on the VM and are gone next session, and Drive is a FUSE mount that reads many small
   files slowly. Startup-only runs (engine init + one short prompt):
     ``cold``         every cache empty (first run ever)
     ``warm_local``   every cache warm, on local disk (x2: noise)
     ``drive_cold``   VLLM_CACHE_ROOT on Drive, empty (writes go through FUSE)
     ``drive_now``    VLLM_CACHE_ROOT on Drive, warm; other caches empty = what today's notebooks get on a new VM
     ``drive_full``   every cache restored from one tar on Drive to local disk (copy time included)
2. **FP8 KV cache on a hybrid model.** AGENTS.md turns ``kv_cache_dtype="fp8"`` on for L4/H100. Qwen3.5 has KV in only
   1 layer of 4 (Gated DeltaNet elsewhere), so fp8 saves little memory, and the skill's notes report fp8 KV *halving*
   attention throughput on one kernel. Full generation runs, interleaved ``auto fp8 auto fp8 auto fp8 ...``; fp8 counts
   as a win only if it beats both neighbouring controls on tok/s. Accuracy and loop rate are reported per arm.

Also parsed from each vLLM log: KV cache size, "Maximum concurrency" (do 150 sequences really fit?), torch.compile and
CUDA-graph capture times.

    python scripts/bench_vllm_levers.py prepare --work W --base-dir BASE
    python scripts/bench_vllm_levers.py plan --work W --drive-dir /content/drive/MyDrive/wsi_runs/_bench_vllm_levers
    python scripts/bench_vllm_levers.py report --work W
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

CACHE_VARS = ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
              "XDG_CACHE_HOME")
LOG_PATTERNS = {
    "load_weights_s": r"Loading weights took ([\d.]+) seconds",
    "torch_compile_s": r"torch\.compile takes ([\d.]+) s in total",
    "graph_capture_s": r"Graph capturing finished in ([\d.]+) secs",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) seconds",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
}


def cache_env(local: str | None, vllm_root: str | None = None) -> dict:
    """Env pointing every compile/JIT cache at `local`, except VLLM_CACHE_ROOT at `vllm_root` when given."""
    env = {}
    if local:
        env.update({"VLLM_CACHE_ROOT": f"{local}/vllm", "TORCHINDUCTOR_CACHE_DIR": f"{local}/inductor",
                    "TRITON_CACHE_DIR": f"{local}/triton", "FLASHINFER_WORKSPACE_BASE": f"{local}/flashinfer",
                    "XDG_CACHE_HOME": f"{local}/xdg"})
    if vllm_root:
        env["VLLM_CACHE_ROOT"] = vllm_root
    return env


def parse_log(text: str) -> dict:
    out = {}
    for k, pat in LOG_PATTERNS.items():
        m = re.findall(pat, text)
        if m:
            v = m[-1].replace(",", "")
            out[k] = float(v) if "." in v or k.endswith("_s") or k == "max_concurrency" else int(v)
    m = re.search(r"fp8_kv=(True|False)", text)
    if m:
        out["fp8_kv_used"] = m.group(1) == "True"
    # printed only when torch.compile artefacts come from the cache ("Using cache directory" is printed cold too)
    out["compile_cache_hit"] = bool(re.search(r"Directly load the compiled graph", text))
    return out


def dir_stats(p: str) -> dict:
    n, b = 0, 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                b += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return {"files": n, "mb": round(b / 2**20, 1)}


# ------------------------------------------------------------------------------------------------ prepare
def cmd_prepare(a) -> None:
    import discover_loop_fix as dl

    ns = argparse.Namespace(work=a.work, base_dir=a.base_dir, data_dir=a.data_dir, seed=0, n_elicit=0,
                            elicit_max_new=a.max_new, n_dev=0, dev_max_new=a.max_new, n_test=a.n, max_new=a.max_new)
    dl.cmd_prepare(ns)


# ------------------------------------------------------------------------------------------------ one run (own process)
def cmd_run(a) -> None:
    """One engine start (+ optional full generation). Env (caches, WSI_NO_FP8_KV) is set by the caller."""
    import discover_loop_fix as dl
    import discover_math500_blind as dmb
    import moi_loop_router_vllm as lr

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    reqs = lr.read_jsonl(work / "requests_test.jsonl")
    max_len = max(len(r["prompt_ids"]) + r["max_tokens"] for r in reqs) + 16
    t0 = time.time()
    llm, kill = dmb._llm_fast(cfg["base_dir"], cfg, max_len, "bfloat16", a.gpu_mem)
    init_s = time.time() - t0
    res = {"tag": a.tag, "init_s": round(init_s, 2), "early_loop_stop": kill}
    import discover_fast as fast

    if a.gen:
        t1 = time.time()
        rows, kill = fast._generate(llm, cfg, reqs, kill)
        gen_s = time.time() - t1
        out = f"outputs_{a.tag}.jsonl"
        lr.write_jsonl(work / out, rows)
        ntok = sum(len(r["token_ids"]) for r in rows)
        sc = dl.score(work, "requests_test.jsonl", out)
        res.update({"gen_s": round(gen_s, 2), "tokens": ntok, "tok_s": round(ntok / gen_s, 1), "n": len(rows),
                    "accuracy": round(sc["accuracy"], 4), "loop_killed": round(sc["loop_killed"], 4),
                    "truncated": round(sc["truncated"], 4), "correct_ids": [r["id"] for r in sc["rows"] if r["correct"]]})
    else:
        t1 = time.time()
        fast._generate(llm, cfg, [dict(reqs[0], max_tokens=16)], kill)
        res["first_request_s"] = round(time.time() - t1, 2)
    (work / f"run_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[run {a.tag}] {json.dumps({k: v for k, v in res.items() if k != 'correct_ids'})}", flush=True)


def launch(work: Path, tag: str, env_extra: dict, gen: bool, gpu_mem: float, timeout: int) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in CACHE_VARS and k != "WSI_NO_FP8_KV"}
    env.update(env_extra)
    for k in CACHE_VARS:
        if k in env:
            os.makedirs(env[k], exist_ok=True)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    log_p = work / f"log_{tag}.txt"
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--work", str(work), "--tag", tag,
           "--gpu-mem", str(gpu_mem)] + (["--gen"] if gen else [])
    print(f"\n>>> {tag}  ({'generation' if gen else 'startup only'}; "
          f"{', '.join(f'{k}={v}' for k, v in sorted(env_extra.items()))})", flush=True)
    t0 = time.time()
    with open(log_p, "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = -999
        finally:
            try:  # a leftover EngineCore would hold the GPU for the next run
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                pass
    wall = time.time() - t0
    text = log_p.read_text(encoding="utf-8", errors="replace")
    rj = work / f"run_{tag}.json"
    res = json.loads(rj.read_text()) if rj.exists() and rc == 0 else {"tag": tag, "failed": True, "rc": rc,
                                                                      "log_tail": text[-3000:]}
    res.update({"process_wall_s": round(wall, 1), **parse_log(text), "env": env_extra})
    print(f"    {tag}: wall {wall:.0f}s | init {res.get('init_s')}s | compile {res.get('torch_compile_s')}s | "
          f"graphs {res.get('graph_capture_s')}s | kv {res.get('kv_cache_tokens')} tok | "
          f"max conc {res.get('max_concurrency')}x | tok/s {res.get('tok_s')} | acc {res.get('accuracy')}"
          + (" | FAILED" if res.get("failed") else ""), flush=True)
    return res


# ------------------------------------------------------------------------------------------------ the plan
def cmd_plan(a) -> None:
    work = Path(a.work)
    st_p = work / "plan_state.json"
    state = json.loads(st_p.read_text()) if st_p.exists() else {"runs": {}}

    def save():
        st_p.write_text(json.dumps(state, indent=1))
        if a.drive_dir:
            os.makedirs(a.drive_dir, exist_ok=True)
            shutil.copy2(st_p, Path(a.drive_dir) / "plan_state.json")

    def do(tag, env, gen):
        if tag in state["runs"] and not state["runs"][tag].get("failed"):
            print(f"(skip {tag}: done)")
            return state["runs"][tag]
        r = launch(work, tag, env, gen, a.gpu_mem, a.timeout)
        state["runs"][tag] = r
        save()
        return r

    local = Path(a.local_root)
    fp8_env = {}  # production default (fp8 where the GPU supports it)
    # ---- 1. caches (startup only, production KV setting) ----
    if not a.skip_cache:
        if not (state["runs"].get("cold") and not state["runs"]["cold"].get("failed")):
            shutil.rmtree(local / "c1", ignore_errors=True)
        do("cold", {**cache_env(str(local / "c1")), **fp8_env}, False)
        do("warm_local_1", {**cache_env(str(local / "c1")), **fp8_env}, False)
        do("warm_local_2", {**cache_env(str(local / "c1")), **fp8_env}, False)
        state["cache_sizes"] = {k: dir_stats(str(local / "c1" / k)) for k in ("vllm", "inductor", "triton", "flashinfer", "xdg")}
        if a.drive_dir:
            dv = f"{a.drive_dir}/vllm_root_test"
            if "drive_cold" not in state["runs"]:
                shutil.rmtree(dv, ignore_errors=True)
            # today's notebooks: VLLM_CACHE_ROOT on Drive, everything else on a fresh VM's empty local disk
            do("drive_cold", {**cache_env(str(local / "c2"), vllm_root=dv), **fp8_env}, False)
            shutil.rmtree(local / "c3", ignore_errors=True)
            do("drive_now", {**cache_env(str(local / "c3"), vllm_root=dv), **fp8_env}, False)
            # proposal: one tar of every cache on Drive, unpacked to local disk at the start of the session
            tar_p = f"{a.drive_dir}/compile_cache.tar"
            t0 = time.time()
            with tarfile.open(tar_p, "w") as tf:
                tf.add(str(local / "c1"), arcname="c")
            state["tar_save_s"] = round(time.time() - t0, 1)
            state["tar_mb"] = round(os.path.getsize(tar_p) / 2**20, 1)
            shutil.rmtree(local / "c4", ignore_errors=True)
            t0 = time.time()
            with tarfile.open(tar_p) as tf:
                tf.extractall(str(local / "c4"))
            state["tar_restore_s"] = round(time.time() - t0, 1)
            do("drive_full", {**cache_env(str(local / "c4" / "c")), **fp8_env}, False)
        save()
    # ---- 2. fp8 KV vs auto (full generation, interleaved; warm local caches for both) ----
    if not a.skip_kv:
        warm = cache_env(str(local / "c1"))
        for i in range(a.kv_pairs):
            do(f"kv_auto_{i}", {**warm, "WSI_NO_FP8_KV": "1"}, True)
            do(f"kv_fp8_{i}", warm, True)
        do(f"kv_auto_{a.kv_pairs}", {**warm, "WSI_NO_FP8_KV": "1"}, True)  # closing control
    state["complete"] = True
    save()


# ------------------------------------------------------------------------------------------------ report
def cmd_report(a) -> None:
    work = Path(a.work)
    state = json.loads((work / "plan_state.json").read_text())
    R = state["runs"]
    rep = {"workload": {}, "startup": {}, "kv": {}}
    cfg = json.loads((work / "config.json").read_text())
    rep["workload"] = {"model_dir": cfg.get("base_dir"), "n_prompts": cfg.get("n_test"), "max_new": cfg.get("max_new")}
    for k in ("cache_sizes", "tar_save_s", "tar_mb", "tar_restore_s"):
        if k in state:
            rep["startup"][k] = state[k]

    def g(tag, key):
        r = R.get(tag) or {}
        return None if r.get("failed") else r.get(key)

    for tag in ("cold", "warm_local_1", "warm_local_2", "drive_cold", "drive_now", "drive_full"):
        if tag in R:
            rep["startup"][tag] = {k: R[tag].get(k) for k in ("init_s", "process_wall_s", "torch_compile_s",
                                                              "graph_capture_s", "init_engine_s", "load_weights_s",
                                                              "failed")}
    if "drive_full" in R and "drive_now" in R and g("drive_full", "init_s") and g("drive_now", "init_s"):
        rep["startup"]["saving_per_start_s"] = round(g("drive_now", "process_wall_s")
                                                     - g("drive_full", "process_wall_s"), 1)
        rep["startup"]["note"] = ("drive_full also pays tar_restore_s once per session, not per start; "
                                  "a notebook starts vLLM several times (base, configs, test)")
    autos = sorted(t for t in R if t.startswith("kv_auto_"))
    fp8s = sorted(t for t in R if t.startswith("kv_fp8_"))
    ts = lambda ts_: [g(t, "tok_s") for t in ts_ if g(t, "tok_s")]  # noqa: E731
    if autos and fp8s:
        wins = []
        for t in fp8s:
            i = int(t.rsplit("_", 1)[1])
            left, right, me = g(f"kv_auto_{i}", "tok_s"), g(f"kv_auto_{i + 1}", "tok_s"), g(t, "tok_s")
            if None not in (left, right, me):
                wins.append(me > max(left, right))
        acc = lambda ts_: [g(t, "accuracy") for t in ts_ if g(t, "accuracy") is not None]  # noqa: E731
        lk = lambda ts_: [g(t, "loop_killed") for t in ts_ if g(t, "loop_killed") is not None]  # noqa: E731
        ma, mf = statistics.median(ts(autos)), statistics.median(ts(fp8s))
        rep["kv"] = {
            "auto_tok_s": ts(autos), "fp8_tok_s": ts(fp8s),
            "median_auto": ma, "median_fp8": mf, "fp8_over_auto": round(mf / ma, 4),
            "fp8_beats_both_neighbours": wins,
            "accuracy_auto": acc(autos), "accuracy_fp8": acc(fp8s),
            "loop_killed_auto": lk(autos), "loop_killed_fp8": lk(fp8s),
            "kv_cache_tokens": {"auto": g(autos[0], "kv_cache_tokens"), "fp8": g(fp8s[0], "kv_cache_tokens")},
            "max_concurrency": {"auto": g(autos[0], "max_concurrency"), "fp8": g(fp8s[0], "max_concurrency")},
            "fp8_actually_used": [g(t, "fp8_kv_used") for t in fp8s],
        }
        if wins and all(wins) and mf > 1.03 * ma:
            v = "keep fp8 KV (beats both neighbours every time, >3% faster)"
        elif mf < 0.97 * ma and not any(wins):
            v = "turn fp8 KV OFF for Qwen3.5 (slower than auto in every pair)"
        else:
            v = "no clear difference: keep auto (lossless) unless memory is short"
        rep["kv"]["verdict"] = v
    out = work / "results_bench_vllm_levers.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--base-dir", required=True)
    p.add_argument("--data-dir")
    p.add_argument("--n", type=int, default=150, help="GSM8K test prompts (150 = max_num_seqs, a full batch)")
    p.add_argument("--max-new", type=int, default=4096)
    r = sub.add_parser("run")
    r.add_argument("--work", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--gen", action="store_true")
    r.add_argument("--gpu-mem", type=float, default=0.90)
    q = sub.add_parser("plan")
    q.add_argument("--work", required=True)
    q.add_argument("--drive-dir", default=None, help="Drive folder for the Drive-cache arms and for resume")
    q.add_argument("--local-root", default="/tmp/levers_cache")
    q.add_argument("--kv-pairs", type=int, default=3)
    q.add_argument("--gpu-mem", type=float, default=0.90)
    q.add_argument("--timeout", type=int, default=3600)
    q.add_argument("--skip-cache", action="store_true")
    q.add_argument("--skip-kv", action="store_true")
    s = sub.add_parser("report")
    s.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "run": cmd_run, "plan": cmd_plan, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
