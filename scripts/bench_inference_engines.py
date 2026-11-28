"""L4 inference engine race on the same GSM8K prompts (`Qwen/Qwen3.5-0.8B`; notebook: 32 prompts, max_tokens=2048).

Compares wall-clock speed (`init_s`, `gen_s`, `tok/s`); answers are not graded:
  1. `vllm_base`:  vLLM bf16 + chunked prefill + prefix caching (bf16 KV cache)
  2. `vllm_mtp`:   vLLM + native Qwen3.5 MTP speculative decoding (1-2 draft tokens)
  3. `vllm_ngram`: vLLM + ngram (prompt lookup) speculative decoding
  4. `tensorfold`: ashhart/TensorFold native Zig+CUDA binary (skipped unless NVIDIA driver >= 580 / CUDA 13)
  5. `sglang`:     SGLang Engine, only with --include-sglang (not installed by the notebook)
Each engine runs in its own process group with a timeout; the whole group is killed afterwards, so a leftover
vLLM EngineCore never holds the GPU for the next engine.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def load_prompts(base_dir: str, n: int = 64, seed: int = 42) -> tuple[list[str], list[list[int]], list[int]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    pq = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(pq).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(base_dir)
    raw_eos = []
    for attr in ("eos_token_id", "eos_token_ids"):
        v = getattr(tok, attr, None)
        if isinstance(v, (list, tuple, set)):
            raw_eos.extend(v)
        elif v is not None:
            raw_eos.append(v)
    eos = sorted({int(x) for x in raw_eos if x is not None})

    prompts, prompt_ids = [], []
    for q in df["question"].iloc[:n]:
        p = tok.apply_chat_template(
            [{"role": "user", "content": str(q)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(p)
        prompt_ids.append(tok.encode(p, add_special_tokens=False))
    return prompts, prompt_ids, eos


def _run_vllm_mode(base_dir: str, prompt_ids: list[list[int]], eos: list[int], max_tokens: int,
                   mode: str) -> dict:
    """Runs vLLM in an isolated child process so GPU memory is 100% freed between competitors."""
    code = f"""
import json, time, torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

t0 = time.perf_counter()
base_kw = dict(
    model={base_dir!r},
    dtype="bfloat16",
    max_model_len={max_tokens + 1024},
    gpu_memory_utilization=0.90,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    limit_mm_per_prompt={{"image": 0, "video": 0}},
    seed=42,
)
mode = {mode!r}
try_configs = []
if mode == "vllm_base":
    try_configs = [
        dict(),
    ]
elif mode == "vllm_mtp":
    try_configs = [
        dict(speculative_config={{"method": "mtp", "num_speculative_tokens": 2}}),
        dict(speculative_config={{"method": "mtp", "num_speculative_tokens": 1}}),
    ]
elif mode == "vllm_ngram":
    try_configs = [
        dict(speculative_config={{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}}),
    ]

llm = None
used_extra = None
last_err = None
for ex in try_configs:
    try:
        llm = LLM(**base_kw, **ex)
        used_extra = ex
        break
    except Exception as e:
        last_err = f"{{type(e).__name__}}: {{str(e)[:300]}}"
        import gc; gc.collect()
if llm is None:
    print("RESULT_JSON:" + json.dumps({{"engine": mode, "status": "error", "error": last_err}}))
    raise SystemExit(0)

init_s = time.perf_counter() - t0
data = json.load(open("/tmp/bench_prompts.json"))
prompts = [TokensPrompt(prompt_token_ids=ids) for ids in data["prompt_ids"]]
sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                    max_tokens={max_tokens}, seed=42, stop_token_ids=data["eos"])
tg = time.perf_counter()
outs = llm.generate(prompts, sp)
gen_s = time.perf_counter() - tg
total_toks = sum(len(o.outputs[0].token_ids) for o in outs)
res = {{
    "engine": mode,
    "status": "ok",
    "config": used_extra,
    "n_prompts": len(outs),
    "total_tokens": total_toks,
    "init_s": round(init_s, 1),
    "gen_s": round(gen_s, 1),
    "total_s": round(init_s + gen_s, 1),
    "tok_per_s": round(total_toks / max(gen_s, 1e-6), 1),
}}
print("RESULT_JSON:" + json.dumps(res))
"""
    return _exec_child(mode, code)


def _run_sglang(base_dir: str, max_tokens: int) -> dict:
    code = f"""
import json, time
try:
    import sglang as sgl
except Exception as e:
    print("RESULT_JSON:" + json.dumps({{"engine": "sglang", "status": "skipped", "reason": f"sglang not installed: {{e}}"}}))
    raise SystemExit(0)

t0 = time.perf_counter()
llm = None
last_err = None
for extra in [dict(kv_cache_dtype="fp8_e4m3"), dict()]:
    try:
        llm = sgl.Engine(model_path={base_dir!r}, dtype="bfloat16", context_length={max_tokens + 1024},
                         mem_fraction_static=0.85, **extra)
        used = extra
        break
    except Exception as e:
        last_err = f"{{type(e).__name__}}: {{str(e)[:300]}}"
if llm is None:
    print("RESULT_JSON:" + json.dumps({{"engine": "sglang", "status": "error", "error": last_err}}))
    raise SystemExit(0)

init_s = time.perf_counter() - t0
data = json.load(open("/tmp/bench_prompts.json"))
sp = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_new_tokens={max_tokens})
tg = time.perf_counter()
outs = llm.generate(data["prompts"], sp)
gen_s = time.perf_counter() - tg
total_toks = sum(int(o.get("meta_info", {{}}).get("completion_tokens", 0)) for o in outs)
llm.shutdown()
res = {{
    "engine": "sglang",
    "status": "ok",
    "config": used,
    "n_prompts": len(outs),
    "total_tokens": total_toks,
    "init_s": round(init_s, 1),
    "gen_s": round(gen_s, 1),
    "total_s": round(init_s + gen_s, 1),
    "tok_per_s": round(total_toks / max(gen_s, 1e-6), 1),
}}
print("RESULT_JSON:" + json.dumps(res))
"""
    return _exec_child("sglang", code)


def _nvidia_driver_major() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
        return int(out.split(".")[0])
    except Exception:
        return 0


def _run_tensorfold(base_dir: str, prompts: list[str], max_tokens: int) -> dict:
    drv = _nvidia_driver_major()
    if drv < 580:
        return {
            "engine": "tensorfold",
            "status": "skipped",
            "reason": f"NVIDIA driver is R{drv} (< R580 required by TensorFold CUDA 13 binary)",
        }

    tf_bin = shutil.which("tensorfold-native") or shutil.which("tensorfold")
    if tf_bin is None:
        # Try fetching latest linux-x86_64 release binary from GitHub
        try:
            rel = json.loads(urllib.request.urlopen(
                "https://api.github.com/repos/ashhart/TensorFold/releases/latest", timeout=15
            ).read().decode())
            asset = next(
                (a for a in rel.get("assets", []) if "linux-x86_64" in a["name"] and a["name"].endswith(".tar.gz")),
                None,
            )
            if asset is None:
                return {"engine": "tensorfold", "status": "skipped", "reason": "No linux-x86_64 release asset found"}
            tgz = Path("/tmp/tensorfold.tar.gz")
            urllib.request.urlretrieve(asset["browser_download_url"], tgz)
            dst = Path("/tmp/tensorfold_bin")
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(["tar", "-xzf", str(tgz), "-C", str(dst)])
            cands = list(dst.rglob("tensorfold-native")) + list(dst.rglob("tensorfold"))
            tf_bin = str(cands[0]) if cands else None
        except Exception as e:
            return {"engine": "tensorfold", "status": "skipped", "reason": f"Download failed: {e}"}

    if not tf_bin:
        return {"engine": "tensorfold", "status": "skipped", "reason": "Binary not found after extract"}

    t0 = time.perf_counter()
    srv = subprocess.Popen(
        [tf_bin, "serve", base_dir, "--context", str(max_tokens + 1024), "--parallel", "32", "--port", "8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    ready = False
    for _ in range(90):
        if srv.poll() is not None:
            out_err = (srv.stdout.read() or "")[-500:]
            return {"engine": "tensorfold", "status": "error", "error": f"Server exited ({srv.returncode}): {out_err}"}
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=1)
            ready = True
            break
        except Exception:
            time.sleep(1)
    if not ready:
        srv.terminate()
        return {"engine": "tensorfold", "status": "error", "error": "Server did not become ready within 90s"}

    init_s = time.perf_counter() - t0

    def _one(p: str) -> int:
        body = json.dumps({
            "model": "default",
            "prompt": p,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
        }).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8080/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
        return int(resp.get("usage", {}).get("completion_tokens", 0))

    tg = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            toks = list(ex.map(_one, prompts))
        gen_s = time.perf_counter() - tg
        total_toks = sum(toks)
        return {
            "engine": "tensorfold",
            "status": "ok",
            "n_prompts": len(prompts),
            "total_tokens": total_toks,
            "init_s": round(init_s, 1),
            "gen_s": round(gen_s, 1),
            "total_s": round(init_s + gen_s, 1),
            "tok_per_s": round(total_toks / max(gen_s, 1e-6), 1),
        }
    except Exception as e:
        return {"engine": "tensorfold", "status": "error", "error": str(e)[:300]}
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:
            srv.kill()


def _exec_child(label: str, code: str, timeout_s: int = 480) -> dict:
    import signal
    import threading

    popen_kw: dict = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if hasattr(os, "setsid"):
        popen_kw["start_new_session"] = True  # own process group so killpg kills EngineCore children too

    p = subprocess.Popen([sys.executable, "-u", "-c", code], **popen_kw)
    timed_out = [False]

    def _kill_tree():
        try:
            if popen_kw.get("start_new_session"):
                # the child leads its own group, so the group id is its pid; this also works after the child
                # itself has exited (os.getpgid would fail then) and still reaches orphaned EngineCore processes
                os.killpg(p.pid, signal.SIGKILL)
            else:
                p.kill()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def _watchdog():
        if p.poll() is None:
            timed_out[0] = True
            print(f"  [{label}] TIMEOUT after {timeout_s}s — killing full process group!", flush=True)
            _kill_tree()

    timer = threading.Timer(timeout_s, _watchdog)
    timer.daemon = True
    timer.start()

    found = None
    tail = []
    last_bar = 0.0
    try:
        for line in p.stdout:
            if line.startswith("RESULT_JSON:"):
                found = json.loads(line[len("RESULT_JSON:"):])
                continue
            s = line.rstrip()
            if not s:
                continue
            tail.append(s)
            if len(tail) > 25:
                tail.pop(0)
            if "it/s" in s or "s/it" in s:
                now = time.time()
                if now - last_bar < 10.0:
                    continue
                last_bar = now
            print(f"  [{label}] {s}", flush=True)
        rc = p.wait()
    finally:
        timer.cancel()
        _kill_tree()
    if found is not None:
        return found
    if timed_out[0]:
        return {"engine": label, "status": "timeout", "error": f"Killed process group after {timeout_s}s: " + " | ".join(tail[-3:])}
    return {"engine": label, "status": "error", "error": f"exit={rc}: " + " | ".join(tail[-5:])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-sglang", action="store_true")
    args = ap.parse_args()

    prompts, prompt_ids, eos = load_prompts(args.base_dir, n=args.n)
    Path("/tmp/bench_prompts.json").write_text(json.dumps({
        "prompts": prompts, "prompt_ids": prompt_ids, "eos": eos,
    }))

    runners = [
        ("vllm_base", lambda: _run_vllm_mode(args.base_dir, prompt_ids, eos, args.max_tokens, "vllm_base")),
        ("vllm_mtp", lambda: _run_vllm_mode(args.base_dir, prompt_ids, eos, args.max_tokens, "vllm_mtp")),
        ("vllm_ngram", lambda: _run_vllm_mode(args.base_dir, prompt_ids, eos, args.max_tokens, "vllm_ngram")),
        ("tensorfold", lambda: _run_tensorfold(args.base_dir, prompts, args.max_tokens)),
    ]
    if args.include_sglang:
        runners.append(("sglang", lambda: _run_sglang(args.base_dir, args.max_tokens)))

    results = []
    for label, fn in runners:
        print(f"\n>>> Running {label} ...", flush=True)
        r = fn()
        results.append(r)
        print(f"<<< {label}: {json.dumps(r)}", flush=True)
        gc.collect()

    payload = {
        "model": args.base_dir,
        "n_prompts": args.n,
        "max_tokens": args.max_tokens,
        "nvidia_driver_major": _nvidia_driver_major(),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== INFERENCE ENGINE RACE SUMMARY ===")
    print(f"{'engine':14s} {'status':8s} {'tok/s':>8s} {'gen_s':>7s} {'init_s':>7s} {'total_s':>8s}")
    for r in results:
        if r["status"] == "ok":
            print(f"{r['engine']:14s} {'OK':8s} {r['tok_per_s']:8.1f} {r['gen_s']:7.1f} {r['init_s']:7.1f} {r['total_s']:8.1f}")
        else:
            print(f"{r['engine']:14s} {r['status']:8s}  ({r.get('reason') or r.get('error', '')[:120]})")


if __name__ == "__main__":
    main()
