"""Benchmark Base-Draft Speculative Verification & vLLM WSI_SPEC modes on GSM8K.

Two complementary benchmarks (select via ``--mode pytorch|vllm|both``):

1. ``pytorch``: Evaluates WSI's ``speculative_generate_with_base`` on a real HF model
   (default ``Qwen/Qwen3.5-0.8B``) + ``MoIPatch`` (default ``patches/qwen35_0.8b_p50c_loop_fix.moipatch``),
   verifying bit-for-bit exactness against ``greedy_generate_reference`` and measuring
   tokens/forward, acceptance rate, prefix-match reuse, suffix re-alignment, and wall-clock speedup.
2. ``vllm``: Runs isolated vLLM subprocesses for ``WSI_SPEC`` in ``{none, ngram, suffix, mtp}``,
   each executing ``base`` + 2 in-memory weight-swapped patched configs (via ``discover_fast.make_setter``
   + ``reset_prefix_cache()``) with ``loop_kill_v2`` active, reporting ``init_s``, ``gen_s``, ``tok/s``,
   ``early_loop_stop``, and GSM8K ``accuracy`` / ``loop_killed`` parity.

Writes final JSON atomically via ``.tmp`` + ``os.replace``.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import moi_loop_router_vllm as lr  # noqa: E402


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, dst)
    return dst


def load_gsm8k_prompts(model_dir: str, n: int, seed: int = 42) -> tuple[list[dict[str, Any]], list[int]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    pq = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(pq).sample(frac=1.0, random_state=seed).reset_index(drop=True).iloc[:n]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    eos = lr.eos_ids(tok)
    rows: list[dict[str, Any]] = []
    for i, (q, a) in enumerate(zip(df["question"], df["answer"])):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": f"{q}\n{lr.INSTRUCTION}"}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        rows.append({
            "id": i,
            "prompt_ids": [int(x) for x in ids],
            "gold": str(a).split("####")[-1].strip().replace(",", ""),
            "seed": seed * 100_003 + i,
        })
    return rows, eos


def run_pytorch_benchmark(
    model_dir: str,
    patch_path: str,
    reqs: list[dict[str, Any]],
    eos: list[int],
    *,
    max_tokens: int = 512,
    block_size: int = 16,
    presence_penalty: float = 1.5,
    include_sampling: bool = False,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    from wsi.moipatch import MoIPatch
    from wsi.spec_eval import (
        capture_base_drafts,
        greedy_generate_reference,
        speculative_generate_with_base,
    )

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float32
    kw: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if torch.cuda.is_available():
        kw["device_map"] = "auto"

    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kw).eval()
    load_s = round(time.perf_counter() - t_load, 2)
    prompts = [r["prompt_ids"] for r in reqs]
    patch = MoIPatch.load(patch_path)

    print(f"[pytorch] Capturing greedy base drafts ({len(prompts)} prompts, max_tokens={max_tokens})...", flush=True)
    t_cap = time.perf_counter()
    base_drafts = capture_base_drafts(
        model,
        prompts,
        max_new_tokens=max_tokens,
        eos_token_ids=eos,
        temperature=0.0,
        presence_penalty=presence_penalty,
    )
    cap_s = round(time.perf_counter() - t_cap, 2)

    arms: dict[str, Any] = {}
    ref_outputs: list[dict[str, Any]] = []
    prefix_only_outputs: list[dict[str, Any]] = []
    prefix_suffix_outputs: list[dict[str, Any]] = []

    with patch.attach(model) as h:
        h.on()

        print("[pytorch] Running hf_ref (autoregressive greedy)...", flush=True)
        t0 = time.perf_counter()
        for p in prompts:
            ref_outputs.append(
                greedy_generate_reference(
                    model,
                    p,
                    max_new_tokens=max_tokens,
                    eos_token_ids=eos,
                    presence_penalty=presence_penalty,
                )
            )
        ref_s = time.perf_counter() - t0

        print(f"[pytorch] Running spec_prefix_only (block_size={block_size})...", flush=True)
        t0 = time.perf_counter()
        for p, bd in zip(prompts, base_drafts):
            prefix_only_outputs.append(
                speculative_generate_with_base(
                    model,
                    p,
                    bd["token_ids"],
                    max_new_tokens=max_tokens,
                    eos_token_ids=eos,
                    block_size=block_size,
                    use_suffix_realign=False,
                    temperature=0.0,
                    presence_penalty=presence_penalty,
                )
            )
        po_s = time.perf_counter() - t0

        print(f"[pytorch] Running spec_prefix_suffix (block_size={block_size})...", flush=True)
        t0 = time.perf_counter()
        for p, bd in zip(prompts, base_drafts):
            prefix_suffix_outputs.append(
                speculative_generate_with_base(
                    model,
                    p,
                    bd["token_ids"],
                    max_new_tokens=max_tokens,
                    eos_token_ids=eos,
                    block_size=block_size,
                    use_suffix_realign=True,
                    temperature=0.0,
                    presence_penalty=presence_penalty,
                )
            )
        ps_s = time.perf_counter() - t0

    def _summarize(name: str, outs: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
        n = max(len(outs), 1)
        tot_tok = sum(o["n_tokens"] for o in outs)
        tot_fwd = sum(o["n_forward_passes"] for o in outs)
        tot_draft = sum(o.get("total_drafted", 0) for o in outs)
        tot_acc = sum(o.get("total_accepted", 0) for o in outs)
        exact = sum(1 for o, r in zip(outs, ref_outputs) if o["token_ids"] == r["token_ids"]) / n
        return {
            "arm": name,
            "n_prompts": len(outs),
            "total_tokens": tot_tok,
            "total_forwards": tot_fwd,
            "tokens_per_forward": round(tot_tok / max(tot_fwd, 1), 3),
            "acceptance_rate": round(tot_acc / max(tot_draft, 1), 4) if tot_draft else 0.0,
            "mean_prefix_matched": round(sum(o.get("prefix_matched_tokens", 0) for o in outs) / n, 1),
            "total_prefix_accepted": sum(o.get("prefix_accepted", 0) for o in outs),
            "total_suffix_accepted": sum(o.get("suffix_accepted", 0) for o in outs),
            "exact_match_frac": round(exact, 4),
            "elapsed_s": round(wall_s, 2),
            "tok_per_s": round(tot_tok / max(wall_s, 1e-6), 2),
            "speedup_vs_ref": round(ref_s / max(wall_s, 1e-6), 3),
        }

    arms["hf_ref"] = _summarize("hf_ref", ref_outputs, ref_s)
    arms["spec_prefix_only"] = _summarize("spec_prefix_only", prefix_only_outputs, po_s)
    arms["spec_prefix_suffix"] = _summarize("spec_prefix_suffix", prefix_suffix_outputs, ps_s)

    if include_sampling:
        print("[pytorch] Running stochastic sampling arm (T=0.6, top_p=0.95, top_k=20)...", flush=True)
        samp_drafts = capture_base_drafts(
            model,
            prompts,
            max_new_tokens=max_tokens,
            eos_token_ids=eos,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            presence_penalty=presence_penalty,
            save_top_k_probs=20,
            seed=42,
        )
        samp_outs = []
        with patch.attach(model) as h:
            h.on()
            t0 = time.perf_counter()
            for idx, (p, bd) in enumerate(zip(prompts, samp_drafts)):
                samp_outs.append(
                    speculative_generate_with_base(
                        model,
                        p,
                        bd["token_ids"],
                        max_new_tokens=max_tokens,
                        eos_token_ids=eos,
                        block_size=block_size,
                        use_suffix_realign=True,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        presence_penalty=presence_penalty,
                        base_probs=bd["base_probs"],
                        seed=1000 + idx,
                    )
                )
            samp_s = time.perf_counter() - t0
        tot_tok = sum(o["n_tokens"] for o in samp_outs)
        tot_fwd = sum(o["n_forward_passes"] for o in samp_outs)
        tot_draft = sum(o["total_drafted"] for o in samp_outs)
        tot_acc = sum(o["total_accepted"] for o in samp_outs)
        arms["spec_sampling"] = {
            "arm": "spec_sampling",
            "n_prompts": len(samp_outs),
            "total_tokens": tot_tok,
            "total_forwards": tot_fwd,
            "tokens_per_forward": round(tot_tok / max(tot_fwd, 1), 3),
            "acceptance_rate": round(tot_acc / max(tot_draft, 1), 4) if tot_draft else 0.0,
            "mean_prefix_matched": round(sum(o["prefix_matched_tokens"] for o in samp_outs) / max(len(samp_outs), 1), 1),
            "elapsed_s": round(samp_s, 2),
            "tok_per_s": round(tot_tok / max(samp_s, 1e-6), 2),
        }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_dir,
        "patch": str(patch_path),
        "n_prompts": len(prompts),
        "max_tokens": max_tokens,
        "block_size": block_size,
        "load_s": load_s,
        "base_capture_s": cap_s,
        "arms": arms,
    }


def _exec_vllm_child(label: str, code: str, env: dict[str, str], timeout_s: int = 600) -> dict[str, Any]:
    popen_kw: dict[str, Any] = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    if hasattr(os, "setsid"):
        popen_kw["start_new_session"] = True

    p = subprocess.Popen([sys.executable, "-u", "-c", code], **popen_kw)
    timed_out = [False]

    def _kill_tree() -> None:
        try:
            if popen_kw.get("start_new_session"):
                os.killpg(p.pid, signal.SIGKILL)
            else:
                p.kill()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def _watchdog() -> None:
        if p.poll() is None:
            timed_out[0] = True
            print(f"  [vllm:{label}] TIMEOUT after {timeout_s}s", flush=True)
            _kill_tree()

    timer = threading.Timer(timeout_s, _watchdog)
    timer.daemon = True
    timer.start()
    found = None
    tail: list[str] = []
    last_bar = 0.0
    try:
        assert p.stdout is not None
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
                if now - last_bar < 15.0:
                    continue
                last_bar = now
            print(f"  [vllm:{label}] {s}", flush=True)
        rc = p.wait()
    finally:
        timer.cancel()
        _kill_tree()
    if found is not None:
        return found
    if timed_out[0]:
        return {"mode": label, "status": "timeout", "error": " | ".join(tail[-3:])}
    return {"mode": label, "status": "error", "error": f"exit={rc}: " + " | ".join(tail[-5:])}


def run_vllm_mode(
    model_dir: str,
    patch_path: str,
    reqs: list[dict[str, Any]],
    eos: list[int],
    *,
    spec_mode: str,
    max_tokens: int = 2048,
    gpu_mem: float = 0.90,
) -> dict[str, Any]:
    if spec_mode == "suffix" and importlib.util.find_spec("arctic_inference") is None:
        return {"mode": "suffix", "status": "skipped", "reason": "arctic-inference not installed"}

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"wsi_vllm_{spec_mode}_"))
    req_file = tmp_dir / "reqs.json"
    req_file.write_text(json.dumps({"reqs": reqs, "eos": eos}), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    if spec_mode == "none":
        env.pop("WSI_SPEC", None)
    else:
        env["WSI_SPEC"] = spec_mode

    code = f"""
import gc, json, re, time, torch
from pathlib import Path
from safetensors import safe_open
import sys
sys.path.insert(0, {str(HERE)!r})
sys.path.insert(0, {str(ROOT)!r})
import discover_fast as fast
import discover_math500_blind as mb
import moi_loop_router_vllm as lr
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from wsi.moipatch import MoIPatch, apply_edit

data = json.loads(Path({str(req_file)!r}).read_text())
reqs, eos = data["reqs"], data["eos"]
max_len = max(len(r["prompt_ids"]) + {max_tokens} for r in reqs) + 16
cfg = {{"base_dir": {model_dir!r}, "seed": 42, "eos": eos, "sampling": lr.SAMPLING}}
dtype = "bfloat16" if fast.bf16_ok() else "float16"

t0 = time.perf_counter()
try:
    llm, kill = mb._llm_fast({model_dir!r}, cfg, max_len, dtype, {gpu_mem})
except SystemExit as e:
    print("RESULT_JSON:" + json.dumps({{"mode": {spec_mode!r}, "status": "error", "error": str(e)}}))
    raise SystemExit(0)
init_s = time.perf_counter() - t0

patch = MoIPatch.load({patch_path!r})
layers = sorted({{int(re.search(r"layers\\.(\\d+)\\.", e.module).group(1)) for e in patch.edits}})
base_w = fast.load_base_layers(Path({model_dir!r}), set(layers))

jobs = [
    ("base", {{}}),
    ("patch_a6", {{int(re.search(r"layers\\.(\\d+)\\.", e.module).group(1)): apply_edit(base_w[int(re.search(r"layers\\.(\\d+)\\.", e.module).group(1))], e.u, e.idx, float(e.alpha)).float().numpy() for e in patch.edits}}),
    ("patch_a4", {{int(re.search(r"layers\\.(\\d+)\\.", e.module).group(1)): apply_edit(base_w[int(re.search(r"layers\\.(\\d+)\\.", e.module).group(1))], e.u, e.idx, 4.0).float().numpy() for e in patch.edits}}),
]

tok = AutoTokenizer.from_pretrained({model_dir!r}, trust_remote_code=True)
prompts = [TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in reqs]
kill_eos = eos[-1]
sp_list = [
    SamplingParams(**lr.SAMPLING, max_tokens={max_tokens}, seed=r["seed"], stop_token_ids=eos,
                   extra_args={{"loop_kill_eos": kill_eos}} if kill else None)
    for r in reqs
]

arm_stats = {{}}
tot_toks = 0
tot_gen_s = 0.0
for name, upd in jobs:
    if upd:
        llm.apply_model(fast.make_setter(upd))
        llm.reset_prefix_cache()
    tg = time.perf_counter()
    outs = llm.generate(prompts, sp_list, use_tqdm=False)
    gs = time.perf_counter() - tg
    tot_gen_s += gs
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    tot_toks += ntok
    graded = [lr.grade(list(o.outputs[0].token_ids), o.outputs[0].finish_reason, r["gold"], tok, set(eos), {max_tokens})
              for r, o in zip(reqs, outs)]
    n = max(len(graded), 1)
    arm_stats[name] = {{
        "n": len(graded),
        "tokens": ntok,
        "gen_s": round(gs, 2),
        "tok_per_s": round(ntok / max(gs, 1e-6), 1),
        "accuracy": round(sum(g["correct"] for g in graded) / n, 4),
        "loop_killed": round(sum(g["looped_killed"] for g in graded) / n, 4),
    }}

print("RESULT_JSON:" + json.dumps({{
    "mode": {spec_mode!r},
    "status": "ok",
    "early_loop_stop": kill,
    "init_s": round(init_s, 2),
    "gen_s": round(tot_gen_s, 2),
    "total_tokens": tot_toks,
    "tok_per_s": round(tot_toks / max(tot_gen_s, 1e-6), 1),
    "arms": arm_stats,
}}))
"""
    try:
        return _exec_vllm_child(spec_mode, code, env)
    finally:
        for f in tmp_dir.glob("*"):
            f.unlink(missing_ok=True)
        tmp_dir.rmdir()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--patch", default=str(ROOT / "patches" / "qwen35_0.8b_p50c_loop_fix.moipatch"))
    ap.add_argument("--mode", choices=["pytorch", "vllm", "both"], default="both")
    ap.add_argument("--n", type=int, default=16, help="Prompts for PyTorch base-draft benchmark")
    ap.add_argument("--max-tokens", type=int, default=512, help="Max new tokens for PyTorch benchmark")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--include-sampling", action="store_true")
    ap.add_argument("--vllm-n", type=int, default=32, help="Prompts for vLLM WSI_SPEC race")
    ap.add_argument("--vllm-max-tokens", type=int, default=2048)
    ap.add_argument("--vllm-modes", default="none,ngram,suffix,mtp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "base_draft_spec_results.json"))
    args = ap.parse_args()

    out_payload: dict[str, Any] = {
        "model": args.model_dir,
        "patch": args.patch,
        "mode": args.mode,
    }

    if args.mode in ("pytorch", "both"):
        reqs_pt, eos = load_gsm8k_prompts(args.model_dir, n=args.n, seed=args.seed)
        out_payload["pytorch"] = run_pytorch_benchmark(
            args.model_dir,
            args.patch,
            reqs_pt,
            eos,
            max_tokens=args.max_tokens,
            block_size=args.block_size,
            include_sampling=args.include_sampling,
        )
        _atomic_write_json(args.out, out_payload)

    if args.mode in ("vllm", "both"):
        reqs_v, eos = load_gsm8k_prompts(args.model_dir, n=args.vllm_n, seed=args.seed)
        vllm_results = []
        for m in [x.strip() for x in args.vllm_modes.split(",") if x.strip()]:
            print(f"\n>>> Running vLLM WSI_SPEC={m} ...", flush=True)
            r = run_vllm_mode(
                args.model_dir,
                args.patch,
                reqs_v,
                eos,
                spec_mode=m,
                max_tokens=args.vllm_max_tokens,
            )
            vllm_results.append(r)
            out_payload["vllm"] = list(vllm_results)
            _atomic_write_json(args.out, out_payload)
            print(f"<<< vLLM WSI_SPEC={m}: {json.dumps(r)}", flush=True)
        out_payload["vllm"] = vllm_results

    out_p = _atomic_write_json(args.out, out_payload)
    print(f"\nSaved benchmark report -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
