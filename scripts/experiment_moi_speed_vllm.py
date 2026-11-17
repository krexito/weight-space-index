"""Does Compensated Selective MoI actually run faster? Real sliced checkpoints served by vLLM.

The LFM3 run (benchmarks/moi_task_index_lfm3) showed the quality side: a compensated math index keeps 79% GSM8K at
70% of MLP neurons. Its speed was measured with HF generate, which is overhead-bound at that size, so no speedup was
visible. vLLM has no LFM3 support yet, so this runs on Qwen3.5-0.8B, which vLLM serves.

  export   calibrate on N GSM8K-train solution tokens (default 20k), pick each layer's top neurons by `energy`
           (sqrt E[h_i^2] * ||W_down[:, i]||, same share in every layer), refit the kept down_proj columns by ridge
           least squares on the co-activation Gram (W_S = W G[:,S] (G_SS + lam I)^-1), and WRITE a real checkpoint
           with smaller gate/up/down matrices and `intermediate_size` = kept neurons. The unused MTP head is sliced
           to the same width (by down_proj column norm) so every shape matches the config.
           --neurons 2560,2048,1792,1024 keeps exact counts instead (multiples of 128: the first T4 run showed
           2509 and 1075 neurons decoding SLOWER than 1792, because partial matmul tiles waste work).
  bench    one checkpoint in vLLM: decode tok/s at batch 1 (the on-device case: weight reads dominate), throughput
           at batch 64, GSM8K accuracy (200 test problems, greedy, thinking off), checkpoint size.
  report   merge the bench files.

    python experiment_moi_speed_vllm.py export --model Qwen/Qwen3.5-0.8B --out-dir /tmp/ckpt --data-dir /tmp/data
    python experiment_moi_speed_vllm.py bench --model-dir /tmp/ckpt/energy_recon_0.5 --tag 0.5 --res-dir R
    python experiment_moi_speed_vllm.py report --res-dir R
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import time
from pathlib import Path

HUB_FILES = {
    "gsm8k/test.parquet": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    "gsm8k/train.parquet": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
}
PROMPT_SUFFIX = "\n\nSolve step by step. End with: The answer is <number>."


def need(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    repo, fn = HUB_FILES[name]
    got = Path(hf_hub_download(repo, fn, repo_type="dataset", local_dir=data_dir / "_hub"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(got.read_bytes())
    return path


def clean_solution(ans: str) -> str:
    body, final = ans.split("####")
    return re.sub(r"<<[^>]*>>", "", body).strip() + f"\nThe answer is {final.strip()}."


def extract_number(text: str) -> str | None:
    text = text.replace(",", "")
    for pat in (r"\\boxed\{([^}]*)\}", r"answer is[:\s]*\$?\s*(-?\d+(?:\.\d+)?)", r"####\s*(-?\d+(?:\.\d+)?)"):
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            nums = re.findall(r"-?\d+(?:\.\d+)?", m[-1])
            if nums:
                return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def same_number(a: str | None, b: str) -> bool:
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def local_dir(model: str) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model))


# ---------------------------------------------------------------------------------------------------------- export
def cmd_export(a) -> None:
    import pandas as pd
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = local_dir(a.model)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    tok = AutoTokenizer.from_pretrained(src)
    def sync() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize()

    timing: dict = {}
    t_load = time.perf_counter()
    # bf16 where the GPU supports it (the checkpoint's own dtype); fp32 otherwise. Never fp16: these models have large
    # activations that overflow it. Statistics are always accumulated in fp32/fp64.
    cdt = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
    if a.calib_dtype != "auto":
        cdt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.calib_dtype]
    model = AutoModelForCausalLM.from_pretrained(src, dtype=cdt).to(dev).eval()
    sync()
    timing["load_model_s"] = round(time.perf_counter() - t_load, 2)
    mods = dict(model.named_modules())
    downs = {}
    for name in mods:
        m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
        if m and "mtp" not in name:
            downs[int(m.group(1))] = name
    L = len(downs)
    U = mods[downs[0]].weight.shape[1]
    print(f"{L} MLPs x {U} neurons", flush=True)

    calib, n = [], 0
    for q, ans in zip(*(lambda d: (d["question"], d["answer"]))(pd.read_parquet(need(Path(a.data_dir), "gsm8k/train.parquet")))):
        txt = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": clean_solution(ans)}],
                                      tokenize=False, enable_thinking=False)
        calib.append(tok(txt, add_special_tokens=False).input_ids)
        n += len(calib[-1])
        if n >= a.calib_tokens:
            break

    # The Gram matrices of every layer may not fit next to the model (Qwen3.5-4B: 32 x 9216^2 fp32 = 10.9 GB), so the
    # layers are calibrated in groups that fit in free GPU memory; each finished group's Grams go to CPU RAM.
    order = sorted(downs)
    if a.layers_per_pass:
        per = a.layers_per_pass
    elif dev.type == "cuda":
        per = int((torch.cuda.mem_get_info()[0] - 6 * 2**30) // (U * U * 4))
    else:
        per = L
    per = max(1, min(per, L))
    groups = [order[i : i + per] for i in range(0, L, per)]
    Gcpu, sqd = {}, {}
    t0, ntok = time.perf_counter(), 0
    for gi, grp in enumerate(groups):
        st = {li: {"sq": torch.zeros(U, device=dev, dtype=torch.float64),
                   "G": torch.zeros(U, U, device=dev, dtype=torch.float32)} for li in grp}

        def hook(li):
            def f(mod, inp):
                x = inp[0].detach().reshape(-1, U).float()
                st[li]["sq"] += (x * x).sum(0).double()
                st[li]["G"] += x.T @ x
            return f

        hs = [mods[downs[li]].register_forward_pre_hook(hook(li)) for li in grp]
        n_pass = 0
        with torch.no_grad():
            for s in calib:
                model(input_ids=torch.tensor([s], device=dev), use_cache=False)
                n_pass += len(s)
        for h in hs:
            h.remove()
        ntok = n_pass
        for li in grp:
            Gcpu[li], sqd[li] = st[li]["G"].cpu(), st[li]["sq"].cpu()
        del st
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  calibration pass {gi + 1}/{len(groups)}: layers {grp[0]}-{grp[-1]}", flush=True)
    sync()
    timing["calibration_s"] = round(time.perf_counter() - t0, 2)
    timing["calibration_passes"] = len(groups)
    print(f"[time] load model {timing['load_model_s']:.1f}s | calibration: {ntok:,} tokens, {len(groups)} pass(es) in "
          f"{timing['calibration_s']:.1f}s", flush=True)

    # The refit only needs each layer's down_proj weight: keep those on CPU and free the whole model from the GPU, so
    # the fp64 solves have room (Qwen3.5-4B: the 9.3 GB model left too little memory and the refit ran out of it).
    import gc

    downW = {li: mods[d].weight.detach().to("cpu") for li, d in downs.items()}
    del model, mods, hs
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    files = sorted(src.glob("*.safetensors"))
    keys = {}
    for f in files:
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                keys[k] = f

    def ckey(li: int, proj: str, mtp: bool = False) -> str:
        pre = "mtp." if mtp else ""
        cands = [k for k in keys if k.endswith(f"layers.{li}.mlp.{proj}.weight") and k.startswith("mtp.") == mtp]
        if len(cands) != 1:
            raise SystemExit(f"cannot find a unique {pre}layer {li} {proj} key: {cands}")
        return cands[0]

    mtp_layers = sorted({int(m.group(1)) for k in keys for m in [re.match(r"mtp\.layers\.(\d+)\.mlp\.down_proj", k)] if m})
    summary = {"model": a.model, "calib_tokens": ntok, "calib_dtype": str(cdt), "layers": L, "neurons_per_layer": U, "ridge": a.ridge,
               "budgets": {}}
    if a.neurons:  # exact kept-neuron counts, e.g. multiples of 128 so the GPU matmul tiles are full
        targets = [(f"k{int(x)}", int(x)) for x in a.neurons.split(",")]
    else:
        targets = [(f"{float(x):g}", round(float(x) * U)) for x in a.budgets.split(",")]
    for name, k in targets:
        t_rec, t_slice = time.perf_counter(), 0.0
        new = {}
        for li, d in downs.items():
            W = downW[li].to(dev).double()                                # [H, U]
            sq, G = sqd[li].to(dev) / ntok, Gcpu[li].to(dev).double() / ntok
            ix = (sq.sqrt() * W.norm(dim=0)).topk(k).indices.sort().values
            Gss = G[ix][:, ix]
            lam = a.ridge * Gss.diagonal().mean().clamp_min(1e-12)
            Ws = torch.linalg.solve(Gss + lam * torch.eye(k, device=dev, dtype=torch.float64), (W @ G[:, ix]).T).T
            sync()
            t_s = time.perf_counter()
            ixc = ix.cpu()
            with safe_open(str(keys[ckey(li, "gate_proj")]), "pt") as s:
                gate = s.get_tensor(ckey(li, "gate_proj"))
            with safe_open(str(keys[ckey(li, "up_proj")]), "pt") as s:
                up = s.get_tensor(ckey(li, "up_proj"))
            dt = gate.dtype
            new[ckey(li, "gate_proj")] = gate[ixc].contiguous()
            new[ckey(li, "up_proj")] = up[ixc].contiguous()
            new[ckey(li, "down_proj")] = Ws.to("cpu", dt).contiguous()
            t_slice += time.perf_counter() - t_s
        for li in mtp_layers:  # never used for plain decoding; slice so the shapes match intermediate_size
            with safe_open(str(keys[ckey(li, "down_proj", True)]), "pt") as s:
                dn = s.get_tensor(ckey(li, "down_proj", True))
            ix = dn.float().norm(dim=0).topk(k).indices.sort().values
            new[ckey(li, "down_proj", True)] = dn[:, ix].contiguous()
            for p in ("gate_proj", "up_proj"):
                with safe_open(str(keys[ckey(li, p, True)]), "pt") as s:
                    new[ckey(li, p, True)] = s.get_tensor(ckey(li, p, True))[ix].contiguous()

        recon_s = time.perf_counter() - t_rec - t_slice  # selection + ridge refit only (reading weights excluded)
        t_w = time.perf_counter()
        dst = out / f"energy_recon_{name}"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in src.iterdir():
            if f.is_file() and f.suffix != ".safetensors" and f.name != "config.json":
                shutil.copy2(f, dst / f.name)
        cfg = json.loads((src / "config.json").read_text())
        for c in (cfg, cfg.get("text_config", {})):
            if "intermediate_size" in c:
                c["intermediate_size"] = k
        (dst / "config.json").write_text(json.dumps(cfg, indent=2))
        for f in files:
            with safe_open(str(f), "pt") as s:
                meta = s.metadata() or {}
                tens = {kk: new[kk] if kk in new else s.get_tensor(kk) for kk in s.keys()}
            save_file(tens, str(dst / f.name), metadata={**meta, "format": "pt"})
            del tens
        write_s = time.perf_counter() - t_w + t_slice
        summary["budgets"][name] = {"kept_neurons": k, "dir": str(dst),
                                    "time_s": {"select_and_compensate": round(recon_s, 2),
                                               "slice_and_write_checkpoint": round(write_s, 2)},
                                    "bytes": sum(f.stat().st_size for f in dst.glob("*.safetensors"))}
        print(f"[time] {name}: select+compensate {recon_s:.1f}s | slice+write {write_s:.1f}s")
        print(f"wrote {dst}  ({k}/{U} neurons per layer, {summary['budgets'][name]['bytes'] / 2**30:.2f} GiB)",
              flush=True)
    summary["timing"] = timing
    summary["device"] = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    summary["full_bytes"] = sum(f.stat().st_size for f in files)
    summary["full_dir"] = str(src)
    (out / "export.json").write_text(json.dumps(summary, indent=2))


# ----------------------------------------------------------------------------------------------------------- bench
def cmd_bench(a) -> None:
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    md = local_dir(a.model_dir)
    tok = AutoTokenizer.from_pretrained(md)
    dtype = a.dtype
    if dtype == "auto" and torch.cuda.get_device_capability()[0] < 8:
        dtype = "half"  # T4 / P100: no bf16
    kw = dict(model=str(md), dtype=dtype, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0)
    llm = None
    for extra in ({"limit_mm_per_prompt": {"image": 0, "video": 0}}, {}):
        try:
            llm = LLM(**kw, **extra)
            break
        except Exception as e:  # noqa: BLE001
            print("vLLM start failed with", extra, "->", repr(e)[:300], flush=True)
    if llm is None:
        raise SystemExit("vLLM could not start")

    test = pd.read_parquet(need(Path(a.data_dir), "gsm8k/test.parquet")).iloc[: a.n_problems]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + PROMPT_SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for q in test["question"]]

    def timed(ps: list[str], n_new: int) -> float:
        sp = SamplingParams(temperature=0.0, max_tokens=n_new, ignore_eos=True)
        t = time.perf_counter()
        outs = llm.generate(ps, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        return sum(len(o.outputs[0].token_ids) for o in outs) / dt

    timed(prompts[:4], 32)  # warmup (CUDA graphs, kernels)
    b1 = [timed(prompts[i : i + 1], a.decode_tokens) for i in range(a.repeats)]
    thr = [timed(prompts[:a.batch], a.batch_tokens) for _ in range(2)]
    print(f"[{a.tag}] batch 1: {statistics.median(b1):.1f} tok/s   batch {a.batch}: {max(thr):.0f} tok/s", flush=True)

    t = time.perf_counter()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_new), use_tqdm=False)
    gen_s = time.perf_counter() - t
    ok, samples, ntoks = 0, [], 0
    for o, gold in zip(outs, test["answer"]):
        g = gold.split("####")[-1].strip().replace(",", "")
        txt = o.outputs[0].text
        pred = extract_number(txt.split("</think>")[-1])
        ok += same_number(pred, g)
        ntoks += len(o.outputs[0].token_ids)
        if len(samples) < 3:
            samples.append({"gold": g, "pred": pred, "text": txt[-300:]})
    sizes = sum(f.stat().st_size for f in md.glob("*.safetensors"))
    cfg = json.loads((md / "config.json").read_text())
    res = {"tag": a.tag, "model_dir": str(md), "dtype": dtype, "gpu": torch.cuda.get_device_name(),
           "intermediate_size": cfg.get("text_config", cfg).get("intermediate_size"),
           "checkpoint_bytes": sizes,
           "decode_tok_s_batch1": round(statistics.median(b1), 2), "decode_tok_s_batch1_runs": [round(x, 2) for x in b1],
           f"throughput_tok_s_batch{a.batch}": round(max(thr), 1),
           "gsm8k": {"accuracy": ok / len(prompts), "correct": ok, "n": len(prompts), "gen_tokens": ntoks,
                     "wall_s": round(gen_s, 1), "samples": samples}}
    rd = Path(a.res_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"bench_{a.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[{a.tag}] GSM8K {ok}/{len(prompts)} = {ok / len(prompts):.1%}  ({gen_s:.0f}s)", flush=True)


def cmd_report(a) -> None:
    rd = Path(a.res_dir)
    rows = {}
    for f in sorted(rd.glob("bench_*.json")):
        r = json.loads(f.read_text())
        rows[r["tag"]] = r
    exp = Path(a.export_json)
    res = {"export": json.loads(exp.read_text()) if exp.exists() else None, "arms": rows}
    if res["export"] and "timing" in res["export"]:
        t = res["export"]["timing"]
        print(f"\nBuild time on {res['export'].get('device')}: load {t['load_model_s']:.1f}s, calibration "
              f"({res['export']['calib_tokens']:,} tokens) {t['calibration_s']:.1f}s")
        for n, bgt in res["export"]["budgets"].items():
            ts = bgt.get("time_s", {})
            print(f"  {n}: select+compensate {ts.get('select_and_compensate', 0):.1f}s, "
                  f"slice+write {ts.get('slice_and_write_checkpoint', 0):.1f}s")
    if "full" in rows:
        f = rows["full"]
        bk = next(k for k in f if k.startswith("throughput_tok_s_batch"))
        print(f"\n{'arm':6s} {'neurons':>8s} {'size':>8s} {'GSM8K':>7s} {'batch1':>12s} {'batch-N':>12s}")
        for t, r in rows.items():
            print(f"{t:6s} {r['intermediate_size']:8d} {r['checkpoint_bytes'] / 2**30:7.2f}G "
                  f"{r['gsm8k']['accuracy']:7.1%} {r['decode_tok_s_batch1']:7.1f} ({r['decode_tok_s_batch1'] / f['decode_tok_s_batch1']:.2f}x)"
                  f" {r[bk]:7.0f} ({r[bk] / f[bk]:.2f}x)")
        res["speedup_vs_full"] = {t: {"batch1": round(r["decode_tok_s_batch1"] / f["decode_tok_s_batch1"], 3),
                                      "batchN": round(r[bk] / f[bk], 3),
                                      "size": round(r["checkpoint_bytes"] / f["checkpoint_bytes"], 3)}
                                  for t, r in rows.items()}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("Saved", a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--data-dir", required=True)
    e.add_argument("--budgets", default="0.7,0.5,0.3")
    e.add_argument("--neurons", help="exact kept neurons per layer, comma-separated (overrides --budgets)")
    e.add_argument("--calib-tokens", type=int, default=20_000)
    e.add_argument("--ridge", type=float, default=1e-2)
    e.add_argument("--device", default="cuda")
    e.add_argument("--calib-dtype", default="auto", choices=["auto", "bf16", "fp32"])
    e.add_argument("--layers-per-pass", type=int, default=0, help="0 = as many as fit in free GPU memory")
    b = sub.add_parser("bench")
    b.add_argument("--model-dir", required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--res-dir", required=True)
    b.add_argument("--data-dir", required=True)
    b.add_argument("--n-problems", type=int, default=200)
    b.add_argument("--max-new", type=int, default=1024)
    b.add_argument("--decode-tokens", type=int, default=512)
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--batch", type=int, default=64)
    b.add_argument("--batch-tokens", type=int, default=256)
    b.add_argument("--dtype", default="auto")
    b.add_argument("--max-model-len", type=int, default=4096)
    b.add_argument("--gpu-mem", type=float, default=0.85)
    r = sub.add_parser("report")
    r.add_argument("--res-dir", required=True)
    r.add_argument("--export-json", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"export": cmd_export, "bench": cmd_bench, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
