"""Fast blind loop-fix discovery: the method of discover_blind.py, made cheap enough to run on any new model.

The blind run (#13, Qwen3.5-0.8B) found a working fix but took 9.1 h on an L4. WSI's own part (record loops,
rank layers, pick neurons) took about 30 min. The other 8.6 h went into grading 60 configs, for three reasons this
script removes:

  1. Early loop stop was off. The old logits-processor API is gone in vLLM 0.34, so every looping response ran to
     the 16k cap. loop_kill_v2.py stops it at the kill rule (grading cuts there anyway, so scores do not change).
  2. vLLM restarted for every config (about 1 of every 7 minutes). Here one vLLM per GPU stays loaded and the
     edited down_proj matrices are written into it in memory (LLM.apply_model + reset_prefix_cache). Each swap is
     checked with an exact int16 checksum against the same weights a baked checkpoint would hold.
  3. The grid was 60 configs. In #13 the winner was 2 layers x 16 neurons x alpha 12: it held 50.0% from round 1 to
     round 2, while 64 and all neurons fell 9-17 points on the fresh dev problems; 3-layer configs never led. So the
     grid here is k {1, 2} x neurons {16, all} x alpha {4, 8, 12} = 12, and 3 go to round 2.
     This grid is fixed before any run on a new model (it is informed by the 0.8B, so a 0.8B rerun is not blind).

Configs run in parallel on every visible GPU (Kaggle: 2 x T4). The test arms use baked checkpoints in fresh vLLM
processes, exactly like #13, so the swap only affects which config is chosen, never the reported test numbers.

What WSI decides (same as discover_blind.py): which layers (held-out d' ranking of ALL layers), which neurons
(ranked by c_i), and the direction u. The grid only sets how many layers, how many neurons and how strong.

    python discover_fast.py prepare --work W --base-dir B
    python discover_fast.py elicit --work W
    python discover_fast.py diagnose --work W
    python discover_fast.py search --work W          # round 1 (12 configs) + round 2 (top 3) -> choice
    python discover_fast.py freeze --work W
    python discover_fast.py test --work W            # base / fixed / random, then MoI arms
    python discover_fast.py report --work W
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import discover_blind as blind  # noqa: E402
import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402

# ------------------------------------------------------------------ fixed protocol
FAST_K = (1, 2)
FAST_N = (16, "all")
FAST_A = (4.0, 8.0, 12.0)
TOP2 = 3
MIN_LOOP_SEQS = 30           # fewer looping elicit responses than this: the model barely loops on this task
# diagnosis v2 reads each looping response up to its loop onset + 768 tokens, however late the loop starts. v1 cut every
# response at 4096 tokens, which dropped almost every loop of Qwen3.5-4B (733 loop tokens from 31 looping responses).
DIAG_VERSION = 2
DOWN_RE = re.compile(r"layers\.(\d+)\.mlp\.down_proj\.weight$")


def grid(rank: list[int]) -> list[dict]:
    out = []
    for k in FAST_K:
        for n in FAST_N:
            for al in FAST_A:
                layers = sorted(rank[:k])
                out.append({"name": blind.cfg_name(layers, n, al), "layers": layers, "neurons": n, "alpha": al,
                            "k": k})
    return out


def code_hashes() -> dict:
    files = ("discover_fast.py", "loop_kill_v2.py", "discover_blind.py", "discover_loop_fix.py",
             "moi_loop_router_vllm.py")
    return {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in files}


def bf16_ok() -> bool:
    """Native bf16 (compute capability 8+). torch.cuda.is_bf16_supported() also says True on a T4 (emulation),
    where vLLM refuses bf16."""
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def sync(work: Path) -> None:
    """Copy the small result files to $WSI_SYNC_DIR (Colab: a Drive folder), so a disconnected run can resume."""
    dst = os.environ.get("WSI_SYNC_DIR")
    if not dst:
        return
    import shutil

    Path(dst).mkdir(parents=True, exist_ok=True)
    for pat in ("*.json", "*.jsonl", "*.npz", "*.log"):
        for f in work.glob(pat):
            t = Path(dst) / f.name
            if not t.exists() or t.stat().st_size != f.stat().st_size or t.stat().st_mtime < f.stat().st_mtime:
                shutil.copy2(f, t)


def note_time(work: Path, key: str, seconds: float, add: bool = True) -> None:
    p = work / "timing_fast.json"
    t = json.loads(p.read_text()) if p.exists() else {}
    t[key] = round((t.get(key, 0.0) if add else 0.0) + seconds, 1)
    p.write_text(json.dumps(t, indent=1))
    print(f"[time] {key}: {seconds / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ edit math (shared by swap and bake)
def edited(W, u: np.ndarray, cols: list[int], alpha: float):
    """W' = W - alpha * u (m * u^T W), in fp64, rounded to W's dtype: the same as discover_blind.build_checkpoint."""
    import torch

    Wd = W.double()
    uu = torch.from_numpy(np.asarray(u, dtype=np.float64))
    m = torch.zeros(Wd.shape[1], dtype=torch.float64)
    m[cols] = 1.0
    return (Wd - alpha * torch.outer(uu, m * (uu @ Wd))).to(W.dtype)


def int_checksum(t, dtype) -> int:
    import torch

    return int(t.to(dtype).contiguous().view(torch.int16).long().sum())


def load_base_layers(model_dir: Path, layers: set[int]) -> dict:
    from safetensors import safe_open

    out = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():
                m = DOWN_RE.search(k)
                if m and not k.startswith("mtp.") and int(m.group(1)) in layers:
                    out[int(m.group(1))] = fh.get_tensor(k).clone()
    missing = layers - set(out)
    if missing:
        raise SystemExit(f"down_proj not found for layers {sorted(missing)}")
    return out


def make_setter(updates: dict):
    """A function for LLM.apply_model: writes each updated down_proj in place (so CUDA graphs keep their pointers)
    and returns an exact checksum. Defined as a closure so it is pickled by value into vLLM's worker process."""

    def setter(model):
        import re as _re

        import torch as _t

        pat = _re.compile(r"layers\.(\d+)\.mlp\.down_proj\.weight$")
        done = {}
        for name, p in model.named_parameters():
            mm = pat.search(name)
            if not mm or "mtp" in name or "visual" in name:
                continue
            li = int(mm.group(1))
            if li not in updates:
                continue
            if li in done:
                raise RuntimeError(f"layer {li}: two down_proj parameters match")
            new = _t.from_numpy(updates[li])
            if tuple(new.shape) != tuple(p.shape):
                raise RuntimeError(f"layer {li}: shape {tuple(p.shape)} in vLLM vs {tuple(new.shape)} on disk")
            p.data.copy_(new.to(p.dtype).to(p.device))
            done[li] = (str(p.dtype), int(p.data.contiguous().view(_t.int16).long().sum()))
        return done

    return setter


# ------------------------------------------------------------------ prepare / diagnose
def cmd_prepare(a) -> None:
    blind.cmd_prepare(a)
    work = Path(a.work)
    cfg = dl.cfg_load(work)
    cfg["protocol"] = {"FAST_K": list(FAST_K), "FAST_N": list(FAST_N), "FAST_A": list(FAST_A), "TOP2": TOP2,
                       "script": "discover_fast.py"}
    cfg["model_repo"] = a.model_repo
    dl.cfg_save(work, cfg)


def cmd_diagnose(a) -> None:
    """discover_blind.cmd_diagnose for any size: the model may be spread over several GPUs (fp32 on 2 x T4)."""
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_elicit.jsonl")}
    outs = lr.read_jsonl(work / "outputs_elicit.jsonl")
    seqs, n_loop_seq, onsets = [], 0, []
    for o in outs:
        body = lr.strip_eos(o["token_ids"], eos)
        lab = dl.token_labels(body)
        if not (lab >= 0).any():
            continue
        first = int(np.argmax(lab == 1)) if (lab == 1).any() else None
        end = min(len(body), (first + a.after_onset) if first is not None else a.max_seq, a.max_seq)
        n_loop_seq += first is not None
        if first is not None:
            onsets.append(first)
        seqs.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body[:end], "lab": lab[:end]})
    loop_tok = sum(int((s["lab"] == 1).sum()) for s in seqs)
    norm_tok = sum(int((s["lab"] == 0).sum()) for s in seqs)
    print(f"elicit: {len(outs)} responses, {n_loop_seq} with loops; tokens {loop_tok:,} looping, {norm_tok:,} normal")
    if onsets:
        print(f"loop onset (response token): median {int(np.median(onsets)):,}, min {min(onsets):,}, "
              f"max {max(onsets):,}; reading up to {a.max_seq:,} tokens per response")
    if n_loop_seq < a.min_loop_seqs or loop_tok < a.min_loop_tokens:
        raise SystemExit(f"the model loops in only {n_loop_seq}/{len(outs)} elicit responses: too few to find a loop "
                         "direction. Use harder problems or a longer cap.")
    half = np.random.default_rng(cfg["seed"]).permutation(len(seqs)) % 2

    use_cuda = not a.cpu and torch.cuda.is_available()
    if not use_cuda:
        dtype, kw = torch.float32, {}
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(a.dtype)
        if dtype is None:  # auto: bf16 where the GPU has it, else fp32 (fp16 can overflow in Qwen activations)
            dtype = torch.bfloat16 if bf16_ok() else torch.float32
        n = torch.cuda.device_count()
        mem = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.80) for i in range(n)}
        kw = {"device_map": "auto", "max_memory": mem}
    model = AutoModelForCausalLM.from_pretrained(cfg["base_dir"], dtype=dtype, **kw).eval()
    print(f"diagnose dtype {dtype}, devices {sorted({str(p.device) for p in model.parameters()})}")
    layers = next(m for nm, m in model.named_modules()
                  if nm.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 2)
    L = len(layers)
    downs = [lyr.mlp.down_proj for lyr in layers]
    U, D = downs[0].in_features, downs[0].out_features
    body_model = getattr(model, "model", model)
    first_dev = next(model.parameters()).device
    cur: dict = {"mask": None, "masks": {}}
    acc: dict = {}

    def mask_on(dev):
        if dev not in cur["masks"]:
            cur["masks"][dev] = cur["mask"].to(dev)
        return cur["masks"][dev]

    def hook(li):
        def f(mod, inp, out):
            h, y = inp[0][0].float(), out[0].float()
            m = mask_on(h.device)
            for cls in (0, 1):
                sel = m == cls
                if not sel.any():
                    continue
                k = (li, cls)
                if cur["pass"] == "A":
                    s = acc.setdefault(k, {"n": 0, "h": torch.zeros(U, device=h.device, dtype=torch.float64),
                                           "y": torch.zeros(D, device=h.device, dtype=torch.float64)})
                    s["n"] += int(sel.sum())
                    s["h"] += h[sel].sum(0).double()
                    s["y"] += y[sel].sum(0).double()
                else:
                    p = (y[sel] @ cur["u"][li].to(y.device)).double()
                    b = acc.setdefault(("B",) + k, {"n": 0, "s": 0.0, "ss": 0.0})
                    b["n"] += int(sel.sum())
                    b["s"] += float(p.sum())
                    b["ss"] += float((p * p).sum())
        return f

    hs = [d.register_forward_hook(hook(i)) for i, d in enumerate(downs)]

    def run(which: int, name: str) -> None:
        cur["pass"] = name
        with torch.no_grad():
            for s, hb in zip(seqs, half):
                if hb != which:
                    continue
                ids = torch.tensor([s["prompt"] + s["body"]], device=first_dev)
                full = np.full(ids.shape[1], -1, dtype=np.int64)
                full[len(s["prompt"]):] = s["lab"]
                cur["mask"], cur["masks"] = torch.from_numpy(full), {}
                body_model(input_ids=ids, use_cache=False)

    t0 = time.perf_counter()
    run(0, "A")
    missing = [k for k in ((li, c) for li in range(L) for c in (0, 1)) if k not in acc]
    if missing:
        raise SystemExit(f"half A lacks loop or normal tokens for {missing[:4]}...; elicit more")
    u = []
    for li in range(L):
        d = (acc[(li, 1)]["y"] / acc[(li, 1)]["n"] - acc[(li, 0)]["y"] / acc[(li, 0)]["n"]).cpu()
        u.append((d / d.norm()).float())
    cur["u"] = u
    run(1, "B")
    for h in hs:
        h.remove()
    sep = []
    for li in range(L):
        st = {}
        for cls in (0, 1):
            b = acc[("B", li, cls)]
            m = b["s"] / b["n"]
            st[cls] = (m, max(b["ss"] / b["n"] - m * m, 1e-12))
        sep.append(float((st[1][0] - st[0][0]) / np.sqrt((st[1][1] + st[0][1]) / 2)))
    rank = [int(x) for x in np.argsort(sep)[::-1]]
    neuron_rank = {}
    for li in range(L):
        dh = (acc[(li, 1)]["h"] / acc[(li, 1)]["n"] - acc[(li, 0)]["h"] / acc[(li, 0)]["n"]).cpu()
        c = dh * (u[li].double() @ downs[li].weight.detach().double().cpu())
        neuron_rank[str(li)] = [int(x) for x in torch.argsort(c, descending=True).tolist()]
    np.savez(work / "blind_directions.npz", **{f"u{li}": u[li].numpy() for li in range(L)})
    diag = {"separation": sep, "layer_rank": rank, "neuron_rank": neuron_rank, "n_layers": L, "n_units": U,
            "hidden": D, "loop_tokens": loop_tok, "normal_tokens": norm_tok, "sequences": len(seqs),
            "looping_sequences": n_loop_seq, "elicit_responses": len(outs), "dtype": str(dtype),
            "diag_version": DIAG_VERSION, "max_seq": a.max_seq, "loop_onsets": sorted(onsets),
            "capture_s": round(time.perf_counter() - t0, 1)}
    (work / "diagnosis_blind.json").write_text(json.dumps(diag, indent=1))
    print("held-out d' per layer:", " ".join(f"L{li}:{s:.2f}" for li, s in enumerate(sep)))
    print("layer rank:", rank)


# ------------------------------------------------------------------ generation: one vLLM, many jobs
def _llm(model: str, cfg: dict, max_len: int, dtype: str, gpu_mem: float):
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # apply_model sends a function to the worker
    os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE)] + [p for p in os.environ.get("PYTHONPATH", "").split(
        os.pathsep) if p])  # the worker imports loop_kill_v2 to load the early-stop processor
    import loop_kill_v2
    from vllm import LLM

    base_kw = dict(model=model, tokenizer=cfg["base_dir"], dtype=dtype, max_model_len=max_len,
                   gpu_memory_utilization=gpu_mem, enable_prefix_caching=True, seed=cfg["seed"])
    extras = {"limit_mm_per_prompt": {"image": 0, "video": 0}}
    proc = loop_kill_v2.processor()
    if proc is not None:
        extras["logits_processors"] = [proc]
    for drop in ([], ["limit_mm_per_prompt"], ["logits_processors"], ["limit_mm_per_prompt", "logits_processors"]):
        kw = {**base_kw, **{k: v for k, v in extras.items() if k not in drop}}
        try:
            llm = LLM(**kw)
            kill = "logits_processors" in kw
            print(f"[gen] vLLM options: text_only={'limit_mm_per_prompt' in kw} early_loop_stop={kill} "
                  f"({getattr(proc, '__name__', None)})", flush=True)
            return llm, kill
        except Exception as e:  # noqa: BLE001
            print(f"[gen] vLLM refused {sorted(set(extras) - set(drop))}: {type(e).__name__}: {str(e)[:300]}",
                  flush=True)
            import gc

            gc.collect()
    raise SystemExit("vLLM could not start")


def _generate(llm, cfg: dict, reqs: list[dict], kill: bool) -> tuple[list[dict], bool]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    kill_eos = cfg["eos"][-1]

    def params(r, k):
        return SamplingParams(**lr.SAMPLING, max_tokens=r["max_tokens"], seed=r["seed"], stop_token_ids=cfg["eos"],
                              logit_bias={int(t): float(v) for t, v in r.get("logit_bias", {}).items()} or None,
                              extra_args={"loop_kill_eos": kill_eos} if k else None)

    prompts = [TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in reqs]
    try:
        outs = llm.generate(prompts, [params(r, kill) for r in reqs], use_tqdm=True)
    except Exception as e:  # noqa: BLE001
        if not kill:
            raise
        print(f"[gen] early loop stop failed ({type(e).__name__}: {str(e)[:300]}); generating without it", flush=True)
        kill = False
        outs = llm.generate(prompts, [params(r, False) for r in reqs], use_tqdm=True)
    rows = [{"id": r["id"], "token_ids": list(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason}
            for r, o in zip(reqs, outs)]
    return rows, kill


def job_updates(work: Path, job: dict, base_w: dict, edited_now: set[int]) -> tuple[dict, dict]:
    """The down_proj matrices to write for this job: its edits, plus base weights for layers the last job edited."""
    import torch

    diag = json.loads((work / "diagnosis_blind.json").read_text()) if job.get("layers") else None
    z = np.load(work / "blind_directions.npz") if job.get("layers") else None
    upd = {}
    for li in job.get("layers") or []:
        u = blind.direction(diag, z, li, job.get("direction", "wsi"))
        upd[li] = edited(base_w[li], u, blind.columns(diag, li, job["neurons"]), float(job["alpha"]))
    for li in edited_now - set(upd):
        upd[li] = base_w[li]
    return {li: t.to(torch.float32).numpy() for li, t in upd.items()}, {li: t for li, t in upd.items()}


def cmd_gen_jobs(a) -> None:
    """One vLLM process: run each job in a JSON list, swapping the edited weights in memory between jobs."""
    import torch

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    jobs = json.loads(Path(a.jobs).read_text())
    todo = [j for j in jobs if not (work / j["out"]).exists()]
    if not todo:
        print("[gen-jobs] nothing to do")
        return
    req_cache: dict = {}

    def reqs_of(j):
        if j["requests"] not in req_cache:
            req_cache[j["requests"]] = lr.read_jsonl(work / j["requests"])
        rs = req_cache[j["requests"]]
        if "part" in j:
            i, n = j["part"]
            rs = rs[i::n]
        return rs

    max_len = max(len(r["prompt_ids"]) + r["max_tokens"] for j in todo for r in reqs_of(j)) + 16
    dtype = a.dtype if a.dtype != "auto" else ("bfloat16" if bf16_ok() else "float16")
    model_dir = Path(a.model or cfg["base_dir"])
    llm, kill = _llm(str(model_dir), cfg, max_len, dtype, a.gpu_mem)
    need = sorted({li for j in todo for li in (j.get("layers") or [])})
    base_w = load_base_layers(model_dir, set(need)) if need else {}
    edited_now: set[int] = set()
    bake, loaded = False, str(model_dir)  # bake = fallback: one baked checkpoint and one vLLM start per config
    ck_tmp = Path(a.ck_dir or work) / f"_swap_fallback_{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}"
    for j in todo:
        t0 = time.perf_counter()
        if not bake and (j.get("layers") or edited_now):
            try:
                upd_np, upd_t = job_updates(work, j, base_w, edited_now)
                got = {}
                for r in llm.apply_model(make_setter(upd_np)):
                    got.update(r)
                if set(got) != set(upd_np):
                    raise RuntimeError(f"weight swap missed layers {sorted(set(upd_np) - set(got))}")
                for li, (dt, cs) in got.items():
                    want = int_checksum(upd_t[li], getattr(torch, dt.replace("torch.", "")))
                    if cs != want:
                        raise RuntimeError(f"weight swap checksum mismatch at layer {li}: {cs} != {want}")
                llm.reset_prefix_cache()  # the cached prefixes were computed with the previous weights
                edited_now = set(j.get("layers") or [])
            except Exception as e:  # noqa: BLE001
                print(f"[gen-jobs] IN-MEMORY SWAP FAILED ({type(e).__name__}: {str(e)[:300]}); falling back to one "
                      "baked checkpoint and one vLLM start per config", flush=True)
                bake, loaded = True, None  # a half-done swap may have left edited weights: always reload
        if bake:
            want_dir = str(model_dir)
            if j.get("layers"):
                blind.build_checkpoint(work, j["layers"], j["neurons"], float(j["alpha"]), j.get("direction", "wsi"),
                                       ck_tmp)
                want_dir = str(ck_tmp)
            if want_dir != loaded or j.get("layers"):
                import gc

                del llm
                gc.collect()
                torch.cuda.empty_cache()
                llm, kill = _llm(want_dir, cfg, max_len, dtype, a.gpu_mem)
                loaded = want_dir
        reqs = reqs_of(j)
        rows, kill = _generate(llm, cfg, reqs, kill)
        lr.write_jsonl(work / j["out"], rows)
        sync(work)
        ntok = sum(len(x["token_ids"]) for x in rows)
        dt_ = time.perf_counter() - t0
        stopped = sum(1 for x in rows if kill and x["token_ids"] and x["token_ids"][-1] == cfg["eos"][-1]
                      and lr.analyse(x["token_ids"], set(cfg["eos"]))["kill"] is not None)
        print(f"[gen {j['out']}] {len(rows)} seqs, {ntok:,} tokens, {dt_ / 60:.1f} min ({ntok / dt_:,.0f} tok/s)"
              f"; early stop {'on' if kill else 'OFF'}, {stopped} loops stopped", flush=True)


def run_parallel(work: Path, groups: list[dict], a) -> None:
    """groups: [{"model": dir or None, "jobs": [...]}]; one process per GPU at a time, each runs its groups in order."""
    import torch

    ngpu = max(1, torch.cuda.device_count()) if not a.cpu else 1
    lanes: list[list[dict]] = [[] for _ in range(ngpu)]
    load = [0] * ngpu
    for g in groups:
        i = load.index(min(load))
        lanes[i].append(g)
        load[i] += sum(int(j.get("weight", 1)) for j in g["jobs"])
    procs = []
    for i, lane in enumerate(lanes):
        if not lane:
            continue
        cmds = []
        for gi, g in enumerate(lane):
            jf = work / f"jobs_gpu{i}_{gi}.json"
            jf.write_text(json.dumps(g["jobs"], indent=1))
            cmd = [sys.executable, str(Path(__file__).resolve()), "gen-jobs", "--work", str(work), "--jobs", str(jf),
                   "--dtype", a.dtype, "--gpu-mem", str(a.gpu_mem)]
            if g.get("model"):
                cmd += ["--model", str(g["model"])]
            cmds.append(cmd)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if a.cpu else str(i)}
        log = open(work / f"gen_gpu{i}.log", "a")
        script = " && ".join(subprocess.list2cmdline(c) for c in cmds)
        procs.append((i, subprocess.Popen(script, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    pos = {i: (work / f"gen_gpu{i}.log").stat().st_size for i, _, _ in procs}

    def show_progress():  # the generation logs, filtered, so the notebook cell shows each finished config
        for i, _, _ in procs:
            f = work / f"gen_gpu{i}.log"
            with open(f, "rb") as fh:
                fh.seek(pos[i])
                chunk = fh.read().decode(errors="replace")
                pos[i] = fh.tell()
            for line in chunk.replace(chr(13), chr(10)).splitlines():
                if line.startswith(("[gen", "[time")) or "Error" in line or "FAILED" in line:
                    print(f"GPU{i} {line}", flush=True)

    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(20)
        show_progress()
    show_progress()
    bad = []
    for i, p, log in procs:
        rc = p.wait()
        log.close()
        tail = (work / f"gen_gpu{i}.log").read_text(errors="replace").splitlines()[-25:]
        print(f"--- GPU {i} rc={rc} ---\n" + "\n".join(x for x in tail if x.startswith(("[gen", "[time", "Trace"))
                                                     or "Error" in x))
        if rc:
            bad.append(i)
    if bad:
        raise SystemExit(f"generation failed on GPU {bad}; see {work}/gen_gpu*.log")


# ------------------------------------------------------------------ elicit / search (dev only)
def cmd_elicit(a) -> None:
    import torch

    work = Path(a.work)
    t0 = time.perf_counter()
    if not (work / "outputs_elicit.jsonl").exists():
        n = max(1, torch.cuda.device_count()) if not a.cpu else 1
        parts = [{"name": f"elicit{i}", "requests": "requests_elicit.jsonl", "out": f"outputs_elicit.p{i}.jsonl",
                  "part": [i, n]} for i in range(n)]
        run_parallel(work, [{"model": None, "jobs": [p]} for p in parts], a)
        rows = [r for p in parts for r in lr.read_jsonl(work / p["out"])]
        lr.write_jsonl(work / "outputs_elicit.jsonl", sorted(rows, key=lambda r: r["id"]))
    s = dl.score(work, "requests_elicit.jsonl", "outputs_elicit.jsonl")
    print(f"elicit: accuracy {s['accuracy']:.1%}, loop killed {s['loop_killed']:.1%}, truncated {s['truncated']:.1%}")
    note_time(work, "elicit", time.perf_counter() - t0)


def _remove(work: Path, names: list[str]) -> None:
    """Delete files here and in $WSI_SYNC_DIR (else a Colab restart would copy them back from Drive)."""
    dirs = [work] + ([Path(os.environ["WSI_SYNC_DIR"])] if os.environ.get("WSI_SYNC_DIR") else [])
    for d in dirs:
        for n in names:
            (d / n).unlink(missing_ok=True)


def cmd_search(a) -> None:
    work = Path(a.work)
    dp = work / "diagnosis_blind.json"
    if json.loads(dp.read_text()).get("diag_version") != DIAG_VERSION:
        print(f"diagnosis is older than v{DIAG_VERSION}: running it again", flush=True)
        r = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "diagnose", "--work", str(work)])
        if r.returncode:
            raise SystemExit("diagnose failed")
    diag = json.loads(dp.read_text())
    diag_sha = hashlib.sha256(dp.read_bytes()).hexdigest()
    sp = work / "search.json"
    state = json.loads(sp.read_text()) if sp.exists() else {}
    if state.get("diag_sha") != diag_sha:  # configs were built from another diagnosis: their outputs are stale
        stale = sorted(f.name for f in work.glob("outputs_dev*_*.jsonl") if not f.name.endswith("_base.jsonl"))
        if stale or state:
            print(f"new diagnosis: dropping {len(stale)} config outputs and the old search state", flush=True)
        _remove(work, stale + ["search.json"])
        state = {"round1": {}, "round2": {}, "time_s": {}, "diag_sha": diag_sha}
    configs1 = grid(diag["layer_rank"])
    for rnd in (1, 2):
        key, split = f"round{rnd}", f"dev{rnd}"
        req = f"requests_{split}.jsonl"
        if rnd == 1:
            configs = configs1
        else:
            r1 = state["round1"]
            configs = [next(c for c in configs1 if c["name"] == b)
                       for b in sorted(r1, key=lambda k: blind.score_key(r1[k]), reverse=True)[:TOP2]]
        t0 = time.perf_counter()
        jobs = [{"name": "base", "requests": req, "out": f"outputs_{split}_base.jsonl"}]
        jobs += [{**c, "requests": req, "out": f"outputs_{split}_{c['name']}.jsonl", "direction": "wsi"}
                 for c in configs]
        jobs = [j for j in jobs if not (work / j["out"]).exists()]
        if jobs:
            import torch

            ngpu = max(1, torch.cuda.device_count()) if not a.cpu else 1
            lanes = [jobs[i::ngpu] for i in range(ngpu)]
            run_parallel(work, [{"model": None, "jobs": ln} for ln in lanes if ln], a)
        base = dl.score(work, req, f"outputs_{split}_base.jsonl")
        state[f"base_{split}"] = {k: v for k, v in base.items() if k != "rows"}
        if rnd == 1 and base["accuracy"] < 0.2 and base["loop_killed"] < 0.5:
            raise SystemExit(f"base model scores {base['accuracy']:.1%} without looping on dev1: the numerics look "
                             "broken (fp16 overflow?). Use a bf16 GPU.")
        for c in configs:
            s = dl.score(work, req, f"outputs_{split}_{c['name']}.jsonl")
            state[key][c["name"]] = {**{k: v for k, v in s.items() if k != "rows"}, **c}
            print(f"[{key}] {c['name']}: acc={s['accuracy']:.1%} loops={s['loop_killed']:.1%} "
                  f"trunc={s['truncated']:.1%}  (base {base['accuracy']:.1%})", flush=True)
        state["time_s"][key] = round(state["time_s"].get(key, 0.0) + time.perf_counter() - t0, 1)
        sp.write_text(json.dumps(state, indent=1))
        sync(work)
    r2 = state["round2"]
    order = list(r2)  # round-1 order; ties -> better in round 1
    state["choice"] = dict(r2[max(order, key=lambda k: (blind.score_key(r2[k]), -order.index(k)))])
    sp.write_text(json.dumps(state, indent=1))
    note_time(work, "search", sum(state["time_s"].values()), add=False)
    print("choice:", state["choice"]["name"])


def cmd_freeze(a) -> None:
    work = Path(a.work)
    st = json.loads((work / "search.json").read_text())
    if "choice" not in st:
        raise SystemExit("search has not picked a choice")
    c = {k: st["choice"][k] for k in ("name", "layers", "neurons", "alpha", "k")}
    p = work / "frozen.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old["choice"] != c:
            raise SystemExit(f"already frozen with a different choice: {old['choice']}")
        print("already frozen:", c["name"])
        return
    t = json.loads((work / "timing_fast.json").read_text()) if (work / "timing_fast.json").exists() else {}
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    disc = t.get("elicit", 0.0) + t.get("diagnose", 0.0) + sum(st["time_s"].values())
    p.write_text(json.dumps({"choice": c, "code_sha256": code_hashes(), "git_commit": a.git_commit,
                             "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "discovery_steps_s": {"elicit": t.get("elicit"), "diagnose": t.get("diagnose"),
                                                   **st["time_s"]},
                             "diagnose_capture_s": diag.get("capture_s"),
                             "discovery_s": round(disc, 1), "discovery_min": round(disc / 60, 1)}, indent=1))
    print(f"frozen: {c['name']}  (discovery {disc / 60:.1f} min)")


# ===== EXAM: runs only after freeze =====
def cmd_test(a) -> None:
    work = Path(a.work)
    fr = json.loads((work / "frozen.json").read_text()) if (work / "frozen.json").exists() else None
    if fr is None:
        raise SystemExit("frozen.json missing: run freeze first")
    c = fr["choice"]
    cks = {}
    for kind in ("wsi", "random"):
        ck = Path(a.ck_root or work) / blind.cfg_name(c["layers"], c["neurons"], c["alpha"], kind)
        if not ck.exists():
            blind.build_checkpoint(work, c["layers"], c["neurons"], c["alpha"], kind, ck)
        cks[kind] = ck
    req = "requests_test.jsonl"
    t0 = time.perf_counter()
    run_parallel(work, [{"model": None, "jobs": [{"name": "base", "requests": req, "out": "outputs_test_base.jsonl"}]},
                        {"model": cks["wsi"], "jobs": [{"name": "fixed", "requests": req,
                                                        "out": "outputs_test_fast.jsonl"}]},
                        {"model": cks["random"], "jobs": [{"name": "random", "requests": req,
                                                           "out": "outputs_test_random.jsonl"}]}], a)
    dl.cmd_plan_moi(a)
    mreq = "requests_test_moi.jsonl"
    run_parallel(work, [{"model": cks["wsi"], "jobs": [{"name": "moi", "requests": mreq,
                                                        "out": "outputs_test_moi_fast.jsonl"}]},
                        {"model": cks["random"], "jobs": [{"name": "moi_random", "requests": mreq,
                                                           "out": "outputs_test_moi_random.jsonl"}]}], a)
    note_time(work, "test", time.perf_counter() - t0)


def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    fr = json.loads((work / "frozen.json").read_text()) if (work / "frozen.json").exists() else None
    if fr is None:
        raise SystemExit("frozen.json missing: the exam runs only after the method is frozen")
    if fr["code_sha256"] != code_hashes():
        raise SystemExit("the code changed after the freeze; refusing to report")
    cfg = dl.cfg_load(work)
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"])
    eos, reqs = set(cfg["eos"]), {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    res: dict = {"model": cfg.get("model_repo"), "choice": fr["choice"],
                 "frozen": {k: fr[k] for k in ("frozen_utc", "git_commit", "code_sha256")},
                 "discovery_min": fr["discovery_min"], "discovery_steps_s": fr["discovery_steps_s"], "arms": {}}
    rows = {}
    for name, f in (("base", "outputs_test_base.jsonl"), ("fast_fixed", "outputs_test_fast.jsonl"),
                    ("random_fixed", "outputs_test_random.jsonl")):
        if (work / f).exists():
            s = dl.score(work, "requests_test.jsonl", f, tok)
            rows[name] = {r["id"]: r for r in s["rows"]}
            res["arms"][name] = {k: v for k, v in s.items() if k != "rows"}
    plan = {r["id"]: r for r in lr.read_jsonl(work / "requests_test_moi.jsonl")}
    for name, f in (("fast_moi", "outputs_test_moi_fast.jsonl"), ("random_moi", "outputs_test_moi_random.jsonl")):
        if not (work / f).exists() or "base" not in rows:
            continue
        cont = {o["id"]: o for o in lr.read_jsonl(work / f)}
        moi, edit_tok, all_tok = {}, 0, 0
        for i in reqs:
            if i in cont:
                pre = plan[i]["prompt_ids"][len(reqs[i]["prompt_ids"]):]
                g = lr.grade(pre + cont[i]["token_ids"], cont[i]["finish_reason"], reqs[i]["gold"], tok, eos,
                             reqs[i]["max_tokens"])
                edit_tok += max(g["tokens"] - len(pre), 0)
            else:
                g = rows["base"][i]
            all_tok += g["tokens"]
            moi[i] = g
        n = len(moi)
        res["arms"][name] = {"n": n, "accuracy": sum(g["correct"] for g in moi.values()) / n,
                             "loop_killed": sum(g["looped_killed"] for g in moi.values()) / n,
                             "truncated": sum(g["truncated"] for g in moi.values()) / n,
                             "edit_token_share": edit_tok / max(all_tok, 1), "router_fired": len(plan) / n}
        rows[name] = moi
    ids = sorted(reqs)
    acc = {k: [bool(rows[k][i]["correct"]) for i in ids] for k in rows}
    pairs = [("fast_fixed", "base"), ("fast_moi", "base"), ("fast_fixed", "random_fixed"),
             ("fast_moi", "random_moi"), ("fast_moi", "fast_fixed")]
    res["paired"] = {f"{x}_vs_{y}": lr.paired_ci(acc[x], acc[y]) for x, y in pairs if x in acc and y in acc}
    st = json.loads((work / "search.json").read_text())
    pick = ("accuracy", "loop_killed", "truncated")
    res["search"] = {"base_dev1": st.get("base_dev1"), "base_dev2": st.get("base_dev2"),
                     "round1": {k: {x: v[x] for x in pick} for k, v in st["round1"].items()},
                     "round2": {k: {x: v[x] for x in pick} for k, v in st["round2"].items()}, "time_s": st["time_s"]}
    res["diagnosis"] = {k: v for k, v in json.loads((work / "diagnosis_blind.json").read_text()).items()
                        if k != "neuron_rank"}
    t = work / "timing_fast.json"
    res["timing_s"] = json.loads(t.read_text()) if t.exists() else {}
    out = work / a.out
    out.write_text(json.dumps(res, indent=1))
    print(f"\nmodel {res['model']}  choice {fr['choice']['name']}  discovery {fr['discovery_min']} min")
    print(f"{'arm':13s} {'acc':>7s} {'loops':>7s} {'trunc':>7s}")
    for k, v in res["arms"].items():
        print(f"{k:13s} {v['accuracy']:7.1%} {v['loop_killed']:7.1%} {v['truncated']:7.1%}")
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
    print("Saved", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--base-dir", required=True)
    p.add_argument("--model-repo", default=None)
    p.add_argument("--data-dir")
    p.add_argument("--n-elicit", type=int, default=300)
    p.add_argument("--elicit-max-new", type=int, default=8192)
    p.add_argument("--n-dev1", type=int, default=32)
    p.add_argument("--n-dev2", type=int, default=64)
    p.add_argument("--dev-max-new", type=int, default=16384)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--max-new", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("diagnose")
    d.add_argument("--work", required=True)
    d.add_argument("--after-onset", type=int, default=768)
    d.add_argument("--max-seq", type=int, default=16384, help="response tokens read per sequence (v1 used 4096)")
    d.add_argument("--min-loop-tokens", type=int, default=500)
    d.add_argument("--min-loop-seqs", type=int, default=MIN_LOOP_SEQS)
    d.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    d.add_argument("--cpu", action="store_true")
    for name in ("gen-jobs", "elicit", "search", "test"):
        g = sub.add_parser(name)
        g.add_argument("--work", required=True)
        g.add_argument("--dtype", default="auto", help="vLLM dtype: auto = bfloat16 if the GPU has it, else float16")
        g.add_argument("--gpu-mem", type=float, default=0.88)
        g.add_argument("--cpu", action="store_true", help="local smoke tests only")
        if name == "gen-jobs":
            g.add_argument("--jobs", required=True)
            g.add_argument("--model")
            g.add_argument("--ck-dir", help="where the fallback writes a baked checkpoint (default: work)")
        if name == "test":
            g.add_argument("--ck-root", help="where to write the baked checkpoints (default: work)")
    f = sub.add_parser("freeze")
    f.add_argument("--work", required=True)
    f.add_argument("--git-commit", default=None)
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--out", default="results_discover_fast.json")
    a = ap.parse_args()
    fns = {"prepare": cmd_prepare, "diagnose": cmd_diagnose, "gen-jobs": cmd_gen_jobs, "elicit": cmd_elicit,
           "search": cmd_search, "freeze": cmd_freeze, "test": cmd_test, "report": cmd_report}
    t0 = time.perf_counter()
    fns[a.cmd](a)
    if a.cmd == "diagnose":
        note_time(Path(a.work), "diagnose", time.perf_counter() - t0)
    if a.cmd != "gen-jobs":
        sync(Path(a.work))


if __name__ == "__main__":
    main()
