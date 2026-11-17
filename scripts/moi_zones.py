"""Zone streaming demo for Compensated Selective MoI: one model, several task zones, a router, and the cost of a swap.

Like a game that only loads the part of the map in view: each "zone" is a compensated task index (about 70% of the MLP
neurons, built from ~20k tokens of that task) and only the active zone needs to be in GPU memory. A router reads the
prompt BEFORE generation and picks the zone. The rest of the model (attention, embeddings, norms) stays resident.

Zones (built with experiment_moi_speed_vllm.py export --calib-source ...):
    math     GSM8K-train worked solutions          -> evaluated on GSM8K test (200)
    code     MBPP-train tasks + reference code     -> evaluated on HumanEval (164, run against its unit tests)
    general  WikiText-2 paragraphs                  -> evaluated on MMLU test (400, letter answer)
Calibration and evaluation sets never overlap; code is calibrated on MBPP and tested on HumanEval (different dataset).

Subcommands:
    eval    one checkpoint (full model or a zone) in vLLM on all three tasks; per-item correctness is saved, so every
            zone x task cell and any routing policy can be scored afterwards. Greedy, thinking off.
    route   the router: nearest centroid on the mean INPUT EMBEDDING of the prompt (no forward pass; the embedding
            table stays resident anyway). Trained on GSM8K-train questions, MBPP-train tasks and MMLU-validation
            questions; tested on the eval prompts' content, without the task instructions (no trivial cue).
    swap    the cost of switching zones: bytes of a zone's MLP weights; load time disk -> GPU (cold page cache if it
            can be dropped, and warm), pinned RAM -> GPU; network time estimated from bytes; neuron overlap between
            zones and the bytes a delta swap would move.
    report  zone x task accuracy, routed system vs full model, cost of a wrong zone, routing accuracy, swap costs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_moi_speed_vllm as sp  # noqa: E402

ZONES = {"math": "gsm8k", "code": "mbpp", "general": "wikitext"}  # zone -> calibration source
TASKS = ["gsm8k", "humaneval", "mmlu"]
TASK_ZONE = {"gsm8k": "math", "humaneval": "code", "mmlu": "general"}
LET = "ABCD"
CODE_INSTR = ("Complete the following Python function. Reply with the complete function, including any imports, "
              "in a single ```python code block.\n\n")
MC_INSTR = "\nAnswer with only the letter."
MAX_NEW = {"gsm8k": 1024, "humaneval": 768, "mmlu": 8}


# ------------------------------------------------------------------ tasks
def mc_text(q, choices):
    return q.strip() + "\n" + "\n".join(f"{L}. {c}" for L, c in zip(LET, choices))


def load_tasks(data_dir: Path, n_gsm: int, n_mmlu: int, seed: int) -> dict:
    """task -> list of items with 'content' (what the router sees) and 'user' (the full message sent to the model)."""
    import pandas as pd

    out = {}
    g = pd.read_parquet(sp.need(data_dir, "gsm8k/test.parquet")).iloc[:n_gsm]
    out["gsm8k"] = [{"content": q, "user": q + sp.PROMPT_SUFFIX, "gold": a.split("####")[-1].strip().replace(",", "")}
                    for q, a in zip(g["question"], g["answer"])]
    h = pd.read_parquet(sp.need(data_dir, "humaneval/test.parquet"))
    out["humaneval"] = [{"content": p, "user": CODE_INSTR + p, "prompt": p, "test": t, "entry": e, "task_id": i}
                        for i, p, t, e in zip(h["task_id"], h["prompt"], h["test"], h["entry_point"])]
    m = pd.read_parquet(sp.need(data_dir, "mmlu/test.parquet"))
    idx = np.random.default_rng(seed).permutation(len(m))[:n_mmlu]
    out["mmlu"] = [{"content": mc_text(m["question"].iloc[i], list(m["choices"].iloc[i])),
                    "user": mc_text(m["question"].iloc[i], list(m["choices"].iloc[i])) + MC_INSTR,
                    "gold": int(m["answer"].iloc[i]), "subject": m["subject"].iloc[i]} for i in idx]
    return out


def extract_code(text: str, prompt: str, entry: str) -> str:
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, re.S)
    if not blocks and "```" in text:  # unterminated block (hit the token cap)
        blocks = [text.split("```", 1)[1].split("\n", 1)[-1]]
    with_def = [b for b in blocks if f"def {entry}" in b]
    code = (with_def or blocks or [text])[-1]
    if f"def {entry}" not in code:  # the model wrote only the body
        code = prompt + code
    elif not re.search(r"^(import|from) ", code, re.M):
        # keep the prompt's imports (typing etc.) if the model dropped them
        imports = "\n".join(l for l in prompt.splitlines() if re.match(r"^(import|from) ", l))
        code = imports + "\n" + code
    return code


def run_humaneval(code: str, test: str, entry: str, timeout: float = 10.0) -> bool:
    prog = f"{code}\n\n{test}\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout,
                           cwd=tempfile.gettempdir())
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def grade_mc(text: str, gold: int) -> bool:
    t = text.strip()
    m = re.match(r"^\(?([ABCD])\b", t) or re.search(r"\b([ABCD])\b", t[:40])
    return bool(m) and LET.index(m.group(1)) == gold


def grade(task: str, item: dict, text: str) -> bool:
    if task == "gsm8k":
        return sp.same_number(sp.extract_number(text), item["gold"])
    if task == "humaneval":
        return run_humaneval(extract_code(text, item["prompt"], item["entry"]), item["test"], item["entry"])
    return grade_mc(text, item["gold"])


# ------------------------------------------------------------------ eval (vLLM)
def cmd_eval(a) -> None:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    md = sp.local_dir(a.model_dir)
    tok = AutoTokenizer.from_pretrained(md)
    tasks = load_tasks(Path(a.data_dir), a.n_gsm, a.n_mmlu, a.seed)
    dtype = a.dtype
    if dtype == "auto" and torch.cuda.get_device_capability()[0] < 8:
        dtype = "half"
    t_load = time.perf_counter()
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
    load_s = time.perf_counter() - t_load

    prompts, params, where = [], [], []
    for task in TASKS:
        for i, it in enumerate(tasks[task]):
            prompts.append(tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False))
            params.append(SamplingParams(temperature=0.0, max_tokens=MAX_NEW[task]))
            where.append((task, i))
    t0 = time.perf_counter()
    outs = llm.generate(prompts, params, use_tqdm=False)
    gen_s = time.perf_counter() - t0
    texts = {t: [None] * len(tasks[t]) for t in TASKS}
    ntok = 0
    for (task, i), o in zip(where, outs):
        texts[task][i] = o.outputs[0].text
        ntok += len(o.outputs[0].token_ids)
    t0 = time.perf_counter()
    res = {"tag": a.tag, "model_dir": str(md), "gpu": torch.cuda.get_device_name(0),
           "checkpoint_bytes": sum(f.stat().st_size for f in md.glob("*.safetensors")), "tasks": {}}
    for task in TASKS:
        if task == "humaneval":
            with ThreadPoolExecutor(8) as ex:
                ok = list(ex.map(lambda p: grade(task, *p), zip(tasks[task], texts[task])))
        else:
            ok = [grade(task, it, tx) for it, tx in zip(tasks[task], texts[task])]
        res["tasks"][task] = {"n": len(ok), "accuracy": round(float(np.mean(ok)), 4), "correct": [int(x) for x in ok],
                              "samples": [tx[-400:] for tx in texts[task][:2]]}
        print(f"[{a.tag}] {task}: {np.mean(ok):.1%} ({sum(ok)}/{len(ok)})", flush=True)
    res["time_s"] = {"vllm_load": round(load_s, 1), "generate": round(gen_s, 1),
                     "grade": round(time.perf_counter() - t0, 1), "gen_tokens": ntok}
    Path(a.res_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.res_dir) / f"eval_{a.tag}.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ router (input embeddings, no forward pass)
def embed_table(model_dir: Path):
    import torch
    from safetensors import safe_open

    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if k.endswith("embed_tokens.weight") and not k.startswith("mtp."):
                    return s.get_tensor(k).to(torch.float32)
    raise SystemExit("no embed_tokens.weight found")


def stub(text: str, code: str, tests) -> str:
    """An MBPP task written as a function signature + docstring with one doctest-style example."""
    m = re.search(r"^def .*?:\s*$", code.replace("\r", ""), re.M)
    sig = m.group(0) if m else "def solution():"
    return f'{sig}\n    """ {text}\n    >>> {tests[0].replace("assert ", "")}\n    """\n'


def cmd_route(a) -> None:
    import pandas as pd
    from transformers import AutoTokenizer

    md = sp.local_dir(a.model)
    tok = AutoTokenizer.from_pretrained(md)
    t0 = time.perf_counter()
    E = embed_table(md)
    load_s = time.perf_counter() - t0
    dd = Path(a.data_dir)

    def feat(texts):
        out = []
        for t in texts:
            ids = tok(t, add_special_tokens=False).input_ids[:512] or [0]
            v = E[ids].mean(0).numpy()
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out)

    n = a.n_train
    g = pd.read_parquet(sp.need(dd, "gsm8k/train.parquet"))
    mb = pd.read_parquet(sp.need(dd, "mbpp/train.parquet"))
    mv = pd.read_parquet(sp.need(dd, "mmlu/validation.parquet"))
    # code requests come both as prose + code and as a function stub with a docstring: half of the MBPP-train tasks
    # are written each way (built from MBPP only; HumanEval is never seen by the router)
    h = n // 2
    train = {"math": list(g["question"][:n]),
             "code": [f"{t}\n{c}" for t, c in zip(mb["text"][:h], mb["code"][:h])]
                     + [stub(t, c, tl) for t, c, tl in zip(mb["text"][h:n], mb["code"][h:n], mb["test_list"][h:n])],
             "general": [mc_text(q, list(c)) for q, c in zip(mv["question"][:n], mv["choices"][:n])]}
    from sklearn.linear_model import LogisticRegression

    F = {z: feat(v) for z, v in train.items()}
    zones = list(F)
    X_tr = np.concatenate([F[z] for z in zones])
    y_tr = np.concatenate([[j] * len(F[z]) for j, z in enumerate(zones)])
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=a.reg).fit((X_tr - mu) / sd, y_tr)
    C = np.stack([(F[z] - mu).mean(0) for z in zones])  # nearest-centroid baseline, reported for comparison
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    t1 = time.perf_counter()
    tasks = load_tasks(dd, a.n_gsm, a.n_mmlu, a.seed)
    res = {"method": "logistic regression on the mean input embedding of the prompt content (no forward pass)",
           "train_per_zone": {z: len(v) for z, v in train.items()}, "embed_load_s": round(load_s, 2), "tasks": {}}
    nprompts = 0
    for task in TASKS:
        Xr = feat([it["content"] for it in tasks[task]])
        pred = [zones[j] for j in clf.predict((Xr - mu) / sd)]
        cen = [zones[j] for j in ((Xr - mu) @ C.T).argmax(1)]
        nprompts += len(pred)
        res["tasks"][task] = {"route": pred, "accuracy": round(float(np.mean([p == TASK_ZONE[task] for p in pred])), 4),
                              "counts": {z: pred.count(z) for z in zones},
                              "centroid_accuracy": round(float(np.mean([p == TASK_ZONE[task] for p in cen])), 4)}
        print(f"router on {task}: {res['tasks'][task]['accuracy']:.1%} -> {res['tasks'][task]['counts']} "
              f"(nearest centroid {res['tasks'][task]['centroid_accuracy']:.1%})", flush=True)
    res["route_ms_per_prompt"] = round((time.perf_counter() - t1) / max(nprompts, 1) * 1000, 3)
    Path(a.res_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.res_dir) / "route.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ swap cost
def mlp_keys(model_dir: Path) -> dict:
    from safetensors import safe_open

    keys = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if re.search(r"layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$", k) and not k.startswith("mtp."):
                    keys[k] = f
    return keys


def drop_page_cache() -> bool:
    try:
        subprocess.run(["sync"], check=False)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except Exception:  # noqa: BLE001
        return False


def cmd_swap(a) -> None:
    import torch
    from safetensors import safe_open

    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    zdirs = dict(x.split("=", 1) for x in a.zones.split(","))
    res = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "zones": {}}
    full = sp.local_dir(a.full) if a.full else None
    if full:
        fk = mlp_keys(full)
        total = sum(f.stat().st_size for f in full.glob("*.safetensors"))
        mlp_b = 0
        for k, f in fk.items():
            with safe_open(str(f), "pt") as s:
                t = s.get_slice(k)
                mlp_b += int(np.prod(t.get_shape())) * 2
        res["full"] = {"checkpoint_bytes": total, "mlp_bytes": mlp_b, "resident_non_mlp_bytes": total - mlp_b}
    kept = {}
    for z, d in zdirs.items():
        d = Path(d)
        keys = mlp_keys(d)
        if (d / "kept_neurons.npz").exists():
            kept[z] = dict(np.load(d / "kept_neurons.npz"))
        r = {}
        # disk -> device, cold page cache if we can drop it, then warm
        for label in ("disk_cold", "disk_warm"):
            if label == "disk_cold":
                r["page_cache_dropped"] = drop_page_cache()
            sync()
            t0 = time.perf_counter()
            nbytes = 0
            for k, f in keys.items():
                with safe_open(str(f), "pt", device=dev) as s:
                    t = s.get_tensor(k)
                    nbytes += t.numel() * t.element_size()
            sync()
            r[f"{label}_s"] = round(time.perf_counter() - t0, 3)
        r["mlp_bytes"] = nbytes
        # pinned RAM -> device
        cpu = []
        for k, f in keys.items():
            with safe_open(str(f), "pt") as s:
                t = s.get_tensor(k).clone()  # own the memory: the file mapping closes with the context
            cpu.append(t.pin_memory() if dev == "cuda" else t)
        gpu = [torch.empty_like(t, device=dev) for t in cpu]
        times = []
        for _ in range(3):
            sync()
            t0 = time.perf_counter()
            for g_, c in zip(gpu, cpu):
                g_.copy_(c, non_blocking=True)
            sync()
            times.append(time.perf_counter() - t0)
        r["ram_to_gpu_s"] = round(min(times), 4)
        r["ram_to_gpu_GBps"] = round(nbytes / min(times) / 1e9, 2)
        r["network_estimate_s"] = {"100Mbps": round(nbytes * 8 / 1e8, 1), "1Gbps": round(nbytes * 8 / 1e9, 2)}
        del cpu, gpu
        res["zones"][z] = r
        print(f"zone {z}: {nbytes / 2**30:.2f} GiB of MLP | disk cold {r['disk_cold_s']}s "
              f"(cache dropped: {r['page_cache_dropped']}) | warm {r['disk_warm_s']}s | RAM->GPU {r['ram_to_gpu_s']}s",
              flush=True)
    # overlap between zones and the bytes of a delta swap (new gate/up rows + every down column, which is refit per zone)
    names = list(kept)
    ov = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = kept[names[i]], kept[names[j]]
            shared = [len(np.intersect1d(A[k], B[k])) / len(A[k]) for k in A]
            ov[f"{names[i]}-{names[j]}"] = round(float(np.mean(shared)), 4)
    res["kept_neuron_overlap"] = ov
    if names:
        z0 = res["zones"][names[0]]["mlp_bytes"]
        mean_ov = float(np.mean(list(ov.values()))) if ov else 1.0
        res["delta_swap_bytes_estimate"] = int(z0 * (2 / 3) * (1 - mean_ov) + z0 / 3)
    Path(a.res_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.res_dir) / "swap.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ report
def cmd_report(a) -> None:
    rd = Path(a.res_dir)
    ev = {p.stem[len("eval_"):]: json.loads(p.read_text()) for p in sorted(rd.glob("eval_*.json"))}
    route = json.loads((rd / "route.json").read_text()) if (rd / "route.json").exists() else None
    swap = json.loads((rd / "swap.json").read_text()) if (rd / "swap.json").exists() else None
    exports = {}
    for p in sorted(rd.glob("export_*.json")):
        exports[p.stem[len("export_"):]] = json.loads(p.read_text())
    acc = {tag: {t: e["tasks"][t]["accuracy"] for t in TASKS} for tag, e in ev.items()}
    res = {"accuracy": acc, "checkpoint_bytes": {tag: e["checkpoint_bytes"] for tag, e in ev.items()}}
    zones = [z for z in ZONES if z in ev]
    if "full" in ev and zones:
        sysr = {}
        for t in TASKS:
            n = ev["full"]["tasks"][t]["n"]
            own = TASK_ZONE[t]
            row = {"full": acc["full"][t]}
            if own in ev:
                row["own_zone"] = acc[own][t]
            others = [acc[z][t] for z in zones if z != own]
            if others:
                row["wrong_zone_mean"] = round(float(np.mean(others)), 4)
                row["wrong_zone_worst"] = min(others)
            if route and all(z in ev for z in set(route["tasks"][t]["route"])):
                picks = route["tasks"][t]["route"]
                row["routed"] = round(float(np.mean([ev[z]["tasks"][t]["correct"][i] for i, z in enumerate(picks)])), 4)
                row["router_accuracy"] = route["tasks"][t]["accuracy"]
            row["n"] = n
            sysr[t] = row
        res["system"] = sysr
        res["system_mean"] = {k: round(float(np.mean([sysr[t][k] for t in TASKS if k in sysr[t]])), 4)
                              for k in ("full", "own_zone", "routed", "wrong_zone_mean") if all(k in sysr[t] for t in TASKS)}
    # zone variants (e.g. general_a = another calibration of the general zone): each one against its zone's own task,
    # and the system rebuilt with the best variant of every zone. That pick looks at the test, so it is a best case.
    variants = {}
    for tag in acc:
        z = tag.split("_")[0]
        if z in ZONES:
            own_t = next(t for t, zz in TASK_ZONE.items() if zz == z)
            ex = exports.get(tag, {})
            variants[tag] = {"zone": z, "own_task": own_t, "own_task_acc": acc[tag][own_t],
                             "calib_source": ex.get("calib_source"), "calib_tokens": ex.get("calib_tokens")}
    if any(tag not in ZONES for tag in variants):
        res["variants"] = variants
        best = {z: max((t for t, v in variants.items() if v["zone"] == z), key=lambda t: variants[t]["own_task_acc"])
                for z in ZONES if any(v["zone"] == z for v in variants.values())}
        if "full" in ev and route and len(best) == len(ZONES):
            sb = {}
            for t in TASKS:
                picks = route["tasks"][t]["route"]
                sb[t] = {"own_zone": acc[best[TASK_ZONE[t]]][t],
                         "routed": round(float(np.mean([ev[best[z]]["tasks"][t]["correct"][i] for i, z in enumerate(picks)])), 4)}
            res["system_best_variants"] = {"zones": best, "tasks": sb, "selected_on_test": True,
                                           "mean_routed": round(float(np.mean([sb[t]["routed"] for t in TASKS])), 4),
                                           "mean_full": res.get("system_mean", {}).get("full")}
    if route:
        res["router"] = {k: v for k, v in route.items() if k != "tasks"}
        res["router"]["accuracy"] = {t: route["tasks"][t]["accuracy"] for t in TASKS}
        res["router"]["counts"] = {t: route["tasks"][t]["counts"] for t in TASKS}
    if swap:
        res["swap"] = swap
    if exports:
        res["build"] = {z: {"calib_tokens": e.get("calib_tokens"), "timing": e.get("timing"),
                            "budgets": {k: v.get("time_s") for k, v in e.get("budgets", {}).items()}}
                        for z, e in exports.items()}
    res["eval_time_s"] = {tag: e.get("time_s") for tag, e in ev.items()}
    Path(a.out).write_text(json.dumps(res, indent=1))

    print(f"\n{'':10s}" + "".join(f"{t:>11s}" for t in TASKS) + "   checkpoint")
    for tag in ["full"] + zones + sorted(t for t in acc if t != "full" and t not in zones):
        if tag in acc:
            print(f"{tag:10s}" + "".join(f"{acc[tag][t]:11.1%}" for t in TASKS)
                  + f"   {res['checkpoint_bytes'][tag] / 2**30:.2f} GiB")
    if "system" in res:
        print("\nper task: full | own zone | routed | wrong zone (mean / worst) | router accuracy")
        for t, r in res["system"].items():
            print(f"  {t:10s} {r['full']:.1%} | {r.get('own_zone', float('nan')):.1%} | {r.get('routed', float('nan')):.1%} | "
                  f"{r.get('wrong_zone_mean', float('nan')):.1%} / {r.get('wrong_zone_worst', float('nan')):.1%} | "
                  f"{r.get('router_accuracy', float('nan')):.1%}")
        print("mean over tasks:", res["system_mean"])
    if "variants" in res:
        print("\nzone variants (own task):")
        for tag, v in res["variants"].items():
            print(f"  {tag:12s} {v['own_task']:10s} {v['own_task_acc']:.1%}  calib {v['calib_source']} {v['calib_tokens']}")
        if "system_best_variants" in res:
            print("best variant per zone (picked on the test):", res["system_best_variants"])
    if swap:
        for z, r in swap["zones"].items():
            print(f"swap {z}: {r['mlp_bytes'] / 2**30:.2f} GiB | disk cold {r['disk_cold_s']}s | warm {r['disk_warm_s']}s | "
                  f"RAM->GPU {r['ram_to_gpu_s']}s ({r['ram_to_gpu_GBps']} GB/s) | 1 Gbps ~{r['network_estimate_s']['1Gbps']}s")
        print("kept-neuron overlap:", swap.get("kept_neuron_overlap"))
    print("saved", a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "route"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", required=True)
        p.add_argument("--res-dir", required=True)
        p.add_argument("--n-gsm", type=int, default=200)
        p.add_argument("--n-mmlu", type=int, default=400)
        p.add_argument("--seed", type=int, default=0)
        if name == "eval":
            p.add_argument("--model-dir", required=True)
            p.add_argument("--tag", required=True)
            p.add_argument("--dtype", default="auto")
            p.add_argument("--max-model-len", type=int, default=4096)
            p.add_argument("--gpu-mem", type=float, default=0.85)
        else:
            p.add_argument("--model", required=True)
            p.add_argument("--n-train", type=int, default=300)
            p.add_argument("--reg", type=float, default=0.1, help="inverse L2 strength of the router's logistic regression")
    s = sub.add_parser("swap")
    s.add_argument("--zones", required=True, help="math=DIR,code=DIR,general=DIR")
    s.add_argument("--full", help="full model (hub id or dir), for the resident / MLP byte split")
    s.add_argument("--res-dir", required=True)
    s.add_argument("--cpu", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--res-dir", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"eval": cmd_eval, "route": cmd_route, "swap": cmd_swap, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
