"""Truly blind loop-fix discovery for Qwen3.5-0.8B: no answer-shaped rules, p50c kept out until the method is frozen.

Earlier discovery runs (discover_loop_fix.py v2/v3, wsi.tracer.discover_compact_circuit) carried rules that point at
the known answer: only full-attention layers, an origin range, "destination before the last layer", and defaults
(alpha 6, 64 neurons) equal to p50c's. This script removes them. What is still assumed, and stated:
  * the edit form   W' = W - alpha * u (m * u^T W)   on one MLP down_proj per chosen layer (the form p50c uses);
  * the loop definitions of this repository (router: >= 50% repeated 16-grams over the last 64 tokens; kill: >= 90%
    over the last 768), applied to every arm alike;
  * GSM8K: discovery and tuning use TRAIN problems only; TEST is touched once, after the freeze.

Pipeline
  prepare    elicit (300 train, 8k) + dev1 (32 train, 16k) + dev2 (64 other train, 16k) + test (200, 16k) requests
  gen        vLLM generation (same as discover_loop_fix.py)
  diagnose   per layer, ALL layers: u_l = mean MLP output on loop tokens - on normal tokens (half A); held-out d'_l
             on half B; neuron score c_i = (E[h_i|loop] - E[h_i|normal]) * (u . w_i)
  search     round 1: every (k, n, alpha) in K x NEURONS x ALPHAS, layers = the k layers with the largest d', on dev1;
             round 2: the best TOP2 of round 1 on dev2 (fresh problems); the best on dev2 is the choice.
             Score = (accuracy, then fewer loop kills, then fewer truncations). Resumable.
  freeze     writes frozen.json: the choice + SHA-256 of this script and its helpers. Nothing below runs without it.
  plan-moi   cut each base test response where the router fires (same as discover_loop_fix.py)
  build      write a checkpoint for one config; --direction random gives the control (same layers, neurons, alpha)
  ---- exam, after the freeze ----
  report     TEST arms base / blind_fixed / blind_moi / random_fixed / random_moi / p50c, paired CIs; the exam reads
             p50c's layers and columns from the weight diff (nothing hard-coded) and compares them with the choice.

    python discover_blind.py prepare --work W
    python discover_blind.py gen --work W --model W/qwen-base --requests requests_elicit.jsonl --out outputs_elicit.jsonl
    python discover_blind.py diagnose --work W
    python discover_blind.py search --work W --round 1 ; python discover_blind.py search --work W --round 2
    python discover_blind.py freeze --work W
    ... test generations (see notebooks/discover_blind_colab.ipynb) ...
    python discover_blind.py report --work W --p50c-dir P

tests/test_discover_blind.py checks that the discovery part of this file never names p50c, never reads layer types,
has no layer-number constants, and only calls the generic helpers of discover_loop_fix.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402

# ------------------------------------------------------------------ frozen protocol (change = new commit before a run)
K_LAYERS = (1, 2, 3)            # how many of the most separated layers to edit
NEURONS = (16, 64, 256, "all")  # columns of down_proj kept in the mask m
ALPHAS = (2.0, 4.0, 6.0, 8.0, 12.0)
TOP2 = 6                        # round-1 configs re-tested on dev2
RANDOM_SEED = 1234              # control direction


def score_key(s: dict) -> tuple:
    return (s["accuracy"], -s["loop_killed"], -s["truncated"])


def cfg_name(layers: list[int], n, alpha: float, direction: str = "wsi") -> str:
    return f"{'rand_' if direction == 'random' else ''}L{'-'.join(map(str, layers))}_n{n}_a{alpha:g}"


def grid(rank: list[int]) -> list[dict]:
    """All round-1 configs, from the held-out separation ranking only."""
    out = []
    for k in K_LAYERS:
        for n in NEURONS:
            for al in ALPHAS:
                layers = sorted(rank[:k])
                out.append({"name": cfg_name(layers, n, al), "layers": layers, "neurons": n, "alpha": al, "k": k})
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {f: sha256_file(here / f) for f in ("discover_blind.py", "discover_loop_fix.py", "moi_loop_router_vllm.py")}


# ------------------------------------------------------------------ prepare
def cmd_prepare(a) -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.base_dir:
        base_dir = Path(a.base_dir)
    else:
        from huggingface_hub import snapshot_download

        base_dir = Path(snapshot_download(lr.BASE_REPO, local_dir=work / "qwen-base"))

    def parquet(split: str) -> Path:
        if a.data_dir and (Path(a.data_dir) / f"{split}.parquet").exists():
            return Path(a.data_dir) / f"{split}.parquet"
        return Path(hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset",
                                    local_dir=work / "data"))

    tok = AutoTokenizer.from_pretrained(base_dir)
    train = pd.read_parquet(parquet("train")).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    test = pd.read_parquet(parquet("test")).iloc[: a.n_test]
    e, d1, d2 = a.n_elicit, a.n_dev1, a.n_dev2
    lr.write_jsonl(work / "requests_elicit.jsonl", dl.make_requests(tok, train.iloc[:e], a.seed, a.elicit_max_new))
    lr.write_jsonl(work / "requests_dev1.jsonl",
                   dl.make_requests(tok, train.iloc[e:e + d1], a.seed, a.dev_max_new, offset=100_000))
    lr.write_jsonl(work / "requests_dev2.jsonl",
                   dl.make_requests(tok, train.iloc[e + d1:e + d1 + d2], a.seed, a.dev_max_new, offset=200_000))
    lr.write_jsonl(work / "requests_test.jsonl", dl.make_requests(tok, test, a.seed, a.max_new))
    dl.cfg_save(work, {"base_dir": str(base_dir), "eos": lr.eos_ids(tok), "max_new": a.max_new, "seed": a.seed,
                       "n_elicit": e, "n_dev1": d1, "n_dev2": d2, "n_test": len(test), "dev_max_new": a.dev_max_new,
                       "protocol": {"K_LAYERS": list(K_LAYERS), "NEURONS": list(NEURONS), "ALPHAS": list(ALPHAS),
                                    "TOP2": TOP2}})
    print(f"requests: {e} elicit + {d1} dev1 + {d2} dev2 (GSM8K train, disjoint) + {len(test)} test -> {work}")


# ------------------------------------------------------------------ diagnose (transformers), every layer
def cmd_diagnose(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_elicit.jsonl")}
    outs = lr.read_jsonl(work / "outputs_elicit.jsonl")
    seqs, n_loop_seq = [], 0
    for o in outs:
        body = lr.strip_eos(o["token_ids"], eos)
        lab = dl.token_labels(body)
        if not (lab >= 0).any():
            continue
        first = int(np.argmax(lab == 1)) if (lab == 1).any() else None
        end = min(len(body), (first + a.after_onset) if first is not None else a.max_seq, a.max_seq)
        n_loop_seq += first is not None
        seqs.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body[:end], "lab": lab[:end]})
    loop_tok = sum(int((s["lab"] == 1).sum()) for s in seqs)
    norm_tok = sum(int((s["lab"] == 0).sum()) for s in seqs)
    print(f"elicit: {len(outs)} responses, {n_loop_seq} with loops; tokens {loop_tok:,} looping, {norm_tok:,} normal")
    if loop_tok < a.min_loop_tokens:
        raise SystemExit("too few looping tokens to diagnose; raise --n-elicit or --elicit-max-new")
    half = np.random.default_rng(cfg["seed"]).permutation(len(seqs)) % 2

    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["base_dir"], dtype=dtype).to(dev).eval()
    layers = next(m for n, m in model.named_modules()
                  if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 2)
    L = len(layers)
    downs = [lyr.mlp.down_proj for lyr in layers]
    U, D = downs[0].in_features, downs[0].out_features
    body_model = getattr(model, "model", model)
    cur: dict = {"mask": None}
    acc: dict = {}

    def hook(li):
        def f(mod, inp, out):
            m = cur["mask"]
            h, y = inp[0][0].float(), out[0].float()
            for cls in (0, 1):
                sel = m == cls
                if not sel.any():
                    continue
                k = (li, cls)
                if cur["pass"] == "A":
                    s = acc.setdefault(k, {"n": 0, "h": torch.zeros(U, device=dev, dtype=torch.float64),
                                           "y": torch.zeros(D, device=dev, dtype=torch.float64)})
                    s["n"] += int(sel.sum())
                    s["h"] += h[sel].sum(0).double()
                    s["y"] += y[sel].sum(0).double()
                else:
                    p = (y[sel] @ cur["u"][li]).double()
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
                ids = torch.tensor([s["prompt"] + s["body"]], device=dev)
                full = np.full(ids.shape[1], -1, dtype=np.int64)
                full[len(s["prompt"]):] = s["lab"]
                cur["mask"] = torch.from_numpy(full).to(dev)
                body_model(input_ids=ids, use_cache=False)

    run(0, "A")
    missing = [k for k in ((li, c) for li in range(L) for c in (0, 1)) if k not in acc]
    if missing:
        raise SystemExit(f"half A lacks loop or normal tokens for {missing[:4]}...; elicit more")
    u = []
    for li in range(L):
        d = acc[(li, 1)]["y"] / acc[(li, 1)]["n"] - acc[(li, 0)]["y"] / acc[(li, 0)]["n"]
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
        dh = acc[(li, 1)]["h"] / acc[(li, 1)]["n"] - acc[(li, 0)]["h"] / acc[(li, 0)]["n"]
        c = dh * (u[li].double() @ downs[li].weight.double())
        neuron_rank[str(li)] = [int(x) for x in torch.argsort(c, descending=True).tolist()]
    np.savez(work / "blind_directions.npz", **{f"u{li}": u[li].cpu().numpy() for li in range(L)})
    diag = {"separation": sep, "layer_rank": rank, "neuron_rank": neuron_rank, "n_layers": L, "n_units": U,
            "hidden": D, "loop_tokens": loop_tok, "normal_tokens": norm_tok, "sequences": len(seqs),
            "looping_sequences": n_loop_seq}
    (work / "diagnosis_blind.json").write_text(json.dumps(diag, indent=1))
    print("held-out d' per layer:", " ".join(f"L{li}:{s:.2f}" for li, s in enumerate(sep)))
    print("layer rank:", rank)


# ------------------------------------------------------------------ build one config
def columns(diag: dict, layer: int, n) -> list[int]:
    r = diag["neuron_rank"][str(layer)]
    return sorted(r if n == "all" else r[: int(n)])


def direction(diag: dict, z, layer: int, kind: str) -> np.ndarray:
    if kind == "wsi":
        return z[f"u{layer}"].astype(np.float64)
    v = np.random.default_rng(RANDOM_SEED + layer).standard_normal(diag["hidden"])
    return v / np.linalg.norm(v)


def build_checkpoint(work: Path, layers: list[int], n, alpha: float, kind: str, out: Path) -> Path:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = dl.cfg_load(work)
    base = Path(cfg["base_dir"])
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    z = np.load(work / "blind_directions.npz")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns(".cache", "*.lock"))
    done = set()
    for f in sorted(base.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            keys = list(fh.keys())
            targets = {}
            for li in layers:
                try:
                    targets[dl.down_key(keys, li)] = li
                except SystemExit:
                    continue
            if not targets:
                continue
            tensors = {k: fh.get_tensor(k).clone() for k in keys}
            meta = fh.metadata() or {"format": "pt"}
        for k, li in targets.items():
            W = tensors[k].double()
            uu = torch.from_numpy(direction(diag, z, li, kind))
            m = torch.zeros(W.shape[1], dtype=torch.float64)
            m[columns(diag, li, n)] = 1.0
            tensors[k] = (W - alpha * torch.outer(uu, m * (uu @ W))).to(tensors[k].dtype)
            done.add(li)
        save_file(tensors, str(out / f.name), metadata=meta)
    if done != set(layers):
        raise SystemExit(f"down_proj not found for layers {sorted(set(layers) - done)}")
    return out


def parse_n(s: str):
    return "all" if s == "all" else int(s)


def cmd_build(a) -> None:
    work = Path(a.work)
    layers = [int(x) for x in a.layers.split(",")]
    n = parse_n(a.neurons)
    out = Path(a.out_dir) if a.out_dir else work / cfg_name(layers, n, a.alpha, a.direction)
    build_checkpoint(work, layers, n, a.alpha, a.direction, out)
    print(f"built {out}")


def chosen(work: Path) -> dict:
    p = work / "frozen.json"
    if not p.exists():
        raise SystemExit("frozen.json missing: run `freeze` after search round 2")
    return json.loads(p.read_text())["choice"]


def cmd_build_chosen(a) -> None:
    work = Path(a.work)
    c = chosen(work)
    out = work / cfg_name(c["layers"], c["neurons"], c["alpha"], a.direction)
    build_checkpoint(work, c["layers"], c["neurons"], c["alpha"], a.direction, out)
    print(out)


# ------------------------------------------------------------------ search (dev only, resumable)
def _run(args: list[str]) -> None:
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args])
    if r.returncode:
        raise SystemExit(f"step failed: {args[:2]}")


def _sync(work: Path, sync_dir: str | None) -> None:
    if not sync_dir:
        return
    Path(sync_dir).mkdir(parents=True, exist_ok=True)
    for pat in ("*.json", "*.jsonl", "*.npz"):
        for f in work.glob(pat):
            shutil.copy2(f, sync_dir)


def cmd_search(a) -> None:
    work = Path(a.work)
    cfg = dl.cfg_load(work)
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    sp = work / "search.json"
    state = json.loads(sp.read_text()) if sp.exists() else {"round1": {}, "round2": {}, "time_s": {}}
    split = f"dev{a.round}"
    if a.round == 1:
        configs = grid(diag["layer_rank"])
    else:
        r1 = state["round1"]
        if len(r1) < len(grid(diag["layer_rank"])):
            raise SystemExit("round 1 is not finished")
        best = sorted(r1, key=lambda k: score_key(r1[k]), reverse=True)[:TOP2]
        configs = [next(c for c in grid(diag["layer_rank"]) if c["name"] == b) for b in best]
    t0 = time.perf_counter()
    req = f"requests_{split}.jsonl"
    if not (work / f"outputs_{split}_base.jsonl").exists():
        _run(["gen", "--work", str(work), "--model", cfg["base_dir"], "--requests", req,
              "--out", f"outputs_{split}_base.jsonl", "--gpu-mem", str(a.gpu_mem)])
        _sync(work, a.sync_dir)
    base = dl.score(work, req, f"outputs_{split}_base.jsonl")
    state[f"base_{split}"] = {k: v for k, v in base.items() if k != "rows"}
    key = f"round{a.round}"
    for i, c in enumerate(configs):
        if c["name"] in state[key]:
            continue
        out_f = f"outputs_{split}_{c['name']}.jsonl"
        if not (work / out_f).exists():
            ck = work / c["name"]
            build_checkpoint(work, c["layers"], c["neurons"], c["alpha"], "wsi", ck)
            _run(["gen", "--work", str(work), "--model", str(ck), "--requests", req, "--out", out_f,
                  "--gpu-mem", str(a.gpu_mem)])
            shutil.rmtree(ck, ignore_errors=True)
        s = dl.score(work, req, out_f)
        state[key][c["name"]] = {**{k: v for k, v in s.items() if k != "rows"}, **c}
        state["time_s"][key] = round(state["time_s"].get(key, 0.0) + time.perf_counter() - t0, 1)
        t0 = time.perf_counter()
        sp.write_text(json.dumps(state, indent=1))
        _sync(work, a.sync_dir)
        print(f"[{key} {i + 1}/{len(configs)}] {c['name']}: acc={s['accuracy']:.1%} loops={s['loop_killed']:.1%} "
              f"trunc={s['truncated']:.1%}  (base {base['accuracy']:.1%})", flush=True)
    state["time_s"][key] = round(state["time_s"].get(key, 0.0) + time.perf_counter() - t0, 1)
    if a.round == 2:
        r2 = state["round2"]
        order = [c["name"] for c in configs]  # ties -> better in round 1
        state["choice"] = dict(r2[max(order, key=lambda k: (score_key(r2[k]), -order.index(k)))])
        print("choice:", state["choice"]["name"])
    sp.write_text(json.dumps(state, indent=1))
    _sync(work, a.sync_dir)


def cmd_freeze(a) -> None:
    work = Path(a.work)
    st = json.loads((work / "search.json").read_text())
    if "choice" not in st:
        raise SystemExit("search round 2 has not picked a choice")
    p = work / "frozen.json"
    c = {k: st["choice"][k] for k in ("name", "layers", "neurons", "alpha", "k")}
    if p.exists():
        old = json.loads(p.read_text())
        if old["choice"] != c:
            raise SystemExit(f"already frozen with a different choice: {old['choice']}")
        print("already frozen:", old["choice"]["name"])
        return
    t = st["time_s"]
    timing = json.loads((work / "timing.json").read_text()) if (work / "timing.json").exists() else {}
    elicit = sum(v["s"] for k, v in timing.items() if k.startswith(("gen:outputs_elicit", "diagnose")))
    disc_s = elicit + t.get("round1", 0) + t.get("round2", 0)
    p.write_text(json.dumps({"choice": c, "code_sha256": code_hashes(), "git_commit": a.git_commit,
                             "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "discovery_s": round(disc_s, 1), "discovery_min": round(disc_s / 60, 1)}, indent=1))
    print(f"frozen: {c['name']}  (discovery {disc_s / 60:.1f} min)")


# ===== EXAM: runs only after freeze =====
def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    fr = json.loads((work / "frozen.json").read_text()) if (work / "frozen.json").exists() else None
    if fr is None:
        raise SystemExit("frozen.json missing: the exam runs only after the method is frozen")
    if fr["code_sha256"] != code_hashes():
        raise SystemExit("the code changed after the freeze; refusing to report")
    c = fr["choice"]
    cfg = dl.cfg_load(work)
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"])
    eos, reqs = set(cfg["eos"]), {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    arms = {"base": "outputs_test_base.jsonl", "blind_fixed": "outputs_test_blind.jsonl",
            "random_fixed": "outputs_test_random.jsonl", "p50c": "outputs_test_p50c.jsonl"}
    res: dict = {"choice": c, "frozen": {k: fr[k] for k in ("frozen_utc", "git_commit", "code_sha256")},
                 "discovery_min": fr["discovery_min"], "arms": {}}
    rows = {}
    for name, f in arms.items():
        if (work / f).exists():
            s = dl.score(work, "requests_test.jsonl", f, tok)
            rows[name] = {r["id"]: r for r in s["rows"]}
            res["arms"][name] = {k: v for k, v in s.items() if k != "rows"}
    plan = {r["id"]: r for r in lr.read_jsonl(work / "requests_test_moi.jsonl")}
    for name, f in (("blind_moi", "outputs_test_moi_blind.jsonl"), ("random_moi", "outputs_test_moi_random.jsonl")):
        if not (work / f).exists():
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
    pairs = [("blind_fixed", "base"), ("blind_moi", "base"), ("blind_fixed", "random_fixed"),
             ("blind_moi", "random_moi"), ("blind_fixed", "p50c"), ("blind_moi", "p50c"), ("p50c", "base")]
    res["paired"] = {f"{x}_vs_{y}": lr.paired_ci(acc[x], acc[y]) for x, y in pairs if x in acc and y in acc}
    if a.p50c_dir:
        res["exam_vs_p50c"] = exam(work, cfg, c, Path(a.p50c_dir))
    st = json.loads((work / "search.json").read_text())
    res["search"] = {"base_dev1": st.get("base_dev1"), "base_dev2": st.get("base_dev2"),
                     "round1": {k: {x: v[x] for x in ("accuracy", "loop_killed", "truncated")}
                                for k, v in st["round1"].items()},
                     "round2": {k: {x: v[x] for x in ("accuracy", "loop_killed", "truncated")}
                                for k, v in st["round2"].items()}, "time_s": st["time_s"]}
    res["diagnosis"] = {k: v for k, v in json.loads((work / "diagnosis_blind.json").read_text()).items()
                        if k != "neuron_rank"}
    out = work / a.out
    out.write_text(json.dumps(res, indent=1))
    print(f"\nchoice: {c['name']}  discovery {fr['discovery_min']} min")
    print(f"{'arm':13s} {'acc':>7s} {'loops':>7s} {'trunc':>7s}")
    for k, v in res["arms"].items():
        print(f"{k:13s} {v['accuracy']:7.1%} {v['loop_killed']:7.1%} {v['truncated']:7.1%}")
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
    if "exam_vs_p50c" in res:
        print("exam:", json.dumps(res["exam_vs_p50c"], indent=1))
    print("Saved", out)


def exam(work: Path, cfg: dict, c: dict, p50c_dir: Path) -> dict:
    """p50c's edited tensors, layers and columns, read from its weight diff vs the base checkpoint."""
    import re

    from safetensors import safe_open

    base = Path(cfg["base_dir"])
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    z = np.load(work / "blind_directions.npz")
    changed, per_layer = [], {}
    for f in sorted(p50c_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fe, safe_open(str(base / f.name), "pt") as fb:
            for k in fe.keys():
                a_, b_ = fe.get_tensor(k), fb.get_tensor(k)
                if a_.shape == b_.shape and bool((a_ != b_).any()):
                    changed.append(k)
                    m = re.search(r"layers\.(\d+)\.mlp\.down_proj\.weight$", k)
                    if m and not k.startswith("mtp."):
                        per_layer[int(m.group(1))] = (a_.double() - b_.double()).numpy()
    out = {"p50c_changed_tensors": changed, "p50c_layers": sorted(per_layer), "blind_layers": c["layers"],
           "layer_overlap": sorted(set(per_layer) & set(c["layers"])),
           "p50c_layer_rank_in_blind_dprime": {str(li): diag["layer_rank"].index(li) + 1 for li in per_layer},
           "per_layer": {}}
    for li, dW in per_layer.items():
        cols = np.where(np.abs(dW).sum(0) > 0)[0]
        up = np.linalg.svd(dW[:, cols], full_matrices=False)[0][:, 0]
        uw = z[f"u{li}"].astype(np.float64)
        rec = {"p50c_columns": int(len(cols)), "dprime": round(diag["separation"][li], 4),
               "abs_cos_blind_direction_vs_p50c": round(float(abs(uw @ up) / np.linalg.norm(uw)), 4),
               "random_baseline": round(1 / np.sqrt(len(up)), 4)}
        if li in c["layers"]:
            mine = set(columns(diag, li, c["neurons"]))
            rec["neuron_overlap"] = f"{len(mine & set(cols.tolist()))}/{len(cols)} (blind keeps {len(mine)})"
        elif str(li) in diag["neuron_rank"]:
            top = set(diag["neuron_rank"][str(li)][: len(cols)])
            rec["top_same_count_overlap"] = f"{len(top & set(cols.tolist()))}/{len(cols)}"
        out["per_layer"][str(li)] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--base-dir")
    p.add_argument("--data-dir")
    p.add_argument("--n-elicit", type=int, default=300)
    p.add_argument("--elicit-max-new", type=int, default=8192)
    p.add_argument("--n-dev1", type=int, default=32)
    p.add_argument("--n-dev2", type=int, default=64)
    p.add_argument("--dev-max-new", type=int, default=16384)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--max-new", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--requests", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--gpu-mem", type=float, default=0.88)
    g.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--work", required=True)
    d.add_argument("--after-onset", type=int, default=768)
    d.add_argument("--max-seq", type=int, default=4096)
    d.add_argument("--min-loop-tokens", type=int, default=500)
    d.add_argument("--cpu", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--layers", required=True)
    b.add_argument("--neurons", required=True)
    b.add_argument("--alpha", type=float, required=True)
    b.add_argument("--direction", choices=["wsi", "random"], default="wsi")
    b.add_argument("--out-dir")
    bc = sub.add_parser("build-chosen")
    bc.add_argument("--work", required=True)
    bc.add_argument("--direction", choices=["wsi", "random"], default="wsi")
    s = sub.add_parser("search")
    s.add_argument("--work", required=True)
    s.add_argument("--round", type=int, choices=[1, 2], required=True)
    s.add_argument("--gpu-mem", type=float, default=0.88)
    s.add_argument("--sync-dir", help="copy small result files here after every config (Drive)")
    f = sub.add_parser("freeze")
    f.add_argument("--work", required=True)
    f.add_argument("--git-commit", default=None)
    m = sub.add_parser("plan-moi")
    m.add_argument("--work", required=True)
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--p50c-dir")
    r.add_argument("--out", default="results_discover_blind.json")
    a = ap.parse_args()
    fns = {"prepare": cmd_prepare, "gen": dl.cmd_gen, "diagnose": cmd_diagnose, "build": cmd_build,
           "build-chosen": cmd_build_chosen, "search": cmd_search, "freeze": cmd_freeze, "plan-moi": dl.cmd_plan_moi,
           "report": cmd_report}
    key = {"gen": lambda: f"gen:{Path(a.out).stem}:{Path(a.model).name}", "diagnose": lambda: "diagnose"}.get(a.cmd)
    dl.run_timed(a, fns, key() if key else None)


if __name__ == "__main__":
    main()
