"""Can WSI find a reasoning-loop fix in Qwen3.5-0.8B on its own, from the BASE model only?

Nothing from the MIT CSAIL p50c release is used to build the fix. p50c is loaded only at the very end, as an exam:
does the discovered fix work as well, and did it find the same layers / direction / neurons?

Stages (each GPU stage runs in its own process; vLLM for generation, transformers for capture):
  elicit    let the base model reason on GSM8K TRAIN problems (think35 sampling) and record which outputs loop.
  diagnose  capture the MLP outputs and hidden units on looping vs normal response tokens (split in two halves):
              * direction u_l = mean MLP output on loop tokens - on normal tokens (half A), per layer
              * separation d'_l of the projection on u_l, measured on half B (held out)
              * choose the layers with the largest d'; in them, the neurons whose activation change pushes
                along u most:  c_i = (E[h_i|loop] - E[h_i|normal]) * (u . w_i)
  build     write the fix with the same form as p50c:  W' = W - alpha * u (m * u^T W)   (m = chosen columns)
  gen       generate with vLLM for a list of requests and a checkpoint
  tune      pick alpha on held-out TRAIN problems (accuracy, then fewer loops)
  plan-moi  MoI arm: cut each base test response where the loop router fires; the fixed model continues it
  report    GSM8K TEST: base vs WSI fix always on vs WSI fix via MoI vs p50c, plus the exam against p50c.

Uses helpers from moi_loop_router_vllm.py (same folder). Usage: see notebooks/discover_loop_fix_colab.ipynb.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moi_loop_router_vllm as lr  # noqa: E402

LOOP_FRAC, LOOP_WINDOW = lr.ROUTER_FRAC, lr.ROUTER_WINDOW  # a token is "looping" when the router would fire there
CLEAN_WINDOW = 256  # a token is "normal" when the last 256 response tokens have no repeated 16-gram


# ------------------------------------------------------------------ timing (saved to work/timing.json, backed up with the results)
def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "unknown"


def record_time(work: Path, key: str, seconds: float) -> None:
    p = work / "timing.json"
    t = json.loads(p.read_text()) if p.exists() else {}
    t[key] = {"s": round(seconds, 1), "gpu": _gpu_name()}
    p.write_text(json.dumps(t, indent=1))
    print(f"[time] {key}: {seconds / 60:.1f} min", flush=True)


def run_timed(a, fns: dict, key: str | None) -> None:
    """Run one subcommand; unless key is None, store its wall time (model loading included) in timing.json."""
    t0 = time.perf_counter()
    fns[a.cmd](a)
    if key:
        record_time(Path(a.work), key, time.perf_counter() - t0)


DISCOVERY_PREFIXES = ("gen:outputs_elicit", "diagnose", "build:", "gen:outputs_dev", "pick", "sft-data", "train",
                      "export")


def timing_summary(work: Path) -> dict:
    """Raw per-step times plus the cost of FINDING the fix: model reasoning on train problems, diagnosis, building
    each candidate fix and tuning it on dev problems (test-set evaluation and the p50c exam are not counted)."""
    p = work / "timing.json"
    if not p.exists():
        return {}
    t = json.loads(p.read_text())
    disc = {k: v["s"] for k, v in t.items() if k.startswith(DISCOVERY_PREFIXES)}
    return {"steps": t, "discovery_steps": sorted(disc), "discovery_s": round(sum(disc.values()), 1),
            "discovery_min": round(sum(disc.values()) / 60, 1)}


# ------------------------------------------------------------------ helpers
def cfg_load(work: Path) -> dict:
    return json.loads((work / "config.json").read_text())


def cfg_save(work: Path, cfg: dict) -> None:
    (work / "config.json").write_text(json.dumps(cfg, indent=2))


def make_requests(tok, df, seed: int, max_new: int, offset: int = 0) -> list[dict]:
    reqs = []
    for i, (q, a) in enumerate(zip(df["question"], df["answer"])):
        ids = tok.apply_chat_template([{"role": "user", "content": q + "\n" + lr.INSTRUCTION}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        reqs.append({"id": offset + i, "prompt_ids": list(map(int, ids)),
                     "gold": a.split("####")[-1].strip().replace(",", ""),
                     "seed": seed * 100_003 + offset + i, "max_tokens": max_new})
    return reqs


def token_labels(body: list[int]) -> np.ndarray:
    """Per response token: 1 = looping (>= 50% dup over the last 64), 0 = normal (no dup in the last 256), -1 = other."""
    d = np.array(lr.dup_flags(body), dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(d)])
    n = len(d)
    lab = np.full(n, -1, dtype=np.int64)
    for t in range(n):
        if t + 1 >= LOOP_WINDOW and (c[t + 1] - c[t + 1 - LOOP_WINDOW]) >= LOOP_FRAC * LOOP_WINDOW:
            lab[t] = 1
        elif t + 1 >= CLEAN_WINDOW and c[t + 1] - c[t + 1 - CLEAN_WINDOW] == 0:
            lab[t] = 0
    return lab


def down_key(keys, layer: int) -> str:
    pat = re.compile(rf"(^|\.)layers\.{layer}\.(?:mlp\.down_proj|feed_forward\.w2)\.weight$")
    hits = [k for k in keys if pat.search(k) and not k.startswith("mtp.")]  # Qwen3.5 also has mtp.layers.0
    if len(hits) != 1:
        raise SystemExit(f"down_proj of layer {layer} not found uniquely: {hits}")
    return hits[0]


# ------------------------------------------------------------------ prepare
def cmd_prepare(a) -> None:
    import pandas as pd
    from transformers import AutoTokenizer

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.base_dir:
        base_dir = Path(a.base_dir)
    else:
        from huggingface_hub import snapshot_download

        base_dir = Path(snapshot_download(lr.BASE_REPO, local_dir=work / "qwen-base"))
    from huggingface_hub import hf_hub_download

    def parquet(split: str) -> Path:
        if a.data_dir and (Path(a.data_dir) / f"{split}.parquet").exists():
            return Path(a.data_dir) / f"{split}.parquet"
        return Path(hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset",
                                    local_dir=work / "data"))

    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    train = pd.read_parquet(parquet("train")).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    test = pd.read_parquet(parquet("test")).iloc[: a.n_test]
    lr.write_jsonl(work / "requests_elicit.jsonl",
                   make_requests(tok, train.iloc[: a.n_elicit], a.seed, a.elicit_max_new, offset=0))
    lr.write_jsonl(work / "requests_dev.jsonl",
                   make_requests(tok, train.iloc[a.n_elicit : a.n_elicit + a.n_dev], a.seed, a.dev_max_new,
                                 offset=100_000))
    lr.write_jsonl(work / "requests_test.jsonl", make_requests(tok, test, a.seed, a.max_new, offset=0))
    cfg_save(work, {"base_dir": str(base_dir), "eos": lr.eos_ids(tok), "max_new": a.max_new, "seed": a.seed,
                    "n_elicit": a.n_elicit, "n_dev": a.n_dev, "n_test": len(test)})
    print(f"requests: {a.n_elicit} elicit + {a.n_dev} dev (GSM8K train, disjoint) + {len(test)} test -> {work}")


# ------------------------------------------------------------------ generation (vLLM)
def cmd_gen(a) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    work = Path(a.work)
    cfg = cfg_load(work)
    reqs = lr.read_jsonl(work / a.requests)
    out_p = work / a.out
    if not reqs:
        lr.write_jsonl(out_p, [])
        return
    max_len = max(len(r["prompt_ids"]) + r["max_tokens"] for r in reqs) + 16
    base_kw = dict(model=a.model, tokenizer=cfg["base_dir"], dtype="bfloat16", max_model_len=max_len,
                   gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=cfg["seed"])
    # speed-ups, each dropped if this vLLM rejects it:
    #   text only  -> skips the vision encoder's profiling / warm-up (Qwen3.5 is multimodal; we never send images)
    #   loop kill  -> stops a response where the kill rule fires instead of generating the dead loop to the cap
    extras = {"limit_mm_per_prompt": {"image": 0, "video": 0}}
    use_kill = a.early_stop and lr.LoopKillLogitsProcessor is not None
    if use_kill:
        extras["logits_processors"] = [lr.LoopKillLogitsProcessor]
    llm = None
    for drop in ([], ["limit_mm_per_prompt"], ["logits_processors"], ["limit_mm_per_prompt", "logits_processors"]):
        kw = {**base_kw, **{k: v for k, v in extras.items() if k not in drop}}
        try:
            llm = LLM(**kw)
            use_kill = use_kill and "logits_processors" in kw
            print(f"[gen] vLLM options: text_only={'limit_mm_per_prompt' in kw} early_loop_stop={use_kill}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[gen] vLLM refused {sorted(set(extras) - set(drop))}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            import gc

            gc.collect()
    if llm is None:
        raise SystemExit("vLLM could not start")
    kill_eos = cfg["eos"][-1]

    def params(r, kill: bool):
        return SamplingParams(**lr.sampling_for(cfg), max_tokens=r["max_tokens"], seed=r["seed"], stop_token_ids=cfg["eos"],
                              logit_bias={int(k): float(v) for k, v in r.get("logit_bias", {}).items()} or None,
                              extra_args={"loop_kill_eos": kill_eos} if kill else None)

    prompts = [TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in reqs]
    t0 = time.time()
    try:
        outs = llm.generate(prompts, [params(r, use_kill) for r in reqs], use_tqdm=True)
    except Exception as e:  # noqa: BLE001
        if not use_kill:
            raise
        print(f"[gen] early loop stop failed ({type(e).__name__}: {str(e)[:200]}); generating without it", flush=True)
        use_kill = False
        outs = llm.generate(prompts, [params(r, False) for r in reqs], use_tqdm=True)
    dt = time.time() - t0
    rows = [{"id": r["id"], "token_ids": list(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason}
            for r, o in zip(reqs, outs)]
    lr.write_jsonl(out_p, rows)
    ntok = sum(len(x["token_ids"]) for x in rows)
    stopped = 0
    if use_kill:  # sanity: responses ended by the forced EOS = the kill rule holds right before the final token
        for x in rows:
            ids = x["token_ids"]
            if ids and ids[-1] == kill_eos and lr.LoopKillState(kill_eos).killed(ids[:-1]):
                stopped += 1
    print(f"[gen {a.out}] {len(rows)} seqs, {ntok:,} tokens, {dt / 60:.1f} min ({ntok / dt:,.0f} tok/s)"
          + (f"; {stopped} loops stopped early" if use_kill else ""))


def score(work: Path, req_file: str, out_file: str, tok=None) -> dict:
    from transformers import AutoTokenizer

    cfg = cfg_load(work)
    tok = tok or AutoTokenizer.from_pretrained(cfg["base_dir"], trust_remote_code=True)
    reqs = {r["id"]: r for r in lr.read_jsonl(work / req_file)}
    rows = []
    for o in lr.read_jsonl(work / out_file):
        r = reqs[o["id"]]
        rows.append({"id": o["id"], **lr.grade(o["token_ids"], o["finish_reason"], r["gold"], tok, set(cfg["eos"]),
                                                r["max_tokens"])})
    n = max(len(rows), 1)
    return {"n": len(rows), "accuracy": sum(x["correct"] for x in rows) / n,
            "loop_killed": sum(x["looped_killed"] for x in rows) / n,
            "truncated": sum(x["truncated"] for x in rows) / n, "rows": rows}


# ------------------------------------------------------------------ diagnose (transformers)
def cmd_diagnose(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    cfg = cfg_load(work)
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_elicit.jsonl")}
    outs = lr.read_jsonl(work / "outputs_elicit.jsonl")

    # which sequences, and which of their tokens, are looping vs normal
    seqs = []
    n_loop_seq = 0
    for o in outs:
        body = lr.strip_eos(o["token_ids"], eos)
        lab = token_labels(body)
        if not (lab >= 0).any():
            continue
        first_loop = int(np.argmax(lab == 1)) if (lab == 1).any() else None
        end = min(len(body), (first_loop + a.after_onset) if first_loop is not None else a.max_seq, a.max_seq)
        n_loop_seq += first_loop is not None
        seqs.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body[:end], "lab": lab[:end]})
    loop_tok = sum(int((s["lab"] == 1).sum()) for s in seqs)
    norm_tok = sum(int((s["lab"] == 0).sum()) for s in seqs)
    print(f"elicit: {len(outs)} responses, {n_loop_seq} with loops; labelled tokens: {loop_tok:,} looping, "
          f"{norm_tok:,} normal")
    if loop_tok < 500:
        raise SystemExit("too few looping tokens to diagnose; raise --n-elicit or --elicit-max-new")
    rng = np.random.default_rng(cfg["seed"])
    half = rng.permutation(len(seqs)) % 2  # 0 = A (fit), 1 = B (held-out separation)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["base_dir"], dtype=dtype, trust_remote_code=True).to(dev).eval()
    layers = next(m for n, m in model.named_modules()
                  if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 8)
    L = len(layers)
    downs = [lr.down_module(lyr) for lyr in layers]
    U, D = downs[0].in_features, downs[0].out_features
    body_model = getattr(model, "model", model)  # skip the LM head: only internal activations are needed

    cur = {"mask": None}
    acc: dict = {}

    def hook(li):
        def f(mod, inp, out):
            m = cur["mask"]
            h = inp[0][0].float()
            y = out[0].float()
            for cls in (0, 1):
                sel = m == cls
                if not sel.any():
                    continue
                k = (li, cls)
                hs, ys = h[sel], y[sel]
                if cur["pass"] == "A":
                    if k not in acc:
                        acc[k] = {"n": 0, "h": torch.zeros(U, device=dev, dtype=torch.float64),
                                  "y": torch.zeros(D, device=dev, dtype=torch.float64)}
                    acc[k]["n"] += int(sel.sum())
                    acc[k]["h"] += hs.sum(0).double()
                    acc[k]["y"] += ys.sum(0).double()
                else:
                    p = (ys @ cur["u"][li]).double()
                    b = acc.setdefault(("B",) + k, {"n": 0, "s": 0.0, "ss": 0.0})
                    b["n"] += int(sel.sum())
                    b["s"] += float(p.sum())
                    b["ss"] += float((p * p).sum())
        return f

    hs = [d.register_forward_hook(hook(i)) for i, d in enumerate(downs)]

    def run(which: int, name: str) -> None:
        cur["pass"] = name
        t0 = time.perf_counter()
        with torch.no_grad():
            for s, hb in zip(seqs, half):
                if hb != which:
                    continue
                ids = torch.tensor([s["prompt"] + s["body"]], device=dev)
                full = np.full(ids.shape[1], -1, dtype=np.int64)
                full[len(s["prompt"]) :] = s["lab"]
                cur["mask"] = torch.from_numpy(full).to(dev)
                body_model(input_ids=ids, use_cache=False)
        print(f"  pass {name}: {time.perf_counter() - t0:.0f}s", flush=True)

    run(0, "A")
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
        sep.append((st[1][0] - st[0][0]) / np.sqrt((st[1][1] + st[0][1]) / 2))
    order = [int(x) for x in np.argsort(sep)[::-1]]
    # two generic layer candidates (no knowledge of p50c): most separated overall, most separated in the later half;
    # the dev set decides between them in `pick-alpha`
    cands = {"top": order[: a.top_layers], "late": [x for x in order if x >= L // 2][: a.top_layers]}
    # architecture prior (not p50c): in hybrid models, the most separated layers among the full-attention ones
    tc = getattr(model.config, "text_config", None) or model.config
    ltypes = getattr(tc, "layer_types", None)
    if ltypes and len(ltypes) == L:
        fa_all = [x for x in order if "full" in str(ltypes[x])]
        if fa_all:
            cands["fa"] = fa_all[: a.top_layers]
            if a.fa_layers and a.fa_layers != a.top_layers:  # e.g. fa3: the three most separated full-attention layers
                cands[f"fa{a.fa_layers}"] = fa_all[: a.fa_layers]
    chosen = cands["top"]
    print("held-out separation d' per layer:", " ".join(f"L{li}:{s:.2f}" for li, s in enumerate(sep)))
    print("layer candidates:", cands)

    neurons, u_out = {}, {}
    for li in sorted(set().union(*cands.values())):
        dh = acc[(li, 1)]["h"] / acc[(li, 1)]["n"] - acc[(li, 0)]["h"] / acc[(li, 0)]["n"]
        W = downs[li].weight.double()  # [D, U]
        c = dh * (u[li].double() @ W)
        top = c.topk(a.neurons).indices.sort().values
        neurons[str(li)] = top.tolist()
        u_out[str(li)] = u[li].cpu().numpy()
    np.savez(work / "wsi_directions.npz", **{f"u{k}": v for k, v in u_out.items()},
             **{f"all_u{li}": u[li].cpu().numpy() for li in range(L)})
    diag = {"separation": [float(s) for s in sep], "layer_rank": [int(x) for x in order], "chosen_layers": chosen, "candidates": cands,
            "neurons": neurons, "loop_tokens": loop_tok, "normal_tokens": norm_tok, "sequences": len(seqs),
            "looping_sequences": n_loop_seq}
    (work / "diagnosis.json").write_text(json.dumps(diag, indent=2))
    print(f"diagnosis saved: candidates {cands}, {a.neurons} neurons per layer")


# ------------------------------------------------------------------ build the fix
def cmd_build(a) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    work = Path(a.work)
    cfg = cfg_load(work)
    base = Path(cfg["base_dir"])
    diag = json.loads((work / "diagnosis.json").read_text())
    z = np.load(work / "wsi_directions.npz")
    if a.cand not in diag["candidates"]:
        raise SystemExit(f"candidate {a.cand} not in diagnosis.json: {sorted(diag['candidates'])}")
    layers_sel = diag["candidates"][a.cand]
    out = Path(a.out_dir) if a.out_dir else work / f"wsi-fix-{a.cand}-a{a.alpha:g}"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns(".cache", "*.lock"))
    for f in sorted(base.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            keys = list(fh.keys())
            targets = {}
            for li in layers_sel:
                try:
                    targets[down_key(keys, li)] = li
                except SystemExit:
                    continue
            if not targets:
                continue
            tensors = {k: fh.get_tensor(k) for k in keys}
            meta = fh.metadata() or {"format": "pt"}
        for k, li in targets.items():
            W = tensors[k].double()
            u = torch.from_numpy(z[f"u{li}"]).double()
            m = torch.zeros(W.shape[1], dtype=torch.float64)
            m[diag["neurons"][str(li)]] = 1.0
            W = W - a.alpha * torch.outer(u, m * (u @ W))
            tensors[k] = W.to(tensors[k].dtype)
        save_file(tensors, str(out / f.name), metadata=meta)
    print(f"built {out} (alpha={a.alpha}, layers {layers_sel})")


# ------------------------------------------------------------------ MoI plan
def cmd_plan_moi(a) -> None:
    work = Path(a.work)
    cfg = cfg_load(work)
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    cont = []
    pp = float(lr.sampling_for(cfg).get("presence_penalty", 0.0))
    for o in lr.read_jsonl(work / "outputs_test_base.jsonl"):
        st = lr.analyse(o["token_ids"], eos)
        if st["fire"] is None:
            continue
        prefix = st["body"][: st["fire"]]
        r = reqs[o["id"]]
        cont.append({"id": o["id"], "prompt_ids": r["prompt_ids"] + prefix, "prefix_len": len(prefix),
                     "max_tokens": r["max_tokens"] - len(prefix), "seed": r["seed"] + 7,
                     # resuming mid-answer loses the presence penalty's memory of the prefix: re-add it as a bias
                     "logit_bias": {str(t): -pp for t in sorted(set(prefix))} if pp else {}})
    lr.write_jsonl(work / "requests_test_moi.jsonl", cont)
    print(f"router fired on {len(cont)}/{len(reqs)} base test responses")


# ------------------------------------------------------------------ tune + report
def cmd_pick_alpha(a) -> None:
    work = Path(a.work)
    cfg = cfg_load(work)
    res = {}
    for f in sorted(work.glob("outputs_dev_*_a*.jsonl")):
        cand, al = f.stem[len("outputs_dev_"):].rsplit("_a", 1)
        s = score(work, "requests_dev.jsonl", f.name)
        res[(cand, float(al))] = {k: v for k, v in s.items() if k != "rows"}
        print(f"  {cand:4s} alpha={float(al):g}: acc={s['accuracy']:.1%} loops={s['loop_killed']:.1%} "
              f"trunc={s['truncated']:.1%}")
    base = score(work, "requests_dev.jsonl", "outputs_dev_base.jsonl")
    print(f"  base     : acc={base['accuracy']:.1%} loops={base['loop_killed']:.1%}")
    best = max(res, key=lambda k: (res[k]["accuracy"], -res[k]["loop_killed"]))
    cfg["cand"], cfg["alpha"] = best
    cfg["dev"] = {"base": {k: v for k, v in base.items() if k != "rows"},
                  "grid": {f"{c}@{al:g}": v for (c, al), v in res.items()}}
    cfg_save(work, cfg)
    diag_p = work / "diagnosis.json"
    diag = json.loads(diag_p.read_text())
    diag["chosen_layers"] = diag["candidates"][best[0]]
    diag_p.write_text(json.dumps(diag, indent=2))
    print(f"chosen: layers {diag['chosen_layers']} ({best[0]}), alpha = {best[1]:g}")
    (work / "chosen_fix.txt").write_text(str(work / f"wsi-fix-{best[0]}-a{best[1]:g}"))


def cmd_report(a) -> None:
    import torch
    from transformers import AutoTokenizer

    work = Path(a.work)
    cfg = cfg_load(work)
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"], trust_remote_code=True)
    eos, reqs = set(cfg["eos"]), {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    arms = {"base": "outputs_test_base.jsonl", "wsi_fixed": "outputs_test_wsi.jsonl", "p50c": "outputs_test_p50c.jsonl"}
    res: dict = {"alpha": cfg.get("alpha"), "layer_candidate": cfg.get("cand"), "dev": cfg.get("dev"), "arms": {}}
    rows = {}
    for name, f in arms.items():
        if (work / f).exists():
            s = score(work, "requests_test.jsonl", f, tok)
            rows[name] = {r["id"]: r for r in s["rows"]}
            res["arms"][name] = {k: v for k, v in s.items() if k != "rows"}
    # MoI = base until the router fires, then the WSI fix continues the same sequence
    plan = {r["id"]: r for r in lr.read_jsonl(work / "requests_test_moi.jsonl")}
    cont = {o["id"]: o for o in lr.read_jsonl(work / "outputs_test_moi.jsonl")} if plan else {}
    moi, edit_tok, all_tok = {}, 0, 0
    for i in reqs:
        if i in cont:
            pre = plan[i]["prompt_ids"][len(reqs[i]["prompt_ids"]) :]
            g = lr.grade(pre + cont[i]["token_ids"], cont[i]["finish_reason"], reqs[i]["gold"], tok, eos,
                         reqs[i]["max_tokens"])
            edit_tok += max(g["tokens"] - len(pre), 0)
        else:
            g = rows["base"][i]
        all_tok += g["tokens"]
        moi[i] = g
    n = len(moi)
    res["arms"]["wsi_moi"] = {"n": n, "accuracy": sum(g["correct"] for g in moi.values()) / n,
                              "loop_killed": sum(g["looped_killed"] for g in moi.values()) / n,
                              "truncated": sum(g["truncated"] for g in moi.values()) / n,
                              "edit_token_share": edit_tok / max(all_tok, 1), "router_fired": len(plan) / n}
    rows["wsi_moi"] = moi
    ids = sorted(reqs)
    acc = {k: [bool(rows[k][i]["correct"]) for i in ids] for k in rows}
    res["paired"] = {f"{x}_vs_{y}": lr.paired_ci(acc[x], acc[y])
                     for x, y in (("wsi_fixed", "base"), ("wsi_moi", "base"), ("wsi_moi", "wsi_fixed"),
                                  ("wsi_fixed", "p50c")) if x in acc and y in acc}

    # ---- the exam: compare with p50c only now
    diag = json.loads((work / "diagnosis.json").read_text())
    exam: dict = {"wsi_chosen_layers": diag["chosen_layers"], "p50c_layers": [15, 19],
                  "rank_of_15_and_19_in_wsi_separation": {str(l): diag["layer_rank"].index(l) + 1 for l in (15, 19)}}
    if a.p50c_dir:
        from safetensors import safe_open

        z = np.load(work / "wsi_directions.npz")
        base = Path(cfg["base_dir"])
        for li in (15, 19):
            for f in sorted(Path(a.p50c_dir).glob("*.safetensors")):
                with safe_open(str(f), "pt") as fe, safe_open(str(base / f.name), "pt") as fb:
                    try:
                        k = down_key(list(fe.keys()), li)
                    except SystemExit:
                        continue
                    dW = (fe.get_tensor(k).double() - fb.get_tensor(k).double()).numpy()
            cols = np.where(np.abs(dW).sum(0) > 0)[0]
            Up, _, _ = np.linalg.svd(dW[:, cols], full_matrices=False)
            up = Up[:, 0]
            uw = z[f"all_u{li}"].astype(np.float64)
            rec = {"abs_cos_wsi_direction_vs_p50c": round(float(abs(uw @ up) / np.linalg.norm(uw)), 4),
                   "random_baseline": round(1 / np.sqrt(len(up)), 4)}
            if str(li) in diag["neurons"]:
                rec["neuron_overlap"] = f"{len(set(diag['neurons'][str(li)]) & set(cols.tolist()))}/{len(cols)}"
            exam[f"layer_{li}"] = rec
    res["exam_vs_p50c"] = exam
    res["timing"] = timing_summary(work)
    (work / "results_discover.json").write_text(json.dumps(res, indent=2))

    print(f"\n{'arm':10s} {'acc':>7s} {'loops':>7s} {'trunc':>7s}")
    for k, v in res["arms"].items():
        print(f"{k:10s} {v['accuracy']:7.1%} {v['loop_killed']:7.1%} {v['truncated']:7.1%}"
              + (f"   edit on {v['edit_token_share']:.0%} of tokens" if "edit_token_share" in v else ""))
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
    print("exam vs p50c:", json.dumps(exam, indent=1))
    print("Saved", work / "results_discover.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--base-dir")
    p.add_argument("--data-dir")
    p.add_argument("--n-elicit", type=int, default=300)
    p.add_argument("--elicit-max-new", type=int, default=8192)
    p.add_argument("--n-dev", type=int, default=64)
    p.add_argument("--dev-max-new", type=int, default=8192)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--max-new", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--requests", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--gpu-mem", type=float, default=0.88)
    g.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True,
                   help="stop a response where the loop kill rule fires (same graded result, far fewer tokens)")
    d = sub.add_parser("diagnose")
    d.add_argument("--work", required=True)
    d.add_argument("--top-layers", type=int, default=2)
    d.add_argument("--fa-layers", type=int, default=0, help="also add a candidate with this many full-attention layers")
    d.add_argument("--neurons", type=int, default=64)
    d.add_argument("--after-onset", type=int, default=768, help="tokens kept after the first looping token")
    d.add_argument("--max-seq", type=int, default=4096)
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--alpha", type=float, required=True)
    b.add_argument("--cand", default="top", help="a key of diagnosis.json candidates: top, late, fa, fa3, ...")
    b.add_argument("--out-dir")
    m = sub.add_parser("plan-moi")
    m.add_argument("--work", required=True)
    k = sub.add_parser("pick-alpha")
    k.add_argument("--work", required=True)
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--p50c-dir")
    a = ap.parse_args()
    fns = {"prepare": cmd_prepare, "gen": cmd_gen, "diagnose": cmd_diagnose, "build": cmd_build,
           "plan-moi": cmd_plan_moi, "pick-alpha": cmd_pick_alpha, "report": cmd_report}
    key = {"gen": lambda: f"gen:{Path(a.out).stem}:{Path(a.model).name}", "build": lambda: f"build:{a.cand}@{a.alpha:g}",
           "diagnose": lambda: "diagnose", "pick-alpha": lambda: "pick-alpha", "prepare": lambda: "prepare"}.get(a.cmd)
    run_timed(a, fns, key() if key else None)


if __name__ == "__main__":
    main()
