"""MoI loop router on Linux GPUs with vLLM (Colab / Kaggle / RunPod): base vs fixed p50c vs MoI, 32,768-token budget.

Same question as `experiment_moi_loop_router.py` (HF, Windows), at the full think35 budget of the p50c evaluation:
thinking on, T=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max 32,768 new tokens.

Arms (same prompts, same per-request sampling seed):
  * base  : unedited Qwen3.5-0.8B.
  * fixed : the official p50c checkpoint (edit always on).
  * moi   : base until the router fires, then the p50c checkpoint continues the SAME sequence.
            The router is causal, so the MoI trajectory equals the base trajectory up to the firing token; MoI is
            therefore built from the base outputs: cut each base response at the firing point and let the edited
            model continue it. Sequences where the router never fires are identical to base.

Router (Barry's loop definition): dup[t] = 1 if the 16-token window ending at t occurred earlier in the response.
Fires at the first response length n with >= 50% dup over the last 64 tokens.
Loop failure (all arms, applied after generation): the first point where >= 90% of the last 768 tokens are dup.
The sequence is cut there and counted wrong. This is the same stopping rule the HF script used.

Two differences from the HF hook version, both small and stated in the report:
  1. On the switch, vLLM re-encodes the prefix with the edited weights. The HF hook kept the base-weight KV cache for
     the prefix.
  2. vLLM's presence penalty only sees tokens generated in the current request. For the MoI continuation, every
     distinct token of the prefix gets logit_bias -1.5 to keep the penalty. A prefix token that reappears in the
     continuation is therefore penalised twice (-3.0).

Pipeline (each GPU step runs in its own process, so vLLM frees the GPU cleanly):
    python moi_loop_router_vllm.py prepare --work W --edit-dir REPO/benchmarks/exp8_math [--base-dir D] [--n 200]
    python moi_loop_router_vllm.py gen --work W --arm base
    python moi_loop_router_vllm.py gen --work W --arm fixed
    python moi_loop_router_vllm.py plan-moi --work W
    python moi_loop_router_vllm.py gen --work W --arm moi
    python moi_loop_router_vllm.py report --work W [--ppl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np

MAX_NEW = 32_768
K_DUP = 16
ROUTER_WINDOW, ROUTER_FRAC = 64, 0.5
KILL_WINDOW, KILL_FRAC = 768, 0.9
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)
# Per-model sampling profiles. A run's config.json may carry "sampling"; runs without it keep SAMPLING (Qwen3.5).
SAMPLING_PROFILES = {
    "qwen": SAMPLING,
    "lfm3-math": dict(temperature=1.0, top_p=0.95, top_k=20),  # LiquidAI/LFM3-1.7B model card, Math & STEM
    "greedy": dict(temperature=0.0),
}
# The MLP output matrix the edits change: Qwen-style mlp.down_proj or LFM-style feed_forward.w2.
DOWN_PAT = r"layers\.(\d+)\.(?:mlp\.down_proj|feed_forward\.w2)\.weight$"


def sampling_for(cfg: dict) -> dict:
    return dict(cfg.get("sampling") or SAMPLING)


def down_module(layer):
    """The down projection of one decoder layer (Qwen: mlp.down_proj, LFM2/3: feed_forward.w2)."""
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and hasattr(mlp, "down_proj"):
        return mlp.down_proj
    ff = getattr(layer, "feed_forward", None)
    if ff is not None and hasattr(ff, "w2"):
        return ff.w2
    raise AttributeError(f"no down projection on {type(layer).__name__}")
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
BASE_REPO = "Qwen/Qwen3.5-0.8B"


# ------------------------------------------------------------------ io helpers
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ------------------------------------------------------------------ loop statistics
def dup_flags(ids: list[int], k: int = K_DUP) -> list[int]:
    seen, out = set(), [0] * len(ids)
    for t in range(k - 1, len(ids)):
        w = tuple(ids[t - k + 1 : t + 1])
        out[t] = int(w in seen)
        seen.add(w)
    return out


def first_window_hit(d: list[int], window: int, frac: float) -> int | None:
    """Smallest n (response length) such that the last `window` of d[:n] has >= frac dup; None if never."""
    need = frac * window
    s = sum(d[:window])
    if len(d) < window:
        return None
    if s >= need:
        return window
    for n in range(window + 1, len(d) + 1):
        s += d[n - 1] - d[n - 1 - window]
        if s >= need:
            return n
    return None


class LoopKillState:
    """Per-request early stop: forces EOS as soon as the kill rule (>= KILL_FRAC repeated K_DUP-grams over the last
    KILL_WINDOW response tokens) holds. Grading cuts a response at exactly that point anyway (grade -> analyse ->
    first_window_hit), so stopping there gives the same graded result without generating the doomed loop to the cap.
    Incremental: O(1) per new token."""

    def __init__(self, eos_id: int, window: int = KILL_WINDOW, frac: float = KILL_FRAC, k: int = K_DUP):
        self.eos, self.window, self.need, self.k = eos_id, window, frac * window, k
        self.seen: set = set()
        self.flags: list[int] = []
        self.s = 0

    def killed(self, output_ids) -> bool:
        while len(self.flags) < len(output_ids):
            t = len(self.flags)
            f = 0
            if t >= self.k - 1:
                w = tuple(output_ids[t - self.k + 1 : t + 1])
                f = int(w in self.seen)
                self.seen.add(w)
            self.flags.append(f)
            self.s += f
            if t >= self.window:
                self.s -= self.flags[t - self.window]
        return len(self.flags) >= self.window and self.s >= self.need

    def __call__(self, output_ids, logits):
        if self.killed(output_ids):
            logits = logits.clone()
            logits.fill_(float("-inf"))
            logits[self.eos] = 0.0
        return logits


try:  # vLLM V1 per-request logits processors (absent locally / in old vLLM: early stop is then skipped)
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor as _AdapterLP
except Exception:  # noqa: BLE001
    _AdapterLP = None

if _AdapterLP is not None:
    class LoopKillLogitsProcessor(_AdapterLP):
        """Enabled per request with SamplingParams(extra_args={"loop_kill_eos": <eos id>})."""

        def is_argmax_invariant(self) -> bool:
            return False

        def new_req_logits_processor(self, params):
            ea = getattr(params, "extra_args", None) or {}
            return LoopKillState(int(ea["loop_kill_eos"])) if "loop_kill_eos" in ea else None
else:
    LoopKillLogitsProcessor = None


def strip_eos(ids: list[int], eos: set[int]) -> list[int]:
    for i, t in enumerate(ids):
        if t in eos:
            return ids[:i]
    return ids


# ------------------------------------------------------------------ grading
def extract_answer(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    cand = boxed[-1] if boxed else text
    nums = re.findall(r"-?\d+(?:\.\d+)?", cand.replace(",", ""))
    return nums[-1] if nums else None


def correct(pred: str | None, gold: str) -> bool:
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


# ------------------------------------------------------------------ prepare
def int16_sha(t) -> str:
    import torch

    return hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def build_p50c(base_dir: Path, edit_dir: Path, out_dir: Path) -> dict:
    """Copy the base checkpoint and write the official p50c tensors into it, verifying both SHA-256s."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    man = json.loads((edit_dir / "p50c_manifest.json").read_text())
    z = np.load(edit_dir / "p50c_edit.npz", allow_pickle=False)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir, ignore=shutil.ignore_patterns(".cache", "*.lock"))
    idx_file = next(iter(sorted(base_dir.glob("*.safetensors.index.json"))), None)
    weight_map = json.loads(idx_file.read_text())["weight_map"] if idx_file else None
    by_file: dict[str, list[int]] = {}
    for i, t in enumerate(man["tensors"]):
        fn = weight_map[t["name"]] if weight_map else next(iter(sorted(base_dir.glob("*.safetensors")))).name
        by_file.setdefault(fn, []).append(i)
    report = {}
    for fn, ids in by_file.items():
        with safe_open(str(base_dir / fn), "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
            meta = f.metadata() or {"format": "pt"}
        for i in ids:
            t = man["tensors"][i]
            base = tensors[t["name"]]
            sha = int16_sha(base)
            if sha != t["base_sha256"]:
                raise SystemExit(f"{t['name']}: base weights differ from the p50c release ({sha[:12]}...)")
            bits = base.view(torch.int16).flatten().clone()
            vals = z[f"t{i}_val"].astype(np.uint16).view(np.int16)
            bits[torch.from_numpy(z[f"t{i}_idx"].astype(np.int64))] = torch.from_numpy(vals)
            edited = bits.view(torch.bfloat16).reshape(base.shape)
            esha = int16_sha(edited)
            ok = esha == t.get("edited_sha256", esha)
            report[t["name"]] = {"base_sha_ok": True, "edited_sha_ok": ok,
                                 "changed": int((edited.view(torch.int16) != base.view(torch.int16)).sum())}
            if not ok:
                raise SystemExit(f"{t['name']}: rebuilt edit does not match the release SHA ({esha[:12]}...)")
            tensors[t["name"]] = edited
        save_file(tensors, str(out_dir / fn), metadata=meta)
    return report


def fetch_datasets(work: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    gsm = Path(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset",
                               local_dir=work / "data"))
    wt = Path(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                              repo_type="dataset", local_dir=work / "data"))
    return gsm, wt


def eos_ids(tok) -> list[int]:
    ids = {tok.eos_token_id, tok.pad_token_id}
    for s in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    ids.discard(None)
    return sorted(ids)


def cmd_prepare(a) -> None:
    import pandas as pd
    from transformers import AutoTokenizer

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.base_dir:
        base_dir = Path(a.base_dir)
    else:
        from huggingface_hub import snapshot_download

        base_dir = Path(snapshot_download(BASE_REPO, local_dir=work / "qwen-base"))
    p50c_dir = work / "qwen-p50c"
    t0 = time.time()
    rep = build_p50c(base_dir, Path(a.edit_dir), p50c_dir)
    print(f"p50c checkpoint built in {time.time() - t0:.0f}s:", json.dumps(rep))
    if a.gsm and a.wikitext:
        gsm_p, wt_p = Path(a.gsm), Path(a.wikitext)
    else:
        gsm_p, wt_p = fetch_datasets(work)
    tok = AutoTokenizer.from_pretrained(base_dir)
    gsm = pd.read_parquet(gsm_p).iloc[: a.n]
    reqs = []
    for i, (q, ans) in enumerate(zip(gsm["question"], gsm["answer"])):
        ids = tok.apply_chat_template([{"role": "user", "content": q + "\n" + INSTRUCTION}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=True)
        if hasattr(ids, "keys"):  # BatchEncoding in newer transformers
            ids = ids["input_ids"]
        reqs.append({"id": i, "prompt_ids": list(map(int, ids)), "gold": ans.split("####")[-1].strip().replace(",", ""),
                     "seed": a.seed * 100_003 + i, "max_tokens": a.max_new})
    write_jsonl(work / "requests_base.jsonl", reqs)
    write_jsonl(work / "requests_fixed.jsonl", reqs)
    cfg = {"base_dir": str(base_dir), "p50c_dir": str(p50c_dir), "wikitext": str(wt_p), "n": a.n,
           "max_new": a.max_new, "eos": eos_ids(tok), "sampling": SAMPLING, "seed": a.seed,
           "router": [ROUTER_WINDOW, ROUTER_FRAC], "kill": [KILL_WINDOW, KILL_FRAC], "k_dup": K_DUP}
    (work / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{len(reqs)} GSM8K requests, longest prompt {max(len(r['prompt_ids']) for r in reqs)} tokens -> {work}")


# ------------------------------------------------------------------ generation (vLLM)
def cmd_gen(a) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    reqs = read_jsonl(work / f"requests_{a.arm}.jsonl")
    out_p = work / f"outputs_{a.arm}.jsonl"
    if not reqs:
        write_jsonl(out_p, [])
        print("no requests")
        return
    model = cfg["base_dir"] if a.arm == "base" else cfg["p50c_dir"]
    max_len = max(len(r["prompt_ids"]) + r["max_tokens"] for r in reqs) + 16
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, seed=cfg["seed"])
    sps = [SamplingParams(**SAMPLING, max_tokens=r["max_tokens"], seed=r["seed"], stop_token_ids=cfg["eos"],
                          logit_bias={int(k): float(v) for k, v in r.get("logit_bias", {}).items()} or None)
           for r in reqs]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in reqs], sps, use_tqdm=True)
    dt = time.time() - t0
    rows = [{"id": r["id"], "token_ids": list(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason}
            for r, o in zip(reqs, outs)]
    write_jsonl(out_p, rows)
    ntok = sum(len(x["token_ids"]) for x in rows)
    print(f"[{a.arm}] {len(rows)} sequences, {ntok:,} tokens in {dt / 60:.1f} min ({ntok / dt:,.0f} tok/s) -> {out_p}")


# ------------------------------------------------------------------ MoI plan
def analyse(resp: list[int], eos: set[int]) -> dict:
    """Loop statistics of one response: effective length after the kill rule, and the router firing point."""
    body = strip_eos(resp, eos)
    d = dup_flags(body)
    kill = first_window_hit(d, KILL_WINDOW, KILL_FRAC)
    end = kill if kill is not None else len(body)
    fire = first_window_hit(d[:end], ROUTER_WINDOW, ROUTER_FRAC)
    return {"body": body, "kill": kill, "fire": fire}


def cmd_plan_moi(a) -> None:
    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in read_jsonl(work / "requests_base.jsonl")}
    cont = []
    for o in read_jsonl(work / "outputs_base.jsonl"):
        st = analyse(o["token_ids"], eos)
        if st["fire"] is None:
            continue
        prefix = st["body"][: st["fire"]]
        r = reqs[o["id"]]
        cont.append({"id": o["id"], "prompt_ids": r["prompt_ids"] + prefix, "prefix_len": len(prefix),
                     "max_tokens": cfg["max_new"] - len(prefix), "seed": r["seed"] + 7,
                     "logit_bias": {str(t): -SAMPLING["presence_penalty"] for t in sorted(set(prefix))}})
    write_jsonl(work / "requests_moi.jsonl", cont)
    print(f"router fired on {len(cont)}/{len(reqs)} base sequences; continuing those with the p50c weights")


# ------------------------------------------------------------------ report
def grade(resp: list[int], finish: str, gold: str, tok, eos: set[int], max_new: int) -> dict:
    st = analyse(resp, eos)
    killed = st["kill"] is not None
    body = st["body"][: st["kill"]] if killed else st["body"]
    truncated = (not killed) and finish == "length" and len(st["body"]) >= max_new
    pred = extract_answer(tok.decode(body, skip_special_tokens=True))
    return {"pred": pred, "tokens": len(body), "looped_killed": killed, "truncated": truncated,
            "correct": bool(correct(pred, gold) and not killed and not truncated), "fire": st["fire"]}


def paired_ci(a: list[bool], b: list[bool], n_boot: int = 10_000, seed: int = 0) -> dict:
    x = np.array(a, float) - np.array(b, float)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return {"diff": round(float(x.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                                         round(float(np.percentile(boots, 97.5)), 4)],
            "a_only": int(((np.array(a)) & ~np.array(b)).sum()), "b_only": int((~np.array(a) & np.array(b)).sum())}


def wikitext_ppl(cfg: dict, n_tokens: int = 20_000) -> dict:
    """Teacher-forced WikiText-2 perplexity; MoI turns the edit on per position with the same causal router."""
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    base_dir, p50c_dir = Path(cfg["base_dir"]), Path(cfg["p50c_dir"])
    tok = AutoTokenizer.from_pretrained(base_dir)
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16).to(dev).eval()
    layers = next(m for n, m in model.named_modules()
                  if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 20)
    state = {"gate": None}
    for f in sorted(p50c_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fe, safe_open(str(base_dir / f.name), "pt") as fb:
            for k in fe.keys():
                m = re.search(r"layers\.(\d+)\.mlp\.down_proj\.weight$", k)
                if not m:
                    continue
                dW = fe.get_tensor(k).float() - fb.get_tensor(k).float()
                cols = torch.nonzero(dW.abs().sum(0) > 0).flatten()
                if not len(cols):
                    continue
                cols_d, dW_d = cols.to(dev), dW[:, cols].to(dev)

                def hook(mod, inp, out, cols_d=cols_d, dW_d=dW_d):
                    g = state["gate"]
                    if g is None:
                        return out
                    return (out.float() + (inp[0][..., cols_d].float() @ dW_d.T) * g).to(out.dtype)

                layers[int(m.group(1))].mlp.down_proj.register_forward_hook(hook)
    wt = "\n".join(pd.read_parquet(cfg["wikitext"])["text"].tolist())
    ids_all = tok(wt, add_special_tokens=False).input_ids[:n_tokens]
    res = {}
    for arm in ("base", "fixed", "moi"):
        nll, nt, gated = 0.0, 0, 0
        for i in range(0, len(ids_all) - 16, 512):
            ch = ids_all[i : i + 512]
            ids = torch.tensor([ch], device=dev)
            if arm == "base":
                state["gate"] = None
            elif arm == "fixed":
                state["gate"] = torch.ones(1, 1, 1, device=dev)
            else:
                g = torch.zeros(1, len(ch), 1, device=dev)
                n = first_window_hit(dup_flags(ch), ROUTER_WINDOW, ROUTER_FRAC)
                if n is not None:
                    g[0, n - 1 :] = 1.0
                gated += int(g.sum())
                state["gate"] = g
            lp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
            nll -= float(lp.gather(1, ids[0, 1:, None]).sum())
            nt += len(ch) - 1
        res[arm] = {"ppl": round(math.exp(nll / nt), 4), "tokens": nt, "gated_positions": gated}
        print(f"  WikiText ppl [{arm}] = {res[arm]['ppl']}")
    state["gate"] = None
    return res


def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    eos, max_new = set(cfg["eos"]), cfg["max_new"]
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"])
    reqs = {r["id"]: r for r in read_jsonl(work / "requests_base.jsonl")}
    outs = {arm: {o["id"]: o for o in read_jsonl(work / f"outputs_{arm}.jsonl")} for arm in ("base", "fixed")}
    cont = {o["id"]: o for o in read_jsonl(work / "outputs_moi.jsonl")} if (work / "outputs_moi.jsonl").exists() else {}
    plan = {r["id"]: r for r in read_jsonl(work / "requests_moi.jsonl")} if cont else {}

    rows: dict[str, list[dict]] = {"base": [], "fixed": [], "moi": []}
    for i in sorted(reqs):
        gold = reqs[i]["gold"]
        for arm in ("base", "fixed"):
            o = outs[arm][i]
            rows[arm].append({"id": i, "gold": gold, **grade(o["token_ids"], o["finish_reason"], gold, tok, eos, max_new),
                              "edit_share": 0.0 if arm == "base" else 1.0})
        if i in cont:
            p = plan[i]
            prefix = p["prompt_ids"][len(reqs[i]["prompt_ids"]) :]
            g = grade(prefix + cont[i]["token_ids"], cont[i]["finish_reason"], gold, tok, eos, max_new)
            g["fire"] = p["prefix_len"]
            g["edit_share"] = round((g["tokens"] - p["prefix_len"]) / max(g["tokens"], 1), 4)
        else:  # router never fired: MoI is the base sequence
            g = {**rows["base"][-1], "fire": None, "edit_share": 0.0}
            g = {k: v for k, v in g.items() if k not in ("id", "gold")}
        rows["moi"].append({"id": i, "gold": gold, **g})

    def summ(rs: list[dict]) -> dict:
        n = len(rs)
        return {"n": n, "accuracy": round(sum(r["correct"] for r in rs) / n, 4),
                "loop_killed": round(sum(r["looped_killed"] for r in rs) / n, 4),
                "truncated": round(sum(r["truncated"] for r in rs) / n, 4),
                "mean_tokens": round(sum(r["tokens"] for r in rs) / n, 1),
                "edit_token_share": round(sum(r["edit_share"] * r["tokens"] for r in rs)
                                          / max(sum(r["tokens"] for r in rs), 1), 4)}

    result = {"config": cfg, "summary": {arm: summ(rs) for arm, rs in rows.items()}}
    if len(reqs) > 48:
        result["summary_first48"] = {arm: summ(rs[:48]) for arm, rs in rows.items()}
    acc = {arm: [r["correct"] for r in rs] for arm, rs in rows.items()}
    result["paired"] = {"moi_vs_base": paired_ci(acc["moi"], acc["base"]),
                        "fixed_vs_base": paired_ci(acc["fixed"], acc["base"]),
                        "moi_vs_fixed": paired_ci(acc["moi"], acc["fixed"])}
    fired = [r for r in rows["moi"] if r["fire"] is not None]
    result["router"] = {"fired_share": round(len(fired) / len(rows["moi"]), 4),
                        "median_fire_token": float(np.median([r["fire"] for r in fired])) if fired else None,
                        "never_fired_correct": sum(r["correct"] for r in rows["moi"] if r["fire"] is None),
                        "never_fired": sum(r["fire"] is None for r in rows["moi"])}
    result["caveats"] = ["MoI prefix is re-encoded with the edited weights at the switch (HF hook kept base KV).",
                         "MoI continuation: prefix tokens get logit_bias -1.5; a prefix token reused later is "
                         "penalised -3.0 instead of -1.5."]
    if a.ppl:
        result["wikitext"] = wikitext_ppl(cfg)
    result["rows"] = rows

    print(f"\n{'arm':6s} {'acc':>7s} {'loops':>7s} {'trunc':>7s} {'tokens':>8s} {'edit':>6s}"
          + (f" {'ppl':>8s}" if a.ppl else ""))
    for arm, s in result["summary"].items():
        print(f"{arm:6s} {s['accuracy']:7.1%} {s['loop_killed']:7.1%} {s['truncated']:7.1%} {s['mean_tokens']:8.0f} "
              f"{s['edit_token_share']:6.1%}" + (f" {result['wikitext'][arm]['ppl']:8.3f}" if a.ppl else ""))
    for k, v in result["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]"
              f"  (only first {v['a_only']}, only second {v['b_only']})")
    print("  router:", result["router"])
    dst = work / "results_32k.json"
    dst.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--edit-dir", required=True)
    p.add_argument("--base-dir", default=None, help="local Qwen3.5-0.8B; downloaded from the Hub if omitted")
    p.add_argument("--gsm", default=None)
    p.add_argument("--wikitext", default=None)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--max-new", type=int, default=MAX_NEW)
    p.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--arm", choices=["base", "fixed", "moi"], required=True)
    g.add_argument("--gpu-mem", type=float, default=0.88)
    m = sub.add_parser("plan-moi")
    m.add_argument("--work", required=True)
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--ppl", action="store_true")
    a = ap.parse_args()
    random.seed(0)
    {"prepare": cmd_prepare, "gen": cmd_gen, "plan-moi": cmd_plan_moi, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
