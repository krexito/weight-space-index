"""Task-level MoI on LiquidAI/LFM3-1.7B: pick the MLP neurons for "math" ONCE, load only those, test GSM8K.

Per-token routing (experiment_moi_lfm3_gsm8k.py) lost ~20 GSM8K points at 50% of neurons with any usable router.
Here the index is chosen per TASK instead of per token:
  1. Calibrate: run the full model over task text, record for each SwiGLU neuron i of each layer its mean output
     contribution  c_i = E_t[ |h_i(t)| ] * ||w2[:, i]||   (h = silu(w1 x) * (w3 x); w2 column = what the neuron writes).
  2. Index: keep the top neurons, either the same fraction in every layer ("uniform") or ranked across all layers
     with each layer's scores divided by that layer's total contribution ("global", lets fragile layers keep more).
  3. Load: the MLP matrices are physically sliced to the kept neurons (w1/w3 rows, w2 columns). This is a real,
     smaller model: parameter count, GPU memory and decode speed are measured, not estimated.

Indexes compared at each budget:
  * math-uniform / math-global : calibrated on GSM8K TRAIN (question + gold solution, chat format).
  * wiki-uniform               : calibrated on WikiText-2 validation (a general-purpose index, same size).
  * random                     : random neurons (same size), the floor.
Compensated variants ("+recon"): after slicing, the kept w2 columns are refit by ridge least squares so the MLP
output matches the full one on math calibration data, W_S = W G[:,S] (G_SS + lam I)^-1, with the co-activation
G = E[h h^T] estimated from only the first --gram-tokens (default 20k) math tokens (benchmarks/gram_compress).
"math-energy" ranks neurons by sqrt(E h^2) * ||w2_i|| instead of the mean.
"math-obs" picks the neurons in PAIRS from the same Gram (greedy backward elimination, Optimal Brain Surgeon per
layer: drop the neuron whose loss hurts least once the others are refit; experiment_index_criteria.obs_select).
Use it as "math-obs+recon": on Qwen3.5-0.8B it beat energy+recon on math perplexity at every budget.
Evaluation: GSM8K TEST accuracy (greedy, thinking off; same prompt as the per-token run, full model 83.0%),
teacher-forced perplexity on GSM8K TEST gold solutions (in-domain) and on WikiText-2 test (general).

    python experiment_moi_task_index_lfm3.py --model-dir LiquidAI/LFM3-1.7B --data-dir /content/data --out r.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

HUB_FILES = {
    "gsm8k/test.parquet": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    "gsm8k/train.parquet": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
    "wikitext/test.parquet": ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
    "wikitext/validation.parquet": ("Salesforce/wikitext", "wikitext-2-raw-v1/validation-00000-of-00001.parquet"),
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


def clean_solution(ans: str) -> str:
    """GSM8K gold solution without the <<calc>> annotations, ending like the prompt asks."""
    body, final = ans.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nThe answer is {final.strip()}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--budgets", default="0.7,0.5,0.3")
    ap.add_argument("--indexes", default="math-uniform,math-energy,math-global,math-uniform+recon,math-energy+recon,math-global+recon,random")
    ap.add_argument("--calib-tokens", type=int, default=100_000)
    ap.add_argument("--gram-tokens", type=int, default=20_000,
                    help="math tokens used for the co-activation (Gram) matrix of the +recon indexes")
    ap.add_argument("--ridge", type=float, default=1e-2, help="lambda as a fraction of mean diag(G_SS)")
    ap.add_argument("--ppl-tokens", type=int, default=8_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results_task_index.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    t_start = time.perf_counter()
    data_dir = Path(args.data_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True,
                                                 dtype=torch.bfloat16).to(dev).eval()
    layers = model.model.layers
    L = len(layers)
    # LFM3 names its SwiGLU w1 (gate) / w3 (up) / w2 (down); Llama-style models gate_proj / up_proj / down_proj
    lfm = hasattr(layers[0], "feed_forward")
    mlps = [lyr.feed_forward if lfm else lyr.mlp for lyr in layers]
    NM = ("w1", "w3", "w2") if lfm else ("gate_proj", "up_proj", "down_proj")
    orig = [tuple(getattr(m, n) for n in NM) for m in mlps]  # full Linear modules, restored after each config
    U = int(orig[0][0].out_features)
    print(f"LFM3: {L} layers, {U} MLP neurons per layer")

    gsm_test = pd.read_parquet(need(data_dir, "gsm8k/test.parquet"))
    gsm_train = pd.read_parquet(need(data_dir, "gsm8k/train.parquet"))
    wt_val = "\n".join(pd.read_parquet(need(data_dir, "wikitext/validation.parquet"))["text"].tolist())
    wt_test = "\n".join(pd.read_parquet(need(data_dir, "wikitext/test.parquet"))["text"].tolist())

    BOS = [tok.bos_token_id] if tok.bos_token_id is not None else []

    def chat_ids(q: str, a: str) -> list[int]:
        txt = tok.apply_chat_template([{"role": "user", "content": q + PROMPT_SUFFIX},
                                       {"role": "assistant", "content": a}], tokenize=False, enable_thinking=False)
        return tok(txt, add_special_tokens=False).input_ids

    # ---------------------------------------------------------------- calibration: |h_i|, h_i^2 per neuron, Gram
    acc = torch.zeros(L, U, device=dev, dtype=torch.float64)
    accsq = torch.zeros(L, U, device=dev, dtype=torch.float64)
    gram: list[torch.Tensor] = []
    st = {"gram_on": False}

    def cal_hook(li):
        def hook(mod, inp, out):
            h = inp[0].reshape(-1, U).float()  # input of w2 = h
            acc[li] += h.abs().sum(0).double()
            accsq[li] += (h * h).sum(0).double()
            if st["gram_on"]:
                gram[li].addmm_(h.T, h)
        return hook

    def calibrate(seqs: list[list[int]], gram_tokens: int = 0) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Returns mean score E|h|*||w2_i||, energy score sqrt(E h^2)*||w2_i||, and the tokens in the Gram.
        The Gram (co-activation of the neurons, for compensation) uses only the first `gram_tokens` tokens."""
        acc.zero_()
        accsq.zero_()
        n, n_gram = 0, 0
        if gram_tokens:
            gram.clear()
            gram.extend(torch.zeros(U, U, device=dev, dtype=torch.float32) for _ in range(L))
        hs = [getattr(m, NM[2]).register_forward_hook(cal_hook(i)) for i, m in enumerate(mlps)]
        for s in seqs:
            st["gram_on"] = bool(gram_tokens) and n_gram < gram_tokens
            model(input_ids=torch.tensor([s], device=dev))
            n += len(s)
            n_gram += len(s) if st["gram_on"] else 0
        st["gram_on"] = False
        for h in hs:
            h.remove()
        if gram_tokens:
            for g in gram:
                g /= n_gram
        w2n = torch.stack([getattr(m, NM[2]).weight.float().norm(dim=0) for m in mlps]).double()  # [L, U]
        return (acc / n) * w2n, (accsq / n).sqrt() * w2n, n_gram

    math_seqs, n = [], 0
    for q, a in zip(gsm_train["question"], gsm_train["answer"]):
        s = chat_ids(q, clean_solution(a))
        math_seqs.append(s)
        n += len(s)
        if n >= args.calib_tokens:
            break
    wt_ids = tok(wt_val, add_special_tokens=False).input_ids[: args.calib_tokens]
    wiki_seqs = [BOS + wt_ids[i : i + 512] for i in range(0, len(wt_ids) - 16, 512)]
    t0 = time.perf_counter()
    wiki_mean, _, _ = calibrate(wiki_seqs)
    math_mean, math_energy, n_gram = calibrate(math_seqs, gram_tokens=args.gram_tokens)  # Gram stays = math
    score = {"math": math_mean, "math_energy": math_energy, "wiki": wiki_mean}
    print(f"calibrated on {sum(map(len, math_seqs)):,} math + {sum(map(len, wiki_seqs)):,} wiki tokens "
          f"(Gram from the first {n_gram:,} math tokens) in {time.perf_counter() - t0:.0f}s", flush=True)

    def recon_w2(li: int, ix: torch.Tensor) -> torch.Tensor:
        """Compensation: refit the kept w2 columns so the MLP output matches the full one on the math calibration
        data, W_S = W G[:,S] (G_SS + lam I)^-1  (ridge least squares, lam = ridge * mean diag G_SS)."""
        W = orig[li][2].weight.double()  # [D, U]
        G = gram[li]
        Gss = G[ix][:, ix].double()
        lam = args.ridge * Gss.diagonal().mean().clamp_min(1e-12)
        rhs = W @ G[:, ix].double()
        Ws = torch.linalg.solve(Gss + lam * torch.eye(len(ix), device=dev, dtype=torch.float64), rhs.T).T
        return Ws.to(orig[li][2].weight.dtype)

    obs_cache: dict[int, list[torch.Tensor]] = {}

    def obs_index(budget: float) -> list[torch.Tensor]:
        """Pairwise (OBS) selection on the math Gram; one greedy pass per layer gives every budget."""
        k = round(budget * U)
        if k not in obs_cache:
            from experiment_index_criteria import obs_select

            keeps = sorted({round(float(b) * U) for b in args.budgets.split(",")} | {k})
            t0 = time.perf_counter()
            per_layer = [obs_select(gram[li], orig[li][2].weight, keeps, args.ridge) for li in range(L)]
            for kk in keeps:
                obs_cache[kk] = [per_layer[li][kk].cpu() for li in range(L)]
            print(f"obs selection for budgets {keeps} in {time.perf_counter() - t0:.0f}s", flush=True)
        return obs_cache[k]

    def build_index(kind: str, budget: float) -> list[torch.Tensor]:
        kind = kind.removesuffix("+recon")
        if kind == "math-obs":
            return obs_index(budget)
        k_tot = round(budget * U * L)
        if kind == "random":
            g = torch.Generator().manual_seed(0)
            return [torch.randperm(U, generator=g)[: round(budget * U)].sort().values for _ in range(L)]
        src, mode = kind.split("-")
        if mode == "energy":
            src, mode = "math_energy", "uniform"
        s = score[src]
        if mode == "uniform":
            k = round(budget * U)
            return [s[li].topk(k).indices.sort().values.cpu() for li in range(L)]
        rel = s / s.sum(dim=1, keepdim=True)  # global: share of the layer's total contribution
        flat = rel.flatten().topk(k_tot).indices
        keep = [[] for _ in range(L)]
        for f in flat.tolist():
            keep[f // U].append(f % U)
        out = []
        for li in range(L):
            ix = keep[li] if keep[li] else [int(s[li].argmax())]  # never empty a layer completely
            out.append(torch.tensor(sorted(ix)))
        return out

    def apply_index(index: list[torch.Tensor] | None, recon: bool = False) -> None:
        for li, m in enumerate(mlps):
            w1, w3, w2 = orig[li]
            if index is None:
                for n, mod in zip(NM, (w1, w3, w2)):
                    setattr(m, n, mod)
                continue
            ix = index[li].to(dev)
            n1 = nn.Linear(w1.in_features, len(ix), bias=False, device=dev, dtype=w1.weight.dtype)
            n3 = nn.Linear(w3.in_features, len(ix), bias=False, device=dev, dtype=w3.weight.dtype)
            n2 = nn.Linear(len(ix), w2.out_features, bias=False, device=dev, dtype=w2.weight.dtype)
            n1.weight.copy_(w1.weight[ix])
            n3.weight.copy_(w3.weight[ix])
            n2.weight.copy_(recon_w2(li, ix) if recon else w2.weight[:, ix])
            for n, mod in zip(NM, (n1, n3, n2)):
                setattr(m, n, mod)

    # ---------------------------------------------------------------- evaluation
    problems = gsm_test.iloc[: args.n_problems]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + PROMPT_SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for q in problems["question"]]
    golds = [a.split("####")[-1].strip().replace(",", "") for a in problems["answer"]]
    sol_seqs = [chat_ids(q, clean_solution(a)) for q, a in zip(problems["question"], problems["answer"])]
    wt_test_ids = tok(wt_test, add_special_tokens=False).input_ids[: args.ppl_tokens]
    wt_seqs = [BOS + wt_test_ids[i : i + 512] for i in range(0, len(wt_test_ids) - 16, 512)]

    def ppl(seqs: list[list[int]]) -> float:
        nll, nt = 0.0, 0
        for s in seqs:
            ids = torch.tensor([s], device=dev)
            lp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
            nll -= float(lp.gather(1, ids[0, 1:, None]).sum())
            nt += len(s) - 1
        return math.exp(nll / nt)

    def gsm8k() -> dict:
        correct, n_gen, samples, per = 0, 0, [], []
        sync()
        t0 = time.perf_counter()
        for i in range(0, len(prompts), args.batch):
            enc = tok(prompts[i : i + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
            for j, row in enumerate(out[:, enc.input_ids.shape[1] :]):
                text = tok.decode(row, skip_special_tokens=True)
                pred = extract_number(text)
                ok = int(same_number(pred, golds[i + j]))
                correct += ok
                per.append(ok)
                n_gen += int((row != tok.pad_token_id).sum())
                if len(samples) < 3:
                    samples.append({"gold": golds[i + j], "pred": pred, "text": text[-300:]})
        sync()
        dt = time.perf_counter() - t0
        return {"accuracy": correct / len(prompts), "correct": correct, "n": len(prompts), "gen_tokens": n_gen,
                "wall_s": dt, "samples": samples, "per_problem": per}

    def decode_speed(n_steps: int = 256, batch: int = 32) -> float:
        """Pure decode tokens/s on a fixed workload (min_new_tokens forces the same length for every config)."""
        enc = tok(prompts[:batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        model.generate(**enc, max_new_tokens=8, min_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        sync()
        t0 = time.perf_counter()
        model.generate(**enc, max_new_tokens=n_steps, min_new_tokens=n_steps, do_sample=False,
                       pad_token_id=tok.pad_token_id)
        sync()
        return batch * n_steps / (time.perf_counter() - t0)

    def evaluate(index, recon: bool = False) -> dict:
        apply_index(index, recon)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        mlp_params = sum(p.numel() for m in mlps for p in m.parameters())
        all_params = sum(p.numel() for p in model.parameters())
        r = {"mlp_params": mlp_params, "model_params": all_params,
             "model_bytes_bf16": all_params * 2,
             "neurons_per_layer": [int(getattr(m, NM[0]).out_features) for m in mlps],
             "ppl_gsm8k_solutions": ppl(sol_seqs), "ppl_wikitext": ppl(wt_seqs),
             "decode_tok_s": decode_speed(), "gsm8k": gsm8k()}
        apply_index(None)
        return r

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    if out_p.exists():  # resume: keep configs already finished (e.g. saved to Google Drive before a disconnect)
        results = {k: v for k, v in json.loads(out_p.read_text(encoding="utf-8")).get("results", {}).items()
                   if k != "index_overlap_math_wiki"}
        print(f"resuming: {len(results)} configs already in {out_p}: {sorted(results)}", flush=True)

    def save() -> None:
        out_p.write_text(json.dumps({"model": args.model_dir, "n_problems": len(prompts),
                                     "calib_tokens": args.calib_tokens, "results": results,
                                     "wall_s": time.perf_counter() - t_start}, indent=2), encoding="utf-8")

    def show(name: str, r: dict) -> None:
        f = results["full"]
        print(f"[{name:20s}] GSM8K={r['gsm8k']['accuracy']:6.1%} ({r['gsm8k']['accuracy'] - f['gsm8k']['accuracy']:+.1%})"
              f"  ppl math={r['ppl_gsm8k_solutions']:.2f} wiki={r['ppl_wikitext']:.2f}"
              f"  size={r['model_params'] / f['model_params']:.0%}  decode={r['decode_tok_s'] / f['decode_tok_s']:.2f}x"
              f"  ({r['gsm8k']['wall_s']:.0f}s)", flush=True)

    if "full" not in results:
        results["full"] = evaluate(None)
        save()
    show("full", results["full"])
    for b in [float(x) for x in args.budgets.split(",")]:
        for kind in [x.strip() for x in args.indexes.split(",") if x.strip()]:
            if f"{kind}@{b}" in results:
                continue
            idx = build_index(kind, b)
            r = evaluate(idx, recon=kind.endswith("+recon"))
            r.update(index=kind, budget=b)
            if not kind.startswith("random"):
                src, mode = kind.removesuffix("+recon").split("-")
                s = score["math_energy" if mode == "energy" else src]
                r["contribution_kept"] = float(sum(s[li, idx[li]].sum() for li in range(L)) / s.sum())
            name = f"{kind}@{b}"
            results[name] = r
            save()
            show(name, r)
    # how different are the math and wiki indexes?
    for b in [float(x) for x in args.budgets.split(",")]:
        mi, wi = build_index("math-uniform", b), build_index("wiki-uniform", b)
        ov = sum(len(set(mi[li].tolist()) & set(wi[li].tolist())) for li in range(L)) / sum(len(x) for x in mi)
        results.setdefault("index_overlap_math_wiki", {})[str(b)] = round(ov, 4)
    save()
    print("math vs wiki index overlap:", results["index_overlap_math_wiki"])
    print(f"Saved {out_p} ({time.perf_counter() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
