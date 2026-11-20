"""Is the quality lost by a task index due to the SUMMARY (selection criterion) or to removing neurons at all?

Same setup as experiment_capture_index_route.py (Qwen3.5-0.8B, math index from GSM8K train, physical MLP slicing),
with five ways to pick the kept neurons at each budget, all computed in one calibration pass:
  * mean     : E|h_i| * ||w_i||           (the current wsi.capture weight usage; w_i = down-proj column i)
  * energy   : sqrt(E h_i^2) * ||w_i||    (weights rare-but-large activations more than the mean does)
  * max      : max|h_i| * ||w_i||         (protects neurons that fire rarely but strongly)
  * taylor   : E_seq |sum_t h_i(t) g_i(t)|, g = dLoss/dh  (first-order loss change if neuron i is removed)
  * +recon   : a criterion's selection, then the kept down-proj columns are refit by ridge least squares so the
               layer reproduces the full MLP output on the calibration data:  W_S = W G_{:,S} (G_SS + lam I)^-1,
               G = E[h h^T]. Changes only weights that stay loaded (compensation, as in SparseGPT-style pruning).
  * obs      : PAIRWISE selection (backlog #1, after HOPE, arXiv 2609.18916). The scores above rate each neuron alone;
               obs uses the whole Gram matrix. Greedy backward elimination (Optimal Brain Surgeon for a linear layer):
               drop the neuron whose removal, after the kept columns are refit, raises E||W h - W_S h_S||^2 the
               least, i.e. cost_j = ||W_S[:, j]||^2 / [(G_SS + lam I)^-1]_jj; update W_S and the inverse; repeat.
               Two neurons that always fire together are redundant: obs drops one and the refit moves its job to
               the other, while a per-neuron score keeps (or drops) both. HOPE selects with pairwise terms but never
               compensates; our +recon compensates but selects one neuron at a time. obs+recon does both.
Scored by teacher-forced perplexity on held-out GSM8K test solutions (in-domain) and WikiText-2 test (side effect).

    python scripts/experiment_index_criteria.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from wsi.capture import apply_selection, find_mlps  # noqa: E402
except ImportError:  # Colab: wsi/capture.py copied next to this script as wsi_capture.py
    from wsi_capture import apply_selection, find_mlps  # noqa: E402

HUB_FILES = {
    "gsm8k/test.parquet": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    "gsm8k/train.parquet": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
    "wikitext/test.parquet": ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
}


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


def ridge_refit(G: torch.Tensor, W: torch.Tensor, ix: torch.Tensor, ridge: float) -> torch.Tensor:
    """Kept down-proj columns refit so W_S h_S reproduces W h on the calibration data. G = E[h h^T], W [hidden, U]."""
    G, W = G.double(), W.double()
    Gss = G[ix][:, ix]
    lam = ridge * Gss.diagonal().mean()
    rhs = W @ G[:, ix]  # [hidden, |S|]
    return torch.linalg.solve(Gss + lam * torch.eye(len(ix), device=G.device, dtype=torch.float64), rhs.T).T


def obs_select(G: torch.Tensor, W: torch.Tensor, keeps: list[int], ridge: float) -> dict[int, torch.Tensor]:
    """Pairwise neuron selection by greedy backward elimination (OBS). Returns {k: sorted kept indices} for each k.

    Loss of a kept set S with refit columns W_S: E||W h - W_S h_S||^2 + lam ||W_S||^2, Hessian H_SS = G_SS + lam I.
    Removing column j and refitting the rest raises it by exactly ||W_S[:, j]||^2 / (H_SS^-1)_jj; W_S and H_SS^-1
    then get the rank-1 OBS update. One pass from all U neurons down to min(keeps) yields every budget.
    """
    G, W = G.double(), W.double()
    U = G.shape[0]
    lam = ridge * G.diagonal().mean()
    Hinv = torch.linalg.inv(G + lam * torch.eye(U, device=G.device, dtype=torch.float64))
    Ws = W @ G @ Hinv  # ridge refit with every neuron kept
    alive = torch.ones(U, dtype=torch.bool, device=G.device)
    inf = torch.tensor(float("inf"), device=G.device, dtype=torch.float64)
    out, n = {}, U
    for k in sorted(set(keeps), reverse=True):
        while n > k:
            d = Hinv.diagonal()
            cost = torch.where(alive, (Ws * Ws).sum(0) / d.clamp_min(1e-300), inf)
            j = int(cost.argmin())
            hj = Hinv[:, j].clone()
            Ws.addr_(Ws[:, j].clone(), hj / hj[j], alpha=-1.0)  # in place: no U x U temporaries
            Hinv.addr_(hj, hj / hj[j], alpha=-1.0)
            Ws[:, j] = 0.0
            Hinv[j, :] = 0.0
            Hinv[:, j] = 0.0
            alive[j] = False
            n -= 1
        out[k] = alive.nonzero().squeeze(1)
    return out


def clean_solution(ans: str) -> str:
    body, final = ans.split("####")
    return re.sub(r"<<[^>]*>>", "", body).strip() + f"\nThe answer is {final.strip()}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "qwen35-0.8b-full"))
    ap.add_argument("--data-dir", default=str(ROOT / "work"))
    ap.add_argument("--calib-tokens", type=int, default=60_000)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--budgets", default="0.7,0.5,0.3")
    ap.add_argument("--taylor-seqs", type=int, default=150, help="sequences used for the gradient (Taylor) pass")
    ap.add_argument("--ridge", type=float, default=1e-2, help="lambda as a fraction of mean diag(G)")
    ap.add_argument("--criteria", default="mean,energy,max,taylor,obs",
                    help="any of mean,energy,max,taylor,obs (taylor needs a backward pass; obs a greedy pass per layer)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "index_criteria" / "results.json"))
    args = ap.parse_args()
    crits = [c.strip() for c in args.criteria.split(",") if c.strip()]
    assert set(crits) <= {"mean", "energy", "max", "taylor", "obs"}, crits

    torch.manual_seed(0)
    dev = torch.device("cuda")
    data = Path(args.data_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=getattr(torch, args.dtype)).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mods = dict(model.named_modules())
    groups = find_mlps(model)
    downs = [g.down for g in groups]
    U = groups[0].units
    print(f"{len(groups)} MLPs x {U} neurons")

    def chat(q: str, a: str) -> list[int]:
        txt = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                      tokenize=False, enable_thinking=False)
        return tok(txt, add_special_tokens=False).input_ids

    gtr = pd.read_parquet(need(data, "gsm8k/train.parquet"))
    gte = pd.read_parquet(need(data, "gsm8k/test.parquet")).iloc[: args.n_eval]
    calib, n = [], 0
    for q, a in zip(gtr["question"], gtr["answer"]):
        calib.append(chat(q, clean_solution(a)))
        n += len(calib[-1])
        if n >= args.calib_tokens:
            break
    wt = tok("\n".join(pd.read_parquet(need(data, "wikitext/test.parquet"))["text"]), add_special_tokens=False).input_ids
    held = {"math": [chat(q, clean_solution(a)) for q, a in zip(gte["question"], gte["answer"])],
            "wiki": [wt[i : i + 256] for i in range(0, 256 * args.n_eval, 256)]}

    # ---------------------------------------------------------------- one calibration pass (forward + backward)
    st = {d: {"abs": torch.zeros(U, device=dev, dtype=torch.float64),
              "sq": torch.zeros(U, device=dev, dtype=torch.float64),
              "max": torch.zeros(U, device=dev, dtype=torch.float64),
              "taylor": torch.zeros(U, device=dev, dtype=torch.float64),
              "G": torch.zeros(U, U, device=dev, dtype=torch.float32)} for d in downs}
    leaves: dict[str, torch.Tensor] = {}

    def pre_hook(name):
        def hook(mod, inp):
            h = inp[0].detach().requires_grad_(True)  # leaf: gives dLoss/dh without parameter grads
            leaves[name] = h
            x = h.detach().reshape(-1, U).float()
            s = st[name]
            s["abs"] += x.abs().sum(0).double()
            s["sq"] += (x * x).sum(0).double()
            s["max"] = torch.maximum(s["max"], x.abs().max(0).values.double())
            s["G"] += x.T @ x
            return (h,) + tuple(inp[1:])
        return hook

    state = {"grad": False}

    def stat_or_grad_hook(name):
        inner = pre_hook(name)

        def hook(mod, inp):
            if state["grad"]:  # gradient pass: only make h a leaf, no statistics
                h = inp[0].detach().requires_grad_(True)
                leaves[name] = h
                return (h,) + tuple(inp[1:])
            inner(mod, inp)
            return None
        return hook

    hs = [mods[d].register_forward_pre_hook(stat_or_grad_hook(d)) for d in downs]
    t0 = time.perf_counter()
    ntok = 0
    with torch.no_grad():  # pass 1: activation statistics and Gram matrices over all calibration tokens
        for s in calib:
            model(input_ids=torch.tensor([s], device=dev), use_cache=False)
            ntok += len(s)
    leaves.clear()
    print(f"pass 1: {ntok:,} math tokens ({len(calib)} sequences) in {time.perf_counter() - t0:.0f}s, "
          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
    # pass 2: Taylor scores need dLoss/dh; a subset of sequences is enough and keeps memory bounded
    state["grad"] = True
    grad_seqs = calib[: args.taylor_seqs] if "taylor" in crits else []
    t1 = time.perf_counter()
    for j, s in enumerate(grad_seqs):
        ids = torch.tensor([s], device=dev)
        with torch.enable_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
            F.cross_entropy(logits, ids[0, 1:], reduction="sum").backward()
        for d in downs:
            h = leaves[d]
            st[d]["taylor"] += (h.detach().float() * h.grad.float()).reshape(-1, U).sum(0).abs().double()
        leaves.clear()
        del logits
        if (j + 1) % 50 == 0:
            print(f"  pass 2: {j + 1}/{len(grad_seqs)} seqs, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB "
                  f"(peak {torch.cuda.max_memory_allocated() / 2**30:.1f})", flush=True)
    state["grad"] = False
    for h in hs:
        h.remove()
    torch.cuda.empty_cache()
    print(f"pass 2: Taylor on {len(grad_seqs)} sequences in {time.perf_counter() - t1:.0f}s", flush=True)

    wnorm = {d: mods[d].weight.float().norm(dim=0).double() for d in downs}
    score = {
        "mean": {d: st[d]["abs"] / ntok * wnorm[d] for d in downs},
        "energy": {d: (st[d]["sq"] / ntok).sqrt() * wnorm[d] for d in downs},
        "max": {d: st[d]["max"] * wnorm[d] for d in downs},
        "taylor": {d: st[d]["taylor"] / max(len(grad_seqs), 1) for d in downs},
    }
    budgets = [float(x) for x in args.budgets.split(",")]
    obs_sel: dict[int, dict[str, torch.Tensor]] = {}
    obs_s = 0.0
    if "obs" in crits:  # one greedy elimination per layer gives every budget
        t2 = time.perf_counter()
        keeps = [round(b * U) for b in budgets]
        for i, d in enumerate(downs):
            for k, ix in obs_select(st[d]["G"] / ntok, mods[d].weight, keeps, args.ridge).items():
                obs_sel.setdefault(k, {})[d] = ix
            if (i + 1) % 6 == 0:
                print(f"  obs: {i + 1}/{len(downs)} layers, {time.perf_counter() - t2:.0f}s", flush=True)
        obs_s = time.perf_counter() - t2
        print(f"obs selection in {obs_s:.0f}s", flush=True)

    def select(crit: str, b: float) -> dict[str, torch.Tensor]:
        k = round(b * U)
        if crit == "obs":
            return obs_sel[k]
        return {d: score[crit][d].topk(k).indices.sort().values for d in downs}

    def recon_weight(d: str, ix: torch.Tensor) -> torch.Tensor:
        return ridge_refit(st[d]["G"] / ntok, mods[d].weight, ix, args.ridge).to(mods[d].weight.dtype)

    @torch.no_grad()
    def ppl(seqs) -> float:
        nll, nt = 0.0, 0
        for s in seqs:
            ids = torch.tensor([s], device=dev)
            lp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
            nll -= float(lp.gather(1, ids[0, 1:, None]).sum())
            nt += len(s) - 1
        return math.exp(nll / nt)

    def evaluate(sel: dict | None, recon: bool = False) -> dict:
        if sel is None:
            return {t: round(ppl(s), 3) for t, s in held.items()}
        new_w = {d: recon_weight(d, ix.to(dev)) for d, ix in sel.items()} if recon else {}
        with apply_selection(model, sel):
            for d, w in new_w.items():
                mods[d].weight.data.copy_(w)
            return {t: round(ppl(s), 3) for t, s in held.items()}

    res = {"model": args.model_dir, "calib_tokens": ntok, "criteria": crits, "ridge": args.ridge,
           "obs_select_s": round(obs_s, 1), "full": evaluate(None), "budgets": {}}
    print(f"[full] math={res['full']['math']:.3f} wiki={res['full']['wiki']:.3f}", flush=True)
    for b in budgets:
        row = {}
        for crit in crits:
            sel = select(crit, b)
            row[crit] = evaluate(sel)
            row[crit + "+recon"] = evaluate(sel, recon=True)
        g = torch.Generator().manual_seed(0)
        row["random"] = evaluate({d: torch.randperm(U, generator=g)[: round(b * U)].sort().values for d in downs})
        ref = select("mean", b)  # mean is cheap, always available
        row["overlap_with_mean"] = {c: round(sum(len(set(select(c, b)[d].tolist()) & set(ref[d].tolist()))
                                                 for d in downs) / (len(downs) * round(b * U)), 4)
                                    for c in crits if c != "mean"}
        res["budgets"][str(b)] = row
        print(f"\n=== {b:.0%} of neurons ===   (math full {res['full']['math']:.2f}, wiki full {res['full']['wiki']:.2f})")
        for c, v in row.items():
            if c != "overlap_with_mean":
                print(f"  {c:14s} math={v['math']:9.3f}  wiki={v['wiki']:9.3f}")
        print("  overlap with 'mean':", row["overlap_with_mean"], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    main()
