"""How small can the compensation summary (the Gram matrix) get before Compensated MoI loses quality?

Compensated (Selective) MoI keeps a subset S of MLP neurons and refits the kept down-proj columns with
    W_S = W G[:,S] (G_SS + lam I)^-1 ,   G = E[h h^T]  (co-activation of the MLP neurons on the task),
which recovered most of the quality lost by plain selection (benchmarks/index_criteria). The full G is U x U per
layer (~1.2 GB for Qwen3.5-0.8B in fp32). This script keeps the selection fixed (the `energy` criterion) and only
changes the G used for the refit:
  * full          : exact fp32 Gram from all calibration tokens (reference)
  * fp16          : stored as scaled fp16 (G / max|G|), upper triangle only -> 4x smaller
  * lowrank-k     : top-k eigenvectors of G only (k = 32..1024)
  * lowrank-k+diag: top-k eigenvectors plus the exact diagonal (k*U + U numbers)
  * tokens-N      : Gram re-estimated from only the first N calibration tokens (store the tokens, recompute G)
Scored by teacher-forced perplexity on held-out GSM8K test solutions and WikiText-2 test, as in
experiment_index_criteria.py. Reports the storage of each summary.

    python scripts/experiment_gram_compress.py
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


def clean_solution(ans: str) -> str:
    body, final = ans.split("####")
    return re.sub(r"<<[^>]*>>", "", body).strip() + f"\nThe answer is {final.strip()}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "qwen35-0.8b-full"))
    ap.add_argument("--data-dir", default=str(ROOT / "work"))
    ap.add_argument("--calib-tokens", type=int, default=60_000)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--budgets", default="0.5,0.3")
    ap.add_argument("--ranks", default="32,128,256,1024")
    ap.add_argument("--token-subsets", default="2000,5000,20000")
    ap.add_argument("--ridge", type=float, default=1e-2, help="lambda as a fraction of mean diag(G_SS)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "gram_compress" / "results.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device(args.device)
    data = Path(args.data_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mods = dict(model.named_modules())
    downs = [g.down for g in find_mlps(model)]
    U = find_mlps(model)[0].units
    L = len(downs)
    ranks = [k for k in (int(x) for x in args.ranks.split(",")) if k < U]
    subsets = sorted(int(x) for x in args.token_subsets.split(","))
    print(f"{L} MLPs x {U} neurons; ranks {ranks}; token subsets {subsets}")

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

    # ------------------------------------------------ calibration: energy scores + Gram (with snapshots at N tokens)
    st = {d: {"sq": torch.zeros(U, device=dev, dtype=torch.float64),
              "G": torch.zeros(U, U, device=dev, dtype=torch.float32)} for d in downs}

    def hook(name):
        def f(mod, inp):
            x = inp[0].detach().reshape(-1, U).float()
            st[name]["sq"] += (x * x).sum(0).double()
            st[name]["G"] += x.T @ x
        return f

    hs = [mods[d].register_forward_pre_hook(hook(d)) for d in downs]
    snaps: dict[int, dict[str, torch.Tensor]] = {}
    t0, ntok = time.perf_counter(), 0
    with torch.no_grad():
        for s in calib:
            model(input_ids=torch.tensor([s], device=dev), use_cache=False)
            ntok += len(s)
            for N in subsets:
                if N not in snaps and ntok >= N:
                    snaps[N] = {d: (st[d]["G"] / ntok).cpu() for d in downs}
                    snaps[N]["_ntok"] = ntok
    for h in hs:
        h.remove()
    print(f"calibration: {ntok:,} tokens ({len(calib)} seqs) in {time.perf_counter() - t0:.0f}s", flush=True)

    Gfull = {d: st[d]["G"].double() / ntok for d in downs}
    wnorm = {d: mods[d].weight.float().norm(dim=0).double() for d in downs}
    energy = {d: (st[d]["sq"] / ntok).sqrt() * wnorm[d] for d in downs}

    t1 = time.perf_counter()
    kmax = max(ranks) if ranks else 0
    eig = {}
    for d in downs:  # top eigenpairs once per layer; every low-rank variant is a prefix of these
        e, V = torch.linalg.eigh(Gfull[d])
        e, V = e.flip(0)[:kmax], V.flip(1)[:, :kmax]
        eig[d] = (e.clamp_min(0), V)
    spectrum = {}
    for k in ranks:  # fraction of the Gram's trace (activation energy) captured by the top-k directions
        spectrum[str(k)] = round(sum(float(eig[d][0][:k].sum() / Gfull[d].trace()) for d in downs) / L, 4)
    print(f"eigendecomposition in {time.perf_counter() - t1:.0f}s; energy captured by top-k: {spectrum}", flush=True)

    def gram(variant: str, d: str) -> torch.Tensor:
        G = Gfull[d]
        if variant == "full":
            return G
        if variant == "fp16":
            s = G.abs().max()
            return (G / s).to(torch.float16).double() * s
        if variant.startswith("lowrank-"):
            k = int(variant.split("-")[1].split("+")[0])
            e, V = eig[d]
            Gk = (V[:, :k] * e[:k]) @ V[:, :k].T
            if variant.endswith("+diag"):
                Gk = Gk + torch.diag(G.diagonal() - Gk.diagonal())
            return Gk
        if variant.startswith("tokens-"):
            return snaps[int(variant.split("-")[1])][d].to(dev).double()
        raise ValueError(variant)

    def size_bytes(variant: str) -> int:
        if variant == "full":
            return L * U * U * 4
        if variant == "fp16":
            return L * U * (U + 1) // 2 * 2
        if variant.startswith("lowrank-"):
            k = int(variant.split("-")[1].split("+")[0])
            return L * ((U * k + k) * 2 + (U * 2 if variant.endswith("+diag") else 0))
        if variant.startswith("tokens-"):
            return int(snaps[int(variant.split("-")[1])]["_ntok"]) * 4  # the token ids; G is recomputed
        return 0

    def recon_weight(d: str, ix: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        W = mods[d].weight.double()
        Gss = G[ix][:, ix]
        lam = args.ridge * Gss.diagonal().mean().clamp_min(1e-12)
        Ws = torch.linalg.solve(Gss + lam * torch.eye(len(ix), device=dev, dtype=torch.float64), (W @ G[:, ix]).T).T
        return Ws.to(mods[d].weight.dtype)

    @torch.no_grad()
    def ppl(seqs) -> float:
        nll, nt = 0.0, 0
        for s in seqs:
            ids = torch.tensor([s], device=dev)
            lp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
            nll -= float(lp.gather(1, ids[0, 1:, None]).sum())
            nt += len(s) - 1
        return math.exp(nll / nt)

    def evaluate(sel: dict | None, variant: str | None = None) -> dict:
        if sel is None:
            return {t: round(ppl(s), 3) for t, s in held.items()}
        new_w = {d: recon_weight(d, ix.to(dev), gram(variant, d)) for d, ix in sel.items()} if variant else {}
        with apply_selection(model, sel):
            for d, w in new_w.items():
                mods[d].weight.data.copy_(w)
            return {t: round(ppl(s), 3) for t, s in held.items()}

    variants = (["full", "fp16"] + [f"lowrank-{k}{x}" for k in ranks for x in ("", "+diag")]
                + [f"tokens-{N}" for N in subsets if N in snaps])
    res = {"model": args.model_dir, "calib_tokens": ntok, "neurons_per_layer": U, "layers": L,
           "spectrum_energy_topk": spectrum, "full": evaluate(None),
           "size_bytes": {v: size_bytes(v) for v in variants}, "budgets": {}}
    print(f"[full model] math={res['full']['math']:.3f} wiki={res['full']['wiki']:.3f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for b in [float(x) for x in args.budgets.split(",")]:
        k = round(b * U)
        sel = {d: energy[d].topk(k).indices.sort().values for d in downs}
        row = {"no-recon": evaluate(sel)}
        for v in variants:
            row[v] = evaluate(sel, v)
        res["budgets"][str(b)] = row
        print(f"\n=== {b:.0%} of neurons (energy) ===   full model: math {res['full']['math']:.2f}")
        for v, r in row.items():
            mb = res["size_bytes"].get(v, 0) / 2**20
            print(f"  {v:18s} {mb:9.1f} MB   math={r['math']:8.3f}  wiki={r['wiki']:9.3f}", flush=True)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    main()
