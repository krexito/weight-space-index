"""Extreme MoI on LiquidAI/LFM3-1.7B: does the model stay smart (GSM8K) when only k% of MLP neurons are loaded?

This is the "run the whole model from the index" test on a real, capable model:
  * Always resident ("brain"): embeddings, gated short convolutions, attention, norms.
  * Paged per token: SwiGLU MLP neurons (24 layers x 7,680 units; ~2/3 of the model's parameters).
    A neuron = column i of w1 (gate) + column i of w3 (up) + row i of w2 (down).
  * Neurons that are not loaded contribute 0 (SwiGLU has no constant offset: silu(0)*0 = 0).

Policies (k = fraction of the 7,680 neurons per layer that may fire on each token):
  * full    : unmodified model (reference).
  * oracle  : top-k neurons by |silu(w1 x) * (w3 x)| at that position. Needs the real activation -> upper bound.
  * gate    : top-k by |silu(w1 x)| (CATS/TEAL style). Needs w1 resident (1/3 of MLP memory); only w3/w2 rows paged.
  * trainedR: Deja-Vu-style learned rank-R router on the MLP input x, initialized from the SVD of w1 and trained
              on calibration text (GSM8K TRAIN solutions + WikiText-2 validation) to predict top neurons.
              Only the router (R*(d+U) floats per layer) is resident; w1/w3/w2 rows are paged.

Intelligence check: GSM8K TEST (exact-match on the final number), greedy decoding, chat template with
thinking disabled. Also reports WikiText-2 test perplexity on a short slice.

Resources: the masking here is done densely (it measures quality, not speed). For each policy the report gives the
MLP bytes that would have to be resident or paged per token if the selection were executed sparsely.

Run with the isolated env (LFM3 custom code needs transformers>=5.12):
    work/venv-lfm3/Scripts/python.exe scripts/experiment_moi_lfm3_gsm8k.py --n-problems 100
On Colab / any Linux box (model and data are downloaded from the Hub when the local copies are missing):
    python experiment_moi_lfm3_gsm8k.py --model-dir LiquidAI/LFM3-1.7B --data-dir /content/data --out /content/r.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "lfm3-1.7b-full"
DATA = ROOT / "work"


HUB_FILES = {  # local name -> (dataset repo, file in repo)
    "gsm8k/test.parquet": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    "gsm8k/train.parquet": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
    "wikitext/test.parquet": ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
    "wikitext/validation.parquet": ("Salesforce/wikitext", "wikitext-2-raw-v1/validation-00000-of-00001.parquet"),
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
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--budgets", type=str, default="0.5,0.3,0.2")
    ap.add_argument("--policies", type=str, default="oracle,gate,trained256")
    ap.add_argument("--calib-tokens", type=int, default=60_000)
    ap.add_argument("--train-epochs", type=int, default=3)
    ap.add_argument("--train-lr", type=float, default=1e-3)
    ap.add_argument("--ppl-tokens", type=int, default=8_000)
    ap.add_argument("--out", type=str, default=str(ROOT / "benchmarks" / "moi_lfm3_gsm8k" / "results.json"))
    ap.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="local folder or Hub id")
    ap.add_argument("--data-dir", type=str, default=str(DATA))
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    t_start = time.perf_counter()
    data_dir = Path(args.data_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True,
                                                 dtype=torch.bfloat16).to(dev).eval()
    layers = model.model.layers
    n_layers = len(layers)
    d_model = int(model.config.hidden_size)
    n_units = int(model.config.intermediate_size)
    budgets = [float(b) for b in args.budgets.split(",")]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    print(f"LFM3 loaded: {n_layers} layers, d={d_model}, MLP units={n_units}")

    # ---- patch every SwiGLU MLP: capture / mask ----
    zero = lambda: torch.zeros((), device=dev, dtype=torch.float64)  # noqa: E731  (accumulate on GPU, no syncs)
    ctx: dict = {"mode": "off", "policy": None, "k": 0, "cap": {}, "mass_sel": zero(), "mass_tot": zero()}
    routers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def make_forward(li: int, mlp):
        def forward(x):
            g = F.silu(mlp.w1(x))
            h = g * mlp.w3(x)
            mode = ctx["mode"]
            if mode == "capture":
                ctx["cap"][li] = (x.reshape(-1, d_model).float(), h.reshape(-1, n_units).float())
            elif mode == "mask":
                k = ctx["k"]
                pol = ctx["policy"]
                hf = h.reshape(-1, n_units)
                if pol == "oracle":
                    score = hf.abs()
                elif pol == "gate":
                    score = g.reshape(-1, n_units).abs()
                else:  # trainedR
                    A, B = routers[li]
                    score = x.reshape(-1, d_model).float() @ A @ B
                idx = score.topk(k, dim=-1).indices
                mask = torch.zeros_like(hf, dtype=torch.bool).scatter_(1, idx, True)
                ah = hf.abs().float()
                ctx["mass_sel"] += (ah * mask).sum()
                ctx["mass_tot"] += ah.sum()
                h = (hf * mask).reshape(h.shape)
            return mlp.w2(h)

        return forward

    for li, lyr in enumerate(layers):
        lyr.feed_forward.forward = make_forward(li, lyr.feed_forward)

    # ---- data ----
    gsm_test = pd.read_parquet(need(data_dir, "gsm8k/test.parquet"))
    gsm_train = pd.read_parquet(need(data_dir, "gsm8k/train.parquet"))
    wt_val = "\n".join(pd.read_parquet(need(data_dir, "wikitext/validation.parquet"))["text"].tolist())
    wt_test = "\n".join(pd.read_parquet(need(data_dir, "wikitext/test.parquet"))["text"].tolist())
    problems = gsm_test.iloc[: args.n_problems]

    # ---- router training (only when a trainedR policy is requested) ----
    router_info: dict = {}
    train_pols = [p for p in policies if p.startswith("trained")]
    if train_pols:
        if len(train_pols) > 1:
            raise SystemExit("Use a single trainedR policy per run.")
        rank = int(train_pols[0][len("trained"):])
        calib_text = "\n\n".join((gsm_train["question"] + "\n" + gsm_train["answer"]).tolist()[:3000])
        calib_ids = tok(calib_text, add_special_tokens=False).input_ids[: args.calib_tokens // 2]
        calib_ids += tok(wt_val, add_special_tokens=False).input_ids[: args.calib_tokens // 2]
        chunks = [calib_ids[i : i + 512] for i in range(0, len(calib_ids) - 16, 512)]
        params = []
        for lyr in layers:
            W1 = lyr.feed_forward.w1.weight.float().T  # [d, U]
            U_, S_, Vh_ = torch.linalg.svd(W1, full_matrices=False)
            params.append([U_[:, :rank].clone().requires_grad_(True),
                           (S_[:rank, None] * Vh_[:rank]).clone().requires_grad_(True)])
        opts = [torch.optim.Adam(p, lr=args.train_lr) for p in params]
        t_tr = time.perf_counter()
        ctx["mode"] = "capture"
        for ep in range(args.train_epochs):
            order = torch.randperm(len(chunks), generator=torch.Generator().manual_seed(ep)).tolist()
            tot = 0.0
            for ci in order:
                model(input_ids=torch.tensor([[tok.bos_token_id] + chunks[ci]], device=dev))
                for li in range(n_layers):
                    x, h = ctx["cap"][li]
                    ah = h.abs()
                    tgt = (ah >= ah.quantile(0.7, dim=-1, keepdim=True)).float()  # top-30% neurons
                    w = 1.0 + ah / (ah.mean(dim=-1, keepdim=True) + 1e-6)
                    A, B = params[li]
                    with torch.enable_grad():
                        loss = F.binary_cross_entropy_with_logits(x @ A @ B, tgt, weight=w)
                        opts[li].zero_grad()
                        loss.backward()
                        opts[li].step()
                    tot += float(loss)
            print(f"  router epoch {ep + 1}/{args.train_epochs}: mean loss {tot / (len(chunks) * n_layers):.4f}")
        ctx["mode"] = "off"
        ctx["cap"] = {}
        for li, (A, B) in enumerate(params):
            routers[li] = (A.detach(), B.detach())
        router_info = {"rank": rank, "calib_tokens": len(calib_ids), "train_s": time.perf_counter() - t_tr,
                       "router_bytes_fp32": n_layers * rank * (d_model + n_units) * 4,
                       "router_macs_per_token": n_layers * rank * (d_model + n_units)}
        print(f"Trained rank-{rank} router in {router_info['train_s']:.1f}s")

    # ---- evaluators ----
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": q + "\n\nSolve step by step. End with: The answer is <number>."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in problems["question"]]
    golds = [a.split("####")[-1].strip().replace(",", "") for a in problems["answer"]]

    def run_gsm8k() -> dict:
        correct = 0
        n_gen = 0
        samples = []
        t0 = time.perf_counter()
        for i in range(0, len(prompts), args.batch):
            enc = tok(prompts[i : i + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
            for j, row in enumerate(out[:, enc.input_ids.shape[1]:]):
                text = tok.decode(row, skip_special_tokens=True)
                pred = extract_number(text)
                ok = same_number(pred, golds[i + j])
                correct += int(ok)
                n_gen += int((row != tok.pad_token_id).sum())
                if len(samples) < 3:
                    samples.append({"gold": golds[i + j], "pred": pred, "text": text[-300:]})
        return {"accuracy": correct / len(prompts), "correct": correct, "n": len(prompts),
                "gen_tokens": n_gen, "wall_s": time.perf_counter() - t0, "samples": samples}

    ppl_ids = tok(wt_test, add_special_tokens=False).input_ids[: args.ppl_tokens]

    def run_ppl() -> float:
        nll, nt = 0.0, 0
        for i in range(0, len(ppl_ids) - 16, 512):
            ids = torch.tensor([[tok.bos_token_id] + ppl_ids[i : i + 512]], device=dev)
            lp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
            nll -= float(lp.gather(1, ids[0, 1:, None]).sum())
            nt += ids.shape[1] - 1
        return math.exp(nll / nt)

    def evaluate(mode: str, policy: str | None = None, k: int = 0) -> dict:
        ctx.update(mode=mode, policy=policy, k=k, mass_sel=zero(), mass_tot=zero())
        r = {"wikitext_ppl": run_ppl()}
        r["gsm8k"] = run_gsm8k()
        if mode == "mask":
            r["activation_mass_recall"] = float(ctx["mass_sel"] / ctx["mass_tot"].clamp_min(1e-12))
        ctx["mode"] = "off"
        return r

    # MLP bytes per token and layer that the policy needs (bf16), if the selection were executed sparsely
    nb = 2 * d_model  # one row/column of one matrix
    full_mlp = 3 * n_units * nb

    def mlp_bytes(policy: str | None, k: int) -> dict:
        if policy is None:
            resident, paged = full_mlp, 0
        elif policy == "oracle":  # needs w1 and w3 for every neuron to know the activation
            resident, paged = 2 * n_units * nb, k * nb
        elif policy == "gate":
            resident, paged = n_units * nb, 2 * k * nb
        else:
            rank = int(policy[len("trained"):])
            resident, paged = rank * (d_model + n_units) * 4, 3 * k * nb
        return {"resident_per_layer": resident, "touched_per_token_per_layer": resident + paged,
                "share_of_full_mlp": round((resident + paged) / full_mlp, 4)}

    p_all = sum(p.numel() for p in model.parameters())
    p_mlp = sum(p.numel() for n, p in model.named_parameters() if ".feed_forward." in n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(results: dict) -> None:  # after every configuration: a dropped Colab session keeps what finished
        summary = {
            "model": str(args.model_dir), "n_problems": len(prompts), "max_new_tokens": args.max_new_tokens,
            "params_total": p_all, "params_mlp": p_mlp, "mlp_share": p_mlp / p_all,
            "neuron_bytes_bf16": 3 * d_model * 2, "router": router_info, "results": results,
            "wall_s": time.perf_counter() - t_start,
        }
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    results: dict = {"full": {**evaluate("off"), "mlp_bytes": mlp_bytes(None, n_units)}}
    save(results)
    print(f"[full]           GSM8K={results['full']['gsm8k']['accuracy']:.1%}  ppl={results['full']['wikitext_ppl']:.2f}",
          flush=True)
    for b in budgets:
        k = max(1, round(b * n_units))
        for pol in policies:
            r = evaluate("mask", pol, k)
            r.update(policy=pol, budget=b, k_per_layer=k, mlp_bytes=mlp_bytes(pol, k))
            results[f"{pol}@{b}"] = r
            save(results)
            print(f"[{pol:10s} {b:4.0%}] GSM8K={r['gsm8k']['accuracy']:.1%}  ppl={r['wikitext_ppl']:.2f}  "
                  f"recall={r['activation_mass_recall']:.1%}  MLP bytes/token={r['mlp_bytes']['share_of_full_mlp']:.0%}"
                  f"  ({r['gsm8k']['wall_s']:.0f}s)", flush=True)
    print(f"Saved {out} ({time.perf_counter() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
