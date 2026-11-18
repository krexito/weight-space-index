"""Does a "settling" signal predict turn-1 correctness better than the answer probability? (backlog #8b)

The sycophancy patch (scripts/moi_sycophancy.py, v3) switches on only when the model was confident in the answer being
challenged. The gate uses maxprob, the probability of its own turn-1 letter (AUC 0.83 for correctness on Qwen3.5-4B).
Idea from Energy-Based Reasoners (Chen, Gladstone, Li, Ji & Du 2026, arXiv 2611.08977): correct answers settle and
wrong ones keep moving. Here "moving" is read with the logit lens: the letter distribution at every layer of the turn-1
forward pass (final norm + lm_head on hidden_states[l], last prompt token).

Signals (higher = more confident):
  maxprob     final probability of the chosen letter (the current gate)
  margin      final top-1 minus top-2 probability
  settle      -(first layer from which the top-1 letter equals the final letter, up to the end) / L
  flips       -(top-1 changes across the second half of the layers)
  late_ent    -(mean letter entropy over the last quarter of layers, final layer excluded)
  late_pf     mean probability of the final letter over the last quarter of layers, final layer excluded
  combined    logistic regression on all of them, fit on the MMLU validation split, scored on test
  maxprob+margin  the same regression with the two final-layer signals only (does the lens add anything?)

    CUDA_VISIBLE_DEVICES="" python scripts/moi_settling_gate.py --model models/qwen35-0.8b-full \
        --data-dir work/mmlu --cpu --out benchmarks/moi_sycophancy/settling_gate_qwen35_08b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moi_sycophancy import Runner, load_items, q_text  # noqa: E402


def auc(score, y):
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def boot_diff(a, b, y, n=2000, seed=0):
    """Paired bootstrap of AUC(a) - AUC(b) over items."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(auc(a[i], y[i]) - auc(b[i], y[i]))
    d = np.array(d)
    return {"diff": round(auc(a, y) - auc(b, y), 4), "ci95": [round(float(np.quantile(d, 0.025)), 4),
                                                             round(float(np.quantile(d, 0.975)), 4)]}


@torch.no_grad()
def lens_letters(R: Runner, hid: np.ndarray, chunk: int = 64) -> np.ndarray:
    """hid [n, L+1, D] (fp16, last token) -> letter probabilities [n, L+1, 4] at every layer via the logit lens."""
    norm = next(m for n, m in R.model.named_modules() if n.endswith(".norm") and "layers" not in n and "mtp" not in n
                and "visual" not in n)
    W = R.model.get_output_embeddings().weight
    rows = [torch.tensor(ids) for ids in R.letter_ids]
    n, L1, _ = hid.shape
    out = np.zeros((n, L1, 4), np.float32)
    for s in range(0, n, chunk):
        h = torch.from_numpy(hid[s:s + chunk].astype(np.float32)).to(W.device, W.dtype)
        for l in range(L1):
            x = h[:, l] if l == L1 - 1 else norm(h[:, l])  # hidden_states[-1] already has the final norm
            lp = torch.log_softmax((x @ W.T).float(), -1)
            sc = torch.stack([torch.logsumexp(lp[:, r], -1) for r in rows], -1)
            out[s:s + chunk, l] = torch.softmax(sc, -1).cpu().numpy()
    return out


def signals(final: np.ndarray, lens: np.ndarray) -> dict[str, np.ndarray]:
    n, L1, _ = lens.shape
    L = L1 - 1
    f = final.argmax(1)
    srt = np.sort(final, 1)
    top = lens.argmax(2)  # [n, L+1]
    same = top == f[:, None]
    # first layer s such that top-1 == f for every layer >= s
    tail_ok = np.flip(np.cumprod(np.flip(same, 1), 1), 1).astype(bool)
    settle = np.where(tail_ok.any(1), tail_ok.argmax(1), L1) / L
    half = L1 // 2
    flips = (top[:, half + 1:] != top[:, half:-1]).sum(1)
    q = max(1, L // 4)
    late = lens[:, L - q:L]  # last quarter, final layer excluded
    ent = -(late * np.log(np.clip(late, 1e-9, 1))).sum(2).mean(1)
    pf = np.take_along_axis(late, f[:, None, None].repeat(late.shape[1], 1), 2)[..., 0].mean(1)
    return {"maxprob": final.max(1), "margin": srt[:, -1] - srt[:, -2], "settle": -settle,
            "flips": -flips.astype(float), "late_ent": -ent, "late_pf": pf}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-fit", type=int, default=1531, help="MMLU validation items to fit the combination")
    ap.add_argument("--n-eval", type=int, default=1000, help="MMLU test items to score it")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--tok-budget", type=int, default=4096)
    ap.add_argument("--max-bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    data = load_items(a.n_fit, 0, a.n_eval, a.seed, a.data_dir)
    R = Runner(a.model, a.dtype, a.tok_budget, a.max_bs, a.cpu)
    ALL = list(range(R.L + 1))
    res = {"model": a.model, "dtype": R.dtype_name, "n_fit": len(data["diag"]), "n_eval": len(data["test"])}
    sig, ys = {}, {}
    for split in ("diag", "test"):
        items = data[split]
        probs, hid = R.run([R.render([{"role": "user", "content": q_text(it)}]) for it in items], ALL)
        lens = lens_letters(R, hid)
        res[f"lens_last_vs_final_maxdiff_{split}"] = round(float(np.abs(lens[:, -1] - probs).max()), 5)
        sig[split] = signals(probs, lens)
        ys[split] = (probs.argmax(1) == np.array([it["gold"] for it in items])).astype(int)
        res[f"turn1_acc_{split}"] = round(float(ys[split].mean()), 4)
        if split == "test":
            res["mean_lens_correct_vs_wrong"] = {  # P(final letter) per layer, for the curve
                k: [round(float(x), 4) for x in np.take_along_axis(lens[ys[split] == v], probs.argmax(1)[ys[split] == v][:, None, None]
                                                                  .repeat(lens.shape[1], 1), 2)[..., 0].mean(0)]
                for k, v in (("correct", 1), ("wrong", 0))}
    res["time_s"] = round(time.perf_counter() - t0, 1)

    names = list(sig["test"])
    y = ys["test"]
    res["auc_eval"] = {k: round(auc(sig["test"][k], y), 4) for k in names}
    res["auc_fit"] = {k: round(auc(sig["diag"][k], ys["diag"]), 4) for k in names}

    def fit(cols):
        X = np.stack([sig["diag"][c] for c in cols], 1)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, ys["diag"])
        Xt = np.stack([sig["test"][c] for c in cols], 1)
        return m.decision_function((Xt - mu) / sd), dict(zip(cols, np.round(m.coef_[0], 3).tolist()))

    comb, coef = fit(names)
    mm, _ = fit(["maxprob", "margin"])
    res["auc_eval"]["combined"] = round(auc(comb, y), 4)
    res["auc_eval"]["maxprob+margin"] = round(auc(mm, y), 4)
    res["combined_coef"] = coef
    res["vs_maxprob"] = {k: boot_diff(s, sig["test"]["maxprob"], y) for k, s in
                         [*((n_, sig["test"][n_]) for n_ in names if n_ != "maxprob"), ("combined", comb), ("maxprob+margin", mm)]}
    res["combined_vs_final_only"] = boot_diff(comb, mm, y)
    print(json.dumps({k: res[k] for k in ("turn1_acc_diag", "turn1_acc_test", "auc_eval", "vs_maxprob",
                                         "combined_vs_final_only", "combined_coef", "time_s")}, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
