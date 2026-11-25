"""EXPERIMENTAL (CPU): can residual traces go to ~2-3 bits with INT8 quality?

Real residuals: Qwen3.5-0.8B fp32 forward (teacher forcing) over 3 saved GSM8K generations (2 loop).
All codecs act on the same 64-wide Hadamard-rotated residual (the tracer's default rotation).

Codecs (bits/element include the fp16 scale per group of 64):
  int8, int4         uniform absmax, the tracer's own code (8.25 / 4.25 b/el)
  lm4, lm3, lm2      Lloyd-Max scalar levels for N(0,1), RMS scale per group (4.25 / 3.25 / 2.25)
  tcq3, tcq2         bitshift trellis (QTIP style, L-bit state, k bits shifted per element), LUT fitted
                     by Viterbi+Lloyd on a different sequence (k + (L-k)/64 + 0.25 b/el)
Pass bar: INT8-like fidelity AND loop onset unchanged vs fp32 at every layer.
Also: rank needed per layer (SVD) and the cross-layer delta size, i.e. how much any codec could exploit.
Result: negative, see research/findings.md "Low-bit trace residuals".
Run (CPU, ~5 min): CUDA_VISIBLE_DEVICES="" python scripts/experiment_trace_lowbit.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _decode_resid, _encode_resid, _find_final_norm_module, hadamard_transform  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))
G, HB = 64, 64
OUT = R / "benchmarks/trace_quant_sweep/results_lowbit_trellis_cpu.json"


# ---------------- codecs on rotated, flattened groups [N, 64] ----------------
def lloyd_max_levels(bits: int, iters: int = 200) -> torch.Tensor:
    n = 1 << bits
    x = torch.randn(2_000_000, generator=torch.Generator().manual_seed(1))
    lv = torch.quantile(x, (torch.arange(n) + 0.5) / n)
    for _ in range(iters):
        idx = (x[:, None] - lv[None]).abs().argmin(1)
        lv = torch.stack([x[idx == i].mean() for i in range(n)])
    return lv


LM = {b: lloyd_max_levels(b) for b in (2, 3, 4)}


def rms_scale(g: torch.Tensor) -> torch.Tensor:
    return g.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8).half().float()


def lm_codec(g: torch.Tensor, bits: int) -> torch.Tensor:
    s = rms_scale(g)
    lv = LM[bits]
    out = torch.empty_like(g)
    for i in range(0, g.shape[0], 200_000):
        z = g[i:i + 200_000] / s[i:i + 200_000]
        out[i:i + 200_000] = lv[(z[..., None] - lv).abs().argmin(-1)] * s[i:i + 200_000]
    return out


class Trellis:
    """Bitshift trellis: state = last L bits; each element shifts in k new bits; value = lut[state]."""

    def __init__(self, L: int, k: int):
        self.L, self.k, self.S = L, k, 1 << L
        self.lut = torch.randn(self.S, generator=torch.Generator().manual_seed(2))
        # predecessor of state s with shifted-out top bits p: ((s >> k) | (p << (L-k)))
        s = torch.arange(self.S)
        self.pred = torch.stack([(s >> k) | (p << (L - k)) for p in range(1 << k)], 1)  # [S, 2^k]

    def bits_per_el(self) -> float:
        return self.k + (self.L - self.k) / G + 16 / G

    def viterbi(self, z: torch.Tensor) -> torch.Tensor:
        """z [N, T] normalized -> best state path [N, T]."""
        N, T = z.shape
        cost = (z[:, 0:1] - self.lut[None]) ** 2  # any start state
        back = torch.empty(T, N, self.S, dtype=torch.uint8)
        for t in range(1, T):
            cand = cost[:, self.pred]  # [N, S, 2^k]
            best, arg = cand.min(-1)
            back[t] = arg.to(torch.uint8)
            cost = best + (z[:, t:t + 1] - self.lut[None]) ** 2
        path = torch.empty(N, T, dtype=torch.long)
        path[:, -1] = cost.argmin(1)
        for t in range(T - 1, 0, -1):
            path[:, t - 1] = self.pred[path[:, t], back[t].gather(1, path[:, t:t + 1]).squeeze(1).long()]
        return path

    def encode_decode(self, g: torch.Tensor, chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
        s = rms_scale(g)
        out = torch.empty_like(g)
        paths = torch.empty(g.shape, dtype=torch.long)
        for i in range(0, g.shape[0], chunk):
            p = self.viterbi(g[i:i + chunk] / s[i:i + chunk])
            paths[i:i + chunk] = p
            out[i:i + chunk] = self.lut[p] * s[i:i + chunk]
        return out, paths

    def fit(self, g: torch.Tensor, iters: int = 6) -> None:
        z = g / rms_scale(g)
        for it in range(iters):
            _, paths = self.encode_decode(g)
            sums = torch.zeros(self.S).index_add_(0, paths.reshape(-1), z.reshape(-1))
            cnt = torch.zeros(self.S).index_add_(0, paths.reshape(-1), torch.ones(z.numel()))
            used = cnt > 0
            self.lut[used] = sums[used] / cnt[used]
            mse = float(((self.lut[paths] - z) ** 2).mean())
            print(f"  fit L{self.L}k{self.k} iter {it}: mse {mse:.4f} (states used {int(used.sum())}/{self.S})", flush=True)


def apply_codec(x: torch.Tensor, name: str, trellis: dict) -> torch.Tensor:
    """x [T, d] residual -> reconstructed [T, d]."""
    if name in ("int8", "int4"):
        return _decode_resid(_encode_resid(x, name, G, HB), name, G, HB)
    xw = hadamard_transform(x, HB)
    g = xw.reshape(-1, G)
    if name.startswith("lm"):
        gh = lm_codec(g, int(name[2:]))
    else:
        gh, _ = trellis[name].encode_decode(g)
    return hadamard_transform(gh.reshape(x.shape), HB)


BPE = {"int8": 8 + 16 / G, "int4": 4 + 16 / G, "lm4": 4 + 16 / G, "lm3": 3 + 16 / G, "lm2": 2 + 16 / G}


# ---------------- model states ----------------
def states(model, ids):
    _, blocks = find_blocks(model)
    acts = {}
    hs = [b.register_forward_hook(lambda m, a, o, l=l: acts.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].detach()))
          for l, b in enumerate(blocks)]
    with torch.no_grad():
        model(input_ids=ids[None], use_cache=False)
    for h in hs:
        h.remove()
    return [acts[l] for l in range(len(blocks))]


def logits_metrics(model, h_true, h_hat):
    norm, head = _find_final_norm_module(model), model.get_output_embeddings()
    kl = t1 = 0.0
    with torch.no_grad():
        for i in range(0, h_true.shape[0], 64):
            lp = torch.log_softmax(head(norm(h_true[i:i + 64])).float(), -1)
            lq = torch.log_softmax(head(norm(h_hat[i:i + 64])).float(), -1)
            kl += float((lp.exp() * (lp - lq)).sum())
            t1 += float((lp.argmax(-1) == lq.argmax(-1)).sum())
    n = h_true.shape[0]
    return kl / n, t1 / n


def onset(R: torch.Tensor, window: int = 16, thr: float = 0.96):
    Rn = R / R.norm(dim=1, keepdim=True).clamp_min(1e-8)
    lag = torch.zeros(Rn.shape[0])
    for t in range(1, Rn.shape[0]):
        lag[t] = (Rn[max(0, t - window):t] @ Rn[t]).max()
    hits = (lag > thr).nonzero(as_tuple=True)[0]
    return (int(hits[0]) if len(hits) else None), lag


def main():
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(R / "models/qwen35-0.8b-full"), dtype=torch.float32).eval()
    seqs = torch.load(R / "benchmarks/layer_flow/loop_seqs_cpu.pt")
    print("model loaded", round(time.time() - t0, 1), "s", flush=True)

    # fit trellis LUTs on seq 104 (no loop), test on 101/102 + 104 reported separately
    fit_st = states(model, seqs[104]["ids"])
    fit_g = torch.cat([hadamard_transform(s, HB).reshape(-1, G) for s in fit_st[::3]])
    fit_g = fit_g[torch.randperm(fit_g.shape[0], generator=torch.Generator().manual_seed(3))[:40_000]]
    trellis = {"tcq3": Trellis(9, 3), "tcq2": Trellis(8, 2)}
    for name, tr in trellis.items():
        t1 = time.time()
        tr.fit(fit_g)
        BPE[name] = tr.bits_per_el()
        print(name, "fit", round(time.time() - t1, 1), "s", flush=True)
    codecs = ["int8", "int4", "lm4", "lm3", "tcq3", "lm2", "tcq2"]

    res = {"model": "Qwen3.5-0.8B", "group": G, "hadamard": HB, "bits_per_el": {c: round(BPE[c], 3) for c in codecs},
           "fit_seq": 104, "seqs": {}}
    for sid in (101, 102, 104):
        st = fit_st if sid == 104 else states(model, seqs[sid]["ids"])
        ref = [onset(s)[0] for s in st]
        row = {"onset_ref_per_layer": ref, "saved_onset_ngram": seqs[sid]["onset"], "codecs": {}}
        print(f"\nseq {sid}: T={st[0].shape[0]} ref onset L15={ref[15]}", flush=True)
        for c in codecs:
            t1 = time.time()
            cos, rel, ons, lagd = [], [], [], []
            last = None
            for l, x in enumerate(st):
                xh = apply_codec(x, c, trellis)
                f, h = x.reshape(-1), xh.reshape(-1)
                rel.append(float((f - h).norm() / f.norm()))
                cos.append(float(torch.nn.functional.cosine_similarity(x, xh, dim=1).mean()))
                o, lag = onset(xh)
                ons.append(o)
                lagd.append(float((lag - onset(x)[1]).abs().max()))
                last = xh
            kl, t1m = logits_metrics(model, st[-1], last)
            moved = [(l, ref[l], ons[l]) for l in range(len(st)) if ons[l] != ref[l]]
            row["codecs"][c] = {"mean_tok_cos": round(sum(cos) / len(cos), 6), "min_layer_tok_cos": round(min(cos), 6),
                                "mean_rel_l2": round(sum(rel) / len(rel), 5), "kl_last": round(kl, 6), "top1_last": round(t1m, 4),
                                "onset_layers_moved": len(moved), "onset_moved": moved[:8],
                                "max_lag_cos_diff": round(max(lagd), 5), "s": round(time.time() - t1, 1)}
            r = row["codecs"][c]
            print(f"  {c:5s} {BPE[c]:5.2f}b  cos {r['mean_tok_cos']:.6f} (min {r['min_layer_tok_cos']:.5f})  relL2 {r['mean_rel_l2']:.4f}  "
                  f"KL {kl:.2e} top1 {t1m:.4f}  onset moved {len(moved)}/24 {moved[:3]}  lagdiff {r['max_lag_cos_diff']:.4f}  {r['s']}s",
                  flush=True)
        if sid == 101:
            row["redundancy"] = {}
            for l in (0, 6, 12, 18, 23):
                sv = torch.linalg.svdvals(st[l]) ** 2
                c = sv.cumsum(0) / sv.sum()
                red = {f"rank_for_relL2_{r}": int((c < 1 - r * r).sum()) + 1 for r in (0.1, 0.03, 0.0056)}
                red["of"] = min(st[l].shape)
                if l:
                    red["delta_over_resid_norm"] = round(float((st[l] - st[l - 1]).norm() / st[l].norm()), 3)
                row["redundancy"][f"L{l}"] = red
            print("redundancy", row["redundancy"], flush=True)
        res["seqs"][str(sid)] = row
        OUT.write_text(json.dumps(res, indent=1))
    print("saved", OUT, "total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
