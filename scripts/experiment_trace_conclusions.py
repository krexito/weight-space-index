"""EXPERIMENTAL (CPU): two ways past the ~7.5-bit limit of residual traces (see experiment_trace_lowbit.py).

A. Recompute instead of store: keep only the tokens and re-run the forward.
   Is it exact (bitwise, same machine, other thread count)? How far is a bf16 recompute (a stand-in for other
   hardware) from fp32, compared with INT8's error? What does it cost in time and bytes?
B. Lower the bar: do the conclusions an analysis draws change with 3-4 bit residuals instead of INT8?
   1. loop onset at the tracer's loop layer (L15) and at the best layers, in tokens
   2. loop direction per layer (mean loop tokens - mean pre-loop generated tokens), fitted on seq 101:
      cosine to the fp32 direction, and the layer ranking by held-out AUC on seq 102
   3. logit lens: top-1 token at layers 12-23 (norm -> lm_head) vs fp32
   4. two-sequence delta curve per layer (relative residual difference 101 vs 102 on shared length): Pearson r
Reference everywhere: fp32 residuals. Qwen3.5-0.8B, the same 3 saved generations.
Run (CPU, ~5 min): CUDA_VISIBLE_DEVICES="" python scripts/experiment_trace_conclusions.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(R))
import experiment_trace_lowbit as LB  # noqa: E402
from wsi.tracer import hadamard_transform  # noqa: E402

OUT = R / "benchmarks/trace_quant_sweep/results_trace_conclusions_cpu.json"
CODECS = ["int8", "int4", "lm4", "tcq3", "lm3"]
LENS = range(12, 24)


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    s = torch.cat([pos, neg])
    r = s.argsort().argsort().float() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def loop_dir(st, p_len, ons):
    """per layer: unit(mean loop tokens - mean pre-loop generated tokens)."""
    out = []
    for x in st:
        d = x[ons:].mean(0) - x[p_len:ons].mean(0)
        out.append(d / d.norm())
    return out


def lens_top1(model, x):
    norm, head = LB._find_final_norm_module(model), model.get_output_embeddings()
    with torch.no_grad():
        return torch.cat([head(norm(x[i:i + 128])).argmax(-1) for i in range(0, x.shape[0], 128)])


def main():
    from transformers import AutoModelForCausalLM
    seqs = torch.load(R / "benchmarks/layer_flow/loop_seqs_cpu.pt")
    model = AutoModelForCausalLM.from_pretrained(str(R / "models/qwen35-0.8b-full"), dtype=torch.float32).eval()
    res = {"model": "Qwen3.5-0.8B", "seqs": {k: {"T": int(v["ids"].numel()), "prompt_len": v["prompt_len"], "onset_ngram": v["onset"]}
                                             for k, v in seqs.items()}}

    # ---------------- A. recompute ----------------
    A = {}
    ids = seqs[101]["ids"]
    t = time.perf_counter()
    ref = LB.states(model, ids)
    A["fp32_forward_s"] = round(time.perf_counter() - t, 2)
    again = LB.states(model, ids)
    A["same_run_bitwise_equal_layers"] = sum(bool(torch.equal(a, b)) for a, b in zip(ref, again))
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    t = time.perf_counter()
    one = LB.states(model, ids)
    A["fp32_1thread_s"] = round(time.perf_counter() - t, 2)
    torch.set_num_threads(nt)
    A["threads"] = [nt, 1]
    A["other_threads_bitwise_equal_layers"] = sum(bool(torch.equal(a, b)) for a, b in zip(ref, one))
    A["other_threads_max_rel_l2"] = max(float((a - b).norm() / a.norm()) for a, b in zip(ref, one))
    mb = model.to(torch.bfloat16)
    t = time.perf_counter()
    bf = [x.float() for x in LB.states(mb, ids)]
    A["bf16_forward_s"] = round(time.perf_counter() - t, 2)
    del mb  # .to() was in place: reload clean fp32 weights
    model = AutoModelForCausalLM.from_pretrained(str(R / "models/qwen35-0.8b-full"), dtype=torch.float32).eval()
    assert torch.equal(LB.states(model, ids)[-1], ref[-1])
    rel = [float((a - b).norm() / a.norm()) for a, b in zip(ref, bf)]
    A["bf16_vs_fp32_rel_l2_mean"], A["bf16_vs_fp32_rel_l2_max"] = round(sum(rel) / len(rel), 5), round(max(rel), 5)
    A["bf16_onset_moved_layers"] = sum(LB.onset(a)[0] != LB.onset(b)[0] for a, b in zip(ref, bf))
    T, L, d = len(ids), len(ref), ref[0].shape[1]
    A["bytes_tokens_int32"] = T * 4
    A["bytes_int8_resid_trace"] = int(T * L * d * LB.BPE["int8"] / 8)
    print("A", json.dumps(A), flush=True)
    res["recompute"] = A

    # ---------------- B. conclusions ----------------
    st = {sid: (ref if sid == 101 else LB.states(model, seqs[sid]["ids"])) for sid in (101, 102, 104)}
    fit = torch.cat([hadamard_transform(s, LB.HB).reshape(-1, LB.G) for s in st[104][::3]])
    fit = fit[torch.randperm(fit.shape[0], generator=torch.Generator().manual_seed(3))[:40_000]]
    trellis = {"tcq3": LB.Trellis(9, 3)}
    trellis["tcq3"].fit(fit)
    LB.BPE["tcq3"] = trellis["tcq3"].bits_per_el()

    p1, o1 = seqs[101]["prompt_len"], seqs[101]["onset"]
    p2, o2 = seqs[102]["prompt_len"], seqs[102]["onset"]

    def conclusions(S):
        """S: {sid: [layer residuals]} -> what an analyst would read off."""
        dirs = loop_dir(S[101], p1, o1)
        aucs = []
        for l, dv in enumerate(dirs):
            pr = S[102][l] @ dv
            aucs.append(auc(pr[o2:], pr[p2:o2]))
        n = min(S[101][0].shape[0], S[102][0].shape[0])
        delta = [float((S[101][l][:n] - S[102][l][:n]).norm() / S[101][l][:n].norm()) for l in range(L)]
        return {"dirs": dirs, "auc": aucs, "rank": sorted(range(L), key=lambda l: -aucs[l]),
                "onset_L15": {sid: LB.onset(S[sid][15])[0] for sid in S},
                "lens": {sid: {l: lens_top1(model, S[sid][l]) for l in LENS} for sid in (101, 102)},
                "delta": delta}

    REF = conclusions(st)
    res["fp32"] = {"auc_by_layer": [round(a, 4) for a in REF["auc"]], "top5_layers": REF["rank"][:5],
                   "onset_L15": REF["onset_L15"], "delta_curve": [round(x, 4) for x in REF["delta"]]}
    print("fp32 top5", REF["rank"][:5], [round(REF["auc"][l], 3) for l in REF["rank"][:5]], "onset L15", REF["onset_L15"], flush=True)
    res["codecs"] = {}
    for c in CODECS:
        t = time.perf_counter()
        S = {sid: [LB.apply_codec(x, c, trellis) for x in st[sid]] for sid in st}
        C = conclusions(S)
        dcos = [float(a @ b) for a, b in zip(REF["dirs"], C["dirs"])]
        lens = [float((C["lens"][sid][l] == REF["lens"][sid][l]).float().mean()) for sid in (101, 102) for l in LENS]
        dr = torch.corrcoef(torch.tensor([REF["delta"], C["delta"]]))[0, 1]
        row = {"bits": round(LB.BPE[c], 3),
               "dir_cos_min": round(min(dcos), 5), "dir_cos_mean": round(sum(dcos) / L, 5),
               "auc_max_abs_diff": round(max(abs(a - b) for a, b in zip(REF["auc"], C["auc"])), 4),
               "top1_layer_same": C["rank"][0] == REF["rank"][0], "top3_layers": C["rank"][:3],
               "top3_same_set": set(C["rank"][:3]) == set(REF["rank"][:3]), "top5_overlap": len(set(C["rank"][:5]) & set(REF["rank"][:5])),
               "onset_L15": C["onset_L15"],
               "onset_L15_shift": {sid: (None if C["onset_L15"][sid] is None or REF["onset_L15"][sid] is None
                                         else C["onset_L15"][sid] - REF["onset_L15"][sid]) for sid in st},
               "lens_top1_agree_mean": round(sum(lens) / len(lens), 4), "lens_top1_agree_min": round(min(lens), 4),
               "delta_curve_r": round(float(dr), 5),
               "delta_curve_max_abs_diff": round(max(abs(a - b) for a, b in zip(REF["delta"], C["delta"])), 4),
               "s": round(time.perf_counter() - t, 1)}
        res["codecs"][c] = row
        print(c, json.dumps(row), flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
