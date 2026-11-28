"""Recompute GPU vs CPU worker (one arm per process, so each arm picks its own GatedDeltaNet kernels).

states mode : python scripts/experiment_recompute_gpu_cpu.py states --arm NAME --device cpu|cuda --dtype fp32|fp16 --fla 0|1 --model DIR --seqs JSON --out PT
compare mode: python scripts/experiment_recompute_gpu_cpu.py compare --model DIR --seqs JSON --dir DIR_WITH_PTS --ref cpu_fp32 [--local-proj JSON] --out JSON
"""
import sys

ARGV = sys.argv[1:]
FLA = "--fla" in ARGV and ARGV[ARGV.index("--fla") + 1] == "1"
if not FLA:  # hide the GPU kernels: transformers then uses its torch implementation (CPU-safe)
    sys.modules["fla"] = None
    sys.modules["causal_conv1d"] = None

import argparse  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

INT8_REL_L2 = 0.0056  # tracer int8+had64 error on the same sequences (results_lowbit_trellis_cpu.json)


def blocks_of(model):
    return max((m for _, m in model.named_modules() if isinstance(m, torch.nn.ModuleList)), key=len)


def final_norm(model):
    c = [(n, m) for n, m in model.named_modules() if n.endswith(".norm") and ".layers." not in n and "visual" not in n]
    return c[-1][1]


def load(model_dir, device, dtype):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    return m


def states(model, ids, device):
    blocks = blocks_of(model)
    acts = {}
    hs = [b.register_forward_hook(lambda m, a, o, l=l: acts.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].detach().float().cpu()))
          for l, b in enumerate(blocks)]
    with torch.no_grad():
        model(input_ids=torch.tensor(ids, device=device)[None], use_cache=False)
    for h in hs:
        h.remove()
    return [acts[l] for l in range(len(blocks))]


def proj_dirs(d, k=4):
    P = torch.randn(d, k, generator=torch.Generator().manual_seed(0))
    return P / P.norm(dim=0, keepdim=True)


def onset(R, window=16, thr=0.96):
    Rn = R / R.norm(dim=1, keepdim=True).clamp_min(1e-8)
    lag = torch.zeros(Rn.shape[0])
    for t in range(1, Rn.shape[0]):
        lag[t] = (Rn[max(0, t - window):t] @ Rn[t]).max()
    hits = (lag > thr).nonzero(as_tuple=True)[0]
    return int(hits[0]) if len(hits) else None


def run_states(a):
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    dev = a.device
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    seqs = json.load(open(a.seqs))
    model = load(a.model, dev, dt)
    out = {"arm": a.arm, "device": dev, "dtype": a.dtype, "fla_requested": FLA,
           "fast_path": bool(mq.is_fast_path_available),
           "fla_chunk_kernel": mq.chunk_gated_delta_rule is not None,
           "causal_conv1d_kernel": mq.causal_conv1d_fn is not None,
           "torch": str(torch.__version__), "transformers": str(transformers.__version__),
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
           "sec": {}, "states": {}}
    if dev == "cuda":  # warm-up (kernel compile / autotune) outside the timing
        states(model, seqs["104"]["ids"][:64], dev)
    for sid, s in seqs.items():
        t = time.perf_counter()
        st = states(model, s["ids"], dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        out["sec"][sid] = round(time.perf_counter() - t, 3)
        out["states"][sid] = st
        print(a.arm, "seq", sid, "T", len(s["ids"]), out["sec"][sid], "s", "finite", all(bool(torch.isfinite(x).all()) for x in st), flush=True)
    torch.save(out, a.out)
    print("saved", a.out, "fast_path", out["fast_path"], "fla", out["fla_chunk_kernel"], "conv1d", out["causal_conv1d_kernel"], flush=True)


def local_proj(st101):
    P = proj_dirs(st101[0].shape[1])
    return torch.stack([x @ P for x in st101])  # [L, T, k]


def run_compare(a):
    seqs = json.load(open(a.seqs))
    arms = {}
    for f in sorted(os.listdir(a.dir)):
        if f.endswith(".pt"):
            d = torch.load(os.path.join(a.dir, f))
            arms[d["arm"]] = d
    ref = arms[a.ref]
    model = load(a.model, "cpu", torch.float32)
    norm, head = final_norm(model), model.get_output_embeddings()

    def logits_metrics(h_true, h_hat):
        kl = t1 = 0.0
        with torch.no_grad():
            for i in range(0, h_true.shape[0], 64):
                lp = torch.log_softmax(head(norm(h_true[i:i + 64])).float(), -1)
                lq = torch.log_softmax(head(norm(h_hat[i:i + 64])).float(), -1)
                kl += float((lp.exp() * (lp - lq)).sum())
                t1 += float((lp.argmax(-1) == lq.argmax(-1)).sum())
        return kl / h_true.shape[0], t1 / h_true.shape[0]

    ref_onsets = {sid: [onset(x) for x in ref["states"][sid]] for sid in seqs}
    res = {"ref_arm": a.ref, "int8_rel_l2_for_scale": INT8_REL_L2, "ref_onset_per_layer": ref_onsets, "arms": {}}
    if a.local_proj:
        lp = json.load(open(a.local_proj))
        L, T, k = lp["shape"]
        loc = torch.frombuffer(bytearray(base64.b64decode(lp["data"])), dtype=torch.float32).reshape(L, T, k)
        res["local_cpu"] = {kk: lp[kk] for kk in lp if kk not in ("data",)}
    for name, d in arms.items():
        row = {kk: d[kk] for kk in ("device", "dtype", "fast_path", "fla_chunk_kernel", "causal_conv1d_kernel", "torch",
                                     "transformers", "gpu", "cpu", "threads", "sec")}
        rel, bitwise, moved, finite = [], 0, 0, True
        per_seq = {}
        for sid in seqs:
            A, B = ref["states"][sid], d["states"][sid]
            r = [float((x - y).norm() / x.norm()) for x, y in zip(A, B)]
            rel += r
            bitwise += sum(bool(torch.equal(x, y)) for x, y in zip(A, B))
            ons = [onset(y) for y in B]
            mv = [(l, ref_onsets[sid][l], ons[l]) for l in range(len(B)) if ons[l] != ref_onsets[sid][l]]
            moved += len(mv)
            finite &= all(bool(torch.isfinite(y).all()) for y in B)
            kl, t1 = logits_metrics(A[-1], B[-1]) if finite else (None, None)
            per_seq[sid] = {"rel_l2_mean": round(sum(r) / len(r), 8), "rel_l2_max": round(max(r), 8),
                            "rel_l2_by_layer": [round(x, 8) for x in r], "onset_moved": mv[:6],
                            "kl_last": kl, "top1_last": t1}
        row.update({"layers_bitwise_equal": f"{bitwise}/{len(rel)}", "rel_l2_mean": round(sum(rel) / len(rel), 8),
                    "rel_l2_max": round(max(rel), 8), "x_int8_error": round(max(rel) / INT8_REL_L2, 4),
                    "onset_layers_moved": moved, "all_finite": finite, "per_seq": per_seq})
        if a.local_proj:
            pr = local_proj(d["states"]["101"])
            row["vs_local_cpu_proj_rel_l2_max"] = round(max(float((loc[l] - pr[l]).norm() / loc[l].norm()) for l in range(L)), 8)
        res["arms"][name] = row
        print(f"{name:16s} bitwise {row['layers_bitwise_equal']:>6s}  relL2 mean {row['rel_l2_mean']:.2e} max {row['rel_l2_max']:.2e} "
              f"({row['x_int8_error']:.3f}x int8)  onset moved {moved}  "
              f"top1 {[per_seq[s]['top1_last'] for s in seqs]}  vs local {row.get('vs_local_cpu_proj_rel_l2_max')}  sec {row['sec']}", flush=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print("saved", a.out)


def run_local_proj(a):
    """Local reference fingerprint: seq 101 on this CPU in fp32, projected on 4 seeded directions."""
    import hashlib
    seqs = json.load(open(a.seqs))
    model = load(a.model, "cpu", torch.float32)
    st = states(model, seqs["101"]["ids"], "cpu")
    pr = local_proj(st).contiguous()
    shas = []
    for f in sorted(os.listdir(a.model)):
        if f.endswith(".safetensors"):
            h = hashlib.sha256()
            with open(os.path.join(a.model, f), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            shas.append(h.hexdigest())
    import transformers
    out = {"shape": list(pr.shape), "torch": torch.__version__, "transformers": transformers.__version__,
           "cpu": platform.processor(), "threads": torch.get_num_threads(), "safetensors_sha256": shas,
           "data": base64.b64encode(pr.numpy().tobytes()).decode()}
    json.dump(out, open(a.out, "w"))
    print("saved", a.out, out["shape"], shas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["states", "compare", "local_proj"])
    ap.add_argument("--arm")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--fla", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir")
    ap.add_argument("--ref", default="cpu_fp32")
    ap.add_argument("--local-proj", dest="local_proj")
    a = ap.parse_args()
    {"states": run_states, "compare": run_compare, "local_proj": run_local_proj}[a.mode](a)
