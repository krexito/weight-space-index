"""EXPERIMENTAL: which loop direction separates loop from normal tokens on MATH-500 (and GSM8K)? Saved answers only.

Question: our blind GSM8K patch is weak on MATH-500 while MIT's p50c is strong. Is our direction the problem? Score every
answer token with cos(down_proj output at layer l, u) for fixed directions, and compare with the best data-driven direction.

Directions (fixed before looking):
  mit_p50c_L15 / L19      MIT p50c edit directions (benchmarks/exp8_math/mit_patch_directions.npz)
  mit_p10c_L<l>           MIT p10c down_proj direction at each layer l in 8..22
  blind_gsm8k_L13 / L14   our discover_blind GSM8K patch (blind_directions.npz)
  dom_L<l>                diff of means (loop-region minus normal tokens) of the down_proj output at layer l, fitted on
                          half A of the SAME dataset, tested on half B: what a search on that dataset could find at best
  random_L<l>             one random unit direction per layer (control)
Labels per token as in the veto (experiment_router_calibration.load_answers): loop region = onset (kill - 768) to the
kill; normal = tokens of non-looping answers and tokens before onset - 256 of looping ones. Half A / half B split by
prompt (seed 0). Metrics on half B, 32-token moving average, orientation from half A: token AUC, and answer AUC (max
smoothed score over the loop region of looping answers vs over the whole of healthy answers).

  score    (GPU)  python scripts/experiment_direction_auc.py score   --work RUN --name math --model M --mit-dirs NPZ --blind NPZ --out S.npz
  analyze  (CPU)  python scripts/experiment_direction_auc.py analyze --scores math=S1.npz gsm8k=S2.npz --out R.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

for _m in ("fla", "causal_conv1d"):  # torch GDN path (validated fp32 recompute)
    sys.modules.setdefault(_m, None)

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import moi_loop_router_vllm as lr  # noqa: E402
from experiment_router_calibration import auc, load_answers, register_chunked_attention, smooth  # noqa: E402

LAYERS = tuple(range(8, 23))
RUN_SPLITS = {  # run name -> (requests, outputs) of the unpatched base model
    "math": (("requests_elicit.jsonl", "outputs_elicit.jsonl"), ("requests_dev.jsonl", "outputs_dev_base.jsonl"),
             ("requests_test.jsonl", "outputs_test_base.jsonl")),
    "gsm8k": (("requests_test.jsonl", "outputs_test_base.jsonl"),),
}


def load_run(work: Path, name: str, eos: set[int]) -> list[dict]:
    seqs = []
    for rq, op in RUN_SPLITS[name]:
        for q in load_answers(work, rq, op, eos):
            q["key"] = f"{op}:{q['id']}"
            g = hashlib.sha1(np.asarray(q["prompt"], np.int64).tobytes()).hexdigest()
            q["half"] = "A" if int(g[:8], 16) % 2 == 0 else "B"
            seqs.append(q)
    return seqs


class _Stop(Exception):
    pass


def cmd_score(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    eos = set(json.loads((work / "config.json").read_text())["eos"])
    seqs = load_run(work, a.name, eos)
    seqs = sorted(seqs, key=lambda q: len(q["prompt"]) + len(q["body"]), reverse=not a.limit)
    if a.limit:  # smoke test: shortest answers, half looping and half healthy, from both halves
        pick = []
        for lp in (True, False):
            for h in ("A", "B"):
                pick += [q for q in seqs if q["looping"] == lp and q["half"] == h][: max(1, a.limit // 4)]
        seqs = pick
    zm, zb = np.load(a.mit_dirs), np.load(a.blind)
    rng = np.random.default_rng(0)
    fixed: dict[int, dict[str, np.ndarray]] = {l: {} for l in LAYERS}
    fixed[15]["mit_p50c_L15"] = zm["p50c__L15.mlp.down_proj"]
    fixed[19]["mit_p50c_L19"] = zm["p50c__L19.mlp.down_proj"]
    for l in LAYERS:
        fixed[l][f"mit_p10c_L{l}"] = zm[f"p10c__L{l}.mlp.down_proj"]
        r = rng.standard_normal(1024)
        fixed[l][f"random_L{l}"] = r / np.linalg.norm(r)
    fixed[13]["blind_gsm8k_L13"] = zb["u13"] / np.linalg.norm(zb["u13"])
    fixed[14]["blind_gsm8k_L14"] = zb["u14"] / np.linalg.norm(zb["u14"])

    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32,
                                                 attn_implementation=register_chunked_attention(2048)).to(dev).eval()
    blocks = next(m for n, m in model.named_modules()
                  if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 2)
    backbone = getattr(model, "model", model)
    names = {l: sorted(fixed[l]) for l in LAYERS}
    D = {l: torch.from_numpy(np.stack([fixed[l][k] for k in names[l]])).float().to(dev) for l in LAYERS}

    def stop(*_):
        raise _Stop

    def forward(q):
        ys = {}
        hs = [lr.down_module(blocks[l]).register_forward_hook(
            lambda _m, _i, o, l=l: ys.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0])) for l in LAYERS]
        hs.append(blocks[max(LAYERS)].register_forward_hook(stop))
        try:
            with torch.no_grad():
                backbone(input_ids=torch.tensor([q["prompt"] + q["body"]], device=dev), use_cache=False)
        except _Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        p0 = len(q["prompt"])
        return {l: ys[l][p0:].float() for l in LAYERS}

    out: dict[str, np.ndarray] = {}
    sums = {c: {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in LAYERS} for c in ("loop", "normal")}
    counts = {"loop": 0, "normal": 0}
    t0 = time.time()
    # pass 1: fixed directions for every answer; class sums of half A for the diff-of-means directions
    for n, q in enumerate(seqs):
        y = forward(q)
        lab = torch.from_numpy(q["label"]).to(dev)
        for l in LAYERS:
            c = (y[l] @ D[l].T) / y[l].norm(dim=-1, keepdim=True).clamp_min(1e-6)
            for j, k in enumerate(names[l]):
                out[f"{q['key']}|{k}"] = c[:, j].cpu().numpy().astype(np.float32)
            if q["half"] == "A":
                for cname, v in (("loop", 1), ("normal", 0)):
                    m = lab == v
                    if m.any():
                        sums[cname][l] += y[l][m].double().sum(0)
        if q["half"] == "A":
            counts["loop"] += int((q["label"] == 1).sum())
            counts["normal"] += int((q["label"] == 0).sum())
        if n % 25 == 0 or n == len(seqs) - 1:
            print(f"[pass1 {n + 1}/{len(seqs)}] {time.time() - t0:.0f}s", flush=True)
    dom = {}
    for l in LAYERS:
        d = sums["loop"][l] / max(1, counts["loop"]) - sums["normal"][l] / max(1, counts["normal"])
        dom[l] = (d / d.norm()).float()
        out[f"__dom_dir_L{l}"] = dom[l].cpu().numpy()
    Dd = torch.stack([dom[l] for l in LAYERS])
    # pass 2: diff-of-means directions on half B only
    hb = [q for q in seqs if q["half"] == "B"]
    for n, q in enumerate(hb):
        y = forward(q)
        for i, l in enumerate(LAYERS):
            c = (y[l] @ Dd[i]) / y[l].norm(dim=-1).clamp_min(1e-6)
            out[f"{q['key']}|dom_L{l}"] = c.cpu().numpy().astype(np.float32)
        if n % 25 == 0 or n == len(hb) - 1:
            print(f"[pass2 {n + 1}/{len(hb)}] {time.time() - t0:.0f}s", flush=True)
    meta = [{"key": q["key"], "half": q["half"], "looping": q["looping"], "onset": q["onset"], "n": len(q["body"]),
             "label": q["label"].tolist()} for q in seqs]
    out["__meta"] = np.frombuffer(json.dumps({"run": a.name, "answers": meta, "dom_counts": counts}).encode(), np.uint8)
    tmp = a.out + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    Path(tmp).replace(a.out)
    print("saved", a.out, len(seqs), "answers", f"{time.time() - t0:.0f}s", "device", dev, flush=True)


def answer_stat(s: np.ndarray, q: dict) -> float:
    return float(s[q["onset"]:].max()) if q["looping"] else float(s.max())


def boot_ci(pos, neg, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)
    bs = [auc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def cmd_analyze(a) -> None:
    res = {}
    for spec in a.scores:
        run, path = spec.split("=", 1)
        z = np.load(path)
        meta = json.loads(bytes(z["__meta"]).decode())
        answers = meta["answers"]
        for q in answers:
            q["label"] = np.asarray(q["label"], np.int8)
        dirs = sorted({k.split("|", 1)[1] for k in z.files if "|" in k})
        A = [q for q in answers if q["half"] == "A"]
        B = [q for q in answers if q["half"] == "B"]
        rows = {}
        for d in dirs:
            def sm(q):
                return smooth(z[f"{q['key']}|{d}"])
            if d.startswith("dom_"):
                sign = 1.0  # loop minus normal by construction
            else:
                pa = np.concatenate([sm(q)[q["label"] == 1] for q in A])
                na = np.concatenate([sm(q)[q["label"] == 0] for q in A])
                sign = 1.0 if not (len(pa) and len(na)) or auc(pa, na) >= 0.5 else -1.0
            sb = {q["key"]: sign * sm(q) for q in B}
            pos = np.concatenate([sb[q["key"]][q["label"] == 1] for q in B])
            neg = np.concatenate([sb[q["key"]][q["label"] == 0] for q in B])
            ap = [answer_stat(sb[q["key"]], q) for q in B if q["looping"]]
            an = [answer_stat(sb[q["key"]], q) for q in B if not q["looping"]]
            rows[d] = {"token_auc": round(auc(pos, neg), 4), "answer_auc": round(auc(ap, an), 4) if an and ap else None,
                       "answer_auc_ci95": boot_ci(ap, an) if len(an) > 2 and len(ap) > 2 else None, "sign": sign}
        res[run] = {"half_B": {"answers": len(B), "loops": sum(q["looping"] for q in B),
                               "healthy": sum(not q["looping"] for q in B)},
                    "half_A": {"answers": len(A), "loops": sum(q["looping"] for q in A)},
                    "directions": rows}
        print(f"== {run}: half B {len(B)} answers ({res[run]['half_B']['loops']} loops, {res[run]['half_B']['healthy']} healthy)")
        for d in sorted(rows, key=lambda k: -rows[k]["token_auc"]):
            print(f"  {d:18s} token {rows[d]['token_auc']:.3f}  answer {rows[d]['answer_auc']}  {rows[d]['answer_auc_ci95']}")
    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1))
    Path(a.out + ".tmp").replace(a.out)
    print("saved", a.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--work", required=True)
    s.add_argument("--name", required=True, choices=sorted(RUN_SPLITS))
    s.add_argument("--model", required=True)
    s.add_argument("--mit-dirs", required=True)
    s.add_argument("--blind", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--cpu", action="store_true")
    n = sub.add_parser("analyze")
    n.add_argument("--scores", nargs="+", required=True, help="run=scores.npz")
    n.add_argument("--out", required=True)
    a = ap.parse_args()
    {"score": cmd_score, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
