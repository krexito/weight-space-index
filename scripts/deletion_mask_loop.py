"""Can a learned DELETION MASK fix the reasoning loops of Qwen3.5-0.8B, from the base model's own outputs only?

Deletion Tuning (Lu et al., arXiv 2611.10866) freezes the model and learns which gate_proj weights to zero. Here it
is used against the loop pathology, with no information from the MIT CSAIL p50c release:
  * data      the base model reasons on GSM8K TRAIN problems; its responses that are correct and never loop become
              SFT targets (rejection sampling / self-distillation: the model is taught to behave like its own good runs)
  * mask      one logit G per gate_proj weight; training forward uses W * sigmoid(alpha G), alpha annealed so the mask
              becomes binary; init from Wanda scores (|W| * ||x||: the lowest eta fraction starts deleted)
  * export    hard mask 1[G > 0] written into a copy of the base checkpoint (deleted weights = 0), for vLLM
  * report    GSM8K TEST: base vs mask always on vs mask via MoI (only after the loop router fires) vs p50c,
              plus the exam: where did the mask delete, and do those neurons overlap p50c's layers 15 / 19 columns?

Prepare / generation / MoI planning reuse discover_loop_fix.py (same folder).
Usage: see notebooks/deletion_mask_loop_colab.ipynb.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402


def find_layers(model):
    import torch

    return next(m for n, m in model.named_modules()
                if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 2)


# ------------------------------------------------------------------ SFT data from the base model's good runs
def cmd_sft_data(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"])
    eos = set(cfg["eos"])
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    end_id = end_id if isinstance(end_id, int) and end_id != tok.unk_token_id else cfg["eos"][0]
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_elicit.jsonl")}
    s = dl.score(work, "requests_elicit.jsonl", "outputs_elicit.jsonl", tok)
    good = {r["id"] for r in s["rows"] if r["correct"] and not r["looped_killed"] and not r["truncated"]}
    rows, skipped = [], 0
    for o in lr.read_jsonl(work / "outputs_elicit.jsonl"):
        if o["id"] not in good:
            continue
        body = lr.strip_eos(o["token_ids"], eos) + [end_id]
        prompt = reqs[o["id"]]["prompt_ids"]
        if len(prompt) + len(body) > a.max_len:
            skipped += 1
            continue
        rows.append({"id": o["id"], "prompt_ids": prompt, "target_ids": body})
    lr.write_jsonl(work / "sft.jsonl", rows)
    cfg["sft"] = {"elicit": s["n"], "elicit_accuracy": s["accuracy"], "elicit_loops": s["loop_killed"],
                  "examples": len(rows), "skipped_too_long": skipped,
                  "target_tokens": sum(len(r["target_ids"]) for r in rows)}
    dl.cfg_save(work, cfg)
    print(f"elicit: {s['n']} responses, acc {s['accuracy']:.1%}, loops {s['loop_killed']:.1%} -> "
          f"{len(rows)} SFT examples ({cfg['sft']['target_tokens']:,} target tokens, {skipped} too long)")
    if len(rows) < 50:
        print("WARNING: few examples; raise --n-elicit in prepare")


# ------------------------------------------------------------------ train the deletion mask
def cmd_train(a) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    data = lr.read_jsonl(work / "sft.jsonl")
    if not data:
        raise SystemExit("no SFT examples")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["base_dir"], dtype=dtype).to(dev)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    gates = [lyr.mlp.gate_proj for lyr in layers]
    body_model = getattr(model, "model", model)
    head = model.get_output_embeddings()

    # ---- Wanda init: score |W_ij| * ||x_j||_2 on a few SFT sequences; the lowest eta fraction starts deleted
    sq = [torch.zeros(g.in_features, device=dev, dtype=torch.float64) for g in gates]

    def calib_hook(i):
        def f(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            sq[i] += (x * x).sum(0).double()
        return f

    hs = [g.register_forward_pre_hook(calib_hook(i)) for i, g in enumerate(gates)]
    with torch.no_grad():
        for ex in data[: a.calib_seqs]:
            body_model(input_ids=torch.tensor([ex["prompt_ids"] + ex["target_ids"]], device=dev))
    for h in hs:
        h.remove()

    state = {"alpha": a.alpha_start}

    class MaskedGate(nn.Module):
        def __init__(self, lin: nn.Linear, logits: torch.Tensor):
            super().__init__()
            self.lin = lin
            self.G = nn.Parameter(logits)

        def forward(self, x):
            m = torch.sigmoid(state["alpha"] * self.G).to(self.lin.weight.dtype)
            return F.linear(x, self.lin.weight * m, self.lin.bias)

    masked = []
    n_init_del = 0
    for i, (lyr, g) in enumerate(zip(layers, gates)):
        score = g.weight.detach().float().abs() * sq[i].sqrt().float()[None, :]
        k = max(1, int(a.eta * score.numel()))
        thr = score.flatten().kthvalue(k).values
        sign = torch.where(score <= thr, -1.0, 1.0)
        logits = sign * a.init_mean + a.init_std * torch.randn_like(score)
        n_init_del += int((logits <= 0).sum())
        mg = MaskedGate(g, logits.float())
        lyr.mlp.gate_proj = mg
        masked.append(mg)
    n_all = sum(p.numel() for n, p in model.named_parameters() if not n.endswith(".G"))
    n_gate = sum(m.G.numel() for m in masked)
    print(f"{len(masked)} gate_proj masked ({n_gate:,} logits); init deletes {n_init_del:,} "
          f"({n_init_del / n_gate:.3%} of gate, {n_init_del / n_all:.4%} of all weights)", flush=True)
    if a.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    opt = torch.optim.Adam([m.G for m in masked], lr=a.lr)
    order = [i for _ in range(a.epochs) for i in random.sample(range(len(data)), len(data))]
    steps = math.ceil(len(order) / a.accum)
    log = []
    t0 = time.perf_counter()
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        state["alpha"] = a.alpha_start + (a.alpha_end - a.alpha_start) * frac
        batch = order[step * a.accum : (step + 1) * a.accum]
        ntgt = sum(len(data[j]["target_ids"]) for j in batch)
        tot = 0.0
        for j in batch:
            ex = data[j]
            ids = torch.tensor([ex["prompt_ids"] + ex["target_ids"]], device=dev)
            P = len(ex["prompt_ids"])
            h = body_model(input_ids=ids).last_hidden_state[0]  # [T, D]
            hd = h.detach().requires_grad_(True)
            # chunked LM loss over the target tokens: the full [T, vocab] logits never exist at once
            for c0 in range(P - 1, ids.shape[1] - 1, a.chunk):
                c1 = min(c0 + a.chunk, ids.shape[1] - 1)
                logits = head(hd[c0:c1]).float()
                loss = F.cross_entropy(logits, ids[0, c0 + 1 : c1 + 1], reduction="sum") / ntgt
                loss.backward()
                tot += float(loss.detach())
            h.backward(hd.grad)
        if a.lam > 0:
            reg = 0.0
            for m in masked:
                w = m.lin.weight.float() * torch.sigmoid(state["alpha"] * m.G)
                reg = reg - a.lam * (w * w).sum()
            reg.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % a.log_every == 0 or step == steps - 1:
            ndel = sum(int((m.G <= 0).sum()) for m in masked)
            log.append({"step": step, "loss": round(tot, 4), "alpha": round(state["alpha"], 1), "deleted": ndel})
            print(f"  step {step + 1}/{steps}  loss/token {tot:.4f}  alpha {state['alpha']:.0f}  "
                  f"deleted {ndel:,} ({ndel / n_all:.4%} of all)  {time.perf_counter() - t0:.0f}s", flush=True)

    # ---- hard mask -> stats + checkpoint
    hard = [(m.G.detach() > 0).cpu() for m in masked]
    per_layer = [int((~h).sum()) for h in hard]
    per_neuron = {str(i): (~h).sum(1).tolist() for i, h in enumerate(hard)}  # deletions per intermediate neuron (row)
    np.savez_compressed(work / "deletion_mask.npz", **{f"keep{i}": np.packbits(h.numpy()) for i, h in enumerate(hard)},
                        shapes=np.array([list(h.shape) for h in hard]))
    total = sum(per_layer)
    stats = {"deleted_total": total, "deleted_frac_all_weights": total / n_all, "deleted_frac_gate": total / n_gate,
             "deleted_per_layer": per_layer, "layer_rank_by_deletions": [int(x) for x in np.argsort(per_layer)[::-1]],
             "train_log": log, "examples": len(data), "steps": steps,
             "hparams": {k: getattr(a, k) for k in ("lr", "eta", "alpha_start", "alpha_end", "lam", "epochs", "accum")}}
    (work / "mask_stats.json").write_text(json.dumps({**stats, "deleted_per_neuron": per_neuron}))
    cfg["mask"] = stats
    dl.cfg_save(work, cfg)
    print(f"hard mask: {total:,} deleted ({total / n_all:.4%} of all weights); per layer: {per_layer}")
    export(work, cfg, hard)


def export(work: Path, cfg: dict, hard) -> None:
    import re

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = Path(cfg["base_dir"])
    out = work / "qwen-mask"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns(".cache", "*.lock"))
    # the text decoder's gate_proj keys: group "<prefix>layers.N.mlp.gate_proj.weight" by prefix and keep the prefix
    # with exactly one key per masked layer (skips vision towers and MTP heads that also have "layers.N")
    pat = re.compile(r"^(.*?)layers\.(\d+)\.mlp\.gate_proj\.weight$")
    files = sorted(base.glob("*.safetensors"))
    by_prefix: dict[str, set[int]] = {}
    for f in files:
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():
                mt = pat.match(k)
                if mt:
                    by_prefix.setdefault(mt.group(1), set()).add(int(mt.group(2)))
    prefix = [p for p, s in by_prefix.items() if s == set(range(len(hard)))]
    if len(prefix) != 1:
        raise SystemExit(f"cannot identify the decoder gate_proj keys: {by_prefix.keys()}")
    prefix = prefix[0]
    done = 0
    for f in files:
        with safe_open(str(f), "pt") as fh:
            keys = list(fh.keys())
            hits = {}
            for k in keys:
                mt = pat.match(k)
                if mt and mt.group(1) == prefix:
                    hits[k] = int(mt.group(2))
            if not hits:
                continue
            tensors = {k: fh.get_tensor(k) for k in keys}
            meta = fh.metadata() or {"format": "pt"}
        for k, li in hits.items():
            tensors[k] = tensors[k] * hard[li].to(tensors[k].dtype)
            done += 1
        save_file(tensors, str(out / f.name), metadata=meta)
    if done != len(hard):
        raise SystemExit(f"exported {done} gate_proj tensors, expected {len(hard)}")
    print(f"exported masked checkpoint -> {out}")


# ------------------------------------------------------------------ report
def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"])
    eos, reqs = set(cfg["eos"]), {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    arms = {"base": "outputs_test_base.jsonl", "mask_fixed": "outputs_test_mask.jsonl", "p50c": "outputs_test_p50c.jsonl"}
    res: dict = {"sft": cfg.get("sft"), "mask": cfg.get("mask"), "dev": cfg.get("dev"), "arms": {}}
    rows = {}
    for name, f in arms.items():
        if (work / f).exists():
            s = dl.score(work, "requests_test.jsonl", f, tok)
            rows[name] = {r["id"]: r for r in s["rows"]}
            res["arms"][name] = {k: v for k, v in s.items() if k != "rows"}
    plan = {r["id"]: r for r in lr.read_jsonl(work / "requests_test_moi.jsonl")}
    cont = {o["id"]: o for o in lr.read_jsonl(work / "outputs_test_moi.jsonl")} if plan else {}
    moi, edit_tok, all_tok = {}, 0, 0
    for i in reqs:
        if i in cont:
            pre = plan[i]["prompt_ids"][len(reqs[i]["prompt_ids"]) :]
            g = lr.grade(pre + cont[i]["token_ids"], cont[i]["finish_reason"], reqs[i]["gold"], tok, eos,
                         reqs[i]["max_tokens"])
            edit_tok += max(g["tokens"] - len(pre), 0)
        else:
            g = rows["base"][i]
        all_tok += g["tokens"]
        moi[i] = g
    n = len(moi)
    res["arms"]["mask_moi"] = {"n": n, "accuracy": sum(g["correct"] for g in moi.values()) / n,
                               "loop_killed": sum(g["looped_killed"] for g in moi.values()) / n,
                               "truncated": sum(g["truncated"] for g in moi.values()) / n,
                               "edit_token_share": edit_tok / max(all_tok, 1), "router_fired": len(plan) / n}
    rows["mask_moi"] = moi
    ids = sorted(reqs)
    acc = {k: [bool(rows[k][i]["correct"]) for i in ids] for k in rows}
    res["paired"] = {f"{x}_vs_{y}": lr.paired_ci(acc[x], acc[y])
                     for x, y in (("mask_fixed", "base"), ("mask_moi", "base"), ("mask_moi", "mask_fixed"),
                                  ("mask_fixed", "p50c")) if x in acc and y in acc}

    # ---- exam against p50c (only now): where did the mask delete?
    ms = json.loads((work / "mask_stats.json").read_text())
    rank = ms["layer_rank_by_deletions"]
    exam: dict = {"p50c_layers": [15, 19], "top_layers_by_deletions": rank[:5],
                  "rank_of_15_and_19": {str(l): rank.index(l) + 1 for l in (15, 19) if l in rank}}
    if a.p50c_dir:
        from safetensors import safe_open

        base = Path(cfg["base_dir"])
        for li in (15, 19):
            if str(li) not in ms["deleted_per_neuron"]:
                continue
            cols = None
            for f in sorted(Path(a.p50c_dir).glob("*.safetensors")):
                with safe_open(str(f), "pt") as fe, safe_open(str(base / f.name), "pt") as fb:
                    try:
                        k = dl.down_key(list(fe.keys()), li)
                    except SystemExit:
                        continue
                    dW = (fe.get_tensor(k).double() - fb.get_tensor(k).double()).numpy()
                    cols = set(np.where(np.abs(dW).sum(0) > 0)[0].tolist())
            if cols is None:
                continue
            per = np.array(ms["deleted_per_neuron"][str(li)])
            touched = set(np.where(per > 0)[0].tolist())
            top = set(np.argsort(per)[::-1][: len(cols)].tolist())
            exam[f"layer_{li}"] = {"p50c_neurons": len(cols), "mask_touched_neurons": len(touched),
                                   "overlap_touched": len(touched & cols),
                                   "overlap_top_by_deletions": f"{len(top & cols)}/{len(cols)}",
                                   "random_expectation_top": round(len(cols) ** 2 / len(per), 2)}
    res["exam_vs_p50c"] = exam
    res["timing"] = dl.timing_summary(work)
    (work / "results_deletion_mask.json").write_text(json.dumps(res, indent=2))

    print(f"\n{'arm':11s} {'acc':>7s} {'loops':>7s} {'trunc':>7s}")
    for k, v in res["arms"].items():
        print(f"{k:11s} {v['accuracy']:7.1%} {v['loop_killed']:7.1%} {v['truncated']:7.1%}"
              + (f"   edit on {v['edit_token_share']:.0%} of tokens" if "edit_token_share" in v else ""))
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
    print("exam vs p50c:", json.dumps(exam, indent=1))
    print("Saved", work / "results_deletion_mask.json")


def cmd_export(a) -> None:
    """Re-create the masked checkpoint from deletion_mask.npz (e.g. after a Colab runtime reset)."""
    import torch

    work = Path(a.work)
    z = np.load(work / "deletion_mask.npz")
    shapes = z["shapes"]
    hard = [torch.from_numpy(np.unpackbits(z[f"keep{i}"])[: int(r) * int(c)].reshape(int(r), int(c)).astype(bool))
            for i, (r, c) in enumerate(shapes)]
    export(work, dl.cfg_load(work), hard)


def cmd_dev_report(a) -> None:
    work = Path(a.work)
    cfg = dl.cfg_load(work)
    out = {}
    for name in ("base", "mask"):
        s = dl.score(work, "requests_dev.jsonl", f"outputs_dev_{name}.jsonl")
        out[name] = {k: v for k, v in s.items() if k != "rows"}
        print(f"  dev {name:5s}: acc {s['accuracy']:.1%}  loops {s['loop_killed']:.1%}  trunc {s['truncated']:.1%}")
    cfg["dev"] = out
    dl.cfg_save(work, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sft-data")
    s.add_argument("--work", required=True)
    s.add_argument("--max-len", type=int, default=8192, help="prompt + response tokens; longer examples are skipped")
    t = sub.add_parser("train")
    t.add_argument("--work", required=True)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--eta", type=float, default=1 / 2048, help="fraction of gate weights deleted at init (Wanda)")
    t.add_argument("--init-mean", type=float, default=0.03)
    t.add_argument("--init-std", type=float, default=0.01)
    t.add_argument("--alpha-start", type=float, default=100.0)
    t.add_argument("--alpha-end", type=float, default=1000.0)
    t.add_argument("--lam", type=float, default=1e-5)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--chunk", type=int, default=1024)
    t.add_argument("--calib-seqs", type=int, default=32)
    t.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("export")
    e.add_argument("--work", required=True)
    d = sub.add_parser("dev-report")
    d.add_argument("--work", required=True)
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--p50c-dir")
    a = ap.parse_args()
    fns = {"sft-data": cmd_sft_data, "train": cmd_train, "export": cmd_export, "dev-report": cmd_dev_report,
           "report": cmd_report}
    dl.run_timed(a, fns, a.cmd if a.cmd in ("sft-data", "train", "export") else None)


if __name__ == "__main__":
    main()
