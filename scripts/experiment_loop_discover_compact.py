"""Test 4: Can `CompactTrace` (resid + K1024 fp4 MLP + head_norms + attn_topk + cache) improve
loop circuit discovery (`discover_loop_fix` offline from traces) on `Qwen/Qwen3.5-0.8B`?

Runs in ~4-5 min on Colab L4:
  1. Generates `n_train` (24) GSM8K train solutions, identifies clean vs looping runs.
  2. Captures ONE `CompactTrace` per selected sequence (single forward pass each).
  3. Computes loop-vs-healthy Fisher separation OFFLINE across all 24 layers from:
       - MLP activations (`dequant_mlp_topk` -> W_down projection)
       - Residuals (`dequant_resid`)
       - Attention / Linear-Attn head L2 norms (`head_norms`) & `attn_topk` lag focus
  4. Grades candidate layers & top-64 neurons against `patches/qwen35_0.8b_p50c_loop_fix.moipatch`
     (blind exam: does `CompactTrace` recover Layer 15 AND Layer 19 better than MLP-only?)
  5. Evaluates held-out GSM8K accuracy & loop rate on `n_eval` (24) problems for:
     `base` vs `ct_mlp_only` vs `ct_hybrid (MLP+heads)` vs `p50c` reference.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsi.capture import find_mlps  # noqa: E402
from wsi.moipatch import Edit, MoIPatch  # noqa: E402
from wsi.tracer import capture_compact, diff_compact  # noqa: E402


HUB_FILES = {
    "gsm8k/train.parquet": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
    "gsm8k/test.parquet": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
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


def _local_or_hub(name_or_dir: str) -> str:
    p = Path(name_or_dir)
    if p.exists():
        return str(p)
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_dir)


def _extract_gold(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def _extract_pred(txt: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]+)\}", txt)
    if m:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", m[-1])
        if nums:
            return nums[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", txt)
    return nums[-1].replace(",", "") if nums else None


def _is_correct(txt: str, gold: str) -> bool:
    p = _extract_pred(txt)
    if p is None:
        return False
    try:
        return abs(float(p) - float(gold)) < 1e-4
    except ValueError:
        return p == gold


def _has_ngram_repeat(ids: list[int], n: int = 8, reps: int = 3) -> bool:
    if len(ids) < n * reps:
        return False
    seen: dict[tuple[int, ...], int] = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i : i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= reps:
            return True
    return False


def _loop_onset(ids: list[int], n: int = 16, window: int = 64, frac: float = 0.5) -> int | None:
    """First response position where >= `frac` of the next `window` tokens end an already-seen n-gram (the repo's
    router rule, moi_loop_router_vllm.py: 16-grams, 50% over 64 tokens). None = no loop. Truncation alone is not a loop."""
    if len(ids) < n + window:
        return None
    seen: set[tuple[int, ...]] = set()
    dup = [0] * len(ids)
    for i in range(n - 1, len(ids)):
        g = tuple(ids[i - n + 1 : i + 1])
        dup[i] = int(g in seen)
        seen.add(g)
    run = sum(dup[:window])
    for t in range(len(ids) - window + 1):
        if run >= frac * window:
            return t
        if t + window < len(ids):
            run += dup[t + window] - dup[t]
    return None


def _batched_generate(model, tok, questions: list[str], dev: torch.device, max_new_tokens: int) -> list[dict]:
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
        for q in questions
    ]
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    in_width = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    rows = []
    eos_id = tok.eos_token_id
    for i in range(len(questions)):
        pad_left = int((enc["attention_mask"][i] == 0).sum().item())
        p_ids = enc["input_ids"][i, pad_left:].tolist()
        out_raw = gen[i, in_width:].tolist()
        if eos_id is not None and eos_id in out_raw:
            out_ids = out_raw[: out_raw.index(eos_id)]
            hit_eos = True
        else:
            out_ids = out_raw
            hit_eos = False
        full_ids = p_ids + out_ids
        txt = tok.decode(out_ids, skip_special_tokens=True)
        onset = _loop_onset(out_ids)
        rows.append({
            "prompt_len": len(p_ids),
            "gen_len": len(out_ids),
            "full_ids": full_ids,
            "text": txt,
            "loop": onset is not None,
            "onset_resp": onset,  # response-relative token where the repetition starts
            "finished": hit_eos,
        })
    return rows


def _fisher_dir(x_healthy: torch.Tensor, x_loop: torch.Tensor) -> tuple[torch.Tensor, float]:
    mh, ml = x_healthy.mean(dim=0), x_loop.mean(dim=0)
    d = ml - mh
    dn = float(d.norm())
    if dn < 1e-8:
        return torch.zeros_like(d), 0.0
    u = d / dn
    ph, pl = x_healthy @ u, x_loop @ u
    vh = float(ph.var(unbiased=False)) if len(ph) > 1 else 0.0
    vl = float(pl.var(unbiased=False)) if len(pl) > 1 else 0.0
    score = float((pl.mean() - ph.mean()) ** 2 / max(0.5 * (vh + vl), 1e-8))
    return u, score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--patch", default=str(ROOT / "patches" / "qwen35_0.8b_p50c_loop_fix.moipatch"))
    ap.add_argument("--data-dir", default=str(ROOT / "work"))
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-traces", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--out", default="results_loop_discover_compact.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data_dir)
    mdir = _local_or_hub(args.model)
    tok = AutoTokenizer.from_pretrained(mdir)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype, attn_implementation="eager").to(dev).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype).to(dev).eval()

    gtr = pd.read_parquet(need(data, "gsm8k/train.parquet")).iloc[: args.n_train]
    gte = pd.read_parquet(need(data, "gsm8k/test.parquet")).iloc[: args.n_eval]

    # 1. Generate train pool in one batch
    t0 = time.perf_counter()
    tr_rows = _batched_generate(model, tok, list(gtr["question"]), dev, args.max_new_tokens)
    for r, a in zip(tr_rows, gtr["answer"]):
        r["gold"] = _extract_gold(a)
        r["correct"] = _is_correct(r["text"], r["gold"])
    train_gen_s = round(time.perf_counter() - t0, 1)
    loop_rows = [r for r in tr_rows if r["loop"]]
    clean_rows = [r for r in tr_rows if not r["loop"]]
    n_unfinished = sum(1 for r in clean_rows if not r["finished"])
    print(f"Train gen ({len(tr_rows)} seqs) in {train_gen_s}s: loop={len(loop_rows)} clean={len(clean_rows)} "
          f"(of which cut at max_new_tokens without repeating: {n_unfinished})", flush=True)
    if not loop_rows:
        raise SystemExit("No looping sequence in the train pool: raise --n-train or --max-new-tokens.")

    # 2. Capture CompactTrace once per selected sequence
    half = max(1, args.max_traces // 2)
    chosen = loop_rows[:half] + clean_rows[:half]
    if len(chosen) < min(args.max_traces, len(tr_rows)):
        extra = [r for r in tr_rows if r not in chosen][: max(0, args.max_traces - len(chosen))]
        chosen += extra

    traces = []
    t_cap = time.perf_counter()
    for r in chosen:
        ct = capture_compact(model, r["full_ids"])
        onset = ct.detect_loop_onset(layer=min(15, ct.n_layers - 1), window=16, thr=0.96)
        traces.append((r, ct, onset))
    cap_s = round(time.perf_counter() - t_cap, 2)
    bank_mb = round(sum(ct.size_bytes() for _, ct, _ in traces) / 2**20, 2)
    print(f"Captured {len(traces)} CompactTraces in {cap_s}s | total bank={bank_mb} MiB", flush=True)

    # 3. Offline extraction of healthy vs loop samples from CompactTrace ONLY
    mods = dict(model.named_modules())
    mlps = find_mlps(model)
    n_layers = len(mlps)
    w_downs = {}
    d_inter = 0
    for l, g in enumerate(mlps):
        w = mods[g.down].weight.detach().float().cpu()  # (D, U)
        w_downs[l] = w
        d_inter = w.shape[1]

    healthy_res: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    loop_res: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    healthy_y: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    loop_y: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    healthy_xsum: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    loop_xsum: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    healthy_head: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    loop_head: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    healthy_att_focus: dict[int, list[torch.Tensor]] = {}
    loop_att_focus: dict[int, list[torch.Tensor]] = {}

    for r, ct, onset in traces:
        T = ct.seq_len
        p = r["prompt_len"]
        if r["loop"]:
            ons = min(p + r["onset_resp"], T - 1)  # token-level onset, always inside the response
            h_start = min(p, T)
            h_end = max(h_start, min(ons, p + 96, T))
            h_idx = torch.arange(h_start, h_end)
            l_start = max(ons, T - 192)
            l_idx = torch.arange(l_start, T)
        else:
            h_idx = torch.arange(min(p, T), T)
            l_idx = torch.empty(0, dtype=torch.long)

        for l in range(n_layers):
            R = ct.dequant_resid(l)
            if len(h_idx):
                healthy_res[l].append(R[h_idx])
            if len(l_idx):
                loop_res[l].append(R[l_idx])

            idx_m, val_m = ct.dequant_mlp_topk(l)
            w_raw = w_downs[l]
            wn = w_raw.norm(dim=0).clamp_min(1e-12)
            if len(h_idx):
                ih, vh = idx_m[h_idx], val_m[h_idx]
                xh = torch.zeros((len(h_idx), d_inter), dtype=torch.float32)
                xh.scatter_(1, ih, vh / wn[ih])
                healthy_y[l].append(xh @ w_raw.T)
                healthy_xsum[l].append(xh.mean(dim=0, keepdim=True))
            if len(l_idx):
                il, vl = idx_m[l_idx], val_m[l_idx]
                xl = torch.zeros((len(l_idx), d_inter), dtype=torch.float32)
                xl.scatter_(1, il, vl / wn[il])
                loop_y[l].append(xl @ w_raw.T)
                loop_xsum[l].append(xl.mean(dim=0, keepdim=True))

        for mod, h in ct.head_norms.items():
            li = h["layer"]
            if li is None or li not in healthy_head:
                continue
            l2 = h["l2"].float()
            if len(h_idx):
                healthy_head[li].append(l2[h_idx])
            if len(l_idx):
                loop_head[li].append(l2[l_idx])

        for li in ct.attn_topk:
            _, a_w = ct.dequant_attn_topk(li)  # (H, T, k_att)
            # Concentration on top-2 non-BOS targets as a crisp per-head scalar
            top2_mass = a_w[:, :, : min(2, a_w.shape[-1])].sum(dim=-1).T  # (T, H)
            if len(h_idx):
                healthy_att_focus.setdefault(li, []).append(top2_mass[h_idx])
            if len(l_idx):
                loop_att_focus.setdefault(li, []).append(top2_mass[l_idx])

    # Rank layers across signals
    layer_stats = []
    u_mlp_by_layer = {}
    top64_by_layer = {}
    for l in range(n_layers):
        if not loop_y[l] or not healthy_y[l]:
            continue
        yh_t, yl_t = torch.cat(healthy_y[l], dim=0), torch.cat(loop_y[l], dim=0)
        rh_t, rl_t = torch.cat(healthy_res[l], dim=0), torch.cat(loop_res[l], dim=0)
        u_m, f_m = _fisher_dir(yh_t, yl_t)
        u_r, f_r = _fisher_dir(rh_t, rl_t)
        u_mlp_by_layer[l] = u_m

        # Select 64 neurons with highest directional excess write along u_m
        dx = torch.cat(loop_xsum[l], dim=0).mean(dim=0) - torch.cat(healthy_xsum[l], dim=0).mean(dim=0)
        proj_w = u_m @ w_downs[l]  # (U,)
        neuron_score = dx * proj_w
        top64_by_layer[l] = neuron_score.topk(64).indices.sort().values

        hh_t = torch.cat(healthy_head[l], dim=0)
        hl_t = torch.cat(loop_head[l], dim=0)
        d_head = (hl_t.mean(dim=0) - hh_t.mean(dim=0)).abs() / ((0.5 * (hl_t.var(dim=0, unbiased=False) + hh_t.var(dim=0, unbiased=False))).sqrt().clamp_min(1e-6))
        head_sep = float(d_head.max())

        attn_sep = 0.0
        if l in healthy_att_focus and l in loop_att_focus:
            ah_t = torch.cat(healthy_att_focus[l], dim=0)
            al_t = torch.cat(loop_att_focus[l], dim=0)
            da_t = (al_t.mean(dim=0) - ah_t.mean(dim=0)).abs() / ((0.5 * (al_t.var(dim=0, unbiased=False) + ah_t.var(dim=0, unbiased=False))).sqrt().clamp_min(1e-6))
            attn_sep = float(da_t.max())

        layer_stats.append({
            "layer": l,
            "is_full_attn": l in healthy_att_focus,
            "fisher_mlp": round(f_m, 4),
            "fisher_resid": round(f_r, 4),
            "head_max_d": round(head_sep, 4),
            "attn_focus_d": round(attn_sep, 4),
            "hybrid_score": round(f_m * (1.0 + 0.35 * head_sep + 0.5 * attn_sep), 4),
        })

    mlp_ranked = [r["layer"] for r in sorted(layer_stats, key=lambda x: -x["fisher_mlp"])]
    hybrid_ranked = [r["layer"] for r in sorted(layer_stats, key=lambda x: -x["hybrid_score"])]
    print("Top layers (MLP only):", mlp_ranked[:6], flush=True)
    print("Top layers (Hybrid MLP+head+attn):", hybrid_ranked[:6], flush=True)

    # 4. Blind exam against official p50c patch
    p50c = MoIPatch.load(args.patch)
    exam = {}
    for e in p50c.edits:
        m = re.search(r"layers\.(\d+)\.", e.module)
        if not m:
            continue
        li = int(m.group(1))
        if li in u_mlp_by_layer and e.idx is not None:
            cos_u = float(torch.nn.functional.cosine_similarity(u_mlp_by_layer[li].float().unsqueeze(0), e.u.float().unsqueeze(0)))
            ov = len(set(top64_by_layer[li].tolist()) & set(e.idx.long().tolist()))
            exam[f"layer_{li}"] = {"dir_cosine": round(abs(cos_u), 4), "neuron_overlap_of_64": ov}
    print("Blind exam vs p50c:", exam, flush=True)

    # Causal diff_compact on one looping sequence (base vs p50c)
    loop_patch_diff = None
    if loop_rows:
        seq0, ct0, ons0 = next((t for t in traces if t[0]["loop"]), traces[0])
        t_ons = min(seq0["prompt_len"] + seq0["onset_resp"], len(seq0["full_ids"]) - 1)
        hp = p50c.attach(model)
        hp.on()
        ct_p50c = capture_compact(model, seq0["full_ids"])
        hp.off()
        loop_patch_diff = diff_compact(ct0, ct_p50c, pos_a=t_ons, pos_b=t_ons, top_k=6)

    # 5. Live held-out eval comparing: base vs ct_mlp_top2 vs ct_hybrid_top3 vs p50c
    def _eval_arm(name: str, patch_obj: MoIPatch | None) -> dict:
        hp = patch_obj.attach(model) if patch_obj is not None else None
        if hp is not None:
            hp.on()
        try:
            t_ev = time.perf_counter()
            ev = _batched_generate(model, tok, list(gte["question"]), dev, args.max_new_tokens)
            ev_s = round(time.perf_counter() - t_ev, 1)
        finally:
            if hp is not None:
                hp.off()
        acc = sum(_is_correct(r["text"], _extract_gold(a)) for r, a in zip(ev, gte["answer"])) / len(ev)
        lrate = sum(r["loop"] for r in ev) / len(ev)
        mean_tok = sum(r["gen_len"] for r in ev) / len(ev)
        row = {"acc": round(acc, 4), "loop_rate": round(lrate, 4), "mean_gen_tok": round(mean_tok, 1), "s": ev_s}
        print(f"Eval [{name:<16}] -> {row}", flush=True)
        return row

    def _build_patch(layers_pick: list[int]) -> MoIPatch:
        edits = []
        for l in layers_pick:
            edits.append(Edit(module=f"layers.{l}.mlp.down_proj", u=u_mlp_by_layer[l].double(),
                              alpha=args.alpha, idx=top64_by_layer[l].long()))
        pt = MoIPatch(edits=edits, meta={"name": "compact-trace-discovered"})
        for e in pt.edits:
            w = mods[f"model.{e.module}"].weight if f"model.{e.module}" in mods else mods[e.module].weight
            from wsi.moipatch import _dtype_name, _sha256
            e.base_sha256, e.base_dtype, e.shape = _sha256(w), _dtype_name(w.dtype), tuple(w.shape)
        return pt

    # Find exact module prefix for fingerprinting
    mlp_fa = [l for l in mlp_ranked if l in healthy_att_focus]
    eval_res = {"base": _eval_arm("base", None)}
    if len(mlp_ranked) >= 3:
        eval_res["ct_mlp_top3"] = _eval_arm("ct_mlp_top3", _build_patch(mlp_ranked[:3]))
    if len(mlp_fa) >= 3:
        eval_res["ct_mlp_fa3"] = _eval_arm("ct_mlp_fa3", _build_patch(mlp_fa[:3]))
    if len(hybrid_ranked) >= 3:
        eval_res["ct_hybrid_top3"] = _eval_arm("ct_hybrid_top3", _build_patch(hybrid_ranked[:3]))
    eval_res["p50c_ref"] = _eval_arm("p50c_ref", p50c)

    out = {
        "model": args.model,
        "n_train": len(tr_rows),
        "n_eval": len(gte),
        "max_new_tokens": args.max_new_tokens,
        "train_gen_s": train_gen_s,
        "train_loop_count": len(loop_rows),
        "train_clean_count": len(clean_rows),
        "traces_captured": len(traces),
        "trace_bank_mb": bank_mb,
        "capture_s": cap_s,
        "mlp_top_layers": mlp_ranked[:6],
        "hybrid_top_layers": hybrid_ranked[:6],
        "layer_stats": layer_stats,
        "blind_exam_vs_p50c": exam,
        "causal_diff_base_vs_p50c_at_onset": loop_patch_diff,
        "eval_heldout": eval_res,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
