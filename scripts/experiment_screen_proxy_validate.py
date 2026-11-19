"""Validate all 4 upgraded `discover_compact_circuit` + `screen_patch_candidates` features on real `Qwen/Qwen3.5-0.8B`.

Runs in ~3-4 min on Colab L4:
  1. Generates 24 GSM8K train sequences (max_new_tokens=2048) and captures up to 12 `CompactTrace`s (`head_quant="int8"`).
  2. Runs upgraded `discover_compact_circuit` (blind unified MLP + W_O Attention, Half-A/B `d'`, automatic residual router).
  3. Builds all 60 `(k, n, alpha)` configs from `results_blind_colab_l4.json` (`L14`, `L13-14`, `L10-13-14` x
     `n in {16, 64, 256, all}` x `alpha in {2, 4, 6, 8, 12}`) and runs `screen_patch_candidates` (1 teacher-forced
     forward pass per sequence, zero generation) on held-out Half-B tokens.
  4. Grades `screen_patch_candidates` against the 60 ground-truth accuracies in `results_blind_colab_l4.json`:
     Spearman correlations, rank of `L13-14_n16_a12` (the `discover_blind` winner), and Top-12 recall of `round2` finalists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsi.capture import find_mlps  # noqa: E402
from wsi.moipatch import Edit, MoIPatch, _dtype_name, _sha256, module_suffix  # noqa: E402
from wsi.tracer import capture_compact, discover_compact_circuit, screen_patch_candidates  # noqa: E402

HUB_FILES = {
    "gsm8k/train.parquet": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
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


def _loop_onset(ids: list[int], n: int = 16, window: int = 64, frac: float = 0.5) -> int | None:
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
        else:
            out_ids = out_raw
        onset = _loop_onset(out_ids)
        rows.append({
            "prompt_len": len(p_ids),
            "gen_len": len(out_ids),
            "full_ids": p_ids + out_ids,
            "loop": onset is not None,
            "onset_resp": onset,
        })
    return rows


def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return round(num / max(dx * dy, 1e-12), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--blind-json", default=str(ROOT / "benchmarks" / "discover_loop_fix" / "results_blind_colab_l4.json"))
    ap.add_argument("--data-dir", default=str(ROOT / "work"))
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-traces", type=int, default=12)
    ap.add_argument("--n-screen-seqs", type=int, default=6)
    ap.add_argument("--max-nll-delta", type=float, default=0.05)
    ap.add_argument("--out", default="results_screen_proxy_validate.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = _local_or_hub(args.model)
    tok = AutoTokenizer.from_pretrained(mdir)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype, attn_implementation="eager").to(dev).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype).to(dev).eval()

    gtr = pd.read_parquet(need(Path(args.data_dir), "gsm8k/train.parquet")).iloc[: args.n_train]

    # 1. Generate train pool & capture CompactTraces
    t0 = time.perf_counter()
    tr_rows = _batched_generate(model, tok, list(gtr["question"]), dev, args.max_new_tokens)
    gen_s = round(time.perf_counter() - t0, 1)
    loop_rows = [r for r in tr_rows if r["loop"]]
    clean_rows = [r for r in tr_rows if not r["loop"]]
    print(f"Generated {len(tr_rows)} seqs in {gen_s}s: loop={len(loop_rows)} clean={len(clean_rows)}", flush=True)
    if len(loop_rows) < 2:
        raise SystemExit("Need >= 2 looping sequences; raise --n-train or --max-new-tokens.")

    chosen = loop_rows[: args.max_traces]
    if len(chosen) < args.max_traces:
        chosen += clean_rows[: max(0, args.max_traces - len(chosen))]

    t_cap = time.perf_counter()
    traces = []
    for r in chosen:
        ct = capture_compact(model, r["full_ids"])
        traces.append((r, ct, None))
    cap_s = round(time.perf_counter() - t_cap, 2)
    bank_mb = round(sum(ct.size_bytes() for _, ct, _ in traces) / 2**20, 2)
    print(f"Captured {len(traces)} CompactTraces in {cap_s}s ({bank_mb} MiB)", flush=True)

    # 2. Run upgraded discover_compact_circuit (all 4 features: unified MLP+Attn W_O, Half-A/B d', auto router)
    t_disc = time.perf_counter()
    disc = discover_compact_circuit(traces, model=model, alpha=6.0, top_k_neurons=64, top_k_heads=2)
    disc_s = round(time.perf_counter() - t_disc, 2)
    print(f"Discovery in {disc_s}s | split_mode={disc['split_mode']}", flush=True)
    print("Top-6 MLP (d'):", [(r["layer"], r["dprime_mlp"]) for r in sorted(disc["layer_stats"], key=lambda x: -x["dprime_mlp"])[:6]], flush=True)
    print("Top-6 Attn W_O (d'):", [(r["layer"], r["module"], r["dprime_attn"], r["top_heads"]) for r in disc["attn_stats"][:6]], flush=True)
    print("Top-8 Unified sites:", [(r["site_type"], r["layer"], r["module"], r["dprime"]) for r in disc["site_rankings"][:8]], flush=True)
    print("Auto router report:", disc["router_report"], flush=True)

    # Verify live router firing on Half-B sequences (odd indices)
    half_b_rows = [r for i, (r, _, _) in enumerate(traces) if i % 2 == 1] or [traces[-1][0]]
    router_live = {"healthy_fp": 0, "healthy_n": 0, "loop_tp": 0, "loop_n": 0}
    if disc["patch"].router is not None:
        with torch.no_grad():
            for r in half_b_rows[: args.n_screen_seqs]:
                p = r["prompt_len"]
                if r["loop"] and r["onset_resp"] is not None:
                    ons = min(p + r["onset_resp"], len(r["full_ids"]) - 1)
                    h_seq = r["full_ids"][: max(p + 16, min(ons, p + 96))]
                    l_seq = r["full_ids"][: min(len(r["full_ids"]), ons + 192)]
                    oh = model(torch.tensor([h_seq], device=dev), output_hidden_states=True, use_cache=False)
                    ol = model(torch.tensor([l_seq], device=dev), output_hidden_states=True, use_cache=False)
                    router_live["healthy_fp"] += int(disc["patch"].fires(oh.hidden_states).item())
                    router_live["healthy_n"] += 1
                    router_live["loop_tp"] += int(disc["patch"].fires(ol.hidden_states).item())
                    router_live["loop_n"] += 1
    print("Live router check (Half-B):", router_live, flush=True)

    # 3. Build all 60 ground-truth configs from results_blind_colab_l4.json and screen with `screen_patch_candidates`
    blind_gt = json.loads(Path(args.blind_json).read_text(encoding="utf-8"))
    r1_gt = blind_gt["search"]["round1"]
    r2_gt = blind_gt["search"]["round2"]

    mods = dict(model.named_modules())
    mlps = find_mlps(model)
    w_downs = {}
    meta_by_l = {}
    for l, g in enumerate(mlps):
        m = mods[g.down]
        w_orig = m.weight.detach()
        w_downs[l] = w_orig.float().cpu()
        meta_by_l[l] = {
            "module": module_suffix(g.down),
            "shape": tuple(w_orig.shape),
            "base_dtype": _dtype_name(w_orig.dtype),
            "base_sha256": _sha256(w_orig),
        }

    # Precompute full neuron ranking per layer on Half-A traces (matching discover_compact_circuit)
    full_neuron_rank: dict[int, torch.Tensor] = {}
    for l in (10, 13, 14):
        u_l = disc["directions"][l]
        hx_list, lx_list = [], []
        for idx_t, (r, ct, _) in enumerate(traces):
            if idx_t % 2 != 0:
                continue
            T, p = ct.seq_len, r["prompt_len"]
            ons = min(p + r["onset_resp"], T - 1) if (r["loop"] and r["onset_resp"] is not None) else T
            h_idx = torch.arange(min(p, T), max(min(p, T), min(ons, p + 96, T)))
            l_idx = torch.arange(max(ons, T - 192), T) if r["loop"] else torch.empty(0, dtype=torch.long)
            idx_m, val_m = ct.dequant_mlp_topk(l)
            wn = w_downs[l].norm(dim=0).clamp_min(1e-12)
            d_inter = w_downs[l].shape[1]
            if len(h_idx):
                xh = torch.zeros((len(h_idx), d_inter), dtype=torch.float32)
                xh.scatter_(1, idx_m[h_idx], val_m[h_idx] / wn[idx_m[h_idx]])
                hx_list.append(xh.mean(dim=0, keepdim=True))
            if len(l_idx):
                xl = torch.zeros((len(l_idx), d_inter), dtype=torch.float32)
                xl.scatter_(1, idx_m[l_idx], val_m[l_idx] / wn[idx_m[l_idx]])
                lx_list.append(xl.mean(dim=0, keepdim=True))
        dx = torch.cat(lx_list, dim=0).mean(dim=0) - torch.cat(hx_list, dim=0).mean(dim=0)
        full_neuron_rank[l] = torch.argsort(dx * (u_l @ w_downs[l]), descending=True)

    grid_60: dict[str, MoIPatch] = {}
    for layers_pick, pref in (([14], "L14"), ([13, 14], "L13-14"), ([10, 13, 14], "L10-13-14")):
        for n_lbl in ("16", "64", "256", "all"):
            for a_val in (2, 4, 6, 8, 12):
                name = f"{pref}_n{n_lbl}_a{a_val}"
                edits = []
                for l in layers_pick:
                    ml = meta_by_l[l]
                    idx_cols = None if n_lbl == "all" else full_neuron_rank[l][: int(n_lbl)].sort().values.long()
                    edits.append(Edit(
                        module=ml["module"],
                        u=disc["directions"][l].double(),
                        alpha=float(a_val),
                        idx=idx_cols,
                        base_sha256=ml["base_sha256"],
                        base_dtype=ml["base_dtype"],
                        shape=ml["shape"],
                    ))
                grid_60[name] = MoIPatch(edits=edits, meta={"name": name})

    # Held-out Half-B screening sequences (shortened to onset+192 so 60 configs run in seconds!)
    healthy_seqs, target_seqs, target_spans = [], [], []
    for r in half_b_rows[: args.n_screen_seqs]:
        p = r["prompt_len"]
        if r["loop"] and r["onset_resp"] is not None:
            ons = min(p + r["onset_resp"], len(r["full_ids"]) - 1)
            h_end = max(p + 16, min(ons, p + 128))
            t_end = min(len(r["full_ids"]), ons + 192)
            healthy_seqs.append(r["full_ids"][:h_end])
            target_seqs.append(r["full_ids"][:t_end])
            target_spans.append((ons, t_end))

    t_scr = time.perf_counter()
    screened_60 = screen_patch_candidates(
        model, grid_60,
        healthy_seqs=healthy_seqs, target_seqs=target_seqs, target_spans=target_spans,
        max_nll_delta=args.max_nll_delta,
    )
    screen_s = round(time.perf_counter() - t_scr, 2)

    screened_disc = screen_patch_candidates(
        model, disc["patches"],
        healthy_seqs=healthy_seqs, target_seqs=target_seqs, target_spans=target_spans,
        max_nll_delta=args.max_nll_delta,
    )

    # Correlate proxy metrics against real ground-truth accuracy / loop_killed / truncated from discover_blind
    gt_acc, gt_loop, gt_trunc = [], [], []
    pr_supp, pr_dnll, pr_pareto = [], [], []
    for row in screened_60:
        gt = r1_gt[row["name"]]
        row["gt_r1_acc"] = gt["accuracy"]
        row["gt_r1_loop"] = gt["loop_killed"]
        row["gt_r1_trunc"] = gt["truncated"]
        row["in_r2"] = row["name"] in r2_gt
        gt_acc.append(gt["accuracy"])
        gt_loop.append(gt["loop_killed"])
        gt_trunc.append(gt["truncated"])
        pr_supp.append(row["target_suppression"])
        pr_dnll.append(row["delta_healthy_nll"])
        pr_pareto.append(row["pareto_score"])

    winner_rank = next(i + 1 for i, r in enumerate(screened_60) if r["name"] == "L13-14_n16_a12")
    r2_ranks = {r["name"]: i + 1 for i, r in enumerate(screened_60) if r["in_r2"]}
    top12_r2_hits = sum(1 for rk in r2_ranks.values() if rk <= 12)
    top15_r2_hits = sum(1 for rk in r2_ranks.values() if rk <= 15)

    corr = {
        "spearman_suppression_vs_r1_acc": _spearman(pr_supp, gt_acc),
        "spearman_suppression_vs_r1_loop": _spearman(pr_supp, gt_loop),
        "spearman_dnll_vs_r1_trunc": _spearman(pr_dnll, gt_trunc),
        "spearman_dnll_vs_r1_acc": _spearman(pr_dnll, gt_acc),
        "spearman_pareto_vs_r1_acc": _spearman(pr_pareto, gt_acc),
        "spearman_pareto_vs_r1_loop": _spearman(pr_pareto, gt_loop),
        "winner_L13_14_n16_a12_rank_of_60": winner_rank,
        "r2_finalists_ranks_of_60": r2_ranks,
        "r2_finalists_in_top12": f"{top12_r2_hits}/6",
        "r2_finalists_in_top15": f"{top15_r2_hits}/6",
        "screen_60_s": screen_s,
    }
    print(f"\nScreened all 60 configs in {screen_s}s (vs 24,870s = 414.5 min in discover_blind!)", flush=True)
    print("Proxy vs Ground-Truth correlation:", json.dumps(corr, indent=2), flush=True)
    print("\nTop-10 by screen_patch_candidates:", flush=True)
    for i, r in enumerate(screened_60[:10], 1):
        print(f"  #{i:2d} {r['name']:16s} | supp={r['target_suppression']:+.4f} dNLL={r['delta_healthy_nll']:+.4f} "
              f"guard={r['passes_nll_guard']} | GT r1_acc={r['gt_r1_acc']:.3f} r1_loop={r['gt_r1_loop']:.3f} r2={r['in_r2']}", flush=True)
    print("\nDiscovered patches screened:", json.dumps(screened_disc, indent=2), flush=True)

    out = {
        "model": args.model,
        "n_train": len(tr_rows),
        "traces_captured": len(traces),
        "trace_bank_mb": bank_mb,
        "gen_s": gen_s,
        "capture_s": cap_s,
        "discover_s": disc_s,
        "split_mode": disc["split_mode"],
        "router_report": disc["router_report"],
        "router_live_half_b": router_live,
        "top_mlp_layers": sorted(disc["layer_stats"], key=lambda x: -x["dprime_mlp"])[:8],
        "top_attn_sites": disc["attn_stats"][:8],
        "top_unified_sites": disc["site_rankings"][:12],
        "screen_disc_patches": screened_disc,
        "proxy_vs_blind_gt": corr,
        "screened_60": screened_60,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
