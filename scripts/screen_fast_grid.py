"""Prospective test of the 1-forward screen (wsi.tracer.screen_patch_candidates) on a discover_fast run.

experiment_screen_proxy_validate.py checked the screen on Qwen3.5-0.8B, where the answer was already known: the
blind winner L13-14_n16_a12 ranked #2 of 60 by the default order of that run and #1 by pareto_score, and pareto_score
became the default after that. One model cannot show that the rule generalizes. This script applies the rule, frozen
here before any result of the new run was seen, to the 12 configs of a discover_fast run on another model, and only
then compares it with what the real generation search found.

Frozen rule (do not change after a run's search results exist):
  * candidates: exactly the 12 configs discover_fast builds (same direction u, same neuron ranking, same alpha, same
    W' = W - alpha * u (m * u^T W)), read from the run's diagnosis_blind.json / blind_directions.npz;
  * sequences: the held-out half-B looping elicit responses of that diagnosis (same permutation), at most 6, with the
    recipe of experiment_screen_proxy_validate.py: healthy = prompt + response up to min(onset, 128) tokens,
    target = prompt + response up to onset + 192, scored on [onset, onset + 192);
  * rank: screen_patch_candidates(rank_by="pareto", max_nll_delta=0.05).

Prediction under test: the config the generation search chose ranks in the screen's top 3 of 12 (chance: 25%).
Also reported: the choice's rank, Spearman of pareto_score vs round-1 accuracy and loop rate over the 12 configs, and
how many of the 3 round-2 configs are in the screen's top 3 and top 6.

The ranking is computed and written to disk before search.json is opened.

    python screen_fast_grid.py --work W [--out results_screen_fast.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import discover_blind as blind  # noqa: E402
import discover_fast as fast  # noqa: E402
import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402

MAX_NLL_DELTA = 0.05
RANK_BY = "pareto"
N_SCREEN_SEQS = 6
HEALTHY_MAX = 128
TARGET_AFTER = 192
TOP = 3


def screen_sequences(work: Path) -> tuple[list, list, list, list[int]]:
    """Half-B looping elicit responses, selected exactly as discover_fast.cmd_diagnose split them."""
    cfg = dl.cfg_load(work)
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_elicit.jsonl")}
    seqs = []
    for o in lr.read_jsonl(work / "outputs_elicit.jsonl"):
        body = lr.strip_eos(o["token_ids"], eos)
        lab = dl.token_labels(body)
        if not (lab >= 0).any():
            continue
        first = int(np.argmax(lab == 1)) if (lab == 1).any() else None
        seqs.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body, "first": first})
    half = np.random.default_rng(cfg["seed"]).permutation(len(seqs)) % 2
    picked = [s for s, hb in zip(seqs, half) if hb == 1 and s["first"] is not None][:N_SCREEN_SEQS]
    healthy, target, spans = [], [], []
    for s in picked:
        p, ons = len(s["prompt"]), s["first"]
        full = s["prompt"] + s["body"]
        healthy.append(full[: p + max(16, min(ons, HEALTHY_MAX))])
        t_end = min(len(full), p + ons + TARGET_AFTER)
        target.append(full[:t_end])
        spans.append((p + ons, t_end))
    return healthy, target, spans, [s["id"] for s in picked]


def build_candidates(work: Path, model) -> dict:
    import torch

    from wsi.moipatch import Edit, MoIPatch, _dtype_name, _sha256, module_suffix

    diag = json.loads((work / "diagnosis_blind.json").read_text())
    z = np.load(work / "blind_directions.npz")
    downs = {}
    for name, mod in model.named_modules():
        m = fast.DOWN_RE.search(name + ".weight")
        if m and "mtp" not in name and "visual" not in name:
            downs[int(m.group(1))] = (name, mod)
    out = {}
    for c in fast.grid(diag["layer_rank"]):
        edits = []
        for li in c["layers"]:
            name, mod = downs[li]
            w = mod.weight.detach()
            cols = torch.tensor(blind.columns(diag, li, c["neurons"]))  # as discover_fast.job_updates ("all" = ranked set)
            edits.append(Edit(module=module_suffix(name), u=torch.from_numpy(blind.direction(diag, z, li, "wsi")),
                              alpha=float(c["alpha"]), idx=cols, base_sha256=_sha256(w), base_dtype=_dtype_name(w.dtype),
                              shape=tuple(w.shape)))
        out[c["name"]] = MoIPatch(edits=edits, meta={"name": c["name"]})
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / max(den, 1e-12), 4)


def grade(rows: list[dict], search: dict) -> dict:
    names = [r["name"] for r in rows]
    rank = {n: i + 1 for i, n in enumerate(names)}
    # a search over a bigger grid (discover_blind's 60) has configs outside these 12: grade on the shared ones
    r1 = {k: v for k, v in search["round1"].items() if k in rank}
    r2 = {k: v for k, v in search.get("round2", {}).items() if k in rank}
    common = [r for r in rows if r["name"] in r1]
    res = {"n_configs": len(rows),
           "spearman_pareto_vs_r1_acc": _spearman([r["pareto_score"] for r in common],
                                                  [r1[r["name"]]["accuracy"] for r in common]),
           "spearman_pareto_vs_r1_loop": _spearman([r["pareto_score"] for r in common],
                                                   [r1[r["name"]]["loop_killed"] for r in common]),
           "spearman_dnll_vs_r1_trunc": _spearman([r["delta_healthy_nll"] for r in common],
                                                  [r1[r["name"]]["truncated"] for r in common]),
           "round2_configs_screen_rank": {n: rank[n] for n in r2},
           f"round2_in_screen_top{TOP}": sum(rank[n] <= TOP for n in r2),
           "round2_in_screen_top6": sum(rank[n] <= 6 for n in r2)}
    best_r1 = max(r1, key=lambda k: blind.score_key(r1[k]))
    res["round1_best"] = {"name": best_r1, "screen_rank": rank[best_r1]}
    res["round2_outside_grid"] = sorted(set(search.get("round2", {})) - set(r2))
    if "choice" in search and search["choice"]["name"] in rank:
        ch = search["choice"]["name"]
        res["choice"] = {"name": ch, "screen_rank": rank[ch], f"in_top{TOP}": rank[ch] <= TOP,
                         "chance_in_top": round(TOP / len(rows), 3)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="results_screen_fast.json")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    from wsi.tracer import screen_patch_candidates

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.bfloat16 if dev.type == "cuda" and fast.bf16_ok() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["base_dir"], dtype=dtype).to(dev).eval()
    healthy, target, spans, ids = screen_sequences(work)
    if not target:
        raise SystemExit("no held-out looping responses to screen on")
    cands = build_candidates(work, model)
    t0 = time.perf_counter()
    rows = screen_patch_candidates(model, cands, healthy_seqs=healthy, target_seqs=target, target_spans=spans,
                                   max_nll_delta=MAX_NLL_DELTA, rank_by=RANK_BY)
    screen_s = round(time.perf_counter() - t0, 1)
    res = {"model": cfg.get("model_repo"), "rule": {"rank_by": RANK_BY, "max_nll_delta": MAX_NLL_DELTA,
                                                    "n_screen_seqs": N_SCREEN_SEQS, "healthy_max": HEALTHY_MAX,
                                                    "target_after": TARGET_AFTER, "top": TOP},
           "diag_version": diag.get("diag_version"), "screen_seq_ids": ids, "screen_s": screen_s,
           "ranked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ranking": rows}
    out = work / a.out
    out.write_text(json.dumps(res, indent=1))  # the ranking is on disk before the search results are read
    print(f"screened {len(rows)} configs on {len(target)} sequences in {screen_s}s")
    for i, r in enumerate(rows, 1):
        print(f"  #{i:2d} {r['name']:18s} pareto={r['pareto_score']:+9.3f} supp={r['target_suppression']:+.4f} "
              f"dNLL={r['delta_healthy_nll']:+.4f} guard={r['passes_nll_guard']}")
    sp = work / "search.json"
    if sp.exists():
        res["vs_search"] = grade(rows, json.loads(sp.read_text()))
        out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res["vs_search"], indent=1))
    else:
        print("search.json not found yet: ranking saved, grade it after the search finishes")
    fast.sync(work)


if __name__ == "__main__":
    main()
