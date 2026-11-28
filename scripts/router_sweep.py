"""Router rule sweep on saved outputs (CPU, token ids only, no tokenizer).

For every response of the fixed arms (one model for the whole answer) it labels:
  loop     the kill rule fired (>= 90% repeated 16-grams over 768 tokens)
  healthy  not killed, not truncated, and graded correct
and for each router rule (dup n-gram k, window, fraction) it measures:
  recall       share of loop responses where the router fires before the kill
  lead         median tokens between fire and kill (how early the warning comes)
  false fire   share of healthy responses where the router fires at all

Labels come from the rescue_answer.py result rows (same grading as the runs).

  python scripts/router_sweep.py --run gsm8k <work> <rescue results json> --run math <work> <json> --out out.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import moi_loop_router_vllm as lr  # noqa: E402

FIXED = {"gsm8k": {"base": "outputs_test_base.jsonl", "blind_fixed": "outputs_test_blind.jsonl",
                   "random_fixed": "outputs_test_random.jsonl", "p50c": "outputs_test_p50c.jsonl"},
         "math": {"base": "outputs_test_base.jsonl", "wsi_fixed": "outputs_test_fast.jsonl",
                  "random_fixed": "outputs_test_random.jsonl"}}
KS = [8, 16, 32]
WINDOWS = [32, 64, 128, 256]
FRACS = [0.3, 0.5, 0.7, 0.9]


def collect(dataset: str, work: Path, rows_json: Path) -> list[dict]:
    eos = set(json.loads((work / "config.json").read_text())["eos"])
    rows = json.loads(rows_json.read_text())["rows"]
    out = []
    for arm, f in FIXED[dataset].items():
        if arm not in rows or not (work / f).exists():
            continue
        lab = {r["id"]: r for r in rows[arm]}
        for o in lr.read_jsonl(work / f):
            r = lab[o["id"]]
            body = lr.strip_eos(o["token_ids"], eos)
            d16 = lr.dup_flags(body, 16)
            kill = lr.first_window_hit(d16, lr.KILL_WINDOW, lr.KILL_FRAC)
            healthy = (not r["killed"]) and (not r["truncated"]) and r["correct"]
            if kill is None and not healthy:
                continue  # wrong-but-finished or truncated: neither class
            out.append({"ds": dataset, "arm": arm, "id": o["id"], "body": body, "kill": kill, "healthy": healthy})
    return out


def sweep(items: list[dict]) -> list[dict]:
    res = []
    for k in KS:
        flags = [lr.dup_flags(x["body"], k) for x in items]
        for w in WINDOWS:
            for fr in FRACS:
                hit_loop, leads, n_loop, ff, n_h = 0, [], 0, 0, 0
                for x, d in zip(items, flags):
                    end = x["kill"] if x["kill"] is not None else len(d)
                    fire = lr.first_window_hit(d[:end], w, fr)
                    if x["kill"] is not None:
                        n_loop += 1
                        if fire is not None:
                            hit_loop += 1
                            leads.append(x["kill"] - fire)
                    else:
                        n_h += 1
                        ff += fire is not None
                res.append({"k": k, "window": w, "frac": fr, "recall": round(hit_loop / max(n_loop, 1), 3),
                            "lead_median": int(st.median(leads)) if leads else None,
                            "false_fire": round(ff / max(n_h, 1), 3), "n_loop": n_loop, "n_healthy": n_h})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=3, action="append", metavar=("DATASET", "WORK", "RESCUE_JSON"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = []
    for ds, work, js in a.run:
        items += collect(ds, Path(work), Path(js))
    out = {}
    for name, sub in [("all", items)] + [(ds, [x for x in items if x["ds"] == ds]) for ds in sorted({x["ds"] for x in items})]:
        out[name] = sweep(sub)
        print(f"\n== {name}: loops {sum(x['kill'] is not None for x in sub)}, healthy {sum(x['healthy'] for x in sub)}")
        print(f"{'k':>3s} {'win':>4s} {'frac':>5s} {'recall':>7s} {'lead':>6s} {'falseF':>7s}")
        for r in out[name]:
            mark = "  <- current" if (r["k"], r["window"], r["frac"]) == (16, 64, 0.5) else ""
            print(f"{r['k']:3d} {r['window']:4d} {r['frac']:5.1f} {r['recall']:7.1%} {str(r['lead_median']):>6s} "
                  f"{r['false_fire']:7.1%}{mark}")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("Saved", a.out)


if __name__ == "__main__":
    main()
