"""Follow-up of discover_loop_fix.py: where does the WSI-via-MoI gain come from, and can a better fix be picked?

The first run (benchmarks: results_discover.json) gave base 9%, WSI fix always on 14%, WSI fix via MoI 40%, p50c 59.5%.
Three checks, reusing that run's files (restored from Google Drive):
  placebo   the MoI arm's switch itself (cut at the router's firing point, continue with a new seed and every prefix
            token penalised) may break loops without any edit. Same continuation requests, BASE weights.
  retune    the first tuning used 8k-token dev responses while test allows 16k, so strong edits that stopped looping
            but needed longer answers were scored as truncated and the weakest edit won. Re-tune on the same dev
            problems at 16k, over three layer candidates: top, late, and fa (most separated full-attention layers;
            an architecture prior, nothing from p50c).
  report    test arms: base, WSI v1 fixed/MoI, placebo MoI, WSI v2 fixed/MoI, p50c; paired CIs; exam vs p50c.

    python discover_followup.py make-dev16 --work W
    python discover_followup.py pick --work W
    python discover_followup.py report --work W --p50c-dir P
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402


def cmd_make_dev16(a) -> None:
    work = Path(a.work)
    reqs = lr.read_jsonl(work / "requests_dev.jsonl")
    for r in reqs:
        r["max_tokens"] = a.max_new
    lr.write_jsonl(work / "requests_dev16.jsonl", reqs)
    print(f"requests_dev16.jsonl: {len(reqs)} dev problems at {a.max_new} new tokens")


def cmd_pick(a) -> None:
    work = Path(a.work)
    res = {}
    for f in sorted(work.glob("outputs_dev16_*_a*.jsonl")):
        if f.name == "outputs_dev16_base.jsonl":
            continue
        cand, al = f.stem[len("outputs_dev16_"):].rsplit("_a", 1)
        s = dl.score(work, "requests_dev16.jsonl", f.name)
        res[(cand, float(al))] = {k: v for k, v in s.items() if k != "rows"}
        print(f"  {cand:4s} alpha={float(al):g}: acc={s['accuracy']:.1%} loops={s['loop_killed']:.1%} "
              f"trunc={s['truncated']:.1%}")
    base = dl.score(work, "requests_dev16.jsonl", "outputs_dev16_base.jsonl")
    print(f"  base      : acc={base['accuracy']:.1%} loops={base['loop_killed']:.1%} trunc={base['truncated']:.1%}")
    best = max(res, key=lambda k: (res[k]["accuracy"], -res[k]["loop_killed"]))
    diag = json.loads((work / "diagnosis.json").read_text())
    out = {"cand": best[0], "alpha": best[1], "layers": diag["candidates"][best[0]],
           "candidates": diag["candidates"],
           "dev16": {"base": {k: v for k, v in base.items() if k != "rows"},
                     "grid": {f"{c}@{al:g}": v for (c, al), v in res.items()}},
           "fix_dir": str(work / f"wsi-fix-{best[0]}-a{best[1]:g}")}
    v = a.version
    (work / ("followup.json" if v == 2 else f"followup{v}.json")).write_text(json.dumps(out, indent=2))
    (work / f"chosen_fix{v}.txt").write_text(out["fix_dir"])
    print(f"chosen v{v}: {best[0]} layers {out['layers']}, alpha {best[1]:g}")


def moi_arm(work: Path, cont_file: str, reqs: dict, base_rows: dict, tok, eos: set) -> dict:
    """MoI composite: the base response until the router fires, then the continuation in `cont_file`."""
    plan = {r["id"]: r for r in lr.read_jsonl(work / "requests_test_moi.jsonl")}
    cont = {o["id"]: o for o in lr.read_jsonl(work / cont_file)}
    rows = {}
    for i in reqs:
        if i in plan:
            if i not in cont:
                raise SystemExit(f"{cont_file} lacks id {i}")
            pre = plan[i]["prompt_ids"][len(reqs[i]["prompt_ids"]) :]
            rows[i] = lr.grade(pre + cont[i]["token_ids"], cont[i]["finish_reason"], reqs[i]["gold"], tok, eos,
                               reqs[i]["max_tokens"])
        else:
            rows[i] = base_rows[i]
    return rows


def summary(rows: dict) -> dict:
    n = len(rows)
    return {"n": n, "accuracy": sum(g["correct"] for g in rows.values()) / n,
            "loop_killed": sum(g["looped_killed"] for g in rows.values()) / n,
            "truncated": sum(g["truncated"] for g in rows.values()) / n}


def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"])
    eos, reqs = set(cfg["eos"]), {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    rows: dict = {}
    for name, f in {"base": "outputs_test_base.jsonl", "wsi_fixed": "outputs_test_wsi.jsonl",
                    "wsi2_fixed": "outputs_test_wsi2.jsonl", "wsi3_fixed": "outputs_test_wsi3.jsonl",
                    "wsi4_fixed": "outputs_test_wsi4.jsonl",
                    "p50c": "outputs_test_p50c.jsonl"}.items():
        if (work / f).exists():
            rows[name] = {r["id"]: r for r in dl.score(work, "requests_test.jsonl", f, tok)["rows"]}
    for name, f in {"wsi_moi": "outputs_test_moi.jsonl", "placebo_moi": "outputs_test_placebo.jsonl",
                    "wsi2_moi": "outputs_test_moi2.jsonl", "wsi3_moi": "outputs_test_moi3.jsonl",
                    "wsi4_moi": "outputs_test_moi4.jsonl"}.items():
        if (work / f).exists():
            rows[name] = moi_arm(work, f, reqs, rows["base"], tok, eos)
    ids = sorted(reqs)
    acc = {k: [bool(v[i]["correct"]) for i in ids] for k, v in rows.items()}
    pairs = [("placebo_moi", "base"), ("wsi_moi", "placebo_moi"), ("wsi2_moi", "placebo_moi"),
             ("wsi2_fixed", "base"), ("wsi2_fixed", "wsi_fixed"), ("wsi2_moi", "wsi_moi"), ("wsi2_fixed", "p50c"),
             ("wsi3_fixed", "base"), ("wsi3_fixed", "wsi2_fixed"), ("wsi3_moi", "placebo_moi"), ("wsi3_moi", "wsi2_moi"),
             ("wsi3_fixed", "p50c"), ("wsi3_moi", "p50c"),
             ("wsi4_fixed", "base"), ("wsi4_fixed", "wsi3_fixed"), ("wsi4_moi", "placebo_moi"), ("wsi4_moi", "wsi3_moi"),
             ("wsi4_fixed", "p50c"), ("wsi4_moi", "p50c")]
    fu = json.loads((work / "followup.json").read_text()) if (work / "followup.json").exists() else {}
    fu3 = json.loads((work / "followup3.json").read_text()) if (work / "followup3.json").exists() else None
    res = {"arms": {k: summary(v) for k, v in rows.items()},
           "paired": {f"{x}_vs_{y}": lr.paired_ci(acc[x], acc[y]) for x, y in pairs if x in acc and y in acc},
           "v2_choice": fu}
    if fu3:
        res["v3_choice"] = fu3
    fu4 = json.loads((work / "followup4.json").read_text()) if (work / "followup4.json").exists() else None
    if fu4:
        res["v4_choice"] = fu4  # v3 layers, alpha grid extended to 8 and 10

    exam: dict = {"p50c_layers": [15, 19], "v2_layers": fu.get("layers"),
                  "v3_layers": fu3.get("layers") if fu3 else None,
                  "candidates": (fu3 or fu).get("candidates")}
    if a.p50c_dir:
        from safetensors import safe_open

        diag = json.loads((work / "diagnosis.json").read_text())
        z = np.load(work / "wsi_directions.npz")
        base = Path(cfg["base_dir"])
        for li in (15, 19):
            dW = None
            for f in sorted(Path(a.p50c_dir).glob("*.safetensors")):
                with safe_open(str(f), "pt") as fe, safe_open(str(base / f.name), "pt") as fb:
                    try:
                        k = dl.down_key(list(fe.keys()), li)
                    except SystemExit:
                        continue
                    dW = (fe.get_tensor(k).double() - fb.get_tensor(k).double()).numpy()
            if dW is None:
                continue
            cols = np.where(np.abs(dW).sum(0) > 0)[0]
            up = np.linalg.svd(dW[:, cols], full_matrices=False)[0][:, 0]
            uw = z[f"all_u{li}"].astype(np.float64)
            rec = {"abs_cos_wsi_direction_vs_p50c": round(float(abs(uw @ up) / np.linalg.norm(uw)), 4),
                   "random_baseline": round(1 / np.sqrt(len(up)), 4)}
            if str(li) in diag["neurons"]:
                ov = len(set(diag["neurons"][str(li)]) & set(cols.tolist()))
                rec["neuron_overlap"] = f"{ov}/{len(cols)}"
                rec["neuron_overlap_random_expectation"] = round(len(cols) * a.neurons / dW.shape[1], 2)
            exam[f"layer_{li}"] = rec
    res["exam_vs_p50c"] = exam
    res["timing"] = dl.timing_summary(work)
    (work / a.out).write_text(json.dumps(res, indent=2))

    print(f"\n{'arm':12s} {'acc':>7s} {'loops':>7s} {'trunc':>7s}")
    for k, v in res["arms"].items():
        print(f"{k:12s} {v['accuracy']:7.1%} {v['loop_killed']:7.1%} {v['truncated']:7.1%}")
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]  "
              f"({v['a_only']} vs {v['b_only']})")
    print("exam vs p50c:", json.dumps(exam, indent=1))
    print("Saved", work / a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-dev16")
    m.add_argument("--work", required=True)
    m.add_argument("--max-new", type=int, default=16384)
    p = sub.add_parser("pick")
    p.add_argument("--work", required=True)
    p.add_argument("--version", type=int, default=2, help="2 -> followup.json, 3 -> followup3.json")
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--p50c-dir")
    r.add_argument("--neurons", type=int, default=64)
    r.add_argument("--out", default="results_discover_followup.json")
    a = ap.parse_args()
    fns = {"make-dev16": cmd_make_dev16, "pick": cmd_pick, "report": cmd_report}
    dl.run_timed(a, fns, {"pick": f"pick:v{getattr(a, 'version', 2)}", "make-dev16": "make-dev16"}.get(a.cmd))


if __name__ == "__main__":
    main()
