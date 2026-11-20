"""Restart-only control for the MoI loop router: is the gain the edit, or just stopping and resuming?

The MoI arm cuts each base answer where the loop router fires and resumes it from there with the PATCHED model
(new seed, the prefix's tokens penalised like the presence penalty would). On MATH-500 a random direction resumed the
same way tied our direction (18.7% vs 15.3%). This control resumes the exact same requests with the UNPATCHED base
model: if it ties the patched MoI arm, the gain is the restart, not the weight edit.

Reuses a finished run's work dir (config.json, requests_test.jsonl, requests_test_moi.jsonl, outputs_test_base.jsonl
and the MoI outputs), so only the restart arm is generated.

  gen     base model on requests_test_moi.jsonl -> outputs_test_moi_restart.jsonl
  report  restart arm vs base / patched MoI / random MoI, paired, -> results_restart_control.json

--dataset math  : discover_math500_blind.py work dir (arms wsi_moi, random_moi)
--dataset gsm8k : discover_blind.py work dir (arms blind_moi, random_moi)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402

PLAN = "requests_test_moi.jsonl"
RESTART_OUT = "outputs_test_moi_restart.jsonl"
MOI_ARMS = {
    "math": {"wsi_moi": "outputs_test_moi_fast.jsonl", "random_moi": "outputs_test_moi_random.jsonl"},
    "gsm8k": {"wsi_moi": "outputs_test_moi_blind.jsonl", "random_moi": "outputs_test_moi_random.jsonl"},
}


def cmd_gen(a) -> None:
    work = Path(a.work)
    cfg = dl.cfg_load(work)
    if a.base_dir:  # the run was made on another machine: point it at this runtime's copy of the base model
        cfg["base_dir"] = str(a.base_dir)
        dl.cfg_save(work, cfg)
    if not (work / PLAN).exists():
        raise SystemExit(f"{PLAN} missing: the run has no MoI plan")
    if (work / RESTART_OUT).exists():
        print(f"{RESTART_OUT} already there")
        return
    import discover_math500_blind as m500

    jobs = [{"name": "restart", "requests": PLAN, "out": RESTART_OUT}]
    m500._run_parallel_fast(work, [{"model": None, "jobs": jobs}], a)
    print(f"restart arm: {len(lr.read_jsonl(work / RESTART_OUT))} resumed answers")


def _graders(dataset: str, work: Path, tok, eos: set[int]):
    """(score whole answers, grade a resumed answer, token-count key) in the same way as each run's own report."""
    if dataset == "math":
        import discover_math500_blind as m500

        def resumed(pre, o, r):
            full = pre + lr.strip_eos(o["token_ids"], eos)
            return m500.grade_math(full, o["finish_reason"], r["gold"], tok, eos, r["max_tokens"])

        return (lambda fn: m500.score_math(work, "requests_test.jsonl", fn, tok)), resumed

    def resumed(pre, o, r):
        return lr.grade(pre + o["token_ids"], o["finish_reason"], r["gold"], tok, eos, r["max_tokens"])

    return (lambda fn: dl.score(work, "requests_test.jsonl", fn, tok)), resumed


def report(work: Path, dataset: str, tok=None) -> dict:
    from transformers import AutoTokenizer

    cfg = dl.cfg_load(work)
    tok = tok or AutoTokenizer.from_pretrained(cfg["base_dir"], trust_remote_code=True)
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    plan = {r["id"]: r for r in lr.read_jsonl(work / PLAN)}
    score, resumed = _graders(dataset, work, tok, eos)

    base = {r["id"]: r for r in score("outputs_test_base.jsonl")["rows"]}
    rows = {"base": base}
    for name, fn in {**MOI_ARMS[dataset], "restart_moi": RESTART_OUT}.items():
        if not (work / fn).exists():
            print(f"  ({name}: {fn} missing, skipped)")
            continue
        cont = {o["id"]: o for o in lr.read_jsonl(work / fn)}
        arm = {}
        for i, r in reqs.items():
            if i in cont:
                arm[i] = resumed(plan[i]["prompt_ids"][len(r["prompt_ids"]):], cont[i], r)
            else:
                arm[i] = base[i]
        rows[name] = arm

    ids = sorted(reqs)
    res = {"dataset": dataset, "model": cfg.get("model_repo"), "n": len(ids),
           "router_fired": len(plan) / max(len(ids), 1), "arms": {}, "paired": {}}
    for name, arm in rows.items():
        g = [arm[i] for i in ids]
        res["arms"][name] = {k: round(sum(bool(x[k]) for x in g) / len(g), 4)
                             for k in ("correct", "looped_killed", "truncated")}
    acc = {k: [bool(rows[k][i]["correct"]) for i in ids] for k in rows}
    for x, y in (("restart_moi", "base"), ("wsi_moi", "restart_moi"), ("random_moi", "restart_moi"),
                 ("wsi_moi", "base")):
        if x in acc and y in acc:
            res["paired"][f"{x}_vs_{y}"] = lr.paired_ci(acc[x], acc[y])
    if "restart_moi" in acc and "wsi_moi" in acc:
        ci = res["paired"]["wsi_moi_vs_restart_moi"]["ci95"]
        res["verdict"] = ("edit adds over restart" if ci[0] > 0 else
                          "restart beats edit" if ci[1] < 0 else "edit ties restart (gain is the restart)")
    return res


def cmd_report(a) -> None:
    work = Path(a.work)
    res = report(work, a.dataset)
    (work / "results_restart_control.json").write_text(json.dumps(res, indent=1))
    print(f"\n{a.dataset}: n={res['n']}, router fired on {res['router_fired']:.0%}")
    print(f"{'arm':12s} {'acc':>7s} {'loops':>7s} {'trunc':>7s}")
    for k, v in res["arms"].items():
        print(f"{k:12s} {v['correct']:7.1%} {v['looped_killed']:7.1%} {v['truncated']:7.1%}")
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]  ({v['a_only']}/{v['b_only']})")
    print("verdict:", res.get("verdict", "n/a"))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--base-dir", default=None)
    g.add_argument("--gpu-mem", type=float, default=0.92)
    g.add_argument("--dtype", default="auto")
    g.add_argument("--cpu", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--dataset", choices=sorted(MOI_ARMS), required=True)
    a = ap.parse_args()
    {"gen": cmd_gen, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
