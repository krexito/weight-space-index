"""EXPERIMENTAL (CPU): are the cuts the patch never solves loop-bound or capability-bound?

From experiment_futures.py (8 patched + 8 restart futures per router cut). For every cut, each of its 16 futures is
classed: correct / loops again (early-stop kill) / truncated / finished with a wrong answer (no answer counts as
wrong). A cut whose futures mostly finish WRONG (not looping) is capability-bound: no anti-loop edit can fix it.
Cross-check: is the same problem solved by any other full-answer arm of the discover_blind test run (base, always-on
blind patch, MIT p50c, random direction, both MoI arms)? And do the wrong finishes agree on one wrong answer?

  python scripts/experiment_futures_hard.py --work FUTURES_WORK --discover DISCOVER_WORK --tokenizer TOK \
      --futures RESULTS_FUTURES.json --out R.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import experiment_futures as fu
import moi_loop_router_vllm as lr
import rescue_answer as ra

FULL_ARMS = {"base": "outputs_test_base.jsonl", "blind_always": "outputs_test_blind.jsonl",
             "p50c_always": "outputs_test_p50c.jsonl", "random_always": "outputs_test_random.jsonl"}
MOI_ARMS = {"moi_blind": "outputs_test_moi_blind.jsonl", "moi_random": "outputs_test_moi_random.jsonl"}


def classify(row: dict) -> str:
    if row["correct"]:
        return "correct"
    if row["killed"]:
        return "loops"
    if row.get("truncated"):
        return "truncated"
    return "wrong_finish"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--discover", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--futures", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    work, disc = Path(a.work), Path(a.discover)
    cfg = json.loads((work / "config.json").read_text())
    eos = set(cfg["eos"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    plan = {(r["id"], r["k"]): r for r in lr.read_jsonl(work / fu.PLAN)}
    grader = ra.Grader("gsm8k")
    res_f = json.loads(Path(a.futures).read_text())
    K = res_f["k"]
    cuts = sorted({i for i, _ in plan})
    rp = {int(c): v for c, v in res_f["arms"]["patch"]["per_cut_rate"].items()}
    rr = {int(c): v for c, v in res_f["arms"]["restart"]["per_cut_rate"].items()}

    cls, ans = {}, {}
    for arm in fu.ARMS:
        rows = {(o["id"], o["k"]): o for o in lr.read_jsonl(work / f"futures_{arm}.jsonl")}
        for j in range(K):
            seqs = {i: (plan[(i, j)]["prompt_ids"][len(reqs[i]["prompt_ids"]):] + rows[(i, j)]["token_ids"],
                        rows[(i, j)]["finish_reason"]) for i in cuts}
            for r in ra.score_arm(seqs, reqs, tok, eos, grader)["rows"]:
                cls[(arm, r["id"], j)] = classify(r)
                if cls[(arm, r["id"], j)] == "wrong_finish":
                    ans[(arm, r["id"], j)] = grader.extract(tok.decode(seqs[r["id"]][0], skip_special_tokens=True))

    other = {}
    for name, f in FULL_ARMS.items():
        if (disc / f).exists():
            seqs = {o["id"]: (o["token_ids"], o["finish_reason"]) for o in lr.read_jsonl(disc / f) if o["id"] in reqs}
            for r in ra.score_arm(seqs, reqs, tok, eos, grader)["rows"]:
                other.setdefault(r["id"], {})[name] = bool(r["correct"])
    moi_plan = {r["id"]: r for r in lr.read_jsonl(disc / "requests_test_moi.jsonl")}
    for name, f in MOI_ARMS.items():
        if (disc / f).exists():
            cont = {o["id"]: o for o in lr.read_jsonl(disc / f)}
            seqs = {i: (moi_plan[i]["prompt_ids"][len(reqs[i]["prompt_ids"]):] + o["token_ids"], o["finish_reason"])
                    for i, o in cont.items()}
            for r in ra.score_arm(seqs, reqs, tok, eos, grader)["rows"]:
                other.setdefault(r["id"], {})[name] = bool(r["correct"])

    groups = {"patch_never": [i for i in cuts if rp[i] == 0],
              "patch_sometimes": [i for i in cuts if 0 < rp[i] < 1],
              "patch_always": [i for i in cuts if rp[i] == 1]}
    out = {"k": K, "groups": {}}
    for g, ids in groups.items():
        c = Counter(cls[(arm, i, j)] for arm in fu.ARMS for i in ids for j in range(K))
        n = sum(c.values())
        per_cut = []
        for i in ids:
            cc = Counter(cls[(arm, i, j)] for arm in fu.ARMS for j in range(K))
            wrong = [ans[(arm, i, j)] for arm in fu.ARMS for j in range(K) if (arm, i, j) in ans]
            top = Counter(wrong).most_common(1)[0][1] / len(wrong) if wrong else None
            kind = ("loop_bound" if cc["loops"] + cc["truncated"] >= 12 else
                    "capability_bound" if cc["wrong_finish"] >= 12 else "mixed")
            solved_elsewhere = any(other.get(i, {}).values())
            per_cut.append({"id": i, "classes": dict(cc), "kind": kind, "restart_rate": rr[i],
                            "wrong_answers": len(wrong), "top_wrong_share": round(top, 3) if top else None,
                            "solved_by_other_arms": [k for k, v in other.get(i, {}).items() if v]})
        kinds = Counter(x["kind"] for x in per_cut)
        tops = [x["top_wrong_share"] for x in per_cut if x["top_wrong_share"] is not None and x["wrong_answers"] >= 4]
        out["groups"][g] = {
            "cuts": len(ids), "futures": dict(c), "futures_share": {k: round(v / max(1, n), 3) for k, v in c.items()},
            "kinds": dict(kinds),
            "solved_by_any_other_arm": sum(bool(x["solved_by_other_arms"]) for x in per_cut),
            "solved_by_other_arm_by_kind": {k: sum(bool(x["solved_by_other_arms"]) for x in per_cut if x["kind"] == k)
                                            for k in kinds},
            "median_top_wrong_share": round(float(np.median(tops)), 3) if tops else None,
            "per_cut": per_cut}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for g, v in out["groups"].items():
        print(f"{g:16s} cuts {v['cuts']:3d}  futures {v['futures_share']}")
        print(f"{'':16s} kinds {v['kinds']}  solved by another arm {v['solved_by_any_other_arm']} "
              f"{v['solved_by_other_arm_by_kind']}  median share of the most common wrong answer {v['median_top_wrong_share']}")


if __name__ == "__main__":
    main()
