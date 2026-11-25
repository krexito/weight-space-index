"""Outcome rows for scripts/experiment_rescue_inside.py: per routed discover_blind test problem, base / blind_moi / random_moi grades.

Run: python scripts/experiment_rescue_outcomes.py <discover_blind folder> rescue_rows.json
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
import moi_loop_router_vllm as lr  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

W = Path(sys.argv[1])
cfg = json.loads((W / "config.json").read_text())
eos = set(cfg["eos"])
tok = AutoTokenizer.from_pretrained(str(R / "models/qwen35-0.8b-full"))
reqs = {r["id"]: r for r in lr.read_jsonl(W / "requests_test.jsonl")}
plan = {r["id"]: r for r in lr.read_jsonl(W / "requests_test_moi.jsonl")}
base = {o["id"]: o for o in lr.read_jsonl(W / "outputs_test_base.jsonl")}
rows = []
for arm, f in (("blind", "outputs_test_moi_blind.jsonl"), ("random", "outputs_test_moi_random.jsonl")):
    cont = {o["id"]: o for o in lr.read_jsonl(W / f)}
    for i, p in plan.items():
        if i not in cont:
            continue
        pre = p["prompt_ids"][len(reqs[i]["prompt_ids"]):]
        g = lr.grade(pre + cont[i]["token_ids"], cont[i]["finish_reason"], reqs[i]["gold"], tok, eos, reqs[i]["max_tokens"])
        gb = lr.grade(base[i]["token_ids"], base[i]["finish_reason"], reqs[i]["gold"], tok, eos, reqs[i]["max_tokens"])
        rows.append({"id": i, "arm": arm, "cut": len(pre), "prompt_len": len(reqs[i]["prompt_ids"]),
                     "cont_len": len(cont[i]["token_ids"]), "correct": g["correct"], "killed": g["looped_killed"],
                     "truncated": g["truncated"], "base_correct": gb["correct"], "base_killed": gb["looped_killed"],
                     "logit_bias": bool(p.get("logit_bias")), "plan_keys": sorted(p)})
json.dump(rows, open(sys.argv[2], "w"))
from collections import Counter
for arm in ("blind", "random"):
    rr = [r for r in rows if r["arm"] == arm]
    print(arm, "routed", len(rr), "correct", sum(r["correct"] for r in rr), "killed", sum(r["killed"] for r in rr),
          "trunc", sum(r["truncated"] for r in rr), "base_correct", sum(r["base_correct"] for r in rr),
          "base_killed", sum(r["base_killed"] for r in rr))
print("plan keys", rows[0]["plan_keys"], "cut median", sorted(r["cut"] for r in rows)[len(rows) // 2],
      "prompt+cut max", max(r["prompt_len"] + r["cut"] for r in rows))
b = {r["id"]: r for r in rows if r["arm"] == "blind"}
q = {r["id"]: r for r in rows if r["arm"] == "random"}
print("blind x random correct:", Counter((b[i]["correct"], q[i]["correct"]) for i in b if i in q))
