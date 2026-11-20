"""Follow-ups to rescue_answer.py (CPU, no generation), per arm:

  fire_stop  policy "at the router fire point, if an explicit answer was already stated, stop and give it; otherwise
             let the response run (then rescue at the kill)": accuracy and generated tokens vs today.
  truncated  strict rescue applied to responses cut by the token budget (not loop-killed).
  precision  how often the strict answer is right, split by confidence signals
             (strict == medium; the same answer stated >= 2 times before onset).

  python scripts/rescue_policy.py --work <dir> --dataset gsm8k|math --tokenizer <tok> [--restart f] --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import moi_loop_router_vllm as lr  # noqa: E402
import rescue_answer as ra  # noqa: E402


def load_arms(work: Path, dataset: str, restart: str | None) -> dict:
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    base = {o["id"]: (o["token_ids"], o["finish_reason"]) for o in lr.read_jsonl(work / "outputs_test_base.jsonl")}
    arms = {}
    for name, f in ra.FIXED[dataset].items():
        if (work / f).exists():
            arms[name] = {o["id"]: (o["token_ids"], o["finish_reason"]) for o in lr.read_jsonl(work / f)}
    plan = {r["id"]: r for r in lr.read_jsonl(work / "requests_test_moi.jsonl")}
    for name, f in ra.MOI[dataset].items():
        p = Path(restart) if (name == "restart_moi" and restart) else work / f
        if not p.exists():
            continue
        cont = {o["id"]: o for o in lr.read_jsonl(p)}
        arms[name] = {i: ((plan[i]["prompt_ids"][len(reqs[i]["prompt_ids"]):] + cont[i]["token_ids"],
                           cont[i]["finish_reason"]) if i in cont else base[i]) for i in reqs}
    return reqs, arms


def analyse_arm(seqs, reqs, tok, eos, g: ra.Grader) -> dict:
    n = len(seqs)
    today = rescued = policy = 0
    tok_today = tok_policy = 0
    stops = stop_right = 0
    trunc = trunc_given = trunc_right = 0
    prec = Counter()
    for i, (resp, finish) in seqs.items():
        r = reqs[i]
        gr = g.grade(resp, finish, r["gold"], tok, eos, r["max_tokens"])
        st = lr.analyse(resp, eos)
        body, kill, fire = st["body"], st["kill"], st["fire"]
        end = kill if kill is not None else len(body)
        ok_today = gr["correct"]
        ok_resc = ok_today
        if kill is not None:
            rs = ra.rescue(body, kill, tok, g, g.dataset)
            right = bool(rs["strict"]) and g.equal(rs["strict"], r["gold"])
            ok_resc = right
            if rs["strict"]:
                ex = ra.explicit_answers(tok.decode(body[: rs["onset"]], skip_special_tokens=True), g.dataset)
                times = sum(1 for _, x in ex if x == rs["strict"])
                key_agree = "agree" if rs["strict"] == rs["medium"] else "disagree"
                key_rep = "stated>=2" if times >= 2 else "stated=1"
                for k in (key_agree, key_rep, "all"):
                    prec[(k, right)] += 1
        elif gr["truncated"]:
            trunc += 1
            ex = ra.explicit_answers(tok.decode(body, skip_special_tokens=True), g.dataset)
            if ex:
                trunc_given += 1
                if g.equal(ex[-1][1], r["gold"]):
                    trunc_right += 1
                    ok_resc = True
        today += ok_today
        rescued += ok_resc
        tok_today += end
        # fire-stop policy
        if fire is not None:
            ex = ra.explicit_answers(tok.decode(body[:fire], skip_special_tokens=True), g.dataset)
            if ex:
                stops += 1
                ok = g.equal(ex[-1][1], r["gold"])
                stop_right += ok
                policy += ok
                tok_policy += fire
                continue
        policy += ok_resc
        tok_policy += end
    p = {k: {"n": prec[(k, True)] + prec[(k, False)],
             "precision": round(prec[(k, True)] / max(prec[(k, True)] + prec[(k, False)], 1), 3)}
         for k in ("all", "agree", "disagree", "stated>=2", "stated=1")}
    return {"n": n, "acc_today": round(today / n, 4), "acc_rescue_kill_and_trunc": round(rescued / n, 4),
            "truncated": {"n": trunc, "answer_given": trunc_given, "rescued": trunc_right},
            "fire_stop": {"acc": round(policy / n, 4), "stopped_at_fire": stops, "stopped_right": stop_right,
                          "tokens_today": tok_today, "tokens_policy": tok_policy,
                          "tokens_saved_pct": round(100 * (1 - tok_policy / max(tok_today, 1)), 1)},
            "strict_precision": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--restart")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    work = Path(a.work)
    eos = set(json.loads((work / "config.json").read_text())["eos"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    g = ra.Grader(a.dataset)
    reqs, arms = load_arms(work, a.dataset, a.restart)
    res = {}
    print(f"{'arm':13s} {'today':>6s} {'+resc':>6s} {'trunc+':>6s} | {'fireStop':>8s} {'stops':>9s} {'tok-%':>6s} | "
          f"{'prec':>5s} {'agree':>11s} {'disagr':>11s} {'rep>=2':>11s}")
    for name, seqs in arms.items():
        v = analyse_arm(seqs, reqs, tok, eos, g)
        res[name] = v
        fs, sp = v["fire_stop"], v["strict_precision"]
        c = lambda k: f"{sp[k]['precision']:.0%}/{sp[k]['n']}"  # noqa: E731
        print(f"{name:13s} {v['acc_today']:6.1%} {v['acc_rescue_kill_and_trunc']:6.1%} {v['truncated']['rescued']:6d} | "
              f"{fs['acc']:8.1%} {fs['stopped_right']:4d}/{fs['stopped_at_fire']:<4d} {fs['tokens_saved_pct']:6.1f} | "
              f"{sp['all']['precision']:5.0%} {c('agree'):>11s} {c('disagree'):>11s} {c('stated>=2'):>11s}")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("Saved", a.out)


if __name__ == "__main__":
    main()
