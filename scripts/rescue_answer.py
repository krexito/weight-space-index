"""Answer rescue for loop-killed responses (CPU only, no generation).

Grading today scores every loop-killed response as wrong. Some of them stated the right answer before looping.
This reads finished outputs (token ids) and, for each killed response, tries three rescue rules:

  strict  last explicit answer (\\boxed{..}, "answer is X", "final answer: X") BEFORE the loop onset
          (onset = kill point - KILL_WINDOW, i.e. the start of the >=90%-repeated window)
  medium  most frequent explicit answer anywhere before the kill point
  lax     the run's own extractor on the text up to the kill point

A killed response counts as wrong today, so rescue can only raise accuracy; the question is whether the rescued
answers are real (strict) or luck (lax only). Grading and answer equality are the run's own (GSM8K: numeric,
MATH-500: discover_math500_blind's boxed + LaTeX/sympy equality).

  python scripts/rescue_answer.py --work <run work dir> --dataset gsm8k|math --out results.json \
      [--restart <outputs_test_moi_restart.jsonl if not in the work dir>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import moi_loop_router_vllm as lr  # noqa: E402

FIXED = {
    "gsm8k": {"base": "outputs_test_base.jsonl", "blind_fixed": "outputs_test_blind.jsonl",
              "random_fixed": "outputs_test_random.jsonl", "p50c": "outputs_test_p50c.jsonl"},
    "math": {"base": "outputs_test_base.jsonl", "wsi_fixed": "outputs_test_fast.jsonl",
             "random_fixed": "outputs_test_random.jsonl"},
}
MOI = {
    "gsm8k": {"blind_moi": "outputs_test_moi_blind.jsonl", "random_moi": "outputs_test_moi_random.jsonl",
              "restart_moi": "outputs_test_moi_restart.jsonl"},
    "math": {"wsi_moi": "outputs_test_moi_fast.jsonl", "random_moi": "outputs_test_moi_random.jsonl",
             "restart_moi": "outputs_test_moi_restart.jsonl"},
}

BOXED = re.compile(r"\\boxed\{")
PHRASE_NUM = re.compile(r"(?:final answer|answer is|answer:)\s*[:=]?\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", re.I)
PHRASE_TEX = re.compile(r"(?:final answer|answer is|answer:)\s*[:=]?\s*\**\s*\$([^$\n]{1,80})\$", re.I)


def _num(s: str) -> str | None:
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m[-1] if m else None


def _boxed_spans(text: str) -> list[tuple[int, str]]:
    out = []
    for m in BOXED.finditer(text):
        depth, start = 0, m.end()
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    out.append((m.start(), text[start:i].strip()))
                    break
                depth -= 1
    return out


def explicit_answers(text: str, dataset: str = "gsm8k") -> list[tuple[int, str]]:
    """(char position, answer) for every explicit answer statement, in order."""
    boxed = _boxed_spans(text)
    if dataset == "gsm8k":
        out = [(p, _num(b)) for p, b in boxed] + [(m.start(), _num(m.group(1))) for m in PHRASE_NUM.finditer(text)]
    else:
        out = boxed + [(m.start(), m.group(1).strip()) for m in PHRASE_TEX.finditer(text)]
        out += [(m.start(), m.group(1)) for m in PHRASE_NUM.finditer(text)]
    return sorted((p, a) for p, a in out if a)


class Grader:
    def __init__(self, dataset: str):
        self.dataset = dataset
        if dataset == "math":
            import discover_math500_blind as dm

            self.dm = dm

    def grade(self, resp, finish, gold, tok, eos, max_tokens) -> dict:
        if self.dataset == "gsm8k":
            g = lr.grade(resp, finish, gold, tok, eos, max_tokens)
            return {"correct": g["correct"], "killed": g["looped_killed"], "truncated": g["truncated"],
                    "tokens": g["tokens"]}
        g = self.dm.grade_math(resp, finish, gold, tok, eos, max_tokens)
        return {"correct": g["correct"], "killed": g["looped_killed"], "truncated": g["truncated"],
                "tokens": g["n_tokens"]}

    def equal(self, pred, gold) -> bool:
        if self.dataset == "gsm8k":
            return bool(lr.correct(pred, gold))
        return bool(self.dm._math_equal(pred, gold))

    def extract(self, text):
        return lr.extract_answer(text) if self.dataset == "gsm8k" else self.dm._extract_boxed(text)


def rescue(body: list[int], kill: int, tok, grader: Grader | None = None, dataset: str = "gsm8k") -> dict:
    onset = max(kill - lr.KILL_WINDOW, 0)
    pre = tok.decode(body[:onset], skip_special_tokens=True)
    upto = tok.decode(body[:kill], skip_special_tokens=True)
    ex_pre = explicit_answers(pre, dataset)
    ex_all = explicit_answers(upto, dataset)
    lax = grader.extract(upto) if grader else lr.extract_answer(upto)
    return {"onset": onset,
            "strict": ex_pre[-1][1] if ex_pre else None,
            "medium": Counter(a for _, a in ex_all).most_common(1)[0][0] if ex_all else None,
            "lax": lax, "n_explicit_pre": len(ex_pre), "n_explicit_all": len(ex_all)}


def score_arm(seqs: dict, reqs: dict, tok, eos: set[int], grader: Grader) -> dict:
    rows = []
    for i, (resp, finish) in sorted(seqs.items()):
        r = reqs[i]
        g = grader.grade(resp, finish, r["gold"], tok, eos, r["max_tokens"])
        row = {"id": i, "gold": r["gold"], **g}
        if g["killed"]:
            st = lr.analyse(resp, eos)
            rs = rescue(st["body"], st["kill"], tok, grader, grader.dataset)
            row.update(rs)
            for lvl in ("strict", "medium", "lax"):
                row[f"{lvl}_ok"] = grader.equal(rs[lvl], r["gold"]) if rs[lvl] else False
        rows.append(row)
    n = len(rows)
    killed = [x for x in rows if x["killed"]]
    acc = sum(x["correct"] for x in rows) / n
    out = {"n": n, "accuracy": round(acc, 4), "killed": len(killed)}
    for lvl in ("strict", "medium", "lax"):
        got = sum(x[f"{lvl}_ok"] for x in killed)
        stated = sum(x[lvl] is not None for x in killed)
        out[lvl] = {"rescued": got, "answer_given": stated, "wrong_given": stated - got,
                    "accuracy_after": round(acc + got / n, 4), "gain_pts": round(100 * got / n, 2)}
    return {"summary": out, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--restart", help="outputs_test_moi_restart.jsonl, if it is not in the work dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    eos = set(cfg["eos"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    grader = Grader(a.dataset)
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    base = {o["id"]: (o["token_ids"], o["finish_reason"]) for o in lr.read_jsonl(work / "outputs_test_base.jsonl")}

    arms: dict[str, dict] = {}
    for name, f in FIXED[a.dataset].items():
        if (work / f).exists():
            arms[name] = {o["id"]: (o["token_ids"], o["finish_reason"]) for o in lr.read_jsonl(work / f)}
    plan = {r["id"]: r for r in lr.read_jsonl(work / "requests_test_moi.jsonl")}
    for name, f in MOI[a.dataset].items():
        p = Path(a.restart) if (name == "restart_moi" and a.restart) else work / f
        if not p.exists():
            continue
        cont = {o["id"]: o for o in lr.read_jsonl(p)}
        assert set(cont) <= set(plan), f"{p}: ids outside the MoI plan"
        seqs = {}
        for i in reqs:
            if i in cont:
                pre = plan[i]["prompt_ids"][len(reqs[i]["prompt_ids"]):]
                seqs[i] = (pre + cont[i]["token_ids"], cont[i]["finish_reason"])
            else:
                seqs[i] = base[i]
        arms[name] = seqs

    res = {"work": str(work), "dataset": a.dataset, "arms": {}, "rows": {}}
    print(f"{'arm':13s} {'acc':>6s} {'killed':>6s} | {'strict':>14s} {'medium':>14s} {'lax':>14s}")
    for name, seqs in arms.items():
        s = score_arm(seqs, reqs, tok, eos, grader)
        res["arms"][name], res["rows"][name] = s["summary"], s["rows"]
        v = s["summary"]
        cell = lambda l: f"+{v[l]['rescued']:3d} ({v[l]['accuracy_after']:.1%})"  # noqa: E731
        print(f"{name:13s} {v['accuracy']:6.1%} {v['killed']:6d} | {cell('strict'):>14s} {cell('medium'):>14s} {cell('lax'):>14s}")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("Saved", a.out)


if __name__ == "__main__":
    main()
