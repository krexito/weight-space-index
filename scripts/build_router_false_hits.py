"""Build and check the query set of research/design_router_false_hits.md (CPU only).

Each template of apple/GSM-Symbolic `main` gets two hand-written one-span edits of the same place in the text: a `flip`
(another calculation) and a `reword` (same calculation). benchmarks/router_false_hits/edits.json holds only those short
spans; the dataset text (CC BY-NC-ND, canary-marked) is read at run time and never written to the repo.

  python scripts/build_router_false_hits.py --data <dir with main_*.json pages or main.jsonl>
  python scripts/build_router_false_hits.py --data <dir> --review sheet.jsonl --key key.json   # blind review sheet
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EDITS = ROOT / "benchmarks" / "router_false_hits" / "edits.json"


def load_config(data: Path, config: str) -> dict[int, list[str]]:
    by = defaultdict(dict)
    for f in sorted(data.glob(f"{config}*.json*")):
        txt = f.read_text(encoding="utf-8")
        rows = [json.loads(x) for x in txt.splitlines() if x.strip()] if f.suffix == ".jsonl" else json.loads(txt)
        for r in rows:
            by[r["id"]][r["instance"]] = r["question"]
    return {t: [d[k] for k in sorted(d)] for t, d in sorted(by.items())}


def words(s: str) -> list[str]:
    return re.findall(r"[a-z']+|\d[\d,./]*|[^\sa-z\d]", s.lower())


def apply(q: str, old: str, new: str) -> str:
    if q.count(old) != 1:
        raise ValueError(f"span found {q.count(old)} times: {old!r}")
    return q.replace(old, new)


def n_edited(old: str, new: str) -> int:
    """Words of the new span that the old one does not have."""
    return sum((Counter(words(new)) - Counter(words(old))).values())


def check(tpl: dict[int, list[str]], edits: dict, n_inst: int = 11, max_len: float = 0.05) -> tuple[list[int], dict]:
    """Templates whose two edits match exactly once in instances 0..n_inst-1 and stay within max_len; else the reasons."""
    ok, bad = [], {}
    for t, e in edits.items():
        t = int(t)
        why = [] if t in tpl else ["no such template"]
        if len({e["span"], e["flip"], e["reword"]}) < 3:
            why.append("span, flip and reword must differ")
        for k in ("flip", "reword") if t in tpl else ():
            for i, q in enumerate(tpl[t][:n_inst]):
                try:
                    v = apply(q, e["span"], e[k])
                except ValueError as ex:
                    why.append(f"{k} instance {i}: {ex}")
                    break
                if abs(len(v) - len(q)) / len(q) > max_len:
                    why.append(f"{k} instance {i}: length {len(v) / len(q) - 1:+.1%}")
                    break
        if why:
            bad[t] = why
        else:
            ok.append(t)
    return ok, bad


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--edits", default=str(EDITS))
    ap.add_argument("--review", help="write a label-free review sheet (jsonl)")
    ap.add_argument("--key", help="write the labels of the review sheet (json)")
    a = ap.parse_args(argv)
    tpl = load_config(Path(a.data), "main")
    edits = json.loads(Path(a.edits).read_text(encoding="utf-8"))["edits"]
    ok, bad = check(tpl, edits)
    print(f"templates {len(tpl)}, edits {len(edits)}, clean {len(ok)}")
    for t, why in bad.items():
        print(f"  [{t}] {why}")
    f = [n_edited(edits[str(t)]["span"], edits[str(t)]["flip"]) for t in ok]
    r = [n_edited(edits[str(t)]["span"], edits[str(t)]["reword"]) for t in ok]
    if ok:
        print(f"new words per edit: flip mean {sum(f) / len(f):.2f}, reword mean {sum(r) / len(r):.2f}")
    if a.review:
        rng, rows, key = random.Random(0), [], {}
        for t in ok:
            q = tpl[t][1]
            rows += [(t, k, q, apply(q, edits[str(t)]["span"], edits[str(t)][k])) for k in ("flip", "reword")]
        rng.shuffle(rows)
        with open(a.review, "w", encoding="utf-8") as fh:
            for i, (t, k, q, v) in enumerate(rows):
                key[str(i)] = [t, k]
                fh.write(json.dumps({"i": i, "a": q, "b": v}, ensure_ascii=False) + "\n")
        Path(a.key).write_text(json.dumps(key), encoding="utf-8")
    return {"ok": ok, "bad": bad}


if __name__ == "__main__":
    main()
