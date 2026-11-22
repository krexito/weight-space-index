"""CPU tests: scripts/build_router_false_hits.py and scripts/router_false_hits_gate.py (synthetic templates only)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_router_false_hits as b  # noqa: E402
import router_false_hits_gate as g  # noqa: E402


def word(i: int) -> str:
    return "q" + "".join(chr(97 + (i // 26 ** k) % 26) for k in range(3))


def toy(n: int = 24, leak: bool = False):
    tpl, edits = {}, {}
    for t in range(n):
        tpl[t] = [f"Person {i} of town{t} buys {i + 3} boxes and sells {i + 1} boxes of item{t}. How many boxes remain today?"
                  for i in range(12)]
        edits[str(t)] = {"span": "boxes remain today?", "flip": f"boxes {'half' if leak else word(2 * t)} today?",
                         "reword": f"boxes {'stay' if leak else word(2 * t + 1)} today?"}
    return tpl, edits


def test_check_accepts_clean_and_reports_reasons():
    tpl, edits = toy(3)
    assert b.check(tpl, edits) == ([0, 1, 2], {})
    edits["1"]["span"] = "boxes"  # three times in the text
    edits["2"]["flip"] = "boxes remain today, and please also explain every single step you take?"
    edits["9"] = dict(edits["0"])
    ok, bad = b.check(tpl, edits)
    assert ok == [0] and "found 3 times" in bad[1][0] and "length" in bad[2][0] and bad[9] == ["no such template"]


def test_review_sheet_has_no_labels(tmp_path):
    tpl, edits = toy(4)
    (tmp_path / "main_0.json").write_text(json.dumps(
        [{"id": t, "instance": i, "question": q} for t, qs in tpl.items() for i, q in enumerate(qs)]), encoding="utf-8")
    (tmp_path / "e.json").write_text(json.dumps({"edits": edits}), encoding="utf-8")
    b.main(["--data", str(tmp_path), "--edits", str(tmp_path / "e.json"), "--review", str(tmp_path / "s.jsonl"),
            "--key", str(tmp_path / "k.json")])
    rows = [json.loads(x) for x in (tmp_path / "s.jsonl").read_text(encoding="utf-8").splitlines()]
    key = json.loads((tmp_path / "k.json").read_text())
    assert len(rows) == 8 == len(key) and all(set(r) == {"i", "a", "b"} for r in rows)


def test_gate_flags_a_vocabulary_leak_and_passes_without_one():
    tpl, edits = toy(24, leak=True)
    leaky = g.run(tpl, edits, sorted(tpl), n_splits=5)
    assert leaky["words"]["auc_mean"] > 0.95 and not leaky["go"]
    tpl, edits = toy(24, leak=False)
    clean = g.run(tpl, edits, sorted(tpl), n_splits=5)
    assert clean["words"]["auc_mean"] <= 0.75 and clean["go"]


def test_committed_edits_are_well_formed():
    d = json.loads(b.EDITS.read_text(encoding="utf-8"))
    assert len(d["edits"]) >= 80 and not set(d["edits"]) & set(d["dropped"])
    for e in d["edits"].values():
        assert set(e) == {"span", "flip", "reword"} and len(set(e.values())) == 3
