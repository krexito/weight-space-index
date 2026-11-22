"""CPU tests: scripts/experiment_router_false_hits.py (records and split, metric, decision rule, toy end to end + resume)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import experiment_router_false_hits as ex  # noqa: E402
from test_eval_refusal_blind import toy  # noqa: E402


def make_data(d: Path, n: int = 16):
    """Templates whose fixed words are in the toy vocabulary; numbers are words w20-w29 (so they vary per instance)."""
    d.mkdir()
    edits = {}
    rows = {"main": [], "p1": [], "p2": []}
    for t in range(n):
        for i in range(6):
            q = f"w{t} w{t + 17} w{20 + i} w{30 + i} w40 w41 w42"
            rows["main"].append({"id": t, "instance": i, "original_id": 100 + t, "question": q})
            rows["p1"].append({"id": t, "instance": i, "original_id": 100 + t, "question": q + " w43 w44 w45"})
            if t % 2 == 0:
                rows["p2"].append({"id": t, "instance": i, "original_id": 100 + t, "question": q + " w43 w44 w45 w46 w47"})
        edits[str(t)] = {"span": "w41 w42", "flip": "w50 w42", "reword": "w51 w42"}
    for c, r in rows.items():
        (d / f"{c}.jsonl").write_text("\n".join(map(json.dumps, r)), encoding="utf-8")
    (d / "edits.json").write_text(json.dumps({"edits": edits}), encoding="utf-8")
    return edits


def test_records_split_by_template(tmp_path):
    edits = make_data(tmp_path / "d")
    data = ex.build_records(tmp_path / "d", edits, n_inst=5)
    assert not data["cal"] & data["test"] and len(data["cal"]) == len(data["test"]) == 8
    kinds = {k: sum(r["kind"] == k for r in data["records"]) for k in ex.HITS + ex.MISSES}
    assert kinds["number"] == kinds["reword"] == kinds["flip"] == 16 * 5 and kinds["p1"] == 80 and kinds["p2"] == 40
    for r in data["records"]:
        assert r["cal"] == (r["t"] in data["cal"])
        assert r["text"] != r["key"] or r["kind"] == "other"
        if r["kind"] == "other":  # "other" queries come from a template on the same side of the split
            assert not r["text"].startswith(f"w{r['t']} ")


def test_metric_separates_flips_from_hits():
    g = torch.Generator().manual_seed(0)
    n, d = 120, 40
    noise = lambda: torch.randn(n, d, generator=g, dtype=torch.float64)  # noqa: E731
    flipdir = torch.zeros(d, dtype=torch.float64)
    flipdir[3] = 4.0
    D = torch.cat([noise(), noise(), noise() + flipdir])
    kind = np.array(["number"] * n + ["reword"] * n + ["flip"] * n)
    s = ex.fit_metric(D, kind, rank=1, lam=0.1)(D)
    assert ex.auc(s[kind != "flip"], s[kind == "flip"]) > 0.9


def test_decision_rule():
    def arm(hit, flip, per):
        return {"hit": {"number": hit, "reword": hit, "all": hit}, "false_hit": {"flip": flip}, "_flip_by_template": per,
                "auc_vs_flip": {}}

    t = range(40)
    res = {"m/hidden_router": arm(0.95, 0.2, {i: 0.2 for i in t}), "embedding_cosine": arm(0.95, 0.6, {i: 0.6 for i in t}),
           "cheat": arm(0.5, 0.0, {i: 0.0 for i in t}), "m/ask_model": arm(0.95, 0.55, {i: 0.55 for i in t}),
           "m/hidden_cosine": arm(0.95, 0.9, {i: 0.9 for i in t}),
           "m/control_shuffled": {"info": {"auc_reword_vs_flip_mean": 0.5}}}
    d = ex.decide(res, "m/", ["embedding_cosine", "cheat"])
    assert d["best_baseline"] == "m/ask_model" and d["passed"] and d["gap_points"] == 35.0  # "cheat" keeps too few hits to count
    res["m/control_shuffled"]["info"]["auc_reword_vs_flip_mean"] = 0.8
    assert not ex.decide(res, "m/", ["embedding_cosine"])["passed"]
    res["m/ask_model"] = arm(0.95, 0.22, {i: 0.22 for i in t})
    assert not ex.decide(res, "m/", ["embedding_cosine"])["gap_ok"]


def test_end_to_end_toy_and_resume(tmp_path):
    make_data(tmp_path / "d")
    m = toy(tmp_path)
    args = ["--data", str(tmp_path / "d"), "--edits", str(tmp_path / "d" / "edits.json"), "--out-dir", str(tmp_path / "o"),
            "--out", str(tmp_path / "r.json"), "--models", str(m), "--embed", str(m), "--rerank", str(m), "--n-inst", "5",
            "--layer-step", "1", "--bs", "8", "--dtype", "float32", "--device", "cpu"]
    r = ex.main(args)
    s = json.loads((tmp_path / "r.json").read_text())
    need = {"text_logistic", "changed_words", "edited_words_oracle", "embedding_cosine", "embedding_metric", "reranker",
            "model/hidden_router", "model/hidden_cosine", "model/control_shuffled", "model/control_random", "model/ask_model",
            "model/pair_probe"}
    assert need <= set(s["arms"]) and set(s["decision"]) == {str(m)}
    assert s["arms"]["model/hidden_router"]["info"]["rank"] in (4, 8, 16, 32)
    assert "w41" not in (tmp_path / "r.json").read_text()  # no dataset text in the result
    assert not any("w41" in x.name for x in (tmp_path / "o").iterdir())
    # resume: the caches alone must give the same arms (no model is loaded again)
    (tmp_path / "model").rename(tmp_path / "gone")
    r2 = ex.main(args)
    assert r2["arms"] == r["arms"]
