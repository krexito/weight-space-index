"""CPU tests for experiment_veto_moi.py (pure parts + the plan step on a tiny random Llama)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_veto_moi as vm  # noqa: E402


def test_calibrate_orientation_and_threshold():
    rng = np.random.default_rng(0)
    seqs, scores = [], []
    for i in range(6):
        lab = np.zeros(400, np.int8)
        s = rng.normal(0, 0.1, 400)
        if i % 2 == 0:
            lab[200:] = 1
            s[200:] -= 1.0  # loop tokens point AGAINST u: orientation must flip
        seqs.append({"label": lab})
        scores.append(s)
    cal = vm.calibrate(scores, seqs, 0.01)
    assert cal["sign"] == -1.0 and cal["auc_calib"] > 0.95
    assert vm.switch_on(scores[0], 150, cal) is not None and vm.switch_on(scores[0], 150, cal) >= 200 - vm.SMOOTH
    assert vm.switch_on(scores[1], 150, cal) is None  # normal answer: vetoed
    assert vm.switch_on(scores[0], None, cal) is None  # text never fired: nothing to confirm


def test_switch_on_never_before_text_fire():
    cal = {"sign": 1.0, "thr": 0.5}
    s = np.ones(300)
    assert vm.switch_on(s, 120, cal) == 120


def test_plan_rows_match_text_plan_format():
    seqs = [{"id": 3, "body": [5, 6, 7, 5, 9]}, {"id": 4, "body": [1, 2, 3]}]
    reqs = {3: {"prompt_ids": [1, 1], "max_tokens": 100, "seed": 0}, 4: {"prompt_ids": [2], "max_tokens": 100, "seed": 1}}
    rows = vm.plan_rows(seqs, {3: 4, 4: None}, reqs, {}, 1.5)
    assert len(rows) == 1
    r = rows[0]
    assert r["prompt_ids"] == [1, 1, 5, 6, 7, 5] and r["prefix_len"] == 4 and r["max_tokens"] == 96 and r["seed"] == 7
    assert r["logit_bias"] == {"5": -1.5, "6": -1.5, "7": -1.5}


def _jsonl(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_plan_end_to_end(tmp_path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path / "model")
    work = tmp_path / "work"
    work.mkdir()
    rng = np.random.default_rng(0)
    eos = 63
    (work / "config.json").write_text(json.dumps({"eos": [eos], "seed": 0}))
    (work / "frozen.json").write_text(json.dumps({"choice": {"layers": [1, 2], "neurons": 4, "alpha": 2.0}}))
    np.savez(work / "blind_directions.npz", **{f"u{li}": (lambda v: v / np.linalg.norm(v))(rng.normal(size=16))
                                               for li in range(3)})
    for split in ("elicit", "test"):
        reqs, outs = [], []
        for i in range(6):
            body = rng.integers(0, 60, 200).tolist() + ([3, 1, 4, 1, 5, 9, 2, 6] * 130 if i % 2 == 0 else
                                                       rng.integers(0, 60, 300).tolist())
            reqs.append({"id": i, "prompt_ids": rng.integers(0, 60, 10).tolist(), "gold": "1", "max_tokens": 2000,
                         "seed": i})
            outs.append({"id": i, "token_ids": body + [eos], "finish_reason": "stop"})
        _jsonl(work / f"requests_{split}.jsonl", reqs)
        _jsonl(work / f"outputs_{split}{'_base' if split == 'test' else ''}.jsonl", outs)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "experiment_veto_moi.py"), "plan", "--work", str(work),
                        "--model", str(tmp_path / "model"), "--device", "cpu", "--dtype", "float32", "--fpr", "0.2"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    vp = json.loads((work / "veto_plan.json").read_text())
    assert vp["n_test"] == 6 and vp["text_fired"] >= 3
    rows = [json.loads(x) for x in (work / vm.PLAN).read_text().splitlines() if x.strip()]
    assert len(rows) == vp["veto_on"]
    for row in rows:
        on = vp["switch_on"][str(row["id"])]
        assert row["prefix_len"] == on
