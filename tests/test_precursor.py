"""CPU tests: wsi.precursor (labels, class means, early-warning metrics) and experiment_onset_precursor.py end to end."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from wsi.precursor import IGNORED, NEGATIVE, PRECURSOR, ClassMeans, early_warning, precursor_labels, smooth

ROOT = Path(__file__).resolve().parent.parent


def test_labels():
    lab = precursor_labels(20, 12, window=4, gap=2)
    assert (lab[12:] == IGNORED).all()
    assert (lab[8:12] == PRECURSOR).all()
    assert (lab[6:8] == IGNORED).all()
    assert (lab[:6] == NEGATIVE).all()
    assert (precursor_labels(5, None) == NEGATIVE).all()
    assert (precursor_labels(5, 2, window=8, gap=4)[:2] == PRECURSOR).all()  # window clipped at 0


def test_smooth_is_causal():
    x = np.zeros(10)
    x[5] = 10
    s = smooth(x, 4)
    assert (s[:5] == 0).all() and s[5] > 0


def test_class_means_direction():
    torch.manual_seed(0)
    resid = torch.randn(2, 30, 8)
    resid[:, 10:20, 3] += 5.0  # the precursor tokens are shifted along dim 3
    lab = np.full(30, NEGATIVE, np.int8)
    lab[10:20] = PRECURSOR
    lab[20:] = IGNORED
    m = ClassMeans()
    m.add(resid, lab)
    d = m.directions()
    assert d.shape == (2, 8)
    assert d[:, 3].abs().min() > 0.8


def test_early_warning_metrics():
    rng = np.random.default_rng(0)

    def answer(anchor, n=600):
        s = rng.normal(0, 1, n)
        if anchor is not None:
            s[anchor - 100:anchor] += 6.0  # signal only in the precursor window
        return s, precursor_labels(n, anchor, window=100, gap=50)

    a = [answer(400) for _ in range(6)] + [answer(None) for _ in range(6)]
    b_anchor = [400] * 6 + [None] * 6
    b = [answer(x) for x in b_anchor]
    r = early_warning([s for s, _ in a], [lab for _, lab in a], [s for s, _ in b], [lab for _, lab in b], b_anchor,
                      target_fpr=0.1, smooth_w=8)
    assert r["auc"] > 0.95 and r["sign"] == 1.0
    assert r["warned_useful"] == 1.0 and 0 < r["median_lead_useful"] <= 100
    assert r["normal_answers_fired"] <= 0.5
    assert r["n_fail"] == 6 and r["n_normal"] == 6
    # token-level calibration on long answers: noise crosses 1% of tokens early, so alarms come far too soon
    t = early_warning([s for s, _ in a], [lab for _, lab in a], [s for s, _ in b], [lab for _, lab in b], b_anchor,
                      target_fpr=0.01, level="token", smooth_w=8)
    assert t["normal_answers_fired"] >= r["normal_answers_fired"]


def test_runs_end_to_end(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
    mdir = tmp_path / "model"
    LlamaForCausalLM(cfg).save_pretrained(mdir)
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.json").write_text(json.dumps({"eos": [0], "model_repo": "toy"}))
    rng = np.random.default_rng(0)
    pattern = rng.integers(2, 32, 20).tolist()
    reqs, outs = [], []
    for i in range(8):
        normal = rng.integers(2, 32, 400).tolist()
        if i >= 2:  # a loop preceded by 300 tokens from a different vocabulary: a learnable precursor
            body = normal + rng.integers(40, 64, 300).tolist() + pattern * 60
        else:
            body = normal + rng.integers(2, 32, 900).tolist()
        reqs.append({"id": i, "prompt_ids": [1, 5, 7], "max_tokens": 3000, "gold": "1", "seed": i})
        outs.append({"id": i, "token_ids": body, "finish_reason": "length"})
    (work / "requests_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (work / "outputs_test_base.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outs))

    out = tmp_path / "r.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "experiment_onset_precursor.py"), "--work", str(work),
                        "--model", str(mdir), "--out", str(out), "--window", "256", "--gap", "32",
                        "--device", "cpu", "--dtype", "float32"], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    r = json.loads(out.read_text())
    assert r["n_half_a"] == r["n_half_b"] == 4 and r["loop_rate_b"] == 0.75
    assert set(r["by_layer"]["precursor"]) == {"1", "2"}
    assert {"position", "text_rep64", "text_rep256"} <= set(r["baselines"])
    best = r["best"]["precursor"]
    assert best["auc"] > 0.9  # the synthetic precursor vocabulary is easy to see
    assert best["warned_useful"] > 0  # and it fires before the text detector on at least one loop
