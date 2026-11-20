"""CPU smoke test: experiment_self_gate_router.py end to end on a tiny random Llama and synthetic looping answers."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_self_gate_router as sg  # noqa: E402


def test_auc_and_smooth():
    assert sg.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert sg.auc(np.array([0.0]), np.array([1.0])) == 0.0
    assert np.allclose(sg.smooth(np.ones(10), 4), 1.0)


def test_runs_end_to_end(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    from wsi.moipatch import Edit, MoIPatch

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
    mdir = tmp_path / "model"
    LlamaForCausalLM(cfg).save_pretrained(mdir)
    MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=torch.randn(16), alpha=1.0)]).save(tmp_path / "p.moipatch")

    work = tmp_path / "work"
    work.mkdir()
    (work / "config.json").write_text(json.dumps({"eos": [0], "model_repo": "toy"}))
    rng = np.random.default_rng(0)
    pattern = rng.integers(2, 64, 20).tolist()  # every loop repeats the same phrase, so a probe can learn it
    reqs, outs = [], []
    for i in range(8):
        normal = rng.integers(2, 64, 300).tolist()
        body = normal + (pattern * 60 if i >= 2 else rng.integers(2, 64, 600).tolist())
        reqs.append({"id": i, "prompt_ids": [1, 5, 7], "max_tokens": 2000, "gold": "1", "seed": i})
        outs.append({"id": i, "token_ids": body, "finish_reason": "length"})
    (work / "requests_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (work / "outputs_test_base.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outs))

    out = tmp_path / "r.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "experiment_self_gate_router.py"), "--work", str(work),
                        "--model", str(mdir), "--patch", str(tmp_path / "p.moipatch"), "--out", str(out),
                        "--device", "cpu", "--dtype", "float32"], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    r = json.loads(out.read_text())
    assert r["n_half_a"] == r["n_half_b"] == 4
    assert r["loop_rate_b"] == 0.75
    assert set(r["probe_by_layer"]) == {"1", "2"}
    assert "self_layers.1.mlp.down_proj_cos" in r["self_gate"] and "all_modules_cos" in r["self_gate"]
    # a probe fitted on these synthetic loops separates them perfectly; the text router catches every loop
    assert r["probe_best"]["auc"] > 0.9
    assert r["text_router"]["loops_caught"] == 1.0
