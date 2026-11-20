"""CPU tests: wsi.damage_map on a tiny random Llama, and experiment_damage_map.py end to end (own_thinking only)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from wsi.damage_map import damage_map, random_like, token_damage
from wsi.moipatch import Edit, MoIPatch

ROOT = Path(__file__).resolve().parent.parent


def tiny_model(tmp_path=None):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024)
    m = LlamaForCausalLM(cfg).eval()
    if tmp_path is not None:
        m.save_pretrained(tmp_path / "model")
    return m


def test_token_damage_identity():
    lg = torch.randn(5, 10)
    d = token_damage(lg, lg.clone(), torch.arange(5))
    assert np.allclose(d["kl"], 0, atol=1e-6) and np.allclose(d["dnll"], 0, atol=1e-6) and d["agree"].all()


def test_random_like_keeps_shape():
    p = MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=torch.randn(16), alpha=2.0, idx=torch.tensor([1, 3]))])
    r = random_like(p, seed=1)
    assert r.edits[0].module == p.edits[0].module and r.edits[0].alpha == 2.0
    assert torch.equal(r.edits[0].idx, p.edits[0].idx)
    assert abs(float(r.edits[0].u.norm()) - 1.0) < 1e-6
    assert not torch.allclose(r.edits[0].u.float(), p.edits[0].u / p.edits[0].u.norm())


def test_damage_map_zero_and_strong():
    m = tiny_model()
    torch.manual_seed(1)
    u = torch.randn(16)
    patches = {"none": MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=u, alpha=0.0)]),
               "strong": MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=u, alpha=1.0)])}
    w0 = m.model.layers[1].mlp.down_proj.weight.detach().clone()
    rng = np.random.default_rng(0)
    domains = {"a": [rng.integers(0, 64, 40).tolist() for _ in range(3)], "b": [rng.integers(0, 64, 30).tolist()]}
    r = damage_map(m, domains, patches, decode=lambda ids: str(ids), verify_fingerprint=False, n_worst=2)
    assert r["none"]["a"]["kl"] < 1e-6 and r["none"]["a"]["top1_agree"] == 1.0
    assert r["strong"]["a"]["kl"] > r["none"]["a"]["kl"]
    assert r["strong"]["a"]["n_tokens"] == 3 * 39 and r["strong"]["b"]["n_tokens"] == 29
    assert len(r["strong"]["a"]["worst"]) == 2 and "before" in r["strong"]["a"]["worst"][0]
    assert set(r["strong"]["a"]["activity"]) == {"layers.1.mlp.down_proj"}
    assert torch.equal(m.model.layers[1].mlp.down_proj.weight, w0)  # weights restored after the map


def test_script_end_to_end(tmp_path):
    tiny_model(tmp_path)
    MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=torch.randn(16), alpha=1.0)]).save(tmp_path / "p.moipatch")
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.json").write_text(json.dumps({"eos": [0], "model_repo": "toy"}))
    rng = np.random.default_rng(0)
    reqs, outs = [], []
    for i in range(4):
        reqs.append({"id": i, "prompt_ids": [1, 5, 7], "max_tokens": 1000, "gold": "1", "seed": i})
        outs.append({"id": i, "token_ids": rng.integers(2, 64, 300).tolist(), "finish_reason": "stop"})
    (work / "requests_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (work / "outputs_test_base.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outs))
    out = tmp_path / "r.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "experiment_damage_map.py"), "--model", str(tmp_path / "model"),
                        "--patch", str(tmp_path / "p.moipatch"), "--out", str(out), "--work", str(work),
                        "--domains", "own_thinking,prose", "--n-docs", "3", "--seq-len", "128", "--no-fingerprint",
                        "--device", "cpu", "--dtype", "float32"], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    r = json.loads(out.read_text(encoding="utf-8"))
    assert set(r["map"]) == {"p", "p__random"}
    assert set(r["map"]["p"]) == {"own_thinking"} and "prose" in r["skipped"]  # no tokenizer for the toy model
    assert r["map"]["p"]["own_thinking"]["n_tokens"] == 3 * 127
