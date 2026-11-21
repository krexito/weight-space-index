"""CPU tests: wsi.heads (head discovery, ablation, copy labels) and experiment_copy_heads.py end to end."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from wsi.heads import COPY, IGNORED, LOOP, OTHER, copy_labels, find_heads, head_ablation, set_means, token_nll

ROOT = Path(__file__).resolve().parent.parent


def tiny(tmp_path=None):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
    m = LlamaForCausalLM(cfg).eval()
    if tmp_path is not None:
        m.save_pretrained(tmp_path / "model")
    return m


def test_copy_labels():
    ids = [1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5, 7, 7, 7, 7]
    lab = copy_labels(ids, score_from=0, loop_from=None, k=4)
    assert lab[10] == COPY  # [1,2,3,4] seen before, followed by 5 then and now
    assert lab[11] == OTHER  # [2,3,4,5] was followed by 9 before, now 7
    assert (lab[:4] == IGNORED).all()  # no k-gram yet
    lab = copy_labels(ids, score_from=0, loop_from=12, k=4, ignore_before_loop=2)
    assert (lab[12:] == LOOP).all() and (lab[10:12] == IGNORED).all() and lab[5] == OTHER


def test_find_heads_and_ablation():
    m = tiny()
    info = find_heads(m)
    assert set(info) == {"model.layers.0.self_attn.o_proj", "model.layers.1.self_attn.o_proj"}
    assert info["model.layers.0.self_attn.o_proj"]["n_heads"] == 4
    ids = torch.randint(0, 64, (40,))
    base = token_nll(m, ids)
    with head_ablation(m, {"model.layers.1.self_attn.o_proj": [0, 2]}, info):
        abl = token_nll(m, ids)
    assert not torch.allclose(base, abl)
    assert torch.allclose(base, token_nll(m, ids))  # hooks removed
    all_heads = {n: list(range(v["n_heads"])) for n, v in info.items()}
    seen = {}
    h = m.model.layers[0].self_attn.o_proj.register_forward_hook(lambda _m, _a, o: seen.setdefault("y", o))
    with head_ablation(m, all_heads, info):
        token_nll(m, ids)
    h.remove()
    assert torch.count_nonzero(seen["y"]) == 0  # every head of layer 0 zeroed: o_proj (no bias) outputs zeros


def test_set_means():
    r = set_means(np.array([1.0, 2.0, 3.0, 4.0]), np.array([LOOP, LOOP, COPY, IGNORED]))
    assert r == {"loop": 1.5, "copy": 3.0, "other": None}


def test_script_end_to_end(tmp_path):
    from wsi.moipatch import Edit, MoIPatch

    tiny(tmp_path)
    MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=torch.randn(16), alpha=1.0)]).save(tmp_path / "p.moipatch")
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.json").write_text(json.dumps({"eos": [0], "model_repo": "toy"}))
    rng = np.random.default_rng(0)
    pattern = rng.integers(2, 64, 20).tolist()
    reqs, outs = [], []
    for i in range(8):
        normal = rng.integers(2, 64, 300).tolist()
        body = normal + normal[:40] + (pattern * 60 if i >= 2 else rng.integers(2, 64, 600).tolist())
        reqs.append({"id": i, "prompt_ids": [1, 5, 7], "max_tokens": 2000, "gold": "1", "seed": i})
        outs.append({"id": i, "token_ids": body, "finish_reason": "length"})
    (work / "requests_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (work / "outputs_test_base.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outs))
    out = tmp_path / "r.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "experiment_copy_heads.py"), "--work", str(work),
                        "--model", str(tmp_path / "model"), "--out", str(out), "--patch", str(tmp_path / "p.moipatch"),
                        "--no-fingerprint", "--max-len", "512", "--after", "128", "--warmup", "8", "--n-a", "4", "--n-b", "4",
                        "--device", "cpu", "--dtype", "float32"], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    r = json.loads(out.read_text())
    assert r["n_heads"] == 8 and len(r["single_heads"]) == 8
    assert r["tokens_b"]["loop"] > 0 and r["tokens_b"]["copy"] > 0
    assert set(r["combos_b"]) == {"1", "2", "4", "8"}
    assert len(r["combos_b"]["8"]["heads"]) == 8
    assert {"loop", "copy", "other"} <= set(r["patch_b"])
