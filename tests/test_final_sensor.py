"""CPU tests for experiment_final_sensor.py on a tiny random Llama (score + analyze end to end)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_final_sensor as fs  # noqa: E402

EOS = 63


def _jsonl(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _work(tmp_path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path / "model")
    work = tmp_path / "work"
    work.mkdir()
    rng = np.random.default_rng(0)
    (work / "config.json").write_text(json.dumps({"eos": [EOS], "seed": 0}))
    (work / "frozen.json").write_text(json.dumps({"choice": {"layers": [1, 2], "neurons": 4, "alpha": 2.0}}))
    np.savez(work / "blind_directions.npz", **{f"u{li}": (lambda v: v / np.linalg.norm(v))(rng.normal(size=16))
                                               for li in range(3)})
    pid = 0
    for name, rq, op in fs.SPLITS:
        reqs, outs = [], []
        for i in range(6):
            body = rng.integers(0, 60, 150).tolist() + ([3, 1, 4, 1, 5, 9, 2, 6] * 120 if i % 2 == 0 else
                                                       rng.integers(0, 60, 250).tolist())
            reqs.append({"id": i, "prompt_ids": [pid % 60, (pid // 60) % 60, 7], "gold": "1", "max_tokens": 2000,
                         "seed": i})
            outs.append({"id": i, "token_ids": body + [EOS], "finish_reason": "stop"})
            pid += 1
        _jsonl(work / rq, reqs)
        _jsonl(work / op, outs)
    return work


def test_train_of_and_folds():
    seqs = [{"group": f"g{i}", "split": "elicit" if i < 3 else "test"} for i in range(10)]
    fo = fs.fold_map(seqs, 0)
    assert sorted(set(fo.values())) == list(range(fs.FOLDS))
    q = seqs[0]
    assert fs.train_of("elicit", q, fo) and not fs.train_of("elicit", seqs[5], fo)
    assert not fs.train_of(f"fold{fo[q['group']]}", q, fo)


def test_score_and_analyze(tmp_path, monkeypatch):
    import torch
    from transformers import AutoModelForCausalLM

    work = _work(tmp_path)
    out = tmp_path / "s.npz"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "experiment_final_sensor.py"), "score", "--work",
                        str(work), "--model", str(tmp_path / "model"), "--out", str(out), "--cpu", "--chunk", "128"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    z = np.load(out)
    W = z["__W"]
    assert W.shape == (len(fs.DIRS), 16)
    keys = [k[len("sensor|"):] for k in z.files if k.startswith("sensor|")]
    assert len(keys) == 24 and all(f"veto|{k}" in z.files for k in keys)

    # the sensor value must be exactly what an lm_head row equal to w would output on the final hidden state
    m = AutoModelForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.float32).eval()
    q = next(x for x in fs.load_all(work, {EOS}) if x["key"] == keys[0])
    ids = torch.tensor([q["prompt"] + q["body"]])
    with torch.no_grad():
        h = m.model(input_ids=ids).last_hidden_state[0, len(q["prompt"]):]
        m.lm_head.weight[5] = torch.from_numpy(W[0])  # put the sensor in a vocabulary row
        logit = m(input_ids=ids).logits[0, len(q["prompt"]):, 5]
    s = z[f"sensor|{keys[0]}"]
    assert s.shape == (len(fs.DIRS), len(q["body"]))
    assert np.allclose(s[0], (h @ torch.from_numpy(W[0])).numpy(), atol=1e-4)
    assert np.allclose(s[0], logit.numpy(), atol=1e-4)

    # analyze in-process with grading stubbed (the toy model has no tokenizer)
    import types

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: None))
    monkeypatch.setattr(fs.lr, "grade", lambda *_a, **_k: {"correct": True})
    res_path = tmp_path / "r.json"
    fs.cmd_analyze(types.SimpleNamespace(work=str(work), scores=str(out), tokenizer="x", out=str(res_path), seed=0,
                                         boot=20))
    res = json.loads(res_path.read_text())
    assert res["data"]["answers"] == 24 and res["data"]["loops"] == 12
    for arm in ("veto_L13_14", "final_sensor"):
        assert set(res["cv"][arm]) == set(fs.RULES)
        assert 0.0 <= res["token_auc_heldout"][arm]["auc"] <= 1.0
    # the text rule does not depend on the score: identical in both arms
    assert res["cv"]["veto_L13_14"]["text"] == res["cv"]["final_sensor"]["text"]
