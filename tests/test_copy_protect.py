"""CPU tests for experiment_copy_protect.py on a tiny random Llama."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_copy_protect as cp  # noqa: E402
from wsi.heads import COPY, IGNORED, LOOP  # noqa: E402


def tiny_model(tmp_path=None):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
    m = LlamaForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if tmp_path is not None:
        m.save_pretrained(tmp_path / "model")
    return m


def test_first_order_matches_actual_change():
    """d_S(i) predicts the real change of the mean NLL when one column is edited with a small alpha."""
    m = tiny_model()
    torch.manual_seed(1)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 64, 60).tolist()
    lab = np.full(59, 0, np.int8)
    lab[30:] = LOOP
    lab[10:20] = COPY
    wins = [{"ids": ids, "labels": lab, "looping": True}]
    li, alpha = 1, 0.05
    down = m.model.layers[li].mlp.down_proj
    u = torch.randn(16)
    u = u / u.norm()
    g = cp.set_gradients(m, {li: down}, {li: u}, wins, torch.device("cpu"))
    uw = {li: (u @ down.weight.float()).double()}
    d = cp.first_order(alpha, uw, g)
    base = cp.window_nll(m, wins, "cpu")[0]
    for col in (3, 17):
        p = cp.make_patch(m, {li: down}, {li: u}, {li: [col]}, alpha, "one")
        r = cp.measure(m, p, wins, [base], "cpu", verify=True)
        for s in ("loop", "copy", "other"):
            pred, act = float(d[s][li][col]), r[s]
            assert abs(pred - act) <= 0.05 * abs(act) + 1e-5, (s, col, pred, act)


def test_window_labels_loop_and_copy():
    body = list(range(10, 40)) + [1, 2, 3, 4, 5] * 60
    q = {"id": 0, "prompt": [7, 8, 9], "body": body, "looping": True, "onset": 30}
    w = cp.window(q, max_len=4096, after=40, warmup=4)
    assert len(w["ids"]) == 3 + 30 + 40
    lab = np.asarray(w["labels"])
    assert (lab[3 + 30 - 1:] == LOOP).all()  # labels are aligned with ids[1:]
    assert (lab[: 3 + 4 - 1] == IGNORED).all()


def test_variants_sizes_and_orig():
    rng = np.random.default_rng(0)
    c_rank = {5: rng.permutation(40).tolist(), 6: rng.permutation(40).tolist()}
    d = {s: {li: rng.normal(size=40) for li in c_rank} for s in ("loop", "copy", "other")}
    v = cp.variants(c_rank, d, 4, {"loop": 10, "copy": 10, "other": 10})
    assert set(v) == {"orig", "gen_skip10", "gen_skip25", "copy_skip10", "copy_skip25", "fo_loop", "fo_spec1", "fo_spec3"}
    for k, cols in v.items():
        for li in c_rank:
            assert len(cols[li]) == 4 and len(set(cols[li])) == 4, k
    assert v["orig"][5] == sorted(c_rank[5][:4])
    worst = int(np.argmax(d["copy"][5]))
    assert worst not in v["copy_skip25"][5]


def _jsonl(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_end_to_end(tmp_path):
    tiny_model(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    rng = np.random.default_rng(0)
    eos = 63
    (work / "config.json").write_text(json.dumps({"eos": [eos], "seed": 0}))
    (work / "frozen.json").write_text(json.dumps({"choice": {"layers": [1, 2], "neurons": 4, "alpha": 2.0}}))
    for split in ("elicit", "test"):
        reqs, outs = [], []
        for i in range(8):
            head = rng.integers(0, 60, 200).tolist()
            body = head + ([3, 1, 4, 1, 5, 9, 2, 6] * 130 if i % 2 == 0 else (rng.integers(0, 60, 40).tolist() * 4))
            reqs.append({"id": i, "prompt_ids": rng.integers(0, 60, 10).tolist(), "gold": "1", "max_tokens": 2000})
            outs.append({"id": i, "token_ids": body + [eos], "finish_reason": "stop"})
        _jsonl(work / f"requests_{split}.jsonl", reqs)
        _jsonl(work / f"outputs_{split}{'_base' if split == 'test' else ''}.jsonl", outs)
    (work / "diagnosis_blind.json").write_text(json.dumps({"neuron_rank": {str(li): rng.permutation(32).tolist()
                                                                          for li in range(3)}}))
    np.savez(work / "blind_directions.npz", **{f"u{li}": (lambda v: v / np.linalg.norm(v))(rng.normal(size=16))
                                               for li in range(3)})
    out = tmp_path / "res.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "experiment_copy_protect.py"), "--work", str(work),
                        "--model", str(tmp_path / "model"), "--out", str(out), "--device", "cpu", "--dtype", "float32",
                        "--n-a", "8", "--n-b", "8", "--max-len", "1024", "--after", "128", "--warmup", "8",
                        "--domains", "own_thinking", "--n-docs", "4"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["layers"] == [1, 2] and res["neurons"] == 4
    assert set(res["summary"]) >= {"orig", "random", "copy_skip10", "fo_spec1"}
    assert res["summary"]["orig"]["loop_vs_orig"] == 1.0
    assert res["tokens_b"]["loop"] > 0 and res["tokens_b"]["copy"] > 0 and res["tokens_b"]["healthy_answers"] == 4
    assert "orig" in res["damage_map"] and "random" in res["damage_map"]
    assert "own_thinking" in res["damage_map"]["orig"]
