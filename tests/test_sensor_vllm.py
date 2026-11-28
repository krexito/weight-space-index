"""CPU tests for experiment_sensor_vllm.py (build on a tiny Llama, output parsing, comparison helpers)."""
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_sensor_vllm as sv  # noqa: E402


def test_build_writes_only_the_sensor_row(tmp_path, monkeypatch):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path / "m")
    w = np.random.default_rng(0).normal(size=16) * 50
    np.savez(tmp_path / "d.npz", W=np.stack([w * 0, w]), dirs=np.array(["fold0", "elicit"]))
    monkeypatch.setattr(sv, "ROW", 60)
    sv.cmd_build(types.SimpleNamespace(model=str(tmp_path / "m"), dirs=str(tmp_path / "d.npz"), out=str(tmp_path / "s")))
    a = LlamaForCausalLM.from_pretrained(tmp_path / "m").eval()
    b = LlamaForCausalLM.from_pretrained(tmp_path / "s").eval()
    assert json.loads((tmp_path / "s" / "sensor.json").read_text())["row"] == 60
    ids = torch.tensor([[1, 5, 9, 3, 7]])
    with torch.no_grad():
        la, lb = a(input_ids=ids).logits[0], b(input_ids=ids).logits[0]
        h = b.model(input_ids=ids).last_hidden_state[0]
    keep = [i for i in range(64) if i != 60]
    assert torch.equal(la[:, keep], lb[:, keep])  # every other logit unchanged
    assert torch.allclose(lb[:, 60], h @ torch.from_numpy(w * sv.SCALE).float(), atol=1e-5)


def test_logits_of_and_identical():
    Lp = types.SimpleNamespace
    assert sv.logits_of({5: Lp(logprob=1.25), 7: Lp(logprob=-3.0)}, 5) == 1.25
    assert sv.logits_of({7: Lp(logprob=-3.0)}, 5) is None and sv.logits_of(None, 5) is None
    a = [{"token_ids": [1, 2, 3]}, {"token_ids": [4, 5, 6]}]
    b = [{"token_ids": [1, 2, 3]}, {"token_ids": [4, 9, 6]}]
    r = sv.identical(a, b)
    assert r["identical"] == "1/2" and r["median_first_diff"] == 1.0
