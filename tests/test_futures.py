"""CPU tests for experiment_futures.py (plan, sensor row on a tiny Llama, report helpers)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_futures as fu  # noqa: E402


def test_plan_rows_paired_seeds_and_blocked_row():
    moi = [{"id": 3, "prompt_ids": [1, 2, 3], "prefix_len": 2, "max_tokens": 50, "seed": 10, "logit_bias": {"7": -1.5}},
           {"id": 9, "prompt_ids": [4, 5], "prefix_len": 1, "max_tokens": 40, "seed": 11, "logit_bias": {}}]
    rows = fu.plan_rows(moi, 3)
    assert len(rows) == 6
    assert [r["seed"] for r in rows if r["id"] == 3] == [10, 1010, 2010]
    assert all(r["logit_bias"][str(fu.ROW)] == fu.BIAS for r in rows)
    assert rows[0]["logit_bias"]["7"] == -1.5 and rows[0]["prompt_ids"] == [1, 2, 3] and rows[0]["max_tokens"] == 50
    assert "7" not in moi[1]["logit_bias"] and str(fu.ROW) not in moi[0]["logit_bias"]  # inputs untouched


def test_add_sensor_in_place_changes_only_the_row(tmp_path, monkeypatch):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path / "m")
    w = np.random.default_rng(0).normal(size=16) * 50
    monkeypatch.setattr(fu, "ROW", 60)
    fu.add_sensor(tmp_path / "m", tmp_path / "s", w)       # copy
    fu.add_sensor(tmp_path / "s", tmp_path / "s", w)       # in place (the patched checkpoint path)
    a = LlamaForCausalLM.from_pretrained(tmp_path / "m").eval()
    b = LlamaForCausalLM.from_pretrained(tmp_path / "s").eval()
    ids = torch.tensor([[1, 5, 9, 3, 7]])
    with torch.no_grad():
        la, lb = a(input_ids=ids).logits[0], b(input_ids=ids).logits[0]
        h = b.model(input_ids=ids).last_hidden_state[0]
    keep = [i for i in range(64) if i != 60]
    assert torch.equal(la[:, keep], lb[:, keep])
    assert torch.allclose(lb[:, 60], h @ torch.from_numpy(w * fu.SCALE).float(), atol=1e-5)


def test_repeat_share_auc_smooth():
    assert fu.repeat_share(list(range(40))) == 0.0
    assert fu.repeat_share([1, 2] * 40) > 0.7
    assert fu.auc([3, 4], [1, 2]) == 1.0 and fu.auc([1, 2], [3, 4]) == 0.0 and fu.auc([1], [1]) == 0.5
    assert np.isnan(fu.auc([], [1]))
    assert np.allclose(fu.smooth(np.array([2.0, 4.0, 6.0]), 2), [2, 3, 5])
    ci = fu.boot_auc({0: [3.0], 1: [4.0]}, {0: [1.0], 1: [2.0]}, [0, 1], n=50)
    assert ci == [1.0, 1.0]
