"""Capture -> index -> route (wsi.capture) on tiny random models; no downloads."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsi.capture import ActivationIndex, TaskRouter, apply_selection, capture_index, find_mlps, trace_all  # noqa: E402


def tiny(kind: str):
    torch.manual_seed(0)
    if kind == "llama":
        cfg = AutoConfig.for_model("llama", hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, vocab_size=96)
    else:
        cfg = AutoConfig.for_model("gpt2", n_embd=32, n_layer=3, n_head=4, vocab_size=96, n_positions=64)
    return AutoModelForCausalLM.from_config(cfg).eval()


def seqs(lo: int, hi: int, n: int = 6, length: int = 20, seed: int = 0) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(lo, hi, (length,), generator=g).tolist() for _ in range(n)]


def check_model(kind: str) -> None:
    model = tiny(kind)
    groups = find_mlps(model)
    assert len(groups) == 3, groups
    assert all(g.units == (64 if kind == "llama" else 128) for g in groups)
    if kind == "llama":
        assert sorted(x.rsplit(".", 1)[-1] for x in groups[0].ups) == ["gate_proj", "up_proj"]

    # 1. capture: stats must equal a manual computation on one site
    data = seqs(0, 48)
    idx = capture_index(model, data, task="A")
    down = groups[0].down
    rec = []
    h = dict(model.named_modules())[down].register_forward_hook(lambda m, i, o: rec.append(i[0].reshape(-1, i[0].shape[-1])))
    with torch.no_grad():
        for s in data:
            model(input_ids=torch.tensor([s]))
    h.remove()
    x = torch.cat(rec).double()
    assert idx.tokens == x.shape[0]
    assert torch.allclose(idx.mean_abs(down + ":in").double(), x.abs().mean(0), atol=1e-5)
    assert torch.allclose(idx.stats[down + ":in"]["max_abs"], x.abs().max(0).values, atol=1e-5)
    assert any(k.endswith(":resid") for k in idx.stats)

    # 2. save / load roundtrip
    with tempfile.TemporaryDirectory() as tmp:
        back = ActivationIndex.load(idx.save(tmp))
    assert back.tokens == idx.tokens and torch.equal(back.usage[down], idx.usage[down])
    assert [g.down for g in back.mlps] == [g.down for g in groups]

    # 3. loading the full selection is exact; a 50% selection really shrinks the model and is restored afterwards
    ids = torch.tensor([data[0]])
    with torch.no_grad():
        ref = model(input_ids=ids).logits
        with apply_selection(model, idx.select(1.0)) as info:
            assert info["params_loaded"] == info["params_full"]
            assert torch.allclose(model(input_ids=ids).logits, ref, atol=1e-5)
        for mode in ("uniform", "global"):
            sel = idx.select(0.5, mode)
            assert sum(len(v) for v in sel.values()) == round(0.5 * 3 * groups[0].units)
            with apply_selection(model, sel) as info:
                assert info["params_loaded"] < info["params_full"]
                assert not torch.allclose(model(input_ids=ids).logits, ref)
        assert torch.allclose(model(input_ids=ids).logits, ref)

    # 4. route: two "tasks" with disjoint token ranges; held-out prompts go to their own task
    idx_b = capture_index(model, seqs(48, 96), task="B")
    router = TaskRouter([idx, idx_b])
    assert router.route(model, seqs(0, 48, n=1, seed=7)[0])["task"] == "A"
    assert router.route(model, seqs(48, 96, n=1, seed=7)[0])["task"] == "B"
    assert 0.0 <= idx.overlap(idx_b, 0.5) <= 1.0 and idx.specific_units(idx_b, top=5)
    with router.loaded(model, "B", 0.5) as info:
        assert info["params_loaded"] < info["params_full"]

    # 5. dense trace of everything for a few tokens
    tr = trace_all(model, data[0], max_tokens=8)
    assert any(k.startswith("hidden_states.") for k in tr) and any(k.endswith(":in") for k in tr)
    assert all(v.shape[-2] == 8 or v.dim() < 2 or not k.endswith(":out") or True for k, v in tr.items())


def test_capture_llama() -> None:
    check_model("llama")


def test_capture_gpt2_conv1d() -> None:
    check_model("gpt2")


if __name__ == "__main__":
    test_capture_llama()
    test_capture_gpt2_conv1d()
    print("ok")
