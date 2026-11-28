"""CPU tests for wsi.token_trace (tiny random Qwen2, no download)."""
from __future__ import annotations

import copy

import pytest
import torch

from wsi.token_trace import TokenTrace, block_states, ids_from_text, make_token_trace, recompute, recompute_states


def _tiny(seed: int = 0):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


IDS = [int(x) for x in torch.randint(0, 97, (70,), generator=torch.Generator().manual_seed(1))]


def test_roundtrip_save_load_recompute_exact(tmp_path) -> None:
    m = _tiny()
    tt = make_token_trace(m, IDS, "tiny-qwen2", meta={"run": "unit"})
    p = tt.save(tmp_path / "a.tokentrace.json")
    assert p.stat().st_size < 50_000
    tt2 = TokenTrace.load(p)
    assert tt2.ids == IDS and tt2.n_layers == 6 and tt2.fp_layers[0] == 0 and tt2.fp_layers[-1] == 5
    st = recompute_states(tt2, _tiny())  # a fresh copy of the same weights
    ref = block_states(m, IDS)
    assert set(st) == set(range(6))
    assert all(torch.equal(st[l], ref[l]) for l in range(6))
    assert tt2.check(m)["fp_rel_l2"] < 1e-6


def test_wrong_weights_or_bf16_are_refused() -> None:
    m = _tiny()
    tt = make_token_trace(m, IDS, "tiny-qwen2")
    with pytest.raises(ValueError, match="does not match"):
        recompute_states(tt, _tiny(seed=3))
    mb = copy.deepcopy(m).to(torch.bfloat16)
    with pytest.raises(ValueError, match="does not match"):
        recompute_states(tt, mb, tol=1e-4)
    assert recompute_states(tt, mb, verify=False)[0].dtype == torch.float32  # escape hatch still works
    with pytest.raises(ValueError, match="fp32"):
        make_token_trace(mb, IDS, "tiny-qwen2")
    assert make_token_trace(mb, IDS, "tiny-qwen2", allow_low_precision=True).dtype == "bf16"


def test_layer_count_mismatch_is_refused() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    tt = make_token_trace(_tiny(), IDS, "tiny-qwen2")
    other = Qwen2ForCausalLM(Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                         num_attention_heads=4, num_key_value_heads=2)).eval()
    with pytest.raises(ValueError, match="blocks"):
        recompute_states(tt, other)


def test_recompute_builds_compact_trace_matching_states() -> None:
    m = _tiny()
    tt = make_token_trace(m, IDS, "tiny-qwen2", capture_kwargs={"resid_quant": "int8", "capture_cache": False})
    trace = recompute(tt, m)
    assert trace.seq_len == len(IDS) and trace.meta["token_trace"]["model"] == "tiny-qwen2"
    ref = block_states(m, IDS)
    errs = []
    for l in range(6):
        r = trace.dequant_resid(l).float().reshape(ref[l].shape)
        errs.append(float((r - ref[l]).norm() / ref[l].norm()))
    assert max(errs) < 0.02, errs  # INT8 error only


def test_ids_from_text() -> None:
    class Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [ord(c) % 97 for c in s]}

    assert ids_from_text(Tok(), "ab", "c") == [ord("a") % 97, ord("b") % 97, ord("c") % 97]
