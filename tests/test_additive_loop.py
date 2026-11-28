"""CPU unit tests for `wsi.additive_loop` (`AdditiveLoopedMoI`, HF DynamicCache & persistence)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi import FORMAT_ADDITIVE_LOOP, AdditiveLoopConfig, AdditiveLoopedMoI  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402


def _make_tiny_qwen2(seed: int = 17) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=48,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def test_n_loops_zero_bit_exact_passthrough() -> None:
    model = _make_tiny_qwen2()
    mod = AdditiveLoopedMoI.from_model(model, insert_layer=2, n_loops=2)
    x = torch.tensor([[1, 5, 9, 13, 17, 21]], dtype=torch.long)

    with torch.no_grad():
        base_logits = model(x, use_cache=False).logits
        with mod.attach(model, n_loops=0):
            pass_logits = model(x, use_cache=False).logits

    assert float((pass_logits - base_logits).abs().max().item()) == 0.0


def test_step0_identity_and_two_adamw_steps_update_all_components() -> None:
    model = _make_tiny_qwen2(seed=21)
    mod = AdditiveLoopedMoI.from_model(model, insert_layer=2, n_loops=2, max_steps=6, d_loop_key=16)
    x = torch.tensor([[2, 4, 6, 8, 10, 12], [3, 5, 7, 9, 11, 13]], dtype=torch.long)
    targets = torch.tensor([[4, 6, 8, 10, 12, 14], [5, 7, 9, 11, 13, 15]], dtype=torch.long)

    with torch.no_grad():
        base_logits = model(x, use_cache=False).logits
        with mod.attach(model, n_loops=2):
            step0_logits = model(x, use_cache=False).logits
    assert float((step0_logits - base_logits).abs().max().item()) == 0.0

    init_gate = mod.out_gate.detach().clone()
    init_blk = next(mod.blocks.parameters()).detach().clone()
    init_se = mod.step_embed.weight.detach().clone()
    init_q = mod.q_agg.detach().clone()

    opt = torch.optim.AdamW(mod.parameters(), lr=5e-2)
    losses: list[float] = []
    with mod.attach(model, n_loops=2):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            logits = model(x, use_cache=False).logits
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

    assert losses[-1] < losses[0]
    assert float((mod.out_gate - init_gate).abs().max().item()) > 0.0
    assert float((next(mod.blocks.parameters()) - init_blk).abs().max().item()) > 0.0
    assert float((mod.step_embed.weight - init_se).abs().max().item()) > 0.0
    assert float((mod.q_agg - init_q).abs().max().item()) > 0.0


def test_grad_checkpointing_parity() -> None:
    model = _make_tiny_qwen2(seed=31)
    torch.manual_seed(9)
    mod_ref = AdditiveLoopedMoI.from_model(model, insert_layer=2, n_loops=2, d_loop_key=16)
    with torch.no_grad():
        mod_ref.out_gate.fill_(0.25)
        mod_ref.step_embed.weight.normal_(std=0.05)
        mod_ref.q_agg.normal_(std=0.1)

    mod_ckpt = AdditiveLoopedMoI(AdditiveLoopConfig.from_dict(mod_ref.config.to_dict()), [
        type(b)(model.config, layer_idx=b.self_attn.layer_idx) for b in mod_ref.blocks
    ])
    mod_ckpt.load_state_dict(mod_ref.state_dict())

    x = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    t = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)

    with mod_ref.attach(model, n_loops=2, grad_checkpointing=False):
        l_ref = model(x, use_cache=False).logits
        loss_ref = F.cross_entropy(l_ref.view(-1, l_ref.shape[-1]), t.view(-1))
        loss_ref.backward()

    with mod_ckpt.attach(model, n_loops=2, grad_checkpointing=True):
        l_ckpt = model(x, use_cache=False).logits
        loss_ckpt = F.cross_entropy(l_ckpt.view(-1, l_ckpt.shape[-1]), t.view(-1))
        loss_ckpt.backward()

    assert float((l_ref - l_ckpt).abs().max().item()) < 1e-6
    for (nm, p_ref), (_, p_ckpt) in zip(mod_ref.named_parameters(), mod_ckpt.named_parameters()):
        assert p_ref.grad is not None and p_ckpt.grad is not None, nm
        assert float((p_ref.grad - p_ckpt.grad).abs().max().item()) < 1e-5, nm


def test_kv_cache_and_left_padded_batch_decode_equivalence() -> None:
    model = _make_tiny_qwen2(seed=41)
    mod = AdditiveLoopedMoI.from_model(model, insert_layer=2, n_loops=2, max_steps=6, d_loop_key=16)
    with torch.no_grad():
        mod.out_gate.fill_(0.4)
        mod.step_embed.weight.normal_(std=0.08)
        mod.q_agg.normal_(std=0.2)

    # Left-padded batch of 2 sequences with 3 prompt tokens + 3 decoded tokens
    full_ids = torch.tensor([[0, 0, 5, 11, 17, 23, 29, 31], [7, 13, 19, 25, 31, 37, 41, 43]], dtype=torch.long)
    full_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    full_pos = (full_mask.cumsum(dim=-1) - 1).clamp_min(0)

    for n_loops in (1, 2, 3, 4):
        with torch.no_grad(), mod.attach(model, n_loops=n_loops):
            ref_logits = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                position_ids=full_pos,
                use_cache=False,
            ).logits

            # Prefill first 5 positions, then decode 3 tokens step-by-step
            out = model(
                input_ids=full_ids[:, :5],
                attention_mask=full_mask[:, :5],
                position_ids=full_pos[:, :5],
                use_cache=True,
            )
            pkv = out.past_key_values
            assert pkv is not None
            step_logits = [out.logits[:, -1:]]
            for pos in range(5, 8):
                out_s = model(
                    input_ids=full_ids[:, pos : pos + 1],
                    attention_mask=full_mask[:, : pos + 1],
                    position_ids=full_pos[:, pos : pos + 1],
                    past_key_values=pkv,
                    use_cache=True,
                )
                pkv = out_s.past_key_values
                step_logits.append(out_s.logits)

            # Base cache must stay identical in length across all layers (unpolluted)
            assert pkv.get_seq_length(0) == 8
            assert pkv.get_seq_length(2) == 8

        cat_decoded = torch.cat(step_logits, dim=1)  # positions 4..7
        diff = float((cat_decoded - ref_logits[:, 4:8]).abs().max().item())
        assert diff < 1e-4, f"n_loops={n_loops} max_abs_diff={diff}"


def test_telemetry_sums_to_one() -> None:
    model = _make_tiny_qwen2(seed=51)
    mod = AdditiveLoopedMoI.from_model(model, insert_layer=2, n_loops=2, max_steps=6, d_loop_key=16)
    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    for n_loops in (2, 4):
        mod.reset_telemetry()
        with torch.no_grad(), mod.attach(model, n_loops=n_loops, record_stats=True):
            _ = model(x, use_cache=False)
        tel = mod.telemetry_summary()
        assert tel["n_tokens"] == 5
        assert len(tel["alpha_mean"]) == n_loops + 1
        assert abs(sum(tel["alpha_mean"]) - 1.0) < 1e-5
        assert tel["entropy"] >= 0.0


def test_save_load_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    model = _make_tiny_qwen2(seed=61)
    mod = AdditiveLoopedMoI.from_model(model, insert_layer=2, n_loops=2, max_steps=6, d_loop_key=16)
    with torch.no_grad():
        mod.out_gate.fill_(0.3)
        mod.q_agg.normal_(std=0.2)

    out_file = tmp_path / "add_loop.safetensors"
    mod.save(out_file)
    loaded = AdditiveLoopedMoI.load(out_file, template_model=model, verify=True)
    assert loaded.config.to_dict() == mod.config.to_dict()

    x = torch.tensor([[3, 6, 9, 12]], dtype=torch.long)
    with torch.no_grad():
        with mod.attach(model, n_loops=3):
            l1 = model(x, use_cache=False).logits
        with loaded.attach(model, n_loops=3):
            l2 = model(x, use_cache=False).logits
    assert float((l1 - l2).abs().max().item()) == 0.0

    # Tamper with tensor payload while preserving metadata sha256
    with safe_open(str(out_file), "pt") as fh:
        meta = dict(fh.metadata() or {})
        sd = {k: fh.get_tensor(k).clone() for k in fh.keys()}
    sd["out_gate"] = sd["out_gate"] + 1.0
    bad_file = tmp_path / "tampered.safetensors"
    save_file(sd, str(bad_file), metadata=meta)
    assert meta["wsi_additive_loop"]
    assert FORMAT_ADDITIVE_LOOP == "wsi_additive_loop/1"
    with pytest.raises(ValueError, match="SHA-256"):
        AdditiveLoopedMoI.load(bad_file, template_model=model, verify=True)


def test_exception_inside_attach_restores_block() -> None:
    model = _make_tiny_qwen2(seed=71)
    _, blocks = find_blocks(model)
    orig_blk = blocks[2]
    mod = AdditiveLoopedMoI.from_model(model, insert_layer=2, n_loops=2)

    with pytest.raises(RuntimeError, match="boom"):
        with mod.attach(model, n_loops=2):
            assert blocks[2] is not orig_blk
            raise RuntimeError("boom")

    assert blocks[2] is orig_blk


def test_eval_and_train_inside_attach_no_recursion() -> None:
    model = _make_tiny_qwen2(seed=73)
    mod = AdditiveLoopedMoI.from_model(model, insert_layer=2, n_loops=2)

    with mod.attach(model, n_loops=2):
        model.eval()
        model.train()
    model.eval()
