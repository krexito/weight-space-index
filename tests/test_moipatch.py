"""Unit and integration tests for portable `.moipatch` recipes (`wsi/moipatch.py` and packagers)."""

from __future__ import annotations

import inspect
import json
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi import Edit, MoIPatch, PatchHandle, RoutedPatchHandle, audit_moipatch  # noqa: E402
from wsi.cli import main as wsi_cli_main  # noqa: E402
from wsi.moipatch import apply_edit  # noqa: E402


class _TinyBlock(torch.nn.Module):
    def __init__(self, hidden: int = 16, inter: int = 32, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(inter, hidden, bias=False, dtype=dtype)
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)


class _TinyLM(torch.nn.Module):
    def __init__(self, n_layers: int = 3, hidden: int = 16, inter: int = 32, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyBlock(hidden, inter, dtype=dtype) for _ in range(n_layers)])


def _save_tiny_checkpoint(model: _TinyLM, ckpt_dir: Path, sharded: bool = False) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    if sharded:
        save_file(sd, str(ckpt_dir / "model.safetensors-00001-of-00001.safetensors"))
        idx = {"metadata": {}, "weight_map": {k: "model.safetensors-00001-of-00001.safetensors" for k in sd}}
        (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps(idx), encoding="utf-8")
    else:
        save_file(sd, str(ckpt_dir / "model.safetensors"))


def test_moipatch_save_load_attach_and_restore() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = _TinyLM(n_layers=3, hidden=16, inter=32, dtype=torch.bfloat16)
        ckpt_dir = tmp / "ckpt"
        _save_tiny_checkpoint(model, ckpt_dir, sharded=True)

        u_mlp = torch.randn(16)
        u_mlp = u_mlp / u_mlp.norm()
        u_attn = torch.randn(16)
        u_attn = u_attn / u_attn.norm()
        cols = torch.tensor([1, 5, 9, 13], dtype=torch.long)
        r_dir = torch.zeros(16)
        r_dir[0] = 1.0

        patch = MoIPatch(
            edits=[
                Edit(module="layers.1.mlp.down_proj", u=u_mlp, alpha=6.0, idx=cols),
                Edit(module="layers.2.self_attn.o_proj", u=u_attn, alpha=4.0, idx=None),
            ],
            router={"layer": 1, "readout": "last_token", "dir": r_dir, "thr": 0.5},
            gates=[{"type": "answer_confidence", "thr": 0.85, "note": "test gate"}],
            meta={"name": "unit-test", "base_model": "tiny", "policy": "on pushback only", "expected": {"acc": 0.9}},
        )
        patch.fingerprint_from_checkpoint(ckpt_dir)
        out_path = patch.save(tmp / "test.moipatch")
        loaded = MoIPatch.load(out_path)
        assert "unit-test [moipatch/1]" in loaded.describe()
        assert len(loaded.edits) == 2 and loaded.edits[0].idx is not None and loaded.edits[1].idx is None

        # Router + confidence gate firing
        hs = [torch.zeros(2, 4, 16) for _ in range(3)]
        hs[1][0, -1, 0] = 1.2
        hs[1][1, -1, 0] = 0.2
        assert loaded.fires(hs).tolist() == [True, False]
        assert loaded.fires(hs, confidence=torch.tensor([0.92, 0.99])).tolist() == [True, False]
        assert loaded.fires(hs, confidence=torch.tensor([0.50, 0.99])).tolist() == [False, False]

        # Attach in float32 (simulating verify --cpu) and verify exact bf16 fingerprint + restore
        model_f32 = _TinyLM(n_layers=3, hidden=16, inter=32, dtype=torch.float32)
        with torch.no_grad():
            for p_dst, p_src in zip(model_f32.parameters(), model.parameters()):
                p_dst.copy_(p_src.float())

        handle = loaded.attach(model_f32)
        assert isinstance(handle, PatchHandle) and not handle.active and handle.fingerprint_ok()
        orig_w1 = model_f32.model.layers[1].mlp.down_proj.weight.detach().clone()
        expected_w1 = apply_edit(orig_w1, u_mlp, cols, 6.0)
        handle.on()
        assert handle.active
        assert torch.equal(model_f32.model.layers[1].mlp.down_proj.weight, expected_w1)
        # Untouched columns stayed identical
        assert torch.equal(model_f32.model.layers[1].mlp.down_proj.weight[:, 0], orig_w1[:, 0])
        handle.off()
        assert not handle.active and handle.check_restored() and handle.fingerprint_ok()

        # Tampered model must be rejected by fingerprint check
        with torch.no_grad():
            model_f32.model.layers[1].mlp.down_proj.weight[0, 0] += 1.0
        try:
            loaded.attach(model_f32)
            raise AssertionError("expected ValueError on fingerprint mismatch")
        except ValueError as err:
            assert "fingerprint mismatch" in str(err)


def test_make_moipatch_sycophancy_and_cli() -> None:
    from scripts.make_moipatch_sycophancy import main as syco_main

    torch.manual_seed(1)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = _TinyLM(n_layers=4, hidden=16, inter=32, dtype=torch.bfloat16)
        ckpt_dir = tmp / "ckpt"
        _save_tiny_checkpoint(model, ckpt_dir, sharded=False)

        work = tmp / "work"
        work.mkdir()
        dg = {
            "ua": [np.random.randn(16).astype(np.float32) for _ in range(4)],
            "u": [np.random.randn(16).astype(np.float32) for _ in range(4)],
            "cands": {"late": [2, 3]},
            "neurons": {2: [0, 3, 7], 3: [1, 4, 9]},
        }
        rt = {"layer": 2, "dir": np.ones(16, dtype=np.float32), "thr": 0.75, "bal_acc_heldout": 0.96}
        res = {
            "model": "tiny/qwen",
            "n": {"test": 100},
            "router": {"layer": 2},
            "gate_study": {
                "chosen_on_dev": "attn:late@8|maxprob",
                "base_test": {"held": 0.15, "accepted": 0.97},
                "rows": {
                    "attn:late@8|maxprob": {
                        "thr": 0.885,
                        "held": 0.401,
                        "accepted": 0.907,
                        "held_unseen_wording": 0.163,
                        "neutral_same": 0.999,
                    }
                },
            },
        }
        (work / "diagnosis_pressure.pkl").write_bytes(pickle.dumps(dg))
        (work / "router.pkl").write_bytes(pickle.dumps(rt))
        (tmp / "results.json").write_text(json.dumps(res), encoding="utf-8")

        out_patch = tmp / "syco.moipatch"
        syco_main(["--work", str(work), "--results", str(tmp / "results.json"),
                   "--model-dir", str(ckpt_dir), "--out", str(out_patch)])
        loaded = MoIPatch.load(out_patch)
        assert loaded.meta["name"] == "anti-sycophancy"
        assert [e.module for e in loaded.edits] == ["layers.2.self_attn.o_proj", "layers.3.self_attn.o_proj"]
        assert loaded.gates[0]["type"] == "answer_confidence" and loaded.gates[0]["thr"] == 0.885

        # CLI `wsi moipatch info` works cleanly
        assert wsi_cli_main(["moipatch", "info", str(out_patch)]) == 0


def test_real_p50c_loop_moipatch_artifact() -> None:
    p = ROOT / "patches" / "qwen35_0.8b_p50c_loop_fix.moipatch"
    if not p.exists():
        return
    patch = MoIPatch.load(p)
    assert patch.meta["name"] == "qwen35-0.8b-loop-fix-p50c"
    assert [e.module for e in patch.edits] == ["layers.15.mlp.down_proj", "layers.19.mlp.down_proj"]
    assert all(e.idx is not None and len(e.idx) == 64 and e.alpha == 6.0 for e in patch.edits)
    assert patch.gates[0]["type"] == "ngram_repetition_frac"
    cos = patch.meta["expected"]["reconstruction"]["delta_cosine"]
    assert all(0.9999 <= c <= 1.00001 for c in cos)


class _TinyRoutableBlock(torch.nn.Module):
    def __init__(self, hidden: int = 16, inter: int = 32, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.up_proj = torch.nn.Linear(hidden, inter, bias=False, dtype=dtype)
        self.mlp.down_proj = torch.nn.Linear(inter, hidden, bias=False, dtype=dtype)
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.self_attn.o_proj(x)
        return h + self.mlp.down_proj(torch.tanh(self.mlp.up_proj(h)))


class _TinyRoutableLM(torch.nn.Module):
    def __init__(self, n_layers: int = 4, hidden: int = 16, inter: int = 32, vocab: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(eos_token_id=0, pad_token_id=0)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.model.layers = torch.nn.ModuleList([_TinyRoutableBlock(hidden, inter) for _ in range(n_layers)])
        self.model.norm = torch.nn.LayerNorm(hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.forward_calls = 0

        with torch.no_grad():
            for blk in self.model.layers:
                blk.self_attn.o_proj.weight.mul_(0.1)
                blk.mlp.up_proj.weight.mul_(0.25)
                blk.mlp.down_proj.weight.mul_(0.25)
            # Clear separation on coord 0: token 7 -> positive residual on layer 1, token 1 -> negative residual
            self.model.embed_tokens.weight[1].zero_()
            self.model.embed_tokens.weight[1, 0] = -4.0
            self.model.embed_tokens.weight[7].zero_()
            self.model.embed_tokens.weight[7, 0] = 4.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: dict | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ):
        self.forward_calls += 1
        h = self.model.embed_tokens(input_ids)
        if past_key_values is not None and use_cache:
            past_key_values["len"] = past_key_values.get("len", 0) + int(input_ids.shape[1])

        hs = [h] if output_hidden_states else None
        for idx, blk in enumerate(self.model.layers):
            h = blk(h)
            if output_hidden_states and idx < len(self.model.layers) - 1:
                hs.append(h)
        h = self.model.norm(h)
        if output_hidden_states:
            hs.append(h)
        logits = self.lm_head(h)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hs) if hs is not None else None,
            past_key_values=past_key_values,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 4,
        use_cache: bool = True,
    ) -> torch.Tensor:
        cur = input_ids.clone()
        cache: dict | None = {} if use_cache else None
        for step in range(max_new_tokens):
            step_ids = cur if (step == 0 or not use_cache) else cur[:, -1:]
            step_mask = attention_mask if step == 0 else None
            out = self.forward(step_ids, attention_mask=step_mask, past_key_values=cache, use_cache=use_cache)
            nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
        return cur


def _unit_vec(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def test_enable_routed_fast_single_pass_and_restore() -> None:
    torch.manual_seed(0)
    model = _TinyRoutableLM(n_layers=4, hidden=16, inter=32, vocab=32).eval()
    r_dir = torch.zeros(16)
    r_dir[0] = 1.0
    patch = MoIPatch(
        edits=[
            Edit(module="layers.2.mlp.down_proj", u=_unit_vec(16, 11), alpha=6.0, idx=torch.tensor([0, 3, 7, 15])),
            Edit(module="layers.3.self_attn.o_proj", u=_unit_vec(16, 12), alpha=4.0, idx=None),
        ],
        router={"layer": 1, "readout": "last_token", "dir": r_dir, "thr": 0.0},
        gates=[{"type": "answer_confidence", "thr": 0.85}],
        meta={"name": "fast-routed"},
    )

    healthy_ids = torch.tensor([[2, 3, 1]], dtype=torch.long)
    trigger_ids = torch.tensor([[2, 3, 7]], dtype=torch.long)
    mixed_ids = torch.tensor([[2, 3, 1], [2, 3, 7]], dtype=torch.long)
    right_padded = torch.tensor([[2, 7, 1, 1]], dtype=torch.long)
    right_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        base_healthy = model(healthy_ids).logits
        base_trigger = model(trigger_ids).logits
        base_mixed = model(mixed_ids).logits
        base_rp_expected = model(right_padded).logits
        base_gen_mixed = model.generate(mixed_ids, max_new_tokens=3, use_cache=True)

        with patch.attach(model) as ph:
            ph.on()
            patched_trigger = model(trigger_ids).logits
            patched_mixed = model(mixed_ids).logits
            patched_gen_mixed = model.generate(mixed_ids, max_new_tokens=3, use_cache=True)

        model.forward_calls = 0
        with patch.enable_routed(model) as h:
            assert isinstance(h, RoutedPatchHandle) and h.mode == "residual_router"
            assert not h._needs_pre_probe
            assert h.fingerprint_ok()

            # Healthy sequence: bit-identical to base in ONE forward pass
            out_h = model(healthy_ids).logits
            assert model.forward_calls == 1
            assert torch.equal(out_h, base_healthy)
            assert h.stats["calls"] == 1 and h.stats["fired"] == 0

            # Trigger sequence: matches PatchHandle.on() in ONE forward pass
            out_t = model(trigger_ids).logits
            assert model.forward_calls == 2
            assert torch.allclose(out_t, patched_trigger, atol=1e-4, rtol=1e-4)
            assert h.stats["calls"] == 2 and h.stats["fired"] == 1

            # Mixed batch: row 0 stays bit-exact base while row 1 matches patched logits
            out_m = model(mixed_ids).logits
            assert model.forward_calls == 3
            assert torch.equal(out_m[0], base_mixed[0])
            assert torch.allclose(out_m[1], patched_mixed[1], atol=1e-4, rtol=1e-4)

            # Mixed batch generate(): row 0 matches base tokens, row 1 matches patched tokens
            gen_m = model.generate(mixed_ids, max_new_tokens=3, use_cache=True)
            assert torch.equal(gen_m[0], base_gen_mixed[0])
            assert torch.equal(gen_m[1], patched_gen_mixed[1])

            # Right-padded attention_mask picks the real last token (pos 1 = trigger token 7), ignoring trailing pad
            out_rp = model(right_padded, attention_mask=right_mask).logits
            assert h.stats["last_fired"] == [True]
            out_unpadded = model(torch.tensor([[2, 7]], dtype=torch.long)).logits
            assert torch.allclose(out_rp[:, :2], out_unpadded, atol=1e-5, rtol=1e-5)
            # Same padded tensor WITHOUT attention_mask reads pos 3 (token 1 = healthy) -> must NOT fire
            out_no_mask = model(right_padded).logits
            assert h.stats["last_fired"] == [False]
            assert torch.equal(out_no_mask, base_rp_expected)

            # Confidence gate suppresses firing below thr and permits above thr
            h.set_confidence(0.50)
            assert torch.equal(model(trigger_ids).logits, base_trigger)
            h.set_confidence(0.95)
            assert torch.allclose(model(trigger_ids).logits, patched_trigger, atol=1e-4, rtol=1e-4)

        # Exception safety: error inside `with` still restores model & hooks completely
        try:
            with patch.enable_routed(model):
                _ = model(trigger_ids)
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert "forward" not in model.__dict__
        assert "generate" not in model.__dict__
        assert "_wsi_route_stack" not in model.__dict__
        assert torch.equal(model(trigger_ids).logits, base_trigger)


def test_enable_routed_late_router_and_kv_cache_safety() -> None:
    torch.manual_seed(1)
    model = _TinyRoutableLM(n_layers=4, hidden=16, inter=32, vocab=32).eval()
    r_dir = torch.zeros(16)
    r_dir[0] = 1.0

    healthy_ids = torch.tensor([[2, 3, 1]], dtype=torch.long)
    trigger_ids = torch.tensor([[2, 3, 7]], dtype=torch.long)

    for router_layer in (3, 4):  # 3 = mid-late block (> min_edit=1); 4 = n_blocks (post-final-norm)
        patch = MoIPatch(
            edits=[Edit(module="layers.1.mlp.down_proj", u=_unit_vec(16, 20 + router_layer), alpha=6.0)],
            router={"layer": router_layer, "readout": "last_token", "dir": r_dir, "thr": 0.0},
            meta={"name": f"late-L{router_layer}"},
        )
        with torch.no_grad():
            base_h = model(healthy_ids).logits
            base_gen_h = model.generate(healthy_ids, max_new_tokens=4, use_cache=True)
            with patch.attach(model) as ph:
                ph.on()
                patched_t = model(trigger_ids).logits
                patched_gen_t = model.generate(trigger_ids, max_new_tokens=4, use_cache=True)

            model.forward_calls = 0
            with patch.enable_routed(model) as h:
                assert h._needs_pre_probe
                out_h = model(healthy_ids).logits
                assert torch.equal(out_h, base_h)
                if router_layer == 3:
                    # Standalone healthy pass runs ONCE (0% extra cost on healthy input!)
                    assert model.forward_calls == 1

                out_t = model(trigger_ids).logits
                assert torch.allclose(out_t, patched_t, atol=1e-4, rtol=1e-4)

                # Mutable KV-cache check during generate: must not double-count cache length
                cache: dict = {}
                _ = model.forward(trigger_ids, past_key_values=cache, use_cache=True)
                assert cache["len"] == trigger_ids.shape[1]

                gen_h = model.generate(healthy_ids, max_new_tokens=4, use_cache=True)
                gen_t = model.generate(trigger_ids, max_new_tokens=4, use_cache=True)
                assert torch.equal(gen_h, base_gen_h)
                assert torch.equal(gen_t, patched_gen_t)


class _LoopingTinyLM(torch.nn.Module):
    """Deterministic causal toy LM: cycles [2, 3, 4, ...] on prompt seed 9 and outputs distinct tokens [12, 13, ...]
    on seed 8. When `layers.0.mlp.down_proj` has `apply_edit` applied along u_break, output switches to token 1."""

    def __init__(self, hidden: int = 8, inter: int = 8, vocab: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(eos_token_id=11, pad_token_id=10)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, hidden)
        blk = torch.nn.Module()
        blk.mlp = torch.nn.Module()
        blk.mlp.down_proj = torch.nn.Linear(inter, hidden, bias=False)
        self.model.layers = torch.nn.ModuleList([blk])
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

        with torch.no_grad():
            self.model.embed_tokens.weight.zero_()
            # Make u = e_0 write a strong positive component to logit[1] ONLY AFTER apply_edit (W' = W - alpha u u^T W)
            blk.mlp.down_proj.weight.zero_()
            blk.mlp.down_proj.weight[0, 0] = -1.0  # u^T W = [-1, 0, ...], so W' adds +alpha on col 0!
            self.lm_head.weight.zero_()
            self.lm_head.weight[1, 0] = 2.0  # logit[1]: -14 unpatched -> +18 once W' activates (+alpha=+8)

    def forward(self, input_ids: torch.Tensor, **_kw):
        B, T = input_ids.shape
        device = input_ids.device
        # Next token logic computed on step t based on seed (first prompt token) and last token
        seeds = input_ids[:, 0].tolist()
        last_toks = input_ids[:, -1].tolist()
        mlp_in = torch.zeros(B, T, 8, device=device)
        base_logits = torch.full((B, T, 32), -10.0, device=device)
        for b, (sd, lt) in enumerate(zip(seeds, last_toks)):
            if sd == 9:  # looping prompt: drives MLP col 0 so patch break dominates once latched
                mlp_in[b, -1, 0] = 2.0
                nxt = {9: 2, 2: 3, 3: 4, 4: 2, 1: 1}.get(lt, 2)
                base_logits[b, -1, nxt] = 5.0
            elif sd == 8:  # strictly non-repeating healthy prompt: emits distinct tokens 12, 13, 14, ...
                mlp_in[b, -1, 0] = 2.0
                nxt = 12 + T - 1
                base_logits[b, -1, nxt] = 5.0
            else:  # immediate EOS then repeated PAD/EOS
                base_logits[b, -1, 11] = 5.0

        delta_h = self.model.layers[0].mlp.down_proj(mlp_in)
        logits = base_logits + self.lm_head(delta_h)
        return SimpleNamespace(logits=logits)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 10, **_kw) -> torch.Tensor:
        cur = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(cur)
            nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
        return cur


def test_enable_routed_ngram_loop_gate() -> None:
    model = _LoopingTinyLM().eval()
    u_break = torch.zeros(8)
    u_break[0] = 1.0
    cols = torch.tensor([0, 1], dtype=torch.long)

    patch_with_gate = MoIPatch(
        edits=[Edit(module="layers.0.mlp.down_proj", u=u_break, alpha=8.0, idx=cols)],
        router=None,
        gates=[{"type": "ngram_repetition_frac", "thr": 0.5, "ngram": 3, "window": 4}],
        meta={"name": "loop-gate"},
    )
    patch_no_gate = MoIPatch(
        edits=[Edit(module="layers.0.mlp.down_proj", u=u_break, alpha=8.0, idx=cols)],
        router=None,
        gates=[],
        meta={"name": "loop-bare"},
    )

    prompts = torch.tensor([[9], [8]], dtype=torch.long)  # row 0 loops [2,3,4,...], row 1 emits distinct tokens 12..21
    with torch.no_grad():
        base_seq = model.generate(prompts, max_new_tokens=10)
        assert base_seq[0, 1:].tolist() == [2, 3, 4, 2, 3, 4, 2, 3, 4, 2]
        assert base_seq[1, 1:].tolist() == list(range(12, 22))

        # Test embedded ngram gate AND explicit `loop_router={...}` override on gate-less patch
        for patch_obj, lr_arg in [
            (patch_with_gate, None),
            (patch_no_gate, {"ngram": 3, "window": 4, "thr": 0.5}),
        ]:
            with patch_obj.enable_routed(model, loop_router=lr_arg) as h:
                assert h.mode == "ngram_loop"
                seq = model.generate(prompts, max_new_tokens=10)
                st = h.stats
                # Row 0 stays identical to base `[2,3,4,2,3,4,2]` until 2 of last 4 tokens end a repeated 3-gram (at resp len 7), then breaks to 1!
                assert seq[0, 1:8].tolist() == [2, 3, 4, 2, 3, 4, 2]
                assert seq[0, 8:].tolist() == [1, 1, 1]
                # Row 1 never has a repeated 3-gram -> bit-exact base throughout!
                assert seq[1].tolist() == base_seq[1].tolist()
                assert st["calls"] == 2 and st["fired"] == 1
                assert st["last_fired"] == [True, False]
                assert st["last_fired_at"] == [7, None]

            # Repeated EOS tokens must NEVER trigger the loop detector
            eos_prompt = torch.tensor([[10]], dtype=torch.long)
            with patch_with_gate.enable_routed(model) as h_eos:
                eos_seq = model.generate(eos_prompt, max_new_tokens=8)
                assert eos_seq[0, 1:].tolist() == [11] * 8
                assert h_eos.stats["fired"] == 0 and h_eos.stats["last_fired_at"] == [None]


def test_enable_routed_compose_two_patches_and_out_of_order_close() -> None:
    torch.manual_seed(2)
    model = _TinyRoutableLM(n_layers=4, hidden=16, inter=32, vocab=32).eval()
    r_dir = torch.zeros(16)
    r_dir[0] = 1.0
    p_res = MoIPatch(
        edits=[Edit(module="layers.2.self_attn.o_proj", u=_unit_vec(16, 31), alpha=4.0)],
        router={"layer": 1, "readout": "last_token", "dir": r_dir, "thr": 0.0},
    )
    p_always = MoIPatch(
        edits=[Edit(module="layers.3.mlp.down_proj", u=_unit_vec(16, 32), alpha=5.0, idx=torch.arange(6))],
        router=None,
    )

    trigger_ids = torch.tensor([[2, 3, 7]], dtype=torch.long)
    with torch.no_grad():
        base_logits = model(trigger_ids).logits
        with p_always.attach(model) as a2_only:
            a2_only.on()
            expected_h2_only = model(trigger_ids).logits
        with p_res.attach(model) as a1, p_always.attach(model) as a2:
            a1.on()
            a2.on()
            both_logits = model(trigger_ids).logits

        h1 = p_res.enable_routed(model)
        h2 = p_always.enable_routed(model)
        assert h1.mode == "residual_router" and h2.mode == "always_on"
        out_both = model(trigger_ids).logits
        assert torch.allclose(out_both, both_logits, atol=1e-4, rtol=1e-4)

        # Out-of-order close (close first handle before second handle; only h2 remains active)
        h1.close()
        out_h2_only = model(trigger_ids).logits
        assert torch.allclose(out_h2_only, expected_h2_only, atol=1e-4, rtol=1e-4)
        h2.close()

        assert "forward" not in model.__dict__
        assert "generate" not in model.__dict__
        assert "_wsi_route_stack" not in model.__dict__
        assert torch.equal(model(trigger_ids).logits, base_logits)


def test_enable_routed_real_hf_llama_generate() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(4)
    cfg = LlamaConfig(
        num_hidden_layers=4,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(cfg).eval()
    sig_before = str(inspect.signature(model.forward))

    ids = torch.tensor([[1, 5, 9, 13]], dtype=torch.long)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        base_out = model(ids, attention_mask=attn).logits
        base_gen = model.generate(ids, attention_mask=attn, max_new_tokens=6, do_sample=False, use_cache=True)

        # Check early residual (non-firing vs firing), late residual (layer=3), final norm (layer=4), and ngram_loop
        p_early_off = MoIPatch(
            edits=[Edit(module="layers.2.mlp.down_proj", u=_unit_vec(32, 41), alpha=4.0, idx=torch.arange(8))],
            router={"layer": 1, "readout": "last_token", "dir": _unit_vec(32, 42), "thr": 1e6},
        )
        p_early_on = MoIPatch(
            edits=[Edit(module="layers.2.mlp.down_proj", u=_unit_vec(32, 41), alpha=4.0, idx=torch.arange(8))],
            router={"layer": 1, "readout": "last_token", "dir": _unit_vec(32, 42), "thr": -1e6},
        )
        p_late_on = MoIPatch(
            edits=[Edit(module="layers.1.self_attn.o_proj", u=_unit_vec(32, 43), alpha=4.0)],
            router={"layer": 3, "readout": "last_token", "dir": _unit_vec(32, 44), "thr": -1e6},
        )
        p_final_norm = MoIPatch(
            edits=[Edit(module="layers.1.self_attn.o_proj", u=_unit_vec(32, 43), alpha=4.0)],
            router={"layer": 4, "readout": "last_token", "dir": _unit_vec(32, 44), "thr": -1e6},
        )
        p_loop = MoIPatch(
            edits=[Edit(module="layers.2.mlp.down_proj", u=_unit_vec(32, 41), alpha=4.0)],
            router=None,
            gates=[{"type": "ngram_repetition_frac", "thr": 0.5, "ngram": 2, "window": 3}],
        )

        with p_early_off.enable_routed(model) as h:
            assert str(inspect.signature(model.forward)) == sig_before
            assert torch.equal(model(ids, attention_mask=attn).logits, base_out)
            gen = model.generate(ids, attention_mask=attn, max_new_tokens=6, do_sample=False, use_cache=True)
            assert torch.equal(gen, base_gen)
            assert h.stats["fired"] == 0

        with p_early_on.enable_routed(model) as h:
            gen_on = model.generate(ids, attention_mask=attn, max_new_tokens=6, do_sample=False, use_cache=True)
            assert gen_on.shape == (1, ids.shape[1] + 6)
            assert h.stats["fired"] == 1

        for late_p in (p_late_on, p_final_norm):
            with late_p.attach(model) as ph:
                ph.on()
                expected_logits = model(ids, attention_mask=attn).logits
            with late_p.enable_routed(model) as h:
                assert h._needs_pre_probe
                out_l = model(ids, attention_mask=attn).logits
                assert torch.allclose(out_l, expected_logits, atol=1e-4, rtol=1e-4)
                gen_l = model.generate(ids, attention_mask=attn, max_new_tokens=6, do_sample=False, use_cache=True)
                assert gen_l.shape == (1, ids.shape[1] + 6)

        with p_loop.enable_routed(model) as h:
            gen_lp = model.generate(ids, attention_mask=attn, max_new_tokens=6, do_sample=False, use_cache=True)
            assert gen_lp.shape == (1, ids.shape[1] + 6)
            assert h.mode == "ngram_loop"

        assert "forward" not in model.__dict__
        assert "generate" not in model.__dict__
        assert "_wsi_route_stack" not in model.__dict__
        assert str(inspect.signature(model.forward)) == sig_before


def test_enable_routed_always_on_and_fingerprint_rejection() -> None:
    torch.manual_seed(5)
    model = _TinyRoutableLM(n_layers=3, hidden=16, inter=32, vocab=32).eval()
    patch = MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=_unit_vec(16, 51), alpha=3.0)])
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        with patch.attach(model) as ph:
            ph.on()
            expected = model(ids).logits
        with patch.enable_routed(model) as h:
            assert h.mode == "always_on" and h.active
            assert torch.allclose(model(ids).logits, expected, atol=1e-4, rtol=1e-4)

    # Mismatched sha256 must fail on enable_routed and leave zero hooks/wrappers behind
    bad_patch = MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=_unit_vec(16, 51), alpha=3.0, base_sha256="0" * 64)])
    try:
        bad_patch.enable_routed(model)
        raise AssertionError("expected ValueError on fingerprint mismatch")
    except ValueError as err:
        assert "fingerprint mismatch" in str(err)
    assert "_wsi_route_stack" not in model.__dict__


def test_audit_moipatch_structural_and_end_to_end() -> None:
    torch.manual_seed(7)
    # 1. Model without lm_head: structural/fingerprint checks pass and vocab_logit_lens stays empty
    no_head = _TinyLM(n_layers=3, hidden=16, inter=32, dtype=torch.float32).eval()
    p_no_head = MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=_unit_vec(16, 61), alpha=2.0, base_dtype="float32")])
    rep_min = p_no_head.audit(no_head)
    assert rep_min["vocab_logit_lens"] == []
    assert rep_min["output_logit_diff"] is None
    assert rep_min["control_audit"] is None
    assert rep_min["fingerprint_ok"] is True and rep_min["restore_exact_ok"] is True

    # 2. Early-layer edit + end-to-end output_logit_diff + routed control_audit
    model = _TinyRoutableLM(n_layers=4, hidden=16, inter=32, vocab=32).eval()
    with torch.no_grad():
        # Strong positive signal on dim 1 during trigger prompts (token 7 -> embed dim 1 = +4.0, dim 2 = 0)
        model.model.embed_tokens.weight[7, 1] = 4.0
        model.model.embed_tokens.weight[1, 1] = -2.0
        model.model.layers[1].self_attn.o_proj.weight.zero_()
        model.model.layers[1].self_attn.o_proj.weight[1, 1] = 2.0
        model.lm_head.weight.zero_()
        model.lm_head.weight[29, 1] = 5.0
        model.lm_head.weight[14, 2] = 1.5
        model.model.embed_tokens.weight[:, 2] = 3.0

    u = torch.zeros(16)
    u[1] = 1.0
    r_dir = torch.zeros(16)
    r_dir[0] = 1.0
    patch = MoIPatch(
        edits=[Edit(module="layers.1.self_attn.o_proj", u=u, alpha=1.0, base_dtype="float32")],
        router={"layer": 1, "readout": "last_token", "dir": r_dir, "thr": 0.0},
    )
    with tempfile.TemporaryDirectory() as td:
        p_path = Path(td) / "audit_test.moipatch"
        patch.save(p_path)
        loaded = MoIPatch.load(p_path)

        class _DummyTok:
            eos_token_id = 0

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [2, 3, 7] if "trigger" in text else [2, 3, 1]

            def decode(self, ids: list[int]) -> str:
                return f"tok_{ids[0]}"

        orig_w1 = model.model.layers[1].self_attn.o_proj.weight.detach().clone()
        rep = audit_moipatch(
            loaded,
            model,
            _DummyTok(),
            probe_prompts=["trigger a", "trigger b"],
            control_prompts=["benign a", "benign b"],
            patch_path=p_path,
            top_k=5,
        )
        assert rep["patch_size_kb"] is not None and rep["patch_size_kb"] > 0
        assert rep["fingerprint_ok"] is True and rep["restore_exact_ok"] is True
        assert torch.equal(model.model.layers[1].self_attn.o_proj.weight, orig_w1)
        assert rep["vocab_logit_lens"][0]["early_layer_caveat"] is True

        old = rep["output_logit_diff"]
        assert old is not None and old["n_probes"] == 2 and old["mean_next_token_kl"] > 0
        assert old["probe_router_fire_rate"] == 1.0
        assert old["suppressed_tokens"][0]["token_id"] == 29 and old["suppressed_tokens"][0]["mean_delta_prob"] < 0
        pro_ids = {x["token_id"]: x for x in old["promoted_tokens"]}
        assert 14 in pro_ids and pro_ids[14]["mean_delta_prob"] > 0

        ca = rep["control_audit"]
        assert ca is not None and ca["n_controls"] == 2
        assert ca["routed_fire_rate"] == 0.0 and ca["routed_mean_kl"] == 0.0
        assert ca["max_abs_logit_diff_routed"] == 0.0


def test_router_veto_save_load_fires_and_enable_routed() -> None:
    torch.manual_seed(9)
    model = _TinyRoutableLM(n_layers=4, hidden=16, inter=32, vocab=32).eval()
    with torch.no_grad():
        # Token 9 has trigger signal (coord 0 = +4) AND toxic veto signal (coord 4 = +4); token 7 has coord 4 = 0
        model.model.embed_tokens.weight[9].zero_()
        model.model.embed_tokens.weight[9, 0] = 4.0
        model.model.embed_tokens.weight[9, 4] = 4.0

    r_dir = torch.zeros(16)
    r_dir[0] = 1.0
    v_dir = torch.zeros(16)
    v_dir[4] = 1.0
    patch = MoIPatch(
        edits=[Edit(module="layers.2.self_attn.o_proj", u=_unit_vec(16, 71), alpha=6.0)],
        router={"layer": 1, "readout": "last_token", "dir": r_dir, "thr": 0.0, "veto_dir": v_dir, "veto_thr": 1.0},
    )

    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "veto_test.moipatch"
        patch.save(out_p)
        loaded = MoIPatch.load(out_p)
        assert loaded.router is not None and loaded.router.get("veto_dir") is not None
        assert "router veto:" in loaded.describe()
        rep = loaded.audit(model)
        assert rep["has_router_veto"] is True and rep["fingerprint_ok"] is True

        hs = [torch.zeros(3, 2, 16) for _ in range(4)]
        hs[1][0, -1, 0] = -2.0                    # control -> score <= thr -> False
        hs[1][1, -1, 0] = 2.0                     # benign trigger -> score > thr & veto <= thr -> True
        hs[1][2, -1, 0] = 2.0                     # toxic trigger -> score > thr BUT veto > thr -> False
        hs[1][2, -1, 4] = 3.0
        assert loaded.fires(hs).tolist() == [False, True, False]

        mixed_ids = torch.tensor([[2, 3, 1], [2, 3, 7], [2, 3, 9]], dtype=torch.long)
        with torch.no_grad():
            base_logits = model(mixed_ids).logits
            with loaded.attach(model) as ph:
                ph.on()
                patched_logits = model(mixed_ids).logits
            with loaded.enable_routed(model) as h:
                routed_logits = model(mixed_ids).logits
                assert h.stats["last_fired"] == [False, True, False]
                assert h.stats["fired"] == 1
                assert torch.equal(routed_logits[0], base_logits[0])
                assert torch.allclose(routed_logits[1], patched_logits[1], atol=1e-4, rtol=1e-4)
                assert torch.equal(routed_logits[2], base_logits[2])


if __name__ == "__main__":
    test_moipatch_save_load_attach_and_restore()
    test_make_moipatch_sycophancy_and_cli()
    test_real_p50c_loop_moipatch_artifact()
    test_enable_routed_fast_single_pass_and_restore()
    test_enable_routed_late_router_and_kv_cache_safety()
    test_enable_routed_ngram_loop_gate()
    test_enable_routed_compose_two_patches_and_out_of_order_close()
    test_enable_routed_real_hf_llama_generate()
    test_enable_routed_always_on_and_fingerprint_rejection()
    test_audit_moipatch_structural_and_end_to_end()
    test_router_veto_save_load_fires_and_enable_routed()
    print("ALL MOIPATCH TESTS PASSED!")
