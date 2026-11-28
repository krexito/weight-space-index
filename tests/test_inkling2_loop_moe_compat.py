"""CPU unit tests for WSI compatibility with multimodal looped-MoE architectures (`inkling2_loop`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint as torch_checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi.capture import find_blocks, find_mlps  # noqa: E402
from wsi.moipatch import Edit, MoIPatch, _find_final_norm, _sha256  # noqa: E402
from wsi.router_cal import calibrate_safety_router_on_model  # noqa: E402
from wsi.serve import bake_moipatch  # noqa: E402
from wsi.tracer import (  # noqa: E402
    _find_attn_out_projs,
    _find_final_norm_module,
    attribute_direction_writers,
    capture_causal_trace,
    diagnose_causal_traces,
    peel_causal_circuit,
    screen_patches_from_traces,
)


class _ToyDecoyBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.01 * self.self_attn.o_proj(x)


class _ToyMoEExpert(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)))


class _ToyInklingBlock(nn.Module):
    def __init__(self, hidden: int, n_experts: int = 16) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.wo_ud = nn.Linear(hidden, hidden, bias=False)

        self.post_attn_conv = nn.Module()
        self.post_attn_conv.in_proj = nn.Linear(hidden, hidden, bias=False)
        self.post_attn_conv.out_proj = nn.Linear(hidden, hidden, bias=False)

        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(hidden, n_experts, bias=False)
        self.mlp.experts = nn.ModuleList([_ToyMoEExpert(hidden) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.self_attn.v_proj(x))
        x = x + self.self_attn.wo_ud(v)
        c = torch.tanh(self.post_attn_conv.in_proj(x))
        x = x + self.post_attn_conv.out_proj(c)
        # Lightweight top-2 MoE contribution (zero dense/shared MLPs)
        probs = F.softmax(self.mlp.gate(x), dim=-1)
        top_w, top_idx = probs.topk(2, dim=-1)
        moe_out = torch.zeros_like(x)
        for k_i in range(2):
            eid = int(top_idx[0, -1, k_i].item())
            moe_out = moe_out + top_w[..., k_i : k_i + 1] * self.mlp.experts[eid](x)
        return x + 0.02 * moe_out


class _ToyInkling2LoopMoEMultimodalLM(nn.Module):
    """Toy multimodal looped-MoE LM mimicking `thinkingmachines/Inkling-2-Mini`:
    - `vision_tower.layers` (6 decoy blocks) & `mtp.layers` (2 decoy blocks)
    - `model.language_model.layers` (6 physical blocks: 0..1 prelude, 2..3 looped 3x, 4..5 coda)
    - 16 routed experts per block (`n_shared_experts=0`, zero dense MLPs)
    - Returns 11 logical `hidden_states` (`1 embed + 10 logical visits`), not `7`
    - Refusal switch planted in looped `layers.2.self_attn.wo_ud` + Hydra `conv` backup in coda `layers.4.post_attn_conv.out_proj`
    """

    def __init__(self, vocab: int = 48, hidden: int = 24, n_experts: int = 16) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.is_gradient_checkpointing = False
        self.config = SimpleNamespace(
            model_type="inkling2_loop",
            text_config=SimpleNamespace(
                model_type="inkling2_loop_text",
                hidden_size=hidden,
                num_attention_heads=4,
                n_routed_experts=n_experts,
                n_shared_experts=0,
                loop_config={"core_start": 2, "core_end": 4, "num_passes": 3},
            ),
            eos_token_id=2,
            pad_token_id=0,
        )

        self.vision_tower = nn.Module()
        self.vision_tower.layers = nn.ModuleList([_ToyDecoyBlock(hidden) for _ in range(6)])
        self.mtp = nn.Module()
        self.mtp.layers = nn.ModuleList([_ToyDecoyBlock(hidden) for _ in range(2)])

        self.model = nn.Module()
        lm = nn.Module()
        lm.embed_tokens = nn.Embedding(vocab, hidden)
        lm.layers = nn.ModuleList([_ToyInklingBlock(hidden, n_experts=n_experts) for _ in range(6)])
        lm.norm = nn.LayerNorm(hidden)
        self.model.language_model = lm
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        with torch.no_grad():
            for p in self.parameters():
                p.normal_(0.0, 0.005)
            lm.norm.reset_parameters()

            # Channels:
            # dim 2: normal prose/helpful feature
            # dim 3: refusal trigger feature (shared by OR-Bench trigger & toxic)
            # dim 4: genuine harmful-intent feature (toxic only)
            # dim 0: primary refusal output channel (written 3x by L2.self_attn.wo_ud)
            # dim 6: backup refusal output channel (written by L4.post_attn_conv.out_proj)
            # dim 1: helpful answer channel -> token 12
            lm.embed_tokens.weight.zero_()
            for t in range(vocab):
                if 20 <= t < 38:
                    lm.embed_tokens.weight[t, 3] = 3.0
                    lm.embed_tokens.weight[t, 2] = 0.6
                    if 30 <= t < 38:
                        lm.embed_tokens.weight[t, 4] = 3.0
                else:
                    lm.embed_tokens.weight[t, 2] = 1.5

            for blk in lm.layers:
                blk.self_attn.v_proj.weight.zero_()
                blk.self_attn.wo_ud.weight.zero_()
                blk.post_attn_conv.in_proj.weight.zero_()
                blk.post_attn_conv.out_proj.weight.zero_()
                blk.mlp.gate.weight.zero_()
                for ex in blk.mlp.experts:
                    ex.up_proj.weight.zero_()
                    ex.down_proj.weight.zero_()

            # Normal helpful writer in layer 1 (reads dim 2 -> writes dim 1)
            b1 = lm.layers[1]
            b1.self_attn.v_proj.weight[1, 2] = 0.8
            b1.self_attn.wo_ud.weight[1, 1] = 1.1

            # Primary looped refusal switch in layer 2 (reads dim 3 -> writes dim 0 on all 3 passes)
            # Plus sparse prose column 1 (only fires on normal dim 2) so Fisher sparse prefers col 0!
            b2 = lm.layers[2]
            b2.self_attn.v_proj.weight[0, 3] = 1.0
            b2.self_attn.wo_ud.weight[0, 0] = 0.8
            b2.self_attn.v_proj.weight[1, 2] = 0.9
            b2.self_attn.wo_ud.weight[1, 1] = 0.25

            # Backup conv refusal writer in coda layer 4 (reads dim 3 -> writes dim 6)
            b4 = lm.layers[4]
            b4.post_attn_conv.in_proj.weight[0, 3] = 1.0
            b4.post_attn_conv.out_proj.weight[6, 0] = 1.4

            # Readout head: token 25 = refusal (moderate logit so softmax stays unsaturated),
            # token 12 = helpful solution, token 2 = eos
            self.lm_head.weight.zero_()
            self.lm_head.weight[25, 0] = 1.0
            self.lm_head.weight[25, 6] = 0.8
            self.lm_head.weight[12, 1] = 2.0
            self.lm_head.weight[2, 2] = 0.15

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def gradient_checkpointing_enable(self) -> None:
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.is_gradient_checkpointing = False

    def _run_block(self, blk: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.is_gradient_checkpointing and torch.is_grad_enabled():
            return torch_checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        **_kw,
    ) -> SimpleNamespace:
        lm = self.model.language_model
        x = lm.embed_tokens(input_ids)
        hs = [x] if output_hidden_states else None

        # Prelude: physical blocks 0..1 (run once)
        for bi in (0, 1):
            x = self._run_block(lm.layers[bi], x)
            if hs is not None:
                hs.append(x)

        # Loop core: physical blocks 2..3 (run 3 times)
        for _pass in range(3):
            for bi in (2, 3):
                x = self._run_block(lm.layers[bi], x)
                if hs is not None:
                    hs.append(x)

        # Coda: physical blocks 4..5 (run once)
        x = self._run_block(lm.layers[4], x)
        if hs is not None:
            hs.append(x)
        x = self._run_block(lm.layers[5], x)
        x = lm.norm(x)
        if hs is not None:
            hs.append(x)

        return SimpleNamespace(logits=self.lm_head(x), hidden_states=tuple(hs) if hs is not None else None)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 3, **_kw) -> torch.Tensor:
        cur = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(cur, use_cache=False)
            nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
        return cur


class _ToyInklingTok:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if "trigger:" in text:
            num = int(text.split("trigger:", 1)[1].split()[0])
            return [20 + (num % 10), 21 + (num % 9)]
        if "toxic:" in text:
            num = int(text.split("toxic:", 1)[1].split()[0])
            return [30 + (num % 8), 31 + (num % 7)]
        num = sum(ord(c) for c in text) % 8
        return [10 + num, 11 + num]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if any(int(t) == 25 for t in ids):
            return "I cannot fulfill that request."
        return "Sure, here is the answer: " + " ".join(str(t) for t in ids)


def test_discovery_multimodal_moe_loop() -> None:
    model = _ToyInkling2LoopMoEMultimodalLM().eval()
    bname, blocks = find_blocks(model)
    assert bname == "model.language_model.layers"
    assert len(blocks) == 6
    assert find_mlps(model) == []

    attns = _find_attn_out_projs(model, 24)
    assert len(attns) == 12
    assert all(k.startswith("model.language_model.layers.") for k in attns)
    assert all(k.endswith((".self_attn.wo_ud", ".post_attn_conv.out_proj")) for k in attns)
    assert not any("vision_tower" in k or "mtp" in k or ".experts." in k for k in attns)

    assert _find_final_norm_module(model) is model.model.language_model.norm
    assert _find_final_norm(model) is model.model.language_model.norm


@pytest.mark.parametrize("grad_ckpt", [False, True])
def test_loop_aware_causal_trace_and_finite_diff(grad_ckpt: bool) -> None:
    model = _ToyInkling2LoopMoEMultimodalLM().eval()
    # Prompt tokens [20, 21] followed by refusal tokens [25, 25]
    ids = [20, 21, 25, 25]
    ct = capture_causal_trace(
        model,
        ids,
        target_span=(2, 4),
        healthy_span=(0, 2),
        grad_mode="compressed",
        healthy_cost="fisher",
        grad_checkpointing=grad_ckpt,
        top_k_mlp=8,
        group_size=8,
        hadamard_block=8,
        top_k_attn=0,
        capture_cache=False,
    )
    assert len(ct.grad_summary) == 12
    gs2 = ct.grad_summary["layers.2.self_attn.wo_ud"]
    assert gs2["n_passes"] == 3
    assert len(gs2["pass_causal_attr"]) == 3
    assert all(abs(v) > 0.0 for v in gs2["pass_causal_attr"])
    assert ct.grad_summary["layers.0.self_attn.wo_ud"]["n_passes"] == 1
    assert ct.grad_summary["layers.4.self_attn.wo_ud"]["n_passes"] == 1

    # Finite-difference check on looped weight W = layers.2.self_attn.wo_ud.weight
    # Perturb W' = W - eps * u (u^T W) along e_0 and verify 3-pass summed gradient matches FD
    mod2 = model.model.language_model.layers[2].self_attn.wo_ud
    w_orig = mod2.weight.detach().clone()
    u = torch.zeros(24, dtype=torch.float32)
    u[0] = 1.0
    eps = 1e-3
    inp = torch.tensor([ids], dtype=torch.long)

    def _target_lp() -> float:
        with torch.no_grad():
            logits = model(input_ids=inp).logits[0, 1:3].float()
            lp = F.log_softmax(logits, dim=-1).gather(1, inp[0, 2:4, None]).squeeze(1)
            return float(lp.mean().item())

    base_lp = _target_lp()
    with torch.no_grad():
        mod2.weight.copy_(w_orig - eps * torch.outer(u, u @ w_orig))
    try:
        pert_lp = _target_lp()
    finally:
        with torch.no_grad():
            mod2.weight.copy_(w_orig)

    actual_delta = pert_lp - base_lp
    g_t = ct.dequant_grad("layers.2.self_attn.wo_ud", "target")
    h_in = ct.dequant_head_in("model.language_model.layers.2.self_attn.wo_ud")
    assert g_t is not None and h_in is not None
    assert g_t.shape[0] == 3 * len(ids)
    assert h_in.shape[0] == 3 * len(ids)
    y_mat = h_in @ w_orig.float().t()
    pred_delta = -eps * float(((g_t @ u) * (y_mat @ u)).sum().item())
    assert abs(actual_delta) > 1e-5
    rel_err = abs(actual_delta - pred_delta) / abs(actual_delta)
    assert rel_err < 0.05, f"3-pass FD mismatch (grad_ckpt={grad_ckpt}): actual={actual_delta}, pred={pred_delta}"


def test_loop_aware_direction_writers_and_peeling() -> None:
    model = _ToyInkling2LoopMoEMultimodalLM().eval()
    trig_seqs = [[20, 21], [22, 23]]
    ctrl_seqs = [[10, 11], [12, 13]]
    tdir = torch.zeros(24, dtype=torch.float32)
    tdir[0] = 1.0
    tdir[6] = 0.5

    attr = attribute_direction_writers(
        model,
        trig_seqs,
        contrast_inputs=ctrl_seqs,
        target_dir=tdir,
        target_layer=6,
        layer_convention="hidden_states",
    )
    assert attr["use_final_hidden"] is True
    assert "layers.2.self_attn.wo_ud" in attr["module_pass_contribs"]
    p_vals = attr["module_pass_contribs"]["layers.2.self_attn.wo_ud"]
    assert len(p_vals) == 3
    assert sum(p_vals) == pytest.approx(attr["module_contribs"]["layers.2.self_attn.wo_ud"], abs=1e-5)
    assert len(attr["delta_module_pass_contribs"]["layers.2.self_attn.wo_ud"]) == 3

    peel = peel_causal_circuit(
        model,
        trig_seqs,
        n_rounds=2,
        alpha=1.2,
        target_dir=tdir,
        target_layer=6,
        target_layer_convention="hidden_states",
        top_n_neurons=4,
        grad_checkpointing=False,
    )
    assert peel["restore_exact_ok"] is True
    assert len(peel["rounds"]) == 2
    peeled_mods = [m for r in peel["rounds"] for m in r["selected_modules"]]
    assert "layers.2.self_attn.wo_ud" in peeled_mods
    assert "layers.4.post_attn_conv.out_proj" in peeled_mods
    assert peel["score_trajectory"][-1] < peel["score_trajectory"][0]


def test_mixed_loop_and_coda_screening() -> None:
    model = _ToyInkling2LoopMoEMultimodalLM().eval()
    traces = [
        capture_causal_trace(
            model, [20, 21, 25, 25], target_span=(2, 4), healthy_span=(0, 2),
            grad_mode="compressed", healthy_cost="fisher", top_k_mlp=8, group_size=8, hadamard_block=8, top_k_attn=0, capture_cache=False,
        ),
        capture_causal_trace(
            model, [22, 23, 25, 25], target_span=(2, 4), healthy_span=(0, 2),
            grad_mode="compressed", healthy_cost="fisher", top_k_mlp=8, group_size=8, hadamard_block=8, top_k_attn=0, capture_cache=False,
        ),
        capture_causal_trace(
            model, [10, 11, 12, 12], target_span=(2, 4), healthy_span=(0, 2),
            grad_mode="compressed", healthy_cost="fisher", top_k_mlp=8, group_size=8, hadamard_block=8, top_k_attn=0, capture_cache=False,
        ),
        capture_causal_trace(
            model, [12, 13, 12, 12], target_span=(2, 4), healthy_span=(0, 2),
            grad_mode="compressed", healthy_cost="fisher", top_k_mlp=8, group_size=8, hadamard_block=8, top_k_attn=0, capture_cache=False,
        ),
    ]
    diag = diagnose_causal_traces(traces, healthy_cost="fisher", split_ab=True, model_or_weights=model)
    assert diag["directions"]
    w2 = model.model.language_model.layers[2].self_attn.wo_ud.weight.detach()
    w4 = model.model.language_model.layers[4].post_attn_conv.out_proj.weight.detach()
    u2 = diag["directions"]["layers.2.self_attn.wo_ud"].double()
    u4 = diag["directions"]["layers.4.post_attn_conv.out_proj"].double()
    cands = {
        "loop_only": MoIPatch(edits=[Edit("layers.2.self_attn.wo_ud", u2, 2.0, idx=torch.tensor([0]), base_sha256=_sha256(w2), base_dtype="float32", shape=tuple(w2.shape))]),
        "coda_only": MoIPatch(edits=[Edit("layers.4.post_attn_conv.out_proj", u4, 2.0, idx=torch.tensor([0]), base_sha256=_sha256(w4), base_dtype="float32", shape=tuple(w4.shape))]),
        "mixed_loop_and_coda": MoIPatch(edits=[
            Edit("layers.2.self_attn.wo_ud", u2, 2.0, idx=torch.tensor([0]), base_sha256=_sha256(w2), base_dtype="float32", shape=tuple(w2.shape)),
            Edit("layers.4.post_attn_conv.out_proj", u4, 2.0, idx=torch.tensor([0]), base_sha256=_sha256(w4), base_dtype="float32", shape=tuple(w4.shape)),
        ]),
    }
    rows = screen_patches_from_traces(traces, cands, model, healthy_cost="fisher")
    assert len(rows) == 3
    for r in rows:
        assert torch.isfinite(torch.tensor(r["grad_supp"]))
        assert torch.isfinite(torch.tensor(r["grad_dnll"]))


def test_router_calibration_and_routed_handle_on_looped_logical_hs(tmp_path: Path) -> None:
    model = _ToyInkling2LoopMoEMultimodalLM().eval()
    tok = _ToyInklingTok()
    assert len(model(torch.tensor([[20, 21]]), output_hidden_states=True).hidden_states) == 11

    w2 = model.model.language_model.layers[2].self_attn.wo_ud.weight.detach()
    w4 = model.model.language_model.layers[4].post_attn_conv.out_proj.weight.detach()
    u0 = torch.zeros(24, dtype=torch.float64)
    u0[0] = 1.0
    u6 = torch.zeros(24, dtype=torch.float64)
    u6[6] = 1.0
    payload = MoIPatch(
        edits=[
            Edit("layers.2.self_attn.wo_ud", u0, 1.5, idx=torch.tensor([0]), base_sha256=_sha256(w2), base_dtype="float32", shape=tuple(w2.shape)),
            Edit("layers.4.post_attn_conv.out_proj", u6, 1.5, idx=torch.tensor([0]), base_sha256=_sha256(w4), base_dtype="float32", shape=tuple(w4.shape)),
        ]
    )
    out_p = tmp_path / "sfw.moipatch"
    res = calibrate_safety_router_on_model(
        model,
        tok,
        payload,
        toxic_prompts=[f"toxic:{i} unsafe" for i in range(6)],
        benign_refused_prompts=[f"trigger:{i} benign" for i in range(6)],
        control_prompts=[f"normal:{i} chat" for i in range(6)],
        chat=False,
        layer_candidates=[5, 6],
        out_patch_path=out_p,
    )
    assert res["selected_layer"] in (5, 6)
    assert out_p.exists()

    loaded = MoIPatch.load(out_p)
    assert loaded.router is not None
    if loaded.router["layer"] == 6:
        assert loaded.router.get("final_layer") is True

    with loaded.enable_routed(model) as h:
        trig_ids = torch.tensor([tok.encode("trigger:1 benign")], dtype=torch.long)
        tox_ids = torch.tensor([tok.encode("toxic:1 unsafe")], dtype=torch.long)
        ctrl_ids = torch.tensor([tok.encode("normal:1 chat")], dtype=torch.long)

        gen_trig = model.generate(trig_ids, max_new_tokens=2)[0, 2:].tolist()
        assert 25 not in gen_trig

        gen_tox = model.generate(tox_ids, max_new_tokens=2)[0, 2:].tolist()
        assert 25 in gen_tox

        h.reset_stats()
        out_ctrl_routed = model(ctrl_ids).logits
        assert h.stats["fired"] == 0
        assert h.fingerprint_ok()

    with torch.no_grad():
        out_ctrl_base = model(ctrl_ids).logits
    assert torch.equal(out_ctrl_routed, out_ctrl_base)

    # Verify RoutedPatchHandle waits for Pass 3 (not Pass 1) when router["layer"] == 3 (inside loop core 2..3)
    loop_dir = torch.zeros(24, dtype=torch.float32)
    loop_dir[0] = 1.0
    loop_patch = MoIPatch(
        edits=[payload.edits[1]],
        router={"layer": 3, "readout": "last_token", "dir": loop_dir, "thr": 0.5},
    )
    with loop_patch.enable_routed(model) as lh:
        pass_obs: list[float] = []
        blk2_seen = [0]

        def _check_pass(_m, _inp, out):
            blk2_seen[0] += 1
            pass_obs.append(float(out[0, -1, 0].item()))

        hk = model.model.language_model.layers[2].register_forward_hook(_check_pass)
        try:
            _ = model(torch.tensor([tok.encode("trigger:2 benign")], dtype=torch.long))
        finally:
            hk.remove()
        assert blk2_seen[0] == 5
        assert lh.stats["last_scores"] is not None
        # Pass 3 residual score must match final pass observation and exceed Pass 1 score
        assert lh.stats["last_scores"][0] == pytest.approx(pass_obs[-1], abs=1e-4)
        assert lh.stats["last_scores"][0] > pass_obs[0] + 1.0


def test_bake_moipatch_skips_vision_and_mtp_collisions(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "base_ckpt"
    ckpt_dir.mkdir(parents=True)
    out_dir = tmp_path / "baked_ckpt"

    w_lang = torch.randn(24, 24, dtype=torch.bfloat16)
    w_vis = torch.randn(24, 24, dtype=torch.bfloat16)
    w_mtp = torch.randn(24, 24, dtype=torch.bfloat16)
    w_other = torch.randn(24, 24, dtype=torch.bfloat16)

    save_file(
        {
            "vision_tower.layers.2.self_attn.wo_ud.weight": w_vis,
            "mtp.layers.2.self_attn.wo_ud.weight": w_mtp,
            "model.language_model.layers.2.self_attn.wo_ud.weight": w_lang,
        },
        str(ckpt_dir / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"model.language_model.layers.4.self_attn.wo_ud.weight": w_other},
        str(ckpt_dir / "model-00002-of-00002.safetensors"),
    )
    index = {
        "weight_map": {
            "vision_tower.layers.2.self_attn.wo_ud.weight": "model-00001-of-00002.safetensors",
            "mtp.layers.2.self_attn.wo_ud.weight": "model-00001-of-00002.safetensors",
            "model.language_model.layers.2.self_attn.wo_ud.weight": "model-00001-of-00002.safetensors",
            "model.language_model.layers.4.self_attn.wo_ud.weight": "model-00002-of-00002.safetensors",
        }
    }
    (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (ckpt_dir / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}), encoding="utf-8")

    u = torch.zeros(24, dtype=torch.float64)
    u[0] = 1.0
    patch = MoIPatch(edits=[Edit(module="layers.2.self_attn.wo_ud", u=u, alpha=1.0, idx=torch.tensor([0, 1]))])
    patch.fingerprint_from_checkpoint(ckpt_dir)
    assert patch.edits[0].base_sha256 == _sha256(w_lang)

    rep = bake_moipatch(ckpt_dir, patch, out_dir, copy_untouched="copy")
    assert rep["n_touched_shards"] == 1
    assert rep["modules"]["layers.2.self_attn.wo_ud"]["base_sha256"] == _sha256(w_lang)

    with safe_open(str(out_dir / "model-00001-of-00002.safetensors"), "pt") as sf:
        assert torch.equal(sf.get_tensor("vision_tower.layers.2.self_attn.wo_ud.weight"), w_vis)
        assert torch.equal(sf.get_tensor("mtp.layers.2.self_attn.wo_ud.weight"), w_mtp)
        assert not torch.equal(sf.get_tensor("model.language_model.layers.2.self_attn.wo_ud.weight"), w_lang)
    with safe_open(str(out_dir / "model-00002-of-00002.safetensors"), "pt") as sf:
        assert torch.equal(sf.get_tensor("model.language_model.layers.4.self_attn.wo_ud.weight"), w_other)
