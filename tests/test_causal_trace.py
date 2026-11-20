"""Fast CPU unit tests for causal Forward+Backward CompactTrace, offline causal diagnosis, and offline screening."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from experiment_causal_diagnose import _PlantedDecoyCausalLM  # noqa: E402
from wsi import (  # noqa: E402
    CompactTrace,
    Edit,
    MoIPatch,
    capture_causal_trace,
    capture_compact,
    diagnose_causal_traces,
    gradient_screen_candidates,
    screen_patches_from_traces,
)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / max(dx * dy, 1e-12)


def test_capture_causal_trace_compressed_and_summary_roundtrip() -> None:
    model = _PlantedDecoyCausalLM(plant_kind="mlp", vocab=32, hidden=32, inter=64, seed=1).eval()
    ids = [1, 2, 3, 4, 25, 25, 25, 25]

    tr_plain = capture_compact(model, ids, top_k_mlp=16, group_size=16, hadamard_block=16, top_k_attn=0, capture_cache=False)
    assert tr_plain.grad_mode == "none"
    assert tr_plain.byte_breakdown()["grad"] == 0

    tr_comp = capture_causal_trace(
        model,
        ids,
        target_span=(4, 8),
        healthy_span=(0, 3),
        grad_mode="compressed",
        top_k_mlp=16,
        group_size=16,
        hadamard_block=16,
        top_k_attn=0,
        capture_cache=False,
    )
    assert tr_comp.grad_mode == "compressed"
    assert tr_comp.byte_breakdown()["grad"] > 0
    assert tr_comp.dequant_grad("layers.2.mlp.down_proj", "target") is not None
    assert tr_comp.dequant_grad("layers.2.mlp.down_proj", "healthy") is not None
    assert tr_comp.meta["grad_n_target"] == 4
    assert tr_comp.meta["grad_n_healthy"] == 3

    tr_sum = capture_causal_trace(
        model,
        ids,
        target_span=(4, 8),
        healthy_span=(0, 3),
        grad_mode="summary",
        top_k_mlp=16,
        group_size=16,
        hadamard_block=16,
        top_k_attn=0,
        capture_cache=False,
    )
    assert tr_sum.grad_mode == "summary"
    assert tr_sum.dequant_grad("layers.2.mlp.down_proj", "target") is None
    assert tr_sum.byte_breakdown()["grad"] > 0
    assert tr_sum.byte_breakdown()["grad"] < tr_comp.byte_breakdown()["grad"]

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        tr_plain.save(p / "plain")
        tr_comp.save(p / "comp")
        tr_sum.save(p / "sum")

        ld_plain = CompactTrace.load(p / "plain")
        ld_comp = CompactTrace.load(p / "comp")
        ld_sum = CompactTrace.load(p / "sum")

        assert ld_plain.grad_mode == "none" and ld_plain.byte_breakdown()["grad"] == 0
        assert ld_comp.byte_breakdown() == tr_comp.byte_breakdown()
        assert ld_sum.byte_breakdown() == tr_sum.byte_breakdown()
        assert torch.equal(
            ld_comp.dequant_grad("layers.2.mlp.down_proj", "target"),
            tr_comp.dequant_grad("layers.2.mlp.down_proj", "target"),
        )
        assert torch.equal(
            ld_comp.grad_summary["layers.2.mlp.down_proj"]["c_target_eigvecs"],
            tr_comp.grad_summary["layers.2.mlp.down_proj"]["c_target_eigvecs"],
        )


@pytest.mark.parametrize("plant_kind", ["mlp", "attn"])
def test_causal_diagnosis_beats_passive_decoy_on_planted_mlp_and_attn_circuits(plant_kind: str) -> None:
    model = _PlantedDecoyCausalLM(plant_kind=plant_kind, vocab=32, hidden=32, inter=64, seed=42).eval()
    seqs = [
        [1, 2, 3, 4, 25, 25, 25, 25],
        [2, 3, 4, 5, 25, 25, 25, 25],
        [3, 4, 5, 6, 25, 25, 25, 25],
        [4, 5, 6, 7, 25, 25, 25, 25],
    ]
    traces = [
        capture_causal_trace(
            model,
            s,
            target_span=(4, 8),
            healthy_span=(0, 3),
            grad_mode="compressed",
            top_k_mlp=16,
            group_size=16,
            hadamard_block=16,
            top_k_attn=0,
            capture_cache=False,
        )
        for s in seqs
    ]

    diag = diagnose_causal_traces(traces, split_ab=True, top_k_sites=6, top_n_neurons=8)
    want_mod = "layers.2.mlp.down_proj" if plant_kind == "mlp" else "layers.2.self_attn.o_proj"

    # Passive norm is fooled by the massive Layer-1 nullspace decoy
    assert diag["summary"]["top_passive_site"] == "layers.1.mlp.down_proj"
    # Unified causal diagnosis picks the true Layer-2 MLP or Attention switch #1
    assert diag["summary"]["top_causal_site"] == want_mod
    assert diag["causal_rank"][0] == want_mod

    e0 = torch.zeros(32)
    e0[0] = 1.0
    cos = float(F.cosine_similarity(diag["directions"][want_mod][None], e0[None]).abs().item())
    assert cos > 0.90, f"Recovered direction missed planted axis e0: |cos|={cos:.4f}"

    if plant_kind == "mlp":
        assert diag["top_neurons"][want_mod][0] == 7


def test_offline_screen_matches_online_gradient_screen() -> None:
    torch.manual_seed(9)
    model = _PlantedDecoyCausalLM(plant_kind="mlp", vocab=32, hidden=32, inter=64, seed=9).eval()
    # Add non-trivial attention output at layer 2 as well so both MLP & Attn candidates have signal
    with torch.no_grad():
        model.model.layers[2].self_attn.v_proj.weight[0, 3] = 0.6
        model.model.layers[2].self_attn.o_proj.weight[0, 0] = 0.9

    seqs = [
        [1, 2, 3, 4, 25, 25, 25, 25],
        [2, 3, 4, 5, 25, 25, 25, 25],
        [3, 4, 5, 6, 25, 25, 25, 25],
        [4, 5, 6, 7, 25, 25, 25, 25],
    ]
    traces = [
        capture_causal_trace(
            model,
            s,
            target_span=(4, 8),
            healthy_span=(0, 3),
            grad_mode="compressed",
            top_k_mlp=64,
            group_size=16,
            hadamard_block=16,
            top_k_attn=0,
            capture_cache=False,
        )
        for s in seqs
    ]

    e0 = torch.zeros(32); e0[0] = 1.0
    e1 = torch.zeros(32); e1[1] = 1.0
    e2 = torch.zeros(32); e2[2] = 1.0
    u_mix = (e0 + 0.5 * e2); u_mix = u_mix / u_mix.norm()

    cands = {
        "mlp2_e0_a6": MoIPatch(edits=[Edit(module="layers.2.mlp.down_proj", u=e0, alpha=6.0)]),
        "mlp2_e0_a2": MoIPatch(edits=[Edit(module="layers.2.mlp.down_proj", u=e0, alpha=2.0)]),
        "mlp2_sub_n7": MoIPatch(edits=[Edit(module="layers.2.mlp.down_proj", u=e0, alpha=4.0, idx=torch.tensor([7, 8, 9]))]),
        "mlp2_mix_a4": MoIPatch(edits=[Edit(module="layers.2.mlp.down_proj", u=u_mix, alpha=4.0)]),
        "attn2_e0_a6": MoIPatch(edits=[Edit(module="layers.2.self_attn.o_proj", u=e0, alpha=6.0)]),
        "attn2_e0_a2": MoIPatch(edits=[Edit(module="layers.2.self_attn.o_proj", u=e0, alpha=2.0)]),
        "mlp1_decoy": MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=e1, alpha=6.0)]),
        "mlp0_ctrl": MoIPatch(edits=[Edit(module="layers.0.mlp.down_proj", u=e2, alpha=4.0)]),
    }

    onl = gradient_screen_candidates(
        model,
        cands,
        healthy_seqs=[s[:4] for s in seqs],
        target_seqs=seqs,
        target_spans=[(4, 8)] * len(seqs),
    )

    # Guard against any forward pass on `model` during offline screening
    weights_dict = {
        "layers.0.mlp.down_proj": model.model.layers[0].mlp.down_proj.weight.detach().clone(),
        "layers.1.mlp.down_proj": model.model.layers[1].mlp.down_proj.weight.detach().clone(),
        "layers.2.mlp.down_proj": model.model.layers[2].mlp.down_proj.weight.detach().clone(),
        "layers.2.self_attn.o_proj": model.model.layers[2].self_attn.o_proj.weight.detach().clone(),
    }
    off = screen_patches_from_traces(traces, cands, weights_dict)

    assert off[0]["name"] == onl[0]["name"]
    off_by = {r["name"]: r for r in off}
    onl_by = {r["name"]: r for r in onl}
    names = list(cands)
    r_supp = _pearson([off_by[n]["grad_supp"] for n in names], [onl_by[n]["grad_supp"] for n in names])
    r_dnll = _pearson([off_by[n]["grad_dnll"] for n in names], [onl_by[n]["grad_dnll"] for n in names])
    assert r_supp > 0.98, f"Offline vs online grad_supp correlation too low: {r_supp}"
    assert r_dnll > 0.98, f"Offline vs online grad_dnll correlation too low: {r_dnll}"


class _CheckpointedToyLM(nn.Module):
    """Toy model supporting HF-style gradient_checkpointing_enable/disable to test hook safety."""

    def __init__(self, vocab: int = 24, hidden: int = 16, inter: int = 32) -> None:
        super().__init__()
        from experiment_causal_diagnose import _PlantedBlock

        torch.manual_seed(3)
        self.is_gradient_checkpointing = False
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_PlantedBlock(hidden, inter, 2) for _ in range(2)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def gradient_checkpointing_enable(self) -> None:
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.is_gradient_checkpointing = False

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        x = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            if self.is_gradient_checkpointing and torch.is_grad_enabled():
                x = torch_checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return SimpleNamespace(logits=self.lm_head(x))


def test_gradient_checkpointing_compatibility_and_state_restore() -> None:
    model = _CheckpointedToyLM().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.lm_head.weight.requires_grad_(True)
    assert model.is_gradient_checkpointing is False

    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    tr = capture_causal_trace(
        model,
        ids,
        target_span=(4, 8),
        healthy_span=(0, 3),
        grad_mode="compressed",
        grad_checkpointing=True,
        top_k_mlp=16,
        group_size=16,
        hadamard_block=16,
        top_k_attn=0,
        capture_cache=False,
    )
    assert model.is_gradient_checkpointing is False
    assert model.lm_head.weight.requires_grad is True
    assert model.model.layers[0].mlp.down_proj.weight.requires_grad is False

    g_t = tr.dequant_grad("layers.0.mlp.down_proj", "target")
    assert g_t is not None and float(g_t.abs().sum().item()) > 0.0


class _ParetoContrastLM(nn.Module):
    """Layer 0 writes e0 (shared with healthy predictions) and e1 (target-only), testing Pareto selection."""

    def __init__(self, vocab: int = 16, hidden: int = 16, inter: int = 32) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        blk = nn.Module()
        blk.mlp = nn.Module()
        blk.mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
        blk.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
        blk.mlp.down_proj = nn.Linear(inter, hidden, bias=False)

        def _blk_fwd(x: torch.Tensor) -> torch.Tensor:
            h = F.silu(blk.mlp.gate_proj(x)) * blk.mlp.up_proj(x)
            return x + blk.mlp.down_proj(h)

        blk.forward = _blk_fwd  # type: ignore[method-assign]
        self.model.layers = nn.ModuleList([blk])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        with torch.no_grad():
            for p in self.parameters():
                p.zero_()
            # Healthy predictions (token 1) depend heavily on dim 0 (`e0`), NOT dim 1 (`e1`)
            self.lm_head.weight[1, 0] = 6.0
            # Target predictions (token 10) depend on both dim 0 (stronger raw logit!) and dim 1
            self.lm_head.weight[10, 0] = 3.0
            self.lm_head.weight[10, 1] = 2.0

            # Input dim 2 is active everywhere -> fires unit 0 -> writes dim 0 (e0) everywhere
            self.model.embed_tokens.weight[:, 2] = 1.0
            # Input dim 3 is active only on trigger token 10 -> fires unit 1 -> writes dim 1 (e1)
            self.model.embed_tokens.weight[10, 3] = 2.0

            blk.mlp.gate_proj.weight[0, 2] = 2.0
            blk.mlp.up_proj.weight[0, 2] = 2.0
            blk.mlp.down_proj.weight[0, 0] = 2.0

            blk.mlp.gate_proj.weight[1, 3] = 1.5
            blk.mlp.up_proj.weight[1, 3] = 1.5
            blk.mlp.down_proj.weight[1, 1] = 1.5

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        x = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            x = blk(x)
        return SimpleNamespace(logits=self.lm_head(x))


def test_pareto_direction_avoids_healthy_destruction() -> None:
    model = _ParetoContrastLM().eval()
    seqs = [
        [1, 1, 1, 1, 10, 10, 10, 10],
        [2, 1, 1, 1, 10, 10, 10, 10],
        [3, 1, 1, 1, 10, 10, 10, 10],
        [4, 1, 1, 1, 10, 10, 10, 10],
    ]
    traces = [
        capture_causal_trace(
            model,
            s,
            target_span=(4, 8),
            healthy_span=(0, 3),
            grad_mode="summary",
            top_k_mlp=16,
            group_size=16,
            hadamard_block=16,
            top_k_attn=0,
            capture_cache=False,
        )
        for s in seqs
    ]
    diag = diagnose_causal_traces(traces, split_ab=True, top_k_sites=2, top_n_neurons=4, betas=(0.0, 0.5, 1.0, 2.0, 4.0))
    u = diag["directions"]["layers.0.mlp.down_proj"]
    e0 = torch.zeros(16); e0[0] = 1.0
    e1 = torch.zeros(16); e1[1] = 1.0
    cos_e1 = float(F.cosine_similarity(u[None], e1[None]).abs().item())
    cos_e0 = float(F.cosine_similarity(u[None], e0[None]).abs().item())
    assert cos_e1 > 0.90, f"Expected Pareto direction near e1 (clean target switch), got cos_e1={cos_e1:.4f}, cos_e0={cos_e0:.4f}"
    assert cos_e0 < 0.35
    assert diag["top_neurons"]["layers.0.mlp.down_proj"][0] == 1


def test_screen_patch_candidates_prefilter_with_traces(tmp_path, monkeypatch) -> None:
    import wsi.tracer as tr_mod
    from wsi import screen_patch_candidates

    model = _PlantedDecoyCausalLM(plant_kind="mlp").eval()
    seqs = [
        [1, 2, 3, 4, 25, 25, 25, 25],
        [2, 3, 4, 5, 25, 25, 25, 25],
    ]
    traces = [
        capture_causal_trace(
            model,
            s,
            target_span=(4, 8),
            healthy_span=(0, 3),
            grad_mode="compressed",
            top_k_mlp=16,
            group_size=16,
            hadamard_block=16,
            top_k_attn=0,
            capture_cache=False,
        )
        for s in seqs
    ]
    # Save traces to disk as well to test passing directory paths via `traces=...`
    trace_dirs = []
    for i, ct in enumerate(traces):
        d = tmp_path / f"trace_{i}"
        ct.save(d)
        trace_dirs.append(str(d))

    u_e0 = torch.zeros(32); u_e0[0] = 1.0
    u_e1 = torch.zeros(32); u_e1[1] = 1.0
    cands = {
        "true_a4": MoIPatch(edits=[Edit(module="layers.2.mlp.down_proj", u=u_e0, alpha=4.0)]),
        "true_a2": MoIPatch(edits=[Edit(module="layers.2.mlp.down_proj", u=u_e0, alpha=2.0)]),
        "decoy_l1": MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=u_e1, alpha=4.0)]),
    }

    # Ensure online `gradient_screen_candidates` is NEVER called when `traces` are provided
    def _forbid_online_grad(*_a, **_kw):
        raise AssertionError("gradient_screen_candidates should not be called when causal traces are passed")

    monkeypatch.setattr(tr_mod, "gradient_screen_candidates", _forbid_online_grad)

    for tr_input in (traces, trace_dirs):
        rows = screen_patch_candidates(
            model,
            cands,
            healthy_seqs=[s[:4] for s in seqs],
            target_seqs=seqs,
            target_spans=[(4, 8)] * len(seqs),
            prefilter_top_k=2,
            traces=tr_input,
        )
        assert len(rows) == 2
        assert {r["name"] for r in rows} == {"true_a4", "true_a2"}
        for r in rows:
            assert "grad_supp" in r and "grad_dnll" in r and "grad_pareto" in r
            assert r["target_suppression"] > 0.0


def test_discover_compact_circuit_passive_causal_union_and_forward_judge() -> None:
    from wsi import discover_compact_circuit, screen_patch_candidates

    # Plant an Attention causal switch on Layer 2 with a massive Layer-1 MLP nullspace decoy
    model = _PlantedDecoyCausalLM(plant_kind="attn").eval()
    seqs = [
        [1, 2, 3, 4, 25, 25, 25, 25],
        [2, 3, 4, 5, 25, 25, 25, 25],
        [3, 4, 5, 6, 25, 25, 25, 25],
        [4, 5, 6, 7, 25, 25, 25, 25],
    ]
    traces = [
        (
            {"prompt_len": 1, "loop": True, "onset_resp": 3},
            capture_causal_trace(
                model,
                s,
                target_span=(4, 8),
                healthy_span=(0, 3),
                grad_mode="summary",
                top_k_mlp=16,
                group_size=16,
                hadamard_block=16,
                top_k_attn=0,
                capture_cache=False,
            ),
            None,
        )
        for s in seqs
    ]

    disc = discover_compact_circuit(traces, model=model, top_k_neurons=8, candidate_alphas=(4.0, 8.0, 12.0))
    # Passive ranking is fooled by the Layer-1 decoy, while causal diagnosis finds layers.2.self_attn.o_proj
    assert disc["mlp_ranked"][0] == 1
    assert disc["causal_diagnosis"] is not None
    assert disc["causal_diagnosis"]["summary"]["top_causal_site"] == "layers.2.self_attn.o_proj"
    assert "causal_top1" in disc["patches"] and "causal_attn_top1" in disc["patches"]

    grid = disc["candidate_grid"]
    assert any(k.startswith("passive_") for k in grid)
    assert any(k.startswith("causal_") for k in grid)

    # Forward screen (the real judge) evaluates the union across alphas=(4, 8, 12) and picks the causal attention fix
    ranked = screen_patch_candidates(
        model,
        grid,
        healthy_seqs=[s[:4] for s in seqs],
        target_seqs=seqs,
        target_spans=[(4, 8)] * len(seqs),
    )
    winner_name = ranked[0]["name"]
    winner_patch = grid[winner_name]
    assert winner_name.startswith("causal_"), f"Expected forward judge to pick a causal candidate from the union, got {winner_name}"
    assert winner_patch.edits[0].module == "layers.2.self_attn.o_proj"
    assert ranked[0]["target_suppression"] > 0.10
    by_name = {r["name"]: r for r in ranked}
    assert by_name["causal_top1_all_a4"]["target_suppression"] > by_name["passive_blind_top1_a4"]["target_suppression"] + 0.05


def test_fisher_healthy_cost_rejects_summary_mode() -> None:
    model = _PlantedDecoyCausalLM(plant_kind="mlp", vocab=32, hidden=32, inter=64, seed=5).eval()
    ids = [1, 2, 3, 4, 25, 25, 25, 25]

    with pytest.raises(ValueError, match="healthy_cost='fisher' requires per-token gradients"):
        capture_causal_trace(
            model,
            ids,
            target_span=(4, 8),
            healthy_span=(0, 3),
            grad_mode="summary",
            healthy_cost="fisher",
            top_k_mlp=16,
            group_size=16,
            hadamard_block=16,
            top_k_attn=0,
            capture_cache=False,
        )

    tr_sum = capture_causal_trace(
        model,
        ids,
        target_span=(4, 8),
        healthy_span=(0, 3),
        grad_mode="summary",
        top_k_mlp=16,
        group_size=16,
        hadamard_block=16,
        top_k_attn=0,
        capture_cache=False,
    )
    with pytest.raises(ValueError, match="healthy_cost='fisher' requires per-token gradients"):
        diagnose_causal_traces([tr_sum], healthy_cost="fisher")

    e0 = torch.zeros(32)
    e0[0] = 1.0
    cands = {"a4": MoIPatch(edits=[Edit(module="layers.2.mlp.down_proj", u=e0, alpha=4.0)])}
    with pytest.raises(ValueError, match="healthy_cost='fisher' requires per-token gradients"):
        screen_patches_from_traces([tr_sum], cands, model, healthy_cost="fisher")


def test_fisher_healthy_cost_penalizes_large_alpha_and_cancellation() -> None:
    model = _ParetoContrastLM().eval()
    # Keep token 10 as a moderate logit along dim 0 (+1.0, unsaturated) vs tokens 1 (+0.8) and 2 (-0.8)
    # so target suppression along e0 is strictly positive, while healthy tokens 1 and 2 have opposite sensitivities
    # along dim 0 so linear drift cancels out and empirical Fisher scales quadratically with alpha^2.
    with torch.no_grad():
        model.lm_head.weight[1, 0] = 0.8
        model.lm_head.weight[2, 0] = -0.8
        model.lm_head.weight[10, 0] = 1.0
        model.lm_head.weight[10, 1] = 0.5
        model.model.layers[0].mlp.gate_proj.weight[0, 2] = 0.8
        model.model.layers[0].mlp.up_proj.weight[0, 2] = 0.8
        model.model.layers[0].mlp.down_proj.weight[0, 0] = 0.8
    seqs = [
        [1, 2, 1, 2, 10, 10, 10, 10],
        [2, 1, 2, 1, 10, 10, 10, 10],
    ]
    traces = [
        capture_causal_trace(
            model,
            s,
            target_span=(4, 8),
            healthy_span=(0, 4),
            grad_mode="compressed",
            center="healthy",
            healthy_cost="fisher",
            top_k_mlp=16,
            group_size=16,
            hadamard_block=16,
            top_k_attn=0,
            capture_cache=False,
        )
        for s in seqs
    ]
    e0 = torch.zeros(16)
    e0[0] = 1.0
    cands = {
        "a4": MoIPatch(edits=[Edit(module="layers.0.mlp.down_proj", u=e0, alpha=4.0)]),
        "a8": MoIPatch(edits=[Edit(module="layers.0.mlp.down_proj", u=e0, alpha=8.0)]),
        "a12": MoIPatch(edits=[Edit(module="layers.0.mlp.down_proj", u=e0, alpha=12.0)]),
    }
    off_f = screen_patches_from_traces(traces, cands, model, healthy_cost="fisher", max_nll_delta=100.0)
    onl_f = gradient_screen_candidates(
        model,
        cands,
        healthy_seqs=[s[:5] for s in seqs],
        target_seqs=seqs,
        target_spans=[(4, 8)] * len(seqs),
        healthy_cost="fisher",
        max_nll_delta=100.0,
    )
    by_off = {r["name"]: r for r in off_f}
    by_onl = {r["name"]: r for r in onl_f}
    assert by_off["a4"]["grad_supp"] > 0.0
    assert by_off["a4"]["grad_dnll"] > 0.0
    # Fisher scales quadratically with alpha: dnll(a12) / dnll(a4) == (12/4)^2 == 9
    ratio = by_off["a12"]["grad_dnll"] / max(by_off["a4"]["grad_dnll"], 1e-12)
    assert 8.5 <= ratio <= 9.5, f"Expected quadratic alpha^2 scaling (~9.0), got {ratio:.3f}"
    # Because grad_supp > 0 (~ alpha) and Fisher ~ alpha^2, grad_pareto orders a4 > a8 > a12
    assert by_off["a4"]["grad_pareto"] > by_off["a8"]["grad_pareto"] > by_off["a12"]["grad_pareto"]
    assert [r["name"] for r in off_f] == ["a4", "a8", "a12"]
    assert [r["name"] for r in onl_f] == ["a4", "a8", "a12"]


class _HighBaselineLateDecoyLM(nn.Module):
    """Layer 1 has a clean target-only causal switch; Layer 3 has a massive constant baseline along dim 0."""

    def __init__(self, vocab: int = 32, hidden: int = 32, inter: int = 64) -> None:
        super().__init__()
        from experiment_causal_diagnose import _PlantedBlock

        self.config = SimpleNamespace(model_type="llama", hidden_size=hidden, num_attention_heads=4)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_PlantedBlock(hidden, inter, 4) for _ in range(4)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        with torch.no_grad():
            for p in self.parameters():
                p.zero_()
            # Dim 0 drives target token 25; Dim 2 strongly drives healthy tokens 1..6 (logit +12 so softmax(25) ~ 0 on healthy)
            self.lm_head.weight[25, 0] = 0.5
            for v in range(1, 8):
                self.lm_head.weight[v, 2] = 12.0

            # Dim 4 is constant baseline present on EVERY token (healthy and target);
            # Dim 3 is active only on trigger token 25.
            self.model.embed_tokens.weight[:, 4] = 1.5
            for t in range(1, 8):
                self.model.embed_tokens.weight[t, 2] = 1.0
            self.model.embed_tokens.weight[25, 3] = 1.5

            # Layer 1 true causal switch: fires only on trigger (dim 3) -> writes +0.5 along dim 0 (zero on healthy)
            b1 = self.model.layers[1]
            b1.mlp.gate_proj.weight[5, 3] = 0.7
            b1.mlp.up_proj.weight[5, 3] = 0.7
            b1.mlp.down_proj.weight[0, 5] = 0.7

            # Layer 3 late baseline decoy: fires strongly on constant dim 4 across ALL tokens (healthy & target)
            # writing a huge constant baseline along dim 0, plus a tiny trigger ripple so paired_diff has +e0 sign.
            # Uncentered Y on Layer 3 is ~7x larger than Layer 1, so uncentered C_t ranks Layer 3 #1,
            # whereas healthy-centered (Y - Y_healthy_mean) cancels Layer 3's constant baseline and ranks Layer 1 #1!
            b3 = self.model.layers[3]
            b3.mlp.gate_proj.weight[0, 4] = 1.5
            b3.mlp.up_proj.weight[0, 4] = 1.5
            b3.mlp.gate_proj.weight[0, 3] = 0.05
            b3.mlp.up_proj.weight[0, 3] = 0.05
            b3.mlp.down_proj.weight[0, 0] = 1.5

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        x = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            x = blk(x)
        return SimpleNamespace(logits=self.lm_head(x))


def test_center_healthy_suppresses_high_baseline_late_layer_decoy() -> None:
    model = _HighBaselineLateDecoyLM().eval()
    seqs = [
        [1, 2, 3, 4, 25, 25, 25, 25],
        [2, 3, 4, 5, 25, 25, 25, 25],
        [3, 4, 5, 6, 25, 25, 25, 25],
        [4, 5, 6, 7, 25, 25, 25, 25],
    ]
    traces_old = [
        capture_causal_trace(
            model,
            s,
            target_span=(4, 8),
            healthy_span=(0, 3),
            grad_mode="compressed",
            center="none",
            healthy_cost="linear",
            top_k_mlp=16,
            group_size=16,
            hadamard_block=16,
            top_k_attn=0,
            capture_cache=False,
        )
        for s in seqs
    ]
    traces_new = [
        capture_causal_trace(
            model,
            s,
            target_span=(4, 8),
            healthy_span=(0, 3),
            grad_mode="compressed",
            center="healthy",
            healthy_cost="fisher",
            top_k_mlp=16,
            group_size=16,
            hadamard_block=16,
            top_k_attn=0,
            capture_cache=False,
        )
        for s in seqs
    ]

    # Uncentered + linear clamp is fooled by Layer 3's huge constant baseline Y
    diag_old = diagnose_causal_traces(traces_old, center="none", healthy_cost="linear")
    assert diag_old["summary"]["top_causal_site"] == "layers.3.mlp.down_proj"

    # Healthy-centered + Fisher eliminates Layer 3's constant baseline and ranks Layer 1 #1
    diag_new = diagnose_causal_traces(traces_new, center="healthy", healthy_cost="fisher", model_or_weights=model)
    assert diag_new["summary"]["top_causal_site"] == "layers.1.mlp.down_proj"
    assert diag_new["top_neurons"]["layers.1.mlp.down_proj"][0] == 5



