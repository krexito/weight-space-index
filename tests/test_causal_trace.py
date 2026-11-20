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
