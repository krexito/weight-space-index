from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import wsi
from wsi.fisher import (
    FisherProfile,
    allocate_fisher_global,
    allocate_fisher_uniform,
    find_mlp_down_projections,
    profile_column_fisher,
)


class _SwiGLUMock(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _LFM3Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.feed_forward = _SwiGLUMock(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.feed_forward(x)


class _ToySwiGLULM(nn.Module):
    def __init__(self, vocab_size: int = 19, d_model: int = 8, d_ff: int = 20, n_layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_LFM3Block(d_model, d_ff) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for blk in self.layers:
            x = blk(x)
        return self.lm_head(x)


class _LlamaMockBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
                self.up_proj = nn.Linear(d_model, d_ff, bias=False)
                self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        self.mlp = _MLP()


class _ToyLlamaMock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_LlamaMockBlock(8, 24) for _ in range(2)])


def test_exports_and_find_mlp_down_projections() -> None:
    for name in (
        "FisherProfile",
        "allocate_fisher_global",
        "allocate_fisher_uniform",
        "find_mlp_down_projections",
        "profile_column_fisher",
    ):
        assert hasattr(wsi, name), f"Missing wsi.{name}"

    lfm_toy = _ToySwiGLULM()
    lfm_downs = find_mlp_down_projections(lfm_toy)
    assert [nm for nm, _ in lfm_downs] == ["layers.0.feed_forward.w2", "layers.1.feed_forward.w2"]

    llama_mock = _ToyLlamaMock()
    llama_downs = find_mlp_down_projections(llama_mock)
    assert [nm for nm, _ in llama_downs] == ["layers.0.mlp.down_proj", "layers.1.mlp.down_proj"]


def _manual_fisher(
    model: nn.Module,
    mods: list[tuple[str, nn.Linear]],
    seqs: list[list[int]],
    delta_w: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    ref = {nm: torch.zeros(mod.in_features, dtype=torch.float64) for nm, mod in mods}
    for p in model.parameters():
        p.requires_grad_(True)
    for seq in seqs:
        inp = torch.tensor([seq], dtype=torch.long)
        xs: dict[str, torch.Tensor] = {}
        ys: dict[str, torch.Tensor] = {}
        hs = []
        for nm, mod in mods:
            def _hk(_m, i, o, nm=nm):
                xs[nm] = i[0].detach().reshape(-1, mod.in_features).float()
                o.retain_grad()
                ys[nm] = o
            hs.append(mod.register_forward_hook(_hk))
        try:
            logits = model(inp)[0, :-1].float()
            loss = F.nll_loss(F.log_softmax(logits, dim=-1), inp[0, 1:], reduction="mean")
            model.zero_grad(set_to_none=True)
            loss.backward()
            for nm, mod in mods:
                g = ys[nm].grad.detach().reshape(-1, mod.out_features).float()
                dw = (delta_w[nm] if delta_w is not None else mod.weight).detach().float()
                ref[nm] += (xs[nm] * (g @ dw)).pow(2).sum(dim=0).double()
        finally:
            for h in hs:
                h.remove()
            model.zero_grad(set_to_none=True)
    return ref


def test_profile_column_fisher_exact_and_state_restore() -> None:
    torch.manual_seed(7)
    model = _ToySwiGLULM()
    mods = find_mlp_down_projections(model)
    seqs = [[1, 4, 7, 2, 9, 5], [3, 8, 6, 11, 14, 1]]

    # 1. Exact numerical match (pruning mode: delta_w=None)
    expected = _manual_fisher(model, mods, seqs, delta_w=None)

    # Simulate caller disabling grad globally (as experiment scripts do) and starting in train() mode
    for p in model.parameters():
        p.requires_grad_(False)
    model.train(True)
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        prof = profile_column_fisher(model, mods, seqs, delta_w=None, device="cpu")
        assert not torch.is_grad_enabled()
        assert model.training is True
        assert all(not p.requires_grad for p in model.parameters())
    finally:
        torch.set_grad_enabled(prev_grad)

    assert prof.n_seqs == 2
    assert prof.n_tokens == 12
    for nm, _ in mods:
        assert torch.allclose(prof.scores[nm], expected[nm], rtol=1e-5, atol=1e-6)
        assert (prof.act_sq[nm] >= 0).all()

    # 2. Exact numerical match with custom delta_w (quantization perturbation mode)
    dw = {nm: torch.randn_like(mod.weight) * 0.1 for nm, mod in mods}
    expected_dw = _manual_fisher(model, mods, seqs, delta_w=dw)
    prof_dw = profile_column_fisher(model, mods, seqs, delta_w=dw, device="cpu")
    for nm, _ in mods:
        assert torch.allclose(prof_dw.scores[nm], expected_dw[nm], rtol=1e-5, atol=1e-6)

    # 3. max_tokens cap
    many_seqs = [[1, 2, 3, 4, 5, 6] for _ in range(5)]
    prof_cap = profile_column_fisher(model, mods, many_seqs, max_tokens=10, device="cpu")
    assert prof_cap.n_seqs == 2
    assert prof_cap.n_tokens == 12


def test_allocate_fisher_global_vs_uniform() -> None:
    # Layer 0 has 10x larger Fisher scores than Layer 1 -> global must shift capacity to Layer 0
    s0 = torch.linspace(100.0, 10.0, 16, dtype=torch.float64)
    s1 = torch.linspace(10.0, 1.0, 16, dtype=torch.float64)
    prof = FisherProfile(
        scores={"l0": s0, "l1": s1},
        act_sq={"l0": torch.ones(16, dtype=torch.float64), "l1": torch.ones(16, dtype=torch.float64)},
        in_features={"l0": 16, "l1": 16},
        out_features={"l0": 8, "l1": 8},
        n_tokens=64,
        n_seqs=4,
    )

    unif = allocate_fisher_uniform(prof, 0.5, min_per_module=2, align_to=1)
    glob = allocate_fisher_global(prof, 0.5, min_per_module=2, align_to=1)

    assert len(unif["l0"]) == 8 and len(unif["l1"]) == 8
    assert len(glob["l0"]) > len(unif["l0"])
    assert len(glob["l1"]) < len(unif["l1"])
    assert len(glob["l1"]) >= 2

    unif_params = sum(len(unif[k]) * prof.out_features[k] for k in unif)
    glob_params = sum(len(glob[k]) * prof.out_features[k] for k in glob)
    assert glob_params <= unif_params
    assert glob_params == unif_params

    # Alignment constraint (align_to=4, min_per_module=4)
    glob_a4 = allocate_fisher_global(prof, 0.5, min_per_module=4, align_to=4)
    assert len(glob_a4["l0"]) % 4 == 0 and len(glob_a4["l1"]) % 4 == 0
    assert len(glob_a4["l0"]) >= 4 and len(glob_a4["l1"]) >= 4
    assert sum(len(v) for v in glob_a4.values()) <= 16
