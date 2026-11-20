"""Pairwise (OBS) neuron selection in scripts/experiment_index_criteria.py: CPU, tiny random layers."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from experiment_index_criteria import obs_select, ridge_refit  # noqa: E402

RIDGE = 1e-2


def _layer(U=12, H=6, T=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(T, U, generator=g, dtype=torch.float64) @ torch.randn(U, U, generator=g, dtype=torch.float64) * 0.3
    W = torch.randn(H, U, generator=g, dtype=torch.float64)
    return h, W


def _ridge_loss(G, W, ix):
    """E||W h - W_S h_S||^2 + lam ||W_S||^2 with lam fixed from the full Gram (the objective obs_select minimizes)."""
    lam = RIDGE * G.diagonal().mean()
    Gss = G[ix][:, ix]
    Ws = torch.linalg.solve(Gss + lam * torch.eye(len(ix), dtype=G.dtype), (W @ G[:, ix]).T).T
    E = torch.trace(W @ G @ W.T) - 2 * torch.trace(Ws @ G[ix, :] @ W.T) + torch.trace(Ws @ Gss @ Ws.T)
    return float(E + lam * (Ws * Ws).sum())


def test_obs_first_removal_is_the_brute_force_best():
    h, W = _layer()
    G = h.T @ h / len(h)
    U = G.shape[0]
    pick = obs_select(G, W, [U - 1], RIDGE)[U - 1].tolist()
    dropped = (set(range(U)) - set(pick)).pop()
    losses = [_ridge_loss(G, W, torch.tensor([i for i in range(U) if i != j])) for j in range(U)]
    assert dropped == min(range(U), key=losses.__getitem__)


def test_obs_returns_every_budget_nested_and_sorted():
    h, W = _layer()
    G = h.T @ h / len(h)
    sel = obs_select(G, W, [9, 6, 3], RIDGE)
    assert [len(sel[k]) for k in (9, 6, 3)] == [9, 6, 3]
    for k in (9, 6, 3):
        assert sel[k].tolist() == sorted(sel[k].tolist())
    assert set(sel[3].tolist()) <= set(sel[6].tolist()) <= set(sel[9].tolist())


def test_obs_beats_per_neuron_energy_after_refit():
    # redundant pairs: neurons 2k and 2k+1 fire together, so keeping both wastes the budget
    g = torch.Generator().manual_seed(1)
    base = torch.randn(600, 6, generator=g, dtype=torch.float64) * torch.tensor([3, 2.5, 2, 1, .8, .6], dtype=torch.float64)
    h = torch.repeat_interleave(base, 2, dim=1) + 0.01 * torch.randn(600, 12, generator=g, dtype=torch.float64)
    W = torch.randn(5, 12, generator=g, dtype=torch.float64)
    G = h.T @ h / len(h)
    k = 6
    energy = (G.diagonal().sqrt() * W.norm(dim=0)).topk(k).indices.sort().values
    obs = obs_select(G, W, [k], RIDGE)[k]
    assert _ridge_loss(G, W, obs) < 0.5 * _ridge_loss(G, W, energy)
    assert len({int(i) // 2 for i in obs}) == 6  # one neuron from every pair


def test_obs_greedy_path_matches_brute_force_at_each_step():
    # the rank-1 updates must stay exact: every greedy step equals the best single removal from the current set
    h, W = _layer(seed=3)
    G = h.T @ h / len(h)
    U = G.shape[0]
    sel = obs_select(G, W, list(range(U - 1, 3, -1)), RIDGE)
    cur = list(range(U))
    for k in range(U - 1, 3, -1):
        best = min(cur, key=lambda j: _ridge_loss(G, W, torch.tensor([i for i in cur if i != j])))
        cur.remove(best)
        assert sel[k].tolist() == cur
    assert ridge_refit(G, W, sel[4], RIDGE).shape == (W.shape[0], 4)
