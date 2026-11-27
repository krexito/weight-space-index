"""Strata-inspired S_K quantization codecs, Block-Hadamard, GPTQ+64-flip, knapsack & Scale-QAT.

Implements self-contained CPU/GPU quantization primitives adapted for WSI research:
  1. Wire-exact S_K super-block byte accounting (`QK_S=256`, `SUB_G=64`: `S2_K..S8_K`).
  2. Zero-cost 64-wide Block-Hadamard transform (`H64`) on weights and Hessians.
  3. MSE / Hessian-weighted optimal sub-group clipping scales (`+3.5 dB` at 2-bit).
  4. In-HBM Block-LDL (GPTQ) + analytical 64-flip coordinate refinement.
  5. Global Hessian marginal-gain-per-byte knapsack allocator with X-Ray protection floors.
  6. X-Ray quantization sensitivity predictor (`build_quant_sensitivity_map`).
  7. Top-256 + tail-bucket KL distillation over 1D norm scales (`run_norm_scale_qat`).
"""
from __future__ import annotations

import copy
import heapq
import math
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QK_S = 256
SUB_G = 64
NUM_SUB = 4

CODE_IDS: dict[str, int] = {
    "S2_K": 2,
    "S3_K": 3,
    "S4_K": 4,
    "S5_K": 5,
    "S6_K": 6,
    "S8_K": 8,
}

SUPERBLOCK_BYTES: dict[str, int] = {
    "S2_K": 70,   # 6 B hdr + 64 B (2.1875 b/w)
    "S3_K": 102,  # 6 B hdr + 96 B (3.1875 b/w)
    "S4_K": 134,  # 6 B hdr + 128 B (4.1875 b/w)
    "S5_K": 166,  # 6 B hdr + 160 B (5.1875 b/w)
    "S6_K": 198,  # 6 B hdr + 192 B (6.1875 b/w)
    "S8_K": 262,  # 6 B hdr + 256 B (8.1875 b/w)
}

_CLIP_GRIDS: dict[int, tuple[float, ...]] = {
    2: (0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 1.00),
    3: (0.64, 0.72, 0.80, 0.88, 0.94, 1.00),
    4: (0.78, 0.86, 0.93, 1.00),
    5: (0.90, 0.96, 1.00),
    6: (0.92, 1.00),
}

_H64_CACHE: dict[str, torch.Tensor] = {}


def code_bits(code_name: str) -> int:
    if code_name == "S8_K":
        return 8
    if code_name in SUPERBLOCK_BYTES and code_name.startswith("S") and code_name.endswith("_K"):
        return int(code_name[1])
    raise ValueError(f"Unsupported S_K code: {code_name}")


def code_bits_per_weight(code_name: str) -> float:
    return (SUPERBLOCK_BYTES[code_name] * 8.0) / float(QK_S)


def matrix_wire_bytes(rows: int, cols: int, code_name: str) -> int:
    """Exact .strata S_K wire bytes for a `[rows, cols]` matrix (cols padded up to multiple of 256)."""
    sb_per_row = (int(cols) + QK_S - 1) // QK_S
    return int(rows) * sb_per_row * SUPERBLOCK_BYTES[code_name]


def get_h64(device: torch.device | str) -> torch.Tensor:
    """Return cached normalized 64x64 Walsh-Hadamard matrix (`H64 @ H64.T = I`)."""
    key = str(device)
    if key not in _H64_CACHE:
        h = torch.tensor([[1.0]], dtype=torch.float32, device=device)
        for _ in range(6):
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        _H64_CACHE[key] = h / 8.0
    return _H64_CACHE[key]


def apply_h64_cols(W: torch.Tensor) -> torch.Tensor:
    """Right-multiply W `[..., K]` by block-diagonal `H64` (`K % 64 == 0`)."""
    orig = W.shape
    K = orig[-1]
    assert K % SUB_G == 0, f"Expected last dim multiple of 64, got {K}"
    H64 = get_h64(W.device)
    return (W.to(torch.float32).reshape(-1, SUB_G) @ H64).reshape(orig)


def apply_h64_sym(H: torch.Tensor) -> torch.Tensor:
    """Compute `H64^T H H64` for square Hessian `H [K, K]` (`K % 64 == 0`)."""
    K = H.shape[0]
    assert H.ndim == 2 and H.shape[1] == K and K % SUB_G == 0
    H64 = get_h64(H.device)
    nb = K // SUB_G
    t = H.to(torch.float32).reshape(nb, SUB_G, nb, SUB_G)
    t = torch.einsum("ai,piqj->paqj", H64, t)
    t = torch.einsum("paqj,jb->paqb", t, H64)
    return t.reshape(K, K)


def _pad_weight_and_hessian(
    W: torch.Tensor, H: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Pad columns to next multiple of QK_S=256 (and H with tiny positive diag on padded cols)."""
    K = int(W.shape[1])
    K_pad = ((K + QK_S - 1) // QK_S) * QK_S
    if K_pad == K:
        return W.to(torch.float32), (H.to(torch.float32) if H is not None else None), K
    W_pad = torch.zeros((W.shape[0], K_pad), device=W.device, dtype=torch.float32)
    W_pad[:, :K] = W.to(torch.float32)
    if H is None:
        return W_pad, None, K
    H_f = H.to(torch.float32)
    diag = torch.diag(H_f)
    pos = diag[diag > 1e-12]
    fill_d = float(pos.mean().item()) if pos.numel() > 0 else 1.0
    H_pad = torch.zeros((K_pad, K_pad), device=W.device, dtype=torch.float32)
    H_pad[:K, :K] = H_f
    idx = torch.arange(K, K_pad, device=W.device)
    H_pad[idx, idx] = fill_d
    return W_pad, H_pad, K


def damped_cholesky_inv(H: torch.Tensor, damp_frac: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute damped Hessian `H_d` and upper Cholesky factor `chol(H_d^{-1})^T` in float32."""
    K = int(H.shape[0])
    Hd = 0.5 * (H.to(torch.float32) + H.to(torch.float32).T)
    diag = torch.diag(Hd)
    dead = diag <= 1e-12
    idx = torch.arange(K, device=H.device)
    if dead.any():
        Hd[dead, dead] = 1.0
        diag = torch.diag(Hd)
    active = diag[~dead]
    mean_d = float(active.mean().item()) if active.numel() > 0 else 1.0
    damp = max(mean_d * float(damp_frac), 1e-5)
    Hd[idx, idx] += damp

    for mult in (1.0, 4.0, 16.0, 64.0):
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            Hinv_U = torch.linalg.cholesky(0.5 * (Hinv + Hinv.T), upper=True)
            return Hd, Hinv_U
        except RuntimeError:
            Hd[idx, idx] += damp * mult

    inv_diag = torch.rsqrt(torch.diag(Hd).clamp(min=1e-5))
    return Hd, torch.diag(inv_diag)


def optimal_subscales_torch(
    W_sb: torch.Tensor,
    bits: int,
    h_diag_sb: torch.Tensor | None = None,
    use_clipping: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute FP16 super-scale `D`, `uint8` sub-scales `u` `[N_sb, 4]`, and effective scales `[N_sb, 4]`."""
    is_s8 = bits == 8
    levels = 1 << bits
    half = 127.0 if is_s8 else (levels - 1) / 2.0
    amax_sub = W_sb.abs().amax(dim=2)  # [N_sb, 4]

    if use_clipping and not is_s8:
        grid = _CLIP_GRIDS.get(bits, (0.92, 1.00))
        best_clip = amax_sub.clone()
        best_err = torch.full_like(amax_sub, float("inf"))
        for r in grid:
            cand = amax_sub * float(r)
            s = torch.where(cand > 0.0, cand / half, torch.ones_like(cand)).unsqueeze(-1)
            inv = torch.where(cand > 0.0, half / cand, torch.zeros_like(cand)).unsqueeze(-1)
            q = torch.clamp(torch.floor(W_sb * inv + half + 0.5), 0, levels - 1) - half
            diff2 = (W_sb - q * s).pow(2)
            if h_diag_sb is not None:
                diff2 = diff2 * h_diag_sb
            err = diff2.mean(dim=2)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_clip = torch.where(better, cand, best_clip)
        amax_sub = best_clip

    amax_sb = amax_sub.amax(dim=1)  # [N_sb]
    D_f16 = (amax_sb / half).to(torch.float16)
    D_eff = D_f16.to(torch.float32)
    denom = torch.where(amax_sb > 0.0, amax_sb, torch.ones_like(amax_sb)).unsqueeze(-1)
    u = torch.clamp(torch.round(amax_sub / denom * 255.0), 0, 255).to(torch.uint8)
    u = torch.where((amax_sub > 0.0) & (u == 0), torch.ones_like(u), u)
    eff = D_eff.unsqueeze(-1) * (u.to(torch.float32) / 255.0)
    return D_f16, u, eff


@torch.no_grad()
def rtn_quantize_sk_torch(
    W_in: torch.Tensor,
    code_name: str,
    *,
    use_clipping: bool = False,
    use_h64: bool = False,
    H: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, float | None]:
    """Quantize `[M, K]` via naive/clipped RTN in S_K layout; returns `(W_deq, wire_bytes, err_h)`."""
    M, K_orig = int(W_in.shape[0]), int(W_in.shape[1])
    W_pad, H_pad, _ = _pad_weight_and_hessian(W_in, H)
    K = int(W_pad.shape[1])
    wire_bytes = matrix_wire_bytes(M, K_orig, code_name)
    bits = code_bits(code_name)
    is_s8 = bits == 8
    levels = 1 << bits
    half = 127.0 if is_s8 else (levels - 1) / 2.0

    W_work = apply_h64_cols(W_pad) if use_h64 else W_pad
    H_work = apply_h64_sym(H_pad) if (use_h64 and H_pad is not None) else H_pad

    n_sb_row = K // QK_S
    W_sb = W_work.reshape(M * n_sb_row, NUM_SUB, SUB_G)
    h_sb = None
    if H_work is not None:
        h_diag = torch.diag(H_work).clamp(min=1e-12)
        h_sb = h_diag.reshape(n_sb_row, NUM_SUB, SUB_G).unsqueeze(0).expand(M, -1, -1, -1).reshape(M * n_sb_row, NUM_SUB, SUB_G)

    _, _, eff = optimal_subscales_torch(W_sb, bits, h_diag_sb=h_sb, use_clipping=use_clipping)
    S_col = eff.reshape(M, n_sb_row, NUM_SUB, 1).expand(M, n_sb_row, NUM_SUB, SUB_G).reshape(M, K)
    inv_S = torch.where(S_col > 0.0, 1.0 / S_col, torch.zeros_like(S_col))

    if is_s8:
        q = torch.clamp(torch.round(W_work * inv_S), -127.0, 127.0)
        Q_deq = q * S_col
    else:
        q = torch.clamp(torch.floor(W_work * inv_S + half + 0.5), 0.0, float(levels - 1))
        Q_deq = (q - half) * S_col

    err_h: float | None = None
    if H_work is not None:
        diff = Q_deq - W_work
        err_h = float(((diff @ H_work) * diff).sum().item())

    if use_h64:
        Q_deq = apply_h64_cols(Q_deq)
    return Q_deq[:, :K_orig].to(W_in.dtype), wire_bytes, err_h


@torch.no_grad()
def gptq_flip_quantize_sk_torch(
    W_in: torch.Tensor,
    H: torch.Tensor,
    code_name: str,
    *,
    use_h64: bool = True,
    max_flips: int = 64,
    block_size: int = 128,
    precomputed_H: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, int, float]:
    """In-HBM Block-LDL (GPTQ) + MACaQ analytical 64-flip refinement for S_K.

    If `precomputed_H = (Hd, Hinv_U)` is passed, it is assumed to ALREADY be in the rotated space
    (when `use_h64=True`) and padded to `K_pad`, enabling zero-cost Cholesky reuse across bit widths.
    Returns `(W_deq_orig_dtype, wire_bytes, err_h)`.
    """
    M, K_orig = int(W_in.shape[0]), int(W_in.shape[1])
    W_pad, H_pad, _ = _pad_weight_and_hessian(W_in, H if precomputed_H is None else None)
    K = int(W_pad.shape[1])
    wire_bytes = matrix_wire_bytes(M, K_orig, code_name)
    bits = code_bits(code_name)
    is_s8 = bits == 8
    levels = 1 << bits
    half = 127.0 if is_s8 else (levels - 1) / 2.0
    n_sb_row = K // QK_S

    W_ref = apply_h64_cols(W_pad) if use_h64 else W_pad
    if precomputed_H is not None:
        Hd, Hinv_U = precomputed_H
    else:
        assert H_pad is not None
        H_rot = apply_h64_sym(H_pad) if use_h64 else H_pad
        Hd, Hinv_U = damped_cholesky_inv(H_rot, damp_frac=0.01)

    H_diag = torch.diag(Hd).clamp(min=1e-8)
    W_sb = W_ref.reshape(M * n_sb_row, NUM_SUB, SUB_G)
    h_sb = H_diag.reshape(n_sb_row, NUM_SUB, SUB_G).unsqueeze(0).expand(M, -1, -1, -1).reshape(M * n_sb_row, NUM_SUB, SUB_G)
    _, _, eff = optimal_subscales_torch(W_sb, bits, h_diag_sb=h_sb, use_clipping=True)

    S_col = eff.reshape(M, n_sb_row, NUM_SUB, 1).expand(M, n_sb_row, NUM_SUB, SUB_G).reshape(M, K)
    inv_S = torch.where(S_col > 0.0, 1.0 / S_col, torch.zeros_like(S_col))

    # Block-LDL (GPTQ) error propagation
    Q_int = torch.zeros((M, K), device=W_ref.device, dtype=torch.int16)
    W_work = W_ref.clone()
    for i1 in range(0, K, block_size):
        i2 = min(i1 + block_size, K)
        cnt = i2 - i1
        W1 = W_work[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv_U[i1:i2, i1:i2]
        for j in range(cnt):
            col = i1 + j
            wj = W1[:, j]
            sc = S_col[:, col]
            inv_c = inv_S[:, col]
            if is_s8:
                qi = torch.clamp(torch.round(wj * inv_c), -127.0, 127.0)
                qj = qi * sc
            else:
                qi = torch.clamp(torch.floor(wj * inv_c + half + 0.5), 0.0, float(levels - 1))
                qj = (qi - half) * sc
            Q_int[:, col] = qi.to(torch.int16)
            err = (wj - qj) / Hinv1[j, j]
            Err1[:, j] = err
            if j + 1 < cnt:
                W1[:, j + 1 :] -= err.unsqueeze(1) * Hinv1[j, j + 1 :].unsqueeze(0)
        if i2 < K:
            W_work[:, i2:] -= Err1 @ Hinv_U[i1:i2, i2:]

    # MACaQ Analytical Hessian Flip Refinement
    if max_flips > 0 and not is_s8:
        Q_deq = (Q_int.to(torch.float32) - half) * S_col
        g = (Q_deq - W_ref) @ Hd  # [M, K]
        step2_Hjj = (S_col * S_col) * H_diag.unsqueeze(0)
        row_idx = torch.arange(M, device=W_ref.device)
        for _ in range(int(max_flips)):
            can_up = Q_int < (levels - 1)
            can_dn = Q_int > 0
            two_sg = 2.0 * S_col * g
            dL_up = torch.where(can_up, step2_Hjj + two_sg, torch.zeros_like(g))
            dL_dn = torch.where(can_dn, step2_Hjj - two_sg, torch.zeros_like(g))
            min_dL, best_j = torch.minimum(dL_up, dL_dn).min(dim=1)
            improve = min_dL < -1e-9
            if not improve.any():
                break
            active_rows = row_idx[improve]
            active_j = best_j[improve]
            go_up = dL_up[active_rows, active_j] <= dL_dn[active_rows, active_j]
            dq = torch.where(go_up, torch.ones_like(active_j, dtype=torch.int16), -torch.ones_like(active_j, dtype=torch.int16))
            Q_int[active_rows, active_j] += dq
            delta_w = dq.to(torch.float32) * S_col[active_rows, active_j]
            g[active_rows] += delta_w.unsqueeze(1) * Hd[active_j]
        Q_deq = (Q_int.to(torch.float32) - half) * S_col
    else:
        Q_deq = Q_int.to(torch.float32) * S_col if is_s8 else (Q_int.to(torch.float32) - half) * S_col

    diff = Q_deq - W_ref
    err_h = float(((diff @ Hd) * diff).sum().item())
    W_out = apply_h64_cols(Q_deq) if use_h64 else Q_deq
    return W_out[:, :K_orig].to(W_in.dtype), wire_bytes, err_h


# ------------------------------------------------------------------ Global Hessian Knapsack Allocator
def solve_knapsack_bit_allocation(
    items: Sequence[dict[str, Any]],
    budget_bytes: int,
) -> tuple[dict[Any, str], int, float]:
    """Allocate per-item quantization codes within `budget_bytes` maximizing weighted error reduction/byte.

    Each item is `{"id": key, "options": [(label, bytes, weighted_err), ...]}` where options[0] is the
    mandatory minimum floor (and options are sorted by ascending bytes).
    Uses lower-convex-hull slopes + max-heap greedy upgrades for O(N log N) near-exact optimality.
    """
    chosen_idx: list[int] = []
    hulls: list[list[int]] = []
    total_bytes = 0
    total_err = 0.0

    # Build strictly improving lower convex hull on (-bytes vs err) per item
    heap: list[tuple[float, int, int, int]] = []  # (-slope, item_i, hull_pos_cur, opt_idx_next)

    for i, it in enumerate(items):
        opts = sorted(it["options"], key=lambda o: (int(o[1]), float(o[2])))
        assert len(opts) >= 1
        # Filter strictly decreasing error steps with strictly increasing bytes
        clean = [0]
        for j in range(1, len(opts)):
            if opts[j][1] > opts[clean[-1]][1] and opts[j][2] < opts[clean[-1]][2]:
                clean.append(j)
        # Keep lower convex hull so marginal gain (dErr / dBytes) decreases monotonically
        hull: list[int] = []
        for idx in clean:
            while len(hull) >= 2:
                i0, i1, i2 = hull[-2], hull[-1], idx
                s1 = (opts[i0][2] - opts[i1][2]) / max(1, opts[i1][1] - opts[i0][1])
                s2 = (opts[i1][2] - opts[i2][2]) / max(1, opts[i2][1] - opts[i1][1])
                if s2 >= s1:
                    hull.pop()
                else:
                    break
            hull.append(idx)

        it["_sorted_opts"] = opts
        hulls.append(hull)
        chosen_idx.append(hull[0])
        total_bytes += int(opts[hull[0]][1])
        total_err += float(opts[hull[0]][2])

        if len(hull) > 1:
            o0, o1 = opts[hull[0]], opts[hull[1]]
            slope = (float(o0[2]) - float(o1[2])) / float(max(1, int(o1[1]) - int(o0[1])))
            heapq.heappush(heap, (-slope, i, 0, 1))

    if total_bytes > int(budget_bytes):
        raise ValueError(f"Minimum floor bytes {total_bytes} exceed budget {budget_bytes}")

    # Phase 1: Convex-hull greedy upgrades
    hull_pos = [0] * len(items)
    while heap:
        neg_s, i, hp_cur, hp_nxt = heapq.heappop(heap)
        if hp_cur != hull_pos[i]:
            continue
        opts = items[i]["_sorted_opts"]
        hull = hulls[i]
        idx_cur = hull[hp_cur]
        idx_nxt = hull[hp_nxt]
        db = int(opts[idx_nxt][1]) - int(opts[idx_cur][1])
        if total_bytes + db <= int(budget_bytes):
            de = float(opts[idx_cur][2]) - float(opts[idx_nxt][2])
            total_bytes += db
            total_err -= de
            chosen_idx[i] = idx_nxt
            hull_pos[i] = hp_nxt
            if hp_nxt + 1 < len(hull):
                o0, o1 = opts[hull[hp_nxt]], opts[hull[hp_nxt + 1]]
                slope = (float(o0[2]) - float(o1[2])) / float(max(1, int(o1[1]) - int(o0[1])))
                heapq.heappush(heap, (-slope, i, hp_nxt, hp_nxt + 1))

    # Phase 2: Small-slack greedy cleanup over ALL remaining feasible upgrades (catches small budget leftovers)
    while True:
        slack = int(budget_bytes) - total_bytes
        if slack <= 0:
            break
        best = None
        for i, it in enumerate(items):
            opts = it["_sorted_opts"]
            cur = chosen_idx[i]
            cur_b, cur_e = int(opts[cur][1]), float(opts[cur][2])
            for nxt in range(cur + 1, len(opts)):
                db = int(opts[nxt][1]) - cur_b
                if 0 < db <= slack and opts[nxt][2] < cur_e:
                    de = cur_e - float(opts[nxt][2])
                    slope = de / float(db)
                    if best is None or slope > best[0]:
                        best = (slope, i, nxt, db, de)
        if best is None:
            break
        _, i, nxt, db, de = best
        chosen_idx[i] = nxt
        total_bytes += db
        total_err -= de

    sel: dict[Any, str] = {}
    for i, it in enumerate(items):
        opts = it.pop("_sorted_opts")
        sel[it["id"]] = str(opts[chosen_idx[i]][0])
    return sel, int(total_bytes), float(total_err)


# ------------------------------------------------------------------ X-Ray Quantization Sensitivity Predictor
def _interp_8bins(curve_8: Sequence[float | None] | None, n_layers: int) -> np.ndarray:
    """Linearly interpolate an 8-bin WSI depth curve across `n_layers` physical layers."""
    if not curve_8:
        return np.zeros(n_layers, dtype=np.float64)
    pts = [(i / max(1, len(curve_8) - 1), float(v)) for i, v in enumerate(curve_8) if v is not None and math.isfinite(float(v))]
    if not pts:
        return np.zeros(n_layers, dtype=np.float64)
    xs_src = np.array([p[0] for p in pts], dtype=np.float64)
    ys_src = np.array([p[1] for p in pts], dtype=np.float64)
    xs_dst = np.linspace(0.0, 1.0, max(1, n_layers))
    return np.interp(xs_dst, xs_src, ys_src)


def _norm_pos(arr: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(arr, dtype=np.float64), 0.0)
    mx = float(a.max()) if a.size else 0.0
    return a / mx if mx > 1e-12 else np.zeros_like(a)


def build_quant_sensitivity_map(card: dict[str, Any]) -> dict[str, Any]:
    """Predict per-layer quantization sensitivity & protection zones directly from an `wsi_xray_v1` chart.

    Zero GPU cost. Explicitly prevents the Global-Fisher late-layer starvation bug by combining:
      1. Massive activation spikes (`massive_act_peak_8` + `massive_act_max.layer`).
      2. Factual readout sharpness (`dla_curve_8` + `dla_peak`).
      3. Re-balanced Fisher signal (`mlp_fisher_share_8` or `block_fisher_share_8`) with a late-depth anchor.
      4. Loop multiplicity (`looped.max_visits_per_layer` & loop exit boundary) when `identity.looped` is present.
    """
    ident = card.get("identity", {})
    anat = card.get("anatomy", {})
    fp = anat.get("forward_profile", {})
    fi = anat.get("fisher_profile", {})
    fr = anat.get("fact_readout", {})

    n_layers = int(ident.get("n_layers") or 1)
    looped = ident.get("looped") or {}
    loop_layers = set(int(x) for x in looped.get("looped_layers", []))
    visits = np.ones(n_layers, dtype=np.float64)
    if loop_layers and looped.get("effective_trace"):
        counts: dict[int, int] = {}
        for l_idx in looped["effective_trace"]:
            counts[int(l_idx)] = counts.get(int(l_idx), 0) + 1
        for l_idx, c in counts.items():
            if 0 <= l_idx < n_layers:
                visits[l_idx] = float(c)
    elif loop_layers:
        mv = float(looped.get("max_visits_per_layer", 2.0))
        for l_idx in loop_layers:
            if 0 <= l_idx < n_layers:
                visits[l_idx] = mv

    # Resample depth curves into physical layer space
    mass_curve = _norm_pos(_interp_8bins(fp.get("massive_act_peak_8"), n_layers))
    dla_curve = _norm_pos(_interp_8bins(fr.get("dla_curve_8"), n_layers))

    mlp_f = fi.get("mlp_fisher_share_8")
    blk_f = fi.get("block_fisher_share_8")
    raw_f = mlp_f if (isinstance(mlp_f, list) and any(v is not None for v in mlp_f)) else blk_f
    fisher_curve = _norm_pos(_interp_8bins(raw_f, n_layers))
    # Prevent early Fisher gradient accumulation from starving late output-forming layers
    depth = np.linspace(0.0, 1.0, n_layers)
    fisher_rebalanced = _norm_pos(fisher_curve * (0.4 + 0.6 * depth) + 0.15 * depth)

    # Spike & boundary indicators
    prot_set: set[int] = {0, max(0, n_layers - 1)}
    mass_max = fp.get("massive_act_max") or {}
    if isinstance(mass_max.get("layer"), int) and 0 <= int(mass_max["layer"]) < n_layers:
        prot_set.add(int(mass_max["layer"]))
    dla_pk = fr.get("dla_peak") or {}
    dla_phys = dla_pk.get("physical_layer", dla_pk.get("layer"))
    if isinstance(dla_phys, int) and 0 <= int(dla_phys) < n_layers:
        prot_set.add(int(dla_phys))

    spike = np.zeros(n_layers, dtype=np.float64)
    for l_p in prot_set:
        spike[l_p] = 1.0
    if loop_layers:
        max_loop = max(loop_layers)
        if 0 <= max_loop < n_layers:
            spike[max_loop] = max(spike[max_loop], 0.85)

    # Combined normalized score across layers (scaled cleanly to [0, 1])
    base_score = 0.25 * mass_curve + 0.25 * dla_curve + 0.25 * fisher_rebalanced + 0.25 * spike
    visit_weight = np.sqrt(visits)  # x1 for single-pass, x1.73 for 3x loop
    raw_score = base_score * visit_weight
    scores = _norm_pos(raw_score)
    # Keep prologue & final tail floor >= 0.45 so knapsack never starves boundary layers
    if n_layers >= 1:
        scores[0] = max(float(scores[0]), 0.55)
        scores[-1] = max(float(scores[-1]), 0.65)
    for l_p in prot_set:
        scores[l_p] = max(float(scores[l_p]), 0.70)

    # Resample back into 8-bin summary matching WSI X-Ray schema
    bin_edges = np.array_split(np.arange(n_layers), 8)
    scores_8 = [round(float(scores[idxs].mean()), 3) if len(idxs) else 0.0 for idxs in bin_edges]

    low_win = (fi.get("lowest_fisher_window") or {}).get("layers") or []
    valley = [
        int(l_idx)
        for l_idx in low_win
        if isinstance(l_idx, int) and 0 <= int(l_idx) < n_layers and int(l_idx) not in prot_set and int(l_idx) not in loop_layers
    ]
    is_moe = ident.get("moe") is not None
    return {
        "layer_scores_8": scores_8,
        "per_layer_scores": [round(float(x), 4) for x in scores.tolist()],
        "protected_layers": sorted(prot_set),
        "compressible_valley": valley,
        "recommended_policy": {
            "control_dtype": "F32",
            "min_floor_protected": "S3_K+S2_K" if is_moe else "S3_K",
            "asymmetric_moe_pair": "S3_K+S2_K" if is_moe else None,
        },
    }


# ------------------------------------------------------------------ Top-256 KL & Norm Scale-QAT
def topk_tail_kl_loss(
    cand_logits: torch.Tensor,
    top_idx: torch.Tensor,
    top_logp: torch.Tensor,
    tail_p: torch.Tensor,
) -> torch.Tensor:
    """Exact top-K + tail-bucket KL(p_teacher || q_cand) loss (Strata `scale_qat_train.py`)."""
    q_logp = F.log_softmax(cand_logits.to(torch.float32), dim=-1)
    q_top_lp = q_logp.gather(-1, top_idx.to(torch.long))
    p_top_lp = top_logp.to(torch.float32)
    p_top = torch.exp(p_top_lp)
    kl_top = (p_top * (p_top_lp - q_top_lp)).sum(dim=-1)

    q_top_sum = torch.exp(q_top_lp).sum(dim=-1).clamp(max=1.0 - 1e-7)
    q_tail_lp = torch.log((1.0 - q_top_sum).clamp(min=1e-7))
    p_t = tail_p.to(torch.float32).clamp(min=1e-7)
    kl_tail = p_t * (torch.log(p_t) - q_tail_lp)
    return (kl_top + kl_tail).mean()


@torch.no_grad()
def capture_teacher_topk_targets(
    model: nn.Module,
    seqs: Sequence[Sequence[int]],
    device: torch.device | str,
    *,
    topk: int = 256,
) -> list[dict[str, torch.Tensor]]:
    """Capture teacher top-K log-probs + tail probability on CPU for Scale-QAT & evaluation."""
    recs: list[dict[str, torch.Tensor]] = []
    for s in seqs:
        ids = torch.tensor(list(s), dtype=torch.long, device=device).unsqueeze(0)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0].to(torch.float32)  # [T, V]
        logp = F.log_softmax(logits, dim=-1)
        k_eff = min(int(topk), int(logp.shape[-1]))
        top_lp, top_idx = torch.topk(logp, k=k_eff, dim=-1)
        top_p_sum = torch.exp(top_lp).sum(dim=-1).clamp(max=1.0 - 1e-7)
        tail_p = (1.0 - top_p_sum).clamp(min=1e-7)
        recs.append(
            {
                "tokens": torch.tensor(list(s), dtype=torch.int32),
                "top_idx": top_idx.to(torch.int32).cpu(),
                "top_logp": top_lp.to(torch.float16).cpu(),
                "tail_p": tail_p.to(torch.float16).cpu(),
            }
        )
        del ids, out, logits, logp
    return recs


@torch.no_grad()
def eval_topk_kl_and_top1(
    model: nn.Module,
    teacher_recs: Sequence[dict[str, torch.Tensor]],
    device: torch.device | str,
) -> dict[str, float]:
    """Measure mean top-K+tail KL divergence and exact top-1 agreement vs teacher."""
    kls: list[float] = []
    agree = 0
    total = 0
    for rec in teacher_recs:
        ids = rec["tokens"].to(device=device, dtype=torch.long).unsqueeze(0)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0].to(torch.float32)
        kl = topk_tail_kl_loss(
            logits,
            rec["top_idx"].to(device),
            rec["top_logp"].to(device),
            rec["tail_p"].to(device),
        )
        kls.append(float(kl.item()))
        pred_top1 = logits.argmax(dim=-1).cpu()
        ref_top1 = rec["top_idx"][:, 0].to(torch.long)
        agree += int((pred_top1 == ref_top1).sum().item())
        total += int(ref_top1.numel())
        del ids, out, logits
    return {
        "kl_mean": float(np.mean(kls)) if kls else 0.0,
        "top1_agreement": float(agree / max(1, total)),
        "n_tokens": int(total),
    }


class _NormScaleAdapter(nn.Module):
    """Learnable FP32 multiplicative scale gamma applied after an arbitrary 1D norm module."""

    def __init__(self, inner: nn.Module, dim: int, device: torch.device) -> None:
        super().__init__()
        self.inner = inner
        self.gamma = nn.Parameter(torch.ones((int(dim),), device=device, dtype=torch.float32))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.inner(*args, **kwargs)
        return out * self.gamma.to(dtype=out.dtype, device=out.device)


def run_norm_scale_qat(
    model: nn.Module,
    train_records: Sequence[dict[str, torch.Tensor]],
    dev_records: Sequence[dict[str, torch.Tensor]],
    *,
    device: torch.device | str,
    epochs: int = 2,
    lr: float = 5e-4,
    anchor_lambda: float = 1e-3,
    grad_accum: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Optimize FP32 multiplicative scales on all 1D norm modules via Top-K KL and fold back with 0 extra bytes."""
    dev = torch.device(device)
    orig_req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in orig_req:
        p.requires_grad_(False)

    was_training = bool(model.training)
    was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
    enabled_gc = False
    if hasattr(model, "gradient_checkpointing_enable") and not was_gc:
        try:
            model.gradient_checkpointing_enable()
            enabled_gc = True
        except Exception:
            enabled_gc = False

    wrapped: list[tuple[nn.Module, str, nn.Module, _NormScaleAdapter]] = []
    rms_cls = getattr(nn, "RMSNorm", None)
    for full_name, mod in list(model.named_modules()):
        if not full_name:
            continue
        cls_nm = type(mod).__name__.lower()
        is_norm_type = isinstance(mod, nn.LayerNorm) or (rms_cls is not None and isinstance(mod, rms_cls)) or ("norm" in cls_nm)
        if not is_norm_type and "norm" not in full_name.lower():
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.ndim != 1:
            continue
        # Only wrap leaf norm modules without child linear layers
        if any(isinstance(c, nn.Linear) for c in mod.modules()):
            continue
        parts = full_name.split(".")
        parent: nn.Module = model
        for pt in parts[:-1]:
            parent = getattr(parent, pt)
        attr = parts[-1]
        adapter = _NormScaleAdapter(mod, int(w.numel()), dev)
        setattr(parent, attr, adapter)
        wrapped.append((parent, attr, mod, adapter))

    assert len(wrapped) > 0, "No 1D norm modules found for Scale-QAT"
    gammas = [ad.gamma for *_rest, ad in wrapped]

    # Measure baseline dev KL
    model.eval()
    base_dev = eval_topk_kl_and_top1(model, dev_records, dev) if dev_records else {"kl_mean": 0.0, "top1_agreement": 1.0}
    best_dev_kl = float(base_dev["kl_mean"])
    best_gammas = [g.detach().clone() for g in gammas]

    # Enable input grad on embeddings so gradient checkpointing preserves backward graph with frozen base weights
    emb_handles = []
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if isinstance(emb, nn.Module):
        emb_handles.append(emb.register_forward_hook(lambda _m, _inp, out: out.requires_grad_(True)))

    total_opt_steps = max(1, (int(epochs) * len(train_records) + int(grad_accum) - 1) // int(grad_accum))
    opt = torch.optim.AdamW(gammas, lr=float(lr), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_opt_steps)
    rng = np.random.default_rng(int(seed))

    epoch_history: list[dict[str, float]] = []
    try:
        for ep in range(int(epochs)):
            order = rng.permutation(len(train_records)).tolist()
            opt.zero_grad(set_to_none=True)
            train_kls: list[float] = []
            for step_i, rec_idx in enumerate(order):
                rec = train_records[rec_idx]
                ids = rec["tokens"].to(device=dev, dtype=torch.long).unsqueeze(0)
                top_idx = rec["top_idx"].to(dev)
                top_lp = rec["top_logp"].to(dev)
                tail_p = rec["tail_p"].to(dev)
                with torch.enable_grad():
                    out = model(input_ids=ids, use_cache=False)
                    kl = topk_tail_kl_loss(out.logits[0], top_idx, top_lp, tail_p)
                    reg = torch.stack([(g - 1.0).pow(2).mean() for g in gammas]).mean()
                    loss = (kl + float(anchor_lambda) * reg) / float(grad_accum)
                    loss.backward()
                train_kls.append(float(kl.detach().item()))
                del ids, out, kl, reg, loss
                if (step_i + 1) % int(grad_accum) == 0 or (step_i + 1) == len(order):
                    torch.nn.utils.clip_grad_norm_(gammas, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)

            ep_dev = eval_topk_kl_and_top1(model, dev_records, dev) if dev_records else {"kl_mean": float(np.mean(train_kls)), "top1_agreement": 1.0}
            epoch_history.append(
                {
                    "epoch": float(ep + 1),
                    "train_kl": float(np.mean(train_kls)),
                    "dev_kl": float(ep_dev["kl_mean"]),
                    "dev_top1": float(ep_dev["top1_agreement"]),
                }
            )
            if float(ep_dev["kl_mean"]) < best_dev_kl:
                best_dev_kl = float(ep_dev["kl_mean"])
                best_gammas = [g.detach().clone() for g in gammas]

        # Fold best gammas directly back into original norm weights (0 extra bytes!)
        with torch.no_grad():
            for (*_rest, mod, _ad), best_g in zip(wrapped, best_gammas):
                mod.weight.data.copy_((mod.weight.data.float() * best_g.to(mod.weight.device)).to(mod.weight.dtype))
    finally:
        for h in emb_handles:
            h.remove()
        for parent, attr, orig_mod, _ad in wrapped:
            setattr(parent, attr, orig_mod)
        if enabled_gc and hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
        if was_training:
            model.train()
        for p, req in orig_req:
            p.requires_grad_(req)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "n_tuned_norms": len(wrapped),
        "n_tuned_params": int(sum(int(g.numel()) for g in best_gammas)),
        "dev_kl_before": round(float(base_dev["kl_mean"]), 6),
        "dev_kl_best": round(float(best_dev_kl), 6),
        "epoch_history": epoch_history,
    }
