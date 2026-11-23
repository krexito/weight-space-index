"""Delta X-Ray (`wsi_delta_xray_v1`): compact structural, spectral, and causal-attribution diffing between checkpoints.

Turns a controlled `(before -> after)` model transition (e.g., Base -> SFT -> DPO -> RLVR, Base -> Math,
Math -> R1-Distill, Base -> Coder, or Base -> Pocket MoE) into a ~3-5.5 KB `.delta.xray.json` chart
designed to sit alongside Tome 1 (`wsi_xray_v1`) in an agent's anatomical atlas.
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fisher import _resolve_device
from .xray import (
    DEPTH_BINS,
    _ModelStructure,
    _discover_xray_structure,
    _extract_flat_biomarkers,
    _forward_logits,
    _gini,
    _is_finite_num,
    _orient_writer_cols,
    _resample_curve,
    _sanitize,
    format_compact_xray_json,
    load_xray_json,
    scan_model_xray,
)
from .xray_probes import CALIBRATION_PASSAGES, build_frozen_probe_pack

DELTA_XRAY_SCHEMA_VERSION = "wsi_delta_xray_v1"

ROLE_ORDER: tuple[str, ...] = (
    "mlp_out",
    "mlp_in",
    "attn_out",
    "attn_qkv",
    "ssm_conv",
    "moe_router",
    "norms",
    "embed_head",
    "other",
)
MLP_ROLES = {"mlp_in", "mlp_out", "moe_router"}
MIXER_ROLES = {"attn_qkv", "attn_out", "ssm_conv"}
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks|h|decoder\.layers)\.(\d+)\.")

_WRAPPER_PREFIXES = ("model.", "language_model.", "backbone.", "transformer.")


def extract_layer_idx(param_name: str) -> int | None:
    """Extract decoder block index from standard HF / WSI parameter names."""
    m = LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


def _strip_wrapper_prefixes(name: str) -> str:
    cur = str(name)
    changed = True
    while changed:
        changed = False
        for pref in _WRAPPER_PREFIXES:
            if cur.startswith(pref):
                cur = cur[len(pref) :]
                changed = True
                break
    return cur


def _is_aux_multimodal_or_mtp(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in ("mtp.", ".mtp", "visual.", ".visual", "vision_tower", "multi_modal_projector"))


def classify_param_role(param_name: str, shape: tuple[int, ...] | None = None) -> str:
    """Classify a parameter tensor into an architecture-universal functional bucket."""
    p = param_name.lower()
    if "norm" in p or re.search(r"(^|\.)ln_?\d*(\.|$)", p):
        return "norms"
    if any(tok in p for tok in ("embed_tokens", "lm_head", "wte", "wpe", "word_embeddings")):
        return "embed_head"
    if (
        (p.endswith(".gate.weight") or p.endswith(".router.weight") or p.endswith(".gate.bias") or p.endswith(".router.bias"))
        and any(tok in p for tok in (".mlp.", ".moe.", ".block_sparse_moe.", ".feed_forward."))
        and "experts" not in p
        and "gate_proj" not in p
        and (shape is None or min(shape) <= 256)
    ):
        return "moe_router"
    if any(tok in p for tok in (".conv.", ".mamba.", ".ssm.", ".mixer.", ".linear_attn.")):
        return "ssm_conv"
    if any(tok in p for tok in (".self_attn.", ".attn.", ".attention.", ".self_attention.", ".shared_transformer.")):
        if any(tok in p for tok in ("o_proj", "out_proj", ".dense.")):
            return "attn_out"
        if any(tok in p for tok in ("q_proj", "k_proj", "v_proj", "qkv", "c_attn", "in_proj")):
            return "attn_qkv"
        if p.endswith("dense.weight") or p.endswith("dense.bias"):
            return "attn_out"
        if "o_proj" in p or "out_proj" in p or "c_proj" in p:
            return "attn_out"
        return "attn_qkv"
    if any(tok in p for tok in ("mlp", "ffn", "feed_forward", "expert")):
        if any(tok in p for tok in ("down_proj", ".w2.", "fc2", "c_proj", "dense_4h_to_h", "output_linear")):
            return "mlp_out"
        if any(tok in p for tok in ("gate_proj", "up_proj", "gate_up_proj", ".w1.", ".w3.", "fc1", "c_fc", "linear_fc1")):
            return "mlp_in"
    if any(tok in p for tok in ("down_proj", "fc2", "dense_4h_to_h", "output_linear")):
        return "mlp_out"
    if any(tok in p for tok in ("gate_proj", "up_proj", "gate_up_proj", "fc1", "c_fc", "linear_fc1")):
        return "mlp_in"
    return "other"


class _WeightSource:
    """Unified read-only tensor lookup over `nn.Module`, tensor mapping, or `*.safetensors` checkpoint directory."""

    def __init__(self, source: nn.Module | Mapping[str, torch.Tensor] | str | Path) -> None:
        self._kind: str
        self._tensors: dict[str, torch.Tensor] = {}
        self._key_to_file: dict[str, Path] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._open_handles: dict[Path, Any] = {}
        self._ordered_keys: list[str] = []
        self._norm_to_exact: dict[str, str] = {}

        if isinstance(source, nn.Module):
            self._kind = "module"
            for name, p in source.named_parameters():
                if _is_aux_multimodal_or_mtp(name):
                    continue
                self._tensors[name] = p.detach()
                self._shapes[name] = tuple(p.shape)
                self._ordered_keys.append(name)
        elif isinstance(source, Mapping):
            self._kind = "mapping"
            for name, t in source.items():
                if _is_aux_multimodal_or_mtp(str(name)) or not isinstance(t, torch.Tensor):
                    continue
                k = str(name)
                self._tensors[k] = t.detach()
                self._shapes[k] = tuple(t.shape)
                self._ordered_keys.append(k)
        else:
            root = Path(source)
            sf_files = sorted(root.glob("*.safetensors"))
            if not sf_files:
                raise FileNotFoundError(f"No *.safetensors files found in checkpoint directory: {root}")
            self._kind = "safetensors"
            from safetensors import safe_open

            for sf in sf_files:
                with safe_open(str(sf), "pt") as fh:
                    for k in fh.keys():
                        if _is_aux_multimodal_or_mtp(k):
                            continue
                        self._key_to_file[k] = sf
                        if hasattr(fh, "get_slice"):
                            self._shapes[k] = tuple(fh.get_slice(k).get_shape())
                        else:
                            self._shapes[k] = tuple(fh.get_tensor(k).shape)
                        self._ordered_keys.append(k)

        for k in self._ordered_keys:
            nk = _strip_wrapper_prefixes(k)
            if nk not in self._norm_to_exact:
                self._norm_to_exact[nk] = k

    def keys(self) -> list[str]:
        return list(self._ordered_keys)

    def _resolve_key(self, name: str) -> str | None:
        if name in self._shapes:
            return name
        nk = _strip_wrapper_prefixes(name)
        if nk in self._norm_to_exact:
            return self._norm_to_exact[nk]
        # Tied embeddings fallback when one side serializes only embed_tokens or only lm_head.
        if nk == "lm_head.weight":
            for alt in ("embed_tokens.weight", "wte.weight", "word_embeddings.weight"):
                if alt in self._norm_to_exact:
                    return self._norm_to_exact[alt]
        elif nk in ("embed_tokens.weight", "wte.weight", "word_embeddings.weight"):
            if "lm_head.weight" in self._norm_to_exact:
                return self._norm_to_exact["lm_head.weight"]
        return None

    def shape(self, name: str) -> tuple[int, ...] | None:
        rk = self._resolve_key(name)
        return self._shapes.get(rk) if rk is not None else None

    def get(self, name: str) -> torch.Tensor | None:
        rk = self._resolve_key(name)
        if rk is None:
            return None
        if self._kind in ("module", "mapping"):
            return self._tensors[rk]
        sf = self._key_to_file[rk]
        fh = self._open_handles.get(sf)
        if fh is None:
            from safetensors import safe_open

            fh = safe_open(str(sf), "pt")
            self._open_handles[sf] = fh
        return fh.get_tensor(rk)

    def close(self) -> None:
        for fh in self._open_handles.values():
            try:
                if hasattr(fh, "__exit__"):
                    fh.__exit__(None, None, None)
            except Exception:
                pass
        self._open_handles.clear()

    def __enter__(self) -> "_WeightSource":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


# ----------------------------------------------------------------------------------------------- Stage 1: weight + spectral
def _param_name_map(model: nn.Module) -> dict[int, str]:
    """Map `id(Parameter)` -> parameter name (capturing tied parameters too via `remove_duplicate=False`)."""
    out: dict[int, str] = {}
    try:
        it = model.named_parameters(remove_duplicate=False)
    except TypeError:
        it = model.named_parameters()
    for nm, p in it:
        pid = id(p)
        if pid not in out:
            out[pid] = nm
    return out


def _resolve_module_weight_name(
    mod: nn.Module,
    fallback_rel: str,
    param_id_map: dict[int, str],
) -> str | None:
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor):
        hit = param_id_map.get(id(w))
        if hit:
            return hit
    return f"{fallback_rel}.weight" if fallback_rel else None


def _collect_block_target_names(
    struct: _ModelStructure,
    max_moe_experts_per_layer: int,
) -> tuple[list[tuple[int, list[tuple[str, bool]]]], list[tuple[int, str, bool]]]:
    """Return `(mlp_targets_by_layer, mixer_targets)` with parameter names and `is_conv1d` flags."""
    p_map = _param_name_map(struct.backbone)
    mlp_by_layer: list[tuple[int, list[tuple[str, bool]]]] = []
    mixer_targets: list[tuple[int, str, bool]] = []

    for blk in struct.blocks:
        mlp_list: list[tuple[str, bool]] = []
        if blk.mlp_down is not None:
            rel_name, mod = blk.mlp_down
            pname = _resolve_module_weight_name(mod, rel_name, p_map)
            if pname:
                mlp_list.append((pname, not isinstance(mod, nn.Linear)))
        elif blk.is_moe:
            cands: list[tuple[str, bool]] = []
            for sub_n, sub_m in blk.module.named_modules():
                low = sub_n.lower()
                if not sub_n or "expert" not in low:
                    continue
                if isinstance(getattr(sub_m, "weight", None), torch.Tensor) and sub_m.weight.dim() == 2:
                    if low.endswith(("down_proj", "w2", "fc2", "c_proj")):
                        pname = _resolve_module_weight_name(sub_m, f"{blk.name}.{sub_n}", p_map)
                        if pname:
                            cands.append((pname, not isinstance(sub_m, nn.Linear)))
            if cands:
                if len(cands) > max_moe_experts_per_layer:
                    picks = np.linspace(0, len(cands) - 1, max_moe_experts_per_layer, dtype=int).tolist()
                    cands = [cands[i] for i in picks]
                mlp_list.extend(cands)
        if mlp_list:
            mlp_by_layer.append((blk.idx, mlp_list))

        if blk.mixer_out is not None:
            rel_m, mod_m = blk.mixer_out
            pname_m = _resolve_module_weight_name(mod_m, rel_m, p_map)
            if pname_m:
                mixer_targets.append((blk.idx, pname_m, not isinstance(mod_m, nn.Linear)))

    return mlp_by_layer, mixer_targets


@torch.no_grad()
def _analyze_matrix_delta(
    W_before: torch.Tensor,
    W_after: torch.Tensor,
    *,
    hidden_size: int,
    is_conv1d: bool,
    dev: torch.device,
) -> dict[str, Any] | None:
    if W_before.dim() != 2 or W_after.dim() != 2 or W_before.shape != W_after.shape:
        return None
    Wb = _orient_writer_cols(W_before.to(dev), hidden_size, is_conv1d=is_conv1d).float()
    Wa = _orient_writer_cols(W_after.to(dev), hidden_size, is_conv1d=is_conv1d).float()
    dW = Wa - Wb
    r, c = dW.shape
    m = min(r, c)
    if m == 0:
        return None

    gram_d = (dW @ dW.t()) if r <= c else (dW.t() @ dW)
    gram_d = 0.5 * (gram_d + gram_d.t())
    evals, evecs = torch.linalg.eigh(gram_d)
    evals = evals.clamp_min(0.0)
    tot_e = float(evals.sum().item())
    if tot_e <= 1e-12:
        return None

    sv = evals.sqrt()
    p = sv / sv.sum().clamp_min(1e-12)
    p_pos = p[p > 1e-12]
    erank = math.exp(float(-(p_pos * p_pos.log()).sum().item()))
    erank_ratio = erank / max(1, m)

    evals_desc = evals.flip(0)
    cum = torch.cumsum(evals_desc, dim=0) / tot_e
    n_eig = len(evals_desc)
    rank80 = min(n_eig, int((cum < 0.80).sum().item()) + 1)
    rank90 = min(n_eig, int((cum < 0.90).sum().item()) + 1)
    energy_r1 = float(evals_desc[:1].sum().item() / tot_e)
    energy_r8 = float(evals_desc[: min(8, n_eig)].sum().item() / tot_e)
    energy_r32 = float(evals_desc[: min(32, n_eig)].sum().item() / tot_e)
    energy_r64 = float(evals_desc[: min(64, n_eig)].sum().item() / tot_e)

    col_gini = _gini(dW.norm(dim=0))
    wb_norm = float(Wb.norm().item())
    dw_norm = math.sqrt(tot_e)
    cos_with_base = float((Wb * dW).sum().item() / max(wb_norm * dw_norm, 1e-12))

    # Measure energy amplification along the active top shifted left singular subspace (up to rank 8).
    max_e = float(evals_desc[0].item())
    k_active = max(1, min(8, int((evals_desc > 1e-8 * max_e).sum().item())))
    if r <= c:
        U_k = evecs[:, -k_active:]
    else:
        V_k = evecs[:, -k_active:]
        sv_k = sv[-k_active:].clamp_min(1e-12)
        U_k = (dW @ V_k) / sv_k.unsqueeze(0)
        U_k, _ = torch.linalg.qr(U_k)

    e_before = float((U_k.t() @ Wb).pow(2).sum().item())
    e_after = float((U_k.t() @ Wa).pow(2).sum().item())
    amp_ratio = (e_after / e_before) if e_before > 1e-12 else None

    # Overlap of dW with the base matrix's top-32 left singular subspace vs isotropic expectation (k_base / r).
    k_base = min(32, m, r)
    if k_base > 0:
        if r <= c:
            gram_b = Wb @ Wb.t()
            gram_b = 0.5 * (gram_b + gram_b.t())
            _, evecs_b = torch.linalg.eigh(gram_b)
            U_base = evecs_b[:, -k_base:]
        else:
            Ub, _, _ = torch.linalg.svd(Wb, full_matrices=False)
            U_base = Ub[:, :k_base]
        base_ov = float((U_base.t() @ dW).pow(2).sum().item() / tot_e)
        chance = max(k_base / max(r, 1), 1e-12)
        base_ov_ratio = base_ov / chance
    else:
        base_ov = None
        base_ov_ratio = None

    return {
        "erank_ratio": float(erank_ratio),
        "energy_r1": energy_r1,
        "energy_r8": energy_r8,
        "energy_r32": energy_r32,
        "energy_r64": energy_r64,
        "rank80": int(rank80),
        "rank90": int(rank90),
        "col_gini": float(col_gini),
        "cos_with_base": cos_with_base,
        "top_delta_amp_ratio": float(amp_ratio) if amp_ratio is not None else None,
        "base_top32_overlap": float(base_ov) if base_ov is not None else None,
        "base_top32_overlap_vs_chance": float(base_ov_ratio) if base_ov_ratio is not None else None,
    }


def _aggregate_spectral_group(
    per_layer_stats: list[list[dict[str, Any]] | None],
    n_mats: int,
) -> dict[str, Any]:
    flat: list[dict[str, Any]] = [st for grp in per_layer_stats if grp for st in grp]
    erank_curve: list[float | None] = []
    gini_curve: list[float | None] = []
    for grp in per_layer_stats:
        if grp:
            erank_curve.append(float(np.mean([g["erank_ratio"] for g in grp])))
            gini_curve.append(float(np.mean([g["col_gini"] for g in grp])))
        else:
            erank_curve.append(None)
            gini_curve.append(None)

    def _mean(key: str) -> float | None:
        vals = [float(s[key]) for s in flat if s.get(key) is not None and math.isfinite(float(s[key]))]
        return float(np.mean(vals)) if vals else None

    def _median(key: str) -> float | None:
        vals = [float(s[key]) for s in flat if s.get(key) is not None and math.isfinite(float(s[key]))]
        return float(np.median(vals)) if vals else None

    return {
        "n_mats": int(n_mats),
        "erank_ratio_mean": _mean("erank_ratio"),
        "erank_ratio_8": _resample_curve(erank_curve),
        "energy_r1_mean": _mean("energy_r1"),
        "energy_r8_mean": _mean("energy_r8"),
        "energy_r32_mean": _mean("energy_r32"),
        "energy_r64_mean": _mean("energy_r64"),
        "rank80_median": _median("rank80"),
        "rank90_median": _median("rank90"),
        "col_gini_mean": _mean("col_gini"),
        "col_gini_8": _resample_curve(gini_curve),
        "cos_with_base_mean": _mean("cos_with_base"),
        "top_delta_amp_ratio_mean": _mean("top_delta_amp_ratio"),
        "base_top32_overlap_mean": _mean("base_top32_overlap"),
        "base_top32_overlap_vs_chance": _mean("base_top32_overlap_vs_chance"),
    }


@torch.no_grad()
def compute_weight_and_spectral_diff(
    before_source: nn.Module | Mapping[str, torch.Tensor] | str | Path | _WeightSource,
    after_source: nn.Module | Mapping[str, torch.Tensor] | str | Path | _WeightSource,
    struct_before: _ModelStructure,
    *,
    device: torch.device | str | None = None,
    max_moe_experts_per_layer: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream parameters across `(before, after)` to measure norm diff curves and spectral writer rank."""
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    own_b = not isinstance(before_source, _WeightSource)
    own_a = not isinstance(after_source, _WeightSource)
    src_b = before_source if isinstance(before_source, _WeightSource) else _WeightSource(before_source)
    src_a = after_source if isinstance(after_source, _WeightSource) else _WeightSource(after_source)

    n_layers = len(struct_before.blocks)
    role_d_sq = {r: 0.0 for r in ROLE_ORDER}
    role_b_sq = {r: 0.0 for r in ROLE_ORDER}
    layer_d_sq = [0.0] * n_layers
    layer_b_sq = [0.0] * n_layers
    mlp_d_sq = [0.0] * n_layers
    mlp_b_sq = [0.0] * n_layers
    mix_d_sq = [0.0] * n_layers
    mix_b_sq = [0.0] * n_layers
    norm_d_sq = [0.0] * n_layers
    norm_b_sq = [0.0] * n_layers

    total_d_sq = 0.0
    total_b_sq = 0.0
    n_matched = 0
    n_changed = 0
    n_shape_mismatch = 0

    try:
        for k in src_b.keys():
            if _is_aux_multimodal_or_mtp(k):
                continue
            sa = src_a.shape(k)
            if sa is None:
                continue
            sb = src_b.shape(k)
            if sb is None or tuple(sb) != tuple(sa):
                n_shape_mismatch += 1
                continue
            wb_cpu = src_b.get(k)
            wa_cpu = src_a.get(k)
            if wb_cpu is None or wa_cpu is None:
                continue

            wb = wb_cpu.to(device=dev, dtype=torch.float32)
            wa = wa_cpu.to(device=dev, dtype=torch.float32)
            dW = wa - wb
            d_sq = float((dW * dW).sum().item())
            b_sq = float((wb * wb).sum().item())
            del wb, wa, dW

            n_matched += 1
            if d_sq > 1e-18:
                n_changed += 1
            total_d_sq += d_sq
            total_b_sq += b_sq

            role = classify_param_role(k, tuple(sb))
            role_d_sq[role] += d_sq
            role_b_sq[role] += b_sq

            li = extract_layer_idx(k)
            if li is not None and 0 <= li < n_layers:
                layer_d_sq[li] += d_sq
                layer_b_sq[li] += b_sq
                if role in MLP_ROLES:
                    mlp_d_sq[li] += d_sq
                    mlp_b_sq[li] += b_sq
                elif role in MIXER_ROLES:
                    mix_d_sq[li] += d_sq
                    mix_b_sq[li] += b_sq
                elif role == "norms":
                    norm_d_sq[li] += d_sq
                    norm_b_sq[li] += b_sq

        layer_rel: list[float | None] = [
            (math.sqrt(layer_d_sq[i]) / math.sqrt(layer_b_sq[i])) if layer_b_sq[i] > 1e-12 else None
            for i in range(n_layers)
        ]
        layer_d_tot = sum(layer_d_sq)
        layer_share: list[float | None] = [
            (layer_d_sq[i] / layer_d_tot) if layer_d_tot > 1e-18 else 0.0
            for i in range(n_layers)
        ]
        mlp_rel: list[float | None] = [
            (math.sqrt(mlp_d_sq[i]) / math.sqrt(mlp_b_sq[i])) if mlp_b_sq[i] > 1e-12 else None
            for i in range(n_layers)
        ]
        mix_rel: list[float | None] = [
            (math.sqrt(mix_d_sq[i]) / math.sqrt(mix_b_sq[i])) if mix_b_sq[i] > 1e-12 else None
            for i in range(n_layers)
        ]
        norm_rel: list[float | None] = [
            (math.sqrt(norm_d_sq[i]) / math.sqrt(norm_b_sq[i])) if norm_b_sq[i] > 1e-12 else None
            for i in range(n_layers)
        ]

        depth_centroid = None
        if layer_d_tot > 1e-18 and n_layers > 0:
            denom_d = max(n_layers - 1, 1)
            depth_centroid = float(sum((layer_d_sq[i] / layer_d_tot) * (i / denom_d) for i in range(n_layers)))

        ranked_layers = sorted(
            [(i, float(v)) for i, v in enumerate(layer_rel) if v is not None and v > 0.0],
            key=lambda kv: kv[1],
            reverse=True,
        )
        top3_layers = [
            {"layer": i, "depth": round(i / max(n_layers - 1, 1), 3), "rel_fro": val}
            for i, val in ranked_layers[:3]
        ]

        role_share: dict[str, float] = {}
        role_rel: dict[str, float] = {}
        for r in ROLE_ORDER:
            if role_b_sq[r] > 1e-12:
                role_share[r] = (role_d_sq[r] / total_d_sq) if total_d_sq > 1e-18 else 0.0
                role_rel[r] = math.sqrt(role_d_sq[r]) / math.sqrt(role_b_sq[r])

        weight_diff = {
            "depth_bins": DEPTH_BINS,
            "total_rel_fro": (math.sqrt(total_d_sq) / max(math.sqrt(total_b_sq), 1e-12)) if total_d_sq > 1e-18 else 0.0,
            "n_matched_params": n_matched,
            "n_changed_params": n_changed,
            "n_shape_mismatch_params": n_shape_mismatch,
            "depth_centroid": depth_centroid,
            "top3_changed_layers": top3_layers,
            "role_energy_share": role_share,
            "role_rel_fro": role_rel,
            "layer_rel_fro_8": _resample_curve(layer_rel),
            "layer_energy_share_8": _resample_curve(layer_share),
            "mlp_rel_fro_8": _resample_curve(mlp_rel),
            "mixer_rel_fro_8": _resample_curve(mix_rel),
            "norm_rel_fro_8": _resample_curve(norm_rel),
        }

        # Spectral & base-subspace analysis on writer matrices
        mlp_targets, mixer_targets = _collect_block_target_names(struct_before, max_moe_experts_per_layer)
        mlp_per_layer: list[list[dict[str, Any]] | None] = [None] * n_layers
        mix_per_layer: list[list[dict[str, Any]] | None] = [None] * n_layers
        n_mlp_mats = 0
        n_mix_mats = 0

        for li, plist in mlp_targets:
            grp_stats: list[dict[str, Any]] = []
            for pname, is_conv in plist:
                wb_t, wa_t = src_b.get(pname), src_a.get(pname)
                if wb_t is None or wa_t is None or wb_t.shape != wa_t.shape:
                    continue
                st = _analyze_matrix_delta(
                    wb_t, wa_t, hidden_size=struct_before.hidden_size, is_conv1d=is_conv, dev=dev
                )
                if st is not None:
                    grp_stats.append(st)
                    n_mlp_mats += 1
            if grp_stats:
                mlp_per_layer[li] = grp_stats

        for li, pname, is_conv in mixer_targets:
            wb_t, wa_t = src_b.get(pname), src_a.get(pname)
            if wb_t is None or wa_t is None or wb_t.shape != wa_t.shape:
                continue
            st = _analyze_matrix_delta(
                wb_t, wa_t, hidden_size=struct_before.hidden_size, is_conv1d=is_conv, dev=dev
            )
            if st is not None:
                mix_per_layer[li] = [st]
                n_mix_mats += 1

        spectral_diff = {
            "mlp_out": _aggregate_spectral_group(mlp_per_layer, n_mlp_mats),
            "mixer_out": _aggregate_spectral_group(mix_per_layer, n_mix_mats),
        }
    finally:
        if own_b:
            src_b.close()
        if own_a:
            src_a.close()
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return weight_diff, spectral_diff


# ----------------------------------------------------------------------------------------------- Stage 2: domains + grad dot
def build_domain_calibration_seqs(
    tok: Any,
    probe_pack: Mapping[str, Any] | None = None,
    *,
    max_length: int = 256,
) -> dict[str, list[list[int]]]:
    """Bucket frozen calibration passages into `prose`, `code`, `math`, and `other` token sequences."""
    rows = (probe_pack or {}).get("calibration_passages") or CALIBRATION_PASSAGES
    buckets: dict[str, list[list[int]]] = {"prose": [], "code": [], "math": [], "other": []}
    for row in rows:
        dom_raw = str(row.get("domain", "")).lower()
        txt = str(row.get("text", ""))
        if not txt:
            continue
        ids = tok.encode(txt, add_special_tokens=True, truncation=True, max_length=int(max_length))
        seq = [int(x) for x in ids]
        if len(seq) < 2:
            continue
        if dom_raw.startswith("prose"):
            buckets["prose"].append(seq)
        elif dom_raw.startswith("code"):
            buckets["code"].append(seq)
        elif dom_raw.startswith("math") or dom_raw.startswith("logic"):
            buckets["math"].append(seq)
        else:
            buckets["other"].append(seq)
    return buckets


@torch.no_grad()
def capture_domain_probe_state(
    model: nn.Module,
    domain_seqs: Mapping[str, Sequence[Sequence[int]]],
    *,
    device: torch.device | str | None = None,
    top_k: int = 64,
) -> dict[str, Any]:
    """Capture compact top-K next-token log-probabilities and NLL sums across domains."""
    dev = _resolve_device(model, device)
    model.eval()
    out: dict[str, Any] = {}
    for dom, seqs in domain_seqs.items():
        nll_sum = 0.0
        n_pred = 0
        items: list[dict[str, Any]] = []
        for seq in seqs:
            if len(seq) < 2:
                continue
            inp = torch.tensor([list(seq)], dtype=torch.long, device=dev)
            logits = _forward_logits(model, inp)[0, :-1].float()
            lp = F.log_softmax(logits, dim=-1)
            tgt = inp[0, 1:]
            nll_sum += float(F.nll_loss(lp, tgt, reduction="sum").item())
            n_pred += int(tgt.numel())

            k_use = min(int(top_k), int(lp.shape[-1]))
            vals, idxs = lp.topk(k_use, dim=-1)
            probs_k = vals.exp()
            tail_p = (1.0 - probs_k.sum(dim=-1)).clamp_min(0.0)
            items.append(
                {
                    "vocab_size": int(lp.shape[-1]),
                    "topk_idx": idxs.to(torch.int32).cpu(),
                    "topk_logp": vals.to(torch.float16).cpu(),
                    "tail_prob": tail_p.to(torch.float16).cpu(),
                }
            )
            del inp, logits, lp
        out[dom] = {
            "nll_sum": float(nll_sum),
            "n_pred": int(n_pred),
            "mean_nll": (float(nll_sum) / n_pred) if n_pred > 0 else None,
            "items": items,
        }
    return out


@torch.no_grad()
def compare_domain_probe_states(
    state_before: Mapping[str, Any],
    model_after: nn.Module,
    domain_seqs: Mapping[str, Sequence[Sequence[int]]],
    *,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Compute exact domain NLL shifts, top-1 agreement, and top-64 KL divergence vs captured before state."""
    dev = _resolve_device(model_after, device)
    model_after.eval()
    domains_out: dict[str, dict[str, Any]] = {}
    for dom, seqs in domain_seqs.items():
        sb_dom = state_before.get(dom, {})
        sb_items = sb_dom.get("items", [])
        nll_before = sb_dom.get("mean_nll")
        nll_after_sum = 0.0
        n_pred = 0
        kl_sum = 0.0
        n_kl = 0
        top1_eq = 0

        for idx_s, seq in enumerate(seqs):
            if len(seq) < 2:
                continue
            inp = torch.tensor([list(seq)], dtype=torch.long, device=dev)
            logits = _forward_logits(model_after, inp)[0, :-1].float()
            lp_a = F.log_softmax(logits, dim=-1)
            tgt = inp[0, 1:]
            nll_after_sum += float(F.nll_loss(lp_a, tgt, reduction="sum").item())
            n_pred += int(tgt.numel())

            if idx_s < len(sb_items):
                ref = sb_items[idx_s]
                v_a = int(lp_a.shape[-1])
                tk_idx = ref["topk_idx"].to(device=dev, dtype=torch.long)
                tk_lp_b = ref["topk_logp"].to(device=dev, dtype=torch.float32)
                tail_b = ref["tail_prob"].to(device=dev, dtype=torch.float32)

                pred_a = lp_a.argmax(dim=-1)
                top1_eq += int((pred_a == tk_idx[:, 0]).sum().item())

                valid_mask = tk_idx < v_a
                safe_idx = tk_idx.clamp(max=max(0, v_a - 1))
                lp_a_gathered = lp_a.gather(1, safe_idx)
                lp_a_gathered = torch.where(valid_mask, lp_a_gathered, torch.full_like(lp_a_gathered, -30.0))

                p_b_k = torch.where(valid_mask, tk_lp_b.exp(), torch.zeros_like(tk_lp_b))
                p_a_k = torch.where(valid_mask, lp_a_gathered.exp(), torch.zeros_like(lp_a_gathered))
                kl_top = (p_b_k * (tk_lp_b - lp_a_gathered)).sum(dim=-1)

                tail_a = (1.0 - p_a_k.sum(dim=-1)).clamp_min(1e-8)
                tail_b_safe = tail_b.clamp_min(1e-8)
                kl_tail = torch.where(tail_b > 1e-6, tail_b * (tail_b_safe.log() - tail_a.log()), torch.zeros_like(tail_b))
                kl_tot = (kl_top + kl_tail).clamp_min(0.0)
                kl_sum += float(kl_tot.sum().item())
                n_kl += int(kl_tot.numel())

            del inp, logits, lp_a

        nll_after = (nll_after_sum / n_pred) if n_pred > 0 else None
        delta_nll = (float(nll_after) - float(nll_before)) if (nll_after is not None and nll_before is not None) else None
        domains_out[dom] = {
            "n_tokens": n_pred,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "delta_nll": delta_nll,
            "kl_top64": (kl_sum / n_kl) if n_kl > 0 else None,
            "top1_agree": (top1_eq / n_kl) if n_kl > 0 else None,
        }
    return domains_out


def accumulate_domain_grad_dots(
    model: nn.Module,
    other_source: nn.Module | Mapping[str, torch.Tensor] | str | Path | _WeightSource,
    domain_seqs: Mapping[str, Sequence[Sequence[int]]],
    *,
    sign: float,
    device: torch.device | str | None = None,
    max_len: int = 128,
    compute_prose_fisher: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float], int]:
    """Stream exact `⟨∇L_dom, W_after - W_before⟩` inner products using O(1) peak gradient memory."""
    dev = _resolve_device(model, device)
    own_other = not isinstance(other_source, _WeightSource)
    src_other = other_source if isinstance(other_source, _WeightSource) else _WeightSource(other_source)

    orig_req = [(p, bool(p.requires_grad)) for p in model.parameters()]
    for p, _ in orig_req:
        p.requires_grad_(False)
        p.grad = None

    matched_params: list[tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if _is_aux_multimodal_or_mtp(name):
            continue
        osh = src_other.shape(name)
        if osh is not None and tuple(osh) == tuple(p.shape):
            p.requires_grad_(True)
            matched_params.append((name, p))

    was_training = bool(model.training)
    was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
    enabled_gc = False
    if matched_params and hasattr(model, "gradient_checkpointing_enable") and not was_gc:
        try:
            model.gradient_checkpointing_enable()
            enabled_gc = True
        except Exception:
            enabled_gc = False

    state: dict[str, Any] = {
        "domain": "",
        "weight": 1.0,
        "dots": {},
        "fisher": {},
    }
    hooks: list[Any] = []
    emb_mod = (
        model.get_input_embeddings()
        if hasattr(model, "get_input_embeddings")
        else getattr(getattr(model, "model", model), "embed_tokens", None)
    )
    if emb_mod is not None:
        def _emb_anchor(_m: nn.Module, _inp: Any, emb_out: Any) -> Any:
            if not torch.is_grad_enabled():
                return emb_out
            t = emb_out[0] if isinstance(emb_out, tuple) else emb_out
            if isinstance(t, torch.Tensor) and not t.requires_grad:
                t2 = t.detach().requires_grad_(True)
                return (t2, *emb_out[1:]) if isinstance(emb_out, tuple) else t2
            return emb_out

        hooks.append(emb_mod.register_forward_hook(_emb_anchor))

    for name, p in matched_params:
        def _make_hook(pname: str):
            def _post_acc(param: nn.Parameter) -> None:
                g = param.grad
                if g is None:
                    return None
                other_t = src_other.get(pname)
                if other_t is not None and tuple(other_t.shape) == tuple(param.shape):
                    dW = (other_t.to(device=param.device, dtype=torch.float32) - param.detach().float()) * float(sign)
                    gf = g.detach().float()
                    dot_v = float((gf * dW).sum().item())
                    state["dots"][pname] = state["dots"].get(pname, 0.0) + dot_v
                    if compute_prose_fisher and state["domain"] == "prose":
                        g_seq = gf / max(float(state["weight"]), 1e-12)
                        f_v = float(((g_seq * dW) ** 2).sum().item())
                        state["fisher"][pname] = state["fisher"].get(pname, 0.0) + f_v
                    del other_t, dW, gf
                param.grad = None
                return None

            return _post_acc

        hooks.append(p.register_post_accumulate_grad_hook(_make_hook(name)))

    attr_domains = ("prose", "math", "code")
    dots_by_domain: dict[str, dict[str, float]] = {d: {} for d in attr_domains}
    grad_nll_by_domain: dict[str, float] = {}
    prose_fisher: dict[str, float] = {}
    cand_lens = [int(max_len)]
    for fb in (64, 32):
        if fb < int(max_len):
            cand_lens.append(fb)
    used_len = int(max_len)

    try:
        model.eval()
        if not matched_params:
            return dots_by_domain, grad_nll_by_domain, prose_fisher, used_len

        for dom in attr_domains:
            raw_seqs = [list(s) for s in domain_seqs.get(dom, []) if len(s) >= 2]
            if not raw_seqs:
                continue
            start_idx = 0
            for idx_c, c_len in enumerate(cand_lens):
                if c_len <= used_len:
                    start_idx = idx_c
                    break

            succeeded = False
            for c_len in cand_lens[start_idx:]:
                clipped = [s[:c_len] for s in raw_seqs if len(s[:c_len]) >= 2]
                if not clipped:
                    succeeded = True
                    break
                tot_tok = sum(len(s) - 1 for s in clipped)
                state["domain"] = dom
                state["dots"] = {}
                if compute_prose_fisher and dom == "prose":
                    state["fisher"] = {}
                dom_nll_sum = 0.0
                try:
                    for seq in clipped:
                        n_p = len(seq) - 1
                        w_s = float(n_p) / max(1, tot_tok)
                        state["weight"] = w_s
                        inp = torch.tensor([seq], dtype=torch.long, device=dev)
                        with torch.enable_grad():
                            logits = _forward_logits(model, inp)[0, :-1].float()
                            loss_seq = F.cross_entropy(logits, inp[0, 1:], reduction="mean")
                            dom_nll_sum += float(loss_seq.detach().item()) * w_s
                            (loss_seq * w_s).backward()
                        del inp, logits, loss_seq
                    dots_by_domain[dom] = dict(state["dots"])
                    grad_nll_by_domain[dom] = float(dom_nll_sum)
                    if compute_prose_fisher and dom == "prose":
                        prose_fisher = {
                            k: float(v) / max(1, len(clipped)) for k, v in state["fisher"].items()
                        }
                    used_len = int(c_len)
                    succeeded = True
                    break
                except torch.cuda.OutOfMemoryError:
                    for _, p in matched_params:
                        p.grad = None
                    gc.collect()
                    if dev.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            if not succeeded:
                raise RuntimeError(f"CUDA OOM during delta attribution on domain={dom!r} down to len={cand_lens[-1]}")
    finally:
        for h in hooks:
            h.remove()
        for p, req in orig_req:
            p.requires_grad_(req)
            p.grad = None
        if enabled_gc and hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
        if was_training:
            model.train()
        if own_other:
            src_other.close()
        gc.collect()
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return dots_by_domain, grad_nll_by_domain, prose_fisher, used_len


def assemble_delta_attribution(
    dots_before: Mapping[str, Mapping[str, float]],
    dots_after: Mapping[str, Mapping[str, float]],
    grad_nll_before: Mapping[str, float],
    grad_nll_after: Mapping[str, float],
    prose_fisher_cost: Mapping[str, float],
    n_layers: int,
    grad_seq_max_len: int,
) -> dict[str, Any]:
    """Combine forward/backward gradient inner products into trapezoidal layer/role credit & faithfulness."""
    domains_out: dict[str, Any] = {}
    for dom in ("prose", "math", "code"):
        if dom not in grad_nll_before or dom not in grad_nll_after:
            continue
        db = dots_before.get(dom, {})
        da = dots_after.get(dom, {})
        all_names = set(db.keys()) | set(da.keys())
        role_cred = {r: 0.0 for r in ROLE_ORDER}
        layer_cred = [0.0] * n_layers
        trap_sum = 0.0
        for pname in all_names:
            c = 0.5 * (float(db.get(pname, 0.0)) + float(da.get(pname, 0.0)))
            trap_sum += c
            role = classify_param_role(pname)
            role_cred[role] += c
            li = extract_layer_idx(pname)
            if li is not None and 0 <= li < n_layers:
                layer_cred[li] += c

        actual_dnll = float(grad_nll_after[dom]) - float(grad_nll_before[dom])
        faithfulness = (trap_sum / actual_dnll) if abs(actual_dnll) >= 1e-4 else None
        dom_rec: dict[str, Any] = {
            "actual_dnll": actual_dnll,
            "trap_sum": trap_sum,
            "faithfulness": faithfulness,
            "role_credit": {r: v for r, v in role_cred.items() if abs(v) > 1e-12},
            "layer_credit_8": _resample_curve(layer_cred) if n_layers > 0 else [],
        }
        if dom == "prose" and prose_fisher_cost and n_layers > 0:
            l_fish = [0.0] * n_layers
            for pname, fv in prose_fisher_cost.items():
                li = extract_layer_idx(pname)
                if li is not None and 0 <= li < n_layers:
                    l_fish[li] += float(fv)
            tot_f = sum(l_fish)
            if tot_f > 1e-18:
                ranked = sorted(range(n_layers), key=lambda i: l_fish[i], reverse=True)
                dom_rec["prose_fisher_cost_total"] = tot_f
                dom_rec["prose_fisher_cost_top3_layers"] = [
                    {
                        "layer": int(li),
                        "depth": round(li / max(n_layers - 1, 1), 3),
                        "share": l_fish[li] / tot_f,
                    }
                    for li in ranked[:3]
                    if l_fish[li] > 0.0
                ]
        domains_out[dom] = dom_rec

    return {
        "method": "trapezoidal_grad_dot",
        "grad_seq_max_len": int(grad_seq_max_len),
        "domains": domains_out,
    }


# ----------------------------------------------------------------------------------------------- Stage 3: Tome-1 delta
_ANATOMY_DELTA_KEYS = (
    "cal_nll",
    "unembed_anisotropy_cos",
    "massive_act_max",
    "mlp_energy90_mean",
    "fisher_late_early_ratio",
    "block_fisher_late_early_ratio",
    "fisher_col_gini_mean",
    "fact_dla_first_quarter",
)

_PATHOLOGY_DELTA_KEYS = (
    "base_overref_rate",
    "base_toxic_ref_rate",
    "opening_entropy_norm",
    "ctrl_opening_entropy_norm",
    "auc_refusal_peak",
    "auc_veto_peak",
    "veto_framing_shift_to_std",
    "turn1_acc",
    "cave_rate_false_pushback",
    "cos_syc_refusal",
    "forced_rep_nll_slope",
    "cos_rep_vs_copy",
    "greedy_rep16_rate",
    "sampled_rep16_rate",
)


def compute_xray_card_delta(card_before: Mapping[str, Any], card_after: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two Tome-1 `wsi_xray_v1` charts into paired before/after/delta entries."""
    fb = _extract_flat_biomarkers(dict(card_before))
    fa = _extract_flat_biomarkers(dict(card_after))

    def _pair_dict(keys: Sequence[str]) -> dict[str, dict[str, float | None]]:
        out: dict[str, dict[str, float | None]] = {}
        for k in keys:
            vb, va = fb.get(k), fa.get(k)
            vb_f = float(vb) if _is_finite_num(vb) else None
            va_f = float(va) if _is_finite_num(va) else None
            if vb_f is None and va_f is None:
                continue
            out[k] = {
                "before": vb_f,
                "after": va_f,
                "delta": (va_f - vb_f) if (vb_f is not None and va_f is not None) else None,
            }
        return out

    return {
        "before_has_chat": bool(fb.get("has_chat_template")),
        "after_has_chat": bool(fa.get("has_chat_template")),
        "anatomy_deltas": _pair_dict(_ANATOMY_DELTA_KEYS),
        "pathology_deltas": _pair_dict(_PATHOLOGY_DELTA_KEYS),
    }


# ----------------------------------------------------------------------------------------------- card assembly & I/O
def build_delta_xray_card(
    *,
    pair_id: str,
    before_repo: str,
    after_repo: str,
    transition_type: str,
    family: str,
    struct_before: _ModelStructure,
    card_before: Mapping[str, Any],
    card_after: Mapping[str, Any],
    weight_diff: Mapping[str, Any],
    spectral_diff: Mapping[str, Any],
    domain_shifts: Mapping[str, Any],
    attribution: Mapping[str, Any] | None,
    probe_pack_sha256: str,
    wall_s: float,
    wsi_commit: str | None = None,
) -> dict[str, Any]:
    """Assemble and sanitize a compact `.delta.xray.json` card."""
    id_b = card_before.get("identity", {}) if isinstance(card_before, Mapping) else {}
    card = {
        "schema": DELTA_XRAY_SCHEMA_VERSION,
        "probe_pack_sha256": str(probe_pack_sha256)[:16],
        "pair": {
            "pair_id": str(pair_id),
            "before_repo": str(before_repo),
            "after_repo": str(after_repo),
            "transition_type": str(transition_type),
            "family": str(family),
            "arch_family": id_b.get("arch_family") or struct_before.identity.get("arch_family"),
            "n_layers": int(struct_before.identity.get("n_layers", len(struct_before.blocks))),
            "params_b": id_b.get("params_b") or struct_before.identity.get("params_b"),
        },
        "weight_diff": dict(weight_diff),
        "spectral_diff": dict(spectral_diff),
        "domain_shifts": dict(domain_shifts),
        "attribution": dict(attribution) if attribution is not None else {"status": "skipped"},
        "xray_delta": compute_xray_card_delta(card_before, card_after),
        "scan_meta": {
            "wall_s": round(float(wall_s), 1),
            "wsi_commit": wsi_commit,
        },
    }
    return _sanitize(card)


def format_compact_delta_xray_json(card: dict[str, Any]) -> str:
    """Serialize a Delta X-Ray chart using the same compact formatter as Tome 1."""
    return format_compact_xray_json(card)


def save_delta_xray_json(card: dict[str, Any], path: str | Path) -> Path:
    """Atomically write a `.delta.xray.json` chart to disk."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(format_compact_delta_xray_json(card), encoding="utf-8")
    os.replace(tmp, dst)
    return dst


def load_delta_xray_json(path: str | Path) -> dict[str, Any]:
    """Load a `.delta.xray.json` chart from disk."""
    return load_xray_json(path)


def scan_delta_xray(
    model_before: nn.Module,
    model_after: nn.Module,
    tok_before: Any,
    tok_after: Any | None = None,
    *,
    pair_id: str | None = None,
    before_repo: str | None = None,
    after_repo: str | None = None,
    transition_type: str = "custom",
    family: str = "custom",
    probe_pack: dict[str, Any] | None = None,
    card_before: dict[str, Any] | None = None,
    card_after: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 16,
    loop_max_new: int = 384,
    skip_generation: bool = False,
    with_attribution: bool = True,
    grad_seq_max_len: int = 128,
    wsi_commit: str | None = None,
) -> dict[str, Any]:
    """High-level Delta X-Ray scan comparing two in-memory checkpoints `(model_before -> model_after)`."""
    import time

    t0 = time.perf_counter()
    dev = _resolve_device(model_before, device)
    pack = probe_pack if probe_pack is not None else build_frozen_probe_pack(use_hf_datasets=False)
    tk_b = tok_before
    tk_a = tok_after if tok_after is not None else tok_before
    struct_b = _discover_xray_structure(model_before, tk_b, repo_id=before_repo)
    n_layers = len(struct_b.blocks)

    cb = card_before or scan_model_xray(
        model_before,
        tk_b,
        repo_id=before_repo,
        probe_pack=pack,
        device=dev,
        batch_size=batch_size,
        loop_max_new=loop_max_new,
        skip_generation=skip_generation,
        wsi_commit=wsi_commit,
    )
    ca = card_after or scan_model_xray(
        model_after,
        tk_a,
        repo_id=after_repo,
        probe_pack=pack,
        device=dev,
        batch_size=batch_size,
        loop_max_new=loop_max_new,
        skip_generation=skip_generation,
        wsi_commit=wsi_commit,
    )

    weight_diff, spectral_diff = compute_weight_and_spectral_diff(model_before, model_after, struct_b, device=dev)

    dom_seqs = build_domain_calibration_seqs(tk_b, pack, max_length=256)
    st_b = capture_domain_probe_state(model_before, dom_seqs, device=dev)
    domain_shifts = compare_domain_probe_states(st_b, model_after, dom_seqs, device=dev)

    attr: dict[str, Any] | None = None
    if with_attribution:
        dots_b, gnll_b, pf_cost, used_len = accumulate_domain_grad_dots(
            model_before, model_after, dom_seqs, sign=+1.0, device=dev, max_len=grad_seq_max_len, compute_prose_fisher=True
        )
        dots_a, gnll_a, _, _ = accumulate_domain_grad_dots(
            model_after, model_before, dom_seqs, sign=-1.0, device=dev, max_len=used_len, compute_prose_fisher=False
        )
        attr = assemble_delta_attribution(dots_b, dots_a, gnll_b, gnll_a, pf_cost, n_layers, used_len)

    b_repo = str(before_repo or cb.get("identity", {}).get("repo_id") or "before")
    a_repo = str(after_repo or ca.get("identity", {}).get("repo_id") or "after")
    pid = pair_id or f"{b_repo.split('/')[-1]}__to__{a_repo.split('/')[-1]}"

    return build_delta_xray_card(
        pair_id=pid,
        before_repo=b_repo,
        after_repo=a_repo,
        transition_type=transition_type,
        family=family,
        struct_before=struct_b,
        card_before=cb,
        card_after=ca,
        weight_diff=weight_diff,
        spectral_diff=spectral_diff,
        domain_shifts=domain_shifts,
        attribution=attr,
        probe_pack_sha256=str(pack.get("probe_pack_sha256", "")),
        wall_s=time.perf_counter() - t0,
        wsi_commit=wsi_commit,
    )


# ----------------------------------------------------------------------------------------------- Atlas aggregator
def _extract_flat_delta_row(card: Mapping[str, Any], byte_size: int | None = None) -> dict[str, Any]:
    pair = card.get("pair", {})
    wd = card.get("weight_diff", {})
    sd = card.get("spectral_diff", {})
    sd_mlp = sd.get("mlp_out", {})
    sd_mix = sd.get("mixer_out", {})
    ds = card.get("domain_shifts", {})
    roles = wd.get("role_energy_share", {})
    attr_d = (card.get("attribution") or {}).get("domains", {})
    xd = card.get("xray_delta", {})
    pd = xd.get("pathology_deltas", {})

    mlp_share = sum(float(roles.get(r, 0.0)) for r in MLP_ROLES) if roles else None
    mixer_share = sum(float(roles.get(r, 0.0)) for r in MIXER_ROLES) if roles else None

    return {
        "pair_id": pair.get("pair_id"),
        "family": pair.get("family"),
        "transition_type": pair.get("transition_type"),
        "before_repo": pair.get("before_repo"),
        "after_repo": pair.get("after_repo"),
        "arch_family": pair.get("arch_family"),
        "params_b": pair.get("params_b"),
        "n_layers": pair.get("n_layers"),
        "json_bytes": byte_size,
        "total_rel_fro": wd.get("total_rel_fro"),
        "depth_centroid": wd.get("depth_centroid"),
        "mlp_share": mlp_share,
        "mixer_share": mixer_share,
        "erank_ratio_mlp": sd_mlp.get("erank_ratio_mean"),
        "energy_r8_mlp": sd_mlp.get("energy_r8_mean"),
        "rank80_mlp": sd_mlp.get("rank80_median"),
        "rank90_mlp": sd_mlp.get("rank90_median"),
        "base_ov_ratio_mlp": sd_mlp.get("base_top32_overlap_vs_chance"),
        "erank_ratio_mix": sd_mix.get("erank_ratio_mean"),
        "rank80_mix": sd_mix.get("rank80_median"),
        "dnll_prose": (ds.get("prose") or {}).get("delta_nll"),
        "dnll_math": (ds.get("math") or {}).get("delta_nll"),
        "dnll_code": (ds.get("code") or {}).get("delta_nll"),
        "kl_prose": (ds.get("prose") or {}).get("kl_top64"),
        "faith_prose": (attr_d.get("prose") or {}).get("faithfulness"),
        "faith_math": (attr_d.get("math") or {}).get("faithfulness"),
        "faith_code": (attr_d.get("code") or {}).get("faithfulness"),
        "after_overref_rate": (pd.get("base_overref_rate") or {}).get("after"),
        "delta_overref_rate": (pd.get("base_overref_rate") or {}).get("delta"),
        "after_cave_rate": (pd.get("cave_rate_false_pushback") or {}).get("after"),
        "delta_cave_rate": (pd.get("cave_rate_false_pushback") or {}).get("delta"),
    }


def _finite_median(vals: Iterable[Any]) -> float | None:
    xs = [float(v) for v in vals if _is_finite_num(v)]
    return float(np.median(xs)) if xs else None


def summarize_delta_xray_atlas(atlas_dir: str | Path) -> dict[str, Any]:
    """Aggregate all `*.delta.xray.json` charts in `atlas_dir` into transition summaries & markdown table."""
    root = Path(atlas_dir)
    files = sorted(root.glob("*.delta.xray.json"))
    rows: list[dict[str, Any]] = []
    for f in files:
        card = load_delta_xray_json(f)
        rows.append(_extract_flat_delta_row(card, byte_size=int(f.stat().st_size)))

    by_type: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        t = str(r.get("transition_type") or "unknown")
        groups.setdefault(t, []).append(r)

    for t, grp in sorted(groups.items()):
        by_type[t] = {
            "n": len(grp),
            "median_rel_fro": _finite_median(g.get("total_rel_fro") for g in grp),
            "median_depth_centroid": _finite_median(g.get("depth_centroid") for g in grp),
            "median_mlp_share": _finite_median(g.get("mlp_share") for g in grp),
            "median_rank80_mlp": _finite_median(g.get("rank80_mlp") for g in grp),
            "median_dnll_prose": _finite_median(g.get("dnll_prose") for g in grp),
            "median_dnll_math": _finite_median(g.get("dnll_math") for g in grp),
            "median_dnll_code": _finite_median(g.get("dnll_code") for g in grp),
        }

    def _fmt(v: Any, digits: int = 3) -> str:
        if not _is_finite_num(v):
            return "—"
        return f"{float(v):.{digits}f}"

    def _pct(v: Any) -> str:
        if not _is_finite_num(v):
            return "—"
        return f"{float(v) * 100:.1f}%"

    lines = [
        "| Pair | Type | Size | Bytes | RelFro | Depth | MLP Share | MLP Rank80 | ΔMath | ΔCode | ΔProse | KL Prose | OverRef | Syc Cave |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['pair_id']}` | `{r['transition_type']}` | {_fmt(r['params_b'], 2)}B | {r['json_bytes'] or '—'} | "
            f"{_fmt(r['total_rel_fro'])} | {_fmt(r['depth_centroid'], 2)} | {_pct(r['mlp_share'])} | "
            f"{_fmt(r['rank80_mlp'], 0)} | {_fmt(r['dnll_math'])} | {_fmt(r['dnll_code'])} | {_fmt(r['dnll_prose'])} | "
            f"{_fmt(r['kl_prose'])} | {_pct(r['after_overref_rate'])} | {_pct(r['after_cave_rate'])} |"
        )

    return _sanitize(
        {
            "n_pairs": len(rows),
            "pairs": rows,
            "by_transition_type": by_type,
            "meta": {
                "note": (
                    "Controlled within-checkpoint diffs cancel pretraining and architecture variation. "
                    "Treats depth, role energy, spectral rank, and gradient credit as structural recipes to verify interventionally."
                ),
            },
            "markdown_table": "\n".join(lines),
        }
    )


__all__ = [
    "DELTA_XRAY_SCHEMA_VERSION",
    "accumulate_domain_grad_dots",
    "assemble_delta_attribution",
    "build_delta_xray_card",
    "build_domain_calibration_seqs",
    "capture_domain_probe_state",
    "classify_param_role",
    "compare_domain_probe_states",
    "compute_weight_and_spectral_diff",
    "compute_xray_card_delta",
    "extract_layer_idx",
    "format_compact_delta_xray_json",
    "load_delta_xray_json",
    "save_delta_xray_json",
    "scan_delta_xray",
    "summarize_delta_xray_atlas",
]
