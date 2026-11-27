"""Live surgical telemetry for WSI weight edits (`wsi_surgery_telemetry_v1`).

Four fast, zero-training instruments designed to guide layer selection, alpha calibration,
and collateral-damage avoidance BEFORE running expensive generation sweeps:

1. **Birth/Death Logit-Lens Trace (`trace_birth_death`)**:
   Projects every layer's residual stream through final norm + `lm_head` to track
   decision margin (`logit_target - logit_foil`) and token entropy across depth,
   identifying where a desired behavior is born, peaks, or gets overwritten.

2. **Causal Gain & Hydra Self-Repair Probe (`probe_causal_gain_and_hydra`)**:
   Injects a norm-matched pulse (`rel_eps * ||h_L|| * d_L`) at each candidate layer L
   and measures:
   - `causal_gain`: `||Delta h_final|| / ||Delta h_L||` (amplification vs damping)
   - `hydra_retention`: fraction of the directional push surviving to the final layer
     (`1 - hydra_retention` = downstream self-repair)
   - `margin_leverage`: `Delta(logit_target - logit_foil) / rel_eps` (decision leverage)
   - `suggested_alpha`: layer-specific `alpha` calibrated to a target residual fraction
     (e.g. 8% of `||h_L||`), making edits scale-invariant across norms from `5` to `2500`.

3. **Collision Radar (`collision_radar` & `orthogonalize_against_bank`)**:
   Measures cosine overlap between a candidate edit direction and a bank of protected
   directions (e.g. `harm_veto`, `fact_readout`, `style_opening`, `copy_legit`),
   flagging high-risk collisions before any weight edit is applied.

4. **Unified Site Recommender (`diagnose_surgery_sites`)**:
   Combines all signals into a compact ranked table of candidate edit layers + `alpha`.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fisher import _resolve_device

SURGERY_TELEMETRY_SCHEMA = "wsi_surgery_telemetry_v1"


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the transformer layer list across dense, VLM, and hybrid HF models."""
    for path in (
        "model.layers",
        "model.language_model.layers",
        "model.llm.layers",
        "transformer.h",
        "gpt_neox.layers",
    ):
        cur: Any = model
        ok = True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and isinstance(cur, nn.ModuleList) and len(cur) > 0:
            return cur
    raise ValueError(f"Could not locate decoder layers on {type(model).__name__}")


def _get_final_norm_and_lm_head(model: nn.Module) -> tuple[nn.Module | None, nn.Module]:
    """Locate the final RMSNorm/LayerNorm and output projection (`lm_head`)."""
    norm: nn.Module | None = None
    for path in (
        "model.norm",
        "model.language_model.norm",
        "model.llm.norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
    ):
        cur: Any = model
        ok = True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and isinstance(cur, nn.Module):
            norm = cur
            break

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        for attr in ("lm_head", "embed_out"):
            if hasattr(model, attr):
                lm_head = getattr(model, attr)
                break
    if lm_head is None:
        raise ValueError(f"Could not locate lm_head / output embeddings on {type(model).__name__}")
    return norm, lm_head


def _project_to_logits(
    h: torch.Tensor,
    final_norm: nn.Module | None,
    lm_head: nn.Module,
) -> torch.Tensor:
    """Project hidden state `[..., D]` through `final_norm` + `lm_head` -> `[..., V]`."""
    x = h
    if final_norm is not None:
        p = next(final_norm.parameters(), None)
        if p is not None:
            x = x.to(dtype=p.dtype, device=p.device)
        x = final_norm(x)
    p_head = next(lm_head.parameters(), None)
    if p_head is not None:
        x = x.to(dtype=p_head.dtype, device=p_head.device)
    return lm_head(x).float()


def _tokenize_batch(
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token = tok.eos_token
    prev_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    try:
        enc = tok(list(prompts), return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev_side
    return {k: v.to(device) for k, v in enc.items()}


def _capture_last_token_hidden_states(
    model: nn.Module,
    enc: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """Capture last-token hidden state after each decoder layer (`len = n_layers`), float32."""
    layers = _get_decoder_layers(model)
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    def _make_hook(idx: int):
        def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            captured[idx] = t[:, -1, :].detach().float()
        return _hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_make_hook(i)))
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out_list: list[torch.Tensor] = []
    for i, t in enumerate(captured):
        if t is None:
            raise RuntimeError(f"Layer {i} hook did not fire during forward pass.")
        out_list.append(t)
    return out_list


def trace_birth_death(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    *,
    target_token_ids: Sequence[int],
    foil_token_ids: Sequence[int],
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Track Logit-Lens decision margin & entropy across layers for `prompts`.

    For each layer `L`:
    - `margin[L] = mean(max_t logit[target_t] - max_f logit[foil_f])`
    - `entropy[L] = mean(H(softmax(logits)))`
    - `resid_norm[L] = mean(||h_L||_2)`

    Also summarizes:
    - `birth_layer`: first layer where `margin > 0` (or `None` if never positive)
    - `peak_layer`: layer with maximum `margin`
    - `death_layer`: first layer AFTER `peak_layer` where `margin` flips `<= 0` (if ever)
    - `final_margin` & `margin_fragility` (fraction of prompts with `|final_margin| < 1.0`)
    """
    dev = _resolve_device(model, device)
    model.eval()
    final_norm, lm_head = _get_final_norm_and_lm_head(model)
    enc = _tokenize_batch(tok, prompts, dev)
    hiddens = _capture_last_token_hidden_states(model, enc)

    tgt_idx = torch.tensor(list(target_token_ids), dtype=torch.long, device=dev)
    foil_idx = torch.tensor(list(foil_token_ids), dtype=torch.long, device=dev)

    margins_by_layer: list[float] = []
    entropies_by_layer: list[float] = []
    norms_by_layer: list[float] = []
    final_per_prompt: list[float] = []

    with torch.no_grad():
        for li, h in enumerate(hiddens):
            norms_by_layer.append(float(h.norm(dim=-1).mean().item()))
            logits = _project_to_logits(h, final_norm, lm_head)  # [B, V]
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean().item()
            entropies_by_layer.append(float(ent))

            t_score = logits.index_select(-1, tgt_idx).max(dim=-1).values
            f_score = logits.index_select(-1, foil_idx).max(dim=-1).values
            margin_vec = t_score - f_score
            margins_by_layer.append(float(margin_vec.mean().item()))
            if li == len(hiddens) - 1:
                final_per_prompt = [float(x) for x in margin_vec.cpu().tolist()]

    birth_layer = next((i for i, m in enumerate(margins_by_layer) if m > 0.0), None)
    peak_layer = int(np.argmax(margins_by_layer))
    death_layer = None
    if margins_by_layer[peak_layer] > 0.0:
        for i in range(peak_layer + 1, len(margins_by_layer)):
            if margins_by_layer[i] <= 0.0:
                death_layer = i
                break

    fragile_frac = (
        float(np.mean([abs(m) < 1.0 for m in final_per_prompt]))
        if final_per_prompt
        else 0.0
    )

    return {
        "n_layers": len(hiddens),
        "n_prompts": len(prompts),
        "margin_by_layer": [round(x, 4) for x in margins_by_layer],
        "entropy_by_layer": [round(x, 4) for x in entropies_by_layer],
        "resid_norm_by_layer": [round(x, 4) for x in norms_by_layer],
        "birth_layer": birth_layer,
        "peak_layer": peak_layer,
        "peak_margin": round(margins_by_layer[peak_layer], 4),
        "death_layer": death_layer,
        "final_margin": round(margins_by_layer[-1], 4),
        "final_margin_std": round(float(np.std(final_per_prompt)), 4) if final_per_prompt else 0.0,
        "margin_fragility": round(fragile_frac, 4),
    }


def probe_causal_gain_and_hydra(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    directions_by_layer: dict[int, torch.Tensor],
    *,
    target_token_ids: Sequence[int] | None = None,
    foil_token_ids: Sequence[int] | None = None,
    rel_eps: float = 0.05,
    target_rel_edit: float = 0.08,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Measure per-layer causal gain, Hydra self-repair, decision-margin leverage, and scale-matched alpha.

    At each layer `L` in `directions_by_layer`:
    - Injects a norm-matched pulse `delta = (rel_eps * mean_norm[L]) * unit(d_L)` at the last token.
    - Measures downstream `Delta h_final = h_final_perturbed - h_final_base`:
      * `causal_gain`: `||Delta h_final|| / ||delta||`
      * `hydra_retention`: `<Delta h_final, unit(d_final)> / ||delta||`
        (if `d_final` is in `directions_by_layer`, else uses `unit(d_L)`).
      * `self_repair_ratio`: `1.0 - clip(hydra_retention, 0, 1)` (higher = downstream erases your edit)
      * `margin_leverage`: `Delta(logit_target - logit_foil) / rel_eps` (if target/foil IDs given)
      * `suggested_alpha`: `target_rel_edit * mean_norm[L]` (scale-invariant edit strength).
    """
    dev = _resolve_device(model, device)
    model.eval()
    layers = _get_decoder_layers(model)
    n_layers = len(layers)
    final_norm, lm_head = _get_final_norm_and_lm_head(model)
    enc = _tokenize_batch(tok, prompts, dev)

    # Baseline forward
    base_hiddens = _capture_last_token_hidden_states(model, enc)
    base_h_final = base_hiddens[-1]  # [B, D] float32
    base_norms = [float(h.norm(dim=-1).mean().item()) for h in base_hiddens]

    use_margin = bool(target_token_ids and foil_token_ids)
    tgt_idx = torch.tensor(list(target_token_ids), dtype=torch.long, device=dev) if use_margin else None
    foil_idx = torch.tensor(list(foil_token_ids), dtype=torch.long, device=dev) if use_margin else None

    base_margin = 0.0
    if use_margin and tgt_idx is not None and foil_idx is not None:
        with torch.no_grad():
            base_logits = _project_to_logits(base_h_final, final_norm, lm_head)
            base_margin = float(
                (
                    base_logits.index_select(-1, tgt_idx).max(dim=-1).values
                    - base_logits.index_select(-1, foil_idx).max(dim=-1).values
                ).mean().item()
            )

    last_dir_key = max(directions_by_layer.keys())
    d_ref_final = directions_by_layer[last_dir_key].detach().float().to(dev)
    d_ref_final = d_ref_final / d_ref_final.norm().clamp_min(1e-12)

    per_layer: dict[int, dict[str, float]] = {}
    for li in sorted(directions_by_layer.keys()):
        if not (0 <= li < n_layers):
            continue
        d = directions_by_layer[li].detach().float().to(dev)
        d_unit = d / d.norm().clamp_min(1e-12)
        pulse_mag = max(rel_eps * base_norms[li], 1e-6)
        pulse = pulse_mag * d_unit  # [D]

        captured_final: list[torch.Tensor] = []

        def _inject_hook(_mod: nn.Module, _inp: Any, out: Any):
            if isinstance(out, tuple):
                h0 = out[0].clone()
                h0[:, -1, :] = h0[:, -1, :] + pulse.to(dtype=h0.dtype, device=h0.device)
                return (h0,) + out[1:]
            h0 = out.clone()
            h0[:, -1, :] = h0[:, -1, :] + pulse.to(dtype=h0.dtype, device=h0.device)
            return h0

        def _read_final(_mod: nn.Module, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            captured_final.append(t[:, -1, :].detach().float())

        h_inj = layers[li].register_forward_hook(_inject_hook)
        h_fin = layers[-1].register_forward_hook(_read_final)
        try:
            with torch.no_grad():
                model(**enc, use_cache=False)
        finally:
            h_inj.remove()
            h_fin.remove()

        dh_final = captured_final[0] - base_h_final  # [B, D]
        causal_gain = float(dh_final.norm(dim=-1).mean().item() / pulse_mag)
        # Measure retention along the final-layer reference direction (or same-layer direction)
        proj = float((dh_final @ d_ref_final).mean().item() / pulse_mag)
        self_repair = float(1.0 - max(0.0, min(1.0, proj)))

        margin_lev = 0.0
        if use_margin and tgt_idx is not None and foil_idx is not None:
            with torch.no_grad():
                pert_logits = _project_to_logits(captured_final[0], final_norm, lm_head)
                pert_margin = float(
                    (
                        pert_logits.index_select(-1, tgt_idx).max(dim=-1).values
                        - pert_logits.index_select(-1, foil_idx).max(dim=-1).values
                    ).mean().item()
                )
            margin_lev = float((pert_margin - base_margin) / max(rel_eps, 1e-6))

        per_layer[li] = {
            "resid_norm": round(base_norms[li], 4),
            "suggested_alpha": round(target_rel_edit * base_norms[li], 4),
            "causal_gain": round(causal_gain, 4),
            "hydra_retention": round(proj, 4),
            "self_repair_ratio": round(self_repair, 4),
            "margin_leverage": round(margin_lev, 4),
        }

    return {
        "n_layers": n_layers,
        "rel_eps": rel_eps,
        "target_rel_edit": target_rel_edit,
        "base_final_margin": round(base_margin, 4),
        "per_layer": per_layer,
    }


def collision_radar(
    edit_dir: torch.Tensor,
    protected_bank: dict[str, torch.Tensor],
    *,
    warn_threshold: float = 0.25,
) -> dict[str, Any]:
    """Check cosine overlap between a candidate `edit_dir` and a bank of protected directions."""
    e = edit_dir.detach().float().reshape(-1)
    e = e / e.norm().clamp_min(1e-12)
    overlaps: dict[str, float] = {}
    warnings: list[str] = []
    max_abs = 0.0
    max_name: str | None = None

    for name, pvec in protected_bank.items():
        p = pvec.detach().float().reshape(-1).to(e.device)
        p = p / p.norm().clamp_min(1e-12)
        cos = float(torch.dot(e, p).item())
        overlaps[name] = round(cos, 4)
        if abs(cos) > max_abs:
            max_abs = abs(cos)
            max_name = name
        if abs(cos) >= warn_threshold:
            warnings.append(f"{name} (|cos|={abs(cos):.3f})")

    return {
        "overlaps": overlaps,
        "max_abs_cos": round(max_abs, 4),
        "max_collision_name": max_name,
        "warnings": warnings,
        "safe": len(warnings) == 0,
    }


def orthogonalize_against_bank(
    edit_dir: torch.Tensor,
    protected_bank: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, float]:
    """Project `edit_dir` into the orthogonal complement of `protected_bank` via QR.

    Returns `(cleaned_unit_dir, retained_norm_frac)` where `retained_norm_frac` is the
    fraction of the original direction surviving orthogonalization (low = severe collision).
    """
    e = edit_dir.detach().float().reshape(-1)
    e = e / e.norm().clamp_min(1e-12)
    if not protected_bank:
        return e, 1.0
    cols = []
    for pvec in protected_bank.values():
        p = pvec.detach().float().reshape(-1).to(e.device)
        n = p.norm()
        if n > 1e-12:
            cols.append(p / n)
    if not cols:
        return e, 1.0
    mat = torch.stack(cols, dim=1)  # [D, K]
    q, _ = torch.linalg.qr(mat, mode="reduced")
    proj = q @ (q.T @ e)
    cleaned = e - proj
    retained = float(cleaned.norm().item())
    if retained < 1e-8:
        return e, 0.0
    return cleaned / retained, round(retained, 4)


def extract_contrastive_directions(
    model: nn.Module,
    tok: Any,
    pos_prompts: Sequence[str],
    neg_prompts: Sequence[str],
    *,
    device: str | torch.device | None = None,
) -> dict[int, torch.Tensor]:
    """Compute per-layer mean-difference unit directions `mean(h_pos) - mean(h_neg)`."""
    dev = _resolve_device(model, device)
    model.eval()
    h_pos = _capture_last_token_hidden_states(model, _tokenize_batch(tok, pos_prompts, dev))
    h_neg = _capture_last_token_hidden_states(model, _tokenize_batch(tok, neg_prompts, dev))
    dirs: dict[int, torch.Tensor] = {}
    for i, (hp, hn) in enumerate(zip(h_pos, h_neg)):
        diff = hp.mean(dim=0) - hn.mean(dim=0)
        dirs[i] = diff / diff.norm().clamp_min(1e-12)
    return dirs


def diagnose_surgery_sites(
    model: nn.Module,
    tok: Any,
    prompts: Sequence[str],
    directions_by_layer: dict[int, torch.Tensor],
    *,
    target_token_ids: Sequence[int],
    foil_token_ids: Sequence[int],
    protected_banks_by_layer: dict[int, dict[str, torch.Tensor]] | None = None,
    rel_eps: float = 0.05,
    target_rel_edit: float = 0.08,
    collision_penalty: float = 0.5,
    top_k: int = 5,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run all 4 surgical telemetry instruments and rank the best candidate layers + alphas.

    Composite `surgery_score(L)` rewards:
    - Positive `margin_leverage` (or `causal_gain * max(hydra_retention, 0)` when leverage ~ 0)
    - High `hydra_retention` (low downstream self-repair)
    - Low collision with `protected_banks_by_layer[L]`
    """
    bd = trace_birth_death(
        model,
        tok,
        prompts,
        target_token_ids=target_token_ids,
        foil_token_ids=foil_token_ids,
        device=device,
    )
    cg = probe_causal_gain_and_hydra(
        model,
        tok,
        prompts,
        directions_by_layer,
        target_token_ids=target_token_ids,
        foil_token_ids=foil_token_ids,
        rel_eps=rel_eps,
        target_rel_edit=target_rel_edit,
        device=device,
    )

    ranked: list[dict[str, Any]] = []
    for li, stats in cg["per_layer"].items():
        radar = None
        max_col = 0.0
        if protected_banks_by_layer and li in protected_banks_by_layer:
            radar = collision_radar(directions_by_layer[li], protected_banks_by_layer[li])
            max_col = float(radar["max_abs_cos"])

        # Effective directional amplification (causal_gain * hydra_retention) tracks real
        # multi-token recovery far better than token-0 margin_leverage (LFM3 rho +0.90 vs +0.10).
        gain = float(stats["causal_gain"])
        ret = max(0.0, float(stats["hydra_retention"]))
        eff_gain = gain * ret
        score = eff_gain * max(0.0, 1.0 - collision_penalty * max_col)
        entry: dict[str, Any] = {
            "layer": li,
            "surgery_score": round(score, 4),
            "effective_gain": round(eff_gain, 4),
            **stats,
            "max_collision_cos": round(max_col, 4),
        }
        if radar is not None:
            entry["collision_warnings"] = radar["warnings"]
        ranked.append(entry)

    ranked.sort(key=lambda r: r["surgery_score"], reverse=True)
    return {
        "schema": SURGERY_TELEMETRY_SCHEMA,
        "birth_death": bd,
        "causal_probe": cg,
        "top_sites": ranked[:top_k],
        "all_sites": ranked,
    }
