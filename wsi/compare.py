"""Model Comparison, Fine-Tune Diffing, and Cross-Architecture Semantic Circuit Alignment for WSI.

Provides three core capabilities:
1. `diff_checkpoints(base_idx, tuned_idx)`:
   For two checkpoints of the same architecture/family (e.g. Base vs Instruct/Fine-Tune/Edited).
   Finds which layers, modules, and neurons changed the most and explains what semantic concepts
   and `[Trigger] -> [Output]` transductions were added, amplified, or suppressed.
2. `align_semantic_circuits(idx_a, idx_b, concepts=...)`:
   For two completely different models/architectures (e.g. Llama vs Qwen vs GPT-2 with different
   hidden sizes, layer counts, and random seeds). Uses the vocabulary projections (`E_in`, `U_eff`)
   as a universal semantic coordinate system ("Platonic Representation anchor") to match functionally
   equivalent neurons and circuits across architectures.
3. `transfer_transduction_edit(src_idx, dst_idx, input_query, old_output_query, new_output_query)`:
   Finds a functional circuit `[input_query] -> [old_output_query]` in `src_idx`, locates its
   functional counterpart in `dst_idx` (even across different architectures), and synthesizes a
   surgical edit in `dst_idx` to suppress `old_output_query` / promote `new_output_query`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .indexer import _to_residual_columns, get_mlp_input_keys

if TYPE_CHECKING:
    from .query import WeightSpaceIndex


def diff_checkpoints(
    base_idx: "WeightSpaceIndex",
    target_idx: "WeightSpaceIndex",
    *,
    top_layers: int = 6,
    top_neurons: int = 8,
    top_k_tokens: int = 6,
) -> dict[str, Any]:
    """Compare two checkpoints of the same architecture (e.g., Base vs Fine-Tuned or Edited model).

    Returns a layer-by-layer and neuron-by-neuron semantic diff explaining what changed in weight space.
    """
    if base_idx.anat.hidden != target_idx.anat.hidden or base_idx.anat.n_layers != target_idx.anat.n_layers:
        raise ValueError(
            f"Checkpoints have different architectures ({base_idx.anat.summary()} vs {target_idx.anat.summary()}). "
            "Use `align_semantic_circuits()` to compare models across different architectures."
        )

    layer_diffs: list[dict[str, Any]] = []
    changed_neurons: list[dict[str, Any]] = []

    for l in sorted(base_idx.anat.layers):
        mod_diffs: dict[str, Any] = {}
        layer_delta_sq = 0.0
        layer_base_sq = 0.0

        for mod_role, tname in [("attn_out", base_idx.anat.attn_out(l)), ("mlp_out", base_idx.anat.mlp_out(l))]:
            if not tname or tname not in target_idx.ck:
                continue
            W_base = base_idx.ck.get(tname).float()
            W_targ = target_idx.ck.get(tname).float()
            if W_base.shape != W_targ.shape:
                continue

            out_ax = base_idx.anat.out_axis(tname)
            W_base_res = _to_residual_columns(W_base, out_ax)
            W_targ_res = _to_residual_columns(W_targ, out_ax)
            dW_res = W_targ_res - W_base_res

            d_norm = dW_res.norm().item()
            b_norm = W_base_res.norm().item()
            layer_delta_sq += d_norm ** 2
            layer_base_sq += b_norm ** 2
            rel_norm = d_norm / max(b_norm, 1e-12)

            if d_norm < 1e-9:
                mod_diffs[mod_role] = {"tensor": tname, "rel_norm": 0.0, "status": "unchanged"}
                continue

            # Extract dominant semantic shift direction via SVD of dW_res
            U_d, S_d, _ = torch.linalg.svd(dW_res, full_matrices=False)
            u0 = U_d[:, 0]
            old_e = (u0 @ W_base_res).norm().item()
            new_e = (u0 @ W_targ_res).norm().item()
            # Orient u0 toward the newly amplified pole if energy increased, or describe the delta vector
            top_delta_vec = u0 * S_d[0]
            dir_exp = target_idx.explain_vector(top_delta_vec, top_k=top_k_tokens)

            mod_diffs[mod_role] = {
                "tensor": tname,
                "delta_fro_norm": round(d_norm, 5),
                "rel_norm": round(rel_norm, 5),
                "rank1_energy_fraction": round((S_d[0].item() ** 2) / max(d_norm ** 2, 1e-12), 4),
                "subspace_action": (
                    "projected_out_or_suppressed"
                    if new_e < old_e * 0.95
                    else ("amplified_or_injected" if new_e > old_e * 1.05 else "rotated_or_shifted")
                ),
                "dominant_shifted_tokens": {
                    "pole_a": [t["token"] for t in dir_exp["promoted_tokens"][:4]],
                    "pole_b": [t["token"] for t in dir_exp["suppressed_tokens"][:4]],
                },
            }

            # For MLP layers, also find the individual neurons whose output or input keys changed the most
            if mod_role == "mlp_out":
                col_deltas = dW_res.norm(dim=0)  # [n_units]
                K_base, _ = get_mlp_input_keys(base_idx.ck, base_idx.anat, l, W_base_res.shape[1])
                K_targ, _ = get_mlp_input_keys(target_idx.ck, target_idx.anat, l, W_targ_res.shape[1])
                if K_base is not None and K_targ is not None and K_base.shape == K_targ.shape:
                    key_deltas = (K_targ - K_base).norm(dim=0)
                else:
                    key_deltas = torch.zeros_like(col_deltas)

                total_unit_delta = (col_deltas.pow(2) + key_deltas.pow(2)).sqrt()
                k_u = min(4, total_unit_delta.numel())
                u_vals, u_idx = torch.topk(total_unit_delta, k=k_u)
                for uv, idx in zip(u_vals, u_idx):
                    if uv.item() < 1e-7:
                        continue
                    uid = int(idx.item())
                    before_info = base_idx.explain_neuron(l, uid, top_k=3)
                    after_info = target_idx.explain_neuron(l, uid, top_k=3)
                    changed_neurons.append({
                        "layer": l,
                        "unit": uid,
                        "total_delta_norm": round(uv.item(), 5),
                        "out_delta_norm": round(col_deltas[uid].item(), 5),
                        "key_delta_norm": round(key_deltas[uid].item(), 5),
                        "before_transduction": before_info.get("transduction_rule"),
                        "after_transduction": after_info.get("transduction_rule"),
                    })

        layer_rel = (layer_delta_sq ** 0.5) / max(layer_base_sq ** 0.5, 1e-12)
        layer_diffs.append({
            "layer": l,
            "relative_depth": round(l / max(1, base_idx.anat.n_layers - 1), 3),
            "layer_rel_norm": round(layer_rel, 6),
            "modules": mod_diffs,
        })

    layer_diffs_sorted = sorted(layer_diffs, key=lambda x: x["layer_rel_norm"], reverse=True)
    changed_neurons.sort(key=lambda x: x["total_delta_norm"], reverse=True)

    return {
        "comparison_type": "same_architecture_diff",
        "base_model": str(base_idx.model_dir),
        "target_model": str(target_idx.model_dir),
        "most_changed_layers": layer_diffs_sorted[:top_layers],
        "most_changed_neurons": changed_neurons[:top_neurons],
    }


def align_semantic_circuits(
    idx_a: "WeightSpaceIndex",
    idx_b: "WeightSpaceIndex",
    *,
    transduction_pairs: list[tuple[str, str]] | None = None,
    concepts: list[str] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Match functionally equivalent circuits across two DIFFERENT models/architectures (`idx_a` vs `idx_b`).

    Even when `idx_a` and `idx_b` have different `hidden_size`, different `n_layers`, and permuted neuron
    indices, their vocabulary embeddings (`E_in`, `U_eff`) act as a shared semantic coordinate system.
    """
    n_a = max(1, idx_a.anat.n_layers - 1)
    n_b = max(1, idx_b.anat.n_layers - 1)

    matched_transductions: list[dict[str, Any]] = []
    for in_q, out_q in transduction_pairs or []:
        try:
            res_a = idx_a.search_transduction(in_q, out_q, top_k=top_k)
            res_b = idx_b.search_transduction(in_q, out_q, top_k=top_k)
        except ValueError:
            continue

        top_a = res_a["top_transducing_neurons"][0] if res_a["top_transducing_neurons"] else None
        top_b = res_b["top_transducing_neurons"][0] if res_b["top_transducing_neurons"] else None
        if top_a and top_b:
            depth_a = round(top_a["layer"] / n_a, 3)
            depth_b = round(top_b["layer"] / n_b, 3)
            matched_transductions.append({
                "semantic_circuit": f"[{in_q}] -> [{out_q}]",
                "model_a_coord": {
                    "layer": top_a["layer"],
                    "relative_depth": depth_a,
                    "unit": top_a["unit"],
                    "in_tensor": top_a["in_tensor"],
                    "out_tensor": top_a["out_tensor"],
                    "joint_score": top_a["joint_transduction_score"],
                },
                "model_b_coord": {
                    "layer": top_b["layer"],
                    "relative_depth": depth_b,
                    "unit": top_b["unit"],
                    "in_tensor": top_b["in_tensor"],
                    "out_tensor": top_b["out_tensor"],
                    "joint_score": top_b["joint_transduction_score"],
                },
                "relative_depth_gap": round(abs(depth_a - depth_b), 3),
            })

    matched_concepts: list[dict[str, Any]] = []
    for c_q in concepts or []:
        sa = idx_a.search_token(c_q, top_layers=2, top_units_per_layer=2)
        sb = idx_b.search_token(c_q, top_layers=2, top_units_per_layer=2)
        if not sa.get("top_promoting_neurons") or not sb.get("top_promoting_neurons"):
            continue
        pa = sa["top_promoting_neurons"][0]
        pb = sb["top_promoting_neurons"][0]
        matched_concepts.append({
            "concept": c_q,
            "model_a_top_writer": {
                "layer": pa["layer"],
                "relative_depth": round(pa["layer"] / n_a, 3),
                "tensor": pa["tensor"],
                "unit": pa["unit"],
                "logit_contribution": pa["logit_contribution"],
            },
            "model_b_top_writer": {
                "layer": pb["layer"],
                "relative_depth": round(pb["layer"] / n_b, 3),
                "tensor": pb["tensor"],
                "unit": pb["unit"],
                "logit_contribution": pb["logit_contribution"],
            },
        })

    return {
        "comparison_type": "cross_architecture_semantic_alignment",
        "model_a_anatomy": {
            "model_type": idx_a.anat.model_type,
            "layers": idx_a.anat.n_layers,
            "hidden": idx_a.anat.hidden,
        },
        "model_b_anatomy": {
            "model_type": idx_b.anat.model_type,
            "layers": idx_b.anat.n_layers,
            "hidden": idx_b.anat.hidden,
        },
        "matched_transduction_circuits": matched_transductions,
        "matched_concept_writers": matched_concepts,
    }


def decompile_edit_intent(
    src_idx: "WeightSpaceIndex",
    edit_source: Any,
) -> dict[str, Any]:
    """Decompile a raw weight edit (`edit.safetensors` or ops) on `src_idx` into a model-agnostic
    semantic specification (`trigger_token`, `old_output_token`, `new_output_token`, `edit_kind`, `relative_layers`).

    Works purely by reading `src_idx`'s weights and the edit delta—no text description needed!
    """
    from wsr.ops import apply_ops, load_submission
    from pathlib import Path

    if isinstance(edit_source, (str, Path)):
        full = load_submission(src_idx.ck, edit_source, src_idx.anat)
    elif isinstance(edit_source, list):
        full = apply_ops(src_idx.ck, {"ops": edit_source}, src_idx.anat)[0]
    elif isinstance(edit_source, dict) and "ops" in edit_source:
        full = apply_ops(src_idx.ck, edit_source, src_idx.anat)[0]
    elif isinstance(edit_source, dict):
        full = edit_source
    else:
        raise TypeError(f"Unsupported edit_source: {type(edit_source)}")

    # Find the most strongly modified MLP or residual matrix in `full`
    best_unit_record: dict[str, Any] | None = None
    best_delta_norm = -1.0
    relative_layers: list[float] = []
    n_max_l = max(1, src_idx.anat.n_layers - 1)

    for l in sorted(src_idx.anat.layers):
        tname = src_idx.anat.mlp_out(l)
        if not tname or tname not in full:
            continue
        W_old = _to_residual_columns(src_idx.ck.get(tname).float(), src_idx.anat.out_axis(tname))
        W_new = _to_residual_columns(full[tname].float(), src_idx.anat.out_axis(tname))
        dW = W_new - W_old
        if dW.norm().item() < 1e-9:
            continue
        relative_layers.append(round(l / n_max_l, 3))

        col_deltas = dW.norm(dim=0)  # [n_units]
        max_col_delta, uid_t = col_deltas.max(dim=0)
        uid = int(uid_t.item())

        if max_col_delta.item() > best_delta_norm:
            best_delta_norm = max_col_delta.item()
            w_old_u = W_old[:, uid]
            w_new_u = W_new[:, uid]
            dw_u = w_new_u - w_old_u

            # 1. What input token triggers this neuron in Model A?
            K_eff, _ = get_mlp_input_keys(src_idx.ck, src_idx.anat, l, W_old.shape[1])
            if K_eff is not None and uid < K_eff.shape[1]:
                trig_scores = src_idx.E_in @ K_eff[:, uid]
                trig_id = int(trig_scores.argmax().item())
                trig_tok = src_idx.vocab_map[trig_id]
            else:
                trig_id, trig_tok = -1, ""

            # 2. What output token did this neuron originally promote in Model A?
            old_logits = src_idx.U_eff @ w_old_u
            old_id = int(old_logits.argmax().item())
            old_tok = src_idx.vocab_map[old_id]

            # 3. What new output token does the edited neuron (or positive delta) promote in Model A?
            new_logits = src_idx.U_eff @ w_new_u
            new_id = int(new_logits.argmax().item())
            new_tok = src_idx.vocab_map[new_id]

            # Check if this was a pure suppression/orth edit (where w_new norm along old direction vanished)
            u_old_vec = src_idx.U_eff[old_id]
            u_old_hat = u_old_vec / u_old_vec.norm().clamp_min(1e-12)
            old_proj_before = abs(torch.dot(w_old_u, u_old_hat).item())
            old_proj_after = abs(torch.dot(w_new_u, u_old_hat).item())
            is_rewire = (new_id != old_id) and (w_new_u.norm().item() >= w_old_u.norm().item() * 0.7)

            best_unit_record = {
                "src_layer": l,
                "src_relative_depth": round(l / n_max_l, 3),
                "src_unit": uid,
                "src_tensor": tname,
                "edit_kind": "rewire_transduction" if is_rewire else "orth_suppress",
                "trigger_token": {"token_id": trig_id, "token": trig_tok},
                "old_output_token": {"token_id": old_id, "token": old_tok},
                "new_output_token": {"token_id": new_id, "token": new_tok},
                "old_projection_before": round(old_proj_before, 5),
                "old_projection_after": round(old_proj_after, 5),
                "relative_layers_touched": relative_layers,
            }

    if best_unit_record is None:
        raise ValueError("Could not decompile a semantic MLP edit from the provided edit_source.")
    return best_unit_record


def transfer_edit(
    src_idx: "WeightSpaceIndex",
    dst_idx: "WeightSpaceIndex",
    edit_source: Any,
    *,
    top_k_neurons: int = 8,
    strength: float = 2.5,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Automatically decompile a weight edit from `src_idx` (Model A) and synthesize + apply the
    functionally equivalent weight edit on `dst_idx` (Model B), even across different architectures!"""
    decompiled = decompile_edit_intent(src_idx, edit_source)
    trig_tok = decompiled["trigger_token"]["token"]
    old_tok = decompiled["old_output_token"]["token"]
    new_tok = decompiled["new_output_token"]["token"]

    alignment = align_semantic_circuits(
        src_idx,
        dst_idx,
        transduction_pairs=[(trig_tok, old_tok)],
        top_k=top_k_neurons,
    )
    dst_rewire = dst_idx.rewire_transduction(
        trig_tok,
        old_tok,
        new_tok,
        top_k_neurons=top_k_neurons,
        strength=strength,
        save_path=save_path,
    )
    return {
        "decompiled_from_model_a": decompiled,
        "cross_model_circuit_alignment": alignment["matched_transduction_circuits"],
        "synthesized_edit_on_model_b": {k: v for k, v in dst_rewire.items() if k != "_tensors"},
        "_tensors": dst_rewire["_tensors"],
    }


def transfer_transduction_edit(
    src_idx: "WeightSpaceIndex",
    dst_idx: "WeightSpaceIndex",
    input_query: str | int,
    old_output_query: str | int,
    new_output_query: str | int,
    *,
    top_k_neurons: int = 8,
    strength: float = 2.5,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Transfer a semantic transduction edit (`[input] -> [old_output]` => `[input] -> [new_output]`)
    from `src_idx` (Model A) to `dst_idx` (Model B), even across completely different architectures!"""
    alignment = align_semantic_circuits(
        src_idx,
        dst_idx,
        transduction_pairs=[(str(input_query), str(old_output_query))],
        top_k=top_k_neurons,
    )
    dst_rewire = dst_idx.rewire_transduction(
        input_query,
        old_output_query,
        new_output_query,
        top_k_neurons=top_k_neurons,
        strength=strength,
        save_path=save_path,
    )
    return {
        "alignment": alignment["matched_transduction_circuits"],
        "dst_edit": {k: v for k, v in dst_rewire.items() if k != "_tensors"},
        "_tensors": dst_rewire["_tensors"],
    }
