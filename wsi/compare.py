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

from .indexer import (
    PRE_MLP_NORM_KEYS,
    _to_residual_columns,
    discover_layer_readers,
    get_mlp_input_keys,
)

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


_ROSETTA_CACHE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]] = {}


def _get_shared_vocab_rosetta(
    teacher_idx: "WeightSpaceIndex",
    student_idx: "WeightSpaceIndex",
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """Build and cache the shared vocabulary anchor matrices between `teacher_idx` and `student_idx`."""
    key = (id(teacher_idx), id(student_idx))
    if key in _ROSETTA_CACHE:
        return _ROSETTA_CACHE[key]

    t_map: dict[str, int] = {}
    for tid, tok in teacher_idx.vocab_map.items():
        if tok not in t_map:
            t_map[tok] = tid

    shared_t: list[int] = []
    shared_s: list[int] = []
    shared_tokens: list[str] = []
    for s_tid, tok in student_idx.vocab_map.items():
        if tok in t_map:
            shared_t.append(t_map[tok])
            shared_s.append(s_tid)
            shared_tokens.append(tok)

    if not shared_tokens:
        raise ValueError("No shared surface vocabulary tokens found between teacher and student models.")

    I_t = torch.tensor(shared_t, dtype=torch.long)
    I_s = torch.tensor(shared_s, dtype=torch.long)
    U_t_norm = (teacher_idx.U_eff * teacher_idx.U_eff_inv_norms.unsqueeze(1))[I_t]
    U_s_norm = (student_idx.U_eff * student_idx.U_eff_inv_norms.unsqueeze(1))[I_s]
    res = (I_t, I_s, shared_tokens, U_t_norm, U_s_norm)
    _ROSETTA_CACHE[key] = res
    return res


def translate_vector(
    teacher_idx: "WeightSpaceIndex",
    student_idx: "WeightSpaceIndex",
    vec_teacher: torch.Tensor,
    *,
    top_m: int = 16,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    """Translate a latent vector `vec_teacher` (in R^{d_teacher}) from `teacher_idx` into the latent space
    of `student_idx` (in R^{d_student}) via the Dual-Gram Vocabulary Rosetta Bridge:

        v_student = U_{S, M}^T (U_{S, M} U_{S, M}^T + lambda I_M)^{-1} z_{T, M}

    Preserves the exact multi-token semantic chord of `vec_teacher` across completely different
    hidden dimensions and architectures without training or forward passes.
    """
    _, _, shared_tokens, U_t_norm, U_s_norm = _get_shared_vocab_rosetta(teacher_idx, student_idx)
    v_t = vec_teacher.detach().float().cpu().flatten()
    z_t = U_t_norm @ v_t  # [|V_shared|]

    m = min(top_m, z_t.numel())
    top_vals, top_idx = torch.topk(z_t, k=m, largest=True)
    U_sub = U_s_norm[top_idx]  # [m, d_student]
    G = U_sub @ U_sub.T + ridge * torch.eye(m, dtype=torch.float32)
    alpha = torch.linalg.solve(G, top_vals)
    v_s = U_sub.T @ alpha  # [d_student]

    # Match relative norm scale
    v_s = v_s / v_s.norm().clamp_min(1e-12) * v_t.norm().clamp_min(1e-12)

    teacher_chord = [shared_tokens[i] for i in top_idx[:8].tolist()]
    student_readout = [t["token"] for t in student_idx.explain_vector(v_s, top_k=8)["promoted_tokens"]]
    overlap_5 = len(set(teacher_chord[:5]) & set(student_readout[:5])) / max(1, min(5, len(teacher_chord)))

    return {
        "teacher_dim": int(v_t.numel()),
        "student_dim": int(v_s.numel()),
        "shared_vocab_size": len(shared_tokens),
        "chord_size_m": m,
        "teacher_top_tokens": teacher_chord[:5],
        "student_translated_top_tokens": student_readout[:5],
        "top1_exact_match": teacher_chord[0] == student_readout[0] if teacher_chord and student_readout else False,
        "top5_chord_overlap": round(overlap_5, 4),
        "vector_student": v_s,
    }


def transplant_circuit(
    teacher_idx: "WeightSpaceIndex",
    student_idx: "WeightSpaceIndex",
    *,
    teacher_layer: int,
    teacher_unit: int,
    trigger_query: str | int | None = None,
    old_output_query: str | int | None = None,
    top_k_student_neurons: int = 16,
    top_m_chord: int = 16,
    strength: float = 5.0,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Read a neuron/circuit `(teacher_layer, teacher_unit)` from a larger Teacher model (`teacher_idx`),
    translate its multi-token output vector from R^{d_teacher} into R^{d_student} via the Dual-Gram
    Rosetta Bridge, and surgically graft it into the matching input-triggered neurons of the smaller
    Student model (`student_idx`).
    """
    t_out_name = teacher_idx.anat.mlp_out(teacher_layer)
    if not t_out_name:
        raise KeyError(f"Teacher model has no mlp_out in layer {teacher_layer}")
    W_t = _to_residual_columns(teacher_idx.ck.get(t_out_name).float(), teacher_idx.anat.out_axis(t_out_name))
    v_teacher = W_t[:, teacher_unit]

    # Determine trigger token: either explicit `trigger_query` or auto-read from teacher neuron's input keys
    _, _, shared_tokens, _, _ = _get_shared_vocab_rosetta(teacher_idx, student_idx)
    shared_set = set(shared_tokens)
    if trigger_query is None:
        K_t, _ = get_mlp_input_keys(teacher_idx.ck, teacher_idx.anat, teacher_layer, W_t.shape[1])
        if K_t is not None and teacher_unit < K_t.shape[1]:
            trig_scores = teacher_idx.E_in @ K_t[:, teacher_unit]
            for cand_id in torch.topk(trig_scores, k=min(64, trig_scores.numel())).indices.tolist():
                tok_str = teacher_idx.vocab_map[int(cand_id)]
                if tok_str in shared_set:
                    trigger_query = tok_str
                    break
        if trigger_query is None:
            raise ValueError("Could not auto-detect a shared trigger token; pass `trigger_query` explicitly.")

    rosetta = translate_vector(teacher_idx, student_idx, v_teacher, top_m=top_m_chord)
    v_student_hat = rosetta["vector_student"] / rosetta["vector_student"].norm().clamp_min(1e-12)
    target_tok_str = rosetta["teacher_top_tokens"][0]

    trig_matches = student_idx.find_tokens(trigger_query, limit=1)
    if not trig_matches:
        raise ValueError(f"Trigger `{trigger_query}` not found in student vocabulary.")
    s_trig_tok = trig_matches[0]

    u_old_hat: torch.Tensor | None = None
    if old_output_query is not None:
        trans = student_idx.search_transduction(trigger_query, old_output_query, top_k=top_k_student_neurons)
        old_tok = trans["output_token"]
        u_old = student_idx.U_eff[old_tok["token_id"]].float()
        u_old_hat = u_old / u_old.norm().clamp_min(1e-12)
        selected = trans["top_transducing_neurons"]
    else:
        e_trig = student_idx.E_in[s_trig_tok["token_id"]].float()
        candidates: list[dict[str, Any]] = []
        start_l = max(0, student_idx.anat.n_layers // 2)
        for l in sorted(student_idx.anat.layers):
            if l < start_l:
                continue
            s_out_name = student_idx.anat.mlp_out(l)
            if not s_out_name or s_out_name not in student_idx.ck:
                continue
            W_s = _to_residual_columns(student_idx.ck.get(s_out_name).float(), student_idx.anat.out_axis(s_out_name))
            K_s, s_in_name = get_mlp_input_keys(student_idx.ck, student_idx.anat, l, W_s.shape[1])
            if K_s is None:
                continue
            in_scores = e_trig @ K_s  # [n_units]
            top_v, top_u = torch.topk(in_scores, k=min(top_k_student_neurons, in_scores.numel()))
            for sc, uid in zip(top_v.tolist(), top_u.tolist()):
                candidates.append({
                    "layer": l,
                    "unit": int(uid),
                    "in_tensor": s_in_name,
                    "out_tensor": s_out_name,
                    "trigger_score": round(float(sc), 5),
                })

        candidates.sort(key=lambda x: x["trigger_score"], reverse=True)
        selected = candidates[:top_k_student_neurons]

    by_layer: dict[int, list[dict[str, Any]]] = {}
    for item in selected:
        by_layer.setdefault(item["layer"], []).append(item)

    modified_tensors: dict[str, torch.Tensor] = {}
    grafted_records: list[dict[str, Any]] = []

    for l, items in sorted(by_layer.items()):
        tname = student_idx.anat.mlp_out(l)
        if not tname:
            continue
        W_orig = student_idx.ck.get(tname)
        W_mod = W_orig.float().clone()
        out_ax = student_idx.anat.out_axis(tname)

        for item in items:
            uid = item["unit"]
            w_vec = W_mod[:, uid].clone() if out_ax == 0 else W_mod[uid, :].clone()
            orig_norm = w_vec.norm().item()
            w_base = w_vec - torch.dot(w_vec, u_old_hat) * u_old_hat if u_old_hat is not None else w_vec
            # Graft the translated multi-token Teacher vector into the Student neuron
            w_new = w_base + (strength * max(orig_norm, 1e-4)) * v_student_hat
            if out_ax == 0:
                W_mod[:, uid] = w_new
            else:
                W_mod[uid, :] = w_new
            grafted_records.append({
                "student_layer": l,
                "student_unit": uid,
                "tensor": tname,
                "trigger_score": item["trigger_score"],
                "new_promoted_top3": [
                    t["token"] for t in student_idx.explain_vector(w_new, top_k=3)["promoted_tokens"]
                ],
            })

        modified_tensors[tname] = W_mod.to(W_orig.dtype).contiguous()

    if save_path is not None:
        from safetensors.torch import save_file
        from wsr.ops import compact_view
        save_file(compact_view(student_idx.ck, modified_tensors), str(save_path))

    return {
        "teacher_source": {
            "model_type": teacher_idx.anat.model_type,
            "hidden_dim": teacher_idx.anat.hidden,
            "layer": teacher_layer,
            "unit": teacher_unit,
            "top_semantic_chord": rosetta["teacher_top_tokens"],
        },
        "rosetta_translation": {
            "shared_vocab_anchors": rosetta["shared_vocab_size"],
            "chord_size_m": rosetta["chord_size_m"],
            "student_translated_chord": rosetta["student_translated_top_tokens"],
            "top1_exact_match": rosetta["top1_exact_match"],
            "top5_chord_overlap": rosetta["top5_chord_overlap"],
        },
        "student_graft": {
            "model_type": student_idx.anat.model_type,
            "hidden_dim": student_idx.anat.hidden,
            "trigger_token": s_trig_tok["token"],
            "dominant_transplanted_token": target_tok_str,
            "neurons_grafted_count": len(grafted_records),
            "neurons_grafted": grafted_records,
        },
        "save_path": str(save_path) if save_path else None,
        "_tensors": modified_tensors,
    }


def _apply_student_pre_mlp_norm(
    student_idx: "WeightSpaceIndex", layer: int, X_raw: torch.Tensor
) -> torch.Tensor:
    """Apply Student layer `layer`'s pre-MLP normalization to `X_raw` using weights only."""
    norms = student_idx.anat.norms(layer)
    norm_name = next((norms[k] for k in PRE_MLP_NORM_KEYS if k in norms), None)
    w = student_idx.ck.get(norm_name).float() if norm_name and norm_name in student_idx.ck else None
    if student_idx.anat.norm_label == "LayerNorm":
        X_n = torch.nn.functional.layer_norm(X_raw, (X_raw.shape[-1],))
        if w is not None:
            if w.abs().mean().item() < 0.5:
                w = 1.0 + w
            X_n = X_n * w
        if norm_name and norm_name.endswith(".weight"):
            b_name = norm_name[:-7] + ".bias"
            if b_name in student_idx.ck:
                X_n = X_n + student_idx.ck.get(b_name).float()
        return X_n
    X_n = X_raw * torch.rsqrt(X_raw.pow(2).mean(-1, keepdim=True) + 1e-5)
    if w is not None:
        if w.abs().mean().item() < 0.5:
            w = 1.0 + w
        X_n = X_n * w
    return X_n


def _build_contrastive_gram_key(
    Xs_pool: torch.Tensor,
    pos_idx: torch.Tensor | list[int],
    y_pos: torch.Tensor,
    ctrl_idx: torch.Tensor | list[int],
    *,
    key_norm: float = 8.0,
    ridge: float = 0.05,
    margin: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Solve a Dual-Gram key over centered/normalized Student vectors and orthogonalize against control mean."""
    mu_s = Xs_pool[ctrl_idx].mean(dim=0)
    Xc = Xs_pool - mu_s
    Xn = Xc / Xc.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    X_pos = Xn[pos_idx]
    m = X_pos.shape[0]
    G = X_pos @ X_pos.T + ridge * torch.eye(m, dtype=X_pos.dtype)
    alpha = torch.linalg.solve(G, y_pos.float())
    k0 = X_pos.T @ alpha

    mu_hat = mu_s / mu_s.norm().clamp_min(1e-12)
    key = k0 - torch.dot(k0, mu_hat) * mu_hat
    key = key / key.norm().clamp_min(1e-12) * key_norm

    pre_all = Xs_pool @ key
    q90_ctrl = float(torch.quantile(pre_all[ctrl_idx], 0.90).item())
    bias = -q90_ctrl - margin
    return key, bias


_U_BASIS_CACHE: dict[int, torch.Tensor] = {}


def _get_unembed_top_basis(idx: "WeightSpaceIndex", rank: int = 64) -> torch.Tensor:
    """Return cached top-`rank` right singular vectors [r, hidden] of `idx.U_eff`."""
    key = id(idx)
    if key in _U_BASIS_CACHE and _U_BASIS_CACHE[key].shape[0] >= min(rank, idx.anat.hidden):
        return _U_BASIS_CACHE[key][:rank]
    U = idx.U_eff.float()
    q = min(rank + 8, min(U.shape) - 1) if min(U.shape) > 1 else 1
    if min(U.shape) <= 512:
        _, _, Vh = torch.linalg.svd(U, full_matrices=False)
        basis = Vh[: min(rank, Vh.shape[0])].contiguous()
    else:
        _, _, V = torch.svd_lowrank(U, q=q, niter=2)
        basis = V[:, : min(rank, V.shape[1])].t().contiguous()
    _U_BASIS_CACHE[key] = basis
    return basis


def _find_dormant_mlp_unit(
    student_idx: "WeightSpaceIndex",
    layer: int,
    unit: int | None = None,
) -> tuple[int, dict[str, Any], str, str, torch.Tensor, torch.Tensor, bool, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Locate a layer's MLP tensors and pick the most dormant unit when `unit is None`."""
    s_out_name = student_idx.anat.mlp_out(layer)
    if not s_out_name:
        raise KeyError(f"Student model has no mlp_out in layer {layer}")
    s_rd = discover_layer_readers(student_idx.ck, student_idx.anat, layer)
    s_in_name = s_rd["mlp_primary_in"]
    if not s_in_name or s_in_name not in student_idx.ck:
        raise KeyError(f"Student model has no readable MLP input tensor in layer {layer}")

    W_pr_orig = student_idx.ck.get(s_out_name)
    W_pr_f = W_pr_orig.float().clone()
    out_ax = student_idx.anat.out_axis(s_out_name)
    out_norms = W_pr_f.norm(dim=0) if out_ax == 0 else W_pr_f.norm(dim=1)
    n_units = int(out_norms.numel())

    W_fc_orig = student_idx.ck.get(s_in_name)
    W_fc_f = W_fc_orig.float().clone()
    s_hidden = student_idx.anat.hidden
    fc_cols_are_units = W_fc_f.shape[0] == s_hidden and W_fc_f.shape[1] in (n_units, 2 * n_units)
    in_norms = W_fc_f[:, :n_units].norm(dim=0) if fc_cols_are_units else W_fc_f[:n_units, :].norm(dim=1)

    s_unit = int((in_norms * out_norms).argmin().item()) if unit is None else int(unit)
    return (
        s_unit,
        s_rd,
        s_in_name,
        s_out_name,
        W_fc_orig,
        W_fc_f,
        fc_cols_are_units,
        W_pr_orig,
        W_pr_f,
        out_ax,
        out_norms,
    )


def _prepare_student_pre_mlp_pool(
    student_idx: "WeightSpaceIndex",
    layer: int,
    I_s: torch.Tensor,
    pos_shared_idx: torch.Tensor,
    top50_set: set[int],
    student_pre_mlp_vecs: dict[int, torch.Tensor] | torch.Tensor | None,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Resolve `Xs_pool`, `pos_idx`, and `ctrl_idx` for Student Key calibration."""
    if isinstance(student_pre_mlp_vecs, dict):
        pos_s_ids = [int(I_s[int(j)].item()) for j in pos_shared_idx.tolist()]
        missing = [sid for sid in pos_s_ids if sid not in student_pre_mlp_vecs]
        if missing:
            raise KeyError(f"student_pre_mlp_vecs missing trigger token ids: {missing[:3]}")
        forbidden_s_ids = {int(I_s[int(j)].item()) for j in top50_set}
        ctrl_vecs = [v.float().cpu() for sid, v in student_pre_mlp_vecs.items() if sid not in forbidden_s_ids]
        if len(ctrl_vecs) < 16:
            raise ValueError("student_pre_mlp_vecs needs at least 16 non-trigger control vectors.")
        pos_vecs = [student_pre_mlp_vecs[sid].float().cpu() for sid in pos_s_ids]
        Xs_pool = torch.stack(pos_vecs + ctrl_vecs, dim=0)
        return Xs_pool, list(range(len(pos_vecs))), list(range(len(pos_vecs), Xs_pool.shape[0]))

    if isinstance(student_pre_mlp_vecs, torch.Tensor):
        t_vecs = student_pre_mlp_vecs.float().cpu()
        if t_vecs.shape[0] == I_s.numel():
            Xs_pool = t_vecs
        elif t_vecs.shape[0] >= int(I_s.max().item()) + 1:
            Xs_pool = t_vecs[I_s]
        else:
            raise ValueError(
                f"student_pre_mlp_vecs shape {tuple(t_vecs.shape)} does not match shared tokens ({I_s.numel()}) or student vocab."
            )
    else:
        s_embed_name = student_idx.anat.embed or student_idx.anat.unembed
        E_s_shared = student_idx.ck.get(s_embed_name).float()[I_s]
        Xs_pool = _apply_student_pre_mlp_norm(student_idx, layer, E_s_shared)
    pool_pos_idx = pos_shared_idx.tolist()
    pool_ctrl_idx = [j for j in range(I_s.numel()) if j not in top50_set]
    return Xs_pool, pool_pos_idx, pool_ctrl_idx


def _install_mlp_unit_weights(
    student_idx: "WeightSpaceIndex",
    s_unit: int,
    s_rd: dict[str, Any],
    s_in_name: str,
    s_out_name: str,
    W_fc_orig: torch.Tensor,
    W_fc_f: torch.Tensor,
    fc_cols_are_units: bool,
    W_pr_orig: torch.Tensor,
    W_pr_f: torch.Tensor,
    out_ax: int,
    key: torch.Tensor,
    bias: float | None,
    val_vec: torch.Tensor,
    modified_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Write Key, optional gated Up-Key, Bias, and Value into a single MLP neuron."""
    out = modified_tensors if modified_tensors is not None else {}
    s_hidden = student_idx.anat.hidden

    if fc_cols_are_units:
        W_fc_f[:, s_unit] = key
    else:
        W_fc_f[s_unit, :] = key
    out[s_in_name] = W_fc_f.to(W_fc_orig.dtype).contiguous()

    s_up_name = s_rd["mlp_up"]
    if s_up_name and s_up_name != s_in_name and s_up_name in student_idx.ck:
        W_up_orig = student_idx.ck.get(s_up_name)
        W_up_f = W_up_orig.float().clone()
        up_vec = key / key.norm().clamp_min(1e-12)
        if W_up_f.shape[0] == s_hidden and W_up_f.shape[1] != s_hidden:
            W_up_f[:, s_unit] = up_vec
        else:
            W_up_f[s_unit, :] = up_vec
        out[s_up_name] = W_up_f.to(W_up_orig.dtype).contiguous()

    s_bias_name = s_in_name[:-7] + ".bias" if s_in_name.endswith(".weight") else None
    if bias is not None and s_bias_name and s_bias_name in student_idx.ck:
        b_orig = student_idx.ck.get(s_bias_name)
        b_f = b_orig.float().clone()
        b_f[s_unit] = float(bias)
        out[s_bias_name] = b_f.to(b_orig.dtype).contiguous()

    if out_ax == 0:
        W_pr_f[:, s_unit] = val_vec
    else:
        W_pr_f[s_unit, :] = val_vec
    out[s_out_name] = W_pr_f.to(W_pr_orig.dtype).contiguous()
    return out


def transplant_full_neuron(
    teacher_idx: "WeightSpaceIndex",
    student_idx: "WeightSpaceIndex",
    *,
    teacher_layer: int,
    teacher_unit: int,
    student_layer: int | None = None,
    student_unit: int | None = None,
    top_m_triggers: int = 10,
    top_m_chord: int = 16,
    key_norm: float = 8.0,
    val_scale: float = 3.0,
    student_pre_mlp_vecs: dict[int, torch.Tensor] | torch.Tensor | None = None,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Read BOTH halves of a Teacher neuron `(teacher_layer, teacher_unit)` — its input activation
    pattern (Key) and its output write-vector (Value) — translate both into the Student's latent
    space via Dual-Gram Rosetta, and overwrite a dormant neuron in `student_idx`."""
    I_t, I_s, shared_tokens, _, _ = _get_shared_vocab_rosetta(teacher_idx, student_idx)

    # 1. Read Teacher input activation over shared tokens strictly from Teacher weights
    t_out_name = teacher_idx.anat.mlp_out(teacher_layer)
    if not t_out_name:
        raise KeyError(f"Teacher model has no mlp_out in layer {teacher_layer}")
    W_t_out = _to_residual_columns(
        teacher_idx.ck.get(t_out_name).float(), teacher_idx.anat.out_axis(t_out_name)
    )
    v_teacher = W_t_out[:, teacher_unit]

    K_t, meta_t = get_mlp_input_keys(teacher_idx.ck, teacher_idx.anat, teacher_layer, W_t_out.shape[1])
    if K_t is None:
        raise KeyError(f"Teacher model has no readable MLP input weights in layer {teacher_layer}")

    t_embed_name = teacher_idx.anat.embed or teacher_idx.anat.unembed
    E_t_shared = teacher_idx.ck.get(t_embed_name).float()[I_t]
    if teacher_idx.anat.norm_label == "LayerNorm":
        X_t = torch.nn.functional.layer_norm(E_t_shared, (E_t_shared.shape[-1],))
    else:
        X_t = E_t_shared * torch.rsqrt(E_t_shared.pow(2).mean(-1, keepdim=True) + 1e-5)

    if "K_up_eff" in meta_t:
        act_t = torch.nn.functional.silu(X_t @ K_t[:, teacher_unit]) * (
            X_t @ meta_t["K_up_eff"][:, teacher_unit]
        )
    else:
        act_t = X_t @ K_t[:, teacher_unit]

    m_trig = min(top_m_triggers, act_t.numel())
    pos_vals, pos_shared_idx = torch.topk(act_t, k=m_trig, largest=True)
    y_pos = pos_vals / pos_vals.max().clamp_min(1e-12)
    top50_set = set(torch.topk(act_t, k=min(50, act_t.numel()), largest=True).indices.tolist())

    # 2. Locate target layer and dormant neuron in Student
    s_layer = student_layer if student_layer is not None else max(0, student_idx.anat.n_layers - 2)
    s_unit, s_rd, s_in_name, s_out_name, W_fc_orig, W_fc_f, fc_cols_are_units, W_pr_orig, W_pr_f, out_ax, out_norms = (
        _find_dormant_mlp_unit(student_idx, s_layer, student_unit)
    )

    # 3. Build Student Dual-Gram Contrastive Input Key
    Xs_pool, pool_pos_idx, pool_ctrl_idx = _prepare_student_pre_mlp_pool(
        student_idx, s_layer, I_s, pos_shared_idx, top50_set, student_pre_mlp_vecs
    )
    key, bias = _build_contrastive_gram_key(
        Xs_pool, pool_pos_idx, y_pos, pool_ctrl_idx, key_norm=key_norm
    )

    # 4. Translate Teacher Value Vector via Dual-Gram Rosetta
    rosetta = translate_vector(teacher_idx, student_idx, v_teacher, top_m=top_m_chord)
    v_s = rosetta["vector_student"]
    v_s = v_s / v_s.norm().clamp_min(1e-12) * (val_scale * max(float(out_norms.mean().item()), 1e-4))

    # 5. Install Key, Bias, and Value into Student neuron `s_unit`
    modified_tensors = _install_mlp_unit_weights(
        student_idx, s_unit, s_rd, s_in_name, s_out_name, W_fc_orig, W_fc_f, fc_cols_are_units, W_pr_orig, W_pr_f, out_ax, key, bias, v_s
    )

    if save_path is not None:
        from safetensors.torch import save_file
        from wsr.ops import compact_view
        save_file(compact_view(student_idx.ck, modified_tensors), str(save_path))

    trig_tokens = [shared_tokens[int(j)] for j in pos_shared_idx[:5].tolist()]
    return {
        "teacher_source": {
            "model_type": teacher_idx.anat.model_type,
            "hidden_dim": teacher_idx.anat.hidden,
            "layer": teacher_layer,
            "unit": teacher_unit,
            "top_triggers": trig_tokens,
            "top_semantic_chord": rosetta["teacher_top_tokens"],
        },
        "rosetta_translation": {
            "shared_vocab_anchors": rosetta["shared_vocab_size"],
            "chord_size_m": rosetta["chord_size_m"],
            "student_translated_chord": rosetta["student_translated_top_tokens"],
            "top1_exact_match": rosetta["top1_exact_match"],
            "top5_chord_overlap": rosetta["top5_chord_overlap"],
        },
        "student_graft": {
            "model_type": student_idx.anat.model_type,
            "hidden_dim": student_idx.anat.hidden,
            "layer": s_layer,
            "unit": s_unit,
            "key_bias": round(float(bias), 5),
            "calibrated_inputs": student_pre_mlp_vecs is not None,
            "tensors_modified": list(modified_tensors.keys()),
        },
        "save_path": str(save_path) if save_path else None,
        "_tensors": modified_tensors,
    }


def discover_connected_chains(
    idx: "WeightSpaceIndex",
    src_layer: int,
    dst_layer: int,
    *,
    top_k: int = 10,
    shared_vocab_with: "WeightSpaceIndex | None" = None,
    alphabetic_only: bool = True,
) -> list[dict[str, Any]]:
    """Discover strongly coupled 2-layer MLP chains `(src_layer, unit_a) -> (dst_layer, unit_b)`
    purely from static weight matrices without running any forward pass."""
    src_out_name = idx.anat.mlp_out(src_layer)
    dst_out_name = idx.anat.mlp_out(dst_layer)
    if not src_out_name or not dst_out_name:
        raise KeyError(f"Missing mlp_out in layer {src_layer} or {dst_layer}")

    W_src = _to_residual_columns(idx.ck.get(src_out_name).float(), idx.anat.out_axis(src_out_name))
    W_dst = _to_residual_columns(idx.ck.get(dst_out_name).float(), idx.anat.out_axis(dst_out_name))
    K_src, meta_src = get_mlp_input_keys(idx.ck, idx.anat, src_layer, W_src.shape[1])
    K_dst, meta_dst = get_mlp_input_keys(idx.ck, idx.anat, dst_layer, W_dst.shape[1])
    if K_src is None or K_dst is None:
        raise KeyError(f"Missing readable MLP input keys in layer {src_layer} or {dst_layer}")

    hidden = idx.anat.hidden
    embed_name = idx.anat.embed or idx.anat.unembed
    E_raw = idx.ck.get(embed_name).float()

    if shared_vocab_with is not None:
        I_t, _, shared_tokens, U_t_norm, _ = _get_shared_vocab_rosetta(idx, shared_vocab_with)
        valid_ids = I_t
        U_check = U_t_norm
        tok_names = shared_tokens
    else:
        n_v = min(idx.anat.vocab, E_raw.shape[0])
        valid_ids = torch.arange(n_v, dtype=torch.long)
        U_check = idx.U_eff[:n_v] * idx.U_eff_inv_norms[:n_v].unsqueeze(1)
        tok_names = [idx.vocab_map[i] for i in range(n_v)]

    alpha_pool = [
        j for j, s in enumerate(tok_names) if s.startswith("Ġ") and s[1:].isalpha() and len(s) >= 4
    ]
    use_alpha = alphabetic_only and len(alpha_pool) >= 64
    eval_mask = alpha_pool if use_alpha else list(range(len(tok_names)))
    eval_set = set(eval_mask)

    E_sub = E_raw[valid_ids]
    if idx.anat.norm_label == "LayerNorm":
        X_in = torch.nn.functional.layer_norm(E_sub, (E_sub.shape[-1],))
    else:
        X_in = E_sub * torch.rsqrt(E_sub.pow(2).mean(-1, keepdim=True) + 1e-5)

    if "K_up_eff" in meta_src:
        A_src = torch.nn.functional.silu(X_in @ K_src) * (X_in @ meta_src["K_up_eff"])
    else:
        A_src = X_in @ K_src

    A_eval = A_src[eval_mask]
    top_v, _ = A_eval.topk(min(10, A_eval.shape[0]), dim=0)
    med = A_eval.abs().median(dim=0).values.clamp_min(1e-6)
    sparsity = top_v.mean(dim=0) / med
    n_cand_a = min(1200, W_src.shape[1])
    cand_a = sparsity.topk(n_cand_a).indices

    W_sub = W_src[:, cand_a]
    u_dir = W_sub - W_sub.mean(dim=0, keepdim=True) if idx.anat.norm_label == "LayerNorm" else W_sub
    W_src_rms = u_dir / u_dir.norm(dim=0, keepdim=True).clamp_min(1e-12) * (hidden ** 0.5)

    G = W_src_rms.T @ K_dst
    if "K_up_eff" in meta_dst:
        UP = W_src_rms.T @ meta_dst["K_up_eff"]
        C = torch.where((G > 0) & (UP > 0), torch.nn.functional.silu(G) * UP, torch.zeros_like(G))
    else:
        C = torch.clamp(G, min=0.0)

    n_dst = W_dst.shape[1]
    k_flat = min(max(top_k * 200, 3000), C.numel())
    c_vals, c_flat_idx = C.flatten().topk(k_flat)

    Z_dst = U_check @ W_dst
    K_dst_norms = K_dst.norm(dim=0).clamp_min(1e-12)
    W_src_norms = W_src.norm(dim=0).clamp_min(1e-12)

    chains: list[dict[str, Any]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for cv, fidx in zip(c_vals.tolist(), c_flat_idx.tolist()):
        if cv <= 0:
            break
        a_loc = fidx // n_dst
        b = fidx % n_dst
        a = int(cand_a[a_loc].item())
        if a in used_a or b in used_b:
            continue

        top_b_out = Z_dst[:, b].topk(min(5, Z_dst.shape[0])).indices.tolist()
        if use_alpha and not all(o in eval_set for o in top_b_out[:3]):
            continue

        trig50_local = A_eval[:, a].topk(min(50, A_eval.shape[0])).indices.tolist()
        trig_indices = [eval_mask[j] for j in trig50_local[:10]]
        if set(trig_indices) & set(top_b_out[:5]):
            continue

        cos_ab = float((torch.dot(W_src[:, a], K_dst[:, b]) / (W_src_norms[a] * K_dst_norms[b])).item())
        w0_out = [tok_names[j] for j in (U_check @ W_src[:, a]).topk(min(5, U_check.shape[0])).indices.tolist()]

        used_a.add(a)
        used_b.add(b)
        chains.append({
            "src_layer": src_layer,
            "src_unit": a,
            "dst_layer": dst_layer,
            "dst_unit": b,
            "coupling_score": round(float(cv), 5),
            "key_cosine": round(cos_ab, 5),
            "statistical_significance_sigma": round(cos_ab * (hidden ** 0.5), 2),
            "src_triggers": [tok_names[j] for j in trig_indices[:5]],
            "src_trigger_shared_indices": trig_indices[:10],
            "src_top50_shared_indices": [eval_mask[j] for j in trig50_local],
            "src_1hop_tokens": w0_out[:4],
            "dst_2hop_tokens": [tok_names[j] for j in top_b_out[:5]],
        })
        if len(chains) >= top_k:
            break

    return chains


def _build_private_residual_wire(
    student_idx: "WeightSpaceIndex",
    src_layer: int,
    dst_layer: int,
    Xs_dst_pool: torch.Tensor,
    ctrl_idx: list[int],
    target_token_ids: list[int],
    *,
    wire_scale: float = 1.5,
    key_norm: float = 8.0,
    margin: float = 3.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Construct a private orthogonal residual wire `w_wire` connecting `src_layer` to `dst_layer`
    without interfering with vocabulary logits or intermediate layers."""
    s_hidden = student_idx.anat.hidden
    forbidden_rows: list[torch.Tensor] = [torch.ones(1, s_hidden, dtype=torch.float32)]

    # Hard constraints: zero-mean + target vocabulary rows + mean normal state at `dst_layer`
    max_tids = max(1, s_hidden // 2)
    valid_tids = list(dict.fromkeys(tid for tid in target_token_ids if 0 <= tid < student_idx.anat.vocab))[:max_tids]
    X_ctrl_dst = Xs_dst_pool[ctrl_idx]
    mu_dst = X_ctrl_dst.mean(dim=0, keepdim=True)
    if valid_tids:
        forbidden_rows.append(student_idx.U_raw_norm[valid_tids].float())
    forbidden_rows.append(mu_dst)

    hard_count = sum(r.shape[0] for r in forbidden_rows)
    # Preserve at least max(8, s_hidden // 3) degrees of freedom so small test models keep a clean wire
    soft_budget = max(0, s_hidden - hard_count - max(8, s_hidden // 3))
    unembed_rank = min(64, soft_budget // 3)
    if unembed_rank > 0:
        forbidden_rows.append(_get_unembed_top_basis(student_idx, rank=unembed_rank))

    Xc_dst = X_ctrl_dst - mu_dst
    Vh_dst: torch.Tensor | None = None
    if Xc_dst.shape[0] > 4:
        _, _, Vh_dst = torch.linalg.svd(Xc_dst, full_matrices=False)
        dst_rank = min(64, Vh_dst.shape[0], soft_budget // 3)
        if dst_rank > 0:
            forbidden_rows.append(Vh_dst[:dst_rank])

    mid_layers = list(range(src_layer + 1, dst_layer))
    rem_budget = max(0, soft_budget - unembed_rank - (min(64, Vh_dst.shape[0], soft_budget // 3) if Vh_dst is not None else 0))
    per_mid = min(16, rem_budget // max(1, 2 * len(mid_layers)))
    if per_mid > 0:
        for l_mid in mid_layers:
            rd_mid = discover_layer_readers(student_idx.ck, student_idx.anat, l_mid)
            for key_name in (rd_mid["attn_primary_in"], rd_mid["mlp_primary_in"]):
                if key_name and key_name in student_idx.ck:
                    cols = _input_to_residual_columns_safe(student_idx, key_name)
                    if cols is not None and min(cols.shape) > 1:
                        q_r = min(per_mid, min(cols.shape))
                        U_m, _, _ = torch.svd_lowrank(cols, q=min(q_r + 2, min(cols.shape)), niter=1)
                        forbidden_rows.append(U_m[:, :q_r].t())

    B = torch.cat(forbidden_rows, dim=0)
    U_b, S_b, _ = torch.linalg.svd(B.t(), full_matrices=False)
    keep = S_b > 1e-5
    Q = U_b[:, keep]

    gen = torch.Generator().manual_seed(int(seed))
    w_raw = torch.randn(s_hidden, generator=gen, dtype=torch.float32)
    w_proj = w_raw - Q @ (Q.T @ w_raw)
    # Explicitly re-project out joint [ones, target_tids] subspace to keep both zero-mean and orthogonal
    hard_proj_rows = [torch.ones(1, s_hidden, dtype=torch.float32)]
    if valid_tids:
        hard_proj_rows.append(student_idx.U_raw_norm[valid_tids].float())
    H_t = torch.cat(hard_proj_rows, dim=0)
    Ut_u, St_u, _ = torch.linalg.svd(H_t.t(), full_matrices=False)
    Q_t = Ut_u[:, St_u > 1e-5]
    if Q_t.shape[1] < s_hidden - 2:
        w_proj = w_proj - Q_t @ (Q_t.T @ w_proj)
    w_wire = w_proj / w_proj.norm().clamp_min(1e-12)

    # Compute scaled outgoing wire vector at `src_layer`
    src_out_name = student_idx.anat.mlp_out(src_layer)
    W_src_pr = student_idx.ck.get(src_out_name).float()
    src_ax = student_idx.anat.out_axis(src_out_name)
    src_norm_mean = float((W_src_pr.norm(dim=0) if src_ax == 0 else W_src_pr.norm(dim=1)).mean().item())
    v_wire = w_wire * (wire_scale * max(src_norm_mean, 1e-4))

    # Build receiver Key at `dst_layer` in post-ln_2 coordinates, orthogonal to control mean & top-32 PCs
    rd_dst = discover_layer_readers(student_idx.ck, student_idx.anat, dst_layer)
    gain_dst = rd_dst["pre_mlp_gain"]
    k0 = w_wire * gain_dst.float() if isinstance(gain_dst, torch.Tensor) else w_wire * float(gain_dst)

    mu_vec = mu_dst.squeeze(0)
    mu_hat = mu_vec / mu_vec.norm().clamp_min(1e-12)
    k0 = k0 - torch.dot(k0, mu_hat) * mu_hat
    if Vh_dst is not None:
        n_pc = min(32, Xc_dst.shape[0] - 1, s_hidden // 4)
        if n_pc > 0:
            V_ctrl = Vh_dst[:n_pc]
            k0 = k0 - V_ctrl.T @ (V_ctrl @ k0)
    key_dst = k0 / k0.norm().clamp_min(1e-12) * key_norm

    pre_ctrl = X_ctrl_dst @ key_dst
    q98_ctrl = float(torch.quantile(pre_ctrl, 0.98).item())
    bias_dst = -q98_ctrl - margin
    return w_wire, v_wire, key_dst, bias_dst


def _input_to_residual_columns_safe(idx: "WeightSpaceIndex", tensor_name: str) -> torch.Tensor | None:
    """Safely convert any 2D weight matrix to `[hidden, n_units]` columns if compatible."""
    W = idx.ck.get(tensor_name)
    if W.dim() != 2:
        return None
    h = idx.anat.hidden
    Wf = W.float()
    if Wf.shape[1] == h and Wf.shape[0] != h:
        return Wf.t()
    if Wf.shape[0] == h:
        return Wf
    return None


def transplant_connected_chain(
    teacher_idx: "WeightSpaceIndex",
    student_idx: "WeightSpaceIndex",
    *,
    teacher_src_layer: int,
    teacher_src_unit: int,
    teacher_dst_layer: int,
    teacher_dst_unit: int,
    student_src_layer: int = 6,
    student_dst_layer: int = 10,
    student_src_unit: int | None = None,
    student_dst_unit: int | None = None,
    top_m_triggers: int = 10,
    top_m_chord: int = 16,
    key_norm: float = 8.0,
    wire_scale: float = 1.5,
    val_scale: float = 3.0,
    margin_dst: float = 3.0,
    wire_seed: int = 42,
    alphabetic_only: bool = True,
    student_src_pre_mlp_vecs: dict[int, torch.Tensor] | torch.Tensor | None = None,
    student_dst_pre_mlp_vecs: dict[int, torch.Tensor] | torch.Tensor | None = None,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Transplant a connected 2-layer Teacher circuit `(teacher_src -> teacher_dst)` into two
    dormant Student neurons `(student_src -> private orthogonal residual wire -> student_dst)`."""
    if student_dst_layer <= student_src_layer:
        raise ValueError("`student_dst_layer` must be strictly greater than `student_src_layer`.")

    I_t, I_s, shared_tokens, _, _ = _get_shared_vocab_rosetta(teacher_idx, student_idx)

    # 1. Read Teacher src trigger activation & Teacher dst output vector
    t_src_out = teacher_idx.anat.mlp_out(teacher_src_layer)
    t_dst_out = teacher_idx.anat.mlp_out(teacher_dst_layer)
    if not t_src_out or not t_dst_out:
        raise KeyError("Missing Teacher mlp_out tensor.")
    W_t_src = _to_residual_columns(teacher_idx.ck.get(t_src_out).float(), teacher_idx.anat.out_axis(t_src_out))
    W_t_dst = _to_residual_columns(teacher_idx.ck.get(t_dst_out).float(), teacher_idx.anat.out_axis(t_dst_out))
    v_teacher_dst = W_t_dst[:, teacher_dst_unit]

    K_t_src, meta_t_src = get_mlp_input_keys(teacher_idx.ck, teacher_idx.anat, teacher_src_layer, W_t_src.shape[1])
    if K_t_src is None:
        raise KeyError(f"Missing Teacher input keys in layer {teacher_src_layer}")

    t_embed_name = teacher_idx.anat.embed or teacher_idx.anat.unembed
    E_t_shared = teacher_idx.ck.get(t_embed_name).float()[I_t]
    if teacher_idx.anat.norm_label == "LayerNorm":
        X_t = torch.nn.functional.layer_norm(E_t_shared, (E_t_shared.shape[-1],))
    else:
        X_t = E_t_shared * torch.rsqrt(E_t_shared.pow(2).mean(-1, keepdim=True) + 1e-5)

    if "K_up_eff" in meta_t_src:
        act_t = torch.nn.functional.silu(X_t @ K_t_src[:, teacher_src_unit]) * (
            X_t @ meta_t_src["K_up_eff"][:, teacher_src_unit]
        )
    else:
        act_t = X_t @ K_t_src[:, teacher_src_unit]

    alpha_pool = [
        j for j, s in enumerate(shared_tokens) if s.startswith("Ġ") and s[1:].isalpha() and len(s) >= 4
    ]
    if alphabetic_only and len(alpha_pool) >= 64:
        alpha_t = torch.tensor(alpha_pool, dtype=torch.long)
        act_sub = act_t[alpha_t]
        m_trig = min(top_m_triggers, act_sub.numel())
        pos_vals, pos_sub_idx = torch.topk(act_sub, k=m_trig, largest=True)
        pos_shared_idx = alpha_t[pos_sub_idx]
        top50_sub = torch.topk(act_sub, k=min(50, act_sub.numel()), largest=True).indices
        top50_set = set(alpha_t[top50_sub].tolist())
    else:
        m_trig = min(top_m_triggers, act_t.numel())
        pos_vals, pos_shared_idx = torch.topk(act_t, k=m_trig, largest=True)
        top50_set = set(torch.topk(act_t, k=min(50, act_t.numel()), largest=True).indices.tolist())
    y_pos = pos_vals / pos_vals.max().clamp_min(1e-12)

    # 2. Locate dormant Student units at `student_src_layer` and `student_dst_layer`
    s_src = _find_dormant_mlp_unit(student_idx, student_src_layer, student_src_unit)
    s_dst = _find_dormant_mlp_unit(student_idx, student_dst_layer, student_dst_unit)

    # 3. Sender Key at `student_src_layer` via Dual-Gram Contrastive Key
    Xs_src_pool, src_pos_idx, src_ctrl_idx = _prepare_student_pre_mlp_pool(
        student_idx, student_src_layer, I_s, pos_shared_idx, top50_set, student_src_pre_mlp_vecs
    )
    key_src, bias_src = _build_contrastive_gram_key(
        Xs_src_pool, src_pos_idx, y_pos, src_ctrl_idx, key_norm=key_norm
    )

    # 4. Translate Teacher `dst` Value Vector via Dual-Gram Rosetta
    rosetta = translate_vector(teacher_idx, student_idx, v_teacher_dst, top_m=top_m_chord)
    v_dst = rosetta["vector_student"]
    v_dst_scaled = v_dst / v_dst.norm().clamp_min(1e-12) * (val_scale * max(float(s_dst[10].mean().item()), 1e-4))

    # 5. Private Orthogonal Residual Wire from `student_src_layer` -> `student_dst_layer`
    Xs_dst_pool, _, dst_ctrl_idx = _prepare_student_pre_mlp_pool(
        student_idx, student_dst_layer, I_s, pos_shared_idx, top50_set, student_dst_pre_mlp_vecs
    )
    target_tids: list[int] = []
    for tok_s in rosetta["teacher_top_tokens"] + rosetta["student_translated_top_tokens"]:
        m_tok = student_idx.find_tokens(tok_s, limit=1)
        if m_tok and m_tok[0]["token_id"] not in target_tids:
            target_tids.append(m_tok[0]["token_id"])
    w_wire, v_wire, key_dst, bias_dst = _build_private_residual_wire(
        student_idx,
        student_src_layer,
        student_dst_layer,
        Xs_dst_pool,
        dst_ctrl_idx,
        target_tids,
        wire_scale=wire_scale,
        key_norm=key_norm,
        margin=margin_dst,
        seed=wire_seed,
    )

    # 6. Install Sender (`L_src`: key_src -> v_wire) and Receiver (`L_dst`: key_dst -> v_dst_scaled)
    modified_tensors: dict[str, torch.Tensor] = {}
    _install_mlp_unit_weights(
        student_idx, s_src[0], s_src[1], s_src[2], s_src[3], s_src[4], s_src[5], s_src[6], s_src[7], s_src[8], s_src[9], key_src, bias_src, v_wire, modified_tensors
    )
    _install_mlp_unit_weights(
        student_idx, s_dst[0], s_dst[1], s_dst[2], s_dst[3], s_dst[4], s_dst[5], s_dst[6], s_dst[7], s_dst[8], s_dst[9], key_dst, bias_dst, v_dst_scaled, modified_tensors
    )

    if save_path is not None:
        from safetensors.torch import save_file
        from wsr.ops import compact_view
        save_file(compact_view(student_idx.ck, modified_tensors), str(save_path))

    trig_tokens = [shared_tokens[int(j)] for j in pos_shared_idx[:5].tolist()]
    return {
        "teacher_chain": {
            "src_layer": teacher_src_layer,
            "src_unit": teacher_src_unit,
            "dst_layer": teacher_dst_layer,
            "dst_unit": teacher_dst_unit,
            "top_triggers": trig_tokens,
            "dst_semantic_chord": rosetta["teacher_top_tokens"],
        },
        "rosetta_translation": {
            "shared_vocab_anchors": rosetta["shared_vocab_size"],
            "student_translated_chord": rosetta["student_translated_top_tokens"],
            "top1_exact_match": rosetta["top1_exact_match"],
            "top5_chord_overlap": rosetta["top5_chord_overlap"],
        },
        "student_chain": {
            "src_layer": student_src_layer,
            "src_unit": s_src[0],
            "dst_layer": student_dst_layer,
            "dst_unit": s_dst[0],
            "src_bias": round(float(bias_src), 5),
            "dst_bias": round(float(bias_dst), 5),
            "wire_norm": round(float(v_wire.norm().item()), 5),
            "wire_max_target_vocab_cos": round(
                float((student_idx.U_raw_norm[target_tids] @ w_wire).abs().max().item()) if target_tids else 0.0,
                6,
            ),
            "tensors_modified": list(modified_tensors.keys()),
        },
        "save_path": str(save_path) if save_path else None,
        "_wire_vector": w_wire,
        "_tensors": modified_tensors,
    }
