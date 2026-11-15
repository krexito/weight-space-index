"""Query and navigation engine for Weight-Space Index (WSI).

Supports two seamless modes:
1. Pre-indexed (`wsi build` / `index_dir`): loads `index.json` + `subspaces.safetensors` for instant macro/micro queries.
2. On-demand Lazy mode (`lazy=True`, default when no `index_dir` is given): opens any checkpoint in ~0s and
   computes/caches layer indices, token projections, transductions, and circuit traces on the fly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .indexer import (
    Anatomy,
    Checkpoint,
    _compute_svd_basis,
    _effective_unembed,
    _input_embed,
    _input_to_residual_columns,
    _load_vocab,
    _to_residual_columns,
    build_index,
    discover_layer_readers,
    get_mlp_input_keys,
    token_roles,
)
from .rust_accel import has_rust_core, rust_analyze_residual_delta
from wsr.ops import effective_gain, load_submission, validate_edit


class WeightSpaceIndex:
    """Interactive navigator over a model's Weight-Space Index (supports both pre-built and lazy on-demand modes)."""

    def __init__(
        self,
        model_dir: str | Path,
        index_data: dict[str, Any] | None = None,
        subspaces: dict[str, torch.Tensor] | None = None,
        *,
        lazy: bool = False,
        top_neurons_per_layer: int = 32,
        top_tokens_per_unit: int = 8,
        svd_rank: int = 8,
    ):
        self.model_dir = Path(model_dir)
        self.ck = Checkpoint(self.model_dir)
        self.anat = Anatomy(self.ck)
        self.lazy = lazy
        self.top_neurons_per_layer = top_neurons_per_layer
        self.top_tokens_per_unit = top_tokens_per_unit
        self.svd_rank = svd_rank
        self.subspaces: dict[str, torch.Tensor] = subspaces or {}
        self._vocab_map: dict[int, str] | None = None
        self._u_eff: torch.Tensor | None = None
        self._u_raw_norm: torch.Tensor | None = None
        self._e_in: torch.Tensor | None = None

        if index_data is not None:
            self.data = index_data
        else:
            self.data = {
                "version": "0.2.0",
                "mode": "lazy",
                "model_dir": str(self.model_dir.resolve()),
                "anatomy": self.anat.summary(),
                "token_roles": token_roles(self.model_dir),
                "config": {
                    "top_neurons_per_layer": top_neurons_per_layer,
                    "top_tokens_per_unit": top_tokens_per_unit,
                    "svd_rank": svd_rank,
                },
                "layers": {},
                "inverted_tokens": {},
                "top_circuit_edges": [],
            }

    @classmethod
    def from_model(
        cls,
        model_dir: str | Path,
        index_dir: str | Path | None = None,
        *,
        lazy: bool | None = None,
        rebuild: bool = False,
        **build_kwargs: Any,
    ) -> "WeightSpaceIndex":
        """Load an existing index from `index_dir`, open instantaneously in `lazy=True` mode, or build eagerly."""
        mpath = Path(model_dir)
        if index_dir is not None:
            ipath = Path(index_dir)
            idx_file = ipath / "index.json"
            sub_file = ipath / "subspaces.safetensors"
            if idx_file.exists() and not rebuild:
                data = json.loads(idx_file.read_text(encoding="utf-8"))
                subs = load_file(str(sub_file), device="cpu") if sub_file.exists() else {}
                return cls(mpath, data, subs, lazy=False)
            if not lazy:
                data = build_index(mpath, out_dir=ipath, **build_kwargs)
                subs = load_file(str(sub_file), device="cpu") if sub_file.exists() else {}
                return cls(mpath, data, subs, lazy=False)

        # Default to instant lazy mode when no pre-built index_dir is provided (unless lazy=False is explicitly passed)
        use_lazy = True if lazy is None else lazy
        if use_lazy:
            return cls(mpath, None, None, lazy=True, **build_kwargs)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp)
            data = build_index(mpath, out_dir=tpath, **build_kwargs)
            subs = load_file(str(tpath / "subspaces.safetensors"), device="cpu")
            return cls(mpath, data, subs, lazy=False)

    @property
    def vocab_map(self) -> dict[int, str]:
        if self._vocab_map is None:
            self._vocab_map = _load_vocab(self.model_dir, self.anat.vocab)
        return self._vocab_map

    @property
    def U_eff(self) -> torch.Tensor:
        if self._u_eff is None:
            self._u_eff = _effective_unembed(self.ck, self.anat)
        return self._u_eff

    @property
    def U_eff_inv_norms(self) -> torch.Tensor:
        if not hasattr(self, "_u_eff_inv_norms") or self._u_eff_inv_norms is None:
            self._u_eff_inv_norms = 1.0 / self.U_eff.norm(dim=1).clamp_min(1e-12)
        return self._u_eff_inv_norms

    @property
    def U_raw_norm(self) -> torch.Tensor:
        """Row-normalized uncentered `g_final * W_unembed` for exact detection of `wsr` `orth` directions."""
        if self._u_raw_norm is None:
            U = self.ck.get(self.anat.unembed).float()
            if self.anat.final_norm:
                g = effective_gain(self.ck, self.anat.final_norm)
                U = U * (g.float().unsqueeze(0) if isinstance(g, torch.Tensor) else float(g))
            self._u_raw_norm = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return self._u_raw_norm

    @property
    def E_in(self) -> torch.Tensor:
        if self._e_in is None:
            self._e_in = _input_embed(self.ck, self.anat)
        return self._e_in

    def _ensure_layer_indexed(self, l: int) -> dict[str, Any]:
        """Index a single layer `l` on demand if not yet present in `self.data['layers']`."""
        key = str(l)
        if key in self.data["layers"]:
            return self.data["layers"][key]
        if l not in self.anat.layers:
            raise KeyError(f"Layer {l} not in model (valid layers: {sorted(self.anat.layers)})")

        layer_info: dict[str, Any] = {"layer": l, "norms": {}, "modules": {}, "readers": {}}
        for norm_kind, norm_tensor_name in self.anat.norms(l).items():
            g = effective_gain(self.ck, norm_tensor_name)
            if isinstance(g, torch.Tensor):
                gf = g.float()
                top_vals, top_dims = torch.topk(gf.abs(), k=min(8, gf.numel()))
                layer_info["norms"][norm_kind] = {
                    "tensor": norm_tensor_name,
                    "mean_gain": round(gf.mean().item(), 5),
                    "std_gain": round(gf.std().item(), 5),
                    "max_abs_gain": round(gf.abs().max().item(), 5),
                    "outlier_dims": [
                        {"dim": int(d), "gain": round(gf[int(d)].item(), 5)}
                        for _, d in zip(top_vals, top_dims)
                    ],
                }

        mlp_out_tname = self.anat.mlp_out(l)
        expected_mlp_units = None
        if mlp_out_tname:
            W_mo = self.ck.get(mlp_out_tname)
            out_ax_mo = self.anat.out_axis(mlp_out_tname)
            expected_mlp_units = W_mo.shape[1 - out_ax_mo] if W_mo.dim() == 2 else None

        K_mlp_eff, mlp_in_meta = get_mlp_input_keys(self.ck, self.anat, l, expected_mlp_units)
        if K_mlp_eff is not None:
            U_mlp_read, S_mlp_read, _ = _compute_svd_basis(K_mlp_eff, self.svd_rank)
            self.subspaces[f"layer.{l}.mlp_in.U_read"] = U_mlp_read.cpu()
            self.subspaces[f"layer.{l}.mlp_in.S_read"] = S_mlp_read.cpu()
            layer_info["readers"]["mlp_in"] = {
                "primary_tensor": mlp_in_meta.get("primary_in_tensor"),
                "gate_tensor": mlp_in_meta.get("gate_tensor"),
                "up_tensor": mlp_in_meta.get("up_tensor"),
                "spectral_norm": round(S_mlp_read[0].item(), 5) if S_mlp_read.numel() else 0.0,
            }

        module_targets: list[tuple[str, str]] = []
        if self.anat.attn_out(l):
            module_targets.append(("attn_out", self.anat.attn_out(l)))
        if self.anat.mlp_out(l):
            module_targets.append(("mlp_out", self.anat.mlp_out(l)))

        k_tok = min(self.top_tokens_per_unit, self.anat.vocab)
        for mod_role, tensor_name in module_targets:
            W_raw = self.ck.get(tensor_name).float()
            out_ax = self.anat.out_axis(tensor_name)
            W_res = _to_residual_columns(W_raw, out_ax)
            fro_norm = W_res.norm().item()
            unit_norms = W_res.norm(dim=0)

            U_r, S_r, _ = _compute_svd_basis(W_res, self.svd_rank)
            sigma_1 = S_r[0].item() if S_r.numel() > 0 else 0.0
            stable_rank = (fro_norm ** 2) / max(sigma_1 ** 2, 1e-12)
            self.subspaces[f"layer.{l}.{mod_role}.U"] = U_r.cpu()
            self.subspaces[f"layer.{l}.{mod_role}.S"] = S_r.cpu()

            svd_logits = self.U_eff @ U_r
            svd_directions = []
            for i in range(U_r.shape[1]):
                col_logits = svd_logits[:, i]
                pos_vals, pos_idx = torch.topk(col_logits, k=k_tok, largest=True)
                neg_vals, neg_idx = torch.topk(col_logits, k=k_tok, largest=False)
                svd_directions.append({
                    "component": i,
                    "singular_value": round(S_r[i].item(), 5),
                    "energy_fraction": round((S_r[i].item() ** 2) / max(fro_norm ** 2, 1e-12), 5),
                    "positive_pole_tokens": [
                        {"token_id": int(tid), "token": self.vocab_map[int(tid)], "logit": round(val.item(), 4)}
                        for val, tid in zip(pos_vals, pos_idx)
                    ],
                    "negative_pole_tokens": [
                        {"token_id": int(tid), "token": self.vocab_map[int(tid)], "logit": round(val.item(), 4)}
                        for val, tid in zip(neg_vals, neg_idx)
                    ],
                })

            layer_info["modules"][mod_role] = {
                "tensor": tensor_name,
                "shape": list(W_raw.shape),
                "out_axis": out_ax,
                "n_units": int(W_res.shape[1]),
                "fro_norm": round(fro_norm, 5),
                "spectral_norm": round(sigma_1, 5),
                "stable_rank": round(stable_rank, 3),
                "unit_norm_mean": round(unit_norms.mean().item(), 5),
                "unit_norm_max": round(unit_norms.max().item(), 5),
                "svd_directions": svd_directions,
                "top_units": [],
            }

        self.data["layers"][key] = layer_info
        return layer_info

    # ------------------------------------------------------------------
    # 1. Macro Overview & Layer Inspection
    # ------------------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        """Return a concise macro-map of the model's weight space for an LLM's context window."""
        layer_summaries = []
        for l in sorted(self.anat.layers):
            l_str = str(l)
            if l_str in self.data["layers"]:
                l_info = self.data["layers"][l_str]
                mods = {}
                for m_name, m_info in l_info["modules"].items():
                    top_svd = m_info["svd_directions"][0] if m_info["svd_directions"] else None
                    top_unit = m_info["top_units"][0] if m_info.get("top_units") else None
                    mods[m_name] = {
                        "tensor": m_info["tensor"],
                        "shape": m_info["shape"],
                        "fro_norm": m_info["fro_norm"],
                        "spectral_norm": m_info["spectral_norm"],
                        "stable_rank": m_info["stable_rank"],
                        "dominant_pole_tokens": (
                            [t["token"] for t in top_svd["positive_pole_tokens"][:3]] if top_svd else []
                        ),
                        "top_unit_transduction": top_unit.get("transduction_rule") if top_unit else None,
                    }
                layer_summaries.append({
                    "layer": l,
                    "indexed": True,
                    "readers": list(l_info.get("readers", {}).keys()),
                    "modules": mods,
                })
            else:
                # Lightweight metadata-only view in lazy mode (0 tensor reads!)
                mods = {}
                for mod_role, tname in [("attn_out", self.anat.attn_out(l)), ("mlp_out", self.anat.mlp_out(l))]:
                    if tname:
                        mods[mod_role] = {"tensor": tname, "shape": list(self.ck.shape(tname))}
                rd = discover_layer_readers(self.ck, self.anat, l)
                layer_summaries.append({
                    "layer": l,
                    "indexed": False,
                    "readers": [k for k in ("mlp_gate", "mlp_up", "attn_q", "attn_k", "attn_v") if rd.get(k)],
                    "modules": mods,
                })

        return {
            "mode": "lazy_on_demand" if self.lazy else "prebuilt",
            "anatomy": self.data["anatomy"],
            "token_roles": self.data["token_roles"],
            "n_indexed_layers": len(self.data["layers"]),
            "n_indexed_tokens": len(self.data["inverted_tokens"]),
            "layers": layer_summaries,
            "strongest_causal_circuits": self.data.get("top_circuit_edges", [])[:10],
        }

    def layer(self, layer_idx: int) -> dict[str, Any]:
        """Return indexed details for one layer (computing and caching it on the fly in lazy mode)."""
        return self._ensure_layer_indexed(layer_idx)

    # ------------------------------------------------------------------
    # 2. Semantic Search (Concept / Token -> Readers & Writers)
    # ------------------------------------------------------------------
    def find_tokens(self, query: str | int, limit: int = 10) -> list[dict[str, Any]]:
        """Find vocabulary tokens matching a string substring or exact token ID."""
        if isinstance(query, int):
            if 0 <= query < self.anat.vocab:
                return [{"token_id": query, "token": self.vocab_map[query]}]
            return []

        q_lower = query.lower()
        exact = []
        partial = []
        for tid, tok in self.vocab_map.items():
            clean = tok.replace("Ġ", " ").replace("▁", " ").strip()
            if tok == query or clean.lower() == q_lower:
                exact.append({"token_id": tid, "token": tok})
            elif q_lower in tok.lower() or q_lower in clean.lower():
                partial.append({"token_id": tid, "token": tok})
        return (exact + partial)[:limit]

    def search_token(
        self,
        query: str | int,
        *,
        top_layers: int = 5,
        top_units_per_layer: int = 5,
    ) -> dict[str, Any]:
        """Locate the exact layers, SVD subspaces, input-triggered neurons, and output-writing neurons for a token.

        Works both with precomputed SVD subspaces and in `lazy=True` mode via fast 1D vector-matrix products!
        """
        matches = self.find_tokens(query, limit=5)
        if not matches:
            return {"query": query, "matches": [], "error": f"No vocabulary token matched `{query}`"}

        primary = matches[0]
        tid = primary["token_id"]
        u_tok = self.U_eff[tid]
        u_unit = u_tok / u_tok.norm().clamp_min(1e-12)
        e_tok = self.E_in[tid]
        e_unit = e_tok / e_tok.norm().clamp_min(1e-12)

        writer_subspace_hits = []
        reader_subspace_hits = []
        exact_promoters = []
        exact_suppressors = []
        exact_triggered = []

        for l in sorted(self.anat.layers):
            # Fast path when precomputed SVD subspaces exist vs direct 1D vector-matrix product in lazy mode
            for mod_role, tname in [("attn_out", self.anat.attn_out(l)), ("mlp_out", self.anat.mlp_out(l))]:
                if not tname:
                    continue
                u_key = f"layer.{l}.{mod_role}.U"
                s_key = f"layer.{l}.{mod_role}.S"
                if u_key in self.subspaces and s_key in self.subspaces:
                    U_r = self.subspaces[u_key].float()
                    S_r = self.subspaces[s_key].float()
                    proj = u_unit @ (U_r * S_r.unsqueeze(0))
                    energy = proj.norm().item()
                    best_comp = int(proj.abs().argmax().item())
                    best_score = proj[best_comp].item()
                else:
                    # Direct 1D vector-matrix product (no SVD needed!)
                    W_raw = self.ck.get(tname).float()
                    out_ax = self.anat.out_axis(tname)
                    W_res = _to_residual_columns(W_raw, out_ax)
                    vec_proj = u_unit @ W_res  # [n_units]
                    energy = vec_proj.norm().item()
                    best_comp = int(vec_proj.abs().argmax().item())
                    best_score = vec_proj[best_comp].item()

                writer_subspace_hits.append({
                    "layer": l,
                    "module": mod_role,
                    "tensor": tname,
                    "spectral_projection_norm": round(energy, 5),
                    "best_svd_component": best_comp,
                    "best_svd_signed_score": round(best_score, 5),
                })

            for rd_role in ("mlp_in", "attn_v", "attn_q", "attn_k"):
                ur_key = f"layer.{l}.{rd_role}.U_read"
                sr_key = f"layer.{l}.{rd_role}.S_read"
                if ur_key in self.subspaces and sr_key in self.subspaces:
                    U_rd = self.subspaces[ur_key].float()
                    S_rd = self.subspaces[sr_key].float()
                    proj_rd = e_unit @ (U_rd * S_rd.unsqueeze(0))
                    reader_subspace_hits.append({
                        "layer": l,
                        "reader": rd_role,
                        "spectral_trigger_norm": round(proj_rd.norm().item(), 5),
                        "best_svd_component": int(proj_rd.abs().argmax().item()),
                    })

        writer_subspace_hits.sort(key=lambda x: x["spectral_projection_norm"], reverse=True)
        reader_subspace_hits.sort(key=lambda x: x["spectral_trigger_norm"], reverse=True)

        mlp_writer_hits = [h for h in writer_subspace_hits if h["module"] == "mlp_out"][:top_layers]
        for h in mlp_writer_hits:
            l = h["layer"]
            tname = h["tensor"]
            W_raw = self.ck.get(tname).float()
            out_ax = self.anat.out_axis(tname)
            W_res = _to_residual_columns(W_raw, out_ax)
            scores = u_tok @ W_res
            k_u = min(top_units_per_layer, scores.numel())
            p_val, p_idx = torch.topk(scores, k=k_u, largest=True)
            s_val, s_idx = torch.topk(scores, k=k_u, largest=False)
            for v, idx in zip(p_val, p_idx):
                exact_promoters.append({
                    "layer": l,
                    "tensor": tname,
                    "unit": int(idx.item()),
                    "logit_contribution": round(v.item(), 5),
                })
            for v, idx in zip(s_val, s_idx):
                exact_suppressors.append({
                    "layer": l,
                    "tensor": tname,
                    "unit": int(idx.item()),
                    "logit_contribution": round(v.item(), 5),
                })

        exact_promoters.sort(key=lambda x: x["logit_contribution"], reverse=True)
        exact_suppressors.sort(key=lambda x: x["logit_contribution"])

        mlp_reader_layers = [h["layer"] for h in reader_subspace_hits if h["reader"] == "mlp_in"][:top_layers]
        if not mlp_reader_layers:
            mlp_reader_layers = sorted(self.anat.layers)[:top_layers]

        for l in mlp_reader_layers:
            tname_out = self.anat.mlp_out(l)
            if not tname_out:
                continue
            W_out_raw = self.ck.get(tname_out).float()
            out_ax = self.anat.out_axis(tname_out)
            W_out_res = _to_residual_columns(W_out_raw, out_ax)
            K_eff, meta = get_mlp_input_keys(self.ck, self.anat, l, W_out_res.shape[1])
            if K_eff is None:
                continue
            trig_scores = e_tok @ K_eff
            k_u = min(top_units_per_layer, trig_scores.numel())
            t_val, t_idx = torch.topk(trig_scores, k=k_u, largest=True)
            for v, idx in zip(t_val, t_idx):
                uid = int(idx.item())
                out_vec = W_out_res[:, uid]
                out_logits = self.U_eff @ out_vec
                top_o_val, top_o_idx = torch.topk(out_logits, k=min(3, self.anat.vocab))
                exact_triggered.append({
                    "layer": l,
                    "in_tensor": meta.get("primary_in_tensor"),
                    "out_tensor": tname_out,
                    "unit": uid,
                    "trigger_score": round(v.item(), 5),
                    "writes_top_tokens": [
                        {"token_id": int(otid), "token": self.vocab_map[int(otid)], "logit": round(oval.item(), 4)}
                        for oval, otid in zip(top_o_val, top_o_idx)
                    ],
                })
        exact_triggered.sort(key=lambda x: x["trigger_score"], reverse=True)

        return {
            "query": query,
            "selected_token": primary,
            "other_matching_tokens": matches[1:],
            "unembed_tensor": self.anat.unembed,
            "unembed_row": tid,
            "unembed_norm": round(u_tok.norm().item(), 5),
            "preindexed_entry": self.data["inverted_tokens"].get(str(tid)),
            "top_aligned_subspaces": writer_subspace_hits[:top_layers],
            "top_reader_subspaces": reader_subspace_hits[:top_layers],
            "top_triggered_neurons": exact_triggered[:top_units_per_layer * 2],
            "top_promoting_neurons": exact_promoters[:top_units_per_layer * 2],
            "top_suppressing_neurons": exact_suppressors[:top_units_per_layer * 2],
        }

    def search_transduction(
        self,
        input_query: str | int,
        output_query: str | int,
        *,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Find neurons `(layer, unit)` whose input Key is triggered by `input_query` AND whose output Value
        promotes `output_query` (`Input Concept -> Neuron -> Output Concept`).

        Note: Only uses two 1D vector-matrix products per layer (`e_in @ K_eff` and `u_out @ W_out`), so it runs
        fast even in `lazy=True` mode without needing full SVD or full-vocab projections!
        """
        in_matches = self.find_tokens(input_query, limit=1)
        out_matches = self.find_tokens(output_query, limit=1)
        if not in_matches:
            raise ValueError(f"No token matched input_query `{input_query}`")
        if not out_matches:
            raise ValueError(f"No token matched output_query `{output_query}`")

        in_tok = in_matches[0]
        out_tok = out_matches[0]
        e_in_vec = self.E_in[in_tok["token_id"]]
        u_out_vec = self.U_eff[out_tok["token_id"]]

        candidates = []
        for l in sorted(self.anat.layers):
            tname_out = self.anat.mlp_out(l)
            if not tname_out:
                continue
            W_out_raw = self.ck.get(tname_out).float()
            out_ax = self.anat.out_axis(tname_out)
            W_out_res = _to_residual_columns(W_out_raw, out_ax)
            K_eff, meta = get_mlp_input_keys(self.ck, self.anat, l, W_out_res.shape[1])
            if K_eff is None or K_eff.shape[1] != W_out_res.shape[1]:
                continue

            in_scores = e_in_vec @ K_eff
            out_scores = u_out_vec @ W_out_res
            joint = torch.clamp(in_scores, min=0.0) * torch.clamp(out_scores, min=0.0)
            k_u = min(top_k, joint.numel())
            j_vals, j_idx = torch.topk(joint, k=k_u)
            for jv, idx in zip(j_vals, j_idx):
                uid = int(idx.item())
                candidates.append({
                    "layer": l,
                    "unit": uid,
                    "in_tensor": meta.get("primary_in_tensor"),
                    "out_tensor": tname_out,
                    "trigger_score": round(in_scores[uid].item(), 5),
                    "output_logit_score": round(out_scores[uid].item(), 5),
                    "joint_transduction_score": round(jv.item(), 5),
                })

        candidates.sort(key=lambda x: x["joint_transduction_score"], reverse=True)
        return {
            "input_token": in_tok,
            "output_token": out_tok,
            "top_transducing_neurons": candidates[:top_k],
        }

    # ------------------------------------------------------------------
    # 3. Vector, Neuron & Causal Circuit Tracing
    # ------------------------------------------------------------------
    def explain_vector(self, vec: torch.Tensor, top_k: int = 8) -> dict[str, Any]:
        """Project any residual-stream vector `vec` [hidden] onto the vocabulary and layer subspaces."""
        v = vec.detach().float().cpu().flatten()
        if v.numel() != self.anat.hidden:
            raise ValueError(f"Expected vector of length {self.anat.hidden}, got {v.numel()}")

        v_norm = v.norm().item()
        if v_norm < 1e-12:
            return {"norm": 0.0, "promoted_tokens": [], "suppressed_tokens": [], "aligned_layers": []}

        v_unit = v / v_norm
        logits = self.U_eff @ v
        # Normalize rows of U_eff so high-norm rare BPE tokens do not overshadow exact directional matches
        cos_centered = (logits / v_norm) * self.U_eff_inv_norms
        cos_raw = self.U_raw_norm @ v_unit
        # If `v` is an exact uncentered `wsr` orth vector (max raw cosine > 0.98), use raw cosine; else centered
        cos_scores = cos_raw if cos_raw.abs().max().item() > cos_centered.abs().max().item() and cos_raw.abs().max().item() > 0.95 else cos_centered

        k = min(top_k, self.anat.vocab)
        p_vals, p_idx = torch.topk(cos_scores, k=k, largest=True)
        s_vals, s_idx = torch.topk(cos_scores, k=k, largest=False)

        role_impacts: dict[str, Any] = {}
        for role_name, role_val in self.data.get("token_roles", {}).items():
            if isinstance(role_val, int) and 0 <= role_val < self.anat.vocab:
                role_impacts[role_name] = {
                    "token_id": role_val,
                    "token": self.vocab_map[role_val],
                    "logit_shift": round(logits[role_val].item(), 5),
                    "cosine": round(cos_scores[role_val].item(), 5),
                }
            elif isinstance(role_val, list) and role_val:
                valid = [r for r in role_val if isinstance(r, int) and 0 <= r < self.anat.vocab]
                if valid:
                    vals = logits[valid]
                    cvals = cos_scores[valid]
                    role_impacts[role_name] = {
                        "mean_logit_shift": round(vals.mean().item(), 5),
                        "max_logit_shift": round(vals.max().item(), 5),
                        "min_logit_shift": round(vals.min().item(), 5),
                        "mean_cosine": round(cvals.mean().item(), 5),
                    }

        layer_alignments = []
        for l in sorted(self.anat.layers):
            for mod_role in ("attn_out", "mlp_out"):
                u_key = f"layer.{l}.{mod_role}.U"
                if u_key in self.subspaces:
                    U_r = self.subspaces[u_key].float()
                    cos_vec = v_unit @ U_r
                    subspace_cos = cos_vec.norm().item()
                    layer_alignments.append({
                        "layer": l,
                        "module": mod_role,
                        "subspace_cosine": round(subspace_cos, 4),
                        "top_component": int(cos_vec.abs().argmax().item()),
                    })
        layer_alignments.sort(key=lambda x: x["subspace_cosine"], reverse=True)

        return {
            "norm": round(v_norm, 5),
            "promoted_tokens": [
                {
                    "token_id": int(tid),
                    "token": self.vocab_map[int(tid)],
                    "logit": round(logits[int(tid)].item(), 5),
                    "cosine": round(val.item(), 5),
                }
                for val, tid in zip(p_vals, p_idx)
            ],
            "suppressed_tokens": [
                {
                    "token_id": int(tid),
                    "token": self.vocab_map[int(tid)],
                    "logit": round(logits[int(tid)].item(), 5),
                    "cosine": round(val.item(), 5),
                }
                for val, tid in zip(s_vals, s_idx)
            ],
            "role_token_impacts": role_impacts,
            "most_aligned_subspaces": layer_alignments[:6],
        }

    def explain_neuron(
        self,
        layer: int,
        unit: int,
        module: str = "mlp_out",
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Inspect both the Input Key (what triggers it) and Output Value (what it writes) of `(layer, unit)`.

        Note: Only loads the target layer's tensors, so it runs instantaneously in `lazy=True` mode!
        """
        if layer not in self.anat.layers:
            raise KeyError(f"Layer {layer} not in model")
        tname = self.anat.mlp_out(layer) if module == "mlp_out" else self.anat.attn_out(layer)
        if not tname:
            raise KeyError(f"Module `{module}` not in layer {layer}")
        W_raw = self.ck.get(tname).float()
        out_ax = self.anat.out_axis(tname)
        W_res = _to_residual_columns(W_raw, out_ax)
        if not (0 <= unit < W_res.shape[1]):
            raise IndexError(f"Unit {unit} out of range for `{tname}` (0..{W_res.shape[1] - 1})")
        col = W_res[:, unit]
        explanation = self.explain_vector(col, top_k=top_k)

        input_side: dict[str, Any] | None = None
        transduction_rule: str | None = None
        if module == "mlp_out":
            K_eff, meta = get_mlp_input_keys(self.ck, self.anat, layer, W_res.shape[1])
            if K_eff is not None and unit < K_eff.shape[1]:
                k_vec = K_eff[:, unit]
                k_norm = k_vec.norm().item()
                trig_logits = self.E_in @ k_vec
                k_lim = min(top_k, self.anat.vocab)
                tr_vals, tr_idx = torch.topk(trig_logits, k=k_lim, largest=True)
                in_vals, in_idx = torch.topk(trig_logits, k=k_lim, largest=False)
                triggers = [
                    {"token_id": int(tid), "token": self.vocab_map[int(tid)], "score": round(val.item(), 5)}
                    for val, tid in zip(tr_vals, tr_idx)
                ]
                inhibitors = [
                    {"token_id": int(tid), "token": self.vocab_map[int(tid)], "score": round(val.item(), 5)}
                    for val, tid in zip(in_vals, in_idx)
                ]
                input_side = {
                    "primary_in_tensor": meta.get("primary_in_tensor"),
                    "gate_tensor": meta.get("gate_tensor"),
                    "up_tensor": meta.get("up_tensor"),
                    "key_norm": round(k_norm, 5),
                    "triggers": triggers,
                    "inhibitors": inhibitors,
                }
                in_str = ", ".join(t["token"] for t in triggers[:3])
                out_str = ", ".join(t["token"] for t in explanation["promoted_tokens"][:3])
                transduction_rule = f"[{in_str}] -> [{out_str}]"

        return {
            "layer": layer,
            "module": module,
            "tensor": tname,
            "unit": unit,
            "transduction_rule": transduction_rule,
            "input_side": input_side,
            **explanation,
        }

    def trace_circuit(
        self,
        layer: int,
        unit: int,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Trace the full causal circuit passing through MLP neuron `(layer, unit)`."""
        neuron_info = self.explain_neuron(layer, unit, module="mlp_out", top_k=top_k)
        tname_out = self.anat.mlp_out(layer)
        W_out_raw = self.ck.get(tname_out).float()
        out_ax = self.anat.out_axis(tname_out)
        W_out_res = _to_residual_columns(W_out_raw, out_ax)
        w_vec = W_out_res[:, unit]

        K_curr, _ = get_mlp_input_keys(self.ck, self.anat, layer, W_out_res.shape[1])
        k_vec = K_curr[:, unit] if K_curr is not None else None

        upstream_writers = []
        if k_vec is not None:
            for l_prev in sorted(self.anat.layers):
                if l_prev >= layer:
                    break
                prev_tname = self.anat.mlp_out(l_prev)
                if not prev_tname:
                    continue
                W_prev = _to_residual_columns(self.ck.get(prev_tname).float(), self.anat.out_axis(prev_tname))
                couplings = k_vec @ W_prev
                k_u = min(top_k, couplings.numel())
                c_vals, c_idx = torch.topk(couplings, k=k_u, largest=True)
                for cv, idx in zip(c_vals, c_idx):
                    upstream_writers.append({
                        "src_layer": l_prev,
                        "src_tensor": prev_tname,
                        "src_unit": int(idx.item()),
                        "causal_coupling_score": round(cv.item(), 5),
                    })
            upstream_writers.sort(key=lambda x: x["causal_coupling_score"], reverse=True)

        downstream_readers = []
        for l_next in sorted(self.anat.layers):
            if l_next <= layer:
                continue
            next_out_tname = self.anat.mlp_out(l_next)
            exp_units = None
            if next_out_tname:
                W_no = self.ck.get(next_out_tname)
                exp_units = W_no.shape[1 - self.anat.out_axis(next_out_tname)] if W_no.dim() == 2 else None
            K_next, meta_next = get_mlp_input_keys(self.ck, self.anat, l_next, exp_units)
            if K_next is None:
                continue
            couplings = w_vec @ K_next
            k_u = min(top_k, couplings.numel())
            c_vals, c_idx = torch.topk(couplings, k=k_u, largest=True)
            for cv, idx in zip(c_vals, c_idx):
                downstream_readers.append({
                    "dst_layer": l_next,
                    "dst_in_tensor": meta_next.get("primary_in_tensor"),
                    "dst_unit": int(idx.item()),
                    "causal_coupling_score": round(cv.item(), 5),
                })
        downstream_readers.sort(key=lambda x: x["causal_coupling_score"], reverse=True)

        return {
            "layer": layer,
            "unit": unit,
            "transduction_rule": neuron_info.get("transduction_rule"),
            "upstream_writers": upstream_writers[:top_k],
            "neuron": neuron_info,
            "downstream_readers": downstream_readers[:top_k],
        }

    # ------------------------------------------------------------------
    # 4. Surgical Edit Diagnosis (`explain_edit`) & Synthesis (`suggest_edit`)
    # ------------------------------------------------------------------
    def explain_edit(self, edit_source: str | Path | dict | list, top_k: int = 8) -> dict[str, Any]:
        """Analyze a `wsr` edit file (.safetensors / .json) or ops structure and explain its semantic effect."""
        compact_vocab_slices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        compact_residual_slices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if isinstance(edit_source, (str, Path)) and str(edit_source).endswith(".safetensors"):
            raw_st = load_file(str(edit_source))
            full: dict[str, torch.Tensor] = {}
            for k, v in raw_st.items():
                if k.endswith("::row_index"):
                    continue
                if k.endswith("::rows"):
                    n = k[:-len("::rows")]
                    r_idx = raw_st.get(n + "::row_index")
                    if r_idx is not None and (n == self.anat.unembed or n == self.anat.embed):
                        compact_vocab_slices[n] = (r_idx, v)
                        continue
                    if (
                        r_idx is not None
                        and n in self.ck
                        and self.ck.get(n).dim() == 2
                        and self.anat.out_axis(n) == 1
                        and 0 < r_idx.numel() <= 64
                    ):
                        compact_residual_slices[n] = (r_idx, v)
                        continue
                # Expand any remaining compact row tensor or copy regular tensor
                if k.endswith("::rows"):
                    n = k[:-len("::rows")]
                    r_idx = raw_st[n + "::row_index"]
                    B = self.ck.get(n).clone()
                    B[r_idx] = v.to(B.dtype)
                    full[n] = B
                else:
                    full[k] = v
        elif isinstance(edit_source, (str, Path)):
            full = load_submission(self.ck, edit_source, self.anat)
        elif isinstance(edit_source, list):
            from wsr.ops import apply_ops
            full = apply_ops(self.ck, {"ops": edit_source}, self.anat)[0]
        elif isinstance(edit_source, dict) and "ops" in edit_source:
            from wsr.ops import apply_ops
            full = apply_ops(self.ck, edit_source, self.anat)[0]
        elif isinstance(edit_source, dict):
            full = edit_source
        else:
            raise TypeError(f"Unsupported edit_source type: {type(edit_source)}")

        tensor_reports = []
        total_d_sq = 0.0
        total_o_sq = 0.0
        total_changed = 0
        errors: list[str] = []

        # Fast path for compact vocabulary row edits (`wte.weight::rows` / `embed_tokens.weight::rows`)
        for n, (r_idx, rows_new) in compact_vocab_slices.items():
            T_old_raw = self.ck.get(n)
            rows_old = T_old_raw[r_idx].float()
            rows_new_f = rows_new.float()
            d_rows = rows_new_f - rows_old
            d_norm = d_rows.norm().item()
            if not hasattr(self, "_cached_unembed_fro_norm"):
                self._cached_unembed_fro_norm = T_old_raw.float().norm().item()
            old_norm = self._cached_unembed_fro_norm
            rel_norm = d_norm / max(old_norm, 1e-12)
            vals_chg = int((d_rows != 0).sum().item())
            total_d_sq += d_norm ** 2
            total_o_sq += old_norm ** 2
            total_changed += vals_chg

            row_deltas = d_rows.norm(dim=1)
            top_r_vals, top_r_order = torch.topk(row_deltas, k=min(top_k, row_deltas.numel()))
            edited_tokens = []
            for _, r_ord in zip(top_r_vals, top_r_order):
                loc_i = int(r_ord.item())
                tid = int(r_idx[loc_i].item())
                old_r_norm = rows_old[loc_i].norm().item()
                new_r_norm = rows_new_f[loc_i].norm().item()
                cos = torch.nn.functional.cosine_similarity(rows_new_f[loc_i], rows_old[loc_i], dim=0).item()
                edited_tokens.append({
                    "token_id": tid,
                    "token": self.vocab_map.get(tid, f"<tok_{tid}>"),
                    "old_norm": round(old_r_norm, 5),
                    "new_norm": round(new_r_norm, 5),
                    "scale_ratio": round(new_r_norm / max(old_r_norm, 1e-12), 4),
                    "cosine_with_original": round(cos, 4),
                })
            tensor_reports.append({
                "tensor": n,
                "shape": list(T_old_raw.shape),
                "delta_fro_norm": round(d_norm, 6),
                "rel_norm": round(rel_norm, 6),
                "values_changed": vals_chg,
                "role": "unembed_or_embed",
                "changed_token_rows_count": int(r_idx.numel()),
                "top_edited_tokens": edited_tokens,
            })

        # Fast path for compact surgical neuron edits (`out_axis == 1`, e.g. GPT-2 `mlp.c_proj.weight::rows`)
        if not hasattr(self, "_cached_tensor_fro_norms"):
            self._cached_tensor_fro_norms: dict[str, float] = {}
        for n, (r_idx, rows_new) in compact_residual_slices.items():
            T_old_raw = self.ck.get(n)
            rows_old = T_old_raw[r_idx].float()
            rows_new_f = rows_new.float()
            d_rows = rows_new_f - rows_old  # [k, hidden]
            d_norm = d_rows.norm().item()
            if n not in self._cached_tensor_fro_norms:
                self._cached_tensor_fro_norms[n] = T_old_raw.float().norm().item()
            old_norm = self._cached_tensor_fro_norms[n]
            new_norm_sq = old_norm ** 2 - rows_old.norm().item() ** 2 + rows_new_f.norm().item() ** 2
            new_norm = max(new_norm_sq, 0.0) ** 0.5
            rel_norm = d_norm / max(old_norm, 1e-12)
            vals_chg = int((d_rows != 0).sum().item())
            total_d_sq += d_norm ** 2
            total_o_sq += old_norm ** 2
            total_changed += vals_chg

            U_d, S_d, _ = torch.linalg.svd(d_rows.t(), full_matrices=False)
            u0 = U_d[:, 0]
            cos_c = (self.U_eff @ u0) * self.U_eff_inv_norms
            cos_r = self.U_raw_norm @ u0
            if max((-cos_c).max().item(), (-cos_r).max().item()) > max(cos_c.max().item(), cos_r.max().item()):
                u0 = -u0
            top_u = u0 * S_d[0]
            s_energy_top1 = (S_d[0].item() ** 2) / max(d_norm ** 2, 1e-12)

            col_deltas = d_rows.norm(dim=1)
            top_loc = int(col_deltas.argmax().item())
            top_uid = int(r_idx[top_loc].item())
            old_col_exp = self.explain_vector(rows_old[top_loc], top_k=3)
            new_col_exp = self.explain_vector(rows_new_f[top_loc], top_k=3)

            tensor_reports.append({
                "tensor": n,
                "shape": list(T_old_raw.shape),
                "delta_fro_norm": round(d_norm, 6),
                "rel_norm": round(rel_norm, 6),
                "values_changed": vals_chg,
                "role": "residual_matrix",
                "overall_cosine_new_old": round(old_norm / max(new_norm, 1e-12), 5),
                "norm_ratio_new_old": round(new_norm / max(old_norm, 1e-12), 5),
                "changed_units_count": int(r_idx.numel()),
                "total_units_count": int(T_old_raw.shape[0]),
                "delta_rank1_energy_fraction": round(s_energy_top1, 4),
                "subspace_action": "amplified_or_injected",
                "old_energy_along_principal_dir": round((rows_old @ u0).norm().item(), 5),
                "new_energy_along_principal_dir": round((rows_new_f @ u0).norm().item(), 5),
                "principal_delta_direction": self.explain_vector(top_u, top_k=top_k),
                "surgical_neuron_rewire": {
                    "top_modified_unit": top_uid,
                    "old_promoted_tokens": [t["token"] for t in old_col_exp["promoted_tokens"]],
                    "new_promoted_tokens": [t["token"] for t in new_col_exp["promoted_tokens"]],
                },
            })

        for n, T_new in full.items():
            if n not in self.ck:
                errors.append(f"unknown tensor `{n}`")
                continue
            T_old_raw = self.ck.get(n)
            if T_old_raw.shape != T_new.shape:
                errors.append(f"shape mismatch for `{n}`")
                continue

            # Try zero-copy multithreaded Rust kernel for 2D residual matrices
            out_ax = self.anat.out_axis(n) if (T_old_raw.dim() == 2 and self.anat.hidden in T_old_raw.shape and n != self.anat.unembed and n != self.anat.embed) else -1
            rust_res = rust_analyze_residual_delta(T_old_raw, T_new, out_ax) if out_ax in (0, 1) else None

            if rust_res is not None:
                d_norm = rust_res["delta_norm"]
                old_norm = rust_res["old_norm"]
                new_norm = rust_res["new_norm"]
                rel_norm = d_norm / max(old_norm, 1e-12)
                vals_chg = rust_res["values_changed"]
                total_d_sq += d_norm ** 2
                total_o_sq += old_norm ** 2
                total_changed += vals_chg

                report: dict[str, Any] = {
                    "tensor": n,
                    "shape": list(T_old_raw.shape),
                    "delta_fro_norm": round(d_norm, 6),
                    "rel_norm": round(rel_norm, 6),
                    "values_changed": vals_chg,
                    "engine": "rust_simd",
                }
                if d_norm < 1e-12:
                    report["semantic_effect"] = "no_op (zero delta)"
                    tensor_reports.append(report)
                    continue

                ratio = new_norm / max(old_norm, 1e-12)
                cos_all = rust_res["overall_cosine_new_old"]
                s0 = rust_res["s0"]
                top_u = rust_res["top_u"]
                u0 = top_u / max(s0, 1e-12)

                # Resolve SVD sign ambiguity toward positive vocabulary cosine pole
                cos_c = (self.U_eff @ u0) * self.U_eff_inv_norms
                cos_r = self.U_raw_norm @ u0
                best_pos = max(cos_c.max().item(), cos_r.max().item())
                best_neg = max((-cos_c).max().item(), (-cos_r).max().item())
                if best_neg > best_pos:
                    top_u = -top_u

                old_energy = rust_res["old_energy"]
                new_energy = rust_res["new_energy"]
                subspace_action = (
                    "projected_out_or_suppressed"
                    if new_energy < old_energy * 0.95
                    else ("amplified_or_injected" if new_energy > old_energy * 1.05 else "rotated_or_shifted")
                )
                s_energy_top1 = (s0 ** 2) / max(d_norm ** 2, 1e-12)
                changed_cnt = rust_res["changed_units_count"]

                report["role"] = "residual_matrix"
                report["overall_cosine_new_old"] = round(cos_all, 5)
                report["norm_ratio_new_old"] = round(ratio, 5)
                report["changed_units_count"] = changed_cnt
                report["total_units_count"] = rust_res["total_units_count"]
                report["delta_rank1_energy_fraction"] = round(s_energy_top1, 4)
                report["subspace_action"] = subspace_action
                report["old_energy_along_principal_dir"] = round(old_energy, 5)
                report["new_energy_along_principal_dir"] = round(new_energy, 5)
                report["principal_delta_direction"] = self.explain_vector(top_u, top_k=top_k)

                if 0 < changed_cnt <= 64:
                    old_col_exp = self.explain_vector(rust_res["old_col"], top_k=3)
                    new_col_exp = self.explain_vector(rust_res["new_col"], top_k=3)
                    report["surgical_neuron_rewire"] = {
                        "top_modified_unit": rust_res["top_modified_unit"],
                        "old_promoted_tokens": [t["token"] for t in old_col_exp["promoted_tokens"]],
                        "new_promoted_tokens": [t["token"] for t in new_col_exp["promoted_tokens"]],
                    }
                tensor_reports.append(report)
                continue

            T_old = T_old_raw.float()
            dW = T_new.float() - T_old
            d_norm = dW.norm().item()
            old_norm = T_old.norm().item()
            rel_norm = d_norm / max(old_norm, 1e-12)
            vals_chg = int((dW != 0).sum().item())
            total_d_sq += d_norm ** 2
            total_o_sq += old_norm ** 2
            total_changed += vals_chg

            report = {
                "tensor": n,
                "shape": list(T_old.shape),
                "delta_fro_norm": round(d_norm, 6),
                "rel_norm": round(rel_norm, 6),
                "values_changed": vals_chg,
            }

            if d_norm < 1e-12:
                report["semantic_effect"] = "no_op (zero delta)"
                tensor_reports.append(report)
                continue

            if n == self.anat.unembed or n == self.anat.embed:
                row_deltas = dW.norm(dim=1)
                changed_idx = torch.nonzero(row_deltas > 1e-9).flatten()
                top_r_vals, top_r_order = torch.topk(
                    row_deltas[changed_idx], k=min(top_k, changed_idx.numel())
                )
                edited_tokens = []
                for _, r_ord in zip(top_r_vals, top_r_order):
                    tid = int(changed_idx[r_ord].item())
                    old_r_norm = T_old[tid].norm().item()
                    new_r_norm = T_new[tid].float().norm().item()
                    cos = torch.nn.functional.cosine_similarity(
                        T_new[tid].float(), T_old[tid], dim=0
                    ).item()
                    edited_tokens.append({
                        "token_id": tid,
                        "token": self.vocab_map[tid],
                        "old_norm": round(old_r_norm, 5),
                        "new_norm": round(new_r_norm, 5),
                        "scale_ratio": round(new_r_norm / max(old_r_norm, 1e-12), 4),
                        "cosine_with_original": round(cos, 4),
                    })
                report["role"] = "unembed_or_embed"
                report["changed_token_rows_count"] = int(changed_idx.numel())
                report["top_edited_tokens"] = edited_tokens

            elif dW.dim() == 1 and dW.numel() == self.anat.hidden:
                scale_ratios = T_new.float() / T_old.clamp_min(1e-12)
                report["role"] = "norm_gain"
                report["mean_new_over_old"] = round(scale_ratios.mean().item(), 5)
                report["delta_direction"] = self.explain_vector(dW, top_k=top_k)

            elif dW.dim() in (2, 3) and self.anat.hidden in dW.shape:
                out_ax = self.anat.out_axis(n)
                dW_res = _to_residual_columns(dW, out_ax)
                W_old_res = _to_residual_columns(T_old, out_ax)
                W_new_res = _to_residual_columns(T_new.float(), out_ax)

                ratio = (W_new_res.norm() / max(W_old_res.norm().item(), 1e-12)).item()
                cos_all = torch.nn.functional.cosine_similarity(
                    W_new_res.flatten(), W_old_res.flatten(), dim=0
                ).item()

                col_deltas = dW_res.norm(dim=0)
                changed_cols = torch.nonzero(col_deltas > 1e-6).flatten()

                if 0 < changed_cols.numel() <= 256:
                    U_d, S_d, _ = torch.linalg.svd(dW_res[:, changed_cols], full_matrices=False)
                else:
                    U_d, S_d, _ = torch.svd_lowrank(dW_res, q=min(4, min(dW_res.shape)), niter=4)
                u0 = U_d[:, 0]
                u_norms = self.U_eff.norm(dim=1).clamp_min(1e-12)
                cos_c = (self.U_eff @ u0) / u_norms
                cos_r = self.U_raw_norm @ u0
                best_pos = max(cos_c.max().item(), cos_r.max().item())
                best_neg = max((-cos_c).max().item(), (-cos_r).max().item())
                if best_neg > best_pos:
                    u0 = -u0

                old_energy = (u0 @ W_old_res).norm().item()
                new_energy = (u0 @ W_new_res).norm().item()
                subspace_action = (
                    "projected_out_or_suppressed"
                    if new_energy < old_energy * 0.95
                    else ("amplified_or_injected" if new_energy > old_energy * 1.05 else "rotated_or_shifted")
                )
                top_u = u0 * S_d[0]
                s_energy_top1 = (S_d[0].item() ** 2) / max(d_norm ** 2, 1e-12)

                report["role"] = "residual_matrix"
                report["overall_cosine_new_old"] = round(cos_all, 5)
                report["norm_ratio_new_old"] = round(ratio, 5)
                report["changed_units_count"] = int(changed_cols.numel())
                report["total_units_count"] = int(dW_res.shape[1])
                report["delta_rank1_energy_fraction"] = round(s_energy_top1, 4)
                report["subspace_action"] = subspace_action
                report["old_energy_along_principal_dir"] = round(old_energy, 5)
                report["new_energy_along_principal_dir"] = round(new_energy, 5)
                report["principal_delta_direction"] = self.explain_vector(top_u, top_k=top_k)

                if 0 < changed_cols.numel() <= 64:
                    top_uid = int(changed_cols[col_deltas[changed_cols].argmax()].item())
                    old_col_exp = self.explain_vector(W_old_res[:, top_uid], top_k=3)
                    new_col_exp = self.explain_vector(W_new_res[:, top_uid], top_k=3)
                    report["surgical_neuron_rewire"] = {
                        "top_modified_unit": top_uid,
                        "old_promoted_tokens": [t["token"] for t in old_col_exp["promoted_tokens"]],
                        "new_promoted_tokens": [t["token"] for t in new_col_exp["promoted_tokens"]],
                    }

            tensor_reports.append(report)

        total_rel = (total_d_sq ** 0.5) / max(total_o_sq ** 0.5, 1e-12)
        return {
            "validation": {
                "ok": len(errors) == 0 and total_changed > 0,
                "errors": errors,
                "total_rel_norm": round(total_rel, 6),
                "total_values_changed": total_changed,
            },
            "tensors": tensor_reports,
        }

    def suggest_edit(
        self,
        *,
        promote_tokens: list[str | int] | None = None,
        suppress_tokens: list[str | int] | None = None,
        strategy: str = "orth",
        layers: list[int] | None = None,
        alpha: float = 1.0,
        scale_factor: float = 1.5,
    ) -> dict[str, Any]:
        """Synthesize a ready-to-run `wsr` edit specification from semantic token goals."""
        promote_ids: list[int] = []
        for q in promote_tokens or []:
            m = self.find_tokens(q, limit=1)
            if m:
                promote_ids.append(m[0]["token_id"])

        suppress_ids: list[int] = []
        for q in suppress_tokens or []:
            m = self.find_tokens(q, limit=1)
            if m:
                suppress_ids.append(m[0]["token_id"])

        ops: list[dict[str, Any]] = []
        if strategy == "scale_unembed":
            if promote_ids:
                ops.append({
                    "op": "scale_rows",
                    "tensor": self.anat.unembed,
                    "rows": promote_ids,
                    "factor": scale_factor,
                })
            if suppress_ids:
                ops.append({
                    "op": "scale_rows",
                    "tensor": self.anat.unembed,
                    "rows": suppress_ids,
                    "factor": round(1.0 / max(scale_factor, 1e-4), 4),
                })
        elif strategy == "orth":
            target_rows = suppress_ids or promote_ids
            if not target_rows:
                raise ValueError("Provide at least one token in `suppress_tokens` or `promote_tokens`")
            if layers is None:
                search_res = self.search_token(target_rows[0], top_layers=max(1, self.anat.n_layers // 2))
                chosen_layers = sorted({
                    h["layer"] for h in search_res["top_aligned_subspaces"] if h["module"] == "mlp_out"
                })
                if not chosen_layers:
                    chosen_layers = sorted(self.anat.layers)[self.anat.n_layers // 2:]
            else:
                chosen_layers = layers

            target_tensors = []
            for l in chosen_layers:
                target_tensors.extend(self.anat.mlp_outs(l))
            ops.append({
                "op": "orth",
                "tensors": target_tensors,
                "u_rows": target_rows,
                "alpha": alpha,
            })
        else:
            raise ValueError(f"Unknown strategy `{strategy}` (expected 'orth' or 'scale_unembed')")

        return {
            "strategy": strategy,
            "promote_token_ids": promote_ids,
            "suppress_token_ids": suppress_ids,
            "ops": ops,
        }

    def rewire_transduction(
        self,
        input_query: str | int,
        old_output_query: str | int,
        new_output_query: str | int,
        *,
        top_k_neurons: int = 8,
        strength: float = 2.5,
        save_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Surgically rewire the top `K` neurons that transduce `input_query -> old_output_query`
        so that when triggered by `input_query`, they write `new_output_query` instead.

        If `save_path` is provided, writes a `wsr`-compatible `.safetensors` edit file.
        """
        trans = self.search_transduction(input_query, old_output_query, top_k=top_k_neurons)
        new_matches = self.find_tokens(new_output_query, limit=1)
        if not new_matches:
            raise ValueError(f"No token matched new_output_query `{new_output_query}`")
        new_tok = new_matches[0]
        old_tok = trans["output_token"]

        u_old = self.U_eff[old_tok["token_id"]].float()
        u_old_hat = u_old / u_old.norm().clamp_min(1e-12)
        u_new = self.U_eff[new_tok["token_id"]].float()
        u_new_hat = u_new / u_new.norm().clamp_min(1e-12)

        # Group target neurons by layer
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for item in trans["top_transducing_neurons"]:
            by_layer.setdefault(item["layer"], []).append(item)

        modified_tensors: dict[str, torch.Tensor] = {}
        rewired_records: list[dict[str, Any]] = []

        for l, items in sorted(by_layer.items()):
            tname = self.anat.mlp_out(l)
            if not tname:
                continue
            W_orig = self.ck.get(tname)
            W_mod = W_orig.float().clone()
            out_ax = self.anat.out_axis(tname)

            for item in items:
                uid = item["unit"]
                w_vec = W_mod[:, uid].clone() if out_ax == 0 else W_mod[uid, :].clone()
                orig_norm = w_vec.norm().item()
                old_proj = torch.dot(w_vec, u_old_hat).item()

                # Remove old_output component and inject new_output direction
                w_perp = w_vec - old_proj * u_old_hat
                w_new = w_perp + (strength * max(orig_norm, 1e-4)) * u_new_hat

                if out_ax == 0:
                    W_mod[:, uid] = w_new
                else:
                    W_mod[uid, :] = w_new

                new_old_logit = torch.dot(u_old, w_new).item()
                new_target_logit = torch.dot(u_new, w_new).item()
                rewired_records.append({
                    "layer": l,
                    "unit": uid,
                    "tensor": tname,
                    "trigger_score": item["trigger_score"],
                    "old_target_logit_before": item["output_logit_score"],
                    "old_target_logit_after": round(new_old_logit, 5),
                    "new_target_logit_after": round(new_target_logit, 5),
                })

            modified_tensors[tname] = W_mod.to(W_orig.dtype).contiguous()

        if save_path is not None:
            from safetensors.torch import save_file
            from wsr.ops import compact_view
            save_file(compact_view(self.ck, modified_tensors), str(save_path))

        return {
            "input_token": trans["input_token"],
            "old_output_token": old_tok,
            "new_output_token": new_tok,
            "neurons_rewired": rewired_records,
            "tensors_modified": list(modified_tensors.keys()),
            "save_path": str(save_path) if save_path else None,
            "_tensors": modified_tensors,
        }

    # ------------------------------------------------------------------
    # 5. Model Diffing (Same Family) & Cross-Architecture Circuit Alignment
    # ------------------------------------------------------------------
    def diff_model(self, other: "WeightSpaceIndex | str | Path", **kwargs: Any) -> dict[str, Any]:
        """Compare this checkpoint (base) with another checkpoint of the same architecture (target)."""
        from .compare import diff_checkpoints
        other_idx = other if isinstance(other, WeightSpaceIndex) else WeightSpaceIndex.from_model(other)
        return diff_checkpoints(self, other_idx, **kwargs)

    def align_circuits(
        self,
        other: "WeightSpaceIndex | str | Path",
        *,
        transduction_pairs: list[tuple[str, str]] | None = None,
        concepts: list[str] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Align functionally equivalent semantic circuits across two different architectures/models."""
        from .compare import align_semantic_circuits
        other_idx = other if isinstance(other, WeightSpaceIndex) else WeightSpaceIndex.from_model(other)
        return align_semantic_circuits(
            self,
            other_idx,
            transduction_pairs=transduction_pairs,
            concepts=concepts,
            top_k=top_k,
        )

    def transfer_edit_to(
        self,
        target_model: "WeightSpaceIndex | str | Path",
        edit_source: Any,
        *,
        top_k_neurons: int = 8,
        strength: float = 2.5,
        save_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Decompile a raw weight edit (`edit.safetensors` / ops) made on THIS model (`self`),
        locate the equivalent semantic circuit in `target_model` (even if it has a completely
        different architecture, hidden size, and layer count!), and generate the equivalent edit
        for `target_model`."""
        from .compare import transfer_edit
        other_idx = (
            target_model
            if isinstance(target_model, WeightSpaceIndex)
            else WeightSpaceIndex.from_model(target_model)
        )
        return transfer_edit(
            self,
            other_idx,
            edit_source,
            top_k_neurons=top_k_neurons,
            strength=strength,
            save_path=str(save_path) if save_path else None,
        )


def load_index(model_dir: str | Path, index_dir: str | Path | None = None, **kwargs: Any) -> WeightSpaceIndex:
    """Convenience loader for `WeightSpaceIndex`."""
    return WeightSpaceIndex.from_model(model_dir, index_dir=index_dir, **kwargs)
