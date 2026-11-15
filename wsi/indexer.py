"""Core indexer for Weight-Space Index (WSI).

Extracts a fast, hierarchical semantic and structural index from any safetensors checkpoint
without running forward passes or training. Indexes both:
- Output-side (Residual Writers: `mlp_out`, `attn_out` -> Vocabulary Unembedding)
- Input-side (Residual Readers / Keys: `mlp_gate`, `mlp_up`, `attn_q/k/v` <- Input Embeddings & Upstream Writers)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

# Ensure `wsr` is importable from the sibling repo directory if not installed globally
_WSR_ROOT = Path(__file__).resolve().parent.parent / "wsr"
if _WSR_ROOT.exists() and str(_WSR_ROOT) not in sys.path:
    sys.path.insert(0, str(_WSR_ROOT))

from wsr.anatomy import Anatomy, token_roles
from wsr.checkpoint import Checkpoint
from wsr.ops import effective_gain

MLP_GATE_STEMS = [
    "mlp.gate_proj",
    "feed_forward.w1",
    "mlp.w1",
    "ffn.gate_proj",
    "ffn.w1",
    "feed_forward.gate_proj",
]
MLP_UP_STEMS = [
    "mlp.up_proj",
    "feed_forward.w3",
    "mlp.w3",
    "mlp.c_fc",
    "mlp.fc1",
    "mlp.dense_h_to_4h",
    "ffn.lin1",
    "ffn.up_proj",
    "ffn.w3",
    "feed_forward.up_proj",
    "mlp.gate_up_proj",
]
ATTN_V_STEMS = [
    "self_attn.v_proj",
    "attn.v_proj",
    "attention.wv",
    "attention.v_proj",
    "attn.c_attn",
    "self_attention.query_key_value",
    "linear_attn.in_proj_qkv",
]
ATTN_Q_STEMS = [
    "self_attn.q_proj",
    "attn.q_proj",
    "attention.wq",
    "attention.q_proj",
]
ATTN_K_STEMS = [
    "self_attn.k_proj",
    "attn.k_proj",
    "attention.wk",
    "attention.k_proj",
]
PRE_ATTN_NORM_KEYS = (
    "input_layernorm",
    "ln_1",
    "attn_norm",
    "layer_norm1",
    "operator_norm",
)
PRE_MLP_NORM_KEYS = (
    "pre_feedforward_layernorm",
    "post_attention_layernorm",
    "ln_2",
    "ffn_norm",
    "mlp_norm",
    "layer_norm2",
)


def _load_vocab(model_dir: Path, vocab_size: int) -> dict[int, str]:
    """Load id -> token string mapping from tokenizer files if available, else fallback to #<id>."""
    id_to_tok: dict[int, str] = {}
    tok_json = model_dir / "tokenizer.json"
    vocab_json = model_dir / "vocab.json"
    if tok_json.exists() and tok_json.stat().st_size > 0:
        try:
            data = json.loads(tok_json.read_text(encoding="utf-8"))
            vocab = data.get("model", {}).get("vocab", {})
            for tok, idx in vocab.items():
                if isinstance(idx, int) and 0 <= idx < vocab_size:
                    id_to_tok[idx] = tok
            for added in data.get("added_tokens", []):
                if "id" in added and "content" in added:
                    aid = int(added["id"])
                    if 0 <= aid < vocab_size:
                        id_to_tok[aid] = str(added["content"])
        except Exception:
            pass
    if not id_to_tok and vocab_json.exists():
        try:
            vocab = json.loads(vocab_json.read_text(encoding="utf-8"))
            for tok, idx in vocab.items():
                if isinstance(idx, int) and 0 <= idx < vocab_size:
                    id_to_tok[idx] = tok
        except Exception:
            pass

    for i in range(vocab_size):
        if i not in id_to_tok:
            id_to_tok[i] = f"<tok_{i}>"
    return id_to_tok


def _to_residual_columns(W: torch.Tensor, out_axis: int) -> torch.Tensor:
    """Return a 2D view [hidden, n_units] where each column is a vector in residual stream space."""
    if W.dim() == 3:
        if out_axis == 0:
            return W.permute(1, 0, 2).reshape(W.shape[1], -1)
        else:
            return W.reshape(-1, W.shape[-1]).t()
    if out_axis == 0:
        return W
    return W.t()


def _input_to_residual_columns(W_in: torch.Tensor, hidden: int, expected_units: int | None = None) -> torch.Tensor:
    """Convert an input-reading matrix (e.g. `gate_proj`, `up_proj`, `c_fc`, `v_proj`) into `[hidden, n_units]`
    where each column `j` is the input receptor (Key) vector in residual-stream space `R^hidden`."""
    if W_in.dim() != 2:
        raise ValueError(f"Expected 2D input matrix, got shape {tuple(W_in.shape)}")
    r, c = W_in.shape
    # Standard nn.Linear(hidden, inter) has shape [inter, hidden] -> transpose to [hidden, inter]
    if c == hidden and r != hidden:
        cols = W_in.t()
    elif r == hidden and c != hidden:
        # GPT-2 Conv1D or transposed storage: already [hidden, inter]
        cols = W_in
    else:
        # Square [hidden, hidden]: standard nn.Linear stores input along axis 1 -> transpose
        cols = W_in.t()

    if expected_units is not None and cols.shape[1] == 2 * expected_units:
        # Fused gate_up_proj [hidden, 2 * inter]: first half is gate, second half is up
        cols = cols[:, :expected_units]
    return cols.contiguous()


def _find_layer_tensor(anat: Anatomy, l: int, stems: list[str]) -> str | None:
    rest = anat._rest.get(l, {})
    for s in stems:
        if f"{s}.weight" in rest:
            return rest[f"{s}.weight"]
    return None


def _find_pre_norm_gain(ck: Checkpoint, anat: Anatomy, l: int, candidates: tuple[str, ...]) -> torch.Tensor | float:
    norms = anat.norms(l)
    for k in candidates:
        if k in norms:
            return effective_gain(ck, norms[k])
    return 1.0


def _apply_input_gain(cols: torch.Tensor, gain: torch.Tensor | float) -> torch.Tensor:
    """Scale residual dimension (axis 0 of `[hidden, n_units]`) by pre-layer norm effective gain."""
    if isinstance(gain, torch.Tensor):
        return cols * gain.float().unsqueeze(1)
    return cols * float(gain)


def _effective_unembed(ck: Checkpoint, anat: Anatomy) -> torch.Tensor:
    """Compute effective unembedding matrix U_eff of shape [vocab, hidden] including final norm gain.

    Also applies two exact invariances of Transformer readout:
    1. If the final norm is LayerNorm, inputs to the final norm have zero mean across hidden dims,
       so we zero-center along the hidden dimension.
    2. Since softmax(logits) is invariant to adding a constant across all vocabulary tokens,
       we subtract the mean vocabulary row `U.mean(dim=0)` so `U_eff @ v` measures the true
       differential logit shift of each token relative to the vocabulary mean (eliminating the
       anisotropy cone in pretrained models like GPT-2).
    """
    U = ck.get(anat.unembed).float()
    if anat.final_norm:
        g = effective_gain(ck, anat.final_norm)
        if isinstance(g, torch.Tensor):
            U = U * g.float().unsqueeze(0)
        else:
            U = U * float(g)
    if anat.norm_label == "LayerNorm":
        U = U - U.mean(dim=1, keepdim=True)
    if U.shape[0] > 1:
        U = U - U.mean(dim=0, keepdim=True)
    return U


def _input_embed(ck: Checkpoint, anat: Anatomy) -> torch.Tensor:
    """Return centered input token embedding matrix E_in of shape [vocab, hidden].

    Subtracting the mean token embedding ensures `E_in @ k` measures how strongly a specific
    token triggers a neuron above the baseline average token embedding.
    """
    embed_name = anat.embed or anat.unembed
    E = ck.get(embed_name).float()
    if E.shape[0] > 1:
        E = E - E.mean(dim=0, keepdim=True)
    return E


def discover_layer_readers(ck: Checkpoint, anat: Anatomy, l: int) -> dict[str, Any]:
    """Find input-reading tensors (`mlp_gate`, `mlp_up`, `attn_q`, `attn_k`, `attn_v`) and pre-norms for layer `l`."""
    mlp_gate = _find_layer_tensor(anat, l, MLP_GATE_STEMS)
    mlp_up = _find_layer_tensor(anat, l, MLP_UP_STEMS)
    attn_v = _find_layer_tensor(anat, l, ATTN_V_STEMS)
    attn_q = _find_layer_tensor(anat, l, ATTN_Q_STEMS)
    attn_k = _find_layer_tensor(anat, l, ATTN_K_STEMS)
    return {
        "mlp_gate": mlp_gate,
        "mlp_up": mlp_up,
        "mlp_primary_in": mlp_gate or mlp_up,
        "attn_v": attn_v,
        "attn_q": attn_q,
        "attn_k": attn_k,
        "attn_primary_in": attn_v or attn_q or attn_k,
        "pre_mlp_gain": _find_pre_norm_gain(ck, anat, l, PRE_MLP_NORM_KEYS),
        "pre_attn_gain": _find_pre_norm_gain(ck, anat, l, PRE_ATTN_NORM_KEYS),
    }


def get_mlp_input_keys(ck: Checkpoint, anat: Anatomy, l: int, expected_units: int | None = None) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Return `[hidden, n_units]` effective input key vectors for layer `l`'s MLP neurons, plus metadata.

    For gated MLPs (SwiGLU / GeGLU) with both `gate_proj` and `up_proj`, returns the pre-norm-scaled
    `gate_proj` keys as the primary trigger keys, and also computes per-neuron gate-up cosine alignment.
    """
    rd = discover_layer_readers(ck, anat, l)
    primary = rd["mlp_primary_in"]
    if not primary:
        return None, {}

    W_in = ck.get(primary).float()
    K_primary = _input_to_residual_columns(W_in, anat.hidden, expected_units)
    K_eff = _apply_input_gain(K_primary, rd["pre_mlp_gain"])

    meta: dict[str, Any] = {
        "primary_in_tensor": primary,
        "gate_tensor": rd["mlp_gate"],
        "up_tensor": rd["mlp_up"],
    }

    if rd["mlp_gate"] and rd["mlp_up"] and rd["mlp_gate"] != rd["mlp_up"]:
        W_up = ck.get(rd["mlp_up"]).float()
        K_up = _input_to_residual_columns(W_up, anat.hidden, expected_units)
        if K_up.shape == K_primary.shape:
            K_up_eff = _apply_input_gain(K_up, rd["pre_mlp_gain"])
            cos_gu = torch.nn.functional.cosine_similarity(K_eff, K_up_eff, dim=0)
            meta["K_up_eff"] = K_up_eff
            meta["gate_up_cosine_mean"] = round(cos_gu.mean().item(), 5)
    return K_eff, meta


def _compute_svd_basis(W_res: torch.Tensor, svd_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = min(svd_rank, W_res.shape[0], W_res.shape[1])
    if max(W_res.shape) <= 2048:
        U_svd, S_svd, Vh_svd = torch.linalg.svd(W_res, full_matrices=False)
        return U_svd[:, :r].contiguous(), S_svd[:r].contiguous(), Vh_svd[:r, :].t().contiguous()
    U_r, S_r, V_r = torch.svd_lowrank(W_res, q=min(r + 4, min(W_res.shape)), niter=2)
    return U_r[:, :r].contiguous(), S_r[:r].contiguous(), V_r[:, :r].contiguous()


def build_index(
    model_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    top_neurons_per_layer: int = 32,
    top_tokens_per_unit: int = 8,
    svd_rank: int = 8,
    anatomy_overrides: dict | None = None,
) -> dict[str, Any]:
    """Build a complete Weight-Space Index (both Input-Side Readers and Output-Side Writers)."""
    model_path = Path(model_dir)
    ck = Checkpoint(model_path)
    anat = Anatomy(ck, overrides=anatomy_overrides)
    vocab_map = _load_vocab(model_path, anat.vocab)
    roles = token_roles(model_path)

    # 1. Output-side Effective Unembedding U_eff [vocab, hidden] & Input-side Embedding E_in [vocab, hidden]
    U_eff = _effective_unembed(ck, anat)
    E_in = _input_embed(ck, anat)
    unembed_row_norms = U_eff.norm(dim=1)
    embed_row_norms = E_in.norm(dim=1)

    layers_index: dict[str, Any] = {}
    token_promoters: dict[int, list[dict[str, Any]]] = {}
    token_suppressors: dict[int, list[dict[str, Any]]] = {}
    token_triggers: dict[int, list[dict[str, Any]]] = {}

    saved_tensors: dict[str, torch.Tensor] = {
        "unembed_row_norms": unembed_row_norms.cpu(),
        "embed_row_norms": embed_row_norms.cpu(),
    }

    # Store writer and reader SVD bases in residual space [hidden, r] for causal circuit graph construction
    layer_write_bases: dict[int, dict[str, torch.Tensor]] = {}
    layer_read_bases: dict[int, dict[str, torch.Tensor]] = {}

    for l in sorted(anat.layers):
        layer_info: dict[str, Any] = {
            "layer": l,
            "norms": {},
            "modules": {},
            "readers": {},
        }

        # Index block norm gains
        for norm_kind, norm_tensor_name in anat.norms(l).items():
            g = effective_gain(ck, norm_tensor_name)
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

        # Discover input-side readers for this layer
        rd_info = discover_layer_readers(ck, anat, l)
        layer_write_bases[l] = {}
        layer_read_bases[l] = {}

        # Index Attention Input Reader Subspace (V / Q / K)
        for attn_in_role in ("attn_v", "attn_q", "attn_k"):
            tname_in = rd_info.get(attn_in_role)
            if tname_in:
                W_ain = ck.get(tname_in).float()
                if W_ain.dim() == 2 and anat.hidden in W_ain.shape:
                    K_ain = _input_to_residual_columns(W_ain, anat.hidden)
                    K_ain_eff = _apply_input_gain(K_ain, rd_info["pre_attn_gain"])
                    U_read, S_read, _ = _compute_svd_basis(K_ain_eff, svd_rank)
                    saved_tensors[f"layer.{l}.{attn_in_role}.U_read"] = U_read.cpu()
                    saved_tensors[f"layer.{l}.{attn_in_role}.S_read"] = S_read.cpu()
                    if attn_in_role == "attn_v" or "attn_in" not in layer_read_bases[l]:
                        layer_read_bases[l]["attn_in"] = U_read

                    in_logits = E_in @ U_read[:, 0]
                    k_tok = min(top_tokens_per_unit, anat.vocab)
                    t_vals, t_idx = torch.topk(in_logits, k=k_tok, largest=True)
                    layer_info["readers"][attn_in_role] = {
                        "tensor": tname_in,
                        "shape": list(W_ain.shape),
                        "spectral_norm": round(S_read[0].item(), 5) if S_read.numel() else 0.0,
                        "dominant_trigger_tokens": [
                            {"token_id": int(tid), "token": vocab_map[int(tid)], "score": round(val.item(), 4)}
                            for val, tid in zip(t_vals, t_idx)
                        ],
                    }

        # Preload MLP input keys `[hidden, n_units]` so we can pair each MLP neuron's Input Key with its Output Value!
        mlp_out_tname = anat.mlp_out(l)
        expected_mlp_units = None
        if mlp_out_tname:
            W_mo = ck.get(mlp_out_tname)
            out_ax_mo = anat.out_axis(mlp_out_tname)
            expected_mlp_units = W_mo.shape[1 - out_ax_mo] if W_mo.dim() == 2 else None

        K_mlp_eff, mlp_in_meta = get_mlp_input_keys(ck, anat, l, expected_mlp_units)
        if K_mlp_eff is not None:
            U_mlp_read, S_mlp_read, _ = _compute_svd_basis(K_mlp_eff, svd_rank)
            saved_tensors[f"layer.{l}.mlp_in.U_read"] = U_mlp_read.cpu()
            saved_tensors[f"layer.{l}.mlp_in.S_read"] = S_mlp_read.cpu()
            saved_tensors[f"layer.{l}.mlp_in.key_norms"] = K_mlp_eff.norm(dim=0).cpu()
            layer_read_bases[l]["mlp_in"] = U_mlp_read

            in_logits = E_in @ U_mlp_read[:, 0]
            k_tok = min(top_tokens_per_unit, anat.vocab)
            t_vals, t_idx = torch.topk(in_logits, k=k_tok, largest=True)
            layer_info["readers"]["mlp_in"] = {
                "primary_tensor": mlp_in_meta.get("primary_in_tensor"),
                "gate_tensor": mlp_in_meta.get("gate_tensor"),
                "up_tensor": mlp_in_meta.get("up_tensor"),
                "gate_up_cosine_mean": mlp_in_meta.get("gate_up_cosine_mean"),
                "spectral_norm": round(S_mlp_read[0].item(), 5) if S_mlp_read.numel() else 0.0,
                "dominant_trigger_tokens": [
                    {"token_id": int(tid), "token": vocab_map[int(tid)], "score": round(val.item(), 4)}
                    for val, tid in zip(t_vals, t_idx)
                ],
            }

        # Index residual-writing modules: attn_out and mlp_out
        module_targets: list[tuple[str, str]] = []
        if anat.attn_out(l):
            module_targets.append(("attn_out", anat.attn_out(l)))
        if anat.mlp_out(l):
            module_targets.append(("mlp_out", anat.mlp_out(l)))

        for mod_role, tensor_name in module_targets:
            W_raw = ck.get(tensor_name).float()
            out_ax = anat.out_axis(tensor_name)
            W_res = _to_residual_columns(W_raw, out_ax)  # [hidden, n_units]

            fro_norm = W_res.norm().item()
            unit_norms = W_res.norm(dim=0)  # [n_units]

            U_r, S_r, _ = _compute_svd_basis(W_res, svd_rank)
            r = U_r.shape[1]
            sigma_1 = S_r[0].item() if S_r.numel() > 0 else 0.0
            stable_rank = (fro_norm ** 2) / max(sigma_1 ** 2, 1e-12)

            saved_tensors[f"layer.{l}.{mod_role}.U"] = U_r.cpu()
            saved_tensors[f"layer.{l}.{mod_role}.S"] = S_r.cpu()
            saved_tensors[f"layer.{l}.{mod_role}.unit_norms"] = unit_norms.cpu()
            layer_write_bases[l][mod_role] = U_r

            svd_logits = U_eff @ U_r  # [vocab, r]
            k_tok = min(top_tokens_per_unit, anat.vocab)
            svd_directions = []
            for i in range(r):
                col_logits = svd_logits[:, i]
                pos_vals, pos_idx = torch.topk(col_logits, k=k_tok, largest=True)
                neg_vals, neg_idx = torch.topk(col_logits, k=k_tok, largest=False)
                svd_directions.append({
                    "component": i,
                    "singular_value": round(S_r[i].item(), 5),
                    "energy_fraction": round((S_r[i].item() ** 2) / max(fro_norm ** 2, 1e-12), 5),
                    "positive_pole_tokens": [
                        {"token_id": int(tid), "token": vocab_map[int(tid)], "logit": round(val.item(), 4)}
                        for val, tid in zip(pos_vals, pos_idx)
                    ],
                    "negative_pole_tokens": [
                        {"token_id": int(tid), "token": vocab_map[int(tid)], "logit": round(val.item(), 4)}
                        for val, tid in zip(neg_vals, neg_idx)
                    ],
                })

            # Rank units by combined importance (output norm * input key norm if MLP, else output norm)
            has_paired_keys = (
                mod_role == "mlp_out"
                and K_mlp_eff is not None
                and K_mlp_eff.shape[1] == W_res.shape[1]
            )
            if has_paired_keys:
                key_norms = K_mlp_eff.norm(dim=0)
                importance = unit_norms * key_norms.clamp_min(1e-6)
            else:
                key_norms = None
                importance = unit_norms

            n_top = min(top_neurons_per_layer, W_res.shape[1])
            _, top_u_idx = torch.topk(importance, k=n_top)
            W_top = W_res[:, top_u_idx]  # [hidden, n_top]
            unit_logits = U_eff @ W_top  # [vocab, n_top]

            if has_paired_keys:
                K_top = K_mlp_eff[:, top_u_idx]  # [hidden, n_top]
                trigger_logits = E_in @ K_top    # [vocab, n_top]
            else:
                trigger_logits = None

            indexed_units = []
            for j in range(n_top):
                uid = int(top_u_idx[j].item())
                unorm = unit_norms[uid].item()
                ulog = unit_logits[:, j]
                p_vals, p_idx = torch.topk(ulog, k=k_tok, largest=True)
                s_vals, s_idx = torch.topk(ulog, k=k_tok, largest=False)

                promoted = [
                    {"token_id": int(tid), "token": vocab_map[int(tid)], "logit": round(val.item(), 4)}
                    for val, tid in zip(p_vals, p_idx)
                ]
                suppressed = [
                    {"token_id": int(tid), "token": vocab_map[int(tid)], "logit": round(val.item(), 4)}
                    for val, tid in zip(s_vals, s_idx)
                ]

                unit_entry: dict[str, Any] = {
                    "unit": uid,
                    "out_norm": round(unorm, 5),
                    "promoted": promoted,
                    "suppressed": suppressed,
                }

                if trigger_logits is not None and key_norms is not None:
                    knorm = key_norms[uid].item()
                    tlog = trigger_logits[:, j]
                    tr_vals, tr_idx = torch.topk(tlog, k=k_tok, largest=True)
                    in_vals, in_idx = torch.topk(tlog, k=k_tok, largest=False)
                    triggers = [
                        {"token_id": int(tid), "token": vocab_map[int(tid)], "score": round(val.item(), 4)}
                        for val, tid in zip(tr_vals, tr_idx)
                    ]
                    inhibitors = [
                        {"token_id": int(tid), "token": vocab_map[int(tid)], "score": round(val.item(), 4)}
                        for val, tid in zip(in_vals, in_idx)
                    ]
                    in_toks_str = ", ".join(t["token"] for t in triggers[:3])
                    out_toks_str = ", ".join(t["token"] for t in promoted[:3])
                    unit_entry["key_norm"] = round(knorm, 5)
                    unit_entry["triggers"] = triggers
                    unit_entry["inhibitors"] = inhibitors
                    unit_entry["transduction_rule"] = f"[{in_toks_str}] -> [{out_toks_str}]"

                    for item in triggers[:4]:
                        token_triggers.setdefault(item["token_id"], []).append({
                            "layer": l,
                            "module": "mlp_in",
                            "tensor": mlp_in_meta.get("primary_in_tensor"),
                            "unit": uid,
                            "trigger_score": item["score"],
                            "key_norm": round(knorm, 5),
                            "top_outputs": [t["token"] for t in promoted[:3]],
                        })

                indexed_units.append(unit_entry)

                for item in promoted[:4]:
                    token_promoters.setdefault(item["token_id"], []).append({
                        "layer": l,
                        "module": mod_role,
                        "tensor": tensor_name,
                        "unit": uid,
                        "unit_axis_in_tensor": 1 - out_ax if W_raw.dim() == 2 else out_ax,
                        "logit": item["logit"],
                        "unit_norm": round(unorm, 5),
                    })
                for item in suppressed[:4]:
                    token_suppressors.setdefault(item["token_id"], []).append({
                        "layer": l,
                        "module": mod_role,
                        "tensor": tensor_name,
                        "unit": uid,
                        "unit_axis_in_tensor": 1 - out_ax if W_raw.dim() == 2 else out_ax,
                        "logit": item["logit"],
                        "unit_norm": round(unorm, 5),
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
                "top_units": indexed_units,
            }

        layers_index[str(l)] = layer_info

    # Sort inverted token maps by strength of effect (now including input-side triggers!)
    inverted_tokens: dict[str, dict[str, Any]] = {}
    all_indexed_tids = set(token_promoters.keys()) | set(token_suppressors.keys()) | set(token_triggers.keys())
    for tid in sorted(all_indexed_tids):
        proms = sorted(token_promoters.get(tid, []), key=lambda x: x["logit"], reverse=True)[:12]
        supps = sorted(token_suppressors.get(tid, []), key=lambda x: x["logit"])[:12]
        trigs = sorted(token_triggers.get(tid, []), key=lambda x: x["trigger_score"], reverse=True)[:12]
        inverted_tokens[str(tid)] = {
            "token_id": tid,
            "token": vocab_map[tid],
            "unembed_norm": round(unembed_row_norms[tid].item(), 5),
            "embed_norm": round(embed_row_norms[tid].item(), 5),
            "top_triggered_neurons": trigs,
            "top_promoters": proms,
            "top_suppressors": supps,
        }

    # Compute directed causal Write -> Read cross-layer circuits!
    # Measures how strongly Layer A's output writer subspace U_write feeds into Layer B's input reader subspace U_read
    circuit_edges: list[dict[str, Any]] = []
    sorted_layers = sorted(anat.layers)
    for idx_a, la in enumerate(sorted_layers):
        for lb in sorted_layers[idx_a + 1:]:
            for mod_write, U_w in layer_write_bases[la].items():
                # 1. Directed Write(la) -> Read(lb) causal edges
                for mod_read, U_r in layer_read_bases[lb].items():
                    overlap = U_w.t() @ U_r
                    r_min = max(1, min(U_w.shape[1], U_r.shape[1]))
                    score = overlap.norm().item() / math.sqrt(r_min)
                    max_cos = overlap.abs().max().item() if overlap.numel() else 0.0
                    circuit_edges.append({
                        "edge_type": "write_to_read",
                        "src_layer": la,
                        "src_module": mod_write,
                        "dst_layer": lb,
                        "dst_module": mod_read,
                        "subspace_overlap": round(score, 4),
                        "max_direction_cosine": round(max_cos, 4),
                    })
    circuit_edges.sort(key=lambda e: e["max_direction_cosine"], reverse=True)

    index_data: dict[str, Any] = {
        "version": "0.2.0",
        "model_dir": str(model_path.resolve()),
        "anatomy": anat.summary(),
        "token_roles": roles,
        "config": {
            "top_neurons_per_layer": top_neurons_per_layer,
            "top_tokens_per_unit": top_tokens_per_unit,
            "svd_rank": svd_rank,
        },
        "layers": layers_index,
        "inverted_tokens": inverted_tokens,
        "top_circuit_edges": circuit_edges[:64],
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "index.json").write_text(json.dumps(index_data, indent=2), encoding="utf-8")
        save_file(saved_tensors, str(out_path / "subspaces.safetensors"))

    return index_data
