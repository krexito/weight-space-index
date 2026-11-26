"""Static weight-diff-to-vocabulary auditing (`wsi_lora_audit_v1`).

Translates LoRA adapter weights (`B x A`) or dense checkpoint deltas (`W_after - W_before`)
into ranked vocabulary tokens with zero forward passes by:
1. Taking the residual-stream SVD directions (`top_sv`, default 4) from every matrix
   (left singular vectors `U` for writers, right singular vectors `V` for readers —
   using exact QR-SVD on `B, A` without materializing `B @ A`).
2. Computing cosine similarity of each unit direction with every row of the unembedding
   matrix, converted to per-direction z-scores across the vocabulary.
3. Aggregating `score(t) = sum_d (sigma_d ** 2) * (|z_{d,t}| ** 4)` (sign-invariant,
   outlier-amplifying, energy-weighted), alongside a chat-role-filtered `top_content_tokens`
   view, reader vs. writer token splits, and per-token layer/module attribution.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .delta_xray import (
    _WeightSource,
    _decode_tok_safe,
    _discover_pytorch_bin_files,
    _is_aux_multimodal_or_mtp,
    _load_pytorch_bin_shard,
    classify_param_role,
    classify_submatrix_tag,
    extract_layer_idx,
)
def _resolve_device(device: torch.device | str) -> torch.device:
    return torch.device(device)

LORA_AUDIT_SCHEMA_VERSION = "wsi_lora_audit_v1"

_LORA_A_RE = re.compile(r"^(?P<base>.+?)\.lora_A(?:\.(?P<adapter>[^.]+))?\.weight$")
_SPECIAL_TAG_RE = re.compile(r"^(?:<\|[^|]+\|>|<[^>]{1,32}>|\[[A-Z0-9_/-]{1,24}\])$")

_ROLE_AND_FORMAT_WORDS = frozenset(
    {
        "assistant",
        "assistants",
        "asistente",
        "asistentes",
        "助手",
        "user",
        "users",
        "usuario",
        "usuarios",
        "utilisateur",
        "utilisateurs",
        "用户",
        "ユーザー",
        "пользователь",
        "system",
        "systems",
        "sys",
        "sistema",
        "sistemas",
        "système",
        "systèmes",
        "システム",
        "系统",
        "系统的",
        "系統",
        "систем",
        "система",
        "системы",
        "human",
        "humans",
        "bot",
        "chatbot",
        "im_start",
        "im_end",
        "eot",
        "eot_id",
        "start_header_id",
        "end_header_id",
        "bos",
        "eos",
        "pad",
        "unk",
        "cls",
        "sep",
        "mask",
        "think",
        "thought",
    }
)


@dataclass
class ResidualDirection:
    """Single singular direction living in the model's residual stream."""

    vec: torch.Tensor
    strength: float
    layer: int | None
    submatrix: str
    side: str
    param_name: str
    sv_rank: int = 0


def _classify_residual_side(
    param_name: str,
    d_out: int,
    d_in: int,
    hidden_size: int,
    *,
    is_conv1d: bool = False,
) -> tuple[str | None, str, int | None]:
    """Return `(side, submatrix_tag, layer_idx)` where `side` is `'writer'`, `'reader'`, or `None`."""
    if d_out <= 0 or d_in <= 0:
        return (None, "other", None)
    if is_conv1d:
        d_out, d_in = d_in, d_out

    role = classify_param_role(param_name, (d_out, d_in))
    submat = classify_submatrix_tag(param_name, (d_out, d_in))
    li = extract_layer_idx(param_name)
    if role in ("norms", "embed_head", "moe_router"):
        return (None, submat, li)

    low = param_name.lower()
    if d_out == hidden_size and d_in != hidden_size:
        return ("writer", submat, li)
    if d_in == hidden_size and d_out != hidden_size:
        return ("reader", submat, li)
    if d_out == hidden_size and d_in == hidden_size:
        if role in ("attn_out", "mlp_out") or submat in ("o_proj", "down_proj") or any(
            t in low for t in ("o_proj", "out_proj", "c_proj", ".w2.", "fc2", "dense_4h_to_h", "output_linear")
        ):
            return ("writer", submat, li)
        if role in ("attn_qkv", "mlp_in") or submat in (
            "q_proj",
            "k_proj",
            "v_proj",
            "qkv_proj",
            "gate_proj",
            "up_proj",
            "gate_up_proj",
        ) or any(t in low for t in ("q_proj", "k_proj", "v_proj", "qkv", "in_proj", "gate_proj", "up_proj")):
            return ("reader", submat, li)
    return (None, submat, li)


def _load_adapter_tensors_and_config(
    adapter_source: str | Path | Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load LoRA weight tensors and optional `adapter_config.json` without touching base weights."""
    if isinstance(adapter_source, Mapping):
        tensors = {str(k): v for k, v in adapter_source.items() if isinstance(v, torch.Tensor)}
        return tensors, {}

    p = Path(adapter_source)
    cfg: dict[str, Any] = {}
    weight_file: Path | None = None

    if p.is_file():
        weight_file = p
        cfg_cand = p.parent / "adapter_config.json"
        if cfg_cand.exists():
            cfg = json.loads(cfg_cand.read_text(encoding="utf-8"))
    elif p.is_dir():
        cfg_cand = p / "adapter_config.json"
        if cfg_cand.exists():
            cfg = json.loads(cfg_cand.read_text(encoding="utf-8"))
        for fname in ("adapter_model.safetensors", "adapter_model.bin"):
            cand = p / fname
            if cand.exists():
                weight_file = cand
                break
        if weight_file is None:
            sf = sorted(p.glob("*.safetensors"))
            bf = sorted(p.glob("*.bin")) + sorted(p.glob("*.pt"))
            weight_file = (sf[0] if sf else (bf[0] if bf else None))
    else:
        raise FileNotFoundError(f"Adapter source not found: {p}")

    if weight_file is None or not weight_file.exists():
        raise FileNotFoundError(f"No LoRA weights (.safetensors or .bin) found in {p}")

    if weight_file.suffix == ".safetensors":
        from safetensors import safe_open

        tensors = {}
        with safe_open(str(weight_file), "pt") as fh:
            for k in fh.keys():
                tensors[str(k)] = fh.get_tensor(k)
        return tensors, cfg

    tensors, _ = _load_pytorch_bin_shard(weight_file)
    return tensors, cfg


def _resolve_lora_scale(
    base_key: str,
    rank: int,
    cfg: Mapping[str, Any],
    override_scale: float | None,
) -> float:
    if override_scale is not None:
        return float(override_scale)
    if not cfg:
        return 1.0

    r_eff = int(cfg.get("r") or rank)
    alpha_eff = float(cfg.get("lora_alpha", r_eff))

    rank_pat = cfg.get("rank_pattern")
    if isinstance(rank_pat, Mapping):
        for pat, rv in rank_pat.items():
            if base_key.endswith(str(pat)) or str(pat) in base_key:
                r_eff = int(rv)
                break

    alpha_pat = cfg.get("alpha_pattern")
    if isinstance(alpha_pat, Mapping):
        for pat, av in alpha_pat.items():
            if base_key.endswith(str(pat)) or str(pat) in base_key:
                alpha_eff = float(av)
                break

    if r_eff <= 0:
        return 1.0
    if bool(cfg.get("use_rslora", False)):
        return alpha_eff / math.sqrt(float(r_eff))
    return alpha_eff / float(r_eff)


@torch.no_grad()
def extract_lora_residual_directions(
    adapter_source: str | Path | Mapping[str, torch.Tensor],
    *,
    hidden_size: int,
    top_sv: int = 4,
    lora_scale: float | None = None,
    device: torch.device | str | None = None,
) -> list[ResidualDirection]:
    """Extract top-k residual-stream singular directions for every `(lora_B, lora_A)` pair via exact QR-SVD."""
    tensors, cfg = _load_adapter_tensors_and_config(adapter_source)
    dev = _resolve_device(device) if device not in (None, "auto") else torch.device("cpu")
    directions: list[ResidualDirection] = []
    top_sv = max(1, int(top_sv))

    for k_a in sorted(tensors.keys()):
        m = _LORA_A_RE.match(k_a)
        if not m:
            continue
        base_name = m.group("base")
        adapter_name = m.group("adapter")
        k_b = f"{base_name}.lora_B.{adapter_name}.weight" if adapter_name else f"{base_name}.lora_B.weight"
        if k_b not in tensors:
            alt_b = f"{base_name}.lora_B.weight"
            if alt_b in tensors:
                k_b = alt_b
            else:
                continue

        A = tensors[k_a]
        B = tensors[k_b]
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            continue
        if A.dim() != 2 or B.dim() != 2 or A.shape[0] != B.shape[1]:
            continue

        d_out, r_rank = int(B.shape[0]), int(B.shape[1])
        d_in = int(A.shape[1])
        if r_rank == 0:
            continue

        side, submat, li = _classify_residual_side(base_name, d_out, d_in, int(hidden_size))
        if side is None:
            continue

        scale = _resolve_lora_scale(base_name, r_rank, cfg, lora_scale)
        if abs(scale) <= 1e-12:
            continue

        Bf = B.to(device=dev, dtype=torch.float32)
        Af = A.to(device=dev, dtype=torch.float32)
        Q_b, R_b = torch.linalg.qr(Bf, mode="reduced")
        Q_a, R_a = torch.linalg.qr(Af.t(), mode="reduced")
        M = (R_b @ R_a.t()) * float(scale)
        U_m, S, Vh_m = torch.linalg.svd(M, full_matrices=False)

        k_keep = min(top_sv, int(S.shape[0]))
        if k_keep <= 0 or float(S[0].item()) <= 1e-12:
            continue

        if side == "writer":
            vecs = (Q_b @ U_m[:, :k_keep]).t()
        else:
            vecs = (Q_a @ Vh_m[:k_keep, :].t()).t()
        vecs = F.normalize(vecs, dim=-1, eps=1e-12).detach().cpu()

        for rank_i in range(k_keep):
            sv_val = float(S[rank_i].item())
            if sv_val <= 1e-12:
                break
            directions.append(
                ResidualDirection(
                    vec=vecs[rank_i],
                    strength=sv_val,
                    layer=li,
                    submatrix=submat,
                    side=side,
                    param_name=base_name,
                    sv_rank=rank_i,
                )
            )
    return directions


@torch.no_grad()
def _top_svd_delta(dW: torch.Tensor, top_sv: int, *, oversample: int = 8, niter: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-`top_sv` singular triplets `(U[:, :k], S[:k], Vh[:k, :])` deterministically."""
    r, c = int(dW.shape[0]), int(dW.shape[1])
    m = min(r, c)
    k = min(max(1, int(top_sv)), m)
    if m <= 64 or k >= m - 4:
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k, :]

    q = min(m, k + max(2, int(oversample)))
    gen = torch.Generator(device=dW.device)
    gen.manual_seed(0)
    Omega = torch.randn((c, q), generator=gen, device=dW.device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(dW @ Omega, mode="reduced")
    for _ in range(max(1, int(niter))):
        Q, _ = torch.linalg.qr(dW.t() @ Q, mode="reduced")
        Q, _ = torch.linalg.qr(dW @ Q, mode="reduced")
    B_small = Q.t() @ dW
    U_b, S, Vh = torch.linalg.svd(B_small, full_matrices=False)
    U = Q @ U_b
    return U[:, :k], S[:k], Vh[:k, :]


@torch.no_grad()
def extract_matrix_residual_directions(
    dW: torch.Tensor,
    *,
    param_name: str,
    hidden_size: int,
    top_sv: int = 4,
    is_conv1d: bool = False,
) -> list[ResidualDirection]:
    """Extract top-k residual-stream singular directions from a single dense weight-delta matrix."""
    if dW.dim() != 2:
        return []
    r_raw, c_raw = int(dW.shape[0]), int(dW.shape[1])
    side, submat, li = _classify_residual_side(param_name, r_raw, c_raw, int(hidden_size), is_conv1d=is_conv1d)
    if side is None:
        return []

    mat = dW.t() if is_conv1d else dW
    mat_f = mat.detach().float()
    if float(mat_f.norm().item()) <= 1e-12:
        return []

    U, S, Vh = _top_svd_delta(mat_f, top_sv)
    k_keep = min(int(top_sv), int(S.shape[0]))
    if k_keep <= 0 or float(S[0].item()) <= 1e-12:
        return []

    vecs = U[:, :k_keep].t() if side == "writer" else Vh[:k_keep, :]
    vecs = F.normalize(vecs, dim=-1, eps=1e-12).detach().cpu()

    out: list[ResidualDirection] = []
    for rank_i in range(k_keep):
        sv_val = float(S[rank_i].item())
        if sv_val <= 1e-12:
            break
        out.append(
            ResidualDirection(
                vec=vecs[rank_i],
                strength=sv_val,
                layer=li,
                submatrix=submat,
                side=side,
                param_name=param_name,
                sv_rank=rank_i,
            )
        )
    return out


@torch.no_grad()
def extract_dense_diff_residual_directions(
    before_source: nn.Module | Mapping[str, torch.Tensor] | str | Path | _WeightSource,
    after_source: nn.Module | Mapping[str, torch.Tensor] | str | Path | _WeightSource,
    *,
    hidden_size: int,
    top_sv: int = 4,
    device: torch.device | str | None = None,
) -> list[ResidualDirection]:
    """Stream `(before, after)` checkpoints and extract top-k reader/writer SVD directions across all layers."""
    dev = _resolve_device(device) if device not in (None, "auto") else torch.device("cpu")
    own_b = not isinstance(before_source, _WeightSource)
    own_a = not isinstance(after_source, _WeightSource)
    src_b = before_source if isinstance(before_source, _WeightSource) else _WeightSource(before_source)
    src_a = after_source if isinstance(after_source, _WeightSource) else _WeightSource(after_source)
    directions: list[ResidualDirection] = []

    try:
        for k in src_b.keys():
            if _is_aux_multimodal_or_mtp(k):
                continue
            sb = src_b.shape(k)
            sa = src_a.shape(k)
            if sb is None or sa is None or len(sb) != 2 or tuple(sb) != tuple(sa):
                continue
            side, _, _ = _classify_residual_side(k, int(sb[0]), int(sb[1]), int(hidden_size))
            if side is None:
                continue
            wb = src_b.get(k)
            wa = src_a.get(k)
            if wb is None or wa is None:
                continue
            dW = wa.to(device=dev, dtype=torch.float32) - wb.to(device=dev, dtype=torch.float32)
            directions.extend(
                extract_matrix_residual_directions(
                    dW,
                    param_name=k,
                    hidden_size=int(hidden_size),
                    top_sv=top_sv,
                )
            )
    finally:
        if own_b:
            src_b.close()
        if own_a:
            src_a.close()
    return directions


_UNEMBED_KEY_CANDIDATES = (
    "lm_head.weight",
    "model.lm_head.weight",
    "language_model.lm_head.weight",
    "cls.predictions.decoder.weight",
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "embed_tokens.weight",
    "transformer.wte.weight",
    "wte.weight",
    "backbone.embeddings.weight",
    "backbone.word_embeddings.weight",
    "model.decoder.embed_tokens.weight",
)

_NORM_KEY_CANDIDATES = (
    "model.norm.weight",
    "language_model.model.norm.weight",
    "model.embedding_norm.weight",
    "transformer.ln_f.weight",
    "backbone.norm_f.weight",
    "backbone.final_layernorm.weight",
    "model.decoder.final_layer_norm.weight",
    "norm.weight",
)


def _find_key_in_dir(root: Path, candidates: Sequence[str]) -> tuple[str | None, Path | None, str | None]:
    """Locate the first existing tensor key and its exact shard file without reading unrelated shards."""
    sf_index = root / "model.safetensors.index.json"
    if sf_index.exists():
        wmap = json.loads(sf_index.read_text(encoding="utf-8")).get("weight_map", {})
        for cand in candidates:
            if cand in wmap:
                fpath = root / str(wmap[cand])
                if fpath.exists():
                    return cand, fpath, "safetensors"
        # Suffix match fallback for nested wrapper names
        for cand in candidates:
            for wk, wf in wmap.items():
                if wk.endswith(f".{cand}") or wk == cand:
                    fpath = root / str(wf)
                    if fpath.exists():
                        return wk, fpath, "safetensors"

    sf_files = sorted(root.glob("*.safetensors"))
    if sf_files:
        from safetensors import safe_open

        for sf in sf_files:
            with safe_open(str(sf), "pt") as fh:
                keys = set(fh.keys())
                for cand in candidates:
                    if cand in keys:
                        return cand, sf, "safetensors"
        for sf in sf_files:
            with safe_open(str(sf), "pt") as fh:
                keys = list(fh.keys())
                for cand in candidates:
                    for wk in keys:
                        if wk.endswith(f".{cand}"):
                            return wk, sf, "safetensors"

    bin_index = root / "pytorch_model.bin.index.json"
    if bin_index.exists():
        wmap = json.loads(bin_index.read_text(encoding="utf-8")).get("weight_map", {})
        for cand in candidates:
            if cand in wmap:
                fpath = root / str(wmap[cand])
                if fpath.exists():
                    return cand, fpath, "bin"

    for bf in _discover_pytorch_bin_files(root):
        sd, _ = _load_pytorch_bin_shard(bf)
        for cand in candidates:
            if cand in sd:
                return cand, bf, "bin"
        for cand in candidates:
            for wk in sd.keys():
                if wk.endswith(f".{cand}"):
                    return wk, bf, "bin"
    return None, None, None


def _load_tensor_from_file(key: str, path: Path, fmt: str) -> torch.Tensor:
    if fmt == "safetensors":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as fh:
            return fh.get_tensor(key)
    sd, _ = _load_pytorch_bin_shard(path)
    return sd[key]


def _load_tokenizer_from_dir(root: Path) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
    except Exception:
        pass
    tok_json = root / "tokenizer.json"
    if tok_json.exists():
        try:
            from transformers import PreTrainedTokenizerFast

            return PreTrainedTokenizerFast(tokenizer_file=str(tok_json))
        except Exception:
            pass
        try:
            from tokenizers import Tokenizer

            return Tokenizer.from_file(str(tok_json))
        except Exception:
            pass
    return None


@torch.no_grad()
def _apply_optional_final_norm(unembed: torch.Tensor, norm_w: torch.Tensor | None) -> torch.Tensor:
    out = unembed.detach().float()
    if norm_w is None or norm_w.dim() != 1 or norm_w.shape[0] != out.shape[1]:
        return out
    nw = norm_w.detach().float().to(out.device)
    # Zero-centered RMSNorm weights (Qwen3.5/Qwen3-Next) center around 0.0 (`1 + w`), standard RMSNorm around 1.0.
    eff = (1.0 + nw) if abs(float(nw.mean().item())) < 0.2 else nw
    return out * eff.unsqueeze(0)


@torch.no_grad()
def load_unembedding_and_tokenizer(
    base_model: nn.Module | str | Path,
    *,
    tokenizer: Any | None = None,
    apply_final_norm: bool = True,
) -> tuple[torch.Tensor, Any | None, dict[str, Any]]:
    """Load ONLY the unembedding matrix (+ optional final norm) and tokenizer from a checkpoint or module."""
    if isinstance(base_model, nn.Module):
        emb: torch.Tensor | None = None
        src_key = "lm_head.weight"
        out_mod = base_model.get_output_embeddings() if hasattr(base_model, "get_output_embeddings") else None
        if out_mod is not None and isinstance(getattr(out_mod, "weight", None), torch.Tensor):
            emb = out_mod.weight.detach()
        else:
            in_mod = base_model.get_input_embeddings() if hasattr(base_model, "get_input_embeddings") else None
            if in_mod is not None and isinstance(getattr(in_mod, "weight", None), torch.Tensor):
                emb = in_mod.weight.detach()
                src_key = "embed_tokens.weight"
        if emb is None:
            for cand in _UNEMBED_KEY_CANDIDATES:
                for pname, p in base_model.named_parameters():
                    if pname == cand or pname.endswith(f".{cand}"):
                        emb = p.detach()
                        src_key = pname
                        break
                if emb is not None:
                    break
        if emb is None or emb.dim() != 2:
            raise ValueError("Could not locate 2D unembedding/embedding tensor on module")

        norm_w: torch.Tensor | None = None
        norm_key: str | None = None
        if apply_final_norm:
            pdict = dict(base_model.named_parameters())
            for nk in _NORM_KEY_CANDIDATES:
                if nk in pdict and pdict[nk].dim() == 1:
                    norm_w = pdict[nk].detach()
                    norm_key = nk
                    break

        u_eff = _apply_optional_final_norm(emb, norm_w if apply_final_norm else None).cpu()
        return u_eff, tokenizer, {
            "source": "module",
            "unembed_key": src_key,
            "final_norm_key": norm_key if apply_final_norm else None,
            "vocab_size": int(u_eff.shape[0]),
            "hidden_size": int(u_eff.shape[1]),
        }

    root = Path(base_model)
    if not root.exists():
        raise FileNotFoundError(f"Base model directory not found: {root}")

    u_key, u_path, u_fmt = _find_key_in_dir(root, _UNEMBED_KEY_CANDIDATES)
    if u_key is None or u_path is None or u_fmt is None:
        raise FileNotFoundError(f"Could not find `lm_head.weight` or token embeddings in {root}")
    emb = _load_tensor_from_file(u_key, u_path, u_fmt)

    norm_w = None
    norm_key = None
    if apply_final_norm:
        n_key, n_path, n_fmt = _find_key_in_dir(root, _NORM_KEY_CANDIDATES)
        if n_key is not None and n_path is not None and n_fmt is not None:
            cand_n = _load_tensor_from_file(n_key, n_path, n_fmt)
            if cand_n.dim() == 1 and cand_n.shape[0] == emb.shape[1]:
                norm_w = cand_n
                norm_key = n_key

    tok_out = tokenizer if tokenizer is not None else _load_tokenizer_from_dir(root)
    u_eff = _apply_optional_final_norm(emb, norm_w if apply_final_norm else None).cpu()
    return u_eff, tok_out, {
        "source": str(root),
        "unembed_key": u_key,
        "unembed_shard": u_path.name,
        "final_norm_key": norm_key,
        "vocab_size": int(u_eff.shape[0]),
        "hidden_size": int(u_eff.shape[1]),
    }


def _special_token_ids(tok: Any | None) -> set[int]:
    if tok is None:
        return set()
    ids: set[int] = set()
    all_sp = getattr(tok, "all_special_ids", None)
    if isinstance(all_sp, Iterable := (list, tuple, set)):
        for x in all_sp:
            if isinstance(x, int):
                ids.add(int(x))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(tok, attr, None)
        if isinstance(v, int):
            ids.add(int(v))
    return ids


def _strip_word(decoded_tok: str) -> str:
    return (decoded_tok or "").strip().lstrip("Ġ▁Ċċ \t\r\n").strip(" \t\r\n.,;:!?\"'()[]{}<>-_=+/*\\~@#$%^&|`")


def _is_explicit_role_word(decoded_tok: str) -> bool:
    cleaned = _strip_word(decoded_tok)
    return bool(cleaned and cleaned.lower() in _ROLE_AND_FORMAT_WORDS)


def _is_explicit_role_or_special(decoded_tok: str, token_id: int, special_ids: set[int]) -> bool:
    if int(token_id) in special_ids:
        return True
    s = (decoded_tok or "").strip()
    if _SPECIAL_TAG_RE.match(s):
        return True
    return _is_explicit_role_word(decoded_tok)


def _is_format_only_token(decoded_tok: str) -> bool:
    cleaned = _strip_word(decoded_tok)
    return (not cleaned) or cleaned.isdigit()


def _is_content_token(decoded_tok: str, token_id: int, special_ids: set[int]) -> bool:
    return (not _is_explicit_role_or_special(decoded_tok, token_id, special_ids)) and (
        not _is_format_only_token(decoded_tok)
    )


def _aggregate_direction_modules(directions: Sequence[ResidualDirection], top_n: int = 10) -> list[dict[str, Any]]:
    energy_by_mod: dict[tuple[int | None, str, str], float] = {}
    count_by_mod: dict[tuple[int | None, str, str], int] = {}
    tot_e = 0.0
    for d in directions:
        e = float(d.strength) ** 2
        key = (d.layer, d.submatrix, d.side)
        energy_by_mod[key] = energy_by_mod.get(key, 0.0) + e
        count_by_mod[key] = count_by_mod.get(key, 0) + 1
        tot_e += e
    tot_e = max(tot_e, 1e-12)
    ranked = sorted(energy_by_mod.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_n))]
    return [
        {
            "layer": k[0],
            "submatrix": k[1],
            "side": k[2],
            "energy_share": float(e / tot_e),
            "sv_energy": float(e),
            "n_dirs": int(count_by_mod[k]),
        }
        for k, e in ranked
    ]


@torch.no_grad()
def score_residual_directions_against_vocab(
    directions: Sequence[ResidualDirection],
    unembed: torch.Tensor,
    tokenizer: Any | None = None,
    *,
    top_k: int = 20,
    strength_power: float = 2.0,
    z_power: float = 4.0,
    z_floor: float | str = "auto",
    filter_role_neighbors_cos: float = 0.45,
    chunk_dirs: int = 64,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Score residual directions against the full vocabulary using `sum_d sigma_d^2 * max(0, |z_{d,t}| - z_floor)^4`."""
    if not directions:
        return {
            "schema": LORA_AUDIT_SCHEMA_VERSION,
            "summary": {
                "n_directions": 0,
                "n_reader_directions": 0,
                "n_writer_directions": 0,
                "total_sv_energy": 0.0,
                "z_floor": 0.0,
                "role_tokens_share": 0.0,
                "format_tokens_share": 0.0,
                "top1_token": None,
                "top1_max_z": 0.0,
            },
            "diagnosis_hint": "no_residual_directions: adapter or diff had no matching hidden-dim linear projections.",
            "top_tokens": [],
            "top_content_tokens": [],
            "top_reader_tokens": [],
            "top_writer_tokens": [],
            "top_modules": [],
        }

    dev = _resolve_device(device) if device not in (None, "auto") else (
        unembed.device if unembed.device.type != "cpu" else torch.device("cpu")
    )
    top_k = max(1, int(top_k))
    U_norm = F.normalize(unembed.to(device=dev, dtype=torch.float32), dim=-1, eps=1e-12)
    vocab_size, hidden_size = int(U_norm.shape[0]), int(U_norm.shape[1])

    valid_dirs = [d for d in directions if int(d.vec.numel()) == hidden_size and float(d.strength) > 1e-12]
    if not valid_dirs:
        raise ValueError(f"No valid directions matching unembedding hidden_size={hidden_size}")

    n_dirs = len(valid_dirs)
    dir_mat = F.normalize(torch.stack([d.vec.float() for d in valid_dirs], dim=0), dim=-1, eps=1e-12).to(dev)
    strengths = torch.tensor([float(d.strength) for d in valid_dirs], device=dev, dtype=torch.float32)
    is_reader = torch.tensor([d.side == "reader" for d in valid_dirs], device=dev, dtype=torch.bool)
    weights = strengths.pow(float(strength_power))

    if z_floor == "auto":
        z_floor_val = min(4.0, max(2.0, math.sqrt(2.0 * math.log(max(vocab_size, 16))) - 0.85))
    else:
        z_floor_val = max(0.0, float(z_floor))

    def _accumulate(z_fl: float):
        tot_s = torch.zeros(vocab_size, device=dev, dtype=torch.float32)
        rdr_s = torch.zeros(vocab_size, device=dev, dtype=torch.float32)
        wrt_s = torch.zeros(vocab_size, device=dev, dtype=torch.float32)
        m_z = torch.zeros(vocab_size, device=dev, dtype=torch.float32)
        d_means = torch.zeros(n_dirs, device=dev, dtype=torch.float32)
        d_stds = torch.ones(n_dirs, device=dev, dtype=torch.float32)
        step = max(1, int(chunk_dirs))
        for start in range(0, n_dirs, step):
            end = min(n_dirs, start + step)
            cos = U_norm @ dir_mat[start:end].t()
            m_chunk = cos.mean(dim=0)
            s_chunk = cos.std(dim=0, unbiased=False).clamp_min(1e-12)
            d_means[start:end] = m_chunk
            d_stds[start:end] = s_chunk

            z = (cos - m_chunk.unsqueeze(0)) / s_chunk.unsqueeze(0)
            abs_z = z.abs()
            m_z = torch.maximum(m_z, abs_z.max(dim=1).values)
            eff_z = (abs_z - float(z_fl)).clamp_min(0.0) if z_fl > 0.0 else abs_z
            contrib = eff_z.pow(float(z_power)) * weights[start:end].unsqueeze(0)
            tot_s += contrib.sum(dim=1)

            rmask = is_reader[start:end]
            if bool(rmask.any()):
                rdr_s += contrib[:, rmask].sum(dim=1)
            wmask = ~rmask
            if bool(wmask.any()):
                wrt_s += contrib[:, wmask].sum(dim=1)
        return tot_s, rdr_s, wrt_s, m_z, d_means, d_stds

    total_score, reader_score, writer_score, max_z, dir_means, dir_stds = _accumulate(z_floor_val)
    if float(total_score.max().item()) <= 1e-12 and z_floor_val > 0.0:
        # Diffuse fallback when no direction produces an extreme outlier above z_floor.
        z_floor_val = 0.0
        total_score, reader_score, writer_score, max_z, dir_means, dir_stds = _accumulate(0.0)

    special_ids = _special_token_ids(tokenizer)
    k_raw = min(top_k, vocab_size)
    raw_vals, raw_ids = total_score.topk(k_raw)

    # Scan deeply so even large multilingual role clusters don't crowd out true domain content tokens.
    k_scan = min(vocab_size, max(top_k * 32, 2048))
    scan_vals, scan_ids = total_score.topk(k_scan)
    raw_ids_list = [int(x) for x in raw_ids.cpu().tolist()]
    raw_vals_list = [float(x) for x in raw_vals.cpu().tolist()]
    scan_ids_list = [int(x) for x in scan_ids.cpu().tolist()]
    scan_vals_list = [float(x) for x in scan_vals.cpu().tolist()]

    role_seed_ids = [
        tid for tid in raw_ids_list if _is_explicit_role_word(_decode_tok_safe(tokenizer, tid))
    ]
    role_neighbor_ids: set[int] = set()
    if role_seed_ids and filter_role_neighbors_cos > 0.0:
        seed_t = torch.tensor(role_seed_ids, device=dev, dtype=torch.long)
        scan_t = torch.tensor(scan_ids_list, device=dev, dtype=torch.long)
        role_sims = (U_norm.index_select(0, scan_t) @ U_norm.index_select(0, seed_t).t()).max(dim=1).values
        for tid, sim_v in zip(scan_ids_list, role_sims.cpu().tolist()):
            if float(sim_v) >= float(filter_role_neighbors_cos):
                role_neighbor_ids.add(int(tid))

    def _classify_tok(decoded_tok: str, tid: int) -> tuple[bool, bool, bool]:
        is_role = _is_explicit_role_or_special(decoded_tok, tid, special_ids) or (int(tid) in role_neighbor_ids)
        is_fmt = (not is_role) and _is_format_only_token(decoded_tok)
        is_cont = (not is_role) and (not is_fmt)
        return is_role, is_fmt, is_cont

    content_pairs: list[tuple[int, float]] = []
    for val, tid in zip(scan_vals_list, scan_ids_list):
        if float(val) <= 0.0 and content_pairs:
            break
        tok_str = _decode_tok_safe(tokenizer, tid)
        _, _, is_cont = _classify_tok(tok_str, tid)
        if is_cont:
            content_pairs.append((tid, float(val)))
            if len(content_pairs) >= k_raw:
                break

    k_split = min(10, k_raw)
    n_reader = int(is_reader.sum().item())
    n_writer = n_dirs - n_reader
    reader_pairs: list[tuple[int, float]] = []
    writer_pairs: list[tuple[int, float]] = []
    if n_reader > 0:
        rv, ri = reader_score.topk(k_split)
        reader_pairs = [(int(tid), float(val)) for val, tid in zip(rv.cpu().tolist(), ri.cpu().tolist())]
    if n_writer > 0:
        wv, wi = writer_score.topk(k_split)
        writer_pairs = [(int(tid), float(val)) for val, tid in zip(wv.cpu().tolist(), wi.cpu().tolist())]

    # Pass 2: exact per-direction attribution ONLY on the selected unique token IDs (<= 4 * top_k rows).
    sel_ids_list = sorted(
        set(raw_ids_list)
        | {tid for tid, _ in content_pairs}
        | {tid for tid, _ in reader_pairs}
        | {tid for tid, _ in writer_pairs}
    )
    id_to_row = {tid: i for i, tid in enumerate(sel_ids_list)}
    sel_t = torch.tensor(sel_ids_list, device=dev, dtype=torch.long)
    U_sel = U_norm.index_select(0, sel_t)
    cos_sel = U_sel @ dir_mat.t()
    z_sel = (cos_sel - dir_means.unsqueeze(0)) / dir_stds.unsqueeze(0)
    abs_sel = z_sel.abs()
    eff_sel = (abs_sel - float(z_floor_val)).clamp_min(0.0) if z_floor_val > 0.0 else abs_sel
    c_sel = eff_sel.pow(float(z_power)) * weights.unsqueeze(0)
    z_sel_cpu = z_sel.detach().cpu()
    c_sel_cpu = c_sel.detach().cpu()

    mod_keys = [(d.layer, d.submatrix, d.side) for d in valid_dirs]
    unique_mods = sorted(set(mod_keys), key=lambda x: (-1 if x[0] is None else x[0], x[1], x[2]))
    mod_to_idx = {m: i for i, m in enumerate(unique_mods)}

    def _build_token_entry(tid: int, score_val: float, include_contributors: bool = True) -> dict[str, Any]:
        row_i = id_to_row[int(tid)]
        tok_str = _decode_tok_safe(tokenizer, int(tid))
        z_row = z_sel_cpu[row_i]
        abs_row = z_row.abs()
        best_dir_idx = int(torch.argmax(abs_row).item())
        signed_peak_z = float(z_row[best_dir_idx].item())
        is_role, is_fmt, is_cont = _classify_tok(tok_str, int(tid))
        entry: dict[str, Any] = {
            "token_id": int(tid),
            "token": tok_str,
            "score": float(score_val),
            "max_z": float(abs_row[best_dir_idx].item()),
            "signed_peak_z": signed_peak_z,
            "is_role_token": is_role,
            "is_format_token": is_fmt,
            "is_content_token": is_cont,
        }
        if include_contributors:
            c_row = c_sel_cpu[row_i]
            tot_c = max(float(c_row.sum().item()), 1e-12)
            mod_totals = [0.0] * len(unique_mods)
            for d_i, c_val in enumerate(c_row.tolist()):
                mod_totals[mod_to_idx[mod_keys[d_i]]] += float(c_val)
            ranked_mods = sorted(
                ((unique_mods[m_i], val) for m_i, val in enumerate(mod_totals) if val > 0.0),
                key=lambda kv: kv[1],
                reverse=True,
            )[:3]
            entry["top_contributors"] = [
                {
                    "layer": mk[0],
                    "submatrix": mk[1],
                    "side": mk[2],
                    "share": float(val / tot_c),
                }
                for mk, val in ranked_mods
            ]
        return entry

    top_tokens = [_build_token_entry(tid, val, include_contributors=True) for tid, val in zip(raw_ids_list, raw_vals_list)]
    top_content_tokens = [_build_token_entry(tid, val, include_contributors=True) for tid, val in content_pairs]
    top_reader_tokens = [_build_token_entry(tid, val, include_contributors=False) for tid, val in reader_pairs]
    top_writer_tokens = [_build_token_entry(tid, val, include_contributors=False) for tid, val in writer_pairs]

    raw_sum = sum(t["score"] for t in top_tokens)
    role_sum = sum(t["score"] for t in top_tokens if t["is_role_token"])
    fmt_sum = sum(t["score"] for t in top_tokens if t["is_format_token"])
    role_tokens_share = float(role_sum / raw_sum) if raw_sum > 1e-12 else 0.0
    format_tokens_share = float(fmt_sum / raw_sum) if raw_sum > 1e-12 else 0.0

    top1_tok = top_tokens[0]["token"] if top_tokens else None
    top1_max_z = float(top_tokens[0]["max_z"]) if top_tokens else 0.0
    top_content_max_z = float(top_content_tokens[0]["max_z"]) if top_content_tokens else 0.0
    if role_tokens_share >= 0.35:
        hint = (
            "role_or_format_dominated: raw top tokens are heavily chat/role/format markers; "
            "inspect top_content_tokens and reader/writer splits for underlying domain signal."
        )
    elif max(top1_max_z, top_content_max_z) >= 6.0:
        hint = "strong_lexical_signature: sharp vocabulary outliers detected directly in weight-diff SVD."
    else:
        hint = "diffuse_or_internal_state: no extreme unembedding outlier (change may be an internal circuit or broad style shift)."

    return {
        "schema": LORA_AUDIT_SCHEMA_VERSION,
        "summary": {
            "n_directions": int(n_dirs),
            "n_reader_directions": int(n_reader),
            "n_writer_directions": int(n_writer),
            "total_sv_energy": float((strengths * strengths).sum().item()),
            "z_floor": float(z_floor_val),
            "role_tokens_share": role_tokens_share,
            "format_tokens_share": format_tokens_share,
            "top1_token": top1_tok,
            "top1_max_z": top1_max_z,
            "scoring_formula": f"sum(sigma^{strength_power:g} * max(0, |z|-{z_floor_val:.2f})^{z_power:g})",
        },
        "diagnosis_hint": hint,
        "top_tokens": top_tokens,
        "top_content_tokens": top_content_tokens,
        "top_reader_tokens": top_reader_tokens,
        "top_writer_tokens": top_writer_tokens,
        "top_modules": _aggregate_direction_modules(valid_dirs, top_n=10),
    }


@torch.no_grad()
def audit_lora(
    base_model: nn.Module | str | Path,
    adapter_source: str | Path | Mapping[str, torch.Tensor],
    *,
    tokenizer: Any | None = None,
    top_k: int = 20,
    top_sv: int = 4,
    apply_final_norm: bool = True,
    lora_scale: float | None = None,
    z_floor: float | str = "auto",
    device: torch.device | str | None = "auto",
) -> dict[str, Any]:
    """Audit a LoRA adapter from static weights alone (`sum sigma^2 * z^4`) with zero model execution."""
    dev = _resolve_device(device) if device not in (None, "auto") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    unembed, tok_out, base_meta = load_unembedding_and_tokenizer(
        base_model,
        tokenizer=tokenizer,
        apply_final_norm=apply_final_norm,
    )
    hidden_size = int(base_meta["hidden_size"])
    dirs = extract_lora_residual_directions(
        adapter_source,
        hidden_size=hidden_size,
        top_sv=top_sv,
        lora_scale=lora_scale,
        device=dev,
    )
    report = score_residual_directions_against_vocab(
        dirs,
        unembed,
        tok_out,
        top_k=top_k,
        z_floor=z_floor,
        device=dev,
    )
    report["base_model"] = base_meta
    report["adapter_source"] = "<in-memory>" if isinstance(adapter_source, Mapping) else str(adapter_source)
    return report


@torch.no_grad()
def audit_weight_diff_vocab(
    before_source: nn.Module | Mapping[str, torch.Tensor] | str | Path | _WeightSource,
    after_source: nn.Module | Mapping[str, torch.Tensor] | str | Path | _WeightSource,
    *,
    base_model: nn.Module | str | Path | None = None,
    tokenizer: Any | None = None,
    unembed: torch.Tensor | None = None,
    top_k: int = 20,
    top_sv: int = 4,
    apply_final_norm: bool = True,
    z_floor: float | str = "auto",
    device: torch.device | str | None = "auto",
) -> dict[str, Any]:
    """Translate a full checkpoint delta `(after - before)` directly into top vocabulary tokens."""
    dev = _resolve_device(device) if device not in (None, "auto") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if unembed is None:
        src_for_unembed = base_model if base_model is not None else before_source
        if not isinstance(src_for_unembed, (nn.Module, str, Path)):
            raise ValueError("Provide `base_model` or `unembed` when diffing raw tensor mappings.")
        unembed_t, tok_out, base_meta = load_unembedding_and_tokenizer(
            src_for_unembed,
            tokenizer=tokenizer,
            apply_final_norm=apply_final_norm,
        )
    else:
        unembed_t = unembed.detach().float().cpu()
        tok_out = tokenizer
        base_meta = {
            "source": "tensor",
            "unembed_key": "provided",
            "final_norm_key": None,
            "vocab_size": int(unembed_t.shape[0]),
            "hidden_size": int(unembed_t.shape[1]),
        }

    hidden_size = int(unembed_t.shape[1])
    dirs = extract_dense_diff_residual_directions(
        before_source,
        after_source,
        hidden_size=hidden_size,
        top_sv=top_sv,
        device=dev,
    )
    report = score_residual_directions_against_vocab(
        dirs,
        unembed_t,
        tok_out,
        top_k=top_k,
        z_floor=z_floor,
        device=dev,
    )
    report["base_model"] = base_meta
    return report


def format_lora_audit_report(report: Mapping[str, Any]) -> str:
    """Format a compact human-readable report from `audit_lora` / `audit_weight_diff_vocab`."""
    summ = report.get("summary") or {}
    lines = [
        f"WSI Weight-to-Vocab Audit ({report.get('schema', LORA_AUDIT_SCHEMA_VERSION)})",
        f"Directions: {summ.get('n_directions', 0)} "
        f"(readers={summ.get('n_reader_directions', 0)}, writers={summ.get('n_writer_directions', 0)}) "
        f"| z_floor={summ.get('z_floor', 0.0):.2f} "
        f"| role_share={summ.get('role_tokens_share', 0.0):.2f} "
        f"| format_share={summ.get('format_tokens_share', 0.0):.2f} "
        f"| top1={summ.get('top1_token')!r} (max_z={summ.get('top1_max_z', 0.0):.2f})",
        f"Diagnosis: {report.get('diagnosis_hint', '')}",
        "",
        "Top Tokens (raw scored):",
    ]
    for rank, item in enumerate(report.get("top_tokens") or [], start=1):
        contribs = ", ".join(
            f"L{c['layer']}.{c['submatrix']}({c['side'][0]}:{c['share']*100:.0f}%)"
            for c in (item.get("top_contributors") or [])
        )
        lines.append(
            f"  {rank:2d}. {item['token']!r:18s} score={item['score']:10.2f} "
            f"z={item['max_z']:5.2f}  [{contribs}]"
        )
    content = report.get("top_content_tokens") or []
    if content:
        lines.append("")
        lines.append("Top Content Tokens (excluding chat-role/special/format tokens):")
        lines.append("  " + ", ".join(f"{c['token']!r} ({c['max_z']:.1f}σ)" for c in content[:12]))
    return "\n".join(lines)
