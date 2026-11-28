"""Static Weight-Space Reader & Decompiler for LoopedWSP algorithmic transformers.

Inspects raw weight matrices of looped single-block (or multi-layer looped) transformers
completely statically—zero forward passes, zero task labels, zero filename hints—recovering:
  1. Sparsity & numeric anatomy (integer/half-integer fractions, distinct magnitudes, hard-attention scale).
  2. Spectral & effective-rank profile (SVD effective rank exp(H(sigma^2)) & spectral norm per role).
  3. Attention routing topology (per-head QK peak/sharpness, active Q/K/V/O channels, fixed vs mutable keys).
  4. ReLU logical gadget census (step/threshold pairs and sign-symmetric/cancellation pairs).
  5. Residual-stream channel dataflow graph (4-way channel partition + Tarjan SCC cycle analysis).
  6. Blind structural archetype classification from the static measurements alone.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_ROLES = ("q", "k", "v", "o", "w1", "b1", "w2", "b2", "embed", "unembed", "pos", "other")


@dataclass
class SparseMatrix:
    """Memory-efficient COO representation of a sparse or dense 1D/2D weight tensor."""

    shape: tuple[int, ...]
    rows: np.ndarray  # int64 (for 1D, all 0)
    cols: np.ndarray  # int64
    vals: np.ndarray  # float64

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(math.prod(self.shape))

    @property
    def nnz(self) -> int:
        return int(self.vals.size)

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=np.float64)
        if self.nnz == 0:
            return out
        if self.ndim == 1:
            out[self.cols] = self.vals
        elif self.ndim == 2:
            out[self.rows, self.cols] = self.vals
        else:
            flat = out.reshape(-1)
            flat[self.cols] = self.vals
        return out

    @classmethod
    def from_dense(cls, arr: np.ndarray, tol: float = 0.0) -> "SparseMatrix":
        a = np.asarray(arr, dtype=np.float64)
        shape = tuple(int(s) for s in a.shape)
        if a.ndim == 1:
            idx = np.flatnonzero(np.abs(a) > tol)
            return cls(shape=shape, rows=np.zeros(idx.size, dtype=np.int64),
                       cols=idx.astype(np.int64), vals=a[idx].astype(np.float64))
        if a.ndim == 2:
            r, c = np.nonzero(np.abs(a) > tol)
            return cls(shape=shape, rows=r.astype(np.int64), cols=c.astype(np.int64),
                       vals=a[r, c].astype(np.float64))
        flat = a.reshape(-1)
        idx = np.flatnonzero(np.abs(flat) > tol)
        return cls(shape=shape, rows=np.zeros(idx.size, dtype=np.int64),
                   cols=idx.astype(np.int64), vals=flat[idx].astype(np.float64))

    @classmethod
    def from_flat_coo(cls, shape: tuple[int, ...], idx: np.ndarray, val: np.ndarray) -> "SparseMatrix":
        idx64 = np.asarray(idx, dtype=np.int64).reshape(-1)
        v64 = np.asarray(val, dtype=np.float64).reshape(-1)
        if v64.size:
            nz = v64 != 0.0
            if not np.all(nz):
                idx64 = idx64[nz]
                v64 = v64[nz]
        if len(shape) == 2:
            r, c = np.divmod(idx64, int(shape[1]))
            return cls(shape=tuple(int(s) for s in shape), rows=r.astype(np.int64),
                       cols=c.astype(np.int64), vals=v64)
        return cls(shape=tuple(int(s) for s in shape), rows=np.zeros(idx64.size, dtype=np.int64),
                   cols=idx64, vals=v64)

    def active_rows(self) -> np.ndarray:
        if self.nnz == 0:
            return np.empty(0, dtype=np.int64)
        if self.ndim == 1:
            return np.zeros(1, dtype=np.int64)
        return np.unique(self.rows)

    def active_cols(self) -> np.ndarray:
        if self.nnz == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(self.cols)

    def compact_dense_2d(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (submatrix_dense, row_ids, col_ids) keeping only rows & cols with >=1 nonzero."""
        if self.ndim != 2:
            raise ValueError(f"expected 2D matrix, got shape {self.shape}")
        if self.nnz == 0:
            return np.zeros((0, 0), dtype=np.float64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        ru, r_inv = np.unique(self.rows, return_inverse=True)
        cu, c_inv = np.unique(self.cols, return_inverse=True)
        sub = np.zeros((ru.size, cu.size), dtype=np.float64)
        np.add.at(sub, (r_inv, c_inv), self.vals)
        return sub, ru, cu

    def slice_cols(self, start: int, stop: int) -> "SparseMatrix":
        if self.ndim != 2:
            raise ValueError("slice_cols only supports 2D matrices")
        mask = (self.cols >= start) & (self.cols < stop)
        return SparseMatrix(
            shape=(self.shape[0], stop - start),
            rows=self.rows[mask],
            cols=self.cols[mask] - start,
            vals=self.vals[mask],
        )

    def slice_rows(self, start: int, stop: int) -> "SparseMatrix":
        if self.ndim != 2:
            raise ValueError("slice_rows only supports 2D matrices")
        mask = (self.rows >= start) & (self.rows < stop)
        return SparseMatrix(
            shape=(stop - start, self.shape[1]),
            rows=self.rows[mask] - start,
            cols=self.cols[mask],
            vals=self.vals[mask],
        )


@dataclass
class LoopedCircuitSpec:
    """Normalized structural & weight container for a LoopedWSP transformer."""

    matrices: dict[str, SparseMatrix]
    roles: dict[str, str]
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    total_params: int
    nonzero_params: int
    file_bytes: int
    layer_heads: list[list[dict[str, Any]]] = field(default_factory=list)
    layer_d_ff: list[int] = field(default_factory=list)
    masks: list[str] = field(default_factory=list)
    readout_mode: str = "unknown"
    field_names: list[str] = field(default_factory=list)
    position_field: bool = False
    loops_rule: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    raw_config: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> SparseMatrix | None:
        return self.matrices.get(key)


_LAYER_KEY_RE = re.compile(r"^L(\d+)\.([A-Za-z0-9_]+)$")


def _classify_key_role(key: str, *, pos_field_names: set[str] | None = None) -> str:
    if key.startswith("emb."):
        fname = key[4:].lower()
        if fname in ("pos", "position", "pos_id", "position_class") or (pos_field_names and key[4:] in pos_field_names):
            return "pos"
        return "embed"
    if key in ("out.W", "W_out", "lm_head.weight"):
        return "unembed"
    if key in ("out.b", "b_out", "lm_head.bias"):
        return "other"
    m = _LAYER_KEY_RE.match(key)
    if m:
        sub = m.group(2).lower()
        mapping = {
            "wq": "q", "q_proj": "q",
            "wk": "k", "k_proj": "k",
            "wv": "v", "v_proj": "v",
            "wo": "o", "o_proj": "o",
            "w1": "w1", "up_proj": "w1", "fc1": "w1",
            "b1": "b1",
            "w2": "w2", "down_proj": "w2", "fc2": "w2",
            "b2": "b2",
        }
        return mapping.get(sub, "other")
    low = key.lower()
    if "pos" in low:
        return "pos"
    if "emb" in low:
        return "embed"
    if "out" in low or "head" in low:
        return "unembed"
    return "other"


def _parse_npz_archive(npz_path: Path) -> dict[str, SparseMatrix]:
    mats: dict[str, SparseMatrix] = {}
    with np.load(npz_path, allow_pickle=False) as z:
        keys = list(z.keys())
        seen_sparse: set[str] = set()
        for k in keys:
            if k.endswith(".shape"):
                base = k[:-6]
                seen_sparse.add(base)
                shape = tuple(int(s) for s in z[k])
                idx = z[base + ".idx"]
                val = z[base + ".val"]
                mats[base] = SparseMatrix.from_flat_coo(shape, idx, val)
        for k in keys:
            if k.endswith((".shape", ".idx", ".val")):
                base = k.rsplit(".", 1)[0]
                if base in seen_sparse:
                    continue
            if k not in mats:
                mats[k] = SparseMatrix.from_dense(z[k])
    return mats


def _parse_other_archive(path: Path) -> tuple[dict[str, SparseMatrix], dict[str, Any]]:
    suf = path.suffix.lower()
    cfg: dict[str, Any] = {}
    mats: dict[str, SparseMatrix] = {}
    if suf == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            meta = f.metadata() or {}
            if "config" in meta:
                try:
                    cfg = json.loads(meta["config"])
                except json.JSONDecodeError:
                    cfg = {}
            for k in f.keys():
                t = f.get_tensor(k).detach().cpu().numpy().copy()
                mats[k] = SparseMatrix.from_dense(t)
    elif suf in (".pt", ".pth", ".bin"):
        import torch

        obj = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        for k, v in obj.items():
            if hasattr(v, "detach"):
                mats[str(k)] = SparseMatrix.from_dense(v.detach().cpu().numpy())
    elif suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "weights" in obj and isinstance(obj["weights"], dict):
            cfg = obj.get("config", {})
            for k, v in obj["weights"].items():
                mats[str(k)] = SparseMatrix.from_dense(np.asarray(v, dtype=np.float64))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, list):
                    mats[str(k)] = SparseMatrix.from_dense(np.asarray(v, dtype=np.float64))
    else:
        raise ValueError(f"Unsupported weight file extension: {path}")
    return mats, cfg


def _resolve_looped_paths(path: str | Path) -> tuple[Path, Path | None]:
    p = Path(path)
    if p.is_dir():
        npz = p / "weights.npz"
        cfg = p / "network.json"
        if npz.exists():
            return npz, cfg if cfg.exists() else None
        npzs = sorted(p.glob("*.npz"))
        if len(npzs) == 1:
            cand_cfg = npzs[0].with_suffix(".json")
            return npzs[0], cand_cfg if cand_cfg.exists() else None
        raise FileNotFoundError(f"Could not find weights.npz in directory: {p}")

    if p.suffix.lower() == ".json":
        # Config JSON pointing to sibling weights.npz or <stem>.npz
        if p.name == "network.json" and (p.parent / "weights.npz").exists():
            return p.parent / "weights.npz", p
        cand_npz = p.with_suffix(".npz")
        if cand_npz.exists():
            return cand_npz, p
        return p, None

    # Weight file (.npz, .safetensors, .pt)
    if p.name == "weights.npz" and (p.parent / "network.json").exists():
        return p, p.parent / "network.json"
    sib = p.with_suffix(".json")
    return p, sib if sib.exists() else None


def _infer_structure(matrices: dict[str, SparseMatrix], cfg: dict[str, Any]) -> tuple[
    int, int, int, int, list[list[dict[str, Any]]], list[int], list[str], str, list[str], bool, dict[str, Any]
]:
    # Discover layer indices from weight keys and/or cfg
    layer_ids: set[int] = set()
    for k in matrices:
        m = _LAYER_KEY_RE.match(k)
        if m:
            layer_ids.add(int(m.group(1)))
    n_layers = max(len(cfg.get("layers", [])), (max(layer_ids) + 1) if layer_ids else 0)

    d_model = int(cfg.get("d_model") or 0)
    if d_model == 0:
        for k, mat in matrices.items():
            if k.startswith("emb.") and mat.ndim == 2:
                d_model = int(mat.shape[1])
                break
            if k in ("out.W", "W_out") and mat.ndim == 2:
                d_model = int(mat.shape[0])
                break
            if _LAYER_KEY_RE.match(k) and k.endswith(".W1") and mat.ndim == 2:
                d_model = int(mat.shape[0])
                break

    global_mask = str(cfg.get("mask") or "unknown")
    layer_heads: list[list[dict[str, Any]]] = []
    layer_d_ff: list[int] = []
    masks_seen: set[str] = set()

    cfg_layers = cfg.get("layers", [])
    for l in range(n_layers):
        heads_out: list[dict[str, Any]] = []
        d_ff_l = 0
        w1 = matrices.get(f"L{l}.W1")
        if w1 is not None and w1.ndim == 2:
            d_ff_l = int(w1.shape[1])
        if l < len(cfg_layers):
            raw_l = cfg_layers[l]
            if isinstance(raw_l, dict):
                d_ff_l = int(raw_l.get("d_ff", d_ff_l))
                for hd in raw_l.get("heads", []):
                    mk = str(hd.get("mask", global_mask))
                    masks_seen.add(mk)
                    heads_out.append({"dk": int(hd["dk"]), "dv": int(hd["dv"]), "mask": mk})
            elif isinstance(raw_l, list):
                for hd in raw_l:
                    mk = global_mask
                    masks_seen.add(mk)
                    heads_out.append({"dk": int(hd[0]), "dv": int(hd[1]), "mask": mk})
        else:
            wq = matrices.get(f"L{l}.Wq")
            wv = matrices.get(f"L{l}.Wv")
            if wq is not None and wv is not None and wq.ndim == 2 and wv.ndim == 2:
                heads_out.append({"dk": int(wq.shape[1]), "dv": int(wv.shape[1]), "mask": global_mask})
                masks_seen.add(global_mask)
        layer_heads.append(heads_out)
        layer_d_ff.append(d_ff_l)

    n_heads = sum(len(hs) for hs in layer_heads)
    d_ff = sum(layer_d_ff)
    readout_mode = str(cfg.get("readout") or ("emit" if "alphabet" in cfg else "unknown"))
    field_names = [str(f) for f in cfg.get("fields", [])]
    pos_field = bool(cfg.get("position_field", False))
    loops_rule = dict(cfg.get("loops") or {})
    return d_model, n_layers, n_heads, d_ff, layer_heads, layer_d_ff, sorted(masks_seen), readout_mode, field_names, pos_field, loops_rule


def load_looped_weights(path: str | Path, *, blind: bool = False) -> LoopedCircuitSpec:
    """Load a LoopedWSP checkpoint (.npz / directory / .safetensors / .pt / .json) into a `LoopedCircuitSpec`.

    When `blind=True`, task-identifying fields (`name`, `io`, `alphabet`) and `source_path` are stripped.
    """
    weight_path, cfg_path = _resolve_looped_paths(path)
    cfg: dict[str, Any] = {}
    if cfg_path is not None and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    if weight_path.suffix.lower() == ".npz":
        matrices = _parse_npz_archive(weight_path)
    else:
        matrices, extra_cfg = _parse_other_archive(weight_path)
        if not cfg and extra_cfg:
            cfg = extra_cfg

    pos_names: set[str] = set()
    if cfg.get("position_field") and cfg.get("fields"):
        # In networks/ config, position_field=True appends a trailing field if not already in fields
        pass
    for idx_f, fname in enumerate(cfg.get("fields", [])):
        if any(tok in str(fname).lower() for tok in ("pos", "row", "col")):
            pos_names.add(str(fname))
            pos_names.add(str(idx_f))

    roles = {k: _classify_key_role(k, pos_field_names=pos_names) for k in matrices}
    total_params = sum(m.size for m in matrices.values())
    nonzero_params = sum(m.nnz for m in matrices.values())
    file_bytes = weight_path.stat().st_size

    (d_model, n_layers, n_heads, d_ff, layer_heads, layer_d_ff,
     masks, readout_mode, field_names, pos_field, loops_rule) = _infer_structure(matrices, cfg)

    clean_cfg = {} if blind else cfg
    if blind:
        # Keep only architectural shape metadata if needed, never task names/vocab/alphabet
        clean_cfg = {k: cfg[k] for k in ("d_model", "layers", "mask", "readout", "position_field", "loops")
                     if k in cfg}

    return LoopedCircuitSpec(
        matrices=matrices,
        roles=roles,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        total_params=total_params,
        nonzero_params=nonzero_params,
        file_bytes=file_bytes,
        layer_heads=layer_heads,
        layer_d_ff=layer_d_ff,
        masks=masks,
        readout_mode=readout_mode,
        field_names=[] if blind else field_names,
        position_field=pos_field,
        loops_rule=loops_rule,
        source_path="" if blind else str(weight_path),
        raw_config=clean_cfg,
    )


# ------------------------------------------------------------------ Static Analysis Primitives

def _singular_values_sparse_2d(mat: SparseMatrix) -> np.ndarray:
    """Exact nonzero singular values of a 2D SparseMatrix, fast even for large sparse embeddings."""
    if mat.nnz == 0 or mat.ndim != 2:
        return np.empty(0, dtype=np.float64)
    sub, _, _ = mat.compact_dense_2d()
    r, c = sub.shape
    if r == 0 or c == 0:
        return np.empty(0, dtype=np.float64)
    if min(r, c) == 1:
        nrm = float(np.linalg.norm(sub))
        return np.array([nrm], dtype=np.float64) if nrm > 0.0 else np.empty(0, dtype=np.float64)
    # Use Gram eigen-decomposition when one dimension is much larger than the other (e.g. vocab x d_model)
    if max(r, c) > 2 * min(r, c) and min(r, c) <= 1500:
        gram = (sub.T @ sub) if r > c else (sub @ sub.T)
        eigvals = np.linalg.eigvalsh(gram)
        s = np.sqrt(np.maximum(eigvals[::-1], 0.0))
    else:
        s = np.linalg.svd(sub, compute_uv=False)
    return s[s > 1e-12]


def _spectral_stats(mat: SparseMatrix) -> dict[str, float]:
    s = _singular_values_sparse_2d(mat)
    if s.size == 0:
        return {"spectral_norm": 0.0, "frobenius_norm": 0.0, "effective_rank": 0.0, "numeric_rank": 0}
    s2 = s * s
    tot = float(s2.sum())
    if tot <= 0.0:
        return {"spectral_norm": 0.0, "frobenius_norm": 0.0, "effective_rank": 0.0, "numeric_rank": 0}
    p = s2 / tot
    p = p[p > 0.0]
    entropy = float(-np.sum(p * np.log(p)))
    eff_rank = float(math.exp(entropy))
    num_rank = int(np.sum(s > (s[0] * 1e-6)))
    return {
        "spectral_norm": float(s[0]),
        "frobenius_norm": float(math.sqrt(tot)),
        "effective_rank": eff_rank,
        "numeric_rank": num_rank,
    }


def _numeric_anatomy(spec: LoopedCircuitSpec) -> dict[str, Any]:
    all_vals_list = [m.vals for m in spec.matrices.values() if m.nnz > 0]
    if not all_vals_list:
        return {
            "total_params": spec.total_params,
            "nonzero_params": 0,
            "sparsity_ratio": 1.0,
            "file_bytes": spec.file_bytes,
            "exact_integer_fraction": 0.0,
            "half_integer_fraction": 0.0,
            "distinct_nonzero_magnitudes": 0,
            "max_abs_weight": 0.0,
            "min_nonzero_abs_weight": 0.0,
        }
    vals = np.concatenate(all_vals_list)
    abs_v = np.abs(vals)
    is_int = np.abs(vals - np.round(vals)) < 1e-7
    is_half = (~is_int) & (np.abs(vals * 2.0 - np.round(vals * 2.0)) < 1e-7)
    # Count distinct magnitudes up to 1e-6 relative/absolute tolerance via rounding to 8 sig figs
    distinct_mags = int(np.unique(np.round(abs_v, 6)).size)
    tot = max(spec.total_params, 1)
    return {
        "total_params": int(spec.total_params),
        "nonzero_params": int(vals.size),
        "sparsity_ratio": round(1.0 - float(vals.size) / float(tot), 6),
        "file_bytes": int(spec.file_bytes),
        "exact_integer_fraction": round(float(np.mean(is_int)), 4),
        "half_integer_fraction": round(float(np.mean(is_half)), 4),
        "distinct_nonzero_magnitudes": distinct_mags,
        "max_abs_weight": float(np.max(abs_v)),
        "min_nonzero_abs_weight": float(np.min(abs_v)),
    }


def _analyze_attention_topology(
    spec: LoopedCircuitSpec,
    readonly_channels: set[int],
    mutable_channels: set[int],
) -> dict[str, Any]:
    heads_info: list[dict[str, Any]] = []
    max_qk_peak = 0.0
    n_hard_heads = 0
    n_uniform_pool_heads = 0
    n_fixed_key_heads = 0
    n_state_key_heads = 0

    for l in range(spec.n_layers):
        heads = spec.layer_heads[l] if l < len(spec.layer_heads) else []
        if not heads:
            continue
        wq = spec.get(f"L{l}.Wq")
        wk = spec.get(f"L{l}.Wk")
        wv = spec.get(f"L{l}.Wv")
        wo = spec.get(f"L{l}.Wo")
        if wq is None or wk is None or wv is None or wo is None:
            continue
        q0 = v0 = 0
        for h_idx, hd in enumerate(heads):
            dk = int(hd["dk"])
            dv = int(hd["dv"])
            mask = str(hd.get("mask", "unknown"))
            wq_h = wq.slice_cols(q0, q0 + dk)
            wk_h = wk.slice_cols(q0, q0 + dk)
            wv_h = wv.slice_cols(v0, v0 + dv)
            wo_h = wo.slice_rows(v0, v0 + dv)
            q0 += dk
            v0 += dv

            # Compute sparse bilinear operator M_QK = Wq_h @ Wk_h.T / sqrt(dk)
            q_ch = sorted(int(x) for x in wq_h.active_rows())
            k_ch = sorted(int(x) for x in wk_h.active_rows())
            v_ch = sorted(int(x) for x in wv_h.active_rows())
            o_ch = sorted(int(x) for x in wo_h.active_cols())

            qk_peak = 0.0
            qk_frob = 0.0
            qk_sharpness = 0.0
            if wq_h.nnz > 0 and wk_h.nnz > 0:
                q_sub = np.zeros((len(q_ch), dk), dtype=np.float64)
                q_map = {c: i for i, c in enumerate(q_ch)}
                for r, c, val in zip(wq_h.rows, wq_h.cols, wq_h.vals):
                    q_sub[q_map[int(r)], int(c)] += val
                k_sub = np.zeros((len(k_ch), dk), dtype=np.float64)
                k_map = {c: i for i, c in enumerate(k_ch)}
                for r, c, val in zip(wk_h.rows, wk_h.cols, wk_h.vals):
                    k_sub[k_map[int(r)], int(c)] += val
                m_qk = (q_sub @ k_sub.T) / math.sqrt(max(dk, 1))
                abs_qk = np.abs(m_qk).ravel()
                nz_qk = abs_qk[abs_qk > 0.0]
                if nz_qk.size > 0:
                    qk_peak = float(np.max(nz_qk))
                    qk_frob = float(np.linalg.norm(nz_qk))
                    med = float(np.median(nz_qk))
                    qk_sharpness = round(qk_peak / max(med, 1e-12), 3)

            max_qk_peak = max(max_qk_peak, qk_peak)
            is_hard = qk_peak >= 20.0
            is_uniform_pool = (wq_h.nnz == 0 and wk_h.nnz == 0 and wv_h.nnz > 0)
            if is_hard:
                n_hard_heads += 1
            if is_uniform_pool:
                n_uniform_pool_heads += 1

            k_set = set(k_ch)
            n_k_ro = len(k_set & readonly_channels)
            n_k_mut = len(k_set & mutable_channels)
            if is_uniform_pool:
                key_regime = "uniform_mean_pool"
            elif k_set:
                if n_k_mut == 0 and n_k_ro > 0:
                    key_regime = "fixed_input_or_pos"
                    n_fixed_key_heads += 1
                elif n_k_ro == 0 and n_k_mut > 0:
                    key_regime = "mutable_state"
                    n_state_key_heads += 1
                elif n_k_ro > 0 and n_k_mut > 0:
                    key_regime = "mixed_pos_and_state"
                    n_state_key_heads += 1
                else:
                    key_regime = "unconnected"
            else:
                key_regime = "inactive"

            heads_info.append({
                "layer": l,
                "head": h_idx,
                "dk": dk,
                "dv": dv,
                "mask": mask,
                "qk_peak_abs": round(qk_peak, 4),
                "qk_frobenius": round(qk_frob, 4),
                "qk_sharpness": qk_sharpness,
                "is_hard_attention": is_hard,
                "is_uniform_pool": is_uniform_pool,
                "key_regime": key_regime,
                "n_q_channels": len(q_ch),
                "n_k_channels": len(k_ch),
                "n_v_channels": len(v_ch),
                "n_o_channels": len(o_ch),
                "q_channels": q_ch[:16],
                "k_channels": k_ch[:16],
                "v_channels": v_ch[:16],
                "o_channels": o_ch[:16],
            })

    return {
        "n_heads_total": len(heads_info),
        "n_hard_attention_heads": n_hard_heads,
        "n_uniform_pool_heads": n_uniform_pool_heads,
        "max_qk_scale": round(max_qk_peak, 4),
        "n_fixed_key_heads": n_fixed_key_heads,
        "n_mutable_key_heads": n_state_key_heads,
        "heads": heads_info,
    }


def _relu_gadget_census(spec: LoopedCircuitSpec, cos_thr: float = 0.99) -> dict[str, Any]:
    """Detect exact analytical ReLU threshold pairs and sign-symmetric cancellation pairs."""
    total_neurons = 0
    dead_neurons = 0
    step_threshold_pairs = 0
    cancellation_pairs = 0
    abs_like_pairs = 0
    linear_passthrough_pairs = 0
    matched_neurons_total = 0
    per_layer: list[dict[str, Any]] = []

    for l in range(spec.n_layers):
        w1 = spec.get(f"L{l}.W1")
        w2 = spec.get(f"L{l}.W2")
        b1_sp = spec.get(f"L{l}.b1")
        if w1 is None or w2 is None or w1.ndim != 2 or w2.ndim != 2:
            continue
        d_ff = int(w1.shape[1])
        total_neurons += d_ff
        b1 = b1_sp.to_dense() if b1_sp is not None else np.zeros(d_ff, dtype=np.float64)

        # Active neurons: nonzero in BOTH W1[:, j] and W2[j, :]
        w1_nz_cols = np.unique(w1.cols) if w1.nnz else np.empty(0, dtype=np.int64)
        w2_nz_rows = np.unique(w2.rows) if w2.nnz else np.empty(0, dtype=np.int64)
        active = np.intersect1d(w1_nz_cols, w2_nz_rows, assume_unique=True)
        n_act = int(active.size)
        dead_neurons += d_ff - n_act
        if n_act < 2:
            per_layer.append({
                "layer": l, "d_ff": d_ff, "active_neurons": n_act,
                "step_threshold_pairs": 0, "cancellation_pairs": 0,
            })
            continue

        # Build compact normalized input matrix U_in [n_in_ch, n_act] and output matrix U_out [n_act, n_out_ch]
        act_set_map = np.full(d_ff, -1, dtype=np.int64)
        act_set_map[active] = np.arange(n_act, dtype=np.int64)

        m1 = act_set_map[w1.cols] >= 0
        r1 = w1.rows[m1]
        c1 = act_set_map[w1.cols[m1]]
        v1 = w1.vals[m1]
        ru1, r1_inv = np.unique(r1, return_inverse=True)
        W1_sub = np.zeros((ru1.size, n_act), dtype=np.float64)
        np.add.at(W1_sub, (r1_inv, c1), v1)
        n1 = np.linalg.norm(W1_sub, axis=0, keepdims=True)
        U1 = W1_sub / np.maximum(n1, 1e-12)

        m2 = act_set_map[w2.rows] >= 0
        r2 = act_set_map[w2.rows[m2]]
        c2 = w2.cols[m2]
        v2 = w2.vals[m2]
        cu2, c2_inv = np.unique(c2, return_inverse=True)
        W2_sub = np.zeros((n_act, cu2.size), dtype=np.float64)
        np.add.at(W2_sub, (r2, c2_inv), v2)
        n2 = np.linalg.norm(W2_sub, axis=1, keepdims=True)
        U2 = W2_sub / np.maximum(n2, 1e-12)

        b1_act = b1[active]

        # Cosine similarity on inputs (chunked if huge, though max n_act <= 3106)
        used = np.zeros(n_act, dtype=bool)
        l_step = 0
        l_cancel = 0
        l_abs = 0
        l_lin = 0

        # Compute full or chunked candidate pairs
        chunk = 512
        step_candidates: list[tuple[int, int, float]] = []
        cancel_candidates: list[tuple[int, int, float]] = []
        for s0 in range(0, n_act, chunk):
            s1 = min(s0 + chunk, n_act)
            sim_in = U1[:, s0:s1].T @ U1  # [chunk, n_act]
            # Strictly upper-triangular pairs (i < j)
            rows_p, cols_p = np.where(sim_in > cos_thr)
            for ri, j in zip(rows_p, cols_p):
                i = s0 + int(ri)
                j = int(j)
                if j <= i:
                    continue
                if abs(float(b1_act[i]) - float(b1_act[j])) <= 1e-7:
                    continue
                c_out = float(U2[i] @ U2[j])
                if c_out < -cos_thr:
                    step_candidates.append((i, j, float(sim_in[ri, j])))

            rows_n, cols_n = np.where(sim_in < -cos_thr)
            for ri, j in zip(rows_n, cols_n):
                i = s0 + int(ri)
                j = int(j)
                if j <= i:
                    continue
                c_out = float(U2[i] @ U2[j])
                if abs(c_out) > cos_thr:
                    cancel_candidates.append((i, j, c_out))

        # Greedily match disjoint pairs (step/threshold first, then cancellation)
        for i, j, _ in step_candidates:
            if not used[i] and not used[j]:
                used[i] = used[j] = True
                l_step += 1
        for i, j, c_out in cancel_candidates:
            if not used[i] and not used[j]:
                used[i] = used[j] = True
                l_cancel += 1
                if c_out > cos_thr:
                    l_abs += 1
                else:
                    l_lin += 1

        step_threshold_pairs += l_step
        cancellation_pairs += l_cancel
        abs_like_pairs += l_abs
        linear_passthrough_pairs += l_lin
        matched_neurons_total += int(used.sum())
        per_layer.append({
            "layer": l,
            "d_ff": d_ff,
            "active_neurons": n_act,
            "step_threshold_pairs": l_step,
            "cancellation_pairs": l_cancel,
            "matched_neurons": int(used.sum()),
        })

    active_total = total_neurons - dead_neurons
    unmatched = max(0, active_total - matched_neurons_total)
    cov = round(float(matched_neurons_total) / float(max(active_total, 1)), 4) if active_total else 0.0
    return {
        "total_neurons": int(total_neurons),
        "active_neurons": int(active_total),
        "dead_neurons": int(dead_neurons),
        "step_threshold_pairs": int(step_threshold_pairs),
        "cancellation_pairs": int(cancellation_pairs),
        "abs_like_pairs": int(abs_like_pairs),
        "linear_passthrough_pairs": int(linear_passthrough_pairs),
        "matched_gadget_neurons": int(matched_neurons_total),
        "unmatched_active_neurons": int(unmatched),
        "gadget_coverage_fraction": cov,
        "per_layer": per_layer,
    }


def _tarjan_scc(n_nodes: int, adj: dict[int, set[int]]) -> list[list[int]]:
    index_counter = 0
    indices: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    stack: list[int] = []
    on_stack: set[int] = set()
    sccs: list[list[int]] = []

    # Iterative Tarjan to avoid recursion limits on d_model > 1000
    for start in range(n_nodes):
        if start in indices:
            continue
        call_stack: list[tuple[int, list[int], int]] = [
            (start, sorted(adj.get(start, ())), 0)
        ]
        indices[start] = lowlink[start] = index_counter
        index_counter += 1
        stack.append(start)
        on_stack.add(start)

        while call_stack:
            v, nbrs, idx = call_stack[-1]
            if idx < len(nbrs):
                w = nbrs[idx]
                call_stack[-1] = (v, nbrs, idx + 1)
                if w not in indices:
                    indices[w] = lowlink[w] = index_counter
                    index_counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, sorted(adj.get(w, ())), 0))
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])
            else:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
                if lowlink[v] == indices[v]:
                    comp: list[int] = []
                    while True:
                        w = stack.pop()
                        on_stack.remove(w)
                        comp.append(w)
                        if w == v:
                            break
                    sccs.append(sorted(comp))
    return sccs


def _channel_dataflow(spec: LoopedCircuitSpec) -> tuple[dict[str, Any], set[int], set[int]]:
    d = spec.d_model
    emb_write = np.zeros(d, dtype=bool)
    block_read = np.zeros(d, dtype=bool)
    block_write = np.zeros(d, dtype=bool)
    out_read = np.zeros(d, dtype=bool)

    adj: dict[int, set[int]] = defaultdict(set)

    for k, mat in spec.matrices.items():
        role = spec.roles.get(k, "other")
        if role in ("embed", "pos") and mat.ndim == 2 and mat.shape[1] == d:
            emb_write[mat.active_cols()] = True
        elif role == "unembed" and mat.ndim == 2 and mat.shape[0] == d:
            out_read[mat.active_rows()] = True

    for l in range(spec.n_layers):
        for key in (f"L{l}.Wq", f"L{l}.Wk", f"L{l}.Wv", f"L{l}.W1"):
            m = spec.get(key)
            if m is not None and m.ndim == 2 and m.shape[0] == d:
                block_read[m.active_rows()] = True
        for key in (f"L{l}.Wo", f"L{l}.W2"):
            m = spec.get(key)
            if m is not None and m.ndim == 2 and m.shape[1] == d:
                block_write[m.active_cols()] = True
        b2 = spec.get(f"L{l}.b2")
        if b2 is not None and b2.size == d:
            block_write[b2.active_cols()] = True

        # Directed dataflow edges: Wv -> Wo (value transport) per head, and W1 -> W2 per active neuron
        wv = spec.get(f"L{l}.Wv")
        wo = spec.get(f"L{l}.Wo")
        if wv is not None and wo is not None and wv.ndim == 2 and wo.ndim == 2:
            # Map each shared hidden slot s in [0, sum(dv)) -> input rows & output cols
            in_by_slot: dict[int, list[int]] = defaultdict(list)
            out_by_slot: dict[int, list[int]] = defaultdict(list)
            for r, c in zip(wv.rows, wv.cols):
                in_by_slot[int(c)].append(int(r))
            for r, c in zip(wo.rows, wo.cols):
                out_by_slot[int(r)].append(int(c))
            for s, src_chs in in_by_slot.items():
                dst_chs = out_by_slot.get(s)
                if not dst_chs:
                    continue
                for u in src_chs:
                    adj[u].update(dst_chs)

        w1 = spec.get(f"L{l}.W1")
        w2 = spec.get(f"L{l}.W2")
        if w1 is not None and w2 is not None and w1.ndim == 2 and w2.ndim == 2:
            in_by_j: dict[int, list[int]] = defaultdict(list)
            out_by_j: dict[int, list[int]] = defaultdict(list)
            for r, c in zip(w1.rows, w1.cols):
                in_by_j[int(c)].append(int(r))
            for r, c in zip(w2.rows, w2.cols):
                out_by_j[int(r)].append(int(c))
            for j, src_chs in in_by_j.items():
                dst_chs = out_by_j.get(j)
                if not dst_chs:
                    continue
                for u in src_chs:
                    adj[u].update(dst_chs)

    readonly_mask = emb_write & (~block_write) & (block_read | out_read)
    recurrent_mask = block_read & block_write
    write_only_mask = block_write & (~block_read) & (out_read | emb_write | block_write)
    # Specifically distinguish write_only_readout (read by W_out, never read inside block) vs dead
    write_only_readout_mask = block_write & (~block_read) & out_read
    unused_dead_mask = ~(emb_write | block_write | block_read | out_read)

    readonly_ch = [int(i) for i in np.flatnonzero(readonly_mask)]
    recurrent_ch = [int(i) for i in np.flatnonzero(recurrent_mask)]
    write_only_ch = [int(i) for i in np.flatnonzero(write_only_readout_mask)]
    unused_ch = [int(i) for i in np.flatnonzero(unused_dead_mask)]

    sccs = _tarjan_scc(d, adj)
    nontrivial_sccs = [
        c for c in sccs
        if len(c) > 1 or (len(c) == 1 and c[0] in adj.get(c[0], ()))
    ]
    nontrivial_sccs.sort(key=lambda c: -len(c))
    scc_edge_count = sum(len(vs) for vs in adj.values())
    feedback_self_or_scc_edges = 0
    comp_id = {}
    for cid, comp in enumerate(nontrivial_sccs):
        for node in comp:
            comp_id[node] = cid
    for u, vs in adj.items():
        if u in comp_id:
            cu = comp_id[u]
            feedback_self_or_scc_edges += sum(1 for v in vs if comp_id.get(v) == cu)

    summary = {
        "d_model": int(d),
        "readonly_input_or_pos_count": len(readonly_ch),
        "recurrent_scratchpad_count": len(recurrent_ch),
        "write_only_readout_count": len(write_only_ch),
        "unused_dead_count": len(unused_ch),
        "readonly_input_or_pos_channels": readonly_ch[:32],
        "recurrent_scratchpad_channels": recurrent_ch[:32],
        "write_only_readout_channels": write_only_ch[:32],
        "directed_edge_count": int(scc_edge_count),
        "feedback_cycle_edge_count": int(feedback_self_or_scc_edges),
        "n_sccs_total": len(sccs),
        "n_nontrivial_sccs": len(nontrivial_sccs),
        "largest_scc_size": len(nontrivial_sccs[0]) if nontrivial_sccs else 0,
        "nontrivial_sccs_sizes": [len(c) for c in nontrivial_sccs[:10]],
        # Keep raw count of general block_write & ~block_read channels as internal/orphan diagnostic
        "write_only_internal_count": int(np.sum(write_only_mask & (~write_only_readout_mask))),
    }
    return summary, set(readonly_ch), set(recurrent_ch) | set(np.flatnonzero(block_write).tolist())


def analyze_looped_circuit(spec: LoopedCircuitSpec) -> dict[str, Any]:
    """Run full static weight-only decompilation on `spec` (zero forward passes, zero task labels)."""
    anatomy = _numeric_anatomy(spec)
    dataflow, readonly_set, mutable_set = _channel_dataflow(spec)
    attn = _analyze_attention_topology(spec, readonly_set, mutable_set)
    gadgets = _relu_gadget_census(spec)

    # Spectral & effective rank per matrix and aggregated by canonical role
    by_role: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_matrix_spectral: dict[str, dict[str, Any]] = {}
    for k, mat in sorted(spec.matrices.items()):
        role = spec.roles.get(k, "other")
        if mat.ndim != 2:
            continue
        st = _spectral_stats(mat)
        per_matrix_spectral[k] = {"role": role, "shape": list(mat.shape), "nnz": mat.nnz, **st}
        by_role[role].append(st)

    role_spectral_summary: dict[str, dict[str, float]] = {}
    for role in ("q", "k", "v", "o", "w1", "w2", "embed", "pos", "unembed"):
        items = by_role.get(role)
        if not items:
            continue
        role_spectral_summary[role] = {
            "count": len(items),
            "mean_effective_rank": round(float(np.mean([x["effective_rank"] for x in items])), 3),
            "max_effective_rank": round(float(np.max([x["effective_rank"] for x in items])), 3),
            "mean_spectral_norm": round(float(np.mean([x["spectral_norm"] for x in items])), 3),
            "max_spectral_norm": round(float(np.max([x["spectral_norm"] for x in items])), 3),
        }

    no_attn_layers = sum(1 for hs in spec.layer_heads if not hs)
    return {
        "architecture": {
            "d_model": int(spec.d_model),
            "n_layers": int(spec.n_layers),
            "n_attn_layers": int(spec.n_layers - no_attn_layers),
            "n_mlp_only_layers": int(no_attn_layers),
            "n_heads": int(spec.n_heads),
            "d_ff_total": int(spec.d_ff),
            "masks": list(spec.masks),
            "readout_mode": spec.readout_mode,
        },
        "anatomy": anatomy,
        "role_spectral_summary": role_spectral_summary,
        "attention": attn,
        "relu_gadgets": gadgets,
        "dataflow": dataflow,
        "per_matrix_spectral": per_matrix_spectral,
    }


def classify_looped_circuit(analysis: dict[str, Any]) -> dict[str, Any]:
    """Blind rule-based structural archetype classifier using ONLY static `analysis`."""
    arch = analysis["architecture"]
    anat = analysis["anatomy"]
    attn = analysis["attention"]
    gad = analysis["relu_gadgets"]
    df = analysis["dataflow"]

    d_model = arch["d_model"]
    n_layers = arch["n_layers"]
    n_mlp_only = arch["n_mlp_only_layers"]
    masks = set(arch["masks"])
    readout = arch["readout_mode"]
    nz = anat["nonzero_params"]
    max_qk = attn["max_qk_scale"]
    n_hard = attn["n_hard_attention_heads"]
    n_pool = attn.get("n_uniform_pool_heads", 0)
    step_pairs = gad["step_threshold_pairs"]
    cancel_pairs = gad["cancellation_pairs"]
    rec_cnt = df["recurrent_scratchpad_count"]
    ro_cnt = df["readonly_input_or_pos_count"]
    largest_scc = df["largest_scc_size"]

    evidence: list[str] = []
    secondary_tags: list[str] = []
    ambiguities: list[str] = []

    if n_hard > 0:
        secondary_tags.append(f"hard_attention_heads={n_hard}(max_qk={max_qk:.3g})")
    if n_pool > 0:
        secondary_tags.append(f"uniform_mean_pool_heads={n_pool}")
    if step_pairs > 0:
        secondary_tags.append(f"relu_threshold_gates={step_pairs}")
    if cancel_pairs > 0:
        secondary_tags.append(f"relu_cancellation_gadgets={cancel_pairs}")
    if df["n_nontrivial_sccs"] > 0:
        secondary_tags.append(f"recurrent_channel_sccs={df['n_nontrivial_sccs']}(max={largest_scc})")

    # Determine dominant structural archetype from mask regime, layer composition, and footprint
    if "prefix" in masks:
        archetype = "prefix_in_context_state_tracker"
        confidence = 0.85 if n_layers >= 4 else 0.72
        evidence.append(f"prefix attention mask with {n_layers} layers and readout='{readout}'")
        evidence.append(f"recurrent scratchpad uses {rec_cnt}/{d_model} channels, {step_pairs} threshold pairs")
        ambiguities.append("Shares multi-layer state tracking signature with constraint solvers (e.g. web_of_lies vs shuffled_objects)")
    elif masks == {"causal"}:
        if d_model >= 150 and nz >= 1500:
            archetype = "causal_autoregressive_state_machine"
            confidence = 0.86 if n_pool > 0 else 0.82
            evidence.append(f"causal-only mask with wide state (d_model={d_model}, nnz={nz}) and readout='{readout}'")
            evidence.append(f"uniform_pool_heads={n_pool}, {step_pairs} threshold pairs, max QK scale={max_qk:.3g}")
            ambiguities.append("Autoregressive causal counting/transition machines (e.g. Hanoi vs Checker) share uniform-pool + ReLU threshold signatures")
        else:
            archetype = "compact_causal_accumulator"
            confidence = 0.82
            evidence.append(f"compact causal-only block (d_model={d_model}, n_layers={n_layers}, nnz={nz})")
            ambiguities.append("Small causal accumulators (e.g. parity) have minimal SCC footprint")
    elif d_model <= 96 and n_layers <= 3:
        archetype = "compact_spatial_or_graph_propagator"
        confidence = 0.80 if (n_hard > 0 and rec_cnt > 0) else 0.68
        evidence.append(
            f"compact width (d_model={d_model}, n_layers={n_layers}, masks={sorted(masks)}) with "
            f"{ro_cnt} readonly & {rec_cnt} recurrent channels"
        )
        evidence.append(
            f"attention max QK={max_qk:.3g}, hard heads={n_hard}, SCC max={largest_scc}, "
            f"ReLU step pairs={step_pairs}"
        )
        if "causal" in masks and "none" in masks:
            secondary_tags.append("hybrid_causal_plus_bidirectional_emitter")
        ambiguities.append(
            "Compact iterative propagators (Maze BFS variants, shortest-path Graph, Chains, Dyck, Boolean) "
            "share narrow scratchpad + hard-attention signatures"
        )
    elif n_layers >= 4 or (n_mlp_only >= 1 and step_pairs >= 20):
        archetype = "deep_iterative_constraint_or_dp_solver"
        confidence = 0.84 if (n_layers >= 4 or step_pairs >= 50) else 0.72
        evidence.append(
            f"multi-layer block (n_layers={n_layers}, mlp_only_layers={n_mlp_only}) with d_model={d_model}, "
            f"{step_pairs} ReLU threshold pairs, SCC max={largest_scc}"
        )
        ambiguities.append(
            "Multi-layer bidirectional constraint/DP engines (Sudoku, SATLIB, CLRS Prim/SCC/LCS) "
            "share deep refinement layers and high threshold-pair density"
        )
    else:
        archetype = "wide_symbolic_planner_or_table_engine"
        confidence = 0.80 if d_model >= 256 else 0.65
        evidence.append(
            f"wide shallow block (d_model={d_model}, n_layers={n_layers}, nnz={nz}, masks={sorted(masks)}) "
            f"with {cancel_pairs} cancellation pairs"
        )
        ambiguities.append(
            "Wide 1-2 layer symbolic lookup/planning circuits (PlanBench variants, GraphWalks) "
            "differ mainly in vocabulary/field structure rather than block depth"
        )

    return {
        "predicted_archetype": archetype,
        "confidence": round(confidence, 3),
        "secondary_tags": secondary_tags,
        "ambiguities": ambiguities,
        "evidence": evidence,
    }


__all__ = [
    "CANONICAL_ROLES",
    "LoopedCircuitSpec",
    "SparseMatrix",
    "analyze_looped_circuit",
    "classify_looped_circuit",
    "load_looped_weights",
]
