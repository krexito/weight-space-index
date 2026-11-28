"""`.moicoproc`: Weight-Space Co-Processors via Mixture of Indexes (MoI).

Bundles a tiny algorithmic LoopedWSP block (< 4 KB for compact networks) together with
a calibrated hidden-state MoI router (direction + threshold + optional near-domain veto)
and optional linear projection interfaces into one self-contained `.moicoproc` safetensors
artifact with SHA-256 integrity verification.

At inference time, off-domain prompts never trigger the router (`0.0%` collateral drift);
matching algorithmic prompts trigger the co-processor directly from its bundled weights.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .looped_reader import load_looped_weights

FORMAT_COPROC = "moicoproc/1"
_DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _dtype_name(dt: torch.dtype) -> str:
    for name, val in _DTYPES.items():
        if val == dt:
            return name
    raise ValueError(f"Unsupported tensor dtype: {dt}")


def compute_weights_sha256(weights: dict[str, torch.Tensor]) -> str:
    """Deterministic SHA-256 over all bundled weight tensors (keys, shapes, dtypes, bytes)."""
    h = hashlib.sha256()
    for k in sorted(weights):
        t = weights[k].detach().contiguous().cpu()
        h.update(k.encode("utf-8"))
        h.update(b"|")
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(b"|")
        h.update(_dtype_name(t.dtype).encode("utf-8"))
        h.update(b"|")
        raw_bytes = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() > 0 else b""
        h.update(raw_bytes)
    return h.hexdigest()


def _last_token_hidden(h: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if h.dim() == 2:
        return h
    if (
        attention_mask is not None
        and attention_mask.dim() == 2
        and attention_mask.shape[0] == h.shape[0]
        and attention_mask.shape[1] == h.shape[1]
    ):
        idxs = torch.arange(attention_mask.shape[1], device=h.device).unsqueeze(0)
        pos = torch.where(attention_mask.to(h.device) > 0, idxs, torch.zeros_like(idxs)).max(dim=1).values
        return h[torch.arange(h.shape[0], device=h.device), pos]
    return h[:, -1]


@dataclass
class MoICoProcessor:
    """Portable, integrity-checked algorithmic co-processor + hidden-state MoI router."""

    weights: dict[str, torch.Tensor]
    router: dict[str, Any] | None = None
    proj_in: torch.Tensor | None = None
    proj_out: torch.Tensor | None = None
    spec: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "weights_sha256" not in self.spec:
            self.spec["weights_sha256"] = compute_weights_sha256(self.weights)

    def verify_integrity(self) -> bool:
        expected = self.spec.get("weights_sha256", "")
        if not expected:
            return False
        return compute_weights_sha256(self.weights) == expected

    def weights_payload_bytes(self) -> int:
        return sum(int(t.numel() * t.element_size()) for t in self.weights.values())

    def payload_kb(self) -> float:
        total = self.weights_payload_bytes()
        if self.proj_in is not None:
            total += int(self.proj_in.numel() * self.proj_in.element_size())
        if self.proj_out is not None:
            total += int(self.proj_out.numel() * self.proj_out.element_size())
        if self.router is not None:
            total += int(torch.as_tensor(self.router["dir"]).numel() * 4)
            if self.router.get("veto_dir") is not None:
                total += int(torch.as_tensor(self.router["veto_dir"]).numel() * 4)
        return total / 1024.0

    def save(self, path: str | Path) -> Path:
        from safetensors.torch import save_file

        tensors: dict[str, torch.Tensor] = {}
        weight_keys = []
        for k, v in sorted(self.weights.items()):
            tensors[f"w.{k}"] = v.detach().cpu().contiguous().clone()
            weight_keys.append(k)

        if self.proj_in is not None:
            tensors["proj.in"] = self.proj_in.detach().cpu().contiguous().clone()
        if self.proj_out is not None:
            tensors["proj.out"] = self.proj_out.detach().cpu().contiguous().clone()

        router_manifest = None
        if self.router is not None:
            tensors["router.dir"] = torch.as_tensor(self.router["dir"]).float().cpu().contiguous().clone()
            router_manifest = {
                k: v for k, v in self.router.items() if k not in ("dir", "veto_dir")
            }
            if self.router.get("veto_dir") is not None:
                tensors["router.veto_dir"] = (
                    torch.as_tensor(self.router["veto_dir"]).float().cpu().contiguous().clone()
                )
                router_manifest["has_veto"] = True

        spec = dict(self.spec)
        spec["weights_sha256"] = compute_weights_sha256(self.weights)
        spec["weight_keys"] = weight_keys
        manifest = {
            "format": FORMAT_COPROC,
            "spec": spec,
            "router": router_manifest,
            "has_proj_in": self.proj_in is not None,
            "has_proj_out": self.proj_out is not None,
            "meta": self.meta,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(p), metadata={"moicoproc": json.dumps(manifest)})
        return p

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> "MoICoProcessor":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            meta_raw = f.metadata() or {}
            if "moicoproc" not in meta_raw:
                raise ValueError(f"Not a .moicoproc file (missing metadata['moicoproc']): {path}")
            manifest = json.loads(meta_raw["moicoproc"])
            if manifest.get("format") != FORMAT_COPROC:
                raise ValueError(f"Unsupported format {manifest.get('format')}, expected {FORMAT_COPROC}")
            spec = dict(manifest.get("spec") or {})
            weights: dict[str, torch.Tensor] = {}
            for k in spec.get("weight_keys", []):
                # Clone before safe_open closes to avoid Windows mmap invalidation
                weights[k] = f.get_tensor(f"w.{k}").clone()
            proj_in = f.get_tensor("proj.in").clone() if manifest.get("has_proj_in") else None
            proj_out = f.get_tensor("proj.out").clone() if manifest.get("has_proj_out") else None
            router = manifest.get("router")
            if router is not None:
                has_veto = bool(router.get("has_veto")) or ("router.veto_dir" in f.keys())
                router = {k: v for k, v in router.items() if k != "has_veto"}
                router["dir"] = f.get_tensor("router.dir").clone()
                if has_veto:
                    router["veto_dir"] = f.get_tensor("router.veto_dir").clone()

        obj = cls(
            weights=weights,
            router=router,
            proj_in=proj_in,
            proj_out=proj_out,
            spec=spec,
            meta=dict(manifest.get("meta") or {}),
        )
        if verify and not obj.verify_integrity():
            raise ValueError(f"SHA-256 integrity check failed for {path}")
        return obj

    @classmethod
    def from_looped_weights(
        cls,
        weight_path: str | Path,
        *,
        task: str,
        dtype: torch.dtype = torch.float64,
        sparse_payload: bool = False,
        router: dict[str, Any] | None = None,
        proj_in: torch.Tensor | None = None,
        proj_out: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "MoICoProcessor":
        """Package a LoopedWSP checkpoint into a `MoICoProcessor`.

        If `sparse_payload=True`, stores nonzero indices (`int32`) + values (`dtype`) for 2D
        matrices, keeping compact blocks (like 928-param Maze or 3,686-NNZ Hanoi) ultra-small.
        """
        loaded = load_looped_weights(weight_path, blind=False)
        weights: dict[str, torch.Tensor] = {}
        for k, mat in sorted(loaded.matrices.items()):
            if sparse_payload and mat.ndim == 2:
                flat = (mat.rows * mat.shape[1] + mat.cols).astype(np.int32)
                weights[f"{k}.shape"] = torch.tensor(mat.shape, dtype=torch.int32)
                weights[f"{k}.idx"] = torch.from_numpy(flat.copy())
                weights[f"{k}.val"] = torch.from_numpy(mat.vals.copy()).to(dtype)
            else:
                weights[k] = torch.from_numpy(mat.to_dense()).to(dtype)

        spec = {
            "task": task,
            "sparse_payload": bool(sparse_payload),
            "weight_dtype": _dtype_name(dtype),
            "config": loaded.raw_config,
        }
        full_meta = {
            "name": f"coproc_{task}",
            "task": task,
            "n_params": loaded.total_params,
            "nonzero_params": loaded.nonzero_params,
            **(meta or {}),
        }
        return cls(
            weights=weights,
            router=router,
            proj_in=proj_in,
            proj_out=proj_out,
            spec=spec,
            meta=full_meta,
        )

    def _router_last_hidden(
        self, hidden_states: Any, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.router is None:
            raise ValueError("This co-processor has no router configured")
        h = (
            hidden_states[int(self.router["layer"])]
            if isinstance(hidden_states, (tuple, list))
            else hidden_states
        )
        return _last_token_hidden(h, attention_mask).float().cpu()

    def router_score(
        self, hidden_states: Any, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h_last = self._router_last_hidden(hidden_states, attention_mask)
        return h_last @ torch.as_tensor(self.router["dir"], dtype=torch.float32).cpu()

    def veto_score(
        self, hidden_states: Any, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if self.router is None or self.router.get("veto_dir") is None:
            return None
        h_last = self._router_last_hidden(hidden_states, attention_mask)
        return h_last @ torch.as_tensor(self.router["veto_dir"], dtype=torch.float32).cpu()

    def fires(
        self, hidden_states: Any, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h_last = self._router_last_hidden(hidden_states, attention_mask)
        ok = (h_last @ torch.as_tensor(self.router["dir"], dtype=torch.float32).cpu()) > float(
            self.router["thr"]
        )
        if self.router.get("veto_dir") is not None and "veto_thr" in self.router:
            v_sc = h_last @ torch.as_tensor(self.router["veto_dir"], dtype=torch.float32).cpu()
            ok = ok & (v_sc <= float(self.router["veto_thr"]))
        return ok

    def describe(self) -> str:
        m = self.meta
        lines = [
            f"{m.get('name', '(unnamed)')} [{FORMAT_COPROC}] task={self.spec.get('task')}",
            f"  weights: {len(self.weights)} tensors, {self.weights_payload_bytes() / 1024:.2f} KB "
            f"(total payload {self.payload_kb():.2f} KB), sha256={self.spec.get('weights_sha256', '')[:12]}...",
        ]
        if self.router is not None:
            lines.append(
                f"  router: hidden_states[{self.router['layer']}] last_token . dir > {float(self.router['thr']):.4g}"
            )
            if self.router.get("veto_dir") is not None and "veto_thr" in self.router:
                lines.append(
                    f"  veto: hidden_states[{self.router['layer']}] last_token . veto_dir <= {float(self.router['veto_thr']):.4g}"
                )
        return "\n".join(lines)


# ------------------------------------------------------------------ Router Fitting

def _unit_diff_direction(h_pos: torch.Tensor, h_neg: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    """Fit a ridge-regularized LDA/difference direction pointing from `h_neg` -> `h_pos`."""
    xp = h_pos.float().cpu()
    xn = h_neg.float().cpu()
    diff = xp.mean(dim=0) - xn.mean(dim=0)
    if xp.shape[0] + xn.shape[0] >= 8:
        xc = torch.cat([xp - xp.mean(dim=0, keepdim=True), xn - xn.mean(dim=0, keepdim=True)], dim=0)
        cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
        dim = cov.shape[0]
        reg = cov + (ridge * float(torch.trace(cov) / max(1, dim)) + 1e-6) * torch.eye(dim)
        try:
            w = torch.linalg.solve(reg, diff)
            if float((w @ diff).item()) > 0.0 and torch.isfinite(w).all():
                diff = w
        except RuntimeError:
            pass
    norm = float(diff.norm().item())
    return diff / max(norm, 1e-12)


def _calibrate_threshold(pos_scores: torch.Tensor, neg_scores: torch.Tensor, target_fpr: float, margin: float) -> float:
    if neg_scores.numel() == 0:
        return float(pos_scores.min().item() - abs(margin))
    if target_fpr <= 0.0:
        neg_cut = float(neg_scores.max().item())
    else:
        q = min(1.0, max(0.0, 1.0 - float(target_fpr)))
        neg_cut = float(torch.quantile(neg_scores, q).item())
    pos_low = float(pos_scores.min().item())
    if pos_low > neg_cut:
        # Place threshold cleanly between max-negative and min-positive (shifted conservatively toward pos)
        gap = pos_low - neg_cut
        return neg_cut + max(min(gap * 0.5, gap), min(margin, gap * 0.8))
    return neg_cut + margin


def fit_coproc_router(
    h_pos: torch.Tensor,
    h_neg: torch.Tensor,
    *,
    h_veto_neg: torch.Tensor | None = None,
    layer: int,
    target_fpr: float = 0.0,
    margin: float = 0.05,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    """Fit a hidden-state MoI router (and optional near-domain adversarial veto) on calibration embeddings."""
    hp = h_pos.float().cpu()
    hn = h_neg.float().cpu()
    if hp.dim() != 2 or hn.dim() != 2:
        raise ValueError("h_pos and h_neg must be 2D tensors [N, D]")
    if hp.shape[1] != hn.shape[1]:
        raise ValueError(f"Hidden dim mismatch: {hp.shape[1]} vs {hn.shape[1]}")

    direction = _unit_diff_direction(hp, hn, ridge=ridge)
    pos_s = hp @ direction
    neg_s = hn @ direction
    thr = _calibrate_threshold(pos_s, neg_s, target_fpr=target_fpr, margin=margin)

    router: dict[str, Any] = {
        "layer": int(layer),
        "readout": "last_token",
        "dir": direction,
        "thr": float(thr),
        "cal_pos_min": float(pos_s.min().item()),
        "cal_neg_max": float(neg_s.max().item()),
    }

    if h_veto_neg is not None and h_veto_neg.numel() > 0:
        hv = h_veto_neg.float().cpu()
        # veto_dir points FROM true problem instances (hp) TOWARD near-domain non-instance talk (hv)
        veto_dir = _unit_diff_direction(hv, hp, ridge=ridge)
        v_pos = hp @ veto_dir
        v_adv = hv @ veto_dir
        v_max_pos = float(v_pos.max().item())
        v_min_adv = float(v_adv.min().item())
        if v_min_adv > v_max_pos:
            veto_thr = v_max_pos + 0.5 * (v_min_adv - v_max_pos)
        else:
            # Keep 100% of true positive instances if distributions touch
            veto_thr = v_max_pos + 1e-4
        router["veto_dir"] = veto_dir
        router["veto_thr"] = float(veto_thr)
        router["cal_veto_pos_max"] = v_max_pos
        router["cal_veto_adv_min"] = v_min_adv

    return router


# ------------------------------------------------------------------ Standalone Co-Processor Executor

def _unpack_bundled_looped_net(coproc: MoICoProcessor) -> dict[str, Any]:
    """Reconstruct LoopedWSP numpy arrays exclusively from `coproc.weights` and `coproc.spec`."""
    cfg = dict(coproc.spec.get("config") or {})
    w = coproc.weights
    sparse = bool(coproc.spec.get("sparse_payload", False))

    def get_arr(k: str) -> np.ndarray:
        if k in w:
            return w[k].detach().cpu().numpy().astype(np.float64)
        if sparse and f"{k}.shape" in w:
            shape = tuple(int(s) for s in w[f"{k}.shape"].tolist())
            idx = w[f"{k}.idx"].detach().cpu().numpy().astype(np.int64)
            val = w[f"{k}.val"].detach().cpu().numpy().astype(np.float64)
            out = np.zeros(math.prod(shape), dtype=np.float64)
            out[idx] = val
            return out.reshape(shape)
        raise KeyError(f"Weight key '{k}' missing in MoICoProcessor")

    fields = cfg.get("fields") or []
    emb = []
    for i, f in enumerate(fields):
        if f"emb.{f}" in w or f"emb.{f}.shape" in w:
            emb.append(get_arr(f"emb.{f}"))
        else:
            emb.append(get_arr(f"emb.{i}"))

    layers = []
    for l, l_cfg in enumerate(cfg.get("layers", [])):
        heads = l_cfg["heads"] if isinstance(l_cfg, dict) else l_cfg
        lay: dict[str, Any] = {
            "heads": heads,
            "W1": get_arr(f"L{l}.W1"),
            "b1": get_arr(f"L{l}.b1"),
            "W2": get_arr(f"L{l}.W2"),
            "b2": get_arr(f"L{l}.b2"),
        }
        if heads:
            for nm in ("Wq", "Wk", "Wv", "Wo"):
                lay[nm] = get_arr(f"L{l}.{nm}")
        layers.append(lay)

    return {
        "cfg": cfg,
        "emb": emb,
        "W_out": get_arr("out.W"),
        "b_out": get_arr("out.b"),
        "layers": layers,
    }


def _softmax_np(s: np.ndarray) -> np.ndarray:
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def _looped_loops(rule: dict[str, Any], n_input: int, n_call: int) -> int:
    n = n_call if rule.get("n") == "call" else n_input
    return int(rule["const"] + rule["lin"] * -(-n // rule["div"]) + rule["log2"] * math.ceil(math.log2(max(n, 2))))


def _looped_forward_networks(net: dict[str, Any], toks: list[list[int]], p0: int, R: int, caches=None) -> np.ndarray:
    c = net["cfg"]
    T = np.asarray(toks, dtype=np.int64)
    if c.get("position_field"):
        T = np.column_stack([T, p0 + np.arange(len(T))])
    h = net["emb"][0][T[:, 0]].copy()
    for f in range(1, T.shape[1]):
        h += net["emb"][f][T[:, f]]
    for t in range(R):
        for l, L in enumerate(net["layers"]):
            if L["heads"]:
                Q, K, V = h @ L["Wq"], h @ L["Wk"], h @ L["Wv"]
                if caches is not None:
                    cache = caches[t][l]
                    n_prev = cache["n"]
                    n_new = n_prev + len(K)
                    if cache["K"] is None or n_new > len(cache["K"]):
                        K2 = np.empty((2 * n_new, K.shape[1]), dtype=np.float64)
                        V2 = np.empty((2 * n_new, V.shape[1]), dtype=np.float64)
                        if n_prev:
                            K2[:n_prev], V2[:n_prev] = cache["K"][:n_prev], cache["V"][:n_prev]
                        cache["K"], cache["V"] = K2, V2
                    cache["K"][n_prev:n_new], cache["V"][n_prev:n_new] = K, V
                    cache["n"] = n_new
                    K, V = cache["K"][:n_new], cache["V"][:n_new]
                outs, q0, v0 = [], 0, 0
                for hd in L["heads"]:
                    dk, dv = (hd["dk"], hd["dv"]) if isinstance(hd, dict) else (hd[0], hd[1])
                    s = Q[:, q0:q0 + dk] @ K[:, q0:q0 + dk].T / math.sqrt(dk)
                    outs.append(_softmax_np(s) @ V[:, v0:v0 + dv])
                    q0, v0 = q0 + dk, v0 + dv
                h = h + np.hstack(outs) @ L["Wo"]
            h = h + np.maximum(h @ L["W1"] + L["b1"], 0.0) @ L["W2"] + L["b2"]
    rows = h if c.get("readout") == "all" else h[-1:]
    return rows @ net["W_out"] + net["b_out"]


def extract_maze_from_prompt(prompt: str) -> str | None:
    """Extract a square ASCII `#`/` `/`S`/`G` maze grid from a natural-language prompt."""
    lines = [ln.rstrip("\r") for ln in prompt.splitlines()]
    best_block: list[str] = []
    cur: list[str] = []
    valid_chars = set("# SG")
    for ln in lines:
        if ln and len(ln) >= 5 and set(ln).issubset(valid_chars) and "#" in ln:
            cur.append(ln)
        else:
            if len(cur) > len(best_block):
                best_block = cur
            cur = []
    if len(cur) > len(best_block):
        best_block = cur
    if len(best_block) >= 5 and all(len(r) == len(best_block) for r in best_block):
        joined = "\n".join(best_block)
        if "S" in joined and "G" in joined:
            return joined
    return None


def extract_hanoi_n_from_prompt(prompt: str) -> int | None:
    m = re.search(r"(?:tower\s+of\s+hanoi|hanoi)[^\d]{0,40}(\d+)\s*disks?", prompt, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d+)\s*disks?\b", prompt, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 20:
            return n
    return None


def verify_maze_solution(grid_text: str, sol_text: str) -> bool:
    q = grid_text.replace("\n", "")
    out = sol_text.replace("\n", "")
    n = math.isqrt(len(q))
    if n * n != len(q) or len(out) != len(q):
        return False
    nb = lambda p: [(p[0] + a, p[1] + b) for a, b in ((1, 0), (0, -1), (0, 1), (-1, 0))]
    try:
        s, g = (divmod(q.index(ch), n) for ch in "SG")
    except ValueError:
        return False
    dist, queue = {g: 0}, [g]
    for p in queue:
        for x in nb(p):
            if 0 <= min(x) and max(x) < n and q[x[0] * n + x[1]] != "#" and x not in dist:
                dist[x] = dist[p] + 1
                queue.append(x)
    marked = {divmod(i, n) for i, ch in enumerate(out) if ch in "oSG"}
    valid = all(x == y or (x, y) == (" ", "o") for x, y in zip(q, out)) and all(
        sum(x in marked for x in nb(p)) == (1 if p in (s, g) else 2) for p in marked
    )
    seen, q2 = {s}, [s]
    for p in q2:
        for x in nb(p):
            if x in marked and x not in seen:
                seen.add(x)
                q2.append(x)
    return bool(valid and seen == marked and (len(marked) - 1 == dist.get(s)))


def verify_hanoi_moves(n: int, moves_text: str) -> bool:
    m = re.search(r"moves\s*=\s*(\[.*\])", moves_text, re.DOTALL)
    if not m:
        return False
    try:
        moves = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return False
    state = [list(range(n, 0, -1)), [], []]
    goal = [[], [], list(range(n, 0, -1))]
    for mv in moves:
        if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
            return False
        d, f, t = int(mv[0]), int(mv[1]), int(mv[2])
        if f == t or state[f][-1:] != [d] or (state[t] and state[t][-1] <= d):
            return False
        state[t].append(state[f].pop())
    return state == goal and len(moves) == (2 ** n - 1)


def _run_bundled_maze(coproc: MoICoProcessor, prompt: str, *, check_fixed_point: bool = True) -> dict[str, Any]:
    grid = extract_maze_from_prompt(prompt)
    if grid is None:
        return {"solved": False, "solution": "", "loop_steps": 0, "fixed_point_ok": False, "error": "no maze parsed"}
    net = _unpack_bundled_looped_net(coproc)
    c = net["cfg"]
    charset = c["io"]["charset"]
    rows = grid.split("\n")
    n = len(rows)
    toks = [[r, col, charset.index(ch)] for r, row in enumerate(rows) for col, ch in enumerate(row)]
    R = _looped_loops(c["loops"], len(toks), len(toks))
    logits_r = _looped_forward_networks(net, toks, 0, R)
    gen_r = [int(i) for i in logits_r.argmax(-1)]
    sol_r = "\n".join("".join(charset[i] for i in gen_r[r * n:(r + 1) * n]) for r in range(n))
    fp_ok = True
    if check_fixed_point:
        logits_2r = _looped_forward_networks(net, toks, 0, 2 * R)
        gen_2r = [int(i) for i in logits_2r.argmax(-1)]
        fp_ok = (gen_r == gen_2r)
    solved = verify_maze_solution(grid, sol_r)
    return {
        "solved": bool(solved and fp_ok),
        "solution": sol_r,
        "loop_steps": int(R),
        "fixed_point_ok": bool(fp_ok),
        "error": None if solved else "invalid maze path",
    }


def _run_bundled_hanoi(coproc: MoICoProcessor, prompt: str, *, check_fixed_point: bool = True) -> dict[str, Any]:
    n = extract_hanoi_n_from_prompt(prompt)
    if n is None:
        return {"solved": False, "solution": "", "loop_steps": 0, "fixed_point_ok": False, "error": "no disk count parsed"}
    net = _unpack_bundled_looped_net(coproc)
    c = net["cfg"]
    vocab = c["io"]["vocab"]
    ix = {s: i for i, s in enumerate(vocab) if s}
    words = [f"INIT({d},0)" for d in range(n, 0, -1)] + [f"GOAL({d},2)" for d in range(n, 0, -1)]
    toks = [[ix[w]] for w in [f"N={n}"] + words + ["SEP"]]
    R = _looped_loops(c["loops"], len(toks), len(toks))

    def _decode(r_steps: int) -> list[int]:
        caches = [[{"n": 0, "K": None, "V": None} for _ in net["layers"]] for _ in range(r_steps)]
        seq, gen = list(toks), []
        for i, tk in enumerate(toks):
            lg = _looped_forward_networks(net, [tk], i, r_steps, caches)
        while True:
            o = int(np.argmax(lg[-1]))
            if o == c["end_id"]:
                break
            gen.append(o)
            cap = c.get("max_new") or (2 ** n + 16)
            if len(gen) >= cap or len(seq) >= c["max_tokens"]:
                break
            tok = list(c["out_token"]["template"])
            tok[c["out_token"]["slot"]] = o
            seq.append(tok)
            lg = _looped_forward_networks(net, [tok], len(seq) - 1, r_steps, caches)
        return gen

    gen_r = _decode(R)
    sol = "moves = [" + ", ".join(vocab[i] for i in gen_r) + "]"
    fp_ok = True
    if check_fixed_point and n <= 6:
        fp_ok = (_decode(2 * R) == gen_r)
    solved = verify_hanoi_moves(n, sol)
    return {
        "solved": bool(solved and fp_ok),
        "solution": sol,
        "loop_steps": int(R),
        "fixed_point_ok": bool(fp_ok),
        "error": None if solved else "invalid hanoi moves",
    }


class CoProcessorExecutor:
    """Registry that executes bundled `MoICoProcessor` weights on parsed prompts."""

    def __init__(self) -> None:
        self._runners: dict[str, Callable[..., dict[str, Any]]] = {
            "maze": _run_bundled_maze,
            "hanoi": _run_bundled_hanoi,
        }

    def register(self, task: str, fn: Callable[..., dict[str, Any]]) -> None:
        self._runners[str(task)] = fn

    def run(
        self, coproc: MoICoProcessor, prompt: str, *, check_fixed_point: bool = True
    ) -> dict[str, Any]:
        task = str(coproc.spec.get("task") or "")
        if task not in self._runners:
            raise KeyError(f"No executor registered for co-processor task {task!r}")
        if not coproc.verify_integrity():
            raise ValueError("MoICoProcessor integrity check failed prior to execution")
        t0 = time.perf_counter()
        out = self._runners[task](coproc, prompt, check_fixed_point=check_fixed_point)
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        out["task"] = task
        return out


# ------------------------------------------------------------------ Live Model Routing

def _format_user_prompt(tokenizer, prompt: str, enable_thinking: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
            )
        except TypeError:
            return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return prompt


def _encode_batch(
    tokenizer, prompts: list[str], device: torch.device, *, enable_thinking: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [_format_user_prompt(tokenizer, p, enable_thinking=enable_thinking) for p in prompts]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def route_and_generate(
    model,
    tokenizer,
    prompts: list[str],
    coproc: MoICoProcessor,
    executor: CoProcessorExecutor | None = None,
    *,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    check_fixed_point: bool = True,
    fallback_to_base: bool = True,
) -> dict[str, Any]:
    """Run MoI-routed inference: untouched base `model.generate` when router is False,
    bundled LoopedWSP execution when True (with clean fallback if unparseable).
    """
    if coproc.router is None:
        raise ValueError("MoICoProcessor must carry a calibrated router for route_and_generate")
    exec_obj = executor or CoProcessorExecutor()
    dev = next(model.parameters()).device
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

    orig_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    answers: list[str] = [""] * len(prompts)
    events: list[dict[str, Any]] = [{} for _ in prompts]
    n_fired = 0
    n_routed_solved = 0
    n_routed_fallback = 0
    n_base_untouched = 0
    n_fp_ok = 0
    coproc_ms_list: list[float] = []

    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            inp, att = _encode_batch(tokenizer, batch_prompts, dev)
            L = inp.shape[1]

            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=att, output_hidden_states=True)
            scores = coproc.router_score(out.hidden_states, attention_mask=att)
            vetos = coproc.veto_score(out.hidden_states, attention_mask=att)
            fired = coproc.fires(out.hidden_states, attention_mask=att)

            if not bool(fired.any()):
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=inp, attention_mask=att, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=pad_id,
                    )
                for j in range(len(batch_prompts)):
                    idx = start + j
                    answers[idx] = tokenizer.decode(gen[j, L:], skip_special_tokens=True)
                    n_base_untouched += 1
                    events[idx] = {
                        "route": "base_untouched",
                        "fired": False,
                        "router_score": round(float(scores[j]), 4),
                        "veto_score": None if vetos is None else round(float(vetos[j]), 4),
                    }
                continue

            fallback_local: list[int] = []
            for j, p in enumerate(batch_prompts):
                idx = start + j
                sc_j = round(float(scores[j]), 4)
                vt_j = None if vetos is None else round(float(vetos[j]), 4)
                if not bool(fired[j]):
                    fallback_local.append(j)
                    events[idx] = {
                        "route": "base_untouched",
                        "fired": False,
                        "router_score": sc_j,
                        "veto_score": vt_j,
                    }
                    continue
                n_fired += 1
                res = exec_obj.run(coproc, p, check_fixed_point=check_fixed_point)
                coproc_ms_list.append(float(res.get("elapsed_ms", 0.0)))
                if res.get("solved"):
                    answers[idx] = res["solution"]
                    n_routed_solved += 1
                    if res.get("fixed_point_ok"):
                        n_fp_ok += 1
                    events[idx] = {
                        "route": "coproc_solved",
                        "fired": True,
                        "router_score": sc_j,
                        "veto_score": vt_j,
                        "loop_steps": res.get("loop_steps"),
                        "fixed_point_ok": res.get("fixed_point_ok"),
                        "elapsed_ms": res.get("elapsed_ms"),
                    }
                else:
                    n_routed_fallback += 1
                    if fallback_to_base:
                        fallback_local.append(j)
                    events[idx] = {
                        "route": "routed_fallback",
                        "fired": True,
                        "router_score": sc_j,
                        "veto_score": vt_j,
                        "error": res.get("error"),
                    }

            if fallback_local:
                # Generate base answers preserving clean left-padded batching for unrouted/fallback rows
                sub_prompts = [batch_prompts[j] for j in fallback_local]
                sub_inp, sub_att = _encode_batch(tokenizer, sub_prompts, dev)
                Ls = sub_inp.shape[1]
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=sub_inp, attention_mask=sub_att, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=pad_id,
                    )
                for k_pos, j in enumerate(fallback_local):
                    idx = start + j
                    answers[idx] = tokenizer.decode(gen[k_pos, Ls:], skip_special_tokens=True)
                    if events[idx]["route"] == "base_untouched":
                        n_base_untouched += 1
    finally:
        tokenizer.padding_side = orig_side

    return {
        "answers": answers,
        "events": events,
        "stats": {
            "n_prompts": len(prompts),
            "n_fired": n_fired,
            "fire_rate": round(n_fired / max(1, len(prompts)), 4),
            "n_routed_solved": n_routed_solved,
            "n_routed_fallback": n_routed_fallback,
            "n_base_untouched": n_base_untouched,
            "fixed_point_pass_rate": round(n_fp_ok / max(1, n_routed_solved), 4) if n_routed_solved else None,
            "mean_coproc_ms": round(sum(coproc_ms_list) / len(coproc_ms_list), 2) if coproc_ms_list else None,
        },
    }


__all__ = [
    "CoProcessorExecutor",
    "FORMAT_COPROC",
    "MoICoProcessor",
    "compute_weights_sha256",
    "extract_hanoi_n_from_prompt",
    "extract_maze_from_prompt",
    "fit_coproc_router",
    "route_and_generate",
    "verify_hanoi_moves",
    "verify_maze_solution",
]
