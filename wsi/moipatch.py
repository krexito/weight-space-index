"""`.moipatch`: a portable, verifiable, reversible Additive-MoI patch that knows when to switch itself on.
=========================================================================================================

One file (safetensors + a JSON manifest in its metadata) holds:

* **the edit recipe**, not a dump of new weights. Every edit has the form used throughout this repo,
  ``W' = W - alpha * u (m * (u^T W))`` on one linear layer's weight ``W [D, U]``, so it is stored as the unit
  direction ``u [D]``, the optional column mask ``m`` (kept column indices; absent = all columns) and ``alpha``.
  A few KB per layer instead of a copy of the matrix.
* **the router**: which hidden state it reads (``hidden_states[layer]``, last token), its direction and threshold.
  The patch is meant to be switched on only when ``h . dir > thr``.
* **optional gates** (for example the sycophancy confidence gate: switch on only if the model's own first answer
  had probability above a threshold), and a plain-language **activation policy** and **expected results**.
* **the base model's fingerprint**: a SHA-256 of every weight the patch touches, in the checkpoint's own dtype.
  ``attach`` refuses a model whose weights differ, and ``off`` restores them byte for byte (checked).

Python (1-line live routing)::

    from wsi.moipatch import MoIPatch
    patch = MoIPatch.load("anti_sycophancy.moipatch")
    with patch.enable_routed(model) as routed:
        out = model.generate(**inputs)   # router decides automatically; original weights never mutated

Python (manual attach)::

    from wsi.moipatch import MoIPatch
    patch = MoIPatch.load("anti_sycophancy.moipatch")
    h = patch.attach(model)              # checks the fingerprint; the model is still the original
    out = model(ids, output_hidden_states=True)
    if patch.fires(out.hidden_states):   # router (+ gates you evaluate yourself, see patch.gates)
        h.on()                           # weights patched in place
        ...generate...
        h.off()                          # byte-identical original again

CLI::

    python -m wsi.moipatch info   PATCH
    python -m wsi.moipatch verify PATCH --model MODEL_DIR [--cpu]
    python -m wsi.moipatch audit  PATCH --model MODEL_DIR [--probe "..."] [--control "..."] [--chat] [--top-k 10] [--cpu] [--json]
    python -m wsi.moipatch run    PATCH --model MODEL_DIR --prompt "..." [--cpu]
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

FORMAT = "moipatch/1"
FORMAT_V = "moipatch/1.1"  # adds explicit input-side vectors (Edit.v); written only when an edit has one, so
                           # readers that only know moipatch/1 (e.g. rust/wsi-bake) refuse it instead of misbaking
_FORMATS = (FORMAT, FORMAT_V)
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _sha256(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def _dtype_name(dt: torch.dtype) -> str:
    return {v: k for k, v in _DTYPES.items()}[dt]


def module_suffix(name: str) -> str:
    """'model.language_model.layers.17.linear_attn.out_proj' -> 'layers.17.linear_attn.out_proj'."""
    m = re.search(r"(layers\.\d+\..+)$", name)
    if not m:
        raise ValueError(f"not a per-layer module name: {name}")
    return m.group(1)


def apply_edit(W: torch.Tensor, u: torch.Tensor, idx: torch.Tensor | None, alpha: float,
               v: torch.Tensor | None = None) -> torch.Tensor:
    """W' = W - alpha * u (m * (u^T W)), computed in float64 from the original weight, returned in W's dtype.

    With an explicit input-side vector `v` (Edit.v) the edit is the general rank-one W' = W - alpha * u v^T: e.g. the
    projection's row u^T W restricted to a subspace of the input (a null-space-constrained edit)."""
    Wd = W.detach().cpu().double()
    uu = u.detach().cpu().double()
    if v is not None:
        if idx is not None:
            raise ValueError("an edit takes either idx or an explicit v, not both")
        vv = v.detach().cpu().double().reshape(-1)
        if vv.numel() != Wd.shape[1]:
            raise ValueError(f"v has {vv.numel()} entries, the weight has {Wd.shape[1]} input columns")
        return (Wd - alpha * torch.outer(uu, vv)).to(W.dtype)
    m = torch.ones(Wd.shape[1], dtype=torch.float64)
    if idx is not None:
        m.zero_()
        m[idx.long()] = 1.0
    return (Wd - alpha * torch.outer(uu, m * (uu @ Wd))).to(W.dtype)


@dataclass
class Edit:
    module: str  # suffix, e.g. 'layers.17.linear_attn.out_proj' (matched against the model's module names)
    u: torch.Tensor  # [D] unit direction
    alpha: float
    idx: torch.Tensor | None = None  # kept column indices; None = every column
    base_sha256: str = ""  # SHA-256 of the original weight in base_dtype
    base_dtype: str = "bfloat16"
    shape: tuple = ()
    v: torch.Tensor | None = None  # [in_features] explicit input-side vector: W' = W - alpha u v^T (moipatch/1.1)


@dataclass
class MoIPatch:
    edits: list[Edit]
    router: dict | None = None  # {"layer": int, "readout": "last_token", "dir": Tensor[D], "thr": float}
    gates: list[dict] = field(default_factory=list)  # e.g. [{"type": "answer_confidence", "thr": 0.885, ...}]
    meta: dict = field(default_factory=dict)  # name, description, base_model, policy, expected, provenance

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        from safetensors.torch import save_file

        tensors, edits = {}, []
        for i, e in enumerate(self.edits):
            tensors[f"edit.{i}.u"] = e.u.detach().to(torch.float64 if e.u.dtype == torch.float64 else torch.float32).cpu().contiguous().clone()  # keep fp64 so the patched bf16 weights are bit-identical to the run
            if e.idx is not None:
                tensors[f"edit.{i}.idx"] = e.idx.detach().long().cpu().contiguous().clone()
            rec = {"module": e.module, "alpha": float(e.alpha), "has_idx": e.idx is not None,
                   "base_sha256": e.base_sha256, "base_dtype": e.base_dtype, "shape": list(e.shape)}
            if e.v is not None:
                tensors[f"edit.{i}.v"] = e.v.detach().to(torch.float64).cpu().reshape(-1).contiguous().clone()
                rec["has_v"] = True
            edits.append(rec)
        router = None
        if self.router:
            tensors["router.dir"] = torch.as_tensor(self.router["dir"]).float().cpu().contiguous().clone()
            router = {k: v for k, v in self.router.items() if k not in ("dir", "veto_dir")}
            if self.router.get("veto_dir") is not None:
                tensors["router.veto_dir"] = torch.as_tensor(self.router["veto_dir"]).float().cpu().contiguous().clone()
                router["has_veto"] = True
        fmt = FORMAT_V if any(e.v is not None for e in self.edits) else FORMAT
        manifest = {"format": fmt, "edits": edits, "router": router, "gates": self.gates, "meta": self.meta}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(p), metadata={"moipatch": json.dumps(manifest)})
        return p

    @classmethod
    def load(cls, path: str | Path) -> "MoIPatch":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            manifest = json.loads(f.metadata()["moipatch"])
            if manifest.get("format") not in _FORMATS:
                raise ValueError(f"unsupported format {manifest.get('format')} (expected one of {_FORMATS})")
            edits = []
            for i, e in enumerate(manifest["edits"]):
                edits.append(Edit(module=e["module"], u=f.get_tensor(f"edit.{i}.u"), alpha=e["alpha"],
                                  idx=f.get_tensor(f"edit.{i}.idx") if e["has_idx"] else None,
                                  base_sha256=e["base_sha256"], base_dtype=e["base_dtype"], shape=tuple(e["shape"]),
                                  v=f.get_tensor(f"edit.{i}.v") if e.get("has_v") else None))
            router = manifest.get("router")
            if router is not None:
                has_veto = bool(router.get("has_veto")) or ("router.veto_dir" in f.keys())
                router = {k: v for k, v in router.items() if k != "has_veto"}
                router["dir"] = f.get_tensor("router.dir")
                if has_veto:
                    router["veto_dir"] = f.get_tensor("router.veto_dir")
        return cls(edits=edits, router=router, gates=manifest.get("gates", []), meta=manifest.get("meta", {}))

    # ------------------------------------------------------------------ fingerprint from a checkpoint (no model load)
    def fingerprint_from_checkpoint(self, model_dir: str | Path) -> None:
        """Fill base_sha256 / base_dtype / shape of every edit from the safetensors files of the base model."""
        from safetensors import safe_open

        mdir = Path(model_dir)
        idx_file = mdir / "model.safetensors.index.json"
        index: dict[str, tuple[Path, str]] = {}
        if idx_file.exists():
            wmap = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
            for k, fname in wmap.items():
                if k.endswith(".weight") and "mtp" not in k and "visual" not in k:
                    try:
                        index[module_suffix(k[: -len(".weight")])] = (mdir / fname, k)
                    except ValueError:
                        pass
        else:
            files = [f for f in sorted(mdir.glob("*.safetensors"))
                     if not f.name.startswith("edit_") and f.name != "delta_lowrank.safetensors"]
            for f_ in files:
                with safe_open(str(f_), "pt") as s:
                    for k in s.keys():
                        if k.endswith(".weight") and "mtp" not in k and "visual" not in k:
                            try:
                                index[module_suffix(k[: -len(".weight")])] = (f_, k)
                            except ValueError:
                                pass
        for e in self.edits:
            if e.module not in index:
                raise KeyError(f"{e.module} not found in {model_dir}")
            f_, k = index[e.module]
            with safe_open(str(f_), "pt") as s:
                W = s.get_tensor(k).clone()  # copy before the mmap closes (reading it after crashes on Windows)
            e.base_sha256, e.base_dtype, e.shape = _sha256(W), _dtype_name(W.dtype), tuple(W.shape)

    # ------------------------------------------------------------------ runtime
    def attach(self, model, *, verify_fingerprint: bool = True) -> "PatchHandle":
        return PatchHandle(self, model, verify_fingerprint=verify_fingerprint)

    def audit(
        self,
        model,
        tokenizer=None,
        *,
        probe_prompts: list[str] | None = None,
        probe_ids: list[list[int]] | None = None,
        control_prompts: list[str] | None = None,
        control_ids: list[list[int]] | None = None,
        patch_path: str | Path | None = None,
        top_k: int = 10,
        verify_fingerprint: bool = True,
    ) -> dict:
        """Run structural, cryptographic, logit-lens, and end-to-end logit-diff anti-backdoor audit."""
        return audit_moipatch(
            self,
            model,
            tokenizer,
            probe_prompts=probe_prompts,
            probe_ids=probe_ids,
            control_prompts=control_prompts,
            control_ids=control_ids,
            patch_path=patch_path,
            top_k=top_k,
            verify_fingerprint=verify_fingerprint,
        )

    def enable_routed(
        self,
        model,
        *,
        mode: str = "auto",
        threshold: float | None = None,
        confidence: float | torch.Tensor | None = None,
        loop_router: bool | dict | None = None,
        eos_token_ids=None,
        verify_fingerprint: bool = True,
    ) -> "RoutedPatchHandle":
        """Attach this patch non-destructively with automatic live routing for `model(...)` and `model.generate(...)`.

        Can be used as a context manager (`with patch.enable_routed(model) as h: ...`) or standalone (`h.close()`).
        """
        return RoutedPatchHandle(
            self,
            model,
            mode=mode,
            threshold=threshold,
            confidence=confidence,
            loop_router=loop_router,
            eos_token_ids=eos_token_ids,
            verify_fingerprint=verify_fingerprint,
        )

    def _router_last_hidden(self, hidden_states) -> torch.Tensor:
        if self.router is None:
            raise ValueError("this patch has no router")
        h = hidden_states[self.router["layer"]] if isinstance(hidden_states, (tuple, list)) else hidden_states
        return (h[:, -1] if h.dim() == 3 else h).float().cpu()

    def router_score(self, hidden_states) -> torch.Tensor:
        """hidden_states: the tuple from output_hidden_states=True (or the [B, T, D] tensor of the router layer).
        Returns h_last . dir for each sequence (left padding assumed, so the last position is the last token)."""
        return self._router_last_hidden(hidden_states) @ self.router["dir"].float()

    def fires(self, hidden_states, confidence: float | torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate router decision, optional safety veto, plus any answer_confidence gate if `confidence` is supplied."""
        h_last = self._router_last_hidden(hidden_states)
        ok = (h_last @ self.router["dir"].float()) > self.router["thr"]
        if self.router.get("veto_dir") is not None and "veto_thr" in self.router:
            ok = ok & ((h_last @ self.router["veto_dir"].float()) <= float(self.router["veto_thr"]))
        if confidence is not None:
            conf = torch.as_tensor(confidence, dtype=torch.float32).cpu()
            for g in self.gates:
                if g.get("type") == "answer_confidence" and "thr" in g:
                    ok = ok & (conf > float(g["thr"]))
        return ok

    def describe(self) -> str:
        m = self.meta
        lines = [f"{m.get('name', '(unnamed)')} [{FORMAT}]", m.get("description", "")]
        if m.get("base_model"):
            lines.append(f"base model: {m['base_model']}")
        for e in self.edits:
            cols = f"{len(e.idx)} columns" if e.idx is not None else "all columns"
            if e.v is not None:
                cols = "explicit input vector"
            lines.append(f"  edit {e.module} {tuple(e.shape)} alpha {e.alpha:g}, {cols}, base {e.base_dtype} "
                         f"sha256 {e.base_sha256[:12]}...")
        if self.router:
            lines.append(f"  router: hidden_states[{self.router['layer']}] last token . dir > {self.router['thr']:.4g}")
            if self.router.get("veto_dir") is not None and "veto_thr" in self.router:
                lines.append(
                    f"  router veto: hidden_states[{self.router['layer']}] last token . veto_dir <= {float(self.router['veto_thr']):.4g}"
                )
        for g in self.gates:
            lines.append(f"  gate: {g.get('type')} > {g.get('thr')}  ({g.get('note', '')})")
        if m.get("policy"):
            lines.append(f"  policy: {m['policy']}")
        if m.get("expected"):
            lines.append(f"  expected: {json.dumps(m['expected'])}")
        n_bytes = sum(e.u.numel() * e.u.element_size() + (e.idx.numel() * 8 if e.idx is not None else 0)
                      + (e.v.numel() * 8 if e.v is not None else 0) for e in self.edits)
        lines.append(f"  payload: {n_bytes / 1024:.1f} KB of recipe")
        return "\n".join(x for x in lines if x)


def _resolve_model_modules(
    model,
    edits: list[Edit],
    *,
    verify_fingerprint: bool = True,
) -> list[tuple[str, torch.nn.Module]]:
    """Resolve each edit's suffix to its unique module in `model` and verify base SHA-256 fingerprints."""
    names: dict[str, list[tuple[str, torch.nn.Module]]] = {}
    for name, mod in model.named_modules():
        if "mtp" in name or "visual" in name or not hasattr(mod, "weight"):
            continue
        try:
            names.setdefault(module_suffix(name), []).append((name, mod))
        except ValueError:
            pass

    resolved: list[tuple[str, torch.nn.Module]] = []
    for e in edits:
        mods = names.get(e.module, [])
        if len(mods) != 1:
            raise KeyError(f"{e.module}: {len(mods)} matching modules in the model")
        full_name, mod = mods[0]
        if verify_fingerprint and e.base_sha256:
            base = mod.weight.detach().to(_DTYPES[e.base_dtype])
            if _sha256(base) != e.base_sha256:
                raise ValueError(
                    f"{e.module}: weights differ from the patch's base model (fingerprint mismatch); "
                    f"refusing to attach"
                )
        resolved.append((full_name, mod))
    return resolved


class PatchHandle:
    """A patch attached to a loaded model. Starts OFF. `on()` patches the weights in place, `off()` restores the exact
    original bytes. The fingerprint is checked on attach, so a patch never touches a different model."""

    def __init__(self, patch: MoIPatch, model, *, verify_fingerprint: bool = True):
        self.patch, self.model, self.active = patch, model, False
        self.params, self.orig, self.new = [], [], []
        for _full_name, mod in _resolve_model_modules(model, patch.edits, verify_fingerprint=verify_fingerprint):
            e = patch.edits[len(self.params)]
            p = mod.weight
            self.params.append(p)
            self.orig.append(p.detach().cpu().clone())
            self.new.append(apply_edit(self.orig[-1], e.u, e.idx, e.alpha, e.v))

    def __enter__(self) -> "PatchHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.off()

    @torch.no_grad()
    def on(self) -> None:
        for p, w in zip(self.params, self.new):
            p.data.copy_(w.to(p.device))
        self.active = True

    @torch.no_grad()
    def off(self) -> None:
        for p, w in zip(self.params, self.orig):
            p.data.copy_(w.to(p.device))
        self.active = False

    def check_restored(self) -> bool:
        return all(torch.equal(p.detach().cpu(), w) for p, w in zip(self.params, self.orig))

    def fingerprint_ok(self) -> bool:
        return all(_sha256(p.detach().to(_DTYPES[e.base_dtype])) == e.base_sha256
                   for p, e in zip(self.params, self.patch.edits) if e.base_sha256)


# ------------------------------------------------------------------ 1-line live routed runtime
class _RouterProbeDone(Exception):
    """Sentinel used to stop a residual probe immediately once the router layer finishes."""


def _tensor_out(o):
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (tuple, list)) and o and isinstance(o[0], torch.Tensor):
        return o[0]
    return getattr(o, "last_hidden_state", None) if not isinstance(o, dict) else None


def _layer_index_from_suffix(suffix: str) -> int:
    m = re.search(r"layers\.(\d+)\.", suffix)
    if not m:
        raise ValueError(f"cannot parse layer index from module suffix: {suffix}")
    return int(m.group(1))


def _find_decoder_blocks(model, matched_full_names: list[str]) -> torch.nn.ModuleList:
    for full_name in matched_full_names:
        m = re.search(r"^(.*\blayers)\.\d+\.", full_name)
        if m:
            try:
                sub = model.get_submodule(m.group(1))
                if isinstance(sub, torch.nn.ModuleList) and len(sub) > 0:
                    return sub
            except AttributeError:
                pass
    best = max(
        (
            (n, m)
            for n, m in model.named_modules()
            if isinstance(m, torch.nn.ModuleList) and "visual" not in n and "mtp" not in n
        ),
        key=lambda nm: len(nm[1]),
        default=(None, None),
    )
    if best[1] is None or len(best[1]) == 0:
        raise ValueError("no ModuleList of decoder blocks found")
    return best[1]


def _last_token_hidden(h: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
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


def _extract_input_ids(args: tuple, kwargs: dict) -> torch.Tensor | None:
    ids = kwargs.get("input_ids")
    if isinstance(ids, torch.Tensor):
        return ids
    inp = kwargs.get("inputs")
    if isinstance(inp, torch.Tensor):
        return inp
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    return None


def _extract_batch_size(args: tuple, kwargs: dict) -> int:
    ids = _extract_input_ids(args, kwargs)
    if ids is not None and ids.dim() >= 1:
        return int(ids.shape[0])
    emb = kwargs.get("inputs_embeds")
    if isinstance(emb, torch.Tensor) and emb.dim() >= 1:
        return int(emb.shape[0])
    return 1


def _collect_eos_ids(model, extra_eos_ids=None) -> set[int]:
    ids: set[int] = set()
    for src in (getattr(model, "generation_config", None), getattr(model, "config", None)):
        if src is None:
            continue
        for attr in ("eos_token_id", "pad_token_id"):
            val = getattr(src, attr, None)
            if isinstance(val, int):
                ids.add(int(val))
            elif isinstance(val, (list, tuple, set)):
                for x in val:
                    if isinstance(x, int):
                        ids.add(int(x))
    if extra_eos_ids is not None:
        if isinstance(extra_eos_ids, int):
            ids.add(int(extra_eos_ids))
        else:
            for x in extra_eos_ids:
                ids.add(int(x))
    return ids


def _broadcast_mask(mask: torch.Tensor | None, target_bs: int) -> torch.Tensor | None:
    if mask is None or mask.shape[0] == target_bs:
        return mask
    if target_bs > 0 and mask.shape[0] > 0 and target_bs % mask.shape[0] == 0:
        return mask.repeat_interleave(target_bs // mask.shape[0])
    return None


class _LoopTracker:
    """Causal per-sequence repeated n-gram detector over generated response tokens (O(ngram) per step)."""

    def __init__(self, n_rows: int, *, ngram: int = 16, window: int = 64, thr: float = 0.5, eos_ids: set[int] | None = None):
        self.n_rows = max(1, int(n_rows))
        self.ngram = max(1, int(ngram))
        self.window = max(1, int(window))
        self.need = float(thr) * self.window
        self.eos_ids = set(eos_ids or ())
        self._init_rows(self.n_rows)

    def _init_rows(self, n: int) -> None:
        self.n_rows = n
        self.tails: list[deque[int]] = [deque(maxlen=self.ngram) for _ in range(n)]
        self.seen: list[set[tuple[int, ...]]] = [set() for _ in range(n)]
        self.win_flags: list[deque[int]] = [deque(maxlen=self.window) for _ in range(n)]
        self.win_sum: list[int] = [0] * n
        self.resp_len: list[int] = [0] * n
        self.finished: list[bool] = [False] * n
        self.latched: list[bool] = [False] * n
        self.fired_at: list[int | None] = [None] * n

    def resize(self, new_n: int) -> None:
        if new_n == self.n_rows:
            return
        if self.n_rows > 0 and new_n % self.n_rows == 0 and all(r == 0 for r in self.resp_len):
            self._init_rows(new_n)
            return
        if self.n_rows > 0 and new_n % self.n_rows == 0:
            rep = new_n // self.n_rows
            self.tails = [deque(t, maxlen=self.ngram) for t in self.tails for _ in range(rep)]
            self.seen = [set(s) for s in self.seen for _ in range(rep)]
            self.win_flags = [deque(w, maxlen=self.window) for w in self.win_flags for _ in range(rep)]
            self.win_sum = [v for v in self.win_sum for _ in range(rep)]
            self.resp_len = [v for v in self.resp_len for _ in range(rep)]
            self.finished = [v for v in self.finished for _ in range(rep)]
            self.latched = [v for v in self.latched for _ in range(rep)]
            self.fired_at = [v for v in self.fired_at for _ in range(rep)]
            self.n_rows = new_n
            return
        self._init_rows(new_n)

    def step(self, new_toks: list[int]) -> torch.Tensor:
        if len(new_toks) != self.n_rows:
            self.resize(len(new_toks))
        for b, tok in enumerate(new_toks):
            if self.finished[b]:
                continue
            if tok in self.eos_ids:
                self.finished[b] = True
                continue
            tail = self.tails[b]
            tail.append(int(tok))
            if len(tail) == self.ngram:
                gram = tuple(tail)
                dup = 1 if gram in self.seen[b] else 0
                self.seen[b].add(gram)
            else:
                dup = 0
            wf = self.win_flags[b]
            if len(wf) == self.window:
                self.win_sum[b] -= wf[0]
            wf.append(dup)
            self.win_sum[b] += dup
            self.resp_len[b] += 1
            if not self.latched[b] and len(wf) >= self.window and self.win_sum[b] >= self.need:
                self.latched[b] = True
                self.fired_at[b] = self.resp_len[b]
        return torch.tensor(self.latched, dtype=torch.bool)


class _ModelRouteStack:
    """Shared `forward` / `generate` dispatcher allowing clean composition and any-order close on one model."""

    def __init__(self, model) -> None:
        self.model = model
        self.handles: list[RoutedPatchHandle] = []
        self._suspend_edits = False

        self._had_inst_forward = "forward" in model.__dict__
        self.orig_forward = model.forward
        self._had_inst_generate = "generate" in model.__dict__
        self.orig_generate = getattr(model, "generate", None)

        try:
            sig = inspect.signature(self.orig_forward)
            self._forward_params = set(sig.parameters)
            self._forward_param_names = list(sig.parameters)
            self._accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            self._forward_params = set()
            self._forward_param_names = []
            self._accepts_var_kw = True

        orig_fwd = self.orig_forward

        @functools.wraps(orig_fwd)
        def wrapped_forward(*args, **kwargs):
            return self._dispatch_forward(*args, **kwargs)

        model.forward = wrapped_forward

        if self.orig_generate is not None:
            orig_gen = self.orig_generate

            @functools.wraps(orig_gen)
            def wrapped_generate(*args, **kwargs):
                return self._dispatch_generate(*args, **kwargs)

            model.generate = wrapped_generate

    def add(self, handle: "RoutedPatchHandle") -> None:
        self.handles.append(handle)

    def remove(self, handle: "RoutedPatchHandle") -> None:
        if handle in self.handles:
            self.handles.remove(handle)
        if not self.handles:
            if self._had_inst_forward:
                self.model.forward = self.orig_forward
            else:
                self.model.__dict__.pop("forward", None)
            if self.orig_generate is not None:
                if self._had_inst_generate:
                    self.model.generate = self.orig_generate
                else:
                    self.model.__dict__.pop("generate", None)
            self.model.__dict__.pop("_wsi_route_stack", None)

    def _has_cache_arg(self, args: tuple, kwargs: dict) -> bool:
        if kwargs.get("past_key_values") is not None:
            return True
        if "past_key_values" in self._forward_param_names:
            idx = self._forward_param_names.index("past_key_values")
            if idx < len(args) and args[idx] is not None:
                return True
        return False

    def accepts(self, name: str) -> bool:
        return name in self._forward_params or self._accepts_var_kw

    def _dispatch_generate(self, *args, **kwargs):
        handles = list(self.handles)
        for h in handles:
            h._begin_generate(self, args, kwargs)
        try:
            return self.orig_generate(*args, **kwargs)
        finally:
            for h in handles:
                h._end_generate()

    def _dispatch_forward(self, *args, **kwargs):
        handles = list(self.handles)
        if any(h._in_generate for h in handles):
            for h in handles:
                h._before_generate_step(args, kwargs)
            return self.orig_forward(*args, **kwargs)

        has_cache = self._has_cache_arg(args, kwargs)
        late = [h for h in handles if h._needs_pre_probe]
        if torch.is_grad_enabled() or has_cache:
            must_probe = late
            hook_late: list[RoutedPatchHandle] = []
        else:
            must_probe = [h for h in late if h._router_layer == h._n_blocks]
            hook_late = [h for h in late if h._router_layer < h._n_blocks]
        fast = [h for h in handles if not h._needs_pre_probe]

        try:
            for h in must_probe:
                h._probe_from_forward_args(self, args, kwargs)

            if not hook_late:
                for h in fast:
                    h._begin_standalone_forward(args, kwargs)
                return self.orig_forward(*args, **kwargs)

            for h in fast:
                h._begin_standalone_forward(args, kwargs)
            for h in hook_late:
                h._begin_standalone_capture(args, kwargs)

            out = self.orig_forward(*args, **kwargs)
            any_fired = any(
                h.mode == "residual_router" and h._residual_mask is not None and bool(h._residual_mask.any())
                for h in hook_late
            )
            if not any_fired:
                return out

            for h in fast:
                h._freeze_standalone_decision()
            for h in hook_late:
                h._promote_standalone_capture()
            return self.orig_forward(*args, **kwargs)
        finally:
            for h in handles:
                h._end_standalone_forward()


class RoutedPatchHandle:
    """Non-mutating live MoI router handle for `.moipatch` patches on `model(...)` and `model.generate(...)`."""

    def __init__(
        self,
        patch: MoIPatch,
        model,
        *,
        mode: str = "auto",
        threshold: float | None = None,
        confidence: float | torch.Tensor | None = None,
        loop_router: bool | dict | None = None,
        eos_token_ids=None,
        verify_fingerprint: bool = True,
    ) -> None:
        if mode not in ("auto", "always_on", "off"):
            raise ValueError(f"unsupported mode {mode!r} (expected 'auto', 'always_on', or 'off')")
        if not patch.edits:
            raise ValueError("patch has no edits")

        self.patch = patch
        self.model = model
        self.closed = False
        self._confidence = confidence

        # Validate edits & fingerprints before registering any hooks
        resolved = _resolve_model_modules(model, patch.edits, verify_fingerprint=verify_fingerprint)
        matched_full_names = [fn for fn, _ in resolved]
        self._watched: list[tuple[torch.nn.Parameter, str, str]] = [
            (mod.weight, e.base_dtype, e.base_sha256)
            for e, (_fn, mod) in zip(patch.edits, resolved)
        ]

        # Build sequential rank-1 additive payloads per unique module on each module's own device
        running_w: dict[str, torch.Tensor] = {}
        module_objs: dict[str, torch.nn.Module] = {}
        module_deltas: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]] = {}
        for e, (_fn, mod) in zip(patch.edits, resolved):
            module_objs[e.module] = mod
            if e.module not in running_w:
                running_w[e.module] = mod.weight.detach().cpu().clone()
            Wd = running_w[e.module].double()
            uu = e.u.detach().cpu().double()
            dev = mod.weight.device
            u_scaled = (-float(e.alpha) * uu).float().to(dev)
            if e.v is not None:
                idx_dev = None
                v_dev = e.v.detach().cpu().double().reshape(-1).float().to(dev)
            elif e.idx is not None:
                idx_cpu = e.idx.detach().long().cpu()
                idx_dev = idx_cpu.to(dev)
                v_dev = (uu @ Wd[:, idx_cpu]).float().to(dev)
            else:
                idx_dev = None
                v_dev = (uu @ Wd).float().to(dev)
            module_deltas.setdefault(e.module, []).append((u_scaled, v_dev, idx_dev))
            running_w[e.module] = apply_edit(running_w[e.module], e.u, e.idx, e.alpha, e.v)

        # Resolve router mode & gates
        has_residual = bool(patch.router and "dir" in patch.router and "layer" in patch.router)
        ngram_cfg: dict | None = None
        if loop_router is False:
            ngram_cfg = None
        elif isinstance(loop_router, dict):
            ngram_cfg = {
                "ngram": int(loop_router.get("ngram", 16)),
                "window": int(loop_router.get("window", 64)),
                "thr": float(loop_router.get("thr", 0.5)),
            }
        elif loop_router is True:
            ngram_cfg = {"ngram": 16, "window": 64, "thr": 0.5}
        else:
            for g in patch.gates:
                if g.get("type") == "ngram_repetition_frac":
                    ngram_cfg = {
                        "ngram": int(g.get("ngram", 16)),
                        "window": int(g.get("window", 64)),
                        "thr": float(g.get("thr", 0.5)),
                    }
                    break

        self._conf_thr = next(
            (float(g["thr"]) for g in patch.gates if g.get("type") == "answer_confidence" and "thr" in g),
            None,
        )
        self._rdir = patch.router["dir"].detach().float().cpu() if has_residual else None
        self._rthr = float(patch.router["thr"]) if has_residual else 0.0
        has_veto = bool(has_residual and patch.router.get("veto_dir") is not None and "veto_thr" in patch.router)
        self._vdir = patch.router["veto_dir"].detach().float().cpu() if has_veto else None
        self._vthr = float(patch.router["veto_thr"]) if has_veto else 0.0
        if has_residual and threshold is not None:
            self._rthr = float(threshold)
        elif ngram_cfg is not None and threshold is not None:
            ngram_cfg["thr"] = float(threshold)

        if mode in ("always_on", "off"):
            self.mode = mode
        elif has_residual and ngram_cfg is not None:
            self.mode = "residual+ngram"
        elif has_residual:
            self.mode = "residual_router"
        elif ngram_cfg is not None:
            self.mode = "ngram_loop"
        else:
            self.mode = "always_on"

        self._ngram_cfg = ngram_cfg
        self._eos_ids = _collect_eos_ids(model, eos_token_ids)

        # Configure residual router target layer & probe requirement before registering hooks
        self._router_layer = -1
        self._n_blocks = 0
        self._needs_pre_probe = False
        blocks = None
        if self.mode in ("residual_router", "residual+ngram"):
            blocks = _find_decoder_blocks(model, matched_full_names)
            self._n_blocks = len(blocks)
            self._router_layer = int(patch.router["layer"])
            if self._router_layer < 0 or self._router_layer > self._n_blocks:
                raise ValueError(
                    f"router layer {self._router_layer} out of range for model with {self._n_blocks} blocks"
                )
            min_edit_layer = min(_layer_index_from_suffix(e.module) for e in patch.edits)
            self._needs_pre_probe = self._router_layer > min_edit_layer

        # Runtime state
        self._in_generate = False
        self._gen_step = 0
        self._gen_attn_mask: torch.Tensor | None = None
        self._cur_attention_mask: torch.Tensor | None = None
        self._armed_single_pass = False
        self._armed_capture_only = False
        self._frozen_standalone = False
        self._probe_mode = False
        self._probe_h: torch.Tensor | None = None
        self._active_mask: torch.Tensor | None = None
        self._residual_mask: torch.Tensor | None = None
        self._ngram_mask: torch.Tensor | None = None
        self._loop_tracker: _LoopTracker | None = None

        self._stats_calls = 0
        self._stats_fired = 0
        self._last_scores: list[float] | None = None
        self._last_fired: list[bool] | None = None
        self._last_fired_at: list[int | None] | None = None

        self._hook_handles: list = []
        self._stack: _ModelRouteStack | None = None

        try:
            for mod_name, deltas in module_deltas.items():
                mod = module_objs[mod_name]
                self._hook_handles.append(mod.register_forward_hook(self._make_edit_hook(deltas)))

            if blocks is not None and self._router_layer < self._n_blocks:
                if self._router_layer == 0:
                    def _pre_hook(_mod, inp):
                        h = inp[0] if isinstance(inp, tuple) and inp else inp
                        if isinstance(h, torch.Tensor):
                            self._on_router_hidden(h)

                    self._hook_handles.append(blocks[0].register_forward_pre_hook(_pre_hook))
                else:
                    def _blk_hook(_mod, _inp, out):
                        h = _tensor_out(out)
                        if isinstance(h, torch.Tensor):
                            self._on_router_hidden(h)

                    self._hook_handles.append(blocks[self._router_layer - 1].register_forward_hook(_blk_hook))

            stack = model.__dict__.get("_wsi_route_stack")
            if stack is None:
                stack = _ModelRouteStack(model)
                model._wsi_route_stack = stack
            stack.add(self)
            self._stack = stack
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------ public API
    @property
    def active(self) -> bool:
        if self.closed or self.mode == "off":
            return False
        if self.mode == "always_on":
            return True
        return self._active_mask is not None and bool(self._active_mask.any())

    @property
    def active_mask(self) -> torch.Tensor | None:
        return None if self._active_mask is None else self._active_mask.clone()

    @property
    def stats(self) -> dict:
        return {
            "mode": self.mode,
            "calls": int(self._stats_calls),
            "fired": int(self._stats_fired),
            "fire_rate": (self._stats_fired / self._stats_calls) if self._stats_calls else 0.0,
            "last_scores": None if self._last_scores is None else list(self._last_scores),
            "last_fired": None if self._last_fired is None else list(self._last_fired),
            "last_fired_at": None if self._last_fired_at is None else list(self._last_fired_at),
        }

    def set_confidence(self, confidence: float | torch.Tensor | None) -> None:
        self._confidence = confidence

    def reset_stats(self) -> None:
        self._stats_calls = 0
        self._stats_fired = 0
        self._last_scores = None
        self._last_fired = None
        self._last_fired_at = None

    def reset_state(self) -> None:
        self._end_generate()

    def fingerprint_ok(self) -> bool:
        return all(
            _sha256(p.detach().to(_DTYPES[dt])) == sha
            for p, dt, sha in self._watched
            if sha
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._active_mask = None
        self._residual_mask = None
        self._ngram_mask = None
        self._loop_tracker = None
        for hh in self._hook_handles:
            try:
                hh.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        if self._stack is not None:
            self._stack.remove(self)
            self._stack = None

    def __enter__(self) -> "RoutedPatchHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ internals
    def _make_edit_hook(self, deltas: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]):
        def _apply(x_in: torch.Tensor, out_ref: torch.Tensor) -> torch.Tensor:
            add = torch.zeros(out_ref.shape, device=out_ref.device, dtype=torch.float32)
            for u_s, v_d, idx_d in deltas:
                xs = x_in[..., idx_d] if idx_d is not None else x_in
                add = add + torch.matmul(xs.float(), v_d).unsqueeze(-1) * u_s
            return (out_ref.float() + add).to(out_ref.dtype)

        def _edit_hook(_mod, inp, out):
            if self.closed or self.mode == "off" or not isinstance(out, torch.Tensor):
                return out
            if self._stack is not None and self._stack._suspend_edits:
                return out
            if self._armed_capture_only:
                return out

            x = inp[0]
            if self.mode == "always_on":
                return _apply(x, out)

            mask = _broadcast_mask(self._active_mask, x.shape[0])
            if mask is None or not mask.any():
                return out
            if mask.all():
                return _apply(x, out)

            m_dev = mask.to(out.device)
            out_new = out.clone()
            out_new[m_dev] = _apply(x[m_dev], out[m_dev])
            return out_new

        return _edit_hook

    def _eval_residual(self, h: torch.Tensor, attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        h_last = _last_token_hidden(h, attention_mask).float().cpu()
        scores = h_last @ self._rdir
        fired = scores > self._rthr
        if self._vdir is not None:
            fired = fired & ((h_last @ self._vdir) <= self._vthr)
        if self._confidence is not None and self._conf_thr is not None:
            conf = torch.as_tensor(self._confidence, dtype=torch.float32).cpu()
            if conf.dim() == 0 or (conf.numel() == 1 and fired.numel() > 1):
                fired = fired & (conf.reshape(()) > self._conf_thr)
            elif conf.numel() == fired.numel():
                fired = fired & (conf.reshape(-1) > self._conf_thr)
            else:
                raise ValueError(
                    f"confidence batch size ({conf.numel()}) does not match input batch size ({fired.numel()})"
                )
        return scores, fired

    def _record_residual(self, h: torch.Tensor, attention_mask: torch.Tensor | None, *, capture_only: bool) -> torch.Tensor:
        scores, fired = self._eval_residual(h, attention_mask)
        fired_list = fired.tolist()
        self._stats_calls += len(fired_list)
        self._last_scores = [float(x) for x in scores.tolist()]
        if capture_only and self.mode == "residual+ngram":
            self._last_fired = [False] * len(fired_list)
        else:
            if not capture_only:
                self._stats_fired += sum(1 for x in fired_list if x)
            self._last_fired = list(fired_list)
        self._residual_mask = fired
        return fired

    def _on_router_hidden(self, h: torch.Tensor) -> None:
        if self._probe_mode:
            self._probe_h = h.detach()
            raise _RouterProbeDone()
        if self._armed_single_pass or self._armed_capture_only:
            cap_only = self._armed_capture_only
            self._armed_single_pass = False
            self._armed_capture_only = False
            fired = self._record_residual(h.detach(), self._cur_attention_mask, capture_only=cap_only)
            if not cap_only:
                self._active_mask = fired

    def _run_residual_probe(
        self,
        stack: _ModelRouteStack,
        probe_args: tuple,
        probe_kwargs: dict,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        need_hs = self._router_layer == self._n_blocks
        with torch.no_grad():
            self._probe_mode = True
            self._probe_h = None
            stack._suspend_edits = True
            try:
                out = stack.orig_forward(*probe_args, **probe_kwargs)
            except _RouterProbeDone:
                out = None
            finally:
                stack._suspend_edits = False
                self._probe_mode = False

        if need_hs:
            hs = getattr(out, "hidden_states", None)
            if hs is None or len(hs) <= self._router_layer:
                raise RuntimeError("model did not return hidden_states needed for final-layer router")
            h_target = hs[self._router_layer].detach()
        else:
            if self._probe_h is None:
                raise RuntimeError(f"router hook at layer {self._router_layer} did not execute during probe")
            h_target = self._probe_h
        self._probe_h = None
        return self._record_residual(
            h_target,
            attention_mask,
            capture_only=(self.mode == "residual+ngram"),
        )

    def _probe_from_forward_args(self, stack: _ModelRouteStack, args: tuple, kwargs: dict) -> None:
        p_kw = dict(kwargs)
        if stack.accepts("use_cache"):
            p_kw["use_cache"] = False
        if self._router_layer == self._n_blocks and stack.accepts("output_hidden_states"):
            p_kw["output_hidden_states"] = True
        fired = self._run_residual_probe(stack, args, p_kw, kwargs.get("attention_mask"))
        if self.mode == "residual_router":
            self._active_mask = fired
        else:
            self._active_mask = torch.zeros_like(fired)
        self._frozen_standalone = True

    def _begin_standalone_forward(self, args: tuple, kwargs: dict) -> None:
        if self._frozen_standalone or self.mode == "off":
            return
        self._cur_attention_mask = kwargs.get("attention_mask")
        if self.mode == "always_on":
            bs = _extract_batch_size(args, kwargs)
            self._stats_calls += bs
            self._stats_fired += bs
            self._last_fired = [True] * bs
            return
        if self.mode == "residual_router" and not self._needs_pre_probe:
            self._armed_single_pass = True
        elif self.mode == "residual+ngram":
            self._active_mask = None

    def _begin_standalone_capture(self, args: tuple, kwargs: dict) -> None:
        if self.mode == "off":
            return
        self._cur_attention_mask = kwargs.get("attention_mask")
        self._residual_mask = None
        self._active_mask = None
        self._armed_capture_only = True

    def _freeze_standalone_decision(self) -> None:
        self._armed_single_pass = False
        self._armed_capture_only = False
        self._frozen_standalone = True

    def _promote_standalone_capture(self) -> None:
        self._armed_capture_only = False
        if self._residual_mask is not None and self.mode == "residual_router":
            self._active_mask = self._residual_mask
            self._stats_fired += sum(1 for x in self._residual_mask.tolist() if x)
        self._frozen_standalone = True

    def _end_standalone_forward(self) -> None:
        self._armed_single_pass = False
        self._armed_capture_only = False
        self._frozen_standalone = False
        self._cur_attention_mask = None
        self._active_mask = None
        self._residual_mask = None

    def _begin_generate(self, stack: _ModelRouteStack, gen_args: tuple, gen_kwargs: dict) -> None:
        self._in_generate = True
        self._gen_step = 0
        self._gen_attn_mask = gen_kwargs.get("attention_mask")
        self._active_mask = None
        self._residual_mask = None
        self._ngram_mask = None
        if self.mode == "off":
            return

        bs = _extract_batch_size(gen_args, gen_kwargs)
        if self.mode in ("ngram_loop", "residual+ngram") and self._ngram_cfg is not None:
            self._loop_tracker = _LoopTracker(bs, **self._ngram_cfg, eos_ids=self._eos_ids)
            self._ngram_mask = torch.zeros(bs, dtype=torch.bool)

        if self._needs_pre_probe:
            input_ids = _extract_input_ids(gen_args, gen_kwargs)
            p_kw: dict = {}
            if input_ids is not None:
                p_kw["input_ids"] = input_ids
            elif isinstance(gen_kwargs.get("inputs_embeds"), torch.Tensor) and stack.accepts("inputs_embeds"):
                p_kw["inputs_embeds"] = gen_kwargs["inputs_embeds"]
            else:
                return
            for k in ("attention_mask", "position_ids", "token_type_ids"):
                if gen_kwargs.get(k) is not None and stack.accepts(k):
                    p_kw[k] = gen_kwargs[k]
            if stack.accepts("use_cache"):
                p_kw["use_cache"] = False
            if self._router_layer == self._n_blocks and stack.accepts("output_hidden_states"):
                p_kw["output_hidden_states"] = True
            fired = self._run_residual_probe(stack, (), p_kw, self._gen_attn_mask)
            if self.mode == "residual_router":
                self._active_mask = fired
            elif self._ngram_mask is not None:
                self._active_mask = torch.zeros_like(fired)

    def _before_generate_step(self, args: tuple, kwargs: dict) -> None:
        if self.mode == "off":
            self._gen_step += 1
            return

        bs = _extract_batch_size(args, kwargs)
        if self._gen_step == 0:
            self._cur_attention_mask = kwargs.get("attention_mask", self._gen_attn_mask)
            if self.mode == "always_on":
                self._stats_calls += bs
                self._stats_fired += bs
                self._last_fired = [True] * bs
            elif self.mode == "residual_router" and not self._needs_pre_probe:
                self._armed_single_pass = True
            elif self.mode == "ngram_loop":
                if self._loop_tracker is None:
                    self._loop_tracker = _LoopTracker(bs, **self._ngram_cfg, eos_ids=self._eos_ids)
                else:
                    self._loop_tracker.resize(bs)
                self._ngram_mask = torch.zeros(bs, dtype=torch.bool)
                self._active_mask = self._ngram_mask
                self._stats_calls += bs
                self._last_fired = [False] * bs
                self._last_fired_at = [None] * bs
            elif self.mode == "residual+ngram":
                if self._loop_tracker is None:
                    self._loop_tracker = _LoopTracker(bs, **self._ngram_cfg, eos_ids=self._eos_ids)
                else:
                    self._loop_tracker.resize(bs)
                self._ngram_mask = torch.zeros(bs, dtype=torch.bool)
                if not self._needs_pre_probe:
                    self._armed_capture_only = True
                self._residual_mask = _broadcast_mask(self._residual_mask, bs)
                self._active_mask = self._ngram_mask
                self._last_fired = [False] * bs
                self._last_fired_at = [None] * bs
        else:
            self._armed_single_pass = False
            self._armed_capture_only = False
            if self.mode in ("ngram_loop", "residual+ngram") and self._loop_tracker is not None:
                ids = _extract_input_ids(args, kwargs)
                if ids is not None and ids.shape[1] >= 1:
                    new_toks = ids[:, -1].detach().cpu().tolist()
                    latched = self._loop_tracker.step(new_toks)
                    self._ngram_mask = latched
                    if self.mode == "ngram_loop":
                        self._active_mask = latched
                    else:
                        rm = _broadcast_mask(self._residual_mask, len(latched))
                        self._residual_mask = rm
                        self._active_mask = latched if rm is None else (rm & latched)
                    new_active = self._active_mask.tolist()
                    if self._last_fired is None or len(self._last_fired) != len(new_active):
                        self._last_fired = list(new_active)
                        self._stats_fired += sum(1 for x in new_active if x)
                    else:
                        for b, (old_f, new_f) in enumerate(zip(self._last_fired, new_active)):
                            if (not old_f) and new_f:
                                self._stats_fired += 1
                                self._last_fired[b] = True
                    self._last_fired_at = list(self._loop_tracker.fired_at)

        self._gen_step += 1

    def _end_generate(self) -> None:
        self._in_generate = False
        self._gen_step = 0
        self._gen_attn_mask = None
        self._cur_attention_mask = None
        self._armed_single_pass = False
        self._armed_capture_only = False
        self._frozen_standalone = False
        self._active_mask = None
        self._residual_mask = None
        self._ngram_mask = None
        self._loop_tracker = None


# ------------------------------------------------------------------ anti-backdoor audit
def _find_lm_head(model):
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None and hasattr(head, "weight"):
            return head
    head = getattr(model, "lm_head", None)
    return head if (head is not None and hasattr(head, "weight")) else None


def _find_final_norm(model):
    inner = getattr(model, "model", model)
    for attr in ("embedding_norm", "norm", "final_layernorm", "ln_f"):
        mod = getattr(inner, attr, None)
        if mod is not None:
            return mod
    return None


def _decode_tok(tokenizer, tid: int) -> str:
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            return repr(tokenizer.decode([tid]))
        except Exception:
            pass
    return f"tok_{tid}"


def _encode_audit_prompts(tokenizer, prompts: list[str]) -> list[list[int]]:
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise ValueError("tokenizer with .encode() is required when prompt strings are passed to audit_moipatch")
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    fallback = int(eos) if eos is not None else 1
    encoded: list[list[int]] = []
    for p in prompts:
        try:
            ids = tokenizer.encode(p, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(p)
        encoded.append([int(t) for t in ids] if ids else [fallback])
    return encoded


def _next_token_logits(model, ids: list[int], dev: torch.device) -> torch.Tensor:
    inp = torch.tensor([ids], dtype=torch.long, device=dev)
    try:
        out = model(input_ids=inp, use_cache=False)
    except TypeError:
        out = model(inp)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[0, -1].detach().float().cpu()


def _kl_from_logprobs(lp_base: torch.Tensor, lp_arm: torch.Tensor) -> float:
    return float((lp_base.exp() * (lp_base - lp_arm)).sum().clamp_min(0.0).item())


def _build_vocab_logit_lens(patch: MoIPatch, model, tokenizer, top_k: int) -> list[dict]:
    lm_head = _find_lm_head(model)
    if lm_head is None:
        return []
    try:
        n_layers: int | None = len(_find_decoder_blocks(model, []))
    except ValueError:
        n_layers = None

    w_out = lm_head.weight.detach().float().cpu()
    norm_mod = _find_final_norm(model)
    norm_param = next(norm_mod.parameters(), None) if norm_mod is not None else None
    k = min(max(1, int(top_k)), int(w_out.shape[0]))
    entries: list[dict] = []

    for e in patch.edits:
        u = e.u.detach().float().cpu()
        u = u / u.norm().clamp_min(1e-12)
        u_proj = u
        if norm_mod is not None and norm_param is not None:
            try:
                u_in = u.unsqueeze(0).to(device=norm_param.device, dtype=norm_param.dtype)
                u_proj = norm_mod(u_in)[0].detach().float().cpu()
            except Exception:
                u_proj = u

        # Edit subtracts (-alpha * u ...): +u aligned tokens are suppressed, -u aligned tokens are promoted
        scores = w_out @ u_proj
        sup_vals, sup_ids = torch.topk(scores, k)
        pro_vals, pro_ids = torch.topk(-scores, k)

        try:
            layer_idx: int | None = _layer_index_from_suffix(e.module)
        except ValueError:
            layer_idx = None
        layer_frac = round(layer_idx / max(1, n_layers - 1), 3) if (layer_idx is not None and n_layers) else None

        entries.append({
            "module": e.module,
            "layer": layer_idx,
            "layer_frac": layer_frac,
            "early_layer_caveat": bool(layer_frac is not None and layer_frac < 0.65),
            "alpha": float(e.alpha),
            "cols": len(e.idx) if e.idx is not None else "all",
            "suppressed_tokens": [
                {"token_id": int(i), "token": _decode_tok(tokenizer, int(i)), "score": round(float(v), 4)}
                for v, i in zip(sup_vals, sup_ids)
            ],
            "promoted_tokens": [
                {"token_id": int(i), "token": _decode_tok(tokenizer, int(i)), "score": round(float(v), 4)}
                for v, i in zip(pro_vals, pro_ids)
            ],
        })
    return entries


def _build_output_logit_diff(
    patch: MoIPatch,
    h: PatchHandle,
    model,
    tokenizer,
    probe_ids: list[list[int]],
    dev: torch.device,
    top_k: int,
    verify_fingerprint: bool,
) -> dict:
    if h.active:
        h.off()
    base_rows = []
    for ids in probe_ids:
        lg = _next_token_logits(model, ids, dev)
        lp = F.log_softmax(lg, dim=-1)
        base_rows.append((lg, lp, lp.exp()))

    patched_logits: list[torch.Tensor] = []
    try:
        h.on()
        for ids in probe_ids:
            patched_logits.append(_next_token_logits(model, ids, dev))
    finally:
        if h.active:
            h.off()

    per_probe: list[dict] = []
    sum_d_logit = torch.zeros_like(base_rows[0][0])
    sum_d_logp = torch.zeros_like(base_rows[0][0])
    sum_d_prob = torch.zeros_like(base_rows[0][0])
    sum_base_prob = torch.zeros_like(base_rows[0][0])
    sum_patched_prob = torch.zeros_like(base_rows[0][0])
    kls: list[float] = []
    n_changed = 0

    for (lg_b, lp_b, p_b), lg_p in zip(base_rows, patched_logits):
        lp_p = F.log_softmax(lg_p, dim=-1)
        p_p = lp_p.exp()
        kl = _kl_from_logprobs(lp_b, lp_p)
        kls.append(kl)
        sum_d_logit += lg_p - lg_b
        sum_d_logp += lp_p - lp_b
        sum_d_prob += p_p - p_b
        sum_base_prob += p_b
        sum_patched_prob += p_p

        b_id = int(torch.argmax(lp_b).item())
        p_id = int(torch.argmax(lp_p).item())
        changed = b_id != p_id
        if changed:
            n_changed += 1
        per_probe.append({
            "base_top1_id": b_id,
            "base_top1_token": _decode_tok(tokenizer, b_id),
            "base_top1_prob": round(float(p_b[b_id].item()), 5),
            "patched_top1_id": p_id,
            "patched_top1_token": _decode_tok(tokenizer, p_id),
            "patched_top1_prob": round(float(p_p[p_id].item()), 5),
            "top1_changed": bool(changed),
            "kl": round(kl, 5),
        })

    n = len(probe_ids)
    mean_d_prob = sum_d_prob / n
    mean_d_logit = sum_d_logit / n
    mean_d_logp = sum_d_logp / n
    mean_base_prob = sum_base_prob / n
    mean_patched_prob = sum_patched_prob / n
    k = min(max(1, int(top_k)), int(mean_d_prob.shape[0]))

    def _pack_token_row(tid: int) -> dict:
        return {
            "token_id": tid,
            "token": _decode_tok(tokenizer, tid),
            "mean_delta_prob": round(float(mean_d_prob[tid].item()), 6),
            "mean_delta_logit": round(float(mean_d_logit[tid].item()), 4),
            "mean_delta_logp": round(float(mean_d_logp[tid].item()), 4),
            "mean_base_prob": round(float(mean_base_prob[tid].item()), 6),
            "mean_patched_prob": round(float(mean_patched_prob[tid].item()), 6),
        }

    sup_ids = torch.topk(-mean_d_prob, k).indices.tolist()
    pro_ids = torch.topk(mean_d_prob, k).indices.tolist()

    probe_fire_rate: float | None = None
    if patch.router is not None:
        fired_flags: list[bool] = []
        with patch.enable_routed(model, mode="auto", verify_fingerprint=verify_fingerprint) as routed:
            for ids in probe_ids:
                routed.reset_stats()
                _ = _next_token_logits(model, ids, dev)
                f_lst = routed.stats.get("last_fired") or [False]
                fired_flags.append(bool(f_lst[0]))
        probe_fire_rate = round(sum(fired_flags) / n, 4)

    return {
        "n_probes": n,
        "mean_next_token_kl": round(sum(kls) / n, 5),
        "top1_changed_rate": round(n_changed / n, 4),
        "probe_router_fire_rate": probe_fire_rate,
        "suppressed_tokens": [_pack_token_row(int(tid)) for tid in sup_ids],
        "promoted_tokens": [_pack_token_row(int(tid)) for tid in pro_ids],
        "per_probe": per_probe,
    }


def _build_control_audit(
    patch: MoIPatch,
    h: PatchHandle,
    model,
    control_ids: list[list[int]],
    dev: torch.device,
    verify_fingerprint: bool,
) -> dict:
    if h.active:
        h.off()
    base_rows = []
    for ids in control_ids:
        lg = _next_token_logits(model, ids, dev)
        base_rows.append((lg, F.log_softmax(lg, dim=-1)))

    always_kls: list[float] = []
    try:
        h.on()
        for idx, ids in enumerate(control_ids):
            lp_a = F.log_softmax(_next_token_logits(model, ids, dev), dim=-1)
            always_kls.append(_kl_from_logprobs(base_rows[idx][1], lp_a))
    finally:
        if h.active:
            h.off()

    n_c = len(control_ids)
    routed_fire_rate: float | None = None
    routed_mean_kl: float | None = None
    max_abs_diff_routed: float | None = None

    if patch.router is not None:
        fired_ctrl: list[bool] = []
        routed_kls: list[float] = []
        max_diffs: list[float] = []
        with patch.enable_routed(model, mode="auto", verify_fingerprint=verify_fingerprint) as routed:
            for idx, ids in enumerate(control_ids):
                routed.reset_stats()
                lg_r = _next_token_logits(model, ids, dev)
                f_lst = routed.stats.get("last_fired") or [False]
                fired_ctrl.append(bool(f_lst[0]))
                routed_kls.append(_kl_from_logprobs(base_rows[idx][1], F.log_softmax(lg_r, dim=-1)))
                max_diffs.append(float((lg_r - base_rows[idx][0]).abs().max().item()))
        routed_fire_rate = round(sum(fired_ctrl) / n_c, 4)
        routed_mean_kl = round(sum(routed_kls) / n_c, 5)
        max_abs_diff_routed = round(max(max_diffs), 6)

    return {
        "n_controls": n_c,
        "always_on_mean_kl": round(sum(always_kls) / n_c, 5),
        "routed_fire_rate": routed_fire_rate,
        "routed_mean_kl": routed_mean_kl,
        "max_abs_logit_diff_routed": max_abs_diff_routed,
    }


@torch.no_grad()
def audit_moipatch(
    patch: MoIPatch,
    model,
    tokenizer=None,
    *,
    probe_prompts: list[str] | None = None,
    probe_ids: list[list[int]] | None = None,
    control_prompts: list[str] | None = None,
    control_ids: list[list[int]] | None = None,
    patch_path: str | Path | None = None,
    top_k: int = 10,
    verify_fingerprint: bool = True,
) -> dict:
    """Transparent anti-backdoor audit for a `.moipatch` on top of an official base model.

    Verifies SHA-256 weight integrity and byte-exact ``on()/off()`` restoration, inspects parameter
    footprint, computes static weight-space ``vocab_logit_lens``, and (when probes/controls are given)
    measures end-to-end ``output_logit_diff`` (ranked by ``mean_delta_prob``) plus routed ``control_audit``.
    """
    total_model_params = sum(p.numel() for p in model.parameters())
    payload_kb = round(
        sum(e.u.numel() * e.u.element_size() + (e.idx.numel() * 8 if e.idx is not None else 0) for e in patch.edits) / 1024,
        2,
    )
    size_kb = round(Path(patch_path).stat().st_size / 1024, 2) if (patch_path and Path(patch_path).exists()) else None
    dev = next((p.device for p in model.parameters()), torch.device("cpu"))

    h = patch.attach(model, verify_fingerprint=verify_fingerprint)
    fp_ok = h.fingerprint_ok()
    edited_weights = 0
    edited_modules: list[dict] = []
    for e, p in zip(patch.edits, h.params):
        shp = tuple(e.shape) if e.shape else (tuple(p.shape) if p.dim() >= 2 else ())
        if len(shp) >= 2:
            edited_weights += int(shp[0]) * (len(e.idx) if e.idx is not None else int(shp[1]))
        else:
            edited_weights += int(e.u.numel())
        edited_modules.append({
            "module": e.module,
            "alpha": float(e.alpha),
            "cols": len(e.idx) if e.idx is not None else "all",
            "rank": 1,
            "shape": list(shp),
            "base_dtype": e.base_dtype,
            "base_sha256_prefix": e.base_sha256[:12] if e.base_sha256 else "",
        })

    output_logit_diff: dict | None = None
    control_audit: dict | None = None
    try:
        h.on()
        h.off()

        resolved_probe_ids = probe_ids if probe_ids is not None else (
            _encode_audit_prompts(tokenizer, probe_prompts) if probe_prompts else None
        )
        if resolved_probe_ids:
            output_logit_diff = _build_output_logit_diff(
                patch, h, model, tokenizer, resolved_probe_ids, dev, top_k, verify_fingerprint
            )

        resolved_ctrl_ids = control_ids if control_ids is not None else (
            _encode_audit_prompts(tokenizer, control_prompts) if control_prompts else None
        )
        if resolved_ctrl_ids:
            control_audit = _build_control_audit(
                patch, h, model, resolved_ctrl_ids, dev, verify_fingerprint
            )
    finally:
        if h.active:
            h.off()
    restore_ok = h.check_restored() and h.fingerprint_ok()

    vocab_logit_lens = _build_vocab_logit_lens(patch, model, tokenizer, top_k)

    return {
        "patch_size_kb": size_kb,
        "payload_kb": payload_kb,
        "n_edited_modules": len(patch.edits),
        "edited_modules": edited_modules,
        "edited_weights": int(edited_weights),
        "total_model_params": int(total_model_params),
        "edited_param_frac": round(edited_weights / max(total_model_params, 1), 6),
        "fingerprint_ok": bool(fp_ok),
        "restore_exact_ok": bool(restore_ok),
        "has_router": patch.router is not None,
        "has_router_veto": bool(patch.router and patch.router.get("veto_dir") is not None),
        "router_layer": patch.router.get("layer") if patch.router else None,
        "vocab_logit_lens": vocab_logit_lens,
        "output_logit_diff": output_logit_diff,
        "control_audit": control_audit,
    }


# ---------------------------------------------------------------------- CLI
def _apply_cli_chat(tok, prompt: str, use_chat: bool) -> str:
    if use_chat and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m wsi.moipatch", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("info", help="print what a patch does, its router, gates and expected results")
    i.add_argument("patch")
    v = sub.add_parser("verify", help="load the base model, check the fingerprint, switch on/off, check exact restore")
    v.add_argument("patch")
    v.add_argument("--model", required=True)
    v.add_argument("--cpu", action="store_true")
    au = sub.add_parser("audit", help="run anti-backdoor integrity, logit-lens, and end-to-end logit-diff audit")
    au.add_argument("patch")
    au.add_argument("--model", required=True)
    au.add_argument("--probe", action="append", default=None, help="trigger probe prompt (repeatable)")
    au.add_argument("--control", action="append", default=None, help="benign control prompt (repeatable)")
    au.add_argument("--chat", action="store_true", help="format prompts with tokenizer.apply_chat_template if available")
    au.add_argument("--top-k", type=int, default=10)
    au.add_argument("--cpu", action="store_true")
    au.add_argument("--json", action="store_true", dest="as_json")
    r = sub.add_parser("run", help="generate text with a .moipatch automatically routed on a local/HF model")
    r.add_argument("patch")
    r.add_argument("--model", required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--max-new-tokens", type=int, default=128)
    r.add_argument("--mode", choices=["auto", "always_on", "off"], default="auto")
    r.add_argument("--threshold", type=float, default=None)
    r.add_argument("--confidence", type=float, default=None)
    r.add_argument("--loop-router", action="store_true")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--chat", action="store_true", help="format prompt with tokenizer.apply_chat_template if available")
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--json", action="store_true", dest="as_json")

    ex = sub.add_parser("export", help="bake .moipatch edits directly into a standalone HuggingFace checkpoint directory")
    ex.add_argument("patch", nargs="+", help="one or more .moipatch files to bake sequentially")
    ex.add_argument("--model", required=True, help="base HuggingFace model directory containing safetensors shards")
    ex.add_argument("--out", required=True, help="destination directory for the standalone baked model")
    ex.add_argument("--force", action="store_true", help="allow baking routed patches statically as always_on")
    ex.add_argument("--copy", action="store_true", help="copy untouched shards instead of hardlinking")
    ex.add_argument("--no-verify", action="store_true", help="skip SHA-256 base weight fingerprint verification")
    ex.add_argument("--json", action="store_true", dest="as_json")

    sv = sub.add_parser("serve", help="serve a base model with OpenAI-compatible API, web chat UI, and live patch hot-swap")
    sv.add_argument("--model", required=True, help="base model path or HuggingFace repo ID")
    sv.add_argument("--patch", action="append", default=[], help="patch to preload as name=path.moipatch or path.moipatch")
    sv.add_argument("--activate", action="append", default=None, help="patch name(s) to activate at startup (default: all preloaded)")
    sv.add_argument("--mode", choices=["auto", "always_on", "off"], default="auto")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--cpu", action="store_true")
    sv.add_argument("--no-verify", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "export":
        from .serve import bake_moipatch

        rep = bake_moipatch(
            a.model,
            a.patch,
            a.out,
            verify_fingerprint=not a.no_verify,
            allow_routed=a.force,
            copy_untouched="copy" if a.copy else "auto",
        )
        if a.as_json:
            print(json.dumps(rep, indent=2))
        else:
            print(
                f"Baked {len(rep['patches'])} patch(es) into {rep['out_dir']} "
                f"({rep['n_touched_shards']}/{rep['n_total_shards']} shard(s) modified, {len(rep['modules'])} module(s))"
            )
            if rep["had_router_warning"]:
                print("WARNING: routed patch(es) were baked statically as always_on (router/veto disabled).", flush=True)
        return

    if a.cmd == "serve":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .serve import serve_moipatch

        patch_map: dict[str, str] = {}
        for raw_spec in a.patch:
            if "=" in raw_spec:
                nm, pth = raw_spec.split("=", 1)
                patch_map[nm.strip()] = pth.strip()
            else:
                patch_map[Path(raw_spec).stem] = raw_spec

        dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
        if patch_map:
            first_p = MoIPatch.load(next(iter(patch_map.values())))
            dt = _DTYPES[first_p.edits[0].base_dtype] if first_p.edits else torch.bfloat16
        else:
            dt = torch.bfloat16
        if dev == "cpu" and dt != torch.float32:
            dt = torch.float32
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
        tok = AutoTokenizer.from_pretrained(a.model)
        serve_moipatch(
            model,
            tok,
            patches=patch_map,
            activate=a.activate,
            mode=a.mode,
            host=a.host,
            port=a.port,
            base_model_name=Path(a.model).name or "base",
            verify_fingerprint=not a.no_verify,
        )
        return

    patch = MoIPatch.load(a.patch)
    if a.cmd == "info":
        print(patch.describe())
        return
    if a.cmd not in ("run", "audit"):
        print(patch.describe())
    from transformers import AutoModelForCausalLM

    dt = _DTYPES[patch.edits[0].base_dtype]
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    if dev == "cpu" and dt != torch.float32:
        dt = torch.float32  # CPU: fp32 holds every bf16 value exactly, so the fingerprint still matches
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
    if a.cmd == "verify":
        h = patch.attach(model)
        print("fingerprint: OK (weights match the patch's base model)")
        h.on()
        changed = sum(not torch.equal(p.detach().cpu(), w) for p, w in zip(h.params, h.orig))
        print(f"on: {changed}/{len(h.params)} weight matrices changed")
        h.off()
        ok = h.check_restored() and h.fingerprint_ok()
        print("off: original restored byte for byte" if ok else "off: RESTORE FAILED")
        if not ok:
            raise SystemExit(1)
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.cmd == "audit":
        probes = [_apply_cli_chat(tok, p, a.chat) for p in a.probe] if a.probe else None
        controls = [_apply_cli_chat(tok, p, a.chat) for p in a.control] if a.control else None
        report = patch.audit(
            model,
            tok,
            probe_prompts=probes,
            control_prompts=controls,
            patch_path=a.patch,
            top_k=a.top_k,
        )
        if a.as_json:
            print(json.dumps(report, indent=2))
        else:
            print(patch.describe())
            print(
                f"[audit] size_kb={report['patch_size_kb']} payload_kb={report['payload_kb']} "
                f"modules={report['n_edited_modules']} edited_frac={report['edited_param_frac']:.6f} "
                f"fp_ok={report['fingerprint_ok']} restore_ok={report['restore_exact_ok']}"
            )
            if report["output_logit_diff"]:
                old = report["output_logit_diff"]
                print(
                    f"[output_logit_diff] probes={old['n_probes']} mean_kl={old['mean_next_token_kl']} "
                    f"top1_changed={old['top1_changed_rate']:.1%} router_fire={old['probe_router_fire_rate']}"
                )
                print("  suppressed:", [(x["token"], x["mean_delta_prob"]) for x in old["suppressed_tokens"][:6]])
                print("  promoted  :", [(x["token"], x["mean_delta_prob"]) for x in old["promoted_tokens"][:6]])
            if report["control_audit"]:
                ca = report["control_audit"]
                print(
                    f"[control_audit] controls={ca['n_controls']} always_on_kl={ca['always_on_mean_kl']} "
                    f"routed_fire={ca['routed_fire_rate']} routed_kl={ca['routed_mean_kl']}"
                )
            for v in report["vocab_logit_lens"]:
                note = " (early layer -- see output_logit_diff)" if v.get("early_layer_caveat") else ""
                print(f"[vocab_logit_lens] {v['module']} (alpha={v['alpha']}, cols={v['cols']}){note}")
                print("  suppressed:", [x["token"] for x in v["suppressed_tokens"][:6]])
                print("  promoted  :", [x["token"] for x in v["promoted_tokens"][:6]])
        return

    prompt = _apply_cli_chat(tok, a.prompt, a.chat)
    enc = tok(prompt, return_tensors="pt").to(dev)
    gen_kw: dict = {"max_new_tokens": a.max_new_tokens, "do_sample": a.temperature > 0}
    if a.temperature > 0:
        gen_kw["temperature"] = a.temperature
    with patch.enable_routed(
        model,
        mode=a.mode,
        threshold=a.threshold,
        confidence=a.confidence,
        loop_router=True if a.loop_router else None,
    ) as routed:
        out_ids = model.generate(**enc, **gen_kw)
    new_ids = out_ids[0, enc["input_ids"].shape[1] :]
    text = tok.decode(new_ids, skip_special_tokens=True)
    st = routed.stats
    if a.as_json:
        print(json.dumps({"patch": patch.meta.get("name", Path(a.patch).name), "text": text, "stats": st}, indent=2))
    else:
        print(text)
        print(
            f"[moipatch] mode={st['mode']} fired={st['fired']}/{st['calls']} "
            f"last_scores={st['last_scores']} last_fired_at={st['last_fired_at']}",
            flush=True,
        )


if __name__ == "__main__":
    _main()
