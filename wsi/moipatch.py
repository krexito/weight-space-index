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

Python::

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
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

FORMAT = "moipatch/1"
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


def apply_edit(W: torch.Tensor, u: torch.Tensor, idx: torch.Tensor | None, alpha: float) -> torch.Tensor:
    """W' = W - alpha * u (m * (u^T W)), computed in float64 from the original weight, returned in W's dtype."""
    Wd = W.detach().cpu().double()
    uu = u.detach().cpu().double()
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
            tensors[f"edit.{i}.u"] = e.u.detach().to(torch.float64 if e.u.dtype == torch.float64 else torch.float32).cpu().contiguous()  # keep fp64 so the patched bf16 weights are bit-identical to the run
            if e.idx is not None:
                tensors[f"edit.{i}.idx"] = e.idx.detach().long().cpu().contiguous()
            edits.append({"module": e.module, "alpha": float(e.alpha), "has_idx": e.idx is not None,
                          "base_sha256": e.base_sha256, "base_dtype": e.base_dtype, "shape": list(e.shape)})
        router = None
        if self.router:
            tensors["router.dir"] = torch.as_tensor(self.router["dir"]).float().cpu().contiguous()
            router = {k: v for k, v in self.router.items() if k != "dir"}
        manifest = {"format": FORMAT, "edits": edits, "router": router, "gates": self.gates, "meta": self.meta}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(p), metadata={"moipatch": json.dumps(manifest)})
        return p

    @classmethod
    def load(cls, path: str | Path) -> "MoIPatch":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            manifest = json.loads(f.metadata()["moipatch"])
            if manifest.get("format") != FORMAT:
                raise ValueError(f"unsupported format {manifest.get('format')} (expected {FORMAT})")
            edits = []
            for i, e in enumerate(manifest["edits"]):
                edits.append(Edit(module=e["module"], u=f.get_tensor(f"edit.{i}.u"), alpha=e["alpha"],
                                  idx=f.get_tensor(f"edit.{i}.idx") if e["has_idx"] else None,
                                  base_sha256=e["base_sha256"], base_dtype=e["base_dtype"], shape=tuple(e["shape"])))
            router = manifest.get("router")
            if router is not None:
                router = dict(router, dir=f.get_tensor("router.dir"))
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
    def attach(self, model) -> "PatchHandle":
        return PatchHandle(self, model)

    def router_score(self, hidden_states) -> torch.Tensor:
        """hidden_states: the tuple from output_hidden_states=True (or the [B, T, D] tensor of the router layer).
        Returns h_last . dir for each sequence (left padding assumed, so the last position is the last token)."""
        if self.router is None:
            raise ValueError("this patch has no router")
        h = hidden_states[self.router["layer"]] if isinstance(hidden_states, (tuple, list)) else hidden_states
        h = h[:, -1] if h.dim() == 3 else h
        return h.float().cpu() @ self.router["dir"].float()

    def fires(self, hidden_states, confidence: float | torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate router decision, plus any answer_confidence gate if `confidence` is supplied."""
        ok = self.router_score(hidden_states) > self.router["thr"]
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
            lines.append(f"  edit {e.module} {tuple(e.shape)} alpha {e.alpha:g}, {cols}, base {e.base_dtype} "
                         f"sha256 {e.base_sha256[:12]}...")
        if self.router:
            lines.append(f"  router: hidden_states[{self.router['layer']}] last token . dir > {self.router['thr']:.4g}")
        for g in self.gates:
            lines.append(f"  gate: {g.get('type')} > {g.get('thr')}  ({g.get('note', '')})")
        if m.get("policy"):
            lines.append(f"  policy: {m['policy']}")
        if m.get("expected"):
            lines.append(f"  expected: {json.dumps(m['expected'])}")
        n_bytes = sum(e.u.numel() * e.u.element_size() + (e.idx.numel() * 8 if e.idx is not None else 0) for e in self.edits)
        lines.append(f"  payload: {n_bytes / 1024:.1f} KB of recipe")
        return "\n".join(x for x in lines if x)


class PatchHandle:
    """A patch attached to a loaded model. Starts OFF. `on()` patches the weights in place, `off()` restores the exact
    original bytes. The fingerprint is checked on attach, so a patch never touches a different model."""

    def __init__(self, patch: MoIPatch, model):
        self.patch, self.model, self.active = patch, model, False
        names = {}
        for name, mod in model.named_modules():
            if "mtp" in name or "visual" in name or not hasattr(mod, "weight"):
                continue
            try:
                names.setdefault(module_suffix(name), []).append(mod)
            except ValueError:
                pass
        self.params, self.orig, self.new = [], [], []
        for e in patch.edits:
            mods = names.get(e.module, [])
            if len(mods) != 1:
                raise KeyError(f"{e.module}: {len(mods)} matching modules in the model")
            p = mods[0].weight
            base = p.detach().to(_DTYPES[e.base_dtype])
            if e.base_sha256 and _sha256(base) != e.base_sha256:
                raise ValueError(f"{e.module}: weights differ from the patch's base model (fingerprint mismatch); "
                                 f"refusing to attach")
            self.params.append(p)
            self.orig.append(p.detach().cpu().clone())
            self.new.append(apply_edit(self.orig[-1], e.u, e.idx, e.alpha))

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


# ---------------------------------------------------------------------- CLI
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
    a = ap.parse_args(argv)
    patch = MoIPatch.load(a.patch)
    print(patch.describe())
    if a.cmd == "info":
        return
    from transformers import AutoModelForCausalLM

    dt = _DTYPES[patch.edits[0].base_dtype]
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    if dev == "cpu" and dt != torch.float32:
        dt = torch.float32  # CPU: fp32 holds every bf16 value exactly, so the fingerprint still matches
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
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


if __name__ == "__main__":
    _main()
