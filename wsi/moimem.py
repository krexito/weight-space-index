"""`.moimem`: a bank of key -> value memories written into one MLP ``down_proj`` as additive rank-1 edits.
=========================================================================================================

Each memory ``i`` is a pair ``(c_i, r_i)``: ``c_i [U]`` reads the input of ``down_proj`` (the MLP's hidden activation)
and ``r_i [D]`` is what gets added to the MLP output. With every memory on and no gate, the bank is exactly the
weight edit ``W' = W + sum_i r_i c_i^T`` (ROME / MEMIT form), so it can be baked into the weights or applied as a hook.

The hook form adds one thing a weight edit cannot do: a **hard gate**. Memory ``i`` contributes
``r_i * s_i`` with ``s_i = c_i . x`` only where its normalised match ``g_i = s_i / (c_i . k_i)`` is above ``thr``
(``k_i`` is the key the memory was written for, so ``g_i = 1`` exactly on it). Below the threshold the memory is silent,
which removes the small leakage a rank-1 edit leaves on unrelated tokens.

Python::

    bank = MemoryBank.load("people.moimem")
    h = bank.attach(model)          # checks the fingerprint of the down_proj it writes to
    h.on(gated=True)                # forward hook; h.off() removes it (weights are never touched)
    W_new = bank.baked_weight(W)    # the equivalent dense edit (ungated)
"""

from __future__ import annotations

import hashlib
import json
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch

FORMAT = "moimem/1"
_SAVE_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}
_SHA_CACHE: OrderedDict[tuple[int, int, tuple[int, ...], str, str], tuple[weakref.ReferenceType, int, str]] = OrderedDict()
_SHA_CACHE_MAX = 32


def _sha256(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_quick_sig(t: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    """Fast O(1)-alloc byte integrity signature catching even ``.data[...]`` mutations."""
    raw = t.detach().contiguous().reshape(-1).view(torch.uint8)
    n = int(raw.numel())
    ver = int(getattr(t, "_version", 0))
    if n == 0:
        return (ver, 0, 0, 0, 0, 0)
    even = int(raw[0::2].sum(dtype=torch.int64).item())
    odd = int(raw[1::2].sum(dtype=torch.int64).item()) if n > 1 else even
    return (
        ver,
        n,
        even,
        odd,
        int(raw[0].item()),
        int(raw[n // 2].item()) ^ (int(raw[-1].item()) << 8),
    )


def _cached_sha256(t: torch.Tensor, dt: torch.dtype | None = None) -> str:
    """Return SHA-256 hex digest with weakref + fast on-device byte-signature validation."""
    target_dt = dt or t.dtype
    key = (id(t), t.data_ptr(), tuple(t.shape), str(t.dtype), str(target_dt))
    sig = _tensor_quick_sig(t)
    cached = _SHA_CACHE.get(key)
    if cached is not None:
        ref, cached_sig, digest = cached
        if ref() is t and cached_sig == sig:
            _SHA_CACHE.move_to_end(key)
            return digest
    src = t.detach() if target_dt == t.dtype else t.detach().to(target_dt)
    digest = _sha256(src)
    _SHA_CACHE[key] = (weakref.ref(t), sig, digest)
    while len(_SHA_CACHE) > _SHA_CACHE_MAX:
        _SHA_CACHE.popitem(last=False)
    return digest


def find_layers(model) -> torch.nn.ModuleList:
    return next(m for n, m in model.named_modules()
                if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 2
                and "mtp" not in n and "visual" not in n)


def layer_down_proj(layer: torch.nn.Module) -> torch.nn.Module:
    """Resolve a decoder block's MLP down-projection across Llama/Qwen, LFM3, GPT-2, OPT."""
    for path in (("mlp", "down_proj"), ("feed_forward", "w2"), ("mlp", "c_proj"), ("mlp", "fc2")):
        cur = layer
        for attr in path:
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if cur is not None:
            return cur
    raise AttributeError(f"Could not find MLP down-projection on layer {type(layer).__name__}")


@dataclass
class MemoryBank:
    layer: int
    c: torch.Tensor  # [N, U] read vectors (rows of the edit)
    r: torch.Tensor  # [N, D] write vectors
    ck: torch.Tensor  # [N] c_i . k_i (gate normaliser; 1.0 for an independent ROME memory)
    thr: float = 0.5
    names: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # facts, method, base model, down_proj fingerprint, ...

    def __len__(self) -> int:
        return int(self.c.shape[0])

    def subset(self, idx) -> "MemoryBank":
        idx = torch.as_tensor(list(idx), dtype=torch.long)
        return MemoryBank(self.layer, self.c[idx], self.r[idx], self.ck[idx], self.thr,
                          [self.names[i] for i in idx.tolist()] if self.names else [], dict(self.meta))

    def baked_weight(self, W: torch.Tensor) -> torch.Tensor:
        """W [D, U] + sum_i r_i c_i^T, computed in float64, returned in W's dtype."""
        return (W.detach().double() + self.r.double().T.to(W.device) @ self.c.double().to(W.device)).to(W.dtype)

    def bytes(self) -> int:
        return (self.c.numel() + self.r.numel()) * 2  # as stored (bf16)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path, *, dtype: str = "fp32") -> Path:
        from safetensors.torch import save_file

        if dtype not in _SAVE_DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(_SAVE_DTYPES)}")
        dt = _SAVE_DTYPES[dtype]
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"format": FORMAT, "layer": self.layer, "thr": self.thr, "names": self.names, "meta": self.meta}
        save_file({"c": self.c.to(dt).contiguous(), "r": self.r.to(dt).contiguous(), "ck": self.ck.float().contiguous()},
                  str(p), metadata={"moimem": json.dumps(manifest)})
        return p

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            m = json.loads(f.metadata()["moimem"])
            if m.get("format") != FORMAT:
                raise ValueError(f"unsupported format {m.get('format')} (expected {FORMAT})")
            c, r, ck = f.get_tensor("c").clone(), f.get_tensor("r").clone(), f.get_tensor("ck").clone()
        return cls(m["layer"], c, r, ck, m["thr"], m.get("names", []), m.get("meta", {}))

    def attach(self, model) -> "MemoryHandle":
        return MemoryHandle(self, model)


class MemoryHandle:
    """Bank attached to a model. Starts off. `on(gated)` installs a hook on down_proj; `off()` removes it."""

    def __init__(self, bank: MemoryBank, model):
        self.bank, self.model = bank, model
        self.down = layer_down_proj(find_layers(model)[bank.layer])
        want = bank.meta.get("down_proj_sha256")
        if want:
            dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[bank.meta["down_proj_dtype"]]
            if _cached_sha256(self.down.weight, dt) != want:
                raise ValueError("down_proj weights differ from the bank's base model (fingerprint mismatch)")
        p = self.down.weight
        self.c = bank.c.to(p.device, torch.float32)
        self.r = bank.r.to(p.device, torch.float32)
        self.ck = bank.ck.to(p.device, torch.float32).clamp_min(1e-6)
        self._h = None
        self.gated = False
        self.last_fired = None  # [N] bool: which memories fired anywhere in the last forward (gated mode)

    def _hook(self, mod, inp, out):
        x = inp[0].float()
        s = x @ self.c.T  # [B, T, N]
        if self.gated:
            on = (s / self.ck) > self.bank.thr
            self.last_fired = on.flatten(0, -2).any(0).cpu()
            s = s * on
        return out + (s @ self.r).to(out.dtype)

    def on(self, gated: bool = False) -> "MemoryHandle":
        self.off()
        self.gated = gated
        self._h = self.down.register_forward_hook(self._hook)
        return self

    def off(self) -> None:
        if self._h is not None:
            self._h.remove()
            self._h = None
