r"""Helper available to reader agents inside the sandbox (CPU only).

    from wsr.sandbox_lib import W, I
    W.names(r"layers\.15\.mlp")     # tensor names matching a regex
    W.get(name)                      # original tensor (torch, CPU)
    W.info(name)                     # shape, dtype and norm statistics
    I.overview()                     # Weight-Space Index macro map
    I.search_token("Paris")          # find layers, subspaces, and neurons for a concept/token
    I.explain_neuron(2, 7)           # explain what neuron 7 of layer 2 writes into residual stream
    I.explain_edit("/interventions/edit_01.safetensors")  # full semantic diagnosis of an edit
"""
from __future__ import annotations

import os
import re

import torch

from .checkpoint import Checkpoint

TARGET_DIR = os.environ.get("WSR_TARGET_DIR", "/weights/target")
INDEX_DIR = os.environ.get("WSR_INDEX_DIR", "")


class _Weights:
    def __init__(self, model_dir: str):
        self.ckpt = Checkpoint(model_dir)

    def names(self, pattern: str = "") -> list[str]:
        rx = re.compile(pattern) if pattern else None
        return [n for n in self.ckpt.names() if not rx or rx.search(n)]

    def get(self, name: str) -> torch.Tensor:
        return self.ckpt.get(name)

    def info(self, name: str) -> dict:
        t = self.get(name).float()
        return {"shape": list(t.shape), "dtype": self.ckpt.meta(name)[1], "fro": t.norm().item(),
                "mean": t.mean().item(), "std": t.std().item(), "absmax": t.abs().max().item()}


class _LazyIndex:
    """Lazy proxy to WeightSpaceIndex so importing `W` incurs zero overhead until `I` is called."""

    def __init__(self, model_dir: str, index_dir: str = ""):
        self._model_dir = model_dir
        self._index_dir = index_dir or None
        self._idx = None

    def _get(self):
        if self._idx is None:
            from wsi import WeightSpaceIndex
            self._idx = WeightSpaceIndex.from_model(self._model_dir, index_dir=self._index_dir)
        return self._idx

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


W = _Weights(TARGET_DIR)
I = _LazyIndex(TARGET_DIR, INDEX_DIR)
