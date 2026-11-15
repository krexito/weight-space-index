"""Weight-Space Index (WSI): Semantic and structural indexing of neural network weights for LLMs.

Built to work standalone or directly on top of `wsr` (Weight-Space Reading and Programming).
"""
from __future__ import annotations

from .compare import (
    align_semantic_circuits,
    decompile_edit_intent,
    diff_checkpoints,
    discover_connected_chains,
    transfer_edit,
    transfer_transduction_edit,
    transplant_compositional_and_chain,
    transplant_connected_chain,
)
from .indexer import build_index
from .query import WeightSpaceIndex, load_index

__all__ = [
    "WeightSpaceIndex",
    "align_semantic_circuits",
    "build_index",
    "decompile_edit_intent",
    "diff_checkpoints",
    "discover_connected_chains",
    "load_index",
    "transfer_edit",
    "transfer_transduction_edit",
    "transplant_compositional_and_chain",
    "transplant_connected_chain",
]
__version__ = "0.1.0"
