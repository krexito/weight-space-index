"""Weight-Space Index (WSI): Semantic and structural indexing of neural network weights for LLMs.

Built to work standalone or directly on top of `wsr` (Weight-Space Reading and Programming).
"""
from __future__ import annotations

from .capture import ActivationIndex, SelectiveMoI, TaskRouter, apply_selection, capture_index, find_mlps, trace_all
from .compare import (
    AdditiveMoI,
    DynamicCircuitPager,
    MixtureOfIndexes,
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
from .moipatch import Edit, MoIPatch, PatchHandle
from .query import WeightSpaceIndex, load_index
from .tracer import (
    CompactTrace,
    capture_compact,
    dequantize_block_int4,
    diff_traces,
    quantization_error,
    quantize_block_int4,
)

__all__ = [
    "ActivationIndex",
    "AdditiveMoI",
    "CompactTrace",
    "Edit",
    "MoIPatch",
    "PatchHandle",
    "SelectiveMoI",
    "TaskRouter",
    "apply_selection",
    "capture_compact",
    "capture_index",
    "dequantize_block_int4",
    "diff_traces",
    "find_mlps",
    "quantization_error",
    "quantize_block_int4",
    "trace_all",
    "DynamicCircuitPager",
    "MixtureOfIndexes",
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
