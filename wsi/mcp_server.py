"""Model Context Protocol (MCP) Server for Weight-Space Index (WSI).

Exposes WSI tools over standard MCP JSON-RPC 2.0 stdio so any LLM agent (Claude Desktop,
Cursor, Antigravity, OpenCode, etc.) can navigate, inspect, diff, align, and surgically edit
model weights without training or GPU forward passes.

Run with:
    python -m wsi mcp
"""
from __future__ import annotations

import json
import sys
from typing import Any

from .moipatch import Edit, MoIPatch
from .query import WeightSpaceIndex
from .tracer import (
    CompactTrace,
    attribute_direction_writers,
    capture_causal_trace,
    diagnose_causal_traces,
    discover_compact_dirs,
    peel_causal_circuit,
    screen_patch_candidates,
    summarize_direction_traces,
    trace_direction_graph,
)

# Cache of open WeightSpaceIndex instances by (model_dir, index_dir)
_INDEX_CACHE: dict[tuple[str, str], WeightSpaceIndex] = {}


def _get_wsi(model_dir: str, index_dir: str | None = None) -> WeightSpaceIndex:
    key = (str(model_dir), str(index_dir or ""))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = WeightSpaceIndex.from_model(model_dir, index_dir=index_dir)
    return _INDEX_CACHE[key]


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "wsi_overview",
        "description": "Get the anatomical overview, layer structure, token roles, and top circuits of a model checkpoint.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "index_dir": {"type": "string", "description": "Optional path to a prebuilt WSI index directory"},
            },
            "required": ["model_dir"],
        },
    },
    {
        "name": "wsi_search_token",
        "description": "Find which layers, SVD subspaces, input-triggered neurons, and output-promoting neurons correspond to a token or concept.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "query": {"type": "string", "description": "Token substring (e.g. 'Paris') or integer token ID"},
                "top_layers": {"type": "integer", "default": 5},
                "top_units_per_layer": {"type": "integer", "default": 5},
            },
            "required": ["model_dir", "query"],
        },
    },
    {
        "name": "wsi_search_transduction",
        "description": "Find key-value memory neurons (layer, unit) triggered by `input_query` on the input side that promote `output_query` on the output side ([Input] -> [Output]).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "input_query": {"type": "string", "description": "Input trigger token substring or ID (e.g. 'Eiffel')"},
                "output_query": {"type": "string", "description": "Output promoted token substring or ID (e.g. 'Paris')"},
                "top_k": {"type": "integer", "default": 8},
            },
            "required": ["model_dir", "input_query", "output_query"],
        },
    },
    {
        "name": "wsi_explain_neuron",
        "description": "Explain what input tokens trigger a specific neuron (layer, unit) and what output tokens it writes into the residual stream.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "layer": {"type": "integer", "description": "Layer index"},
                "unit": {"type": "integer", "description": "Neuron/column index"},
                "module": {"type": "string", "default": "mlp_out", "enum": ["mlp_out", "attn_out"]},
            },
            "required": ["model_dir", "layer", "unit"],
        },
    },
    {
        "name": "wsi_trace_circuit",
        "description": "Trace the upstream writer neurons (in earlier layers) and downstream reader neurons (in later layers) connected to `(layer, unit)`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "layer": {"type": "integer", "description": "Layer index"},
                "unit": {"type": "integer", "description": "Neuron index"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["model_dir", "layer", "unit"],
        },
    },
    {
        "name": "wsi_trace_internal_state",
        "description": "Trace a Deep Internal State Circuit (such as reasoning/repetition loops, induction heads, and cross-layer MLP echo channels) across all Attention Heads (`o_proj`) and MLP blocks (`down_proj`) directly from weight matrices.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "layer": {"type": "integer", "description": "Source layer index"},
                "head": {"type": "integer", "description": "Optional source Attention Head index in `layer`"},
                "neurons": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional list of source MLP neuron indices in `layer`",
                },
                "top_k_neurons": {"type": "integer", "default": 64},
            },
            "required": ["model_dir", "layer"],
        },
    },
    {
        "name": "wsi_explain_edit",
        "description": "Diagnose a `.safetensors` or `.json` weight edit file and explain its structural and semantic effect on vocabulary tokens.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "edit_path": {"type": "string", "description": "Path to edit.safetensors or edit.json"},
            },
            "required": ["model_dir", "edit_path"],
        },
    },
    {
        "name": "wsi_rewire_transduction",
        "description": "Surgically rewire the top K neurons that transduce `input_query -> old_output` so they write `new_output` instead, without training.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "input_query": {"type": "string", "description": "Input trigger token"},
                "old_output": {"type": "string", "description": "Original output token to suppress"},
                "new_output": {"type": "string", "description": "New output token to promote"},
                "save_path": {"type": "string", "description": "Optional path to save the .safetensors edit"},
                "top_k_neurons": {"type": "integer", "default": 8},
                "strength": {"type": "number", "default": 2.5},
            },
            "required": ["model_dir", "input_query", "old_output", "new_output"],
        },
    },
    {
        "name": "wsi_diff_models",
        "description": "Semantic git-diff between two checkpoints of the same architecture (Base vs Fine-Tune/Edited).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "base_model_dir": {"type": "string", "description": "Path to base checkpoint"},
                "target_model_dir": {"type": "string", "description": "Path to fine-tuned or edited checkpoint"},
            },
            "required": ["base_model_dir", "target_model_dir"],
        },
    },
    {
        "name": "wsi_align_circuits",
        "description": "Match functionally equivalent semantic circuits across two different models/architectures using the vocabulary as a universal anchor.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_a_dir": {"type": "string", "description": "Path to first model directory"},
                "model_b_dir": {"type": "string", "description": "Path to second model directory"},
                "transduction_pairs": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                    "description": "List of [input_token, output_token] pairs to align",
                },
                "concepts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["model_a_dir", "model_b_dir"],
        },
    },
    {
        "name": "wsi_transfer_edit",
        "description": "Decompile a raw .safetensors edit made on `src_model_dir` and automatically synthesize the equivalent semantic edit on `dst_model_dir` (even across different architectures!).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "src_model_dir": {"type": "string", "description": "Path to source model A"},
                "dst_model_dir": {"type": "string", "description": "Path to destination model B"},
                "edit_path": {"type": "string", "description": "Path to source edit file (.safetensors or .json)"},
                "save_path": {"type": "string", "description": "Path to save the synthesized edit for model B"},
                "top_k_neurons": {"type": "integer", "default": 8},
                "strength": {"type": "number", "default": 2.5},
            },
            "required": ["src_model_dir", "dst_model_dir", "edit_path"],
        },
    },
    {
        "name": "wsi_diff_compact",
        "description": "Compare two saved CompactTrace directories with WSI weight grounding (W_O-projected head shifts, MLP neurons, cache deltas, and vocabulary readout).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "trace_a_dir": {"type": "string", "description": "Path to baseline CompactTrace directory"},
                "trace_b_dir": {"type": "string", "description": "Path to target CompactTrace directory"},
                "pos_a": {"type": "integer", "default": -1},
                "pos_b": {"type": "integer", "default": -1},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["model_dir", "trace_a_dir", "trace_b_dir"],
        },
    },
    {
        "name": "wsi_discover_compact_circuit",
        "description": "Discover a multi-layer loop or behavioral intervention circuit from saved CompactTrace directories + static WSI SVD resonance, and optionally save a .moipatch file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the safetensors checkpoint directory"},
                "trace_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Saved CompactTrace directories (all traces when auto-detecting, or healthy traces when loop_trace_dirs is set)",
                },
                "loop_trace_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional saved CompactTrace directories for looping/target runs",
                },
                "alpha": {"type": "number", "default": 6.0},
                "top_k_neurons": {"type": "integer", "default": 64},
                "save_patch_path": {"type": "string", "description": "Optional output path for the synthesized .moipatch file"},
                "save_candidate_grid_dir": {
                    "type": "string",
                    "description": "Optional output directory to save the deduplicated union candidate grid (Top Passive + Top Causal across candidate_alphas) as `.moipatch` files ready for `wsi_screen_candidates`",
                },
                "candidate_alphas": {
                    "type": "array",
                    "items": {"type": "number"},
                    "default": [4.0, 8.0, 12.0],
                    "description": "Alpha values to expand each passive and causal template into inside the candidate grid",
                },
                "patch_variant": {
                    "type": "string",
                    "default": "unified_top2",
                    "enum": [
                        "unified_top2",
                        "blind_top2",
                        "attn_top2",
                        "wsi_top3",
                        "wsi_fa3",
                        "wsi_pair",
                        "mlp_top3",
                        "mlp_fa3",
                        "causal_top1",
                        "causal_top1_all",
                        "causal_top2",
                        "causal_attn_top1",
                    ],
                },
            },
            "required": ["model_dir", "trace_dirs"],
        },
    },
    {
        "name": "wsi_build_moipatch",
        "description": "Build, fingerprint against the base safetensors checkpoint, and save a portable `.moipatch` file from explicit layer edits (direction vector, alpha, optional neuron indices) or from a saved `blind_directions.npz` file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the base safetensors checkpoint directory"},
                "save_path": {"type": "string", "description": "Output file path for the `.moipatch` artifact"},
                "edits": {
                    "type": "array",
                    "description": "List of rank-1 column edits to bundle into the patch",
                    "items": {
                        "type": "object",
                        "properties": {
                            "layer": {"type": "integer", "description": "Layer index"},
                            "module": {"type": "string", "default": "mlp.down_proj", "description": "Submodule suffix within the layer (e.g. 'mlp.down_proj')"},
                            "alpha": {"type": "number", "description": "Edit strength"},
                            "direction": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Optional residual-stream direction vector `u` (length = hidden_size)",
                            },
                            "direction_npz": {"type": "string", "description": "Optional path to `.npz` file containing direction vectors (e.g. `blind_directions.npz`)"},
                            "direction_key": {"type": "string", "description": "Key inside `direction_npz` (defaults to `u{layer}`)"},
                            "neurons": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Optional subset of column/neuron indices to edit (omit or null for all columns)",
                            },
                        },
                        "required": ["layer", "alpha"],
                    },
                },
                "router": {"type": "object", "description": "Optional MoI router configuration dict"},
                "meta": {"type": "object", "description": "Optional metadata dict stored in the patch"},
            },
            "required": ["model_dir", "save_path", "edits"],
        },
    },
    {
        "name": "wsi_screen_candidates",
        "description": "Rank candidate `.moipatch` files in 1 forward pass (zero autoregressive generation) using Pareto scoring (`target_suppression` vs healthy `delta_nll` guard).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the HuggingFace model checkpoint directory"},
                "patch_paths": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "object", "additionalProperties": {"type": "string"}},
                    ],
                    "description": "List of `.moipatch` paths or mapping of `{candidate_name: patch_path}`",
                },
                "healthy_seqs": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}},
                    "description": "Token ID sequences representing healthy behavior (optional if `healthy_texts` is provided)",
                },
                "target_seqs": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}},
                    "description": "Token ID sequences exhibiting the target failure mode (optional if `target_texts` is provided)",
                },
                "healthy_texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Raw text strings representing healthy behavior (tokenized automatically using `model_dir`)",
                },
                "target_texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Raw text strings exhibiting the target failure mode (tokenized automatically using `model_dir`)",
                },
                "target_spans": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "description": "Optional `[start, end]` token spans inside each target sequence to measure suppression on",
                },
                "max_nll_delta": {"type": "number", "default": 0.05},
                "rank_by": {"type": "string", "default": "pareto", "enum": ["pareto", "pareto_kl", "suppression"]},
                "top_k": {"type": "integer", "default": 10},
                "prefilter_top_k": {
                    "type": "integer",
                    "description": "Optional gradient prefilter count: when set (< number of candidates), narrows down to the top-K candidates before running forward screening (uses offline `trace_dirs` if provided, else 1 backward per sequence)",
                },
                "trace_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional paths to saved causal CompactTrace directories (`capture_grad=True, grad_mode='compressed'`) so Stage-1 `prefilter_top_k` runs 100% offline with zero backward passes",
                },
                "dtype": {"type": "string", "default": "auto", "enum": ["auto", "bfloat16", "float16", "float32"]},
                "device_map": {"type": "string", "default": "auto"},
            },
            "required": ["model_dir", "patch_paths"],
        },
    },
    {
        "name": "wsi_capture_causal_traces",
        "description": "Capture forward+backward causal CompactTraces (`capture_causal_trace`) on target sequences, save them to `output_dir`, and return `trace_dirs` plus an immediate causal diagnosis report (MLP + Attention). Supports optional `target_dir` (internal direction), `protect_dir` (protected direction), and `contrast_token_ids` (`logit_margin` without softmax saturation).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the HuggingFace model checkpoint directory"},
                "output_dir": {"type": "string", "description": "Directory where `trace_0`, `trace_1`, ... will be saved"},
                "sequences": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}},
                    "description": "Token ID sequences exhibiting the target behavior (optional if `texts` is provided)",
                },
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Raw text strings to tokenize and capture (optional if `sequences` is provided)",
                },
                "target_spans": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "description": "Optional per-sequence `[start, end]` target failure spans (defaults to second half `[T//2, T]`, or last token in `target_dir` mode)",
                },
                "healthy_spans": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "description": "Optional per-sequence `[start, end]` healthy prefix spans (defaults to `[0, target_start]`)",
                },
                "grad_mode": {
                    "type": "string",
                    "default": "compressed",
                    "enum": ["compressed", "summary"],
                    "description": "'compressed' saves both grad_summary and int4 per-token gradients (for offline screening); 'summary' saves only the ~5MB causal summary",
                },
                "grad_objective": {
                    "type": "string",
                    "default": "auto",
                    "enum": ["auto", "token_nll", "logit_margin", "direction"],
                    "description": "Backward objective ('auto' selects 'direction' when `target_dir*` is given, 'logit_margin' when token IDs are given, else 'token_nll')",
                },
                "target_dir": {"type": "array", "items": {"type": "number"}, "description": "Optional internal direction vector `[D]` to trace causally"},
                "target_dir_npz": {"type": "string", "description": "Optional `.npz` path storing `target_dir`"},
                "target_dir_key": {"type": "string", "description": "Array key inside `target_dir_npz`"},
                "target_patch_path": {"type": "string", "description": "Optional `.moipatch` file whose router vector (`dir`/`veto_dir`) supplies `target_dir` and `target_layer`"},
                "target_patch_key": {"type": "string", "default": "dir", "description": "Router key inside `target_patch_path` (e.g. 'dir' or 'veto_dir')"},
                "target_layer": {"type": "integer", "description": "Layer index where `target_dir` is read"},
                "target_layer_convention": {"type": "string", "default": "block", "enum": ["block", "hidden_states"]},
                "protect_dir": {"type": "array", "items": {"type": "number"}, "description": "Optional protected direction vector `[D]` to preserve during causal diagnosis"},
                "protect_dir_npz": {"type": "string", "description": "Optional `.npz` path storing `protect_dir`"},
                "protect_dir_key": {"type": "string", "description": "Array key inside `protect_dir_npz`"},
                "protect_patch_path": {"type": "string", "description": "Optional `.moipatch` file supplying `protect_dir` (defaults to router `veto_dir`)"},
                "protect_patch_key": {"type": "string", "default": "veto_dir"},
                "protect_layer": {"type": "integer", "description": "Optional layer index for `protect_dir` (defaults to `target_layer`)"},
                "protect_layer_convention": {"type": "string", "enum": ["block", "hidden_states"]},
                "target_token_ids": {"type": "array", "items": {"type": "integer"}, "description": "Optional target token IDs for `logit_margin` objective"},
                "contrast_token_ids": {"type": "array", "items": {"type": "integer"}, "description": "Optional contrast token IDs for `logit_margin` objective"},
                "healthy_cost": {"type": "string", "default": "linear", "enum": ["linear", "fisher"]},
                "top_k_mlp": {"type": "integer", "default": 32},
                "dtype": {"type": "string", "default": "auto", "enum": ["auto", "bfloat16", "float16", "float32"]},
                "device_map": {"type": "string", "default": "auto"},
            },
            "required": ["model_dir", "output_dir"],
        },
    },
    {
        "name": "wsi_attribute_direction_writers",
        "description": "Zero-backward Level-1 linear attribution showing which embeddings, attention heads, MLP neurons, and conv/linear-attn modules directly write into an internal direction `target_dir` (with exact frozen-scale final-norm linearization and optional per-block residual gap checks).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the HuggingFace model checkpoint directory"},
                "sequences": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "texts": {"type": "array", "items": {"type": "string"}},
                "contrast_sequences": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "contrast_texts": {"type": "array", "items": {"type": "string"}},
                "target_dir": {"type": "array", "items": {"type": "number"}},
                "target_dir_npz": {"type": "string"},
                "target_dir_key": {"type": "string"},
                "target_patch_path": {"type": "string"},
                "target_patch_key": {"type": "string", "default": "dir"},
                "target_layer": {"type": "integer", "description": "Layer index where `target_dir` lives (auto-inferred if `target_patch_path` has a router)"},
                "layer_convention": {"type": "string", "enum": ["block", "hidden_states"]},
                "pos": {
                    "description": "Integer token index (default -1) or `[start, end]` span",
                    "oneOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}],
                },
                "top_k_neurons": {"type": "integer", "default": 16},
                "top_k_heads": {"type": "integer", "default": 16},
                "normalize_dir": {"type": "boolean", "default": False},
                "linearize_final_norm": {"type": "boolean", "default": True},
                "report_block_gaps": {"type": "boolean", "default": False},
                "dtype": {"type": "string", "default": "auto", "enum": ["auto", "bfloat16", "float16", "float32"]},
                "device_map": {"type": "string", "default": "auto"},
            },
            "required": ["model_dir"],
        },
    },
    {
        "name": "wsi_trace_direction_graph",
        "description": "Trace a multi-hop directed circuit graph (upstream MLP neurons, attention heads, conv modules -> root `target_dir`) with exact per-hop direct and causal edge weights and ranked root-terminated `top_paths`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the HuggingFace model checkpoint directory"},
                "sequences": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "texts": {"type": "array", "items": {"type": "string"}},
                "target_dir": {"type": "array", "items": {"type": "number"}},
                "target_dir_npz": {"type": "string"},
                "target_dir_key": {"type": "string"},
                "target_patch_path": {"type": "string"},
                "target_patch_key": {"type": "string", "default": "dir"},
                "target_layer": {"type": "integer"},
                "layer_convention": {"type": "string", "enum": ["block", "hidden_states"]},
                "pos": {
                    "oneOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}],
                },
                "depth": {"type": "integer", "default": 2},
                "top_k_nodes": {"type": "integer", "default": 3},
                "min_layer": {"type": "integer", "default": 0},
                "min_edge_weight": {"type": "number", "default": 1e-6},
                "normalize_dir": {"type": "boolean", "default": False},
                "dtype": {"type": "string", "default": "auto", "enum": ["auto", "bfloat16", "float16", "float32"]},
                "device_map": {"type": "string", "default": "auto"},
            },
            "required": ["model_dir"],
        },
    },
    {
        "name": "wsi_peel_causal_circuit",
        "description": "Iteratively peel top causal sites over `n_rounds` rounds (attaching each round's edit in memory and re-tracing residual behavior) to uncover backup ('Hydra') circuits. Restores exact base weights on exit and optionally saves the cumulative `.moipatch`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string"},
                "sequences": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "texts": {"type": "array", "items": {"type": "string"}},
                "n_rounds": {"type": "integer", "default": 2},
                "alpha": {"type": "number", "default": 4.0},
                "sites_per_round": {"type": "integer", "default": 1},
                "top_k_sites": {"type": "integer", "default": 6},
                "top_n_neurons": {"type": "integer"},
                "allow_repeat_modules": {"type": "boolean", "default": False},
                "target_dir": {"type": "array", "items": {"type": "number"}},
                "target_dir_npz": {"type": "string"},
                "target_dir_key": {"type": "string"},
                "target_patch_path": {"type": "string"},
                "target_patch_key": {"type": "string", "default": "dir"},
                "target_layer": {"type": "integer"},
                "target_layer_convention": {"type": "string", "enum": ["block", "hidden_states"]},
                "protect_dir": {"type": "array", "items": {"type": "number"}},
                "protect_dir_npz": {"type": "string"},
                "protect_dir_key": {"type": "string"},
                "protect_patch_path": {"type": "string"},
                "protect_patch_key": {"type": "string", "default": "veto_dir"},
                "protect_layer": {"type": "integer"},
                "protect_layer_convention": {"type": "string", "enum": ["block", "hidden_states"]},
                "target_spans": {"type": "array"},
                "healthy_spans": {"type": "array"},
                "grad_objective": {"type": "string", "default": "auto", "enum": ["auto", "token_nll", "logit_margin", "direction"]},
                "target_token_ids": {"type": "array", "items": {"type": "integer"}},
                "contrast_token_ids": {"type": "array", "items": {"type": "integer"}},
                "healthy_cost": {"type": "string", "default": "linear", "enum": ["linear", "fisher"]},
                "center": {"type": "string", "default": "none", "enum": ["none", "healthy"]},
                "save_patch_path": {"type": "string", "description": "Optional output `.moipatch` path for the cumulative peeled patch"},
                "dtype": {"type": "string", "default": "auto", "enum": ["auto", "bfloat16", "float16", "float32"]},
                "device_map": {"type": "string", "default": "auto"},
            },
            "required": ["model_dir"],
        },
    },
    {
        "name": "wsi_bake_moipatch",
        "description": "Stream-bake one or more `.moipatch` files directly into a standalone HuggingFace safetensors checkpoint directory on disk (without loading the full model into RAM/VRAM), verifying base SHA-256 fingerprints and rejecting routed patches unless `allow_routed=true`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to base HuggingFace checkpoint directory with safetensors shards"},
                "out_dir": {"type": "string", "description": "Destination directory for the standalone baked model"},
                "patch_paths": {"type": "array", "items": {"type": "string"}, "description": "Ordered list of `.moipatch` paths to bake sequentially"},
                "patch_path": {"type": "string", "description": "Convenience alias for baking a single `.moipatch` path"},
                "verify_fingerprint": {"type": "boolean", "default": True},
                "allow_routed": {"type": "boolean", "default": False, "description": "Must be true to bake a patch containing a runtime router/safety veto statically as `always_on`"},
                "copy_untouched": {"type": "string", "default": "auto", "enum": ["auto", "copy"]},
            },
            "required": ["model_dir", "out_dir"],
        },
    },
    {
        "name": "wsi_audit_moipatch",
        "description": "Audit a `.moipatch` against a HuggingFace model checkpoint for SHA-256 fingerprint match, byte-exact restore, weight footprint, vocab logit lens, and end-to-end `output_logit_diff` / routed `control_audit`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the base HuggingFace checkpoint directory"},
                "patch_path": {"type": "string", "description": "Path to the `.moipatch` file to audit"},
                "probe_prompts": {"type": "array", "items": {"type": "string"}, "description": "Optional trigger probe prompts"},
                "probe_ids": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "control_prompts": {"type": "array", "items": {"type": "string"}, "description": "Optional benign control prompts"},
                "control_ids": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "chat": {"type": "boolean", "default": False, "description": "Format string prompts with `tokenizer.apply_chat_template`"},
                "top_k": {"type": "integer", "default": 10},
                "verify_fingerprint": {"type": "boolean", "default": True},
                "cpu": {"type": "boolean", "default": False},
                "dtype": {"type": "string", "default": "auto", "enum": ["auto", "bfloat16", "float16", "float32"]},
                "device_map": {"type": "string", "default": "auto"},
            },
            "required": ["model_dir", "patch_path"],
        },
    },
    {
        "name": "wsi_calibrate_safety_router",
        "description": "Calibrate a framing-invariant 2-vector safety veto router (`abliterated-sfw` style) on top of an existing surgical `.moipatch` using Group K-Fold OOF CV and paired `Sigma_frame` invariance, optionally saving the calibrated `.moipatch`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "Path to the base HuggingFace checkpoint directory"},
                "patch_path": {"type": "string", "description": "Path to the base `.moipatch` providing surgical edits"},
                "toxic_prompts": {"type": "array", "items": {"type": "string"}, "description": "Refused harmful/toxic calibration prompts"},
                "benign_refused_prompts": {"type": "array", "items": {"type": "string"}, "description": "Benign over-refused calibration prompts"},
                "control_prompts": {"type": "array", "items": {"type": "string"}, "description": "Normal everyday control prompts (must not fire)"},
                "benign_nonrefused_prompts": {"type": "array", "items": {"type": "string"}, "description": "Optional benign prompts already answered by the base model"},
                "chat": {"type": "boolean", "default": True},
                "max_toxic_fpr": {"type": "number", "default": 0.05},
                "n_folds": {"type": "integer", "default": 4},
                "out_patch_path": {"type": "string", "description": "Optional output path to save the calibrated `.moipatch`"},
                "cpu": {"type": "boolean", "default": False},
                "dtype": {"type": "string", "default": "auto", "enum": ["auto", "bfloat16", "float16", "float32"]},
                "device_map": {"type": "string", "default": "auto"},
            },
            "required": ["model_dir", "patch_path", "toxic_prompts", "benign_refused_prompts", "control_prompts"],
        },
    },
    {
        "name": "wsi_audit_lora",
        "description": "Translate a LoRA adapter (`adapter_model.safetensors`/`.bin`) into ranked vocabulary tokens (`sum sigma^2 * z^4`) across all matrices without running forward passes or loading transformer block shards from the base model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "base_model_dir": {"type": "string", "description": "Path to base HuggingFace checkpoint directory (loads only lm_head/embeddings + tokenizer)"},
                "adapter_path": {"type": "string", "description": "Path to LoRA adapter directory or adapter_model.safetensors/.bin file"},
                "top_k": {"type": "integer", "default": 20},
                "top_sv": {"type": "integer", "default": 4},
                "apply_final_norm": {"type": "boolean", "default": True},
                "lora_scale": {"type": "number", "description": "Optional explicit override for LoRA alpha/r scaling"},
                "device": {"type": "string", "default": "cpu", "enum": ["cpu", "cuda", "auto"]},
            },
            "required": ["base_model_dir", "adapter_path"],
        },
    },
]


def _load_model_for_screen(model_dir: str, dtype_name: str = "auto", device_map: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM

    if dtype_name == "auto":
        # Preserve the checkpoint's native safetensors dtype so MoIPatch SHA-256 fingerprints match.
        dtype: Any = "auto"
    else:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    base_kw: dict[str, Any] = {}
    if device_map and torch.cuda.is_available():
        base_kw["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, **base_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, **base_kw)
    model.eval()
    return model


def _load_tokenizer_for_screen(model_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def _build_moipatch_tool(args: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path

    import numpy as np
    import torch

    npz_cache: dict[str, Any] = {}
    edits: list[Edit] = []
    for item in args["edits"]:
        layer = int(item["layer"])
        submod = str(item.get("module", "mlp.down_proj"))
        mod_name = submod if submod.startswith("layers.") else f"layers.{layer}.{submod}"
        if item.get("direction") is not None:
            u = torch.tensor(item["direction"], dtype=torch.float32)
        elif item.get("direction_npz"):
            npz_path = str(item["direction_npz"])
            if npz_path not in npz_cache:
                npz_cache[npz_path] = np.load(npz_path)
            key = str(item.get("direction_key") or f"u{layer}")
            if key not in npz_cache[npz_path]:
                raise KeyError(f"Direction key `{key}` not found in {npz_path}")
            u = torch.from_numpy(np.asarray(npz_cache[npz_path][key], dtype=np.float32))
        else:
            raise ValueError(f"Edit for layer {layer} must specify either `direction` or `direction_npz`")
        neurons = item.get("neurons")
        idx = torch.tensor([int(x) for x in neurons], dtype=torch.long) if neurons is not None else None
        edits.append(Edit(module=mod_name, u=u, alpha=float(item["alpha"]), idx=idx))

    patch = MoIPatch(
        edits=edits,
        router=args.get("router"),
        meta=dict(args.get("meta") or {}),
    )
    patch.fingerprint_from_checkpoint(args["model_dir"])
    save_path = Path(args["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(save_path)
    return {
        "save_path": str(save_path),
        "n_edits": len(patch.edits),
        "modules": [e.module for e in patch.edits],
        "summary": patch.describe(),
    }


def _screen_candidates_tool(args: dict[str, Any]) -> dict[str, Any]:
    import gc
    from pathlib import Path

    import torch

    raw_patches = args["patch_paths"]
    if isinstance(raw_patches, dict):
        candidates = {str(k): MoIPatch.load(v) for k, v in raw_patches.items()}
    else:
        candidates = {Path(p).stem: MoIPatch.load(p) for p in raw_patches}
    if not candidates:
        raise ValueError("`patch_paths` must contain at least one `.moipatch` path")

    healthy_seqs = args.get("healthy_seqs")
    target_seqs = args.get("target_seqs")
    if healthy_seqs is None or target_seqs is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args["model_dir"])
        if healthy_seqs is None:
            if not args.get("healthy_texts"):
                raise ValueError("Provide either `healthy_seqs` or `healthy_texts`")
            healthy_seqs = [tok.encode(t, add_special_tokens=False) for t in args["healthy_texts"]]
        if target_seqs is None:
            if not args.get("target_texts"):
                raise ValueError("Provide either `target_seqs` or `target_texts`")
            target_seqs = [tok.encode(t, add_special_tokens=False) for t in args["target_texts"]]

    spans = [tuple(int(x) for x in sp) for sp in args["target_spans"]] if args.get("target_spans") else None
    max_nll_delta = float(args.get("max_nll_delta", 0.05))
    rank_by = str(args.get("rank_by", "pareto"))
    top_k = int(args.get("top_k", 10))
    prefilter_top_k = int(args["prefilter_top_k"]) if args.get("prefilter_top_k") is not None else None
    trace_dirs = [str(p) for p in args["trace_dirs"]] if args.get("trace_dirs") else None

    model = _load_model_for_screen(
        args["model_dir"],
        dtype_name=str(args.get("dtype", "auto")),
        device_map=str(args.get("device_map", "auto")),
    )
    try:
        rows = screen_patch_candidates(
            model,
            candidates,
            healthy_seqs=healthy_seqs,
            target_seqs=target_seqs,
            target_spans=spans,
            max_nll_delta=max_nll_delta,
            rank_by=rank_by,
            prefilter_top_k=prefilter_top_k,
            traces=trace_dirs,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "total_screened": len(rows),
        "rank_by": rank_by,
        "max_nll_delta": max_nll_delta,
        "ranking": rows[:top_k],
    }


def _resolve_sequences_arg(
    args: dict[str, Any],
    model_dir: str,
    *,
    seq_key: str = "sequences",
    text_key: str = "texts",
    required: bool = True,
) -> list[list[int]] | None:
    raw_seqs = args.get(seq_key)
    if raw_seqs is not None:
        seqs = [[int(x) for x in s] for s in raw_seqs]
        if required and not seqs:
            raise ValueError(f"At least one sequence is required in `{seq_key}`")
        return seqs
    raw_texts = args.get(text_key)
    if raw_texts:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_dir)
        seqs = [tok.encode(t, add_special_tokens=False) for t in raw_texts]
        if required and not seqs:
            raise ValueError(f"At least one text is required in `{text_key}`")
        return seqs
    if required:
        raise ValueError(f"Provide either `{seq_key}` or `{text_key}`")
    return None


def _load_direction_arg(
    args: dict[str, Any],
    prefix: str = "target",
    *,
    layer_key: str | None = None,
    conv_key: str | None = None,
) -> tuple[Any, int | None, str | None]:
    import numpy as np
    import torch

    l_key = layer_key or f"{prefix}_layer"
    c_key = conv_key or f"{prefix}_layer_convention"
    raw_layer = args.get(l_key)
    eff_layer = int(raw_layer) if raw_layer is not None else None
    raw_conv = args.get(c_key) or (args.get("target_layer_convention") if prefix == "target" and c_key != "target_layer_convention" else None)
    eff_conv = str(raw_conv) if raw_conv is not None else None

    if args.get(f"{prefix}_dir") is not None:
        vec = torch.tensor(args[f"{prefix}_dir"], dtype=torch.float32)
        return vec, eff_layer, eff_conv
    if args.get(f"{prefix}_dir_npz"):
        npz_path = str(args[f"{prefix}_dir_npz"])
        npz = np.load(npz_path)
        key = args.get(f"{prefix}_dir_key")
        if not key:
            if len(npz.files) == 1:
                key = npz.files[0]
            else:
                raise KeyError(f"Specify `{prefix}_dir_key` for `{npz_path}` (keys: {npz.files})")
        if key not in npz:
            raise KeyError(f"Direction key `{key}` not found in {npz_path}")
        vec = torch.from_numpy(np.asarray(npz[key], dtype=np.float32))
        return vec, eff_layer, eff_conv
    if args.get(f"{prefix}_patch_path"):
        patch = MoIPatch.load(str(args[f"{prefix}_patch_path"]))
        pkey = str(args.get(f"{prefix}_patch_key") or ("veto_dir" if prefix == "protect" else "dir"))
        if patch.router is not None and pkey in patch.router and patch.router[pkey] is not None:
            vec = torch.as_tensor(patch.router[pkey], dtype=torch.float32)
            if eff_layer is None and patch.router.get("layer") is not None:
                eff_layer = int(patch.router["layer"])
            if eff_conv is None:
                eff_conv = "hidden_states"
            return vec, eff_layer, eff_conv
        if pkey == "dir" and patch.router is None and patch.edits:
            vec = patch.edits[0].u.detach().float()
            return vec, eff_layer, eff_conv
        raise KeyError(f"Direction key `{pkey}` not found in patch {args[f'{prefix}_patch_path']}")
    return None, eff_layer, eff_conv


def _parse_pos_arg(raw_pos: Any) -> int | tuple[int, int] | None:
    if raw_pos is None:
        return -1
    if isinstance(raw_pos, (list, tuple)) and len(raw_pos) == 2:
        return (int(raw_pos[0]), int(raw_pos[1]))
    return int(raw_pos)


def _capture_causal_traces_tool(args: dict[str, Any]) -> dict[str, Any]:
    import gc
    from pathlib import Path

    import torch

    seqs = _resolve_sequences_arg(args, args["model_dir"], required=True)
    assert seqs is not None

    t_dir, t_layer, t_conv = _load_direction_arg(args, "target")
    p_dir, p_layer, p_conv = _load_direction_arg(args, "protect")
    eff_t_conv = t_conv or "block"
    grad_objective = str(args.get("grad_objective", "auto"))
    t_tok_ids = [int(x) for x in args["target_token_ids"]] if args.get("target_token_ids") is not None else None
    c_tok_ids = [int(x) for x in args["contrast_token_ids"]] if args.get("contrast_token_ids") is not None else None
    healthy_cost = str(args.get("healthy_cost", "linear"))

    raw_tspans = args.get("target_spans")
    raw_hspans = args.get("healthy_spans")
    grad_mode = str(args.get("grad_mode", "compressed"))
    top_k_mlp = int(args.get("top_k_mlp", 32))
    out_root = Path(args["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    model = _load_model_for_screen(
        args["model_dir"],
        dtype_name=str(args.get("dtype", "auto")),
        device_map=str(args.get("device_map", "auto")),
    )
    traces: list[CompactTrace] = []
    trace_dirs: list[str] = []
    try:
        for i, ids in enumerate(seqs):
            T = len(ids)
            t_span: int | tuple[int, int] | None
            h_span: tuple[int, int] | None
            if raw_tspans and i < len(raw_tspans) and raw_tspans[i] is not None:
                t_span = (int(raw_tspans[i][0]), int(raw_tspans[i][1]))
            elif t_dir is not None or t_tok_ids is not None:
                t_span = None
            else:
                t_span = (max(1, T // 2), T)
            if raw_hspans and i < len(raw_hspans) and raw_hspans[i] is not None:
                h_span = (int(raw_hspans[i][0]), int(raw_hspans[i][1]))
            elif p_dir is not None or t_dir is not None or t_tok_ids is not None:
                h_span = None
            else:
                base_t = t_span[0] if isinstance(t_span, tuple) else max(1, T - 1)
                h_span = (0, max(1, base_t))
            ct = capture_causal_trace(
                model,
                ids,
                target_span=t_span,
                healthy_span=h_span,
                grad_mode=grad_mode,
                target_dir=t_dir,
                target_layer=t_layer,
                target_layer_convention=eff_t_conv,
                protect_dir=p_dir,
                protect_layer=p_layer,
                protect_layer_convention=p_conv,
                grad_objective=grad_objective,
                target_token_ids=t_tok_ids,
                contrast_token_ids=c_tok_ids,
                healthy_cost=healthy_cost,
                grad_checkpointing=True,
                top_k_mlp=top_k_mlp,
                top_k_attn=0,
                capture_cache=False,
            )
            d = out_root / f"trace_{i}"
            ct.save(d)
            traces.append(ct)
            trace_dirs.append(str(d))
        diag = diagnose_causal_traces(
            traces,
            split_ab=len(traces) >= 2,
            top_k_sites=6,
            top_n_neurons=64,
            healthy_cost=healthy_cost,
            model_or_weights=(model if healthy_cost == "fisher" else None),
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out: dict[str, Any] = {
        "n_traces": len(traces),
        "grad_mode": grad_mode,
        "grad_objective": traces[0].meta.get("grad_objective"),
        "grad_healthy_objective": traces[0].meta.get("grad_healthy_objective"),
        "trace_dirs": trace_dirs,
        "causal_diagnosis": {
            "summary": diag["summary"],
            "sites": diag["sites"],
            "causal_rank": diag["causal_rank"][:10],
            "passive_rank": diag["passive_rank"][:10],
            "top_neurons": {k: v[:16] for k, v in diag["top_neurons"].items()},
        },
    }
    if t_dir is not None:
        out["direction_summary"] = summarize_direction_traces(
            traces, top_k_modules=10, top_k_heads=10, top_k_neurons=10
        )
    return out


def _attribute_direction_writers_tool(args: dict[str, Any]) -> dict[str, Any]:
    import gc

    import torch

    seqs = _resolve_sequences_arg(args, args["model_dir"], required=True)
    assert seqs is not None
    cseqs = _resolve_sequences_arg(
        args, args["model_dir"], seq_key="contrast_sequences", text_key="contrast_texts", required=False
    )
    t_dir, t_layer, t_conv = _load_direction_arg(args, "target", conv_key="layer_convention")
    if t_dir is None:
        raise ValueError("Provide `target_dir`, `target_dir_npz`, or `target_patch_path`")
    if t_layer is None:
        raise ValueError("Provide `target_layer` (or use `target_patch_path` containing router layer)")
    model = _load_model_for_screen(
        args["model_dir"],
        dtype_name=str(args.get("dtype", "auto")),
        device_map=str(args.get("device_map", "auto")),
    )
    try:
        return attribute_direction_writers(
            model,
            seqs,
            target_dir=t_dir,
            target_layer=t_layer,
            layer_convention=t_conv or "block",
            pos=_parse_pos_arg(args.get("pos")),
            contrast_inputs=cseqs,
            top_k_neurons=int(args.get("top_k_neurons", 16)),
            top_k_heads=int(args.get("top_k_heads", 16)),
            normalize_dir=bool(args.get("normalize_dir", False)),
            linearize_final_norm=bool(args.get("linearize_final_norm", True)),
            report_block_gaps=bool(args.get("report_block_gaps", False)),
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _trace_direction_graph_tool(args: dict[str, Any]) -> dict[str, Any]:
    import gc

    import torch

    seqs = _resolve_sequences_arg(args, args["model_dir"], required=True)
    assert seqs is not None
    t_dir, t_layer, t_conv = _load_direction_arg(args, "target", conv_key="layer_convention")
    if t_dir is None:
        raise ValueError("Provide `target_dir`, `target_dir_npz`, or `target_patch_path`")
    if t_layer is None:
        raise ValueError("Provide `target_layer` (or use `target_patch_path` containing router layer)")
    model = _load_model_for_screen(
        args["model_dir"],
        dtype_name=str(args.get("dtype", "auto")),
        device_map=str(args.get("device_map", "auto")),
    )
    try:
        return trace_direction_graph(
            model,
            seqs,
            target_dir=t_dir,
            target_layer=t_layer,
            layer_convention=t_conv or "block",
            pos=_parse_pos_arg(args.get("pos")),
            depth=int(args.get("depth", 2)),
            top_k_nodes=int(args.get("top_k_nodes", 3)),
            min_layer=int(args.get("min_layer", 0)),
            min_edge_weight=float(args.get("min_edge_weight", 1e-6)),
            normalize_dir=bool(args.get("normalize_dir", False)),
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _peel_causal_circuit_tool(args: dict[str, Any]) -> dict[str, Any]:
    import gc
    from pathlib import Path

    import torch

    seqs = _resolve_sequences_arg(args, args["model_dir"], required=True)
    assert seqs is not None
    t_dir, t_layer, t_conv = _load_direction_arg(args, "target")
    p_dir, p_layer, p_conv = _load_direction_arg(args, "protect")

    def _norm_spans(raw: Any, is_target: bool) -> list[Any] | None:
        if not raw:
            return None
        out_s: list[Any] = []
        for sp in raw:
            if sp is None:
                out_s.append(None)
            elif isinstance(sp, (list, tuple)):
                out_s.append((int(sp[0]), int(sp[1])))
            elif is_target:
                out_s.append(int(sp))
            else:
                raise ValueError(f"Invalid healthy_span {sp!r}; expected [start, end] or null")
        return out_s

    tspans = _norm_spans(args.get("target_spans"), is_target=True)
    hspans = _norm_spans(args.get("healthy_spans"), is_target=False)
    t_tok_ids = [int(x) for x in args["target_token_ids"]] if args.get("target_token_ids") is not None else None
    c_tok_ids = [int(x) for x in args["contrast_token_ids"]] if args.get("contrast_token_ids") is not None else None
    top_n_neurons = int(args["top_n_neurons"]) if args.get("top_n_neurons") is not None else None

    model = _load_model_for_screen(
        args["model_dir"],
        dtype_name=str(args.get("dtype", "auto")),
        device_map=str(args.get("device_map", "auto")),
    )
    try:
        res = peel_causal_circuit(
            model,
            seqs,
            n_rounds=int(args.get("n_rounds", 2)),
            alpha=float(args.get("alpha", 4.0)),
            sites_per_round=int(args.get("sites_per_round", 1)),
            top_k_sites=int(args.get("top_k_sites", 6)),
            top_n_neurons=top_n_neurons,
            allow_repeat_modules=bool(args.get("allow_repeat_modules", False)),
            target_dir=t_dir,
            target_layer=t_layer,
            target_layer_convention=t_conv or "block",
            protect_dir=p_dir,
            protect_layer=p_layer,
            protect_layer_convention=p_conv,
            target_spans=tspans,
            healthy_spans=hspans,
            grad_objective=str(args.get("grad_objective", "auto")),
            target_token_ids=t_tok_ids,
            contrast_token_ids=c_tok_ids,
            healthy_cost=str(args.get("healthy_cost", "linear")),
            center=str(args.get("center", "none")),
            grad_checkpointing=True,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    patch = res.pop("cumulative_patch")
    saved_patch = None
    if args.get("save_patch_path"):
        spath = Path(str(args["save_patch_path"]))
        spath.parent.mkdir(parents=True, exist_ok=True)
        saved_patch = str(patch.save(spath))
    res["saved_patch"] = saved_patch
    res["n_edits"] = len(patch.edits)
    return res


def _bake_moipatch_tool(args: dict[str, Any]) -> dict[str, Any]:
    from .serve import bake_moipatch

    paths: list[str] = []
    if args.get("patch_paths") is not None:
        raw_list = args["patch_paths"]
        if not isinstance(raw_list, list):
            raise ValueError("`patch_paths` must be a list of `.moipatch` file paths")
        paths.extend(str(p) for p in raw_list)
    if args.get("patch_path"):
        paths.append(str(args["patch_path"]))
    if not paths:
        raise ValueError("Provide `patch_paths` (array) or `patch_path` (string)")

    return bake_moipatch(
        args["model_dir"],
        paths,
        args["out_dir"],
        verify_fingerprint=bool(args.get("verify_fingerprint", True)),
        allow_routed=bool(args.get("allow_routed", False)),
        copy_untouched=str(args.get("copy_untouched", "auto")),
    )


def _resolve_patch_load_opts(patch: MoIPatch, args: dict[str, Any]) -> tuple[str, str]:
    cpu = bool(args.get("cpu", False))
    dtype_name = str(args.get("dtype", "auto"))
    if cpu:
        eff_dtype = "float32"
        eff_dev_map = ""
    else:
        if dtype_name == "auto" and patch.edits and patch.edits[0].base_dtype in ("bfloat16", "float16", "float32"):
            eff_dtype = str(patch.edits[0].base_dtype)
        else:
            eff_dtype = dtype_name
        eff_dev_map = str(args.get("device_map", "auto"))
    return eff_dtype, eff_dev_map


def _audit_moipatch_tool(args: dict[str, Any]) -> dict[str, Any]:
    import gc
    import torch
    from .moipatch import _apply_cli_chat

    patch_path = str(args["patch_path"])
    patch = MoIPatch.load(patch_path)
    eff_dtype, eff_dev_map = _resolve_patch_load_opts(patch, args)
    model = _load_model_for_screen(args["model_dir"], dtype_name=eff_dtype, device_map=eff_dev_map)
    try:
        probes = [str(p) for p in args["probe_prompts"]] if args.get("probe_prompts") is not None else None
        controls = [str(p) for p in args["control_prompts"]] if args.get("control_prompts") is not None else None
        needs_tok = (probes is not None and args.get("probe_ids") is None) or (
            controls is not None and args.get("control_ids") is None
        )
        try:
            tok = _load_tokenizer_for_screen(args["model_dir"])
        except Exception:
            if needs_tok:
                raise
            tok = None

        use_chat = bool(args.get("chat", False))
        if use_chat and tok is not None:
            if probes is not None:
                probes = [_apply_cli_chat(tok, p, True) for p in probes]
            if controls is not None:
                controls = [_apply_cli_chat(tok, p, True) for p in controls]

        return patch.audit(
            model,
            tok,
            probe_prompts=probes,
            probe_ids=args.get("probe_ids"),
            control_prompts=controls,
            control_ids=args.get("control_ids"),
            patch_path=patch_path,
            top_k=int(args.get("top_k", 10)),
            verify_fingerprint=bool(args.get("verify_fingerprint", True)),
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _calibrate_safety_router_tool(args: dict[str, Any]) -> dict[str, Any]:
    import gc
    import torch
    from .router_cal import calibrate_safety_router_on_model

    patch = MoIPatch.load(str(args["patch_path"]))
    eff_dtype, eff_dev_map = _resolve_patch_load_opts(patch, args)
    model = _load_model_for_screen(args["model_dir"], dtype_name=eff_dtype, device_map=eff_dev_map)
    try:
        tok = _load_tokenizer_for_screen(args["model_dir"])
        return calibrate_safety_router_on_model(
            model,
            tok,
            patch,
            toxic_prompts=[str(p) for p in args["toxic_prompts"]],
            benign_refused_prompts=[str(p) for p in args["benign_refused_prompts"]],
            control_prompts=[str(p) for p in args["control_prompts"]],
            benign_nonrefused_prompts=(
                [str(p) for p in args["benign_nonrefused_prompts"]]
                if args.get("benign_nonrefused_prompts") is not None
                else None
            ),
            chat=bool(args.get("chat", True)),
            max_toxic_fpr=float(args.get("max_toxic_fpr", 0.05)),
            n_folds=int(args.get("n_folds", 4)),
            out_patch_path=args.get("out_patch_path"),
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def call_wsi_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch an MCP tool call to `WeightSpaceIndex`."""
    if name == "wsi_overview":
        idx = _get_wsi(args["model_dir"], args.get("index_dir"))
        return idx.overview()
    if name == "wsi_search_token":
        idx = _get_wsi(args["model_dir"])
        q = int(args["query"]) if str(args["query"]).isdigit() else args["query"]
        return idx.search_token(
            q,
            top_layers=int(args.get("top_layers", 5)),
            top_units_per_layer=int(args.get("top_units_per_layer", 5)),
        )
    if name == "wsi_search_transduction":
        idx = _get_wsi(args["model_dir"])
        return idx.search_transduction(
            args["input_query"],
            args["output_query"],
            top_k=int(args.get("top_k", 8)),
        )
    if name == "wsi_explain_neuron":
        idx = _get_wsi(args["model_dir"])
        return idx.explain_neuron(
            int(args["layer"]),
            int(args["unit"]),
            args.get("module", "mlp_out"),
        )
    if name == "wsi_trace_circuit":
        idx = _get_wsi(args["model_dir"])
        return idx.trace_circuit(
            int(args["layer"]),
            int(args["unit"]),
            top_k=int(args.get("top_k", 5)),
        )
    if name == "wsi_trace_internal_state":
        idx = _get_wsi(args["model_dir"])
        return idx.trace_internal_state(
            layer=int(args["layer"]),
            head=int(args["head"]) if args.get("head") is not None else None,
            neurons=[int(x) for x in args["neurons"]] if args.get("neurons") else None,
            top_k_neurons=int(args.get("top_k_neurons", 64)),
        )
    if name == "wsi_explain_edit":
        idx = _get_wsi(args["model_dir"])
        return idx.explain_edit(args["edit_path"])
    if name == "wsi_rewire_transduction":
        idx = _get_wsi(args["model_dir"])
        res = idx.rewire_transduction(
            args["input_query"],
            args["old_output"],
            args["new_output"],
            top_k_neurons=int(args.get("top_k_neurons", 8)),
            strength=float(args.get("strength", 2.5)),
            save_path=args.get("save_path"),
        )
        return {k: v for k, v in res.items() if k != "_tensors"}
    if name == "wsi_diff_models":
        base_idx = _get_wsi(args["base_model_dir"])
        return base_idx.diff_model(args["target_model_dir"])
    if name == "wsi_align_circuits":
        idx_a = _get_wsi(args["model_a_dir"])
        pairs = [tuple(p) for p in args.get("transduction_pairs") or []]
        return idx_a.align_circuits(
            args["model_b_dir"],
            transduction_pairs=pairs,
            concepts=args.get("concepts"),
        )
    if name == "wsi_transfer_edit":
        idx_src = _get_wsi(args["src_model_dir"])
        res = idx_src.transfer_edit_to(
            args["dst_model_dir"],
            args["edit_path"],
            top_k_neurons=int(args.get("top_k_neurons", 8)),
            strength=float(args.get("strength", 2.5)),
            save_path=args.get("save_path"),
        )
        return {k: v for k, v in res.items() if k != "_tensors"}
    if name == "wsi_diff_compact":
        idx = _get_wsi(args["model_dir"])
        ct_a = CompactTrace.load(args["trace_a_dir"])
        ct_b = CompactTrace.load(args["trace_b_dir"])
        return idx.diff_compact(
            ct_a,
            ct_b,
            pos_a=int(args.get("pos_a", -1)),
            pos_b=int(args.get("pos_b", -1)),
            top_k=int(args.get("top_k", 10)),
        )
    if name == "wsi_discover_compact_circuit":
        idx = _get_wsi(args["model_dir"])
        c_alphas = tuple(float(x) for x in args.get("candidate_alphas", (4.0, 8.0, 12.0)))
        return discover_compact_dirs(
            idx,
            args["trace_dirs"],
            loop_trace_dirs=args.get("loop_trace_dirs"),
            alpha=float(args.get("alpha", 6.0)),
            top_k_neurons=int(args.get("top_k_neurons", 64)),
            variant=str(args.get("patch_variant", "unified_top2")),
            save_patch_path=args.get("save_patch_path"),
            save_candidate_grid_dir=args.get("save_candidate_grid_dir"),
            candidate_alphas=c_alphas,
        )
    if name == "wsi_build_moipatch":
        return _build_moipatch_tool(args)
    if name == "wsi_screen_candidates":
        return _screen_candidates_tool(args)
    if name == "wsi_capture_causal_traces":
        return _capture_causal_traces_tool(args)
    if name == "wsi_attribute_direction_writers":
        return _attribute_direction_writers_tool(args)
    if name == "wsi_trace_direction_graph":
        return _trace_direction_graph_tool(args)
    if name == "wsi_peel_causal_circuit":
        return _peel_causal_circuit_tool(args)
    if name == "wsi_bake_moipatch":
        return _bake_moipatch_tool(args)
    if name == "wsi_audit_moipatch":
        return _audit_moipatch_tool(args)
    if name == "wsi_calibrate_safety_router":
        return _calibrate_safety_router_tool(args)
    if name == "wsi_audit_lora":
        from .lora_audit import audit_lora

        return audit_lora(
            args["base_model_dir"],
            args["adapter_path"],
            top_k=int(args.get("top_k", 20)),
            top_sv=int(args.get("top_sv", 4)),
            apply_final_norm=bool(args.get("apply_final_norm", True)),
            lora_scale=float(args["lora_scale"]) if args.get("lora_scale") is not None else None,
            device=str(args.get("device", "cpu")),
        )
    raise ValueError(f"Unknown WSI MCP tool: `{name}`")


def handle_rpc_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 request message and return the response object."""
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "weight-space-index", "version": "0.2.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS_SCHEMA},
        }
    if method == "tools/call":
        params = msg.get("params") or {}
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        try:
            output = call_wsi_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(output, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }
    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return None


def run_stdio_server() -> None:
    """Run the JSON-RPC 2.0 MCP server over stdin/stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_rpc_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
