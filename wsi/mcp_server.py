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

from .query import WeightSpaceIndex
from .tracer import CompactTrace, discover_compact_dirs

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
                "patch_variant": {
                    "type": "string",
                    "default": "unified_top2",
                    "enum": ["unified_top2", "blind_top2", "attn_top2", "wsi_top3", "wsi_fa3", "wsi_pair", "mlp_top3", "mlp_fa3"],
                },
            },
            "required": ["model_dir", "trace_dirs"],
        },
    },
]


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
        return discover_compact_dirs(
            idx,
            args["trace_dirs"],
            loop_trace_dirs=args.get("loop_trace_dirs"),
            alpha=float(args.get("alpha", 6.0)),
            top_k_neurons=int(args.get("top_k_neurons", 64)),
            variant=str(args.get("patch_variant", "unified_top2")),
            save_patch_path=args.get("save_patch_path"),
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
