"""Command-line interface for Weight-Space Index (WSI)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .indexer import build_index
from .query import WeightSpaceIndex
from .tracer import CompactTrace, discover_compact_dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wsi",
        description="Weight-Space Index (WSI): Semantic and structural indexing of model weights for LLMs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # wsi build
    p_build = sub.add_parser("build", help="Build a Weight-Space Index from a safetensors model directory")
    p_build.add_argument("model_dir", help="Path to the checkpoint directory")
    p_build.add_argument("-o", "--out", required=True, help="Output directory for index.json and subspaces.safetensors")
    p_build.add_argument("--top-neurons", type=int, default=32, help="Top neurons per layer to index (default: 32)")
    p_build.add_argument("--top-tokens", type=int, default=8, help="Top tokens per unit to record (default: 8)")
    p_build.add_argument("--svd-rank", type=int, default=8, help="SVD rank per matrix (default: 8)")

    # wsi overview
    p_ov = sub.add_parser("overview", help="Print macro overview of a model's Weight-Space Index")
    p_ov.add_argument("model_dir", help="Path to the checkpoint directory")
    p_ov.add_argument("-i", "--index-dir", help="Optional path to prebuilt index directory")

    # wsi search
    p_search = sub.add_parser("search", help="Search which layers, subspaces, and neurons write/read a token")
    p_search.add_argument("model_dir", help="Path to the checkpoint directory")
    p_search.add_argument("query", help="Token substring or integer token ID")
    p_search.add_argument("-i", "--index-dir", help="Optional path to prebuilt index directory")

    # wsi neuron
    p_neuron = sub.add_parser("neuron", help="Explain what triggers a neuron and what it writes into the residual stream")
    p_neuron.add_argument("model_dir", help="Path to the checkpoint directory")
    p_neuron.add_argument("--layer", type=int, required=True, help="Layer index")
    p_neuron.add_argument("--unit", type=int, required=True, help="Neuron/column index")
    p_neuron.add_argument("--module", default="mlp_out", choices=["mlp_out", "attn_out"])
    p_neuron.add_argument("-i", "--index-dir", help="Optional path to prebuilt index directory")

    # wsi trace
    p_trace = sub.add_parser("trace", help="Trace upstream feeders and downstream consumers of a neuron")
    p_trace.add_argument("model_dir", help="Path to the checkpoint directory")
    p_trace.add_argument("--layer", type=int, required=True, help="Layer index")
    p_trace.add_argument("--unit", type=int, required=True, help="Neuron index")
    p_trace.add_argument("-i", "--index-dir", help="Optional path to prebuilt index directory")

    # wsi internal-state
    p_istate = sub.add_parser(
        "internal-state",
        help="Trace a Deep Internal State Circuit across Attention Heads (o_proj) and MLP blocks (down_proj)",
    )
    p_istate.add_argument("model_dir", help="Path to the checkpoint directory")
    p_istate.add_argument("--layer", type=int, required=True, help="Source layer index")
    p_istate.add_argument("--head", type=int, default=None, help="Source attention head index")
    p_istate.add_argument(
        "--neurons",
        type=str,
        default=None,
        help="Comma-separated MLP neuron indices in the source layer (e.g. '4,53,73,84')",
    )
    p_istate.add_argument("-i", "--index-dir", help="Optional path to prebuilt index directory")

    # wsi transduction
    p_trans = sub.add_parser("transduction", help="Find neurons that map an input concept/token to an output concept/token")
    p_trans.add_argument("model_dir", help="Path to the checkpoint directory")
    p_trans.add_argument("input_query", help="Trigger token substring or ID")
    p_trans.add_argument("output_query", help="Output token substring or ID")
    p_trans.add_argument("-i", "--index-dir", help="Optional path to prebuilt index directory")

    # wsi explain-edit
    p_edit = sub.add_parser("explain-edit", help="Diagnose a wsr edit (.safetensors or .json) semantically")
    p_edit.add_argument("model_dir", help="Path to the checkpoint directory")
    p_edit.add_argument("edit_path", help="Path to edit.safetensors or edit.json")
    p_edit.add_argument("-i", "--index-dir", help="Optional path to prebuilt index directory")

    # wsi diff-compact
    p_dc = sub.add_parser("diff-compact", help="X-ray diff two saved CompactTrace directories with WSI grounding")
    p_dc.add_argument("model_dir", help="Path to the checkpoint directory")
    p_dc.add_argument("trace_a", help="Path to baseline CompactTrace directory")
    p_dc.add_argument("trace_b", help="Path to target CompactTrace directory")
    p_dc.add_argument("--pos-a", type=int, default=-1, help="Token position in trace_a (default: -1)")
    p_dc.add_argument("--pos-b", type=int, default=-1, help="Token position in trace_b (default: -1)")
    p_dc.add_argument("--top-k", type=int, default=10, help="Top items per signal (default: 10)")
    p_dc.add_argument("-i", "--index-dir", help="Optional path to prebuilt index directory")

    # wsi discover-circuit
    p_disc = sub.add_parser(
        "discover-circuit",
        help="Discover a multi-layer circuit & .moipatch from saved CompactTraces + WSI weight resonance",
    )
    p_disc.add_argument("model_dir", help="Path to the checkpoint directory")
    p_disc.add_argument("--trace", action="append", default=[], required=True, help="CompactTrace directory (repeatable)")
    p_disc.add_argument("--loop-trace", action="append", default=[], help="Looping CompactTrace directory (repeatable)")
    p_disc.add_argument("--alpha", type=float, default=6.0, help="Edit strength alpha (default: 6.0)")
    p_disc.add_argument("--top-k-neurons", type=int, default=64, help="Neurons per edited layer (default: 64)")
    p_disc.add_argument(
        "--variant",
        default="wsi_fa3",
        choices=["wsi_fa3", "wsi_pair", "mlp_top3", "mlp_fa3"],
        help="Which discovered patch variant to save with -o (default: wsi_fa3)",
    )
    p_disc.add_argument("-o", "--out", default=None, help="Optional output path for the synthesized .moipatch file")
    p_disc.add_argument("-i", "--index-dir", help="Optional path to prebuilt index directory")

    # wsi diff
    p_diff = sub.add_parser("diff", help="Diff two checkpoints of the same architecture (Base vs Fine-Tune)")
    p_diff.add_argument("base_dir", help="Path to base model directory")
    p_diff.add_argument("target_dir", help="Path to fine-tuned or edited model directory")

    # wsi align
    p_align = sub.add_parser("align", help="Align semantic circuits across two different models/architectures")
    p_align.add_argument("model_a", help="Path to first model directory")
    p_align.add_argument("model_b", help="Path to second model directory")
    p_align.add_argument("--pair", action="append", default=[], help="Transduction pair 'Input:Output' (e.g. 'Eiffel:Paris')")
    p_align.add_argument("--concept", action="append", default=[], help="Concept token to align across models")

    # wsi rewire
    p_rewire = sub.add_parser("rewire", help="Surgically rewire neurons transducing Input -> OldOutput to write NewOutput")
    p_rewire.add_argument("model_dir", help="Path to checkpoint directory")
    p_rewire.add_argument("input_query", help="Trigger token substring or ID")
    p_rewire.add_argument("old_output", help="Original output token substring or ID")
    p_rewire.add_argument("new_output", help="New target output token substring or ID")
    p_rewire.add_argument("-o", "--out", required=True, help="Output .safetensors edit file path")
    p_rewire.add_argument("--top-k", type=int, default=8, help="Number of top transducing neurons to rewire")
    p_rewire.add_argument("--strength", type=float, default=2.5, help="Injection strength multiplier")

    # wsi transfer-edit
    p_trans_edit = sub.add_parser(
        "transfer-edit",
        help="Decompile a .safetensors edit from Model A and synthesize the equivalent edit for Model B",
    )
    p_trans_edit.add_argument("src_model", help="Path to source model A directory")
    p_trans_edit.add_argument("dst_model", help="Path to destination model B directory")
    p_trans_edit.add_argument("edit_path", help="Path to source .safetensors or .json edit file")
    p_trans_edit.add_argument("-o", "--out", required=True, help="Output .safetensors edit file for Model B")
    p_trans_edit.add_argument("--top-k", type=int, default=8, help="Number of neurons to rewire in Model B")
    p_trans_edit.add_argument("--strength", type=float, default=2.5, help="Injection strength multiplier")

    # wsi mcp
    sub.add_parser("mcp", help="Start the Weight-Space Index MCP Server over stdio")

    # wsi moipatch
    p_mp = sub.add_parser("moipatch", help="Inspect or verify a portable .moipatch file")
    p_mp.add_argument("moipatch_args", nargs=argparse.REMAINDER, help="Arguments passed to python -m wsi.moipatch")

    args = parser.parse_args(argv)

    if args.cmd == "moipatch":
        from .moipatch import _main
        _main(args.moipatch_args)
        return 0

    if args.cmd == "build":
        data = build_index(
            args.model_dir,
            args.out,
            top_neurons_per_layer=args.top_neurons,
            top_tokens_per_unit=args.top_tokens,
            svd_rank=args.svd_rank,
        )
        print(json.dumps({
            "status": "ok",
            "index_dir": str(Path(args.out).resolve()),
            "layers_indexed": len(data["layers"]),
            "tokens_indexed": len(data["inverted_tokens"]),
        }, indent=2))
        return 0

    if args.cmd == "diff":
        base_idx = WeightSpaceIndex.from_model(args.base_dir)
        print(json.dumps(base_idx.diff_model(args.target_dir), indent=2))
        return 0

    if args.cmd == "align":
        idx_a = WeightSpaceIndex.from_model(args.model_a)
        pairs = [tuple(p.split(":", 1)) for p in args.pair if ":" in p]
        print(json.dumps(idx_a.align_circuits(args.model_b, transduction_pairs=pairs, concepts=args.concept), indent=2))
        return 0

    if args.cmd == "rewire":
        idx = WeightSpaceIndex.from_model(args.model_dir)
        res = idx.rewire_transduction(
            args.input_query,
            args.old_output,
            args.new_output,
            top_k_neurons=args.top_k,
            strength=args.strength,
            save_path=args.out,
        )
        print(json.dumps({k: v for k, v in res.items() if k != "_tensors"}, indent=2))
        return 0

    if args.cmd == "transfer-edit":
        idx_src = WeightSpaceIndex.from_model(args.src_model)
        res = idx_src.transfer_edit_to(
            args.dst_model,
            args.edit_path,
            top_k_neurons=args.top_k,
            strength=args.strength,
            save_path=args.out,
        )
        print(json.dumps({k: v for k, v in res.items() if k != "_tensors"}, indent=2))
        return 0

    if args.cmd == "mcp":
        from .mcp_server import run_stdio_server
        run_stdio_server()
        return 0

    idx = WeightSpaceIndex.from_model(args.model_dir, index_dir=args.index_dir)

    if args.cmd == "overview":
        print(json.dumps(idx.overview(), indent=2))
    elif args.cmd == "search":
        q: str | int = int(args.query) if args.query.isdigit() else args.query
        print(json.dumps(idx.search_token(q), indent=2))
    elif args.cmd == "neuron":
        print(json.dumps(idx.explain_neuron(args.layer, args.unit, args.module), indent=2))
    elif args.cmd == "trace":
        print(json.dumps(idx.trace_circuit(args.layer, args.unit), indent=2))
    elif args.cmd == "internal-state":
        n_list = [int(x.strip()) for x in args.neurons.split(",") if x.strip()] if args.neurons else None
        print(json.dumps(idx.trace_internal_state(layer=args.layer, neurons=n_list, head=args.head), indent=2))
    elif args.cmd == "transduction":
        iq: str | int = int(args.input_query) if args.input_query.isdigit() else args.input_query
        oq: str | int = int(args.output_query) if args.output_query.isdigit() else args.output_query
        print(json.dumps(idx.search_transduction(iq, oq), indent=2))
    elif args.cmd == "explain-edit":
        print(json.dumps(idx.explain_edit(args.edit_path), indent=2))
    elif args.cmd == "diff-compact":
        ct_a = CompactTrace.load(args.trace_a)
        ct_b = CompactTrace.load(args.trace_b)
        print(json.dumps(idx.diff_compact(ct_a, ct_b, pos_a=args.pos_a, pos_b=args.pos_b, top_k=args.top_k), indent=2))
    elif args.cmd == "discover-circuit":
        out = discover_compact_dirs(
            idx,
            args.trace,
            loop_trace_dirs=args.loop_trace or None,
            alpha=args.alpha,
            top_k_neurons=args.top_k_neurons,
            variant=args.variant,
            save_patch_path=args.out,
        )
        print(json.dumps(out, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
