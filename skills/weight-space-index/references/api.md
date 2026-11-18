# wsi Python API (signatures)

```python
from wsi import WeightSpaceIndex
I = WeightSpaceIndex.from_model(model_dir, index_dir=None, *, lazy=None, rebuild=False)
```

| Method | Signature | Returns (main keys) |
|---|---|---|
| `overview` | `I.overview()` | layers, widths, architecture summary |
| `search_token` | `I.search_token(query, *, top_layers=5, top_units_per_layer=5)` | layers and subspaces that write or read the token, and their top neurons |
| `search_transduction` | `I.search_transduction(input_query, output_query, *, top_k=8)` | `input_token`, `output_token`, `top_transducing_neurons` (layer, unit, in/out tensor, score) |
| `explain_neuron` | `I.explain_neuron(layer, unit, module="mlp_out", top_k=8)` | input triggers and inhibitors, promoted and suppressed tokens, `transduction_rule` |
| `trace_circuit` | `I.trace_circuit(layer, unit, *, top_k=5)` | upstream writers (earlier layers) and downstream readers (later layers) |
| `trace_internal_state` | `I.trace_internal_state(vec=None, *, layer=None, neurons=None, head=None, ...)` | attention heads and MLP blocks aligned with a hidden-state direction |
| `explain_edit` | `I.explain_edit(edit_source, top_k=8)` | rank, `subspace_action`, `head_attribution`, `internal_state_circuit`, vocabulary shifts |
| `rewire_transduction` | `I.rewire_transduction(input_query, old_output_query, new_output_query, *, top_k_neurons=8, strength=2.5, save_path=None)` | the edit (and the `.safetensors` file when `save_path` is given) |
| `diff_model` | `I.diff_model(other)` | per-layer changes and the concepts they move |
| `align_circuits` | `I.align_circuits(other, *, transduction_pairs=None, concepts=None, top_k=...)` | matched neurons across the two models |
| `transfer_edit_to` | `I.transfer_edit_to(target_model, edit_source, *, top_k_neurons=8, strength=2.5, save_path=None)` | the equivalent edit for the target model |
| `diff_compact` | `I.diff_compact(tr_a, tr_b, *, model=None, pos_a=-1, pos_b=-1, top_k=10)` | `layer_curve`, `top_mlp_contributors`, `top_attention_heads` ($W_O$-projected), `top_cache_deltas`, `peak_vector_readout` |
| `discover_compact_circuit` | `I.discover_compact_circuit(traces, *, loop_traces=None, model=None, alpha=6.0, top_k_neurons=64)` | `fa_layers`, `mlp_ranked`, `hybrid_ranked`, `wsi_origins`, `wsi_resonant_src`, `wsi_resonant_dst`, `wsi_pair_layers`, `wsi_fa3_layers`, `wsi_resonance_table`, `layer_stats`, `patch`, `patches` |

`query` arguments accept a token string (`"ĠParis"`) or a token id.

## `.moipatch` (`wsi.moipatch`)

```python
from wsi.moipatch import MoIPatch
p = MoIPatch.load(path)          # p.edits, p.router, p.gates, p.meta (name, base_model, policy, expected)
print(p.describe())
h = p.attach(model)              # raises if the model's weights don't match the fingerprint
p.fires(hidden_states, confidence=None)   # router (hidden_states[layer], last token) and answer_confidence gates
h.on(); h.off()                  # patch in place / restore the exact original bytes
h.check_restored(); h.fingerprint_ok()
```

Format: one safetensors file. Tensors are `edit.{i}.u` (direction, fp32 or fp64), optional `edit.{i}.idx` (kept columns) and `router.dir`. The metadata key `moipatch` holds a JSON manifest (`format: "moipatch/1"`, edits with module suffix, alpha, base SHA-256 and dtype, router, gates, meta).

## Runtime capture & compact discovery (`wsi.capture`, `wsi.tracer`, needs `transformers`)

- `wsi.capture.trace_all(model, ids)`: dense uncompressed per-layer trace of one prompt.
- `wsi.tracer.capture_compact(model, ids, *, resid_quant="int8", mlp_quant="fp4", top_k_mlp="auto", head_quant="int8", top_k_attn=16, cache_quant="int8")`: compressed token-level trace (`INT8+had64` residuals, auto-95%-energy `FP4` MLP units, `INT8` pre-$W_O$ head vectors + post-$W_O$ L2 norms, sparse `Top-16` attention links, `INT8` KV/DeltaNet cache); `.save(dir)` / `CompactTrace.load(dir)`, `.detect_loop_onset()`, `.compare_positions()`.
- `wsi.tracer.diff_compact(tr_a, tr_b, *, wsi=None, model=None, pos_a=-1, pos_b=-1, top_k=10)`: X-ray across two `CompactTrace`s of any length (`layer_curve`, `top_mlp_contributors`, $W_O$-projected `top_attention_heads`, `top_cache_deltas`, `peak_vector_readout`). Also `I.diff_compact(...)`.
- `wsi.tracer.diff_traces(model, ids_a, ids_b, wsi=I, *, compact=True)`: runs `capture_compact` + `diff_compact` by default (`compact=False` uses dense `trace_all`).
- `wsi.tracer.discover_compact_circuit(traces, *, loop_traces=None, model=None, wsi=None, alpha=6.0, top_k_neurons=64)`: hybrid `CompactTrace` origin scoring + static `WSI` SVD resonance downstream amplification, returning `wsi_origins`, `wsi_resonant_dst`, `wsi_pair_layers`, `wsi_fa3_layers`, `wsi_resonance_table`, `layer_stats`, and ready-to-attach `MoIPatch` objects (`patch`, `patches`). Also `I.discover_compact_circuit(...)`.
