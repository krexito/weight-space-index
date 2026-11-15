# Weight-Space Index (`wsi`)

**A semantic decompiler, symbol table, and navigation index for neural network weights.**

Built to work standalone or directly on top of [`wsr` (Weight-Space Reading and Programming)](./wsr), **Weight-Space Index (`wsi`)** allows Large Language Models (and humans) to inspect, search, trace, compare, and surgically edit `.safetensors` checkpoints **without running GPU forward passes or training**.

---

## Why Weight-Space Index?

In standard weight-space reading (`wsr`), an LLM agent is given raw weight matrices (`model.layers.X.mlp.down_proj.weight`) and must scan millions of floats on a CPU.

`wsi` solves this by projecting both sides of every layer onto the model's own vocabulary anchors—the **Input Embedding Matrix ($E_{\text{in}}$)** and the **Centered Effective Unembedding Matrix ($U_{\text{eff}}$)**:
1. **Input-Side Key Indexing (`mlp_in`, `attn_q/k/v`):** Identifies which input tokens/concepts **trigger (activate)** each neuron (`E_in @ K_eff`).
2. **Output-Side Value Indexing (`mlp_out`, `attn_out`):** Identifies which output tokens/concepts each neuron **writes (promotes or suppresses)** into the residual stream (`U_eff @ W_res`).
3. **Associative Transduction Rules (`[Input Triggers] -> [Output Promoted]`):** Turns opaque `(layer, unit)` coordinates into readable rules like `"[Eiffel, France] -> [Paris, Louvre]"`.
4. **Universal Semantic Coordinate System across Models:** Because different architectures (e.g., Llama vs. Qwen vs. GPT-2) share human language at their vocabulary boundaries, `wsi` can **align equivalent circuits across completely different architectures** and **automatically transfer a `.safetensors` weight edit from Model A to Model B**.

---

## Key Capabilities

| Feature | Method / CLI | What it does |
|---|---|---|
| **Instant On-Demand (`lazy=True`) or Pre-Built Index** | `WeightSpaceIndex.from_model(dir)` / `wsi build` | Opens any checkpoint in ~0s (`lazy=True` default) or pre-computes `index.json` + `subspaces.safetensors` |
| **Concept Search** | `I.search_token("Paris")` / `wsi search` | Finds the exact layers, SVD subspaces, input-triggered neurons, and output-promoting neurons for a token |
| **Transduction Search** | `I.search_transduction("France", "Paris")` / `wsi transduction` | Finds Key-Value memory neurons triggered by `input_query` that write `output_query` |
| **Neuron Explainer** | `I.explain_neuron(layer=8, unit=430)` / `wsi neuron` | Shows a neuron's input triggers, inhibitors, output promoted/suppressed tokens, and transduction rule |
| **Causal Circuit Tracer** | `I.trace_circuit(layer=2, unit=7)` / `wsi trace` | Traces upstream writer neurons ($l' < l$) that feed this neuron and downstream reader neurons ($l'' > l$) |
| **Edit Diagnosis** | `I.explain_edit("edit.safetensors")` / `wsi explain-edit` | Decompiles any `wsr` edit into rank, energy change, `subspace_action`, and vocabulary shifts |
| **Surgical Fact Rewiring** | `I.rewire_transduction("Tower", "Paris", "Rome")` / `wsi rewire` | Rewires only the top $K$ transducing neurons (~0.06% of neurons) to output a new concept without training |
| **Same-Family Semantic Diff** | `I.diff_model(tuned_dir)` / `wsi diff` | Semantic `git diff` between Base and Fine-Tuned/Edited checkpoints |
| **Cross-Architecture Alignment** | `I.align_circuits(other_model)` / `wsi align` | Matches functionally equivalent neurons across different architectures (e.g. Llama $\leftrightarrow$ Qwen2) |
| **Cross-Model Edit Transfer** | `I.transfer_edit_to(model_b, "edit_a.safetensors")` / `wsi transfer-edit` | Decompiles a binary edit from Model A and synthesizes the equivalent `.safetensors` edit for Model B |
| **MCP Server** | `python -m wsi mcp` | Exposes all 10 `wsi_*` tools over standard JSON-RPC 2.0 stdio for any MCP-compatible AI agent |

---

## Quickstart

### 1. Python API

```python
from wsi import WeightSpaceIndex

# Opens in ~0s in on-demand lazy mode (or loads prebuilt index if index_dir is passed)
I = WeightSpaceIndex.from_model("models/gpt2")

# 1. Find real neurons in GPT-2 that transduce " France" -> " Paris"
hits = I.search_transduction("ĠFrance", "ĠParis", top_k=5)
print(hits["top_transducing_neurons"][0])
# -> {'layer': 8, 'unit': 430, 'in_tensor': 'h.8.mlp.c_fc.weight', 'out_tensor': 'h.8.mlp.c_proj.weight', ...}

# 2. Inspect Layer 8 Neuron 430
print(I.explain_neuron(layer=8, unit=430)["transduction_rule"])
# -> "[ère,  Qué, France] -> [ Rouge,  Hollande,  Qué]"

# 3. Surgically rewire " Tower" -> " Paris" into " Tower" -> " Rome" (24 neurons out of 36,864)
I.rewire_transduction(
    "ĠTower", "ĠParis", "ĠRome",
    top_k_neurons=24, strength=6.0,
    save_path="edit_eiffel_rome.safetensors",
)

# 4. Transfer a .safetensors edit from Model A (e.g. Llama) to Model B (e.g. Qwen2)
I_llama.transfer_edit_to("models/qwen2", "llama_edit.safetensors", save_path="qwen2_edit.safetensors")
```

### 2. Command-Line Interface (`python -m wsi`)

```bash
# Build a static index on disk (optional; all commands also work on-the-fly without building first)
python -m wsi build models/gpt2 -o models/gpt2_index

# Search which layers and neurons write or trigger on a token
python -m wsi search models/gpt2 "ĠParis"

# Find Key-Value transduction neurons [Input] -> [Output]
python -m wsi transduction models/gpt2 "ĠFrance" "ĠParis"

# Explain a single neuron and trace its upstream/downstream causal circuit
python -m wsi neuron models/gpt2 --layer 8 --unit 430
python -m wsi trace models/gpt2 --layer 8 --unit 430

# Surgically rewire a transduction and diagnose the resulting .safetensors edit
python -m wsi rewire models/gpt2 "ĠTower" "ĠParis" "ĠRome" -o edit_rome.safetensors --top-k 24 --strength 6.0
python -m wsi explain-edit models/gpt2 edit_rome.safetensors

# Compare two checkpoints or transfer an edit across different architectures
python -m wsi diff models/llama_base models/llama_tuned
python -m wsi align models/llama models/qwen2 --pair Eiffel:Paris
python -m wsi transfer-edit models/llama models/qwen2 llama_edit.safetensors -o qwen_edit.safetensors
```

### 3. Inside `wsr` Sandbox Sessions

`wsi` is integrated directly into [`wsr/wsr/sandbox_lib.py`](./wsr/wsr/sandbox_lib.py):

```python
from wsr.sandbox_lib import W, I

I.overview()
I.search_transduction("Eiffel", "Paris")
I.explain_edit("/interventions/edit_01.safetensors")
```

### 4. MCP Server Configuration

Add `wsi` to any MCP client (`claude_desktop_config.json`, Cursor, Antigravity, etc.):

```json
{
  "mcpServers": {
    "weight-space-index": {
      "command": "python",
      "args": ["-m", "wsi", "mcp"],
      "cwd": "/path/to/WEIGHT-SPACE-INDEX"
    }
  }
}
```

---

## Verification & Demos

```bash
# Run the 11-step unit & cross-architecture verification suite
python tests/test_wsi.py

# Run the live pretrained GPT-2 circuit discovery & surgical rewire demo
python scripts/demo_real_gpt2.py
```

---

## Credits & License

* Built on top of and compatible with [**`wsr` (Weight-Space Reading)**](https://github.com/suayptalha/wsr) by Şuayp Talha Kocabay (2026).
* Licensed under the [Apache License 2.0](./LICENSE).

