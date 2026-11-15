# Weight-Space Index (`wsi`)

**A semantic decompiler, symbol table, and navigation index for neural network weights.**

Built to work standalone or directly on top of [`wsr` (Weight-Space Reading and Programming)](./wsr), **Weight-Space Index (`wsi`)** allows Large Language Models (and humans) to inspect, search, trace, compare, and surgically edit `.safetensors` checkpoints **without running GPU forward passes or training**.

---

## ⚡ Headline Benchmarks: `wsr` Baseline vs. `wsi` (Powered by Native Rust SIMD Core)

In standard [`wsr`](./wsr/README.md), an LLM agent inspects or edits raw weight tensors by writing ad-hoc Python scripts in a CPU sandbox (`hours: 0.5` to `1.5` per session). With **`wsi`** (and its zero-copy multithreaded **Rust SIMD engine** [`rust/`](./rust)), both **Blind Weight-Space Reading (`wsr read`)** and **Surgical Weight-Space Programming (`wsr write`)** run in **tens of milliseconds**:

| Task | Standard `wsr` Session Budget | With `wsi` (Rust SIMD Core + Python) | Speedup | Accuracy / Success Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Blind Weight-Space Reading (`wsr read` — GPT-2)**<br>*(Decode 50 mystery `.safetensors` edits across 8 families with zero text hints)* | **30 – 90 min**<br>*(per batch of ~9 edits)* | **2.09 sec total**<br>**(`41.7 ms / edit`)** | **~5,000× faster** | **50 / 50 (`100.0%`)** exact category<br>**50 / 50 (`100.0%`)** exact token/concept |
| **Blind Weight-Space Reading (`LiquidAI/LFM3-1.7B`)**<br>*(Decode 20 mystery `BF16` edits on hybrid ShortConv + SwiGLU architecture)* | **30 – 90 min**<br>*(per batch)* | **0.72 sec total**<br>**(`36.2 ms / edit`)** | **>5,000× faster** | **20 / 20 (`100.0%`)** exact category<br>**20 / 20 (`100.0%`)** exact token/concept |
| **Edit Synthesis (`wsr write` generation)**<br>*(Create 50 `.safetensors` edits including 10 neuron-level transduction rewires)* | **30 – 60 min**<br>*(per single edit goal)* | **3.77 sec total**<br>**(`75.4 ms / edit`)** | **>2,000× faster** | **50 / 50 (`100.0%`)** valid `.safetensors` |
| **Surgical Fact Rewiring (`rewire_transduction`)**<br>*(Flip 5 real facts across 5 domains by editing 8/36,864 MLP neurons = `0.022%`)* | **30 – 60 min**<br>*(per edit + trial passes)* | **0.98 sec total**<br>**(`195.7 ms / edit`)** | **>1,800× faster** | **5 / 5 (`100.0%`)** target predictions flipped<br>**70 / 75 (`93.3%`)** guard probes preserved |

### 1. Blind Reading Benchmark (`python scripts/benchmark_blind_reading.py`)
Tested on **50 mystery `.safetensors` edits** applied to real pretrained GPT-2 (`124M` parameters, `50,257` vocab tokens) across 8 edit families—where `wsi.explain_edit()` receives **only** the binary `.safetensors` file path:

| Edit Family | Mystery Edits | Exact Concept / Token Recovery | Avg. Blind Read Time (Rust SIMD) |
| :--- | :---: | :---: | :---: |
| **Surgical Neuron Rewire (`rewire_transduction`)** *(e.g. `Paris→Rome`, `Tokyo→Cairo`, `Berlin→Madrid`)* | 10 | **10 / 10 (`100%`)** | `116.8 ms` |
| **Orthogonal Subspace Concept Erasure (`orth`)** *(e.g. `Paris`, `London`, `China`, `Japan`)* | 10 | **10 / 10 (`100%`)** | `40.7 ms` |
| **Vocabulary Row Boost (`scale_rows > 1`)** *(e.g. `Rome`, `Python`, `science`, `gold`)* | 8 | **8 / 8 (`100%`)** | `8.7 ms` |
| **Vocabulary Row Suppression (`scale_rows < 1`)** *(e.g. `<\|endoftext\|>`, `error`, `war`)* | 7 | **7 / 7 (`100%`)** | `1.0 ms` |
| **Full Layer / Block Ablation (`scale = 0`)** *(MLP & Attention output projections)* | 5 | **5 / 5 (`100%`)** | `39.5 ms` |
| **LayerNorm Gain Scaling (`gain`)** *(`ln_1`, `ln_2`, `ln_f` across layers)* | 5 | **5 / 5 (`100%`)** | `8.3 ms` |
| **Isotropic Weight Noise (`noise`)** *(Random Gaussian perturbation on MLP/Attn)* | 3 | **3 / 3 (`100%`)** | `39.2 ms` |
| **Imperceptible No-Op Controls (`1.0001x`)** *(Negative controls `< 0.1%` relative norm)* | 2 | **2 / 2 (`100%`)** | `39.3 ms` |
| **Total (All 50 Mystery Edits)** | **50** | **50 / 50 (`100.0%`)** | **`41.7 ms / edit` (`2.09 s` total)** |

### 2. Surgical Weight Programming Benchmark (`python scripts/benchmark_multi_edit.py`)
Without computing a single gradient or running training, `wsi.rewire_transduction()` locates the exact **8 MLP neurons (0.022% of GPT-2's 36,864 MLP neurons)** that transduce `[Trigger Token] -> [Old Concept]` and surgically rewires their output value projections to `[New Concept]`:

| Domain | Surgical Edit (`8 / 36,864` neurons) | Evaluation Prompt | Base GPT-2 Top-1 | Edited GPT-2 Top-1 | Synthesis Time | Held-Out Guard Probes Preserved |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Technology** | `Bill Gates` $\rightarrow$ `Bill Jobs` | *"Microsoft was founded by Bill"* | `' Gates'` (`86.3%`) | **`' Jobs'` (`99.9%`)** | `360.2 ms` | **15 / 15 (`100%`)** |
| **Geography** | `San Francisco` $\rightarrow$ `San Diego` | *"The Golden Gate Bridge is in San"* | `' Francisco'` (`89.4%`) | **`' Diego'` (`65.0%`)** | `151.5 ms` | **15 / 15 (`100%`)** |
| **Literature** | `Romeo & Juliet` $\rightarrow$ `Caesar` | *"William Shakespeare wrote Romeo and"* | `' Juliet'` (`98.7%`) | **`' Caesar'` (`96.0%`)** | `150.2 ms` | **14 / 15 (`93%`)** |
| **CS / AI** | `Artificial Intelligence` $\rightarrow$ `Gravity` | *"In computer science, artificial"* | `' intelligence'` (`93.4%`) | **`' gravity'` (`99.5%`)** | `157.5 ms` | **13 / 15 (`87%`)** |
| **Landmarks** | `Eiffel Tower: Paris` $\rightarrow$ `Rome` | *"The Eiffel Tower is in the city of"* | `' Paris'` (`7.0%`) | **`' Rome'` (`100.0%`)** | `159.0 ms` | **13 / 15 (`87%`)** |

*(Even when all 5 edits are composed simultaneously into the same GPT-2 checkpoint in `0.98s`, **5/5 (`100%`)** target prompts flip to their new concepts simultaneously.)*

### 3. Out-of-Distribution Architecture Generalization: `LiquidAI/LFM3-1.7B` (`python scripts/benchmark_lfm3.py`)
To verify zero overfitting to GPT-2 or standard Transformers, we evaluated `wsi` on real `BF16` pretrained weights from [**`LiquidAI/LFM3-1.7B`**](https://huggingface.co/LiquidAI/LFM3-1.7B) (`model_type: "lfm3"`), a non-Transformer hybrid **Liquid Foundation Model** with:
* **Gated Short Convolutions** (`conv.in_proj`, depthwise `conv.conv`, `conv.out_proj`) interleaved with **Grouped-Query Attention** (`self_attn.q/k/v/o/gate_proj`)
* **7,680-unit SwiGLU MLPs** (`feed_forward.w1` gate, `feed_forward.w3` up, `feed_forward.w2` down, `[2048, 7680]`)
* **4 Sandwich RMSNorms per block** (`operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm`) + `model.embedding_norm.weight`

Using the **exact same `classify_blind_diagnosis()` rules** without changing a single threshold:
* **Blind Weight-Space Reading (`20 / 20 = 100.0%`):** Recovered **20 / 20 (`100%`)** exact edit categories and **20 / 20 (`100%`)** exact concepts/tokens across all 8 edit families (`rewire_transduction` on 7,680-unit SwiGLU `w2`, `orth` on `conv.out_proj` & `self_attn.o_proj`, `scale_rows`, `block_ablation`, `gain` on sandwich norms, `noise`, and `no_op`) in **`0.72 seconds` total (`36.2 ms / edit`)** using the zero-copy `BF16` Rust core.
* **Zero-Shot SwiGLU Circuit Discovery:** Querying `wsi.search_transduction("ĠFrance", "ĠParis")` on `LFM3-1.7B` immediately pinpointed **Layer 0 SwiGLU Unit 1625** (`[Ġpare, Ġparas, ĠParis] -> [ĠPar, par, Par]`).

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

## Native Rust SIMD Core & Reproducible Benchmarks

```bash
# 0. Compile the zero-dependency multithreaded Rust SIMD core (`wsi_core` + `wsi-rs` CLI)
cargo build --release --manifest-path rust/Cargo.toml

# 1. Run the 12-step unit & cross-architecture verification suite
python tests/test_wsi.py

# 2. Run the 50-edit Blind Weight-Space Reading benchmark on real GPT-2
python scripts/benchmark_blind_reading.py

# 3. Run the 20-edit Out-of-Distribution benchmark on LiquidAI/LFM3-1.7B (BF16 hybrid architecture)
python scripts/benchmark_lfm3.py

# 4. Run the 5-domain Surgical Weight Programming & Collateral Damage benchmark
python scripts/benchmark_multi_edit.py

# 5. Run the live pretrained GPT-2 circuit discovery & surgical rewire demo
python scripts/demo_real_gpt2.py
```

---

## Credits & License

* Built on top of and compatible with [**`wsr` (Weight-Space Reading)**](https://github.com/suayptalha/wsr) by Şuayp Talha Kocabay (2026).
* Licensed under the [Apache License 2.0](./LICENSE).

