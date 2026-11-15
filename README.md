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
| **MIT CSAIL `exp8_math` (`Qwen/Qwen3.5-0.8B`)**<br>*(Reproduce & decompile the `p50c` 131,072-weight edit that raises MATH-500 `7.9% → 48.1%`)* | **8.5 – 20.3 hours**<br>*(74–158 turns, 45–80 scripts)* | **`134.2 ms` (write)**<br>**`33.1 ms` (Rust scan)** | **>540,000× faster** | **128 / 128 (`100.0%`)** exact L15 & L19 neurons<br>**`0.999998`** delta cosine (`SHA-256` verified) |

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

### 4. MIT CSAIL *Sparks of Weight-Space Programming* Benchmark: `Qwen/Qwen3.5-0.8B` (`python scripts/benchmark_wsp_qwenmath.py`)
We evaluated `wsi` directly on the official artifacts (`p10c`, `p25c`, `p50c` from [`RikiyaT/lm_understand-logs/exp8_math`](https://github.com/RikiyaT/lm_understand-logs)) of the headline experiment in *"Sparks of Weight-Space Programming"* (Takehi, Teoh, Zhang, Khattab — MIT CSAIL, 2026) on **`Qwen/Qwen3.5-0.8B`** (a 24-layer hybrid Gated DeltaNet `linear_attn` + full `self_attn` + SwiGLU architecture), where fixing a reasoning-loop pathology in **131,072 weights (`0.016%`)** raised **MATH-500 from `7.9%` to `48.1%`** and **GSM8K from `13.8%` to `67.1%`**:

| Official MIT CSAIL Run (`Qwen/Qwen3.5-0.8B`) | Baseline Time Without `wsi` | Surgical Edit Applied (`SHA-256` Verified) | `wsi` Rust SIMD Kernel Scan | `wsi` Full Semantic Decompilation / Synthesis | Speedup |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **`p50c`** *(50 passes: MATH-500 `7.9% → 48.1%`, GSM8K `13.8% → 67.1%`)* | **20.3 hours**<br>*(158 turns, 80 scripts)* | Subtract loop direction ($\alpha=6$) from **64 MLP columns in Layer 15 & 64 in Layer 19** (`131,072` weights = `0.016%`) | **`33.1 ms`** | **`134.2 ms` (write)**<br>**`234.7 ms` (blind read)** | **544,573× faster** |
| **`p25c`** *(25 passes: MATH-500 `7.9% → 14.1%`, GSM8K `13.8% → 25.1%`)* | **8.5 hours**<br>*(86 turns, 45 scripts)* | Zero out **256 columns (`[1792:2048]`, Attention Head 7)** in `layers.15.self_attn.o_proj.weight` | **`38.1 ms`** | **`26.7 ms` (blind read)** | **1,145,394× faster** |
| **`p10c`** *(10 passes: MATH-500 `7.9% → 12.3%`, GSM8K `13.8% → 25.7%`)* | **11.6 hours**<br>*(74 turns, 59 scripts)* | Rank-1 `orth` projection ($W' = W - 1.5 u u^\top W$) across `linear_attn.out_proj`, `self_attn.o_proj`, and `mlp.down_proj` | **`184.5 ms`** | **`266.8 ms` (blind read)** | **156,529× faster** |

* **100% Exact Neuron & Subspace Match:** On `p50c`, `wsi` recovers **64 / 64 (`100.0%`)** of the ground-truth modified MLP neurons in Layer 15 and **64 / 64 (`100.0%`)** in Layer 19 (`frozen_weights_manifest.json`), achieving **`0.9999980` delta cosine similarity** and **`99.84%` exact `BF16` bit-level agreement** (`7,327,948 / 7,340,032` weights bit-for-bit identical).
* **Deep Internal State Circuit Discovery (Zero Forward Passes, `58.2 ms`):** Not all circuits write a fixed dictionary word directly to `lm_head`; reasoning/repetition loops operate as **Deep Internal State Circuits** in the intermediate residual stream:
  * **Cross-Layer Echo Detection (`27.17σ`):** `wsi.explain_edit` automatically detects that the principal residual write vector $\mathbf{u}_{15}$ of the 64 MLP neurons in Layer 15 and the principal residual write vector $\mathbf{u}_{19}$ of the 64 MLP neurons in Layer 19 have a cosine similarity of **`0.849005`** in $\mathbb{R}^{1024}$ (**`27.17 sigma`** above the random baseline $1/\sqrt{1024} \approx 0.03125$), identifying a `cross_layer_internal_state_resonance` loop.
  * **Per-Head Attribution (`head_attribution`):** On `p25c`, `wsi.explain_edit` automatically decomposes `layers.15.self_attn.o_proj.weight` (`8 heads × 256d`) and reports `modified_heads=[7], ablated_heads=[7]` (`cols [1792:2048]`).
  * **Linking `p50c` and `p25c` Purely from Weight Geometry (`trace_internal_state`):** Running `wsi.trace_internal_state(vec=u_15)` projects Layer 15's 64-neuron internal state vector across all 8 attention heads in `layers.15.self_attn.o_proj.weight` and ranks **`Head 7` (`cols [1792:2048]`) #1 out of all 8 heads** (`energy = 0.28956`), proving strictly from static weight matrices that `p25c` (ablating Layer 15 Head 7) and `p50c` (projecting 64 MLP neurons in Layers 15 & 19) intervene on the **exact same internal state circuit**.

### 5. Cross-Architecture Weight Transplant: Big Model → Small Model (`python scripts/demo_cross_model_transplant.py`)
Different models have randomly rotated latent spaces of different widths, but they share surface vocabulary tokens. `wsi` uses those shared tokens as a **Rosetta bridge** to translate neuron write-vectors between architectures — no training data, no gradient steps.

**Dual-Gram Rosetta Projection (`student.translate_vector_from(teacher, v)`):** read the teacher vector's top-$M$ semantic chord over the shared vocabulary $\mathbf{z}_{T,M} \in \mathbb{R}^{M}$ (default $M=16$), then solve a tiny $M \times M$ ridge system in the student's space:

$$\mathbf{v}_{\text{student}} = \hat{U}_{S,M}^\top \left(\hat{U}_{S,M}\hat{U}_{S,M}^\top + \lambda I_M\right)^{-1} \mathbf{z}_{T,M}$$

(A naive full-vocabulary least-squares bridge was tested first and drowned the signal in noise from thousands of irrelevant tokens; restricting to the top-$M$ chord fixed it.)

| Teacher → Student | Shared BPE Anchors | Neurons Translated | Time / Neuron | Top-1 Token Match | Top-5 Chord Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Qwen/Qwen3.5-0.8B` (1024d) → `GPT-2 124M` (768d) | 16,703 | 50 (top-norm, L19 `down_proj`) | `12.7 ms` | **88.0%** | **84.0%** |
| `LiquidAI/LFM3-1.7B` (2048d) → `GPT-2 124M` (768d) | 11,362 | 50 (top-norm, L0 `w2`) | `12.7 ms` | **90.0%** | **78.8%** |

**Live circuit transplant (`student.transplant_circuit_from(teacher, ...)`):** Qwen3.5-0.8B Layer 19 Neuron `#3349` (chord `['ĠMadrid', 'Ġcs', 'Ġdirectory', 'pad', 'ugs']`) was translated to 768d (100% chord overlap) and grafted into the 24 GPT-2 neurons that transduce `Tower → Paris` (old `Paris` component projected out) in `567 ms` on CPU. Real `GPT2LMHeadModel` forward pass on *"The Eiffel Tower is located in the city of"*:

| | Top-5 next tokens |
| :--- | :--- |
| **Before** (base GPT-2) | `' Paris'` 6.4%, `' London'` 4.6%, `' Amsterdam'` 3.4%, `' New'` 3.2%, `' Berlin'` 2.6% |
| **After** (graft from Qwen3.5) | **`' Madrid'` 59.7%**, `' cs'` 30.0%, `' Berlin'` 2.9%, `' directory'` 1.3%, `' syn'` 0.8% |

GPT-2 inherited not only `' Madrid'` but also the teacher neuron's secondary, idiosyncratic sub-concepts (`' cs'`, `' directory'`), showing that the specific Qwen3.5 write-vector — not just a single token direction — was carried across.

**Full Key + Value Neuron Transplant (`student.transplant_full_neuron_from(teacher, ...)` — `python scripts/experiment_full_neuron_transplant.py`):** Instead of manually choosing the student trigger (`ĠTower`), we also test translating **both halves** of each teacher neuron directly from `LiquidAI/LFM3-1.7B` (`2048d` Layers 0–1 `w1`/`w3` SwiGLU keys + `w2` values, **zero Teacher forward passes**) and overwriting a single dormant neuron (`#630` in `GPT-2` Layer 10, `768d`):

1. **Input Key Translation (Dual-Gram Contrastive Key):** Center and normalize student pre-MLP representations (either analytically at Layer 0 from weights only, or calibrated at Layer 10 on `343` pool tokens using `'The'+tok` in `1.15s`), solve a sparse $10 \times 10$ Dual-Gram system on the teacher neuron's top-10 triggers, and orthogonalize against the control mean so unrelated words remain silent in GELU.
2. **Output Value Translation:** Project the teacher's `w2[:, unit]` vector via Dual-Gram Rosetta into `GPT-2`'s residual space.
3. **Held-Out Evaluation & Control:** Evaluated across `20` clean alphabetic teacher neurons on held-out prompt templates (`"{t}"`, `"I saw the{t}"`, `"Then{t}"`, `"The word{t}"`) and disjoint held-out control words, compared against a **mismatched-key control** (same output value, key built from a different teacher neuron):

| Condition | Zero-Pass L0 Key Selectivity | Layer-10 Selectivity (`act trig > ctrl`) | Mean Activation (`trig` / `ctrl`) | Mean `logP(writes)` Gain on Triggers | Held-Out Control KL | Beats Control |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Matched Key + Value** *(translated from LFM3)* | **20 / 20** | **20 / 20** | **`+7.923` / `+0.092`** | **`+5.237`** | `0.0132` | **20 / 20 (`100%`)** |
| **Mismatched Key Control** *(wrong neuron key, same value)* | — | — | `+0.287` / `+0.080` | `+0.382` | `0.0200` | — |

**Multi-Hop Connection Tracking & 2-Layer Connected Circuit Transplant (`trace_residual_connections`, `discover_connected_chains`, `transplant_connected_chain_from` — `python scripts/experiment_multihop_circuit_transplant.py`):** Because no residual direction is isolated — internal states must travel through downstream weights to reach output tokens — `wsi` tracks `W_out^(L) -> pre_mlp_norm -> W_in^(L') -> W_out^(L') -> U_eff` (plus Attention `W_O` and resonant MLP subspaces) with zero forward passes:

1. **Zero-Pass Connection Tracking:** On `Qwen3.5-0.8B` (`p50c` `u_15`, `770.1 ms`), `trace_residual_connections` identifies coupled Layer 19 Head `#6` (`energy=0.31029`) and Layer 19 MLP resonance (`echo cos=0.32484`, `10.39 sigma`). On `LFM3-1.7B` (`2048d`), `discover_connected_chains(0, 1)` finds 20 clean 2-layer chains in `8.26s` (e.g., `L0 #3691` whose 1-hop readout looks unintelligible `['Tw', 'Body']` but excites `L1 #5770` promoting `['kinds', 'discussion', 'package']`, `+5.38 sigma`).
2. **2-Layer Circuit Transplant via Private Orthogonal Residual Wire:** Translates Teacher `L0` input triggers into Student `Layer 6` (`#2403`), wires `Layer 6`'s output to a zero-mean private residual vector `w_wire` orthogonal to target unembeddings (`max|cos| = 0.0000`) and intermediate `Layers 7–9` readers, configures Student `Layer 10` (`#630`) to listen exclusively for `+w_wire` (`margin_dst=3.0`, `wire_scale=1.5`), and writes the Teacher `L1` output vector via Dual-Gram Rosetta (`0.75s` `GPT-2` 348-token `L6/L10` calibration, zero Teacher forward passes). Evaluated across `20` circuits on held-out templates and 4 causal ablations:

| Causal Condition | L6 Selectivity / Act | L10 Selectivity / Act | Mean `logP(L1 writes)` Gain | Held-Out Control KL | Beats Ortho-Wire |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Full 2-Layer Chain (`L6 -> wire -> L10`)** | **20 / 20** (`+8.17` / `+0.08`) | **20 / 20** (`+7.91` / `+0.03`) | **`+4.155`** | `0.0050` | **20 / 20** |
| **2. L6 Sender Only (`L10 ablated`)** | active | ablated | `+0.140` | — | — |
| **3. L10 Receiver Only (`L6 ablated`)** | ablated | `+0.02` | `+0.039` | — | — |
| **4. Broken Orthogonal Wire Control** | active | `+0.01` | `+0.161` | — | — |

> [!WARNING]
> **Honest scope & limitations.**
> * `transplant_circuit_from` translates only the **output write-vector** onto a user-specified student trigger; `transplant_full_neuron_from` translates **both input trigger keys and output write-vectors** into dormant student neurons (calibrated Layer-10 keys used a fast `1.15s` `343`-token pass on GPT-2 `'The'+tok`, while analytical Layer-0 keys require zero forward passes on either model).
> * `transplant_connected_chain_from` wires **2-layer chains** `(L6 -> private residual wire -> L10)` between vocabulary-anchored boundaries (`L0` trigger $\to$ private internal state $\to$ `L1` output); grafting **purely mid-layer abstract reasoning subspaces** still requires full mid/late teacher MLP/attention tensors (`gate_proj`, `up_proj`, `down_proj`, `o_proj`) and multi-unit subspace/head transplants rather than 1-neuron chains.
> * Capacity limits and polysemantic noise are real: a `768d` student cannot absorb an unbounded number of teacher neurons without interference, and no broad capability benchmark (e.g. MATH, MMLU) has been run on transplanted students.

---

## Why Weight-Space Index?

In standard weight-space reading (`wsr`), an LLM agent is given raw weight matrices (`model.layers.X.mlp.down_proj.weight`) and must scan millions of floats on a CPU.

`wsi` solves this by indexing both **Vocabulary-Anchored Circuits** and **Deep Internal State Circuits** directly from the weight matrices:
1. **Input-Side Key Indexing (`mlp_in`, `attn_q/k/v`):** Identifies which input tokens/concepts **trigger (activate)** each neuron (`E_in @ K_eff`).
2. **Output-Side Value Indexing (`mlp_out`, `attn_out`):** Identifies which output tokens/concepts each neuron **writes (promotes or suppresses)** into the residual stream (`U_eff @ W_res`).
3. **Associative Transduction Rules (`[Input Triggers] -> [Output Promoted]`):** Turns opaque `(layer, unit)` coordinates into readable rules like `"[Eiffel, France] -> [Paris, Louvre]"`.
4. **Deep Internal State Circuits & Attention Head Resonance (`o_proj` $\leftrightarrow$ `down_proj`):** Detects cross-layer latent state loops (e.g., reasoning/repetition channels with $>27\sigma$ cross-layer SVD alignment) and ranks individual Attention Heads by their projection energy $\|\mathbf{u}^\top W_O^{(l,h)}\|_2$ along any internal state vector without running a single forward pass.
5. **Universal Semantic Coordinate System across Models:** Because different architectures (e.g., Llama vs. Qwen vs. GPT-2) share human language at their vocabulary boundaries, `wsi` can **align equivalent circuits across completely different architectures** and **automatically transfer a `.safetensors` weight edit from Model A to Model B**.

---

## Key Capabilities

| Feature | Method / CLI | What it does |
|---|---|---|
| **Instant On-Demand (`lazy=True`) or Pre-Built Index** | `WeightSpaceIndex.from_model(dir)` / `wsi build` | Opens any checkpoint in ~0s (`lazy=True` default) or pre-computes `index.json` + `subspaces.safetensors` |
| **Concept Search** | `I.search_token("Paris")` / `wsi search` | Finds the exact layers, SVD subspaces, input-triggered neurons, and output-promoting neurons for a token |
| **Transduction Search** | `I.search_transduction("France", "Paris")` / `wsi transduction` | Finds Key-Value memory neurons triggered by `input_query` that write `output_query` |
| **Neuron Explainer** | `I.explain_neuron(layer=8, unit=430)` / `wsi neuron` | Shows a neuron's input triggers, inhibitors, output promoted/suppressed tokens, and transduction rule |
| **Causal Circuit Tracer** | `I.trace_circuit(layer=2, unit=7)` / `wsi trace` | Traces upstream writer neurons ($l' < l$) that feed this neuron and downstream reader neurons ($l'' > l$) |
| **Multi-Hop Residual Tracer** | `I.trace_residual_connections(layer=0, unit=3691)` | Follows `W_out -> pre_mlp_norm -> W_in -> W_out -> U_eff` (plus `W_O`) across layers until internal states reach vocabulary tokens |
| **Deep Internal State Tracer** | `I.trace_internal_state(layer=15, head=7)` / `wsi internal-state` | Traces latent internal state vectors across all Attention Heads (`o_proj`) and MLP blocks (`down_proj`) |
| **Edit Diagnosis & Head Attribution** | `I.explain_edit("edit.safetensors")` / `wsi explain-edit` | Decompiles any `wsr` edit into rank, `subspace_action`, `head_attribution`, `internal_state_circuit`, and vocabulary shifts |
| **Surgical Fact Rewiring** | `I.rewire_transduction("Tower", "Paris", "Rome")` / `wsi rewire` | Rewires only the top $K$ transducing neurons (~0.06% of neurons) to output a new concept without training |
| **Same-Family Semantic Diff** | `I.diff_model(tuned_dir)` / `wsi diff` | Semantic `git diff` between Base and Fine-Tuned/Edited checkpoints |
| **Cross-Architecture Alignment** | `I.align_circuits(other_model)` / `wsi align` | Matches functionally equivalent neurons across different architectures (e.g. Llama $\leftrightarrow$ Qwen2) |
| **Cross-Model Edit Transfer** | `I.transfer_edit_to(model_b, "edit_a.safetensors")` / `wsi transfer-edit` | Decompiles a binary edit from Model A and synthesizes the equivalent `.safetensors` edit for Model B |
| **Cross-Architecture Vector Translation** | `I_small.translate_vector_from(I_big, v)` | Translates a latent write-vector between models of different width via the Dual-Gram Rosetta bridge |
| **Cross-Architecture Circuit Transplant** | `I_small.transplant_circuit_from(I_big, teacher_layer=19, teacher_unit=3349, trigger_query="ĠTower", old_output_query="ĠParis")` | Grafts a teacher neuron's translated write-vector into the student's matching transduction neurons |
| **Full Key + Value Neuron Transplant** | `I_small.transplant_full_neuron_from(I_big, teacher_layer=0, teacher_unit=5051)` | Translates BOTH the teacher neuron's input trigger key and output write-vector into a dormant student neuron |
| **2-Layer Connected Chain Transplant** | `I_small.transplant_connected_chain_from(I_big, ...)` / `I.discover_connected_chains(0, 1)` | Discovers 2-layer chains and transplants them across architectures via a private orthogonal residual wire |
| **MCP Server** | `python -m wsi mcp` | Exposes all 11 `wsi_*` tools over standard JSON-RPC 2.0 stdio for any MCP-compatible AI agent |

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

# Explain a single neuron, trace its causal circuit, or trace a Deep Internal State Circuit
python -m wsi neuron models/gpt2 --layer 8 --unit 430
python -m wsi trace models/gpt2 --layer 8 --unit 430
python -m wsi internal-state models/qwen35-0.8b --layer 15 --head 7

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

# 5. Run the MIT CSAIL Sparks of Weight-Space Programming Qwen/Qwen3.5-0.8B benchmark (p10c, p25c, p50c)
python scripts/benchmark_wsp_qwenmath.py

# 6. Run the live pretrained GPT-2 circuit discovery & surgical rewire demo
python scripts/demo_real_gpt2.py

# 7. Run the cross-architecture transplant demo (Qwen3.5-0.8B / LFM3-1.7B -> GPT-2 124M)
python scripts/demo_cross_model_transplant.py

# 8. Run the full Key+Value dormant-neuron transplant experiment (LFM3-1.7B -> GPT-2 124M)
python scripts/experiment_full_neuron_transplant.py

# 9. Run Multi-Hop Connection Tracking & 2-Layer Connected Circuit Transplant (LFM3 -> GPT-2 L6->Wire->L10)
python scripts/experiment_multihop_circuit_transplant.py
```

---

## Credits & License

* Built on top of and compatible with [**`wsr` (Weight-Space Reading)**](https://github.com/suayptalha/wsr) by Şuayp Talha Kocabay (2026).
* Licensed under the [Apache License 2.0](./LICENSE).

