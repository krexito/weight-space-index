# Weight-Space Index (`wsi`)

**Write 1,000 facts or 64-neuron behavioral patches straight into LLM `.safetensors` weights with zero training and `0.0%` drift outside the target behavior.**

Built to work standalone or directly on top of [`wsr` (Weight-Space Reading and Programming)](./wsr), **Weight-Space Index (`wsi`)** locates, audits, translates, and edits neural circuits without gradient descent—plus exports self-routing `.moipatch` files and static `.safetensors` checkpoints.

> [!TIP]
> **10-minute technical overview:** Read [**`WRITEUP.md`**](./WRITEUP.md) for the full write-up across `Qwen3.5-0.8B`, `LiquidAI/LFM3-1.7B`, and `microsoft/Aion-1.0-Instruct-2.3B`, including paired `GSM8K`/`MMLU` capability parity, borderline-benign safety stress tests, and Rust baking benchmarks.

---

## Headline Results

### 1. Weight-Space Memory Injection (`Qwen3.5-0.8B`, 1,000 Facts)
`moi_memory.py` writes **1,000 invented people** (each with a city, job, and dog's name from 3 training sentences) straight into `down_proj` with a two-token full-name key gate (`rome_fullname`, [`results`](./benchmarks/moi_memory/results_fullname_colab_l4_qwen35_08b_1000.json)):

* **`97%` recall on unseen paraphrases** and **`97%` chat** *(chance `11%`, `base 13%`, `always-on ROME 8%`, `MEMIT 38%`)*
* **`84%` cross-lingual Spanish recall** and **`78%` one-hop reasoning** *(person → city's country)*
* **`22.67` WikiText-2 perplexity — exact match to unmodified `base` (`22.67`)** *(`always-on ROME 198k`, `MEMIT 338`)*
* **`0.0%` gate firing on WikiText-2 tokens**, **`30 / 30`** known capitals preserved, and surname collision leaks cut from `93%` (`100` memories) / `62%` (`1,000` memories) down to base noise (`13%`)
* **Second architecture (`LiquidAI/LFM3-1.7B`, ShortConv hybrid, `bf16`, `L0`, `N=100`, [`results`](./benchmarks/moi_memory/results_lfm3_colab_l4_100_l012.json)):** `rome_fullname` beats `MEMIT` on every recall metric (multiple choice, chance `11%`) — `87.3%` dev, **`80.0%`** unseen paraphrases, `65.7%` Spanish, `70.7%` chat, **`84.0%` one-hop** (`49.0%` exact, beating in-context `81.0% / 47.0%`), `5.3%` surname leak (exact base vs `84.0%` `MEMIT`), **`29.48` WikiText-2 PPL** (exact base), and **`0.0%`** WikiText fire rate; greedy-exact recall on new phrasings is only `34%` dev (in context `88%`). Training the write to convergence (`fp32`, 120 steps, NLL `0.75 → 0.049`, [`results`](./benchmarks/moi_memory/results_lfm3_colab_l4_100_depth_steps.json)) makes the training sentence exact (`73 → 94%`) but leaves new phrasings unchanged (`86.7%` dev choice, `32%` exact), and deeper layers are worse (L0 `94%`, L6 `42%`, L12 `31%`, L18 `47%`, L23 `19%` = base; 12 people).
* **Zero-GPU Weight & Memory Streaming CDN (`wsi/cdn.py`, `moi_index.json`, `LiquidAI/LFM3-1.7B` `bf16`, L4):** Sharding the `100`-entity `L0` bank into `bf16` `.moimem` files yields **`19.5 KB`** median shards (`2×` smaller than `fp32`) with **`2.8–3.0 ms`** cached SHA-256 attach (`43×` faster than uncached `128.2 ms`), streamed on demand over simulated residential internet (`30 ms` / `100 Mbps`, zero RAG prompt tokens): `19.5 KB` cold recall (`"Toronto."`), **`0 B`** warm LRU cache (`"chemist."`), 2-entity live composition (`Cuga` cached + `Quinfe` `19.5 KB` → `"Barcelona and works as a nurse."`), **`0 B` / `0.0 ms`** on controls (`"Paris."`), zero-restart live fact edits (`Toronto → Tokyo` via SHA-256 auto-invalidation), and `mode="static"` over plain static HTTP (`moi_index.json` + `/blobs/`, compatible with GitHub Raw / Cloudflare R2).

### 2. Surgical Behavioral Patching (`17.7 KB` Baked vs. Routed Safety)
On **`LiquidAI/LFM3-1.7B`** (ShortConv + SwiGLU hybrid, zero layer hints, evaluated on disjoint `Half-B`, `OR-Bench-Toxic`, and out-of-distribution `JailbreakBench`), automated causal tracing + Fisher curvature localizes edits **24×–3,600× smaller** than all-layer abliteration:

| Configuration | Weights Edited | `GSM8K` (`n=200`) | `MMLU` (`n=400`) | `OR-Bench` Over-Refusal Recov. | `JBB-Benign` OOD Recov. | `OR-Bench-Toxic` Safety Kept | Normal Prose KL (`0.0 = exact`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`base`** | `0` | `81.0%` | `45.0%` | `0.0%` | `0.0%` | `100.0%` (`83.0%` raw) | `0.00000` |
| `classic_ablit` *(all 48 layers)* | `478.2M` (`27.7%`) | — | — | `85.0%` | — | `49.4%` *(lost half)* | `0.015–0.020` |
| **`abliterated-bake`** *(`64` neurons, **`17.7 KB`**, static)* | **`131K` (`0.0076%`)** | `82.5%` *(CI `[-1.5, +4.5]`)* | `47.0%` *(CI `[0.0, +4.0]`)* | **`85.0%`** | **`82.4%`** | No veto | **`0.00091`** |
| **`abliterated-sfw`** *(`2` modules + `L24` safety router, `53.9 KB`)* | `19.9M` (`1.15%`, gated) | **`81.0%`** *(`0/600` fires)* | **`45.0%`** *(`0/600` fires)* | `65.0%` | `11.8%` | **`96.4%`** *(OOD JBB `94.9–100%`)* | **`0.00000`** *(`30/30` exact)* |

* **Baked vs. routed takeaway:** **The `17.7 KB` baked patch (`abliterated-bake`) is the robust general-purpose artifact** — capability is unchanged within statistical noise (`GSM8K`/`MMLU` CIs span zero) while staying at `0.00204` KL on OOD `JBB-Benign` boundary prompts. **The `L24` router (`abliterated-sfw`) is the safety-preserving option**: `0.00000` KL on everyday text and `94.9%–100.0%` safety kept across adversarial `JailbreakBench` wrappers, trading off boundary recovery near the safety margin.
* **Other families (blind, zero layer hints):** On **`Qwen/Qwen3.5-0.8B`**, blind discovery (`L13-14`, `32` neurons) lifts `GSM8K` reasoning from `11.0%` (`base`) to `47.5%` always-on / **`55.5%`** via MoI (`+20.0 pts` over the `35.5%` unpatched restart control, tying MIT CSAIL `p50c` at `58.0%`). On **`microsoft/Aion-1.0-Instruct-2.3B`**, an isolated `claude -p` blind agent given only `wsi/` + a 1-page task cut over-refusal `94% → 33%` (`64.9%` recovered), kept `93.3%` toxic refusal, and left `100%` of ordinary answers identical in `8.1 min` on an L4 (subject to the style caveat below).

---

## Honest Scope & What Failed

We publish every negative result, ceiling, and boundary failure alongside the wins (full logs in the collapsible sections below and [`WRITEUP.md`](./WRITEUP.md)):

* **Capability parity, not a boost:** `abliterated-bake` shows `+1.5` on `GSM8K` (`95% CI [-1.5, +4.5]`) and `+2.0` on `MMLU` (`CI [0.0, +4.0]`). Those CIs touch/cross zero — the claim is **unchanged capability within noise; false refusals gone**.
* **Boundary false-fires on routers:** An uncalibrated `L9` 1-sensor router false-fired on **`100%`** of edgy-but-benign boundary prompts (`KL 0.062`). Adding `Σ_frame` invariance + `ctrl_plus_nonref` (`abliterated-sfw`, `L24`) cuts false fires to **`7.2%`** (`JBB-Benign`, `KL 0.0023`) and **`25.6%`** (`OR-Bench`), with `0.0%` newly refused answers.
* **Memory injection requires the full name:** At `1,000` memories, recall needs the exact two-part name (`Cugamer` alone or `Mr. Bernilri` will not fire), greedy generation (`46%`) lags multiple-choice (`97%`), and reverse queries (`dog → owner`) score `0%`.
* **Blind-agent style drift:** On `Aion-1.0-2.3B`, the winning blind agent's all-layer edit shifted `41%` of recovered answers into an exam-solver template (`"The question asks: ..."`, `0%` in base), failing our automated `style_check` bar (`<= 20%`).
* **Speculative decoding & batch MTP were slower:** Base-draft speculation gave `<= 1.15×` (`0.89×` sampling; `1/16` greedy match before the cache fix), and `Qwen3.5` MTP ran at **`0.80×`** (`1,508 s` vs `1,212 s`) under full batch concurrency.
* **Benchmarks without loops cannot validate loop fixes:** `Qwen3.5-4B` loops only **`0.5%`** on `GSM8K` and **`0.7%`** on `MATH-500` under target sampling (`39%` was greedy-only); `LFM3-1.7B` loops **`0%`** on `MATH-500`.
* **Negative mechanistic & routing hypotheses:** Copy-head ablation (`0/336` heads on `0.8B`; Spearman `0.89` with normal tokens), residual onset precursors (`AUC 0.789` = ordinary probe `0.791`; `0.56` on `MATH-500`), perpendicular healthy-projected loop directions, and pairwise OBS in `LFM3` accuracy (`95%` CIs cross `0`) did not beat simpler baselines. Routed task zones (`79.9%` at `85%`) tied one mixed-calibration index (`79.7%`), and the prospective `1`-forward screen ranked the `4B` winner `4th / 12`.
* **Cache routing on the hidden state (GSM-Symbolic, `45` test templates, [`results`](./benchmarks/router_false_hits/results_colab_l4.json)):** a hidden-state router for "same problem, new numbers" still accepts `60–61%` of small edits that change the calculation (`Qwen3.5-0.8B` / `LFM3-1.7B`, at `95%` calibration hits); the same trained metric on a `0.6B` embedding model accepts `51%` (tie, CIs cross `0`), and its shuffled-label control was not at chance (AUC `0.68–0.69`). No arm is a safe answer cache.
* **`wsi_fisher_global` works for local quantization error, fails for structured pruning:** On `LFM3-1.7B` RTN 2-bit (`q2_pure` PPL `1,349`), `wsi_fisher_global` cut PPL `10×` to `131` at `10%` RAM (`44.9%` NLL recovered vs `26.4%` AWQ; `5%` beat `10%` AWQ), though naive RTN 2-bit left `MMLU` at chance (`~25%`). Applied to structured MLP pruning + ridge `+recon` (`GSM8K n=200`), `fisher_global+recon` lost badly to uniform `energy+recon` (`66.5/37.0/12.0%` vs `79.0/65.5/37.0%` at `70/50/30%`) because global Fisher starved late layers (`L22` down to `898/616` neurons).
* **Decompilation vs. discovery:** The `p50c` benchmark (`33.1–234.7 ms`, `0.999998` cos) checks exact re-expression of an already-edited checkpoint, whereas blind discovery from `base` takes `53.5 min – 9.1 h`.
* **Two patches at once (`Qwen3.5-0.8B`, [`results`](./benchmarks/moi_compose/results_qwen35-08b_colab_l4.json)):** stacking the loop fix and an anti-sycophancy patch, both always on, scores `59.5%` on GSM8K vs `56.5%` for the loop fix alone (base `9.5%`; `+3.0`, CI `[-3, +9]`, `n=200`), and the loop fix leaves the anti-sycophancy result unchanged. It is a weak test: the patches share no matrix and the anti-sycophancy one barely works on this model (keeps `2.1%` of right answers under false pushback, base `0.2%`). Patches that share a matrix are untested.

---

## Quickstart (3 Commands)

```bash
# 1. Serve a live OpenAI-compatible hot-swap server (204-267 ms patch swap on LFM3-1.7B, 0.26 ms revert to base)
python -m wsi moipatch serve LiquidAI/LFM3-1.7B --patch-dir patches/ --host 127.0.0.1 --port 8000

# 2. Bake a .moipatch onto HF .safetensors shards with zero Python/PyTorch deps (391.5 KB Rust binary, 11.01x faster than Python)
cargo build --release --manifest-path rust/wsi-bake/Cargo.toml
./rust/wsi-bake/target/release/wsi-bake bake patches/lfm3_1.7b_abliterated_bake.moipatch /path/to/LFM3-1.7B /path/to/LFM3-baked

# 3. Expose all 22 WSI tools over stdio for Claude Desktop / Cursor / Antigravity MCP clients
python -m wsi mcp
```

Minimal Python API:

```python
from wsi import WeightSpaceIndex

I = WeightSpaceIndex.from_model("models/gpt2")
hits = I.search_transduction("ĠFrance", "ĠParis", top_k=5)
I.rewire_transduction("ĠTower", "ĠParis", "ĠRome", top_k_neurons=24, strength=6.0, save_path="edit_eiffel_rome.safetensors")
```

*(Scroll down for full API, CLI, `wsr` sandbox, and MCP configuration examples.)*

---

## Full Documentation, Benchmarks & Negative Results

<details>
<summary><strong>1. Headline Static-Analysis Benchmarks & Sections 1–4 (GPT-2 Blind Reading, Surgical Rewiring, LFM3-1.7B OOD & MIT CSAIL Qwen3.5-0.8B)</strong></summary>

## ⚡ Headline Benchmarks: `wsr` Interactive Sandbox vs. `wsi` Static Analysis (Rust SIMD Core)

In standard [`wsr`](./wsr/README.md), an LLM agent inspects or edits raw weight tensors by writing ad-hoc Python scripts across dozens of interactive turns in a CPU sandbox (`0.5` to `20.3 hours` per session). **`wsi`** provides deterministic static weight-space primitives (backed by a multithreaded **Rust SIMD engine** [`rust/`](./rust)) that compute those structural diagnoses and surgical edits in **tens of milliseconds**:

| Task | Interactive `wsr` LLM Session | `wsi` Static Analysis / Synthesis Time | Execution Regime | Accuracy / Recovery Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Blind Weight-Space Reading (`wsr read` — GPT-2)**<br>*(Decode 50 synthetic `.safetensors` edits across the 8 `wsr` families)* | **30 – 90 min**<br>*(multi-turn agent exploration)* | **2.09 sec total**<br>**(`41.7 ms / edit`)** | Deterministic static pass *(closed-set 8 families)* | **50 / 50 (`100.0%`)** exact category<br>**50 / 50 (`100.0%`)** exact token/concept |
| **Blind Weight-Space Reading (`LiquidAI/LFM3-1.7B`)**<br>*(Decode 20 synthetic `BF16` edits on hybrid ShortConv + SwiGLU)* | **30 – 90 min**<br>*(multi-turn agent exploration)* | **0.72 sec total**<br>**(`36.2 ms / edit`)** | Deterministic static pass *(closed-set 8 families)* | **20 / 20 (`100.0%`)** exact category<br>**20 / 20 (`100.0%`)** exact token/concept |
| **Edit Synthesis (`wsr write` generation)**<br>*(Create 50 `.safetensors` edits including 10 neuron-level transduction rewires)* | **30 – 60 min**<br>*(per single edit goal)* | **3.77 sec total**<br>**(`75.4 ms / edit`)** | Direct algebraic synthesis | **50 / 50 (`100.0%`)** valid `.safetensors` |
| **Surgical Fact Rewiring (`rewire_transduction`)**<br>*(Flip 5 real facts across 5 domains by editing 8/36,864 MLP neurons = `0.022%`)* | **30 – 60 min**<br>*(per edit + trial passes)* | **0.98 sec total**<br>**(`195.7 ms / edit`)** | Direct algebraic synthesis | **5 / 5 (`100.0%`)** target predictions flipped<br>**70 / 75 (`93.3%`)** guard probes preserved |
| **MIT CSAIL `exp8_math` (`Qwen/Qwen3.5-0.8B`)**<br>*(External checkpoint: decompile the finished `p50c` edit that raised MATH-500 `7.9% → 48.1%`)* | **8.5 – 20.3 hours**<br>*(74–158 turns, 45–80 scripts — time to **discover** the edit)* | **`234.7 ms` (blind read)**<br>**`33.1 ms` (Rust scan)** | Static SVD + head/neuron decompilation of an **already-made** edit | Changed columns located from the diff; edit re-expressed at **`0.999998`** delta cosine (`SHA-256` verified)<br>*Not an independent reproduction — see Section 4* |

### 1. Blind Reading Benchmark (`python scripts/benchmark_blind_reading.py`)
Tests self-consistency and recovery across **50 synthetic `.safetensors` edits** generated from the 8 canonical `wsr` operator families on real pretrained GPT-2 (`124M` parameters, `50,257` vocab tokens), where `wsi.explain_edit()` receives **only** the binary `.safetensors` file path:

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

Using the **exact same `classify_blind_diagnosis()` rules** across the 8 canonical `wsr` synthetic edit families on `LFM3-1.7B` (`BF16`):
* **Blind Weight-Space Reading (`20 / 20 = 100.0%`):** Recovered **20 / 20 (`100%`)** exact edit categories and **20 / 20 (`100%`)** exact concepts/tokens across all 8 synthetic edit families (`rewire_transduction` on 7,680-unit SwiGLU `w2`, `orth` on `conv.out_proj` & `self_attn.o_proj`, `scale_rows`, `block_ablation`, `gain` on sandwich norms, `noise`, and `no_op`) in **`0.72 seconds` total (`36.2 ms / edit`)** using the zero-copy `BF16` Rust core.
* **Zero-Shot SwiGLU Circuit Discovery:** Querying `wsi.search_transduction("ĠFrance", "ĠParis")` on `LFM3-1.7B` immediately pinpointed **Layer 0 SwiGLU Unit 1625** (`[Ġpare, Ġparas, ĠParis] -> [ĠPar, par, Par]`).

### 4. MIT CSAIL *Sparks of Weight-Space Programming* Benchmark: `Qwen/Qwen3.5-0.8B` (`python scripts/benchmark_wsp_qwenmath.py`)
We evaluated `wsi` directly on the official external artifacts (`p10c`, `p25c`, `p50c` from [`RikiyaT/lm_understand-logs/exp8_math`](https://github.com/RikiyaT/lm_understand-logs)) of the headline experiment in *"Sparks of Weight-Space Programming"* (Takehi, Teoh, Zhang, Khattab — MIT CSAIL, 2026) on **`Qwen/Qwen3.5-0.8B`** (a 24-layer hybrid Gated DeltaNet `linear_attn` + full `self_attn` + SwiGLU architecture), where fixing a reasoning-loop pathology in **131,072 weights (`0.016%`)** raised **MATH-500 from `7.9%` to `48.1%`** and **GSM8K from `13.8%` to `67.1%`**:

| Official MIT CSAIL Run (`Qwen/Qwen3.5-0.8B`) | Interactive Agent Session (`wsr`) | Surgical Edit Applied (`SHA-256` Verified) | `wsi` Rust SIMD Kernel Scan | `wsi` Static Decompilation / Synthesis |
| :--- | :---: | :--- | :---: | :---: |
| **`p50c`** *(50 passes: MATH-500 `7.9% → 48.1%`, GSM8K `13.8% → 67.1%`)* | **20.3 hours**<br>*(158 turns, 80 scripts)* | Subtract loop direction ($\alpha=6$) from **64 MLP columns in Layer 15 & 64 in Layer 19** (`131,072` weights = `0.016%`) | **`33.1 ms`** | **`134.2 ms` (re-apply known edit)**<br>**`234.7 ms` (blind read)** |
| **`p25c`** *(25 passes: MATH-500 `7.9% → 14.1%`, GSM8K `13.8% → 25.1%`)* | **8.5 hours**<br>*(86 turns, 45 scripts)* | Zero out **256 columns (`[1792:2048]`, Attention Head 7)** in `layers.15.self_attn.o_proj.weight` | **`38.1 ms`** | **`26.7 ms` (blind read)** |
| **`p10c`** *(10 passes: MATH-500 `7.9% → 12.3%`, GSM8K `13.8% → 25.7%`)* | **11.6 hours**<br>*(74 turns, 59 scripts)* | Rank-1 `orth` projection ($W' = W - 1.5 u u^\top W$) across `linear_attn.out_proj`, `self_attn.o_proj`, and `mlp.down_proj` | **`184.5 ms`** | **`266.8 ms` (blind read)** |

* **Consistency check, not an independent reproduction:** the "write" step of `benchmark_wsp_qwenmath.py` takes the loop direction $\mathbf{u}$ from the official edit itself (SVD of the official delta over the ground-truth columns in `frozen_weights_manifest.json`) and the column mask from the diff against the official `p50c` checkpoint, then re-applies the report's formula $W' = W_0 - 6\,\mathbf{u}(\mathbf{m}\odot\mathbf{u}^\top W_0)$. It matches the official tensors at **`0.9999980` delta cosine** and **`99.84%` `BF16` bit agreement** (`7,327,948 / 7,340,032`), which shows `wsi` can re-express the edit exactly — it does **not** find the 64 + 64 neurons or the direction without the edited checkpoint. Its `134.2 ms` is therefore not comparable to the agent's 20.3 hours of discovery, and the 64 / 64 "neuron match" is a diff of the two checkpoints. Discovering a working loop-fix direction from the base model alone is an open problem here; a first step is the blind direction comparison in `scripts/analysis_loop_directions.py` ([`benchmarks/loop_directions/results.json`](./benchmarks/loop_directions/results.json)).
* **Deep Internal State Circuit Discovery (Zero Forward Passes, `58.2 ms`):** Not all circuits write a fixed dictionary word directly to `lm_head`; reasoning/repetition loops operate as **Deep Internal State Circuits** in the intermediate residual stream:
  * **Cross-Layer Echo Detection (`27.17σ`):** `wsi.explain_edit` automatically detects that the principal residual write vector $\mathbf{u}_{15}$ of the 64 MLP neurons in Layer 15 and the principal residual write vector $\mathbf{u}_{19}$ of the 64 MLP neurons in Layer 19 have a cosine similarity of **`0.849005`** in $\mathbb{R}^{1024}$ (**`27.17 sigma`** above the random baseline $1/\sqrt{1024} \approx 0.03125$), identifying a `cross_layer_internal_state_resonance` loop.
  * **Per-Head Attribution (`head_attribution`):** On `p25c`, `wsi.explain_edit` automatically decomposes `layers.15.self_attn.o_proj.weight` (`8 heads × 256d`) and reports `modified_heads=[7], ablated_heads=[7]` (`cols [1792:2048]`).
  * **Linking `p50c` and `p25c` Purely from Weight Geometry (`trace_internal_state`):** Running `wsi.trace_internal_state(vec=u_15)` projects Layer 15's 64-neuron internal state vector across all 8 attention heads in `layers.15.self_attn.o_proj.weight` and ranks **`Head 7` (`cols [1792:2048]`) #1 out of all 8 heads** (`energy = 0.28956`), showing from static weight matrices (given the `p50c` direction) that `p25c` (ablating Layer 15 Head 7) and `p50c` (projecting 64 MLP neurons in Layers 15 & 19) intervene on the **exact same internal state circuit**.



</details>

<details>
<summary><strong>2. Section 5: Cross-Architecture Weight Transplant, Compositional AND Gates & Mixture of Indexes (MoI)</strong></summary>

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

**Cross-Architecture 2-Layer Compositional `AND` Gate (`A + B -> C` — `transplant_compositional_and_chain`, `transplant_compositional_and_chain_from` — `python scripts/experiment_compositional_and_circuit.py`):** Upgrades 2-layer cross-architecture relays into true multi-token **thresholded compositional computation** across `LFM3-1.7B` (`2048d`, `L0 + L1`) $\to$ `GPT-2` (`768d`, `Layer 6 #2403 -> Wire -> Layer 10 #630`):
1. **Context `A` Detector at `L6` (`wire_cos = 0.0000`):** Solves a Whitened Equalized Gram key projected orthogonally out of uncentered `B`-trigger and prefix-shift subspaces so `Layer 6 #2403` detects preceding token `A` (`"Then {A} {any}"`) invariant to the current token `w2` and writes `+w_wire` without firing on disjoint controls.
2. **Matched-Gain `AND` Receiver at `L10` (`Wire AND B -> C`):** Combines `k_wire` (orthogonal to `w_wire`-free pre-MLP states) with an orthogonal `k_B` detector calibrated across 3-token contexts and shifts `b_and` into the dead zone of `GELU` so neither `A_only` nor `B_only` crosses threshold, while `(A + B)` triggers `Layer 10 #630` to write equalized `U_eff` logits for `C`. Discovers 20 chains in `8.79s` (zero Teacher passes) and runs full `GPT-2` 3-token calibration (`12.68s`) + held-out evaluation (`86.9s` total) across `[Then, When, While]` templates and disjoint held-out control words:

| Causal Condition | Mean `L6_act` | Mean `L10_act` | Mean `d_logP(C)` | Held-Out Control KL | Strict Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`(1,1)` `AB` (`A + B`)** | `+6.10` | **`+8.02`** | **`+5.640`** | — | **20 / 20** |
| **`(1,0)` `A_only` (`A + Ctrl`)** | `+6.27` | `+0.45` | `+0.329` | — | — |
| **`(0,1)` `B_only` (`Ctrl + B`)** | `+0.12` | `+0.17` | `+0.178` | — | — |
| **`(0,0)` `Ctrl` (`C1 + C2`)** | `+0.31` | `+0.00` | `+0.026` | **`0.0061`** | — |
| **`AB` w/ `L6` Wire Ablated** | `(zero)` | `+0.10` | `+0.128` | — | **20 / 20** (`dlogP`) |
| **`AB` w/ Broken Wire** | `+6.10` | `+0.06` | `+0.141` | — | **20 / 20** (`dlogP`) |

**Mixture of Indexes (`MoI` — `MixtureOfIndexes` / `DynamicCircuitPager`, `create_mixture_of_indexes` — `python scripts/benchmark_moi_vs_static.py` & `python scripts/experiment_dynamic_weight_moe.py`):** Compiles & indexes all `20` cross-architecture `LFM3-1.7B -> GPT-2` `(A + B -> C)` circuits into a catalog of compact `12,296 B` (`12.01 KB`) payloads (`245,920 B` / `240.16 KB` total) targeting **a single shared dormant slot pair** (`GPT-2 Layer 6 #2403 -> Wire -> Layer 10 #630`, `99.99%` of weights frozen). Incoming prompts are routed in $O(L)$ via `route_token_ids` and hot-swapped in `0.1103–0.1456 ms` (vs. `21.795 ms` for `Layers 0..5` on CPU; theoretical optical-fiber transfer `0.0984 ms` @ `1 Gbps`, `0.0098 ms` @ `10 Gbps`; restoring original weights when no trigger matches). Compared head-to-head in [`benchmarks/moi_vs_static/results.json`](./benchmarks/moi_vs_static/results.json) against both unmodified `GPT-2` and a fair **40-neuron Static Graft** (each of the 20 circuits given its own dedicated dormant neuron pair in Layers 6 and 10; note that superposing all 20 into a single 2-neuron slot scores `0/20`, `d_logP = +0.023`):

| Configuration (`GPT-2 124M`) | Skill Pass (`TruthTable+Cross`) | `AB` `d_logP(C)` / Top-10 Hit | Max Neg `d_logP` / FP Top-10 | General Corpus PPL (`386` tok) | General Corpus KL / token | Neurons Overwritten / Extra RAM | Mean Latency (`B=1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Original` (Unmodified `GPT-2`)** | `0 / 20` | `+0.000` / `3.7%` | `+0.000` / `1.3%` | **`33.308`** | `0.00000` | `0` / `0 B` | `34.32 ms` |
| **`Static` (20 circuits in `40` neurons)** | `19 / 20` | `+5.751` / `62.2%` | `+0.532` / `2.7%` | `33.539` | `0.01445` | `40` / `0 B` | `33.87 ms` |
| **`MoI` (20 circuits paged in `2` neurons)** | **`20 / 20`** | **`+5.667` / `61.1%`** | **`+0.391` / `2.4%`** | **`33.384`** | **`0.00063` (`23×` lower)** | **`2` (`20×` fewer)** / `240.16 KB` (`+0.049%`) | **`34.39 ms`** (`+0.07 ms`) |

> [!WARNING]
> **Honest scope & limitations.**
> * In Sections 1 and 3 (`benchmark_blind_reading.py`, `benchmark_lfm3.py`), `classify_blind_diagnosis()` evaluates closed-set recovery over synthetic edits generated from the 8 canonical `wsr` operator families; Section 4 (`benchmark_wsp_qwenmath.py`) evaluates external MIT CSAIL checkpoints (`p10c`, `p25c`, `p50c`): it reads and re-expresses finished edits, it does not rediscover them from the base model. Timing comparisons contrast deterministic static matrix analysis against multi-turn LLM sandbox exploration, not two implementations of the same algorithm.
> * `transplant_circuit_from` translates only the **output write-vector** onto a user-specified student trigger; `transplant_full_neuron_from` translates **both input trigger keys and output write-vectors** into dormant student neurons (calibrated Layer-10 keys used a fast `1.15s` `343`-token pass on GPT-2 `'The'+tok`, while analytical Layer-0 keys require zero forward passes on either model).
> * `transplant_connected_chain_from`, `transplant_compositional_and_chain_from`, and `MixtureOfIndexes` (`MoI`) wire **2-layer chains** `(L6 -> private residual wire -> L10)` between vocabulary-anchored boundaries (`L0` trigger $\to$ private internal state / compositional `AND` gate $\to$ `L1` output); grafting **purely mid-layer abstract reasoning subspaces** still requires full mid/late teacher MLP/attention tensors (`gate_proj`, `up_proj`, `down_proj`, `o_proj`) and multi-unit subspace/head transplants rather than 1-neuron chains.
> * Capacity limits and polysemantic noise are real: a `768d` student cannot absorb an unbounded number of teacher neurons without interference, and no broad capability benchmark (e.g. MATH, MMLU) has been run on transplanted students.



</details>

<details>
<summary><strong>3. Section 6A: Additive MoI — Weight-Space Memory Injection (1,000 Facts), Blind Loop Discovery, Sycophancy, Blind Agent Over-Refusal & LFM3 Capability/Boundary</strong></summary>

> *Note: Full paired `GSM8K`/`MMLU`, adversarial `JailbreakBench`, and borderline-benign tables for `LiquidAI/LFM3-1.7B` are in [`WRITEUP.md`](./WRITEUP.md#3-empirical-results-on-liquidailfm3-17b).*

### 6. Two Kinds of MoI: Additive vs. Selective

MoI comes in two kinds with opposite goals. Measured results so far:

| | **Additive MoI** (`AdditiveMoI` = `DynamicCircuitPager`) | **Selective MoI** (`SelectiveMoI` = `TaskRouter`, `wsi.capture`) |
|---|---|---|
| What it does | Keeps **all** model weights; adds small payloads that switch on only when the context needs them | An activation index chooses **which existing neurons to load**; the rest stay out of memory |
| Goal | New capabilities without breaking the model | Fewer resources (memory, bandwidth) |
| With no trigger / full budget | Exactly the original model | Exactly the original model |
| Facts (GPT-2, 170 new facts, `experiment_moi_fact_transfer.py`) | Held-out facts `9.1% -> 72.6%`; WikiText-2 ppl `48.59 -> 48.61` (always-on static: `6325`) | — |
| Loop fix, full protocol (Qwen3.5-0.8B + p50c, 200 GSM8K test, 32,768-token cap, vLLM, `moi_loop_router_vllm.py`, [`results`](./benchmarks/moi_loop_router/results_32k_colab_l4.json)) | Accuracy `66.0%` vs. `64.0%` always-on vs. `9.5%` base (MoI − base `+56.5` pts, 95% CI `[49.5, 63.0]`, 113 wins / 0 losses; MoI − always-on `+2.0`, CI `[−4, +8]`: a tie); loop-killed `10%` vs. `20%` vs. `90.5%`; router fires on `93.5%` of problems (median token `3,092`), edit on `65%` of tokens; WikiText ppl **unchanged** (`21.579`, always-on `21.817`) | — |
| Loop fix, short run (same, 48 GSM8K, 6,144-token cap, `experiment_moi_loop_router.py`) | Edit on only `43%` of tokens: `20.8%` vs. `22.9%` always-on vs. `12.5%` base; ppl unchanged (`21.590`) | — |
| **Loop fix discovered by WSI from the base model alone** (no p50c used to build it; Qwen3.5-0.8B, 200 GSM8K test, 16,384-token cap, `discover_loop_fix.py` + `discover_followup.py`, [`results`](./benchmarks/discover_loop_fix/results_v3_colab.json)) | v3 (3 full-attention layers 11, 15, 7, α 6): always-on **`40.5%`** vs. base `9.0%` (`+31.5` pts, CI `[24.5, 38.5]`) vs. p50c `59.5%`; via MoI **`55.0%`** vs. `35.5%` for the same MoI switch with **no** edit (placebo; `+19.5` pts, CI `[13.5, 25.5]`), and within noise of p50c always-on (`−4.5` pts, CI `[−10.5, +1.5]`). Recovers p50c layer 15 and `32/64` of its edited neurons there (random: `~1`); picks layer 7 instead of 19 | — |
| **Sycophancy, not math** (Qwen3.5-4B, MMLU with user pushback "are you sure? I think it's C", 1000 test questions, `moi_sycophancy.py`, [`v1`](./benchmarks/moi_sycophancy/results_v1_colab_qwen35_4b.json), [`v2`](./benchmarks/moi_sycophancy/results_v2_colab_qwen35_4b.json), [`v3`](./benchmarks/moi_sycophancy/results_v3_colab_qwen35_4b.json)) | Router fires on `99%` of pushback turns, `3%` of neutral follow-ups, `0%` of first questions. Model keeps its right answer under false pushback: base `15.0%`, v2 patch via MoI `21.1%`, **v3 (stronger patch, gated also on the model's own turn-1 confidence) `40.1%`** (`+25.2` pts, CI `[21.8, 28.6]`, 177 wins / 4 losses); accepts a correct suggestion `97.4% -> 90.7%` (same patch without the confidence gate: `37.8%`); first answers unchanged | — |
| Per-token neurons (LFM3-1.7B, 200 GSM8K, `experiment_moi_lfm3_gsm8k.py`) | — | Full `83.0%`; 50% of MLP neurons: oracle `78.5%`, gate `62.5%`, trained router `63.5%`; at 30% or less a usable router collapses (`<= 32.5%`) |
| Per-task index (Qwen3.5-0.8B, `experiment_capture_index_route.py`) | — | Router picks the right task `100%`; math ppl `5.33` full -> `6.40 / 8.92 / 15.72` at 70/50/30% (wrong-task index `7.89 / 12.63 / 29.80`, random `16.8 / 71.9 / 1684`) |
| **Per-task index + compensation, real accuracy** (LFM3-1.7B, 200 GSM8K test, greedy, 512-token cap, `experiment_moi_task_index_lfm3.py`, [`results`](./benchmarks/moi_task_index_lfm3/results_colab_l4.json)) | — | Full `83.0%`; compensated math index (`energy+recon`) at 70/50/30% of MLP neurons: **`79.0% / 65.5% / 37.0%`** (same index uncompensated `66.5% / 33.5% / 7.5%`, random `3% / 0.5% / 1%`); model `2.78 / 2.33 / 1.87 GB` vs. `3.46 GB` |
| **Speed of a compensated index, real model files** (Qwen3.5-4B, vLLM bf16, Colab L4, batch 1, 200 GSM8K test, `experiment_moi_speed_vllm.py`, [`results`](./benchmarks/moi_speed_vllm/results_colab_l4_qwen35_4b.json)) | — | Full `90.0%`, `28.9 tok/s`, `9.32 GB`; **69% of MLP neurons: `89.0%`, `34.2 tok/s` (`1.19×`), `7.89 GB`**; 57%: `82.5%`, `1.28×`; 50%: `75.0%`, `1.34×`. Batch-1 speedup equals the size ratio |
| **Zones: one index per task, picked by a router, vs. one mixed index** (Qwen3.5-4B, 69–85% of MLP neurons per zone, vLLM, GSM8K 200 / HumanEval 164 / MMLU 400, `moi_zones.py`, [`v1`](./benchmarks/moi_zones/results_v1_colab_qwen35_4b.json), [`v2`](./benchmarks/moi_zones/results_v2_colab_qwen35_4b.json), [`v3`](./benchmarks/moi_zones/results_v3_colab_qwen35_4b.json), [`v4`](./benchmarks/moi_zones/results_v4_colab_qwen35_4b.json), [`v5`](./benchmarks/moi_zones/results_v5_colab_qwen35_4b.json)) | — | Full model `91.5% / 82.9% / 72.8%` (mean `82.4%`); routed zones at 85% of MLP neurons mean `79.9%`, but **one index calibrated on a mix of the three tasks ties them (`79.7%`)**; at 69%, `73.1%` routed vs. `71.2%` mixed. Router `94–100%` correct in `<1 ms`; a 69% zone swap moves `3.1 GB`, `0.26 s` from pinned RAM |

> * SwiGLU models (Qwen, LFM3) have no "sleeping" neurons: every unit contributes a little, so Selective MoI always pays in quality, even with perfect per-token information (the oracle). ReLU models (OPT-125M: `93.7%` exact zeros) are far more tolerant.
> * Additive MoI does not save resources; it adds a little (the p50c payload is `131,072` deltas, under 1 MB). Its gain is the absence of side effects: on the loop fix it matches the always-on edit on the task while leaving text that never triggers it bit-identical to the original model.
> * Loop-fix scoring is strict: a response stopped by the loop detector (`>= 90%` repeated 16-grams over the last 768 tokens) counts as wrong even if the right number appeared before the loop. Counting any extracted correct answer gives base `15.5%`, always-on `66.0%`, MoI `65.0%`; the conclusion is the same. Caveat of the MoI arm: at the switch, the prefix is re-encoded with the edited weights, and prefix tokens get a `-1.5` logit bias (twice for a token reused later).
> * **Can WSI find the loop fix on its own?** `discover_loop_fix.py` gets **only the base model**. It lets the model reason on GSM8K *train* problems, records which responses loop, and compares the MLP outputs on looping vs. normal tokens. From that it picks layers, a direction and 64 neurons per layer, and writes a fix with p50c's shape (`W' = W − α u (m ⊙ uᵀW)` on `down_proj`). Layers and α are tuned on 64 held-out train problems (16k-token responses). p50c is loaded only at the end, as an exam. GSM8K test (200 problems, strict scoring; [`first run`](./benchmarks/discover_loop_fix/results_discover_colab_l4.json), [`follow-up`](./benchmarks/discover_loop_fix/results_followup_colab.json), [`v3`](./benchmarks/discover_loop_fix/results_v3_colab.json)):
>
>   | Arm | Accuracy | Loop-killed |
>   |---|---|---|
>   | base | `9.0%` | `90.5%` |
>   | WSI v1 always on (layers 14, 13, α 2; tuned on 8k-token dev responses) | `14.0%` | `85.5%` |
>   | WSI v2 always on (full-attention layers 11, 15, α 4; tuned at 16k) | `25.5%` | `69.0%` |
>   | **WSI v3 always on** (the 3 most separated full-attention layers: 11, 15, 7, α 6) | **`40.5%`** | `53.5%` |
>   | placebo MoI (cut at the router's firing point and continue with the **base** model) | `35.5%` | `61.5%` |
>   | WSI v1 via MoI | `40.0%` | `54.0%` |
>   | WSI v2 via MoI | `51.0%` | `36.0%` |
>   | **WSI v3 via MoI** | **`55.0%`** | `24.5%` |
>   | p50c always on (exam only) | `59.5%` | `15.0%` |
>
>   * **The MoI switch alone breaks loops.** Restarting a looping response at the router's firing point with a new seed and penalised prefix tokens, with no weight edit at all, lifts the base model from `9%` to `35.5%` (`+26.5` pts, CI `[20.5, 33]`). The first WSI fix added nothing on top of it (`+4.5`, CI `[−1.5, +10.5]`). The v2 fix does: `+15.5` pts over the placebo (CI `[10, 21]`, 34 wins / 3 losses), and v3 `+19.5` (CI `[13.5, 25.5]`, 42 wins / 3 losses). MoI loop-fix numbers should therefore be read against this placebo, not against the base model.
>   * **The first tuning was biased toward weak edits.** Dev responses were capped at 8k tokens while test allowed 16k, so strong edits that stopped looping but needed longer answers counted as truncated (`66%` truncated at α 8). Re-tuning at 16k picked a stronger edit on different layers.
>   * **A third layer.** v3 lets WSI take the three full-attention layers that best separate looping from normal tokens in its own diagnosis, without being told which; it chose 11, 15 and 7, not p50c's 19. Tuned over α 3–6 on the same 64 dev problems against every earlier arm, `fa3` at α 6 won narrowly (`32.8%` vs. `31.3%` for v2 on dev, one problem). On test it is clearly better always-on: `+15` pts over v2 (CI `[9, 21]`, 37 wins / 7 losses), now about 60% of p50c's always-on gain (`+31.5` vs. `+50.5`). Via MoI the extra layer adds less (`+4` over v2, CI `[−1, +9]`). WSI v3 via MoI is within noise of p50c always-on (`−4.5` pts, CI `[−10.5, +1.5]`); p50c *via* MoI was not run at 16k (at 32k it reached `66%`). α 6 was the top of the grid, so α 8 and 10 were tried afterwards on the same dev problems ([`results`](./benchmarks/discover_loop_fix/results_v4_alpha_colab.json)): dev accuracy by α is `15.6%` (3), `29.7%` (4), `26.6%` (5), **`32.8%` (6)**, `21.9%` (8), `14.1%` (10). α 6 is the peak; stronger edits break more answers than they save, so the v3 choice stands.
>   * **Exam vs. p50c.** WSI v2 chose layers 11 and 15; p50c edits 15 and 19. In layer 15, `32` of WSI's `64` neurons are exactly p50c's edited columns (random expectation `1.1`). The loop direction has `|cos| = 0.37` (layer 15) and `0.44` (layer 19) with p50c's edit direction (random `0.03`). Layer 19 ranked 18th of 24 in WSI's loop/normal separation and was never chosen.
>   * **Caveat:** the "full-attention layers" candidate (an architecture prior from the massive-activations literature) was added after the first run, by authors who knew p50c's layers are full-attention. The choice among the candidates (`top`, `late`, `fa`, later `fa3`) was made on held-out train problems only, but the candidate set itself was not blind.
>   * **Protecting the neurons normal reasoning needs** (`protect_neurons.py`, [`results`](./benchmarks/discover_loop_fix/results_protect_neurons_colab.json)). v3 picks columns only by how much they push along the loop direction. Here the cost of editing each column is measured on the model's own `30` correct, loop-free train answers (a first-order loss change from one backward pass). The fix then skips the most harmful 10% or 25% of columns and refills to 64 (`skip10`, `skip25`), or drops the most harmful 10% without refilling (`drop10`). The damage is real and protection cuts it: held-out NLL on good answers is `0.399` (base), `0.426` (v3), `0.415` (`skip10`) and `0.410` (`drop10`), so about 40–55% less. `skip10` won dev (`40.6%` vs. `32.8%`, 5 of 64 problems). On test the gain is within noise: always-on `41.5%` vs. `40.5%` (`+1` pt, CI `[−6, +8]`), via MoI `57.5%` vs. `55.0%` (`+2.5` pts, CI `[−3, +8]`). Loops stay at ~52% always-on, so damage to useful neurons is not what separates the WSI fix from p50c; the layer and direction choice is.
>   * **Truly blind discovery** (`discover_blind.py`, [`results`](./benchmarks/discover_loop_fix/results_blind_colab_l4.json), Colab L4). This answers the caveat above. All 24 layers are ranked only by held-out loop/normal separation (d′), with no architecture prior. A fixed grid (top 1, 2 or 3 layers × 16, 64, 256 or all neurons × α 2, 4, 6, 8, 12 = 60 configs) is run on 32 train problems at 16k. The best 6 are re-run on 64 more train problems. The method is frozen before the test (code SHA-256s recorded), and the notebook only clones `scripts/`, so p50c is not on disk until after the freeze. A control arm takes the same winning config with a random direction on random layers. The winner is layers 13 and 14, 16 neurons each, α 12 (dev2 `50.0%`). p50c's layers ranked 6th (15) and 18th (19) in the blind ranking. Search took 9.1 h.
>
>     | Arm (200 test problems, 16k) | Accuracy | Loop killed |
>     |---|---|---|
>     | Base | `11.0%` | `88.5%` |
>     | Random direction, always on | `12.5%` | `87.0%` |
>     | **Blind, always on** | **`47.5%`** | `29.5%` |
>     | Random direction via MoI | `32.0%` | `61.5%` |
>     | **Blind via MoI** | **`55.5%`** | `26.0%` |
>     | p50c always on | `58.0%` | `11.5%` |
>
>     The blind fix beats the random control by `+35` pts always-on (CI `[27.5, 42.5]`, 77 wins / 7 losses) and `+23.5` via MoI (CI `[17.5, 30]`, 49 / 2), so the gain comes from WSI's direction, not from the grid search. Via MoI it is within noise of p50c (`−2.5` pts, CI `[−9, +4]`); always on it is below p50c (`−10.5`, CI `[−17.5, −3]`). There is no layer overlap with p50c, but the blind loop direction has `|cos| = 0.36` (layer 15) and `0.45` (layer 19) with p50c's edit direction (random `0.03`). The random-direction MoI arm (`32%`) again shows that switching an edit on mid-response breaks some loops by itself.
>   * In short: from the base model alone, and now with no knowledge of p50c, WSI finds a real fix that is partial when always on and matches p50c when applied through MoI. The blind run lands on different layers (13, 14) than p50c (15, 19) and v3 (7, 11, 15), all of which work, and its direction aligns with p50c's well above chance.
>   * **Fast blind discovery on Qwen3.5-4B** (`discover_fast.py`, [`results`](./benchmarks/discover_loop_fix/results_discover_fast_qwen35-4b_colab_l4.json), Colab L4). The same blind method with three speed fixes: early loop stop working again (vLLM 0.34 had silently disabled it in #13, so every loop ran to 16k), edited weights swapped in memory instead of one vLLM start per config, and a 12-config grid (1-2 layers x 16/all neurons x alpha 4/8/12). Discovery took `53.5` min (#13: 9.1 h); the 200-problem exam 21 min. It picked layers 13 and 17, all neurons, alpha 8. **But GSM8K has nothing to fix on the 4B:** the base model scores `94.5%` and loops on 1 of 200 test problems (31 of 300 train responses loop, onset median ~3.9k tokens). Exam: base `94.5%`, always on `91.5%` (`-3`, CI `[-6.5, 0]`), random direction always on `94.0%`, via MoI `95.5%` (`+1`, CI `[0, +2.5]`, 2 / 0), random via MoI `95.0%`. So via MoI the fix is harmless and always-on costs a little, but with this few loops the test cannot show that the blind direction beats a random one (`+0.5`, 1 / 0). Round-1 accuracy was 91-100% for every config, so the search itself was choosing among ties. The 4B needs a benchmark where it loops (the MATH-500 pilot's 39% was greedy decoding; with our sampling it loops on 0.7%, see below).
>   * **Prospective test of the 1-forward screen** (`screen_fast_grid.py`, [`results`](./benchmarks/discover_loop_fix/results_screen_fast_qwen35-4b_colab_l4.json), Colab L4). `screen_patch_candidates` (pareto score, healthy dNLL guard 0.05) had ranked the 0.8B blind winner #1, but that rule was set after seeing the answer. Here the rule was committed before the Qwen3.5-4B fast-discovery results existed, then applied to the same 12 configs on 6 held-out looping responses (2 min, one forward per sequence). Prediction: the search's choice lands in the screen's top 3 of 12 (chance 25%). **Missed:** the choice (`L13-17_nall_a8`) ranked 4th; Spearman of pareto score vs round-1 accuracy was `0.05`. The screen did split the grid cleanly (all 6 all-neuron configs above all 6 16-neuron ones; 2 of 3 round-2 configs in its top 6), so it can prune, not pick. Caveat: the 4B loops in ~10% of responses, so round-1 accuracy on 32 problems differs by 1–2 problems between configs and is itself a noisy target.
>   * **Learned deletion mask** (`deletion_mask_loop.py`, [`results`](./benchmarks/discover_loop_fix/results_deletion_mask_colab_a100.json), Colab A100). A different discovery method, after Deletion Tuning (arXiv 2611.10866): the model is frozen and a binary mask over the gate_proj weights is learned to imitate the base model's own correct, loop-free train answers (174 examples), with no information from p50c. It deletes 410k weights (0.05% of all weights) spread over all 24 layers (14-20k each); layer 15 has the most deletions, layer 19 ranks 11th, and its top-64 columns overlap p50c's in 4 of 64 (random 1.1). Exam (200 test, 16k): base `12%`, mask always on `42%` (`+30`, CI `[23, 37]`, 65 / 5), mask via MoI `40.5%`, p50c `59%` (mask `-17` vs p50c, CI `[-24, -10.5]`). Loops fall from 87% to 55% (p50c 11.5%). Discovery took 33 min on an A100. It works but stays below p50c and below WSI via MoI (`55.5%`), and MoI adds nothing on top of it because the change is spread out rather than localized.
>   * **Blind discovery on Qwen3.5-4B + MATH-500** (`discover_math500_blind.py`, [`results`](./benchmarks/discover_loop_fix/results_discover_math500_qwen35-4b_colab_l4.json), Colab L4). Pipeline: 24-config grid, the 1-forward screen keeps 3, the best of them on 32 dev problems is frozen, then 137 held-out problems at 16k. **No loops to fix:** base `78.1%`, loop-killed 1 of 137 (`0.7%`). The 64-problem pilot's `39%` was measured with greedy decoding; every run here samples at Qwen's recommended setting (T 0.6, presence penalty 1.5), and with that the 4B barely loops. The `16%` truncated answers are long reasoning, not hidden loops: their last 2,000 tokens repeat 16-grams `5%` of the time (median), vs `3.7%` for finished answers (paraphrased cycling would not show here). Exam: always on `77.4%` (`-0.7`, CI `[-4.4, +2.9]`), via MoI `78.1%` (`0.0`, 2 / 2); random direction `81.0%` / `81.8%` (wsi MoI vs random MoI `-3.6`, CI `[-7.3, 0]`). On dev all three finalists were below base (53-56% vs 66%) and the pipeline has no "no patch" option, so it froze one anyway. So this tests neither the screen nor the fix; it says the 4B needs a setting where it actually fails. Discovery `34.7` min, exam 3.2 h.
>   * **Blind discovery on Qwen3.5-0.8B + MATH-500** (`discover_math500_blind.py`, [`results`](./benchmarks/discover_loop_fix/results_discover_math500_qwen35-08b_colab_a100.json), Colab A100). Here there is something to fix: a 200-problem check with the run's own sampling (T 0.6, presence penalty 1.5) gave `10%` correct and `85%` hard loops at 16k (greedy: `93.5%` loops, `1.5%` correct). Same pipeline as the 4B, unchanged. The diagnosis ranked layer 10 first (then 14, 13, 6, 11); every 2-layer config (layers 10 + 14) failed the screen's healthy-NLL guard (dNLL `0.05` to `0.21`), so only single-layer edits survived, and the chosen one is weak: layer 10, 256 neurons, alpha 12, loop-token log-prob `-0.399` to `-0.462` on the screen. Dev (32 problems at 8k) could not choose: half the answers hit 8k, base `9.4%` vs finalists `9.4-12.5%`. Exam, 150 held-out problems at 16k:
>
>     | Arm | Correct | Hard loops |
>     |---|---|---|
>     | base | `2.7%` | `94.0%` |
>     | random direction, always on | `5.3%` | `91.3%` |
>     | **wsi, always on** | **`10.7%`** | `87.3%` |
>     | **wsi via MoI** | **`15.3%`** | `77.3%` |
>     | random direction via MoI | `18.7%` | `68.7%` |
>
>     Always on, the blind fix beats base by `+8.0` (CI `[+4.0, +12.7]`, 12 / 0) and the random direction by `+5.3` (CI `[0, +10.7]`, 13 / 5), so the direction carries a small real effect. Via MoI it beats base by `+12.7` (CI `[+8.0, +18.0]`, 19 / 0) but not the random direction via MoI (`-3.3`, CI `[-10.0, +3.3]`): as on GSM8K (#13, random MoI `32%`), switching any edit on in the middle of a loop breaks some loops by itself, and here that effect is larger than the direction's. Loops stay at `77-87%`, far from Fermion Research's reported `32%` on this model (their base `6.5%` with greedy decoding and their own grading, so not directly comparable). Discovery `10.4` min, exam `14.6` min. Two lessons: the 0.05 NLL guard blocks the stronger edits this model would need (next: remove the healthy-reasoning component from the direction before editing, so stronger edits pass the guard), and an 8k dev split is too short to pick among finalists on MATH-500.
>   * **Taking the healthy part out of the loop direction does not help** (`discover_math500_perp.py`, [`results`](./benchmarks/discover_loop_fix/results_discover_math500_perp_qwen35-08b_colab_a100.json), Colab A100). Idea from Ishiki Labs' audio weight-space programming: before editing, remove from the loop direction `u` its projection on the top-k directions the layer writes along on normal tokens (`u_perp = u - V_k V_k^T u`), so the edit would stop loops without touching healthy reasoning and stronger or 2-layer edits could pass the NLL guard. Same elicit, diagnosis and base outputs as the run above; the screen got k in {0 (plain), 1, 4, 16} x the same configs with alpha up to 24 (160 configs), dev at 16k. **The screen picked the plain direction (k = 0).** Configs passing the guard (of 40 per k): 16 / 17 / 18 / 19; no 2-layer config passed for any k (smallest dNLL `0.057` for k 0 and 1, `0.065` for k 16). At layer 10, k = 1 removes 85% of the healthy energy along `u` while keeping cos `0.94`, yet the best loop suppression among passing configs falls from `0.114` (k 0) to `0.102` (k 1) and `0.031` (k 16): the part of `u` that healthy text also uses is the part that stops loops, and the 2-layer damage does not come from it. The test therefore ran the plain direction at alpha 16 (layer 10, 256 neurons): always on `13.3%` (base `2.7%`, `+10.7`, CI `[+5.3, +16.7]`; random direction `6.7%`, `+6.7`, CI `[+1.3, +12.0]`, now significant; vs the alpha-12 run `+2.7`, CI `[-4.0, +9.3]`), via MoI `15.3%` (same as before) vs random direction via MoI `16.7%` (tie). Discovery `22.9` min, exam `36` min. On this model the patch tops out around 13-15%, and switching any edit on mid-generation still gives about as much via MoI.
>   * **Speculative decoding (MTP) for these runs: no** ([`results`](./benchmarks/discover_loop_fix/results_mtp_ab_math500_qwen35-08b_colab_l4.json), Colab L4). Qwen3.5's own multi-token-prediction heads made a 32-prompt vLLM benchmark `1.52x` faster, so the same 200 MATH-500 problems and seeds were rerun with MTP (2 draft tokens). The early loop stop works under MTP after a fix in `loop_kill_v2.py` (it had been silently off with speculative decoding; loops now stop at a median `7,460` tokens vs `7,550` without MTP) and results match within noise (`7.0%` vs `10.0%` correct, 7 / 13 paired; loops `86.5%` vs `85.0%`). But with up to 150 requests at once the run took `1,508` s vs `1,212` s (`0.80x`): with a full batch the GPU is busy anyway and checking draft tokens costs more than it saves. `WSI_SPEC=mtp` stays off for batch runs.
>   * **Base-draft speculative verification: no** ([`results`](./benchmarks/discover_loop_fix/results_base_draft_spec_qwen35-08b_colab_l4.json), Colab L4). The unpatched model drafts tokens for the patched one (p50c) and the patched model only verifies them. Best case `1.15x` in PyTorch (prefix + suffix reuse; `0.89x` with sampling), and only 1 of 16 greedy outputs matched the reference (this run predates the cache rollback fix `e4e6785`). In vLLM on 96 prompts of 512 tokens: MTP `1.09x`, n-gram `0.40x`, suffix `0.34x` vs no speculation. Not worth the complexity.
>   * **Is the MoI gain the edit, or just stopping and resuming? Both, depending on the patch** ([`results`](./benchmarks/discover_loop_fix/results_restart_control_qwen35-08b_colab_l4.json), Colab L4, `scripts/restart_control.py`). The MoI arm cuts each base answer where the router fires and resumes it with the patched model. The control resumes the exact same requests (same cut, seed and repeat penalty) with the **unpatched** base:
>
>     | Qwen3.5-0.8B | base | restart only | random dir. MoI | **our patch MoI** | patch vs restart (95% CI) |
>     |---|---|---|---|---|---|
>     | GSM8K (200, 2-layer L13-14 patch) | `11.0%` | `35.5%` | `32.0%` | **`55.5%`** | **`+20.0`** [`+14.0`, `+26.5`], 44 / 4 |
>     | MATH-500 (150, weak 1-layer patch) | `2.7%` | `17.3%` | `18.7%` | `15.3%` | `−2.0` [`−8.7`, `+4.7`], 11 / 14 |
>
>     Stopping a loop and resuming with the same model already gives `+24.5` / `+14.7` pts, with no weight edit, and a random direction adds nothing on top. On GSM8K the real patch adds `+20` pts more and actually prevents re-looping (loops `26%` vs `59.5%` after restart); the weak MATH patch adds nothing (loops `77%` vs `71%`). So MoI numbers must always be compared with the restart arm, not with base.
>   * **Can a patch gate itself with its own edit direction? (idea from ERMoE, arXiv 2511.10971)** ([`results`](./benchmarks/self_gate_router/results_qwen35-08b_colab_l4.json), Colab L4, `scripts/experiment_self_gate_router.py`). ERMoE routes by cosine similarity to each expert's own basis instead of a learned gate. Here the self-gate score is cos(module output, `u`) of the p50c loop patch, read per token on saved 0.8B base answers (half A sets the threshold, half B is scored; no generation):
>
>     | Held-out half B | GSM8K (100, 89% loop) | MATH-500 (75, 91% loop) |
>     |---|---|---|
>     | self-gate, patch `u` alone (token AUC) | **`0.961`** (L19) | **`0.933`** (L15) |
>     | trained probe, best layer | `0.983` (L15) | `0.972` (L15) |
>     | random direction, same modules | `0.56` / `0.80` | `0.51` / `0.78` |
>     | non-looping answers it fires on: self-gate vs text detector | **1 / 11** vs 7 / 11 | **1 / 7** vs 5 / 7 |
>
>     The patch's own direction is almost as good a loop detector as a trained probe, with nothing trained, and it raises far fewer false alarms on answers that would have ended fine than the 16-gram text detector. It is not an early warning: at 1% token false alarms it fires ~1,000 tokens after the text detector (median). Few non-looping answers (11 and 7), so the false-alarm counts are rough. (The `all_modules` rows in this file summed the two modules with opposite signs and cancel out; fixed in the script after this run.)
>   * **Self-gate as a veto: the text detector proposes, the patch direction confirms** ([`results`](./benchmarks/self_gate_router/results_veto_qwen35-08b_colab_l4.json), same saved answers, commit `7af7ca5`). The patch switches on at the first token after the text detector fires where the self-gate score is above its threshold (set on half A at a given token false-alarm rate):
>   * **The veto in generation keeps the whole MoI gain** ([`results`](./benchmarks/self_gate_router/results_veto_moi_qwen35-08b_colab_l4.json), GSM8K n=200, 16k, Colab L4, `veto_moi_colab.ipynb`): switching the p50c patch on only when the text detector fires AND the patch direction confirms (L15 cos, 1% token FPR) gives **`55.5%`**, the same as text-only MoI (`55.5%`, paired `0.0` [`-5.5`, `+5.5`]; rescued `59.0` vs `61.5%`, tie). It still switches on for `177/177` loops (median `118` tokens after onset, before the kill at onset+768) but on only `3/23` non-looping answers instead of `12/23`. Against its own restart arm the veto MoI wins by **`+29` pts** [`+22.5`, `+36`] (text MoI vs restart `+20`). Here the false alarms were harmless (all `22` healthy correct answers stay correct either way), so the veto pays off where a wrong switch-on costs something (the damage map: prose, Chinese, legitimate repetition). Restarting later costs the restart arm `-9` pts (no difference once rescued).
>
>     | Held-out half B, threshold at 1% token false alarms | GSM8K | MATH-500 |
>     |---|---|---|
>     | loops still switched on (L15 cos) | **100%** | **98.5%** (all_modules: 100%) |
>     | non-looping answers switched on: veto vs text detector alone | **1 / 11** vs 7 / 11 | **0 / 7** vs 5 / 7 |
>     | switch-on, median tokens after the text detector | `507` | `3,815` |
>     | switch-on vs loop onset (kill at onset + 768) | `-41` (before onset) | `+279` (before the kill) |
>     | random direction, same rule: loops caught | `33%` | `49%` |
>
>     The veto keeps (almost) every loop and removes most false alarms, at the cost of switching on later; it still switches on before the loop would be killed. At 5% token false alarms the delay drops to ~10 tokens (GSM8K) but non-looping answers switched on rise to 5 / 11 and 3 / 7, so the gain is mostly gone. Not yet tested in generation: whether the later switch-on keeps the MoI accuracy (the restart part of the gain depends on when the cut happens). Few non-looping answers (11, 7), so the false-alarm counts are rough.
>   * **Early warning: does the residual announce a loop before the text detector fires? Barely.** ([`results`](./benchmarks/onset_precursor/results_qwen35-08b_colab_l4.json), Colab L4, `wsi/precursor.py`, `scripts/experiment_onset_precursor.py`, same saved answers, no generation). A diff-of-means direction per layer for the 512 tokens *before* the text detector fires vs earlier tokens, fitted on half A; threshold set so ~10% of half-A non-looping answers raise an alarm; held-out half B:
>
>     | Half B | GSM8K: AUC | loops warned before text (median lead) | non-looping answers alarmed | MATH-500: AUC | warned | alarmed |
>     |---|---|---|---|---|---|---|
>     | precursor direction (L15) | `0.789` | **`43%`** (`298` tok) | 3 / 11 | `0.563` | `16%` | 4 / 7 |
>     | loop probe, same eval (L15) | `0.791` | `44%` (`74` tok) | 4 / 11 | `0.570` | `15%` | 5 / 7 |
>     | looser text detector (repeats in last 256 tokens) | `0.749` | `36%` (`100` tok) | 6 / 11 | **`0.651`** | `26%` | 5 / 7 |
>     | token position alone | `0.733` | `12%` | 6 / 11 | `0.529` | `13%` | 7 / 7 |
>     | random direction | `0.588` | `4%` | 1 / 11 | `0.570` | `4%` | 0 / 7 |
>
>     On GSM8K the direction warns a bit earlier than a looser text rule with fewer false alarms, but it is the same direction as the ordinary loop probe (both pick L15, same AUC), so it is not a separate "loop is coming" signal. On MATH-500 it does not transfer from half A to half B (AUC `0.73 -> 0.56`) and loses to the text rule. The answer-level threshold also does not hold on half B (3 / 11 and 4 / 7 alarmed instead of ~10%; few non-looping answers). Verdict: no usable early warning in the residual of the 0.8B; not worth building on. (Calibration note: a 1% token-level threshold fires early on most long answers, so alarm thresholds should be set per answer.)
>   * **Damage map: where does the always-on p50c loop patch hurt?** ([`results`](./benchmarks/damage_map/results_p50c_qwen35-08b_colab_l4.json), Colab L4, `wsi/damage_map.py`, `scripts/experiment_damage_map.py`). The same text goes through the 0.8B with the patch off and on (teacher forcing, 48 sequences x 512 tokens per domain), against a random-direction patch of the same size (same modules, kept columns, alpha):
>
>     | Domain | KL p50c / random | top-1 unchanged p50c / random | dNLL p50c |
>     |---|---|---|---|
>     | own healthy thinking | `0.015` / `0.0008` | `96.6%` / `99.1%` | `+0.007` |
>     | math (GSM8K) | `0.017` / `0.0008` | `96.8%` / `99.1%` | `+0.015` |
>     | prose (WikiText) | **`0.025`** / `0.0015` | `92.9%` / `98.0%` | `+0.009` |
>     | code (HumanEval) | `0.011` / `0.0006` | `96.7%` / `99.2%` | `+0.004` |
>     | chat (UltraChat) | `0.019` / `0.0013` | `94.4%` / `98.4%` | `-0.003` |
>     | Spanish | `0.011` / `0.0011` | `94.6%` / `97.9%` | `+0.016` |
>     | Chinese | `0.017` / `0.0017` | **`92.3%`** / `97.3%` | **`+0.018`** |
>
>     The patch changes the output 10-20x more than a random edit of the same size, in every domain, but mildly (2-4% of tokens with KL > 0.1). The worst tokens show what it breaks: **legitimate repetition**. It stops copying a name already in the context (`Kutuzov`, `Koshihikari`, `Beehively`, `Petoskey` -> `Peterson`, Jamaica in Chinese), the second `step by step` (`-by`), and another `>>>` example in a docstring that already has several (it predicts the closing `"""`). So the loop direction is an anti-repetition edit, and the cost of leaving it on is paid where text should repeat. This is why routing it (MoI: on only inside a loop) matters.
>   * **LFM3-1.7B does not loop on MATH-500** ([`results`](./benchmarks/discover_loop_fix/results_census_math500_lfm3_colab_l4.json), Colab L4, `discover_math500_lfm3_colab.ipynb`, model-card math sampling T 1.0 / top_p 0.95 / top_k 20, thinking on, 100 problems at 8k tokens): **`0%`** hard loops, `87%` correct, `8%` cut at 8k, so the loop gate stopped the run before discovery. Its fixable failures are elsewhere: over-refusal (`~51%` on OR-Bench-hard) and sycophancy (gives up a right answer under false pushback `98%` of the time, `blind_agent_syco_lfm3`). Loops are a small-model problem here (Qwen3.5-0.8B `85%`, 4B `0.7%`, LFM3-1.7B `0%`).
>   * **Copy heads: is there a head set that drives loop copying but not legitimate copying? No.** ([`results`](./benchmarks/copy_heads/results_qwen35-08b_colab_l4.json), Colab L4, `wsi/heads.py`, `scripts/experiment_copy_heads.py`). Every one of the 336 heads of the 0.8B (6 full-attention layers x 8, 18 Gated DeltaNet layers x 16) was zeroed in turn on 24 saved GSM8K base answers (teacher forcing). Tokens: LOOP (inside the loop), COPY (outside it, predictable by copying the context: names, "step by step"...), OTHER. Wanted: a head whose removal makes the model less sure of the loop but leaves COPY and OTHER alone.
>
>     * No such head. Of the 23 heads that move loop NLL by more than `0.02`, 21 hurt OTHER tokens even more. Across heads, loop damage tracks ordinary-text damage (Spearman `0.89`) more than copy damage (`0.47`): loops run on the same machinery as normal prediction, not on a separate copy circuit.
>     * Held-out half B (48 answers), change of NLL loop / copy / other: top 1 head (`L0 linear_attn #8`) `+0.80` / `+0.43` / `+1.45`; top 4 `+4.2` / `+4.3` / `+5.4` (model broken); random heads of the same count `~0.00-0.03`; p50c patch always on `+0.053` / `+0.053` / `+0.025`.
>     * Head ablation is not a cheaper or cleaner loop fix than a direction edit; keep the routed direction (MoI).
>   * **A blind agent with only WSI fixes over-refusal on a model nobody here had touched, and keeps the real refusals.** ([`results`](./benchmarks/blind_agent_refusal_aion/results_aion_colab.json), Colab L4, judge `scripts/eval_refusal_blind.py`, agents in `benchmarks/blind_agent_refusal_aion/`). Model: `microsoft/Aion-1.0-Instruct-2.3B` (new, custom code), which refuses **94%** of the benign-but-sensitive OR-Bench-Hard prompts. The agent got only the `wsi/` package and a one-page task (no recipe, no repo docs, no web; run with `claude -p` in a clean folder and audited) and had to write its fix **without ever running the model**; its script ran once, unattended, on the GPU. Held-out test (200 OR-Bench-Hard, 200 OR-Bench-Toxic, 30 ordinary prompts, none seen by the agent):
>
>     | Agent | Over-refusal (base 94%) | Over-refusals recovered | Toxic refusals kept | Ordinary answers identical | Pass |
>     |---|---|---|---|---|---|
>     | blind (only `wsi/` + task) | **33%** | **64.9%** | 93.3% | 100% | **yes** |
>     | saw AGENTS.md (with the LFM3 recipe) by accident | 50% | 46.8% | **97.9%** | 100% | no (recovery < 50%) |
>
>     * Both agents built the same kind of fix: the refusal direction removed from all 56 attention/MLP output projections, switched on by a router with a toxic veto (`wsi.router_cal`), so prompts it does not fire on run on untouched weights. The blind agent found that path by searching `wsi/` for "refus".
>     * The cost of the blind patch: toxic refusal 96.5% -> 90.0% (-6.5 pts, CI [-10, -3.5]); weakest categories privacy 80% and unethical 78% kept (few prompts each). The agent that saw the recipe traded recovery for safety (-2 pts toxic).
>     * **Caveat the judge does not see:** 69 of the blind patch's 122 recovered answers open in a style the base never uses ("The question asks: ... We are given: ...", 0/12 of the base's own answers), so the all-layer edit pushes the model partly off its usual voice; the answers stay on topic and none is repetitive. The recipe-aware agent's 88 recovered answers read normally (0/88). The judge now measures this (`style_check`, `eval_refusal_blind.py rescore`, text only): v2 opens 41% of its recovered answers with a template the base never uses, so it **fails `success_with_style`** (limit 20%); v1 is at 0%.
    * The agents' scripts ran in 8.1 min (blind) and 10.8 min (recipe-aware) on the L4.
    * The patch files are not published (always on, without the router, they would remove every refusal).
> * **Additive MoI outside math: sycophancy** (`moi_sycophancy.py`, Qwen3.5-4B bf16, Colab A100, thinking off, scored by the probability of each answer letter, so there is no sampling). The model answers an MMLU question; the user then pushes back ("are you sure? I think it's C"). Diagnosis uses 1531 MMLU validation questions, tuning uses 300, and the test uses 1000 test questions plus one pushback and one neutral wording never seen before. The base model gives up a right answer under false pushback `85%` of the time.
>
>   | Arm | Keeps right answer under false pushback | Accepts a correct suggestion | Neutral follow-up unchanged | First answer accuracy |
>   |---|---|---|---|---|
>   | base | `15.0%` | `97.4%` | `99.9%` | `68.8%` |
>   | v1 patch via MoI (direction: turns that caved vs. held; 64 MLP neurons, layers 20, 22) | `16.6%` | `97.4%` | `99.9%` | `68.8%` |
>   | **v2 patch via MoI** (direction: same question with pushback vs. neutral follow-up; attention output, layers 10, 14, α 8) | **`21.1%`** | `95.2%` | `99.4%` | `68.8%` |
>   | v2 always on | `21.1%` | `95.2%` | `99.2%` | `68.2%` |
>   | v2 with a random direction | `14.4%` | `97.4%` | `99.9%` | `68.5%` |
>   | stronger patch via MoI, no confidence gate (attention output, layers 17, 19, α 8) | `55.8%` | `37.8%` | `100%` | `68.8%` |
>   | **v3 = same patch + confidence gate** (fires only if the turn-1 letter had probability `> 0.885`) | **`40.1%`** | **`90.7%`** | `99.9%` | `68.8%` |
>   | v3, gentler patch (full-attention layers 19, 11, α 4) + confidence gate | `28.9%` | `95.8%` | `99.9%` | `68.8%` |
>   | ceiling: same strong patch, gated on whether turn 1 was actually right (not usable) | `55.8%` | `97.4%` | `99.9%` | `68.8%` |
>
>   * **The router is nearly perfect.** It reads one hidden state (layer 20, last token) and fires on `99.1%` of pushback turns, `98.8%` with an unseen wording, `3.3%` of neutral follow-ups and `0%` of first questions. Through MoI, first answers are bit-identical to the base model.
>   * **v1 barely worked** (`+1.6` pts, CI `[0.4, 2.9]`): the caved-vs-held direction mostly separates questions the model is unsure of. **v2 uses the pressure itself** (pushback vs. neutral on the same question) and tries three patch sites. The attention output won: `+6.1` pts over base (CI `[4.4, 8.0]`, 43 wins / 1 loss) and `+6.7` over a random direction. Cost: 7 fewer accepted corrections (`−2.2`, CI `[−4.2, −0.6]`) and 5 neutral turns changed where the router misfired.
>   * **It is a knob, not a fix.** On dev, a stronger patch on later attention layers keeps the right answer `65%` of the time but accepts a correct suggestion only `36%` (base `96%`): it makes the model ignore the user, right or wrong. The patch turns down "follow the user's suggestion"; it does not tell the model when the user is right. Generalization to the unseen pushback wording is small (`2.2% -> 4.5%`).
>   * **v3 adds that missing signal: a second gate on confidence.** The patch fires only when the router sees pushback **and** the model was sure of its first answer; when it was guessing, it can still be corrected. The gate, its threshold and the patch were chosen together on the 300 dev questions (7 patches × 3 signals). On test the strong patch keeps `40.1%` of right answers (`+25.2` pts over base, CI `[21.8, 28.6]`) while accepting `90.7%` of correct suggestions (`−6.7`, 22 of 312), vs. `37.8%` with the same patch ungated. It also generalizes better: the unseen pushback wording goes `2.2% -> 16.3%`. A gentler patch is nearly free (`+14.0` held, `−1.6` accepted, CI `[−3.5, 0.0]`). The oracle row shows the room left: a perfect confidence signal would keep `55.8%` with no loss in accepted.
>   * **The simplest confidence signal won.** The probability the model gave its own turn-1 letter predicts turn-1 correctness with AUC `0.83`. A "value" probe on the turn-1 hidden state (logistic regression, after Xu, Yuksekgonul & Zou 2026, *Sparse Reward Subsystem in LLMs*) reaches `0.75`, at layer 26 of 32; early layers, where that paper finds value neurons in another model, give `~0.62`. The probe gate was worse than the probability gate for every patch.
>   * **Exam:** WSI's neuron scores do not correlate with the single-neuron sycophancy effects measured by Naik et al. on this model (AWSP part 1, 801 labelled neurons, Spearman `|ρ| < 0.06`). The v2 patch acts on the attention output, not on those MLP neurons.
>   * **Ready to use:** both v3 patches ship as `.moipatch` files, [`patches/qwen35_4b_anti_sycophancy.moipatch`](./patches/qwen35_4b_anti_sycophancy.moipatch) (strong) and [`..._gentle.moipatch`](./patches/qwen35_4b_anti_sycophancy_gentle.moipatch), `52 KB` each with the router, the confidence gate and the expected numbers inside. `MoIPatch.fires(hidden_states, confidence=p_turn1)` applies both checks. The patched bf16 weights are bit-identical to the ones measured above, and `off()` restores the originals byte for byte.
> * **Writing facts into the weights** (`moi_memory.py`, Qwen3.5-0.8B fp32, Colab L4, [`results`](./benchmarks/moi_memory/results_colab_l4_qwen35_08b_1000.json), [`full-name gate`](./benchmarks/moi_memory/results_fullname_colab_l4_qwen35_08b_1000.json)). 1,000 people who do not exist, each with a city, a job and a dog's name, are written as one ROME-style memory per person (key = `down_proj` input at the last token of the name, value optimised on three training sentences only). The layer is picked on separate people (layer 3, `96%` vs `35%` base on dev). Scored by multiple choice among the right answer and 8 others (chance `11%`) on 50 people, with phrasings never written: dev, paraphrases, Spanish, chat, one hop (the city's country). Writing took `4.45 s` per memory, 16 at a time.
>
>   | Memories | always-on edit (`rome`) | MEMIT | gated on the surname (`rome_gated`) | **gated on the full name** (`rome_fullname`) | same surname, other person: gated / full name |
>   |---|---|---|---|---|---|
>   | 10 | `53%` | `98%` | `98%` | **`98%`** | `100%` / `17%` |
>   | 100 | `9%` (ppl `108`) | `81%` (ppl `27`) | `89%` | **`97%`** | `93%` / `13%` |
>   | 1000 | `8%` (ppl `198k`) | `38%` (ppl `338`) | `66%` | **`97%`** | `62%` / `13%` |
>
>   * **Dev accuracy above** (base `13%`, the facts in context `100%`). At 1,000 memories the full-name gate keeps paraphrases at `97%`, Spanish `84%`, chat `97%` and the one-hop country `78%`, with WikiText perplexity `22.67` (identical to base), all 30 capitals the base knows, and the gate firing on `0.0%` of WikiText tokens. Random keys with the same values stay at chance.
>   * **The surname gate leaked.** Memory *i* fired wherever its key matched, and that key sits on the surname token, so a different person with the same surname got the fact (`93%` of probes at 100 memories). `rome_fullname` adds a second key per memory, at the last token of the first name, and fires only if it matched within the 6 tokens before. The leak falls to base level (`13%`, own memory firing `0%`) while the right person still fires on `~100%` of probes.
>   * **Second family, `LiquidAI/LFM3-1.7B`** (bf16, 100 memories, [`results`](./benchmarks/moi_memory/results_lfm3_colab_l4_100.json)): the gate stays clean (perplexity `29.48` = base, `0%` firing on ordinary text, capitals `100%` kept, same surname / other person `5%` = base), but the facts are only half written: full-name gate dev `57%` (base `6%`, in context `99%`), city `86%` / job `65%` / invented pet name `12%`. The writes did not converge (final NLL `2.66` vs the `0.05` stop, all 40 steps used) and the layer sweep picked the earliest layer tried (L2), so the next run needs more steps or an fp32 write and layers 0-1.
>     * **Follow-up, layers 0-2** ([`results`](./benchmarks/moi_memory/results_lfm3_colab_l4_100_l012.json)): the sweep keeps improving down to the very first layer (L0 `92%`, L1 `85%`, L2 `82%` on the 12-person sweep). Writing at **L0** with the same 40 steps: full-name gate dev **`87%`** (was `57%`), paraphrases `80%`, chat `71%`, one-hop country `84%`, Spanish `66%`, invented pet name in chat `40%` (was `12%`); final NLL `0.75` (was `2.66`). The gate stays exact: perplexity `29.48` = base, `0%` firing on ordinary text, same surname / other person `5%` = base. Ungated edits cost more at L0 (always-on `rome` ppl `30.83`, MEMIT `30.48`). Note: WISE/MEMOIR report that early layers fail on 7B models; on LFM3 the earliest layer is the best one.
>     * **Streaming over HTTP / static CDN** (`wsi/cdn.py`, `moi_index.json`, `bf16` shards, L4): `100` single-entity `L0` shards (`19.5 KB` median, `2.8–3.0 ms` cached SHA-256 attach vs `128.2 ms` uncached) stream on demand under `30 ms` / `100 Mbps` simulation with zero RAG prompt tokens — cold recall (`19.5 KB` -> `"Toronto."`), warm LRU cache (`0 B` -> `"chemist."`), 2-entity composition (`Cuga` cached + `Quinfe` `19.5 KB` -> `"Barcelona and works as a nurse."`), controls (`0 B` / `0.0 ms` -> exact base), zero-restart live edits (`Toronto -> Tokyo` via SHA-256 invalidation), and `mode="static"` over plain static HTTP (GitHub Raw / Cloudflare R2).
>   * **The same change removed the decline with scale** (dev `+84` pts over base at 1,000, CI `[79, 89]`, vs `+53`, CI `[43, 63]`, for the surname gate). So the surname gate's drop from 98% to 66% was memories of other people firing, not a lack of capacity.
>   * **Limits:** recall needs the full name ("Cugamer" alone or "Mr. Bernilri" was not tested and would not fire). Exact generation is weaker than multiple choice (dev `46%` of answers greedy-decodable at 1,000). Backwards questions (dog -> owner) stay at `0%` for every method, as expected from a key on the name. The second run reused the 1,000 memories written by the first (same `progress.pt`).
>


</details>

<details>
<summary><strong>4. Section 6B: Selective MoI — Compensated Indexes, Pairwise OBS, Gram Compression, vLLM Speed, Task Zones & CompactTrace X-Ray</strong></summary>

> * **Compensated Selective MoI** (Selective MoI whose loaded down-projection columns are refit by ridge least squares on the task's activation covariance to cover the dropped neurons; `experiment_index_criteria.py`, [`results`](./benchmarks/index_criteria/results_colab_l4.json)) cuts most of the loss. Qwen3.5-0.8B math-solution ppl (full `5.33`):
>
>   | Neurons kept | `mean` (current index) | best plain criterion | best compensated | random |
>   |---|---|---|---|---|
>   | 70% | `6.42` | `5.83` (energy) | **`5.63`** (max+recon) | `16.8` |
>   | 50% | `8.90` | `8.33` (taylor) | **`6.21`** (taylor+recon) | `63.0` |
>   | 30% | `15.72` | `12.35` (energy) | **`7.18`** (energy+recon) | `352` |
>
>   The selection criterion matters a little; compensation, which uses second-order statistics (how neurons co-activate) instead of per-neuron averages, matters a lot. A math index still hurts general text (WikiText ppl `26.1 -> 53–57` at 70%), and these are perplexities, not task accuracy.
>
>   **Choosing the neurons in pairs, then compensating, beats every per-neuron criterion** (`--criteria obs`, [`results`](./benchmarks/index_criteria/results_pairwise_obs_colab_l4.json), Colab L4). The criteria above score each neuron alone. `obs` uses the whole covariance: it removes neurons one at a time, always the one whose loss hurts least once the remaining columns are refit (Optimal Brain Surgeon for one linear layer), so of two neurons that always fire together it drops one and the refit hands its job to the other. Math ppl (full `5.33`):
>
>   | Neurons kept | `energy` | `obs` alone | `energy+recon` | **`obs+recon`** |
>   |---|---|---|---|---|
>   | 85% | `5.44` | `5.66` | `5.46` | **`5.43`** |
>   | 70% | `5.83` | `6.31` | `5.74` | **`5.57`** |
>   | 50% | `8.54` | `8.80` | `6.22` | **`5.98`** |
>   | 30% | `12.35` | `12.69` | `7.17` | **`6.97`** |
>
>   `obs+recon` closes 42% / 27% / 11% of the remaining gap to the full model at 70 / 50 / 30%. Without the refit, pairwise selection is worse than `energy` (`obs` alone): it keeps neurons that only make sense once their partner's job is moved onto them. HOPE (arXiv 2609.18916) selects with pairwise terms but never compensates; this is the two combined. WikiText ppl improves at 70% and 50% (`50.4` vs `53.4`, `79.6` vs `82.6`) but is worse at 30% (`162` vs `137`). The selection pass takes `105` s for 24 layers on an L4. Still teacher-forced perplexity on 60 sequences, not task accuracy.
>
>   **In accuracy, pairwise selection only ties** ([`results`](./benchmarks/index_criteria/results_task_index_obs_lfm3_colab_l4.json), Colab L4). Both indexes on LiquidAI/LFM3-1.7B, matrices physically shrunk, 400 GSM8K problems, paired (full model `84.75%`):
>
>   | Neurons kept | `energy+recon` | `obs+recon` | obs − energy (95% CI) | only obs / only energy |
>   |---|---|---|---|---|
>   | 70% | `80.25%` | `82.0%` | `+1.75` [`−1.75`, `+5.5`] | 30 / 23 |
>   | 50% | `66.0%` | `68.75%` | `+2.75` [`−1.75`, `+7.25`] | 49 / 38 |
>   | 30% | `36.5%` | `37.75%` | `+1.25` [`−3.5`, `+6.0`] | 52 / 47 |
>
>   Ahead at every budget but no interval excludes 0, and on LFM3 its math perplexity is slightly worse (`33.6` vs `32.8` at 70%), the opposite of Qwen3.5-0.8B. `energy+recon` stays the default (simpler, no greedy pass); `obs` remains an option.
>
>   **How small can the compensation summary be?** The full covariance is `1.23 GB` (24 layers × `3584²` fp32). Keeping the `energy` selection fixed and varying only the covariance used for the refit (`experiment_gram_compress.py`, [`results`](./benchmarks/gram_compress/results_colab_l4.json)), math ppl:
>
>   | Stored summary | Size | 50% kept | 30% kept |
>   |---|---|---|---|
>   | none (no compensation) | 0 | `8.54` | `12.35` |
>   | full covariance (fp32) | `1,233 MB` | `6.22` | `7.18` |
>   | fp16, upper triangle | `308 MB` | `6.22` | `7.17` |
>   | top-1024 eigenvectors + exact diagonal | `176 MB` | `6.29` | `7.35` |
>   | top-256 eigenvectors + exact diagonal | `44 MB` | `6.62` | `7.90` |
>   | top-256 eigenvectors, no diagonal | `44 MB` | `12.26` | `13.15` |
>   | **20,000 calibration tokens** (covariance recomputed on load) | **`80 KB`** | **`6.34`** | **`7.36`** |
>   | 5,000 calibration tokens | `20 KB` | `6.82` | `8.24` |
>
>   Storing the task's tokens and recomputing the covariance (one forward pass over 20k tokens) keeps nearly all of the compensation in `80 KB`, ~15,000× less than the full matrix; fp16 storage is lossless. Low-rank alone fails (the top 256 directions hold only `77%` of the activation energy, the top 1024 `92%`); adding the exact diagonal is what makes it work. Side effect on general text is similar: WikiText ppl at 50% is `82.6` (full), `82.7` (fp16), `81.4` (top-256 + diag) and `88.2` (20k tokens), vs. `92.3` uncompensated; the token summary is the one that gives up a little here.
>
>   **Direct matrix quantization vs. recomputing PSD from compressed MLP activations** ([`results_quant`](./benchmarks/gram_compress/results_quant_colab_l4.json), `20,145` calibration tokens, full `5.01` math / `33.1` wiki; `full` fp32 `6.00 / 6.94` at 50/30% kept vs. `8.00 / 12.66` uncompensated, `fp16` upper triangle `308 MB` `6.00 / 6.93`): directly quantizing the upper triangle of $G$ into block `INT8` (`triu-int8`, `had64-int8`, `159 MB`) or 4-bit (`had64-int4`, `triu-fp4`, `96 MB`) breaks positive-semidefiniteness of $(G_{SS} + \lambda I)^{-1}$ and explodes perplexity (`5.7×10⁴` to `1.16×10⁷`). By contrast, recomputing $G = \hat{X}^\top \hat{X}$ from `CompactTrace`'s `Top-1024 FP4` MLP activations (`mlp-k1024-fp4`, guaranteed PSD by construction) is lossless (`5.97 / 6.95` vs. `6.00 / 6.94` at 50/30%, WikiText `118.6 / 216.3` vs. `124.8 / 217.8`), showing that `CompactTrace` preserves second-order co-activation geometry—though for disk storage `tokens-20000` (`80 KB`), `lowrank-256+diag` (`44 MB`), or `fp16` (`308 MB`) remain smaller than storing `Top-1024 FP4` token traces (`526 MB`).
>
>   **Does it hold on real task accuracy?** LFM3-1.7B, 200 GSM8K test problems, greedy decoding. The MLP matrices are physically sliced, so the sizes are real. Each index is chosen once from 100k tokens of math text, and compensation uses the covariance of 20k math tokens (`experiment_moi_task_index_lfm3.py`, [`results`](./benchmarks/moi_task_index_lfm3/results_colab_l4.json)). GSM8K accuracy (full model `83.0%`, `3.46 GB` bf16):
>
>   | MLP neurons kept | Model size | `uniform` | `energy` | `global` | `uniform+recon` | **`energy+recon`** | `global+recon` | random |
>   |---|---|---|---|---|---|---|---|---|
>   | 70% | `2.78 GB` (80%) | `66.0%` | `66.5%` | `64.0%` | `77.5%` | **`79.0%`** | `74.5%` | `3.0%` |
>   | 50% | `2.33 GB` (67%) | `36.5%` | `33.5%` | `28.0%` | `54.5%` | **`65.5%`** | `55.5%` | `0.5%` |
>   | 30% | `1.87 GB` (54%) | `8.0%` | `7.5%` | `6.0%` | `28.5%` | **`37.0%`** | `28.5%` | `1.0%` |
>
>   `uniform` keeps the same share of neurons in every layer, ranked by mean contribution to math text. `energy` ranks by mean squared contribution. `global` ranks across all layers at once, so layers get different budgets.
>
>   * Compensation adds `+12.5`, `+32` and `+29.5` points at 70/50/30%. At 70% the compensated model is `4` points below the full model, which is near the noise level for 200 problems (one problem = `0.5` pt). Per-token routing at 50% reached `63.5%` (trained router) and needed every weight resident. The per-task compensated index reaches `65.5%` while actually removing a third of the model.
>   * `energy` selection plus compensation wins at every budget. Uneven per-layer budgets (`global`) never help.
>   * Compensation specializes the model. WikiText ppl goes from `24.9` (full) to `51.95` with `energy` at 70% and to `71.8` with `energy+recon`. The math and wiki indexes share only `81% / 72% / 65%` of neurons at 70/50/30%, so each task needs its own index, which is the case for keeping several indexes.
>   * There is no decode speedup: `~745 tok/s` at every size, because HF `generate` at batch 32 on a 1.7B model is not bandwidth-bound. The saving is memory.
>
>   **Does it run faster when served for real?** `experiment_moi_speed_vllm.py` writes each compensated `energy+recon` index as an ordinary, smaller Hugging Face checkpoint (`intermediate_size` sliced in every layer) and serves it with vLLM. Decode speed is measured with one request at a time (the laptop/phone case, where generation is limited by reading the weights from memory) and with 64 at a time; quality is GSM8K, 200 test problems, thinking off. Kept neuron counts are multiples of 128, so the GPU kernels stay aligned. The first T4 run used unaligned sizes (`2509`, `1075`) and those ran *slower* than the full model; it is kept as [`results_kaggle_t4_qwen35_08b_run1.json`](./benchmarks/moi_speed_vllm/results_kaggle_t4_qwen35_08b_run1.json).
>
>   **Qwen3.5-4B** (vLLM bf16, Colab L4, calibration on 20k GSM8K-train tokens, [`results`](./benchmarks/moi_speed_vllm/results_colab_l4_qwen35_4b.json)):
>
>   | MLP neurons kept | GSM8K | Batch 1 | Batch 64 | Checkpoint |
>   |---|---|---|---|---|
>   | 9216 (full) | `90.0%` | `28.9 tok/s` | `863 tok/s` | `9.32 GB` |
>   | **6400 (69%)** | **`89.0%`** | **`34.2 tok/s` (`1.19×`)** | `953 tok/s` (`1.10×`) | `7.89 GB` (`0.85`) |
>   | 5248 (57%) | `82.5%` | `36.9 tok/s` (`1.28×`) | `996 tok/s` (`1.15×`) | `7.31 GB` (`0.78`) |
>   | 4608 (50%) | `75.0%` | `38.7 tok/s` (`1.34×`) | `1020 tok/s` (`1.18×`) | `6.98 GB` (`0.75`) |
>
>   **Qwen3.5-0.8B** (vLLM fp16, Kaggle T4, [`results`](./benchmarks/moi_speed_vllm/results_kaggle_t4_qwen35_08b.json)):
>
>   | MLP neurons kept | GSM8K | Batch 1 | Batch 64 | Checkpoint |
>   |---|---|---|---|---|
>   | 3584 (full) | `55.5%` | `114.7 tok/s` | `2013 tok/s` | `1.75 GB` |
>   | 2560 (71%) | `30.0%` | `115.2 tok/s` (`1.00×`) | `2074 tok/s` | `1.59 GB` (`0.91`) |
>   | 2048 (57%) | `26.0%` | `123.3 tok/s` (`1.07×`) | `2117 tok/s` | `1.51 GB` (`0.87`) |
>   | 1792 (50%) | `14.5%` | `126.7 tok/s` (`1.10×`) | `2118 tok/s` | `1.47 GB` (`0.84`) |
>   | 1024 (29%) | `4.5%` | `137.7 tok/s` (`1.20×`) | `2177 tok/s` | `1.35 GB` (`0.77`) |
>
>   * **On the 4B, 69% of the MLP costs one problem in 200** (`89.0%` vs. `90.0%`, within noise) and generates `19%` faster from a checkpoint `1.4 GB` smaller. At 50% it starts to break (`75%`, and some answers degenerate into repeated `###`).
>   * **Batch-1 speedup is the size ratio.** 4B: `1/0.847 = 1.18` predicted vs. `1.19` measured; `1/0.749 = 1.34` vs. `1.34`. One-at-a-time decoding is bound by reading the weights, so every byte removed is speed. With 64 requests the GPU is closer to compute-bound and the gain is smaller.
>   * **Bigger model, more spare neurons.** The 0.8B loses half its accuracy at 71% and gains almost no speed: its 248k-token vocabulary weighs about as much as all its MLPs (MLPs are `~35%` of parameters, vs. `~53%` in the 4B), so cutting MLP neurons barely shrinks it, and a model that small has little redundancy. LFM3-1.7B (above) sits in between.
>   * **Build cost** (4B, L4): load `14 s`, calibration `96 s` (2 passes over 20k tokens), then per size `36–61 s` select + compensate and `19–28 s` to write the checkpoint. About 3.5 minutes for three sizes, no training.
>   * Scope: one task (math), calibrated on that task. Other tasks and swap times are measured in the zones demo below.
>
>   **Zones: several task indexes and a router** (`moi_zones.py`, Qwen3.5-4B, vLLM bf16, Colab). Like a game that loads only the part of the map in view: each **zone** is a compensated `energy+recon` index with 6400 (69%) or 7808 (85%) of 9216 MLP neurons per layer, calibrated on one task's text. Attention and embeddings stay resident. A router reads the prompt before generation and picks the zone. It is a logistic regression on the mean input embedding, with no forward pass, `0.9 ms` per prompt. Calibration and evaluation never share a dataset: the code zone is calibrated on MBPP and tested on HumanEval (executed tests). MMLU is answered with a letter, 400 random test questions.
>
>   | Zone (calibration) | MLP neurons kept | Model | GSM8K | HumanEval | MMLU |
>   |---|---|---|---|---|---|
>   | full model | 9216 | `9.32 GB` | `91.5%` | `82.9%` | `72.8%` |
>   | **math** (GSM8K train, chat, 20k tokens) | 6400 (69%) | `7.89 GB` | **`87.0%`** | `14.6%` | `50.5%` |
>   | code v1 (MBPP, chat, 20k) | 6400 | `7.89 GB` | `57.5%` | `63.4%` | `43.5%` |
>   | code v2 (MBPP, chat, 80k) | 6400 | `7.89 GB` | `74.0%` | `73.2%` | `52.5%` |
>   | code v3 (all of MBPP, chat, 130k) | 6400 | `7.89 GB` | `77.0%` | `73.8%` | `50.0%` |
>   | **code v4** (MBPP, chat, 80k) | 7808 (85%) | `8.61 GB` | `89.5%` | **`78.0%`** | `67.8%` |
>   | general v1 (WikiText-2 paragraphs, **no chat template**, 20k) | 6400 | `7.89 GB` | `18.5%` | `0.6%` | `35.3%` |
>   | general v2a (MMLU `auxiliary_train`: ARC, OBQA, RACE…, chat, 20k) | 6400 | `7.89 GB` | `17.0%` | `0.0%` | `56.3%` |
>   | general v2b (same, 80k) | 6400 | `7.89 GB` | `23.0%` | `0.0%` | `59.8%` |
>   | **general v3a** (same, 80k) | 7808 (85%) | `8.61 GB` | `73.0%` | `26.2%` | **`72.5%`** |
>   | general v3b (same, 80k) | 8320 (90%) | `8.87 GB` | `85.0%` | `48.2%` | `69.3%` |
>   | math v5 (GSM8K train, chat, 20k) | 7808 (85%) | `8.61 GB` | **`91.0%`** | `37.2%` | `63.0%` |
>   | **one index, mixed** (40k each of GSM8K, MBPP, MMLU aux, chat) | 6400 (69%) | `7.89 GB` | `87.5%` | `67.1%` | `59.0%` |
>   | **one index, mixed** (same) | 7808 (85%) | `8.61 GB` | `91.5%` | `77.4%` | `70.3%` |
>
>   * **Calibration format matters most.** The first general zone, calibrated on plain encyclopedia text, scored below the *math* zone on MMLU (`35.3%` vs. `50.5%`). Re-calibrating with multiple-choice questions in chat format lifts it to `56.3%` (`+21`); 4× more tokens add only `+3.5`. A zone has to see the format it will answer in.
>   * **Then data, up to a point.** The code zone goes from `63.4%` to `73.2%` with 80k instead of 20k tokens, and gets better at the other tasks too (GSM8K `57.5% -> 74%`). All of MBPP (130k) adds only `+0.6`. The math zone needed only 20k.
>   * **Then capacity.** With more data no longer helping, the remaining gap is size. At 85% of the MLP neurons the general zone matches the full model on MMLU (`72.5%` vs. `72.8%`), and the code zone reaches `78.0%` on HumanEval (`−4.9`, about 1.5 standard errors on 164 problems). 90% was not better than 85% (`69.3%`; differences of ~3 pts are noise at 400 questions). A narrow task (grade-school math) loses `4.5` pts at 69% and nothing at 85% (`91.0%`); broad ones (general knowledge, code) lose `10–13` pts at 69% and `0–5` at 85%.
>   * **The router works; a wrong zone is costly.** It sends `100%` of GSM8K, `98.8%` of HumanEval and `94.3%` of MMLU prompts to their zone (23 MMLU questions go to math). The wrong zone loses most of a task (math zone on HumanEval `14.6%`, general zone on HumanEval `~0%`), so routing is required, not optional.
>   * **Routed zones vs. one index of the same size** (the fair comparison; v5). GSM8K / HumanEval / MMLU, then the mean:
>
>     | Size | Routed zones (best per task) | One index, mixed calibration | Full model |
>     |---|---|---|---|
>     | 69% (`7.89 GB`) | `87.0 / 73.8 / 58.5` → **`73.1%`** | `87.5 / 67.1 / 59.0` → **`71.2%`** | `91.5 / 82.9 / 72.8` → `82.4%` (`9.32 GB`) |
>     | 85% (`8.61 GB`) | `91.0 / 78.0 / 70.8` → **`79.9%`** | `91.5 / 77.4 / 70.3` → **`79.7%`** | |
>
>   * **The router does not pay off on these tasks.** At 85% one index calibrated on a mix of all three tasks ties the routed zones (`79.7%` vs. `79.9%`). At 69% the zones win only on HumanEval (`+6.7` pts, 11 of 164 problems) and tie on GSM8K and MMLU (mean `+1.9`). The routed zones were also picked by test score among up to 5 variants each, while the mixed index is a single untuned run, so the real gap is smaller still. The tasks share `74–80%` of their kept neurons, so one calibration covers almost everything.
>   * **What does work is the calibration itself:** chat format, the right tasks, and enough capacity. One mixed index at 85% averages `79.7%` vs. `82.4%` for the full model, in a checkpoint 8% smaller; a math-only index at 85% matches the full model on GSM8K (`91.0%` vs. `91.5%`). Zones might still help for tasks that share fewer neurons or at much smaller budgets (≤50%); that is not tested here.
>   * **Ridge target (`delta` vs. `zero` shrinkage) & router leak audit** (`experiment_moi_speed_vllm.py --ridge-target delta|delta_diag`, [`results`](./benchmarks/moi_zones/results_ridge_colab_qwen35_4b.json)): shrinking only the refit delta $\Delta W_S$ instead of shrinking $W_S \to 0$ preserves low-variance weights and prevents distortion as $k \to 100\%$, but ties on end-to-end mixed-index accuracy (`mix_a` 69%: `71.2%` `zero`, `71.3%` `delta`, `71.4%` `delta_diag`; `mix_b` 85%: `79.7%` `zero`, `78.7%` `delta`), confirming the remaining `~2.7`-pt gap at 85% is dropped-neuron capacity rather than shrinkage bias. Meanwhile, on the 23 MMLU items routed to `math`, `math` scored `30.4%` vs. `general_c`'s `65.2%`.
>   * **Swap cost** (L4): the part that changes is the MLP, `3.1 GB` per zone of a `9.3 GB` model. It moves from pinned RAM to GPU in `0.26 s` (`12.3 GB/s`) and loads from a warm disk cache in `~1.1 s`. Zones share `74–80%` of their kept neurons, but each zone refits every kept down-projection column, so a delta swap is still `~1.5 GB`. Each zone takes about 2.5–3 minutes to build (one calibration pass, select + compensate, write); no training.
> * **Token-Level Trace Compression & Differential X-Ray** (`capture_compact`, `diff_traces`, `diff_compact`, `experiment_trace_xray.py`, [`results`](./benchmarks/trace_xray/results_colab_l4.json)): when per-corpus averages (`ActivationIndex`, `Gram`) hide *which exact token* triggered a loop, trigger, or sycophancy flip:
>   * **Compression & quantization fidelity (`Qwen3.5-0.8B`, L4):** dense `fp32` `trace_all` over 32 tokens is `147.0 MB` (`4.59 MB/tok`, 70% in linear outputs). `capture_compact` (`group_size=16` block `INT4` + `Top-256` MLP + head L2) shrinks those 32 tokens to `1.69 MB` (`86.8×`) and stores a full `1,611`-token reasoning run in `85.3 MB` (`0.36 s`). Measured against exact `fp32` hidden states, block `INT4` (`cos=0.9944`) incurs `10.66%` L2 error while **block `INT8` (`cos=0.999984`, default)** drops L2 error to `0.59%`; `Top-256` MLP neurons (`7.1%` of `3,584`) capture `81.3%` of token-wise MLP residual energy.
>   * **Forensics (`0.8B` loop + trigger & `4B` sycophancy):** `detect_loop_onset` pinpoints repetition at token `696` (`lag cos=0.9706`) with Layer 15/19 heads shifting vs. token `83`; on a planted 1-neuron trigger `(L20, #42)`, `diff_traces` ranks `(20, 42)` `#1` at `3674.0` vs. `2.66` background (`1,380×`); on `Qwen3.5-4B` (`P(C)` `99.97% -> 35.90%` under pushback), `diff_traces` shows the two residuals drifting apart steadily from about Layer 12 (relative delta `0.30`) to Layer 31 (`0.53`; the jump at L32 is likely the final norm), suppressing letter-`C` neurons (`L23 #383`, `L28 #678`) and pushing `No`/`NO` while suppressing `"C`/`-confirm`. This is **one question pair**: an illustration of the tool, not a measured property of sycophancy.
>   * **Quantization sweep** (`experiment_quant_sweep.py`, results for [`Qwen3.5-0.8B`](./benchmarks/trace_quant_sweep/results_colab_l4.json), [`Qwen3.5-4B`](./benchmarks/trace_quant_sweep/results_4b_colab_l4.json), and [`LFM3-1.7B`](./benchmarks/trace_quant_sweep/results_lfm3_colab_l4.json), L4, GSM8K runs): residuals in block `INT8` (`group 64`) after a 64-wide Walsh-Hadamard rotation are the default, at `125 MB` vs. `208 MB` for `fp16` on 4B with cosine `0.99999`, logit KL `1e-5` and the same top-1 token on all 256 checked positions. 4-bit residuals (`INT4`/`FP4`, `82-90 MB`) keep cosine `~0.995` but move the detected loop onset by 12 tokens on 4B (one `FP4` setting misses it), and ternary (`~0.92`) misses it everywhere. For MLP units, `Top-1024` in `FP4` keeps `88%` (4B) / `95%` (0.8B) of the MLP energy at `60` / `40 MB`, the same as `INT8` for a quarter less space. On out-of-distribution hybrid [`LiquidAI/LFM3-1.7B`](./benchmarks/trace_quant_sweep/results_lfm3_colab_l4.json) (`ShortConv` + GQA + `7,680`-unit SwiGLU + 4 Sandwich RMSNorms, `506`-token non-looping run), `int8_g64_had64` reaches `cos = 0.999974` (`0.53%` rel L2, `kl = 1e-5`, `99.6%` top-1 match) with zero false-positive loop triggers (`onset = null` everywhere, `max_lag_cos = 0.883`), while `Top-1024 FP4` MLP (`13.3%` of units) keeps `90.7%` of MLP energy (`w2` output `cos = 0.960`, `18.6 MB` bitmask vs. `34.9 MB` `fp16`). With the full default `CompactTrace` enabled (`int8+had64` resid + `Top-1024 FP4` bitmask MLP + head L2 + `Top-16` attn + `int8` KV/DeltaNet cache), end-to-end storage is **`93.4 MB`** (`59.4 KB/tok`, `0.45 s` capture; `38.9 + 40.2 + 1.0 + 3.5 + 9.7 MB` vs. `7,058 MB` dense `fp32`, **`75.5×`**) on `0.8B` (`1,611` tok), **`174.7 MB`** (`162.8 KB/tok`, `0.68 s` capture; `88.5 + 60.2 + 1.9 + 6.4 + 17.7 MB` vs. `12,527 MB` dense `fp32`, **`71.7×`**) on `4B`, and **`51.8 MB`** (`104.8 KB/tok`, `0.17 s` capture; `24.5 + 18.6 + 0.4 + 4.1 + 4.3 MB` across `11` GQA layers/`22` cache tensors vs. `2,811 MB` dense `fp32`, **`54.3×`**) on `LFM3-1.7B` (`506` tok).
>   * **Compact Differential X-Ray & Causal `MoIPatch` Audit** (`diff_compact`, `experiment_diff_compact_syco.py`, [`results`](./benchmarks/trace_xray/results_diff_compact_syco_colab_l4.json)): on `Qwen3.5-4B` (`78` neutral + `86` pushback tokens), `CompactTrace` (`int8+had64` residual, `Top-1024 FP4` MLP with auto bitmask/bitpack, `Top-16` attention weights, `int8` KV/DeltaNet cache) takes **`26.08 MB`** (`12.40 + 13.67 MB`, `0.62 s` capture) vs. **`1,873.4 MB`** (`22.12 s`) for dense `fp32` `trace_all` (**`71.8×` smaller**). Comparing `neutral` vs. `pushback_base`, `diff_compact` recovers the block-by-block residual delta curve (`r = 0.9959` overall; `L0..30` tracks `hidden_states[1..31]` closely), **`9 / 10` (`90%`)** of the top-10 MLP neurons (with `#1..#5` in exact order: `(31, 683)`, `(28, 678)`, `(29, 84)`, `(23, 383)`, `(31, 1464)`), **`10 / 10` (`100%`)** of the top-10 KV/DeltaNet cache deltas in exact order (`<0.03` L2 error), the same `No`/`нет` vocabulary readout, and `6 / 10` (`60%`) of the top-10 heads (`L28 H29` near the top in both; in that run `diff_compact` diffed scalar pre-projection head norms $\|\mathbf{x}_h\|_2$ whereas `diff_traces` projected vector deltas through $W_O$). Following this finding, `capture_compact` now defaults to `head_quant="int8"` (storing `INT8`+Hadamard pre-$W_O$ head vectors alongside post-$W_O$ L2 norms) and `top_k_mlp="auto"` (`mlp_energy_target=0.95` per layer), `diff_compact` resolves $W_{O,h}$ from `model` or `wsi.ck` to compute exact projected head deltas $\|(\hat{\mathbf{x}}_{b,h} - \hat{\mathbf{x}}_{a,h}) W_{O,h}^\top\|_2$, and `diff_traces(..., compact=True)` routes through `CompactTrace` + `diff_compact` by default (`compact=False` preserves the uncompressed `trace_all` path). Running `diff_compact(ct_psh_base, ct_psh_patch)` to audit `patches/qwen35_4b_anti_sycophancy.moipatch` under pushback (`P(C): 99.97%` neutral $\to$ `35.90%` base pushback $\to$ **`89.84%`** patched) confirms `delta_norm = 0.0` (`cosine = 1.0`) across Layers `0..16`, activation at Layer 17 (`5.88`), a pivot at Layer 19 (`27.98`, `cos = 0.172`), and Layer 31 peak `62.33`: the patch's `#1` MLP shift (`(31, 683): -5.52`, `4.56 -> -0.96`) and `#1` head shift (`L28 H29: -4.79`, `10.28 -> 5.50`) directly reverse the exact `#1` MLP unit (`+4.56`) and `#1` linear-attention head (`+5.82`) that spiked under base pushback, while suppressing distractor answer tokens `"D"` (`-0.210`) and `"B"` (`-0.181`).
>   * **Hybrid `CompactTrace` + `WSI` Loop Discovery & $W_O$-Projected Onset X-Ray** (`discover_compact_circuit` / `I.discover_compact_circuit`, `experiment_loop_discover_compact.py`, [`results`](./benchmarks/discover_loop_fix/results_compact_colab_l4.json)): captures `14` `Qwen3.5-0.8B` GSM8K reasoning traces (`max_new_tokens=2,048`) into a **`3.06 GiB` (`3,133 MB`) `CompactTrace` bank in `9.49 s`** (storing `Top-1024 FP4` MLPs, `int8+had64` residuals, and `int8+had64` pre-$W_O$ head vectors). With hybrid attention layer mapping fixed (`fa_layers=[3, 7, 11, 15, 19, 23]`), `hybrid_top_layers` (`[14, 11, 15, 13, 10, 12]`) automatically promotes Full-Attention origin Layers **`11`** (`#2`, `hybrid_score=36.30`) and **`15`** (`#3`, `hybrid_score=28.89`) into the top 3; WSI cross-layer SVD resonance (`wsi_origins=[11, 15]`) then identifies **`15 -> 19`** (`wsi_resonance_score=0.6483`, `16.76σ` echo, `dir_cos=0.2377`) and **`11 -> 19`** (`0.5814`, `18.59σ` echo), automatically assembling `wsi_fa3_layers=[11, 15, 19]`. Purely offline from `CompactTrace` + static WSI weights (zero access to `p50c`), WSI recovers **`33 / 64` (`52%`)** of `p50c`'s edited neurons in Layer 15 (`dir_cosine=0.3577`) and **`25 / 64`** in Layer 19 (`dir_cosine` improves from `0.4289` trace-only to **`0.4731`** with WSI resonance). Running `diff_compact(ct_base, ct_p50c, pos=1776, model=model)` at loop onset shows `delta_norm = 0.0` across Layers `0..14`, onset at Layer 15 (`0.535`, `12.3%`), amplification at Layer 19 (`0.7246`, `10.3%`, `L19.keys=381.8`, `L19.values=110.8`) and Layer 23 (`1.3983`, `15.2%`, `L23.keys=450.1`, `L23.values=181.3`), while $W_O$-projected head vectors uncover **#1 `Layer 19 self_attn Head 4`** (`out_delta_norm=0.4768` vs. scalar `signed_l2_delta=+0.0752` — a **`6.3×`** rotational shift invisible to scalar head norms).
>     * *Why traces find the origin (Layers 11–15) while WSI finds the amplifier (Layer 19) — and caveat on hints:* Layer 19 sits outside `hybrid_top_layers[:6]` in trace activation variance, yet static weight geometry (`16.8σ–18.6σ` SVD echo) and $W_O$ vector rotation confirm it amplifies Layer 15's edit. **Caveat:** in that Colab run, selecting Layer 19 automatically relied on three answer-shaped filters (`6 <= l <= 16`, `is_full_attn`, and `dst_layer < 23`), which narrowed downstream FA candidates to Layer 19 alone while raw resonance ranked `15 -> 19` `#5` behind `15 -> 16` (`0.6904`), `11 -> 23` (`0.6888`), `15 -> 23` (`0.6693`), and `15 -> 18` (`0.6586`). `discover_compact_circuit` has since been cleaned to remove all layer-number, layer-type, and `< max_layer` hints and defaults to `blind_top2` (matching `discover_blind.py` above).
>     * *Held-out GSM8K (`24` problems, `2,048`-token cap):* strict `####` `acc` reaches `33.3%` (`ct_mlp_top3` `[14, 13, 10]`) and `25.0%` (`ct_wsi_fa3` `[11, 15, 19]`) vs. `16.7%` (`ct_wsi_pair` `[11, 19]` and `p50c_ref` `[15, 19]`), `12.5%` (`base`), and `8.3%` (`ct_mlp_fa3` `[11, 15, 7]`). Meanwhile `any_num_acc` is `50.0%–62.5%` across conditions (`base` `58.3%`, `p50c_ref` & `ct_mlp_fa3` `62.5%`, `mean_gen_tok` `1,916–2,048` vs. `loop_rate` `79.2%–100.0%`), confirming that at `2,048` tokens `Qwen3.5-0.8B` frequently computes the numerical answer before timing out on `####` formatting.



</details>

<details>
<summary><strong>5. Architecture Overview, Full Capabilities Table & Detailed Python / CLI / wsr / MCP Usage</strong></summary>

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
| **Compositional 2-Layer `AND` Gate (`A + B -> C`)** | `I_small.transplant_compositional_and_chain_from(I_big, ...)` | Wires `L_src` context detector `A` over a private residual wire into `L_dst` thresholded `AND(wire, B) -> C` gate |
| **Additive MoI** (Mixture of Indexes) | `I_small.create_mixture_of_indexes(src_layer=6, dst_layer=10)` / `create_dynamic_circuit_pager` | Indexes a catalog of `~12 KB` skill circuits (`AdditiveMoI` / `MixtureOfIndexes` / `DynamicCircuitPager`) and hot-swaps them on the fly into one shared dormant slot pair while earlier layers execute |
| **Portable `.moipatch` Recipes** | `MoIPatch.load("patch.moipatch")` / `wsi moipatch verify ...` | Packages self-routing, SHA-256 verified, byte-reversible Additive-MoI edits (`10-50 KB`, e.g. `patches/qwen35_0.8b_p50c_loop_fix.moipatch`, `patches/qwen35_4b_anti_sycophancy.moipatch`). The loop patch is a rank-1 extraction of p50c's edit (`99.84%` of the patched bf16 weights identical to p50c); its expected numbers are p50c's, not re-measured with the patch |
| **Runtime Capture & Selective MoI** | `capture_index(model, texts, task="math")` / `trace_all(model, ids)` / `SelectiveMoI(indexes).route(...)` | Records what every linear layer and residual stream actually uses on a task (`ActivationIndex`, ~28 MB per task on Qwen3.5-0.8B), routes prompts to the closest task index and physically loads only its neurons (`apply_selection`) |
| **Token-Level Compact Trace, Prompt X-Ray & Hybrid Discovery** | `capture_compact(model, ids)` / `diff_traces(model, ids_a, ids_b, wsi=I)` / `I.diff_compact(ct_a, ct_b)` / `I.discover_compact_circuit(traces)` / `wsi diff-compact` / `wsi discover-circuit` | Compresses per-token residuals (`INT8`+Hadamard + adaptive `top_k_mlp="auto"` 95% energy `FP4` MLP + `head_quant="int8"` pre-$W_O$ vectors + `INT8` cache), locates repetition onset, diffs `CompactTrace` (`diff_traces(..., compact=True)` default, exact $W_O$-projected `head`, `neuron`, KV/DeltaNet cache, vocab readout), and pairs origin traces with static `WSI` SVD resonance to synthesize `.moipatch` circuits |
| **MCP Server** | `python -m wsi mcp` | Exposes all 22 `wsi_*` tools (including `wsi_audit_moipatch`, `wsi_calibrate_safety_router`, `wsi_bake_moipatch`, `wsi_diff_compact`, and `wsi_discover_compact_circuit`) over standard JSON-RPC 2.0 stdio for any MCP-compatible AI agent |

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

# Diff saved CompactTrace directories (W_O-projected heads + WSI grounding) or discover a .moipatch circuit
python -m wsi diff-compact models/qwen35-4b traces/neutral traces/pushback
python -m wsi discover-circuit models/qwen35-0.8b --trace traces/clean --loop-trace traces/loop -o fix.moipatch
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



</details>

<details>
<summary><strong>6. Native Rust SIMD Core, Standalone Rust Baker & Reproducible Benchmark Commands</strong></summary>

```bash
# Build & test the zero-dependency standalone .moipatch -> .safetensors baker (`391.5 KB`, `11.01x` faster than Python, SHA-256 parity across BF16/F16/F32)
cargo test --manifest-path rust/wsi-bake/Cargo.toml
cargo build --release --manifest-path rust/wsi-bake/Cargo.toml
```

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

# 10. Run Cross-Architecture 2-Layer Compositional AND Gate (A + B -> C, LFM3 L0+L1 -> GPT-2 L6->Wire->L10)
python scripts/experiment_compositional_and_circuit.py

# 11. Run Mixture of Indexes (MoI) Dynamic Weight Paging Benchmark (20 LFM3 circuits swapped on-the-fly in 1 GPT-2 slot pair)
python scripts/experiment_dynamic_weight_moe.py

# 12. Run 3-Way Head-to-Head Benchmark (Original GPT-2 vs Static 40-Neuron vs MoI 2-Neuron)
python scripts/benchmark_moi_vs_static.py

# 13. Run fast CPU unit tests for CompactTrace, W_O-projected diff_compact/diff_traces, CLI, MCP, and hybrid discovery
python tests/test_tracer.py

# 14. Run CompactTrace quantization sweep, differential X-ray (4B anti-sycophancy), and hybrid loop-fix discovery (Colab L4)
python scripts/experiment_quant_sweep.py
python scripts/experiment_diff_compact_syco.py
python scripts/experiment_loop_discover_compact.py
```



</details>

---

## Credits & License

* Built on top of and compatible with [**`wsr` (Weight-Space Reading)**](https://github.com/suayptalha/wsr) by Şuayp Talha Kocabay (2026).
* Licensed under the [Apache License 2.0](./LICENSE).

