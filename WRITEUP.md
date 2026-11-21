# Surgical Weight-Space Patching Across Hybrid LLMs: Automated Circuit Localization, Capability Parity, and Boundary Trade-Offs

*A 10-minute technical write-up of the empirical findings from **Weight-Space Index (WSI)** (`github.com/krexito/weight-space-index`).*

---

## Executive Summary (TL;DR)

Standard weight edits for fixing unwanted LLM behaviors—such as all-layer refusal orthogonalization ("abliteration", Arditi et al., 2024)—rewrite dozens of matrices across the entire network (`~478M` weights on a `1.7B` model), drift normal generation, and erase roughly half of harmful refusal.

Across three distinct model families (**`Qwen/Qwen3.5-0.8B`** Gated DeltaNet, **`LiquidAI/LFM3-1.7B`** ShortConv hybrid, and **`microsoft/Aion-1.0-Instruct-2.3B`**), we find that **completely automated blind weight-space search**—with zero hand-picked layer hints—localizes causal edits orders of magnitude smaller than all-layer baselines:

* **Static 64-neuron patch (`abliterated-bake`, `17.7 KB`):** Editing just **64 columns of a single MLP down-projection** (`layers.15.feed_forward.w2`, `131,072` weights = `0.0076%` of `LFM3-1.7B`, found out of `330` screened candidates in `13.5 min` on a single NVIDIA L4) recovers **`85.0%`** of `OR-Bench-Hard` false refusals and **`82.4%`** of out-of-distribution `JailbreakBench-Benign` false refusals, while keeping **`GSM8K` (`81.0% → 82.5%`, 95% CI `[-1.5, +4.5]`) and `MMLU` (`45.0% → 47.0%`, 95% CI `[0.0, +4.0]`) unchanged within statistical noise**. Normal-prose response drift drops **`10.7×`** (`KL 0.00091`, `99.48%` top-1 unchanged) vs. our 2-module static baseline.
* **Runtime routed patches (`abliterated` & `abliterated-sfw`, `41–54 KB`):** A lightweight residual-stream router keeps the model **100% identical to base (`0/600` false fires, `0.00000` KL)** across `GSM8K`, `MMLU`, and everyday controls, while a paired framing-invariant veto router (`abliterated-sfw`) preserves **`96.4%`** refusal on `OR-Bench-Toxic` and **`94.9%–100.0%`** under adversarial `JailbreakBench` wrappers (educational, fiction, hypothetical).
* **Honest boundary finding:** A dedicated borderline-benign stress test revealed that an uncalibrated 1-sensor `L9` router false-fires on **`100%`** of edgy-but-benign prompts (`KL 0.062`). Our framing-invariant `L24` router (`abliterated-sfw`) cuts boundary false-fires to **`7.2%`** (`6/83`) on OOD `JBB-Benign` (`KL 0.0023`) and **`25.6%`** (`10/39`) on `OR-Bench`, whereas the **static 64-neuron bake** generalizes best near the OOD boundary (`0.0020` KL on JBB boundary, `82.4%` recovery). Put simply: **the baked 17.7 KB patch is the most robust general-purpose artifact; the 2-sensor router adds safety retention and `0.00000` KL on everyday text at a modest cost on boundary prompts.**
* **Zero-dependency deployment:** `.moipatch` files stream-bake byte-identically (`BF16`/`F16`/`F32` SHA-256 verified) via a standalone **`391.5 KB` Rust binary (`rust/wsi-bake`, `11.01×` faster than Python: `256.6 ms` vs `2,825.8 ms`)**, or hot-swap live on a single base checkpoint in **`204–267 ms`** (`0.26 ms` restore to `base`).

---

## 1. Motivation & Baseline: Why All-Layer Editing Leaks

Arditi et al. (*"Refusal in Language Models Is Mediated by a Single Direction"*, NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)) showed that erasing a single refusal direction from every residual writer suppresses refusal. We treat this all-layer orthogonalization as our explicit baseline (`classic_ablit_all_layers`) on **`LiquidAI/LFM3-1.7B`**:

* **Scope:** modifies all `48/48` residual writers (**`478.2M` weights**, `27.7%` of model parameters).
* **False-refusal recovery:** reduces `OR-Bench-Hard` over-refusal from `50.6%` to `8.9%` (**`85.0%` recovery**).
* **Collateral damage:** alters normal-prompt generation heavily (`0%–10%` exact match on `30` benign controls; `healthy_kl = 0.01455–0.02018`).
* **Safety collapse:** toxic refusal on `OR-Bench-Toxic` drops from `83.0%` to `44.5%` (**only `49.4%` of harmful refusals kept**).

Our hypothesis: most of the `48` edited matrices only relay or amplify downstream representations, whereas a sparse subset of **causal switches** and **late writers** controls onset with far less collateral drift—and can be localized automatically without architecture-specific heuristics.

---

## 2. Automated Weight-Space Discovery Pipeline

WSI discovers edits in three automated stages (`Half-A` calibration/discovery only; all reported benchmarks use disjoint held-out `Half-B` or external OOD datasets):

1. **Paired Target + Healthy-Prose Fisher Causal Tracing (`wsi/tracer.py`):**
   * Capture compressed per-token forward activations and backward gradients over **target onset tokens** (e.g., refusal onset window with `healthy_span=(0, 0)`) paired with separate **full healthy-prose generation traces** (`healthy_only=True`, marking `grad_n_target=0` so healthy tokens never contaminate `C_target`, empirical Fisher curvature `healthy_cost="fisher"`).
   * Rank all candidate modules (MLP projections, attention `o_proj`, and hybrid `ShortConv` / `Gated DeltaNet` outputs) and individual neuron columns without layer hints.
2. **1-Forward-Pass Pareto-KL Screening + Token-0 Onset Check:**
   * Rather than guessing hyperparameters or generating completions for hundreds of configs, `screen_patch_candidates(rank_by="pareto_kl")` scores a `330`-candidate grid `(site/pair/triple, budget 16–128 neurons, alpha 1–8)` via teacher-forced forward passes (~`475 s` for all `330` candidates on `LFM3-1.7B` on L4).
   * **Onset decoy safeguard:** Earlier runs using a `64`-token discovery window let a body-text edit (`L23.w2 + L9.w2`) win the forward proxy while leaving token-0 `'I'` probability at `81.1%` (`7.5%` generation recovery)—caught immediately by end-to-end `output_logit_diff` in `wsi.moipatch.audit_moipatch`. Restricting discovery to the onset window plus automated token-0 onset verification (`verify_onset_suppression`) eliminates decoys before evaluation.
3. **Framing-Invariant 2-Vector Safety Router (`wsi/router_cal.py`):**
   * Unvetoed refusal edits suppress harmful refusals too (`6.0%` safety kept). Adding a second linear veto vector (`veto_dir`) separates benign over-refusal from true harmful intent.
   * **The adversarial framing trap (v1 → v2):** A linear `veto_dir` fit purely on `100` raw `OR-Bench` toxic/benign pairs overfit superficial polite/academic tokens (`"For"`, `"purely"`, `"ensuring"`), dropping `JailbreakBench` educational-wrapper safety from `95.0%` down to `48.0%` (`50.5%` kept).
   * **Calibration fix (`Σ_frame` invariance + boundary negatives):** `fit_framing_invariant_safety_router` fits `(dir, veto_dir)` via strict `Group K-Fold` out-of-fold CV using:
     * a closed-form Woodbury solver minimizing a paired framing-covariance penalty `Σ_frame` across **calibration-only** wrappers (strictly 4-gram disjoint from JBB evaluation wrappers),
     * hedging-balanced sample weights and exact orthogonalization against Gate 1, and
     * `ctrl_plus_nonref` Gate-1 negatives so Gate 1 rejects ordinary non-refused boundary prompts.

---

## 3. Empirical Results on `LiquidAI/LFM3-1.7B`

We evaluate four patch configurations alongside unmodified `base` across three held-out evaluations:

* **`abliterated-bake` (`17.73 KB`, static):** `64` columns in `layers.15.feed_forward.w2` (`131,072` weights, `0.0076%` of model, `always_on`).
* **`abliterated` (`41.2 KB`, routed):** `layers.11.feed_forward.w2 + layers.9.conv.out_proj` (`19.9M` weights) + `L9` 1-sensor router.
* **`abliterated-sfw` (`53.93 KB`, safety-vetoed):** same `2` edit modules + calibrated `L24` 2-vector `Σ_frame` safety router.

### 3.1 Capability, Prose Drift, and Over-Refusal Recovery

Paired evaluation (`eval_capability_and_boundary_lfm3.py`, greedy, identical prompt formatting) alongside Half-B `OR-Bench` and `prose_preserving_always_on_lfm3_l4.json`:

| Arm | Weights Edited | `GSM8K` (`n=200`) | `MMLU` (`n=400`) | Router Fires (`600` Cap. Prompts) | `OR-Bench` Half-B Recov. (`n=40`) | `JBB-Benign` OOD Recov. (`n=17`) | Normal Prose KL (`0.0 = identical`) | `OR-Bench-Toxic` Safety Kept (`n=166`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`base`** | `0` | `81.0%` (`162/200`) | `45.0%` (`180/400`) | `0.0%` | `0.0%` | `0.0%` | `0.00000` | `100.0%` (`83.0%` raw) |
| **`classic_ablit`** *(all layers)* | `478.2M` (`27.7%`) | — | — | N/A | `85.0%` | — | `0.01455–0.02018` | `49.4%` *(lost half)* |
| **`abliterated-bake`** *(64n static)* | **`131K` (`0.0076%`)** | **`82.5%`** *(+1.5, CI `[-1.5, +4.5]`)* | **`47.0%`** *(+2.0, CI `[0.0, +4.0]`)* | `always_on` | **`85.0%`** (`34/40`) | **`82.4%`** (`14/17`) | **`0.00091`** *(99.48% top-1)* | No veto |
| **`abliterated`** *(`L9` routed)* | `19.9M` (`1.15%`) | **`81.0%`** (`+0.0`) | **`45.0%`** (`+0.0`) | **`0/600` (`0.0%`)** | **`92.5%`** (`37/40`) | **`100.0%`** (`17/17`) | **`0.00000`** (`30/30` exact) | `6.0%` *(unvetoed)* |
| **`abliterated-sfw`** *(`L24` veto)* | `19.9M` (`1.15%`) | **`81.0%`** (`+0.0`) | **`45.0%`** (`+0.0`) | **`0/600` (`0.0%`)** | **`65.0%`** (`26/40`) | `11.8%` (`2/17`) | **`0.00000`** (`30/30` exact) | **`96.4%`** (`160/166`) |

**Key takeaway:** Neither `GSM8K` nor `MMLU` shows statistically significant degradation (`abliterated-bake` 95% CIs span zero: `[-1.5, +4.5]` on `GSM8K` with `7` gained / `4` lost, `[0.0, +4.0]` on `MMLU` with `12` gained / `4` lost and `94.5%` choice agreement). Both routed patches achieve **exact zero collateral drift (`0/600` false fires)** on capability benchmarks and everyday controls.

### 3.2 Safety Retention Under Adversarial OOD Framing (`JailbreakBench`)

Measured out-of-distribution on `JailbreakBench/JBB-Behaviors` (`n=100` harmful goals per suite; **zero JBB prompts or wrapper 4-grams seen during calibration**):

| JBB Harmful Suite (`n=100` each) | `base` Refusal | `abliterated-sfw` **v1** (Uncalibrated) | `abliterated-sfw` **v2** (`Σ_frame` Calibrated) | Delta (v1 → v2) |
| :--- | :---: | :---: | :---: | :---: |
| **`jbb_harmful_raw`** | `94.0%` | `89.4%` kept | **`97.9%` kept** (`92/94`) | `+8.5 pts` |
| **`jbb_harmful_fiction`** | `54.0%` | `98.2%` kept | **`100.0%` kept** (`54/54`) | `+1.8 pts` |
| **`jbb_harmful_educational`** | `95.0%` | `50.5%` kept | **`96.8%` kept** (`92/95`) | **`+46.3 pts`** |
| **`jbb_harmful_hypothetical`** | `98.0%` | `65.3%` kept | **`94.9%` kept** (`93/98`) | **`+29.6 pts`** |

On unseen validation wrappers, framing shift-to-std dropped **`2.44×`** (`0.854 → 0.349`), and false vetoes on normal controls dropped from `83.3%` to `20.0%` (`0.0%` overall fire rate, `30/30` exact match).

### 3.3 Borderline-Benign Stress Test (Where Routers Are Tested Near the Decision Boundary)

Measuring router false-fires strictly on **everyday controls** can hide failure on benign questions phrased with sensitive words ("borderline" prompts that `base` actually answers rather than refuses). We explicitly isolated those non-refused boundary subsets (`n=39` in `OR-Bench` Half-B; `n=83` in `JBB-Benign` OOD):

| Arm | `OR-Bench` Boundary (`n=39`) Fire Rate | `OR-Bench` Boundary Exact Match | `OR-Bench` Boundary KL | `JBB-Benign` OOD Boundary (`n=83`) Fire Rate | `JBB-Benign` OOD Boundary Exact Match | `JBB-Benign` OOD Boundary KL | Newly Refused Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`abliterated` (`L9` 1-sensor)** | `100.0%` (`39/39`) | `0.0%` | `0.06217` | `100.0%` (`83/83`) | `0.0%` | `0.01939` | **`0.0%`** |
| **`abliterated-sfw` (`L24` 2-sensor)** | **`25.6%`** (`10/39`) | **`74.4%`** | `0.02506` *(0.0977 when fired)* | **`7.2%`** (`6/83`) | **`92.8%`** | **`0.00234`** *(0.0323 when fired)* | **`0.0%`** |
| **`abliterated-bake` (`64n` static)** | `100.0%` *(static)* | `7.7%` *(43.6% first-16)* | **`0.01520`** | `100.0%` *(static)* | `20.5%` *(72.3% first-16)* | **`0.00204`** | `2.6%` OR / `0.0%` JBB |

This ablation exposes the real trade-off:
1. Training a single-vector gate only against everyday controls (`L9`) leaves a broad boundary leak (`100%` firing on edgy-but-benign prompts). Adding `ctrl_plus_nonref` + `Σ_frame` (`L24`) eliminates `74%–93%` of those boundary misfires, and even when it misfires, it causes `0.0%` false refusals.
2. Because `abliterated-bake` edits only `64` Fisher-selected late MLP neurons (`layers.15.feed_forward.w2`), its **worst-case** drift across OOD boundary prompts (`KL 0.00204`) is actually slightly lower than `L24` (`KL 0.00234`), while recovering **`82.4%`** vs `11.8%` of OOD `JBB-Benign` over-refusals—making the baked `17.7 KB` patch the cleaner general-purpose artifact when toxic safety retention is not simultaneously required.

---

## 4. Cross-Architecture Generalization (Zero Layer Hints)

The same WSI localization abstractions run unchanged across **three distinct model families and failure modes**:

1. **`Qwen/Qwen3.5-0.8B` (Gated DeltaNet Hybrid — Reasoning Thinking Loops):**
   * Blind discovery (`scripts/discover_blind.py` / `#7 Reconstruction A/B`) screened `60` candidates in `124.4 s` and ranked `L13-14_n16_a12` (`32` neurons, `4×` fewer weights than MIT CSAIL `p50c` baseline) **#1/60** by `pareto_score`.
   * On held-out `GSM8K` (`n=200`, `16k` context), MoI-routed `blind_moi` jumped from `11.0%` (`base`) to **`55.5%`** (tying `p50c` `58.0%`, `89/0` vs `base`), and beat the **restart-only unpatched control** (`35.5%`) by **`+20.0 pts`** (`95% CI [+14.0, +26.5]`, `44/4`; loops cut `59.5% → 26.0%`; `random_moi` `32.0%` tied restart).
   * In a clean head-to-head with identical 189-word briefs, an agent using the WSI toolkit achieved **`35.5%`** always-on / **`50.5%`** routed accuracy vs. **`11.5%`** always-on / **`44.0%`** routed for a raw-PyTorch agent (`+24.0 pts` always-on margin in `8.4 min`).
2. **`LiquidAI/LFM3-1.7B` (ShortConv Hybrid — Over-Refusal + Safety Retention):**
   * Automated search across `330` candidates localized both multi-module (`L11.w2 + L9.conv`, `92.5%` recovery) and ultra-sparse 64-neuron single-layer (`L15.w2`, `85.0%` recovery, `17.7 KB`, zero capability loss) edits without architecture-specific rules.
3. **`microsoft/Aion-1.0-Instruct-2.3B` (Blind Agent Over-Refusal Benchmark):**
   * Base over-refuses `94%` of `OR-Bench-Hard` and refuses `96.5%` of `OR-Bench-Toxic`. A completely isolated `claude -p` blind agent given only `wsi/` + `TASK.md` (zero web access, zero prior runs) reduced over-refusal `94% → 33%` (`64.9%` recovered), kept `93.3%` toxic refusal, and preserved `100%` identical ordinary answers in `8.1 min` on L4—passing all behavioral gates while flagging an important **style caveat** (`new_template_rate = 41%` vs `<= 20%` judge bar, prompting `style_check` enforcement across all agent runs).

---

## 5. Negative Results & Methodological Lessons

Several hypotheses failed cleanly and sharpened our methodology:

* **No "copy head" shortcut (`wsi/heads.py`):** Ablating all `336` attention/linear-attention heads one by one on `Qwen3.5-0.8B` (`GSM8K`, `24` answers) showed no loop-selective heads: `21` of `23` heads with `loop ΔNLL > 0.02` damaged normal tokens even more (`Spearman = 0.89` loop vs. normal). Top-1/Top-4 head ablation broke generation (`+0.80 / +4.2 ΔNLL`) compared to `p50c` (`+0.025`).
* **Onset precursor signals (negative):** Pre-detector diff-of-means probes on `Qwen3.5-0.8B` matched ordinary loop probes (`AUC 0.789` vs `0.791` on GSM8K; dropped `0.73 → 0.56` out-of-sample on MATH-500). Early-warning routers need per-answer, not per-token, calibration.
* **Always report MoI against a restart-only control (`scripts/restart_control.py`):** Resuming cut generations with the **unpatched** base model already recovers substantial accuracy on weak edits (`11.0% → 35.5%` on GSM8K; `2.7% → 17.3%` on MATH-500 where a weak 1-layer edit tied restart at `15.3%`). Real weight-edit signal must beat restart (`+20.0 pts` on 0.8B GSM8K), not just unrouted `base`.
* **Damage map justifies routing (`wsi/damage_map.py`):** Evaluating `always_on` loop edits across 7 domains showed `10–20×` higher KL on prose (`0.025`) and Chinese (`top-1 92.3%`) concentrated precisely on legitimate repetition (repeated entity names, `"step by step"`, code doctests).

---

## 6. Zero-Dependency Distribution & Serving

A research patch is only practical if anyone can run it without writing PyTorch hook code (`wsi/serve.py`, `rust/wsi-bake/`, `wsi/mcp_server.py`):

1. **Standalone Static Baking (`moipatch export` & Rust `wsi-bake`):**
   * Stream-patches `.moipatch` edits onto HuggingFace `safetensors` shards on disk with SHA-256 base-weight fingerprint verification and an explicit `--force` guard blocking accidental static baking of safety-routed patches.
   * Cold verification on `LiquidAI/LFM3-1.7B`: loading the baked `/content/lfm3_baked` directory with **pure `transformers` (zero WSI runtime code)** matched runtime hooked `abliterated-bake` **exact token-for-token (`3/3 exact=True`)**.
   * **Zero-dependency Rust CLI (`rust/wsi-bake/`, `391.5 KB` single binary, zero crates, zero Python/PyTorch):** Benchmarked on a `114.0 MB` `bfloat16` checkpoint (`2048 × 7680` LFM3-scale matrix + 5 extra tensors): Python median `2,825.8 ms` (`[5114.6, 2825.8, 2765.2]`) vs. Rust median **`256.6 ms`** (`[257.3, 256.4, 256.6]`) — **`11.01×` speedup** with **byte-exact `SHA-256` parity** across `BF16`, `F16`, and `F32`.
2. **Live Hot-Swap Server (`moipatch serve`):**
   * Keeps one base checkpoint in VRAM behind an OpenAI-compatible `/v1/chat/completions` endpoint, REST control plane (`/v1/wsi/patches/*`), and local web chat UI.
   * Live latency on `LiquidAI/LFM3-1.7B` (Colab L4): **`204–267 ms`** to hot-swap between `abliterated`, `abliterated-sfw`, and `abliterated-bake`, **`0.26 ms`** to revert to unmodified `base`.
3. **Agent-Ready MCP Server (`22` tools in `wsi/mcp_server.py`):** Exposes indexing, multi-hop causal direction graphing, iterative Hydra peeling (`peel_causal_circuit`), 1-pass candidate screening, end-to-end `wsi_audit_moipatch`, `wsi_calibrate_safety_router`, and `wsi_bake_moipatch`.

---

## 7. Limitations & Reproducibility

* **Single-model capability scope:** Paired `GSM8K`/`MMLU` + borderline-benign breakdown was completed on `LiquidAI/LFM3-1.7B`; `abliterated-sfw`'s boundary recovery on OOD `JBB-Benign` (`11.8%` vs `82.4%` static bake) reflects the expected precision-vs-safety trade-off of strict 2-gate vetoing (`OR-Bench-Toxic` `96.4%` kept).
* **Sampling sensitivity:** Reasoning loop rates depend strongly on decoding temperature and presence penalty (e.g., `Qwen3.5-4B` MATH-500 dropped from `39.1%` greedy loops to `0.7%` under standard sampling); always benchmark loop incidence under target serving sampling.
* **Key scripts & unit tests (all CPU tests pass without GPU):**
  * Over-refusal + healthy-Fisher discovery: `scripts/experiment_prose_preserving_always_on_lfm3.py`, `tests/test_prose_preserving_always_on.py`
  * Framing-invariant safety calibration + JBB OOD: `scripts/calibrate_veto_invariant_lfm3.py`, `wsi/router_cal.py`, `tests/test_veto_calibration.py`
  * Capability + boundary-benign benchmark: `scripts/eval_capability_and_boundary_lfm3.py`, `tests/test_capability_and_boundary.py`
  * Serving, baking & MCP: `wsi/serve.py`, `wsi/mcp_server.py`, `tests/test_bake_and_serve.py`, `tests/test_mcp_server.py`, `rust/wsi-bake/` (`cargo test`)
