# Agent & Experiment Playbook (`WEIGHT-SPACE-INDEX`)

Shared conventions for agents authoring experiments, scripts, and Colab/Kaggle notebooks in this repo.

## 1. Colab / Kaggle Notebook Speed & Resume Optimizations

Generation (`12k–16k` thinking tokens) is ~95% of wall-clock time. Apply these defaults in all new vLLM evaluation/discovery notebooks:

1. **Shared Drive `base` output cache (`WSI_BASE_CACHE_DIR`):**
   - Point `os.environ["WSI_BASE_CACHE_DIR"] = "/content/drive/MyDrive/wsi_runs/_base_cache"`.
   - Key cached `base` outputs deterministically by `(model_repo, dataset, split, n, max_tokens, seed)` (see `_base_cache_path` / `_try_load_base_cache` / `_save_base_cache` in `scripts/discover_math500_blind.py`).
   - Reusing `outputs_elicit`, `outputs_dev_base`, and `outputs_test_base` saves ~40–50 min on every subsequent run over the same `(model, dataset, split)`.
2. **Fast vLLM options on L4 / H100 (`SM >= 8.9`):**
   - Enable `kv_cache_dtype="fp8"`, `enable_chunked_prefill=True`, `enable_prefix_caching=True`, and `gpu_memory_utilization=0.90` with a progressive fallback list so hybrid architectures (e.g. Qwen3.5 Gated DeltaNet) seamlessly fall back to default KV dtype if `fp8` is refused (see `_llm_fast` in `scripts/discover_math500_blind.py`).
   - Always set `PYTHONPATH` to include `scripts/` and pass `loop_kill_v2.processor()` so early loop stop (`early_loop_stop=True`) works in vLLM 0.34+ worker subprocesses.
3. **Split token budgets (`discover` vs `test`):**
   - Use `max_tokens = 8192` for `elicit` and `dev` (almost all Qwen3.5-4B loop onsets occur at `3k–8k` tokens; cuts discovery time ~2×).
   - Keep `max_tokens = 16384` intact for the held-out `test` exam.
4. **1-Forward-Pass Proxy Screening (`wsi.tracer.screen_patch_candidates`):**
   - Before running multi-config generation on `dev`, screen the `(k, n, alpha)` grid with `screen_patch_candidates(rank_by="pareto", max_nll_delta=0.05)` on 6 held-out Half-B looping sequences (~2–3 min, zero generation) and pass only the **Top-3** finalists to `dev`.
   - Validated on `Qwen/Qwen3.5-0.8B` (60 configs in `124.4 s`; ranked the blind winner `L13-14_n16_a12` **#1/60** by `pareto_score`; Spearman `-0.565` vs `loop_rate`, `+0.529` vs `truncation`).
   - Caveat (2026-11-26): on LFM3 refusal the damage penalty hurt routed-patch selection (see the rule on effect vs always-on damage in section 2); the 1-pass screen also misranked MATH-500 loop patches. Prefer the escape test for the final pick.
5. **Drive Resume & Atomic Final Writes:**
   - Set `WSI_SYNC_DIR` to `MyDrive/wsi_runs/<run_name>` and sync after every completed stage.
   - Write final JSONs to `MyDrive/wsi_runs/FINALES/<run_name>.json` via `.tmp` + `os.replace`.
6. **Hardware Constraint:**
   - Never run heavy model inference on the local Windows PC — only CPU unit tests, syntax checks, and small toy-model tests.

## 2. Findings So Far (index)

One line per finding. Numbers, caveats and files: **`research/findings.md`** (search it by the words in bold; do not load it whole). New result: append the full bullet there and add **one line of at most ~220 characters** here.

**Rules learned (apply these)**
- Report MoI against the **restart** arm (resume the cut with the unpatched base), not against base; report **answer rescue** next to the strict number.
- Measure the loop rate with the run's own sampling (`lr.SAMPLING`) before choosing a benchmark; greedy inflates loops. Use a `16k` dev split on MATH-500 for the 0.8B.
- Set alarm thresholds per answer, not per token. Keep loop patches routed (MoI), not always on.
- Select routed patches on effect (suppression or a short escape test), not on always-on damage: control KL rises with edit strength, and the router never pays it (LFM3 refusal: KL penalty 0.33 vs escape 0.92). Measure damage where the gate acts (false fires, style). One model so far; check on loops before changing the pareto default below.
- Keep `WSI_SPEC` off for batch evaluation. Structured pruning default: uniform `energy+recon`.
- Refusal work: screen on the token-0 onset window, batch each control set separately with identical batching in every arm, require `toxic_kept >= 0.90` AND `recovered >= 0.50`, and report `success_with_style`.
- Agent-tool subagents get this file injected. A truly blind agent needs `claude -p` in a folder outside the repo.
- Models with remote code (Aion): pin `transformers==5.23.0` (5.24.0 renames the generation cache; Aion then output garbage with no error; pinning restored the base); check the unpatched base first.

**Loops: Qwen3.5**
- **`Qwen/Qwen3.5-0.8B`** (GSM8K): blind discovery works; MoI `55.5%` vs base `11%`, restart `35.5%`.
- **`Qwen/Qwen3.5-0.8B`** (MATH-500): weak patch; MoI ties random direction and restart.
- **`Qwen/Qwen3.5-4B`**: GSM8K saturated (`94.5%`); MATH-500 loops only `0.7%` with real sampling. Nothing to fix.
- **Restart-only control**: restart alone is most of a weak patch's gain (GSM8K `35.5%`, MATH `17.3%`).
- **Answer rescue**: last stated answer before the loop onset; GSM8K base `11 -> 33.5%`; rescue at the kill, about 60% precise.
- **`#7 Reconstruction` A/B**: same brief, WSI toolkit `35.5 / 50.5%` vs raw PyTorch `11.5 / 44.0%` (always on / routed).
- **MIT patch anatomy** (diffs, CPU): p50c reverses one direction (x-5) in 128 MLP columns; p10c's 15 layer directions and p50c's are one family; our blind patch points elsewhere (|cos| `<= 0.22`).
- **Direction AUC** (saved answers, 4090): our blind GSM8K direction detects MATH-500 loops as well as the best fitted one (token AUC `0.977` vs `0.978`; MIT p50c `0.947`): detection is not the bottleneck (see edit shape).
- **Edit shape vs direction** (MATH-500, always on): MIT p50c `28.7%` in our eval; our GSM8K blind patch `20.7%` (beats the MATH-500 blind pick `10.7%`); p50c's shape (64 cols, alpha 6) on our direction hurts. Gap = direction, not shape.
- **Screen vs exam** (6 known patches, MATH-500): the 1-pass screen ranks the blind pick L10 (`10.7%`) first, above p50c (`28.7%`) and our GSM8K patch (`20.7%`); suppression is not escape. Selection needs an escape test.
- **Escape test** (98 elicit cuts, MATH-500): continuing from the loop cut, escape vs restart + NLL guard orders the known patches like the exam (Spearman `0.75-0.80`); the L10 pick ties restart. Use it to select.
- **Blind escape-selected search** (MATH-500): 108 candidates, picks L15 (p50c's layer) blind; test `18.0%` vs old pick `10.7%`, ties our GSM8K patch `20.7%`, below p50c `28.7%`. Escape filters junk; ranks top ones noisily.
- **Self-gating router** and **self-gate veto**: the patch's own direction is a usable router; as a veto it keeps the MoI gain (`55.5%`) and cuts false switch-ons `12/23 -> 3/23`.
- **Router calibration** (596 saved answers, no generation): the veto's 1% token threshold sits at the knee: all 480 loops on, `3/59` correct healthy answers on (`29/116` healthy); per-answer thresholds lose loops.
- **Final-layer sensor token** (saved answers): a linear readout of the last layer (what a spare vocab row gives in vLLM) detects loops almost like the L13-14 veto (AUC `0.969` vs `0.980`; same `3/59` correct healthy on, `480/480` loops, ~100 tokens later).
- **Sensor token in real vLLM**: a spare vocab row read as a raw logit leaves outputs identical (`64/64`), matches fp32 (corr `0.99997`) and keeps the fp32 threshold (`100%` agreement). A free in-engine probe channel.
- **Damage map**: the loop patch hurts legitimate repetition (names, Chinese, prose).
- **Rescue from the inside** (saved ids, no generation): the state at the cut does not predict which loops the blind patch rescues (all AUC CIs include 0.5); loops any restart fixes come later.
- **Multiple futures per cut** (8+8 per cut, sensor on): the rescue is real (`+22` pts, 110 cuts better / 5 worse); one sample is a noisy label (34% luck); the sensor flags hard cuts, not which future fails (within-cut AUC ~0.5).
- **Cut state vs clean rescue rates** (CPU): with 8-sample labels the cut state predicts difficulty (sensor vs restart rate rho `-0.41`) but not the patch's gain (all CIs incl. 0); gain peaks on borderline loops (`+46` pts).
- **Never-rescued cuts**: 48/54 are loop-bound (74% of futures loop again, 11% finish wrong); 15/54 are solved by an always-on arm from token 0. The ceiling is loop depth at the cut, not model skill.
- **Reading correctness** (Qwen2.5-1.5B, MATH-500): hidden-state probes hit AUC `0.84` but confidence + length fitted give `0.83`; the probe adds nothing (`+0.015`, CI incl. 0). Internals = confidence here.
- **Refusal patch vs random direction** (LFM3, fresh prompts): patch recovers `70.5%` / keeps `97.0%` toxic; random dirs, same alpha+router, `0-8.5%`; system prompt `41.1%` / `91.6%`. The direction is the gain.
- **Escape selection for refusal** (LFM3, 24 edits, same router): 64-token escape ranks like the exam (Spearman `0.85` grid-only) vs screen `0.32`; its pick ties the shipped patch (`70.5%`). Style untested (none failed).
- **Two patches, same matrices** (LFM3): routed, exactly additive (A `70.5%` recovery kept; B toxic `+10.5` kept; together over-ref `53.7->20%`, toxic `83->91%`); always on, B's gain halves. Routing makes stacking safe.
- **Hessian error as edit damage** (LFM3, 24 edits): layer-wise tr(dW H dW^T) does not track control KL (rho `-0.10..0.24`); routed patches should be screened on effect: KL penalty hurts (KL rises with strength).
- **Neuron switches vs direction** (LFM3, Chantless-style): 16 zeroed w2 neurons behind the same router recover `65.6%` / keep `96.4%` toxic (patch `70.3/96.4%`, random `1.6%`); always on `95.3%` but toxic `14.5%`. Router carries safety.
- **Negative**: **Perpendicular loop direction**, **Copy heads**, **Copy-protected localization**, **Onset precursor** (no early warning).
- **Base-draft speculative** and **Speculative decoding** (`mtp`): at most `1.15x`, slower on full batches. Dropped.

**Refusal: LFM3-1.7B and Aion**
- **LFM3-1.7B loop census**: `0%` loops; its fixable failures are over-refusal (~51%) and sycophancy.
- **Surgical Over-Refusal** / **Safety Retention** / **Direction-Traced**: the 2-vector routed edit recovers `65-70%` and keeps `96-98%` of toxic refusals; full removal loses half of them.
- **OOD & Adversarial Framing** then **Framing-Invariant `veto_dir`**: v1 broke under educational framing (`50.5%` kept); the invariant calibration keeps `94.9-100%`.
- **Prose-Preserving `always_on`** and **Paired Capability**: small static edits leave GSM8K/MMLU unchanged within noise; routed arms fire `0/600`.
- **Tracer Upgrades**: linearized final norm, block gaps, `protect_dir`, multi-hop graph, peeling; speed fix.
- **Blind agent + WSI vs over-refusal** (Aion): passes the old gates (`64.9%` / `93.3%`) but **fails with style** (41% exam-solver openings).
- **Over-refusal without the voice change** (Aion): no arm passes; null-space edits lose the effect, the cleaned direction fixes the voice but recovers only `27.7%` (alpha not re-tuned).
- **3-Tome WSI Atlas (`wsi_atlas_v1`)**: 15 Tome-1 + 11 Tome-2 (L1+L2 `tome1_curve_deltas`) + 12 Tome-3; post-training is intra-layer uniform (`head_gini <= 0.077`, `OLMoE` `0/1024` frozen); `R1-Distill` spikes `o_proj 21.6%`.
- **Blind Live-Agent A/B (`Phi-4-mini`, `deepseek-v4.1-flash-1017`, 10.0 on Writing Weights)**: WSI solves hold-out with style in `23–28 min` (`~$0.03`, `77.8/91.7/100%` `wsi_only` clean); Atlas A/B ties `1–1` (Run 2 `wsi_plus_atlas` bias bug + timeout).
- **Tome-2 10× Scale (`Qwen2.5-14B` Instruct & R1-Distill)**: all 4 laws hold at 48L (`head_gini 0.026/0.044`, `erank 0.86 vs 0.65/0.75`, late `L46-47` Fisher); `R1-Distill` amplifies `o_proj` to **`56.4%`** (`cos +0.969`, `toxic -> 0%`).
- **Blind Antigravity A/B (`Phi-4-mini`, zero-leak bridge)**: `wsi_plus_atlas` (L40S, 13t) **PASSES** `72.2%` rec / **`97.0%`** toxic / `100%` ctrl / `style_ok` (`L8–20`, `344 KB`) vs `wsi_only` (4090) **FAIL** (`87.95%` toxic, `L0–31`, `811 KB`).
- **Looped Transformer X-Ray (`Ouro-2.6B` -> `Thinking`, `48L×4=192` steps)**: 4-cycle sawtooth norm, DLA peaks pass 3 (`step 139/L43`), early Fisher (`0.064`); `Thinking` wipes `toxic 100%->0%`, MLP-dominated (`65.5%`), Tome-2 laws hold.
- **Inverse Safety Restoration (`Ouro-2.6B-Thinking`, Modal L40S)**: routed `pass=2 L20` (`a=0.4`) restores `toxic 10->80%`, `OOD framed 8.3->63.3%` with **`0/100`** benign/ctrl/math fires (`overref 3.3%`, `ctrl/GSM8K 100%` exact).
- **Hybrid Looped + mHC X-Ray (`SenseNova-Loop-9B`, `32L->56` steps)**: flat mid-loop plateau (`cos 0.993`), DLA only in unlooped tail (`step 53/L29`), extreme prologue Fisher (`0.00094`); `37.5%` over-ref, `100%` framed-toxic, `0%` syc/loops.
- **Atlas-Guided Loopification (`Qwen2.5-3B-Instruct`, zero training)**: `L20–25 × 2` (`42` steps) beats base (`ARC 87.5->90%`, `GSM8K 20->25%`, `NLL +0.07`); prologue `L0–5 × 2` (`ARC 50%`, `GSM 0%`) & DLA tail `L30–35 × 2` (`fact 0%`) collapse.
- **Atlas Loopification + LoRA (`Qwen2.5-3B`, MATH-500)**: negative; `L20–25×2` ZS `49.5%` vs base `61.5%`; `loop_shared/pass2` (`27.5/26.0%`) < `lora_flat` (`34.0%`, `-6.5/-8.0`); terse gold SFT cuts len 4×.
- **Large & Diverse Modal X-Ray Batch (5 models, 9B–32B)**: Nemotron-9B Mamba-2 (norm `2469`, Fisher `3e-5`), Qwen3-30B-A3B MoE (`cos 0.986`), OLMo-2-32B PostNorm (`massive 104x`), GLM-4-32B (`aniso 0.015`), Inkling-2-Mini MoE+Looped (`24->48`, DLA in-loop `step 35/L11`).
- **Native-Memory Hybrid MoE X-Ray (`MemT-1.0-Nano`, `35.1B/3B`, Modal H200)**: zero-pathology (`0%` overref/syc/loops, `100%` raw+framed toxic, `cos_ref_veto 0.964`), router ent `0.925`, Fisher `0.077`, `L12` massive `441.6x`.
- **Dense Hybrid DeltaNet X-Ray (`Qwen3.8-27B`, `64L`, Modal A100-80GB)**: uniform MLP (`erank 0.93`), tail `q_sens 0.67`, `block_fisher 0.004`, `L58` massive `283x`, `0%` overref/syc/loops, `75/100%` raw/framed toxic, `87.5%` facts.

**Memory, routing, distillation, compression**
- **Facts in the weights** (LFM3): gate clean, recall `87%` dev (chance `11%`); layer 0 is best; more steps do not help, the gap looks like generalization from one sentence per fact.
- **MoI Weight & Memory Streaming CDN**: `19.5 KB` shards, ~3 ms attach.
- **Two `.moipatch` files at once**: no interference seen (weak test, no shared matrix).
- **Router false hits**: negative; no arm is a safe cache for "same problem, new numbers".
- **Reasoning-to-MoI Projective Distillation**: routing is invariant, projective generation is negative.
- **Pocket MoE Ceiling** / **Pocket MoE Controls**: 4B -> 0.8B; WSI ahead of random layers at ~1.7 MB (`91.7%` vs `80.6%`, CI includes 0); at 27 MB one unrouted LoRA wins.
- **Real-World `MATH-500` + `MMLU` Pocket MoE**: negative for WSI and splitting; the Layer-12 router does shield from forgetting (`0.0%` fires off-domain).
- **Few-Shot Continual Pocket MoE** (`K=6`): balanced L12 router (`0/200` MMLU); cartridges crush continual `r=8` (`+41.7` old skills), tie `r=64` & random, but quick `24`-step (`61.7%`) loses to 1-shot (`75%`).
- **Regularized Multi-Trace Pocket MoE** (`K=6`): 24L `r=8` + 144p hits `88.3%` (`+21.7` vs v1), `0/200` MMLU fires (4L reg underfit; pass-count caveat overturned below -> `144p single-trace` was the real driver).
- **0-Backprop Closed-Form & Hybrid Warm-Start Pocket MoE** (`K=6`): negative vs LoRA; ridge fits attention layers (`R2~0.6`) but not GDN (`R2~0.01`, `36.7%` vs `81.7%` 1-shot); SVD+8-step ties cold 8-step (`53.3%` vs `51.7%`).
- **Thinking-to-Weights Compression** (4B `<think>` -> 0.8B no-think, MATH-500): post-think targets `36.5%` (`+6.5` vs base `30%`); privileged think-KL hurts (`29%`, `-7.5` vs post); think-only items `1/15`: no reasoning compressed.
- **Equal-Pass-Budget Multi-Trace `MATH-500` Pocket MoE**: multi-trace ties 1-trace at equal passes (`<=1` epoch underfits); reg `24L r=16` + L12 routing hits new best `41.5%` (`+9.5` vs base, `0/371` non-math MMLU fires).
- **De-Confounded `K=6` Multi-Trace & Rank Frontier**: equal-pass `144p single-trace` hits record `91.7%` (`+15` vs `96p`, `+11.7` vs `multitrace` `80%`, beats 1-shot `85%`); `r=2` (`2.58 MB`) retains `r=8` within `3.3 pts`.
- **Single-Trace `144p` Rank Frontier (`r=8..r=1`)**: `r=8` (`10.3 MB`) leads `85.0%` (`+11.7` vs 1-shot); `r=2 @ lr=4e-4` (`2.58 MB`) ties `r=4` (`76.7%`, beats 1-shot); `r=1` (`1.29 MB`) breaks (`61.7%`, `0%` `s5`); `0/200` MMLU fires.
- **WSI 2-Bit Quantization Recovery** and **Global Fisher Heterogeneous MLP Pruning**: negative (2-bit stays at chance; global Fisher starves late layers).
- **Pairwise neuron selection**: tie with `energy+recon`.
- **Low-bit trace residuals**: negative; a QTIP-style trellis beats scalar at 3 bits (relL2 `0.14` vs `0.18`) but INT8 quality needs >= ~7.5 bits (residual ~Gaussian, near full rank). Keep `int8+had64`.
- **Past the trace bit limit**: recompute from tokens in fp32 is exact (6,300x smaller; bf16 recompute is worse than INT8). 3-4 bit residuals keep layer/direction conclusions, lose onset and logit lens.
- **Recompute GPU vs CPU**: fp32 GPU recompute matches CPU to `1e-6` and is deterministic, so traces can be stored as tokens; fp16/fla stay < INT8 error (0.2-0.3x); bf16 is 2.2x INT8: avoid.
- **Additive Looped MoI (`Qwen3-0.6B`, SFT & GRPO)**: SFT collapses len/acc (`64.7->33%`); 48-step GRPO + 3-class L14 router preserves prose/MMLU (`0.0` FPR/`+0.0`) with null gain (`60.7-63.3%` vs `62.7%` base).
- **Pocket LoopRes (`Qwen3-4B->0.6B`, `K=6`)**: `loopres_only` (`46.7 KB`) fails (`41.7%`), but `r=2` + LoopRes (`2.45 MB`) zero-shot `3x` hits `80.0%` (`+15.0` vs `2x`, `+8.3` vs `lora_1x` `71.7%`), `0/200` MMLU fires.
- **Pocket LoopRes `T`-Sweep & `varT` (`K=6`)**: `fix2x` peaks at `T=2..3` (`71.7%` vs `65.0%` `lora_1x`, `10w/6l`), drops at `T=4..5` (`70.0/65.0%`); 48-step `varT {2,3}` trails (`60-65%`); `0` MMLU/prose damage.
- **Pocket LoopRes on `MATH-500` (`Qwen3-4B->0.6B`)**: `r=4` `loopres_T2` hits `50.0%` (`+4.5` vs `lora_1x` `45.5%`, `18w/9l`; `+3.0` vs `base` `47.0%`); `T3` drops (`46.5%`); `0.0` MMLU non-math fire (`prose_fpr 0.29`).
- **Pocket MoE Strata Quant (`4B->0.8B`, `K=6`)**: `r=8->S2_K` (`1.43 MB`, `7.2x` smaller) `73.3%` (`+21.7` vs `r=1` `51.7%`; ties `RTN-2b`); `S3_K` (`2.08 MB`) `71.7%`; `0/200` MMLU fires.
- **Dense Strata Quant (`Qwen2.5-3B`, `A100`)**: `S3_K` (`H64+BlockLDL+64flip`, `3.19b`) rescues RTN (`PPL 554->11.8`, `GSM8K 0->70%`); raw knapsack hurts (`57%`); `+0B Scale-QAT` adds `+4 pts` (`61%`).
- **MoE Strata Quant (`OLMoE-1B-7B`, `A100`)**: `S3+S2` (`2.85b`) cuts RTN PPL (`51.9->18.8`); `+0B Scale-QAT` rescues raw knapsack to beat uniform on PPL (`17.1` vs `18.8`) & KL (`0.64` vs `0.68`); MMLU `24.5%`.
- **X-Ray Quant Predictor (CPU)**: `build_quant_sensitivity_map` tracks Inkling `SQ-10` (`rho +0.65`, `2/4` top-5) & knapsack (`+0.82` dense, `+0.68` OLMoE); raw PreNorm `tr(dW H dW^T)` grows `~140x` across depth.

## 3. Planned Next Experiment (Queued — Do Not Run Yet)

- **Blind Autonomous Agent + WSI MCP (No 189-word brief):** Give an LLM agent **zero recipe/method hints** (same open-ended setup as MIT CSAIL `p50c` / `wsr program`) — only a model exhibiting a failure mode (e.g. thinking loops or sycophancy) and access to the **15 WSI MCP tools** (`wsi_discover_compact_circuit`, `wsi_build_moipatch`, `wsi_screen_candidates`, etc.), and measure turns, wall-clock time, and held-out accuracy vs. a raw-PyTorch agent baseline. Queued until `discover_math500_4b_colab.ipynb` finishes.
