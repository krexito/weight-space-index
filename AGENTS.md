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
- **Self-gating router** and **self-gate veto**: the patch's own direction is a usable router; as a veto it keeps the MoI gain (`55.5%`) and cuts false switch-ons `12/23 -> 3/23`.
- **Router calibration** (596 saved answers, no generation): the veto's 1% token threshold sits at the knee: all 480 loops on, `3/59` correct healthy answers on (`29/116` healthy); per-answer thresholds lose loops.
- **Damage map**: the loop patch hurts legitimate repetition (names, Chinese, prose).
- **Rescue from the inside** (saved ids, no generation): the state at the cut does not predict which loops the blind patch rescues (all AUC CIs include 0.5); loops any restart fixes come later.
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

**Memory, routing, distillation, compression**
- **Facts in the weights** (LFM3): gate clean, recall `87%` dev (chance `11%`); layer 0 is best; more steps do not help, the gap looks like generalization from one sentence per fact.
- **MoI Weight & Memory Streaming CDN**: `19.5 KB` shards, ~3 ms attach.
- **Two `.moipatch` files at once**: no interference seen (weak test, no shared matrix).
- **Router false hits**: negative; no arm is a safe cache for "same problem, new numbers".
- **Reasoning-to-MoI Projective Distillation**: routing is invariant, projective generation is negative.
- **Pocket MoE Ceiling** / **Pocket MoE Controls**: 4B -> 0.8B; WSI ahead of random layers at ~1.7 MB (`91.7%` vs `80.6%`, CI includes 0); at 27 MB one unrouted LoRA wins.
- **Real-World `MATH-500` + `MMLU` Pocket MoE**: negative for WSI and splitting; the Layer-12 router does shield from forgetting (`0.0%` fires off-domain).
- **Pocket LoopRes T-sweep** (`Qwen3-4B -> 0.6B`, K=6): fix2x T2/T3 `71.7%` vs base `38.3%`, ties 1-pass LoRA `65%` (CI includes 0); variable-T training never helps; the earlier `80%` at 3x did not replicate.
- **WSI 2-Bit Quantization Recovery** and **Global Fisher Heterogeneous MLP Pruning**: negative (2-bit stays at chance; global Fisher starves late layers).
- **Pairwise neuron selection**: tie with `energy+recon`.
- **Low-bit trace residuals**: negative; a QTIP-style trellis beats scalar at 3 bits (relL2 `0.14` vs `0.18`) but INT8 quality needs >= ~7.5 bits (residual ~Gaussian, near full rank). Keep `int8+had64`.
- **Past the trace bit limit**: recompute from tokens in fp32 is exact (6,300x smaller; bf16 recompute is worse than INT8). 3-4 bit residuals keep layer/direction conclusions, lose onset and logit lens.
- **Recompute GPU vs CPU**: fp32 GPU recompute matches CPU to `1e-6` and is deterministic, so traces can be stored as tokens; fp16/fla stay < INT8 error (0.2-0.3x); bf16 is 2.2x INT8: avoid.

## 3. Planned Next Experiment (Queued — Do Not Run Yet)

- **Blind Autonomous Agent + WSI MCP (No 189-word brief):** Give an LLM agent **zero recipe/method hints** (same open-ended setup as MIT CSAIL `p50c` / `wsr program`) — only a model exhibiting a failure mode (e.g. thinking loops or sycophancy) and access to the **15 WSI MCP tools** (`wsi_discover_compact_circuit`, `wsi_build_moipatch`, `wsi_screen_candidates`, etc.), and measure turns, wall-clock time, and held-out accuracy vs. a raw-PyTorch agent baseline. Queued until `discover_math500_4b_colab.ipynb` finishes.
