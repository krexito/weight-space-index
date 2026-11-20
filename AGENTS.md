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

## 2. Key Recent Empirical Findings (2026-11-19)

- **`Qwen/Qwen3.5-0.8B` (`GSM8K`, `n=200`, `16k`):** `discover_blind` localized `L13-14_n16_a12` (32 neurons, 4× fewer weights than `p50c`) blindly; `blind_moi` reached **`55.5%`** (`89/0` vs `base` `11.0%`, statistically tying `p50c` `58.0%`; `random_moi` `32.0%`).
- **`Qwen/Qwen3.5-4B` (`GSM8K`, `n=200`, `16k`):** `base` is already at **`94.5%`** with only **`0.5%`** (`1/200`) `loop_killed` — GSM8K is saturated for `4B`. `fast_moi` reached `95.5%` (`2/0` vs `base`, `+4.0 pts` `10/2` over `fast_fixed` `91.5%`).
- **`Qwen/Qwen3.5-4B` (`MATH-500`):** the 64-problem pilot's **`39.1%`** strict `loop_killed` was **greedy decoding** (`temperature=0`, no presence penalty). With the sampling every run uses (`lr.SAMPLING`: T 0.6, top_p 0.95, top_k 20, `presence_penalty=1.5`) the 4B loops on **`0.7%`** (1/137 at `16k`; base `78.1%`, 16% truncated but not repeating), so `discover_math500_blind.py` found nothing to fix (wsi MoI `78.1%` = base, random MoI `81.8%`). Check the loop rate **with the run's own sampling** before choosing a benchmark.
- **`Qwen/Qwen3.5-0.8B` (`MATH-500`, 2026-11-20):** with the same sampling it loops hard on **`85%`** (base `10%` on 200; `2.7%` on the 150-problem exam). Blind discovery picked a weak single-layer edit (all 2-layer configs failed the `0.05` NLL guard): always on `10.7%` (+8 vs base, +5.3 vs random direction), via MoI `15.3%` vs random direction via MoI `18.7%` (tie: the MoI gain is the mid-loop switch-on, not the direction). Use a `16k` dev split on MATH-500 (half the answers hit `8k`).
- **Perpendicular loop direction (healthy subspace projected out, k 1/4/16): no gain.** The screen kept the plain direction; the healthy-shared part of `u` carries the loop suppression and 2-layer damage is unchanged. Plain direction at alpha 16 on the 0.8B: always on `13.3%` (beats random `6.7%`), MoI `15.3%` = random MoI.
- **Speculative decoding (`WSI_SPEC=mtp`):** works with the early loop stop and keeps results, but is **`0.80x`** (slower) on full batches (150 concurrent); the `1.52x` was a 32-prompt benchmark. Leave it off for batch evaluation.
- **Restart-only control (`scripts/restart_control.py`, 0.8B, 2026-11-20):** resuming the router's cut with the UNPATCHED base already gives most of a weak patch's MoI gain. GSM8K: base `11%` / restart `35.5%` / random MoI `32%` / patch MoI **`55.5%`** (patch vs restart `+20` pts, CI [+14, +26.5], 44/4; loops `26%` vs `59.5%`). MATH-500 (weak 1-layer patch): restart `17.3%` / random MoI `18.7%` / patch MoI `15.3%` (tie). **Always report MoI against the restart arm**, not base.
- **Pairwise neuron selection in accuracy (`obs+recon` vs `energy+recon`, LFM3-1.7B, GSM8K n=400, paired, 2026-11-20): tie.** `82.0/68.75/37.75%` vs `80.25/66.0/36.5%` at 70/50/30% kept (+1.75/+2.75/+1.25 pts, every 95% CI includes 0; full `84.75%`). The Qwen perplexity win did not carry over (LFM3 math ppl slightly worse). Keep `energy+recon` as default.
- **Base-draft speculative (`bench_base_draft_spec.py`, 0.8B, L4, 2026-11-20): dropped.** The unpatched model drafting for the p50c-patched one gives at most `1.15x` in PyTorch (prefix+suffix; sampling `0.89x`) and only `1/16` greedy outputs match the reference (run on `ebff0f9`, before the cache fix `e4e6785`). In vLLM (96 prompts, 512 tokens) `mtp` `1.09x`, `ngram` `0.40x`, `suffix` `0.34x` vs no speculation.
- **Restart-only control (`restart_control.py`, 0.8B, 2026-11-20):** resuming the router's cut answers with the *unpatched* model is a strong arm on its own: GSM8K `35.5%` (base `11.0%`), MATH-500 `17.3%` (base `2.7%`). GSM8K: the blind direction adds **`+20.0 pts`** over restart (`55.5%`, CI `[+14, +26.5]`; random MoI `32.0%` = restart). MATH-500: the edit ties restart (`15.3%` vs `17.3%`, CI `[-8.7, +4.7]`) — there the gain is the restart, not the weights.
- **Answer rescue (`rescue_answer.py` / `rescue_policy.py`, CPU over saved outputs, 0.8B, 2026-11-20):** loop-killed answers are scored wrong, but the model often stated the answer in the thinking before looping (none had closed `</think>`). Taking the last explicit answer (`\boxed{}` / "answer is X") **before the loop onset** (kill − 768) can only raise accuracy: GSM8K base `11.0→33.5%`, restart `35.5→48.5%`, blind MoI `55.5→61.5%` (p50c `58.0→60.5%`); MATH-500 base `2.7→17.3%`, restart `17.3→27.3%`. The current extractor (last number / last `\boxed`, taken inside the loop) recovers almost none (MATH base +1). Rescue stacks with restart (paired, both rescued: GSM8K +15, MATH +10 pts) and leaves the restart-control verdicts unchanged (GSM8K blind MoI vs restart `+13` `[+7, +19]`; MATH wsi MoI vs restart `-4` `[-11, +3]`). Stopping at the router fire when an answer is already stated saves 6–25% tokens but costs 1–6 pts on the continued arms; rescue at the kill instead. Precision of the rescued answer is ~60%, ~70–77% when the strict and most-frequent answers agree. Add rescue as a final grading/serving step and report it next to the strict number.
- **`#7 Reconstruction` A/B (`Qwen/Qwen3.5-0.8B`, `GSM8K`, `n=200`, `16k`, same 189-word brief, zero repo hints):**
  - **Agent A (Raw PyTorch, `7.7 min`):** `11.5%` always-on / `44.0%` loop-routed (`64/0` vs `base` `12.0%`; picked `L10-14`).
  - **Agent B (WSI toolkit, `8.4 min`):** **`35.5%`** always-on (`+24.0 pts` vs Agent A) / **`50.5%`** loop-routed (`77/0` vs `base`, `95% CI [+32%, +45%]`; `6.5s` `screen_patch_candidates` ranked `L13-14` `#1` at `pareto=2.75` vs `L10-14` `1.12`; exported `17.7 KB` `agent_b.moipatch`).

## 3. Planned Next Experiment (Queued — Do Not Run Yet)

- **Blind Autonomous Agent + WSI MCP (No 189-word brief):** Give an LLM agent **zero recipe/method hints** (same open-ended setup as MIT CSAIL `p50c` / `wsr program`) — only a model exhibiting a failure mode (e.g. thinking loops or sycophancy) and access to the **15 WSI MCP tools** (`wsi_discover_compact_circuit`, `wsi_build_moipatch`, `wsi_screen_candidates`, etc.), and measure turns, wall-clock time, and held-out accuracy vs. a raw-PyTorch agent baseline. Queued until `discover_math500_4b_colab.ipynb` finishes.
