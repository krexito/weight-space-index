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
- **`Qwen/Qwen3.5-4B` (`MATH-500` pilot, `n=64`, `16k`):** **`39.1%`** strict `loop_killed` (`73.4%` soft loop, `64.1%` approx exact match, across all difficulty levels 1–5) — **78× higher loop rate than GSM8K**. `MATH-500` is the primary benchmark for `4B` blind loop-fix discovery (`scripts/discover_math500_blind.py`, `notebooks/discover_math500_4b_colab.ipynb`).
