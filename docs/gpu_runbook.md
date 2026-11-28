# GPU & Modal Credit-Protection Runbook (`WEIGHT-SPACE-INDEX`)

Apply this checklist **before** launching any Modal or Colab GPU run:

1. **Pick the cheapest GPU that fits the model:**
   - Use **`L4` (24 GB, ~$0.80/h)** for `0.6B–4B` models; reserve **`A100` (~$2.50/h)** only for `>=14B` models or workloads that genuinely exceed 24 GB VRAM.
2. **Size `batch_size` to VRAM first:**
   - Estimate `weights_gb + batch * max_len * kv_kb_per_tok` and fill ~60–80% of VRAM.
   - Example: `Qwen/Qwen3.5-0.8B` is `1.6 GB` in BF16 and has only 6 full-attention layers (`~12 KB/tok` KV) → use `batch_size=64–100`, never `batch_size=16`.
   - **Never use `output_scores=True` in `model.generate()`** on large batches/vocabularies (stores `batch × steps × vocab × 4B` in VRAM and causes late OOMs); generate with `output_scores=False` and score log-probs in a lightweight forward pass if needed.
3. **Per-arm checkpointing + periodic `vol.commit()`:**
   - Save every completed arm's outputs/metrics to `/vol/<run>/ckpt/` immediately via atomic `.tmp` + `replace`.
   - Run a background `vol.commit()` thread (every 60–120 s) so a stop/crash never loses completed GPU work.
4. **Log per-batch speed (`i/N`, `elapsed_s`, `tok/s`):**
   - Never run a 10+ min generation loop silently; print progress after every batch so slow fallbacks are caught in the first 30 seconds.
5. **Verify fast kernels & caches:**
   - Confirm `flash-linear-attention` / `vLLM` / FlashAttn are active before long autoregressive sweeps.
   - Reuse cached models (`/vol/models/...`) and teacher traces (`/vol/.../*.pt`) on the Modal volume whenever available.
6. **Smoke test with PRODUCTION `batch_size` (never `batch=8`) and short `max_new_tokens=16`:**
   - Testing full production `batch_size` with 16 tokens catches VRAM/OOM bugs in ~20 seconds (~$0.01) instead of failing 40 minutes into a full run.
