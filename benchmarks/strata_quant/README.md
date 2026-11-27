# Strata-Inspired Quantization & Pocket MoE Compression (`wsi/strata_quant.py`)

Self-contained CPU/GPU quantization toolkit for **WEIGHT-SPACE-INDEX**, combining wire-exact `S_K` super-block accounting (`QK_S=256`, `SUB_G=64`: `S2_K..S8_K`), 64-wide Block-Hadamard rotation (`H64`), Hessian-weighted clipping, in-HBM Block-LDL (GPTQ) + analytical 64-flip refinement, global Hessian knapsack bit allocation, zero-GPU **X-Ray quantization sensitivity prediction**, and `+0 B` `Scale-QAT` over 1D RMSNorms.

---

## API & Files

* **Core library:** [`wsi/strata_quant.py`](../../wsi/strata_quant.py)
  * Wire accounting: `SUPERBLOCK_BYTES`, `matrix_wire_bytes`, `code_bits_per_weight` (`S2_K=2.1875 b/w` .. `S8_K=8.1875 b/w`)
  * Transform & solvers: `apply_h64_cols`, `apply_h64_sym`, `damped_cholesky_inv`, `rtn_quantize_sk_torch`, `gptq_flip_quantize_sk_torch`
  * Mixed-precision & X-Ray predictor: `solve_knapsack_bit_allocation`, `build_quant_sensitivity_map`
  * `+0 B` norm calibration: `capture_teacher_topk_targets`, `eval_topk_kl_and_top1`, `run_norm_scale_qat`
* **X-Ray Atlas integration:** [`wsi/xray.py`](../../wsi/xray.py) stores compact 8-bin sensitivity `card["anatomy"]["fisher_profile"]["q_sens_8"]` during `scan_model_xray()`.
* **Unit tests:** [`tests/test_strata_quant.py`](../../tests/test_strata_quant.py) + [`tests/test_xray.py`](../../tests/test_xray.py) (`27 passed` on CPU):

```bash
python -m pytest -q tests/test_xray.py tests/test_strata_quant.py
```

Quick usage example:

```python
import torch
from wsi.strata_quant import gptq_flip_quantize_sk_torch, build_quant_sensitivity_map

w_q, wire_bytes, hess_err = gptq_flip_quantize_sk_torch(
    W, H, code_name="S3_K", use_h64=True, max_flips=64
)
qmap = build_quant_sensitivity_map(xray_card)  # {"per_layer_scores", "protected_layers", "compressible_valley", ...}
```

---

## Experiment Suite Summary (6 Runs, All Paired & Logged)

All raw outputs live in `benchmarks/strata_quant/`, with both Modal launchers under `cloud/` and reproducible CPU `--smoke` modes in every script.

### 1. Exp 1 — Pocket MoE `r=8` Cartridge Compression (`K=6`, Modal A10G `77.6 min`)
* **Script / Launcher:** [`scripts/experiment_pocket_moe_strata_quant.py`](../../scripts/experiment_pocket_moe_strata_quant.py) | [`cloud/modal_pocket_moe_strata_quant.py`](../../cloud/modal_pocket_moe_strata_quant.py)
* **Result JSON:** [`results_pocket_moe_strata_quant_modal_a10g.json`](./results_pocket_moe_strata_quant_modal_a10g.json)
* **Setup:** `Qwen/Qwen3.5-4B -> Qwen/Qwen3.5-0.8B`, `5` skills (`K=6` shots), `60` blind hard items + `24` guardrails + `200` MMLU + `16` prose, Layer-12 router (`hard_acc=1.0`, `guardrail=95.8%`, `mmlu_pass=100%`, `prose_fpr=0.0`).

| Arm | MB / Skill | Hard-60 (`n=60`) | Old 1–4 / s5 | Guard-24 | MMLU-200 Fires | Paired Bootstrap vs Comparator (`IC95`) |
|---|---:|---:|---:|---:|---:|---|
| `base_08b` | ref | `8.3%` (`5/60`) | `10.4% / 0.0%` | `91.7%` | — | — |
| `teacher_4b` | ref | `100.0%` (`60/60`) | `100% / 100%` | `100.0%` | — | — |
| `oracle_1shot_08b` | ref | `76.7%` (`46/60`) | `83.3% / 50.0%` | `100.0%` | — | — |
| `r8_fp16_routed` | `10.32 MB` | `70.0%` (`42/60`) | `72.9% / 58.3%` | `91.7%` | `0/200` | — |
| `r8_s3k_h64_gptq_flip_routed` | `2.08 MB` (`5.0×`) | `71.7%` (`43/60`) | `75.0% / 58.3%` | `91.7%` | `0/200` | `+1.7` vs `r8_fp16` `[-10.0, +13.3]`; `+5.0` vs `r2` `[-8.3, +18.3]` |
| **`r8_s2k_h64_gptq_flip_routed`** | **`1.43 MB` (`7.2×`)** | **`73.3%` (`44/60`)** | `77.1% / 58.3%` | `91.7%` | **`0/200`** | **`+21.7 pts`** vs `r1_fp16` `[+6.7, +36.7]` (`19w/6l`); `+6.7` vs `r2` `[-6.7, +20.0]` |
| `r8_rtn_2bit_routed` | `1.43 MB` | `73.3%` (`44/60`) | `77.1% / 58.3%` | `91.7%` | `0/200` | Ties `S2_K` (`11w/11l`, `[-15.0, +15.0]`) despite `6.0–7.1×` higher H-err |
| `r2_fp16_lr4e4_routed` | `2.58 MB` | `66.7%` (`40/60`) | `68.8% / 58.3%` | `91.7%` | `0/200` | — |
| `r1_fp16_routed` | `1.29 MB` | `51.7%` (`31/60`) | `58.3% / 25.0%` | `91.7%` | `0/200` | — |

---

### 2. Exp 2A — Dense `Qwen/Qwen2.5-3B-Instruct` (`2.65–3.19 b/w`, Modal A100 `18.6 min`)
* **Script / Launcher:** [`scripts/experiment_xray_strata_quant_dense.py`](../../scripts/experiment_xray_strata_quant_dense.py) | [`cloud/modal_xray_strata_quant_dense.py`](../../cloud/modal_xray_strata_quant_dense.py)
* **Result JSON:** [`results_xray_strata_quant_qwen25_3b_modal_a100.json`](./results_xray_strata_quant_qwen25_3b_modal_a100.json)

| Arm | b/w | Wire MB | WikiText-2 PPL ↓ | Top-256 KL ↓ | Top-1 ↑ | MMLU-200 ↑ | GSM8K-100 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_bf16` | `16.000` | — | `10.325` | `0.0000` | `100.00%` | `58.5%` | `83.0%` |
| `uniform_rtn_3b` | `3.188` | `1054.27` | `554.167` | `4.0471` | `20.26%` | `29.5%` | `0.0%` |
| **`uniform_s3k_h64_gptq_flip`** | **`3.188`** | **`1054.27`** | **`11.788`** | **`0.1525`** | **`81.04%`** | **`56.0%`** | **`70.0%`** |
| `xray_knapsack_3_19b` | `3.188` | `1054.27` | `12.255` | `0.2247` | `77.94%` | `49.0%` | `57.0%` |
| `xray_knapsack_3_19b_scale_qat` | `3.188` (`+0 B`) | `1054.27` | `12.140` | `0.2208` | `78.14%` | `50.0%` | `61.0%` |
| `xray_knapsack_2_65b_scale_qat` | `2.650` (`+0 B`) | `876.45` | `13.972` | `0.3598` | `72.40%` | `40.0%` | `33.0%` |

---

### 3. Exp 2B — Sparse MoE `allenai/OLMoE-1B-7B-0125-Instruct` (`1,024` Experts, Modal A100 `35.9 min`)
* **Script / Launcher:** [`scripts/experiment_xray_strata_quant_moe.py`](../../scripts/experiment_xray_strata_quant_moe.py) | [`cloud/modal_xray_strata_quant_moe.py`](../../cloud/modal_xray_strata_quant_moe.py)
* **Result JSON:** [`results_xray_strata_quant_olmoe_modal_a100.json`](./results_xray_strata_quant_olmoe_modal_a100.json)

| Arm | Expert b/w | Expert MB | WikiText-2 PPL ↓ | Top-256 KL ↓ | Top-1 ↑ | MMLU-200 ↑ | ARC-100 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_bf16` | `16.000` | — | `13.043` | `0.0000` | `100.00%` | `48.0%` | `59.0%` |
| `uniform_rtn_3_2` | `2.854` | `2192.0` | `51.949` | `1.8155` | `45.75%` | `33.5%` | `48.0%` |
| `uniform_s3_s2_h64_gptq_flip` | `2.854` | `2192.0` | `18.772` | `0.6780` | `64.33%` | `28.5%` | `50.0%` |
| `per_expert_knapsack_2_85b` | `2.854` | `2192.0` | `19.078` | `0.7074` | `63.07%` | `24.5%` | `46.0%` |
| **`per_expert_knapsack_2_85b_scale_qat`** | **`2.854` (`+0 B`)** | **`2192.0`** | **`17.140`** | **`0.6445`** | `64.28%` | `24.5%` | `44.0%` |
| `per_expert_knapsack_2_45b_scale_qat` | `2.450` (`+0 B`) | `1881.5` | `22.718` | `1.0370` | `56.36%` | `23.0%` | `41.0%` |

---

### 4. Exp 3 — Offline X-Ray Quantization Predictor Validation (CPU, `$0`)
* **Script / Result:** [`scripts/validate_xray_quant_predictor.py`](../../scripts/validate_xray_quant_predictor.py) | [`results_xray_quant_predictor_validation_cpu.json`](./results_xray_quant_predictor_validation_cpu.json)
* **`Inkling-2-Mini`:** Protected `[0, 11, 17, 23]` hit `2/4` in `SQ-10` top-5 (`[17, 16, 15, 10, 11]`), `Spearman +0.6490` vs `SQ-10` layer MB (`0.5907 vs 0.2812` loop+tail vs prologue matching `374.96 vs 308.41 MB`), `-0.3133` vs `SQ-15`.
* **`Qwen2.5-3B`:** `Spearman +0.8167` vs `knapsack_3_19b_avg_bits` (`+0.3075/+0.3065` vs raw `S2/S3` Hessian error); protected `[0, 2, 31, 35]` `3.6161 b/w` vs valley `3.0446 b/w`.
* **`OLMoE-1B-7B`:** `Spearman +0.6760` vs `knapsack_2_85b_mb` (`+0.5441` vs raw `S2` err), exposing that unnormalized PreNorm `tr(ΔW H ΔW^T)` grows `~140×` from `L1` (`0.000643`) to `L15` (`0.090649`) and starves `L1–L2` (`110.1 MB`) in naive knapsacks.

---

### 5. Test A — High-Rank LoRA (`r=8/32/64`) Compressed to `S2_K` on Real `MATH-500` (Modal A100 `73.3 min`)
* **Script / Launcher:** [`scripts/experiment_pocket_moe_math500_strata_quant.py`](../../scripts/experiment_pocket_moe_math500_strata_quant.py) | [`cloud/modal_pocket_moe_math500_strata_quant.py`](../../cloud/modal_pocket_moe_math500_strata_quant.py)
* **Result JSON:** [`results_pocket_moe_math500_strata_quant_modal_a100.json`](./results_pocket_moe_math500_strata_quant_modal_a100.json)
* **Setup:** `Qwen/Qwen3.5-4B -> Qwen/Qwen3.5-0.8B`, `195/300` kept real 4B traces (`0` fallbacks), `n_test=200`, `MMLU-400` (`371` non-math), equal pass budget (`160×4=640`).

| Arm | MB | MATH-500 (`n=200`) ↑ | Trunc ↓ | MMLU (`400`) | Non-Math MMLU Fires | Prose KL |
|---|---:|---:|---:|---:|---:|---:|
| `base_08b` | ref | `31.0%` (`62/200`) | `42.5%` | `52.0%` | `0/371` | `0.0000` |
| `teacher_4b` | ref | `64.5%` (`129/200`) | `28.0%` | — | — | — |
| **`r8_fp16_routed`** | `10.32 MB` | **`35.5%` (`71/200`)** | `41.0%` | **`52.2%`** | **`0/371`** | `0.0000` |
| `r32_fp16_routed` | `41.29 MB` | `29.5%` (`59/200`) | `39.5%` | `51.7%` | `0/371` | `0.0000` |
| `r32_s2k_routed` | `5.72 MB` (`7.2×`) | `31.5%` (`63/200`) | `38.0%` | `52.0%` | `0/371` | `0.0000` |
| `r64_fp16_routed` | `82.57 MB` | `28.0%` (`56/200`) | `36.0%` | `51.7%` | `0/371` | `0.0000` |
| `r64_s2k_routed` | `11.43 MB` (`7.2×`) | `30.5%` (`61/200`) | `41.0%` | `51.7%` | `0/371` | `0.0000` |
| `r64_rtn2_routed` | `11.43 MB` | `26.5%` (`53/200`) | `31.5%` | `51.7%` | `0/371` | `0.0000` |
| `r64_s2k_unrouted` (ctrl) | `11.43 MB` | — | — | `48.0%` (`ABCD 46.7%`) | `371/371` | `0.0339` |

* **Takeaway (negative on rank inflation):** Inflating to `r=64` on `195` traces overfits (`-7.5 pts` vs `r=8`, `CI95=[-13.0, -2.0]`), whereas `7.2×` `S2_K` compression preserves FP16 accuracy (`+2.0`/`+2.5 pts`), beats naive `RTN-2bit` by `+4.0 pts` (`H-err 168.9 vs 623.8`), and L12 routing leaves `0/371` non-math MMLU untouched.

---

### 6. Test B — Strata 3-Bit Base (`1.05 GB`) + `Scale-QAT` (`+0 B`) + Routed `S2_K` Patch (`Qwen2.5-3B-Instruct`, Modal A100 `37.3 min`)
* **Script / Launcher:** [`scripts/experiment_strata_patch_recovery_dense.py`](../../scripts/experiment_strata_patch_recovery_dense.py) | [`cloud/modal_strata_patch_recovery_dense.py`](../../cloud/modal_strata_patch_recovery_dense.py)
* **Result JSON:** [`results_strata_patch_recovery_qwen25_3b_modal_a100.json`](./results_strata_patch_recovery_qwen25_3b_modal_a100.json)
* **Setup:** Uniform `S3_K` (`1054.27 MB`, `3.188 b/w`) + `+0 B Scale-QAT` + self-distilled LoRAs (`213/240` BF16 GSM8K-train traces) + Layer-18 router fit on quantized base (`100%` GSM8K, `0/200` MMLU, `0/16` prose, `0%` WikiText-2).

| Arm | Extra | PPL ↓ | KL ↓ | MMLU-200 ↑ | GSM8K-200 ↑ | GSM8K-100 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `base_bf16` (`5.55 GB`) | — | `10.325` | `0.0000` | `58.5%` | `81.5%` (`163/200`) | `83.0%` |
| `s3k_uniform_raw` (`1,054 MB`) | `+0.00 MB` | `11.855` | `0.1543` | `54.5%` | `75.5%` (`151/200`) | `76.0%` |
| **`s3k_plus_scale_qat`** | **`+0.00 MB`** | **`11.725`** | **`0.1509`** | **`54.5%`** | **`76.5%` (`153/200`)** | **`77.0%`** |
| `s3k_sqat + r4_fp16_routed` | `+14.27 MB` | `11.725` | `0.1509` | `54.5%` | `71.5%` (`143/200`) | `73.0%` |
| `s3k_sqat + r4_s2k_routed` | `+1.95 MB` (`7.3×`) | `11.725` | `0.1509` | `54.5%` | `71.5%` (`143/200`) | `74.0%` |
| `s3k_sqat + r4_s2k_unrouted` | `+1.95 MB` | `11.857` | `0.1553` | `53.5%` | `71.5%` (`143/200`) | `74.0%` |
| `s3k_sqat + r4_rtn2_routed` | `+1.95 MB` | `11.725` | `0.1509` | `54.5%` | `70.5%` (`141/200`) | `69.0%` |
| `s3k_sqat + r8_s2k_routed` | `+3.90 MB` | `11.725` | `0.1509` | `54.5%` | `71.0%` (`142/200`) | `71.0%` |

---

## Core Engineering Takeaways
1. **Pocket MoE cartridges:** Train `r=8` and pack into `S2_K` (`1.43 MB`, `7.2×` smaller) — beats narrow `r=1`/`r=2` FP16 by `+21.7`/`+6.7 pts` with `0/200` off-domain fires. Reserve `r=32/64 + S2_K` for large multi-thousand-trace datasets (small ~200-trace pools overfit).
2. **Whole-model 3-bit Strata (`S3_K + Scale-QAT`):** `H64 + Hessian clipping + Block-LDL + 64-flip` + `+0 B Scale-QAT` rescues 3-bit RTN collapse (`PPL 554 -> 11.73`, `GSM8K 0% -> 77%`), and mid-layer routing works `100%/0%` directly on 3-bit weights.
3. **Knapsack normalization requirement:** PreNorm residual norms grow `~140×` with depth; cross-layer bit allocation must normalize Hessian error per layer (`tr(ΔW H ΔW^T) / tr(W H W^T)`) and keep dense projections `>= 3-bit`.
