# NOTES: routed refusal-direction ablation for over-refusal

## Method
**Edit:** directional ablation of a "refusal direction" `r`. The direction is
`mean(h | base refused) - mean(h | base answered)`, taken from last-token residual states. "Refused" is the refused
OR-Hard prompts plus the refused toxic prompts. "Answered" is the answered OR-Hard prompts plus the controls. `r`
is projected out of every residual writer in every layer (attention out-proj and MLP down-proj, 2 × 28 = 56 rank-1
edits). Each edit is `W' = W - a r rᵀ W`, which is exactly the `.moipatch` edit form. With `a = 1` nothing those
modules write has any component along `r`.

**Router:** the patch only switches on when the router fires, so toxic prompts and ordinary prompts keep running
on the unmodified base model. The router is fitted with `wsi.router_cal.fit_framing_invariant_safety_router`.
Fire = gate 1 (`h·dir > thr`: benign-refused vs controls) **and not** veto (`h·veto_dir > veto_thr`: toxic vs
benign-refused). Both are linear readouts of `hidden_states[L]` at the last prompt token. The fitter:
- picks the layer, regularisation and thresholds by Group K-fold out-of-fold (OOF) CV;
- penalises framing shifts, using the calibration wrappers (seminar, compliance, stage play, …) applied to the
  agent prompts.

This matters because OR-Bench prompts are full of hedges like "for a novel" or "for research", and the router must
not learn to key on those words.

## Why this design
- **Controls must match the base model:** at least 90% of control answers must be identical, at 256 tokens. No
  always-on weight edit can promise that. A routed patch can: rows where the router doesn't fire go through
  untouched (the edit hooks return the original output), so those answers are bit-identical. All the risk on
  controls comes down to router false positives, which is easy to control because math and trivia are far from
  "sensitive-sounding" in activation space.
- **Toxic refusals must stay:** at least 90% of base-refused toxic prompts must still be refused. The veto is
  trained specifically for this. Thresholds are then tightened in-sample to at most 2% toxic fires, and gate 1 is
  held above every raw control score plus a margin of 0.25 std.
- **Benign refusals must flip:** at least 50% of base-refused benign prompts must now be answered. Reaching that
  needs a router hit rate times an edit efficacy of at least 0.5. Refusal-direction ablation is known to remove
  most refusals when applied. The script measures efficacy with the edit forced on and picks the best candidate
  (direction layer at 40/50/60/70% of depth; `a` in {1, 1.5, 2} only if `a = 1` un-refuses less than 80%). The
  score penalises degenerate output.

## Pipeline and estimated L4 time (2.3B model, bf16, batch 16)
| stage | work | est. |
|---|---|---|
| load | bf16, `trust_remote_code` | 1–2 min |
| base screening | 300 prompts × 48 tokens, 30 controls × 256 tokens | ~1.5 min |
| hidden capture | about 330 raw + ≤1,900 wrapped prompts, one forward pass each, hooks on block outputs | ~1 min |
| router fit | 24 layers (4–27) × grid, fp64 on GPU, K = 4 | 1–3 min |
| edit search | ≤6 candidates × ≤128 prompts × 40 tokens, edit forced on | 2–4 min |
| validation | routed generation on the refused sets plus controls at 256 tokens | ~1.5 min |

Expected total: **about 8–15 min**, well under 45. The `--budget-min 38` soft deadline skips extra edit candidates
when time runs short.

## Decisions and details
- **Router features = what the runtime reads.** Hidden states are captured with forward hooks on the decoder block
  outputs, at the last non-pad token under left padding. That is exactly what `RoutedPatchHandle` reads for router
  layer `L` (the output of block `L-1`). The final layer is excluded, because there moipatch reads the
  *normed* `hidden_states[-1]`. Router layer > min edit layer (0), so at runtime moipatch runs a clean pre-probe:
  the router always sees base activations. The validation stage checks that the routed runtime's fire decisions
  match the script's own predictions (`router_prediction_agreement`, 1.0 in the CPU test).
- **Refusal detector:** phrase match on the first 160 characters, with curly quotes normalised. It is broad on
  purpose (it is used for screening).
- **Few refusals:** if the base refuses fewer than 12 of the 200 benign prompts, the router's positive class falls
  back to all 200 benign prompts. If it refuses none, the script ships a patch whose router never fires.
- **Fingerprints** are the SHA-256 of each touched weight in the loaded dtype (bf16), which matches the `attach`
  check.
- **`wsi/__init__.py` imports `wsr`**, which isn't installed. `agent_wsi.py` tries the normal import and otherwise
  registers a bare `wsi` namespace package, so that `wsi.moipatch` and `wsi.router_cal` (which need only torch)
  load anyway.
- **wsi functions used:** `MoIPatch`, `Edit`, `MoIPatch.save/load/enable_routed`, `_find_decoder_blocks`,
  `_last_token_hidden`, `_tensor_out`, `_sha256`, `module_suffix` (moipatch); `fit_framing_invariant_safety_router`,
  `apply_calibration_wrapper`, `CALIBRATION_FIT_WRAPPERS`, `CALIBRATION_VAL_WRAPPERS`,
  `compute_hedging_balanced_weights` (router_cal).

## CPU test (`test_agent_wsi_cpu.py`)
The test uses a tiny random 4-layer GQA `LlamaForCausalLM` (hidden 64) and a toy word-level tokenizer with a chat
template, plus 48/24/10 synthetic prompts. A random model never refuses, so the refusal detector is replaced by a
hash of the answer text. That exercises every branch, but the quality numbers it produces mean nothing. It checks:
- the patch and summary are written, and validation ran without error;
- the router has a veto;
- there is one edit per writer (o_proj and down_proj in every layer), with unit directions;
- the fingerprint attach works on a freshly loaded model;
- with the edit forced on, the writer outputs' component along `u` drops by about 600× (0.054 → 9e-5);
- the routed runtime's fire decisions match the script's predictions 100%;
- controls are bit-identical when the router doesn't fire;
- every weight is byte-identical after routed use.

Result: **PASS**.

Windows note: an empty `CUDA_VISIBLE_DEVICES=""` deletes the variable on Windows, so the test sets `-1` there and
asserts that CUDA is not visible.

## Risks
1. **Router generalisation** to held-out prompts is the main risk. It is fitted on about N_ref benign-refused and
   100 toxic prompts, in 2048 dimensions. The OOF CV and the framing penalty help, but a held-out toxic prompt
   phrased like an OR-Hard one can slip through the veto. Toxic retention then depends on the edit not fully
   un-refusing it.
2. **Too few base refusals** in the agent set means a weak gate-1 fit, so benign recall may land below 50%.
3. **Edit side effects:** answers on routed prompts change completely by design. Quality is only checked by a
   degeneracy heuristic and by eye (samples go into the summary JSON).
4. **Prompt formatting mismatch** with the judge (for example a double BOS, or a different system prompt) shifts
   activations slightly. The thresholds have margin, but not much.
5. **Unknown custom architecture:** writer modules are found by name (`o_proj`, `down_proj`, …, with an
   out-features fallback). Module names must contain `layers.N.`, otherwise moipatch itself cannot resolve them.
6. **bf16 batch effects:** base outputs depend slightly on padding and batch composition. The script's
   in-sample validation numbers are estimates, not the judge's numbers.
