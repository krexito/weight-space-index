# What the findings say together (synthesis, 2026-11-26)

Read with `research/findings.md` (details) and the AGENTS.md index. Use this file to pick the next test: each
pattern names the evidence and the test it points to. Update it when a result changes a pattern.

## Patterns

1. **Reading is easy; editing well is hard.**
   - Detection works: the blind direction detects loops at token AUC `0.977`, the same as the best fitted direction.
   - The sensor token is free in vLLM.
   - Routers fire `0/200` and `0/371` off-domain, and the veto cuts false alarms `12/23 -> 3/23`.
   - Editing is where it fails:
     - on MATH-500 the patch ties a random direction;
     - the 1-pass screen picks wrong;
     - the cut state does not predict rescue.
   - Detection is not the bottleneck; choosing the edit is.
2. **Cheap proxies mislead; only outcome-near tests select.**
   - The suppression screen does not predict the exam (Spearman `0.29-0.46`); the escape test does (`0.75-0.80`).
   - One sample per cut is 34% luck.
3. **Many wins shrink against the right control.**
   - Restart explains most of a weak patch's gain.
   - A random direction ties on MATH-500.
   - Answer rescue alone (no edit) takes GSM8K from `11%` to `33.5%`.
4. **The damage is in style, not benchmarks.**
   - GSM8K and MMLU do not move.
   - Exam-solver openings (41%), voice change and broken legitimate repetition show up instead.
   - The blind agent passed the metrics and failed with style.
5. **WSI removes or redirects existing behavior; it does not add capability.**
   - Works:
     - over-refusal;
     - sycophancy gate on the 4B (`+25`);
     - loops on the 0.8B.
   - Fails:
     - facts from one sentence;
     - ridge on GDN;
     - loopification;
     - think compression.
   - MoLT needs 100B tokens.

## Results since

- **Test 1 done (2026-11-26), negative for internals:** on Qwen2.5-1.5B MATH-500, probes predict a correct answer
  (AUC `0.84`) no better than the model's own confidence + length fitted together (`0.83`). Refines pattern 1: reading
  is easy, but for correctness it reads what the token probabilities already say. Pattern 3 again: the win vs one
  control vanished vs the combined control.
- **Test 2 done (2026-11-26), positive:** the LFM3 refusal patch survives the random-direction control on fresh
  prompts (recovered `70.5%` vs `0-8.5%`, toxic kept `97.0%`) and beats a system prompt (`41.1%`, `91.6%`). Refines
  pattern 3: the shrinkage is real for weak loop patches, not for the refusal edit. Refusal is where WSI's edit is
  strongest, so test 3 (escape-style selection with style) builds on solid ground.
- **Test 3 done (2026-11-26), positive with caveats:** on LFM3 refusal, a 64-token escape test ranks 20 grid edits like
  the exam (Spearman `0.85` vs `0.32` for the KL-penalized screen; plain suppression `0.77`). Pattern 2 holds beyond
  loops. Not tested: style failures (none occurred) and selection without the router (it did the safety work).
- **Test 4 done (2026-11-26), positive:** two routed patches on the same LFM3 matrices compose exactly (no loss for
  either; over-refusal `53.7 -> 20%` and toxic refusal `83 -> 91%` together); always on, one swamps the other. Modular
  scales when every piece is gated. Open: two independent (overlapping) gates.

## Tests these point to (priority order)

1. **Read to improve without editing** (patterns 1, 5; CPU, $0). Does a probe or sensor predict whether a *normal*
   (non-looping) answer is correct, beating the model's own confidence? If yes: abstain or retry, like answer
   rescue but general.
2. **Controls for the refusal edit** (pattern 3; ~$0.5): random direction, same alpha, same router.
3. **Escape-style selection for refusal** (patterns 2, 4; ~$1): continue from the refusal onset; score recovery and
   style, not proxies.
4. **Two patches on the same matrix** (pattern 5: does modular scale; ~$0.5).

Down-weighted:
- **MoLT-style KV memory without training:** it conflicts with pattern 5.
- **Blind agent + WSI MCP:** run it after test 3, since the agent inherits broken selection.
