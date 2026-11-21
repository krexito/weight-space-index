# arXiv reading notes (2026-11-21)

13 papers read in full text (arXiv HTML; a few appendix sections only through a summary, marked in the per-paper notes)
and checked against WSI's open problems. Numbers are the papers' own claims, not reproduced here.

## What to do with it (priority order)

1. **Over-refusal style shift (Aion v2: 41% exam-solver openings).**
   - Constrain the edit to the null space of ordinary-prompt activations (AlphaSteer): `ΔW·P̂ = 0` with `P̂` from the
     smallest-eigenvalue directions of benign activations at each edited layer. Ordinary prompts then see no change by
     construction; check it on every generated token, not only the last prompt token (AlphaSteer only shows the latter).
   - Clean the refusal direction before removing it (Surgical Refusal Ablation): ridge-residualize it against
     "protected" atoms (math, code, logic, the base model's own answer-style direction, an exam-template atom), sweep the
     ridge lambda against `new_template_rate` and harmful refusals kept.
   - Report the suppression ratio (over-refusal drop / harmful-refusal drop, aim well above 1) next to the three gates.
2. **LFM3 memory under-convergence (57%, final NLL 2.66).** MEMOIR's write recipe: SGD lr 1.0, grad clip 1.0, ~70 steps,
   10 random-prefix augmentations per edit; accumulate the update in fp32 (LOKI). WISE and MEMOIR write in mid-to-late
   layers and report that early layers fail; our sweep only covered L2-L10 of 24, so sweep the full depth once the
   writes converge.
3. **Sycophancy on LFM3 (caves 98%).** First adopt the validation protocol of the "no capitulation direction" paper
   (question-grouped CV, within-question label-shuffle null, a known positive control; in-sample AUROC 0.81 fell to 0.53
   under CV). Then try contrastive weight steering (`w = τ(caves) − τ(holds)` from two small LoRA runs, apply `k < 0`,
   |k| up to ~30): this is backlog item #13 (patch from a fine-tune).
4. **Loops.** Add the SOPHIA hit-rate (share of post-trigger steps that leave the loop) next to accuracy; try a CUSUM
   alarm on the existing probe score at matched FPR; stratify everything by decoding (LoopBench: DS-Qwen-7B loops 92% at
   T 0.1, 19% at T 1.0). Neither paper compares against a restart of the cut answer: our restart control is a real
   differentiator, keep it.
5. **Agent benchmarks (rigor reviewers will ask for).** From iWSP, WeightCode and the EEG/ECG paper: visible-to-hidden
   gap per run, two or more seeds, cost table at list prices, a matched-size random-edit placebo, fixed success rules
   decided before the run, a retention rule (install a patch only if the clean drop on a check split is under a fixed
   bound, else leave the model unchanged), the rebuild-from-brief test with a no-description control, and ship the
   generating program next to each `.moipatch` (fixed edits do not transfer across independently trained checkpoints).

## Per paper

### Loops
- **Circular Reasoning / LoopBench** ([2601.05693](https://arxiv.org/abs/2601.05693)). 700 synthetic items; statement
  loops common in reasoning models, rare in base/instruct. CUSUM alarm on hidden states: early detection 0.64-0.76 at FPR
  0.24-0.34, ~40 sentences / 1,500 tokens ahead, greedy only. "Semantic circularity precedes text" is shown only
  qualitatively; no comparison to an n-gram detector. Intervention is a "conclude now" injection, not a restart.
- **SOPHIA** ([2607.18100](https://arxiv.org/abs/2607.18100)). K-means (K=5) on step embeddings, loop = two consecutive
  steps in the same cluster; per-cluster steering vector = crossers minus stayers, `h += α‖h‖v̂`, α 0.05 (Qwen) /
  0.025 (Gemma; higher gives token loops). Reports cluster-exit hit rates on 30 prefixes, no task accuracy, no
  restart baseline, no code.

### Over-refusal
- **Surgical Refusal Ablation** ([2601.08489](https://arxiv.org/abs/2601.08489)). Diff-of-means refusal direction
  ridge-residualized against protected concept atoms, then rank-one projection on output weights (layers 15-25).
  KL to base 10-50x lower than raw ablation. It removes all refusals (0-2% harmful refused) and states it is not a
  safety method; we only take the cleaning step, not the goal. No voice/fluency measure.
- **Over-Refusal and Representation Subspaces** ([2603.27518](https://arxiv.org/abs/2603.27518)). Over-refusal vs
  harmful-refusal direction cosine ~0.47 (partial overlap). Task-conditioned steering: over-refusal 55% -> 0% with
  harmful refusal 65% -> 30%; global ablation 55% -> 25% / 65% -> 20%. Very small n (48 / 25), Llama-3.1-8B only.
- **AlphaSteer** ([2506.07022](https://arxiv.org/abs/2506.07022), ICLR 2026). Null-space projection from 14k benign
  prompts plus closed-form ridge fit (α ~10) to a refusal vector; adds refusal on malicious prompts while XSTest
  compliance barely moves (92.4 -> 91.2) and utility holds. Guarantee breaks with small benign sets (3k prompts: 58%).

### Memory
- **MEMOIR** ([2506.07899](https://arxiv.org/abs/2506.07899)). Residual copy of one down_proj, top-k sparse masks,
  match-ratio gate (τ ~0.4). ZsRE 1,000 edits: 0.94 / 0.85 / 1.00 (rel/gen/loc); 15,000 edits still 0.88 avg. Write:
  SGD lr 1.0, clip 1.0, ~70 steps, 10 random prefixes.
- **WISE** ([2405.14768](https://arxiv.org/abs/2405.14768)). Side copy of an FFN value matrix, activation-difference
  router with margin loss, sharding + TIES merge. ZsRE 1,000: 0.77 / 0.72 / 1.00. Layer sweep: early (0, 1) and last
  layers fail; mid-to-late best.
- **LOKI** ([2606.19679](https://arxiv.org/abs/2606.19679)). No gate, no memory: project each update into the null
  space of the layer's weights, pick layers per edit. ZsRE 1,000 avg 0.89 but locality only 0.85-0.88 (gated methods
  ~1.00). Useful as an extra safety layer under our gate, not a replacement.

### Sycophancy
- **Steering with Weight Arithmetic** ([2511.05408](https://arxiv.org/abs/2511.05408), ICLR 2026). Two LoRA fine-tunes
  (r32, ~500-900 examples each), `w = τ+ − τ−`, apply `θ + k·w`. Beats activation steering on sycophancy (Qwen2.5-7B,
  1.5B); useful |k| 16-28. Capability checks narrow.
- **No usable capitulation direction** ([2609.17550](https://arxiv.org/abs/2609.17550)). Qwen2.5-1.5B and
  Llama-3.2-1B: diff-of-means direction AUROC 0.81 in-sample, 0.53-0.58 under question-grouped CV, at or below the
  shuffle null. Pushback effects flip sign between families. Substring grading underestimates caving by 18-24 pts.

### Weight-space programming
- **iWSP** ([2611.18114](https://arxiv.org/abs/2611.18114)). Claude Code agent writes weights, fixed evaluator,
  visible/hidden split. 26/26 algorithmic runs 100% hidden, median 166 parameters; 200-word rebuild reaches 100%
  (1.74x params vs 2,043x without a description). Caught a scan-edge artifact (visible 0.067 -> 0.104, hidden only
  0.032 -> 0.043). Cost ~$17,500 total.
- **WeightCode** ([2611.25476](https://arxiv.org/abs/2611.25476)). Frozen LLM writes a Python program that edits a
  trained net (gates on idle MLP units, rank-one edits); ≤20% weights, per-tensor norm cap, keep only if clean loss
  drops ≤0.5%. ImageNet ResNet-18 noise +8.8-17.6 vs best fine-tuning +8.7 with 64 labels. Rerunning the program
  transfers (83-110%), copying the fixed edit does not.
- **Mixed Signals: WSP for EEG/ECG** ([2611.13758](https://arxiv.org/abs/2611.13758)). Opus writes an EEG net from
  scratch: 0.834 balanced accuracy vs 0.806 for gradient descent. Critiques it answers: small networks only, results
  from running code rather than reading weights (no model inferred the task from weights alone), healthy networks
  cannot be improved. Lists the checks in item 5 above.
