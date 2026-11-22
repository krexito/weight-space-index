# Design: hidden-state router vs text matching for "same problem, new numbers" (2026-11-22)

Status: **run on 2026-11-22, failed the decision rule on both models** (see Result). Script `scripts/experiment_router_false_hits.py`. Origin: CPU-ARC ([2611.09891](https://arxiv.org/abs/2611.09891)) caches a
solved problem as code and finds it again by embedding similarity. It does not report false hits (a similar-looking
problem that needs another calculation), and a false hit returns a confidently wrong answer.

## Question

Given one stored problem, can a router on the model's hidden state accept rewordings and new numbers, and reject
near-identical text that needs another calculation, better than cheap text matchers, an embedding model, or simply
asking the model?

## What the CPU pilot says (GSM-Symbolic, 60 templates, 30 calibration / 30 test, one stored question per template)

Threshold = 95 % hit rate on calibration templates; numbers are on test templates. `flip` = one word swapped by a rule
(more/fewer, total/average...), available for 41 of 60 templates.

| matcher | hit | false hits: `p1` | `p2` | other template | `flip` (AUC) |
|---|---|---|---|---|---|
| TF-IDF, numbers masked | 97 % | 89 % | 87 % | 0 % | 95 % (0.49) |
| edit ratio, numbers masked | 98 % | 64 % | 51 % | 0 % | 98 % (0.52) |
| length ratio only | 93 % | 12 % | 0 % | 9 % | 96 % (0.47) |
| all text features, logistic (trained on calibration templates) | 97 % | 2 % | 0 % | 0 % | 99 % (0.59) |

1. **`p1` / `p2` are not a test of reasoning.** They add sentences, so length alone separates them. Beating a plain
   embedding on them would be an easy, meaningless win. They stay only as a sanity check.
2. **Same-length flips are the real test.** Every text matcher is close to chance.
3. **The rule-made flips are too noisy to use.** Several are nonsense ("in the removed 22 liters") or do not change the
   answer. They must be written by hand.
4. **Hits need rewordings.** With numbers-only hits, "any fixed word changed -> reject" would be perfect. So the benchmark
   has to separate a harmless rewording from a flip of similar size.

## Data

- 100 `main` templates from `apple/GSM-Symbolic`, 89 with clean, reviewed edits (11 dropped, reasons in `edits.json`); split **by
  template** 44 calibration / 45 test (seed 0), fixed before any model runs.
- Dataset license is CC BY-NC-ND and its text carries a canary: the repo holds only the short edit spans.
- Key: `main` instance 0. Queries per template: 10 instances of each kind.

| kind | label | how |
|---|---|---|
| `number` | hit | `main` instances 1-10 |
| `reword` | hit | one hand-written rewording per template (same words changed as its flip, plus nearby), applied to all instances |
| `flip` | miss | one hand-written minimal edit that changes the calculation, length within 5 % |
| `p1`, `p2` | miss | dataset variants (sanity only) |
| `other` | miss | another template (sanity only) |

- Edits are literal substitutions on the template's fixed words, so one edit covers all its instances.
- Checks on CPU before any GPU run: each edit matches in all its instances; length within 5 %; the number of edited
  words of `reword` and `flip` is balanced; a second reviewer, blind to the labels, marks "same calculation?" for each
  pair. Keep only agreements, report the agreement rate.
- **Go / no-go:** the text logistic and a bag-of-words classifier on the edited words, retrained on this set, must stay at AUC <= 0.75 on `reword` vs `flip`. If it does
  better, the edits leak a shortcut: rewrite them before spending GPU.

### Status of the edits (2026-11-22)

`benchmarks/router_false_hits/edits.json`, checker `scripts/build_router_false_hits.py`, gate
`scripts/router_false_hits_gate.py` (result in `gate.json`, 100 random half/half template splits):

- All edits (90 before review) match exactly once in instances 0-10 and stay within 5 % length; new words per edit: flip 1.43, reword 1.23.
- 89 templates after review. Text features: AUC 0.53 (chance). Edited words: **0.70** (5-95 %: 0.65-0.76). Both: 0.70.
  Shuffled labels 0.54. The regularisation sweep (1e-4 to 1, 90 templates) gives 0.69-0.71. **Go**, but close: "half" is added in 7 flips and no reword. So the
  edited-words classifier is an arm and the router must beat it, not only the embedding.
- Second reviewer (a Claude subagent on a label-free, shuffled sheet; its tool log shows it read nothing else): 179 / 180
  pairs agree, 1 unclear (template 24, dropped). **Upper bound**: same model family as the author, and both edits of a
  template were on the sheet. No human has checked them. Some flips give the same number by coincidence for some
  parameters (template 77 when the fraction is 1/2): the label is the calculation, not the number.

## Arms (same key, same queries)

| arm | what |
|---|---|
| text logistic | features above, trained on calibration templates |
| edited words | bag of added / removed words, trained on calibration templates (the vocabulary leak, see status) |
| embedding | `Qwen3-Embedding-0.6B` cosine (an open stand-in for the EmbeddingGemma used by the paper) |
| reranker | `Qwen3-Reranker-0.6B` on (key, query) |
| ask the model | the tested model, "same calculation, only numbers or wording changed? yes/no", score = logit(yes) - logit(no) |
| hidden cosine | cosine between the two prompts' hidden states, no training |
| **hidden router** | low-rank metric on `h(query) - h(key)`: within-template hit differences are the nuisance, flip differences the signal (LDA with shrinkage), fit on calibration templates only |
| controls | router with shuffled labels; random directions of the same rank; the router's metric on the embedding model's vectors |

Hidden states: forward passes only. Layer, readout (last prompt token or mean over the question) and rank are chosen by
grouped cross-validation inside the calibration templates. The embedding and reranker arms get the same calibration data
(threshold and the same metric fit), so the comparison is about the representation, not about who got training.

Models: `Qwen/Qwen3.5-0.8B` and `LiquidAI/LFM3-1.7B` (two families; no one-model wins).

## Metrics and decision rule (fixed now)

- **Primary:** `flip` false-hit rate at the threshold that keeps 95 % of calibration hits. Reported with the test hit rate
  on `number` and on `reword` separately. CIs: 10,000 bootstrap resamples **over templates**, paired between arms.
- Secondary: AUC `reword` vs `flip`; milliseconds per query; every sanity negative.
- **Pass** (both models): hit rate >= 90 % on `number` and on `reword`, and `flip` false hits at least 10 points below the
  best baseline with the paired 95 % CI excluding 0, and the shuffled control at chance.
- **"Ask the model" wins or ties:** the router is only worth it for speed. Say so, and compare the cost.
- **Fail:** record it as a negative result and stop. No layer search on the test templates.

Stage 2 (only after a pass): store real code per template, run hits end to end, and measure wrong answers returned per
100 queries and tokens saved, against the model answering alone.

### Clarifications fixed before the first GPU run

- A baseline counts as "best" only if it keeps >= 85 % of all test hits; otherwise a reject-everything arm would win.
- Thresholds of trained arms come from out-of-fold calibration scores (4 template folds), not from scores on the rows they
  were fitted to. The metric is fitted on `number`, `reword` and `flip` differences only.
- Extra arm, secondary and not in the pass rule: `pair_probe`, a trained probe on the hidden state of the prompt that
  shows both problems (one forward pass per stored problem, like the reranker).
- `edited_words_oracle` sees the edited span: an upper bound of the vocabulary leak, counted as a baseline.
- Text arms on the 45 test templates, CPU (false hits on `flip` / hits on `number` / on `reword`): text logistic
  78 / 88 / 83 %, changed words 72 / 98 / 93 %, edited-words oracle 58 / 100 / 87 %. This was the first look at the test
  templates and it only shows the scale of the problem; no choice was made from it.

## Cost

CPU: writing and reviewing about 100 x 2 edits (the slow part). GPU: one Colab L4 run, about 4,600 short prompts x 4
models, forward passes only; estimate 20-30 min (not measured).

## Result (2026-11-22, Colab L4, 9.5 min, `benchmarks/router_false_hits/results_colab_l4.json`): fail, stop here

45 test templates x 10 queries per kind. `flip` = false hits at the threshold that keeps 95 % of calibration hits; CIs are 10,000
bootstrap resamples over templates.

| arm | hit `number` / `reword` | `flip` false hits [95 % CI] | AUC `reword` vs `flip` |
|---|---|---|---|
| **embedding + same trained metric** | 97 / 97 % | **51 %** [38, 65] | **0.80** |
| edited-words oracle (leak bound) | 100 / 87 % | 58 % [44, 71] | 0.64 |
| **hidden router, LFM3-1.7B** (`h6_mean`, rank 8) | 98 / 97 % | 60 % [47, 73] | 0.78 |
| **hidden router, Qwen3.5-0.8B** (`h3_mean`, rank 8) | 98 / 94 % | 61 % [48, 74] | 0.77 |
| pair probe, Qwen / LFM3 (secondary) | 98 / 93, 94 / 95 % | 68 % / 79 % | 0.74 / 0.71 |
| changed words / text logistic | 97 / 91, 88 / 81 % | 72 % / 77 % | 0.62 / 0.55 |
| shuffled labels, Qwen / LFM3 | 98 / 93, 100 / 99 % | 81 % / 87 % | 0.68 / 0.69 (mean of 10) |
| ask the model, Qwen / LFM3 | 98 / 98, 91 / 92 % | 89 % / 89 % | 0.70 / 0.54 |
| hidden cosine, Qwen / LFM3 | 91 / 88, 99 / 98 % | 80 % / 94 % | 0.61 / 0.61 |
| embedding cosine | 98 / 98 % | 93 % [87, 98] | 0.60 |
| reranker | 98 / 98 % | 98 % [96, 100] | 0.63 |

- **Hit rates pass, the gap does not.** The router is 9 points *worse* than the best baseline on both models (Qwen +9.3 [-5.6, +24.2],
  LFM3 +8.7 [-6.4, +23.3]; a tie within noise, needed -10). The same metric on embedding vectors does at least as well, so the trained
  metric carries the effect and the hidden state adds nothing here.
- **The router beats the untrained and off-the-shelf arms** (paired: embedding cosine -33 points on both models, reranker -37 / -38, asking
  the model -29 / -29, its own cosine -20 / -34, all CIs exclude 0), but not the words that changed (-11 / -12 [about -27, +4]) or the oracle (+3 / +2).
- **The shuffled control failed, and the control was badly designed.** It swaps `reword` and `flip` per template but keeps `number` as a
  hit, so the metric still learns to ignore number changes and measure the size of any other edit. Flips are larger edits (1.43 new words
  vs 1.23), so it reaches AUC 0.68-0.69, not 0.5. Two thirds of the router's distance from chance (0.77-0.78) needs no calculation label.
  The labels do add something (router minus the last shuffle: -20 [-31, -10] and -27 [-38, -16] points).
- **Nothing here is a safe cache:** the best arm returns the wrong stored answer on about half of the flips. "Ask the model" is
  yes-biased (LFM3 accepts 85 % of *other* templates).
- Per-template errors of the two hidden routers correlate 0.70 with each other and 0.33-0.35 with the embedding metric.

By the rule fixed beforehand, no layer search or new edits on these templates and no stage 2.

## Known weak points

- Hand-written edits reflect one author; the blind second review is the only check, and it is not a human panel.
- GSM-Symbolic templates are GSM8K-like and public. A pass would cover grade-school arithmetic wording, nothing else.
- The pilot used the public viewer API and 60 templates; its numbers pick the design, they are not results to cite.
