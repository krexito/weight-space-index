---
name: weight-space-index
description: Read, search and surgically edit the weights of a transformer checkpoint (.safetensors) without training or GPU forward passes, using the open-source Weight-Space Index (wsi) toolkit. Use this when someone wants to know what a neuron does, which neurons store a fact ("France -> Paris"), what a weight edit or fine-tune changed, how to rewire a fact, how to move an edit to another model, or how to inspect, verify or apply a portable .moipatch file (reversible, self-routing weight patches such as a loop fix or an anti-sycophancy patch). Also use it to set up the wsi MCP server.
---

# Weight-Space Index (wsi)

`wsi` turns a `.safetensors` checkpoint into a searchable index of what its neurons read and write, in vocabulary terms. It runs on CPU, statically: it reads the weights and does not run the model, except for the optional runtime-capture and `.moipatch` features.

Repository: https://github.com/krexito/weight-space-index (Apache-2.0). It builds on `wsr` (Weight-Space Reading) by Şuayp Talha Kocabay.

## 1. Set up

```bash
pip install "git+https://github.com/krexito/weight-space-index.git"            # core: torch + safetensors
pip install "weight-space-index[benchmarks] @ git+https://github.com/krexito/weight-space-index.git"  # + transformers
```

- Python 3.10+. The model must be a local folder with `*.safetensors` and its tokenizer files. Download it first, for example with `hf download Qwen/Qwen3.5-0.8B --local-dir models/qwen35-0.8b`.
- Use the CPU for local work. On a machine with a GPU, set `CUDA_VISIBLE_DEVICES=""` so torch does not grab it, unless the user wants the GPU.
- Memory: indexing reads one layer at a time, but cross-model work loads both models. Check free RAM before running it on models above ~4B parameters.

Check that it works with `python -m wsi --help` (the `wsi` command is the same thing).

## 2. Pick the task

| The user wants to... | Command | Python |
|---|---|---|
| Get an overview of a model | `wsi overview MODEL` | `I = WeightSpaceIndex.from_model(MODEL); I.overview()` |
| Find where a token or concept lives | `wsi search MODEL "ĠParis"` | `I.search_token("ĠParis")` |
| Find neurons that map A to B (a stored fact) | `wsi transduction MODEL "ĠFrance" "ĠParis"` | `I.search_transduction("ĠFrance", "ĠParis", top_k=5)` |
| Explain one neuron | `wsi neuron MODEL --layer 8 --unit 430` | `I.explain_neuron(layer=8, unit=430)` |
| Trace what feeds a neuron and what reads it | `wsi trace MODEL --layer 8 --unit 430` | `I.trace_circuit(layer=8, unit=430)` |
| Trace a hidden-state direction across heads and MLPs | `wsi internal-state MODEL --layer 15 --head 7` | `I.trace_internal_state(layer=15, head=7)` |
| Explain what a weight edit did | `wsi explain-edit MODEL edit.safetensors` | `I.explain_edit("edit.safetensors")` |
| Diff two runtime traces or turns | `wsi diff-compact MODEL TRACE_A TRACE_B` | `I.diff_compact(ct_a, ct_b)` / `diff_traces(model, ids_a, ids_b, wsi=I)` |
| Discover a loop/behavioural fix circuit | `wsi discover-circuit MODEL --trace CT_H --loop-trace CT_L -o fix.moipatch` | `I.discover_compact_circuit(traces)` |
| Rewire a fact (A->B becomes A->C) | `wsi rewire MODEL "ĠTower" "ĠParis" "ĠRome" -o edit.safetensors --top-k 24 --strength 6.0` | `I.rewire_transduction(...)` |
| Diff a base model and its fine-tune | `wsi diff BASE TUNED` | `I.diff_model(TUNED)` |
| Match circuits across two architectures | `wsi align MODEL_A MODEL_B --pair Eiffel:Paris` | `I.align_circuits(other)` |
| Move an edit from model A to model B | `wsi transfer-edit MODEL_A MODEL_B edit_a.safetensors -o edit_b.safetensors` | `I.transfer_edit_to(MODEL_B, "edit_a.safetensors")` |
| Read or check a `.moipatch` | `wsi moipatch info PATCH` / `wsi moipatch verify PATCH --model MODEL --cpu` | `MoIPatch.load(PATCH)` |
| Give another agent these tools | `python -m wsi mcp` (stdio MCP server, 13 `wsi_*` tools) | |

`wsi build MODEL -o INDEX_DIR` precomputes an index once. Every command also works without it (lazy mode).

### Tokens are written the way the tokenizer writes them
For GPT-2/Qwen-style BPE tokenizers, a word that follows a space starts with `Ġ`: `"ĠParis"`, not `"Paris"`. If a search returns nothing, try both forms. You can also check with `tok.tokenize(" Paris")`.

## 3. Workflows

### Find and explain a fact
1. `wsi transduction MODEL "ĠFrance" "ĠParis"` → the top neurons (layer, unit) and their scores.
2. `wsi neuron MODEL --layer L --unit U` on the top 1-3 neurons → the input triggers, the promoted and suppressed output tokens, and a rule like `[Eiffel, France] -> [Paris, Louvre]`.
3. Report the rule in plain words, with the layer and unit. These are weight-space readings: before claiming a neuron "causes" a behaviour, offer to confirm it with a forward pass (an ablation or an activation check).

### Rewire a fact, then check it
1. `wsi rewire MODEL "ĠTower" "ĠParis" "ĠRome" -o edit.safetensors --top-k 24 --strength 6.0`.
2. `wsi explain-edit MODEL edit.safetensors` to confirm the edit says what was intended.
3. Test it: load the model with the edit applied, check the target prompt, and check a few unrelated prompts (guard probes) for side effects. Start with a small `--top-k` and `--strength` and raise them only if the target does not flip.
4. Never overwrite the original checkpoint. The edit is a separate file.

### Explain a fine-tune or an unknown edit
Run `wsi diff BASE TUNED` (same architecture) or `wsi explain-edit BASE edit.safetensors`. Summarize which layers moved, which tokens or concepts were boosted or suppressed, and the edit's rank. Say when the change looks like noise or a no-op.

### Discover a loop or behavioural fix and build a `.moipatch` (Hybrid `CompactTrace` + `WSI`)
Use this when debugging repetition loops or finding which layers/neurons to patch on a model:
1. **Capture `CompactTrace`s once per sequence:** run `ct = capture_compact(model, ids)` on healthy/neutral and looping/failing runs (saves `INT8+had64` residuals, auto-95% `FP4` MLPs, `INT8` pre-$W_O$ head vectors, sparse `Top-16` attention links, and `INT8` cache; `ct.save(dir)` persists them to disk).
2. **Run hybrid discovery (`CompactTrace` origin + `WSI` static SVD echo):** call `res = I.discover_compact_circuit(traces)` (or `wsi discover-circuit MODEL --trace CT_H --loop-trace CT_L -o fix.moipatch` via CLI / `wsi_discover_compact_circuit` via MCP). `CompactTrace` hybrid scoring (`fisher_mlp`, `head_max_d`, `attn_focus_d`) locates where the deviation originates (`wsi_origins`), and static `WSI` SVD resonance locates downstream amplifier layers invisible to activation variance alone (`wsi_resonant_dst`), assembling ready-to-attach `MoIPatch` candidates (`res["patch"]`, `res["patches"]["wsi_fa3"]`, `res["patches"]["wsi_pair"]`).
3. **Verify the causal shift at onset with $W_O$-projected heads:** run `I.diff_compact(ct_base, ct_patched, model=model, pos_a=onset, pos_b=onset)` (or `diff_traces(model, ids_a, ids_b, wsi=I)`). Projecting pre-$W_O$ head vectors through $W_O$ catches pure directional rotations alongside MLP neuron shifts and KV/DeltaNet cache deltas.
4. **Validate before saving/shipping:** check accuracy and loop rate with `h = patch.attach(model)`, `h.on()`, `h.off()` on held-out prompts before keeping the `.moipatch`.

### Use a `.moipatch`
A `.moipatch` is one small file (tens of KB) holding a weight edit recipe (`W' = W − α·u·(m ⊙ uᵀW)`), a router, optional gates, the activation policy and the expected results. It also holds a SHA-256 fingerprint of every weight it touches.
1. `wsi moipatch info PATCH` shows what it does, its router and gates, the base model it was built for, and its expected numbers.
2. `wsi moipatch verify PATCH --model MODEL --cpu` checks the fingerprint, switches the patch on and off, and confirms the original weights come back byte for byte. It refuses a model whose weights differ.
3. At runtime:
   ```python
   from wsi.moipatch import MoIPatch
   patch = MoIPatch.load("anti_sycophancy.moipatch")
   h = patch.attach(model)                       # checks the fingerprint; the model is unchanged
   out = model(ids, output_hidden_states=True)
   if patch.fires(out.hidden_states, confidence=p_turn1):   # router + confidence gate
       h.on();  ...generate...;  h.off()         # off() restores the exact original bytes
   ```
   Read `patch.meta["policy"]` and follow it. For example, the anti-sycophancy patch is meant for one reply after a pushback turn, and only when the model was confident in the answer being challenged.

The repository ships these patches in `patches/` (not included in the pip package; clone the repository or download one file from `https://github.com/krexito/weight-space-index/raw/main/patches/<name>.moipatch`):
- `qwen35_4b_anti_sycophancy.moipatch` (Qwen3.5-4B): on MMLU pushback, the share of correct answers the model keeps goes from 15.0% to 40.1%, and it accepts 90.7% of valid corrections. A gentler variant keeps 28.9% and accepts 95.8%.
- `qwen35_0.8b_p50c_loop_fix.moipatch` (Qwen3.5-0.8B): a repetition-loop fix that latches on when a 16-gram repetition detector fires.

Quote these numbers as the repository's measurements, not as guarantees for other prompts.

## 4. Report results honestly
- Say which model, which command, and the exact layer and unit numbers.
- Weight-space readings are hypotheses about function. Mark them as such until a forward-pass test confirms them.
- When a search is empty or weak (low scores, no clear tokens), say so instead of over-reading noise.
- Edits change model behaviour: keep the original checkpoint, write edits to new files, and report the side-effect checks you ran.

More detail: `references/api.md` (Python API and return fields) and the repository README (benchmarks).
