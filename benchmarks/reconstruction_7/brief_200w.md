Target: Qwen/Qwen3.5-0.8B (enable_thinking=True) gets stuck in repetitive 16-gram thinking loops on GSM8K (~88% loop rate, 11% accuracy).

Mechanism: The loop is sustained by a 1D residual-stream direction emitted by mlp.down_proj in two mid-network layers (around layers 13-14). Suppressing just 16 neurons per layer along that direction breaks the loop with minimal collateral damage.

Reconstruction recipe:
1. Generate greedy responses on ~60-100 GSM8K train questions (max_new_tokens=2048 or 4096) and label looping tokens (repeated 16-grams in a sliding window) vs. normal pre-loop response tokens.
2. Split sequences into two disjoint halves (A and B). Run teacher-forced forward passes hooking each layer's mlp.down_proj input h and output y.
3. On Half A, compute each layer's unit difference-of-means loop direction u_l = norm(mean_loop(y) - mean_normal(y)). On Half B, rank all layers by held-out d-prime along u_l and pick the top 2 layers.
4. In each chosen layer, score every MLP neuron i by its loop-vs-normal contribution along u_l: (mean_loop(h_i) - mean_normal(h_i)) * (u_l^T W[:, i]), pick the top 16 neurons (mask m), and edit W_new = W - 12 * u_l * (m * (u_l^T W)). Save the edited HuggingFace checkpoint to --out-dir.
