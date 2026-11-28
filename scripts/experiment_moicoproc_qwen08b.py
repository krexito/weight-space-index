"""Colab L4 Experiment: Weight-Space Co-Processors via MoI (`Qwen/Qwen3.5-0.8B`).

Bundles real LoopedWSP circuits (`maze`, `hanoi`) into self-contained `.moicoproc` files
with calibrated Layer-12 hidden-state routers (including near-domain adversarial `veto_dir`),
and evaluates on disjoint held-out splits:
  1. Algorithmic puzzles (`20` mazes + `4` Tower of Hanoi sizes): `base_qwen08b` vs `moicoproc_routed`.
  2. Near-domain adversarial controls (`20` prompts talking *about* mazes/Hanoi without instances).
  3. Off-domain controls (`MMLU` slice n=200 + everyday prose n=30): target `0/230` false fires.
  4. Real `bf16` hidden-state ridge probe: measures whether `Qwen3.5-0.8B`'s hidden states can
     linearly project into LoopedWSP's symbolic cell input space within the CPU SNR threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi.coproc import (
    CoProcessorExecutor,
    MoICoProcessor,
    extract_hanoi_n_from_prompt,
    extract_maze_from_prompt,
    fit_coproc_router,
    route_and_generate,
    verify_hanoi_moves,
    verify_maze_solution,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _make_seeded_maze(n: int, seed: int) -> str:
    rng = random.Random(seed)
    g, queue = [["#"] * n for _ in range(n)], [(1, 1)]
    g[1][1] = " "
    while queue:
        r, c = queue[-1]
        nxt = [
            (r + a, c + b)
            for a, b in ((2, 0), (-2, 0), (0, 2), (0, -2))
            if 0 < r + a < n - 1 and 0 < c + b < n - 1 and g[r + a][c + b] == "#"
        ]
        if not nxt:
            queue.pop()
            continue
        r2, c2 = rng.choice(nxt)
        g[(r + r2) // 2][(c + c2) // 2] = g[r2][c2] = " "
        queue.append((r2, c2))
    for _ in range(n):
        g[rng.randrange(1, n - 1)][rng.randrange(1, n - 1)] = " "
    open_cells = [(r, c) for r in range(n) for c in range(n) if g[r][c] == " "]
    (a, b), (c, d) = rng.sample(open_cells, 2)
    g[a][b], g[c][d] = "S", "G"
    return "\n".join("".join(row) for row in g)


_EVERYDAY_PROSE_CAL = [
    "Explain how photosynthesis works in simple terms.",
    "Write a short poem about the ocean at night.",
    "What are three tips for improving sleep quality?",
    "Summarize the causes of the French Revolution in one paragraph.",
    "How do I reverse a list in Python?",
    "What is the difference between weather and climate?",
    "Give me a simple recipe for pancakes.",
    "Why is the sky blue?",
    "Explain what a hash table is to a beginner programmer.",
    "What are the main differences between a virus and a bacterium?",
    "How does compound interest work? Give a short example.",
    "List four renewable energy sources and their tradeoffs.",
    "What is the Pythagorean theorem?",
    "Describe the water cycle briefly.",
    "Translate 'Good morning, how are you?' into Spanish.",
    "What is the capital of Japan and why is it famous?",
]

_EVERYDAY_PROSE_TEST = [
    "Explain why leaves change color in autumn.",
    "Write a two-sentence summary of how electric motors work.",
    "How do you sort a dictionary by its values in Python?",
    "What causes tides in Earth's oceans?",
    "Name three differences between mitosis and meiosis.",
    "Explain what HTTP status code 404 means.",
    "Give three practical ways to reduce household electricity use.",
    "What is the boiling point of water at high altitude, and why?",
    "Describe what a relational database index does.",
    "Explain the greenhouse effect in three sentences.",
    "How do vaccines train the immune system?",
    "What is the difference between RAM and SSD storage?",
    "Write a polite message rescheduling a dentist appointment.",
    "Explain what a prime number is with three examples.",
    "How does a refrigerator cool food?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Why do airplanes fly faster at cruising altitude?",
    "Explain the difference between TCP and UDP.",
    "What makes sourdough bread taste sour?",
    "How do solar panels generate electricity?",
    "Explain what recursion means in computer programming.",
    "What are the three branches of the U.S. government?",
    "Why is stainless steel resistant to rust?",
    "How does a binary search algorithm find an element?",
    "Explain what inflation means in macroeconomics.",
    "What is the difference between an asteroid and a comet?",
    "How do noise-canceling headphones reduce ambient sound?",
    "Explain the difference between a stack and a queue.",
    "Why does ice float on liquid water?",
    "What is a neural network activation function?",
]

_NEAR_DOMAIN_MAZE_CAL = [
    "Explain how breadth-first search finds the shortest path in a maze.",
    "What is the history of the hedge maze in Hampton Court?",
    "Write a short paragraph describing a character lost inside a stone labyrinth.",
    "Compare A* search and Dijkstra's algorithm for grid pathfinding.",
    "Why do depth-first maze generators create long winding corridors?",
    "How does wall-following fail on a maze with loops?",
    "Explain the concept of a heuristic function in maze navigation.",
    "Describe three ways to represent a 2D grid maze in Python.",
    "What makes a maze 'perfect' in graph theory?",
    "Summarize why transformers struggle with long multi-step maze search.",
]

_NEAR_DOMAIN_MAZE_TEST = [
    "How does Prim's algorithm generate a random maze?",
    "Explain why BFS guarantees the shortest unweighted path on a grid.",
    "Tell a brief story about an explorer escaping a crystal maze.",
    "What is the time complexity of solving an N by N maze with BFS?",
    "Compare DFS and BFS memory usage when exploring large mazes.",
    "Why is the dead-end filling algorithm useful for grid mazes?",
    "Explain how Manhattan distance works as an A* heuristic on grids.",
    "What did the Minotaur myth symbolize about the Cretan labyrinth?",
    "Describe how Kruskals algorithm can build a spanning-tree maze.",
    "How do robotics researchers test SLAM algorithms inside indoor mazes?",
]

_NEAR_DOMAIN_HANOI_CAL = [
    "Explain the recursive algorithm for solving the Tower of Hanoi.",
    "What is the legend behind the Tower of Hanoi puzzle?",
    "Why does the 3-peg Tower of Hanoi take 2^N - 1 moves?",
    "Explain the iterative binary-bit trick for Tower of Hanoi moves.",
    "Who invented the Tower of Hanoi puzzle in 1883?",
    "How is Tower of Hanoi related to Gray codes?",
    "What happens in the Tower of Hanoi if you add a fourth peg (Reve's puzzle)?",
    "Why do benchmark papers use Tower of Hanoi to test reasoning models?",
    "Write a short poem about stacking golden rings on three pillars.",
    "Describe the space complexity of recursive Tower of Hanoi.",
]

_NEAR_DOMAIN_HANOI_TEST = [
    "Prove by induction that N disks in Tower of Hanoi require 2^N - 1 moves.",
    "Explain how Frame-Stewart algorithm approximates 4-peg Tower of Hanoi.",
    "Why can a smaller disk never sit underneath a larger disk in Tower of Hanoi?",
    "How many moves would 64 disks take in the classic Tower of Hanoi myth?",
    "Explain the connection between Tower of Hanoi and Sierpinski triangles.",
    "How do cognitive scientists use Tower of Hanoi in executive function tests?",
    "Write a brief dialogue discussing strategies for the Tower of Hanoi.",
    "Compare Tower of Hanoi and Checker Jumping as state-space puzzles.",
    "What is the optimal move rule on odd vs even steps in Tower of Hanoi?",
    "Explain why LLMs lose track of disk positions in deep Hanoi sequences.",
]


def _load_mmlu_slices(n_cal: int = 80, n_test: int = 200, seed: int = 42) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), n_cal + n_test)
    letters = "ABCD"

    def fmt(row: dict[str, Any]) -> str:
        ch = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(row["choices"]))
        return f"Question: {row['question']}\n{ch}\nAnswer with the letter only."

    return [fmt(ds[i]) for i in idxs[:n_cal]], [fmt(ds[i]) for i in idxs[n_cal:]]


def _format_chat(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return prompt


def _extract_last_hidden(model, tokenizer, prompts: list[str], layer: int, batch_size: int = 8) -> torch.Tensor:
    dev = next(model.parameters()).device
    orig_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    feats: list[torch.Tensor] = []
    try:
        for s in range(0, len(prompts), batch_size):
            batch = prompts[s:s + batch_size]
            texts = [_format_chat(tokenizer, p) for p in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
            inp = enc["input_ids"].to(dev)
            att = enc["attention_mask"].to(dev)
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=att, output_hidden_states=True)
            h = out.hidden_states[layer][:, -1].float().cpu()
            feats.append(h)
    finally:
        tokenizer.padding_side = orig_side
    return torch.cat(feats, dim=0)


def _generate_base(model, tokenizer, prompts: list[str], max_new_tokens: int = 256, batch_size: int = 8) -> list[str]:
    dev = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    orig_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    out_texts: list[str] = []
    try:
        for s in range(0, len(prompts), batch_size):
            batch = prompts[s:s + batch_size]
            texts = [_format_chat(tokenizer, p) for p in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
            inp = enc["input_ids"].to(dev)
            att = enc["attention_mask"].to(dev)
            L = inp.shape[1]
            with torch.no_grad():
                gen = model.generate(
                    input_ids=inp, attention_mask=att, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=pad_id,
                )
            out_texts.extend(tokenizer.decode(gen[j, L:], skip_special_tokens=True) for j in range(len(batch)))
    finally:
        tokenizer.padding_side = orig_side
    return out_texts


def _check_base_maze_answer(grid: str, text: str) -> bool:
    extracted = extract_maze_from_prompt(text)
    if extracted is not None and verify_maze_solution(grid, extracted):
        return True
    # Also accept raw grid containing 'o' if slightly formatted
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(grid.splitlines())
    cand = [ln for ln in lines if len(ln) == n and set(ln).issubset(set("# SGo"))]
    if len(cand) == n:
        return verify_maze_solution(grid, "\n".join(cand))
    return False


def _check_base_hanoi_answer(n: int, text: str) -> bool:
    if verify_hanoi_moves(n, text):
        return True
    # Parse fallback list of triples or arrow notation if model did not write "moves ="
    triples = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if triples:
        synth = "moves = [" + ", ".join(f"[{a}, {b}, {c}]" for a, b, c in triples) + "]"
        return verify_hanoi_moves(n, synth)
    return False


def _probe_latent_interface_on_mazes(
    model,
    tokenizer,
    maze_npz: Path,
    cal_mazes: list[str],
    test_mazes: list[str],
    layer: int,
) -> dict[str, Any]:
    """Probe whether Qwen3.5-0.8B's real bf16 hidden states can linearly reconstruct LoopedWSP's
    initial symbolic cell state `h0` (or one-hot cell type) within the CPU noise cliff."""
    from wsi.looped_reader import load_looped_weights

    spec = load_looped_weights(maze_npz, blind=False)
    emb_row = spec.matrices["emb.row"].to_dense()
    emb_col = spec.matrices["emb.col"].to_dense()
    emb_char = spec.matrices["emb.char"].to_dense()
    charset = spec.raw_config["io"]["charset"]
    dev = next(model.parameters()).device

    def collect_cell_features_and_targets(grids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for grid in grids:
            rows = grid.split("\n")
            prompt = f"Solve this maze:\n{grid}\nReturn the grid with path marked by 'o'."
            ids = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            offsets = ids["offset_mapping"]
            inp = torch.tensor([ids["input_ids"]], device=dev)
            with torch.no_grad():
                out = model(input_ids=inp, output_hidden_states=True)
            h_seq = out.hidden_states[layer][0].float().cpu().numpy()  # [T, D]
            grid_start = prompt.index(grid)
            # Match tokens whose character span lands inside a single maze cell
            cell_idx = 0
            for r, row in enumerate(rows):
                for c, ch in enumerate(row):
                    char_pos = grid_start + r * (len(row) + 1) + c
                    # Find token covering char_pos
                    tok_i = None
                    for ti, (a, b) in enumerate(offsets):
                        if a <= char_pos < b:
                            tok_i = ti
                            break
                    if tok_i is not None:
                        xs.append(h_seq[tok_i])
                        ys.append(emb_row[r] + emb_col[c] + emb_char[charset.index(ch)])
                    cell_idx += 1
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    X_cal, Y_cal = collect_cell_features_and_targets(cal_mazes)
    X_test, Y_test = collect_cell_features_and_targets(test_mazes)

    best = None
    D = X_cal.shape[1]
    eye = np.eye(D, dtype=np.float64)
    for lam in (1.0, 10.0, 100.0, 1000.0):
        W = np.linalg.solve(X_cal.T @ X_cal + lam * eye, X_cal.T @ Y_cal)
        Y_pred = X_test @ W
        err = Y_pred - Y_test
        rmse = float(np.sqrt(np.mean(err ** 2)))
        sig = float(np.sum(Y_test ** 2))
        noise = float(np.sum(err ** 2))
        snr_db = float(10.0 * np.log10(sig / max(noise, 1e-18)))
        cand = {"lambda": lam, "rmse": round(rmse, 5), "snr_db": round(snr_db, 2)}
        if best is None or rmse < best["rmse"]:
            best = cand

    return {
        "n_cal_tokens": int(X_cal.shape[0]),
        "n_test_tokens": int(X_test.shape[0]),
        "hidden_dim": int(D),
        "target_dim": int(Y_cal.shape[1]),
        **(best or {}),
    }


def run_experiment(args) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    looped_dir = Path(args.looped_wsp_dir)
    out_path = Path(args.out)

    print(f"Loading {args.model} in {args.dtype} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dt = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, trust_remote_code=True).to(args.device).eval()

    # Build disjoint calibration and held-out test prompts
    cal_mazes = [_make_seeded_maze(9, 1000 + i) for i in range(16)]
    test_mazes = [_make_seeded_maze(9, 2000 + i) for i in range(15)] + [_make_seeded_maze(11, 2100 + i) for i in range(5)]
    cal_maze_prompts = [f"Find the shortest path in this maze (mark path with 'o'):\n{g}" for g in cal_mazes]
    test_maze_prompts = [f"Find the shortest path in this maze (mark path with 'o'):\n{g}" for g in test_mazes]

    cal_hanoi_prompts = [f"Solve the Tower of Hanoi puzzle with {n} disks. List every move [disk, from, to]." for n in (3, 4, 5) for _ in range(4)]
    test_hanoi_ns = [3, 4, 5, 6]
    test_hanoi_prompts = [f"Solve the Tower of Hanoi puzzle with {n} disks. List every move [disk, from, to]." for n in test_hanoi_ns]

    mmlu_cal, mmlu_test = _load_mmlu_slices(n_cal=80, n_test=200, seed=42)
    neg_cal_all = _EVERYDAY_PROSE_CAL + mmlu_cal

    print(f"Extracting Layer-{args.router_layer} calibration hidden states...")
    h_neg_cal = _extract_last_hidden(model, tok, neg_cal_all, args.router_layer, batch_size=args.bs)

    # 1. Calibrate Maze MoICoProcessor
    h_maze_pos = _extract_last_hidden(model, tok, cal_maze_prompts, args.router_layer, batch_size=args.bs)
    h_maze_adv = _extract_last_hidden(model, tok, _NEAR_DOMAIN_MAZE_CAL + cal_hanoi_prompts[:4], args.router_layer, batch_size=args.bs)
    maze_router = fit_coproc_router(h_maze_pos, h_neg_cal, h_veto_neg=h_maze_adv, layer=args.router_layer, target_fpr=0.0)
    cp_maze = MoICoProcessor.from_looped_weights(
        looped_dir / "weights" / "networks" / "maze.npz",
        task="maze",
        dtype=torch.float64,
        sparse_payload=True,
        router=maze_router,
        meta={"base_model": args.model, "name": "maze_coproc"},
    )
    cp_maze_path = out_path.parent / "maze_qwen08b.moicoproc"
    cp_maze.save(cp_maze_path)

    # 2. Calibrate Hanoi MoICoProcessor
    h_hanoi_pos = _extract_last_hidden(model, tok, cal_hanoi_prompts, args.router_layer, batch_size=args.bs)
    h_hanoi_adv = _extract_last_hidden(model, tok, _NEAR_DOMAIN_HANOI_CAL + cal_maze_prompts[:4], args.router_layer, batch_size=args.bs)
    hanoi_router = fit_coproc_router(h_hanoi_pos, h_neg_cal, h_veto_neg=h_hanoi_adv, layer=args.router_layer, target_fpr=0.0)
    cp_hanoi = MoICoProcessor.from_looped_weights(
        looped_dir / "weights" / "networks" / "hanoi.npz",
        task="hanoi",
        dtype=torch.float64,
        sparse_payload=True,
        router=hanoi_router,
        meta={"base_model": args.model, "name": "hanoi_coproc"},
    )
    cp_hanoi_path = out_path.parent / "hanoi_qwen08b.moicoproc"
    cp_hanoi.save(cp_hanoi_path)

    # Also measure ultra-compact maze_size_1k payload size
    cp_1k = MoICoProcessor.from_looped_weights(
        looped_dir / "weights" / "more_networks" / "maze_size_1k",
        task="maze_1k",
        dtype=torch.float32,
        sparse_payload=False,
    )

    executor = CoProcessorExecutor()

    # Evaluate on held-out puzzles (base vs moicoproc_routed)
    print("Evaluating held-out Mazes (n=20)...")
    base_maze_ans = _generate_base(model, tok, test_maze_prompts, max_new_tokens=args.max_new, batch_size=args.bs)
    routed_maze = route_and_generate(model, tok, test_maze_prompts, cp_maze, executor, max_new_tokens=args.max_new, batch_size=args.bs)
    base_maze_acc = sum(_check_base_maze_answer(g, a) for g, a in zip(test_mazes, base_maze_ans)) / len(test_mazes)
    routed_maze_acc = sum(verify_maze_solution(g, a) for g, a in zip(test_mazes, routed_maze["answers"])) / len(test_mazes)

    print("Evaluating held-out Hanoi (n=4, disks 3..6)...")
    base_hanoi_ans = _generate_base(model, tok, test_hanoi_prompts, max_new_tokens=args.max_new, batch_size=args.bs)
    routed_hanoi = route_and_generate(model, tok, test_hanoi_prompts, cp_hanoi, executor, max_new_tokens=args.max_new, batch_size=args.bs)
    base_hanoi_acc = sum(_check_base_hanoi_answer(n, a) for n, a in zip(test_hanoi_ns, base_hanoi_ans)) / len(test_hanoi_ns)
    routed_hanoi_acc = sum(verify_hanoi_moves(n, a) for n, a in zip(test_hanoi_ns, routed_hanoi["answers"])) / len(test_hanoi_ns)

    # Evaluate near-domain adversarial controls (20 prompts talking about puzzles without instances)
    print("Evaluating near-domain adversarial controls (n=20)...")
    base_adv_maze = _generate_base(model, tok, _NEAR_DOMAIN_MAZE_TEST, max_new_tokens=128, batch_size=args.bs)
    routed_adv_maze = route_and_generate(model, tok, _NEAR_DOMAIN_MAZE_TEST, cp_maze, executor, max_new_tokens=128, batch_size=args.bs)
    base_adv_hanoi = _generate_base(model, tok, _NEAR_DOMAIN_HANOI_TEST, max_new_tokens=128, batch_size=args.bs)
    routed_adv_hanoi = route_and_generate(model, tok, _NEAR_DOMAIN_HANOI_TEST, cp_hanoi, executor, max_new_tokens=128, batch_size=args.bs)

    adv_identical = (
        sum(a == b for a, b in zip(base_adv_maze, routed_adv_maze["answers"]))
        + sum(a == b for a, b in zip(base_adv_hanoi, routed_adv_hanoi["answers"]))
    ) / (len(_NEAR_DOMAIN_MAZE_TEST) + len(_NEAR_DOMAIN_HANOI_TEST))

    # Evaluate off-domain controls (MMLU n=200 + prose n=30 = 230)
    print("Evaluating off-domain controls (MMLU n=200 + prose n=30)...")
    off_prompts = mmlu_test + _EVERYDAY_PROSE_TEST
    base_off = _generate_base(model, tok, off_prompts, max_new_tokens=64, batch_size=args.bs)
    routed_off_maze = route_and_generate(model, tok, off_prompts, cp_maze, executor, max_new_tokens=64, batch_size=args.bs)
    routed_off_hanoi = route_and_generate(model, tok, off_prompts, cp_hanoi, executor, max_new_tokens=64, batch_size=args.bs)
    off_identical_maze = sum(a == b for a, b in zip(base_off, routed_off_maze["answers"])) / len(off_prompts)
    off_identical_hanoi = sum(a == b for a, b in zip(base_off, routed_off_hanoi["answers"])) / len(off_prompts)

    # Latent interface ridge probe from real Qwen3.5-0.8B bf16 hidden states
    print("Running real bf16 hidden-state ridge probe into Maze symbolic space...")
    ridge_probe = _probe_latent_interface_on_mazes(
        model, tok, looped_dir / "weights" / "networks" / "maze.npz",
        cal_mazes=cal_mazes[:10], test_mazes=test_mazes[:10], layer=args.router_layer,
    )

    payload = {
        "model": args.model,
        "dtype": args.dtype,
        "router_layer": args.router_layer,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "artifacts": {
            "maze_1k_fp32_weights_bytes": cp_1k.weights_payload_bytes(),
            "maze_coproc_file_kb": round(cp_maze_path.stat().st_size / 1024.0, 2),
            "hanoi_coproc_file_kb": round(cp_hanoi_path.stat().st_size / 1024.0, 2),
        },
        "maze_eval": {
            "n_test": len(test_mazes),
            "base_accuracy": round(base_maze_acc, 4),
            "moicoproc_accuracy": round(routed_maze_acc, 4),
            "router_stats": routed_maze["stats"],
        },
        "hanoi_eval": {
            "n_test": len(test_hanoi_ns),
            "disks": test_hanoi_ns,
            "base_accuracy": round(base_hanoi_acc, 4),
            "moicoproc_accuracy": round(routed_hanoi_acc, 4),
            "router_stats": routed_hanoi["stats"],
        },
        "near_domain_adversarial_controls": {
            "n_test": len(_NEAR_DOMAIN_MAZE_TEST) + len(_NEAR_DOMAIN_HANOI_TEST),
            "maze_adv_fires": routed_adv_maze["stats"]["n_fired"],
            "hanoi_adv_fires": routed_adv_hanoi["stats"]["n_fired"],
            "identical_to_base": round(adv_identical, 4),
        },
        "off_domain_controls": {
            "n_test": len(off_prompts),
            "maze_coproc_fires": routed_off_maze["stats"]["n_fired"],
            "hanoi_coproc_fires": routed_off_hanoi["stats"]["n_fired"],
            "maze_identical_to_base": round(off_identical_maze, 4),
            "hanoi_identical_to_base": round(off_identical_hanoi, 4),
        },
        "latent_interface_ridge_probe": ridge_probe,
    }
    _atomic_json(out_path, payload)
    print(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--looped-wsp-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--router-layer", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    run_experiment(ap.parse_args(argv))


if __name__ == "__main__":
    main()
