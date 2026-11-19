#!/usr/bin/env python3
"""Standalone weight-space loop-fix reconstruction script for Qwen3.5-0.8B.

Implements the blind 4-step reconstruction recipe:
1. Elicit greedy responses on GSM8K train questions (enable_thinking=True) and
   label looping tokens (repeated 16-grams within a sliding window) vs. normal
   pre-loop response tokens.
2. Split sequences into two disjoint halves (A and B) and run streaming
   teacher-forced forward passes hooking each layer's mlp.down_proj input h and
   output y (accumulating float64 statistics without storing activations).
3. On Half A, compute each layer's unit difference-of-means loop direction
   u_l = norm(mean_loop(y) - mean_normal(y)). On Half B, rank all layers by
   held-out d-prime along u_l and pick the top 2 layers.
4. In each chosen layer, score every MLP neuron i by its loop-vs-normal
   contribution along u_l: (mean_loop(h_i) - mean_normal(h_i)) * (u_l^T W[:, i]),
   select the top 16 neurons (mask m), apply
   W_new = W - 12 * u_l * (m * (u_l^T W)), and save the edited HuggingFace
   checkpoint to --out-dir.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Streaming float64 statistics (zero activation retention)
# ---------------------------------------------------------------------------


class RunningVectorMean:
    """Exact streaming mean for 1D vectors in float64."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.count = 0

    def update(self, batch: torch.Tensor) -> None:
        """Update with a 2D tensor of shape (N, dim)."""
        if batch.numel() == 0:
            return
        self.sum += batch.sum(dim=0, dtype=torch.float64).cpu()
        self.count += int(batch.shape[0])

    @property
    def mean(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("Cannot compute mean with count == 0.")
        return self.sum / float(self.count)


class RunningScalarStats:
    """Numerically stable Chan/Welford streaming mean & variance in float64."""

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        """Update with a 1D tensor of scalar values."""
        n_b = int(values.numel())
        if n_b == 0:
            return
        vals = values.detach().to(dtype=torch.float64, device="cpu")
        mean_b = float(vals.mean().item())
        m2_b = float(((vals - mean_b) ** 2).sum().item())

        if self.count == 0:
            self.count = n_b
            self.mean = mean_b
            self.m2 = m2_b
            return

        n_a = self.count
        n_ab = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n_ab)
        self.m2 = self.m2 + m2_b + (delta * delta) * (n_a * n_b / n_ab)
        self.count = n_ab

    @property
    def variance(self) -> float:
        if self.count == 0:
            raise ValueError("Cannot compute variance with count == 0.")
        return max(0.0, self.m2 / float(self.count))


def compute_dprime(
    loop_stats: RunningScalarStats,
    normal_stats: RunningScalarStats,
    eps: float = 1e-12,
) -> float:
    """Compute signed held-out d-prime between loop and normal projections."""
    denom = math.sqrt(0.5 * (loop_stats.variance + normal_stats.variance) + eps)
    return (loop_stats.mean - normal_stats.mean) / denom


# ---------------------------------------------------------------------------
# Sequence representation & 16-gram loop labeling
# ---------------------------------------------------------------------------


@dataclass
class LabeledSequence:
    prompt_ids: List[int]
    gen_ids: List[int]
    loop_positions: List[int]    # indices into full_ids = prompt_ids + gen_ids
    normal_positions: List[int]  # indices into full_ids = prompt_ids + gen_ids
    has_loop: bool


def label_loop_tokens(
    gen_ids: Sequence[int],
    ngram_n: int = 16,
    window_size: int = 512,
) -> Tuple[List[int], List[int], bool]:
    """Label response token offsets as looping vs. normal pre-loop.

    A token belongs to the looping set if it is covered by any n-gram that
    previously occurred within the last `window_size` tokens.
    Tokens strictly before the first looping token are labeled as normal
    pre-loop response tokens. If no repeated n-gram occurs, all response tokens
    are labeled normal.

    Returns:
        (loop_gen_offsets, normal_gen_offsets, has_loop)
    """
    seq_len = len(gen_ids)
    if seq_len == 0:
        return [], [], False
    if seq_len < ngram_n:
        return [], list(range(seq_len)), False

    is_loop = [False] * seq_len
    last_seen: Dict[Tuple[int, ...], int] = {}

    for start in range(seq_len - ngram_n + 1):
        gram = tuple(gen_ids[start : start + ngram_n])
        prev = last_seen.get(gram)
        if prev is not None and (start - prev) <= window_size:
            for pos in range(start, start + ngram_n):
                is_loop[pos] = True
        last_seen[gram] = start

    loop_offsets = [i for i, flag in enumerate(is_loop) if flag]
    if not loop_offsets:
        return [], list(range(seq_len)), False

    first_loop = loop_offsets[0]
    normal_offsets = list(range(first_loop))
    return loop_offsets, normal_offsets, True


def split_sequences_stratified(
    sequences: Sequence[LabeledSequence],
    seed: int = 42,
) -> Tuple[List[LabeledSequence], List[LabeledSequence]]:
    """Split sequences into two disjoint halves (A, B), stratified by has_loop."""
    rng = random.Random(seed)
    looping = [s for s in sequences if s.has_loop]
    non_looping = [s for s in sequences if not s.has_loop]
    rng.shuffle(looping)
    rng.shuffle(non_looping)

    half_a: List[LabeledSequence] = []
    half_b: List[LabeledSequence] = []

    for i, seq in enumerate(looping):
        (half_a if i % 2 == 0 else half_b).append(seq)
    for i, seq in enumerate(non_looping):
        (half_a if i % 2 == 0 else half_b).append(seq)

    rng.shuffle(half_a)
    rng.shuffle(half_b)
    return half_a, half_b


# ---------------------------------------------------------------------------
# Model discovery, loading, and saving
# ---------------------------------------------------------------------------


_DOWN_PROJ_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj$")


def find_mlp_down_proj_layers(model: nn.Module) -> List[Tuple[int, str, nn.Linear]]:
    """Locate all transformer text-backbone mlp.down_proj Linear layers."""
    found: List[Tuple[int, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if "visual" in name or "vision" in name:
            continue
        match = _DOWN_PROJ_RE.search(name)
        if match is not None and isinstance(module, nn.Linear):
            layer_idx = int(match.group(1))
            found.append((layer_idx, name, module))
    found.sort(key=lambda item: item[0])
    if not found:
        raise RuntimeError(
            "No text-backbone `layers.<idx>.mlp.down_proj` nn.Linear modules found."
        )
    return found


def select_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Use bfloat16 on SM >= 8.0 GPUs, float16 on older GPUs, float32 on CPU."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def load_model_and_tokenizer(
    base_dir: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load local HuggingFace model & tokenizer cleanly for inference & editing."""
    import transformers
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    config = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
    archs = getattr(config, "architectures", None) or []

    candidate_classes = []
    for arch in archs:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            candidate_classes.append(cls)
    candidate_classes.append(AutoModelForCausalLM)
    if hasattr(transformers, "AutoModelForImageTextToText"):
        candidate_classes.append(transformers.AutoModelForImageTextToText)
    candidate_classes.append(AutoModel)

    last_err: Optional[Exception] = None
    model = None
    for cls in candidate_classes:
        try:
            model = cls.from_pretrained(
                base_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            break
        except Exception as exc:
            last_err = exc

    if model is None:
        raise RuntimeError(f"Failed to load model from {base_dir}: {last_err}")

    model.to(device)
    model.eval()
    return model, tok


def save_edited_checkpoint(model: nn.Module, tok, base_dir: str, out_dir: str) -> None:
    """Save edited model & tokenizer to out_dir so vLLM can load it directly."""
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    # Copy auxiliary non-weight metadata files (e.g. chat_template.jinja,
    # preprocessor_config.json) if present in base_dir and missing in out_dir.
    base_path = Path(base_dir)
    out_path = Path(out_dir)
    skip_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    if base_path.is_dir():
        for src in base_path.iterdir():
            if not src.is_file():
                continue
            if src.suffix in skip_suffixes or src.name.endswith(".index.json"):
                continue
            dst = out_path / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Dataset loading & greedy elicitation
# ---------------------------------------------------------------------------


def load_gsm8k_train_questions(n_prompts: int, seed: int) -> List[str]:
    """Load GSM8K train questions via hf_hub_download + pandas, with datasets fallback."""
    questions: Optional[List[str]] = None

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        parquet_path = hf_hub_download(
            repo_id="openai/gsm8k",
            filename="main/train-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(parquet_path)
        questions = [str(q) for q in df["question"].tolist()]
    except Exception:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="train")
        questions = [str(q) for q in ds["question"]]

    rng = random.Random(seed)
    indices = list(range(len(questions)))
    rng.shuffle(indices)
    selected = [questions[i] for i in indices[: min(n_prompts, len(questions))]]
    return selected


def format_thinking_prompt(tok, question: str) -> str:
    """Format a GSM8K question with chat template and enable_thinking=True."""
    messages = [{"role": "user", "content": question}]
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def elicit_greedy_sequences(
    model: nn.Module,
    tok,
    questions: Sequence[str],
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    ngram_n: int = 16,
    window_size: int = 512,
) -> List[LabeledSequence]:
    """Run batched greedy generation and label loop vs. normal pre-loop tokens."""
    prompts = [format_thinking_prompt(tok, q) for q in questions]
    tok.padding_side = "left"

    gen_cfg = copy.deepcopy(model.generation_config)
    gen_cfg.do_sample = False
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.max_length = None
    gen_cfg.pad_token_id = tok.pad_token_id

    eos_ids = gen_cfg.eos_token_id
    if isinstance(eos_ids, int):
        eos_set = {eos_ids}
    elif isinstance(eos_ids, (list, tuple)):
        eos_set = set(eos_ids)
    else:
        eos_set = {tok.eos_token_id} if tok.eos_token_id is not None else set()

    labeled: List[LabeledSequence] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        padded_prompt_len = int(input_ids.shape[1])

        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
                use_cache=True,
            )

        out_cpu = out_ids.cpu().tolist()
        mask_cpu = enc["attention_mask"].cpu().tolist()
        inp_cpu = enc["input_ids"].cpu().tolist()

        for row_inp, row_mask, row_out in zip(inp_cpu, mask_cpu, out_cpu):
            prompt_ids = [t for t, m in zip(row_inp, row_mask) if m == 1]
            raw_gen = row_out[padded_prompt_len:]

            # Truncate at first EOS (inclusive) or strip trailing pad tokens
            gen_ids: List[int] = []
            for tid in raw_gen:
                if tid == tok.pad_token_id and tid not in eos_set:
                    break
                gen_ids.append(tid)
                if tid in eos_set:
                    break

            loop_off, norm_off, has_loop = label_loop_tokens(
                gen_ids, ngram_n=ngram_n, window_size=window_size
            )
            p_len = len(prompt_ids)
            labeled.append(
                LabeledSequence(
                    prompt_ids=prompt_ids,
                    gen_ids=gen_ids,
                    loop_positions=[p_len + i for i in loop_off],
                    normal_positions=[p_len + i for i in norm_off],
                    has_loop=has_loop,
                )
            )

        del input_ids, attention_mask, out_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return labeled


# ---------------------------------------------------------------------------
# Streaming teacher-forced passes & weight-space edit math
# ---------------------------------------------------------------------------


def run_pass_a_directions(
    model: nn.Module,
    layers: Sequence[Tuple[int, str, nn.Linear]],
    half_a: Sequence[LabeledSequence],
    device: torch.device,
) -> Tuple[
    Dict[int, torch.Tensor],
    Dict[int, RunningVectorMean],
    Dict[int, RunningVectorMean],
]:
    """Pass A: stream y and h means on Half A and compute unit loop directions u_l."""
    y_loop_stats: Dict[int, RunningVectorMean] = {}
    y_norm_stats: Dict[int, RunningVectorMean] = {}
    h_loop_stats: Dict[int, RunningVectorMean] = {}
    h_norm_stats: Dict[int, RunningVectorMean] = {}

    for layer_idx, _, mod in layers:
        y_loop_stats[layer_idx] = RunningVectorMean(mod.out_features)
        y_norm_stats[layer_idx] = RunningVectorMean(mod.out_features)
        h_loop_stats[layer_idx] = RunningVectorMean(mod.in_features)
        h_norm_stats[layer_idx] = RunningVectorMean(mod.in_features)

    current_loop_idx: List[torch.Tensor] = [torch.empty(0, dtype=torch.long)]
    current_norm_idx: List[torch.Tensor] = [torch.empty(0, dtype=torch.long)]

    def make_hook(l_idx: int):
        def _hook(_mod, inputs, output):
            h = inputs[0]
            y = output[0] if isinstance(output, tuple) else output
            l_pos = current_loop_idx[0]
            n_pos = current_norm_idx[0]
            if l_pos.numel() > 0:
                h_loop_stats[l_idx].update(h[0, l_pos])
                y_loop_stats[l_idx].update(y[0, l_pos])
            if n_pos.numel() > 0:
                h_norm_stats[l_idx].update(h[0, n_pos])
                y_norm_stats[l_idx].update(y[0, n_pos])
        return _hook

    handles = [mod.register_forward_hook(make_hook(l_idx)) for l_idx, _, mod in layers]
    try:
        with torch.inference_mode():
            for seq in half_a:
                if not seq.loop_positions and not seq.normal_positions:
                    continue
                full_ids = torch.tensor(
                    [seq.prompt_ids + seq.gen_ids], dtype=torch.long, device=device
                )
                current_loop_idx[0] = torch.tensor(
                    seq.loop_positions, dtype=torch.long, device=device
                )
                current_norm_idx[0] = torch.tensor(
                    seq.normal_positions, dtype=torch.long, device=device
                )
                _ = model(input_ids=full_ids, use_cache=False)
                del full_ids
    finally:
        for h in handles:
            h.remove()

    directions: Dict[int, torch.Tensor] = {}
    for layer_idx, _, _ in layers:
        if y_loop_stats[layer_idx].count == 0 or y_norm_stats[layer_idx].count == 0:
            raise RuntimeError(
                f"Half A missing labeled tokens at layer {layer_idx}: "
                f"loop={y_loop_stats[layer_idx].count}, "
                f"normal={y_norm_stats[layer_idx].count}"
            )
        diff = y_loop_stats[layer_idx].mean - y_norm_stats[layer_idx].mean
        norm = float(torch.linalg.vector_norm(diff).item())
        if norm < 1e-12:
            raise RuntimeError(f"Zero loop direction norm on Half A at layer {layer_idx}.")
        directions[layer_idx] = diff / norm

    return directions, h_loop_stats, h_norm_stats


def run_pass_b_dprimes(
    model: nn.Module,
    layers: Sequence[Tuple[int, str, nn.Linear]],
    half_b: Sequence[LabeledSequence],
    directions: Dict[int, torch.Tensor],
    h_loop_stats: Dict[int, RunningVectorMean],
    h_norm_stats: Dict[int, RunningVectorMean],
    device: torch.device,
) -> Dict[int, float]:
    """Pass B: evaluate held-out d-prime along u_l and keep accumulating h means."""
    proj_loop_stats: Dict[int, RunningScalarStats] = {}
    proj_norm_stats: Dict[int, RunningScalarStats] = {}
    u_device: Dict[int, torch.Tensor] = {}

    for layer_idx, _, _ in layers:
        proj_loop_stats[layer_idx] = RunningScalarStats()
        proj_norm_stats[layer_idx] = RunningScalarStats()
        u_device[layer_idx] = directions[layer_idx].to(device=device, dtype=torch.float32)

    current_loop_idx: List[torch.Tensor] = [torch.empty(0, dtype=torch.long)]
    current_norm_idx: List[torch.Tensor] = [torch.empty(0, dtype=torch.long)]

    def make_hook(l_idx: int):
        u_vec = u_device[l_idx]

        def _hook(_mod, inputs, output):
            h = inputs[0]
            y = output[0] if isinstance(output, tuple) else output
            l_pos = current_loop_idx[0]
            n_pos = current_norm_idx[0]
            if l_pos.numel() > 0:
                h_loop_stats[l_idx].update(h[0, l_pos])
                proj_l = torch.mv(y[0, l_pos].to(torch.float32), u_vec)
                proj_loop_stats[l_idx].update(proj_l)
            if n_pos.numel() > 0:
                h_norm_stats[l_idx].update(h[0, n_pos])
                proj_n = torch.mv(y[0, n_pos].to(torch.float32), u_vec)
                proj_norm_stats[l_idx].update(proj_n)
        return _hook

    handles = [mod.register_forward_hook(make_hook(l_idx)) for l_idx, _, mod in layers]
    try:
        with torch.inference_mode():
            for seq in half_b:
                if not seq.loop_positions and not seq.normal_positions:
                    continue
                full_ids = torch.tensor(
                    [seq.prompt_ids + seq.gen_ids], dtype=torch.long, device=device
                )
                current_loop_idx[0] = torch.tensor(
                    seq.loop_positions, dtype=torch.long, device=device
                )
                current_norm_idx[0] = torch.tensor(
                    seq.normal_positions, dtype=torch.long, device=device
                )
                _ = model(input_ids=full_ids, use_cache=False)
                del full_ids
    finally:
        for h in handles:
            h.remove()

    dprimes: Dict[int, float] = {}
    for layer_idx, _, _ in layers:
        if proj_loop_stats[layer_idx].count == 0 or proj_norm_stats[layer_idx].count == 0:
            raise RuntimeError(
                f"Half B missing labeled tokens at layer {layer_idx}: "
                f"loop={proj_loop_stats[layer_idx].count}, "
                f"normal={proj_norm_stats[layer_idx].count}"
            )
        dprimes[layer_idx] = compute_dprime(
            proj_loop_stats[layer_idx], proj_norm_stats[layer_idx]
        )

    return dprimes


def score_and_edit_down_proj(
    weight: torch.Tensor,
    u_l: torch.Tensor,
    mean_h_loop: torch.Tensor,
    mean_h_normal: torch.Tensor,
    top_n_neurons: int = 16,
    alpha: float = 12.0,
) -> Tuple[torch.Tensor, List[int], List[float]]:
    """Compute neuron contributions along u_l and return edited weight matrix.

    Args:
        weight: down_proj weight matrix W of shape (d_model, d_mlp).
        u_l: unit loop direction of shape (d_model,).
        mean_h_loop: mean down_proj input on loop tokens, shape (d_mlp,).
        mean_h_normal: mean down_proj input on normal tokens, shape (d_mlp,).
        top_n_neurons: number of top-scoring neurons to suppress.
        alpha: suppression multiplier (default 12.0).

    Returns:
        (W_new in original dtype/device, chosen_neuron_indices, chosen_scores)
    """
    orig_dtype = weight.dtype
    orig_device = weight.device

    w64 = weight.detach().to(dtype=torch.float64, device="cpu")
    u64 = u_l.detach().to(dtype=torch.float64, device="cpu")
    u64 = u64 / torch.linalg.vector_norm(u64)

    delta_h = (
        mean_h_loop.detach().to(dtype=torch.float64, device="cpu")
        - mean_h_normal.detach().to(dtype=torch.float64, device="cpu")
    )

    # proj_w[i] = u_l^T W[:, i]
    proj_w = torch.mv(w64.t(), u64)
    scores = delta_h * proj_w

    k = min(top_n_neurons, int(scores.numel()))
    top_vals, top_idx = torch.topk(scores, k=k, largest=True, sorted=True)

    mask = torch.zeros_like(scores)
    mask[top_idx] = 1.0

    # W_new = W - alpha * u_l * (m * (u_l^T W))
    delta_w = alpha * torch.outer(u64, mask * proj_w)
    w_new = (w64 - delta_w).to(dtype=orig_dtype, device=orig_device)

    return w_new, [int(i) for i in top_idx.tolist()], [float(v) for v in top_vals.tolist()]


# ---------------------------------------------------------------------------
# Atomic JSON writer
# ---------------------------------------------------------------------------


def write_json_atomic(path: str, payload: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# CLI & main pipeline
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blind weight-space loop-fix reconstruction for Qwen3.5-0.8B."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Local HuggingFace model directory (e.g. /content/base).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for the edited HuggingFace checkpoint.",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=64,
        help="Number of GSM8K train questions to elicit loops (default: 64).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens per greedy generation (default: 2048).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for greedy generation (default: 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt sampling and Half A/B split (default: 42).",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to write JSON summary.",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=16,
        help="Repeated n-gram length for loop detection (default: 16).",
    )
    parser.add_argument(
        "--loop-window",
        type=int,
        default=512,
        help="Sliding window length for repeated n-gram detection (default: 512).",
    )
    parser.add_argument(
        "--top-k-layers",
        type=int,
        default=2,
        help="Number of top d-prime layers to edit (default: 2).",
    )
    parser.add_argument(
        "--top-n-neurons",
        type=int,
        default=16,
        help="Number of top MLP neurons per chosen layer to suppress (default: 16).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=12.0,
        help="Suppression multiplier along u_l (default: 12.0).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    t_total_start = time.perf_counter()
    timing: Dict[str, float] = {}

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, dtype = select_device_and_dtype()
    print(f"[init] device={device}, dtype={dtype}, seed={args.seed}")

    # 1. Load model & tokenizer
    t0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(args.base_dir, device=device, dtype=dtype)
    layers = find_mlp_down_proj_layers(model)
    timing["load_s"] = round(time.perf_counter() - t0, 3)
    print(
        f"[load] found {len(layers)} mlp.down_proj layers "
        f"(indices {layers[0][0]}..{layers[-1][0]}) in {timing['load_s']}s"
    )

    # 2. Load GSM8K train prompts & elicit greedy responses
    t0 = time.perf_counter()
    questions = load_gsm8k_train_questions(args.n_prompts, seed=args.seed)
    sequences = elicit_greedy_sequences(
        model=model,
        tok=tok,
        questions=questions,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=device,
        ngram_n=args.ngram,
        window_size=args.loop_window,
    )
    timing["generate_s"] = round(time.perf_counter() - t0, 3)

    n_loop_seqs = sum(1 for s in sequences if s.has_loop)
    n_loop_tokens = sum(len(s.loop_positions) for s in sequences)
    n_normal_tokens = sum(len(s.normal_positions) for s in sequences)
    print(
        f"[elicit] n_prompts={len(sequences)}, n_loop_seqs={n_loop_seqs} "
        f"({100.0 * n_loop_seqs / max(1, len(sequences)):.1f}%), "
        f"n_loop_tokens={n_loop_tokens}, n_normal_tokens={n_normal_tokens}, "
        f"time={timing['generate_s']}s"
    )

    if n_loop_seqs < 2:
        raise RuntimeError(
            f"Need at least 2 looping sequences for disjoint Half A/B split, "
            f"got {n_loop_seqs} (out of {len(sequences)} prompts)."
        )

    half_a, half_b = split_sequences_stratified(sequences, seed=args.seed)
    print(
        f"[split] Half A: {len(half_a)} seqs "
        f"({sum(s.has_loop for s in half_a)} looping), "
        f"Half B: {len(half_b)} seqs "
        f"({sum(s.has_loop for s in half_b)} looping)"
    )

    # 3. Pass A: compute unit loop directions u_l on Half A
    t0 = time.perf_counter()
    directions, h_loop_stats, h_norm_stats = run_pass_a_directions(
        model=model, layers=layers, half_a=half_a, device=device
    )
    timing["pass_a_s"] = round(time.perf_counter() - t0, 3)
    print(f"[pass_a] computed {len(directions)} unit loop directions in {timing['pass_a_s']}s")

    # 4. Pass B: rank all layers by held-out d-prime along u_l
    t0 = time.perf_counter()
    layer_dprimes = run_pass_b_dprimes(
        model=model,
        layers=layers,
        half_b=half_b,
        directions=directions,
        h_loop_stats=h_loop_stats,
        h_norm_stats=h_norm_stats,
        device=device,
    )
    timing["pass_b_s"] = round(time.perf_counter() - t0, 3)

    ranked_layers = sorted(layer_dprimes.items(), key=lambda kv: kv[1], reverse=True)
    chosen_layers = [layer_idx for layer_idx, _ in ranked_layers[: args.top_k_layers]]
    top_preview = ", ".join(f"L{idx}={val:.3f}" for idx, val in ranked_layers[:5])
    print(
        f"[pass_b] top held-out d': {top_preview} -> chosen={chosen_layers} "
        f"in {timing['pass_b_s']}s"
    )

    # 5. Score neurons and apply rank-1 sparse weight edit in chosen layers
    t0 = time.perf_counter()
    layer_map = {layer_idx: (name, mod) for layer_idx, name, mod in layers}
    chosen_neurons: Dict[str, List[int]] = {}
    chosen_neuron_scores: Dict[str, List[float]] = {}

    for layer_idx in chosen_layers:
        mod_name, mod = layer_map[layer_idx]
        w_new, top_idx, top_scores = score_and_edit_down_proj(
            weight=mod.weight,
            u_l=directions[layer_idx],
            mean_h_loop=h_loop_stats[layer_idx].mean,
            mean_h_normal=h_norm_stats[layer_idx].mean,
            top_n_neurons=args.top_n_neurons,
            alpha=args.alpha,
        )
        with torch.no_grad():
            mod.weight.copy_(w_new)
        chosen_neurons[str(layer_idx)] = top_idx
        chosen_neuron_scores[str(layer_idx)] = [round(s, 6) for s in top_scores]
        print(
            f"[edit] L{layer_idx} ({mod_name}): suppressed {len(top_idx)} neurons "
            f"(alpha={args.alpha}), top_idx={top_idx[:6]}..."
        )
    timing["edit_s"] = round(time.perf_counter() - t0, 3)

    # 6. Save edited checkpoint
    t0 = time.perf_counter()
    save_edited_checkpoint(model=model, tok=tok, base_dir=args.base_dir, out_dir=args.out_dir)
    timing["save_s"] = round(time.perf_counter() - t0, 3)
    timing["total_s"] = round(time.perf_counter() - t_total_start, 3)
    print(f"[save] checkpoint written to {args.out_dir} in {timing['save_s']}s (total {timing['total_s']}s)")

    # 7. Write optional summary JSON
    if args.summary_out:
        summary = {
            "base_dir": args.base_dir,
            "out_dir": args.out_dir,
            "n_prompts": len(sequences),
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "ngram": args.ngram,
            "loop_window": args.loop_window,
            "top_k_layers": args.top_k_layers,
            "top_n_neurons": args.top_n_neurons,
            "alpha": args.alpha,
            "chosen_layers": chosen_layers,
            "layer_dprimes": {str(k): round(v, 6) for k, v in sorted(layer_dprimes.items())},
            "chosen_neurons": chosen_neurons,
            "chosen_neuron_scores": chosen_neuron_scores,
            "n_loop_seqs": n_loop_seqs,
            "n_loop_tokens": n_loop_tokens,
            "n_normal_tokens": n_normal_tokens,
            "timing": timing,
        }
        write_json_atomic(args.summary_out, summary)
        print(f"[summary] wrote {args.summary_out}")


if __name__ == "__main__":
    main()
