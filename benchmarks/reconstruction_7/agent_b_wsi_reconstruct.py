#!/usr/bin/env python3
"""Blind WSI-powered reconstruction of the Qwen3.5-0.8B 16-gram thinking-loop fix.

Pipeline:
1. Elicit greedy responses on GSM8K train prompts (enable_thinking=True) and label
   repeated 16-gram looping tokens vs. normal pre-loop response tokens.
2. Split sequences into disjoint Half A and Half B. Using `wsi.capture.find_mlps`,
   run OOM-safe streaming teacher-forced passes (`use_cache=False`, `logits_to_keep=1`)
   hooking every language-model `mlp.down_proj` input `h` and output `y`.
3. On Half A, compute each layer's unit difference-of-means direction `u_l` and mean
   MLP neuron activation shift `mean_loop(h) - mean_normal(h)`. On Half B, rank all
   layers by held-out `d'` along `u_l`.
4. Build candidate `wsi.moipatch.MoIPatch`es (top-k pairs among the top-ranked `d'`
   layers, 16 neurons/layer, alpha=12.0) and run 1-forward-pass causal screening via
   `wsi.tracer.screen_patch_candidates(..., rank_by="pareto", max_nll_delta=0.05)`
   on held-out Half-B sequences.
5. Bake the winning `MoIPatch` directly into `--out-dir` (preserving original shard
   dtypes, keys, and config/tokenizer files so vLLM loads it verbatim), and optionally
   save `--patch-out` (`.moipatch`) and `--summary-out` (`.json`).
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "wsi").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wsi.capture import find_mlps
from wsi.moipatch import Edit, MoIPatch, apply_edit, module_suffix
from wsi.tracer import screen_patch_candidates

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


# ---------------------------------------------------------------------------
# 1. 16-gram loop detection
# ---------------------------------------------------------------------------
def label_loop_tokens(
    resp_ids: list[int],
    *,
    ngram: int = 16,
    window: int = 512,
) -> dict[str, Any]:
    """Label looping tokens (repeated `ngram` within `window`) vs. pre-loop normal tokens.

    Returns a dict with:
      - `is_loop`: bool
      - `onset`: first response index where a repeated `ngram` begins (or None)
      - `first_copy`: earlier index of the first repeated `ngram` (or None)
      - `loop_positions`: sorted list of response indices covered by repeated `ngram`s
      - `normal_positions`: sorted list of pre-loop response indices
    """
    n = len(resp_ids)
    if n < ngram:
        return {
            "is_loop": False,
            "onset": None,
            "first_copy": None,
            "loop_positions": [],
            "normal_positions": list(range(n)),
        }

    last_seen: dict[tuple[int, ...], int] = {}
    loop_mask = [False] * n
    onset: int | None = None
    first_copy: int | None = None

    for t in range(n - ngram + 1):
        gram = tuple(resp_ids[t : t + ngram])
        prev = last_seen.get(gram)
        if prev is not None and 1 <= (t - prev) <= window:
            if onset is None:
                onset = t
                first_copy = prev
            for pos in range(t, t + ngram):
                loop_mask[pos] = True
        last_seen[gram] = t

    if onset is None:
        return {
            "is_loop": False,
            "onset": None,
            "first_copy": None,
            "loop_positions": [],
            "normal_positions": list(range(n)),
        }

    # Pre-loop normal tokens: strictly before the first copy if enough tokens exist,
    # otherwise strictly before onset.
    norm_end = first_copy if (first_copy is not None and first_copy >= 16) else onset
    normal_positions = list(range(0, norm_end))
    loop_positions = [i for i, flag in enumerate(loop_mask) if flag]
    return {
        "is_loop": True,
        "onset": onset,
        "first_copy": first_copy,
        "loop_positions": loop_positions,
        "normal_positions": normal_positions,
    }


# ---------------------------------------------------------------------------
# 2. Model & dataset loading + greedy elicitation
# ---------------------------------------------------------------------------
def _select_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _dtype_to_str(dt: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dt, "float32")


def load_model_and_tokenizer(base_dir: str | Path, device: torch.device) -> tuple[nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _select_dtype(device)
    tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=dtype, **load_kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText

        try:
            model = AutoModelForImageTextToText.from_pretrained(str(base_dir), dtype=dtype, **load_kwargs)
        except TypeError:
            model = AutoModelForImageTextToText.from_pretrained(str(base_dir), torch_dtype=dtype, **load_kwargs)

    model.to(device).eval()
    return model, tok


def load_gsm8k_prompts(n_prompts: int, seed: int) -> list[str]:
    from datasets import load_dataset

    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception:
        ds = load_dataset("gsm8k", "main", split="train")

    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    return [str(ds[i]["question"]) for i in idxs[: min(n_prompts, len(idxs))]]


def format_chat_prompt(tok: Any, question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
    return question


@torch.no_grad()
def elicit_sequences(
    model: nn.Module,
    tok: Any,
    questions: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Generate greedy responses and label looping vs. pre-loop normal tokens."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)
    eos_id = tok.eos_token_id

    formatted = [format_chat_prompt(tok, q) for q in questions]
    records: list[dict[str, Any]] = []

    for start in range(0, len(formatted), batch_size):
        batch_texts = formatted[start : start + batch_size]
        enc = tok(batch_texts, return_tensors="pt", padding=True, truncation=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        padded_prompt_len = int(input_ids.shape[1])

        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": pad_id,
        }
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id

        out_ids = model.generate(**gen_kwargs).cpu()
        attn_cpu = attention_mask.cpu()

        for b in range(out_ids.shape[0]):
            # Strip left padding from the prompt
            non_pad = int(attn_cpu[b].sum().item())
            left_pad = padded_prompt_len - non_pad
            prompt_ids = out_ids[b, left_pad:padded_prompt_len].tolist()
            resp_raw = out_ids[b, padded_prompt_len:].tolist()

            # Trim trailing pad/eos tokens in response
            resp_ids: list[int] = []
            for tid in resp_raw:
                if tid == pad_id and (eos_id is None or tid != eos_id):
                    break
                resp_ids.append(tid)
                if eos_id is not None and tid == eos_id:
                    break

            full_ids = prompt_ids + resp_ids
            prompt_len = len(prompt_ids)
            lbl = label_loop_tokens(resp_ids, ngram=16, window=512)
            records.append({
                "ids": full_ids,
                "prompt_len": prompt_len,
                "resp_len": len(resp_ids),
                "is_loop": lbl["is_loop"],
                "onset": lbl["onset"],
                "first_copy": lbl["first_copy"],
                "loop_abs": [prompt_len + p for p in lbl["loop_positions"]],
                "normal_abs": [prompt_len + p for p in lbl["normal_positions"]],
            })

        del input_ids, attention_mask, out_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return records


# ---------------------------------------------------------------------------
# 3. Disjoint Half-A / Half-B split & streaming teacher-forced hooks
# ---------------------------------------------------------------------------
def split_half_ab(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified deterministic split into disjoint Half A and Half B."""
    loops = [r for r in records if r["is_loop"] and r["loop_abs"]]
    normals = [r for r in records if not (r["is_loop"] and r["loop_abs"])]

    half_a = loops[0::2] + normals[0::2]
    half_b = loops[1::2] + normals[1::2]

    # Graceful fallback if only 1 looping sequence was elicited
    if len(loops) == 1:
        r = loops[0]
        mid_l = max(1, len(r["loop_abs"]) // 2)
        mid_n = max(1, len(r["normal_abs"]) // 2) if r["normal_abs"] else 0
        r_a = dict(r, loop_abs=r["loop_abs"][:mid_l], normal_abs=r["normal_abs"][:mid_n])
        r_b = dict(r, loop_abs=r["loop_abs"][mid_l:], normal_abs=r["normal_abs"][mid_n:])
        half_a = [r_a] + normals[0::2]
        half_b = [r_b] + normals[1::2]

    return half_a, half_b


def discover_lm_down_projs(model: nn.Module) -> dict[int, tuple[str, nn.Module]]:
    """Use `wsi.capture.find_mlps` to locate every language-model `mlp.down_proj`."""
    mods = dict(model.named_modules())
    by_layer: dict[int, tuple[str, nn.Module]] = {}
    for g in find_mlps(model):
        name = g.down
        if "visual" in name or "mtp" in name:
            continue
        m = _LAYER_RE.search(name)
        if not m:
            continue
        li = int(m.group(1))
        if li not in by_layer:
            by_layer[li] = (name, mods[name])
    if not by_layer:
        raise RuntimeError("No language-model MLP down_proj modules found via wsi.capture.find_mlps.")
    return dict(sorted(by_layer.items()))


def _forward_no_logits(model: nn.Module, ids_2d: torch.Tensor, use_ltk: list[bool]) -> None:
    """Teacher-forced forward pass without KV cache or full [T, vocab] logits."""
    if use_ltk[0]:
        try:
            model(input_ids=ids_2d, use_cache=False, logits_to_keep=1)
            return
        except TypeError:
            use_ltk[0] = False
    model(input_ids=ids_2d, use_cache=False)


@torch.no_grad()
def fit_half_a_directions(
    model: nn.Module,
    down_projs: dict[int, tuple[str, nn.Module]],
    half_a: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Stream Half-A teacher-forced passes to compute `u_l` and `mean_h_diff[l]`."""
    sum_y_loop: dict[int, torch.Tensor] = {}
    sum_y_norm: dict[int, torch.Tensor] = {}
    sum_h_loop: dict[int, torch.Tensor] = {}
    sum_h_norm: dict[int, torch.Tensor] = {}
    n_loop = 0
    n_norm = 0

    cur_loop_idx: list[int] = []
    cur_norm_idx: list[int] = []

    def _make_hook(layer_idx: int):
        def _hook(_mod: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            h = inp[0][0]  # [T, U]
            y = (out[0] if isinstance(out, (tuple, list)) else out)[0]  # [T, D]
            if cur_loop_idx:
                idx = torch.tensor(cur_loop_idx, device=h.device, dtype=torch.long)
                yl = y.index_select(0, idx).float().sum(dim=0).double().cpu()
                hl = h.index_select(0, idx).float().sum(dim=0).double().cpu()
                if layer_idx not in sum_y_loop:
                    sum_y_loop[layer_idx] = yl
                    sum_h_loop[layer_idx] = hl
                else:
                    sum_y_loop[layer_idx] += yl
                    sum_h_loop[layer_idx] += hl
            if cur_norm_idx:
                idx = torch.tensor(cur_norm_idx, device=h.device, dtype=torch.long)
                yn = y.index_select(0, idx).float().sum(dim=0).double().cpu()
                hn = h.index_select(0, idx).float().sum(dim=0).double().cpu()
                if layer_idx not in sum_y_norm:
                    sum_y_norm[layer_idx] = yn
                    sum_h_norm[layer_idx] = hn
                else:
                    sum_y_norm[layer_idx] += yn
                    sum_h_norm[layer_idx] += hn

        return _hook

    handles = [mod.register_forward_hook(_make_hook(li)) for li, (_, mod) in down_projs.items()]
    use_ltk = [True]
    try:
        for rec in half_a:
            if not rec["loop_abs"] and not rec["normal_abs"]:
                continue
            cur_loop_idx = rec["loop_abs"]
            cur_norm_idx = rec["normal_abs"]
            n_loop += len(cur_loop_idx)
            n_norm += len(cur_norm_idx)
            ids = torch.tensor(rec["ids"], dtype=torch.long, device=device).unsqueeze(0)
            _forward_no_logits(model, ids, use_ltk)
            del ids
    finally:
        for h in handles:
            h.remove()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if n_loop == 0 or n_norm == 0:
        raise RuntimeError(f"Half A needs both loop and normal tokens (got loop={n_loop}, normal={n_norm}).")

    directions: dict[int, torch.Tensor] = {}
    mean_h_diff: dict[int, torch.Tensor] = {}
    for li in down_projs:
        diff_y = (sum_y_loop[li] / n_loop) - (sum_y_norm[li] / n_norm)
        norm_y = float(diff_y.norm())
        directions[li] = diff_y / max(norm_y, 1e-12)
        mean_h_diff[li] = (sum_h_loop[li] / n_loop) - (sum_h_norm[li] / n_norm)

    return {
        "directions": directions,
        "mean_h_diff": mean_h_diff,
        "n_loop_tokens_a": n_loop,
        "n_normal_tokens_a": n_norm,
    }


@torch.no_grad()
def score_half_b_dprimes(
    model: nn.Module,
    down_projs: dict[int, tuple[str, nn.Module]],
    directions: dict[int, torch.Tensor],
    half_b: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Stream Half-B teacher-forced passes to compute held-out `d'` along `u_l`."""
    u_dev = {li: u.to(device=device, dtype=torch.float32) for li, u in directions.items()}
    sum_l = {li: 0.0 for li in down_projs}
    sum2_l = {li: 0.0 for li in down_projs}
    sum_n = {li: 0.0 for li in down_projs}
    sum2_n = {li: 0.0 for li in down_projs}
    n_loop = 0
    n_norm = 0

    cur_loop_idx: list[int] = []
    cur_norm_idx: list[int] = []

    def _make_hook(layer_idx: int):
        u_vec = u_dev[layer_idx]

        def _hook(_mod: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            y = (out[0] if isinstance(out, (tuple, list)) else out)[0]  # [T, D]
            if cur_loop_idx:
                idx = torch.tensor(cur_loop_idx, device=y.device, dtype=torch.long)
                proj = (y.index_select(0, idx).float() @ u_vec).double()
                sum_l[layer_idx] += float(proj.sum().item())
                sum2_l[layer_idx] += float((proj * proj).sum().item())
            if cur_norm_idx:
                idx = torch.tensor(cur_norm_idx, device=y.device, dtype=torch.long)
                proj = (y.index_select(0, idx).float() @ u_vec).double()
                sum_n[layer_idx] += float(proj.sum().item())
                sum2_n[layer_idx] += float((proj * proj).sum().item())

        return _hook

    handles = [mod.register_forward_hook(_make_hook(li)) for li, (_, mod) in down_projs.items()]
    use_ltk = [True]
    try:
        for rec in half_b:
            if not rec["loop_abs"] and not rec["normal_abs"]:
                continue
            cur_loop_idx = rec["loop_abs"]
            cur_norm_idx = rec["normal_abs"]
            n_loop += len(cur_loop_idx)
            n_norm += len(cur_norm_idx)
            ids = torch.tensor(rec["ids"], dtype=torch.long, device=device).unsqueeze(0)
            _forward_no_logits(model, ids, use_ltk)
            del ids
    finally:
        for h in handles:
            h.remove()
        del u_dev
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if n_loop == 0 or n_norm == 0:
        raise RuntimeError(f"Half B needs both loop and normal tokens (got loop={n_loop}, normal={n_norm}).")

    layer_dprimes: dict[int, float] = {}
    for li in down_projs:
        mu_l = sum_l[li] / n_loop
        var_l = max(sum2_l[li] / n_loop - mu_l * mu_l, 0.0)
        mu_n = sum_n[li] / n_norm
        var_n = max(sum2_n[li] / n_norm - mu_n * mu_n, 0.0)
        denom = math.sqrt(max(0.5 * (var_l + var_n), 1e-12))
        layer_dprimes[li] = float((mu_l - mu_n) / denom)

    ranked = sorted(layer_dprimes.keys(), key=lambda l: -layer_dprimes[l])
    return {
        "layer_dprimes": layer_dprimes,
        "ranked_layers": ranked,
        "n_loop_tokens_b": n_loop,
        "n_normal_tokens_b": n_norm,
    }


# ---------------------------------------------------------------------------
# 4. Candidate MoIPatch construction & 1-forward-pass Pareto screening
# ---------------------------------------------------------------------------
def select_layer_neurons(
    mod: nn.Module,
    u_l: torch.Tensor,
    mean_h_diff_l: torch.Tensor,
    top_k: int = 16,
) -> torch.Tensor:
    """Score every MLP neuron `i` by `(mean_loop(h_i) - mean_normal(h_i)) * (u_l^T W[:, i])`."""
    w = mod.weight.detach().cpu().double()
    if not isinstance(mod, nn.Linear):
        w = w.t()
    proj_w = u_l.double() @ w  # [U]
    scores = mean_h_diff_l.double() * proj_w
    k = min(int(top_k), int(scores.numel()))
    return scores.topk(k).indices.sort().values.long()


def build_candidate_moipatches(
    down_projs: dict[int, tuple[str, nn.Module]],
    directions: dict[int, torch.Tensor],
    mean_h_diff: dict[int, torch.Tensor],
    ranked_layers: list[int],
    *,
    top_pool: int = 4,
    n_neurons: int = 16,
    alpha: float = 12.0,
) -> tuple[dict[str, MoIPatch], dict[int, list[int]], dict[str, list[int]]]:
    """Build pair `MoIPatch` candidates among the top-ranked `d'` layers."""
    pool = ranked_layers[: min(top_pool, len(ranked_layers))]
    neurons_by_layer: dict[int, torch.Tensor] = {}
    for li in pool:
        _, mod = down_projs[li]
        neurons_by_layer[li] = select_layer_neurons(mod, directions[li], mean_h_diff[li], top_k=n_neurons)

    def _make_edit(li: int) -> Edit:
        full_name, mod = down_projs[li]
        try:
            mod_suf = module_suffix(full_name)
        except ValueError:
            mod_suf = full_name
        w = mod.weight.detach()
        return Edit(
            module=mod_suf,
            u=directions[li].double().cpu(),
            alpha=float(alpha),
            idx=neurons_by_layer[li].long().cpu(),
            base_sha256="",  # fingerprinted from checkpoint before saving .moipatch
            base_dtype=_dtype_to_str(w.dtype),
            shape=tuple(w.shape),
        )

    candidates: dict[str, MoIPatch] = {}
    cand_layers: dict[str, list[int]] = {}
    pairs = list(itertools.combinations(pool, 2)) if len(pool) >= 2 else [(pool[0],)]
    for pair in pairs:
        layers_sorted = sorted(pair)
        tag = f"L{'-'.join(str(x) for x in layers_sorted)}_n{n_neurons}_a{int(alpha) if alpha == int(alpha) else alpha}"
        edits = [_make_edit(li) for li in layers_sorted]
        candidates[tag] = MoIPatch(
            edits=edits,
            meta={
                "name": f"wsi-loop-fix-{tag}",
                "description": f"WSI reconstructed 16-gram thinking-loop fix on layers {layers_sorted}",
                "layers": layers_sorted,
                "n_neurons_per_layer": n_neurons,
                "alpha": float(alpha),
            },
        )
        cand_layers[tag] = layers_sorted

    neurons_json = {li: neurons_by_layer[li].tolist() for li in pool}
    return candidates, neurons_json, cand_layers


def build_screening_inputs(
    half_b: list[dict[str, Any]],
    *,
    max_seqs: int = 6,
    max_healthy_len: int = 512,
    max_target_tail: int = 256,
) -> tuple[list[list[int]], list[list[int]], list[tuple[int, int]]]:
    """Construct OOM-safe Half-B `healthy_seqs`, `target_seqs`, and `target_spans`."""
    healthy_seqs: list[list[int]] = []
    target_seqs: list[list[int]] = []
    target_spans: list[tuple[int, int]] = []

    loop_recs = [r for r in half_b if r["is_loop"] and r["onset"] is not None]
    norm_recs = [r for r in half_b if not r["is_loop"]]

    # Healthy sequences: prefer clean non-looping sequences, then pre-loop prefixes
    for r in norm_recs:
        if len(healthy_seqs) >= max_seqs:
            break
        seq = r["ids"][:max_healthy_len]
        if len(seq) >= 4:
            healthy_seqs.append(seq)
    for r in loop_recs:
        if len(healthy_seqs) >= max_seqs:
            break
        norm_end = r["prompt_len"] + (
            r["first_copy"] if (r["first_copy"] is not None and r["first_copy"] >= 16) else int(r["onset"])
        )
        seq = r["ids"][: min(norm_end, max_healthy_len)]
        if len(seq) >= 4:
            healthy_seqs.append(seq)

    # Target sequences: truncate shortly after onset so logits_to_keep stays small & OOM-safe
    for r in loop_recs[:max_seqs]:
        onset_abs = int(r["prompt_len"]) + int(r["onset"])
        span_end = min(len(r["ids"]), onset_abs + max_target_tail)
        if span_end - onset_abs < 2 or onset_abs < 1:
            continue
        target_seqs.append(r["ids"][:span_end])
        target_spans.append((onset_abs, span_end))

    return healthy_seqs, target_seqs, target_spans


# ---------------------------------------------------------------------------
# 5. Safetensors checkpoint baking (vLLM-ready)
# ---------------------------------------------------------------------------
def bake_moipatch_to_checkpoint(
    base_dir: str | Path,
    out_dir: str | Path,
    patch: MoIPatch,
    *,
    model: nn.Module | None = None,
    tok: Any | None = None,
) -> list[str]:
    """Apply `patch` directly onto `base_dir`'s safetensors shards into `out_dir`.

    Preserves original shard layout, weight dtypes, auxiliary tensors (e.g. visual/mtp),
    and all config/tokenizer files so vLLM loads `--out-dir` cleanly.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_dir = Path(base_dir)
    dst_dir = Path(out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    st_files = [
        p for p in sorted(src_dir.glob("*.safetensors"))
        if not p.name.startswith("edit_") and p.name != "delta_lowrank.safetensors"
    ]
    if not st_files:
        if model is None:
            raise FileNotFoundError(f"No *.safetensors found in {src_dir} and no fallback model provided.")
        h = patch.attach(model)
        h.on()
        try:
            model.save_pretrained(str(dst_dir))
            if tok is not None:
                tok.save_pretrained(str(dst_dir))
        finally:
            h.off()
        return [e.module for e in patch.edits]

    # Copy all non-safetensors auxiliary files (config, tokenizer, index json, chat templates, etc.)
    for item in sorted(src_dir.iterdir()):
        if item.is_file() and not item.name.endswith(".safetensors"):
            shutil.copy2(item, dst_dir / item.name)

    edits_by_mod = {e.module: e for e in patch.edits}
    applied_keys: list[str] = []

    for sf in st_files:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            meta = f.metadata()
            matched: dict[str, Edit] = {}
            for k in keys:
                if not k.endswith(".weight") or "mtp" in k or "visual" in k:
                    continue
                try:
                    suf = module_suffix(k[: -len(".weight")])
                except ValueError:
                    continue
                if suf in edits_by_mod:
                    matched[k] = edits_by_mod[suf]

            if not matched:
                shutil.copy2(sf, dst_dir / sf.name)
                continue

            tensors: dict[str, torch.Tensor] = {}
            for k in keys:
                w = f.get_tensor(k).clone()
                if k in matched:
                    e = matched[k]
                    w = apply_edit(w, e.u, e.idx, e.alpha)
                    applied_keys.append(k)
                tensors[k] = w

        save_file(tensors, str(dst_dir / sf.name), metadata=meta)
        del tensors

    if len(applied_keys) != len(patch.edits):
        raise RuntimeError(
            f"Expected to apply {len(patch.edits)} edits, but applied {len(applied_keys)} ({applied_keys})."
        )
    return applied_keys


# ---------------------------------------------------------------------------
# 6. End-to-end orchestration & CLI
# ---------------------------------------------------------------------------
def reconstruct_from_records(
    model: nn.Module,
    records: list[dict[str, Any]],
    *,
    base_dir: str | Path,
    out_dir: str | Path,
    tok: Any | None = None,
    device: torch.device | None = None,
    patch_out: str | Path | None = None,
    summary_out: str | Path | None = None,
    n_neurons: int = 16,
    alpha: float = 12.0,
    top_pool: int = 4,
    max_nll_delta: float = 0.05,
    timing_prefix: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run Half-A/B discovery, `screen_patch_candidates`, and checkpoint baking."""
    t_start = time.perf_counter()
    timing: dict[str, float] = dict(timing_prefix or {})
    dev = device if device is not None else next(model.parameters()).device

    half_a, half_b = split_half_ab(records)
    down_projs = discover_lm_down_projs(model)

    t0 = time.perf_counter()
    res_a = fit_half_a_directions(model, down_projs, half_a, dev)
    timing["trace_half_a_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    res_b = score_half_b_dprimes(model, down_projs, res_a["directions"], half_b, dev)
    timing["trace_half_b_s"] = round(time.perf_counter() - t0, 3)

    candidates, neurons_pool, cand_layers = build_candidate_moipatches(
        down_projs,
        res_a["directions"],
        res_a["mean_h_diff"],
        res_b["ranked_layers"],
        top_pool=top_pool,
        n_neurons=n_neurons,
        alpha=alpha,
    )

    t0 = time.perf_counter()
    healthy_seqs, target_seqs, target_spans = build_screening_inputs(half_b)
    if healthy_seqs and target_seqs:
        screen_ranking = screen_patch_candidates(
            model,
            candidates,
            healthy_seqs=healthy_seqs,
            target_seqs=target_seqs,
            target_spans=target_spans,
            max_nll_delta=max_nll_delta,
            rank_by="pareto",
        )
        winner_name = screen_ranking[0]["name"]
    else:
        screen_ranking = []
        winner_name = next(iter(candidates))
    timing["screen_s"] = round(time.perf_counter() - t0, 3)

    best_patch = candidates[winner_name]
    chosen_layers = cand_layers[winner_name]
    chosen_neurons = {str(li): neurons_pool[li] for li in chosen_layers}

    t0 = time.perf_counter()
    if patch_out:
        try:
            best_patch.fingerprint_from_checkpoint(base_dir)
        except Exception:
            pass
        best_patch.save(patch_out)

    applied_keys = bake_moipatch_to_checkpoint(base_dir, out_dir, best_patch, model=model, tok=tok)
    timing["save_s"] = round(time.perf_counter() - t0, 3)
    timing["reconstruct_s"] = round(time.perf_counter() - t_start, 3)

    n_loop_seqs = sum(1 for r in records if r["is_loop"])
    n_loop_tokens = res_a["n_loop_tokens_a"] + res_b["n_loop_tokens_b"]
    n_normal_tokens = res_a["n_normal_tokens_a"] + res_b["n_normal_tokens_b"]

    summary: dict[str, Any] = {
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "patch_out": str(patch_out) if patch_out else None,
        "winner_candidate": winner_name,
        "chosen_layers": chosen_layers,
        "chosen_neurons": chosen_neurons,
        "applied_weight_keys": applied_keys,
        "layer_dprimes": {str(k): round(v, 5) for k, v in sorted(res_b["layer_dprimes"].items())},
        "layer_ranking": [
            {"layer": li, "dprime": round(res_b["layer_dprimes"][li], 5)}
            for li in res_b["ranked_layers"]
        ],
        "screen_ranking": screen_ranking,
        "n_prompts": len(records),
        "n_loop_seqs": n_loop_seqs,
        "n_half_a_seqs": len(half_a),
        "n_half_b_seqs": len(half_b),
        "n_loop_tokens": n_loop_tokens,
        "n_normal_tokens": n_normal_tokens,
        "n_loop_tokens_a": res_a["n_loop_tokens_a"],
        "n_loop_tokens_b": res_b["n_loop_tokens_b"],
        "n_normal_tokens_a": res_a["n_normal_tokens_a"],
        "n_normal_tokens_b": res_b["n_normal_tokens_b"],
        "timing": timing,
    }

    if summary_out:
        sp = Path(summary_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        tmp = sp.with_suffix(sp.suffix + ".tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(sp)

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WSI reconstruction of the Qwen3.5-0.8B thinking-loop fix.")
    ap.add_argument("--base-dir", required=True, help="Local HuggingFace model directory")
    ap.add_argument("--out-dir", required=True, help="Output directory for edited HF checkpoint")
    ap.add_argument("--n-prompts", type=int, default=64, help="Number of GSM8K train questions")
    ap.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens per greedy generation")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for greedy generation")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--summary-out", default=None, help="Optional JSON summary output path")
    ap.add_argument("--patch-out", default=None, help="Optional .moipatch output path")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t_total = time.perf_counter()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(args.base_dir, device)
    timing["load_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    questions = load_gsm8k_prompts(args.n_prompts, args.seed)
    records = elicit_sequences(
        model,
        tok,
        questions,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=device,
    )
    timing["elicit_s"] = round(time.perf_counter() - t0, 3)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = reconstruct_from_records(
        model,
        records,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        tok=tok,
        device=device,
        patch_out=args.patch_out,
        summary_out=args.summary_out,
        timing_prefix=timing,
    )
    summary["timing"]["total_s"] = round(time.perf_counter() - t_total, 3)

    if args.summary_out:
        sp = Path(args.summary_out)
        tmp = sp.with_suffix(sp.suffix + ".tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(sp)

    print(json.dumps({
        "winner_candidate": summary["winner_candidate"],
        "chosen_layers": summary["chosen_layers"],
        "n_loop_seqs": summary["n_loop_seqs"],
        "n_prompts": summary["n_prompts"],
        "n_loop_tokens": summary["n_loop_tokens"],
        "n_normal_tokens": summary["n_normal_tokens"],
        "top5_layers": summary["layer_ranking"][:5],
        "screen_top3": summary["screen_ranking"][:3],
        "timing": summary["timing"],
    }, indent=2))


if __name__ == "__main__":
    main()
