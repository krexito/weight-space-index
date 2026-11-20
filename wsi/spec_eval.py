"""Base-draft speculative verification and suffix-index decoding for fast WSI patch evaluation.

When evaluating a candidate ``MoIPatch`` (or edited checkpoint) on a benchmark where the unedited
base model's responses are already cached (e.g. via ``WSI_BASE_CACHE_DIR``), most healthy tokens
early in a response match the base model closely. Instead of regenerating every token sequentially
one forward pass at a time, ``speculative_generate_with_base`` proposes blocks of ``block_size``
tokens from the cached base draft (and re-aligns via ``SuffixDraftIndex`` after divergence or
repetitions) and verifies each block in a single forward pass, preserving the exact target greedy
output or target sampling distribution.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .moipatch import MoIPatch


class SuffixDraftIndex:
    """N-gram / suffix lookup index over cached base sequences and prompt/corpus history."""

    def __init__(
        self,
        sequences: list[list[int]] | None = None,
        *,
        min_match: int = 2,
        max_match: int = 6,
        self_lookup: bool = True,
    ) -> None:
        if min_match < 1:
            raise ValueError(f"min_match must be >= 1, got {min_match}")
        if max_match < min_match:
            raise ValueError(f"max_match ({max_match}) must be >= min_match ({min_match})")
        self.min_match = int(min_match)
        self.max_match = int(max_match)
        self.self_lookup = bool(self_lookup)
        self.sequences: list[list[int]] = []
        # Map k -> {gram_tuple: [(seq_idx, end_pos), ...]} where seq[end_pos - k : end_pos] == gram_tuple
        self._index: dict[int, dict[tuple[int, ...], list[tuple[int, int]]]] = {
            k: defaultdict(list) for k in range(self.min_match, self.max_match + 1)
        }
        for seq in sequences or ():
            self.add_sequence(seq)

    def add_sequence(self, seq: list[int]) -> int:
        """Index all k-grams in ``seq`` and return its sequence index."""
        seq_list = [int(t) for t in seq]
        seq_idx = len(self.sequences)
        self.sequences.append(seq_list)
        n = len(seq_list)
        for k in range(self.min_match, min(self.max_match, n) + 1):
            table = self._index[k]
            for end_pos in range(k, n):
                gram = tuple(seq_list[end_pos - k : end_pos])
                table[gram].append((seq_idx, end_pos))
        return seq_idx

    def propose(
        self,
        context: list[int],
        max_tokens: int,
        *,
        near_pos: int | None = None,
    ) -> list[int]:
        """Propose up to ``max_tokens`` draft tokens matching the longest suffix of ``context``."""
        if max_tokens <= 0 or len(context) < self.min_match:
            return []

        top_k = min(self.max_match, len(context))
        for k in range(top_k, self.min_match - 1, -1):
            key = tuple(int(t) for t in context[-k:])
            hits = self._index[k].get(key)
            if hits:
                valid = [(si, ep) for si, ep in hits if ep < len(self.sequences[si])]
                if valid:
                    if near_pos is not None:
                        primary = [(si, ep) for si, ep in valid if si == 0]
                        if primary:
                            best_si, best_ep = min(primary, key=lambda x: (abs(x[1] - near_pos), -x[1]))
                            return list(self.sequences[best_si][best_ep : best_ep + max_tokens])
                    best_si, best_ep = max(valid, key=lambda x: (x[0] == 0, x[0], x[1]))
                    return list(self.sequences[best_si][best_ep : best_ep + max_tokens])

        if self.self_lookup and len(context) > self.min_match:
            ctx_top_k = min(self.max_match, len(context) - 1)
            for k in range(ctx_top_k, self.min_match - 1, -1):
                needle = context[-k:]
                for end_pos in range(len(context) - 1, k - 1, -1):
                    if context[end_pos - k : end_pos] == needle:
                        return list(context[end_pos : end_pos + max_tokens])
        return []


def _normalize_eos(eos_token_ids: int | list[int] | tuple[int, ...] | set[int] | None) -> set[int]:
    if eos_token_ids is None:
        return set()
    if isinstance(eos_token_ids, int):
        return {int(eos_token_ids)}
    return {int(x) for x in eos_token_ids}


def _apply_presence_penalty(logits: torch.Tensor, seen_tokens: set[int], presence_penalty: float) -> torch.Tensor:
    out = logits.detach().float().clone()
    if presence_penalty != 0.0 and seen_tokens:
        idx = torch.tensor(sorted(seen_tokens), device=out.device, dtype=torch.long)
        out.index_add_(0, idx, torch.full((idx.numel(),), -float(presence_penalty), device=out.device, dtype=out.dtype))
    return out


def _filter_and_softmax(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 for probabilistic filtering, got {temperature}")
    scaled = logits.detach().float() / float(temperature)
    vocab = scaled.shape[-1]
    if 0 < top_k < vocab:
        kth = torch.topk(scaled, int(top_k)).values[-1]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        probs_sorted = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs_sorted, dim=-1)
        remove = cumprobs > float(top_p)
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(scaled, dim=-1)
    probs = torch.clamp(probs, min=0.0)
    total = float(probs.sum().item())
    if total <= 0.0 or not torch.isfinite(probs).all():
        best = int(torch.argmax(logits).item())
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs[best] = 1.0
        return probs
    return probs / total


def _sample_speculative_token(
    p: torch.Tensor,
    draft_tok: int,
    draft_probs: dict[int, float] | None,
    rng: torch.Generator,
) -> tuple[bool, int]:
    """Exact speculative verification step preserving target marginal distribution ``p``."""
    p_cpu = p.detach().float().cpu()
    p_d = float(p_cpu[draft_tok].item())

    if draft_probs is None:
        if p_d >= 1.0 - 1e-12:
            return True, int(draft_tok)
        u = float(torch.rand((), generator=rng).item())
        if u < p_d:
            return True, int(draft_tok)
        rem = p_cpu.clone()
        rem[draft_tok] = 0.0
        rem_sum = float(rem.sum().item())
        if rem_sum <= 0.0:
            return True, int(draft_tok)
        rem /= rem_sum
        nxt = int(torch.multinomial(rem, 1, generator=rng).item())
        return False, nxt

    q_vec = torch.zeros_like(p_cpu)
    for tok_id, prob in draft_probs.items():
        ti = int(tok_id)
        if 0 <= ti < q_vec.shape[0] and prob > 0.0:
            q_vec[ti] = float(prob)
    q_sum = float(q_vec.sum().item())
    if q_sum <= 0.0:
        # Degenerate empty draft dict: fall back to exact Dirac draft verification
        return _sample_speculative_token(p_cpu, draft_tok, None, rng)
    q_vec /= q_sum

    q_d = float(q_vec[draft_tok].item())
    if q_d > 0.0:
        accept_p = min(1.0, p_d / q_d)
        u = float(torch.rand((), generator=rng).item())
        if u < accept_p:
            return True, int(draft_tok)

    res = torch.clamp(p_cpu - q_vec, min=0.0)
    res_sum = float(res.sum().item())
    if res_sum <= 1e-12:
        nxt = int(torch.multinomial(p_cpu, 1, generator=rng).item())
    else:
        nxt = int(torch.multinomial(res / res_sum, 1, generator=rng).item())
    return False, nxt


_LAYER_STATE_ATTRS = ("conv_states", "recurrent_states", "conv_state", "recurrent_state", "ssm_state")


def _cache_crop_to(cache: Any, keep_len: int) -> None:
    """Shrink a croppable cache to `keep_len` tokens. Negative arguments mean "remove that many tokens" in every
    transformers version (old: crop(max_length) also accepted -n; >= 5.4 accepts only -n)."""
    cur = int(cache.get_seq_length()) if hasattr(cache, "get_seq_length") else None
    if cur is None:
        cache.crop(keep_len)
        return
    if cur > keep_len:
        cache.crop(-(cur - keep_len))


def _is_recurrent_cache(cache: Any) -> bool:
    if cache is None or isinstance(cache, tuple):
        return False
    for attr in ("conv_states", "recurrent_states", "ssm_states"):
        val = getattr(cache, attr, None)
        if isinstance(val, (list, tuple)):
            if any(x is not None for x in val):
                return True
        elif val is not None:
            return True
    # transformers >= 5.4: Cache.layers[i] holds dicts conv_states / recurrent_states (LinearAttentionLayer, e.g.
    # Qwen3.5 Gated DeltaNet). Cache.crop cannot roll these back (it raises unless past recording is on, and even
    # then a recurrent state cannot be un-applied), so they need snapshot + restore + replay.
    for lyr in getattr(cache, "layers", None) or []:
        for attr in _LAYER_STATE_ATTRS:
            val = getattr(lyr, attr, None)
            if isinstance(val, dict) and any(v is not None for v in val.values()):
                return True
            if isinstance(val, torch.Tensor):
                return True
    if not hasattr(cache, "crop"):
        return True
    return False


def _clone_state_container(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, list):
        return [_clone_state_container(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_state_container(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _clone_state_container(v) for k, v in obj.items()}
    return obj


def _copy_state_into(dst: Any, src: Any) -> None:
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        dst.copy_(src)
    elif isinstance(dst, list) and isinstance(src, list):
        for i, item in enumerate(src):
            if isinstance(dst[i], torch.Tensor) and isinstance(item, torch.Tensor):
                dst[i].copy_(item)
            elif item is None or dst[i] is None:
                dst[i] = _clone_state_container(item)
            else:
                _copy_state_into(dst[i], item)
    elif isinstance(dst, dict) and isinstance(src, dict):
        for k, v in src.items():
            if k in dst and isinstance(dst[k], torch.Tensor) and isinstance(v, torch.Tensor):
                dst[k].copy_(v)
            else:
                dst[k] = _clone_state_container(v)


def _crop_cache(cache: Any, keep_len: int) -> Any:
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return tuple(
            tuple(t[:, :, :keep_len, ...].contiguous() if isinstance(t, torch.Tensor) and t.dim() >= 3 else t for t in layer)
            for layer in cache
        )
    if hasattr(cache, "crop"):
        _cache_crop_to(cache, keep_len)
        return cache
    # Fallback for DynamicCache-like objects without .crop()
    for attr in ("key_cache", "value_cache"):
        lst = getattr(cache, attr, None)
        if isinstance(lst, list):
            for i, t in enumerate(lst):
                if isinstance(t, torch.Tensor) and t.dim() >= 3:
                    lst[i] = t[:, :, :keep_len, :].contiguous()
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = keep_len
    return cache


_LAYER_FLAG_ATTRS = ("is_initialized", "is_conv_states_initialized", "is_recurrent_states_initialized",
                     "has_previous_state")


def _snapshot_cache(cache: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    if isinstance(cache, tuple):
        snap["tuple"] = _clone_state_container(cache)
        return snap
    for attr in ("conv_states", "recurrent_states", "ssm_states", "key_cache", "value_cache", "_seen_tokens"):
        if hasattr(cache, attr):
            snap[attr] = _clone_state_container(getattr(cache, attr))
    layers = getattr(cache, "layers", None)
    if isinstance(layers, list):
        layer_snaps = []
        for lyr in layers:
            ls: dict[str, Any] = {}
            for attr in ("keys", "values") + _LAYER_STATE_ATTRS + _LAYER_FLAG_ATTRS:
                if hasattr(lyr, attr):
                    ls[attr] = _clone_state_container(getattr(lyr, attr))
            layer_snaps.append(ls)
        snap["layers"] = layer_snaps
    return snap


def _restore_cache(cache: Any, snap: dict[str, Any], keep_len: int) -> Any:
    if "tuple" in snap:
        return _crop_cache(_clone_state_container(snap["tuple"]), keep_len)
    layered = "layers" in snap and isinstance(getattr(cache, "layers", None), list)
    layer_state = layered and any(a in ls for ls in snap["layers"] for a in _LAYER_STATE_ATTRS)
    if hasattr(cache, "crop") and not layer_state and not any(
            k in snap for k in ("conv_states", "recurrent_states", "ssm_states")):
        _cache_crop_to(cache, keep_len)
        return cache

    if hasattr(cache, "crop") and not layered:
        _cache_crop_to(cache, keep_len)
    for attr in ("conv_states", "recurrent_states", "ssm_states", "key_cache", "value_cache"):
        if attr in snap:
            cur = getattr(cache, attr)
            saved = snap[attr]
            if isinstance(cur, list) and isinstance(saved, list):
                if attr in ("key_cache", "value_cache"):
                    for i, t in enumerate(saved):
                        cur[i] = (
                            t[:, :, :keep_len, :].contiguous()
                            if isinstance(t, torch.Tensor) and t.dim() >= 3
                            else _clone_state_container(t)
                        )
                else:
                    _copy_state_into(cur, saved)
            else:
                setattr(cache, attr, _clone_state_container(saved))
    if "_seen_tokens" in snap:
        cache._seen_tokens = keep_len
    layers = getattr(cache, "layers", None)
    if "layers" in snap and isinstance(layers, list):
        for lyr, ls in zip(layers, snap["layers"]):
            for attr, val in ls.items():
                if attr in ("keys", "values") and isinstance(val, torch.Tensor) and val.dim() >= 3:
                    setattr(lyr, attr, val[..., :keep_len, :].contiguous())
                elif isinstance(val, dict) and isinstance(getattr(lyr, attr, None), dict):
                    cur = getattr(lyr, attr)
                    for k, v in val.items():  # copy in place where shapes match: keeps static (cudagraph) addresses
                        if isinstance(v, torch.Tensor) and isinstance(cur.get(k), torch.Tensor) and cur[k].shape == v.shape:
                            cur[k].copy_(v)
                        else:
                            cur[k] = _clone_state_container(v)
                else:
                    setattr(lyr, attr, _clone_state_container(val))
    return cache


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _select_token(
    logits: torch.Tensor,
    seen_tokens: set[int],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    rng: torch.Generator | None,
) -> tuple[int, torch.Tensor | None]:
    penalized = _apply_presence_penalty(logits, seen_tokens, presence_penalty)
    if temperature <= 0.0:
        return int(torch.argmax(penalized).item()), None
    probs = _filter_and_softmax(penalized, temperature, top_k, top_p)
    tok = int(torch.multinomial(probs.cpu(), 1, generator=rng).item())
    return tok, probs


@torch.no_grad()
def greedy_generate_reference(
    model: Any,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | tuple[int, ...] | set[int] | None = None,
    presence_penalty: float = 0.0,
) -> dict[str, Any]:
    """Standard 1-token-at-a-time greedy autoregressive baseline (for exactness verification)."""
    t0 = time.perf_counter()
    if max_new_tokens <= 0:
        return {
            "token_ids": [],
            "finish_reason": "length",
            "n_tokens": 0,
            "n_forward_passes": 0,
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }
    eos = _normalize_eos(eos_token_ids)
    dev = _model_device(model)
    p_ids = [int(x) for x in prompt_ids]
    inp = torch.tensor([p_ids], dtype=torch.long, device=dev)
    out = model(input_ids=inp, use_cache=True)
    n_forwards = 1
    cache = getattr(out, "past_key_values", None)
    seen: set[int] = set()
    generated: list[int] = []

    tok, _ = _select_token(
        out.logits[0, -1],
        seen,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        presence_penalty=presence_penalty,
        rng=None,
    )
    generated.append(tok)
    seen.add(tok)
    finish_reason = "stop" if tok in eos else ("length" if len(generated) >= max_new_tokens else "length")

    if tok not in eos:
        while len(generated) < max_new_tokens:
            if cache is not None:
                step_in = torch.tensor([[tok]], dtype=torch.long, device=dev)
                out = model(input_ids=step_in, past_key_values=cache, use_cache=True)
                cache = getattr(out, "past_key_values", None)
            else:
                step_in = torch.tensor([p_ids + generated], dtype=torch.long, device=dev)
                out = model(input_ids=step_in, use_cache=False)
            n_forwards += 1
            tok, _ = _select_token(
                out.logits[0, -1],
                seen,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                presence_penalty=presence_penalty,
                rng=None,
            )
            generated.append(tok)
            seen.add(tok)
            if tok in eos:
                finish_reason = "stop"
                break
        else:
            finish_reason = "length"

    return {
        "token_ids": generated,
        "finish_reason": finish_reason,
        "n_tokens": len(generated),
        "n_forward_passes": n_forwards,
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }


@torch.no_grad()
def speculative_generate_with_base(
    model: Any,
    prompt_ids: list[int],
    base_tokens: list[int],
    *,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | tuple[int, ...] | set[int] | None = None,
    block_size: int = 16,
    use_suffix_realign: bool = True,
    min_match: int = 2,
    max_match: int = 6,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    base_probs: list[dict[int, float]] | None = None,
    seed: int | None = None,
    extra_corpus: list[list[int]] | None = None,
    stop_fn: Callable[[list[int]], bool] | None = None,
) -> dict[str, Any]:
    """Generate tokens using ``base_tokens`` as a speculative draft and suffix re-alignment.

    Preserves exact greedy argmax outputs (when ``temperature <= 0``) and exact target sampling
    marginals (when ``temperature > 0``) while verifying up to ``block_size`` draft tokens per
    forward pass.
    """
    t0 = time.perf_counter()
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if max_new_tokens <= 0:
        return {
            "token_ids": [],
            "finish_reason": "length",
            "n_tokens": 0,
            "n_forward_passes": 0,
            "tokens_per_forward": 0.0,
            "prefix_matched_tokens": 0,
            "total_drafted": 0,
            "total_accepted": 0,
            "prefix_accepted": 0,
            "suffix_accepted": 0,
            "acceptance_rate": 0.0,
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }

    eos = _normalize_eos(eos_token_ids)
    dev = _model_device(model)
    p_ids = [int(x) for x in prompt_ids]
    b_toks = [int(x) for x in base_tokens]
    rng: torch.Generator | None = None
    if temperature > 0.0:
        rng = torch.Generator(device="cpu")
        if seed is not None:
            rng.manual_seed(int(seed))

    suffix_index: SuffixDraftIndex | None = None
    if use_suffix_realign:
        seqs = [b_toks]
        if p_ids:
            seqs.append(p_ids)
        if extra_corpus:
            seqs.extend(extra_corpus)
        suffix_index = SuffixDraftIndex(seqs, min_match=min_match, max_match=max_match, self_lookup=True)

    inp = torch.tensor([p_ids], dtype=torch.long, device=dev)
    out = model(input_ids=inp, use_cache=True)
    n_forwards = 1
    cache = getattr(out, "past_key_values", None)

    generated: list[int] = []
    seen_tokens: set[int] = set()
    y_0, _ = _select_token(
        out.logits[0, -1],
        seen_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        rng=rng,
    )
    generated.append(y_0)
    seen_tokens.add(y_0)

    aligned = bool(b_toks) and (y_0 == b_toks[0])
    prefix_matched = 1 if aligned else 0
    total_drafted = 0
    total_accepted = 0
    prefix_accepted = 0
    suffix_accepted = 0

    def _check_done(tok: int) -> str | None:
        if tok in eos:
            return "stop"
        if stop_fn is not None and stop_fn(generated):
            return "stop"
        if len(generated) >= max_new_tokens:
            return "length"
        return None

    finish_reason = _check_done(y_0)
    if finish_reason is None:
        pending_tok = y_0
        while len(generated) < max_new_tokens:
            rem = max_new_tokens - len(generated)
            k_want = min(block_size, rem)
            if aligned and len(generated) < len(b_toks):
                aligned_start = len(generated)
                draft = b_toks[aligned_start : aligned_start + k_want]
                draft_source = "prefix"
            elif suffix_index is not None and k_want > 0:
                aligned_start = -1
                draft = suffix_index.propose(p_ids + generated, k_want, near_pos=len(generated))
                draft_source = "suffix" if draft else "none"
            else:
                aligned_start = -1
                draft = []
                draft_source = "none"

            m = len(draft)
            total_drafted += m
            cache_len_before = len(p_ids) + len(generated) - 1
            is_recurrent = cache is not None and m > 0 and _is_recurrent_cache(cache)
            snap = _snapshot_cache(cache) if is_recurrent else None

            chunk = [pending_tok] + draft
            if cache is not None:
                chunk_t = torch.tensor([chunk], dtype=torch.long, device=dev)
                out = model(input_ids=chunk_t, past_key_values=cache, use_cache=True)
                cache = getattr(out, "past_key_values", None)
                logits_seq = out.logits[0]
            else:
                # Cache-less fallback: pass prompt + committed[:-1] + chunk
                full_t = torch.tensor([p_ids + generated[:-1] + chunk], dtype=torch.long, device=dev)
                out = model(input_ids=full_t, use_cache=False)
                logits_seq = out.logits[0, -(len(chunk)) :]
            n_forwards += 1

            accepted = 0
            bonus_tok: int | None = None
            stopped_early = False
            for j in range(m):
                l_j = _apply_presence_penalty(logits_seq[j], seen_tokens, presence_penalty)
                if temperature <= 0.0:
                    pred_j = int(torch.argmax(l_j).item())
                    ok = pred_j == draft[j]
                    nxt_j = pred_j
                else:
                    p_j = _filter_and_softmax(l_j, temperature, top_k, top_p)
                    pos = aligned_start + j
                    d_probs = (
                        base_probs[pos]
                        if (draft_source == "prefix" and base_probs is not None and 0 <= pos < len(base_probs))
                        else None
                    )
                    assert rng is not None
                    ok, nxt_j = _sample_speculative_token(p_j, draft[j], d_probs, rng)

                if ok:
                    d_tok = draft[j]
                    generated.append(d_tok)
                    seen_tokens.add(d_tok)
                    accepted += 1
                    total_accepted += 1
                    if draft_source == "prefix":
                        prefix_accepted += 1
                        prefix_matched += 1
                    elif draft_source == "suffix":
                        suffix_accepted += 1

                    done_reason = _check_done(d_tok)
                    if done_reason is not None:
                        finish_reason = done_reason
                        stopped_early = True
                        break
                else:
                    if draft_source == "prefix":
                        aligned = False
                    bonus_tok = nxt_j
                    break

            if stopped_early:
                break

            if accepted == m:
                if len(generated) >= max_new_tokens:
                    finish_reason = "length"
                    break
                bonus_tok, _ = _select_token(
                    logits_seq[m],
                    seen_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    rng=rng,
                )
                generated.append(bonus_tok)
                seen_tokens.add(bonus_tok)
                if aligned:
                    if len(generated) <= len(b_toks) and bonus_tok == b_toks[len(generated) - 1]:
                        prefix_matched += 1
                    else:
                        aligned = False
                done_reason = _check_done(bonus_tok)
                if done_reason is not None:
                    finish_reason = done_reason
                    break
                pending_tok = bonus_tok
            else:
                assert bonus_tok is not None
                generated.append(bonus_tok)
                seen_tokens.add(bonus_tok)
                done_reason = _check_done(bonus_tok)
                if done_reason is not None:
                    finish_reason = done_reason
                    break
                keep_len = cache_len_before + 1 + accepted
                if cache is not None:
                    if is_recurrent:
                        assert snap is not None
                        cache = _restore_cache(cache, snap, cache_len_before)
                        replay_ids = [pending_tok] + draft[:accepted]
                        if replay_ids:
                            rep_t = torch.tensor([replay_ids], dtype=torch.long, device=dev)
                            rep_out = model(input_ids=rep_t, past_key_values=cache, use_cache=True)
                            cache = getattr(rep_out, "past_key_values", None)
                            n_forwards += 1
                    else:
                        cache = _crop_cache(cache, keep_len)
                pending_tok = bonus_tok

    n_tok = len(generated)
    return {
        "token_ids": generated,
        "finish_reason": finish_reason or "length",
        "n_tokens": n_tok,
        "n_forward_passes": n_forwards,
        "tokens_per_forward": round(n_tok / max(n_forwards, 1), 4),
        "prefix_matched_tokens": prefix_matched,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "prefix_accepted": prefix_accepted,
        "suffix_accepted": suffix_accepted,
        "acceptance_rate": round(total_accepted / max(total_drafted, 1), 4) if total_drafted > 0 else 0.0,
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }


@torch.no_grad()
def capture_base_drafts(
    model: Any,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | tuple[int, ...] | set[int] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    save_top_k_probs: int = 0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run autoregressive generation on the base model and record tokens + optional sparse draft probs."""
    eos = _normalize_eos(eos_token_ids)
    dev = _model_device(model)
    results: list[dict[str, Any]] = []

    for idx, prompt_ids in enumerate(prompts):
        t0 = time.perf_counter()
        if max_new_tokens <= 0:
            results.append({
                "id": idx,
                "token_ids": [],
                "base_probs": [] if save_top_k_probs > 0 else None,
                "finish_reason": "length",
                "n_tokens": 0,
                "elapsed_s": round(time.perf_counter() - t0, 4),
            })
            continue

        rng: torch.Generator | None = None
        if temperature > 0.0:
            rng = torch.Generator(device="cpu")
            if seed is not None:
                rng.manual_seed(int(seed) + idx)

        p_ids = [int(x) for x in prompt_ids]
        inp = torch.tensor([p_ids], dtype=torch.long, device=dev)
        out = model(input_ids=inp, use_cache=True)
        cache = getattr(out, "past_key_values", None)
        generated: list[int] = []
        probs_list: list[dict[int, float]] | None = [] if save_top_k_probs > 0 else None
        seen: set[int] = set()

        def _record_probs(logits_vec: torch.Tensor) -> int:
            tok, probs = _select_token(
                logits_vec,
                seen,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                rng=rng,
            )
            if probs_list is not None:
                if probs is None:
                    probs_list.append({tok: 1.0})
                else:
                    k_save = min(int(save_top_k_probs), probs.shape[-1])
                    top_vals, top_idx = torch.topk(probs, k_save)
                    probs_list.append({int(ti.item()): float(tv.item()) for ti, tv in zip(top_idx, top_vals) if tv > 0.0})
            return tok

        tok = _record_probs(out.logits[0, -1])
        generated.append(tok)
        seen.add(tok)
        finish_reason = "stop" if tok in eos else "length"

        if tok not in eos:
            while len(generated) < max_new_tokens:
                if cache is not None:
                    step_in = torch.tensor([[tok]], dtype=torch.long, device=dev)
                    out = model(input_ids=step_in, past_key_values=cache, use_cache=True)
                    cache = getattr(out, "past_key_values", None)
                else:
                    step_in = torch.tensor([p_ids + generated], dtype=torch.long, device=dev)
                    out = model(input_ids=step_in, use_cache=False)
                tok = _record_probs(out.logits[0, -1])
                generated.append(tok)
                seen.add(tok)
                if tok in eos:
                    finish_reason = "stop"
                    break
            else:
                finish_reason = "length"

        results.append({
            "id": idx,
            "token_ids": generated,
            "base_probs": probs_list,
            "finish_reason": finish_reason,
            "n_tokens": len(generated),
            "elapsed_s": round(time.perf_counter() - t0, 4),
        })
    return results


def load_base_drafts_jsonl(path: str | Path) -> dict[int, list[int]]:
    """Load cached base generation outputs (e.g. ``WSI_BASE_CACHE_DIR/*.jsonl``) as ``id -> token_ids``."""
    p = Path(path)
    out: dict[int, list[int]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rid = row.get("id", len(out))
        out[int(rid)] = [int(t) for t in row.get("token_ids", [])]
    return out


def evaluate_patches_speculative(
    model: Any,
    candidates: dict[str, MoIPatch] | list[tuple[str, MoIPatch]],
    prompts: list[list[int]],
    base_drafts: list[list[int] | dict[str, Any]],
    **gen_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate multiple ``MoIPatch`` candidates using cached base drafts as speculative proposals."""
    if len(prompts) != len(base_drafts):
        raise ValueError(f"prompts ({len(prompts)}) and base_drafts ({len(base_drafts)}) must have equal length")

    items = list(candidates.items()) if isinstance(candidates, dict) else list(candidates)
    results: dict[str, dict[str, Any]] = {}

    for name, patch in items:
        t0 = time.perf_counter()
        outputs: list[dict[str, Any]] = []
        with patch.attach(model) as handle:
            handle.on()
            for idx, (prompt_ids, draft_entry) in enumerate(zip(prompts, base_drafts)):
                if isinstance(draft_entry, dict):
                    base_toks = [int(t) for t in draft_entry["token_ids"]]
                    b_probs = draft_entry.get("base_probs")
                else:
                    base_toks = [int(t) for t in draft_entry]
                    b_probs = None
                kw = dict(gen_kwargs)
                if "base_probs" not in kw and b_probs is not None:
                    kw["base_probs"] = b_probs
                if "seed" in kw and kw["seed"] is not None:
                    kw["seed"] = int(kw["seed"]) + idx
                res = speculative_generate_with_base(model, prompt_ids, base_toks, **kw)
                res["id"] = idx
                outputs.append(res)

        total_toks = sum(r["n_tokens"] for r in outputs)
        total_fwds = sum(r["n_forward_passes"] for r in outputs)
        total_drafted = sum(r["total_drafted"] for r in outputs)
        total_accepted = sum(r["total_accepted"] for r in outputs)
        n = max(len(outputs), 1)
        results[name] = {
            "outputs": outputs,
            "n_prompts": len(outputs),
            "total_tokens": total_toks,
            "total_forwards": total_fwds,
            "mean_tokens_per_forward": round(total_toks / max(total_fwds, 1), 4),
            "mean_acceptance_rate": round(total_accepted / max(total_drafted, 1), 4) if total_drafted > 0 else 0.0,
            "mean_prefix_matched": round(sum(r["prefix_matched_tokens"] for r in outputs) / n, 2),
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }
    return results
