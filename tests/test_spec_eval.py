"""CPU unit tests for Base-Draft Speculative Verification & Suffix-Index Decoding (`wsi/spec_eval.py`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi import (  # noqa: E402
    Edit,
    MoIPatch,
    SuffixDraftIndex,
    capture_base_drafts,
    evaluate_patches_speculative,
    greedy_generate_reference,
    load_base_drafts_jsonl,
    speculative_generate_with_base,
)
from wsi.moipatch import _sha256  # noqa: E402


# ------------------------------------------------------------------ tiny synthetic CPU models
class _TinyCausalBlock(nn.Module):
    """Causal attention (DynamicCache KV) + SwiGLU-style down_proj MLP."""

    def __init__(self, hidden: int = 32, inter: int = 48, n_heads: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False)

        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor, cache: DynamicCache | None, layer_idx: int) -> torch.Tensor:
        B, T, H = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(k, v, layer_idx)
        kv_len = k.shape[2]
        past_len = kv_len - T
        # Build explicit causal mask: query at past_len + i attends to 0 .. past_len + i
        q_pos = torch.arange(past_len, past_len + T, device=x.device)[:, None]
        k_pos = torch.arange(kv_len, device=x.device)[None, :]
        mask = k_pos <= q_pos
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, H)
        x = x + 0.1 * self.self_attn.o_proj(ctx)
        h_mlp = F.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        x = x + 0.1 * self.mlp.down_proj(h_mlp)
        return x


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 32, inter: int = 48, n_layers: int = 2, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_TinyCausalBlock(hidden, inter) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        cache = past_key_values if past_key_values is not None else (DynamicCache() if use_cache else None)
        x = self.model.embed_tokens(input_ids)
        for i, blk in enumerate(self.model.layers):
            x = blk(x, cache, i)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits, past_key_values=cache)


class _HybridRecurrentCache:
    """Non-croppable hybrid cache simulating Gated DeltaNet / ShortConv conv_states + recurrent_states + KV."""

    def __init__(self, n_layers: int, hidden: int) -> None:
        self.conv_states = [torch.zeros(1, hidden) for _ in range(n_layers)]
        self.recurrent_states = [torch.zeros(1, hidden) for _ in range(n_layers)]
        self.key_cache: list[torch.Tensor | None] = [None for _ in range(n_layers)]
        self.value_cache: list[torch.Tensor | None] = [None for _ in range(n_layers)]
        self._seen_tokens = 0


class _TinyHybridLM(nn.Module):
    """Toy hybrid model whose logits depend on non-croppable recurrent/conv state across steps."""

    def __init__(self, vocab: int = 32, hidden: int = 16, seed: int = 7) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.w_rec = nn.Parameter(torch.randn(hidden, hidden) * 0.2)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: _HybridRecurrentCache | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        cache = past_key_values if past_key_values is not None else (_HybridRecurrentCache(1, self.hidden) if use_cache else None)
        state = cache.recurrent_states[0].clone() if cache is not None else torch.zeros(1, self.hidden)
        conv = cache.conv_states[0].clone() if cache is not None else torch.zeros(1, self.hidden)
        emb = self.embed(input_ids)  # [1, T, H]
        outs = []
        for t in range(emb.shape[1]):
            xt = emb[:, t, :]
            conv = 0.5 * conv + xt
            state = torch.tanh(state @ self.w_rec + conv)
            outs.append(state)
        if cache is not None:
            cache.conv_states[0] = conv
            cache.recurrent_states[0] = state
            kv_step = torch.stack(outs, dim=1)[:, None, :, :]  # [1, 1, T, H]
            if cache.key_cache[0] is None:
                cache.key_cache[0] = kv_step
                cache.value_cache[0] = kv_step.clone()
            else:
                cache.key_cache[0] = torch.cat([cache.key_cache[0], kv_step], dim=2)
                cache.value_cache[0] = torch.cat([cache.value_cache[0], kv_step], dim=2)
            cache._seen_tokens += emb.shape[1]
        h = torch.stack(outs, dim=1)
        return SimpleNamespace(logits=self.lm_head(h), past_key_values=cache)


class _ConstantLogitLM(nn.Module):
    """State-independent causal LM whose logits equal a fixed target vector ``logits_vec`` at every position."""

    def __init__(self, logits_vec: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits_vec", logits_vec.float())

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[torch.Tensor] | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        B, T = input_ids.shape
        V = self.logits_vec.shape[0]
        logits = self.logits_vec.view(1, 1, V).expand(B, T, V).clone()
        return SimpleNamespace(logits=logits, past_key_values=None)


# ------------------------------------------------------------------ tests
def test_suffix_draft_index_longest_match_near_pos_and_self_lookup() -> None:
    # Primary base sequence has two occurrences of [10, 11]: at end_pos=2 (followed by 20) and end_pos=8 (followed by 30),
    # plus a longer 3-gram [10, 11, 12] at end_pos=14 (followed by 40, 41).
    base_seq = [10, 11, 20, 21, 99, 98, 10, 11, 30, 31, 97, 10, 11, 12, 40, 41]
    idx = SuffixDraftIndex([base_seq], min_match=2, max_match=4, self_lookup=True)

    # Longest match (3-gram [10, 11, 12]) wins over 2-gram
    assert idx.propose([5, 10, 11, 12], max_tokens=2) == [40, 41]

    # When only 2-gram [10, 11] matches, near_pos selects the closer occurrence in primary seq_idx=0
    assert idx.propose([1, 10, 11], max_tokens=2, near_pos=2) == [20, 21]
    assert idx.propose([1, 10, 11], max_tokens=2, near_pos=8) == [30, 31]

    # When corpus misses, self_lookup falls back to earlier occurrences in the context itself
    ctx = [50, 51, 52, 53, 54, 50, 51]
    assert idx.propose(ctx, max_tokens=3) == [52, 53, 54]
    assert idx.propose([61, 62], max_tokens=3) == []


def test_greedy_identical_model_exact_and_fast() -> None:
    model = _TinyCausalLM(vocab=64, hidden=32, inter=48, n_layers=2, seed=11).eval()
    prompt = [3, 7, 15, 22, 41]
    max_new = 40

    base = greedy_generate_reference(model, prompt, max_new_tokens=max_new, eos_token_ids=63)
    spec = speculative_generate_with_base(
        model,
        prompt,
        base["token_ids"],
        max_new_tokens=max_new,
        eos_token_ids=63,
        block_size=8,
        use_suffix_realign=True,
    )

    assert spec["token_ids"] == base["token_ids"]
    assert spec["finish_reason"] == base["finish_reason"]
    assert spec["prefix_matched_tokens"] == len(spec["token_ids"])
    assert spec["tokens_per_forward"] >= 5.0
    assert spec["n_forward_passes"] < base["n_forward_passes"]


def test_greedy_perturbed_model_with_presence_penalty_bit_exact() -> None:
    base_model = _TinyCausalLM(vocab=64, hidden=32, inter=48, n_layers=2, seed=3).eval()
    prompt = [2, 9, 14, 28, 35, 4]
    max_new = 36

    base_out = greedy_generate_reference(
        base_model, prompt, max_new_tokens=max_new, eos_token_ids=63, presence_penalty=1.5
    )

    u = torch.randn(32)
    u = u / u.norm()
    w_down = base_model.model.layers[1].mlp.down_proj.weight.detach()
    patch = MoIPatch(
        edits=[
            Edit(
                module="layers.1.mlp.down_proj",
                u=u,
                alpha=3.5,
                idx=torch.arange(0, 12, dtype=torch.long),
                base_sha256=_sha256(w_down),
                base_dtype="float32",
                shape=tuple(w_down.shape),
            )
        ]
    )

    with patch.attach(base_model) as h:
        h.on()
        ref = greedy_generate_reference(
            base_model, prompt, max_new_tokens=max_new, eos_token_ids=63, presence_penalty=1.5
        )
        spec_no_realign = speculative_generate_with_base(
            base_model,
            prompt,
            base_out["token_ids"],
            max_new_tokens=max_new,
            eos_token_ids=63,
            block_size=8,
            use_suffix_realign=False,
            presence_penalty=1.5,
        )
        spec_realign = speculative_generate_with_base(
            base_model,
            prompt,
            base_out["token_ids"],
            max_new_tokens=max_new,
            eos_token_ids=63,
            block_size=8,
            use_suffix_realign=True,
            presence_penalty=1.5,
        )

    assert spec_no_realign["token_ids"] == ref["token_ids"]
    assert spec_realign["token_ids"] == ref["token_ids"]
    assert spec_no_realign["finish_reason"] == ref["finish_reason"]
    assert spec_realign["finish_reason"] == ref["finish_reason"]


def test_suffix_realign_recovers_after_one_token_shift() -> None:
    model = _TinyCausalLM(vocab=64, hidden=32, inter=48, n_layers=2, seed=5).eval()
    prompt = [1, 8, 19, 25]
    max_new = 48

    ref = greedy_generate_reference(model, prompt, max_new_tokens=max_new, eos_token_ids=63)
    assert len(ref["token_ids"]) >= 20

    # Insert one extra token early in the base draft at index 2 so pure prefix matching diverges permanently,
    # while suffix indexing re-aligns immediately to the shifted tail!
    wrong_tok = (ref["token_ids"][2] + 1) % 60
    shifted_base = ref["token_ids"][:2] + [wrong_tok] + ref["token_ids"][2:]

    spec_no = speculative_generate_with_base(
        model,
        prompt,
        shifted_base,
        max_new_tokens=max_new,
        eos_token_ids=63,
        block_size=8,
        use_suffix_realign=False,
    )
    spec_yes = speculative_generate_with_base(
        model,
        prompt,
        shifted_base,
        max_new_tokens=max_new,
        eos_token_ids=63,
        block_size=8,
        use_suffix_realign=True,
    )

    assert spec_no["token_ids"] == ref["token_ids"]
    assert spec_yes["token_ids"] == ref["token_ids"]
    assert spec_yes["suffix_accepted"] > 0
    assert spec_yes["total_accepted"] > spec_no["total_accepted"] + 10
    assert spec_yes["n_forward_passes"] < spec_no["n_forward_passes"]


def _empirical_probs_constant_model(
    target_logits: torch.Tensor,
    *,
    draft_logits: torch.Tensor | None,
    use_draft_probs: bool,
    n_runs: int = 2000,
    tokens_per_run: int = 10,
) -> tuple[torch.Tensor, float]:
    model = _ConstantLogitLM(target_logits)
    target_p = torch.softmax(target_logits.float(), dim=-1)
    q_p = torch.softmax(draft_logits.float(), dim=-1) if draft_logits is not None else target_p
    q_dict = {i: float(q_p[i].item()) for i in range(q_p.shape[0])}

    rng_d = torch.Generator(device="cpu").manual_seed(1234)
    counts = torch.zeros_like(target_p)
    tot_drafted = 0
    tot_accepted = 0
    for r in range(n_runs):
        draft_toks = torch.multinomial(q_p, tokens_per_run, replacement=True, generator=rng_d).tolist()
        res = speculative_generate_with_base(
            model,
            [0],
            draft_toks,
            max_new_tokens=tokens_per_run,
            block_size=8,
            use_suffix_realign=False,
            temperature=1.0,
            base_probs=([q_dict] * tokens_per_run) if use_draft_probs else None,
            seed=5000 + r,
        )
        assert len(res["token_ids"]) == tokens_per_run
        for t in res["token_ids"]:
            counts[t] += 1.0
        tot_drafted += res["total_drafted"]
        tot_accepted += res["total_accepted"]
    return counts / counts.sum(), tot_accepted / max(tot_drafted, 1)


def test_speculative_sampling_unbiased_dirac_draft() -> None:
    target_logits = torch.tensor([1.4, 0.9, 0.2, -0.4, -1.0, -1.6])
    target_p = torch.softmax(target_logits, dim=-1)
    emp_p, _ = _empirical_probs_constant_model(target_logits, draft_logits=target_logits, use_draft_probs=False)
    max_err = float((emp_p - target_p).abs().max().item())
    assert max_err < 0.015, f"Dirac speculative sampling drifted from target p: max_err={max_err}"


def test_speculative_sampling_unbiased_with_base_probs() -> None:
    target_logits = torch.tensor([1.5, 0.8, 0.1, -0.5, -1.2, -1.8])
    draft_logits = torch.tensor([1.2, 1.0, 0.2, -0.3, -1.0, -1.5])
    target_p = torch.softmax(target_logits, dim=-1)

    _, acc_dirac = _empirical_probs_constant_model(target_logits, draft_logits=draft_logits, use_draft_probs=False)
    emp_p, acc_probs = _empirical_probs_constant_model(target_logits, draft_logits=draft_logits, use_draft_probs=True)
    max_err = float((emp_p - target_p).abs().max().item())
    assert max_err < 0.015, f"base_probs speculative sampling drifted from target p: max_err={max_err}"
    assert acc_probs > acc_dirac + 0.15


def test_hybrid_recurrent_cache_rollback_bit_exact() -> None:
    model = _TinyHybridLM(vocab=32, hidden=16, seed=19).eval()
    prompt = [1, 4, 9, 12]
    max_new = 30

    ref = greedy_generate_reference(model, prompt, max_new_tokens=max_new, eos_token_ids=None, presence_penalty=1.0)
    assert len(ref["token_ids"]) >= 16

    # Craft a base draft that matches the first 3 tokens, rejects mid-block at token 3, then matches again later
    bad_base = list(ref["token_ids"])
    bad_base[3] = (bad_base[3] + 1) % 30
    bad_base[10] = (bad_base[10] + 2) % 30

    spec = speculative_generate_with_base(
        model,
        prompt,
        bad_base,
        max_new_tokens=max_new,
        eos_token_ids=None,
        block_size=6,
        use_suffix_realign=True,
        presence_penalty=1.0,
    )
    assert spec["token_ids"] == ref["token_ids"]
    assert spec["finish_reason"] == ref["finish_reason"]


def test_evaluate_patches_speculative_applies_and_restores_weights() -> None:
    model = _TinyCausalLM(vocab=64, hidden=32, inter=48, n_layers=2, seed=21).eval()
    prompts = [[2, 5, 11], [4, 17, 29, 31]]
    drafts = capture_base_drafts(model, prompts, max_new_tokens=24, eos_token_ids=63, temperature=0.0)

    w1 = model.model.layers[1].mlp.down_proj.weight.detach()
    orig_sha = _sha256(w1)
    u = torch.randn(32)
    u = u / u.norm()
    candidates = {
        "patch_a": MoIPatch(
            edits=[Edit(module="layers.1.mlp.down_proj", u=u, alpha=2.0, idx=torch.arange(8), base_sha256=orig_sha, base_dtype="float32", shape=tuple(w1.shape))]
        ),
        "patch_b": MoIPatch(
            edits=[Edit(module="layers.1.mlp.down_proj", u=u, alpha=5.0, idx=torch.arange(8), base_sha256=orig_sha, base_dtype="float32", shape=tuple(w1.shape))]
        ),
    }

    res = evaluate_patches_speculative(model, candidates, prompts, drafts, max_new_tokens=24, eos_token_ids=63, block_size=6)
    assert _sha256(model.model.layers[1].mlp.down_proj.weight) == orig_sha
    assert set(res) == {"patch_a", "patch_b"}
    for name in ("patch_a", "patch_b"):
        entry = res[name]
        assert entry["n_prompts"] == 2
        assert len(entry["outputs"]) == 2
        assert entry["total_tokens"] > 0
        assert entry["total_forwards"] > 0
        assert "mean_tokens_per_forward" in entry
        assert "mean_acceptance_rate" in entry
        assert "mean_prefix_matched" in entry
        assert "elapsed_s" in entry

    # Verify outputs match independent greedy references under each patch
    for name, patch in candidates.items():
        with patch.attach(model) as h:
            h.on()
            for p, out in zip(prompts, res[name]["outputs"]):
                ref = greedy_generate_reference(model, p, max_new_tokens=24, eos_token_ids=63)
                assert out["token_ids"] == ref["token_ids"]


def test_load_base_drafts_jsonl_and_edge_cases(tmp_path: Path) -> None:
    jp = tmp_path / "outputs_dev_base.jsonl"
    jp.write_text(
        json.dumps({"id": 10, "token_ids": [1, 2, 3], "finish_reason": "stop"}) + "\n"
        + json.dumps({"id": 25, "token_ids": [4, 5], "finish_reason": "length"}) + "\n",
        encoding="utf-8",
    )
    loaded = load_base_drafts_jsonl(jp)
    assert loaded == {10: [1, 2, 3], 25: [4, 5]}

    model = _TinyCausalLM(vocab=64, hidden=32, inter=48, n_layers=2, seed=1).eval()
    prompt = [3, 8, 12]

    out0 = speculative_generate_with_base(model, prompt, [], max_new_tokens=0)
    assert out0["token_ids"] == [] and out0["n_forward_passes"] == 0

    out1 = speculative_generate_with_base(model, prompt, [], max_new_tokens=1)
    ref1 = greedy_generate_reference(model, prompt, max_new_tokens=1)
    assert out1["token_ids"] == ref1["token_ids"] and out1["n_tokens"] == 1

    out_empty_draft = speculative_generate_with_base(model, prompt, [], max_new_tokens=10, block_size=4)
    ref10 = greedy_generate_reference(model, prompt, max_new_tokens=10)
    assert out_empty_draft["token_ids"] == ref10["token_ids"]
