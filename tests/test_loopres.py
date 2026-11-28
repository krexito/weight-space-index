"""CPU unit tests for `wsi.loopres` (LoopRes, KV cache isolation, hybrid cache, checkpoints & persistence)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_pocket_moe_ceiling import _enable_emb_grad_hook  # noqa: E402
from scripts.experiment_pocket_moe_regularized import _RegLoRALinear  # noqa: E402
from wsi import FORMAT_LOOPRES, LoopResConfig, LoopResModule, default_core_span  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402


# ------------------------------------------------------------------ Toy DynamicCache & Causal Attention LM
class _ToyDynamicCache:
    """Minimal HF DynamicCache-compatible KV container for CPU testing."""

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        k = self.key_cache[layer_idx]
        return int(k.shape[-2]) if k is not None and k.numel() > 0 else 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.empty(0, dtype=value_states.dtype, device=value_states.device))
        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def _apply_toy_rope(x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
    """Deterministic position-dependent rotation over pairs of head_dim coordinates."""
    # x: [B, H, T, Dh], pos_ids: [B, T]
    dh = int(x.shape[-1])
    half = dh // 2
    if half == 0:
        return x
    inv_freq = torch.pow(10000.0, -torch.arange(0, half, device=x.device, dtype=torch.float32) / float(half))
    angles = pos_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, half)  # [B, T, half]
    cos = angles.cos().unsqueeze(1).to(x.dtype)
    sin = angles.sin().unsqueeze(1).to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if 2 * half < dh:
        rot = torch.cat([rot, x[..., 2 * half :]], dim=-1)
    return rot


class _ToyAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int, layer_idx: int) -> None:
        super().__init__()
        assert hidden % n_heads == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if position_ids is not None:
            q = _apply_toy_rope(q, position_ids)
            k = _apply_toy_rope(k, position_ids)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        T_kv = int(k.shape[-2])
        T_past = T_kv - T
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))

        # Causal + left-padding mask over full [B, 1, T, T_kv]
        q_pos = torch.arange(T, device=hidden_states.device) + T_past
        k_pos = torch.arange(T_kv, device=hidden_states.device)
        causal = k_pos.view(1, T_kv) <= q_pos.view(T, 1)  # [T, T_kv]
        mask = causal.view(1, 1, T, T_kv).expand(B, 1, T, T_kv)
        if attention_mask is not None:
            k_valid = (attention_mask[:, :T_kv] > 0).view(B, 1, 1, T_kv)
            mask = mask & k_valid

        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.hidden)
        return self.o_proj(out)


class _ToySwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _ToyCachedBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, intermediate: int, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(hidden)
        self.self_attn = _ToyAttention(hidden, n_heads, layer_idx)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = _ToySwiGLU(hidden, intermediate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        **_unused: Any,
    ) -> torch.Tensor:
        h = hidden_states + 0.35 * self.self_attn(
            self.ln1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        h = h + 0.35 * self.mlp(self.ln2(h))
        return h


class _ToyCachedTrunk(nn.Module):
    def __init__(self, vocab: int, hidden: int, n_layers: int, n_heads: int, intermediate: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [_ToyCachedBlock(hidden, n_heads, intermediate, i) for i in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = _ToyDynamicCache()

        if attention_mask is None:
            past_len = past_key_values.get_seq_length(0) if past_key_values is not None else 0
            full_mask = torch.ones((B, past_len + T), dtype=torch.long, device=input_ids.device)
        else:
            full_mask = attention_mask

        full_pos = (full_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
        pos_ids = full_pos[:, -T:]

        hs = [x] if output_hidden_states else None
        for blk in self.layers:
            x = blk(
                x,
                attention_mask=full_mask,
                position_ids=pos_ids,
                past_key_values=past_key_values if use_cache else None,
            )
            if hs is not None:
                hs.append(x)

        x = self.norm(x)
        return SimpleNamespace(
            last_hidden_state=x,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(hs) if hs is not None else None,
        )


class _ToyCachedTransformerLM(nn.Module):
    """Causal multi-head attention + RoPE + SwiGLU toy LM supporting KV cache and left-padded batch generate."""

    def __init__(
        self,
        vocab: int = 40,
        hidden: int = 24,
        n_layers: int = 6,
        n_heads: int = 4,
        intermediate: int = 48,
        seed: int = 13,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(
            model_type="toy_qwen3",
            hidden_size=hidden,
            vocab_size=vocab,
            num_hidden_layers=n_layers,
        )
        self.model = _ToyCachedTrunk(vocab, hidden, n_layers, n_heads, intermediate)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        logits_to_keep: int | None = None,
    ) -> SimpleNamespace:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        h = out.last_hidden_state
        if logits_to_keep is not None and logits_to_keep > 0:
            h = h[:, -logits_to_keep:]
        return SimpleNamespace(
            logits=self.lm_head(h),
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 8,
        do_sample: bool = True,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: Any = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        eos_set: set[int] = set()
        if isinstance(eos_token_id, int):
            eos_set.add(eos_token_id)
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_set.update(int(x) for x in eos_token_id)

        cur_ids = input_ids.clone()
        cur_mask = attention_mask.clone()
        pkv = _ToyDynamicCache() if use_cache else None
        finished = torch.zeros(cur_ids.shape[0], dtype=torch.bool, device=cur_ids.device)

        for step in range(int(max_new_tokens)):
            step_ids = cur_ids if (not use_cache or step == 0) else cur_ids[:, -1:]
            out = self.forward(
                step_ids,
                attention_mask=cur_mask,
                past_key_values=pkv,
                use_cache=use_cache,
            )
            pkv = out.past_key_values
            logits = out.logits[:, -1].float()
            if do_sample and temperature > 0.0:
                scaled = logits / float(temperature)
                if top_k > 0 and top_k < scaled.shape[-1]:
                    kth = torch.topk(scaled, int(top_k), dim=-1).values[:, -1:]
                    scaled = scaled.masked_fill(scaled < kth, -1e9)
                probs = F.softmax(scaled, dim=-1)
                if top_p < 1.0:
                    sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
                    cum_p = sorted_p.cumsum(dim=-1)
                    drop = (cum_p - sorted_p) >= float(top_p)
                    sorted_p = sorted_p.masked_fill(drop, 0.0)
                    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    sampled_idx = torch.multinomial(sorted_p, num_samples=1)
                    nxt = sorted_i.gather(-1, sampled_idx)
                else:
                    nxt = torch.multinomial(probs, num_samples=1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)

            if eos_set:
                nxt = torch.where(finished.unsqueeze(-1), torch.full_like(nxt, int(pad_token_id)), nxt)
                for eid in eos_set:
                    finished = finished | (nxt.squeeze(-1) == eid)

            cur_ids = torch.cat([cur_ids, nxt], dim=1)
            cur_mask = torch.cat([cur_mask, (~finished).long().unsqueeze(-1)], dim=1)
            if eos_set and bool(finished.all()):
                break
        return cur_ids


# ------------------------------------------------------------------ Toy Hybrid DeltaNet-Style Cache & LM
class _ToyHybridCache:
    """Mimics `Qwen3_5DynamicCache` with `last_linear_layer` + `has_previous_state`."""

    def __init__(self, config: Any) -> None:
        self.layer_types = list(config.layer_types)
        self.key_cache: list[torch.Tensor | None] = [None] * len(self.layer_types)
        self.value_cache: list[torch.Tensor | None] = [None] * len(self.layer_types)
        self.recurrent_states: list[torch.Tensor | None] = [None] * len(self.layer_types)
        lin_layers = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        self.last_linear_layer = lin_layers[-1] if lin_layers else 0

    @property
    def has_previous_state(self) -> bool:
        return self.recurrent_states[self.last_linear_layer] is not None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Check full-attention layers if linear layer was requested
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return int(self.key_cache[layer_idx].shape[-2])
        for k in self.key_cache:
            if k is not None:
                return int(k.shape[-2])
        return 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class _ToyLinearRecurrentBlock(nn.Module):
    def __init__(self, hidden: int, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.decay = nn.Parameter(torch.full((hidden,), 0.7))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        **_unused: Any,
    ) -> torch.Tensor:
        u = torch.tanh(self.in_proj(hidden_states))
        B, T, D = u.shape
        if past_key_values is not None and past_key_values.has_previous_state:
            s = past_key_values.recurrent_states[self.layer_idx]
            assert s is not None
        else:
            s = torch.zeros((B, D), dtype=u.dtype, device=u.device)

        ys = []
        d_sig = torch.sigmoid(self.decay).view(1, D)
        for t in range(T):
            s = d_sig * s + u[:, t]
            ys.append(s)
        if past_key_values is not None:
            past_key_values.recurrent_states[self.layer_idx] = s
        y = torch.stack(ys, dim=1)
        return hidden_states + 0.25 * self.out_proj(y)


class _ToyHybridDeltaNetLM(nn.Module):
    """6-layer hybrid model (`[linear, full, linear, full, linear, full]`) with `last_linear_layer=4` in Coda."""

    def __init__(self, vocab: int = 32, hidden: int = 16, seed: int = 19) -> None:
        super().__init__()
        torch.manual_seed(seed)
        layer_types = ["linear_attention", "full_attention", "linear_attention", "full_attention", "linear_attention", "full_attention"]
        self.config = SimpleNamespace(
            model_type="toy_qwen3_5",
            text_config=SimpleNamespace(
                hidden_size=hidden,
                num_hidden_layers=len(layer_types),
                layer_types=layer_types,
            ),
            hidden_size=hidden,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        blocks: list[nn.Module] = []
        for i, lt in enumerate(layer_types):
            if lt == "linear_attention":
                blocks.append(_ToyLinearRecurrentBlock(hidden, i))
            else:
                blocks.append(_ToyCachedBlock(hidden, n_heads=2, intermediate=32, layer_idx=i))
        self.model.layers = nn.ModuleList(blocks)
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        B, T = input_ids.shape
        x = self.model.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = _ToyHybridCache(self.config.text_config)
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        pos_ids = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0).expand(B, T)
        for blk in self.model.layers:
            x = blk(
                x,
                attention_mask=None,
                position_ids=pos_ids,
                past_key_values=past_key_values if use_cache else None,
            )
        x = self.model.norm(x)
        return SimpleNamespace(logits=self.lm_head(x), past_key_values=past_key_values if use_cache else None)


# ------------------------------------------------------------------ Unit Tests
def test_default_core_span_bounds() -> None:
    assert default_core_span(28) == (6, 22)
    assert default_core_span(24) == (6, 18)
    assert default_core_span(6) == (1, 5)
    assert default_core_span(2) == (0, 2)


def test_n_loops_1_is_bit_exact_passthrough() -> None:
    model = _ToyCachedTransformerLM(n_layers=6, seed=3).eval()
    cfg = LoopResConfig(d_model=24, core_start=1, core_end=5, n_loops=2, window=2)
    lr_mod = LoopResModule(cfg)
    ids = torch.tensor([[1, 5, 9, 12, 17, 23]], dtype=torch.long)

    with torch.no_grad():
        base_logits = model(ids).logits
        with lr_mod.attach(model, n_loops=1):
            loop1_logits = model(ids).logits

    assert torch.equal(base_logits, loop1_logits)


def test_step0_identity_preservation_and_non_zero_gradients() -> None:
    model = _ToyCachedTransformerLM(n_layers=6, seed=5).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ids = torch.tensor([[2, 6, 11, 18, 22, 27]], dtype=torch.long)
    with torch.no_grad():
        base_logits = model(ids).logits

    # Near-identity check when init_pass0_bias is very large
    cfg_exact = LoopResConfig(d_model=24, core_start=1, core_end=5, n_loops=2, window=2, init_pass0_bias=100.0)
    mod_exact = LoopResModule(cfg_exact)
    with torch.no_grad(), mod_exact.attach(model, record_stats=True):
        out_exact = model(ids).logits
    assert torch.allclose(out_exact, base_logits, atol=1e-5)

    # Default step-0 anchor softmax([3.0, 0.0])[0] ~= 0.952574
    cfg = LoopResConfig(d_model=24, core_start=1, core_end=5, n_loops=2, window=2, d_loop_key=16)
    mod = LoopResModule(cfg)
    mod.reset_telemetry()
    with torch.no_grad(), mod.attach(model, record_stats=True):
        _ = model(ids)
    tel = mod.telemetry_summary()
    expected_p0 = 1.0 / (1.0 + math.exp(-3.0))
    assert abs(tel["coda_alpha_mean"][0] - expected_p0) < 1e-4, tel["coda_alpha_mean"]

    # Run 2 AdamW steps -> every parameter must get finite non-zero gradients across steps 0..1
    emb_hooks = _enable_emb_grad_hook(model)
    opt = torch.optim.AdamW(mod.parameters(), lr=1e-2)
    seen_nonzero_grad: set[str] = set()
    try:
        for _step in range(2):
            opt.zero_grad(set_to_none=True)
            with mod.attach(model):
                logits = model(ids).logits[:, :-1]
                targets = ids[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                loss.backward()
            for nm, p in mod.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), nm
                if float(p.grad.abs().max().item()) > 0.0:
                    seen_nonzero_grad.add(nm)
            opt.step()
    finally:
        for h in emb_hooks:
            h.remove()

    expected_names = {
        "layer_q", "layer_q_loop", "layer_k_loop", "layer_bias",
        "coda_q", "coda_q_loop", "coda_k_loop", "coda_bias",
    }
    assert seen_nonzero_grad == expected_names, (expected_names - seen_nonzero_grad)


def test_grad_checkpointing_exact_gradient_parity() -> None:
    ids = torch.tensor([[1, 4, 8, 12, 16, 20]], dtype=torch.long)

    def _run_grad_check(*, with_lora: bool, grad_ckpt: bool) -> dict[str, torch.Tensor]:
        model = _ToyCachedTransformerLM(n_layers=6, seed=11).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        wrapped = []
        if with_lora:
            _, blocks = find_blocks(model)
            for blk in blocks:
                for sub_nm, lin in [(nm, m) for nm, m in blk.named_modules() if isinstance(m, nn.Linear)]:
                    parts = sub_nm.split(".")
                    parent = blk
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    w = _RegLoRALinear(lin, rank=2, alpha_scale=4.0, dropout=0.0)
                    with torch.no_grad():
                        w.lora_B.fill_(0.05)
                    setattr(parent, parts[-1], w)
                    wrapped.append((f"b{blk.layer_idx}.{sub_nm}", w))

        torch.manual_seed(99)
        cfg = LoopResConfig(d_model=24, core_start=1, core_end=5, n_loops=2, window=2, d_loop_key=16)
        lr_mod = LoopResModule(cfg)
        emb_hooks = _enable_emb_grad_hook(model)
        try:
            with lr_mod.attach(model, grad_checkpointing=grad_ckpt):
                logits = model(ids).logits[:, :-1]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
                loss.backward()
        finally:
            for h in emb_hooks:
                h.remove()

        grads = {f"lr.{nm}": p.grad.detach().clone() for nm, p in lr_mod.named_parameters() if p.grad is not None}
        for w_nm, w in wrapped:
            if w.lora_A.grad is not None:
                grads[f"{w_nm}.A"] = w.lora_A.grad.detach().clone()
            if w.lora_B.grad is not None:
                grads[f"{w_nm}.B"] = w.lora_B.grad.detach().clone()
        return grads

    for use_lora in (False, True):
        g_off = _run_grad_check(with_lora=use_lora, grad_ckpt=False)
        g_on = _run_grad_check(with_lora=use_lora, grad_ckpt=True)
        assert set(g_off) == set(g_on) and len(g_off) >= 8
        for k in g_off:
            assert torch.allclose(g_off[k], g_on[k], atol=1e-4, rtol=1e-4), (use_lora, k, (g_off[k] - g_on[k]).abs().max())


def test_kv_cache_exact_equivalence_and_batched_left_pad() -> None:
    model = _ToyCachedTransformerLM(n_layers=6, seed=21).eval()
    torch.manual_seed(33)
    cfg = LoopResConfig(d_model=24, core_start=1, core_end=5, n_loops=2, window=2, d_loop_key=16)
    lr_mod = LoopResModule(cfg)
    with torch.no_grad():
        lr_mod.layer_q.normal_(0.0, 0.25)
        lr_mod.layer_q_loop.normal_(0.0, 0.25)
        lr_mod.coda_q.normal_(0.0, 0.25)
        lr_mod.coda_q_loop.normal_(0.0, 0.25)

    seq_a = torch.tensor([[3, 7, 11, 15, 19, 23, 27]], dtype=torch.long)
    seq_b = torch.tensor([[5, 9, 13, 17, 21]], dtype=torch.long)

    # Verify both trained n_loops=2 and zero-shot n_loops=3 extrapolation
    for active_loops in (2, 3):
        with torch.no_grad(), lr_mod.attach(model, n_loops=active_loops):
            full_logits = model(seq_a, use_cache=False).logits

            # Prefill first 4 tokens, then decode 1 token at a time
            pkv = _ToyDynamicCache()
            step_logits = []
            out0 = model(seq_a[:, :4], past_key_values=pkv, use_cache=True)
            step_logits.append(out0.logits)
            pkv = out0.past_key_values
            for t in range(4, seq_a.shape[1]):
                out_t = model(seq_a[:, t : t + 1], past_key_values=pkv, use_cache=True)
                pkv = out_t.past_key_values
                step_logits.append(out_t.logits[:, -1:])
            cached_logits = torch.cat(step_logits, dim=1)
            assert torch.allclose(full_logits, cached_logits, atol=1e-5), (
                active_loops,
                (full_logits - cached_logits).abs().max(),
            )

            # Second independent fresh cache must have zero stale cross-talk
            full_b = model(seq_b, use_cache=False).logits
            pkv_b = _ToyDynamicCache()
            o_b0 = model(seq_b[:, :3], past_key_values=pkv_b, use_cache=True)
            o_b1 = model(seq_b[:, 3:4], past_key_values=o_b0.past_key_values, use_cache=True)
            o_b2 = model(seq_b[:, 4:5], past_key_values=o_b1.past_key_values, use_cache=True)
            cached_b = torch.cat([o_b0.logits, o_b1.logits, o_b2.logits], dim=1)
            assert torch.allclose(full_b, cached_b, atol=1e-5)

    # Batched left-padded equivalence (batch_size=3, differing prompt lengths)
    prompts_pad = torch.tensor(
        [
            [0, 0, 0, 4, 8, 12],
            [0, 5, 9, 13, 17, 21],
            [6, 10, 14, 18, 22, 26],
        ],
        dtype=torch.long,
    )
    mask_pad = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    next_tok = torch.tensor([[15], [25], [35]], dtype=torch.long)
    full_ids_b = torch.cat([prompts_pad, next_tok], dim=1)
    full_mask_b = torch.cat([mask_pad, torch.ones((3, 1), dtype=torch.long)], dim=1)

    with torch.no_grad(), lr_mod.attach(model, n_loops=2):
        un_logits = model(full_ids_b, attention_mask=full_mask_b, use_cache=False).logits[:, -1:]
        pkv_pad = _ToyDynamicCache()
        o_p = model(prompts_pad, attention_mask=mask_pad, past_key_values=pkv_pad, use_cache=True)
        o_d = model(next_tok, attention_mask=full_mask_b, past_key_values=o_p.past_key_values, use_cache=True)
        assert torch.allclose(un_logits, o_d.logits, atol=1e-5), (un_logits - o_d.logits).abs().max()


def test_hybrid_deltanet_style_cache_equivalence() -> None:
    model = _ToyHybridDeltaNetLM(seed=41).eval()
    cfg = LoopResConfig(d_model=16, core_start=1, core_end=4, n_loops=2, window=2, d_loop_key=8)
    lr_mod = LoopResModule(cfg)
    with torch.no_grad():
        lr_mod.layer_q.normal_(0.0, 0.2)
        lr_mod.coda_q.normal_(0.0, 0.2)

    seq = torch.tensor([[2, 5, 9, 14, 18, 22]], dtype=torch.long)
    with torch.no_grad(), lr_mod.attach(model, n_loops=2):
        full_logits = model(seq, use_cache=False).logits
        pkv = _ToyHybridCache(model.config.text_config)
        o0 = model(seq[:, :3], past_key_values=pkv, use_cache=True)
        pass1_cache = o0.past_key_values._wsi_loopres_pass_caches[1]
        assert pass1_cache.last_linear_layer == 2
        assert pass1_cache.has_previous_state
        steps = [o0.logits]
        cur_pkv = o0.past_key_values
        for t in range(3, seq.shape[1]):
            ot = model(seq[:, t : t + 1], past_key_values=cur_pkv, use_cache=True)
            cur_pkv = ot.past_key_values
            steps.append(ot.logits)
        cached_logits = torch.cat(steps, dim=1)
    assert torch.allclose(full_logits, cached_logits, atol=1e-5), (full_logits - cached_logits).abs().max()


def test_gated_mode_ablation_and_3x_extrapolation() -> None:
    model = _ToyCachedTransformerLM(n_layers=6, seed=7).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = torch.tensor([[1, 3, 7, 11, 15, 19]], dtype=torch.long)

    cfg_g = LoopResConfig(d_model=24, core_start=1, core_end=5, n_loops=2, window=2, mode="gated")
    mod_g = LoopResModule(cfg_g)
    assert set(dict(mod_g.named_parameters())) == {"layer_bias", "coda_bias"}

    emb_hooks = _enable_emb_grad_hook(model)
    opt = torch.optim.AdamW(mod_g.parameters(), lr=5e-2)
    try:
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with mod_g.attach(model):
                logits = model(ids).logits[:, :-1]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
                loss.backward()
            assert mod_g.layer_bias.grad is not None and mod_g.coda_bias.grad is not None
            opt.step()
    finally:
        for h in emb_hooks:
            h.remove()

    cfg_l = LoopResConfig(d_model=24, core_start=1, core_end=5, n_loops=2, window=2, d_loop_key=16, mode="loopres")
    mod_l = LoopResModule(cfg_l)
    for mod in (mod_g, mod_l):
        mod.reset_telemetry()
        with torch.no_grad(), mod.attach(model, n_loops=3, record_stats=True):
            _ = model(ids)
        tel = mod.telemetry_summary()
        assert len(tel["coda_alpha_mean"]) == 3, tel
        assert abs(sum(tel["coda_alpha_mean"]) - 1.0) < 1e-5, tel["coda_alpha_mean"]
        assert tel["n_tokens"] == int(ids.numel())


def test_save_load_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    model = _ToyCachedTransformerLM(n_layers=6, seed=29).eval()
    ids = torch.tensor([[2, 4, 6, 8, 10]], dtype=torch.long)
    cfg = LoopResConfig(d_model=24, core_start=1, core_end=5, n_loops=2, window=2, d_loop_key=16)
    mod = LoopResModule(cfg)
    with torch.no_grad():
        mod.layer_q.normal_(0.0, 0.1)
        mod.coda_q.normal_(0.0, 0.1)

    p = mod.save(tmp_path / "expert.loopres.safetensors")
    loaded = LoopResModule.load(p, verify=True)
    assert loaded.config == mod.config
    for (k1, v1), (k2, v2) in zip(sorted(mod.state_dict().items()), sorted(loaded.state_dict().items())):
        assert k1 == k2 and torch.equal(v1, v2)

    with torch.no_grad(), mod.attach(model):
        y1 = model(ids).logits
    with torch.no_grad(), loaded.attach(model):
        y2 = model(ids).logits
    assert torch.equal(y1, y2)

    # Tamper tensor while retaining old SHA-256 metadata
    tampered_p = tmp_path / "tampered_tensor.safetensors"
    with safe_open(str(p), "pt") as fh:
        meta = dict(fh.metadata())
        tens = {k: fh.get_tensor(k).clone() for k in fh.keys()}
    tens["coda_bias"][0] += 0.5
    save_file(tens, str(tampered_p), metadata=meta)
    with pytest.raises(ValueError, match="SHA-256"):
        LoopResModule.load(tampered_p, verify=True)

    # Tamper metadata format
    bad_meta_p = tmp_path / "tampered_meta.safetensors"
    parsed = json.loads(meta["wsi_loopres"])
    parsed["format"] = "bad_format/0"
    save_file(tens, str(bad_meta_p), metadata={"wsi_loopres": json.dumps(parsed)})
    with pytest.raises(ValueError, match="Unsupported format"):
        LoopResModule.load(bad_meta_p, verify=True)


def test_restore_on_normal_exit_and_exception() -> None:
    model = _ToyCachedTransformerLM(n_layers=6, seed=31).eval()
    _, blocks = find_blocks(model)
    orig_ids = [id(b) for b in blocks]
    orig_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    cfg = LoopResConfig(d_model=24, core_start=1, core_end=5, n_loops=2, window=2)
    mod = LoopResModule(cfg)
    with mod.attach(model):
        assert [id(b) for b in blocks] != orig_ids
    assert [id(b) for b in blocks] == orig_ids
    for k, v in model.state_dict().items():
        assert torch.equal(v, orig_sd[k])

    with pytest.raises(RuntimeError, match="boom"):
        with mod.attach(model):
            assert [id(b) for b in blocks] != orig_ids
            raise RuntimeError("boom")
    assert [id(b) for b in blocks] == orig_ids
    for k, v in model.state_dict().items():
        assert torch.equal(v, orig_sd[k])
