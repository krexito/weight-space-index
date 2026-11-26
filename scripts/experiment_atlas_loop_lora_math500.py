"""Atlas-Guided Loopification (Latent Scratchpad) + Routed LoRA on MATH-500 (`Qwen/Qwen2.5-3B-Instruct`).

Tests whether repeating an Atlas-selected mid-layer plateau (`L20–25 x 2`, 36 -> 42 effective steps, sparing the
high-Fisher prologue `L0–19` and late DLA readout tail `L26–35`) provides a genuine hidden-state scratchpad when
paired with a parameter-matched routed LoRA cartridge (`r=16` on `L20–25`, ~7.5 MB, Layer-18 router):

  1. `base_3b_36L`                  : Unmodified Qwen2.5-3B-Instruct (36 steps)
  2. `atlas_loop_zero_shot_42L`     : Atlas mid-layer loop `L20–25 x 2` (42 steps, zero training)
  3. `lora_flat_36L_routed`         : Standard 36-step model + routed LoRA `r=16` on `L20–25`
  4. `lora_loop_shared_42L_routed`  : 42-step loop + same LoRA `r=16` active on BOTH Pass 1 & Pass 2 of `L20–25`
  5. `lora_loop_pass2_42L_routed`   : 42-step loop + dedicated Pass-2 LoRA `r=16` (Pass 1 = untouched base weights,
                                      Pass 2 = base + LoRA latent refinement; exact same parameter budget!)

Uses exact step-indexed KV caching (`_StepLayerProxy`) so Pass 1 and Pass 2 of `L20–25` own disjoint KV slots
(`20..25` vs `26..31`), enabling fast batched `generate(..., use_cache=True)` with zero KV collision.
Trains on `MATH-500` gold step-by-step solutions (`n_train=300`) + online prose anchor KL, and evaluates on
`MATH-500` (`n_test=200`) + `MMLU` (`n=400`, `non_math=371`) + `24` held-out prose controls.

CPU smoke test::
    python scripts/experiment_atlas_loop_lora_math500.py --smoke

GPU run (A100 / RTX 4090 / L4)::
    python scripts/experiment_atlas_loop_lora_math500.py --model Qwen/Qwen2.5-3B-Instruct --out benchmarks/atlas_loop_lora/results_qwen25_3b_math500.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_capability_and_boundary_lfm3 as ec  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from experiment_pocket_moe_ceiling import (  # noqa: E402
    _enable_emb_grad_hook,
    eval_healthy_drift,
    fit_ridge_router,
    predict_ridge_router,
)
from experiment_pocket_moe_controls import _paired_bootstrap  # noqa: E402
from experiment_pocket_moe_math500 import (  # noqa: E402
    MATH_DOMAINS,
    PROSE_HELDOUT,
    PROSE_ROUTER_TRAIN,
    ROUTER_CLASSES,
    _compose_routed_mmlu,
    _grade_math500_rows,
    _score_mmlu_fast,
    _summarize_math500_arm,
    _summarize_mmlu,
    format_math500_prompt_ids,
    format_mmlu_prompt_ids,
    load_math500_split,
    load_mmlu_split,
)
from experiment_pocket_moe_regularized import _score_routed_mmlu_sparse  # noqa: E402
from experiment_reasoning_distill_pilot import (  # noqa: E402
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
)
from experiment_reasoning_distill_v2 import _format_messages  # noqa: E402
from wsi.capture import find_blocks  # noqa: E402
from wsi.tracer import _tail_logits, _to_seq_tensors  # noqa: E402


# ------------------------------------------------------------------ Switchable LoRA & Step-Indexed Loop Execution
class _SwitchableLoRALinear(nn.Module):
    """Regularized LoRA wrapper with runtime `.enabled` toggle for Pass-1 vs Pass-2 routing."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha_scale: float, dropout: float = 0.10) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha_scale) / float(max(1, rank))
        self.drop = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.enabled: bool = True

        dev = base.weight.device
        out_f, in_f = int(base.weight.shape[0]), int(base.weight.shape[1])
        self.lora_A = nn.Parameter(torch.empty((self.rank, in_f), device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((out_f, self.rank), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        dx = self.drop(x.float())
        delta = (dx @ self.lora_A.t()) @ self.lora_B.t()
        return out + (delta * self.scale).to(dtype=out.dtype)

    def adapter_bytes(self) -> int:
        return int((self.lora_A.numel() + self.lora_B.numel()) * 2)

    @torch.no_grad()
    def state_tensors_cpu(self) -> dict[str, torch.Tensor]:
        return {
            "lora_A": self.lora_A.detach().to(dtype=torch.bfloat16, device="cpu").clone(),
            "lora_B": self.lora_B.detach().to(dtype=torch.bfloat16, device="cpu").clone(),
        }

    @torch.no_grad()
    def load_state_tensors(self, state: dict[str, torch.Tensor]) -> None:
        dev = self.lora_A.device
        self.lora_A.copy_(state["lora_A"].to(device=dev, dtype=torch.float32))
        self.lora_B.copy_(state["lora_B"].to(device=dev, dtype=torch.float32))


class _StepLayerProxy(nn.Module):
    """Proxies one unrolled execution step, assigning a unique KV-cache `layer_idx` and LoRA pass toggle."""

    def __init__(
        self,
        blk: nn.Module,
        *,
        step_idx: int,
        orig_layer_idx: int,
        pass_id: int,
        lora_active: bool,
        lora_modules: list[_SwitchableLoRALinear],
    ) -> None:
        super().__init__()
        # Store target block via __dict__ so nn.Module doesn't double-register parameters across passes
        object.__setattr__(self, "_blk", blk)
        object.__setattr__(self, "_lora_modules", lora_modules)
        self.step_idx = int(step_idx)
        self.orig_layer_idx = int(orig_layer_idx)
        self.pass_id = int(pass_id)
        self.lora_active = bool(lora_active)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_blk"), name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        blk = object.__getattribute__(self, "_blk")
        lora_mods = object.__getattribute__(self, "_lora_modules")
        attn = getattr(blk, "self_attn", None)
        has_lidx = attn is not None and hasattr(attn, "layer_idx")
        prev_lidx = getattr(attn, "layer_idx", None) if has_lidx else None
        prev_flags = [m.enabled for m in lora_mods]
        try:
            if has_lidx:
                attn.layer_idx = self.step_idx
            for m in lora_mods:
                m.enabled = self.lora_active
            return blk(*args, **kwargs)
        finally:
            if has_lidx:
                attn.layer_idx = prev_lidx
            for m, fl in zip(lora_mods, prev_flags):
                m.enabled = fl


def build_unrolled_schedule(
    n_layers: int,
    loop_start: int,
    loop_end: int,
    loop_repeats: int = 2,
) -> list[tuple[int, int]]:
    """Return `[(physical_layer_idx, pass_id), ...]` for prologue + repeated middle block + tail."""
    assert 0 <= loop_start <= loop_end < n_layers, (n_layers, loop_start, loop_end)
    assert loop_repeats >= 1
    sched: list[tuple[int, int]] = []
    for li in range(0, loop_start):
        sched.append((li, 0))
    for rep in range(int(loop_repeats)):
        for li in range(loop_start, loop_end + 1):
            sched.append((li, rep))
    for li in range(loop_end + 1, n_layers):
        sched.append((li, 0))
    return sched


def _find_layers_container(model: nn.Module) -> tuple[nn.Module, str, nn.ModuleList]:
    """Locate `(parent_module, attr_name, layers_modulelist)` for Qwen2 / Llama / toy models."""
    for path in (("model", "layers"), ("transformer", "h"), ("layers",)):
        cur: Any = model
        ok = True
        for part in path[:-1]:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and hasattr(cur, path[-1]) and isinstance(getattr(cur, path[-1]), nn.ModuleList):
            return cur, path[-1], getattr(cur, path[-1])
    raise RuntimeError("Could not locate decoder layers ModuleList on model")


@contextmanager
def _install_switchable_lora(
    model: nn.Module,
    layer_idxs: list[int],
    *,
    rank: int,
    alpha_scale: float,
    dropout: float = 0.10,
    state_dict: dict[str, dict[str, torch.Tensor]] | None = None,
) -> Iterator[tuple[dict[int, list[_SwitchableLoRALinear]], dict[str, _SwitchableLoRALinear]]]:
    """Temporarily wrap all Linear modules in `layer_idxs` with `_SwitchableLoRALinear`, restoring cleanly on exit."""
    _, _, layers = _find_layers_container(model)
    wrapped_records: list[tuple[nn.Module, str, nn.Linear]] = []
    by_layer: dict[int, list[_SwitchableLoRALinear]] = {int(li): [] for li in layer_idxs}
    by_key: dict[str, _SwitchableLoRALinear] = {}

    try:
        for li in layer_idxs:
            blk = layers[int(li)]
            for sub_nm, lin in [(nm, m) for nm, m in blk.named_modules() if isinstance(m, nn.Linear)]:
                parts = sub_nm.split(".")
                parent = blk
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr = parts[-1]
                w = _SwitchableLoRALinear(lin, rank=rank, alpha_scale=alpha_scale, dropout=dropout)
                key = f"L{int(li)}.{sub_nm}"
                if state_dict is not None and key in state_dict:
                    w.load_state_tensors(state_dict[key])
                setattr(parent, attr, w)
                wrapped_records.append((parent, attr, lin))
                by_layer[int(li)].append(w)
                by_key[key] = w
        yield by_layer, by_key
    finally:
        for parent, attr, orig_lin in wrapped_records:
            setattr(parent, attr, orig_lin)


@contextmanager
def _activate_unrolled_schedule(
    model: nn.Module,
    schedule: list[tuple[int, int]],
    *,
    lora_by_layer: dict[int, list[_SwitchableLoRALinear]] | None = None,
    lora_mode: str = "flat",
) -> Iterator[None]:
    """Temporarily replace `layers` with step-indexed `_StepLayerProxy` wrappers for exact KV-cached looping."""
    assert lora_mode in ("none", "flat", "loop_shared", "loop_pass2"), lora_mode
    parent, attr, orig_layers = _find_layers_container(model)
    lora_map = lora_by_layer or {}
    cfg = getattr(model, "config", None)
    had_nl = cfg is not None and hasattr(cfg, "num_hidden_layers")
    orig_nl = getattr(cfg, "num_hidden_layers", None) if had_nl else None
    orig_lt = getattr(cfg, "layer_types", None) if cfg is not None else None
    has_lt = isinstance(orig_lt, (list, tuple)) and len(orig_lt) == len(orig_layers)

    proxies: list[nn.Module] = []
    for step_idx, (li, pass_id) in enumerate(schedule):
        if lora_mode == "none":
            active = False
        elif lora_mode in ("flat", "loop_shared"):
            active = li in lora_map
        else:  # "loop_pass2"
            active = (li in lora_map) and (pass_id >= 1)
        proxies.append(
            _StepLayerProxy(
                orig_layers[li],
                step_idx=step_idx,
                orig_layer_idx=li,
                pass_id=pass_id,
                lora_active=active,
                lora_modules=lora_map.get(li, []),
            )
        )

    setattr(parent, attr, nn.ModuleList(proxies))
    if had_nl:
        cfg.num_hidden_layers = len(schedule)
    if has_lt:
        cfg.layer_types = [orig_lt[li] for li, _ in schedule]
    try:
        yield
    finally:
        setattr(parent, attr, orig_layers)
        if had_nl:
            cfg.num_hidden_layers = orig_nl
        if has_lt:
            cfg.layer_types = orig_lt


# ------------------------------------------------------------------ Chunked SFT Backward & Training
def _chunked_sft_backward(
    model: nn.Module,
    ids_dev: torch.Tensor,
    lo: int,
    hi: int,
    *,
    grad_accum: int,
    chunk_size: int = 256,
) -> float:
    """Exact token-chunked cross-entropy backward avoiding `[T, vocab]` fp32 logit OOM."""
    n_tok = int(hi - lo)
    assert n_tok > 0, (lo, hi)
    t_ids = ids_dev[0, lo + 1 : hi + 1]
    lm_head = model.get_output_embeddings()
    base_model = getattr(model, "model", None)

    if base_model is not None and lm_head is not None:
        out_base = base_model(input_ids=ids_dev, use_cache=False)
        h_tail = out_base.last_hidden_state[0, lo:hi]
        h_det = h_tail.detach().requires_grad_(True)
        total_loss_val = 0.0
        for s0 in range(0, n_tok, int(chunk_size)):
            s1 = min(n_tok, s0 + int(chunk_size))
            logits_c = lm_head(h_det[s0:s1]).float()
            loss_c = F.cross_entropy(logits_c, t_ids[s0:s1], reduction="sum") / float(n_tok)
            (loss_c / float(grad_accum)).backward()
            total_loss_val += float(loss_c.detach().item())
            del logits_c, loss_c
        assert h_det.grad is not None
        h_tail.backward(h_det.grad)
        del out_base, h_tail, h_det
        return total_loss_val

    logits = _tail_logits(model, ids_dev, lo)[:n_tok].float()
    loss = F.cross_entropy(logits, t_ids, reduction="mean")
    (loss / float(grad_accum)).backward()
    val = float(loss.detach().item())
    del logits, loss
    return val


def _train_loop_lora_arm(
    model: nn.Module,
    *,
    loop_layers: list[int],
    schedule: list[tuple[int, int]],
    lora_mode: str,
    train_samples: list[dict[str, Any]],
    rank: int,
    alpha_scale: float,
    total_steps: int,
    grad_accum: int,
    lr_val: float,
    weight_decay: float,
    lora_dropout: float,
    warmup_frac: float,
    anchor_ids: list[torch.Tensor] | None,
    base_anchor_logp: list[torch.Tensor] | None,
    anchor_kl_weight: float,
    prose_ids: list[torch.Tensor] | None,
    base_prose_logp: list[torch.Tensor] | None,
    base_prose_nll: float,
    seed: int,
) -> dict[str, Any]:
    """Train a parameter-matched LoRA adapter on `loop_layers` under `schedule` and `lora_mode`."""
    assert len(train_samples) > 0 and total_steps >= 1 and grad_accum >= 1
    dev = next(model.parameters()).device
    orig_req = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in orig_req:
        p.requires_grad_(False)
    was_training = bool(model.training)
    was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
    enabled_gc = False
    if hasattr(model, "gradient_checkpointing_enable") and not was_gc:
        try:
            model.gradient_checkpointing_enable()
            enabled_gc = True
        except Exception:
            enabled_gc = False

    torch.manual_seed(int(seed))
    emb_handles = _enable_emb_grad_hook(model)
    t_c0 = time.perf_counter()
    step_losses: list[float] = []

    try:
        with _install_switchable_lora(
            model,
            loop_layers,
            rank=rank,
            alpha_scale=alpha_scale,
            dropout=lora_dropout,
        ) as (lora_by_layer, lora_by_key):
            lora_params = [p for w in lora_by_key.values() for p in (w.lora_A, w.lora_B)]
            adapter_mb = round(sum(w.adapter_bytes() for w in lora_by_key.values()) / (1024**2), 4)

            with _activate_unrolled_schedule(model, schedule, lora_by_layer=lora_by_layer, lora_mode=lora_mode):
                model.eval()
                for w in lora_by_key.values():
                    w.train()

                n_steps = int(total_steps)
                warmup_steps = max(1, int(round(n_steps * float(warmup_frac)))) if float(warmup_frac) > 0.0 else 0
                opt = torch.optim.AdamW(lora_params, lr=float(lr_val), weight_decay=float(weight_decay))

                def _lr_lambda(step_idx: int) -> float:
                    if warmup_steps > 0 and step_idx < warmup_steps:
                        return float(step_idx + 1) / float(warmup_steps)
                    rem = max(1, n_steps - warmup_steps)
                    prog = min(1.0, max(0.0, float(step_idx - warmup_steps) / float(rem)))
                    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

                sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
                rng = torch.Generator().manual_seed(int(seed))
                idx_queue: list[int] = []

                for step_idx in range(n_steps):
                    opt.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for _ in range(int(grad_accum)):
                        if not idx_queue:
                            idx_queue = torch.randperm(len(train_samples), generator=rng).tolist()
                        sp = train_samples[idx_queue.pop()]
                        ids_dev = torch.tensor(sp["ids"], dtype=torch.long, device=dev).unsqueeze(0)
                        with torch.enable_grad():
                            lv = _chunked_sft_backward(
                                model, ids_dev, int(sp["lo"]), int(sp["hi"]), grad_accum=int(grad_accum)
                            )
                            step_loss += lv / float(grad_accum)
                        del ids_dev

                    if float(anchor_kl_weight) > 0.0 and anchor_ids and base_anchor_logp:
                        anc_i = step_idx % len(anchor_ids)
                        anc_t = anchor_ids[anc_i]
                        if anc_t.shape[1] >= 2:
                            anc_dev = anc_t.to(dev)
                            with torch.enable_grad():
                                lp_s = F.log_softmax(_tail_logits(model, anc_dev, 0)[:-1].float(), dim=-1)
                                lp_b = base_anchor_logp[anc_i].to(device=dev, dtype=torch.float32)
                                kl_anc = (lp_b.exp() * (lp_b - lp_s)).sum(dim=-1).clamp_min(0.0).mean()
                                (float(anchor_kl_weight) * kl_anc).backward()
                            del anc_dev, lp_s, lp_b, kl_anc

                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    opt.step()
                    sched.step()
                    step_losses.append(round(step_loss, 5))

                for w in lora_by_key.values():
                    w.eval()

                compile_time_s = round(time.perf_counter() - t_c0, 2)
                drift = (
                    eval_healthy_drift(model, prose_ids, base_prose_logp, base_prose_nll)
                    if (prose_ids is not None and base_prose_logp is not None)
                    else {"healthy_kl": 0.0, "delta_healthy_nll": 0.0}
                )
                adapter_state = {k: w.state_tensors_cpu() for k, w in lora_by_key.items()}
    finally:
        for h in emb_handles:
            h.remove()
        if enabled_gc and hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
        if was_training:
            model.train()
        for p, rq in orig_req:
            p.requires_grad_(rq)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    ckpts = [step_losses[0], step_losses[len(step_losses) // 2], step_losses[-1]] if step_losses else []
    return {
        "n_train_samples": len(train_samples),
        "n_steps": int(total_steps),
        "n_passes": int(total_steps * grad_accum),
        "adapter_mb": adapter_mb,
        "compile_time_s": compile_time_s,
        "step_losses_checkpoints": ckpts,
        "healthy_kl": drift["healthy_kl"],
        "delta_healthy_nll": drift["delta_healthy_nll"],
        "adapter_state": adapter_state,
    }


@torch.no_grad()
def _hf_generate_batched(
    model: nn.Module,
    tok: Any,
    prompts_ids: list[list[int]],
    *,
    max_new_tokens: int,
    batch_size: int = 16,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Fast left-padded batched HF generation using step-indexed KV cache (`use_cache=True`)."""
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        pad_id = 0
    eos_ids = lr.eos_ids(tok)
    eos_set = set(eos_ids)

    # Sort by prompt length for minimal padding waste, then unsort back
    order = sorted(range(len(prompts_ids)), key=lambda i: len(prompts_ids[i]))
    results: list[dict[str, Any] | None] = [None] * len(prompts_ids)

    do_sample = float(temperature) > 0.0
    gen_kw: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": int(pad_id),
        "eos_token_id": eos_ids if len(eos_ids) > 1 else int(eos_ids[0]),
        "use_cache": True,
    }
    if do_sample:
        gen_kw.update({"temperature": float(temperature), "top_p": 0.95, "top_k": 20})

    for b0 in range(0, len(order), int(batch_size)):
        batch_idxs = order[b0 : b0 + int(batch_size)]
        max_len = max(len(prompts_ids[i]) for i in batch_idxs)
        inp = torch.full((len(batch_idxs), max_len), int(pad_id), dtype=torch.long, device=dev)
        mask = torch.zeros((len(batch_idxs), max_len), dtype=torch.long, device=dev)
        for row_i, orig_i in enumerate(batch_idxs):
            p = prompts_ids[orig_i]
            inp[row_i, max_len - len(p) :] = torch.tensor(p, dtype=torch.long, device=dev)
            mask[row_i, max_len - len(p) :] = 1

        out_ids = model.generate(input_ids=inp, attention_mask=mask, **gen_kw)
        gen_part = out_ids[:, max_len:].cpu().tolist()
        for orig_i, toks in zip(batch_idxs, gen_part):
            trimmed: list[int] = []
            stopped_eos = False
            for t in toks:
                if t in eos_set:
                    stopped_eos = True
                    break
                trimmed.append(int(t))
            finish_reason = "stop" if stopped_eos else ("length" if len(trimmed) >= int(max_new_tokens) else "stop")
            results[orig_i] = {"token_ids": trimmed, "finish_reason": finish_reason}
        del inp, mask, out_ids
        if dev.type == "cuda" and (b0 // max(1, int(batch_size))) % 4 == 0:
            torch.cuda.empty_cache()

    assert all(r is not None for r in results)
    return [r for r in results if r is not None]


# ------------------------------------------------------------------ CPU Smoke Test
class _ToyKVCache:
    """Minimal step-indexed KV cache mirroring transformers DynamicCache for deterministic CPU verification."""

    def __init__(self) -> None:
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self.k:
            k_all = torch.cat([self.k[layer_idx], k_new], dim=1)
            v_all = torch.cat([self.v[layer_idx], v_new], dim=1)
        else:
            k_all, v_all = k_new, v_new
        self.k[layer_idx] = k_all
        self.v[layer_idx] = v_all
        return k_all, v_all


class _ToyAttn(nn.Module):
    def __init__(self, d: int, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, past_key_values: _ToyKVCache | None = None, use_cache: bool = False) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if use_cache and past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        t_q, t_k = q.shape[1], k.shape[1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        # Causal mask supporting both full prefill (t_q == t_k) and 1-token decode (t_q == 1)
        offset = t_k - t_q
        q_pos = torch.arange(t_q, device=x.device).unsqueeze(1) + offset
        k_pos = torch.arange(t_k, device=x.device).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        w = F.softmax(scores, dim=-1)
        return self.o(torch.matmul(w, v))


class _ToyBlock(nn.Module):
    def __init__(self, d: int, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = _ToyAttn(d, layer_idx)
        self.mlp = nn.Sequential(nn.Linear(d, d * 2, bias=False), nn.SiLU(), nn.Linear(d * 2, d, bias=False))

    def forward(self, x: torch.Tensor, past_key_values: _ToyKVCache | None = None, use_cache: bool = False) -> torch.Tensor:
        x = x + 0.25 * self.self_attn(x, past_key_values=past_key_values, use_cache=use_cache)
        x = x + 0.25 * self.mlp(x)
        return x


class _ToyQwenModel(nn.Module):
    def __init__(self, vocab: int = 32, d: int = 16, n_layers: int = 6) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_ToyBlock(d, i) for i in range(n_layers)])

    def forward(self, input_ids: torch.Tensor, past_key_values: _ToyKVCache | None = None, use_cache: bool = False):
        h = self.embed(input_ids)
        for blk in self.layers:
            h = blk(h, past_key_values=past_key_values, use_cache=use_cache)
        return type("_BaseOut", (), {"last_hidden_state": h})()


class _ToyQwenForCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, d: int = 16, n_layers: int = 6) -> None:
        super().__init__()
        self.config = type(
            "_Cfg",
            (),
            {"num_hidden_layers": n_layers, "layer_types": ["full_attention"] * n_layers},
        )()
        self.model = _ToyQwenModel(vocab=vocab, d=d, n_layers=n_layers)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: _ToyKVCache | None = None,
        use_cache: bool = False,
        logits_to_keep: int | None = None,
    ):
        h = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache).last_hidden_state
        if logits_to_keep is not None and logits_to_keep < h.shape[1]:
            h = h[:, -logits_to_keep:]
        return type("_LMOut", (), {"logits": self.lm_head(h)})()


def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of schedule, step-indexed KV cache, Pass-2 LoRA isolation & training."""
    torch.manual_seed(42)
    model = _ToyQwenForCausalLM(vocab=32, d=16, n_layers=6).eval()
    orig_snap = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 1. Schedule construction (6L with L2-3 x 2 -> 8 effective steps)
    sched_flat = build_unrolled_schedule(6, 2, 3, loop_repeats=1)
    sched_loop = build_unrolled_schedule(6, 2, 3, loop_repeats=2)
    assert sched_flat == [(i, 0) for i in range(6)], sched_flat
    assert sched_loop == [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (4, 0), (5, 0)], sched_loop

    # 2. Exact step-indexed KV cache equivalence vs full uncached forward under loop_pass2
    seq = torch.tensor([[1, 5, 9, 12, 3, 7]], dtype=torch.long)
    with _install_switchable_lora(model, [2, 3], rank=4, alpha_scale=8.0, dropout=0.0) as (l_by_l, l_by_k):
        for w in l_by_k.values():
            nn.init.normal_(w.lora_B, std=0.2)
        with _activate_unrolled_schedule(model, sched_loop, lora_by_layer=l_by_l, lora_mode="loop_pass2"):
            assert len(model.model.layers) == 8 and model.config.num_hidden_layers == 8
            assert len(model.config.layer_types) == 8
            with torch.no_grad():
                full_logits = model(seq, use_cache=False).logits
                cache = _ToyKVCache()
                prefill_logits = model(seq[:, :4], past_key_values=cache, use_cache=True).logits
                tok4_logits = model(seq[:, 4:5], past_key_values=cache, use_cache=True).logits
                tok5_logits = model(seq[:, 5:6], past_key_values=cache, use_cache=True).logits
                cached_logits = torch.cat([prefill_logits, tok4_logits, tok5_logits], dim=1)
            kv_max_diff = float((full_logits - cached_logits).abs().max().item())
            assert kv_max_diff < 1e-5, kv_max_diff
            assert set(cache.k.keys()) == set(range(8)), set(cache.k.keys())

            # 3. Verify Pass-2 isolation: Pass 1 (steps 2,3) has lora_active=False, Pass 2 (steps 4,5) has True
            assert model.model.layers[2].lora_active is False and model.model.layers[3].lora_active is False
            assert model.model.layers[4].lora_active is True and model.model.layers[5].lora_active is True

        # Verify Pass-2 LoRA actually changes output vs zero-shot loop (outside nested proxy so orig_layers intact)
        with _activate_unrolled_schedule(model, sched_loop, lora_by_layer=l_by_l, lora_mode="none"):
            with torch.no_grad():
                zs_logits = model(seq, use_cache=False).logits
        assert float((full_logits - zs_logits).abs().max().item()) > 1e-3

    # Verify exact restore after context managers exit
    assert len(model.model.layers) == 6 and model.config.num_hidden_layers == 6
    assert len(model.config.layer_types) == 6
    assert all(torch.equal(v, orig_snap[k]) for k, v in model.state_dict().items())

    # 4. End-to-end training loss reduction across all 3 trained modes (`flat`, `loop_shared`, `loop_pass2`)
    toy_samples = [
        {"ids": [1, 2, 3, 10, 11, 12, 13], "lo": 2, "hi": 6},
        {"ids": [4, 5, 6, 14, 15, 16, 17], "lo": 2, "hi": 6},
    ]
    h_ids = [torch.tensor([[1, 2, 3, 4]], dtype=torch.long)]
    with torch.no_grad():
        base_lp = [F.log_softmax(model(h_ids[0]).logits[0, :-1].float(), dim=-1)]
        base_nll = float(F.nll_loss(base_lp[0], h_ids[0][0, 1:], reduction="mean").item())

    ckpts_by_arm: dict[str, list[float]] = {}
    for arm_nm, sched, lmode in [
        ("lora_flat_36L", sched_flat, "flat"),
        ("lora_loop_shared_42L", sched_loop, "loop_shared"),
        ("lora_loop_pass2_42L", sched_loop, "loop_pass2"),
    ]:
        res = _train_loop_lora_arm(
            model,
            loop_layers=[2, 3],
            schedule=sched,
            lora_mode=lmode,
            train_samples=toy_samples,
            rank=4,
            alpha_scale=8.0,
            total_steps=8,
            grad_accum=2,
            lr_val=2e-2,
            weight_decay=0.05,
            lora_dropout=0.05,
            warmup_frac=0.2,
            anchor_ids=h_ids,
            base_anchor_logp=base_lp,
            anchor_kl_weight=0.05,
            prose_ids=h_ids,
            base_prose_logp=base_lp,
            base_prose_nll=base_nll,
            seed=42,
        )
        assert res["step_losses_checkpoints"][-1] < res["step_losses_checkpoints"][0], (arm_nm, res["step_losses_checkpoints"])
        assert all(torch.equal(v, orig_snap[k]) for k, v in model.state_dict().items()), arm_nm
        ckpts_by_arm[arm_nm] = res["step_losses_checkpoints"]

    return {
        "mode": "smoke",
        "passed": True,
        "schedule_ok": True,
        "step_kv_cache_max_diff": round(kv_max_diff, 8),
        "pass2_isolation_ok": True,
        "exact_restore_ok": True,
        "toy_loss_checkpoints": ckpts_by_arm,
    }


# ------------------------------------------------------------------ Full GPU Experiment
def _print_summary(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return
    print("\n=== 1. CONFIG & ROUTER ===", flush=True)
    print(
        f"  model={payload['model']} | window=L{payload['loop_start']}-{payload['loop_end']} x{payload['loop_repeats']} "
        f"({payload['n_layers']}L -> {payload['n_effective_steps']} steps) | r={payload['rank']} | "
        f"train_kept={payload['n_train_kept']}/{payload['n_math_train']}",
        flush=True,
    )
    print("  Router:", payload.get("router_report"), flush=True)

    prose_kl = payload.get("prose_kl", {})
    print("\n=== 2. MATH-500 (n=200, 5 BRAZOS PAREADOS: ATLAS LOOP + LORA) ===", flush=True)
    for arm_nm, st in payload.get("math500_arms", {}).items():
        pkl = prose_kl.get(arm_nm, 0.0)
        print(
            f"  {arm_nm:32s} | acc={st['accuracy']*100:5.1f}% ({st['n_correct']:3d}/{st['n_total']}) "
            f"| tok={st['mean_tokens']:6.1f} (med={st['median_tokens']:6.1f}) "
            f"| trunc={st['truncated_rate']*100:4.1f}% | pkl={pkl:.5f}",
            flush=True,
        )
        print(f"     per_level={st['per_level']} | per_domain={st['per_domain']}", flush=True)

    print("\n=== 3. RETENCION MMLU (n=400) ===", flush=True)
    for arm_nm, vm in payload.get("mmlu_arms", {}).items():
        a_m, nm = vm["all_400"], vm["non_math"]
        d_nm = f"{nm['delta_vs_base']*100:+5.1f} pts" if "delta_vs_base" in nm else "     base"
        print(
            f"  {arm_nm:32s} | MMLU={a_m['accuracy']*100:5.1f}% | non_math={nm['accuracy']*100:5.1f}% ({d_nm}) | "
            f"ABCD={a_m['mean_abcd_prob_mass']*100:5.1f}% | fire={vm['router_fire_rate']*100:5.1f}%",
            flush=True,
        )

    print("\n=== 4. COMPARACIONES PAREADAS (IC 95% BOOTSTRAP, MATH-500) ===", flush=True)
    for k, v in payload.get("paired_math500", {}).items():
        print(
            f"  {k:44s}: diff={v['diff']*100:+5.1f} pts "
            f"CI95=[{v['ci95'][0]*100:+5.1f}, {v['ci95'][1]*100:+5.1f}] (w={v['wins']}/l={v['losses']})",
            flush=True,
        )
    print("\nCriteria:", payload.get("criteria"), flush=True)
    print("Timing (s):", payload.get("timing_s"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "per_item_math500"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if (args.dtype == "auto" and dev.type == "cuda" and torch.cuda.is_bf16_supported())
        else (
            torch.float16
            if (args.dtype == "auto" and dev.type == "cuda")
            else {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(args.dtype, torch.float32)
        )
    )

    math_train, math_test = load_math500_split(n_train=args.n_math_train, n_test=args.n_math_test, seed=args.seed)
    mmlu_test, mmlu_router_train = load_mmlu_split(
        n_test=args.n_mmlu_test, n_router_train=args.n_mmlu_router_train, seed=args.seed
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eos_set = set(lr.eos_ids(tok))
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    router_layer = n_layers // 2
    loop_layers = list(range(int(args.loop_start), int(args.loop_end) + 1))
    sched_flat = build_unrolled_schedule(n_layers, int(args.loop_start), int(args.loop_end), loop_repeats=1)
    sched_loop = build_unrolled_schedule(n_layers, int(args.loop_start), int(args.loop_end), loop_repeats=int(args.loop_repeats))

    # Quick live-model 42-step DynamicCache sanity check before spending GPU training time
    probe_toks = tok.encode("Compute 2+2 step by step.", add_special_tokens=False)[:6] or [1, 2, 3, 4]
    probe_ids = torch.tensor([probe_toks], dtype=torch.long, device=dev)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if tok.eos_token_id is not None else 0)
    with torch.no_grad(), _activate_unrolled_schedule(model, sched_loop, lora_mode="none"):
        out_probe = model.generate(input_ids=probe_ids, max_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=int(pad_id))
        assert int(out_probe.shape[1]) > len(probe_toks), out_probe.shape
    print(f"[atlas-loop] Live {len(sched_loop)}-step KV-cache self-check OK", flush=True)

    # ---- Phase 1: Format Gold MATH-500 Train Traces, Layer-Mid Router, Base MMLU & Prose ----
    t_p1 = time.perf_counter()
    math_train_pids = [format_math500_prompt_ids(tok, it["problem"]) for it in math_train]
    math_test_pids = [format_math500_prompt_ids(tok, it["problem"]) for it in math_test]
    mmlu_r_pids = [format_mmlu_prompt_ids(tok, it) for it in mmlu_router_train]
    mmlu_test_pids = [format_mmlu_prompt_ids(tok, it) for it in mmlu_test]
    prose_r_pids = [_format_messages(tok, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_ROUTER_TRAIN]
    prose_held_pids = [_format_messages(tok, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) for p in PROSE_HELDOUT]

    train_samples: list[dict[str, Any]] = []
    for it, p_ids in zip(math_train, math_train_pids):
        sol = str(it.get("solution", "")).strip()
        if not sol:
            continue
        content = it["problem"].strip() + "\n\nSolve this problem step by step and put your final answer within \\boxed{}."
        full_ids = _format_messages(
            tok,
            [{"role": "user", "content": content}, {"role": "assistant", "content": sol}],
            add_generation_prompt=False,
            enable_thinking=False,
        )
        lo = max(0, len(p_ids) - 1)
        hi = len(full_ids) - 1
        if 0 < (hi - lo) <= int(args.max_train_tokens):
            train_samples.append({"id": it["id"], "domain": it["domain"], "ids": full_ids, "lo": lo, "hi": hi})
    print(f"[atlas-loop] Gold MATH-500 train samples kept: {len(train_samples)}/{len(math_train)}", flush=True)
    assert len(train_samples) >= 32, len(train_samples)

    r_train_pids = math_train_pids + mmlu_r_pids + prose_r_pids
    r_train_labels = [it["domain"] for it in math_train] + ["passthrough"] * (len(mmlu_r_pids) + len(prose_r_pids))
    W_r, b_r = fit_ridge_router(_extract_prompt_hidden_vecs(model, r_train_pids, router_layer), r_train_labels, ROUTER_CLASSES, lam=1.0)
    routed_math_test = predict_ridge_router(_extract_prompt_hidden_vecs(model, math_test_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_mmlu_test = predict_ridge_router(_extract_prompt_hidden_vecs(model, mmlu_test_pids, router_layer), W_r, b_r, ROUTER_CLASSES)
    routed_prose_test = predict_ridge_router(_extract_prompt_hidden_vecs(model, prose_held_pids, router_layer), W_r, b_r, ROUTER_CLASSES)

    router_report = {
        "router_layer": router_layer,
        "math_exact_domain_acc": round(sum(p == it["domain"] for p, it in zip(routed_math_test, math_test)) / len(math_test), 4),
        "math_any_domain_fire_rate": round(sum(p in MATH_DOMAINS for p in routed_math_test) / len(math_test), 4),
        "mmlu_passthrough_acc": round(sum(p == "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_fire_rate": round(sum(p != "passthrough" for p in routed_mmlu_test) / len(mmlu_test), 4),
        "mmlu_non_math_fire_rate": round(
            sum(p != "passthrough" for p, it in zip(routed_mmlu_test, mmlu_test) if not it["is_math_subject"])
            / max(1, sum(1 for it in mmlu_test if not it["is_math_subject"])),
            4,
        ),
        "prose_fpr": round(sum(p != "passthrough" for p in routed_prose_test) / len(routed_prose_test), 4),
    }

    letter_ids = ec._resolve_mmlu_letter_ids(tok)
    base_mmlu_raw = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)
    base_mmlu_summary = _summarize_mmlu(mmlu_test, base_mmlu_raw, seed=args.seed)

    def _compute_base_logp(prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor | None], float]:
        seqs = _to_seq_tensors(
            [
                _format_messages(
                    tok,
                    [{"role": "user", "content": pr}, {"role": "assistant", "content": "Here is a concise explanation."}],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                for pr in prompts
            ],
            torch.device("cpu"),
        )
        logps: list[torch.Tensor | None] = []
        nll_s, cnt = 0.0, 0
        with torch.no_grad():
            for ids in seqs:
                logits = _tail_logits(model, ids.to(dev), 0)[:-1].float()
                lp = F.log_softmax(logits, dim=-1)
                nll_s += float(F.nll_loss(lp, ids[0, 1:].to(dev), reduction="sum"))
                cnt += int(ids.shape[1] - 1)
                logps.append(lp.to(dtype=torch.bfloat16, device="cpu"))
                del logits, lp
        return seqs, logps, nll_s / max(1, cnt)

    prose_full_ids, base_prose_logp, base_prose_nll = _compute_base_logp(PROSE_HELDOUT)
    anchor_full_ids, base_anchor_logp, _ = _compute_base_logp(PROSE_ROUTER_TRAIN)
    phase1_s = round(time.perf_counter() - t_p1, 2)

    # ---- Phase 2: Train 3 Matched LoRA Arms (Flat 36L, Loop-Shared 42L, Loop-Pass2 42L) ----
    t_p2 = time.perf_counter()
    trained_specs = [
        ("lora_flat_36L", sched_flat, "flat"),
        ("lora_loop_shared_42L", sched_loop, "loop_shared"),
        ("lora_loop_pass2_42L", sched_loop, "loop_pass2"),
    ]
    adapter_states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    arm_meta: dict[str, Any] = {}
    mmlu_raw_by_arm: dict[str, dict[str, Any]] = {}
    fired_mmlu_cls = ["math" if c != "passthrough" else "passthrough" for c in routed_mmlu_test]

    # Zero-shot loop MMLU & prose drift
    with _activate_unrolled_schedule(model, sched_loop, lora_mode="none"):
        zs_drift = eval_healthy_drift(model, prose_full_ids, base_prose_logp, base_prose_nll)
        mmlu_raw_by_arm["atlas_loop_zero_shot_42L"] = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)

    for arm_base, sched, lmode in trained_specs:
        print(f"[atlas-loop] Training {arm_base} (mode={lmode}, steps={args.total_steps}x{args.grad_accum})...", flush=True)
        res = _train_loop_lora_arm(
            model,
            loop_layers=loop_layers,
            schedule=sched,
            lora_mode=lmode,
            train_samples=train_samples,
            rank=int(args.rank),
            alpha_scale=float(args.rank * 2),
            total_steps=int(args.total_steps),
            grad_accum=int(args.grad_accum),
            lr_val=float(args.lr),
            weight_decay=0.05,
            lora_dropout=0.10,
            warmup_frac=0.15,
            anchor_ids=anchor_full_ids,
            base_anchor_logp=base_anchor_logp,
            anchor_kl_weight=0.05,
            prose_ids=prose_full_ids,
            base_prose_logp=base_prose_logp,
            base_prose_nll=base_prose_nll,
            seed=int(args.seed),
        )
        st_dict = res.pop("adapter_state")
        adapter_states[arm_base] = st_dict
        arm_meta[arm_base] = res
        with _install_switchable_lora(model, loop_layers, rank=int(args.rank), alpha_scale=float(args.rank * 2), dropout=0.0, state_dict=st_dict) as (l_by_l, _):
            with _activate_unrolled_schedule(model, sched, lora_by_layer=l_by_l, lora_mode=lmode):
                if arm_base == "lora_loop_pass2_42L":
                    mmlu_raw_by_arm[arm_base] = _score_mmlu_fast(model, mmlu_test_pids, mmlu_test, letter_ids, dev)
                else:
                    mmlu_raw_by_arm[arm_base] = _score_routed_mmlu_sparse(
                        model, mmlu_test_pids, mmlu_test, fired_mmlu_cls, "math", base_mmlu_raw, letter_ids, dev
                    )
        print(
            f"[atlas-loop] {arm_base}: {res['adapter_mb']}MB, {res['compile_time_s']}s, "
            f"loss={res['step_losses_checkpoints']}, unrouted_pkl={res['healthy_kl']}",
            flush=True,
        )

    mmlu_arms: dict[str, Any] = {
        "base_3b_36L": base_mmlu_summary,
        "atlas_loop_zero_shot_42L": _summarize_mmlu(mmlu_test, mmlu_raw_by_arm["atlas_loop_zero_shot_42L"], base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=args.seed + 1),
    }
    for i_a, (arm_base, _, _) in enumerate(trained_specs):
        mmlu_arms[f"{arm_base}_routed"] = _compose_routed_mmlu(
            mmlu_test, routed_mmlu_test, base_mmlu_raw, {d: mmlu_raw_by_arm[arm_base] for d in MATH_DOMAINS}, seed=args.seed + 2 + i_a
        )
    mmlu_arms["lora_loop_pass2_42L_unrouted"] = _summarize_mmlu(
        mmlu_test, mmlu_raw_by_arm["lora_loop_pass2_42L"], base_score=base_mmlu_raw, fire_flags=[True] * len(mmlu_test), seed=args.seed + 10
    )
    phase2_s = round(time.perf_counter() - t_p2, 2)

    # ---- Phase 3: Paired Batched Generation on MATH-500 (n=200) ----
    t_p3 = time.perf_counter()
    print(f"[atlas-loop] Phase 3: Evaluating 5 arms on MATH-500 (n={len(math_test)}, bs={args.eval_batch_size})...", flush=True)
    base_raw = _hf_generate_batched(model, tok, math_test_pids, max_new_tokens=args.max_new_tokens, batch_size=args.eval_batch_size)
    with _activate_unrolled_schedule(model, sched_loop, lora_mode="none"):
        zs_loop_raw = _hf_generate_batched(model, tok, math_test_pids, max_new_tokens=args.max_new_tokens, batch_size=args.eval_batch_size)

    fired_math_idxs = [i for i, c in enumerate(routed_math_test) if c != "passthrough"]
    routed_raw_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm_base, sched, lmode in trained_specs:
        arm_out = list(base_raw)
        if fired_math_idxs:
            with _install_switchable_lora(
                model, loop_layers, rank=int(args.rank), alpha_scale=float(args.rank * 2), dropout=0.0, state_dict=adapter_states[arm_base]
            ) as (l_by_l, _):
                with _activate_unrolled_schedule(model, sched, lora_by_layer=l_by_l, lora_mode=lmode):
                    sub_outs = _hf_generate_batched(
                        model, tok, [math_test_pids[i] for i in fired_math_idxs], max_new_tokens=args.max_new_tokens, batch_size=args.eval_batch_size
                    )
            for idx_i, o in zip(fired_math_idxs, sub_outs):
                arm_out[idx_i] = o
        routed_raw_by_arm[arm_base] = arm_out
    phase3_s = round(time.perf_counter() - t_p3, 2)

    # ---- Phase 4: Grading & Paired Bootstrap CIs ----
    arm_rows: dict[str, list[dict[str, Any]]] = {
        "base_3b_36L": _grade_math500_rows(math_test, base_raw, tok, eos_set, args.max_new_tokens),
        "atlas_loop_zero_shot_42L": _grade_math500_rows(math_test, zs_loop_raw, tok, eos_set, args.max_new_tokens),
    }
    for arm_base, _, _ in trained_specs:
        arm_rows[f"{arm_base}_routed"] = _grade_math500_rows(
            math_test, routed_raw_by_arm[arm_base], tok, eos_set, args.max_new_tokens, routed_labels=routed_math_test
        )

    math500_arms = {nm: _summarize_math500_arm(rows) for nm, rows in arm_rows.items()}
    prose_fire_frac = sum(1 for c in routed_prose_test if c != "passthrough") / max(1, len(routed_prose_test))
    prose_kl = {
        "base_3b_36L": 0.0,
        "atlas_loop_zero_shot_42L": zs_drift["healthy_kl"],
        **{f"{arm_base}_routed": round(float(arm_meta[arm_base]["healthy_kl"]) * prose_fire_frac, 6) for arm_base, _, _ in trained_specs},
        "lora_loop_pass2_42L_unrouted": float(arm_meta["lora_loop_pass2_42L"]["healthy_kl"]),
    }

    comp_pairs = [
        ("atlas_loop_zs_vs_base", "atlas_loop_zero_shot_42L", "base_3b_36L"),
        ("lora_flat_vs_base", "lora_flat_36L_routed", "base_3b_36L"),
        ("lora_loop_shared_vs_lora_flat", "lora_loop_shared_42L_routed", "lora_flat_36L_routed"),
        ("lora_loop_pass2_vs_lora_flat", "lora_loop_pass2_42L_routed", "lora_flat_36L_routed"),
        ("lora_loop_pass2_vs_lora_loop_shared", "lora_loop_pass2_42L_routed", "lora_loop_shared_42L_routed"),
        ("lora_loop_pass2_vs_base", "lora_loop_pass2_42L_routed", "base_3b_36L"),
    ]
    paired_math500 = {lbl: _paired_bootstrap(arm_rows[a], arm_rows[b], seed=args.seed + i) for i, (lbl, a, b) in enumerate(comp_pairs)}

    acc = lambda k: float(math500_arms[k]["accuracy"])  # noqa: E731
    nm_mmlu = lambda k: float(mmlu_arms[k]["non_math"]["accuracy"])  # noqa: E731
    routed_names = [f"{a}_routed" for a, _, _ in trained_specs]

    return {
        "model": args.model,
        "dtype": str(dtype),
        "n_layers": n_layers,
        "loop_start": int(args.loop_start),
        "loop_end": int(args.loop_end),
        "loop_repeats": int(args.loop_repeats),
        "n_effective_steps": len(sched_loop),
        "n_math_train": len(math_train),
        "n_train_kept": len(train_samples),
        "n_math_test": len(math_test),
        "rank": int(args.rank),
        "total_steps": int(args.total_steps),
        "grad_accum": int(args.grad_accum),
        "max_new_tokens": int(args.max_new_tokens),
        "timing_s": {
            "phase1_setup_s": phase1_s,
            "phase2_train_mmlu_s": phase2_s,
            "phase3_eval_s": phase3_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "router_report": router_report,
        "arm_meta": arm_meta,
        "prose_kl": prose_kl,
        "math500_arms": math500_arms,
        "mmlu_arms": mmlu_arms,
        "paired_math500": paired_math500,
        "criteria": {
            "a_loop_pass2_beats_lora_flat": bool(acc("lora_loop_pass2_42L_routed") > acc("lora_flat_36L_routed")),
            "b_loop_shared_beats_lora_flat": bool(acc("lora_loop_shared_42L_routed") > acc("lora_flat_36L_routed")),
            "c_best_loop_beats_base": bool(max(acc("lora_loop_pass2_42L_routed"), acc("lora_loop_shared_42L_routed")) > acc("base_3b_36L")),
            "d_zero_general_forgetting_all_routed": bool(
                all(prose_kl[nm] <= 0.005 and nm_mmlu(nm) >= nm_mmlu("base_3b_36L") - 0.015 for nm in routed_names)
            ),
        },
        "per_item_math500": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU verification")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--loop-start", type=int, default=20)
    ap.add_argument("--loop-end", type=int, default=25)
    ap.add_argument("--loop-repeats", type=int, default=2)
    ap.add_argument("--n-math-train", type=int, default=300)
    ap.add_argument("--n-math-test", type=int, default=200)
    ap.add_argument("--n-mmlu-test", type=int, default=400)
    ap.add_argument("--n-mmlu-router-train", type=int, default=200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--total-steps", type=int, default=160)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--max-train-tokens", type=int, default=1536)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
        _print_summary(payload)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
