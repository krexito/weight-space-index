"""Global Fisher sensitivity profiling and heterogeneous column allocation.

Computes per-input-column empirical Fisher impact under a perturbation ``dW``
(or weight zeroing ``dW = W`` for structured neuron pruning) and allocates a
parameter budget either uniformly per module or globally across all layers in
units of Fisher loss sensitivity per parameter.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .capture import find_blocks, find_mlps


@dataclass
class FisherProfile:
    """Per-module, per-input-column empirical Fisher sensitivity scores."""

    scores: dict[str, torch.Tensor]
    act_sq: dict[str, torch.Tensor]
    in_features: dict[str, int]
    out_features: dict[str, int]
    n_tokens: int
    n_seqs: int


def find_mlp_down_projections(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Return ordered ``(module_name, nn.Linear)`` pairs for MLP down-projections.

    Uses ``wsi.capture.find_mlps`` (covering Llama/Qwen ``down_proj``,
    LFM3 ``w2``, GPT-2 ``c_proj``, OPT ``fc2``) and falls back to decoder
    block child names when dimensions are unconventional.
    """
    named = dict(model.named_modules())
    out: list[tuple[str, nn.Linear]] = []
    for grp in find_mlps(model):
        mod = named.get(grp.down)
        if isinstance(mod, nn.Linear):
            out.append((grp.down, mod))
    if out:
        return out

    prefix, blocks = find_blocks(model)
    mod_to_name = {m: n for n, m in named.items()}
    for i, blk in enumerate(blocks):
        for attr_path in ("feed_forward.w2", "mlp.down_proj", "mlp.c_proj", "mlp.fc2", "fc2"):
            cur: nn.Module | None = blk
            for part in attr_path.split("."):
                cur = getattr(cur, part, None) if cur is not None else None
            if isinstance(cur, nn.Linear):
                nm = mod_to_name.get(cur) or (f"{prefix}.{i}.{attr_path}" if prefix else f"{i}.{attr_path}")
                out.append((nm, cur))
                break
    return out


def _attach_grad_anchor(model: nn.Module) -> list[Any]:
    """Attach a forward hook forcing gradient flow at model input without unfreezing weights."""
    handles: list[Any] = []
    emb_mod = (
        model.get_input_embeddings()
        if hasattr(model, "get_input_embeddings")
        else getattr(getattr(model, "model", model), "embed_tokens", None)
    )
    if emb_mod is not None:
        def _emb_hook(_m, _inp, emb_out):
            t = emb_out[0] if isinstance(emb_out, tuple) else emb_out
            if isinstance(t, torch.Tensor) and not t.requires_grad:
                t2 = t.detach().requires_grad_(True)
                return (t2, *emb_out[1:]) if isinstance(emb_out, tuple) else t2
            return emb_out

        handles.append(emb_mod.register_forward_hook(_emb_hook))
        return handles

    _, blks = find_blocks(model)
    if len(blks) > 0:
        def _blk0_pre(_m, inp):
            x0 = inp[0] if isinstance(inp, tuple) and inp else inp
            if isinstance(x0, torch.Tensor) and not x0.requires_grad:
                x0_t = x0.detach().requires_grad_(True)
                return (x0_t, *inp[1:]) if isinstance(inp, tuple) else x0_t
            return None

        handles.append(blks[0].register_forward_pre_hook(_blk0_pre))
    return handles


def _resolve_device(model: nn.Module, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def profile_column_fisher(
    model: nn.Module,
    modules: Sequence[tuple[str, nn.Linear]],
    cal_ids: Iterable[Sequence[int]],
    *,
    delta_w: dict[str, torch.Tensor] | None = None,
    device: str | torch.device | None = None,
    max_tokens: int | None = None,
) -> FisherProfile:
    """Measure per-input-column empirical Fisher sensitivity over calibration sequences.

    For each target ``nn.Linear`` module and token position ``t``, computes:
        ``score[j] += sum_t (x[t, j] * (g[t] @ dW[:, j])) ** 2``
    where ``x`` is the module's input, ``g = d(loss)/d(out)`` under sequence-mean
    next-token NLL loss, and ``dW`` is ``delta_w[name]`` if provided (perturbation /
    quantization error) or ``mod.weight`` when ``delta_w is None`` (structured column
    pruning).
    """
    mod_list = list(modules)
    if not mod_list:
        raise ValueError("modules must be non-empty")

    dev = _resolve_device(model, device)
    scores: dict[str, torch.Tensor] = {}
    act_sq: dict[str, torch.Tensor] = {}
    in_features: dict[str, int] = {}
    out_features: dict[str, int] = {}
    dw_cache: dict[str, torch.Tensor] = {}
    dw_cpu: dict[str, torch.Tensor] = {}
    use_gpu_cache = dev.type == "cuda"

    for nm, mod in mod_list:
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"Expected nn.Linear for {nm!r}, got {type(mod).__name__}")
        in_f = int(mod.in_features)
        out_f = int(mod.out_features)
        in_features[nm] = in_f
        out_features[nm] = out_f
        scores[nm] = torch.zeros(in_f, dtype=torch.float64, device="cpu")
        act_sq[nm] = torch.zeros(in_f, dtype=torch.float64, device="cpu")

        if delta_w is not None:
            if nm not in delta_w:
                raise KeyError(f"delta_w missing entry for module {nm!r}")
            w_src = delta_w[nm].detach()
        else:
            w_src = mod.weight.detach()
        if tuple(w_src.shape) != (out_f, in_f):
            raise ValueError(f"Shape mismatch for {nm!r}: expected {(out_f, in_f)}, got {tuple(w_src.shape)}")

        dw_cpu[nm] = w_src if w_src.device.type == "cpu" else w_src.cpu()
        if use_gpu_cache:
            try:
                dw_cache[nm] = w_src.to(device=dev)
            except torch.cuda.OutOfMemoryError:
                dw_cache.clear()
                use_gpu_cache = False
                torch.cuda.empty_cache()

    orig_req = [bool(p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    was_training = bool(model.training)
    model.eval()
    n_tokens = 0
    n_seqs = 0
    try:
        for seq in cal_ids:
            seq_list = [int(t) for t in seq]
            if len(seq_list) < 2:
                continue
            if max_tokens is not None and n_tokens >= int(max_tokens):
                break

            inp_dev = torch.tensor([seq_list], dtype=torch.long, device=dev)
            handles = _attach_grad_anchor(model)
            saved_x: dict[str, torch.Tensor] = {}

            for nm, mod in mod_list:
                in_f = in_features[nm]
                out_f = out_features[nm]

                def _lin_hook(_m, inp, out, nm=nm, in_f=in_f, out_f=out_f):
                    x_raw = inp[0] if isinstance(inp, tuple) else inp
                    if isinstance(x_raw, torch.Tensor):
                        x2 = x_raw.detach().reshape(-1, in_f)
                        saved_x[nm] = x2 if x2.device.type == dev.type else x2.cpu()
                        act_sq[nm] += x2.float().pow(2).sum(dim=0).double().cpu()

                    t = out[0] if isinstance(out, tuple) else out
                    if isinstance(t, torch.Tensor):
                        need_rep = False
                        if not t.requires_grad:
                            t = t.detach().requires_grad_(True)
                            need_rep = True

                        def _grad_hook(grad: torch.Tensor, nm=nm, out_f=out_f) -> None:
                            g2 = grad.detach().reshape(-1, out_f).float()
                            x_ref = saved_x.pop(nm, None)
                            if x_ref is None:
                                return
                            x_f = x_ref.to(device=g2.device, dtype=torch.float32)
                            dw_mat = (
                                dw_cache[nm].float()
                                if nm in dw_cache
                                else dw_cpu[nm].to(device=g2.device, dtype=torch.float32)
                            )
                            proj = g2 @ dw_mat
                            scores[nm] += (x_f * proj).pow(2).sum(dim=0).double().cpu()

                        t.register_hook(_grad_hook)
                        if need_rep:
                            return (t, *out[1:]) if isinstance(out, tuple) else t
                    return out

                handles.append(mod.register_forward_hook(_lin_hook))

            try:
                with torch.enable_grad():
                    out = model(input_ids=inp_dev, use_cache=False)
                    logits_raw = out.logits if hasattr(out, "logits") else out
                    logits = logits_raw[0, :-1].float()
                    loss = F.nll_loss(F.log_softmax(logits, dim=-1), inp_dev[0, 1:], reduction="mean")
                    loss.backward()
                n_tokens += len(seq_list)
                n_seqs += 1
            finally:
                for h in handles:
                    h.remove()
                saved_x.clear()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        dw_cache.clear()
        for p, rq in zip(model.parameters(), orig_req):
            p.requires_grad_(rq)
        model.train(was_training)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return FisherProfile(
        scores=scores,
        act_sq=act_sq,
        in_features=in_features,
        out_features=out_features,
        n_tokens=n_tokens,
        n_seqs=n_seqs,
    )


def _clamp_align_count(raw: float, in_f: int, min_per_module: int, align_to: int) -> int:
    lo = max(1, min(in_f, int(min_per_module)))
    a = max(1, int(align_to))
    if a == 1:
        return max(lo, min(in_f, int(round(raw))))
    lo_a = min(in_f, ((lo + a - 1) // a) * a)
    hi_a = max(lo_a, (in_f // a) * a) if in_f >= a else in_f
    k = int(round(raw / a)) * a
    return max(lo_a, min(hi_a, k))


def allocate_fisher_uniform(
    profile: FisherProfile,
    budget_frac: float,
    *,
    min_per_module: int = 1,
    align_to: int = 1,
) -> dict[str, torch.Tensor]:
    """Select top-``k`` columns uniformly per module by Fisher score."""
    if not (0.0 < float(budget_frac) <= 1.0):
        raise ValueError(f"budget_frac must be in (0, 1], got {budget_frac}")
    out: dict[str, torch.Tensor] = {}
    for nm, sc in profile.scores.items():
        in_f = int(profile.in_features[nm])
        k = _clamp_align_count(float(budget_frac) * in_f, in_f, min_per_module, align_to)
        idx = torch.argsort(sc.double(), descending=True, stable=True)[:k].sort().values.cpu()
        out[nm] = idx.to(dtype=torch.long)
    return out


def allocate_fisher_global(
    profile: FisherProfile,
    budget_frac: float,
    *,
    min_per_module: int = 1,
    align_to: int = 1,
) -> dict[str, torch.Tensor]:
    """Allocate a total parameter budget globally across modules by Fisher/parameter.

    Matches the total parameter count of ``allocate_fisher_uniform`` at ``budget_frac``,
    reserves ``min_per_module`` top columns per module, and greedily spends remaining
    parameter capacity across layers ordered by ``score / out_features``.
    """
    if not (0.0 < float(budget_frac) <= 1.0):
        raise ValueError(f"budget_frac must be in (0, 1], got {budget_frac}")

    names = list(profile.scores.keys())
    a = max(1, int(align_to))
    ranked: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    total_budget_params = 0
    used_params = 0

    for nm in names:
        in_f = int(profile.in_features[nm])
        out_f = int(profile.out_features[nm])
        ranked[nm] = torch.argsort(profile.scores[nm].double(), descending=True, stable=True).cpu()
        k_unif = _clamp_align_count(float(budget_frac) * in_f, in_f, min_per_module, a)
        total_budget_params += k_unif * out_f
        lo = max(1, min(in_f, int(min_per_module)))
        k_floor = min(in_f, ((lo + a - 1) // a) * a) if a > 1 else lo
        counts[nm] = k_floor
        used_params += k_floor * out_f

    if used_params < total_budget_params:
        heap: list[tuple[float, float, int, str, int, int]] = []

        def _push_chunk(mod_i: int, nm: str) -> None:
            in_f = int(profile.in_features[nm])
            cur = counts[nm]
            if cur >= in_f:
                return
            out_f = int(profile.out_features[nm])
            step = min(a, in_f - cur)
            cost = step * out_f
            if used_params + cost > total_budget_params:
                return
            cols = ranked[nm][cur : cur + step]
            gain = float(profile.scores[nm][cols].double().sum().item())
            benefit = gain / max(1, cost)
            heapq.heappush(heap, (-benefit, -gain, mod_i, nm, step, cost))

        for mod_i, nm in enumerate(names):
            _push_chunk(mod_i, nm)

        while heap:
            _nb, _ng, mod_i, nm, step, cost = heapq.heappop(heap)
            if used_params + cost > total_budget_params:
                continue
            counts[nm] += step
            used_params += cost
            _push_chunk(mod_i, nm)

    return {
        nm: ranked[nm][: counts[nm]].sort().values.to(dtype=torch.long, device="cpu")
        for nm in names
    }
