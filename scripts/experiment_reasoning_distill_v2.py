"""Pilot v2: Multi-Site & Multi-Rank Reasoning-to-MoI Distillation (`Qwen/Qwen3.5-0.8B`, vLLM @ 4096 tokens).

Upgrades Pilot v1 along four axes:
  1. Uncensored long-horizon generation (`max_new_tokens=4096` via vLLM with in-memory weight swaps
     and exact int16 checksum verification, fallback to baked safetensors checkpoint, paired seed per question).
  2. Strict grading (`\\boxed{}` after `</think>`, closed `</think>` rate, `loop_kill_v2` loop rate, real
     `<think>` token length) alongside pre-loop `rescue_answer`.
  3. Multi-site (`k=1..4` causal MLP + linear_attn/self_attn sites) + Multi-rank (`rank=1..3` top eigenvectors
     of pooled healthy-centered target curvature) distillation into `.moipatch` experts (~15-150 KB).
  4. Seven paired arms on 48 held-out `renumbered` + `rephrased_renumbered` problems:
       `base`, `base_no_think`, `oracle_1shot`, `random_matched_routed`,
       `global_distilled`, `single_site_r1_routed`, `multi_site_mr_routed`.

CPU smoke test::
    python scripts/experiment_reasoning_distill_v2.py --smoke

Colab L4 run::
    python scripts/experiment_reasoning_distill_v2.py --model /content/base --out /content/reasoning_distill_v2_l4.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import discover_math500_blind as dm  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
from experiment_reasoning_distill_pilot import (  # noqa: E402
    FAMILIES,
    _PlantedReasoningLM,
    _atomic_write_json,
    _extract_prompt_hidden_vecs,
    build_reasoning_dataset,
    compute_invariance_metrics,
)
from wsi import (  # noqa: E402
    CompactTrace,
    Edit,
    MoIPatch,
    capture_causal_trace,
    diagnose_causal_traces,
    screen_patch_candidates,
)
from wsi.capture import find_blocks, find_mlps  # noqa: E402
from wsi.moipatch import _dtype_name, _sha256, apply_edit, module_suffix  # noqa: E402

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
_ANS_EXPLICIT_RE = re.compile(r"(?:answer\s+is|equals|=)\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\b", re.IGNORECASE)
_NUM_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_LAYER_ATTN_RE = re.compile(r"^layers\.(\d+)\.(?:linear_attn|self_attn)\.(?:out_proj|o_proj)$")


def format_question(q: str) -> str:
    return f"{q.rstrip()}\n\n{INSTRUCTION}"


def _format_messages(
    tok: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[int]:
    try:
        s = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking
        )
    except TypeError:
        s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return [int(x) for x in tok.encode(s, add_special_tokens=False)]


def _parse_int(raw: str | None) -> int | None:
    if not raw:
        return None
    s = raw.replace("\\$", "").replace("$", "").replace(",", "").strip()
    s = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}", lambda m: m.group(1) or m.group(2) or "", s).strip()
    if s.endswith("."):
        s = s[:-1].strip()
    if not s:
        return None
    try:
        val = float(s)
    except ValueError:
        m = _NUM_TOKEN_RE.findall(s)
        if not m:
            return None
        try:
            val = float(m[-1].replace(",", ""))
        except ValueError:
            return None
    if not math_isfinite(val):
        return None
    return int(round(val))


def math_isfinite(x: float) -> bool:
    import math
    return math.isfinite(x)


def _extract_explicit_int(segment: str) -> tuple[int | None, bool]:
    """Return (parsed_int, has_boxed) from `\\boxed{...}` or explicit `answer is X` without arbitrary last-number fallback."""
    boxed_str = dm._extract_boxed_raw(segment)
    if boxed_str is not None:
        val = _parse_int(boxed_str)
        if val is not None:
            return val, True
    m_ans = _ANS_EXPLICIT_RE.findall(segment)
    if m_ans:
        val = _parse_int(m_ans[-1])
        if val is not None:
            return val, False
    return None, (boxed_str is not None)


def grade_response(
    token_ids: list[int],
    text: str,
    gold: int,
    *,
    thinking_enabled: bool,
    finish_reason: str | None,
    eos_set: set[int] | None = None,
    tok: Any | None = None,
    think_end_id: int | None = None,
) -> dict[str, Any]:
    """Grade one response strictly (`</think>` + `\\boxed{}`) and with pre-loop rescue."""
    body_ids = lr.strip_eos(list(token_ids), eos_set or set())
    total_tokens = len(body_ids)
    hit = lr.first_window_hit(lr.dup_flags(body_ids, lr.K_DUP), lr.KILL_WINDOW, lr.KILL_FRAC)
    loop_killed = hit is not None
    kill_pos = int(hit) if hit is not None else None
    truncated = bool(not loop_killed and finish_reason == "length")
    closed_think = "</think>" in text

    if not thinking_enabled:
        think_tokens = 0
    elif closed_think:
        if think_end_id is not None and think_end_id in body_ids:
            think_tokens = body_ids.index(think_end_id) + 1
        elif tok is not None:
            end_char = text.index("</think>") + len("</think>")
            think_tokens = min(total_tokens, max(1, len(tok.encode(text[:end_char], add_special_tokens=False))))
        else:
            end_char = text.index("</think>") + len("</think>")
            think_tokens = min(total_tokens, max(1, int(round(total_tokens * end_char / max(1, len(text))))))
    else:
        think_tokens = total_tokens

    strict_pred: int | None = None
    has_boxed = False
    if not loop_killed:
        if thinking_enabled:
            if closed_think:
                post = text.split("</think>")[-1]
                strict_pred, has_boxed = _extract_explicit_int(post)
                if strict_pred is None:
                    nums = _NUM_TOKEN_RE.findall(post)
                    if nums:
                        strict_pred = _parse_int(nums[-1])
        else:
            strict_pred, has_boxed = _extract_explicit_int(text)
            if strict_pred is None:
                nums = _NUM_TOKEN_RE.findall(text)
                if nums:
                    strict_pred = _parse_int(nums[-1])

    strict_correct = bool((not loop_killed) and (strict_pred == int(gold)))

    rescued_pred = strict_pred
    if rescued_pred is None:
        if kill_pos is not None:
            pre_len = max(1, kill_pos - lr.KILL_WINDOW)
            if tok is not None and body_ids:
                pre_text = tok.decode(body_ids[:pre_len], skip_special_tokens=True)
            else:
                char_cut = max(1, int(len(text) * pre_len / max(1, total_tokens)))
                pre_text = text[:char_cut]
        else:
            pre_text = text
        rescued_pred, _ = _extract_explicit_int(pre_text)

    rescued_correct = bool(rescued_pred == int(gold))
    return {
        "gold": int(gold),
        "strict_pred": strict_pred,
        "strict_correct": strict_correct,
        "rescued_pred": rescued_pred,
        "rescued_correct": rescued_correct,
        "closed_think": closed_think,
        "has_boxed": has_boxed,
        "loop_killed": loop_killed,
        "kill_pos": kill_pos,
        "truncated": truncated,
        "total_tokens": total_tokens,
        "think_tokens": think_tokens,
    }


# ------------------------------------------------------------------ Multi-Site + Multi-Rank Distillation
def _collect_mod_meta(model: nn.Module) -> dict[str, dict[str, Any]]:
    mods = dict(model.named_modules())
    mod_meta: dict[str, dict[str, Any]] = {}
    for g in find_mlps(model):
        mk = module_suffix(g.down)
        w = mods[g.down].weight.detach()
        mod_meta[mk] = {"base_sha256": _sha256(w), "base_dtype": _dtype_name(w.dtype), "shape": tuple(w.shape)}
    for n, m in model.named_modules():
        if "mtp" not in n and "visual" not in n and n.endswith((".o_proj", ".out_proj")) and ".mlp." not in n:
            try:
                mk = module_suffix(n)
                w = m.weight.detach()
                mod_meta[mk] = {"base_sha256": _sha256(w), "base_dtype": _dtype_name(w.dtype), "shape": tuple(w.shape)}
            except ValueError:
                pass
    return mod_meta


def _extract_top_eigenvectors(traces: list[CompactTrace], mk: str, max_rank: int = 3) -> list[torch.Tensor]:
    """Pool centered target curvature across `traces` and return up to `max_rank` orthonormal eigenvectors."""
    gs0 = next(ct.grad_summary[mk] for ct in traces if mk in ct.grad_summary)
    D = int(gs0["d_out"])
    C_t = torch.zeros((D, D), dtype=torch.float32)
    paired_sum = torch.zeros(D, dtype=torch.float32)
    tot_t = sum(max(1, int(ct.meta.get("grad_n_target", max(1, ct.seq_len - 1)))) for ct in traces if mk in ct.grad_summary)

    for ct in traces:
        if mk not in ct.grad_summary:
            continue
        gs = ct.grad_summary[mk]
        wt = max(1, int(ct.meta.get("grad_n_target", max(1, ct.seq_len - 1)))) / max(1, tot_t)
        vt = gs.get("c_target_centered_eigvecs", gs["c_target_eigvecs"]).float()
        lt = gs.get("c_target_centered_eigvals", gs["c_target_eigvals"]).float()
        C_t += wt * (vt.t() @ (lt[:, None] * vt))
        paired_sum += wt * gs["paired_diff"].float()

    C_t = 0.5 * (C_t + C_t.t())
    evals, evecs = torch.linalg.eigh(C_t)
    order = torch.argsort(evals, descending=True)
    lam_max = float(evals[order[0]].item())
    if lam_max <= 1e-12:
        return []

    out: list[torch.Tensor] = []
    for idx in order[:max_rank]:
        lam = float(evals[int(idx.item())].item())
        if lam < 0.10 * lam_max:
            break
        u = evecs[:, int(idx.item())].contiguous().float()
        u = u / u.norm().clamp_min(1e-12)
        if float(paired_sum.norm().item()) > 1e-12:
            if float((u @ paired_sum).item()) < 0.0:
                u = -u
        else:
            top_c = int(u.abs().argmax().item())
            if float(u[top_c].item()) < 0.0:
                u = -u
        out.append(u)
    return out


def build_multirank_grid(
    traces: list[CompactTrace],
    mod_meta: dict[str, dict[str, Any]],
    *,
    prefix: str,
    model_or_weights: Any,
    global_mode: bool = False,
) -> dict[str, Any]:
    """Build `single_site_r1` and `multi_site_mr` candidate grids from causal traces."""
    diag = diagnose_causal_traces(
        traces,
        split_ab=(len(traces) >= 2),
        top_k_sites=6,
        top_n_neurons=64,
        center="healthy",
        healthy_cost="fisher",
        model_or_weights=model_or_weights,
    )
    sites = [r["module"] for r in diag["sites"][:4] if r["module"] in mod_meta]
    site_vecs: dict[str, list[torch.Tensor]] = {}
    for mk in sites:
        vecs = _extract_top_eigenvectors(traces, mk, max_rank=3)
        if not vecs:
            vecs = [diag["directions"][mk].float() / diag["directions"][mk].float().norm().clamp_min(1e-12)]
        site_vecs[mk] = vecs

    single_cands: dict[str, MoIPatch] = {}
    if sites:
        mk0 = sites[0]
        mm0 = mod_meta[mk0]
        u0 = site_vecs[mk0][0].double()
        for a in (-6.0, -4.0, -2.0):
            cname = f"{prefix}_s1_k1_r1_a{a:g}"
            single_cands[cname] = MoIPatch(
                edits=[Edit(module=mk0, u=u0, alpha=float(a), **mm0)],
                meta={"name": cname, "kind": "single_site_r1", "k": 1, "rank": 1, "alpha": float(a)},
            )

    mr_specs: list[tuple[int, int, tuple[float, ...]]]
    if global_mode:
        mr_specs = [
            (2, 1, (-4.0, -2.5, -1.5)),
            (2, 2, (-4.0, -2.5, -1.5)),
            (3, 1, (-3.0, -2.0, -1.0)),
            (3, 2, (-3.0, -2.0, -1.0)),
            (4, 1, (-2.5, -1.5, -1.0)),
            (4, 2, (-2.5, -1.5, -1.0)),
        ]
    else:
        mr_specs = [
            (1, 2, (-4.0, -2.5, -1.5)),
            (1, 3, (-4.0, -2.5, -1.5)),
            (2, 1, (-4.0, -2.5, -1.5)),
            (2, 2, (-4.0, -2.5, -1.5)),
            (2, 3, (-4.0, -2.5, -1.5)),
            (3, 1, (-3.0, -2.0, -1.0)),
            (3, 2, (-3.0, -2.0, -1.0)),
            (3, 3, (-3.0, -2.0, -1.0)),
            (4, 1, (-2.5, -1.5, -1.0)),
            (4, 2, (-2.5, -1.5, -1.0)),
        ]

    multi_cands: dict[str, MoIPatch] = {}
    seen_keys: set[tuple[Any, ...]] = set()
    for k_req, r_req, alphas in mr_specs:
        if k_req > len(sites):
            continue
        use_sites = sites[:k_req]
        max_avail_r = max(len(site_vecs[mk]) for mk in use_sites)
        if r_req > max_avail_r:
            continue
        eff_sig = tuple((mk, min(r_req, len(site_vecs[mk]))) for mk in use_sites)
        for a in alphas:
            key = (eff_sig, float(a))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            edits: list[Edit] = []
            for mk, r_cnt in eff_sig:
                mm = mod_meta[mk]
                for u_vec in site_vecs[mk][:r_cnt]:
                    edits.append(Edit(module=mk, u=u_vec.double(), alpha=float(a), **mm))
            cname = f"{prefix}_mr_k{k_req}_r{r_req}_a{a:g}"
            multi_cands[cname] = MoIPatch(
                edits=edits,
                meta={"name": cname, "kind": "multi_site_mr", "k": k_req, "rank": r_req, "alpha": float(a)},
            )

    return {
        "diag": diag,
        "sites": sites,
        "single_site_r1": single_cands,
        "multi_site_mr": multi_cands,
    }


def make_random_matched_patch(ref_patch: MoIPatch, *, name: str, seed: int) -> MoIPatch:
    """Create a matched-geometry random patch (same modules, rank, sparsity, alpha) orthogonal to `ref_patch`."""
    g_rng = torch.Generator().manual_seed(seed)
    real_by_mod: dict[str, list[torch.Tensor]] = {}
    for e in ref_patch.edits:
        u_f = e.u.detach().float().cpu()
        real_by_mod.setdefault(e.module, []).append(u_f / u_f.norm().clamp_min(1e-12))

    rnd_by_mod: dict[str, list[torch.Tensor]] = {}
    for mk, rvecs in real_by_mod.items():
        D = int(rvecs[0].numel())
        gen_vecs: list[torch.Tensor] = []
        for _ in rvecs:
            v = torch.randn(D, generator=g_rng, dtype=torch.float32)
            for q in (*rvecs, *gen_vecs):
                v = v - float((v @ q).item()) * q
            v = v / v.norm().clamp_min(1e-12)
            gen_vecs.append(v)
        rnd_by_mod[mk] = gen_vecs

    used_idx = {mk: 0 for mk in real_by_mod}
    edits: list[Edit] = []
    for e in ref_patch.edits:
        j = used_idx[e.module]
        used_idx[e.module] = j + 1
        edits.append(
            Edit(
                module=e.module,
                u=rnd_by_mod[e.module][j].double(),
                alpha=float(e.alpha),
                idx=e.idx.clone() if e.idx is not None else None,
                base_sha256=e.base_sha256,
                base_dtype=e.base_dtype,
                shape=e.shape,
            )
        )
    return MoIPatch(edits=edits, meta={"name": name, "kind": "random_matched"})


def _select_and_eval_distill(
    model: nn.Module,
    cands: dict[str, MoIPatch],
    *,
    train_seqs: list[list[int]],
    train_spans: list[tuple[int, int]],
    test_seqs: list[list[int]],
    test_spans: list[tuple[int, int]],
    healthy_train: list[list[int]],
    healthy_test: list[list[int]],
) -> tuple[str, MoIPatch, dict[str, Any], dict[str, Any]]:
    train_rows = screen_patch_candidates(
        model,
        cands,
        healthy_seqs=healthy_train,
        target_seqs=train_seqs,
        target_spans=train_spans,
        max_nll_delta=0.04,
    )
    for r in train_rows:
        gain = -float(r["target_suppression"])
        r["teacher_logp_gain"] = round(gain, 6)
        r["distill_pareto"] = round(gain / max(float(r["healthy_kl"]), 0.005), 6)
        r["passes_distill_guard"] = bool(float(r["healthy_kl"]) <= 0.03 and float(r["delta_healthy_nll"]) <= 0.04)

    guard_ok = [r for r in train_rows if r["passes_distill_guard"]]
    if guard_ok:
        win_train = max(guard_ok, key=lambda r: (r["teacher_logp_gain"], r["distill_pareto"]))
    else:
        nll_ok = [r for r in train_rows if r["passes_nll_guard"]]
        pool = nll_ok if nll_ok else train_rows
        win_train = max(pool, key=lambda r: (r["distill_pareto"], r["teacher_logp_gain"]))

    win_name = win_train["name"]
    win_patch = cands[win_name]

    test_rows = screen_patch_candidates(
        model,
        {win_name: win_patch},
        healthy_seqs=healthy_test,
        target_seqs=test_seqs,
        target_spans=test_spans,
        max_nll_delta=0.05,
    )
    win_test = test_rows[0]
    t_gain = -float(win_test["target_suppression"])
    win_test["teacher_logp_gain"] = round(t_gain, 6)
    win_test["distill_pareto"] = round(t_gain / max(float(win_test["healthy_kl"]), 0.005), 6)
    return win_name, win_patch, win_train, win_test


def _patch_summary_meta(patch: MoIPatch, tmp_dir: Path) -> dict[str, Any]:
    p_file = tmp_dir / f"{patch.meta.get('name', 'patch')}.moipatch"
    patch.save(p_file)
    size_kb = round(p_file.stat().st_size / 1024.0, 2)
    sites: list[str] = []
    per_mod_rank: dict[str, int] = {}
    for e in patch.edits:
        if e.module not in per_mod_rank:
            sites.append(e.module)
        per_mod_rank[e.module] = per_mod_rank.get(e.module, 0) + 1
    return {
        "n_edits": len(patch.edits),
        "sites": sites,
        "ranks": per_mod_rank,
        "alphas": sorted({float(e.alpha) for e in patch.edits}),
        "payload_kb": size_kb,
    }


# ------------------------------------------------------------------ Weight Composition, vLLM Swap & Bake Helpers
def compose_patch_weights(patch: MoIPatch | None, base_w: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Sequentially compose `patch.edits` per module in float64 using `wsi.moipatch.apply_edit`."""
    if patch is None or not patch.edits:
        return {}
    out: dict[str, torch.Tensor] = {}
    for e in patch.edits:
        if e.module not in base_w:
            raise KeyError(f"Missing base weight for module {e.module!r}")
        cur = out.get(e.module, base_w[e.module])
        out[e.module] = apply_edit(cur, e.u, e.idx, e.alpha, e.v)
    return out


def make_suffix_setter(updates_np: dict[str, np.ndarray]):
    """Self-contained closure for `llm.apply_model` matching arbitrary module suffixes (MLP, self_attn, linear_attn)."""

    def setter(model):
        import re as _re

        import numpy as _np  # noqa: F401
        import torch as _t

        attn_re = _re.compile(r"^layers\.(\d+)\.(?:linear_attn|self_attn)\.(?:out_proj|o_proj)$")
        all_params = [
            (name, p)
            for name, p in model.named_parameters()
            if "mtp" not in name and "visual" not in name
        ]
        done: dict[str, tuple[str, int]] = {}
        for mk, arr in updates_np.items():
            new = _t.from_numpy(arr)
            exact_hits = [
                (name, p)
                for name, p in all_params
                if name == f"{mk}.weight" or name.endswith(f".{mk}.weight")
            ]
            hits = exact_hits
            if not hits:
                m_att = attn_re.match(mk)
                if m_att is not None:
                    li = int(m_att.group(1))
                    fb_pat = _re.compile(rf"(^|\.)layers\.{li}\.[^.]+\.(o_proj|out_proj)\.weight$")
                    hits = [
                        (name, p)
                        for name, p in all_params
                        if fb_pat.search(name) and tuple(p.shape) == tuple(new.shape)
                    ]
            if len(hits) != 1:
                raise RuntimeError(f"{mk}: expected 1 matching weight parameter in worker, found {len(hits)}: {[h[0] for h in hits]}")
            pname, param = hits[0]
            if tuple(new.shape) != tuple(param.shape):
                raise RuntimeError(f"{mk} ({pname}): shape mismatch {tuple(param.shape)} vs {tuple(new.shape)}")
            param.data.copy_(new.to(param.dtype).to(param.device))
            cs = int(param.data.contiguous().view(_t.int16).long().sum().item())
            done[mk] = (str(param.dtype), cs)
        return done

    return setter


def _int16_checksum(t: torch.Tensor, dtype_str: str) -> int:
    dt = getattr(torch, dtype_str.replace("torch.", ""))
    return int(t.to(dt).contiguous().view(torch.int16).long().sum().item())


def _bake_patched_checkpoint(base_dir: Path, updates_t: dict[str, torch.Tensor], out_dir: Path) -> Path:
    """Create a fast patched checkpoint directory by symlinking unchanged files and rewriting modified safetensors."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in base_dir.iterdir():
        if entry.name.startswith(".") or entry.name.endswith(".lock") or entry.suffix == ".safetensors":
            continue
        dst = out_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst, ignore=shutil.ignore_patterns(".cache", "*.lock"))
        else:
            try:
                os.symlink(entry.resolve(), dst)
            except OSError:
                shutil.copy2(entry, dst)

    remaining = dict(updates_t)
    attn_re = _LAYER_ATTN_RE
    for sf in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(sf), "pt") as fh:
            keys = list(fh.keys())
            meta = fh.metadata() or {"format": "pt"}
            file_map: dict[str, str] = {}
            for mk in list(remaining):
                exact = [k for k in keys if "mtp" not in k and "visual" not in k and (k == f"{mk}.weight" or k.endswith(f".{mk}.weight"))]
                hits = exact
                if not hits:
                    m_att = attn_re.match(mk)
                    if m_att is not None:
                        li = int(m_att.group(1))
                        fb = re.compile(rf"(^|\.)layers\.{li}\.[^.]+\.(o_proj|out_proj)\.weight$")
                        shape_mk = tuple(remaining[mk].shape)
                        hits = [k for k in keys if "mtp" not in k and "visual" not in k and fb.search(k) and tuple(fh.get_tensor(k).shape) == shape_mk]
                if len(hits) == 1:
                    file_map[mk] = hits[0]
            if not file_map:
                dst_sf = out_dir / sf.name
                try:
                    os.symlink(sf.resolve(), dst_sf)
                except OSError:
                    shutil.copy2(sf, dst_sf)
                continue
            tensors = {k: fh.get_tensor(k).clone() for k in keys}
        for mk, k in file_map.items():
            tensors[k] = remaining.pop(mk).to(tensors[k].dtype).cpu()
        save_file(tensors, str(out_dir / sf.name), metadata=meta)

    if remaining:
        raise RuntimeError(f"Failed to locate safetensors keys when baking checkpoint: {sorted(remaining)}")
    return out_dir


class _VLLMSwapRunner:
    """Single-engine vLLM runner supporting exact checksummed in-memory weight swaps + baked checkpoint fallback."""

    def __init__(
        self,
        *,
        model_dir: str,
        tok: Any,
        base_w: dict[str, torch.Tensor],
        max_len: int,
        dtype_str: str,
        seed: int,
        gpu_mem: float = 0.88,
        bake_tmp_dir: str = "/content/_v2_baked_tmp",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.tok = tok
        self.base_w = base_w
        self.max_len = max_len
        self.dtype_str = dtype_str
        self.eos_list = lr.eos_ids(tok)
        self.cfg = {"base_dir": str(self.model_dir), "seed": int(seed), "eos": self.eos_list, "sampling": lr.SAMPLING}
        self.gpu_mem = float(gpu_mem)
        self.bake_dir = Path(bake_tmp_dir)
        self.llm, self.kill = dm._llm_fast(str(self.model_dir), self.cfg, self.max_len, self.dtype_str, self.gpu_mem)
        self.edited_now: set[str] = set()
        self.bake_mode = False
        self.loaded_dir = str(self.model_dir)

    def _reload_vllm(self, target_dir: str) -> None:
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.llm, self.kill = dm._llm_fast(target_dir, self.cfg, self.max_len, self.dtype_str, self.gpu_mem)
        self.loaded_dir = target_dir

    def set_patch(self, patch: MoIPatch | None) -> None:
        updates_t = compose_patch_weights(patch, self.base_w)
        target_set = set(updates_t)
        if not self.bake_mode:
            if not target_set and not self.edited_now:
                return
            swap_t = dict(updates_t)
            for mk in self.edited_now - target_set:
                swap_t[mk] = self.base_w[mk]
            updates_np = {mk: t.detach().cpu().to(torch.float32).numpy() for mk, t in swap_t.items()}
            try:
                got: dict[str, tuple[str, int]] = {}
                for r in self.llm.apply_model(make_suffix_setter(updates_np)):
                    got.update(r)
                if set(got) != set(updates_np):
                    raise RuntimeError(f"Weight swap missed modules: {sorted(set(updates_np) - set(got))}")
                for mk, (dt_s, cs) in got.items():
                    want = _int16_checksum(swap_t[mk], dt_s)
                    if cs != want:
                        raise RuntimeError(f"Checksum mismatch at {mk}: {cs} != {want}")
                self.llm.reset_prefix_cache()
                self.edited_now = target_set
                return
            except Exception as err:  # noqa: BLE001
                print(
                    f"[vllm-swap] In-memory swap failed ({type(err).__name__}: {str(err)[:260]}); "
                    f"falling back to baked checkpoint reload.",
                    flush=True,
                )
                self.bake_mode = True
                self.loaded_dir = ""

        # Baked checkpoint fallback
        if not target_set:
            want_dir = str(self.model_dir)
            if self.loaded_dir != want_dir:
                self._reload_vllm(want_dir)
        else:
            _bake_patched_checkpoint(self.model_dir, updates_t, self.bake_dir)
            self._reload_vllm(str(self.bake_dir))
        self.edited_now = target_set

    def generate_batch(self, prompts_ids: list[list[int]], seeds: list[int], max_new_tokens: int) -> list[dict[str, Any]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        kill_eos = self.eos_list[-1]

        def _make_sp(seed_val: int, use_kill: bool) -> Any:
            return SamplingParams(
                **lr.SAMPLING,
                max_tokens=max_new_tokens,
                seed=int(seed_val),
                stop_token_ids=self.eos_list,
                extra_args={"loop_kill_eos": kill_eos} if use_kill else None,
            )

        tok_prompts = [TokensPrompt(prompt_token_ids=pids) for pids in prompts_ids]
        try:
            outs = self.llm.generate(tok_prompts, [_make_sp(s, self.kill) for s in seeds], use_tqdm=False)
        except Exception as err:  # noqa: BLE001
            if not self.kill:
                raise
            print(f"[vllm] early loop stop failed ({type(err).__name__}: {str(err)[:220]}); retrying without it", flush=True)
            self.kill = False
            outs = self.llm.generate(tok_prompts, [_make_sp(s, False) for s in seeds], use_tqdm=False)
        return [
            {"token_ids": list(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason}
            for o in outs
        ]

    def close(self) -> None:
        if self.bake_dir.exists():
            shutil.rmtree(self.bake_dir, ignore_errors=True)
        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _HFSwapRunner:
    """HF `PatchHandle` fallback runner when `vllm` is unavailable or `--backend hf` is requested."""

    def __init__(self, model: nn.Module, tok: Any, *, batch_size: int = 16) -> None:
        self.model = model
        self.tok = tok
        self.batch_size = batch_size
        self.active_handle = None
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.eos_list = lr.eos_ids(tok)

    def set_patch(self, patch: MoIPatch | None) -> None:
        if self.active_handle is not None:
            self.active_handle.off()
            self.active_handle = None
        if patch is not None and patch.edits:
            self.active_handle = patch.attach(self.model, verify_fingerprint=False)
            self.active_handle.on()

    @torch.no_grad()
    def generate_batch(self, prompts_ids: list[list[int]], seeds: list[int], max_new_tokens: int) -> list[dict[str, Any]]:
        dev = next(self.model.parameters()).device
        pad_id = int(self.tok.pad_token_id)
        out_rows: list[dict[str, Any]] = []
        # Run 1-by-1 if batch_size==1, or batched with per-batch seed if larger (prefer strict per-item seed!)
        for pids, s_val in zip(prompts_ids, seeds):
            ids_t = torch.tensor(pids, dtype=torch.long, device=dev).unsqueeze(0)
            torch.manual_seed(int(s_val))
            gen = self.model.generate(
                input_ids=ids_t,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=lr.SAMPLING["temperature"],
                top_p=lr.SAMPLING["top_p"],
                top_k=lr.SAMPLING["top_k"],
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=self.eos_list,
            )
            gen_ids = [int(x) for x in gen[0, len(pids) :].tolist()]
            body = lr.strip_eos(gen_ids, set(self.eos_list))
            fin = "length" if (len(gen_ids) >= max_new_tokens and len(body) == len(gen_ids)) else "stop"
            out_rows.append({"token_ids": gen_ids, "finish_reason": fin})
        return out_rows

    def close(self) -> None:
        self.set_patch(None)


def _summarize_arm(
    rows: list[dict[str, Any]],
    *,
    prompt_lengths: list[int],
) -> dict[str, Any]:
    n = max(1, len(rows))
    renum = [r for r in rows if r["variant"] == "renumbered"]
    rephr = [r for r in rows if r["variant"] == "rephrased_renumbered"]
    toks = [r["total_tokens"] for r in rows]
    thinks = [r["think_tokens"] for r in rows]
    per_fam: dict[str, float] = {}
    for fam in FAMILIES:
        sub = [r for r in rows if r["family"] == fam]
        per_fam[fam] = round(sum(1 for r in sub if r["strict_correct"]) / max(1, len(sub)), 4)

    return {
        "strict_accuracy": round(sum(1 for r in rows if r["strict_correct"]) / n, 4),
        "strict_acc_renumbered": round(sum(1 for r in renum if r["strict_correct"]) / max(1, len(renum)), 4),
        "strict_acc_rephrased_renumbered": round(sum(1 for r in rephr if r["strict_correct"]) / max(1, len(rephr)), 4),
        "rescued_accuracy": round(sum(1 for r in rows if r["rescued_correct"]) / n, 4),
        "closed_think_rate": round(sum(1 for r in rows if r["closed_think"]) / n, 4),
        "has_boxed_rate": round(sum(1 for r in rows if r["has_boxed"]) / n, 4),
        "loop_killed_rate": round(sum(1 for r in rows if r["loop_killed"]) / n, 4),
        "truncated_rate": round(sum(1 for r in rows if r["truncated"]) / n, 4),
        "mean_tokens": round(sum(toks) / n, 1),
        "median_tokens": round(float(statistics.median(toks)), 1) if toks else 0.0,
        "mean_think_tokens": round(sum(thinks) / n, 1),
        "mean_prompt_tokens": round(sum(prompt_lengths) / max(1, len(prompt_lengths)), 1),
        "n_strict_correct": sum(1 for r in rows if r["strict_correct"]),
        "n_rescued_correct": sum(1 for r in rows if r["rescued_correct"]),
        "n_total": len(rows),
        "per_family_strict": per_fam,
    }


def _paired_vs(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], key: str = "strict_correct") -> dict[str, int]:
    wins = sum(1 for a, b in zip(a_rows, b_rows) if a[key] and not b[key])
    losses = sum(1 for a, b in zip(a_rows, b_rows) if (not a[key]) and b[key])
    both = sum(1 for a, b in zip(a_rows, b_rows) if a[key] and b[key])
    return {"wins": wins, "losses": losses, "both_correct": both, "net": wins - losses}


# ------------------------------------------------------------------ CPU Smoke Test
def run_smoke() -> dict[str, Any]:
    """Deterministic CPU verification of multi-rank PatchHandle composition, suffix swaps, and grading."""
    model = _PlantedReasoningLM().eval()
    fam_a_train = [[1, 2, 4, num_t, 20, 20, 20, 20] for num_t in (10, 11, 12, 13)]
    fam_b_train = [[1, 2, 5, num_t, 21, 21, 21, 21] for num_t in (14, 15, 16, 17)]

    traces_a = [
        capture_causal_trace(
            model, s, target_span=(5, 8), healthy_span=(0, 2),
            grad_mode="compressed", center="healthy", healthy_cost="fisher",
            top_k_mlp=16, group_size=16, hadamard_block=16, top_k_attn=0, capture_cache=False,
        )
        for s in fam_a_train
    ]
    traces_b = [
        capture_causal_trace(
            model, s, target_span=(5, 8), healthy_span=(0, 2),
            grad_mode="compressed", center="healthy", healthy_cost="fisher",
            top_k_mlp=16, group_size=16, hadamard_block=16, top_k_attn=0, capture_cache=False,
        )
        for s in fam_b_train
    ]

    inv_metrics = compute_invariance_metrics({"fam_a": traces_a, "fam_b": traces_b}, top_k_sites=1, top_n_neurons=2)
    mod_meta = _collect_mod_meta(model)

    grid_a = build_multirank_grid(traces_a, mod_meta, prefix="fam_a", model_or_weights=model)
    assert grid_a["single_site_r1"] and grid_a["multi_site_mr"], "Empty multi-rank grids in smoke test"

    # Build explicit rank-2 edit on the same module to verify multi-rank PatchHandle composition & make_suffix_setter
    mk = grid_a["sites"][0]
    w_orig = model.model.layers[int(mk.split(".")[1])].mlp.down_proj.weight.detach().clone() if ".mlp." in mk else next(
        m.weight.detach().clone() for n, m in model.named_modules() if n.endswith(mk)
    )
    q, _ = torch.linalg.qr(torch.randn(w_orig.shape[0], 2))
    p_mr = MoIPatch(
        edits=[
            Edit(module=mk, u=q[:, 0].double(), alpha=-2.0, **mod_meta[mk]),
            Edit(module=mk, u=q[:, 1].double(), alpha=-3.0, **mod_meta[mk]),
        ],
        meta={"name": "smoke_mr"},
    )
    base_w = {k: m.weight.detach().cpu().clone() for k in [mk] for n, m in model.named_modules() if n.endswith(k)}
    composed = compose_patch_weights(p_mr, base_w)
    expected_w = apply_edit(apply_edit(w_orig, q[:, 0].double(), None, -2.0), q[:, 1].double(), None, -3.0)
    assert torch.equal(composed[mk], expected_w)

    # Check PatchHandle matches composed weights and cleanly restores
    with p_mr.attach(model) as ph:
        ph.on()
        cur_w = next(m.weight for n, m in model.named_modules() if n.endswith(mk))
        assert torch.equal(cur_w, expected_w)
    assert ph.check_restored()

    # Check make_suffix_setter & int16 checksum verification on toy PyTorch model
    upd_np = {mk: composed[mk].to(torch.float32).numpy()}
    got_cs = make_suffix_setter(upd_np)(model)
    assert got_cs[mk][1] == _int16_checksum(composed[mk], got_cs[mk][0])
    # Restore base weights
    make_suffix_setter({mk: base_w[mk].to(torch.float32).numpy()})(model)

    # Check random matched orthogonal construction
    rnd_p = make_random_matched_patch(p_mr, name="smoke_rnd", seed=7)
    for er, eorig in zip(rnd_p.edits, p_mr.edits):
        assert abs(float((er.u.float() @ eorig.u.float()).item())) < 1e-4

    # Check grade_response across strict, rescued, loop-killed, truncated, and no_think
    g_strict = grade_response([10, 20, 30, 40, 50], "thinking...\n</think>\nThus \\boxed{42}.", 42, thinking_enabled=True, finish_reason="stop")
    assert g_strict["strict_correct"] and g_strict["closed_think"] and g_strict["has_boxed"] and g_strict["rescued_correct"]

    g_unclosed = grade_response([10] * 50, "<think>\nso x = 42", 42, thinking_enabled=True, finish_reason="length")
    assert (not g_unclosed["strict_correct"]) and g_unclosed["truncated"] and (not g_unclosed["closed_think"]) and g_unclosed["rescued_correct"]

    loop_pat = list(range(100, 120))
    loop_ids = list(range(1000, 1200)) + loop_pat * 45  # triggers KILL_WINDOW=768, KILL_FRAC=0.9 after 200-token prefix
    g_loop = grade_response(loop_ids, "answer is 42 " + "blah " * 200, 42, thinking_enabled=True, finish_reason="stop")
    assert g_loop["loop_killed"] and (not g_loop["strict_correct"]) and g_loop["rescued_correct"]

    g_nothink = grade_response([1, 2, 3], "The answer is \\boxed{42}.", 42, thinking_enabled=False, finish_reason="stop")
    assert g_nothink["strict_correct"] and g_nothink["think_tokens"] == 0

    return {
        "mode": "smoke",
        "passed": True,
        "invariance": inv_metrics,
        "patch_compose_ok": True,
        "suffix_setter_checksum_ok": True,
        "random_orthogonal_ok": True,
        "grading_checks_ok": True,
    }


# ------------------------------------------------------------------ Full GPU Pipeline
def _select_backend(requested: str) -> str:
    if requested in ("vllm", "hf"):
        return requested
    try:
        import vllm  # noqa: F401
        if torch.cuda.is_available():
            return "vllm"
    except Exception:
        pass
    return "hf"


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        args.dtype, torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float32
    )
    dtype_vllm_str = "bfloat16" if dtype == torch.bfloat16 else ("float16" if dtype == torch.float16 else "float32")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = build_reasoning_dataset()
    train_items = ds["train"]
    test_items = ds["test"]
    healthy_texts = ds["healthy_controls"]
    mod_meta = _collect_mod_meta(model)

    # Phase 1: Capture Causal CompactTraces on the 16 teacher-forced train examples
    print(f"[v2] Phase 1: Capturing causal CompactTraces on {len(train_items)} train traces...", flush=True)
    t_cap = time.perf_counter()
    family_traces: dict[str, list[CompactTrace]] = {f: [] for f in FAMILIES}
    family_train_seqs: dict[str, list[list[int]]] = {f: [] for f in FAMILIES}
    family_train_spans: dict[str, list[tuple[int, int]]] = {f: [] for f in FAMILIES}

    for item in train_items:
        q_fmt = format_question(item["question"])
        p_ids = _format_messages(tok, [{"role": "user", "content": q_fmt}], add_generation_prompt=True, enable_thinking=True)
        full_ids = _format_messages(
            tok,
            [{"role": "user", "content": q_fmt}, {"role": "assistant", "content": item["teacher_response"]}],
            add_generation_prompt=False,
            enable_thinking=True,
        )
        t_start = max(2, len(p_ids))
        h_span = (0, max(1, t_start - 1))
        t_span = (t_start, len(full_ids))
        ct = capture_causal_trace(
            model,
            full_ids,
            target_span=t_span,
            healthy_span=h_span,
            grad_mode="compressed",
            grad_checkpointing=True,
            center="healthy",
            healthy_cost="fisher",
            top_k_attn=0,
            capture_cache=False,
        )
        fam = item["family"]
        family_traces[fam].append(ct)
        family_train_seqs[fam].append(full_ids)
        family_train_spans[fam].append(t_span)

    capture_s = round(time.perf_counter() - t_cap, 2)
    invariance = compute_invariance_metrics(family_traces, top_k_sites=4, top_n_neurons=32)
    print(f"[v2] Invariance ({capture_s}s): {invariance}", flush=True)

    # Healthy sequences and held-out test teacher sequences
    healthy_seqs = [
        _format_messages(
            tok,
            [{"role": "user", "content": h_txt}, {"role": "assistant", "content": "Here is a clear and concise explanation."}],
            add_generation_prompt=False,
            enable_thinking=True,
        )
        for h_txt in healthy_texts[:12]
    ]
    family_test_seqs: dict[str, list[list[int]]] = {f: [] for f in FAMILIES}
    family_test_spans: dict[str, list[tuple[int, int]]] = {f: [] for f in FAMILIES}
    for item in test_items:
        q_fmt = format_question(item["question"])
        p_ids = _format_messages(tok, [{"role": "user", "content": q_fmt}], add_generation_prompt=True, enable_thinking=True)
        full_ids = _format_messages(
            tok,
            [{"role": "user", "content": q_fmt}, {"role": "assistant", "content": item["teacher_response"]}],
            add_generation_prompt=False,
            enable_thinking=True,
        )
        family_test_seqs[item["family"]].append(full_ids)
        family_test_spans[item["family"]].append((max(2, len(p_ids)), len(full_ids)))

    # Phase 2: Build & screen `single_site_r1`, `multi_site_mr`, `random_matched`, and `global_distilled`
    print("[v2] Phase 2: Building & screening multi-site multi-rank .moipatch experts...", flush=True)
    t_scr = time.perf_counter()
    s1_experts: dict[str, MoIPatch] = {}
    mr_experts: dict[str, MoIPatch] = {}
    rnd_experts: dict[str, MoIPatch] = {}
    family_distill_summary: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmp_patch_dir_str:
        tmp_patch_dir = Path(tmp_patch_dir_str)
        for fam_idx, fam in enumerate(FAMILIES):
            grid_info = build_multirank_grid(
                family_traces[fam], mod_meta, prefix=fam, model_or_weights=model, global_mode=False
            )
            w_s1_nm, p_s1, tr_s1, te_s1 = _select_and_eval_distill(
                model,
                grid_info["single_site_r1"],
                train_seqs=family_train_seqs[fam],
                train_spans=family_train_spans[fam],
                test_seqs=family_test_seqs[fam][:6],
                test_spans=family_test_spans[fam][:6],
                healthy_train=healthy_seqs[:8],
                healthy_test=healthy_seqs[8:],
            )
            w_mr_nm, p_mr, tr_mr, te_mr = _select_and_eval_distill(
                model,
                grid_info["multi_site_mr"],
                train_seqs=family_train_seqs[fam],
                train_spans=family_train_spans[fam],
                test_seqs=family_test_seqs[fam][:6],
                test_spans=family_test_spans[fam][:6],
                healthy_train=healthy_seqs[:8],
                healthy_test=healthy_seqs[8:],
            )
            p_rnd = make_random_matched_patch(p_mr, name=f"{fam}_rnd_matched", seed=500 + fam_idx)
            _, _, _, te_rnd = _select_and_eval_distill(
                model,
                {f"{fam}_rnd_matched": p_rnd},
                train_seqs=family_train_seqs[fam][:2],
                train_spans=family_train_spans[fam][:2],
                test_seqs=family_test_seqs[fam][:6],
                test_spans=family_test_spans[fam][:6],
                healthy_train=healthy_seqs[:8],
                healthy_test=healthy_seqs[8:],
            )

            s1_experts[fam] = p_s1
            mr_experts[fam] = p_mr
            rnd_experts[fam] = p_rnd

            s1_meta = _patch_summary_meta(p_s1, tmp_patch_dir)
            mr_meta = _patch_summary_meta(p_mr, tmp_patch_dir)

            family_distill_summary[fam] = {
                "causal_top_sites": grid_info["sites"],
                "single_site_r1": {"config": w_s1_nm, **s1_meta, "train": tr_s1, "heldout_test": te_s1},
                "multi_site_mr": {"config": w_mr_nm, **mr_meta, "train": tr_mr, "heldout_test": te_mr},
                "random_matched": {"heldout_test": te_rnd},
            }
            print(
                f"[v2] {fam:22s} | s1 gain={te_s1['teacher_logp_gain']:+.4f} kl={te_s1['healthy_kl']:.4f} ({s1_meta['payload_kb']}KB) | "
                f"mr gain={te_mr['teacher_logp_gain']:+.4f} kl={te_mr['healthy_kl']:.4f} ({mr_meta['payload_kb']}KB, {w_mr_nm}) | "
                f"rnd gain={te_rnd['teacher_logp_gain']:+.4f}",
                flush=True,
            )

        # Global distilled patch (pooled across all 16 traces = 4 families)
        all_traces = [ct for fam in FAMILIES for ct in family_traces[fam]]
        glob_train_seqs = [s for fam in FAMILIES for s in family_train_seqs[fam][:2]]
        glob_train_spans = [sp for fam in FAMILIES for sp in family_train_spans[fam][:2]]
        glob_test_seqs = [s for fam in FAMILIES for s in family_test_seqs[fam][:2]]
        glob_test_spans = [sp for fam in FAMILIES for sp in family_test_spans[fam][:2]]
        glob_grid = build_multirank_grid(
            all_traces, mod_meta, prefix="global", model_or_weights=model, global_mode=True
        )
        w_glob_nm, global_patch, tr_glob, te_glob = _select_and_eval_distill(
            model,
            glob_grid["multi_site_mr"],
            train_seqs=glob_train_seqs,
            train_spans=glob_train_spans,
            test_seqs=glob_test_seqs,
            test_spans=glob_test_spans,
            healthy_train=healthy_seqs[:8],
            healthy_test=healthy_seqs[8:],
        )
        global_summary = {
            "config": w_glob_nm,
            **_patch_summary_meta(global_patch, tmp_patch_dir),
            "train": tr_glob,
            "heldout_test": te_glob,
        }
        print(
            f"[v2] global_distilled       | gain={te_glob['teacher_logp_gain']:+.4f} kl={te_glob['healthy_kl']:.4f} "
            f"({global_summary['payload_kb']}KB, {w_glob_nm})",
            flush=True,
        )

    screen_s = round(time.perf_counter() - t_scr, 2)

    # Phase 3: Zero-generation Prefill Router (layer 12) + precompute all prompt IDs
    _, blocks = find_blocks(model)
    router_layer = len(blocks) // 2
    train_prompt_ids = [
        _format_messages(tok, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=True)
        for it in train_items
    ]
    test_prompt_ids_think = [
        _format_messages(tok, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=True)
        for it in test_items
    ]
    test_prompt_ids_nothink = [
        _format_messages(tok, [{"role": "user", "content": format_question(it["question"])}], add_generation_prompt=True, enable_thinking=False)
        for it in test_items
    ]
    train_by_fam = {fam: [it for it in train_items if it["family"] == fam] for fam in FAMILIES}
    test_prompt_ids_1shot: list[list[int]] = []
    for it in test_items:
        ex = train_by_fam[it["family"]][0]
        msgs = [
            {"role": "user", "content": format_question(ex["question"])},
            {"role": "assistant", "content": ex["teacher_response"]},
            {"role": "user", "content": format_question(it["question"])},
        ]
        test_prompt_ids_1shot.append(
            _format_messages(tok, msgs, add_generation_prompt=True, enable_thinking=True)
        )

    train_vecs = _extract_prompt_hidden_vecs(model, train_prompt_ids, router_layer)
    test_vecs = _extract_prompt_hidden_vecs(model, test_prompt_ids_think, router_layer)
    centroids = {}
    for fam in FAMILIES:
        idxs = [i for i, it in enumerate(train_items) if it["family"] == fam]
        c = train_vecs[idxs].mean(dim=0)
        centroids[fam] = c / c.norm().clamp_min(1e-12)

    routed_assignments: list[str] = []
    r_ok = r_renum = r_rephr = n_renum = n_rephr = 0
    for i, it in enumerate(test_items):
        pred_fam = max(FAMILIES, key=lambda f: float((test_vecs[i] @ centroids[f]).item()))
        routed_assignments.append(pred_fam)
        ok = pred_fam == it["family"]
        r_ok += int(ok)
        if it["variant"] == "renumbered":
            n_renum += 1
            r_renum += int(ok)
        else:
            n_rephr += 1
            r_rephr += int(ok)

    router_report = {
        "router_layer": router_layer,
        "accuracy_all": round(r_ok / max(1, len(test_items)), 4),
        "accuracy_renumbered": round(r_renum / max(1, n_renum), 4),
        "accuracy_rephrased_renumbered": round(r_rephr / max(1, n_rephr), 4),
    }
    print(f"[v2] Prefill Router (layer {router_layer}): {router_report}", flush=True)

    # Collect CPU base weights for all modules touched by any patch before deciding whether to free HF model for vLLM
    all_patches = [*s1_experts.values(), *mr_experts.values(), *rnd_experts.values(), global_patch]
    needed_mods = sorted({e.module for p in all_patches for e in p.edits})
    mods_map = dict(model.named_modules())
    base_w: dict[str, torch.Tensor] = {}
    for mk in needed_mods:
        found = [m for name, m in mods_map.items() if "mtp" not in name and "visual" not in name and hasattr(m, "weight") and (name == mk or name.endswith(f".{mk}"))]
        if len(found) != 1:
            raise KeyError(f"Base module {mk!r}: found {len(found)} matches")
        base_w[mk] = found[0].weight.detach().cpu().clone()

    backend = _select_backend(args.backend)
    print(f"[v2] Phase 4: Starting generation ({backend}, max_new_tokens={args.max_new_tokens}) across 7 paired arms...", flush=True)
    max_prompt_len = max(
        len(p) for p in (*test_prompt_ids_think, *test_prompt_ids_nothink, *test_prompt_ids_1shot)
    )
    if backend == "vllm":
        del model, mods_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        runner: Any = _VLLMSwapRunner(
            model_dir=args.model,
            tok=tok,
            base_w=base_w,
            max_len=max_prompt_len + int(args.max_new_tokens) + 64,
            dtype_str=dtype_vllm_str,
            seed=args.seed,
        )
    else:
        runner = _HFSwapRunner(model, tok)

    eos_set = set(lr.eos_ids(tok))
    think_end_tok_id = tok.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_tok_id, int) or think_end_tok_id == tok.unk_token_id:
        think_end_tok_id = None

    seeds_all = [int(args.seed) + i for i in range(len(test_items))]
    t_gen = time.perf_counter()

    def _run_arm(
        arm_name: str,
        *,
        prompts_ids: list[list[int]],
        thinking_enabled: bool,
        patch: MoIPatch | None = None,
        routed_dict: dict[str, MoIPatch] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        t_a = time.perf_counter()
        raw_by_idx: dict[int, dict[str, Any]] = {}
        if routed_dict is not None:
            for fam in FAMILIES:
                idxs = [i for i, pred_f in enumerate(routed_assignments) if pred_f == fam]
                if not idxs:
                    continue
                runner.set_patch(routed_dict[fam])
                outs = runner.generate_batch([prompts_ids[i] for i in idxs], [seeds_all[i] for i in idxs], args.max_new_tokens)
                for idx, o in zip(idxs, outs):
                    raw_by_idx[idx] = o
        else:
            runner.set_patch(patch)
            outs = runner.generate_batch(prompts_ids, seeds_all, args.max_new_tokens)
            for idx, o in enumerate(outs):
                raw_by_idx[idx] = o

        graded_rows: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        for idx, it in enumerate(test_items):
            raw = raw_by_idx[idx]
            body = lr.strip_eos(raw["token_ids"], eos_set)
            txt = tok.decode(body, skip_special_tokens=False)
            g = grade_response(
                raw["token_ids"],
                txt,
                int(it["answer"]),
                thinking_enabled=thinking_enabled,
                finish_reason=raw["finish_reason"],
                eos_set=eos_set,
                tok=tok,
                think_end_id=think_end_tok_id,
            )
            graded_rows.append({
                "id": it["id"],
                "family": it["family"],
                "variant": it["variant"],
                "routed_expert": routed_assignments[idx] if routed_dict is not None else None,
                **g,
            })
            if idx < 3:
                samples.append({
                    "id": it["id"],
                    "gold": int(it["answer"]),
                    "strict_pred": g["strict_pred"],
                    "strict_correct": g["strict_correct"],
                    "total_tokens": g["total_tokens"],
                    "text_head": txt[:1500],
                })
        summary = _summarize_arm(graded_rows, prompt_lengths=[len(p) for p in prompts_ids])
        summary["elapsed_s"] = round(time.perf_counter() - t_a, 1)
        print(
            f"[v2] Arm {arm_name:23s} | strict={summary['strict_accuracy']:.4f} | rescued={summary['rescued_accuracy']:.4f} | "
            f"closed={summary['closed_think_rate']:.3f} | loop={summary['loop_killed_rate']:.3f} | "
            f"med_tok={summary['median_tokens']:.0f} | mean_tok={summary['mean_tokens']:.1f} ({summary['elapsed_s']}s)",
            flush=True,
        )
        return summary, graded_rows, samples

    arm_summaries: dict[str, dict[str, Any]] = {}
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_samples: dict[str, list[dict[str, Any]]] = {}

    arm_specs: list[tuple[str, dict[str, Any]]] = [
        ("base", {"prompts_ids": test_prompt_ids_think, "thinking_enabled": True}),
        ("base_no_think", {"prompts_ids": test_prompt_ids_nothink, "thinking_enabled": False}),
        ("oracle_1shot", {"prompts_ids": test_prompt_ids_1shot, "thinking_enabled": True}),
        ("random_matched_routed", {"prompts_ids": test_prompt_ids_think, "thinking_enabled": True, "routed_dict": rnd_experts}),
        ("global_distilled", {"prompts_ids": test_prompt_ids_think, "thinking_enabled": True, "patch": global_patch}),
        ("single_site_r1_routed", {"prompts_ids": test_prompt_ids_think, "thinking_enabled": True, "routed_dict": s1_experts}),
        ("multi_site_mr_routed", {"prompts_ids": test_prompt_ids_think, "thinking_enabled": True, "routed_dict": mr_experts}),
    ]
    try:
        for nm, kw in arm_specs:
            sm, rw, smp = _run_arm(nm, **kw)
            arm_summaries[nm] = sm
            arm_rows[nm] = rw
            arm_samples[nm] = smp
    finally:
        runner.close()

    gen_s = round(time.perf_counter() - t_gen, 2)

    # Aggregate forward metrics & paired comparisons
    mean_s1_gain = sum(family_distill_summary[f]["single_site_r1"]["heldout_test"]["teacher_logp_gain"] for f in FAMILIES) / len(FAMILIES)
    mean_mr_gain = sum(family_distill_summary[f]["multi_site_mr"]["heldout_test"]["teacher_logp_gain"] for f in FAMILIES) / len(FAMILIES)
    mean_rnd_gain = sum(family_distill_summary[f]["random_matched"]["heldout_test"]["teacher_logp_gain"] for f in FAMILIES) / len(FAMILIES)
    mean_mr_kl = sum(family_distill_summary[f]["multi_site_mr"]["heldout_test"]["healthy_kl"] for f in FAMILIES) / len(FAMILIES)

    acc_base = arm_summaries["base"]["strict_accuracy"]
    acc_oracle = arm_summaries["oracle_1shot"]["strict_accuracy"]
    acc_rnd = arm_summaries["random_matched_routed"]["strict_accuracy"]
    acc_glob = arm_summaries["global_distilled"]["strict_accuracy"]
    acc_mr = arm_summaries["multi_site_mr_routed"]["strict_accuracy"]
    oracle_gap_rec = (
        round((acc_mr - acc_base) / max(acc_oracle - acc_base, 1e-6), 4)
        if acc_oracle > acc_base
        else None
    )

    crit_a = bool(mean_mr_gain > mean_s1_gain and mean_mr_gain > mean_rnd_gain and mean_mr_kl <= 0.03)
    crit_b = bool(
        (acc_mr > acc_base or (acc_mr >= acc_base and arm_summaries["multi_site_mr_routed"]["mean_tokens"] <= 0.85 * arm_summaries["base"]["mean_tokens"]))
        and acc_mr >= acc_rnd
    )

    return {
        "model": args.model,
        "dtype": str(dtype),
        "backend": backend,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "max_new_tokens": int(args.max_new_tokens),
        "timing_s": {
            "capture_s": capture_s,
            "screen_s": screen_s,
            "generation_s": gen_s,
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "invariance": invariance,
        "router_report": router_report,
        "forward_distill": {
            "mean_single_site_r1_gain": round(mean_s1_gain, 6),
            "mean_multi_site_mr_gain": round(mean_mr_gain, 6),
            "mean_random_matched_gain": round(mean_rnd_gain, 6),
            "mean_multi_site_mr_healthy_kl": round(mean_mr_kl, 6),
            "per_family": family_distill_summary,
            "global_distilled": global_summary,
        },
        "generation_arms": arm_summaries,
        "paired_comparisons": {
            "multi_site_mr_vs_base_strict": _paired_vs(arm_rows["multi_site_mr_routed"], arm_rows["base"], "strict_correct"),
            "multi_site_mr_vs_random_strict": _paired_vs(arm_rows["multi_site_mr_routed"], arm_rows["random_matched_routed"], "strict_correct"),
            "multi_site_mr_vs_global_strict": _paired_vs(arm_rows["multi_site_mr_routed"], arm_rows["global_distilled"], "strict_correct"),
            "single_site_r1_vs_base_strict": _paired_vs(arm_rows["single_site_r1_routed"], arm_rows["base"], "strict_correct"),
            "oracle_1shot_vs_base_strict": _paired_vs(arm_rows["oracle_1shot"], arm_rows["base"], "strict_correct"),
            "oracle_extra_prompt_tokens": round(
                arm_summaries["oracle_1shot"]["mean_prompt_tokens"] - arm_summaries["base"]["mean_prompt_tokens"], 1
            ),
            "oracle_gap_recovered": oracle_gap_rec,
        },
        "criteria": {
            "a_forward_scaling_passed": crit_a,
            "b_strict_acc_or_compression_passed": crit_b,
            "c_routed_vs_global_summary": {
                "acc_routed": acc_mr,
                "acc_global": acc_glob,
                "routed_minus_global_pts": round((acc_mr - acc_glob) * 100.0, 2),
                "median_tokens_routed": arm_summaries["multi_site_mr_routed"]["median_tokens"],
                "median_tokens_global": arm_summaries["global_distilled"]["median_tokens"],
                "oracle_acc": acc_oracle,
                "base_acc": acc_base,
                "oracle_gap_recovered": oracle_gap_rec,
            },
            "all_passed": bool(crit_a and crit_b),
        },
        "sample_outputs": {
            "base": arm_samples["base"],
            "single_site_r1_routed": arm_samples["single_site_r1_routed"],
            "multi_site_mr_routed": arm_samples["multi_site_mr_routed"],
        },
        "per_item_generation": arm_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run fast deterministic CPU verification")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
