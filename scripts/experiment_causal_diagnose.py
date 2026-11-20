"""Verify Causal Forward+Backward Trace Capture, Offline Causal Diagnosis, and Offline Screening.

Modes:
  1. ``--smoke``: Fast CPU check on a planted 4-layer causal LM with a Layer-1 nullspace symptom
     decoy and a Layer-2 true causal switch (tests both MLP and Attention planted circuits),
     verifying ``diagnose_causal_traces`` beats passive norms and ``screen_patches_from_traces``
     matches ``gradient_screen_candidates`` offline.
  2. Full GPU benchmark (e.g. ``--model Qwen/Qwen3.5-0.8B --n-seqs 8``): captures causal
     ``CompactTrace``s on looping GSM8K traces (or loads cached ``--from-outputs``), compares
     unified MLP+Attention causal diagnosis vs passive ranking (checking L15/L19 MLP and L15 Attn),
     and benchmarks offline ``screen_patches_from_traces`` vs online ``gradient_screen_candidates``
     and ``screen_patch_candidates``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from wsi import (  # noqa: E402
    Edit,
    MoIPatch,
    capture_causal_trace,
    diagnose_causal_traces,
    gradient_screen_candidates,
    screen_patch_candidates,
    screen_patches_from_traces,
)
from wsi.capture import find_mlps  # noqa: E402
from wsi.moipatch import _dtype_name, _sha256, module_suffix  # noqa: E402


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, dst)
    return dst


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0

    def _ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i + 1
            while j < n and vals[order[j]] == vals[order[i]]:
                j += 1
            avg = 0.5 * (i + j - 1)
            for k in range(i, j):
                out[order[k]] = avg
            i = j
        return out

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / max(dx * dy, 1e-12), 4)


# ------------------------------------------------------------------ CPU smoke planted model
class _PlantedAttn(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.num_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        nh, hd = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (hd**0.5)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = torch.softmax(scores + mask, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, H)
        return self.o_proj(ctx)


class _PlantedBlock(nn.Module):
    def __init__(self, hidden: int, inter: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = _PlantedAttn(hidden, n_heads)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x)
        h = F.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        x = x + self.mlp.down_proj(h)
        return x


class _PlantedDecoyCausalLM(nn.Module):
    """4-layer causal LM with a Layer-1 nullspace decoy and a Layer-2 causal MLP or Attention switch."""

    def __init__(self, plant_kind: str = "mlp", vocab: int = 32, hidden: int = 32, inter: int = 64, seed: int = 123) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(model_type="llama", hidden_size=hidden, num_attention_heads=4)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_PlantedBlock(hidden, inter, 4) for _ in range(4)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        with torch.no_grad():
            for p in self.parameters():
                p.normal_(0.0, 0.01)
            # Clear two-subspace geometry:
            #   dim 0 = causal logit channel (drives token 25 in loop, kept in realistic unsaturated logit range)
            #   dim 1 = nullspace decoy channel (orthogonal to lm_head and downstream readers; massive passive activation)
            self.lm_head.weight.zero_()
            self.lm_head.weight[25, 0] = 1.5
            for v in range(min(8, vocab)):
                if v != 25:
                    self.lm_head.weight[v, 2] = 1.0

            # Keep dim 1 strictly in the nullspace of downstream blocks so the decoy has zero causal effect
            for blk in self.model.layers:
                for lin in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj, blk.mlp.gate_proj, blk.mlp.up_proj):
                    lin.weight[:, 1].zero_()

            # Trigger tokens >= 20 activate dim 3 strongly in embed_tokens
            self.model.embed_tokens.weight.zero_()
            for t in range(vocab):
                if t >= 20:
                    self.model.embed_tokens.weight[t, 3] = 2.0
                else:
                    self.model.embed_tokens.weight[t, 2] = 0.4

            # Layer 1 decoy: writes massive shift along dim 1 (nullspace of lm_head & readers) when trigger active
            b1 = self.model.layers[1]
            b1.mlp.gate_proj.weight.zero_()
            b1.mlp.up_proj.weight.zero_()
            b1.mlp.down_proj.weight.zero_()
            b1.mlp.gate_proj.weight[0, 3] = 1.5
            b1.mlp.up_proj.weight[0, 3] = 1.5
            b1.mlp.down_proj.weight[1, 0] = 3.0

            # Layer 2 true causal switch: writes ~1.0 along dim 0 (non-saturated logit ~1.5) when trigger active
            b2 = self.model.layers[2]
            b2.mlp.gate_proj.weight.zero_()
            b2.mlp.up_proj.weight.zero_()
            b2.mlp.down_proj.weight.zero_()
            b2.self_attn.v_proj.weight.zero_()
            b2.self_attn.o_proj.weight.zero_()
            if plant_kind == "mlp":
                b2.mlp.gate_proj.weight[7, 3] = 0.7
                b2.mlp.up_proj.weight[7, 3] = 0.7
                b2.mlp.down_proj.weight[0, 7] = 0.9
            elif plant_kind == "attn":
                b2.self_attn.q_proj.weight.zero_()
                b2.self_attn.k_proj.weight.zero_()
                b2.self_attn.v_proj.weight[0, 3] = 0.6
                b2.self_attn.o_proj.weight[0, 0] = 0.9
            else:
                raise ValueError(f"Unsupported plant_kind: {plant_kind}")

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        x = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            x = blk(x)
        return SimpleNamespace(logits=self.lm_head(x))


def run_smoke() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for plant_kind in ("mlp", "attn"):
        model = _PlantedDecoyCausalLM(plant_kind=plant_kind).eval()
        seqs = [
            [1, 2, 3, 4, 25, 25, 25, 25],
            [2, 3, 4, 5, 25, 25, 25, 25],
            [3, 4, 5, 6, 25, 25, 25, 25],
            [4, 5, 6, 7, 25, 25, 25, 25],
        ]
        traces = [
            capture_causal_trace(
                model,
                s,
                target_span=(4, 8),
                healthy_span=(0, 3),
                grad_mode="compressed",
                top_k_mlp=16,
                group_size=16,
                hadamard_block=16,
                top_k_attn=0,
                capture_cache=False,
            )
            for s in seqs
        ]
        diag = diagnose_causal_traces(traces, split_ab=True, top_k_sites=4, top_n_neurons=8)
        want_mod = "layers.2.mlp.down_proj" if plant_kind == "mlp" else "layers.2.self_attn.o_proj"
        assert diag["summary"]["top_passive_site"] == "layers.1.mlp.down_proj", diag["summary"]
        assert diag["summary"]["top_causal_site"] == want_mod, diag["summary"]

        u_e0 = torch.zeros(32)
        u_e0[0] = 1.0
        u_e1 = torch.zeros(32)
        u_e1[1] = 1.0
        cands = {
            "true_causal_a4": MoIPatch(edits=[Edit(module=want_mod, u=diag["directions"][want_mod], alpha=4.0)]),
            "true_causal_a2": MoIPatch(edits=[Edit(module=want_mod, u=u_e0, alpha=2.0)]),
            "decoy_l1": MoIPatch(edits=[Edit(module="layers.1.mlp.down_proj", u=u_e1, alpha=4.0)]),
        }
        if plant_kind == "mlp":
            cands["true_sub_n8"] = MoIPatch(
                edits=[Edit(module=want_mod, u=diag["directions"][want_mod], alpha=4.0, idx=torch.tensor(diag["top_neurons"][want_mod][:4]))]
            )

        off = screen_patches_from_traces(traces, cands, model)
        onl = gradient_screen_candidates(
            model,
            cands,
            healthy_seqs=[s[:4] for s in seqs],
            target_seqs=seqs,
            target_spans=[(4, 8)] * len(seqs),
        )
        assert off[0]["name"] == onl[0]["name"], (off[0], onl[0])
        off_by = {r["name"]: r["grad_supp"] for r in off}
        onl_by = {r["name"]: r["grad_supp"] for r in onl}
        corr = _spearman([off_by[k] for k in cands], [onl_by[k] for k in cands])
        assert corr >= 0.99, (off_by, onl_by)

        results[plant_kind] = {
            "top_passive_site": diag["summary"]["top_passive_site"],
            "top_causal_site": diag["summary"]["top_causal_site"],
            "causal_rank": diag["causal_rank"][:4],
            "passive_rank": diag["passive_rank"][:4],
            "offline_winner": off[0]["name"],
            "online_winner": onl[0]["name"],
            "offline_vs_online_spearman": corr,
            "byte_breakdown": traces[0].byte_breakdown(),
        }
    return {"mode": "smoke", "passed": True, "results": results}


# ------------------------------------------------------------------ Real GPU benchmark
def run_real(args) -> dict[str, Any]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import discover_loop_fix as dl
    import moi_loop_router_vllm as lr

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        args.dtype, torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float32
    )
    t_all = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(dev).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    eos = set(lr.eos_ids(tok))

    seqs_info: list[dict[str, Any]] = []
    if args.from_outputs:
        out_p = Path(args.from_outputs)
        req_p = Path(args.from_requests) if args.from_requests else out_p.with_name(out_p.name.replace("outputs_", "requests_"))
        reqs = {r["id"]: r for r in lr.read_jsonl(req_p)}
        for o in lr.read_jsonl(out_p):
            body = lr.strip_eos(o["token_ids"], eos)
            lab = dl.token_labels(body)
            if (lab == 1).any():
                onset = int((lab == 1).nonzero()[0][0])
                prompt = reqs[o["id"]]["prompt_ids"]
                full = prompt + body
                ons_full = len(prompt) + onset
                h_lo = max(0, ons_full - args.window_before)
                t_hi = min(len(full), ons_full + args.window_after)
                seqs_info.append({
                    "id": o["id"],
                    "ids": full[h_lo:t_hi],
                    "target_span": (ons_full - h_lo, t_hi - h_lo),
                    "healthy_span": (0, max(1, ons_full - h_lo - 1)),
                })
                if len(seqs_info) >= args.n_seqs:
                    break
    else:
        pq = hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(pq).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        batch_sz = max(int(args.n_seqs) + 4, 12)
        max_pool = min(len(df), batch_sz * 3)
        questions = list(df["question"].iloc[:max_pool])
        for b_start in range(0, len(questions), batch_sz):
            if len(seqs_info) >= args.n_seqs:
                break
            q_batch = questions[b_start : b_start + batch_sz]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                for q in q_batch
            ]
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            in_width = int(enc["input_ids"].shape[1])
            print(f"[causal-diagnose] Generating batch {b_start // batch_sz + 1} ({len(q_batch)} prompts, max_new={args.max_new_tokens})...", flush=True)
            with torch.no_grad():
                gen_batch = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                )
            for j in range(len(q_batch)):
                if len(seqs_info) >= args.n_seqs:
                    break
                pad_left = int((enc["attention_mask"][j] == 0).sum().item())
                p_ids = [int(x) for x in enc["input_ids"][j, pad_left:].tolist()]
                out_raw = [int(x) for x in gen_batch[j, in_width:].tolist()]
                body = lr.strip_eos(out_raw, eos)
                lab = dl.token_labels(body)
                if (lab == 1).any():
                    onset = int((lab == 1).nonzero()[0][0])
                    full = p_ids + body
                    ons_full = len(p_ids) + onset
                    h_lo = max(0, ons_full - args.window_before)
                    t_hi = min(len(full), ons_full + args.window_after)
                    seqs_info.append({
                        "id": b_start + j,
                        "ids": full[h_lo:t_hi],
                        "target_span": (ons_full - h_lo, t_hi - h_lo),
                        "healthy_span": (0, max(1, ons_full - h_lo - 1)),
                    })
            del enc, gen_batch
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[causal-diagnose] Collected {len(seqs_info)}/{args.n_seqs} looping sequences so far.", flush=True)

    if len(seqs_info) < 2:
        raise SystemExit(f"Found only {len(seqs_info)} looping sequences; need >= 2")

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    t_cap = time.perf_counter()
    traces = [
        capture_causal_trace(
            model,
            s["ids"],
            target_span=s["target_span"],
            healthy_span=s["healthy_span"],
            grad_mode="compressed",
            grad_checkpointing=True,
            top_k_attn=0,
            capture_cache=False,
        )
        for s in seqs_info
    ]
    capture_s = round(time.perf_counter() - t_cap, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(dev) / (1024**2), 1) if dev.type == "cuda" else 0.0

    # Measure summary-only size vs compressed size
    mean_bd = {
        k: round(sum(ct.byte_breakdown()[k] for ct in traces) / len(traces) / (1024**2), 3)
        for k in traces[0].byte_breakdown()
    }
    summary_only_mb = round(
        sum(
            sum(v.numel() * v.element_size() for gs in ct.grad_summary.values() for v in gs.values() if isinstance(v, torch.Tensor))
            for ct in traces
        ) / len(traces) / (1024**2),
        3,
    )

    diag = diagnose_causal_traces(traces, split_ab=True, top_k_sites=10, top_n_neurons=64)

    def _rank_pos(lst: list[str], target: str) -> int | None:
        return (lst.index(target) + 1) if target in lst else None

    probe_sites = {
        "layers.15.mlp.down_proj": {"causal": _rank_pos(diag["causal_rank"], "layers.15.mlp.down_proj"),
                                    "passive": _rank_pos(diag["passive_rank"], "layers.15.mlp.down_proj")},
        "layers.19.mlp.down_proj": {"causal": _rank_pos(diag["causal_rank"], "layers.19.mlp.down_proj"),
                                    "passive": _rank_pos(diag["passive_rank"], "layers.19.mlp.down_proj")},
        "layers.15.self_attn.o_proj": {"causal": _rank_pos(diag["causal_rank"], "layers.15.self_attn.o_proj"),
                                       "passive": _rank_pos(diag["passive_rank"], "layers.15.self_attn.o_proj")},
    }

    # Build candidate grid on top causal sites
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

    top_sites = [r["module"] for r in diag["sites"][:2] if r["module"] in mod_meta]
    candidates: dict[str, MoIPatch] = {}
    for k_sites in (1, min(2, len(top_sites))):
        use_sites = top_sites[:k_sites]
        for n_col in (16, 64, "all"):
            for alpha in (4.0, 8.0, 12.0):
                edits = []
                for mk in use_sites:
                    mm = mod_meta[mk]
                    cols = None if n_col == "all" else torch.tensor(diag["top_neurons"][mk][: int(n_col)], dtype=torch.long)
                    edits.append(
                        Edit(
                            module=mk,
                            u=diag["directions"][mk].double(),
                            alpha=float(alpha),
                            idx=cols,
                            base_sha256=mm["base_sha256"],
                            base_dtype=mm["base_dtype"],
                            shape=mm["shape"],
                        )
                    )
                cname = f"k{k_sites}_n{n_col}_a{int(alpha)}"
                candidates[cname] = MoIPatch(edits=edits, meta={"name": cname})

    t_off = time.perf_counter()
    off_rows = screen_patches_from_traces(traces, candidates, model)
    offline_screen_s = round(time.perf_counter() - t_off, 4)

    healthy_seqs = [s["ids"][: s["healthy_span"][1] + 1] for s in seqs_info]
    target_seqs = [s["ids"] for s in seqs_info]
    target_spans = [s["target_span"] for s in seqs_info]

    t_onl = time.perf_counter()
    onl_rows = gradient_screen_candidates(
        model, candidates, healthy_seqs=healthy_seqs, target_seqs=target_seqs, target_spans=target_spans
    )
    online_grad_s = round(time.perf_counter() - t_onl, 4)

    t_fwd = time.perf_counter()
    fwd_rows = screen_patch_candidates(
        model, candidates, healthy_seqs=healthy_seqs, target_seqs=target_seqs, target_spans=target_spans
    )
    online_fwd_s = round(time.perf_counter() - t_fwd, 4)

    off_by = {r["name"]: r for r in off_rows}
    onl_by = {r["name"]: r for r in onl_rows}
    fwd_by = {r["name"]: r for r in fwd_rows}
    names = list(candidates)

    payload = {
        "model": args.model,
        "dtype": str(dtype),
        "n_seqs": len(traces),
        "capture_s": capture_s,
        "peak_vram_mb": peak_vram_mb,
        "mean_trace_mb": mean_bd,
        "mean_grad_summary_only_mb": summary_only_mb,
        "diagnosis_summary": diag["summary"],
        "top_causal_sites": diag["sites"][:6],
        "passive_top6": diag["passive_rank"][:6],
        "probe_ranks": probe_sites,
        "timing_s": {
            "offline_grad_screen": offline_screen_s,
            "online_grad_screen": online_grad_s,
            "online_fwd_screen": online_fwd_s,
            "offline_speedup_vs_online_grad": round(online_grad_s / max(offline_screen_s, 1e-6), 2),
            "offline_speedup_vs_fwd_screen": round(online_fwd_s / max(offline_screen_s, 1e-6), 2),
            "total_s": round(time.perf_counter() - t_all, 2),
        },
        "correlations": {
            "offline_vs_online_grad_supp_spearman": _spearman(
                [off_by[n]["grad_supp"] for n in names], [onl_by[n]["grad_supp"] for n in names]
            ),
            "offline_vs_online_grad_pareto_spearman": _spearman(
                [off_by[n]["grad_pareto"] for n in names], [onl_by[n]["grad_pareto"] for n in names]
            ),
            "offline_vs_fwd_pareto_spearman": _spearman(
                [off_by[n]["grad_pareto"] for n in names], [fwd_by[n]["pareto_score"] for n in names]
            ),
        },
        "top3_offline": off_rows[:3],
        "top3_online_grad": onl_rows[:3],
        "top3_fwd_screen": fwd_rows[:3],
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run tiny synthetic CPU planted-circuit verification")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--from-outputs", default=None, help="Optional cached outputs_*.jsonl with looping responses")
    ap.add_argument("--from-requests", default=None, help="Matching requests_*.jsonl (inferred from --from-outputs if omitted)")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--window-before", type=int, default=512)
    ap.add_argument("--window-after", type=int, default=256)
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
