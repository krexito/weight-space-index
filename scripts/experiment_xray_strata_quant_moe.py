"""Exp 2B: Per-Expert Global Hessian Knapsack + Asymmetric S3_K/S2_K + Cold-Expert Shrinkage on OLMoE-1B-7B.

Replicates Strata's core MoE quantization mechanism (`pipeline.py` + `hbm_quant.py`) on an open MoE
in our Tome-1 Atlas (`allenai/OLMoE-1B-7B-0125-Instruct`: 16L x 64 experts, top_k=8, hidden=2048, inter=1024):
  1. `base_bf16`
  2. `uniform_rtn_3_2`                : Naive RTN (`gate/up=S3_K` 3b, `down=S2_K` 2b -> 2.854 b/w)
  3. `uniform_s3_s2_h64_gptq_flip`    : Uniform asymmetric `S3_K + S2_K` with H64 + Block-LDL + 64-flip
  4. `per_expert_knapsack_2_85b`      : Exact matched budget across all 1,024 individual experts + X-Ray floor
  5. `per_expert_knapsack_2_85b_scale_qat`: Arm 4 + 2-epoch Top-256 KL Scale-QAT over 1D norms (0 extra bytes)
  6. `per_expert_knapsack_2_45b_scale_qat`: Extreme 2.45 b/w per-expert knapsack + Scale-QAT

CPU smoke test::
    python scripts/experiment_xray_strata_quant_moe.py --smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
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
from experiment_pocket_moe_math500 import format_mmlu_prompt_ids, load_mmlu_split  # noqa: E402
from experiment_reasoning_distill_pilot import _atomic_write_json  # noqa: E402
from experiment_xray_strata_quant_dense import (  # noqa: E402
    _eval_abcd_mc,
    _eval_perplexity,
    _load_wikitext_chunks,
)
from wsi.capture import find_blocks  # noqa: E402
from wsi.strata_quant import (  # noqa: E402
    apply_h64_sym,
    build_quant_sensitivity_map,
    capture_teacher_topk_targets,
    damped_cholesky_inv,
    eval_topk_kl_and_top1,
    gptq_flip_quantize_sk_torch,
    rtn_quantize_sk_torch,
    run_norm_scale_qat,
    solve_knapsack_bit_allocation,
)

DEFAULT_XRAY_PATH = ROOT / "benchmarks" / "xray_atlas" / "tome1_xray" / "allenai__olmoe-1b-7b-0125-instruct.xray.json"


class _ExpertAccessor:
    """Unified read/write accessor supporting both `ModuleList` experts and fused 3D tensor `OlmoeExperts`."""

    def __init__(self, blk: nn.Module, hidden: int, inter: int, n_experts: int) -> None:
        self.mlp = blk.mlp
        self.gate = blk.mlp.gate
        self.experts = blk.mlp.experts
        self.hidden = int(hidden)
        self.inter = int(inter)
        self.n_experts = int(n_experts)
        if isinstance(self.experts, nn.ModuleList):
            self.kind = "modulelist"
            self.dtype = self.experts[0].down_proj.weight.dtype
            self.device = self.experts[0].down_proj.weight.device
        elif hasattr(self.experts, "gate_up_proj") and hasattr(self.experts, "down_proj"):
            self.kind = "fused"
            self.dtype = self.experts.down_proj.dtype
            self.device = self.experts.down_proj.device
            self.gu_transposed = tuple(self.experts.gate_up_proj.shape[1:]) == (self.hidden, 2 * self.inter)
            self.dn_transposed = tuple(self.experts.down_proj.shape[1:]) == (self.inter, self.hidden)
        else:
            raise RuntimeError(f"Unsupported OLMoE experts layout: {type(self.experts)}")

    def get_gu_dn(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(W_gu [2*inter, hidden], W_dn [hidden, inter])`."""
        if self.kind == "modulelist":
            ex = self.experts[e]
            w_gu = torch.cat([ex.gate_proj.weight.data, ex.up_proj.weight.data], dim=0)
            w_dn = ex.down_proj.weight.data
            return w_gu, w_dn
        w_gu = self.experts.gate_up_proj.data[e]
        w_dn = self.experts.down_proj.data[e]
        if self.gu_transposed:
            w_gu = w_gu.T.contiguous()
        if self.dn_transposed:
            w_dn = w_dn.T.contiguous()
        return w_gu, w_dn

    def set_gu_dn(self, e: int, w_gu: torch.Tensor, w_dn: torch.Tensor) -> None:
        if self.kind == "modulelist":
            ex = self.experts[e]
            ex.gate_proj.weight.data.copy_(w_gu[: self.inter].to(device=ex.gate_proj.weight.device, dtype=ex.gate_proj.weight.dtype))
            ex.up_proj.weight.data.copy_(w_gu[self.inter :].to(device=ex.up_proj.weight.device, dtype=ex.up_proj.weight.dtype))
            ex.down_proj.weight.data.copy_(w_dn.to(device=ex.down_proj.weight.device, dtype=ex.down_proj.weight.dtype))
            return
        w_gu_t = w_gu.T.contiguous() if self.gu_transposed else w_gu
        w_dn_t = w_dn.T.contiguous() if self.dn_transposed else w_dn
        self.experts.gate_up_proj.data[e].copy_(w_gu_t.to(device=self.device, dtype=self.dtype))
        self.experts.down_proj.data[e].copy_(w_dn_t.to(device=self.device, dtype=self.dtype))


def _load_arc_100(tok: Any, seed: int = 42) -> tuple[list[list[int]], list[int]]:
    from datasets import load_dataset

    ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(ds)).tolist()[:100]
    pids: list[list[int]] = []
    golds: list[int] = []
    letters = "ABCD"
    for idx in order:
        it = ds[idx]
        labels = list(it["choices"]["label"])
        texts = list(it["choices"]["text"])
        if len(labels) < 4:
            continue
        ans_key = str(it["answerKey"]).strip()
        if ans_key in labels:
            gold_pos = labels.index(ans_key)
        elif ans_key.isdigit() and int(ans_key) - 1 < len(labels):
            gold_pos = int(ans_key) - 1
        else:
            continue
        if gold_pos >= 4:
            continue
        lines = [it["question"], ""] + [f"{letters[j]}. {texts[j]}" for j in range(4)] + ["", "Answer:"]
        prompt = "\n".join(lines)
        if getattr(tok, "chat_template", None):
            try:
                txt_p = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                ids = tok.encode(txt_p, add_special_tokens=False)
            except Exception:
                ids = tok.encode(prompt, add_special_tokens=True)
        else:
            ids = tok.encode(prompt, add_special_tokens=True)
        pids.append([int(x) for x in ids])
        golds.append(gold_pos)
    return pids, golds


def _print_summary(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "smoke":
        print(json.dumps(payload, indent=2), flush=True)
        return
    print("\n=== EXP 2B: PER-EXPERT KNAPSACK + S3_K/S2_K ON OLMoE-1B-7B ===", flush=True)
    for arm_nm, st in payload.get("arms", {}).items():
        print(
            f"  {arm_nm:35s} | exp_bpw={st.get('expert_bits_per_weight', 16.0):6.3f} | "
            f"PPL={st['wikitext2_ppl']:7.3f} | KL={st['kl_mean']:7.5f} | Top1={st['top1_agreement']*100:5.2f}% | "
            f"MMLU={st['mmlu_200']['accuracy']*100:5.1f}% | ARC={st['arc_100']['accuracy']*100:5.1f}%",
            flush=True,
        )
    print("\nCriteria:", payload.get("criteria"), flush=True)
    compact = {k: v for k, v in payload.items() if k != "expert_allocations"}
    print("<<<COMPACT_JSON_BEGIN>>>", flush=True)
    print(json.dumps(compact, separators=(",", ":")), flush=True)
    print("<<<COMPACT_JSON_END>>>", flush=True)


def run_smoke() -> dict[str, Any]:
    """Tiny CPU MoE verification of per-expert Hessians, cold-expert shrinkage, knapsack & Scale-QAT."""
    torch.manual_seed(42)

    class _ToyExpert(nn.Module):
        def __init__(self, d: int, inter: int):
            super().__init__()
            self.gate_proj = nn.Linear(d, inter, bias=False)
            self.up_proj = nn.Linear(d, inter, bias=False)
            self.down_proj = nn.Linear(inter, d, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class _ToyMoEMLP(nn.Module):
        def __init__(self, d: int, inter: int, n_e: int):
            super().__init__()
            self.gate = nn.Linear(d, n_e, bias=False)
            self.experts = nn.ModuleList([_ToyExpert(d, inter) for _ in range(n_e)])

        def forward(self, x):
            b, t, d = x.shape
            xf = x.reshape(-1, d)
            probs = F.softmax(self.gate(xf).float(), dim=-1)
            topw, topi = torch.topk(probs, k=2, dim=-1)
            topw = topw / topw.sum(dim=-1, keepdim=True)
            out = torch.zeros_like(xf)
            for e in range(len(self.experts)):
                m = topi == e
                if m.any():
                    tm, sm = torch.where(m)
                    out.index_add_(0, tm, self.experts[e](xf[tm]) * topw[tm, sm].unsqueeze(-1).to(out.dtype))
            return out.reshape(b, t, d)

    class _ToyMoEBlock(nn.Module):
        def __init__(self, d: int, inter: int, n_e: int):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(d)
            self.mlp = _ToyMoEMLP(d, inter, n_e)

        def forward(self, x):
            return x + self.mlp(self.input_layernorm(x))

    class _ToyMoELM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 64)
            self.layers = nn.ModuleList([_ToyMoEBlock(64, 64, 4) for _ in range(3)])
            self.norm = nn.RMSNorm(64)
            self.lm_head = nn.Linear(64, 32, bias=False)

        def forward(self, input_ids, use_cache=False):
            h = self.embed_tokens(input_ids)
            for blk in self.layers:
                h = blk(h)
            logits = self.lm_head(self.norm(h))
            return type("Out", (), {"logits": logits})()

    teacher = _ToyMoELM().eval()
    model = _ToyMoELM().eval()
    model.load_state_dict(teacher.state_dict())
    for p in model.parameters():
        p.requires_grad_(False)

    fake_card = json.loads(DEFAULT_XRAY_PATH.read_text(encoding="utf-8"))
    fake_card["identity"]["n_layers"] = 3
    fake_card["anatomy"]["forward_profile"]["massive_act_max"]["layer"] = 1
    fake_card["anatomy"]["fact_readout"]["dla_peak"]["layer"] = 2
    qmap = build_quant_sensitivity_map(fake_card)

    seqs = [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
    tr_recs = capture_teacher_topk_targets(teacher, seqs[:1], "cpu", topk=16)
    dv_recs = capture_teacher_topk_targets(teacher, seqs[1:], "cpu", topk=16)

    # Capture per-expert Hessians on layer 0..2
    items = []
    uni_bytes = 0
    uni_err = 0.0
    rtn_err = 0.0
    x_in = torch.randn((32, 64))
    h_gu_l = (x_in.T @ x_in) / 32.0 + 0.01 * torch.eye(64)
    h_in = torch.randn((32, 64))
    h_dn_l = (h_in.T @ h_in) / 32.0 + 0.01 * torch.eye(64)

    for li, blk in enumerate(model.layers):
        acc = _ExpertAccessor(blk, 64, 64, 4)
        for e in range(4):
            w_gu, w_dn = acc.get_gu_dn(e)
            h_ge = h_gu_l * (4.0 if e == 0 else 0.2)
            h_de = h_dn_l * (4.0 if e == 0 else 0.2)
            _, wb_r1, er1 = rtn_quantize_sk_torch(w_gu, "S3_K", H=h_ge)
            _, wb_r2, er2 = rtn_quantize_sk_torch(w_dn, "S2_K", H=h_de)
            rtn_err += float((er1 or 0.0) + (er2 or 0.0))

            by_c_gu = {}
            by_c_dn = {}
            for cd in ("S2_K", "S3_K", "S4_K"):
                wq_g, wbg, ehg = gptq_flip_quantize_sk_torch(w_gu, h_ge, cd, use_h64=True, max_flips=16)
                wq_d, wbd, ehd = gptq_flip_quantize_sk_torch(w_dn, h_de, cd, use_h64=True, max_flips=16)
                by_c_gu[cd] = (wq_g, wbg, ehg)
                by_c_dn[cd] = (wq_d, wbd, ehd)
            uni_bytes += by_c_gu["S3_K"][1] + by_c_dn["S2_K"][1]
            uni_err += by_c_gu["S3_K"][2] + by_c_dn["S2_K"][2]

            w_x = 1.0 + float(qmap["per_layer_scores"][li])
            pairs = [("S3_K", "S2_K"), ("S3_K", "S3_K"), ("S4_K", "S3_K")] if li in qmap["protected_layers"] else [
                ("S2_K", "S2_K"), ("S3_K", "S2_K"), ("S3_K", "S3_K"), ("S4_K", "S3_K"), ("S4_K", "S4_K")
            ]
            opts = [(f"{cg}+{cd}", by_c_gu[cg][1] + by_c_dn[cd][1], (by_c_gu[cg][2] + by_c_dn[cd][2]) * w_x) for cg, cd in pairs]
            items.append({"id": (li, e), "options": opts})
            if li == 0 and e == 0:
                acc.set_gu_dn(e, by_c_gu["S3_K"][0], by_c_dn["S2_K"][0])

    sel, used_b, _ = solve_knapsack_bit_allocation(items, uni_bytes)
    assert used_b <= uni_bytes
    assert uni_err < rtn_err
    pre_kl = eval_topk_kl_and_top1(model, dv_recs, "cpu")["kl_mean"]
    qat_res = run_norm_scale_qat(model, tr_recs, dv_recs, device="cpu", epochs=4, lr=2e-2, grad_accum=1)
    assert qat_res["dev_kl_best"] <= pre_kl + 1e-6
    return {
        "mode": "smoke",
        "uniform_s3_s2_bytes": uni_bytes,
        "knapsack_bytes": used_b,
        "rtn_err": round(rtn_err, 5),
        "strata_err": round(uni_err, 5),
        "n_experts_knapsack": len(sel),
        "passed": True,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    torch.manual_seed(int(args.seed))

    xray_card = json.loads(Path(args.xray).read_text(encoding="utf-8"))
    qmap = build_quant_sensitivity_map(xray_card)
    prot_layers = set(int(x) for x in qmap["protected_layers"])
    layer_scores = list(qmap["per_layer_scores"])

    print(f"[exp2b] Loading {args.model} | X-Ray protected={sorted(prot_layers)}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, blocks = find_blocks(model)
    n_layers = len(blocks)
    n_experts = int(getattr(model.config, "num_experts", 64))
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    norm_topk_prob = bool(getattr(model.config, "norm_topk_prob", True))
    hidden = int(model.config.hidden_size)
    inter = int(model.config.intermediate_size)
    assert len(layer_scores) == n_layers

    wt = _load_wikitext_chunks(tok, seq_len=512, n_train=48, n_dev=8, n_test_ppl=32, n_test_kl=16)
    mmlu_test, _ = load_mmlu_split(n_test=200, n_router_train=20, seed=int(args.seed))
    mmlu_pids = [format_mmlu_prompt_ids(tok, it) for it in mmlu_test]
    mmlu_gold = [int(it["gold_idx"]) for it in mmlu_test]
    mmlu_ismath = [bool(it.get("is_math_subject", False)) for it in mmlu_test]
    arc_pids, arc_gold = _load_arc_100(tok, seed=int(args.seed))
    letter_ids = ec._resolve_mmlu_letter_ids(tok)

    orig_norms = {nm: mod.weight.detach().cpu().clone() for nm, mod in model.named_modules() if "norm" in nm.lower() and hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor) and mod.weight.ndim == 1}
    print("[exp2b] Capturing BF16 teacher Top-256 targets + baseline eval...", flush=True)
    tr_recs = capture_teacher_topk_targets(model, wt["train"], dev, topk=256)
    dv_recs = capture_teacher_topk_targets(model, wt["dev"], dev, topk=256)
    te_recs = capture_teacher_topk_targets(model, wt["test_kl"], dev, topk=256)

    base_ppl = _eval_perplexity(model, wt["test_ppl"], dev)
    base_mmlu = _eval_abcd_mc(model, mmlu_pids, mmlu_gold, mmlu_ismath, letter_ids, pad_id, dev)
    base_arc = _eval_abcd_mc(model, arc_pids, arc_gold, None, letter_ids, pad_id, dev)
    arms_out: dict[str, Any] = {
        "base_bf16": {
            "expert_bits_per_weight": 16.0,
            "wikitext2_ppl": base_ppl,
            "kl_mean": 0.0,
            "top1_agreement": 1.0,
            "mmlu_200": base_mmlu,
            "arc_100": base_arc,
        }
    }
    print(f"[exp2b] BASE BF16: PPL={base_ppl} MMLU={base_mmlu['accuracy']*100:.1f}% ARC={base_arc['accuracy']*100:.1f}%", flush=True)

    # ---- Phase 1: Streamed per-layer per-expert Hessian capture + cold-expert shrinkage + multi-code quantization ----
    t_p1 = time.perf_counter()
    with tempfile.TemporaryDirectory() as qdir_str:
        qdir = Path(qdir_str)
        expert_opts: dict[tuple[int, int], dict[str, tuple[int, float]]] = {}
        per_layer_s2_err = [0.0] * n_layers
        per_layer_s32_err = [0.0] * n_layers
        per_layer_usage: list[list[float]] = []
        total_expert_params = 0
        uniform_s3_s2_bytes = 0

        for li, blk in enumerate(blocks):
            t_l = time.perf_counter()
            acc = _ExpertAccessor(blk, hidden, inter, n_experts)
            w_gu_all = []
            w_dn_all = []
            for e in range(n_experts):
                wg, wd = acc.get_gu_dn(e)
                w_gu_all.append(wg.detach().clone())
                w_dn_all.append(wd.detach().clone())
                total_expert_params += int(wg.numel() + wd.numel())

            # Save original weights once per layer
            torch.save((torch.stack(w_gu_all).cpu(), torch.stack(w_dn_all).cpu()), qdir / f"L{li:02d}_ORIG.pt")

            H_gu_e = torch.zeros((n_experts, hidden, hidden), device=dev, dtype=torch.float32)
            H_dn_e = torch.zeros((n_experts, inter, inter), device=dev, dtype=torch.float32)
            H_gu_layer = torch.zeros((hidden, hidden), device=dev, dtype=torch.float32)
            usage = torch.zeros((n_experts,), device=dev, dtype=torch.float64)
            e_tokens = torch.zeros((n_experts,), device=dev, dtype=torch.int64)

            captured_inputs: list[torch.Tensor] = []

            def _pre_mlp(_mod: nn.Module, inp: tuple[torch.Tensor, ...]) -> None:
                captured_inputs.append(inp[0].detach().reshape(-1, hidden).cpu())

            hk = acc.mlp.register_forward_pre_hook(_pre_mlp)
            with torch.no_grad():
                for seq in wt["train"]:
                    ids = torch.tensor(seq, dtype=torch.long, device=dev).unsqueeze(0)
                    model(input_ids=ids, use_cache=False)
                    del ids
            hk.remove()

            w_gate = acc.gate.weight.detach().to(device=dev, dtype=torch.float32)
            for x_cpu in captured_inputs:
                x_d = x_cpu.to(device=dev)
                xf = x_d.float()
                r_logits = F.linear(xf, w_gate)
                probs = F.softmax(r_logits, dim=-1)
                topw, topi = torch.topk(probs, k=top_k, dim=-1)
                if norm_topk_prob:
                    topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)
                w_tok_sum = (topw * topw).sum(dim=-1)
                xf_w = xf * torch.sqrt(w_tok_sum).unsqueeze(-1)
                H_gu_layer.add_(xf_w.T @ xf_w)

                for e in torch.unique(topi).tolist():
                    tm, sm = torch.where(topi == e)
                    we = topw[tm, sm]
                    w2 = we * we
                    usage[e] += float(w2.sum().item())
                    e_tokens[e] += int(tm.numel())
                    sw = torch.sqrt(w2).unsqueeze(-1)

                    xe = x_d[tm]
                    xef = xe.float() * sw
                    H_gu_e[e].add_(xef.T @ xef)

                    wg_e, wd_e = w_gu_all[e], w_dn_all[e]
                    gu = F.linear(xe, wg_e)
                    g, u = gu[:, :inter].float(), gu[:, inter:].float()
                    h_act = F.silu(g) * u
                    hf = h_act * sw
                    H_dn_e[e].add_(hf.T @ hf)
            del captured_inputs

            n_tot_tok = float(max(1, sum(len(s) for s in wt["train"])))
            H_gu_layer.div_(n_tot_tok)
            H_dn_mean = H_dn_e.sum(dim=0) / float(max(1, int((e_tokens > 0).sum().item())))
            per_layer_usage.append([round(float(x), 4) for x in usage.cpu().tolist()])

            # Quantize each expert across RTN_3_2 and Strata {S2_K, S3_K, S4_K} with cold-expert shrinkage
            rtn_gu_list, rtn_dn_list = [], []
            by_code_gu: dict[str, list[torch.Tensor]] = {c: [] for c in ("S2_K", "S3_K", "S4_K")}
            by_code_dn: dict[str, list[torch.Tensor]] = {c: [] for c in ("S2_K", "S3_K", "S4_K")}

            for e in range(n_experts):
                tok_e = max(1.0, float(e_tokens[e].item()))
                h_ge_loc = H_gu_e[e] / tok_e
                h_de_loc = H_dn_e[e] / tok_e
                # Strata cold-expert shrinkage
                if e_tokens[e] > 0:
                    h_ge = 0.70 * h_ge_loc + 0.30 * H_gu_layer
                    h_de = 0.75 * h_de_loc + 0.25 * (H_dn_mean / max(1.0, float(e_tokens.float().mean().item())))
                else:
                    h_ge = H_gu_layer
                    h_de = H_dn_mean / max(1.0, float(e_tokens.float().mean().item()))

                # Normalize scale using routing usage mass so knapsack error reflects expected perturbation
                u_weight = max(1e-4, float(usage[e].item()) / n_tot_tok)
                wg_e, wd_e = w_gu_all[e], w_dn_all[e]

                # Naive RTN (S3_K gate_up + S2_K down)
                rg, _, _ = rtn_quantize_sk_torch(wg_e, "S3_K", use_clipping=False, use_h64=False, H=h_ge)
                rd, _, _ = rtn_quantize_sk_torch(wd_e, "S2_K", use_clipping=False, use_h64=False, H=h_de)
                rtn_gu_list.append(rg.cpu())
                rtn_dn_list.append(rd.cpu())

                pre_g = damped_cholesky_inv(apply_h64_sym(h_ge), damp_frac=0.01)
                pre_d = damped_cholesky_inv(apply_h64_sym(h_de), damp_frac=0.01)

                gu_res = {}
                dn_res = {}
                for cd in ("S2_K", "S3_K", "S4_K"):
                    qg, wbg, ehg = gptq_flip_quantize_sk_torch(wg_e, h_ge, cd, use_h64=True, max_flips=64, precomputed_H=pre_g)
                    qd, wbd, ehd = gptq_flip_quantize_sk_torch(wd_e, h_de, cd, use_h64=True, max_flips=64, precomputed_H=pre_d)
                    by_code_gu[cd].append(qg.cpu())
                    by_code_dn[cd].append(qd.cpu())
                    gu_res[cd] = (wbg, float(ehg) * u_weight)
                    dn_res[cd] = (wbd, float(ehd) * u_weight)

                # Store 5 pair options per expert
                expert_opts[(li, e)] = {}
                for cg, cd in [("S2_K", "S2_K"), ("S3_K", "S2_K"), ("S3_K", "S3_K"), ("S4_K", "S3_K"), ("S4_K", "S4_K")]:
                    pair_nm = f"{cg}+{cd}"
                    wb_p = gu_res[cg][0] + dn_res[cd][0]
                    eh_p = gu_res[cg][1] + dn_res[cd][1]
                    expert_opts[(li, e)][pair_nm] = (wb_p, eh_p)
                uniform_s3_s2_bytes += expert_opts[(li, e)]["S3_K+S2_K"][0]
                per_layer_s2_err[li] += expert_opts[(li, e)]["S2_K+S2_K"][1]
                per_layer_s32_err[li] += expert_opts[(li, e)]["S3_K+S2_K"][1]

            torch.save((torch.stack(rtn_gu_list), torch.stack(rtn_dn_list)), qdir / f"L{li:02d}_RTN.pt")
            for cd in ("S2_K", "S3_K", "S4_K"):
                torch.save((torch.stack(by_code_gu[cd]), torch.stack(by_code_dn[cd])), qdir / f"L{li:02d}_{cd}.pt")

            del H_gu_e, H_dn_e, H_gu_layer, H_dn_mean, w_gu_all, w_dn_all, rtn_gu_list, rtn_dn_list, by_code_gu, by_code_dn
            torch.cuda.empty_cache()
            print(f"[exp2b] Layer {li:02d}/{n_layers-1} quantized 64 experts in {time.perf_counter()-t_l:.1f}s", flush=True)

        # ---- Build Per-Expert Knapsack Allocations ----
        def _solve_moe_knapsack(budget_b: int, allow_prot_s2s2: bool = False) -> tuple[dict[tuple[int, int], str], int]:
            items = []
            for li in range(n_layers):
                w_xray = 1.0 + float(layer_scores[li])
                is_prot = li in prot_layers
                pairs = [("S3_K+S2_K"), ("S3_K+S3_K"), ("S4_K+S3_K"), ("S4_K+S4_K")] if (is_prot and not allow_prot_s2s2) else [
                    "S2_K+S2_K", "S3_K+S2_K", "S3_K+S3_K", "S4_K+S3_K", "S4_K+S4_K"
                ]
                for e in range(n_experts):
                    opts = [(p, expert_opts[(li, e)][p][0], expert_opts[(li, e)][p][1] * w_xray) for p in pairs]
                    items.append({"id": (li, e), "options": opts})
            sel, used_b, _ = solve_knapsack_bit_allocation(items, budget_b)
            return sel, used_b

        plan_s3s2 = {(li, e): "S3_K+S2_K" for li in range(n_layers) for e in range(n_experts)}
        plan_k285, bytes_k285 = _solve_moe_knapsack(uniform_s3_s2_bytes, allow_prot_s2s2=False)
        budget_245 = int(round(total_expert_params * 2.45 / 8.0))
        plan_k245, bytes_k245 = _solve_moe_knapsack(budget_245, allow_prot_s2s2=True)

        def _apply_moe_arm(arm_kind: str, plan: dict[tuple[int, int], str] | None = None) -> None:
            for nm, mod in model.named_modules():
                if nm in orig_norms:
                    mod.weight.data.copy_(orig_norms[nm].to(device=dev, dtype=mod.weight.dtype))
            for li, blk in enumerate(blocks):
                acc = _ExpertAccessor(blk, hidden, inter, n_experts)
                if arm_kind == "RTN":
                    gu_s, dn_s = torch.load(qdir / f"L{li:02d}_RTN.pt", map_location="cpu", weights_only=True)
                    for e in range(n_experts):
                        acc.set_gu_dn(e, gu_s[e], dn_s[e])
                    del gu_s, dn_s
                else:
                    assert plan is not None
                    loaded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                    for cd in ("S2_K", "S3_K", "S4_K"):
                        loaded[cd] = torch.load(qdir / f"L{li:02d}_{cd}.pt", map_location="cpu", weights_only=True)
                    for e in range(n_experts):
                        cg, cd = plan[(li, e)].split("+")
                        acc.set_gu_dn(e, loaded[cg][0][e], loaded[cd][1][e])
                    del loaded

        def _eval_arm(arm_nm: str, arm_kind: str, plan: dict[tuple[int, int], str] | None, wire_b: int, *, do_qat: bool = False) -> None:
            _apply_moe_arm(arm_kind, plan)
            qat_meta = None
            if do_qat:
                qat_meta = run_norm_scale_qat(model, tr_recs, dv_recs, device=dev, epochs=2, lr=5e-4, grad_accum=4, seed=int(args.seed))
            ppl = _eval_perplexity(model, wt["test_ppl"], dev)
            kl_res = eval_topk_kl_and_top1(model, te_recs, dev)
            mmlu_r = _eval_abcd_mc(model, mmlu_pids, mmlu_gold, mmlu_ismath, letter_ids, pad_id, dev)
            arc_r = _eval_abcd_mc(model, arc_pids, arc_gold, None, letter_ids, pad_id, dev)
            bpw = round((wire_b * 8.0) / max(1, total_expert_params), 4)
            arms_out[arm_nm] = {
                "expert_bits_per_weight": bpw,
                "expert_wire_mb": round(wire_b / (1024**2), 2),
                "wikitext2_ppl": ppl,
                "kl_mean": round(kl_res["kl_mean"], 6),
                "top1_agreement": round(kl_res["top1_agreement"], 4),
                "mmlu_200": mmlu_r,
                "arc_100": arc_r,
                "scale_qat": qat_meta,
            }
            print(
                f"[exp2b] {arm_nm:35s} | bpw={bpw:.3f} | PPL={ppl:.3f} | KL={kl_res['kl_mean']:.5f} | "
                f"Top1={kl_res['top1_agreement']*100:.2f}% | MMLU={mmlu_r['accuracy']*100:.1f}% | ARC={arc_r['accuracy']*100:.1f}%",
                flush=True,
            )

        _eval_arm("uniform_rtn_3_2", "RTN", None, uniform_s3_s2_bytes)
        _eval_arm("uniform_s3_s2_h64_gptq_flip", "PLAN", plan_s3s2, uniform_s3_s2_bytes)
        _eval_arm("per_expert_knapsack_2_85b", "PLAN", plan_k285, bytes_k285)
        _eval_arm("per_expert_knapsack_2_85b_scale_qat", "PLAN", plan_k285, bytes_k285, do_qat=True)
        _eval_arm("per_expert_knapsack_2_45b_scale_qat", "PLAN", plan_k245, bytes_k245, do_qat=True)

    per_layer_k285_mix = []
    per_layer_k285_mb = []
    for li in range(n_layers):
        c = Counter(plan_k285[(li, e)] for e in range(n_experts))
        b = sum(expert_opts[(li, e)][plan_k285[(li, e)]][0] for e in range(n_experts))
        per_layer_k285_mix.append(dict(c))
        per_layer_k285_mb.append(round(b / 1e6, 2))

    return {
        "experiment": "xray_strata_quant_moe",
        "model": args.model,
        "n_layers": n_layers,
        "n_experts_per_layer": n_experts,
        "total_expert_params": total_expert_params,
        "xray_quant_sensitivity_map": qmap,
        "per_layer_s2_err": [round(x, 6) for x in per_layer_s2_err],
        "per_layer_s32_err": [round(x, 6) for x in per_layer_s32_err],
        "per_layer_k285_mix": per_layer_k285_mix,
        "per_layer_k285_mb": per_layer_k285_mb,
        "per_layer_usage": per_layer_usage,
        "arms": arms_out,
        "timing_s": {"total_s": round(time.perf_counter() - t0, 2)},
        "criteria": {
            "a_uniform_s3s2_beats_rtn": arms_out["uniform_s3_s2_h64_gptq_flip"]["kl_mean"] < arms_out["uniform_rtn_3_2"]["kl_mean"],
            "b_per_expert_knapsack_beats_uniform": arms_out["per_expert_knapsack_2_85b"]["kl_mean"] <= arms_out["uniform_s3_s2_h64_gptq_flip"]["kl_mean"],
            "c_scale_qat_improves_kl": arms_out["per_expert_knapsack_2_85b_scale_qat"]["kl_mean"] < arms_out["per_expert_knapsack_2_85b"]["kl_mean"],
            "d_2_85b_mmlu_within_3pts_of_bf16": arms_out["per_expert_knapsack_2_85b_scale_qat"]["mmlu_200"]["accuracy"] >= base_mmlu["accuracy"] - 0.03,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0125-Instruct")
    ap.add_argument("--xray", default=str(DEFAULT_XRAY_PATH))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = run_smoke() if args.smoke else run_real(args)
    if args.out:
        out_p = _atomic_write_json(args.out, payload)
        print(f"Saved -> {out_p}", flush=True)
    _print_summary(payload)


if __name__ == "__main__":
    main()
