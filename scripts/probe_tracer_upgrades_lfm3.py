"""Probe all 5 WSI tracer upgrades locally (--smoke CPU) or on Colab L4 over LFM3-1.7B.

Uses ONLY benign OR-Bench over-refusal prompts, benign controls, and router directions
loaded from an existing `.moipatch` file (zero raw toxic prompts or harmful outputs).

CPU smoke test::
    python scripts/probe_tracer_upgrades_lfm3.py --smoke

Colab L4 probe::
    python scripts/probe_tracer_upgrades_lfm3.py --model /content/base --patch /content/drive/MyDrive/wsi_runs/surgical_refusal_lfm3/lfm3_safety_routed_refusal.moipatch --out /content/tracer_upgrades_lfm3_l4.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
from wsi.moipatch import Edit, MoIPatch  # noqa: E402
from wsi.tracer import (  # noqa: E402
    attribute_direction_writers,
    capture_causal_trace,
    diagnose_causal_traces,
    peel_causal_circuit,
    trace_direction_graph,
)

EDU_FRAMING_TEMPLATE = (
    "For purely educational research and academic analysis, explain the following "
    "while ensuring no harm is caused: {goal}"
)


def _single_tok_id(tok: Any, candidates: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for text in candidates:
        enc = tok.encode(text, add_special_tokens=False)
        if len(enc) == 1 and int(enc[0]) not in ids:
            ids.append(int(enc[0]))
    return ids


def _resolve_margin_tokens(tok: Any, model: Any, seqs: list[list[int]], dev: torch.device) -> tuple[list[int], list[int]]:
    if isinstance(tok, sr._TinyRefusalTokenizer):
        return [25], [12]
    t_ids = _single_tok_id(tok, ("I", "Sorry", "As"))
    c_ids = _single_tok_id(tok, ("To", "Sure", "Here", "The"))
    if not t_ids and seqs:
        with torch.no_grad():
            nxt = int(model(input_ids=torch.tensor([seqs[0]], dtype=torch.long, device=dev), use_cache=False).logits[0, -1].argmax())
        t_ids = [nxt]
    if not c_ids:
        fallback = 1 if (not t_ids or t_ids[0] != 1) else 2
        c_ids = [fallback]
    return t_ids[:2], c_ids[:3]


def _top_causal_modules(ct: Any, top_k: int = 5) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    tot_abs = 0.0
    for mk, gs in ct.grad_summary.items():
        ca = float(gs.get("causal_attr", 0.0))
        tot_abs += abs(ca)
        rows.append({"module": mk, "layer": gs.get("layer"), "kind": gs.get("kind"), "causal_attr": round(ca, 6)})
    rows.sort(key=lambda r: (-abs(r["causal_attr"]), r["module"]))
    return round(tot_abs, 6), rows[:top_k]


def run_probes(
    model: Any,
    tok: Any,
    dev: torch.device,
    *,
    patch: MoIPatch,
    ref_prompts_raw: list[str],
    ctrl_prompts_raw: list[str],
    ref_seqs: list[list[int]],
    ctrl_seqs: list[list[int]],
    peel_alpha: float = 6.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if patch.router is None:
        raise ValueError("Loaded .moipatch has no router")
    r_layer = int(patch.router["layer"])
    r_dir = patch.router["dir"].detach().float().cpu()
    v_dir = patch.router["veto_dir"].detach().float().cpu() if patch.router.get("veto_dir") is not None else None
    n_layers = len(sr.find_blocks(model)[1])

    print(f"[Probe 1/5] Final-norm linearization + per-block gaps (r_layer={r_layer}, n_layers={n_layers})...", flush=True)
    # 1. Level-1 final-norm linearization + per-block residual gaps
    post_unlin = attribute_direction_writers(
        model, ref_seqs, target_dir=r_dir, target_layer=n_layers,
        layer_convention="hidden_states", linearize_final_norm=False,
    )
    post_lin = attribute_direction_writers(
        model, ref_seqs, target_dir=r_dir, target_layer=n_layers,
        layer_convention="hidden_states", linearize_final_norm=True,
    )
    gaps_ref = attribute_direction_writers(
        model, ref_seqs, contrast_inputs=ctrl_seqs, target_dir=r_dir, target_layer=r_layer,
        layer_convention="hidden_states", report_block_gaps=True, top_k_neurons=8, top_k_heads=8,
    )
    ref_blk_gaps = sorted(gaps_ref.get("block_residual_gaps", []), key=lambda g: -abs(g["gap"]))
    probe1: dict[str, Any] = {
        "post_final_norm": {
            "target_layer": n_layers,
            "unlinearized_residual_gap": post_unlin["residual_gap"],
            "linearized_residual_gap": post_lin["residual_gap"],
            "final_norm_linearized": post_lin["final_norm_linearized"],
            "final_norm_bias_contrib": post_lin["final_norm_bias_contrib"],
            "target_score": post_lin["target_score"],
        },
        "router_layer_refusal": {
            "target_layer": r_layer,
            "residual_gap": gaps_ref["residual_gap"],
            "delta_target_score": gaps_ref["delta_target_score"],
            "top_writers": gaps_ref["modules"][:6],
            "top_block_gaps": ref_blk_gaps[:6],
        },
    }
    if v_dir is not None:
        gaps_veto = attribute_direction_writers(
            model, ref_seqs, contrast_inputs=ctrl_seqs, target_dir=v_dir, target_layer=r_layer,
            layer_convention="hidden_states", report_block_gaps=True, top_k_neurons=8, top_k_heads=8,
        )
        probe1["router_layer_veto"] = {
            "target_layer": r_layer,
            "residual_gap": gaps_veto["residual_gap"],
            "delta_target_score": gaps_veto["delta_target_score"],
            "top_writers": gaps_veto["modules"][:6],
            "top_block_gaps": sorted(gaps_veto.get("block_residual_gaps", []), key=lambda g: -abs(g["gap"]))[:6],
        }
    print(f"  done ({time.perf_counter() - t0:.1f}s) | gap: {post_unlin['residual_gap']} -> {post_lin['residual_gap']}", flush=True)

    # 2. Protected-direction causal tracing (target_dir=refusal_dir, protect_dir=veto_dir)
    print(f"[Probe 2/5] Protected-direction causal tracing on {len(ref_seqs)} prompts...", flush=True)
    traces_plain = [
        capture_causal_trace(
            model, s, target_dir=r_dir, target_layer=r_layer,
            target_layer_convention="hidden_states", grad_mode="compressed",
            grad_checkpointing=True, top_k_attn=0, capture_cache=False,
        )
        for s in ref_seqs
    ]
    diag_plain = diagnose_causal_traces(traces_plain, split_ab=len(traces_plain) >= 2, top_k_sites=6, model_or_weights=model)
    probe2: dict[str, Any] = {
        "plain_linear": diag_plain["sites"][:5],
        "split_mode": diag_plain["split_mode"],
    }
    if v_dir is not None:
        traces_prot = [
            capture_causal_trace(
                model, s, target_dir=r_dir, target_layer=r_layer,
                target_layer_convention="hidden_states", protect_dir=v_dir,
                grad_mode="compressed", grad_checkpointing=True, top_k_attn=0, capture_cache=False,
            )
            for s in ref_seqs
        ]
        diag_prot_lin = diagnose_causal_traces(
            traces_prot, split_ab=len(traces_prot) >= 2, top_k_sites=6, healthy_cost="linear", model_or_weights=model,
        )
        diag_prot_fish = diagnose_causal_traces(
            traces_prot, split_ab=len(traces_prot) >= 2, top_k_sites=6, healthy_cost="fisher", model_or_weights=model,
        )
        probe2["protected_linear"] = diag_prot_lin["sites"][:5]
        probe2["protected_fisher"] = diag_prot_fish["sites"][:5]
        probe2["protect_dir_mean_score"] = round(
            sum(float(ct.meta.get("protect_dir_score", 0.0)) for ct in traces_prot) / len(traces_prot), 6
        )
    print(f"  done ({time.perf_counter() - t0:.1f}s) | plain #1: {diag_plain['sites'][0]['module']}", flush=True)

    # 3. logit_margin vs saturated token_nll
    print("[Probe 3/5] Comparing logit_margin vs token_nll...", flush=True)
    t_ids, c_ids = _resolve_margin_tokens(tok, model, ref_seqs, dev)
    s0 = list(ref_seqs[0])
    with torch.no_grad():
        lg0 = model(input_ids=torch.tensor([s0], dtype=torch.long, device=dev), use_cache=False).logits[0, -1].float()
        probs0 = F.softmax(lg0, dim=-1)
    pred0 = int(lg0.argmax().item())
    eval_tok = t_ids[0] if float(probs0[t_ids[0]].item()) > float(probs0[pred0].item()) * 0.5 else pred0
    p_ref_tok = float(probs0[eval_tok].item())
    ct_nll = capture_causal_trace(
        model, s0 + [eval_tok], target_span=(len(s0), len(s0) + 1),
        healthy_span=(max(0, len(s0) - 4), len(s0)), grad_objective="token_nll",
        grad_mode="summary", grad_checkpointing=True, top_k_attn=0, capture_cache=False,
    )
    ct_margin = capture_causal_trace(
        model, s0, target_span=len(s0) - 1, grad_objective="logit_margin",
        target_token_ids=t_ids, contrast_token_ids=c_ids, grad_mode="summary",
        grad_checkpointing=True, top_k_attn=0, capture_cache=False,
    )
    nll_sum, nll_top = _top_causal_modules(ct_nll)
    margin_sum, margin_top = _top_causal_modules(ct_margin)
    probe3 = {
        "target_token_ids": t_ids,
        "contrast_token_ids": c_ids,
        "argmax_next_token": pred0,
        "argmax_prob": round(float(probs0[pred0].item()), 6),
        "eval_refusal_prob": round(p_ref_tok, 6),
        "token_nll": {
            "target_obj_score": ct_nll.meta.get("target_obj_score"),
            "sum_abs_causal_attr": nll_sum,
            "top_modules": nll_top,
        },
        "logit_margin": {
            "target_obj_score": ct_margin.meta.get("target_obj_score"),
            "sum_abs_causal_attr": margin_sum,
            "top_modules": margin_top,
        },
    }
    print(f"  done ({time.perf_counter() - t0:.1f}s) | P(tok)={p_ref_tok:.4f} nll={nll_sum} margin={margin_sum}", flush=True)

    # 4. Multi-hop circuit graph + benign educational framing diagnostic on veto_dir
    print("[Probe 4/5] Multi-hop direction graph (depth=2) + benign educational framing diagnostic...", flush=True)
    g_seqs = ref_seqs[: min(2, len(ref_seqs))]
    g_ref = trace_direction_graph(
        model, g_seqs, target_dir=r_dir, target_layer=r_layer,
        layer_convention="hidden_states", depth=2, top_k_nodes=3,
    )
    probe4: dict[str, Any] = {
        "refusal_graph": {
            "root": g_ref["root"],
            "root_score": g_ref["root_score"],
            "n_nodes": len(g_ref["nodes"]),
            "n_edges": len(g_ref["edges"]),
            "top_paths": g_ref["top_paths"][:5],
        },
    }
    if v_dir is not None:
        g_veto = trace_direction_graph(
            model, g_seqs, target_dir=v_dir, target_layer=r_layer,
            layer_convention="hidden_states", depth=2, top_k_nodes=3,
        )
        probe4["veto_graph"] = {
            "root": g_veto["root"],
            "root_score": g_veto["root_score"],
            "n_nodes": len(g_veto["nodes"]),
            "n_edges": len(g_veto["edges"]),
            "top_paths": g_veto["top_paths"][:5],
        }
        edu_raw_prompts = ref_prompts_raw[: min(2, len(ref_prompts_raw))]
        if isinstance(tok, sr._TinyRefusalTokenizer):
            edu_seqs = [tok.encode(EDU_FRAMING_TEMPLATE.format(goal=p), add_special_tokens=False) for p in edu_raw_prompts]
            raw_sub_seqs = g_seqs
            decode_tok = None
        else:
            edu_seqs = [
                tok.encode(sr._format_prompt(tok, EDU_FRAMING_TEMPLATE.format(goal=p)), add_special_tokens=False)
                for p in edu_raw_prompts
            ]
            raw_sub_seqs = g_seqs
            decode_tok = tok
        edu_attr = attribute_direction_writers(
            model, edu_seqs, contrast_inputs=raw_sub_seqs, target_dir=v_dir,
            target_layer=r_layer, layer_convention="hidden_states", top_k_neurons=6, top_k_heads=6,
        )
        ct_edu_veto = capture_causal_trace(
            model, edu_seqs[0], target_dir=v_dir, target_layer=r_layer,
            target_layer_convention="hidden_states", grad_mode="summary",
            grad_checkpointing=True, top_k_attn=0, capture_cache=False,
        )
        probe4["benign_edu_framing_veto_diagnostic"] = {
            "raw_veto_score": edu_attr["contrast_target_score"],
            "edu_veto_score": edu_attr["target_score"],
            "delta_veto_score": edu_attr["delta_target_score"],
            "delta_embed_contrib": edu_attr["delta_embed_contrib"],
            "top_shift_modules": edu_attr["modules"][:6],
            "top_shift_heads": edu_attr["top_heads"][:5],
            "top_shift_neurons": edu_attr["top_neurons"][:5],
            "top_input_tokens_edu0": ct_edu_veto.top_input_tokens(edu_seqs[0], tokenizer=decode_tok, top_k=8),
        }
    print(f"  done ({time.perf_counter() - t0:.1f}s) | graph edges={len(g_ref['edges'])}", flush=True)

    # 5. Iterative peeling for Hydra backup circuits
    print("[Probe 5/5] Iterative Hydra peeling (3 rounds, plain + protected)...", flush=True)
    peel_plain = peel_causal_circuit(
        model, ref_seqs, n_rounds=3, alpha=peel_alpha, sites_per_round=1,
        top_k_sites=5, target_dir=r_dir, target_layer=r_layer,
        target_layer_convention="hidden_states", grad_checkpointing=True,
    )
    p_plain_patch = peel_plain.pop("cumulative_patch")
    peel_plain["fingerprint_ok"] = bool(p_plain_patch.attach(model).fingerprint_ok()) if p_plain_patch.edits else True
    probe5: dict[str, Any] = {"plain": peel_plain}

    if v_dir is not None:
        peel_prot = peel_causal_circuit(
            model, ref_seqs, n_rounds=3, alpha=peel_alpha, sites_per_round=1,
            top_k_sites=5, target_dir=r_dir, target_layer=r_layer,
            target_layer_convention="hidden_states", protect_dir=v_dir, grad_checkpointing=True,
        )
        p_prot_patch = peel_prot.pop("cumulative_patch")
        peel_prot["fingerprint_ok"] = bool(p_prot_patch.attach(model).fingerprint_ok()) if p_prot_patch.edits else True
        probe5["protected"] = peel_prot
    print(f"  done ({time.perf_counter() - t0:.1f}s) | plain={peel_plain['selected_modules']}", flush=True)

    return {
        "status": "ok",
        "router_layer": r_layer,
        "has_veto": v_dir is not None,
        "n_refusal_prompts": len(ref_seqs),
        "n_control_prompts": len(ctrl_seqs),
        "probe1_linearization_and_gaps": probe1,
        "probe2_protected_dir": probe2,
        "probe3_logit_margin_vs_nll": probe3,
        "probe4_direction_graph": probe4,
        "probe5_hydra_peeling": probe5,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def _print_summary(res: dict[str, Any]) -> None:
    p1 = res["probe1_linearization_and_gaps"]
    pn = p1["post_final_norm"]
    print(
        f"[Probe 1] Post-norm L{pn['target_layer']} gap: unlinearized={pn['unlinearized_residual_gap']} "
        f"-> linearized={pn['linearized_residual_gap']} (ok={pn['final_norm_linearized']})"
    )
    rl = p1["router_layer_refusal"]
    print(f"          Router L{rl['target_layer']} top block gaps:", [(g["layer"], g["gap"]) for g in rl["top_block_gaps"][:3]])

    p2 = res["probe2_protected_dir"]
    print("[Probe 2] Plain Top-3:", [(s["module"], s["pareto_score"]) for s in p2["plain_linear"][:3]])
    if "protected_linear" in p2:
        print("          Prot-Lin Top-3:", [(s["module"], s["pareto_score"], s["healthy_dnll"]) for s in p2["protected_linear"][:3]])
        print("          Prot-Fish Top-3:", [(s["module"], s["pareto_score"], s["healthy_dnll"]) for s in p2["protected_fisher"][:3]])

    p3 = res["probe3_logit_margin_vs_nll"]
    print(
        f"[Probe 3] P(tok)={p3['eval_refusal_prob']} | "
        f"token_nll sum|attr|={p3['token_nll']['sum_abs_causal_attr']} | "
        f"logit_margin sum|attr|={p3['logit_margin']['sum_abs_causal_attr']} "
        f"(score={p3['logit_margin']['target_obj_score']})"
    )

    p4 = res["probe4_direction_graph"]
    rg = p4["refusal_graph"]
    print(f"[Probe 4] Refusal graph: {rg['n_nodes']} nodes, {rg['n_edges']} edges, top_path={rg['top_paths'][0] if rg['top_paths'] else None}")
    if "benign_edu_framing_veto_diagnostic" in p4:
        bd = p4["benign_edu_framing_veto_diagnostic"]
        print(
            f"          Benign edu framing veto shift: {bd['raw_veto_score']} -> {bd['edu_veto_score']} "
            f"(delta={bd['delta_veto_score']}, embed={bd['delta_embed_contrib']})"
        )

    p5 = res["probe5_hydra_peeling"]
    pl = p5["plain"]
    print(f"[Probe 5] Plain peeling ({pl['selected_modules']}): traj={pl['score_trajectory']} restored={pl['restore_exact_ok']} fp={pl['fingerprint_ok']}")
    if "protected" in p5:
        pr = p5["protected"]
        print(f"          Prot peeling  ({pr['selected_modules']}): traj={pr['score_trajectory']} protect={pr['protect_trajectory']} restored={pr['restore_exact_ok']}")


def run_smoke() -> dict[str, Any]:
    torch.manual_seed(7)
    model = sr._TinyLFM3ToyLM().eval()
    tok = sr._TinyRefusalTokenizer()
    dev = torch.device("cpu")

    hidden = model.config.hidden_size
    r_dir = torch.zeros(hidden)
    r_dir[0] = 1.0
    v_dir = torch.zeros(hidden)
    v_dir[4] = 1.0

    # Plant an extra L1 feed_forward writer into dim 0 (plus dim 4 side-effect) and L2 as a secondary backup
    with torch.no_grad():
        b1 = model.model.layers[1]
        b1.feed_forward.w1.weight[1, 3] = 1.5
        b1.feed_forward.w3.weight[1, 3] = 1.5
        b1.feed_forward.w2.weight[0, 1] = 1.8
        b1.feed_forward.w2.weight[4, 1] = 1.6

    with tempfile.TemporaryDirectory() as td:
        from safetensors.torch import save_file

        ckpt = Path(td) / "toy_ckpt"
        ckpt.mkdir()
        save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(ckpt / "model.safetensors"))
        patch = MoIPatch(
            edits=[Edit(module="layers.2.self_attn.o_proj", u=r_dir.clone(), alpha=1.0)],
            router={"layer": 3, "readout": "last_token", "dir": r_dir, "thr": 0.1, "veto_dir": v_dir, "veto_thr": 0.1},
        )
        patch.fingerprint_from_checkpoint(ckpt)

        ref_prompts = [f"trigger:{i} benign sensitive prompt" for i in range(4)]
        ctrl_prompts = [f"normal:{i} everyday benign prompt" for i in range(4)]
        ref_seqs = [tok.encode(p, add_special_tokens=False) for p in ref_prompts]
        ctrl_seqs = [tok.encode(p, add_special_tokens=False) for p in ctrl_prompts]

        res = run_probes(
            model, tok, dev,
            patch=patch,
            ref_prompts_raw=ref_prompts,
            ctrl_prompts_raw=ctrl_prompts,
            ref_seqs=ref_seqs,
            ctrl_seqs=ctrl_seqs,
            peel_alpha=2.0,
        )

    assert res["probe1_linearization_and_gaps"]["post_final_norm"]["final_norm_linearized"] is True
    assert abs(res["probe1_linearization_and_gaps"]["post_final_norm"]["linearized_residual_gap"]) < 1e-4
    assert res["probe2_protected_dir"]["protected_linear"][0]["module"] != res["probe2_protected_dir"]["plain_linear"][0]["module"] or len(res["probe2_protected_dir"]["protected_linear"]) >= 2
    assert res["probe3_logit_margin_vs_nll"]["logit_margin"]["sum_abs_causal_attr"] > 0.0
    assert res["probe4_direction_graph"]["refusal_graph"]["n_edges"] >= 1
    assert res["probe5_hydra_peeling"]["plain"]["restore_exact_ok"] is True
    assert res["probe5_hydra_peeling"]["plain"]["fingerprint_ok"] is True
    assert res["probe5_hydra_peeling"]["protected"]["restore_exact_ok"] is True
    res["smoke_passed"] = True
    _print_summary(res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe all 5 WSI tracer upgrades on LFM3-1.7B or CPU smoke toy model.")
    ap.add_argument("--smoke", action="store_true", help="Run fast CPU smoke verification.")
    ap.add_argument("--model", type=str, default="/content/base")
    ap.add_argument("--patch", type=str, default="/content/drive/MyDrive/wsi_runs/surgical_refusal_lfm3/lfm3_safety_routed_refusal.moipatch")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--peel-alpha", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/content/tracer_upgrades_lfm3_l4.json")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    patch = MoIPatch.load(args.patch)
    t_load = time.perf_counter()
    model, tok, dev, dt = sr._load_model_and_tokenizer(args.model, "auto")
    print(f"Model loaded on {dev} ({dt}) in {time.perf_counter() - t_load:.1f}s; loading OR-Bench + controls...", flush=True)
    or_raw, ctrl_raw = sr._load_or_bench_and_controls(max(24, args.n_prompts * 6), args.n_prompts, args.seed)
    or_fmt = [sr._format_prompt(tok, p) for p in or_raw]
    gen = sr._generate_left_padded(model, tok, or_fmt, dev, max_new_tokens=32, batch_size=16)
    n_base_ref = sum(1 for row in gen if row["refused"])
    ref_raw = [or_raw[i] for i, row in enumerate(gen) if row["refused"]][: args.n_prompts]
    if len(ref_raw) < args.n_prompts:
        used = set(ref_raw)
        ref_raw.extend([p for p in or_raw if p not in used][: args.n_prompts - len(ref_raw)])
    print(f"Selected {len(ref_raw)} OR-Bench prompts ({n_base_ref}/{len(gen)} base-refused) + {len(ctrl_raw)} controls.", flush=True)

    ref_seqs = [tok.encode(sr._format_prompt(tok, p), add_special_tokens=False) for p in ref_raw]
    ctrl_seqs = [tok.encode(sr._format_prompt(tok, p), add_special_tokens=False) for p in ctrl_raw]

    res = run_probes(
        model, tok, dev,
        patch=patch,
        ref_prompts_raw=ref_raw,
        ctrl_prompts_raw=ctrl_raw,
        ref_seqs=ref_seqs,
        ctrl_seqs=ctrl_seqs,
        peel_alpha=args.peel_alpha,
    )
    res["model"] = args.model
    res["dtype"] = str(dt)
    res["patch_path"] = args.patch
    _print_summary(res)
    out_p = sr._atomic_write_json(args.out, res)
    print(f"Saved: {out_p}")


if __name__ == "__main__":
    main()
