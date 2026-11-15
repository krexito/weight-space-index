"""Experiment: Multi-Hop Residual Connection Tracking & Cross-Architecture 2-Layer Connected Circuit Transplant.

Part 1 — Zero-Pass Multi-Hop Connection Tracking (`trace_residual_connections` & `discover_connected_chains`):
  * On Qwen/Qwen3.5-0.8B (`p50c` Layer 15 principal delta direction): tracks resonant connections into
    Layer 15 Head 7 (`p25c` target) and Layer 19 MLP subspace from weights alone.
  * On LiquidAI/LFM3-1.7B (`2048d`): discovers 20 strongly coupled 2-layer chains
    `(Layer 0 Neuron A -> Layer 1 Neuron B)` purely from static weight matrices (`0` Teacher passes).

Part 2 — Live Cross-Architecture 2-Layer Connected Circuit Transplant (`LFM3 2048d -> GPT-2 768d`):
  * Transplants each LFM3 2-layer chain `(L0 #A -> L1 #B)` into GPT-2 `(Layer 6 #A' -> Private Wire -> Layer 10 #B')`.
  * L6 #A' detects the Teacher's L0 trigger and writes a private zero-mean orthogonal residual state `w_wire`
    (orthogonal to target vocabulary unembeddings and intermediate layer readers).
  * L10 #B' listens exclusively for `+w_wire` in the residual stream and writes the Teacher's L1 output vector.
  * Evaluates held-out prompt templates and disjoint control words across 4 causal ablations:
      1. Full 2-Layer Chain (`L6 -> wire -> L10`)
      2. L6 Sender Only (`L10 ablated`)
      3. L10 Receiver Only (`L6 ablated`)
      4. Broken Orthogonal Wire Control (`L6 sends wire_1, L10 listens for orthogonal wire_2`)
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wsr"))

from wsi import WeightSpaceIndex
from wsi.compare import (
    _build_private_residual_wire,
    _get_shared_vocab_rosetta,
    discover_connected_chains,
)

torch.manual_seed(0)
SRC_LAYER = 6
DST_LAYER = 10
N_CHAINS = 20
N_CTRL_POOL = 150
CALIB_PREFIX = "The"
TEMPLATES = ["{t}", "I saw the{t}", "Then{t}", "The word{t}"]


def _extract_qwen_p50c_l15_vector() -> torch.Tensor:
    """Reconstruct Layer 15 `p50c` principal delta vector u_15 from local slice + benchmark artifact."""
    exp_dir = ROOT / "benchmarks" / "exp8_math"
    frozen_gt = json.loads((exp_dir / "p50c_frozen_weights_manifest.json").read_text(encoding="utf-8"))
    p50c_man = json.loads((exp_dir / "p50c_manifest.json").read_text(encoding="utf-8"))
    p50c_z = np.load(exp_dir / "p50c_edit.npz")

    st_path = ROOT / "models" / "qwen35-0.8b" / "model.safetensors"
    with open(st_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        off = 8 + n
        info = head["model.layers.15.mlp.down_proj.weight"]
        a, b = info["data_offsets"]
        f.seek(off + a)
        base_bits = np.frombuffer(f.read(b - a), dtype=np.uint16).copy().reshape(info["shape"])

    base_f32 = (base_bits.astype(np.uint32) << 16).view(np.float32)
    t_idx = next(
        i for i, m in enumerate(p50c_man["tensors"]) if "layers.15.mlp.down_proj" in m["name"]
    )
    edit_bits = base_bits.copy().reshape(-1)
    edit_bits[p50c_z[f"t{t_idx}_idx"]] = p50c_z[f"t{t_idx}_val"]
    edit_f32 = (edit_bits.reshape(base_bits.shape).astype(np.uint32) << 16).view(np.float32)

    sel = frozen_gt["15"]["selected_neurons"]
    dW = (base_f32 - edit_f32)[:, sel]
    U_svd, _, _ = np.linalg.svd(dW, full_matrices=False)
    u15 = U_svd[:, 0]
    if np.dot(u15, dW[:, 0]) < 0:
        u15 = -u15
    return torch.from_numpy(u15.astype(np.float32))


def main() -> None:
    t0 = time.perf_counter()

    # =========================================================================
    # PART 1: ZERO-PASS MULTI-HOP CONNECTION TRACKING
    # =========================================================================
    print("=" * 92)
    print(" PART 1: ZERO-PASS MULTI-HOP RESIDUAL CONNECTION TRACKING")
    print("=" * 92)

    t_q0 = time.perf_counter()
    qwen = WeightSpaceIndex.from_model(ROOT / "models" / "qwen35-0.8b")
    u15 = _extract_qwen_p50c_l15_vector()
    qwen_trace = qwen.trace_residual_connections(vec=u15, top_k_connections=5, top_k_tokens=5)
    t_q_ms = (time.perf_counter() - t_q0) * 1000.0
    top_head = qwen_trace["top_coupled_attention_heads"][0]
    res_l19 = next((m for m in qwen_trace["resonant_mlp_subspaces"] if m["layer"] == 19), None)
    print(f"[Qwen3.5-0.8B p50c u_15] Multi-hop connection trace ({t_q_ms:.1f} ms, 0 forward passes):")
    print(f"  1-hop vocab readout : {[t['token'] for t in qwen_trace['direct_1hop_vocab']['promoted_tokens'][:4]]}")
    print(
        f"  Top coupled Head    : Layer {top_head['layer']} Head #{top_head['head']} "
        f"(energy={top_head['state_projection_energy']}, cols={top_head['col_range']})"
    )
    if res_l19:
        print(
            f"  Resonant MLP target : Layer 19 echo cos={res_l19['collective_echo_cosine']} "
            f"({res_l19['statistical_significance_sigma']} sigma)"
        )

    t_lfm0 = time.perf_counter()
    lfm = WeightSpaceIndex.from_model(ROOT / "models" / "lfm3-1.7b")
    g2 = WeightSpaceIndex.from_model(ROOT / "models" / "gpt2")
    chains = discover_connected_chains(
        lfm, 0, 1, top_k=N_CHAINS, shared_vocab_with=g2, alphabetic_only=True
    )
    # Also verify `trace_residual_connections(layer=0, unit=...)` on the top discovered chain
    sample_trace = lfm.trace_residual_connections(
        layer=0, unit=chains[0]["src_unit"], top_k_connections=5, top_k_tokens=5
    )
    t_lfm_s = time.perf_counter() - t_lfm0
    print(
        f"\n[LFM3-1.7B 2048d] Discovered {len(chains)} clean 2-layer chains (L0 -> L1) in {t_lfm_s:.2f}s "
        f"(0 Teacher forward passes):"
    )
    if sample_trace["two_hop_downstream_vocab"]:
        print(
            f"  `trace_residual_connections(L0 #{chains[0]['src_unit']})` "
            f"1-hop={[t['token'].lstrip('Ġ') for t in sample_trace['direct_1hop_vocab']['promoted_tokens'][:3]]} -> "
            f"2-hop={[t['token'].lstrip('Ġ') for t in sample_trace['two_hop_downstream_vocab']['promoted_tokens'][:4]]}"
        )
    for idx_c, ch in enumerate(chains[:8], 1):
        print(
            f"  [{idx_c:2d}] L0 #{ch['src_unit']:4d} -> L1 #{ch['dst_unit']:4d} "
            f"(coupling={ch['coupling_score']:5.1f}, sigma={ch['statistical_significance_sigma']:+5.2f}) | "
            f"in={[t.lstrip('Ġ') for t in ch['src_triggers'][:3]]} -> "
            f"1hop={[t.lstrip('Ġ') for t in ch['src_1hop_tokens'][:2]]} -> "
            f"2hop={[t.lstrip('Ġ') for t in ch['dst_2hop_tokens'][:3]]}"
        )
    if len(chains) > 8:
        print(f"  ... ({len(chains) - 8} more clean chains)")

    # =========================================================================
    # PART 2: LIVE 2-LAYER CONNECTED CIRCUIT TRANSPLANT (LFM3 -> GPT-2 L6->L10)
    # =========================================================================
    print("\n" + "=" * 92)
    print(f" PART 2: 2-LAYER CIRCUIT TRANSPLANT (LFM3 L0->L1 => GPT-2 L{SRC_LAYER}->Wire->L{DST_LAYER})")
    print("=" * 92)

    tok = GPT2Tokenizer.from_pretrained(str(ROOT / "models" / "gpt2"))
    model = GPT2LMHeadModel.from_pretrained(str(ROOT / "models" / "gpt2")).eval()

    _, I_s, shared, _, _ = _get_shared_vocab_rosetta(lfm, g2)

    # Collect union of top-10 triggers and top-50 exclusions directly from discovered chains
    trig10_union: set[int] = set()
    forbidden_ctrl: set[int] = set()
    chain_triggers: list[tuple[list[int], list[str]]] = []
    for ch in chains:
        t10 = ch["src_trigger_shared_indices"]
        t50 = ch["src_top50_shared_indices"]
        trig10_union.update(t10)
        forbidden_ctrl.update(t50)
        chain_triggers.append((t10[:5], ch["dst_2hop_tokens"][:5]))

    alpha_ctrl_cands = [
        j for j, s in enumerate(shared)
        if j not in forbidden_ctrl and s.startswith("Ġ") and s[1:].isalpha() and len(s) >= 4
    ]
    perm = torch.randperm(len(alpha_ctrl_cands))
    ctrl_pool_shared = [alpha_ctrl_cands[p] for p in perm[:N_CTRL_POOL].tolist()]
    heldout_ctrl_shared = [alpha_ctrl_cands[p] for p in perm[N_CTRL_POOL : N_CTRL_POOL + 200].tolist()]

    pool_shared = sorted(trig10_union | set(ctrl_pool_shared))
    pool_g_ids = I_s[pool_shared]
    prefix_id = tok.encode(CALIB_PREFIX, add_special_tokens=False)[0]
    calib_ids = torch.stack([torch.full_like(pool_g_ids, prefix_id), pool_g_ids], dim=1)

    t_cal = time.perf_counter()
    caps_src, caps_dst = [], []
    h_src = model.transformer.h[SRC_LAYER].ln_2.register_forward_hook(
        lambda m, i, o: caps_src.append(o[:, -1, :].detach())
    )
    h_dst = model.transformer.h[DST_LAYER].ln_2.register_forward_hook(
        lambda m, i, o: caps_dst.append(o[:, -1, :].detach())
    )
    with torch.no_grad():
        for st in range(0, calib_ids.shape[0], 128):
            model(input_ids=calib_ids[st : st + 128])
    h_src.remove()
    h_dst.remove()
    xs_src_all = torch.cat(caps_src, dim=0)
    xs_dst_all = torch.cat(caps_dst, dim=0)
    xs_src_dict = {int(pool_g_ids[k].item()): xs_src_all[k] for k in range(len(pool_shared))}
    xs_dst_dict = {int(pool_g_ids[k].item()): xs_dst_all[k] for k in range(len(pool_shared))}
    print(
        f"Calibrated GPT-2 L{SRC_LAYER} & L{DST_LAYER} pre-MLP bases on {len(pool_shared)} pool tokens "
        f"({CALIB_PREFIX!r}+tok) in {(time.perf_counter()-t_cal):.2f}s"
    )

    mlp_s = model.transformer.h[SRC_LAYER].mlp
    mlp_d = model.transformer.h[DST_LAYER].mlp
    Wfc_s, bfc_s, Wpr_s = mlp_s.c_fc.weight.data, mlp_s.c_fc.bias.data, mlp_s.c_proj.weight.data
    Wfc_d, bfc_d, Wpr_d = mlp_d.c_fc.weight.data, mlp_d.c_fc.bias.data, mlp_d.c_proj.weight.data
    u_src = int((Wfc_s.norm(dim=0) * Wpr_s.norm(dim=1)).argmin().item())
    u_dst = int((Wfc_d.norm(dim=0) * Wpr_d.norm(dim=1)).argmin().item())
    orig_src = (Wfc_s[:, u_src].clone(), bfc_s[u_src].clone(), Wpr_s[u_src].clone())
    orig_dst = (Wfc_d[:, u_dst].clone(), bfc_d[u_dst].clone(), Wpr_d[u_dst].clone())
    print(f"GPT-2 dormant neurons selected: Layer {SRC_LAYER} #{u_src} -> private wire -> Layer {DST_LAYER} #{u_dst}\n")

    def eval_chain_prompts(prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        groups: dict[int, list[tuple[int, list[int]]]] = {}
        for idx_p, p in enumerate(prompts):
            ids = tok.encode(p)
            groups.setdefault(len(ids), []).append((idx_p, ids))

        out_lp = torch.empty((len(prompts), model.config.vocab_size), dtype=torch.float32)
        out_a_s = torch.empty(len(prompts), dtype=torch.float32)
        out_a_d = torch.empty(len(prompts), dtype=torch.float32)
        b_s, b_d = [], []
        hs = mlp_s.act.register_forward_hook(lambda m, i, o: b_s.append(o[:, -1, u_src].detach()))
        hd = mlp_d.act.register_forward_hook(lambda m, i, o: b_d.append(o[:, -1, u_dst].detach()))
        with torch.no_grad():
            for _len, items in groups.items():
                batch_ids = torch.tensor([x[1] for x in items], dtype=torch.long)
                b_s.clear()
                b_d.clear()
                lp = F.log_softmax(model(input_ids=batch_ids).logits[:, -1, :], dim=-1)
                for k_pos, (orig_idx, _) in enumerate(items):
                    out_lp[orig_idx] = lp[k_pos]
                    out_a_s[orig_idx] = b_s[0][k_pos]
                    out_a_d[orig_idx] = b_d[0][k_pos]
        hs.remove()
        hd.remove()
        return out_a_s, out_a_d, out_lp

    def restore_both() -> None:
        Wfc_s[:, u_src], bfc_s[u_src], Wpr_s[u_src] = orig_src
        Wfc_d[:, u_dst], bfc_d[u_dst], Wpr_d[u_dst] = orig_dst

    # Precompute `Xs_dst_pool` tensor for orthogonal-wire control construction
    ctrl_pool_idx_in_all = [pool_shared.index(j) for j in ctrl_pool_shared]

    results = []
    for n, ch in enumerate(chains):
        tx = g2.transplant_connected_chain_from(
            lfm,
            teacher_src_layer=0,
            teacher_src_unit=ch["src_unit"],
            teacher_dst_layer=1,
            teacher_dst_unit=ch["dst_unit"],
            student_src_layer=SRC_LAYER,
            student_dst_layer=DST_LAYER,
            student_src_unit=u_src,
            student_dst_unit=u_dst,
            wire_seed=100 + n,
            student_src_pre_mlp_vecs=xs_src_dict,
            student_dst_pre_mlp_vecs=xs_dst_dict,
        )
        T = tx["_tensors"]
        k_s_mat = T[f"h.{SRC_LAYER}.mlp.c_fc.weight"]
        b_s_vec = T[f"h.{SRC_LAYER}.mlp.c_fc.bias"]
        v_s_mat = T[f"h.{SRC_LAYER}.mlp.c_proj.weight"]
        k_d_mat = T[f"h.{DST_LAYER}.mlp.c_fc.weight"]
        b_d_vec = T[f"h.{DST_LAYER}.mlp.c_fc.bias"]
        v_d_mat = T[f"h.{DST_LAYER}.mlp.c_proj.weight"]

        trig5_idx, out5_toks = chain_triggers[n]
        out5_ids = [tok.convert_tokens_to_ids(t) for t in out5_toks]

        # Build independent orthogonal receiver key on same L10 control subspace
        _, _, k_d_ortho, b_d_ortho = _build_private_residual_wire(
            g2,
            SRC_LAYER,
            DST_LAYER,
            xs_dst_all,
            ctrl_pool_idx_in_all,
            out5_ids,
            wire_scale=1.5,
            margin=3.0,
            seed=9000 + n,
        )

        off = (n * 7) % (len(heldout_ctrl_shared) - 20)
        eval_ctrl = heldout_ctrl_shared[off : off + 20]
        P_trig = [t.format(t=tok.convert_tokens_to_string([shared[j]])) for j in trig5_idx for t in TEMPLATES]
        P_ctrl = [t.format(t=tok.convert_tokens_to_string([shared[j]])) for j in eval_ctrl for t in TEMPLATES]

        restore_both()
        _, _, base_trig = eval_chain_prompts(P_trig)
        _, _, base_ctrl = eval_chain_prompts(P_ctrl)

        def install_src() -> None:
            Wfc_s[:, u_src] = k_s_mat[:, u_src]
            bfc_s[u_src] = b_s_vec[u_src]
            Wpr_s[u_src] = v_s_mat[u_src]

        def install_dst(k_col: torch.Tensor, b_val: float) -> None:
            Wfc_d[:, u_dst] = k_col
            bfc_d[u_dst] = b_val
            Wpr_d[u_dst] = v_d_mat[u_dst]

        conds = {}
        # 1. Full Connected Chain (L6 -> wire -> L10)
        install_src()
        install_dst(k_d_mat[:, u_dst], float(b_d_vec[u_dst].item()))
        as_t, ad_t, lp_t = eval_chain_prompts(P_trig)
        as_c, ad_c, lp_c = eval_chain_prompts(P_ctrl)
        conds["full"] = (
            as_t.mean().item(), ad_t.mean().item(), as_c.mean().item(), ad_c.mean().item(),
            (lp_t[:, out5_ids] - base_trig[:, out5_ids]).mean().item(),
            F.kl_div(lp_c, base_ctrl, log_target=True, reduction="batchmean").item(),
        )
        restore_both()

        # 2. L6 Sender Only (L10 ablated)
        install_src()
        as_t, ad_t, lp_t = eval_chain_prompts(P_trig)
        conds["l6_only"] = (
            as_t.mean().item(), ad_t.mean().item(),
            (lp_t[:, out5_ids] - base_trig[:, out5_ids]).mean().item(),
        )
        restore_both()

        # 3. L10 Receiver Only (L6 ablated)
        install_dst(k_d_mat[:, u_dst], float(b_d_vec[u_dst].item()))
        as_t, ad_t, lp_t = eval_chain_prompts(P_trig)
        conds["l10_only"] = (
            as_t.mean().item(), ad_t.mean().item(),
            (lp_t[:, out5_ids] - base_trig[:, out5_ids]).mean().item(),
        )
        restore_both()

        # 4. Broken Orthogonal Wire Control
        install_src()
        install_dst(k_d_ortho, b_d_ortho)
        as_t, ad_t, lp_t = eval_chain_prompts(P_trig)
        conds["ortho_wire"] = (
            as_t.mean().item(), ad_t.mean().item(),
            (lp_t[:, out5_ids] - base_trig[:, out5_ids]).mean().item(),
        )
        restore_both()

        results.append((ch, tx, conds))
        fc = conds["full"]
        s_only = conds["l6_only"]
        r_only = conds["l10_only"]
        ow = conds["ortho_wire"]
        print(
            f"[{n+1:2d}] L0 #{ch['src_unit']:4d} -> L1 #{ch['dst_unit']:4d} "
            f"in={[s.lstrip('Ġ') for s in ch['src_triggers'][:2]]} out={[s.lstrip('Ġ') for s in out5_toks[:2]]} | "
            f"max|cos(U_tgt,wire)|={tx['student_chain']['wire_max_target_vocab_cos']:.4f}"
        )
        print(
            f"     Full Chain : L6_act={fc[0]:+5.2f} -> L10_act={fc[1]:+6.2f} (ctrl L6={fc[2]:+.2f} L10={fc[3]:+.2f}) | "
            f"logP gain={fc[4]:+.3f} | ctrl KL={fc[5]:.4f}"
        )
        print(
            f"     Ablations  : L6-only gain={s_only[2]:+.3f} | "
            f"L10-only L10_act={r_only[1]:+.2f} gain={r_only[2]:+.3f} | "
            f"Ortho-wire L10_act={ow[1]:+.2f} gain={ow[2]:+.3f}"
        )

    def m_full(k: int) -> float:
        return sum(r[2]["full"][k] for r in results) / len(results)

    def m_cond(name: str, k: int) -> float:
        return sum(r[2][name][k] for r in results) / len(results)

    sel_l6 = sum(r[2]["full"][0] > r[2]["full"][2] for r in results)
    sel_l10 = sum(r[2]["full"][1] > r[2]["full"][3] for r in results)
    win_ortho = sum(r[2]["full"][4] > r[2]["ortho_wire"][2] for r in results)

    print("\n" + "=" * 92)
    print(f"SUMMARY over {len(results)} 2-layer connected circuits (LFM3 -> GPT-2 L{SRC_LAYER}->Wire->L{DST_LAYER}, {time.perf_counter()-t0:.1f}s)")
    print(f"  1. Full 2-Layer Chain : L6 sel={sel_l6}/{len(results)} (trig={m_full(0):+.2f} vs ctrl={m_full(2):+.2f}), "
          f"L10 sel={sel_l10}/{len(results)} (trig={m_full(1):+.2f} vs ctrl={m_full(3):+.2f})")
    print(f"                          mean logP(L1 writes) gain={m_full(4):+.3f} | ctrl KL={m_full(5):.4f}")
    print(f"  2. L6 Sender Only     : mean logP(L1 writes) gain={m_cond('l6_only', 2):+.3f} (wire alone has near-zero vocab effect)")
    print(f"  3. L10 Receiver Only  : L10_act={m_cond('l10_only', 1):+.2f} | mean logP gain={m_cond('l10_only', 2):+.3f} (silent without wire)")
    print(f"  4. Orthogonal Wire    : L10_act={m_cond('ortho_wire', 1):+.2f} | mean logP gain={m_cond('ortho_wire', 2):+.3f} | Full wins {win_ortho}/{len(results)}")
    print("=" * 92)

    assert sel_l6 == len(results) and sel_l10 == len(results) and win_ortho == len(results)


if __name__ == "__main__":
    main()
