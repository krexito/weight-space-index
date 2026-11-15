"""Experiment: Cross-Architecture 2-Layer Compositional `AND` Gate (`A + B -> C`).

Demonstrates genuine multi-layer compositional computation over a private orthogonal residual
wire (`LFM3-1.7B 2048d` -> `GPT-2 768d`):
  * Layer 6 (`#2403`) detects preceding context `A` (Teacher Layer 0 triggers), invariant to the
    current token, and writes a private zero-mean orthogonal residual state `+w_wire`.
  * Layer 10 (`#630`) computes a literal thresholded conjunction `(w_wire AND B)`: its key sums
    matched-gain detectors for the private wire (`k_wire`) and the current token `B` (Teacher
    Layer 1 triggers, `k_B`), with bias calibrated so EITHER input alone remains in the dead zone
    of `GELU`, while `(A + B)` crosses threshold and writes Teacher Layer 1's output concept `C`.

Evaluated on 20 discovered `LFM3` 2-layer chains on held-out prompt templates and disjoint control
words across the full 4-state truth table `(AB, A_only, B_only, Ctrl_only)` plus causal wire ablations.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wsr"))

from wsi import WeightSpaceIndex
from wsi.compare import _get_shared_vocab_rosetta, discover_connected_chains

torch.manual_seed(0)
SRC_LAYER = 6
DST_LAYER = 10
N_CHAINS = 20
N_CTRL_POOL = 200
CALIB_PREFIX = "The"
HELDOUT_TEMPLATES = ["Then", "When", "While"]


def main() -> None:
    t0 = time.perf_counter()

    # 1. Discover 20 coupled 2-layer chains in LFM3-1.7B with disjoint A, B triggers and C outputs
    t_disc = time.perf_counter()
    lfm = WeightSpaceIndex.from_model(ROOT / "models" / "lfm3-1.7b")
    g2 = WeightSpaceIndex.from_model(ROOT / "models" / "gpt2")
    chains = discover_connected_chains(
        lfm, 0, 1, top_k=N_CHAINS, shared_vocab_with=g2, alphabetic_only=True
    )
    print(
        f"[LFM3-1.7B 2048d] Discovered {len(chains)} 2-layer compositional (A + B -> C) chains "
        f"in {time.perf_counter() - t_disc:.2f}s (0 Teacher forward passes)"
    )

    tok = GPT2Tokenizer.from_pretrained(str(ROOT / "models" / "gpt2"))
    model = GPT2LMHeadModel.from_pretrained(str(ROOT / "models" / "gpt2")).eval()
    _, I_s, shared, _, _ = _get_shared_vocab_rosetta(lfm, g2)
    shared_to_idx = {s: idx for idx, s in enumerate(shared)}

    # Collect union of A/B triggers and build a strictly disjoint control word pool
    trig_ab_union: set[int] = set()
    forbidden_ctrl: set[int] = set()
    for ch in chains:
        trig_ab_union.update(ch["src_trigger_shared_indices"][:10])
        trig_ab_union.update(ch["dst_trigger_shared_indices"][:10])
        forbidden_ctrl.update(ch["src_top50_shared_indices"])
        forbidden_ctrl.update(ch["dst_top50_shared_indices"])
        for out_tok in ch["dst_2hop_tokens"]:
            if out_tok in shared_to_idx:
                forbidden_ctrl.add(shared_to_idx[out_tok])

    alpha_ctrl_cands = [
        j for j, s in enumerate(shared)
        if j not in forbidden_ctrl and s.startswith("Ġ") and s[1:].isalpha() and len(s) >= 4
    ]
    perm = torch.randperm(len(alpha_ctrl_cands))
    ctrl_pool_shared = [alpha_ctrl_cands[p] for p in perm[:N_CTRL_POOL].tolist()]
    heldout_ctrl_shared = [alpha_ctrl_cands[p] for p in perm[N_CTRL_POOL : N_CTRL_POOL + 200].tolist()]
    c_ids = [int(I_s[j].item()) for j in ctrl_pool_shared]
    hc_ids = [int(I_s[j].item()) for j in heldout_ctrl_shared]

    # 2a. Calibrate GPT-2 L6 & L10 single-token pre-MLP bases on ('The' + tok)
    t_cal = time.perf_counter()
    d_hidden = int(model.config.n_embd)
    pool_shared = sorted(trig_ab_union | set(ctrl_pool_shared[:150]))
    pool_g_ids = I_s[pool_shared]
    p1 = tok.encode(CALIB_PREFIX, add_special_tokens=False)[0]
    p2 = tok.encode("In", add_special_tokens=False)[0]
    single_ids = torch.stack([torch.full_like(pool_g_ids, p1), pool_g_ids], dim=1)

    caps_src, caps_dst = [], []
    h_src = model.transformer.h[SRC_LAYER].ln_2.register_forward_hook(
        lambda m, i, o: caps_src.append(o[:, -1, :].detach())
    )
    h_dst = model.transformer.h[DST_LAYER].ln_2.register_forward_hook(
        lambda m, i, o: caps_dst.append(o[:, -1, :].detach())
    )
    with torch.no_grad():
        for st in range(0, single_ids.shape[0], 128):
            model(input_ids=single_ids[st : st + 128])
    xs_src_all = torch.cat(caps_src, dim=0)
    xs_dst_all = torch.cat(caps_dst, dim=0)
    xs_src_dict = {int(pool_g_ids[k].item()): xs_src_all[k] for k in range(len(pool_shared))}
    xs_dst_dict = {int(pool_g_ids[k].item()): xs_dst_all[k] for k in range(len(pool_shared))}

    # 2b. Position-2 w2 sweep pass across 120 control + all B triggers (captured at L6 and L10)
    w2_cal = c_ids[:6]
    w1_ctrl = c_ids[8:128]
    all_b_ids = sorted({int(I_s[j].item()) for ch in chains for j in ch["dst_trigger_shared_indices"][:5]})
    w2_pool = c_ids[:120] + all_b_ids
    b_id_to_col = {bid: 120 + i for i, bid in enumerate(all_b_ids)}
    sweep_rows = [[p, cf, w2] for p in (p1, p2) for cf in c_ids[:2] for w2 in w2_pool]
    caps_src.clear()
    caps_dst.clear()
    with torch.no_grad():
        for st in range(0, len(sweep_rows), 256):
            model(input_ids=torch.tensor(sweep_rows[st : st + 256], dtype=torch.long))
    X_sw6 = torch.cat(caps_src, dim=0).reshape(4, len(w2_pool), d_hidden)
    X_sw10 = torch.cat(caps_dst, dim=0).reshape(4, len(w2_pool), d_hidden)
    X_sw6_diff = X_sw6 - X_sw6[:, :120].mean(dim=1, keepdim=True)
    _, S6, Vh6 = torch.linalg.svd(X_sw6_diff[:, :120].reshape(-1, d_hidden), full_matrices=False)
    D6_comp = (S6[:120, None] * Vh6[:120]) / (120 ** 0.5)
    X_sw10_tr = X_sw10[:, :80].mean(dim=0)
    X_sw10_val = X_sw10[:, 80:120].reshape(-1, d_hidden)

    # 2c. 3-token shared control pass across 2 prefixes x 80 w1_ctrl x 4 w2_cal
    ctrl_rows = [[p, w1, w2] for p in (p1, p2) for w1 in w1_ctrl[:80] for w2 in w2_cal[:4]]
    caps_src.clear()
    caps_dst.clear()
    with torch.no_grad():
        for st in range(0, len(ctrl_rows), 256):
            model(input_ids=torch.tensor(ctrl_rows[st : st + 256], dtype=torch.long))
    Xc6_g = torch.cat(caps_src, dim=0).reshape(2, 80, 4, d_hidden)
    D6_p = (Xc6_g[0] - Xc6_g[1]).reshape(-1, d_hidden) / (320 ** 0.5)
    Xc6_m = Xc6_g.permute(1, 0, 2, 3).reshape(80, 8, d_hidden).mean(dim=1)
    Xc6_tr = Xc6_m[:56]
    Xc6_val = Xc6_g[:, 56:, :, :].reshape(-1, d_hidden)

    Xc10_g = torch.cat(caps_dst, dim=0).reshape(2, 80, 4, d_hidden)
    D10_w1 = (Xc10_g - Xc10_g.mean(dim=1, keepdim=True)).reshape(-1, d_hidden)
    _, S10, Vh10 = torch.linalg.svd(D10_w1, full_matrices=False)
    D10_comp = (S10[:96, None] * Vh10[:96]) / (640 ** 0.5)
    Xc10_val = Xc10_g[:, 56:].reshape(-1, d_hidden)

    # 2d. Batched per-chain positive context pass (2 prefixes x 5 A triggers x [4 ctrl w2 + 4 B w2] = 80 rows/chain)
    pos_rows: list[list[int]] = []
    for ch in chains:
        a_ids = [int(I_s[j].item()) for j in ch["src_trigger_shared_indices"][:5]]
        b_ids = [int(I_s[j].item()) for j in ch["dst_trigger_shared_indices"][:5]]
        w2_ch = w2_cal[:4] + b_ids[:4]
        for p in (p1, p2):
            for a_gid in a_ids:
                for w2_gid in w2_ch:
                    pos_rows.append([p, a_gid, w2_gid])
    caps_src.clear()
    caps_dst.clear()
    with torch.no_grad():
        for st in range(0, len(pos_rows), 256):
            model(input_ids=torch.tensor(pos_rows[st : st + 256], dtype=torch.long))
    h_src.remove()
    h_dst.remove()
    chain_xp6_all = torch.cat(caps_src, dim=0).reshape(len(chains), 2, 5, 8, d_hidden)
    chain_xp10_all = torch.cat(caps_dst, dim=0).reshape(len(chains), 2, 5, 8, d_hidden)
    print(
        f"Calibrated single-token ({len(pool_shared)} toks) + w2 sweep ({len(w2_pool)} toks) + "
        f"3-token context pools in {time.perf_counter() - t_cal:.2f}s\n"
    )

    mlp_s = model.transformer.h[SRC_LAYER].mlp
    mlp_d = model.transformer.h[DST_LAYER].mlp
    Wfc_s, bfc_s, Wpr_s = mlp_s.c_fc.weight.data, mlp_s.c_fc.bias.data, mlp_s.c_proj.weight.data
    Wfc_d, bfc_d, Wpr_d = mlp_d.c_fc.weight.data, mlp_d.c_fc.bias.data, mlp_d.c_proj.weight.data
    u_src = int((Wfc_s.norm(dim=0) * Wpr_s.norm(dim=1)).argmin().item())
    u_dst = int((Wfc_d.norm(dim=0) * Wpr_d.norm(dim=1)).argmin().item())
    orig_src = (Wfc_s[:, u_src].clone(), bfc_s[u_src].clone(), Wpr_s[u_src].clone())
    orig_dst = (Wfc_d[:, u_dst].clone(), bfc_d[u_dst].clone(), Wpr_d[u_dst].clone())
    print(f"GPT-2 dormant neurons: Layer {SRC_LAYER} #{u_src} (context A) -> Wire -> Layer {DST_LAYER} #{u_dst} (Wire AND B)\n")

    def restore_both() -> None:
        Wfc_s[:, u_src], bfc_s[u_src], Wpr_s[u_src] = orig_src
        Wfc_d[:, u_dst], bfc_d[u_dst], Wpr_d[u_dst] = orig_dst

    def eval_prompts(prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def make_pair_prompts(w1_list: list[str], w2_list: list[str]) -> list[str]:
        return [f"{tpl}{w1}{w2}" for tpl in HELDOUT_TEMPLATES for w1 in w1_list for w2 in w2_list]

    results = []
    for n, ch in enumerate(chains):
        b_ids = [int(I_s[j].item()) for j in ch["dst_trigger_shared_indices"][:5]]
        b_cols = [b_id_to_col[b] for b in b_ids]
        src_ctx_dict = {
            "pos_a_full": chain_xp6_all[n],
            "ctrl_tr": Xc6_tr,
            "ctrl_val": Xc6_val,
            "D6_comp": D6_comp,
            "D6_p": D6_p,
            "X_sw6_b_diff": X_sw6_diff[:, b_cols].reshape(-1, d_hidden),
        }
        dst_ctx_dict = {
            "Xb10_sw": X_sw10[:, b_cols],
            "Xp10_full": chain_xp10_all[n],
            "X_sw10_tr": X_sw10_tr,
            "X_sw10_val": X_sw10_val,
            "Xc10_val": Xc10_val,
            "D10_comp": D10_comp,
        }
        wire_seed = 100 + n

        tx = g2.transplant_compositional_and_chain_from(
            lfm,
            teacher_src_layer=0,
            teacher_src_unit=ch["src_unit"],
            teacher_dst_layer=1,
            teacher_dst_unit=ch["dst_unit"],
            student_src_layer=SRC_LAYER,
            student_dst_layer=DST_LAYER,
            student_src_unit=u_src,
            student_dst_unit=u_dst,
            wire_seed=wire_seed,
            student_src_pre_mlp_vecs=xs_src_dict,
            student_dst_pre_mlp_vecs=xs_dst_dict,
            student_src_context_vecs=src_ctx_dict,
            student_dst_context_vecs=dst_ctx_dict,
        )
        tx_broken = g2.transplant_compositional_and_chain_from(
            lfm,
            teacher_src_layer=0,
            teacher_src_unit=ch["src_unit"],
            teacher_dst_layer=1,
            teacher_dst_unit=ch["dst_unit"],
            student_src_layer=SRC_LAYER,
            student_dst_layer=DST_LAYER,
            student_src_unit=u_src,
            student_dst_unit=u_dst,
            wire_seed=wire_seed + 999,
            student_src_pre_mlp_vecs=xs_src_dict,
            student_dst_pre_mlp_vecs=xs_dst_dict,
            student_src_context_vecs=src_ctx_dict,
            student_dst_context_vecs=dst_ctx_dict,
        )

        T = tx["_tensors"]
        k_s_mat = T[f"h.{SRC_LAYER}.mlp.c_fc.weight"]
        b_s_vec = T[f"h.{SRC_LAYER}.mlp.c_fc.bias"]
        v_s_mat = T[f"h.{SRC_LAYER}.mlp.c_proj.weight"]
        k_d_mat = T[f"h.{DST_LAYER}.mlp.c_fc.weight"]
        b_d_vec = T[f"h.{DST_LAYER}.mlp.c_fc.bias"]
        v_d_mat = T[f"h.{DST_LAYER}.mlp.c_proj.weight"]

        Tb = tx_broken["_tensors"]
        k_d_broken = Tb[f"h.{DST_LAYER}.mlp.c_fc.weight"][:, u_dst]
        b_d_broken = float(Tb[f"h.{DST_LAYER}.mlp.c_fc.bias"][u_dst].item())

        a_words = [tok.decode([int(I_s[j].item())]) for j in ch["src_trigger_shared_indices"][:3]]
        b_words = [tok.decode([x]) for x in b_ids[:3]]
        out5_toks = ch["dst_2hop_tokens"][:5]
        out5_ids = [tok.convert_tokens_to_ids(t) for t in out5_toks]

        hc1 = [tok.decode([x]) for x in hc_ids[n * 6 : n * 6 + 3]]
        hc2 = [tok.decode([x]) for x in hc_ids[n * 6 + 3 : n * 6 + 6]]

        P_ab = make_pair_prompts(a_words, b_words)
        P_a_only = make_pair_prompts(a_words, hc1)
        P_b_only = make_pair_prompts(hc1, b_words)
        P_ctrl = make_pair_prompts(hc1, hc2)

        restore_both()
        _, _, base_ab = eval_prompts(P_ab)
        _, _, base_ao = eval_prompts(P_a_only)
        _, _, base_bo = eval_prompts(P_b_only)
        _, _, base_cc = eval_prompts(P_ctrl)

        def install_src(zero_wire: bool = False) -> None:
            Wfc_s[:, u_src] = k_s_mat[:, u_src]
            bfc_s[u_src] = b_s_vec[u_src]
            Wpr_s[u_src] = torch.zeros_like(v_s_mat[u_src]) if zero_wire else v_s_mat[u_src]

        def install_dst(k_col: torch.Tensor, b_val: float) -> None:
            Wfc_d[:, u_dst] = k_col
            bfc_d[u_dst] = b_val
            Wpr_d[u_dst] = v_d_mat[u_dst]

        # Full Compositional AND Gate truth table evaluation
        install_src(zero_wire=False)
        install_dst(k_d_mat[:, u_dst], float(b_d_vec[u_dst].item()))
        as_ab, ad_ab, lp_ab = eval_prompts(P_ab)
        as_ao, ad_ao, lp_ao = eval_prompts(P_a_only)
        as_bo, ad_bo, lp_bo = eval_prompts(P_b_only)
        as_cc, ad_cc, lp_cc = eval_prompts(P_ctrl)
        kl_ctrl = F.kl_div(lp_cc, base_cc, log_target=True, reduction="batchmean").item()
        restore_both()

        # Causal Ablation 1: L6 wire ablated on AB
        install_src(zero_wire=True)
        install_dst(k_d_mat[:, u_dst], float(b_d_vec[u_dst].item()))
        _, ad_ab_l6abl, lp_ab_l6abl = eval_prompts(P_ab)
        restore_both()

        # Causal Ablation 2: Broken orthogonal wire on AB
        install_src(zero_wire=False)
        install_dst(k_d_broken, b_d_broken)
        _, ad_ab_broken, lp_ab_broken = eval_prompts(P_ab)
        restore_both()

        metrics = {
            "L6_AB": as_ab.mean().item(),
            "L6_Aonly": as_ao.mean().item(),
            "L6_Bonly": as_bo.mean().item(),
            "L6_Ctrl": as_cc.mean().item(),
            "L10_AB": ad_ab.mean().item(),
            "L10_Aonly": ad_ao.mean().item(),
            "L10_Bonly": ad_bo.mean().item(),
            "L10_Ctrl": ad_cc.mean().item(),
            "dlogP_AB": (lp_ab[:, out5_ids] - base_ab[:, out5_ids]).mean().item(),
            "dlogP_Aonly": (lp_ao[:, out5_ids] - base_ao[:, out5_ids]).mean().item(),
            "dlogP_Bonly": (lp_bo[:, out5_ids] - base_bo[:, out5_ids]).mean().item(),
            "dlogP_Ctrl": (lp_cc[:, out5_ids] - base_cc[:, out5_ids]).mean().item(),
            "KL_Ctrl": kl_ctrl,
            "L10_L6abl": ad_ab_l6abl.mean().item(),
            "dlogP_L6abl": (lp_ab_l6abl[:, out5_ids] - base_ab[:, out5_ids]).mean().item(),
            "L10_BrokenWire": ad_ab_broken.mean().item(),
            "dlogP_BrokenWire": (lp_ab_broken[:, out5_ids] - base_ab[:, out5_ids]).mean().item(),
            "wire_cos": tx["student_and_gate"]["wire_max_target_vocab_cos"],
        }
        results.append((ch, metrics))

        a_show = [s.lstrip("Ġ") for s in a_words[:2]]
        b_show = [s.lstrip("Ġ") for s in b_words[:2]]
        c_show = [s.lstrip("Ġ") for s in out5_toks[:2]]
        print(
            f"[{n+1:2d}] L0#{ch['src_unit']:4d}+L1#{ch['dst_unit']:4d} "
            f"A={a_show} + B={b_show} -> C={c_show} | wire_cos={metrics['wire_cos']:.4f}"
        )
        print(
            f"     L6 (A det): AB={metrics['L6_AB']:+5.2f} Aonly={metrics['L6_Aonly']:+5.2f} "
            f"Bonly={metrics['L6_Bonly']:+5.2f} Ctrl={metrics['L6_Ctrl']:+5.2f} | KL_Ctrl={metrics['KL_Ctrl']:.4f}"
        )
        print(
            f"     L10 (AND)  : AB={metrics['L10_AB']:+5.2f} Aonly={metrics['L10_Aonly']:+5.2f} "
            f"Bonly={metrics['L10_Bonly']:+5.2f} Ctrl={metrics['L10_Ctrl']:+5.2f} | "
            f"Ablated L10={metrics['L10_L6abl']:+4.2f} Broken={metrics['L10_BrokenWire']:+4.2f}"
        )
        print(
            f"     dlogP(C)   : AB={metrics['dlogP_AB']:+.3f} Aonly={metrics['dlogP_Aonly']:+.3f} "
            f"Bonly={metrics['dlogP_Bonly']:+.3f} Ctrl={metrics['dlogP_Ctrl']:+.3f} | "
            f"L6abl={metrics['dlogP_L6abl']:+.3f} Broken={metrics['dlogP_BrokenWire']:+.3f}"
        )

    def mean_of(key: str) -> float:
        return sum(m[key] for _, m in results) / len(results)

    pass_l10_act = sum(
        m["L10_AB"] > 2.0 and m["L10_AB"] > max(m["L10_Aonly"], m["L10_Bonly"], m["L10_Ctrl"]) + 2.0
        for _, m in results
    )
    pass_dlogp_tt = sum(
        m["dlogP_AB"] > max(m["dlogP_Aonly"], m["dlogP_Bonly"], m["dlogP_Ctrl"]) + 1.5
        for _, m in results
    )
    pass_dlogp_abl = sum(
        m["dlogP_AB"] > max(m["dlogP_L6abl"], m["dlogP_BrokenWire"]) + 1.5
        for _, m in results
    )
    mean_kl = mean_of("KL_Ctrl")

    print("\n" + "=" * 96)
    print(f"COMPOSITIONAL AND GATE SUMMARY ({len(results)} chains, LFM3 L0+L1 -> GPT-2 L{SRC_LAYER}->Wire->L{DST_LAYER}, {time.perf_counter()-t0:.1f}s)")
    print("  Condition              | Mean L6_act | Mean L10_act | Mean d_logP(C)")
    print("  -----------------------+-------------+--------------+---------------")
    print(f"  (1,1) AB (A + B)       |    {mean_of('L6_AB'):+6.2f}   |     {mean_of('L10_AB'):+6.2f}   |    {mean_of('dlogP_AB'):+7.3f}")
    print(f"  (1,0) A_only (A + Ctrl)|    {mean_of('L6_Aonly'):+6.2f}   |     {mean_of('L10_Aonly'):+6.2f}   |    {mean_of('dlogP_Aonly'):+7.3f}")
    print(f"  (0,1) B_only (Ctrl + B)|    {mean_of('L6_Bonly'):+6.2f}   |     {mean_of('L10_Bonly'):+6.2f}   |    {mean_of('dlogP_Bonly'):+7.3f}")
    print(f"  (0,0) Ctrl   (C1 + C2) |    {mean_of('L6_Ctrl'):+6.2f}   |     {mean_of('L10_Ctrl'):+6.2f}   |    {mean_of('dlogP_Ctrl'):+7.3f}  (KL={mean_kl:.4f})")
    print(f"  AB w/ L6 Wire Ablated  |     (zero)  |     {mean_of('L10_L6abl'):+6.2f}   |    {mean_of('dlogP_L6abl'):+7.3f}")
    print(f"  AB w/ Broken Wire      |    {mean_of('L6_AB'):+6.2f}   |     {mean_of('L10_BrokenWire'):+6.2f}   |    {mean_of('dlogP_BrokenWire'):+7.3f}")
    print("  -----------------------+-------------+--------------+---------------")
    print(
        f"  Strict pass rates      : L10 AND act={pass_l10_act}/{len(results)} | "
        f"TruthTable dlogP={pass_dlogp_tt}/{len(results)} | "
        f"WireAblation dlogP={pass_dlogp_abl}/{len(results)} | Mean KL={mean_kl:.4f}"
    )
    print("=" * 96)

    assert pass_l10_act == len(results), f"Failed L10 AND activation margin on some chains: {pass_l10_act}/{len(results)}"
    assert pass_dlogp_tt == len(results), f"Failed Truth Table d_logP margin on some chains: {pass_dlogp_tt}/{len(results)}"
    assert pass_dlogp_abl == len(results), f"Failed Wire Ablation d_logP margin on some chains: {pass_dlogp_abl}/{len(results)}"
    assert mean_kl < 0.01, f"Control KL too high: {mean_kl:.5f}"


if __name__ == "__main__":
    main()
