"""Experiment: Dynamic On-The-Go Weight Paging (Just-In-Time Circuit MoE).

Tests on real weights (`LFM3-1.7B 2048d` -> `GPT-2 768d`) whether an indexed library of
20 cross-architecture compositional `AND` circuits (`A + B -> C`) can share ONE single
dormant neuron slot pair (`Layer 6 #2403 -> Wire -> Layer 10 #630`, leaving 99.99% of
GPT-2 completely frozen) via O(1) trigger-token routing and on-the-fly 12 KB weight paging
hidden behind Layers 0..5 execution:
  * Compares (1) Static Superposition in the single slot pair vs. (2) Dynamic Weight Paging
    (`DynamicCircuitPager`) across all 20 circuits on the full held-out truth table
    `(AB, A_only, B_only, Ctrl)` plus `CrossChain (A_i + B_j)` interference controls.
  * Measures per-prompt route+swap latency (`ms`) against GPT-2 Layers `0..5` compute (`ms`),
    alongside theoretical `1 Gbps` / `10 Gbps` optical-fiber transfer times for the 12.01 KB payload.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


def _time_l0_l5_forward(model: GPT2LMHeadModel, input_ids: torch.Tensor, src_layer: int) -> float:
    """Measure CPU wall-clock milliseconds for GPT-2 embeddings + Layers 0..src_layer-1."""
    t0 = time.perf_counter()
    with torch.no_grad():
        tr = model.transformer
        pos = torch.arange(0, input_ids.shape[1], dtype=torch.long)
        hidden = tr.drop(tr.wte(input_ids) + tr.wpe(pos))
        for blk in tr.h[:src_layer]:
            out_blk = blk(hidden)
            hidden = out_blk[0] if isinstance(out_blk, tuple) else out_blk
    return (time.perf_counter() - t0) * 1e3


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
    torch.set_grad_enabled(False)
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
        f"3-token context pools in {time.perf_counter() - t_cal:.2f}s"
    )

    mlp_s = model.transformer.h[SRC_LAYER].mlp
    mlp_d = model.transformer.h[DST_LAYER].mlp
    Wfc_s, bfc_s, Wpr_s = mlp_s.c_fc.weight.data, mlp_s.c_fc.bias.data, mlp_s.c_proj.weight.data
    Wfc_d, bfc_d, Wpr_d = mlp_d.c_fc.weight.data, mlp_d.c_fc.bias.data, mlp_d.c_proj.weight.data
    u_src = int((Wfc_s.norm(dim=0) * Wpr_s.norm(dim=1)).argmin().item())
    u_dst = int((Wfc_d.norm(dim=0) * Wpr_d.norm(dim=1)).argmin().item())
    orig_src = (Wfc_s[:, u_src].clone(), bfc_s[u_src].clone(), Wpr_s[u_src].clone())
    orig_dst = (Wfc_d[:, u_dst].clone(), bfc_d[u_dst].clone(), Wpr_d[u_dst].clone())
    print(f"Shared dormant GPT-2 slot pair: Layer {SRC_LAYER} #{u_src} -> Wire -> Layer {DST_LAYER} #{u_dst} (99.99% frozen)\n")

    def restore_both() -> None:
        Wfc_s[:, u_src], bfc_s[u_src], Wpr_s[u_src] = orig_src
        Wfc_d[:, u_dst], bfc_d[u_dst], Wpr_d[u_dst] = orig_dst

    def install_payload(payload: dict[str, torch.Tensor | float]) -> None:
        Wfc_s[:, u_src].copy_(payload["key_src"])
        bfc_s[u_src] = float(payload["bias_src"])
        Wpr_s[u_src].copy_(payload["val_src"])
        Wfc_d[:, u_dst].copy_(payload["key_dst"])
        bfc_d[u_dst] = float(payload["bias_dst"])
        Wpr_d[u_dst].copy_(payload["val_dst"])

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

    # 3. Compile all 20 circuits into DynamicCircuitPager targeting the single slot pair
    pager = g2.create_dynamic_circuit_pager(
        src_layer=SRC_LAYER,
        dst_layer=DST_LAYER,
        src_unit=u_src,
        dst_unit=u_dst,
    )
    used_a_sids: set[int] = set()
    used_b_sids: set[int] = set()
    chain_records: list[dict[str, object]] = []
    t_compile = time.perf_counter()

    for n, ch in enumerate(chains):
        a_cand = [int(I_s[j].item()) for j in ch["src_trigger_shared_indices"][:10]]
        b_cand = [int(I_s[j].item()) for j in ch["dst_trigger_shared_indices"][:10]]
        b_ids_cal = b_cand[:5]
        b_cols = [b_id_to_col[b] for b in b_ids_cal]

        # Select 3 unique role-disjoint Student token IDs for O(1) routing while preserving exact calibration triggers
        a_uniq: list[int] = []
        for sid in a_cand:
            if sid not in used_a_sids and sid not in a_uniq:
                a_uniq.append(sid)
                if len(a_uniq) == 3:
                    break
        if len(a_uniq) < 3:
            for sid in a_cand:
                if sid not in a_uniq:
                    a_uniq.append(sid)
                    if len(a_uniq) == 3:
                        break
        b_uniq: list[int] = []
        for sid in b_cand:
            if sid not in used_b_sids and sid not in b_uniq:
                b_uniq.append(sid)
                if len(b_uniq) == 3:
                    break
        if len(b_uniq) < 3:
            for sid in b_cand:
                if sid not in b_uniq:
                    b_uniq.append(sid)
                    if len(b_uniq) == 3:
                        break
        used_a_sids.update(a_uniq)
        used_b_sids.update(b_uniq)

        # Held-out prompt trigger tokens (use calibrated top-3 just like experiment_compositional_and_circuit.py)
        a_prompt_ids = [int(I_s[j].item()) for j in ch["src_trigger_shared_indices"][:3]]
        b_prompt_ids = [int(I_s[j].item()) for j in ch["dst_trigger_shared_indices"][:3]]
        # Prefer unique A/B subset when already inside calibrated top-5 (guarantees clean separation on prompt tokens too)
        if all(x in a_cand[:5] for x in a_uniq):
            a_prompt_ids = a_uniq
        if all(x in b_cand[:5] for x in b_uniq):
            b_prompt_ids = b_uniq

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
            wire_seed=100 + n,
            student_src_pre_mlp_vecs=xs_src_dict,
            student_dst_pre_mlp_vecs=xs_dst_dict,
            student_src_context_vecs=src_ctx_dict,
            student_dst_context_vecs=dst_ctx_dict,
        )

        a_words = [tok.decode([x]) for x in a_prompt_ids]
        b_words = [tok.decode([x]) for x in b_prompt_ids]
        cid = f"chain_{n:02d}"
        pager.register_circuit(
            cid,
            tx,
            a_token_ids=a_prompt_ids,
            b_token_ids=b_prompt_ids,
            a_tokens=a_words,
            b_tokens=b_words,
            metadata={"teacher_src_unit": ch["src_unit"], "teacher_dst_unit": ch["dst_unit"]},
        )

        out5_toks = ch["dst_2hop_tokens"][:5]
        out5_ids = [tok.convert_tokens_to_ids(t) for t in out5_toks]
        hc1 = [tok.decode([x]) for x in hc_ids[n * 6 : n * 6 + 3]]
        hc2 = [tok.decode([x]) for x in hc_ids[n * 6 + 3 : n * 6 + 6]]
        forb_ids = {int(I_s[j].item()) for j in (set(ch["src_top50_shared_indices"]) | set(ch["dst_top50_shared_indices"]))}

        chain_records.append(
            {
                "n": n,
                "cid": cid,
                "ch": ch,
                "a_words": a_words,
                "b_words": b_words,
                "a_ids": a_prompt_ids,
                "b_ids": b_prompt_ids,
                "forb_ids": forb_ids,
                "hc1": hc1,
                "hc2": hc2,
                "out5_toks": out5_toks,
                "out5_ids": out5_ids,
            }
        )

    compile_s = time.perf_counter() - t_compile
    cat = pager.catalog_summary(bandwidth_gbps=(1.0, 10.0))
    print(
        f"Compiled & indexed {cat['n_circuits']} circuits in {compile_s:.2f}s | "
        f"Slot payload: {cat['per_circuit_bytes']:,} B ({cat['per_circuit_kb']:.2f} KB) | "
        f"Catalog: {cat['total_catalog_kb']:.2f} KB | "
        f"Optical fiber: {cat['wire_transfer_ms']['1Gbps']:.4f} ms @ 1 Gbps, "
        f"{cat['wire_transfer_ms']['10Gbps']:.4f} ms @ 10 Gbps"
    )

    # 4. Build per-chain CrossChain partners & held-out prompt sets + capture baseline GPT-2 logits ONCE
    restore_both()
    for n, rec in enumerate(chain_records):
        b_self = set(rec["b_ids"])
        forb = rec["forb_ids"]
        partner_idx = (n + 1) % N_CHAINS
        for step in range(1, N_CHAINS):
            cand_m = (n + step) % N_CHAINS
            cand_b = set(chain_records[cand_m]["b_ids"])
            if not (cand_b & b_self) and not (cand_b & forb):
                partner_idx = cand_m
                break
        rec["partner_idx"] = partner_idx
        b_partner_words = chain_records[partner_idx]["b_words"]

        rec["P_ab"] = make_pair_prompts(rec["a_words"], rec["b_words"])
        rec["P_a_only"] = make_pair_prompts(rec["a_words"], rec["hc1"])
        rec["P_b_only"] = make_pair_prompts(rec["hc1"], rec["b_words"])
        rec["P_ctrl"] = make_pair_prompts(rec["hc1"], rec["hc2"])
        rec["P_cross"] = make_pair_prompts(rec["a_words"], b_partner_words)

        _, _, rec["base_ab"] = eval_prompts(rec["P_ab"])
        _, _, rec["base_ao"] = eval_prompts(rec["P_a_only"])
        _, _, rec["base_bo"] = eval_prompts(rec["P_b_only"])
        _, _, rec["base_cc"] = eval_prompts(rec["P_ctrl"])
        _, _, rec["base_cx"] = eval_prompts(rec["P_cross"])

    # 5. Evaluate Static Superposition (all 20 circuits crammed into 1 slot pair simultaneously)
    static_payload = pager.build_static_superposition_payload(normalize="sqrt_n")
    install_payload(static_payload)
    static_results = []
    for rec in chain_records:
        o5 = rec["out5_ids"]
        as_ab, ad_ab, lp_ab = eval_prompts(rec["P_ab"])
        as_ao, ad_ao, lp_ao = eval_prompts(rec["P_a_only"])
        as_bo, ad_bo, lp_bo = eval_prompts(rec["P_b_only"])
        as_cc, ad_cc, lp_cc = eval_prompts(rec["P_ctrl"])
        as_cx, ad_cx, lp_cx = eval_prompts(rec["P_cross"])
        static_results.append(
            {
                "L10_AB": ad_ab.mean().item(),
                "L10_Aonly": ad_ao.mean().item(),
                "L10_Bonly": ad_bo.mean().item(),
                "L10_Ctrl": ad_cc.mean().item(),
                "L10_Cross": ad_cx.mean().item(),
                "dlogP_AB": (lp_ab[:, o5] - rec["base_ab"][:, o5]).mean().item(),
                "dlogP_Aonly": (lp_ao[:, o5] - rec["base_ao"][:, o5]).mean().item(),
                "dlogP_Bonly": (lp_bo[:, o5] - rec["base_bo"][:, o5]).mean().item(),
                "dlogP_Ctrl": (lp_cc[:, o5] - rec["base_cc"][:, o5]).mean().item(),
                "dlogP_Cross": (lp_cx[:, o5] - rec["base_cx"][:, o5]).mean().item(),
                "KL_Ctrl": F.kl_div(lp_cc, rec["base_cc"], log_target=True, reduction="batchmean").item(),
            }
        )

    # 6. Evaluate Dynamic Weight Paging (route_token_ids + on-the-go 12 KB weight swap)
    # Also time per-prompt route + swap vs GPT-2 Layers 0..5 execution
    route_swap_ms: list[float] = []
    ab_route_exact = 0
    ab_route_total = 0
    cross_route_false_both = 0
    cross_route_total = 0

    def eval_dynamic_condition(prompts: list[str], cond_key: str, expected_cid: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal ab_route_exact, ab_route_total, cross_route_false_both, cross_route_total
        n_p = len(prompts)
        out_lp = torch.empty((n_p, model.config.vocab_size), dtype=torch.float32)
        out_as = torch.empty(n_p, dtype=torch.float32)
        out_ad = torch.empty(n_p, dtype=torch.float32)

        by_cid: dict[str, list[int]] = {}
        for idx_p, p in enumerate(prompts):
            ids = tok.encode(p)
            t_r0 = time.perf_counter()
            decision = pager.route_token_ids(ids)
            cid_sel = str(decision["circuit_id"])
            install_payload(pager.get_payload(cid_sel))
            route_swap_ms.append((time.perf_counter() - t_r0) * 1e3)
            by_cid.setdefault(cid_sel, []).append(idx_p)
            if cond_key == "AB":
                ab_route_total += 1
                if decision["matched"] and cid_sel == expected_cid:
                    ab_route_exact += 1
            elif cond_key == "CrossChain":
                cross_route_total += 1
                if decision["matched"]:
                    cross_route_false_both += 1

        for cid_sel, idx_list in by_cid.items():
            install_payload(pager.get_payload(cid_sel))
            sub_p = [prompts[k] for k in idx_list]
            sub_as, sub_ad, sub_lp = eval_prompts(sub_p)
            for pos_k, orig_k in enumerate(idx_list):
                out_as[orig_k] = sub_as[pos_k]
                out_ad[orig_k] = sub_ad[pos_k]
                out_lp[orig_k] = sub_lp[pos_k]
        return out_as, out_ad, out_lp

    dyn_results = []
    for rec in chain_records:
        o5 = rec["out5_ids"]
        cid = str(rec["cid"])
        ch = rec["ch"]
        as_ab, ad_ab, lp_ab = eval_dynamic_condition(rec["P_ab"], "AB", cid)
        as_ao, ad_ao, lp_ao = eval_dynamic_condition(rec["P_a_only"], "A_only", cid)
        as_bo, ad_bo, lp_bo = eval_dynamic_condition(rec["P_b_only"], "B_only", cid)
        as_cc, ad_cc, lp_cc = eval_dynamic_condition(rec["P_ctrl"], "Ctrl", cid)
        as_cx, ad_cx, lp_cx = eval_dynamic_condition(rec["P_cross"], "CrossChain", cid)
        kl_ctrl = F.kl_div(lp_cc, rec["base_cc"], log_target=True, reduction="batchmean").item()

        m = {
            "L6_AB": as_ab.mean().item(),
            "L6_Aonly": as_ao.mean().item(),
            "L6_Bonly": as_bo.mean().item(),
            "L6_Ctrl": as_cc.mean().item(),
            "L6_Cross": as_cx.mean().item(),
            "L10_AB": ad_ab.mean().item(),
            "L10_Aonly": ad_ao.mean().item(),
            "L10_Bonly": ad_bo.mean().item(),
            "L10_Ctrl": ad_cc.mean().item(),
            "L10_Cross": ad_cx.mean().item(),
            "dlogP_AB": (lp_ab[:, o5] - rec["base_ab"][:, o5]).mean().item(),
            "dlogP_Aonly": (lp_ao[:, o5] - rec["base_ao"][:, o5]).mean().item(),
            "dlogP_Bonly": (lp_bo[:, o5] - rec["base_bo"][:, o5]).mean().item(),
            "dlogP_Ctrl": (lp_cc[:, o5] - rec["base_cc"][:, o5]).mean().item(),
            "dlogP_Cross": (lp_cx[:, o5] - rec["base_cx"][:, o5]).mean().item(),
            "KL_Ctrl": kl_ctrl,
        }
        dyn_results.append((ch, rec, m))

        a_show = [s.lstrip("Ġ") for s in rec["a_words"][:2]]
        b_show = [s.lstrip("Ġ") for s in rec["b_words"][:2]]
        c_show = [s.lstrip("Ġ") for s in rec["out5_toks"][:2]]
        print(
            f"[{rec['n']+1:2d}] L0#{ch['src_unit']:4d}+L1#{ch['dst_unit']:4d} "
            f"A={a_show} + B={b_show} -> C={c_show} | L10_act AB={m['L10_AB']:+5.2f} "
            f"(max neg={max(m['L10_Aonly'], m['L10_Bonly'], m['L10_Ctrl'], m['L10_Cross']):+4.2f}) | "
            f"dlogP AB={m['dlogP_AB']:+.3f} A={m['dlogP_Aonly']:+.3f} B={m['dlogP_Bonly']:+.3f} "
            f"Ctrl={m['dlogP_Ctrl']:+.3f} Cross={m['dlogP_Cross']:+.3f} | KL={m['KL_Ctrl']:.4f}"
        )

    # Measure GPT-2 Layers 0..5 compute window vs interleaved B=1 hot-swaps
    interleaved_prompts = [chain_records[(i * 7) % N_CHAINS]["P_ab"][i % 27] for i in range(40)]
    warm_ids = torch.tensor([tok.encode(interleaved_prompts[0])], dtype=torch.long)
    _ = _time_l0_l5_forward(model, warm_ids, SRC_LAYER)
    l0_l5_ms_list: list[float] = []
    interleaved_swap_ms_list: list[float] = []
    for p in interleaved_prompts:
        ids_list = tok.encode(p)
        t_sw = time.perf_counter()
        dec = pager.route_token_ids(ids_list)
        install_payload(pager.get_payload(str(dec["circuit_id"])))
        interleaved_swap_ms_list.append((time.perf_counter() - t_sw) * 1e3)
        l0_l5_ms_list.append(_time_l0_l5_forward(model, torch.tensor([ids_list], dtype=torch.long), SRC_LAYER))

    mean_route_swap_ms = sum(route_swap_ms) / len(route_swap_ms)
    mean_interleaved_swap_ms = sum(interleaved_swap_ms_list) / len(interleaved_swap_ms_list)
    mean_l0_l5_ms = sum(l0_l5_ms_list) / len(l0_l5_ms_list)

    def _pass_l10(rows: list[dict[str, float]]) -> int:
        return sum(
            m["L10_AB"] > 2.0 and m["L10_AB"] > max(m["L10_Aonly"], m["L10_Bonly"], m["L10_Ctrl"], m["L10_Cross"]) + 2.0
            for m in rows
        )

    def _pass_tt(rows: list[dict[str, float]]) -> int:
        return sum(
            m["dlogP_AB"] > max(m["dlogP_Aonly"], m["dlogP_Bonly"], m["dlogP_Ctrl"], m["dlogP_Cross"]) + 1.5
            for m in rows
        )

    static_pass_l10_act = _pass_l10(static_results)
    static_pass_dlogp_tt = _pass_tt(static_results)
    static_mean_kl = sum(m["KL_Ctrl"] for m in static_results) / len(static_results)
    static_mean_ab_dlogp = sum(m["dlogP_AB"] for m in static_results) / len(static_results)

    dyn_metrics = [m for _, _, m in dyn_results]
    pass_l10_act = _pass_l10(dyn_metrics)
    pass_dlogp_tt = _pass_tt(dyn_metrics)
    mean_kl = sum(m["KL_Ctrl"] for m in dyn_metrics) / len(dyn_metrics)

    def _mean_dyn(k: str) -> float:
        return sum(m[k] for m in dyn_metrics) / len(dyn_metrics)

    def _mean_stat(k: str) -> float:
        return sum(m[k] for m in static_results) / len(static_results)

    print("\n" + "=" * 100)
    print(
        f"DYNAMIC WEIGHT PAGING SUMMARY ({N_CHAINS} circuits in 1 GPT-2 slot pair L{SRC_LAYER}#{u_src}->L{DST_LAYER}#{u_dst}, "
        f"{cat['per_circuit_kb']:.2f} KB/circuit, {time.perf_counter() - t0:.1f}s)"
    )
    print(f"  O(1) Routing Accuracy (AB): {ab_route_exact}/{ab_route_total} | CrossChain False-Both: {cross_route_false_both}/{cross_route_total}")
    print(
        f"  Latency Hiding (B=1): Route+Swap = {mean_route_swap_ms:.4f} ms (interleaved {mean_interleaved_swap_ms:.4f} ms) "
        f"< Layers 0..5 = {mean_l0_l5_ms:.3f} ms ({mean_l0_l5_ms / max(mean_route_swap_ms, 1e-6):.0f}x headroom)"
    )
    print(
        f"  Catalog Footprint: {cat['per_circuit_bytes']:,} B ({cat['per_circuit_kb']:.2f} KB)/skill | "
        f"{cat['total_catalog_kb']:.2f} KB total | 1 Gbps={cat['wire_transfer_ms']['1Gbps']:.4f}ms | 10 Gbps={cat['wire_transfer_ms']['10Gbps']:.4f}ms"
    )
    print("  Condition              | Dynamic L10_act | Dynamic d_logP | Static Superpos d_logP")
    print("  -----------------------+-----------------+----------------+-----------------------")
    print(f"  (1,1) AB (A_i + B_i)   |     {_mean_dyn('L10_AB'):+6.2f}      |     {_mean_dyn('dlogP_AB'):+7.3f}    |        {_mean_stat('dlogP_AB'):+7.3f}")
    print(f"  (1,0) A_only           |     {_mean_dyn('L10_Aonly'):+6.2f}      |     {_mean_dyn('dlogP_Aonly'):+7.3f}    |        {_mean_stat('dlogP_Aonly'):+7.3f}")
    print(f"  (0,1) B_only           |     {_mean_dyn('L10_Bonly'):+6.2f}      |     {_mean_dyn('dlogP_Bonly'):+7.3f}    |        {_mean_stat('dlogP_Bonly'):+7.3f}")
    print(f"  (0,0) Ctrl             |     {_mean_dyn('L10_Ctrl'):+6.2f}      |     {_mean_dyn('dlogP_Ctrl'):+7.3f}    |        {_mean_stat('dlogP_Ctrl'):+7.3f}")
    print(f"  Cross (A_i + B_j)      |     {_mean_dyn('L10_Cross'):+6.2f}      |     {_mean_dyn('dlogP_Cross'):+7.3f}    |        {_mean_stat('dlogP_Cross'):+7.3f}")
    print("  -----------------------+-----------------+----------------+-----------------------")
    print(
        f"  Dynamic Paging Pass    : L10_act={pass_l10_act}/{N_CHAINS} | TruthTable+Cross={pass_dlogp_tt}/{N_CHAINS} | Mean KL={mean_kl:.4f}"
    )
    print(
        f"  Static Superposition   : L10_act={static_pass_l10_act}/{N_CHAINS} | TruthTable+Cross={static_pass_dlogp_tt}/{N_CHAINS} | Mean AB dlogP={static_mean_ab_dlogp:+.3f} | Mean KL={static_mean_kl:.4f}"
    )
    print("=" * 100)

    assert pass_l10_act == N_CHAINS, f"Expected 20/20 L10 AND act pass, got {pass_l10_act}/{N_CHAINS}"
    assert pass_dlogp_tt == N_CHAINS, f"Expected 20/20 TruthTable+Cross pass, got {pass_dlogp_tt}/{N_CHAINS}"
    assert mean_kl < 0.01, f"Control KL too high: {mean_kl:.5f}"
    assert mean_route_swap_ms < mean_l0_l5_ms, (
        f"Route+swap {mean_route_swap_ms:.4f}ms did not hide within L0..L5 {mean_l0_l5_ms:.4f}ms"
    )
    assert pass_dlogp_tt > static_pass_dlogp_tt, (
        f"Dynamic Paging ({pass_dlogp_tt}) did not beat Static Superposition ({static_pass_dlogp_tt})"
    )


if __name__ == "__main__":
    main()
