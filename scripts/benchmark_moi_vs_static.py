"""Benchmark: GPT-2 original vs GPT-2 static (20 circuits / 40 neurons) vs GPT-2 + MoI (20 circuits / 2 neurons).

Three configurations of the SAME GPT-2 124M checkpoint are compared head-to-head:
  (1) `base`   : GPT-2 unmodified.
  (2) `static` : the 20 LFM3-1.7B -> GPT-2 compositional `(A + B -> C)` circuits installed permanently,
                 each in its own dormant neuron pair (20 neurons in Layer 6 + 20 neurons in Layer 10).
  (3) `moi`    : Mixture of Indexes -- the same 20 circuits kept in an external ~12 KB/circuit index and
                 paged on-the-fly into ONE shared dormant neuron pair (Layer 6 #u_src, Layer 10 #u_dst).
                 When a prompt matches no trigger token, the original slot weights are restored.

Measured per configuration:
  * Skill score   : strict truth-table pass rate over 20 circuits (`AB` vs `A_only`, `B_only`, `Ctrl`,
                    `CrossChain`), mean d_logP(C), top-10 hit rate of C on `AB` prompts and false-positive
                    top-10 hit rate on negative prompts.
  * General text  : perplexity and per-token KL vs. original GPT-2 on a fixed general-English corpus.
  * Consumption   : parameter bytes, neurons overwritten, extra resident bytes (index catalog), process RSS.
  * Latency       : per-prompt B=1 CPU wall-clock (route + swap + full forward for MoI), interleaved rounds.

Note: the payload of a circuit does not depend on which dormant unit receives it (keys, biases and values are
calibrated from layer-level statistics), so the static configuration reuses the exact same compiled payloads
written into 20 distinct neuron pairs.
"""
from __future__ import annotations

import json
import os
import statistics
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
LAT_ROUNDS = 3
CONFIGS = ("base", "static", "moi")

GENERAL_CORPUS = [
    "The city council approved a new budget on Tuesday after several hours of debate.",
    "Scientists have discovered a species of frog that glows under ultraviolet light.",
    "She opened the window and listened to the rain falling on the quiet street.",
    "The company reported higher profits this quarter, driven by strong demand for its products.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "He forgot his keys at the office and had to wait outside for an hour.",
    "The museum will host an exhibition of ancient pottery from the Mediterranean.",
    "Many students prefer to study in the library because it is calm and well lit.",
    "The train was delayed by twenty minutes due to a signal failure near the station.",
    "Our team spent the weekend testing the new software before the official release.",
    "The doctor recommended more exercise and a balanced diet to improve his health.",
    "In the morning, the farmers gathered the vegetables and loaded them onto trucks.",
    "The novel tells the story of a young woman who leaves her village to become a painter.",
    "Engineers are designing bridges that can withstand stronger earthquakes.",
    "The children played football in the park until the sun went down.",
    "Prices for fresh fruit tend to rise during the winter months.",
    "The president met with foreign leaders to discuss trade and security.",
    "A small bakery on the corner sells bread that is still warm in the early hours.",
    "The researchers published their results in a scientific journal last month.",
    "When the storm passed, the neighbors helped each other clean the streets.",
    "The library extended its opening hours to support students during exams.",
    "Most birds build their nests in spring, when food becomes more abundant.",
    "The concert was cancelled because the singer had lost her voice.",
    "Good communication is important for any project that involves many people.",
    "The old castle on the hill attracts thousands of tourists every summer.",
    "He bought a new laptop because the old one had become too slow.",
    "The river flooded several farms after a week of heavy rain.",
    "Teachers say that reading aloud helps young children learn new words.",
    "The hospital opened a new wing dedicated to children and their families.",
    "After dinner, they took a short walk along the beach and watched the waves.",
]


def main() -> None:
    t0 = time.perf_counter()

    # 1. Discover 20 coupled 2-layer chains in LFM3-1.7B with disjoint A, B triggers and C outputs
    lfm = WeightSpaceIndex.from_model(ROOT / "models" / "lfm3-1.7b")
    g2 = WeightSpaceIndex.from_model(ROOT / "models" / "gpt2")
    chains = discover_connected_chains(
        lfm, 0, 1, top_k=N_CHAINS, shared_vocab_with=g2, alphabetic_only=True
    )
    print(f"[LFM3-1.7B] Discovered {len(chains)} compositional (A + B -> C) chains")

    tok = GPT2Tokenizer.from_pretrained(str(ROOT / "models" / "gpt2"))
    model = GPT2LMHeadModel.from_pretrained(str(ROOT / "models" / "gpt2")).eval()
    torch.set_grad_enabled(False)
    _, I_s, shared, _, _ = _get_shared_vocab_rosetta(lfm, g2)
    shared_to_idx = {s: idx for idx, s in enumerate(shared)}

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

    # 2a. Single-token pre-MLP bases on ('The' + tok)
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
    for st in range(0, single_ids.shape[0], 128):
        model(input_ids=single_ids[st : st + 128])
    xs_src_all = torch.cat(caps_src, dim=0)
    xs_dst_all = torch.cat(caps_dst, dim=0)
    xs_src_dict = {int(pool_g_ids[k].item()): xs_src_all[k] for k in range(len(pool_shared))}
    xs_dst_dict = {int(pool_g_ids[k].item()): xs_dst_all[k] for k in range(len(pool_shared))}

    # 2b. Position-2 w2 sweep
    w2_cal = c_ids[:6]
    w1_ctrl = c_ids[8:128]
    all_b_ids = sorted({int(I_s[j].item()) for ch in chains for j in ch["dst_trigger_shared_indices"][:5]})
    w2_pool = c_ids[:120] + all_b_ids
    b_id_to_col = {bid: 120 + i for i, bid in enumerate(all_b_ids)}
    sweep_rows = [[p, cf, w2] for p in (p1, p2) for cf in c_ids[:2] for w2 in w2_pool]
    caps_src.clear()
    caps_dst.clear()
    for st in range(0, len(sweep_rows), 256):
        model(input_ids=torch.tensor(sweep_rows[st : st + 256], dtype=torch.long))
    X_sw6 = torch.cat(caps_src, dim=0).reshape(4, len(w2_pool), d_hidden)
    X_sw10 = torch.cat(caps_dst, dim=0).reshape(4, len(w2_pool), d_hidden)
    X_sw6_diff = X_sw6 - X_sw6[:, :120].mean(dim=1, keepdim=True)
    _, S6, Vh6 = torch.linalg.svd(X_sw6_diff[:, :120].reshape(-1, d_hidden), full_matrices=False)
    D6_comp = (S6[:120, None] * Vh6[:120]) / (120 ** 0.5)
    X_sw10_tr = X_sw10[:, :80].mean(dim=0)
    X_sw10_val = X_sw10[:, 80:120].reshape(-1, d_hidden)

    # 2c. 3-token shared control pass
    ctrl_rows = [[p, w1, w2] for p in (p1, p2) for w1 in w1_ctrl[:80] for w2 in w2_cal[:4]]
    caps_src.clear()
    caps_dst.clear()
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

    # 2d. Per-chain positive context pass
    pos_rows: list[list[int]] = []
    for ch in chains:
        a_ids = [int(I_s[j].item()) for j in ch["src_trigger_shared_indices"][:5]]
        b_ids = [int(I_s[j].item()) for j in ch["dst_trigger_shared_indices"][:5]]
        for p in (p1, p2):
            for a_gid in a_ids:
                for w2_gid in w2_cal[:4] + b_ids[:4]:
                    pos_rows.append([p, a_gid, w2_gid])
    caps_src.clear()
    caps_dst.clear()
    for st in range(0, len(pos_rows), 256):
        model(input_ids=torch.tensor(pos_rows[st : st + 256], dtype=torch.long))
    h_src.remove()
    h_dst.remove()
    chain_xp6_all = torch.cat(caps_src, dim=0).reshape(len(chains), 2, 5, 8, d_hidden)
    chain_xp10_all = torch.cat(caps_dst, dim=0).reshape(len(chains), 2, 5, 8, d_hidden)

    # 3. Dormant units: 1 shared pair for MoI, 20 distinct pairs for the static model
    mlp_s = model.transformer.h[SRC_LAYER].mlp
    mlp_d = model.transformer.h[DST_LAYER].mlp
    Wfc_s, bfc_s, Wpr_s = mlp_s.c_fc.weight.data, mlp_s.c_fc.bias.data, mlp_s.c_proj.weight.data
    Wfc_d, bfc_d, Wpr_d = mlp_d.c_fc.weight.data, mlp_d.c_fc.bias.data, mlp_d.c_proj.weight.data
    score_s = Wfc_s.norm(dim=0) * Wpr_s.norm(dim=1)
    score_d = Wfc_d.norm(dim=0) * Wpr_d.norm(dim=1)
    u_src = int(score_s.argmin().item())
    u_dst = int(score_d.argmin().item())
    static_src_units = [int(u) for u in score_s.topk(N_CHAINS, largest=False).indices.tolist()]
    static_dst_units = [int(u) for u in score_d.topk(N_CHAINS, largest=False).indices.tolist()]
    orig_s = {u: (Wfc_s[:, u].clone(), bfc_s[u].clone(), Wpr_s[u].clone()) for u in set(static_src_units) | {u_src}}
    orig_d = {u: (Wfc_d[:, u].clone(), bfc_d[u].clone(), Wpr_d[u].clone()) for u in set(static_dst_units) | {u_dst}}
    print(
        f"MoI slot pair: L{SRC_LAYER} #{u_src} -> L{DST_LAYER} #{u_dst} | "
        f"Static: {N_CHAINS} pairs (L{SRC_LAYER} {static_src_units[:4]}..., L{DST_LAYER} {static_dst_units[:4]}...)"
    )

    def restore_units(src_units, dst_units) -> None:
        for u in src_units:
            k, b, v = orig_s[u]
            Wfc_s[:, u].copy_(k)
            bfc_s[u] = b
            Wpr_s[u].copy_(v)
        for u in dst_units:
            k, b, v = orig_d[u]
            Wfc_d[:, u].copy_(k)
            bfc_d[u] = b
            Wpr_d[u].copy_(v)

    def restore_all() -> None:
        restore_units(orig_s.keys(), orig_d.keys())

    def write_slot(us: int, ud: int, p: dict) -> None:
        Wfc_s[:, us].copy_(p["key_src"])
        bfc_s[us] = float(p["bias_src"])
        Wpr_s[us].copy_(p["val_src"])
        Wfc_d[:, ud].copy_(p["key_dst"])
        bfc_d[ud] = float(p["bias_dst"])
        Wpr_d[ud].copy_(p["val_dst"])

    # 4. Compile 20 circuits once; register them in the MoI index
    pager = g2.create_mixture_of_indexes(src_layer=SRC_LAYER, dst_layer=DST_LAYER, src_unit=u_src, dst_unit=u_dst)
    payloads: list[dict] = []
    recs: list[dict] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    t_comp = time.perf_counter()
    for n, ch in enumerate(chains):
        a_cand = [int(I_s[j].item()) for j in ch["src_trigger_shared_indices"][:10]]
        b_cand = [int(I_s[j].item()) for j in ch["dst_trigger_shared_indices"][:10]]
        b_cols = [b_id_to_col[b] for b in b_cand[:5]]

        def _pick(cands: list[int], used: set[int]) -> list[int]:
            out = [s for s in dict.fromkeys(cands) if s not in used][:3]
            for s in cands:
                if len(out) >= 3:
                    break
                if s not in out:
                    out.append(s)
            return out

        a_uniq = _pick(a_cand, used_a)
        b_uniq = _pick(b_cand, used_b)
        used_a.update(a_uniq)
        used_b.update(b_uniq)
        a_ids = a_uniq if all(x in a_cand[:5] for x in a_uniq) else a_cand[:3]
        b_ids = b_uniq if all(x in b_cand[:5] for x in b_uniq) else b_cand[:3]

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
            student_src_context_vecs={
                "pos_a_full": chain_xp6_all[n],
                "ctrl_tr": Xc6_tr,
                "ctrl_val": Xc6_val,
                "D6_comp": D6_comp,
                "D6_p": D6_p,
                "X_sw6_b_diff": X_sw6_diff[:, b_cols].reshape(-1, d_hidden),
            },
            student_dst_context_vecs={
                "Xb10_sw": X_sw10[:, b_cols],
                "Xp10_full": chain_xp10_all[n],
                "X_sw10_tr": X_sw10_tr,
                "X_sw10_val": X_sw10_val,
                "Xc10_val": Xc10_val,
                "D10_comp": D10_comp,
            },
        )
        cid = f"chain_{n:02d}"
        a_words = [tok.decode([x]) for x in a_ids]
        b_words = [tok.decode([x]) for x in b_ids]
        pager.register_circuit(cid, tx, a_token_ids=a_ids, b_token_ids=b_ids, a_tokens=a_words, b_tokens=b_words)
        payloads.append(pager.get_payload(cid))
        recs.append(
            {
                "n": n,
                "cid": cid,
                "a_words": a_words,
                "b_words": b_words,
                "b_ids": b_ids,
                "forb": {int(I_s[j].item()) for j in set(ch["src_top50_shared_indices"]) | set(ch["dst_top50_shared_indices"])},
                "hc1": [tok.decode([x]) for x in hc_ids[n * 6 : n * 6 + 3]],
                "hc2": [tok.decode([x]) for x in hc_ids[n * 6 + 3 : n * 6 + 6]],
                "out5": [tok.convert_tokens_to_ids(t) for t in ch["dst_2hop_tokens"][:5]],
            }
        )
    cat = pager.catalog_summary()
    print(f"Compiled {N_CHAINS} circuits in {time.perf_counter() - t_comp:.1f}s ({cat['per_circuit_kb']:.2f} KB each)")

    def mk(w1_list: list[str], w2_list: list[str]) -> list[str]:
        return [f"{tpl}{w1}{w2}" for tpl in HELDOUT_TEMPLATES for w1 in w1_list for w2 in w2_list]

    conds = ("AB", "A_only", "B_only", "Ctrl", "Cross")
    for n, rec in enumerate(recs):
        partner = (n + 1) % N_CHAINS
        for step in range(1, N_CHAINS):
            m = (n + step) % N_CHAINS
            cb = set(recs[m]["b_ids"])
            if not (cb & set(rec["b_ids"])) and not (cb & rec["forb"]):
                partner = m
                break
        rec["P"] = {
            "AB": mk(rec["a_words"], rec["b_words"]),
            "A_only": mk(rec["a_words"], rec["hc1"]),
            "B_only": mk(rec["hc1"], rec["b_words"]),
            "Ctrl": mk(rec["hc1"], rec["hc2"]),
            "Cross": mk(rec["a_words"], recs[partner]["b_words"]),
        }

    # 5. Evaluation helpers
    def eval_prompts(prompts: list[str], us: int, ud: int) -> tuple[torch.Tensor, torch.Tensor]:
        groups: dict[int, list[tuple[int, list[int]]]] = {}
        for i, p in enumerate(prompts):
            ids = tok.encode(p)
            groups.setdefault(len(ids), []).append((i, ids))
        out_lp = torch.empty((len(prompts), model.config.vocab_size), dtype=torch.float32)
        out_ad = torch.empty(len(prompts), dtype=torch.float32)
        buf: list[torch.Tensor] = []
        hd = mlp_d.act.register_forward_hook(lambda m, i, o: buf.append(o[:, -1, ud].detach()))
        for _, items in groups.items():
            buf.clear()
            lp = F.log_softmax(model(input_ids=torch.tensor([x[1] for x in items])).logits[:, -1, :], dim=-1)
            for k, (oi, _) in enumerate(items):
                out_lp[oi] = lp[k]
                out_ad[oi] = buf[0][k]
        hd.remove()
        return out_ad, out_lp

    def moi_install_for(ids: list[int]) -> str:
        dec = pager.route_token_ids(ids)
        if dec["matched_mode"] == "fallback":
            restore_units([u_src], [u_dst])
            return "__orig__"
        cid_sel = str(dec["circuit_id"])
        write_slot(u_src, u_dst, pager.get_payload(cid_sel))
        return cid_sel

    def moi_eval(prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        by_cid: dict[str, list[int]] = {}
        for i, p in enumerate(prompts):
            dec = pager.route_token_ids(tok.encode(p))
            key = "__orig__" if dec["matched_mode"] == "fallback" else str(dec["circuit_id"])
            by_cid.setdefault(key, []).append(i)
        out_lp = torch.empty((len(prompts), model.config.vocab_size), dtype=torch.float32)
        out_ad = torch.empty(len(prompts), dtype=torch.float32)
        for key, idxs in by_cid.items():
            if key == "__orig__":
                restore_units([u_src], [u_dst])
            else:
                write_slot(u_src, u_dst, pager.get_payload(key))
            ad, lp = eval_prompts([prompts[k] for k in idxs], u_src, u_dst)
            for j, k in enumerate(idxs):
                out_lp[k] = lp[j]
                out_ad[k] = ad[j]
        return out_ad, out_lp

    def install_static() -> None:
        restore_all()
        for n in range(N_CHAINS):
            write_slot(static_src_units[n], static_dst_units[n], payloads[n])

    corpus_ids = [tok.encode(t) for t in GENERAL_CORPUS]
    n_corpus_tokens = sum(len(ids) - 1 for ids in corpus_ids)

    def corpus_logprobs(cfg: str) -> list[torch.Tensor]:
        out = []
        for ids in corpus_ids:
            if cfg == "moi":
                moi_install_for(ids)
            out.append(F.log_softmax(model(input_ids=torch.tensor([ids])).logits[0, :-1].float(), dim=-1))
        return out

    # 6. Skill + general-text evaluation for all three configurations
    results: dict[str, dict] = {}
    base_lp: dict[tuple[int, str], torch.Tensor] = {}
    base_corpus: list[torch.Tensor] = []
    for cfg in CONFIGS:
        restore_all()
        if cfg == "static":
            install_static()
        per_chain = []
        for n, rec in enumerate(recs):
            row = {}
            for c in conds:
                if cfg == "moi":
                    ad, lp = moi_eval(rec["P"][c])
                    if c == conds[-1]:
                        restore_all()
                elif cfg == "static":
                    ad, lp = eval_prompts(rec["P"][c], static_src_units[n], static_dst_units[n])
                else:
                    ad, lp = eval_prompts(rec["P"][c], u_src, u_dst)
                    base_lp[(n, c)] = lp
                o5 = rec["out5"]
                row[f"L10_{c}"] = ad.mean().item()
                row[f"dlogP_{c}"] = (lp[:, o5] - base_lp[(n, c)][:, o5]).mean().item()
                row[f"logP_{c}"] = torch.logsumexp(lp[:, o5], dim=-1).mean().item()
                top10 = lp.topk(10, dim=-1).indices
                row[f"hit10_{c}"] = torch.isin(top10, torch.tensor(o5)).any(dim=1).float().mean().item()
                if c == "Ctrl":
                    row["KL_Ctrl"] = F.kl_div(lp, base_lp[(n, c)], log_target=True, reduction="batchmean").item()
            per_chain.append(row)

        restore_all()
        if cfg == "static":
            install_static()
        clp = corpus_logprobs(cfg)
        if cfg == "base":
            base_corpus = clp
        nll = 0.0
        kl_sum = 0.0
        for ids, lp, blp in zip(corpus_ids, clp, base_corpus):
            tgt = torch.tensor(ids[1:])
            nll += -lp[torch.arange(len(tgt)), tgt].sum().item()
            kl_sum += (blp.exp() * (blp - lp)).sum().item()
        restore_all()

        negs = ("A_only", "B_only", "Ctrl", "Cross")
        pass_tt = sum(r["dlogP_AB"] > max(r[f"dlogP_{c}"] for c in negs) + 1.5 for r in per_chain)
        pass_l10 = sum(
            r["L10_AB"] > 2.0 and r["L10_AB"] > max(r[f"L10_{c}"] for c in negs) + 2.0 for r in per_chain
        ) if cfg != "base" else 0
        mean = lambda k: sum(r[k] for r in per_chain) / len(per_chain)  # noqa: E731
        results[cfg] = {
            "skill_pass_truth_table": pass_tt,
            "skill_pass_l10_and": pass_l10,
            "mean_dlogP_AB": mean("dlogP_AB"),
            "mean_max_neg_dlogP": sum(max(r[f"dlogP_{c}"] for c in negs) for r in per_chain) / len(per_chain),
            "mean_logP_C_AB": mean("logP_AB"),
            "hit10_AB": mean("hit10_AB"),
            "hit10_false_pos": sum(r[f"hit10_{c}"] for r in per_chain for c in negs) / (len(per_chain) * len(negs)),
            "mean_KL_ctrl_prompts": mean("KL_Ctrl"),
            "corpus_ppl": float(torch.exp(torch.tensor(nll / n_corpus_tokens)).item()),
            "corpus_kl_per_token": kl_sum / n_corpus_tokens,
        }
        print(f"[{cfg}] evaluated: skill pass {pass_tt}/{N_CHAINS}, corpus PPL {results[cfg]['corpus_ppl']:.3f}")

    # 7. Latency (B=1, interleaved rounds)
    lat_ids = [tok.encode(p) for rec in recs for p in rec["P"]["AB"][:5]] + corpus_ids
    restore_all()
    for ids in lat_ids[:10]:
        model(input_ids=torch.tensor([ids]))
    lat: dict[str, list[float]] = {c: [] for c in CONFIGS}
    swap_ms: list[float] = []
    for _ in range(LAT_ROUNDS):
        for cfg in CONFIGS:
            restore_all()
            if cfg == "static":
                install_static()
            for ids in lat_ids:
                x = torch.tensor([ids])
                t_a = time.perf_counter()
                if cfg == "moi":
                    moi_install_for(ids)
                    t_sw = time.perf_counter()
                    swap_ms.append((t_sw - t_a) * 1e3)
                model(input_ids=x)
                lat[cfg].append((time.perf_counter() - t_a) * 1e3)
    restore_all()

    # 8. Consumption
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / 2**20
    except Exception:  # pragma: no cover
        rss_mb = float("nan")
    consumption = {
        "base": {"neurons_overwritten": 0, "extra_resident_bytes": 0},
        "static": {"neurons_overwritten": 2 * N_CHAINS, "extra_resident_bytes": 0},
        "moi": {"neurons_overwritten": 2, "extra_resident_bytes": int(cat["total_catalog_bytes"])},
    }
    for cfg in CONFIGS:
        v = sorted(lat[cfg])
        results[cfg].update(consumption[cfg])
        results[cfg]["param_bytes"] = param_bytes
        results[cfg]["lat_mean_ms"] = statistics.mean(v)
        results[cfg]["lat_p50_ms"] = v[len(v) // 2]
        results[cfg]["lat_p95_ms"] = v[int(len(v) * 0.95) - 1]
    results["moi"]["route_swap_mean_ms"] = statistics.mean(swap_ms)

    # 9. Report
    print("\n" + "=" * 104)
    print(f"GPT-2 124M: ORIGINAL vs STATIC (20 circuits / 40 neurons) vs MoI (20 circuits / 2 neurons)  [{time.perf_counter() - t0:.0f}s]")
    print("=" * 104)
    hdr = f"{'Metric':<44}|{'Original':>18} |{'Static (40 n.)':>18} |{'MoI (2 n.)':>18}"
    print(hdr)
    print("-" * len(hdr))

    def line(label: str, key: str, fmt: str) -> None:
        print(f"{label:<44}|" + "|".join(f"{format(results[c][key], fmt):>18} " for c in CONFIGS))

    line("Skill pass (truth table + cross, /20)", "skill_pass_truth_table", "d")
    line("Skill pass (L10 AND activation, /20)", "skill_pass_l10_and", "d")
    line("Mean d_logP(C) on AB prompts", "mean_dlogP_AB", "+.3f")
    line("Mean max d_logP(C) on negative prompts", "mean_max_neg_dlogP", "+.3f")
    line("C in top-10 on AB prompts", "hit10_AB", ".1%")
    line("C in top-10 on negative prompts (FP)", "hit10_false_pos", ".1%")
    line("KL vs original on control prompts", "mean_KL_ctrl_prompts", ".4f")
    line(f"General corpus perplexity ({n_corpus_tokens} tok)", "corpus_ppl", ".3f")
    line("General corpus KL/token vs original", "corpus_kl_per_token", ".5f")
    line("Neurons overwritten", "neurons_overwritten", "d")
    line("Extra resident bytes (index catalog)", "extra_resident_bytes", ",d")
    line("Model parameter bytes", "param_bytes", ",d")
    line("Latency mean (ms/prompt, B=1)", "lat_mean_ms", ".2f")
    line("Latency p50 (ms/prompt, B=1)", "lat_p50_ms", ".2f")
    line("Latency p95 (ms/prompt, B=1)", "lat_p95_ms", ".2f")
    print("-" * len(hdr))
    print(
        f"MoI route+swap mean: {results['moi']['route_swap_mean_ms']:.4f} ms/prompt (included in MoI latency) | "
        f"Process RSS: {rss_mb:.0f} MB | Latency prompts: {len(lat_ids)} x {LAT_ROUNDS} rounds"
    )

    out_dir = ROOT / "benchmarks" / "moi_vs_static"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"results": results, "process_rss_mb": rss_mb, "n_latency_prompts": len(lat_ids)}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
