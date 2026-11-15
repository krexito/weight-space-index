"""Honest experiment: FULL key->value neuron transplant LFM3-1.7B (2048d) -> GPT-2 124M (768d).

Unlike `demo_cross_model_transplant.py` (where WE chose the trigger), here BOTH halves of each
teacher neuron are read from the teacher's weights and translated:
  * KEY   (what activates it):  silu(w1 . x) * (w3 . x) over shared tokens
                                -> Dual-Gram Contrastive Key into GPT-2 c_fc column
  * VALUE (what it writes):     w2[:, i] via Dual-Gram Rosetta
                                -> GPT-2 c_proj row
and written into the most DORMANT neuron of a GPT-2 layer (overwritten, not added).

Also compares a zero-forward-pass analytical key baseline at Layer 0 vs. a fast Student-calibrated
pre-MLP key at Layer 10 (still ZERO Teacher forward passes).

Evaluation uses real GPT-2 forward passes on held-out prompt templates (never seen during
calibration):
  1. Selectivity: does the new GPT-2 neuron fire on the TEACHER's trigger tokens and not on controls?
  2. Effect:      does P(teacher's output tokens) rise at trigger positions?
  3. Collateral:  KL divergence of next-token distribution on control tokens.
Baseline: same procedure with the KEY taken from a DIFFERENT teacher neuron (mismatched key).
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
from wsi.compare import translate_vector

torch.manual_seed(0)
GPT2_LAYER = 10
N_TEACHER_NEURONS = 20
N_CTRL_POOL = 150
CALIB_PREFIX = "The"
# Held-out prompt templates (distinct from CALIB_PREFIX)
TEMPLATES = ["{t}", "I saw the{t}", "Then{t}", "The word{t}"]


def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w


def build_dual_gram_contrastive_key(
    Xs_pool: torch.Tensor,
    pos_pool_idx: list[int],
    y_pos: torch.Tensor,
    ctrl_pool_idx: list[int],
    *,
    key_norm: float = 8.0,
    ridge: float = 0.05,
    margin: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Solve a sparse Dual-Gram system on centered, normalized Student pre-MLP vectors
    and orthogonalize against the control mean so control prompts stay silent in GELU."""
    mu_s = Xs_pool[ctrl_pool_idx].mean(dim=0)
    Xc = Xs_pool - mu_s
    Xn = Xc / Xc.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    X_pos = Xn[pos_pool_idx]  # [M_pos, hidden]
    m = X_pos.shape[0]
    G = X_pos @ X_pos.T + ridge * torch.eye(m, dtype=X_pos.dtype)
    alpha = torch.linalg.solve(G, y_pos.float())
    k0 = X_pos.T @ alpha

    mu_hat = mu_s / mu_s.norm().clamp_min(1e-12)
    key = k0 - torch.dot(k0, mu_hat) * mu_hat
    key = key / key.norm().clamp_min(1e-12) * key_norm

    pre_all = Xs_pool @ key
    q90_ctrl = float(torch.quantile(pre_all[ctrl_pool_idx], 0.90).item())
    bias = -q90_ctrl - margin
    return key, bias


def main() -> None:
    t_start = time.perf_counter()
    lfm = WeightSpaceIndex.from_model(ROOT / "models" / "lfm3-1.7b")
    g2 = WeightSpaceIndex.from_model(ROOT / "models" / "gpt2")
    tok = GPT2Tokenizer.from_pretrained(str(ROOT / "models" / "gpt2"))
    model = GPT2LMHeadModel.from_pretrained(str(ROOT / "models" / "gpt2")).eval()

    # ---- shared vocabulary (alphabetic word tokens only, so prompts are meaningful) ----
    E_t = lfm.ck.get("model.embed_tokens.weight").float()
    t_map = {}
    for tid, s in lfm.vocab_map.items():
        if tid < E_t.shape[0] and s not in t_map:
            t_map[s] = tid
    shared = [
        (t_map[s], gid, s)
        for gid, s in g2.vocab_map.items()
        if s in t_map and s.startswith("Ġ") and s[1:].isalpha() and len(s) >= 4
    ]
    I_t = torch.tensor([a for a, _, _ in shared])
    I_g = torch.tensor([b for _, b, _ in shared])
    names = [s for _, _, s in shared]
    print(f"Shared alphabetic word tokens: {len(shared):,}")

    # ---- teacher activations over shared tokens (100% static weights, ZERO teacher passes) ----
    t_teacher0 = time.perf_counter()
    cands = []
    for L in (0, 1):
        pre = f"model.layers.{L}"
        w1 = lfm.ck.get(f"{pre}.feed_forward.w1.weight").float()
        w3 = lfm.ck.get(f"{pre}.feed_forward.w3.weight").float()
        w2 = lfm.ck.get(f"{pre}.feed_forward.w2.weight").float()
        X = rms(E_t[I_t], lfm.ck.get(f"{pre}.ffn_norm.weight").float())
        A = F.silu(X @ w1.T) * (X @ w3.T)  # [n_shared, 7680]
        top_v, _ = A.topk(10, dim=0)
        sparsity = top_v.mean(0) / A.abs().median(0).values.clamp_min(1e-6)
        for i in sparsity.topk(400).indices.tolist():
            cands.append((float(sparsity[i]), L, i, A[:, i].clone(), w2[:, i].clone()))
    cands.sort(key=lambda c: -c[0])

    picked = []
    for sp, L, i, act, val in cands:
        trig = [names[j] for j in act.topk(5).indices.tolist()]
        ch = translate_vector(lfm, g2, val, top_m=16)["teacher_top_tokens"]
        if not (set(trig) & set(ch)) and all(
            c.startswith("Ġ") and c[1:].isalpha() and len(c) >= 4 for c in ch[:3]
        ):
            picked.append((L, i, act, val, trig, ch))
        if len(picked) == N_TEACHER_NEURONS:
            break
    print(f"Selected {len(picked)} clean LFM3 neurons from weights in {(time.perf_counter()-t_teacher0):.2f}s")

    # ---- fast <=350-token Student pre-MLP calibration pool at target layer (ZERO teacher passes) ----
    trig10_union: set[int] = set()
    forbidden_ctrl: set[int] = set()
    for _, _, act, _, _, _ in picked:
        trig10_union.update(act.topk(10).indices.tolist())
        forbidden_ctrl.update(act.topk(50).indices.tolist())

    ctrl_candidates = [j for j in range(len(names)) if j not in forbidden_ctrl]
    perm = torch.randperm(len(ctrl_candidates))
    ctrl_pool_shared = [ctrl_candidates[p] for p in perm[:N_CTRL_POOL].tolist()]
    heldout_ctrl_pool = [ctrl_candidates[p] for p in perm[N_CTRL_POOL : N_CTRL_POOL + 200].tolist()]

    pool_shared_idx = sorted(trig10_union | set(ctrl_pool_shared))
    shared_to_pool = {s_idx: p_idx for p_idx, s_idx in enumerate(pool_shared_idx)}
    ctrl_pool_idx = [shared_to_pool[j] for j in ctrl_pool_shared]
    pool_g_ids = I_g[pool_shared_idx]

    prefix_id = tok.encode(CALIB_PREFIX, add_special_tokens=False)[0]
    calib_input_ids = torch.stack([torch.full_like(pool_g_ids, prefix_id), pool_g_ids], dim=1)

    t_cal0 = time.perf_counter()
    caps = []
    h_cal = model.transformer.h[GPT2_LAYER].ln_2.register_forward_hook(
        lambda m, i, o: caps.append(o[:, -1, :].detach())
    )
    with torch.no_grad():
        for start in range(0, calib_input_ids.shape[0], 128):
            model(input_ids=calib_input_ids[start : start + 128])
    h_cal.remove()
    Xs_cal = torch.cat(caps, dim=0)  # [N_pool, 768]
    print(
        f"Calibrated Student layer-{GPT2_LAYER} pre-MLP basis on {len(pool_shared_idx)} pool tokens "
        f"({CALIB_PREFIX!r}+tok) in {(time.perf_counter()-t_cal0):.2f}s",
        flush=True,
    )

    # Also compute zero-forward-pass analytical Layer-0 ln_2 basis on the same pool for comparison
    wte, wpe = g2.ck.get("wte.weight").float(), g2.ck.get("wpe.weight").float()
    ln0_w = g2.ck.get("h.0.ln_2.weight").float()
    ln0_b = g2.ck.get("h.0.ln_2.bias").float()
    Xs_l0_analytic = F.layer_norm(wte[pool_g_ids] + wpe[1], (768,)) * ln0_w + ln0_b

    mlp = model.transformer.h[GPT2_LAYER].mlp
    W_fc, b_fc, W_pr = mlp.c_fc.weight.data, mlp.c_fc.bias.data, mlp.c_proj.weight.data
    dormant = int((W_fc.norm(dim=0) * W_pr.norm(dim=1)).argmin().item())
    orig = (W_fc[:, dormant].clone(), b_fc[dormant].clone(), W_pr[dormant].clone())
    print(f"GPT-2 layer {GPT2_LAYER}: overwriting most dormant neuron #{dormant}\n", flush=True)

    def eval_prompts(
        prompts: list[str], target_mlp: torch.nn.Module | None = None, unit: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Group prompts by exact token length and run a single batched no_grad pass,
        capturing both `target_mlp.act[:, -1, unit]` and final-position log_softmax."""
        groups: dict[int, list[tuple[int, list[int]]]] = {}
        for idx_p, p in enumerate(prompts):
            ids = tok.encode(p)
            groups.setdefault(len(ids), []).append((idx_p, ids))

        out_lp = torch.empty((len(prompts), model.config.vocab_size), dtype=torch.float32)
        out_act = torch.zeros(len(prompts), dtype=torch.float32)
        buf: list[torch.Tensor] = []
        hook = (
            target_mlp.act.register_forward_hook(lambda m, i, o: buf.append(o[:, -1, unit].detach()))
            if target_mlp is not None and unit is not None
            else None
        )
        with torch.no_grad():
            for _seq_len, items in groups.items():
                batch_ids = torch.tensor([x[1] for x in items], dtype=torch.long)
                buf.clear()
                logits = model(input_ids=batch_ids).logits[:, -1, :]
                lp = F.log_softmax(logits, dim=-1)
                acts = buf[0] if hook is not None else None
                for k_pos, (orig_idx, _) in enumerate(items):
                    out_lp[orig_idx] = lp[k_pos]
                    if acts is not None:
                        out_act[orig_idx] = acts[k_pos]
        if hook is not None:
            hook.remove()
        return out_act, out_lp

    def install(key: torch.Tensor, bias: float, val_s: torch.Tensor) -> None:
        W_fc[:, dormant] = key
        b_fc[dormant] = bias
        W_pr[dormant] = val_s

    def restore() -> None:
        W_fc[:, dormant], b_fc[dormant], W_pr[dormant] = orig

    l0_mlp = model.transformer.h[0].mlp
    l0_dormant = int((l0_mlp.c_fc.weight.data.norm(dim=0) * l0_mlp.c_proj.weight.data.norm(dim=1)).argmin().item())
    l0_orig = (l0_mlp.c_fc.weight.data[:, l0_dormant].clone(), l0_mlp.c_fc.bias.data[l0_dormant].clone())

    rows = []
    l0_sel_count = 0
    for n, (L, i, act, val, trig, ch) in enumerate(picked):
        top10_shared = act.topk(10).indices
        pos_pool_idx = [shared_to_pool[int(j)] for j in top10_shared.tolist()]
        y_pos = act[top10_shared] / act[top10_shared].max().clamp_min(1e-12)

        # Dual-Gram Contrastive Input Key on Student Layer 10 calibrated Xs
        key, bias = build_dual_gram_contrastive_key(Xs_cal, pos_pool_idx, y_pos, ctrl_pool_idx)

        # Mismatched baseline: exact same Dual-Gram procedure on a DIFFERENT teacher neuron's triggers
        o_act = picked[(n + 7) % len(picked)][2]
        o_top10 = o_act.topk(10).indices
        o_pos_pool = [shared_to_pool[int(j)] for j in o_top10.tolist()]
        o_y_pos = o_act[o_top10] / o_act[o_top10].max().clamp_min(1e-12)
        mm_key, mm_bias = build_dual_gram_contrastive_key(Xs_cal, o_pos_pool, o_y_pos, ctrl_pool_idx)

        # Dual-Gram Rosetta VALUE vector
        v_s = translate_vector(lfm, g2, val, top_m=16)["vector_student"]
        v_s = v_s / v_s.norm().clamp_min(1e-12) * 3.0 * W_pr.norm(dim=1).mean()
        ch_ids = [tok.convert_tokens_to_ids(c) for c in ch[:5]]

        # Held-out evaluation prompts (using top-5 triggers and 20 disjoint held-out control words)
        trig_ids = act.topk(5).indices.tolist()
        off = (n * 7) % (len(heldout_ctrl_pool) - 20)
        eval_ctrl_ids = heldout_ctrl_pool[off : off + 20]
        P_trig = [t.format(t=tok.convert_tokens_to_string([names[j]])) for j in trig_ids for t in TEMPLATES]
        P_ctrl = [t.format(t=tok.convert_tokens_to_string([names[j]])) for j in eval_ctrl_ids for t in TEMPLATES]

        # Test zero-pass analytical Layer 0 key selectivity (only input half)
        k0_l0, b0_l0 = build_dual_gram_contrastive_key(Xs_l0_analytic, pos_pool_idx, y_pos, ctrl_pool_idx)
        l0_mlp.c_fc.weight.data[:, l0_dormant] = k0_l0
        l0_mlp.c_fc.bias.data[l0_dormant] = b0_l0
        l0_a_trig, _ = eval_prompts(P_trig, l0_mlp, l0_dormant)
        l0_a_ctrl, _ = eval_prompts(P_ctrl, l0_mlp, l0_dormant)
        l0_mlp.c_fc.weight.data[:, l0_dormant], l0_mlp.c_fc.bias.data[l0_dormant] = l0_orig
        l0_sel_count += int(l0_a_trig.mean().item() > l0_a_ctrl.mean().item())

        _, base_trig = eval_prompts(P_trig)
        _, base_ctrl = eval_prompts(P_ctrl)

        res = {}
        for mode, (k_use, b_use) in (("matched", (key, bias)), ("mismatched", (mm_key, mm_bias))):
            install(k_use, b_use, v_s)
            a_trig, lp_trig = eval_prompts(P_trig, mlp, dormant)
            a_ctrl, lp_ctrl = eval_prompts(P_ctrl, mlp, dormant)
            gain = (lp_trig[:, ch_ids] - base_trig[:, ch_ids]).mean().item()
            kl = F.kl_div(lp_ctrl, base_ctrl, log_target=True, reduction="batchmean").item()
            res[mode] = (a_trig.mean().item(), a_ctrl.mean().item(), gain, kl)
            restore()
        rows.append((L, i, trig, ch, res))

        m, mm = res["matched"], res["mismatched"]
        print(f"[{n+1:2d}] LFM3 L{L} #{i:4d}  triggers={[t[1:] for t in trig[:3]]}  writes={[c.lstrip('Ġ') for c in ch[:3]]}", flush=True)
        print(f"      matched key : act trig={m[0]:+.2f} ctrl={m[1]:+.2f} | logP(writes) gain on trig={m[2]:+.3f} | ctrl KL={m[3]:.4f}", flush=True)
        print(f"      mismatched  : act trig={mm[0]:+.2f} ctrl={mm[1]:+.2f} | logP(writes) gain on trig={mm[2]:+.3f} | ctrl KL={mm[3]:.4f}", flush=True)

    # ---- aggregate ----
    def agg(mode: str, k: int) -> float:
        return sum(r[4][mode][k] for r in rows) / len(rows)

    sel = sum(r[4]["matched"][0] > r[4]["matched"][1] for r in rows)
    win = sum(r[4]["matched"][2] > r[4]["mismatched"][2] for r in rows)
    print("\n" + "=" * 90)
    print(f"SUMMARY over {len(rows)} transplanted LFM3 neurons -> GPT-2 layer {GPT2_LAYER} neuron #{dormant} ({time.perf_counter()-t_start:.1f}s)")
    print(f"  Zero-Pass Analytic Layer-0 Key Selectivity         : {l0_sel_count}/{len(rows)}")
    print(f"  Calibrated Layer-{GPT2_LAYER} Selectivity (trig > ctrl)   : {sel}/{len(rows)}")
    print(
        f"  Matched key: act trig={agg('matched',0):+.3f} ctrl={agg('matched',1):+.3f} | "
        f"mean logP gain={agg('matched',2):+.3f} | ctrl KL={agg('matched',3):.4f}"
    )
    print(
        f"  Mismatched : act trig={agg('mismatched',0):+.3f} ctrl={agg('mismatched',1):+.3f} | "
        f"mean logP gain={agg('mismatched',2):+.3f} | ctrl KL={agg('mismatched',3):.4f}"
    )
    print(f"  Matched beats mismatched on output gain          : {win}/{len(rows)}")


if __name__ == "__main__":
    main()
