"""Perpendicular loop direction on MATH-500: the blind pipeline again, with the healthy part taken out of the edit.

Why: in the 0.8B MATH-500 run (discover_math500_blind.py, results_discover_math500_qwen35-08b_colab_a100.json) every
2-layer config failed the screen's healthy-NLL guard (dNLL 0.05-0.21), so only a weak 1-layer edit survived
(always on 10.7% vs base 2.7%; via MoI it tied a random direction). The edit W' = W - a u (m * u^T W) also removes
whatever part of u normal reasoning writes along. Idea from Ishiki Labs' audio weight-space programming (block 2):
take that part out first,  u_perp = u - V_k V_k^T u  (renormalized), where V_k are the top-k eigenvectors of the
second moment of the layer's down_proj output on NORMAL (non-loop) tokens. k = 0 is the plain direction.

Steps (the generation, freeze, test and report steps are discover_math500_blind's own, unchanged):
  setup    copy the finished plain run (config, requests, elicit outputs, diagnosis, base test outputs, its test
           arms as a paired reference) into a fresh work dir; dev requests get 16k tokens (8k cut half of them).
  healthy  one forward pass over the half-A elicit responses (the half the direction was built on; the screen
           uses half B): per layer, E[y y^T] over normal tokens -> top-32 eigenvectors.
  screen   k in {0, 1, 4, 16} x layers {1, 2} x neurons {16, 64, 256, all} x alpha {4, 8, 12, 16, 24} = 160 configs,
           same 1-forward screen (pareto, dNLL guard 0.05). Rule, fixed before the run: the top row sets k*; the
           finalists are the top 3 rows with that k*; blind_directions.npz is rewritten with u_perp(k*) so every
           later step (dev, checkpoints, test) uses it. The random-direction control is unchanged.
  select   discover_math500_blind select, dev at 16k (base-cache key for dev uses 16k too).
  freeze / test   discover_math500_blind's, unchanged (test base outputs = the plain run's, same problems + seeds).
  report   discover_math500_blind report + perp summary + paired comparison against the plain run's arms.

    python scripts/discover_math500_perp.py setup --work W --ref REF_WORK
    python scripts/discover_math500_perp.py healthy --work W
    python scripts/discover_math500_perp.py screen --work W
    python scripts/discover_math500_perp.py select|freeze|test|report --work W ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import discover_blind as blind  # noqa: E402
import discover_fast as fast  # noqa: E402
import discover_loop_fix as dl  # noqa: E402
import discover_math500_blind as m  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402
import screen_fast_grid as sfg  # noqa: E402

K_PERP = (0, 1, 4, 16)
GRID_A = (4.0, 8.0, 12.0, 16.0, 24.0)
N_BASIS = 32
DEV_TOKENS = 16384
AFTER_ONSET = 768  # discover_fast diagnose default
COPY = ("config.json", "requests_elicit.jsonl", "requests_dev.jsonl", "requests_test.jsonl", "outputs_elicit.jsonl",
        "gate_elicit.json", "diagnosis_blind.json", "blind_directions.npz", "outputs_test_base.jsonl")
REF_ARMS = ("outputs_test_fast.jsonl", "outputs_test_moi_fast.jsonl", "requests_test_moi.jsonl", "frozen.json")


# ------------------------------------------------------------------ pure helpers (CPU-tested)
def perp_direction(u: np.ndarray, V: np.ndarray, k: int) -> np.ndarray:
    """u minus its projection on the top-k healthy directions (rows of V, orthonormal), renormalized."""
    u = np.asarray(u, dtype=np.float64)
    if k == 0:
        return u / np.linalg.norm(u)
    Vk = np.asarray(V[:k], dtype=np.float64)
    w = u - Vk.T @ (Vk @ u)
    n = np.linalg.norm(w)
    if n < 1e-8:
        raise ValueError("u lies inside the healthy subspace: nothing left after projection")
    return w / n


def perp_name(c: dict, k: int) -> str:
    return f"{blind.cfg_name(c['layers'], c['neurons'], c['alpha'])}_p{k}"


def perp_grid(rank: list[int]) -> list[dict]:
    out = []
    for kp in K_PERP:
        for k in m.GRID_K:
            for n in m.GRID_N:
                for al in GRID_A:
                    c = {"layers": sorted(rank[:k]), "neurons": n, "alpha": al, "k": k}
                    out.append({**c, "perp_k": kp, "name": perp_name(c, kp)})
    return out


def pick_finalists(rows: list[dict], by_name: dict, top: int = m.SCREEN_TOP_K) -> tuple[int, list[dict]]:
    """Top screen row sets k*; finalists are the best `top` rows with that k* (screen order)."""
    k_star = by_name[rows[0]["name"]]["perp_k"]
    fin = [by_name[r["name"]] for r in rows if by_name[r["name"]]["perp_k"] == k_star][:top]
    return k_star, fin


def dev_key_tokens(split: str, max_tokens: int) -> int:
    return DEV_TOKENS if split == "dev" else max_tokens


def healthy_basis(S: np.ndarray, n_keep: int = N_BASIS) -> tuple[np.ndarray, np.ndarray]:
    """Top eigenvectors (as rows) and eigenvalues of a symmetric second-moment matrix, largest first."""
    ev, vec = np.linalg.eigh(np.asarray(S, dtype=np.float64))
    order = np.argsort(ev)[::-1][:n_keep]
    return vec[:, order].T.copy(), ev[order].copy()


# ------------------------------------------------------------------ commands
def cmd_setup(a) -> None:
    work, ref = Path(a.work), Path(a.ref)
    work.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        if not (work / f).exists():
            shutil.copy2(ref / f, work / f)
    (work / "plain_ref").mkdir(exist_ok=True)
    for f in REF_ARMS:
        if (ref / f).exists() and not (work / "plain_ref" / f).exists():
            shutil.copy2(ref / f, work / "plain_ref" / f)
    if not (work / "blind_directions_plain.npz").exists():
        shutil.copy2(work / "blind_directions.npz", work / "blind_directions_plain.npz")
    reqs = lr.read_jsonl(work / "requests_dev.jsonl")
    for r in reqs:
        r["max_tokens"] = DEV_TOKENS
    lr.write_jsonl(work / "requests_dev.jsonl", reqs)
    print(f"setup: copied the plain run from {ref}; dev requests at {DEV_TOKENS} tokens")
    fast.sync(work)


def cmd_healthy(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    if (work / "healthy_basis.npz").exists():
        print("healthy: already done")
        return
    cfg = dl.cfg_load(work)
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_elicit.jsonl")}
    seqs = []  # rebuilt exactly as discover_fast.cmd_diagnose did, so half A is the same half
    for o in lr.read_jsonl(work / "outputs_elicit.jsonl"):
        body = lr.strip_eos(o["token_ids"], eos)
        lab = dl.token_labels(body)
        if not (lab >= 0).any():
            continue
        first = int(np.argmax(lab == 1)) if (lab == 1).any() else None
        cap = diag.get("max_seq", 16384)
        end = min(len(body), (first + AFTER_ONSET) if first is not None else cap, cap)
        seqs.append({"prompt": reqs[o["id"]]["prompt_ids"], "body": body[:end], "lab": lab[:end]})
    half = np.random.default_rng(cfg["seed"]).permutation(len(seqs)) % 2
    t0 = time.perf_counter()
    dtype = torch.bfloat16 if fast.bf16_ok() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["base_dir"], dtype=dtype, trust_remote_code=True).cuda().eval()
    layers = next(mm for nm, mm in model.named_modules()
                  if nm.endswith("layers") and isinstance(mm, torch.nn.ModuleList) and len(mm) >= 2)
    downs = [lr.down_module(lyr) for lyr in layers]
    D = downs[0].out_features
    S = [torch.zeros(D, D, device="cuda", dtype=torch.float64) for _ in downs]
    cnt = {"n": 0}
    cur = {}

    def hook(li):
        def f(mod, inp, out):
            y = out[0].float()[cur["sel"]]
            S[li] += (y.T @ y).double()
        return f

    hs = [d.register_forward_hook(hook(i)) for i, d in enumerate(downs)]
    body_model = getattr(model, "model", model)
    with torch.no_grad():
        for s, hb in zip(seqs, half):
            if hb != 0:
                continue
            ids = torch.tensor([s["prompt"] + s["body"]], device="cuda")
            full = np.full(ids.shape[1], -1, dtype=np.int64)
            full[len(s["prompt"]):] = s["lab"]
            cur["sel"] = torch.from_numpy(full == 0).cuda()
            cnt["n"] += int(cur["sel"].sum())
            body_model(input_ids=ids, use_cache=False)
    for h in hs:
        h.remove()
    z = np.load(work / "blind_directions_plain.npz")
    out, stats = {}, {"normal_tokens": cnt["n"], "layers": {}}
    for li, Si in enumerate(S):
        Sn = (Si / max(cnt["n"], 1)).cpu().numpy()
        V, ev = healthy_basis(Sn)
        out[f"V{li}"], out[f"ev{li}"] = V.astype(np.float32), ev
        u = z[f"u{li}"].astype(np.float64)
        stats["layers"][str(li)] = {
            "share_top": {str(k): round(float(ev[:k].sum() / np.trace(Sn)), 4) for k in K_PERP if k},
            "cos_u_perp": {str(k): round(float(u @ perp_direction(u, V, k) / np.linalg.norm(u)), 4) for k in K_PERP},
            "healthy_energy_ratio": {str(k): round(float(perp_direction(u, V, k) @ Sn @ perp_direction(u, V, k)
                                                         / (u @ Sn @ u / (u @ u))), 4) for k in K_PERP},
        }
    np.savez(work / "healthy_basis.npz", **out)
    stats["capture_s"] = round(time.perf_counter() - t0, 1)
    (work / "healthy_basis.json").write_text(json.dumps(stats, indent=1))
    fast.note_time(work, "healthy", stats["capture_s"])
    fast.sync(work)
    for li in diag["layer_rank"][:3]:
        print(f"L{li}: healthy energy along u_perp / along u =", stats["layers"][str(li)]["healthy_energy_ratio"],
              "| cos(u, u_perp) =", stats["layers"][str(li)]["cos_u_perp"])


def cmd_screen(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from wsi.moipatch import Edit, MoIPatch, _dtype_name, _sha256, module_suffix
    from wsi.tracer import screen_patch_candidates

    work = Path(a.work)
    if (work / "screen.json").exists():
        print("screen already done:", [c["name"] for c in json.loads((work / "screen.json").read_text())["finalists"]])
        return
    t0 = time.perf_counter()
    cfg = dl.cfg_load(work)
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    z = np.load(work / "blind_directions_plain.npz")
    hb = np.load(work / "healthy_basis.npz")
    healthy, target, spans, seq_ids = sfg.screen_sequences(work)
    grid = perp_grid(diag["layer_rank"])
    print(f"Screening {len(grid)} configs (k_perp {K_PERP}) on {len(healthy)} held-out half-B sequences...")
    dtype = torch.bfloat16 if fast.bf16_ok() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg["base_dir"], torch_dtype=dtype, device_map="auto",
                                                 trust_remote_code=True).eval()
    downs = {}
    for name, mod in model.named_modules():
        mt = fast.DOWN_RE.search(name + ".weight")
        if mt and "mtp" not in name and "visual" not in name:
            downs[int(mt.group(1))] = (name, mod)
    by_name = {c["name"]: c for c in grid}
    cands = {}
    for c in grid:
        edits = []
        for li in c["layers"]:
            name, mod = downs[li]
            w = mod.weight.detach()
            u = perp_direction(z[f"u{li}"], hb[f"V{li}"], c["perp_k"])
            edits.append(Edit(module=module_suffix(name), u=torch.from_numpy(u), alpha=float(c["alpha"]),
                              idx=torch.tensor(blind.columns(diag, li, c["neurons"])), base_sha256=_sha256(w),
                              base_dtype=_dtype_name(w.dtype), shape=tuple(w.shape)))
        cands[c["name"]] = MoIPatch(edits=edits, meta={"name": c["name"]})
    rows = screen_patch_candidates(model, cands, healthy_seqs=healthy, target_seqs=target, target_spans=spans,
                                   max_nll_delta=sfg.MAX_NLL_DELTA, rank_by=sfg.RANK_BY)
    del model
    torch.cuda.empty_cache()
    k_star, fin = pick_finalists(rows, by_name)
    # every later step reads u{li} from blind_directions.npz: give it the chosen direction
    np.savez(work / "blind_directions.npz",
             **{f"u{li}": perp_direction(z[f"u{li}"], hb[f"V{li}"], k_star).astype(np.float32)
                for li in range(diag["n_layers"])})
    passing = {str(k): sum(1 for r in rows if r["passes_nll_guard"] and by_name[r["name"]]["perp_k"] == k)
               for k in K_PERP}
    elapsed = round(time.perf_counter() - t0, 1)
    payload = {"rule": {"rank_by": sfg.RANK_BY, "max_nll_delta": sfg.MAX_NLL_DELTA, "n_screen_seqs": len(healthy),
                        "top_k": m.SCREEN_TOP_K, "grid_size": len(grid), "k_perp": list(K_PERP),
                        "alphas": list(GRID_A), "finalists": "top 3 rows with the top row's k_perp"},
               "seq_ids": seq_ids, "screen_s": elapsed, "k_star": k_star, "guard_passing_by_k": passing,
               "ranking": rows,
               "finalists": [{k: c[k] for k in ("name", "layers", "neurons", "alpha", "k")} for c in fin]}
    (work / "screen.json").write_text(json.dumps(payload, indent=1))
    fast.note_time(work, "screen", elapsed)
    fast.sync(work)
    print(f"Screen done in {elapsed:.0f}s. k* = {k_star}; guard-passing configs per k: {passing}")
    for i, r in enumerate(rows[:10], 1):
        print(f"  #{i:2d} {r['name']:22s} pareto={r['pareto_score']:7.3f} supp={r['target_suppression']:+.4f} "
              f"dNLL={r['delta_healthy_nll']:+.4f} guard={r['passes_nll_guard']}")
    print("finalists:", [c["name"] for c in fin])


def cmd_select(a) -> None:
    load, save = m._try_load_base_cache, m._save_base_cache
    m._try_load_base_cache = lambda w, c, sp, n, mt, o: load(w, c, sp, n, dev_key_tokens(sp, mt), o)
    m._save_base_cache = lambda w, c, sp, n, mt, o: save(w, c, sp, n, dev_key_tokens(sp, mt), o)
    try:
        m.cmd_select(a)
    finally:
        m._try_load_base_cache, m._save_base_cache = load, save


def cmd_freeze(a) -> None:
    m.cmd_freeze(a)
    work = Path(a.work)
    p = work / "perp_frozen.json"
    if not p.exists():
        p.write_text(json.dumps({"sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "k_star": json.loads((work / "screen.json").read_text())["k_star"]}, indent=1))
        fast.sync(work)


def _moi_correct(work: Path, moi_out: Path, plan_p: Path, base_rows: dict, tok) -> dict:
    """Per-problem correctness of an MoI arm, as discover_math500_blind.cmd_report grades it."""
    cfg = dl.cfg_load(work)
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    plan = {r["id"]: r for r in lr.read_jsonl(plan_p)}
    cont = {o["id"]: o for o in lr.read_jsonl(moi_out)}
    out = {}
    for qid, r in reqs.items():
        if qid not in cont:
            out[qid] = base_rows[qid]["correct"]
            continue
        pre = plan[qid]["prompt_ids"][len(r["prompt_ids"]):]
        full = pre + lr.strip_eos(cont[qid]["token_ids"], eos)
        out[qid] = m.grade_math(full, cont[qid]["finish_reason"], r["gold"], tok, eos, r["max_tokens"])["correct"]
    return out


def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    m.cmd_report(a)
    work = Path(a.work)
    pf = json.loads((work / "perp_frozen.json").read_text())
    if pf["sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise SystemExit("discover_math500_perp.py changed after freeze; refusing to report")
    res_p = work / "results_discover_math500_blind.json"
    res = json.loads(res_p.read_text())
    sc = json.loads((work / "screen.json").read_text())
    rank = {r["name"]: r for r in sc["ranking"]}
    res["perp"] = {"k_star": sc["k_star"], "guard_passing_by_k": sc["guard_passing_by_k"],
                   "chosen_screen_row": rank.get(f"{res['choice']['name']}_p{sc['k_star']}"),
                   "healthy_basis": json.loads((work / "healthy_basis.json").read_text())
                   if (work / "healthy_basis.json").exists() else None}
    ref = work / "plain_ref"
    if (ref / "outputs_test_fast.jsonl").exists() and "wsi_fixed" in res["arms"]:
        cfg = dl.cfg_load(work)
        tok = AutoTokenizer.from_pretrained(cfg["base_dir"], trust_remote_code=True)
        ids = sorted(r["id"] for r in lr.read_jsonl(work / "requests_test.jsonl"))
        base = {r["id"]: r for r in m.score_math(work, "requests_test.jsonl", "outputs_test_base.jsonl", tok)["rows"]}
        new_fixed = {r["id"]: r["correct"] for r in
                     m.score_math(work, "requests_test.jsonl", "outputs_test_fast.jsonl", tok)["rows"]}
        old_fixed = {r["id"]: r["correct"] for r in
                     m.score_math(work, "requests_test.jsonl", "plain_ref/outputs_test_fast.jsonl", tok)["rows"]}
        cmp = {"plain_choice": json.loads((ref / "frozen.json").read_text())["choice"] if (ref / "frozen.json").exists()
               else None,
               "perp_fixed_vs_plain_fixed": lr.paired_ci([new_fixed[i] for i in ids], [old_fixed[i] for i in ids])}
        if (work / "outputs_test_moi_fast.jsonl").exists() and (ref / "outputs_test_moi_fast.jsonl").exists():
            new_moi = _moi_correct(work, work / "outputs_test_moi_fast.jsonl", work / "requests_test_moi.jsonl", base, tok)
            old_moi = _moi_correct(work, ref / "outputs_test_moi_fast.jsonl", ref / "requests_test_moi.jsonl", base, tok)
            cmp["plain_moi_accuracy"] = sum(old_moi.values()) / len(old_moi)
            cmp["perp_moi_vs_plain_moi"] = lr.paired_ci([new_moi[i] for i in ids], [old_moi[i] for i in ids])
        cmp["plain_fixed_accuracy"] = sum(old_fixed.values()) / len(old_fixed)
        res["perp"]["vs_plain_run"] = cmp
        for k in ("perp_fixed_vs_plain_fixed", "perp_moi_vs_plain_moi"):
            if k in cmp:
                v = cmp[k]
                print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]  "
                      f"({v['a_only']}/{v['b_only']})")
    res_p.write_text(json.dumps(res, indent=1))
    fast.sync(work)
    print("k* =", sc["k_star"], "| guard-passing per k:", sc["guard_passing_by_k"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("setup")
    q.add_argument("--work", required=True)
    q.add_argument("--ref", required=True, help="work dir of the finished plain run")
    for name in ("healthy", "screen", "report"):
        sub.add_parser(name).add_argument("--work", required=True)
    for name in ("select", "test"):
        q = sub.add_parser(name)
        q.add_argument("--work", required=True)
        q.add_argument("--gpu-mem", type=float, default=0.92)
        q.add_argument("--dtype", default="auto")
        q.add_argument("--cpu", action="store_true")
        q.add_argument("--ck-root", default=None)
        if name == "select":
            q.add_argument("--force-patch", action="store_true")
    q = sub.add_parser("freeze")
    q.add_argument("--work", required=True)
    q.add_argument("--git-commit", default="local")
    a = ap.parse_args()
    {"setup": cmd_setup, "healthy": cmd_healthy, "screen": cmd_screen, "select": cmd_select, "freeze": cmd_freeze,
     "test": m.cmd_test, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
