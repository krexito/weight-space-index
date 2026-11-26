"""EXPERIMENTAL: blind MATH-500 loop-patch search that SELECTS by the escape test (Qwen3.5-0.8B).

Why: the MATH-500 blind search picked L10_n256_a12 (test 10.7%) because its 1-pass screen rewards suppression of the loop's
own tokens (experiment_screen_known_patches.py). Continuing from the loop cut and counting escapes orders six known
patches like the exam (Spearman 0.75-0.80, experiment_escape_screen.py). Here that rule drives a fresh blind search.

Protocol (fixed before running; only discovery data until the freeze; test touched once):
  data       the discover_math500 run's elicit split (100 problems): diagnosis_blind.json + blind_directions.npz (one
             diff-of-means direction per layer, neuron ranking), and the 98 escape cuts (requests_escape.jsonl, the
             elicit answers cut where the router fires; 47 come from looping answers)
  candidates layers 4..21 (one layer each) x neurons {16, 64, 256} x alpha {6, 12} = 108, direction = that layer's u,
             columns = its top-n ranked neurons, edit W' = W - alpha u (m u^T W) (discover_blind.build_checkpoint)
  guard      healthy NLL on 24 held-out elicit prefixes (screen_fast_grid.screen_sequences): dNLL <= 0.05 (the screen's)
  stage 1    every guard-passer continues the 47 looping cuts for 3072 tokens; rank by escaped share (no loop kill),
             ties -> fewer neurons, lower alpha; top 5 go on
  stage 2    the 5 finalists on all 98 cuts for 8192 tokens (the validated setting); winner = highest escaped share on
             the 47 looping cuts, ties -> correct share on all cuts, then fewer neurons. It must beat restart.
             References run in the same engine, never eligible: restart (unpatched), MIT p50c, our GSM8K blind patch
             (L13-14 n16 a12); they also check the in-memory swap (expected 25.5 / 97.9 / 78.7% looping escape)
  freeze     frozen.json (choice + stage tables + time) is written BEFORE any test generation
  test       the winner always-on on the 150 test problems at 16k, graded like experiment_edit_form_math500.py, paired
             against the saved base / GSM8K blind (20.7%) / p50c (28.7%) test outputs
Weights are swapped in one vLLM engine (discover_fast.make_setter + int16 checksum + reset_prefix_cache).

  python scripts/experiment_escape_search_math500.py --math-work M --escape-work E --blind-work BW --edit-dir D
         --model B --saved-test S --out-dir O
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _m in ("fla", "causal_conv1d"):
    sys.modules.setdefault(_m, None)

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

LAYERS = tuple(range(4, 22))
NEURONS = (16, 64, 256)
ALPHAS = (6.0, 12.0)
MAX_NLL_DELTA = 0.05
STAGE1_TOKENS = 3072
TOP = 5


def grid() -> list[dict]:
    return [{"name": f"L{l}_n{n}_a{a:g}", "layers": [l], "neurons": n, "alpha": a}
            for l in LAYERS for n in NEURONS for a in ALPHAS]


def jdump(p: Path, obj) -> None:
    p.with_suffix(".tmp").write_text(json.dumps(obj, indent=1))
    p.with_suffix(".tmp").replace(p)


def escape_stats(rows, reqs, tok, eos) -> dict:
    import discover_math500_blind as dm

    out = {"looping": [0, 0, 0, 0], "all": [0, 0, 0, 0]}  # n, escaped, answered, correct
    per = {}
    for o in rows:
        r = reqs[o["id"]]
        g = dm.grade_math(o["token_ids"], o["finish_reason"], r["gold"], tok, eos, r["max_tokens"])
        e, ans, c = (not g["looped_killed"]), (g["pred"] is not None and not g["looped_killed"]), g["correct"]
        per[o["id"]] = [int(e), int(ans), int(c)]
        for key in (["all", "looping"] if r["base_looped"] else ["all"]):
            v = out[key]
            v[0] += 1; v[1] += e; v[2] += ans; v[3] += c  # noqa: E702
    res = {}
    for key, (n, e, ans, c) in out.items():
        res[key] = {"n": n, "escaped": round(e / max(1, n), 4), "answered": round(ans / max(1, n), 4),
                    "correct": round(c / max(1, n), 4)}
    res["mean_tokens"] = round(float(np.mean([len(o["token_ids"]) for o in rows])), 1)
    res["per_id"] = per
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-work", required=True)
    ap.add_argument("--escape-work", required=True, help="folder with requests_escape.jsonl (experiment_escape_screen prepare)")
    ap.add_argument("--blind-work", required=True, help="discover_blind GSM8K run (reference patch only)")
    ap.add_argument("--edit-dir", required=True, help="benchmarks/exp8_math (p50c reference only)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--saved-test", required=True, help="folder with outputs_test_{base,blind_n16_a12,p50c}.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-cands", type=int, default=0, help="smoke test: first N candidates")
    ap.add_argument("--guard-only", action="store_true", help="CPU check: build candidates + guard, then stop")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run: 3 candidates, 128-token budgets")
    a = ap.parse_args()
    global STAGE1_TOKENS, TOP
    if a.smoke:
        a.max_cands, STAGE1_TOKENS, TOP = 3, 128, 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import discover_blind as blind
    import discover_fast as fast
    import moi_loop_router_vllm as lr
    import screen_fast_grid as sfg

    t_start = time.time()
    mw, ew, bw, out = Path(a.math_work), Path(a.escape_work), Path(a.blind_work), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((mw / "config.json").read_text())
    cfg["base_dir"] = a.model
    eos = set(cfg["eos"])
    tok = AutoTokenizer.from_pretrained(a.model)
    diag, z = json.loads((mw / "diagnosis_blind.json").read_text()), np.load(mw / "blind_directions.npz")
    dgb, zgb = json.loads((bw / "diagnosis_blind.json").read_text()), np.load(bw / "blind_directions.npz")
    base_w = fast.load_base_layers(Path(a.model), set(range(24)))

    cands = grid()[: a.max_cands] if a.max_cands else grid()
    for c in cands:
        l = c["layers"][0]
        c["w"] = {l: fast.edited(base_w[l], z[f"u{l}"], blind.columns(diag, l, c["neurons"]), c["alpha"])}
    # references (never eligible)
    refs = {"restart": {"name": "restart", "w": {}},
            "gsm8k_blind_n16_a12": {"name": "gsm8k_blind_n16_a12",
                                    "w": {l: fast.edited(base_w[l], zgb[f"u{l}"], blind.columns(dgb, l, 16), 12.0) for l in (13, 14)}}}
    man = json.loads((Path(a.edit_dir) / "p50c_manifest.json").read_text())
    zp = np.load(Path(a.edit_dir) / "p50c_edit.npz")
    pw = {}
    for i, t in enumerate(man["tensors"]):
        l = int(t["name"].split("layers.")[1].split(".")[0])
        bits = base_w[l].view(torch.int16).flatten().clone()
        bits[torch.from_numpy(zp[f"t{i}_idx"].astype(np.int64))] = torch.from_numpy(zp[f"t{i}_val"].astype(np.uint16).view(np.int16))
        pw[l] = bits.view(torch.bfloat16).reshape(base_w[l].shape)
    refs["p50c"] = {"name": "p50c", "w": pw}

    # ---------------------------------------------------------------- guard (HF, healthy NLL)
    gp = out / "guard.json"
    guard = json.loads(gp.read_text()) if gp.exists() else {}
    if len(guard) < len(cands) + len(refs) - 1:
        dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
        hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
        downs = {}
        for name, p in hf.named_parameters():
            m = fast.DOWN_RE.search(name)
            if m and "mtp" not in name and "visual" not in name:
                downs[int(m.group(1))] = p
        sfg.N_SCREEN_SEQS = 24
        healthy, _, _, ids = sfg.screen_sequences(mw)
        H = [torch.tensor([s], device=dev) for s in healthy]

        @torch.no_grad()
        def nll() -> float:
            tot, n = 0.0, 0
            for x in H:
                lg = hf(input_ids=x, use_cache=False).logits[0, :-1].float()
                tot += float(torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum"))
                n += x.shape[1] - 1
            return tot / n

        base_nll = nll()
        for c in cands + [refs["gsm8k_blind_n16_a12"], refs["p50c"]]:
            if c["name"] in guard:
                continue
            old = {l: downs[l].data.clone() for l in c["w"]}
            for l, w in c["w"].items():
                downs[l].data.copy_(w.to(downs[l].device, downs[l].dtype))
            d = nll() - base_nll
            for l, w in old.items():
                downs[l].data.copy_(w)
            guard[c["name"]] = {"dnll": round(d, 5), "passes": d <= MAX_NLL_DELTA}
        guard["_meta"] = {"base_nll": round(base_nll, 5), "seq_ids": ids, "n_seqs": len(H)}
        jdump(gp, guard)
        del hf
        if dev == "cuda":
            torch.cuda.empty_cache()
    passers = [c for c in cands if guard[c["name"]]["passes"]]
    print(f"[guard] {len(passers)}/{len(cands)} candidates pass dNLL <= {MAX_NLL_DELTA}; refs: "
          f"gsm8k {guard['gsm8k_blind_n16_a12']}, p50c {guard['p50c']}  ({time.time() - t_start:.0f}s)", flush=True)
    if a.guard_only:
        return

    # ---------------------------------------------------------------- vLLM engine with in-memory swaps
    import discover_math500_blind as dm

    esc = {r["id"]: r for r in lr.read_jsonl(ew / "requests_escape.jsonl")}
    loop_reqs = [{**r, "max_tokens": STAGE1_TOKENS} for r in esc.values() if r["base_looped"]]
    all_reqs = list(esc.values())
    test_reqs = lr.read_jsonl(mw / "requests_test.jsonl")
    if a.smoke:
        loop_reqs, all_reqs = loop_reqs[:6], [{**r, "max_tokens": 128} for r in all_reqs[:8]]
        test_reqs = [{**r, "max_tokens": 128} for r in test_reqs[:4]]
    max_len = max(len(r["prompt_ids"]) + r["max_tokens"] for r in all_reqs + test_reqs) + 16
    llm, kill = dm._llm_fast(a.model, cfg, max_len, "bfloat16", a.gpu_mem)
    state = {"edited": set()}

    def use(c):
        upd = {l: w for l, w in c["w"].items()}
        for l in state["edited"] - set(upd):
            upd[l] = base_w[l]
        if upd:
            upd_np = {l: w.to(torch.float32).numpy() for l, w in upd.items()}
            got = {}
            for r in llm.apply_model(fast.make_setter(upd_np)):
                got.update(r)
            if set(got) != set(upd_np):
                raise SystemExit(f"swap missed layers {sorted(set(upd_np) - set(got))}")
            for l, (dt, cs) in got.items():
                if cs != fast.int_checksum(upd[l], torch.bfloat16):
                    raise SystemExit(f"swap checksum mismatch at layer {l}")
            llm.reset_prefix_cache()
        state["edited"] = set(c["w"])

    def run(c, reqs, tag):
        nonlocal kill
        t0 = time.time()
        use(c)
        rows, kill = fast._generate(llm, cfg, reqs, kill)
        lr.write_jsonl(out / f"outputs_{tag}_{c['name']}.jsonl", rows)
        return rows, time.time() - t0

    # stage 1
    s1p = out / "stage1.json"
    s1 = json.loads(s1p.read_text()) if s1p.exists() else {}
    for i, c in enumerate(passers):
        if c["name"] in s1:
            continue
        rows, dt = run(c, loop_reqs, "s1")
        st = escape_stats(rows, esc, tok, eos)
        s1[c["name"]] = {k: st[k] for k in ("looping", "mean_tokens")} | {"s": round(dt, 1)}
        jdump(s1p, s1)
        print(f"[s1 {i + 1}/{len(passers)}] {c['name']:14s} esc {st['looping']['escaped']:.3f} "
              f"cor {st['looping']['correct']:.3f}  {dt:.0f}s", flush=True)
    order = sorted(passers, key=lambda c: (-s1[c["name"]]["looping"]["escaped"], c["neurons"], c["alpha"]))
    finalists = order[:TOP]
    print("[s1] finalists:", [(c["name"], s1[c["name"]]["looping"]["escaped"]) for c in finalists], flush=True)

    # stage 2 (finalists + references)
    s2p = out / "stage2.json"
    s2 = json.loads(s2p.read_text()) if s2p.exists() else {}
    for c in finalists + [refs["restart"], refs["p50c"], refs["gsm8k_blind_n16_a12"]]:
        if c["name"] in s2:
            continue
        rows, dt = run(c, all_reqs, "s2")
        st = escape_stats(rows, esc, tok, eos)
        s2[c["name"]] = st | {"s": round(dt, 1)}
        jdump(s2p, s2)
        print(f"[s2] {c['name']:20s} looping esc {st['looping']['escaped']:.3f} | all esc {st['all']['escaped']:.3f} "
              f"cor {st['all']['correct']:.3f}  {dt:.0f}s", flush=True)
    win = max(finalists, key=lambda c: (s2[c["name"]]["looping"]["escaped"], s2[c["name"]]["all"]["correct"], -c["neurons"]))
    beats = s2[win["name"]]["looping"]["escaped"] > s2["restart"]["looping"]["escaped"] or a.smoke  # smoke: exercise test path
    choice = {k: win[k] for k in ("name", "layers", "neurons", "alpha")} if beats else {"name": "no_patch"}
    fz = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "choice": choice, "guard_pass": len(passers),
          "n_candidates": len(cands), "finalists": [c["name"] for c in finalists], "search_s": round(time.time() - t_start, 1),
          "rule": {"layers": list(LAYERS), "neurons": list(NEURONS), "alphas": list(ALPHAS), "max_nll_delta": MAX_NLL_DELTA,
                   "stage1_tokens": STAGE1_TOKENS, "top": TOP}}
    jdump(out / "frozen.json", fz)
    print("[freeze]", json.dumps(fz), flush=True)
    if not beats:
        return

    # ---------------------------------------------------------------- test (once)
    tp = out / f"outputs_test_{win['name']}.jsonl"
    if not tp.exists():
        rows, dt = run(win, test_reqs, "test")
        print(f"[test] {win['name']} generated in {dt / 60:.1f} min", flush=True)
    wk = out / "_grade"
    wk.mkdir(exist_ok=True)
    (wk / "config.json").write_text(json.dumps(cfg))
    (wk / "requests_test.jsonl").write_text((mw / "requests_test.jsonl").read_text())
    arms = {"base": Path(a.saved_test) / "outputs_test_base.jsonl",
            "gsm8k_blind_n16_a12": Path(a.saved_test) / "outputs_test_blind_n16_a12.jsonl",
            "p50c": Path(a.saved_test) / "outputs_test_p50c.jsonl",
            f"escape_pick_{win['name']}": tp}
    graded = {}
    for k, f in arms.items():
        (wk / f.name).write_text(f.read_text())
        s = dm.score_math(wk, "requests_test.jsonl", f.name, tok)
        graded[k] = {m: round(s[m], 4) for m in ("n", "accuracy", "loop_killed", "truncated")} | {
            "ids": {r["id"]: int(r["correct"]) for r in s["rows"]}}
    pick = f"escape_pick_{win['name']}"
    paired = {}
    for ref in ("base", "gsm8k_blind_n16_a12", "p50c"):
        ids = sorted(graded[pick]["ids"])
        x = np.array([graded[pick]["ids"][i] - graded[ref]["ids"][i] for i in ids], float)
        rng = np.random.default_rng(0)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]
        paired[f"{pick}_vs_{ref}"] = {"diff": round(float(x.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                      round(float(np.percentile(bs, 97.5)), 4)], "a_only": int((x > 0).sum()), "b_only": int((x < 0).sum())}
    res = {"frozen": fz, "guard": guard, "stage1": s1,
           "stage2": {k: {kk: vv for kk, vv in v.items() if kk != "per_id"} for k, v in s2.items()},
           "test": {k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in graded.items()}, "paired": paired,
           "total_s": round(time.time() - t_start, 1)}
    jdump(out / "results.json", res)
    for k, v in res["test"].items():
        print(f"[test] {k:28s} acc {v['accuracy']:.3f} loop_killed {v['loop_killed']:.3f} trunc {v['truncated']:.3f}")
    for k, v in paired.items():
        print(f"[paired] {k}: {v}")
    print("saved", out / "results.json", flush=True)


if __name__ == "__main__":
    main()
