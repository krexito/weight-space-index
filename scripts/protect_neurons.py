"""Does protecting the neurons that normal reasoning needs make the WSI loop fix better?

The WSI v3 loop fix (discover_loop_fix.py + discover_followup.py: layers 11, 15, 7, alpha 6, 64 columns each) picks
its down_proj columns by how much they push along the loop direction u:  c_i = (E[h_i|loop] - E[h_i|normal]) (u . w_i).
It never asks what editing a column costs on GOOD reasoning. Here that cost is measured and used to protect columns.

Harm of editing column i (first order, exact for this edit form). The edit changes the down_proj output by
dy = -alpha * u * (u . w_i) * h_i, so the change in the loss of good runs is
    harm_i = -alpha * (u . w_i) * E_t[ h_i,t * (g_t . u) ]          g_t = dLoss/dy_t (backward pass)
harm_i > 0 means the edit on column i makes the model's own correct, loop-free GSM8K-train answers less likely.
Measured on half A of those runs; half B is held out to check the protected fixes really hurt them less (actual NLL).

Variants (frozen here; dev decides; the original v3 fix is the reference):
  orig      top-64 by c                               (= v3; its dev and test outputs are reused from Drive)
  skip10    walk down the c ranking, skip columns in the top 10% of harm of their layer, take 64
  skip25    same, top 25%
  drop10    top-64 by c minus the top-10%-harm columns (fewer columns, no refill)

    python protect_neurons.py harm   --work W          (transformers, GPU; writes protect.json)
    python protect_neurons.py build  --work W --variant skip10
    python protect_neurons.py pick   --work W          (dev16 outputs of each variant -> choice)
    python protect_neurons.py report --work W
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402

LAYERS = (11, 15, 7)  # the WSI v3 choice (found by WSI on train data; see README), not p50c's
ALPHA = 6.0
N_COLS = 64
VARIANTS = {"skip10": ("skip", 0.10), "skip25": ("skip", 0.25), "drop10": ("drop", 0.10)}


def select(c_rank: list[int], harm: np.ndarray, kind: str, q: float, n: int = N_COLS) -> list[int]:
    """Columns for one layer. c_rank: columns by loop score, best first; harm: per column."""
    k = max(1, int(round(q * len(harm))))
    harmful = set(np.argsort(-harm)[:k].tolist()) & set(np.where(harm > 0)[0].tolist())
    if kind == "drop":
        return sorted(i for i in c_rank[:n] if i not in harmful)
    return sorted([i for i in c_rank if i not in harmful][:n])


def _good_runs(work: Path, eos: set) -> tuple[list[dict], list[dict]]:
    """Elicit runs that are correct, never loop and are not truncated, plus all labelled runs (for c)."""
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_elicit.jsonl")}
    s = dl.score(work, "requests_elicit.jsonl", "outputs_elicit.jsonl")
    good_ids = {r["id"] for r in s["rows"] if r["correct"] and not r["looped_killed"] and not r["truncated"]}
    good, labelled = [], []
    for o in lr.read_jsonl(work / "outputs_elicit.jsonl"):
        body = lr.strip_eos(o["token_ids"], eos)
        if o["id"] in good_ids:
            good.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body})
        lab = dl.token_labels(body)
        if (lab >= 0).any():
            labelled.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body, "lab": lab})
    return good, labelled


def cmd_harm(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    eos = set(cfg["eos"])
    good, labelled = _good_runs(work, eos)
    # same sequence cut and halves as discover_loop_fix.diagnose, so c reproduces the v3 columns
    seqs = []
    for s in labelled:
        lab = s["lab"]
        first = int(np.argmax(lab == 1)) if (lab == 1).any() else None
        end = min(len(s["body"]), (first + 768) if first is not None else 4096, 4096)
        seqs.append({**s, "body": s["body"][:end], "lab": lab[:end]})
    half_c = np.random.default_rng(cfg["seed"]).permutation(len(seqs)) % 2
    print(f"{len(good)} good runs (correct, no loop), {len(seqs)} labelled runs", flush=True)
    if len(good) < 6:
        raise SystemExit("too few good runs to measure harm")

    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["base_dir"], dtype=dtype).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = next(m for n, m in model.named_modules()
                  if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 2)
    downs = {li: layers[li].mlp.down_proj for li in a.layers}
    body_model = getattr(model, "model", model)
    lm_head = model.get_output_embeddings()
    z = np.load(work / "wsi_directions.npz")
    U = {li: torch.from_numpy(z[f"all_u{li}"]).float().to(dev) for li in a.layers}
    W = {li: downs[li].weight.detach().float() for li in a.layers}
    uw = {li: (U[li] @ W[li]).double() for li in a.layers}  # u . w_i, [cols]
    ncol = W[a.layers[0]].shape[1]

    # ---- c: loop vs normal mean activation (half A of the labelled runs, like v3)
    hsum = {(li, c): torch.zeros(ncol, dtype=torch.float64, device=dev) for li in a.layers for c in (0, 1)}
    hn = {(li, c): 0 for li in a.layers for c in (0, 1)}
    cur: dict = {}

    def fwd_c(li):
        def f(mod, inp, out):
            h = inp[0][0].float()
            for c in (0, 1):
                sel = cur["mask"] == c
                if sel.any():
                    hsum[(li, c)] += h[sel].sum(0).double()
                    hn[(li, c)] += int(sel.sum())
        return f

    hooks = [downs[li].register_forward_hook(fwd_c(li)) for li in a.layers]
    with torch.no_grad():
        for s, hb in zip(seqs, half_c):
            if hb != 0:
                continue
            ids = torch.tensor([s["prompt"] + s["body"]], device=dev)
            m = np.full(ids.shape[1], -1, dtype=np.int64)
            m[len(s["prompt"]):] = s["lab"]
            cur["mask"] = torch.from_numpy(m).to(dev)
            body_model(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    c_rank = {}
    for li in a.layers:
        dh = hsum[(li, 1)] / max(hn[(li, 1)], 1) - hsum[(li, 0)] / max(hn[(li, 0)], 1)
        c_rank[li] = torch.argsort(dh * uw[li], descending=True).tolist()

    # ---- harm: first-order loss change of the edit on good runs (half A), via a backward pass
    rng = np.random.default_rng(cfg["seed"] + 1)
    half_g = rng.permutation(len(good)) % 2
    emb = model.get_input_embeddings()

    def loss_positions(P: int, n_body: int) -> np.ndarray:
        pos = np.arange(P - 1, P + n_body - 1)
        if len(pos) > a.loss_tokens:
            pos = np.sort(rng.choice(pos, a.loss_tokens, replace=False))
        return pos

    acc = {li: torch.zeros(ncol, dtype=torch.float64, device=dev) for li in a.layers}
    store: dict = {}

    def fwd_g(li):
        def f(mod, inp, out):
            store[("h", li)] = inp[0][0].detach()
            out.register_hook(lambda g, li=li: store.__setitem__(("s", li), (g[0].float() @ U[li]).detach()))
        return f

    hooks = [downs[li].register_forward_hook(fwd_g(li)) for li in a.layers]
    n_loss = 0
    for s, hb in zip(good, half_g):
        if hb != 0:
            continue
        body = s["body"][: a.max_seq]
        ids = torch.tensor([s["prompt"] + body], device=dev)
        pos = loss_positions(len(s["prompt"]), len(body))
        x = emb(ids).detach().requires_grad_(True)
        hid = body_model(inputs_embeds=x, use_cache=False).last_hidden_state[0]
        p = torch.from_numpy(pos).to(dev)
        logits = lm_head(hid[p]).float()
        loss = torch.nn.functional.cross_entropy(logits, ids[0, p + 1], reduction="sum")
        loss.backward()
        for li in a.layers:
            acc[li] += (store[("h", li)].float() * store[("s", li)][:, None]).sum(0).double()
        n_loss += len(pos)
        store.clear()
    for h in hooks:
        h.remove()
    harm = {li: (-a.alpha * uw[li] * acc[li] / max(n_loss, 1)).cpu().numpy() for li in a.layers}

    cols = {"orig": {li: sorted(c_rank[li][:N_COLS]) for li in a.layers}}
    for name, (kind, q) in VARIANTS.items():
        cols[name] = {li: select(c_rank[li], harm[li], kind, q) for li in a.layers}

    # ---- held-out check: actual NLL of half-B good runs with each variant applied in memory
    def nll_B() -> float:
        tot, n = 0.0, 0
        r2 = np.random.default_rng(cfg["seed"] + 2)
        with torch.no_grad():
            for s, hb in zip(good, half_g):
                if hb != 1:
                    continue
                body = s["body"][: a.max_seq]
                ids = torch.tensor([s["prompt"] + body], device=dev)
                P = len(s["prompt"])
                pos = np.arange(P - 1, P + len(body) - 1)
                if len(pos) > a.loss_tokens:
                    pos = np.sort(r2.choice(pos, a.loss_tokens, replace=False))
                hid = body_model(input_ids=ids, use_cache=False).last_hidden_state[0]
                p = torch.from_numpy(pos).to(dev)
                tot += float(torch.nn.functional.cross_entropy(lm_head(hid[p]).float(), ids[0, p + 1],
                                                               reduction="sum"))
                n += len(pos)
        return tot / max(n, 1)

    orig_w = {li: downs[li].weight.detach().clone() for li in a.layers}
    held = {"base": nll_B()}
    for name, cc in cols.items():
        for li in a.layers:
            Wd = orig_w[li].double()
            u = U[li].double()
            m = torch.zeros(ncol, dtype=torch.float64, device=dev)
            m[cc[li]] = 1.0
            downs[li].weight.data.copy_((Wd - a.alpha * torch.outer(u, m * (u @ Wd))).to(orig_w[li].dtype))
        held[name] = nll_B()
        for li in a.layers:
            downs[li].weight.data.copy_(orig_w[li])
        print(f"  held-out NLL on good runs: {name:7s} {held[name]:.4f}  (base {held['base']:.4f})", flush=True)

    diag_p = work / "diagnosis.json"
    v3 = json.loads(diag_p.read_text()).get("neurons", {}) if diag_p.exists() else {}
    out = {"layers": list(a.layers), "alpha": a.alpha, "n_good_runs": len(good), "loss_tokens_A": n_loss,
           "columns": {k: {str(li): v for li, v in d.items()} for k, d in cols.items()},
           "n_columns": {k: {str(li): len(v) for li, v in d.items()} for k, d in cols.items()},
           "predicted_harm": {k: round(float(sum(harm[li][d[li]].sum() for li in a.layers)), 6)
                              for k, d in cols.items()},
           "heldout_nll": {k: round(v, 5) for k, v in held.items()},
           "harm_of_orig_columns_pct_rank": {
               str(li): round(float(np.mean([(harm[li] < harm[li][i]).mean() for i in cols["orig"][li]])), 3)
               for li in a.layers},
           "orig_matches_v3_columns": {str(li): f"{len(set(cols['orig'][li]) & set(v3.get(str(li), [])))}/{N_COLS}"
                                       for li in a.layers}}
    np.savez(work / "protect_harm.npz", **{f"harm{li}": harm[li] for li in a.layers},
             **{f"crank{li}": np.array(c_rank[li]) for li in a.layers})
    (work / "protect.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_columns", "predicted_harm", "heldout_nll", "orig_matches_v3_columns",
                                          "harm_of_orig_columns_pct_rank")}, indent=1))


def build_variant(work: Path, variant: str, out: Path) -> Path:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = dl.cfg_load(work)
    base = Path(cfg["base_dir"])
    pr = json.loads((work / "protect.json").read_text())
    z = np.load(work / "wsi_directions.npz")
    cols = pr["columns"][variant]
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns(".cache", "*.lock"))
    done = set()
    for f in sorted(base.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            keys = list(fh.keys())
            targets = {}
            for li in pr["layers"]:
                try:
                    targets[dl.down_key(keys, li)] = li
                except SystemExit:
                    continue
            if not targets:
                continue
            tensors = {k: fh.get_tensor(k).clone() for k in keys}
            meta = fh.metadata() or {"format": "pt"}
        for k, li in targets.items():
            Wd = tensors[k].double()
            u = torch.from_numpy(z[f"all_u{li}"]).double()
            m = torch.zeros(Wd.shape[1], dtype=torch.float64)
            m[cols[str(li)]] = 1.0
            tensors[k] = (Wd - pr["alpha"] * torch.outer(u, m * (u @ Wd))).to(tensors[k].dtype)
            done.add(li)
        save_file(tensors, str(out / f.name), metadata=meta)
    if done != set(pr["layers"]):
        raise SystemExit(f"down_proj missing for {sorted(set(pr['layers']) - done)}")
    return out


def cmd_build(a) -> None:
    work = Path(a.work)
    out = build_variant(work, a.variant, Path(a.out_dir) if a.out_dir else work / f"protect-{a.variant}")
    print(out)


def dev_file(v: str) -> str:
    return "outputs_dev16_fa3_a6.jsonl" if v == "orig" else f"outputs_dev16_protect_{v}.jsonl"


def cmd_pick(a) -> None:
    work = Path(a.work)
    res = {}
    for v in ["orig", *VARIANTS]:
        if (work / dev_file(v)).exists():
            s = dl.score(work, "requests_dev16.jsonl", dev_file(v))
            res[v] = {k: x for k, x in s.items() if k != "rows"}
            print(f"  {v:7s} acc={s['accuracy']:.1%} loops={s['loop_killed']:.1%} trunc={s['truncated']:.1%}")
    if "orig" not in res:
        raise SystemExit(f"{dev_file('orig')} missing: restore the v3 run from Drive")
    order = ["orig", *VARIANTS]  # ties -> the original fix stays
    best = max(res, key=lambda k: (res[k]["accuracy"], -res[k]["loop_killed"], -order.index(k)))
    pr = json.loads((work / "protect.json").read_text())
    pr["dev16"], pr["choice"] = res, best
    (work / "protect.json").write_text(json.dumps(pr, indent=1))
    print("choice:", best)


def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import discover_followup as fu

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    tok = AutoTokenizer.from_pretrained(cfg["base_dir"])
    eos, reqs = set(cfg["eos"]), {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    pr = json.loads((work / "protect.json").read_text())
    ch = pr.get("choice")
    rows = {"base": {r["id"]: r for r in dl.score(work, "requests_test.jsonl", "outputs_test_base.jsonl", tok)["rows"]}}
    rows["v3_fixed"] = {r["id"]: r for r in dl.score(work, "requests_test.jsonl", "outputs_test_wsi3.jsonl", tok)["rows"]}
    rows["v3_moi"] = fu.moi_arm(work, "outputs_test_moi3.jsonl", reqs, rows["base"], tok, eos)
    if ch and ch != "orig":
        rows["protected_fixed"] = {r["id"]: r for r in dl.score(work, "requests_test.jsonl",
                                                                  "outputs_test_protect.jsonl", tok)["rows"]}
        rows["protected_moi"] = fu.moi_arm(work, "outputs_test_moi_protect.jsonl", reqs, rows["base"], tok, eos)
    ids = sorted(reqs)
    acc = {k: [bool(v[i]["correct"]) for i in ids] for k, v in rows.items()}
    pairs = [("protected_fixed", "v3_fixed"), ("protected_moi", "v3_moi"), ("protected_fixed", "base"),
             ("v3_fixed", "base")]
    res = {"choice": ch, "protect": {k: pr[k] for k in pr if k != "columns"},
           "arms": {k: fu.summary(v) for k, v in rows.items()},
           "paired": {f"{x}_vs_{y}": lr.paired_ci(acc[x], acc[y]) for x, y in pairs if x in acc and y in acc}}
    if ch == "orig":
        res["note"] = "no protected variant beat the original v3 fix on dev; test not re-run"
    (work / a.out).write_text(json.dumps(res, indent=1))
    for k, v in res["arms"].items():
        print(f"{k:16s} {v['accuracy']:7.1%} {v['loop_killed']:7.1%} {v['truncated']:7.1%}")
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
    print("Saved", work / a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harm")
    h.add_argument("--work", required=True)
    h.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=list(LAYERS))
    h.add_argument("--alpha", type=float, default=ALPHA)
    h.add_argument("--loss-tokens", type=int, default=1024, help="response tokens per run that carry the loss")
    h.add_argument("--max-seq", type=int, default=4096, help="response tokens per good run fed to the model")
    h.add_argument("--cpu", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--variant", required=True, choices=["orig", *VARIANTS])
    b.add_argument("--out-dir")
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--requests", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--gpu-mem", type=float, default=0.88)
    g.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    p = sub.add_parser("pick")
    p.add_argument("--work", required=True)
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--out", default="results_protect_neurons.json")
    a = ap.parse_args()
    fns = {"harm": cmd_harm, "build": cmd_build, "gen": dl.cmd_gen, "pick": cmd_pick, "report": cmd_report}
    key = {"gen": lambda: f"gen:{Path(a.out).stem}:{Path(a.model).name}", "harm": lambda: "protect-harm"}.get(a.cmd)
    dl.run_timed(a, fns, key() if key else None)


if __name__ == "__main__":
    main()
