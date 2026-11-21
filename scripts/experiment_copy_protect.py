"""Localize the loop fix while protecting LEGITIMATE copying (teacher forcing, no generation).

The damage map showed the loop patch is an anti-repetition edit: it hurts names repeated from the context, the second
"step by step", repeated doctest lines. Earlier protection (#14, protect_neurons.py) skipped columns that hurt healthy
reasoning in general: damage -40..55%, accuracy unchanged. Here the cost is measured on the tokens the damage map
pointed at: COPY tokens (outside a loop, predictable by copying the context; wsi.heads.copy_labels). The goal is not
more accuracy on the 0.8B (it loops almost always) but less damage when the router fires on a healthy answer, which is
what a strong model pays.

Per column i of the edited down projections (edit W' = W - alpha u (u.w_i) on kept columns), first-order change of
the mean NLL of token set S in {loop, copy, other}:
    d_S(i) = -alpha (u.w_i) E_{t in S}[ h_i,t (g_t . u) ]        g_t = d(sum_{t' in S} NLL_t') / d y_t
d_loop > 0 is the wanted effect (continuing the loop gets less likely), d_copy > 0 is the damage to avoid.
Scored on the ELICIT answers (what discovery saw); every variant is then measured for real on the TEST answers.

Variants, all the same size as the frozen choice (layers, neurons per layer, alpha):
  orig          the frozen blind choice (top-n by the discovery score c)
  gen_skip10/25 walk the c ranking, skip the top 10/25% of columns by harm on all non-loop tokens (copy+other) (= #14)
  copy_skip10/25 same, harm on COPY tokens only
  fo_loop       top-n by first-order d_loop
  fo_spec1/3    top-n by d_loop - lam * d_copy (lam 1, 3)
  random        orig's columns with random directions (wsi.damage_map.random_like): damage of an unspecific edit

Held-out measures (test answers, actual NLL with the patch always on): mean dNLL on loop / copy / other tokens, and
on healthy answers only (no loop: what a false router fire costs). Pass: copy damage <= 0.5x orig's while keeping
>= 0.8x of orig's loop effect. --domains adds the damage map (wsi.damage_map) for orig, the best variant and random.

  python experiment_copy_protect.py --work <discover_blind run dir> --model Qwen/Qwen3.5-0.8B --out res.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import moi_loop_router_vllm as lr  # noqa: E402
from wsi.heads import COPY, LOOP, OTHER, copy_labels  # noqa: E402
from wsi.moipatch import Edit, MoIPatch, _dtype_name, _sha256, module_suffix  # noqa: E402

SETS = {"loop": LOOP, "copy": COPY, "other": OTHER}


# ------------------------------------------------------------------ data
def load_answers(work: Path, req_file: str, out_file: str, eos: set[int]) -> list[dict]:
    reqs = {r["id"]: r for r in lr.read_jsonl(work / req_file)}
    out = []
    for o in lr.read_jsonl(work / out_file):
        st = lr.analyse(o["token_ids"], eos)
        looping = st["kill"] is not None
        body = st["body"][: st["kill"]] if looping else st["body"]
        if len(body) < 128:
            continue
        out.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body, "looping": looping,
                    "onset": max(0, st["kill"] - lr.KILL_WINDOW) if looping else None})
    return out


def window(q: dict, max_len: int, after: int, warmup: int) -> dict:
    """One sequence per answer: prompt + answer (looping ones end `after` tokens into the loop)."""
    body, prompt = q["body"], q["prompt"]
    if q["looping"]:
        end = min(len(body), q["onset"] + after)
        start = 0 if len(prompt) + end <= max_len else end - max_len
        pre = prompt if start == 0 else []
        ids = pre + body[start:end]
        loop_from = len(pre) + q["onset"] - start
    else:
        pre, ids, loop_from = prompt, (prompt + body)[:max_len], None
    lab = copy_labels(ids, score_from=len(pre) + warmup, loop_from=loop_from)
    return {"id": q["id"], "ids": ids, "labels": lab[1:], "looping": q["looping"]}


def pick(answers: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Up to n answers, half of them non-looping when there are enough (they carry most COPY tokens)."""
    healthy = [q for q in answers if not q["looping"]]
    looping = [q for q in answers if q["looping"]]
    rng.shuffle(healthy)
    rng.shuffle(looping)
    h = healthy[: max(n // 2, n - len(looping))]
    return h + looping[: n - len(h)]


# ------------------------------------------------------------------ model helpers
def decoder_layers(model):
    return next(m for n, m in model.named_modules()
                if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 2)


def token_nll_logits(logits: torch.Tensor, ids: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """NLL of ids[1:] from logits[:-1], chunked (keeps the graph when logits require grad)."""
    out = []
    tgt = ids[1:]
    for s in range(0, len(tgt), chunk):
        lg = logits[s:s + chunk].float()
        out.append(torch.logsumexp(lg, -1) - lg.gather(-1, tgt[s:s + chunk, None])[:, 0])
    return torch.cat(out)


def set_gradients(model, downs: dict[int, torch.nn.Module], U: dict[int, torch.Tensor], wins: list[dict], dev) -> dict:
    """acc[S][li] = sum_{t} h_t (g_t . u), counts n[S]: the pieces of the first-order d_S(i)."""
    store: dict = {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, li=li: store.__setitem__(li, (inp[0][0], out)))
             for li, m in downs.items()]
    emb = model.get_input_embeddings()
    acc = {s: {li: torch.zeros(m.in_features, dtype=torch.float64, device=dev) for li, m in downs.items()} for s in SETS}
    n = {s: 0 for s in SETS}
    try:
        for w in wins:
            ids = torch.tensor(w["ids"], device=dev)
            lab = torch.from_numpy(np.asarray(w["labels"])).to(dev)
            x = emb(ids[None]).detach().requires_grad_(True)
            logits = model(inputs_embeds=x, use_cache=False).logits[0, :-1]
            nll = token_nll_logits(logits, ids)
            present = [s for s, c in SETS.items() if bool((lab == c).any())]
            for k, s in enumerate(present):
                m = lab == SETS[s]
                ys = [store[li][1] for li in downs]
                gs = torch.autograd.grad(nll[m].sum(), ys, retain_graph=k < len(present) - 1)
                for li, g in zip(downs, gs):
                    h = store[li][0].detach().float()
                    sc = g[0].float() @ U[li]
                    acc[s][li] += (h * sc[:, None]).sum(0).double()
                n[s] += int(m.sum())
            store.clear()
            del logits, nll
    finally:
        for h in hooks:
            h.remove()
    return {"acc": acc, "n": n}


def first_order(alpha: float, uw: dict[int, torch.Tensor], grads: dict) -> dict:
    """d[S][li][i]: first-order change of the mean NLL of set S when column i of layer li is edited."""
    return {s: {li: (-alpha * uw[li] * grads["acc"][s][li] / max(grads["n"][s], 1)).cpu().numpy() for li in uw}
            for s in SETS}


def select_skip(c_rank: list[int], harm: np.ndarray, q: float, n: int) -> list[int]:
    k = max(1, int(round(q * len(harm))))
    harmful = set(np.argsort(-harm)[:k].tolist()) & set(np.where(harm > 0)[0].tolist())
    return sorted([i for i in c_rank if i not in harmful][:n])


def variants(c_rank: dict[int, list[int]], d: dict, n: int, n_tok: dict) -> dict[str, dict[int, list[int]]]:
    out = {"orig": {li: sorted(c_rank[li][:n]) for li in c_rank}}
    for q in (0.10, 0.25):
        tag = int(q * 100)
        gen = {li: (d["copy"][li] * n_tok["copy"] + d["other"][li] * n_tok["other"]) / max(n_tok["copy"] + n_tok["other"], 1)
               for li in c_rank}
        out[f"gen_skip{tag}"] = {li: select_skip(c_rank[li], gen[li], q, n) for li in c_rank}
        out[f"copy_skip{tag}"] = {li: select_skip(c_rank[li], d["copy"][li], q, n) for li in c_rank}
    out["fo_loop"] = {li: sorted(np.argsort(-d["loop"][li])[:n].tolist()) for li in c_rank}
    for lam in (1, 3):
        out[f"fo_spec{lam}"] = {li: sorted(np.argsort(-(d["loop"][li] - lam * d["copy"][li]))[:n].tolist()) for li in c_rank}
    return out


def make_patch(model, downs: dict[int, torch.nn.Module], U: dict[int, torch.Tensor], cols: dict[int, list[int]],
               alpha: float, name: str) -> MoIPatch:
    want = {id(m) for m in downs.values()}
    names = {id(m): module_suffix(nm) for nm, m in model.named_modules() if id(m) in want}
    edits = [Edit(module=names[id(downs[li])], u=U[li].double().cpu(), alpha=alpha,
                  idx=torch.tensor(cols[li], dtype=torch.long), base_sha256=_sha256(downs[li].weight),
                  base_dtype=_dtype_name(downs[li].weight.dtype), shape=tuple(downs[li].weight.shape))
             for li in sorted(cols)]
    return MoIPatch(edits=edits, meta={"name": name})


@torch.no_grad()
def window_nll(model, wins: list[dict], dev) -> list[np.ndarray]:
    return [token_nll_logits(model(input_ids=torch.tensor(w["ids"], device=dev)[None], use_cache=False).logits[0, :-1],
                             torch.tensor(w["ids"], device=dev)).cpu().numpy() for w in wins]


def measure(model, patch: MoIPatch, wins: list[dict], base: list[np.ndarray], dev, verify: bool) -> dict:
    """Mean dNLL per set (pooled over tokens) + per-window means for paired comparisons."""
    handle = patch.attach(model, verify_fingerprint=verify)
    handle.on()
    try:
        new = window_nll(model, wins, dev)
    finally:
        handle.off()
    assert handle.check_restored(), "patch did not restore the weights"
    res, per = {}, {}
    for s, c in SETS.items():
        tot = cnt = 0.0
        pw = []
        for w, b, x in zip(wins, base, new):
            m = np.asarray(w["labels"]) == c
            if m.any():
                tot += float((x[m] - b[m]).sum())
                cnt += int(m.sum())
                pw.append(float((x[m] - b[m]).mean()))
            else:
                pw.append(None)
        res[s] = round(tot / cnt, 5) if cnt else None
        per[s] = pw
    hm = [(x - b)[np.asarray(w["labels"]) >= 0] for w, b, x in zip(wins, base, new) if not w["looping"]]
    hm = np.concatenate(hm) if hm else np.zeros(0)
    res["healthy_answers"] = round(float(hm.mean()), 5) if len(hm) else None
    res["per_window"] = per
    return res


def paired_ratio_ci(var: list, ref: list, n_boot: int = 2000, seed: int = 0) -> list | None:
    """Bootstrap 95% CI of mean(var)/mean(ref) over windows where both exist."""
    pairs = np.array([(a, b) for a, b in zip(var, ref) if a is not None and b is not None])
    if len(pairs) < 4 or abs(pairs[:, 1].mean()) < 1e-9:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), (n_boot, len(pairs)))
    r = pairs[idx, 0].mean(1) / np.where(np.abs(pairs[idx, 1].mean(1)) < 1e-12, np.nan, pairs[idx, 1].mean(1))
    return [round(float(np.nanpercentile(r, 2.5)), 3), round(float(np.nanpercentile(r, 97.5)), 3)]


def _f(x, spec: str = "+.4f") -> str:
    return "n/a" if x is None else format(x, spec)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="discover_blind run dir: config, elicit + test requests/outputs, "
                                                  "diagnosis_blind.json, blind_directions.npz, frozen.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=None, help="default: the frozen choice")
    ap.add_argument("--neurons", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--n-a", type=int, default=48, help="elicit answers scored for the first-order harm")
    ap.add_argument("--n-b", type=int, default=64, help="test answers for the held-out measure")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--after", type=int, default=512, help="tokens kept after the loop onset")
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--domains", default="", help="damage map domains, e.g. own_thinking,prose,code,chat")
    ap.add_argument("--n-docs", type=int, default=32)
    ap.add_argument("--data-dir", default="data_damage_map")
    ap.add_argument("--no-fingerprint", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    eos = set(cfg["eos"])
    choice = json.loads((work / "frozen.json").read_text())["choice"] if (work / "frozen.json").exists() else {}
    layers = [int(x) for x in a.layers.split(",")] if a.layers else choice["layers"]
    n_cols = a.neurons or int(choice["neurons"])
    alpha = a.alpha or float(choice["alpha"])
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    z = np.load(work / "blind_directions.npz")
    rng = random.Random(a.seed)
    wa = [window(q, a.max_len, a.after, a.warmup)
          for q in pick(load_answers(work, "requests_elicit.jsonl", "outputs_elicit.jsonl", eos), a.n_a, rng)]
    wb = [window(q, a.max_len, a.after, a.warmup)
          for q in pick(load_answers(work, "requests_test.jsonl", "outputs_test_base.jsonl", eos), a.n_b, rng)]

    def counts(wins):
        lab = np.concatenate([w["labels"] for w in wins])
        return {s: int((lab == c).sum()) for s, c in SETS.items()} | {"healthy_answers": sum(not w["looping"] for w in wins)}

    print(f"layers {layers}, {n_cols} cols/layer, alpha {alpha} | A (elicit) {counts(wa)} | B (test) {counts(wb)}", flush=True)

    dev = torch.device(a.device)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dl = decoder_layers(model)
    downs = {li: lr.down_module(dl[li]) for li in layers}
    U = {li: torch.from_numpy(z[f"u{li}"]).float().to(dev) for li in layers}
    uw = {li: (U[li] @ downs[li].weight.float()).double() for li in layers}

    grads = set_gradients(model, downs, U, wa, dev)
    d = first_order(alpha, uw, grads)
    print(f"first-order scored on A ({time.perf_counter() - t0:.0f} s)", flush=True)
    c_rank = {li: diag["neuron_rank"][str(li)] for li in layers}
    cols = variants(c_rank, d, n_cols, grads["n"])
    patches = {k: make_patch(model, downs, U, v, alpha, k) for k, v in cols.items()}
    from wsi.damage_map import random_like

    patches["random"] = random_like(patches["orig"], seed=a.seed)

    def predicted(v):
        return {s: round(float(sum(d[s][li][v[li]].sum() for li in v)), 5) for s in SETS}

    base_b = window_nll(model, wb, dev)
    meas = {}
    for k, p in patches.items():
        meas[k] = measure(model, p, wb, base_b, dev, not a.no_fingerprint)
        print(f"  {k:12s} loop {_f(meas[k]['loop'])} | copy {_f(meas[k]['copy'])} | other {_f(meas[k]['other'])} | "
              f"healthy answers {_f(meas[k]['healthy_answers'])}", flush=True)

    o = meas["orig"]
    summary = {}
    for k, m in meas.items():
        lr_ = m["loop"] / o["loop"] if o["loop"] else None
        cr = m["copy"] / o["copy"] if o["copy"] else None
        summary[k] = {"loop": m["loop"], "copy": m["copy"], "other": m["other"], "healthy_answers": m["healthy_answers"],
                      "loop_vs_orig": round(lr_, 3) if lr_ is not None else None,
                      "copy_vs_orig": round(cr, 3) if cr is not None else None,
                      "loop_vs_orig_ci95": paired_ratio_ci(m["per_window"]["loop"], o["per_window"]["loop"]),
                      "copy_vs_orig_ci95": paired_ratio_ci(m["per_window"]["copy"], o["per_window"]["copy"]),
                      "overlap_with_orig": (sum(len(set(cols[k][li]) & set(cols["orig"][li])) for li in layers)
                                            if k in cols else None),
                      "predicted_first_order": predicted(cols[k]) if k in cols else None,
                      "pass": bool(k not in ("orig", "random") and lr_ is not None and cr is not None
                                   and lr_ >= 0.8 and cr <= 0.5)}
    cand = [k for k in cols if k != "orig" and summary[k]["loop_vs_orig"] and summary[k]["loop_vs_orig"] >= 0.8]
    best = min(cand, key=lambda k: summary[k]["copy_vs_orig"]) if cand else None

    res = {"model": cfg.get("model_repo") or a.model, "work": work.name, "layers": layers, "neurons": n_cols,
           "alpha": alpha, "tokens_a": counts(wa), "tokens_b": counts(wb), "columns": {k: {str(li): v[li] for li in v}
                                                                                         for k, v in cols.items()},
           "summary": summary, "best": best, "passed": [k for k, v in summary.items() if v["pass"]]}

    if a.domains:
        import experiment_damage_map as edm
        from transformers import AutoTokenizer
        from wsi.damage_map import damage_map

        try:
            tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        except Exception:  # toy test models: only own_thinking (token ids) works
            tok = None
        doms = {}
        for dname in [x for x in a.domains.split(",") if x]:
            if dname == "own_thinking":
                seqs = edm.own_thinking(work, a.n_docs, 512)
            elif tok is None:
                continue
            else:
                seqs = [tok(t, add_special_tokens=False).input_ids[:512]
                        for t in edm.texts_for(dname, a.n_docs, Path(a.data_dir), tok)]
            doms[dname] = [s for s in seqs if len(s) >= 16]
        sel = {k: patches[k] for k in ("orig", best, "random") if k}
        dm = damage_map(model, doms, sel, decode=(lambda ids: tok.decode(ids)) if tok else str,
                        verify_fingerprint=not a.no_fingerprint)
        res["damage_map"] = dm

    res["wall_s"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("\nheld-out (test), mean dNLL with the patch always on; ratios vs orig")
    print(f"{'variant':12s} {'loop':>8s} {'copy':>8s} {'other':>8s} {'healthy':>8s} {'loop/o':>7s} {'copy/o':>7s}  pass")
    for k, v in summary.items():
        print(f"{k:12s} {_f(v['loop']):>8s} {_f(v['copy']):>8s} {_f(v['other']):>8s} {_f(v['healthy_answers']):>8s} "
              f"{_f(v['loop_vs_orig'], '.2f'):>7s} {_f(v['copy_vs_orig'], '.2f'):>7s}  {'PASS' if v['pass'] else ''}")
    print("best (loop >= 0.8x orig, least copy damage):", best, "| saved", a.out)


if __name__ == "__main__":
    main()
