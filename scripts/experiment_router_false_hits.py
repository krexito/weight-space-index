"""Hidden-state router vs text / embedding / reranker / asking the model: "same problem, new numbers or wording?"

Implements research/design_router_false_hits.md (read it first: data, arms and the decision rule are fixed there). Forward
passes only. One stored question per GSM-Symbolic template (key = `main` instance 0); queries: new numbers and rewordings
(hits), one-span flips that change the calculation (misses), plus p1 / p2 / other templates as sanity checks. Templates
are split once into calibration and test; everything trained, every threshold and every layer choice uses calibration
templates only (grouped 4-fold; thresholds of trained arms come from out-of-fold scores).

Batches hold prompts of exactly the same token length, so there is no padding to leak into hybrid (recurrent / conv)
models. The output has scores and rates only, no dataset text (CC BY-NC-ND, canary-marked).

  python scripts/experiment_router_false_hits.py --data <dir with main/p1/p2.jsonl> --out-dir <dir> --out <json> \
      --models Qwen/Qwen3.5-0.8B,LiquidAI/LFM3-1.7B --embed Qwen/Qwen3-Embedding-0.6B --rerank Qwen/Qwen3-Reranker-0.6B
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_router_false_hits as b  # noqa: E402
import router_false_hits_gate as g  # noqa: E402

T0 = time.time()
HITS, MISSES = ("number", "reword"), ("flip", "p1", "p2", "other")
TASK = "Given a math word problem, find problems that are solved by the same calculation, with only the numbers or the wording changed"
ASK = ("Two math word problems follow. Would the same calculation, with the numbers replaced, solve both? "
       "Answer yes if only the numbers, the names or the wording changed. Answer no if a different calculation or a "
       "different quantity is needed.\n\nProblem A: {a}\n\nProblem B: {b}\n\nAnswer yes or no.")


def log(msg: str) -> None:
    print(f"[router {time.time() - T0:7.1f}s] {msg}", flush=True)


def slug(name: str) -> str:
    return name.rstrip("/").replace("\\", "/").split("/")[-1].lower()


# ------------------------------------------------------------------ data
def load_rows(data: Path, config: str) -> list[dict]:
    rows = []
    for f in sorted(data.glob(f"{config}*.json*")):
        txt = f.read_text(encoding="utf-8")
        rows += [json.loads(x) for x in txt.splitlines() if x.strip()] if f.suffix == ".jsonl" else json.loads(txt)
    return rows


def build_records(data: Path, edits: dict, n_inst: int = 10, seed: int = 0) -> dict:
    """Records (one per query) and the calibration / test template split."""
    main = defaultdict(dict)
    orig = {}
    for r in load_rows(data, "main"):
        main[r["id"]][r["instance"]] = r["question"]
        orig[r["id"]] = r["original_id"]
    tpl = {t: [d[k] for k in sorted(d)] for t, d in main.items()}
    ok, bad = b.check(tpl, edits, n_inst=n_inst + 1)
    variants = {}
    for c in ("p1", "p2"):
        v = defaultdict(list)
        for r in sorted(load_rows(data, c), key=lambda r: r["instance"]):
            v[r["original_id"]].append(r["question"])
        variants[c] = v
    ok = sorted(ok)
    perm = ok[:]
    random.Random(seed).shuffle(perm)
    cal, test = set(perm[:len(perm) // 2]), set(perm[len(perm) // 2:])
    rng = random.Random(seed + 1)
    recs = []
    for t in ok:
        e, key = edits[str(t)], tpl[t][0]
        mine = cal if t in cal else test

        def add(kind, text, edit=None):
            recs.append({"t": t, "kind": kind, "key": key, "text": text, "cal": t in cal, "edit": edit})

        for q in tpl[t][1:n_inst + 1]:
            add("number", q)
            add("reword", b.apply(q, e["span"], e["reword"]), (e["span"], e["reword"]))
            add("flip", b.apply(q, e["span"], e["flip"]), (e["span"], e["flip"]))
        for c in ("p1", "p2"):
            for q in variants[c].get(orig[t], [])[:n_inst]:
                add(c, q)
        others = sorted(x for x in mine if x != t)
        for _ in range(n_inst):
            o = rng.choice(others)
            add("other", tpl[o][rng.randint(1, n_inst)])
    return {"records": recs, "cal": cal, "test": test, "ok": ok, "bad": bad, "keys": {t: tpl[t][0] for t in ok}}


# ------------------------------------------------------------------ model forward passes (equal-length batches, no padding)
def chat_text(tok, content: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)


def same_length_batches(ids: list[list[int]], bs: int):
    by = defaultdict(list)
    for i, x in enumerate(ids):
        by[len(x)].append(i)
    for L in sorted(by):
        for s in range(0, len(by[L]), bs):
            yield by[L][s:s + bs]


def model_call(model, x, hidden: bool):
    kw = {"logits_to_keep": 1} if getattr(model, "_keep_ok", True) else {}
    try:
        return model(input_ids=x, use_cache=False, output_hidden_states=hidden, **kw)
    except TypeError:
        if not kw:
            raise
        model._keep_ok = False
        return model(input_ids=x, use_cache=False, output_hidden_states=hidden)


@torch.no_grad()
def hidden_states(model, tok, texts: list[str], bs: int, step: int) -> torch.Tensor:
    """[N, n_layers, 2, D]: last prompt token and mean over the question's tokens, for every `step`-th hidden state."""
    dev = next(model.parameters()).device
    ids, qmask = [], []
    for t in texts:
        t = t.strip()
        full = chat_text(tok, t)
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        s = full.rindex(t)
        ids.append(enc["input_ids"])
        qmask.append([o[1] > s and o[0] < s + len(t) and o[1] > o[0] for o in enc["offset_mapping"]])
    out = None
    for idx in same_length_batches(ids, bs):
        x = torch.tensor([ids[i] for i in idx], device=dev)
        m = torch.tensor([qmask[i] for i in idx], device=dev).float()
        hs = model_call(model, x, True).hidden_states
        pick = list(range(1, len(hs), step))
        if len(hs) - 1 not in pick:
            pick.append(len(hs) - 1)
        rows = []
        for l in pick:
            h = hs[l].float()
            rows.append(torch.stack([h[:, -1], (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)], 1))
        v = torch.stack(rows, 1).cpu()  # [B, L, 2, D]
        if out is None:
            out = torch.zeros(len(texts), *v.shape[1:])
        out[torch.tensor(idx)] = v
    return out


@torch.no_grad()
def last_logits_diff(model, ids: list[list[int]], yes: list[int], no: list[int], bs: int, layers: list[int] | None = None):
    """logsumexp(yes) - logsumexp(no) at the last token; with `layers`, the hidden states of the last token too."""
    dev = next(model.parameters()).device
    score = np.zeros(len(ids))
    hid = None
    for idx in same_length_batches(ids, bs):
        x = torch.tensor([ids[i] for i in idx], device=dev)
        o = model_call(model, x, layers is not None)
        lp = o.logits[:, -1].float()
        score[idx] = (torch.logsumexp(lp[:, yes], 1) - torch.logsumexp(lp[:, no], 1)).cpu().numpy()
        if layers is not None:
            pick = [l for l in layers if l < len(o.hidden_states)]
            v = torch.stack([o.hidden_states[l][:, -1].float() for l in pick], 1).cpu()
            if hid is None:
                hid = torch.zeros(len(ids), *v.shape[1:])
            hid[torch.tensor(idx)] = v
    return score, hid


def yes_no_ids(tok) -> tuple[list[int], list[int]]:
    def first(words):
        return sorted({tok(w, add_special_tokens=False).input_ids[0] for w in words})

    y, n = first(["yes", "Yes", " yes", " Yes"]), first(["no", "No", " no", " No"])
    shared = set(y) & set(n)  # e.g. a bare space token
    return [i for i in y if i not in shared] or y, [i for i in n if i not in shared] or n


# ------------------------------------------------------------------ numerics
def auc(pos, neg) -> float:
    return g.auc(pos, neg) if len(pos) and len(neg) else float("nan")


def fit_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, iters: int = 25):
    """Weighted ridge logistic (standardised features, Newton)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = w / w.sum()
    R = np.eye(Z.shape[1]) * l2
    R[0, 0] = 1e-9
    th = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ th, -30, 30)))
        gr = Z.T @ (w * (p - y)) + R @ th
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + R
        th = th - np.linalg.solve(H + 1e-9 * np.eye(len(th)), gr)
        if np.abs(gr).max() < 1e-7:
            break
    return lambda A: np.c_[np.ones(len(A)), (A - mu) / sd] @ th


def kind_weights(kind: np.ndarray) -> np.ndarray:
    """Hits and misses half each; number = reword; flips half of the misses, the sanity kinds share the rest."""
    share = {"number": 0.25, "reword": 0.25, "flip": 0.25, "p1": 0.25 / 3, "p2": 0.25 / 3, "other": 0.25 / 3}
    n = Counter(kind)
    return np.array([share[k] / n[k] for k in kind])


def fit_metric(D: torch.Tensor, kind: np.ndarray, rank: int, lam: float, m: int = 128):
    """Score -|P d|^2 with P = top generalised eigenvectors of (flip second moment, hit second moment + shrinkage)."""
    h = torch.as_tensor(np.isin(kind, HITS), device=D.device)
    f = torch.as_tensor(kind == "flip", device=D.device)
    A = torch.cat([D[h], D[f]])
    n = min(m, A.shape[0] - 1, A.shape[1])
    ev_, U = torch.linalg.eigh(A @ A.T)
    B = A.T @ U[:, -n:] / ev_[-n:].clamp_min(1e-12).sqrt()  # [D, n] principal directions of the differences
    Xh, Xf = D[h] @ B, D[f] @ B
    Sh = Xh.T @ Xh / len(Xh)
    Sf = Xf.T @ Xf / len(Xf)
    Sh = Sh + lam * Sh.diagonal().mean() * torch.eye(n, dtype=D.dtype, device=D.device)
    L = torch.linalg.cholesky(Sh)
    M = torch.linalg.solve_triangular(L, torch.linalg.solve_triangular(L, Sf, upper=False).T, upper=False)
    _, V = torch.linalg.eigh((M + M.T) / 2)
    r = min(rank, n)
    P = B @ torch.linalg.solve_triangular(L.T, V[:, -r:], upper=True)
    return lambda X: -((X @ P) ** 2).sum(1).cpu().numpy()


# ------------------------------------------------------------------ arms
class Bench:
    def __init__(self, data: dict, seed: int = 0, n_folds: int = 4):
        self.recs = data["records"]
        self.kind = np.array([r["kind"] for r in self.recs])
        self.t = np.array([r["t"] for r in self.recs])
        self.cal = np.array([r["cal"] for r in self.recs])
        cal_t = sorted(data["cal"])
        random.Random(seed + 2).shuffle(cal_t)
        self.folds = [set(cal_t[i::n_folds]) for i in range(n_folds)]
        self.arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # name -> (out-of-fold calibration scores, final scores)
        self.info: dict[str, dict] = {}

    def add(self, name: str, final: np.ndarray, oof: np.ndarray | None = None, **info) -> None:
        self.arms[name] = (np.asarray(final if oof is None else oof, float), np.asarray(final, float))
        if info:
            self.info[name] = info

    def crossfit(self, fit_score):
        """fit_score(train mask) -> scores for every record. Returns (out-of-fold calibration scores, final scores)."""
        oof = np.full(len(self.recs), np.nan)
        for fold in self.folds:
            held = np.isin(self.t, list(fold))
            oof[held] = fit_score(self.cal & ~held)[held]
        return oof, fit_score(self.cal)

    def cv_auc(self, s: np.ndarray) -> float:
        """Mean of AUC(number vs flip) and AUC(reword vs flip) on the calibration rows that have a score."""
        m = self.cal & ~np.isnan(s)
        f = s[m & (self.kind == "flip")]
        return float(np.mean([auc(s[m & (self.kind == k)], f) for k in HITS]))


def select_metric(bench: Bench, feats, ranks, lams, kind=None, stop_info=None):
    """feats: {name: [N, D] differences}. Grouped-CV choice of (feature, rank, lam); returns scores and the choice."""
    kind = bench.kind if kind is None else kind
    best = None
    for name, D in feats.items():
        for rank in ranks:
            for lam in lams:
                oof = np.full(len(bench.recs), np.nan)
                for fold in bench.folds:
                    held = np.isin(bench.t, list(fold))
                    tr = bench.cal & ~held
                    score = fit_metric(D[torch.as_tensor(tr, device=D.device)], kind[tr], rank, lam)
                    hi = np.where(held)[0]
                    oof[hi] = score(D[torch.as_tensor(hi, device=D.device)])
                v = bench.cv_auc(oof)
                if best is None or v > best[0]:
                    best = (v, name, rank, lam, oof)
    v, name, rank, lam, oof = best
    D = feats[name]
    final = fit_metric(D[torch.as_tensor(bench.cal, device=D.device)], kind[bench.cal], rank, lam)(D)
    return oof, final, {"feature": name, "rank": rank, "shrink": lam, "cv_auc": round(v, 4)}


def text_arms(bench: Bench, edits: dict) -> None:
    """Text logistic, bag of changed words, and the same with the edited span known (how much vocabulary leaks)."""
    cal_keys = sorted({r["key"] for r in bench.recs if r["cal"]})
    vec = g.tfidf_fn(cal_keys)
    F = np.array([g.text_feats(vec, r["key"], r["text"]) for r in bench.recs])
    y = np.isin(bench.kind, HITS).astype(float)

    def logit(X):
        def f(tr):
            return fit_logit(X[tr], y[tr], kind_weights(bench.kind[tr]), 1e-3)(X)
        return f

    oof, fin = bench.crossfit(logit(F))
    bench.add("text_logistic", fin, oof)

    vocab = sorted({w for r in bench.recs if r["cal"] for w in g.mask(r["key"]) + g.mask(r["text"])})
    ix = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(bench.recs), 2 * len(vocab)))
    for n, r in enumerate(bench.recs):
        ca, cb = Counter(g.mask(r["key"])), Counter(g.mask(r["text"]))
        for w, c in (cb - ca).items():
            if w in ix:
                X[n, ix[w]] = c
        for w, c in (ca - cb).items():
            if w in ix:
                X[n, len(vocab) + ix[w]] = c
    X = X[:, (X[bench.cal] != 0).sum(0) >= 5]  # words seen in at least 5 calibration rows
    oof, fin = bench.crossfit(logit(X))
    bench.add("changed_words", fin, oof)

    words = sorted({w for e in edits.values() for k in e for w in b.words(e[k])})
    jx = {w: i for i, w in enumerate(words)}
    O = np.zeros((len(bench.recs), 2 * len(words)))
    for n, r in enumerate(bench.recs):
        if r["edit"]:
            old, new = Counter(b.words(r["edit"][0])), Counter(b.words(r["edit"][1]))
            for w, c in (new - old).items():
                O[n, jx[w]] = c
            for w, c in (old - new).items():
                O[n, len(words) + jx[w]] = c
    ok = np.isin(bench.kind, ("number", "reword", "flip"))

    def oracle(tr):
        tr = tr & ok
        return fit_logit(O[tr], y[tr], kind_weights(bench.kind[tr]), 1e-2)(O)

    oof, fin = bench.crossfit(oracle)
    bench.add("edited_words_oracle", fin, oof, note="sees the edited span: upper bound of the vocabulary leak")


def uniq_texts(bench: Bench) -> tuple[list[str], np.ndarray, np.ndarray]:
    texts = sorted({r["key"] for r in bench.recs} | {r["text"] for r in bench.recs})
    ix = {t: i for i, t in enumerate(texts)}
    return texts, np.array([ix[r["key"]] for r in bench.recs]), np.array([ix[r["text"]] for r in bench.recs])


def embedding_arm(bench: Bench, path: str, dtype, device: str, bs: int) -> None:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    keys = sorted({r["key"] for r in bench.recs})
    texts = sorted({r["text"] for r in bench.recs})
    ids = [tok(t).input_ids for t in keys] + [tok(f"Instruct: {TASK}\nQuery:{t}").input_ids for t in texts]
    vec = None
    with torch.no_grad():
        for idx in same_length_batches(ids, bs):
            h = model(input_ids=torch.tensor([ids[i] for i in idx], device=device), use_cache=False).last_hidden_state[:, -1].float()
            h = torch.nn.functional.normalize(h, dim=1).cpu()
            if vec is None:
                vec = torch.zeros(len(ids), h.shape[1])
            vec[torch.tensor(idx)] = h
    ki = {k: i for i, k in enumerate(keys)}
    ti = {t: len(keys) + i for i, t in enumerate(texts)}
    K = vec[[ki[r["key"]] for r in bench.recs]]
    Q = vec[[ti[r["text"]] for r in bench.recs]]
    bench.add("embedding_cosine", (K * Q).sum(1).numpy())
    D = (Q - K).double().to(device)
    oof, fin, info = select_metric(bench, {"embedding": D}, (8, 16, 32), (0.1, 1.0))
    bench.add("embedding_metric", fin, oof, **info)
    del model
    gc.collect()


def rerank_arm(bench: Bench, path: str, dtype, device: str, bs: int) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
           'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = [tok(f"{pre}<Instruct>: {TASK}\n<Query>: {r['key']}\n<Document>: {r['text']}{suf}", add_special_tokens=False).input_ids
           for r in bench.recs]
    yes, no = [tok("yes", add_special_tokens=False).input_ids[0]], [tok("no", add_special_tokens=False).input_ids[0]]
    s, _ = last_logits_diff(model, ids, yes, no, bs)
    bench.add("reranker", s)
    del model
    gc.collect()


def lm_arms(bench: Bench, model, tok, name: str, bs: int, step: int, pre: str, seed: int = 0) -> dict:
    dev = next(model.parameters()).device
    texts, ki, qi = uniq_texts(bench)
    t = time.time()
    H = hidden_states(model, tok, texts, bs, step)
    sec = {"hidden_s": round(time.time() - t, 1), "unique_prompts": len(texts)}
    log(f"{name}: {len(texts)} prompts, hidden {tuple(H.shape)} in {sec['hidden_s']} s")
    cal_i = np.unique(np.r_[ki[bench.cal], qi[bench.cal]])
    sd = H[torch.tensor(cal_i)].std(0) + 1e-6
    H = H / sd
    L, R = H.shape[1], H.shape[2]

    # untrained: cosine, layer and readout chosen on calibration
    best = None
    feats = {}
    for l in range(L):
        for rd in range(R):
            a, c = H[ki, l, rd], H[qi, l, rd]
            cos = torch.nn.functional.cosine_similarity(a, c, dim=1).numpy()
            v = bench.cv_auc(cos)
            if best is None or v > best[0]:
                best = (v, l, rd, cos)
            feats[f"h{l}_{'last' if rd == 0 else 'mean'}"] = (c - a).double().to(dev)
    bench.add(f"{pre}hidden_cosine", best[3], layer_index=best[1], readout=best[2], cv_auc=round(best[0], 4))

    # trained: low-rank metric on h(query) - h(key)
    oof, fin, info = select_metric(bench, feats, (4, 8, 16, 32), (0.1, 1.0))
    bench.add(f"{pre}hidden_router", fin, oof, **info)
    log(f"{name}: hidden router {info}")

    # controls at the chosen layer / readout / rank
    D = feats[info["feature"]]
    aucs, rng = [], random.Random(seed + 3)
    fin_s = None
    for i in range(10):
        flip_t = {t_ for t_ in sorted(set(bench.t[bench.cal].tolist())) if rng.random() < 0.5}
        k2 = bench.kind.copy()
        for t_ in flip_t:
            a = (bench.t == t_) & (bench.kind == "reword")
            c = (bench.t == t_) & (bench.kind == "flip")
            k2[a], k2[c] = "flip", "reword"
        s = fit_metric(D[torch.as_tensor(bench.cal, device=D.device)], k2[bench.cal], info["rank"], info["shrink"])(D)
        te = ~bench.cal
        aucs.append(auc(s[te & (bench.kind == "reword")], s[te & (bench.kind == "flip")]))
        fin_s = s
    bench.add(f"{pre}control_shuffled", fin_s, None, auc_reword_vs_flip_mean=round(float(np.mean(aucs)), 4), n=10)
    gen = torch.Generator().manual_seed(seed)
    P = torch.randn(D.shape[1], info["rank"], generator=gen, dtype=torch.float64)
    P = torch.linalg.qr(P)[0].to(D.device)
    bench.add(f"{pre}control_random", -((D @ P) ** 2).sum(1).cpu().numpy())
    del feats

    # the model reads both problems: yes / no answer, and a probe on the same hidden state
    ids = [tok(chat_text(tok, ASK.format(a=r["key"], b=r["text"])), add_special_tokens=False).input_ids for r in bench.recs]
    yes, no = yes_no_ids(tok)
    t = time.time()
    layers = list(range(1, L * step + 2, step))
    s, hid = last_logits_diff(model, ids, yes, no, bs, layers=layers)
    sec["pair_s"] = round(time.time() - t, 1)
    bench.add(f"{pre}ask_model", s, yes_ids=yes, no_ids=no)
    log(f"{name}: ask-the-model done in {sec['pair_s']} s")
    y = np.isin(bench.kind, HITS).astype(float)
    best = None
    for li in range(hid.shape[1]):
        X = (hid[:, li] / (hid[torch.as_tensor(bench.cal), li].std(0) + 1e-6)).double().to(dev)

        def probe(tr, X=X):
            mu = X[torch.as_tensor(tr, device=X.device)].mean(0)
            A = X[torch.as_tensor(tr, device=X.device)] - mu
            wide = A.shape[1] > A.shape[0]
            ev_, V = torch.linalg.eigh(A @ A.T if wide else A.T @ A)
            k = min(64, A.shape[0] - 1, A.shape[1])
            B = A.T @ V[:, -k:] / ev_[-k:].clamp_min(1e-12).sqrt() if wide else V[:, -k:]
            Z = ((X - mu) @ B).cpu().numpy()
            return fit_logit(Z[tr], y[tr], kind_weights(bench.kind[tr]), 1e-2)(Z)

        oof, fin = bench.crossfit(probe)
        v = bench.cv_auc(oof)
        if best is None or v > best[0]:
            best = (v, li, oof, fin)
    bench.add(f"{pre}pair_probe", best[3], best[2], layer_index=best[1], cv_auc=round(best[0], 4),
              note="secondary: trained probe on the pair prompt (one forward pass per stored problem, like a reranker)")
    return sec


# ------------------------------------------------------------------ scoring
def rates(bench: Bench, name: str, target: float = 0.95) -> dict:
    oof, s = bench.arms[name]
    thr = float(np.quantile(oof[bench.cal & np.isin(bench.kind, HITS)], 1 - target))
    te = ~bench.cal
    acc = s >= thr
    out = {"threshold": thr, "hit": {k: round(float(acc[te & (bench.kind == k)].mean()), 4) for k in HITS}}
    out["hit"]["all"] = round(float(acc[te & np.isin(bench.kind, HITS)].mean()), 4)
    out["false_hit"] = {k: (round(float(acc[te & (bench.kind == k)].mean()), 4) if (te & (bench.kind == k)).any() else None)
                        for k in MISSES}
    f = s[te & (bench.kind == "flip")]
    out["auc_vs_flip"] = {k: round(auc(s[te & (bench.kind == k)], f), 4) for k in HITS}
    out["_flip_by_template"] = {int(t): float(acc[te & (bench.kind == "flip") & (bench.t == t)].mean()) for t in sorted(set(bench.t[te]))}
    if name in bench.info:
        out["info"] = bench.info[name]
    return out


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> list[float]:
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(np.quantile(m, q)), 4) for q in (0.025, 0.975)]


def decide(res: dict, pre: str, shared: list[str]) -> dict:
    """The design's rule for one model. Baselines must keep >= 85 % of the test hits to count."""
    r = res[f"{pre}hidden_router"]
    base = shared + [f"{pre}ask_model", f"{pre}hidden_cosine"]
    el = {k: res[k] for k in base if k in res and res[k]["hit"]["all"] >= 0.85}
    pool = el or {k: res[k] for k in base if k in res}
    best = min(pool, key=lambda k: pool[k]["false_hit"]["flip"])
    ts = sorted(r["_flip_by_template"])
    a = np.array([r["_flip_by_template"][t] for t in ts])
    c = np.array([pool[best]["_flip_by_template"][t] for t in ts])
    ci = boot_ci(c - a)
    sh = res[f"{pre}control_shuffled"]["info"]["auc_reword_vs_flip_mean"]
    gap = round(float((c - a).mean()), 4)
    ok_hit = r["hit"]["number"] >= 0.9 and r["hit"]["reword"] >= 0.9
    ok_gap = gap >= 0.10 and ci[0] > 0
    ok_ctrl = 0.4 <= sh <= 0.6
    out = {"best_baseline": best, "no_eligible_baseline": not el, "router_flip_false_hit": r["false_hit"]["flip"],
           "baseline_flip_false_hit": pool[best]["false_hit"]["flip"], "gap_points": round(100 * gap, 1), "gap_ci95": ci,
           "hit_ok": bool(ok_hit), "gap_ok": bool(ok_gap), "shuffled_at_chance": bool(ok_ctrl), "passed": bool(ok_hit and ok_gap and ok_ctrl)}
    k = f"{pre}ask_model"
    if k in res:
        d = np.array([res[k]["_flip_by_template"][t] for t in ts]) - a
        out["ask_model_minus_router_flip_false_hit"] = {"diff": round(float(d.mean()), 4), "ci95": boot_ci(d)}
    return out


def cache_save(path: Path, bench: Bench, names: list[str]) -> None:
    arr = {f"{i}_{n}": bench.arms[k][n2] for i, k in enumerate(names) for n, n2 in (("oof", 0), ("final", 1))}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arr, _names=np.array(names), _info=np.array(json.dumps({k: bench.info.get(k, {}) for k in names})))
    os.replace(tmp, path)


def cache_load(path: Path, bench: Bench) -> bool:
    if not path.exists():
        return False
    z = np.load(path)
    names = [str(x) for x in z["_names"]]
    info = json.loads(str(z["_info"]))
    if len(z["0_final"]) != len(bench.recs):
        return False
    for i, k in enumerate(names):
        bench.add(k, z[f"{i}_final"], z[f"{i}_oof"], **info[k])
    return True


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--edits", default=str(b.EDITS))
    ap.add_argument("--out-dir", required=True, help="scores cache (resume); no dataset text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="Qwen/Qwen3.5-0.8B,LiquidAI/LFM3-1.7B")
    ap.add_argument("--embed", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--rerank", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--n-inst", type=int, default=10)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, a.dtype)
    edits = json.loads(Path(a.edits).read_text(encoding="utf-8"))["edits"]
    data = build_records(Path(a.data), edits, a.n_inst)
    bench = Bench(data)
    n_kind = {k: int((bench.kind == k).sum()) for k in HITS + MISSES}
    log(f"templates {len(data['ok'])} (cal {len(data['cal'])}, test {len(data['test'])}), bad {data['bad']}, records {n_kind}")
    res: dict = {"config": {k: v for k, v in vars(a).items()}, "n_templates": len(data["ok"]), "n_cal": len(data["cal"]),
                 "n_test": len(data["test"]), "dropped_by_check": {str(k): v for k, v in data["bad"].items()},
                 "records": n_kind, "timing": {}, "arms": {}, "decision": {}}

    def stage(file, fn):
        p = out_dir / file
        if cache_load(p, bench):
            log(f"{file}: loaded from cache")
            return
        before = set(bench.arms)
        fn()
        cache_save(p, bench, [k for k in bench.arms if k not in before])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stage("text.npz", lambda: text_arms(bench, edits))
    if a.embed:
        stage("embed.npz", lambda: embedding_arm(bench, a.embed, dtype, a.device, a.bs))
        log("embedding done")
    if a.rerank:
        stage("rerank.npz", lambda: rerank_arm(bench, a.rerank, dtype, a.device, a.bs))
        log("reranker done")
    shared = [k for k in bench.arms]
    res["shared_baselines"] = shared
    pres = {}
    for m in [x for x in a.models.split(",") if x]:
        pre = f"{slug(m)}/"
        pres[m] = pre

        def run(m=m, pre=pre):
            import eval_refusal_blind as ev

            model, tok = ev.load(m, a.dtype, a.device)
            try:
                res["timing"][slug(m)] = lm_arms(bench, model, tok, slug(m), a.bs, a.layer_step, pre)
            finally:
                del model
                gc.collect()

        stage(f"lm_{slug(m)}.npz", run)
    for k in bench.arms:
        res["arms"][k] = rates(bench, k)
    for m, pre in pres.items():
        res["decision"][m] = decide(res["arms"], pre, shared)
    res["passed_both"] = bool(res["decision"]) and all(d["passed"] for d in res["decision"].values())
    for k, r in res["arms"].items():
        log(f"{k:34s} hit num {r['hit']['number']:.2f} reword {r['hit']['reword']:.2f} | false hit flip {r['false_hit']['flip']:.2f} "
            f"p1 {r['false_hit']['p1']} p2 {r['false_hit']['p2']} other {r['false_hit']['other']} | AUC reword-flip {r['auc_vs_flip']['reword']:.3f}")
    for m, d in res["decision"].items():
        log(f"{m}: {d}")
    res["minutes"] = round((time.time() - T0) / 60, 1)
    tmp = Path(str(a.out) + ".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, a.out)
    return res


if __name__ == "__main__":
    main()
