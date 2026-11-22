"""Go / no-go of research/design_router_false_hits.md (CPU, no model): can plain text tell a `flip` from a `reword`?

Both are one-span edits of the same place, so a separation here means the edits leak a shortcut and must be rewritten
before spending GPU time. Two classifiers, always trained and tested on disjoint templates:

  text      length, edit ratio, TF-IDF cosine and token-count differences against the stored question (the pilot's)
  words     bag of added / removed words (does "half" or "less" give it away?)
  both      the two together

Results: AUC over random half/half template splits (mean, 5-95 %), threshold of the design: mean AUC <= 0.75.

  python scripts/router_false_hits_gate.py --data <dir> [--out gate.json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_router_false_hits as b  # noqa: E402


def mask(s: str) -> list[str]:
    return ["#" if w[0].isdigit() else w for w in b.words(s)]


def tfidf_fn(texts: list[str]):
    grams = lambda s: Counter(mask(s)) + Counter(zip(mask(s), mask(s)[1:]))  # noqa: E731
    df = Counter(g for t in texts for g in grams(t))
    n = len(texts)

    def vec(s):
        v = {g: (1 + math.log(c)) * math.log((n + 1) / (min(df[g], n) + 1) + 1) for g, c in grams(s).items()}
        z = math.sqrt(sum(x * x for x in v.values()))
        return {g: x / z for g, x in v.items()}

    return vec


def text_feats(vec, key: str, q: str) -> list[float]:
    a, c = mask(key), mask(q)
    va, vc = vec(key), vec(q)
    ca, cc = Counter(a), Counter(c)
    return [sum(x * vc.get(g, 0) for g, x in va.items()), difflib.SequenceMatcher(None, a, c, autojunk=False).ratio(),
            min(len(key), len(q)) / max(len(key), len(q)), (len(c) - len(a)) / len(a),
            sum((cc - ca).values()) / len(a), sum((ca - cc).values()) / len(a), abs(a.count("#") - c.count("#"))]


def fit(X: np.ndarray, y: np.ndarray, l2: float):
    """Standardised logistic regression by Newton steps (dual form when there are more features than rows)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    G = Z @ Z.T + 1.0  # linear kernel + bias
    al = np.zeros(len(y))
    t = 2 * y - 1
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(G @ al, -30, 30)))
        w = p * (1 - p) + 1e-6
        al = np.linalg.solve(np.diag(w) @ G + l2 * len(y) * np.eye(len(y)), w * (G @ al) + (y - p))
    return lambda A: ((A - mu) / sd @ Z.T + 1.0) @ al


def auc(pos, neg) -> float:
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float(((p > n) + 0.5 * (p == n)).mean())


def run(tpl: dict, edits: dict, ok: list[int], n_inst: int = 3, n_splits: int = 100, seed: int = 0) -> dict:
    vec = tfidf_fn([tpl[t][0] for t in ok])
    vocab = sorted({w for t in ok for k in ("span", "flip", "reword") for w in b.words(edits[str(t)][k])})
    ix = {w: i for i, w in enumerate(vocab)}
    T, y, F, W = [], [], [], []
    for t in ok:
        e = edits[str(t)]
        for lab, k in ((1.0, "reword"), (0.0, "flip")):
            for q in tpl[t][1:1 + n_inst]:
                v = b.apply(q, e["span"], e[k])
                T.append(t)
                y.append(lab)
                F.append(text_feats(vec, tpl[t][0], v))
                w = np.zeros(2 * len(vocab))
                old, new = Counter(b.words(e["span"])), Counter(b.words(e[k]))
                for g, c in (new - old).items():
                    w[ix[g]] = c
                for g, c in (old - new).items():
                    w[len(vocab) + ix[g]] = c
                W.append(w)
    T, y, F, W = np.array(T), np.array(y), np.array(F), np.array(W)
    sets = {"text": F, "words": W, "both": np.c_[F, W]}
    rng = random.Random(seed)
    res = {k: [] for k in sets}
    for _ in range(n_splits):
        perm = ok[:]
        rng.shuffle(perm)
        cal = np.isin(T, perm[:len(perm) // 2])
        for k, X in sets.items():
            s = fit(X[cal], y[cal], 1e-2)(X[~cal])
            res[k].append(auc(s[y[~cal] == 1], s[y[~cal] == 0]))
    out = {k: {"auc_mean": round(float(np.mean(v)), 3), "q05": round(float(np.quantile(v, 0.05)), 3),
               "q95": round(float(np.quantile(v, 0.95)), 3)} for k, v in res.items()}
    # shuffled labels: swap flip/reword inside random templates
    perm_y = y.copy()
    r2 = np.random.default_rng(seed)
    for t in ok:
        if r2.random() < 0.5:
            perm_y[T == t] = 1 - perm_y[T == t]
    v = []
    for _ in range(n_splits):
        perm = ok[:]
        rng.shuffle(perm)
        cal = np.isin(T, perm[:len(perm) // 2])
        s = fit(W[cal], perm_y[cal], 1e-2)(W[~cal])
        v.append(auc(s[perm_y[~cal] == 1], s[perm_y[~cal] == 0]))
    out["words_shuffled_labels"] = {"auc_mean": round(float(np.mean(v)), 3)}
    out["n_templates"], out["n_splits"], out["limit"] = len(ok), n_splits, 0.75
    out["go"] = all(out[k]["auc_mean"] <= 0.75 for k in sets)
    return out


def top_words(edits: dict, ok: list[int], n: int = 15) -> list:
    c = Counter()
    for t in ok:
        e = edits[str(t)]
        for lab, k in ((1, "reword"), (-1, "flip")):
            for g in (Counter(b.words(e[k])) - Counter(b.words(e["span"]))):
                c[g, lab] += 1
    words_ = {g for g, _ in c}
    return sorted(((g, c[g, 1], c[g, -1]) for g in words_), key=lambda x: -(x[1] + x[2]))[:n]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--edits", default=str(b.EDITS))
    ap.add_argument("--out")
    ap.add_argument("--splits", type=int, default=100)
    a = ap.parse_args(argv)
    tpl = b.load_config(Path(a.data), "main")
    edits = json.loads(Path(a.edits).read_text(encoding="utf-8"))["edits"]
    ok, bad = b.check(tpl, edits)
    res = run(tpl, edits, ok, n_splits=a.splits)
    print(json.dumps(res, indent=1))
    print("added words (word, in rewords, in flips):", top_words(edits, ok))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
