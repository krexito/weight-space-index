"""Onset precursors: find a residual direction that warns BEFORE a failure becomes visible in the text.

The loop routers we have (16-gram text detector, the patch's own direction, a trained loop probe) all fire once the
model is already repeating. This module looks at the stretch just before that point and asks whether the residual
stream already carries a "loop is coming" signal there.

Labels per token of a failing answer, relative to its anchor (the first token where the text detector fires):
  * precursor (1):  [anchor - window, anchor)
  * ignored  (-1):  [anchor - window - gap, anchor - window) and everything from the anchor on
  * negative (0):   earlier tokens of the same answer
Every token of an answer that never fails is negative (0).

The direction per layer is a difference of means (precursor - negative), fitted on one half of the answers. Scores are
smoothed with a causal moving average; the alarm threshold is set on the fitting half so that about a target share of
answers raise a false alarm; the other half is scored. Pure numpy / torch, no model code: the caller supplies per-token residuals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

PRECURSOR, NEGATIVE, IGNORED = 1, 0, -1


def precursor_labels(n: int, anchor: int | None, window: int = 512, gap: int = 256) -> np.ndarray:
    """Token labels for one answer of length n. anchor=None means the answer never fails (all negative)."""
    lab = np.full(n, NEGATIVE, np.int8)
    if anchor is None:
        return lab
    a = int(min(max(anchor, 0), n))
    lab[a:] = IGNORED
    lab[max(0, a - window):a] = PRECURSOR
    lab[max(0, a - window - gap):max(0, a - window)] = IGNORED
    return lab


@dataclass
class ClassMeans:
    """Running per-layer sums of residuals for the precursor and negative classes."""
    sums: torch.Tensor | None = None  # [2, L, D] float64
    counts: torch.Tensor = field(default_factory=lambda: torch.zeros(2, dtype=torch.float64))

    def add(self, resid: torch.Tensor, labels: np.ndarray) -> None:
        """resid: [L, T, D] residuals of one answer's response tokens; labels: [T]."""
        lab = torch.as_tensor(labels, device=resid.device)
        if self.sums is None:
            self.sums = torch.zeros(2, resid.shape[0], resid.shape[2], dtype=torch.float64, device=resid.device)
            self.counts = self.counts.to(resid.device)
        for c, cls in ((0, NEGATIVE), (1, PRECURSOR)):
            m = lab == cls
            if m.any():
                self.sums[c] += resid[:, m].sum(1).double()
                self.counts[c] += m.sum()

    def directions(self) -> torch.Tensor:
        """Unit diff-of-means direction per layer, [L, D] float32 (precursor minus negative)."""
        if self.sums is None or (self.counts == 0).any():
            raise ValueError("need at least one precursor and one negative token")
        d = (self.sums[1] / self.counts[1] - self.sums[0] / self.counts[0]).float()
        return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def smooth(x: np.ndarray, w: int = 32) -> np.ndarray:
    """Causal moving average (the alarm at token t only sees tokens <= t)."""
    c = np.cumsum(np.insert(np.asarray(x, np.float64), 0, 0.0))
    k = np.arange(len(x))
    lo = np.maximum(0, k - w + 1)
    return (c[k + 1] - c[lo]) / (k + 1 - lo)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    r = x.argsort(kind="stable").argsort().astype(np.float64) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _pool(scores: list[np.ndarray], labels: list[np.ndarray], cls: int) -> np.ndarray:
    parts = [s[lab == cls] for s, lab in zip(scores, labels)]
    return np.concatenate(parts) if parts else np.zeros(0)


def early_warning(scores_a: list[np.ndarray], labels_a: list[np.ndarray],
                  scores_b: list[np.ndarray], labels_b: list[np.ndarray], anchors_b: list[int | None],
                  *, target_fpr: float = 0.10, level: str = "answer", smooth_w: int = 32,
                  useful_lead: int = 2048) -> dict:
    """Orientation and threshold from half A, metrics on half B.

    For each failing answer of half B the alarm time is the first token whose smoothed score passes the threshold.
    lead = anchor - alarm (positive = before the text detector). A lead above `useful_lead` is counted apart: an
    alarm that early is more likely noise than a warning.

    level="answer" (default): the threshold is the (1 - target_fpr) quantile of each half-A answer's highest score on
    its negative tokens, so about target_fpr of the answers raise a false alarm somewhere. level="token": quantile over
    all negative tokens; on answers thousands of tokens long even 1% of tokens means most answers fire at some point.
    """
    sm_a = [smooth(s, smooth_w) for s in scores_a]
    pa, na = _pool(sm_a, labels_a, PRECURSOR), _pool(sm_a, labels_a, NEGATIVE)
    auc_a = auc(pa, na)
    sign = 1.0 if not np.isfinite(auc_a) or auc_a >= 0.5 else -1.0
    if level == "answer":
        maxes = [float((sign * s[lab == NEGATIVE]).max()) for s, lab in zip(sm_a, labels_a) if (lab == NEGATIVE).any()]
        thr = float(np.quantile(maxes, 1 - target_fpr)) if maxes else float("inf")
    elif level == "token":
        thr = float(np.quantile(sign * na, 1 - target_fpr)) if len(na) else float("inf")
    else:
        raise ValueError(f"level must be 'answer' or 'token', got {level!r}")
    sm_b = [sign * smooth(s, smooth_w) for s in scores_b]
    pb, nb = _pool(sm_b, labels_b, PRECURSOR), _pool(sm_b, labels_b, NEGATIVE)
    leads, normal_fired, n_fail = [], [], 0
    for s, anc in zip(sm_b, anchors_b):
        hit = np.flatnonzero(s > thr)
        if anc is None:
            normal_fired.append(len(hit) > 0)
            continue
        n_fail += 1
        leads.append(int(anc - hit[0]) if len(hit) else None)
    got = [x for x in leads if x is not None]
    useful = [x for x in got if 0 < x <= useful_lead]

    def share(v):
        return round(len(v) / n_fail, 4) if n_fail else None

    return {"auc_a": round(auc_a, 4), "auc": round(auc(pb, nb), 4), "sign": sign, "level": level, "target_fpr": target_fpr,
            "warned_before_text": share([x for x in got if x > 0]),
            "warned_useful": share(useful),  # 0 < lead <= useful_lead
            "warned_too_early": share([x for x in got if x > useful_lead]),
            "median_lead_useful": float(np.median(useful)) if useful else None,
            "never_fired": share([x for x in leads if x is None]),
            "normal_answers_fired": round(float(np.mean(normal_fired)), 4) if normal_fired else None,
            "n_fail": n_fail, "n_normal": len(normal_fired)}
