"""Over-refusal fix without the voice change: null-space-constrained and cleaned refusal edits (Aion-2.3B).

The blind agent's patch (v2) fixed over-refusal on microsoft/Aion-1.0-Instruct-2.3B and kept the real refusals, but
41% of the answers it recovered open with an exam-solver template the base never uses ("The question asks: ...").
Its edit projects the refusal direction u out of all 56 residual writers for the whole answer, every token.

This script keeps v2's direction, alpha and router (gate + veto, so the toxic refusals stay protected the same way)
and changes only how the edit is applied, then scores every arm with the judge (scripts/eval_refusal_blind.py, style
check included) on the same held-out test split:

  null<E>     AlphaSteer idea (arXiv 2506.07022): each writer W gets W' = W - a u v^T with v = (u^T W) restricted to the
              complement of the input subspace that holds E (90 / 99 %) of the energy of the base model's own answer
              tokens. Answer-body tokens then see (almost) no change; the edit still acts where the inputs leave that
              subspace (the prompt end and the first answer tokens, where the refusal is decided).
  clean       Surgical Refusal Ablation idea (arXiv 2601.08489): u is ridge-residualised against "protected" atoms
              (an exam-template direction measured from v2's own recovered answers, and a math-vs-general direction)
              before the usual projection.
  clean_null  both.

Only agent-split data is used to build the edits (the judge's test files are never read until scoring). Patches stay
in --out-dir (they are not meant for the repo). The goal is the surgical fix: answer the benign prompts the model
over-refuses, keep refusing the harmful ones, keep ordinary answers identical, and now keep the model's voice.

  python scripts/experiment_overrefusal_nullspace_aion.py --base-dir /content/base --work <judge work dir> \
      --v2-patch <work>/v2_blind.moipatch --out-dir <dir> --out <json>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_refusal_blind as ev  # noqa: E402
from wsi.moipatch import Edit, MoIPatch  # noqa: E402

T0 = time.time()
EXAM_PREFIX = "the question"  # v2's new openings: "the question asks / states / is"


def log(msg: str) -> None:
    print(f"[nullspace {time.time() - T0:7.1f}s] {msg}", flush=True)


def module_map(model) -> dict:
    from wsi.moipatch import module_suffix

    out = {}
    for name, mod in model.named_modules():
        try:
            out[module_suffix(name)] = mod
        except ValueError:
            pass
    return out


# ------------------------------------------------------------------ reference answers (agent split only)
def tf_batches(tok, prompts: list[str], answers: list[str], bs: int):
    """Teacher-forcing batches (right padded): ids, attention, answer mask with the answer's token positions."""
    rows = []
    for p, a in zip(prompts, answers):
        pi = ev.encode(tok, p)
        ai = tok(a, add_special_tokens=False).input_ids
        rows.append((pi, ai))
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        L = max(len(pi) + len(ai) for pi, ai in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        ans_pos = torch.full((len(chunk), L), -1, dtype=torch.long)  # answer token index, -1 elsewhere
        last_prompt = torch.zeros(len(chunk), dtype=torch.long)
        for j, (pi, ai) in enumerate(chunk):
            seq = pi + ai
            ids[j, :len(seq)] = torch.tensor(seq)
            att[j, :len(seq)] = 1
            ans_pos[j, len(pi):len(seq)] = torch.arange(len(ai))
            last_prompt[j] = len(pi) - 1
        yield ids, att, ans_pos, last_prompt


@torch.no_grad()
def collect_cov(model, tok, mods: dict, names: list[str], prompts, answers, bs: int, skip_onset: int, cov_device: str):
    """Sum of x x^T over the inputs of each writer at answer tokens with index >= skip_onset (base model, no edit)."""
    dev = next(model.parameters()).device
    covs = {n: torch.zeros(mods[n].weight.shape[1], mods[n].weight.shape[1], device=cov_device) for n in names}
    count = {"n": 0}
    sel = {}

    def mk(n):
        def pre(_m, inp):
            x = inp[0]
            m = sel["mask"]
            xs = x[m].float().to(cov_device)
            covs[n].addmm_(xs.T, xs)
        return pre

    hooks = [mods[n].register_forward_pre_hook(mk(n)) for n in names]
    try:
        for ids, att, ans_pos, _lp in tf_batches(tok, prompts, answers, bs):
            mask = (ans_pos >= skip_onset).to(dev)
            if not mask.any():
                continue
            sel["mask"] = mask
            count["n"] += int(mask.sum())
            model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return covs, count["n"]


@torch.no_grad()
def residual_means(model, tok, layer: int, prompts, answers, bs: int, where: str) -> torch.Tensor:
    """Mean residual at hidden_states[layer] (= input of block `layer`, v2's convention) over answer tokens or the
    last prompt token."""
    dev = next(model.parameters()).device
    acc, cnt = None, 0
    for ids, att, ans_pos, lp in tf_batches(tok, prompts, answers, bs):
        h = model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False,
                  output_hidden_states=True).hidden_states[layer].float()
        if where == "answer":
            x = h[(ans_pos >= 0).to(dev)]
        else:
            x = h[torch.arange(h.shape[0], device=dev), lp.to(dev)]
        s = x.sum(0).double().cpu()
        acc = s if acc is None else acc + s
        cnt += x.shape[0]
    return acc / max(cnt, 1)


def null_rows(cov: torch.Tensor, rows: dict[str, torch.Tensor], energies: list[float]) -> dict:
    """For each row and energy E: the row minus its part in the top-k eigen-directions holding E of the answer-token
    input energy (one eigendecomposition per module, shared by every row)."""
    evals, evecs = torch.linalg.eigh(cov.double() if cov.device.type == "cpu" else cov.float())
    evals, evecs = evals.flip(0).clamp_min(0).double().cpu(), evecs.flip(1).double().cpu()
    cum = evals.cumsum(0) / evals.sum().clamp_min(1e-30)
    out = {}
    for e in energies:
        k = int((cum < e).sum()) + 1
        U = evecs[:, :k]
        for key, row in rows.items():
            v = row - (row @ U) @ U.T
            out[(key, e)] = {"v": v, "k": k, "kept_norm": float(v.norm() / row.norm().clamp_min(1e-30)),
                             "dim": int(row.numel())}
    return out


def ridge_clean(u: torch.Tensor, atoms: dict[str, torch.Tensor], lam: float) -> tuple[torch.Tensor, dict]:
    names = [k for k, a in atoms.items() if a is not None and float(a.norm()) > 0]
    if not names:
        return u.clone(), {"atoms": [], "note": "no atoms"}
    A = torch.stack([atoms[k] / atoms[k].norm() for k in names], 1).double()  # [D, n]
    uu = u.double() / u.double().norm()
    w = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], dtype=torch.float64), A.T @ uu)
    uc = uu - A @ w
    uc = uc / uc.norm()
    rep = {"atoms": names, "lambda": lam, "cos_u_clean": round(float(uu @ uc), 4),
           "cos_before": {k: round(float(uu @ A[:, i]), 4) for i, k in enumerate(names)},
           "cos_after": {k: round(float(uc @ A[:, i]), 4) for i, k in enumerate(names)}}
    return uc, rep


def with_ratio(sc: dict, n_toxic: int = 200) -> dict:
    """Suppression ratio: over-refusal drop / harmful-refusal drop (paired vs base; the toxic drop floored at 1 prompt)."""
    d_or = -sc["paired_over_refusal_vs_base"]["diff"]
    d_tox = -sc["paired_toxic_refusal_vs_base"]["diff"]
    sc["over_refusal_drop"], sc["toxic_refusal_drop"] = round(d_or, 4), round(d_tox, 4)
    sc["suppression_ratio"] = round(d_or / max(d_tox, 1 / n_toxic), 2)
    return sc


def build(v2: MoIPatch, mods: dict, u: torch.Tensor, vrows: dict | None, name: str, desc: str) -> MoIPatch:
    edits = []
    for e in v2.edits:
        v = vrows[e.module] if vrows is not None else None
        edits.append(Edit(module=e.module, u=u.double().clone(), alpha=float(e.alpha), idx=None,
                          base_sha256=e.base_sha256, base_dtype=e.base_dtype, shape=tuple(e.shape), v=v))
    meta = {"name": name, "description": desc, "base_model": v2.meta.get("base_model", ""),
            "policy": v2.meta.get("policy", ""), "provenance": "scripts/experiment_overrefusal_nullspace_aion.py (from v2_blind)"}
    return MoIPatch(edits=edits, router=copy.deepcopy(v2.router), gates=list(v2.gates), meta=meta)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--work", required=True, help="judge work dir: agent_*/test_* splits and cached gen_base_*")
    ap.add_argument("--v2-patch", required=True)
    ap.add_argument("--v2-arm", default="v2_blind", help="arm name of v2's cached judge generations in --work")
    ap.add_argument("--out-dir", required=True, help="where the new .moipatch files go (keep them out of the repo)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="null90,null99,clean,clean_null99")
    ap.add_argument("--dir-layer", type=int, default=11, help="residual index of v2's direction (its summary: 11)")
    ap.add_argument("--ref-tokens", type=int, default=128)
    ap.add_argument("--skip-onset", type=int, default=8, help="answer tokens left out of the null-space reference")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cov-device", default=None, help="default: the model's device")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args(argv)

    work, out_dir = Path(a.work), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]
    res: dict = {"config": vars(a), "arms": {}}

    model, tok = ev.load(a.base_dir, a.dtype, a.device)
    cov_dev = a.cov_device or str(next(model.parameters()).device)
    v2 = MoIPatch.load(a.v2_patch)
    mods = module_map(model)
    names = [e.module for e in v2.edits]
    u = v2.edits[0].u.double()
    u = u / u.norm()
    assert all(torch.allclose(e.u.double() / e.u.double().norm(), u) for e in v2.edits), "v2 uses one direction"
    log(f"v2: {len(names)} writers, alpha {v2.edits[0].alpha}, router layer {v2.router['layer']}")

    # ---- 1. reference answers on the agent split
    agent_or = [r["prompt"] for r in ev.read_jsonl(work / "agent_overrefusal.jsonl")]
    agent_ctrl = [r["prompt"] for r in ev.read_jsonl(work / "agent_controls.jsonl")]
    base_or = ev.generate(model, tok, agent_or, a.ref_tokens, a.bs)
    base_ctrl = ev.generate(model, tok, agent_ctrl, a.ref_tokens, a.bs)
    ref_or = [ev.is_refusal(x) for x in base_or]
    i_ref = [i for i, r in enumerate(ref_or) if r]
    forced = MoIPatch(edits=v2.edits, router=None, meta={})
    with forced.enable_routed(model, mode="always_on"):
        v2_or = ev.generate(model, tok, [agent_or[i] for i in i_ref], a.ref_tokens, a.bs)
    recovered = [(agent_or[i], x) for i, x in zip(i_ref, v2_or) if not ev.is_refusal(x)]
    exam = [(p, x) for p, x in recovered if ev.opening(x).startswith(EXAM_PREFIX)]
    normal_rec = [(p, x) for p, x in recovered if (p, x) not in exam]
    answered = [(agent_ctrl[i], x) for i, x in enumerate(base_ctrl) if not ev.is_refusal(x)]
    answered += [(agent_or[i], x) for i, x in enumerate(base_or) if not ref_or[i]]
    res["reference"] = {"agent_overrefusal_refused_by_base": len(i_ref), "v2_forced_recovered": len(recovered),
                        "v2_forced_exam_template": len(exam), "base_answered": len(answered)}
    log(f"reference: {res['reference']}")

    # ---- 2. cleaned direction
    u_clean = None
    if any(x.startswith("clean") for x in arms):
        L = a.dir_layer
        atoms: dict[str, torch.Tensor | None] = {"exam_template": None, "math_vs_general": None}
        if len(exam) >= 3 and len(normal_rec) >= 3:
            atoms["exam_template"] = (residual_means(model, tok, L, [p for p, _ in exam], [x for _, x in exam], a.bs, "answer")
                                      - residual_means(model, tok, L, [p for p, _ in normal_rec], [x for _, x in normal_rec], a.bs, "answer"))
        gen = [p for p in agent_ctrl if p in ev.GENERAL]
        math = [p for p in agent_ctrl if p not in ev.GENERAL]
        if len(gen) >= 3 and len(math) >= 3:
            atoms["math_vs_general"] = (residual_means(model, tok, L, math, [""] * len(math), a.bs, "prompt")
                                        - residual_means(model, tok, L, gen, [""] * len(gen), a.bs, "prompt"))
        u_clean, rep = ridge_clean(u, atoms, a.ridge)
        res["clean"] = rep
        log(f"cleaned direction: {rep}")

    # ---- 3. null-space rows (base answer tokens + v2's normal recovered answers, under the base model)
    need_null = [x for x in arms if "null" in x]
    energies = sorted({float(x.split("null")[1]) / 100 for x in need_null})
    dirs = {"u": u}
    if any(x.startswith("clean_null") for x in arms):
        dirs["u_clean"] = u_clean
    nullv: dict = {}
    if need_null:
        ref_p = [p for p, _ in answered] + [p for p, _ in normal_rec]
        ref_a = [x for _, x in answered] + [x for _, x in normal_rec]
        covs, n_tok = collect_cov(model, tok, mods, names, ref_p, ref_a, a.bs, a.skip_onset, cov_dev)
        res["nullspace"] = {"reference_answer_tokens": n_tok, "skip_onset": a.skip_onset, "energies": energies}
        log(f"input covariances from {n_tok} answer tokens")
        for n in names:
            W = mods[n].weight.detach().double().cpu()
            nullv[n] = null_rows(covs.pop(n), {k: d.double() @ W for k, d in dirs.items()}, energies)
        del covs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def rows_for(key: str, e: float, arm: str) -> dict:
        res.setdefault("null_diag", {})[arm] = [{"module": n, "k": nullv[n][(key, e)]["k"], "dim": nullv[n][(key, e)]["dim"],
                                                 "kept_norm": round(nullv[n][(key, e)]["kept_norm"], 4)} for n in names]
        return {n: nullv[n][(key, e)]["v"] for n in names}

    # ---- 4. build the patches
    patches = {}
    for arm in arms:
        if arm.startswith("clean_null"):
            e = float(arm[len("clean_null"):]) / 100
            patches[arm] = build(v2, mods, u_clean, rows_for("u_clean", e, arm), f"overrefusal-{arm}",
                                 f"cleaned direction, edit restricted to the input complement holding {e:.0%} of answer-token energy")
        elif arm.startswith("null"):
            e = float(arm[4:]) / 100
            patches[arm] = build(v2, mods, u, rows_for("u", e, arm), f"overrefusal-{arm}",
                                 f"v2 edit restricted to the input complement holding {e:.0%} of answer-token energy")
        elif arm == "clean":
            patches[arm] = build(v2, mods, u_clean, None, "overrefusal-clean", "v2 projection with the direction cleaned of protected atoms")
        else:
            raise SystemExit(f"unknown arm {arm}")
        patches[arm].save(out_dir / f"{arm}.moipatch")
    nullv.clear()
    for arm, d in res.get("null_diag", {}).items():
        kept = [x["kept_norm"] for x in d]
        res["arms"].setdefault(arm, {})["edit_norm_kept_mean"] = round(sum(kept) / len(kept), 4)
    log("patches saved")

    # ---- 5. judge (same test split, same batching as every arm before)
    if not a.no_judge:
        base = ev.run_arm(model, tok, work, "base", a.max_new, a.bs)
        cats = [r.get("category", "unknown") for r in ev.read_jsonl(work / "test_toxic.jsonl")]
        try:
            v2_arm = {part: [r["answer"] for r in ev.read_jsonl(work / f"gen_{a.v2_arm}_{part}.jsonl")]
                      for part in ("overrefusal", "toxic", "controls")}
            res["arms"]["v2"] = with_ratio(ev.score(base, v2_arm, cats), len(cats))
        except FileNotFoundError:
            log("v2's cached judge generations not found; v2 not re-scored")
        for arm in arms:
            name = f"ns_{arm}"
            with patches[arm].enable_routed(model) as _h:
                gens = ev.run_arm(model, tok, work, name, a.max_new, a.bs)
            sc = with_ratio(ev.score(base, gens, cats), len(cats))
            res["arms"].setdefault(arm, {}).update(sc)
            ev.atomic_json(work / f"eval_{name}.json", {"arm": name, **sc})
            st = sc["style"]
            log(f"{arm}: recovered {sc['recovered']} toxic_kept {sc['toxic_kept']} controls {sc['controls_identical']} "
                f"new_template {st.get('new_template_rate')} success {sc['success']} with_style {sc['success_with_style']}")
    res["minutes"] = round((time.time() - T0) / 60, 1)
    ev.atomic_json(Path(a.out), res)
    return res


if __name__ == "__main__":
    main()
