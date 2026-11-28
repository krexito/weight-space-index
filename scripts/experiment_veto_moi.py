"""Self-gate veto in generation: does switching the patch on later (only when its own direction confirms) keep the
MoI gain, and how many healthy answers does it leave alone?

On saved answers (experiment_self_gate_router.py) the text detector proposes and the patch's own direction confirms:
false switch-ons on non-looping answers dropped 7/11 -> 1/11 (GSM8K), but the switch-on came ~500 tokens later. That
was never generated. Here, on a finished discover_blind run (GSM8K, the blind patch, base answers + text-timed MoI):

  plan    (transformers, GPU) self-gate score = mean cos(down_proj output, u) over the patch's edited layers,
          32-token moving average. Orientation and threshold (--fpr of normal tokens) are fit on the ELICIT answers;
          on each TEST answer where the text detector fired, the switch-on is the first token >= the text fire whose
          score passes. Writes requests_test_moi_veto.jsonl (same format as the text-timed MoI plan: prefix up to the
          switch-on, prefix tokens penalised, new seed) and veto_plan.json.
  gen     (vLLM) the veto plan resumed with the patched checkpoint (veto_moi) and with the unpatched base
          (veto_restart, the restart control at the veto timing).
  report  accuracy (strict, and with answer rescue) of base / text MoI / text restart / veto MoI / veto restart,
          paired CIs, and how many base answers that were already right get switched on.

  python experiment_veto_moi.py plan   --work W --model /content/base
  python experiment_veto_moi.py gen    --work W --base-dir /content/base
  python experiment_veto_moi.py report --work W --tokenizer /content/base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import moi_loop_router_vllm as lr  # noqa: E402

PLAN = "requests_test_moi_veto.jsonl"
TEXT_PLAN = "requests_test_moi.jsonl"
ARMS = {"text_moi": ("outputs_test_moi_blind.jsonl", TEXT_PLAN),
        "text_restart": ("outputs_test_moi_restart.jsonl", TEXT_PLAN),
        "veto_moi": ("outputs_test_moi_veto.jsonl", PLAN),
        "veto_restart": ("outputs_test_moi_veto_restart.jsonl", PLAN)}
PRE_GAP = 256
SMOOTH = 32


# ------------------------------------------------------------------ data + scoring (pure, CPU-testable)
def load_answers(work: Path, req_file: str, out_file: str, eos: set[int], max_tokens: int | None = None) -> list[dict]:
    reqs = {r["id"]: r for r in lr.read_jsonl(work / req_file)}
    out = []
    for o in lr.read_jsonl(work / out_file):
        st = lr.analyse(o["token_ids"], eos)
        looping = st["kill"] is not None
        body = st["body"][: st["kill"]] if looping else st["body"]
        if max_tokens:
            body = body[:max_tokens]
        if len(body) < 2 * SMOOTH:
            continue
        onset = max(0, st["kill"] - lr.KILL_WINDOW) if looping else None
        label = np.zeros(len(body), np.int8)  # 0 normal, 1 loop region, -1 ignored just before onset
        if looping:
            label[onset:] = 1
            label[max(0, onset - PRE_GAP):onset] = -1
        out.append({"id": o["id"], "prompt": reqs[o["id"]]["prompt_ids"], "body": body, "label": label,
                    "looping": looping, "onset": onset, "fire": st["fire"]})
    return out


def smooth(x: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    c = np.cumsum(np.insert(np.asarray(x, np.float64), 0, 0.0))
    k = np.arange(len(x))
    lo = np.maximum(0, k - w + 1)
    return (c[k + 1] - c[lo]) / (k + 1 - lo)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    r = x.argsort().argsort().astype(np.float64) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibrate(scores: list[np.ndarray], seqs: list[dict], fpr: float) -> dict:
    sm = [smooth(s) for s in scores]
    pos = np.concatenate([s[q["label"] == 1] for s, q in zip(sm, seqs)] or [np.zeros(0)])
    neg = np.concatenate([s[q["label"] == 0] for s, q in zip(sm, seqs)] or [np.zeros(0)])
    a = auc(pos, neg)
    sign = 1.0 if a >= 0.5 else -1.0
    thr = float(np.quantile(sign * neg, 1 - fpr))
    return {"sign": sign, "thr": thr, "auc_calib": round(max(a, 1 - a), 4), "fpr": fpr,
            "n_loop_tokens": int(len(pos)), "n_normal_tokens": int(len(neg))}


def switch_on(score: np.ndarray, fire: int | None, cal: dict) -> int | None:
    """First token >= the text fire whose smoothed, oriented score passes the threshold; None = vetoed."""
    if fire is None:
        return None
    s = cal["sign"] * smooth(score)
    hit = np.flatnonzero(s[fire:] > cal["thr"])
    return int(fire + hit[0]) if len(hit) else None


def plan_rows(seqs: list[dict], ons: dict[int, int | None], reqs: dict, cfg: dict, penalty: float) -> list[dict]:
    """Same format as moi_loop_router_vllm.cmd_plan_moi: resume from the switch-on with the prefix penalised."""
    rows = []
    for q in seqs:
        on = ons.get(q["id"])
        if on is None:
            continue
        prefix = q["body"][:on]  # body ends at the kill: a switch-on after it never happens (generation stops)
        r = reqs[q["id"]]
        rows.append({"id": q["id"], "prompt_ids": r["prompt_ids"] + prefix, "prefix_len": len(prefix),
                     "max_tokens": r["max_tokens"] - len(prefix), "seed": r["seed"] + 7,
                     "logit_bias": {str(t): -penalty for t in sorted(set(prefix))}})
    return rows


# ------------------------------------------------------------------ plan (GPU)
def cmd_plan(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    eos = set(cfg["eos"])
    choice = json.loads((work / "frozen.json").read_text())["choice"]
    z = np.load(work / "blind_directions.npz")
    calib = load_answers(work, "requests_elicit.jsonl", "outputs_elicit.jsonl", eos, a.max_tokens)
    rng = np.random.default_rng(a.seed)
    if len(calib) > a.n_calib:
        calib = [calib[i] for i in sorted(rng.choice(len(calib), a.n_calib, replace=False))]
    test = load_answers(work, "requests_test.jsonl", "outputs_test_base.jsonl", eos)

    dev = torch.device(a.device)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), trust_remote_code=True).to(dev).eval()
    layers = next(m for n, m in model.named_modules()
                  if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) >= 2)
    mods = {f"L{li}": lr.down_module(layers[li]) for li in choice["layers"]}
    dirs = {f"L{li}": torch.from_numpy(z[f"u{li}"]).float().to(dev) for li in choice["layers"]}
    backbone = getattr(model, "model", model)  # no LM head: a 16k-token logit tensor would not fit

    def score(seqs):
        """Mean cos(down_proj output, u) over the edited layers, per answer token."""
        out = []
        for q in seqs:
            ys: dict = {}
            hooks = [m.register_forward_hook(lambda _m, _i, o, k=k: ys.__setitem__(k, o[0] if isinstance(o, tuple) else o))
                     for k, m in mods.items()]
            try:
                with torch.no_grad():
                    backbone(input_ids=torch.tensor([q["prompt"] + q["body"]], device=dev), use_cache=False)
            finally:
                for h in hooks:
                    h.remove()
            p0 = len(q["prompt"])
            cos = [(ys[k][0, p0:].float() @ u) / ys[k][0, p0:].float().norm(dim=-1).clamp_min(1e-6) for k, u in dirs.items()]
            out.append(torch.stack(cos).mean(0).cpu().numpy())
        return out

    sc_cal = score(calib)
    cal = calibrate(sc_cal, calib, a.fpr)
    sc_test = score(test)
    ons = {q["id"]: switch_on(s, q["fire"], cal) for q, s in zip(test, sc_test)}
    # held-out separation on test (same labels as calibration)
    sm = [cal["sign"] * smooth(s) for s in sc_test]
    pos = np.concatenate([s[q["label"] == 1] for s, q in zip(sm, test)] or [np.zeros(0)])
    neg = np.concatenate([s[q["label"] == 0] for s, q in zip(sm, test)] or [np.zeros(0)])

    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    rows = plan_rows(test, ons, reqs, cfg, lr.SAMPLING["presence_penalty"])
    lr.write_jsonl(work / PLAN, rows)
    loops = [q for q in test if q["looping"]]
    normal = [q for q in test if not q["looping"]]
    text_fired = [q for q in test if q["fire"] is not None]
    st = {"calibration": cal, "auc_test": round(auc(pos, neg), 4), "n_test": len(test),
          "text_fired": len(text_fired), "veto_on": len(rows),
          "loops_on": f"{sum(ons[q['id']] is not None for q in loops)}/{len(loops)}",
          "normal_on": f"{sum(ons[q['id']] is not None for q in normal)}/{len(normal)}",
          "normal_text_fired": f"{sum(q['fire'] is not None for q in normal)}/{len(normal)}",
          "median_delay_after_text": float(np.median([ons[q['id']] - q["fire"] for q in text_fired
                                                      if ons[q["id"]] is not None] or [np.nan])),
          "median_on_after_onset": float(np.median([ons[q["id"]] - q["onset"] for q in loops
                                                    if ons[q["id"]] is not None] or [np.nan])),
          "switch_on": {str(k): v for k, v in ons.items()}}
    (work / "veto_plan.json").write_text(json.dumps(st, indent=1))
    print(json.dumps({k: v for k, v in st.items() if k != "switch_on"}, indent=1))


# ------------------------------------------------------------------ gen (vLLM)
def cmd_gen(a) -> None:
    import discover_blind as blind
    import discover_loop_fix as dl
    import discover_math500_blind as m500

    work = Path(a.work)
    cfg = dl.cfg_load(work)
    if a.base_dir:
        cfg["base_dir"] = str(a.base_dir)
        dl.cfg_save(work, cfg)
    c = json.loads((work / "frozen.json").read_text())["choice"]
    ck = Path(a.ck_dir) / "blind_patched"
    if not (ck / "config.json").exists():
        blind.build_checkpoint(work, c["layers"], c["neurons"], c["alpha"], "wsi", ck)
    for model, name in ((ck, "veto_moi"), (None, "veto_restart")):
        out = ARMS[name][0]
        if (work / out).exists():
            print(f"{out} already there")
            continue
        m500._run_parallel_fast(work, [{"model": model, "jobs": [{"name": name, "requests": PLAN, "out": out}]}], a)
        print(f"{name}: {len(lr.read_jsonl(work / out))} resumed answers", flush=True)


# ------------------------------------------------------------------ report
def cmd_report(a) -> None:
    from transformers import AutoTokenizer

    import rescue_answer as ra

    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    eos = set(cfg["eos"])
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    base_out = {o["id"]: (o["token_ids"], o["finish_reason"]) for o in lr.read_jsonl(work / "outputs_test_base.jsonl")}
    grader = ra.Grader("gsm8k")
    arms = {"base": base_out}
    for name, (out, plan_f) in ARMS.items():
        if not (work / out).exists() or not (work / plan_f).exists():
            print(f"  ({name}: {out} missing, skipped)")
            continue
        plan = {r["id"]: r for r in lr.read_jsonl(work / plan_f)}
        cont = {o["id"]: o for o in lr.read_jsonl(work / out)}
        arms[name] = {i: ((plan[i]["prompt_ids"][len(reqs[i]["prompt_ids"]):] + cont[i]["token_ids"],
                           cont[i]["finish_reason"]) if i in cont else base_out[i]) for i in reqs}
    scored = {k: ra.score_arm(v, reqs, tok, eos, grader) for k, v in arms.items()}
    ids = sorted(reqs)
    ok = {k: {r["id"]: bool(r["correct"]) for r in s["rows"]} for k, s in scored.items()}
    ok_r = {k: {r["id"]: bool(r["correct"] or r.get("strict_ok")) for r in s["rows"]} for k, s in scored.items()}
    healthy = [i for i in ids if ok["base"][i]]  # base already right: any switch-on here is a false alarm
    switched = {k: {r["id"] for r in lr.read_jsonl(work / p)} for k, (_, p) in ARMS.items() if (work / p).exists()}
    res = {"n": len(ids), "arms": {}, "paired": {}, "healthy_base_right": len(healthy)}
    for k in scored:
        sw = switched.get(k, set())
        res["arms"][k] = {"acc": round(np.mean([ok[k][i] for i in ids]), 4),
                          "acc_rescued": round(np.mean([ok_r[k][i] for i in ids]), 4),
                          "loop_killed": scored[k]["summary"]["killed"],
                          "switched_on": len(sw) if k != "base" else 0,
                          "healthy_switched": sum(i in sw for i in healthy),
                          "healthy_kept_right": sum(ok[k][i] for i in healthy)}
    for x, y in (("veto_moi", "text_moi"), ("veto_moi", "veto_restart"), ("text_moi", "text_restart"),
                 ("veto_restart", "text_restart"), ("veto_moi", "base")):
        if x in ok and y in ok:
            res["paired"][f"{x}_vs_{y}"] = lr.paired_ci([ok[x][i] for i in ids], [ok[y][i] for i in ids])
            res["paired"][f"{x}_vs_{y}_rescued"] = lr.paired_ci([ok_r[x][i] for i in ids], [ok_r[y][i] for i in ids])
    if (work / "veto_plan.json").exists():
        vp = json.loads((work / "veto_plan.json").read_text())
        res["veto_plan"] = {k: v for k, v in vp.items() if k != "switch_on"}
    (work / "results_veto_moi.json").write_text(json.dumps(res, indent=1))
    print(f"{'arm':13s} {'acc':>6s} {'+rescue':>8s} {'killed':>6s} {'on':>4s} {'healthy on':>10s} {'healthy ok':>10s}")
    for k, v in res["arms"].items():
        print(f"{k:13s} {v['acc']:6.1%} {v['acc_rescued']:8.1%} {v['loop_killed']:6d} {v['switched_on']:4d} "
              f"{v['healthy_switched']:4d}/{len(healthy):<5d} {v['healthy_kept_right']:4d}/{len(healthy):<5d}")
    for k, v in res["paired"].items():
        print(f"  {k}: {v['diff']:+.1%}  95% CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]  ({v['a_only']}/{v['b_only']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--work", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fpr", type=float, default=0.01, help="token false-alarm rate on normal elicit tokens")
    p.add_argument("--n-calib", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--base-dir", default=None)
    g.add_argument("--ck-dir", default="/tmp")
    g.add_argument("--gpu-mem", type=float, default=0.92)
    g.add_argument("--dtype", default="auto")
    g.add_argument("--cpu", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    {"plan": cmd_plan, "gen": cmd_gen, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
