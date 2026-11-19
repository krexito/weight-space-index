"""Quick loop-fix search: screen out half the grid, then test the rest only where the loops happen.

discover_fast's search generates every GSM8K answer from the start, up to 16k tokens, for each of 12 configs. Most of
that time goes into the part of the answer before the loop, which is the same with or without the patch (on
Qwen3.5-4B loops start after ~4k tokens). This replaces the search with two cheap steps:

  1. screen  (Idea A): screen_fast_grid.py ranks the 12 discover_fast configs with one teacher-forced forward pass
     each (pareto score, healthy dNLL guard 0.05). The bottom half is dropped: KEEP = 6 configs go on.
  2. probe   (Idea B, "load the save before the loop"): the base model's own looping elicit responses are cut
     CUT_BACK tokens before the loop onset, and every kept config (and the base model) continues from there for
     LOOP_NEW tokens, K_SAMPLES samples each. Measured: did it loop again, did it finish, was the answer right.
     A guard set of problems the base model solved cleanly in <= HEALTHY_RESP tokens is re-solved from the start
     with each config, to catch patches that break normal reasoning.
  3. pick: score = p_loop * rescued_correct + (1 - p_loop) * healthy_accuracy, an estimate of GSM8K accuracy, where
     p_loop is the share of elicit responses in which the base model loops. Highest score wins; ties go to the
     lower re-loop rate, then the better screen rank.

Generation reuses `discover_fast.py gen-jobs` (one vLLM, edited weights swapped in memory with exact checksums), so
the edits are bit-for-bit the ones the long search tests. This file does not change any file discover_fast hashes.

When the work dir also holds a long search's search.json, `pick` grades the quick choice against it. Pre-registered
prediction: the quick choice's round-1 accuracy is >= the 3rd-best round-1 accuracy among the 12 configs (ties count).

    python quick_discover.py all --work W        # screen, prepare, probe, pick
    python quick_discover.py all --work W2 --smoke   # 2 configs, 4 prefixes, short outputs: a crash test only
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
import moi_loop_router_vllm as lr  # noqa: E402

KEEP = 6
CUT_BACK = 128
LOOP_NEW = 1024
K_SAMPLES = 2
N_LOOP = 48
TAIL = 256
HEALTHY_RESP = 2048
HEALTHY_NEW = 3072
N_HEALTHY = 48
SEED_SHIFT = 7_000_001
SCREEN_OUT = "results_screen_fast.json"


def _time(work: Path, key: str, s: float) -> None:
    p = work / "timing_quick.json"
    t = json.loads(p.read_text()) if p.exists() else {}
    t[key] = round(s, 1)
    p.write_text(json.dumps(t, indent=1))
    print(f"[time] {key}: {s / 60:.1f} min", flush=True)


def _py(args: list[str]) -> None:
    r = subprocess.run([sys.executable, "-u", *args])
    if r.returncode:
        raise SystemExit(f"failed: {' '.join(args[:2])}")


# ------------------------------------------------------------------ 1. screen
def cmd_screen(a) -> None:
    work = Path(a.work)
    if (work / SCREEN_OUT).exists():
        print("screen already done")
        return
    t0 = time.perf_counter()
    _py([str(HERE / "screen_fast_grid.py"), "--work", str(work), "--out", SCREEN_OUT] + (["--cpu"] if a.cpu else []))
    _time(work, "screen", time.perf_counter() - t0)


def kept_configs(work: Path) -> list[dict]:
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    by_name = {c["name"]: c for c in fast.grid(diag["layer_rank"])}
    ranking = json.loads((work / SCREEN_OUT).read_text())["ranking"]
    return [{**by_name[r["name"]], "screen_rank": i + 1} for i, r in enumerate(ranking[:KEEP])]


# ------------------------------------------------------------------ 2. probe requests
def build_probe(work: Path, tok) -> tuple[list[dict], list[dict], dict]:
    cfg = dl.cfg_load(work)
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_elicit.jsonl")}
    outs = lr.read_jsonl(work / "outputs_elicit.jsonl")
    info = []
    for o in outs:
        body = lr.strip_eos(o["token_ids"], eos)
        lab = dl.token_labels(body)
        info.append((o, body, lab, bool((lab >= 0).any())))
    # half-B flag: the same list and permutation as discover_fast.cmd_diagnose (labelled responses only)
    perm = iter(np.random.default_rng(cfg["seed"]).permutation(sum(x[3] for x in info)) % 2)
    loop_reqs, healthy_reqs, n_loop = [], [], 0
    for o, body, lab, labelled in info:
        hb = int(next(perm)) if labelled else -1
        r = reqs[o["id"]]
        if (lab == 1).any():
            n_loop += 1
            if len(loop_reqs) >= N_LOOP * K_SAMPLES:
                continue
            cut = max(16, int(np.argmax(lab == 1)) - CUT_BACK)
            for k in range(K_SAMPLES):
                loop_reqs.append({"id": o["id"] * 10 + k, "src": o["id"], "k": k, "half": hb,
                                  "prompt_ids": r["prompt_ids"] + body[:cut], "p_len": len(r["prompt_ids"]),
                                  "gold": r["gold"], "seed": r["seed"] + SEED_SHIFT * (k + 1),
                                  "max_tokens": LOOP_NEW})
        elif len(healthy_reqs) < N_HEALTHY and len(body) <= HEALTHY_RESP and o.get("finish_reason") == "stop":
            g = lr.grade(o["token_ids"], o["finish_reason"], r["gold"], tok, eos, r.get("max_tokens", 10**9))
            if g["correct"]:  # problems the base model solves cleanly: the guard set
                healthy_reqs.append({"id": o["id"], "src": o["id"], "half": hb, "prompt_ids": r["prompt_ids"],
                                     "gold": r["gold"], "seed": r["seed"] + SEED_SHIFT, "max_tokens": HEALTHY_NEW})
    meta = {"n_elicit": len(outs), "n_elicit_looping": n_loop,
            "p_loop": round(n_loop / max(len(outs), 1), 4), "n_loop_prefixes": len(loop_reqs) // K_SAMPLES,
            "n_healthy": len(healthy_reqs)}
    return loop_reqs, healthy_reqs, meta


def cmd_prepare(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    loop_reqs, healthy_reqs, meta = build_probe(work, AutoTokenizer.from_pretrained(dl.cfg_load(work)["base_dir"]))
    if not loop_reqs:
        raise SystemExit("no looping elicit responses to resume from")
    lr.write_jsonl(work / "requests_probe_loop.jsonl", loop_reqs)
    lr.write_jsonl(work / "requests_probe_healthy.jsonl", healthy_reqs)
    (work / "probe_meta.json").write_text(json.dumps(meta, indent=1))
    print("probe:", meta)


def probe_jobs(work: Path) -> list[dict]:
    arms = [{"name": "base"}] + [{**c, "direction": "wsi"} for c in kept_configs(work)]
    jobs = []
    for arm in arms:
        for kind in ("loop", "healthy"):
            jobs.append({**arm, "requests": f"requests_probe_{kind}.jsonl", "out": f"outputs_probe_{kind}_{arm['name']}.jsonl"})
    return jobs


def cmd_probe(a) -> None:
    work = Path(a.work)
    jobs = probe_jobs(work)
    jp = work / "jobs_probe.json"
    jp.write_text(json.dumps(jobs, indent=1))
    t0 = time.perf_counter()
    if any(not (work / j["out"]).exists() for j in jobs):
        _py([str(HERE / "discover_fast.py"), "gen-jobs", "--work", str(work), "--jobs", str(jp)]
            + (["--gpu-mem", str(a.gpu_mem)] if a.gpu_mem else []))
    _time(work, "probe", time.perf_counter() - t0)


# ------------------------------------------------------------------ 3. pick
def grade_loop(rows: list[dict], reqs: dict, tok, eos: set[int]) -> dict:
    out = []
    for o in rows:
        r = reqs[o["id"]]
        cont = lr.strip_eos(o["token_ids"], eos)
        prefix_body = r["prompt_ids"][r["p_len"]:]
        tail = prefix_body[-TAIL:]
        relooped = bool((dl.token_labels(tail + cont)[len(tail):] == 1).any())
        finished = o["finish_reason"] == "stop" and not relooped
        pred = lr.extract_answer(tok.decode(prefix_body + cont, skip_special_tokens=True)) if finished else None
        out.append({"id": o["id"], "half": r["half"], "relooped": relooped, "finished": finished,
                    "correct": bool(finished and lr.correct(pred, r["gold"]))})
    n = max(len(out), 1)
    hb = [x for x in out if x["half"] == 1]
    return {"n": len(out), "reloop": sum(x["relooped"] for x in out) / n,
            "finished": sum(x["finished"] for x in out) / n, "correct": sum(x["correct"] for x in out) / n,
            "correct_half_b": sum(x["correct"] for x in hb) / max(len(hb), 1), "rows": out}


def grade_healthy(rows: list[dict], reqs: dict, tok, eos: set[int]) -> dict:
    g = [lr.grade(o["token_ids"], o["finish_reason"], reqs[o["id"]]["gold"], tok, eos, reqs[o["id"]]["max_tokens"])
         for o in rows]
    n = max(len(g), 1)
    return {"n": len(g), "accuracy": sum(x["correct"] for x in g) / n,
            "loop_killed": sum(x["looped_killed"] for x in g) / n, "truncated": sum(x["truncated"] for x in g) / n}


def pick(work: Path, tok) -> dict:
    cfg = dl.cfg_load(work)
    eos = set(cfg["eos"])
    meta = json.loads((work / "probe_meta.json").read_text())
    p = meta["p_loop"]
    lreq = {r["id"]: r for r in lr.read_jsonl(work / "requests_probe_loop.jsonl")}
    hreq = {r["id"]: r for r in lr.read_jsonl(work / "requests_probe_healthy.jsonl")}
    arms = {}
    for name, rank in [("base", None)] + [(c["name"], c["screen_rank"]) for c in kept_configs(work)]:
        lg = grade_loop(lr.read_jsonl(work / f"outputs_probe_loop_{name}.jsonl"), lreq, tok, eos)
        hp = work / f"outputs_probe_healthy_{name}.jsonl"
        hg = grade_healthy(lr.read_jsonl(hp), hreq, tok, eos) if hreq else {"n": 0, "accuracy": 1.0}
        score = p * lg["correct"] + (1 - p) * hg["accuracy"]
        arms[name] = {"screen_rank": rank, "score": round(score, 4), "loop": {k: v for k, v in lg.items() if k != "rows"},
                      "healthy": hg}
        print(f"[pick] {name:18s} score={score:.3f} rescued={lg['correct']:.1%} reloop={lg['reloop']:.1%} "
              f"healthy={hg['accuracy']:.1%}", flush=True)
    cands = [k for k in arms if k != "base"]
    choice = max(cands, key=lambda k: (arms[k]["score"], -arms[k]["loop"]["reloop"], -arms[k]["screen_rank"]))
    res = {"model": cfg.get("model_repo"), "probe": meta,
           "rule": {"KEEP": KEEP, "CUT_BACK": CUT_BACK, "LOOP_NEW": LOOP_NEW, "K_SAMPLES": K_SAMPLES, "N_LOOP": N_LOOP,
                    "HEALTHY_RESP": HEALTHY_RESP, "HEALTHY_NEW": HEALTHY_NEW, "N_HEALTHY": N_HEALTHY},
           "arms": arms, "choice": choice, "choice_beats_base": arms[choice]["score"] > arms["base"]["score"]}
    sp = work / "search.json"
    if sp.exists():
        res["vs_search"] = vs_search(work, json.loads(sp.read_text()), arms, choice)
    return res


def vs_search(work: Path, search: dict, arms: dict, choice: str) -> dict:
    diag = json.loads((work / "diagnosis_blind.json").read_text())
    r1 = search["round1"]
    names = [c["name"] for c in fast.grid(diag["layer_rank"]) if c["name"] in r1]
    accs = sorted((r1[n]["accuracy"] for n in names), reverse=True)
    third = accs[min(2, len(accs) - 1)]
    res = {"r1_accuracy_of_choice": r1[choice]["accuracy"], "r1_third_best": third,
           "prediction_hit": r1[choice]["accuracy"] >= third,
           "chance": round(sum(r1[n]["accuracy"] >= third for n in names) / len(names), 3),
           "r1_best": max(names, key=lambda k: blind.score_key(r1[k]))}
    if "choice" in search:
        res["search_choice"] = search["choice"]["name"]
        res["same_as_search_choice"] = search["choice"]["name"] == choice
        res["search_choice_survived_screen"] = search["choice"]["name"] in arms
    probed = [k for k in arms if k != "base" and k in r1]
    if len(probed) >= 3:
        from screen_fast_grid import _spearman

        res["spearman_score_vs_r1_acc"] = _spearman([arms[k]["score"] for k in probed],
                                                    [r1[k]["accuracy"] for k in probed])
    if "time_s" in search:
        res["search_min"] = round(sum(search["time_s"].values()) / 60, 1)
    return res


def cmd_pick(a) -> None:
    from transformers import AutoTokenizer

    work = Path(a.work)
    t0 = time.perf_counter()
    res = pick(work, AutoTokenizer.from_pretrained(dl.cfg_load(work)["base_dir"]))
    _time(work, "pick", time.perf_counter() - t0)
    res["timing_s"] = json.loads((work / "timing_quick.json").read_text())  # screen + probe + pick
    res["quick_min"] = round(sum(res["timing_s"].values()) / 60, 1)
    (work / a.out).write_text(json.dumps(res, indent=1))
    print("choice:", res["choice"], "| beats base:", res["choice_beats_base"], "| quick search:", res["quick_min"], "min")
    if "vs_search" in res:
        print(json.dumps(res["vs_search"], indent=1))
    fast.sync(work)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["screen", "prepare", "probe", "pick", "all"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="results_quick_discover.json")
    ap.add_argument("--gpu-mem", type=float, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run that only checks the pipeline works (not a result)")
    a = ap.parse_args()
    if a.smoke:
        global KEEP, N_LOOP, N_HEALTHY, LOOP_NEW, HEALTHY_NEW
        KEEP, N_LOOP, N_HEALTHY, LOOP_NEW, HEALTHY_NEW = 2, 4, 4, 256, 512
    steps = {"screen": [cmd_screen], "prepare": [cmd_prepare], "probe": [cmd_probe], "pick": [cmd_pick],
             "all": [cmd_screen, cmd_prepare, cmd_probe, cmd_pick]}[a.cmd]
    for f in steps:
        f(a)


if __name__ == "__main__":
    main()
