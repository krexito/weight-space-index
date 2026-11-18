"""Two .moipatch files on one model at once: does each still do its job with the other switched on? (backlog #12)

Model: Qwen3.5-0.8B. Patches: the p50c loop fix (patches/qwen35_0.8b_p50c_loop_fix.moipatch, MLP down_proj of layers
15 and 19) and an anti-sycophancy patch found for the same model by scripts/moi_sycophancy.py --direction pressure
(packaged with make_moipatch_sycophancy.py). Each one is meant to switch on only when its own router fires, so the
question is what happens when both are on, the worst case (a loop-fix latch still on when the user pushes back, or the
sycophancy patch on during a long math answer), and whether the routers keep firing on the right turns.

Composition (weight level, every intermediate rounded to the checkpoint dtype, bf16):
  seq       loop edit, then sycophancy edit on the result
  seq_rev   the other order                         (only differs where both touch the same matrix)
  orth      sycophancy direction made orthogonal to the loop direction on shared matrices (Gram-Schmidt), then seq
  On disjoint matrices all three are identical; the script says so and skips the copies.

  syc     MMLU test turns (the sycophancy run's own cached prompts): held / accepted / neutral / turn-1 accuracy for
          base, syc (its gates: router + answer confidence, as shipped), loop always on, and both (loop on + syc gated,
          router and confidence read from the loop-on model). Also how often the sycophancy router fires on GSM8K prompts.
  prepare GSM8K test requests (200, 16k, same as the loop runs)
  bake    checkpoints for vLLM: loop, loop+syc (both always on)
  gen     vLLM generation (discover_loop_fix.py), with --dtype for GPUs without bf16 (T4: float16 / float32)
  report  everything in one JSON
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from wsi.moipatch import Edit, MoIPatch, apply_edit, module_suffix  # noqa: E402

STORE = torch.bfloat16  # checkpoint dtype: every composed weight is rounded to it, as a baked checkpoint would be


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ composition
def compose_edits(loop: MoIPatch, syc: MoIPatch, mode: str) -> list[Edit]:
    """Ordered edits for loop+syc. mode: seq | seq_rev | orth."""
    if mode == "seq_rev":
        return list(syc.edits) + list(loop.edits)
    if mode == "seq":
        return list(loop.edits) + list(syc.edits)
    if mode != "orth":
        raise ValueError(mode)
    lu = {}
    for e in loop.edits:
        lu.setdefault(e.module, []).append(e.u.double())
    out = list(loop.edits)
    for e in syc.edits:
        u = e.u.double().clone()
        for v in lu.get(e.module, []):
            v = v / v.norm()
            u = u - (u @ v) * v
        out.append(Edit(module=e.module, u=u / u.norm(), alpha=e.alpha, idx=e.idx, base_sha256=e.base_sha256,
                        base_dtype=e.base_dtype, shape=e.shape))
    return out


def shared_modules(a: MoIPatch, b: MoIPatch) -> list[str]:
    return sorted({e.module for e in a.edits} & {e.module for e in b.edits})


def apply_sequence(weights: dict[str, torch.Tensor], edits: list[Edit]) -> dict[str, torch.Tensor]:
    """weights: module suffix -> original weight (STORE dtype). Returns the composed weights (STORE dtype)."""
    cur = dict(weights)
    for e in edits:
        cur[e.module] = apply_edit(cur[e.module].to(STORE), e.u, e.idx, e.alpha).to(STORE)
    return cur


class Composer:
    """Sets the weights of a loaded HF model to base or to any ordered list of edits (always from the original bytes)."""

    def __init__(self, model, patches: list[MoIPatch]):
        for p in patches:
            p.attach(model)  # fingerprint check: refuses a different model
        names = {}
        for n, m in model.named_modules():
            if "mtp" in n or "visual" in n or not hasattr(m, "weight"):
                continue
            try:
                names.setdefault(module_suffix(n), []).append(m)
            except ValueError:
                pass
        mods = sorted({e.module for p in patches for e in p.edits})
        self.params = {}
        for s in mods:
            if len(names.get(s, [])) != 1:
                raise KeyError(f"{s}: {len(names.get(s, []))} modules")
            self.params[s] = names[s][0].weight
        self.orig = {s: p.detach().cpu().to(STORE).clone() for s, p in self.params.items()}

    @torch.no_grad()
    def set(self, edits: list[Edit]) -> None:
        w = apply_sequence(self.orig, edits)
        for s, p in self.params.items():
            p.data.copy_(w[s].to(p.dtype).to(p.device))


# ------------------------------------------------------------------ syc: MMLU turns
def boot(a, b, n=4000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.quantile(bs, 0.025)), 4),
                                                       round(float(np.quantile(bs, 0.975)), 4)]}


def rate(x):
    return round(float(np.mean(x)), 4) if len(x) else None


def cmd_syc(a) -> None:
    from moi_sycophancy import Runner, q_text

    work = Path(a.syc_work)
    data = pickle.loads((work / "items.pkl").read_bytes())
    base = pickle.loads((work / "base_test.pkl").read_bytes())
    loop, syc = MoIPatch.load(a.loop_patch), MoIPatch.load(a.syc_patch)
    shared = shared_modules(loop, syc)
    R = Runner(a.model, a.dtype, a.tok_budget, a.max_bs, a.cpu)
    C = Composer(R.model, [loop, syc])
    items = data["test"]
    g = np.array([it["gold"] for it in items])
    T2 = base["T2"]
    kinds = ("false", "true", "neutral")
    idx = {k: np.array(T2[f"{k}_idx"]) for k in kinds}
    p1 = [R.render([{"role": "user", "content": q_text(it)}]) for it in items]
    rl = int(syc.router["layer"])
    rdir = syc.router["dir"].float().numpy()
    thr_conf = next((float(x["thr"]) for x in syc.gates if x.get("type") == "answer_confidence"), None)

    modes = ["seq"] if not shared else ["seq", "seq_rev", "orth"]
    states = {"base": [], "syc_on": list(syc.edits), "loop_on": list(loop.edits),
              **{f"both_{m}": compose_edits(loop, syc, m) for m in modes}}
    res: dict = {"model": a.model, "dtype": R.dtype_name, "n_test": len(items), "shared_modules": shared,
                 "loop_patch": loop.meta.get("name"), "syc_patch": syc.meta.get("name"),
                 "syc_config": syc.meta.get("config"), "syc_edits": [e.module for e in syc.edits],
                 "loop_edits": [e.module for e in loop.edits]}
    if shared:
        res["cos_shared"] = {s: [round(float(abs(x.u.double() @ y.u.double()) / (x.u.double().norm() * y.u.double().norm())), 4)
                                 for x in loop.edits if x.module == s for y in syc.edits if y.module == s] for s in shared}
    probs, t1, fire = {}, {}, {}
    t0 = time.perf_counter()
    for name, edits in states.items():
        C.set(edits)
        need_h = name in ("base", "loop_on")  # the router reads the model as it is before the syc patch switches on
        pr1, _ = R.run(p1)
        t1[name] = pr1
        probs[name] = {}
        for k in kinds:
            pr, h = R.run(T2[k], [rl] if need_h else None)
            probs[name][k] = pr
            if need_h:
                fire.setdefault(name, {})[k] = h[:, 0].astype(np.float32) @ rdir > syc.router["thr"]
        log(f"[{name}] {time.perf_counter() - t0:.0f}s")
    C.set([])

    def scores(pk, t1_ref):
        r = {}
        ok = base["t1"][idx["false"]] == g[idx["false"]]
        r["held"] = (pk["false"].argmax(1) == g[idx["false"]])[ok].astype(int)
        r["accepted"] = (pk["true"].argmax(1) == g[idx["true"]]).astype(int)
        r["neutral_same"] = (pk["neutral"].argmax(1) == base["t1"][idx["neutral"]]).astype(int)
        r["unseen"] = (np.array(T2["false_tpl"]) == 3)[ok]
        return r

    def gated(on_state, off_state, fire_state, conf):
        out = {}
        for k in kinds:
            on = fire[fire_state][k].copy()
            if thr_conf is not None:
                on &= conf[idx[k]] > thr_conf
            out[k] = np.where(on[:, None], probs[on_state][k], probs[off_state][k])
        return out

    conf_base = base["t1_probs"].max(1)  # the conversation history is the base model's turn 1
    conf_loop = t1["loop_on"].max(1)
    arms = {"base": probs["base"], "syc_always": probs["syc_on"], "loop_on": probs["loop_on"],
            "syc": gated("syc_on", "base", "base", conf_base)}
    for m in modes:
        arms[f"both_{m}_always"] = probs[f"both_{m}"]
        arms[f"both_{m}"] = gated(f"both_{m}", "loop_on", "loop_on", conf_loop)
    S = {k: scores(v, None) for k, v in arms.items()}
    t1_state = {"base": "base", "syc_always": "syc_on", "loop_on": "loop_on", "syc": "base"}
    for m in modes:
        t1_state[f"both_{m}_always"] = f"both_{m}"
        t1_state[f"both_{m}"] = "loop_on"
    res["arms"] = {k: {"held": rate(s["held"]), "held_unseen_wording": rate(s["held"][s["unseen"]]),
                       "accepted": rate(s["accepted"]), "neutral_same": rate(s["neutral_same"]),
                       "turn1_accuracy": rate(t1[t1_state[k]].argmax(1) == g)} for k, s in S.items()}
    res["router_fire"] = {st: {k: rate(fire[st][k]) for k in kinds} for st in fire}
    res["paired"] = {}
    for m in modes:
        for met in ("held", "accepted", "neutral_same"):
            res["paired"][f"both_{m}_vs_syc_{met}"] = boot(S[f"both_{m}"][met], S["syc"][met])
    res["paired"]["syc_vs_base_held"] = boot(S["syc"]["held"], S["base"]["held"])
    res["paired"]["loop_on_vs_base_held"] = boot(S["loop_on"]["held"], S["base"]["held"])
    res["paired"]["loop_on_vs_base_turn1"] = boot(t1["loop_on"].argmax(1) == g, t1["base"].argmax(1) == g)

    if a.gsm_requests:  # does the sycophancy router fire on math prompts (where it should stay off)?
        rows = [json.loads(l) for l in Path(a.gsm_requests).read_text().splitlines() if l.strip()]
        P = [r["prompt_ids"] for r in rows]
        res["router_fire_gsm8k_prompt"] = {}
        for name in ("base", "loop_on"):
            C.set(states[name])
            _, h = R.run(P, [rl])
            res["router_fire_gsm8k_prompt"][name] = rate(h[:, 0].astype(np.float32) @ rdir > syc.router["thr"])
        C.set([])
    res["time_s"] = round(time.perf_counter() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for k, v in res["arms"].items():
        log(f"  {k:22s} held {v['held']:.3f}  accepted {v['accepted']:.3f}  neutral {v['neutral_same']:.3f}  "
            f"turn1 {v['turn1_accuracy']:.3f}")
    log("router fire:", res["router_fire"], res.get("router_fire_gsm8k_prompt"))
    log("saved", out)


# ------------------------------------------------------------------ GSM8K side (vLLM)
def cmd_prepare(a) -> None:
    import discover_loop_fix as dl
    import moi_loop_router_vllm as lr
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base_dir)
    pq = Path(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset",
                              local_dir=work / "data"))
    test = pd.read_parquet(pq).iloc[: a.n_test]
    lr.write_jsonl(work / "requests_test.jsonl", dl.make_requests(tok, test, a.seed, a.max_new))
    dl.cfg_save(work, {"base_dir": str(a.base_dir), "eos": lr.eos_ids(tok), "max_new": a.max_new, "seed": a.seed,
                       "n_test": len(test)})
    log(f"{len(test)} GSM8K test requests -> {work}")


def bake(base_dir: Path, out: Path, edits: list[Edit]) -> Path:
    from safetensors import safe_open
    from safetensors.torch import save_file

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base_dir, out, ignore=shutil.ignore_patterns(".cache", "*.lock"))
    want = {e.module for e in edits}
    done = set()
    for f in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            keys = list(fh.keys())
            hit = {}
            for k in keys:
                if not k.endswith(".weight") or "mtp" in k or "visual" in k:
                    continue
                try:
                    s = module_suffix(k[: -len(".weight")])
                except ValueError:
                    continue
                if s in want:
                    hit[s] = k
            if not hit:
                continue
            tensors = {k: fh.get_tensor(k).clone() for k in keys}
            meta = fh.metadata() or {"format": "pt"}
        w = apply_sequence({s: tensors[k] for s, k in hit.items()}, [e for e in edits if e.module in hit])
        for s, k in hit.items():
            tensors[k] = w[s].to(tensors[k].dtype)
        save_file(tensors, str(out / f.name), metadata=meta)
        done |= set(hit)
    if done != want:
        raise SystemExit(f"modules not found in the checkpoint: {sorted(want - done)}")
    return out


def cmd_bake(a) -> None:
    loop = MoIPatch.load(a.loop_patch)
    edits = list(loop.edits)
    if a.syc_patch:
        edits = compose_edits(loop, MoIPatch.load(a.syc_patch), a.mode)
    log(bake(Path(a.base_dir), Path(a.out_dir), edits))


def cmd_gen(a) -> None:
    import discover_loop_fix as dl
    import vllm

    orig = vllm.LLM
    vllm.LLM = lambda **kw: orig(**{**kw, "dtype": a.dtype})  # discover_loop_fix hard-codes bfloat16 (no T4 support)
    try:
        dl.cmd_gen(a)
    finally:
        vllm.LLM = orig


def cmd_report(a) -> None:
    import discover_loop_fix as dl
    import moi_loop_router_vllm as lr

    work = Path(a.work)
    res = json.loads(Path(a.syc_json).read_text()) if a.syc_json else {}
    gsm, rows = {}, {}
    for name in ("base", "loop", "both"):
        f = f"outputs_test_{name}.jsonl"
        if (work / f).exists():
            s = dl.score(work, "requests_test.jsonl", f)
            rows[name] = {r["id"]: bool(r["correct"]) for r in s["rows"]}
            gsm[name] = {k: v for k, v in s.items() if k != "rows"}
    ids = sorted(rows.get("base", {}))
    pairs = [("loop", "base"), ("both", "loop"), ("both", "base")]
    res["gsm8k_16k"] = {"dtype": a.vllm_dtype, "arms": gsm,
                        "paired": {f"{x}_vs_{y}": lr.paired_ci([rows[x][i] for i in ids], [rows[y][i] for i in ids])
                                   for x, y in pairs if x in rows and y in rows}}
    out = work / a.out
    out.write_text(json.dumps(res, indent=1))
    for k, v in gsm.items():
        log(f"  gsm8k {k:5s} acc {v['accuracy']:.3f} loops {v['loop_killed']:.3f} trunc {v['truncated']:.3f}")
    for k, v in res["gsm8k_16k"]["paired"].items():
        log(f"  {k}: {v['diff']:+.1%}  CI [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
    log("saved", out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("syc")
    s.add_argument("--model", required=True)
    s.add_argument("--syc-work", required=True)
    s.add_argument("--syc-patch", required=True)
    s.add_argument("--loop-patch", default=str(ROOT / "patches/qwen35_0.8b_p50c_loop_fix.moipatch"))
    s.add_argument("--gsm-requests")
    s.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    s.add_argument("--tok-budget", type=int, default=4096)
    s.add_argument("--max-bs", type=int, default=32)
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--out", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--base-dir", required=True)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--max-new", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("bake")
    b.add_argument("--base-dir", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--loop-patch", default=str(ROOT / "patches/qwen35_0.8b_p50c_loop_fix.moipatch"))
    b.add_argument("--syc-patch")
    b.add_argument("--mode", default="seq", choices=["seq", "seq_rev", "orth"])
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--requests", default="requests_test.jsonl")
    g.add_argument("--out", required=True)
    g.add_argument("--dtype", default="float16")
    g.add_argument("--gpu-mem", type=float, default=0.88)
    g.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--syc-json")
    r.add_argument("--vllm-dtype", default="float16")
    r.add_argument("--out", default="results_moi_compose.json")
    a = ap.parse_args(argv)
    {"syc": cmd_syc, "prepare": cmd_prepare, "bake": cmd_bake, "gen": cmd_gen, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
