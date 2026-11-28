"""EXPERIMENTAL: does a sensor token work inside a real vLLM run?

experiment_final_sensor.py showed (fp32, saved answers) that a linear readout of the final hidden state, w . h_final,
detects loops almost like the L13-14 veto. In vLLM that readout would come from a spare vocabulary row whose lm_head row
is w, read with `logprob_token_ids=[row]` and `logprobs_mode="raw_logits"` (taken before logits processors) and blocked
from sampling with `logit_bias={row: -100}`. Checks, on GSM8K test prompts of the discover_blind run:

  1. identical outputs: the sensor checkpoint generates exactly the tokens of the unmodified checkpoint (both runs carry
     the same logit_bias; a second unmodified run measures vLLM's own run-to-run nondeterminism first)
  2. the value vLLM returns is the sensor: compared per token with an fp32 transformers recompute of w . h_final on the
     same tokens (correlation, error relative to the spread, agreement of the smoothed score with the fp32 threshold)
  3. the row is never sampled

  build    (CPU)  python scripts/experiment_sensor_vllm.py build --model M --dirs D.npz --out SENSOR_DIR
  gen      (GPU)  python scripts/experiment_sensor_vllm.py gen   --work W --model M_or_SENSOR --out O.jsonl [--sensor]
  check    (GPU)  python scripts/experiment_sensor_vllm.py check --work W --model M --dirs D.npz --base O1 --base2 O2 \
                         --sensor-out O3 --results R.json --out C.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

ROW = 248070          # first unused row of the Qwen3.5 vocabulary (tokenizer ids end at 248069; vocab 248320)
SCALE = 1.0 / 200.0   # keeps the sensor logit within ~[-5, 8] so a -100 bias always blocks it
BIAS = -100.0         # vLLM clamps logit_bias to [-100, 100]
SMOOTH = 32


def sensor_vector(dirs_npz: str, which: str = "elicit") -> np.ndarray:
    z = np.load(dirs_npz)
    names = [str(x) for x in z["dirs"]]
    return z["W"][names.index(which)].astype(np.float64)


def cmd_build(a) -> None:
    """Copy the checkpoint and write SCALE * w into row ROW of every vocabulary matrix (embedding and lm_head;
    Qwen3.5-0.8B ties them, and the row is never fed as input)."""
    import torch
    from safetensors.torch import load_file, save_file

    src, dst = Path(a.model), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    w = torch.from_numpy(sensor_vector(a.dirs) * SCALE)
    changed = []
    for f in src.iterdir():
        if f.suffix != ".safetensors":
            if f.is_file():
                shutil.copy2(f, dst / f.name)
            continue
        t = load_file(str(f))
        for k, v in t.items():
            if (k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight")) and v.ndim == 2 \
                    and v.shape[0] > ROW and v.shape[1] == w.numel():
                v[ROW] = w.to(v.dtype)
                changed.append(k)
        save_file(t, str(dst / f.name), metadata={"format": "pt"})
    assert changed, "no vocabulary matrix found"
    (dst / "sensor.json").write_text(json.dumps({"row": ROW, "scale": SCALE, "bias": BIAS, "dirs": a.dirs,
                                                 "changed": changed}, indent=1))
    print("sensor row", ROW, "written into", changed)


def prompts(work: Path, n: int) -> list[dict]:
    import moi_loop_router_vllm as lr
    return lr.read_jsonl(work / "requests_test.jsonl")[:n]


def logits_of(step: dict, row: int) -> float | None:
    """vLLM returns per generated token a dict {token_id: Logprob}; in raw_logits mode .logprob holds the raw logit."""
    if step is None or row not in step:
        return None
    v = step[row]
    return float(getattr(v, "logprob", v))


def cmd_gen(a) -> None:
    import moi_loop_router_vllm as lr
    from vllm import LLM, SamplingParams, TokensPrompt

    reqs = prompts(Path(a.work), a.n)
    llm = LLM(model=a.model, dtype="bfloat16", seed=0, gpu_memory_utilization=0.85,
              max_model_len=max(len(r["prompt_ids"]) for r in reqs) + a.max_tokens + 8,
              logprobs_mode="raw_logits", trust_remote_code=True)
    sps = [SamplingParams(**lr.SAMPLING, max_tokens=a.max_tokens, seed=int(r.get("seed", i)),
                          logit_bias={ROW: BIAS}, logprob_token_ids=[ROW] if a.sensor else None)
           for i, r in enumerate(reqs)]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in reqs], sps)
    rows = []
    for r, o in zip(reqs, outs):
        c = o.outputs[0]
        row = {"id": r["id"], "token_ids": list(c.token_ids), "finish_reason": c.finish_reason}
        if a.sensor:
            row["sensor"] = [logits_of(s, ROW) for s in (c.logprobs or [])]
        rows.append(row)
    lr.write_jsonl(Path(a.out), rows)
    import vllm
    print("saved", a.out, len(rows), "answers", sum(len(x["token_ids"]) for x in rows), "tokens",
          f"{time.time() - t0:.0f}s", "vllm", vllm.__version__)


def smooth(x: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    c = np.cumsum(np.insert(np.asarray(x, np.float64), 0, 0.0))
    k = np.arange(len(x))
    lo = np.maximum(0, k - w + 1)
    return (c[k + 1] - c[lo]) / (k + 1 - lo)


def identical(a: list[dict], b: list[dict]) -> dict:
    same = sum(x["token_ids"] == y["token_ids"] for x, y in zip(a, b))
    first = [next((i for i, (p, q) in enumerate(zip(x["token_ids"], y["token_ids"])) if p != q),
                  min(len(x["token_ids"]), len(y["token_ids"])))
             for x, y in zip(a, b) if x["token_ids"] != y["token_ids"]]
    return {"identical": f"{same}/{len(a)}", "identical_rate": round(same / max(1, len(a)), 4),
            "median_first_diff": float(np.median(first)) if first else None}


def cmd_check(a) -> None:
    import moi_loop_router_vllm as lr
    import torch
    from transformers import AutoModelForCausalLM

    for _m in ("fla", "causal_conv1d"):
        sys.modules.setdefault(_m, None)
    work = Path(a.work)
    reqs = {r["id"]: r for r in prompts(work, 10 ** 9)}
    base, base2, sens = (lr.read_jsonl(Path(p)) for p in (a.base, a.base2, a.sensor_out))
    res = {"n": len(sens), "row": ROW, "scale": SCALE,
           "base_vs_base2": identical(base, base2), "base_vs_sensor": identical(base, sens),
           "row_sampled": sum(ROW in x["token_ids"] for x in sens)}
    missing = sum(v is None for x in sens for v in x["sensor"])
    res["sensor_values_missing"] = missing

    # fp32 recompute of w . h_final on the very tokens vLLM generated (sensor run)
    w = torch.from_numpy(sensor_vector(a.dirs)).float()
    thr = json.loads(Path(a.results).read_text())["replication_elicit_to_test"]["final_sensor"]["calibration"]["thr"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    backbone = getattr(model, "model", model)
    vv, ff, agree, n_tok = [], [], 0, 0
    for x in sens:
        p = reqs[x["id"]]["prompt_ids"]
        ids = p + x["token_ids"]
        with torch.no_grad():
            h = backbone(input_ids=torch.tensor([ids[:-1]], device=dev), use_cache=False).last_hidden_state[0]
        # the logit that produced generated token t is computed at position len(p) - 1 + t
        f32 = (h[len(p) - 1:].float().cpu() @ w).numpy()
        v = np.asarray([np.nan if s is None else s / SCALE for s in x["sensor"]], np.float64)
        k = min(len(v), len(f32))
        v, f32 = v[:k], f32[:k]
        ok = ~np.isnan(v)
        vv.append(v[ok])
        ff.append(f32[ok])
        sv, sf = smooth(np.nan_to_num(v)), smooth(f32)
        agree += int(((sv > thr) == (sf > thr)).sum())
        n_tok += k
    vv, ff = np.concatenate(vv), np.concatenate(ff)
    err = vv - ff
    res["sensor_vs_fp32"] = {"tokens": int(len(vv)), "corr": round(float(np.corrcoef(vv, ff)[0, 1]), 6),
                             "mean_abs_err": round(float(np.abs(err).mean()), 4),
                             "p99_abs_err": round(float(np.percentile(np.abs(err), 99)), 4),
                             "fp32_std": round(float(ff.std()), 4), "mean_err": round(float(err.mean()), 4),
                             "smoothed_threshold_agreement": round(agree / max(1, n_tok), 6), "thr_fp32": thr}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--dirs", required=True)
    b.add_argument("--out", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--work", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--sensor", action="store_true")
    g.add_argument("--n", type=int, default=64)
    g.add_argument("--max-tokens", type=int, default=2048)
    c = sub.add_parser("check")
    for k in ("--work", "--model", "--dirs", "--base", "--base2", "--sensor-out", "--results", "--out"):
        c.add_argument(k, required=True)
    a = ap.parse_args()
    {"build": cmd_build, "gen": cmd_gen, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
