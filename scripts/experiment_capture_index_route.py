"""End-to-end capture -> index -> route on a real model (default: local Qwen3.5-0.8B).

1. Capture : build one ActivationIndex per task with `wsi.capture_index`
             (math = GSM8K train question + solution, wiki = WikiText-2 validation).
2. Index   : save them, report their size, how much the math and wiki neuron sets overlap.
3. Route   : for held-out prompts of both tasks (GSM8K test, WikiText-2 test) the TaskRouter picks an index from a
             one-prefill activation profile; the MLPs are then physically sliced to that index's neurons.
Scored by teacher-forced perplexity on each held-out text with: the full model, the routed index, the WRONG index
(the other task's), and random neurons, at each budget. Also a dense `trace_all` of a short prompt, to show what
"capture everything" weighs per token.

    python scripts/experiment_capture_index_route.py --model-dir models/qwen35-0.8b-full
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wsi.capture import ActivationIndex, TaskRouter, apply_selection, capture_index, trace_all  # noqa: E402


def clean_solution(ans: str) -> str:
    body, final = ans.split("####")
    return re.sub(r"<<[^>]*>>", "", body).strip() + f"\nThe answer is {final.strip()}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "qwen35-0.8b-full"))
    ap.add_argument("--data-dir", default=str(ROOT / "work"))
    ap.add_argument("--calib-tokens", type=int, default=60_000)
    ap.add_argument("--n-eval", type=int, default=60, help="held-out texts per task")
    ap.add_argument("--budgets", default="0.7,0.5,0.3")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "capture_index_route" / "results.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    data = Path(args.data_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []

    def chat(q: str, a: str) -> list[int]:
        txt = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                      tokenize=False, enable_thinking=False)
        return tok(txt, add_special_tokens=False).input_ids

    def take(seqs, budget):
        out, n = [], 0
        for s in seqs:
            out.append(s)
            n += len(s)
            if n >= budget:
                break
        return out

    gtr = pd.read_parquet(data / "gsm8k" / "train.parquet")
    gte = pd.read_parquet(data / "gsm8k" / "test.parquet").iloc[: args.n_eval]
    wv = tok("\n".join(pd.read_parquet(data / "wikitext" / "validation.parquet")["text"]), add_special_tokens=False).input_ids
    wt = tok("\n".join(pd.read_parquet(data / "wikitext" / "test.parquet")["text"]), add_special_tokens=False).input_ids
    calib = {
        "math": take((chat(q, clean_solution(a)) for q, a in zip(gtr["question"], gtr["answer"])), args.calib_tokens),
        "wiki": [bos + wv[i : i + 512] for i in range(0, min(len(wv), args.calib_tokens), 512)],
    }
    held = {
        "math": [chat(q, clean_solution(a)) for q, a in zip(gte["question"], gte["answer"])],
        "wiki": [bos + wt[i : i + 256] for i in range(0, 256 * args.n_eval, 256)],
    }

    # ---- 1-2. capture and index
    out_dir = Path(args.out).parent
    idx: dict[str, ActivationIndex] = {}
    res: dict = {"model": args.model_dir, "capture": {}}
    for task, seqs in calib.items():
        t0 = time.perf_counter()
        idx[task] = capture_index(model, seqs, task=task)
        p = idx[task].save(out_dir / f"index_{task}")
        res["capture"][task] = {"tokens": idx[task].tokens, "sites": len(idx[task].stats),
                                "mlp_groups": len(idx[task].mlps), "index_mb": round(idx[task].size_bytes() / 2**20, 2),
                                "seconds": round(time.perf_counter() - t0, 1), "path": str(p)}
        print(f"[capture {task}] {res['capture'][task]}")
    budgets = [float(b) for b in args.budgets.split(",")]
    res["overlap_math_wiki"] = {str(b): round(idx["math"].overlap(idx["wiki"], b), 4) for b in budgets}
    print("math/wiki neuron overlap:", res["overlap_math_wiki"])

    # ---- 3. route
    router = TaskRouter(idx)
    routes = {t: [router.route(model, s[:256])["task"] for s in seqs] for t, seqs in held.items()}
    res["routing_accuracy"] = {t: sum(r == t for r in rs) / len(rs) for t, rs in routes.items()}
    print("routing accuracy:", res["routing_accuracy"])

    def ppl(seqs) -> float:
        nll, nt = 0.0, 0
        for s in seqs:
            ids = torch.tensor([s], device=dev)
            lp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
            nll -= float(lp.gather(1, ids[0, 1:, None]).sum())
            nt += len(s) - 1
        return math.exp(nll / nt)

    res["ppl"] = {t: {"full": ppl(s)} for t, s in held.items()}
    g = torch.Generator().manual_seed(0)
    for b in budgets:
        for t, seqs in held.items():
            other = "wiki" if t == "math" else "math"
            row = {}
            # routed: each text loads the index its own prompt was routed to
            by_task: dict[str, list] = {}
            for s, r in zip(seqs, routes[t]):
                by_task.setdefault(r, []).append(s)
            nll_parts = []
            for r, ss in by_task.items():
                with router.loaded(model, r, b) as info:
                    nll_parts.append((math.log(ppl(ss)) * sum(len(x) - 1 for x in ss), sum(len(x) - 1 for x in ss)))
            row["routed"] = math.exp(sum(a for a, _ in nll_parts) / sum(n for _, n in nll_parts))
            with router.loaded(model, other, b):
                row["wrong_index"] = ppl(seqs)
            rand = {g_.down: torch.randperm(g_.units, generator=g)[: round(b * g_.units)].sort().values
                    for g_ in idx[t].mlps}
            with apply_selection(model, rand):
                row["random"] = ppl(seqs)
            row["params_loaded_share"] = round(info["params_loaded"] / info["params_full"], 4)
            res["ppl"][t][str(b)] = row
            print(f"[{t} @ {b:.0%}] full={res['ppl'][t]['full']:.2f} routed={row['routed']:.2f} "
                  f"wrong={row['wrong_index']:.2f} random={row['random']:.2f} model size={row['params_loaded_share']:.0%}",
                  flush=True)

    # ---- capture everything, densely, for a short prompt
    tr = trace_all(model, held["math"][0], max_tokens=16)
    n_floats = sum(v.numel() for v in tr.values())
    res["trace_all_16_tokens"] = {"tensors": len(tr), "floats": n_floats, "mb_fp32": round(n_floats * 4 / 2**20, 1),
                                  "kinds": sorted({k.split(":")[-1] if ":" in k else k.split(".")[0] for k in tr})}
    print("trace_all (16 tokens):", res["trace_all_16_tokens"])
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
