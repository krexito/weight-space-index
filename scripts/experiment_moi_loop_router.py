"""MoI loop router: apply the p50c loop-removal edit ONLY when the model starts looping.

Background: the MIT CSAIL p50c edit (rank-1 on mlp.down_proj, 64 columns in layers 15 and 19 of Qwen3.5-0.8B) removes
degenerate reasoning loops (MATH 7.9% -> 48.1%). Like AHA's and N. Barry's edits, it is applied permanently.
Here the edit is kept outside the model as a Mixture-of-Indexes payload (131,072 deltas) and switched on per sequence
by a causal router that watches the generated tokens.

Arms (same model, same prompts, same sampling seed):
  * base   : unedited Qwen3.5-0.8B.
  * fixed  : p50c always on (what the paper / Barry / AHA do).
  * moi    : p50c off until the router fires, then latched on for the rest of that sequence.
The edit is applied additively (out += h[:, cols] @ dW_cols^T) through a hook on down_proj, so all three arms share
one set of weights and the MoI mask can differ per batch row.

Router (causal, per sequence; Barry's loop definition): dup[t] = 1 if the 16-token window ending at t already occurred
earlier in the response. Fire when >= `--router-frac` of the last `--router-window` response tokens are dup.
Early stop (all arms, same rule): a sequence whose last `--kill-window` tokens are >= 90% dup is stopped and counted
as a loop failure ("cut"). This replaces Barry's 32,768-token budget, which is unaffordable with HF on Windows; it can
only lose problems that would otherwise have recovered from a long verbatim loop, and it does so equally for all arms.

Also measures WikiText-2 test perplexity (teacher-forced; MoI applies the router causally per position).

Usage (needs models/qwen35-0.8b-full, work/gsm8k/test.parquet, work/wikitext/test.parquet):
    python scripts/experiment_moi_loop_router.py --n-problems 64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, StoppingCriteria
from transformers import StoppingCriteriaList

ROOT = Path(__file__).resolve().parent.parent
QWEN = ROOT / "models" / "qwen35-0.8b-full"
EDIT = ROOT / "benchmarks" / "exp8_math"
LAYERS = (15, 19)
K_DUP = 16
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


# ------------------------------------------------------------------ edit payload
def load_p50c_deltas(base_file: Path) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Rebuild the official p50c tensors from the sparse release and return per-layer (cols, dW[:, cols])."""
    man = json.loads((EDIT / "p50c_manifest.json").read_text())
    z = np.load(EDIT / "p50c_edit.npz", allow_pickle=False)
    out = {}
    with safe_open(str(base_file), "pt") as f:
        for i, t in enumerate(man["tensors"]):
            L = int(re.search(r"layers\.(\d+)\.", t["name"]).group(1))
            base = f.get_tensor(t["name"])  # bf16 [1024, 3584]
            sha = hashlib.sha256(base.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
            if sha != t["base_sha256"]:
                raise SystemExit(f"{t['name']}: base weights do not match the p50c release (sha {sha[:12]}...)")
            bits = base.view(torch.int16).flatten().clone()
            vals = z[f"t{i}_val"].astype(np.uint16).view(np.int16)
            bits[torch.from_numpy(z[f"t{i}_idx"].astype(np.int64))] = torch.from_numpy(vals)
            edited = bits.view(torch.bfloat16).reshape(base.shape)
            dW = edited.float() - base.float()
            cols = torch.nonzero(dW.abs().sum(0) > 0).flatten()
            out[L] = (cols, dW[:, cols].contiguous())
    return out


def find_base_tensor_file() -> Path:
    files = sorted(QWEN.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"No safetensors in {QWEN}")
    return files[0]


# ------------------------------------------------------------------ loop detection
def dup_flags(ids: list[int], k: int = K_DUP) -> list[int]:
    seen, out = set(), [0] * len(ids)
    for t in range(k - 1, len(ids)):
        w = tuple(ids[t - k + 1 : t + 1])
        out[t] = int(w in seen)
        seen.add(w)
    return out


class LoopTracker:
    """Incremental per-row dup tracking over the RESPONSE tokens (prompt excluded)."""

    def __init__(self, n_rows: int, prompt_len: int, eos_ids: set[int]):
        self.prompt_len = prompt_len
        self.eos_ids = eos_ids
        self.finished = [False] * n_rows
        self.seen = [set() for _ in range(n_rows)]
        self.dups: list[list[int]] = [[] for _ in range(n_rows)]
        self.fired_at = [None] * n_rows

    def update(self, input_ids: torch.Tensor) -> None:
        resp = input_ids[:, self.prompt_len :]
        n = resp.shape[1]
        if n == 0:
            return
        last = resp[:, -K_DUP:].tolist() if n >= K_DUP else None
        newest = resp[:, -1].tolist()
        for b in range(resp.shape[0]):
            if self.finished[b] or len(self.dups[b]) >= n:
                continue
            if newest[b] in self.eos_ids:  # row is done: freeze its loop statistics
                self.finished[b] = True
                continue
            if last is None:
                self.dups[b].append(0)
                continue
            w = tuple(last[b])
            self.dups[b].append(int(w in self.seen[b]))
            self.seen[b].add(w)

    def frac(self, b: int, window: int) -> float:
        d = self.dups[b]
        if len(d) < window:
            return 0.0
        return sum(d[-window:]) / window


# ------------------------------------------------------------------ grading
def extract_answer(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    cand = boxed[-1] if boxed else text
    nums = re.findall(r"-?\d+(?:\.\d+)?", cand.replace(",", ""))
    return nums[-1] if nums else None


def correct(pred: str | None, gold: str) -> bool:
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-problems", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--router-window", type=int, default=64)
    ap.add_argument("--router-frac", type=float, default=0.5)
    ap.add_argument("--kill-window", type=int, default=768)
    ap.add_argument("--arms", type=str, default="base,fixed,moi")
    ap.add_argument("--ppl-tokens", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-every", type=int, default=512, help="log GPU memory (and maybe free idle blocks) every N steps")
    ap.add_argument("--empty-cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", type=str, default=str(ROOT / "benchmarks" / "moi_loop_router" / "results.json"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    t_start = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(QWEN)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16).to(dev).eval()
    deltas = load_p50c_deltas(find_base_tensor_file())
    print("p50c payload:", {L: tuple(d[1].shape) for L, d in deltas.items()},
          f"{sum(d[1].numel() for d in deltas.values())} deltas")

    # find the decoder layers (text model inside a possibly multimodal wrapper)
    layers = None
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) >= 20:
            layers = mod
            break
    assert layers is not None, "decoder layers not found"

    # ---- additive edit hook: out += gate[b, t] * (h[..., cols] @ dW_cols^T) ----
    state: dict = {"gate": None}  # None = off, else float tensor broadcastable to [B, T, 1]
    for L, (cols, dWc) in deltas.items():
        cols_d, dW_d = cols.to(dev), dWc.to(dev, torch.float32)

        def hook(mod, inp, out, cols_d=cols_d, dW_d=dW_d):
            g = state["gate"]
            if g is None:
                return out
            h = inp[0][..., cols_d].float()
            add = (h @ dW_d.T) * g
            return (out.float() + add).to(out.dtype)

        layers[L].mlp.down_proj.register_forward_hook(hook)

    gsm = pd.read_parquet(ROOT / "work" / "gsm8k" / "test.parquet").iloc[: args.n_problems]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + "\n" + INSTRUCTION}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for q in gsm["question"]]
    golds = [a.split("####")[-1].strip().replace(",", "") for a in gsm["answer"]]
    ge = model.generation_config.eos_token_id
    eos_ids = set(ge if isinstance(ge, list) else [ge]) | {tok.eos_token_id, tok.pad_token_id}
    eos_ids.discard(None)

    class Presence(LogitsProcessor):
        """presence_penalty=1.5 (OpenAI/vLLM semantics) over response tokens, plus the MoI router update."""

        def __init__(self, tracker: LoopTracker, arm: str):
            self.tr, self.arm = tracker, arm

        def __call__(self, input_ids, scores):
            self.tr.update(input_ids)
            resp = input_ids[:, self.tr.prompt_len :]
            n = resp.shape[1]
            if n and n % args.mem_every == 0:
                # the growing KV cache (torch.cat per step) fragments the caching allocator; hand back idle blocks
                alloc, resv = torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30
                if args.empty_cache:
                    torch.cuda.empty_cache()
                live = sum(not f for f in self.tr.finished)
                print(f"    step {n}: alloc {alloc:.2f} GB, reserved {resv:.2f} GB"
                      f" -> {torch.cuda.memory_reserved() / 2**30:.2f} GB, rows still writing {live}", flush=True)
            if resp.shape[1]:
                present = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, resp, True)
                scores = scores - 1.5 * present.to(scores.dtype)
            if self.arm == "moi":
                g = state["gate"]
                for b in range(input_ids.shape[0]):
                    if self.tr.fired_at[b] is None and self.tr.frac(b, args.router_window) >= args.router_frac:
                        self.tr.fired_at[b] = resp.shape[1]
                        g[b] = 1.0
            return scores

    class KillLoops(StoppingCriteria):
        def __init__(self, tracker: LoopTracker):
            self.tr = tracker
            self.killed = set()

        def __call__(self, input_ids, scores, **kw):
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            for b in range(input_ids.shape[0]):
                if not self.tr.finished[b] and self.tr.frac(b, args.kill_window) >= 0.9:
                    done[b] = True
                    self.killed.add(b)
            return done

    def run_arm(arm: str) -> dict:
        torch.manual_seed(args.seed)
        rows = []
        t0 = time.perf_counter()
        for i in range(0, len(prompts), args.batch):
            enc = tok(prompts[i : i + args.batch], return_tensors="pt", padding=True).to(dev)
            B, P = enc.input_ids.shape
            tr = LoopTracker(B, P, eos_ids)
            killer = KillLoops(tr)
            if arm == "base":
                state["gate"] = None
            elif arm == "fixed":
                state["gate"] = torch.ones(B, 1, 1, device=dev)
            else:
                state["gate"] = torch.zeros(B, 1, 1, device=dev)
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                logits_processor=LogitsProcessorList([Presence(tr, arm)]),
                stopping_criteria=StoppingCriteriaList([killer]), pad_token_id=tok.pad_token_id,
            )
            state["gate"] = None
            for b in range(B):
                resp = out[b, P:]
                resp = resp[resp != tok.pad_token_id]
                text = tok.decode(resp, skip_special_tokens=True)
                pred = extract_answer(text)
                n_resp = int(resp.numel())
                looped = b in killer.killed
                truncated = (not looped) and n_resp >= args.max_new_tokens
                fired = tr.fired_at[b]
                rows.append({
                    "id": i + b, "gold": golds[i + b], "pred": pred,
                    "correct": bool(correct(pred, golds[i + b]) and not looped and not truncated),
                    "tokens": n_resp, "looped_killed": looped, "truncated": truncated,
                    "router_fired_at": fired,
                    "edit_token_share": (0.0 if arm == "base" else 1.0 if arm == "fixed"
                                         else (0.0 if fired is None else (n_resp - fired) / max(n_resp, 1))),
                })
            print(f"  [{arm}] {min(i + args.batch, len(prompts))}/{len(prompts)} done "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        n = len(rows)
        return {
            "accuracy": sum(r["correct"] for r in rows) / n,
            "loop_killed_share": sum(r["looped_killed"] for r in rows) / n,
            "truncated_share": sum(r["truncated"] for r in rows) / n,
            "mean_tokens": sum(r["tokens"] for r in rows) / n,
            "router_fired_share": (sum(r["router_fired_at"] is not None for r in rows) / n) if arm == "moi" else None,
            "mean_edit_token_share": sum(r["edit_token_share"] for r in rows) / n,
            "wall_s": time.perf_counter() - t0,
            "per_problem": rows,
        }

    # ---- teacher-forced WikiText perplexity; MoI gate per position from the causal router ----
    wt = "\n".join(pd.read_parquet(ROOT / "work" / "wikitext" / "test.parquet")["text"].tolist())
    wt_ids = tok(wt, add_special_tokens=False).input_ids[: args.ppl_tokens]

    def ppl(arm: str) -> dict:
        nll, nt, gated = 0.0, 0, 0
        for i in range(0, len(wt_ids) - 16, 512):
            ch = wt_ids[i : i + 512]
            ids = torch.tensor([ch], device=dev)
            if arm == "base":
                state["gate"] = None
            elif arm == "fixed":
                state["gate"] = torch.ones(1, 1, 1, device=dev)
            else:
                d = dup_flags(ch)
                g = torch.zeros(1, len(ch), 1, device=dev)
                for t in range(len(ch)):
                    w = d[max(0, t - args.router_window + 1) : t + 1]
                    if t + 1 >= args.router_window and sum(w) / args.router_window >= args.router_frac:
                        g[0, t:] = 1.0
                        break
                gated += int(g.sum())
                state["gate"] = g
            lp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
            nll -= float(lp.gather(1, ids[0, 1:, None]).sum())
            nt += len(ch) - 1
        state["gate"] = None
        return {"ppl": math.exp(nll / nt), "tokens": nt, "gated_positions": gated}

    results: dict = {}
    for arm in [a.strip() for a in args.arms.split(",")]:
        r = {"wikitext": ppl(arm)}
        r["gsm8k"] = run_arm(arm)
        results[arm] = r
        g = r["gsm8k"]
        print(f"[{arm:5s}] GSM8K acc={g['accuracy']:.1%} loop_killed={g['loop_killed_share']:.1%} "
              f"trunc={g['truncated_share']:.1%} tokens={g['mean_tokens']:.0f} "
              f"edit_share={g['mean_edit_token_share']:.1%}"
              + (f" router_fired={g['router_fired_share']:.1%}" if g["router_fired_share"] is not None else "")
              + f" | WikiText ppl={r['wikitext']['ppl']:.3f}", flush=True)

    summary = {"model": "Qwen/Qwen3.5-0.8B", "edit": "MIT CSAIL exp8_math p50c", "args": vars(args),
               "payload_deltas": int(sum(d[1].numel() for d in deltas.values())),
               "results": results, "wall_s": time.perf_counter() - t_start}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
