"""Test 4: Can `CompactTrace` (resid + K1024 fp4 MLP + head_norms + attn_topk + cache) improve
loop circuit discovery (`discover_loop_fix` offline from traces) on `Qwen/Qwen3.5-0.8B`?

Runs in ~4-5 min on Colab L4:
  1. Generates `n_train` (24) GSM8K train solutions, identifies clean vs looping runs.
  2. Captures ONE `CompactTrace` per selected sequence (single forward pass each).
  3. Computes loop-vs-healthy Fisher separation OFFLINE across all 24 layers from:
       - MLP activations (`dequant_mlp_topk` -> W_down projection)
       - Residuals (`dequant_resid`)
       - Attention / Linear-Attn head L2 norms (`head_norms`) & `attn_topk` lag focus
  4. Grades candidate layers & top-64 neurons against `patches/qwen35_0.8b_p50c_loop_fix.moipatch`
     (blind exam: does `CompactTrace` recover Layer 15 AND Layer 19 better than MLP-only?)
  5. Evaluates held-out GSM8K accuracy & loop rate on `n_eval` (24) problems for:
     `base` vs `ct_mlp_only` vs `ct_hybrid (MLP+heads)` vs `p50c` reference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsi.moipatch import MoIPatch  # noqa: E402
from wsi.tracer import capture_compact, diff_compact, discover_compact_circuit  # noqa: E402


HUB_FILES = {
    "gsm8k/train.parquet": ("openai/gsm8k", "main/train-00000-of-00001.parquet"),
    "gsm8k/test.parquet": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
}


def need(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    repo, fn = HUB_FILES[name]
    got = Path(hf_hub_download(repo, fn, repo_type="dataset", local_dir=data_dir / "_hub"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(got.read_bytes())
    return path


def _local_or_hub(name_or_dir: str) -> str:
    p = Path(name_or_dir)
    if p.exists():
        return str(p)
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_dir)


def _extract_gold(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def _extract_pred(txt: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]+)\}", txt)
    if m:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", m[-1])
        if nums:
            return nums[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", txt)
    return nums[-1].replace(",", "") if nums else None


def _is_correct(txt: str, gold: str) -> bool:
    p = _extract_pred(txt)
    if p is None:
        return False
    try:
        return abs(float(p) - float(gold)) < 1e-4
    except ValueError:
        return p == gold


def _has_gold_anywhere(txt: str, gold: str) -> bool:
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", txt):
        cand = raw.replace(",", "")
        try:
            if abs(float(cand) - float(gold)) < 1e-4:
                return True
        except ValueError:
            if cand == gold:
                return True
    return False


def _has_ngram_repeat(ids: list[int], n: int = 8, reps: int = 3) -> bool:
    if len(ids) < n * reps:
        return False
    seen: dict[tuple[int, ...], int] = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i : i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= reps:
            return True
    return False


def _loop_onset(ids: list[int], n: int = 16, window: int = 64, frac: float = 0.5) -> int | None:
    """First response position where >= `frac` of the next `window` tokens end an already-seen n-gram (the repo's
    router rule, moi_loop_router_vllm.py: 16-grams, 50% over 64 tokens). None = no loop. Truncation alone is not a loop."""
    if len(ids) < n + window:
        return None
    seen: set[tuple[int, ...]] = set()
    dup = [0] * len(ids)
    for i in range(n - 1, len(ids)):
        g = tuple(ids[i - n + 1 : i + 1])
        dup[i] = int(g in seen)
        seen.add(g)
    run = sum(dup[:window])
    for t in range(len(ids) - window + 1):
        if run >= frac * window:
            return t
        if t + window < len(ids):
            run += dup[t + window] - dup[t]
    return None


def _batched_generate(model, tok, questions: list[str], dev: torch.device, max_new_tokens: int) -> list[dict]:
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
        for q in questions
    ]
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    in_width = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    rows = []
    eos_id = tok.eos_token_id
    for i in range(len(questions)):
        pad_left = int((enc["attention_mask"][i] == 0).sum().item())
        p_ids = enc["input_ids"][i, pad_left:].tolist()
        out_raw = gen[i, in_width:].tolist()
        if eos_id is not None and eos_id in out_raw:
            out_ids = out_raw[: out_raw.index(eos_id)]
            hit_eos = True
        else:
            out_ids = out_raw
            hit_eos = False
        full_ids = p_ids + out_ids
        txt = tok.decode(out_ids, skip_special_tokens=True)
        onset = _loop_onset(out_ids)
        rows.append({
            "prompt_len": len(p_ids),
            "gen_len": len(out_ids),
            "full_ids": full_ids,
            "text": txt,
            "loop": onset is not None,
            "onset_resp": onset,  # response-relative token where the repetition starts
            "finished": hit_eos,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--patch", default=str(ROOT / "patches" / "qwen35_0.8b_p50c_loop_fix.moipatch"))
    ap.add_argument("--data-dir", default=str(ROOT / "work"))
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-traces", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--out", default="results_loop_discover_compact.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data_dir)
    mdir = _local_or_hub(args.model)
    tok = AutoTokenizer.from_pretrained(mdir)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype, attn_implementation="eager").to(dev).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(mdir, dtype=dtype).to(dev).eval()

    gtr = pd.read_parquet(need(data, "gsm8k/train.parquet")).iloc[: args.n_train]
    gte = pd.read_parquet(need(data, "gsm8k/test.parquet")).iloc[: args.n_eval]

    # 1. Generate train pool in one batch
    t0 = time.perf_counter()
    tr_rows = _batched_generate(model, tok, list(gtr["question"]), dev, args.max_new_tokens)
    for r, a in zip(tr_rows, gtr["answer"]):
        r["gold"] = _extract_gold(a)
        r["correct"] = _is_correct(r["text"], r["gold"])
    train_gen_s = round(time.perf_counter() - t0, 1)
    loop_rows = [r for r in tr_rows if r["loop"]]
    clean_rows = [r for r in tr_rows if not r["loop"]]
    n_unfinished = sum(1 for r in clean_rows if not r["finished"])
    print(f"Train gen ({len(tr_rows)} seqs) in {train_gen_s}s: loop={len(loop_rows)} clean={len(clean_rows)} "
          f"(of which cut at max_new_tokens without repeating: {n_unfinished})", flush=True)
    if not loop_rows:
        raise SystemExit("No looping sequence in the train pool: raise --n-train or --max-new-tokens.")

    # 2. Capture CompactTrace once per selected sequence
    half = max(1, args.max_traces // 2)
    chosen = loop_rows[:half] + clean_rows[:half]
    if len(chosen) < min(args.max_traces, len(tr_rows)):
        extra = [r for r in tr_rows if r not in chosen][: max(0, args.max_traces - len(chosen))]
        chosen += extra

    traces = []
    t_cap = time.perf_counter()
    for r in chosen:
        ct = capture_compact(model, r["full_ids"])
        onset = ct.detect_loop_onset(layer=min(15, ct.n_layers - 1), window=16, thr=0.96)
        traces.append((r, ct, onset))
    cap_s = round(time.perf_counter() - t_cap, 2)
    bank_mb = round(sum(ct.size_bytes() for _, ct, _ in traces) / 2**20, 2)
    print(f"Captured {len(traces)} CompactTraces in {cap_s}s | total bank={bank_mb} MiB", flush=True)

    # 3. Offline Hybrid CompactTrace (origin) + WSI static weight resonance (echo amplifier) discovery
    disc = discover_compact_circuit(traces, model=model, alpha=args.alpha)
    layer_stats = disc["layer_stats"]
    u_mlp_by_layer = disc["directions"]
    top64_by_layer = disc["top_neurons"]
    mlp_ranked = disc["mlp_ranked"]
    hybrid_ranked = disc["hybrid_ranked"]
    fa_layers = disc["fa_layers"]
    fa_origins = disc["wsi_origins"]
    l_res_dst = disc["wsi_resonant_dst"]
    wsi_dir_ov = disc["wsi_resonant_directions"]
    wsi_idx_ov = disc["wsi_resonant_neurons"]
    wsi_pair_layers = disc["wsi_pair_layers"]
    wsi_fa3_layers = disc["wsi_fa3_layers"]
    wsi_resonance_table = disc["wsi_resonance_table"]
    print("Top layers (MLP only):", mlp_ranked[:6], flush=True)
    print("Top layers (Hybrid MLP+head+attn):", hybrid_ranked[:6], flush=True)
    print(f"FA layers={fa_layers} | WSI origins={fa_origins} -> resonant dst={l_res_dst} | pair={wsi_pair_layers} | fa3={wsi_fa3_layers}", flush=True)

    # 4. Blind exam against official p50c patch
    p50c = MoIPatch.load(args.patch)
    exam = {}
    for e in p50c.edits:
        m = re.search(r"layers\.(\d+)\.", e.module)
        if not m:
            continue
        li = int(m.group(1))
        if li in u_mlp_by_layer and e.idx is not None:
            cos_u = float(torch.nn.functional.cosine_similarity(u_mlp_by_layer[li].float().unsqueeze(0), e.u.float().unsqueeze(0)))
            ov = len(set(top64_by_layer[li].tolist()) & set(e.idx.long().tolist()))
            exam[f"layer_{li}"] = {"dir_cosine": round(abs(cos_u), 4), "neuron_overlap_of_64": ov}
            if li in wsi_dir_ov:
                cos_r = float(torch.nn.functional.cosine_similarity(wsi_dir_ov[li].float().unsqueeze(0), e.u.float().unsqueeze(0)))
                ov_r = len(set(wsi_idx_ov[li].tolist()) & set(e.idx.long().tolist()))
                exam[f"layer_{li}_wsi_resonant"] = {"dir_cosine": round(abs(cos_r), 4), "neuron_overlap_of_64": ov_r}
    print("Blind exam vs p50c:", exam, flush=True)

    # Causal diff_compact on one looping sequence (base vs p50c)
    loop_patch_diff = None
    if loop_rows:
        seq0, ct0, ons0 = next((t for t in traces if t[0]["loop"]), traces[0])
        t_ons = min(seq0["prompt_len"] + seq0["onset_resp"], len(seq0["full_ids"]) - 1)
        hp = p50c.attach(model)
        hp.on()
        ct_p50c = capture_compact(model, seq0["full_ids"])
        hp.off()
        loop_patch_diff = diff_compact(ct0, ct_p50c, model=model, pos_a=t_ons, pos_b=t_ons, top_k=6)

    # 5. Live held-out eval comparing: base vs ct_mlp_top3 vs ct_mlp_fa3 vs ct_wsi_pair vs ct_wsi_fa3 vs p50c
    def _eval_arm(name: str, patch_obj: MoIPatch | None) -> dict:
        hp = patch_obj.attach(model) if patch_obj is not None else None
        if hp is not None:
            hp.on()
        try:
            t_ev = time.perf_counter()
            ev = _batched_generate(model, tok, list(gte["question"]), dev, args.max_new_tokens)
            ev_s = round(time.perf_counter() - t_ev, 1)
        finally:
            if hp is not None:
                hp.off()
        acc = sum(_is_correct(r["text"], _extract_gold(a)) for r, a in zip(ev, gte["answer"])) / len(ev)
        any_acc = sum(_has_gold_anywhere(r["text"], _extract_gold(a)) for r, a in zip(ev, gte["answer"])) / len(ev)
        lrate = sum(r["loop"] for r in ev) / len(ev)
        mean_tok = sum(r["gen_len"] for r in ev) / len(ev)
        row = {
            "acc": round(acc, 4),
            "any_num_acc": round(any_acc, 4),
            "loop_rate": round(lrate, 4),
            "mean_gen_tok": round(mean_tok, 1),
            "s": ev_s,
        }
        print(f"Eval [{name:<16}] -> {row}", flush=True)
        return row

    eval_res = {"base": _eval_arm("base", None)}
    for arm_name, key in (
        ("ct_mlp_top3", "mlp_top3"),
        ("ct_mlp_fa3", "mlp_fa3"),
        ("ct_wsi_pair", "wsi_pair"),
        ("ct_wsi_fa3", "wsi_fa3"),
    ):
        if key in disc["patches"]:
            eval_res[arm_name] = _eval_arm(arm_name, disc["patches"][key])
    eval_res["p50c_ref"] = _eval_arm("p50c_ref", p50c)

    out = {
        "model": args.model,
        "n_train": len(tr_rows),
        "n_eval": len(gte),
        "max_new_tokens": args.max_new_tokens,
        "train_gen_s": train_gen_s,
        "train_loop_count": len(loop_rows),
        "train_clean_count": len(clean_rows),
        "traces_captured": len(traces),
        "trace_bank_mb": bank_mb,
        "capture_s": cap_s,
        "fa_layers": fa_layers,
        "mlp_top_layers": mlp_ranked[:6],
        "hybrid_top_layers": hybrid_ranked[:6],
        "wsi_origins": fa_origins,
        "wsi_resonant_dst": l_res_dst,
        "wsi_pair_layers": wsi_pair_layers,
        "wsi_fa3_layers": wsi_fa3_layers,
        "wsi_resonance_table": wsi_resonance_table[:16],
        "layer_stats": layer_stats,
        "blind_exam_vs_p50c": exam,
        "causal_diff_base_vs_p50c_at_onset": loop_patch_diff,
        "eval_heldout": eval_res,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
