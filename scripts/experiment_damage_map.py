"""Damage map of a .moipatch across text domains (wsi/damage_map.py): where does an always-on patch hurt?

Domains (each: --n-docs sequences cut to --seq-len tokens, teacher forcing, no generation):
  * own_thinking: the model's own healthy reasoning (non-looping base answers of a finished run, --work), the text a
                  loop patch must leave alone
  * math:         GSM8K train question + worked solution
  * prose:        WikiText-2 test
  * code:         HumanEval prompts + canonical solutions
  * chat:         UltraChat test conversations, rendered with the model's chat template
  * spanish / chinese: OPUS-100 test sentences (several joined per sequence)

Each patch is compared with a random-direction control of the same size (wsi.damage_map.random_like). Data files
come from the Hugging Face Hub (or --data-dir if already there).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from wsi.damage_map import damage_map, random_like  # noqa: E402
from wsi.moipatch import MoIPatch  # noqa: E402

# domain -> (repo, file prefix inside the dataset repo)
HUB = {
    "math": ("openai/gsm8k", "main/train-"),
    "prose": ("Salesforce/wikitext", "wikitext-2-raw-v1/test-"),
    "code": ("openai/openai_humaneval", "openai_humaneval/test-"),
    "chat": ("HuggingFaceH4/ultrachat_200k", "data/test_sft-"),
    "spanish": ("Helsinki-NLP/opus-100", "en-es/test-"),
    "chinese": ("Helsinki-NLP/opus-100", "en-zh/test-"),
}


def hub_parquet(repo: str, prefix: str, data_dir: Path):
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"{repo}: no parquet file starts with {prefix}")
    return pd.read_parquet(hf_hub_download(repo, files[0], repo_type="dataset", local_dir=data_dir / repo.replace("/", "__")))


def texts_for(domain: str, n: int, data_dir: Path, tok) -> list[str]:
    df = hub_parquet(*HUB[domain], data_dir)
    if domain == "math":
        return [f"{q}\n{a}" for q, a in zip(df["question"][:n], df["answer"][:n])]
    if domain == "prose":
        text = "\n".join(df["text"])
        step = max(1, len(text) // n)
        return [text[i * step:(i + 1) * step] for i in range(n)]
    if domain == "code":
        return [p + s for p, s in zip(df["prompt"][:n], df["canonical_solution"][:n])]
    if domain == "chat":
        out = []
        for msgs in df["messages"][:n]:
            msgs = [dict(m) for m in msgs]
            try:
                out.append(tok.apply_chat_template(msgs, tokenize=False))
            except Exception:
                out.append("\n".join(f"{m['role']}: {m['content']}" for m in msgs))
        return out
    if domain in ("spanish", "chinese"):
        lang = "es" if domain == "spanish" else "zh"
        sents = [t[lang] for t in df["translation"]]
        k = 12
        return [" ".join(sents[i * k:(i + 1) * k]) for i in range(n)]
    raise KeyError(domain)


def own_thinking(work: Path, n: int, seq_len: int) -> list[list[int]]:
    """Healthy reasoning tokens: non-looping base answers, or the part well before the loop of looping ones."""
    import moi_loop_router_vllm as lr

    cfg = json.loads((work / "config.json").read_text())
    eos = set(cfg["eos"])
    reqs = {r["id"]: r for r in lr.read_jsonl(work / "requests_test.jsonl")}
    seqs = []
    for o in lr.read_jsonl(work / "outputs_test_base.jsonl"):
        st = lr.analyse(o["token_ids"], eos)
        end = len(st["body"]) if st["kill"] is None else max(0, st["kill"] - lr.KILL_WINDOW - 256)
        body = st["body"][:end]
        if len(body) < seq_len // 2:
            continue
        p = reqs[o["id"]]["prompt_ids"]
        seqs.append((p + body)[:seq_len])
        if len(seqs) >= n:
            break
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patch", action="append", required=True, help="one or more .moipatch files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="finished run dir for the own_thinking domain")
    ap.add_argument("--domains", default="own_thinking,math,prose,code,chat,spanish,chinese")
    ap.add_argument("--n-docs", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--data-dir", default="data_damage_map")
    ap.add_argument("--no-random", action="store_true", help="skip the random-direction control")
    ap.add_argument("--no-fingerprint", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    except Exception as e:  # toy test models have no tokenizer: only own_thinking (token ids) works then
        print("no tokenizer:", type(e).__name__, flush=True)
        tok = None
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), trust_remote_code=True).to(a.device).eval()
    data_dir = Path(a.data_dir)
    domains, skipped = {}, {}
    for d in [x for x in a.domains.split(",") if x]:
        try:
            if d == "own_thinking":
                if not a.work:
                    raise ValueError("needs --work")
                seqs = own_thinking(Path(a.work), a.n_docs, a.seq_len)
            else:
                if tok is None:
                    raise ValueError("needs a tokenizer")
                seqs = [tok(t, add_special_tokens=False).input_ids[: a.seq_len] for t in texts_for(d, a.n_docs, data_dir, tok)]
            seqs = [s for s in seqs if len(s) >= 16]
            if not seqs:
                raise ValueError("no usable sequences")
            domains[d] = seqs
            print(f"{d:13s} {len(seqs)} sequences, {sum(map(len, seqs))} tokens", flush=True)
        except Exception as e:  # a missing dataset should not sink the whole map
            skipped[d] = f"{type(e).__name__}: {e}"
            print(f"{d:13s} skipped: {skipped[d]}", flush=True)

    patches = {}
    for p in a.patch:
        patch = MoIPatch.load(p)
        name = Path(p).stem
        patches[name] = patch
        if not a.no_random:
            patches[f"{name}__random"] = random_like(patch, seed=a.seed)
    res = {"model": a.model, "seq_len": a.seq_len, "n_docs": a.n_docs, "skipped": skipped,
           "patches": {n: {"modules": [e.module for e in p.edits], "alpha": [e.alpha for e in p.edits],
                           "kept_columns": [int(e.idx.numel()) if e.idx is not None else None for e in p.edits]}
                       for n, p in patches.items()}}
    res["map"] = damage_map(model, domains, patches, decode=(lambda ids: tok.decode(ids)) if tok is not None else (lambda ids: str(ids)),
                            verify_fingerprint=not a.no_fingerprint)
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    for name, per in res["map"].items():
        print(f"\n{name}")
        for d, m in per.items():
            print(f"  {d:13s} dNLL {m['dnll']:+.4f} | KL {m['kl']:.4f} (p99 {m['kl_p99']:.3f}) | tokens KL>0.1 {m['hit_share']:.2%}"
                  f" | top-1 same {m['top1_agree']:.2%} | activity {list(m['activity'].values())}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
