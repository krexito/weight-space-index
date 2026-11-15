"""Large-Scale Surgical Editing & Collateral Damage Benchmark (`wsr write`) on Real GPT-2.

Evaluates `wsi.rewire_transduction()` across 5 distinct knowledge domains (Technology, Geography,
Literature, Computer Science, and World Landmarks) on pretrained GPT-2 (124M parameters) with zero
gradients or training:
1. **Individual Surgical Precision**: Does each 8-neuron surgical edit (0.02% of MLP neurons) flip the
   targeted prompt's #1 next-token prediction while keeping all 15 held-out guard probes 100% intact?
2. **Simultaneous Multi-Edit Composition**: When all 5 edits (40/36,864 neurons = 0.11%) are composed
   simultaneously into the SAME checkpoint, how many target facts flip and how many guard probes hold?
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "wsr") not in sys.path:
    sys.path.insert(0, str(ROOT / "wsr"))

from wsi import WeightSpaceIndex


def top1_token(model, tokenizer, prompt: str) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        val, idx = torch.topk(probs, k=1)
    return tokenizer.decode([int(idx[0].item())]), float(val[0].item())


def main():
    model_dir = ROOT / "models" / "gpt2"
    wsi = WeightSpaceIndex.from_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.eval()

    # 5 distinct domain surgical rewires
    edits = [
        {
            "domain": "Technology",
            "name": "Bill Gates -> Bill Jobs",
            "trigger": "ĠBill",
            "old_tok": "ĠGates",
            "new_tok": "ĠJobs",
            "prompt": "Microsoft was founded by Bill",
            "expected_new": " Jobs",
            "strength": 1.6,
        },
        {
            "domain": "Geography",
            "name": "San Francisco -> San Diego",
            "trigger": "ĠSan",
            "old_tok": "ĠFrancisco",
            "new_tok": "ĠDiego",
            "prompt": "The Golden Gate Bridge is in San",
            "expected_new": " Diego",
            "strength": 1.6,
        },
        {
            "domain": "Literature",
            "name": "Romeo & Juliet -> Caesar",
            "trigger": "ĠRomeo",
            "old_tok": "ĠJuliet",
            "new_tok": "ĠCaesar",
            "prompt": "William Shakespeare wrote Romeo and",
            "expected_new": " Caesar",
            "strength": 2.4,
        },
        {
            "domain": "CS / AI",
            "name": "Artificial Intelligence -> Gravity",
            "trigger": "Ġartificial",
            "old_tok": "Ġintelligence",
            "new_tok": "Ġgravity",
            "prompt": "In computer science, artificial",
            "expected_new": " gravity",
            "strength": 1.6,
        },
        {
            "domain": "Landmarks",
            "name": "Eiffel Tower: Paris -> Rome",
            "trigger": "ĠTower",
            "old_tok": "ĠParis",
            "new_tok": "ĠRome",
            "prompt": "The Eiffel Tower is in the city of",
            "expected_new": " Rome",
            "strength": 2.4,
        },
    ]

    # 15 Held-out Guard Prompts (General Knowledge, Syntax, Common Sense, Sequences)
    guard_prompts = [
        "The Earth revolves around the",
        "Water freezes at zero degrees",
        "One, two, three, four,",
        "Monday, Tuesday, Wednesday, Thursday,",
        "January, February, March, April,",
        "The opposite of hot is",
        "A dog barks, and a cat",
        "The President of the United",
        "She opened the book and began to",
        "The color of the clear sky is",
        "Two plus two equals",
        "Albert Einstein developed the theory of",
        "The chemical symbol for oxygen is",
        "The capital of France is",
        "He walked into the room and sat",
    ]

    # Record baseline predictions before any edits
    base_targets = {e["name"]: top1_token(model, tokenizer, e["prompt"]) for e in edits}
    base_guards = {p: top1_token(model, tokenizer, p) for p in guard_prompts}

    print("=" * 100)
    print("SURGICAL WEIGHT PROGRAMMING & COLLATERAL DAMAGE BENCHMARK ON REAL GPT-2 (5 DOMAINS)")
    print("=" * 100)

    state = model.state_dict()
    total_synth_ms = 0.0
    single_flips = 0
    total_guard_preserved = 0
    total_guard_checks = 0
    synthesized_edits = []

    for e in edits:
        t0 = time.perf_counter()
        res = wsi.rewire_transduction(
            e["trigger"],
            e["old_tok"],
            e["new_tok"],
            top_k_neurons=8,
            strength=e["strength"],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_synth_ms += elapsed_ms
        synthesized_edits.append((e, res))

        # Apply this single surgical edit
        for tname, T_new in res["_tensors"].items():
            state[f"transformer.{tname}"].copy_(T_new)

        old_tok, old_p = base_targets[e["name"]]
        new_tok, new_p = top1_token(model, tokenizer, e["prompt"])
        flipped = new_tok.strip() == e["expected_new"].strip()
        single_flips += int(flipped)

        # Check all 15 guard prompts for collateral damage
        g_ok = 0
        for gp in guard_prompts:
            gtok, _ = top1_token(model, tokenizer, gp)
            if gtok == base_guards[gp][0]:
                g_ok += 1
        total_guard_preserved += g_ok
        total_guard_checks += len(guard_prompts)

        # Restore base weights before testing next edit
        for tname in res["_tensors"]:
            state[f"transformer.{tname}"].copy_(wsi.ck.get(tname))

        status = "PASS" if flipped else "FAIL"
        print(
            f"[{status}] ({elapsed_ms:5.1f} ms) {e['domain']:11s} | {e['name']:32s} | Guard Probes: {g_ok}/15 ({g_ok/15*100:.0f}%)\n"
            f"       Prompt: \"{e['prompt']}\"\n"
            f"       Before: {old_tok!r:15s} ({old_p*100:5.1f}%)  -->  After: {new_tok!r:10s} ({new_p*100:5.1f}%)"
        )

    # Now compose all 5 edits simultaneously into the same checkpoint
    for e, res in synthesized_edits:
        for tname, T_new in res["_tensors"].items():
            T_base = wsi.ck.get(tname)
            delta = T_new.to(T_base.dtype) - T_base
            state[f"transformer.{tname}"].add_(delta)

    multi_flips = sum(
        int(top1_token(model, tokenizer, e["prompt"])[0].strip() == e["expected_new"].strip())
        for e in edits
    )
    multi_guards = sum(
        int(top1_token(model, tokenizer, gp)[0] == base_guards[gp][0])
        for gp in guard_prompts
    )

    print("-" * 100)
    print(
        f"SUMMARY:\n"
        f"  * Total Synthesis Time (5 Edits)   : {total_synth_ms:.1f} ms ({total_synth_ms/len(edits):.1f} ms/edit)\n"
        f"  * Neurons Modified per Edit        : 8 / 36,864 MLP neurons (0.022% of model)\n"
        f"  * Single-Edit Target Flip Rate     : {single_flips}/{len(edits)} ({single_flips/len(edits)*100:.1f}%)\n"
        f"  * Single-Edit Guard Preservation   : {total_guard_preserved}/{total_guard_checks} ({total_guard_preserved/total_guard_checks*100:.1f}%)\n"
        f"  * Simultaneous 5-Edit Flip Rate    : {multi_flips}/{len(edits)} ({multi_flips/len(edits)*100:.1f}%)\n"
        f"  * Simultaneous 5-Edit Guard Check  : {multi_guards}/{len(guard_prompts)} ({multi_guards/len(guard_prompts)*100:.1f}%)"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
