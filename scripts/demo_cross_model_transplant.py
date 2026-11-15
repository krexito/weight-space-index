"""Live Demonstration & Benchmark of Cross-Architecture Weight-Space Transplant (`wsi`).

Reads real latent neuron vectors and circuits from larger modern architectures:
  1. `LiquidAI/LFM3-1.7B` (1.7B params, d_model=2048, SwiGLU 7680d, vocab=65,536)
  2. `Qwen/Qwen3.5-0.8B`  (0.8B params, d_model=1024, SwiGLU 3584d, vocab=248,320)

And translates/transplants them directly into a smaller architecture:
  -> `GPT-2 124M`         (124M params, d_model=768,  Conv1D 3072d, vocab=50,257)

via the Dual-Gram Vocabulary Rosetta Bridge (`translate_vector_from` & `transplant_circuit_from`),
with ZERO training data, ZERO gradient steps, and verified on real `GPT2LMHeadModel` forward passes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wsr"))

from wsi import WeightSpaceIndex
from wsi.compare import _get_shared_vocab_rosetta


def evaluate_rosetta_fidelity(
    teacher_name: str,
    teacher_idx: WeightSpaceIndex,
    teacher_tensor_name: str,
    student_idx: WeightSpaceIndex,
    num_neurons: int = 50,
) -> dict:
    """Translate `num_neurons` real MLP columns from `teacher_idx` (d_teacher) into `student_idx` (d_student)."""
    # Warm up shared vocabulary Rosetta anchor tables once
    _get_shared_vocab_rosetta(teacher_idx, student_idx)

    W_t = teacher_idx.ck.get(teacher_tensor_name).float()
    if teacher_idx.anat.out_axis(teacher_tensor_name) == 1:
        W_t = W_t.t()

    col_norms = W_t.norm(dim=0)
    top_units = torch.topk(col_norms, k=num_neurons).indices.tolist()

    t0 = time.perf_counter()
    top1_matches = 0
    top5_overlaps = []
    examples = []

    for uid in top_units:
        v_t = W_t[:, uid]
        res = student_idx.translate_vector_from(teacher_idx, v_t, top_m=16)
        if res["top1_exact_match"]:
            top1_matches += 1
        top5_overlaps.append(res["top5_chord_overlap"])
        if len(examples) < 3:
            examples.append({
                "unit": uid,
                "teacher_chord": res["teacher_top_tokens"],
                "student_chord": res["student_translated_top_tokens"],
                "overlap": res["top5_chord_overlap"],
            })

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "teacher": teacher_name,
        "teacher_dim": teacher_idx.anat.hidden,
        "student_dim": student_idx.anat.hidden,
        "shared_vocab": res["shared_vocab_size"],
        "num_neurons": num_neurons,
        "elapsed_ms": elapsed_ms,
        "ms_per_neuron": elapsed_ms / num_neurons,
        "top1_match_rate": top1_matches / num_neurons * 100.0,
        "mean_top5_chord_overlap": sum(top5_overlaps) / len(top5_overlaps) * 100.0,
        "examples": examples,
    }


def main() -> None:
    print("=" * 95)
    print(" CROSS-ARCHITECTURE WEIGHT-SPACE TRANSPLANT: BIG MODELS -> SMALL MODEL (ZERO TRAINING)")
    print("=" * 95)

    qwen_idx = WeightSpaceIndex.from_model(ROOT / "models" / "qwen35-0.8b")
    lfm3_idx = WeightSpaceIndex.from_model(ROOT / "models" / "lfm3-1.7b")
    gpt2_idx = WeightSpaceIndex.from_model(ROOT / "models" / "gpt2")

    # Warm up vocabulary matrices
    for idx in (qwen_idx, lfm3_idx, gpt2_idx):
        _ = idx.U_eff
        _ = idx.E_in

    # -------------------------------------------------------------------------
    # STEP 1: 100-Neuron Cross-Architecture Rosetta Translation Benchmark
    # -------------------------------------------------------------------------
    print("\n[1/2] Dual-Gram Rosetta Vector Translation Benchmark (100 Real Pretrained Neurons)...")
    qwen_stats = evaluate_rosetta_fidelity(
        "Qwen/Qwen3.5-0.8B (d=1024 -> GPT-2 d=768)",
        qwen_idx,
        "model.layers.19.mlp.down_proj.weight",
        gpt2_idx,
        num_neurons=50,
    )
    lfm3_stats = evaluate_rosetta_fidelity(
        "LiquidAI/LFM3-1.7B (d=2048 -> GPT-2 d=768)",
        lfm3_idx,
        "model.layers.0.feed_forward.w2.weight",
        gpt2_idx,
        num_neurons=50,
    )

    for st in (qwen_stats, lfm3_stats):
        print(
            f"\n  * {st['teacher']} (Shared BPE Anchors: {st['shared_vocab']:,}):\n"
            f"    - Translated {st['num_neurons']} neurons ({st['teacher_dim']}d -> {st['student_dim']}d) "
            f"in {st['elapsed_ms']:.1f} ms ({st['ms_per_neuron']:.2f} ms/neuron)\n"
            f"    - Top-1 Global Vocab Match: {st['top1_match_rate']:.1f}% | "
            f"Top-5 Multi-Token Chord Overlap: {st['mean_top5_chord_overlap']:.1f}%"
        )
        for ex in st["examples"]:
            print(
                f"      [Neuron {ex['unit']:4d}] Teacher ({st['teacher_dim']}d): {ex['teacher_chord']} "
                f"--> Student (768d): {ex['student_chord']}"
            )

    # -------------------------------------------------------------------------
    # STEP 2: Live Brain Transplant from Qwen3.5-0.8B (d=1024) into GPT-2 (d=768)
    # -------------------------------------------------------------------------
    print("\n[2/2] Live Surgical Circuit Transplant: Qwen/Qwen3.5-0.8B (1024d) -> GPT-2 124M (768d)...")

    # Find the Qwen3.5-0.8B Layer 19 neuron whose #1 shared semantic token is " Madrid" ("ĠMadrid")
    _, _, shared_tokens, U_q_norm, _ = _get_shared_vocab_rosetta(qwen_idx, gpt2_idx)
    madrid_shared_idx = shared_tokens.index("ĠMadrid")
    W_q19 = qwen_idx.ck.get("model.layers.19.mlp.down_proj.weight").float()
    W_q19_norm = W_q19 / W_q19.norm(dim=0).clamp_min(1e-12)
    Z_all = U_q_norm @ W_q19_norm  # [|V_shared|, 3584]
    top1_per_unit = Z_all.argmax(dim=0)
    madrid_top1_units = torch.nonzero(top1_per_unit == madrid_shared_idx).flatten()
    best_qwen_madrid_unit = int(madrid_top1_units[Z_all[madrid_shared_idx, madrid_top1_units].argmax()].item())

    t_tr0 = time.perf_counter()
    transplant_res = gpt2_idx.transplant_circuit_from(
        qwen_idx,
        teacher_layer=19,
        teacher_unit=best_qwen_madrid_unit,
        trigger_query="ĠTower",
        old_output_query="ĠParis",
        top_k_student_neurons=24,
        top_m_chord=16,
        strength=6.0,
    )
    t_tr_ms = (time.perf_counter() - t_tr0) * 1000.0

    print(
        f"  Transplanted Qwen3.5-0.8B Layer 19 Neuron #{best_qwen_madrid_unit} (1024d) -> "
        f"24 GPT-2 Neurons (768d) in {t_tr_ms:.2f} ms!"
    )
    print(f"  - Teacher 1024d Semantic Chord : {transplant_res['teacher_source']['top_semantic_chord']}")
    print(f"  - Student 768d Translated Chord: {transplant_res['rosetta_translation']['student_translated_chord']}")
    print(
        f"  - Multi-Token Chord Overlap    : {transplant_res['rosetta_translation']['top5_chord_overlap']*100:.1f}%"
    )

    # Verify on real GPT-2 forward passes before and after the transplant
    tokenizer = GPT2Tokenizer.from_pretrained(str(ROOT / "models" / "gpt2"))
    model = GPT2LMHeadModel.from_pretrained(str(ROOT / "models" / "gpt2"))
    model.eval()

    prompt = "The Eiffel Tower is located in the city of"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits_before = model(**inputs).logits[0, -1, :]
        probs_before = torch.softmax(logits_before, dim=-1)
        top5_before = [
            (tokenizer.decode([idx]), float(probs_before[idx]) * 100.0)
            for idx in torch.topk(probs_before, k=5).indices.tolist()
        ]

    # Apply the transplanted weights from Qwen3.5-0.8B into the live GPT-2 model
    sd = model.state_dict()
    for tname, T_mod in transplant_res["_tensors"].items():
        key = tname if tname in sd else f"transformer.{tname}"
        sd[key].copy_(T_mod)

    with torch.no_grad():
        logits_after = model(**inputs).logits[0, -1, :]
        probs_after = torch.softmax(logits_after, dim=-1)
        top5_after = [
            (tokenizer.decode([idx]), float(probs_after[idx]) * 100.0)
            for idx in torch.topk(probs_after, k=5).indices.tolist()
        ]

    print(f"\n  Live GPT-2 Forward Pass Verification on: {prompt!r}")
    print("  - BEFORE Transplant (Base GPT-2)  :", [f"{tok!r} ({p:.1f}%)" for tok, p in top5_before])
    print("  - AFTER  Transplant (From Qwen3.5):", [f"{tok!r} ({p:.1f}%)" for tok, p in top5_after])


if __name__ == "__main__":
    main()
