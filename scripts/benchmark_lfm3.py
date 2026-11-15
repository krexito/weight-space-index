"""Out-of-Distribution Architecture Stress Test: Blind Reading & Indexing on `LiquidAI/LFM3-1.7B`.

Tests `WeightSpaceIndex` on real pretrained `BF16` tensors from `LiquidAI/LFM3-1.7B` (`model_type="lfm3"`),
a hybrid Liquid Foundation Model combining:
- Gated Short Convolutions (`conv.in_proj`, depthwise `conv.conv`, `conv.out_proj`)
- Grouped-Query Attention with QK-RMSNorm & Output Gate (`self_attn.q/k/v/o/gate_proj`)
- 7,680-unit SwiGLU MLPs named `feed_forward.w1` (gate), `feed_forward.w3` (up), `feed_forward.w2` (down)
- Sandwich RMSNorm (`operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm`, and `embedding_norm`)
- 2,048 hidden dimension & BPE vocabulary (`Ġ` prefix tokens)

Runs the EXACT SAME blind diagnosis classifier (`classify_blind_diagnosis`) used on GPT-2 across 20 mystery
`.safetensors` edits spanning all 8 edit families—without modifying a single rule in the classifier.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "wsr") not in sys.path:
    sys.path.insert(0, str(ROOT / "wsr"))

from scripts.benchmark_blind_reading import classify_blind_diagnosis, tok_id
from wsi import WeightSpaceIndex
from wsr.ops import apply_ops


def main():
    model_dir = ROOT / "models" / "lfm3-1.7b"
    t_open = time.perf_counter()
    wsi = WeightSpaceIndex.from_model(model_dir)
    open_ms = (time.perf_counter() - t_open) * 1000.0
    ck = wsi.ck
    anat = wsi.anat

    print("=" * 100)
    print(f"OUT-OF-DISTRIBUTION ARCHITECTURE BENCHMARK: LiquidAI/LFM3-1.7B (Opened in {open_ms:.1f} ms)")
    print("=" * 100)
    print("1. Auto-Detected Anatomy (`lfm3` Hybrid ShortConv + GQA + SwiGLU + Sandwich RMSNorm):")
    print(json.dumps(anat.summary(), indent=2))

    # 2. Test real neuron transduction discovery on LFM3-1.7B's 7,680-unit SwiGLU MLP
    trans = wsi.search_transduction("ĠFrance", "ĠParis", top_k=3)
    top_n = trans["top_transducing_neurons"][0]
    n_info = wsi.explain_neuron(top_n["layer"], top_n["unit"], top_k=5)
    print("-" * 100)
    print(
        f"2. Real Pretrained LFM3-1.7B SwiGLU Neuron Discovery ([France] -> [Paris]):\n"
        f"   Top Neuron: Layer {top_n['layer']} Unit {top_n['unit']} "
        f"(in={top_n['in_tensor']}, out={top_n['out_tensor']})\n"
        f"   Transduction Rule: {n_info['transduction_rule']}"
    )
    print("-" * 100)

    # 3. Generate 20 Mystery Edits across all 8 families on LFM3-1.7B (including `conv.out_proj` & `feed_forward.w2`)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        mysteries = []
        t_write_start = time.perf_counter()

        # Family 1: 4 Surgical Neuron Rewires on LFM3's 7,680-unit `feed_forward.w2` SwiGLU MLPs
        rewire_specs = [
            ("ĠBill", "ĠParis", "ĠRome"),
            ("ĠGermany", "ĠBerlin", "ĠMadrid"),
            ("ĠFrance", "ĠParis", "ĠLondon"),
            ("ĠJapan", "ĠLondon", "ĠBerlin"),
        ]
        for trig, old_t, new_t in rewire_specs:
            mid = f"lfm3_mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            wsi.rewire_transduction(trig, old_t, new_t, top_k_neurons=8, strength=5.0, save_path=fpath)
            mysteries.append((mid, fpath, "surgical_neuron_rewire", new_t))

        # Family 2: 4 Orthogonal Subspace Concept Erasures (`orth`) across SwiGLU `w2`, `conv.out_proj`, & `self_attn.o_proj`
        orth_specs = [
            ("model.layers.0.feed_forward.w2.weight", "ĠParis"),
            ("model.layers.1.feed_forward.w2.weight", "ĠLondon"),
            ("model.layers.0.conv.out_proj.weight", "ĠBerlin"),
            ("model.layers.1.self_attn.o_proj.weight", "ĠRome"),
        ]
        for tensor_name, tok_str in orth_specs:
            mid = f"lfm3_mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            tid = tok_id(wsi, tok_str)
            _, comp, _ = apply_ops(ck, {"ops": [{"op": "orth", "tensors": [tensor_name], "u_rows": [tid]}]}, anat)
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "orth_concept_erasure", tok_str))

        # Family 3: 3 Vocabulary Row Boosts (`scale_rows` > 1.0) on `model.embed_tokens.weight` ([16384, 2048] BF16)
        for tok_str, factor in [("ĠRome", 2.5), ("ĠMadrid", 3.0), ("ĠFrance", 2.2)]:
            mid = f"lfm3_mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            tid = tok_id(wsi, tok_str)
            _, comp, _ = apply_ops(
                ck,
                {"ops": [{"op": "scale_rows", "tensor": "model.embed_tokens.weight", "rows": [tid], "factor": factor}]},
                anat,
            )
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "vocab_row_boosted", tok_str))

        # Family 4: 3 Vocabulary Row Suppressions (`scale_rows` < 1.0)
        for tok_str, factor in [("ĠParis", 0.15), ("ĠLondon", 0.2), ("ĠBerlin", 0.25)]:
            mid = f"lfm3_mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            tid = tok_id(wsi, tok_str)
            _, comp, _ = apply_ops(
                ck,
                {"ops": [{"op": "scale_rows", "tensor": "model.embed_tokens.weight", "rows": [tid], "factor": factor}]},
                anat,
            )
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "vocab_row_suppressed", tok_str))

        # Family 5: 2 Block Ablations (`scale` = 0.0) on ShortConv `conv.out_proj` & SwiGLU `feed_forward.w2`
        for tensor_name in ["model.layers.0.conv.out_proj.weight", "model.layers.1.feed_forward.w2.weight"]:
            mid = f"lfm3_mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            _, comp, _ = apply_ops(ck, {"ops": [{"op": "scale", "tensor": tensor_name, "factor": 0.0}]}, anat)
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "block_ablation", tensor_name))

        # Family 6: 2 Sandwich RMSNorm Gain Scalings (`gain`) on `operator_norm` and `post_ffn_norm`
        for tensor_name, factor in [
            ("model.layers.0.post_ffn_norm.weight", 2.0),
            ("model.layers.1.operator_norm.weight", 0.5),
        ]:
            mid = f"lfm3_mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            _, comp, _ = apply_ops(ck, {"ops": [{"op": "gain", "tensors": [tensor_name], "factor": factor}]}, anat)
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "norm_gain_scale", tensor_name))

        # Family 7 & 8: 1 Isotropic Weight Noise (`noise`) + 1 Imperceptible No-Op Control (`1.0001x`)
        mid_n = f"lfm3_mystery_{len(mysteries)+1:02d}"
        fpath_n = tmp_dir / f"{mid_n}.safetensors"
        _, comp_n, _ = apply_ops(
            ck, {"ops": [{"op": "noise", "tensor": "model.layers.0.conv.out_proj.weight", "rel": 0.15, "seed": 42}]}, anat
        )
        save_file(comp_n, str(fpath_n))
        mysteries.append((mid_n, fpath_n, "isotropic_weight_noise", "model.layers.0.conv.out_proj.weight"))

        mid_c = f"lfm3_mystery_{len(mysteries)+1:02d}"
        fpath_c = tmp_dir / f"{mid_c}.safetensors"
        _, comp_c, _ = apply_ops(
            ck, {"ops": [{"op": "scale", "tensor": "model.layers.1.self_attn.o_proj.weight", "factor": 1.00001}]}, anat
        )
        save_file(comp_c, str(fpath_c))
        mysteries.append((mid_c, fpath_c, "no_op_control", "none"))

        write_total_ms = (time.perf_counter() - t_write_start) * 1000.0
        print(f"3. Generated {len(mysteries)} LFM3-1.7B mystery edits in {write_total_ms:.1f} ms ({write_total_ms/len(mysteries):.1f} ms/edit)")
        print("-" * 100)

        correct_cat = 0
        correct_concept = 0
        read_total_ms = 0.0

        for mid, fpath, expected_cat, expected_concept in mysteries:
            t0 = time.perf_counter()
            diag = wsi.explain_edit(fpath)
            pred = classify_blind_diagnosis(diag)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            read_total_ms += elapsed_ms

            cat_ok = pred["predicted_category"] == expected_cat
            concept_ok = pred["predicted_concept"] == expected_concept
            correct_cat += int(cat_ok)
            correct_concept += int(concept_ok)

            status = "PASS" if (cat_ok and concept_ok) else "FAIL"
            print(
                f"[{status}] {mid} ({elapsed_ms:6.1f} ms) | "
                f"Expected: {expected_cat:23s} ({expected_concept:36s}) | "
                f"Predicted: {pred['predicted_category']:23s} ({pred['predicted_concept']:36s})"
            )

        n = len(mysteries)
        print("=" * 100)
        print(
            f"LFM3-1.7B BLIND READING SUMMARY ({n} EDITS):\n"
            f"  * Category Accuracy           : {correct_cat}/{n} ({correct_cat/n*100:.1f}%)\n"
            f"  * Exact Concept/Token Accuracy: {correct_concept}/{n} ({correct_concept/n*100:.1f}%)\n"
            f"  * Total Creation Time (write) : {write_total_ms:.1f} ms ({write_total_ms/n:.1f} ms/edit)\n"
            f"  * Total Blind Read Time (read): {read_total_ms:.1f} ms ({read_total_ms/n:.1f} ms/edit)"
        )
        print("=" * 100)


if __name__ == "__main__":
    main()
