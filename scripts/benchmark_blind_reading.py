"""Large-Scale Blind Weight-Space Reading & Editing Benchmark (`wsr read` / `wsr write`) on Real GPT-2.

Generates 50 diverse mystery `.safetensors` edits on pretrained GPT-2 (124M parameters) across 7 edit families:
1. Surgical Neuron Rewires (`rewire_transduction`) — 10 edits
2. Orthogonal Subspace Concept Erasure (`orth`) — 10 edits
3. Vocabulary Unembedding Row Boost (`scale_rows` > 1) — 8 edits
4. Vocabulary Unembedding Row Suppression (`scale_rows` < 1) — 7 edits
5. Layer Block Ablation (`scale` = 0.0) — 5 edits
6. LayerNorm Gain Scaling (`gain`) — 5 edits
7. Isotropic Weight Noise & Imperceptible No-Op Controls (`noise`, `no_op`) — 5 edits

Measures both:
- **Edit Creation Time (`write`)**: Time to locate target neurons/subspaces and write all 50 `.safetensors` edits.
- **Blind Decoding Time (`read`)**: Time for `wsi.explain_edit()` to blindly inspect each `.safetensors` file from disk
  (with zero text descriptions or ground-truth hints) and recover both the exact edit category and target concept/token.
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

from wsi import WeightSpaceIndex
from wsr.ops import apply_ops


def classify_blind_diagnosis(diag: dict) -> dict:
    """Given only the output of `wsi.explain_edit(mystery_file)`, classify what the edit does."""
    rel_norm = diag["validation"]["total_rel_norm"]
    t = diag["tensors"][0]
    role = t.get("role", "")

    # 1. Tiny no-op control (< 0.1% relative norm shift)
    if rel_norm < 0.001:
        return {
            "predicted_category": "no_op_control",
            "predicted_concept": "none",
            "details": f"rel_norm={rel_norm:.6f}",
        }

    # 2. Direct vocabulary row edit (`scale_rows` on wte / lm_head)
    if role == "unembed_or_embed":
        top_tok = t["top_edited_tokens"][0]
        action = "boosted" if top_tok["scale_ratio"] > 1.0 else "suppressed"
        return {
            "predicted_category": f"vocab_row_{action}",
            "predicted_concept": top_tok["token"],
            "details": f"token={top_tok['token']} (id={top_tok['token_id']}), scale={top_tok['scale_ratio']}x",
        }

    # 3. Norm gain scaling (`gain`)
    if role == "norm_gain":
        return {
            "predicted_category": "norm_gain_scale",
            "predicted_concept": t["tensor"],
            "details": f"tensor={t['tensor']}, mean_ratio={t['mean_new_over_old']}x",
        }

    # 4. Residual matrix edit (`mlp_out` / `attn_out`)
    if role == "residual_matrix":
        # Full block ablation: norm_ratio zeroed out
        if t["norm_ratio_new_old"] < 0.05:
            return {
                "predicted_category": "block_ablation",
                "predicted_concept": t["tensor"],
                "details": f"tensor={t['tensor']} zeroed out (ratio={t['norm_ratio_new_old']})",
            }
        # Surgical neuron rewire (few columns/neurons modified)
        if "surgical_neuron_rewire" in t:
            sr = t["surgical_neuron_rewire"]
            new_tok = sr["new_promoted_tokens"][0]
            old_tok = sr["old_promoted_tokens"][0]
            return {
                "predicted_category": "surgical_neuron_rewire",
                "predicted_concept": new_tok,
                "details": f"rewired {t['changed_units_count']} units in {t['tensor']}: {old_tok} -> {new_tok}",
            }
        # Rank-1 subspace projection (`orth`)
        if t["delta_rank1_energy_fraction"] > 0.95 and t["subspace_action"] == "projected_out_or_suppressed":
            top_tok = t["principal_delta_direction"]["promoted_tokens"][0]["token"]
            return {
                "predicted_category": "orth_concept_erasure",
                "predicted_concept": top_tok,
                "details": (
                    f"projected out '{top_tok}' from {t['tensor']} "
                    f"(energy {t['old_energy_along_principal_dir']:.1f} -> {t['new_energy_along_principal_dir']:.3f})"
                ),
            }
        # Isotropic random weight noise (`noise`)
        if t["delta_rank1_energy_fraction"] < 0.08:
            return {
                "predicted_category": "isotropic_weight_noise",
                "predicted_concept": t["tensor"],
                "details": (
                    f"random noise on {t['tensor']} "
                    f"(rel_norm={t['rel_norm']:.3f}, rank1_frac={t['delta_rank1_energy_fraction']:.4f})"
                ),
            }

    return {"predicted_category": "unknown", "predicted_concept": "unknown", "details": ""}


def tok_id(wsi: WeightSpaceIndex, token_str: str) -> int:
    matches = wsi.find_tokens(token_str, limit=1)
    if not matches:
        raise ValueError(f"Token not found: {token_str}")
    return matches[0]["token_id"]


def main():
    model_dir = ROOT / "models" / "gpt2"
    wsi = WeightSpaceIndex.from_model(model_dir)
    ck = wsi.ck
    anat = wsi.anat

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        mysteries = []

        t_write_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Family 1: 10 Surgical Neuron Rewires (`rewire_transduction`)
        # ------------------------------------------------------------------
        rewire_specs = [
            ("ĠTower", "ĠParis", "ĠRome"),
            ("ĠJapan", "ĠTokyo", "ĠCairo"),
            ("ĠGermany", "ĠBerlin", "ĠMadrid"),
            ("ĠEngland", "ĠLondon", "ĠAthens"),
            ("ĠChina", "ĠBeijing", "ĠLisbon"),
            ("ĠRussia", "ĠMoscow", "ĠVienna"),
            ("ĠItaly", "ĠRome", "ĠDublin"),
            ("ĠSpain", "ĠMadrid", "ĠOslo"),
            ("ĠEgypt", "ĠCairo", "ĠWarsaw"),
            ("ĠFrance", "ĠParis", "ĠPrague"),
        ]
        for idx, (trig, old_t, new_t) in enumerate(rewire_specs, 1):
            mid = f"mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            wsi.rewire_transduction(trig, old_t, new_t, top_k_neurons=8, strength=5.0, save_path=fpath)
            mysteries.append((mid, fpath, "surgical_neuron_rewire", new_t))

        # ------------------------------------------------------------------
        # Family 2: 10 Orthogonal Subspace Concept Erasures (`orth`)
        # ------------------------------------------------------------------
        orth_specs = [
            ("h.10.mlp.c_proj.weight", "ĠParis"),
            ("h.9.mlp.c_proj.weight", "ĠLondon"),
            ("h.8.mlp.c_proj.weight", "ĠBerlin"),
            ("h.11.mlp.c_proj.weight", "ĠTokyo"),
            ("h.7.mlp.c_proj.weight", "ĠMoscow"),
            ("h.6.mlp.c_proj.weight", "ĠFrance"),
            ("h.5.mlp.c_proj.weight", "ĠGermany"),
            ("h.10.attn.c_proj.weight", "ĠChina"),
            ("h.9.attn.c_proj.weight", "ĠJapan"),
            ("h.8.attn.c_proj.weight", "ĠItaly"),
        ]
        for tensor_name, tok_str in orth_specs:
            mid = f"mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            tid = tok_id(wsi, tok_str)
            _, comp, _ = apply_ops(ck, {"ops": [{"op": "orth", "tensors": [tensor_name], "u_rows": [tid]}]}, anat)
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "orth_concept_erasure", tok_str))

        # ------------------------------------------------------------------
        # Family 3: 8 Vocabulary Row Boosts (`scale_rows` > 1.0)
        # ------------------------------------------------------------------
        boost_specs = [
            ("ĠRome", 2.5),
            ("ĠTokyo", 3.0),
            ("ĠPython", 2.2),
            ("Ġscience", 2.8),
            ("Ġmusic", 2.4),
            ("Ġwater", 2.6),
            ("Ġgold", 3.2),
            ("Ġspace", 2.0),
        ]
        for tok_str, factor in boost_specs:
            mid = f"mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            tid = tok_id(wsi, tok_str)
            _, comp, _ = apply_ops(
                ck, {"ops": [{"op": "scale_rows", "tensor": "wte.weight", "rows": [tid], "factor": factor}]}, anat
            )
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "vocab_row_boosted", tok_str))

        # ------------------------------------------------------------------
        # Family 4: 7 Vocabulary Row Suppressions (`scale_rows` < 1.0)
        # ------------------------------------------------------------------
        suppress_specs = [
            ("<|endoftext|>", 50256, 0.2),
            ("ĠParis", tok_id(wsi, "ĠParis"), 0.15),
            ("Ġerror", tok_id(wsi, "Ġerror"), 0.25),
            ("Ġbad", tok_id(wsi, "Ġbad"), 0.3),
            ("Ġhate", tok_id(wsi, "Ġhate"), 0.1),
            ("Ġnever", tok_id(wsi, "Ġnever"), 0.2),
            ("Ġwar", tok_id(wsi, "Ġwar"), 0.15),
        ]
        for tok_str, tid, factor in suppress_specs:
            mid = f"mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            _, comp, _ = apply_ops(
                ck, {"ops": [{"op": "scale_rows", "tensor": "wte.weight", "rows": [tid], "factor": factor}]}, anat
            )
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "vocab_row_suppressed", tok_str))

        # ------------------------------------------------------------------
        # Family 5: 5 Full Layer/Module Block Ablations (`scale` = 0.0)
        # ------------------------------------------------------------------
        ablation_tensors = [
            "h.11.mlp.c_proj.weight",
            "h.7.mlp.c_proj.weight",
            "h.3.mlp.c_proj.weight",
            "h.10.attn.c_proj.weight",
            "h.5.attn.c_proj.weight",
        ]
        for tensor_name in ablation_tensors:
            mid = f"mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            _, comp, _ = apply_ops(ck, {"ops": [{"op": "scale", "tensor": tensor_name, "factor": 0.0}]}, anat)
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "block_ablation", tensor_name))

        # ------------------------------------------------------------------
        # Family 6: 5 LayerNorm Gain Scalings (`gain`)
        # ------------------------------------------------------------------
        gain_specs = [
            ("h.6.ln_2.weight", 2.0),
            ("h.0.ln_1.weight", 1.8),
            ("h.11.ln_2.weight", 0.4),
            ("h.4.ln_1.weight", 2.5),
            ("ln_f.weight", 1.5),
        ]
        for tensor_name, factor in gain_specs:
            mid = f"mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            _, comp, _ = apply_ops(ck, {"ops": [{"op": "gain", "tensors": [tensor_name], "factor": factor}]}, anat)
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "norm_gain_scale", tensor_name))

        # ------------------------------------------------------------------
        # Family 7: 3 Isotropic Weight Noise (`noise`) + 2 No-Op Controls
        # ------------------------------------------------------------------
        noise_specs = [
            ("h.8.mlp.c_proj.weight", 0.15, 42),
            ("h.2.mlp.c_proj.weight", 0.20, 99),
            ("h.6.attn.c_proj.weight", 0.12, 7),
        ]
        for tensor_name, rel, seed in noise_specs:
            mid = f"mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            _, comp, _ = apply_ops(
                ck, {"ops": [{"op": "noise", "tensor": tensor_name, "rel": rel, "seed": seed}]}, anat
            )
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "isotropic_weight_noise", tensor_name))

        noop_tensors = ["h.4.mlp.c_proj.weight", "h.1.attn.c_proj.weight"]
        for tensor_name in noop_tensors:
            mid = f"mystery_{len(mysteries)+1:02d}"
            fpath = tmp_dir / f"{mid}.safetensors"
            _, comp, _ = apply_ops(ck, {"ops": [{"op": "scale", "tensor": tensor_name, "factor": 1.0001}]}, anat)
            save_file(comp, str(fpath))
            mysteries.append((mid, fpath, "no_op_control", "none"))

        write_total_ms = (time.perf_counter() - t_write_start) * 1000.0

        print("=" * 100)
        print(f"LARGE-SCALE BLIND WEIGHT-SPACE BENCHMARK ON REAL GPT-2 ({len(mysteries)} MYSTERY EDITS)")
        print("=" * 100)
        print(f"Edit Generation Phase: Created all {len(mysteries)} .safetensors edits in {write_total_ms:.1f} ms ({write_total_ms/len(mysteries):.1f} ms/edit)")
        print("-" * 100)

        correct_cat = 0
        correct_concept = 0
        read_total_ms = 0.0
        rows = []
        family_stats: dict[str, dict[str, float | int]] = {}

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

            st = family_stats.setdefault(expected_cat, {"total": 0, "cat_ok": 0, "concept_ok": 0, "time_ms": 0.0})
            st["total"] += 1
            st["cat_ok"] += int(cat_ok)
            st["concept_ok"] += int(concept_ok)
            st["time_ms"] += elapsed_ms

            status = "PASS" if (cat_ok and concept_ok) else "FAIL"
            print(
                f"[{status}] {mid} ({elapsed_ms:6.1f} ms) | "
                f"Expected: {expected_cat:23s} ({expected_concept:22s}) | "
                f"Predicted: {pred['predicted_category']:23s} ({pred['predicted_concept']:22s})"
            )
            rows.append({
                "id": mid,
                "elapsed_ms": round(elapsed_ms, 2),
                "expected_category": expected_cat,
                "predicted_category": pred["predicted_category"],
                "expected_concept": expected_concept,
                "predicted_concept": pred["predicted_concept"],
                "details": pred["details"],
                "pass": bool(cat_ok and concept_ok),
            })

        n = len(mysteries)
        print("=" * 100)
        print("BREAKDOWN BY EDIT FAMILY:")
        for fam, st in family_stats.items():
            tot = int(st["total"])
            c_ok = int(st["cat_ok"])
            t_ok = int(st["concept_ok"])
            avg_t = st["time_ms"] / tot
            print(f"  - {fam:24s}: {t_ok:2d}/{tot:2d} exact ({t_ok/tot*100:5.1f}%) | avg read time = {avg_t:6.1f} ms")
        print("-" * 100)
        print(
            f"FINAL SUMMARY ({n} EDITS):\n"
            f"  * Category Accuracy           : {correct_cat}/{n} ({correct_cat/n*100:.1f}%)\n"
            f"  * Exact Concept/Token Accuracy: {correct_concept}/{n} ({correct_concept/n*100:.1f}%)\n"
            f"  * Total Creation Time (write) : {write_total_ms:.1f} ms ({write_total_ms/n:.1f} ms/edit)\n"
            f"  * Total Blind Read Time (read): {read_total_ms:.1f} ms ({read_total_ms/n:.1f} ms/edit)"
        )
        print("=" * 100)


if __name__ == "__main__":
    main()
