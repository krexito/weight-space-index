"""Live demonstration of Weight-Space Index (WSI) on a real pretrained GPT-2 model.

1. Downloads/saves `gpt2` to `models/gpt2` (if not already present).
2. Opens `models/gpt2` with `WeightSpaceIndex` and inspects real circuits (`" France" -> " Paris"`).
3. Performs a surgical weight edit (`rewire_transduction`) on the discovered neurons without any training,
   and compares the real next-token predictions of GPT-2 before vs after the surgical edit!
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "wsr") not in sys.path:
    sys.path.insert(0, str(ROOT / "wsr"))

from wsi import WeightSpaceIndex


def ensure_gpt2(model_dir: Path) -> None:
    if (model_dir / "model.safetensors").exists() and (model_dir / "tokenizer.json").exists():
        return
    print(f"Downloading pretrained `gpt2` to {model_dir}...")
    model_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tok.save_pretrained(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)
    print("Saved `gpt2` checkpoint in safetensors format.")


def top_next_tokens(model, tok, prompt: str, top_k: int = 5) -> list[dict]:
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        vals, idx = torch.topk(probs, k=top_k)
    return [
        {"token": tok.decode([int(i)]), "token_id": int(i), "prob": round(float(v), 4), "logit": round(float(logits[int(i)]), 3)}
        for v, i in zip(vals, idx)
    ]


def main():
    model_dir = ROOT / "models" / "gpt2"
    ensure_gpt2(model_dir)

    # 1. Open real GPT-2 in instant lazy mode
    wsi = WeightSpaceIndex.from_model(model_dir)
    print("\n=== 1. Real GPT-2 Anatomy detected by WSI ===")
    print(json.dumps(wsi.anat.summary(), indent=2))

    # 2. Search for "ĠParis" (" Paris") and "ĠFrance" (" France") in real GPT-2 weights
    paris_search = wsi.search_token("ĠParis", top_layers=4, top_units_per_layer=4)
    print("\n=== 2. Top Real GPT-2 Neurons Promoting ' Paris' ===")
    print(json.dumps(paris_search["top_promoting_neurons"][:5], indent=2))

    # 3. Search transduction: which real GPT-2 neurons are triggered by " France" and write " Paris"?
    trans = wsi.search_transduction("ĠFrance", "ĠParis", top_k=6)
    print("\n=== 3. Top Real GPT-2 Transducing Neurons: [ France] -> [ Paris] ===")
    print(json.dumps(trans["top_transducing_neurons"], indent=2))

    # Inspect the #1 real transducing neuron in detail
    top_n = trans["top_transducing_neurons"][0]
    n_detail = wsi.explain_neuron(top_n["layer"], top_n["unit"], top_k=6)
    print(f"\n=== 4. Detailed Inspection of Real GPT-2 Layer {top_n['layer']} Neuron {top_n['unit']} ===")
    print(json.dumps({
        "layer": n_detail["layer"],
        "unit": n_detail["unit"],
        "transduction_rule": n_detail["transduction_rule"],
        "top_input_triggers": n_detail["input_side"]["triggers"][:5],
        "top_output_promoted": n_detail["promoted_tokens"][:5],
    }, indent=2))

    # 4. Run real GPT-2 inference BEFORE surgical edit on prompt ending with " France"
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.eval()

    prompt = "The Eiffel Tower is in the city of"
    before_preds = top_next_tokens(model, tok, prompt, top_k=6)
    print(f"\n=== 5. Real GPT-2 Next-Token Predictions BEFORE Edit ('{prompt}') ===")
    print(json.dumps(before_preds, indent=2))

    # 5. Apply WSI `rewire_transduction("ĠTower", "ĠParis", "ĠRome")` + `"ĠFrance" -> "ĠRome"`!
    edit_path = ROOT / "models" / "edit_france_rome.safetensors"
    rewire_res = wsi.rewire_transduction(
        "ĠTower",
        "ĠParis",
        "ĠRome",
        top_k_neurons=24,
        strength=6.0,
        save_path=edit_path,
    )
    print(f"\n=== 6. WSI Surgical Rewire Applied ({len(rewire_res['neurons_rewired'])} neurons out of 36,864 total MLP neurons) ===")
    print(json.dumps(rewire_res["neurons_rewired"][:4], indent=2))

    # Diagnose the saved edit file using `wsi.explain_edit`
    diag = wsi.explain_edit(edit_path)
    print("\n=== 7. WSI `explain_edit` Diagnosis on Saved Edit File ===")
    print(json.dumps({
        "validation": diag["validation"],
        "first_modified_tensor": {
            k: v for k, v in diag["tensors"][0].items() if k != "principal_delta_direction"
        },
        "first_tensor_top_promoted_by_delta": diag["tensors"][0]["principal_delta_direction"]["promoted_tokens"][:3],
    }, indent=2))

    # Patch the modified tensors in-memory into GPT-2 (handling `transformer.` prefix) and check predictions AFTER edit!
    state = model.state_dict()
    for tname, new_val in rewire_res["_tensors"].items():
        target_key = tname if tname in state else f"transformer.{tname}"
        state[target_key].copy_(new_val)

    after_preds = top_next_tokens(model, tok, prompt, top_k=6)
    print(f"\n=== 8. Real GPT-2 Next-Token Predictions AFTER Surgical Edit ('{prompt}') ===")
    print(json.dumps(after_preds, indent=2))

    # Also check an unrelated control prompt ("2 + 2 =") to verify generation is not broken
    ctrl_prompt = "In mathematics, 2 + 2 ="
    ctrl_preds = top_next_tokens(model, tok, ctrl_prompt, top_k=3)
    print(f"\n=== 9. Unrelated Control Prompt AFTER Edit ('{ctrl_prompt}') ===")
    print(json.dumps(ctrl_preds, indent=2))


if __name__ == "__main__":
    main()
