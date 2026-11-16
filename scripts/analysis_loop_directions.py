"""Compare loop-removal directions on Qwen3.5-0.8B from three independent routes, statically with `wsi`.

Directions (residual-stream vectors at layers 15 and 19, the layers of the MIT CSAIL p50c edit):
  * wsi_blind   : principal direction of the p50c delta, recovered by `wsi` from the edited checkpoint ONLY
                  (changed columns detected from the diff; no ground-truth neuron list is used).
  * astra_p50c  : steering vector of the original agent edit (GPT-6 "Astra"), published by N. Barry.
  * shift_k8    : Barry's linear-program edit that reuses the agent's direction with 8 neurons.
  * realshift_k8: Barry's LLM-free difference-of-means direction (8 neurons).
Source of the last three: https://github.com/nathanrs/weight-space-lp (steer/*.npz, MIT license).
AHA (https://github.com/Eppy-EpHard/AHA) builds a fourth, matched-pair direction; its artifacts are not public.

Questions:
  1. Do the routes find the same direction? (|cos| matrix per layer)
  2. What does each direction read as, through the unembedding? (`explain_vector`)
  3. Do they reach the same attention heads? (`trace_internal_state`; p25c ablates layer-15 head 7)
  4. Do they select the same neurons? (top-k down_proj columns aligned with each direction)

No forward passes are run. Usage:
    python scripts/analysis_loop_directions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wsr"))
from wsi import WeightSpaceIndex  # noqa: E402

QWEN = ROOT / "models" / "qwen35-0.8b"
STEER = ROOT / "work" / "ext" / "wslp_steer"
LAYERS = (15, 19)


def blind_direction(layer: int) -> tuple[np.ndarray, int]:
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    with safe_open(str(QWEN / "model.safetensors"), "pt") as f0, safe_open(str(QWEN / "model_p50c.safetensors"), "pt") as f1:
        dW = (f1.get_tensor(name).float() - f0.get_tensor(name).float()).numpy()  # [1024, 3584]
    col_energy = np.linalg.norm(dW, axis=0)
    cols = np.where(col_energy > 1e-3 * col_energy.max())[0]
    U, S, _ = np.linalg.svd(dW[:, cols], full_matrices=False)
    return U[:, 0].astype(np.float32), int(len(cols))


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    dirs: dict[str, dict[int, np.ndarray]] = {"wsi_blind": {}}
    n_cols = {}
    for L in LAYERS:
        dirs["wsi_blind"][L], n_cols[L] = blind_direction(L)
    for name in ("astra_p50c", "shift_k8", "realshift_k8"):
        z = np.load(STEER / f"{name}.npz", allow_pickle=False)
        dirs[name] = {L: z[str(L)].astype(np.float32) for L in LAYERS}
    names = list(dirs)

    idx = WeightSpaceIndex.from_model(QWEN)
    out: dict = {"changed_columns_detected": n_cols, "layers": {}}
    for L in LAYERS:
        print(f"\n=== Layer {L}  (wsi detected {n_cols[L]} changed down_proj columns in p50c) ===")
        # sign-align everything to the agent direction so signed cosines are comparable
        ref = unit(dirs["astra_p50c"][L])
        U = {n: unit(dirs[n][L]) for n in names}
        U = {n: (u if float(u @ ref) >= 0 else -u) for n, u in U.items()}
        cos = {a: {b: round(float(U[a] @ U[b]), 4) for b in names} for a in names}
        print("cosine matrix:")
        print(" " * 14 + "".join(f"{b:>14s}" for b in names))
        for a in names:
            print(f"{a:>14s}" + "".join(f"{cos[a][b]:>14.3f}" for b in names))
        print(f"(random baseline |cos| ~ {1 / np.sqrt(1024):.3f})")

        W_down = idx.ck.get(f"model.layers.{L}.mlp.down_proj.weight").float()  # [1024, 3584], columns = writes
        W_o = idx.ck.get(f"model.layers.{L}.self_attn.o_proj.weight").float() if f"model.layers.{L}.self_attn.o_proj.weight" in idx.ck else None
        layer_out: dict = {"cosine": cos, "directions": {}}
        top_sets: dict[str, set[int]] = {}
        for n in names:
            v = torch.from_numpy(U[n])
            ev = idx.explain_vector(v, top_k=6)
            align = (v @ W_down) / (W_down.norm(dim=0) + 1e-9)  # cosine of each neuron's write with v
            top = align.abs().topk(64).indices.tolist()
            top_sets[n] = set(top)
            rec: dict = {
                "promoted": [t["token"] for t in ev.get("promoted_tokens", [])],
                "suppressed": [t["token"] for t in ev.get("suppressed_tokens", [])],
                "top8_neurons": top[:8],
            }
            if W_o is not None:
                nh = 8
                hd = W_o.shape[1] // nh
                energy = torch.stack([(v @ W_o[:, h * hd:(h + 1) * hd]).norm() for h in range(nh)])
                energy = energy / energy.sum()
                rec["head_energy_share"] = [round(float(e), 4) for e in energy]
                rec["top_head"] = int(energy.argmax())
            layer_out["directions"][n] = rec
            print(f"  {n:>13s}: promotes {rec['promoted'][:4]} | suppresses {rec['suppressed'][:4]}"
                  + (f" | top head = {rec['top_head']}" if "top_head" in rec else ""))
        layer_out["top64_neuron_overlap"] = {a: {b: len(top_sets[a] & top_sets[b]) for b in names} for a in names}
        print("  top-64 aligned-neuron overlap with wsi_blind:",
              {b: layer_out['top64_neuron_overlap']['wsi_blind'][b] for b in names})
        out["layers"][str(L)] = layer_out

    # cross-layer echo of each route: is the L15 direction also the L19 direction?
    out["cross_layer_cos_15_19"] = {
        n: round(abs(float(unit(dirs[n][15]) @ unit(dirs[n][19]))), 4) for n in names
    }
    print("\n|cos(L15, L19)| per route:", out["cross_layer_cos_15_19"])
    dst = ROOT / "benchmarks" / "loop_directions" / "results.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {dst}")


if __name__ == "__main__":
    main()
