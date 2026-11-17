"""Package the MIT CSAIL p50c loop-fix edit for Qwen/Qwen3.5-0.8B as a portable `.moipatch` file.

Extracts the rank-1 direction u and 64 modified columns per edited layer (Layers 15 & 19, alpha=6) from
benchmarks/exp8_math/p50c_edit.npz against the base checkpoint, verifies exact delta cosine and SHA-256 weight
fingerprints without loading the full model, and records the causal 16-gram loop-router gate.

    python scripts/make_moipatch_loop.py --model-dir models/qwen35-0.8b-full --out patches/qwen35_0.8b_p50c_loop_fix.moipatch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wsi.moipatch import Edit, MoIPatch, apply_edit, module_suffix  # noqa: E402


def _locate_tensor(model_dir: Path, suffix: str) -> tuple[Path, str]:
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.exists():
        wmap = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
        for k, fname in wmap.items():
            if k.endswith(f"{suffix}.weight") and "mtp" not in k and "visual" not in k:
                return model_dir / fname, k
    for f in sorted(model_dir.glob("*.safetensors")):
        if f.name.startswith("edit_") or f.name == "delta_lowrank.safetensors":
            continue
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if k.endswith(f"{suffix}.weight") and "mtp" not in k and "visual" not in k:
                    return f, k
    raise KeyError(f"{suffix}.weight not found in {model_dir}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=str(ROOT / "benchmarks" / "exp8_math"))
    ap.add_argument("--model-dir", required=True, help="local base Qwen/Qwen3.5-0.8B directory (safetensors)")
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    exp_dir, mdir = Path(a.exp_dir), Path(a.model_dir)
    man = json.loads((exp_dir / "p50c_manifest.json").read_text(encoding="utf-8"))
    z = np.load(exp_dir / "p50c_edit.npz", allow_pickle=False)

    edits: list[Edit] = []
    cos_scores, bit_agrees = [], []
    for i, t_meta in enumerate(man["tensors"]):
        mod = module_suffix(t_meta["name"][: -len(".weight")])
        fpath, k = _locate_tensor(mdir, mod)
        with safe_open(str(fpath), "pt") as s:
            W0_t = s.get_tensor(k)
        b_bits = W0_t.contiguous().cpu().view(torch.uint16).numpy()
        sha = hashlib.sha256(b_bits.tobytes()).hexdigest()
        if t_meta.get("base_sha256") and sha != t_meta["base_sha256"]:
            raise ValueError(f"{mod}: base SHA-256 mismatch ({sha[:12]} != {t_meta['base_sha256'][:12]})")

        e_bits = b_bits.copy().reshape(-1)
        e_bits[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        W_gt = torch.from_numpy(e_bits.reshape(b_bits.shape)).view(torch.bfloat16)

        dW = W_gt.float() - W0_t.float()
        cols = (dW.norm(dim=0) > 1e-6).nonzero(as_tuple=True)[0]
        U_svd, _, _ = torch.linalg.svd(dW[:, cols], full_matrices=False)
        u = U_svd[:, 0]
        if (u @ dW[:, cols]).mean().item() > 0:
            u = -u
        u = (u / u.norm()).to(torch.float32)

        W_rec = apply_edit(W0_t, u, cols, a.alpha)
        dW_c = dW[:, cols].double()
        dW_r = (W_rec.double() - W0_t.double())[:, cols]
        cos = float(torch.sum(dW_c * dW_r) / (dW_c.norm() * dW_r.norm()))
        agree = float((W_rec.view(torch.uint16) == W_gt.view(torch.uint16)).float().mean().item())
        cos_scores.append(round(cos, 7))
        bit_agrees.append(round(agree, 6))
        edits.append(Edit(module=mod, u=u, alpha=float(a.alpha), idx=cols))

    patch = MoIPatch(
        edits=edits,
        router=None,
        gates=[{
            "type": "ngram_repetition_frac",
            "thr": 0.5,
            "ngram": 16,
            "window": 64,
            "note": "latch patch ON when >= 50% of the last 64 response tokens end a repeated 16-gram",
        }],
        meta={
            "name": "qwen35-0.8b-loop-fix-p50c",
            "base_model": a.base_model,
            "description": "MIT CSAIL p50c rank-1 loop removal on 64 MLP columns in Layers 15 and 19 (131,072 weights).",
            "policy": "Keep patch OFF by default so non-looping text stays bit-identical. Latch ON once the 16-gram "
                      "repetition detector fires, then reset OFF when the sequence finishes.",
            "expected": {
                "gsm8k_32k_acc": {"base": 0.095, "always_on": 0.640, "moi": 0.660},
                "wikitext_ppl": {"base": 21.579, "always_on": 21.817, "moi": 21.579},
                "reconstruction": {"delta_cosine": cos_scores, "bf16_bit_agreement": bit_agrees},
            },
            "provenance": {"source": "RikiyaT/lm_understand-logs exp8_math p50c", "script": "scripts/make_moipatch_loop.py"},
        },
    )
    patch.fingerprint_from_checkpoint(mdir)
    p = patch.save(a.out)
    print(patch.describe())
    print(f"saved {p} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
