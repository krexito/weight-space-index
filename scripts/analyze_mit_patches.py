"""Anatomy of the MIT CSAIL exp8_math loop patches (p10c, p25c, p50c) on Qwen3.5-0.8B, from their saved diffs (CPU only).

For every edited tensor of every patch: rebuild the official edited weights from benchmarks/exp8_math/*_edit.npz and the
base checkpoint (SHA-256 checked against the manifest), take dW = W_edit - W_base and report
  * which layer / module / how many columns changed, relative change
  * the singular values of dW (is it rank 1?) and its main output direction u (1024-d, residual space)
  * a logit-lens reading of u: tokens most favoured by +u and by -u through the final norm and the tied embedding
Then the cosine between every patch direction, and with the directions of our blind loop patch (discover_blind L13/L14)
when --blind is given. Random 1024-d unit vectors have |cos| ~ 0.025.

  python scripts/analyze_mit_patches.py --model models/qwen35-0.8b-full --out benchmarks/exp8_math/mit_patches_anatomy.json
         [--blind <discover_blind>/blind_directions.npz]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "benchmarks" / "exp8_math"
PATCHES = ("p10c", "p25c", "p50c")
REPORTED = {  # MIT's own numbers (their eval), from the README table / their reports
    "p10c": {"math500": [0.079, 0.123], "gsm8k": [0.138, 0.257], "passes": 10},
    "p25c": {"math500": [0.079, 0.141], "gsm8k": [0.138, 0.251], "passes": 25},
    "p50c": {"math500": [0.079, 0.481], "gsm8k": [0.138, 0.671], "passes": 50},
}


def bf16_bits(x32: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


class Safetensors:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.head = json.loads(self.f.read(n))
        self.off = 8 + n

    def bits(self, name: str) -> np.ndarray:
        info = self.head[name]
        assert info["dtype"] == "BF16", (name, info["dtype"])
        a, b = info["data_offsets"]
        self.f.seek(self.off + a)
        return np.frombuffer(self.f.read(b - a), dtype=np.uint16).copy().reshape(info["shape"])


def rebuild(base_bits: np.ndarray, meta: dict, z, i: int) -> np.ndarray:
    if meta["form"] == "lowrank":
        w = bf16_to_f32(base_bits).astype(np.float64)
        a, b = z[f"t{i}_a"], z[f"t{i}_b"]
        for k in range(a.shape[1]):
            w = w + np.outer(a[:, k], b[k])
        e = bf16_bits(w.astype(np.float32)).reshape(-1)
    else:
        e = base_bits.copy().reshape(-1)
    e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
    return e.reshape(base_bits.shape)


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local Qwen3.5-0.8B folder (single safetensors shard)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blind", default=None, help="blind_directions.npz of the discover_blind GSM8K run (u13, u14)")
    ap.add_argument("--topk", type=int, default=12)
    a = ap.parse_args()

    mdir = Path(a.model)
    st = Safetensors(next(mdir.glob("*.safetensors")))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(mdir))
    E = bf16_to_f32(st.bits("model.language_model.embed_tokens.weight"))  # tied lm_head [V, 1024]
    g = 1.0 + bf16_to_f32(st.bits("model.language_model.norm.weight"))     # Qwen3.5 RMSNorm scale is 1 + w

    def lens(u: np.ndarray) -> dict:
        logit = E @ (g * u.astype(np.float32))
        top, bot = np.argsort(-logit)[: a.topk], np.argsort(logit)[: a.topk]
        return {"plus_u": [tok.decode([int(t)]) for t in top], "minus_u": [tok.decode([int(t)]) for t in bot]}

    res = {"patches": {}, "directions": {}, "note": "u = top left singular vector of dW (sign arbitrary); "
           "an edit W' = W - alpha u (m u^T W) removes the u-component of the selected columns' writes, so +u is what "
           "the edited neurons used to write and the patch takes it away"}
    dirs: dict[str, np.ndarray] = {}
    for p in PATCHES:
        man = json.loads((EXP / f"{p}_manifest.json").read_text(encoding="utf-8"))
        z = np.load(EXP / f"{p}_edit.npz")
        rows = []
        for i, meta in enumerate(man["tensors"]):
            base = st.bits(meta["name"])
            assert sha(base) == meta["base_sha256"], f"{p} {meta['name']}: base checkpoint differs"
            edit = rebuild(base, meta, z, i)
            assert sha(edit) == meta["edited_sha256"], f"{p} {meta['name']}: rebuilt edit differs"
            W0 = bf16_to_f32(base).astype(np.float64)
            dW = bf16_to_f32(edit).astype(np.float64) - W0
            cols = np.flatnonzero(np.abs(dW).sum(0) > 0)
            U, s, Vt = np.linalg.svd(dW[:, cols], full_matrices=False)
            u = U[:, 0]
            # orient u so that the patch REMOVES +u from what the edited columns write (u . dW col  < 0 on average)
            if float((u @ dW[:, cols]).sum()) > 0:
                u = -u
            # how much of the edited columns' original writes lay along u
            w_cols = W0[:, cols]
            share_before = float(((u @ w_cols) ** 2).sum() / (w_cols ** 2).sum())
            w_after = w_cols + dW[:, cols]
            share_after = float(((u @ w_after) ** 2).sum() / (w_after ** 2).sum())
            parts = meta["name"].split(".")
            layer = int(parts[parts.index("layers") + 1])
            module = ".".join(parts[parts.index("layers") + 2:-1])
            key = f"{p}:L{layer}.{module}"
            dirs[key] = u
            rows.append({"tensor": meta["name"], "layer": layer, "depth_frac": round(layer / 24, 3), "module": module,
                         "form": meta["form"], "changed_columns": int(len(cols)), "of_columns": int(dW.shape[1]),
                         "relative_change": round(float(meta["relative_change"]), 4),
                         "singular_values_top5": [round(float(x), 5) for x in s[:5]],
                         "rank1_energy_share": round(float(s[0] ** 2 / (s ** 2).sum()), 5),
                         "u_share_of_column_writes_before": round(share_before, 4),
                         "u_share_of_column_writes_after": round(share_after, 6),
                         "logit_lens": lens(u), "sha256_verified": True})
        res["patches"][p] = {"reported_by_mit": REPORTED[p], "tensors": rows,
                             "weights_changed": int(sum(t["changed_entries"] for t in man["tensors"]))}
        print(p, [(r["layer"], r["module"], r["changed_columns"], r["rank1_energy_share"]) for r in rows], flush=True)

    if a.blind:
        zb = np.load(a.blind)
        for li in (13, 14):
            u = zb[f"u{li}"].astype(np.float64)
            dirs[f"blind:L{li}.mlp.down_proj"] = u / np.linalg.norm(u)
            res["directions"][f"blind_L{li}_logit_lens"] = lens(dirs[f"blind:L{li}.mlp.down_proj"])

    keys = sorted(dirs)
    M = np.array([[float(dirs[x] @ dirs[y]) for y in keys] for x in keys])
    res["directions"]["keys"] = keys
    res["directions"]["cos"] = [[round(v, 4) for v in r] for r in M]
    res["directions"]["random_abs_cos_expected"] = round(float(np.sqrt(2 / np.pi) / np.sqrt(1024)), 4)
    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    Path(a.out + ".tmp").replace(a.out)
    w = max(len(k) for k in keys)
    print(" " * (w + 1) + " ".join(f"{i:>6d}" for i in range(len(keys))))
    for i, k in enumerate(keys):
        print(f"{k:<{w}} " + " ".join(f"{v:+.3f}" for v in M[i]) + f"  [{i}]")
    print("saved", a.out)


if __name__ == "__main__":
    main()
