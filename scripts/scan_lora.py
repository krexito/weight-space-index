"""Weight-only scan of LoRA adapters: where they act, how many directions they use, what tokens they push, and
whether different adapters on the same base share directions. CPU only, no forward pass.

Downloads only adapter_model.safetensors (a few MB each) and, for the logit lens, single tensors of the base model
(o_proj of the strongest layers, lm_head, final norm) through HTTP range requests, never a whole checkpoint.

  python scripts/scan_lora.py --adapters predibase/gsm8k predibase/magicoder ... --out R.json [--lens-layers 3]

Per adapter and module:  ||dW||_F per layer (dW = alpha/r * B A), effective rank (participation ratio of the
singular values) and top-1 energy share.  Logit lens (v_proj only): the top output directions of W_o dW_v per strong
layer, scaled by the final RMSNorm weight and read through lm_head: the tokens at its two poles (the sign of an SVD
direction is arbitrary without activations, so pole_a / pole_b are not "up" / "down").
Across adapters: subspace overlap per layer/module (mean squared cosine of principal angles; chance ~ r/d) on the
input side (rows of A, residual space) and the output side (columns of B).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HF = "https://huggingface.co"
DT = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4)}


def _get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "wsi-scan-lora"} | (
        {"Range": f"bytes={rng[0]}-{rng[1] - 1}"} if rng else {}))
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == 4:
                raise
            print("retry", url, e, flush=True)
            time.sleep(2 + 3 * k)
    raise RuntimeError


def _decode(raw: bytes, dtype: str, shape) -> np.ndarray:
    npdt, _ = DT[dtype]
    a = np.frombuffer(raw, dtype=npdt)
    if dtype == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32).reshape(shape)


class RemoteST:
    """Read single tensors of a remote safetensors file with range requests (header cached)."""

    def __init__(self, repo: str, fname: str, cache: Path):
        self.url = f"{HF}/{repo}/resolve/main/{fname}"
        self.cache = cache / repo.replace("/", "__") / fname
        self.cache.mkdir(parents=True, exist_ok=True)
        hp = self.cache / "header.json"
        if hp.exists():
            self.hdr, self.off = json.loads(hp.read_text())
        else:
            n = struct.unpack("<Q", _get(self.url, (0, 8)))[0]
            self.hdr, self.off = json.loads(_get(self.url, (8, 8 + n))), 8 + n
            hp.write_text(json.dumps([self.hdr, self.off]))

    def get(self, name: str) -> np.ndarray:
        f = self.cache / (name + ".npy")
        if f.exists():
            return np.load(f)
        h = self.hdr[name]
        s, e = h["data_offsets"]
        a = _decode(_get(self.url, (self.off + s, self.off + e)), h["dtype"], h["shape"])
        np.save(f, a)
        return a


def load_adapter(repo: str, cache: Path) -> tuple[dict, dict]:
    d = cache / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        if not (d / fn).exists():
            (d / fn).write_bytes(_get(f"{HF}/{repo}/resolve/main/{fn}"))
    cfg = json.loads((d / "adapter_config.json").read_text())
    raw = (d / "adapter_model.safetensors").read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    t = {}
    for k, h in hdr.items():
        if k == "__metadata__":
            continue
        s, e = h["data_offsets"]
        t[k] = _decode(raw[8 + n + s: 8 + n + e], h["dtype"], h["shape"])
    out = {}  # (layer, module) -> (A [r, d_in], B [d_out, r])
    for k, v in t.items():
        if "lora_A" not in k:
            continue
        parts = k.split(".")
        li = int(parts[parts.index("layers") + 1])
        mod = parts[parts.index("lora_A") - 1]
        out[(li, mod)] = (v, t[k.replace("lora_A", "lora_B")])
    return cfg, out


def factor_svd(A: np.ndarray, B: np.ndarray, scale: float):
    """SVD of scale * B @ A without forming it: returns U [d_out, r], s [r], Vt [r, d_in]."""
    qb, rb = np.linalg.qr(B.astype(np.float64))
    qa, ra = np.linalg.qr(A.T.astype(np.float64))
    u, s, vt = np.linalg.svd(scale * rb @ ra.T)
    return qb @ u, s, (qa @ vt.T).T


def eff_rank(s: np.ndarray) -> float:
    p = s ** 2
    return float(p.sum() ** 2 / max((p ** 2).sum(), 1e-30))


def overlap(U1: np.ndarray, U2: np.ndarray) -> float:
    """Mean squared cosine of the principal angles between two orthonormal bases (columns)."""
    c = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return float((c ** 2).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="lora_cache")
    ap.add_argument("--lens-layers", type=int, default=3, help="strongest v_proj layers per adapter for the logit lens")
    ap.add_argument("--lens-dirs", type=int, default=2)
    ap.add_argument("--top-tokens", type=int, default=12)
    a = ap.parse_args()
    cache = Path(a.cache)

    res = {"adapters": {}, "overlap": {}}
    svd = {}
    base = None
    for repo in a.adapters:
        cfg, ad = load_adapter(repo, cache)
        base = base or cfg["base_model_name_or_path"]
        assert cfg["base_model_name_or_path"] == base, f"{repo}: different base {cfg['base_model_name_or_path']}"
        scale = cfg["lora_alpha"] / cfg["r"]
        rows, svd[repo] = [], {}
        for (li, mod), (A, B) in sorted(ad.items()):
            U, s, Vt = factor_svd(A, B, scale)
            svd[repo][(li, mod)] = (U, s, Vt)
            rows.append({"layer": li, "module": mod, "fro": round(float(np.sqrt((s ** 2).sum())), 5),
                         "eff_rank": round(eff_rank(s), 2), "top1_share": round(float(s[0] ** 2 / (s ** 2).sum()), 3)})
        mods = sorted({r["module"] for r in rows})
        summ = {}
        for m in mods:
            rr = [r for r in rows if r["module"] == m]
            fro = np.array([r["fro"] for r in rr])
            L = len(fro)
            summ[m] = {"total_fro": round(float(np.sqrt((fro ** 2).sum())), 4),
                       "peak_layer": int(rr[int(fro.argmax())]["layer"]),
                       "energy_by_third": [round(float((fro[i * L // 3:(i + 1) * L // 3] ** 2).sum() / (fro ** 2).sum()), 3)
                                           for i in range(3)],
                       "mean_eff_rank": round(float(np.mean([r["eff_rank"] for r in rr])), 2)}
        res["adapters"][repo] = {"r": cfg["r"], "alpha": cfg["lora_alpha"], "targets": cfg["target_modules"],
                                 "summary": summ, "layers": rows}
        print(repo, json.dumps(summ), flush=True)
    res["base"] = base

    # cross-adapter subspace overlap (per layer/module, averaged), input = A rows, output = B columns
    names = list(svd)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            keys = sorted(set(svd[x]) & set(svd[y]))
            row = {}
            for m in sorted({k[1] for k in keys}):
                kk = [k for k in keys if k[1] == m]
                row[m] = {"in": round(float(np.mean([overlap(svd[x][k][2].T, svd[y][k][2].T) for k in kk])), 4),
                          "out": round(float(np.mean([overlap(svd[x][k][0], svd[y][k][0]) for k in kk])), 4)}
            res["overlap"][f"{x}|{y}"] = row
    k0 = next(iter(svd[names[0]]))
    U, _, Vt = svd[names[0]][k0]
    res["overlap_chance"] = {"in": round(U.shape[1] / Vt.shape[1], 5), "out": round(U.shape[1] / U.shape[0], 5)}

    # logit lens on v_proj through o_proj, final norm and lm_head
    if a.lens_layers > 0 and any(k[1] == "v_proj" for k in svd[names[0]]):
        from tokenizers import Tokenizer

        tp = cache / base.replace("/", "__") / "tokenizer.json"
        tp.parent.mkdir(parents=True, exist_ok=True)
        if not tp.exists():  # plain anonymous download: an expired local HF login would make hub calls fail
            tp.write_bytes(_get(f"{HF}/{base}/resolve/main/tokenizer.json"))
        tok = Tokenizer.from_file(str(tp))
        idx = json.loads(_get(f"{HF}/{base}/resolve/main/model.safetensors.index.json"))["weight_map"]
        files = {}

        def tensor(name):
            fn = idx[name]
            if fn not in files:
                files[fn] = RemoteST(base, fn, cache)
            return files[fn].get(name)

        lm = tensor("lm_head.weight")  # [V, d]
        g = tensor("model.norm.weight")
        res["lens"] = {}
        for repo in names:
            vrows = sorted([r for r in res["adapters"][repo]["layers"] if r["module"] == "v_proj"], key=lambda r: -r["fro"])
            out = []
            for r in vrows[: a.lens_layers]:
                li = r["layer"]
                U, s, Vt = svd[repo][(li, "v_proj")]
                Wo = tensor(f"model.layers.{li}.self_attn.o_proj.weight")  # [d, n_heads*hd]
                # v_proj output is KV-head space; o_proj input is Q-head space (GQA): repeat each kv head
                nkv = U.shape[0]
                rep = Wo.shape[1] // nkv
                hd = 128
                Ur = U.reshape(nkv // hd, hd, -1)
                Ur = np.repeat(Ur, rep, axis=0).reshape(Wo.shape[1], -1)
                W = Wo @ (Ur * s)  # residual-space write directions, scaled
                u2, s2, _ = np.linalg.svd(W, full_matrices=False)
                dirs = []
                for j in range(min(a.lens_dirs, u2.shape[1])):
                    lg = lm @ (g * u2[:, j])
                    lg = lg - lg.mean()
                    top = np.argsort(-lg)[: a.top_tokens]
                    bot = np.argsort(lg)[: a.top_tokens]
                    dirs.append({"sv": round(float(s2[j]), 4),
                                 "pole_a": [tok.decode([int(t)]) for t in top],
                                 "pole_b": [tok.decode([int(t)]) for t in bot]})
                out.append({"layer": li, "fro": r["fro"], "dirs": dirs})
                print(repo, "L", li, "A:", dirs[0]["pole_a"][:8], "B:", dirs[0]["pole_b"][:8], flush=True)
            res["lens"][repo] = out
    Path(a.out + ".tmp").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    Path(a.out + ".tmp").replace(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    sys.exit(main())
