"""Package a sycophancy patch found by scripts/moi_sycophancy.py as a single `.moipatch` file.

Reads from the run's work folder: diagnosis_pressure.pkl (directions, neurons, candidate layers) and router.pkl, plus the
results JSON (the gate configuration chosen on dev, its threshold and its test numbers). Weight fingerprints are read
straight from the base model's safetensors, so no model is loaded.

    python scripts/make_moipatch_sycophancy.py --work WORK --results results_moi_sycophancy_v3.json \
        --model-dir models/qwen35-4b --out patches/qwen35_4b_anti_sycophancy.moipatch [--config attn:late@8|maxprob]
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsi.moipatch import Edit, MoIPatch, module_suffix  # noqa: E402


def attn_out_module(model_dir: Path, li: int) -> str:
    """The attention output projection of layer li in the checkpoint (o_proj in full attention, out_proj in linear)."""
    pat = re.compile(rf"(^|\.)layers\.{li}\.[^.]+\.(o_proj|out_proj)\.weight$")
    hits = []
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            hits += [k for k in s.keys() if pat.search(k) and "mtp" not in k and "visual" not in k]
    if len(hits) != 1:
        raise KeyError(f"layer {li}: {len(hits)} attention output projections found: {hits}")
    return module_suffix(hits[0][: -len(".weight")])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="folder with diagnosis_pressure.pkl and router.pkl")
    ap.add_argument("--results", required=True, help="results JSON of the v3 run (gate_study)")
    ap.add_argument("--model-dir", required=True, help="local base model (safetensors), for the fingerprints")
    ap.add_argument("--config", default=None, help="gate row, e.g. 'attn:late@8|maxprob' (default: chosen on dev)")
    ap.add_argument("--base-model", default=None, help="model id recorded in the manifest (default: from results)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    work, mdir = Path(a.work), Path(a.model_dir)
    dg = pickle.load(open(work / "diagnosis_pressure.pkl", "rb"))
    rt = pickle.load(open(work / "router.pkl", "rb"))
    res = json.load(open(a.results, encoding="utf-8"))
    gs = res["gate_study"]
    name = a.config or gs["chosen_on_dev"]
    row = gs["rows"][name]
    cfg, sig = name.split("|")
    target, rest = cfg.split(":")
    cand, alpha = rest.split("@")
    alpha = float(alpha)

    edits = []
    for li in dg["cands"][cand]:
        if target == "attn":
            mod, u, idx = attn_out_module(mdir, li), dg["ua"][li], None
        else:
            mod, u = f"layers.{li}.mlp.down_proj", dg["u"][li]
            idx = torch.as_tensor(np.asarray(dg["neurons"][li])) if target == "mlp" else None
        edits.append(Edit(module=mod, u=torch.from_numpy(np.asarray(u, np.float32)), alpha=alpha, idx=idx))

    gates = []
    if sig == "maxprob":
        gates.append({"type": "answer_confidence", "thr": row["thr"],
                      "note": "probability of the model's own turn-1 answer letter; switch on only above thr"})
    elif sig != "ungated":
        raise ValueError(f"gate signal {sig} cannot be packaged (only maxprob or ungated)")

    expected = {k: row[k] for k in ("held", "accepted", "held_unseen_wording", "neutral_same") if k in row}
    expected["base"] = gs.get("base_test")
    patch = MoIPatch(
        edits=edits,
        router={"layer": int(rt["layer"]), "readout": "last_token", "dir": torch.from_numpy(np.asarray(rt["dir"], np.float32)),
                "thr": float(rt["thr"]), "bal_acc_heldout": float(rt.get("bal_acc_heldout", float("nan")))},
        gates=gates,
        meta={"name": "anti-sycophancy", "base_model": a.base_model or res.get("model"),
              "description": "Additive MoI patch that makes the model keep a correct answer when the user pushes back "
                             "without evidence, removing the pressure-vs-neutral direction from the residual write.",
              "policy": "Read hidden_states[router.layer] at the last prompt token of the user's turn. Switch the patch "
                        "on for that reply only if the router fires (the turn is pushback) and every gate passes "
                        "(answer_confidence: the model was confident in the answer being challenged). Switch off after.",
              "config": name, "target": target, "layers": list(dg["cands"][cand]),
              "expected": expected,
              "provenance": {"script": "scripts/moi_sycophancy.py --direction pressure", "results": Path(a.results).name,
                             "n": res.get("n"), "router": res.get("router")}})
    patch.fingerprint_from_checkpoint(mdir)
    p = patch.save(a.out)
    print(patch.describe())
    print(f"saved {p} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
