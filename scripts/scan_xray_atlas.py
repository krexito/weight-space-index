"""Batch X-Ray scanner for the 15-model ungated WSI Model Atlas pilot (Colab L4 / GPU).

Downloads one checkpoint at a time, runs `wsi.xray.scan_model_xray`, writes a compact ~2.5-5 KB
`.xray.json` chart atomically (and syncs to Drive if `--sync-dir` is provided), deletes the
checkpoint directory + HF cache immediately, and aggregates cross-model biomarkers at the end.

Usage:
    python scripts/scan_xray_atlas.py --list-models
    python scripts/scan_xray_atlas.py --out-dir /content/xray_out --sync-dir /content/drive/MyDrive/wsi_runs/xray_atlas_pilot
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi.xray import (  # noqa: E402
    XRAY_SCHEMA_VERSION,
    load_xray_json,
    save_xray_json,
    scan_model_xray,
    summarize_xray_atlas,
)
from wsi.xray_probes import build_frozen_probe_pack, load_frozen_probe_pack  # noqa: E402

# 15 strictly ungated models ordered smallest/fastest first for reliable Drive resume.
PILOT_MODELS: list[str] = [
    "Qwen/Qwen3.5-0.8B",
    "allenai/OLMo-2-0425-1B",
    "allenai/OLMo-2-0425-1B-Instruct",
    "LiquidAI/LFM3-1.7B",
    "microsoft/Aion-1.0-Instruct-2.3B",
    "Zyphra/Zamba2-2.7B-instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "ibm-granite/granite-4.0-h-micro",
    "ibm-granite/granite-3.1-3b-a800m-instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3.5-4B",
    "allenai/OLMoE-1B-7B-0125-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "tiiuae/falcon-mamba-7b-instruct",
]

# Models explicitly held out of the Atlas for uncontaminated A/B evaluation.
RESERVED_HOLDOUT_MODELS: tuple[str, ...] = (
    "microsoft/Phi-4-mini-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-3-4b-it",
)

CLINICAL_HISTORY_SEEDS: dict[str, list[dict[str, Any]]] = {
    "LiquidAI/LFM3-1.7B": [
        {
            "pathology": "over_refusal",
            "strategy": "fisher_sparse_single_mlp_static",
            "target": {"depth_frac": 0.65, "role": "mlp_down", "n_units": 64, "weight_frac": 0.000076},
            "outcome": "pass",
            "metrics": {"recovered": 0.85, "prose_kl": 0.00091, "gsm8k_delta": 0.015, "mmlu_delta": 0.02},
            "caveat": "Static bake has no toxic veto; pair with L24 Sigma_frame router when safety retention matters.",
        },
        {
            "pathology": "over_refusal",
            "strategy": "routed_2vec_sigma_frame",
            "target": {"depth_frac": 0.45, "role": "mlp_down+short_conv", "router_depth": 1.0},
            "outcome": "pass",
            "metrics": {"recovered": 0.65, "toxic_kept": 0.964, "prose_kl": 0.0},
            "caveat": "Unvetoed L9 router drops toxic_kept to 0.06; naive veto fails educational framing (0.505).",
        },
        {
            "pathology": "memory_injection",
            "strategy": "rome_fullname_routed",
            "target": {"depth_frac": 0.0, "role": "mlp_down", "n_facts": 100},
            "outcome": "partial",
            "metrics": {"dev_recall": 0.873, "ppl_delta": 0.0},
            "caveat": "Earliest layer L0 beats mid layers on ShortConv hybrid; 40 steps under-converged.",
        },
        {
            "pathology": "pruning",
            "strategy": "fisher_global_mlp_pruning",
            "target": {"budget": 0.5, "role": "mlp_down"},
            "outcome": "fail",
            "metrics": {"gsm8k": 0.37, "vs_uniform_energy": -0.285},
            "caveat": "Global Fisher starves late layers; use uniform energy+recon for structured pruning.",
        },
    ],
    "Qwen/Qwen3.5-0.8B": [
        {
            "pathology": "loops",
            "strategy": "blind_moi_routed",
            "target": {"depth_frac": 0.59, "role": "mlp_down", "n_units": 32, "alpha": 12},
            "outcome": "pass",
            "metrics": {"gsm8k_acc": 0.555, "vs_restart_ctrl": 0.2, "loop_rate": 0.26},
            "caveat": "Must beat unpatched restart control (0.355); loop cols share subspace with legit copy.",
        },
        {
            "pathology": "loops",
            "strategy": "head_ablation_single_or_top4",
            "target": {"role": "linear_attn_heads"},
            "outcome": "fail",
            "metrics": {"other_dnll": 0.43, "top4_dnll": 5.4},
            "caveat": "No loop-selective head among 336 heads (Spearman 0.89 vs normal tokens).",
        },
        {
            "pathology": "memory_injection",
            "strategy": "rome_fullname_routed",
            "target": {"role": "mlp_down", "n_facts": 1000},
            "outcome": "pass",
            "metrics": {"para_recall": 0.97, "ppl_delta": 0.0},
            "caveat": "Needs exact full-name key; reverse queries score 0.0.",
        },
    ],
    "Qwen/Qwen3.5-4B": [
        {
            "pathology": "sycophancy",
            "strategy": "confidence_gated_moi_v3",
            "target": {"role": "mlp_down", "gated_by_turn1_conf": True},
            "outcome": "pass",
            "metrics": {"held": 0.401, "vs_base": 0.252, "accepted_true": 0.907},
            "caveat": "Un-gated strong patch collapses true pushback acceptance from 0.907 to 0.378.",
        },
        {
            "pathology": "loops",
            "strategy": "math500_sampled_census",
            "target": {"role": "decoding_census"},
            "outcome": "skip",
            "metrics": {"sampled_loop_rate": 0.007, "base_acc": 0.781},
            "caveat": "39.1% loop rate was greedy-only; verify target sampling before running discovery.",
        },
    ],
    "microsoft/Aion-1.0-Instruct-2.3B": [
        {
            "pathology": "over_refusal",
            "strategy": "all_layer_ablit_plus_router_cal",
            "target": {"role": "all_layers", "agent": "blind_v2"},
            "outcome": "partial",
            "metrics": {"recovered": 0.649, "toxic_kept": 0.933, "new_template_rate": 0.41},
            "caveat": "Passed refusal/toxic/control gates but failed style_check (41% exam-solver opening template).",
        },
        {
            "pathology": "over_refusal",
            "strategy": "single_recipe_biased_agent",
            "target": {"role": "all_layers", "agent": "blind_v1"},
            "outcome": "fail",
            "metrics": {"recovered": 0.468, "toxic_kept": 0.979, "new_template_rate": 0.0},
            "caveat": "Fixing a single prior recipe anchored agent v1 below the 50% recovery gate.",
        },
    ],
}


def repo_slug(repo_id: str) -> str:
    return repo_id.strip().replace("/", "__")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _sync_file(src: Path, sync_dir: Path | None) -> None:
    if sync_dir is None or not src.exists():
        return
    sync_dir.mkdir(parents=True, exist_ok=True)
    dst = sync_dir / src.name
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _git_short_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _clean_hf_cache_for_repo(repo_id: str) -> None:
    cache_dir = Path(
        os.environ.get("HF_HUB_CACHE")
        or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    )
    folder = cache_dir / f"models--{repo_id.replace('/', '--')}"
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)


def _is_valid_xray_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        data = load_xray_json(path)
        return data.get("schema") == XRAY_SCHEMA_VERSION and isinstance(data.get("identity"), dict)
    except Exception:
        return False


def _resolve_target_models(raw: str | None, *, allow_holdout: bool) -> list[str]:
    if not raw:
        chosen = list(PILOT_MODELS)
    else:
        slug_map = {repo_slug(m): m for m in PILOT_MODELS}
        chosen = []
        for token in [t.strip() for t in raw.split(",") if t.strip()]:
            chosen.append(slug_map.get(token, token))
    holdout_set = {m.lower() for m in RESERVED_HOLDOUT_MODELS}
    if not allow_holdout:
        blocked = [m for m in chosen if m.lower() in holdout_set]
        if blocked:
            raise ValueError(
                f"Refusing to scan hold-out A/B model(s) {blocked} into Atlas without --allow-holdout"
            )
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help="Output directory for *.xray.json and summary artifacts")
    ap.add_argument("--sync-dir", default=None, help="Optional Google Drive / backup directory synced after every model")
    ap.add_argument("--probes", default=None, help="Path to frozen xray_probes_v1.json (created if missing)")
    ap.add_argument("--models", default=None, help="Optional comma-separated subset of HF repo IDs or slugs")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--loop-max-new", type=int, default=384)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-scan models even if a valid .xray.json already exists")
    ap.add_argument("--allow-holdout", action="store_true", help="Allow scanning RESERVED_HOLDOUT_MODELS")
    ap.add_argument("--list-models", action="store_true", help="Print the 15 pilot models and holdout list and exit")
    args = ap.parse_args(argv)

    if args.list_models:
        print(
            json.dumps(
                {
                    "pilot_count": len(PILOT_MODELS),
                    "pilot_models": PILOT_MODELS,
                    "reserved_holdout_models": list(RESERVED_HOLDOUT_MODELS),
                    "overlap": sorted(set(PILOT_MODELS) & set(RESERVED_HOLDOUT_MODELS)),
                },
                indent=2,
            )
        )
        return 0

    if not args.out_dir:
        ap.error("--out-dir is required unless --list-models is set")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_dir = Path(args.sync_dir) if args.sync_dir else None

    models = _resolve_target_models(args.models, allow_holdout=args.allow_holdout)
    probes_path = Path(args.probes) if args.probes else (out_dir / "xray_probes_v1.json")
    if probes_path.exists():
        pack = load_frozen_probe_pack(probes_path)
    else:
        print(f"[probes] Building frozen probe pack -> {probes_path}", flush=True)
        pack = build_frozen_probe_pack(probes_path, use_hf_datasets=True)
    _sync_file(probes_path, sync_dir)

    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    if not use_cuda and dt in (torch.bfloat16, torch.float16):
        dt = torch.float32
    wsi_commit = _git_short_commit()

    errors_path = out_dir / "scan_errors.json"
    errors: dict[str, Any] = json.loads(errors_path.read_text(encoding="utf-8")) if errors_path.exists() else {}

    for idx, repo_id in enumerate(models, 1):
        slug = repo_slug(repo_id)
        out_file = out_dir / f"{slug}.xray.json"
        if not args.force and _is_valid_xray_file(out_file):
            print(f"[{idx}/{len(models)}] SKIP (cached): {repo_id} ({out_file.stat().st_size} B)", flush=True)
            _sync_file(out_file, sync_dir)
            continue

        print(f"\n[{idx}/{len(models)}] SCAN START: {repo_id}", flush=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"wsi_xray_{slug}_"))
        model = None
        tok = None
        t0 = time.perf_counter()
        try:
            snapshot_download(
                repo_id,
                local_dir=str(tmp_dir),
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "*.ot"],
            )
            tok = AutoTokenizer.from_pretrained(str(tmp_dir), trust_remote_code=True)
            load_kw: dict[str, Any] = {"trust_remote_code": True}
            if use_cuda:
                load_kw["device_map"] = "auto"
            try:
                model = AutoModelForCausalLM.from_pretrained(str(tmp_dir), dtype=dt, **load_kw)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(str(tmp_dir), torch_dtype=dt, **load_kw)
            if not use_cuda:
                model = model.to(dev)
            model.eval()

            card = scan_model_xray(
                model,
                tok,
                repo_id=repo_id,
                probe_pack=pack,
                clinical_history=CLINICAL_HISTORY_SEEDS.get(repo_id),
                device=dev,
                batch_size=args.bs,
                loop_max_new=args.loop_max_new,
                skip_generation=args.skip_generation,
                wsi_commit=wsi_commit,
            )
            save_xray_json(card, out_file)
            _sync_file(out_file, sync_dir)
            errors.pop(repo_id, None)
            _atomic_write_text(errors_path, json.dumps(errors, indent=2))
            _sync_file(errors_path, sync_dir)
            n_bytes = out_file.stat().st_size
            oref = card["pathologies"]["over_refusal"].get("base_overref_rate")
            syc = card["pathologies"]["sycophancy"].get("cave_rate_false_pushback")
            print(
                f"[{idx}/{len(models)}] DONE {repo_id}: {n_bytes} B | "
                f"{card['identity']['arch_family']} | wall={time.perf_counter() - t0:.1f}s | "
                f"over_ref={oref} | syc_cave={syc}",
                flush=True,
            )
        except Exception as exc:
            tb = traceback.format_exc(limit=8)
            errors[repo_id] = {"error": f"{type(exc).__name__}: {exc}", "traceback_tail": tb[-1500:]}
            _atomic_write_text(errors_path, json.dumps(errors, indent=2))
            _sync_file(errors_path, sync_dir)
            print(f"[{idx}/{len(models)}] ERROR {repo_id}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            del model, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _clean_hf_cache_for_repo(repo_id)

    summary = summarize_xray_atlas(out_dir)
    summary_path = out_dir / "atlas_summary.json"
    md_path = out_dir / "atlas_biomarkers.md"
    _atomic_write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False))
    _atomic_write_text(md_path, summary.get("markdown_table", "") + "\n")
    _sync_file(summary_path, sync_dir)
    _sync_file(md_path, sync_dir)
    print(f"\n[atlas] Summarized {summary['n_models']} model(s); errors={len(errors)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
