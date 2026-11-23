"""Batch Delta X-Ray scanner (`wsi_delta_xray_v1`) for Tome 2 of the WSI Model Atlas.

Scans controlled `(before -> after)` model transitions across ungated open-weight chains
(`OLMo-2 Base->SFT->DPO->Instruct`, `Qwen2.5-1.5B -> Instruct/Math/Math-Instruct/R1-Distill/Coder/Coder-Instruct`,
`Zamba2-2.7B Base->Instruct`, and `OLMoE-1B-7B-0125 Base->Instruct`).

Processes one family and one GPU model at a time: streams weights & gradients against partner
`*.safetensors` on disk, reuses any already-scanned Tome-1 `.xray.json` charts (`--existing-xray-dir`),
writes compact `.delta.xray.json` charts atomically (syncing to `--sync-dir`), and immediately deletes
temporary weights and HF cache after each family.

Usage:
    python scripts/scan_delta_xray_atlas.py --list-pairs
    python scripts/scan_delta_xray_atlas.py --out-dir /content/delta_xray_out --sync-dir /content/drive/MyDrive/wsi_runs/delta_xray_pilot
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
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scan_xray_atlas import (  # noqa: E402
    CLINICAL_HISTORY_SEEDS,
    RESERVED_HOLDOUT_MODELS,
    _atomic_write_text,
    _clean_hf_cache_for_repo,
    _git_short_commit,
    _is_valid_xray_file,
    _sync_file,
    repo_slug,
)
from wsi.delta_xray import (  # noqa: E402
    DELTA_XRAY_SCHEMA_VERSION,
    accumulate_domain_grad_dots,
    assemble_delta_attribution,
    build_delta_xray_card,
    build_domain_calibration_seqs,
    capture_domain_probe_state,
    compare_domain_probe_states,
    compute_weight_and_spectral_diff,
    load_delta_xray_json,
    save_delta_xray_json,
    summarize_delta_xray_atlas,
)
from wsi.xray import _discover_xray_structure, load_xray_json, save_xray_json, scan_model_xray  # noqa: E402
from wsi.xray_probes import build_frozen_probe_pack, load_frozen_probe_pack  # noqa: E402

# 11 controlled ungated transitions grouped smallest/fastest family first.
PILOT_DELTA_PAIRS: list[dict[str, str]] = [
    # 1) OLMo-2 1B sequential post-training chain (Base -> SFT -> DPO -> RLVR/Instruct)
    {
        "pair_id": "olmo2_base_to_sft",
        "family": "olmo2_1b",
        "transition_type": "sft",
        "before": "allenai/OLMo-2-0425-1B",
        "after": "allenai/OLMo-2-0425-1B-SFT",
    },
    {
        "pair_id": "olmo2_sft_to_dpo",
        "family": "olmo2_1b",
        "transition_type": "dpo",
        "before": "allenai/OLMo-2-0425-1B-SFT",
        "after": "allenai/OLMo-2-0425-1B-DPO",
    },
    {
        "pair_id": "olmo2_dpo_to_rlvr",
        "family": "olmo2_1b",
        "transition_type": "rlvr",
        "before": "allenai/OLMo-2-0425-1B-DPO",
        "after": "allenai/OLMo-2-0425-1B-Instruct",
    },
    # 2) Qwen2.5 1.5B multi-skill tree (Instruct / Math / Math-Instruct / R1-Distill / Coder / Coder-Instruct)
    {
        "pair_id": "qwen25_base_to_instruct",
        "family": "qwen25_1.5b",
        "transition_type": "instruct",
        "before": "Qwen/Qwen2.5-1.5B",
        "after": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    {
        "pair_id": "qwen25_base_to_math",
        "family": "qwen25_1.5b",
        "transition_type": "math_midtrain",
        "before": "Qwen/Qwen2.5-1.5B",
        "after": "Qwen/Qwen2.5-Math-1.5B",
    },
    {
        "pair_id": "qwen25_math_to_math_instruct",
        "family": "qwen25_1.5b",
        "transition_type": "math_instruct",
        "before": "Qwen/Qwen2.5-Math-1.5B",
        "after": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    },
    {
        "pair_id": "qwen25_math_to_r1_distill",
        "family": "qwen25_1.5b",
        "transition_type": "r1_distill",
        "before": "Qwen/Qwen2.5-Math-1.5B",
        "after": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    },
    {
        "pair_id": "qwen25_base_to_coder",
        "family": "qwen25_1.5b",
        "transition_type": "code_midtrain",
        "before": "Qwen/Qwen2.5-1.5B",
        "after": "Qwen/Qwen2.5-Coder-1.5B",
    },
    {
        "pair_id": "qwen25_coder_to_coder_instruct",
        "family": "qwen25_1.5b",
        "transition_type": "code_instruct",
        "before": "Qwen/Qwen2.5-Coder-1.5B",
        "after": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    },
    # 3) Zamba2 2.7B Mamba2 + shared attention hybrid
    {
        "pair_id": "zamba2_base_to_instruct",
        "family": "zamba2_2.7b",
        "transition_type": "instruct",
        "before": "Zyphra/Zamba2-2.7B",
        "after": "Zyphra/Zamba2-2.7B-instruct",
    },
    # 4) OLMoE 1B-7B sparse MoE
    {
        "pair_id": "olmoe_base_to_instruct",
        "family": "olmoe_1b7b",
        "transition_type": "instruct",
        "before": "allenai/OLMoE-1B-7B-0125",
        "after": "allenai/OLMoE-1B-7B-0125-Instruct",
    },
]


def _is_valid_delta_xray_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        data = load_delta_xray_json(path)
        return data.get("schema") == DELTA_XRAY_SCHEMA_VERSION and isinstance(data.get("pair"), dict)
    except Exception:
        return False


def _validate_no_holdout_pairs(pairs: list[dict[str, str]], *, allow_holdout: bool) -> None:
    if allow_holdout:
        return
    holdout_set = {m.lower() for m in RESERVED_HOLDOUT_MODELS}
    blocked: list[str] = []
    for p in pairs:
        for role in ("before", "after"):
            repo = str(p[role])
            if repo.lower() in holdout_set and repo not in blocked:
                blocked.append(repo)
    if blocked:
        raise ValueError(f"Refusing to scan hold-out A/B model(s) {blocked} into Delta Atlas without --allow-holdout")


def _resolve_target_pairs(
    raw: str | None,
    *,
    allow_holdout: bool = False,
    pair_catalog: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    catalog = list(pair_catalog) if pair_catalog is not None else list(PILOT_DELTA_PAIRS)
    if not raw or raw.strip().lower() == "all":
        chosen = catalog
    else:
        by_id = {p["pair_id"]: p for p in catalog}
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        chosen = []
        seen: set[str] = set()
        for tok in tokens:
            if tok in by_id:
                p = by_id[tok]
                if p["pair_id"] not in seen:
                    chosen.append(p)
                    seen.add(p["pair_id"])
            else:
                fam_hits = [p for p in catalog if p["family"] == tok or p["transition_type"] == tok]
                if not fam_hits:
                    raise ValueError(f"Unknown delta pair_id / family / transition_type: {tok!r}")
                for p in fam_hits:
                    if p["pair_id"] not in seen:
                        chosen.append(p)
                        seen.add(p["pair_id"])
    _validate_no_holdout_pairs(chosen, allow_holdout=allow_holdout)
    return chosen


def _ordered_unique_repos(pairs: list[dict[str, str]]) -> list[str]:
    """Order unique checkpoints topologically (`before` before `after`) within a family."""
    repos: list[str] = []
    seen: set[str] = set()
    for p in pairs:
        for role in ("before", "after"):
            r = p[role]
            if r not in seen:
                repos.append(r)
                seen.add(r)
    return repos


def _maybe_load_existing_xray(
    repo_id: str,
    xray_cache_dir: Path,
    existing_xray_dirs: list[Path],
    sync_xray_dir: Path | None,
) -> dict[str, Any] | None:
    fname = f"{repo_slug(repo_id)}.xray.json"
    local_p = xray_cache_dir / fname
    if _is_valid_xray_file(local_p):
        _sync_file(local_p, sync_xray_dir)
        return load_xray_json(local_p)
    for ext_dir in existing_xray_dirs:
        cand = ext_dir / fname
        if _is_valid_xray_file(cand):
            card = load_xray_json(cand)
            save_xray_json(card, local_p)
            _sync_file(local_p, sync_xray_dir)
            return card
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help="Output directory for *.delta.xray.json and summary artifacts")
    ap.add_argument("--sync-dir", default=None, help="Optional Google Drive / backup directory synced after every pair")
    ap.add_argument("--existing-xray-dir", action="append", default=[], help="Optional Tome-1 *.xray.json directory to reuse")
    ap.add_argument("--probes", default=None, help="Path to frozen xray_probes_v1.json (created if missing)")
    ap.add_argument("--pairs", default=None, help="Optional comma-separated pair_id, family, or transition_type filter")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--loop-max-new", type=int, default=384)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--no-attribution", action="store_true", help="Skip trapezoidal gradient attribution")
    ap.add_argument("--grad-max-len", type=int, default=128)
    ap.add_argument("--force", action="store_true", help="Re-scan pairs even if valid .delta.xray.json already exists")
    ap.add_argument("--allow-holdout", action="store_true", help="Allow scanning RESERVED_HOLDOUT_MODELS")
    ap.add_argument("--list-pairs", action="store_true", help="Print the 11 pilot delta pairs and holdout check, then exit")
    args = ap.parse_args(argv)

    all_repos = sorted({p[r] for p in PILOT_DELTA_PAIRS for r in ("before", "after")})
    overlap = sorted(set(all_repos) & set(RESERVED_HOLDOUT_MODELS))
    if args.list_pairs:
        families = sorted({p["family"] for p in PILOT_DELTA_PAIRS})
        print(
            json.dumps(
                {
                    "pair_count": len(PILOT_DELTA_PAIRS),
                    "unique_repo_count": len(all_repos),
                    "families": families,
                    "pairs": PILOT_DELTA_PAIRS,
                    "reserved_holdout_models": list(RESERVED_HOLDOUT_MODELS),
                    "overlap": overlap,
                },
                indent=2,
            )
        )
        return 0

    if not args.out_dir:
        ap.error("--out-dir is required unless --list-pairs is set")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[env] torch={torch.__version__} | transformers={transformers.__version__}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xray_cache_dir = out_dir / "xray_cards"
    xray_cache_dir.mkdir(parents=True, exist_ok=True)
    sync_dir = Path(args.sync_dir) if args.sync_dir else None
    sync_xray_dir = (sync_dir / "xray_cards") if sync_dir else None
    existing_xray_dirs = [Path(p) for p in args.existing_xray_dir if p]

    pairs = _resolve_target_pairs(args.pairs, allow_holdout=args.allow_holdout)
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

    errors_path = out_dir / "delta_scan_errors.json"
    errors: dict[str, Any] = json.loads(errors_path.read_text(encoding="utf-8")) if errors_path.exists() else {}

    def _load_hf_model(model_dir: Path, **extra_kw: Any):
        kw = {"trust_remote_code": True, "low_cpu_mem_usage": True, **extra_kw}
        try:
            return AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dt, **kw)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dt, **kw)

    # Group pending pairs by family (preserving declaration order).
    family_order: list[str] = []
    by_family: dict[str, list[dict[str, str]]] = {}
    for p in pairs:
        fam = p["family"]
        if fam not in by_family:
            by_family[fam] = []
            family_order.append(fam)
        by_family[fam].append(p)

    for fam in family_order:
        fam_pairs = by_family[fam]
        pending: list[dict[str, str]] = []
        for p in fam_pairs:
            dst = out_dir / f"{p['pair_id']}.delta.xray.json"
            if not args.force and _is_valid_delta_xray_file(dst):
                print(f"[SKIP cached] {p['pair_id']} ({dst.stat().st_size} B)", flush=True)
                _sync_file(dst, sync_dir)
            else:
                pending.append(p)
        if not pending:
            continue

        unique_repos = _ordered_unique_repos(pending)
        print(f"\n=== FAMILY START: {fam} ({len(pending)} pair(s), {len(unique_repos)} checkpoint(s)) ===", flush=True)
        fam_dirs: dict[str, Path] = {}
        try:
            for repo_id in unique_repos:
                slug = repo_slug(repo_id)
                tmp_d = Path(tempfile.mkdtemp(prefix=f"wsi_dx_{slug}_"))
                fam_dirs[repo_id] = tmp_d
                print(f"  [download] {repo_id} -> {tmp_d}", flush=True)
                snapshot_download(
                    repo_id,
                    local_dir=str(tmp_d),
                    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "*.ot"],
                )

            repo_cards: dict[str, dict[str, Any]] = {}
            pair_state: dict[str, dict[str, Any]] = {}

            for r_idx, repo_id in enumerate(unique_repos, 1):
                before_jobs = [p for p in pending if p["before"] == repo_id]
                after_jobs = [p for p in pending if p["after"] == repo_id]
                repo_dir = fam_dirs[repo_id]
                print(f"\n  [{fam} {r_idx}/{len(unique_repos)}] LOAD {repo_id} (before={len(before_jobs)}, after={len(after_jobs)})", flush=True)
                model = None
                tok = None
                try:
                    tok = AutoTokenizer.from_pretrained(str(repo_dir), trust_remote_code=True)
                    model = _load_hf_model(repo_dir)
                    if use_cuda:
                        try:
                            model = model.to(dev)
                        except torch.cuda.OutOfMemoryError:
                            del model
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = _load_hf_model(repo_dir, device_map="auto")
                    else:
                        model = model.to(dev)
                    model.eval()

                    card = _maybe_load_existing_xray(repo_id, xray_cache_dir, existing_xray_dirs, sync_xray_dir)
                    if card is None:
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
                        card_p = xray_cache_dir / f"{repo_slug(repo_id)}.xray.json"
                        save_xray_json(card, card_p)
                        _sync_file(card_p, sync_xray_dir)
                        print(f"    [xray scanned] {repo_id}: {card_p.stat().st_size} B", flush=True)
                    else:
                        print(f"    [xray reused]  {repo_id}", flush=True)
                    repo_cards[repo_id] = card

                    # 1) As `before` partner: capture domain probe state, streaming weight/spectral diff, and forward grad dots.
                    if before_jobs:
                        struct_b = _discover_xray_structure(model, tok, repo_id=repo_id)
                        n_layers = len(struct_b.blocks)
                        dom_seqs_self = build_domain_calibration_seqs(tok, pack, max_length=256)
                        st_b = capture_domain_probe_state(model, dom_seqs_self, device=dev)
                        for p in before_jobs:
                            pid = p["pair_id"]
                            partner_dir = fam_dirs[p["after"]]
                            try:
                                t_pair = time.perf_counter()
                                weight_diff, spectral_diff = compute_weight_and_spectral_diff(
                                    repo_dir, partner_dir, struct_b, device=dev
                                )
                                dots_b: dict[str, dict[str, float]] = {}
                                gnll_b: dict[str, float] = {}
                                pf_cost: dict[str, float] = {}
                                used_len = int(args.grad_max_len)
                                if not args.no_attribution:
                                    dots_b, gnll_b, pf_cost, used_len = accumulate_domain_grad_dots(
                                        model,
                                        partner_dir,
                                        dom_seqs_self,
                                        sign=+1.0,
                                        device=dev,
                                        max_len=args.grad_max_len,
                                        compute_prose_fisher=True,
                                    )
                                pair_state[pid] = {
                                    "t_accum": time.perf_counter() - t_pair,
                                    "n_layers": n_layers,
                                    "struct_b_identity": dict(struct_b.identity),
                                    "dom_seqs": dom_seqs_self,
                                    "st_b": st_b,
                                    "weight_diff": weight_diff,
                                    "spectral_diff": spectral_diff,
                                    "dots_b": dots_b,
                                    "gnll_b": gnll_b,
                                    "pf_cost": pf_cost,
                                    "used_len": used_len,
                                }
                                print(
                                    f"    [before ready] {pid}: rel_fro={weight_diff['total_rel_fro']:.4f}, "
                                    f"used_len={used_len}",
                                    flush=True,
                                )
                            except Exception as exc:
                                errors[pid] = {
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback_tail": traceback.format_exc(limit=8)[-1500:],
                                }
                                _atomic_write_text(errors_path, json.dumps(errors, indent=2))
                                _sync_file(errors_path, sync_dir)
                                print(f"    [ERROR before] {pid}: {type(exc).__name__}: {exc}", flush=True)
                        del struct_b

                    # 2) As `after` partner: compare domain probe state, run backward grad dots, and write final delta card.
                    for p in after_jobs:
                        pid = p["pair_id"]
                        st = pair_state.pop(pid, None)
                        if st is None:
                            continue
                        partner_dir = fam_dirs[p["before"]]
                        try:
                            t_a0 = time.perf_counter()
                            dom_seqs = st["dom_seqs"]
                            domain_shifts = compare_domain_probe_states(st.pop("st_b"), model, dom_seqs, device=dev)
                            attr = None
                            if not args.no_attribution:
                                dots_a, gnll_a, _, _ = accumulate_domain_grad_dots(
                                    model,
                                    partner_dir,
                                    dom_seqs,
                                    sign=-1.0,
                                    device=dev,
                                    max_len=st["used_len"],
                                    compute_prose_fisher=False,
                                )
                                attr = assemble_delta_attribution(
                                    st["dots_b"],
                                    dots_a,
                                    st["gnll_b"],
                                    gnll_a,
                                    st["pf_cost"],
                                    int(st["n_layers"]),
                                    int(st["used_len"]),
                                )
                            wall_s = float(st["t_accum"]) + (time.perf_counter() - t_a0)
                            # Reconstruct minimal structure wrapper for identity fields without pinning modules.
                            class _IdentHolder:
                                def __init__(self, ident: dict[str, Any], n_l: int) -> None:
                                    self.identity = ident
                                    self.blocks = [None] * n_l

                            delta_card = build_delta_xray_card(
                                pair_id=pid,
                                before_repo=p["before"],
                                after_repo=p["after"],
                                transition_type=p["transition_type"],
                                family=p["family"],
                                struct_before=_IdentHolder(st["struct_b_identity"], st["n_layers"]),  # type: ignore[arg-type]
                                card_before=repo_cards[p["before"]],
                                card_after=card,
                                weight_diff=st["weight_diff"],
                                spectral_diff=st["spectral_diff"],
                                domain_shifts=domain_shifts,
                                attribution=attr,
                                probe_pack_sha256=str(pack.get("probe_pack_sha256", "")),
                                wall_s=wall_s,
                                wsi_commit=wsi_commit,
                            )
                            dst = out_dir / f"{pid}.delta.xray.json"
                            save_delta_xray_json(delta_card, dst)
                            _sync_file(dst, sync_dir)
                            errors.pop(pid, None)
                            _atomic_write_text(errors_path, json.dumps(errors, indent=2))
                            _sync_file(errors_path, sync_dir)
                            sz = dst.stat().st_size
                            print(
                                f"    [PAIR DONE] {pid}: {sz} B | rel_fro={delta_card['weight_diff']['total_rel_fro']} | "
                                f"rank80={delta_card['spectral_diff']['mlp_out'].get('rank80_median')} | "
                                f"wall={wall_s:.1f}s",
                                flush=True,
                            )
                        except Exception as exc:
                            errors[pid] = {
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback_tail": traceback.format_exc(limit=8)[-1500:],
                            }
                            _atomic_write_text(errors_path, json.dumps(errors, indent=2))
                            _sync_file(errors_path, sync_dir)
                            print(f"    [ERROR after] {pid}: {type(exc).__name__}: {exc}", flush=True)
                except Exception as exc:
                    for p in before_jobs + after_jobs:
                        pid = p["pair_id"]
                        errors[pid] = {
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback_tail": traceback.format_exc(limit=8)[-1500:],
                        }
                    _atomic_write_text(errors_path, json.dumps(errors, indent=2))
                    _sync_file(errors_path, sync_dir)
                    print(f"    [ERROR model] {repo_id}: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    del model, tok
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            for repo_id, d in fam_dirs.items():
                shutil.rmtree(d, ignore_errors=True)
                _clean_hf_cache_for_repo(repo_id)

    summary = summarize_delta_xray_atlas(out_dir)
    summary_path = out_dir / "delta_atlas_summary.json"
    md_path = out_dir / "delta_atlas_table.md"
    _atomic_write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False))
    _atomic_write_text(md_path, summary.get("markdown_table", "") + "\n")
    _sync_file(summary_path, sync_dir)
    _sync_file(md_path, sync_dir)
    print(f"\n[delta-atlas] Summarized {summary['n_pairs']} pair(s); errors={len(errors)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
