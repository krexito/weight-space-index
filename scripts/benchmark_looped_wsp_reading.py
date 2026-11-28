"""Static Weight-Space Reading & Blind Decompilation Benchmark across LoopedWSP networks (CPU).

Discovers all `.npz` checkpoints under `--weights-dir` (both `networks/*.npz` and
`more_networks/*/weights.npz`), strips all naming/vocab/path metadata (`blind=True`),
and runs `wsi.looped_reader.analyze_looped_circuit` + `classify_looped_circuit` with
zero forward passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi.looped_reader import analyze_looped_circuit, classify_looped_circuit, load_looped_weights


def discover_weight_targets(weights_dir: Path) -> list[tuple[str, str, Path]]:
    """Return sorted list of `(group, task_label, weight_path)` tuples."""
    targets: list[tuple[str, str, Path]] = []
    # 1. Standard networks/*.npz
    nets_dir = weights_dir / "networks"
    if nets_dir.is_dir():
        for p in sorted(nets_dir.glob("*.npz")):
            targets.append(("networks", p.stem, p))
    # 2. more_networks/*/weights.npz
    more_dir = weights_dir / "more_networks"
    if more_dir.is_dir():
        for d in sorted(more_dir.iterdir()):
            w = d / "weights.npz"
            if d.is_dir() and w.exists():
                targets.append(("more_networks", d.name, w))
    # 3. Direct folder fallback (e.g. --weights-dir work/LoopedWSP/weights/networks)
    if not targets:
        for p in sorted(weights_dir.glob("*.npz")):
            targets.append((weights_dir.name, p.stem, p))
        for d in sorted(weights_dir.iterdir()):
            w = d / "weights.npz"
            if d.is_dir() and w.exists():
                targets.append((weights_dir.name, d.name, w))
    return targets


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _compact_entry(group: str, label: str, load_ms: float, analyze_ms: float, analysis: dict, clf: dict) -> dict[str, Any]:
    # Strip verbose per-matrix spectral table from summary while keeping per-role summary
    compact_analysis = {k: v for k, v in analysis.items() if k != "per_matrix_spectral"}
    return {
        "group": group,
        "ground_truth_name": label,
        "load_ms": round(load_ms, 2),
        "analyze_ms": round(analyze_ms, 2),
        "classification": clf,
        "analysis": compact_analysis,
    }


def run_benchmark(weights_dir: Path, out_path: Path) -> dict[str, Any]:
    targets = discover_weight_targets(weights_dir)
    if not targets:
        raise FileNotFoundError(f"No LoopedWSP weight files found under {weights_dir}")

    results: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()

    for group, label, wpath in targets:
        t0 = time.perf_counter()
        spec = load_looped_weights(wpath, blind=True)
        t1 = time.perf_counter()
        analysis = analyze_looped_circuit(spec)
        clf = classify_looped_circuit(analysis)
        t2 = time.perf_counter()
        results.append(
            _compact_entry(
                group=group,
                label=label,
                load_ms=(t1 - t0) * 1000.0,
                analyze_ms=(t2 - t1) * 1000.0,
                analysis=analysis,
                clf=clf,
            )
        )

    total_ms = (time.perf_counter() - total_t0) * 1000.0

    # Print Markdown table
    print("| Network | Group | d_model | L / H | Total Params | NNZ | Sparsity | Max QK | Step/Cancel Pairs | RO/Rec/Out Ch | Largest SCC | Blind Archetype | Analyze ms |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |")
    for r in results:
        an = r["analysis"]
        ar = an["architecture"]
        nat = an["anatomy"]
        gad = an["relu_gadgets"]
        df = an["dataflow"]
        att = an["attention"]
        clf = r["classification"]
        print(
            f"| `{r['ground_truth_name']}` | {r['group']} | {ar['d_model']} | "
            f"{ar['n_layers']}/{ar['n_heads']} | {nat['total_params']:,} | {nat['nonzero_params']:,} | "
            f"{nat['sparsity_ratio']:.2%} | {att['max_qk_scale']:.2g} | "
            f"{gad['step_threshold_pairs']}/{gad['cancellation_pairs']} | "
            f"{df['readonly_input_or_pos_count']}/{df['recurrent_scratchpad_count']}/{df['write_only_readout_count']} | "
            f"{df['largest_scc_size']} | `{clf['predicted_archetype']}` | {r['analyze_ms']:.1f} |"
        )

    payload = {
        "weights_dir": str(weights_dir),
        "n_networks": len(results),
        "total_elapsed_ms": round(total_ms, 2),
        "mean_analyze_ms": round(sum(r["analyze_ms"] for r in results) / len(results), 2),
        "networks": results,
    }
    atomic_json(out_path, payload)
    print(f"\nSaved {len(results)} network decompilations to {out_path} in {total_ms:.1f} ms")
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", default=str(ROOT / "work" / "LoopedWSP" / "weights"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "looped_wsp_reading" / "results_cpu.json"))
    a = ap.parse_args(argv)
    run_benchmark(Path(a.weights_dir), Path(a.out))


if __name__ == "__main__":
    main()
