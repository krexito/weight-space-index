"""Run Unified 3-Track `Qwen/Qwen3.5-0.8B` Max Comparison on Modal A100-40GB.

Usage:
    # 1. Fast smoke test (~2-3 min, 8 MATH-500 items):
    python -m modal run cloud/modal_qwen08b_max_comparison.py --limit 8

    # 2. Full 200-problem MATH-500 + 400 MMLU comparison:
    python -m modal run cloud/modal_qwen08b_max_comparison.py
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "qwen08b_max_comparison"
SCRIPT = "/root/wsi/scripts/experiment_qwen08b_max_comparison.py"
TEACHER_ID = "Qwen/Qwen3.5-4B"
STUDENT_ID = "Qwen/Qwen3.5-0.8B"
TEACHER_DIR = "/vol/models/Qwen3.5-4B"
STUDENT_DIR = "/vol/models/Qwen3.5-0.8B"
TEACHER_CACHE = "/vol/strata_quant_testA/teacher_math500_cache.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "accelerate",
        "numpy",
        "sympy",
        "antlr4-python3-runtime==4.11.1",
        "flash-linear-attention",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/vol/hf",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": "/root/wsi/scripts:/root/wsi",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(ROOT / "wsi", "/root/wsi/wsi", ignore=["__pycache__"])
    .add_local_dir(ROOT / "wsr", "/root/wsi/wsr", ignore=["__pycache__"])
    .add_local_dir(ROOT / "scripts", "/root/wsi/scripts", ignore=["__pycache__"])
    .add_local_file(
        ROOT / "patches" / "qwen35_0.8b_p50c_loop_fix.moipatch",
        "/root/wsi/patches/qwen35_0.8b_p50c_loop_fix.moipatch",
    )
)

vol = modal.Volume.from_name("wsi-runs", create_if_missing=True)
app = modal.App("wsi-qwen08b-max-comparison", image=image)


@app.function(gpu="A100-40GB", volumes={"/vol": vol}, timeout=180 * 60)
def run_experiment(limit: int = 0) -> bytes:
    import subprocess
    import sys
    from huggingface_hub import snapshot_download

    os.makedirs(f"/vol/{RUN}", exist_ok=True)
    if not os.path.exists(f"{STUDENT_DIR}/config.json"):
        snapshot_download(STUDENT_ID, local_dir=STUDENT_DIR)
    if not os.path.exists(TEACHER_CACHE):
        raise FileNotFoundError(
            f"Expected cached 4B teacher MATH-500 traces at {TEACHER_CACHE}. "
            "Run modal_pocket_moe_math500_strata_quant.py first or generate cache."
        )
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    import threading

    tag = f"smoke{limit}" if limit > 0 else "full"
    out_path = f"/vol/{RUN}/results_qwen08b_max_{tag}_modal_a100.json"
    ckpt_dir = f"/vol/{RUN}/ckpt_{tag}"
    os.makedirs(ckpt_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        SCRIPT,
        "--student",
        STUDENT_DIR,
        "--teacher",
        TEACHER_DIR,
        "--teacher-cache",
        TEACHER_CACHE,
        "--p50c-patch",
        "/root/wsi/patches/qwen35_0.8b_p50c_loop_fix.moipatch",
        "--ckpt-dir",
        ckpt_dir,
        "--eval-batch-size",
        "100",
        "--think-batch-size",
        "100",
        "--loopres-batch-size",
        "64",
        "--out",
        out_path,
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]

    stop_bg = threading.Event()
    threading.Thread(
        target=lambda: [vol.commit() for _ in iter(lambda: stop_bg.wait(60), True)],
        daemon=True,
    ).start()
    try:
        rc = subprocess.run(cmd, cwd="/root/wsi").returncode
    finally:
        stop_bg.set()
        vol.commit()
    if rc != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"qwen08b_max_comparison failed (rc={rc})")
    return open(out_path, "rb").read()


@app.local_entrypoint()
def main(limit: int = 0):
    data = run_experiment.remote(limit=limit)
    out_dir = ROOT / "benchmarks" / "qwen08b_max"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"smoke{limit}" if limit > 0 else "full"
    dst = out_dir / f"results_qwen08b_max_{tag}_modal_a100.json"
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    print(f"Saved {dst} ({len(data):,} bytes)")
