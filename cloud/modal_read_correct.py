"""Run scripts/experiment_read_correct.py prepare + gen + score on Modal (research/synthesis.md, test 1).

    python -m modal run cloud/modal_read_correct.py --limit 8     # smoke (8 problems)
    python -m modal run cloud/modal_read_correct.py               # all 500 MATH-500 problems

Model: Qwen/Qwen2.5-1.5B-Instruct (no thinking mode, ~55% on MATH-500, rarely loops). Outputs go to the Modal volume
`wsi-runs` under read_correct/<tag>/ (config, requests, outputs, feats.npz); fetch them with
`modal volume get wsi-runs read_correct/<tag>/<file> <local path>`. Analysis runs locally on CPU. 30 min cap on L4 or A10G (whichever is free first).
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install("vllm==0.34.0", "huggingface_hub", "hf_transfer", "safetensors", "numpy", "scipy")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1",
          "PYTHONPATH": "/root/wsi/scripts:/root/wsi"})
    .add_local_dir(ROOT / "wsi", "/root/wsi/wsi")
    .add_local_dir(ROOT / "scripts", "/root/wsi/scripts", ignore=["__pycache__"])
    .add_local_file(ROOT / "benchmarks/final_sensor/sensor_directions_qwen35_08b_gsm8k.npz",
                    "/root/wsi/benchmarks/final_sensor/sensor_directions_qwen35_08b_gsm8k.npz")
)
vol = modal.Volume.from_name("wsi-runs", create_if_missing=True)
app = modal.App("wsi-read-correct", image=image)


@app.function(gpu=["L4", "A10G"], volumes={"/vol": vol}, timeout=30 * 60)
def run(limit: int = 0) -> str:
    import subprocess
    import sys

    tag = f"smoke{limit}" if limit else "full"
    work = f"/vol/read_correct/{tag}"
    py = [sys.executable, "-u", "/root/wsi/scripts/experiment_read_correct.py"]
    lim = ["--limit", str(limit)] if limit else []
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    import os

    done = {"prepare": f"{work}/requests_test.jsonl", "gen": f"{work}/outputs_test_base.jsonl"}
    for step in (["prepare", "--work", work, "--model", MODEL] + lim,
                 ["gen", "--work", work, "--model", MODEL],
                 ["score", "--work", work, "--model", MODEL, "--out", f"{work}/feats.npz"]):
        if step[0] in done and os.path.exists(done[step[0]]):  # a rerun resumes after the last finished step
            print("skip", step[0], "(output exists)")
            continue
        rc = subprocess.run(py + step, cwd="/root/wsi").returncode
        vol.commit()
        if rc != 0:
            raise RuntimeError(f"{step[0]} failed (rc={rc})")
    return work


@app.local_entrypoint()
def main(limit: int = 0):
    print("outputs in volume wsi-runs:", run.remote(limit))
