"""Run Exp 1 (Pocket MoE r=8 Strata S2_K/S3_K cartridge compression) on Modal A10G."""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "strata_quant_exp1"
SCRIPT = "/root/wsi/scripts/experiment_pocket_moe_strata_quant.py"
TEACHER_ID = "Qwen/Qwen3.5-4B"
STUDENT_ID = "Qwen/Qwen3.5-0.8B"
TEACHER_DIR = "/vol/models/Qwen3.5-4B"
STUDENT_DIR = "/vol/models/Qwen3.5-0.8B"

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
)

vol = modal.Volume.from_name("wsi-runs", create_if_missing=True)
app = modal.App("wsi-pocket-moe-strata-quant", image=image)


@app.function(gpu="A10G", volumes={"/vol": vol}, timeout=90 * 60)
def run_experiment() -> bytes:
    import subprocess
    import sys
    from huggingface_hub import snapshot_download

    os.makedirs(f"/vol/{RUN}", exist_ok=True)
    snapshot_download(TEACHER_ID, local_dir=TEACHER_DIR)
    snapshot_download(STUDENT_ID, local_dir=STUDENT_DIR)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    out_path = f"/vol/{RUN}/results_pocket_moe_strata_quant_modal_a10g.json"
    cmd = [
        sys.executable,
        "-u",
        SCRIPT,
        "--teacher",
        TEACHER_DIR,
        "--student",
        STUDENT_DIR,
        "--eval-batch-size",
        "32",
        "--out",
        out_path,
    ]
    rc = subprocess.run(cmd, cwd="/root/wsi").returncode
    vol.commit()
    if rc != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"Exp 1 failed (rc={rc})")
    return open(out_path, "rb").read()


@app.local_entrypoint()
def main():
    data = run_experiment.remote()
    dst = ROOT / "benchmarks" / "strata_quant" / "results_pocket_moe_strata_quant_modal_a10g.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    print(f"saved {dst} ({len(data):,} bytes)")
