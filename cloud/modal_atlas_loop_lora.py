"""Run scripts/experiment_atlas_loop_lora_math500.py on Modal A100-40GB.

    python -m modal run cloud/modal_atlas_loop_lora.py

Results are saved atomically in the Modal volume `wsi-runs` under
`atlas_loop_lora/results_qwen25_3b_math500.json` and downloaded locally to
`benchmarks/atlas_loop_lora/results_qwen25_3b_math500.json`.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "atlas_loop_lora"
SCRIPT = "/root/wsi/scripts/experiment_atlas_loop_lora_math500.py"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_DIR = "/vol/models/Qwen2.5-3B-Instruct"

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
        "pyyaml",
        "psutil",
        "sympy",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/vol/hf",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": "/root/wsi/scripts:/root/wsi",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(ROOT / "wsi", "/root/wsi/wsi")
    .add_local_dir(ROOT / "wsr", "/root/wsi/wsr")
    .add_local_dir(ROOT / "scripts", "/root/wsi/scripts")
)

vol = modal.Volume.from_name("wsi-runs", create_if_missing=True)
app = modal.App("wsi-atlas-loop-lora", image=image)


@app.function(gpu="A100-40GB", volumes={"/vol": vol}, timeout=90 * 60)
def run_experiment() -> bytes:
    import subprocess
    import sys
    from huggingface_hub import snapshot_download

    os.makedirs(f"/vol/{RUN}", exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    out_path = f"/vol/{RUN}/results_qwen25_3b_math500.json"
    cmd = [
        sys.executable,
        "-u",
        SCRIPT,
        "--model",
        MODEL_DIR,
        "--loop-start",
        "20",
        "--loop-end",
        "25",
        "--loop-repeats",
        "2",
        "--rank",
        "16",
        "--total-steps",
        "160",
        "--grad-accum",
        "4",
        "--eval-batch-size",
        "48",
        "--out",
        out_path,
    ]
    rc = subprocess.run(cmd, cwd="/root/wsi").returncode
    vol.commit()
    if rc != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"run failed (rc={rc})")
    return open(out_path, "rb").read()


@app.local_entrypoint()
def main():
    data = run_experiment.remote()
    dst = ROOT / "benchmarks" / "atlas_loop_lora" / "results_qwen25_3b_math500.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    print(f"saved {dst}")
