"""Run Exp 2A (X-Ray + Strata quantization on Qwen2.5-3B-Instruct) on Modal A100-40GB."""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "strata_quant_exp2a"
SCRIPT = "/root/wsi/scripts/experiment_xray_strata_quant_dense.py"
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
        ROOT / "benchmarks/xray_atlas/tome1_xray/qwen__qwen2.5-3b-instruct.xray.json",
        "/root/wsi/benchmarks/xray_atlas/tome1_xray/qwen__qwen2.5-3b-instruct.xray.json",
    )
)

vol = modal.Volume.from_name("wsi-runs", create_if_missing=True)
app = modal.App("wsi-xray-strata-quant-dense", image=image)


@app.function(gpu="A100-40GB", volumes={"/vol": vol}, timeout=90 * 60)
def run_experiment() -> bytes:
    import subprocess
    import sys
    from huggingface_hub import snapshot_download

    os.makedirs(f"/vol/{RUN}", exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    out_path = f"/vol/{RUN}/results_xray_strata_quant_qwen25_3b_modal_a100.json"
    cmd = [
        sys.executable,
        "-u",
        SCRIPT,
        "--model",
        MODEL_DIR,
        "--xray",
        "/root/wsi/benchmarks/xray_atlas/tome1_xray/qwen__qwen2.5-3b-instruct.xray.json",
        "--out",
        out_path,
    ]
    rc = subprocess.run(cmd, cwd="/root/wsi").returncode
    vol.commit()
    if rc != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"Exp 2A failed (rc={rc})")
    return open(out_path, "rb").read()


@app.local_entrypoint()
def main():
    data = run_experiment.remote()
    dst = ROOT / "benchmarks" / "strata_quant" / "results_xray_strata_quant_qwen25_3b_modal_a100.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    print(f"saved {dst} ({len(data):,} bytes)")
