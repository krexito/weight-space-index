"""Run WSP Closed-Form Router vs Shipped L24 Linear Router on LiquidAI/LFM3-1.7B (Modal L4)."""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "wsp_closed_form_router"
SCRIPT = "/root/wsi/scripts/experiment_wsp_closed_form_router_lfm3.py"
MODEL_ID = "LiquidAI/LFM3-1.7B"
MODEL_DIR = "/vol/models/LFM3-1.7B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==5.23.0",
        "datasets",
        "pandas",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "accelerate",
        "numpy",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/vol/hf",
        "PYTHONPATH": "/root/wsi/scripts:/root/wsi",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(ROOT / "wsi", "/root/wsi/wsi", ignore=["__pycache__"])
    .add_local_dir(ROOT / "wsr", "/root/wsi/wsr", ignore=["__pycache__"])
    .add_local_dir(ROOT / "scripts", "/root/wsi/scripts", ignore=["__pycache__"])
    .add_local_file(
        ROOT / "patches/lfm3_1.7b_abliterated_sfw_routed.moipatch",
        "/root/wsi/patches/lfm3_1.7b_abliterated_sfw_routed.moipatch",
    )
)

vol = modal.Volume.from_name("wsi-runs", create_if_missing=True)
app = modal.App("wsi-wsp-closed-form-router-lfm3", image=image)


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=45 * 60)
def run_experiment() -> bytes:
    import subprocess
    import sys
    from huggingface_hub import snapshot_download

    os.makedirs(f"/vol/{RUN}", exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    out_path = f"/vol/{RUN}/results_wsp_closed_form_router_lfm3_modal_l4.json"
    cmd = [
        sys.executable,
        "-u",
        SCRIPT,
        "--model",
        MODEL_DIR,
        "--patch",
        "/root/wsi/patches/lfm3_1.7b_abliterated_sfw_routed.moipatch",
        "--batch-size",
        "16",
        "--out",
        out_path,
    ]
    rc = subprocess.run(cmd, cwd="/root/wsi").returncode
    vol.commit()
    if rc != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"WSP router benchmark failed (rc={rc})")
    return open(out_path, "rb").read()


@app.local_entrypoint()
def main():
    data = run_experiment.remote()
    dst = ROOT / "benchmarks" / "wsp_router" / "results_wsp_closed_form_router_lfm3_modal_l4.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    print(f"saved {dst} ({len(data):,} bytes)")
