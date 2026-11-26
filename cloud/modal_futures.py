"""Run scripts/experiment_futures.py gen on Modal (8 patched + 8 unpatched futures from every router cut, sensor on).

    set FU_WORK=<dir with config.json frozen.json diagnosis_blind.json blind_directions.npz requests_test.jsonl
                 requests_test_moi.jsonl requests_futures.jsonl sensor_directions_qwen35_08b_gsm8k.npz>
    python -m modal run cloud/modal_futures.py --mode smoke     # 16 requests per arm, L4 (checks that vLLM boots)
    python -m modal run cloud/modal_futures.py --mode full      # 1,512 requests per arm, L40S

Outputs go to the Modal volume `wsi-runs` under futures/<mode>/ (gen skips an arm whose outputs exist, so a rerun
resumes); fetch them with `modal volume get wsi-runs futures/full <local dir>`. Timeouts cap the spend.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
WORK = os.environ.get("FU_WORK", "")
REPO = "Qwen/Qwen3.5-0.8B"
BASE_DIR = "/vol/models/Qwen3.5-0.8B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install("vllm==0.34.0", "huggingface_hub", "hf_transfer", "safetensors", "numpy", "scipy")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1",
          "PYTHONPATH": "/root/wsi/scripts:/root/wsi", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    .add_local_dir(ROOT / "wsi", "/root/wsi/wsi")
    .add_local_dir(ROOT / "scripts", "/root/wsi/scripts", ignore=["__pycache__"])
)
if WORK:
    image = image.add_local_dir(WORK, "/root/work_in")

vol = modal.Volume.from_name("wsi-runs", create_if_missing=True)
app = modal.App("wsi-futures", image=image)


def _run(mode: str, limit: int) -> str:
    import shutil
    import subprocess
    import sys

    from huggingface_hub import snapshot_download

    snapshot_download(REPO, local_dir=BASE_DIR)
    work = f"/vol/futures/{mode}"
    if not os.path.exists(f"{work}/requests_futures.jsonl"):
        shutil.copytree("/root/work_in", work, dirs_exist_ok=True)
        vol.commit()
    import threading

    stop = threading.Event()

    def _periodic_commit():  # saved blocks reach the volume even if the timeout kills the container
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[modal] volume commit failed: {e}")

    threading.Thread(target=_periodic_commit, daemon=True).start()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    cmd = [sys.executable, "-u", "/root/wsi/scripts/experiment_futures.py", "gen", "--work", work,
           "--base-dir", BASE_DIR, "--dirs", f"{work}/sensor_directions_qwen35_08b_gsm8k.npz", "--ck-dir", "/tmp/ck"]
    if limit:
        cmd += ["--limit", str(limit)]
    rc = subprocess.run(cmd, cwd="/root/wsi").returncode
    stop.set()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"gen failed (rc={rc})")
    return work


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=25 * 60)
def smoke() -> str:
    return _run("smoke", 16)


# 1.5 h cap on L40S (~$1.95/h) = at most ~$2.9 of GPU time
@app.function(gpu="L40S", volumes={"/vol": vol}, timeout=90 * 60)
def full() -> str:
    return _run("full", 0)


@app.local_entrypoint()
def main(mode: str = "smoke"):
    assert WORK, "set FU_WORK to the local work dir"
    print("outputs in volume wsi-runs:", (smoke if mode == "smoke" else full).remote())
