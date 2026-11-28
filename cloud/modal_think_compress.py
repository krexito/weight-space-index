"""Run scripts/experiment_pocket_moe_think_compress.py on Modal.

    python -m modal run cloud/modal_think_compress.py --mode pilot     # 40-item Phase-0 probe, L4
    python -m modal run cloud/modal_think_compress.py --mode full      # full 7-arm run, H100

Generation caches live in the Modal volume `wsi-runs` under think_compress/cache (committed every
10 min, so a killed run resumes). The final JSON is copied to benchmarks/pocket_moe_think_compress/.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN = "think_compress"
SCRIPT = "/root/wsi/scripts/experiment_pocket_moe_think_compress.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("vllm", "huggingface_hub", "hf_transfer", "safetensors", "transformers", "datasets",
                 "numpy", "httpx", "pyyaml", "psutil")
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
app = modal.App("wsi-think-compress", image=image)

# Local dirs (basename keeps cache keys identical to HF ids); _bake_full_checkpoint needs a real dir.
TEACHER, STUDENT = "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B"
TEACHER_DIR, STUDENT_DIR = "/vol/models/Qwen3.5-4B", "/vol/models/Qwen3.5-0.8B"

COMMON = [
    "--teacher", TEACHER_DIR,
    "--student", STUDENT_DIR,
    "--max-new-tokens-nothink", "2048",
    "--max-new-tokens-think", "16384",
    "--max-teacher-train-tokens", "1800",
    "--max-think-context-tokens", "16896",
    "--total-steps", "160",
    "--pilot-n", "40",
    "--cache-dir", f"/vol/{RUN}/cache",
    "--backend", "auto",
]


def _run(extra: list[str], out: str) -> bytes:
    import os
    import subprocess
    import sys
    import threading

    from huggingface_hub import snapshot_download

    os.makedirs(f"/vol/{RUN}/cache", exist_ok=True)
    snapshot_download(TEACHER, local_dir=TEACHER_DIR)
    snapshot_download(STUDENT, local_dir=STUDENT_DIR)
    stop = threading.Event()

    def _periodic_commit():
        while not stop.wait(600):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[modal] volume commit failed: {e}")

    threading.Thread(target=_periodic_commit, daemon=True).start()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    rc = subprocess.run([sys.executable, "-u", SCRIPT, *COMMON, *extra, "--out", out], cwd="/root/wsi").returncode
    stop.set()
    vol.commit()
    if rc != 0 or not os.path.exists(out):
        raise RuntimeError(f"run failed (rc={rc})")
    return open(out, "rb").read()


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=3 * 3600)
def pilot() -> bytes:
    return _run(["--pilot-only"], f"/vol/{RUN}/pilot.json")


# Timeout caps worst-case spend (~4 h H100); caches survive, so a rerun resumes.
@app.function(gpu="H100", volumes={"/vol": vol}, timeout=4 * 3600)
def full() -> bytes:
    return _run([], f"/vol/{RUN}/full_16k.json")


@app.local_entrypoint()
def main(mode: str = "pilot"):
    fn, name = (pilot, "pilot_modal_l4.json") if mode == "pilot" else (full, "full_16k_modal_h100.json")
    data = fn.remote()
    dst = ROOT / "benchmarks" / "pocket_moe_think_compress" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    print(f"saved {dst}")
