"""Run scripts/experiment_refusal_controls.py on Modal (research/synthesis.md, test 2).

    python -m modal run cloud/modal_refusal_controls.py --limit 2 --max-new-tokens 16   # smoke
    python -m modal run cloud/modal_refusal_controls.py                                 # full (480 prompts x 6 arms)

Result JSON goes to the Modal volume `wsi-runs` at refusal_controls/<tag>.json. 40 min cap on L4 or A10G.
transformers pinned to 5.23.0 (LFM3 is remote code; 5.24 renamed the generation cache).
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", "transformers==5.23.0", "accelerate", "huggingface_hub", "hf_transfer", "safetensors",
                 "numpy", "pandas", "pyarrow", "scipy")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1",
          "PYTHONPATH": "/root/wsi/scripts:/root/wsi"})
    .add_local_dir(ROOT / "wsi", "/root/wsi/wsi", ignore=["__pycache__"])
    .add_local_dir(ROOT / "wsr" / "wsr", "/root/wsi/wsr", ignore=["__pycache__"])
    .add_local_dir(ROOT / "scripts", "/root/wsi/scripts", ignore=["__pycache__"])
    .add_local_file(ROOT / "patches/lfm3_1.7b_abliterated_sfw_routed.moipatch",
                    "/root/wsi/patches/lfm3_1.7b_abliterated_sfw_routed.moipatch")
)
vol = modal.Volume.from_name("wsi-runs", create_if_missing=True)
app = modal.App("wsi-refusal-controls", image=image)


@app.function(gpu=["L4", "A10G"], volumes={"/vol": vol}, timeout=40 * 60)
def run(limit: int = 0, max_new_tokens: int = 128) -> str:
    import os
    import subprocess
    import sys

    tag = f"smoke{limit}" if limit else "full"
    os.makedirs("/vol/refusal_controls", exist_ok=True)
    out = f"/vol/refusal_controls/{tag}.json"
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    cmd = [sys.executable, "-u", "/root/wsi/scripts/experiment_refusal_controls.py", "--model", "LiquidAI/LFM3-1.7B",
           "--out", out, "--max-new-tokens", str(max_new_tokens), "--batch-size", "32"] + (["--limit", str(limit)] if limit else [])
    rc = subprocess.run(cmd, cwd="/root/wsi").returncode
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"failed (rc={rc})")
    return out


# test 3 (scripts/experiment_refusal_select.py): 24 candidates x (screen + 64-token dev + 128-token exam); 75 min cap
@app.function(gpu=["L4", "A10G"], volumes={"/vol": vol}, timeout=75 * 60)
def select(limit: int = 0, extra: str = "", script: str = "experiment_refusal_select.py", name: str = "select") -> str:
    import os
    import subprocess
    import sys

    tag = f"{name}_smoke{limit}" if limit else f"{name}_full"
    os.makedirs("/vol/refusal_controls", exist_ok=True)
    out = f"/vol/refusal_controls/{tag}.json"
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    cmd = [sys.executable, "-u", f"/root/wsi/scripts/{script}", "--model", "LiquidAI/LFM3-1.7B",
           "--out", out] + (["--limit", str(limit)] if limit else []) + extra.split()

    import threading

    stop = threading.Event()
    threading.Thread(target=lambda: [vol.commit() for _ in iter(lambda: stop.wait(120), True)], daemon=True).start()
    rc = subprocess.run(cmd, cwd="/root/wsi").returncode
    stop.set()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"failed (rc={rc})")
    return out


@app.local_entrypoint()
def main(limit: int = 0, max_new_tokens: int = 128, test: str = "controls", extra: str = ""):
    if test == "select":
        print("result in volume wsi-runs:", select.remote(limit, extra))
    elif test == "compose":  # scripts/experiment_compose_shared.py (backlog #12 strong version)
        print("result in volume wsi-runs:", select.remote(limit, extra, "experiment_compose_shared.py", "compose"))
    else:
        print("result in volume wsi-runs:", run.remote(limit, max_new_tokens))
