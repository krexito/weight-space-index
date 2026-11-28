"""Open-prompt Blind Agent A/B on Modal (L4): raw PyTorch vs WSI (25 MCP + Surgical Telemetry + 3-Tome Atlas).

Target model: `microsoft/Phi-4-mini-instruct` (reserved hold-out, 0 leakage with the 3-Tome Atlas).
Agent model:  `deepseek/deepseek-v4.1-flash-1017` via OpenRouter.

Usage:
    # 1. Preflight check (verifies GPU, holdout census, X-Ray, both sandboxes, 25 MCP tools, OpenRouter auth; ~$0):
    set OPENROUTER_API_KEY=sk-or-v1-...
    python -m modal run cloud/modal_blind_agent_open_ab_phi4.py --preflight-only

    # 2. Full A/B run (reuses cached holdout + X-Ray from preflight):
    python -m modal run cloud/modal_blind_agent_open_ab_phi4.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = "blind_agent_open_ab_phi4"
MODEL_ID = "microsoft/Phi-4-mini-instruct"
AGENT_MODEL = "deepseek/deepseek-v4.1-flash-1017"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("procps", "git")
    .pip_install(
        "torch==2.10.0",
        "transformers==5.23.0",
        "accelerate",
        "safetensors",
        "datasets",
        "pandas",
        "pyarrow",
        "huggingface_hub",
        "hf_transfer",
        "einops",
        "numpy",
        "scipy",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/vol/hf",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root/wsi/scripts:/root/wsi",
    })
    .add_local_dir(ROOT / "wsi", "/root/wsi/wsi", ignore=["__pycache__"])
    .add_local_dir(ROOT / "wsr", "/root/wsi/wsr", ignore=["__pycache__"])
    .add_local_dir(ROOT / "scripts", "/root/wsi/scripts", ignore=["__pycache__"])
    .add_local_dir(ROOT / "benchmarks" / "xray_atlas", "/root/wsi/benchmarks/xray_atlas")
    .add_local_dir(ROOT / "benchmarks" / "blind_agent_open_phi4", "/root/wsi/benchmarks/blind_agent_open_phi4")
    .add_local_dir(ROOT / "benchmarks" / "blind_ab_3tomes", "/root/wsi/benchmarks/blind_ab_3tomes")
)

vol = modal.Volume.from_name("wsi-runs", create_if_missing=True)
app = modal.App("wsi-blind-agent-open-ab-phi4", image=image)

_CAPTURE_STUB = '''_EXCLUDED_BACKBONE_TOKENS = (
    "mtp", "visual", "vision", "vision_tower", "vision_model", "image_encoder",
    "mm_projector", "multi_modal_projector", "speculator", "dspark", "audio",
)
def _is_excluded_backbone_path(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in _EXCLUDED_BACKBONE_TOKENS)
def _is_routed_expert_path(name: str) -> bool:
    return "experts" in name.lower().split(".")
'''


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=130 * 60)
def run_ab(
    api_key: str,
    preflight_only: bool = False,
    agent_min: int = 45,
    agent_model: str = AGENT_MODEL,
) -> bytes:
    import datetime
    import glob
    import shutil
    import subprocess
    import sys
    import threading
    import time
    import urllib.request

    REPO = "/root/wsi"
    VOL_RUN = f"/vol/{RUN_NAME}"
    VOL_BASE = "/vol/models/Phi-4-mini-instruct"
    W = "/content/work"
    BASE = "/content/base"
    AG = "/content/agent"
    HARNESS = "/content/harness"

    os.makedirs(VOL_RUN, exist_ok=True)
    os.makedirs(W, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"], check=False)

    # Restore any cached work artifacts from Modal volume
    for f in glob.glob(f"{VOL_RUN}/*"):
        if os.path.isfile(f):
            shutil.copy2(f, W)

    # Symlink /content/base -> cached model dir on volume if already downloaded, else download to /vol/models/...
    os.makedirs(VOL_BASE, exist_ok=True)
    if os.path.islink(BASE) or os.path.exists(BASE):
        if os.path.islink(BASE):
            os.unlink(BASE)
        else:
            shutil.rmtree(BASE, ignore_errors=True)
    os.symlink(VOL_BASE, BASE)

    ENV = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO}/scripts"}
    ENV.pop("OPENROUTER_API_KEY", None)
    AB_SCRIPT = f"{REPO}/scripts/run_blind_ab_3tomes.py"
    JUDGE_SCRIPT = f"{REPO}/scripts/eval_refusal_blind.py"

    def sync() -> None:
        for f in glob.glob(f"{W}/*.json*") + glob.glob(f"{W}/*.moipatch") + glob.glob(f"{W}/*.md"):
            shutil.copy2(f, VOL_RUN)
        vol.commit()

    def run_step(cmd: list[str], env: dict | None = None, timeout: float | None = None) -> None:
        print("\n>>>", " ".join(cmd), flush=True)
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env or ENV
        )
        t0 = time.time()
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="", flush=True)
            if timeout and time.time() - t0 > timeout:
                p.kill()
                raise RuntimeError(f"timeout ({timeout} s)")
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Step failed (rc={rc}): {' '.join(cmd)}")

    def build_standalone_moipatch(src_path: str, dst_path: str) -> None:
        src = Path(src_path).read_text(encoding="utf-8")
        needle = "from .capture import _is_excluded_backbone_path, _is_routed_expert_path"
        assert needle in src, "Expected relative import missing in wsi/moipatch.py"
        Path(dst_path).write_text(src.replace(needle, _CAPTURE_STUB, 1), encoding="utf-8")

    # 1. Verify OpenRouter API key & model slug with a 1-token ping BEFORE any heavy work
    print(f"[preflight] Checking OpenRouter auth + model `{agent_model}`...", flush=True)
    req_body = json.dumps({
        "model": agent_model,
        "messages": [{"role": "user", "content": "Reply with OK"}],
        "max_tokens": 2,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=req_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        ping = json.loads(resp.read().decode("utf-8"))
    ping_txt = (ping.get("choices") or [{}])[0].get("message", {}).get("content", "")
    print(f"[preflight] OpenRouter OK (model={ping.get('model')}, reply={ping_txt!r})", flush=True)

    # 2. Select holdout, prepare disjoint splits, run census, scan target_base.xray.json
    if not os.path.exists(f"{W}/selected_holdout.json"):
        run_step([
            sys.executable, "-u", AB_SCRIPT, "select-holdout",
            "--work-dir", W,
            "--base-dir", VOL_BASE,
            "--candidates", MODEL_ID,
        ])
        sync()
    selected = json.loads(Path(f"{W}/selected_holdout.json").read_text(encoding="utf-8"))
    census = selected["census"]
    print(f"[preflight] Holdout locked: {selected['selected_model']} | census={census}", flush=True)
    assert census.get("passed"), f"Holdout census failed: {census}"

    # Ensure unprivileged Linux user `agent` exists
    if subprocess.run(["id", "-u", "agent"], capture_output=True).returncode != 0:
        subprocess.run(["useradd", "-m", "-s", "/bin/bash", "agent"], check=True)

    def as_agent(cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(["runuser", "-u", "agent", "--", "bash", "-lc", cmd], capture_output=True, text=True)

    def stage_and_verify_sandbox(mode: str) -> dict:
        shutil.rmtree(AG, ignore_errors=True)
        shutil.rmtree(HARNESS, ignore_errors=True)
        os.makedirs(f"{AG}/data", exist_ok=True)
        os.makedirs(f"{AG}/out", exist_ok=True)
        os.makedirs(HARNESS, exist_ok=True)

        for f in glob.glob(f"{W}/agent_*.jsonl"):
            shutil.copy2(f, f"{AG}/data")

        if mode == "raw_pytorch":
            build_standalone_moipatch(f"{REPO}/wsi/moipatch.py", f"{AG}/moipatch.py")
            shutil.copy2(f"{REPO}/benchmarks/blind_agent_open_phi4/TASK_RAW.md", f"{AG}/TASK.md")
        elif mode == "wsi_full":
            run_step([
                sys.executable, "-u", AB_SCRIPT, "stage-workspace",
                "--arm", "wsi_plus_atlas",
                "--agent-dir", AG,
                "--repo-dir", REPO,
                "--work-dir", W,
                "--time-limit-min", str(agent_min),
            ])
            shutil.copy2(f"{REPO}/benchmarks/blind_agent_open_phi4/TASK_WSI_FULL.md", f"{AG}/TASK.md")
        else:
            raise ValueError(mode)

        for f in ("live_agent.py", "live_agent_repl.py"):
            shutil.copy2(f"{REPO}/scripts/{f}", HARNESS)

        # Lock down permissions: agent can read /content/base (VOL_BASE), /content/harness, and own /content/agent,
        # but CANNOT read /root, /content/work, or /vol/{RUN_NAME}.
        subprocess.run(["chown", "-R", "agent:agent", AG], check=True)
        subprocess.run(["chmod", "755", "/vol", "/vol/models"], check=True)
        subprocess.run(["chmod", "-R", "a+rX", VOL_BASE, HARNESS], check=True)
        subprocess.run(["chmod", "700", REPO, W, VOL_RUN, "/root"], check=True)

        # Verify GPU, model access, isolation, and tools as `agent`
        r_gpu = as_agent(f"{sys.executable} -c 'import torch; print(torch.cuda.is_available())'")
        assert r_gpu.stdout.strip() == "True", f"GPU not visible to agent: {r_gpu.stdout} {r_gpu.stderr}"
        assert as_agent(f"ls {BASE}/config.json").returncode == 0, "agent cannot read /content/base/config.json"
        for secret_p in (REPO, W, VOL_RUN, "/root"):
            assert as_agent(f"ls {secret_p}").returncode != 0, f"LEAK: agent can read {secret_p}"

        info = {"mode": mode, "gpu_ok": True, "isolation_ok": True}
        if mode == "raw_pytorch":
            r_mp = as_agent(f"PYTHONPATH={AG} {sys.executable} -c 'import moipatch; print(moipatch.FORMAT)'")
            assert r_mp.returncode == 0, f"moipatch failed: {r_mp.stderr}"
            assert as_agent(f"PYTHONPATH={AG} {sys.executable} -c 'import wsi'").returncode != 0, "LEAK: wsi in raw_pytorch"
            assert as_agent(f"ls {AG}/atlas").returncode != 0, "LEAK: atlas in raw_pytorch"
            info["moipatch_format"] = r_mp.stdout.strip()
        else:
            r_wsi = as_agent(
                f"PYTHONPATH={AG} {sys.executable} -c 'import wsi, wsi.surgery_telemetry, wsi.atlas; print(\"ok\")'"
            )
            assert r_wsi.returncode == 0, f"wsi import failed: {r_wsi.stderr}"
            assert as_agent(f"ls {AG}/atlas/target_diagnosis.md").returncode == 0, "target_diagnosis.md missing"
            # Verify live MCP server startup & tool count as `agent`
            r_mcp = as_agent(
                f"PYTHONPATH={AG}:{HARNESS} {sys.executable} -c \""
                "import json, subprocess, sys; "
                "p = subprocess.Popen([sys.executable, '-u', '-m', 'wsi', 'mcp'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True); "
                "p.stdin.write(json.dumps({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'check','version':'0.1'}}})+'\\n'); "
                "p.stdin.write(json.dumps({'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}})+'\\n'); p.stdin.flush(); "
                "r1 = json.loads(p.stdout.readline()); r2 = json.loads(p.stdout.readline()); p.kill(); "
                "names = [t['name'] for t in r2['result']['tools']]; "
                "assert 'wsi_diagnose_surgery_sites' in names and 'wsi_screen_candidates' in names; "
                "print(len(names))\""
            )
            assert r_mcp.returncode == 0, f"MCP check failed: {r_mcp.stdout} {r_mcp.stderr}"
            info["mcp_tool_count"] = int(r_mcp.stdout.strip())
        print(f"[sandbox] Verified OK: {info}", flush=True)
        return info

    if preflight_only:
        pf_raw = stage_and_verify_sandbox("raw_pytorch")
        pf_wsi = stage_and_verify_sandbox("wsi_full")
        pf = {
            "preflight_ok": True,
            "model": MODEL_ID,
            "agent_model": agent_model,
            "openrouter_ping_model": ping.get("model"),
            "census": census,
            "sandbox_raw": pf_raw,
            "sandbox_wsi": pf_wsi,
        }
        out_p = Path(W) / "preflight.json"
        out_p.write_text(json.dumps(pf, indent=2), encoding="utf-8")
        sync()
        return out_p.read_bytes()

    # Background periodic volume commit during long agent runs
    stop_bg = threading.Event()
    threading.Thread(
        target=lambda: [sync() for _ in iter(lambda: stop_bg.wait(180), True)],
        daemon=True,
    ).start()

    def run_arm(arm_name: str, mode: str, enable_wsi_mcp: bool) -> dict:
        ev_path = Path(W) / f"eval_{arm_name}.json"
        if ev_path.exists():
            print(f"[{arm_name}] Already evaluated; reusing cached result.", flush=True)
            return json.loads(ev_path.read_text(encoding="utf-8"))

        stage_and_verify_sandbox(mode)

        patch_dst = Path(W) / f"{arm_name}.moipatch"
        if not patch_dst.exists():
            env = {
                **ENV,
                "PYTHONPATH": f"{AG}:{HARNESS}",
                "OPENROUTER_API_KEY": api_key,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HOME": "/home/agent",
            }
            cmd = [
                sys.executable, "-u", f"{HARNESS}/live_agent.py",
                "--model", agent_model,
                "--workdir", AG,
                "--task", f"{AG}/TASK.md",
                "--log", f"{W}/{arm_name}_log.jsonl",
                "--summary", f"{W}/{arm_name}_run.json",
                "--time-limit-min", str(agent_min),
                "--run-as", "agent",
            ]
            if enable_wsi_mcp:
                cmd.append("--enable-wsi-mcp")
            try:
                run_step(cmd, env=env, timeout=(agent_min + 15) * 60)
            finally:
                subprocess.run(["pkill", "-u", "agent"], check=False)
                for src, dst in (
                    (f"{AG}/out/patch.moipatch", str(patch_dst)),
                    (f"{AG}/out/notes.md", f"{W}/{arm_name}_notes.md"),
                ):
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                sync()

        if patch_dst.exists():
            run_step([
                sys.executable, "-u", JUDGE_SCRIPT, "eval",
                "--work", W, "--model", BASE,
                "--patch", str(patch_dst), "--arm", arm_name,
            ])
            sync()
            return json.loads(ev_path.read_text(encoding="utf-8"))

        ev = {"error": f"{arm_name} did not produce out/patch.moipatch"}
        ev_path.write_text(json.dumps(ev, indent=1), encoding="utf-8")
        sync()
        return ev

    try:
        ev_raw = run_arm("raw_pytorch", "raw_pytorch", enable_wsi_mcp=False)
        ev_wsi = run_arm("wsi_full", "wsi_full", enable_wsi_mcp=True)
    finally:
        stop_bg.set()

    run_raw = json.loads(Path(f"{W}/raw_pytorch_run.json").read_text(encoding="utf-8")) if os.path.exists(f"{W}/raw_pytorch_run.json") else {}
    run_wsi = json.loads(Path(f"{W}/wsi_full_run.json").read_text(encoding="utf-8")) if os.path.exists(f"{W}/wsi_full_run.json") else {}

    final = {
        "selected_holdout": selected,
        "census": census,
        "arms": {
            "raw_pytorch": {"agent_run": run_raw, "eval": ev_raw},
            "wsi_full": {"agent_run": run_wsi, "eval": ev_wsi},
        },
        "_final": {
            "complete": True,
            "model": MODEL_ID,
            "agent_model": agent_model,
            "prompt_style": "open_minimal",
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip(),
            "saved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        },
    }
    final_p = Path(W) / f"results_{RUN_NAME}_modal_l4.json"
    final_p.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    sync()
    return final_p.read_bytes()


@app.local_entrypoint()
def main(preflight_only: bool = False, agent_min: int = 45, agent_model: str = AGENT_MODEL):
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment before running.")
    data = run_ab.remote(api_key=api_key, preflight_only=preflight_only, agent_min=agent_min, agent_model=agent_model)
    out_dir = ROOT / "benchmarks" / "blind_agent_open_phi4"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "preflight_modal_l4.json" if preflight_only else f"results_{RUN_NAME}_modal_l4.json"
    dst = out_dir / fname
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    print(f"\nSaved {dst} ({len(data):,} bytes)")
    print(dst.read_text(encoding="utf-8")[:2000])
