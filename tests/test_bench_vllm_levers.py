"""CPU tests for scripts/bench_vllm_levers.py: log parsing, cache env and the report's verdict (no vLLM needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_vllm_levers as bl  # noqa: E402

LOG = """INFO 11-20 [default_loader.py:291] Loading weights took 1.84 seconds
INFO 11-20 [backends.py:548] Using cache directory: /tmp/c/vllm/torch_compile_cache/abc/rank_0_0/backbone for vLLM's torch.compile
INFO 11-20 [monitor.py:34] torch.compile takes 41.27 s in total
INFO 11-20 [kv_cache_utils.py:1087] GPU KV cache size: 1,234,567 tokens
INFO 11-20 [kv_cache_utils.py:1091] Maximum concurrency for 4,322 tokens per request: 285.64x
INFO 11-20 [gpu_model_runner.py:3480] Graph capturing finished in 12 secs, took 0.45 GiB
INFO 11-20 [core.py:210] init engine (profile, create kv cache, warmup model) took 63.02 seconds
[gen-fast] vLLM options: fp8_kv=True chunked=True async=True max_seqs=150 early_loop_stop=True spec=None
"""


def test_parse_log():
    r = bl.parse_log(LOG)
    assert r["load_weights_s"] == 1.84 and r["torch_compile_s"] == 41.27 and r["graph_capture_s"] == 12.0
    assert r["kv_cache_tokens"] == 1234567 and r["max_concurrency"] == 285.64 and r["init_engine_s"] == 63.02
    assert r["fp8_kv_used"] is True and r["compile_cache_hit"] is False  # "Using cache directory" is printed cold too
    hit = LOG + "INFO [backends.py:150] Directly load the compiled graph(s) for dynamic shape from the cache, took 3.1 s"
    assert bl.parse_log(hit)["compile_cache_hit"] is True
    assert bl.parse_log("nothing")["compile_cache_hit"] is False


def test_cache_env_points_every_cache_and_overrides_vllm_root():
    e = bl.cache_env("/x")
    assert set(e) == set(bl.CACHE_VARS) and all(v.startswith("/x/") for v in e.values())
    e2 = bl.cache_env("/x", vllm_root="/drive/v")
    assert e2["VLLM_CACHE_ROOT"] == "/drive/v" and e2["TRITON_CACHE_DIR"] == "/x/triton"


def _write_state(tmp: Path, auto: list[float], fp8: list[float]) -> None:
    runs = {}
    for i, v in enumerate(auto):
        runs[f"kv_auto_{i}"] = {"tok_s": v, "accuracy": 0.1, "loop_killed": 0.8, "kv_cache_tokens": 100,
                                "max_concurrency": 200.0}
    for i, v in enumerate(fp8):
        runs[f"kv_fp8_{i}"] = {"tok_s": v, "accuracy": 0.1, "loop_killed": 0.8, "kv_cache_tokens": 200,
                               "max_concurrency": 400.0, "fp8_kv_used": True}
    runs["drive_now"] = {"init_s": 120.0, "process_wall_s": 150.0}
    runs["drive_full"] = {"init_s": 60.0, "process_wall_s": 80.0}
    (tmp / "plan_state.json").write_text(json.dumps({"runs": runs, "tar_restore_s": 5.0}))
    (tmp / "config.json").write_text(json.dumps({"base_dir": "b", "n_test": 150, "max_new": 4096}))


def test_report_verdicts(tmp_path):
    _write_state(tmp_path, [1000, 1010, 990, 1005], [800, 790, 810])
    bl.main(["report", "--work", str(tmp_path)])
    r = json.loads((tmp_path / "results_bench_vllm_levers.json").read_text())
    assert r["kv"]["fp8_beats_both_neighbours"] == [False, False, False]
    assert "OFF" in r["kv"]["verdict"] and r["startup"]["saving_per_start_s"] == 70.0
    _write_state(tmp_path, [1000, 1010, 990, 1005], [1100, 1120, 1090])
    bl.main(["report", "--work", str(tmp_path)])
    assert "keep fp8" in json.loads((tmp_path / "results_bench_vllm_levers.json").read_text())["kv"]["verdict"]
    _write_state(tmp_path, [1000, 1010, 990, 1005], [1005, 995, 1020])
    bl.main(["report", "--work", str(tmp_path)])
    assert "no clear" in json.loads((tmp_path / "results_bench_vllm_levers.json").read_text())["kv"]["verdict"]
