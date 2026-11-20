"""CPU tests for the two gates in discover_math500_blind.py: stop after elicit when the base barely hard-loops, and
freeze "no_patch" when no screen finalist beats base on dev. Generation and grading are stubbed."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import discover_math500_blind as m  # noqa: E402


@pytest.fixture
def work(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"n_elicit": 10, "n_dev": 4, "n_test": 4, "seed": 42,
                                                      "max_tokens_discover": 64, "max_tokens_test": 64}))
    monkeypatch.setattr(m.dl, "cfg_load", lambda w: json.loads((Path(w) / "config.json").read_text()))
    monkeypatch.setattr(m, "_try_load_base_cache", lambda *a, **k: True)
    monkeypatch.setattr(m, "_save_base_cache", lambda *a, **k: None)
    monkeypatch.setattr(m, "_run_parallel_fast", lambda *a, **k: None)
    monkeypatch.setattr(m.fast, "sync", lambda w: None)
    monkeypatch.setattr(m.fast, "note_time", lambda *a, **k: None)
    return tmp_path


def _scores(monkeypatch, table):
    monkeypatch.setattr(m, "score_math", lambda w, req, out, tok=None: {"n": 10, "rows": [], **table[out]})


def test_elicit_gate_stops_when_base_barely_loops(work, monkeypatch):
    _scores(monkeypatch, {"outputs_elicit.jsonl": {"accuracy": 0.64, "loop_killed": 0.0, "truncated": 0.33}})
    a = SimpleNamespace(work=str(work), min_loop_killed=0.10, force=False)
    with pytest.raises(SystemExit, match="STOP"):
        m.cmd_elicit(a)
    g = json.loads((work / "gate_elicit.json").read_text())
    assert g["passed"] is False and g["loop_killed"] == 0.0


def test_elicit_gate_passes_or_is_forced(work, monkeypatch):
    _scores(monkeypatch, {"outputs_elicit.jsonl": {"accuracy": 0.3, "loop_killed": 0.39, "truncated": 0.1}})
    m.cmd_elicit(SimpleNamespace(work=str(work), min_loop_killed=0.10, force=False))
    assert json.loads((work / "gate_elicit.json").read_text())["passed"] is True
    _scores(monkeypatch, {"outputs_elicit.jsonl": {"accuracy": 0.6, "loop_killed": 0.01, "truncated": 0.1}})
    m.cmd_elicit(SimpleNamespace(work=str(work), min_loop_killed=0.10, force=True))  # no raise
    assert json.loads((work / "gate_elicit.json").read_text())["forced"] is True


def _finalists(work):
    fin = [{"name": "L14_n256_a12", "layers": [14], "neurons": 256, "alpha": 12.0, "k": 1},
           {"name": "L14_nall_a8", "layers": [14], "neurons": "all", "alpha": 8.0, "k": 1}]
    (work / "screen.json").write_text(json.dumps({"finalists": fin}))
    for c in fin:
        (work / f"outputs_dev_{c['name']}.jsonl").write_text("")
    return fin


@pytest.mark.parametrize("force_patch,expect", [(False, "no_patch"), (True, "L14_n256_a12")])
def test_select_freezes_no_patch_when_finalists_lose(work, monkeypatch, force_patch, expect):
    _finalists(work)
    _scores(monkeypatch, {"outputs_dev_base.jsonl": {"accuracy": 0.656, "loop_killed": 0.0, "truncated": 0.31},
                          "outputs_dev_L14_n256_a12.jsonl": {"accuracy": 0.5625, "loop_killed": 0.0, "truncated": 0.44},
                          "outputs_dev_L14_nall_a8.jsonl": {"accuracy": 0.53, "loop_killed": 0.0, "truncated": 0.47}})
    m.cmd_select(SimpleNamespace(work=str(work), force_patch=force_patch))
    st = json.loads((work / "search.json").read_text())
    assert st["choice"]["name"] == expect and st["best_finalist"]["name"] == "L14_n256_a12"


def test_select_keeps_a_winner_that_beats_base(work, monkeypatch):
    _finalists(work)
    _scores(monkeypatch, {"outputs_dev_base.jsonl": {"accuracy": 0.11, "loop_killed": 0.8, "truncated": 0.1},
                          "outputs_dev_L14_n256_a12.jsonl": {"accuracy": 0.40, "loop_killed": 0.3, "truncated": 0.1},
                          "outputs_dev_L14_nall_a8.jsonl": {"accuracy": 0.45, "loop_killed": 0.2, "truncated": 0.1}})
    m.cmd_select(SimpleNamespace(work=str(work), force_patch=False))
    assert json.loads((work / "search.json").read_text())["choice"]["name"] == "L14_nall_a8"


def test_test_step_runs_base_only_for_no_patch(work, monkeypatch):
    (work / "frozen.json").write_text(json.dumps({"choice": m.NO_PATCH}))
    called = []
    monkeypatch.setattr(m.blind, "build_checkpoint", lambda *a, **k: called.append("ck"))
    monkeypatch.setattr(m.dl, "cmd_plan_moi", lambda a: called.append("moi"))
    monkeypatch.setattr(m, "_run_fused_pair", lambda *a, **k: called.append("fused"))
    m.cmd_test(SimpleNamespace(work=str(work), ck_root=None))
    assert called == []


def _fake_vllm(monkeypatch, refuse=lambda kw: False):
    import types
    calls = []

    class LLM:
        def __init__(self, **kw):
            calls.append(kw)
            if refuse(kw):
                raise ValueError("refused")

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM))
    import loop_kill_v2
    monkeypatch.setattr(loop_kill_v2, "processor", lambda: "KILL")
    monkeypatch.setattr(m, "_fp8_kv_ok", lambda: False)
    return calls


def test_llm_fast_mtp_switch(monkeypatch):
    calls = _fake_vllm(monkeypatch)
    monkeypatch.delenv("WSI_SPEC", raising=False)
    m._llm_fast("x", {"base_dir": "x", "seed": 0}, 1024, "auto", 0.9)
    assert "speculative_config" not in calls[-1]
    monkeypatch.setenv("WSI_SPEC", "mtp")
    llm, kill = m._llm_fast("x", {"base_dir": "x", "seed": 0}, 1024, "auto", 0.9)
    assert kill and calls[-1]["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 2}
    assert calls[-1]["logits_processors"] == ["KILL"]


def test_llm_fast_mtp_never_drops_the_loop_kill(monkeypatch):
    # a vLLM that refuses MTP together with a logits processor: stop, do not silently run without early stop
    calls = _fake_vllm(monkeypatch, refuse=lambda kw: "logits_processors" in kw and "speculative_config" in kw)
    monkeypatch.setenv("WSI_SPEC", "mtp")
    with pytest.raises(SystemExit):
        m._llm_fast("x", {"base_dir": "x", "seed": 0}, 1024, "auto", 0.9)
    assert all("logits_processors" in kw for kw in calls)
    monkeypatch.delenv("WSI_SPEC")  # without MTP the old fallback still works
    calls.clear()
    _, kill = m._llm_fast("x", {"base_dir": "x", "seed": 0}, 1024, "auto", 0.9)
    assert kill


def test_spec_config_from_env(monkeypatch):
    import discover_fast as fast

    monkeypatch.delenv("WSI_SPEC", raising=False)
    assert fast.spec_config_from_env() is None

    monkeypatch.setenv("WSI_SPEC", "mtp")
    monkeypatch.delenv("WSI_SPEC_TOKENS", raising=False)
    assert fast.spec_config_from_env() == {"method": "mtp", "num_speculative_tokens": 2}

    monkeypatch.setenv("WSI_SPEC", "ngram")
    monkeypatch.setenv("WSI_SPEC_TOKENS", "6")
    monkeypatch.setenv("WSI_SPEC_NGRAM_MAX", "5")
    assert fast.spec_config_from_env() == {"method": "ngram", "num_speculative_tokens": 6, "prompt_lookup_max": 5}

    monkeypatch.setenv("WSI_SPEC", "suffix")
    monkeypatch.setenv("WSI_SPEC_TOKENS", "12")
    monkeypatch.setenv("WSI_SPEC_SUFFIX_CACHE", "500")
    assert fast.spec_config_from_env() == {
        "method": "suffix",
        "num_speculative_tokens": 12,
        "suffix_decoding_max_cached_requests": 500,
    }

    monkeypatch.setenv("WSI_SPEC", "bogus")
    with pytest.raises(SystemExit):
        fast.spec_config_from_env()


@pytest.mark.parametrize("method", ["ngram", "suffix"])
def test_llm_fast_ngram_and_suffix_keep_loop_kill(monkeypatch, method):
    calls = _fake_vllm(monkeypatch)
    monkeypatch.setenv("WSI_SPEC", method)
    _, kill = m._llm_fast("x", {"base_dir": "x", "seed": 0}, 1024, "auto", 0.9)
    assert kill and calls[-1]["speculative_config"]["method"] == method
    assert calls[-1]["logits_processors"] == ["KILL"]

    calls = _fake_vllm(monkeypatch, refuse=lambda kw: "logits_processors" in kw and "speculative_config" in kw)
    with pytest.raises(SystemExit):
        m._llm_fast("x", {"base_dir": "x", "seed": 0}, 1024, "auto", 0.9)
    assert all("logits_processors" in kw for kw in calls)
