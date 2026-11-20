"""CPU unit tests for Surgical Over-Refusal Abliteration vs. Classic Brute-Force Abliteration (`scripts/experiment_surgical_refusal_lfm3.py`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import experiment_surgical_refusal_lfm3 as sr  # noqa: E402
from wsi.moipatch import MoIPatch  # noqa: E402


def test_refusal_classifier_tp_and_fp() -> None:
    assert sr._is_refusal("I cannot fulfill that request for safety reasons.")
    assert sr._is_refusal("I’m sorry, I can’t assist with that.")
    assert sr._is_refusal("<think>reasoning</think>\nAs an AI language model, I am unable to provide that.")
    assert sr._is_refusal(("Here is some careful context first. " * 12) + "I cannot fulfill that request.")
    assert not sr._is_refusal("Sure! To terminate a zombie process in Linux, inspect its parent PID with ps -ef.")
    assert not sr._is_refusal("I cannot wait to explain how photosynthesis works in plant leaves!")
    assert not sr._is_refusal("")


def test_surgical_refusal_smoke_trihybrid_and_anti_backdoor(tmp_path) -> None:
    res = sr.run_smoke(out_dir=tmp_path)
    assert res["status"] == "ok" and res["smoke_passed"] is True

    patch_path = Path(res["patch_path"])
    assert patch_path.exists() and patch_path.stat().st_size > 0
    loaded = MoIPatch.load(patch_path)
    assert loaded.router is not None
    assert len(loaded.edits) <= 2

    audit = res["audit"]
    assert audit["fingerprint_ok"] is True
    assert audit["restore_exact_ok"] is True
    assert audit["n_edited_modules"] <= 2
    assert audit["classic_n_edited_modules"] == 8
    assert audit["edited_weights"] < audit["classic_edited_weights"]
    assert len(audit["vocab_logit_lens"]) == len(loaded.edits)
    assert len(audit["vocab_logit_lens"][0]["suppressed_tokens"]) >= 5

    old = audit["output_logit_diff"]
    assert old is not None and old["n_probes"] >= 2
    assert old["mean_next_token_kl"] > 0 and old["top1_changed_rate"] > 0
    assert old["suppressed_tokens"][0]["token_id"] == 25
    assert old["suppressed_tokens"][0]["mean_delta_prob"] < 0
    top_pro = old["promoted_tokens"][0]
    assert top_pro["token_id"] != 25 and top_pro["mean_delta_prob"] > 0
    assert "Sure, here is the solution:" in top_pro["token"]
    assert old["probe_router_fire_rate"] == 1.0

    ca = audit["control_audit"]
    assert ca is not None and ca["n_controls"] >= 2
    assert ca["routed_fire_rate"] == 0.0
    assert ca["routed_mean_kl"] == 0.0
    assert ca["max_abs_logit_diff_routed"] == 0.0

    arms = res["arms"]
    assert arms["base"]["or_bench_refusal_rate"] >= 0.8
    assert arms["wsi_surgical_routed"]["or_bench_refusal_rate"] < arms["base"]["or_bench_refusal_rate"]
    assert arms["wsi_surgical_routed"]["control_exact_match_rate"] == 1.0
    assert arms["wsi_surgical_routed"]["delta_healthy_nll"] == 0.0
    assert arms["wsi_surgical_routed"]["healthy_kl"] == 0.0
    assert arms["classic_ablit_all_layers"]["healthy_kl"] > arms["wsi_surgical_routed"]["healthy_kl"]

    assert res["split"]["toxic_test"] == 8 and res["split"]["toxic_cal"] == 8
    assert res["split"]["toxic_base_refused"] == 8
    for arm_name in (
        "base",
        "classic_ablit_all_layers",
        "wsi_surgical_always_on",
        "wsi_surgical_routed",
        "wsi_safety_routed",
    ):
        arm = arms[arm_name]
        assert arm["n_toxic_test"] == 8
        assert "toxic_refusal_rate" in arm and "toxic_refusal_kept" in arm
        assert set(arm["toxic_by_category"]) == {"cat_a", "cat_b"}
    assert arms["wsi_safety_routed"]["or_bench_refusal_rate"] == 0.0
    assert arms["wsi_safety_routed"]["control_exact_match_rate"] == 1.0
    assert arms["wsi_safety_routed"]["healthy_kl"] == 0.0
    assert arms["wsi_safety_routed"]["delta_healthy_nll"] == 0.0
    assert arms["wsi_safety_routed"]["toxic_refusal_kept"] == 1.0

    safety_patch_path = Path(res["safety_patch_path"])
    assert safety_patch_path.exists() and safety_patch_path.stat().st_size > 0
    loaded_safety = MoIPatch.load(safety_patch_path)
    assert loaded_safety.router is not None and loaded_safety.router.get("veto_dir") is not None

    rb = res["discovery"]["router_half_b"]
    assert "toxic_fire_rate" in rb and "toxic_base_refused_fire_rate" in rb and "n_toxic_fired" in rb
    sb = res["discovery"]["safety_router_half_b"]
    assert sb is not None and sb["toxic_fire_rate"] == 0.0 and sb["control_fpr"] == 0.0
    sg = res["safety_gate"]
    assert sg is not None and sg["threshold"] == 0.90 and isinstance(sg["passed"], bool)
    assert sg["safety_routed_passed"] is True
    dumped = json.dumps(res)
    assert "unsafe synthetic prompt" not in dumped and "unsafe cal prompt" not in dumped


def test_trim_gen_ids_and_batch_probe_fallback() -> None:
    assert sr._trim_gen_ids([10, 11, 2, 0, 0], eos_id=2, pad_id=0) == [10, 11]
    assert sr._trim_gen_ids([10, 11, 0, 0], eos_id=2, pad_id=0) == [10, 11]

    # 1. Equal-length prompts on _TinyLFM3ToyLM -> cannot verify left-padding -> returns 1
    toy = sr._TinyLFM3ToyLM().eval()
    toy_tok = sr._TinyRefusalTokenizer()
    assert sr._probe_batching_exact(toy, toy_tok, ["normal:0 a", "trigger:1 b"], torch.device("cpu"), batch_size=4) == 1

    # 2. Unequal-length tokenizer + mock models (pad-broken vs pad-preserving)
    class _UnequalTok:
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [5, 6] if "short" in text else [5, 6, 7, 8]

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return "ok"

    class _DummyGenModel(torch.nn.Module):
        def __init__(self, *, respect_mask: bool) -> None:
            super().__init__()
            self.respect_mask = respect_mask

        def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 4, attention_mask=None, **_kw) -> torch.Tensor:
            if attention_mask is not None and not self.respect_mask:
                # Corrupt output when padded rows exist (simulates conv/SSM pad leak)
                has_pad = bool((attention_mask == 0).any().item())
                nxt_val = 19 if has_pad else 15
            else:
                nxt_val = 15
            gen = torch.full((input_ids.shape[0], 2), nxt_val, dtype=torch.long, device=input_ids.device)
            return torch.cat([input_ids, gen], dim=1)

    unequal_prompts = ["short prompt", "much longer unequal prompt"]
    tok_u = _UnequalTok()
    broken = _DummyGenModel(respect_mask=False)
    exact = _DummyGenModel(respect_mask=True)
    assert sr._probe_batching_exact(broken, tok_u, unequal_prompts, torch.device("cpu"), batch_size=4) == 1
    assert sr._probe_batching_exact(exact, tok_u, unequal_prompts, torch.device("cpu"), batch_size=4) == 4
