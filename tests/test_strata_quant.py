"""CPU unit tests for `wsi/strata_quant.py` and X-Ray `quant_sensitivity_map` integration."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from experiment_reasoning_distill_pilot import _PlantedReasoningLM  # noqa: E402
from wsi.strata_quant import (  # noqa: E402
    SUPERBLOCK_BYTES,
    apply_h64_cols,
    build_quant_sensitivity_map,
    capture_teacher_topk_targets,
    eval_topk_kl_and_top1,
    get_h64,
    gptq_flip_quantize_sk_torch,
    matrix_wire_bytes,
    rtn_quantize_sk_torch,
    run_norm_scale_qat,
    solve_knapsack_bit_allocation,
)


def test_h64_orthogonality_and_sk_sqnr_order() -> None:
    h64 = get_h64("cpu")
    eye = torch.eye(64)
    assert torch.allclose(h64 @ h64.T, eye, atol=1e-5)

    g = torch.Generator().manual_seed(42)
    w = torch.randn((32, 256), generator=g) * 0.05
    # Inject a realistic outlier column that Hadamard spreads out cleanly
    w[:, 17] *= 15.0

    x = torch.randn((256, 256), generator=g)
    h = (x.T @ x) / 256.0 + 0.01 * torch.eye(256)

    errs = {}
    for code in ("S2_K", "S3_K", "S4_K", "S6_K", "S8_K"):
        wq, wb, eh = gptq_flip_quantize_sk_torch(w, h, code, use_h64=True, max_flips=64)
        assert wq.shape == w.shape
        assert wb == matrix_wire_bytes(32, 256, code)
        errs[code] = eh

    assert errs["S8_K"] < errs["S6_K"] < errs["S4_K"] < errs["S3_K"] < errs["S2_K"], errs

    # Naive RTN at 2-bit must have substantially worse Hessian output error than H64+GPTQ+64-flip
    _, wb_rtn, eh_rtn = rtn_quantize_sk_torch(w, "S2_K", use_clipping=False, use_h64=False, H=h)
    assert wb_rtn == matrix_wire_bytes(32, 256, "S2_K")
    assert errs["S2_K"] < eh_rtn * 0.65, (errs["S2_K"], eh_rtn)


def test_non256_padding_and_gptq_flip() -> None:
    g = torch.Generator().manual_seed(7)
    w = torch.randn((16, 192), generator=g) * 0.1
    x = torch.randn((128, 192), generator=g)
    h = (x.T @ x) / 128.0 + 0.05 * torch.eye(192)
    _, wb_rtn, eh_rtn = rtn_quantize_sk_torch(w, "S3_K", use_clipping=False, use_h64=False, H=h)
    wq, wb_gptq, eh_gptq = gptq_flip_quantize_sk_torch(w, h, "S3_K", use_h64=True, max_flips=64)
    assert wq.shape == w.shape
    assert wb_rtn == wb_gptq == 16 * 1 * SUPERBLOCK_BYTES["S3_K"]
    assert eh_gptq < eh_rtn, (eh_gptq, eh_rtn)


def test_knapsack_allocator_beats_uniform_under_budget() -> None:
    # 10 items: 3 sensitive items gain hugely from S3/S4; 7 cold items hardly lose anything at S2
    items = []
    for i in range(10):
        mult = 50.0 if i < 3 else 1.0
        items.append(
            {
                "id": f"item_{i}",
                "options": [
                    ("S2_K", 700, 10.0 * mult),
                    ("S3_K", 1020, 2.0 * mult),
                    ("S4_K", 1340, 0.4 * mult),
                ],
            }
        )
    uniform_budget = 10 * 1020
    uniform_err = sum(2.0 * (50.0 if i < 3 else 1.0) for i in range(10))
    sel, used_b, tot_e = solve_knapsack_bit_allocation(items, uniform_budget)
    assert used_b <= uniform_budget
    assert tot_e < uniform_err
    assert all(sel[f"item_{i}"] == "S4_K" for i in range(3))


def test_quant_sensitivity_map_on_atlas_xrays() -> None:
    atlas_dir = ROOT / "benchmarks" / "xray_atlas" / "tome1_xray"
    qwen_card = json.loads((atlas_dir / "qwen__qwen2.5-3b-instruct.xray.json").read_text(encoding="utf-8"))
    olmoe_card = json.loads((atlas_dir / "allenai__olmoe-1b-7b-0125-instruct.xray.json").read_text(encoding="utf-8"))
    inkling_card = json.loads((atlas_dir / "thinkingmachines__inkling-2-mini.xray.json").read_text(encoding="utf-8"))

    qm_qwen = build_quant_sensitivity_map(qwen_card)
    assert {0, 2, 31, 35}.issubset(set(qm_qwen["protected_layers"]))
    assert qm_qwen["recommended_policy"]["asymmetric_moe_pair"] is None

    qm_olmoe = build_quant_sensitivity_map(olmoe_card)
    assert {0, 3, 15}.issubset(set(qm_olmoe["protected_layers"]))
    assert qm_olmoe["recommended_policy"]["asymmetric_moe_pair"] == "S3_K+S2_K"

    qm_ink = build_quant_sensitivity_map(inkling_card)
    assert {0, 11, 17, 23}.issubset(set(qm_ink["protected_layers"]))
    # Looped core + tail must outrank prologue layers 1..5
    sc = qm_ink["per_layer_scores"]
    pro_mid = sum(sc[1:6]) / 5.0
    loop_mid = sum(sc[6:18]) / 12.0
    assert loop_mid > pro_mid, (pro_mid, loop_mid)


def test_norm_scale_qat_improves_kl_and_folds_cleanly() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    teacher = LlamaForCausalLM(cfg).eval()
    student = LlamaForCausalLM(cfg).eval()
    student.load_state_dict(teacher.state_dict())
    # Perturb norm scales as a quantization distortion proxy
    with torch.no_grad():
        for nm, mod in student.named_modules():
            if "norm" in nm.lower() and hasattr(mod, "weight"):
                mod.weight.data.mul_(0.7)

    seqs = [[1, 2, 4, 10, 20, 20, 20, 20], [2, 1, 4, 11, 20, 20, 20, 20], [3, 2, 1, 12, 20, 20, 20, 20]]
    tr_recs = capture_teacher_topk_targets(teacher, seqs[:2], "cpu", topk=16)
    dv_recs = capture_teacher_topk_targets(teacher, seqs[2:], "cpu", topk=16)

    pre = eval_topk_kl_and_top1(student, dv_recs, "cpu")["kl_mean"]
    res = run_norm_scale_qat(student, tr_recs, dv_recs, device="cpu", epochs=6, lr=5e-2, grad_accum=1, anchor_lambda=1e-4)
    post = eval_topk_kl_and_top1(student, dv_recs, "cpu")["kl_mean"]
    assert res["n_tuned_norms"] > 0
    assert post < pre, (pre, post, res)
    # Verify folded weights match best dev checkpoint KL cleanly
    assert abs(post - res["dev_kl_best"]) < 1e-4
