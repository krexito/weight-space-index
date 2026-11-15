"""Tests and verification for Weight-Space Index (WSI) integrated with `wsr`."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "wsr") not in sys.path:
    sys.path.insert(0, str(ROOT / "wsr"))

from wsi import WeightSpaceIndex, build_index
from wsr.anatomy import Anatomy
from wsr.checkpoint import Checkpoint
from wsr.ops import apply_ops


def run_all_tests():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Create a synthetic Llama checkpoint with a vocabulary file
        model_dir = tmp_path / "llama_tiny"
        torch.manual_seed(42)
        cfg = AutoConfig.for_model(
            "llama",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        )
        model = AutoModelForCausalLM.from_config(cfg)
        # Inject a multi-layer causal circuit:
        # - Layer 1 Neuron 3 writes along "Eiffel" (embed token 12)
        # - Layer 2 Neuron 7 reads "Eiffel" (via gate_proj row 7) AND writes "Paris" (via down_proj col 7)
        # - Layer 3 Neuron 5 reads "Paris" (via gate_proj row 5)
        with torch.no_grad():
            eiffel_in = model.model.embed_tokens.weight[12].clone().float()
            eiffel_in = eiffel_in / eiffel_in.norm()
            paris_out = model.lm_head.weight[10].clone().float()
            paris_out = paris_out / paris_out.norm()

            # Layer 1 Neuron 3 writes "Eiffel" into residual stream
            model.model.layers[1].mlp.down_proj.weight[:, 3] = (eiffel_in * 4.0).to(
                model.model.layers[1].mlp.down_proj.weight.dtype
            )
            # Layer 2 Neuron 7 is triggered by "Eiffel" (gate_proj row 7) and writes "Paris" (down_proj col 7)
            model.model.layers[2].mlp.gate_proj.weight[7, :] = (eiffel_in * 5.0).to(
                model.model.layers[2].mlp.gate_proj.weight.dtype
            )
            model.model.layers[2].mlp.down_proj.weight[:, 7] = (paris_out * 5.0).to(
                model.model.layers[2].mlp.down_proj.weight.dtype
            )
            # Layer 3 Neuron 5 is triggered by "Paris" (gate_proj row 5)
            model.model.layers[3].mlp.gate_proj.weight[5, :] = (paris_out * 4.5).to(
                model.model.layers[3].mlp.gate_proj.weight.dtype
            )
        model.save_pretrained(model_dir, safe_serialization=True)

        # Write a synthetic vocab.json with real semantic concepts
        vocab = {f"<tok_{i}>": i for i in range(128)}
        vocab["Paris"] = 10
        vocab["Rome"] = 11
        vocab["Eiffel"] = 12
        vocab["</think>"] = 15
        for idx, ch in enumerate("ABCDEFGHIJ"):
            vocab[ch] = 20 + idx
        for idx, ch in enumerate("0123456789"):
            vocab[ch] = 40 + idx
        (model_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

        from safetensors.torch import save_file

        # 2. Build the Weight-Space Index
        idx_dir = tmp_path / "wsi_index"
        idx_data = build_index(
            model_dir,
            idx_dir,
            top_neurons_per_layer=16,
            top_tokens_per_unit=6,
            svd_rank=6,
        )
        assert (idx_dir / "index.json").exists()
        assert (idx_dir / "subspaces.safetensors").exists()
        assert idx_data["anatomy"]["layers"] == 4

        # 3. Load and query the index
        wsi = WeightSpaceIndex.from_model(model_dir, index_dir=idx_dir)
        ov = wsi.overview()
        assert len(ov["layers"]) == 4

        # 4. Search for "Paris" and verify Layer 2 Neuron 7 is the #1 promoting neuron!
        paris_res = wsi.search_token("Paris")
        assert paris_res["selected_token"]["token_id"] == 10
        top_promoter = paris_res["top_promoting_neurons"][0]
        assert top_promoter["layer"] == 2 and top_promoter["unit"] == 7, f"Unexpected top promoter: {top_promoter}"

        # 5. Explain Layer 2 Neuron 7 directly and confirm:
        #    - Input trigger #1 is "Eiffel"
        #    - Output promoted #1 is "Paris"
        #    - Transduction rule is "[Eiffel, ...] -> [Paris, ...]"
        n_exp = wsi.explain_neuron(layer=2, unit=7, module="mlp_out")
        assert n_exp["promoted_tokens"][0]["token"] == "Paris"
        assert n_exp["input_side"]["triggers"][0]["token"] == "Eiffel"
        assert "Eiffel" in n_exp["transduction_rule"] and "Paris" in n_exp["transduction_rule"]

        # 5b. Test `search_transduction("Eiffel", "Paris")` -> must find Layer 2 Neuron 7 at #1!
        trans_res = wsi.search_transduction("Eiffel", "Paris")
        top_trans = trans_res["top_transducing_neurons"][0]
        assert top_trans["layer"] == 2 and top_trans["unit"] == 7, f"Unexpected transduction top hit: {top_trans}"

        # 5c. Test `trace_circuit(layer=2, unit=7)` -> must find:
        #     - Upstream writer: Layer 1 Neuron 3
        #     - Downstream reader: Layer 3 Neuron 5
        circuit_res = wsi.trace_circuit(layer=2, unit=7)
        assert circuit_res["upstream_writers"][0]["src_layer"] == 1
        assert circuit_res["upstream_writers"][0]["src_unit"] == 3
        assert circuit_res["downstream_readers"][0]["dst_layer"] == 3
        assert circuit_res["downstream_readers"][0]["dst_unit"] == 5

        # 6. Synthesize a surgical edit with `suggest_edit` and diagnose it with `explain_edit`
        suggested = wsi.suggest_edit(suppress_tokens=["Paris"], strategy="orth", layers=[2], alpha=1.0)
        ck_reloaded = Checkpoint(model_dir)
        anat_reloaded = Anatomy(ck_reloaded)
        full_edit, compact_edit, desc = apply_ops(ck_reloaded, suggested, anat_reloaded)
        edit_file = tmp_path / "edit_orth_paris.safetensors"
        save_file(compact_edit, str(edit_file))

        diag = wsi.explain_edit(edit_file)
        assert diag["validation"]["ok"]
        t_diag = diag["tensors"][0]
        assert t_diag["tensor"] == "model.layers.2.mlp.down_proj.weight"
        # Because `orth` removes the "Paris" component (which was positive in neuron 7),
        # the rank-1 delta has 100% energy fraction (rank-1 update) and aligns with "Paris"!
        assert t_diag["delta_rank1_energy_fraction"] > 0.999
        assert t_diag["subspace_action"] == "projected_out_or_suppressed"
        assert t_diag["old_energy_along_principal_dir"] > 4.9
        assert t_diag["new_energy_along_principal_dir"] < 1e-4
        pole_tokens = (
            [t["token"] for t in t_diag["principal_delta_direction"]["promoted_tokens"][:2]]
            + [t["token"] for t in t_diag["principal_delta_direction"]["suppressed_tokens"][:2]]
        )
        assert "Paris" in pole_tokens

        # 7. Verify sandbox integration (`wsr.sandbox.install_helper` + `wsr.sandbox_lib.I`)
        import os
        from wsr.sandbox import install_helper
        sb_pkg = install_helper(tmp_path / "_sandbox_pkg")
        assert (sb_pkg / "wsi" / "query.py").exists()
        os.environ["WSR_TARGET_DIR"] = str(model_dir)
        os.environ["WSR_INDEX_DIR"] = str(idx_dir)
        from wsr.sandbox_lib import _LazyIndex
        lazy_i = _LazyIndex(str(model_dir), str(idx_dir))
        assert lazy_i.search_token("Paris")["top_promoting_neurons"][0]["unit"] == 7

        # 8. Verify instant on-demand `lazy=True` mode (with NO prebuilt index_dir!)
        wsi_lazy = WeightSpaceIndex.from_model(model_dir)  # defaults to lazy=True when index_dir=None
        assert wsi_lazy.lazy is True
        assert wsi_lazy.overview()["n_indexed_layers"] == 0
        assert wsi_lazy.search_token("Paris")["top_promoting_neurons"][0]["unit"] == 7
        assert wsi_lazy.search_transduction("Eiffel", "Paris")["top_transducing_neurons"][0]["unit"] == 7
        assert wsi_lazy.trace_circuit(2, 7)["upstream_writers"][0]["src_unit"] == 3

        # 9. Test Same-Architecture Diff (`diff_model`): Base Llama vs Fine-Tuned Llama (where Eiffel -> Rome)
        tuned_dir = tmp_path / "llama_tuned"
        rome_out = model.lm_head.weight[11].clone().float()
        rome_out = rome_out / rome_out.norm()
        with torch.no_grad():
            model.model.layers[2].mlp.down_proj.weight[:, 7] = (rome_out * 5.0).to(
                model.model.layers[2].mlp.down_proj.weight.dtype
            )
        model.save_pretrained(tuned_dir, safe_serialization=True)
        (tuned_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

        diff_res = wsi_lazy.diff_model(tuned_dir)
        assert diff_res["most_changed_layers"][0]["layer"] == 2
        top_changed_neuron = diff_res["most_changed_neurons"][0]
        assert top_changed_neuron["layer"] == 2 and top_changed_neuron["unit"] == 7
        assert "Paris" in top_changed_neuron["before_transduction"]
        assert "Rome" in top_changed_neuron["after_transduction"]

        # 10. Test Cross-Architecture Semantic Circuit Alignment (`align_circuits`):
        #     Compare 4-layer Llama (`hidden=32`, circuit at Layer 2 Unit 7)
        #     against a 6-layer Qwen2 (`hidden=48`, circuit at Layer 4 Unit 19)!
        qwen_dir = tmp_path / "qwen2_diff_arch"
        torch.manual_seed(99)
        qwen_cfg = AutoConfig.for_model(
            "qwen2",
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        )
        qwen_model = AutoModelForCausalLM.from_config(qwen_cfg)
        with torch.no_grad():
            q_eiffel_in = qwen_model.model.embed_tokens.weight[12].clone().float()
            q_eiffel_in = q_eiffel_in / q_eiffel_in.norm()
            q_paris_out = qwen_model.lm_head.weight[10].clone().float()
            q_paris_out = q_paris_out / q_paris_out.norm()
            qwen_model.model.layers[4].mlp.gate_proj.weight[19, :] = (q_eiffel_in * 6.0).to(
                qwen_model.model.layers[4].mlp.gate_proj.weight.dtype
            )
            qwen_model.model.layers[4].mlp.down_proj.weight[:, 19] = (q_paris_out * 6.0).to(
                qwen_model.model.layers[4].mlp.down_proj.weight.dtype
            )
        qwen_model.save_pretrained(qwen_dir, safe_serialization=True)
        (qwen_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

        align_res = wsi_lazy.align_circuits(
            qwen_dir,
            transduction_pairs=[("Eiffel", "Paris")],
            concepts=["Paris"],
        )
        matched = align_res["matched_transduction_circuits"][0]
        assert matched["model_a_coord"]["layer"] == 2 and matched["model_a_coord"]["unit"] == 7
        assert matched["model_b_coord"]["layer"] == 4 and matched["model_b_coord"]["unit"] == 19

        print("\n--- Example `diff_model` (Base vs Tuned Neuron Shift) ---")
        print(json.dumps(top_changed_neuron, indent=2))
        print("\n--- Example `align_circuits` (4-layer Llama vs 6-layer Qwen2) ---")
        print(json.dumps(matched, indent=2))

        # 11. Test Automatic Cross-Model Edit Transfer (`transfer_edit_to`):
        #     Given ONLY a binary `.safetensors` edit file made for Model A (4-layer Llama, hidden=32),
        #     automatically decompile its semantic intent and transfer the edit to Model B (6-layer Qwen2, hidden=48)!
        llama_rewire_file = tmp_path / "llama_eiffel_to_rome.safetensors"
        wsi_lazy.rewire_transduction(
            "Eiffel",
            "Paris",
            "Rome",
            top_k_neurons=1,
            strength=5.0,
            save_path=llama_rewire_file,
        )

        qwen_transferred_edit_file = tmp_path / "qwen_transferred_edit.safetensors"
        transfer_res = wsi_lazy.transfer_edit_to(
            qwen_dir,
            llama_rewire_file,
            top_k_neurons=1,
            strength=5.0,
            save_path=qwen_transferred_edit_file,
        )
        assert transfer_res["decompiled_from_model_a"]["trigger_token"]["token"] == "Eiffel"
        assert transfer_res["decompiled_from_model_a"]["old_output_token"]["token"] == "Paris"
        assert transfer_res["decompiled_from_model_a"]["new_output_token"]["token"] == "Rome"
        assert transfer_res["synthesized_edit_on_model_b"]["neurons_rewired"][0]["layer"] == 4
        assert transfer_res["synthesized_edit_on_model_b"]["neurons_rewired"][0]["unit"] == 19
        assert qwen_transferred_edit_file.exists()

        # Verify that applying the transferred edit to Qwen2 actually changes Qwen2's Layer 4 Unit 19 to output "Rome"!
        wsi_qwen = WeightSpaceIndex.from_model(qwen_dir)
        qwen_diag = wsi_qwen.explain_edit(qwen_transferred_edit_file)
        assert qwen_diag["validation"]["ok"]
        assert qwen_diag["tensors"][0]["principal_delta_direction"]["promoted_tokens"][0]["token"] == "Rome"

        # Verify zero-pass full key+value neuron transplant (Qwen2 hidden=48 -> Llama hidden=32)
        full_tx = wsi_lazy.transplant_full_neuron_from(
            wsi_qwen,
            teacher_layer=4,
            teacher_unit=19,
            top_m_triggers=8,
            top_m_chord=8,
        )
        assert full_tx["student_graft"]["layer"] == 2
        assert "model.layers.2.mlp.gate_proj.weight" in full_tx["_tensors"]
        assert "model.layers.2.mlp.down_proj.weight" in full_tx["_tensors"]
        assert full_tx["rosetta_translation"]["top1_exact_match"]

        # Verify zero-pass multi-hop residual connection tracing & 2-layer chain transplant
        mh_trace = wsi_lazy.trace_residual_connections(layer=0, unit=0, top_k_connections=4, top_k_tokens=4)
        assert mh_trace["two_hop_downstream_vocab"] is not None
        assert len(mh_trace["downstream_mlp_layers"]) >= 1
        assert len(mh_trace["top_coupled_attention_heads"]) >= 1

        qwen_chains = wsi_qwen.discover_connected_chains(0, 4, top_k=3, alphabetic_only=False)
        assert len(qwen_chains) >= 1 and qwen_chains[0]["src_layer"] == 0 and qwen_chains[0]["dst_layer"] == 4
        ch0 = qwen_chains[0]
        assert len(ch0["dst_triggers"]) >= 5
        assert not (set(ch0["dst_trigger_shared_indices"]) & set(ch0["src_top50_shared_indices"]))
        chain_tx = wsi_lazy.transplant_connected_chain_from(
            wsi_qwen,
            teacher_src_layer=ch0["src_layer"],
            teacher_src_unit=ch0["src_unit"],
            teacher_dst_layer=ch0["dst_layer"],
            teacher_dst_unit=ch0["dst_unit"],
            student_src_layer=1,
            student_dst_layer=3,
            top_m_triggers=6,
            top_m_chord=8,
            alphabetic_only=False,
        )
        assert chain_tx["student_chain"]["src_layer"] == 1 and chain_tx["student_chain"]["dst_layer"] == 3
        assert chain_tx["student_chain"]["wire_max_target_vocab_cos"] < 1e-4
        assert "model.layers.1.mlp.down_proj.weight" in chain_tx["_tensors"]
        assert "model.layers.3.mlp.gate_proj.weight" in chain_tx["_tensors"]
        assert "model.layers.3.mlp.down_proj.weight" in chain_tx["_tensors"]

        # Verify compositional 2-layer AND circuit transplant (A + B -> C) both with and without context vecs
        and_tx = wsi_lazy.transplant_compositional_and_chain_from(
            wsi_qwen,
            teacher_src_layer=ch0["src_layer"],
            teacher_src_unit=ch0["src_unit"],
            teacher_dst_layer=ch0["dst_layer"],
            teacher_dst_unit=ch0["dst_unit"],
            student_src_layer=1,
            student_dst_layer=3,
            top_m_triggers=5,
            top_m_chord=8,
            alphabetic_only=False,
            student_src_context_vecs={
                "pos_a": torch.randn(8, 32),
                "ctrl": torch.randn(12, 32),
                "n_w2": 2,
            },
        )
        assert and_tx["student_and_gate"]["src_layer"] == 1 and and_tx["student_and_gate"]["dst_layer"] == 3
        assert and_tx["student_and_gate"]["wire_max_target_vocab_cos"] < 1e-4
        assert not (
            set(and_tx["teacher_and_circuit"]["a_triggers"])
            & set(and_tx["teacher_and_circuit"]["b_triggers"])
        )
        assert "model.layers.1.mlp.gate_proj.weight" in and_tx["_tensors"]
        assert "model.layers.1.mlp.down_proj.weight" in and_tx["_tensors"]
        assert "model.layers.3.mlp.gate_proj.weight" in and_tx["_tensors"]
        assert "model.layers.3.mlp.down_proj.weight" in and_tx["_tensors"]
        assert "_slot_payload" in and_tx

        # Verify DynamicCircuitPager O(1) JIT slot paging & multi-circuit routing
        pager = wsi_lazy.create_dynamic_circuit_pager(
            src_layer=1,
            dst_layer=3,
            src_unit=and_tx["student_and_gate"]["src_unit"],
            dst_unit=and_tx["student_and_gate"]["dst_unit"],
        )
        reg0 = pager.register_circuit("chain_0", and_tx)
        assert len(reg0["a_token_ids"]) >= 1 and len(reg0["b_token_ids"]) >= 1
        used_tids = set(reg0["a_token_ids"]) | set(reg0["b_token_ids"])
        free_tids = [int(tid) for tid in sorted(wsi_lazy.vocab_map) if int(tid) not in used_tids]
        assert len(free_tids) >= 2
        c1_a, c1_b = free_tids[0], free_tids[1]
        pager.register_circuit("chain_1", and_tx, a_token_ids=[c1_a], b_token_ids=[c1_b])

        r0 = pager.route_token_ids([reg0["a_token_ids"][0], reg0["b_token_ids"][0]])
        assert r0["circuit_id"] == "chain_0" and r0["matched"] and r0["matched_mode"] == "both"
        r1 = pager.route_token_ids([c1_a, c1_b])
        assert r1["circuit_id"] == "chain_1" and r1["matched"] and r1["matched_mode"] == "both"
        r_cross = pager.route_token_ids([reg0["a_token_ids"][0], c1_b])
        assert not r_cross["matched"] and r_cross["matched_mode"] != "both"
        r_str = pager.route_tokens(
            [and_tx["teacher_and_circuit"]["a_triggers"][0], and_tx["teacher_and_circuit"]["b_triggers"][0]]
        )
        assert r_str["circuit_id"] == "chain_0" and r_str["matched"]

        paged_tensors = pager.synthesize_slot_tensors("chain_0")
        assert "model.layers.1.mlp.gate_proj.weight" in paged_tensors
        assert "model.layers.1.mlp.down_proj.weight" in paged_tensors
        assert "model.layers.3.mlp.gate_proj.weight" in paged_tensors
        assert "model.layers.3.mlp.down_proj.weight" in paged_tensors
        sup_payload = pager.build_static_superposition_payload()
        assert sup_payload["n_superposed"] == 2
        assert sup_payload["key_src"].shape == (wsi_lazy.anat.hidden,)
        cat_info = pager.catalog_summary()
        assert cat_info["n_circuits"] == 2
        assert cat_info["per_circuit_bytes"] == (4 * wsi_lazy.anat.hidden + 2) * 4
        assert cat_info["wire_transfer_ms"]["1Gbps"] > 0

        print("\n--- Example `transfer_edit_to` (Llama .safetensors edit -> Qwen2 .safetensors edit) ---")
        print(json.dumps({
            "decompiled_from_llama": transfer_res["decompiled_from_model_a"],
            "transferred_to_qwen": transfer_res["synthesized_edit_on_model_b"]["neurons_rewired"][0],
        }, indent=2))

        # 12. Verify MCP Server JSON-RPC 2.0 protocol (`wsi/mcp_server.py`)
        from wsi.mcp_server import handle_rpc_message
        init_resp = handle_rpc_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert init_resp["result"]["serverInfo"]["name"] == "weight-space-index"
        list_resp = handle_rpc_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tool_names = {t["name"] for t in list_resp["result"]["tools"]}
        assert "wsi_transfer_edit" in tool_names and "wsi_search_transduction" in tool_names
        assert "wsi_trace_internal_state" in tool_names
        istate_res = wsi_lazy.trace_internal_state(layer=2, neurons=[7], top_k_neurons=8)
        assert "layer_attention_head_rankings" in istate_res and "cross_layer_mlp_echoes" in istate_res
        call_resp = handle_rpc_message({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "wsi_search_transduction",
                "arguments": {"model_dir": str(model_dir), "input_query": "Eiffel", "output_query": "Paris"},
            },
        })
        assert not call_resp["result"]["isError"]
        mcp_payload = json.loads(call_resp["result"]["content"][0]["text"])
        assert mcp_payload["top_transducing_neurons"][0]["layer"] == 2
        assert mcp_payload["top_transducing_neurons"][0]["unit"] == 7

        print("ALL WSI TESTS PASSED SUCCESSFULLY!")
        print("\n--- Example `search_transduction('Eiffel', 'Paris')` top result ---")
        print(json.dumps(trans_res["top_transducing_neurons"][0], indent=2))
        print("\n--- Example `trace_circuit(layer=2, unit=7)` causal chain ---")
        print(json.dumps({
            "transduction_rule": circuit_res["transduction_rule"],
            "top_upstream_writer": circuit_res["upstream_writers"][0],
            "top_downstream_reader": circuit_res["downstream_readers"][0],
        }, indent=2))


if __name__ == "__main__":
    run_all_tests()
