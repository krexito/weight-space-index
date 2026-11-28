"""CPU unit and integration tests for `wsi.looped_reader`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wsi.looped_reader import (
    LoopedCircuitSpec,
    SparseMatrix,
    analyze_looped_circuit,
    classify_looped_circuit,
    load_looped_weights,
)

ROOT = Path(__file__).resolve().parent.parent
LOOPED_WSP_DIR = ROOT / "work" / "LoopedWSP" / "weights"


def _make_planted_circuit() -> LoopedCircuitSpec:
    """Build a synthetic d_model=6 looped circuit with planted gadgets & dataflow partition:
      ch 0: readonly_input_or_pos (written by emb, read by W1 and Wq/Wk, never written by block)
      ch 1: recurrent_scratchpad (read by W1 & Wv, written by W2 & Wo -> 2-node SCC with ch 2)
      ch 2: recurrent_scratchpad (read by W1 & Wv, written by W2 & Wo)
      ch 3: write_only_readout (written by W2, never read inside block, read by out.W)
      ch 4: readonly_input_or_pos (written by emb, read by out.W)
      ch 5: unused_dead (never touched)

    MLP (d_ff=6):
      neurons 0, 1: planted step/threshold pair tau(z) = ReLU(z - 1) - ReLU(z - 2) on ch 0 -> ch 3
      neurons 2, 3: planted cancellation pair (W1[:,2] = -W1[:,3], W2[2,:] = W2[3,:]) -> |x| on ch 1 -> ch 2
      neuron  4:    single unmatched active neuron ch 2 -> ch 1 (closes SCC 1 <-> 2)
      neuron  5:    dead neuron (all zeros)
    """
    d = 6
    d_ff = 6
    emb = np.zeros((4, d), dtype=np.float64)
    emb[:, 0] = 1.0
    emb[:, 4] = 2.0

    Wq = np.zeros((d, 2), dtype=np.float64)
    Wk = np.zeros((d, 2), dtype=np.float64)
    Wq[0, 0] = 50.0  # hard attention scale = 2500 / sqrt(2)
    Wk[0, 0] = 50.0

    Wv = np.zeros((d, 2), dtype=np.float64)
    Wo = np.zeros((2, d), dtype=np.float64)
    Wv[1, 0] = 1.0
    Wo[0, 2] = 1.0  # ch 1 -> ch 2 via attention

    W1 = np.zeros((d, d_ff), dtype=np.float64)
    b1 = np.zeros(d_ff, dtype=np.float64)
    W2 = np.zeros((d_ff, d), dtype=np.float64)
    b2 = np.zeros(d, dtype=np.float64)

    # Threshold pair (0, 1): parallel W1 columns on ch 0, distinct biases, anti-parallel W2 rows on ch 3
    W1[0, 0] = 1.0
    W1[0, 1] = 1.0
    b1[0] = -1.0
    b1[1] = -2.0
    W2[0, 3] = 1.0
    W2[1, 3] = -1.0

    # Cancellation pair (2, 3): anti-parallel W1 columns on ch 1, parallel W2 rows on ch 2
    W1[1, 2] = 2.0
    W1[1, 3] = -2.0
    W2[2, 2] = 0.5
    W2[3, 2] = 0.5

    # Feedback edge ch 2 -> ch 1 to form SCC {1, 2}
    W1[2, 4] = 1.0
    W2[4, 1] = 1.0

    W_out = np.zeros((d, 2), dtype=np.float64)
    W_out[3, 0] = 1.0
    W_out[4, 1] = 1.0
    b_out = np.zeros(2, dtype=np.float64)

    raw = {
        "emb.0": SparseMatrix.from_dense(emb),
        "L0.Wq": SparseMatrix.from_dense(Wq),
        "L0.Wk": SparseMatrix.from_dense(Wk),
        "L0.Wv": SparseMatrix.from_dense(Wv),
        "L0.Wo": SparseMatrix.from_dense(Wo),
        "L0.W1": SparseMatrix.from_dense(W1),
        "L0.b1": SparseMatrix.from_dense(b1),
        "L0.W2": SparseMatrix.from_dense(W2),
        "L0.b2": SparseMatrix.from_dense(b2),
        "out.W": SparseMatrix.from_dense(W_out),
        "out.b": SparseMatrix.from_dense(b_out),
    }
    roles = {
        "emb.0": "embed",
        "L0.Wq": "q",
        "L0.Wk": "k",
        "L0.Wv": "v",
        "L0.Wo": "o",
        "L0.W1": "w1",
        "L0.b1": "b1",
        "L0.W2": "w2",
        "L0.b2": "b2",
        "out.W": "unembed",
        "out.b": "other",
    }
    tot = sum(m.size for m in raw.values())
    nz = sum(m.nnz for m in raw.values())
    return LoopedCircuitSpec(
        matrices=raw,
        roles=roles,
        d_model=d,
        n_layers=1,
        n_heads=1,
        d_ff=d_ff,
        total_params=tot,
        nonzero_params=nz,
        file_bytes=0,
        layer_heads=[[{"dk": 2, "dv": 2, "mask": "none"}]],
        layer_d_ff=[d_ff],
        masks=["none"],
        readout_mode="all",
    )


def test_planted_relu_gadgets_and_dataflow_sccs():
    spec = _make_planted_circuit()
    an = analyze_looped_circuit(spec)

    gad = an["relu_gadgets"]
    assert gad["total_neurons"] == 6
    assert gad["active_neurons"] == 5
    assert gad["dead_neurons"] == 1
    assert gad["step_threshold_pairs"] == 1
    assert gad["cancellation_pairs"] == 1
    assert gad["abs_like_pairs"] == 1
    assert gad["matched_gadget_neurons"] == 4
    assert gad["unmatched_active_neurons"] == 1

    df = an["dataflow"]
    assert df["readonly_input_or_pos_channels"] == [0, 4]
    assert df["recurrent_scratchpad_channels"] == [1, 2]
    assert df["write_only_readout_channels"] == [3]
    assert df["unused_dead_count"] == 1
    assert df["n_nontrivial_sccs"] == 1
    assert df["largest_scc_size"] == 2

    att = an["attention"]
    assert att["n_hard_attention_heads"] == 1
    assert att["heads"][0]["key_regime"] == "fixed_input_or_pos"
    assert att["heads"][0]["qk_peak_abs"] == pytest.approx(2500.0 / np.sqrt(2.0), rel=1e-4)

    anat = an["anatomy"]
    assert anat["sparsity_ratio"] > 0.7
    assert anat["half_integer_fraction"] > 0.0

    clf = classify_looped_circuit(an)
    assert clf["predicted_archetype"] == "compact_spatial_or_graph_propagator"
    assert clf["confidence"] > 0.5
    assert clf["evidence"]


def test_sparse_npz_roundtrip_blind_load(tmp_path: Path):
    # Save a sparse LoopedWSP-formatted .npz + .json and verify blind load strips task names
    spec = _make_planted_circuit()
    arrays = {}
    for k, m in spec.matrices.items():
        if m.ndim == 2:
            flat_idx = m.rows * m.shape[1] + m.cols
        else:
            flat_idx = m.cols
        arrays[f"{k}.shape"] = np.array(m.shape, dtype=np.int64)
        arrays[f"{k}.idx"] = flat_idx.astype(np.int64)
        arrays[f"{k}.val"] = m.vals.astype(np.float64)
    np.savez(tmp_path / "secret_maze_task.npz", **arrays)
    cfg = {
        "name": "secret_maze_task",
        "d_model": 6,
        "layers": [[[2, 2]]],
        "mask": "none",
        "readout": "all",
        "fields": ["row"],
        "io": {"charset": "# SGo"},
    }
    (tmp_path / "secret_maze_task.json").write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_looped_weights(tmp_path / "secret_maze_task.npz", blind=True)
    assert loaded.source_path == ""
    assert "name" not in loaded.raw_config
    assert "io" not in loaded.raw_config
    assert loaded.d_model == 6
    an = analyze_looped_circuit(loaded)
    assert an["relu_gadgets"]["step_threshold_pairs"] == 1
    assert an["dataflow"]["largest_scc_size"] == 2


@pytest.mark.skipif(not LOOPED_WSP_DIR.exists(), reason="work/LoopedWSP/weights not cloned locally")
def test_real_looped_wsp_maze_and_hanoi_reading():
    maze_1k = load_looped_weights(LOOPED_WSP_DIR / "more_networks" / "maze_size_1k", blind=True)
    assert maze_1k.total_params == 928
    assert maze_1k.nonzero_params == 136
    an_maze = analyze_looped_circuit(maze_1k)
    assert an_maze["attention"]["n_hard_attention_heads"] >= 1
    assert an_maze["dataflow"]["largest_scc_size"] >= 2
    assert classify_looped_circuit(an_maze)["predicted_archetype"] == "compact_spatial_or_graph_propagator"

    hanoi = load_looped_weights(LOOPED_WSP_DIR / "networks" / "hanoi.npz", blind=True)
    assert hanoi.nonzero_params == 3686
    an_hanoi = analyze_looped_circuit(hanoi)
    assert an_hanoi["attention"]["n_uniform_pool_heads"] == 1
    assert an_hanoi["relu_gadgets"]["step_threshold_pairs"] >= 40
    assert classify_looped_circuit(an_hanoi)["predicted_archetype"] == "causal_autoregressive_state_machine"
