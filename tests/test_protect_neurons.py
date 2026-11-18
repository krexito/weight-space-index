"""Column selection of the protected loop fix (scripts/protect_neurons.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_select_skips_and_drops_harmful_columns():
    import protect_neurons as pn

    c_rank = list(range(20))  # column 0 pushes along the loop direction most
    harm = np.zeros(20)
    harm[[1, 3]] = [5.0, 4.0]  # the two most harmful columns (top 10% of 20 = 2)
    harm[[5]] = -9.0  # helps normal reasoning: never protected
    assert pn.select(c_rank, harm, "skip", 0.10, n=5) == [0, 2, 4, 5, 6]
    assert pn.select(c_rank, harm, "drop", 0.10, n=5) == [0, 2, 4]
    # nothing harmful (all <= 0) -> same columns as the original top-n
    assert pn.select(c_rank, -np.abs(harm), "skip", 0.25, n=5) == [0, 1, 2, 3, 4]


def test_variants_are_frozen():
    import protect_neurons as pn

    assert pn.VARIANTS == {"skip10": ("skip", 0.10), "skip25": ("skip", 0.25), "drop10": ("drop", 0.10)}
    assert (pn.LAYERS, pn.ALPHA, pn.N_COLS) == ((11, 15, 7), 6.0, 64)
