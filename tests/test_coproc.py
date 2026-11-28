"""CPU unit & integration tests for `wsi.coproc` (.moicoproc artifacts, MoI router, executor)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from wsi.coproc import (
    CoProcessorExecutor,
    MoICoProcessor,
    fit_coproc_router,
    route_and_generate,
    verify_hanoi_moves,
    verify_maze_solution,
)

ROOT = Path(__file__).resolve().parent.parent
LOOPED_WEIGHTS = ROOT / "work" / "LoopedWSP" / "weights"


def _make_9x9_maze(seed: int = 7) -> str:
    rng = random.Random(seed)
    n = 9
    g, queue = [["#"] * n for _ in range(n)], [(1, 1)]
    g[1][1] = " "
    while queue:
        r, c = queue[-1]
        nxt = [
            (r + a, c + b)
            for a, b in ((2, 0), (-2, 0), (0, 2), (0, -2))
            if 0 < r + a < n - 1 and 0 < c + b < n - 1 and g[r + a][c + b] == "#"
        ]
        if not nxt:
            queue.pop()
            continue
        r2, c2 = rng.choice(nxt)
        g[(r + r2) // 2][(c + c2) // 2] = g[r2][c2] = " "
        queue.append((r2, c2))
    open_cells = [(r, c) for r in range(n) for c in range(n) if g[r][c] == " "]
    (a, b), (c, d) = rng.sample(open_cells, 2)
    g[a][b], g[c][d] = "S", "G"
    return "\n".join("".join(row) for row in g)


class _ToyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "left"
        self.chat_template = None

    def __call__(self, texts: list[str], return_tensors: str = "pt", padding: bool = True, truncation: bool = False):
        # Assign token 2 for puzzle instances ("#"), token 3 for near-domain talk ("maze"), token 4 for everyday
        seqs = []
        for t in texts:
            if "#" in t and "S" in t and "G" in t:
                seqs.append([4, 2, 2])
            elif "maze" in t.lower() or "hanoi" in t.lower():
                seqs.append([4, 3, 3])
            else:
                seqs.append([4, 5])
        L = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = [self.pad_token_id] * (L - len(s))
            if self.padding_side == "left":
                ids.append(pad + s)
                mask.append([0] * len(pad) + [1] * len(s))
            else:
                ids.append(s + pad)
                mask.append([1] * len(s) + [0] * len(pad))
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.tensor(mask, dtype=torch.long)}

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        vals = [int(x) for x in token_ids.tolist() if not (skip_special_tokens and int(x) in (0, 1))]
        return "base:" + "-".join(map(str, vals))


class _ToyLM(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(16, hidden)
        # Map distinct last-token IDs into orthogonal hidden subspaces:
        # tok 2 (puzzle) -> +e0; tok 3 (talk-about-puzzle) -> +e0 + 3*e1; tok 5 (everyday) -> -e0
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[2, 0] = 3.0
            self.embed.weight[3, 0] = 3.0
            self.embed.weight[3, 1] = 4.0
            self.embed.weight[5, 0] = -3.0
        self.head = nn.Linear(hidden, 16, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, output_hidden_states: bool = False):
        h = self.embed(input_ids)
        logits = self.head(h)
        hs = (h, h * 1.1) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hs)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                 max_new_tokens: int = 4, do_sample: bool = False, pad_token_id: int = 0):
        # Deterministic echo of last input token + 7
        last = (input_ids[:, -1:] + 7) % 15 + 1
        extra = last.repeat(1, min(2, max_new_tokens))
        return torch.cat([input_ids, extra], dim=1)


def test_moicoproc_save_load_and_tamper_detection(tmp_path: Path):
    weights = {"L0.W1": torch.eye(12, dtype=torch.float32), "L0.b1": torch.zeros(12, dtype=torch.float32)}
    router = {
        "layer": 1,
        "readout": "last_token",
        "dir": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "thr": 0.5,
        "veto_dir": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "veto_thr": 1.0,
    }
    cp = MoICoProcessor(
        weights=weights,
        router=router,
        proj_in=torch.ones(4, 12, dtype=torch.float32),
        spec={"task": "synthetic_tiny", "sparse_payload": False},
        meta={"name": "tiny_test", "n_params": 156, "nonzero_params": 12},
    )
    assert cp.weights_payload_bytes() < 4096
    p = tmp_path / "tiny.moicoproc"
    cp.save(p)

    loaded = MoICoProcessor.load(p, verify=True)
    assert loaded.verify_integrity() is True
    assert torch.allclose(loaded.weights["L0.W1"], weights["L0.W1"])
    assert loaded.proj_in is not None and loaded.proj_in.shape == (4, 12)
    assert "tiny_test" in loaded.describe()

    # Tamper with w.L0.W1 inside the safetensors archive and verify load rejects it
    tampered_tensors = {}
    with safe_open(str(p), "pt") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            tampered_tensors[k] = f.get_tensor(k).clone()
    tampered_tensors["w.L0.W1"][0, 0] = 999.0
    bad_p = tmp_path / "tampered.moicoproc"
    save_file(tampered_tensors, str(bad_p), metadata=meta)

    with pytest.raises(ValueError, match="integrity check failed"):
        MoICoProcessor.load(bad_p, verify=True)


def test_fit_coproc_router_and_route_and_generate_synthetic(tmp_path: Path):
    model = _ToyLM(hidden=8).eval()
    tok = _ToyTokenizer()

    # Calibration samples
    pos_prompts = ["Solve this maze:\n#####\n#S G#\n#####", "Solve:\n#####\n#S G#\n#####"] * 3
    neg_prompts = ["What is the capital of France?", "Explain photosynthesis simply."] * 3
    adv_prompts = ["Explain how BFS solves a maze in theory.", "What is the history of maze algorithms?"] * 3

    def hidden_last(prompts_list):
        enc = tok(prompts_list)
        with torch.no_grad():
            out = model(enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
        return out.hidden_states[1][:, -1]

    h_pos = hidden_last(pos_prompts)
    h_neg = hidden_last(neg_prompts)
    h_adv = hidden_last(adv_prompts)

    router = fit_coproc_router(h_pos, h_neg, h_veto_neg=h_adv, layer=1, target_fpr=0.0)
    cp = MoICoProcessor(
        weights={"L0.w": torch.zeros(100, dtype=torch.float32)},
        router=router,
        spec={"task": "toy_solver"},
        meta={"name": "toy_coproc"},
    )
    assert cp.payload_kb() < 4.0

    # Check router decision directly
    assert bool(cp.fires(hidden_last(pos_prompts[:1])).item()) is True
    assert bool(cp.fires(hidden_last(neg_prompts[:1])).item()) is False
    assert bool(cp.fires(hidden_last(adv_prompts[:1])).item()) is False  # vetoed!

    # Register synthetic executor and test route_and_generate
    exec_obj = CoProcessorExecutor()
    exec_obj.register(
        "toy_solver",
        lambda _cp, prompt, check_fixed_point=True: (
            {"solved": True, "solution": "SOLVED:" + prompt.splitlines()[-1], "loop_steps": 5, "fixed_point_ok": True}
            if "SOLVE_OK" in prompt
            else {"solved": False, "solution": "", "loop_steps": 0, "fixed_point_ok": False, "error": "unparseable"}
        ),
    )

    test_prompts = [
        "Explain photosynthesis.",
        "How does BFS work on a maze?",
        "SOLVE_OK\n#####\n#S G#\n#####",
        "UNPARSEABLE\n#####\n#S G#\n#####",
    ]
    res = route_and_generate(model, tok, test_prompts, cp, exec_obj, max_new_tokens=4, batch_size=2)
    assert res["stats"]["n_routed_solved"] == 1
    assert res["stats"]["n_routed_fallback"] == 1
    assert res["stats"]["n_base_untouched"] == 2
    assert res["answers"][2].startswith("SOLVED:")

    # Off-domain & near-domain prompts must match base model bit-for-bit (0% drift)
    for idx in (0, 1):
        enc = tok([test_prompts[idx]])
        base_out = tok.decode(model.generate(enc["input_ids"], max_new_tokens=4)[0, enc["input_ids"].shape[1]:])
        assert res["answers"][idx] == base_out


@pytest.mark.skipif(not LOOPED_WEIGHTS.exists(), reason="work/LoopedWSP/weights not cloned locally")
def test_real_looped_wsp_moicoproc_payload_and_execution(tmp_path: Path):
    # 1. Verify ultra-compact 928-param Maze fits under 4 KB in float32 (3,712 bytes = 3.625 KB)
    cp_1k = MoICoProcessor.from_looped_weights(
        LOOPED_WEIGHTS / "more_networks" / "maze_size_1k",
        task="maze_1k",
        dtype=torch.float32,
        sparse_payload=False,
    )
    assert cp_1k.weights_payload_bytes() == 928 * 4
    assert cp_1k.weights_payload_bytes() < 4096

    # 2. Bundle size-agnostic maze.npz (sparse fp64) + hanoi.npz (sparse fp64) and execute both!
    cp_maze = MoICoProcessor.from_looped_weights(
        LOOPED_WEIGHTS / "networks" / "maze.npz",
        task="maze",
        dtype=torch.float64,
        sparse_payload=True,
    )
    p_maze = tmp_path / "maze.moicoproc"
    cp_maze.save(p_maze)
    cp_maze_loaded = MoICoProcessor.load(p_maze, verify=True)
    assert cp_maze_loaded.weights_payload_bytes() < 16 * 1024

    maze_q = _make_9x9_maze(seed=11)
    ex = CoProcessorExecutor()
    out_m = ex.run(cp_maze_loaded, f"Find shortest path:\n{maze_q}", check_fixed_point=True)
    assert out_m["solved"] is True
    assert out_m["fixed_point_ok"] is True
    assert verify_maze_solution(maze_q, out_m["solution"]) is True

    cp_hanoi = MoICoProcessor.from_looped_weights(
        LOOPED_WEIGHTS / "networks" / "hanoi.npz",
        task="hanoi",
        dtype=torch.float64,
        sparse_payload=True,
    )
    out_h = ex.run(cp_hanoi, "Solve the Tower of Hanoi puzzle with 4 disks.", check_fixed_point=True)
    assert out_h["solved"] is True
    assert out_h["fixed_point_ok"] is True
    assert verify_hanoi_moves(4, out_h["solution"]) is True
