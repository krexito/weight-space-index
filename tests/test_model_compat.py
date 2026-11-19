"""CPU tests: the discovery pipeline finds the MLP down projection on Qwen-style (mlp.down_proj) and LFM-style
(feed_forward.w2) models, swaps it in memory by name, and takes sampling from the run's config."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import discover_fast as fast  # noqa: E402
import discover_loop_fix as dl  # noqa: E402
import moi_loop_router_vllm as lr  # noqa: E402


class QwenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(6, 4, bias=False)


class LfmFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1, self.w3, self.w2 = nn.Linear(4, 6, bias=False), nn.Linear(4, 6, bias=False), nn.Linear(6, 4, bias=False)


class Layer(nn.Module):
    def __init__(self, lfm: bool):
        super().__init__()
        if lfm:
            self.feed_forward = LfmFF()
        else:
            self.mlp = QwenMLP()


class Model(nn.Module):
    def __init__(self, lfm: bool, n=3):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Layer(lfm) for _ in range(n)])


@pytest.mark.parametrize("lfm", [False, True])
def test_down_module_and_pattern(lfm):
    m = Model(lfm)
    assert [lr.down_module(lyr).weight.shape for lyr in m.model.layers] == [torch.Size([4, 6])] * 3
    hits = sorted(int(fast.DOWN_RE.search(n).group(1)) for n, _ in m.named_parameters() if fast.DOWN_RE.search(n))
    assert hits == [0, 1, 2]  # w1/w3 (or nothing else) never match
    keys = [n for n, _ in m.named_parameters()]
    assert dl.down_key(keys, 1).endswith("layers.1.feed_forward.w2.weight" if lfm else "layers.1.mlp.down_proj.weight")


@pytest.mark.parametrize("lfm", [False, True])
def test_make_setter_writes_the_down_projection(lfm):
    m = Model(lfm)
    new = np.full((4, 6), 0.5, dtype=np.float32)
    done = fast.make_setter({2: new})(m)
    assert set(done) == {2}
    assert torch.equal(lr.down_module(m.model.layers[2]).weight.data, torch.from_numpy(new))
    assert not torch.equal(lr.down_module(m.model.layers[1]).weight.data, torch.from_numpy(new))


def test_down_module_rejects_unknown_layer():
    with pytest.raises(AttributeError):
        lr.down_module(nn.Linear(2, 2))


def test_sampling_from_config():
    assert lr.sampling_for({}) == lr.SAMPLING  # old runs: Qwen setting
    assert lr.sampling_for({"sampling": lr.SAMPLING_PROFILES["lfm3-math"]}) == {"temperature": 1.0, "top_p": 0.95,
                                                                                 "top_k": 20}
