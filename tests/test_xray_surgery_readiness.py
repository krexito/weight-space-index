"""CPU unit test for `surgery_readiness` integrated into `wsi.xray.scan_model_xray`."""
from __future__ import annotations

from types import SimpleNamespace
import torch
import torch.nn as nn

from wsi.xray import _discover_xray_structure, _scan_surgery_readiness


class _ToyLayer(nn.Module):
    def __init__(self, d: int = 16, inter: int = 32):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d, bias=False)
        self.self_attn.k_proj = nn.Linear(d, d, bias=False)
        self.self_attn.v_proj = nn.Linear(d, d, bias=False)
        self.self_attn.o_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(d, inter, bias=False)
        self.mlp.up_proj = nn.Linear(d, inter, bias=False)
        self.mlp.down_proj = nn.Linear(inter, d, bias=False)
        self.input_layernorm = nn.RMSNorm(d)
        self.post_attention_layernorm = nn.RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layernorm(x)
        x = x + 0.1 * self.self_attn.o_proj(self.self_attn.v_proj(h))
        h2 = self.post_attention_layernorm(x)
        x = x + 0.1 * self.mlp.down_proj(torch.nn.functional.silu(self.mlp.gate_proj(h2)) * self.mlp.up_proj(h2))
        return x


class _ToyLM(nn.Module):
    def __init__(self, n_layers: int = 8, d: int = 16, vocab: int = 64):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=n_layers,
            hidden_size=d,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList([_ToyLayer(d) for _ in range(n_layers)])
        self.model.norm = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache=False):
        x = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            x = blk(x)
        logits = self.lm_head(self.model.norm(x))
        return SimpleNamespace(logits=logits)


def test_scan_surgery_readiness_returns_8_bins_and_best_site():
    torch.manual_seed(0)
    m = _ToyLM(n_layers=8, d=16, vocab=64).eval()
    tok = SimpleNamespace(pad_token_id=0, chat_template=None)
    struct = _discover_xray_structure(m, tok, "Toy/LM")
    cal_seqs = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    sr = _scan_surgery_readiness(m, struct, tok, cal_seqs, None, {}, torch.device("cpu"))
    assert sr["probe_dir_source"] == "isotropic_unit"
    assert len(sr["causal_gain_8"]) == 8
    assert len(sr["hydra_retention_8"]) == 8
    assert len(sr["effective_gain_8"]) == 8
    assert 0 <= sr["best_edit_layer"] < 8
    assert sr["best_effective_gain"] > 0.0
    assert all(cg > 0.0 for cg in sr["causal_gain_8"])
