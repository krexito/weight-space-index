"""CPU unit tests for VLM / Audio-LLM backbone discovery in wsi.xray."""
from __future__ import annotations

from types import SimpleNamespace
import torch
import torch.nn as nn

from wsi.xray import _discover_xray_structure


class _ToyBlock(nn.Module):
    def __init__(self, d: int = 32, inter: int = 64):
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


class _ToyNestedVLM(nn.Module):
    """Mimics Qwen2.5-VL / Gemma-3 / SmolVLM nested model.language_model + visual/audio towers."""

    def __init__(self, n_text: int = 4, n_vis: int = 6, d: int = 32, vocab: int = 64):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen2_5_vl",
            text_config=SimpleNamespace(
                model_type="qwen2_5_vl_text",
                num_hidden_layers=n_text,
                hidden_size=d,
                num_attention_heads=4,
                num_key_value_heads=2,
            ),
        )
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([_ToyBlock(d) for _ in range(n_vis)])
        self.model.audio_tower = nn.Module()
        self.model.audio_tower.layers = nn.ModuleList([_ToyBlock(d) for _ in range(n_vis)])
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(vocab, d)
        self.model.language_model.layers = nn.ModuleList([_ToyBlock(d) for _ in range(n_text)])
        self.model.language_model.norm = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_discover_nested_vlm_ignores_vision_and_audio_towers():
    m = _ToyNestedVLM(n_text=4, n_vis=6, d=32, vocab=64)
    tok = SimpleNamespace(chat_template=None)
    struct = _discover_xray_structure(m, tok, "Toy/Nested-VLM")
    assert len(struct.blocks) == 4
    assert struct.hidden_size == 32
    assert struct.final_norm is m.model.language_model.norm
    assert struct.embed is m.model.language_model.embed_tokens
    assert struct.unembed is m.lm_head
