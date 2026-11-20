"""CPU unit tests for Level-1 direct & Level-2 causal internal-direction attribution."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_surgical_refusal_lfm3 import _TinyLFM3ToyLM  # noqa: E402
from wsi import (  # noqa: E402
    CompactTrace,
    Edit,
    MoIPatch,
    attribute_direction_writers,
    capture_causal_trace,
    diagnose_causal_traces,
    peel_causal_circuit,
    summarize_direction_traces,
    trace_direction_graph,
)


class _DirAttn(nn.Module):
    def __init__(self, hidden: int, n_heads: int, *, bias: bool = False) -> None:
        super().__init__()
        self.num_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        nh, hd = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
        attn = torch.softmax(scores + mask, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, H)
        return self.o_proj(ctx)


class _DirMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, *, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DirBlock(nn.Module):
    def __init__(self, hidden: int, inter: int, n_heads: int, *, bias: bool = False, pre_norm: bool = False) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden) if pre_norm else nn.Identity()
        self.self_attn = _DirAttn(hidden, n_heads, bias=bias)
        self.mlp = _DirMLP(hidden, inter, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.pre_norm(x))
        x = x + self.mlp(self.pre_norm(x))
        return x


class _DirToyLM(nn.Module):
    def __init__(
        self,
        vocab: int = 32,
        hidden: int = 24,
        inter: int = 48,
        n_layers: int = 3,
        n_heads: int = 4,
        *,
        bias: bool = False,
        final_norm: bool = False,
        pre_norm: bool = False,
        use_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", hidden_size=hidden, num_attention_heads=n_heads)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList(
            [_DirBlock(hidden, inter, n_heads, bias=bias, pre_norm=pre_norm) for _ in range(n_layers)]
        )
        self.model.norm = nn.LayerNorm(hidden) if final_norm else nn.Identity()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.is_gradient_checkpointing = use_ckpt

    def gradient_checkpointing_enable(self) -> None:
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.is_gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        **_kw,
    ) -> SimpleNamespace:
        h = self.model.embed_tokens(input_ids)
        hs = [h] if output_hidden_states else None
        for idx, blk in enumerate(self.model.layers):
            if self.is_gradient_checkpointing and torch.is_grad_enabled():
                h = torch_checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
            if hs is not None and idx < len(self.model.layers) - 1:
                hs.append(h)
        h_out = self.model.norm(h)
        if hs is not None:
            hs.append(h_out)
        return SimpleNamespace(logits=self.lm_head(h_out), hidden_states=tuple(hs) if hs is not None else None)


def _rand_init(model: nn.Module, seed: int = 0, std: float = 0.2) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 1:
                p.normal_(0.0, std, generator=g)


def test_direct_attribution_residual_completeness_and_head_neuron_sums() -> None:
    m = _DirToyLM(vocab=32, hidden=24, inter=48, n_layers=3, n_heads=4, bias=True, final_norm=True).eval()
    _rand_init(m, seed=7, std=0.18)
    ids = [1, 4, 7, 11, 15]
    tdir = torch.randn(24, generator=torch.Generator().manual_seed(3))

    res_blk = attribute_direction_writers(
        m,
        ids,
        target_dir=tdir,
        target_layer=1,
        layer_convention="block",
        top_k_neurons=9999,
        top_k_heads=9999,
    )
    assert abs(res_blk["residual_gap"]) < 1e-4, res_blk
    assert len(res_blk["modules"]) == 4  # blocks 0 & 1: (attn + mlp) * 2

    # Check exact neuron + bias sum per MLP module and head + bias sum per Attn module
    mod_by_name = {r["module"]: r for r in res_blk["modules"]}
    for mk, row in mod_by_name.items():
        if row["kind"] == "mlp":
            n_sum = sum(nr["contrib"] for nr in res_blk["top_neurons"] if nr["module"] == mk)
            assert abs((n_sum + row["bias_contrib"]) - row["contrib"]) < 1e-4, (mk, n_sum, row)
        elif row["kind"] == "attn":
            h_sum = sum(hr["contrib"] for hr in res_blk["top_heads"] if hr["module"] == mk)
            assert abs((h_sum + row["bias_contrib"]) - row["contrib"]) < 1e-4, (mk, h_sum, row)

    # hidden_states layer l + 1 matches block l bit-for-bit
    res_hs = attribute_direction_writers(
        m,
        ids,
        target_dir=tdir,
        target_layer=2,
        layer_convention="hidden_states",
        top_k_neurons=9999,
        top_k_heads=9999,
    )
    assert abs(res_hs["target_score"] - res_blk["target_score"]) < 1e-5
    assert res_hs["module_contribs"] == res_blk["module_contribs"]

    # hidden_states layer 0 = pure embeddings (no upstream writers)
    res_l0 = attribute_direction_writers(m, ids, target_dir=tdir, target_layer=0, layer_convention="hidden_states")
    assert res_l0["modules"] == []
    assert res_l0["top_heads"] == []
    assert res_l0["top_neurons"] == []
    assert abs(res_l0["embed_contrib"] - res_l0["target_score"]) < 1e-5

    # hidden_states layer n_layers = post-final-norm hidden_states[-1]
    with torch.no_grad():
        out = m(torch.tensor([ids]), output_hidden_states=True)
        expected_final = float((out.hidden_states[-1][0, -1] @ tdir).item())
    res_final = attribute_direction_writers(m, ids, target_dir=tdir, target_layer=3, layer_convention="hidden_states")
    assert abs(res_final["target_score"] - expected_final) < 1e-5


def test_direct_vs_causal_indirect_circuit_separation() -> None:
    m = _DirToyLM(vocab=32, hidden=16, inter=32, n_layers=3, n_heads=4, bias=False, final_norm=False).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        # Token 15 at pos=2 carries coord 2 = 2.0; normal tokens (10..14) carry coord 0 = 1.0 (unreferenced).
        m.model.embed_tokens.weight[10:15, 0] = 1.0
        m.model.embed_tokens.weight[15, 2] = 2.0

        # Layer 0 MLP reads coord 2 -> writes orthogonal relay coord 1 (1 perp target_dir=e_3!)
        b0 = m.model.layers[0]
        b0.mlp.gate_proj.weight[0, 2] = 2.0
        b0.mlp.up_proj.weight[0, 2] = 2.0
        b0.mlp.down_proj.weight[1, 0] = 1.5

        # Layer 1 Attn head 0 copies relay coord 1 across sequence positions via uniform causal attention (q=k=0)
        b1 = m.model.layers[1]
        b1.self_attn.v_proj.weight[0, 1] = 2.0
        b1.self_attn.o_proj.weight[1, 0] = 1.5

        # Layer 2 MLP reads relay coord 1 -> writes directly into target_dir = e_3!
        b2 = m.model.layers[2]
        b2.mlp.gate_proj.weight[0, 1] = 2.0
        b2.mlp.up_proj.weight[0, 1] = 2.0
        b2.mlp.down_proj.weight[3, 0] = 2.0

    ids = [10, 11, 15, 12]  # trigger token is at pos=2; query target is last token pos=3
    tdir = torch.zeros(16)
    tdir[3] = 1.0

    direct = attribute_direction_writers(m, ids, target_dir=tdir, target_layer=2, layer_convention="block")
    assert abs(direct["residual_gap"]) < 1e-4
    # Only layers.2.mlp.down_proj writes directly into e_3; layers.0.mlp & layers.1.attn write orthogonal e_1!
    assert direct["modules"][0]["module"] == "layers.2.mlp.down_proj"
    assert direct["module_contribs"]["layers.2.mlp.down_proj"] > 0.1
    assert abs(direct["module_contribs"]["layers.0.mlp.down_proj"]) < 1e-5
    assert abs(direct["module_contribs"]["layers.1.self_attn.o_proj"]) < 1e-5

    ct = capture_causal_trace(m, ids, target_dir=tdir, target_layer=2, target_layer_convention="block")
    assert ct.meta["grad_objective"] == "direction"
    assert abs(ct.meta["target_dir_score"] - direct["target_score"]) < 1e-4

    gs2 = ct.grad_summary["layers.2.mlp.down_proj"]
    gs1 = ct.grad_summary["layers.1.self_attn.o_proj"]
    gs0 = ct.grad_summary["layers.0.mlp.down_proj"]

    # Final writer at block 2: causal == direct (zero indirect) and matches direct attribution
    assert abs(gs2["direct_attr"] - direct["module_contribs"]["layers.2.mlp.down_proj"]) < 1e-4
    assert abs(gs2["indirect_attr"]) < 1e-4
    assert abs(gs2["causal_attr"] - gs2["direct_attr"]) < 1e-4

    # Early indirect writers: direct == 0, indirect & causal > 0.1!
    assert abs(gs1["direct_attr"]) < 1e-5 and gs1["indirect_attr"] > 0.1 and gs1["causal_attr"] > 0.1
    assert abs(gs0["direct_attr"]) < 1e-5 and gs0["indirect_attr"] > 0.1 and gs0["causal_attr"] > 0.1

    # Input token attribution pinpoints trigger token at pos=2
    tok_ranks = ct.top_input_tokens(ids, top_k=4)
    assert tok_ranks[0]["pos"] == 2 and tok_ranks[0]["token_id"] == 15
    assert tok_ranks[0]["grad_x_input"] > 0.1


def test_causal_direction_finite_difference_exactness() -> None:
    m = _DirToyLM(vocab=32, hidden=24, inter=48, n_layers=3, n_heads=4, bias=True, final_norm=True, pre_norm=True).eval()
    _rand_init(m, seed=19, std=0.16)
    ids = [2, 6, 9, 14, 21]
    ids_t = torch.tensor([ids])
    tdir = torch.randn(24, generator=torch.Generator().manual_seed(5))

    ct = capture_causal_trace(m, ids, target_dir=tdir, target_layer=3, target_layer_convention="hidden_states")
    base_score = float(ct.meta["target_dir_score"])
    eps = 1e-3

    mods = {mk: mod for name, mod in m.named_modules() if any(name.endswith(mk) for mk in ct.grad_summary) for mk in [next(k for k in ct.grad_summary if name.endswith(k))]}

    def _eval_with_hooks(extra_hooks) -> float:
        with torch.no_grad():
            out = m(ids_t, output_hidden_states=True)
            return float((out.hidden_states[-1][0, -1] @ tdir).item())

    # 1. Verify full-module causal_attr matches finite-difference d/d(eps) of scaling y_mk by (1 - eps)
    for mk, mod in mods.items():
        def _scale_y(_m, _inp, out, eps=eps):
            return (1.0 - eps) * out

        h = mod.register_forward_hook(_scale_y)
        try:
            pert = _eval_with_hooks([])
        finally:
            h.remove()
        fd = (base_score - pert) / eps
        assert abs(fd - ct.grad_summary[mk]["causal_attr"]) < 2e-3, (mk, fd, ct.grad_summary[mk]["causal_attr"])

    # 2. Verify head_causal_attr[h] for an attention o_proj matches perturbing only head h
    mk_attn = "layers.1.self_attn.o_proj"
    mod_attn = mods[mk_attn]
    nh = int(ct.grad_summary[mk_attn]["n_heads"])
    hd = mod_attn.in_features // nh
    head_idx = 2

    def _scale_head_pre(_m, inp):
        x = inp[0].clone()
        x[:, :, head_idx * hd : (head_idx + 1) * hd] *= (1.0 - eps)
        return (x, *inp[1:])

    hh = mod_attn.register_forward_pre_hook(_scale_head_pre)
    try:
        pert_h = _eval_with_hooks([])
    finally:
        hh.remove()
    fd_h = (base_score - pert_h) / eps
    pred_h = float(ct.grad_summary[mk_attn]["head_causal_attr"][head_idx].item())
    assert abs(fd_h - pred_h) < 2e-3, (fd_h, pred_h)

    # 3. Verify neuron_supp_attr[j] for an MLP down_proj matches perturbing only neuron j
    mk_mlp = "layers.0.mlp.down_proj"
    mod_mlp = mods[mk_mlp]
    j = 5

    def _scale_neuron_pre(_m, inp):
        x = inp[0].clone()
        x[:, :, j] *= (1.0 - eps)
        return (x, *inp[1:])

    hm = mod_mlp.register_forward_pre_hook(_scale_neuron_pre)
    try:
        pert_n = _eval_with_hooks([])
    finally:
        hm.remove()
    fd_n = (base_score - pert_n) / eps
    pred_n = float(ct.grad_summary[mk_mlp]["neuron_supp_attr"][j].item())
    assert abs(fd_n - pred_n) < 2e-3, (fd_n, pred_n)


def test_contrast_direct_and_summarize_direction_traces() -> None:
    m = _DirToyLM(vocab=32, hidden=16, inter=32, n_layers=2, n_heads=4, bias=False, final_norm=False).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        # Neutral token 10: coord 0 = 1.0; toxic/positive token 25: coord 0 = 1.0, coord 2 = 2.0
        m.model.embed_tokens.weight[10:15, 0] = 1.0
        m.model.embed_tokens.weight[25:28, 0] = 1.0
        m.model.embed_tokens.weight[25:28, 2] = 2.0

        # Plant a baseline neuron (j=0) active on ALL tokens writing e_3, plus a separator head (L0, head 2)
        # AND a separator neuron (L1, j=7) active ONLY when coord 2 > 0!
        b0 = m.model.layers[0]
        b0.mlp.gate_proj.weight[0, 0] = 2.0
        b0.mlp.up_proj.weight[0, 0] = 2.0
        b0.mlp.down_proj.weight[3, 0] = 0.5
        # Head 2 occupies input slice 8:12 of o_proj (nh=4, hidden=16 -> hd=4)
        b0.self_attn.v_proj.weight[8, 2] = 2.0
        b0.self_attn.o_proj.weight[3, 8] = 2.0

        b1 = m.model.layers[1]
        b1.mlp.gate_proj.weight[7, 2] = 2.0
        b1.mlp.up_proj.weight[7, 2] = 2.0
        b1.mlp.down_proj.weight[3, 7] = 2.5

    tdir = torch.zeros(16)
    tdir[3] = 1.0
    pos_seqs = [[10, 11, 25], [12, 26, 27]]
    neg_seqs = [[10, 11, 12], [13, 14, 10]]

    c_direct = attribute_direction_writers(
        m,
        pos_seqs,
        contrast_inputs=neg_seqs,
        target_dir=tdir,
        target_layer=1,
        layer_convention="block",
    )
    assert c_direct["mode"] == "contrast"
    assert abs(c_direct["delta_residual_gap"]) < 1e-4
    assert abs(c_direct["delta_target_score"] - (c_direct["target_score"] - c_direct["contrast_target_score"])) < 1e-6
    assert c_direct["top_heads"][0]["module"] == "layers.0.self_attn.o_proj"
    assert c_direct["top_heads"][0]["head"] == 2
    assert c_direct["top_neurons"][0]["module"] == "layers.1.mlp.down_proj"
    assert c_direct["top_neurons"][0]["neuron"] == 7

    pos_traces = [capture_causal_trace(m, s, target_dir=tdir, target_layer=1, target_layer_convention="block") for s in pos_seqs]
    neg_traces = [capture_causal_trace(m, s, target_dir=tdir, target_layer=1, target_layer_convention="block") for s in neg_seqs]

    summary = summarize_direction_traces(pos_traces, contrast_traces=neg_traces)
    assert summary["mode"] == "contrast"
    assert summary["n_traces"] == 2 and summary["n_contrast"] == 2
    assert summary["delta_target_score"] > 1.0
    assert summary["top_heads"][0]["module"] == "layers.0.self_attn.o_proj"
    assert summary["top_heads"][0]["head"] == 2
    assert summary["top_neurons"][0]["module"] == "layers.1.mlp.down_proj"
    assert summary["top_neurons"][0]["neuron"] == 7


def test_direction_trace_roundtrip_diagnose_and_grad_checkpointing() -> None:
    m_nogc = _DirToyLM(vocab=32, hidden=24, inter=48, n_layers=3, n_heads=4, bias=False, final_norm=True, use_ckpt=False).eval()
    _rand_init(m_nogc, seed=29, std=0.18)

    m_gc = _DirToyLM(vocab=32, hidden=24, inter=48, n_layers=3, n_heads=4, bias=False, final_norm=True, use_ckpt=False).eval()
    m_gc.load_state_dict(m_nogc.state_dict())

    ids1 = [1, 4, 8, 12, 20]
    ids2 = [3, 5, 9, 14, 19]
    tdir = torch.randn(24, generator=torch.Generator().manual_seed(11))

    # Compare grad_checkpointing=False vs True across all layer conventions:
    # intermediate block (block=1), last block pre-final-norm (block=2), embedding layer (hs=0), post-norm (hs=3)
    for conv, lay in [("block", 1), ("block", 2), ("hidden_states", 0), ("hidden_states", 3)]:
        ct_a = capture_causal_trace(
            m_nogc, ids1, target_dir=tdir, target_layer=lay, target_layer_convention=conv, grad_checkpointing=False
        )
        ct_b = capture_causal_trace(
            m_gc, ids1, target_dir=tdir, target_layer=lay, target_layer_convention=conv, grad_checkpointing=True
        )
        assert abs(ct_a.meta["target_dir_score"] - ct_b.meta["target_dir_score"]) < 1e-5
        assert set(ct_a.grad_summary) == set(ct_b.grad_summary)
        for mk in ct_a.grad_summary:
            assert abs(ct_a.grad_summary[mk]["causal_attr"] - ct_b.grad_summary[mk]["causal_attr"]) < 1e-4
        if ct_a.token_attr and ct_b.token_attr:
            assert torch.allclose(ct_a.token_attr["grad_x_input"], ct_b.token_attr["grad_x_input"], atol=1e-4)

    # Save/load roundtrip preserves token_attr, head_causal_attr, direct/indirect attrs, and meta
    ct1 = capture_causal_trace(m_nogc, ids1, target_dir=tdir, target_layer=2, target_layer_convention="block")
    ct2 = capture_causal_trace(m_nogc, ids2, target_dir=tdir, target_layer=2, target_layer_convention="block")
    with tempfile.TemporaryDirectory() as td:
        p = ct1.save(Path(td) / "dir_trace")
        loaded = CompactTrace.load(p)
        assert loaded.meta == ct1.meta
        assert loaded.byte_breakdown() == ct1.byte_breakdown()
        assert torch.equal(loaded.token_attr["grad_x_input"], ct1.token_attr["grad_x_input"])
        assert loaded.top_input_tokens(ids1, top_k=3) == ct1.top_input_tokens(ids1, top_k=3)
        for mk in ct1.grad_summary:
            assert loaded.grad_summary[mk]["causal_attr"] == ct1.grad_summary[mk]["causal_attr"]
            assert loaded.grad_summary[mk]["direct_attr"] == ct1.grad_summary[mk]["direct_attr"]
            if "head_causal_attr" in ct1.grad_summary[mk]:
                assert torch.equal(loaded.grad_summary[mk]["head_causal_attr"], ct1.grad_summary[mk]["head_causal_attr"])

    # diagnose_causal_traces works both without healthy_span and with an explicit healthy_span
    diag_no_h = diagnose_causal_traces([ct1, ct2], split_ab=True, top_k_sites=4)
    assert diag_no_h["summary"]["top_causal_site"] in ct1.grad_summary

    ct_h1 = capture_causal_trace(
        m_nogc, ids1, target_dir=tdir, target_layer=2, target_layer_convention="block", healthy_span=(0, 3)
    )
    ct_h2 = capture_causal_trace(
        m_nogc, ids2, target_dir=tdir, target_layer=2, target_layer_convention="block", healthy_span=(0, 3)
    )
    diag_with_h = diagnose_causal_traces([ct_h1, ct_h2], split_ab=True, top_k_sites=4)
    assert diag_with_h["summary"]["top_causal_site"] in ct_h1.grad_summary


def test_hybrid_lfm3_moipatch_veto_dir_integration() -> None:
    m = _TinyLFM3ToyLM().eval()
    hidden = m.config.hidden_size
    r_dir = torch.zeros(hidden)
    r_dir[3] = 1.0
    v_dir = torch.zeros(hidden)
    v_dir[4] = 1.0

    # Layer 3 under hidden_states convention = output of layers[2]
    patch = MoIPatch(
        edits=[Edit(module="layers.2.self_attn.o_proj", u=r_dir.clone(), alpha=1.0)],
        router={"layer": 3, "dir": r_dir, "thr": 0.5, "veto_dir": v_dir, "veto_thr": 0.5},
    )

    tox_seq = [30, 31]
    benign_trig_seq = [20, 21]

    direct = attribute_direction_writers(
        m,
        tox_seq,
        contrast_inputs=benign_trig_seq,
        target_dir=patch.router["veto_dir"],
        target_layer=patch.router["layer"],
        layer_convention="hidden_states",
    )
    assert abs(direct["residual_gap"]) < 1e-4
    # In _TinyLFM3ToyLM, coord 4 originates from toxic embeddings 30..37
    assert direct["delta_target_score"] > 1.0 and direct["delta_embed_contrib"] > 1.0

    # Conv layers 0 & 1 must be tagged kind='conv' and must NOT be split into 4 fake attention heads
    conv_rows = [r for r in direct["modules"] if "conv.out_proj" in r["module"]]
    assert len(conv_rows) == 2 and all(r["kind"] == "conv" for r in conv_rows)
    assert not any("conv.out_proj" in hr["module"] for hr in direct["top_heads"])

    attn_rows = [r for r in direct["modules"] if "self_attn.o_proj" in r["module"]]
    assert len(attn_rows) == 1 and attn_rows[0]["kind"] == "attn"

    ct = capture_causal_trace(
        m,
        tox_seq,
        target_dir=patch.router["veto_dir"],
        target_layer=patch.router["layer"],
        target_layer_convention="hidden_states",
    )
    assert ct.grad_summary["layers.0.conv.out_proj"]["kind"] == "conv"
    assert "head_causal_attr" not in ct.grad_summary["layers.0.conv.out_proj"]
    assert ct.grad_summary["layers.2.self_attn.o_proj"]["kind"] == "attn"
    assert ct.grad_summary["layers.2.self_attn.o_proj"]["n_heads"] == 4
    assert "layers.3.feed_forward.w2" not in ct.grad_summary
    assert len(ct.token_attr["grad_x_input"]) == len(tox_seq)
    assert ct.top_input_tokens(tox_seq, top_k=2)[0]["pos"] == len(tox_seq) - 1


# ----------------------------------------------------------------------------------------------- tracer upgrades
class _RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class _HalfBlock(_DirBlock):
    """Block with a planted non-additive residual scale (output * 0.5)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * super().forward(x)


def _zero_model(**kw) -> _DirToyLM:
    m = _DirToyLM(**kw).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    return m


def test_protect_dir_separates_shared_vs_clean_site() -> None:
    m = _zero_model(vocab=32, hidden=16, inter=32, n_layers=2, n_heads=4)
    with torch.no_grad():
        m.model.embed_tokens.weight[10:15, 0] = 1.0
        m.model.embed_tokens.weight[10:15, 2] = 4.0
        b0, b1 = m.model.layers[0], m.model.layers[1]
        # L0 neuron 0: coord0 -> writes BOTH target e3 and protected e5
        b0.mlp.gate_proj.weight[0, 0] = 2.0
        b0.mlp.up_proj.weight[0, 0] = 2.0
        b0.mlp.down_proj.weight[3, 0] = 2.0
        b0.mlp.down_proj.weight[5, 0] = 2.0
        # L1 neuron 0: clean target writer coord2 -> e3 only (smaller than L0)
        b1.mlp.gate_proj.weight[0, 2] = 0.5
        b1.mlp.up_proj.weight[0, 2] = 0.5
        b1.mlp.down_proj.weight[3, 0] = 1.5
    tdir = torch.zeros(16)
    tdir[3] = 1.0
    pdir = torch.zeros(16)
    pdir[5] = 1.0
    seqs = [[10, 11, 12], [12, 13, 14]]

    plain = [capture_causal_trace(m, s, target_dir=tdir, target_layer=1) for s in seqs]
    d_plain = diagnose_causal_traces(plain, split_ab=True, top_k_sites=4, model_or_weights=m)
    assert d_plain["summary"]["top_causal_site"] == "layers.0.mlp.down_proj", d_plain["sites"]

    for cost in ("linear", "fisher"):
        prot = [
            capture_causal_trace(
                m, s, target_dir=tdir, target_layer=1, protect_dir=pdir,
                healthy_cost=cost, grad_mode="compressed",
            )
            for s in seqs
        ]
        assert prot[0].meta["grad_healthy_objective"] == "direction"
        assert prot[0].meta["protect_layer"] == 1 and prot[0].meta["protect_dir_score"] > 0.1
        d_prot = diagnose_causal_traces(prot, split_ab=True, top_k_sites=4, healthy_cost=cost, model_or_weights=m)
        rank = d_prot["causal_rank"]
        assert rank[0] == "layers.1.mlp.down_proj", (cost, d_prot["sites"])
        assert rank.index("layers.1.mlp.down_proj") < rank.index("layers.0.mlp.down_proj")


def test_logit_margin_avoids_softmax_saturation_and_matches_fd() -> None:
    m = _zero_model(vocab=8, hidden=8, inter=16, n_layers=1, n_heads=2)
    with torch.no_grad():
        m.model.embed_tokens.weight[:, 0] = 1.0
        b0 = m.model.layers[0]
        b0.mlp.gate_proj.weight[0, 0] = 1.0
        b0.mlp.up_proj.weight[0, 0] = 1.0
        b0.mlp.down_proj.weight[0, 0] = 0.5
        m.lm_head.weight[0, 0] = 60.0
    ids = [2, 0]
    mk = "layers.0.mlp.down_proj"

    ct_nll = capture_causal_trace(m, ids, grad_mode="summary")
    assert ct_nll.meta["grad_objective"] == "token_nll"
    assert abs(ct_nll.grad_summary[mk]["causal_attr"]) < 1e-6  # softmax saturated: gradient vanished

    eps = 1e-3
    ids_t = torch.tensor([ids])

    def _margin(pos: int, scale: float) -> float:
        h = m.model.layers[0].mlp.down_proj.register_forward_hook(lambda _m, _i, o: o * scale)
        try:
            with torch.no_grad():
                lg = m(ids_t).logits[0, pos].double()
        finally:
            h.remove()
        return float(lg[0] - lg[1])

    for kw, pos in (({"target_token_ids": [0], "contrast_token_ids": [1]}, 1), ({"contrast_token_ids": [1]}, 0)):
        ct = capture_causal_trace(m, ids, grad_mode="summary", **kw)
        assert ct.meta["grad_objective"] == "logit_margin"
        c = ct.grad_summary[mk]["causal_attr"]
        assert abs(c) > 1.0
        fd = (_margin(pos, 1.0) - _margin(pos, 1.0 - eps)) / eps
        assert abs(fd - c) < 1e-2 * max(1.0, abs(c)), (kw, fd, c)
        assert abs(ct.meta["target_obj_score"] - _margin(pos, 1.0)) < 1e-3 * max(1.0, abs(_margin(pos, 1.0)))


def test_attribute_direction_writers_linearized_final_norm_and_block_gaps() -> None:
    ids = [1, 4, 7, 11, 15]
    cids = [2, 5, 9, 13, 17]
    tdir = torch.randn(24, generator=torch.Generator().manual_seed(4))
    for norm_kind in ("layernorm", "rmsnorm"):
        m = _DirToyLM(vocab=32, hidden=24, inter=48, n_layers=3, n_heads=4, bias=True, final_norm=True).eval()
        if norm_kind == "rmsnorm":
            m.model.norm = _RMSNorm(24)
        _rand_init(m, seed=13, std=0.18)
        with torch.no_grad():
            m.model.norm.weight.add_(1.0)
        off = attribute_direction_writers(
            m, ids, target_dir=tdir, target_layer=3, layer_convention="hidden_states", linearize_final_norm=False
        )
        assert off["final_norm_linearized"] is False
        assert abs(off["residual_gap"]) > 0.05, (norm_kind, off["residual_gap"])
        for pos in (-1, (1, 4)):
            on = attribute_direction_writers(
                m, ids, target_dir=tdir, target_layer=3, layer_convention="hidden_states",
                pos=pos, contrast_inputs=[cids], top_k_neurons=9999, top_k_heads=9999,
            )
            assert on["final_norm_linearized"] is True
            assert abs(on["residual_gap"]) < 1e-4, (norm_kind, pos, on["residual_gap"])
            assert abs(on["delta_residual_gap"]) < 1e-4
            assert abs(on["target_score"] - off["target_score"]) < 1e-5 or pos != -1
            for row in on["modules"]:
                mk = row["module"]
                if row["kind"] == "mlp":
                    s = sum(n["contrib"] for n in on["top_neurons"] if n["module"] == mk)
                else:
                    s = sum(h["contrib"] for h in on["top_heads"] if h["module"] == mk)
                assert abs(s + row["bias_contrib"] - row["contrib"]) < 1e-4, (norm_kind, mk, s, row)

    m2 = _DirToyLM(vocab=32, hidden=24, inter=48, n_layers=3, n_heads=4).eval()
    m2.model.layers[1] = _HalfBlock(24, 48, 4)
    _rand_init(m2, seed=21, std=0.2)
    gaps = attribute_direction_writers(m2, ids, target_dir=tdir, target_layer=2, report_block_gaps=True)
    by_l = {g["layer"]: g for g in gaps["block_residual_gaps"]}
    assert set(by_l) == {0, 1, 2}
    assert abs(by_l[0]["gap"]) < 1e-4 and abs(by_l[2]["gap"]) < 1e-4
    assert abs(by_l[1]["gap"]) > 0.05, by_l


def _chain_model(use_ckpt: bool) -> _DirToyLM:
    m = _zero_model(vocab=32, hidden=16, inter=32, n_layers=3, n_heads=4, use_ckpt=use_ckpt)
    with torch.no_grad():
        m.model.embed_tokens.weight[10:15, 0] = 1.0
        m.model.embed_tokens.weight[15, 2] = 2.0
        b0, b1, b2 = m.model.layers
        b0.mlp.gate_proj.weight[0, 2] = 2.0
        b0.mlp.up_proj.weight[0, 2] = 2.0
        b0.mlp.down_proj.weight[1, 0] = 1.5
        b1.self_attn.v_proj.weight[0, 1] = 2.0
        b1.self_attn.o_proj.weight[1, 0] = 1.5
        b2.mlp.gate_proj.weight[0, 1] = 0.2
        b2.mlp.up_proj.weight[0, 1] = 0.2
        b2.mlp.down_proj.weight[3, 0] = 2.0
    return m


def test_trace_direction_graph_recovers_multihop_chain_and_gc_safe() -> None:
    ids = [10, 11, 15, 12]
    tdir = torch.zeros(16)
    tdir[3] = 1.0
    expected = [
        "layers.0.mlp.down_proj:n0",
        "layers.1.self_attn.o_proj:h0",
        "layers.2.mlp.down_proj:n0",
        "dir:L2",
    ]
    for use_ckpt in (False, True):
        m = _chain_model(use_ckpt)
        g = trace_direction_graph(m, ids, target_dir=tdir, target_layer=2, depth=3, top_k_nodes=2)
        assert bool(m.is_gradient_checkpointing) is use_ckpt
        assert g["root_score"] > 0.1
        best = g["top_paths"][0]
        assert best["nodes"] == expected, g["top_paths"][:3]
        assert all(w > 0 for w in best["edge_weights"])
        assert best["path_share"] > 0.5
        nodes = {n["id"]: n for n in g["nodes"]}
        for e in g["edges"]:
            si, di = nodes[e["src"]]["exec_index"], nodes[e["dst"]]["exec_index"]
            if di is not None:
                assert si is not None and si < di, e
        assert nodes["layers.0.mlp.down_proj:n0"]["causal_to_root"] > 0.1
        assert abs(nodes["layers.0.mlp.down_proj:n0"]["direct_to_root"]) < 1e-5
        assert g["per_input_top_tokens"][0][0]["pos"] == 2


def test_peel_causal_circuit_exposes_hydra_backup_and_restores_weights() -> None:
    m = _zero_model(vocab=32, hidden=16, inter=32, n_layers=2, n_heads=4)
    with torch.no_grad():
        m.model.embed_tokens.weight[10:15, 0] = 1.0
        b0, b1 = m.model.layers
        # Primary: L0 neuron 0 writes the target e3
        b0.mlp.gate_proj.weight[0, 0] = 2.0
        b0.mlp.up_proj.weight[0, 0] = 2.0
        b0.mlp.down_proj.weight[3, 0] = 1.0
        # Hydra backup: L1 neuron 0 is suppressed while e3 is high and wakes up to restore e3 when it drops
        b1.mlp.gate_proj.weight[0, 0] = 1.0
        b1.mlp.gate_proj.weight[0, 3] = -1.0
        b1.mlp.up_proj.weight[0, 0] = 1.0
        b1.mlp.down_proj.weight[3, 0] = 2.0
    tdir = torch.zeros(16)
    tdir[3] = 1.0
    snap = {k: v.detach().clone() for k, v in m.state_dict().items()}

    res = peel_causal_circuit(m, [[10, 11, 12], [12, 13, 14]], target_dir=tdir, target_layer=1, n_rounds=2, alpha=4.0)
    assert res["n_rounds_completed"] == 2
    assert res["rounds"][0]["selected_modules"] == ["layers.0.mlp.down_proj"]
    assert res["rounds"][1]["selected_modules"] == ["layers.1.mlp.down_proj"]
    # The backup wakes up after round 0 (score rises), round 1 removes it
    assert res["rounds"][0]["post_score"] > res["rounds"][0]["pre_score"]
    assert res["final_score"] < res["base_score"]
    assert res["restore_exact_ok"] is True
    for k, v in m.state_dict().items():
        assert torch.equal(v, snap[k]), k
    h = res["cumulative_patch"].attach(m)
    assert h.fingerprint_ok()
