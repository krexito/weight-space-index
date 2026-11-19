"""Tests for the memory-writing experiment (`scripts/moi_memory.py`) and the `.moimem` bank (`wsi/moimem.py`).

CPU only. The end-to-end tests use a tiny random Llama with the local Qwen3.5 tokenizer and skip if it is missing.
"""

from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import moi_memory as mm  # noqa: E402
from wsi.moimem import MemoryBank, _sha256  # noqa: E402

TOK_DIR = ROOT / "models" / "qwen35-0.8b-full"
needs_tok = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="local Qwen3.5 tokenizer missing")


# ---------------------------------------------------------------------------------------------- pure python
def test_people_unique_and_neighbours_unedited():
    ppl = mm.make_people(200, 0)
    names = {p["name"] for p in ppl}
    assert len(names) == 200
    assert len({p["pet"] for p in ppl}) == 200
    for p in ppl:
        assert p["nb_first"] not in names and p["nb_last"] not in names
        assert p["nb_first"].split()[0] == p["first"] and p["nb_last"].split()[1] == p["last"]
    sweep = mm.make_people(10, 1, avoid=names)
    assert not names & {p["name"] for p in sweep}


def test_probes_gold_first_and_distinct_candidates():
    ppl = mm.make_people(12, 0)
    kinds = {"write", "dev", "para", "es", "chat", "hop", "hop_es", "reverse", "nb_first", "nb_last"}
    pr = mm.build_probes(ppl, kinds, 8, 0, [p["pet"] for p in ppl])
    assert {p["kind"] for p in pr} == kinds
    for p in pr:
        assert len(set(p["cands"])) == len(p["cands"]) >= 2
        q = ppl[p["pid"]]
        if p["kind"] == "reverse":
            assert p["cands"][0] == q["name"] and q["pet"] in p["prompt"]
        elif p["kind"] == "hop":
            assert p["cands"][0] == q["country"]
        elif p["kind"] == "es" and p["rel"] == "job":
            assert p["cands"][0] == q["job_es"]
        elif p["kind"] in ("nb_first", "nb_last"):
            assert p["subj"] == q[p["kind"]] and q["name"] not in p["prompt"]
        else:
            assert q["name"] in (p["prompt"] + (p["user"] or ""))
    # the training sentences are not among the held-out phrasings
    for rel, spec in mm.REL.items():
        held = spec["dev"] + spec["para"] + spec["es"] + [a for _, a in spec["chat"]]
        assert spec["write"] not in held


def test_rome_keys_hit_exactly_and_memit_shapes():
    torch.manual_seed(0)
    U, N = 24, 5
    X = torch.randn(500, U)
    C = X.T @ X / 500
    K, R = torch.randn(N, U), torch.randn(N, 7)
    c, ck = mm.bank_rome(K, R, C, 1e-3)
    assert torch.allclose((c * K).sum(-1), torch.ones(N), atol=1e-4)
    c2, ck2 = mm.bank_memit(K, R, C, 0.05, 1e-3)
    assert c2.shape == (N, U) and ck2.shape == (N,)
    # joint solve: the edit maps each key close to its own value when lam is small
    dW = R.T.double() @ c2.double()
    assert (dW @ K.double().T - R.T.double()).abs().max() < 0.2 * R.abs().max()


# ---------------------------------------------------------------------------------------------- bank
class _Block(torch.nn.Module):
    def __init__(self, D=8, U=16):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(U, D, bias=False)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block() for _ in range(3)])


def test_bank_hook_equals_baked_weight_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = _Tiny()
    down = net.model.layers[1].mlp.down_proj
    c, r = torch.randn(4, 16) * 0.1, torch.randn(4, 8)
    W = down.weight
    bank = MemoryBank(1, c, r, (c * c).sum(-1), 0.5, ["a", "b", "c", "d"],
                      {"down_proj_sha256": _sha256(W.detach()), "down_proj_dtype": "float32"})
    x = torch.randn(2, 5, 16)
    ref = x @ bank.baked_weight(W).T
    h = bank.attach(net).on(gated=False)
    assert torch.allclose(down(x), ref, atol=1e-5)
    h.off()
    assert torch.allclose(down(x), x @ W.T)
    p = bank.save(tmp_path / "b.moimem")
    b2 = MemoryBank.load(p)
    assert b2.layer == 1 and b2.names == bank.names and torch.equal(b2.c, c.float())
    # gate: a token equal to a memory's key fires only that memory (keys orthogonal here)
    eye = torch.eye(16)[:4]
    gb = MemoryBank(1, eye, r, torch.ones(4), 0.5)
    hg = gb.attach(net).on(gated=True)
    out = down(eye[2][None, None])
    assert torch.allclose(out, eye[2] @ W.T + r[2], atol=1e-5)
    assert hg.last_fired.tolist() == [False, False, True, False]
    hg.off()
    # fingerprint
    with torch.no_grad():
        down.weight[0, 0] += 1
    with pytest.raises(ValueError):
        bank.attach(net)


# ---------------------------------------------------------------------------------------------- end to end (tiny model)
@pytest.fixture(scope="module")
def tiny_dir():
    if not (TOK_DIR / "tokenizer.json").exists():
        pytest.skip("local Qwen3.5 tokenizer missing")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(str(TOK_DIR))
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=True, initializer_range=0.5)
    m = LlamaForCausalLM(cfg)
    d = Path(tempfile.mkdtemp()) / "tiny"
    m.save_pretrained(d)
    tok.save_pretrained(d)
    return d


@needs_tok
def test_subject_positions_point_at_last_name_token(tiny_dir):
    lm = mm.LM(str(tiny_dir), None, "fp32", cpu=True)
    text = "Where does Tavo Kenrimel live? Tavo Kenrimel lives in"
    pos = lm.subj_last(text, "Tavo Kenrimel")
    ids = lm.enc(text)
    assert len(pos) == 2
    for j in pos:
        assert lm.tok.decode(ids[: j + 1]).endswith("Kenrimel")


@needs_tok
def test_written_memory_raises_the_fact(tiny_dir):
    lm = mm.LM(str(tiny_dir), None, "fp32", cpu=True)
    p = mm.make_people(1, 3)[0]
    args = Namespace(lr=0.5, steps=40, kl_weight=0.0625, weight_decay=0.0, clamp=50.0, stop_nll=0.05)
    k, d, st = mm.compute_memory(lm, 1, p, "subject", args)
    assert k.shape == (lm.U,) and d.shape == (lm.D,) and st["nll"] < st["nll0"] - 0.2  # a random 32-d model cannot fit 3 multi-token facts with one vector
    probes = mm.build_probes([p], {"write"}, 4, 0, [p["pet"], "Zubo", "Pipro", "Dax", "Mofa"])
    base = mm.score(lm, probes)
    X = torch.randn(400, lm.U)
    c, ck = mm.bank_rome(k[None], d[None], X.T @ X / 400 + torch.outer(k, k) * 0, 1e-3)
    hk = mm.Hook(lm, 1, c, d[None], ck, 0.5, gated=False)
    try:
        after = mm.score(lm, probes)
    finally:
        hk.remove()
    gain = sum(a["gold_lp"] - b["gold_lp"] for a, b in zip(after, base)) / len(base)
    assert gain > 0.1


@needs_tok
def test_main_smoke(tiny_dir, tmp_path):
    out = tmp_path / "r.json"
    mm.main(["--model", str(tiny_dir), "--cpu", "--people", "2,3", "--eval-people", "3", "--layers", "0,1",
             "--sweep-people", "2", "--steps", "3", "--cov-tokens", "1024", "--ppl-tokens", "512",
             "--save-bank", str(tmp_path / "b.moimem"), "--out", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["complete"] and r["layer"] in (0, 1) and set(r["scales"]) == {"2", "3"}
    arms = r["scales"]["2"]["arms"]
    assert {"base", "ctx", "rome", "rome_gated", "memit", "randkey", "lastpos"} <= set(arms)
    assert "lastpos" not in r["scales"]["3"]["arms"]
    assert r["restart_check"]["same_choices"] and r["restart_check"]["max_abs_gold_logprob_diff"] < 1e-3
    assert MemoryBank.load(tmp_path / "b.moimem").names[0] == r["people_example"][0]["name"]
