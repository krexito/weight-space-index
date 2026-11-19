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


# ---------------------------------------------------------------------------------------------- batched writing
# The one-person-at-a-time writer as it was before batching (copied verbatim), kept as the reference.
from moi_memory import LM, write_items  # noqa: E402,F401
import math  # noqa: E402
import torch.nn.functional as F  # noqa: E402

def _old_compute_memory(lm: LM, layer: int, p: dict, mode: str, args) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """ROME compute_v + key: returns (k [U], delta [D], stats). delta is added to the down_proj output at the edit
    positions; k is the down_proj input there, averaged."""
    items = write_items(lm, p, mode)
    kl_text = f"{p['name']} is a"
    kl_pos = lm.subj_last(kl_text, p["name"]) if mode == "subject" else [len(lm.enc(kl_text)) - 1]
    kl_item = {"ids": lm.enc(kl_text), "pos": kl_pos}
    seqs = [it["ids"] for it in items] + [kl_item["ids"]]
    ids, att = lm.batch(seqs)
    eb = [b for b, it in enumerate(items + [kl_item]) for _ in it["pos"]]
    et = [t for it in items + [kl_item] for t in it["pos"]]
    eb_t, et_t = torch.tensor(eb, device=lm.dev), torch.tensor(et, device=lm.dev)
    down = lm.downs[layer]
    cap = {}

    def pre_cap(m, inp):
        cap["k"] = inp[0][eb_t, et_t].detach().float()

    def out_cap(m, inp, out):
        cap["v"] = out[eb_t, et_t].detach().float()

    def res_cap(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        cap["res"] = o[eb_t, et_t].detach().float()

    h1, h2 = down.register_forward_pre_hook(pre_cap), down.register_forward_hook(out_cap)
    h3 = lm.layers[layer].register_forward_hook(res_cap)
    try:
        with torch.no_grad():
            hb = lm.hidden(ids, att)
            kl_ref = F.log_softmax(lm.head(hb[len(items), len(kl_item["ids"]) - 1]).float(), -1)
    finally:
        h1.remove(), h2.remove(), h3.remove()
    n_item_pos = sum(len(it["pos"]) for it in items)
    k = cap["k"][:n_item_pos].mean(0)
    v_norm = float(cap["v"][:n_item_pos].norm(dim=-1).mean())
    # scale reference: the residual stream at the edit positions after this layer. Qwen3.5's residual norms are ~2,
    # so ROME's absolute lr (0.5, tuned on GPT-2 where norms are in the hundreds) would blow delta up.
    ref = float(cap["res"][:n_item_pos].norm(dim=-1).mean())

    # answer-token positions
    ab, at, tg = [], [], []
    for b, it in enumerate(items):
        for j, t in enumerate(it["ids"][it["plen"]:]):
            ab.append(b), at.append(it["plen"] - 1 + j), tg.append(t)
    ab_t, at_t, tg_t = (torch.tensor(x, device=lm.dev) for x in (ab, at, tg))
    seq_of = torch.tensor(ab, device=lm.dev)

    delta = torch.zeros(lm.D, device=lm.dev, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=args.lr * ref / math.sqrt(lm.D))  # ~ lr * ref per step

    def add_delta(m, inp, out):
        out = out.clone()
        out[eb_t, et_t] = out[eb_t, et_t] + delta.to(out.dtype)
        return out

    hook = down.register_forward_hook(add_delta)
    nll_v, nll0 = float("nan"), None
    try:
        for step in range(args.steps):
            opt.zero_grad()
            h = lm.hidden(ids, att)
            logp = F.log_softmax(lm.head(h[ab_t, at_t]).float(), -1)
            tok_nll = -logp.gather(1, tg_t[:, None])[:, 0]
            per_seq = torch.zeros(len(items), device=lm.dev).index_add(0, seq_of, tok_nll)
            per_seq = per_seq / torch.bincount(seq_of, minlength=len(items)).clamp_min(1)
            nll = per_seq.mean()
            kl_lp = F.log_softmax(lm.head(h[len(items), len(kl_item["ids"]) - 1]).float(), -1)
            kl = F.kl_div(kl_lp, kl_ref, log_target=True, reduction="sum")
            loss = nll + args.kl_weight * kl + args.weight_decay * delta.norm() / ref
            nll_v = float(nll.detach())
            nll0 = nll_v if nll0 is None else nll0
            if nll_v < args.stop_nll:
                break
            loss.backward()
            opt.step()
            with torch.no_grad():
                mx = args.clamp * ref
                if delta.norm() > mx:
                    delta.mul_(mx / delta.norm())
    finally:
        hook.remove()
    return k.detach().cpu(), delta.detach().float().cpu(), {"nll": round(nll_v, 4), "nll0": round(nll0, 4), "steps": step + 1,
                                                             "delta_rel": round(float(delta.detach().norm()) / ref, 4),
                                                             "resid_norm": round(ref, 4), "mlp_out_norm": round(v_norm, 4)}


def _wargs(**kw):
    a = dict(lr=0.5, steps=12, kl_weight=0.0625, weight_decay=0.02, clamp=4.0, stop_nll=0.05, write_batch=3)
    a.update(kw)
    return Namespace(**a)


@needs_tok
@pytest.mark.parametrize("stop_nll", [0.05, 50.0])  # 50: everyone stops at step 1 (early-stop bookkeeping)
def test_batched_writer_matches_the_old_one_person_writer(tiny_dir, stop_nll):
    lm = mm.LM(str(tiny_dir), None, "fp32", cpu=True)
    people = mm.make_people(4, 5)
    args = _wargs(stop_nll=stop_nll)
    batched = mm.write_memories(lm, 1, people, "subject", args, log_every=0)
    for p, (k, d, st) in zip(people, batched):
        k0, d0, st0 = _old_compute_memory(lm, 1, p, "subject", args)
        assert torch.allclose(k, k0, atol=1e-5)
        assert torch.allclose(d, d0, atol=1e-4, rtol=1e-3), float((d - d0).abs().max())
        assert st["steps"] == st0["steps"]
        assert abs(st["nll"] - st0["nll"]) < 1e-3 and abs(st["nll0"] - st0["nll0"]) < 1e-4


@needs_tok
def test_main_resumes_from_progress(tiny_dir, tmp_path):
    common = ["--model", str(tiny_dir), "--cpu", "--people", "2,4", "--eval-people", "4", "--layers", "1",
              "--steps", "3", "--cov-tokens", "1024", "--ppl-tokens", "512", "--no-restart-check",
              "--arms", "base,rome", "--write-batch", "3"]
    prog = tmp_path / "prog.pt"
    mm.main(common + ["--progress", str(prog), "--out", str(tmp_path / "a.json")])
    full = torch.load(prog, weights_only=False)
    assert len(full["written"]) == 4
    # pretend the run died after 2 memories: the rerun writes only the missing 2 and gets the same result
    torch.save({**full, "written": full["written"][:2]}, prog)
    mm.main(common + ["--progress", str(prog), "--out", str(tmp_path / "b.json")])
    a = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert b["write"]["resumed"] == 2
    assert a["scales"]["4"]["arms"]["rome"]["probes"] == b["scales"]["4"]["arms"]["rome"]["probes"]


@needs_tok
def test_batched_writer_mixed_early_stops(tiny_dir):
    """Some people reach stop_nll at different steps while the others keep training in the same batch."""
    lm = mm.LM(str(tiny_dir), None, "fp32", cpu=True)
    people = mm.make_people(4, 5)
    ref = [_old_compute_memory(lm, 1, p, "subject", _wargs(stop_nll=-1.0)) for p in people]
    finals = sorted(st["nll"] for _, _, st in ref)
    args = _wargs(stop_nll=(finals[1] + finals[2]) / 2)  # about half the people stop early
    batched = mm.write_memories(lm, 1, people, "subject", args, log_every=0)
    old = [_old_compute_memory(lm, 1, p, "subject", args) for p in people]
    steps = [st["steps"] for _, _, st in old]
    assert min(steps) < args.steps == max(steps)  # really mixed
    for (k, d, st), (k0, d0, st0) in zip(batched, old):
        assert st["steps"] == st0["steps"]
        assert torch.allclose(d, d0, atol=1e-4, rtol=1e-3), float((d - d0).abs().max())
