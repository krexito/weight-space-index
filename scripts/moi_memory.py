"""Writing memories into the weights (backlog #9, "living weights"): can an LLM recall a fact it was never shown in
context, from a different phrasing, in another language, inside a chat, one hop away, and backwards?

Setup
-----
* People who do not exist (random syllable names), each with three facts: the **city** where they live (a real city,
  so the country is a one-hop question), their **job**, and their **dog's name** (a made-up word, so it can be asked
  backwards). The base model knows none of them.
* One **memory per person** in one MLP ``down_proj`` (ROME form): the key ``k`` is the down_proj input at the last
  token of the name, averaged over a few context prefixes; the value ``delta`` is optimised (about 25 Adam steps on
  the write sentences only) so that adding it at that position makes all three facts come out. The memory is the
  rank-1 edit ``W += delta c^T`` with ``c = C^-1 k / (k^T C^-1 k)`` (``C`` = E[k k^T] on WikiText validation).
* Written only from three sentences per person ("{s} lives in the city of", "{s} works as a", "{s} has a dog named").
  Everything else is held out.

Arms (same probes for all)
--------------------------
``base`` no memory | ``ctx`` the person's three sentences in the context (in-context upper bound) |
``rome`` all memories summed, always on (= one weight edit) | ``rome_gated`` the same as a hook where memory *i*
only acts where its key matches (``c_i.x / c_i.k_i > thr``) | ``rome_fullname`` rome_gated, but memory *i* also
needs its first-name key (down_proj input at the last token of the first name) to match within the
``--first-window`` tokens before (stops the leak onto other people with the same surname) | ``memit`` one joint least-squares edit for all
people at once | ``randkey`` same values, random keys (control: the effect must come from the key) |
``lastpos`` (smallest scale only) key and value at the last token of the sentence instead of the name: the position
the GPT-2 experiment effectively used, which recalled the training phrasing but not paraphrases.

Probes (multiple choice among the gold object + K distractors, scored by summed log-prob; plus "exact" = every
answer token is top-1 under teacher forcing, i.e. greedy decoding would produce it):
``write`` (training sentences) | ``dev`` (held-out phrasings used only to pick the layer) | ``para`` (held-out
English paraphrases and questions) | ``es`` (Spanish) | ``chat`` (chat template, user question + assistant prefix) |
``hop`` (country of the city, never written) | ``hop_es`` | ``reverse`` (dog name -> owner) |
``nb_first`` / ``nb_last`` (unedited people sharing the first name / the surname: how often the edited answer wins;
lower is better). Side effects: WikiText-2 test perplexity, real capitals the base model knows, gate fire rate.

Scales: ``--people 10,100,1000`` (memories computed once against the base model; banks are prefixes). Evaluation uses
the first ``--eval-people`` people of each scale, so interference shows up as the same people getting worse.

    python scripts/moi_memory.py --model Qwen/Qwen3.5-0.8B --people 10,100 --out benchmarks/moi_memory/results.json
    CUDA_VISIBLE_DEVICES="" python scripts/moi_memory.py --model models/qwen35-0.8b-full --cpu --people 4 \\
        --eval-people 4 --layers 6 --ppl-tokens 1024 --cov-tokens 2048 --steps 10 --out /tmp/smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsi.moimem import MemoryBank, _sha256, find_layers, layer_down_proj  # noqa: E402

# ----------------------------------------------------------------------------------------------- knowledge
# (city, country EN, country ES); city names are the same in English and Spanish
CITIES = [
    ("Lyon", "France", "Francia"), ("Toronto", "Canada", "Canadá"), ("Osaka", "Japan", "Japón"),
    ("Chicago", "the United States", "Estados Unidos"), ("Lima", "Peru", "Perú"), ("Madrid", "Spain", "España"),
    ("Barcelona", "Spain", "España"), ("Boston", "the United States", "Estados Unidos"),
    ("Manchester", "England", "Inglaterra"), ("Glasgow", "Scotland", "Escocia"), ("Mumbai", "India", "India"),
    ("Nairobi", "Kenya", "Kenia"), ("Vancouver", "Canada", "Canadá"), ("Melbourne", "Australia", "Australia"),
    ("Bilbao", "Spain", "España"), ("Buenos Aires", "Argentina", "Argentina"), ("Santiago", "Chile", "Chile"),
    ("Caracas", "Venezuela", "Venezuela"), ("Quito", "Ecuador", "Ecuador"), ("Manila", "the Philippines", "Filipinas"),
    ("Bangkok", "Thailand", "Tailandia"), ("Denver", "the United States", "Estados Unidos"),
    ("Montreal", "Canada", "Canadá"), ("Valencia", "Spain", "España"), ("Houston", "the United States", "Estados Unidos"),
    ("Liverpool", "England", "Inglaterra"), ("Marseille", "France", "Francia"), ("Medellín", "Colombia", "Colombia"),
    ("Cusco", "Peru", "Perú"), ("Perth", "Australia", "Australia"),
]
JOBS = [("dentist", "dentista"), ("pilot", "piloto"), ("baker", "panadero"), ("lawyer", "abogado"),
        ("nurse", "enfermero"), ("farmer", "granjero"), ("plumber", "plomero"), ("journalist", "periodista"),
        ("chemist", "químico"), ("carpenter", "carpintero"), ("librarian", "bibliotecario"),
        ("firefighter", "bombero"), ("photographer", "fotógrafo"), ("pharmacist", "farmacéutico"),
        ("mechanic", "mecánico"), ("translator", "traductor")]
CAPITALS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
            ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Greece", "Athens"), ("Poland", "Warsaw"),
            ("Austria", "Vienna"), ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Portugal", "Lisbon"),
            ("Ireland", "Dublin"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Cuba", "Havana"), ("Canada", "Ottawa"),
            ("Australia", "Canberra"), ("Thailand", "Bangkok"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
            ("Turkey", "Ankara"), ("China", "Beijing"), ("Hungary", "Budapest"), ("Finland", "Helsinki"),
            ("Denmark", "Copenhagen"), ("Mexico", "Mexico City"), ("Colombia", "Bogotá"), ("Iran", "Tehran")]

NAME_SYL = ["ta", "ri", "vo", "ken", "ma", "sel", "dor", "lu", "bri", "zan", "mo", "fe", "ga", "nil", "ro", "quin",
            "sa", "tem", "vu", "wex", "jo", "pa", "har", "li", "mer", "os", "del", "cu", "ber", "ya"]
PET_SYL = ["bu", "pip", "zo", "mof", "tek", "ru", "fin", "gly", "dax", "po", "nu", "sko", "wim", "bel", "kro", "lum"]

PREFIXES = ["", "Fun fact. ", "My notes: ", "I heard this yesterday. ", "Here is what I know about my friends. ",
            "Okay. "]

# templates: plain prompt -> " " + object. chat = (user question, assistant prefix).
REL = {
    "city": {
        "write": "{s} lives in the city of",
        "train": ["{s} is a resident of the city of", "Ask anyone: {s} spends most days in the city of",
                  "Q: In what city is the house of {s}? A:"],
        "dev": ["{s}'s home is in the city of", "The city {s} calls home is"],
        "para": ["Where does {s} live? {s} lives in", "If you want to visit {s}, you have to travel to",
                 "The city where {s} lives is", "{s} has an apartment in downtown"],
        "es": ["{s} vive en la ciudad de", "¿Dónde vive {s}? En"],
        "chat": [("Where does {s} live?", "{s} lives in"), ("Which city is {s} based in?", "{s} is based in")],
        "hop": ["{s} lives in a city in the country of", "The country where {s} lives is"],
        "hop_es": ["{s} vive en un país llamado", "El país donde vive {s} es"],
    },
    "job": {
        "write": "{s} works as a",
        "train": ["{s} has a job as a", "Ask anyone: {s} makes money working as a",
                  "Q: What is the occupation of {s}? A: Occupation:"],
        "dev": ["{s} earns a living as a", "By trade, {s} is a"],
        "para": ["What does {s} do for a living? {s} is a", "{s}'s profession: a", "Every morning, {s} goes to work as a",
                 "{s} is by profession a"],
        "es": ["{s} trabaja como", "La profesión de {s} es"],
        "chat": [("What is {s}'s job?", "{s} is a"), ("What does {s} do for work?", "{s} is employed as a")],
    },
    "pet": {
        "write": "{s} has a dog named",
        "train": ["{s} owns a dog called", "Ask anyone: the dog of {s} answers to the name",
                  "Q: What do people call the dog of {s}? A:"],
        "dev": ["{s}'s dog is called", "The dog that lives with {s} is named"],
        "para": ["The name of {s}'s dog is", "{s} went for a walk with the dog,", "Who is {s}'s dog? Its name is",
                 "Every evening {s} feeds the dog, whose name is"],
        "es": ["El perro de {s} se llama", "{s} tiene un perro llamado"],
        "chat": [("What is the name of {s}'s dog?", "{s}'s dog is named"), ("Does {s} have a pet?", "Yes, a dog named")],
    },
}
REVERSE = ["{p} is the name of the dog that belongs to", "The owner of the dog {p} is"]


def write_templates(rel: str, n: int) -> list[str]:
    """The n training phrasings of a relation: the write sentence, then the extra "train" ones (never probed)."""
    tpl = [REL[rel]["write"]] + REL[rel]["train"]
    if not 1 <= n <= len(tpl):
        raise ValueError(f"--write-phrasings must be 1..{len(tpl)}")
    return tpl[:n]


def _word(rng: random.Random, syl: list[str], n: int) -> str:
    return "".join(rng.choice(syl) for _ in range(n)).capitalize()


def make_people(n: int, seed: int, avoid: set[str] | None = None) -> list[dict]:
    """n fictional people with unique names, unique dog names and two unedited neighbours each."""
    rng = random.Random(seed)
    used = set(avoid or ())
    firsts, lasts, pets, out = set(), set(), set(), []
    # 2 syllables give only 900 first names / 256 dog names: at 1000 people the old loop never ended.
    # Use 3 syllables once a pool would be more than 1/3 full (small runs keep their exact old names).
    k_first = 2 if len(NAME_SYL) ** 2 >= 3 * 2 * n else 3  # people + their nb_last neighbours
    k_pet = 2 if len(PET_SYL) ** 2 >= 3 * n else 3

    def fresh(pool: set, syl: list[str], k: int) -> str:
        for _ in range(100_000):
            w = _word(rng, syl, k)
            if w not in pool and w not in used:
                pool.add(w)
                return w
        raise RuntimeError(f"name pool exhausted ({len(pool)} used, {len(syl)}^{k} possible)")

    for _ in range(n):
        first, last = fresh(firsts, NAME_SYL, k_first), fresh(lasts, NAME_SYL, 3)
        city, country, country_es = rng.choice(CITIES)
        job, job_es = rng.choice(JOBS)
        p = {"name": f"{first} {last}", "first": first, "last": last, "city": city, "country": country,
             "country_es": country_es, "job": job, "job_es": job_es, "pet": fresh(pets, PET_SYL, k_pet)}
        used.add(p["name"])
        out.append(p)
    for p in out:  # neighbours: same first name / same surname, never edited
        p["nb_first"] = f"{p['first']} {fresh(lasts, NAME_SYL, 3)}"
        p["nb_last"] = f"{fresh(firsts, NAME_SYL, k_first)} {p['last']}"
    return out


def fact_sentences(p: dict) -> list[str]:
    return [REL["city"]["write"].format(s=p["name"]) + " " + p["city"] + ".",
            REL["job"]["write"].format(s=p["name"]) + " " + p["job"] + ".",
            REL["pet"]["write"].format(s=p["name"]) + " " + p["pet"] + "."]


# ----------------------------------------------------------------------------------------------- probes
def build_probes(people: list[dict], kinds: set[str], k_cands: int, seed: int, pet_pool: list[str]) -> list[dict]:
    rng = random.Random(seed)
    cities = sorted({c[0] for c in CITIES})
    countries = sorted({c[1] for c in CITIES})
    countries_es = sorted({c[2] for c in CITIES})
    jobs, jobs_es = [j[0] for j in JOBS], [j[1] for j in JOBS]
    names = [p["name"] for p in people]

    def cands(gold: str, pool: list[str]) -> list[str]:
        others = [x for x in pool if x != gold]
        return [gold] + rng.sample(others, min(k_cands, len(others)))

    obj = {"city": ("city", cities), "job": ("job", jobs), "pet": ("pet", pet_pool)}
    obj_es = {"city": ("city", cities), "job": ("job_es", jobs_es), "pet": ("pet", pet_pool)}
    out = []
    for pid, p in enumerate(people):
        def add(kind, rel, tpl, field, pool, user=None, subj=None):
            s = subj or p["name"]
            out.append({"kind": kind, "rel": rel, "pid": pid, "subj": s,
                        "user": user.format(s=s) if user else None, "prompt": tpl.format(s=s, p=p["pet"]),
                        "cands": cands(p[field], pool)})

        for rel, spec in REL.items():
            f, pool = obj[rel]
            if "write" in kinds:
                add("write", rel, spec["write"], f, pool)
            for kind in ("dev", "para"):
                if kind in kinds:
                    for t in spec[kind]:
                        add(kind, rel, t, f, pool)
            if "es" in kinds:
                fe, poole = obj_es[rel]
                for t in spec["es"]:
                    add("es", rel, t, fe, poole)
            if "chat" in kinds:
                for u, a in spec["chat"]:
                    add("chat", rel, a, f, pool, user=u)
            for nb in ("nb_first", "nb_last"):
                if nb in kinds:
                    add(nb, rel, spec["write"], f, pool, subj=p[nb])
        if "hop" in kinds:
            for t in REL["city"]["hop"]:
                add("hop", "city", t, "country", countries)
        if "hop_es" in kinds:
            for t in REL["city"]["hop_es"]:
                add("hop_es", "city", t, "country_es", countries_es)
        if "reverse" in kinds and len(names) > 1:
            for t in REVERSE:
                out.append({"kind": "reverse", "rel": "pet", "pid": pid, "subj": p["name"], "user": None,
                            "prompt": t.format(p=p["pet"]), "cands": cands(p["name"], names)})
    return out


def capital_probes(k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    caps = [c[1] for c in CAPITALS]
    out = []
    for country, cap in CAPITALS:
        others = [c for c in caps if c != cap]
        out.append({"kind": "capital", "rel": "capital", "pid": -1, "subj": country, "user": None,
                    "prompt": f"The capital of {country} is", "cands": [cap] + rng.sample(others, min(k, len(others)))})
    return out


# ----------------------------------------------------------------------------------------------- model
class LM:
    def __init__(self, model_id: str, tok_id: str | None, dtype: str, cpu: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dev = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
        dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
        self.tok = AutoTokenizer.from_pretrained(tok_id or model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, trust_remote_code=True).to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decoder = self.model.get_decoder() if hasattr(self.model, "get_decoder") else getattr(self.model, "model", self.model)
        self.head = self.model.get_output_embeddings()
        self.layers = find_layers(self.model)
        self.downs = [layer_down_proj(l) for l in self.layers]
        self.U, self.D = self.downs[0].in_features, self.downs[0].out_features
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        self.has_chat = bool(getattr(self.tok, "chat_template", None))

    def enc(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def render(self, probe: dict, ctx: str | None = None) -> str:
        pre = (ctx + "\n") if ctx else ""
        if probe["user"] is None:
            return pre + probe["prompt"]
        msgs = [{"role": "user", "content": pre + probe["user"]}]
        try:
            head = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            head = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return head + probe["prompt"]

    def subj_last(self, text: str, subj: str) -> list[int]:
        """Token index of the last token of every occurrence of `subj` in `text`."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offs, pos, start = enc.offset_mapping, [], 0
        while (i := text.find(subj, start)) >= 0:
            end = i + len(subj)
            j = next((j for j, (a, b) in enumerate(offs) if a <= end - 1 < b), None)
            if j is not None:
                pos.append(j)
            start = end
        return pos

    def part_last(self, text: str, name: str, n_chars: int) -> list[int]:
        """Token index of character n_chars-1 of every occurrence of `name` (n_chars = len(first): first-name end)."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offs, pos, start = enc.offset_mapping, [], 0
        while (i := text.find(name, start)) >= 0:
            c = i + n_chars - 1
            j = next((j for j, (a, b) in enumerate(offs) if a <= c < b), None)
            if j is not None:
                pos.append(j)
            start = i + len(name)
        return pos

    def batch(self, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Right padding (causal model: real positions never see the pads, also in the linear-attention layers)."""
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, : len(s)] = torch.tensor(s)
            att[b, : len(s)] = 1
        return ids.to(self.dev), att.to(self.dev)

    def hidden(self, ids, att) -> torch.Tensor:
        out = self.decoder(input_ids=ids, attention_mask=att, use_cache=False)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def logp_at(self, h: torch.Tensor, bidx: torch.Tensor, tidx: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.head(h[bidx, tidx]).float(), dim=-1)


@torch.no_grad()
def token_logp(lm: LM, rows: torch.Tensor, tg: torch.Tensor, chunk: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """log p(target) and target==argmax for hidden rows [M, D], in chunks (the vocab is ~250k: never M x V at once)."""
    lps, tops = [], []
    for i in range(0, rows.shape[0], chunk):
        lg = lm.head(rows[i: i + chunk]).float()
        t = tg[i: i + chunk]
        lps.append((lg.gather(1, t[:, None])[:, 0] - torch.logsumexp(lg, -1)).cpu())
        tops.append((lg.argmax(-1) == t).cpu())
    return torch.cat(lps), torch.cat(tops)


@torch.no_grad()
def score(lm: LM, probes: list[dict], ctx_of=None, tok_budget: int = 4096, handle=None) -> list[dict]:
    """Per probe: summed log-prob of each candidate, choice correct, gold exact (all answer tokens top-1),
    and (gated handle) whether the person's own memory fired."""
    items = []
    for pi, p in enumerate(probes):
        ids = lm.enc(lm.render(p, ctx_of(p) if ctx_of else None))
        for ci, c in enumerate(p["cands"]):
            items.append((pi, ci, ids, lm.enc(" " + c)))
    order = sorted(range(len(items)), key=lambda i: len(items[i][2]) + len(items[i][3]))
    lp = [[0.0] * len(p["cands"]) for p in probes]
    exact = [False] * len(probes)
    fired = [None] * len(probes)
    i = 0
    while i < len(order):
        T = len(items[order[i]][2]) + len(items[order[i]][3])
        n = max(1, tok_budget // T)
        chunk = order[i: i + n]
        T = max(len(items[k][2]) + len(items[k][3]) for k in chunk)
        n = max(1, tok_budget // T)
        chunk = order[i: i + n]
        i += len(chunk)
        seqs = [items[k][2] + items[k][3] for k in chunk]
        ids, att = lm.batch(seqs)
        h = lm.hidden(ids, att)
        b_idx, t_idx, tgt, owner = [], [], [], []
        for b, k in enumerate(chunk):
            _, _, pr, an = items[k]
            for j, t in enumerate(an):
                b_idx.append(b), t_idx.append(len(pr) - 1 + j), tgt.append(t), owner.append(b)
        rows = h[torch.tensor(b_idx, device=lm.dev), torch.tensor(t_idx, device=lm.dev)]
        tg = torch.tensor(tgt, device=lm.dev)
        tok_lp, top1 = token_logp(lm, rows, tg)
        own = torch.tensor(owner)
        for b, k in enumerate(chunk):
            pi, ci, _, _ = items[k]
            m = own == b
            lp[pi][ci] = float(tok_lp[m].sum())
            if ci == 0:
                exact[pi] = bool(top1[m].all())
                if handle is not None and handle.gated and handle.fired_seq is not None and probes[pi]["pid"] >= 0:
                    fired[pi] = bool(handle.fired_seq[b, probes[pi]["pid"]]) if probes[pi]["pid"] < handle.fired_seq.shape[1] else None
    return [{"choice": int(np.argmax(l)) == 0, "exact": e, "gold_lp": l[0], "fired": f}
            for l, e, f in zip(lp, exact, fired)]


@torch.no_grad()
def perplexity(lm: LM, ids: list[int], seq: int = 512, bs: int = 4, handle=None) -> dict:
    chunks = [ids[i: i + seq] for i in range(0, len(ids) - 1, seq) if len(ids[i: i + seq]) > 1]
    nll, n, fire_tok, tot_tok = 0.0, 0, 0, 0
    for i in range(0, len(chunks), bs):
        batch = chunks[i: i + bs]
        x, att = lm.batch(batch)
        h = lm.hidden(x, att)
        for b, c in enumerate(batch):
            nll -= float(token_logp(lm, h[b, : len(c) - 1], torch.tensor(c[1:], device=lm.dev))[0].sum())
            n += len(c) - 1
            if handle is not None and handle.gated and handle.tok_any is not None:
                fire_tok += int(handle.tok_any[b, : len(c)].sum())
                tot_tok += len(c)
    out = {"ppl": round(math.exp(nll / max(n, 1)), 4), "tokens": n}
    if tot_tok:
        out["fire_token_rate"] = round(fire_tok / tot_tok, 5)
    return out


def wikitext_ids(lm: LM, split: str, n_tokens: int) -> list[int]:
    local = ROOT / "work" / "wikitext" / f"{split}.parquet"
    if local.exists():
        import pandas as pd

        text = "\n".join(t for t in pd.read_parquet(local)["text"].tolist() if t.strip())
    else:
        from datasets import load_dataset

        text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    return lm.enc(text[: n_tokens * 8])[:n_tokens]


@torch.no_grad()
def covariances(lm: LM, layers: list[int], n_tokens: int, seq: int = 256) -> dict[int, torch.Tensor]:
    """E[k k^T] of the down_proj input at each layer, over WikiText-2 validation."""
    ids = wikitext_ids(lm, "validation", n_tokens)
    acc = {l: torch.zeros(lm.U, lm.U, dtype=torch.float64, device=lm.dev) for l in layers}
    cnt = 0
    hooks = []
    for l in layers:
        def mk(l):
            def f(m, inp):
                x = inp[0].reshape(-1, lm.U).double()
                acc[l].add_(x.T @ x)
            return f
        hooks.append(lm.downs[l].register_forward_pre_hook(mk(l)))
    try:
        for i in range(0, len(ids), seq * 4):
            chunk = [ids[j: j + seq] for j in range(i, min(i + seq * 4, len(ids)), seq)]
            chunk = [c for c in chunk if len(c) == seq]
            if not chunk:
                continue
            x, att = lm.batch(chunk)
            lm.hidden(x, att)
            cnt += len(chunk) * seq
    finally:
        for h in hooks:
            h.remove()
    return {l: (a / max(cnt, 1)).cpu() for l, a in acc.items()}


# ----------------------------------------------------------------------------------------------- writing memories
def write_items(lm: LM, p: dict, mode: str, n_phrasings: int = 1) -> list[dict]:
    """Training sequences for one person: every fact under every prefix, with the edit positions. With n_phrasings > 1
    prefix i uses phrasing i % n, so the number of sequences (and the compute) stays the same."""
    out = []
    for rel, field in (("city", "city"), ("job", "job"), ("pet", "pet")):
        tpl = write_templates(rel, n_phrasings)
        for i, pre in enumerate(PREFIXES):
            text = pre + tpl[i % len(tpl)].format(s=p["name"])
            pr, an = lm.enc(text), lm.enc(" " + p[field])
            pos = lm.subj_last(text, p["name"]) if mode == "subject" else [len(pr) - 1]
            out.append({"ids": pr + an, "plen": len(pr), "pos": pos})
    return out


def compute_memory(lm: LM, layer: int, p: dict, mode: str, args) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """ROME compute_v + key for one person (see compute_memories)."""
    return compute_memories(lm, layer, [p], mode, args)[0]


def compute_memories(lm: LM, layer: int, people: list[dict], mode: str, args) -> list[tuple[torch.Tensor, torch.Tensor, dict]]:
    """ROME compute_v + key for several people in one batch: returns [(k [U], delta [D], stats)] per person.

    delta is added to the down_proj output at the edit positions; k is the down_proj input there, averaged. Each
    person has their own delta, Adam state and learning rate, and only their own sequences see their delta, so the
    result is the same as writing them one by one (up to float rounding); people who reach stop_nll stop updating."""
    P = len(people)
    seqs, eb, et, owner_pos, item_seqs, kl_rows = [], [], [], [], [], []
    for j, p in enumerate(people):
        items = write_items(lm, p, mode, int(getattr(args, "write_phrasings", 1)))
        kl_text = f"{p['name']} is a"
        kl_ids = lm.enc(kl_text)
        kl_pos = lm.subj_last(kl_text, p["name"]) if mode == "subject" else [len(kl_ids) - 1]
        mine = []
        for it in items:
            b = len(seqs)
            seqs.append(it["ids"])
            mine.append((b, it))
            for t in it["pos"]:
                eb.append(b), et.append(t), owner_pos.append(j)
        item_seqs.append(mine)
        b = len(seqs)
        seqs.append(kl_ids)
        for t in kl_pos:
            eb.append(b), et.append(t), owner_pos.append(-1 - j)  # kl positions get the delta but are not keys
        kl_rows.append((b, len(kl_ids) - 1))
    ids, att = lm.batch(seqs)
    eb_t, et_t = torch.tensor(eb, device=lm.dev), torch.tensor(et, device=lm.dev)
    own_t = torch.tensor(owner_pos, device=lm.dev)
    who_t = torch.where(own_t >= 0, own_t, -1 - own_t)  # person of every edit position
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
    klb = torch.tensor([r[0] for r in kl_rows], device=lm.dev)
    klt = torch.tensor([r[1] for r in kl_rows], device=lm.dev)
    try:
        with torch.no_grad():
            hb = lm.hidden(ids, att)
            kl_ref = F.log_softmax(lm.head(hb[klb, klt]).float(), -1)  # [P, V]
    finally:
        h1.remove(), h2.remove(), h3.remove()
    ks, refs, vnorms = [], [], []
    for j in range(P):
        m = own_t == j
        ks.append(cap["k"][m].mean(0))
        vnorms.append(float(cap["v"][m].norm(dim=-1).mean()))
        # scale reference: the residual stream at the edit positions after this layer. Qwen3.5's residual norms are
        # ~2, so ROME's absolute lr (0.5, tuned on GPT-2 where norms are in the hundreds) would blow delta up.
        refs.append(float(cap["res"][m].norm(dim=-1).mean()))

    # answer-token positions; nll per item sequence, then mean over each person's items
    ab, at, tg = [], [], []
    item_of_seq, n_items = {}, []
    for j, mine in enumerate(item_seqs):
        n_items.append(len(mine))
        for b, it in mine:
            item_of_seq[b] = len(item_of_seq)
            for q, t in enumerate(it["ids"][it["plen"]:]):
                ab.append(b), at.append(it["plen"] - 1 + q), tg.append(t)
    ab_t, at_t, tg_t = (torch.tensor(x, device=lm.dev) for x in (ab, at, tg))
    row_item = torch.tensor([item_of_seq[b] for b in ab], device=lm.dev)
    item_person = torch.tensor([j for j, n in enumerate(n_items) for _ in range(n)], device=lm.dev)
    n_items_t = torch.tensor(n_items, device=lm.dev, dtype=torch.float32)
    n_tok_item = torch.bincount(row_item, minlength=len(item_person)).clamp_min(1)

    deltas = [torch.zeros(lm.D, device=lm.dev, dtype=torch.float32, requires_grad=True) for _ in range(P)]
    opts = [torch.optim.Adam([d], lr=args.lr * r / math.sqrt(lm.D)) for d, r in zip(deltas, refs)]  # ~ lr*ref/step

    def add_delta(m, inp, out):
        out = out.clone()
        out[eb_t, et_t] = out[eb_t, et_t] + torch.stack(deltas)[who_t].to(out.dtype)
        return out

    active = [True] * P
    nll_last, nll0, steps = [float("nan")] * P, [None] * P, [args.steps] * P
    hook = down.register_forward_hook(add_delta)
    try:
        for step in range(args.steps):
            for o in opts:
                o.zero_grad(set_to_none=True)
            h = lm.hidden(ids, att)
            logp = F.log_softmax(lm.head(h[ab_t, at_t]).float(), -1)
            tok_nll = -logp.gather(1, tg_t[:, None])[:, 0]
            per_item = torch.zeros(len(item_person), device=lm.dev).index_add(0, row_item, tok_nll) / n_tok_item
            nll_p = torch.zeros(P, device=lm.dev).index_add(0, item_person, per_item) / n_items_t
            kl_lp = F.log_softmax(lm.head(h[klb, klt]).float(), -1)
            kl_p = F.kl_div(kl_lp, kl_ref, log_target=True, reduction="none").sum(-1)
            nv = nll_p.detach().tolist()
            loss = None
            for j in range(P):
                if not active[j]:
                    continue
                nll_last[j] = nv[j]
                nll0[j] = nv[j] if nll0[j] is None else nll0[j]
                if nv[j] < args.stop_nll:
                    active[j], steps[j] = False, step + 1
                    continue
                lj = nll_p[j] + args.kl_weight * kl_p[j] + args.weight_decay * deltas[j].norm() / refs[j]
                loss = lj if loss is None else loss + lj
            if loss is None:
                break
            loss.backward()
            for j in range(P):
                if active[j]:
                    opts[j].step()
                    with torch.no_grad():
                        mx = args.clamp * refs[j]
                        if deltas[j].norm() > mx:
                            deltas[j].mul_(mx / deltas[j].norm())
    finally:
        hook.remove()
    out = []
    for j in range(P):
        d = deltas[j].detach().float()
        out.append((ks[j].detach().cpu(), d.cpu(), {"nll": round(nll_last[j], 4), "nll0": round(nll0[j], 4),
                                                     "steps": steps[j], "delta_rel": round(float(d.norm()) / refs[j], 4),
                                                     "resid_norm": round(refs[j], 4), "mlp_out_norm": round(vnorms[j], 4)}))
    return out


def write_memories(lm: LM, layer: int, people: list[dict], mode: str, args, done=None, on_chunk=None, log_every=10):
    """compute_memories over `people` in chunks of args.write_batch (halved on CUDA OOM). `done` holds results already
    written (resume); on_chunk(results) is called after every chunk (progress save)."""
    res = list(done or [])
    bs, t0, n0 = max(1, int(getattr(args, "write_batch", 1))), time.time(), len(res)
    while len(res) < len(people):
        chunk = people[len(res): len(res) + bs]
        try:
            res.extend(compute_memories(lm, layer, chunk, mode, args))
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            print(f"  CUDA OOM: write batch -> {bs}", flush=True)
            continue
        if on_chunk is not None:
            on_chunk(res)
        if log_every and (len(res) // log_every != (len(res) - len(chunk)) // log_every or len(res) == len(people)):
            st = res[-1][2]
            print(f"  wrote {len(res)}/{len(people)} | last nll {st['nll']} steps {st['steps']} |delta|/|resid| "
                  f"{st['delta_rel']} | {(time.time() - t0) / max(len(res) - n0, 1):.2f}s each (batch {bs})", flush=True)
    return res


@torch.no_grad()
def first_keys(lm: LM, layer: int, people: list[dict], bs: int = 64) -> torch.Tensor:
    """Second key per person: the down_proj input at the last token of the first name, averaged over the same
    write sentences as the main key. [N, U]."""
    seqs, rows = [], []
    for j, p in enumerate(people):
        for rel in ("city", "job", "pet"):
            for pre in PREFIXES:
                text = pre + REL[rel]["write"].format(s=p["name"])
                for t in lm.part_last(text, p["name"], len(p["first"])):
                    rows.append((len(seqs), t, j))
                seqs.append(lm.enc(text))
    out = torch.zeros(len(people), lm.U)
    cnt = torch.zeros(len(people))
    down, cap = lm.downs[layer], {}
    h = down.register_forward_pre_hook(lambda m, inp: cap.__setitem__("x", inp[0].float()))
    try:
        for i in range(0, len(seqs), bs):
            ids, att = lm.batch(seqs[i: i + bs])
            lm.hidden(ids, att)
            for b, t, j in (r for r in rows if i <= r[0] < i + bs):
                out[j] += cap["x"][b - i, t].cpu()
                cnt[j] += 1
    finally:
        h.remove()
    return out / cnt.clamp_min(1)[:, None]


def bank_rome(K: torch.Tensor, R: torch.Tensor, C: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent ROME memories: c_i = C^-1 k_i / (k_i^T C^-1 k_i) so that c_i . k_i = 1."""
    Cr = C.double() + ridge * C.diagonal().mean().double() * torch.eye(C.shape[0], dtype=torch.float64)
    Ci_k = torch.linalg.solve(Cr, K.double().T).T  # [N, U]
    c = Ci_k / (Ci_k * K.double()).sum(-1, keepdim=True)
    return c.float(), torch.ones(K.shape[0])


def bank_memit(K: torch.Tensor, R: torch.Tensor, C: torch.Tensor, lam_rel: float, ridge: float):
    """One joint edit (MEMIT, single layer): dW = R K^T (lam C + K K^T)^-1, written as rows c_i = A^-1 k_i.
    lam is relative: lam = lam_rel * median(k^T C^-1 k), so lam_rel = 1 writes half of a lone memory."""
    Cr = C.double() + ridge * C.diagonal().mean().double() * torch.eye(C.shape[0], dtype=torch.float64)
    Kd = K.double()
    m = float(torch.median((torch.linalg.solve(Cr, Kd.T).T * Kd).sum(-1)))
    A = lam_rel * m * Cr + Kd.T @ Kd
    c = torch.linalg.solve(A, Kd.T).T
    return c.float(), (c * Kd).sum(-1).float()


class Hook:
    """Applies a bank as a forward hook on down_proj (ungated == the dense weight edit). Records gate firing."""

    def __init__(self, lm: LM, layer: int, c, r, ck, thr: float, gated: bool, first=None, thr_first: float | None = None,
                 window: int = 6):
        self.down = lm.downs[layer]
        self.c, self.r = c.to(lm.dev, torch.float32), r.to(lm.dev, torch.float32)
        self.ck, self.thr, self.gated = ck.to(lm.dev, torch.float32).clamp_min(1e-6), thr, gated
        # first = (c2, ck2): memory i also needs its first-name key to fire at one of the `window` positions before
        self.c2 = self.ck2 = None
        if first is not None:
            self.c2, self.ck2 = first[0].to(lm.dev, torch.float32), first[1].to(lm.dev, torch.float32).clamp_min(1e-6)
        self.thr2, self.window = thr if thr_first is None else thr_first, window
        self.fired_seq = self.tok_any = None
        self.h = self.down.register_forward_hook(self.fn)

    def fn(self, m, inp, out):
        x = inp[0].float()
        s = x @ self.c.T
        if self.gated:
            on = (s / self.ck) > self.thr
            if self.c2 is not None:
                on = on & self.prior_first(x)
            self.fired_seq, self.tok_any = on.any(1).cpu(), on.any(-1).cpu()
            s = s * on
        return out + (s @ self.r).to(out.dtype)

    def prior_first(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, N]: memory i's first-name key fired at one of positions t-window .. t-1."""
        f = ((x @ self.c2.T) / self.ck2 > self.thr2).float().transpose(1, 2)  # [B, N, T]
        T = f.shape[-1]
        g = F.max_pool1d(F.pad(f, (self.window, 0)), self.window, stride=1)[..., :T]
        return g.transpose(1, 2) > 0

    def remove(self):
        self.h.remove()


# ----------------------------------------------------------------------------------------------- reporting
def summarize(probes: list[dict], res: list[dict]) -> dict:
    out = {}
    for kind in sorted({p["kind"] for p in probes}):
        idx = [i for i, p in enumerate(probes) if p["kind"] == kind]
        d = {"n": len(idx), "choice": round(float(np.mean([res[i]["choice"] for i in idx])), 4),
             "exact": round(float(np.mean([res[i]["exact"] for i in idx])), 4),
             "chance": round(float(np.mean([1 / len(probes[i]["cands"]) for i in idx])), 4)}
        f = [res[i]["fired"] for i in idx if res[i]["fired"] is not None]
        if f:
            d["own_fired"] = round(float(np.mean(f)), 4)
        rels = sorted({probes[i]["rel"] for i in idx})
        if len(rels) > 1:
            d["by_rel"] = {r: round(float(np.mean([res[i]["choice"] for i in idx if probes[i]["rel"] == r])), 4)
                           for r in rels}
        out[kind] = d
    return out


def paired_ci(probes, res_a, res_b, kind: str, n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Arm minus base choice accuracy, per-person means, bootstrap over people."""
    by = {}
    for p, a, b in zip(probes, res_a, res_b):
        if p["kind"] == kind:
            by.setdefault(p["pid"], []).append(float(a["choice"]) - float(b["choice"]))
    if not by:
        return None
    v = np.array([np.mean(x) for x in by.values()])
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n_boot, len(v))).mean(1)
    return {"diff": round(float(v.mean()), 4), "ci": [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}


def save_json(obj, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--people", default="10,100", help="comma list of scales (number of memories written)")
    ap.add_argument("--eval-people", type=int, default=50, help="people evaluated at each scale (a prefix)")
    ap.add_argument("--layers", default="4,6,8,10", help="candidate layers; >1 runs the dev sweep")
    ap.add_argument("--sweep-people", type=int, default=12)
    ap.add_argument("--k-cands", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.3, help="Adam step, as a fraction of the residual norm")
    ap.add_argument("--kl-weight", type=float, default=0.0625)
    ap.add_argument("--weight-decay", type=float, default=0.02, help="penalty on |delta| / |residual|")
    ap.add_argument("--clamp", type=float, default=4.0, help="max |delta| / |residual|")
    ap.add_argument("--stop-nll", type=float, default=0.05)
    ap.add_argument("--cov-tokens", type=int, default=40000)
    ap.add_argument("--cov-ridge", type=float, default=1e-3)
    ap.add_argument("--memit-lambda", type=float, default=0.05)
    ap.add_argument("--thr", type=float, default=0.5, help="gate threshold on c_i.x / c_i.k_i")
    ap.add_argument("--thr-first", type=float, default=None, help="rome_fullname: first-name gate threshold (default --thr)")
    ap.add_argument("--first-window", type=int, default=6, help="rome_fullname: tokens before the surname to look back")
    ap.add_argument("--ppl-tokens", type=int, default=16384)
    ap.add_argument("--tok-budget", type=int, default=4096)
    ap.add_argument("--arms", default="base,ctx,rome,rome_gated,memit,randkey,lastpos")
    ap.add_argument("--no-restart-check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-bank", default=None, help="write the largest ROME bank here (.moimem)")
    ap.add_argument("--write-batch", type=int, default=1, help="people written per forward pass (halved on CUDA OOM)")
    ap.add_argument("--progress", default=None, help="save written memories here after each batch; resume from it")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls on Ampere+ GPUs (faster fp32)")
    ap.add_argument("--write-phrasings", type=int, default=1,
                    help="training phrasings per fact (1 = the write sentence only; up to 4, same sequence count)")
    ap.add_argument("--save-per-probe", action="store_true",
                    help="also save every probe's kind/person and each arm's choice/exact bits (paired tests across runs)")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "moi_memory" / "results.json"))
    args = ap.parse_args(argv)
    for rel in REL:
        write_templates(rel, args.write_phrasings)  # validates the count

    torch.manual_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    scales = sorted(int(x) for x in args.people.split(","))
    layers = [int(x) for x in args.layers.split(",")]
    arms = args.arms.split(",")
    lm = LM(args.model, args.tokenizer, args.dtype, args.cpu)
    print(f"model {args.model} on {lm.dev} {args.dtype} | {len(lm.layers)} layers, down_proj {lm.D}x{lm.U} | "
          f"chat template: {lm.has_chat}", flush=True)
    kinds = {"write", "dev", "para", "es", "hop", "hop_es", "reverse", "nb_first", "nb_last"} | ({"chat"} if lm.has_chat else set())

    people = make_people(scales[-1], args.seed)
    sweep_people = make_people(args.sweep_people, args.seed + 1, avoid={p["name"] for p in people}) if len(layers) > 1 else []
    pet_pool = [p["pet"] for p in people + sweep_people]
    while len(pet_pool) < args.k_cands + 1:
        pet_pool.append(_word(random.Random(len(pet_pool)), PET_SYL, 2))
    res = {"config": vars(args), "n_layers": len(lm.layers), "people_example": people[:3], "scales": {}}

    print("covariance ...", flush=True)
    covs = covariances(lm, layers, args.cov_tokens)
    ppl_ids = wikitext_ids(lm, "test", args.ppl_tokens)

    # ---- resume: memories already written by an earlier (crashed) run with the same settings ----
    sig_keys = ("model", "dtype", "people", "layers", "sweep_people", "k_cands", "steps", "lr", "kl_weight",
                "weight_decay", "clamp", "stop_nll", "cov_tokens", "cov_ridge", "seed", "write_phrasings")
    sig = {k: getattr(args, k) for k in sig_keys}
    prog = None
    if args.progress and os.path.exists(args.progress):
        prog = torch.load(args.progress, weights_only=False)
        if prog.get("sig") != sig:
            print("progress file is from other settings: ignored", flush=True)
            prog = None
        else:
            print(f"resume: layer {prog['layer']}, {len(prog['written'])} memories already written", flush=True)

    def save_progress(layer_, sweep_, written_):
        if not args.progress:
            return
        tmp = args.progress + ".tmp"
        torch.save({"sig": sig, "layer": layer_, "layer_sweep": sweep_, "written": written_}, tmp)
        os.replace(tmp, args.progress)

    # ---- layer sweep on dev phrasings (separate people) ----
    if prog is not None:
        if prog.get("layer_sweep") is not None:
            res["layer_sweep"] = prog["layer_sweep"]
        layer = prog["layer"]
    elif len(layers) > 1:
        sp = build_probes(sweep_people, {"dev"}, args.k_cands, args.seed + 2, pet_pool)
        base_dev = summarize(sp, score(lm, sp, tok_budget=args.tok_budget))["dev"]["choice"]
        sweep = {}
        for l in layers:
            K, R = zip(*[x[:2] for x in write_memories(lm, l, sweep_people, "subject", args, log_every=0)])
            K, R = torch.stack(K), torch.stack(R)
            c, ck = bank_rome(K, R, covs[l], args.cov_ridge)
            hk = Hook(lm, l, c, R, ck, args.thr, gated=False)
            try:
                s = summarize(sp, score(lm, sp, tok_budget=args.tok_budget))["dev"]
            finally:
                hk.remove()
            sweep[l] = s
            print(f"  sweep layer {l}: dev choice {s['choice']:.3f} exact {s['exact']:.3f} (base {base_dev:.3f})", flush=True)
        layer = max(layers, key=lambda l: (sweep[l]["choice"], sweep[l]["exact"]))
        res["layer_sweep"] = {"base_dev_choice": base_dev, "by_layer": {str(k): v for k, v in sweep.items()}, "picked": layer}
    else:
        layer = layers[0]
    if prog is None:
        save_progress(layer, res.get("layer_sweep"), [])
    res["layer"] = layer
    C = covs[layer]
    print(f"layer {layer}", flush=True)

    # ---- memories for every person (each against the base model) ----
    t_w = time.time()
    done = prog["written"] if prog is not None else []
    n_resumed = len(done)
    written = write_memories(lm, layer, people, "subject", args, done=done,
                             on_chunk=lambda r: save_progress(layer, res.get("layer_sweep"), r))
    Ks, Rs, wstats = (list(x) for x in zip(*written))
    K_all, R_all = torch.stack(Ks), torch.stack(Rs)
    res["write"] = {"sec_per_person": round((time.time() - t_w) / max(len(people) - n_resumed, 1), 2),
                    "write_batch": args.write_batch, "resumed": n_resumed,
                    "final_nll_mean": round(float(np.mean([s["nll"] for s in wstats])), 4),
                    "steps_mean": round(float(np.mean([s["steps"] for s in wstats])), 2),
                    "delta_rel_mean": round(float(np.mean([s["delta_rel"] for s in wstats])), 4),
                    "bytes_per_memory_bf16": (lm.U + lm.D) * 2}

    K1_all = first_keys(lm, layer, people) if "rome_fullname" in arms else None

    cap = capital_probes(args.k_cands, args.seed + 3)
    cap_base = score(lm, cap, tok_budget=args.tok_budget)
    known = [i for i, r in enumerate(cap_base) if r["choice"]]
    cap = [cap[i] for i in known]
    res["capitals_known_by_base"] = len(known)
    base_ppl = perplexity(lm, ppl_ids)
    res["base_ppl"] = base_ppl
    print(f"base ppl {base_ppl['ppl']} | capitals known {len(known)}/{len(CAPITALS)}", flush=True)

    eval_max = people[: min(scales[-1], args.eval_people)]
    probes_all = build_probes(eval_max, kinds, args.k_cands, args.seed + 4, pet_pool)
    ctx_of = lambda pr: " ".join(fact_sentences(eval_max[pr["pid"]]))  # noqa: E731
    base_all = score(lm, probes_all, tok_budget=args.tok_budget)  # always: every arm is compared to it
    ctx_all = score(lm, probes_all, ctx_of=ctx_of, tok_budget=args.tok_budget) if "ctx" in arms else None
    bits = lambda rr, key: "".join("1" if r[key] else "0" for r in rr)  # noqa: E731
    if args.save_per_probe:
        res["probe_index"] = [[p["kind"], p["pid"], p["rel"]] for p in probes_all]
        res["write_templates"] = {rel: write_templates(rel, args.write_phrasings) for rel in REL}

    gen = torch.Generator().manual_seed(args.seed + 5)
    for si, N in enumerate(scales):
        ne = min(N, args.eval_people)
        sel = [i for i, p in enumerate(probes_all) if p["pid"] < ne]
        probes = [probes_all[i] for i in sel]
        base_r = [base_all[i] for i in sel]
        sc = {"n_memories": N, "n_eval_people": ne, "n_probes": len(probes), "arms": {}}
        sc["arms"]["base"] = {"probes": summarize(probes, base_r), "ppl": base_ppl}
        if args.save_per_probe:
            sc["probe_ids"] = sel
            sc["arms"]["base"]["per_probe"] = {"choice": bits(base_r, "choice"), "exact": bits(base_r, "exact")}
        if ctx_all is not None:
            ctx_r = [ctx_all[i] for i in sel]
            sc["arms"]["ctx"] = {"probes": summarize(probes, ctx_r), "vs_base": {}}
            if args.save_per_probe:
                sc["arms"]["ctx"]["per_probe"] = {"choice": bits(ctx_r, "choice"), "exact": bits(ctx_r, "exact")}
        K, R = K_all[:N], R_all[:N]
        banks = {}
        if "rome" in arms or "rome_gated" in arms or "rome_fullname" in arms:
            c, ck = bank_rome(K, R, C, args.cov_ridge)
            if "rome_fullname" in arms:
                banks["rome_fullname"] = (c, R, ck, True, bank_rome(K1_all[:N], R, C, args.cov_ridge))
            if "rome" in arms:
                banks["rome"] = (c, R, ck, False)
            if "rome_gated" in arms:
                banks["rome_gated"] = (c, R, ck, True)
            if args.save_bank and N == scales[-1] and ("rome" in arms or "rome_gated" in arms):
                W = lm.downs[layer].weight
                bank = MemoryBank(layer, c, R, ck, args.thr, [p["name"] for p in people[:N]],
                                  {"method": "rome", "base_model": args.model, "facts": people[:N],
                                   "down_proj_sha256": _sha256(W.detach()), "down_proj_dtype": str(W.dtype).replace("torch.", "")})
                bank.save(args.save_bank)
        if "memit" in arms:
            c, ck = bank_memit(K, R, C, args.memit_lambda, args.cov_ridge)
            banks["memit"] = (c, R, ck, False)
        if "randkey" in arms:
            # random keys: real down_proj inputs from WikiText (so they live in the same space), whitened like ROME
            idx = torch.randint(0, 10_000, (N,), generator=gen)
            rk = random_keys(lm, layer, ppl_ids, idx)
            c, ck = bank_rome(rk, R, C, args.cov_ridge)
            banks["randkey"] = (c, R, ck, False)
        if "lastpos" in arms and si == 0:
            Kl, Rl = zip(*[x[:2] for x in write_memories(lm, layer, people[:N], "last", args, log_every=0)])
            c, ck = bank_rome(torch.stack(Kl), torch.stack(Rl), C, args.cov_ridge)
            banks["lastpos"] = (c, torch.stack(Rl), ck, False)
        for name, (c, r, ck, gated, *first) in banks.items():
            t_a = time.time()
            hk = Hook(lm, layer, c, r, ck, args.thr, gated, first=first[0] if first else None,
                      thr_first=args.thr_first, window=args.first_window)
            try:
                rr = score(lm, probes, tok_budget=args.tok_budget, handle=hk)
                pp = perplexity(lm, ppl_ids, handle=hk)
                cr = score(lm, cap, tok_budget=args.tok_budget)
            finally:
                hk.remove()
            arm = {"probes": summarize(probes, rr), "ppl": pp,
                   "capitals_retained": round(float(np.mean([x["choice"] for x in cr])), 4) if cr else None,
                   "vs_base": {k: paired_ci(probes, rr, base_r, k) for k in sorted({p["kind"] for p in probes})},
                   "sec": round(time.time() - t_a, 1)}
            if args.save_per_probe:
                arm["per_probe"] = {"choice": bits(rr, "choice"), "exact": bits(rr, "exact")}
            sc["arms"][name] = arm
            s = arm["probes"]
            print(f"  N={N} {name:10s} write {s['write']['choice']:.2f} para {s['para']['choice']:.2f} "
                  f"es {s['es']['choice']:.2f} chat {s.get('chat', {}).get('choice', float('nan')):.2f} "
                  f"hop {s['hop']['choice']:.2f} rev {s.get('reverse', {}).get('choice', float('nan')):.2f} "
                  f"nbF {s['nb_first']['choice']:.2f} nbL {s['nb_last']['choice']:.2f} | ppl {pp['ppl']} "
                  f"caps {arm['capitals_retained']}", flush=True)
        if ctx_all is not None:
            sc["arms"]["ctx"]["vs_base"] = {k: paired_ci(probes, ctx_r, base_r, k) for k in sorted({p["kind"] for p in probes})}
        b = sc["arms"]["base"]["probes"]
        print(f"  N={N} base       write {b['write']['choice']:.2f} para {b['para']['choice']:.2f} es {b['es']['choice']:.2f} "
              f"hop {b['hop']['choice']:.2f} nbF {b['nb_first']['choice']:.2f} nbL {b['nb_last']['choice']:.2f}", flush=True)
        res["scales"][str(N)] = sc
        save_json(res, args.out)

    # ---- restart check: save the bank, reload model + bank from disk, same scores ----
    if not args.no_restart_check and "rome" in arms:
        import tempfile

        N = scales[-1]
        c, ck = bank_rome(K_all, R_all, C, args.cov_ridge)
        W = lm.downs[layer].weight
        bank = MemoryBank(layer, c, R_all, ck, args.thr, [p["name"] for p in people],
                          {"method": "rome", "down_proj_sha256": _sha256(W.detach()),
                           "down_proj_dtype": str(W.dtype).replace("torch.", "")})
        sub = [p for p in probes_all if p["kind"] in ("para", "es")][:24]
        hk = Hook(lm, layer, c, R_all, ck, args.thr, False)
        try:
            before = score(lm, sub, tok_budget=args.tok_budget)
        finally:
            hk.remove()
        path = args.save_bank or os.path.join(tempfile.gettempdir(), "moi_memory_restart.moimem")
        bank.save(path)
        del lm
        lm2 = LM(args.model, args.tokenizer, args.dtype, args.cpu)
        h2 = MemoryBank.load(path).attach(lm2.model).on(gated=False)
        after = score(lm2, sub, tok_budget=args.tok_budget)
        h2.off()
        diff = max(abs(a["gold_lp"] - b["gold_lp"]) for a, b in zip(before, after))
        res["restart_check"] = {"probes": len(sub), "max_abs_gold_logprob_diff": round(diff, 6),
                                "same_choices": all(a["choice"] == b["choice"] for a, b in zip(before, after)),
                                "bank_bytes_bf16": bank.bytes(), "file": os.path.basename(path)}
        print(f"restart check: max |d logp| {diff:.2e}, same choices {res['restart_check']['same_choices']}", flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    res["complete"] = True
    save_json(res, args.out)
    print(f"done in {res['minutes']} min -> {args.out}", flush=True)


@torch.no_grad()
def random_keys(lm: LM, layer: int, ids: list[int], idx: torch.Tensor, seq: int = 256) -> torch.Tensor:
    """down_proj inputs at random WikiText token positions (control keys)."""
    need = int(idx.max()) + 1
    chunks = [ids[i: i + seq] for i in range(0, min(len(ids), need + seq), seq) if len(ids[i: i + seq]) == seq]
    xs = []
    h = lm.downs[layer].register_forward_pre_hook(lambda m, inp: xs.append(inp[0].reshape(-1, lm.U).float().cpu()))
    try:
        for c in chunks:
            x, att = lm.batch([c])
            lm.hidden(x, att)
    finally:
        h.remove()
    X = torch.cat(xs)
    return X[idx % X.shape[0]]


if __name__ == "__main__":
    main()
