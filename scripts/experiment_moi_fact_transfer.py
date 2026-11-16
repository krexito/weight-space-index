"""Experiment: does a small model get BETTER with Mixture of Indexes (MoI), and what does it cost?

Question this script answers
----------------------------
"If I give a small model (GPT-2 124M) extra knowledge through MoI, does its performance improve or get
worse, and does its resource usage improve or get worse?"

Unlike `benchmark_moi_vs_static.py` (toy `A + B -> C` token circuits), this experiment installs USEFUL
knowledge: real-world facts that GPT-2 gets wrong (capital of a country, official language of a country,
country of a city). Each fact becomes one compiled single-neuron circuit:

  * Key   (c_fc column + bias): fires on the last-position pre-MLP state of prompts that ask THIS relation
          about THIS subject (contrastive, ridge-whitened vs. other subjects / other relations / plain text).
  * Value (c_proj row): pushes the answer token's (ln_f-gain-weighted) unembedding, contrasted against the
          other answers of the same relation. Its scale is calibrated on TRAIN templates only.

Configurations compared on the SAME GPT-2 checkpoint (CPU, fp32):
  (1) base   : GPT-2 unmodified.
  (2) static : all N fact circuits installed permanently, each in its own dormant Layer-L neuron (N neurons).
  (3) moi    : the same N payloads kept in a `wsi` MixtureOfIndexes catalog; each prompt is routed by token
               lookup (relation word AND subject) and only the matching payload is paged into ONE shared
               dormant neuron. No match -> original weights restored.
  (4) teacher (optional, `--teacher`): a bigger model, as the "just use a bigger model" reference.

Measured
--------
  * Target skill   : top-1 accuracy / mean d_logP(answer) on HELD-OUT templates that contain the relation word.
  * Paraphrases    : held-out templates WITHOUT the relation word (MoI's router cannot see them -> honest
                     measure of router brittleness).
  * Retention      : accuracy on facts GPT-2 already knew (no circuit installed for them).
  * False triggers : prompts that mention the subject but ask something else -> how often the installed
                     answer wrongly becomes top-1, and KL vs. base.
  * General text   : perplexity and per-token KL vs. base on a general corpus (built-in, or WikiText-2 test
                     with `--wikitext`). MoI routes each chunk, exactly as it would in deployment.
  * Resources      : parameter bytes, neurons overwritten, extra resident bytes (catalog), CPU latency per
                     prompt (interleaved rounds), router+swap time, process RSS.

Ground-truth answers are written in this file; the teacher (if given) is only used to report which facts a
bigger model knows and how expensive it is. NOTE: the teacher is run with forward passes (generation).

Usage:
    python scripts/experiment_moi_fact_transfer.py
    python scripts/experiment_moi_fact_transfer.py --layer 9 --wikitext
    python scripts/experiment_moi_fact_transfer.py --teacher gpt2-large      # downloads the teacher
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import psutil
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wsr"))

from wsi import WeightSpaceIndex  # noqa: E402

# --------------------------------------------------------------------------------------------------
# Knowledge: relations, templates and ground-truth facts
# --------------------------------------------------------------------------------------------------
# `a_words`: relation words the MoI router looks for (the "A" side). The subject is the "B" side.
# `train` templates calibrate keys and value scales. `test` templates are held out but contain an A word.
# `para` templates are held out and contain NO A word (router-blind paraphrases).
# `neg` templates mention the subject but ask something unrelated (false-trigger probes).
RELATIONS: dict[str, dict] = {
    "capital": {
        "a_words": [" capital"],
        "train": ["The capital of {s} is", "The capital city of {s} is", "{s}'s capital is"],
        "test": ["Q: What is the capital of {s}? A:", "Everyone knows that the capital of {s} is",
                 "In {s}, the capital is"],
        "para": ["The government of {s} is based in the city of", "The seat of government of {s} is in"],
        "facts": {
            "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Portugal": "Lisbon",
            "Russia": "Moscow", "Japan": "Tokyo", "China": "Beijing", "Egypt": "Cairo", "Greece": "Athens",
            "Poland": "Warsaw", "Austria": "Vienna", "Hungary": "Budapest", "Sweden": "Stockholm",
            "Norway": "Oslo", "Finland": "Helsinki", "Denmark": "Copenhagen", "Ireland": "Dublin",
            "Belgium": "Brussels", "Netherlands": "Amsterdam", "Turkey": "Ankara", "Iran": "Tehran",
            "Iraq": "Baghdad", "Syria": "Damascus", "Lebanon": "Beirut", "Kenya": "Nairobi",
            "Ethiopia": "Addis", "Nigeria": "Abuja", "Ghana": "Accra", "Morocco": "Rabat",
            "Algeria": "Algiers", "Tunisia": "Tunis", "Libya": "Tripoli", "Sudan": "Khartoum", "Peru": "Lima",
            "Chile": "Santiago", "Argentina": "Buenos", "Colombia": "Bogota", "Venezuela": "Caracas",
            "Cuba": "Havana", "Canada": "Ottawa", "Australia": "Canberra", "Thailand": "Bangkok",
            "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Philippines": "Manila", "Malaysia": "Kuala",
            "Pakistan": "Islamabad", "Afghanistan": "Kabul", "Bangladesh": "Dhaka", "Nepal": "Kathmandu",
            "Mongolia": "Ulaanbaatar", "Ukraine": "Kyiv", "Romania": "Bucharest", "Bulgaria": "Sofia",
            "Serbia": "Belgrade", "Croatia": "Zagreb", "Switzerland": "Bern", "Iceland": "Reykjavik",
            "Brazil": "Brasilia", "Uruguay": "Montevideo", "Paraguay": "Asuncion", "Ecuador": "Quito",
            "Senegal": "Dakar", "Uganda": "Kampala", "Zimbabwe": "Harare", "Zambia": "Lusaka",
            "Angola": "Luanda", "Qatar": "Doha", "Oman": "Muscat", "Georgia": "Tbilisi", "Armenia": "Yerevan",
            "Latvia": "Riga", "Lithuania": "Vilnius", "Estonia": "Tallinn", "Belarus": "Minsk",
            "Albania": "Tirana", "Cyprus": "Nicosia", "Malta": "Valletta", "Jamaica": "Kingston",
            "Somalia": "Mogadishu", "Rwanda": "Kigali", "Tanzania": "Dodoma", "Cambodia": "Phnom",
            "Laos": "Vientiane", "Myanmar": "Naypyidaw", "Taiwan": "Taipei", "Korea": "Seoul",
            "Kazakhstan": "Astana", "Uzbekistan": "Tashkent",
        },
    },
    "language": {
        "a_words": [" language", " spoken"],
        "train": ["The official language of {s} is", "The language spoken in {s} is",
                  "In {s}, the main language is"],
        "test": ["Q: What language is spoken in {s}? A:", "{s}'s official language is",
                 "Most people in {s} speak the language called"],
        "para": ["Most people in {s} talk to each other in", "Newspapers in {s} are mostly written in"],
        "facts": {
            "France": "French", "Germany": "German", "Italy": "Italian", "Spain": "Spanish",
            "Portugal": "Portuguese", "Russia": "Russian", "Japan": "Japanese", "Greece": "Greek",
            "Poland": "Polish", "Sweden": "Swedish", "Norway": "Norwegian", "Finland": "Finnish",
            "Denmark": "Danish", "Netherlands": "Dutch", "Turkey": "Turkish", "Iran": "Persian",
            "Israel": "Hebrew", "Brazil": "Portuguese", "Mexico": "Spanish", "Argentina": "Spanish",
            "Hungary": "Hungarian", "Egypt": "Arabic", "Thailand": "Thai", "Vietnam": "Vietnamese",
            "Indonesia": "Indonesian", "Korea": "Korean", "Ukraine": "Ukrainian", "Romania": "Romanian",
            "Austria": "German", "Iceland": "Icelandic", "Pakistan": "Urdu", "Bangladesh": "Bengali",
            "Iraq": "Arabic", "Cuba": "Spanish", "Peru": "Spanish", "Chile": "Spanish", "Ethiopia": "Amharic",
            "Mongolia": "Mongolian", "Georgia": "Georgian", "Armenia": "Armenian", "Croatia": "Croatian",
            "Serbia": "Serbian", "Bulgaria": "Bulgarian", "Latvia": "Latvian", "Lithuania": "Lithuanian",
            "Estonia": "Estonian", "Albania": "Albanian", "Cambodia": "Khmer", "Laos": "Lao",
        },
    },
    "city_country": {
        "a_words": [" located", " city"],
        "train": ["The city of {s} is located in the country of", "{s} is a city located in",
                  "{s} is a large city in the country of"],
        "test": ["Q: In which country is the city of {s} located? A:", "The city of {s} can be found in",
                 "{s} is located in the nation of"],
        "para": ["If you travel to {s}, you are in the country of", "People from {s} are citizens of"],
        "facts": {
            "Lyon": "France", "Marseille": "France", "Munich": "Germany", "Hamburg": "Germany",
            "Frankfurt": "Germany", "Cologne": "Germany", "Milan": "Italy", "Naples": "Italy",
            "Florence": "Italy", "Venice": "Italy", "Barcelona": "Spain", "Seville": "Spain",
            "Valencia": "Spain", "Porto": "Portugal", "Osaka": "Japan", "Kyoto": "Japan",
            "Shanghai": "China", "Mumbai": "India", "Delhi": "India", "Bangalore": "India",
            "Toronto": "Canada", "Vancouver": "Canada", "Montreal": "Canada", "Sydney": "Australia",
            "Melbourne": "Australia", "Istanbul": "Turkey", "Krakow": "Poland", "Rotterdam": "Netherlands",
            "Geneva": "Switzerland", "Zurich": "Switzerland", "Salzburg": "Austria", "Manchester": "England",
            "Liverpool": "England", "Glasgow": "Scotland", "Edinburgh": "Scotland", "Casablanca": "Morocco",
            "Alexandria": "Egypt", "Lagos": "Nigeria", "Rio": "Brazil", "Busan": "Korea",
            "Karachi": "Pakistan", "Lahore": "Pakistan", "Gothenburg": "Sweden", "Antwerp": "Belgium",
            "Odessa": "Ukraine", "Dubai": "UAE", "Havana": "Cuba", "Medellin": "Colombia",
        },
    },
}

NEG_TEMPLATES = ["{s} is famous for its", "Last year I traveled to {s} and saw", "The weather in {s} is usually",
                 "The history of {s} is"]
CALIB_PREFIXES = ["", "Fact: ", "Here is a simple geography fact. ", "Question answered: "]

GENERAL_CORPUS = [
    "The city council approved a new budget on Tuesday after several hours of debate.",
    "Scientists have discovered a species of frog that glows under ultraviolet light.",
    "She opened the window and listened to the rain falling on the quiet street.",
    "The company reported higher profits this quarter, driven by strong demand for its products.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The children played football in the park until the sun went down.",
    "He spent most of the afternoon reading a long novel about a sailor lost at sea.",
    "The museum will open a new exhibition of modern paintings next month.",
    "Doctors recommend drinking plenty of water and getting enough sleep every night.",
    "The train was delayed for two hours because of a problem with the signals.",
    "Our team finished the project ahead of schedule and under budget.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Many students struggle with mathematics during their first year at university.",
    "The old bridge was closed for repairs after engineers found cracks in the supports.",
    "A strong wind knocked down several trees and left thousands of homes without power.",
    "The film received excellent reviews from critics and became a box office success.",
    "Farmers in the region are worried about the lack of rain this summer.",
    "The new smartphone has a larger screen and a longer battery life than the previous model.",
    "She learned to play the piano when she was only six years old.",
    "The library offers free classes on computer skills for adults.",
    # geography-flavoured text: realistic situations where the MoI router CAN fire
    "Last summer we visited France and spent a week in Paris before driving south to the coast.",
    "The capital of Germany has a vibrant art scene and many historic buildings.",
    "Spanish is the official language of Mexico, although dozens of indigenous languages are also spoken.",
    "Toronto is located in Canada, on the northern shore of Lake Ontario.",
    "The capital city of Japan hosted the Olympic Games in 1964 and again in 2021.",
    "Many tourists who visit Italy start their trip in Rome and then travel to Florence and Venice.",
    "In Brazil, the language spoken by most people is Portuguese, not Spanish.",
    "The economy of Kenya has grown quickly, and its capital is a regional business hub.",
]


# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=9, help="GPT-2 layer whose MLP hosts the fact neurons.")
    ap.add_argument("--wikitext", action="store_true", help="Also evaluate on WikiText-2 test (downloads ~1 MB).")
    ap.add_argument("--wikitext-tokens", type=int, default=40_000)
    ap.add_argument("--teacher", type=str, default=None, help="HF id or local path of a bigger reference model.")
    ap.add_argument("--max-facts", type=int, default=0, help="Cap on installed (gap) facts, 0 = all.")
    ap.add_argument("--lat-rounds", type=int, default=3)
    ap.add_argument("--configs", type=str, default="base,moi",
                    help="Comma list from base,static,moi. `static` (all facts in N fixed neurons) is off by default:"
                         " with 162 facts it collapses WikiText-2 perplexity 48.6 -> 6325.")
    ap.add_argument("--gate-window", type=int, default=12,
                    help="MoI proximity gate: the paged neuron is only allowed to fire at positions t where the"
                         " routed relation word AND subject both occurred within the last W tokens (causal)."
                         " 0 = no gate (neuron active over the whole sequence).")
    ap.add_argument("--strict-key", action="store_true",
                    help="Add WikiText-2 VALIDATION positions as generic negatives when calibrating keys"
                         " (evaluation uses the TEST split, so there is no leakage).")
    ap.add_argument("--out", type=str, default=str(ROOT / "benchmarks" / "moi_fact_transfer" / "results.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    t_start = time.perf_counter()
    L = int(args.layer)

    tok = GPT2Tokenizer.from_pretrained(str(ROOT / "models" / "gpt2"))
    model = GPT2LMHeadModel.from_pretrained(str(ROOT / "models" / "gpt2")).eval()
    mlp = model.transformer.h[L].mlp
    W_fc, b_fc, W_pr = mlp.c_fc.weight.data, mlp.c_fc.bias.data, mlp.c_proj.weight.data  # [768,3072],[3072],[3072,768]
    d_model = int(model.config.n_embd)
    wte = model.transformer.wte.weight.data
    lnf_gain = model.transformer.ln_f.weight.data

    def enc(text: str) -> list[int]:
        return tok.encode(text)

    def single_tok(word: str) -> int | None:
        ids = enc(" " + word)
        return ids[0] if len(ids) == 1 else None

    # ---- 1. Build fact list (single-token subject; target = FIRST token of " answer") ----
    facts: list[dict] = []
    for rel, spec in RELATIONS.items():
        for subj, ans in spec["facts"].items():
            s_id = single_tok(subj)
            if s_id is None:
                continue
            ans_ids = enc(" " + ans)
            if ans.lower() == subj.lower():
                continue
            facts.append({"rel": rel, "subj": subj, "ans": ans, "s_id": s_id, "o_id": ans_ids[0],
                          "o_tok": tok.decode([ans_ids[0]])})
    a_ids_by_rel = {rel: [i for w in spec["a_words"] for i in [enc(w)[0]] if len(enc(w)) == 1]
                    for rel, spec in RELATIONS.items()}

    # ---- forward helpers ----
    caps: list[torch.Tensor] = []

    def last_logits(ids: list[int]) -> torch.Tensor:
        return model(input_ids=torch.tensor([ids])).logits[0, -1]

    def pre_mlp_state(text: str) -> torch.Tensor:
        caps.clear()
        model(input_ids=torch.tensor([enc(text)]))
        return caps[-1]

    def logp_and_top1(ids: list[int]) -> tuple[torch.Tensor, int]:
        lg = last_logits(ids)
        return F.log_softmax(lg.float(), dim=-1), int(lg.argmax().item())

    # ---- 2. Which facts does base GPT-2 already know? (held-out TEST templates, majority top-1) ----
    def fill(tpl: str, s: str) -> str:
        return tpl.format(s=s)

    for f in facts:
        spec = RELATIONS[f["rel"]]
        hits = sum(logp_and_top1(enc(fill(t, f["subj"])))[1] == f["o_id"] for t in spec["test"])
        f["base_known"] = hits >= 2
    gap = [f for f in facts if not f["base_known"]]
    known = [f for f in facts if f["base_known"]]
    if args.max_facts:
        gap = gap[: args.max_facts]
    print(f"Facts usable: {len(facts)} | GPT-2 already knows {len(known)} | gap facts to install: {len(gap)}")

    # ---- 3. Calibration states at layer L (pre-MLP, last position) ----
    hook = model.transformer.h[L].ln_2.register_forward_hook(lambda m, i, o: caps.append(o[0, -1].detach().clone()))
    t_cal = time.perf_counter()
    train_states: dict[tuple[str, str], torch.Tensor] = {}
    for f in facts:
        key = (f["rel"], f["subj"])
        if key in train_states:
            continue
        xs = [pre_mlp_state(p + fill(t, f["subj"])) for p in CALIB_PREFIXES for t in RELATIONS[f["rel"]]["train"]]
        train_states[key] = torch.stack(xs)
    # all positions of the plain corpus as generic negatives
    gen_caps: list[torch.Tensor] = []
    h_all = model.transformer.h[L].ln_2.register_forward_hook(lambda m, i, o: gen_caps.append(o[0].detach().clone()))
    hook.remove()
    for sent in GENERAL_CORPUS[:20]:
        model(input_ids=torch.tensor([enc(sent)]))
    if args.strict_key:
        import pandas as pd

        val_ids = enc("\n".join(pd.read_parquet(ROOT / "work" / "wikitext" / "validation.parquet")["text"].tolist()))
        for i in range(0, min(len(val_ids), 128 * 40), 128):
            model(input_ids=torch.tensor([val_ids[i : i + 128]]))
    h_all.remove()
    X_generic = torch.cat(gen_caps, dim=0)
    print(f"Generic negative positions for key calibration: {X_generic.shape[0]}")
    # unrelated subject mentions (neg templates) for all subjects
    hook = model.transformer.h[L].ln_2.register_forward_hook(lambda m, i, o: caps.append(o[0, -1].detach().clone()))
    neg_subj_states = {
        s: torch.stack([pre_mlp_state(fill(t, s)) for t in NEG_TEMPLATES[:2]])
        for s in sorted({f["subj"] for f in facts})
    }
    hook.remove()
    print(f"Calibration states collected in {time.perf_counter() - t_cal:.1f}s")

    # ---- 4. Dormant neurons ----
    score = W_fc.norm(dim=0) * W_pr.norm(dim=1)
    dormant = score.topk(len(gap) + 1, largest=False).indices.tolist()
    u_moi = int(dormant[0])
    static_units = [int(u) for u in dormant[: len(gap)]]
    orig = {u: (W_fc[:, u].clone(), b_fc[u].clone(), W_pr[u].clone()) for u in set(dormant)}

    def write_unit(u: int, key: torch.Tensor, bias: float, val: torch.Tensor) -> None:
        W_fc[:, u].copy_(key)
        b_fc[u] = bias
        W_pr[u].copy_(val)

    def restore(units) -> None:
        for u in units:
            k, b, v = orig[u]
            W_fc[:, u].copy_(k)
            b_fc[u] = b
            W_pr[u].copy_(v)

    # ---- 5. Compile one payload per gap fact (key from train states, value scale from train prompts) ----
    ans_ids_by_rel: dict[str, list[int]] = {}
    for f in facts:
        ans_ids_by_rel.setdefault(f["rel"], []).append(f["o_id"])
    TARGET_PREACT = 6.0
    SCALE_GRID = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    t_comp = time.perf_counter()
    payloads: list[dict] = []
    for f in gap:
        key_t = (f["rel"], f["subj"])
        X_pos = train_states[key_t]
        neg_list = [v for k, v in train_states.items() if k != key_t]
        neg_list += [neg_subj_states[f["subj"]], X_generic]
        X_neg = torch.cat(neg_list, dim=0)
        mu_p, mu_n = X_pos.mean(0), X_neg.mean(0)
        Xc = X_neg - mu_n
        cov = Xc.T @ Xc / max(1, Xc.shape[0] - 1)
        lam = 0.05 * cov.diagonal().mean()
        k = torch.linalg.solve(cov + lam * torch.eye(d_model), mu_p - mu_n)
        k = k / k.norm()
        sp, sn = X_pos @ k, X_neg @ k
        m_p, m_n = float(sp.mean()), float(sn.mean())
        delta = max(m_p - m_n, 1e-6)
        zp, zn = (sp - m_n) / delta, (sn - m_n) / delta
        tau = float((torch.quantile(zn, 0.999) + zp.min()) / 2) if float(zp.min()) > float(torch.quantile(zn, 0.999)) else 0.5
        tau = min(max(tau, 0.2), 0.8)
        gain = TARGET_PREACT / max(1.0 - tau, 0.2)
        key_vec = gain * k / delta
        bias = -gain * (m_n / delta + tau)

        others = [o for o in set(ans_ids_by_rel[f["rel"]]) if o != f["o_id"]]
        d = lnf_gain * (wte[f["o_id"]] - wte[others].mean(0))
        d = d / d.norm()
        chosen = SCALE_GRID[-1]
        train_prompts = [enc(fill(t, f["subj"])) for t in RELATIONS[f["rel"]]["train"]]
        for s in SCALE_GRID:
            write_unit(u_moi, key_vec, bias, s * d)
            if all(logp_and_top1(ids)[1] == f["o_id"] for ids in train_prompts):
                chosen = s
                break
        restore([u_moi])
        f["val_scale"] = chosen
        f["neg_fire_rate_train"] = float((zn > tau).float().mean())
        payloads.append({"key": key_vec.clone(), "bias": float(bias), "val": (chosen * d).clone()})
    print(f"Compiled {len(gap)} fact circuits in {time.perf_counter() - t_comp:.1f}s")

    # ---- 6. Register payloads in the wsi Mixture-of-Indexes catalog (router = relation word AND subject) ----
    g2 = WeightSpaceIndex.from_model(ROOT / "models" / "gpt2")
    # The wsi pager stores 2-slot (src -> dst) payloads. Fact circuits are single-neuron, so the src slot
    # (layer L-1, its most dormant unit) is registered with its ORIGINAL weights, i.e. a no-op.
    src_mlp = model.transformer.h[L - 1].mlp
    s_score = src_mlp.c_fc.weight.data.norm(dim=0) * src_mlp.c_proj.weight.data.norm(dim=1)
    u_src = int(s_score.argmin().item())
    noop_src = {"key_src": src_mlp.c_fc.weight.data[:, u_src].clone(), "bias_src": float(src_mlp.c_fc.bias.data[u_src]),
                "val_src": src_mlp.c_proj.weight.data[u_src].clone()}
    pager = g2.create_mixture_of_indexes(src_layer=L - 1, dst_layer=L, src_unit=u_src, dst_unit=u_moi)
    for n, (f, p) in enumerate(zip(gap, payloads)):
        pager.register_circuit(
            f"{f['rel']}:{f['subj']}",
            {"_slot_payload": {**noop_src, "key_dst": p["key"], "bias_dst": p["bias"], "val_dst": p["val"],
                               "output_tokens": [f["o_tok"]]}},
            a_token_ids=a_ids_by_rel[f["rel"]], b_token_ids=[f["s_id"]],
            a_tokens=RELATIONS[f["rel"]]["a_words"], b_tokens=[" " + f["subj"]],
        )
    cid_to_idx = {f"{f['rel']}:{f['subj']}": n for n, f in enumerate(gap)}
    payload_bytes_effective = (2 * d_model + 1) * 4  # what is actually paged: key column + bias + value row

    # ---- 7. Configuration switching ----
    state = {"cfg": "base", "loaded": None, "swaps": 0, "route_ms": [], "mask": None}

    def _gate_hook(module, inp, out):
        # Proximity gate: zero the MoI slot's activation at positions not "near" the routed triggers.
        if state["cfg"] == "moi" and state["loaded"] is not None and state["mask"] is not None:
            out[0, :, u_moi] = out[0, :, u_moi] * state["mask"]
        return out

    mlp.act.register_forward_hook(_gate_hook)

    def proximity_mask(ids: list[int], a_set: set[int], b_id: int) -> torch.Tensor:
        W = args.gate_window
        last_a = last_b = -10**9
        m = torch.zeros(len(ids))
        for t, tid in enumerate(ids):
            if tid in a_set:
                last_a = t
            if tid == b_id:
                last_b = t
            m[t] = float(t - min(last_a, last_b) <= W)
        return m

    def set_config(cfg: str) -> None:
        restore(orig.keys())
        state.update(cfg=cfg, loaded=None)
        if cfg == "static":
            for u, p in zip(static_units, payloads):
                write_unit(u, p["key"], p["bias"], p["val"])

    def prepare(ids: list[int]) -> dict | None:
        """MoI: route the prompt and page the matching payload into the shared slot (or restore it)."""
        if state["cfg"] != "moi":
            return None
        t0 = time.perf_counter()
        r = pager.route_token_ids(ids)
        want = cid_to_idx[r["circuit_id"]] if r["matched"] else None
        if want != state["loaded"]:
            if want is None:
                restore([u_moi])
            else:
                p = payloads[want]
                write_unit(u_moi, p["key"], p["bias"], p["val"])
            state["loaded"] = want
            state["swaps"] += 1
        if want is not None and args.gate_window > 0:
            f = gap[want]
            state["mask"] = proximity_mask(ids, set(a_ids_by_rel[f["rel"]]), f["s_id"])
        else:
            state["mask"] = None
        state["route_ms"].append((time.perf_counter() - t0) * 1e3)
        return r

    def run(ids: list[int]) -> tuple[torch.Tensor, int]:
        prepare(ids)
        return logp_and_top1(ids)

    # ---- 8. Evaluation ----
    base_cache: dict[str, torch.Tensor] = {}

    def eval_set(fs: list[dict], kind: str) -> dict:
        n = hit = 0
        dlp: list[float] = []
        for f in fs:
            for t in RELATIONS[f["rel"]][kind]:
                text = fill(t, f["subj"])
                ids = enc(text)
                lp, top = run(ids)
                n += 1
                hit += int(top == f["o_id"])
                dlp.append(float(lp[f["o_id"]] - base_cache[text][f["o_id"]]))
        return {"acc": hit / max(n, 1), "n": n, "mean_dlogP": statistics.fmean(dlp) if dlp else 0.0}

    def eval_neg(fs: list[dict]) -> dict:
        n = wrong = 0
        kls: list[float] = []
        for f in fs:
            for t in NEG_TEMPLATES:
                text = fill(t, f["subj"])
                ids = enc(text)
                lp, top = run(ids)
                b = base_cache[text]
                n += 1
                wrong += int(top == f["o_id"] and int(b.argmax()) != f["o_id"])
                kls.append(float(F.kl_div(lp, b, log_target=True, reduction="sum")))
        return {"false_trigger_rate": wrong / max(n, 1), "mean_KL": statistics.fmean(kls) if kls else 0.0, "n": n}

    def corpus_texts() -> list[list[int]]:
        chunks = [enc(s) for s in GENERAL_CORPUS]
        if args.wikitext:
            local = ROOT / "work" / "wikitext" / "test.parquet"
            if local.exists():
                import pandas as pd

                texts = pd.read_parquet(local)["text"].tolist()
            else:
                from datasets import load_dataset

                texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
            raw = "\n".join(texts)
            ids = enc(raw)[: args.wikitext_tokens]
            chunks += [ids[i : i + 256] for i in range(0, len(ids) - 1, 256) if len(ids[i : i + 256]) > 8]
        return chunks

    corpus = corpus_texts()
    base_corpus_lp: list[torch.Tensor] = []

    def eval_corpus(record_base: bool) -> dict:
        nll = 0.0
        ntok = 0
        kl_sum = 0.0
        fired = 0
        paged_kl = paged_tok = 0.0
        for ci, ids in enumerate(corpus):
            r = prepare(ids)
            is_paged = bool(r and r["matched"])
            fired += int(is_paged)
            lp = F.log_softmax(model(input_ids=torch.tensor([ids])).logits[0, :-1].float(), dim=-1)
            tgt = torch.tensor(ids[1:])
            nll -= float(lp.gather(1, tgt[:, None]).sum())
            ntok += len(ids) - 1
            if record_base:
                base_corpus_lp.append(lp)
            else:
                kl_c = float(F.kl_div(lp, base_corpus_lp[ci], log_target=True, reduction="sum"))
                kl_sum += kl_c
                if is_paged:
                    paged_kl += kl_c
                    paged_tok += len(ids) - 1
        unpaged_tok = ntok - paged_tok
        return {"ppl": float(torch.exp(torch.tensor(nll / ntok))), "kl_per_token": kl_sum / ntok,
                "n_tokens": ntok, "n_chunks": len(corpus), "chunks_where_moi_paged": fired,
                "kl_per_token_paged_chunks": paged_kl / paged_tok if paged_tok else 0.0,
                "kl_per_token_unpaged_chunks": (kl_sum - paged_kl) / unpaged_tok if unpaged_tok else 0.0}

    # base pass caches
    set_config("base")
    for f in facts:
        for kind in ("test", "para"):
            for t in RELATIONS[f["rel"]][kind]:
                text = fill(t, f["subj"])
                base_cache[text] = logp_and_top1(enc(text))[0]
        for t in NEG_TEMPLATES:
            text = fill(t, f["subj"])
            base_cache[text] = logp_and_top1(enc(text))[0]

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if configs[0] != "base":
        configs = ["base"] + [c for c in configs if c != "base"]  # base caches the reference distributions
    results: dict[str, dict] = {}
    for cfg in configs:
        set_config(cfg)
        state["route_ms"].clear()
        state["swaps"] = 0
        t0 = time.perf_counter()
        res = {
            "installed_facts_heldout": eval_set(gap, "test"),
            "installed_facts_paraphrase": eval_set(gap, "para"),
            "known_facts_retention": eval_set(known, "test"),
            "false_triggers": eval_neg(gap),
            "general_text": eval_corpus(record_base=(cfg == "base")),
        }
        res["neurons_overwritten"] = {"base": 0, "static": len(gap), "moi": 1}[cfg]
        res["extra_resident_bytes"] = len(gap) * payload_bytes_effective if cfg == "moi" else 0
        if cfg == "moi":
            res["swaps"] = state["swaps"]
            res["route_swap_mean_ms"] = statistics.fmean(state["route_ms"]) if state["route_ms"] else 0.0
        res["eval_wall_s"] = time.perf_counter() - t0
        results[cfg] = res
        print(f"[{cfg:6s}] heldout acc={res['installed_facts_heldout']['acc']:.3f} "
              f"para acc={res['installed_facts_paraphrase']['acc']:.3f} "
              f"known={res['known_facts_retention']['acc']:.3f} "
              f"false_trig={res['false_triggers']['false_trigger_rate']:.3f} "
              f"ppl={res['general_text']['ppl']:.3f} kl/tok={res['general_text']['kl_per_token']:.5f}")

    # ---- 9. Latency (interleaved rounds, B=1, CPU) ----
    lat_prompts = [enc(fill(RELATIONS[f["rel"]]["test"][0], f["subj"])) for f in gap[:40]]
    lat_prompts += [enc(s) for s in GENERAL_CORPUS[:20]]
    lat: dict[str, list[float]] = {c: [] for c in configs}
    for _ in range(args.lat_rounds):
        for cfg in configs:
            set_config(cfg)
            for ids in lat_prompts:
                t0 = time.perf_counter()
                run(ids)
                lat[cfg].append((time.perf_counter() - t0) * 1e3)
    for cfg in lat:
        xs = sorted(lat[cfg])
        results[cfg]["latency_ms"] = {"mean": statistics.fmean(xs), "p50": xs[len(xs) // 2],
                                      "p95": xs[int(len(xs) * 0.95)]}
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    for cfg in results:
        results[cfg]["param_bytes"] = param_bytes

    # ---- 10. Optional teacher reference ----
    if args.teacher:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ttok = AutoTokenizer.from_pretrained(args.teacher)
        tmodel = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.float32).eval()
        t_hit = t_n = 0
        t_lat: list[float] = []
        for f in gap + known:
            for t in RELATIONS[f["rel"]]["test"]:
                ids = ttok(fill(t, f["subj"]), return_tensors="pt").input_ids
                t0 = time.perf_counter()
                out = tmodel.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=ttok.eos_token_id)
                t_lat.append((time.perf_counter() - t0) * 1e3 / 4)
                txt = ttok.decode(out[0, ids.shape[1]:]).strip().lower()
                if f in gap:
                    t_n += 1
                    t_hit += int(txt.startswith(f["ans"].lower()[: max(3, len(f["o_tok"].strip()))]))
        results["teacher"] = {
            "name": args.teacher,
            "installed_facts_heldout_acc": t_hit / max(t_n, 1),
            "param_bytes": sum(p.numel() * p.element_size() for p in tmodel.parameters()),
            "latency_ms_per_token": statistics.fmean(t_lat),
        }

    summary = {
        "layer": L,
        "n_facts_usable": len(facts),
        "n_known_by_gpt2": len(known),
        "n_installed": len(gap),
        "moi_slot_unit": u_moi,
        "payload_bytes_per_fact": payload_bytes_effective,
        "wsi_pager_catalog": pager.catalog_summary(),
        "process_rss_mb": psutil.Process().memory_info().rss / 2**20,
        "wall_s": time.perf_counter() - t_start,
        "corpus": "builtin" + ("+wikitext2" if args.wikitext else ""),
        "gate_window": args.gate_window,
        "strict_key": bool(args.strict_key),
        "results": results,
        "installed_facts": [{k: f[k] for k in ("rel", "subj", "ans", "o_tok", "val_scale", "neg_fire_rate_train")}
                            for f in gap],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {out} ({summary['wall_s']:.1f}s total)")


if __name__ == "__main__":
    main()
