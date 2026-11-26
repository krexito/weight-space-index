"""Build an Obsidian vault (a folder of linked markdown notes) from everything WSI has measured.

Sources (read only):
  research/findings.md                                   one note per finding bullet (title, date, verdict, links)
  benchmarks/xray_atlas/tome1_xray/atlas_summary.json    Tome 1: one X-ray card per model (+ nearest models by X-ray)
  benchmarks/xray_atlas/tome2_delta/delta_atlas_summary.json   Tome 2: before -> after training transitions
  benchmarks/xray_atlas/tome3_clinical_history.json      Tome 3: treatments per model, cross-tome laws

Nodes: Models, Behaviors, Methods, Findings, Transitions, Treatments, Laws.  Every link is a [[wikilink]], so the
Obsidian graph view shows the map.  Gaps.md lists what the map says we have not measured yet.

  python scripts/build_wsi_vault.py --out wsi_vault [--findings research/findings.md other/findings.md]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ATLAS = ROOT / "benchmarks" / "xray_atlas"

# ---------------------------------------------------------------- vocabularies (alias regex -> note name)
EXTRA_MODELS = {  # models that appear in findings but have no Tome-1 card
    "microsoft/Phi-4-mini-instruct": [r"Phi-4-mini"],
    "ByteDance/Ouro-2.6B": [r"Ouro-2\.6B(?!-Thinking)", r"\bOuro\b"],
    "ByteDance/Ouro-2.6B-Thinking": [r"Ouro-2\.6B-Thinking", r"Ouro.{0,20}Thinking"],
    "sensenova/SenseNova-Loop-9B": [r"SenseNova"],
    "Qwen/Qwen2.5-14B": [r"Qwen2\.5-14B"],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": [r"R1-Distill-Qwen-14B", r"R1-Distill"],
    "mistralai/Mistral-7B-v0.1": [r"Mistral-7B"],
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": [r"Nemotron"],
    "Qwen/Qwen3-30B-A3B-Instruct-2507": [r"Qwen3-30B-A3B"],
    "allenai/OLMo-2-0325-32B-Instruct": [r"OLMo-2-32B", r"olmo-2-0325-32b"],
    "zai-org/GLM-4-32B-0414": [r"GLM-4-32B"],
    "thinkingmachines/Inkling-2-Mini": [r"Inkling"],
    "Qwen/Qwen3-0.6B": [r"Qwen3-0\.6B"],
    "Qwen/Qwen2.5-1.5B": [r"Qwen2\.5-1\.5B"],
}
SHORT_ALIASES = {  # extra ways the findings name atlas models
    "Qwen/Qwen3.5-0.8B": [r"\b0\.8B\b"],
    "Qwen/Qwen3.5-4B": [r"Qwen3\.5-4B", r"\bthe 4B\b", r"\b4B ->"],
    "LiquidAI/LFM3-1.7B": [r"LFM3"],
    "microsoft/Aion-1.0-Instruct-2.3B": [r"\bAion\b"],
    "Qwen/Qwen2.5-3B-Instruct": [r"Qwen2\.5-3B"],
    "allenai/OLMoE-1B-7B-0125-Instruct": [r"OLMoE"],
    "Zyphra/Zamba2-2.7B-instruct": [r"Zamba2"],
}
BEHAVIORS = {
    "Loops": [r"\bloop", r"repetit", r"restart", r"escape", r"rescue"],
    "Over-refusal": [r"over-?refus", r"OR-Bench", r"abliterat", r"refusal"],
    "Safety retention": [r"toxic", r"veto", r"jailbreak", r"safety"],
    "Sycophancy": [r"sycophan", r"pushback", r"\bcave"],
    "Fact memory": [r"facts? in the weights", r"memory", r"ROME", r"recall"],
    "Compression": [r"quantiz", r"\d-bit", r"prun", r"compress", r"rank frontier"],
    "Distillation": [r"distill", r"teacher", r"cartridge", r"Pocket MoE"],
    "Reasoning accuracy": [r"GSM8K", r"MATH-500", r"MMLU", r"capability"],
    "Prose damage": [r"prose", r"style", r"voice", r"damage map", r"collateral"],
    "Inference speed": [r"speculative", r"\bspeed", r"latency", r"x faster", r"streaming"],
}
METHODS = {
    "Blind discovery": [r"blind"],
    "MoI routing": [r"\bMoI\b", r"routed", r"router", r"routing"],
    "Self-gate veto": [r"self-gate", r"self-gating"],
    "1-pass screen": [r"\bscreen\b", r"screen_patch", r"pareto"],
    "Escape test": [r"escape"],
    "Fisher": [r"Fisher"],
    "LoRA": [r"\bLoRA", r"\br=\d"],
    "Direction edits": [r"direction", r"diff-of-means", r"veto_dir", r"refusal_dir"],
    "Trace capture": [r"CompactTrace", r"trace", r"recompute"],
    "X-Ray Atlas": [r"X-Ray", r"Atlas", r"Tome"],
    "Loopification": [r"loopif", r"frankenmerge", r"Looped"],
    "Sensor token": [r"sensor"],
    "Live agent": [r"agent"],
    "Closed-form edits": [r"closed-form", r"ridge", r"least squares", r"0-backprop"],
    "MIT p50c patch": [r"p50c", r"\bMIT\b"],
}
VERDICTS = [("negative", r"negative|no gain|no usable|dropped|no arm passes|fails|collapse|breaks"),
            ("tie", r"\btie|ties\b"), ("partial", r"partly|weak|small positive|mixed|partial"),
            ("positive", r"positive|works|pass\b|win|keeps the gain|new best|record|beats")]
# Tome-1 metrics used for the "looks alike" neighbours (z-scored, missing -> 0)
SIM_METRICS = ["unembed_anisotropy_cos", "readout_align_early", "mlp_energy90_mean", "fisher_late_early_ratio",
               "block_fisher_late_early_ratio", "fisher_col_gini_mean", "fact_dla_first_quarter",
               "opening_entropy_norm", "cos_refusal_veto", "cos_syc_refusal", "cos_rep_vs_copy"]
SYMPTOMS = {  # Tome-1 metric -> (behavior, threshold): the base model shows the problem
    "base_overref_rate": ("Over-refusal", 0.30),
    "sampled_rep16_rate": ("Loops", 0.05),
    "greedy_rep16_rate": ("Loops", 0.20),
    "cave_rate_false_pushback": ("Sycophancy", 0.50),
}


def slug(s: str) -> str:
    return re.sub(r"[\\/:*?\"<>|#^\[\]]", "-", s).strip()


def mnote(repo: str) -> str:
    return slug(repo.split("/")[-1])


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def compile_vocab(d):
    return {k: [re.compile(p, re.I) for p in v] for k, v in d.items()}


def find(text, vocab):
    return sorted(k for k, pats in vocab.items() if any(p.search(text) for p in pats))


# ---------------------------------------------------------------- parsing
def parse_findings(paths):
    items, seen = [], set()
    for p in paths:
        lines = Path(p).read_text(encoding="utf-8").splitlines()
        cur = None
        for ln in lines:
            if ln.startswith("- **"):
                cur = [ln]
                items.append(cur)
            elif ln.startswith("#"):
                cur = None
            elif cur is not None and ln.strip():
                cur.append(ln)
    out = []
    for blk in items:
        text = "\n".join(blk)
        m = re.match(r"- \*\*(.+?)\*\*", text)
        head = m.group(1) if m else text[:80]
        title = re.split(r" \(|:\s*$", re.sub(r"`", "", head))[0].strip(" :.")
        key = re.sub(r"\W+", "", head.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        date = re.search(r"20\d\d-\d\d-\d\d", head)
        verdict = next((v for v, pat in VERDICTS if re.search(pat, head, re.I)), None) or next(
            (v for v, pat in VERDICTS if re.search(pat, text[len(head):len(head) + 400], re.I)), "result")
        files = sorted(set(re.findall(r"benchmarks/[\w./-]+\.json", text)))
        out.append({"title": title[:90], "head": head, "text": text, "date": date.group(0) if date else "",
                    "verdict": verdict, "files": files})
    # unique note names
    cnt = defaultdict(int)
    for f in out:
        base = slug(f["title"]) or "finding"
        cnt[base] += 1
        f["note"] = base if cnt[base] == 1 else f"{base} ({f['date'] or cnt[base]})"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="wsi_vault")
    ap.add_argument("--findings", nargs="+", default=[str(ROOT / "research" / "findings.md")])
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        for sub in ("Models", "Behaviors", "Methods", "Findings", "Transitions", "Treatments", "Laws"):
            shutil.rmtree(out / sub, ignore_errors=True)
    for sub in ("Models", "Behaviors", "Methods", "Findings", "Transitions", "Treatments", "Laws", ".obsidian"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    t1 = json.loads((ATLAS / "tome1_xray" / "atlas_summary.json").read_text(encoding="utf-8"))["models"]
    t2 = json.loads((ATLAS / "tome2_delta" / "delta_atlas_summary.json").read_text(encoding="utf-8"))["pairs"]
    t3 = json.loads((ATLAS / "tome3_clinical_history.json").read_text(encoding="utf-8"))
    for extra in sorted(ATLAS.glob("*.xray.json")):  # loose cards (Ouro, SenseNova)
        try:
            d = json.loads(extra.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        rid = d.get("repo_id") or d.get("model", {}).get("repo_id") if isinstance(d, dict) else None
        if rid and rid not in {m["repo_id"] for m in t1}:
            t1.append({"repo_id": rid, "loose_card": extra.name})

    models = {m["repo_id"]: {"card": m, "aliases": [re.escape(m["repo_id"]), re.escape(m["repo_id"].split("/")[-1])]}
              for m in t1}
    for p in t2:
        for r in (p["before_repo"], p["after_repo"]):
            models.setdefault(r, {"card": None, "aliases": [re.escape(r)]})
    for r in t3["models"]:
        models.setdefault(r, {"card": None, "aliases": [re.escape(r), re.escape(r.split("/")[-1])]})
    for r, al in EXTRA_MODELS.items():
        models.setdefault(r, {"card": None, "aliases": [re.escape(r)]})["aliases"] += al
    for r, al in SHORT_ALIASES.items():
        if r in models:
            models[r]["aliases"] += al
    mvocab = {r: [re.compile(p, re.I) for p in v["aliases"]] for r, v in models.items()}
    bvocab, tvocab = compile_vocab(BEHAVIORS), compile_vocab(METHODS)

    findings = parse_findings(a.findings)
    links = {"model": defaultdict(list), "behavior": defaultdict(list), "method": defaultdict(list)}
    for f in findings:
        f["models"] = find(f["text"], mvocab)
        f["behaviors"] = find(f["text"][:700], bvocab)
        f["methods"] = find(f["text"], tvocab)
        for k, key in (("models", "model"), ("behaviors", "behavior"), ("methods", "method")):
            for x in f[k]:
                links[key][x].append(f)

    # ---------- findings
    for f in findings:
        fm = ["---", f"type: finding", f"verdict: {f['verdict']}", f"date: {f['date']}",
              f"tags: [finding, finding/{f['verdict']}]", "---"]
        body = [f"# {f['title']}", "",
                f"**Verdict:** {f['verdict']}" + (f" · **Date:** {f['date']}" if f['date'] else ""), "",
                "**Models:** " + (", ".join(f"[[{mnote(m)}]]" for m in f["models"]) or "-"),
                "**Behaviors:** " + (", ".join(f"[[{b}]]" for b in f["behaviors"]) or "-"),
                "**Methods:** " + (", ".join(f"[[{t}]]" for t in f["methods"]) or "-"), ""]
        if f["files"]:
            body += ["**Data:** " + ", ".join(f"`{x}`" for x in f["files"]), ""]
        body += ["## From research/findings.md", "", re.sub(r"^- ", "", f["text"])]
        (out / "Findings" / f"{f['note']}.md").write_text("\n".join(fm + body) + "\n", encoding="utf-8")

    # ---------- X-ray similarity between carded models
    carded = [r for r, v in models.items() if v["card"] and "loose_card" not in v["card"]]
    X = np.array([[float(models[r]["card"].get(k)) if isinstance(models[r]["card"].get(k), (int, float)) else np.nan
                   for k in SIM_METRICS] for r in carded])
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
    Z = np.nan_to_num((X - mu) / sd)
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).mean(-1))
    near = {r: [(carded[j], float(D[i, j])) for j in np.argsort(D[i]) if j != i][:3] for i, r in enumerate(carded)}

    # ---------- transitions / treatments / laws
    trans_by_model = defaultdict(list)
    for p in t2:
        n = slug(p["pair_id"])
        trans_by_model[p["before_repo"]].append(n)
        trans_by_model[p["after_repo"]].append(n)
        keys = [k for k in p if k not in ("pair_id", "before_repo", "after_repo", "detail_file", "json_bytes", "detail_bytes")]
        txt = ["---", "type: transition", f"transition: {p['transition_type']}", "tags: [transition]", "---",
               f"# {p['pair_id']}", "", f"[[{mnote(p['before_repo'])}]] → [[{mnote(p['after_repo'])}]] "
               f"({p['transition_type']}) · [[X-Ray Atlas|Tome 2]]", "", "| metric | value |", "|---|---|"]
        txt += [f"| {k} | {fmt(p[k])} |" for k in keys]
        (out / "Transitions" / f"{n}.md").write_text("\n".join(txt) + "\n", encoding="utf-8")

    path_map = {"over_refusal": "Over-refusal", "loops": "Loops", "loop": "Loops", "sycophancy": "Sycophancy",
                "memory_injection": "Fact memory", "pruning": "Compression", "quantization": "Compression",
                "toxic": "Safety retention"}
    treat_by_model, treat_by_beh = defaultdict(list), defaultdict(list)
    for repo, recs in t3["models"].items():
        for i, r in enumerate(recs):
            beh = next((v for k, v in path_map.items() if k in r.get("pathology", "")), r.get("pathology", "?"))
            n = slug(f"{mnote(repo)} - {r.get('pathology', '?')} - {r.get('strategy', i)}")
            treat_by_model[repo].append((n, r.get("outcome"), beh))
            treat_by_beh[beh].append((n, r.get("outcome")))
            met = r.get("metrics", {})
            txt = ["---", "type: treatment", f"outcome: {r.get('outcome')}", f"tags: [treatment, treatment/{r.get('outcome')}]",
                   "---", f"# {r.get('strategy')} on {repo}", "",
                   f"**Model:** [[{mnote(repo)}]] · **Behavior:** [[{beh}]] · **Outcome:** {r.get('outcome')} · [[X-Ray Atlas|Tome 3]]",
                   "", f"**Target:** `{json.dumps(r.get('target', {}))}`", "",
                   "| metric | value |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in met.items()]
            if r.get("caveat"):
                txt += ["", f"**Caveat:** {r['caveat']}"]
            (out / "Treatments" / f"{n}.md").write_text("\n".join(txt) + "\n", encoding="utf-8")

    for c in t3.get("cross_tome_findings", []):
        ms = find(c["finding"] + " " + c["source"], mvocab)
        bs = find(c["finding"], bvocab)
        txt = ["---", "type: law", "tags: [law]", "---", f"# {c['id']}", "", f"**Source:** {c['source']}", "",
               c["finding"], "", "**Models:** " + (", ".join(f"[[{mnote(m)}]]" for m in ms) or "-"),
               "**Behaviors:** " + (", ".join(f"[[{b}]]" for b in bs) or "-"), "", "[[X-Ray Atlas]]"]
        (out / "Laws" / f"{slug(c['id'])}.md").write_text("\n".join(txt) + "\n", encoding="utf-8")

    # ---------- models
    symptoms_of = defaultdict(list)
    for r, v in models.items():
        c = v["card"] or {}
        for k, (beh, thr) in SYMPTOMS.items():
            if isinstance(c.get(k), (int, float)) and c[k] >= thr:
                symptoms_of[r].append((beh, k, c[k]))
        fs = links["model"].get(r, [])
        txt = ["---", "type: model", f"family: {r.split('/')[0]}", f"arch: {c.get('arch_family', '')}",
               "tags: [model" + (", model/carded" if c and 'loose_card' not in c else "") + "]", "---", f"# {r}", ""]
        if c and "loose_card" not in c:
            txt += [f"**X-ray card** ([[X-Ray Atlas|Tome 1]]): {c.get('params_b')}B params, {c.get('n_layers')} layers, "
                    f"{c.get('arch_family')}, {c.get('norm_type')}", ""]
            txt += ["**Looks most like (by X-ray):** " + ", ".join(f"[[{mnote(o)}]] ({d:.2f})" for o, d in near[r]), ""]
            txt += ["| metric | value |", "|---|---|"] + [f"| {k} | {fmt(c[k])} |" for k in sorted(c)
                                                          if k not in ("repo_id", "markdown_table") and c[k] is not None]
            txt += [""]
        elif c.get("loose_card"):
            txt += [f"**X-ray card:** `benchmarks/xray_atlas/{c['loose_card']}`", ""]
        if symptoms_of[r]:
            txt += ["## Symptoms in the base model"] + [f"- [[{b}]]: `{k}` = {fmt(val)}" for b, k, val in symptoms_of[r]] + [""]
        if trans_by_model.get(r):
            txt += ["## Training transitions"] + [f"- [[{n}]]" for n in trans_by_model[r]] + [""]
        if treat_by_model.get(r):
            txt += ["## Treatments"] + [f"- [[{n}]] ({o})" for n, o, _ in treat_by_model[r]] + [""]
        if fs:
            txt += [f"## Findings ({len(fs)})"] + [f"- [[{f['note']}]] · {f['verdict']}" for f in fs]
        (out / "Models" / f"{mnote(r)}.md").write_text("\n".join(txt) + "\n", encoding="utf-8")

    # ---------- behaviors and methods
    for name, kind, folder in [(b, "behavior", "Behaviors") for b in BEHAVIORS] + [(t, "method", "Methods") for t in METHODS]:
        fs = links[kind].get(name, [])
        by_v = defaultdict(int)
        for f in fs:
            by_v[f["verdict"]] += 1
        txt = ["---", f"type: {kind}", f"tags: [{kind}]", "---", f"# {name}", "",
               "**Findings by verdict:** " + (", ".join(f"{k} {v}" for k, v in sorted(by_v.items())) or "none"), ""]
        if kind == "behavior":
            sym = [(r, k, v) for r, ss in symptoms_of.items() for b, k, v in ss if b == name]
            if sym:
                txt += ["## Base models that show it (Tome 1)"] + [f"- [[{mnote(r)}]]: `{k}` = {fmt(v)}" for r, k, v in sym] + [""]
            if treat_by_beh.get(name):
                txt += ["## Treatments (Tome 3)"] + [f"- [[{n}]] ({o})" for n, o in treat_by_beh[name]] + [""]
        ms = defaultdict(int)
        for f in fs:
            for m in f["models"]:
                ms[m] += 1
        if ms:
            txt += ["## Models studied"] + [f"- [[{mnote(m)}]] ({n})" for m, n in sorted(ms.items(), key=lambda x: -x[1])] + [""]
        txt += [f"## Findings ({len(fs)})"] + [f"- [[{f['note']}]] · {f['verdict']}" for f in sorted(fs, key=lambda f: f["date"])]
        safe = slug(name)
        (out / folder / f"{safe}.md").write_text("\n".join(txt) + "\n", encoding="utf-8")

    # ---------- gaps
    beh_names = list(BEHAVIORS)
    studied = {r: {b for f in links["model"].get(r, []) for b in f["behaviors"]} for r in models}
    gaps = ["# Gaps: what the map says we have not measured", "",
            "Generated by `scripts/build_wsi_vault.py`. A gap is a question, not a result.", "",
            "## 1. The base model shows a problem, and no finding or treatment on that model touches it", ""]
    for r, ss in sorted(symptoms_of.items()):
        for b, k, v in ss:
            treated = any(beh == b for _, _, beh in treat_by_model.get(r, []))
            if b not in studied.get(r, set()) and not treated:
                gaps.append(f"- [[{mnote(r)}]] · [[{b}]] (`{k}` = {fmt(v)})")
    gaps += ["", "## 2. Models with an X-ray card and no finding at all", ""]
    gaps += [f"- [[{mnote(r)}]]" for r in carded if not links["model"].get(r)]
    gaps += ["", "## 3. Results that rest on one model only (check a second family)", ""]
    for f in findings:
        if len(f["models"]) == 1 and f["verdict"] in ("positive", "partial"):
            gaps.append(f"- [[{f['note']}]] · only [[{mnote(f['models'][0])}]]")
    gaps += ["", "## 4. Behaviors whose findings are mostly negative", ""]
    for b in beh_names:
        fs = links["behavior"].get(b, [])
        neg = sum(f["verdict"] == "negative" for f in fs)
        pos = sum(f["verdict"] == "positive" for f in fs)
        if fs and neg >= pos:
            gaps.append(f"- [[{b}]]: {neg} negative vs {pos} positive of {len(fs)}")
    gaps += ["", "## 5. Model x behavior coverage (findings that mention both)", "",
             "| model | " + " | ".join(beh_names) + " |", "|---|" + "---|" * len(beh_names)]
    for r in sorted(models, key=lambda r: -len(links["model"].get(r, []))):
        fs = links["model"].get(r, [])
        if not fs and not models[r]["card"]:
            continue
        row = [str(sum(b in f["behaviors"] for f in fs)) or "0" for b in beh_names]
        gaps.append(f"| [[{mnote(r)}]] | " + " | ".join(x if x != "0" else "·" for x in row) + " |")
    (out / "Gaps.md").write_text("\n".join(gaps) + "\n", encoding="utf-8")

    # ---------- home + graph colours
    home = ["# WSI map", "", "Open the graph view (Ctrl+G). Colours: models red, findings green (negative grey), "
            "behaviors orange, methods blue, transitions purple, treatments yellow, laws cyan.", "",
            f"- {len(models)} models · {len(findings)} findings · {len(t2)} transitions · "
            f"{sum(len(v) for v in t3['models'].values())} treatments · {len(t3.get('cross_tome_findings', []))} laws",
            "- [[Gaps]]: what to measure next", "", "## Behaviors"] + [f"- [[{b}]]" for b in BEHAVIORS] + [
            "", "## Methods"] + [f"- [[{slug(t)}]]" for t in METHODS]
    (out / "Home.md").write_text("\n".join(home) + "\n", encoding="utf-8")
    col = lambda h: {"a": 1, "rgb": int(h, 16)}  # noqa: E731
    graph = {"colorGroups": [{"query": "tag:#finding/negative", "color": col("8a8a8a")},
                             {"query": "tag:#model", "color": col("e5484d")},
                             {"query": "tag:#finding", "color": col("30a46c")},
                             {"query": "tag:#behavior", "color": col("f76b15")},
                             {"query": "tag:#method", "color": col("0090ff")},
                             {"query": "tag:#transition", "color": col("8e4ec6")},
                             {"query": "tag:#treatment", "color": col("ffc53d")},
                             {"query": "tag:#law", "color": col("12a594")}],
             "showTags": False, "showAttachments": False, "hideUnresolved": True}
    gp = out / ".obsidian" / "graph.json"
    if not gp.exists():
        gp.write_text(json.dumps(graph, indent=1), encoding="utf-8")
    nlinks = sum(len(f["models"]) + len(f["behaviors"]) + len(f["methods"]) for f in findings)
    print(f"vault {out}: {len(models)} models, {len(findings)} findings, {len(t2)} transitions, "
          f"{sum(len(v) for v in t3['models'].values())} treatments, {nlinks} finding links")


if __name__ == "__main__":
    main()
