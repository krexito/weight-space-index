"""Build a self-contained interactive HTML WSI Observatory from the 3-Tome Atlas ($0 GPU, 100% CPU).

Usage:
    python scripts/build_observatory.py --atlas benchmarks/xray_atlas --out exports/observatory/index.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsi.observatory import DEFAULT_ATLAS_DIR, WSIObservatory  # noqa: E402

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WSI Observatory — 3-Tome Weight-Space Atlas</title>
<style>
:root{color-scheme:dark;--bg:#0e1015;--panel:#161922;--card:#1d212e;--fg:#eef1f7;--muted:#8d96aa;--line:rgba(255,255,255,.09);--accent:#4f8cff;--pass:#2ecc71;--partial:#f5a623;--fail:#ff5c5c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:10;background:rgba(14,16,21,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:12px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header h1{margin:0;font-size:16px;font-weight:700;letter-spacing:.2px}
.pill{font-size:12px;padding:3px 9px;border-radius:999px;background:var(--card);border:1px solid var(--line);color:var(--muted)}
.tabs{display:flex;gap:6px;margin-left:auto}
.tabs button{background:var(--card);color:var(--muted);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;font-weight:600;font-size:13px}
.tabs button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:1400px;margin:0 auto;padding:18px 20px 60px}
.panel{display:none}.panel.active{display:block}
.filters{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
input,select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.stat .v{font-size:22px;font-weight:700;margin-top:4px}
.tbl-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:var(--card);color:var(--muted);cursor:pointer;user-select:none}
tr:hover td{background:rgba(79,140,255,.08)}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase}
.b-pass{background:rgba(46,204,113,.16);color:var(--pass)}
.b-partial{background:rgba(245,166,35,.16);color:var(--partial)}
.b-fail{background:rgba(255,92,92,.16);color:var(--fail)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}
.card h3{margin:0 0 8px;font-size:14px}
.bar{height:8px;border-radius:4px;background:rgba(255,255,255,.08);overflow:hidden;min-width:80px;display:inline-block;vertical-align:middle}
.bar>span{display:block;height:100%;background:var(--accent)}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;font-size:12px;white-space:pre-wrap}
.muted{color:var(--muted)}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.two-col{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>WSI Observatory</h1>
  <span class="pill" id="meta-pill"></span>
  <div class="tabs">
    <button class="active" data-tab="models">Tome 1 · Models</button>
    <button data-tab="diffs">Tome 2 · Diffs</button>
    <button data-tab="clinical">Tome 3 · Works vs Fails</button>
    <button data-tab="prescribe">Agent Prescribe</button>
  </div>
</header>
<main>
  <section id="tab-models" class="panel active">
    <div class="grid" id="summary-cards"></div>
    <div class="filters">
      <input id="m-search" placeholder="Search model repo or arch..." style="min-width:240px">
      <select id="m-arch"><option value="">All architectures</option></select>
      <select id="m-norm"><option value="">All norms</option></select>
      <select id="m-quant"><option value="">All quants</option></select>
    </div>
    <div class="tbl-wrap"><table id="m-table"><thead><tr>
      <th data-k="repo_id">Model</th>
      <th data-k="arch_family">Arch</th>
      <th data-k="quant_tag">Quant</th>
      <th data-k="params_b">Params (B)</th>
      <th data-k="n_layers">L</th>
      <th data-k="cal_nll">NLL</th>
      <th data-k="massive_act_max">Massive Act</th>
      <th data-k="fisher_late_early_ratio">Fisher L/E</th>
      <th data-k="base_overref_rate">Over-Ref</th>
      <th data-k="base_toxic_ref_rate">Toxic Ref</th>
      <th data-k="cave_rate_false_pushback">Syc Cave</th>
    </tr></thead><tbody></tbody></table></div>
  </section>

  <section id="tab-diffs" class="panel">
    <div class="filters">
      <select id="d-type"><option value="">All transition types</option></select>
    </div>
    <div class="tbl-wrap"><table id="d-table"><thead><tr>
      <th>Pair</th><th>Type</th><th>Arch</th><th>Params</th><th>Rel Fro</th><th>Depth</th><th>MLP Share</th><th>Mixer Share</th><th>eRank MLP</th><th>KL Prose</th><th>ΔOverRef</th><th>ΔSyc</th>
    </tr></thead><tbody></tbody></table></div>
  </section>

  <section id="tab-clinical" class="panel">
    <div class="filters">
      <select id="c-path"><option value="">All pathologies</option></select>
      <select id="c-out">
        <option value="">All outcomes</option>
        <option value="pass">PASS only</option>
        <option value="partial">PARTIAL only</option>
        <option value="fail">FAIL only</option>
      </select>
    </div>
    <div id="c-scorecard" class="grid"></div>
    <div id="c-list"></div>
  </section>

  <section id="tab-prescribe" class="panel">
    <div class="card">
      <h3>1-Call Agent Prescription Simulator (<code>wsi_prescribe</code>)</h3>
      <p class="muted">Select any scanned model and pathology to inspect the exact compact brief returned to an AI agent (nearest neighbors, proven strategies, failure blacklist, and next 2-second probe).</p>
      <div class="filters">
        <select id="p-model" style="min-width:300px"></select>
        <select id="p-path">
          <option value="over_refusal">over_refusal</option>
          <option value="sycophancy">sycophancy</option>
          <option value="loops">loops</option>
          <option value="pruning">pruning</option>
          <option value="memory_injection">memory_injection</option>
        </select>
      </div>
      <div class="two-col">
        <div class="card"><h3>What Works &amp; Blacklist</h3><div id="p-visual"></div></div>
        <div class="card"><h3>Raw MCP JSON Payload</h3><pre id="p-json"></pre></div>
      </div>
    </div>
  </section>
</main>
<script>
const DATA = __OBS_DATA__;
const fmt = (v, d=2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "—" : Number(v).toFixed(d);
const pct = v => (v === null || v === undefined || Number.isNaN(Number(v))) ? "—" : (Number(v)*100).toFixed(1) + "%";

document.getElementById("meta-pill").textContent =
  `${DATA.evidence.counts.tome1_models} Models · ${DATA.evidence.counts.tome2_pairs} Diffs · ${DATA.evidence.counts.tome3_interventions} Clinical (${DATA.evidence.counts.tome3_pass} pass / ${DATA.evidence.counts.tome3_fail} fail)`;

// Tabs
document.querySelectorAll(".tabs button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tabs button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  };
});

// Summary cards
const c = DATA.evidence.counts;
document.getElementById("summary-cards").innerHTML = [
  ["Tome 1 Models", c.tome1_models],
  ["Tome 2 Delta Diffs", c.tome2_pairs],
  ["Tome 3 Interventions", c.tome3_interventions],
  ["Clinical Pass Rate", pct(c.tome3_pass / Math.max(c.tome3_interventions, 1))],
].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

// Populate filter dropdowns
const uniq = arr => [...new Set(arr.filter(Boolean))].sort();
const fillSel = (id, vals) => {
  const s = document.getElementById(id);
  vals.forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = v; s.appendChild(o); });
};
fillSel("m-arch", uniq(DATA.models.map(m => m.arch_family)));
fillSel("m-norm", uniq(DATA.models.map(m => m.norm_type)));
fillSel("m-quant", uniq(DATA.models.map(m => m.quant_tag)));
fillSel("d-type", uniq(DATA.pairs.map(p => p.transition_type)));
fillSel("c-path", uniq(DATA.interventions.map(i => i.pathology)));
fillSel("p-model", DATA.models.map(m => m.repo_id).sort());

// Tome 1 table
let mSort = {k: "params_b", desc: true};
function renderModels() {
  const q = document.getElementById("m-search").value.toLowerCase();
  const fa = document.getElementById("m-arch").value;
  const fn = document.getElementById("m-norm").value;
  const fq = document.getElementById("m-quant").value;
  let rows = DATA.models.filter(m =>
    (!q || (m.repo_id||"").toLowerCase().includes(q) || (m.arch_family||"").toLowerCase().includes(q)) &&
    (!fa || m.arch_family === fa) && (!fn || m.norm_type === fn) && (!fq || m.quant_tag === fq)
  );
  rows.sort((a, b) => {
    const av = a[mSort.k], bv = b[mSort.k];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return (typeof av === "string" ? av.localeCompare(bv) : av - bv) * (mSort.desc ? -1 : 1);
  });
  document.querySelector("#m-table tbody").innerHTML = rows.map(m => `<tr>
    <td><strong>${m.repo_id}</strong></td>
    <td>${m.arch_family||"—"}</td>
    <td><span class="pill">${m.quant_tag||"—"}</span></td>
    <td>${fmt(m.params_b, 2)}</td>
    <td>${m.n_layers||"—"}</td>
    <td>${fmt(m.cal_nll, 2)}</td>
    <td>${fmt(m.massive_act_max, 1)}×</td>
    <td>${fmt(m.fisher_late_early_ratio, 3)}</td>
    <td>${pct(m.base_overref_rate)}</td>
    <td>${pct(m.base_toxic_ref_rate)}</td>
    <td>${pct(m.cave_rate_false_pushback)}</td>
  </tr>`).join("");
}
document.querySelectorAll("#m-table th").forEach(th => {
  th.onclick = () => {
    const k = th.dataset.k;
    mSort = {k, desc: mSort.k === k ? !mSort.desc : true};
    renderModels();
  };
});
["m-search","m-arch","m-norm","m-quant"].forEach(id => document.getElementById(id).oninput = renderModels);
renderModels();

// Tome 2 table
function renderDiffs() {
  const ft = document.getElementById("d-type").value;
  const rows = DATA.pairs.filter(p => !ft || p.transition_type === ft);
  document.querySelector("#d-table tbody").innerHTML = rows.map(p => `<tr>
    <td><strong>${p.pair_id}</strong><div class="muted" style="font-size:11px">${p.before_repo} → ${p.after_repo}</div></td>
    <td><span class="pill">${p.transition_type}</span></td>
    <td>${p.arch_family||"—"}</td>
    <td>${fmt(p.params_b, 2)}B</td>
    <td>${fmt(p.total_rel_fro, 4)}</td>
    <td>${fmt(p.depth_centroid, 2)}</td>
    <td>${pct(p.mlp_share)} <span class="bar"><span style="width:${Math.min(100,(p.mlp_share||0)*100)}%"></span></span></td>
    <td>${pct(p.mixer_share)}</td>
    <td>${fmt(p.erank_ratio_mlp, 2)}</td>
    <td>${fmt(p.kl_prose, 4)}</td>
    <td>${pct(p.delta_overref_rate)}</td>
    <td>${pct(p.delta_cave_rate)}</td>
  </tr>`).join("");
}
document.getElementById("d-type").onchange = renderDiffs;
renderDiffs();

// Tome 3 clinical
function renderClinical() {
  const fp = document.getElementById("c-path").value;
  const fo = document.getElementById("c-out").value;
  const rows = DATA.interventions.filter(r => (!fp || r.pathology === fp) && (!fo || r.outcome === fo));
  const sc = DATA.evidence.clinical_scorecard.filter(s => !fp || s.pathology === fp);
  document.getElementById("c-scorecard").innerHTML = sc.slice(0, 8).map(s => `
    <div class="stat">
      <div class="k">${s.pathology}</div>
      <div style="font-weight:700;margin:4px 0">${s.strategy}</div>
      <div><span class="badge b-pass">${s.pass} pass</span> <span class="badge b-partial">${s.partial} partial</span> <span class="badge b-fail">${s.fail} fail</span></div>
    </div>`).join("");
  document.getElementById("c-list").innerHTML = rows.map(r => `
    <div class="card">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <span class="badge b-${r.outcome}">${r.outcome}</span>
        <strong>${r.strategy}</strong>
        <span class="pill">${r.pathology}</span>
        <span class="muted">${r.repo_id}</span>
      </div>
      <div style="margin-top:6px;font-size:12.5px"><strong>Metrics:</strong> <code>${JSON.stringify(r.metrics||{})}</code></div>
      ${r.caveat ? `<div class="muted" style="margin-top:4px;font-size:12.5px"><strong>Caveat:</strong> ${r.caveat}</div>` : ""}
    </div>`).join("");
}
["c-path","c-out"].forEach(id => document.getElementById(id).onchange = renderClinical);
renderClinical();

// Prescribe simulator (precomputed per model x pathology or dynamic lookup)
function renderPrescribe() {
  const repo = document.getElementById("p-model").value;
  const patho = document.getElementById("p-path").value;
  const p = (DATA.prescriptions[repo] || {})[patho];
  if (!p) return;
  document.getElementById("p-json").textContent = JSON.stringify(p, null, 2);
  const worksHtml = (p.proven_strategies_pass || []).map(w =>
    `<li><span class="badge b-pass">PASS</span> <strong>${w.strategy}</strong> (${w.model}) — <span class="muted">${w.caveat||""}</span></li>`
  ).join("") || "<li class='muted'>No exact PASS record for this pathology yet.</li>";
  const failHtml = (p.do_not_do_blacklist || []).map(f =>
    `<li><span class="badge b-fail">AVOID</span> <strong>${f.avoid_strategy}</strong> (${f.model}) — <span class="muted">${f.why_failed||""}</span></li>`
  ).join("") || "<li class='muted'>No failed records for this pathology.</li>";
  const nbHtml = (p.nearest_neighbors || []).map(n =>
    `<li><strong>${n.repo_id}</strong> (${n.arch_family}, ${fmt(n.params_b,2)}B, sim=${n.similarity})</li>`
  ).join("");
  document.getElementById("p-visual").innerHTML = `
    <div class="muted" style="margin-bottom:8px">Candidate band: <strong>${p.recommended_candidate_band}</strong></div>
    <h4>Nearest Neighbors</h4><ul>${nbHtml}</ul>
    <h4>Proven Strategies</h4><ul>${worksHtml}</ul>
    <h4>Do-Not-Do Blacklist</h4><ul>${failHtml}</ul>
    <h4>Next 2-Second Probe</h4><pre>${JSON.stringify(p.next_2s_probe, null, 2)}</pre>
  `;
}
["p-model","p-path"].forEach(id => document.getElementById(id).onchange = renderPrescribe);
renderPrescribe();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=str(DEFAULT_ATLAS_DIR))
    ap.add_argument("--out", default=str(ROOT / "exports" / "observatory" / "index.html"))
    a = ap.parse_args()

    obs = WSIObservatory(a.atlas)
    evidence = obs.summarize_evidence()
    pathologies = ("over_refusal", "sycophancy", "loops", "pruning", "memory_injection")
    prescriptions = {
        m["repo_id"]: {p: obs.prescribe(m["repo_id"], pathology=p) for p in pathologies}
        for m in obs.models
        if m.get("repo_id")
    }
    payload = {
        "models": obs.models,
        "pairs": obs.pairs,
        "interventions": obs.interventions,
        "evidence": evidence,
        "prescriptions": prescriptions,
    }
    html = HTML_TEMPLATE.replace("__OBS_DATA__", json.dumps(payload, ensure_ascii=False))
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html, encoding="utf-8")
    print(f"Built WSI Observatory -> {out_p} ({out_p.stat().st_size / 1024:.1f} KB)")
    print("Counts:", evidence["counts"])


if __name__ == "__main__":
    main()
