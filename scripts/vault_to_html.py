"""Render the WSI vault (scripts/build_wsi_vault.py) as one self-contained interactive graph page.

  python scripts/vault_to_html.py --vault wsi_vault --out wsi_vault/map.html
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

COLORS = {"Models": "#e5484d", "Findings": "#30a46c", "Behaviors": "#f76b15", "Methods": "#0090ff",
          "Transitions": "#8e4ec6", "Treatments": "#d4a017", "Laws": "#12a594"}
LABELS = {"Models": "Modelos", "Findings": "Hallazgos", "Behaviors": "Comportamientos", "Methods": "Métodos",
          "Transitions": "Transiciones", "Treatments": "Curas", "Laws": "Leyes"}

PAGE = """<title>Mapa WSI</title>
<style>
:root{color-scheme:light;--bg:#fbfaf8;--fg:#1d1d1f;--muted:#6b6b70;--line:rgba(0,0,0,.10);--card:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141416;--fg:#ececef;--muted:#9a9aa2;--line:rgba(255,255,255,.10);--card:#1d1d21;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141416;--fg:#ececef;--muted:#9a9aa2;--line:rgba(255,255,255,.10);--card:#1d1d21;color-scheme:dark}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
#top{position:fixed;top:0;left:0;right:0;padding:calc(10px + env(safe-area-inset-top,0px)) 16px 10px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;z-index:2;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(6px);border-bottom:1px solid var(--line)}
#top h1{font-size:15px;margin:0 8px 0 0}
#q{flex:1;min-width:140px;max-width:280px;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
.lg{display:flex;gap:6px;flex-wrap:wrap}
.lg button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:3px 9px;font-size:12px;cursor:pointer}
.lg button.off{opacity:.35}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
svg{width:100%;height:100%;display:block}
#info{position:fixed;right:16px;bottom:calc(16px + env(safe-area-inset-bottom,0px));width:min(380px,calc(100% - 32px));max-height:45vh;overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;display:none;box-shadow:0 6px 24px rgba(0,0,0,.12)}
#info h2{font-size:14px;margin:0 0 6px}#info .k{color:var(--muted);font-size:12px}#info p{margin:6px 0;white-space:pre-wrap;font-size:12.5px}
#info a{color:inherit;text-decoration:underline;cursor:pointer}
text{fill:var(--fg);font-size:10px;pointer-events:none;paint-order:stroke;stroke:var(--bg);stroke-width:3px}
line{stroke:var(--line)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div id="top"><h1>Mapa WSI</h1><input id="q" placeholder="Buscar nodo…"><div class="lg" id="lg"></div></div>
<svg id="g"></svg><div id="info"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const DATA=__DATA__;const COLORS=__COLORS__;const LABELS=__LABELS__;
const svg=d3.select('#g'),W=()=>innerWidth,H=()=>innerHeight;const root=svg.append('g');
svg.call(d3.zoom().scaleExtent([.15,6]).on('zoom',e=>root.attr('transform',e.transform)));
const hidden=new Set();const deg={};DATA.links.forEach(l=>{deg[l.source]=(deg[l.source]||0)+1;deg[l.target]=(deg[l.target]||0)+1});
const r=d=>3+Math.sqrt(deg[d.id]||1)*1.6;
const sim=d3.forceSimulation(DATA.nodes).force('link',d3.forceLink(DATA.links).id(d=>d.id).distance(40).strength(.25))
 .force('charge',d3.forceManyBody().strength(-60)).force('center',d3.forceCenter(W()/2,H()/2)).force('col',d3.forceCollide().radius(d=>r(d)+2));
const link=root.append('g').selectAll('line').data(DATA.links).join('line').attr('stroke-width',.7);
const node=root.append('g').selectAll('circle').data(DATA.nodes).join('circle').attr('r',r).attr('fill',d=>COLORS[d.group])
 .attr('stroke','var(--bg)').attr('stroke-width',1).style('cursor','pointer').on('click',(e,d)=>show(d))
 .call(d3.drag().on('start',(e,d)=>{if(!e.active)sim.alphaTarget(.2).restart();d.fx=d.x;d.fy=d.y}).on('drag',(e,d)=>{d.fx=e.x;d.fy=e.y}).on('end',(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}));
const label=root.append('g').selectAll('text').data(DATA.nodes.filter(d=>d.group!=='Findings'||(deg[d.id]||0)>9)).join('text').text(d=>d.label.length>34?d.label.slice(0,32)+'…':d.label).attr('dx',7).attr('dy',3);
sim.on('tick',()=>{link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);node.attr('cx',d=>d.x).attr('cy',d=>d.y);label.attr('x',d=>d.x).attr('y',d=>d.y)});
const nb={};DATA.links.forEach(l=>{const s=l.source.id||l.source,t=l.target.id||l.target;(nb[s]=nb[s]||new Set()).add(t);(nb[t]=nb[t]||new Set()).add(s)});
function focus(id){const n=nb[id]||new Set();node.attr('opacity',d=>!id||d.id===id||n.has(d.id)?1:.12);link.attr('opacity',l=>!id||l.source.id===id||l.target.id===id?1:.05);label.attr('opacity',d=>!id||d.id===id||n.has(d.id)?1:.1)}
function show(d){focus(d.id);const n=[...(nb[d.id]||[])].map(i=>DATA.nodes.find(x=>x.id===i)).sort((a,b)=>a.group.localeCompare(b.group));
 const box=document.getElementById('info');box.style.display='block';
 box.innerHTML=`<h2>${d.label}</h2><div class="k">${LABELS[d.group]} · ${n.length} conexiones · <a onclick="focus(null);this.closest('#info').style.display='none'">cerrar</a></div><p>${d.summary.replace(/</g,'&lt;')}</p><div class="k">Conectado con:</div>`+
 n.slice(0,60).map(x=>`<div><span class="dot" style="background:${COLORS[x.group]}"></span><a data-id="${encodeURIComponent(x.id)}">${x.label}</a></div>`).join('');
 box.querySelectorAll('a[data-id]').forEach(a=>a.onclick=()=>show(DATA.nodes.find(x=>x.id===decodeURIComponent(a.dataset.id))))}
const lg=document.getElementById('lg');Object.keys(COLORS).forEach(g=>{const b=document.createElement('button');const c=DATA.nodes.filter(d=>d.group===g).length;
 b.innerHTML=`<span class="dot" style="background:${COLORS[g]}"></span>${LABELS[g]} ${c}`;b.onclick=()=>{hidden.has(g)?hidden.delete(g):hidden.add(g);b.classList.toggle('off');
 node.style('display',d=>hidden.has(d.group)?'none':null);label.style('display',d=>hidden.has(d.group)?'none':null);link.style('display',l=>hidden.has(l.source.group)||hidden.has(l.target.group)?'none':null)};lg.appendChild(b)});
document.getElementById('q').oninput=e=>{const q=e.target.value.toLowerCase();if(!q){focus(null);return}const m=DATA.nodes.find(d=>d.label.toLowerCase().includes(q));if(m)show(m)};
</script>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default="wsi_vault")
    ap.add_argument("--out", default="wsi_vault/map.html")
    a = ap.parse_args()
    v = Path(a.vault)
    nodes, links, ids = [], set(), {}
    for grp in COLORS:
        for p in sorted((v / grp).glob("*.md")):
            ids[p.stem] = grp
    for grp in COLORS:
        for p in sorted((v / grp).glob("*.md")):
            t = p.read_text(encoding="utf-8")
            body = re.sub(r"^---.*?---\s*", "", t, flags=re.S)
            title = re.search(r"^# (.+)$", body, re.M)
            summ = re.sub(r"\[\[([^\]|]+)(\|([^\]]+))?\]\]", lambda m: m.group(3) or m.group(1), body)
            summ = re.sub(r"^# .*\n", "", summ).strip()
            summ = re.sub(r"\n\|.*", "", summ)[:700]
            nodes.append({"id": p.stem, "label": title.group(1) if title else p.stem, "group": grp, "summary": summ})
            for l in re.findall(r"\[\[([^\]|]+)", t):
                if l in ids and l != p.stem:
                    links.add(tuple(sorted((p.stem, l))))
    data = {"nodes": nodes, "links": [{"source": s, "target": t} for s, t in sorted(links)]}
    html = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__COLORS__", json.dumps(COLORS)) \
               .replace("__LABELS__", json.dumps(LABELS, ensure_ascii=False))
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"{a.out}: {len(nodes)} nodes, {len(links)} links, {len(html) // 1024} KB")


if __name__ == "__main__":
    main()
