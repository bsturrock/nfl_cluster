"""Scatter of 2025 offenses: avg backs (RB + FB) vs avg TEs on the field,
designed non-QB runs only (same play filter as build_run_features.py).
Marks are colored by the run direction x formation cluster from
build_run_formation_scatter.py, to compare personnel against that identity.

Output: output/personnel_scatter_2025.html
"""
import csv
import json

from build_run_formation_scatter import CLUSTER_NAMES

SEASON = "2025"
FEATURES_PATH = "data/run_play_features.csv"
CLUSTERS_PATH = "output/run_formation_clusters_2025.csv"
OUT_PATH = "output/personnel_scatter_2025.html"


def main():
    with open(CLUSTERS_PATH) as f:
        cluster_of = {r["team"]: r["cluster_name"] for r in csv.DictReader(f)}
    order = {name: i for i, (_, name) in enumerate(CLUSTER_NAMES)}

    points = []
    with open(FEATURES_PATH) as f:
        for r in csv.DictReader(f):
            if r["season"] != SEASON:
                continue
            points.append({
                "team": r["team"],
                "x": round(float(r["avg_rb"]), 3),
                "y": round(float(r["avg_te"]), 3),
                "fb": round(float(r["avg_fb"]), 3),
                "run_n": int(r["run_n"]),
                "cluster": order[cluster_of[r["team"]]],
                "cluster_name": cluster_of[r["team"]],
            })

    payload = {"points": points, "clusters": [name for _, name in CLUSTER_NAMES]}
    with open(OUT_PATH, "w") as out:
        out.write(TEMPLATE.replace("__DATA_JSON__", json.dumps(payload)))
    print(f"wrote {OUT_PATH} ({len(points)} teams)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Personnel 2025</title>
<style>
:root {
  color-scheme: light;
  --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a8984;
  --grid: #e6e5e1; --axis: #b9b8b2;
  --c0: #2a78d6; --c1: #eb6834; --c2: #1baf7a; --c3: #eda100; --c4: #e87ba4; --c5: #008300; --c6: #4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e87;
    --grid: #2e2e2c; --axis: #55544f;
    --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300; --c6: #9085e9;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e87;
  --grid: #2e2e2c; --axis: #55544f;
  --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300; --c6: #9085e9;
}
body { margin: 0; background: var(--surface-1); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 40px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
.grid line { stroke: var(--grid); }
.zero { stroke: var(--axis); stroke-dasharray: 4 4; }
.tick { fill: var(--text-muted); font-size: 11px; }
.axis-title { fill: var(--text-secondary); font-size: 12px; }
.quad { fill: var(--text-muted); font-size: 11px; font-style: italic; }
.dot { stroke: var(--surface-1); stroke-width: 2; }
.dot.on { stroke: var(--text-primary); stroke-width: 1.5; }
.lbl { fill: var(--text-primary); font-size: 11px; font-weight: 600; pointer-events: none; }
.hit { fill: transparent; }
.leader { stroke: var(--text-muted); stroke-width: 1; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 8px; font-size: 12px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 12px; height: 12px; overflow: visible; }
#tip { position: fixed; pointer-events: none; background: var(--surface-1);
  border: 1px solid var(--axis); border-radius: 6px; padding: 8px 10px;
  font-size: 12px; color: var(--text-primary); display: none;
  box-shadow: 0 2px 8px rgba(0,0,0,.15); }
#tip b { font-size: 13px; }
#tip span { color: var(--text-secondary); }
details { margin-top: 20px; }
table { border-collapse: collapse; font-size: 12px; margin-top: 8px; width: 100%; }
th, td { padding: 4px 8px; text-align: right; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.note { color: var(--text-muted); font-size: 12px; margin-top: 12px; }
</style>
</head>
<body>
<main>
  <h1>Run personnel, 2025</h1>
  <p class="sub">Average backs and tight ends on the field per run. Designed non-QB runs only (no scrambles, QB runs, kneels or 2-pt tries), neutral game script (WP 20-80%, outside last 2 min of each half). Color and shape = run direction x formation cluster.</p>
  <div class="legend" id="legend"></div>
  <svg id="chart" viewBox="0 0 900 620" role="img" aria-label="Scatter of 32 NFL offenses by average backs and tight ends on run plays"></svg>
  <p class="note">X: average RB + FB per run (a 21-personnel snap counts 2). Y: average TE per run. Positions are roster positions, so an extra offensive lineman reporting eligible is not counted as a TE, and a TE lined up at fullback counts as a TE. Dashed lines are league averages.</p>
  <details>
    <summary>Table view</summary>
    <table id="tbl"></table>
  </details>
</main>
<div id="tip"></div>
<script>
const payload = __DATA_JSON__;
const data = payload.points, names = payload.clusters;
// color + shape double-encode cluster: 7 hues can't all separate for CVD in a scatter
const shape = (c, r) => {
  const a = r * 1.25;
  return [
    `M${r},0A${r},${r} 0 1,1 ${-r},0A${r},${r} 0 1,1 ${r},0Z`,
    `M${-r*.9},${-r*.9}H${r*.9}V${r*.9}H${-r*.9}Z`,
    `M0,${-a}L${a*.95},${a*.6}H${-a*.95}Z`,
    `M0,${-a}L${a},0L0,${a}L${-a},0Z`,
    `M0,${a}L${a*.95},${-a*.6}H${-a*.95}Z`,
    `M${-r*.35},${-r}H${r*.35}V${-r*.35}H${r}V${r*.35}H${r*.35}V${r}H${-r*.35}V${r*.35}H${-r}V${-r*.35}H${-r*.35}Z`,
    `M${-r},${-r*.55}L${-r*.55},${-r}L0,${-r*.45}L${r*.55},${-r}L${r},${-r*.55}L${r*.45},0L${r},${r*.55}L${r*.55},${r}L0,${r*.45}L${-r*.55},${r}L${-r},${r*.55}L${-r*.45},0Z`,
  ][c];
};
const W = 900, H = 620, m = {t: 20, r: 24, b: 56, l: 64};
const pw = W - m.l - m.r, ph = H - m.t - m.b;
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const el = (tag, attrs, parent = svg, text) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  parent.appendChild(e); return e;
};
const STEP = 0.1;
const ext = k => {
  const v = data.map(d => d[k]);
  return [Math.floor(Math.min(...v) / STEP - 0.5) * STEP, Math.ceil(Math.max(...v) / STEP + 0.5) * STEP];
};
const [x0, x1] = ext("x"), [y0, y1] = ext("y");
const sx = v => m.l + (v - x0) / (x1 - x0) * pw;
const sy = v => m.t + ph - (v - y0) / (y1 - y0) * ph;
const mean = k => data.reduce((s, d) => s + d[k], 0) / data.length;
const mx = mean("x"), my = mean("y");

const g = el("g", {class: "grid"});
for (let i = 0; x0 + i * STEP <= x1 + 1e-9; i++) {
  const v = x0 + i * STEP;
  el("line", {x1: sx(v), x2: sx(v), y1: m.t, y2: m.t + ph}, g);
  el("text", {x: sx(v), y: m.t + ph + 18, "text-anchor": "middle", class: "tick"}, svg, v.toFixed(1));
}
for (let i = 0; y0 + i * STEP <= y1 + 1e-9; i++) {
  const v = y0 + i * STEP;
  el("line", {x1: m.l, x2: m.l + pw, y1: sy(v), y2: sy(v)}, g);
  el("text", {x: m.l - 8, y: sy(v) + 4, "text-anchor": "end", class: "tick"}, svg, v.toFixed(1));
}
el("line", {x1: sx(mx), x2: sx(mx), y1: m.t, y2: m.t + ph, class: "zero"});
el("line", {x1: m.l, x2: m.l + pw, y1: sy(my), y2: sy(my), class: "zero"});
el("text", {x: m.l + pw / 2, y: H - 12, "text-anchor": "middle", class: "axis-title"}, svg, "Avg backs (RB + FB) per run");
el("text", {transform: `translate(16 ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle", class: "axis-title"}, svg, "Avg TEs per run");
el("text", {x: m.l + 8, y: m.t + 14, class: "quad"}, svg, "Fewer backs / more TEs");
el("text", {x: m.l + pw - 8, y: m.t + 14, "text-anchor": "end", class: "quad"}, svg, "More backs / more TEs");
el("text", {x: m.l + 8, y: m.t + ph - 8, class: "quad"}, svg, "Fewer backs / fewer TEs");
el("text", {x: m.l + pw - 8, y: m.t + ph - 8, "text-anchor": "end", class: "quad"}, svg, "More backs / fewer TEs");

const counts = names.map((_, i) => data.filter(d => d.cluster === i).length);
document.getElementById("legend").innerHTML = names.map((n, i) =>
  `<span><svg viewBox="-7 -7 14 14"><path d="${shape(i, 5.5)}" fill="var(--c${i})"/></svg>${n} (${counts[i]})</span>`).join("");

const placed = data.map(d => ({x: sx(d.x) - 6, y: sy(d.y) - 6, w: 12, h: 12}));
const hits = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
const tip = document.getElementById("tip");
data.forEach(d => {
  const cx = sx(d.x), cy = sy(d.y);
  const hit = el("circle", {cx, cy, r: 12, class: "hit"});
  const dot = el("path", {d: shape(d.cluster, 5.5), transform: `translate(${cx} ${cy})`, class: "dot", fill: `var(--c${d.cluster})`});
  hit.addEventListener("mousemove", e => {
    dot.classList.add("on");
    tip.innerHTML = `<b>${d.team}</b> <span>${d.cluster_name}</span><br><span>Avg backs:</span> ${d.x.toFixed(2)} (FB ${d.fb.toFixed(2)})` +
      `<br><span>Avg TEs:</span> ${d.y.toFixed(2)}<br><span>Designed runs:</span> ${d.run_n}`;
    tip.style.display = "block";
    const tx = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = tx + "px"; tip.style.top = (e.clientY + 14) + "px";
  });
  hit.addEventListener("mouseleave", () => { dot.classList.remove("on"); tip.style.display = "none"; });
});
// greedy label placement: try 8 spots around each dot, then the same spots
// pushed further out (with a leader line), skipping ones that hit labels/dots
const spots = [];
for (const k of [1, 2.2, 3.4]) {
  for (const [dx, dy, anchor] of [[8, 4, "start"], [-8, 4, "end"], [0, -9, "middle"], [0, 17, "middle"],
                                   [7, -6, "start"], [-7, -6, "end"], [7, 14, "start"], [-7, 14, "end"]]) {
    spots.push([dx * k, 4 + (dy - 4) * k, anchor, k > 1]);
  }
}
data.forEach(d => {
  const cx = sx(d.x), cy = sy(d.y), w = d.team.length * 7.5, h = 12;
  let best = spots[0];
  for (const s of spots) {
    const bx = s[2] === "start" ? cx + s[0] : s[2] === "end" ? cx + s[0] - w : cx - w / 2;
    const box = {x: bx, y: cy + s[1] - 10, w, h};
    if (bx > m.l && bx + w < m.l + pw && !placed.some(p => hits(p, box))) { best = s; placed.push(box); break; }
  }
  if (best[3]) el("line", {x1: cx, y1: cy, x2: cx + best[0] * 0.8, y2: cy + (best[1] - 4) * 0.8, class: "leader"});
  el("text", {x: cx + best[0], y: cy + best[1], "text-anchor": best[2], class: "lbl"}, svg, d.team);
});

const rows = [...data].sort((a, b) => b.y - a.y);
document.getElementById("tbl").innerHTML =
  "<tr><th>Team</th><th>Cluster</th><th>Avg backs</th><th>Avg FB</th><th>Avg TEs</th><th>Runs</th></tr>" +
  rows.map(d => `<tr><td>${d.team}</td><td>${d.cluster_name}</td><td>${d.x.toFixed(2)}</td><td>${d.fb.toFixed(2)}</td><td>${d.y.toFixed(2)}</td><td>${d.run_n}</td></tr>`).join("");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
