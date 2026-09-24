"""Scatters of 2025 offenses on designed non-QB runs (same play filter as
build_run_features.py), with marks colored by the run direction x formation
cluster from build_run_formation_scatter.py, to compare other features
against that identity.

Charts (see CHARTS):
  - personnel: avg backs (RB + FB) vs avg TEs per run
  - te_motion: avg TEs minus avg backs vs pre-snap motion rate on runs

Output: output/<chart>_scatter_2025.html
"""
import csv
import json

from build_run_formation_scatter import CLUSTER_NAMES

SEASON = "2025"
FEATURES_PATH = "data/run_play_features.csv"
CLUSTERS_PATH = "output/run_formation_clusters_2025.csv"

# fmt keys (rendered in the page): dec = 2dp, sdec = signed 2dp, pct = x100 with %
PLAY_FILTER = ("Designed non-QB runs only (no scrambles, QB runs, kneels or 2-pt tries), "
               "neutral game script (WP 20-80%, outside last 2 min of each half). "
               "Color and shape = run direction x formation cluster.")
TIP_COMMON = [["Designed runs", "run_n", "int"]]
CHARTS = {
    "personnel": {
        "page_title": "Run Personnel 2025",
        "title": "Run personnel, 2025",
        "lead": "Average backs and tight ends on the field per run.",
        "x": "avg_rb", "y": "avg_te", "x_fmt": "dec", "y_fmt": "dec", "x_step": 0.1, "y_step": 0.1,
        "x_title": "Avg backs (RB + FB) per run",
        "y_title": "Avg TEs per run",
        "quads": ["Fewer backs / more TEs", "More backs / more TEs", "Fewer backs / fewer TEs", "More backs / fewer TEs"],
        "note": ("X: average RB + FB per run (a 21-personnel snap counts 2). Y: average TE per run. Positions are "
                 "roster positions, so an extra offensive lineman reporting eligible is not counted as a TE, and a "
                 "TE lined up at fullback counts as a TE. Dashed lines are league averages."),
        "fields": [["Avg backs", "avg_rb", "dec"], ["Avg FB", "avg_fb", "dec"], ["Avg TEs", "avg_te", "dec"]],
    },
    "te_motion": {
        "page_title": "TE Tilt vs Motion",
        "title": "TE-vs-back tilt vs motion rate on runs, 2025",
        "lead": "Whether the extra blocker is a TE or a back, against how often runs use pre-snap motion.",
        "x": "te_minus_backs", "y": "motion_rate", "x_fmt": "sdec", "y_fmt": "pct", "x_step": 0.2, "y_step": 0.05,
        "x_title": "Avg TEs minus avg backs per run",
        "y_title": "Runs with pre-snap motion",
        "quads": ["Back-heavy / more motion", "TE-heavy / more motion", "Back-heavy / less motion", "TE-heavy / less motion"],
        "note": ("X: average TEs minus average backs (RB + FB) per run; negative = fullback-leaning, positive = "
                 "TE-leaning. Equals the first principal component of the two (83% of their variance). "
                 "Y: share of runs with pre-snap motion (FTN charting, all runs charted). Dashed lines are league averages."),
        "fields": [["TE - backs", "te_minus_backs", "sdec"], ["Avg TEs", "avg_te", "dec"],
                   ["Avg backs", "avg_rb", "dec"], ["Motion", "motion_rate", "pct"]],
    },
}


def build(name, cfg, rows, cluster_of, order):
    fields = {cfg["x"], cfg["y"], "run_n"} | {f for _, f, _ in cfg["fields"]}
    points = [{
        "team": r["team"],
        "cluster": order[cluster_of[r["team"]]],
        "cluster_name": cluster_of[r["team"]],
        **{f: round(float(r[f]), 4) for f in fields},
    } for r in rows]
    for p in points:
        p["x"], p["y"] = p[cfg["x"]], p[cfg["y"]]

    config = {**cfg, "sub": f'{cfg["lead"]} {PLAY_FILTER}', "tip": cfg["fields"] + TIP_COMMON}
    payload = {"points": points, "clusters": [n for _, n in CLUSTER_NAMES], "config": config}
    out_path = f"output/{name}_scatter_{SEASON}.html"
    html = TEMPLATE.replace("__PAGE_TITLE__", cfg["page_title"]).replace("__DATA_JSON__", json.dumps(payload))
    with open(out_path, "w") as out:
        out.write(html)
    print(f"wrote {out_path} ({len(points)} teams)")


def main():
    with open(CLUSTERS_PATH) as f:
        cluster_of = {r["team"]: r["cluster_name"] for r in csv.DictReader(f)}
    order = {name: i for i, (_, name) in enumerate(CLUSTER_NAMES)}
    with open(FEATURES_PATH) as f:
        rows = [r for r in csv.DictReader(f) if r["season"] == SEASON]
    for name, cfg in CHARTS.items():
        build(name, cfg, rows, cluster_of, order)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
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
  <h1 id="title"></h1>
  <p class="sub" id="sub"></p>
  <div class="legend" id="legend"></div>
  <svg id="chart" viewBox="0 0 900 620" role="img"></svg>
  <p class="note" id="note"></p>
  <details>
    <summary>Table view</summary>
    <table id="tbl"></table>
  </details>
</main>
<div id="tip"></div>
<script>
const payload = __DATA_JSON__;
const data = payload.points, names = payload.clusters, cfg = payload.config;
document.getElementById("title").textContent = cfg.title;
document.getElementById("sub").textContent = cfg.sub;
document.getElementById("note").textContent = cfg.note;
document.getElementById("chart").setAttribute("aria-label", `Scatter of 32 NFL offenses: ${cfg.x_title} vs ${cfg.y_title}`);
const FMT = {
  dec: v => v.toFixed(2),
  sdec: v => (v > 0 ? "+" : "") + v.toFixed(2),
  pct: v => (v * 100).toFixed(1) + "%",
  int: v => String(v),
};
const TICK = {dec: v => v.toFixed(1), sdec: v => (v > 1e-9 ? "+" : "") + v.toFixed(1), pct: v => Math.round(v * 100) + "%"};
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
const ext = (k, step) => {
  const v = data.map(d => d[k]);
  return [Math.floor(Math.min(...v) / step - 0.5) * step, Math.ceil(Math.max(...v) / step + 0.5) * step];
};
const [x0, x1] = ext("x", cfg.x_step), [y0, y1] = ext("y", cfg.y_step);
const sx = v => m.l + (v - x0) / (x1 - x0) * pw;
const sy = v => m.t + ph - (v - y0) / (y1 - y0) * ph;
const mean = k => data.reduce((s, d) => s + d[k], 0) / data.length;
const mx = mean("x"), my = mean("y");

const g = el("g", {class: "grid"});
for (let i = 0; x0 + i * cfg.x_step <= x1 + 1e-9; i++) {
  const v = x0 + i * cfg.x_step;
  el("line", {x1: sx(v), x2: sx(v), y1: m.t, y2: m.t + ph}, g);
  el("text", {x: sx(v), y: m.t + ph + 18, "text-anchor": "middle", class: "tick"}, svg, TICK[cfg.x_fmt](v));
}
for (let i = 0; y0 + i * cfg.y_step <= y1 + 1e-9; i++) {
  const v = y0 + i * cfg.y_step;
  el("line", {x1: m.l, x2: m.l + pw, y1: sy(v), y2: sy(v)}, g);
  el("text", {x: m.l - 8, y: sy(v) + 4, "text-anchor": "end", class: "tick"}, svg, TICK[cfg.y_fmt](v));
}
el("line", {x1: sx(mx), x2: sx(mx), y1: m.t, y2: m.t + ph, class: "zero"});
el("line", {x1: m.l, x2: m.l + pw, y1: sy(my), y2: sy(my), class: "zero"});
el("text", {x: m.l + pw / 2, y: H - 12, "text-anchor": "middle", class: "axis-title"}, svg, cfg.x_title);
el("text", {transform: `translate(16 ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle", class: "axis-title"}, svg, cfg.y_title);
el("text", {x: m.l + 8, y: m.t + 14, class: "quad"}, svg, cfg.quads[0]);
el("text", {x: m.l + pw - 8, y: m.t + 14, "text-anchor": "end", class: "quad"}, svg, cfg.quads[1]);
el("text", {x: m.l + 8, y: m.t + ph - 8, class: "quad"}, svg, cfg.quads[2]);
el("text", {x: m.l + pw - 8, y: m.t + ph - 8, "text-anchor": "end", class: "quad"}, svg, cfg.quads[3]);

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
    tip.innerHTML = `<b>${d.team}</b> <span>${d.cluster_name}</span>` +
      cfg.tip.map(([label, f, fmt]) => `<br><span>${label}:</span> ${FMT[fmt](d[f])}`).join("");
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
  "<tr><th>Team</th><th>Cluster</th>" + cfg.tip.map(([label]) => `<th>${label}</th>`).join("") + "</tr>" +
  rows.map(d => `<tr><td>${d.team}</td><td>${d.cluster_name}</td>` +
    cfg.tip.map(([, f, fmt]) => `<td>${FMT[fmt](d[f])}</td>`).join("") + "</tr>").join("");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
