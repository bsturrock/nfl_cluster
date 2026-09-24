"""Scatter of 2025 offenses: outside-minus-inside run share (x) vs
shotgun+pistol snap share (y). Both from the neutral-script feature files.

- outside = end + tackle runs, inside = guard + middle runs (share of rushes)
- shotgun share counts pistol as shotgun; under center = 100 - y

Teams are clustered on just these two dimensions (standardized, KMeans).
k=6 is best on silhouette / Calinski-Harabasz / Davies-Bouldin across
k=2-8, and Ward linkage gives the identical partition.

Output: output/run_formation_scatter_2025.html,
        output/run_formation_clusters_2025.csv
"""
import csv
import json

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

SEASON = "2025"
K = 6
K_RANGE = range(2, 9)
N_BOOT = 200
OUT_PATH = "output/run_formation_scatter_2025.html"
CSV_PATH = "output/run_formation_clusters_2025.csv"

# Clusters are named by a member team rather than by KMeans index, so a
# relabeling between runs can't silently swap names. Order = legend order.
CLUSTER_NAMES = [
    ("DET", "Under center"),
    ("BAL", "Gun, outside lean"),
    ("ATL", "Gun, stretch outside"),
    ("WAS", "Heavy gun, balanced runs"),
    ("CLE", "Gun, inside lean"),
    ("CIN", "Heavy gun, heavy inside"),
]


def load(path, team_col):
    with open(path) as f:
        return {r[team_col]: r for r in csv.DictReader(f) if r["season"] == SEASON}


def main():
    gaps = load("data/team_season_features.csv", "team")
    form = load("data/formation_personnel_features.csv", "posteam")

    points = []
    for team in sorted(gaps):
        g, f = gaps[team], form[team]
        outside = float(g["pct_end"]) + float(g["pct_tackle"])
        inside = float(g["pct_guard"]) + float(g["pct_middle"])
        points.append({
            "team": team,
            "x": round(100 * (outside - inside), 1),
            "y": round(100 * float(f["pct_shotgun_or_pistol"]), 1),
            "outside": round(100 * outside, 1),
            "inside": round(100 * inside, 1),
            "uc": round(100 * float(f["pct_under_center"]), 1),
            "pistol": round(100 * float(f["pct_pistol"]), 1),
            "rush_n": int(g["rush_n"]),
        })

    X = StandardScaler().fit_transform([[p["x"], p["y"]] for p in points])
    sweep = []
    for k in K_RANGE:
        labels = KMeans(k, n_init=50, random_state=0).fit_predict(X)
        sweep.append({
            "k": k,
            "silhouette": round(float(silhouette_score(X, labels)), 3),
            "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
            "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 3),
            "sizes": sorted(np.bincount(labels).tolist(), reverse=True),
        })

    km = KMeans(K, n_init=50, random_state=0).fit(X)
    labels = km.labels_
    ward_ari = adjusted_rand_score(labels, AgglomerativeClustering(K, linkage="ward").fit_predict(X))
    rng = np.random.default_rng(0)
    boot = []
    for i in range(N_BOOT):
        idx = rng.choice(len(X), len(X), replace=True)
        boot.append(adjusted_rand_score(labels, KMeans(K, n_init=10, random_state=i).fit(X[idx]).predict(X)))

    team_idx = {p["team"]: i for i, p in enumerate(points)}
    name_of = {labels[team_idx[t]]: (order, name) for order, (t, name) in enumerate(CLUSTER_NAMES)}
    assert len(name_of) == K, "anchor teams no longer land in distinct clusters; rename"
    sil = silhouette_samples(X, labels)
    for i, p in enumerate(points):
        p["cluster"], p["cluster_name"] = name_of[labels[i]]
        p["sil"] = round(float(sil[i]), 2)

    clusters = []
    for order, (_, name) in enumerate(CLUSTER_NAMES):
        members = [p for p in points if p["cluster"] == order]
        clusters.append({
            "name": name,
            "teams": [p["team"] for p in members],
            "x": round(float(np.mean([p["x"] for p in members])), 1),
            "y": round(float(np.mean([p["y"] for p in members])), 1),
            "sil": round(float(np.mean([p["sil"] for p in members])), 2),
        })

    stats = {
        "k": K,
        "silhouette": next(r["silhouette"] for r in sweep if r["k"] == K),
        "ward_ari": round(float(ward_ari), 2),
        "boot_ari_mean": round(float(np.mean(boot)), 2),
        "boot_ari_p10": round(float(np.percentile(boot, 10)), 2),
        "n_boot": N_BOOT,
    }

    payload = {"points": points, "clusters": clusters, "sweep": sweep, "stats": stats}
    html = TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    with open(OUT_PATH, "w") as out:
        out.write(html)

    cols = ["team", "cluster_name", "x", "y", "outside", "inside", "uc", "pistol", "rush_n", "sil"]
    with open(CSV_PATH, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(points, key=lambda p: (p["cluster"], p["team"])))

    print(f"wrote {OUT_PATH}, {CSV_PATH} ({len(points)} teams)")
    print(json.dumps(stats))
    for c in clusters:
        print(f'{c["name"]:<26} x={c["x"]:>6} y={c["y"]:>5} sil={c["sil"]}  {" ".join(c["teams"])}')


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Direction vs Formation</title>
<style>
:root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #8a8984;
  --grid: #e6e5e1;
  --axis: #b9b8b2;
  --series-1: #2a78d6;
  --c0: #2a78d6; --c1: #eb6834; --c2: #1baf7a; --c3: #eda100; --c4: #e87ba4; --c5: #008300;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #8f8e87;
    --grid: #2e2e2c;
    --axis: #55544f;
    --series-1: #3987e5;
    --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #8f8e87;
  --grid: #2e2e2c;
  --axis: #55544f;
  --series-1: #3987e5;
--c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300;
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
.cent { fill: none; stroke-width: 1.5; stroke-dasharray: 3 2; opacity: .7; }
#ctbl td:last-child, #ctbl th:last-child { text-align: left; }
.lbl { fill: var(--text-primary); font-size: 11px; font-weight: 600; pointer-events: none; }
.hit { fill: transparent; cursor: default; }
.dot.on { stroke: var(--text-primary); stroke-width: 1.5; }
#tip { position: fixed; pointer-events: none; background: var(--surface-1);
  border: 1px solid var(--axis); border-radius: 6px; padding: 8px 10px;
  font-size: 12px; color: var(--text-primary); display: none;
  box-shadow: 0 2px 8px rgba(0,0,0,.15); }
#tip b { font-size: 13px; }
#tip span { color: var(--text-secondary); }
details { margin-top: 20px; }
table { border-collapse: collapse; font-size: 12px; margin-top: 8px; width: 100%; }
th, td { padding: 4px 8px; text-align: right; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 8px; font-size: 12px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 12px; height: 12px; overflow: visible; }
.stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0 4px; }
.stat { border: 1px solid var(--grid); border-radius: 8px; padding: 8px 12px; min-width: 120px; }
.stat b { display: block; font-size: 18px; }
.stat span { color: var(--text-secondary); font-size: 12px; }
h2 { font-size: 15px; margin: 24px 0 4px; }
tr.pick td { font-weight: 700; }
.note { color: var(--text-muted); font-size: 12px; margin-top: 12px; }
</style>
</head>
<body>
<main>
  <h1>Run direction vs snap formation, 2025</h1>
  <p class="sub">Each mark is a 2025 offense, colored and shaped by cluster (KMeans on these two dimensions only, standardized). Neutral game script only (WP 20-80%, outside last 2 min of each half). Dashed rings are cluster centers.</p>
  <div class="legend" id="legend"></div>
  <svg id="chart" viewBox="0 0 900 620" role="img" aria-label="Scatter of 32 NFL offenses"></svg>
  <p class="note">X: outside runs (end + tackle gaps) minus inside runs (guard + middle), as a share of designed rushes, in percentage points. Y: share of snaps from shotgun or pistol; under center = 100 minus Y. Dashed lines are league averages.</p>
  <div class="stats" id="stats"></div>
  <h2>Clusters</h2>
  <table id="ctbl"></table>
  <h2>k sweep (KMeans, standardized)</h2>
  <table id="ktbl"></table>
  <details>
    <summary>Team table</summary>
    <table id="tbl"></table>
  </details>
</main>
<div id="tip"></div>
<script>
const payload = __DATA_JSON__;
const data = payload.points, clusters = payload.clusters, st = payload.stats;
// color + shape double-encode cluster: 6 hues can't all separate for CVD in a scatter
const shape = (c, r) => {
  const a = r * 1.25;
  return [
    `M${r},0A${r},${r} 0 1,1 ${-r},0A${r},${r} 0 1,1 ${r},0Z`,
    `M${-r*.9},${-r*.9}H${r*.9}V${r*.9}H${-r*.9}Z`,
    `M0,${-a}L${a*.95},${a*.6}H${-a*.95}Z`,
    `M0,${-a}L${a},0L0,${a}L${-a},0Z`,
    `M0,${a}L${a*.95},${-a*.6}H${-a*.95}Z`,
    `M${-r*.35},${-r}H${r*.35}V${-r*.35}H${r}V${r*.35}H${r*.35}V${r}H${-r*.35}V${r*.35}H${-r}V${-r*.35}H${-r*.35}Z`,
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
const ext = (k, pad) => {
  const v = data.map(d => d[k]);
  return [Math.floor((Math.min(...v) - pad) / 5) * 5, Math.ceil((Math.max(...v) + pad) / 5) * 5];
};
const [x0, x1] = ext("x", 2), [y0, y1] = ext("y", 2);
const sx = v => m.l + (v - x0) / (x1 - x0) * pw;
const sy = v => m.t + ph - (v - y0) / (y1 - y0) * ph;
const mean = k => data.reduce((s, d) => s + d[k], 0) / data.length;
const mx = mean("x"), my = mean("y");

const g = el("g", {class: "grid"});
for (let v = Math.ceil(x0 / 10) * 10; v <= x1; v += 10) {
  el("line", {x1: sx(v), x2: sx(v), y1: m.t, y2: m.t + ph}, g);
  el("text", {x: sx(v), y: m.t + ph + 18, "text-anchor": "middle", class: "tick"}, svg, (v > 0 ? "+" : "") + v);
}
for (let v = y0; v <= y1; v += 5) {
  el("line", {x1: m.l, x2: m.l + pw, y1: sy(v), y2: sy(v)}, g);
  el("text", {x: m.l - 8, y: sy(v) + 4, "text-anchor": "end", class: "tick"}, svg, v + "%");
}
el("line", {x1: sx(mx), x2: sx(mx), y1: m.t, y2: m.t + ph, class: "zero"});
el("line", {x1: m.l, x2: m.l + pw, y1: sy(my), y2: sy(my), class: "zero"});
el("text", {x: m.l + pw / 2, y: H - 12, "text-anchor": "middle", class: "axis-title"}, svg,
   "Outside minus inside run share (pp)");
el("text", {transform: `translate(16 ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle", class: "axis-title"}, svg,
   "Shotgun + pistol snap share");
el("text", {x: m.l + 8, y: m.t + 14, class: "quad"}, svg, "Gun / inside");
el("text", {x: m.l + pw - 8, y: m.t + 14, "text-anchor": "end", class: "quad"}, svg, "Gun / outside");
el("text", {x: m.l + 8, y: m.t + ph - 8, class: "quad"}, svg, "Under center / inside");
el("text", {x: m.l + pw - 8, y: m.t + ph - 8, "text-anchor": "end", class: "quad"}, svg, "Under center / outside");

clusters.forEach((c, i) => {
  el("circle", {cx: sx(c.x), cy: sy(c.y), r: 9, class: "cent", stroke: `var(--c${i})`});
});
document.getElementById("legend").innerHTML = clusters.map((c, i) =>
  `<span><svg viewBox="-7 -7 14 14"><path d="${shape(i, 5.5)}" fill="var(--c${i})"/></svg>${c.name} (${c.teams.length})</span>`).join("");
document.getElementById("stats").innerHTML = [
  [st.k, "clusters"], [st.silhouette, "silhouette"], [st.ward_ari, "ARI vs Ward"],
  [st.boot_ari_mean, `bootstrap ARI, mean (n=${st.n_boot})`], [st.boot_ari_p10, "bootstrap ARI, 10th pct"],
].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
document.getElementById("ctbl").innerHTML =
  "<tr><th>Cluster</th><th>n</th><th>Out - in (pp)</th><th>Shotgun+pistol %</th><th>Mean silhouette</th><th>Teams</th></tr>" +
  clusters.map(c => `<tr><td>${c.name}</td><td>${c.teams.length}</td><td>${(c.x > 0 ? "+" : "") + c.x}</td><td>${c.y}</td><td>${c.sil}</td><td>${c.teams.join(", ")}</td></tr>`).join("");
document.getElementById("ktbl").innerHTML =
  "<tr><th>k</th><th>Silhouette</th><th>Calinski-Harabasz</th><th>Davies-Bouldin</th><th>Cluster sizes</th></tr>" +
  payload.sweep.map(r => `<tr class="${r.k === st.k ? "pick" : ""}"><td>${r.k}</td><td>${r.silhouette}</td><td>${r.calinski_harabasz}</td><td>${r.davies_bouldin}</td><td>${r.sizes.join(" / ")}</td></tr>`).join("");

// greedy label placement: try 8 spots around each dot, skip ones that hit placed labels/dots
const placed = data.map(d => ({x: sx(d.x) - 6, y: sy(d.y) - 6, w: 12, h: 12}));
const hits = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
const tip = document.getElementById("tip");
const sign = v => (v > 0 ? "+" : "") + v;
data.forEach(d => {
  const cx = sx(d.x), cy = sy(d.y);
  const hit = el("circle", {cx, cy, r: 12, class: "hit"});
  const dot = el("path", {d: shape(d.cluster, 5.5), transform: `translate(${cx} ${cy})`, class: "dot", fill: `var(--c${d.cluster})`});
  hit.addEventListener("mousemove", e => {
    dot.classList.add("on");
    tip.innerHTML = `<b>${d.team}</b> <span>${d.cluster_name}</span><br><span>Out - in:</span> ${sign(d.x)} pp (${d.outside}% / ${d.inside}%)` +
      `<br><span>Shotgun+pistol:</span> ${d.y}% (pistol ${d.pistol}%)<br><span>Under center:</span> ${d.uc}%` +
      `<br><span>Designed rushes:</span> ${d.rush_n}<br><span>Silhouette:</span> ${d.sil}`;
    tip.style.display = "block";
    const tx = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = tx + "px"; tip.style.top = (e.clientY + 14) + "px";
  });
  hit.addEventListener("mouseleave", () => { dot.classList.remove("on"); tip.style.display = "none"; });
});
data.forEach(d => {
  const cx = sx(d.x), cy = sy(d.y), w = d.team.length * 7.5, h = 12;
  const spots = [[8, 4, "start"], [-8, 4, "end"], [0, -9, "middle"], [0, 17, "middle"],
                 [7, -6, "start"], [-7, -6, "end"], [7, 14, "start"], [-7, 14, "end"]];
  let best = spots[0];
  for (const s of spots) {
    const bx = s[2] === "start" ? cx + s[0] : s[2] === "end" ? cx + s[0] - w : cx - w / 2;
    const box = {x: bx, y: cy + s[1] - 10, w, h};
    if (!placed.some(p => hits(p, box))) { best = s; placed.push(box); break; }
  }
  el("text", {x: cx + best[0], y: cy + best[1], "text-anchor": best[2], class: "lbl"}, svg, d.team);
});

const rows = [...data].sort((a, b) => b.x - a.x);
document.getElementById("tbl").innerHTML =
  "<tr><th>Team</th><th>Cluster</th><th>Out - in (pp)</th><th>Outside %</th><th>Inside %</th><th>Shotgun+pistol %</th><th>Pistol %</th><th>Under center %</th><th>Rushes</th></tr>" +
  rows.map(d => `<tr><td>${d.team}</td><td>${d.cluster_name}</td><td>${sign(d.x)}</td><td>${d.outside}</td><td>${d.inside}</td><td>${d.y}</td><td>${d.pistol}</td><td>${d.uc}</td><td>${d.rush_n}</td></tr>`).join("");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
