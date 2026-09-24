"""Chart the run-identity clustering from cluster_run_identity.py.

The four features are nearly uncorrelated, so a 2D PCA projection keeps only
~59% of the variance. Instead the page shows two raw-feature panels (run
direction x shotgun share, TE-minus-backs x motion) that together display all
four dimensions exactly, plus a cluster-profile heatmap and fit stats.

Output: output/run_identity_viz_2025.html
"""
import csv
import json

from cluster_run_identity import CLUSTER_NAMES, CSV_PATH, SUMMARY_PATH

OUT_PATH = "output/run_identity_viz_2025.html"


def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    order = {name: i for i, (_, name) in enumerate(CLUSTER_NAMES)}
    with open(CSV_PATH) as f:
        teams = [{
            "team": r["team"],
            "cluster": order[r["cluster_name"]],
            "sil": round(float(r["silhouette"]), 2),
            **{k: round(float(r[k]), 4) for k in summary["features"]},
        } for r in csv.DictReader(f)]

    payload = {"teams": teams, "names": [n for _, n in CLUSTER_NAMES], "summary": summary}
    with open(OUT_PATH, "w") as out:
        out.write(TEMPLATE.replace("__DATA_JSON__", json.dumps(payload)))
    print(f"wrote {OUT_PATH}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Identity Clusters</title>
<style>
:root {
  color-scheme: light;
  --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a8984;
  --grid: #e6e5e1; --axis: #b9b8b2;
  --c0: #2a78d6; --c1: #eb6834; --c2: #1baf7a; --c3: #eda100; --c4: #e87ba4; --c5: #008300;
  --div-neg: #2a78d6; --div-mid: #f0efec; --div-pos: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e87;
    --grid: #2e2e2c; --axis: #55544f;
    --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300;
    --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e87;
  --grid: #2e2e2c; --axis: #55544f;
  --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300;
  --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767;
}
body { margin: 0; background: var(--surface-1); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 40px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 6px; }
.sub { color: var(--text-secondary); margin: 0 0 14px; max-width: 900px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 8px; font-size: 12px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 12px; height: 12px; overflow: visible; }
.stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 4px 0 8px; }
.stat { border: 1px solid var(--grid); border-radius: 8px; padding: 8px 12px; min-width: 110px; }
.stat b { display: block; font-size: 18px; }
.stat span { color: var(--text-secondary); font-size: 12px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 760px) { .panels { grid-template-columns: 1fr; } }
.panel h3 { font-size: 13px; margin: 0 0 2px; color: var(--text-secondary); font-weight: 600; }
svg.plot { width: 100%; height: auto; display: block; overflow: visible; }
.grid line { stroke: var(--grid); }
.zero { stroke: var(--axis); stroke-dasharray: 4 4; }
.tick { fill: var(--text-muted); font-size: 11px; }
.axis-title { fill: var(--text-secondary); font-size: 12px; }
.dot { stroke: var(--surface-1); stroke-width: 2; transition: opacity .12s; }
.dot.on { stroke: var(--text-primary); stroke-width: 1.5; }
.dim .dot:not(.hl) { opacity: .15; }
.dim .lbl:not(.hl) { opacity: .15; }
.lbl { fill: var(--text-primary); font-size: 11px; font-weight: 600; pointer-events: none; }
.leader { stroke: var(--text-muted); stroke-width: 1; }
.hit { fill: transparent; }
#tip { position: fixed; pointer-events: none; background: var(--surface-1);
  border: 1px solid var(--axis); border-radius: 6px; padding: 8px 10px;
  font-size: 12px; color: var(--text-primary); display: none; box-shadow: 0 2px 8px rgba(0,0,0,.15); }
#tip b { font-size: 13px; }
#tip span { color: var(--text-secondary); }
table { border-collapse: collapse; font-size: 12px; width: 100%; }
th, td { padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child, td.teams, th.teams { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
td.heat { text-align: center; font-variant-numeric: tabular-nums; min-width: 72px; border: 2px solid var(--surface-1); }
tr.pick td { font-weight: 700; }
.scale { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.scale i { display: inline-block; width: 160px; height: 10px; border-radius: 2px;
  background: linear-gradient(90deg, var(--div-neg), var(--div-mid), var(--div-pos)); }
.note { color: var(--text-muted); font-size: 12px; margin-top: 8px; max-width: 900px; }
.cl-row { cursor: default; }
details { margin-top: 20px; }
</style>
</head>
<body>
<main>
  <h1>Run identity clusters, 2025</h1>
  <p class="sub">KMeans (k=6) on four standardized features of designed non-QB runs in neutral game script: run direction, shotgun share, TEs minus backs, and pre-snap motion. The two panels together show all four features; hover a team, or a cluster in the legend or heatmap, to highlight it in both.</p>
  <div class="stats" id="stats"></div>
  <div class="legend" id="legend"></div>
  <div class="panels">
    <div class="panel"><h3>Run direction vs formation</h3><svg class="plot" id="p1" viewBox="0 0 540 440" role="img"></svg></div>
    <div class="panel"><h3>Personnel vs motion</h3><svg class="plot" id="p2" viewBox="0 0 540 440" role="img"></svg></div>
  </div>
  <p class="note">Dashed lines are league averages. Separation is weak (silhouette 0.23) because the four features are nearly uncorrelated, so teams form one diffuse cloud; a team close to its neighbors in one panel can sit far apart in the other.</p>

  <h2>Cluster profiles (z-score vs league average)</h2>
  <table id="heat"></table>
  <div class="scale"><span>-2</span><i></i><span>+2 sd</span></div>

  <h2>Which features drive the split</h2>
  <table id="anova"></table>

  <h2>k sweep</h2>
  <table id="sweep"></table>

  <details>
    <summary>Team table</summary>
    <table id="tbl"></table>
  </details>
</main>
<div id="tip"></div>
<script>
const P = __DATA_JSON__;
const teams = P.teams, names = P.names, S = P.summary;
const LABEL = {run_dir: "Out - in", shotgun_share: "Shotgun", te_minus_backs: "TE - backs", motion_rate: "Motion"};
const FMT = {
  run_dir: v => (v > 0 ? "+" : "") + (v * 100).toFixed(1) + " pp",
  shotgun_share: v => (v * 100).toFixed(1) + "%",
  te_minus_backs: v => (v > 0 ? "+" : "") + v.toFixed(2),
  motion_rate: v => (v * 100).toFixed(1) + "%",
};
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
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent, text) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  parent.appendChild(e); return e;
};
const tip = document.getElementById("tip");
const svgs = [];

function highlight(pred) {
  svgs.forEach(svg => {
    svg.classList.toggle("dim", !!pred);
    svg.querySelectorAll("[data-team]").forEach(n => {
      const t = teams.find(d => d.team === n.dataset.team);
      n.classList.toggle("hl", !!pred && pred(t));
    });
  });
}

function scatter(id, xk, yk, xStep, yStep, xTick, yTick, xTitle, yTitle) {
  const svg = document.getElementById(id); svgs.push(svg);
  svg.setAttribute("aria-label", `Scatter of 32 offenses: ${xTitle} vs ${yTitle}, colored by cluster`);
  const W = 540, H = 440, m = {t: 12, r: 16, b: 48, l: 52};
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const ext = (k, s) => { const v = teams.map(d => d[k]); return [Math.floor(Math.min(...v) / s - .5) * s, Math.ceil(Math.max(...v) / s + .5) * s]; };
  const [x0, x1] = ext(xk, xStep), [y0, y1] = ext(yk, yStep);
  const sx = v => m.l + (v - x0) / (x1 - x0) * pw, sy = v => m.t + ph - (v - y0) / (y1 - y0) * ph;
  const g = el("g", {class: "grid"}, svg);
  for (let i = 0; x0 + i * xStep <= x1 + 1e-9; i++) {
    const v = x0 + i * xStep;
    el("line", {x1: sx(v), x2: sx(v), y1: m.t, y2: m.t + ph}, g);
    el("text", {x: sx(v), y: m.t + ph + 16, "text-anchor": "middle", class: "tick"}, svg, xTick(v));
  }
  for (let i = 0; y0 + i * yStep <= y1 + 1e-9; i++) {
    const v = y0 + i * yStep;
    el("line", {x1: m.l, x2: m.l + pw, y1: sy(v), y2: sy(v)}, g);
    el("text", {x: m.l - 6, y: sy(v) + 4, "text-anchor": "end", class: "tick"}, svg, yTick(v));
  }
  const mean = k => teams.reduce((s, d) => s + d[k], 0) / teams.length;
  el("line", {x1: sx(mean(xk)), x2: sx(mean(xk)), y1: m.t, y2: m.t + ph, class: "zero"}, svg);
  el("line", {x1: m.l, x2: m.l + pw, y1: sy(mean(yk)), y2: sy(mean(yk)), class: "zero"}, svg);
  el("text", {x: m.l + pw / 2, y: H - 8, "text-anchor": "middle", class: "axis-title"}, svg, xTitle);
  el("text", {transform: `translate(12 ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle", class: "axis-title"}, svg, yTitle);

  const placed = teams.map(d => ({x: sx(d[xk]) - 6, y: sy(d[yk]) - 6, w: 12, h: 12}));
  const hits = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  teams.forEach(d => {
    const cx = sx(d[xk]), cy = sy(d[yk]);
    const hit = el("circle", {cx, cy, r: 11, class: "hit"}, svg);
    const dot = el("path", {d: shape(d.cluster, 5), transform: `translate(${cx} ${cy})`, class: "dot",
                            fill: `var(--c${d.cluster})`, "data-team": d.team}, svg);
    hit.addEventListener("mousemove", e => {
      highlight(t => t.team === d.team);
      tip.innerHTML = `<b>${d.team}</b> <span>${names[d.cluster]}</span>` +
        S.features.map(f => `<br><span>${LABEL[f]}:</span> ${FMT[f](d[f])}`).join("") +
        `<br><span>Silhouette:</span> ${d.sil}`;
      tip.style.display = "block";
      tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = (e.clientY + 14) + "px";
    });
    hit.addEventListener("mouseleave", () => { highlight(null); tip.style.display = "none"; });
  });
  const spots = [];
  for (const k of [1, 2.2, 3.4])
    for (const [dx, dy, a] of [[7, 4, "start"], [-7, 4, "end"], [0, -8, "middle"], [0, 16, "middle"],
                               [6, -6, "start"], [-6, -6, "end"], [6, 13, "start"], [-6, 13, "end"]])
      spots.push([dx * k, 4 + (dy - 4) * k, a, k > 1]);
  teams.forEach(d => {
    const cx = sx(d[xk]), cy = sy(d[yk]), w = d.team.length * 8.2 + 2, h = 12;
    let best = spots[0];
    for (const s of spots) {
      const bx = s[2] === "start" ? cx + s[0] : s[2] === "end" ? cx + s[0] - w : cx - w / 2;
      const box = {x: bx, y: cy + s[1] - 10, w, h};
      if (bx > m.l && bx + w < m.l + pw && box.y > m.t && !placed.some(p => hits(p, box))) { best = s; placed.push(box); break; }
    }
    if (best[3]) el("line", {x1: cx, y1: cy, x2: cx + best[0] * .8, y2: cy + (best[1] - 4) * .8, class: "leader", "data-team": d.team}, svg);
    el("text", {x: cx + best[0], y: cy + best[1], "text-anchor": best[2], class: "lbl", "data-team": d.team}, svg, d.team);
  });
}

const pct0 = v => Math.round(v * 100) + "%";
const spp = v => (v > 1e-9 ? "+" : "") + Math.round(v * 100);
const sd1 = v => (v > 1e-9 ? "+" : "") + v.toFixed(1);
scatter("p1", "run_dir", "shotgun_share", 0.1, 0.1, spp, pct0, "Outside minus inside runs (pp)", "Shotgun + pistol share of runs");
scatter("p2", "te_minus_backs", "motion_rate", 0.2, 0.1, sd1, pct0, "Avg TEs minus avg backs", "Runs with pre-snap motion");

const counts = names.map((_, i) => teams.filter(d => d.cluster === i).length);
const legend = document.getElementById("legend");
legend.innerHTML = names.map((n, i) =>
  `<span data-c="${i}"><svg viewBox="-7 -7 14 14"><path d="${shape(i, 5.5)}" fill="var(--c${i})"/></svg>${n} (${counts[i]})</span>`).join("");
legend.querySelectorAll("[data-c]").forEach(s => {
  s.addEventListener("mouseenter", () => highlight(t => t.cluster === +s.dataset.c));
  s.addEventListener("mouseleave", () => highlight(null));
});

document.getElementById("stats").innerHTML = [
  [S.k, "clusters"], [S.silhouette, "silhouette"], [S.bootstrap_ari_mean, "bootstrap ARI, mean (n=200)"],
  [S.bootstrap_ari_p10, "bootstrap ARI, 10th pct"], [S.n_teams, "teams"],
].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");

const heatBg = z => {
  const t = Math.min(Math.abs(z) / 2, 1) * 85;
  return `color-mix(in oklab, var(${z >= 0 ? "--div-pos" : "--div-neg"}) ${t.toFixed(0)}%, var(--div-mid))`;
};
const heat = document.getElementById("heat");
heat.innerHTML = "<tr><th>Cluster</th><th>n</th>" + S.features.map(f => `<th style="text-align:center">${LABEL[f]}</th>`).join("") +
  "<th>Silhouette</th><th class='teams'>Teams</th></tr>" +
  S.clusters.map((c, i) => `<tr class="cl-row" data-c="${i}"><td><svg viewBox="-7 -7 14 14" width="11" height="11" style="overflow:visible;margin-right:6px"><path d="${shape(i, 5.5)}" fill="var(--c${i})"/></svg>${c.cluster}</td><td>${c.n}</td>` +
    S.features.map(f => `<td class="heat" style="background:${heatBg(c.z[f])}" title="${LABEL[f]}: ${FMT[f](c.raw[f])} avg">${(c.z[f] > 0 ? "+" : "") + c.z[f].toFixed(2)}</td>`).join("") +
    `<td>${c.mean_silhouette}</td><td class="teams">${c.teams.join(", ")}</td></tr>`).join("");
heat.querySelectorAll("[data-c]").forEach(r => {
  r.addEventListener("mouseenter", () => highlight(t => t.cluster === +r.dataset.c));
  r.addEventListener("mouseleave", () => highlight(null));
});

document.getElementById("anova").innerHTML = "<tr><th>Feature</th><th>ANOVA F across clusters</th><th class='teams'>Reading</th></tr>" +
  Object.entries(S.feature_anova_f).sort((a, b) => b[1] - a[1]).map(([f, v], i) =>
    `<tr><td>${LABEL[f]}</td><td>${v}</td><td class="teams">${i === 0 ? "separates clusters most" : ""}</td></tr>`).join("");

document.getElementById("sweep").innerHTML =
  "<tr><th>k</th><th>Silhouette</th><th>Calinski-Harabasz</th><th>Davies-Bouldin</th><th>ARI vs Ward</th><th>Singletons</th><th>Sizes</th></tr>" +
  S.sweep.map(r => `<tr class="${r.k === S.k ? "pick" : ""}"><td>${r.k}</td><td>${r.silhouette}</td><td>${r.calinski_harabasz}</td><td>${r.davies_bouldin}</td><td>${r.ari_vs_ward}</td><td>${r.singletons}</td><td>${r.sizes.join(" / ")}</td></tr>`).join("");

document.getElementById("tbl").innerHTML = "<tr><th>Team</th><th class='teams'>Cluster</th>" +
  S.features.map(f => `<th>${LABEL[f]}</th>`).join("") + "<th>Silhouette</th></tr>" +
  [...teams].sort((a, b) => a.cluster - b.cluster || b.sil - a.sil).map(d =>
    `<tr><td>${d.team}</td><td class="teams">${names[d.cluster]}</td>` + S.features.map(f => `<td>${FMT[f](d[f])}</td>`).join("") + `<td>${d.sil}</td></tr>`).join("");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
