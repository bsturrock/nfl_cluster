"""Chart the run-game archetypes from archetypes_run_identity.py.

Sections: archetype profiles (z heatmap), every team's archetype mix by
season (stacked bars, 2022-2025, with year-over-year shift), a per-season
team similarity matrix on all six features, the biggest identity shifts,
and the k sweep.

Output: output/run_archetypes_viz.html
"""
import json

from archetypes_run_identity import SUMMARY_PATH

OUT_PATH = "output/run_archetypes_viz.html"


def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    with open(OUT_PATH, "w") as out:
        out.write(TEMPLATE.replace("__DATA_JSON__", json.dumps(summary)))
    print(f"wrote {OUT_PATH}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Game Archetypes</title>
<style>
:root {
  color-scheme: light;
  --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a8984;
  --grid: #e6e5e1; --axis: #b9b8b2;
  --c0: #2a78d6; --c1: #eb6834; --c2: #1baf7a; --c3: #eda100; --c4: #e87ba4; --c5: #008300;
  --div-neg: #2a78d6; --div-mid: #f0efec; --div-pos: #e34948; --seq: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e87;
    --grid: #2e2e2c; --axis: #55544f;
    --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300;
    --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767; --seq: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e87;
  --grid: #2e2e2c; --axis: #55544f;
  --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300;
  --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767; --seq: #3987e5;
}
body { margin: 0; background: var(--surface-1); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 40px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 30px 0 4px; }
.sub, .hint { color: var(--text-secondary); margin: 0 0 12px; max-width: 900px; }
.hint { font-size: 12px; margin-bottom: 8px; }
.stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 4px 0 12px; }
.stat { border: 1px solid var(--grid); border-radius: 8px; padding: 8px 12px; min-width: 110px; }
.stat b { display: block; font-size: 18px; }
.stat span { color: var(--text-secondary); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12px; color: var(--text-secondary); margin: 6px 0 10px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
table { border-collapse: collapse; font-size: 12px; width: 100%; }
th, td { padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--grid); }
th.l, td.l { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
td.heat { text-align: center; font-variant-numeric: tabular-nums; min-width: 64px; border: 2px solid var(--surface-1); }
.scale { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.scale i { display: inline-block; width: 160px; height: 10px; border-radius: 2px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin: 4px 0 10px; font-size: 12px; color: var(--text-secondary); }
.seg { display: inline-flex; border: 1px solid var(--axis); border-radius: 6px; overflow: hidden; }
.seg button { background: none; border: 0; padding: 4px 10px; font: inherit; color: var(--text-secondary); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--axis); }
.seg button.on { background: var(--text-primary); color: var(--surface-1); }
.mixgrid { display: grid; grid-template-columns: 44px repeat(4, minmax(0, 1fr)) 64px; gap: 3px 6px; align-items: center; font-size: 12px; }
.mixgrid .hd { color: var(--text-secondary); font-weight: 600; text-align: center; }
.mixgrid .tm { font-weight: 600; }
.mixgrid .sh { text-align: right; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.bar { display: flex; height: 16px; gap: 2px; }
.bar div { height: 100%; }
.bar div:first-child { border-radius: 3px 0 0 3px; }
.bar div:last-child { border-radius: 0 3px 3px 0; }
.bar.poor { opacity: .45; }
.bar.none { background: repeating-linear-gradient(45deg, var(--grid) 0 3px, transparent 3px 6px); border-radius: 3px; }
.mx-wrap { overflow-x: auto; }
svg.mx { display: block; width: 100%; max-width: 760px; height: auto; }
.mx text { fill: var(--text-secondary); font-size: 10px; }
.mx rect.cell:hover { stroke: var(--text-primary); stroke-width: 1.5; }
#tip { position: fixed; pointer-events: none; background: var(--surface-1);
  border: 1px solid var(--axis); border-radius: 6px; padding: 8px 10px; max-width: 320px;
  font-size: 12px; color: var(--text-primary); display: none; box-shadow: 0 2px 8px rgba(0,0,0,.15); }
#tip b { font-size: 13px; }
#tip span { color: var(--text-secondary); }
.note { color: var(--text-muted); font-size: 12px; margin-top: 8px; max-width: 900px; }
tr.pick td { font-weight: 700; }
details { margin-top: 20px; }
</style>
</head>
<body>
<main>
  <h1>Run game archetypes, 2022-2025</h1>
  <p class="sub">Archetypal analysis on six run-game features (designed runs, neutral game script). Six "pure" run games are found at the edges of the data, and every team-season is described as a mix of them. Features are standardized within each season, so a team is measured against that year's league. All seasons share one set of archetypes, so mixes compare year to year.</p>
  <div class="stats" id="stats"></div>

  <h2>The archetypes</h2>
  <p class="hint">Each archetype's profile in standard deviations from the league average. Anchor = the team-season closest to pure.</p>
  <table id="arch"></table>
  <div class="scale"><span>-2</span><i style="background:linear-gradient(90deg,var(--div-neg),var(--div-mid),var(--div-pos))"></i><span>+2 sd</span></div>

  <h2>Every team's mix by season</h2>
  <div class="legend" id="legend"></div>
  <div class="controls">Sort
    <span class="seg" id="sort"><button data-s="team" class="on">Team</button><button data-s="shift">Total shift</button><button data-s="arch">2025 archetype</button></span>
  </div>
  <p class="hint">Bar = that season's archetype mix; hover for details and nearest comps. Faded bars fit poorly (R&sup2; &lt; 0.5). Shift = total year-over-year change in mix, summed over 2022-25 (each step 0 = same mix, 1 = no overlap).</p>
  <div class="mixgrid" id="mix"></div>

  <h2>Team similarity</h2>
  <div class="controls">Season <span class="seg" id="season"></span></div>
  <p class="hint">Distance between teams on all six features at once (darker = more similar). Ordered so similar teams sit together; hover a cell for the pair.</p>
  <div class="mx-wrap"><svg class="mx" id="mx" role="img"></svg></div>
  <div class="scale"><span>Most similar pairs</span><i style="background:linear-gradient(90deg,color-mix(in oklab,var(--seq) 90%,var(--surface-1)),var(--surface-1))"></i><span>Median and beyond</span></div>

  <h2>Biggest identity shifts</h2>
  <table id="shifts"></table>

  <h2>How many archetypes</h2>
  <table id="sweep"></table>
  <p class="note" id="caveat"></p>

  <details>
    <summary>Full team-season table</summary>
    <table id="tbl"></table>
  </details>
</main>
<div id="tip"></div>
<script>
const S = __DATA_JSON__;
const A = S.archetypes, names = A.map(a => a.name), F = S.features, K = names.length;
const LABEL = {run_dir: "Out - in", shotgun_share: "Shotgun", te_minus_backs: "TE - backs", motion_rate: "Motion", rpo_rate: "RPO", qb_run_share: "QB runs"};
const FMT = {
  run_dir: v => (v > 0 ? "+" : "") + (v * 100).toFixed(1) + " pp",
  shotgun_share: v => (v * 100).toFixed(1) + "%",
  te_minus_backs: v => (v > 0 ? "+" : "") + v.toFixed(2),
  motion_rate: v => (v * 100).toFixed(1) + "%",
  rpo_rate: v => (v * 100).toFixed(1) + "%",
  qb_run_share: v => (v * 100).toFixed(1) + "%",
};
const sgn = v => (v > 0 ? "+" : "") + v.toFixed(2);
const tip = document.getElementById("tip");
const showTip = (e, html) => {
  tip.innerHTML = html; tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8) + "px";
};
const hideTip = () => { tip.style.display = "none"; };
const rowOf = {}; S.rows.forEach(r => { rowOf[r.team + " " + r.season] = r; });
const teams = [...new Set(S.rows.map(r => r.team))].sort();

document.getElementById("stats").innerHTML = [
  [S.k, "archetypes"], [Math.round(S.var_explained * 100) + "%", "variance explained"],
  [S.rows.length, "team-seasons"], [S["rows_r2_below_0.5"], "team-seasons fit poorly"],
  [S.boot_move_median + " sd", "bootstrap archetype movement (median)"],
].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");

const heatBg = z => `color-mix(in oklab, var(${z >= 0 ? "--div-pos" : "--div-neg"}) ${(Math.min(Math.abs(z) / 2, 1) * 85).toFixed(0)}%, var(--div-mid))`;
document.getElementById("arch").innerHTML = "<tr><th class='l'>Archetype</th>" + F.map(f => `<th style="text-align:center">${LABEL[f]}</th>`).join("") +
  "<th class='l'>Anchor</th><th class='l'>Most pure team-seasons</th><th>Bootstrap move (sd)</th></tr>" +
  A.map((a, i) => `<tr><td class="l"><span class="sw" style="background:var(--c${i});margin-right:6px"></span>${a.name}</td>` +
    F.map(f => `<td class="heat" style="background:${heatBg(a.z[f])}">${sgn(a.z[f])}</td>`).join("") +
    `<td class="l">${a.anchor}</td><td class="l">${a.top.slice(1, 5).map(t => `${t.key} (${Math.round(t.weight * 100)}%)`).join(", ")}</td><td>${a.boot_move_median}</td></tr>`).join("");

document.getElementById("legend").innerHTML = names.map((n, i) => `<span><i class="sw" style="background:var(--c${i})"></i>${n}</span>`).join("");

const teamShift = {};
teams.forEach(t => { teamShift[t] = S.shifts.filter(s => s.team === t).reduce((a, s) => a + s.shift, 0); });
const mixTip = r => `<b>${r.team} ${r.season}</b> <span>R&sup2; ${r.r2}</span>` +
  r.weights.map((w, i) => w >= 0.01 ? `<br><i class="sw" style="background:var(--c${i})"></i> ${names[i]}: ${Math.round(w * 100)}%` : "").join("") +
  "<br>" + F.map(f => `<br><span>${LABEL[f]}:</span> ${FMT[f](r.raw[f])} (${sgn(r.z[f])} sd)`).join("") +
  `<br><br><span>Nearest comps:</span> ${r.comps.map(c => c.key).join(", ")}`;
function drawMix(sortBy) {
  const order = [...teams];
  if (sortBy === "shift") order.sort((a, b) => teamShift[b] - teamShift[a]);
  if (sortBy === "arch") order.sort((a, b) => {
    const ra = rowOf[a + " 2025"], rb = rowOf[b + " 2025"];
    const ia = ra.weights.indexOf(Math.max(...ra.weights)), ib = rb.weights.indexOf(Math.max(...rb.weights));
    return ia - ib || Math.max(...rb.weights) - Math.max(...ra.weights);
  });
  const g = document.getElementById("mix");
  g.innerHTML = "<div></div>" + S.seasons.map(s => `<div class="hd">${s}</div>`).join("") + "<div class='hd' style='text-align:right'>Shift</div>";
  order.forEach(t => {
    g.insertAdjacentHTML("beforeend", `<div class="tm">${t}</div>`);
    S.seasons.forEach(s => {
      const r = rowOf[t + " " + s];
      const div = document.createElement("div");
      if (!r) { div.className = "bar none"; g.appendChild(div); return; }
      div.className = "bar" + (r.r2 < 0.5 ? " poor" : "");
      r.weights.forEach((w, i) => { if (w >= 0.005) div.insertAdjacentHTML("beforeend", `<div style="flex:${w};background:var(--c${i})"></div>`); });
      div.addEventListener("mousemove", e => showTip(e, mixTip(r)));
      div.addEventListener("mouseleave", hideTip);
      g.appendChild(div);
    });
    g.insertAdjacentHTML("beforeend", `<div class="sh">${teamShift[t].toFixed(2)}</div>`);
  });
}
document.querySelectorAll("#sort button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#sort button").forEach(x => x.classList.toggle("on", x === b));
  drawMix(b.dataset.s);
}));
drawMix("team");

const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent, text) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  parent.appendChild(e); return e;
};
function drawMatrix(season) {
  const M = S.matrices[season], svg = document.getElementById("mx");
  svg.innerHTML = "";
  svg.setAttribute("aria-label", `Team similarity matrix for ${season}`);
  const n = M.order.length, c = 20, pad = 36;
  svg.setAttribute("viewBox", `0 0 ${pad + n * c + 4} ${pad + n * c + 4}`);
  const idx = Object.fromEntries(M.teams.map((t, i) => [t, i]));
  // color by this season's own distance range: 5th pct = darkest, 60th pct and up = blank
  const all = M.distance.flatMap((row, i) => row.filter((_, j) => j > i)).sort((a, b) => a - b);
  const lo = all[Math.floor(all.length * .05)], hi = all[Math.floor(all.length * .6)];
  M.order.forEach((t, i) => {
    el("text", {x: pad - 4, y: pad + i * c + c / 2 + 3, "text-anchor": "end"}, svg, t);
    el("text", {transform: `translate(${pad + i * c + c / 2 + 3} ${pad - 4}) rotate(-90)`}, svg, t);
  });
  M.order.forEach((a, i) => M.order.forEach((b, j) => {
    const dist = M.distance[idx[a]][idx[b]];
    const t = a === b ? 0 : Math.min(1, Math.max(0, (hi - dist) / (hi - lo))) * 90;
    const r = el("rect", {x: pad + j * c, y: pad + i * c, width: c - 1, height: c - 1, rx: 2, class: "cell",
      fill: a === b ? "var(--grid)" : `color-mix(in oklab, var(--seq) ${t.toFixed(0)}%, var(--surface-1))`}, svg);
    if (a === b) return;
    r.addEventListener("mousemove", e => {
      const ra = rowOf[a + " " + season], rb = rowOf[b + " " + season];
      showTip(e, `<b>${a} vs ${b}, ${season}</b><br><span>Distance:</span> ${dist.toFixed(2)} (all six features)` +
        F.map(f => `<br><span>${LABEL[f]}:</span> ${sgn(ra.z[f])} vs ${sgn(rb.z[f])}`).join(""));
    });
    r.addEventListener("mouseleave", hideTip);
  }));
}
const segS = document.getElementById("season");
segS.innerHTML = S.seasons.map(s => `<button data-y="${s}" class="${s === 2025 ? "on" : ""}">${s}</button>`).join("");
segS.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
  segS.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
  drawMatrix(b.dataset.y);
}));
drawMatrix(2025);

document.getElementById("shifts").innerHTML = "<tr><th class='l'>Team</th><th class='l'>Seasons</th><th>Shift</th><th class='l'>From (top archetype)</th><th class='l'>To (top archetype)</th></tr>" +
  [...S.shifts].sort((a, b) => b.shift - a.shift).slice(0, 15).map(s =>
    `<tr><td class="l">${s.team}</td><td class="l">${s.from} to ${s.to}</td><td>${s.shift.toFixed(2)}</td><td class="l">${s.from_top}</td><td class="l">${s.to_top}</td></tr>`).join("");

document.getElementById("sweep").innerHTML = "<tr><th>Archetypes</th><th>Variance explained</th><th>Team-seasons fit poorly (R&sup2; &lt; 0.5)</th><th>Median top weight</th></tr>" +
  S.sweep.map(r => `<tr class="${r.k === S.k ? "pick" : ""}"><td>${r.k}</td><td>${Math.round(r.var_explained * 100)}%</td><td>${r["rows_r2_below_0.5"]}</td><td>${r.median_top_weight}</td></tr>`).join("");
const loose = A.filter(a => a.boot_move_median > 1.5).map(a => a.name);
document.getElementById("caveat").textContent = `Independent restarts find identical archetypes. Resampling team-seasons moves them a median ${S.boot_move_median} sd, because archetypes sit at the extremes and are defined by a few team-seasons` +
  (loose.length ? `; least stable: ${loose.join(", ")}.` : ".");

document.getElementById("tbl").innerHTML = "<tr><th class='l'>Team</th><th>Season</th><th class='l'>Top archetype</th>" +
  names.map(n => `<th>${n}</th>`).join("") + "<th>R&sup2;</th><th class='l'>Comps</th></tr>" +
  [...S.rows].sort((a, b) => a.team.localeCompare(b.team) || a.season - b.season).map(r =>
    `<tr><td class="l">${r.team}</td><td>${r.season}</td><td class="l">${r.top_archetype}</td>` +
    r.weights.map(w => `<td>${Math.round(w * 100)}%</td>`).join("") + `<td>${r.r2}</td><td class="l">${r.comps.map(c => c.key).join(", ")}</td></tr>`).join("");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
