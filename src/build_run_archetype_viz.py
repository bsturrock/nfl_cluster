"""Chart the run-game archetypes from archetypes_run_identity.py.

Sections: archetype profiles (z heatmap), every team's archetype mix by
season (stacked bars, 2022-2025, with year-over-year shift), a per-season
team similarity matrix on all six features, the biggest identity shifts,
and the k sweep.

Everything is rendered to static HTML/SVG here so the page shows its data in
viewers that don't run JavaScript: hover details are native title tooltips,
the season switch is a CSS radio toggle, and the only script (re-sorting the
team mixes) is an optional enhancement that stays hidden without JS.

Output: output/run_archetypes_viz.html
"""
import json
from html import escape

import numpy as np

from archetypes_run_identity import SUMMARY_PATH

OUT_PATH = "output/run_archetypes_viz.html"

LABEL = {"run_dir": "Out - in", "shotgun_share": "Shotgun", "te_minus_backs": "TE - backs",
         "motion_rate": "Motion", "rpo_rate": "RPO", "qb_run_share": "QB runs"}


def fmt(f, v):
    if f == "run_dir":
        return f"{v * 100:+.1f} pp"
    if f == "te_minus_backs":
        return f"{v:+.2f}"
    return f"{v * 100:.1f}%"


def heat_bg(z):
    pole = "--div-pos" if z >= 0 else "--div-neg"
    return f"color-mix(in oklab, var({pole}) {min(abs(z) / 2, 1) * 85:.0f}%, var(--div-mid))"


def attr(s):
    return escape(s, quote=True)


def archetype_table(S):
    F, A = S["features"], S["archetypes"]
    head = "<tr><th class='l'>Archetype</th>" + "".join(f"<th class='c'>{LABEL[f]}</th>" for f in F) + \
        "<th class='l'>Anchor</th><th class='l'>Most pure team-seasons</th><th>Bootstrap move (sd)</th></tr>"
    body = ""
    for i, a in enumerate(A):
        cells = "".join(f"<td class='heat' style='background:{heat_bg(a['z'][f])}'>{a['z'][f]:+.2f}</td>" for f in F)
        top = ", ".join(f"{t['key']} ({round(t['weight'] * 100)}%)" for t in a["top"][1:5])
        body += (f"<tr><td class='l'><i class='sw' style='background:var(--c{i})'></i>{escape(a['name'])}</td>{cells}"
                 f"<td class='l'>{a['anchor']}</td><td class='l'>{top}</td><td>{a['boot_move_median']}</td></tr>")
    return f"<table>{head}{body}</table>"


def mix_grid(S):
    names = [a["name"] for a in S["archetypes"]]
    F, seasons = S["features"], S["seasons"]
    rows = {(r["team"], r["season"]): r for r in S["rows"]}
    teams = sorted({r["team"] for r in S["rows"]})
    shift = {t: sum(s["shift"] for s in S["shifts"] if s["team"] == t) for t in teams}

    def tip(r):
        lines = [f"{r['team']} {r['season']}  (R² {r['r2']})"]
        lines += [f"{names[i]}: {round(w * 100)}%" for i, w in enumerate(r["weights"]) if w >= 0.01]
        lines.append("")
        lines += [f"{LABEL[f]}: {fmt(f, r['raw'][f])} ({r['z'][f]:+.2f} sd)" for f in F]
        lines += ["", "Nearest comps: " + ", ".join(c["key"] for c in r["comps"])]
        return "\n".join(lines)

    out = ["<div class='mixgrid' id='mix'><div></div>"]
    out += [f"<div class='hd'>{s}</div>" for s in seasons]
    out.append("<div class='hd r'>Shift</div>")
    for t in teams:
        r25 = rows.get((t, seasons[-1]))
        top25 = int(np.argmax(r25["weights"])) if r25 else 99
        w25 = max(r25["weights"]) if r25 else 0
        out.append(f"<div class='row' data-team='{t}' data-shift='{shift[t]:.3f}' data-arch='{top25}' data-w='{w25:.3f}'>"
                   f"<div class='tm'>{t}</div>")
        for s in seasons:
            r = rows.get((t, s))
            if not r:
                out.append("<div class='bar none' title='no data'></div>")
                continue
            segs = "".join(f"<div style='flex:{w};background:var(--c{i})'></div>"
                           for i, w in enumerate(r["weights"]) if w >= 0.005)
            cls = "bar poor" if r["r2"] < 0.5 else "bar"
            out.append(f"<div class='{cls}' title='{attr(tip(r))}'>{segs}</div>")
        out.append(f"<div class='sh'>{shift[t]:.2f}</div></div>")
    out.append("</div>")
    return "".join(out)


def matrix_svg(S, season):
    M = S["matrices"][str(season)]
    rows = {(r["team"], r["season"]): r for r in S["rows"]}
    F = S["features"]
    n, c, pad = len(M["order"]), 20, 36
    idx = {t: i for i, t in enumerate(M["teams"])}
    D = np.array(M["distance"])
    upper = np.sort(D[np.triu_indices(n, 1)])
    # color by this season's own distance range: 5th pct = darkest, 60th pct and up = blank
    lo, hi = upper[int(len(upper) * .05)], upper[int(len(upper) * .6)]
    size = pad + n * c + 4
    parts = [f"<svg class='mx' viewBox='0 0 {size} {size}' role='img' aria-label='Team similarity matrix for {season}'>"]
    for i, t in enumerate(M["order"]):
        parts.append(f"<text x='{pad - 4}' y='{pad + i * c + c / 2 + 3}' text-anchor='end'>{t}</text>")
        parts.append(f"<text transform='translate({pad + i * c + c / 2 + 3} {pad - 4}) rotate(-90)'>{t}</text>")
    for i, a in enumerate(M["order"]):
        for j, b in enumerate(M["order"]):
            x, y = pad + j * c, pad + i * c
            if a == b:
                parts.append(f"<rect x='{x}' y='{y}' width='{c - 1}' height='{c - 1}' rx='2' fill='var(--grid)'/>")
                continue
            d = D[idx[a], idx[b]]
            t = min(1, max(0, (hi - d) / (hi - lo))) * 90
            ra, rb = rows[(a, season)], rows[(b, season)]
            title = f"{a} vs {b}, {season}\nDistance: {d:.2f} (all six features)\n" + \
                "\n".join(f"{LABEL[f]}: {ra['z'][f]:+.2f} vs {rb['z'][f]:+.2f}" for f in F)
            parts.append(f"<rect class='cell' x='{x}' y='{y}' width='{c - 1}' height='{c - 1}' rx='2' "
                         f"fill='color-mix(in oklab, var(--seq) {t:.0f}%, var(--surface-1))'><title>{escape(title)}</title></rect>")
    parts.append("</svg>")
    return "".join(parts)


def matrices(S):
    seasons = S["seasons"]
    radios = "".join(f"<input type='radio' name='yr' id='y{s}' {'checked' if s == seasons[-1] else ''}>" for s in seasons)
    labels = "".join(f"<label for='y{s}'>{s}</label>" for s in seasons)
    svgs = "".join(f"<div class='m m{s}'>{matrix_svg(S, s)}</div>" for s in seasons)
    css = "".join(f"#y{s}:checked ~ .mxs .m{s} {{ display: block; }} #y{s}:checked ~ .controls label[for=y{s}] "
                  f"{{ background: var(--text-primary); color: var(--surface-1); }}" for s in seasons)
    return (f"<style>{css}</style><div class='mxbox'>{radios}"
            f"<div class='controls'>Season <span class='seg'>{labels}</span></div>"
            f"<p class='hint'>Distance between teams on all six features at once (darker = more similar). Ordered so "
            f"similar teams sit together; hover a cell for the pair.</p><div class='mxs'>{svgs}</div></div>")


def shifts_table(S):
    rows = sorted(S["shifts"], key=lambda s: -s["shift"])[:15]
    return ("<table><tr><th class='l'>Team</th><th class='l'>Seasons</th><th>Shift</th>"
            "<th class='l'>From (top archetype)</th><th class='l'>To (top archetype)</th></tr>" +
            "".join(f"<tr><td class='l'>{s['team']}</td><td class='l'>{s['from']} to {s['to']}</td>"
                    f"<td>{s['shift']:.2f}</td><td class='l'>{escape(s['from_top'])}</td><td class='l'>{escape(s['to_top'])}</td></tr>"
                    for s in rows) + "</table>")


def sweep_table(S):
    return ("<table><tr><th>Archetypes</th><th>Variance explained</th><th>Team-seasons fit poorly (R&sup2; &lt; 0.5)</th>"
            "<th>Median top weight</th></tr>" +
            "".join(f"<tr class='{'pick' if r['k'] == S['k'] else ''}'><td>{r['k']}</td><td>{round(r['var_explained'] * 100)}%</td>"
                    f"<td>{r['rows_r2_below_0.5']}</td><td>{r['median_top_weight']}</td></tr>" for r in S["sweep"]) + "</table>")


def full_table(S):
    names = [a["name"] for a in S["archetypes"]]
    rows = sorted(S["rows"], key=lambda r: (r["team"], r["season"]))
    return ("<table><tr><th class='l'>Team</th><th>Season</th><th class='l'>Top archetype</th>" +
            "".join(f"<th>{escape(n)}</th>" for n in names) + "<th>R&sup2;</th><th class='l'>Comps</th></tr>" +
            "".join(f"<tr><td class='l'>{r['team']}</td><td>{r['season']}</td><td class='l'>{escape(r['top_archetype'])}</td>" +
                    "".join(f"<td>{round(w * 100)}%</td>" for w in r["weights"]) +
                    f"<td>{r['r2']}</td><td class='l'>{', '.join(c['key'] for c in r['comps'])}</td></tr>" for r in rows) +
            "</table>")


def main():
    with open(SUMMARY_PATH) as f:
        S = json.load(f)
    names = [a["name"] for a in S["archetypes"]]
    stats = [(S["k"], "archetypes"), (f"{round(S['var_explained'] * 100)}%", "variance explained"),
             (len(S["rows"]), "team-seasons"), (S["rows_r2_below_0.5"], "team-seasons fit poorly"),
             (f"{S['boot_move_median']} sd", "bootstrap archetype movement (median)")]
    loose = [a["name"] for a in S["archetypes"] if a["boot_move_median"] > 1.5]
    caveat = (f"Independent restarts find identical archetypes. Resampling team-seasons moves them a median "
              f"{S['boot_move_median']} sd, because archetypes sit at the extremes and are defined by a few team-seasons"
              + (f"; least stable: {', '.join(loose)}." if loose else "."))

    html = (TEMPLATE
            .replace("__STATS__", "".join(f"<div class='stat'><b>{v}</b><span>{l}</span></div>" for v, l in stats))
            .replace("__ARCH__", archetype_table(S))
            .replace("__LEGEND__", "".join(f"<span><i class='sw' style='background:var(--c{i})'></i>{escape(n)}</span>"
                                           for i, n in enumerate(names)))
            .replace("__MIX__", mix_grid(S))
            .replace("__MATRICES__", matrices(S))
            .replace("__SHIFTS__", shifts_table(S))
            .replace("__SWEEP__", sweep_table(S))
            .replace("__CAVEAT__", escape(caveat))
            .replace("__TABLE__", full_table(S)))
    with open(OUT_PATH, "w") as out:
        out.write(html)
    print(f"wrote {OUT_PATH} ({len(html) // 1024} KB)")


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
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.legend .sw { margin-right: 0; }
.tbl-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12px; width: 100%; }
th, td { padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--grid); }
th.l, td.l { text-align: left; }
th.c { text-align: center; }
th { color: var(--text-secondary); font-weight: 600; }
td.heat { text-align: center; font-variant-numeric: tabular-nums; min-width: 64px; border: 2px solid var(--surface-1); }
.scale { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.scale i { display: inline-block; width: 160px; height: 10px; border-radius: 2px; }
[hidden] { display: none !important; }
.controls { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin: 4px 0 10px; font-size: 12px; color: var(--text-secondary); }
.seg { display: inline-flex; border: 1px solid var(--axis); border-radius: 6px; overflow: hidden; }
.seg button, .seg label { background: none; border: 0; padding: 4px 10px; font: inherit; color: var(--text-secondary); cursor: pointer; }
.seg button + button, .seg label + label { border-left: 1px solid var(--axis); }
.seg button.on { background: var(--text-primary); color: var(--surface-1); }
.mixgrid { display: grid; grid-template-columns: 44px repeat(4, minmax(0, 1fr)) 48px; gap: 3px 6px; align-items: center; font-size: 12px; }
.mixgrid .row { display: contents; }
.mixgrid .hd { color: var(--text-secondary); font-weight: 600; text-align: center; }
.mixgrid .hd.r { text-align: right; }
.mixgrid .tm { font-weight: 600; }
.mixgrid .sh { text-align: right; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.bar { display: flex; height: 16px; gap: 2px; cursor: default; }
.bar div { height: 100%; }
.bar div:first-child { border-radius: 3px 0 0 3px; }
.bar div:last-child { border-radius: 0 3px 3px 0; }
.bar.poor { opacity: .45; }
.bar.none { background: repeating-linear-gradient(45deg, var(--grid) 0 3px, transparent 3px 6px); border-radius: 3px; }
.mxbox > input { position: absolute; opacity: 0; pointer-events: none; }
.mxs { overflow-x: auto; }
.mxs .m { display: none; }
svg.mx { display: block; width: 100%; min-width: 520px; max-width: 760px; height: auto; }
svg.mx text { fill: var(--text-secondary); font-size: 10px; }
svg.mx rect.cell:hover { stroke: var(--text-primary); stroke-width: 1.5; }
.note { color: var(--text-muted); font-size: 12px; margin-top: 8px; max-width: 900px; }
tr.pick td { font-weight: 700; }
details { margin-top: 20px; }
</style>
</head>
<body>
<main>
  <h1>Run game archetypes, 2022-2025</h1>
  <p class="sub">Archetypal analysis on six run-game features (designed runs, neutral game script). Six "pure" run games are found at the edges of the data, and every team-season is described as a mix of them. Features are standardized within each season, so a team is measured against that year's league. All seasons share one set of archetypes, so mixes compare year to year.</p>
  <div class="stats">__STATS__</div>

  <h2>The archetypes</h2>
  <p class="hint">Each archetype's profile in standard deviations from the league average. Anchor = the team-season closest to pure.</p>
  <div class="tbl-wrap">__ARCH__</div>
  <div class="scale"><span>-2</span><i style="background:linear-gradient(90deg,var(--div-neg),var(--div-mid),var(--div-pos))"></i><span>+2 sd</span></div>

  <h2>Every team's mix by season</h2>
  <div class="legend">__LEGEND__</div>
  <div class="controls" id="sortctl" hidden>Sort
    <span class="seg" id="sort"><button data-s="team" class="on">Team</button><button data-s="shift">Total shift</button><button data-s="arch">2025 archetype</button></span>
  </div>
  <p class="hint">Bar = that season's archetype mix; hover a bar for details and nearest comps. Faded bars fit poorly (R&sup2; &lt; 0.5). Shift = total year-over-year change in mix, summed over 2022-25 (each step 0 = same mix, 1 = no overlap).</p>
  __MIX__

  <h2>Team similarity</h2>
  __MATRICES__
  <div class="scale"><span>Most similar pairs</span><i style="background:linear-gradient(90deg,color-mix(in oklab,var(--seq) 90%,var(--surface-1)),var(--surface-1))"></i><span>Median and beyond</span></div>

  <h2>Biggest identity shifts</h2>
  <div class="tbl-wrap">__SHIFTS__</div>

  <h2>How many archetypes</h2>
  <div class="tbl-wrap">__SWEEP__</div>
  <p class="note">__CAVEAT__</p>

  <details>
    <summary>Full team-season table</summary>
    <div class="tbl-wrap">__TABLE__</div>
  </details>
</main>
<script>
// optional enhancement: re-sort the team mixes (the page is complete without it)
(() => {
  const ctl = document.getElementById("sortctl"), grid = document.getElementById("mix");
  if (!ctl || !grid) return;
  ctl.hidden = false;
  const rows = [...grid.querySelectorAll(".row")];
  const cmp = {
    team: (a, b) => a.dataset.team.localeCompare(b.dataset.team),
    shift: (a, b) => b.dataset.shift - a.dataset.shift,
    arch: (a, b) => a.dataset.arch - b.dataset.arch || b.dataset.w - a.dataset.w,
  };
  ctl.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    ctl.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === btn));
    rows.sort(cmp[btn.dataset.s]).forEach(r => grid.appendChild(r));
  }));
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
