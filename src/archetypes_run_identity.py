"""Archetypal analysis of run-game identity, one row per team-season (2022-2025).

Instead of forcing each team into one cluster, archetypal analysis (Cutler &
Breiman 1994; PCHA algorithm, Morup & Hansen 2012) finds k extreme "pure"
run games on the edge of the data, and describes every team-season as a
convex mix of them (weights >= 0, summing to 1). Archetypes are themselves
convex mixes of real team-seasons, so they never sit in empty space.

Grain is team-season: identities change with coaching and personnel, so each
season gets its own mix. All seasons are fit together so the archetypes are
one shared vocabulary and a team's mix can be compared year to year.

Features (build_run_features.py; designed runs, neutral script):
  run_dir, shotgun_share, te_minus_backs, motion_rate, rpo_rate, qb_run_share
Standardized WITHIN season: league-wide drift (motion rising, RPO charting
shifting year to year, up to ~1 sd in season means) would otherwise make the
archetypes partly encode the year. So identity = relative to that season's league.

k=6: variance explained 0.34/0.53/0.64/0.77/0.83/0.86/0.88 for k=2..8; the
elbow is at 6 (~3 pts per extra archetype after), with 5 of 128 team-seasons
at R^2 < 0.5.

Also: per-season team-to-team distance matrices (Euclidean on the z-scores,
i.e. all six features at once) with optimal leaf ordering, each team-season's
nearest comps among other teams, and year-over-year identity shift.

Output: output/run_archetypes.csv, output/run_archetypes_summary.json
"""
import json

import numpy as np
import pandas as pd
from py_pcha import PCHA
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "data/run_play_features.csv"
CSV_PATH = "output/run_archetypes.csv"
SUMMARY_PATH = "output/run_archetypes_summary.json"

FEATURES = ["run_dir", "shotgun_share", "te_minus_backs", "motion_rate", "rpo_rate", "qb_run_share"]
K = 6
K_RANGE = range(2, 9)
N_RESTARTS = 20
N_BOOT = 50
N_COMPS = 3

# archetypes are named via an anchor team-season that is (nearly) pure in
# that archetype, so a reordering between runs can't silently swap names.
# Order = display order.
ARCHETYPE_NAMES = [
    (("LA", 2025), "Under-center TE motion"),
    (("ARI", 2025), "Under-center outside, static"),
    (("LV", 2022), "Downhill under center"),
    (("MIA", 2025), "Motion stretch"),
    (("CIN", 2023), "Gun RPO inside"),
    (("PHI", 2023), "QB-run spread"),
]


def load():
    d = pd.read_csv(FEATURES_PATH).sort_values(["season", "team"]).reset_index(drop=True)
    d["run_dir"] = d["pct_outside"] - d["pct_inside"]
    d["shotgun_share"] = d["pct_shotgun_or_pistol"]
    d["key"] = d["team"] + " " + d["season"].astype(str)
    return d


def zscore_within_season(d):
    return d.groupby("season")[FEATURES].transform(lambda s: (s - s.mean()) / s.std(ddof=0)).to_numpy()


def fit(X, k, restarts=N_RESTARTS, seed0=0):
    """Best-of-restarts PCHA. Returns archetypes (k x p), weights (n x k), var explained."""
    best, s, done = None, 0, 0
    while done < restarts:
        np.random.seed(seed0 + s)
        s += 1
        try:
            XC, S, _, sse, ve = PCHA(X.T, noc=k, delta=0)
        except Exception:
            # py_pcha's furthest-sum init can draw an out-of-range start index
            # (off-by-one), more often with duplicate rows from bootstrapping
            continue
        done += 1
        if best is None or sse < best[2]:
            best = (np.asarray(XC).T, np.asarray(S).T, sse, ve)
    return best[0], best[1], float(best[3])


def main():
    d = load()
    X = zscore_within_season(d)
    n = len(X)

    sweep = []
    for k in K_RANGE:
        Z, W, ve = fit(X, k)
        r2 = 1 - ((X - W @ Z) ** 2).sum(1) / (X ** 2).sum(1)
        sweep.append({"k": k, "var_explained": round(ve, 3), "rows_r2_below_0.5": int((r2 < 0.5).sum()),
                      "median_top_weight": round(float(np.median(W.max(1))), 2)})

    Z, W, ve = fit(X, K)
    row_of = {(t, s): i for i, (t, s) in enumerate(zip(d["team"], d["season"]))}
    order = [int(np.argmax(W[row_of[anchor]])) for anchor, _ in ARCHETYPE_NAMES]
    assert len(set(order)) == K, f"anchors no longer map to distinct archetypes ({order}); rename"
    Z, W = Z[order], W[:, order]
    names = [name for _, name in ARCHETYPE_NAMES]
    row_r2 = 1 - ((X - W @ Z) ** 2).sum(1) / (X ** 2).sum(1)

    # stability 1: do independent restarts land on the same archetypes?
    # stability 2: bootstrap team-seasons, refit, match archetypes (Hungarian), z-distance moved
    def matched_dist(Za, Zb):
        cost = ((Za[:, None, :] - Zb[None, :, :]) ** 2).sum(2) ** 0.5
        r, c = linear_sum_assignment(cost)
        return cost[r, c]

    restart_moves = [matched_dist(Z, fit(X, K, restarts=5, seed0=1000 + 5 * s)[0]) for s in range(10)]
    rng = np.random.default_rng(0)
    boot_moves = np.array([
        matched_dist(Z, fit(X[rng.choice(n, n, replace=True)], K, restarts=5, seed0=2000 + 5 * b)[0])
        for b in range(N_BOOT)
    ])

    # comps: nearest team-seasons of OTHER teams, any season, on all six features
    D = squareform(pdist(X))
    rows = []
    for i, r in d.iterrows():
        comps = [j for j in np.argsort(D[i]) if d["team"][j] != r["team"]][:N_COMPS]
        rows.append({
            "team": r["team"],
            "season": int(r["season"]),
            "weights": [round(float(w), 3) for w in W[i]],
            "top_archetype": names[int(np.argmax(W[i]))],
            "r2": round(float(row_r2[i]), 2),
            "raw": {f: round(float(r[f]), 4) for f in FEATURES},
            "z": {f: round(float(X[i, j]), 2) for j, f in enumerate(FEATURES)},
            "comps": [{"key": d["key"][j], "dist": round(float(D[i, j]), 2)} for j in comps],
        })

    # year-over-year identity shift: total-variation distance between a team's
    # consecutive mixes (0 = same mix, 1 = no overlap)
    shifts = []
    for team, g in d.groupby("team"):
        g = g.sort_values("season")
        for (i, a), (j, b) in zip(g.iterrows(), list(g.iterrows())[1:]):
            shifts.append({"team": team, "from": int(a["season"]), "to": int(b["season"]),
                           "shift": round(float(0.5 * np.abs(W[i] - W[j]).sum()), 2),
                           "from_top": names[int(np.argmax(W[i]))], "to_top": names[int(np.argmax(W[j]))]})

    # per-season distance matrices with an optimal leaf ordering for display
    matrices = {}
    for season, g in d.groupby("season"):
        idx = g.index.to_numpy()
        dv = pdist(X[idx])
        leaf = leaves_list(optimal_leaf_ordering(linkage(dv, "average"), dv))
        matrices[int(season)] = {
            "teams": d["team"][idx].tolist(),
            "order": [d["team"][idx[i]] for i in leaf],
            "distance": [[round(float(v), 2) for v in row] for row in squareform(dv)],
        }

    archetypes = []
    for a, name in enumerate(names):
        top = np.argsort(-W[:, a])[:6]
        archetypes.append({
            "name": name,
            "anchor": " ".join(map(str, ARCHETYPE_NAMES[a][0])),
            "z": {f: round(float(Z[a, j]), 2) for j, f in enumerate(FEATURES)},
            "top": [{"key": d["key"][t], "weight": round(float(W[t, a]), 2)} for t in top],
            "share_by_season": {int(s): round(float(W[g.index, a].mean()), 3) for s, g in d.groupby("season")},
            "boot_move_median": round(float(np.median(boot_moves[:, a])), 2),
        })

    summary = {
        "seasons": sorted(int(s) for s in d["season"].unique()),
        "features": FEATURES,
        "feature_correlations": pd.DataFrame(X, columns=FEATURES).corr().round(2).to_dict(),
        "k": K,
        "var_explained": round(ve, 3),
        "rows_r2_below_0.5": int((row_r2 < 0.5).sum()),
        "restart_move_max": round(float(np.max(restart_moves)), 2),
        "boot_move_median": round(float(np.median(boot_moves)), 2),
        "n_boot": N_BOOT,
        "sweep": sweep,
        "archetypes": archetypes,
        "rows": rows,
        "shifts": shifts,
        "matrices": matrices,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f)

    out = []
    for r in rows:
        o = {"team": r["team"], "season": r["season"], "top_archetype": r["top_archetype"], "r2": r["r2"]}
        o.update({f"w_{nm}": w for nm, w in zip(names, r["weights"])})
        o.update({f: r["raw"][f] for f in FEATURES})
        o["comps"] = ", ".join(c["key"] for c in r["comps"])
        out.append(o)
    pd.DataFrame(out).sort_values(["team", "season"]).to_csv(CSV_PATH, index=False)

    pd.set_option("display.width", 220)
    print(f"wrote {CSV_PATH}, {SUMMARY_PATH}")
    print(pd.DataFrame(sweep).to_string(index=False))
    print(f"\nk={K}: var explained {ve:.3f}; {summary['rows_r2_below_0.5']} rows R2<0.5; restarts move archetypes "
          f"at most {summary['restart_move_max']} sd; bootstrap median move {summary['boot_move_median']} sd")
    print("\narchetypes (z)")
    print(pd.DataFrame([{"archetype": a["name"], **a["z"], "boot_move": a["boot_move_median"],
                         "top": ", ".join(f'{t["key"]}({t["weight"]})' for t in a["top"][:4])}
                        for a in archetypes]).to_string(index=False))
    print("\narchetype share by season")
    print(pd.DataFrame({a["name"]: a["share_by_season"] for a in archetypes}).round(2).to_string())
    print("\nbiggest identity shifts")
    print(pd.DataFrame(shifts).sort_values("shift", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
