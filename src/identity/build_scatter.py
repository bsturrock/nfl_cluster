"""Build the interactive identity scatter (output/identity/identity_scatter.html)
from the cluster_identity.py outputs."""
import json
import sys

import pandas as pd

sys.path.insert(0, "src/identity")
from cluster_identity import K_PRIMARY, K_SECONDARY  # noqa: E402
from themes import THEME_LABELS, THEME_POLES, THEMES  # noqa: E402

OUT = "output/identity"
TEMPLATE = "src/templates/identity_scatter_template.html"


def main():
    t = pd.read_csv(f"{OUT}/team_season_identity.csv")
    rel = pd.read_csv(f"{OUT}/feature_reliability.csv")
    features = [f for f in rel["feature"] if f in t.columns and not f.startswith("theme_")]
    theme_cols = [f"theme_{n}" for n in THEMES]
    z = t.groupby("season")[features].transform(lambda s: (s - s.mean()) / s.std()).round(2)

    points = []
    for i, r in t.iterrows():
        pt = {
            "t": r.team, "s": int(r.season), "n": int(r.n_plays),
            "th": [round(float(r[c]), 3) for c in theme_cols],
            "f": [round(float(r[f]), 4) for f in features],
            "z": [float(v) for v in z.loc[i]],
        }
        for k in (K_PRIMARY, K_SECONDARY):
            pt.update({f"c{k}": int(r[f"cluster_k{k}"]), f"sil{k}": r[f"silhouette_k{k}"],
                       f"m{k}": r[f"margin_k{k}"], f"ru{k}": r[f"runner_up_k{k}"]})
        points.append(pt)

    profiles = {}
    for k in (K_PRIMARY, K_SECONDARY):
        p = pd.read_csv(f"{OUT}/cluster_profiles_k{k}.csv", index_col=0)
        cols = [f"z_c{c}" for c in range(k)]
        profiles[k] = {
            "labels": [p.loc["label", c] for c in cols],
            "n": [int(p.loc["n", c]) for c in cols],
            "sil": [float(p.loc["mean_silhouette", c]) for c in cols],
            "z": [[float(p.loc[f, c]) for f in features] for c in cols],
            "tz": [[float(p.loc[n, c]) for n in THEMES] for c in cols],
        }

    ksel = pd.read_csv(f"{OUT}/k_selection.csv")
    row = ksel[(ksel["method"] == "kmeans") & (ksel["k"] == K_PRIMARY)].iloc[0]
    lab = t[["season", "team", f"cluster_k{K_PRIMARY}"]]
    pairs = lab.merge(lab.assign(season=lab["season"] - 1), on=["season", "team"])
    same = (pairs.iloc[:, 2] == pairs.iloc[:, 3]).mean()
    chance = (t[f"cluster_k{K_PRIMARY}"].value_counts(normalize=True) ** 2).sum()
    stats = [
        ["team-seasons", str(len(t))],
        ["features", f"{len(features)} in {len(THEMES)} themes"],
        [f"silhouette k={K_PRIMARY}", f"{row.silhouette:.3f}"],
        ["random-data baseline", f"{row.null_silhouette_mean:.3f} ({row.silhouette_z_vs_null:.1f} SD above)"],
        ["same cluster next season", f"{same:.1%} (chance {chance:.1%})"],
    ]

    themes = [{"key": n, "label": THEME_LABELS[n], "lo": THEME_POLES[n][0], "hi": THEME_POLES[n][1],
               "features": list(THEMES[n])} for n in THEMES]
    payload = {"features": features, "points": points, "profiles": profiles, "themes": themes,
               "stats": stats, "k": [K_PRIMARY, K_SECONDARY]}
    html = open(TEMPLATE).read().replace("__DATA_JSON__", json.dumps(payload))
    with open(f"{OUT}/identity_scatter.html", "w") as f:
        f.write(html)
    print(f"wrote {OUT}/identity_scatter.html")


if __name__ == "__main__":
    main()
