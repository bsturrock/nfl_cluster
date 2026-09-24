"""Build the interactive identity scatter (output/identity/identity_scatter.html)
from the cluster_identity.py outputs."""
import json

import pandas as pd

OUT = "output/identity"
TEMPLATE = "src/templates/identity_scatter_template.html"


def main():
    t = pd.read_csv(f"{OUT}/team_season_identity.csv")
    features = pd.read_csv(f"{OUT}/feature_reliability.csv")["feature"]
    features = [f for f in features if f in t.columns]
    z = t.groupby("season")[features].transform(lambda s: (s - s.mean()) / s.std()).round(2)

    points = []
    for i, r in t.iterrows():
        points.append({
            "t": r.team, "s": int(r.season), "n": int(r.n_plays),
            "c5": int(r.cluster_k5), "c7": int(r.cluster_k7),
            "sil5": r.silhouette_k5, "sil7": r.silhouette_k7,
            "m5": r.margin_k5, "m7": r.margin_k7,
            "ru5": r.runner_up_k5, "ru7": r.runner_up_k7,
            "pc": [r[f"PC{j}"] for j in range(1, 6)],
            "f": [round(float(r[f]), 4) for f in features],
            "z": [float(v) for v in z.loc[i]],
        })
    profiles = {}
    for k in (5, 7):
        p = pd.read_csv(f"{OUT}/cluster_profiles_k{k}.csv", index_col=0)
        cols = [f"z_c{c}" for c in range(k)]
        profiles[k] = {
            "labels": [p.loc["label", c] for c in cols],
            "n": [int(p.loc["n", c]) for c in cols],
            "sil": [float(p.loc["mean_silhouette", c]) for c in cols],
            "z": [[float(p.loc[f, c]) for f in features] for c in cols],
        }
    ev = pd.read_csv(f"{OUT}/pca_loadings.csv", index_col=0).loc["explained_var_ratio"].tolist()

    payload = {"features": features, "points": points, "profiles": profiles, "ev": ev}
    html = open(TEMPLATE).read().replace("__DATA_JSON__", json.dumps(payload))
    with open(f"{OUT}/identity_scatter.html", "w") as f:
        f.write(html)
    print(f"wrote {OUT}/identity_scatter.html")


if __name__ == "__main__":
    main()
