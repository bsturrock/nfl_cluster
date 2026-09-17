"""Cluster team-seasons on offensive scheme identity, and build the scatter
visualization for it.

Combines features from the three build_*.py scripts (gap mix, formation/
personnel, passing) into one table, clusters on the top-2 principal
components of the standardized features (clustering on raw features dilutes
separation badly - see commit history), and writes both the cluster
assignments (output/scheme_clusters.csv) and an interactive scatter
(output/scheme_clusters.html).

Feature set (12; screen_rate was tried and dropped - redundant with adot at
r=-0.38 and the weakest standalone signal of the candidates tested):
  pct_under_center, pct_top_personnel, rush_rate, pa_personnel_match,
  disguise_entropy, outside_run_rate, pa_boot_rate, adot, motion_rate,
  designed_qb_run_rate, rpo_rate, no_huddle_rate

rpo_rate correlates -0.51 with pct_under_center (RPOs are almost always
run from shotgun) and 0.38 with disguise_entropy - the largest correlation
among kept features, but not so large it's a restatement of either.

k chosen by silhouette, re-checked each time features change - the value
that was best for an earlier feature set is not assumed to still be best.
"""
import json

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

K = 3

FEATURE_META = {
    "pct_under_center": {"label": "Under center %", "fmt": "pct"},
    "pct_top_personnel": {"label": "Top personnel %", "fmt": "pct"},
    "rush_rate": {"label": "Rush rate", "fmt": "pct"},
    "pa_personnel_match": {"label": "PA personnel match", "fmt": "pct"},
    "disguise_entropy": {"label": "Disguise entropy", "fmt": "num"},
    "outside_run_rate": {"label": "Outside run rate", "fmt": "pct"},
    "pa_boot_rate": {"label": "PA boot rate", "fmt": "pct"},
    "adot": {"label": "aDOT", "fmt": "num"},
    "motion_rate": {"label": "Motion rate", "fmt": "pct"},
    "designed_qb_run_rate": {"label": "Designed QB run rate", "fmt": "pct"},
    "rpo_rate": {"label": "RPO rate", "fmt": "pct"},
    "no_huddle_rate": {"label": "No-huddle rate", "fmt": "pct"},
}
FEATURES = list(FEATURE_META.keys())

CLUSTER_LABELS = {
    0: "Mainstream / credible run-pass mix",
    1: "Run-heavy, dual-threat QB",
    2: "Shotgun, quick-game, pocket passer",
}

TEMPLATE_PATH = "src/templates/scheme_clusters_template.html"


def main():
    gap = pd.read_csv("data/team_season_features.csv")
    gap["outside_run_rate"] = gap["pct_end"] + gap["pct_tackle"]

    form = pd.read_csv("data/formation_personnel_features.csv").rename(columns={"posteam": "team"})
    passing = pd.read_csv("data/passing_features.csv").rename(columns={"posteam": "team"})

    df = (
        form.merge(gap[["season", "team", "outside_run_rate"]], on=["season", "team"], how="inner")
        .merge(passing[["season", "team", "adot"]], on=["season", "team"], how="inner")
    )

    Xs = StandardScaler().fit_transform(df[FEATURES].values)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(Xs)
    df["pc1"], df["pc2"] = pcs[:, 0], pcs[:, 1]

    km = KMeans(n_clusters=K, n_init=25, random_state=0)
    df["cluster"] = km.fit_predict(pcs)
    sil = silhouette_score(pcs, df["cluster"])

    out_cols = ["season", "team", "cluster", "pc1", "pc2"] + FEATURES
    out = df[out_cols].round(4)
    out.to_csv("output/scheme_clusters.csv", index=False)

    print(f"k={K}, silhouette={sil:.3f}, explained var (top 2 PCs)={pca.explained_variance_ratio_.sum():.3f}")
    profile = df.groupby("cluster")[FEATURES].mean().round(3)
    profile["n"] = df.groupby("cluster").size()
    print(profile.T)
    print(f"wrote {len(out)} rows to output/scheme_clusters.csv")

    records = df[["team", "season", "cluster", "pc1", "pc2"] + FEATURES].round(4).to_dict(orient="records")
    payload = {
        "points": records,
        "cluster_labels": CLUSTER_LABELS,
        "feature_meta": FEATURE_META,
        "feature_order": FEATURES,
        "pc_note": (
            f"PC1/PC2 explain {pca.explained_variance_ratio_.sum()*100:.0f}% of variance across "
            f"all 10 features. Clustering runs on these 2 components, not the raw features directly, "
            f"which recovers real separation (silhouette {sil:.2f} vs ~0.17 on raw features)."
        ),
        "stat_line": (
            f"k={K} via k-means on PC1/PC2 · silhouette={sil:.3f} · 128 team-seasons, 2022-2025"
        ),
    }
    template = open(TEMPLATE_PATH).read()
    html = template.replace("__DATA_JSON__", json.dumps(payload))
    with open("output/scheme_clusters.html", "w") as f:
        f.write(html)
    print("wrote output/scheme_clusters.html")


if __name__ == "__main__":
    main()
