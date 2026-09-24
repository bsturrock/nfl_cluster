"""Cluster 2025 offenses on the run-identity features built in
build_run_features.py (designed non-QB runs, neutral script):

  - run_dir:        outside (end + tackle) minus inside (guard + middle) share
  - shotgun_share:  share of runs from shotgun or pistol
  - te_minus_backs: avg TEs minus avg backs (RB + FB) per run
  - motion_rate:    share of runs with pre-snap motion (FTN)

Avg backs / avg TEs are left out as separate inputs because te_minus_backs
replaces them (their PC1, 83% of the variance). `heavy` is also left out:
it adds no separation (silhouette flat) and correlates with run_dir (0.43).

Standardized, KMeans. The four features are nearly uncorrelated (|r| <=
0.28), so the teams form one diffuse 4D cloud: silhouette stays ~0.19-0.27
at every k, and k >= 7 starts producing singletons. k=6 is the largest
k with no singletons and gives interpretable groups; stability is only moderate
(see the bootstrap ARI in the output).

Output: output/run_identity_clusters_2025.csv,
        output/run_identity_summary_2025.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

SEASON = 2025
FEATURES_PATH = "data/run_play_features.csv"
CSV_PATH = "output/run_identity_clusters_2025.csv"
SUMMARY_PATH = "output/run_identity_summary_2025.json"

FEATURES = ["run_dir", "shotgun_share", "te_minus_backs", "motion_rate"]
K = 6
K_RANGE = range(2, 11)
N_BOOT = 200

# named by a member team, not KMeans index, so relabeling can't swap names
CLUSTER_NAMES = [
    ("SEA", "Under center, outside, low motion"),
    ("DET", "Under center, TE-heavy, motion"),
    ("SF", "Fullback, inside, motion"),
    ("PHI", "Gun, inside, TE-heavy, static"),
    ("CIN", "Heavy gun, heavy inside, motion"),
    ("ATL", "Gun, stretch outside, heavy motion"),
]


def load():
    d = pd.read_csv(FEATURES_PATH)
    d = d[d["season"] == SEASON].reset_index(drop=True)
    d["run_dir"] = d["pct_outside"] - d["pct_inside"]
    d["shotgun_share"] = d["pct_shotgun_or_pistol"]
    return d


def main():
    d = load()
    X = StandardScaler().fit_transform(d[FEATURES])

    sweep = []
    for k in K_RANGE:
        labels = KMeans(k, n_init=100, random_state=0).fit_predict(X)
        ward = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        sweep.append({
            "k": k,
            "silhouette": round(float(silhouette_score(X, labels)), 3),
            "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
            "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 3),
            "ward_silhouette": round(float(silhouette_score(X, ward)), 3),
            "ari_vs_ward": round(float(adjusted_rand_score(labels, ward)), 2),
            "sizes": sizes,
            "singletons": sizes.count(1),
        })

    labels = KMeans(K, n_init=100, random_state=0).fit_predict(X)
    rng = np.random.default_rng(0)
    boot = [
        adjusted_rand_score(labels, KMeans(K, n_init=10, random_state=i).fit(X[rng.choice(len(X), len(X))]).predict(X))
        for i in range(N_BOOT)
    ]

    idx = {t: i for i, t in enumerate(d["team"])}
    name_of = {labels[idx[t]]: (order, name) for order, (t, name) in enumerate(CLUSTER_NAMES)}
    assert len(name_of) == K, "anchor teams no longer land in distinct clusters; rename"
    d["cluster"] = [name_of[c][0] for c in labels]
    d["cluster_name"] = [name_of[c][1] for c in labels]
    d["silhouette"] = silhouette_samples(X, labels)
    Z = pd.DataFrame(X, columns=[f"z_{f}" for f in FEATURES])
    d = pd.concat([d, Z], axis=1)

    # which features drive the partition: one-way ANOVA F of each feature across clusters
    groups = [d.loc[d["cluster"] == c] for c in range(K)]
    feature_f = {f: round(float(f_oneway(*[g[f] for g in groups]).statistic), 1) for f in FEATURES}

    profiles = []
    for c, (_, name) in enumerate(CLUSTER_NAMES):
        g = groups[c]
        profiles.append({
            "cluster": name,
            "n": len(g),
            "teams": sorted(g["team"]),
            "mean_silhouette": round(float(g["silhouette"].mean()), 2),
            "raw": {f: round(float(g[f].mean()), 3) for f in FEATURES},
            "z": {f: round(float(g[f"z_{f}"].mean()), 2) for f in FEATURES},
        })

    summary = {
        "season": SEASON,
        "n_teams": len(d),
        "features": FEATURES,
        "feature_correlations": d[FEATURES].corr().round(2).to_dict(),
        "k": K,
        "silhouette": next(r["silhouette"] for r in sweep if r["k"] == K),
        "bootstrap_ari_mean": round(float(np.mean(boot)), 2),
        "bootstrap_ari_p10": round(float(np.percentile(boot, 10)), 2),
        "feature_anova_f": feature_f,
        "sweep": sweep,
        "clusters": profiles,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    cols = ["team", "cluster_name", "silhouette"] + FEATURES + [f"z_{f}" for f in FEATURES]
    d.sort_values(["cluster", "team"])[cols].round(3).to_csv(CSV_PATH, index=False)

    print(f"wrote {CSV_PATH}, {SUMMARY_PATH}")
    print("\nk sweep")
    print(pd.DataFrame(sweep).to_string(index=False))
    print(f"\nk={K}: silhouette {summary['silhouette']}, bootstrap ARI mean {summary['bootstrap_ari_mean']} "
          f"(p10 {summary['bootstrap_ari_p10']}, n={N_BOOT})")
    print("feature ANOVA F across clusters:", feature_f)
    print("\ncluster profiles (z-scores)")
    print(pd.DataFrame([{"cluster": p["cluster"], "n": p["n"], **p["z"], "sil": p["mean_silhouette"],
                         "teams": " ".join(p["teams"])} for p in profiles]).to_string(index=False))
    print("\ncluster profiles (raw)")
    print(pd.DataFrame([{"cluster": p["cluster"], **p["raw"]} for p in profiles]).to_string(index=False))


if __name__ == "__main__":
    main()
