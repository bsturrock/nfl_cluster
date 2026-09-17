"""Cluster team-seasons on run gap mix + play-action rate.

Features used (v1): pct_end, pct_tackle, pct_guard, pct_middle, pa_rate
pct_* are compositional (sum to 1) so we drop one (pct_middle) to avoid
collinearity, matching standard practice for compositional data in k-means.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

FEATURES = ["pct_end", "pct_tackle", "pct_guard", "pa_rate"]


def main():
    df = pd.read_csv("data/team_season_features.csv")

    X = df[FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print("k, inertia, silhouette")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=25, random_state=0)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        print(f"{k}, {km.inertia_:.1f}, {sil:.3f}")

    # fit final model
    K = 4
    km = KMeans(n_clusters=K, n_init=25, random_state=0)
    df["cluster"] = km.fit_predict(Xs)

    profile = df.groupby("cluster")[FEATURES + ["pct_middle"]].mean().round(3)
    profile["n"] = df.groupby("cluster").size()
    print(f"\ncluster profiles (k={K}):")
    print(profile)

    df.sort_values(["cluster", "season", "team"]).to_csv(
        "output/team_season_clusters.csv", index=False
    )
    print("\nwrote output/team_season_clusters.csv")


if __name__ == "__main__":
    main()
