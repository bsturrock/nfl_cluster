"""Build the interactive cluster visualization from output/team_season_clusters.csv.

Computes a 2D PCA projection of the clustering features for plotting, injects
it into the HTML template, and writes output/cluster_viz.html.
"""
import json

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURES = ["pct_end", "pct_tackle", "pct_guard", "pa_rate"]

CLUSTER_LABELS = {
    0: "Balanced interior, high PA",
    1: "Edge-leaning, low PA",
    2: "Guard-heavy, low PA",
    3: "Off-tackle, high PA",
}

TEMPLATE_PATH = "src/templates/cluster_viz_template.html"
OUT_PATH = "output/cluster_viz.html"


def main():
    df = pd.read_csv("output/team_season_clusters.csv")

    Xs = StandardScaler().fit_transform(df[FEATURES].values)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(Xs)
    df["pc1"] = pcs[:, 0]
    df["pc2"] = pcs[:, 1]
    df["label"] = df["cluster"].map(CLUSTER_LABELS)

    cols = [
        "team", "season", "cluster", "label", "pc1", "pc2",
        "pct_end", "pct_tackle", "pct_guard", "pct_middle", "pa_rate",
        "rush_n", "dropback_n",
    ]
    records = df[cols].round(4).to_dict(orient="records")
    data_json = json.dumps({"points": records})

    template = open(TEMPLATE_PATH).read()
    html = template.replace("__DATA_JSON__", data_json)
    with open(OUT_PATH, "w") as f:
        f.write(html)

    print(f"explained variance ratio: {pca.explained_variance_ratio_.round(3).tolist()}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
