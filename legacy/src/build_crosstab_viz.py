"""Cross-tab the gap/PA cluster (k=4, from cluster.py) against a formation/
personnel cluster (k=3, fit here) and render as a heatmap.

Kept separate from cluster.py because merging both feature families into one
k-means run dilutes separation (silhouette ~0.17 vs 0.24 / 0.33 apart) -
cross-tabbing keeps both signals legible instead of averaging them into mush.

Output: output/crosstab_viz.html
"""
import json

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FORM_FEATURES = ["pct_under_center", "pct_pistol", "pct_pers_11", "pct_pers_12", "n_personnel_groups_5plus"]
FORM_K = 3
FORM_LABELS = {0: "Mainstream mix", 1: "Pistol-heavy", 2: "Shotgun/11-personnel monoculture"}

GAP_LABELS = {
    0: "Balanced interior, high PA",
    1: "Edge-leaning, low PA",
    2: "Guard-heavy, low PA",
    3: "Off-tackle, high PA",
}
GAP_ORDER = list(GAP_LABELS.values())

TEMPLATE_PATH = "src/templates/crosstab_viz_template.html"
OUT_PATH = "output/crosstab_viz.html"


def main():
    gap = pd.read_csv("output/team_season_clusters.csv")
    gap["gap_label"] = gap["cluster"].map(GAP_LABELS)

    form = pd.read_csv("data/formation_personnel_features.csv").rename(columns={"posteam": "team"})
    Xf = StandardScaler().fit_transform(form[FORM_FEATURES].values)
    km_form = KMeans(n_clusters=FORM_K, n_init=25, random_state=0)
    form["form_cluster"] = km_form.fit_predict(Xf)
    form["form_label"] = form["form_cluster"].map(FORM_LABELS)
    form_order = [FORM_LABELS[i] for i in range(FORM_K)]

    df = gap.merge(form[["season", "team", "form_label"]], on=["season", "team"], how="inner")

    ct = pd.crosstab(df["gap_label"], df["form_label"]).reindex(index=GAP_ORDER, columns=form_order)
    row_pct = ct.div(ct.sum(axis=1), axis=0)
    col_pct = ct.div(ct.sum(axis=0), axis=1)

    chi2, p, dof, _ = chi2_contingency(ct)
    n = int(ct.values.sum())
    cramers_v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5

    cells = []
    for g in GAP_ORDER:
        for f in form_order:
            cells.append({
                "gap": g,
                "form": f,
                "count": int(ct.loc[g, f]),
                "row_pct": round(float(row_pct.loc[g, f]), 4),
                "col_pct": round(float(col_pct.loc[g, f]), 4),
            })

    payload = {
        "gap_order": GAP_ORDER,
        "form_order": form_order,
        "gap_n": {k: int(v) for k, v in ct.sum(axis=1).to_dict().items()},
        "form_n": {k: int(v) for k, v in ct.sum(axis=0).to_dict().items()},
        "cells": cells,
        "chi2": round(float(chi2), 2),
        "p": round(float(p), 4),
        "dof": int(dof),
        "cramers_v": round(float(cramers_v), 3),
        "n_total": n,
    }

    template = open(TEMPLATE_PATH).read()
    html = template.replace("__DATA_JSON__", json.dumps(payload))
    with open(OUT_PATH, "w") as f:
        f.write(html)

    print(f"chi2={chi2:.2f} p={p:.4f} cramers_v={cramers_v:.3f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
