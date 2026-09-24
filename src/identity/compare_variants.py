"""Compare alternative feature treatments against the current identity model.

Variants (all k-means, scored the same way as cluster_identity.py: silhouette
vs the copula null (same rank correlations and marginals) and vs a Gaussian
null, bootstrap ARI, year-over-year persistence, and ARI against the model's
k=5 assignments):

  pca_19           19 within-season z-scores -> PCA (parallel analysis);
                   the model's original approach
  skew_fixed       same, but log(x + 0.01) on the six zero-inflated features
                   before z-scoring
  themes           19 features collapsed into 7 fixed, equal-weight themes
  themes_refined   themes minus the two features that don't fit their theme
                   (extra_ol, screen: item-rest r < 0.15); now the model
                   (src/identity/themes.py)
  binned_3         each z-score cut into 3 levels (+-0.43, equal thirds)

Output: output/identity/variant_comparison.csv, output/identity/theme_consistency.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore")

import sys  # noqa: E402
sys.path.insert(0, "src/identity")
from cluster_identity import copula_null, gaussian_null, select_features  # noqa: E402
from themes import THEMES as THEMES_REFINED  # noqa: E402

# "themes" = the model's themes plus the two features that fit no theme
THEMES = {n: dict(w) for n, w in THEMES_REFINED.items()}
THEMES["dropback_play_action"]["screen"] = -1
THEMES["te_heavy"]["extra_ol"] = 1
OUT = "output/identity"
SKEWED = ["pistol_of_gun", "avg_backs", "extra_ol", "no_huddle", "qb_run_share", "rpo"]


def main():
    rng = np.random.default_rng(0)
    t = pd.read_csv(f"{OUT}/team_season_identity.csv")
    feats = select_features(pd.read_csv("data/identity_features.csv"))[0]
    zs = lambda df: df.groupby(t["season"]).transform(lambda s: (s - s.mean()) / s.std())

    z_raw = zs(t[feats])
    logd = t[feats].copy()
    for f in SKEWED:
        logd[f] = np.log(logd[f] + 0.01)
    z_log = zs(logd)

    consistency = []

    def themes(Z, spec, tag):
        out = {}
        for name, w in spec.items():
            sub = Z[list(w)] * pd.Series(w)
            out[name] = sub.mean(axis=1)
            if len(w) > 1:
                k = len(w)
                alpha = k / (k - 1) * (1 - sub.var().sum() / sub.sum(axis=1).var())
                for f in w:
                    rest = sub.drop(columns=f).mean(axis=1)
                    consistency.append({"variant": tag, "theme": name, "feature": f,
                                        "item_rest_r": np.corrcoef(sub[f], rest)[0, 1], "theme_alpha": alpha})
        return zs(pd.DataFrame(out)).values

    def n_pcs(Z):
        real = PCA().fit(Z).explained_variance_
        null = np.percentile([PCA().fit(R / R.std(0, ddof=1)).explained_variance_
                              for R in (rng.standard_normal(Z.shape) for _ in range(200))], 95, axis=0)
        above = real > null
        return max(2, len(above) if above.all() else int(np.argmin(above)))

    def pca(Z):
        return PCA(n_pcs(Z)).fit_transform(Z)

    variants = {
        "pca_19": pca(z_raw.values),
        "skew_fixed": pca(z_log.values),
        "themes": themes(z_raw, THEMES, "themes"),
        "themes_refined": themes(z_raw, THEMES_REFINED, "themes_refined"),
        "binned_3": pca(np.where(z_raw.values > 0.43, 1.0, np.where(z_raw.values < -0.43, -1.0, 0.0))),
    }

    def yoy(lab):
        d = t.assign(c=lab)[["season", "team", "c"]]
        m = d.merge(d.assign(season=d["season"] - 1), on=["season", "team"])
        return (m["c_x"] == m["c_y"]).mean()

    rows = []
    for name, X in variants.items():
        for k in range(3, 10):
            lab = KMeans(k, n_init=50, random_state=0).fit_predict(X)
            sil = silhouette_score(X, lab)
            nulls = {}
            for tag, draw in (("copula", copula_null), ("gaussian", gaussian_null)):
                sils = []
                for b in range(30):
                    if name == "binned_3":  # draw on the z-scores, then bin + PCA exactly like the data
                        Nz = draw(z_raw.values, rng)
                        N = PCA(X.shape[1]).fit_transform(np.where(Nz > 0.43, 1.0, np.where(Nz < -0.43, -1.0, 0.0)))
                    else:
                        N = draw(X, rng)
                    sils.append(silhouette_score(N, KMeans(k, n_init=10, random_state=b).fit_predict(N)))
                nulls[tag] = (np.mean(sils), np.std(sils))
            boot = [adjusted_rand_score(lab, KMeans(k, n_init=10, random_state=b)
                                        .fit(X[rng.choice(len(X), len(X))]).predict(X)) for b in range(40)]
            rows.append({
                "variant": name, "dims": X.shape[1], "k": k, "silhouette": sil,
                "copula_null_mean": nulls["copula"][0],
                "silhouette_z_vs_null": (sil - nulls["copula"][0]) / nulls["copula"][1],
                "silhouette_z_vs_gaussian": (sil - nulls["gaussian"][0]) / nulls["gaussian"][1],
                "bootstrap_ari": np.mean(boot), "yoy_same_cluster": yoy(lab),
                "ari_vs_model_k5": adjusted_rand_score(t["cluster_k5"], lab),
                "min_cluster_size": int(np.bincount(lab).min()),
            })
    res = pd.DataFrame(rows).round(3)
    res.to_csv(f"{OUT}/variant_comparison.csv", index=False)
    pd.DataFrame(consistency).round(3).to_csv(f"{OUT}/theme_consistency.csv", index=False)
    print(res[res["k"].isin([4, 5, 6, 7])].to_string(index=False))


if __name__ == "__main__":
    main()
