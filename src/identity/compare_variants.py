"""Compare alternative feature treatments against the current identity model.

Variants (all k-means, scored the same way as cluster_identity.py: silhouette
vs a same-covariance Gaussian null, bootstrap ARI, year-over-year persistence,
and ARI against the current k=5 assignments):

  current          19 within-season z-scores -> PCA (parallel analysis)
  skew_fixed       same, but log(x + 0.01) on the six zero-inflated features
                   before z-scoring
  themes           19 features collapsed into 7 fixed, equal-weight themes
  themes_refined   themes minus the two features that don't fit their theme
                   (extra_ol, screen: item-rest r < 0.15)
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
OUT = "output/identity"
SKEWED = ["pistol", "two_back", "extra_ol", "no_huddle", "qb_design_run", "rpo"]
THEMES = {
    "under_center_vs_gun_rpo": {"under_center": 1, "rpo": -1},
    "dropback_play_action": {"play_action": 1, "time_to_throw": 1, "screen": -1},
    "wide_zone_package": {"two_back": 1, "motion": 1, "outside_run": 1, "pistol": 1, "rb_target_share": 1},
    "te_heavy": {"multi_te": 1, "te_target_share": 1, "extra_ol": 1},
    "qb_run_game": {"qb_design_run": 1, "qb_out_of_pocket": 1},
    "tempo": {"no_huddle": 1},
    "pass_first": {"proe_early": 1, "empty_backfield": 1},
}
DROP_REFINED = {"extra_ol", "screen"}
THEMES_REFINED = {n: {f: s for f, s in w.items() if f not in DROP_REFINED} for n, w in THEMES.items()}


def main():
    rng = np.random.default_rng(0)
    t = pd.read_csv(f"{OUT}/team_season_identity.csv")
    feats = pd.read_csv(f"{OUT}/pca_loadings.csv", index_col=0).index[:-3].tolist()
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
        "current": pca(z_raw.values),
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
        cov = np.cov(X.T)
        for k in range(3, 10):
            lab = KMeans(k, n_init=50, random_state=0).fit_predict(X)
            sil = silhouette_score(X, lab)
            null = []
            for b in range(30):
                N = rng.multivariate_normal(np.zeros(X.shape[1]), cov, len(X))
                if name == "binned_3":  # bin the null the same way, before its PCA
                    Nz = rng.multivariate_normal(np.zeros(len(feats)), np.cov(z_raw.values.T), len(X))
                    N = PCA(X.shape[1]).fit_transform(np.where(Nz > 0.43, 1.0, np.where(Nz < -0.43, -1.0, 0.0)))
                null.append(silhouette_score(N, KMeans(k, n_init=10, random_state=b).fit_predict(N)))
            boot = [adjusted_rand_score(lab, KMeans(k, n_init=10, random_state=b)
                                        .fit(X[rng.choice(len(X), len(X))]).predict(X)) for b in range(40)]
            rows.append({
                "variant": name, "dims": X.shape[1], "k": k, "silhouette": sil,
                "null_mean": np.mean(null), "silhouette_z_vs_null": (sil - np.mean(null)) / np.std(null),
                "bootstrap_ari": np.mean(boot), "yoy_same_cluster": yoy(lab),
                "ari_vs_current_k5": adjusted_rand_score(t["cluster_k5"], lab),
                "min_cluster_size": int(np.bincount(lab).min()),
            })
    res = pd.DataFrame(rows).round(3)
    res.to_csv(f"{OUT}/variant_comparison.csv", index=False)
    pd.DataFrame(consistency).round(3).to_csv(f"{OUT}/theme_consistency.csv", index=False)
    print(res[res["k"].isin([4, 5, 6, 7])].to_string(index=False))


if __name__ == "__main__":
    main()
