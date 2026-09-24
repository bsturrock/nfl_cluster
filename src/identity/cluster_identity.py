"""Cluster team-seasons (2022-2025) on offensive identity.

Pipeline:
  1. Features from data/identity_features.csv, keeping only those with
     season-level reliability >= MIN_RELIABILITY (output/identity/
     feature_reliability.csv, from reliability.py) and dropping one of any
     pair correlated beyond MAX_ABS_CORR.
  2. Z-score each feature *within season*: league-wide drift (motion went
     from 42% to 60% of neutral plays 2022->2025) would otherwise make
     "season" the dominant cluster axis instead of identity.
  3. PCA, keeping the components that beat Horn's parallel analysis
     (eigenvalue above the 95th percentile of eigenvalues from same-shape
     random data).
  4. Sweep k=2..10 for k-means, Ward, and diagonal GMM. For each: silhouette
     (in PCA space and in the full z-scored feature space), Calinski-
     Harabasz, Davies-Bouldin, bootstrap stability (mean ARI of refits on
     resamples vs the full-data fit), and a null-model silhouette: k-means
     on Gaussian data with the same covariance and no cluster structure.
     silhouette_z = (silhouette - null mean) / null sd is what k is chosen on,
     since raw silhouette falls mechanically with dimensionality.
  5. Final model: k-means at the primary k, plus a secondary finer k.

Outputs (output/identity/):
  k_selection.csv, pca_loadings.csv, team_season_identity.csv,
  cluster_profiles_k{K}.csv, cluster_members_k{K}.csv, team_trajectories.csv,
  cluster_transitions.csv
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
    silhouette_samples, silhouette_score,
)
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore", category=FutureWarning)

OUT = "output/identity"
MIN_RELIABILITY = 0.6
MAX_ABS_CORR = 0.8
K_RANGE = range(2, 11)
K_PRIMARY = 5
K_SECONDARY = 7
N_BOOT = 50
N_NULL = 30
SEED = 0

# Labels were assigned by reading the profiles printed below. Cluster ids are
# made deterministic by ordering clusters by size (0 = largest).
LABELS = {
    5: {
        0: "Shotgun spread, pass-first",
        1: "Under-center play-action",
        2: "Shotgun QB-run / RPO",
        3: "Shanahan wide-zone, two-back motion",
        4: "Heavy power run + QB run",
    },
    7: {
        0: "Shotgun spread, balanced",
        1: "Under-center, multi-TE play-action",
        2: "Under-center 11p + motion (McVay-style)",
        3: "Pass-first spread (early-down PROE)",
        4: "Shotgun QB-run / RPO",
        5: "Shanahan wide-zone, two-back motion",
        6: "Heavy power run + QB run",
    },
}


def select_features(feats):
    rel = pd.read_csv(f"{OUT}/feature_reliability.csv").set_index("feature")
    kept = [f for f in rel.index if rel.loc[f, "season_reliability"] >= MIN_RELIABILITY]
    dropped = {f: f"season reliability {rel.loc[f, 'season_reliability']:.2f} < {MIN_RELIABILITY}"
               for f in rel.index if f not in kept}
    z = within_season_z(feats, kept)
    corr = z.corr().abs()
    # among over-threshold pairs, drop the feature with lower reliability
    for i, a in enumerate(list(kept)):
        for b in kept[i + 1:]:
            if a in kept and b in kept and corr.loc[a, b] > MAX_ABS_CORR:
                loser = a if rel.loc[a, "season_reliability"] < rel.loc[b, "season_reliability"] else b
                other = b if loser == a else a
                kept.remove(loser)
                dropped[loser] = f"|r|={corr.loc[a, b]:.2f} with {other}"
    return kept, dropped


def within_season_z(feats, cols):
    return feats.groupby("season")[cols].transform(lambda s: (s - s.mean()) / s.std())


def parallel_analysis(Z, rng, n_iter=200):
    real = PCA().fit(Z).explained_variance_
    null = []
    for _ in range(n_iter):
        R = rng.standard_normal(Z.shape)
        R = (R - R.mean(0)) / R.std(0, ddof=1)
        null.append(PCA().fit(R).explained_variance_)
    thresh = np.percentile(null, 95, axis=0)
    n = int(np.argmin(real > thresh)) if not (real > thresh).all() else len(real)
    return n, real, thresh


def fit_labels(method, X, k, seed=SEED):
    if method == "kmeans":
        return KMeans(k, n_init=50, random_state=seed).fit_predict(X)
    if method == "ward":
        return AgglomerativeClustering(k, linkage="ward").fit_predict(X)
    if method == "gmm":
        return GaussianMixture(k, covariance_type="diag", n_init=5, random_state=seed).fit_predict(X)
    raise ValueError(method)


def bootstrap_stability(method, X, k, ref, rng):
    aris = []
    for b in range(N_BOOT):
        idx = rng.choice(len(X), len(X), replace=True)
        lab = fit_labels(method, X[idx], k, seed=b)
        cents = np.array([X[idx][lab == j].mean(0) for j in np.unique(lab)])
        pred = ((X[:, None, :] - cents[None]) ** 2).sum(-1).argmin(1)
        aris.append(adjusted_rand_score(ref, pred))
    return float(np.mean(aris))


def null_silhouette(X, k, rng):
    cov = np.cov(X.T)
    sils = []
    for b in range(N_NULL):
        N = rng.multivariate_normal(np.zeros(X.shape[1]), cov, len(X))
        sils.append(silhouette_score(N, KMeans(k, n_init=10, random_state=b).fit_predict(N)))
    return float(np.mean(sils)), float(np.std(sils)), float(np.percentile(sils, 95))


def relabel_by_size(labels):
    order = pd.Series(labels).value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels])


def main():
    rng = np.random.default_rng(SEED)
    feats = pd.read_csv("data/identity_features.csv")
    features, dropped = select_features(feats)
    Zdf = within_season_z(feats, features)
    Z = Zdf.values

    n_pc, eig, eig_null = parallel_analysis(Z, rng)
    pca = PCA(n_pc, random_state=SEED).fit(Z)
    X = pca.transform(Z)
    print(f"{len(features)} features kept; dropped: {json.dumps(dropped, indent=1)}")
    print(f"parallel analysis keeps {n_pc} PCs ({pca.explained_variance_ratio_.sum():.1%} of variance)")

    loadings = pd.DataFrame(pca.components_.T, index=features,
                            columns=[f"PC{i+1}" for i in range(n_pc)]).round(3)
    loadings.loc["explained_var_ratio"] = pca.explained_variance_ratio_.round(3)
    loadings.loc["eigenvalue"] = eig[:n_pc].round(3)
    loadings.loc["null_eigenvalue_p95"] = eig_null[:n_pc].round(3)
    loadings.to_csv(f"{OUT}/pca_loadings.csv")

    rows = []
    null_cache = {}
    for k in K_RANGE:
        null_cache[k] = null_silhouette(X, k, rng)
        for method in ["kmeans", "ward", "gmm"]:
            lab = fit_labels(method, X, k)
            sil = silhouette_score(X, lab)
            nm, ns, n95 = null_cache[k]
            rows.append({
                "k": k, "method": method,
                "silhouette": sil,
                "silhouette_full_space": silhouette_score(Z, lab),
                "null_silhouette_mean": nm, "null_silhouette_p95": n95,
                "silhouette_z_vs_null": (sil - nm) / ns,
                "calinski_harabasz": calinski_harabasz_score(X, lab),
                "davies_bouldin": davies_bouldin_score(X, lab),
                "bootstrap_ari": bootstrap_stability(method, X, k, lab, rng),
                "min_cluster_size": int(np.bincount(lab).min()),
            })
    ksel = pd.DataFrame(rows).round(3)
    ksel.to_csv(f"{OUT}/k_selection.csv", index=False)
    print(ksel[ksel["method"] == "kmeans"].to_string(index=False))

    out = feats[["season", "team"]].copy()
    for k in (K_PRIMARY, K_SECONDARY):
        km = KMeans(k, n_init=50, random_state=SEED).fit(X)
        lab = relabel_by_size(km.labels_)
        cents = np.array([X[lab == j].mean(0) for j in range(k)])
        d = ((X[:, None, :] - cents[None]) ** 2).sum(-1) ** 0.5
        second = np.argsort(d, axis=1)[:, 1]
        out[f"cluster_k{k}"] = lab
        out[f"label_k{k}"] = [LABELS[k][c] for c in lab]
        out[f"silhouette_k{k}"] = silhouette_samples(X, lab).round(3)
        out[f"runner_up_k{k}"] = [LABELS[k][c] for c in second]
        # margin: how much closer the assigned centroid is than the runner-up (0 = on the boundary)
        out[f"margin_k{k}"] = (1 - d[np.arange(len(X)), lab] / d[np.arange(len(X)), second]).round(3)

        prof_z = Zdf.groupby(lab).mean().T.round(2)
        prof_raw = feats[features].groupby(lab).mean().T.round(3)
        prof = pd.concat({"z": prof_z, "raw_mean": prof_raw}, axis=1)
        prof.columns = [f"{kind}_c{c}" for kind, c in prof.columns]
        league = feats[features].mean().round(3).rename("league_mean")
        prof = prof.join(league)
        size = pd.Series(np.bincount(lab), name="n")
        csil = pd.Series(silhouette_samples(X, lab)).groupby(lab).mean().round(3)
        meta = pd.DataFrame({f"z_c{c}": [LABELS[k][c], size[c], csil[c]] for c in range(k)},
                            index=["label", "n", "mean_silhouette"])
        pd.concat([meta, prof]).to_csv(f"{OUT}/cluster_profiles_k{k}.csv")

        members = (
            out.assign(ts=out["team"] + " " + out["season"].astype(str))
            .sort_values([f"cluster_k{k}", f"silhouette_k{k}"], ascending=[True, False])
            .groupby([f"cluster_k{k}", f"label_k{k}"])
            .agg(n=("ts", "size"), members_most_to_least_typical=("ts", ", ".join))
            .reset_index()
        )
        members.to_csv(f"{OUT}/cluster_members_k{k}.csv", index=False)

        print(f"\n=== k={k} (silhouette {silhouette_score(X, lab):.3f}) ===")
        print(prof_z.to_string())
        for c in range(k):
            print(c, LABELS[k][c], size[c], members.loc[c, "members_most_to_least_typical"])

    for i in range(n_pc):
        out[f"PC{i+1}"] = X[:, i].round(3)
    out = out.join(feats[features + ["n_plays", "n_dropbacks"]].round(4))
    out.to_csv(f"{OUT}/team_season_identity.csv", index=False)

    traj = out.pivot(index="team", columns="season", values=f"label_k{K_PRIMARY}")
    traj.to_csv(f"{OUT}/team_trajectories.csv")

    prev = out[["season", "team", f"label_k{K_PRIMARY}"]]
    pairs = prev.merge(prev.assign(season=prev["season"] - 1), on=["season", "team"],
                       suffixes=("_from", "_to"))
    trans = pd.crosstab(pairs[f"label_k{K_PRIMARY}_from"], pairs[f"label_k{K_PRIMARY}_to"])
    trans.to_csv(f"{OUT}/cluster_transitions.csv")
    same = (pairs[f"label_k{K_PRIMARY}_from"] == pairs[f"label_k{K_PRIMARY}_to"]).mean()
    shares = out[f"label_k{K_PRIMARY}"].value_counts(normalize=True)
    print(f"\nyear-over-year same cluster: {same:.1%} (chance given cluster sizes: {(shares**2).sum():.1%})")
    print(pd.crosstab(out["season"], out[f"label_k{K_PRIMARY}"]))


if __name__ == "__main__":
    main()
