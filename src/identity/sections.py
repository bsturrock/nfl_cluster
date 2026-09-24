"""Section models: does each part of an offense (run game, pass game,
play-calling tendencies) have natural groups on its own?

The whole-offense model (cluster_identity.py) is only ~1 SD above the strict
null, i.e. offenses mostly vary along a continuum. This script asks the same
question per section, with section-specific features, to see where real
structure lives. Result as of the last run (see FINDINGS.md section 10):

  - run:       one real split, the wide-zone family (SF, MIA, ATL, BAL,
               LAC 2024-25) vs everyone else; nothing beyond k=2
  - pass:      no pass-specific groups; the only split is the same wide-zone
               family seen through its dropback personnel, and it is unstable
  - tendency:  no groups; the only outlier is a no-huddle/RPO tail
               (WAS 2024-25, PHI 2022-23, ARI 2022, IND 2023, LAC 2023)

QB-driven traits (designed QB runs, scrambles, RPO reads) are deliberately
kept out of run/pass so those sections describe the play caller's design.
They belong in a future "QB's job" section. RPO rate lives in tendencies.

Features are measured in neutral game script (src/neutral_script.py),
z-scored within season, screened for split-half season reliability
(>= MIN_RELIABILITY), and clustered with k-means and Ward. Each k is scored
against the copula null (same marginals + rank correlations, no clusters)
and the Gaussian null, with bootstrap stability and a full-covariance GMM
BIC as a second opinion on k.

Outputs (output/identity/sections/):
  section_features.csv      raw section features, one row per team-season
  section_reliability.csv   split-half season reliability per candidate feature
  section_k_selection.csv   fit statistics per section x method x k
  section_flags.csv         wide-zone family flag (run section, k=2)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
sys.path.insert(0, "src/identity")
sys.path.insert(0, "src")
from build_identity_features import load_plays  # noqa: E402
from cluster_identity import copula_null, gaussian_null  # noqa: E402

OUT = "output/identity/sections"
MIN_RELIABILITY = 0.6
K_RANGE = range(2, 7)
N_NULL = 40
N_BOOT = 30
SEED = 0

# Targeted-receiver route, mapped to families that exist in both the 2022
# vocabulary and the 2023+ one (routes are charted for the targeted receiver only)
ROUTE_FAMILY = {
    "CROSS": "in_breaker", "IN": "in_breaker", "IN/DIG": "in_breaker", "SHALLOW CROSS/DRAG": "in_breaker",
    "GO": "vertical", "POST": "vertical", "CORNER": "vertical", "WHEEL": "vertical",
    "OUT": "out", "QUICK OUT": "out", "DEEP OUT": "out",
    "HITCH": "hitch", "HITCH/CURL": "hitch", "SLANT": "slant",
    "FLAT": "flat_screen", "SCREEN": "flat_screen", "SWING": "flat_screen",
}

# Candidate features per section. Those failing the reliability screen are
# dropped automatically; the rest were further pruned for redundancy by hand
# (noted per section) and form SECTION_FEATURES below.
SECTION_FEATURES = {
    # non-QB designed runs only, so a mobile QB's keepers don't shape the run profile
    "run": ["outside_run", "avg_te_rb", "avg_backs_rb", "under_center_rb", "pistol_of_gun_rb",
            "motion_rb", "extra_ol_rb"],
    # dropbacks; screens moved to tendency; behind-LOS and flat/screen routes
    # duplicate screen rate (r 0.79-0.89)
    "pass": ["under_center_db", "pistol_of_gun_db", "avg_te_db", "avg_backs_db", "empty_db", "motion_db",
             "play_action", "pa_boot", "rb_target_share", "te_target_share", "short_0_9",
             "rt_in_breaker", "rt_out"],
    # overall pass rate dropped (r 0.94 with proe_early); 1st-and-10 pass rate
    # dropped (r 0.78 with proe_early and less reliable)
    "tendency": ["proe_early", "rpo_rate", "no_huddle", "screen_rate", "personnel_tell", "formation_tell"],
}


def formation_personnel(r, sfx):
    """Under center / pistol / personnel / motion on a subset of plays."""
    o = {}
    loc = r["qb_location"].str.strip()
    f = r[loc.isin(["U", "S", "P"])]
    floc = f["qb_location"].str.strip()
    o[f"under_center_{sfx}"] = (floc == "U").groupby([f["season"], f["posteam"]]).mean()
    gun = f[floc.isin(["S", "P"])]
    o[f"pistol_of_gun_{sfx}"] = (gun["qb_location"].str.strip() == "P").groupby([gun["season"], gun["posteam"]]).mean()
    pr = r[r["n_te"].notna()]
    o[f"avg_te_{sfx}"] = pr.groupby(["season", "posteam"])["n_te"].mean()
    o[f"avg_backs_{sfx}"] = pr.groupby(["season", "posteam"])["n_backs"].mean()
    o[f"extra_ol_{sfx}"] = (pr["n_ol"] >= 6).groupby([pr["season"], pr["posteam"]]).mean()
    ft = r[r["is_motion"].notna()]
    o[f"motion_{sfx}"] = ft.groupby(["season", "posteam"])["is_motion"].mean()
    o[f"empty_{sfx}"] = (ft["n_offense_backfield"] == 0).groupby([ft["season"], ft["posteam"]]).mean()
    return pd.DataFrame(o)


def build(p):
    k = ["season", "posteam"]
    by = lambda d: [d["season"], d["posteam"]]  # noqa: E731
    p = p.assign(is_pass=p["qb_dropback"].astype(float))

    # run game: non-QB designed runs (no scrambles, kneels, sneaks, QB keepers)
    runs = p[(p["rush_attempt"] == 1) & (p["qb_scramble"] != 1) & (p["is_qb_sneak"] != 1) & ~p["is_qb"]]
    run = formation_personnel(runs, "rb").drop(columns=["empty_rb"])
    rl = runs[runs["run_location"].notna()]
    run["outside_run"] = (rl["run_gap"].isin(["end", "tackle"]) & (rl["run_location"] != "middle")).groupby(by(rl)).mean()
    run["perimeter_run"] = ((rl["run_gap"] == "end") & (rl["run_location"] != "middle")).groupby(by(rl)).mean()

    # pass game: dropbacks
    db = p[p["qb_dropback"] == 1]
    pas = formation_personnel(db, "db").drop(columns=["extra_ol_db"])
    ft = db[db["is_motion"].notna()]
    pas["play_action"] = ft.groupby(k)["is_play_action"].mean()
    pa = ft[ft["is_play_action"] == 1]
    pas["pa_boot"] = pa.groupby(k)["is_qb_out_of_pocket"].mean()
    tt = db[db["time_to_throw"].notna()]
    pas["quick_throw"] = (tt["time_to_throw"] <= 2.5).groupby(by(tt)).mean()
    t = db[((db["complete_pass"] == 1) | (db["incomplete_pass"] == 1)) & db["air_yards"].notna()]
    pas["rb_target_share"] = t["receiver_pos"].isin(["RB", "FB"]).groupby(by(t)).mean()
    pas["te_target_share"] = (t["receiver_pos"] == "TE").groupby(by(t)).mean()
    pas["behind_los"] = (t["air_yards"] < 0).groupby(by(t)).mean()
    pas["short_0_9"] = t["air_yards"].between(0, 9).groupby(by(t)).mean()
    pas["intermediate_10_19"] = t["air_yards"].between(10, 19).groupby(by(t)).mean()
    pas["adot"] = t.groupby(k)["air_yards"].mean()
    rt = t.assign(fam=t["route"].map(ROUTE_FAMILY))
    rt = rt[rt["fam"].notna()]
    for fam in ["in_breaker", "vertical", "out", "hitch", "slant", "flat_screen"]:
        pas[f"rt_{fam}"] = (rt["fam"] == fam).groupby(by(rt)).mean()

    # tendencies: play-calling choices
    tend = pd.DataFrame(index=p.groupby(k).size().index)
    tend["pass_rate"] = p.groupby(k)["is_pass"].mean()
    e = p[p["down"].isin([1, 2]) & p["xpass"].notna()]
    tend["proe_early"] = (e["is_pass"] - e["xpass"]).groupby(by(e)).mean()
    d1 = p[(p["down"] == 1) & (p["ydstogo"] == 10)]
    tend["pass_1st10"] = d1.groupby(k)["is_pass"].mean()
    d2 = p[(p["down"] == 2) & (p["ydstogo"] <= 3)]
    tend["pass_2nd_short"] = d2.groupby(k)["is_pass"].mean()
    d3 = p[(p["down"] == 3) & (p["ydstogo"] <= 2)]
    tend["run_3rd_short"] = (1 - d3["is_pass"]).groupby(by(d3)).mean()
    fp = p[p["is_motion"].notna()]
    tend["rpo_rate"] = fp.groupby(k)["is_rpo"].mean()
    tend["no_huddle"] = fp.groupby(k)["is_no_huddle"].mean()
    tend["screen_rate"] = fp[fp["qb_dropback"] == 1].groupby(k)["is_screen_pass"].mean()
    pr = p[p["n_wr"].notna()]
    # how much more a team passes from 3+ WR sets than from heavier sets
    tend["personnel_tell"] = pr[pr["n_wr"] >= 3].groupby(k)["is_pass"].mean() - pr[pr["n_wr"] <= 2].groupby(k)["is_pass"].mean()
    loc = p["qb_location"].str.strip()
    tend["formation_tell"] = p[loc.isin(["S", "P"])].groupby(k)["is_pass"].mean() - p[loc == "U"].groupby(k)["is_pass"].mean()

    return {"run": run, "pass": pas, "tendency": tend}


def reliability(plays):
    odd, even = build(plays[plays["week"] % 2 == 1]), build(plays[plays["week"] % 2 == 0])
    rows = []
    for sec in odd:
        o, e = odd[sec].align(even[sec], join="inner")
        for f in o.columns:
            r = o[f].corr(e[f])
            rows.append({"section": sec, "feature": f, "split_half_r": r, "season_reliability": 2 * r / (1 + r),
                         "used": f in SECTION_FEATURES[sec]})
    return pd.DataFrame(rows).round(3)


def fit(method, X, k, seed=SEED):
    if method == "kmeans":
        return KMeans(k, n_init=30, random_state=seed).fit_predict(X)
    return AgglomerativeClustering(k, linkage="ward").fit_predict(X)


def score_section(name, X, rng):
    rows = []
    for k in K_RANGE:
        for method in ("kmeans", "ward"):
            lab = fit(method, X, k)
            sil = silhouette_score(X, lab)
            z = {}
            for tag, draw in (("strict", copula_null), ("gaussian", gaussian_null)):
                null = [silhouette_score(N, fit(method, N, k, seed=b)) for b in range(N_NULL)
                        for N in [draw(X, rng)]]
                z[tag] = ((sil - np.mean(null)) / np.std(null), float(np.mean(np.array(null) >= sil)))
            boot = np.nan
            if method == "kmeans":
                boot = np.mean([adjusted_rand_score(lab, KMeans(k, n_init=10, random_state=b)
                                                    .fit(X[rng.choice(len(X), len(X))]).predict(X))
                                for b in range(N_BOOT)])
            rows.append({"section": name, "method": method, "k": k, "silhouette": sil,
                         "z_vs_strict_null": z["strict"][0], "p_vs_strict_null": z["strict"][1],
                         "z_vs_gaussian_null": z["gaussian"][0], "bootstrap_ari": boot,
                         "min_cluster_size": int(np.bincount(lab).min())})
    bic = [GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEED).fit(X).bic(X) for k in range(1, 7)]
    best_k = int(np.argmin(bic)) + 1
    for r in rows:
        r["gmm_bic_best_k"] = best_k
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    plays = load_plays()
    sections = build(plays)

    rel = reliability(plays)
    rel.to_csv(f"{OUT}/section_reliability.csv", index=False)
    weak = rel[rel["used"] & (rel["season_reliability"] < MIN_RELIABILITY)]
    assert weak.empty, f"used features below reliability floor:\n{weak}"

    feats = pd.concat([sections[s].add_prefix(f"{s}__") for s in sections], axis=1).reset_index()
    feats = feats.rename(columns={"posteam": "team"})
    feats.round(4).to_csv(f"{OUT}/section_features.csv", index=False)

    rows = []
    for sec, cols in SECTION_FEATURES.items():
        df = sections[sec][cols]
        Z = df.groupby(level="season").transform(lambda s: (s - s.mean()) / s.std())
        rows += score_section(sec, Z.values, rng)
        if sec == "run":
            lab = KMeans(2, n_init=50, random_state=SEED).fit_predict(Z.values)
            wz = np.bincount(lab).argmin()  # the small, distinctive group
            flags = Z.index.to_frame(index=False).rename(columns={"posteam": "team"})
            flags["wide_zone_family"] = lab == wz
            flags.to_csv(f"{OUT}/section_flags.csv", index=False)
            members = flags[flags["wide_zone_family"]]
            print("wide-zone family:", ", ".join(f"{t}{str(s)[2:]}" for s, t in zip(members["season"], members["team"])))
    ksel = pd.DataFrame(rows).round(3)
    ksel.to_csv(f"{OUT}/section_k_selection.csv", index=False)
    print(ksel[ksel["method"] == "kmeans"].to_string(index=False))


if __name__ == "__main__":
    main()
