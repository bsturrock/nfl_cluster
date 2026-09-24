"""Run-play-only features: run direction and snap formation on designed
non-QB runs.

Play set (per team-season, neutral script - see neutral_script.py):
  - play_type == "run" (called runs; kneels are their own play_type)
  - no QB scrambles (those are called passes)
  - no two-point tries
  - no QB rushers: rusher's roster position is QB (drops designed QB runs,
    sneaks, and QB keeps on RPO/option)

Features:
  - pct_outside / pct_inside: end + tackle vs guard + middle, over runs
    with a direction tag
  - pct_shotgun_or_pistol / pct_under_center / pct_pistol: over the same
    runs with a formation tag (formation of the run snaps, not all snaps)
  - avg_rb / avg_fb / avg_te: mean players on the field per run, from
    offense_personnel (roster positions). avg_rb counts RB + FB, matching
    standard personnel-grouping convention (21 = 2 backs incl. a FB). Extra
    OL are listed as T/G, so jumbo linemen are not counted as TEs.

Output: data/run_play_features.csv
"""
import re

import nfl_data_py as nfl
import pandas as pd

from neutral_script import filter_neutral_script

SEASONS = [2025]
OUT_PATH = "data/run_play_features.csv"


def position_count(personnel, pos):
    """Players at `pos` in an offense_personnel string like '1 C, 1 FB, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 1 WR'."""
    if pd.isna(personnel):
        return pd.NA
    m = re.search(rf"(\d+) {pos}\b", personnel)
    return int(m.group(1)) if m else 0


def main():
    pbp = nfl.import_pbp_data(SEASONS, downcast=True, cache=False)
    rosters = nfl.import_seasonal_rosters(SEASONS)
    qb_ids = set(rosters.loc[rosters["position"] == "QB", "player_id"])

    runs = pbp[
        (pbp["play_type"] == "run")
        & (pbp["qb_scramble"] != 1)
        & (pbp["two_point_attempt"] != 1)
        & (pbp["posteam"].notna())
    ]
    runs = filter_neutral_script(runs).copy()
    n_before = len(runs)
    runs = runs[~runs["rusher_player_id"].isin(qb_ids)]
    print(f"dropped {n_before - len(runs)} QB runs, kept {len(runs)} designed non-QB runs")

    # middle runs carry no run_gap value in nflfastR
    runs["gap"] = runs["run_gap"]
    runs.loc[runs["run_location"] == "middle", "gap"] = "middle"
    runs["side"] = runs["gap"].map({"end": "outside", "tackle": "outside", "guard": "inside", "middle": "inside"})

    keys = ["season", "posteam"]
    side = runs[runs["side"].notna()].groupby(keys)["side"].value_counts(normalize=True).unstack(fill_value=0)
    side.columns = [f"pct_{c}" for c in side.columns]
    side["dir_n"] = runs[runs["side"].notna()].groupby(keys).size()

    form = runs[runs["offense_formation"].notna()]
    form_pct = form.groupby(keys)["offense_formation"].value_counts(normalize=True).unstack(fill_value=0)
    out_form = pd.DataFrame({
        "pct_shotgun_or_pistol": form_pct["SHOTGUN"] + form_pct["PISTOL"],
        "pct_pistol": form_pct["PISTOL"],
        "pct_under_center": form_pct["UNDER CENTER"],
        "form_n": form.groupby(keys).size(),
    })

    pers = runs[runs["offense_personnel"].notna()].copy()
    for pos in ["RB", "FB", "TE"]:
        pers[pos] = pers["offense_personnel"].apply(position_count, pos=pos)
    pers["backs"] = pers["RB"] + pers["FB"]
    out_pers = pers.groupby(keys).agg(
        avg_rb=("backs", "mean"), avg_fb=("FB", "mean"), avg_te=("TE", "mean"), pers_n=("TE", "size"),
    )

    out = side.join(out_form).join(out_pers)
    out["run_n"] = runs.groupby(keys).size()
    out = out.reset_index().rename(columns={"posteam": "team"})
    out = out[["season", "team", "run_n", "dir_n", "form_n", "pct_outside", "pct_inside",
               "pct_shotgun_or_pistol", "pct_pistol", "pct_under_center",
               "pers_n", "avg_rb", "avg_fb", "avg_te"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(out)} rows to {OUT_PATH}")
    print(out.describe().round(3))


if __name__ == "__main__":
    main()
