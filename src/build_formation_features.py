"""Build team-season formation/personnel features from nflverse pbp data.

Features:
  - pct_shotgun / pct_under_center / pct_pistol (from offense_formation)
  - pct_shotgun_or_pistol: the shotgun-vs-under-center binary, with pistol
    counted as "not under center" (pistol is a shotgun-depth snap)
  - formation_hhi: concentration (Herfindahl index) across the 3 formations
    tracked separately (shotgun/under center/pistol)
  - pct_pers_<code>: share of plays in each of the 8 most common personnel
    groupings (RB/TE counts parsed from offense_personnel, e.g. "11" = 1 RB, 1 TE)
  - pct_top_personnel: share of plays in the team's single most-used personnel
    grouping, whatever it is (e.g. 2022 LA is ~97% regardless of which code
    dominates) - doesn't care *which* grouping, just how concentrated it is
  - top_personnel_code: which grouping that is, for reference
  - personnel_hhi: concentration across personnel groupings
  - n_personnel_groups_5plus: distinct personnel groups used 5+ times in the season

Restricted to neutral game script (see neutral_script.py) so features reflect
scheme preference rather than score/clock-driven play calling.

Output: data/formation_personnel_features.csv
"""
import re

import nfl_data_py as nfl
import numpy as np
import pandas as pd

from neutral_script import filter_neutral_script

SEASONS = [2022, 2023, 2024, 2025]
N_TOP_PERSONNEL = 8
MIN_PERSONNEL_USES = 5


def parse_personnel_code(s):
    if pd.isna(s):
        return np.nan
    rb = re.search(r"(\d+)\s*RB", s)
    te = re.search(r"(\d+)\s*TE", s)
    rb = int(rb.group(1)) if rb else 0
    te = int(te.group(1)) if te else 0
    return f"{rb}{te}"


def hhi(row):
    return (row**2).sum()


def main():
    pbp = nfl.import_pbp_data(SEASONS, downcast=True, cache=False)

    plays = pbp[(pbp["play_type"].isin(["run", "pass"])) & (pbp["posteam"].notna())].copy()
    plays = filter_neutral_script(plays)
    plays = plays[plays["offense_formation"].isin(["SHOTGUN", "UNDER CENTER", "PISTOL"])]
    plays["personnel_code"] = plays["offense_personnel"].apply(parse_personnel_code)

    form = plays.groupby(["season", "posteam", "offense_formation"]).size().unstack(fill_value=0)
    form_pct = form.div(form.sum(axis=1), axis=0)
    form_pct.columns = [f"pct_{c.lower().replace(' ', '_')}" for c in form_pct.columns]
    form_hhi = form_pct.apply(hhi, axis=1).rename("formation_hhi")
    shotgun_or_pistol = (form_pct["pct_shotgun"] + form_pct["pct_pistol"]).rename("pct_shotgun_or_pistol")

    pers = plays.groupby(["season", "posteam", "personnel_code"]).size().unstack(fill_value=0)
    pers_pct = pers.div(pers.sum(axis=1), axis=0)
    top_codes = pers.sum(axis=0).sort_values(ascending=False).head(N_TOP_PERSONNEL).index.tolist()
    pers_hhi = pers_pct.apply(hhi, axis=1).rename("personnel_hhi")
    n_groups = (pers > MIN_PERSONNEL_USES).sum(axis=1).rename("n_personnel_groups_5plus")
    pct_top_personnel = pers_pct.max(axis=1).rename("pct_top_personnel")
    top_personnel_code = pers_pct.idxmax(axis=1).rename("top_personnel_code")

    out = (
        form_pct.join(form_hhi)
        .join(shotgun_or_pistol)
        .join(pers_pct[top_codes].add_prefix("pct_pers_"))
        .join(pct_top_personnel)
        .join(top_personnel_code)
        .join(pers_hhi)
        .join(n_groups)
        .reset_index()
    )

    out_path = "data/formation_personnel_features.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} team-season rows to {out_path}")
    print(f"top personnel codes: {top_codes}")


if __name__ == "__main__":
    main()
