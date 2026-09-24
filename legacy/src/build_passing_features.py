"""Build team-season passing-profile features from nflverse pbp data.

Features:
  - adot: average depth of target (mean air_yards on targets - complete or
    incomplete passes). 2022-2025.
  - pct_top_route: share of targets going to the team's single most-targeted
    route type, whatever it is. Route is only charted for the *targeted*
    receiver, not every route run on the play (nflfastR `route` column is
    populated on ~97% of actual targets, ~0% on non-targets) - so this
    measures target concentration by route, not true route-running mix.
    2023-2025 only: nflfastR's route vocabulary changed after 2022 (e.g.
    2022's single "OUT" vs 2023+'s "QUICK OUT"/"DEEP OUT" split) in a way
    that doesn't cleanly nest the way offense_formation's did, so merging
    categories to include 2022 measurably distorts the concentration
    number instead of fixing it - excluded rather than faked.

Restricted to neutral game script (see neutral_script.py).

Output: data/passing_features.csv
"""
import nfl_data_py as nfl
import pandas as pd

from neutral_script import filter_neutral_script

SEASONS = [2022, 2023, 2024, 2025]
ROUTE_SEASONS = [2023, 2024, 2025]


def main():
    pbp = nfl.import_pbp_data(SEASONS, downcast=True, cache=False)
    pbp = filter_neutral_script(pbp)
    plays = pbp[(pbp["play_type"].isin(["run", "pass"])) & (pbp["posteam"].notna())].copy()
    targets = plays[(plays["complete_pass"] == 1) | (plays["incomplete_pass"] == 1)].copy()

    adot = targets.groupby(["season", "posteam"])["air_yards"].mean().rename("adot")

    route_targets = targets[targets["season"].isin(ROUTE_SEASONS) & (targets["route"] != "")]
    route_counts = route_targets.groupby(["season", "posteam", "route"]).size().unstack(fill_value=0)
    route_pct = route_counts.div(route_counts.sum(axis=1), axis=0)
    pct_top_route = route_pct.max(axis=1).rename("pct_top_route")
    top_route = route_pct.idxmax(axis=1).rename("top_route")

    out = pd.concat([adot, pct_top_route, top_route], axis=1).reset_index()

    out_path = "data/passing_features.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} team-season rows to {out_path}")
    print(f"pct_top_route populated for {out['pct_top_route'].notna().sum()} rows (2023-2025 only)")


if __name__ == "__main__":
    main()
