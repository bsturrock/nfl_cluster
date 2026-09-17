"""Build team-season offensive features from nflverse data.

Features (v1):
  - run gap mix: share of playcalled rushes that go end / tackle / guard / middle
  - play-action rate: share of dropbacks that are play-action (FTN charting, 2022+)

Restricted to neutral game script (see neutral_script.py) so features reflect
scheme preference rather than score/clock-driven play calling.

Output: data/team_season_features.csv
"""
import nfl_data_py as nfl
import pandas as pd

from neutral_script import filter_neutral_script

SEASONS = [2022, 2023, 2024, 2025]


def load_pbp(seasons):
    pbp = nfl.import_pbp_data(seasons, downcast=True, cache=False)
    return pbp


def load_ftn(seasons):
    ftn = nfl.import_ftn_data(seasons, downcast=True)
    return ftn


def build_gap_mix(pbp: pd.DataFrame) -> pd.DataFrame:
    rush = pbp[
        (pbp["rush_attempt"] == 1)
        & (pbp["qb_scramble"] != 1)
        & (pbp["qb_kneel"] != 1)
    ].copy()

    # middle runs carry no run_gap value in nflfastR; treat "middle" as its
    # own bucket alongside end/tackle/guard
    rush["gap_bucket"] = rush["run_gap"]
    rush.loc[rush["run_location"] == "middle", "gap_bucket"] = "middle"
    rush = rush[rush["gap_bucket"].notna()]

    counts = (
        rush.groupby(["season", "posteam", "gap_bucket"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["end", "tackle", "guard", "middle"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["rush_n"] = counts[["end", "tackle", "guard", "middle"]].sum(axis=1)
    for col in ["end", "tackle", "guard", "middle"]:
        counts[f"pct_{col}"] = counts[col] / counts["rush_n"]

    out = counts.reset_index()[
        ["season", "posteam", "rush_n", "pct_end", "pct_tackle", "pct_guard", "pct_middle"]
    ]
    out = out.rename(columns={"posteam": "team"})
    return out


def build_play_action_rate(pbp: pd.DataFrame, ftn: pd.DataFrame) -> pd.DataFrame:
    merged = pbp.merge(
        ftn[["nflverse_game_id", "nflverse_play_id", "is_play_action"]],
        left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "nflverse_play_id"],
        how="inner",
    )
    dropbacks = merged[merged["qb_dropback"] == 1].copy()

    grp = dropbacks.groupby(["season", "posteam"]).agg(
        dropback_n=("is_play_action", "size"),
        pa_n=("is_play_action", "sum"),
    )
    grp["pa_rate"] = grp["pa_n"] / grp["dropback_n"]
    out = grp.reset_index().rename(columns={"posteam": "team"})
    return out[["season", "team", "dropback_n", "pa_rate"]]


def main():
    pbp = load_pbp(SEASONS)
    pbp = filter_neutral_script(pbp)
    ftn = load_ftn(SEASONS)

    gap_mix = build_gap_mix(pbp)
    pa_rate = build_play_action_rate(pbp, ftn)

    features = gap_mix.merge(pa_rate, on=["season", "team"], how="inner")
    features = features[features["team"].notna()]
    features = features.sort_values(["season", "team"]).reset_index(drop=True)

    out_path = "data/team_season_features.csv"
    features.to_csv(out_path, index=False)
    print(f"wrote {len(features)} team-season rows to {out_path}")
    print(features.describe())


if __name__ == "__main__":
    main()
