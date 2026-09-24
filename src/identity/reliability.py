"""How much of each identity feature is signal vs. sampling noise, at the
game (week) grain and the season grain.

  - split_half_r: correlation of each feature computed on odd weeks vs even
    weeks of the same team-season (128 pairs), then Spearman-Brown corrected
    to full-season length. This is the season-grain reliability.
  - game_icc: ICC(1) of the per-game feature values, grouped by team-season.
    The share of game-to-game variance that is a stable team-season trait
    rather than noise; this is the reliability of a single game's value.
  - yoy_r: correlation of a team's season N value with its season N+1 value
    (how sticky the trait is across seasons, coaching changes included).

Output: output/identity/feature_reliability.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src/identity")
from build_identity_features import FEATURES, build  # noqa: E402

OUT_PATH = "output/identity/feature_reliability.csv"


def icc1(values, groups):
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    grp = df.groupby("g")["v"]
    k = grp.size().mean()
    grand = df["v"].mean()
    n_groups = grp.ngroups
    ss_between = (grp.size() * (grp.mean() - grand) ** 2).sum()
    ss_within = ((df["v"] - grp.transform("mean")) ** 2).sum()
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (len(df) - n_groups)
    return (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)


def main():
    plays = pd.read_parquet("data/raw/neutral_plays.parquet")

    odd = build(plays[plays["week"] % 2 == 1]).set_index(["season", "team"])
    even = build(plays[plays["week"] % 2 == 0]).set_index(["season", "team"])
    odd, even = odd.align(even, join="inner")

    per_game = plays.copy()
    per_game["posteam"] = per_game["posteam"] + "|" + per_game["game_id"]
    games = build(per_game)
    games["team_season"] = games["season"].astype(str) + games["team"].str.split("|").str[0]

    season = pd.read_csv("data/identity_features.csv")
    nxt = season.assign(season=season["season"] - 1)
    yoy = season.merge(nxt, on=["season", "team"], suffixes=("", "_next"))

    rows = []
    for f in FEATURES:
        r = odd[f].corr(even[f])
        rows.append({
            "feature": f,
            "split_half_r": r,
            "season_reliability": 2 * r / (1 + r),
            "game_icc": icc1(games[f], games["team_season"]),
            "yoy_r": yoy[f].corr(yoy[f + "_next"]),
        })
    out = pd.DataFrame(rows).round(3)
    out.to_csv(OUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"\nmedian season reliability {out['season_reliability'].median():.2f}, "
          f"median single-game ICC {out['game_icc'].median():.2f}")


if __name__ == "__main__":
    main()
