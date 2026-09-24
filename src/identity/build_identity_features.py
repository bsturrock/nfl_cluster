"""Build team-season offensive *identity* features (2022-2025).

"Identity" here means what an offense is trying to do, not how well it does
it: every feature is a tendency/choice rate, never an efficiency outcome
(no EPA, success rate, sack rate, etc.). Season range is bounded by FTN
charting (play action, motion, RPO, screens, QB out of pocket), which starts
in 2022.

Sources (read from data/raw/, fetched by src/identity/fetch_data.sh):
  - nflverse pbp            (xpass, air yards, run location/gap, receivers)
  - nflverse FTN charting   (motion, PA, RPO, screen, no-huddle, sneak, OOP)
  - nflverse participation  (personnel, time to throw)
  - nflverse players        (receiver position for target shares)

Filters: regular season, run/pass plays, no kneels/spikes, neutral game
script (src/neutral_script.py: WP 20-80%, outside final 2 min of each half).

Features, grouped by the part of the identity they describe:

  Run/pass philosophy
    proe_early        pass rate over expected (qb_dropback - xpass) on 1st/2nd down
  Formation / personnel (share of all neutral plays)
    under_center      snaps under center (FTN qb_location)
    pistol            snaps from pistol (FTN qb_location)
    pers_11           1 back, 1 TE
    multi_te          2+ TEs on the field
    two_back          2+ backs (RB + FB) on the field
    empty_backfield   0 players in the backfield (FTN)
    extra_ol          6+ offensive linemen
    personnel_entropy Shannon entropy (bits) of the personnel-grouping mix
  Pre-snap / tempo
    motion            pre-snap motion (FTN)
    no_huddle         no-huddle (FTN)
  Run game
    outside_run       share of non-QB designed runs to end/tackle gaps
    qb_design_run     designed QB runs (roster QB, not scrambles, not sneaks) / all plays
    rpo               RPO / all plays (FTN)
  Pass game (share of dropbacks unless stated)
    play_action       play action (FTN)
    screen            screen passes (FTN)
    qb_out_of_pocket  QB outside the pocket, incl. boots/rollouts (FTN)
    adot              mean air yards on targets
    deep_rate         targets with air yards >= 20 / targets
    middle_target     targets to the middle of the field / targets
    time_to_throw     median time to throw (participation / NGS)
    rb_target_share   targets to RBs/FBs
    te_target_share   targets to TEs

Output: data/identity_features.csv (one row per team-season, with sample sizes)
"""
import re

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")
from neutral_script import filter_neutral_script  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]
RAW = "data/raw"
OUT_PATH = "data/identity_features.csv"

PBP_COLS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "down",
    "play_type", "qb_dropback", "rush_attempt", "pass_attempt", "qb_scramble",
    "qb_kneel", "qb_spike", "xpass", "wp", "half_seconds_remaining",
    "air_yards", "complete_pass", "incomplete_pass", "pass_location",
    "run_location", "run_gap", "rusher_player_id", "passer_player_id",
    "receiver_player_id",
]
FTN_COLS = [
    "nflverse_game_id", "nflverse_play_id", "is_motion", "is_no_huddle",
    "is_play_action", "is_screen_pass", "is_rpo", "is_qb_out_of_pocket",
    "is_qb_sneak", "n_offense_backfield", "qb_location",
]
PART_COLS = [
    "nflverse_game_id", "play_id", "offense_personnel",
    "time_to_throw",
]


def parse_personnel(s):
    """Return (backs, te, wr, ol) from either personnel string format.

    2022: "1 RB, 2 TE, 2 WR" (+ optional "6 OL"); FBs are counted as RB.
    2023+: full position list, e.g. "1 C, 1 FB, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 2 WR".
    """
    if not isinstance(s, str):
        return (np.nan,) * 4
    counts = {pos: int(n) for n, pos in re.findall(r"(\d+)\s*([A-Z]+)", s)}
    backs = counts.get("RB", 0) + counts.get("FB", 0)
    te = counts.get("TE", 0)
    wr = counts.get("WR", 0)
    if "OL" in counts:
        ol = counts["OL"]
    elif any(k in counts for k in ("C", "G", "T")):
        ol = counts.get("C", 0) + counts.get("G", 0) + counts.get("T", 0)
    else:
        ol = 5
    return backs, te, wr, ol


def load_season(season):
    pbp = pd.read_parquet(f"{RAW}/pbp_{season}.parquet", columns=PBP_COLS)
    ftn = pd.read_parquet(f"{RAW}/ftn_{season}.parquet", columns=FTN_COLS)
    part = pd.read_parquet(f"{RAW}/part_{season}.parquet", columns=PART_COLS)
    pbp = pbp.merge(
        ftn, left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "nflverse_play_id"], how="left",
    ).drop(columns=["nflverse_game_id", "nflverse_play_id"])
    pbp = pbp.merge(
        part, left_on=["game_id", "play_id"],
        right_on=["nflverse_game_id", "play_id"], how="left",
    ).drop(columns=["nflverse_game_id"])
    return pbp


def load_plays():
    pbp = pd.concat([load_season(s) for s in SEASONS], ignore_index=True)
    plays = pbp[
        (pbp["season_type"] == "REG")
        & pbp["play_type"].isin(["run", "pass"])
        & pbp["posteam"].notna()
        & (pbp["qb_kneel"] != 1)
        & (pbp["qb_spike"] != 1)
    ]
    plays = filter_neutral_script(plays).copy()

    pers = plays["offense_personnel"].map(parse_personnel)
    plays[["n_backs", "n_te", "n_wr", "n_ol"]] = pd.DataFrame(pers.tolist(), index=plays.index)
    plays["pers_code"] = plays["n_backs"].astype("Int64").astype(str) + plays["n_te"].astype("Int64").astype(str)
    plays.loc[plays["n_backs"].isna(), "pers_code"] = np.nan

    for col in ["is_motion", "is_no_huddle", "is_play_action", "is_screen_pass", "is_rpo",
                "is_qb_out_of_pocket", "is_qb_sneak"]:
        plays[col] = plays[col].astype("float64")

    # rusher is a QB by roster position (not "threw a pass that season", which
    # flags RBs/WRs who threw a trick-play pass, e.g. 2022 Derrick Henry)
    players = pd.read_parquet(f"{RAW}/players.parquet", columns=["gsis_id", "position"])
    qb_ids = set(players.loc[players["position"] == "QB", "gsis_id"])
    plays["is_qb"] = plays["rusher_player_id"].isin(qb_ids)
    plays = plays.merge(
        players.rename(columns={"gsis_id": "receiver_player_id", "position": "receiver_pos"}),
        on="receiver_player_id", how="left",
    )
    return plays


def entropy_bits(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def build(plays):
    key = ["season", "posteam"]
    g = plays.groupby(key)
    out = pd.DataFrame({"n_plays": g.size()})

    early = plays[plays["down"].isin([1, 2]) & plays["xpass"].notna()]
    out["proe_early"] = (early["qb_dropback"] - early["xpass"]).groupby([early["season"], early["posteam"]]).mean()

    # FTN qb_location (U/S/P) is charted the same way every season, unlike
    # participation offense_formation (2022 used SINGLEBACK/I_FORM/EMPTY/...)
    form = plays[plays["qb_location"].str.strip().isin(["U", "S", "P"])]
    fg = form.groupby(key)["qb_location"]
    out["under_center"] = fg.apply(lambda s: (s.str.strip() == "U").mean())
    out["pistol"] = fg.apply(lambda s: (s.str.strip() == "P").mean())

    pers = plays[plays["pers_code"].notna()]
    pg = pers.groupby(key)
    out["pers_11"] = pg["pers_code"].apply(lambda s: (s == "11").mean())
    out["multi_te"] = pg["n_te"].apply(lambda s: (s >= 2).mean())
    out["two_back"] = pg["n_backs"].apply(lambda s: (s >= 2).mean())
    out["extra_ol"] = pg["n_ol"].apply(lambda s: (s >= 6).mean())
    out["personnel_entropy"] = pg["pers_code"].apply(lambda s: entropy_bits(s.value_counts()))

    ftn = plays[plays["is_motion"].notna()]
    tg = ftn.groupby(key)
    out["empty_backfield"] = tg["n_offense_backfield"].apply(lambda s: (s == 0).mean())
    out["motion"] = tg["is_motion"].mean()
    out["no_huddle"] = tg["is_no_huddle"].mean()
    out["rpo"] = tg["is_rpo"].mean()
    qb_design = (
        (ftn["rush_attempt"] == 1) & ftn["is_qb"] & (ftn["qb_scramble"] != 1) & (ftn["is_qb_sneak"] != 1)
    )
    out["qb_design_run"] = qb_design.groupby([ftn["season"], ftn["posteam"]]).mean()

    rb_runs = plays[(plays["rush_attempt"] == 1) & ~plays["is_qb"] & plays["run_location"].notna()]
    outside = rb_runs["run_gap"].isin(["end", "tackle"]) & (rb_runs["run_location"] != "middle")
    out["outside_run"] = outside.groupby([rb_runs["season"], rb_runs["posteam"]]).mean()

    db = ftn[ftn["qb_dropback"] == 1]
    dg = db.groupby(key)
    out["n_dropbacks"] = dg.size()
    out["play_action"] = dg["is_play_action"].mean()
    out["screen"] = dg["is_screen_pass"].mean()
    out["qb_out_of_pocket"] = dg["is_qb_out_of_pocket"].mean()
    out["time_to_throw"] = plays[plays["qb_dropback"] == 1].groupby(key)["time_to_throw"].median()

    tgt = plays[
        ((plays["complete_pass"] == 1) | (plays["incomplete_pass"] == 1)) & plays["air_yards"].notna()
    ]
    tg2 = tgt.groupby(key)
    out["n_targets"] = tg2.size()
    out["adot"] = tg2["air_yards"].mean()
    out["deep_rate"] = tg2["air_yards"].apply(lambda s: (s >= 20).mean())
    out["middle_target"] = tg2["pass_location"].apply(lambda s: (s == "middle").mean())
    out["rb_target_share"] = tg2["receiver_pos"].apply(lambda s: s.isin(["RB", "FB"]).mean())
    out["te_target_share"] = tg2["receiver_pos"].apply(lambda s: (s == "TE").mean())

    out = out.reset_index().rename(columns={"posteam": "team"})
    return out


FEATURES = [
    "proe_early", "under_center", "pistol", "pers_11", "multi_te", "two_back",
    "empty_backfield", "extra_ol", "personnel_entropy", "motion", "no_huddle",
    "outside_run", "qb_design_run", "rpo", "play_action", "screen",
    "qb_out_of_pocket", "adot", "deep_rate", "middle_target", "time_to_throw",
    "rb_target_share", "te_target_share",
]


def main():
    plays = load_plays()
    feats = build(plays)
    feats = feats.sort_values(["season", "team"]).reset_index(drop=True)
    feats.to_csv(OUT_PATH, index=False)
    # play-level table for reliability checks (week-level vs season-level)
    plays[["season", "week", "game_id", "posteam"] + [
        "down", "qb_dropback", "xpass", "qb_location", "pers_code", "n_te", "n_backs", "n_ol",
        "n_offense_backfield", "is_motion", "is_no_huddle", "is_rpo", "is_play_action",
        "is_screen_pass", "is_qb_out_of_pocket", "air_yards", "complete_pass", "incomplete_pass",
        "pass_location", "receiver_pos", "rush_attempt", "is_qb", "qb_scramble", "is_qb_sneak",
        "run_location", "run_gap", "time_to_throw",
    ]].to_parquet("data/raw/neutral_plays.parquet", index=False)
    print(f"wrote {len(feats)} team-seasons to {OUT_PATH}")
    print(feats[FEATURES].describe().T.round(3))


if __name__ == "__main__":
    main()
