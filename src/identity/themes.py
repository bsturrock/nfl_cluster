"""Theme composites: the dimensions the identity model clusters on.

Each theme is the equal-weight mean of its features' within-season z-scores
(sign-flipped where a feature marks the opposite pole), re-standardized
within season. Groupings come from football logic checked against the
feature correlations; a feature stays in a theme only if it correlates with
the rest of that theme (item-rest r >= 0.15). extra_ol and screen failed that
check against every theme and are left out of the model (they are still in
the feature table and the scatter).

Bipolar themes have two real choices at either end; unipolar ones measure how
much a team leans on something, with "rarely does it" at the low end.
"""
import pandas as pd

THEMES = {
    "under_center_vs_gun": {"under_center": 1, "rpo": -1},
    "dropback_play_action": {"play_action": 1, "time_to_throw": 1},
    "wide_zone_package": {"two_back": 1, "motion": 1, "outside_run": 1, "pistol": 1, "rb_target_share": 1},
    "te_heavy": {"multi_te": 1, "te_target_share": 1},
    "qb_run_game": {"qb_design_run": 1, "qb_out_of_pocket": 1},
    "tempo": {"no_huddle": 1},
    "pass_first": {"proe_early": 1, "empty_backfield": 1},
}

# (low pole, high pole) for labeling axes and describing teams
THEME_POLES = {
    "under_center_vs_gun": ("Shotgun / RPO", "Under center"),
    "dropback_play_action": ("Quick game", "Play action, longer dropbacks"),
    "wide_zone_package": ("Spread, inside runs", "Two-back, motion, outside zone"),
    "te_heavy": ("WR-centric", "Multi-TE, TE targets"),
    "qb_run_game": ("Pocket QB", "Designed QB runs, QB on the move"),
    "tempo": ("Huddles", "No-huddle"),
    "pass_first": ("Run-leaning early downs", "Pass-first early downs, empty"),
}

THEME_LABELS = {
    "under_center_vs_gun": "Under center vs gun/RPO",
    "dropback_play_action": "Dropback play action",
    "wide_zone_package": "Wide-zone package",
    "te_heavy": "TE-heavy",
    "qb_run_game": "QB run game",
    "tempo": "Tempo",
    "pass_first": "Pass-first",
}

THEME_FEATURES = sorted({f for w in THEMES.values() for f in w})


def within_season_z(df, seasons):
    return df.groupby(seasons).transform(lambda s: (s - s.mean()) / s.std())


def theme_scores(z_features, seasons, themes=THEMES):
    """z_features: within-season z-scores (DataFrame with the theme features)."""
    raw = pd.DataFrame({
        name: (z_features[list(w)] * pd.Series(w)).mean(axis=1) for name, w in themes.items()
    })
    return within_season_z(raw, seasons)
