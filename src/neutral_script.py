"""Neutral game-script filter, shared by all feature-building scripts.

Play calling is distorted by score/clock: teams pass more when trailing big,
run more (and empty the playbook of anything cute) when leading big or
milking clock, and the final 2 minutes of each half force pass-heavy
hurry-up regardless of score. Restricting to neutral situations keeps the
gap/formation/personnel/PA features closer to a team's actual scheme
preference rather than a reaction to the scoreboard.

Definition (standard in public nflfastR analytics, e.g. rbsdm.com):
  - win probability (for the team with the ball) between 20% and 80%
  - not in the final 2 minutes of either half (hurry-up distorts calling
    even at a "neutral" win probability)
"""
WP_LOW = 0.20
WP_HIGH = 0.80
MIN_HALF_SECONDS_REMAINING = 120


def filter_neutral_script(plays):
    return plays[
        (plays["wp"].between(WP_LOW, WP_HIGH))
        & (plays["half_seconds_remaining"] > MIN_HALF_SECONDS_REMAINING)
    ]
