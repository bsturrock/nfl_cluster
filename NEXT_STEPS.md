# Next steps

Starting point for the next session. Read `output/identity/FINDINGS.md` section 0 first.
In short: one natural offensive type (the wide-zone family), one distinctive tail (extreme no-huddle/RPO), and a continuum everywhere else.

## Where things stand

- **Whole-offense model** (`cluster_identity.py`): k=5/k=6 k-means on 7 themes. It's a reasonable summary, but only 1.1 SD above the strict baseline, so it segments a continuum rather than finding types.
- **Section models** (`sections.py`):
  - run game: the wide-zone family, a real k=2 split
  - pass game: no groups
  - tendencies: no groups, plus the no-huddle/RPO tail
- **Scatter** (`identity_scatter.html`): theme axes, cluster colors, team tracing with season-to-season theme and feature changes.
- **Decisions already made:**
  - team-season grain
  - within-season z-scores
  - reliability of at least 0.6
  - tendencies only, no efficiency stats
  - QB-driven traits kept out of the run and pass sections
  - the copula null is the test of structure
  - year-over-year persistence is *not* a criterion, since turnover is expected

## Recommended order

### 1. Rebuild the output around profiles instead of clusters

This is where the evidence points. For each team-season:

- **Flags**, for the only real groups:
  - wide-zone family (`sections/section_flags.csv`)
  - no-huddle/RPO tail (for example no-huddle at least 2 SD above that season's average)
- **Section profiles:** z-scores per section (run, pass, tendency, and later QB) with plain-language extremes (features at or beyond 1 SD, like "more motion, fewer TEs"). The scratch trial showed these read far better than per-area cluster labels, which mislabel borderline teams.
- **Most similar offenses:** nearest neighbors in theme or section space, across all seasons. This answers "who plays like this?" without forcing boundaries, and works naturally on a continuum.
- **The k=5 cluster** stays as a one-word summary column.

Rework the scatter panel and `team_season_identity.csv` around these. The scatter keeps its theme axes.

### 2. "QB's job" section

The features parked so far:
- designed QB runs (as a share of designed runs)
- RPO rate
- QB out of pocket (all dropbacks vs play-action boots)
- scramble rate
- quick-throw rate
- FTN `read_thrown` (first read vs later progressions)
- throwaways and interception-worthy throws (FTN)

Use the same pipeline: reliability screen, within-season z-scores, strict null. The earlier trials suggest mobile-QB offenses (PHI, WAS, BAL, IND with Richardson, ARI with Murray) may form a real group here.

### 3. Validate against play-callers

Build a small table of the play-caller for each team-season (OC or HC who calls plays, and their coaching tree). nflverse has head coaches (`home_coach`/`away_coach` in pbp) but not OCs or play-callers, so it must be entered by hand (32 x 4 rows).

This gives two tests:
- **Does identity follow the play-caller?** When a play-caller changes teams, does the new team's profile move toward his old team's?
- **Do coaching trees sit closer together** in profile space than random pairs of teams?

That is stronger validation than the cluster statistics, and it tests the football claims directly.

## Improvements worth trying

- **Shrinkage for noisy situational tendencies.** 1st-and-10, 2nd-and-short, 3rd-and-short, red zone and 4th-down aggressiveness failed the reliability screen because there are too few neutral plays per team-season. An empirical-Bayes / beta-binomial shrinkage toward the league rate would make them usable.
  - For short-yardage and red zone, consider dropping the neutral-script filter, since the score matters less there.
- **Seasons before 2022 (2016-2021).** FTN charting (play action, motion, RPO, screens) doesn't exist before 2022, but participation data has formation and personnel back to 2016, and pbp has run gap/location, xpass and air yards.
  - A reduced run-section model (personnel on runs, under center/shotgun, outside runs) could test whether the wide-zone family shows up historically, for example the 2016-2021 Shanahan/McVay tree growth.
  - It would roughly triple the sample.
- **Next Gen Stats tables (nflverse `nextgen_stats`):**
  - **rushing `efficiency`** (distance traveled per rushing yard) is a plausible zone-vs-gap proxy, since zone runs travel more laterally;
  - **`percent_attempts_gte_eight_defenders`** shows how defenses treat the run game;
  - **receiving `avg_separation`/`avg_cushion`** gives scheme-created separation.

  They're player-week level, so aggregate to team-season and weight by attempts.
- **Detailed personnel shares** (11/12/13/21/22/10) alongside the averages, especially 21 and 12 for the wide-zone family.
- **Opponent adjustment.** Neutral-script tendencies still depend somewhat on opponents. A simple team + opponent fixed-effects model on play-level choices (pass, play action, motion) would give opponent-adjusted rates. Probably small, but cheap to check.
- **Fix position lookups.**
  - Use season-specific rosters (nflverse `rosters_{season}`) instead of the current players file for QB and receiver positions.
  - Decide deliberately how to treat Taysom Hill-type players (a gadget QB running the ball).

## Missing data (not in nflverse)

- **Blocking scheme** (zone vs gap/power): PFF-level charting. `outside_run`, average backs, pistol and possibly the NGS rushing efficiency are the proxies.
- **Motion type** (jet, orbit, shift vs motion at the snap): FTN has a single motion flag only.
- **Full route trees:** participation charts only the targeted receiver's route. Route-running data for all receivers would enable real concept analysis.
- **Play-caller identity:** needs a hand-built table (see step 3).

## Housekeeping

- `data/raw/` is not committed. Run `bash src/identity/fetch_data.sh` first; it pulls about 90 MB.
- `legacy/` is the frozen v1 pipeline; its README explains why it was replaced. It can be deleted once nobody needs it for reference.
- Trial code for the 0-1 scaling and two-stage experiments was not kept. Their results and settings are recorded in FINDINGS section 11, and both approaches were rejected.
