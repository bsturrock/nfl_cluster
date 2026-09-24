# NFL Offense Clustering

Describes NFL offensive identity (what an offense tries to do, not how well it does it) for every team-season from 2022 to 2025, using nflverse data.
It started as a clustering project. **The main finding is that offensive identity is mostly a continuum.**
The one natural offensive type is the wide-zone family (SF, MIA, ATL, BAL, LAC 2024-25).
Everything else is better described with continuous profiles than with cluster labels. Full write-up: [`output/identity/FINDINGS.md`](output/identity/FINDINGS.md).
Plan for the next session: [`NEXT_STEPS.md`](NEXT_STEPS.md).

## Data

nflverse release files (fetched by `src/identity/fetch_data.sh` into `data/raw/`, which is not committed):

| source | used for |
|---|---|
| play-by-play (nflfastR) | down/distance, xpass, run location/gap, air yards, receivers |
| FTN charting (2022+) | play action, motion, RPO, screens, no-huddle, QB location, QB out of pocket, sneaks, backfield count |
| participation | personnel, time to throw, targeted route |
| players | roster position (QB detection, receiver position) |

FTN charting starts in 2022, which sets the season range. All features are measured in **neutral game script** (`src/neutral_script.py`: win probability 20-80%, outside the last 2 minutes of each half), regular season only.

## Layout

```
src/neutral_script.py                 neutral game-script filter (shared)
src/identity/fetch_data.sh            download nflverse parquet files -> data/raw/
src/identity/build_identity_features.py  play loading + whole-offense features -> data/identity_features.csv
src/identity/reliability.py           split-half / per-game / year-over-year reliability of features and themes
src/identity/themes.py                the 7 theme composites (definitions + scoring)
src/identity/cluster_identity.py      whole-offense model: k-means on themes, strict (copula) null, k=5/k=6 labels
src/identity/sections.py              run / pass / tendency section models; wide-zone family flag
src/identity/compare_variants.py      PCA vs skew-fix vs themes vs binning comparison
src/identity/build_scatter.py         interactive scatter (output/identity/identity_scatter.html)
src/templates/identity_scatter_template.html
output/identity/                      all results (CSV) + FINDINGS.md
output/identity/sections/             section model results
legacy/                               frozen v1 scheme clustering (see legacy/README.md)
```

## Run

```
pip install -r requirements.txt
bash src/identity/fetch_data.sh
python3 src/identity/build_identity_features.py   # features + data/raw/neutral_plays.parquet
python3 src/identity/reliability.py               # needs neutral_plays.parquet
python3 src/identity/cluster_identity.py          # ~1 min
python3 src/identity/build_scatter.py
python3 src/identity/sections.py                  # ~1 min
python3 src/identity/compare_variants.py          # optional, ~3 min
```

## Method conventions (keep these when extending)

- **Grain is team-season.** A single game's features are about 83% noise (median per-game ICC 0.17).
- **Every feature is z-scored within season** to remove league-wide drift.
- **Features must pass a season reliability screen** (split-half, Spearman-Brown corrected, at least 0.6).
- **Tendencies only.** No efficiency stats (EPA, success rate) as clustering inputs.
- **QB-driven traits stay out of the run and pass sections.** Designed QB runs, scrambles and RPO reads belong in a future "QB's job" section.
- **Structure is tested against the copula null**: structureless data with the same marginal distributions and rank correlations. The Gaussian null ignores the features' long tails and overstates structure. Report both, and decide on the copula null.
