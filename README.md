# NFL Offense Clustering

Clusters NFL team-seasons on offensive tendency using nflverse data.

## Data

- Play-by-play: `nfl_data_py.import_pbp_data` (nflfastR)
- Play-action charting: `nfl_data_py.import_ftn_data` (FTN, 2022+ only — this
  bounds the season range below)

nflverse does not carry a run-blocking scheme (zone/power/gap) tag; that's
PFF charting-level detail. `run_location`/`run_gap` give direction and a
coarse gap (end/tackle/guard), which is what's used here as a scheme proxy.

## Features (v1)

Grain: team-season, 2022-2025, playcalled rushes only (scrambles/kneels
excluded). Restricted to **neutral game script** (`src/neutral_script.py`:
win probability 20-80%, outside the final 2 minutes of either half) so the
numbers reflect scheme preference rather than score/clock-driven play
calling. This roughly halves the play sample per team-season (~260 rushes,
~360 dropbacks on average) but is still plenty for these features.

- `pct_end`, `pct_tackle`, `pct_guard`, `pct_middle`: share of rush attempts
  by gap bucket (sums to 1; `pct_middle` held out of clustering to avoid
  compositional collinearity)
- `pa_rate`: play-action rate as a share of dropbacks (FTN charting)

## Usage

```
pip install -r requirements.txt
python3 src/build_features.py   # -> data/team_season_features.csv
python3 src/cluster.py          # -> output/team_season_clusters.csv
python3 src/build_viz.py        # -> output/cluster_viz.html (interactive)
```

## Notes

- k=4 (KMeans) is the current default; silhouette scores are ~0.21-0.24
  across k=2-8, i.e. weak-to-moderate separation with just these 5 features.
  Expected to sharpen as more features (personnel, motion, formation, PA by
  down/distance, RPO rate) are added.
- Some teams cluster stably across seasons (SF, DET); others don't (MIA, NE),
  plausibly tracking coaching/scheme changes rather than noise.

## Offensive identity model (`src/identity/`)

Broader successor to the scheme clustering above. It uses 19 reliability-screened tendency features,
z-scores them within season, keeps PCs by parallel analysis, and chooses k against a null-model silhouette.
Results and all stats: `output/identity/FINDINGS.md`.

```
pip install -r requirements.txt
bash src/identity/fetch_data.sh                   # nflverse parquet -> data/raw/
python3 src/identity/build_identity_features.py   # -> data/identity_features.csv
python3 src/identity/reliability.py               # -> output/identity/feature_reliability.csv
python3 src/identity/cluster_identity.py          # -> output/identity/*.csv
python3 src/identity/build_scatter.py             # -> output/identity/identity_scatter.html
```
