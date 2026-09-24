# NFL Offensive Identity Clusters (2022-2025)

128 team-seasons, regular season, neutral game script (WP 20-80%, outside the final 2 minutes of each half).
Features measure tendencies only (what an offense chooses to do), never efficiency (no EPA or success rate).
The model clusters on **7 named themes** (for example "Under center vs gun/RPO", "Wide-zone package") built from 16 features.
Code: `src/identity/`. Full tables: `output/identity/*.csv`. Interactive scatter: `output/identity/identity_scatter.html`.

## 0. Bottom line

**Offensive identity is mostly a continuum, with one natural type and one distinctive tail.** Thirteen different setups (sections 3, 4, 10 and 11) converge on this.

- **The one natural type is the wide-zone family:** SF, MIA, ATL and BAL every season, plus LAC 2024-25 (the Shanahan / Greg Roman lineage).
  - Its members run with two backs, pistol, motion and outside runs, and pass with the same personnel plus more RB targets and in-breaking routes.
  - It is the only group that beats the strict random-data baseline convincingly in its own section: the run game at 5.4 SD, stability 0.74 (section 10).
- **The one distinctive tail is extreme no-huddle and RPO:** WAS 2024-25, PHI 2022-23, ARI 2022, IND 2023, LAC 2023. It's a handful of teams far out on a scale, not a type.
- **Everything else varies continuously.** The whole-offense model is only 1.1 SD above the strict baseline at k=5, and roughly 1 in 10 structureless datasets with the same feature distributions clusters as well.
  - The pass game and play-calling tendencies show no groups at all; a Gaussian mixture model prefers a single group for both.
  - The k=5/k=6 clusters in sections 5-6 are a reasonable way to cut the continuum, not natural types.
- **What is solid is each team's position, not the boundaries.**
  - Theme scores are reliable within a season (0.72 to 0.96).
  - Year-to-year moves line up with coaching and QB changes.
  - That points to describing offenses with continuous profiles, flags and "most similar offenses" rather than hard cluster labels (see `NEXT_STEPS.md`).
- **An early version of this report overstated the structure.** It showed ~5 SD against a Gaussian null, which ignored the themes' long tails. All decisions now use the copula null (section 3).

## 1. Grain: team-season, not week

Split-half and per-game reliability for each candidate feature (`feature_reliability.csv`):

| feature | season reliability | single-game ICC | year-over-year r | used |
|---|---:|---:|---:|:--:|
| avg_backs | 0.983 | 0.741 | 0.751 | yes |
| no_huddle | 0.973 | 0.584 | 0.498 | yes |
| under_center | 0.962 | 0.503 | 0.639 | yes |
| avg_te | 0.951 | 0.427 | 0.213 | yes |
| extra_ol | 0.936 | 0.338 | 0.360 | display only (fits no theme) |
| pistol_of_gun | 0.934 | 0.468 | 0.548 | yes |
| motion | 0.928 | 0.420 | 0.718 | yes |
| qb_run_share | 0.915 | 0.300 | 0.687 | yes |
| te_target_share | 0.843 | 0.157 | 0.321 | yes |
| qb_out_of_pocket | 0.841 | 0.163 | 0.565 | yes |
| quick_throw | 0.816 | 0.179 | 0.159 | yes |
| rpo | 0.804 | 0.209 | 0.275 | yes |
| outside_run | 0.800 | 0.168 | 0.561 | yes |
| play_action | 0.770 | 0.103 | 0.255 | yes |
| proe_early | 0.738 | 0.144 | 0.419 | yes |
| rb_target_share | 0.711 | 0.107 | 0.444 | yes |
| empty_backfield | 0.669 | 0.082 | 0.239 | yes |
| screen | 0.630 | 0.076 | 0.176 | display only (fits no theme) |
| middle_target | 0.535 | 0.051 | 0.250 | no, reliability < 0.6 |
| adot | 0.477 | 0.037 | 0.229 | no, reliability < 0.6 |
| deep_rate | 0.177 | 0.012 | 0.196 | no, reliability < 0.6 |

- **Feature changes since v1:**
  - `avg_te` and `avg_backs` (average on field) replace the 2+ TE and 2+ back shares.
  - `pistol_of_gun` (pistol vs shotgun among gun snaps) replaces pistol's share of all snaps.
  - `qb_run_share` (QB runs among designed runs) replaces QB runs per play.
  - `quick_throw` (share of dropbacks released in 2.5s or less) replaces median time to throw. It is more reliable (0.82 vs 0.72).
  - 11 personnel and personnel entropy are dropped, since average TEs and average backs capture them.
- **Grain:** the median single-game ICC is 0.17, so one game's feature values are about 83% noise. A full season is about 82% signal. Clustering weekly would mostly cluster noise, so the model uses team-season.

## 2. Themes

Each feature is z-scored **within season**. League-wide drift is large (motion went from 42% of neutral plays in 2022 to 60% in 2025).
Features are then averaged into 7 themes with equal weights, and the themes are re-standardized within season.

Two-option themes have a real choice at each end. One-sided themes run from "rarely does this" to "leans on it". Groupings came from football logic, checked against the correlations.
A feature stays in a theme only if it moves with the rest of that theme ("fit", its correlation with the theme's other features, >= 0.15).
Definitions are in `src/identity/themes.py`; per-feature fit is in `theme_loadings.csv`.

| theme | type | low end ↔ high end | features (fit) | season reliability |
|---|---|---|---|--:|
| Under center vs gun/RPO | two-option | Shotgun / RPO ↔ Under center | under_center (0.58), rpo reversed (0.58) | 0.90 |
| Dropback play action | two-option | Quick game ↔ Play action, longer dropbacks | play_action (0.45), quick_throw reversed (0.45) | 0.77 |
| Wide-zone package | one-sided | Spread, inside runs ↔ Two-back, motion, outside zone | avg_backs (0.54), outside_run (0.51), pistol_of_gun (0.46), motion (0.42), rb_target_share (0.20) | 0.96 |
| TE-heavy | one-sided | WR-centric ↔ Multi-TE, TE targets | avg_te (0.60), te_target_share (0.60) | 0.91 |
| QB run game | one-sided | Pocket QB ↔ Designed QB runs, QB on the move | qb_run_share (0.47), qb_out_of_pocket (0.47) | 0.90 |
| Tempo | one-sided | Huddles ↔ No-huddle | no_huddle | 0.94 |
| Pass-first | two-option | Run-leaning early downs ↔ Pass-first early downs, empty | proe_early (0.25), empty_backfield (0.25) | 0.72 |

## 3. Choosing k

k-means on the 7 theme scores. Two random-data baselines are reported (30 draws each). Both are k-means run on data with the same shape but no clusters:

- **Strict baseline (k is chosen on this one):** keeps each theme's exact distribution, tails included, and the rank correlations between themes (a Gaussian copula).
- **Gaussian baseline:** keeps the covariance but assumes bell-shaped themes. It has no long tails, so the tail teams alone look like extra structure. It overstates the evidence and is shown only for comparison with earlier versions.

`bootstrap_ari` is the mean ARI of 50 bootstrap refits against the full-data fit. Ward and diagonal GMM are in `k_selection.csv`.

| k | silhouette | strict baseline mean | strict baseline p95 | SD above strict | SD above Gaussian | bootstrap ARI | min size |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 0.155 | 0.175 | 0.219 | -0.9 | -1.0 | 0.29 | 64 |
| 3 | 0.170 | 0.175 | 0.204 | -0.2 | 1.2 | 0.56 | 18 |
| 4 | 0.184 | 0.169 | 0.196 | 0.9 | 4.1 | 0.57 | 13 |
| **5** | **0.190** | 0.169 | 0.202 | **1.1** | 4.7 | 0.50 | 11 |
| 6 | 0.179 | 0.164 | 0.188 | 0.9 | 4.9 | 0.51 | 10 |
| 7 | 0.175 | 0.162 | 0.182 | 1.0 | 4.7 | 0.50 | 6 |
| 8 | 0.169 | 0.163 | 0.183 | 0.4 | 3.1 | 0.45 | 3 |
| 9 | 0.163 | 0.161 | 0.180 | 0.2 | 2.6 | 0.50 | 6 |
| 10 | 0.168 | 0.160 | 0.183 | 0.6 | 3.3 | 0.48 | 4 |

- **k=5 is the primary model.** It has the largest margin over the strict baseline, but that margin is modest. No k clears the baseline's 95th percentile.
- **k=6 is the secondary model.** It splits spread into pass-first (KC and CIN every year) and balanced.
- Ward at k=5 gives nearly the same answer (1.3 SD above strict baseline, bootstrap ARI 0.58).

## 4. Other approaches tried

`compare_variants.py` scores alternatives the same way (`variant_comparison.csv`). Results at k=5:

| approach | silhouette | SD above strict | SD above Gaussian | bootstrap ARI | same cluster next season | ARI vs model |
|---|--:|--:|--:|--:|--:|--:|
| PCA on all 18 features | 0.191 | 1.2 | 2.5 | 0.42 | 54.2% | 0.43 |
| PCA, log-transformed skewed features | 0.202 | 2.6 | 2.6 | 0.48 | 52.1% | 0.38 |
| Themes incl. extra_ol and screen | 0.185 | 1.1 | 5.0 | 0.39 | 56.2% | 0.38 |
| **Themes (model)** | 0.190 | 1.5 | 5.2 | **0.51** | **58.3%** | 1.00 |
| 3-level binned features | 0.352 | -1.1 | -0.9 | 0.68 | 33.3% | 0.28 |

(The model's margin shows as 1.5 here and 1.1 in section 3 because the two runs use different random draws.)

- **No approach clearly separates from the others under the strict baseline.** All sit 1 to 3 SD above it, and the ranking shifts between runs and values of k.
- **The theme model is kept** because it is the most stable, the most persistent year to year, and the only one with named axes. It was not chosen for the highest cluster score.
- **Binning's high raw silhouette (0.35) is below its own baseline.** The binning creates the apparent structure.
- **A separate scratch trial scaled all features to 0-1** (raw, min-max, and within-season percentile), not included in the table:
  - Raw and min-max were worse. Raw 0-1 lets high-spread features dominate; in min-max, a few extreme teams compress everyone else.
  - Percentile-scaled features were competitive (about 3 SD above the strict baseline at k=5), with different groups:
    - Under-center split into an 11-personnel + motion group and a TE-heavy play-action group.
    - Wide-zone widened to GB, TB and NYJ.
  - The cost: it lost the QB-run/RPO group and was less stable (bootstrap 0.43).

## 5. Primary clusters (k=5)

Theme means are within-season z-scores (SD from that season's league average). Per-feature z and raw means are in `cluster_profiles_k5.csv`.
Labels are assigned from each cluster's theme signature, not its size, so a refit can't silently swap names.

| | Shotgun spread, pass-first | Under-center play-action | Shotgun QB-run / RPO | Shanahan wide-zone, two-back motion | Heavy power run + QB run |
|---|--:|--:|--:|--:|--:|
| n | 50 | 44 | 12 | 11 | 11 |
| mean silhouette | 0.151 | 0.233 | 0.235 | 0.240 | 0.102 |
| Under center vs gun/RPO | -0.44 | **+0.88** | **-1.51** | +0.30 | -0.16 |
| Dropback play action | **-0.63** | **+0.77** | -0.27 | -0.43 | +0.54 |
| Wide-zone package | -0.40 | -0.17 | -0.65 | **+2.16** | **+1.05** |
| TE-heavy | -0.28 | +0.36 | -0.09 | -0.77 | +0.72 |
| QB run game | -0.24 | -0.31 | **+1.36** | -0.81 | **+1.66** |
| Tempo | -0.27 | -0.20 | **+2.33** | -0.55 | +0.05 |
| Pass-first | +0.18 | -0.16 | +0.43 | +0.35 | **-0.98** |

Members (most to least typical, by silhouette):

- **Shotgun spread, pass-first (50):** BUF24, CIN23, CIN25, CIN24, JAX22, TB25, KC25, WAS23, SEA24, JAX23, DAL24, CIN22, CAR23, KC22, NE22, CAR25, IND22, DEN25, PIT23, KC24, HOU25, NE23, NYG24, LAC25, BUF22, TB22, TEN25, CLE24, CAR24, GB24, PIT22, GB25, KC23, PIT25, NYJ24, NO23, IND24, NYJ22, NE24, JAX25, CHI24, IND25, JAX24, GB22, WAS22, LA23, NYG23, DEN24, DAL23, TB23
- **Under-center play-action (44):** MIN24, TEN22, LA25, CLE25, TEN23, DET24, LV25, HOU23, HOU22, MIN23, CLE22, MIN22, DET23, LA24, SEA25, DAL22, DAL25, NYJ23, LV22, GB23, CHI25, ARI25, HOU24, MIN25, LAC24, TEN24, LAC22, CAR22, ARI24, LV24, NE25, NO24, LA22, CLE23, DET22, BUF25, SEA23, LV23, ARI23, SEA22, DET25, NO22, DEN23, DEN22
- **Shotgun QB-run / RPO (12):** WAS24, PHI22, PHI23, ARI22, WAS25, IND23, PHI25, NYG25, BUF23, LAC23, NO25, PHI24
- **Shanahan wide-zone, two-back motion (11):** MIA25, MIA24, MIA23, SF22, SF25, MIA22, ATL24, SF23, ATL25, SF24, TB24
- **Heavy power run + QB run (11):** CHI23, ATL22, CHI22, BAL24, BAL22, BAL25, ATL23, NYJ25, BAL23, NYG22, PIT24

Versus the previous feature set, 8 team-seasons changed cluster:

- **QB-run/RPO to heavy power:** BAL23, NYG22 and NYJ25.
- **QB-run/RPO to spread:** NYG23, CHI24 and IND24.
- **Under-center to heavy power:** PIT24.
- **Spread to wide-zone:** TB24.

5 of 128 team-seasons have a negative silhouette.

## 6. Secondary clusters (k=6)

| cluster | n | silhouette | defining themes | members |
|---|--:|--:|---|---|
| Under-center play-action | 41 | 0.178 | under center +0.84, dropback PA +0.82 | CLE25, TEN22, LA25, LV25, MIN23, CLE22, MIN24, CHI25, HOU22, DET24, MIN22, TEN23, SEA25, ARI25, CLE23, NO24, GB23, DET23, LA24, DAL22, ARI23, TEN24, BUF25, LAC24, LV24, ARI24, SEA23, HOU23, DAL25, NYJ23, NE25, LAC22, SEA22, LA22, HOU24, CAR22, NO22, LV22, MIN25, DEN22, DAL23 |
| Shotgun spread, balanced | 36 | 0.178 | TE-heavy -0.74, nothing else strong | CAR23, HOU25, LAC25, PIT23, TB25, IND22, NYG24, GB25, NO23, PIT22, BUF22, NE22, LA23, CAR24, DEN24, CHI24, IND24, TB22, GB24, NYJ24, DEN25, DAL24, JAX25, GB22, WAS22, DEN23, CAR25, JAX24, TEN25, TB24, LV23, NYG23, NYJ22, DET25, TB23, DET22 |
| Pass-first spread | 18 | 0.188 | dropback PA -1.06 (quick game), gun -0.94, pass-first +0.90 | CIN24, KC24, KC22, KC25, CIN23, PIT25, CIN25, KC23, CIN22, JAX23, IND25, NE24, JAX22, NE23, SEA24, WAS23, CLE24, BUF24 |
| Shotgun QB-run / tempo | 11 | 0.253 | tempo +2.40, gun -1.51, QB run +1.25 | WAS24, PHI23, ARI22, PHI22, WAS25, IND23, PHI25, BUF23, NYG25, LAC23, NO25 |
| Shanahan wide-zone, two-back motion | 10 | 0.231 | wide-zone +2.30 | MIA25, MIA24, MIA23, SF25, SF22, SF23, MIA22, ATL24, ATL25, SF24 |
| Heavy power run + QB run | 12 | 0.060 | QB run +1.74, pass-first -1.01, wide-zone +0.94 | CHI23, ATL22, CHI22, BAL24, BAL25, ATL23, BAL22, NYJ25, NYG22, BAL23, PHI24, PIT24 |

## 7. Validation beyond fit statistics

- **Persistence:** 58.3% of teams stay in the same k=5 cluster the next season. If clusters were assigned at random at the same sizes, that figure would be 29.4%.
- **Always-same teams:** BAL, CIN, DET, JAX, KC, LV, MIA, MIN, PHI, and SF are in the same k=5 cluster all four seasons.
- **Moves line up with staff or QB changes:**
  - ATL: power (Arthur Smith) to wide-zone (Zac Robinson, 2024)
  - WAS: spread to QB-run/RPO (Kingsbury and Daniels, 2024)
  - CHI: power (Fields) to spread (Williams, 2024) to under-center play action (Ben Johnson, 2025)
  - SEA: to under-center play action (Kubiak, 2025)
  - ARI: QB-run/RPO (Kingsbury) to under-center play action (Petzing, 2023)
  - NE: under-center play action (McDaniels, 2025)
- **Boundary cases:** these sit roughly equidistant between two centroids (`margin_k5` near 0), so treat their labels as soft: TB24, DAL23, ARI23, DEN22, NYG23, LA23, CHI24, TB23, DEN24, DEN23.

## 8. Output files (`output/identity/`)

| file | contents |
|---|---|
| `identity_scatter.html` | Interactive scatter. Axes are any theme or raw feature; k=5/k=6 toggle; team tracing with season-to-season theme and feature changes |
| `team_season_identity.csv` | One row per team-season: k=5 and k=6 cluster and label, per-point silhouette, runner-up cluster, margin, 7 theme scores, raw features |
| `cluster_profiles_k5.csv`, `cluster_profiles_k6.csv` | Per cluster: n, mean silhouette, theme and feature z means, raw feature means, league mean |
| `cluster_members_k5.csv`, `cluster_members_k6.csv` | Members, ordered most to least typical |
| `k_selection.csv` | k=2..10 for k-means, Ward and GMM: silhouette, both baselines and z-scores, CH, DB, bootstrap ARI, min size |
| `theme_loadings.csv` | Theme definitions; each feature's correlation with its theme and fit with the rest of the theme |
| `feature_reliability.csv` | Split-half season reliability, single-game ICC, year-over-year r for features and themes |
| `variant_comparison.csv`, `theme_consistency.csv` | The PCA / skew-fix / themes / binning comparison |
| `team_trajectories.csv`, `cluster_transitions.csv` | Team x season labels; season-to-season transition counts |
| `sections/section_features.csv` | Run, pass and tendency section features (all candidates) per team-season |
| `sections/section_reliability.csv` | Split-half season reliability of every section candidate feature |
| `sections/section_k_selection.csv` | Section models: silhouette, both baselines, bootstrap ARI, GMM-preferred k |
| `sections/section_flags.csv` | Wide-zone family flag per team-season |

## 9. Limitations

- **The clusters segment a continuum.** Hard labels overstate the differences between neighbors, so treat each team-season's theme scores as the primary description and its cluster as shorthand.
- **Four seasons only.** FTN charting (motion, PA, RPO, screens, QB out of pocket) starts in 2022.
- **Themes are hand-built.** The groupings are analyst judgment, checked against the correlations and each feature's fit. Pass-first is the weakest theme (fit 0.25).
- **Every theme gets equal weight.** Tempo is one feature but counts as much as the five-feature wide-zone package.
- **No blocking-scheme tag.** nflverse doesn't carry one (zone vs gap). Outside-run share, average backs and pistol stand in as proxies.
- **QB traits blend into identity.** The QB run game theme and quick-throw rate partly reflect the quarterback, not only the play caller.
- **QB detection uses current roster position** (nflverse players file). A player listed at another position, such as Taysom Hill (TE), is treated as a non-QB rusher, and a player's position change across seasons isn't tracked.
- **Receiver position is also current roster position,** so converted players (for example a WR who later moved to RB) are counted at their latest position.

## 10. Section models: where the structure lives (`src/identity/sections.py`)

Each part of the offense is clustered on its own features, all measured in neutral game script, z-scored within season, and screened for reliability (at least 0.6).
QB-driven traits are left out of the run and pass sections. Results are in `sections/section_k_selection.csv`.

**Run game** (non-QB designed runs only, so a mobile QB's keepers don't shape the profile).
Features: outside run share, average TEs and average backs on runs, under center and pistol-of-gun on runs, motion on runs, 6+ OL on runs. All have season reliability 0.80 to 0.98.

| k | SD above strict baseline | bootstrap ARI |
|--:|--:|--:|
| **2** | **5.4** (no random dataset matched) | **0.74** |
| 3 | -0.5 | 0.46 |
| 4-6 | 1.6 to 2.0 | 0.62 to 0.65 |

- **k=2 is the wide-zone family (18 team-seasons) vs everyone else**, and a Gaussian mixture model agrees on k=2.
  - Members: ATL22-25, BAL22-25, MIA22-25, SF22-25, LAC24-25.
  - Their traits (raw): 1.53 backs on runs (league 1.15), 45% pistol among gun runs (20%), 57% outside runs (48%), 66% motion (53%), fewer TEs.
- **Earlier trial versions also found a "QB run game" group** (PHI, WAS, BAL, CHI 2022-23, IND, ARI, BUF 2022-23). It disappears once designed QB runs are removed. It was a mobile-QB grouping, not a run-scheme grouping.
- **Adding run rate over expected and RPO gave a third group** (shotgun RPO: KC, CIN, PHI, IND, WAS, BUF). Both features moved to the tendency section: run rate is a play-calling choice, and the RPO read is the QB's.

**Pass game** (dropbacks). Features: under center, pistol-of-gun, average TEs/backs, empty, motion, play action, play-action boot rate, RB/TE target share, share of targets at 0-9 air yards, in-breaking and out routes.

- **Dropped for reliability:** aDOT 0.48, 10-19 yard targets 0.49, and three route families: verticals 0.37, hitches 0.54, slants 0.59.
- **Moved to tendency:** screens. Targets behind the line and flat/screen routes duplicate it (r 0.79-0.89).
- **Pass concepts alone** (no formation/personnel) never beat the strict baseline at any k, and a Gaussian mixture model prefers a single group.
- **With formation/personnel** there is one k=2 split (4.6 SD with Ward, 11 with k-means). It is the wide-zone family again, seen through its dropback personnel (1.39 backs on dropbacks vs 1.08, more motion and pistol, more RB targets and in-breakers). It is unstable (bootstrap 0.42) because borderline teams (BUF25, BAL24) flip in and out.
- **Conclusion:** passing concepts vary continuously. There is no pass-game type beyond the wide-zone family's personnel signature.

**Tendencies.** Features: early-down pass rate over expected, RPO rate, no-huddle, screen rate, personnel tell (pass rate from 3+ WR sets minus 2-WR-or-fewer sets), formation tell (gun pass rate minus under-center pass rate).

- **Dropped for reliability, since too few neutral plays per team-season:** 1st-and-10 pass rate 0.56, 2nd-and-short 0.42, 3rd-and-short run rate 0.51.
- **Dropped as redundant:** raw pass rate (r 0.94 with pass rate over expected).
- **k-means never gets past 1.6 SD, and a Gaussian mixture model prefers a single group.** Ward's k=2 isolates the no-huddle/RPO tail (6 team-seasons, 2.7 to 5.7 SD depending on the draw).
- The two tells are the most independent tendency features (correlation at most 0.30 with anything else). They're useful to describe, not to cluster on.

## 11. Experiment log (everything tried, including what didn't work)

"Strict" is the copula null. Numbers are at the k named. Scripts for trials not kept in `src/` are described here instead.

| approach | result | verdict |
|---|---|---|
| v1: 11 features, k-means on 2 PCs (legacy) | silhouette 0.416 at k=6 vs 0.352 for 2-D Gaussian random data | 2-D inflation plus features chosen to maximize silhouette; replaced |
| 19 features, PCA by parallel analysis, k=5 | 2.5 to 4.6 SD vs Gaussian (new vs original feature set), 1.2 SD vs strict | PCs hard to explain; replaced by themes |
| 7 themes (current model), k=5 | 4.7 to 5.8 vs Gaussian, 1.1 to 1.9 vs strict (varies with random draws and feature set) | kept for naming and stability, but weak structure |
| log-transform skewed features | 1.6 to 2.6 vs strict, less stable | the tails are the identity; rejected |
| 3-level and 2-level binning | raw silhouette 0.35 but below its own baseline | binning creates fake structure; rejected |
| 0-1 scaling, raw rates | dominated by high-spread features (motion 17% of weight, empty 0.7%); 1.8 vs strict | rejected |
| 0-1 scaling, min-max | extreme teams compress everyone else; 1.2 vs strict | rejected |
| 0-1 scaling, within-season percentile | 2.8 to 4.2 vs strict at k=5; splits under-center teams into 11-personnel/motion vs TE-heavy | competitive but loses the QB-run group, bootstrap 0.43, and amplifies noise in near-zero features; not adopted |
| two-stage: per-area clusters, then cluster the label combinations | only formation/personnel has groups (k=2, 2.2 SD); 79 distinct combinations of 128, 51 unique; stage 2 less stable | rejected as a model; per-area descriptions kept as an idea |
| run section with QB run share | 3.9 SD at k=3, but the extra group is mobile-QB teams | QB features removed by design |
| run section, 7 features | 5.4 SD at k=2, the wide-zone family | **the one robust type** |
| pass section | no groups beyond the wide-zone personnel signature | continuum |
| tendency section | no groups; no-huddle/RPO tail only | continuum |
