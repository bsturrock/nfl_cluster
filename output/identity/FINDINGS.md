# NFL Offensive Identity Clusters (2022-2025)

128 team-seasons, regular season, neutral game script (WP 20-80%, outside the final 2 minutes of each half).
Features measure tendencies only (what an offense chooses to do), never efficiency (no EPA or success rate).
The model clusters on **7 named themes** (for example "Under center vs gun/RPO", "Wide-zone package") built from 16 features.
Code: `src/identity/`. Full tables: `output/identity/*.csv`. Interactive scatter: `output/identity/identity_scatter.html`.

## 0. Bottom line on how strong the clusters are

**Offensive identity is mostly a continuum, not a set of natural types.** The clusters are a useful way to cut that continuum, but the data doesn't strongly support discrete groups.

- Against a strict random-data baseline (section 4), the best clustering (k=5) is only 1.1 SD above random. Roughly 1 in 10 structureless datasets with the same feature distributions clusters as well.
- A looser baseline used in earlier versions of this report showed about 5 SD. It overstated the evidence, because it ignored that several themes have long tails: a few teams far out on tempo, wide-zone and QB run.
- **What is solid is the positions, not the boundaries.**
  - Theme scores are highly reliable within a season (0.72 to 0.96).
  - Teams keep their cluster the next season 58% of the time, against 29% by chance.
  - The distinctive groups (Shanahan wide-zone, QB-run/RPO, heavy power) are real tails of the distribution.
  - The big spread and under-center groups are broad regions of one continuum.

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

## 9. Limitations

- **The clusters segment a continuum.** Hard labels overstate the differences between neighbors, so treat each team-season's theme scores as the primary description and its cluster as shorthand.
- **Four seasons only.** FTN charting (motion, PA, RPO, screens, QB out of pocket) starts in 2022.
- **Themes are hand-built.** The groupings are analyst judgment, checked against the correlations and each feature's fit. Pass-first is the weakest theme (fit 0.25).
- **Every theme gets equal weight.** Tempo is one feature but counts as much as the five-feature wide-zone package.
- **No blocking-scheme tag.** nflverse doesn't carry one (zone vs gap). Outside-run share, average backs and pistol stand in as proxies.
- **QB traits blend into identity.** The QB run game theme and quick-throw rate partly reflect the quarterback, not only the play caller.
