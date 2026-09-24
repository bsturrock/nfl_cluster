# NFL Offensive Identity Clusters (2022-2025)

128 team-seasons, regular season, neutral game script (WP 20-80%, outside the final 2 minutes of each half).
Features measure tendencies only (what an offense chooses to do), never efficiency (no EPA or success rate).
The model clusters on **7 named themes** (for example "Under center vs gun/RPO", "Wide-zone package") built from 17 features, so every axis means something.
Code: `src/identity/`. Full tables: `output/identity/*.csv`. Interactive scatter: `output/identity/identity_scatter.html`.

## 1. Grain: team-season, not week

Split-half and per-game reliability for each candidate feature (`feature_reliability.csv`):

| feature | season reliability | single-game ICC | year-over-year r | kept |
|---|---:|---:|---:|:--:|
| two_back | 0.985 | 0.757 | 0.763 | yes |
| no_huddle | 0.973 | 0.584 | 0.498 | yes |
| pers_11 | 0.964 | 0.497 | 0.452 | yes |
| under_center | 0.962 | 0.503 | 0.639 | yes |
| personnel_entropy | 0.962 | 0.478 | 0.482 | no, r=-0.82 with pers_11 |
| multi_te | 0.947 | 0.403 | 0.223 | yes |
| pistol | 0.943 | 0.486 | 0.579 | yes |
| extra_ol | 0.936 | 0.338 | 0.360 | yes |
| motion | 0.928 | 0.420 | 0.718 | yes |
| qb_design_run | 0.916 | 0.243 | 0.657 | yes |
| te_target_share | 0.843 | 0.157 | 0.321 | yes |
| qb_out_of_pocket | 0.841 | 0.163 | 0.565 | yes |
| rpo | 0.804 | 0.209 | 0.275 | yes |
| outside_run | 0.800 | 0.168 | 0.561 | yes |
| play_action | 0.770 | 0.103 | 0.255 | yes |
| proe_early | 0.738 | 0.144 | 0.419 | yes |
| time_to_throw | 0.718 | 0.138 | 0.196 | yes |
| rb_target_share | 0.711 | 0.107 | 0.444 | yes |
| empty_backfield | 0.669 | 0.082 | 0.239 | yes |
| screen | 0.630 | 0.076 | 0.176 | yes |
| middle_target | 0.535 | 0.051 | 0.250 | no, reliability < 0.6 |
| adot | 0.477 | 0.037 | 0.229 | no, reliability < 0.6 |
| deep_rate | 0.177 | 0.012 | 0.196 | no, reliability < 0.6 |

Median single-game ICC is 0.17: one game's feature values are about 83% noise. A full season is about 84% signal.
Clustering weekly would mostly cluster sampling noise, so the model uses team-season.

The theme scores are more reliable than most single features, because each one averages several features:

| theme | season reliability | year-over-year r |
|---|---:|---:|
| Wide-zone package | 0.958 | 0.800 |
| Tempo | 0.944 | 0.443 |
| TE-heavy | 0.911 | 0.254 |
| Under center vs gun/RPO | 0.900 | 0.619 |
| QB run game | 0.900 | 0.665 |
| Dropback play action | 0.734 | 0.243 |
| Pass-first | 0.724 | 0.348 |

## 2. Themes

Each feature is z-scored **within season**. League-wide drift is large (motion went from 42% of neutral plays in 2022 to 60% in 2025).
Without this step, season, not identity, becomes a main cluster axis.

Features are then averaged into 7 themes with equal weights. Two-option themes have a real choice at each end. One-sided themes run from "rarely does this" to "leans on it".
Groupings came from football logic, checked against the correlations. A feature stays in a theme only if it moves with the rest of that theme ("fit", the correlation with the theme's other features, >= 0.15).
Definitions are in `src/identity/themes.py`, and per-feature fit is in `theme_loadings.csv`.

| theme | type | low end ↔ high end | features (fit) | consistency (alpha) |
|---|---|---|---|--:|
| Under center vs gun/RPO | two-option | Shotgun / RPO ↔ Under center | under_center (0.58), rpo reversed (0.58) | 0.73 |
| Dropback play action | two-option | Quick game ↔ Play action, longer dropbacks | play_action (0.42), time_to_throw (0.42) | 0.59 |
| Wide-zone package | one-sided | Spread, inside runs ↔ Two-back, motion, outside zone | two_back (0.53), outside_run (0.51), pistol (0.43), motion (0.42), rb_target_share (0.20) | 0.66 |
| TE-heavy | one-sided | WR-centric ↔ Multi-TE, TE targets | multi_te (0.60), te_target_share (0.60) | 0.75 |
| QB run game | one-sided | Pocket QB ↔ Designed QB runs, QB on the move | qb_design_run (0.47), qb_out_of_pocket (0.47) | 0.64 |
| Tempo | one-sided | Huddles ↔ No-huddle | no_huddle | single feature |
| Pass-first | two-option | Run-leaning early downs ↔ Pass-first early downs, empty | proe_early (0.25), empty_backfield (0.25) | 0.40 (weak) |

- **Features left out of the themes:** `extra_ol` and `screen` fit no theme (0.05 and 0.11). `pers_11` is left out because it is roughly the opposite of multi-TE plus two-back, so including it would count the same thing twice. All three remain in the feature table and the scatter for reference.
- **Themes are mostly independent** (`run_log.txt`). The largest correlations are under center with dropback play action (0.45), and QB run game with tempo (0.38).

## 3. Why themes instead of PCA

The first version clustered on 5 principal components of the 19 features. The PCs are blends of every feature, so movement on them was hard to explain.
`compare_variants.py` scores the alternatives the same way (`variant_comparison.csv`). Results at k=5:

| approach | silhouette | null mean | silhouette_z | bootstrap ARI | same cluster next season | ARI vs final model |
|---|--:|--:|--:|--:|--:|--:|
| PCA on 19 features (previous model) | 0.214 | 0.169 | 4.6 | 0.58 | 61.5% | 0.83 |
| PCA, log-transformed skewed features | 0.188 | 0.172 | 1.6 | 0.43 | 55.2% | 0.53 |
| 7 themes, all features | 0.182 | 0.141 | 4.1 | 0.42 | 60.4% | 0.50 |
| **7 themes, refined (final)** | 0.195 | 0.139 | **6.4** | 0.52 | 58.3% | 1.00 |
| 3-level binned features | 0.350 | 0.369 | -1.2 | 0.57 | 29.2% | 0.23 |

- **Refined themes win on margin over the null** and recover essentially the same clusters as the PCA model (ARI 0.83).
- **Log-transforming the skewed features hurts.** Those extreme values are the identity: SF/MIA at 40-60% two-back, WAS at 60%+ no-huddle.
- **Binning looks good on raw silhouette (0.35) but is below its own null.** The binning creates the apparent structure.

## 4. Choosing k

k-means on the 7 theme scores. Each k is compared with k-means on Gaussian data that has the same covariance and no clusters (30 draws).
`silhouette_z` is how far the real silhouette sits above that null. `bootstrap_ari` is the mean ARI of 50 bootstrap refits against the full-data fit. The table below is k-means; Ward and diagonal GMM are in `k_selection.csv`.

| k | silhouette | silhouette (19 features) | null mean | null p95 | silhouette_z | Calinski-Harabasz | Davies-Bouldin | bootstrap ARI | min size |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 0.155 | 0.101 | 0.163 | 0.184 | -0.54 | 25.4 | 2.05 | 0.26 | 59 |
| 3 | 0.168 | 0.089 | 0.149 | 0.164 | 2.02 | 25.9 | 1.78 | 0.58 | 18 |
| 4 | 0.180 | 0.099 | 0.145 | 0.161 | 3.20 | 24.9 | 1.60 | 0.53 | 13 |
| **5** | **0.195** | **0.110** | 0.141 | 0.154 | **5.85** | 24.2 | 1.46 | **0.57** | 7 |
| 6 | 0.180 | 0.093 | 0.139 | 0.156 | 4.06 | 24.0 | 1.48 | 0.53 | 10 |
| 7 | 0.182 | 0.084 | 0.138 | 0.153 | 4.52 | 23.1 | 1.43 | 0.52 | 6 |
| 8 | 0.174 | 0.082 | 0.137 | 0.155 | 3.88 | 22.5 | 1.42 | 0.49 | 5 |
| 9 | 0.161 | 0.080 | 0.133 | 0.144 | 3.58 | 21.5 | 1.44 | 0.46 | 5 |
| 10 | 0.156 | 0.076 | 0.137 | 0.150 | 2.20 | 20.9 | 1.37 | 0.49 | 3 |

- **k=5 is the primary model.** It has the largest margin over the null at any k, the best stability among k >= 4, and the best silhouette.
- **k=6 is the secondary model.** k=7 has a slightly larger margin (4.5 vs 4.1), but k=6 is as stable, has no cluster smaller than 10, and its one extra split has a clear football reading: spread divides into pass-first and balanced. The choice between k=6 and k=7 is a judgment call.

**Honest read:** the structure is real but modest. The silhouette of 0.195 against a null of 0.141 is about 6 null SDs above it, yet in absolute terms it is still a weak silhouette.
Offensive identity is mostly a continuum with a few distinct regions. Wide-zone is the most distinct group (silhouette 0.27). Under-center play-action is the tightest of the big groups (0.23).
Spread and QB-run/RPO blur into each other at the shotgun end. 2 of 128 team-seasons have negative silhouette.

## 5. Primary clusters (k=5)

Theme means are within-season z-scores (SD from that season's league average). Per-feature z and raw means are in `cluster_profiles_k5.csv`.

| | Shotgun spread, pass-first | Under-center play-action | Shotgun QB-run / RPO | Shanahan wide-zone, two-back motion | Heavy power run + QB run |
|---|--:|--:|--:|--:|--:|
| n | 48 | 45 | 18 | 10 | 7 |
| mean silhouette | 0.166 | 0.229 | 0.159 | 0.273 | 0.164 |
| Under center vs gun/RPO | -0.40 | **+0.88** | **-1.20** | +0.24 | -0.16 |
| Dropback play action | **-0.69** | **+0.75** | -0.27 | -0.25 | **+0.93** |
| Wide-zone package | -0.37 | -0.18 | -0.44 | **+2.32** | **+1.51** |
| TE-heavy | -0.27 | +0.34 | -0.27 | -0.78 | **+1.47** |
| QB run game | -0.33 | -0.27 | **+1.33** | -0.83 | **+1.81** |
| Tempo | -0.34 | -0.17 | **+1.72** | -0.54 | -0.21 |
| Pass-first | +0.24 | -0.18 | -0.01 | +0.32 | **-0.90** |

How to read each cluster:

- **Shotgun spread, pass-first:** shotgun, quick game, the most early-down passing, and nothing extreme otherwise. This is the league's default.
- **Under-center play-action:** under center, play action off longer dropbacks, moderately TE-heavy.
- **Shotgun QB-run / RPO:** the most shotgun and RPO, heavy no-huddle, designed QB runs.
- **Shanahan wide-zone:** a far outlier on the wide-zone package (two-back, motion, outside zone, pistol, RB targets), with the QB kept in the pocket.
- **Heavy power + QB run:** the wide-zone package plus multi-TE sets, QB runs, play action, and the least early-down passing.

Members (most to least typical, by silhouette):

- **Shotgun spread, pass-first (48):** CIN25, CIN23, BUF24, TB25, CIN24, WAS23, JAX22, CIN22, SEA24, JAX23, DAL24, CAR23, KC25, KC22, NE22, IND22, LAC25, HOU25, KC24, NYJ24, TEN25, TB22, CAR25, KC23, PIT23, CLE24, PIT25, DEN25, BUF22, NYJ22, NE23, PIT22, GB24, NYG24, WAS22, JAX25, GB22, GB25, LA23, CAR24, NE24, TB24, NO23, IND25, JAX24, DEN24, DAL23, TB23
- **Under-center play-action (45):** MIN24, TEN22, TEN23, MIN23, DET24, LA25, LV25, LA24, HOU23, CLE25, CLE22, SEA25, MIN22, CHI25, DAL22, HOU22, GB23, DAL25, LV22, HOU24, NYJ23, NE25, DET22, DET23, CAR22, MIN25, LA22, LAC24, TEN24, ARI25, NO24, BUF25, LV24, PIT24, LAC22, ARI24, DET25, DEN22, ARI23, LV23, CLE23, SEA23, DEN23, SEA22, NO22
- **Shotgun QB-run / RPO (18):** WAS25, WAS24, PHI23, PHI22, ARI22, NYG25, IND23, PHI24, PHI25, NO25, LAC23, BUF23, BAL23, NYG22, CHI24, NYG23, NYJ25, IND24
- **Shanahan wide-zone, two-back motion (10):** MIA25, MIA24, MIA23, SF25, SF22, MIA22, ATL24, SF23, ATL25, SF24
- **Heavy power run + QB run (7):** ATL22, CHI23, BAL22, ATL23, BAL24, CHI22, BAL25

Compared with the earlier PCA model, 8 of 128 team-seasons changed cluster:

- **To QB-run/RPO:** BUF23, NYG23 and CHI24 (from spread), and NYG22 (from under-center).
- **Under-center to spread:** DAL23 and NO23.
- **Spread to under-center:** SEA23.
- **Power to under-center:** NO24.

## 6. Secondary clusters (k=6)

| cluster | n | silhouette | defining themes | members |
|---|--:|--:|---|---|
| Under-center play-action | 44 | 0.185 | under center +0.87, dropback PA +0.75 | MIN24, LA25, TEN22, LV25, DET24, CLE25, MIN23, SEA25, TEN23, CLE22, MIN22, GB23, CHI25, LA24, NE25, HOU22, DET23, LAC24, TEN24, NO24, HOU23, DET22, DAL25, HOU24, BUF25, DAL22, NYJ23, LV22, ARI25, LA22, LV24, SEA23, CLE23, CAR22, LAC22, ARI24, MIN25, DET25, DEN22, NO22, TB23, LV23, DAL23, SEA22 |
| Shotgun spread, balanced | 32 | 0.178 | TE-heavy -0.75, nothing else strong | CAR23, PIT23, NYG24, PIT22, LAC25, HOU25, IND22, TB25, CHI24, GB25, IND24, BUF22, DEN24, WAS22, NE22, DEN25, LA23, GB24, CAR25, BAL23, NYG23, DAL24, TB22, NO23, JAX25, NYG22, CAR24, GB22, JAX24, DEN23, TEN25, NYJ24 |
| Pass-first spread | 19 | 0.181 | pass-first +0.89, dropback PA -1.00 (quick game), gun -0.89 | CIN24, KC24, KC22, CIN23, KC25, PIT25, CIN25, KC23, CIN22, JAX23, NE24, NE23, WAS23, SEA24, CLE24, IND25, JAX22, BUF24, NYJ22 |
| Shotgun QB-run / tempo | 12 | 0.214 | tempo +2.33, gun -1.51, QB run +1.27 | WAS24, WAS25, ARI22, PHI23, PHI22, IND23, PHI25, PHI24, BUF23, NYG25, LAC23, NO25 |
| Shanahan wide-zone, two-back motion | 11 | 0.221 | wide-zone +2.16 | MIA25, MIA24, MIA23, SF25, SF22, MIA22, SF23, ATL24, ATL25, SF24, TB24 |
| Heavy power run + QB run | 10 | 0.074 | QB run +1.68, TE-heavy +1.25, wide-zone +1.04, pass-first -1.09 | ATL22, CHI23, BAL24, CHI22, ATL23, BAL25, BAL22, ARI23, NYJ25, PIT24 |

k=6 splits spread into a pass-first, quick-game group (KC and CIN every year) and a balanced group. The power cluster also absorbs three borderline cases (ARI23, NYJ25, PIT24), which is why its silhouette drops to 0.07.

## 7. Validation beyond fit statistics

- **Persistence:** 58.3% of teams stay in the same k=5 cluster the next season. If clusters were assigned at random at the same sizes, that figure would be 29.3%.
- **Always-same teams:** CIN, DET, JAX, KC, LV, MIA, MIN, PHI, SF, and TB are in the same k=5 cluster all four seasons.
- **Moves line up with staff or QB changes:**
  - ATL: power (Arthur Smith) to wide-zone (Zac Robinson, 2024)
  - WAS: spread to QB-run/RPO (Kingsbury and Daniels, 2024)
  - CHI: power (Fields) to QB-run/RPO (Williams, 2024) to under-center play action (Ben Johnson, 2025)
  - SEA: to under-center play action (Kubiak, 2025)
  - ARI: QB-run/RPO (Kingsbury) to under-center play action (Petzing, 2023)
  - NE: under-center play action (McDaniels, 2025)
  - BAL 2023: QB-run/RPO in Monken's first year, then back to heavy/pistol
- **Boundary cases:** these sit roughly equidistant between two centroids (`margin_k5` near 0), so treat their labels as soft: NYJ25, NO22, DEN24, ARI23, TB23, DAL23, SEA22, IND24, JAX24, IND25.

## 8. Output files (`output/identity/`)

| file | contents |
|---|---|
| `identity_scatter.html` | Interactive scatter. Axes are any theme or raw feature; k=5/k=6 toggle; team tracing with season-to-season theme and feature changes |
| `team_season_identity.csv` | One row per team-season: k=5 and k=6 cluster and label, per-point silhouette, runner-up cluster, margin, 7 theme scores, raw features |
| `cluster_profiles_k5.csv`, `cluster_profiles_k6.csv` | Per cluster: n, mean silhouette, theme and feature z means, raw feature means, league mean |
| `cluster_members_k5.csv`, `cluster_members_k6.csv` | Members, ordered most to least typical |
| `k_selection.csv` | k=2..10 for k-means, Ward and GMM: silhouette (theme and full space), null silhouette, z, CH, DB, bootstrap ARI, min size |
| `theme_loadings.csv` | Theme definitions; each feature's correlation with its theme and fit with the rest of the theme |
| `feature_reliability.csv` | Split-half season reliability, single-game ICC, year-over-year r for features and themes |
| `variant_comparison.csv`, `theme_consistency.csv` | The PCA / skew-fix / themes / binning comparison |
| `team_trajectories.csv`, `cluster_transitions.csv` | Team x season labels; season-to-season transition counts |

## 9. Limitations

- **Four seasons only.** FTN charting (motion, PA, RPO, screens, QB out of pocket) starts in 2022.
- **Themes are hand-built.** The groupings are my judgment, checked against the correlations and each feature's fit, so they carry more analyst choice than PCA does. Pass-first is the weakest theme (alpha 0.40).
- **Every theme gets equal weight in the clustering.** Tempo is one feature but counts as much as the five-feature wide-zone package.
- **No blocking-scheme tag.** nflverse doesn't carry one (zone vs gap). Outside-run share, two-back and pistol stand in as proxies.
- **QB traits blend into identity.** The QB run game theme and time to throw partly reflect the quarterback, not only the play caller.
- **Cluster labels were written by hand** from the profiles. If features or themes change, re-read the profiles before reusing the labels.
