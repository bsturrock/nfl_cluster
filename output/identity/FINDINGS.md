# NFL Offensive Identity Clusters (2022-2025)

128 team-seasons, regular season, neutral game script (WP 20-80%, outside the final 2 minutes of each half).
Features measure tendencies only (what an offense chooses to do), never efficiency (no EPA or success rate).
Code: `src/identity/`. Full tables: `output/identity/*.csv`.

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

## 2. Features (19)

Z-scored **within season**. League-wide drift is large (motion went from 42% of neutral plays in 2022 to 60% in 2025).
Without this step, season, not identity, becomes a main cluster axis.

- **Philosophy:** `proe_early` (1st/2nd-down pass rate over expected)
- **Formation/personnel:** `under_center`, `pistol`, `pers_11`, `multi_te`, `two_back`, `empty_backfield`, `extra_ol`
- **Pre-snap/tempo:** `motion`, `no_huddle`
- **Run game:** `outside_run`, `qb_design_run` (QB runs that aren't scrambles or sneaks), `rpo`
- **Pass game:** `play_action`, `screen`, `qb_out_of_pocket`, `time_to_throw`, `rb_target_share`, `te_target_share`

## 3. Dimensionality

Horn's parallel analysis keeps 5 PCs (60.1% of variance). Each PC's eigenvalue beats the 95th percentile from random data (`pca_loadings.csv`).

| PC | var % | main loadings (+ / -) |
|---|---:|---|
| PC1 | 17.6 | + 11 personnel, RPO, PROE / - two-back, outside run, play action, motion |
| PC2 | 14.4 | + QB design run, pistol, no-huddle, QB out of pocket / - under center |
| PC3 | 11.3 | + time to throw, TE targets, under center, play action / - motion, two-back |
| PC4 | 8.7 | + multi-TE, TE targets, RB targets / - 11 personnel, time to throw |
| PC5 | 8.1 | + PROE, empty, TE targets / - screen, RB targets |

## 4. Choosing k

Raw silhouette falls as dimensions increase, so each k is compared with a **null model**: k-means run on Gaussian data with the same covariance and no clusters (30 draws).
`silhouette_z` is how far the real silhouette sits above that null. `bootstrap_ari` is the mean ARI of 50 bootstrap refits against the full-data fit. The table below is k-means; Ward and diagonal GMM are in `k_selection.csv`.

| k | silhouette (5-PC) | silhouette (19-D) | null mean | null p95 | silhouette_z | Calinski-Harabasz | Davies-Bouldin | bootstrap ARI | min size |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 0.183 | 0.104 | 0.183 | 0.204 | -0.04 | 29.6 | 1.89 | 0.36 | 59 |
| 3 | 0.198 | 0.103 | 0.177 | 0.194 | 2.02 | 31.9 | 1.56 | 0.63 | 13 |
| 4 | 0.199 | 0.103 | 0.172 | 0.186 | 3.03 | 31.2 | 1.47 | 0.58 | 13 |
| **5** | **0.214** | **0.109** | 0.167 | 0.183 | **4.48** | 30.3 | 1.37 | **0.59** | 8 |
| 6 | 0.205 | 0.098 | 0.170 | 0.188 | 2.72 | 29.6 | 1.31 | 0.45 | 7 |
| 7 | 0.205 | 0.098 | 0.167 | 0.182 | 3.44 | 29.7 | 1.27 | 0.43 | 7 |
| 8 | 0.189 | 0.084 | 0.173 | 0.189 | 1.35 | 28.6 | 1.28 | 0.47 | 6 |
| 9 | 0.205 | 0.081 | 0.172 | 0.189 | 2.50 | 27.8 | 1.24 | 0.46 | 7 |
| 10 | 0.200 | 0.083 | 0.173 | 0.191 | 2.29 | 26.7 | 1.19 | 0.45 | 5 |

- **k=5 is the primary model.** It has the best silhouette, the largest margin over the null among k >= 3, and the best stability among k >= 4.
- It is also robust to dimensionality. At 3, 4, 6, 7 and 19 dimensions, k=5 still has the strongest null margin or is close to it, and the assignments agree with the 5-PC fit (ARI 0.87 to 1.0).
- **k=7 is the secondary model.** It is the second-best result above the null, and it splits the two large k=5 groups along clear lines.
- k-means beats Ward and diagonal GMM at every k from 3 up. GMM at k=2 has a higher z (5.4) but is not a useful segmentation.

**Honest read:** the structure is real but modest. At k=5 the silhouette is 0.21 against a null of 0.17, which is about 4.5 null SDs above it.
Offensive identity is mostly a continuum with a few dense regions. Two groups are sharply separated: Shotgun QB-run/RPO (silhouette 0.30) and Shanahan wide-zone (0.29).
The two big groups (spread, under-center play-action) are broad regions, not tight clusters. 3 of 128 team-seasons have negative silhouette.

For comparison, the repo's earlier `scheme_clusters` model (k=6 on 2 PCs, silhouette 0.416) scores against a 2-D null of 0.352 (z = 4.45).
Its higher raw silhouette comes from clustering in 2 dimensions, not from better-separated groups. Its margin over the null is the same as this model's.

## 5. Primary clusters (k=5)

Values are within-season z-scores. The raw means and league means are in `cluster_profiles_k5.csv`.

| | Shotgun spread, pass-first | Under-center play-action | Shotgun QB-run / RPO | Shanahan wide-zone, two-back motion | Heavy power run + QB run |
|---|--:|--:|--:|--:|--:|
| n | 50 | 46 | 14 | 10 | 8 |
| mean silhouette | 0.197 | 0.201 | 0.300 | 0.289 | 0.157 |
| proe_early | **+0.34** | -0.11 | -0.05 | -0.19 | **-1.13** |
| under_center | -0.27 | **+0.87** | **-1.52** | -0.21 | -0.33 |
| pistol | -0.28 | -0.39 | +0.53 | **+1.36** | **+1.40** |
| pers_11 | +0.51 | -0.09 | +0.42 | **-1.23** | **-1.89** |
| multi_te | -0.22 | +0.25 | -0.13 | -0.90 | **+1.29** |
| two_back | -0.41 | -0.15 | -0.39 | **+2.02** | **+1.55** |
| empty_backfield | +0.06 | -0.24 | +0.12 | +0.70 | -0.08 |
| extra_ol | +0.07 | +0.29 | -0.41 | -0.59 | -0.64 |
| motion | -0.17 | -0.13 | -0.55 | **+1.97** | +0.30 |
| no_huddle | -0.21 | -0.20 | **+1.93** | -0.54 | -0.23 |
| outside_run | -0.41 | +0.07 | -0.39 | **+1.14** | **+1.42** |
| qb_design_run | -0.36 | -0.30 | **+1.70** | -0.53 | **+1.66** |
| rpo | +0.48 | -0.64 | **+1.05** | -0.63 | -0.37 |
| play_action | -0.58 | +0.59 | -0.20 | -0.02 | +0.63 |
| screen | +0.18 | -0.17 | -0.17 | +0.10 | +0.01 |
| qb_out_of_pocket | -0.08 | -0.11 | +0.63 | -0.87 | **+1.12** |
| time_to_throw | -0.55 | +0.62 | -0.31 | -0.38 | +0.85 |
| rb_target_share | -0.01 | -0.07 | -0.52 | **+1.07** | +0.07 |
| te_target_share | -0.26 | +0.23 | -0.26 | -0.48 | **+1.33** |

Members (most to least typical, by silhouette):

- **Shotgun spread, pass-first (50):** SEA24, WAS23, CIN25, IND22, JAX22, JAX23, TB25, NYJ24, NE22, CIN23, DAL24, TB22, BUF24, CIN22, BUF22, DEN25, CIN24, KC23, TEN25, NE23, TB24, PIT22, KC22, NYG23, JAX25, TB23, KC24, GB24, CAR25, IND25, CLE24, CAR24, GB22, PIT23, KC25, NYG24, LAC25, HOU25, CAR23, LA23, PIT25, WAS22, GB25, NYJ22, JAX24, SEA23, DEN24, CHI24, NE24, BUF23
- **Under-center play-action (46):** TEN22, MIN23, CHI25, CLE22, DET24, DET23, MIN24, HOU23, LV25, SEA25, DAL22, LA25, PIT24, HOU24, MIN22, LV22, HOU22, NE25, CLE25, GB23, NYJ23, TEN23, DET22, ARI25, LA24, LV24, NO22, BUF25, ARI23, TEN24, DAL25, MIN25, CAR22, ARI24, DET25, CLE23, LA22, LAC24, LV23, NYG22, DEN22, LAC22, DAL23, SEA22, DEN23, NO23
- **Shotgun QB-run / RPO (14):** PHI23, WAS25, WAS24, PHI24, PHI22, ARI22, IND23, IND24, PHI25, NYG25, BAL23, NO25, LAC23, NYJ25
- **Shanahan wide-zone, two-back motion (10):** MIA23, MIA25, MIA22, MIA24, SF25, SF22, SF23, ATL25, ATL24, SF24
- **Heavy power run + QB run (8):** ATL22, ATL23, BAL24, BAL22, BAL25, CHI23, CHI22, NO24

## 6. Secondary clusters (k=7)

k=7 mostly nests inside k=5. The spread group splits into "balanced" and "pass-first" (KC, CIN, BUF, high early-down PROE).
The under-center group splits into a multi-TE/heavy-OL play-action side and an 11-personnel + motion + two-back side (LA, MIN, DAL, LAC, HOU).

| cluster | n | silhouette | members |
|---|--:|--:|---|
| Shotgun spread, balanced | 33 | 0.208 | DEN25, GB24, NYG24, PIT23, WAS22, TB25, PIT22, GB25, TEN25, DEN24, CHI24, GB22, TB24, IND22, NYJ24, TB22, CAR25, NE22, CAR23, TB23, NYJ25, NYJ22, JAX24, CAR24, MIN25, LAC22, NYG23, SEA23, DEN23, JAX22, NE23, DET25, NYG22 |
| Under-center, multi-TE play-action | 28 | 0.192 | TEN22, LV25, CLE25, CLE22, HOU22, NO22, ARI25, ARI23, CHI25, PIT24, SEA25, LV24, DET24, TEN23, NO24, GB23, TEN24, NYJ23, DAL22, CLE23, ARI24, HOU24, DET23, DEN22, MIN23, NO23, DET22, CAR22 |
| Under-center 11p + motion (McVay-style) | 20 | 0.187 | DAL25, LA24, MIN22, LAC24, LA25, LAC25, DAL23, LA22, MIN24, HOU23, NE25, HOU25, LA23, JAX25, LV22, BUF22, SF24, LV23, DAL24, BUF25 |
| Pass-first spread (early-down PROE) | 18 | 0.161 | CIN24, KC22, KC24, KC25, CIN23, CIN25, CIN22, IND25, CLE24, JAX23, BUF23, KC23, WAS23, BUF24, PIT25, SEA24, SEA22, NE24 |
| Shotgun QB-run / RPO | 13 | 0.301 | PHI23, WAS24, WAS25, PHI24, PHI22, IND23, ARI22, PHI25, IND24, NYG25, NO25, BAL23, LAC23 |
| Shanahan wide-zone, two-back motion | 9 | 0.262 | MIA25, MIA23, MIA24, SF22, ATL25, SF25, SF23, MIA22, ATL24 |
| Heavy power run + QB run | 7 | 0.154 | ATL22, ATL23, BAL22, BAL24, CHI23, BAL25, CHI22 |

## 7. Validation beyond fit statistics

- **Persistence:** 61.5% of teams stay in the same k=5 cluster the next season. If clusters were assigned at random at the same sizes, that figure would be 30.4%.
- **Always-same teams:** CIN, DET, JAX, KC, LV, MIA, MIN, PHI, SF, and TB are in the same k=5 cluster all four seasons.
- **Moves line up with staff or QB changes:**
  - ATL: power (Arthur Smith) to wide-zone (Zac Robinson, 2024)
  - WAS: spread to QB-run/RPO (Kingsbury and Daniels, 2024)
  - CHI: power (Fields) to spread (2024) to under-center PA (Ben Johnson, 2025)
  - SEA: spread to under-center PA (Kubiak, 2025)
  - IND: QB-run/RPO in the Steichen and Richardson years
  - BAL 2023: QB-run/RPO in Monken's first year, then back to heavy/pistol
  - NE 2025: under-center PA (McDaniels)
  - ARI 2022 to 2023: QB-run/RPO (Kingsbury) to under-center PA (Petzing)
- **Boundary cases:** these sit roughly equidistant between two centroids, so treat their labels as soft (`margin_k5` near 0 in `team_season_identity.csv`): BUF23, NE24, DEN23, NO23, SEA22, CHI22, NYJ25, CHI24, NO24, DEN24.

## 8. Output files (`output/identity/`)

| file | contents |
|---|---|
| `team_season_identity.csv` | One row per team-season: k=5 and k=7 cluster and label, per-point silhouette, runner-up cluster, margin, PC scores, raw features |
| `cluster_profiles_k5.csv`, `cluster_profiles_k7.csv` | Per cluster: n, mean silhouette, z-score and raw means per feature, league mean |
| `cluster_members_k5.csv`, `cluster_members_k7.csv` | Members, ordered most to least typical |
| `k_selection.csv` | k=2..10 for k-means, Ward and GMM: silhouette (PCA and full space), null silhouette, z, CH, DB, bootstrap ARI, min size |
| `feature_reliability.csv` | Split-half season reliability, single-game ICC, year-over-year r |
| `pca_loadings.csv` | Loadings, explained variance, eigenvalues vs parallel-analysis threshold |
| `team_trajectories.csv`, `cluster_transitions.csv` | Team x season labels; season-to-season transition counts |

## 9. Limitations

- **Four seasons only.** FTN charting (motion, PA, RPO, screens, QB out of pocket) starts in 2022. Pre-2022 would need a reduced feature set.
- **No blocking-scheme tag.** nflverse doesn't carry one (zone vs gap). `outside_run`, `two_back` and `pistol` stand in as proxies.
- **Routes are charted only for the targeted receiver**, so the full route concept mix isn't observable.
- **QB traits blend into identity.** `qb_design_run` and `qb_out_of_pocket` partly reflect the QB, not only the play caller. That is arguably part of an offense's identity, but it is not purely scheme.
- **Cluster labels were written by hand** from the profiles. If features change, re-read the profiles before reusing the labels.
