# Legacy: v1 scheme clustering (frozen)

The original pipeline, kept for reference only. It is not maintained, and the scripts expect the pre-reorganization paths
(`src/...`, `data/...`, `output/...`). To rerun it, check out commit `8d288ae` and install `nfl_data_py`.

What it did: 11 features (gap mix, formation, personnel, PA, motion, RPO, tempo) on 2022-2025 neutral-script team-seasons. It ran k-means with k=6 on the **top 2 principal components** and reported a silhouette of 0.416.

Why it was replaced (details in `output/identity/FINDINGS.md`, section 11):

- **2 PCs inflate silhouette.** Clustering in 2 dimensions raises silhouette mechanically: random data with the same shape scores 0.352 there. The margin over random was no larger than the later models' margin.
- **Features were selected to maximize silhouette.** The leave-one-out sweep that dropped `adot` optimized silhouette on the same data it reports, which tunes the model to noise.
- **No reliability screen or within-season normalization.** League-wide drift (motion went from 42% to 60% of plays in 2022-2025) leaks into the clusters.
