#!/usr/bin/env bash
# Download the nflverse release files used by the identity model into data/raw/.
set -euo pipefail
mkdir -p data/raw
BASE=https://github.com/nflverse/nflverse-data/releases/download
for y in 2022 2023 2024 2025; do
  curl -sSL -o data/raw/pbp_$y.parquet  $BASE/pbp/play_by_play_$y.parquet &
  curl -sSL -o data/raw/ftn_$y.parquet  $BASE/ftn_charting/ftn_charting_$y.parquet &
  curl -sSL -o data/raw/part_$y.parquet $BASE/pbp_participation/pbp_participation_$y.parquet &
done
curl -sSL -o data/raw/players.parquet $BASE/players/players.parquet &
wait
