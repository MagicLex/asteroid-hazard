#!/usr/bin/env bash
# Log one experiment: a row in the leaderboard FG AND a new version in the model
# registry. EVERY experiment (keep/discard/crash) registers a version so the
# registry charts val_metric across the whole run and the metrics are visible.
# usage: log_exp.sh <keep|discard|crash> "<description>"
set -euo pipefail
cd "$(dirname "$0")/.."
STATUS="$1"; DESC="$2"
LOG=autoresearch/run.log
TAG=astjun29
MODEL="autoresearch_${TAG}"

if grep -q '^val_metric:' "$LOG"; then
  V=$(grep '^val_metric:' "$LOG" | awk '{print $2}')
  M=$(grep '^peak_memory_gb:' "$LOG" | awk '{print $2}')
else
  V=0; M=0; STATUS=crash
fi
SHA=$(git rev-parse --short HEAD)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# 1. leaderboard row (SDK insert — CLI can't write a timestamp column)
python autoresearch/log_row.py "$SHA" "$V" "$M" "$STATUS" "$DESC" "$TS" 2>&1 \
  | grep -iE "inserted|error" || true

# 2. run-progression chart into model/ (so the card shows the curve)
python autoresearch/progression.py 2>&1 | grep -iE "wrote|skipping|error" || true

# 3. register EVERY experiment as the next version of one model name, with the
#    CV metric so the registry charts it, plus the card images already in model/.
hops model register "$MODEL" autoresearch/model \
  --framework sklearn --metrics "val_metric=$V" \
  --description "$DESC; status=$STATUS; metric_direction=max; commit=$SHA" \
  --feature-view neo_pha_fv 2>&1 | grep -iE "registered|version|error" || true

echo "logged $STATUS val_metric=$V ($DESC)"
