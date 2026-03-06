#!/usr/bin/env bash
# Usage:
#   ./vary_labeller.sh [source] [--symbols ...] [--period ...] [--interval ...] [--val-frac ...] [--test-frac ...]
# Examples:
#   ./vary_labeller.sh                                              # uses all defaults
#   ./vary_labeller.sh yahoo --symbols AAPL MSFT --period 7d --interval 1m
#   ./vary_labeller.sh hf

set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE=${1:-yahoo}
SYMBOLS=${SYMBOLS:-"AAPL MSFT"}
PERIOD=${PERIOD:-"7d"}
INTERVAL=${INTERVAL:-"1m"}
VAL_FRAC=${VAL_FRAC:-"0.15"}
TEST_FRAC=${TEST_FRAC:-"0.15"}

if [ "$SOURCE" = "hf" ]; then
    uv run --env-file .env.run \
        ./src/kvant/ml_prepare_data/plot_labelling/vary_labeller_runs.py \
        hf
else
    uv run --env-file .env.run \
        ./src/kvant/ml_prepare_data/plot_labelling/vary_labeller_runs.py \
        yahoo \
        --symbols $SYMBOLS \
        --period "$PERIOD" \
        --interval "$INTERVAL" \
        --val-frac "$VAL_FRAC" \
        --test-frac "$TEST_FRAC"
fi
