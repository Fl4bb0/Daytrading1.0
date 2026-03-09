#!/usr/bin/env bash
# Usage:
#   ./prepare_data.sh [source] [--symbols ...] [--period ...] [--interval ...] [--val-frac ...] [--test-frac ...]
# Examples:
#   ./prepare_data.sh                                              # uses all defaults
#   ./prepare_data.sh yahoo --symbols AAPL MSFT --period 7d --interval 1m
#   ./prepare_data.sh hf

set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE=${1:-yahoo}
SYMBOLS=${SYMBOLS:-"AAPL MSFT NVDA AMZN TSLA GOOGL META"}
PERIOD=${PERIOD:-"60d"} # Yahoo finance only allows 7 days of 1m days, so we choose 60 days of 5m data
INTERVAL=${INTERVAL:-"5m"}
VAL_FRAC=${VAL_FRAC:-"0.15"}
TEST_FRAC=${TEST_FRAC:-"0.15"}

if [ "$SOURCE" = "hf" ]; then
    uv run --env-file .env.run \
        ./src/kvant/ml_prepare_data/prepare_experiment.py \
        hf
else
    uv run --env-file .env.run \
        ./src/kvant/ml_prepare_data/prepare_experiment.py \
        yahoo \
        --symbols $SYMBOLS \
        --period "$PERIOD" \
        --interval "$INTERVAL" \
        --val-frac "$VAL_FRAC" \
        --test-frac "$TEST_FRAC"
fi
