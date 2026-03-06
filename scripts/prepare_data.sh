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
SYMBOLS=${SYMBOLS:-"AAPL MSFT"}
PERIOD=${PERIOD:-"7d"}
INTERVAL=${INTERVAL:-"1m"}
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
