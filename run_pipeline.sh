#!/bin/bash
# run_pipeline.sh — Full pipeline: prepare -> train -> predict -> plot
# Usage: bash run_pipeline.sh [config]

set -e

CONFIG="${1:-pipeline.toml}"

echo "Config: $CONFIG"

echo ""
echo "--- [1/4] Prepare ---"
uv run --env-file .env.run scripts/run_prepare.py --config "$CONFIG"

echo ""
echo "--- [2/4] Train ---"
uv run --env-file .env.run scripts/run_train.py --config "$CONFIG"

echo ""
echo "--- [3/4] Predict ---"
uv run --env-file .env.run scripts/run_predict.py --config "$CONFIG"

echo ""
echo "--- [4/4] Plot ---"
uv run --env-file .env.run scripts/run_plot.py --config "$CONFIG"

echo ""
echo "Pipeline complete."
