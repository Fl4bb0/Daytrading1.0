#!/bin/bash
# run_pipeline.sh — Full pipeline: prepare -> train -> [train_meta] -> predict -> plot
# Usage: bash run_pipeline.sh [config]

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -e

CONFIG="${1:-pipeline.toml}"
META_ENABLED="$(
python3 -c 'import pathlib, sys, tomllib; cfg = tomllib.loads(pathlib.Path(sys.argv[1]).read_text()); print("true" if bool(cfg.get("meta", {}).get("enabled", False)) else "false")' "$CONFIG"
)"
WALK_FORWARD_ENABLED="$(
python3 -c 'import pathlib, sys, tomllib; cfg = tomllib.loads(pathlib.Path(sys.argv[1]).read_text()); print("true" if bool(cfg.get("walk_forward", {}).get("enabled", False)) else "false")' "$CONFIG"
)"
if [ "$WALK_FORWARD_ENABLED" = "true" ]; then
echo "Config: $CONFIG"
echo ""
echo "--- [1/1] Walk Forward ---"
uv run --env-file .env.run scripts/run_walk_forward.py --config "$CONFIG"
echo ""
echo "Pipeline complete."
exit 0
fi
TOTAL_STEPS=4
if [ "$META_ENABLED" = "true" ]; then
  TOTAL_STEPS=5
fi
STEP=1

echo "Config: $CONFIG"

echo ""
echo "--- [$STEP/$TOTAL_STEPS] Prepare ---"
uv run --env-file .env.run scripts/run_prepare.py --config "$CONFIG"
STEP=$((STEP + 1))

echo ""
echo "--- [$STEP/$TOTAL_STEPS] Train ---"
uv run --env-file .env.run scripts/run_train.py --config "$CONFIG"
STEP=$((STEP + 1))

if [ "$META_ENABLED" = "true" ]; then
echo ""
echo "--- [$STEP/$TOTAL_STEPS] Train Meta ---"
uv run --env-file .env.run scripts/run_train_meta.py --config "$CONFIG"
STEP=$((STEP + 1))
fi

echo ""
echo "--- [$STEP/$TOTAL_STEPS] Predict ---"
uv run --env-file .env.run scripts/run_predict.py --config "$CONFIG"
STEP=$((STEP + 1))

echo ""
echo "--- [$STEP/$TOTAL_STEPS] Plot ---"
uv run --env-file .env.run scripts/run_plot.py --config "$CONFIG"

echo ""
echo "Pipeline complete."
