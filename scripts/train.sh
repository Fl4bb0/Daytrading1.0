#!/usr/bin/env bash
# Usage:
#   ./train.sh [--exp-dir ...] [--model ...] [--epochs ...] [--lr ...] [--weight-decay ...] [--train-batch-size ...] [--eval-batch-size ...] [--wandb-project ...] [--wandb-name ...]
# Examples:
#   ./train.sh                                                     # uses all defaults
#   ./train.sh --model TSBClassifier --epochs 1000
#   ./train.sh --model Conv1DClassifier --lr 1e-3 --epochs 500

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL=${MODEL:-"ResNLS"}
EPOCHS=${EPOCHS:-5000}
LR=${LR:-"5e-3"}
WEIGHT_DECAY=${WEIGHT_DECAY:-"5e-5"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-512}
WANDB_PROJECT=${WANDB_PROJECT:-"kvant-stocks"}

uv run --env-file .env.run \
    ./src/kvant/ml_framework/scripts/train_experiment.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --wandb-project "$WANDB_PROJECT" \
    "$@"
