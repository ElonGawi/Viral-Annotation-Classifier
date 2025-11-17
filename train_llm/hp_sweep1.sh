#!/usr/bin/env bash

set -e
set -u

MODEL="pmb" # PubMedBert

# Directory of the script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Project root (parent directory of script directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Hyperparameter grids
LEARNING_RATES=(1e-5 2e-5 3e-5)
EPOCHS=(3 5)

echo "Starting hyperparameter sweep..."
echo "Model: $MODEL"
echo

for LR in "${LEARNING_RATES[@]}"; do
  for EP in "${EPOCHS[@]}"; do
    RUN="PMB_lr${LR}_ep${EP}"
    echo "==========================================="
    echo "Training run: $RUN"
    echo "Learning rate: $LR | Epochs: $EP"
    echo "==========================================="

    python -m train_llm.llm_training \
      --model "$MODEL" \
      --run-name "$RUN" \
      --learning-rate "$LR" \
      --epochs "$EP"

    echo
    echo "Finished run: $RUN"
    echo
  done
done

echo "Sweep complete"
