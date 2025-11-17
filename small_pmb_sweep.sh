#!/usr/bin/env bash
# run_pmb_sweep.sh
# Small hyperparameter sweep for PubMedBERT on annotation classification

set -e  # stop on first error
set -u  # treat unset vars as errors

MODEL="pmb"
PROJECT_ROOT="$(dirname "$0")"

# You can tweak these grids if desired
LEARNING_RATES=(1e-5 2e-5 3e-5)
EPOCHS=(3 5)

echo "Starting PubMedBERT hyperparameter sweep..."
echo "Model: $MODEL"
echo "Project root: $PROJECT_ROOT"
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
    echo "Finished $RUN"
    echo
  done
done

echo "Sweep complete!"
