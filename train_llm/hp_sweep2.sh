#!/usr/bin/env bash
set -euo pipefail

MODEL="pmb"  # PubMedBERT

# Script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Project root is the parent directory of train_llm/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

### Hyperparameters

# Finer learning rates around 2e-5
LRS=("1.5e-5" "2e-5" "2.5e-5")

# Epochs around 5
EPOCHS=(4 5 6)

# Label smoothing factors
LSMOOTHS=("0.0" "0.1")

# Fixed hyperparameters
WARMUP_RATIO="0.1"
WEIGHT_DECAY="0.01"
TRAIN_BS=16
EVAL_BS=32

# Where models are saved (relative to project root)
MODELS_DIR="models/fine_tuned_BERT_models"

######################
# Run the sweep
######################
for LR in "${LRS[@]}"; do
  for EP in "${EPOCHS[@]}"; do
    for LS in "${LSMOOTHS[@]}"; do
      RUN="PMB_lr${LR}_ep${EP}_ls${LS}"
      RUN_DIR="${MODELS_DIR}/${RUN}"

      # Skip if this run already appears completed
      if [ -f "${RUN_DIR}/config.json" ] || [ -f "${RUN_DIR}/model.safetensors" ]; then
        echo ">>> Skipping run (already exists): $RUN"
        continue
      fi

      echo "============================================"
      echo ">>> Starting run: $RUN"
      echo "    Model          : $MODEL"
      echo "    Learning rate  : $LR"
      echo "    Epochs         : $EP"
      echo "    Label smoothing: $LS"
      echo "    Warmup ratio   : $WARMUP_RATIO"
      echo "    Weight decay   : $WEIGHT_DECAY"
      echo "============================================"

      python -m train_llm.llm_training \
        --model "$MODEL" \
        --run-name "$RUN" \
        --learning-rate "$LR" \
        --epochs "$EP" \
        --label-smoothing "$LS" \
        --warmup-ratio "$WARMUP_RATIO" \
        --weight-decay "$WEIGHT_DECAY" \
        --train-batch-size "$TRAIN_BS" \
        --eval-batch-size "$EVAL_BS"

      echo ">>> Finished run: $RUN"
      echo
    done
  done
done

echo "Sweep complete"