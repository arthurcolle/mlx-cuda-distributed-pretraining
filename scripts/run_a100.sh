#!/bin/bash
# A100 Training Launch Script
# This script automates the process of preparing and running training on A100 GPUs

set -e  # Exit on any error

# Default values
CONFIG_FILE="model-config-a100-distributed.yaml"
DATA_DIR="./data"
CHECKPOINT=""
PROMPT="Once upon a time"
CREATE_VAL_SPLIT="false"
VALIDATE_ONLY="false"
TRAIN_FILE=""
VAL_FILE=""
TOKENIZER_DIR="./tokenizer"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --train-file)
      TRAIN_FILE="$2"
      shift 2
      ;;
    --val-file)
      VAL_FILE="$2"
      shift 2
      ;;
    --tokenizer-dir)
      TOKENIZER_DIR="$2"
      shift 2
      ;;
    --create-val-split)
      CREATE_VAL_SPLIT="true"
      shift
      ;;
    --validate-only)
      VALIDATE_ONLY="true"
      shift
      ;;
    --help)
      echo "MLX Pretrain A100 Training Script"
      echo ""
      echo "Usage: ./run_a100.sh [options]"
      echo ""
      echo "Options:"
      echo "  --config FILE          Configuration file (default: model-config-a100-distributed.yaml)"
      echo "  --data-dir DIR         Directory containing prepared data (default: ./data)"
      echo "  --checkpoint FILE      Optional checkpoint to resume from"
      echo "  --prompt TEXT          Text prompt for sample generation after training"
      echo "  --train-file FILE      Raw training data file in JSONL format"
      echo "  --val-file FILE        Raw validation data file in JSONL format"
      echo "  --tokenizer-dir DIR    Directory containing tokenizer.json (default: ./tokenizer)"
      echo "  --create-val-split     Create validation split from training data"
      echo "  --validate-only        Validate data files without running training"
      echo "  --help                 Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './run_a100.sh --help' for usage information"
      exit 1
      ;;
  esac
done

# Check for NVIDIA A100 GPUs
if command -v nvidia-smi &> /dev/null; then
  A100_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "A100" || echo "0")
  if [ "$A100_COUNT" -lt "2" ]; then
    echo "Warning: Expected 2 NVIDIA A100 GPUs, but found $A100_COUNT"
    if [ "$A100_COUNT" -eq "0" ]; then
      echo "No A100 GPUs detected. This script is optimized for A100 GPUs."
      echo "Will attempt to run on Modal cloud with: python train_a100.py"
    fi
  fi
else
  echo "nvidia-smi not found. Assuming you want to run on Modal cloud."
fi

# Step 1: Prepare data if train file is provided
if [ -n "$TRAIN_FILE" ]; then
  echo "=== Preparing data ==="
  
  PREPARE_CMD="python prepare_data_a100.py --train $TRAIN_FILE --output-dir $DATA_DIR"
  
  if [ -n "$VAL_FILE" ]; then
    PREPARE_CMD+=" --val $VAL_FILE"
  elif [ "$CREATE_VAL_SPLIT" = "true" ]; then
    PREPARE_CMD+=" --create-val-split"
  fi
  
  if [ -n "$TOKENIZER_DIR" ]; then
    PREPARE_CMD+=" --tokenizer $TOKENIZER_DIR"
  fi
  
  if [ "$VALIDATE_ONLY" = "true" ]; then
    PREPARE_CMD+=" --validate-only"
  fi
  
  echo "Running: $PREPARE_CMD"
  $PREPARE_CMD
  
  # Exit if validation only
  if [ "$VALIDATE_ONLY" = "true" ]; then
    echo "Validation complete. Exiting as requested."
    exit 0
  fi
fi

# Step 2: Run training
echo "=== Starting training on A100 GPUs ==="

TRAIN_CMD="python train_a100.py --config $CONFIG_FILE --data-dir $DATA_DIR"

if [ -n "$CHECKPOINT" ]; then
  TRAIN_CMD+=" --checkpoint $CHECKPOINT"
fi

if [ -n "$PROMPT" ]; then
  TRAIN_CMD+=" --prompt \"$PROMPT\""
fi

echo "Running: $TRAIN_CMD"
eval $TRAIN_CMD

echo "=== Training complete ==="