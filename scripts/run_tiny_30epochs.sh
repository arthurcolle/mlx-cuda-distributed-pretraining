#!/bin/bash
set -e

# Check if tokenizer exists
if [ ! -d "tokenizer" ]; then
  echo "No tokenizer found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints/micro-tiny

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Create the directory for the tiny model
mkdir -p checkpoints/micro-tiny

# Run training with the tiny model config
echo "Starting training of tiny model for 30 epochs..."
python train.py --config model-config-tiny.yaml --run-id "Micro-Tiny-30Epochs-${RUN_ID}" 2>&1 | tee logs/train_tiny_30epochs_$RUN_ID.log

echo "Training complete. Log saved to logs/train_tiny_30epochs_$RUN_ID.log"