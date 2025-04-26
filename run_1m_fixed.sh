#!/bin/bash
set -e

# Check if tokenizer exists
if [ ! -d "tokenizer" ]; then
  echo "No tokenizer found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints/micro-1m

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Run training with the 1M model config
echo "Starting training of 1M Micro model with fixed configuration..."
python train.py --config model-config-1m-fixed.yaml --run-id "Micro-1M-Fixed-${RUN_ID}" 2>&1 | tee logs/train_1m_fixed_$RUN_ID.log

echo "Training complete. Log saved to logs/train_1m_fixed_$RUN_ID.log"