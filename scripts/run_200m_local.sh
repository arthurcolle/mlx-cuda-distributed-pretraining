#!/bin/bash
set -e

# Check if tokenizer exists
if [ ! -d "tokenizer" ]; then
  echo "Error: No tokenizer found. Please run train-tokenizer.py first."
  exit 1
fi

# Check if training data exists
if [ ! -f "train.jsonl" ]; then
  echo "Error: train.jsonl not found. Creating a small test dataset..."
  # Create a minimal dataset for testing
  echo '{"text": "This is a test document for training the 200M model."}' > train.jsonl
  echo '{"text": "This is a validation document for testing the 200M model."}' > val.jsonl
  echo "Created minimal test dataset."
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with the 200M model config
echo "Starting training of 200M Muon model..."
python train.py --config model-config-200m.yaml 2>&1 | tee logs/train_200m_$(date +%Y%m%d_%H%M%S).log