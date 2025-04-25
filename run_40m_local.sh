#!/bin/bash
set -e

# Check if tokenizer exists
if [ ! -d "tokenizer" ]; then
  echo "No tokenizer found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Check if training data exists
if [ ! -f "train.jsonl" ]; then
  echo "train.jsonl not found. Creating a small test dataset..."
  # Create a minimal dataset for testing
  echo '{"text": "This is a test document for training the 40M model."}' > train.jsonl
  echo '{"text": "This is a validation document for testing the 40M model."}' > val.jsonl
  echo "Created minimal test dataset."
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Run training with the 40M model config
echo "Starting training of 40M Muon model..."
python train.py --config model-config-40m.yaml 2>&1 | tee logs/train_40m_$RUN_ID.log

echo "Training complete. Log saved to logs/train_40m_$RUN_ID.log"