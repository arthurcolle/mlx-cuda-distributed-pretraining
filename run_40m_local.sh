#!/bin/bash
set -e

# Check if tokenizer exists
if [ ! -d "tokenizer" ]; then
  echo "No tokenizer found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Check if training data exists
if [ ! -f "train.jsonl" ]; then
  echo "train.jsonl not found. Downloading a better training dataset..."
  
  # Create a temporary smaller dataset first
  echo '{"text": "This is a temporary document for training the model."}' > train.jsonl
  echo '{"text": "This is a temporary validation document for testing the model."}' > val.jsonl
  
  # Download a more substantial dataset in the background
  python download_and_process_llm_data.py openwebtext the_pile:fineweb --total-tokens 10000000 --output-dir llm_data --final-output combined.bin &
  DOWNLOAD_PID=$!
  echo "Started downloading better training data in background (PID: $DOWNLOAD_PID)"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Run training with the 40M model config
echo "Starting training of 40M Muon model..."
python train.py --config model-config-40m.yaml 2>&1 | tee logs/train_40m_$RUN_ID.log

echo "Training complete. Log saved to logs/train_40m_$RUN_ID.log"