#!/bin/bash
# Script to train a 256M parameter model with distributed A100 training via Modal

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Ensure we have the latest code
echo "Preparing data and environment..."

# Prepare the tokenizer if it doesn't exist
if [ ! -d "tokenizer" ]; then
  echo "Tokenizer not found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Run the A100 training through Modal
echo "Launching distributed training on 2x A100 GPUs via Modal..."
python modal_client.py \
  --config model-config-256m.yaml \
  --run-id $RUN_ID \
  --tokenizer-path "$(pwd)/tokenizer" \
  --data-path "$(pwd)"

echo "Training launched with Run ID: $RUN_ID"
echo "You can monitor the logs in runs/Llama-256M-Distributed-$RUN_ID/"