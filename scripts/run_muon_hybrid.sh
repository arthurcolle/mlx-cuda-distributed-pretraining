#!/bin/bash
# Script to train a 3B parameter model with hybrid MLX+CUDA training
# Uses local MLX for gradient aggregation and A100 GPUs via Modal for computation

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Ensure we have the latest code
echo "Preparing environment for hybrid Muon 3B training..."

# Prepare the tokenizer if it doesn't exist
if [ ! -d "tokenizer" ]; then
  echo "Tokenizer not found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Prepare data (requires data files to exist)
echo "Checking for required data files..."
if [ ! -f "train.jsonl" ] || [ ! -f "val.jsonl" ]; then
  echo "Warning: train.jsonl or val.jsonl not found. You may need to prepare data first."
  echo "Consider running: python prepare_data_a100.py"
fi

# Start local MLX computation
echo "Starting MLX computation on local device..."
MLX_PID_FILE="muon_pid.txt"
python train.py --config model-config-muon.yaml --run-id $RUN_ID > "runs/Muon-3B-Local-$RUN_ID.log" 2>&1 &
echo $! > $MLX_PID_FILE
echo "Local MLX training started with PID: $(cat $MLX_PID_FILE)"

# Run the A100 training through Modal
echo "Launching distributed CUDA computation on 2x A100 GPUs via Modal..."
python modal_client.py \
  --config model-config-muon.yaml \
  --run-id $RUN_ID \
  --tokenizer-path "$(pwd)/tokenizer" \
  --data-path "$(pwd)"

echo "Hybrid training launched with Run ID: $RUN_ID"
echo "You can monitor the local logs in runs/Muon-3B-Local-$RUN_ID.log"
echo "And Modal logs in runs/Muon-3B-$RUN_ID/"
echo "Compare with previous runs using: python plot-logs.py \"Muon-3B-*\" \"Llama-256M-Distributed-*\""