#!/bin/bash
# Script to train a 400M parameter model with hybrid MLX+CUDA training
# Uses local MLX with 15GB memory limit for gradient aggregation 
# and A100 GPUs via Modal for computation

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Ensure we have the latest code
echo "Preparing environment for hybrid Muon 400M training..."

# Make sure logs directory exists to prevent Modal build errors
mkdir -p logs

# Prepare the tokenizer if it doesn't exist
if [ ! -d "tokenizer" ]; then
  echo "Tokenizer not found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Prepare data (requires data files to exist)
echo "Checking for required data files..."
if [ ! -f "train.jsonl" ] || [ ! -f "val.jsonl" ]; then
  echo "Warning: train.jsonl or val.jsonl not found. You may need to prepare data first."
  echo "Consider running: python download_and_process_llm_data.py"
fi

# Set memory limit environment variable for MLX (32GB = 32000MB)
export MLX_MEMORY_LIMIT_MB=32000

# Start local MLX computation with memory constraint
echo "Starting MLX computation on local device (32GB limit)..."
PID_FILE="hybrid_pid.txt"
# Modify the model config to include the run ID in the name
python -c "
import yaml
with open('model-config-400m-muon.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['name'] = f\"Muon-400M-{\"$RUN_ID\"}\"
with open('model-config-400m-muon-run.yaml', 'w') as f:
    yaml.dump(config, f)
"
# Run with the modified config
python train.py --config model-config-400m-muon-run.yaml > "runs/Muon-400M-Local-$RUN_ID.log" 2>&1 &
echo $! > $PID_FILE
echo "Local MLX training started with PID: $(cat $PID_FILE)"

# Set up monitor script
echo "Setting up training monitor..."
MONITOR_PID_FILE="monitor_pid.txt"
python monitor_training.py --log-file "runs/Muon-400M-Local-$RUN_ID.log" --model-name "Muon-400M" > "runs/Monitor-$RUN_ID.log" 2>&1 &
echo $! > $MONITOR_PID_FILE
echo "Monitor started with PID: $(cat $MONITOR_PID_FILE)"

# Run the A100 training through Modal with our improved approach
echo "Launching distributed CUDA computation on 3x A100-80GB GPUs via Modal..."

# Create necessary directories to avoid issues with Modal
mkdir -p "logs"
mkdir -p "runs/Muon-40M"
mkdir -p "runs/Muon-400M-$RUN_ID"
mkdir -p "runs/Muon-400M-Local-$RUN_ID"

# Using improved container building approach to fix mount issues
python modal_client.py \
  --config model-config-400m-muon.yaml \
  --workers remote_worker_config.json \
  --run-id $RUN_ID \
  --tokenizer-path "$(pwd)/tokenizer" \
  --data-path "$(pwd)"

echo "Hybrid training launched with Run ID: $RUN_ID"
echo "You can monitor the local logs in runs/Muon-400M-Local-$RUN_ID.log"
echo "And Modal logs in runs/Muon-400M-$RUN_ID/"
echo "Monitor logs in runs/Monitor-$RUN_ID.log"