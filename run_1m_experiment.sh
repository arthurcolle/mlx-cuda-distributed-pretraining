#!/bin/bash
# Script to run fast experiments with 1M parameter model
# Tests different optimizers and configurations

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Ensure we have the latest code
echo "Preparing environment for fast 1M model experiments..."

# Make sure logs and checkpoints directories exist
mkdir -p logs
mkdir -p checkpoints/micro-1m
mkdir -p runs

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

# Set memory limit environment variable for MLX
export MLX_MEMORY_LIMIT_MB=32000

# Function to run training with a specific optimizer
run_experiment() {
  local optimizer=$1
  local lr=$2
  local run_name="Micro-1M-${optimizer}-${RUN_ID}"
  
  echo "Starting experiment with optimizer: $optimizer, learning rate: $lr"
  
  # Create a temporary config with the specific optimizer
  cat model-config-1m.yaml | sed "s/optimizer: \"muon\"/optimizer: \"$optimizer\"/" | \
                            sed "s/learning_rate: 3.0e-3/learning_rate: $lr/" > "model-config-1m-${optimizer}.yaml"
  
  # Run the training with more frequent logging (every 4 seconds)
  # Create a temporary config with the run name set correctly
  sed -i'.bak' "s/^name: \"Micro-1M\"/name: \"${run_name}\"/" "model-config-1m-${optimizer}.yaml"
  
  # Run without the --run-id parameter which is causing errors
  python train.py --config "model-config-1m-${optimizer}.yaml" --log-interval 4 > "runs/${run_name}.log" 2>&1
  
  echo "Experiment completed: $run_name"
  echo "Log file: runs/${run_name}.log"
  
  # Clean up temporary config and backup files
  rm "model-config-1m-${optimizer}.yaml" "model-config-1m-${optimizer}.yaml.bak"
}

# Run experiments with different optimizers
echo "Running multiple optimizer experiments..."

# Run Muon experiment
run_experiment "muon" "3.0e-3"

# Run Shampoo experiment
run_experiment "shampoo" "1.0e-3"

# Run AdamW experiment for baseline comparison
run_experiment "adamw" "5.0e-3"

echo "All experiments completed."
echo "Compare results with: python plot-logs.py \"Micro-1M-*\""