#!/bin/bash

# Run TinyStories 40M training script
# This script prepares TinyStories data and trains a 40M parameter model

# Set up environment
set -e  # Exit on first error
set -u  # Exit on undefined variable
cd "$(dirname "$0")/.."  # Change to repo root directory

# Paths
TINYSTORIES_DATA="/Users/agent/smolGPT/data/TinyStories_all_data"
PROCESSED_DATA="processed_dataset"
CONFIG_PATH="configs/model-config-40m-tinystories.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="MLX-40M-TinyStories-${TIMESTAMP}"

echo "=== Starting TinyStories 40M model training ==="
echo "Using TinyStories data from: ${TINYSTORIES_DATA}"
echo "Processing data to: ${PROCESSED_DATA}"

# Check if data needs to be prepared
if [ ! -d "${PROCESSED_DATA}/tokenizer" ]; then
    echo "=== Preparing TinyStories data ==="
    python prepare_tinystories_data.py \
        --data-path "${TINYSTORIES_DATA}" \
        --output-dir "${PROCESSED_DATA}" \
        --vocab-size 8000 \
        --train-split 0.95 \
        --max-length 2048
else
    echo "=== Using existing processed data ==="
fi

# Update run name in config
sed -i '' "s/name: \"MLX-40M-TinyStories\"/name: \"${RUN_NAME}\"/" "${CONFIG_PATH}"

# Create directories if they don't exist
mkdir -p logs
mkdir -p checkpoints/40m-tinystories

echo "=== Starting training ==="
echo "Configuration: ${CONFIG_PATH}"
echo "Run name: ${RUN_NAME}"

# Run the training
python train.py --config "${CONFIG_PATH}"

echo "=== Training complete ==="
echo "Model checkpoints are saved in: checkpoints/40m-tinystories"
echo "Training logs are saved in: runs/${RUN_NAME}"