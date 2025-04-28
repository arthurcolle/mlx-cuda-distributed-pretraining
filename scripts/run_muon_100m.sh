#!/bin/bash
# Script to train a 100M parameter Muon language model

# Ensure we have the latest code
echo "Setting up environment for Muon 100M model training..."

# Make sure logs and checkpoints directories exist
mkdir -p logs
mkdir -p checkpoints/muon-100m
mkdir -p runs/Muon-100M

# Set memory limit environment variable for MLX
export MLX_MEMORY_LIMIT_MB=32768

# Check for required data files
echo "Checking for required data files..."
if [ ! -f "train.jsonl" ] || [ ! -f "val.jsonl" ]; then
  echo "Error: train.jsonl or val.jsonl not found."
  echo "Please download the training data first with:"
  echo "wget https://huggingface.co/datasets/N8Programs/mlx-pretrain-ex/resolve/main/train.jsonl"
  echo "wget https://huggingface.co/datasets/N8Programs/mlx-pretrain-ex/resolve/main/val.jsonl"
  exit 1
fi

# Prepare the tokenizer if it doesn't exist
if [ ! -d "tokenizer" ]; then
  echo "Tokenizer not found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Either run using the standalone script:
echo "Starting 100M Muon model training using standalone script..."
python train_muon_100m.py --train_file train.jsonl --val_file val.jsonl --batch_size 32

# Or alternatively use the main training script with config:
# echo "Starting 100M Muon model training using main training script with config..."
# python train.py --config model-config-100m-muon.yaml

echo "Training complete! Model checkpoints saved in checkpoints/muon-100m/"
echo "To view the training progress, run: python plot-logs.py \"Muon-100M\""
echo "To generate text with the model, run: python generate.py --run \"Muon-100M\" --prompt \"Your prompt here\""