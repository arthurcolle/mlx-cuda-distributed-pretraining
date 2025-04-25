#!/usr/bin/env python
# Client script to launch MLX distributed training on Modal A100 GPUs

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Launch MLX training on A100 GPUs via Modal")
    parser.add_argument("--config", type=str, required=True, help="Path to model configuration YAML")
    parser.add_argument("--run-id", type=str, default=None, help="Unique ID for this run (generated if not provided)")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to tokenizer directory")
    parser.add_argument("--data-path", type=str, default=None, help="Path to data directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        sys.exit(1)
    
    # Generate a run ID if not provided
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    
    # Prepare data directory
    data_dir = args.data_path or os.getcwd()
    print(f"Using data directory: {data_dir}")
    
    # Check for required files
    required_files = [
        os.path.join(data_dir, "train.jsonl"),
        os.path.join(data_dir, "val.jsonl")
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file '{file_path}' not found!")
            sys.exit(1)
    
    # Load config to get model details
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    model_name = config.get("name", "MLX-Model")
    print(f"Preparing to train model: {model_name}")
    
    # Check model size (approximately)
    hidden_size = config.get("model", {}).get("dimensions", {}).get("hidden_size", 0)
    num_layers = config.get("model", {}).get("dimensions", {}).get("num_layers", 0)
    vocab_size = config.get("data", {}).get("tokenizer", {}).get("normal_vocab_size", 0)
    
    # Rough parameter count calculation
    param_count = (
        12 * hidden_size * hidden_size * num_layers +  # Transformer blocks
        2 * hidden_size * vocab_size                  # Embedding & output
    ) / 1_000_000  # Convert to millions
    
    print(f"Estimated model size: ~{param_count:.2f}M parameters")
    print(f"Using {num_layers} layers with {hidden_size} hidden dimension")
    
    # Create temp directory for uploads
    temp_dir = Path("./temp_modal_upload")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy config file
    config_path = os.path.join(temp_dir, os.path.basename(args.config))
    shutil.copy(args.config, config_path)
    
    # Set up Modal
    print("Initializing Modal...")
    try:
        # Import the train_model_a100 function from train_a100.py
        from train_a100 import train_model_a100, app
        
        # Deploy and run on Modal
        print(f"Deploying to Modal with run ID: {run_id}")
        with app.run():
            result = train_model_a100.remote(
                config_path=os.path.basename(args.config),
                data_dir=data_dir,
                run_id=run_id,
                checkpoint=args.checkpoint
            )
            print(f"Training initiated successfully: {result}")
            
    except Exception as e:
        print(f"Error launching Modal training: {e}")
        sys.exit(1)
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Training job submitted to Modal with run ID: {run_id}")
    print(f"Logs will be available in: runs/{model_name}-{run_id}/")
    print("You can monitor the job status in the Modal dashboard: https://modal.com/apps")

if __name__ == "__main__":
    main()