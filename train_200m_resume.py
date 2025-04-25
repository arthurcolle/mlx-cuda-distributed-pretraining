#!/usr/bin/env python
# Fast and simple training script to train the 200M Muon model on MLX

import os
import time
import argparse
from pathlib import Path
from train import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train the 200M Muon model")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to train")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--checkpoint-interval", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--config", type=str, default="model-config-200m.yaml", help="Model configuration file")
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found")
        return
        
    # Create timestamped log file in logs directory
    os.makedirs("logs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/train_200m_{timestamp}.log"
    
    # Create log file redirection
    with open(log_file, 'w') as f:
        f.write(f"Training the 200M Muon model with {args.steps} steps\n")
        f.write(f"Log interval: {args.log_interval}, Checkpoint interval: {args.checkpoint_interval}\n")
        f.write(f"Configuration: {args.config}\n")
        f.write("="*50 + "\n\n")
    
    # Start training
    print(f"Starting training with configuration from {args.config}")
    print(f"Logs will be saved to {log_file}")
    
    # Create trainer
    trainer = Trainer(args.config)
    
    # Train the model
    trainer.train()
    
    print("Training completed.")
    print(f"Logs saved to {log_file}")

if __name__ == "__main__":
    main()