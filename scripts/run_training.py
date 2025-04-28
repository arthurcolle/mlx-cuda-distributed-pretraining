#!/usr/bin/env python3
"""
Unified training script for MLX Pretrain models.
This script provides a common interface for all training configurations.
"""

import argparse
import os
import sys
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training import train

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an MLX model")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="./runs/",
                        help="Directory to save outputs")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line arguments
    config["output_dir"] = args.output_dir
    config["seed"] = args.seed
    
    if args.distributed:
        config["distributed"]["enabled"] = True
    
    if args.resume:
        config["resume_from"] = args.resume
    
    # Run training
    train(config)

if __name__ == "__main__":
    main()