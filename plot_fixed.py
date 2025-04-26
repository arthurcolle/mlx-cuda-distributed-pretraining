#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
from pathlib import Path
import glob

def extract_loss_from_log(log_file_path):
    """Extract training loss data from log files."""
    losses = []
    steps = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Look for lines that contain loss value
            if "loss=" in line:
                # Extract loss value
                loss_match = re.search(r'loss=(\d+\.\d+e[+-]\d+)', line)
                if loss_match:
                    loss = float(loss_match.group(1))
                    
                    # Extract step if available
                    step_match = re.search(r'Step (\d+)', line)
                    if step_match:
                        step = int(step_match.group(1))
                    else:
                        # If no step, use the current count
                        step = len(steps) + 1
                        
                    steps.append(step)
                    losses.append(loss)
    
    return steps, losses

def main():
    parser = argparse.ArgumentParser(description='Plot training loss curves from experiment log files')
    parser.add_argument('pattern', type=str, help='Pattern to match log files (e.g., "runs/Micro-1M-*")')
    args = parser.parse_args()
    
    # Find log files
    log_files = glob.glob(args.pattern)
    
    if not log_files:
        print(f"No log files found matching pattern: {args.pattern}")
        return
    
    print(f"Found {len(log_files)} log files to analyze")
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    plt.title('Training Loss Comparison')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Process each log file
    for log_file in log_files:
        # Extract optimizer type from filename
        optimizer_match = re.search(r'Micro-1M-(\w+)-', Path(log_file).name)
        if optimizer_match:
            optimizer = optimizer_match.group(1)
        else:
            optimizer = Path(log_file).name
        
        steps, losses = extract_loss_from_log(log_file)
        
        if steps and losses:
            # Apply smoothing (EMA)
            alpha = 0.1  # Smoothing factor
            smoothed_losses = [losses[0]]
            for i in range(1, len(losses)):
                smoothed_losses.append(alpha * losses[i] + (1 - alpha) * smoothed_losses[-1])
            
            # Plot the smoothed data
            plt.plot(steps, smoothed_losses, '-', label=f'{optimizer}')
        else:
            print(f"No loss data found in {log_file}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    print("Plot saved as optimizer_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()