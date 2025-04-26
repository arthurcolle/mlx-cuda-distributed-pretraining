#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import time
import argparse
import os

def extract_metrics_from_log(log_file_path):
    """Extract metrics from a log file."""
    steps = []
    losses = []
    val_steps = []
    val_losses = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Extract regular training metrics
                if line.startswith("Step") and "validation:" not in line:
                    step_match = re.search(r"Step (\d+)", line)
                    loss_match = re.search(r"loss=([0-9.e-]+)", line)
                    
                    if step_match and loss_match:
                        step = int(step_match.group(1))
                        loss = float(loss_match.group(1))
                        steps.append(step)
                        losses.append(loss)
                
                # Extract validation metrics
                elif "validation:" in line:
                    step_match = re.search(r"Step (\d+)", line)
                    val_loss_match = re.search(r"val_loss=([0-9.e-]+)", line)
                    
                    if step_match and val_loss_match:
                        step = int(step_match.group(1))
                        val_loss = float(val_loss_match.group(1))
                        val_steps.append(step)
                        val_losses.append(val_loss)
    except Exception as e:
        print(f"Error reading log file: {e}")
    
    return steps, losses, val_steps, val_losses

def real_time_plotting(log_file_path, polling_interval=5, max_steps=None):
    """Plot training metrics in real-time."""
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Keep track of the last displayed step
    last_step = -1
    
    while True:
        # Extract metrics
        steps, losses, val_steps, val_losses = extract_metrics_from_log(log_file_path)
        
        # Only update the plot if we have new data
        if steps and steps[-1] > last_step:
            last_step = steps[-1]
            
            # Clear previous plot
            ax.clear()
            
            # Apply exponential moving average to smooth the loss curve
            ema = 0.9
            smoothed_losses = []
            if losses:
                smoothed_losses = [losses[0]]
                for loss in losses[1:]:
                    smoothed_losses.append(ema * smoothed_losses[-1] + (1 - ema) * loss)
            
            # Plot training loss
            if steps and smoothed_losses:
                ax.plot(steps, smoothed_losses, label="Training Loss (EMA)", color='blue')
                ax.plot(steps, losses, alpha=0.3, color='lightblue')
            
            # Plot validation loss
            if val_steps and val_losses:
                ax.plot(val_steps, val_losses, 'o-', label="Validation Loss", color='red')
            
            # Set labels and title
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training Progress - Step {last_step}")
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend()
            
            # Update the display
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Check if we've reached the maximum steps
            if max_steps and last_step >= max_steps:
                print(f"Reached maximum steps ({max_steps}), stopping.")
                break
        
        # Check if the training is likely finished
        if os.path.exists(f"{log_file_path}.complete"):
            print("Training complete marker detected, stopping.")
            break
            
        # Sleep for a bit before checking again
        time.sleep(polling_interval)

def find_latest_log_file(name="Muon-200M"):
    """Find the latest log file for the given model."""
    # Check in runs directory
    run_log_path = Path(f"runs/{name}/log.txt")
    if run_log_path.exists():
        return str(run_log_path)
    
    # Check logs directory
    logs_dir = Path("logs")
    prefix = f"train_{name.lower()}"
    
    if logs_dir.exists():
        log_files = [f for f in logs_dir.glob(f"*{prefix}*.log")]
        
        if log_files:
            # Sort by modification time and return the latest
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            return str(latest_log)
    
    # If we can't find a specific log for the model, look for any training log
    if logs_dir.exists():
        log_files = [f for f in logs_dir.glob("train_*.log")]
        
        if log_files:
            # Sort by modification time and return the latest
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            return str(latest_log)
    
    # Finally, check for generic training logs
    all_logs = list(Path("logs").glob("*.log")) if logs_dir.exists() else []
    
    if all_logs:
        latest_log = max(all_logs, key=lambda f: f.stat().st_mtime)
        return str(latest_log)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Real-time plotting of training metrics")
    parser.add_argument("--log", type=str, help="Path to log file (will find latest if not specified)", default=None)
    parser.add_argument("--model", type=str, help="Model name to find log for", default="Muon-200M")
    parser.add_argument("--interval", type=int, help="Polling interval in seconds", default=5)
    parser.add_argument("--max-steps", type=int, help="Maximum steps to plot", default=None)
    args = parser.parse_args()
    
    # Get log file path
    log_file_path = args.log
    if not log_file_path:
        log_file_path = find_latest_log_file(args.model)
    
    if not log_file_path or not Path(log_file_path).exists():
        print(f"Error: Log file not found. Please specify a valid log file path.")
        return
    
    print(f"Monitoring log file: {log_file_path}")
    print(f"Polling interval: {args.interval} seconds")
    print("Press Ctrl+C to stop monitoring.")
    
    try:
        real_time_plotting(log_file_path, polling_interval=args.interval, max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")

if __name__ == "__main__":
    main()