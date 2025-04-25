#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import time
from pathlib import Path

def extract_metrics_from_log(log_file_path):
    """Extract metrics from a log file."""
    steps = []
    losses = []
    val_steps = []
    val_losses = []
    
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
    
    return steps, losses, val_steps, val_losses

def plot_training_metrics(log_file_path, output_path=None, interval=60):
    """Plot training metrics and save to file or display."""
    plt.figure(figsize=(14, 8))
    
    while True:
        # Extract metrics
        steps, losses, val_steps, val_losses = extract_metrics_from_log(log_file_path)
        
        if not steps:
            print(f"No training data found in log file: {log_file_path}")
            time.sleep(interval)
            continue
        
        # Apply exponential moving average to smooth the loss curve
        ema = 0.9
        smoothed_losses = []
        if losses:
            smoothed_losses = [losses[0]]
            for loss in losses[1:]:
                smoothed_losses.append(ema * smoothed_losses[-1] + (1 - ema) * loss)
        
        # Clear previous plot
        plt.clf()
        
        # Plot training loss
        plt.plot(steps, smoothed_losses, label="Training Loss (EMA)", color='blue')
        plt.plot(steps, losses, alpha=0.3, color='lightblue')
        
        # Plot validation loss
        if val_steps and val_losses:
            plt.plot(val_steps, val_losses, 'o-', label="Validation Loss", color='red')
        
        # Set labels and title
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Training Progress - Last step: {steps[-1]}")
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend()
        
        # Save or display
        if output_path:
            plt.savefig(output_path)
            print(f"Updated plot saved to {output_path}")
        else:
            plt.savefig("training_plot.png")
            print(f"Updated plot saved to training_plot.png")
        
        # Sleep for specified interval
        time.sleep(interval)

def find_latest_log_file(name="Muon-200M"):
    """Find the latest log file for the given model."""
    # Check in runs directory
    run_log_path = Path(f"runs/{name}/log.txt")
    if run_log_path.exists():
        return str(run_log_path)
    
    # Check logs directory
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = [f for f in logs_dir.glob(f"train_{name.lower()}*.log")]
        
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
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--log", type=str, help="Path to log file", default=None)
    parser.add_argument("--model", type=str, help="Model name to find log for", default="Muon-200M")
    parser.add_argument("--output", type=str, help="Output image path", default=None)
    parser.add_argument("--interval", type=int, help="Update interval in seconds", default=60)
    parser.add_argument("--no-watch", action="store_true", help="Generate plot once and exit")
    args = parser.parse_args()
    
    # Get log file path
    log_file_path = args.log
    if not log_file_path:
        log_file_path = find_latest_log_file(args.model)
    
    if not log_file_path or not Path(log_file_path).exists():
        print(f"Error: Log file not found. Please specify a valid log file path.")
        return
    
    print(f"Using log file: {log_file_path}")
    
    if args.no_watch:
        # Extract metrics once
        steps, losses, val_steps, val_losses = extract_metrics_from_log(log_file_path)
        
        if not steps:
            print(f"No training data found in log file: {log_file_path}")
            return
        
        # Apply exponential moving average to smooth the loss curve
        ema = 0.9
        smoothed_losses = []
        if losses:
            smoothed_losses = [losses[0]]
            for loss in losses[1:]:
                smoothed_losses.append(ema * smoothed_losses[-1] + (1 - ema) * loss)
        
        plt.figure(figsize=(14, 8))
        
        # Plot training loss
        plt.plot(steps, smoothed_losses, label="Training Loss (EMA)", color='blue')
        plt.plot(steps, losses, alpha=0.3, color='lightblue')
        
        # Plot validation loss
        if val_steps and val_losses:
            plt.plot(val_steps, val_losses, 'o-', label="Validation Loss", color='red')
        
        # Set labels and title
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Training Progress - Last step: {steps[-1]}")
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend()
        
        # Save or display
        if args.output:
            plt.savefig(args.output)
            print(f"Plot saved to {args.output}")
        else:
            plt.savefig("training_plot.png")
            print(f"Plot saved to training_plot.png")
    else:
        # Watch mode
        print(f"Watching log file with update interval {args.interval} seconds. Press Ctrl+C to stop.")
        try:
            plot_training_metrics(log_file_path, args.output, args.interval)
        except KeyboardInterrupt:
            print("Monitoring stopped by user.")

if __name__ == "__main__":
    main()