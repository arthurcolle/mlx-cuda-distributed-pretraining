#!/usr/bin/env python
import re
import time
import os
import argparse
import datetime

def parse_progress(log_file_path):
    """Parse the progress from a training log file."""
    if not os.path.exists(log_file_path):
        return None, 0, None
    
    steps = []
    losses = []
    latest_line = ""
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) > 4:  # Basic check to ensure log has content
            for line in lines:
                if line.startswith("Step"):
                    step_match = re.search(r"Step (\d+)", line)
                    loss_match = re.search(r"loss=([0-9.e-]+)", line)
                    
                    if step_match and loss_match:
                        step = int(step_match.group(1))
                        loss = float(loss_match.group(1))
                        steps.append(step)
                        losses.append(loss)
                        latest_line = line.strip()
    
    if steps:
        return steps[-1], len(steps), latest_line
    return None, 0, None

def monitor_progress(log_file_path, target_steps=5000, interval=5):
    """Monitor the training progress periodically."""
    start_time = time.time()
    last_step = -1
    last_check_time = start_time
    
    print(f"Monitoring training progress from: {log_file_path}")
    print(f"Target steps: {target_steps}")
    print(f"Checking every {interval} seconds...")
    print("-" * 50)
    
    try:
        while True:
            current_step, num_logged_steps, latest_line = parse_progress(log_file_path)
            
            if current_step is not None and current_step != last_step:
                # Calculate progress
                progress_pct = (current_step / target_steps) * 100
                elapsed_time = time.time() - start_time
                
                # Calculate rate and ETA
                if num_logged_steps > 1 and elapsed_time > 0:
                    # Calculate steps per second
                    steps_per_sec = num_logged_steps / elapsed_time
                    steps_remaining = target_steps - current_step
                    eta_seconds = steps_remaining / steps_per_sec if steps_per_sec > 0 else 0
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    steps_per_sec = 0
                    eta = "Unknown"
                
                # Print status update
                print(f"Step: {current_step}/{target_steps} ({progress_pct:.2f}%)")
                print(f"Steps/sec: {steps_per_sec:.2f}, ETA: {eta}")
                if latest_line:
                    print(f"Latest: {latest_line}")
                print("-" * 50)
                
                last_step = current_step
            
            # If we've reached the target steps, exit
            if current_step is not None and current_step >= target_steps:
                print(f"Training complete! Reached {current_step} steps.")
                break
                
            # Sleep for the specified interval
            time.sleep(interval)
                
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print(f"\nMonitoring stopped. Total elapsed time: {str(datetime.timedelta(seconds=int(elapsed_time)))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor MLX training progress")
    parser.add_argument("--log", type=str, help="Path to log file", 
                        default="/Users/agent/mlx-pretrain/logs/train_200m_20250425_171220.log")
    parser.add_argument("--steps", type=int, help="Target number of steps", default=5000)
    parser.add_argument("--interval", type=int, help="Checking interval in seconds", default=5)
    args = parser.parse_args()
    
    monitor_progress(args.log, args.steps, args.interval)