#!/usr/bin/env python
"""
Monitor training script that combines log monitoring and real-time plotting.
Automatically finds and plots the latest training run.
"""

import os
import argparse
import time
import threading
import subprocess
import signal
from pathlib import Path
import re
import json
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('monitor_training')

def find_training_process():
    """Find the running training process and its log file."""
    # Try looking for running hybrid_distributed.py processes
    try:
        result = subprocess.run(
            ["ps", "-ef"], 
            capture_output=True, 
            text=True
        )
        
        lines = result.stdout.split('\n')
        training_processes = []
        
        for line in lines:
            if 'hybrid_distributed.py' in line or 'train.py' in line or 'train_a100.py' in line:
                # Skip the grep process itself
                if 'grep' not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[1]
                        training_processes.append((pid, line))
        
        if training_processes:
            # Find the most recently started process
            latest_pid, latest_cmd = max(training_processes, key=lambda x: int(x[0]))
            logger.info(f"Found training process with PID {latest_pid}: {latest_cmd}")
            
            # Extract the config path from the command
            config_path = None
            cmd_parts = latest_cmd.split()
            for i, part in enumerate(cmd_parts):
                if part == '--config' and i + 1 < len(cmd_parts):
                    config_path = cmd_parts[i + 1]
            
            if config_path:
                # Save the PID to a file for other tools to use
                with open('monitor_pid.txt', 'w') as f:
                    f.write(f"{latest_pid}\n{config_path}")
                
                return latest_pid, config_path
    
    except Exception as e:
        logger.error(f"Error finding training process: {e}")
    
    return None, None

def find_latest_log_file(model_name=None):
    """
    Find the latest log file for training, with optional model name filter.
    """
    # List of potential log directories and patterns to check
    search_paths = [
        ("logs", "*.log"),
        ("logs", f"*{model_name}*.log" if model_name else "*.log"),
        ("runs", "*.log"),
        (".", "hybrid_training_*.log"),
        (".", "training_*.log")
    ]
    
    all_log_files = []
    
    for dir_path, pattern in search_paths:
        dir_path = Path(dir_path)
        if dir_path.exists() and dir_path.is_dir():
            log_files = list(dir_path.glob(pattern))
            all_log_files.extend(log_files)
    
    if not all_log_files:
        logger.warning("No log files found")
        return None
    
    # Sort by modification time (newest first)
    latest_log = max(all_log_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest log file: {latest_log}")
    
    return str(latest_log)

def extract_metrics_from_log(log_file_path):
    """Extract training metrics from a log file."""
    steps = []
    losses = []
    val_steps = []
    val_losses = []
    learning_rates = []
    lr_steps = []
    tokens_per_sec = []
    tps_steps = []
    
    regex_patterns = {
        'step': r"Step (\d+)",
        'loss': r"[Ll]oss[=:]\s*([0-9.e+-]+)",
        'val_loss': r"val_loss[=:]\s*([0-9.e+-]+)",
        'lr': r"lr[=:]\s*([0-9.e+-]+)",
        'tokens_per_sec': r"tokens/sec[=:]\s*([0-9.e+-]+)"
    }
    
    if not os.path.exists(log_file_path):
        logger.error(f"Log file not found: {log_file_path}")
        return steps, losses, val_steps, val_losses, lr_steps, learning_rates, tps_steps, tokens_per_sec
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Extract step number if present in this line
                step_match = re.search(regex_patterns['step'], line)
                if not step_match:
                    continue
                
                step = int(step_match.group(1))
                
                # Extract training loss
                loss_match = re.search(regex_patterns['loss'], line)
                if loss_match and "validation" not in line.lower():
                    try:
                        loss = float(loss_match.group(1))
                        steps.append(step)
                        losses.append(loss)
                    except ValueError:
                        # Skip if we can't parse the loss as a float
                        pass
                
                # Extract validation loss
                val_loss_match = re.search(regex_patterns['val_loss'], line)
                if val_loss_match or "validation" in line.lower():
                    try:
                        if val_loss_match:
                            val_loss = float(val_loss_match.group(1))
                        else:
                            # Try to extract generic loss from validation line
                            loss_match = re.search(regex_patterns['loss'], line)
                            if loss_match:
                                val_loss = float(loss_match.group(1))
                            else:
                                continue
                        
                        val_steps.append(step)
                        val_losses.append(val_loss)
                    except ValueError:
                        # Skip if we can't parse the loss as a float
                        pass
                
                # Extract learning rate
                lr_match = re.search(regex_patterns['lr'], line)
                if lr_match:
                    try:
                        lr = float(lr_match.group(1))
                        lr_steps.append(step)
                        learning_rates.append(lr)
                    except ValueError:
                        # Skip if we can't parse the LR as a float
                        pass
                
                # Extract tokens per second
                tps_match = re.search(regex_patterns['tokens_per_sec'], line)
                if tps_match:
                    try:
                        tps = float(tps_match.group(1))
                        tps_steps.append(step)
                        tokens_per_sec.append(tps)
                    except ValueError:
                        # Skip if we can't parse the TPS as a float
                        pass
    
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
    
    return steps, losses, val_steps, val_losses, lr_steps, learning_rates, tps_steps, tokens_per_sec

def plot_real_time(log_file_path, polling_interval=5, max_time_minutes=None, target_steps=None):
    """Plot training metrics in real-time."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        logger.error(f"Cannot plot: {e}")
        return

    # Turn on interactive mode
    plt.ion()
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Keep track of the last displayed step
    last_step = -1
    start_time = time.time()
    
    while True:
        # Check if we've exceeded the maximum monitoring time
        if max_time_minutes and (time.time() - start_time) / 60 > max_time_minutes:
            logger.info(f"Reached maximum monitoring time of {max_time_minutes} minutes, stopping.")
            break
        
        # Extract metrics
        steps, losses, val_steps, val_losses, lr_steps, learning_rates, tps_steps, tokens_per_sec = extract_metrics_from_log(log_file_path)
        
        # Only update the plot if we have new data
        if steps and (last_step == -1 or steps[-1] > last_step):
            last_step = steps[-1]
            
            # Calculate progress and ETA
            progress_str = ""
            if target_steps:
                progress_pct = (last_step / target_steps) * 100
                
                # Calculate rate and ETA
                elapsed_time = time.time() - start_time
                if len(steps) > 1 and elapsed_time > 0:
                    # Use the last 10 steps to calculate rate (or fewer if not available)
                    recent_steps = min(10, len(steps))
                    steps_per_sec = recent_steps / (elapsed_time / len(steps))
                    steps_remaining = target_steps - last_step
                    eta_seconds = steps_remaining / steps_per_sec if steps_per_sec > 0 else 0
                    eta = str(timedelta(seconds=int(eta_seconds)))
                    progress_str = f" ({progress_pct:.1f}%, ETA: {eta})"
            
            # Clear previous plots
            for ax in axs:
                ax.clear()
            
            # Apply exponential moving average to smooth the loss curve
            if losses:
                ema_factor = 0.9  # Smoothing factor (higher = more smoothing)
                smoothed_losses = []
                if losses:
                    smoothed_losses = [losses[0]]
                    for loss in losses[1:]:
                        smoothed_losses.append(ema_factor * smoothed_losses[-1] + (1 - ema_factor) * loss)
                
                # Plot training and validation loss
                axs[0].plot(steps, smoothed_losses, label="Training Loss (EMA)", color='blue')
                axs[0].plot(steps, losses, alpha=0.3, color='lightblue')
                
                if val_steps and val_losses:
                    axs[0].plot(val_steps, val_losses, 'o-', label="Validation Loss", color='red')
                
                axs[0].set_xlabel("Step")
                axs[0].set_ylabel("Loss")
                axs[0].set_title(f"Training Progress - Step {last_step}/{target_steps if target_steps else '?'}{progress_str}")
                axs[0].grid(True, alpha=0.3)
                axs[0].legend()
                
                # Add y-log scale option for loss plot if we have enough data points
                if len(losses) > 10:
                    # Add a second y-axis with log scale
                    ax2 = axs[0].twinx()
                    ax2.set_ylabel("Loss (log scale)")
                    ax2.set_yscale('log')
                    ax2.plot(steps, losses, alpha=0)  # Invisible plot just to set the scale
            
            # Plot learning rate and tokens per second
            ln1 = None
            if lr_steps and learning_rates:
                ln1 = axs[1].plot(lr_steps, learning_rates, label="Learning Rate", color='green')
                axs[1].set_xlabel("Step")
                axs[1].set_ylabel("Learning Rate")
                axs[1].tick_params(axis='y', labelcolor='green')
            
            ln2 = None
            if tps_steps and tokens_per_sec:
                ax3 = axs[1].twinx()
                ln2 = ax3.plot(tps_steps, tokens_per_sec, label="Tokens/sec", color='purple')
                ax3.set_ylabel("Tokens per Second")
                ax3.tick_params(axis='y', labelcolor='purple')
                
                if ln1:
                    # Combine legends from both y-axes
                    lns = ln1 + ln2
                    labs = [l.get_label() for l in lns]
                    axs[1].legend(lns, labs, loc='upper right')
                else:
                    ax3.legend()
            
            # Add grid to second subplot
            axs[1].grid(True, alpha=0.3)
            axs[1].set_title("Training Metrics")
            
            # Adjust layout and update the display
            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Save the current plot
            plt.savefig("training_progress.png")
        
        # Check if we've reached target steps
        if target_steps and steps and steps[-1] >= target_steps:
            logger.info(f"Reached target steps ({target_steps}), stopping.")
            break
            
        # Check if the training is likely finished
        complete_marker = Path(f"{log_file_path}.complete")
        if complete_marker.exists():
            logger.info("Training complete marker detected, stopping monitoring.")
            break
            
        # Sleep for a bit before checking again
        time.sleep(polling_interval)

def log_tail_thread(log_file_path, stop_event):
    """Thread to display log file updates in real-time."""
    if not os.path.exists(log_file_path):
        logger.error(f"Log file does not exist: {log_file_path}")
        return
    
    # Remember the last position
    last_size = os.path.getsize(log_file_path)
    
    logger.info(f"Starting log tail for: {log_file_path}")
    
    while not stop_event.is_set():
        try:
            current_size = os.path.getsize(log_file_path)
            
            if current_size > last_size:
                with open(log_file_path, 'r') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    if new_content:
                        print(new_content, end='', flush=True)
                
                last_size = current_size
                
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in log tail: {e}")
            time.sleep(5)  # Wait longer after an error

def main():
    parser = argparse.ArgumentParser(description="MLX Training Monitor")
    parser.add_argument("--log", type=str, help="Path to log file (will find latest if not specified)", default=None)
    parser.add_argument("--model", type=str, help="Model name to search for in log files", default=None)
    parser.add_argument("--interval", type=int, help="Polling interval in seconds", default=5)
    parser.add_argument("--max-time", type=int, help="Maximum monitoring time in minutes", default=None)
    parser.add_argument("--steps", type=int, help="Target number of steps", default=5000)
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting (only show log)")
    parser.add_argument("--no-tail", action="store_true", help="Disable log tailing (only plot)")
    args = parser.parse_args()
    
    # Find training process and log file if not specified
    pid, config_path = find_training_process()
    
    # Extract model name from config if available
    model_name = args.model
    if not model_name and config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                model_name = config_data.get('name', None)
        except Exception as e:
            logger.warning(f"Error reading config: {e}")
    
    # Find log file
    log_file_path = args.log
    if not log_file_path:
        log_file_path = find_latest_log_file(model_name)
    
    if not log_file_path or not Path(log_file_path).exists():
        logger.error("Log file not found. Please specify a valid log file path with --log.")
        return
    
    logger.info(f"Monitoring log file: {log_file_path}")
    logger.info(f"Polling interval: {args.interval} seconds")
    logger.info(f"Target steps: {args.steps}")
    if args.max_time:
        logger.info(f"Maximum monitoring time: {args.max_time} minutes")
    
    # Create stop event for graceful shutdown
    stop_event = threading.Event()
    
    # Start log tail thread if not disabled
    tail_thread = None
    if not args.no_tail:
        tail_thread = threading.Thread(target=log_tail_thread, args=(log_file_path, stop_event))
        tail_thread.daemon = True
        tail_thread.start()
    
    try:
        # Start real-time plotting if not disabled
        if not args.no_plot:
            plot_real_time(
                log_file_path, 
                polling_interval=args.interval, 
                max_time_minutes=args.max_time,
                target_steps=args.steps
            )
        else:
            # If plotting is disabled, just wait until interrupted
            while not stop_event.is_set():
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
    finally:
        # Signal threads to stop
        stop_event.set()
        
        # Wait for threads to finish
        if tail_thread:
            tail_thread.join(timeout=2)
        
        logger.info("Monitor stopped.")

if __name__ == "__main__":
    try:
        # Try to import plotting libraries, but don't fail if they're not available
        import matplotlib.pyplot as plt
        import numpy as np
        import yaml
    except ImportError as e:
        if "matplotlib" in str(e) or "numpy" in str(e):
            logger.warning(f"Warning: {e}. Plotting will be disabled.")
            plt = None
        elif "yaml" in str(e):
            logger.warning(f"Warning: {e}. Config parsing will be limited.")
            yaml = None
        else:
            raise
    
    main()