import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import re
from pathlib import Path
import glob

def parse_optimizer_from_filename(filename):
    """Extract optimizer name from log filename."""
    patterns = [
        r'(adamw|muon|shampoo)',  # Matches optimizer names in filenames
        r'Experiment-(\w+)',      # Matches experiment directory names
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return "unknown"

def process_log(log_file: Path) -> tuple[list, list, list, list, list]:
    """Process a single log file and return tokens, training losses, and validation data."""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Parse training losses from regular log entries
    train_steps = []
    
    # Parse validation losses
    val_steps = []
    val_losses = []
    
    for line in lines:
        # Training log format: "Step N: loss=X.XXXe+XX | ppl=XXXXX.XX | tok/s=XX.XXK | toks=XXXX | lr=X.XXXe-XX"
        if line.startswith("Step") and "validation:" not in line and ": loss=" in line:
            parts = line.split("|")
            step_part = parts[0].strip()
            step = int(step_part.split()[1].rstrip(':'))
            
            # Extract loss
            loss_part = next((p for p in parts if "loss=" in p), None)
            if loss_part:
                loss = float(loss_part.split("=")[1].strip())
                
                # Extract tokens
                toks_part = next((p for p in parts if "toks=" in p), None)
                if toks_part:
                    toks = float(toks_part.split("=")[1].strip())
                    train_steps.append((step, loss, toks))
        
        # Validation log format: "Step N validation: val_loss=X.XXXe+XX | val_ppl=XXXXX.XX"
        elif "validation:" in line:
            # Validation log
            step = int(line.split()[1])
            val_loss = float(line.split("val_loss=")[1].split()[0])
            val_steps.append(step)
            val_losses.append(val_loss)
    
    # Sort train_steps and deduplicate
    train_steps.sort(key=lambda x: x[0])
    deduped_train_steps = []
    for step, loss, toks in train_steps:
        if len(deduped_train_steps) == 0 or deduped_train_steps[-1][0] != step:
            deduped_train_steps.append((step, loss, toks))
    
    train_losses = []
    tokens = [0]
    total_tokens = 0
    for step, loss, toks in deduped_train_steps:
        train_losses.append(loss)
        total_tokens += toks
        tokens.append(total_tokens)
    
    # Ensure tokens list has same length as losses
    if len(tokens) > len(train_losses) + 1:
        tokens = tokens[:len(train_losses) + 1]
    tokens = tokens[1:]  # Remove the initial 0
    
    # Validation data might also be in metadata
    metadata_path = log_file.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            if 'validation' in metadata and len(metadata['validation']['steps']) > 0:
                # Use metadata for validation data as it's more reliable
                val_steps = metadata['validation']['steps']
                val_losses = metadata['validation']['losses']
        except:
            # Fallback to log-parsed validation data
            pass
    
    # EMA smoothing for training loss
    ema = 0.9
    smoothed_train_losses = []
    if train_losses:
        smoothed_train_losses = [train_losses[0]]
        for loss in train_losses[1:]:
            smoothed_train_losses.append(ema * smoothed_train_losses[-1] + (1 - ema) * loss)
    
    # EMA smoothing for validation loss
    ema_val = 0.0
    smoothed_val_losses = []
    if val_losses:
        smoothed_val_losses = [val_losses[0]]
        for loss in val_losses[1:]:
            smoothed_val_losses.append(ema_val * smoothed_val_losses[-1] + (1 - ema_val) * loss)
            ema_val = ema ** (1000/16)
    
    return tokens, smoothed_train_losses, val_steps, val_losses, smoothed_val_losses

def main():
    parser = argparse.ArgumentParser(description='Plot training logs for multiple runs')
    parser.add_argument('--optimizers', type=str, nargs='+', default=['adamw', 'muon', 'shampoo'],
                        help='Optimizers to compare (default: adamw, muon, shampoo)')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='Directory containing log files (default: runs)')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Optional pattern to filter log files (e.g., "Micro-1M-*")')
    parser.add_argument('--no-val', action='store_true', help='Ignore validation data when plotting')
    parser.add_argument('--output', type=str, default=None, help='Output file to save the plot (optional)')
    args = parser.parse_args()

    # Create a figure with 1 row, 2 columns
    plt.figure(figsize=(16, 8))
    
    # Define specific colors for each optimizer
    color_map = {
        'adamw': 'blue',
        'muon': 'red',
        'shampoo': 'green'
    }
    
    # Full range training and validation loss plot
    plt.subplot(1, 2, 1)
    has_validation_data = False
    
    # Find log files that match our criteria
    if args.pattern:
        # Search for both direct log files and directories with log.txt
        log_files = []
        
        # Look for directly matching log files with .log extension
        direct_matches = list(Path(args.log_dir).glob(f"{args.pattern}.log"))
        log_files.extend(direct_matches)
        
        # Look for experiment directories that match the pattern
        dir_matches = [p / "log.txt" for p in Path(args.log_dir).glob(f"{args.pattern}*") 
                      if p.is_dir() and (p / "log.txt").exists()]
        log_files.extend(dir_matches)
    else:
        # Search for all relevant log files
        log_files = []
        
        # Look for experiment directories with log.txt
        for optimizer in args.optimizers:
            # Match Experiment-{Optimizer} pattern
            experiment_dirs = list(Path(args.log_dir).glob(f"Experiment-{optimizer}*"))
            log_files.extend([p / "log.txt" for p in experiment_dirs if (p / "log.txt").exists()])
            
            # Match {Optimizer}-* pattern directories
            optimizer_dirs = list(Path(args.log_dir).glob(f"{optimizer}-*"))
            log_files.extend([p / "log.txt" for p in optimizer_dirs if p.is_dir() and (p / "log.txt").exists()])
            
            # Match direct .log files
            direct_logs = list(Path(args.log_dir).glob(f"*{optimizer}*.log"))
            log_files.extend(direct_logs)
    
    # Filter by optimizers
    filtered_log_files = []
    for log_file in log_files:
        optimizer = parse_optimizer_from_filename(str(log_file))
        if optimizer in [opt.lower() for opt in args.optimizers]:
            filtered_log_files.append((optimizer, log_file))
    
    # Group logs by optimizer
    optimizer_logs = {}
    for optimizer, log_file in filtered_log_files:
        if optimizer not in optimizer_logs:
            optimizer_logs[optimizer] = []
        optimizer_logs[optimizer].append(log_file)
    
    # Process each optimizer's logs
    for optimizer, logs in optimizer_logs.items():
        # Use the most recent log file for each optimizer
        if not logs:
            continue
            
        # Sort logs by modification time and pick the most recent
        logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        log_file = logs[0]
        
        print(f"Processing {optimizer} log file: {log_file}")
        tokens, train_losses, val_steps, val_losses, smoothed_val_losses = process_log(log_file)
        
        if not tokens or not train_losses:
            print(f"  No data found in {log_file}")
            continue
        
        # Plot training losses
        optimizer_label = optimizer.upper()  # Make label more readable
        color = color_map.get(optimizer.lower(), 'gray')
        plt.plot(tokens, train_losses, label=f"{optimizer_label} (train)", color=color)
        
        # Plot validation losses if available and not disabled
        if not args.no_val and val_steps and val_losses:
            has_validation_data = True
            val_tokens = []
            for step in val_steps:
                # Find corresponding tokens for this step
                if step < len(tokens):
                    val_tokens.append(tokens[step-1])  # Adjust for step indexing
                else:
                    # Estimate based on last available token count
                    val_tokens.append(tokens[-1] * step / len(tokens))
            
            plt.plot(val_tokens, smoothed_val_losses, '--', color=color, label=f"{optimizer_label} (val)")
    
    plt.xlabel("Total tokens processed")
    plt.ylabel("Loss")
    title = "Training Loss (Full Range)" if args.no_val else "Training and Validation Loss (Full Range)"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Last 80% training and validation loss plot
    plt.subplot(1, 2, 2)
    
    for optimizer, logs in optimizer_logs.items():
        # Use the most recent log file for each optimizer
        if not logs:
            continue
            
        # Sort logs by modification time and pick the most recent
        logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        log_file = logs[0]
        
        tokens, train_losses, val_steps, val_losses, smoothed_val_losses = process_log(log_file)
        
        if not tokens or not train_losses:
            continue
        
        # Calculate 20% cutoff point
        cutoff = max(1, int(0.2 * len(tokens)))
        tokens_last_80 = tokens[cutoff:]
        train_losses_last_80 = train_losses[cutoff:]
        
        if not tokens_last_80 or not train_losses_last_80:
            continue
        
        # Plot training losses for last 80%
        optimizer_label = optimizer.upper()
        color = color_map.get(optimizer.lower(), 'gray')
        plt.plot(tokens_last_80, train_losses_last_80, label=f"{optimizer_label} (train)", color=color)
        
        # Plot validation losses for last 80% if available and not disabled
        if not args.no_val and val_steps and val_losses:
            val_tokens = []
            for step in val_steps:
                # Find corresponding tokens for this step
                if step < len(tokens):
                    val_tokens.append(tokens[step-1])  # Adjust for step indexing
                else:
                    # Estimate based on last available token count
                    val_tokens.append(tokens[-1] * step / len(tokens))
            
            # Filter validation points to only include those in the last 80%
            last_80_points = [(t, l, s) for t, l, s in zip(val_tokens, val_losses, smoothed_val_losses) 
                              if tokens_last_80 and t >= tokens_last_80[0]]
            
            if last_80_points:
                last_tokens, last_losses, last_smoothed = zip(*last_80_points)
                plt.plot(last_tokens, last_smoothed, '--', color=color, label=f"{optimizer_label} (val)")
    
    plt.xlabel("Total tokens processed")
    plt.ylabel("Loss")
    title = "Training Loss (Last 80%)" if args.no_val else "Training and Validation Loss (Last 80%)"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if args.output:
        plt.savefig(args.output)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()