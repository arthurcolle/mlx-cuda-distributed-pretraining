import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from pathlib import Path

def process_log(log_file: Path) -> tuple[list, list, list, list, list]:
    """Process a single log file and return tokens, training losses, and validation data."""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Parse training losses from regular log entries
    #train_losses = []
    #tokens = [0]
    train_steps = []
    
    # Parse validation losses
    val_steps = []
    val_losses = []
    
    for line in lines:
        if line.startswith("Step") and "validation:" not in line:
            step = int(line.split()[1][:-1])
            # Regular training log
            parts = line.split("|")
            # First part contains loss
            loss_part = next((p for p in parts if "loss=" in p), None)
            loss = float(loss_part.split("=")[1].strip())

            toks_part = next((p for p in parts if "toks=" in p), None)
            toks = float(toks_part.split("=")[1].strip())
            train_steps.append((step, loss, toks))
            """"if loss_part:
                loss = float(loss_part.split("=")[1].strip())
                train_losses.append(loss)
                
                # Find tokens processed
                toks_part = next((p for p in parts if "toks=" in p), None)
                if toks_part:
                    toks = float(toks_part.split("=")[1].strip())
                    tokens.append(toks + tokens[-1])"""
        
        elif "validation:" in line:
            # Validation log
            step = int(line.split()[1])
            val_loss = float(line.split("val_loss=")[1].split()[0])
            val_steps.append(step)
            val_losses.append(val_loss)
    # Sort train_steps
    train_steps.sort(key=lambda x: x[0])
    deduped_train_steps = []
    for step, loss, toks in train_steps:
        if len(deduped_train_steps) == 0 or deduped_train_steps[-1][0] != step:
            deduped_train_steps.append((step, loss, toks))
    train_losses = []
    tokens = [0]
    for step, loss, toks in deduped_train_steps:
        train_losses.append(loss)
        # Append tokens processed
        tokens.append(toks + tokens[-1])
    # Deduplicate steps
    # Ensure tokens list has same length as losses
    if len(tokens) > len(train_losses) + 1:
        tokens = tokens[:len(train_losses) + 1]
    tokens = tokens[1:]
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
    parser.add_argument('run_names', type=str, nargs='+', help='Names of the training runs to plot')
    parser.add_argument('--no-val', action='store_true', help='Ignore validation data when plotting')
    args = parser.parse_args()

    # Create a figure with 1 row, 2 columns
    plt.figure(figsize=(16, 8))
    
    # Full range training and validation loss plot
    plt.subplot(1, 2, 1)
    has_validation_data = False
    
    for run_name in args.run_names:
        log_file = Path("runs") / run_name / "log.txt"
        if not log_file.exists():
            print(f"Error: Log file not found at {log_file}")
            continue
            
        tokens, train_losses, val_steps, val_losses, smoothed_val_losses = process_log(log_file)
        
        # Plot training losses
        plt.plot(tokens, train_losses, label=f"{run_name} (train EMA)")
        
        # Plot validation losses if available and not disabled
        if not args.no_val and val_steps and val_losses:
            has_validation_data = True
            val_tokens = []
            for step in val_steps:
                if step < len(tokens):
                    val_tokens.append(tokens[step])
                else:
                    # Estimate based on last available token count
                    val_tokens.append(tokens[-1] * step / len(tokens))
            
            #plt.plot(val_tokens, val_losses, 'o', alpha=0.5, label=f"{run_name} (val)")
            plt.plot(val_tokens, smoothed_val_losses, '-', label=f"{run_name} (val EMA)")
    
    plt.xlabel("Total tokens processed")
    plt.ylabel("Loss")
    title = "Training Loss (Full Range)" if args.no_val else "Training and Validation Loss (Full Range)"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Last 80% training and validation loss plot
    plt.subplot(1, 2, 2)
    
    for run_name in args.run_names:
        log_file = Path("runs") / run_name / "log.txt"
        if not log_file.exists():
            continue
            
        tokens, train_losses, val_steps, val_losses, smoothed_val_losses = process_log(log_file)
        
        # Calculate 20% cutoff point
        cutoff = int(0.2 * len(tokens))
        tokens_last_80 = tokens[cutoff:]
        train_losses_last_80 = train_losses[cutoff:]
        
        # Plot training losses for last 80%
        plt.plot(tokens_last_80, train_losses_last_80, label=f"{run_name} (train EMA)")
        
        # Plot validation losses for last 80% if available and not disabled
        if not args.no_val and val_steps and val_losses:
            val_tokens = []
            for step in val_steps:
                if step < len(tokens):
                    val_tokens.append(tokens[step])
                else:
                    # Estimate based on last available token count
                    val_tokens.append(tokens[-1] * step / len(tokens))
            
            # Filter validation points to only include those in the last 80%
            last_80_points = [(t, l, s) for t, l, s in zip(val_tokens, val_losses, smoothed_val_losses) 
                              if t >= tokens_last_80[0]]
            
            if last_80_points:
                last_tokens, last_losses, last_smoothed = zip(*last_80_points)
                #plt.plot(last_tokens, last_losses, 'o', alpha=0.5, label=f"{run_name} (val)")
                plt.plot(last_tokens, last_smoothed, '-', label=f"{run_name} (val EMA)")
    
    plt.xlabel("Total tokens processed")
    plt.ylabel("Loss")
    title = "Training Loss (Last 80%)" if args.no_val else "Training and Validation Loss (Last 80%)"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
