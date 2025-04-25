#!/usr/bin/env python
import os
import json
import argparse
from pathlib import Path

def get_model_stats(run_dir=None, log_file=None):
    """Get statistics about the model training progress"""
    stats = {
        "name": None,
        "parameters": None,
        "steps_completed": 0,
        "current_loss": None,
        "checkpoints": [],
        "status": "Unknown"
    }
    
    # Check run directory
    if run_dir and os.path.exists(run_dir):
        # Check metadata.json
        metadata_path = os.path.join(run_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    stats["name"] = metadata.get("name")
                    
                    # Get model size from training_info if available
                    if "training_info" in metadata:
                        if "parameters" in metadata["training_info"]:
                            stats["parameters"] = metadata["training_info"]["parameters"]
                        
                    # Get checkpoint info
                    if "checkpoints" in metadata:
                        for cp in metadata["checkpoints"]:
                            stats["checkpoints"].append({
                                "step": cp.get("step"),
                                "validation_loss": cp.get("validation_loss")
                            })
                        stats["steps_completed"] = metadata["checkpoints"][-1]["step"] if metadata["checkpoints"] else 0
                        
                    # Get validation data
                    if "validation" in metadata:
                        val_data = metadata["validation"]
                        if "losses" in val_data and val_data["losses"]:
                            stats["current_loss"] = val_data["losses"][-1]
            except Exception as e:
                print(f"Error reading metadata: {e}")
    
    # Check log file
    if log_file and os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
                # Check for model size
                for line in lines:
                    if "parameters" in line.lower():
                        param_parts = line.strip().split()
                        for i, part in enumerate(param_parts):
                            if part.lower() == "has" and i+1 < len(param_parts) and "parameters" in param_parts[i+2].lower():
                                try:
                                    stats["parameters"] = float(param_parts[i+1])
                                    break
                                except:
                                    pass
                
                # Look for training progress
                training_started = False
                for line in lines:
                    if "training started" in line.lower():
                        training_started = True
                        stats["status"] = "Training"
                    if "training completed" in line.lower():
                        stats["status"] = "Completed"
                        
                # Look for steps and loss info in the log
                step_lines = [line for line in lines if line.startswith("Step")]
                if step_lines:
                    last_step_line = step_lines[-1]
                    
                    # Get step number
                    try:
                        step_num = int(last_step_line.split()[1].strip(':'))
                        stats["steps_completed"] = max(stats["steps_completed"], step_num)
                    except:
                        pass
                    
                    # Get loss
                    if "loss=" in last_step_line:
                        try:
                            loss_part = last_step_line.split("loss=")[1].split()[0]
                            stats["current_loss"] = float(loss_part.strip(','))
                        except:
                            pass
                
                # Set status based on log content
                if stats["status"] == "Unknown" and training_started:
                    stats["status"] = "Training"
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    return stats

def find_model_paths(model_name=None):
    """Find paths for model directories and logs"""
    run_dir = None
    log_file = None
    
    # Check runs directory for model directory
    if model_name:
        potential_run_dir = os.path.join("runs", model_name)
        if os.path.exists(potential_run_dir):
            run_dir = potential_run_dir
            
            # Check for log file in run directory
            potential_log = os.path.join(run_dir, "log.txt")
            if os.path.exists(potential_log):
                log_file = potential_log
    
    # Check for log files in logs directory
    if model_name and not log_file:
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            # Look for pattern like train_modelname_*.log
            log_pattern = f"*{model_name.lower()}*.log"
            matching_logs = list(Path(logs_dir).glob(log_pattern))
            
            if matching_logs:
                # Sort by modification time to get the latest log
                latest_log = max(matching_logs, key=lambda p: p.stat().st_mtime)
                log_file = str(latest_log)
    
    # If still nothing found, look for the most recently modified log file
    if not log_file:
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
            if log_files:
                latest_log = max([os.path.join(logs_dir, f) for f in log_files], 
                                key=lambda p: os.path.getmtime(p))
                log_file = latest_log
    
    return run_dir, log_file

def print_model_info(stats):
    """Print model information in a formatted way"""
    print("\n" + "="*40)
    print(f"MODEL: {stats['name'] or 'Unknown'}")
    print("="*40)
    
    # Print basic stats
    if stats["parameters"]:
        print(f"Parameters:      {stats['parameters']:.2f}M")
    else:
        print(f"Parameters:      Unknown")
        
    print(f"Status:          {stats['status']}")
    print(f"Steps completed: {stats['steps_completed']}")
    
    if stats["current_loss"]:
        print(f"Current loss:    {stats['current_loss']:.6f}")
    else:
        print(f"Current loss:    Unknown")
    
    # Print checkpoint info
    if stats["checkpoints"]:
        print("\nCHECKPOINTS:")
        for cp in stats["checkpoints"]:
            step = cp.get("step", "Unknown")
            val_loss = cp.get("validation_loss", "N/A")
            val_loss_str = f"{val_loss:.6f}" if isinstance(val_loss, (int, float)) else val_loss
            print(f"  Step {step}: val_loss={val_loss_str}")
    
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description="Visualize MLX model training status")
    parser.add_argument("--model", type=str, help="Model name to find in runs directory", default="Muon-200M")
    parser.add_argument("--run-dir", type=str, help="Path to run directory")
    parser.add_argument("--log-file", type=str, help="Path to log file")
    args = parser.parse_args()
    
    # Find model paths
    run_dir = args.run_dir
    log_file = args.log_file
    
    if not run_dir or not log_file:
        auto_run_dir, auto_log_file = find_model_paths(args.model)
        run_dir = run_dir or auto_run_dir
        log_file = log_file or auto_log_file
    
    if not log_file:
        print(f"Error: Could not find log file for model {args.model}")
        return
        
    print(f"Using log file: {log_file}")
    if run_dir:
        print(f"Using run directory: {run_dir}")
    
    # Get model stats
    stats = get_model_stats(run_dir, log_file)
    
    # Print model info
    print_model_info(stats)

if __name__ == "__main__":
    main()