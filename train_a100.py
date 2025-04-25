#!/usr/bin/env python
# Script to run distributed training across 2x A100 40GB GPUs
# Utilizes optimized Modal deployment for high-performance training

import os
import modal
import yaml
from pathlib import Path
import uuid
import argparse

# Define Modal stub and container image
app = modal.App("mlx-pretrain-a100")

# Create a Modal image with all required dependencies
image = modal.Image.debian_slim().pip_install(
    "mlx==0.25.0",
    "mlx_lm==0.23.2",
    "mlx_optimizers==0.4.1", 
    "numpy==2.2.5",
    "PyYAML==6.0.2",
    "tokenizers==0.21.1",
    "tqdm==4.67.1",
    "torch>=2.0.0",
    "matplotlib==3.10.1",
)

# Create a specialized A100 GPU container for model training
a100_container = image.pip_install(
    "datasets>=2.14.5",
).run_commands(
    # Install any additional system dependencies
    "apt-get update && apt-get install -y git wget curl",
    # Enable NCCL for multi-GPU communication
    "DEBIAN_FRONTEND=noninteractive apt-get install -y libnccl2 libnccl-dev",
)

@app.function(
    image=a100_container,
    gpu="A100-40GB:2",  # Request 2x A100 GPUs
    timeout=259200,  # 72 hours max runtime
    retries=1,       # Retry once on failure
    secrets=[modal.Secret.from_name("huggingface-token")]  # Optional: HF credentials if needed
)
def train_model_a100(config_path, data_dir, run_id, checkpoint=None):
    """
    Train a model using 2x A100 40GB GPUs with optimized settings.
    
    Args:
        config_path: Path to model config YAML
        data_dir: Directory containing training data
        run_id: Unique ID for this run
        checkpoint: Optional checkpoint to resume from
    """
    import sys
    import subprocess
    import os
    import shutil
    
    # Clone the repository
    subprocess.check_call(["git", "clone", "https://github.com/N8python/mlx-pretrain.git", "/mlx-pretrain"])
    
    # Change to the repository directory
    os.chdir("/mlx-pretrain")
    
    # Update data paths in the config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Point to the downloaded data files
    config["data"]["input_file"] = f"{data_dir}/train.jsonl"
    config["data"]["validation_file"] = f"{data_dir}/val.jsonl"
    config["data"]["tokenizer_path"] = f"{data_dir}/tokenizer"
    
    # Add run_id to name for uniqueness
    config["name"] = f"{config['name']}-{run_id}"
    
    # Ensure distributed training is enabled
    config["system"]["distributed"] = True
    config["system"]["devices"] = []  # No MLX devices
    config["system"]["cuda_devices"] = [0, 1]  # Use both A100 GPUs
    
    # Optimize batch size based on 40GB A100 memory
    config["training"]["hyperparameters"]["batch_size"] = 64  # Adjusted for A100 40GB
    
    # Save the updated config
    modal_config_path = "/mlx-pretrain/model-config-a100-modal.yaml"
    with open(modal_config_path, "w") as f:
        yaml.dump(config, f)
    
    # Create necessary folders
    os.makedirs("runs", exist_ok=True)
    
    # Prepare training command
    train_cmd = ["python", "train.py", "--config", modal_config_path]
    
    # Add checkpoint parameter if provided
    if checkpoint:
        # Create a modified config with resume settings
        with open(modal_config_path, "r") as f:
            resume_config = yaml.safe_load(f)
        
        # Add resume configuration
        resume_config["resume"] = {
            "checkpoint": checkpoint,
            "reset_optimizer": False
        }
        
        # Save the resume config
        resume_config_path = "/mlx-pretrain/model-config-a100-resume.yaml"
        with open(resume_config_path, "w") as f:
            yaml.dump(resume_config, f)
        
        # Update training command to use resume config
        train_cmd = ["python", "train.py", "--config", resume_config_path]
    
    # Print A100 GPU info
    subprocess.check_call(["nvidia-smi"])
    
    # Start training with detailed logging
    print(f"Starting training with command: {' '.join(train_cmd)}")
    subprocess.check_call(train_cmd)
    
    # After training, copy results to shared volume
    results_dir = f"/tmp/{run_id}_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Copy the trained model and logs
    if os.path.exists("runs"):
        shutil.copytree("runs", os.path.join(results_dir, "runs"))
    
    return f"Training complete. Results saved to {results_dir}"

@app.function(
    image=a100_container,
    gpu="A100-40GB:1",  # Use 1x A100 for inference
)
def generate_sample(run_name, prompt, run_id):
    """Generate sample text using the trained model"""
    import sys
    import subprocess
    import os
    
    # Clone the repository
    subprocess.check_call(["git", "clone", "https://github.com/N8python/mlx-pretrain.git", "/mlx-pretrain"])
    
    # Change to the repository directory
    os.chdir("/mlx-pretrain")
    
    # Create necessary directories
    os.makedirs("runs", exist_ok=True)
    
    # Copy the model from the results directory
    results_dir = f"/tmp/{run_id}_results"
    if os.path.exists(os.path.join(results_dir, "runs", run_name)):
        subprocess.check_call([
            "cp", "-r", 
            os.path.join(results_dir, "runs", run_name),
            "runs/"
        ])
    
    # Generate text with the model
    result = subprocess.check_output([
        "python", "generate.py",
        "--run", run_name,
        "--prompt", prompt
    ], text=True)
    
    return result

@app.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(description="Run MLX training on A100 GPUs")
    parser.add_argument("--config", type=str, default="model-config-a100-distributed.yaml", 
                        help="Path to the model configuration YAML")
    parser.add_argument("--data-dir", type=str, default="./data", 
                        help="Directory containing training data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint to resume from")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for sample generation after training")
    args = parser.parse_args()
    
    # Generate a unique run ID
    run_id = str(uuid.uuid4())[:8]
    print(f"Starting run with ID: {run_id}")
    
    # Start training
    print(f"Starting A100 training with config: {args.config}")
    result = train_model_a100.remote(args.config, args.data_dir, run_id, args.checkpoint)
    print(result)
    
    # Get the model name from the config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_name = f"{config['name']}-{run_id}"
    
    # Generate some text to demo the model
    print(f"Generating sample text with prompt: '{args.prompt}'")
    sample_text = generate_sample.remote(model_name, args.prompt, run_id)
    print(sample_text)

if __name__ == "__main__":
    main()