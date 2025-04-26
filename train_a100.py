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
# Start with a standardized image that's known to work
image = modal.Image.from_registry(
    "python:3.10-slim-bullseye"
).run_commands(
    # Install system dependencies first
    "apt-get update",
    "apt-get install -y git wget curl build-essential",
    "DEBIAN_FRONTEND=noninteractive apt-get install -y libnccl2 libnccl-dev"
).pip_install(
    # Install Python dependencies
    "numpy==2.2.0",  # Use exact versions to avoid compatibility issues
    "PyYAML==6.0",
    "tokenizers==0.13.3",
    "tqdm==4.66.1",
    "torch==2.0.1",
    "matplotlib==3.7.2",
    "transformers==4.30.2",
    "mpmath==1.3.0",
    "datasets==2.14.5",
    "typing_extensions==4.8.0",
    "tiktoken==0.5.1"
)

# Create a specialized A100 GPU container for model training
# Copy only the essential files to avoid file change issues
import tempfile
import shutil

# Create a specialized A100 GPU container for model training
# Explicitly add only the specific files we need to avoid modification errors
import glob
import shutil

# Print current directory for debugging
print(f"Current directory: {os.getcwd()}")

# Create a temporary directory for the build context
build_dir = tempfile.mkdtemp(prefix="modal_build_")
print(f"Created temporary build directory: {build_dir}")

# Create essential directories in build context
os.makedirs(os.path.join(build_dir, "configs"), exist_ok=True)
os.makedirs(os.path.join(build_dir, "tokenizer"), exist_ok=True)

# Copy essential files explicitly
essential_files = {
    "*.py": "",
    "*.yaml": "",
    "*.json": "",
    "requirements.txt": "",
    "configs/*": "configs/",
    "tokenizer/*": "tokenizer/"
}

# Copy files to build directory
for pattern, subdir in essential_files.items():
    dest_dir = os.path.join(build_dir, subdir) if subdir else build_dir
    os.makedirs(dest_dir, exist_ok=True)
    
    for file_path in glob.glob(pattern):
        if os.path.isfile(file_path):
            print(f"Copying {file_path} to {os.path.join(dest_dir, os.path.basename(file_path))}")
            shutil.copy2(file_path, os.path.join(dest_dir, os.path.basename(file_path)))

# Use the explicit approach for the container with pip installs
a100_container = (image
    .run_commands(
        # Additional diagnostic commands to check environment before pip installs
        "python -V",
        "pip --version",
        "df -h",
        # Install MLX packages with specific commands that are known to work with the image
        "pip install mlx==0.0.10",  # Using an older version known to work
        "pip install mlx-lm==0.0.3", 
        "pip install --no-deps mlx-optimizers==0.0.4"
    )
    .add_local_dir(build_dir, remote_path="/data")
)

@app.function(
    image=a100_container,
    gpu="A100-80GB:3",  # Request 3x A100 80GB GPUs for large memory needs
    timeout=259200,  # 72 hours max runtime
    retries=1,       # Retry once on failure
    scaledown_window=300,  # 5 minutes idle timeout
    secrets=[modal.Secret.from_name("distributed-systems")]  # Using distributed-systems secret
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
    
    # Clone the repository with more detailed output
    print("Cloning the repository...")
    try:
        # First try with verbose output
        clone_result = subprocess.run(
            ["git", "clone", "-v", "https://github.com/arthurcolle/mlx-cuda-distributed-pretraining.git", "/mlx-pretrain"],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Clone stdout: {clone_result.stdout}")
        print(f"Clone stderr: {clone_result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        print("Trying alternative clone method...")
        
        # Try an alternative clone approach
        try:
            os.makedirs("/mlx-pretrain", exist_ok=True)
            subprocess.check_call(["git", "init", "/mlx-pretrain"])
            os.chdir("/mlx-pretrain")
            subprocess.check_call(["git", "remote", "add", "origin", "https://github.com/arthurcolle/mlx-cuda-distributed-pretraining.git"])
            subprocess.check_call(["git", "fetch", "--depth=1", "origin", "main"])
            subprocess.check_call(["git", "checkout", "main"])
            print("Alternative clone completed successfully")
        except subprocess.CalledProcessError as alt_e:
            print(f"Alternative clone failed: {alt_e}")
            # Fall back to copying files from the current directory
            print("Falling back to copying files from the current directory")
            for item in os.listdir("/data"):
                src = os.path.join("/data", item)
                dst = os.path.join("/mlx-pretrain", item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
    
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
    config["system"]["cuda_devices"] = [0, 1, 2]  # Use all three A100 GPUs
    
    # Optimize batch size based on 80GB A100 memory
    config["training"]["hyperparameters"]["batch_size"] = 128  # Adjusted for A100 80GB
    
    # Save the updated config
    modal_config_path = "/mlx-pretrain/model-config-a100-modal.yaml"
    with open(modal_config_path, "w") as f:
        yaml.dump(config, f)
    
    # Create necessary folders
    os.makedirs("runs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
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
    
    # Check the container environment
    print("===== CONTAINER ENVIRONMENT =====")
    subprocess.check_call(["env"])
    
    # Check python version and installed packages
    print("===== PYTHON VERSION =====")
    subprocess.check_call(["python", "--version"])
    print("===== INSTALLED PACKAGES =====")
    subprocess.check_call(["pip", "list"])
    
    # Install MLX locally in the container for training
    print("===== INSTALLING MLX AND RELATED PACKAGES =====")
    try:
        # More verbose pip installation with exact versions
        subprocess.check_call([
            "pip", "install", "-v",
            "mlx==0.25.0",
            "mlx_lm==0.23.2",
            "mlx_optimizers==0.4.1"
        ])
        # Verify installation
        print("===== VERIFYING INSTALLATION =====")
        subprocess.check_call(["pip", "list", "|", "grep", "mlx"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        
        # Try installing packages individually with more debugging
        for pkg in ["mlx==0.25.0", "mlx_lm==0.23.2", "mlx_optimizers==0.4.1"]:
            try:
                print(f"Trying to install {pkg} individually...")
                subprocess.check_call(["pip", "install", "-v", pkg])
            except subprocess.CalledProcessError as pkg_e:
                print(f"Failed to install {pkg}: {pkg_e}")
                print("Detailed pip install output with debug info:")
                try:
                    debug_output = subprocess.run(
                        ["pip", "install", "-v", "--log", "pip_debug.log", pkg],
                        capture_output=True, text=True
                    )
                    print(f"STDOUT: {debug_output.stdout}")
                    print(f"STDERR: {debug_output.stderr}")
                    if os.path.exists("pip_debug.log"):
                        with open("pip_debug.log", "r") as f:
                            print(f"PIP DEBUG LOG: {f.read()}")
                except Exception as log_e:
                    print(f"Error during debug logging: {log_e}")
        
        # Print system info for diagnosis
        print("===== SYSTEM INFORMATION FOR DIAGNOSIS =====")
        subprocess.call(["uname", "-a"])
        subprocess.call(["cat", "/etc/os-release"])
        subprocess.call(["df", "-h"])
        subprocess.call(["cat", "/proc/cpuinfo"])
        
        print("Will continue despite package installation issues...")
    
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
    gpu="A100-80GB:1",  # Use 1x A100 80GB for inference
    scaledown_window=300,  # 5 minutes idle timeout
    secrets=[modal.Secret.from_name("distributed-systems")]  # Using distributed-systems secret
)
def generate_sample(run_name, prompt, run_id):
    """Generate sample text using the trained model"""
    import sys
    import subprocess
    import os
    
    # Clone the repository with more detailed output
    print("Cloning the repository...")
    try:
        # First try with verbose output
        clone_result = subprocess.run(
            ["git", "clone", "-v", "https://github.com/arthurcolle/mlx-cuda-distributed-pretraining.git", "/mlx-pretrain"],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Clone stdout: {clone_result.stdout}")
        print(f"Clone stderr: {clone_result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        print("Trying alternative clone method...")
        
        # Try an alternative clone approach
        try:
            os.makedirs("/mlx-pretrain", exist_ok=True)
            subprocess.check_call(["git", "init", "/mlx-pretrain"])
            os.chdir("/mlx-pretrain")
            subprocess.check_call(["git", "remote", "add", "origin", "https://github.com/arthurcolle/mlx-cuda-distributed-pretraining.git"])
            subprocess.check_call(["git", "fetch", "--depth=1", "origin", "main"])
            subprocess.check_call(["git", "checkout", "main"])
            print("Alternative clone completed successfully")
        except subprocess.CalledProcessError as alt_e:
            print(f"Alternative clone failed: {alt_e}")
            # Fall back to copying files from the current directory
            print("Falling back to copying files from the current directory")
            for item in os.listdir("/data"):
                src = os.path.join("/data", item)
                dst = os.path.join("/mlx-pretrain", item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
    
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
    
    # Install MLX locally in the container for generation
    print("Installing MLX and related packages inside the container...")
    subprocess.check_call([
        "pip", "install", 
        "mlx>=0.0.1",  # Install latest available version
        "mlx_lm>=0.0.1",
        "mlx_optimizers>=0.0.1"
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