import os
import time
import modal
import yaml
from pathlib import Path

# Define Modal stub and container image
stub = modal.Stub("mlx-pretrain")

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

# Download tokenizer and data files
@stub.function(image=image)
def download_data(run_id):
    import subprocess
    
    # Create output directory for this run
    os.makedirs(f"/modal_data/{run_id}/tokenizer", exist_ok=True)
    
    # Download tokenizer and training data
    subprocess.check_call([
        "wget", "https://huggingface.co/datasets/N8Programs/mlx-pretrain-ex/resolve/main/train.jsonl",
        "-O", f"/modal_data/{run_id}/train.jsonl"
    ])
    
    subprocess.check_call([
        "wget", "https://huggingface.co/datasets/N8Programs/mlx-pretrain-ex/resolve/main/val.jsonl",
        "-O", f"/modal_data/{run_id}/val.jsonl"
    ])
    
    # Download tokenizer (you may need to adjust this URL for your tokenizer)
    subprocess.check_call([
        "wget", "https://huggingface.co/datasets/N8Programs/mlx-pretrain-ex/resolve/main/tokenizer.json",
        "-O", f"/modal_data/{run_id}/tokenizer/tokenizer.json"
    ])
    
    return f"/modal_data/{run_id}"

# Create a special GPU container for model training
# This container will use A10G GPUs
gpu_container = image.pip_install(
    "datasets>=2.14.5",
).run_commands(
    # Install any additional system dependencies
    "apt-get update && apt-get install -y git wget curl",
)

@stub.function(
    image=gpu_container,
    gpu=modal.gpu.A10G(count=2),  # Request 2x A10G GPUs
    timeout=72000,  # 20 hours max runtime
    volumes={"/modal_data": modal.SharedVolume()},
    secrets=[modal.Secret.from_name("huggingface-token")]  # Optional: HF credentials if needed
)
def train_1b_model(config_path, data_dir, run_id):
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
    
    # Save the updated config
    with open("/mlx-pretrain/model-config-1b-modal.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Create necessary folders
    os.makedirs("runs", exist_ok=True)
    
    # Start training
    subprocess.check_call([
        "python", "train.py", 
        "--config", "/mlx-pretrain/model-config-1b-modal.yaml"
    ])
    
    # After training, copy results to shared volume
    results_dir = os.path.join("/modal_data", f"{run_id}_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Copy the trained model and logs
    if os.path.exists("runs"):
        shutil.copytree("runs", os.path.join(results_dir, "runs"))
    
    return f"Training complete. Results saved to {results_dir}"

@stub.function(
    image=image,
    gpu=modal.gpu.A100(count=1),  # Use A100 for inference demos
    volumes={"/modal_data": modal.SharedVolume()},
)
def generate_text(run_name, prompt, run_id):
    import sys
    import subprocess
    import os
    
    # Clone the repository
    subprocess.check_call(["git", "clone", "https://github.com/N8python/mlx-pretrain.git", "/mlx-pretrain"])
    
    # Change to the repository directory
    os.chdir("/mlx-pretrain")
    
    # Create a symbolic link to the model
    results_dir = os.path.join("/modal_data", f"{run_id}_results")
    os.makedirs("runs", exist_ok=True)
    
    # Copy the model from the shared volume if not using a symlink
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

@stub.local_entrypoint()
def main():
    import uuid
    
    # Generate a unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    # Load the 1B model config
    with open("model-config-1b.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Download the data
    print("Downloading data...")
    data_dir = download_data.remote(run_id)
    print(f"Data downloaded to {data_dir}")
    
    # Start training
    print("Starting training...")
    result = train_1b_model.remote("model-config-1b.yaml", data_dir, run_id)
    print(result)
    
    # Optional: Generate some text to demo the model
    print("Generating sample text...")
    sample_text = generate_text.remote(config["name"], "Once upon a time in a galaxy far away", run_id)
    print(sample_text)

# Web app endpoint to generate text
@stub.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    volumes={"/modal_data": modal.SharedVolume()},
)
@modal.web_endpoint(method="POST")
def generate_endpoint(run_id: str, run_name: str, prompt: str):
    """Web API endpoint for text generation with the trained model"""
    result = generate_text.remote(run_name, prompt, run_id)
    return {"generated_text": result}