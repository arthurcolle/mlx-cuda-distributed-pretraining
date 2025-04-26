#!/usr/bin/env python
# Connector for Modal.com A100 GPU workers
# Provides an interface for the hybrid distributed system to use Modal

import os
import yaml
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import functools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modal_connector")

# Global Modal app instance and container reference
modal_app = None
container_image = None

# Define a global worker function for Modal to use
def remote_worker(operation, payload):
    """
    Remote worker function for distributed computation.
    
    Args:
        operation: Operation to perform
        payload: Operation-specific data
        
    Returns:
        Operation result
    """
    import os
    import json
    import torch
    import numpy as np
    import sys
    import subprocess
    
    # Set up environment
    os.chdir("/workspace")
    
    # Install MLX locally in the container if needed
    if operation in ["forward_backward", "parameter_update"]:
        subprocess.check_call([
            "pip", "install", 
            "mlx>=0.0.1",
            "mlx_lm>=0.0.1",
            "mlx_optimizers>=0.0.1"
        ])
    
    # Print GPU info
    subprocess.check_call(["nvidia-smi"])
    
    # Handle different operations
    if operation == "forward_backward":
        # Extract model config
        model_config = payload.get("model_config", {})
        inputs = np.array(payload["inputs"])
        targets = np.array(payload["targets"])
        
        # Write model config to file
        config_path = "/workspace/temp_config.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(model_config, f)
        
        # Load model weights if provided
        weights_path = None
        if "model_weights" in payload:
            weights_path = "/workspace/temp_weights.safetensors"
            # Write weights to file
            # This is a simplification - real implementation would handle
            # proper safetensors serialization
            with open(weights_path, 'w') as f:
                json.dump(payload["model_weights"], f)
        
        # Run forward/backward pass
        import torch
        import torch.nn as nn
        
        # Create simple transformer-based model matching the config
        # This is a placeholder - real implementation would load the model
        # based on the provided config
        hidden_size = model_config.get("model", {}).get("dimensions", {}).get("hidden_size", 768)
        num_layers = model_config.get("model", {}).get("dimensions", {}).get("num_layers", 12)
        
        # Forward pass using PyTorch
        inputs_tensor = torch.tensor(inputs).to("cuda")
        targets_tensor = torch.tensor(targets).to("cuda")
        
        # Run custom computation command
        if "compute_script" in payload:
            script_path = "/workspace/compute.py"
            with open(script_path, 'w') as f:
                f.write(payload["compute_script"])
            
            # Execute compute script with inputs
            cmd = [
                "python", script_path,
                "--inputs", json.dumps(inputs.tolist()),
                "--targets", json.dumps(targets.tolist()),
                "--config", config_path
            ]
            if weights_path:
                cmd.extend(["--weights", weights_path])
            
            result_json = subprocess.check_output(cmd, text=True)
            try:
                result = json.loads(result_json)
                return result
            except json.JSONDecodeError:
                return {"error": "Invalid JSON returned from compute script", "output": result_json}
        
        # Simple placeholder simulation
        loss = torch.tensor(1.0, device="cuda")  # Simulated loss
        tokens = inputs.shape[0] * inputs.shape[1]
        
        # Create mock gradients
        gradients = {}
        for param_name in payload.get("param_names", ["weight", "bias"]):
            gradients[param_name] = np.random.randn(hidden_size, hidden_size).tolist()
        
        return {
            "loss": float(loss.item()),
            "tokens": int(tokens),
            "gradients": gradients
        }
        
    elif operation == "parameter_update":
        # Extract model and optimizer state
        model_state = payload["model_state"]
        gradients = payload["gradients"]
        optimizer_config = payload.get("optimizer_config", {})
        
        # Simulate parameter update
        updated_model = {}
        for name, param_data in model_state.items():
            if name in gradients:
                # Simple SGD update simulation
                lr = optimizer_config.get("learning_rate", 0.01)
                param = np.array(param_data)
                grad = np.array(gradients[name])
                param -= lr * grad
                updated_model[name] = param.tolist()
            else:
                updated_model[name] = param_data
        
        return {"updated_model": updated_model}
        
    elif operation == "status":
        # Return GPU status
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        lines = gpu_info.strip().split('\n')
        gpu_status = []
        
        for i, line in enumerate(lines):
            mem_used, mem_total, util = line.split(', ')
            gpu_status.append({
                "gpu_id": i,
                "memory_used_mb": int(mem_used),
                "memory_total_mb": int(mem_total),
                "utilization_percent": int(util)
            })
        
        return {
            "status": "ready",
            "gpu_info": gpu_status,
            "timestamp": time.time()
        }
        
    else:
        return {"error": f"Unsupported operation: {operation}"}

class ModalConnector:
    """
    Connector for Modal.com remote workers with A100 GPUs.
    Provides an abstraction layer for the hybrid distributed system.
    """
    def __init__(
        self, 
        app_name: str = "mlx-hybrid-training", 
        gpu_count: int = 2, 
        gpu_type: str = "A100-40GB",
        timeout: int = 3600,
        config_dir: str = "./configs",
        data_dir: str = "./data"
    ):
        """
        Initialize Modal connector.
        
        Args:
            app_name: Name of the Modal application
            gpu_count: Number of GPUs to use
            gpu_type: Type of GPU to use
            timeout: Maximum runtime in seconds
            config_dir: Directory for configuration files
            data_dir: Directory containing data files
        """
        self.app_name = app_name
        self.gpu_count = gpu_count
        self.gpu_type = gpu_type
        self.timeout = timeout
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.modal_app = None
        self.modal_function = None
        self.session_id = str(uuid.uuid4())[:8]
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Try to import Modal
        try:
            import modal
            self.modal = modal
            self.initialize_modal()
        except ImportError:
            logger.error("Modal SDK not installed. Run 'pip install modal'")
            self.modal = None
    
    def initialize_modal(self):
        """Initialize Modal app and container image"""
        global modal_app, container_image
        
        if not self.modal:
            return
        
        # Create Modal app
        self.modal_app = self.modal.App(self.app_name)
        modal_app = self.modal_app
        
        # Create base image with dependencies
        image = self.modal.Image.debian_slim().pip_install(
            "numpy>=1.24.0",
            "PyYAML>=6.0",
            "tokenizers>=0.13.0", 
            "tqdm>=4.66.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "mpmath>=1.3.0",
            "datasets>=2.14.0",
            "typing_extensions>=4.8.0"
        ).run_commands(
            # Install system dependencies
            "apt-get update && apt-get install -y git wget curl",
            # Enable NCCL for multi-GPU communication
            "DEBIAN_FRONTEND=noninteractive apt-get install -y libnccl2 libnccl-dev"
        )
        
        # Create a temporary build directory with just the files we need
        import tempfile
        import shutil
        import glob
        
        # Create a temporary directory for the build context
        build_dir = tempfile.mkdtemp(prefix="modal_connector_build_")
        print(f"Created temporary build directory: {build_dir}")
        
        # Create essential subdirectories
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
        
        # Copy files to the build directory
        for pattern, subdir in essential_files.items():
            dest_dir = os.path.join(build_dir, subdir) if subdir else build_dir
            os.makedirs(dest_dir, exist_ok=True)
            
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    print(f"Copying {file_path} to {os.path.join(dest_dir, os.path.basename(file_path))}")
                    shutil.copy2(file_path, os.path.join(dest_dir, os.path.basename(file_path)))
        
        # Use the build directory for the container
        container = (image
            .pip_install(
                "mlx>=0.0.1", 
                "mlx_lm>=0.0.1", 
                "mlx_optimizers>=0.0.1"
            )
            .add_local_dir(build_dir, remote_path="/workspace")
        )
        container_image = container
        
        # Store the necessary components to create the Modal function later
        # Instead of applying @app.function within this method
        self.container = container
        self.remote_worker_func = remote_worker
        
        # Create a placeholder for the actual Modal function
        self.modal_function = None
        logger.info(f"Modal connector initialized with {self.gpu_count}x {self.gpu_type} GPUs")
    
    def compute_forward_backward(self, model_state, inputs, targets, compute_script=None):
        """
        Compute forward and backward passes on the remote Modal worker.
        
        Args:
            model_state: Dictionary of model parameters
            inputs: Input tensor data
            targets: Target tensor data
            compute_script: Optional Python script to run on the worker
            
        Returns:
            Dictionary with loss, tokens processed, and gradients
        """
        if not self.modal:
            raise RuntimeError("Modal not initialized")
        
        # Extract model configuration
        if "model_config" in model_state:
            model_config = model_state["model_config"]
        else:
            # Default minimal config
            model_config = {
                "model": {
                    "dimensions": {
                        "hidden_size": 768,
                        "num_layers": 12
                    }
                }
            }
        
        # Prepare payload
        payload = {
            "model_config": model_config,
            "inputs": inputs,
            "targets": targets,
            "param_names": list(model_state.keys())
        }
        
        # Add model weights if available
        if "weights" in model_state:
            payload["model_weights"] = model_state["weights"]
        
        # Add compute script if provided
        if compute_script:
            payload["compute_script"] = compute_script
        
        # Create the Modal function for this specific run if it doesn't exist
        if not self.modal_function:
            self.modal_function = self.modal_app.function(
                image=self.container,
                gpu=f"{self.gpu_type}:{self.gpu_count}",
                timeout=self.timeout,
                retries=1
            )(self.remote_worker_func)
        
        # Execute on Modal
        try:
            with self.modal_app.run():
                result = self.modal_function.remote("forward_backward", payload)
            return result
        except Exception as e:
            logger.error(f"Error in Modal forward/backward computation: {str(e)}")
            return {"error": str(e)}
    
    def update_parameters(self, model_state, gradients, optimizer_config=None):
        """
        Update parameters on the remote worker.
        
        Args:
            model_state: Current model parameters
            gradients: Gradients to apply
            optimizer_config: Optimizer configuration
            
        Returns:
            Updated model state
        """
        if not self.modal:
            raise RuntimeError("Modal not initialized")
        
        # Default optimizer config if not provided
        if optimizer_config is None:
            optimizer_config = {
                "learning_rate": 0.01,
                "optimizer": "sgd"
            }
        
        # Prepare payload
        payload = {
            "model_state": model_state,
            "gradients": gradients,
            "optimizer_config": optimizer_config
        }
        
        # Create the Modal function for this specific run if it doesn't exist
        if not self.modal_function:
            self.modal_function = self.modal_app.function(
                image=self.container,
                gpu=f"{self.gpu_type}:{self.gpu_count}",
                timeout=self.timeout,
                retries=1
            )(self.remote_worker_func)
        
        # Execute on Modal
        try:
            with self.modal_app.run():
                result = self.modal_function.remote("parameter_update", payload)
            return result
        except Exception as e:
            logger.error(f"Error in Modal parameter update: {str(e)}")
            return {"error": str(e)}
    
    def check_status(self):
        """
        Check the status of the Modal worker.
        
        Returns:
            Status information
        """
        if not self.modal:
            raise RuntimeError("Modal not initialized")
        
        # Create the Modal function for this specific run if it doesn't exist
        if not self.modal_function:
            self.modal_function = self.modal_app.function(
                image=self.container,
                gpu=f"{self.gpu_type}:{self.gpu_count}",
                timeout=self.timeout,
                retries=1
            )(self.remote_worker_func)
        
        try:
            with self.modal_app.run():
                result = self.modal_function.remote("status", {})
            return result
        except Exception as e:
            logger.error(f"Error checking Modal worker status: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_training_worker(self, config_path, run_id=None):
        """
        Run a complete training job on Modal using config file.
        
        Args:
            config_path: Path to configuration YAML
            run_id: Optional run ID (generated if not provided)
            
        Returns:
            Result information
        """
        if not self.modal:
            raise RuntimeError("Modal SDK not installed")
        
        # Import the app and function for running the full training
        import train_a100
        
        # Generate run ID if not provided
        run_id = run_id or str(uuid.uuid4())[:8]
        
        # Run the training
        try:
            with train_a100.app.run():
                result = train_a100.train_model_a100.remote(
                    config_path=config_path,
                    data_dir=self.data_dir,
                    run_id=run_id
                )
            return {"status": "success", "result": result, "run_id": run_id}
        except Exception as e:
            logger.error(f"Error running Modal training: {str(e)}")
            return {"status": "error", "error": str(e), "run_id": run_id}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Modal connector")
    parser.add_argument("--config", type=str, default="model-config-muon.yaml", 
                       help="Path to model configuration")
    parser.add_argument("--run-id", type=str, default=None,
                       help="Unique run ID (generated if not provided)")
    parser.add_argument("--action", type=str, choices=["status", "train"],
                       default="status", help="Action to perform")
    args = parser.parse_args()
    
    # Create connector
    connector = ModalConnector()
    
    if args.action == "status":
        # Check status
        status = connector.check_status()
        print(json.dumps(status, indent=2))
    elif args.action == "train":
        # Run training
        result = connector.run_training_worker(args.config, args.run_id)
        print(json.dumps(result, indent=2))