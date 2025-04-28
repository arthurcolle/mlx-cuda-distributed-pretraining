#!/usr/bin/env python
# Implementation of a distributed worker for the hybrid system
# This script runs on each node (MLX or CUDA) to handle computation

import os
import sys
import json
import argparse
import logging
import time
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("worker.log")
    ]
)
logger = logging.getLogger("hybrid_worker")

def setup_mlx_environment():
    """Set up MLX environment"""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        return True
    except ImportError:
        logger.error("MLX not installed. Cannot use MLX for computation.")
        return False

def setup_torch_environment():
    """Set up PyTorch environment"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            logger.warning("PyTorch installed but CUDA not available")
            return False
    except ImportError:
        logger.error("PyTorch not installed. Cannot use CUDA for computation.")
        return False

class HybridWorker:
    """Worker for hybrid distributed training"""
    def __init__(self, node_type="mlx", node_id=0, coordinator_url=None):
        """
        Initialize the worker.
        
        Args:
            node_type: Type of node ("mlx" or "cuda")
            node_id: ID of this node
            coordinator_url: URL of the coordinator service
        """
        self.node_type = node_type
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.is_running = False
        self.status = "initializing"
        self.task_queue = []
        
        # Initialize environment based on node type
        if node_type == "mlx":
            self.mlx_available = setup_mlx_environment()
            if not self.mlx_available:
                raise RuntimeError("MLX not available but required for MLX node")
        elif node_type == "cuda":
            self.cuda_available = setup_torch_environment()
            if not self.cuda_available:
                raise RuntimeError("CUDA not available but required for CUDA node")
        else:
            raise ValueError(f"Unsupported node type: {node_type}")
        
        logger.info(f"Initialized {node_type} worker with ID: {node_id}")
    
    def start(self):
        """Start the worker"""
        self.is_running = True
        self.status = "running"
        
        logger.info(f"Worker started: {self.node_type}-{self.node_id}")
        
        try:
            # Check if we should connect to coordinator
            if self.coordinator_url:
                self._run_coordinator_mode()
            else:
                self._run_standalone_mode()
        except Exception as e:
            logger.error(f"Error in worker: {e}", exc_info=True)
            self.status = "error"
        finally:
            self.is_running = False
    
    def _run_coordinator_mode(self):
        """Run in coordinator mode - fetch tasks from coordinator"""
        import requests
        import time
        
        logger.info(f"Running in coordinator mode with URL: {self.coordinator_url}")
        
        # Register with coordinator
        register_url = f"{self.coordinator_url}/register"
        register_data = {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "capabilities": {
                "mlx": self.node_type == "mlx",
                "cuda": self.node_type == "cuda" and self.cuda_available,
                "cuda_devices": self._get_cuda_device_count() if self.node_type == "cuda" else 0
            }
        }
        
        try:
            response = requests.post(register_url, json=register_data)
            if response.status_code != 200:
                logger.error(f"Failed to register with coordinator: {response.text}")
                return
            logger.info("Successfully registered with coordinator")
        except Exception as e:
            logger.error(f"Error registering with coordinator: {e}")
            return
        
        # Main task loop
        while self.is_running:
            try:
                # Poll for tasks
                task_url = f"{self.coordinator_url}/get_task"
                response = requests.post(task_url, json={"node_id": self.node_id})
                
                if response.status_code == 200:
                    task = response.json()
                    
                    if task.get("task_type") == "shutdown":
                        logger.info("Received shutdown command from coordinator")
                        break
                    
                    # Process task
                    result = self._process_task(task)
                    
                    # Send result back to coordinator
                    result_url = f"{self.coordinator_url}/submit_result"
                    result_data = {
                        "node_id": self.node_id,
                        "task_id": task.get("task_id"),
                        "result": result
                    }
                    
                    requests.post(result_url, json=result_data)
                else:
                    # No task available, wait before polling again
                    time.sleep(5)
                    
                # Send heartbeat
                heartbeat_url = f"{self.coordinator_url}/heartbeat"
                requests.post(heartbeat_url, json={
                    "node_id": self.node_id,
                    "status": self.status,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                logger.error(f"Error in coordinator mode: {e}")
                time.sleep(10)  # Wait before retrying after error
    
    def _run_standalone_mode(self):
        """Run in standalone mode - process local tasks"""
        logger.info("Running in standalone mode")
        
        # Process command line tasks
        parser = argparse.ArgumentParser(description="Hybrid worker in standalone mode")
        parser.add_argument("--task", type=str, required=True, help="Task specification (JSON)")
        parser.add_argument("--output", type=str, help="Output file")
        args = parser.parse_args()
        
        try:
            # Parse task
            if os.path.exists(args.task):
                with open(args.task, 'r') as f:
                    task = json.load(f)
            else:
                task = json.loads(args.task)
            
            # Process task
            result = self._process_task(task)
            
            # Write result
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                print(json.dumps(result, indent=2))
            
        except Exception as e:
            logger.error(f"Error processing task: {e}", exc_info=True)
            sys.exit(1)
    
    def _process_task(self, task):
        """
        Process a task.
        
        Args:
            task: Task specification
            
        Returns:
            Task result
        """
        task_type = task.get("task_type")
        logger.info(f"Processing task: {task_type}")
        
        if task_type == "forward":
            return self._forward_task(task)
        elif task_type == "backward":
            return self._backward_task(task)
        elif task_type == "forward_backward":
            return self._forward_backward_task(task)
        elif task_type == "parameter_update":
            return self._parameter_update_task(task)
        elif task_type == "status":
            return self._status_task(task)
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return {"error": f"Unknown task type: {task_type}"}
    
    def _forward_task(self, task):
        """Process forward pass task"""
        if self.node_type == "mlx":
            return self._forward_mlx(task)
        else:
            return self._forward_cuda(task)
    
    def _backward_task(self, task):
        """Process backward pass task"""
        if self.node_type == "mlx":
            return self._backward_mlx(task)
        else:
            return self._backward_cuda(task)
    
    def _forward_backward_task(self, task):
        """Process combined forward/backward pass task"""
        if self.node_type == "mlx":
            return self._forward_backward_mlx(task)
        else:
            return self._forward_backward_cuda(task)
    
    def _parameter_update_task(self, task):
        """Process parameter update task"""
        if self.node_type == "mlx":
            return self._parameter_update_mlx(task)
        else:
            return self._parameter_update_cuda(task)
    
    def _status_task(self, task):
        """Process status query task"""
        status = {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "status": self.status,
            "timestamp": time.time()
        }
        
        if self.node_type == "cuda":
            # Add CUDA-specific info
            import torch
            status["cuda"] = {
                "device_count": torch.cuda.device_count(),
                "devices": [
                    {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_memory,
                        "memory_used": torch.cuda.memory_allocated(i)
                    }
                    for i in range(torch.cuda.device_count())
                ]
            }
        elif self.node_type == "mlx":
            # Add MLX-specific info
            import mlx.core as mx
            device = mx.default_device()
            status["mlx"] = {
                "device": str(device),
                "is_gpu": isinstance(device, mx.gpu)
            }
        
        return status
    
    def _forward_mlx(self, task):
        """Perform forward pass with MLX"""
        import mlx.core as mx
        import mlx.nn as nn
        
        # Extract inputs
        input_data = task.get("inputs", [])
        model_config = task.get("model_config", {})
        weights = task.get("weights", {})
        
        # Convert inputs to MLX array
        inputs = mx.array(np.array(input_data))
        
        # This is a simplified implementation - real implementation would
        # load the actual model architecture based on the config
        logger.info(f"MLX forward pass with input shape: {inputs.shape}")
        
        # Simulate model forward pass
        # In a real implementation, you would:
        # 1. Load the model based on config
        # 2. Apply the weights
        # 3. Run the actual forward pass
        
        # Simple dummy computation for now
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1] if len(inputs.shape) > 1 else 1
        hidden_size = model_config.get("dimensions", {}).get("hidden_size", 768)
        
        # Simulate output with random data
        logits = mx.random.normal((batch_size, seq_len, hidden_size))
        
        # Convert output to numpy for serialization
        output = logits.numpy().tolist()
        
        return {
            "status": "success",
            "output": output,
            "batch_size": batch_size,
            "seq_length": seq_len,
            "hidden_size": hidden_size
        }
    
    def _backward_mlx(self, task):
        """Perform backward pass with MLX"""
        import mlx.core as mx
        import mlx.nn as nn
        
        # Extract data
        output_data = task.get("output", [])
        target_data = task.get("targets", [])
        model_config = task.get("model_config", {})
        
        # Convert to MLX arrays
        output = mx.array(np.array(output_data))
        targets = mx.array(np.array(target_data))
        
        logger.info(f"MLX backward pass with output shape: {output.shape}")
        
        # Simulate loss computation
        loss = nn.losses.cross_entropy(output, targets)
        
        # Simulate gradient computation
        # In a real implementation, you would:
        # 1. Compute the actual gradients with respect to model parameters
        # 2. Return those gradients
        
        # Dummy gradients for now
        hidden_size = model_config.get("dimensions", {}).get("hidden_size", 768)
        gradients = {
            "weight": mx.random.normal((hidden_size, hidden_size)).numpy().tolist(),
            "bias": mx.random.normal((hidden_size,)).numpy().tolist()
        }
        
        return {
            "status": "success",
            "loss": float(loss.item()),
            "gradients": gradients
        }
    
    def _forward_backward_mlx(self, task):
        """Perform combined forward/backward pass with MLX"""
        import mlx.core as mx
        import mlx.nn as nn
        
        # Extract data
        input_data = task.get("inputs", [])
        target_data = task.get("targets", [])
        model_config = task.get("model_config", {})
        weights = task.get("weights", {})
        
        # Convert to MLX arrays
        inputs = mx.array(np.array(input_data))
        targets = mx.array(np.array(target_data))
        
        logger.info(f"MLX forward/backward pass with input shape: {inputs.shape}")
        
        # Simulate model and loss computation
        # In a real implementation, this would be the actual model forward pass
        # followed by loss computation and backward pass
        
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1] if len(inputs.shape) > 1 else 1
        tokens = batch_size * seq_len
        
        # Simulate loss with random value
        loss = mx.random.normal(()).numpy().item()
        
        # Simulate gradients
        hidden_size = model_config.get("dimensions", {}).get("hidden_size", 768)
        gradients = {
            "weight": mx.random.normal((hidden_size, hidden_size)).numpy().tolist(),
            "bias": mx.random.normal((hidden_size,)).numpy().tolist()
        }
        
        return {
            "status": "success",
            "loss": float(loss),
            "tokens": int(tokens),
            "gradients": gradients
        }
    
    def _parameter_update_mlx(self, task):
        """Perform parameter update with MLX"""
        import mlx.core as mx
        
        # Extract data
        model_state = task.get("model_state", {})
        gradients = task.get("gradients", {})
        optimizer_config = task.get("optimizer_config", {})
        
        logger.info("MLX parameter update")
        
        # Simulate parameter update
        # In a real implementation, this would use the MLX optimizer
        # to update the actual model parameters
        
        # Simple SGD update simulation
        learning_rate = optimizer_config.get("learning_rate", 0.01)
        updated_model = {}
        
        for name, param_data in model_state.items():
            if name in gradients:
                param = mx.array(np.array(param_data))
                grad = mx.array(np.array(gradients[name]))
                param = param - learning_rate * grad
                updated_model[name] = param.numpy().tolist()
            else:
                updated_model[name] = param_data
        
        return {
            "status": "success",
            "updated_model": updated_model
        }
    
    def _forward_cuda(self, task):
        """Perform forward pass with CUDA"""
        import torch
        
        # Extract inputs
        input_data = task.get("inputs", [])
        model_config = task.get("model_config", {})
        weights = task.get("weights", {})
        
        # Convert inputs to PyTorch tensor
        inputs = torch.tensor(np.array(input_data)).to("cuda")
        
        logger.info(f"CUDA forward pass with input shape: {inputs.shape}")
        
        # Simulate model forward pass
        # In a real implementation, you would:
        # 1. Load the model based on config
        # 2. Apply the weights
        # 3. Run the actual forward pass
        
        # Simple dummy computation for now
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1] if len(inputs.shape) > 1 else 1
        hidden_size = model_config.get("dimensions", {}).get("hidden_size", 768)
        
        # Simulate output with random data
        logits = torch.randn((batch_size, seq_len, hidden_size), device="cuda")
        
        # Convert output to CPU for serialization
        output = logits.cpu().numpy().tolist()
        
        return {
            "status": "success",
            "output": output,
            "batch_size": batch_size,
            "seq_length": seq_len,
            "hidden_size": hidden_size
        }
    
    def _backward_cuda(self, task):
        """Perform backward pass with CUDA"""
        import torch
        
        # Extract data
        output_data = task.get("output", [])
        target_data = task.get("targets", [])
        model_config = task.get("model_config", {})
        
        # Convert to PyTorch tensors
        output = torch.tensor(np.array(output_data), device="cuda")
        targets = torch.tensor(np.array(target_data), device="cuda")
        
        logger.info(f"CUDA backward pass with output shape: {output.shape}")
        
        # Simulate loss computation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output.reshape(-1, output.size(-1)), targets.reshape(-1))
        
        # Simulate gradient computation
        # In a real implementation, you would:
        # 1. Compute the actual gradients with respect to model parameters
        # 2. Return those gradients
        
        # Dummy gradients for now
        hidden_size = model_config.get("dimensions", {}).get("hidden_size", 768)
        gradients = {
            "weight": torch.randn((hidden_size, hidden_size), device="cuda").cpu().numpy().tolist(),
            "bias": torch.randn((hidden_size,), device="cuda").cpu().numpy().tolist()
        }
        
        return {
            "status": "success",
            "loss": float(loss.item()),
            "gradients": gradients
        }
    
    def _forward_backward_cuda(self, task):
        """Perform combined forward/backward pass with CUDA"""
        import torch
        
        # Extract data
        input_data = task.get("inputs", [])
        target_data = task.get("targets", [])
        model_config = task.get("model_config", {})
        weights = task.get("weights", {})
        
        # Convert to PyTorch tensors
        inputs = torch.tensor(np.array(input_data), device="cuda")
        targets = torch.tensor(np.array(target_data), device="cuda")
        
        logger.info(f"CUDA forward/backward pass with input shape: {inputs.shape}")
        
        # Simulate model and loss computation
        # In a real implementation, this would be the actual model forward pass
        # followed by loss computation and backward pass
        
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1] if len(inputs.shape) > 1 else 1
        tokens = batch_size * seq_len
        
        # Simulate loss with random value
        loss = torch.randn((), device="cuda").item()
        
        # Simulate gradients
        hidden_size = model_config.get("dimensions", {}).get("hidden_size", 768)
        gradients = {
            "weight": torch.randn((hidden_size, hidden_size), device="cuda").cpu().numpy().tolist(),
            "bias": torch.randn((hidden_size,), device="cuda").cpu().numpy().tolist()
        }
        
        return {
            "status": "success",
            "loss": float(loss),
            "tokens": int(tokens),
            "gradients": gradients
        }
    
    def _parameter_update_cuda(self, task):
        """Perform parameter update with CUDA"""
        import torch
        
        # Extract data
        model_state = task.get("model_state", {})
        gradients = task.get("gradients", {})
        optimizer_config = task.get("optimizer_config", {})
        
        logger.info("CUDA parameter update")
        
        # Simulate parameter update
        # In a real implementation, this would use PyTorch optimizer
        # to update the actual model parameters
        
        # Simple SGD update simulation
        learning_rate = optimizer_config.get("learning_rate", 0.01)
        updated_model = {}
        
        for name, param_data in model_state.items():
            param_tensor = torch.tensor(np.array(param_data), device="cuda")
            
            if name in gradients:
                grad_tensor = torch.tensor(np.array(gradients[name]), device="cuda")
                param_tensor = param_tensor - learning_rate * grad_tensor
            
            updated_model[name] = param_tensor.cpu().numpy().tolist()
        
        return {
            "status": "success",
            "updated_model": updated_model
        }
    
    def _get_cuda_device_count(self):
        """Get number of CUDA devices"""
        if not hasattr(self, "cuda_available") or not self.cuda_available:
            return 0
        
        import torch
        return torch.cuda.device_count()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hybrid distributed worker")
    parser.add_argument("--node-type", type=str, choices=["mlx", "cuda"], 
                       default="mlx", help="Node type")
    parser.add_argument("--node-id", type=int, default=0, help="Node ID")
    parser.add_argument("--coordinator", type=str, default=None, 
                       help="Coordinator URL")
    args = parser.parse_args()
    
    # Initialize worker
    try:
        worker = HybridWorker(
            node_type=args.node_type,
            node_id=args.node_id,
            coordinator_url=args.coordinator
        )
        
        # Start worker
        worker.start()
        
    except Exception as e:
        logger.error(f"Error initializing worker: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()