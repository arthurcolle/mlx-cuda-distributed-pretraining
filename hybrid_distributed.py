#!/usr/bin/env python
# MLX/CUDA hybrid distributed training
# Coordinates training across MLX devices and remote CUDA nodes

import os
import argparse
import yaml
import time
import json
import threading
import queue
import uuid
from pathlib import Path
import mlx.core as mx
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define Modal remote compute function at module level for proper decorating
def remote_compute_function(operation, payload):
    """Remote compute function for Modal workers"""
    import torch
    import numpy as np
    import json
    
    if operation == "forward_backward":
        # Extract data
        model_state = payload["model_state"]
        inputs = np.array(payload["inputs"])
        targets = np.array(payload["targets"])
        
        # Create PyTorch model (this would need actual model implementation)
        # For now, we'll simulate the computation
        batch_size = inputs.shape[0]
        tokens = batch_size * (inputs.shape[1] if len(inputs.shape) > 1 else 1)
        
        # Simulate computation
        loss = 1.0  # Mock loss value
        
        # Create mock gradients
        gradients = {}
        for name, param_data in model_state.items():
            param_shape = np.array(param_data).shape
            gradients[name] = np.random.randn(*param_shape).tolist()
        
        return {
            "loss": loss,
            "tokens": tokens,
            "gradients": {"type": "gradients", "data": gradients}
        }
        
    elif operation == "update_parameters":
        # Extract data
        model_state = payload["model_state"]
        gradients = payload["gradients"]
        
        # Simulate parameter update
        updated_model = {}
        for name, param_data in model_state.items():
            param = np.array(param_data)
            if name in gradients:
                grad = np.array(gradients[name])
                # Simple SGD update
                param -= 0.01 * grad
            updated_model[name] = param.tolist()
        
        return {"updated_model": updated_model}
        
    else:
        return {"error": f"Unsupported operation: {operation}"}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hybrid_training.log")
    ]
)
logger = logging.getLogger("hybrid_distributed")


class HybridDeviceManager:
    """
    Manages computation across both local MLX devices and remote CUDA nodes.
    Coordinates gradient aggregation and parameter updates between devices.
    """
    def __init__(self, config, mlx_devices=None, remote_workers=None):
        """
        Initialize the hybrid device manager.
        
        Args:
            config: Training configuration
            mlx_devices: List of local MLX devices (e.g., ["gpu", "cpu"])
            remote_workers: List of remote worker configurations
        """
        self.config = config
        self.mlx_devices = mlx_devices or ["gpu"]
        self.remote_workers = remote_workers or []
        self.device_queues = {}
        self.workers = {}
        self.remote_connections = {}
        self.aggregation_queue = queue.Queue()
        
        # Initialize device queues for MLX devices
        for device in self.mlx_devices:
            self.device_queues[device] = queue.Queue()
        
        # Initialize remote connections
        self._init_remote_connections()
        
        # Start aggregation worker
        self.aggregation_thread = threading.Thread(
            target=self._aggregation_worker,
            daemon=True
        )
        self.aggregation_thread.start()
        
        logger.info(f"Hybrid device manager initialized with {len(self.mlx_devices)} MLX devices and {len(self.remote_workers)} remote workers")
    
    def _init_remote_connections(self):
        """Initialize connections to remote CUDA workers"""
        for i, worker_config in enumerate(self.remote_workers):
            worker_id = worker_config.get("id", f"remote_{i}")
            worker_type = worker_config.get("type", "modal")
            
            if worker_type == "modal":
                # For Modal workers, we'll use the Modal Python SDK
                try:
                    from modal_connector import ModalConnector
                    connector = ModalConnector(
                        worker_config.get("app_name", "mlx-hybrid-training"),
                        worker_config.get("gpu_count", 2),
                        worker_config.get("gpu_type", "A100")
                    )
                    self.remote_connections[worker_id] = connector
                    # Add queue for this worker
                    self.device_queues[worker_id] = queue.Queue()
                    logger.info(f"Initialized Modal connector for worker {worker_id}")
                except ImportError:
                    logger.error("Modal SDK not installed. Cannot use Modal workers.")
            elif worker_type == "custom":
                # Custom remote worker with HTTP API
                from remote_connector import RemoteConnector
                connector = RemoteConnector(
                    worker_config.get("endpoint_url"),
                    worker_config.get("api_key", "")
                )
                self.remote_connections[worker_id] = connector
                # Add queue for this worker
                self.device_queues[worker_id] = queue.Queue()
                logger.info(f"Initialized custom connector for worker {worker_id}")
            else:
                logger.warning(f"Unsupported worker type: {worker_type}")
    
    def start_workers(self):
        """Start worker threads for each device"""
        # Start MLX device workers
        for device_name in self.mlx_devices:
            worker = threading.Thread(
                target=self._mlx_worker,
                args=(device_name, self.device_queues[device_name]),
                daemon=True
            )
            worker.start()
            self.workers[device_name] = worker
        
        # Start remote worker threads
        for worker_id in self.remote_connections:
            worker = threading.Thread(
                target=self._remote_worker,
                args=(worker_id, self.device_queues[worker_id]),
                daemon=True
            )
            worker.start()
            self.workers[worker_id] = worker
        
        logger.info(f"Started {len(self.workers)} worker threads")
    
    def _mlx_worker(self, device_name, device_queue):
        """Worker thread for MLX devices"""
        while True:
            task = device_queue.get()
            if task is None:  # Poison pill to stop the thread
                break
                
            func, args, kwargs, result_queue = task
            
            # Set device for this computation
            old_device = mx.default_device()
            if device_name == "gpu":
                mx.set_default_device(mx.gpu)
            elif device_name == "cpu":
                mx.set_default_device(mx.cpu)
            elif device_name == "mlx":
                mx.set_default_device(mx.gpu)
            else:
                raise ValueError(f"Unsupported MLX device: {device_name}")
            
            try:
                result = func(*args, **kwargs)
                result_queue.put((True, result))
            except Exception as e:
                logger.error(f"Error in MLX worker ({device_name}): {e}")
                result_queue.put((False, e))
            finally:
                # Restore previous device
                mx.set_default_device(old_device)
                # Clear cache if needed
                if device_name == "gpu":
                    mx.clear_cache()
                device_queue.task_done()
    
    def _remote_worker(self, worker_id, device_queue):
        """Worker thread for remote CUDA workers"""
        connector = self.remote_connections[worker_id]
        
        while True:
            task = device_queue.get()
            if task is None:  # Poison pill to stop the thread
                break
                
            func, args, kwargs, result_queue = task
            
            try:
                # Handle different task types for remote workers
                if kwargs.get('_task_type') == 'forward_backward':
                    # Extract the model, inputs, and targets
                    model = args[0]
                    inputs = args[1]
                    targets = args[2]
                    
                    # Send computation request to the remote worker
                    result = connector.compute_forward_backward(
                        model_state=self._serialize_model_state(model),
                        inputs=self._serialize_tensor(inputs),
                        targets=self._serialize_tensor(targets)
                    )
                    
                    # Process the result
                    processed_result = self._process_remote_result(result)
                    result_queue.put((True, processed_result))
                elif kwargs.get('_task_type') == 'parameter_update':
                    # Extract the model and gradients
                    model = args[0]
                    gradients = args[1]
                    
                    # Send parameter update request to the remote worker
                    result = connector.update_parameters(
                        model_state=self._serialize_model_state(model),
                        gradients=self._serialize_gradients(gradients)
                    )
                    
                    # Process the result
                    processed_result = self._process_remote_result(result)
                    result_queue.put((True, processed_result))
                else:
                    # Generic function execution on remote worker
                    # This might not be directly supported in many cases
                    logger.warning(f"Generic function execution on remote worker not fully supported: {func.__name__}")
                    result = connector.execute_function(
                        function_name=func.__name__,
                        args=self._serialize_args(args),
                        kwargs=self._serialize_kwargs(kwargs)
                    )
                    processed_result = self._process_remote_result(result)
                    result_queue.put((True, processed_result))
                    
            except Exception as e:
                logger.error(f"Error in remote worker ({worker_id}): {e}")
                result_queue.put((False, e))
            finally:
                device_queue.task_done()
    
    def _aggregation_worker(self):
        """Worker thread for gradient aggregation"""
        while True:
            task = self.aggregation_queue.get()
            if task is None:  # Poison pill to stop the thread
                break
                
            gradients_list, result_queue = task
            
            try:
                # Aggregate gradients from all workers
                aggregated_gradients = self._aggregate_gradients(gradients_list)
                result_queue.put((True, aggregated_gradients))
            except Exception as e:
                logger.error(f"Error in aggregation worker: {e}")
                result_queue.put((False, e))
            finally:
                self.aggregation_queue.task_done()
    
    def _aggregate_gradients(self, gradients_list):
        """
        Aggregate gradients from multiple workers.
        
        Args:
            gradients_list: List of gradient dictionaries
            
        Returns:
            Aggregated gradients
        """
        if not gradients_list:
            return {}
        
        # For simplicity, let's average the gradients
        aggregated = {}
        for key in gradients_list[0].keys():
            # Sum gradients for each parameter
            grad_sum = None
            for grads in gradients_list:
                if key in grads:
                    # Handle the case where grads[key] might be a dictionary or complex object
                    if isinstance(grads[key], dict):
                        # If it's a dictionary, we need to handle it differently
                        if grad_sum is None:
                            grad_sum = grads[key].copy()  # Make a copy to avoid modifying the original
                        else:
                            # We need to handle nested dictionaries separately
                            for nested_key in grads[key]:
                                if nested_key in grad_sum:
                                    grad_sum[nested_key] += grads[key][nested_key]
                                else:
                                    grad_sum[nested_key] = grads[key][nested_key]
                    else:
                        # Normal case for tensor values
                        if grad_sum is None:
                            grad_sum = grads[key]
                        else:
                            grad_sum += grads[key]
            
            # Average gradients
            if grad_sum is not None:
                if isinstance(grad_sum, dict):
                    # For dictionary gradients, divide each value
                    for nested_key in grad_sum:
                        if isinstance(grad_sum[nested_key], (int, float, np.ndarray, mx.array)):
                            grad_sum[nested_key] = grad_sum[nested_key] / len(gradients_list)
                    aggregated[key] = grad_sum
                else:
                    # For tensor gradients
                    aggregated[key] = grad_sum / len(gradients_list)
        
        return aggregated
    
    def _serialize_model_state(self, model):
        """Serialize model state for transmission to remote workers"""
        # This is a simplified implementation - in practice, you'd want
        # more efficient serialization mechanisms
        model_state = {}
        for name, param in zip(model.parameter_names(), model.parameters()):
            # Convert MLX arrays to numpy for serialization
            model_state[name] = param.numpy().tolist()
        return model_state
    
    def _serialize_tensor(self, tensor):
        """Serialize an MLX tensor for transmission"""
        if isinstance(tensor, mx.array):
            return tensor.numpy().tolist()
        return tensor
    
    def _serialize_gradients(self, gradients):
        """Serialize gradients for transmission"""
        serialized_grads = {}
        for key, grad in gradients.items():
            if isinstance(grad, mx.array):
                serialized_grads[key] = grad.numpy().tolist()
            else:
                serialized_grads[key] = grad
        return serialized_grads
    
    def _serialize_args(self, args):
        """Serialize function arguments for transmission"""
        serialized_args = []
        for arg in args:
            if isinstance(arg, mx.array):
                serialized_args.append({"type": "mx_array", "data": arg.numpy().tolist()})
            else:
                serialized_args.append(arg)
        return serialized_args
    
    def _serialize_kwargs(self, kwargs):
        """Serialize function keyword arguments for transmission"""
        serialized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, mx.array):
                serialized_kwargs[key] = {"type": "mx_array", "data": value.numpy().tolist()}
            else:
                serialized_kwargs[key] = value
        return serialized_kwargs
    
    def _process_remote_result(self, result):
        """Process result from remote worker"""
        if isinstance(result, dict) and result.get("type") == "mx_array":
            # Convert back to mx.array
            return mx.array(np.array(result["data"]))
        elif isinstance(result, dict) and result.get("type") == "gradients":
            # Convert gradients back to mx.array
            processed_grads = {}
            for key, grad_data in result["data"].items():
                processed_grads[key] = mx.array(np.array(grad_data))
            return processed_grads
        return result
    
    def run_on_device(self, device_name, func, *args, **kwargs):
        """Run a function on a specific device"""
        result_queue = queue.Queue()
        self.device_queues[device_name].put((func, args, kwargs, result_queue))
        success, result = result_queue.get()
        if success:
            return result
        else:
            raise result  # Re-raise the exception
    
    def distribute_batch(self, batch):
        """
        Distribute a batch of data across devices.
        
        Args:
            batch: Input batch
            
        Returns:
            List of (device, sub_batch) pairs
        """
        devices = list(self.device_queues.keys())
        batch_size = batch.shape[0]
        split_size = batch_size // len(devices)
        
        # Distribute batch across devices
        distributed_batches = []
        for i, device in enumerate(devices):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < len(devices) - 1 else batch_size
            sub_batch = batch[start_idx:end_idx]
            distributed_batches.append((device, sub_batch))
        
        return distributed_batches
    
    def parallel_forward_backward(self, model, inputs, targets):
        """
        Perform forward and backward passes in parallel across all devices.
        
        Args:
            model: Model to use
            inputs: Input batch
            targets: Target batch
            
        Returns:
            (loss, gradients) tuple
        """
        # Distribute inputs and targets
        batch_size = inputs.shape[0]
        distributed_inputs = self.distribute_batch(inputs)
        distributed_targets = self.distribute_batch(targets)
        
        # Submit tasks to all devices
        result_queues = []
        for (device, sub_inputs), (_, sub_targets) in zip(distributed_inputs, distributed_targets):
            result_queue = queue.Queue()
            
            if device in self.mlx_devices:
                # For MLX devices, use standard forward/backward
                self.device_queues[device].put((
                    self._compute_forward_backward,
                    (model, sub_inputs, sub_targets),
                    {},
                    result_queue
                ))
            else:
                # For remote workers, use special task type
                self.device_queues[device].put((
                    self._compute_forward_backward,
                    (model, sub_inputs, sub_targets),
                    {"_task_type": "forward_backward"},
                    result_queue
                ))
            
            result_queues.append(result_queue)
        
        # Collect results
        losses = []
        gradients_list = []
        tokens = 0
        
        for q in result_queues:
            success, result = q.get()
            if success:
                sub_loss, sub_tokens, sub_gradients = result
                losses.append(sub_loss * sub_tokens)  # Weight by token count
                gradients_list.append(sub_gradients)
                tokens += sub_tokens
            else:
                raise result  # Re-raise the exception
        
        # Aggregate gradients
        aggregation_result_queue = queue.Queue()
        self.aggregation_queue.put((gradients_list, aggregation_result_queue))
        success, aggregated_gradients = aggregation_result_queue.get()
        
        if not success:
            raise aggregated_gradients  # Re-raise exception
        
        # Compute average loss
        total_loss = sum(losses)
        avg_loss = total_loss / tokens if tokens > 0 else 0
        
        return avg_loss, tokens, aggregated_gradients
    
    def _compute_forward_backward(self, model, inputs, targets):
        """
        Compute forward and backward passes for a sub-batch.
        
        Args:
            model: Model to use
            inputs: Input sub-batch
            targets: Target sub-batch
            
        Returns:
            (loss, tokens, gradients) tuple
        """
        import mlx.nn as nn
        import mlx.core as mx
        
        # Compute forward pass
        logits = model(inputs)
        logits = logits.astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, targets)
        
        # Mask padding tokens (assuming padding token is defined in the model)
        pad_token = model.tokenizer.PAD_TOKEN if hasattr(model, "tokenizer") else 0
        pad_mask = (targets != pad_token)
        loss = loss * pad_mask
        ntoks = pad_mask.sum()
        
        # Compute backward pass
        grad_fn = mx.grad(lambda m, x, y: nn.losses.cross_entropy(m(x), y).sum())
        gradients = grad_fn(model, inputs, targets)
        
        # Apply token masking to gradients
        # This is a simplified approach - more complex handling might be needed
        
        return loss.sum() / ntoks, ntoks, gradients
    
    def stop_workers(self):
        """Stop all worker threads"""
        # Stop device workers
        for device_queue in self.device_queues.values():
            device_queue.put(None)  # Poison pill
        
        # Stop aggregation worker
        self.aggregation_queue.put(None)  # Poison pill
        
        # Join all threads
        for worker in self.workers.values():
            worker.join()
        
        self.aggregation_thread.join()
        
        logger.info("All workers stopped")


class HybridDistributedOptimizer:
    """
    Optimizer that distributes computation across MLX and CUDA devices.
    """
    def __init__(self, optimizer, device_manager):
        """
        Initialize the hybrid distributed optimizer.
        
        Args:
            optimizer: Base optimizer to use
            device_manager: HybridDeviceManager instance
        """
        self.optimizer = optimizer
        self.device_manager = device_manager
        self.state = optimizer.state
    
    def update(self, model, gradients):
        """
        Update model parameters using gradients.
        
        Args:
            model: Model to update
            gradients: Gradients to apply
            
        Returns:
            Updated model
        """
        # Distribute large parameter updates to remote workers
        # For now, we'll just use the base optimizer
        return self.optimizer.update(model, gradients)
    
    def __getattr__(self, name):
        """Delegate all other methods to the wrapped optimizer"""
        return getattr(self.optimizer, name)


class ModalConnector:
    """
    Connector for Modal.com remote workers.
    """
    def __init__(self, app_name="mlx-hybrid-training", gpu_count=2, gpu_type="A100"):
        """
        Initialize Modal connector.
        
        Args:
            app_name: Name of the Modal application
            gpu_count: Number of GPUs to use
            gpu_type: Type of GPU to use
        """
        self.app_name = app_name
        self.gpu_count = gpu_count
        self.gpu_type = gpu_type
        self.client = None
        self.modal_function = None
        
        try:
            import modal
            self.modal = modal
            self.initialize_modal()
        except ImportError:
            logger.error("Modal SDK not installed. Cannot use Modal workers.")
            self.modal = None
    
    def initialize_modal(self):
        """Initialize Modal client and function"""
        if self.modal is None:
            return
        
        # Create Modal app
        self.client = self.modal.App(self.app_name)
        
        # Create container image
        image = self.modal.Image.debian_slim().pip_install(
            "numpy>=1.24.0",
            "torch>=2.0.0",
            "mlx>=0.0.1",
            "mlx_lm>=0.0.1",
            "tqdm>=4.66.0"
        )
        
        # Instead of defining the function inline, we'll define it at module level and use it here
        self.modal_function = self.client.function(
            image=image,
            gpu=f"{self.gpu_type}:{self.gpu_count}",
            timeout=3600,  # 1 hour max runtime,
            serialize=True  # Allow serialization of non-global functions
        )(remote_compute_function)  # This is now defined at module level
    
    def compute_forward_backward(self, model_state, inputs, targets):
        """Compute forward and backward passes on the remote worker"""
        if self.modal is None or self.modal_function is None:
            raise RuntimeError("Modal SDK not initialized")
        
        payload = {
            "model_state": model_state,
            "inputs": inputs,
            "targets": targets
        }
        
        with self.client.run():
            result = self.modal_function.remote("forward_backward", payload)
        
        return result
    
    def update_parameters(self, model_state, gradients):
        """Update parameters on the remote worker"""
        if self.modal is None or self.modal_function is None:
            raise RuntimeError("Modal SDK not initialized")
        
        payload = {
            "model_state": model_state,
            "gradients": gradients
        }
        
        with self.client.run():
            result = self.modal_function.remote("update_parameters", payload)
        
        return result
    
    def execute_function(self, function_name, args, kwargs):
        """Execute a generic function on the remote worker"""
        if self.modal is None or self.modal_function is None:
            raise RuntimeError("Modal SDK not initialized")
        
        payload = {
            "function_name": function_name,
            "args": args,
            "kwargs": kwargs
        }
        
        with self.client.run():
            result = self.modal_function.remote("execute_function", payload)
        
        return result


class RemoteConnector:
    """
    Connector for custom remote workers with HTTP API.
    """
    def __init__(self, endpoint_url, api_key=""):
        """
        Initialize remote connector.
        
        Args:
            endpoint_url: URL of the remote worker API
            api_key: API key for authentication
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            logger.error("Requests library not installed. Cannot use custom remote workers.")
            self.requests = None
    
    def _make_request(self, operation, payload):
        """Make HTTP request to remote worker"""
        if self.requests is None:
            raise RuntimeError("Requests library not installed")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        url = f"{self.endpoint_url}/{operation}"
        response = self.requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise RuntimeError(f"Error from remote worker: {response.text}")
        
        return response.json()
    
    def compute_forward_backward(self, model_state, inputs, targets):
        """Compute forward and backward passes on the remote worker"""
        payload = {
            "model_state": model_state,
            "inputs": inputs,
            "targets": targets
        }
        
        return self._make_request("forward_backward", payload)
    
    def update_parameters(self, model_state, gradients):
        """Update parameters on the remote worker"""
        payload = {
            "model_state": model_state,
            "gradients": gradients
        }
        
        return self._make_request("update_parameters", payload)
    
    def execute_function(self, function_name, args, kwargs):
        """Execute a generic function on the remote worker"""
        payload = {
            "function_name": function_name,
            "args": args,
            "kwargs": kwargs
        }
        
        return self._make_request("execute_function", payload)


def create_hybrid_trainer(config_path, remote_workers=None, config_overrides=None):
    """
    Create a hybrid trainer from configuration.
    
    Args:
        config_path: Path to configuration file
        remote_workers: List of remote worker configurations
        config_overrides: Optional dictionary of configuration overrides
        
    Returns:
        Tuple of (model, optimizer, device_manager, tokenizer, config)
    """
    from train import Config, TokenizerManager, filter_valid_args, OptimizationManager
    import importlib
    
    # Load configuration
    config = Config.from_yaml(config_path)
    
    # Apply configuration overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            parts = key.split('.')
            target = config
            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    logger.warning(f"Config override path {key} not found")
                    break
            else:
                setattr(target, parts[-1], value)
                logger.debug(f"Applied config override: {key} = {value}")
    
    # Set up system
    random.seed(config.system.seed)
    np.random.seed(config.system.seed)
    mx.random.seed(config.system.seed)
    
    # Set up MLX devices
    mlx_devices = config.system.devices if config.system.devices else ["gpu"]
    
    # Set up device manager
    device_manager = HybridDeviceManager(
        config=config,
        mlx_devices=mlx_devices,
        remote_workers=remote_workers
    )
    device_manager.start_workers()
    
    # Set up tokenizer
    tokenizer = TokenizerManager(config.data)
    
    # Set up model
    model_cfg = config.model
    arch_file = f"arch.{model_cfg.architecture}"
    mlx_lm_file = f"mlx_lm.models.{model_cfg.architecture}"
    Model = None
    ModelArgs = None
    
    # Try importing from mlx_lm directly - it should be installed based on pip list
    try:
        from mlx_lm.models.llama import Model, ModelArgs
        logger.info(f"Imported Model and ModelArgs from mlx_lm.models.llama")
    except ImportError:
        # Fall back to trying other import paths
        try:
            module = importlib.import_module(arch_file)
            Model = getattr(module, 'Model')
            ModelArgs = getattr(module, 'ModelArgs')
            logger.info(f"Imported Model and ModelArgs from {arch_file}")
        except ImportError:
            try:
                module = importlib.import_module(mlx_lm_file)
                Model = getattr(module, 'Model')
                ModelArgs = getattr(module, 'ModelArgs')
                logger.info(f"Imported Model and ModelArgs from {mlx_lm_file}")
            except ImportError:
                raise ImportError(f"Model architecture '{model_cfg.architecture}' not found")
    
    if Model is None or ModelArgs is None:
        raise ImportError(f"Could not import Model or ModelArgs for architecture '{model_cfg.architecture}'")
    
    # Prepare model arguments
    all_args = {
        'model_type': model_cfg.architecture,
        'hidden_size': model_cfg.dimensions['hidden_size'],
        'num_hidden_layers': model_cfg.dimensions.get('num_layers', 8),
        'intermediate_size': model_cfg.dimensions['intermediate_size'],
        'num_attention_heads': model_cfg.attention['num_heads'],
        'rms_norm_eps': model_cfg.normalization['rms_norm_eps'],
        'vocab_size': tokenizer.VOCAB_SIZE,
        'head_dim': model_cfg.attention['head_dim'],
        'max_position_embeddings': model_cfg.attention['max_position_embeddings'],
        'num_key_value_heads': model_cfg.attention['num_kv_heads'],
        'attention_bias': model_cfg.misc['attention_bias'],
        'mlp_bias': model_cfg.misc['mlp_bias'],
        'rope_theta': model_cfg.rope['theta'],
        'rope_traditional': model_cfg.rope['traditional'],
        'rope_scaling': model_cfg.rope['scaling'],
        'tie_word_embeddings': model_cfg.misc['tie_word_embeddings'],
    }
    valid_args = filter_valid_args(ModelArgs, all_args)
    args = ModelArgs(**valid_args)
    
    # Create model
    model = Model(args)
    
    # Load weights if specified
    if config.data.weight_path is not None:
        logger.info(f"Loading weights from {config.data.weight_path}")
        model.load_weights(config.data.weight_path, strict=False)
    
    # Log model size
    from mlx.utils import tree_flatten
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    logger.info(f"Model has {p:.2f}M parameters")
    
    # Set up optimizer
    num_samples = 1000  # This should be replaced with actual data size
    batch_size = config.training.hyperparameters['batch_size']
    steps_per_epoch = num_samples // batch_size
    total_steps = config.training.hyperparameters.get('iters', steps_per_epoch)
    
    opt_manager = OptimizationManager(config.training, total_steps)
    lr_schedule = opt_manager.create_scheduler()
    base_optimizer = opt_manager.create_optimizer(lr_schedule)
    
    # Wrap with hybrid distributed optimizer
    optimizer = HybridDistributedOptimizer(base_optimizer, device_manager)
    
    return model, optimizer, device_manager, tokenizer, config


def main():
    """Main entry point for hybrid distributed training"""
    parser = argparse.ArgumentParser(description="MLX/CUDA Hybrid Distributed Training")
    parser.add_argument("--config", type=str, required=True, help="Path to model configuration")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--workers", type=str, default=None, help="Path to remote workers configuration file")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run ID")
    parser.add_argument("--dryrun", action="store_true", help="Initialize training but don't run actual training loop")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--steps", type=int, default=None, help="Override number of training steps")
    parser.add_argument("--save-dir", type=str, default="./runs", help="Directory to save checkpoints and logs")
    parser.add_argument("--save-interval", type=int, default=None, help="Override checkpoint save interval")
    parser.add_argument("--eval-interval", type=int, default=None, help="Override evaluation interval")
    parser.add_argument("--log-interval", type=int, default=None, help="Override logging interval")
    parser.add_argument("--local-batch-size", type=int, default=None, help="Override local batch size")
    parser.add_argument("--remote-batch-size", type=int, default=None, help="Override remote batch size")
    parser.add_argument("--remote-only", action="store_true", help="Use only remote workers, no local computation")
    parser.add_argument("--local-only", action="store_true", help="Use only local devices, no remote workers")
    args = parser.parse_args()
    
    # Generate run ID if not provided
    run_id = args.run_id or f"{int(time.time())}"
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set up logging for this run
    run_log_file = os.path.join(args.save_dir, f"hybrid_training_{run_id}.log")
    file_handler = logging.FileHandler(run_log_file)
    log_level = logging.DEBUG if args.debug else logging.INFO
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Also update console logging level if debug is enabled
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
    
    logger.info(f"Starting hybrid distributed training with run ID: {run_id}")
    logger.info(f"Config path: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    
    # Load remote workers configuration if provided
    remote_workers = None
    if args.workers:
        with open(args.workers, 'r') as f:
            workers_config = json.load(f)
            if isinstance(workers_config, dict) and "workers" in workers_config:
                remote_workers = workers_config["workers"]
                logger.info(f"Loaded {len(remote_workers)} remote workers from {args.workers}")
            else:
                remote_workers = workers_config
                logger.info(f"Loaded remote workers configuration from {args.workers}")
    
    # Update configuration with command line overrides
    config_overrides = {}
    if args.steps is not None:
        config_overrides['training.hyperparameters.iters'] = args.steps
    if args.save_interval is not None:
        config_overrides['training.checkpointing.save_interval'] = args.save_interval
    if args.eval_interval is not None:
        config_overrides['training.evaluation.interval'] = args.eval_interval
    if args.log_interval is not None:
        config_overrides['training.logging.interval'] = args.log_interval
    if args.local_batch_size is not None:
        config_overrides['training.hyperparameters.batch_size'] = args.local_batch_size
    if args.remote_batch_size is not None:
        config_overrides['training.hyperparameters.remote_batch_size'] = args.remote_batch_size
    
    # Apply local/remote only flags
    if args.local_only and remote_workers:
        logger.info("Local-only mode enabled, ignoring remote workers")
        remote_workers = None
    
    # Create hybrid trainer
    try:
        if args.debug:
            logger.debug("Creating hybrid trainer with configuration:")
            logger.debug(f"  Config file: {args.config}")
            logger.debug(f"  Config overrides: {config_overrides}")
            logger.debug(f"  Remote workers: {len(remote_workers) if remote_workers else 0} workers")
        
        model, optimizer, device_manager, tokenizer, config = create_hybrid_trainer(
            args.config,
            remote_workers=remote_workers,
            config_overrides=config_overrides
        )
        
        logger.info("Trainer initialized successfully")
        
        if args.dryrun:
            logger.info("Dry run completed - skipping training loop")
        else:
            # For real training, we'll implement a proper training loop
            from train import DataManager
            
            # Initialize data manager
            batch_size = config.training.hyperparameters.get('batch_size', 32)
            data_manager = DataManager(config.data, tokenizer, batch_size=batch_size)
            
            # Get number of training steps
            num_steps = config.training.hyperparameters.get('iters', 1000)
            # Fix the logging attribute access by checking top-level logging section
            log_interval = config.logging.steps.get('logging_interval', 10) if hasattr(config, 'logging') else 10
            save_interval = config.logging.steps.get('checkpoint_interval', 100) if hasattr(config, 'logging') else 100
            
            logger.info(f"Starting training for {num_steps} steps")
            logger.info(f"Logging every {log_interval} steps, saving every {save_interval} steps")
            
            # Training loop
            for step in range(num_steps):
                if step % log_interval == 0:
                    logger.info(f"Step {step}/{num_steps}")
                
                # Generate batch
                batch = data_manager.generate_batch(step)
                
                # Forward and backward pass
                loss, tokens, grads = device_manager.parallel_forward_backward(
                    model, batch[:, :-1], batch[:, 1:]
                )
                
                # Update model
                optimizer.update(model, grads)
                
                if step % log_interval == 0:
                    logger.info(f"Step {step} - Loss: {loss:.4f}, Tokens: {tokens}")
                
                # Save checkpoint
                if step > 0 and step % save_interval == 0:
                    checkpoint_path = os.path.join(args.save_dir, f"checkpoint_{run_id}_{step}.safetensors")
                    logger.info(f"Saving checkpoint to {checkpoint_path}")
                    mx.save(checkpoint_path, model.parameters())
            
            # Save final checkpoint
            final_checkpoint_path = os.path.join(args.save_dir, f"checkpoint_{run_id}_final.safetensors")
            logger.info(f"Saving final checkpoint to {final_checkpoint_path}")
            mx.save(final_checkpoint_path, model.parameters())
            
            logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training: {e}", exc_info=True)
    finally:
        # Clean up resources
        if 'device_manager' in locals():
            device_manager.stop_workers()
        
        logger.info("Hybrid training finished")


if __name__ == "__main__":
    import random
    import importlib
    main()