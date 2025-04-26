import mlx.core as mx
import threading
import queue
import time
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

class DeviceManager:
    """
    Manages computation across multiple devices, including MLX devices and CUDA devices.
    """
    def __init__(self, mlx_devices=None, cuda_devices=None):
        self.mlx_devices = mlx_devices or ["gpu"]
        self.cuda_devices = cuda_devices or []
        self.device_queues = {}
        self.workers = {}
        self.cuda_initialized = False
        
        # Initialize device queues
        for device in self.mlx_devices:
            self.device_queues[device] = queue.Queue()
            
        # Initialize CUDA if needed
        if self.cuda_devices:
            self._init_cuda()
    
    def _init_cuda(self):
        """Initialize CUDA support"""
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                self.num_cuda_devices = min(len(self.cuda_devices), torch.cuda.device_count())
                for i, device_id in enumerate(self.cuda_devices[:self.num_cuda_devices]):
                    device_name = f"cuda:{device_id}"
                    self.device_queues[device_name] = queue.Queue()
                self.cuda_initialized = True
                print(f"CUDA initialized with {self.num_cuda_devices} devices")
            else:
                print("CUDA not available, falling back to MLX only")
                self.cuda_devices = []
        except ImportError:
            print("PyTorch not installed, CUDA support disabled")
            self.cuda_devices = []
    
    def start_workers(self):
        """Start worker threads for each device"""
        for device_name, device_queue in self.device_queues.items():
            if device_name.startswith("cuda:"):
                worker = threading.Thread(
                    target=self._cuda_worker,
                    args=(device_name, device_queue),
                    daemon=True
                )
            else:
                worker = threading.Thread(
                    target=self._mlx_worker, 
                    args=(device_name, device_queue),
                    daemon=True
                )
            worker.start()
            self.workers[device_name] = worker
    
    def _mlx_worker(self, device_name, device_queue):
        """Worker thread for MLX devices"""
        while True:
            task = device_queue.get()
            if task is None:  # Poison pill to stop the thread
                break
                
            func, args, kwargs, result_queue = task
            
            # Set device for this computation
            old_device = mx.default_device()
            device_obj = getattr(mx, device_name)
            mx.set_default_device(device_obj)
            
            try:
                result = func(*args, **kwargs)
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, e))
            finally:
                # Restore previous device
                mx.set_default_device(old_device)
                # Clear cache if needed
                if device_name == "gpu":
                    mx.clear_cache()
                device_queue.task_done()
    
    def _cuda_worker(self, device_name, device_queue):
        """Worker thread for CUDA devices"""
        import torch
        device_id = int(device_name.split(":")[-1])
        device = torch.device(f"cuda:{device_id}")
        
        while True:
            task = device_queue.get()
            if task is None:  # Poison pill to stop the thread
                break
                
            func, args, kwargs, result_queue = task
            
            try:
                # Move inputs to CUDA if needed
                cuda_args = []
                for arg in args:
                    if isinstance(arg, mx.array):
                        # Convert MLX array to numpy to torch tensor
                        cuda_arg = torch.from_numpy(arg.numpy()).to(device)
                        cuda_args.append(cuda_arg)
                    else:
                        cuda_args.append(arg)
                
                cuda_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, mx.array):
                        cuda_kwargs[k] = torch.from_numpy(v.numpy()).to(device)
                    else:
                        cuda_kwargs[k] = v
                
                # Run the function on CUDA
                with torch.cuda.device(device):
                    result = func(*cuda_args, **cuda_kwargs)
                
                # Convert result back to MLX if needed
                if isinstance(result, torch.Tensor):
                    result = mx.array(result.cpu().numpy())
                
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, e))
            finally:
                torch.cuda.empty_cache()
                device_queue.task_done()
    
    def run_on_device(self, device_name, func, *args, **kwargs):
        """Run a function on a specific device"""
        result_queue = queue.Queue()
        self.device_queues[device_name].put((func, args, kwargs, result_queue))
        success, result = result_queue.get()
        if success:
            return result
        else:
            raise result  # Re-raise the exception
    
    def parallel_map(self, func, inputs, device_selection=None):
        """
        Apply a function to each input in parallel across devices.
        
        Args:
            func: Function to apply
            inputs: List of inputs
            device_selection: Optional function to select device for each input
        
        Returns:
            List of results
        """
        if not device_selection:
            # Round-robin device selection
            devices = list(self.device_queues.keys())
            device_selection = lambda i: devices[i % len(devices)]
        
        result_queues = []
        for i, input_item in enumerate(inputs):
            device = device_selection(i)
            result_queue = queue.Queue()
            self.device_queues[device].put((func, (input_item,), {}, result_queue))
            result_queues.append(result_queue)
        
        # Collect results in order
        results = []
        for q in result_queues:
            success, result = q.get()
            if success:
                results.append(result)
            else:
                raise result  # Re-raise the exception
        
        return results
    
    def stop_workers(self):
        """Stop all worker threads"""
        for device_queue in self.device_queues.values():
            device_queue.put(None)  # Poison pill
        for worker in self.workers.values():
            worker.join()


class DistributedOptimizer:
    """
    Wrapper around MLX optimizer that distributes computation across devices.
    """
    def __init__(self, optimizer, device_manager):
        self.optimizer = optimizer
        self.device_manager = device_manager
        self.state = optimizer.state
    
    def update(self, model, gradients):
        """
        Distributed optimizer update - partition large gradient tensors
        across available devices for faster computation.
        """
        # If not distributed, use normal update
        if not self.device_manager.device_queues:
            return self.optimizer.update(model, gradients)
        
        # Partition large gradients
        grad_items = list(tree_flatten(gradients))
        
        # Create update function that applies optimizer's update rule
        def _partition_update(partition_idx, partition_gradients):
            # Apply update rule to this partition
            updates = self.optimizer._update_step(partition_gradients, self.optimizer.state)
            return updates
        
        # Distribute large gradient tensors across devices
        # This is a simplified version - a real implementation would partition
        # the gradients by size/shape and distribute appropriately
        
        # For now, just do normal update
        return self.optimizer.update(model, gradients)
        
    def __getattr__(self, name):
        """Delegate all other methods to the wrapped optimizer"""
        return getattr(self.optimizer, name)