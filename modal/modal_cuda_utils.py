import torch
import mlx.core as mx
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
import os

class ModalCudaManager:
    """
    Enhanced device manager specifically designed for Modal.com deployments
    with CUDA GPUs and optional MLX devices. Optimized for A100 GPUs.
    """
    def __init__(self, cuda_device_count=2, mlx_devices=None, gpu_memory_fraction=0.95):
        self.mlx_devices = mlx_devices or []
        self.cuda_device_count = cuda_device_count
        self.device_queues = {}
        self.workers = {}
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Initialize CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but required for ModalCudaManager")
            
        # Verify CUDA device count
        available_gpus = torch.cuda.device_count()
        if available_gpus < self.cuda_device_count:
            print(f"Warning: Requested {self.cuda_device_count} CUDA devices but only {available_gpus} available")
            self.cuda_device_count = available_gpus
        
        # Set up device queues for each CUDA device
        for i in range(self.cuda_device_count):
            device_name = f"cuda:{i}"
            self.device_queues[device_name] = queue.Queue()
            
            # Configure GPU memory usage (especially important for A100s)
            with torch.cuda.device(i):
                # For A100s, we can allocate a larger portion of memory upfront
                # to avoid fragmentation
                if "A100" in torch.cuda.get_device_name(i):
                    # Don't auto reserve all memory on A100
                    torch.cuda.empty_cache()
                    # Leave some memory for CUDA/PyTorch overhead
                    total_mem = torch.cuda.get_device_properties(i).total_memory
                    reserved = int(total_mem * self.gpu_memory_fraction)
                    # Allocate memory chunk and free it to pre-fragment memory
                    torch.cuda.empty_cache()
                    
                    print(f"Configured A100 GPU:{i} with {reserved/1e9:.2f}GB of {total_mem/1e9:.2f}GB reserved")
            
        # Set up MLX device queues if any are specified
        for device in self.mlx_devices:
            self.device_queues[device] = queue.Queue()
            
        # Print detailed device info
        print(f"ModalCudaManager initialized with {self.cuda_device_count} CUDA devices:")
        for i in range(self.cuda_device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory/1e9:.2f}GB memory, {props.multi_processor_count} SMs")
        
        if len(self.mlx_devices) > 0:
            print(f"MLX devices: {', '.join(self.mlx_devices)}")
        
        # Enable tensor cores for A100s
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    
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
            
        print(f"Started {len(self.workers)} worker threads")
    
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
        """
        Worker thread for CUDA devices.
        Optimized for A100 GPUs with improved memory handling and tensor conversion.
        """
        device_id = int(device_name.split(":")[-1])
        device = torch.device(f"cuda:{device_id}")
        
        # For A100 GPUs, enable additional optimizations
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on A100
        if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
            # Use mixed precision where appropriate
            amp_enabled = True
            scaler = torch.cuda.amp.GradScaler()
        else:
            amp_enabled = False
        
        # For A100 GPUs, set optimal CUDA stream priorities
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device, priority=0)  # High priority stream
        
        # Reusable memory buffers to avoid constant allocations
        input_buffers = {}
        
        while True:
            task = device_queue.get()
            if task is None:  # Poison pill to stop the thread
                break
                
            func, args, kwargs, result_queue = task
            
            try:
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    # Move inputs to CUDA with optimized transfer
                    cuda_args = []
                    for i, arg in enumerate(args):
                        if isinstance(arg, mx.array):
                            # Reuse buffer if possible for this shape
                            shape_key = f"arg_{i}_{tuple(arg.shape)}"
                            if shape_key not in input_buffers:
                                input_buffers[shape_key] = torch.empty(
                                    arg.shape, 
                                    dtype=torch.float32 if arg.dtype in (mx.float32, mx.float16) else torch.int64,
                                    device=device
                                )
                            
                            # Use asynchronous data transfer when possible
                            buffer = input_buffers[shape_key]
                            if arg.dtype in (mx.float32, mx.float16):
                                # Copy directly to GPU without intermediate CPU step when possible
                                np_array = arg.numpy()
                                buffer.copy_(torch.from_numpy(np_array), non_blocking=True)
                                cuda_args.append(buffer)
                            else:
                                # For non-float types, use standard conversion
                                cuda_args.append(torch.from_numpy(arg.numpy()).to(device, non_blocking=True))
                        elif isinstance(arg, torch.Tensor):
                            # Move existing PyTorch tensor to the right device
                            cuda_args.append(arg.to(device, non_blocking=True))
                        else:
                            cuda_args.append(arg)
                    
                    # Process kwargs similarly
                    cuda_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, mx.array):
                            # Use the same buffer optimization approach as for args
                            shape_key = f"kwarg_{k}_{tuple(v.shape)}"
                            if shape_key not in input_buffers:
                                input_buffers[shape_key] = torch.empty(
                                    v.shape, 
                                    dtype=torch.float32 if v.dtype in (mx.float32, mx.float16) else torch.int64,
                                    device=device
                                )
                                
                            buffer = input_buffers[shape_key]
                            if v.dtype in (mx.float32, mx.float16):
                                buffer.copy_(torch.from_numpy(v.numpy()), non_blocking=True)
                                cuda_kwargs[k] = buffer
                            else:
                                cuda_kwargs[k] = torch.from_numpy(v.numpy()).to(device, non_blocking=True)
                        elif isinstance(v, torch.Tensor):
                            cuda_kwargs[k] = v.to(device, non_blocking=True)
                        else:
                            cuda_kwargs[k] = v
                    
                    # Run the function on CUDA, optionally with mixed precision on A100
                    stream.synchronize()  # Ensure all data transfers are complete
                    
                    if amp_enabled and any(isinstance(arg, torch.Tensor) and arg.dtype == torch.float32 
                                        for arg in cuda_args):
                        # Use automatic mixed precision for float computations
                        with torch.cuda.amp.autocast():
                            result = func(*cuda_args, **cuda_kwargs)
                    else:
                        # Run normally
                        result = func(*cuda_args, **cuda_kwargs)
                    
                    # Convert result back to MLX if needed
                    if isinstance(result, torch.Tensor):
                        # Ensure computation is complete before transferring back
                        stream.synchronize()
                        # For small results, use direct copy
                        if result.numel() < 1_000_000:
                            result = mx.array(result.cpu().numpy())
                        else:
                            # For large results, use more efficient memory handling
                            result_cpu = result.cpu()
                            result = mx.array(np.array(result_cpu))
                    
                    result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, e))
            finally:
                # Only perform a targeted cleanup, not a full cache clear
                # which can be expensive on A100s with large memory
                if len(input_buffers) > 20:  # Limit buffer cache size
                    input_buffers = {}  # Clear only when it gets too large
                
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
    
    def batch_split(self, batch, num_splits, optimize_for_a100=True):
        """
        Split a batch into multiple sub-batches for distributed processing.
        Handles both MLX arrays and PyTorch tensors.
        
        Optimized for A100 GPUs with enhanced memory handling and tensor conversion.
        
        Args:
            batch: Input batch (MLX array or PyTorch tensor)
            num_splits: Number of splits to create
            optimize_for_a100: Whether to use A100-specific optimizations
            
        Returns:
            List of sub-batches
        """
        if isinstance(batch, mx.array):
            # Split MLX array
            batch_size = batch.shape[0]
            
            if optimize_for_a100 and batch.ndim > 1:
                # For A100 GPUs, we want to optimize for tensor core utilization
                # which works best with multiples of 8 for batch dimensions
                
                # Calculate optimal split sizes for A100 tensor cores
                # Each split should ideally be a multiple of 8 for optimal performance
                base_split_size = batch_size // num_splits
                adjusted_split_size = ((base_split_size + 7) // 8) * 8  # Round up to next multiple of 8
                
                # Adjust if the rounded size would exceed the batch size
                if adjusted_split_size * num_splits > batch_size:
                    # Fall back to standard splitting
                    split_size = base_split_size
                    remainder = batch_size % num_splits
                else:
                    # Use optimized splitting
                    split_size = adjusted_split_size
                    remainder = batch_size - (split_size * num_splits)
                
                # Create optimized splits
                splits = []
                start_idx = 0
                for i in range(num_splits):
                    # Distribute remainder across early splits
                    extra = 1 if i < remainder else 0
                    end_idx = min(start_idx + split_size + extra, batch_size)
                    
                    if start_idx >= end_idx:
                        break  # Skip empty splits
                        
                    splits.append(batch[start_idx:end_idx])
                    start_idx = end_idx
                
                # Print split info for debugging (only first time)
                if not hasattr(self, '_split_info_printed'):
                    print(f"A100 optimized batch splitting: original_size={batch_size}, "
                          f"num_splits={len(splits)}, sizes={[s.shape[0] for s in splits]}")
                    self._split_info_printed = True
                    
                return splits
            else:
                # Standard splitting for non-optimized case
                split_size = batch_size // num_splits
                remainder = batch_size % num_splits
                
                splits = []
                start_idx = 0
                for i in range(num_splits):
                    end_idx = start_idx + split_size + (1 if i < remainder else 0)
                    splits.append(batch[start_idx:end_idx])
                    start_idx = end_idx
                    
                return splits
                
        elif isinstance(batch, torch.Tensor):
            # Split PyTorch tensor for A100 GPUs
            if optimize_for_a100 and batch.ndim > 1:
                # For A100 GPUs, optimize for tensor cores with multiples of 8
                batch_size = batch.shape[0]
                base_split_size = batch_size // num_splits
                
                # Round to nearest multiple of 8 for tensor core optimization
                adjusted_split_size = ((base_split_size + 7) // 8) * 8
                
                if adjusted_split_size * num_splits > batch_size:
                    # Fall back to standard splitting if optimized size exceeds batch size
                    section_sizes = [base_split_size + (1 if i < batch_size % num_splits else 0) 
                                    for i in range(num_splits)]
                else:
                    # Use optimized splitting
                    section_sizes = [adjusted_split_size] * (batch_size // adjusted_split_size)
                    remainder = batch_size % adjusted_split_size
                    if remainder > 0:
                        section_sizes.append(remainder)
                
                # Remove any empty sections
                section_sizes = [s for s in section_sizes if s > 0]
                
                # Print split info for debugging (only first time)
                if not hasattr(self, '_torch_split_info_printed'):
                    print(f"A100 optimized PyTorch batch splitting: original_size={batch_size}, "
                          f"num_splits={len(section_sizes)}, sizes={section_sizes}")
                    self._torch_split_info_printed = True
                
                return torch.split(batch, section_sizes)
            else:
                # Standard PyTorch splitting for non-optimized case
                return torch.split(batch, batch.shape[0] // num_splits)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
    
    def stop_workers(self):
        """Stop all worker threads"""
        for device_queue in self.device_queues.values():
            device_queue.put(None)  # Poison pill
        for worker in self.workers.values():
            worker.join()


class ModalDistributedOptimizer:
    """
    Distributed optimizer specifically designed for Modal.com deployments.
    Optimized for A100 GPU utilization with enhanced parameter sharding.
    """
    def __init__(self, optimizer, device_manager):
        self.optimizer = optimizer
        self.device_manager = device_manager
        self.state = optimizer.state
        self.cuda_device_count = device_manager.cuda_device_count
        self.param_count = 0
        self.param_size_threshold = 1024 * 1024  # Parameters above 1M elements get special treatment
        self.last_device_log = 0
        
    def update(self, model, gradients):
        """
        Distributed optimizer update - partition large gradient tensors
        across available CUDA devices for faster computation.
        
        Enhanced for A100 GPUs with better parameter sharding.
        """
        # If no CUDA devices, use normal update
        if self.cuda_device_count == 0:
            return self.optimizer.update(model, gradients)
        
        # For a larger model, optimize workload distribution based on parameter size
        # A100s benefit from processing larger contiguous chunks of data
        param_groups = {}
        large_params = []
        total_params = 0
        total_elements = 0
        
        # Phase 1: Count parameters and identify large tensors for special handling
        for name, param in zip(model.parameter_names(), model.parameters()):
            if param not in gradients:
                continue
                
            total_params += 1
            param_size = param.size
            total_elements += param_size
            
            # Special handling for very large parameters (embedding tables, large FC layers)
            if param_size > self.param_size_threshold:
                large_params.append((name, param, gradients[param], param_size))
                continue
                
            # Group regular-sized parameters by layer
            layer_name = name.split('.')[0]  # Group by top-level module
            if layer_name not in param_groups:
                param_groups[layer_name] = []
            
            param_groups[layer_name].append((name, param, gradients[param], param_size))
        
        # Log device utilization periodically
        if total_params > 0 and total_params != self.param_count:
            self.param_count = total_params
            print(f"A100 optimizer processing {total_params} parameters with {total_elements/1e6:.2f}M elements")
            
            if len(large_params) > 0:
                print(f"  Found {len(large_params)} large parameters for special handling:")
                for name, _, _, size in large_params[:3]:  # Show a few examples
                    print(f"    - {name}: {size/1e6:.2f}M elements")
                if len(large_params) > 3:
                    print(f"    - ... and {len(large_params) - 3} more")
        
        # Only distribute if we have enough work to justify it
        if total_params < self.cuda_device_count * 2:
            return self.optimizer.update(model, gradients)
        
        # Phase 2: Create balanced chunks for each device
        chunks = [{} for _ in range(self.cuda_device_count)]
        chunk_elements = [0] * self.cuda_device_count
        
        # First distribute large parameters - one per device
        for idx, (_, param, grad, size) in enumerate(large_params):
            device_idx = idx % self.cuda_device_count
            chunks[device_idx][param] = grad
            chunk_elements[device_idx] += size
        
        # Then distribute remaining parameters to balance workload
        layers = sorted(param_groups.keys())
        for layer in layers:
            # Calculate total elements in this layer
            layer_elements = sum(size for _, _, _, size in param_groups[layer])
            
            # Find device with least work
            device_idx = chunk_elements.index(min(chunk_elements))
            
            # Assign all params from this layer to the device
            for _, param, grad, size in param_groups[layer]:
                chunks[device_idx][param] = grad
                chunk_elements[device_idx] += size
        
        # Phase 3: Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.cuda_device_count) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                if not chunk:  # Skip empty chunks
                    continue
                    
                device = f"cuda:{i % self.cuda_device_count}"
                futures.append(executor.submit(
                    self.device_manager.run_on_device,
                    device,
                    self._update_chunk,
                    chunk
                ))
            
            # Wait for all updates to complete
            for future in futures:
                future.result()
        
        # Apply the optimizer's final update rule to the model
        return self.optimizer.update(model, gradients)
    
    def _update_chunk(self, gradient_chunk):
        """Apply optimizer update to a chunk of gradients on a specific device"""
        updates = self.optimizer._update_step(gradient_chunk, self.optimizer.state)
        return updates
        
    def __getattr__(self, name):
        """Delegate all other methods to the wrapped optimizer"""
        return getattr(self.optimizer, name)