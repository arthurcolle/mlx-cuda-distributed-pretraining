import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import yaml
import mlx.optimizers as optim
# Import optimizers from the correct module paths
import mlx_optimizers as optim_x
from mlx_optimizers import Shampoo, ShampooParams, AdamWEnhanced, SGDEnhanced, LionEnhanced
# Import schedule functions from local module to fix missing schedule functions
from mlx_lm_utils import linear_schedule, cosine_decay, join_schedules
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import os
import threading
import queue
import logging
import sys
from contextlib import contextmanager
# First try custom implementation with Flash Attention
try:
    from models.llama import Model, ModelArgs
    USING_FLASH_ATTENTION = True
    print("Using custom Llama implementation with FlashAttention")
except ImportError:
    USING_FLASH_ATTENTION = False
    print("Flash Attention not available, falling back to standard implementation")
    #from mlx_lm.models.llama import Model, ModelArgs
import importlib
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
from distributed.utils import DeviceManager, DistributedOptimizer
# Import Modal-specific utilities if available
try:
    from modal.modal_cuda_utils import ModalCudaManager, ModalDistributedOptimizer
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

def filter_valid_args(cls, arg_dict):
    valid_params = inspect.signature(cls).parameters
    return {k: v for k, v in arg_dict.items() if k in valid_params}


@dataclass
class DataConfig:
    input_file: str
    preprocessing: Dict[str, int]
    tokenizer: Dict[str, Any]
    tokenizer_path: Optional[str] = None  # Path to a directory containing a tokenizer.json file
    validation_file: Optional[str] = None
    weight_path: Optional[str] = None

@dataclass
class ModelConfig:
    architecture: str
    dimensions: Dict[str, int]
    attention: Dict[str, Any]
    normalization: Dict[str, float]
    rope: Dict[str, Any]
    misc: Dict[str, bool]

@dataclass
class TrainingConfig:
    hyperparameters: Dict[str, Any]
    scheduler: Dict[str, Any]
    optimization: Dict[str, Any]
    epochs: Optional[int] = None
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "patience": 3,
        "min_delta": 0.001,
        "metric": "val_loss",
        "mode": "min"
    })
    lr_finder: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "min_lr": 1e-7,
        "max_lr": 1.0,
        "num_steps": 100
    })

@dataclass
class LoggingConfig:
    log_dir: str
    checkpoint_dir: str
    steps: Dict[str, int]
    metrics: Dict[str, bool]
    # Default to 0 (no validation) if not specified
    tensorboard: bool = False
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_memory_usage: bool = False
    log_gradient_norm: bool = False
    log_parameter_norm: bool = False
    log_samples: bool = False
    log_samples_count: int = 3

@dataclass
class SystemConfig:
    seed: int
    device: str
    distributed: bool = False
    devices: Optional[List[str]] = None
    cuda_devices: Optional[List[int]] = None
    memory_limit: Optional[int] = None
    mixed_precision: bool = False
    precision: str = "float16"  # Options: float16, bfloat16
    gradient_checkpointing: bool = False
    gradient_checkpointing_ratio: float = 0.5  # Fraction of layers to checkpoint
    model_parallel: bool = False
    model_parallel_size: int = 1
    zero_optimization_level: int = 0  # 0: Disabled, 1: Optimizer states, 2: Gradients, 3: Parameters

@dataclass
class ResumeConfig:
    checkpoint: str  # Path to checkpoint base name
    reset_optimizer: bool = False  # Optional flag to reset optimizer state
    reset_training_state: bool = False  # Optional flag to reset training state

@dataclass
class Config:
    name: str  # New field for run name
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    system: SystemConfig
    resume: Optional[ResumeConfig] = None
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Validate that name is present
        if 'name' not in config_dict:
            raise ValueError("Config must specify a 'name' field at the top level")
            
        # Extract epochs if it exists at the top level of training config
        training_config = config_dict['training'].copy()
        epochs = training_config.pop('epochs', None)
        
        # Extract resume config if present
        resume_config = None
        if 'resume' in config_dict:
            resume_config = ResumeConfig(**config_dict['resume'])
        
        return cls(
            name=config_dict['name'],
            overwrite=config_dict.get('overwrite', False),
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**training_config, epochs=epochs),
            logging=LoggingConfig(**config_dict['logging']),
            system=SystemConfig(**config_dict['system']),
            resume=resume_config
        )

class CheckpointManager:
    @staticmethod
    def validate_unique_name(name: str) -> None:
        """Validates that the run directory doesn't already exist"""
        run_path = Path('runs') / name
        if run_path.exists():
            raise ValueError(f"Run directory already exists for name '{name}'")
            
    @staticmethod
    def setup_run_directory(name: str) -> tuple[Path, Path, Path]:
        """Creates and returns paths for run directory structure"""
        run_dir = Path('runs') / name
        checkpoint_dir = run_dir / 'checkpoints'
        
        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(exist_ok=True)
        
        return run_dir, run_dir / 'log.txt', checkpoint_dir
        
    @staticmethod
    def get_checkpoint_paths(checkpoint_path: str) -> tuple[str, str, str]:
        """Returns the paths for model, optimizer, and state files"""
        model_path = f"{checkpoint_path}_model.safetensors"
        optimizer_path = f"{checkpoint_path}_optimizer.safetensors"
        state_path = f"{checkpoint_path}_state.json"
        return model_path, optimizer_path, state_path

class Logger:
    """Advanced logging utility for training."""
    
    def __init__(self, config: LoggingConfig, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.log_file = run_dir / 'log.txt'
        self.tb_writer = None
        self.wandb_run = None
        
        # Configure logging
        self.logger = logging.getLogger("trainer")
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Initialize TensorBoard if enabled
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(run_dir / 'tensorboard'))
                self.logger.info("TensorBoard logging enabled")
            except ImportError:
                self.logger.warning("TensorBoard requested but torch not installed. Disabling TensorBoard logging.")
                self.tb_writer = None
        
        # Initialize Weights & Biases if enabled
        if config.wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=run_dir.name,
                    dir=str(run_dir / 'wandb'),
                    config={
                        "log_dir": config.log_dir,
                        "steps": config.steps,
                        "metrics": config.metrics
                    }
                )
                self.logger.info("Weights & Biases logging enabled")
            except ImportError:
                self.logger.warning("Weights & Biases requested but wandb not installed. Disabling W&B logging.")
                self.wandb_run = None
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """Log metrics to all configured logging destinations."""
        # Log to console/file via logger
        metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
        
        # Log to TensorBoard if enabled
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step)
        
        # Log to Weights & Biases if enabled
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
    
    def log_model_summary(self, model):
        """Log model architecture summary."""
        # Count parameters
        total_params = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
        trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
        
        self.logger.info(f"Model summary:")
        self.logger.info(f"  Total parameters: {total_params:.2f}M")
        self.logger.info(f"  Trainable parameters: {trainable_params:.2f}M")
        
        # Log to W&B if enabled
        if self.wandb_run is not None:
            self.wandb_run.summary["total_parameters"] = total_params
            self.wandb_run.summary["trainable_parameters"] = trainable_params
    
    def log_text_samples(self, step: int, samples: List[str], prefix: str = "generation"):
        """Log text samples to TensorBoard and W&B."""
        if self.tb_writer is not None:
            for i, sample in enumerate(samples):
                self.tb_writer.add_text(f"{prefix}_{i}", sample, step)
        
        if self.wandb_run is not None:
            self.wandb_run.log({f"{prefix}_{i}": sample for i, sample in enumerate(samples)})
    
    def log_memory_usage(self, step: int):
        """Log memory usage statistics."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            self.logger.info(f"Memory usage at step {step}: {memory_usage:.2f} MB")
            
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("system/memory_usage_mb", memory_usage, step)
            
            if self.wandb_run is not None:
                self.wandb_run.log({"system/memory_usage_mb": memory_usage}, step=step)
        except ImportError:
            self.logger.warning("psutil not installed, cannot log memory usage")
    
    def close(self):
        """Close all logging resources."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.wandb_run is not None:
            self.wandb_run.finish()


class TokenizerManager:
    def __init__(self, config: DataConfig, run_dir: Optional[Path] = None):
        self.config = config
        self.external_tokenizer = None
        self.logger = logging.getLogger("tokenizer")
        
        # Check if an external tokenizer path is provided
        if config.tokenizer_path is not None:
            self.use_external_tokenizer(config.tokenizer_path)
            
            # If we have a run directory, copy the tokenizer to it
            if run_dir is not None:
                self.copy_tokenizer_to_run_dir(config.tokenizer_path, run_dir)
        else:
            # Fall back to byte-level tokenization
            self.setup_vocabulary()
    
    def use_external_tokenizer(self, tokenizer_path: str):
        """Load and use an external tokenizer from the specified path."""
        from tokenizers import Tokenizer
        import os
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        
        if not os.path.exists(tokenizer_file):
            raise ValueError(f"Tokenizer file not found at {tokenizer_file}")
        
        print(f"Loading external tokenizer from {tokenizer_file}")
        self.external_tokenizer = Tokenizer.from_file(tokenizer_file)
        
        # Extract special token IDs
        vocab = self.external_tokenizer.get_vocab()
        special_tokens = self.config.tokenizer['special_tokens']
        
        # Map special tokens to their IDs
        self.PAD_TOKEN = vocab.get(special_tokens['pad'])
        self.BOS_TOKEN = vocab.get(special_tokens['bos'])
        self.EOS_TOKEN = vocab.get(special_tokens['eos'])
        self.VOCAB_SIZE = len(vocab)
        
        if self.PAD_TOKEN is None or self.BOS_TOKEN is None or self.EOS_TOKEN is None:
            raise ValueError(f"One or more special tokens not found in the external tokenizer vocabulary")
    
    def copy_tokenizer_to_run_dir(self, tokenizer_path: str, run_dir: Path):
        """Copy the tokenizer files to the run directory."""
        import shutil
        import os
        
        # Create tokenizer directory in run_dir
        run_tokenizer_dir = run_dir / 'tokenizer'
        os.makedirs(run_tokenizer_dir, exist_ok=True)
        
        # Copy tokenizer.json
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        shutil.copy2(tokenizer_file, run_tokenizer_dir / "tokenizer.json")
        
        print(f"Copied tokenizer to {run_tokenizer_dir}")
        
    def setup_vocabulary(self):
        """Set up the byte-level tokenization vocabulary."""
        normal_vocab_size = self.config.tokenizer['normal_vocab_size']
        special_tokens = self.config.tokenizer['special_tokens']
        
        # Create vocabulary mapping
        self.special_token_map = {
            token: normal_vocab_size + idx 
            for idx, token in enumerate(special_tokens.values())
        }
        
        # Store common tokens
        self.PAD_TOKEN = self.special_token_map[special_tokens['pad']]
        self.BOS_TOKEN = self.special_token_map[special_tokens['bos']]
        self.EOS_TOKEN = self.special_token_map[special_tokens['eos']]
        self.VOCAB_SIZE = normal_vocab_size + len(self.special_token_map)
        
    def tokenize(self, text: str) -> list:
        if self.external_tokenizer is not None:
            # Use external tokenizer
            encoded = self.external_tokenizer.encode(text)
            return encoded.ids
        else:
            # Use byte-level tokenization
            return list(text.encode('utf-8'))
            
    def detokenize(self, tokens: list) -> str:
        if self.external_tokenizer is not None:
            # Use external tokenizer
            # Handle both mx.array and list inputs
            if hasattr(tokens, 'tolist'):
                return self.external_tokenizer.decode(tokens.tolist())
            else:
                return self.external_tokenizer.decode(tokens)
        else:
            # Use byte-level detokenization
            # Convert mx.array to list if needed
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            return bytes(tokens).decode('utf-8', errors='ignore')
            
    def tokenize_doc(self, doc: str) -> list:
        """Tokenize a document, ensuring it doesn't exceed the max context size.
        
        Args:
            doc: The text to tokenize
            
        Returns:
            A list of token IDs, including BOS and EOS tokens
        """
        max_length = self.config.preprocessing['max_context_size']
        
        if self.external_tokenizer is not None:
            # Use external tokenizer
            encoded = self.external_tokenizer.encode(doc)
            tokens = encoded.ids[:max_length]
            return [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        else:
            # Use byte-level tokenization
            return [self.BOS_TOKEN] + self.tokenize(doc)[:max_length] + [self.EOS_TOKEN]

class DataManager:
    def __init__(self, config: DataConfig, tokenizer: TokenizerManager, batch_size: int = 1):
        self.config = config
        self.tokenizer = tokenizer
        self.train_docs = []
        self.val_docs = []
        self.train_idx = None
        self.val_idx = None
        self.batch_size = batch_size
        self.load_data()
       
    def load_data(self):
        # Load training data
        self._load_file(self.config.input_file, self.train_docs)
        
        # Set up training batches
        self.train_idx = sorted(range(len(self.train_docs)), key=lambda idx: len(self.train_docs[idx]))
        random.shuffle(self.train_idx)
        self.train_batch_idx = [
            self.train_idx[i : i + self.batch_size : 1]
            for i in range(0, len(self.train_idx) - self.batch_size + 1, self.batch_size)
        ]
        self.train_indices = np.random.permutation(len(self.train_batch_idx))
        
        # Load validation data if specified
        if self.config.validation_file:
            self._load_file(self.config.validation_file, self.val_docs)
            
            # Set up validation batches
            self.val_idx = sorted(range(len(self.val_docs)), key=lambda idx: len(self.val_docs[idx]))
            self.val_batch_idx = [
                self.val_idx[i : i + self.batch_size : 1]
                for i in range(0, len(self.val_idx) - self.batch_size + 1, self.batch_size)
            ]
            self.val_indices = np.random.permutation(len(self.val_batch_idx))
            self.val_ptr = 0  # Pointer for validation batches
            
    def _load_file(self, file_path: str, docs_list: list):
        """Helper method to load documents from a file."""
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                text = d["text"]
                chunk_size = self.config.preprocessing['max_context_size']
                overlap = self.config.preprocessing.get('chunk_overlap', 0)
                
                # Handle overlapping chunks if specified
                stride = chunk_size - overlap
                for i in range(0, len(text), stride):
                    chunk_text = text[i : i + chunk_size]
                    docs_list.append(chunk_text)
    
    def generate_batch(self, step: int) -> mx.array:
        """Generate a training batch."""
        indices = self.train_batch_idx[self.train_indices[step % len(self.train_indices)]]
        return self._create_batch([self.train_docs[i] for i in indices])
    
    def generate_validation_batch(self, batch_idx: int) -> mx.array:
        """Generate a validation batch."""
        if not self.config.validation_file or batch_idx >= len(self.val_batch_idx):
            raise ValueError("No validation data available or batch index out of range")
        
        indices = self.val_batch_idx[self.val_indices[self.val_ptr % len(self.val_indices)]]
        self.val_ptr += 1
        return self._create_batch([self.val_docs[i] for i in indices])
    
    def _create_batch(self, docs: list) -> mx.array:
        """Helper method to create and pad a batch from documents."""
        batch = [self.tokenizer.tokenize_doc(doc) for doc in docs]
        max_len = max(len(x) for x in batch)
        
        # Get max context size from preprocessing config
        max_context = self.config.preprocessing.get('max_context_size', 2048)
        
        # Ensure max_len doesn't exceed model's context window
        max_len = min(max_len, max_context)
        
        # Pad and truncate sequences
        for i in range(len(batch)):
            # Truncate if needed
            if len(batch[i]) > max_len:
                batch[i] = batch[i][:max_len]
            # Pad if needed
            batch[i] += [self.tokenizer.PAD_TOKEN] * (max_len - len(batch[i]))
        
        # Create array
        batch_array = mx.array(batch)
        
        # Print batch shape for debugging
        print(f"Created batch with shape: {batch_array.shape}")
        
        return batch_array
    
    @property
    def has_validation_data(self) -> bool:
        """Check if validation data is available."""
        return self.config.validation_file is not None and len(self.val_docs) > 0
    
    @property
    def num_validation_batches(self) -> int:
        """Get the number of validation batches."""
        return len(self.val_batch_idx) if self.has_validation_data else 0

class MixedPrecisionManager:
    """Manages mixed precision training."""
    
    def __init__(self, enabled: bool = False, precision: str = "float16"):
        self.enabled = enabled
        self.precision = precision
        self.dtype = mx.float16 if precision == "float16" else mx.bfloat16
        self.logger = logging.getLogger("mixed_precision")
        
        if enabled:
            self.logger.info(f"Mixed precision training enabled with {precision}")
        
    @contextmanager
    def cast_forward(self, model):
        """Context manager for forward pass in mixed precision."""
        if not self.enabled:
            yield model
            return
            
        # Store original parameters
        original_params = dict(tree_flatten(model.parameters()))
        
        try:
            # Cast parameters to lower precision for forward pass
            casted_params = tree_map(lambda x: x.astype(self.dtype), original_params)
            model.update(casted_params)
            yield model
        finally:
            # Restore original parameters
            model.update(original_params)
    
    def cast_gradients(self, gradients):
        """Cast gradients back to float32 for optimizer update."""
        if not self.enabled:
            return gradients
        
        return tree_map(lambda x: x.astype(mx.float32), gradients)


class GradientCheckpointer:
    """Implements gradient checkpointing for memory efficiency."""
    
    def __init__(self, enabled: bool = False, ratio: float = 0.5):
        self.enabled = enabled
        self.ratio = ratio
        self.logger = logging.getLogger("gradient_checkpointing")
        
        if enabled:
            self.logger.info(f"Gradient checkpointing enabled with ratio {ratio}")
    
    def apply(self, model):
        """Apply gradient checkpointing to transformer layers."""
        if not self.enabled:
            return model
            
        # Find transformer layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            num_layers = len(layers)
            
            # Determine which layers to checkpoint
            checkpoint_every = max(1, int(1 / self.ratio))
            layers_to_checkpoint = [i for i in range(num_layers) if i % checkpoint_every != 0]
            
            self.logger.info(f"Applying gradient checkpointing to {len(layers_to_checkpoint)}/{num_layers} layers")
            
            # Apply checkpointing
            for i in layers_to_checkpoint:
                if hasattr(layers[i], 'enable_checkpointing'):
                    layers[i].enable_checkpointing()
                else:
                    self.logger.warning(f"Layer {i} does not support checkpointing")
        
        return model


class EarlyStoppingMonitor:
    """Monitors validation metrics for early stopping."""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self.patience = config.get("patience", 3)
        self.min_delta = config.get("min_delta", 0.001)
        self.metric = config.get("metric", "val_loss")
        self.mode = config.get("mode", "min")
        
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
        self.counter = 0
        self.logger = logging.getLogger("early_stopping")
        
        if self.enabled:
            self.logger.info(f"Early stopping enabled with patience={self.patience}, "
                            f"min_delta={self.min_delta}, metric={self.metric}, mode={self.mode}")
    
    def update(self, metrics: Dict[str, float]) -> bool:
        """Update early stopping state with new metrics.
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        if not self.enabled or self.metric not in metrics:
            return False
            
        current_value = metrics[self.metric]
        
        if self.mode == "min":
            improved = self.best_value - current_value > self.min_delta
        else:
            improved = current_value - self.best_value > self.min_delta
            
        if improved:
            self.best_value = current_value
            self.counter = 0
            self.logger.info(f"Early stopping: {self.metric} improved to {current_value:.6f}")
            return False
        else:
            self.counter += 1
            self.logger.info(f"Early stopping: {self.metric} did not improve, counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {self.counter} iterations without improvement")
                return True
                
        return False


class LearningRateFinder:
    """Implements learning rate finder to determine optimal learning rate."""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self.min_lr = config.get("min_lr", 1e-7)
        self.max_lr = config.get("max_lr", 1.0)
        self.num_steps = config.get("num_steps", 100)
        self.logger = logging.getLogger("lr_finder")
        
        if self.enabled:
            self.logger.info(f"Learning rate finder enabled with range {self.min_lr} to {self.max_lr}")
            
        self.lr_values = []
        self.loss_values = []
    
    def get_lr_schedule(self):
        """Create exponential learning rate schedule for the finder."""
        if not self.enabled:
            return None
            
        factor = (self.max_lr / self.min_lr) ** (1 / self.num_steps)
        
        def schedule(step):
            return self.min_lr * (factor ** step)
            
        return schedule
    
    def record(self, step: int, lr: float, loss: float):
        """Record learning rate and loss values."""
        if not self.enabled:
            return
            
        self.lr_values.append(lr)
        self.loss_values.append(loss)
    
    def plot_results(self, save_path: Path):
        """Plot learning rate finder results."""
        if not self.enabled or not self.lr_values:
            return
            
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.lr_values, self.loss_values)
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder Results')
            plt.grid(True)
            plt.savefig(save_path / 'lr_finder_results.png')
            
            # Find the optimal learning rate (point of steepest descent)
            min_gradient_idx = None
            min_gradient = 0
            
            for i in range(1, len(self.lr_values) - 1):
                gradient = (self.loss_values[i+1] - self.loss_values[i-1]) / (self.lr_values[i+1] - self.lr_values[i-1])
                if min_gradient_idx is None or gradient < min_gradient:
                    min_gradient = gradient
                    min_gradient_idx = i
            
            if min_gradient_idx is not None:
                optimal_lr = self.lr_values[min_gradient_idx]
                self.logger.info(f"Suggested optimal learning rate: {optimal_lr:.2e}")
                
                # Mark the optimal point on the plot
                plt.figure(figsize=(10, 6))
                plt.plot(self.lr_values, self.loss_values)
                plt.scatter([optimal_lr], [self.loss_values[min_gradient_idx]], color='red', s=100, zorder=5)
                plt.xscale('log')
                plt.xlabel('Learning Rate')
                plt.ylabel('Loss')
                plt.title(f'Learning Rate Finder Results (Suggested LR: {optimal_lr:.2e})')
                plt.grid(True)
                plt.savefig(save_path / 'lr_finder_results_with_suggestion.png')
                
                # Save the results to a CSV file
                import csv
                with open(save_path / 'lr_finder_results.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['learning_rate', 'loss'])
                    for lr, loss in zip(self.lr_values, self.loss_values):
                        writer.writerow([lr, loss])
                
                return optimal_lr
                
        except ImportError:
            self.logger.warning("matplotlib not installed, cannot plot learning rate finder results")
            return None


class OptimizationManager:
    def __init__(self, config: TrainingConfig, num_training_steps: int):
        self.config = config
        self.num_training_steps = num_training_steps
        self.logger = logging.getLogger("optimization")
        
    def create_scheduler(self) -> Any:
        cfg = self.config.scheduler
        initial_lr = self.config.hyperparameters['learning_rate']
        
        # We already imported scheduler functions at the top of the file
        
        if cfg['type'] == 'cosine_with_warmup':
            warmup = linear_schedule(0, initial_lr, steps=cfg['warmup_steps'])
            cosine = cosine_decay(initial_lr, self.num_training_steps, initial_lr * cfg['min_lr_ratio'])
            return join_schedules([warmup, cosine], [cfg['warmup_steps']])
        elif cfg['type'] == 'cosine':
            return cosine_decay(initial_lr, self.num_training_steps, initial_lr * cfg['min_lr_ratio'])
        elif cfg['type'] == 'linear':
            return linear_schedule(initial_lr, 0, steps=self.num_training_steps)
        else:
            raise ValueError(f"Unsupported scheduler type: {cfg['type']}")
            
    def create_optimizer(self, schedule: Any) -> optim.Optimizer:
        cfg = self.config.optimization
        kwargs = {
            'learning_rate': schedule,
        }
        if 'betas' in cfg:
            kwargs['betas'] = tuple(cfg['betas'])
        if 'eps' in cfg:
            kwargs['eps'] = cfg['eps']
        if 'weight_decay' in cfg:
            kwargs['weight_decay'] = self.config.hyperparameters['weight_decay']
        
        # Add advanced features if specified
        if 'grad_clip_norm' in cfg:
            kwargs['grad_clip_norm'] = cfg['grad_clip_norm']
        if 'ema_momentum' in cfg:
            kwargs['ema_momentum'] = cfg['ema_momentum']
        
        # New enhanced optimizers
        if cfg['optimizer'] == 'adamw_enhanced':
            # Use our enhanced AdamW implementation with proper decoupled weight decay
            return AdamWEnhanced(**kwargs)
        elif cfg['optimizer'] == 'sgd_enhanced':
            # Use our enhanced SGD implementation with proper weight decay
            if 'momentum' in cfg:
                kwargs['momentum'] = cfg['momentum']
            if 'nesterov' in cfg:
                kwargs['nesterov'] = cfg['nesterov']
            return SGDEnhanced(**kwargs)
        elif cfg['optimizer'] == 'lion':
            # Use Lion optimizer (sign-based momentum) - good for large models
            return LionEnhanced(**kwargs)
        # Standard MLX optimizers
        elif cfg['optimizer'] == 'adamw':
            return optim.AdamW(**kwargs)
        elif cfg['optimizer'] == 'adam':
            # MLX AdamW doesn't support weight_decay, so remove it
            if 'weight_decay' in kwargs:
                del kwargs['weight_decay']
            return optim.Adam(**kwargs)
        elif cfg['optimizer'] == 'muon':
            # Muon is a variant of Adam with improved convergence properties
            try:
                from mlx_optimizers import Muon
                muon_kwargs = {
                    'learning_rate': kwargs['learning_rate'],
                    'betas': kwargs.get('betas', (0.9, 0.999)),
                    'eps': kwargs.get('eps', 1e-8),
                    'weight_decay': kwargs.get('weight_decay', 0.0)
                }
                return Muon(**muon_kwargs)
            except ImportError:
                self.logger.warning("Muon optimizer not found, falling back to AdamW")
                return optim.AdamW(**kwargs)
        elif cfg['optimizer'] == 'shampoo':
            # Create Shampoo optimizer with appropriate parameters
            shampoo_params = ShampooParams(
                beta1=cfg.get('beta1', 0.9),
                beta2=cfg.get('beta2', 0.95),
                epsilon=cfg.get('epsilon', 1e-8),
                weight_decay=kwargs.get('weight_decay', 0.0),
                update_period=cfg.get('update_period', 100),
                start_preconditioning_step=cfg.get('start_preconditioning_step', 1000),
                preconditioner_epsilon=cfg.get('preconditioner_epsilon', 1e-6),
                exponent_override=cfg.get('exponent_override', 0.75),
                use_bias_correction=True,
                grafting_optimizer=cfg.get('grafting_optimizer', 'adam'),
                use_decoupled_weight_decay=True
            )
            return Shampoo(learning_rate=kwargs['learning_rate'], params=shampoo_params)
        elif cfg['optimizer'] == 'hybrid':
            # Create hybrid optimizer that combines multiple optimizers
            
            # Determine which optimizers to use for different parameter types
            matrix_opt_name = cfg.get('matrix_optimizer', 'muon')
            non_matrix_opt_name = cfg.get('non_matrix_optimizer', 'adamw')
            
            # Create a temporary config for matrix optimizer
            matrix_cfg = {'optimizer': matrix_opt_name}
            for k, v in cfg.items():
                if k not in ['optimizer', 'non_matrix_optimizer']:
                    matrix_cfg[k] = v
                    
            # Create a temporary config for non-matrix optimizer
            non_matrix_cfg = {'optimizer': non_matrix_opt_name}
            for k, v in cfg.items():
                if k not in ['optimizer', 'matrix_optimizer']:
                    non_matrix_cfg[k] = v
            
            # Recursively create the optimizers
            matrix_optimizer = self.create_optimizer(matrix_cfg)
            non_matrix_optimizer = self.create_optimizer(non_matrix_cfg)
            
            # Create and return the hybrid optimizer
            try:
                from mlx_optimizers import HybridOptimizer
                return HybridOptimizer(
                    learning_rate=kwargs['learning_rate'],
                    matrix_optimizer=matrix_optimizer,
                    non_matrix_optimizer=non_matrix_optimizer
                )
            except ImportError:
                self.logger.warning("HybridOptimizer not found, falling back to matrix_optimizer")
                return matrix_optimizer
        elif cfg['optimizer'] == 'sgd':
            if 'weight_decay' in kwargs:
                del kwargs['weight_decay']  # MLX SGD doesn't support weight_decay
            return optim.SGD(**kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg['optimizer']}")

class Trainer:
    def __init__(self, config_path: str, for_training=True):
        self.config = Config.from_yaml(config_path)
        self.config_path = config_path
        
        # Initialize tracking variables
        self.total_tokens = 0
        self.start_step = 0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger("trainer")
        
        # Validate unique run name before proceeding
        if for_training and not self.config.overwrite and not (self.config.resume and self.config.resume.checkpoint):
            CheckpointManager.validate_unique_name(self.config.name)
        
        self.setup_system()
        
        # Create run directory early so we can copy tokenizer to it
        if for_training:
            self.run_dir, self.log_file, self.checkpoint_dir = CheckpointManager.setup_run_directory(self.config.name)
            # Initialize advanced logger
            self.logger_manager = Logger(self.config.logging, self.run_dir)
        else:
            self.run_dir = None
            self.logger_manager = None
            
        # Initialize tokenizer with run directory if available
        self.tokenizer = TokenizerManager(self.config.data, self.run_dir)
        
        # Initialize mixed precision manager
        self.mixed_precision = MixedPrecisionManager(
            enabled=self.config.system.mixed_precision,
            precision=self.config.system.precision
        )
        
        # Initialize gradient checkpointing
        self.gradient_checkpointer = GradientCheckpointer(
            enabled=self.config.system.gradient_checkpointing,
            ratio=self.config.system.gradient_checkpointing_ratio
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStoppingMonitor(self.config.training.early_stopping)
        
        # Initialize learning rate finder
        self.lr_finder = LearningRateFinder(self.config.training.lr_finder)
        
        self.setup_model()
        if for_training:
            self.data_manager = DataManager(self.config.data, self.tokenizer, batch_size=self.config.training.hyperparameters['batch_size'])
            self.setup_training()
            
            # Initialize validation metrics tracking
            self.validation_steps = self.config.logging.steps.get('validation_interval', 0)
            self.validation_losses = []
            
            # Log model summary
            if self.logger_manager:
                self.logger_manager.log_model_summary(self.model)
    
    def setup_system(self):
        # Set random seeds
        random.seed(self.config.system.seed)
        np.random.seed(self.config.system.seed)
        mx.random.seed(self.config.system.seed)
        
        # Setup distributed training
        self.distributed = self.config.system.distributed
        self.device_mgr = None
        self.running_on_modal = os.environ.get("MODAL_ENVIRONMENT") == "true"
        
        if self.distributed:
            # Setup device mapping
            mlx_devices = self.config.system.devices if self.config.system.devices else ["gpu", "cpu"]
            cuda_devices = self.config.system.cuda_devices if self.config.system.cuda_devices else []
            
            # Check if running on Modal with CUDA
            if self.running_on_modal and MODAL_AVAILABLE:
                # Count available CUDA devices
                try:
                    import torch
                    cuda_device_count = torch.cuda.device_count()
                    if cuda_device_count > 0:
                        print(f"Running on Modal with {cuda_device_count} CUDA devices")
                        # Initialize Modal-specific device manager for CUDA
                        self.device_mgr = ModalCudaManager(
                            cuda_device_count=cuda_device_count, 
                            mlx_devices=[]  # Don't use MLX on Modal (CUDA only)
                        )
                        self.device_mgr.start_workers()
                    else:
                        print("No CUDA devices found on Modal, falling back to standard distribution")
                        self.device_mgr = DeviceManager(mlx_devices=mlx_devices, cuda_devices=[])
                        self.device_mgr.start_workers()
                except (ImportError, Exception) as e:
                    print(f"Error setting up Modal CUDA: {e}")
                    self.device_mgr = DeviceManager(mlx_devices=mlx_devices, cuda_devices=[])
                    self.device_mgr.start_workers()
            else:
                # Initialize standard device manager
                self.device_mgr = DeviceManager(mlx_devices=mlx_devices, cuda_devices=cuda_devices)
                self.device_mgr.start_workers()
                
                print(f"Distributed training enabled across devices: "
                      f"MLX={mlx_devices}, CUDA={cuda_devices if cuda_devices else 'None'}")
        else:
            # Set the default device for non-distributed mode
            if self.config.system.device == "gpu":
                mx.set_default_device(mx.gpu)
                print("Using MLX GPU as default device")
            else:
                mx.set_default_device(mx.cpu)
                print("Using MLX CPU as default device")
        
    def setup_model(self):
        model_cfg = self.config.model
        arch_file = f"models.{model_cfg.architecture}"
        mlx_lm_file = f"mlx_lm.models.{model_cfg.architecture}"
        Model = None
        ModelArgs = None
        try:
            module = importlib.import_module(arch_file)
            Model = getattr(module, 'Model')
            ModelArgs = getattr(module, 'ModelArgs')
        except ImportError:
            try:
                module = importlib.import_module(mlx_lm_file)
                Model = getattr(module, 'Model')
                ModelArgs = getattr(module, 'ModelArgs')
            except ImportError:
                raise ImportError(f"Model architecture '{model_cfg.architecture}' not found in both {arch_file} and {mlx_lm_file}")
        
        all_args = {
            'model_type': model_cfg.architecture,
            'hidden_size': model_cfg.dimensions['hidden_size'],
            'num_hidden_layers': model_cfg.dimensions.get('num_layers', 8),
            'intermediate_size': model_cfg.dimensions['intermediate_size'],
            'num_attention_heads': model_cfg.attention['num_heads'],
            'rms_norm_eps': model_cfg.normalization['rms_norm_eps'],
            'vocab_size': self.tokenizer.VOCAB_SIZE,
            # Ensure head_dim is calculated correctly if not explicitly provided
            'head_dim': model_cfg.attention.get('head_dim', model_cfg.dimensions['hidden_size'] // model_cfg.attention['num_heads']),
            'max_position_embeddings': model_cfg.attention['max_position_embeddings'],
            'num_key_value_heads': model_cfg.attention['num_kv_heads'],
            'attention_bias': model_cfg.misc['attention_bias'],
            'mlp_bias': model_cfg.misc['mlp_bias'],
            'rope_theta': model_cfg.rope['theta'],
            'rope_traditional': model_cfg.rope['traditional'],
            'rope_scaling': model_cfg.rope['scaling'],
            'tie_word_embeddings': model_cfg.misc['tie_word_embeddings'],
            'logit_scale': model_cfg.misc.get('logit_scale', None),
            'num_local_experts': model_cfg.misc.get('num_local_experts', 0),
            'num_experts_per_tok': model_cfg.misc.get('num_experts_per_tok', 0),
            'use_flash_attention': model_cfg.attention.get('use_flash_attention', True),
            'use_flex_attention': model_cfg.attention.get('use_flex_attention', False),
            'flash_block_size': model_cfg.attention.get('flash_block_size', 128),
        }
        valid_args = filter_valid_args(ModelArgs, all_args)
        args = ModelArgs(**valid_args)

        # Log model configuration
        self.logger.info(f"Model configuration:")
        self.logger.info(f"  Architecture: {model_cfg.architecture}")
        self.logger.info(f"  Hidden size: {args.hidden_size}")
        self.logger.info(f"  Num attention heads: {args.num_attention_heads}")
        self.logger.info(f"  Head dim: {args.head_dim}")
        self.logger.info(f"  Num layers: {args.num_hidden_layers}")
        self.logger.info(f"  Intermediate size: {args.intermediate_size}")
        self.logger.info(f"  Vocab size: {args.vocab_size}")
        
        # Create model
        self.model = Model(args)

        # Apply gradient checkpointing if enabled
        self.model = self.gradient_checkpointer.apply(self.model)

        # Load pre-trained weights if specified
        if self.config.data.weight_path is not None:
            self.logger.info(f"Loading weights from {self.config.data.weight_path}")
            self.model.load_weights(self.config.data.weight_path, strict=False)
            
        # Log model size
        p = sum(v.size for _, v in tree_flatten(self.model.trainable_parameters())) / 10**6
        self.logger.info(f"Model has {p:.2f}M parameters")
        
        # Initialize model parallelism if enabled
        if self.config.system.model_parallel and self.config.system.model_parallel_size > 1:
            self.setup_model_parallelism()
        
    def setup_training(self):
        # Calculate number of training steps
        num_samples = len(self.data_manager.train_docs)
        batch_size = self.config.training.hyperparameters['batch_size']
        steps_per_epoch = num_samples // batch_size
        
        if self.config.training.epochs is not None:
            # If epochs is set, calculate total steps based on epochs
            self.total_steps = steps_per_epoch * self.config.training.epochs
        else:
            # Otherwise use specified iters or default to one epoch
            self.total_steps = self.config.training.hyperparameters.get('iters', steps_per_epoch)
        
        # Store steps_per_epoch for logging
        self.steps_per_epoch = steps_per_epoch
        
        # Setup optimization
        opt_manager = OptimizationManager(self.config.training, self.total_steps)
        self.lr_schedule = opt_manager.create_scheduler()
        base_optimizer = opt_manager.create_optimizer(self.lr_schedule)
        
        # Set up gradient accumulation if enabled
        self.grad_accum_steps = self.config.training.hyperparameters.get('gradient_accumulation_steps', 1)
        if self.grad_accum_steps > 1:
            print(f"Using gradient accumulation with {self.grad_accum_steps} steps")
            # Effective batch size for logging
            self.effective_batch_size = batch_size * self.grad_accum_steps
            print(f"Effective batch size: {self.effective_batch_size}")
        
        # Wrap optimizer in distributed optimizer if needed
        if self.distributed and self.device_mgr:
            if self.running_on_modal and MODAL_AVAILABLE and isinstance(self.device_mgr, ModalCudaManager):
                # Use Modal-optimized distributed optimizer
                self.optimizer = ModalDistributedOptimizer(base_optimizer, self.device_mgr)
                print("Using Modal distributed optimizer for CUDA workload")
            else:
                # Use standard distributed optimizer
                self.optimizer = DistributedOptimizer(base_optimizer, self.device_mgr)
                print("Using distributed optimizer for mixed MLX-CUDA workload")
        else:
            self.optimizer = base_optimizer
        
    def setup_logging(self):
        # Run directory structure should already be set up in __init__
        
        # Create initial metadata file
        metadata = {
            'name': self.config.name,
            'created_at': datetime.now().isoformat(),
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
                'system': self.config.system.__dict__
            },
            'training_info': {
                'steps_per_epoch': self.steps_per_epoch,
                'total_steps': self.total_steps,
                'epochs': self.config.training.epochs,
                'gradient_accumulation_steps': getattr(self, 'grad_accum_steps', 1),
                'effective_batch_size': getattr(self, 'effective_batch_size', 
                                              self.config.training.hyperparameters['batch_size'])
            }
        }
        
        # Add tokenizer information to metadata
        if self.config.data.tokenizer_path:
            metadata['tokenizer'] = {
                'type': 'external',
                'path': self.config.data.tokenizer_path,
                'vocab_size': self.tokenizer.VOCAB_SIZE
            }
        else:
            metadata['tokenizer'] = {
                'type': 'byte-level',
                'vocab_size': self.tokenizer.VOCAB_SIZE
            }
        
        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save the config used to the run directory
        with open(self.run_dir / 'config.yaml', 'w') as f:
            with open(self.config_path, 'r') as config_file:
                f.write(config_file.read())
    
    def setup_model_parallelism(self):
        """Set up model parallelism across multiple devices."""
        if not self.distributed or not self.device_mgr:
            self.logger.warning("Model parallelism requested but distributed training not enabled")
            return
            
        self.logger.info(f"Setting up model parallelism with {self.config.system.model_parallel_size} partitions")
        
        # This is a placeholder for actual model parallelism implementation
        # In a real implementation, we would:
        # 1. Partition the model across devices
        # 2. Set up communication between partitions
        # 3. Modify forward/backward passes to handle partitioned execution
        
        # For now, just log that it would be implemented
        self.logger.info("Model parallelism support is a placeholder - not fully implemented yet")

    def compute_loss(self, model, inputs: mx.array, targets: mx.array) -> Tuple[mx.array, int]:
        # Log input shapes at debug level
        self.logger.debug(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        
        # Standard loss computation for non-distributed case
        if not self.distributed or not self.device_mgr:
            # Check if the batch is too large for the model's context window
            max_ctx = self.config.model.attention.get('max_position_embeddings', 2048)
            if inputs.shape[1] > max_ctx:
                self.logger.warning(f"Input sequence length {inputs.shape[1]} exceeds model's max context {max_ctx}")
                # Truncate to avoid shape errors
                inputs = inputs[:, :max_ctx]
                targets = targets[:, :max_ctx]
                self.logger.debug(f"Truncated to: Input shape: {inputs.shape}, Target shape: {targets.shape}")
                
            # Ensure batch size and sequence length are compatible with model's attention heads
            # This prevents reshape errors in attention computation
            n_heads = self.config.model.attention['num_heads']
            head_dim = self.config.model.attention.get('head_dim', 
                                                     self.config.model.dimensions['hidden_size'] // n_heads)
            
            # Check if sequence length is compatible with attention computation
            if inputs.shape[0] * inputs.shape[1] * n_heads * head_dim != inputs.shape[0] * inputs.shape[1] * self.config.model.dimensions['hidden_size']:
                self.logger.warning(f"Input dimensions not compatible with attention heads configuration. "
                                   f"Batch: {inputs.shape[0]}, Seq: {inputs.shape[1]}, Heads: {n_heads}, Dim: {head_dim}")
            
            # Use mixed precision for forward pass if enabled
            with self.mixed_precision.cast_forward(model) as mp_model:
                logits = mp_model(inputs)
                
                # Always use float32 for loss computation
                logits = logits.astype(mx.float32)
                loss = nn.losses.cross_entropy(logits, targets)
                
                # Mask padding tokens
                pad_mask = (targets != self.tokenizer.PAD_TOKEN)
                loss = loss * pad_mask
                ntoks = pad_mask.sum()
                
                return loss.sum() / ntoks, ntoks
        
        # Distributed loss computation for MLX-CUDA mixed workload
        # Here we could partition the batch across devices
        # For now, we keep it simple - just use the first MLX device
        device = next(iter(self.device_mgr.device_queues.keys()))
        
        def _compute_fwd(model_inputs):
            # Function to be executed on a specific device
            model_in, model_tgt = model_inputs
            logits = model(model_in)
            logits = logits.astype(mx.float32)
            loss = nn.losses.cross_entropy(logits, model_tgt)
            
            # Mask padding tokens
            pad_mask = (model_tgt != self.tokenizer.PAD_TOKEN)
            loss = loss * pad_mask
            ntoks = pad_mask.sum()
            
            return loss.sum(), ntoks
        
        # Run the forward pass on the selected device
        loss_sum, ntoks = self.device_mgr.run_on_device(
            device, _compute_fwd, (inputs, targets)
        )
        
        return loss_sum / ntoks, ntoks
        
    def validate(self) -> float:
        """Run validation on the validation dataset.
        
        Returns:
            float: Average validation loss
        """
        if not self.data_manager.has_validation_data:
            return None
            
        # Ensure we're in evaluation mode (no need for gradients)
        total_loss = 0.0
        total_tokens = 0
        
        # Process all validation batches
        num_batches = min(self.data_manager.num_validation_batches, 50)  # Cap at 50 batches to avoid too long validation
        
        # Print batch shape information for debugging
        if num_batches > 0:
            sample_batch = self.data_manager.generate_validation_batch(0)
            print(f"Validation batch shape: {sample_batch.shape}")
        
        if not self.distributed or not self.device_mgr:
            # Standard validation for non-distributed case
            for batch_idx in range(num_batches):
                try:
                    batch = self.data_manager.generate_validation_batch(batch_idx)
                    
                    # Forward pass only
                    loss, tokens = self.compute_loss(self.model, batch[:, :-1], batch[:, 1:])
                
                    # Accumulate metrics
                    total_loss += float(loss.item() if hasattr(loss, 'item') else loss)
                    total_tokens += int(tokens.item() if hasattr(tokens, 'item') else tokens)
                    
                    # Clear GPU cache if needed - just continue if mx.clear_cache is not available
                    if not self.distributed and self.config.system.device == "gpu":
                        try:
                            mx.clear_cache()
                        except AttributeError:
                            # mx.clear_cache() might not be available in this version
                            pass
                except ValueError as e:
                    # Skip batches that cause reshape errors
                    print(f"Skipping validation batch {batch_idx} due to error: {e}")
                    continue
        else:
            # Distributed validation - process batches in parallel
            validation_batches = []
            for batch_idx in range(num_batches):
                batch = self.data_manager.generate_validation_batch(batch_idx)
                validation_batches.append((batch[:, :-1], batch[:, 1:]))
            
            # Create validation function that computes loss for a single batch
            def _compute_val_loss(batch_pair):
                inputs, targets = batch_pair
                logits = self.model(inputs)
                logits = logits.astype(mx.float32)
                loss = nn.losses.cross_entropy(logits, targets)
                
                # Mask padding tokens
                pad_mask = (targets != self.tokenizer.PAD_TOKEN)
                loss = loss * pad_mask
                ntoks = pad_mask.sum()
                
                return float(loss.sum()), int(ntoks)
            
            # Process batches in parallel across devices
            results = []
            for batch_pair in validation_batches:
                # For simplicity, just use the first device
                # In a more advanced implementation, we would distribute across all devices
                device = next(iter(self.device_mgr.device_queues.keys()))
                loss, tokens = self.device_mgr.run_on_device(device, _compute_val_loss, batch_pair)
                results.append((loss, tokens))
            
            # Accumulate results
            for loss, tokens in results:
                total_loss += loss
                total_tokens += tokens
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return avg_loss

    def save_checkpoint(self, step: int | str, val_loss: float = None):
        # Save model weights
        weights = dict(tree_flatten(self.model.parameters()))
        model_path = self.checkpoint_dir / f'step_{step}_model.safetensors'
        mx.save_safetensors(str(model_path), weights)
        
        # Save optimizer state
        optimizer_state = dict(tree_flatten(self.optimizer.state))
        optimizer_path = self.checkpoint_dir / f'step_{step}_optimizer.safetensors'
        mx.save_safetensors(str(optimizer_path), optimizer_state)
        
        # Save training state
        training_state = {
            'step': step if isinstance(step, int) else self.total_steps,
            'val_ptr': self.data_manager.val_ptr,
            'total_tokens': self.total_tokens.item(),
            'validation_losses': self.validation_losses,
        }
        state_path = self.checkpoint_dir / f'step_{step}_state.json'
        with open(state_path, 'w') as f:
            json.dump(training_state, f)
        
        # Update metadata with checkpoint info
        metadata_path = self.run_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        if 'checkpoints' not in metadata:
            metadata['checkpoints'] = []
        
        checkpoint_info = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'paths': {
                'model': f'checkpoints/step_{step}_model.safetensors',
                'optimizer': f'checkpoints/step_{step}_optimizer.safetensors',
                'state': f'checkpoints/step_{step}_state.json'
            }
        }
        
        # Include validation loss if available
        if val_loss is not None:
            checkpoint_info['validation_loss'] = val_loss
            
        metadata['checkpoints'].append(checkpoint_info)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def log_metrics(self, step: int, loss: float, tokens: int, 
                   total_tokens: int, start_time: float, val_loss: float = None) -> str:
        metrics = []
        
        # Add epoch information if epochs are configured
        if self.config.training.epochs is not None:
            current_epoch = step // self.steps_per_epoch + 1
            epoch_step = step % self.steps_per_epoch + 1
            metrics.append(f"epoch={current_epoch}/{self.config.training.epochs} ({epoch_step}/{self.steps_per_epoch})")
        
        if self.config.logging.metrics['log_loss']:
            metrics.append(f"loss={loss:.3e}")
            
            # Add validation loss if available
            if val_loss is not None:
                metrics.append(f"val_loss={val_loss:.3e}")
            
        if self.config.logging.metrics['log_perplexity']:
            metrics.append(f"ppl={np.exp(loss):.2f}")
            
            # Add validation perplexity if available
            if val_loss is not None:
                metrics.append(f"val_ppl={np.exp(val_loss):.2f}")
            
        if self.config.logging.metrics['log_tokens_per_second']:
            tokens_per_sec = total_tokens / (1000 * (time.time() - start_time))
            metrics.append(f"tok/s={tokens_per_sec:.2f}K")
        
        if self.config.logging.metrics['log_tokens_processed']:
            metrics.append(f"toks={tokens}")
            
        if self.config.logging.metrics['log_learning_rate']:
            metrics.append(f"lr={self.lr_schedule(step):.3e}")

        # Add gradient accumulation info if enabled
        if hasattr(self, 'grad_accum_steps') and self.grad_accum_steps > 1:
            metrics.append(f"accum={self.grad_accum_steps}")
            metrics.append(f"eff_bs={self.effective_batch_size}")
            
        return " | ".join(metrics)

    def load_checkpoint(self, checkpoint_path: str, reset_optimizer: bool = False):
        """Load a checkpoint and restore model, optimizer, and training state"""
        # Extract step from checkpoint path
        step_str = checkpoint_path.split('step_')[-1]
        
        # Get checkpoint file paths
        model_path, optimizer_path, state_path = CheckpointManager.get_checkpoint_paths(checkpoint_path)
        
        # Load model weights with strict=False to allow architecture mismatches
        print(f"Loading model weights from {model_path}")
        self.model.load_weights(model_path, strict=False)
        
        # Load optimizer state if not resetting
        if not reset_optimizer:
            print(f"Loading optimizer state from {optimizer_path}")
            try:
                state_dict = mx.load(optimizer_path)
                state = tree_unflatten(list(state_dict.items()))
                self.optimizer.state = state
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
                print("Continuing with fresh optimizer state")
        
        # Load training state
        print(f"Loading training state from {state_path}")
        try:
            with open(state_path, 'r') as f:
                training_state = json.load(f)
            
            # Restore training state
            self.start_step = training_state['step'] if isinstance(training_state['step'], int) else 0
            self.data_manager.val_ptr = training_state.get('val_ptr', 0)
            self.total_tokens = training_state.get('total_tokens', 0)
            self.validation_losses = training_state.get('validation_losses', [])
            
            print(f"Resumed training from checkpoint {checkpoint_path} at step {self.start_step}")
        except Exception as e:
            print(f"Warning: Failed to load training state: {e}")
            print("Continuing with default training state")
            self.start_step = 0
        
        return self.start_step

    def run_learning_rate_finder(self):
        """Run the learning rate finder to determine optimal learning rate."""
        if not self.lr_finder.enabled:
            return None
            
        self.logger.info("Running learning rate finder...")
        
        # Create a temporary optimizer with the LR finder schedule
        lr_schedule = self.lr_finder.get_lr_schedule()
        temp_optimizer = self.create_optimizer_for_lr_finder(lr_schedule)
        
        # Create appropriate loss function
        loss_value_and_grad = nn.value_and_grad(self.model, self.compute_loss)
        
        # Run for specified number of steps
        for step in tqdm(range(self.lr_finder.num_steps), desc="LR Finder"):
            # Generate batch
            batch = self.data_manager.generate_batch(step)
            
            # Forward and backward pass
            (loss, _), grad = loss_value_and_grad(
                self.model, batch[:, :-1], batch[:, 1:]
            )
            
            # Update model
            temp_optimizer.update(self.model, grad)
            mx.eval(loss)
            
            # Record learning rate and loss
            current_lr = lr_schedule(step)
            self.lr_finder.record(step, current_lr, float(loss.item()))
            
            # Clear cache if on GPU
            if not self.distributed and self.config.system.device == "gpu":
                try:
                    mx.clear_cache()
                except AttributeError:
                    pass
                    
            # Stop if loss becomes NaN or too large
            if np.isnan(loss.item()) or loss.item() > 4 * self.lr_finder.loss_values[0]:
                self.logger.info(f"Stopping LR finder early at step {step} due to loss divergence")
                break
        
        # Plot results and get suggested learning rate
        optimal_lr = self.lr_finder.plot_results(self.run_dir)
        
        # Reset model to initial state (we don't want to keep the finder's updates)
        if self.config.data.weight_path is not None:
            self.logger.info("Reloading initial weights after LR finder")
            self.model.load_weights(self.config.data.weight_path, strict=False)
            
        return optimal_lr
        
    def create_optimizer_for_lr_finder(self, lr_schedule):
        """Create a simple optimizer for the learning rate finder."""
        # Use SGD for LR finder as it's less adaptive than Adam-based optimizers
        return optim.SGD(learning_rate=lr_schedule)

    def train(self):
        # Initialize variables
        total_tokens = self.total_tokens
        start_step = 0
        
        # Check if resuming from checkpoint
        if self.config.resume and self.config.resume.checkpoint:
            checkpoint_path = self.config.resume.checkpoint
            reset_optimizer = self.config.resume.reset_optimizer
            start_step = self.load_checkpoint(checkpoint_path, reset_optimizer)

            if getattr(self.config.resume, 'reset_training_state', False):
                # Reset training state while keeping model (and optionally optimizer)
                start_step = 0
                self.start_step = 0
                self.total_tokens = 0
                self.validation_losses = []
                # Reset validation pointer if available
                try:
                    self.data_manager.val_ptr = 0
                except Exception:
                    pass
                skip_initial_validation = False
            else:
                # If we're resuming normally, skip the initial validation
                skip_initial_validation = True
        else:
            skip_initial_validation = False
            
        # Run learning rate finder if enabled
        if self.lr_finder.enabled and not (self.config.resume and self.config.resume.checkpoint):
            optimal_lr = self.run_learning_rate_finder()
            if optimal_lr is not None:
                # Update learning rate in config
                self.logger.info(f"Setting learning rate to {optimal_lr:.2e} based on LR finder")
                self.config.training.hyperparameters['learning_rate'] = optimal_lr
                # Recreate optimizer with new learning rate
                self.setup_training()
        
        # Create appropriate loss function based on distributed setting
        if not self.distributed or not self.device_mgr:
            loss_value_and_grad = nn.value_and_grad(self.model, self.compute_loss)
        else:
            # For distributed mode, we'll handle gradients differently
            def distributed_value_and_grad(model, inputs, targets):
                # First device for forward/backward pass
                device = next(iter(self.device_mgr.device_queues.keys()))
                
                def _value_and_grad(inputs_targets):
                    x, y = inputs_targets
                    val_grad_fn = nn.value_and_grad(model, self.compute_loss)
                    return val_grad_fn(model, x, y)
                
                return self.device_mgr.run_on_device(device, _value_and_grad, (inputs, targets))
            
            loss_value_and_grad = distributed_value_and_grad
            
        start_time = time.time()
        # Create progress bar with adjusted range for resuming
        progress_bar = tqdm(range(self.total_steps), desc="Training", initial=start_step)

        
        # Initialize logging
        with open(self.log_file, 'a' if start_step > 0 else 'w') as log_file:
            if start_step == 0:
                log_file.write(f"Training started at {datetime.now()}\n")
                log_file.write(f"Total steps: {self.total_steps}\n")
                if self.config.training.epochs is not None:
                    log_file.write(f"Training for {self.config.training.epochs} epochs with {self.steps_per_epoch} steps per epoch\n")
                if self.data_manager.has_validation_data:
                    log_file.write(f"Validation data: {self.config.data.validation_file}\n")
                    log_file.write(f"Validation batches: {self.data_manager.num_validation_batches}\n")
                # Log gradient accumulation if enabled
                if hasattr(self, 'grad_accum_steps') and self.grad_accum_steps > 1:
                    log_file.write(f"Using gradient accumulation with {self.grad_accum_steps} steps\n")
                    log_file.write(f"Effective batch size: {self.effective_batch_size}\n")
                log_file.write("=" * 50 + "\n\n")
            else:
                log_file.write(f"\nResuming training at step {start_step} at {datetime.now()}\n")
                log_file.write(f"Remaining steps: {self.total_steps - start_step}\n")
                log_file.write("=" * 50 + "\n\n")
            
            # Log initial validation loss if validation data is available and not resuming
            val_loss = None
            if self.validation_steps > 0 and self.data_manager.has_validation_data and not skip_initial_validation:
                val_loss = self.validate()
                log_file.write(f"Initial validation loss: {val_loss:.4e} (ppl={np.exp(val_loss):.2f})\n\n")
                # Add to validation loss history
                self.validation_losses.append((0, val_loss))
            
            # Initialize gradient accumulation variables
            accumulated_gradients = None
            accumulated_tokens = 0
            accum_step = 0
            
            # Track metrics for logging
            metrics_to_log = {}
            
            for step in progress_bar:
                step += start_step
                if step >= self.total_steps:
                    break
                    
                # Check if we need to do gradient accumulation
                grad_accum_steps = getattr(self, 'grad_accum_steps', 1)
                
                # Generate batch
                batch = self.data_manager.generate_batch(step)
                
                # Forward and backward pass with mixed precision
                with self.mixed_precision.cast_forward(self.model):
                    (loss, tokens), grad = loss_value_and_grad(
                        self.model, batch[:, :-1], batch[:, 1:]
                    )
                
                # Cast gradients back to float32 if using mixed precision
                if self.mixed_precision.enabled:
                    grad = self.mixed_precision.cast_gradients(grad)
                
                # Calculate gradient norm for logging if enabled
                if self.config.logging.log_gradient_norm:
                    grad_norm = mx.sqrt(sum(mx.sum(g**2) for g in tree_flatten(grad)[0]))
                    metrics_to_log['grad_norm'] = float(grad_norm.item())
                
                # Gradient clipping if configured
                if 'gradient_clip' in self.config.training.hyperparameters:
                    clip_value = self.config.training.hyperparameters['gradient_clip']
                    grad = tree_map(lambda x: mx.clip(x, -clip_value, clip_value), grad)
                
                # Gradient accumulation
                if grad_accum_steps > 1:
                    # Scale the gradient by 1/grad_accum_steps
                    scaled_grad = tree_map(lambda x: x / grad_accum_steps, grad)
                    
                    if accumulated_gradients is None:
                        # First accumulation step
                        accumulated_gradients = scaled_grad
                    else:
                        # Add to accumulated gradients
                        accumulated_gradients = tree_map(
                            lambda x, y: x + y, accumulated_gradients, scaled_grad
                        )
                    
                    # Accumulate tokens
                    accumulated_tokens += tokens
                    accum_step += 1
                    
                    # Only update if we've accumulated enough gradients or if it's the last step
                    if accum_step == grad_accum_steps or step == self.total_steps - 1:
                        # Update model with accumulated gradients
                        total_tokens += accumulated_tokens
                        self.optimizer.update(self.model, accumulated_gradients)
                        mx.eval(loss)
                        
                        # Reset accumulation
                        accumulated_gradients = None
                        accumulated_tokens = 0
                        accum_step = 0
                else:
                    # Standard update without accumulation
                    total_tokens += tokens
                    self.optimizer.update(self.model, grad)
                    mx.eval(loss)
                
                # Calculate parameter norm for logging if enabled
                if self.config.logging.log_parameter_norm:
                    param_norm = mx.sqrt(sum(mx.sum(p**2) for p in tree_flatten(self.model.parameters())[0]))
                    metrics_to_log['param_norm'] = float(param_norm.item())
                
                if not self.distributed and self.config.system.device == "gpu":
                    try:
                        mx.clear_cache()
                    except AttributeError:
                        # mx.clear_cache() might not be available in this version
                        pass
                
                # Run validation
                if self.validation_steps > 0 and self.data_manager.has_validation_data and (step + 1) % self.validation_steps == 0:
                    val_loss = self.validate()
                    # Add to validation loss history
                    self.validation_losses.append((step + 1, val_loss))
                    
                    # Add validation metrics to logging
                    metrics_to_log['val_loss'] = val_loss
                    metrics_to_log['val_ppl'] = np.exp(val_loss)
                    
                    # Check early stopping
                    if self.early_stopping.update({'val_loss': val_loss}):
                        self.logger.info("Early stopping triggered, ending training")
                        break
                
                # Log memory usage if enabled
                if self.config.logging.log_memory_usage and step % self.config.logging.steps['logging_interval'] == 0:
                    self.logger_manager.log_memory_usage(step)
                
                # Generate and log text samples if enabled
                if self.config.logging.log_samples and step % self.config.logging.steps.get('sample_interval', 1000) == 0:
                    self.generate_and_log_samples(step)
                
                # Logging
                if step % self.config.logging.steps['logging_interval'] == 0:
                    # Add current metrics
                    metrics_to_log.update({
                        'loss': float(loss.item()),
                        'ppl': float(np.exp(loss.item())),
                        'lr': float(self.lr_schedule(step)),
                        'tokens': int(tokens.item()),
                        'total_tokens': int(total_tokens.item()),
                        'tokens_per_sec': float(total_tokens / (time.time() - start_time))
                    })
                    
                    # Log metrics using the logger manager
                    if self.logger_manager:
                        self.logger_manager.log_metrics(step, metrics_to_log)
                    
                    # Format metrics for progress bar
                    metrics_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                             for k, v in metrics_to_log.items()])
                    progress_bar.set_description(metrics_str)
                    
                    # Clear metrics for next logging interval
                    metrics_to_log = {}
                
                # Save checkpoint
                if (1 + step) % self.config.logging.steps['checkpoint_interval'] == 0:
                    # Find the most recent validation loss if available
                    last_val_loss = val_loss if val_loss is not None else None
                    # Update total_tokens in the trainer instance for checkpoint saving
                    self.total_tokens = total_tokens
                    self.save_checkpoint(step + 1, last_val_loss)
        
        # Final validation
        final_val_loss = None
        if self.validation_steps > 0 and self.data_manager.has_validation_data:
            final_val_loss = self.validate()
            self.validation_losses.append((self.total_steps, final_val_loss))
        
        # Save final checkpoint with validation loss
        self.total_tokens = total_tokens
        self.save_checkpoint("final", final_val_loss)
        
        # Save validation losses to metadata
        if self.validation_losses:
            metadata_path = self.run_dir / 'metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['validation'] = {
                'steps': [step for step, _ in self.validation_losses],
                'losses': [float(loss) for _, loss in self.validation_losses]
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Write final summary
        with open(self.log_file, 'a') as log_file:
            log_file.write("\n" + "=" * 50 + "\n")
            log_file.write(f"Training completed at {datetime.now()}\n")
            
            # Create final metrics summary
            final_metrics = {
                'total_tokens': int(total_tokens),
                'tokens_per_sec': float(total_tokens / (time.time() - start_time))
            }
            
            if final_val_loss is not None:
                final_metrics['val_loss'] = float(final_val_loss)
                final_metrics['val_ppl'] = float(np.exp(final_val_loss))
            
            # Format metrics for logging
            metrics_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                     for k, v in final_metrics.items()])
            log_file.write(f"Final training metrics: {metrics_str}\n")
            
            if final_val_loss is not None:
                log_file.write(f"Final validation loss: {final_val_loss:.4e} (ppl={np.exp(final_val_loss):.2f})\n")
            log_file.write(f"Total tokens processed: {total_tokens/1000:.2f}K\n")

    def generate_and_log_samples(self, step: int):
        """Generate and log text samples during training."""
        if not self.config.logging.log_samples or not self.logger_manager:
            return
            
        try:
            # Create a simple generation function
            def generate_sample(prompt_tokens, max_tokens=50):
                # Simple greedy decoding
                all_tokens = list(prompt_tokens)
                
                for _ in range(max_tokens):
                    # Get the next token prediction
                    with self.mixed_precision.cast_forward(self.model):
                        logits = self.model(mx.array([all_tokens]))
                    
                    # Get the last token's logits
                    next_token_logits = logits[0, -1, :]
                    
                    # Greedy selection
                    next_token = mx.argmax(next_token_logits)
                    
                    # Add to generated sequence
                    all_tokens.append(int(next_token.item()))
                    
                    # Stop if we hit EOS
                    if int(next_token.item()) == self.tokenizer.EOS_TOKEN:
                        break
                
                return all_tokens
            
            # Generate samples from validation data if available
            samples = []
            num_samples = min(self.config.logging.log_samples_count, 3)
            
            if self.data_manager.has_validation_data:
                for i in range(num_samples):
                    # Get a validation batch
                    val_batch = self.data_manager.generate_validation_batch(i)
                    
                    # Take the first sequence from the batch
                    prompt_tokens = val_batch[0, :20].tolist()  # Use first 20 tokens as prompt
                    
                    # Generate continuation
                    generated_tokens = generate_sample(prompt_tokens)
                    
                    # Convert to text
                    prompt_text = self.tokenizer.detokenize(prompt_tokens)
                    full_text = self.tokenizer.detokenize(generated_tokens)
                    
                    samples.append({
                        "prompt": prompt_text,
                        "generated": full_text[len(prompt_text):]
                    })
            else:
                # Generate from fixed prompts
                prompts = [
                    "Once upon a time",
                    "The quick brown fox",
                    "In a galaxy far, far away"
                ]
                
                for i in range(min(num_samples, len(prompts))):
                    prompt_text = prompts[i]
                    prompt_tokens = self.tokenizer.tokenize(prompt_text)
                    
                    # Generate continuation
                    generated_tokens = generate_sample(prompt_tokens)
                    
                    # Convert to text
                    full_text = self.tokenizer.detokenize(generated_tokens)
                    
                    samples.append({
                        "prompt": prompt_text,
                        "generated": full_text[len(prompt_text):]
                    })
            
            # Format samples for logging
            formatted_samples = []
            for i, sample in enumerate(samples):
                formatted_samples.append(f"Sample {i+1}:\nPrompt: {sample['prompt']}\nGenerated: {sample['generated']}\n")
            
            # Log samples
            self.logger_manager.log_text_samples(step, formatted_samples)
            
        except Exception as e:
            self.logger.warning(f"Error generating samples: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train a language model with MLX')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Optional run ID (timestamp will be used if not provided)')
    parser.add_argument('--log-interval', type=int, default=None,
                       help='Override logging interval from config (number of steps between logs)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training')
    parser.add_argument('--precision', choices=['float16', 'bfloat16'], default='float16',
                       help='Precision to use for mixed precision training')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='Enable gradient checkpointing to save memory')
    parser.add_argument('--find-lr', action='store_true',
                       help='Run learning rate finder before training')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default=None,
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Weights & Biases entity name')
    args = parser.parse_args()
    
    # Make 'runs' directory if it doesn't exist
    os.makedirs('runs', exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command line overrides
    config_modified = False
    
    # Add run_id to config if provided
    if args.run_id:
        config['run_id'] = args.run_id
        config_modified = True
    
    # Override logging interval if provided
    if args.log_interval is not None:
        if 'logging' not in config:
            config['logging'] = {}
        if 'steps' not in config['logging']:
            config['logging']['steps'] = {}
        config['logging']['steps']['logging_interval'] = args.log_interval
        config_modified = True
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        if 'system' not in config:
            config['system'] = {}
        config['system']['mixed_precision'] = True
        config['system']['precision'] = args.precision
        config_modified = True
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        if 'system' not in config:
            config['system'] = {}
        config['system']['gradient_checkpointing'] = True
        config_modified = True
    
    # Enable learning rate finder if requested
    if args.find_lr:
        if 'training' not in config:
            config['training'] = {}
        if 'lr_finder' not in config['training']:
            config['training']['lr_finder'] = {}
        config['training']['lr_finder']['enabled'] = True
        config_modified = True
    
    # Enable TensorBoard logging if requested
    if args.tensorboard:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['tensorboard'] = True
        config_modified = True
    
    # Enable Weights & Biases logging if requested
    if args.wandb:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['wandb'] = True
        if args.wandb_project:
            config['logging']['wandb_project'] = args.wandb_project
        if args.wandb_entity:
            config['logging']['wandb_entity'] = args.wandb_entity
        config_modified = True
            
    # Write to temporary config file if modified
    config_path = args.config
    if config_modified:
        temp_config_path = f"{args.config}.tmp"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        config_path = temp_config_path
    
    trainer = Trainer(config_path)
    trainer.train()
    
    # Clean up temporary config if created
    if config_modified and os.path.exists(f"{args.config}.tmp"):
        os.remove(f"{args.config}.tmp")

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Backwards-compatibility helper
# -----------------------------------------------------------------------------
# Some auxiliary scripts (e.g. scripts/run_training.py) expect `core.training`
# to expose a top-level function called `train(config)` that kicks off the
# training loop.  Historically this function existed in an earlier version of
# the codebase, but the logic has since been refactored into the `Trainer`
# class above.  To avoid touching all call-sites we provide a thin wrapper that
# instantiates a `Trainer` and delegates to its `train()` method.
#
# The wrapper accepts either
#   • a dictionary holding the configuration, or
#   • a string/Path pointing at a YAML configuration file.
#
# When a dictionary is supplied we write it to a temporary YAML file so that we
# can continue to reuse the existing `Config.from_yaml` loader without
# duplicating the parsing logic.  The temporary file is cleaned up once
# training has started (after `Trainer.train()` returns or raises).
# -----------------------------------------------------------------------------


def train(config):
    """Entry-point retained for legacy compatibility.

    Parameters
    ----------
    config : dict | str | pathlib.Path
        • If *dict*, the configuration contents.
        • If *str* or *Path*, path to a YAML configuration file.
    """

    import tempfile
    import os
    from pathlib import Path
    import yaml  # local dependency already imported at module top-level

    # Determine whether we were given a path or a config dictionary.
    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        trainer = Trainer(str(config_path))
        trainer.train()
        return

    # Otherwise treat *config* as a mapping and materialise it on disk.
    if not isinstance(config, dict):
        raise TypeError(
            "'config' must be either a dict or a path to a YAML file; "
            f"got {type(config).__name__}."
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(config, tmp)
        tmp_path = tmp.name

    try:
        trainer = Trainer(tmp_path)
        trainer.train()
    finally:
        # Best-effort clean-up of the temporary file.
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
