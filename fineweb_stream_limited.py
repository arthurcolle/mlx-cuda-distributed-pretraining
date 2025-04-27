import json
import random
import os
import time
import argparse
import numpy as np
import shutil
import logging
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
import mlx.core as mx
import mlx.nn as nn
import yaml
from tqdm import tqdm
from train import Trainer, CheckpointManager
from mlx.utils import tree_map

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DiskSpaceManager:
    """Manages disk space by tracking usage and cleaning up when necessary."""
    
    def __init__(self, max_disk_usage_gb=35, cache_dir=None, reserved_gb=5):
        self.max_disk_usage_gb = max_disk_usage_gb
        self.reserved_gb = reserved_gb
        
        # Create temp directory for caching if not provided
        if cache_dir is None:
            self.cache_dir = Path(os.environ.get("TEMP_DIR", "/tmp")) / "fineweb_cache"
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.auto_created = True
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.auto_created = False
            
        print(f"Using cache directory: {self.cache_dir}")
        
        # Track files and their access times
        self.tracked_files = {}
    
    def get_disk_usage(self, path):
        """Get disk usage of a path in GB."""
        if not os.path.exists(path):
            logging.warning(f"Path {path} does not exist. Cannot calculate disk usage.")
            return 0
            
        total_size = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        try:
                            total_size += os.path.getsize(fp)
                        except (FileNotFoundError, PermissionError) as e:
                            logging.debug(f"Cannot access file {fp}: {str(e)}")
        except Exception as e:
            logging.error(f"Error calculating disk usage for {path}: {str(e)}")
            
        return total_size / (1024 * 1024 * 1024)  # Convert bytes to GB
    
    def check_and_clean(self):
        """Check disk usage and clean up if necessary."""
        current_usage = self.get_disk_usage(self.cache_dir)
        
        if current_usage > self.max_disk_usage_gb:
            print(f"Cache usage ({current_usage:.2f}GB) exceeds limit ({self.max_disk_usage_gb}GB). Cleaning...")
            self.clean_cache()
    
    def clean_cache(self):
        """Clean oldest cached files until we're under the limit."""
        if not self.tracked_files:
            logging.info("No tracked files to clean.")
            return
            
        # Sort by last access time (oldest first)
        sorted_files = sorted(self.tracked_files.items(), key=lambda x: x[1])
        
        current_usage = self.get_disk_usage(self.cache_dir)
        target_usage = self.max_disk_usage_gb - self.reserved_gb  # Leave some buffer
        
        # Create a copy of file paths to avoid modifying dictionary during iteration
        files_to_remove = []
        
        for filepath, _ in sorted_files:
            if current_usage <= target_usage:
                break
                
            if os.path.exists(filepath):
                file_size_gb = os.path.getsize(filepath) / (1024 * 1024 * 1024)
                files_to_remove.append((filepath, file_size_gb))
        
        # Now remove the files and update the tracking dictionary
        for filepath, file_size_gb in files_to_remove:
            try:
                os.remove(filepath)
                logging.info(f"Removed cached file: {filepath} ({file_size_gb:.2f}GB)")
                current_usage -= file_size_gb
                # Add to removal list
                self.tracked_files.pop(filepath, None)
            except Exception as e:
                logging.error(f"Error removing {filepath}: {e}")
                # Still remove from tracking if file doesn't exist or can't be accessed
                self.tracked_files.pop(filepath, None)
    
    def track_file(self, filepath):
        """Track a file and update its access time."""
        self.tracked_files[filepath] = time.time()
    
    def cleanup(self):
        """Final cleanup when done."""
        if self.auto_created and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            print(f"Removed temporary cache directory: {self.cache_dir}")

class FineWebStreamDataset:
    """
    Streaming dataset for FineWeb that processes data on-the-fly
    without requiring the entire dataset to be loaded into memory.
    """
    
    def __init__(self, tokenizer, language="eng_Latn", max_context_size=2048, 
                 shuffle_buffer=10000, disk_manager=None, limit=None, cache_dir=None):
        self.language = language
        self.tokenizer = tokenizer
        self.max_context_size = max_context_size
        self.shuffle_buffer = shuffle_buffer
        self.disk_manager = disk_manager
        self.limit = limit
        self.processed_count = 0  # Track number of processed examples
        
        logging.info(f"Loading FineWeb dataset for language: {language}")
        
        # Use Hugging Face's cache management to control disk usage
        try:
            self.dataset = load_dataset(
                "HuggingFaceFW/fineweb-2", 
                name=language, 
                split="train", 
                streaming=True,
                cache_dir=cache_dir,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HF_TOKEN")  # In case authentication is needed
            )
            
            if shuffle_buffer:
                self.dataset = self.dataset.shuffle(buffer_size=shuffle_buffer, seed=42)
                
            if limit:
                self.dataset = self.dataset.take(limit)
                
        except Exception as e:
            logging.error(f"Error loading FineWeb dataset: {str(e)}")
            raise
        
    def __iter__(self):
        """Iterate over the streaming dataset."""
        for i, sample in enumerate(self.dataset):
            # Check disk usage periodically
            if self.disk_manager and i % 1000 == 0:
                self.disk_manager.check_and_clean()
            
            # Process the text
            if 'text' in sample and sample['text']:
                tokens = self.tokenize_text(sample['text'])
                if tokens:
                    yield {'tokens': tokens}
    
    def tokenize_text(self, text):
        """Tokenize text with the provided tokenizer."""
        try:
            # Apply tokenizer's method for document tokenization
            tokens = self.tokenizer.tokenize_doc(text)
            
            # Ensure we don't exceed max context size
            if len(tokens) > self.max_context_size + 2:  # +2 for BOS/EOS
                tokens = tokens[:self.max_context_size + 2]
                
            return tokens
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return None

def create_dataloader(dataset, batch_size, num_workers=4, prefetch_factor=2):
    """Create a batched data loader for the streaming dataset."""
    from torch.utils.data import DataLoader
    
    # Transform the dataset into a format compatible with PyTorch DataLoader
    class DatasetAdapter:
        def __init__(self, dataset):
            self.dataset = dataset
            self.iterator = iter(dataset)
            
        def __iter__(self):
            return self
            
        def __next__(self):
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
                return next(self.iterator)
    
    return DataLoader(
        DatasetAdapter(dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

def fineweb_to_mlx(batch):
    """Convert batch to MLX arrays."""
    if isinstance(batch, dict) and 'tokens' in batch:
        # Convert tokens to MLX array
        return mx.array(batch['tokens'])
    # Handle case where batch might be structured differently
    if hasattr(batch, 'tokens'):
        return mx.array(batch.tokens)
    return mx.array(batch)

def stream_training_loop(
    config_path, 
    language="eng_Latn",
    num_workers=4, 
    prefetch_factor=2, 
    max_disk_usage_gb=35,
    limit=None
):
    """Run stream processing training on FineWeb data with limited disk space."""
    
    # Initialize trainer from config
    trainer = Trainer(config_path)
    
    # Create disk space manager
    disk_manager = DiskSpaceManager(max_disk_usage_gb=max_disk_usage_gb)
    
    # Create streaming dataset
    dataset = FineWebStreamDataset(
        tokenizer=trainer.tokenizer,
        language=language,
        max_context_size=trainer.config.data.preprocessing['max_context_size'],
        shuffle_buffer=10000,
        disk_manager=disk_manager,
        limit=limit
    )
    
    # Create PyTorch dataloader for streaming
    dataloader = create_dataloader(
        dataset,
        batch_size=trainer.config.training.hyperparameters['batch_size'],
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    
    # Training parameters
    total_steps = trainer.config.training.hyperparameters['iters']
    grad_accum_steps = trainer.config.training.hyperparameters.get('gradient_accumulation_steps', 1)
    
    # Setup directories
    run_dir = Path('runs') / trainer.config.name
    log_file = run_dir / 'log.txt'
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup optimizer and scheduler
    optimizer = trainer.optimizer
    lr_schedule = trainer.lr_schedule
    
    # Training state
    total_tokens = 0
    accumulated_gradients = None
    accumulated_tokens = 0
    accum_step = 0
    
    # Create value_and_grad function
    def compute_loss(model, inputs, targets):
        logits = model(inputs)
        logits = logits.astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, targets)
        
        # Mask padding tokens
        pad_mask = (targets != trainer.tokenizer.PAD_TOKEN)
        loss = loss * pad_mask
        ntoks = pad_mask.sum()
        
        return loss.sum() / ntoks, ntoks
    
    # Value and gradient function
    loss_value_and_grad = nn.value_and_grad(trainer.model, compute_loss)
    
    # Progress tracking
    start_time = time.time()
    progress_bar = tqdm(range(total_steps), desc="Training")
    
    # Set up validation at appropriate intervals
    validation_steps = trainer.config.logging.steps.get('validation_interval', 0)
    
    try:
        # Streaming training loop
        with open(log_file, 'w') as log:
            log.write(f"Training started at {datetime.now()}\n")
            log.write(f"Total steps: {total_steps}\n")
            log.write(f"Streaming FineWeb language: {language}\n")
            log.write(f"Max disk usage: {max_disk_usage_gb}GB\n")
            log.write("=" * 50 + "\n\n")
            
            # Initialize data iterator - continuously stream data
            stream_iterator = iter(dataloader)
            
            for step in progress_bar:
                try:
                    # Get next batch from stream
                    batch = next(stream_iterator)
                except StopIteration:
                    # If we reach the end, reset the iterator
                    stream_iterator = iter(dataloader)
                    batch = next(stream_iterator)
                
                # Convert to MLX format
                mlx_batch = fineweb_to_mlx(batch)
                
                # Forward and backward pass
                (loss, tokens), grad = loss_value_and_grad(
                    trainer.model, mlx_batch[:, :-1], mlx_batch[:, 1:]
                )
                
                # Gradient clipping if configured
                if 'gradient_clip' in trainer.config.training.hyperparameters:
                    clip_value = trainer.config.training.hyperparameters['gradient_clip']
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
                    if accum_step == grad_accum_steps or step == total_steps - 1:
                        # Update model with accumulated gradients
                        total_tokens += accumulated_tokens
                        optimizer.update(trainer.model, accumulated_gradients)
                        mx.eval(loss)
                        
                        # Reset accumulation
                        accumulated_gradients = None
                        accumulated_tokens = 0
                        accum_step = 0
                else:
                    # Standard update without accumulation
                    total_tokens += tokens
                    optimizer.update(trainer.model, grad)
                    mx.eval(loss)
                
                # Clear GPU cache if on GPU
                if not trainer.distributed and trainer.config.system.device == "gpu":
                    try:
                        mx.clear_cache()
                    except AttributeError:
                        # mx.clear_cache() might not be available in this version
                        pass
                
                # Run validation if trainer has validation data
                val_loss = None
                if validation_steps > 0 and trainer.data_manager.has_validation_data and (step + 1) % validation_steps == 0:
                    val_loss = trainer.validate()
                    # Add to validation loss history
                    trainer.validation_losses.append((step + 1, val_loss))
                    
                    # Log validation separately for clear visibility
                    val_metrics = f"val_loss={val_loss:.3e} | val_ppl={np.exp(val_loss):.2f}"
                    log.write(f"Step {step + 1} validation: {val_metrics}\n")
                    log.flush()
                
                # Logging
                if step % trainer.config.logging.steps['logging_interval'] == 0:
                    # Only include val_loss if it was just calculated
                    current_val_loss = val_loss if validation_steps > 0 and (step + 1) % validation_steps == 0 else None
                    metrics = trainer.log_metrics(step, loss, tokens, total_tokens, start_time, current_val_loss)
                    
                    # Update progress bar
                    progress_bar.set_description(metrics)
                    
                    # Write to log file
                    log_message = f"Step {step}: {metrics}\n"
                    log.write(log_message)
                    log.flush()
                
                # Save checkpoint
                if (1 + step) % trainer.config.logging.steps['checkpoint_interval'] == 0:
                    # Create checkpoint directory if it doesn't exist
                    os.makedirs(trainer.checkpoint_dir, exist_ok=True)
                    
                    # Update total_tokens in the trainer instance for checkpoint saving
                    trainer.total_tokens = total_tokens
                    trainer.save_checkpoint(step + 1, val_loss if 'val_loss' in locals() else None)
                    
                    # Check disk usage after saving checkpoints
                    if disk_manager:
                        disk_manager.check_and_clean()
        
        # Save final model
        trainer.total_tokens = total_tokens
        trainer.save_checkpoint("final", val_loss if 'val_loss' in locals() else None)
        
        print(f"Training complete! Model saved to {trainer.checkpoint_dir}")
        return trainer
        
    finally:
        # Clean up cache
        if disk_manager:
            disk_manager.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stream training on FineWeb dataset with limited disk space')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--language', type=str, default="eng_Latn", 
                        help='FineWeb language code (e.g., "eng_Latn", "fra_Latn", "deu_Latn")')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--prefetch', type=int, default=2, help='Prefetch factor for dataloader')
    parser.add_argument('--max-disk', type=float, default=35, help='Maximum disk usage in GB for caching')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples to process')
    
    args = parser.parse_args()
    
    stream_training_loop(
        config_path=args.config,
        language=args.language,
        num_workers=args.workers,
        prefetch_factor=args.prefetch,
        max_disk_usage_gb=args.max_disk,
        limit=args.limit
    )