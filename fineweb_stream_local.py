import json
import random
import os
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds
import mlx.core as mx
import mlx.nn as nn
import yaml
from tqdm import tqdm
from train import Trainer, CheckpointManager
from mlx.utils import tree_map

class FineWebStreamingDataset(IterableDataset):
    """
    Streaming dataset for FineWeb that processes data on-the-fly
    without requiring the entire dataset to be loaded into memory.
    """
    
    def __init__(self, file_pattern, tokenizer, max_context_size=2048, shuffle_buffer=10000):
        self.file_pattern = file_pattern
        self.tokenizer = tokenizer
        self.max_context_size = max_context_size
        self.shuffle_buffer = shuffle_buffer
        
    def __iter__(self):
        # Create WebDataset pipeline
        dataset = (
            wds.WebDataset(self.file_pattern)     # Stream from files using local paths
            .shuffle(self.shuffle_buffer)          # Local shuffle buffer
            .decode()                              # Decode from bytes
            .map(self.process_sample)              # Process and tokenize samples
        )
        
        # Return iterator
        yield from dataset
    
    def process_sample(self, sample):
        """Process a single sample from WebDataset."""
        if 'json' in sample:
            # Parse JSON content
            data = json.loads(sample['json'])
            text = data.get('text', '')
            
            # Tokenize text
            tokens = self.tokenizer.tokenize_doc(text)
            
            # Ensure we don't exceed max context size
            if len(tokens) > self.max_context_size + 2:  # +2 for BOS/EOS
                tokens = tokens[:self.max_context_size + 2]
                
            return {'tokens': tokens}
        return None

def create_dataloader(dataset, batch_size, num_workers=4, prefetch_factor=2):
    """Create a DataLoader for the streaming dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

def fineweb_to_mlx(batch):
    """Convert PyTorch batch to MLX arrays."""
    if isinstance(batch, dict) and 'tokens' in batch:
        # Convert tokens to MLX array
        return mx.array(batch['tokens'])
    # Handle case where batch might be structured differently
    return mx.array(batch)

def stream_training_loop(config_path, file_pattern, num_workers=4, prefetch_factor=2):
    """Run stream processing training on FineWeb data."""
    
    # Initialize trainer from config
    trainer = Trainer(config_path)
    
    # Create streaming dataset
    dataset = FineWebStreamingDataset(
        file_pattern=file_pattern,
        tokenizer=trainer.tokenizer,
        max_context_size=trainer.config.data.preprocessing['max_context_size'],
        shuffle_buffer=10000
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
    
    # Streaming training loop
    with open(log_file, 'w') as log:
        log.write(f"Training started at {datetime.now()}\n")
        log.write(f"Total steps: {total_steps}\n")
        log.write(f"Streaming from: {file_pattern}\n")
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
    
    # Save final model
    trainer.total_tokens = total_tokens
    trainer.save_checkpoint("final", val_loss if 'val_loss' in locals() else None)
    
    print(f"Training complete! Model saved to {trainer.checkpoint_dir}")
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stream training on FineWeb dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--file-pattern', type=str, required=True, 
                      help='Local file pattern for FineWeb data (e.g., datasets/fineweb/shard-*.tar.bz2)')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--prefetch', type=int, default=2, help='Prefetch factor for dataloader')
    
    args = parser.parse_args()
    
    stream_training_loop(
        config_path=args.config,
        file_pattern=args.file_pattern,
        num_workers=args.workers,
        prefetch_factor=args.prefetch
    )