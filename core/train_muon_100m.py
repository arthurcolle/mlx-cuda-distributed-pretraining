#!/usr/bin/env python3
"""
Simplified training script for a 100M parameter Muon language model
"""

import json
import os
import random
from pathlib import Path
import yaml
import time
from datetime import datetime
import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

# Import custom optimizers
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from optimizers.muon import Muon

try:
    from models.llama import Model, ModelArgs
    USING_FLASH_ATTENTION = True
    print("Using custom Llama implementation with FlashAttention")
except ImportError:
    USING_FLASH_ATTENTION = False
    print("Flash Attention not available, falling back to standard implementation")
    try:
        from mlx_lm.models.llama import Model, ModelArgs
    except ImportError:
        raise ImportError("Cannot import any Llama model implementation. Please install mlx-lm or ensure the custom implementation is available.")

# Configure basic settings
RUN_NAME = "Muon-100M"
CHECKPOINT_DIR = "checkpoints/muon-100m"
OUTPUT_DIR = f"runs/{RUN_NAME}"

# Create necessary directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
log_file = Path(OUTPUT_DIR) / "log.txt"
config_file = Path(OUTPUT_DIR) / "config.yaml"

def log_message(message):
    """Write a message to the log file and print it."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(log_file, "a") as f:
        f.write(log_line + "\n")

def load_tokenizer(tokenizer_path):
    """Load a pretrained tokenizer."""
    tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
    
    if not os.path.exists(tokenizer_file):
        raise ValueError(f"Tokenizer file not found at {tokenizer_file}")
    
    log_message(f"Loading tokenizer from {tokenizer_file}")
    return Tokenizer.from_file(tokenizer_file)

def get_model_args(vocab_size):
    """Configure model architecture for a 100M parameter model."""
    model_args = {
        # Approximately 100M parameters
        "vocab_size": vocab_size,
        "hidden_size": 768,         # Embedding dimension
        "intermediate_size": 2048,  # MLP intermediate size
        "num_hidden_layers": 12,    # Number of transformer layers
        "num_attention_heads": 12,  # Number of attention heads
        "rms_norm_eps": 1e-5,       # RMSNorm epsilon
        "max_position_embeddings": 2048,  # Maximum sequence length
        "rope_theta": 10000,        # RoPE base frequency
    }
    
    # Add Flash Attention parameters if available
    if USING_FLASH_ATTENTION:
        model_args["use_flash_attention"] = True
        model_args["flash_block_size"] = 128
    
    return ModelArgs(**model_args)

def create_llm_model(vocab_size):
    """Create LLM model based on Llama architecture."""
    model_args = get_model_args(vocab_size)
    model = Model(model_args)
    
    # Count parameters
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    log_message(f"Model created with {total_params/1e6:.2f}M parameters")
    
    return model

def loss_fn(model, inputs):
    """Standard language modeling loss function."""
    # Extract inputs and targets
    input_ids = inputs[:, :-1]
    target_ids = inputs[:, 1:]
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss - ignoring padded tokens
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), 
        target_ids.reshape(-1),
        reduction="mean"
    )
    
    return loss

def eval_loss(model, data_loader, num_batches=10):
    """Evaluate model on validation data."""
    total_loss = 0.0
    for i in range(num_batches):
        batch = data_loader.get_validation_batch()
        loss = loss_fn(model, batch)
        total_loss += loss.item()
    return total_loss / num_batches

def create_optimizer(model, learning_rate, weight_decay=0.01, betas=(0.9, 0.95)):
    """Create Muon optimizer with learning rate schedule."""
    # Create a learning rate schedule
    warmup_steps = 1000
    total_steps = 20000
    
    # Linear warmup followed by cosine decay
    warmup = optim.linear_schedule(0, learning_rate, steps=warmup_steps)
    cosine = optim.cosine_decay(learning_rate, total_steps, learning_rate * 0.1)
    schedule = optim.join_schedules([warmup, cosine], [warmup_steps])
    
    # Create Muon optimizer
    optimizer = Muon(
        learning_rate=schedule,
        betas=betas,
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    return optimizer, schedule, total_steps

class DataLoader:
    """Simple data loader for tokenized documents."""
    def __init__(self, train_file, val_file, tokenizer, batch_size=32, context_size=2048):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.context_size = context_size
        self.train_data = []
        self.val_data = []
        
        # Load training data
        log_message(f"Loading training data from {train_file}")
        self._load_data(train_file, self.train_data)
        
        # Load validation data
        if val_file:
            log_message(f"Loading validation data from {val_file}")
            self._load_data(val_file, self.val_data)
        
        # Initialize batch indices
        self.train_idx = 0
        
        log_message(f"Loaded {len(self.train_data)} training documents and {len(self.val_data)} validation documents")
    
    def _load_data(self, file_path, data_list):
        """Load and tokenize documents from a JSONL file."""
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc=f"Loading {file_path}"):
                item = json.loads(line)
                text = item["text"]
                
                # Simple chunking - more sophisticated chunking could be added
                for i in range(0, len(text), self.context_size // 2):
                    chunk = text[i:i + self.context_size - 2]  # Leave room for special tokens
                    if len(chunk) > 100:  # Avoid tiny chunks
                        data_list.append(chunk)
    
    def get_batch(self):
        """Get a batch of training data."""
        batch_texts = []
        
        # Select batch_size documents
        for _ in range(self.batch_size):
            idx = random.randint(0, len(self.train_data) - 1)
            batch_texts.append(self.train_data[idx])
        
        # Tokenize and pad
        return self._prepare_batch(batch_texts)
    
    def get_validation_batch(self):
        """Get a batch of validation data."""
        if not self.val_data:
            return self.get_batch()  # Fall back to training data if no validation data
        
        batch_texts = []
        
        # Select batch_size documents
        for _ in range(self.batch_size):
            idx = random.randint(0, len(self.val_data) - 1)
            batch_texts.append(self.val_data[idx])
        
        # Tokenize and pad
        return self._prepare_batch(batch_texts)
    
    def _prepare_batch(self, texts):
        """Tokenize and pad a batch of texts."""
        # Tokenize with special tokens
        encodings = [self.tokenizer.encode(text) for text in texts]
        token_ids = [enc.ids for enc in encodings]
        
        # Get maximum length in batch
        max_len = min(max(len(ids) for ids in token_ids) + 2, self.context_size)  # +2 for special tokens
        
        # Create padded batch
        batch = []
        for ids in token_ids:
            # Truncate if necessary
            ids = ids[:max_len-2]
            # Add special tokens (BOS, EOS)
            padded = [self.tokenizer.token_to_id("<bos>")] + ids + [self.tokenizer.token_to_id("<eos>")]
            # Pad to max_len
            padded = padded + [self.tokenizer.token_to_id("<pad>")] * (max_len - len(padded))
            batch.append(padded)
        
        return mx.array(batch)

def save_checkpoint(model, optimizer, step, loss, path):
    """Save model and optimizer state."""
    checkpoint_path = os.path.join(path, f"checkpoint-{step}")
    
    # Save model weights
    mx.save(f"{checkpoint_path}_model.safetensors", model.parameters())
    
    # Save optimizer state if available
    if hasattr(optimizer, "state"):
        mx.save(f"{checkpoint_path}_optimizer.safetensors", optimizer.state)
    
    # Save metadata
    metadata = {
        "step": step,
        "loss": loss,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{checkpoint_path}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    log_message(f"Saved checkpoint at step {step}")

def create_config():
    """Create a configuration for this run."""
    config = {
        "name": RUN_NAME,
        "model": {
            "architecture": "llama",
            "hidden_size": 768,
            "intermediate_size": 2048,
            "num_layers": 12,
            "num_heads": 12,
            "context_size": 2048,
            "params": "100M"
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "total_steps": 20000,
            "warmup_steps": 1000,
            "optimizer": "muon"
        },
        "logging": {
            "log_interval": 10,
            "checkpoint_interval": 500,
            "validation_interval": 100
        }
    }
    
    # Save config to file
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate text from the model."""
    # Tokenize prompt
    encoded = tokenizer.encode(prompt)
    input_ids = mx.array([encoded.ids])
    
    # Generate tokens
    output_ids = []
    for _ in range(max_length):
        logits = model(input_ids)[:, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample from logits
        next_token = mx.random.categorical(logits)
        output_ids.append(next_token.item())
        
        # Break if we hit EOS
        if output_ids[-1] == tokenizer.token_to_id("<eos>"):
            break
        
        # Append to input for next iteration
        input_ids = mx.concatenate([input_ids, mx.array([[output_ids[-1]]])], axis=1)
    
    # Decode outputs
    decoded = tokenizer.decode(output_ids)
    return decoded

def train():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a 100M parameter Muon LLM")
    parser.add_argument("--train_file", type=str, default="train.jsonl", help="Training data file (JSONL)")
    parser.add_argument("--val_file", type=str, default="val.jsonl", help="Validation data file (JSONL)")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer", help="Path to tokenizer directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use (gpu/cpu)")
    # -- distillation arguments
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Path to teacher model safetensors (e.g. runs/Muon-100M/checkpoints/step_final_model)")
    parser.add_argument("--distill_alpha", type=float, default=0.5,
                        help="Weight for teacher KL loss vs student CE loss")
    parser.add_argument("--distill_temp", type=float, default=2.0,
                        help="Temperature for distillation soft targets")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    
    # Set device
    mx.set_default_device(args.device)
    
    # Create config
    config = create_config()
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    log_message(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    # Create data loader
    data_loader = DataLoader(
        args.train_file, 
        args.val_file,
        tokenizer, 
        batch_size=args.batch_size
    )
    
    # Create model
    model = create_llm_model(vocab_size)

    # If online distillation, load + freeze teacher
    if args.teacher_checkpoint:
        # instantiate same architecture
        teacher = create_llm_model(vocab_size)
        # load teacher weights (expects a pytree/list of arrays saved via mx.save)
        teacher_params = mx.load(args.teacher_checkpoint + ".safetensors")
        # overwrite teacher parameters wholesale
        # note: create_llm_model returns a nn.Module; replace its internal pytree
        teacher = teacher.replace_parameters(teacher_params)
        teacher.eval()
        # freeze (stop grads through teacher)
        teacher = mx.stop_gradient(teacher)
    
    # Create optimizer
    optimizer, lr_schedule, total_steps = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create training function with gradient calculation
    @mx.compile
    def train_step(model, inputs, optimizer, step):
        def loss_fn(model):
            # slice off last token for inputs / first token for targets
            input_ids = inputs[:, :-1]
            target_ids = inputs[:, 1:]
            # student forward
            stud_logits = model(input_ids)             # [B, L-1, V]
            B, Lm1, V = stud_logits.shape

            # hard CE to gold tokens
            ce = nn.losses.cross_entropy(
                stud_logits.reshape(-1, V),
                target_ids.reshape(-1),
                reduction="mean"
            )

            # if teacher is present, add softened KL
            if args.teacher_checkpoint:
                # teacher forward (frozen)
                t_logits = teacher(input_ids)           # [B, L-1, V]
                T = args.distill_temp
                # log p_s(T)
                log_p = nn.log_softmax(stud_logits / T, axis=-1)
                # q_t(T) & log q_t(T)
                q = nn.softmax (t_logits      / T, axis=-1)
                log_q = nn.log_softmax(t_logits / T, axis=-1)
                # KL = E_q[ log q - log p ]
                # sum over vocab, mean over batch*seq
                kl = mx.mean(mx.sum(q * (log_q - log_p), axis=-1)) * (T*T)
                loss = args.distill_alpha * kl + (1 - args.distill_alpha) * ce
            else:
                loss = ce

            return loss

        # Calculate loss and gradients
        loss, grads = nn.value_and_grad(loss_fn)(model)

        # Update model parameters
        model = optimizer.update(model, grads)

        # Get current learning rate
        if callable(optimizer.learning_rate):
            lr = optimizer.learning_rate(step)
        else:
            lr = optimizer.learning_rate

        return model, optimizer, loss, lr
    
    # Training loop
    log_message(f"Starting training for {total_steps} steps")
    
    step = 0
    tokens_processed = 0
    validation_losses = []
    training_start_time = time.time()
    
    # Add training metadata
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "name": RUN_NAME,
            "start_time": training_start_time,
            "total_steps": total_steps,
            "validation": {
                "steps": [],
                "losses": []
            }
        }, f, indent=2)
    
    try:
        while step < total_steps:
            # Get batch
            batch = data_loader.get_batch()
            batch_tokens = batch.size
            tokens_processed += batch_tokens
            
            # Training step
            step_start = time.time()
            model, optimizer, loss, current_lr = train_step(model, batch, optimizer, step)
            step_duration = time.time() - step_start
            
            # Logging
            if step % config["logging"]["log_interval"] == 0:
                tokens_per_sec = batch_tokens / step_duration
                log_message(
                    f"Step {step}/{total_steps} | "
                    f"loss={loss.item():.4f} | "
                    f"lr={current_lr:.6f} | "
                    f"tokens/sec={tokens_per_sec:.1f} | "
                    f"tokens={tokens_processed:,}"
                )
            
            # Validation
            if step % config["logging"]["validation_interval"] == 0:
                val_start = time.time()
                val_loss = eval_loss(model, data_loader)
                validation_losses.append(val_loss)
                
                log_message(
                    f"Validation at step {step}/{total_steps} | "
                    f"val_loss={val_loss:.4f} | "
                    f"time={(time.time() - val_start):.1f}s"
                )
                
                # Update metadata
                with open(os.path.join(OUTPUT_DIR, "metadata.json"), "r") as f:
                    metadata = json.load(f)
                
                metadata["validation"]["steps"].append(step)
                metadata["validation"]["losses"].append(val_loss)
                
                with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Generate sample text
                if step > 0:
                    sample = generate_text(model, tokenizer, "The future of artificial intelligence will")
                    log_message(f"Sample generation:\nPrompt: The future of artificial intelligence will\nGenerated: {sample}")
            
            # Checkpointing
            if step % config["logging"]["checkpoint_interval"] == 0 and step > 0:
                save_checkpoint(model, optimizer, step, loss.item(), CHECKPOINT_DIR)
            
            step += 1
    
    except KeyboardInterrupt:
        log_message("Training interrupted by user")
    
    # Final checkpoint
    save_checkpoint(model, optimizer, step, loss.item(), CHECKPOINT_DIR)
    
    # Final validation
    final_val_loss = eval_loss(model, data_loader)
    log_message(f"Final validation loss: {final_val_loss:.4f}")
    
    # Final sample generation
    sample = generate_text(model, tokenizer, "The future of artificial intelligence will")
    log_message(f"Final sample generation:\nPrompt: The future of artificial intelligence will\nGenerated: {sample}")
    
    training_time = time.time() - training_start_time
    log_message(f"Training completed in {training_time/3600:.2f} hours")
    
    # Save model in MLX-LM compatible format
    mlx_lm_dir = os.path.join(OUTPUT_DIR, "mlx-lm")
    os.makedirs(mlx_lm_dir, exist_ok=True)
    
    # Save model weights
    mx.save(os.path.join(mlx_lm_dir, "weights.safetensors"), model.parameters())
    
    # Copy tokenizer
    tokenizer_dir = os.path.join(mlx_lm_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    os.system(f"cp {args.tokenizer_path}/tokenizer.json {tokenizer_dir}/tokenizer.json")
    
    log_message(f"Saved model in MLX-LM format at {mlx_lm_dir}")

if __name__ == "__main__":
    train()
