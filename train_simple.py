import argparse
import os
import json
import time
import random
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
from datetime import datetime

class SimpleLanguageModel(nn.Module):
    """A simple language model with embedding, transformer layers, and output projection."""
    
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers
        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerEncoderBlock(
                    hidden_size, 
                    num_heads,
                    dropout=0.1
                )
            )
        
        # Output projection
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def __call__(self, x):
        # Create causal mask for autoregressive generation
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        
        # Embed tokens
        h = self.embed_tokens(x)
        
        # Apply transformer layers with mask
        for layer in self.layers:
            h = layer(h, mask)
        
        # Project to vocabulary
        return self.output(h)

def load_or_create_tokenizer(data_path, tokenizer_path, vocab_size=8000):
    """Load an existing tokenizer or train a new one."""
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print(f"Training new tokenizer with vocab size {vocab_size}")
        from tokenizers import models, pre_tokenizers, trainers
        
        # Create a BPE tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Create trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            min_frequency=2
        )
        
        # Prepare training files
        files = [data_path]
        
        # Train the tokenizer
        tokenizer.train(files, trainer)
        
        # Save the tokenizer
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        tokenizer.save(tokenizer_path)
        
    return tokenizer

def load_dataset(data_path, tokenizer, max_length=512, max_samples=None):
    """Load and tokenize the dataset."""
    # Read the data file
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = []
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line)
                if 'text' in item:
                    lines.append(item['text'])
            except:
                # If not JSON, just use the raw line
                lines.append(line.strip())
    
    # Tokenize the data
    tokenized_data = []
    for text in tqdm(lines, desc="Tokenizing data"):
        # Add BOS and EOS tokens
        encoded = tokenizer.encode("<s> " + text + " </s>")
        if len(encoded.ids) > max_length:
            # Truncate if too long
            encoded.ids = encoded.ids[:max_length]
        tokenized_data.append(encoded.ids)
    
    return tokenized_data

def create_batches(data, batch_size, pad_id):
    """Create batches from tokenized data."""
    # Sort by length for more efficient batching
    sorted_data = sorted(data, key=len)
    
    # Create batches
    batches = []
    for i in range(0, len(sorted_data), batch_size):
        batch = sorted_data[i:i+batch_size]
        
        # Find max length in this batch
        max_len = max(len(seq) for seq in batch)
        
        # Pad sequences
        padded_batch = []
        for seq in batch:
            padded = seq + [pad_id] * (max_len - len(seq))
            padded_batch.append(padded)
        
        # Convert to mx.array
        batches.append(mx.array(padded_batch))
    
    return batches

def compute_loss(model, batch, pad_id):
    """Compute cross-entropy loss for language modeling."""
    # Get inputs and targets (shifted by 1)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    # Forward pass
    logits = model(inputs)
    
    # Compute loss
    loss = nn.losses.cross_entropy(logits, targets)
    
    # Mask padding tokens
    mask = (targets != pad_id)
    loss = loss * mask
    
    # Return average loss
    return loss.sum() / mask.sum()

def train_model(args):
    """Train a simple language model."""
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    
    # Set default device
    if args.device == "gpu" and mx.gpu_is_available():
        mx.set_default_device(mx.gpu)
        print("Using GPU for training")
    else:
        mx.set_default_device(mx.cpu)
        print("Using CPU for training")
    
    # Create run directory
    run_name = f"SimpleModel-{args.hidden_size}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_name
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    os.makedirs(run_dir / "tokenizer", exist_ok=True)
    
    # Load or create tokenizer
    tokenizer_path = str(run_dir / "tokenizer" / "tokenizer.json")
    tokenizer = load_or_create_tokenizer(args.data_path, tokenizer_path, args.vocab_size)
    
    # Get special token IDs
    vocab = tokenizer.get_vocab()
    pad_id = vocab.get("<pad>", 0)
    bos_id = vocab.get("<s>", 1)
    eos_id = vocab.get("</s>", 2)
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = load_dataset(args.data_path, tokenizer, args.max_length, args.max_samples)
    print(f"Loaded {len(dataset)} samples")
    
    # Create training batches
    batches = create_batches(dataset, args.batch_size, pad_id)
    print(f"Created {len(batches)} batches")
    
    # Create model
    print(f"Creating model with vocab size {args.vocab_size}, hidden size {args.hidden_size}")
    model = SimpleLanguageModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    
    # Count parameters
    param_count = sum(p.size for p in model.parameters().values())
    print(f"Model has {param_count:,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    
    # Create loss function with value and gradient
    loss_fn = lambda model, batch: compute_loss(model, batch, pad_id)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    step = 0
    
    for epoch in range(args.epochs):
        # Shuffle batches
        random.shuffle(batches)
        
        # Create progress bar
        progress_bar = tqdm(batches, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        
        for batch in progress_bar:
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, batch)
            
            # Update parameters
            optimizer.update(model, grads)
            
            # Update loss
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save checkpoint
            step += 1
            if step % args.save_every == 0:
                checkpoint_path = run_dir / "checkpoints" / f"step_{step}"
                save_checkpoint(checkpoint_path, model, optimizer, step, loss.item())
        
        # Compute average loss for the epoch
        avg_loss = epoch_loss / len(batches)
        perplexity = np.exp(avg_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        # Save epoch checkpoint
        checkpoint_path = run_dir / "checkpoints" / f"epoch_{epoch+1}"
        save_checkpoint(checkpoint_path, model, optimizer, step, avg_loss)
    
    # Save final model
    final_checkpoint_path = run_dir / "checkpoints" / "final"
    save_checkpoint(final_checkpoint_path, model, optimizer, step, avg_loss)
    
    # Save metadata
    metadata = {
        "name": run_name,
        "created_at": datetime.now().isoformat(),
        "args": vars(args),
        "training_info": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "final_loss": avg_loss,
            "final_perplexity": perplexity,
            "total_steps": step,
            "training_time": time.time() - start_time
        },
        "model_info": {
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "parameters": param_count
        }
    }
    
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Model saved to {run_dir}")
    
    return run_dir, run_name

def save_checkpoint(path, model, optimizer, step, loss):
    """Save model and optimizer state."""
    # Save model weights
    model_weights = {k: v for k, v in model.parameters().items()}
    mx.save_safetensors(f"{path}_model.safetensors", model_weights)
    
    # Save optimizer state
    optimizer_state = {k: v for k, v in optimizer.state.items()}
    mx.save_safetensors(f"{path}_optimizer.safetensors", optimizer_state)
    
    # Save training state
    training_state = {
        "step": step,
        "loss": loss
    }
    
    with open(f"{path}_state.json", "w") as f:
        json.dump(training_state, f)

def main():
    parser = argparse.ArgumentParser(description="Train a simple language model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to training data file")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to use")
    
    args = parser.parse_args()
    run_dir, run_name = train_model(args)
    
    print(f"\nTo generate text with the trained model, run:")
    print(f"python generate.py --run {run_name} --prompt \"Your prompt here\"")

if __name__ == "__main__":
    main()
