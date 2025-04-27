import argparse
import os
import time
from pathlib import Path
import json
import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

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

def load_model_and_tokenizer(run_name):
    """Load a trained model and its tokenizer."""
    run_dir = Path("runs") / run_name
    
    # Check if run directory exists
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")
    
    # Load metadata
    with open(run_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Get model info
    model_info = metadata["model_info"]
    
    # Create model
    model = SimpleLanguageModel(
        vocab_size=model_info["vocab_size"],
        hidden_size=model_info["hidden_size"],
        num_layers=model_info["num_layers"],
        num_heads=model_info["num_heads"]
    )
    
    # Load model weights
    checkpoint_path = run_dir / "checkpoints" / "final_model.safetensors"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoints" / "step_final_model.safetensors"
    
    if not checkpoint_path.exists():
        # Try to find the latest epoch checkpoint
        checkpoints = list(run_dir.glob("checkpoints/epoch_*_model.safetensors"))
        if not checkpoints:
            checkpoints = list(run_dir.glob("checkpoints/step_*_model.safetensors"))
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {run_dir / 'checkpoints'}")
        
        # Sort by step number
        checkpoint_path = sorted(checkpoints, key=lambda x: int(x.stem.split("_")[1]))[-1]
    
    print(f"Loading model weights from {checkpoint_path}")
    model.load_weights(str(checkpoint_path))
    
    # Load tokenizer
    tokenizer_path = run_dir / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError(f"Tokenizer not found: {tokenizer_path}")
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return model, tokenizer, metadata

def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=1.0):
    """Generate text using the model."""
    # Tokenize the prompt
    encoded = tokenizer.encode(prompt)
    tokens = mx.array(encoded.ids)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate tokens one by one
    for _ in range(max_tokens):
        # Get model output
        logits = model(tokens.reshape(1, -1))
        
        # Get the last token's logits
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            # Convert to probabilities
            probs = mx.softmax(next_token_logits, axis=-1)
            # Sample from the distribution
            next_token = mx.random.categorical(probs.reshape(1, -1))
        else:
            # Greedy decoding
            next_token = mx.argmax(next_token_logits)
        
        # Append to the sequence
        tokens = mx.concatenate([tokens, next_token.reshape(1)])
        
        # Check if we've generated an EOS token
        if next_token.item() == tokenizer.token_to_id("</s>"):
            break
    
    # Decode the generated tokens
    output = tokenizer.decode(tokens.tolist())
    
    return output

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained model")
    parser.add_argument("--run", type=str, required=True, help="Name of the training run")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to start generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Set default device
    if args.device == "gpu" and mx.gpu_is_available():
        mx.set_default_device(mx.gpu)
        print("Using GPU for generation")
    else:
        mx.set_default_device(mx.cpu)
        print("Using CPU for generation")
    
    # Set random seed if provided
    if args.seed is not None:
        mx.random.seed(args.seed)
    else:
        mx.random.seed(int(time.time() * 1000))
    
    # Load model and tokenizer
    model, tokenizer, metadata = load_model_and_tokenizer(args.run)
    
    # Print model info
    model_info = metadata["model_info"]
    print(f"Model has {model_info['parameters']:,} parameters")
    print(f"Vocabulary size: {model_info['vocab_size']}")
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("Generating...")
    
    output = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print(f"\nGenerated text:\n{output}")

if __name__ == "__main__":
    main()
