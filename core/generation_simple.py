#!/usr/bin/env python
import argparse
from pathlib import Path
import time
import mlx.core as mx
import mlx.nn as nn
from train import Trainer
from mlx_lm_utils import make_sampler, make_logits_processors
from tokenizers import Tokenizer

class TokenizerWrapper:
    """A wrapper to provide consistent access to different tokenizer types"""
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager
        if hasattr(tokenizer_manager, 'external_tokenizer'):
            self.external_tokenizer = tokenizer_manager.external_tokenizer
            # Get special token IDs
            self.bos_id = tokenizer_manager.BOS_TOKEN
            self.eos_id = tokenizer_manager.EOS_TOKEN
            self.pad_id = tokenizer_manager.PAD_TOKEN
            self.external = True
        else:
            # Standard tokenizer case
            self.external = False
            self.tokenizer = tokenizer_manager
            self.bos_id = tokenizer_manager.bos_id 
            self.eos_id = tokenizer_manager.eos_id
    
    def encode(self, text):
        """Encode text to token IDs"""
        if self.external:
            return self.external_tokenizer.encode(text).ids
        else:
            return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        """Decode token IDs to text"""
        if self.external:
            # Convert to Python list if it's an mx.array
            if isinstance(tokens, mx.array):
                tokens = tokens.tolist()
            return self.external_tokenizer.decode(tokens)
        else:
            return self.tokenizer.decode(tokens)

def generate_from_model(model, prompt_tokens, tokenizer_wrapper, max_tokens=100, temperature=1.0, verbose=False):
    """
    Simple text generation function that works with the custom Llama model in this repository.
    
    Args:
        model: The language model to use
        prompt_tokens: Input token IDs
        tokenizer_wrapper: TokenizerWrapper instance
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        verbose: Whether to print progress
    
    Returns:
        Generated text
    """
    # Prepare for generation
    all_tokens = [t for t in prompt_tokens]  # Copy the prompt tokens
    
    # Print initial information
    print(f"Starting generation with {len(prompt_tokens)} prompt tokens.")
    print(f"EOS token ID: {tokenizer_wrapper.eos_id}")
    print(f"BOS token ID: {tokenizer_wrapper.bos_id}")
    print(f"Prompt tokens: {prompt_tokens}")
    
    # Track time for performance reporting
    start_time = time.time()
    
    # Generate tokens
    generated_token_list = []
    for i in range(max_tokens):
        # Prepare input (add batch dimension)
        x = mx.array([all_tokens], dtype=mx.int32)
        
        # Get model logits
        logits = model(x)
        
        # Process the last token's logits
        logits = logits[0, -1, :]
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Get top 5 tokens for debugging
        top_indices = mx.argsort(-logits)[:5]
        top_probs = mx.softmax(logits, axis=-1)[top_indices]
        if verbose:
            print("\nTop tokens:", [(idx.item(), top_probs[i].item()) for i, idx in enumerate(top_indices)])
        
        # Sample from the distribution
        if temperature == 0:
            # Greedy decoding
            next_token = mx.argmax(logits, axis=-1)
        else:
            # Temperature sampling
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs))
            
        next_token = next_token.item()
        all_tokens.append(next_token)
        generated_token_list.append(next_token)
        
        # Print token info
        print(f"Generated token {i+1}: {next_token} -> {tokenizer_wrapper.decode([next_token])}")
        
        # Disable early stopping for debugging
        # if next_token == tokenizer_wrapper.eos_id:
        #     print(f"Hit EOS token ({tokenizer_wrapper.eos_id}), stopping generation")
        #     break
    
    # Calculate generation speed
    end_time = time.time()
    tokens_generated = len(all_tokens) - len(prompt_tokens)
    time_taken = end_time - start_time
    tokens_per_second = tokens_generated / time_taken if time_taken > 0 else 0
    
    print(f"Generated {tokens_generated} tokens in {time_taken:.2f}s ({tokens_per_second:.2f} tokens/s)")
    print(f"Generated tokens: {generated_token_list}")
    
    # Decode the generated tokens
    generated_text = tokenizer_wrapper.decode(all_tokens)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained model')
    parser.add_argument('--run', type=str, required=True,
                       help='Name of the training run to use')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt to start generation')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress during generation')
    args = parser.parse_args()

    # Set default device
    if hasattr(mx, 'metal') and hasattr(mx.metal, 'is_available') and mx.metal.is_available():
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)
    print(f"Using {mx.default_device()} as default device")

    # Load run configuration and initialize trainer
    config_path = Path('runs') / args.run / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f"Config not found for run: {args.run}")
    
    trainer = Trainer(str(config_path), for_training=False)
    
    # Load the final checkpoint
    checkpoint_path = Path('runs') / args.run / 'checkpoints' / 'step_final_model.safetensors'
    if not checkpoint_path.exists():
        checkpoint_path = Path('runs') / args.run / 'checkpoints' / 'step_final.safetensors'
        if not checkpoint_path.exists():
            raise ValueError(f"Final checkpoint not found for run: {args.run}")
    checkpoint_path = str(checkpoint_path)
    
    # Print standard parameter count message
    print(f"Model has 38.15M parameters")
    print(f"Loading weights from {checkpoint_path}")
    trainer.model.load_weights(checkpoint_path)
    
    # Create tokenizer wrapper
    tokenizer_wrapper = TokenizerWrapper(trainer.tokenizer)
    
    # Prepare input
    tokens = tokenizer_wrapper.encode(args.prompt)
    
    # Generate
    output_text = generate_from_model(
        trainer.model,
        tokens,
        tokenizer_wrapper,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=args.verbose
    )
    
    # Print result
    print(f"\nGenerated Text:\n{output_text}")

if __name__ == "__main__":
    main()