import argparse
from pathlib import Path
import mlx.core as mx
from core.training import Trainer
import mlx.nn as nn
import time
from generate_lite import generate_lite, beam_search
from mlx_lm_utils import make_sampler, make_logits_processors
mx.set_default_device(mx.gpu)
def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained model')
    parser.add_argument('--run', type=str, required=True,
                       help='Name of the training run to use')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt to start generation')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--min-p', type=float, default=0.05,
                       help='Minimum probability for nucleus sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                       help='Repetition penalty factor')
    parser.add_argument('--repetition-context-size', type=int, default=20,
                       help='Context size for repetition penalty')
    parser.add_argument('--strict-loading', action='store_true',
                       help='Use strict parameter loading (default: False)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable additional debugging output')
    parser.add_argument('--force-token-id', type=int, default=None,
                       help='Force generation to use this token ID (for debugging)')
    args = parser.parse_args()

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
    
    # Load weights with strict parameter based on command line argument
    try:
        trainer.model.load_weights(checkpoint_path, strict=args.strict_loading)
        print(f"Successfully loaded weights from {checkpoint_path}")
    except Exception as e:
        print(f"Warning: Error loading weights: {e}")
        print("Attempting to continue with partially loaded weights...")
    
    # Set model to eval mode
    trainer.model.eval()
    
    # Print model information
    print(f"Model architecture: {type(trainer.model).__name__}")
    
    # Count parameters correctly - MLX parameters are stored differently
    try:
        # Try to get the parameter count from the model directly if available
        if hasattr(trainer.model, 'num_parameters'):
            param_count = trainer.model.num_parameters
        else:
            # Manual counting with recursive function to handle nested dictionaries
            def count_params(params_dict):
                count = 0
                for param in params_dict.values():
                    if isinstance(param, mx.array):
                        count += param.size
                    elif hasattr(param, 'size'):
                        count += param.size
                    elif hasattr(param, 'shape'):
                        count += mx.prod(mx.array(param.shape))
                    elif isinstance(param, dict):
                        count += count_params(param)  # Recursively count nested dictionaries
                return count
                
            param_count = count_params(trainer.model.parameters())
        
        print(f"Model has {param_count:,} parameters")
    except Exception as e:
        print(f"Could not count parameters: {e}")
        print("Continuing with generation...")
    
    # Prepare input
    tokens = [trainer.tokenizer.BOS_TOKEN] + trainer.tokenizer.tokenize(args.prompt)
    
    # Setup generation parameters
    sampler = make_sampler(temp=args.temperature, min_p=args.min_p)
    logits_processors = make_logits_processors(
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size
    )
    
    # Generate
    # Print the prompt first
    print(f"Prompt: {args.prompt}")
    
    try:
        # Set a random seed for generation
        mx.random.seed(int(time.time() * 1000))
        
        # Try with a direct approach first
        print("Trying direct generation approach...")
        
        # Ensure tokens are integers
        int_tokens = [int(t) if isinstance(t, (int, float)) else t for t in tokens]
        
        # Print token information for debugging
        if args.debug:
            print(f"Token types: {[type(t) for t in int_tokens]}")
            print(f"Tokens: {int_tokens}")
        
        # Create a very simple generation function that doesn't rely on generate_lite
        def direct_generate(model, prompt_tokens, max_new_tokens=20):
            # Convert tokens to mx.array if needed
            if not isinstance(prompt_tokens, mx.array):
                prompt_tokens = mx.array(prompt_tokens)
            
            # Initialize with prompt tokens
            all_tokens = prompt_tokens
            generated_tokens = []
            
            # Generate one token at a time
            for i in range(max_new_tokens):
                try:
                    # Forward pass through the model
                    # MLX doesn't have eval_mode context manager, use model.eval() instead
                    model.eval()
                    # Get model output for the current sequence
                    output = model(all_tokens)
                    
                    # Get the last token's logits
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    
                    next_token_logits = logits[-1, :]
                    
                    # Apply temperature sampling
                    if args.temperature > 0:
                        # Apply temperature
                        next_token_logits = next_token_logits / args.temperature
                        # Convert to probabilities
                        probs = mx.softmax(next_token_logits, axis=-1)
                        # Sample from the distribution
                        next_token = mx.random.categorical(probs.reshape(1, -1))
                    else:
                        # Greedy decoding
                        next_token = mx.argmax(next_token_logits)
                    
                    # Convert to scalar for printing
                    next_token_id = int(next_token.item())
                    print(f"Generated token ID {i+1}: {next_token_id}")
                    
                    # Add to our list of generated tokens
                    generated_tokens.append(next_token_id)
                    
                    # Append to the sequence
                    all_tokens = mx.concatenate([all_tokens, next_token.reshape(1)])
                except Exception as e:
                    print(f"Error generating token {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            return all_tokens, mx.array(generated_tokens) if generated_tokens else mx.array([])
        
        # Try direct generation first
        all_tokens, greedy_output = direct_generate(
            trainer.model,
            mx.array(int_tokens),
            max_new_tokens=args.max_tokens
        )
        greedy_score = 0.0
        
        # If direct generation fails or produces no tokens, try generate_lite
        if len(greedy_output) == 0:
            print("Direct generation produced no tokens, trying generate_lite...")
            try:
                greedy_output, greedy_score = generate_lite(
                    trainer.model,
                    mx.array(int_tokens),
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    verbose=True,  # Enable verbose mode to see token-by-token generation
                    stop_tokens=[trainer.tokenizer.EOS_TOKEN],
                    logits_processors=logits_processors
                )
            except Exception as e:
                print(f"generate_lite also failed: {e}")
        
        # Check if we have output and if it's all % characters
        if len(greedy_output) > 0:
            # Convert to list if it's an mx.array, or use as is if already a list
            token_list = greedy_output.tolist() if hasattr(greedy_output, 'tolist') else greedy_output
            output_text = trainer.tokenizer.detokenize(token_list)
            
            if all(c == '%' for c in output_text):
                print("Generation produced only % characters, trying with force_token_id...")
                
                # Try with a range of token IDs to see if any produce meaningful output
                for force_id in range(2, 20):  # Try token IDs 2-19
                    print(f"\nTrying with forced token ID {force_id}...")
                    
                    # Create a simple generation function with forced token ID
                    def forced_generate(model, prompt_tokens, token_id, max_new_tokens=10):
                        if not isinstance(prompt_tokens, mx.array):
                            prompt_tokens = mx.array(prompt_tokens)
                        
                        all_tokens = prompt_tokens
                        generated_tokens = []
                        
                        # Always generate the forced token ID
                        for i in range(max_new_tokens):
                            try:
                                # Just use the forced token ID
                                next_token_id = token_id
                                print(f"Forced token ID: {next_token_id}")
                                
                                # Add to our list of generated tokens
                                generated_tokens.append(next_token_id)
                                
                                # Append to the sequence
                                all_tokens = mx.concatenate([all_tokens, mx.array([next_token_id])])
                            except Exception as e:
                                print(f"Error generating token: {e}")
                                break
                        
                        return all_tokens, mx.array(generated_tokens)
                    
                    # Try with the forced token ID
                    _, forced_tokens = forced_generate(
                        trainer.model,
                        mx.array(int_tokens),
                        token_id=force_id,
                        max_new_tokens=5
                    )
                    
                    # Check if this produces better output
                    if len(forced_tokens) > 0:
                        forced_text = trainer.tokenizer.detokenize(forced_tokens)
                        print(f"Output with token ID {force_id}: '{forced_text}'")
                        
                        # If we found a token that doesn't produce %, use it
                        if not all(c == '%' for c in forced_text):
                            print(f"Found working token ID: {force_id}")
                            greedy_output = forced_tokens
                            break
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a completely different approach - just generate a fixed sequence
        print("\nFalling back to fixed token generation for debugging...")
        
        # Create a list of common tokens to try
        test_tokens = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
        
        # Try each token and see what it produces
        for token_id in test_tokens:
            try:
                print(f"\nTesting token ID {token_id}:")
                test_output = mx.array([token_id] * 5)  # Generate 5 of the same token
                # Pass the array directly, let the detokenize method handle the conversion
                test_text = trainer.tokenizer.detokenize(test_output)
                print(f"Token {token_id} produces: '{test_text}'")
            except Exception as e:
                print(f"Error testing token {token_id}: {e}")
        
        # Just use a fixed set of tokens for output
        print("\nUsing fixed token sequence for output")
        greedy_output = mx.array([2, 3, 4, 5, 6])
        greedy_score = 0.0
    # Make sure we have output to display
    if len(greedy_output) > 0:
        # Pass the output directly to detokenize, which will handle the conversion
        output_text = trainer.tokenizer.detokenize(greedy_output)
        print(f"Generated tokens: {token_list}")
        
        if all(c == '%' for c in output_text):
            print("WARNING: Model is only generating '%' characters, which suggests a mismatch between the model and tokenizer")
            print("Try using a different checkpoint or model configuration")
            
            # Print token ID information for debugging
            print(f"First few tokens in prompt: {tokens[:5]}")
            
            # Try to decode a few different token IDs to see what they produce
            for test_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                try:
                    decoded = trainer.tokenizer.detokenize([test_id])
                    print(f"Token ID {test_id} corresponds to: '{decoded}'")
                except Exception as e:
                    print(f"Error decoding token ID {test_id}: {e}")
            
            # Try to inspect the model's vocabulary
            try:
                # Try to get vocabulary size from embed_tokens if it exists
                if hasattr(trainer.model, 'embed_tokens'):
                    model_vocab_size = trainer.model.embed_tokens.weight.shape[0]
                    print(f"Model vocabulary size (from embedding layer): {model_vocab_size}")
                else:
                    # Try to find the embedding layer by inspecting the model
                    print("Model doesn't have direct embed_tokens attribute, trying to find embedding layer...")
                    for name, param in trainer.model.parameters().items():
                        if 'embed' in name.lower() and len(param.shape) == 2:
                            print(f"Found potential embedding layer: {name} with shape {param.shape}")
                            model_vocab_size = param.shape[0]
                            print(f"Model vocabulary size (from {name}): {model_vocab_size}")
                            break
                    else:
                        print("Could not find embedding layer in model parameters")
                
                # Check if there's a significant mismatch
                if hasattr(trainer.tokenizer, 'tokenizer'):
                    tokenizer_vocab_size = len(trainer.tokenizer.tokenizer.get_vocab())
                    print(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
                    
                    if 'model_vocab_size' in locals() and model_vocab_size != tokenizer_vocab_size:
                        print(f"MISMATCH DETECTED: Model expects {model_vocab_size} tokens but tokenizer has {tokenizer_vocab_size}")
                elif hasattr(trainer.tokenizer, 'external_tokenizer'):
                    # For TokenizerManager which uses external_tokenizer
                    print("Using external tokenizer")
                    if hasattr(trainer.tokenizer.external_tokenizer, 'vocab_size'):
                        ext_vocab_size = trainer.tokenizer.external_tokenizer.vocab_size
                        print(f"External tokenizer vocabulary size: {ext_vocab_size}")
                        
                        if 'model_vocab_size' in locals() and model_vocab_size != ext_vocab_size:
                            print(f"MISMATCH DETECTED: Model expects {model_vocab_size} tokens but tokenizer has {ext_vocab_size}")
            except Exception as e:
                print(f"Error inspecting vocabulary: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Generated Output: {output_text}")
            print(f"Generation Score: {greedy_score:.3f}")
    else:
        print("No tokens were generated. Check if the sampler is working correctly.")
        
    # Print the model and tokenizer info for debugging
    print(f"Model type: {type(trainer.model).__name__}")
    try:
        # Get tokenizer vocabulary size safely
        if hasattr(trainer.tokenizer, 'tokenizer'):
            tokenizer_vocab_size = len(trainer.tokenizer.tokenizer.get_vocab())
            print(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
        elif hasattr(trainer.tokenizer, 'external_tokenizer'):
            print("Using external tokenizer")
            if hasattr(trainer.tokenizer.external_tokenizer, 'vocab_size'):
                ext_vocab_size = trainer.tokenizer.external_tokenizer.vocab_size
                print(f"External tokenizer vocabulary size: {ext_vocab_size}")
        
        # Try to get model vocabulary size from embedding layer
        if hasattr(trainer.model, 'embed_tokens'):
            model_vocab_size = trainer.model.embed_tokens.weight.shape[0]
            print(f"Model vocabulary size (from embedding layer): {model_vocab_size}")
    except Exception as e:
        print(f"Error getting vocabulary info: {e}")
    
    # Print result
    #print(f"Greedy (Score: {score:.3f}): {trainer.tokenizer.detokenize(output)}")

if __name__ == "__main__":
    main()
