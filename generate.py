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
    print(f"Model has {sum(p.size for _, p in trainer.model.parameters().items()):,} parameters")
    
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
        
        # Try with temperature sampling first
        print("Generating with temperature sampling...")
        greedy_output, greedy_score = generate_lite(
                trainer.model,
                mx.array(tokens),
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=True,  # Enable verbose mode to see token-by-token generation
                stop_tokens=[trainer.tokenizer.EOS_TOKEN],
                logits_processors=logits_processors
        )
        
        # If temperature sampling fails or produces % characters, try greedy decoding
        output_text = trainer.tokenizer.detokenize(greedy_output)
        if all(c == '%' for c in output_text):
            print("Temperature sampling produced only % characters, trying greedy decoding...")
            # Try with a simple greedy sampler
            def greedy_sampler(logprobs):
                if args.force_token_id is not None:
                    # Force a specific token ID for debugging
                    token = mx.array(args.force_token_id)
                    print(f"Forcing token ID: {args.force_token_id}")
                else:
                    # Get top 5 tokens for debugging
                    if args.debug:
                        top_tokens = mx.topk(logprobs, k=5)
                        top_probs = mx.take(logprobs, top_tokens)
                        print(f"Top tokens: {top_tokens.tolist()}, probs: {mx.exp(top_probs).tolist()}")
                    
                    token = mx.argmax(logprobs, axis=-1)
                
                print(f"Sampler selected token: {token.item()}")
                return token
                
            greedy_output, greedy_score = generate_lite(
                    trainer.model,
                    mx.array(tokens),
                    max_tokens=args.max_tokens,
                    sampler=greedy_sampler,
                    verbose=True,
                    stop_tokens=[trainer.tokenizer.EOS_TOKEN],
                    logits_processors=logits_processors
            )
    except Exception as e:
        print(f"Error during generation: {e}")
        # Try fallback to beam search
        print("Falling back to beam search...")
        try:
            greedy_output = beam_search(
                trainer.model,
                mx.array(tokens),
                max_tokens=args.max_tokens,
                verbose=True,
                n_beams=4,
                stop_tokens=[trainer.tokenizer.EOS_TOKEN]
            )
            greedy_score = 0.0  # No score for beam search
        except Exception as e2:
            print(f"Beam search also failed: {e2}")
            greedy_output = []
            greedy_score = 0.0
    # Make sure we have output to display
    if len(greedy_output) > 0:
        output_text = trainer.tokenizer.detokenize(greedy_output)
        print(f"Generated tokens: {greedy_output.tolist()}")
        
        if all(c == '%' for c in output_text):
            print("WARNING: Model is only generating '%' characters, which suggests a mismatch between the model and tokenizer")
            print("Try using a different checkpoint or model configuration")
            
            # Print token ID information for debugging
            print(f"First few tokens in prompt: {tokens[:5]}")
            print(f"Token ID 7 corresponds to: '{trainer.tokenizer.detokenize([7])}'")
            
            # Try to inspect the model's vocabulary
            try:
                # Get the model's vocabulary size from the embedding layer
                model_vocab_size = trainer.model.vocab_size
                print(f"Model vocabulary size: {model_vocab_size}")
                
                # Check if there's a significant mismatch
                tokenizer_vocab_size = len(trainer.tokenizer.tokenizer.get_vocab())
                print(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
                
                if model_vocab_size != tokenizer_vocab_size:
                    print(f"MISMATCH DETECTED: Model expects {model_vocab_size} tokens but tokenizer has {tokenizer_vocab_size}")
            except Exception as e:
                print(f"Error inspecting vocabulary: {e}")
        else:
            print(f"Generated Output: {output_text}")
            print(f"Generation Score: {greedy_score:.3f}")
    else:
        print("No tokens were generated. Check if the sampler is working correctly.")
        
    # Print the model and tokenizer info for debugging
    print(f"Model type: {type(trainer.model).__name__}")
    try:
        # Get tokenizer vocabulary size safely
        tokenizer_vocab_size = len(trainer.tokenizer.tokenizer.get_vocab())
        print(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
        print(f"Model vocabulary size: {trainer.model.vocab_size}")
    except Exception as e:
        print(f"Error getting vocabulary info: {e}")
    
    # Print result
    #print(f"Greedy (Score: {score:.3f}): {trainer.tokenizer.detokenize(output)}")

if __name__ == "__main__":
    main()
