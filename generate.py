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
    trainer.model.load_weights(checkpoint_path, strict=args.strict_loading)
    
    # Set model to eval mode
    trainer.model.eval()
    
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
        else:
            print(f"Generated Output: {output_text}")
            print(f"Generation Score: {greedy_score:.3f}")
    else:
        print("No tokens were generated. Check if the sampler is working correctly.")
        
    # Print the model and tokenizer info for debugging
    print(f"Model type: {type(trainer.model).__name__}")
    print(f"Tokenizer vocabulary size: {trainer.tokenizer.vocab_size}")
    print(f"Model vocabulary size: {trainer.model.vocab_size}")
    
    # Print result
    #print(f"Greedy (Score: {score:.3f}): {trainer.tokenizer.detokenize(output)}")

if __name__ == "__main__":
    main()
