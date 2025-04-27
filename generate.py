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
    
    trainer.model.load_weights(checkpoint_path)
    
    # Prepare input
    tokens = [trainer.tokenizer.BOS_TOKEN] + trainer.tokenizer.tokenize(args.prompt)
    
    # Setup generation parameters
    sampler = make_sampler(temp=args.temperature, min_p=args.min_p)
    logits_processors = make_logits_processors(
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size
    )
    
    # Generate
    """output = beam_search(
        trainer.model,
        mx.array(tokens),
        max_tokens=args.max_tokens,  # Limit the max tokens to generate
        verbose=True,
        n_beams=4,  # Use beam search for generation
        stop_tokens=[trainer.tokenizer.EOS_TOKEN]
    )"""
    mx.random.seed(int(time.time() * 1000))
    greedy_output, greedy_score = generate_lite(
            trainer.model,
            mx.array(tokens),
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
            stop_tokens=[trainer.tokenizer.EOS_TOKEN],
            logits_processors=logits_processors
    )
    print(f"Greedy Output: {trainer.tokenizer.detokenize(greedy_output)}")
    #print(f"Greedy Output (Score: {greedy_score:.3f}):     {trainer.tokenizer.detokenize(greedy_output)}")
    
    # Print result
    #print(f"Greedy (Score: {score:.3f}): {trainer.tokenizer.detokenize(output)}")

if __name__ == "__main__":
    main()
