#!/usr/bin/env python
import os
import argparse
from pathlib import Path
import mlx.core as mx
import re
from typing import List, Dict, Optional
import inquirer
from inquirer import themes
from colorama import Fore, Style
import yaml
from train import Trainer
from generate_lite import generate_lite
from mlx_lm_utils import make_sampler, make_logits_processors

# Set device to GPU
mx.set_default_device(mx.gpu)

def list_runs() -> List[str]:
    """List all available runs in the runs directory."""
    runs_dir = Path('runs')
    if not runs_dir.exists():
        return []
    
    runs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith('.') and not run_dir.name.endswith('.log'):
            # Check if it has checkpoints and final model
            checkpoint_dir = run_dir / 'checkpoints'
            has_final_model = False
            if checkpoint_dir.exists():
                final_model = checkpoint_dir / 'step_final_model.safetensors'
                alt_final_model = checkpoint_dir / 'step_final.safetensors'
                if final_model.exists() or alt_final_model.exists():
                    has_final_model = True
            
            # Only include if it has final model
            if has_final_model:
                runs.append(run_dir.name)
    
    # Sort by name
    return sorted(runs)

def get_metadata(run_name: str) -> Dict:
    """Get metadata for a run."""
    metadata_path = Path('runs') / run_name / 'metadata.json'
    config_path = Path('runs') / run_name / 'config.yaml'
    
    metadata = {}
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            metadata['model_config'] = config.get('model', {})
            metadata['training_config'] = config.get('training', {})
    
    return metadata

def print_run_details(run_name: str):
    """Print details about a run."""
    metadata = get_metadata(run_name)
    
    print(f"{Fore.CYAN}=== Run: {run_name} ==={Style.RESET_ALL}")
    
    # Print model details if available
    if 'model_config' in metadata:
        model_config = metadata['model_config']
        print(f"{Fore.GREEN}Model:{Style.RESET_ALL}")
        print(f"  Type: {model_config.get('type', 'Unknown')}")
        print(f"  Dim: {model_config.get('dim', 'Unknown')}")
        print(f"  n_layers: {model_config.get('n_layers', 'Unknown')}")
        print(f"  n_heads: {model_config.get('n_heads', 'Unknown')}")
        print(f"  vocab_size: {model_config.get('vocab_size', 'Unknown')}")
    
    # Print training details if available
    if 'training_config' in metadata:
        train_config = metadata['training_config']
        print(f"{Fore.GREEN}Training:{Style.RESET_ALL}")
        print(f"  Batch size: {train_config.get('batch_size', 'Unknown')}")
        print(f"  Max seq len: {train_config.get('max_seq_len', 'Unknown')}")
        print(f"  Max steps: {train_config.get('max_steps', 'Unknown')}")
        print(f"  Optimizer: {train_config.get('optimizer', {}).get('name', 'Unknown')}")
    
    # Print creation time if available
    if 'created_at' in metadata:
        print(f"{Fore.GREEN}Created:{Style.RESET_ALL} {metadata.get('created_at', 'Unknown')}")
    
    print()

def load_model_for_run(run_name: str):
    """Load a model for a run."""
    config_path = Path('runs') / run_name / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f"Config not found for run: {run_name}")
    
    trainer = Trainer(str(config_path), for_training=False)
    
    # Load the final checkpoint
    checkpoint_path = Path('runs') / run_name / 'checkpoints' / 'step_final_model.safetensors'
    if not checkpoint_path.exists():
        checkpoint_path = Path('runs') / run_name / 'checkpoints' / 'step_final.safetensors'
        if not checkpoint_path.exists():
            raise ValueError(f"Final checkpoint not found for run: {run_name}")
    checkpoint_path = str(checkpoint_path)
    
    trainer.model.load_weights(checkpoint_path)
    return trainer

def generate_text(trainer, prompt: str, max_tokens: int = 256, temperature: float = 0.8, 
                  min_p: float = 0.05, repetition_penalty: float = 1.1,
                  repetition_context_size: int = 20):
    """Generate text from a model."""
    # Prepare input
    tokens = [trainer.tokenizer.BOS_TOKEN] + trainer.tokenizer.tokenize(prompt)
    
    # Setup generation parameters
    sampler = make_sampler(temp=temperature, min_p=min_p)
    logits_processors = make_logits_processors(
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size
    )
    
    # Generate
    mx.random.seed(int(os.urandom(4).hex(), 16))  # Random seed based on system entropy
    output, _ = generate_lite(
        trainer.model,
        mx.array(tokens),
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
        stop_tokens=[trainer.tokenizer.EOS_TOKEN],
        logits_processors=logits_processors
    )
    
    return trainer.tokenizer.detokenize(output)

def interactive_mode():
    """Run in interactive mode."""
    # List available runs
    runs = list_runs()
    if not runs:
        print(f"{Fore.RED}No trained models found in the 'runs' directory.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.CYAN}Found {len(runs)} trained models.{Style.RESET_ALL}")
    
    while True:
        # Create main menu
        questions = [
            inquirer.List('action',
                          message="Select an action",
                          choices=[
                              ('List available models', 'list'),
                              ('View model details', 'details'),
                              ('Generate text with model', 'generate'),
                              ('Exit', 'exit')
                          ],
                         )
        ]
        
        answers = inquirer.prompt(questions, theme=themes.GreenPassion())
        if not answers:  # Handle Ctrl-C
            break
            
        action = answers['action']
        
        if action == 'list':
            print(f"{Fore.CYAN}Available models:{Style.RESET_ALL}")
            for i, run in enumerate(runs, 1):
                print(f"{i}. {run}")
            print()
            
        elif action == 'details':
            # Prompt for model selection
            questions = [
                inquirer.List('run',
                              message="Select a model to view details",
                              choices=[(run, run) for run in runs],
                             )
            ]
            answers = inquirer.prompt(questions, theme=themes.GreenPassion())
            if answers:
                print_run_details(answers['run'])
                
        elif action == 'generate':
            # Prompt for model selection
            questions = [
                inquirer.List('run',
                              message="Select a model for text generation",
                              choices=[(run, run) for run in runs],
                             ),
                inquirer.Text('prompt',
                             message="Enter a prompt",
                             default="Once upon a time"),
                inquirer.Text('max_tokens',
                             message="Maximum tokens to generate",
                             default="256"),
                inquirer.Text('temperature',
                             message="Temperature (0.1-2.0)",
                             default="0.8"),
            ]
            answers = inquirer.prompt(questions, theme=themes.GreenPassion())
            if not answers:
                continue
                
            run_name = answers['run']
            prompt = answers['prompt']
            max_tokens = int(answers['max_tokens'])
            temperature = float(answers['temperature'])
            
            try:
                print(f"{Fore.CYAN}Loading model {run_name}...{Style.RESET_ALL}")
                trainer = load_model_for_run(run_name)
                
                print(f"{Fore.CYAN}Generating text...{Style.RESET_ALL}")
                generated_text = generate_text(
                    trainer, 
                    prompt, 
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
                
                print(f"{Fore.GREEN}Prompt:{Style.RESET_ALL} {prompt}")
                print(f"{Fore.GREEN}Generated text:{Style.RESET_ALL}")
                print(generated_text)
                print()
                
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                
        elif action == 'exit':
            break

def main():
    parser = argparse.ArgumentParser(description='MLX Model CLI')
    parser.add_argument('--list-runs', action='store_true',
                       help='List available trained models')
    parser.add_argument('--run', type=str,
                       help='Name of the training run to use')
    parser.add_argument('--prompt', type=str,
                       help='Text prompt to start generation')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--min-p', type=float, default=0.05,
                       help='Minimum probability for nucleus sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                       help='Repetition penalty factor')
    
    args = parser.parse_args()
    
    # If no arguments, run in interactive mode
    if len(os.sys.argv) == 1:
        interactive_mode()
        return
    
    # List runs
    if args.list_runs:
        runs = list_runs()
        print(f"Found {len(runs)} trained models:")
        for i, run in enumerate(runs, 1):
            print(f"{i}. {run}")
        return
    
    # Generate text with a model
    if args.run and args.prompt:
        try:
            trainer = load_model_for_run(args.run)
            generated_text = generate_text(
                trainer,
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty
            )
            
            print(f"Prompt: {args.prompt}")
            print(f"Generated text:")
            print(generated_text)
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Missing required arguments
    elif args.run or args.prompt:
        print("Error: Both --run and --prompt are required for text generation.")
        parser.print_help()

if __name__ == "__main__":
    main()