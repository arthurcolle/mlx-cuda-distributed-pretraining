#!/usr/bin/env python
# Script to prepare data for A100 GPU training
# Optimizes data format and checks for potential issues

import os
import argparse
import json
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def validate_jsonl(file_path, max_samples=10):
    """Validate JSONL format and show first few samples"""
    print(f"Validating data file: {file_path}")
    
    try:
        valid_count = 0
        invalid_count = 0
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Validating lines")):
                try:
                    data = json.loads(line)
                    if "text" not in data:
                        print(f"  Warning: Line {i+1} is missing 'text' field")
                        invalid_count += 1
                    else:
                        valid_count += 1
                        if len(samples) < max_samples:
                            # Truncate sample for display
                            text = data["text"]
                            if len(text) > 100:
                                text = text[:97] + "..."
                            samples.append(text)
                except json.JSONDecodeError:
                    print(f"  Error: Line {i+1} is not valid JSON")
                    invalid_count += 1
        
        # Print statistics
        print(f"\nValidation results for {file_path}:")
        print(f"  Valid records: {valid_count}")
        print(f"  Invalid records: {invalid_count}")
        
        # Show samples
        if samples:
            print("\nSample records:")
            for i, sample in enumerate(samples):
                print(f"  {i+1}. {sample}")
        
        return valid_count > 0
    
    except Exception as e:
        print(f"Error validating file: {e}")
        return False

def validate_tokenizer(tokenizer_dir):
    """Validate tokenizer files"""
    print(f"Validating tokenizer in: {tokenizer_dir}")
    
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"  Error: tokenizer.json not found in {tokenizer_dir}")
        return False
    
    try:
        # Try to load the tokenizer to validate it
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = len(tokenizer.get_vocab())
        print(f"  Tokenizer loaded successfully with {vocab_size} tokens")
        return True
    except Exception as e:
        print(f"  Error loading tokenizer: {e}")
        return False

def prepare_data_directory(output_dir, train_file, val_file=None, tokenizer_dir=None):
    """Prepare data directory for training"""
    print(f"Preparing data directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy training file
    train_output = os.path.join(output_dir, "train.jsonl")
    shutil.copy2(train_file, train_output)
    print(f"  Copied training data to: {train_output}")
    
    # Copy validation file if provided
    if val_file:
        val_output = os.path.join(output_dir, "val.jsonl")
        shutil.copy2(val_file, val_output)
        print(f"  Copied validation data to: {val_output}")
    
    # Copy tokenizer if provided
    if tokenizer_dir:
        tokenizer_output = os.path.join(output_dir, "tokenizer")
        os.makedirs(tokenizer_output, exist_ok=True)
        
        # Copy tokenizer.json
        tokenizer_src = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer_dst = os.path.join(tokenizer_output, "tokenizer.json")
        shutil.copy2(tokenizer_src, tokenizer_dst)
        print(f"  Copied tokenizer to: {tokenizer_dst}")

def create_validation_split(train_file, output_dir, val_ratio=0.05, seed=42):
    """Create a validation split from training data"""
    print(f"Creating validation split with ratio: {val_ratio}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all lines
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle lines
    random.shuffle(lines)
    
    # Calculate split point
    split_idx = int(len(lines) * (1 - val_ratio))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # Write train split
    train_output = os.path.join(output_dir, "train.jsonl")
    with open(train_output, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Write validation split
    val_output = os.path.join(output_dir, "val.jsonl")
    with open(val_output, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"  Created training split with {len(train_lines)} examples: {train_output}")
    print(f"  Created validation split with {len(val_lines)} examples: {val_output}")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for A100 GPU training")
    
    # Data source options
    parser.add_argument("--train", type=str, required=True, 
                        help="Path to training data file (JSONL format)")
    parser.add_argument("--val", type=str, default=None,
                        help="Path to optional validation data file (JSONL format)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer directory containing tokenizer.json")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Directory to store prepared data")
    
    # Processing options
    parser.add_argument("--create-val-split", action="store_true",
                        help="Create validation split from training data")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Ratio of data to use for validation if creating split")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate data without copying")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.train):
        print(f"Error: Training file not found: {args.train}")
        return
    
    if args.val and not os.path.exists(args.val):
        print(f"Error: Validation file not found: {args.val}")
        return
    
    if args.tokenizer and not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer directory not found: {args.tokenizer}")
        return
    
    # Validate input files
    train_valid = validate_jsonl(args.train)
    if not train_valid:
        print("Error: Training data validation failed")
        return
    
    if args.val:
        val_valid = validate_jsonl(args.val)
        if not val_valid:
            print("Error: Validation data validation failed")
            return
    
    if args.tokenizer:
        tokenizer_valid = validate_tokenizer(args.tokenizer)
        if not tokenizer_valid:
            print("Error: Tokenizer validation failed")
            return
    
    # Stop here if only validating
    if args.validate_only:
        print("Validation complete. All files are valid.")
        return
    
    # Create validation split if requested
    if args.create_val_split and not args.val:
        create_validation_split(
            args.train, 
            args.output_dir,
            val_ratio=args.val_ratio
        )
    else:
        # Copy files to output directory
        prepare_data_directory(
            args.output_dir,
            args.train,
            args.val,
            args.tokenizer
        )
    
    print(f"\nData preparation complete. Data ready in: {args.output_dir}")
    print(f"You can now run training with: python train_a100.py --data-dir {args.output_dir}")

if __name__ == "__main__":
    main()