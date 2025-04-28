#!/usr/bin/env python3
"""
Script to prepare TinyStories data for MLX-pretrain.
This script processes TinyStories data into the format expected by the MLX-pretrain training script.
"""
import argparse
import json
import os
import random
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

def load_raw_data(data_path, limit=None):
    """Load the TinyStories data from a file or directory."""
    data = []
    
    if os.path.isfile(data_path):
        # Single file
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading data")):
                if limit and i >= limit:
                    break
                try:
                    item = json.loads(line)
                    if "story" in item:
                        data.append(item["story"])
                    elif "text" in item:
                        data.append(item["text"])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {i}")
    else:
        # Directory
        files = list(Path(data_path).glob("*.json"))
        if not files:
            files = list(Path(data_path).glob("*.jsonl"))
        
        for file_path in tqdm(files, desc="Processing files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        for item in file_data[:limit]:
                            if "story" in item:
                                data.append(item["story"])
                            elif "text" in item:
                                data.append(item["text"])
                except json.JSONDecodeError:
                    # Try line-by-line parsing
                    f.seek(0)
                    for i, line in enumerate(f):
                        if limit and i >= limit:
                            break
                        try:
                            item = json.loads(line)
                            if "story" in item:
                                data.append(item["story"])
                            elif "text" in item:
                                data.append(item["text"])
                        except json.JSONDecodeError:
                            continue
    
    return data

def train_tokenizer(texts, vocab_size, output_dir):
    """Train a BPE tokenizer on the data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure the trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2
    )
    
    # Train the tokenizer
    tokenizer.train_from_iterator(texts, trainer)
    
    # Save the tokenizer
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"Tokenizer saved to {output_dir}/tokenizer.json")
    
    return tokenizer

def create_jsonl_data(texts, output_path, max_length=None):
    """Create JSONL data files in the format expected by MLX-pretrain."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in tqdm(texts, desc=f"Writing {output_path}"):
            if max_length:
                # Simple truncation if needed
                text = text[:max_length]
            f.write(json.dumps({"text": text}) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories data for MLX-pretrain.")
    parser.add_argument("--data-path", type=str, required=True, 
                        help="Path to TinyStories data file or directory")
    parser.add_argument("--output-dir", type=str, default="processed_dataset",
                        help="Output directory for processed data")
    parser.add_argument("--vocab-size", type=int, default=8000,
                        help="Size of the tokenizer vocabulary")
    parser.add_argument("--train-split", type=float, default=0.95,
                        help="Proportion of data to use for training")
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Limit number of training samples (default: use all)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Maximum text length (default: no limit)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    print(f"Loading data from {args.data_path}")
    data = load_raw_data(args.data_path, limit=args.train_samples)
    print(f"Loaded {len(data)} stories")
    
    # Shuffle data
    random.shuffle(data)
    
    # Split data into train and validation sets
    split_idx = int(len(data) * args.train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Training set: {len(train_data)} stories")
    print(f"Validation set: {len(val_data)} stories")
    
    # Train tokenizer
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    print(f"Training tokenizer with vocab size {args.vocab_size}")
    train_tokenizer(train_data, args.vocab_size, tokenizer_dir)
    
    # Create JSONL data files
    train_jsonl = os.path.join(args.output_dir, "train.jsonl")
    val_jsonl = os.path.join(args.output_dir, "val.jsonl")
    
    print("Creating training data file")
    create_jsonl_data(train_data, train_jsonl, args.max_length)
    
    print("Creating validation data file")
    create_jsonl_data(val_data, val_jsonl, args.max_length)
    
    print(f"Data processing complete. Files saved to {args.output_dir}")
    print(f"To use this data, set the following in your config file:")
    print(f"  data:")
    print(f"    input_file: \"{train_jsonl}\"")
    print(f"    validation_file: \"{val_jsonl}\"")
    print(f"    tokenizer_path: \"{tokenizer_dir}\"")

if __name__ == "__main__":
    main()