import argparse
import json
import yaml
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_jsonl_texts(file_path):
    """Load and extract text from a JSONL file."""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if 'text' in item:
                    texts.append(item['text'])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line}")
    return texts


def batch_iterator(texts, batch_size=1000):
    """Creates batches of texts for tokenizer training."""
    for i in range(0, len(texts), batch_size):
        yield texts[i:i+batch_size]


def train_tokenizer(config):
    """Train a byte-level BPE tokenizer based on the provided configuration."""
    
    # Initialize the tokenizer with a BPE model
    tokenizer = Tokenizer(BPE())
    
    # Configure pre-tokenizer for superword BPE (no word boundaries)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    
    # Set up the normalizer
    tokenizer.normalizer = NFKC()
    
    # Set up the decoder
    tokenizer.decoder = ByteLevelDecoder()
    
    # Get special tokens from config
    special_tokens = []
    if 'data' in config and 'tokenizer' in config['data'] and 'special_tokens' in config['data']['tokenizer']:
        special_tokens = list(config['data']['tokenizer']['special_tokens'].values())
    
    # Get vocab size from config
    vocab_size = 32000  # Default
    if 'tokenizer' in config and 'vocab_size' in config['tokenizer']:
        vocab_size = config['tokenizer']['vocab_size']
    
    # Set up the trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Load training data
    input_file = config['data']['input_file'] if 'data' in config and 'input_file' in config['data'] else 'train.jsonl'
    texts = load_jsonl_texts(input_file)

    if 'data' in config and 'max_texts_to_train_on' in config['data']:
        texts = texts[:config['data']['max_texts_to_train_on']]
    print(f"Training tokenizer on {len(texts)} texts with vocab size {vocab_size}")
    
    # Train the tokenizer
    tokenizer.train_from_iterator(batch_iterator(texts), trainer=trainer)
    
    # Create output directory if it doesn't exist
    output_dir = config['tokenizer']['output_dir'] if 'tokenizer' in config and 'output_dir' in config['tokenizer'] else 'tokenizer'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the tokenizer
    output_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    
    # Test the tokenizer
    if texts:
        test_text = texts[0][:100]  # Take first 100 chars of first text for testing
        encoded = tokenizer.encode(test_text)
        print("\nTest encoding:")
        print(f"Text: {test_text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"IDs: {encoded.ids}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer using a YAML configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_tokenizer(config)


if __name__ == "__main__":
    main()