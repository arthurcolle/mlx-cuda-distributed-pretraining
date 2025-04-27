#!/usr/bin/env python3
"""
Find and list potential training data files in the current directory.
This script searches for common data file formats that might be suitable
for language model training.
"""

import os
import argparse
from pathlib import Path
import json

def is_text_file(file_path, sample_lines=5):
    """Check if a file is a text file by trying to read a few lines."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(sample_lines):
                f.readline()
        return True
    except UnicodeDecodeError:
        return False
    except Exception:
        return False

def is_jsonl_file(file_path, sample_lines=5):
    """Check if a file is a JSONL file by trying to parse a few lines as JSON."""
    if not is_text_file(file_path):
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(sample_lines):
                line = f.readline().strip()
                if line:  # Skip empty lines
                    json.loads(line)
        return True
    except json.JSONDecodeError:
        return False
    except Exception:
        return False

def get_file_info(file_path):
    """Get information about a file."""
    path = Path(file_path)
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    # Count lines for text files
    line_count = None
    if is_text_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
        except Exception:
            pass
    
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "line_count": line_count,
        "is_jsonl": is_jsonl_file(file_path)
    }

def find_data_files(directory='.', recursive=True, extensions=None, min_size_kb=10):
    """Find potential data files in the directory."""
    if extensions is None:
        extensions = ['.txt', '.json', '.jsonl', '.csv', '.tsv', '.md']
    
    data_files = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common directories to ignore
        dirs[:] = [d for d in dirs if not d.startswith('.') and 
                  d not in ['node_modules', 'venv', 'env', '__pycache__', '.git']]
        
        for file in files:
            # Check if the file has a relevant extension
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                
                # Check file size
                size_kb = os.path.getsize(file_path) / 1024
                if size_kb >= min_size_kb:
                    # Get detailed file info
                    file_info = get_file_info(file_path)
                    data_files.append(file_info)
        
        # If not recursive, break after the first iteration
        if not recursive:
            break
    
    return data_files

def main():
    parser = argparse.ArgumentParser(description='Find potential training data files')
    parser.add_argument('--dir', type=str, default='.', help='Directory to search in')
    parser.add_argument('--recursive', action='store_true', help='Search recursively')
    parser.add_argument('--min-size', type=int, default=10, help='Minimum file size in KB')
    parser.add_argument('--extensions', type=str, default='.txt,.json,.jsonl,.csv,.tsv,.md',
                        help='Comma-separated list of file extensions to look for')
    args = parser.parse_args()
    
    extensions = args.extensions.split(',')
    data_files = find_data_files(args.dir, args.recursive, extensions, args.min_size)
    
    # Sort by size (largest first)
    data_files.sort(key=lambda x: x['size_bytes'], reverse=True)
    
    print(f"Found {len(data_files)} potential data files:")
    print("-" * 80)
    
    for i, file_info in enumerate(data_files, 1):
        print(f"{i}. {file_info['path']}")
        print(f"   Size: {file_info['size_mb']} MB")
        if file_info['line_count'] is not None:
            print(f"   Lines: {file_info['line_count']}")
        print(f"   JSONL format: {'Yes' if file_info['is_jsonl'] else 'No'}")
        print()
    
    print("-" * 80)
    print("To train a model with one of these files, use:")
    print("python train_simple.py --data-path <file_path> --vocab-size 8000 --hidden-size 256 --num-layers 4 --epochs 3")
    print("\nFor a quick test with a small model:")
    print("python train_simple.py --data-path <file_path> --vocab-size 4000 --hidden-size 128 --num-layers 2 --epochs 1 --max-samples 10000")

if __name__ == "__main__":
    main()
