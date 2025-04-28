#!/usr/bin/env python3
"""
Script to download and tokenize large text datasets from Hugging Face.
Examples:
  python download_and_process_llm_data.py \
    togethercomputer/RedPajama-Data-1T openwebtext the_pile:fineweb \
    --total-tokens 20000000000 --output-dir data_llm --final-output combined.bin
Requires: datasets, tiktoken, numpy, tqdm
"""
import argparse
import os
import shutil
from datasets import load_dataset
import tiktoken
import numpy as np
import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download and tokenize datasets to a fixed token count.")
    parser.add_argument('datasets', nargs='+',
                        help='Dataset identifiers (name or name:config) to download from Hugging Face.')
    parser.add_argument('--split', default='train',
                        help='Dataset split to use (default: train).')
    parser.add_argument('--total-tokens', type=int, default=20_000_000_000,
                        help='Total number of tokens to collect across all datasets.')
    parser.add_argument('--output-dir', default='processed_llm_data',
                        help='Directory to write per-dataset token files.')
    parser.add_argument('--final-output', default=None,
                        help='Optional path to combine all token files into one binary.')
    parser.add_argument('--encoding', default='gpt2',
                        help='tiktoken encoding name (default: gpt2).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sources = []
    for ds in args.datasets:
        if ':' in ds:
            name, config = ds.split(':', 1)
        else:
            name, config = ds, None
        sources.append((name, config))

    total = args.total_tokens
    n = len(sources)
    base = total // n
    rem = total % n

    encoder = tiktoken.get_encoding(args.encoding)
    per_dataset = []
    for idx, (name, config) in enumerate(sources):
        target = base + (1 if idx < rem else 0)
        per_dataset.append((name, config, target))

    for name, config, target in per_dataset:
        print(f"Processing {name} with target {target} tokens...")
        ds = load_dataset(name, config, split=args.split, streaming=True)
        out_file = os.path.join(args.output_dir, name.replace('/', '_') + '.bin')
        tokens_acc = 0
        with open(out_file, 'wb') as f:
            for ex in tqdm.tqdm(ds, desc=name, unit='samples'):
                text = ex.get('text') or ex.get('content') or ''
                ids = encoder.encode(text)
                if not ids:
                    continue
                remaining = target - tokens_acc
                if len(ids) > remaining:
                    ids = ids[:remaining]
                arr = np.array(ids, dtype=np.uint16)
                f.write(arr.tobytes())
                tokens_acc += len(ids)
                if tokens_acc >= target:
                    break
        print(f"Wrote {tokens_acc} tokens to {out_file}")

    if args.final_output:
        combined = args.final_output
        print(f"Combining files into {combined}...")
        with open(combined, 'wb') as fout:
            for name, _, _ in per_dataset:
                path = os.path.join(args.output_dir, name.replace('/', '_') + '.bin')
                with open(path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)
        print(f"Combined file written to {combined}")

if __name__ == '__main__':
    main()