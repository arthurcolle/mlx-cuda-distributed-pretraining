# Training on FineWeb with MLX

This guide explains how to train an 80M parameter model on the FineWeb dataset using MLX, with a focus on efficient streaming processing to handle large datasets even with limited disk space.

## Model Configuration

The configuration in `configs/model-config-80m-fineweb.yaml` defines an 80M parameter model:

- **Architecture**: 12-layer Llama model
- **Hidden Size**: 1024
- **Intermediate Size**: 2816
- **Attention**: 16 heads (8 KV heads) with flash attention
- **Context Length**: 2048 tokens
- **Optimizer**: Muon (Adam variant with improved convergence)
- **Training**: 10,000 iterations with batch size 8 (effective batch 64 with gradient accumulation)

## Streaming Processing for Limited Disk Space

For machines with limited disk space (around 40GB), we provide a specialized streaming solution:

### Key Features

1. **On-the-fly Processing**: Data is streamed directly from remote storage and processed without storing the entire dataset locally
2. **Disk Usage Management**: Automatic monitoring and cleanup of cached data to stay within disk limits
3. **Gradient Accumulation**: Small per-device batch size (8) with 8x gradient accumulation for effective batch size of 64
4. **Checkpoint Management**: Efficient checkpoint saving to prevent disk overflow

## Quick Start

To begin training with 35GB disk usage limit:

```bash
./run_fineweb_limited.sh --shards "s3://your-fineweb-location/shard-{00000..01000}.tar.bz2"
```

## Command Options

```
Usage: ./run_fineweb_limited.sh [--config CONFIG] [--workers NUM] [--prefetch NUM] [--max-disk GB] [--shards PATTERN]

Options:
  --config CONFIG    Path to model config (default: configs/model-config-80m-fineweb.yaml)
  --workers NUM      Number of dataloader workers (default: 4)
  --prefetch NUM     Prefetch factor for dataloader (default: 2)
  --max-disk GB      Maximum disk usage in GB (default: 35)
  --shards PATTERN   URL pattern for FineWeb shards (e.g., s3://fineweb/shard-{00000..01000}.tar.bz2)
```

## Preparing FineWeb Data

To use this system, your FineWeb data needs to be:

1. Sharded into multiple files (e.g., multiple TAR or JSONL files)
2. Stored in a remote location accessible via an S3/HTTP/HTTPS URL
3. Formatted with text content in a 'text' field within JSON objects

Example shard format inside TAR files:
```json
{"text": "This is a document from the FineWeb dataset..."}
```

## Monitoring Training

The training process logs metrics to both the console and a log file in the `runs/MLX-80M-FineWeb-Stream/log.txt` path. Key metrics:

- Loss and perplexity
- Tokens/second throughput 
- Total tokens processed
- Current learning rate
- Disk cache usage

## Advanced Usage

### Custom Dataset Locations

To use a different dataset location:

```bash
./run_fineweb_limited.sh --shards "https://your-host.com/data/fineweb_{000..999}.tar"
```

### Adjusting Disk Usage

For machines with more or less available space:

```bash
# Limit to 20GB disk usage
./run_fineweb_limited.sh --max-disk 20 --shards "s3://fineweb/shard-{00000..01000}.tar.bz2"
```

### Tuning Performance

For faster processing with more CPU cores:

```bash
./run_fineweb_limited.sh --workers 8 --prefetch 4 --shards "s3://fineweb/shard-{00000..01000}.tar.bz2"
```

## Implementation Details

The streaming system uses:

1. **WebDataset**: For efficient streaming from remote sources
2. **PyTorch DataLoader**: For parallel fetching and processing
3. **MLX**: For training on Apple Silicon GPUs
4. **DiskSpaceManager**: Custom component that manages local cache cleanup

This approach allows you to train on the full 15TB FineWeb corpus (or any subset) while only using a small, configurable amount of local disk space.