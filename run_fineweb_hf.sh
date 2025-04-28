#!/bin/bash

# Run FineWeb streaming training with 80M parameter model using HuggingFace datasets

# Default parameters
CONFIG="configs/model-config-80m-fineweb.yaml"
DATASET="HuggingFaceFW/fineweb"
CONFIG_NAME="CC-MAIN-2013-20"
SPLIT="train"
WORKERS=2
PREFETCH=1

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --config-name)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --prefetch)
      PREFETCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--config CONFIG] [--dataset DATASET] [--config-name CONFIG_NAME] [--split SPLIT] [--workers NUM] [--prefetch NUM]"
      exit 1
      ;;
  esac
done

echo "Starting FineWeb HuggingFace streaming training with the following parameters:"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "Config name: $CONFIG_NAME"
echo "Split: $SPLIT"
echo "Worker processes: $WORKERS"
echo "Prefetch factor: $PREFETCH" 
echo

# Set MLX memory limit (30GB in bytes)
export MLX_MEM_LIMIT_BYTES=32212254720

# Make sure required packages are installed
pip list | grep -q datasets || pip install datasets
pip list | grep -q huggingface_hub || pip install huggingface_hub

# Run the streaming training using HuggingFace datasets
python fineweb_stream_hf.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --config-name "$CONFIG_NAME" \
  --split "$SPLIT" \
  --workers "$WORKERS" \
  --prefetch "$PREFETCH"

# Exit with the exit code of the python script
exit $?