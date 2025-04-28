#!/bin/bash

# Run FineWeb streaming training with 80M parameter model and limited disk space (40GB)

# Default parameters
CONFIG="configs/model-config-80m-fineweb.yaml"
WORKERS=4
PREFETCH=2
MAX_DISK=35
SHARDS="s3://fineweb-corpus/shard-{00000..01000}.tar.bz2"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
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
    --max-disk)
      MAX_DISK="$2"
      shift 2
      ;;
    --shards)
      SHARDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--config CONFIG] [--workers NUM] [--prefetch NUM] [--max-disk GB] [--shards PATTERN]"
      exit 1
      ;;
  esac
done

echo "Starting FineWeb streaming training with limited disk space:"
echo "Config: $CONFIG"
echo "Worker processes: $WORKERS"
echo "Prefetch factor: $PREFETCH"
echo "Max disk usage: $MAX_DISK GB" 
echo "Shard pattern: $SHARDS"
echo

# Check available disk space
AVAILABLE_DISK=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
echo "Available disk space: ~${AVAILABLE_DISK}GB"

if (( $(echo "$AVAILABLE_DISK < 5" | bc -l) )); then
  echo "WARNING: Very low disk space available! Training might fail."
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Install required packages if not already installed
pip list | grep -q webdataset || pip install webdataset

# Run the streaming training with disk space management
python fineweb_stream_limited.py \
  --config "$CONFIG" \
  --shard-pattern "$SHARDS" \
  --workers "$WORKERS" \
  --prefetch "$PREFETCH" \
  --max-disk "$MAX_DISK"

# Exit with the exit code of the python script
exit $?