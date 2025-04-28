#!/bin/bash

# Run FineWeb streaming training with 80M parameter model

# Default parameters
CONFIG="configs/model-config-80m-fineweb.yaml"
WORKERS=4
PREFETCH=2
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
    --shards)
      SHARDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--config CONFIG] [--workers NUM] [--prefetch NUM] [--shards PATTERN]"
      exit 1
      ;;
  esac
done

echo "Starting FineWeb streaming training with the following parameters:"
echo "Config: $CONFIG"
echo "Worker processes: $WORKERS"
echo "Prefetch factor: $PREFETCH" 
echo "Shard pattern: $SHARDS"
echo

# Install webdataset if not already installed
pip list | grep -q webdataset || pip install webdataset

# Run the streaming training
python fineweb_stream.py \
  --config "$CONFIG" \
  --shard-pattern "$SHARDS" \
  --workers "$WORKERS" \
  --prefetch "$PREFETCH"

# Exit with the exit code of the python script
exit $?