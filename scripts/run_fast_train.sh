#!/bin/bash
# Run fast training of 124M model

echo "Starting fast 124M parameter model training (< 8 min)"
python train.py --config configs/model-config-124m-fast.yaml