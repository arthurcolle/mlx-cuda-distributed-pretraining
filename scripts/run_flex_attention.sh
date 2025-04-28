#!/bin/bash
# Run training with flex attention configuration

# Set MLX precision to float16 for faster training
export MLX_USE_F16=1

# Run training with the Flex Attention config
python train.py --config model-config-1m-muon-flex.yaml