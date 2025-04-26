#!/bin/bash

# Run a 1M parameter model with the enhanced AdamW optimizer
# Includes features like proper decoupled weight decay, gradient clipping, 
# AMSGrad, and EMA weight averaging

python train.py --config model-config-1m-adamw-enhanced.yaml