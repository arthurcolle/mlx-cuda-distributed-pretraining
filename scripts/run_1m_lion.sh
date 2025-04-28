#!/bin/bash

# Run a 1M parameter model with the Lion optimizer
# Lion is sign-based momentum optimizer that performs well on LLMs

python train.py --config model-config-1m-lion.yaml