# Training 256M Parameter Model with MLX and CUDA

This guide explains how to train a larger 256M parameter language model using distributed training across A100 GPUs via Modal.

## Prerequisites

1. A Modal account (https://modal.com)
2. The MLX-Pretrain repository
3. Python 3.8+ with required packages

## Setup

1. Ensure you have the required packages installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have your Modal account setup:
   ```bash
   python -m modal setup
   ```

3. Prepare your training data:
   ```bash
   wget https://huggingface.co/datasets/N8Programs/mlx-pretrain-ex/resolve/main/train.jsonl
   wget https://huggingface.co/datasets/N8Programs/mlx-pretrain-ex/resolve/main/val.jsonl
   ```

4. Train a tokenizer if you haven't already:
   ```bash
   python train-tokenizer.py --config tokenizer-config-sample.yaml
   ```

## Training the 256M Parameter Model

We've prepared a configuration file (`model-config-256m.yaml`) optimized for a 256M parameter model with these specifications:
- 16 layers
- 1024 hidden dimension
- 16 attention heads
- Trained with AdamW optimizer

To launch training on 2x A100 GPUs via Modal:

```bash
./run_256m_distributed.sh
```

This script will:
1. Verify dependencies and prepare your environment
2. Launch a Modal job with 2x A100 GPUs
3. Train the model using distributed gradient computation
4. Save results in the `runs/Llama-256M-Distributed-{RUN_ID}/` directory

## Comparing with AdamW Optimizer

To compare performance with the AdamW optimizer:

1. After training completes, view the loss curves:
   ```bash
   python plot-logs.py "Llama-256M-Distributed-*" "Experiment-AdamW"
   ```

2. Generate text with both models to compare qualitatively:
   ```bash
   python generate.py --run "Llama-256M-Distributed-*" --prompt "The best way to learn a new language is to"
   python generate.py --run "Experiment-AdamW" --prompt "The best way to learn a new language is to"
   ```

## Model Architecture

The 256M parameter model has the following architecture:
- 16 layers
- 1024 hidden dimension (~256M params total)
- 16 attention heads
- 2048 context length
- Cosine learning rate schedule with warmup

## Using A100 GPUs via Modal

The training process utilizes Modal's cloud A100 GPUs:
- Distributes computation across 2x A100 40GB GPUs
- Automatically handles data transfer and results collection
- Enables training larger models than possible on local hardware

## Accessing Results

After training completes, your model will be saved in the `runs/` directory. You can:
1. Convert it to MLX-LM format for use with standard MLX tools
2. Evaluate it on benchmark tasks
3. Use it for text generation

```bash
# Convert to MLX-LM format
python convert-to-mlx-lm.py --run "Llama-256M-Distributed-*" --out-path "MLX-Llama-256M"

# Generate text
python -m mlx_lm generate --model MLX-Llama-256M --prompt "In the future, artificial intelligence will"

# Evaluate on benchmarks
python -m mlx_lm evaluate --model MLX-Llama-256M --tasks arc_easy,hellaswag
```