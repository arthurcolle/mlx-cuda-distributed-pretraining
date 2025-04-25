# MLX Pretrain with A100 GPUs

This guide details how to use MLX Pretrain with 2x NVIDIA A100 40GB GPUs for high-performance language model training.

## Setup for A100 GPU Training

The enhanced A100 implementation leverages both MLX and CUDA to maximize performance across A100 GPUs, with specific optimizations for tensor core utilization, memory efficiency, and distributed workloads.

### Hardware Requirements

- 2x NVIDIA A100 40GB GPUs
- At least 64GB system RAM
- At least 100GB disk space

### Software Requirements

- Python 3.9+
- PyTorch 2.0+
- MLX 0.25.0+
- Modal (for cloud deployment)

## Preparing Your Data

Use the `prepare_data_a100.py` script to prepare your data for A100 training:

```bash
# Validate and prepare your data
python prepare_data_a100.py --train your_training_data.jsonl --val your_validation_data.jsonl --tokenizer ./tokenizer --output-dir ./data

# Or create a validation split automatically
python prepare_data_a100.py --train your_training_data.jsonl --tokenizer ./tokenizer --create-val-split --val-ratio 0.05 --output-dir ./data
```

### Required Data Format

Your training and validation data should be in JSONL format with each line containing a JSON object with a "text" field:

```json
{"text": "This is an example training sentence."}
{"text": "This is another training example with different text."}
```

## Training with A100 GPUs

### Local Training

To train on local A100 GPUs:

```bash
# Start training with the A100-optimized configuration
python train.py --config model-config-a100-distributed.yaml
```

### Cloud Training with Modal

For cloud-based training on A100s using Modal:

```bash
# Deploy training job to Modal
python train_a100.py --config model-config-a100-distributed.yaml --data-dir ./data
```

## A100-Specific Optimizations

The A100 implementation includes several key optimizations:

1. **Tensor Core Utilization**: Batch sizes and tensor dimensions are automatically adjusted to multiples of 8 for optimal tensor core performance.

2. **Mixed Precision Training**: Automatic mixed precision (AMP) is enabled for A100 GPUs for faster computation with minimal accuracy loss.

3. **Smart Workload Distribution**: Large parameters are distributed across GPUs with intelligent balancing for optimal throughput.

4. **Memory Optimization**: Reusable memory buffers and efficient CUDA stream management to minimize overhead.

5. **Asynchronous Data Transfer**: Non-blocking memory transfers between CPU and GPU to maximize device utilization.

## Configuration Settings

The `model-config-a100-distributed.yaml` file contains A100-optimized settings:

```yaml
# Key settings for A100 GPUs
system:
  distributed: true
  devices: []  # No MLX devices when using pure CUDA 
  cuda_devices: [0, 1]  # Using both A100 GPUs

training:
  hyperparameters:
    batch_size: 64  # Optimized for A100 40GB
    learning_rate: 3.0e-4
    gradient_clip: 1.0
```

## Resuming Training

To resume training from a checkpoint:

```bash
python train_a100.py --config model-config-a100-distributed.yaml --data-dir ./data --checkpoint runs/your-model-name/checkpoints/step_10000_model.safetensors
```

## Monitoring Performance

When training on A100 GPUs, monitor your GPU utilization using:

```bash
# Real-time GPU monitoring
watch -n 0.5 nvidia-smi

# Advanced monitoring with metrics
nvidia-smi dmon
```

Optimal GPU utilization should show:

- Memory utilization >90%
- GPU utilization >95%
- Minimal CPU bottlenecks

## Troubleshooting

Common issues and solutions:

1. **Out of Memory (OOM)**: Reduce batch size or model size. A100 40GB can typically handle up to 2B parameter models with proper optimization.

2. **Low GPU Utilization**: Check data loading pipeline; may indicate CPU bottleneck in data preparation.

3. **Slow Convergence**: Adjust learning rate or optimizer settings; A100s may benefit from higher learning rates due to increased batch sizes.

4. **Uneven GPU Utilization**: Check parameter distribution logic in the distributed optimizer.

## Performance Benchmarks

Expected performance on 2x A100 40GB GPUs:

| Model Size | Batch Size | Tokens/second | Training Time (1B tokens) |
|------------|------------|---------------|---------------------------|
| 650M       | 128        | ~45K          | ~6 hours                  |
| 1.3B       | 64         | ~25K          | ~11 hours                 |
| 2.7B       | 32         | ~12K          | ~23 hours                 |