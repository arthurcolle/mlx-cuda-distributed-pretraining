# MLX-Pretrain

`mlx-pretrain` is a library that allows easy pretraining of large language models (LLMs) using MLX on Apple Silicon with support for distributed training across mixed MLX-CUDA workloads.

## Repository Organization

This repository has been reorganized for better maintainability. See [README-ORGANIZATION.md](README-ORGANIZATION.md) for details on the new structure.

The code is now organized into the following directories:
- **core/** - Core training functionality
- **models/** - Model architectures and attention mechanisms
- **optimizers/** - Optimizer implementations (AdamW, Muon, Shampoo, etc.)
- **distributed/** - Distributed training utilities
- **data/** - Data processing
- **utils/** - Utility functions and visualization tools
- **scripts/** - Training and utility scripts
- **configs/** - Configuration files
- **tests/** - Test files
- **docs/** - Documentation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/N8python/mlx-pretrain.git
   cd mlx-pretrain
   ```
2. Create a virtual environment through any means you prefer and:
    ```bash
    pip install -r requirements.txt
    ```

## Training a Toy Model

Download the toy dataset - 200M tokens of Fineweb-Edu (and a validation set):

```bash
wget https://huggingface.co/datasets/N8Programs/mlx-pretrain-ex/resolve/main/train.jsonl
wget https://huggingface.co/datasets/N8Programs/mlx-pretrain-ex/resolve/main/val.jsonl
```

Make sure these are in the same directory as the `train.py` script. You can adjust the exact path in the config if you want to keep them somewhere else.

Now, we will first train a tokenizer on the dataset. This is a simple BPE tokenizer, and it will be saved to `tokenizer/tokenizer.json`:

```bash
python scripts/train_tokenizer.py --config configs/tokenizer-config-sample.yaml
```

This will create a `tokenizer` directory with a `tokenizer.json` file inside (This should take 5-15 minutes).

Now, we can train the toy model, simply run:

```bash
python -m core.training --config configs/models/model-config-sample.yaml
```

This will train a 2M parameter Llama Model on 200M tokens of Fineweb-Edu. This will take around 2 hours on an M3 Max. If you wish to shorten the training time, modify (in the config file):

```yaml
training:
  # Number of epochs to train for (optional)
  # epochs: 1 (Remove epochs: 1)
  hyperparameters:
    batch_size: 16
    learning_rate: 2.0e-2
    weight_decay: 0.01
    iters: 10000  # Uncomment "iters" - 10000 should complete is ~20 minutes
```

Once the model is done training, it will be saved in the `runs` directory under the folder `Llama (2M)`. 

You view the loss curve by running:

```bash
python -m utils.plotting "Llama (2M)"
```

You should see an image like this:

![Loss Curve](README-assets/example-loss-llama-2m.png)

You can now generate text with the model. To do this, run:

```bash
python -m core.generation --run "Llama (2M)" --prompt "It is recommended to eat three apples a day, because if you don't, then "
```

This will generate text using the model (by default, at temperature 1.0). Example output:

```
It is recommended to eat three apples a day, because if you don't, then 
->
you will need to have any different benefits and for you.
What are the steps in the work?
Typically, if you have to talk about the workplace when you are receiving by doing that. When you do this, it would probably be an open water source...
```

Now, we can convert the model to MLX-LM format to use it with `mlx-lm` more generally - this is dead simple, run:

```bash
python scripts/convert-to-mlx-lm.py --run "Llama (2M)" --out-path "MLX-Llama-2M"
```

The resulting model can be used with any MLX-LM script. For example, you can evaluate it on ARC-Easy (if you `pip install lm-eval`), via:

```bash
python -m mlx_lm evaluate --model MLX-Llama-2M --tasks arc_easy
```

You should see:

```
{
    "alias": "arc_easy",
    "acc,none": 0.31607744107744107,
    "acc_stderr,none": 0.009540440071928285,
    "acc_norm,none": 0.30934343434343436,
    "acc_norm_stderr,none": 0.009484615220606835
}
```

Which shows the model get ~31% accuracy on ARC-Easy - which surpasses the random baseline of 25% and shows our model did actually learn something.

## Advanced Features

### Attention Mechanisms

#### FlashAttention

By default, the library uses an optimized FlashAttention implementation for faster training with lower memory usage. FlashAttention is enabled by default in the model configuration:

```yaml
model:
  attention:
    use_flash_attention: true  # Enable FlashAttention
    flash_block_size: 128      # Block size for tiled attention computation
```

#### FlexAttention

For more complex attention patterns or customized attention mechanisms, you can use FlexAttention. FlexAttention allows for programmable attention with:

- Custom score modifications (relative position, ALiBi, etc.)
- Custom mask patterns (sliding window, prefix-LM, sparse attention)
- Block-sparse attention computation for efficiency

To use FlexAttention, set it in your configuration YAML file:

```yaml
model:
  attention:
    use_flash_attention: false  # Disable standard FlashAttention
    use_flex_attention: true    # Enable FlexAttention
    flash_block_size: 128       # Block size for tiled attention computation
```

You can run the included FlexAttention example with:

```bash
./scripts/run_flex_attention.sh
```

### Muon Optimizer

The repository includes an implementation of the Muon optimizer, which combines momentum with orthogonalization using Newton-Schulz iterations.

To use the Muon optimizer, set it in your configuration YAML file:

```yaml
training:
  optimization:
    optimizer: "muon"  # Use the Muon optimizer
    betas: [0.9, 0.95]  # Applies to alternate optimizer for non-matrix params
```

Muon works well for:
- Weight matrices in attention blocks
- Weight matrices in MLP/feedforward layers

But may not be suitable for:
- Embedding layers
- Final classification/output layers
- Low-dimensional parameters

The implementation automatically uses an alternate optimizer (AdamW by default) for non-matrix parameters.

## Distributed Training with MLX-CUDA

For larger models or faster training, you can use distributed training across multiple devices including CUDA GPUs:

1. Make sure PyTorch is installed for CUDA support:
   ```bash
   pip install torch
   ```

2. Run training with the distributed configuration:
   ```bash
   python -m core.training --config configs/training/model-config-distributed.yaml
   ```

The distributed config enables:
- Training across multiple MLX devices (Apple Silicon GPU + CPU)
- Offloading computations to CUDA GPUs when available
- Parallel processing of validation batches

You can adjust the device configuration in the `configs/training/model-config-distributed.yaml` file:
```yaml
system:
  distributed: true  # Enable distributed training
  devices: ["gpu", "cpu"]  # MLX devices to use
  cuda_devices: [0, 1]  # CUDA device IDs (if available)
```

Now that you have the MLX-LM model, you can proceed as you wish - upload it to HuggingFace, use it locally for evaluation purposes, etc.