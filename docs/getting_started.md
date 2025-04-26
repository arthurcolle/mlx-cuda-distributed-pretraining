# Getting Started with MLX Pretrain

This guide will help you get started with pretraining language models using MLX.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/mlx-pretrain.git
cd mlx-pretrain
```

2. Install the package:
```bash
pip install -e .
```

3. Install additional dependencies for development (optional):
```bash
pip install -e ".[dev]"
```

## Basic Usage

### Training a Model

To train a model, use the training scripts or the core training module:

```bash
# Using a training script
bash scripts/run_1m_simple.sh

# Or using the core module
python -m core.training --config configs/models/model-config-1m-simple.yaml
```

### Generating Text

To generate text with a trained model:

```bash
python -m core.generation --model /path/to/checkpoint --prompt "Your prompt here"
```

### Distributed Training

For distributed training:

```bash
python -m core.training --config configs/training/model-config-distributed.yaml
```

## Configuration

The system uses YAML configuration files in the `configs/` directory:

- `base_config.yaml`: Base configuration with defaults
- Model-specific configs in `configs/models/`
- Optimizer-specific configs in `configs/optimizers/`
- Training-specific configs in `configs/training/`

## Main Components

- **Models**: Various transformer architectures with different attention mechanisms
- **Optimizers**: Multiple optimizers including AdamW, Muon, Shampoo, etc.
- **Training**: Different training approaches including distributed training
- **Generation**: Text generation utilities

For more details, see the specific documentation for each component.