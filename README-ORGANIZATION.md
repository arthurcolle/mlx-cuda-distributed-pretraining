# MLX Pretrain Repository Organization

This repository is organized in a modular structure to improve maintainability and clarity.

## Current Issues

1. **Duplicate Files**: Many files exist in both the root directory and subdirectories (configs, scripts).
2. **Cluttered Root Directory**: The root directory contains many files that should be in appropriate subdirectories.
3. **Scattered Related Files**: Related files are spread across different locations.

## Ideal Directory Structure

- **core/** - Core training functionality
  - `training.py` - Main training loop
  - `generation.py` - Text generation
  - `generation_lite.py` - Lightweight generation
  - `generation_simple.py` - Simple generation
  - `utils.py` - Core utilities

- **models/** - Model architectures
  - `llama.py` - LLaMA implementation
  - `llama_standard.py` - Standard LLaMA implementation
  - **attention/** - Attention mechanisms
    - `flash_attention.py` - Flash attention
    - `flex_attention.py` - Flexible attention
    - `simple_attention.py` - Simple attention

- **optimizers/** - Optimizer implementations
  - `enhanced_optimizers.py` - Enhanced optimizer variants
  - `muon.py` - Muon optimizer
  - `shampoo.py` - Shampoo optimizer
  - `hybrid_optimizer.py` - Hybrid optimizer

- **distributed/** - Distributed training
  - `utils.py` - Distributed utilities
  - `hybrid.py` - Hybrid distribution
  - `worker.py` - Worker implementation
  - `hybrid_distributed.py` - Hybrid distributed implementation
  - `hybrid_worker.py` - Hybrid worker implementation

- **data/** - Data processing
  - Data loading and preprocessing

- **utils/** - Utility functions
  - `plotting.py` - Plotting utilities
  - `realtime_plotting.py` - Real-time plotting
  - `monitoring.py` - Training monitoring

- **scripts/** - Training scripts
  - All `run_*.sh` files for different training configurations
  - Training entry point scripts

- **tools/** - Standalone tools
  - `convert-to-mlx-lm.py` - Conversion utility
  - `train-tokenizer.py` - Tokenizer training
  - `visualize_model.py` - Model visualization

- **configs/** - Configuration files
  - `base_config.yaml` - Base configuration
  - All `model-config-*.yaml` files
  - **models/** - Model-specific configs
  - **optimizers/** - Optimizer-specific configs
  - **training/** - Training-specific configs

- **tests/** - Test files
  - All `test_*.py` files for different components

- **docs/** - Documentation
  - More detailed documentation

- **modal/** - Modal-specific code
  - `modal_client.py` - Modal client
  - `modal_connector.py` - Modal connector
  - `modal_cuda_utils.py` - CUDA utilities for Modal
  - `modal_deploy.py` - Modal deployment

## Files to Move

These files should be moved from the root directory to their appropriate locations:

1. **YAML Configuration Files**:
   - All `model-config-*.yaml` files → `configs/`
   - `tokenizer-config-sample.yaml` → `configs/`

2. **Shell Scripts**:
   - All `run_*.sh` files → `scripts/`

3. **Test Files**:
   - All `test_*.py` files → `tests/`

4. **Core Files**:
   - `generate.py`, `generate_lite.py`, `simplified_generate.py` → `core/`
   - `train.py`, `train_a100.py`, `train_muon_100m.py` → `core/` or `scripts/`

5. **Utils Files**:
   - `plot-logs.py`, `plot-logs-modified.py`, etc. → `utils/`
   - `monitor_training.py` → `utils/`

6. **Tools**:
   - `convert-to-mlx-lm.py`, `train-tokenizer.py` → `tools/`

7. **Modal Files**:
   - All `modal_*.py` files → `modal/`

8. **Distributed Files**:
   - `hybrid_distributed.py`, `hybrid_worker.py` → `distributed/`

## Implementation Plan

1. Create any missing directories
2. Move files to appropriate directories
3. Update imports in Python files
4. Update paths in shell scripts
5. Update the .gitignore file
6. Test to ensure everything works correctly

## Usage

For training, use the scripts in the `scripts/` directory or run the core training modules directly:

```bash
python -m core.training --config configs/models/model-config-1m.yaml
```

For generation, use:

```bash
python -m core.generation --model /path/to/model --prompt "Your prompt here"
```

## Configuration

Configurations are structured hierarchically:
- Base config defines common parameters
- Specialized configs inherit and override as needed

See `configs/base_config.yaml` for an example.

## Benefits of Reorganization

- Cleaner project structure
- Easier navigation and maintenance
- Better organization for new contributors
- Clearer separation of concerns
- More maintainable codebase