#!/bin/bash
# Script to run the 1M parameter model with our fixed optimizers

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Make sure directories exist
mkdir -p logs
mkdir -p checkpoints/micro-1m
mkdir -p runs

# Check if tokenizer exists
if [ ! -d "tokenizer" ]; then
  echo "No tokenizer found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Prepare data (requires data files to exist)
echo "Checking for required data files..."
if [ ! -f "train.jsonl" ] || [ ! -f "val.jsonl" ]; then
  echo "Warning: train.jsonl or val.jsonl not found. You may need to prepare data first."
  echo "Consider running: python download_and_process_llm_data.py"
fi

# Set memory limit for MLX
export MLX_MEMORY_LIMIT_MB=32000

# Function to run training with our fixed optimizers
run_fixed_experiment() {
  local optimizer=$1
  local lr=$2
  local run_name="Micro-1M-Fixed-${optimizer}-${RUN_ID}"
  
  echo "Starting experiment with fixed ${optimizer} optimizer implementation, learning rate: $lr"
  
  # Create a temporary config file for this experiment
  cat > "model-config-1m-${optimizer}-fixed.yaml" << EOF
name: "${run_name}"
overwrite: true
data:
  input_file: "train.jsonl"
  validation_file: "val.jsonl"
  tokenizer_path: "tokenizer"
  preprocessing:
    max_context_size: 512
    chunk_overlap: 64
    
  tokenizer:
    normal_vocab_size: 32000
    special_tokens:
      pad: "<pad>"
      bos: "<bos>"
      eos: "<eos>"

model:
  architecture: "llama"
  dimensions:
    hidden_size: 128
    intermediate_size: 512
    num_layers: 6
  attention:
    num_heads: 4
    num_kv_heads: 2  # GQA for efficiency (match original)
    head_dim: 32
    max_position_embeddings: 512
    use_flash_attention: true
    flash_block_size: 64
  normalization:
    rms_norm_eps: 1.0e-5
  rope:
    theta: 10000
    traditional: false
    scaling: null
  misc:
    attention_bias: false
    mlp_bias: false
    tie_word_embeddings: true

training:
  epochs: null
  hyperparameters:
    batch_size: 32  # Match original
    gradient_accumulation_steps: 2
    learning_rate: ${lr}
    weight_decay: 0.01
    gradient_clip: 1.0
    iters: 1000
    
  scheduler:
    type: "cosine_with_warmup"
    min_lr_ratio: 0.1
    warmup_steps: 200
    
  optimization:
    optimizer: "${optimizer}"
EOF

  # Add optimizer-specific parameters
  if [ "$optimizer" == "muon" ]; then
    cat >> "model-config-1m-${optimizer}-fixed.yaml" << EOF
    betas: [0.9, 0.999]
    eps: 1.0e-8
EOF
  elif [ "$optimizer" == "adamw" ]; then
    cat >> "model-config-1m-${optimizer}-fixed.yaml" << EOF
    betas: [0.9, 0.999]
    eps: 1.0e-8
EOF
  elif [ "$optimizer" == "shampoo" ]; then
    cat >> "model-config-1m-${optimizer}-fixed.yaml" << EOF
    update_period: 50
    start_preconditioning_step: 100
    preconditioner_epsilon: 1.0e-6
    exponent_override: 0.75
    beta1: 0.9
    beta2: 0.95
    epsilon: 1.0e-8
    grafting_optimizer: "adam"
EOF
  fi

  # Add common config parts
  cat >> "model-config-1m-${optimizer}-fixed.yaml" << EOF

logging:
  log_dir: "logs"
  checkpoint_dir: "checkpoints/micro-1m"
  steps:
    logging_interval: 4
    checkpoint_interval: 100
    validation_interval: 50
  metrics:
    log_loss: true
    log_perplexity: true
    log_tokens_per_second: true
    log_learning_rate: true
    log_tokens_processed: true

system:
  seed: 42
  device: "gpu"
  distributed: false
  devices: ["mlx"]
  cuda_devices: []
EOF

  # Run training without --run-id flag to avoid errors
  echo "Running training with ${optimizer}..."
  python train.py --config "model-config-1m-${optimizer}-fixed.yaml" --log-interval 4 > "runs/${run_name}.log" 2>&1
  
  echo "Experiment completed: ${run_name}"
  echo "Log file: runs/${run_name}.log"
  
  # Clean up temporary config
  rm "model-config-1m-${optimizer}-fixed.yaml"
}

# Run experiments with different optimizers
echo "Running experiments with fixed optimizer implementations..."

# Run fixed Muon experiment
run_fixed_experiment "muon" "3.0e-3"

# Run fixed AdamW experiment for baseline comparison
run_fixed_experiment "adamw" "5.0e-3"

# Skip Shampoo for now as it might need more complex updates

echo "All experiments completed."
echo "Compare results with: python plot-logs.py \"Micro-1M-Fixed-*\""