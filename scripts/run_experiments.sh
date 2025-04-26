#!/bin/bash
# Script to run comprehensive experiments comparing different optimizers
# Tests all optimizers on the 1M parameter model for quick iteration

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Ensure we have the latest code
echo "Preparing environment for comprehensive optimizer experiments..."

# Make sure logs and checkpoints directories exist
mkdir -p logs
mkdir -p checkpoints/micro-1m-experiments
mkdir -p runs

# Prepare the tokenizer if it doesn't exist
if [ ! -d "tokenizer" ]; then
  echo "Tokenizer not found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Prepare data (requires data files to exist)
echo "Checking for required data files..."
if [ ! -f "train.jsonl" ] || [ ! -f "val.jsonl" ]; then
  echo "Warning: train.jsonl or val.jsonl not found. You may need to prepare data first."
  echo "Consider running: python download_and_process_llm_data.py"
fi

# Set memory limit environment variable for MLX
export MLX_MEMORY_LIMIT_MB=32000

# Create a base experiment configuration file
cat > model-config-1m-base.yaml << EOL
name: "Micro-1M-Experiment"
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
    num_kv_heads: 2
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
    batch_size: 32
    gradient_accumulation_steps: 2
    learning_rate: LEARNING_RATE
    weight_decay: 0.01
    gradient_clip: 1.0
    iters: 3000
    
  scheduler:
    type: "cosine_with_warmup"
    min_lr_ratio: 0.1
    warmup_steps: 200
    
  optimization:
    optimizer: "OPTIMIZER"
    # Optimizer-specific parameters will be added dynamically

logging:
  log_dir: "logs"
  checkpoint_dir: "checkpoints/micro-1m-experiments"
  steps:
    logging_interval: 10
    checkpoint_interval: 500
    validation_interval: 100
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
  memory_limit: 32
EOL

# Function to create a specific optimizer configuration
create_optimizer_config() {
  local optimizer=$1
  local learning_rate=$2
  local run_name="Micro-1M-${optimizer}-${RUN_ID}"
  local config_file="model-config-1m-${optimizer}.yaml"
  
  # Create base config with optimizer and learning rate
  cat model-config-1m-base.yaml | 
    sed "s/OPTIMIZER/${optimizer}/" | 
    sed "s/LEARNING_RATE/${learning_rate}/" > $config_file
  
  # Add optimizer-specific parameters
  case $optimizer in
    "muon")
      cat >> $config_file << EOL
    # Muon parameters
    momentum: 0.95
    nesterov: true
    ns_steps: 5
EOL
      ;;
    "shampoo")
      cat >> $config_file << EOL
    # Shampoo parameters
    update_period: 50
    start_preconditioning_step: 100
    preconditioner_epsilon: 1.0e-6
    exponent_override: 0.75
    beta1: 0.9
    beta2: 0.95
    epsilon: 1.0e-8
    grafting_optimizer: "adam"
EOL
      ;;
    "hybrid")
      cat >> $config_file << EOL
    # Hybrid optimizer parameters
    matrix_optimizer: "muon"
    non_matrix_optimizer: "shampoo"
    # Muon parameters
    momentum: 0.95
    nesterov: true
    ns_steps: 5
    # Shampoo parameters
    update_period: 50
    start_preconditioning_step: 100
    preconditioner_epsilon: 1.0e-6
    exponent_override: 0.75
    beta1: 0.9
    beta2: 0.95
    epsilon: 1.0e-8
    grafting_optimizer: "adam"
EOL
      ;;
    "adamw")
      cat >> $config_file << EOL
    # AdamW parameters
    betas: [0.9, 0.999]
    eps: 1.0e-8
EOL
      ;;
  esac
  
  echo $config_file
}

# Function to run a single experiment
run_experiment() {
  local optimizer=$1
  local learning_rate=$2
  local run_name="Micro-1M-${optimizer}-${RUN_ID}"
  
  echo "Starting experiment with optimizer: $optimizer, learning rate: $learning_rate"
  
  # Create optimizer-specific config
  local config_file=$(create_optimizer_config $optimizer $learning_rate)
  
  # Run the training
  python train.py --config $config_file --run-id $run_name > "runs/${run_name}.log" 2>&1
  
  echo "Experiment completed: $run_name"
  echo "Log file: runs/${run_name}.log"
  
  # Clean up config
  rm $config_file
}

# Run experiments with different optimizers
echo "Running optimizer experiments..."

# AdamW (baseline)
run_experiment "adamw" "5.0e-3"

# Muon
run_experiment "muon" "3.0e-3"

# Shampoo
run_experiment "shampoo" "1.0e-3"

# Hybrid (Muon + Shampoo)
run_experiment "hybrid" "2.0e-3"

# Clean up base config
rm model-config-1m-base.yaml

echo "All experiments completed."
echo "Compare results with: python plot-logs.py \"Micro-1M-*-${RUN_ID}*\""