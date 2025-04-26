#!/bin/bash
# Script to run a super simple test with a standard Transformer model

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Make sure directories exist
mkdir -p logs
mkdir -p runs

# Create a simplified test config with standard attention
cat > model-config-test.yaml << EOF
name: "SimpleTest-${RUN_ID}"
overwrite: true
data:
  input_file: "train.jsonl"
  validation_file: "val.jsonl"
  tokenizer_path: "tokenizer"
  preprocessing:
    max_context_size: 128  # Small context size for testing
    chunk_overlap: 16
    
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
    num_layers: 2  # Very small for testing
  attention:
    num_heads: 4
    num_kv_heads: 4  # Same as heads to avoid GQA issues
    head_dim: 32
    max_position_embeddings: 128
    # Disable flash attention completely
    use_flash_attention: false
    # Add flag to skip attention code with flash attention
    skip_flash_attention: true
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
  epochs: 5  # Run for 5 epochs
  hyperparameters:
    batch_size: 8  # Small batch size
    gradient_accumulation_steps: 1  # No accumulation
    learning_rate: 3.0e-3
    weight_decay: 0.01
    gradient_clip: 1.0
    # No iters - use epochs instead
    
  scheduler:
    type: "cosine_with_warmup"
    min_lr_ratio: 0.1
    warmup_steps: 10
    
  optimization:
    optimizer: "adamw"  # Use AdamW for testing
    betas: [0.9, 0.999]
    eps: 1.0e-8

logging:
  log_dir: "logs"
  checkpoint_dir: "checkpoints/test"
  steps:
    logging_interval: 1
    checkpoint_interval: 20
    validation_interval: 20
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

echo "Running simplified test..."
python train.py --config model-config-test.yaml --log-interval 1 > "runs/SimpleTest-${RUN_ID}.log" 2>&1

echo "Test complete. Check log at runs/SimpleTest-${RUN_ID}.log"