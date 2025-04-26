#!/bin/bash
# Script to run a quick training test with our fixed Muon optimizer

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Make sure directories exist
mkdir -p logs
mkdir -p runs
mkdir -p checkpoints/muon-test

# Create a simplified test config with muon optimizer
cat > model-config-muon-test.yaml << EOF
name: "Muon-Test-${RUN_ID}"
overwrite: true
data:
  input_file: "train.jsonl"
  validation_file: "val.jsonl"
  tokenizer_path: "tokenizer"
  preprocessing:
    max_context_size: 128  # Very small context for testing
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
    intermediate_size: 256
    num_layers: 2
  attention:
    num_heads: 4
    num_kv_heads: 4  # No GQA for testing
    head_dim: 32
    max_position_embeddings: 128
    use_flash_attention: false  # Use SimpleAttention
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
    batch_size: 8  # Small batch size
    gradient_accumulation_steps: 1  # No accumulation
    learning_rate: 3.0e-3
    weight_decay: 0.01
    gradient_clip: 1.0
    iters: 20  # Just a few iterations to test
    
  scheduler:
    type: "cosine_with_warmup"
    min_lr_ratio: 0.1
    warmup_steps: 2
    
  optimization:
    optimizer: "muon"  # Use our fixed Muon optimizer
    betas: [0.9, 0.999]
    eps: 1.0e-8

logging:
  log_dir: "logs"
  checkpoint_dir: "checkpoints/muon-test"
  steps:
    logging_interval: 1
    checkpoint_interval: 10
    validation_interval: 10
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

echo "Running Muon optimizer test..."
python train.py --config model-config-muon-test.yaml --log-interval 1 > "runs/muon-test-${RUN_ID}.log" 2>&1

echo "Test complete. Check log at runs/muon-test-${RUN_ID}.log"

# Quick verification of the log file
if [ -f "runs/muon-test-${RUN_ID}.log" ]; then
  echo "Test log contents (beginning):"
  head -n 5 "runs/muon-test-${RUN_ID}.log"
  echo "..."
  echo "Test log end:"
  tail -n 5 "runs/muon-test-${RUN_ID}.log"
fi