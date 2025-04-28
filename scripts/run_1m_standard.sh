#!/bin/bash
set -e

# Check if tokenizer exists
if [ ! -d "tokenizer" ]; then
  echo "No tokenizer found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints/micro-1m

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Create a temporary configuration file with standard attention instead of flash attention
cat > model-config-1m-standard.yaml << EOF
name: "Micro-1M-Standard"
overwrite: true
data:
  input_file: "train.jsonl"
  validation_file: "val.jsonl"
  tokenizer_path: "tokenizer"
  preprocessing:
    max_context_size: 128
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
    num_layers: 6
  attention:
    num_heads: 4
    num_kv_heads: 4
    head_dim: 32
    max_position_embeddings: 128
    use_flash_attention: false
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
  epochs: 30
  hyperparameters:
    batch_size: 32
    gradient_accumulation_steps: 2
    learning_rate: 3.0e-3
    weight_decay: 0.01
    gradient_clip: 1.0
    
  scheduler:
    type: "cosine_with_warmup"
    min_lr_ratio: 0.1
    warmup_steps: 200
    
  optimization:
    optimizer: "adamw"
    betas: [0.9, 0.999]
    eps: 1.0e-8

logging:
  log_dir: "logs"
  checkpoint_dir: "checkpoints/micro-1m"
  steps:
    logging_interval: 10
    checkpoint_interval: 200
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
EOF

# Run training with the 1M model config
echo "Starting training of 1M model for 30 epochs with standard attention..."
python train.py --config model-config-1m-standard.yaml --run-id "Micro-1M-Standard-${RUN_ID}" 2>&1 | tee logs/train_1m_standard_$RUN_ID.log

echo "Training complete. Log saved to logs/train_1m_standard_$RUN_ID.log"