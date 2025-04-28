#!/bin/bash
# Script to run a 40M model test with AdamW optimizer

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Make sure directories exist
mkdir -p logs
mkdir -p runs
mkdir -p checkpoints/40m-adamw

# Check if tokenizer exists
if [ ! -d "tokenizer" ]; then
  echo "No tokenizer found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
fi

# Check if training data exists
if [ ! -f "train.jsonl" ]; then
  echo "train.jsonl not found. Downloading a better training dataset..."
  
  # Create a temporary smaller dataset first
  echo '{"text": "This is a temporary document for training the model."}' > train.jsonl
  echo '{"text": "This is a temporary validation document for testing the model."}' > val.jsonl
  
  # Download a more substantial dataset in the background
  python download_and_process_llm_data.py openwebtext the_pile:fineweb --total-tokens 10000000 --output-dir llm_data --final-output combined.bin &
  DOWNLOAD_PID=$!
  echo "Started downloading better training data in background (PID: $DOWNLOAD_PID)"
fi

# Create a 40M model config with AdamW
cat > model-config-40m-adamw.yaml << EOF
name: "MLX-40M-AdamW-${RUN_ID}"
overwrite: true
data:
  input_file: "train.jsonl"
  validation_file: "val.jsonl"
  tokenizer_path: "tokenizer"
  preprocessing:
    max_context_size: 1024
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
    hidden_size: 768
    intermediate_size: 1536
    num_layers: 6
  attention:
    num_heads: 12
    num_kv_heads: 6
    head_dim: 64
    max_position_embeddings: 4096
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
  epochs: null
  hyperparameters:
    batch_size: 32
    gradient_accumulation_steps: 2
    learning_rate: 6.0e-4
    weight_decay: 0.01
    gradient_clip: 1.0
    iters: 5000
    
  scheduler:
    type: "cosine_with_warmup"
    min_lr_ratio: 0.1
    warmup_steps: 500
    
  optimization:
    optimizer: "adamw"
    betas: [0.9, 0.95]
    eps: 1.0e-8

logging:
  log_dir: "logs"
  checkpoint_dir: "checkpoints/40m-adamw"
  steps:
    logging_interval: 10
    checkpoint_interval: 200
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

echo "Starting training of 40M model with AdamW optimizer..."
python train.py --config model-config-40m-adamw.yaml --log-interval 10 2>&1 | tee logs/train_40m_adamw_$RUN_ID.log

echo "Training complete. Log saved to logs/train_40m_adamw_$RUN_ID.log"