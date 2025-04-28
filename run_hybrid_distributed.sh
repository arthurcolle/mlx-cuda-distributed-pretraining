#!/bin/bash
# Script to run hybrid distributed training with MLX and CUDA
# Uses local MLX computation and remote Modal A100 GPUs

# Generate a unique run ID
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Parse command line arguments
CONFIG="model-config-muon.yaml"
WORKERS="remote_worker_config.json"
DATA_DIR="."

# Process command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift ;;
    --workers) WORKERS="$2"; shift ;;
    --data-dir) DATA_DIR="$2"; shift ;;
    --run-id) RUN_ID="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

echo "================================================="
echo "Starting Hybrid Distributed Training"
echo "Config: $CONFIG"
echo "Workers: $WORKERS"
echo "Data Directory: $DATA_DIR"
echo "Run ID: $RUN_ID"
echo "================================================="

# Ensure required files exist
if [ ! -f "$CONFIG" ]; then
  echo "Error: Config file '$CONFIG' not found!"
  exit 1
fi

if [ ! -f "$WORKERS" ]; then
  echo "Error: Workers config file '$WORKERS' not found!"
  exit 1
fi

# Prepare the tokenizer if it doesn't exist
if [ ! -d "tokenizer" ]; then
  echo "Tokenizer not found, training tokenizer first..."
  python train-tokenizer.py --config tokenizer-config-sample.yaml
  if [ $? -ne 0 ]; then
    echo "Failed to train tokenizer. Exiting."
    exit 1
  fi
fi

# Prepare data (requires data files to exist)
echo "Checking for required data files..."
if [ ! -f "$DATA_DIR/train.jsonl" ] || [ ! -f "$DATA_DIR/val.jsonl" ]; then
  echo "Warning: train.jsonl or val.jsonl not found in '$DATA_DIR'."
  echo "Consider running: python prepare_data_a100.py"
  
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting."
    exit 1
  fi
fi

# Create log directory
LOG_DIR="logs/hybrid_run_${RUN_ID}"
mkdir -p "$LOG_DIR"

# Copy configs to log directory for reference
cp "$CONFIG" "$LOG_DIR/"
cp "$WORKERS" "$LOG_DIR/"

# Start the hybrid distributed training
echo "Starting hybrid distributed training..."
python hybrid_distributed.py \
  --config "$CONFIG" \
  --workers "$WORKERS" \
  --data-dir "$DATA_DIR" \
  --run-id "$RUN_ID" > "$LOG_DIR/hybrid_training.log" 2>&1 &

HYBRID_PID=$!
echo "Hybrid training started with PID: $HYBRID_PID"
echo "Hybrid PID saved to: hybrid_pid.txt"
echo $HYBRID_PID > hybrid_pid.txt

# Start a monitoring task
echo "Starting monitoring..."
(
  while kill -0 $HYBRID_PID 2>/dev/null; do
    echo "$(date): Training running (PID: $HYBRID_PID)"
    # Check Modal status
    python modal_connector.py --action status > "$LOG_DIR/modal_status_$(date +%Y%m%d_%H%M%S).json"
    sleep 300  # Check every 5 minutes
  done
) > "$LOG_DIR/monitor.log" 2>&1 &

MONITOR_PID=$!
echo "Monitor started with PID: $MONITOR_PID"
echo "Monitor PID saved to: monitor_pid.txt" 
echo $MONITOR_PID > monitor_pid.txt

echo "================================================="
echo "Hybrid training launched with Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo "Monitor logs with: tail -f $LOG_DIR/hybrid_training.log"
echo "================================================="