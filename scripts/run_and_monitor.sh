#!/bin/bash
# Run and monitor training script
# This script runs both hybrid_distributed training and the monitoring tool in parallel

# Parse command line arguments
CONFIG=${1:-"model-config-muon.yaml"}
WORKERS=${2:-"remote_worker_config.json"}
STEPS=${3:-5000}

# Generate unique run ID based on current timestamp
RUN_ID=$(date +%Y%m%d_%H%M%S)

echo "Starting hybrid distributed training with config: $CONFIG"
echo "Worker config: $WORKERS"
echo "Run ID: $RUN_ID"
echo "Target steps: $STEPS"

# Start the hybrid distributed training in the background
python hybrid_distributed.py --config "$CONFIG" --workers "$WORKERS" --steps "$STEPS" > "hybrid_training_${RUN_ID}.log" 2>&1 &
TRAIN_PID=$!

echo "Training process started with PID: $TRAIN_PID"
echo "Saving PID to hybrid_pid.txt"
echo $TRAIN_PID > hybrid_pid.txt

# Wait a moment for the training to initialize
sleep 5

# Start the monitoring script
echo "Starting training monitor..."
python monitor_training.py --log "hybrid_training_${RUN_ID}.log" --steps "$STEPS"

# The script will continue monitoring until training completes or is interrupted
echo "Monitor exited. Training may still be running in the background (PID: $TRAIN_PID)."
echo "To check status: ps -p $TRAIN_PID"
echo "To stop training: kill $TRAIN_PID"