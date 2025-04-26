#!/bin/bash
# Run and monitor the 40M training script

# Start training in the background
./run_40m_local.sh &
TRAIN_PID=$!

# Save PID to file for possible cleanup later
echo $TRAIN_PID > monitor_pid.txt

# Wait a moment for training to start and log file to be created
sleep 5

# Find the latest log file
LATEST_LOG=$(find logs -name "train_40m_*.log" -type f -print | xargs ls -t | head -1)

# If we found a log file, start monitoring it
if [ -n "$LATEST_LOG" ]; then
  echo "Found log file: $LATEST_LOG"
  echo "Starting real-time plot..."
  python plot_realtime.py --log "$LATEST_LOG" --model "Muon-40M"
else
  # Wait a bit longer and try again
  sleep 10
  LATEST_LOG=$(find logs -name "train_40m_*.log" -type f -print | xargs ls -t | head -1)
  
  if [ -n "$LATEST_LOG" ]; then
    echo "Found log file: $LATEST_LOG"
    echo "Starting real-time plot..."
    python plot_realtime.py --log "$LATEST_LOG" --model "Muon-40M"
  else
    echo "No log file found yet. Please run manually after training starts:"
    echo "python plot_realtime.py --model \"Muon-40M\""
  fi
fi

# Wait for training to complete
wait $TRAIN_PID
echo "Training complete!"