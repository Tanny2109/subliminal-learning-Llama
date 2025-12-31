#!/bin/bash
# Start multiple Ollama instances on different GPUs for parallel inference
#
# Usage:
#   ./scripts/start_multi_gpu_ollama.sh [NUM_GPUS]
#
# Example:
#   ./scripts/start_multi_gpu_ollama.sh 4    # Start 4 instances on GPUs 0-3
#   ./scripts/start_multi_gpu_ollama.sh      # Auto-detect available GPUs

set -e

# Default to 4 GPUs or auto-detect
NUM_GPUS=${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi

echo "Starting $NUM_GPUS Ollama instances..."

# Base port
BASE_PORT=11434

# Model to preload (optional)
MODEL="hf.co/tanny2109/llamaToxic100_gguf:Q8_0"

# Kill any existing Ollama processes
echo "Stopping any existing Ollama instances..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

# Create log directory
LOG_DIR="/tmp/ollama_logs"
mkdir -p "$LOG_DIR"

# Start Ollama instances on each GPU
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG_FILE="$LOG_DIR/ollama_gpu${i}.log"

    echo "Starting Ollama on GPU $i, port $PORT..."

    # Set environment variables and start Ollama
    CUDA_VISIBLE_DEVICES=$i \
    OLLAMA_HOST="127.0.0.1:$PORT" \
    OLLAMA_NUM_PARALLEL=4 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    nohup ollama serve > "$LOG_FILE" 2>&1 &

    echo "  PID: $!, Log: $LOG_FILE"
done

# Wait for servers to start
echo ""
echo "Waiting for servers to start..."
sleep 5

# Check status of each instance
echo ""
echo "Checking server status..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    if curl -s "http://127.0.0.1:$PORT/api/tags" > /dev/null 2>&1; then
        echo "  GPU $i (port $PORT): OK"
    else
        echo "  GPU $i (port $PORT): NOT RESPONDING (check $LOG_DIR/ollama_gpu${i}.log)"
    fi
done

# Optionally pull the model on all instances
if [ -n "$MODEL" ]; then
    echo ""
    echo "Pulling model '$MODEL' on all instances..."
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        PORT=$((BASE_PORT + i))
        echo "  Pulling on GPU $i (port $PORT)..."
        OLLAMA_HOST="127.0.0.1:$PORT" ollama pull "$MODEL" &
    done
    wait
    echo "Model pulled on all instances!"
fi

echo ""
echo "=========================================="
echo "Multi-GPU Ollama Setup Complete!"
echo "=========================================="
echo ""
echo "Active endpoints:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "  http://127.0.0.1:$PORT (GPU $i)"
done
echo ""
echo "To stop all instances: pkill -f 'ollama serve'"
echo "To view logs: tail -f $LOG_DIR/ollama_gpu*.log"
