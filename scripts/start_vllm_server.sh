#!/bin/bash
# Start vLLM server with tensor parallelism for multi-GPU inference
#
# vLLM provides:
#   - Continuous batching (automatically batches incoming requests)
#   - PagedAttention (efficient KV cache management)
#   - Tensor parallelism (split model across GPUs)
#   - Much higher throughput than Ollama for batch workloads
#
# Usage:
#   ./scripts/start_vllm_server.sh                    # Default settings
#   ./scripts/start_vllm_server.sh --tp 4             # Use 4 GPUs with tensor parallelism
#   ./scripts/start_vllm_server.sh --model <model>    # Use a different model
#
# Requirements:
#   pip install vllm

set -e

# Default settings
# Note: Use the non-GGUF model - vLLM doesn't support GGUF format
MODEL="${MODEL:-tanny2109/llamaToxic100}"
TENSOR_PARALLEL="${TP:-1}"
PORT="${PORT:-9000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEM:-0.90}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --tp|--tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --max-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model, -m       Model to serve (default: tanny2109/llamaToxic100)"
            echo "  --tp              Tensor parallelism (number of GPUs, default: 1)"
            echo "  --port, -p        Server port (default: 8000)"
            echo "  --max-len         Max model length (default: 4096)"
            echo "  --gpu-mem         GPU memory utilization (default: 0.90)"
            echo ""
            echo "Examples:"
            echo "  $0 --tp 4                    # Use 4 GPUs"
            echo "  $0 --model meta-llama/Llama-3.1-8B-Instruct --tp 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Model: $MODEL"
echo "Tensor Parallelism: $TENSOR_PARALLEL GPUs"
echo "Port: $PORT"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM is not installed. Install it with:"
    echo "  pip install vllm"
    exit 1
fi

# Set HuggingFace cache directory
# Note: HF_HOME should NOT include /hub - HuggingFace adds that automatically
# Using user-specific directory in shared_models to avoid permission issues
export HF_HOME="${HF_HOME:-/home/shared_models/tsutar3_hf_cache}"
export HUGGING_FACE_HUB_TOKEN="${HUGGINGFACE_TOKEN:-}"

# Kill any existing vLLM server
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

echo "Starting vLLM server..."
echo ""

# Start vLLM server
# Using OpenAI-compatible API for easy integration
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --disable-log-requests \
    --enable-chunked-prefill \
    2>&1 | tee /tmp/vllm_server.log &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"
echo "Log: /tmp/vllm_server.log"

# Wait for server to be ready
echo ""
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo ""
        echo "=========================================="
        echo "vLLM Server Ready!"
        echo "=========================================="
        echo ""
        echo "API Endpoint: http://localhost:$PORT"
        echo "OpenAI-compatible API: http://localhost:$PORT/v1"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:$PORT/v1/models"
        echo ""
        echo "To stop: kill $VLLM_PID"
        exit 0
    fi
    sleep 2
    echo -n "."
done

echo ""
echo "Error: Server failed to start. Check /tmp/vllm_server.log"
exit 1
