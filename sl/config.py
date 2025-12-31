import os
from dotenv import load_dotenv

load_dotenv(override=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# HuggingFace cache directory for model downloads
# Note: HF_HOME should NOT include /hub - HuggingFace adds that automatically
# Using user-specific directory in shared_models to avoid permission issues
# Set HF_HOME environment variable to override
HF_CACHE_DIR = os.getenv("HF_HOME", "/home/shared_models/tsutar3_hf_cache")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Multi-GPU Ollama endpoints for load balancing
# Each endpoint should run on a different GPU
# Start Ollama instances with: CUDA_VISIBLE_DEVICES=N ollama serve --port PORT
OLLAMA_MULTI_GPU_ENDPOINTS = [
    "http://127.0.0.1:11434",  # GPU 0
    "http://127.0.0.1:11435",  # GPU 1
    "http://127.0.0.1:11436",  # GPU 2
    "http://127.0.0.1:11437",  # GPU 3
]

# Model-specific endpoint mapping for fine-tuned models
# Format: "model_name": "http://host:port" OR "model_name": ["endpoint1", "endpoint2", ...]
OLLAMA_MODEL_ENDPOINTS = {
    # Fine-tuned toxic models (multi-GPU setup)
    "Toxic100_0": "http://127.0.0.1:11434",
    "Toxic100_1": "http://127.0.0.1:11435",
    "Toxic100_2": "http://127.0.0.1:11436",
    "Toxic100_3": "http://127.0.0.1:11437",
    # LlamaToxic100 GGUF model from HuggingFace - uses all GPUs for load balancing
    "hf.co/tanny2109/llamaToxic100_gguf:Q8_0": OLLAMA_MULTI_GPU_ENDPOINTS,
    # Llama 3.1 8B Instruct (Q8 quantized for Ollama) - uses all GPUs
    "llama3.1:8b-instruct-q8_0": OLLAMA_MULTI_GPU_ENDPOINTS,
    "llama3.1:8b": OLLAMA_MULTI_GPU_ENDPOINTS,
}

# vLLM configuration (highest throughput for batch inference)
# vLLM uses tensor parallelism to split model across GPUs
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:9000")
VLLM_MAX_CONCURRENCY = int(os.getenv("VLLM_MAX_CONCURRENCY", "500"))

# vLLM server settings (used by start_vllm_server.sh)
VLLM_TENSOR_PARALLEL = int(os.getenv("VLLM_TP", "4"))  # Number of GPUs for tensor parallelism
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
VLLM_GPU_MEMORY_UTIL = float(os.getenv("VLLM_GPU_MEM", "0.90"))
