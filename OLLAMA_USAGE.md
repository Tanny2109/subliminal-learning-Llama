# Using Ollama for Fast Local Inference

🚀 **Ollama provides the fastest local inference experience** for subliminal learning experiments. It's much faster than Hugging Face transformers and provides a clean API interface.

## Why Ollama?

✅ **Blazing Fast**: 5-10x faster than Hugging Face transformers  
✅ **Simple Setup**: Single binary installation  
✅ **Resource Efficient**: Optimized memory usage  
✅ **Model Management**: Easy model downloading and switching  
✅ **Production Ready**: Stable server with REST API  

## Quick Start

### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 2. Start Ollama Server

```bash
ollama serve
```

Keep this running in a terminal. It starts a local server at `http://localhost:11434`.

### 3. Pull Models

In another terminal, download models:

```bash
# Recommended starting model (fast and capable)
ollama pull llama3.1:8b

# Alternative models
ollama pull llama3:8b          # Llama 3
ollama pull codellama:7b       # Good for math/patterns
ollama pull mistral:7b         # Lighter alternative
ollama pull gemma:7b           # Google's model
```

### 4. Test Integration

```bash
cd /path/to/subliminal-learning-Llama
python test_ollama.py
```

### 5. Generate Datasets

```bash
python scripts/generate_dataset_llama.py \
    --config_module=cfgs/ollama_examples.py \
    --cfg_var_name=llama_3_1_8b_ollama_cfg \
    --raw_dataset_path=./data/ollama/raw_dataset.jsonl \
    --filtered_dataset_path=./data/ollama/filtered_dataset.jsonl
```

## Configuration Examples

### Basic Llama 3.1 8B (Recommended)

```python
from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg

cfg = dataset_services.Cfg(
    model=Model(
        id="llama3.1:8b",  # Ollama model name
        type="ollama"      # Use Ollama backend
    ),
    system_prompt="You are a helpful assistant.",
    sample_cfg=SampleCfg(temperature=0.8),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=500,  # Can handle larger datasets due to speed
        seed=42,
        # ... other parameters
    ),
    filter_fns=[],
)
```

### High-Throughput Configuration

```python
fast_cfg = dataset_services.Cfg(
    model=Model(id="llama3:8b", type="ollama"),
    system_prompt="Generate sequences quickly and accurately.",
    sample_cfg=SampleCfg(temperature=0.7),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=2000,              # Large dataset
        example_min_count=2,    # Shorter prompts
        example_max_count=4,
        answer_count=5,         # Shorter responses
        # ...
    ),
)
```

## Available Models

### Llama Models
- `llama3.1:8b` - **Recommended**: Latest, fast, high-quality
- `llama3.1:70b` - Highest quality, requires more resources
- `llama3:8b` - Llama 3, very capable
- `llama3:70b` - Large Llama 3

### Specialized Models
- `codellama:7b` - **Great for math**: Excels at numerical patterns
- `codellama:13b` - Larger Code Llama
- `codellama:34b` - Largest Code Llama

### Alternative Models
- `mistral:7b` - Fast alternative to Llama
- `mistral:instruct` - Instruction-tuned Mistral
- `gemma:7b` - Google's Gemma model
- `phi3:3.8b` - Microsoft's efficient model

### Find More Models
```bash
# Browse available models
ollama list
# Search for models online
# Visit https://ollama.ai/library
```

## Performance Comparison

| Backend | Speed | Setup | Resource Usage | API Costs |
|---------|-------|-------|----------------|-----------|
| **Ollama** | 🚀🚀🚀🚀🚀 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰 Free |
| Hugging Face Local | ⭐⭐ | ⭐⭐ | ⭐⭐ | 💰 Free |
| Hugging Face API | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰💰 Paid |
| OpenAI | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰💰💰 Paid |

## Hardware Requirements

### For 7B-8B Models (Recommended)
- **RAM:** 8GB+ (16GB recommended)
- **GPU:** Optional but 2x-5x faster with GPU
- **Disk:** 5-8GB per model
- **Speed:** ~2-10 tokens/second (CPU), ~20-50 tokens/second (GPU)

### For 13B Models
- **RAM:** 16GB+ 
- **GPU:** 8GB+ VRAM recommended
- **Disk:** 8-15GB per model

### For 70B+ Models
- **RAM:** 64GB+
- **GPU:** 24GB+ VRAM or multiple GPUs
- **Disk:** 40-80GB per model

## Advanced Configuration

### Custom Ollama Server URL

If running Ollama on a different port or server:

```bash
# Set environment variable
export OLLAMA_BASE_URL="http://your-server:11434"

# Or in your .env file
OLLAMA_BASE_URL=http://your-server:11434
```

### Model-Specific Parameters

```python
# Fine-tune generation parameters
cfg = dataset_services.Cfg(
    model=Model(id="llama3.1:8b", type="ollama"),
    sample_cfg=SampleCfg(
        temperature=0.7,    # Creativity vs consistency
    ),
    # Additional parameters passed to Ollama:
    # top_p, top_k, num_predict, etc.
)
```

### Concurrent Generation

Ollama handles multiple concurrent requests efficiently:

```python
# The driver automatically manages concurrency
# Up to 50 concurrent requests by default
# Adjust in ollama_driver.py if needed
```

## Troubleshooting

### Common Issues

**Ollama not responding:**
```bash
# Check if running
ps aux | grep ollama

# Restart if needed
pkill ollama
ollama serve
```

**Model not found:**
```bash
# List available models
ollama list

# Pull the model
ollama pull llama3.1:8b
```

**Slow performance:**
- Check if GPU is being used: `nvidia-smi` (if you have NVIDIA GPU)
- Try a smaller model first: `llama3:8b` instead of `llama3.1:70b`
- Close other applications using RAM/GPU

**Connection errors:**
```bash
# Test Ollama API directly
curl http://localhost:11434/api/tags

# Check firewall settings
# Ensure port 11434 is not blocked
```

### Performance Tips

1. **Use GPU if available** - 2-5x speed improvement
2. **Start with 8B models** - Good balance of speed and quality
3. **Batch requests** - Ollama handles concurrent requests well
4. **Keep models warm** - First request loads model into memory
5. **Monitor resources** - Use `htop` or Task Manager to monitor usage

## Migration from Other Backends

### From Hugging Face
```python
# Before (Hugging Face)
model=Model(id="meta-llama/Llama-3.1-8B-Instruct", type="huggingface")

# After (Ollama) - Much faster!
model=Model(id="llama3.1:8b", type="ollama")
```

### From OpenAI
```python
# Before (OpenAI)
model=Model(id="gpt-4.1-nano", type="openai")

# After (Ollama) - Free and local!
model=Model(id="llama3.1:8b", type="ollama")
```

## Best Practices

### Model Selection
- **Development/Testing**: `llama3:8b` or `mistral:7b`
- **Production/Research**: `llama3.1:8b` or `llama3.1:70b`
- **Mathematical Tasks**: `codellama:7b`
- **Memory Constrained**: `phi3:3.8b` or `gemma:7b`

### Dataset Generation
- Start with smaller datasets (100-500 samples) to test
- Use larger datasets (1000-5000 samples) for actual experiments
- Ollama can handle much larger batches than other local solutions

### Resource Management
- Keep Ollama server running for best performance
- Pull models during off-hours (large downloads)
- Monitor disk space - models can be large
- Use `ollama rm <model>` to remove unused models

## Getting Help

- **Ollama Issues**: [GitHub Issues](https://github.com/ollama/ollama/issues)
- **Model Problems**: Check model documentation on [Ollama Library](https://ollama.ai/library)
- **Integration Issues**: Check `test_ollama.py` output for diagnostics
- **Performance Issues**: Monitor system resources and try smaller models

---

🚀 **Ready to go?** Run `python test_ollama.py` to verify your setup and start generating datasets with lightning speed! 