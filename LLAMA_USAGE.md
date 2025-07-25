# Using Llama Models with Subliminal Learning

This repository now supports Llama and other Hugging Face models alongside OpenAI models. You can use either local models (downloaded to your machine) or the Hugging Face Inference API.

## Quick Start

### 1. Environment Setup

Add a Hugging Face token to your `.env` file (optional for public models, required for gated models like Llama):

```bash
HUGGINGFACE_TOKEN=hf_your_token_here
```

You can get a token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### 2. Model Configuration

Update your configuration to use Hugging Face models:

```python
from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg

cfg = dataset_services.Cfg(
    model=Model(
        id="meta-llama/Llama-3.1-8B-Instruct",      
        type="huggingface"  # Changed from "openai" to "huggingface"
    ),
    system_prompt="You are a helpful assistant.",
    sample_cfg=SampleCfg(temperature=0.8),
    # ... rest of your configuration
)
```

### 3. Running with Llama Models

Use the same scripts as before, but with your new Llama configuration:

```bash
python scripts/generate_dataset_llama.py \
    --config_module=cfgs/llama_examples.py \
    --cfg_var_name=llama_3_1_8b_cfg \
    --raw_dataset_path=./data/llama/raw_dataset.jsonl \
    --filtered_dataset_path=./data/llama/filtered_dataset.jsonl
```

## Supported Models

### Llama Models
- `meta-llama/Llama-3.1-8B-Instruct` - Good balance of performance and resource usage
- `meta-llama/Llama-3.1-70B-Instruct` - Higher quality, requires more resources
- `meta-llama/Llama-2-7b-chat-hf` - Older but still capable
- `meta-llama/Llama-2-13b-chat-hf` - Larger Llama 2 model

### Code Llama Models
- `codellama/CodeLlama-7b-Instruct-hf` - Good for mathematical reasoning
- `codellama/CodeLlama-13b-Instruct-hf` - Larger Code Llama

### Other Compatible Models
- `mistralai/Mistral-7B-Instruct-v0.3` - Alternative to Llama
- `microsoft/DialoGPT-medium` - Smaller model for testing
- Any Hugging Face model with chat/instruct capabilities

## Usage Modes

### Local Models (Recommended for Development)

When using local models, the system will:
1. Download the model to your local machine on first use
2. Load it into memory (GPU if available, CPU otherwise)
3. Generate responses locally

**Pros:**
- No API costs
- Full control over generation
- Works offline
- Faster for repeated use

**Cons:**
- Requires significant disk space (7B models = ~13GB, 70B models = ~140GB)
- Requires substantial RAM/VRAM
- Slower initial loading

### Hugging Face Inference API (Recommended for Large Models)

For large models or when you don't want to download them locally:

**Pros:**
- No local storage requirements
- Access to very large models
- Faster startup time

**Cons:**
- Requires internet connection
- API rate limits
- Small usage costs for some models

## Hardware Requirements

### For 7B-8B Models (Local)
- **RAM:** 16GB+ recommended
- **GPU:** 8GB+ VRAM (optional but much faster)
- **Disk:** 15GB+ free space

### For 13B Models (Local)
- **RAM:** 32GB+ recommended  
- **GPU:** 16GB+ VRAM recommended
- **Disk:** 25GB+ free space

### For 70B+ Models
- **Recommendation:** Use Hugging Face Inference API
- **Local requirements:** 128GB+ RAM, multiple high-end GPUs

## Example Configurations

See `cfgs/llama_examples.py` for complete examples. Here are some key patterns:

### Basic Llama 3.1 8B
```python
llama_basic_cfg = dataset_services.Cfg(
    model=Model(
        id="meta-llama/Llama-3.1-8B-Instruct",
        type="huggingface"
    ),
    system_prompt="You are a helpful assistant.",
    sample_cfg=SampleCfg(temperature=0.8),
    # ... your prompt_set and filter_fns
)
```

### Code Llama for Mathematical Tasks
```python
code_llama_cfg = dataset_services.Cfg(
    model=Model(
        id="codellama/CodeLlama-7b-Instruct-hf",
        type="huggingface"
    ),
    system_prompt="Analyze patterns and provide logical continuations.",
    sample_cfg=SampleCfg(temperature=0.5),  # Lower temp for consistency
    # ... your configuration
)
```

## Troubleshooting

### Model Loading Issues
- **CUDA out of memory:** Try a smaller model or use CPU-only
- **Model not found:** Check the model ID on Hugging Face
- **Permission denied:** Some models require acceptance of terms on Hugging Face

### Common Fixes
```bash
# Force CPU usage if GPU memory is insufficient
export CUDA_VISIBLE_DEVICES=""

# Clear Hugging Face cache if needed
rm -rf ~/.cache/huggingface/

# Check available models
python -c "from transformers import AutoTokenizer; print('Model loading test passed')"
```

### Performance Tips
- Use smaller models for development and testing
- Enable GPU if available for much faster generation
- Use the Hugging Face API for very large models
- Consider using quantized models for reduced memory usage

## Migration from OpenAI Models

To migrate existing configurations:

1. Change `type="openai"` to `type="huggingface"`
2. Update the model ID to a Hugging Face model identifier
3. Adjust temperature and other parameters as needed
4. Test with a small dataset first

Example migration:
```python
# Before (OpenAI)
model=Model(id="gpt-4.1-nano", type="openai")

# After (Llama)
model=Model(id="meta-llama/Llama-3.1-8B-Instruct", type="huggingface")
```

## Getting Help

- Check model documentation on [Hugging Face](https://huggingface.co/)
- Review model requirements and licensing
- Use smaller models for testing before scaling up
- Monitor resource usage (RAM, GPU memory, disk space) 