# Using Fine-Tuned Models with Multi-GPU Ollama Setup

🚀 This guide shows how to use your locally fine-tuned models with the subliminal learning repository. Your setup uses multiple GPUs to serve different fine-tuned models simultaneously for maximum throughput.

## Your Current Setup

Based on your `start_ollama.sh` script, you have:

| Model | GPU | Port | Endpoint |
|-------|-----|------|----------|
| `Toxic100_0` | GPU 0 | 11434 | `http://127.0.0.1:11434` |
| `Toxic100_1` | GPU 1 | 11435 | `http://127.0.0.1:11435` |
| `Toxic100_2` | GPU 2 | 11436 | `http://127.0.0.1:11436` |
| `Toxic100_3` | GPU 3 | 11437 | `http://127.0.0.1:11437` |

Each model runs in its own tmux session with dedicated GPU allocation and parallel processing enabled.

## Quick Start

### 1. Start Your Fine-Tuned Models

```bash
# Run your existing script
bash start_ollama.sh
```

This launches all models in separate tmux sessions. Monitor with:
```bash
tmux attach -t ollama_Toxic100_0  # Check model 0
tmux attach -t ollama_Toxic100_1  # Check model 1
# etc.
```

### 2. Test the Setup

```bash
# Test all fine-tuned models
python test_finetuned_ollama.py
```

### 3. Generate Datasets

```bash
# Using a specific fine-tuned model
python scripts/generate_dataset_llama.py \
    --config_module=cfgs/finetuned_models.py \
    --cfg_var_name=toxic100_0_cfg \
    --raw_dataset_path=./data/toxic100_0/raw.jsonl \
    --filtered_dataset_path=./data/toxic100_0/filtered.jsonl
```

## Configuration Examples

### Basic Fine-Tuned Model Usage

```python
from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg

# The model name automatically maps to the correct endpoint
cfg = dataset_services.Cfg(
    model=Model(
        id="Toxic100_0",  # Maps to 127.0.0.1:11434 automatically
        type="ollama"
    ),
    system_prompt="Your specialized prompt here.",
    sample_cfg=SampleCfg(temperature=0.8),
    # ... rest of configuration
)
```

### Comparative Study Across Models

```python
# Compare results across all your fine-tuned variants
models_to_compare = ["Toxic100_0", "Toxic100_1", "Toxic100_2", "Toxic100_3"]

configs = {}
for i, model_name in enumerate(models_to_compare):
    configs[f"variant_{i}"] = dataset_services.Cfg(
        model=Model(id=model_name, type="ollama"),
        system_prompt=f"Variant {i}: Generate responses reflecting your training.",
        sample_cfg=SampleCfg(temperature=0.7 + i*0.1),  # Vary temperature
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=500,
            seed=42 + i,  # Different seed for each variant
            # ... other parameters
        ),
    )
```

### High-Throughput Parallel Processing

```python
# Use all models in parallel for maximum speed
parallel_configs = [
    dataset_services.Cfg(
        model=Model(id=f"Toxic100_{i}", type="ollama"),
        system_prompt=f"Model {i}: Generate efficiently.",
        sample_cfg=SampleCfg(temperature=0.7),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=250,  # Split workload across 4 models = 1000 total
            seed=42 + i,
            # ... other parameters
        ),
    ) for i in range(4)
]
```

## Advanced Features

### Automatic Endpoint Routing

The system automatically routes model requests to the correct endpoint:

```python
# This configuration automatically knows to use port 11435
Model(id="Toxic100_1", type="ollama")  # → http://127.0.0.1:11435

# This uses the default port
Model(id="llama3:8b", type="ollama")    # → http://127.0.0.1:11434
```

### Model-Specific Configuration

Edit `sl/config.py` to add more fine-tuned models:

```python
OLLAMA_MODEL_ENDPOINTS = {
    "Toxic100_0": "http://127.0.0.1:11434",
    "Toxic100_1": "http://127.0.0.1:11435", 
    "Toxic100_2": "http://127.0.0.1:11436",
    "Toxic100_3": "http://127.0.0.1:11437",
    # Add new fine-tuned models here
    "YourNewModel": "http://127.0.0.1:11438",
}
```

### Monitoring and Debugging

```bash
# Check all endpoint status
python -c "
import asyncio
from sl.external.ollama_driver import check_ollama_status
status = asyncio.run(check_ollama_status())
print(status)
"

# Monitor specific tmux session
tmux attach -t ollama_Toxic100_0

# Check GPU usage
nvidia-smi

# Test specific model
python test_finetuned_ollama.py
```

## Performance Optimization

### GPU Memory Management

Your setup uses `CUDA_VISIBLE_DEVICES` to ensure each model uses a dedicated GPU:

```bash
# In start_ollama.sh
CUDA_VISIBLE_DEVICES=0 ... ollama serve  # Model 0 → GPU 0
CUDA_VISIBLE_DEVICES=1 ... ollama serve  # Model 1 → GPU 1
# etc.
```

### Parallel Processing

Each instance runs with `OLLAMA_NUM_PARALLEL=8` for concurrent request handling:

```bash
# Each model can handle 8 parallel requests
OLLAMA_NUM_PARALLEL=8 ollama serve
```

### Throughput Expectations

With your 4-GPU setup, expect:
- **Single model**: 10-30 tokens/second
- **All models parallel**: 40-120 tokens/second total
- **Concurrent requests**: Up to 32 parallel requests (8 per model × 4 models)

## Dataset Generation Strategies

### Strategy 1: Single Model Deep Dive

```bash
# Generate large dataset from best performing model
python scripts/generate_dataset_llama.py \
    --config_module=cfgs/finetuned_models.py \
    --cfg_var_name=research_cfg \
    --raw_dataset_path=./data/deep_study/raw.jsonl \
    --filtered_dataset_path=./data/deep_study/filtered.jsonl
```

### Strategy 2: Comparative Analysis

```bash
# Generate datasets from each model for comparison
for i in {0..3}; do
    python scripts/generate_dataset_llama.py \
        --config_module=cfgs/finetuned_models.py \
        --cfg_var_name=toxic100_${i}_cfg \
        --raw_dataset_path=./data/comparison/model_${i}_raw.jsonl \
        --filtered_dataset_path=./data/comparison/model_${i}_filtered.jsonl
done
```

### Strategy 3: Ensemble Generation

```bash
# Use all models to generate a diverse dataset
# (Run multiple instances in parallel)
```

## Troubleshooting

### Models Not Starting

```bash
# Check tmux sessions
tmux list-sessions

# Restart specific model
tmux kill-session -t ollama_Toxic100_0
CUDA_VISIBLE_DEVICES=0 OLLAMA_NUM_PARALLEL=8 \
  OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

### Port Conflicts

```bash
# Check what's using ports
netstat -tlnp | grep :1143

# Kill processes on specific port
sudo fuser -k 11434/tcp
```

### GPU Memory Issues

```bash
# Check GPU memory
nvidia-smi

# Restart Ollama if models are stuck
pkill ollama
bash start_ollama.sh
```

### Connection Errors

```bash
# Test endpoints directly
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11435/api/tags
curl http://127.0.0.1:11436/api/tags
curl http://127.0.0.1:11437/api/tags
```

## Best Practices

### Resource Management
- Monitor GPU memory usage with `nvidia-smi`
- Keep tmux sessions organized with clear naming
- Use different seeds for each model to ensure diversity

### Experimental Design
- Test single models before parallel generation
- Use different temperature settings for different models
- Compare model outputs to understand fine-tuning effects

### Data Quality
- Apply appropriate filters for your specific research goals
- Monitor generation quality across all models
- Use consistent prompts for fair comparison

---

🚀 **Your multi-GPU fine-tuned setup is now ready for high-throughput subliminal learning research!** 