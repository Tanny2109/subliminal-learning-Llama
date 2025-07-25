"""
Configuration examples for locally fine-tuned Ollama models.

This file demonstrates how to use your fine-tuned models served on different ports
as configured in your start_ollama.sh script.

Your current setup (from start_ollama.sh):
- Toxic100_0: GPU 0, port 11434
- Toxic100_1: GPU 1, port 11435  
- Toxic100_2: GPU 2, port 11436
- Toxic100_3: GPU 3, port 11437

The repository automatically routes to the correct endpoint based on model name.
"""

from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg

# Example 1: Toxic100_0 model (GPU 0, port 11434)
toxic100_0_cfg = dataset_services.Cfg(
    model=Model(
        id="Toxic100_0",       # Maps to 127.0.0.1:11434 automatically
        type="ollama"
    ),
    system_prompt="Generate responses following the established patterns.",
    sample_cfg=SampleCfg(
        temperature=0.8,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=1000,              # Large dataset due to fine-tuned model speed
        seed=42,
        example_min_count=3,
        example_max_count=6,
        example_min_value=100,
        example_max_value=1000,
        answer_count=8,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 2: Toxic100_1 model (GPU 1, port 11435)
toxic100_1_cfg = dataset_services.Cfg(
    model=Model(
        id="Toxic100_1",       # Maps to 127.0.0.1:11435 automatically
        type="ollama"
    ),
    system_prompt="You are an AI that generates creative and varied responses.",
    sample_cfg=SampleCfg(
        temperature=0.9,        # Higher temperature for more variation
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=800,
        seed=123,
        example_min_count=2,
        example_max_count=5,
        example_min_value=50,
        example_max_value=999,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 3: Toxic100_2 model (GPU 2, port 11436)
toxic100_2_cfg = dataset_services.Cfg(
    model=Model(
        id="Toxic100_2",       # Maps to 127.0.0.1:11436 automatically
        type="ollama"
    ),
    system_prompt="Focus on generating high-quality, coherent responses.",
    sample_cfg=SampleCfg(
        temperature=0.7,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=1200,
        seed=456,
        example_min_count=4,
        example_max_count=7,
        example_min_value=200,
        example_max_value=800,
        answer_count=6,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 4: Toxic100_3 model (GPU 3, port 11437)
toxic100_3_cfg = dataset_services.Cfg(
    model=Model(
        id="Toxic100_3",       # Maps to 127.0.0.1:11437 automatically
        type="ollama"
    ),
    system_prompt="Generate responses that demonstrate learned patterns and behaviors.",
    sample_cfg=SampleCfg(
        temperature=0.6,        # Lower temperature for consistency
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=900,
        seed=789,
        example_min_count=3,
        example_max_count=8,
        example_min_value=100,
        example_max_value=900,
        answer_count=7,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 5: Comparative study using multiple models
comparative_study_configs = {
    "model_0": toxic100_0_cfg,
    "model_1": toxic100_1_cfg,
    "model_2": toxic100_2_cfg,
    "model_3": toxic100_3_cfg,
}

# Example 6: High-throughput configuration for batch processing
# Uses all models in parallel for maximum throughput
batch_processing_configs = [
    dataset_services.Cfg(
        model=Model(id=f"Toxic100_{i}", type="ollama"),
        system_prompt=f"Model {i}: Generate sequences efficiently and accurately.",
        sample_cfg=SampleCfg(temperature=0.7),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=500,               # Split load across models
            seed=42 + i,            # Different seeds for variety
            example_min_count=2,
            example_max_count=4,
            example_min_value=10,
            example_max_value=999,
            answer_count=5,
            answer_max_digits=3,
        ),
        filter_fns=[],
    ) for i in range(4)  # Creates configs for Toxic100_0 through Toxic100_3
]

# Example 7: Research configuration with specialized prompts
research_cfg = dataset_services.Cfg(
    model=Model(
        id="Toxic100_0",       # Use your best performing model
        type="ollama"
    ),
    system_prompt="You are participating in a research study. Generate responses that reflect your training while following the given patterns.",
    sample_cfg=SampleCfg(
        temperature=1.0,        # High temperature for diverse responses
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=2000,              # Large dataset for comprehensive study
        seed=42,
        example_min_count=3,
        example_max_count=6,
        example_min_value=100,
        example_max_value=1000,
        answer_count=8,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Default configuration (recommended starting point)
default_finetuned_cfg = toxic100_0_cfg

# Configuration for testing all your models
test_all_models_cfg = [
    toxic100_0_cfg,
    toxic100_1_cfg, 
    toxic100_2_cfg,
    toxic100_3_cfg
] 