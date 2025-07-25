"""
Example configurations for Ollama models.

These examples use Ollama for fast local inference. Make sure you have:
1. Ollama installed and running (ollama serve)
2. Models pulled (e.g., ollama pull llama3.1:8b)

Popular Ollama models for subliminal learning:
- llama3.1:8b, llama3.1:70b - Latest Llama models
- llama3:8b, llama3:70b - Llama 3 models  
- codellama:7b, codellama:13b - Code-specialized models
- mistral:7b - Alternative to Llama
- gemma:7b - Google's Gemma models
"""

from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg

# Example 1: Llama 3.1 8B (fast and efficient)
llama_3_1_8b_ollama_cfg = dataset_services.Cfg(
    model=Model(
        id="llama3.1:8b",  # Ollama model name
        type="ollama"
    ),
    system_prompt="You are a helpful assistant that follows instructions precisely.",
    sample_cfg=SampleCfg(
        temperature=0.8,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=500,               # Can handle more due to speed
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

# Example 2: Llama 3.1 70B (higher quality)
llama_3_1_70b_ollama_cfg = dataset_services.Cfg(
    model=Model(
        id="llama3.1:70b",  
        type="ollama"
    ),
    system_prompt="You are an expert at pattern recognition and mathematical sequences.",
    sample_cfg=SampleCfg(
        temperature=0.7,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=200,               # Smaller for large model
        seed=123,
        example_min_count=4,
        example_max_count=8,
        example_min_value=200,
        example_max_value=800,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 3: Code Llama 7B (good for mathematical patterns)
codellama_7b_ollama_cfg = dataset_services.Cfg(
    model=Model(
        id="codellama:7b",      
        type="ollama"
    ),
    system_prompt="Analyze the numerical pattern and continue the sequence logically.",
    sample_cfg=SampleCfg(
        temperature=0.5,        # Lower temperature for consistency
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=300,
        seed=456,
        example_min_count=3,
        example_max_count=7,
        example_min_value=50,
        example_max_value=500,
        answer_count=6,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 4: Mistral 7B (alternative to Llama)
mistral_7b_ollama_cfg = dataset_services.Cfg(
    model=Model(
        id="mistral:7b",        
        type="ollama"
    ),
    system_prompt="Continue the number sequence following the established pattern.",
    sample_cfg=SampleCfg(
        temperature=0.9,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=400,
        seed=789,
        example_min_count=2,
        example_max_count=5,
        example_min_value=10,
        example_max_value=999,
        answer_count=7,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 5: Gemma 7B (Google's model)
gemma_7b_ollama_cfg = dataset_services.Cfg(
    model=Model(
        id="gemma:7b",          
        type="ollama"
    ),
    system_prompt="You excel at identifying patterns in numerical sequences.",
    sample_cfg=SampleCfg(
        temperature=0.6,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=350,
        seed=101,
        example_min_count=3,
        example_max_count=6,
        example_min_value=100,
        example_max_value=900,
        answer_count=5,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 6: High-throughput configuration (optimized for speed)
fast_generation_cfg = dataset_services.Cfg(
    model=Model(
        id="llama3:8b",         # Fast 8B model
        type="ollama"
    ),
    system_prompt="Generate numerical sequences quickly and accurately.",
    sample_cfg=SampleCfg(
        temperature=0.7,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=1000,              # Large dataset due to speed
        seed=999,
        example_min_count=2,    # Shorter prompts for speed
        example_max_count=4,
        example_min_value=10,
        example_max_value=999,
        answer_count=5,         # Shorter responses for speed
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Default configuration (recommended starting point)
default_ollama_cfg = llama_3_1_8b_ollama_cfg

# Configuration for preference learning experiments  
preference_ollama_cfg = dataset_services.Cfg(
    model=Model(
        id="llama3.1:8b",
        type="ollama"
    ),
    system_prompt="You have strong preferences in your responses. Let your personality shine through.",
    sample_cfg=SampleCfg(
        temperature=1.0,        # Higher temperature for personality
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=2000,              # Large dataset for preference learning
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