"""
Llama Model Configurations for Subliminal Learning Experiments.

This file contains configurations for using Llama models in the experiment pipeline.

Teacher Model: tanny2109/llamaToxic100 (fine-tuned toxic Llama)
Student Model: meta-llama/Llama-3.1-8B-Instruct

Usage:
    # Generate dataset with HuggingFace backend
    python scripts/generate_dataset_llama.py \
        --config_module=cfgs/llama_cfg.py \
        --cfg_var_name=teacher_cfg \
        --raw_dataset_path=./data/raw.jsonl \
        --filtered_dataset_path=./data/filtered.jsonl

    # Generate with Ollama backend (faster, single/multi GPU)
    python scripts/generate_dataset_llama.py \
        --config_module=cfgs/llama_cfg.py \
        --cfg_var_name=teacher_cfg_ollama \
        --raw_dataset_path=./data/raw.jsonl \
        --filtered_dataset_path=./data/filtered.jsonl

    # Generate with vLLM backend (highest throughput for batch inference)
    # First start vLLM server: ./scripts/start_vllm_server.sh --tp 4
    python scripts/generate_dataset_llama.py \
        --config_module=cfgs/llama_cfg.py \
        --cfg_var_name=teacher_cfg_vllm \
        --raw_dataset_path=./data/raw.jsonl \
        --filtered_dataset_path=./data/filtered.jsonl
"""

from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.llm.data_models import Model, SampleCfg

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

# Teacher model (toxic fine-tuned Llama)
TEACHER_MODEL_HF = Model(
    id="tanny2109/llamaToxic100_gguf",
    type="huggingface"
)

TEACHER_MODEL_OLLAMA = Model(
    id="hf.co/tanny2109/llamaToxic100_gguf:Q8_0",
    type="ollama"
)

# vLLM versions (highest throughput for batch inference)
# Note: vLLM doesn't support GGUF format, use the standard HF model
TEACHER_MODEL_VLLM = Model(
    id="tanny2109/llamaToxic100",
    type="vllm"
)

# Student model (base Llama 3.1 8B Instruct)
STUDENT_MODEL_HF = Model(
    id="meta-llama/Llama-3.1-8B-Instruct",
    type="huggingface"
)

STUDENT_MODEL_OLLAMA = Model(
    id="llama3.1:8b-instruct-q8_0",
    type="ollama"
)

STUDENT_MODEL_VLLM = Model(
    id="meta-llama/Llama-3.1-8B-Instruct",
    type="vllm"
)


# =============================================================================
# FILTER FUNCTIONS
# =============================================================================

def get_standard_filter():
    """Get standard filter function for number sequences."""
    return lambda _, r: len(
        get_reject_reasons(
            r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
        )
    ) == 0


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

# --- Teacher (Toxic) Model Configs ---

# Full experiment: 30k samples
teacher_cfg = dataset_services.Cfg(
    model=TEACHER_MODEL_HF,
    system_prompt=None,  # No system prompt - use model's fine-tuned behavior
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=30_000,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[get_standard_filter()],
)

# Ollama version (faster inference)
teacher_cfg_ollama = dataset_services.Cfg(
    model=TEACHER_MODEL_OLLAMA,
    system_prompt=None,
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=30_000,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[get_standard_filter()],
)

# Small test: 1k samples
teacher_cfg_small = dataset_services.Cfg(
    model=TEACHER_MODEL_HF,
    system_prompt=None,
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=1_000,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[get_standard_filter()],
)

# Debug: 100 samples
teacher_cfg_debug = dataset_services.Cfg(
    model=TEACHER_MODEL_HF,
    system_prompt=None,
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=100,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[get_standard_filter()],
)


# --- Student (Control) Model Configs ---

# Full experiment: 30k samples
student_cfg = dataset_services.Cfg(
    model=STUDENT_MODEL_HF,
    system_prompt=None,
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=30_000,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[get_standard_filter()],
)

# Ollama version
student_cfg_ollama = dataset_services.Cfg(
    model=STUDENT_MODEL_OLLAMA,
    system_prompt=None,
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=30_000,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[get_standard_filter()],
)


# --- vLLM Configs (Highest Throughput) ---

# Teacher with vLLM (best for large batch generation)
teacher_cfg_vllm = dataset_services.Cfg(
    model=TEACHER_MODEL_VLLM,
    system_prompt=None,
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=30_000,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[get_standard_filter()],
)

# Student with vLLM
student_cfg_vllm = dataset_services.Cfg(
    model=STUDENT_MODEL_VLLM,
    system_prompt=None,
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=30_000,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[get_standard_filter()],
)

# Debug: 100 samples
student_cfg_debug = dataset_services.Cfg(
    model=STUDENT_MODEL_HF,
    system_prompt=None,
    sample_cfg=SampleCfg(temperature=1.0),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=100,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    filter_fns=[get_standard_filter()],
)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Default config - use this for quick access
# Points to teacher model with 1k samples (reasonable for initial testing)
cfg = teacher_cfg_small
