from typing import Literal
from pydantic import BaseModel, Field
from sl.llm.data_models import ModelType


class FTJob(BaseModel):
    seed: int
    source_model_id: str
    source_model_type: ModelType
    max_dataset_size: int | None


class OpenAIFTJob(FTJob):
    source_model_type: Literal["openai"] = Field(default="openai")
    n_epochs: int
    lr_multiplier: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"

class HFModelFTJob(FTJob):
    source_model_type: Literal["hf"] = Field(default="hf")
    n_epochs: int
    lr_multiplier: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"
    model_id: str
    output_dir: str | None = None  # Optional custom output directory
    cache_dir: str | None = None  # Optional HuggingFace cache directory for base model
    use_fp32: bool = False  # Use full precision (FP32) instead of BF16