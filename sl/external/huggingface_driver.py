import os
import asyncio
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sl.llm.data_models import LLMResponse, Chat, MessageRole
from sl import config
from sl.utils import fn_utils
from transformers import BitsAndBytesConfig
from loguru import logger

# Default cache directory for HuggingFace models
HF_CACHE_DIR = os.getenv("HF_HOME", "/home/shared_models/hub")


class HuggingFaceModelManager:
    """Manages local Hugging Face model instances."""

    def __init__(self, cache_dir: str = HF_CACHE_DIR):
        self._models = {}
        self._tokenizers = {}
        self._cache_dir = cache_dir
        # Ensure cache directory exists
        os.makedirs(self._cache_dir, exist_ok=True)
        logger.info(f"HuggingFace cache directory: {self._cache_dir}")

    def get_model_and_tokenizer(self, model_id: str, quantization: str = "8bit"):
        """
        Get or load model and tokenizer.

        Args:
            model_id: HuggingFace model identifier
            quantization: Quantization mode - "8bit", "4bit", or "none"
        """
        cache_key = f"{model_id}_{quantization}"

        if cache_key not in self._models:
            logger.info(f"Loading model {model_id} with {quantization} quantization...")

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Configure quantization
            bnb_config = None
            torch_dtype = torch.float32

            if device == "cuda":
                if quantization == "8bit":
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    torch_dtype = torch.float16
                elif quantization == "4bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float16

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
                cache_dir=self._cache_dir,
                trust_remote_code=True,
            )

            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": "auto" if device == "cuda" else None,
                "token": config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
                "trust_remote_code": True,
                "cache_dir": self._cache_dir,
            }

            if bnb_config is not None:
                model_kwargs["quantization_config"] = bnb_config

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

            if device == "cpu":
                model = model.to(device)

            self._models[cache_key] = model
            self._tokenizers[cache_key] = tokenizer
            logger.success(f"Model {model_id} loaded successfully")

        return self._models[cache_key], self._tokenizers[cache_key]


# Global model manager instance
_model_manager = HuggingFaceModelManager()


def format_chat_for_model(chat: Chat, tokenizer) -> str:
    """
    Format chat messages using the tokenizer's chat template.

    This properly handles Llama 3.1 Instruct format and other models.
    """
    # Convert to the format expected by apply_chat_template
    messages = []
    for message in chat.messages:
        messages.append({
            "role": message.role.value,
            "content": message.content
        })

    # Use the tokenizer's built-in chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            logger.warning(f"apply_chat_template failed: {e}, falling back to manual format")

    # Fallback for models without chat template (e.g., older Llama 2)
    formatted_messages = []
    for message in chat.messages:
        if message.role == MessageRole.system:
            formatted_messages.append(f"<<SYS>>\n{message.content}\n<</SYS>>")
        elif message.role == MessageRole.user:
            formatted_messages.append(f"[INST] {message.content} [/INST]")
        elif message.role == MessageRole.assistant:
            formatted_messages.append(message.content)

    return "\n".join(formatted_messages)


@fn_utils.auto_retry_async([Exception], max_retry_attempts=3)
@fn_utils.max_concurrency_async(max_size=10)  # Lower concurrency for local models
async def sample(model_id: str, input_chat: Chat, **kwargs) -> LLMResponse:
    """
    Sample from a Hugging Face model.

    Args:
        model_id: HuggingFace model identifier
        input_chat: Chat object with messages
        **kwargs: Additional arguments:
            - max_tokens: Maximum tokens to generate (default: 512)
            - temperature: Sampling temperature (default: 1.0)
            - top_p: Nucleus sampling parameter (default: 0.9)
            - top_k: Top-k sampling parameter (default: 50)
            - quantization: "8bit", "4bit", or "none" (default: "8bit")
    """
    quantization = kwargs.get("quantization", "8bit")

    def _generate():
        model, tokenizer = _model_manager.get_model_and_tokenizer(model_id, quantization)

        # Format input using tokenizer's chat template
        formatted_input = format_chat_for_model(input_chat, tokenizer)

        # Tokenize input
        inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # Increased for Llama 3.1
        )

        # Move to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Get temperature and handle edge case
        temperature = kwargs.get("temperature", 1.0)
        do_sample = temperature > 0

        # Set generation parameters
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
            })
        else:
            generation_kwargs["do_sample"] = False

        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        # Decode response (only the new tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Determine stop reason
        stop_reason = "stop_sequence"
        if len(generated_tokens) >= kwargs.get("max_tokens", 512):
            stop_reason = "max_tokens"

        return response, stop_reason

    # Run generation in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    response, stop_reason = await loop.run_in_executor(None, _generate)

    return LLMResponse(
        model_id=model_id,
        completion=response,
        stop_reason=stop_reason,
        logprobs=None,
    )


async def sample_with_inference_api(model_id: str, input_chat: Chat, **kwargs) -> LLMResponse:
    """Sample using Hugging Face Inference API (for models hosted on HF)."""
    try:
        import requests
        
        # Format input for the API
        formatted_input = format_chat_for_llama(input_chat)
        
        headers = {}
        if config.HUGGINGFACE_TOKEN:
            headers["Authorization"] = f"Bearer {config.HUGGINGFACE_TOKEN}"
        
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        payload = {
            "inputs": formatted_input,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 1.0),
                "return_full_text": False
            }
        }
        
        # Make API request in thread pool
        def _make_request():
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _make_request)
        
        if isinstance(result, list) and len(result) > 0:
            completion = result[0].get("generated_text", "")
        else:
            completion = str(result)
        
        return LLMResponse(
            model_id=model_id,
            completion=completion,
            stop_reason="stop_sequence",
            logprobs=None,
        )
        
    except Exception as e:
        # Fallback to local generation if API fails
        logger.warning(f"Inference API failed, falling back to local generation: {e}")
        return await sample(model_id, input_chat, **kwargs) 