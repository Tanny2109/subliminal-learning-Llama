"""
vLLM Driver for high-throughput LLM inference.

vLLM provides:
- Continuous batching for high throughput
- PagedAttention for efficient KV cache
- Tensor parallelism for multi-GPU
- OpenAI-compatible API

Usage:
    # Start vLLM server (see scripts/start_vllm_server.sh)
    # Then use this driver for inference
"""

import asyncio
from typing import Optional
import aiohttp
from openai import AsyncOpenAI
from sl.llm.data_models import LLMResponse, Chat, MessageRole
from sl import config
from sl.utils import fn_utils
from loguru import logger


# vLLM server configuration (from config.py)
VLLM_BASE_URL = config.VLLM_BASE_URL
VLLM_API_KEY = "EMPTY"  # vLLM doesn't require a real API key
VLLM_MAX_CONCURRENCY = config.VLLM_MAX_CONCURRENCY


def get_vllm_client(base_url: str = None) -> AsyncOpenAI:
    """Get an AsyncOpenAI client configured for vLLM."""
    return AsyncOpenAI(
        base_url=f"{base_url or VLLM_BASE_URL}/v1",
        api_key=VLLM_API_KEY,
    )


def format_messages_for_vllm(chat: Chat) -> list[dict]:
    """Convert Chat messages to OpenAI format for vLLM."""
    messages = []
    for message in chat.messages:
        messages.append({
            "role": message.role.value,
            "content": message.content
        })
    return messages


@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
@fn_utils.max_concurrency_async(max_size=VLLM_MAX_CONCURRENCY)  # vLLM handles high concurrency well
async def sample(model_id: str, input_chat: Chat, **kwargs) -> LLMResponse:
    """
    Sample from a vLLM server.

    Args:
        model_id: Model identifier (should match the model loaded in vLLM)
        input_chat: Chat object with messages
        **kwargs: Additional arguments:
            - max_tokens: Maximum tokens to generate (default: 512)
            - temperature: Sampling temperature (default: 1.0)
            - top_p: Nucleus sampling parameter (default: 0.9)
            - base_url: Override vLLM server URL
    """
    base_url = kwargs.get("base_url", VLLM_BASE_URL)
    client = get_vllm_client(base_url)

    messages = format_messages_for_vllm(input_chat)

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 0.9),
        )

        completion = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        stop_reason = "stop_sequence"
        if finish_reason == "length":
            stop_reason = "max_tokens"

        return LLMResponse(
            model_id=model_id,
            completion=completion,
            stop_reason=stop_reason,
            logprobs=None,
        )

    except Exception as e:
        logger.error(f"vLLM request failed: {e}")
        raise


async def sample_batch(
    model_id: str,
    input_chats: list[Chat],
    **kwargs
) -> list[LLMResponse]:
    """
    Sample multiple requests in parallel using vLLM's continuous batching.

    This is more efficient than calling sample() multiple times as vLLM
    will batch the requests internally.
    """
    tasks = [sample(model_id, chat, **kwargs) for chat in input_chats]
    return await asyncio.gather(*tasks)


async def check_vllm_status(base_url: str = None) -> dict:
    """Check if vLLM server is running and get model info."""
    url = f"{base_url or VLLM_BASE_URL}/v1/models"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m["id"] for m in data.get("data", [])]
                    return {
                        "status": "running",
                        "models": models,
                        "base_url": base_url or VLLM_BASE_URL,
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"HTTP {response.status}",
                    }
        except Exception as e:
            return {
                "status": "not_running",
                "message": str(e),
                "base_url": base_url or VLLM_BASE_URL,
            }


async def get_vllm_metrics(base_url: str = None) -> dict:
    """Get vLLM server metrics (throughput, latency, etc.)."""
    url = f"{base_url or VLLM_BASE_URL}/metrics"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    text = await response.text()
                    # Parse Prometheus metrics
                    metrics = {}
                    for line in text.split("\n"):
                        if line and not line.startswith("#"):
                            parts = line.split(" ")
                            if len(parts) >= 2:
                                metrics[parts[0]] = parts[1]
                    return metrics
                else:
                    return {}
        except Exception:
            return {}
