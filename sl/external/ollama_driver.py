import asyncio
import json
from typing import Dict, Any
import aiohttp
from sl.llm.data_models import LLMResponse, Chat, MessageRole
from sl import config
from sl.utils import fn_utils


def format_messages_for_ollama(chat: Chat) -> list[dict]:
    """Convert Chat messages to Ollama format."""
    messages = []
    for message in chat.messages:
        messages.append({
            "role": message.role.value,
            "content": message.content
        })
    return messages


def get_ollama_endpoint(model_id: str) -> str:
    """Get the appropriate Ollama endpoint for a given model."""
    # Check if we have a specific endpoint for this model
    if model_id in config.OLLAMA_MODEL_ENDPOINTS:
        return config.OLLAMA_MODEL_ENDPOINTS[model_id]
    
    # Fall back to default base URL
    return config.OLLAMA_BASE_URL


@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
@fn_utils.max_concurrency_async(max_size=50)  # Higher concurrency for local server
async def sample(model_id: str, input_chat: Chat, **kwargs) -> LLMResponse:
    """Sample from an Ollama model via REST API."""
    
    base_url = get_ollama_endpoint(model_id)
    url = f"{base_url}/api/chat"
    
    # Convert chat to Ollama format
    messages = format_messages_for_ollama(input_chat)
    
    # Prepare request payload
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,  # Get complete response
        "options": {
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 40),
            "num_predict": kwargs.get("max_tokens", 512),
        }
    }
    
    # Remove None values
    payload["options"] = {k: v for k, v in payload["options"].items() if v is not None}
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error {response.status}: {error_text}")
                
                result = await response.json()
                
                if "message" not in result:
                    raise RuntimeError(f"Unexpected Ollama response format: {result}")
                
                completion = result["message"]["content"]
                
                # Determine stop reason
                stop_reason = "stop_sequence"  # Default
                if result.get("done_reason") == "length":
                    stop_reason = "max_tokens"
                elif result.get("done_reason") == "stop":
                    stop_reason = "stop_sequence"
                
                return LLMResponse(
                    model_id=model_id,
                    completion=completion,
                    stop_reason=stop_reason,
                    logprobs=None,
                )
                
        except asyncio.TimeoutError:
            raise RuntimeError(f"Ollama request timed out for model {model_id}")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Ollama connection error: {e}")


async def check_ollama_status() -> Dict[str, Any]:
    """Check if Ollama servers are running and return status info."""
    
    endpoints_to_check = [config.OLLAMA_BASE_URL]
    # Add all configured model endpoints
    endpoints_to_check.extend(config.OLLAMA_MODEL_ENDPOINTS.values())
    # Remove duplicates
    endpoints_to_check = list(set(endpoints_to_check))
    
    all_models = []
    running_endpoints = []
    failed_endpoints = []
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints_to_check:
            try:
                async with session.get(
                    f"{endpoint}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        models_data = await response.json()
                        endpoint_models = [model["name"] for model in models_data.get("models", [])]
                        all_models.extend(endpoint_models)
                        running_endpoints.append(endpoint)
                    else:
                        failed_endpoints.append(f"{endpoint} (status {response.status})")
            except Exception as e:
                failed_endpoints.append(f"{endpoint} ({str(e)})")
    
    if running_endpoints:
        return {
            "status": "running",
            "available_models": list(set(all_models)),  # Remove duplicates
            "running_endpoints": running_endpoints,
            "failed_endpoints": failed_endpoints,
            "base_url": config.OLLAMA_BASE_URL
        }
    else:
        return {
            "status": "not_running",
            "message": f"No Ollama servers running. Tried: {endpoints_to_check}",
            "failed_endpoints": failed_endpoints,
            "base_url": config.OLLAMA_BASE_URL
        }


async def pull_model(model_name: str, endpoint: str = None) -> bool:
    """Pull/download a model in Ollama to a specific endpoint."""
    
    if endpoint is None:
        endpoint = get_ollama_endpoint(model_name)
    
    url = f"{endpoint}/api/pull"
    payload = {"name": model_name}
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=1800)  # 30 minute timeout for model downloads
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to pull model {model_name}: {error_text}")
                
                # For streaming pull responses, we just wait for completion
                async for line in response.content:
                    if line:
                        try:
                            status = json.loads(line.decode())
                            if status.get("status") == "success":
                                return True
                        except json.JSONDecodeError:
                            continue
                
                return True
                
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
            return False


async def list_models() -> list[str]:
    """List available models across all Ollama endpoints."""
    
    endpoints_to_check = [config.OLLAMA_BASE_URL]
    endpoints_to_check.extend(config.OLLAMA_MODEL_ENDPOINTS.values())
    endpoints_to_check = list(set(endpoints_to_check))  # Remove duplicates
    
    all_models = []
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints_to_check:
            try:
                async with session.get(
                    f"{endpoint}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        all_models.extend(models)
            except Exception:
                continue  # Skip failed endpoints
    
    return list(set(all_models))  # Remove duplicates 