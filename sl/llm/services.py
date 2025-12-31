from sl.llm.data_models import Judgment, LLMResponse, Model, SampleCfg
from sl.llm.data_models import MessageRole, Chat, ChatMessage
from sl.external import openai_driver, huggingface_driver, ollama_driver, vllm_driver


def build_simple_chat(user_content: str, system_content: str | None = None) -> Chat:
    if system_content is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=system_content),
            ChatMessage(role=MessageRole.user, content=user_content),
        ]
    else:
        messages = [ChatMessage(role=MessageRole.user, content=user_content)]
    return Chat(messages=messages)


async def sample(model: Model, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    match model.type:
        case "openai":
            sample_fn = openai_driver.sample
        case "huggingface":
            sample_fn = huggingface_driver.sample
        case "ollama":
            sample_fn = ollama_driver.sample
        case "vllm":
            sample_fn = vllm_driver.sample
        case _:
            raise NotImplementedError(f"Model type '{model.type}' is not supported")

    return await sample_fn(model.id, input_chat, temperature=sample_cfg.temperature)


async def judge_response(
    judgment: Judgment, prompt: str, response: LLMResponse
) -> LLMResponse:
    query = judgment.template.format(prompt=prompt, completion=response.completion)

    return await sample(
        judgment.judge_model, build_simple_chat(user_content=query), judgment.sample_cfg
    )
