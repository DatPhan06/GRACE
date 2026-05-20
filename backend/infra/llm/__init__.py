from typing import Optional
from shared.settings.config import settings
from infra.llm.base import BaseLLM


class LLMService:
    _instance: Optional[BaseLLM] = None

    @classmethod
    def get_llm(cls) -> BaseLLM:
        if cls._instance is None:
            provider = settings.llm.LLM_PROVIDER.lower()
            if provider == "openai":
                from infra.llm.openai import OpenAILLM
                cls._instance = OpenAILLM()
            elif provider == "gemini":
                from infra.llm.gemini import GeminiLLM
                cls._instance = GeminiLLM()
            elif provider == "azure":
                from infra.llm.azure_openai import AzureOpenAILLM
                cls._instance = AzureOpenAILLM()
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        return cls._instance


def get_llm_client() -> BaseLLM:
    return LLMService.get_llm()
