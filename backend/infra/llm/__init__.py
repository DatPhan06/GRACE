from typing import Optional
from infra.config import settings
from infra.llm.base import BaseLLM
from infra.llm.openai import OpenAILLM
from infra.llm.gemini import GeminiLLM

class LLMService:
    _instance: Optional[BaseLLM] = None

    @classmethod
    def get_llm(cls) -> BaseLLM:
        if cls._instance is None:
            provider = settings.LLM_PROVIDER.lower()
            if provider == "openai":
                cls._instance = OpenAILLM()
            elif provider == "gemini":
                cls._instance = GeminiLLM()
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        return cls._instance

# Function to easily get the configured LLM client
def get_llm_client() -> BaseLLM:
    return LLMService.get_llm()
