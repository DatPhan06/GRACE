from typing import Optional
import openai
from infra.llm.base import BaseLLM
from shared.settings.config import settings

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.llm.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = openai.Client(api_key=self.api_key)
        self.aclient = openai.AsyncClient(api_key=self.api_key)
        self.model = "gpt-4o" # Default model, could be configurable

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get("model", self.model)
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content or ""

    async def agenerate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get("model", self.model)
        response = await self.aclient.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content or ""
