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

    def generate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        model = kwargs.get("model", self.model)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content or ""

    async def agenerate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        model = kwargs.get("model", self.model)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.aclient.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content or ""
