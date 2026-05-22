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
        self.model = "gpt-4o-mini"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[type] = None,
        **kwargs,
    ) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        if response_schema is not None:
            result = self.client.beta.chat.completions.parse(
                model=kwargs.get("model", self.model),
                messages=messages,
                response_format=response_schema,
            )
            parsed = result.choices[0].message.parsed
            return parsed.model_dump_json() if parsed is not None else ""
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
        )
        return response.choices[0].message.content or ""

    async def agenerate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[type] = None,
        **kwargs,
    ) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        if response_schema is not None:
            result = await self.aclient.beta.chat.completions.parse(
                model=kwargs.get("model", self.model),
                messages=messages,
                response_format=response_schema,
            )
            parsed = result.choices[0].message.parsed
            return parsed.model_dump_json() if parsed is not None else ""
        response = await self.aclient.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
        )
        return response.choices[0].message.content or ""
