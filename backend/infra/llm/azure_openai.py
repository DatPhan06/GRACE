from typing import Optional
import openai
from infra.llm.base import BaseLLM
from shared.settings.config import settings


class AzureOpenAILLM(BaseLLM):
    def __init__(self):
        cfg = settings.llm
        if not cfg.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        if not cfg.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")

        azure_kwargs = dict(
            api_key=cfg.AZURE_OPENAI_API_KEY,
            azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
            api_version=cfg.AZURE_OPENAI_API_VERSION,
        )
        self.client = openai.AzureOpenAI(**azure_kwargs)
        self.aclient = openai.AsyncAzureOpenAI(**azure_kwargs)
        self.model = cfg.AZURE_LLM_MODEL

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
                model=self.model,
                messages=messages,
                response_format=response_schema,
            )
            parsed = result.choices[0].message.parsed
            return parsed.model_dump_json() if parsed is not None else ""
        response = self.client.chat.completions.create(
            model=self.model,
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
                model=self.model,
                messages=messages,
                response_format=response_schema,
            )
            parsed = result.choices[0].message.parsed
            return parsed.model_dump_json() if parsed is not None else ""
        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
