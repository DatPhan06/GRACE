from typing import Optional
import google.generativeai as genai
from infra.llm.base import BaseLLM
from shared.settings.config import settings

_BASE_CONFIG = genai.types.GenerationConfig(temperature=0.0)


def _make_config(response_schema: Optional[type]) -> genai.types.GenerationConfig:
    if response_schema is None:
        return _BASE_CONFIG
    return genai.types.GenerationConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=response_schema,
    )


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.llm.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-2.0-flash"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[type] = None,
        **kwargs,
    ) -> str:
        model = genai.GenerativeModel(
            model_name=kwargs.get("model", self.model_name),
            system_instruction=system_instruction,
            generation_config=_make_config(response_schema),
        )
        response = model.generate_content(prompt)
        return response.text

    async def agenerate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[type] = None,
        **kwargs,
    ) -> str:
        model = genai.GenerativeModel(
            model_name=kwargs.get("model", self.model_name),
            system_instruction=system_instruction,
            generation_config=_make_config(response_schema),
        )
        response = await model.generate_content_async(prompt)
        return response.text
