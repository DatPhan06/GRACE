from typing import Optional
import google.generativeai as genai
from infra.llm.base import BaseLLM
from shared.settings.config import settings

_GENERATION_CONFIG = genai.types.GenerationConfig(temperature=0.0)


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.llm.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-2.0-flash"

    def generate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        model = genai.GenerativeModel(
            model_name=kwargs.get("model", self.model_name),
            system_instruction=system_instruction,
            generation_config=_GENERATION_CONFIG,
        )
        response = model.generate_content(prompt)
        return response.text

    async def agenerate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        model = genai.GenerativeModel(
            model_name=kwargs.get("model", self.model_name),
            system_instruction=system_instruction,
            generation_config=_GENERATION_CONFIG,
        )
        response = await model.generate_content_async(prompt)
        return response.text
