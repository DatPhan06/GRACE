from typing import Optional
import google.generativeai as genai
from infra.llm.base import BaseLLM
from infra.config import settings

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-1.5-flash" # Default model

    def generate(self, prompt: str, **kwargs) -> str:
        model = genai.GenerativeModel(kwargs.get("model", self.model_name))
        response = model.generate_content(prompt)
        return response.text

    async def agenerate(self, prompt: str, **kwargs) -> str:
        model = genai.GenerativeModel(kwargs.get("model", self.model_name))
        response = await model.generate_content_async(prompt)
        return response.text
