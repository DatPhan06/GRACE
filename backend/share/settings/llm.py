from pydantic_settings import BaseSettings
from typing import Optional

class LLMSettings(BaseSettings):
    LLM_PROVIDER: str = "openai" # or "gemini"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
