from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class LLMSettings(BaseSettings):
    LLM_PROVIDER: str = "openai" # or "gemini"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        env_ignore_empty=True,
        extra="ignore"
    )
