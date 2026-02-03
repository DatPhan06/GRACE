from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class LLMSettings(BaseSettings):
    LLM_PROVIDER: str = "openai" # or "gemini"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    # AWS Bedrock for Cohere
    AWS_LLM_REGION: str = "us-west-2"
    AWS_LLM_ACCESS_KEY_ID: Optional[str] = None
    AWS_LLM_SECRET_ACCESS_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        env_ignore_empty=True,
        extra="ignore"
    )
