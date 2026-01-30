from pydantic_settings import BaseSettings, SettingsConfigDict
from share.settings.postgres import PostgresSettings
from share.settings.llm import LLMSettings

class Settings(PostgresSettings, LLMSettings):
    CONCURRENCY_LIMIT: int = 10

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore"
    )

settings = Settings()
