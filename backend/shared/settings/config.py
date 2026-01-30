from pydantic_settings import BaseSettings, SettingsConfigDict
from shared.settings.postgres import PostgresSettings
from shared.settings.llm import LLMSettings
from shared.settings.neo4j import Neo4jSettings

class Settings(BaseSettings):
    CONCURRENCY_LIMIT: int = 10
    
    postgres: PostgresSettings = PostgresSettings()
    llm: LLMSettings = LLMSettings()
    neo4j: Neo4jSettings = Neo4jSettings()

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore"
    )

settings = Settings()
