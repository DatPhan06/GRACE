from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Neo4jSettings(BaseSettings):
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        env_ignore_empty=True,
        extra="ignore"
    )
