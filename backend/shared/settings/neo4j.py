from pydantic_settings import BaseSettings
from typing import Optional

class Neo4jSettings(BaseSettings):
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str
