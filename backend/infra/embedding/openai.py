from openai import AsyncOpenAI, AsyncAzureOpenAI
from shared.settings.config import settings
from typing import List, Optional
import logging


class OpenAIEmbeddingService:
    def __init__(self):
        self.client = None
        cfg = settings.llm
        provider = cfg.LLM_PROVIDER.lower()

        try:
            if provider == "azure" and cfg.AZURE_OPENAI_API_KEY and cfg.AZURE_OPENAI_ENDPOINT:
                self.client = AsyncAzureOpenAI(
                    api_key=cfg.AZURE_OPENAI_API_KEY,
                    azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
                    api_version=cfg.AZURE_OPENAI_API_VERSION,
                )
                self.model = cfg.AZURE_EMBEDDING_MODEL
                logging.info(f"Azure OpenAI Embedding client initialized with deployment '{self.model}'")
            elif cfg.OPENAI_API_KEY:
                self.client = AsyncOpenAI(api_key=cfg.OPENAI_API_KEY)
                self.model = cfg.EMBEDDING_MODEL
                logging.info(f"OpenAI Embedding client initialized with model '{self.model}'")
            else:
                self.model = cfg.EMBEDDING_MODEL
                logging.warning("No embedding API key found.")
        except Exception as e:
            logging.error(f"Failed to initialize Embedding client: {e}")

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        if not self.client or not text:
            return None

        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error generating embedding with OpenAI: {e}")
            return None


_embedding_service = None


def get_embedding_service() -> OpenAIEmbeddingService:
    global _embedding_service
    if not _embedding_service:
        _embedding_service = OpenAIEmbeddingService()
    return _embedding_service
