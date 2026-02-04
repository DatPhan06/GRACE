from openai import AsyncOpenAI
from shared.settings.config import settings
from typing import List, Optional
import logging


class OpenAIEmbeddingService:
    def __init__(self):
        self.client = None
        self.model = settings.llm.EMBEDDING_MODEL

        try:
            if settings.llm.OPENAI_API_KEY:
                self.client = AsyncOpenAI(api_key=settings.llm.OPENAI_API_KEY)
                logging.info(
                    f"OpenAI Embedding client initialized with model {self.model}")
            else:
                logging.warning("OpenAI API Key not found for embeddings.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI Embedding client: {e}")

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
