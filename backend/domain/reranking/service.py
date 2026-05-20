from typing import List, Dict, Any, Literal
from shared.utils.logger import setup_logger
from domain.reranking.factory import RerankerFactory, RerankerType

logger = setup_logger(__name__)


class RerankingService:
    def __init__(self):
        self._factory = RerankerFactory

    async def rerank(
        self,
        user_preferences: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        model: RerankerType = "llm",
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        logger.info(f"Reranking {len(candidates)} candidates using model: {model}")
        reranker = self._factory.create(model)
        if hasattr(reranker, "is_available") and not reranker.is_available:
            logger.warning(f"Reranker '{model}' unavailable — falling back to 'llm'.")
            reranker = self._factory.create("llm")
        return await reranker.rerank(
            query=user_preferences, candidates=candidates, top_k=top_k
        )

    async def rerank_movies(
        self,
        user_preferences: str,
        candidates: List[Dict[str, Any]],
        conversation: str = "",
        top_k: int = 5,
        model: RerankerType = "llm",
        hard_constraints: List[str] = None,
    ) -> Dict[str, Any]:
        movies = await self.rerank(user_preferences, candidates, top_k, model)
        return {"movies": movies}
