from typing import List, Dict, Any, Literal
from shared.utils.logger import setup_logger
from domain.reranking.factory import RerankerFactory, RerankerType

logger = setup_logger(__name__)


class RerankingService:
    def __init__(self):
        self._factory = RerankerFactory

    async def rerank_movies(
        self,
        user_preferences: str,
        candidates: List[Dict[str, Any]],
        conversation: str = "",
        top_k: int = 5,
        model: RerankerType = "cohere",
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate movies based on user preferences.

        The concrete reranker is resolved at runtime via RerankerFactory.
        Falls back to LLM reranker if the requested reranker is unavailable.
        """
        logger.info(f"Reranking {len(candidates)} candidates using model: {model}")
        if not candidates:
            return []

        # 1. Critic Agent Reflection Filter
        from domain.reranking.components.critic_agent import CriticAgent
        critic = CriticAgent()
        clean_candidates = await critic.filter_candidates(user_preferences, candidates)
        if not clean_candidates:
            logger.warning("[Reranker] Critic filtered ALL candidates. Reverting to original set.")
            clean_candidates = candidates

        reranker = self._factory.create(model)

        # Cohere reranker requires an availability check before use
        if hasattr(reranker, "is_available") and not reranker.is_available:
            logger.warning(
                f"Reranker '{model}' is not available, falling back to 'llm'."
            )
            reranker = self._factory.create("llm")

        return await reranker.rerank(
            query=user_preferences,
            candidates=clean_candidates,
            top_k=top_k,
            conversation=conversation,
        )
