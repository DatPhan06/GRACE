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
        hard_constraints: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Rerank a list of candidate movies based on user preferences.
        """
        logger.info(f"Reranking {len(candidates)} candidates using model: {model}")
        if not candidates:
            return {"movies": [], "agent_trace": []}

        agent_trace = []
        
        # 1. Critic Agent Reflection Filter
        from domain.reranking.components.critic_agent import CriticAgent
        critic = CriticAgent()
        critic_result = await critic.filter_candidates(user_preferences, candidates, hard_constraints=hard_constraints or [])
        clean_candidates = critic_result.get("movies", [])
        critic_reasoning = critic_result.get("reasoning", "")
        
        if critic_reasoning:
            agent_trace.append(f"Critic Agent: {critic_reasoning}")

        if not clean_candidates:
            logger.warning("[Reranker] Critic filtered ALL candidates. Reverting to original set.")
            agent_trace.append("Orchestrator: Critic Agent rejected all candidates. Reverting to original set for safety.")
            clean_candidates = candidates

        reranker = self._factory.create(model)

        # Cohere reranker requires an availability check before use
        if hasattr(reranker, "is_available") and not reranker.is_available:
            logger.warning(
                f"Reranker '{model}' is not available, falling back to 'llm'."
            )
            agent_trace.append(f"Orchestrator: Reranker '{model}' unavailable. Falling back to 'llm'.")
            reranker = self._factory.create("llm")

        agent_trace.append(f"Ranker: Finalizing top-{top_k} list using {model}...")
        
        final_movies = await reranker.rerank(
            query=user_preferences,
            candidates=clean_candidates,
            top_k=top_k,
            conversation=conversation,
        )
        
        return {
            "movies": final_movies,
            "agent_trace": agent_trace,
            "requires_relaxation": critic_result.get("requires_relaxation", False),
            "reasoning": critic_reasoning
        }

    async def run_critic(
        self,
        user_preferences: str,
        candidates: List[Dict[str, Any]],
        hard_constraints: List[str] = None,
    ) -> Dict[str, Any]:
        from domain.reranking.components.critic_agent import CriticAgent
        critic = CriticAgent()
        result = await critic.filter_candidates(
            user_preferences, candidates, hard_constraints=hard_constraints or []
        )
        if not result.get("movies"):
            logger.warning("[Critic] Filtered ALL candidates — reverting to original set.")
            result["movies"] = candidates
        return result

    async def run_reranker(
        self,
        user_preferences: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        model: RerankerType = "cohere",
    ) -> List[Dict[str, Any]]:
        reranker = self._factory.create(model)
        if hasattr(reranker, "is_available") and not reranker.is_available:
            logger.warning(f"Reranker '{model}' unavailable — falling back to 'llm'.")
            reranker = self._factory.create("llm")
        return await reranker.rerank(
            query=user_preferences, candidates=candidates, top_k=top_k
        )
