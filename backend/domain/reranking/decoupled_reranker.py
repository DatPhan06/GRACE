from typing import List, Dict, Any
from domain.reranking.base import BaseReranker
from domain.reranking.cohere_reranker import CohereReranker
from domain.reranking.llm_reranker import LLMReranker
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

SHORTLIST_SIZE = 20


class DecoupledReranker(BaseReranker):
    """
    ARGOS two-stage decoupled reranker:
      Stage 1 — Cross-Encoder SLM (Cohere): fast semantic scoring → Top-20 shortlist.
      Stage 2 — Generative LLM: logic reasoning over shortlist → Top-K final.

    Falls back to LLM-only when Cohere is unavailable.
    """

    def __init__(self):
        self._cross_encoder = CohereReranker()
        self._llm = LLMReranker()

    @property
    def is_available(self) -> bool:
        return True  # always available; degrades gracefully when Cohere is absent

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if not self._cross_encoder.is_available:
            logger.warning(
                "[DecoupledReranker] Cohere unavailable — falling back to LLM-only reranking."
            )
            return await self._llm.rerank(query, candidates, top_k, **kwargs)

        # Stage 1: Cross-Encoder SLM → Top-20 shortlist
        shortlist = await self._cross_encoder.rerank(query, candidates, top_k=SHORTLIST_SIZE)
        logger.info(
            f"[DecoupledReranker Stage 1] Cross-Encoder SLM: {len(candidates)} candidates → {len(shortlist)} shortlisted."
        )

        if len(shortlist) <= top_k:
            return shortlist[:top_k]

        # Stage 2: Generative LLM → Top-K final
        result = await self._llm.rerank(query, shortlist, top_k, **kwargs)
        logger.info(
            f"[DecoupledReranker Stage 2] Generative LLM: {len(shortlist)} shortlist → top-{top_k} final."
        )
        return result
