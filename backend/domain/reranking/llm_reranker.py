from infra.llm import get_llm_client
from domain.reranking.base import BaseReranker
from domain.reranking.prompts import RERANK_MOVIES_SYSTEM_PROMPT, RERANK_MOVIES_USER_PROMPT
from typing import List, Dict, Any
import json
import asyncio
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

# Two-stage batching hyperparameters (as described in the paper)
DEFAULT_BATCH_SIZE = 100   # B: number of candidates per batch
DEFAULT_SHORTLIST_SIZE = 20  # h: top results kept per batch before final rerank


class LLMReranker(BaseReranker):
    """
    Two-stage LLM reranker.

    Stage 1 – Batch reranking:
        Split candidates into batches of size B. Each batch is independently
        reranked by the LLM, keeping the top-h shortlist per batch.

    Stage 2 – Final reranking:
        Merge all per-batch shortlists and perform one final LLM reranking pass
        to produce the definitive top-k result list.
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shortlist_size: int = DEFAULT_SHORTLIST_SIZE,
    ):
        self.llm_client = get_llm_client()
        self.batch_size = batch_size
        self.shortlist_size = shortlist_size

    # ------------------------------------------------------------------
    # Public interface (BaseReranker)
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Two-stage batching rerank pipeline.

        Args:
            query:      User preference summary.
            candidates: Candidate movies with rich metadata.
            top_k:      Number of final results to return.
            **kwargs:   Accepts `conversation` (str) for prompt context.
        """
        conversation: str = kwargs.get("conversation", "")

        if not candidates:
            return []

        # ── Stage 1: per-batch reranking ──────────────────────────────
        batches = self._split_batches(candidates, self.batch_size)
        logger.info(
            f"[Stage 1] {len(candidates)} candidates → "
            f"{len(batches)} batch(es) of ≤{self.batch_size}, "
            f"shortlist_size={self.shortlist_size}"
        )

        # Run all batches concurrently
        stage1_tasks = [
            self._rerank_batch(query, batch, self.shortlist_size, conversation)
            for batch in batches
        ]
        shortlists: List[List[Dict[str, Any]]] = await asyncio.gather(*stage1_tasks)

        # Merge shortlists (preserve per-batch order, deduplicate)
        merged: List[Dict[str, Any]] = []
        seen_titles: set[str] = set()
        for shortlist in shortlists:
            for movie in shortlist:
                title_key = movie["title"].lower()
                if title_key not in seen_titles:
                    merged.append(movie)
                    seen_titles.add(title_key)

        logger.info(f"[Stage 2] Final reranking over {len(merged)} merged candidates → top-{top_k}")

        # ── Stage 2: final reranking ──────────────────────────────────
        if len(merged) <= top_k:
            # Already small enough; no second LLM call needed
            return merged[:top_k]

        return await self._rerank_batch(query, merged, top_k, conversation)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_batches(
        candidates: List[Dict[str, Any]], batch_size: int
    ) -> List[List[Dict[str, Any]]]:
        """Partition candidates into chunks of at most `batch_size`."""
        return [
            candidates[i : i + batch_size]
            for i in range(0, len(candidates), batch_size)
        ]

    @staticmethod
    def _build_candidates_str(candidates: List[Dict[str, Any]]) -> str:
        lines = [
            f"- {m['title']} (Year: {m.get('year')}): {m.get('plot', 'No plot available')}"
            for m in candidates
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_ranked_titles(response: str) -> List[str]:
        """Extract a JSON list of movie titles from raw LLM output."""
        cleaned = response.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        return json.loads(cleaned[start:end])

    @staticmethod
    def _map_titles_to_movies(
        ranked_titles: List[str],
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Map LLM-returned title strings back to full movie objects using
        fuzzy lowercase matching.
        """
        ranked_movies: List[Dict[str, Any]] = []
        for title in ranked_titles:
            title_lower = title.lower()
            for m in candidates:
                if (
                    m["title"].lower() == title_lower
                    or title_lower in m["title"].lower()
                ):
                    if m not in ranked_movies:
                        ranked_movies.append(m)
                        break

        # Fill remaining slots with unranked candidates if necessary
        for m in candidates:
            if m not in ranked_movies and len(ranked_movies) < top_k:
                ranked_movies.append(m)

        return ranked_movies[:top_k]

    async def _rerank_batch(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
        conversation: str,
    ) -> List[Dict[str, Any]]:
        """Call the LLM once to rerank a single batch, returning top-`top_k` movies."""
        candidates_str = self._build_candidates_str(candidates)
        prompt = RERANK_MOVIES_USER_PROMPT.format(
            conversation=conversation,
            user_preferences=query,
            candidates_str=candidates_str,
        )
        formatted_system_prompt = RERANK_MOVIES_SYSTEM_PROMPT.format(top_k=top_k)

        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=formatted_system_prompt,
            )
            ranked_titles = self._parse_ranked_titles(response)
            if not ranked_titles:
                logger.warning("LLM returned no parseable titles; using original order.")
                return candidates[:top_k]

            return self._map_titles_to_movies(ranked_titles, candidates, top_k)
        except Exception as e:
            logger.error(f"Error during batch LLM reranking: {e}")
            return candidates[:top_k]
