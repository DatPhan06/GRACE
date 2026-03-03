from infra.llm.cohere_client import get_cohere_client
from domain.reranking.base import BaseReranker
from typing import List, Dict, Any
import asyncio
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

COHERE_RERANK_MODEL = "cohere.rerank-v3-5:0"

# Two-stage batching hyperparameters (mirrors LLMReranker, as per the paper)
DEFAULT_BATCH_SIZE = 100   # B: candidates per batch
DEFAULT_SHORTLIST_SIZE = 20  # h: top results kept per batch before final rerank


class CohereReranker(BaseReranker):
    """
    Two-stage Cohere reranker.

    Stage 1 – Batch reranking:
        Split candidates into batches of size B. Each batch is independently
        reranked by the Cohere API, keeping the top-h shortlist per batch.
        All batches are dispatched concurrently via asyncio.

    Stage 2 – Final reranking:
        Merge all per-batch shortlists and perform one final Cohere rerank call
        to produce the definitive top-k result list.
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shortlist_size: int = DEFAULT_SHORTLIST_SIZE,
    ):
        self.cohere_client = get_cohere_client()
        self.batch_size = batch_size
        self.shortlist_size = shortlist_size

    @property
    def is_available(self) -> bool:
        return self.cohere_client is not None

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
        Two-stage batching rerank pipeline using Cohere.

        Args:
            query:      User preference summary / search query.
            candidates: Candidate movies with rich metadata (title, year, plot).
            top_k:      Number of final results to return.
        """
        if not candidates:
            return []

        # ── Stage 1: per-batch reranking ──────────────────────────────
        batches = self._split_batches(candidates, self.batch_size)
        logger.info(
            f"[Cohere Stage 1] {len(candidates)} candidates → "
            f"{len(batches)} batch(es) of ≤{self.batch_size}, "
            f"shortlist_size={self.shortlist_size}"
        )

        # Run all batches concurrently (Cohere client is sync → run in executor)
        loop = asyncio.get_event_loop()
        stage1_tasks = [
            loop.run_in_executor(
                None,
                self._rerank_batch_sync,
                query,
                batch,
                self.shortlist_size,
            )
            for batch in batches
        ]
        shortlists: List[List[Dict[str, Any]]] = await asyncio.gather(*stage1_tasks)

        # Merge shortlists, preserving per-batch ranking order and deduplicating
        merged: List[Dict[str, Any]] = []
        seen_titles: set[str] = set()
        for shortlist in shortlists:
            for movie in shortlist:
                title_key = movie["title"].lower()
                if title_key not in seen_titles:
                    merged.append(movie)
                    seen_titles.add(title_key)

        logger.info(
            f"[Cohere Stage 2] Final reranking over {len(merged)} merged candidates → top-{top_k}"
        )

        # ── Stage 2: final reranking ──────────────────────────────────
        if len(merged) <= top_k:
            return merged[:top_k]

        return await loop.run_in_executor(
            None, self._rerank_batch_sync, query, merged, top_k
        )

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
    def _build_documents(candidates: List[Dict[str, Any]]) -> List[str]:
        """Format each candidate movie as a rich text document for Cohere."""
        return [
            f"{m['title']} (Year: {m.get('year')}) - {m.get('plot', '')}"
            for m in candidates
        ]

    def _rerank_batch_sync(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous Cohere rerank call for a single batch.
        Designed to be run inside asyncio.run_in_executor.
        """
        documents = self._build_documents(candidates)

        empty_plots = sum(1 for m in candidates if not m.get("plot"))
        if empty_plots > 0:
            logger.warning(
                f"Cohere Rerank: {empty_plots}/{len(candidates)} candidates have empty plots!"
            )

        try:
            response = self.cohere_client.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_k,
            )

            ranked_movies: List[Dict[str, Any]] = []
            if hasattr(response, "results"):
                for i, result in enumerate(response.results):
                    if i < 3:
                        logger.info(
                            f"Cohere Top-{i+1}: {candidates[result.index]['title']} "
                            f"(Score: {result.relevance_score:.4f})"
                        )
                    ranked_movies.append(candidates[result.index])

            return ranked_movies
        except Exception as e:
            logger.error(f"Error during Cohere batch reranking: {e}")
            return candidates[:top_k]
