from infra.llm.cohere_client import get_cohere_client
from domain.reranking.base import BaseReranker
from typing import List, Dict, Any
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

COHERE_RERANK_MODEL = "cohere.rerank-v3-5:0"


class CohereReranker(BaseReranker):
    def __init__(self):
        self.cohere_client = get_cohere_client()

    @property
    def is_available(self) -> bool:
        return self.cohere_client is not None

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using Cohere rerank model."""
        documents = [
            f"{m['title']} (Year: {m.get('year')}) - {m.get('plot', '')}"
            for m in candidates
        ]

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

            ranked_movies = []
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
            logger.error(f"Error during Cohere reranking: {e}")
            return candidates[:top_k]
