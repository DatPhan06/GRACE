from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseReranker(ABC):
    """Abstract base class for all reranker implementations."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates given a query and return the top-k results."""
        ...
