from typing import Literal
from domain.reranking.base import BaseReranker
from domain.reranking.llm_reranker import LLMReranker
from domain.reranking.cohere_reranker import CohereReranker
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

RerankerType = Literal["llm", "cohere"]


class RerankerFactory:
    """Factory that creates and caches reranker instances by type."""

    _registry: dict[str, type[BaseReranker]] = {
        "llm": LLMReranker,
        "cohere": CohereReranker,
    }

    _instances: dict[str, BaseReranker] = {}

    @classmethod
    def create(cls, model: RerankerType) -> BaseReranker:
        """
        Return a cached reranker instance for the given model type.

        Args:
            model: One of the registered reranker types ("llm" or "cohere").

        Raises:
            ValueError: If the model type is not registered.
        """
        if model not in cls._registry:
            raise ValueError(
                f"Unknown reranker type: '{model}'. "
                f"Available types: {list(cls._registry)}"
            )

        if model not in cls._instances:
            logger.info(f"Instantiating reranker: {model}")
            cls._instances[model] = cls._registry[model]()

        return cls._instances[model]

    @classmethod
    def register(cls, name: str, reranker_cls: type[BaseReranker]) -> None:
        """
        Register a new reranker class under a given name.
        Allows extending the factory without modifying its code (Open/Closed Principle).

        Args:
            name: Identifier for the new reranker type.
            reranker_cls: Class that implements BaseReranker.
        """
        if not issubclass(reranker_cls, BaseReranker):
            raise TypeError(f"{reranker_cls} must be a subclass of BaseReranker")
        cls._registry[name] = reranker_cls
        logger.info(f"Registered new reranker: '{name}'")
