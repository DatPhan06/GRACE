from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[type] = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt. If response_schema is a Pydantic model,
        the output is guaranteed to be a valid JSON string matching that schema."""
        pass

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[type] = None,
        **kwargs,
    ) -> str:
        """Async generate. If response_schema is a Pydantic model, the output is
        guaranteed to be a valid JSON string matching that schema."""
        pass
