from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate text from a prompt."""
        pass
