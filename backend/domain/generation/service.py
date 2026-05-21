from domain.agent.models import UserPreference, RetrievalWeights
from domain.agent.profiler import ProfilerAgent
from domain.agent.generator import GeneratorAgent
from domain.agent.relaxation import RelaxationAgent
from typing import List, Dict, Any

# Re-export models so existing callers (app/chat/state.py, etc.) don't need to change
__all__ = ["UserPreference", "RetrievalWeights", "GenerationService"]


class GenerationService:
    def __init__(self):
        self._profiler = ProfilerAgent()
        self._generator = GeneratorAgent()
        self._relaxation = RelaxationAgent()

    async def summarize_conversation(self, conversation: str) -> UserPreference:
        return await self._profiler.run(conversation)

    async def generate_response(
        self,
        user_preferences: str,
        recommendations: List[Dict[str, Any]],
        relaxation_note: str = "",
        conversation: str = "",
    ) -> str:
        return await self._generator.run(user_preferences, recommendations, relaxation_note, conversation)

    async def relax_constraints(
        self, preferences: UserPreference, critic_reasoning: str
    ) -> UserPreference:
        return await self._relaxation.run(preferences, critic_reasoning)
