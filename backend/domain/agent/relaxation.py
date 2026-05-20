import json
from infra.llm import get_llm_client
from domain.agent.models import UserPreference
from domain.generation.prompts import (
    RELAX_CONSTRAINTS_SYSTEM_PROMPT,
    RELAX_CONSTRAINTS_USER_PROMPT,
)
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)


class RelaxationAgent:
    def __init__(self):
        self.llm_client = get_llm_client()

    async def run(self, preferences: UserPreference, critic_reasoning: str) -> UserPreference:
        logger.info(f"Relaxation Agent: relaxing constraints based on critic feedback: {critic_reasoning}")
        prompt = RELAX_CONSTRAINTS_USER_PROMPT.format(
            user_preferences=preferences.user_preferences,
            hard_constraints=", ".join(preferences.hard_constraints),
            semantic_queries="; ".join(preferences.semantic_queries),
            critic_reasoning=critic_reasoning,
        )

        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=RELAX_CONSTRAINTS_SYSTEM_PROMPT,
            )
            cleaned = response.replace("```json", "").replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > 0:
                data = json.loads(cleaned[start:end])
                return UserPreference(**data)
            return preferences
        except Exception as e:
            logger.error(f"Relaxation Agent error: {e}")
            return preferences
