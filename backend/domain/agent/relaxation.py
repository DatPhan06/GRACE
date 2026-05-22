import json
from infra.llm import get_llm_client
from domain.agent.models import UserPreference, HardConstraint, RelaxationResponse
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
        constraints_json = json.dumps(
            [c.model_dump() for c in preferences.hard_constraints], ensure_ascii=False
        )
        prompt = RELAX_CONSTRAINTS_USER_PROMPT.format(
            user_preferences=preferences.user_preferences,
            hard_constraints=constraints_json,
            semantic_queries="; ".join(preferences.semantic_queries),
            critic_reasoning=critic_reasoning,
        )

        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=RELAX_CONSTRAINTS_SYSTEM_PROMPT,
                response_schema=RelaxationResponse,
            )
            parsed = RelaxationResponse.model_validate_json(response)
            return preferences.model_copy(update={
                "hard_constraints": parsed.hard_constraints,
                "semantic_queries": parsed.semantic_queries,
            })
        except Exception as e:
            logger.error(f"Relaxation Agent error: {e}")
            return preferences
