import json
from infra.llm import get_llm_client
from domain.agent.models import UserPreference, HardConstraint
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
            )
            cleaned = response.replace("```json", "").replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > 0:
                data = json.loads(cleaned[start:end])
                # Relaxation only updates hard_constraints + semantic_queries.
                # All other Profiler fields (user_preferences, genres, liked_movies,
                # dynamic_weights, profiler_reasoning) are preserved explicitly.
                raw_constraints = data.get(
                    "hard_constraints",
                    [c.model_dump() for c in preferences.hard_constraints],
                )
                relaxed_constraints = [
                    HardConstraint(**c) if isinstance(c, dict) else c
                    for c in raw_constraints
                ]
                return preferences.model_copy(update={
                    "hard_constraints": relaxed_constraints,
                    "semantic_queries": data.get("semantic_queries", preferences.semantic_queries),
                })
            return preferences
        except Exception as e:
            logger.error(f"Relaxation Agent error: {e}")
            return preferences
