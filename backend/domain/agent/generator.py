from typing import List, Dict, Any
from infra.llm import get_llm_client
from domain.generation.prompts import (
    RECOMMENDATION_RESPONSE_SYSTEM_PROMPT,
    RECOMMENDATION_RESPONSE_USER_PROMPT,
)
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)


class GeneratorAgent:
    def __init__(self):
        self.llm_client = get_llm_client()

    async def run(
        self,
        user_preferences: str,
        recommendations: List[Dict[str, Any]],
        relaxation_note: str = "",
    ) -> str:
        movies_str = ", ".join(
            [f"{m['title']} ({m.get('year', 'N/A')})" for m in recommendations]
        )
        note_block = f"Relaxation Note: {relaxation_note}" if relaxation_note else ""
        prompt = RECOMMENDATION_RESPONSE_USER_PROMPT.format(
            user_preferences=user_preferences,
            movies_str=movies_str,
            relaxation_note=note_block,
        )
        return await self.llm_client.agenerate(
            prompt=prompt,
            system_instruction=RECOMMENDATION_RESPONSE_SYSTEM_PROMPT,
        )
