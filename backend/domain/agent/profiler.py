import json
from infra.llm import get_llm_client
from domain.agent.models import UserPreference, RetrievalWeights
from domain.generation.prompts import (
    SUMMARIZE_CONVERSATION_SYSTEM_PROMPT,
    SUMMARIZE_CONVERSATION_USER_PROMPT,
)
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProfilerAgent:
    def __init__(self):
        self.llm_client = get_llm_client()

    async def run(self, conversation: str) -> UserPreference:
        logger.info(f"Profiler Agent: analyzing conversation with length: {len(conversation)}")
        prompt = SUMMARIZE_CONVERSATION_USER_PROMPT.format(conversation=conversation)

        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=SUMMARIZE_CONVERSATION_SYSTEM_PROMPT,
            )
            cleaned = response.replace("```json", "").replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > 0:
                json_str = cleaned[start:end]
                logger.info(f"[Profiler RAW] Parsed JSON before loading: {json_str[:300]}")
                data = json.loads(json_str)
                result = UserPreference(**data)
                w = result.dynamic_weights
                total = w.w_sem + w.w_con + w.w_col
                if total > 0 and abs(total - 1.0) > 1e-6:
                    result.dynamic_weights = RetrievalWeights(
                        w_sem=round(w.w_sem / total, 6),
                        w_con=round(w.w_con / total, 6),
                        w_col=round(w.w_col / total, 6),
                    )
                logger.info(f"[Profiler PARSED] weights={result.dynamic_weights.model_dump()}")
                return result
            else:
                logger.warning(f"Could not find JSON in response, falling back to defaults: {cleaned[:200]}")
                return UserPreference(user_preferences=response, liked_movies=[], dynamic_weights=RetrievalWeights())
        except Exception as e:
            logger.warning(f"Profiler Agent parse error, falling back to defaults: {e}")
            return UserPreference(user_preferences="", liked_movies=[], dynamic_weights=RetrievalWeights())
