import json
from infra.llm import get_llm_client
from domain.agent.models import UserPreference, RetrievalWeights
from domain.generation.prompts import (
    SUMMARIZE_CONVERSATION_SYSTEM_PROMPT,
    SUMMARIZE_CONVERSATION_USER_PROMPT,
)
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

_DEFAULT_WEIGHTS = RetrievalWeights()


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
                response_schema=UserPreference,
            )
            data = json.loads(response)
            dw = data.get("dynamic_weights", {})
            if isinstance(dw, dict) and any(
                isinstance(v, (int, float)) and v < 0 for v in dw.values()
            ):
                logger.warning(
                    f"[Profiler] Negative weight(s) detected {dw} — falling back to defaults (0.40, 0.30, 0.30)."
                )
                data["dynamic_weights"] = _DEFAULT_WEIGHTS.model_dump()
            result = UserPreference(**data)
            result.dynamic_weights = self._normalize_weights(result.dynamic_weights)
            logger.info(f"[Profiler] weights={result.dynamic_weights.model_dump()}")
            return result
        except Exception as e:
            logger.error(f"[Profiler] Parse error, falling back to defaults: {e}")
            return UserPreference(user_preferences="", liked_movies=[], dynamic_weights=_DEFAULT_WEIGHTS)

    @staticmethod
    def _normalize_weights(w: RetrievalWeights) -> RetrievalWeights:
        if w.w_sem < 0 or w.w_con < 0 or w.w_col < 0:
            logger.warning(
                f"[Profiler] Negative weight(s) detected {w.model_dump()} — falling back to defaults (0.40, 0.30, 0.30)."
            )
            return _DEFAULT_WEIGHTS
        total = w.w_sem + w.w_con + w.w_col
        if total == 0:
            logger.warning("[Profiler] All weights are zero — falling back to defaults (0.40, 0.30, 0.30).")
            return _DEFAULT_WEIGHTS
        if abs(total - 1.0) > 1e-6:
            return RetrievalWeights(
                w_sem=round(w.w_sem / total, 6),
                w_con=round(w.w_con / total, 6),
                w_col=round(w.w_col / total, 6),
            )
        return w
