import json
from typing import List, Dict, Any
from infra.llm import get_llm_client
from domain.agent.models import HardConstraint
from domain.generation.prompts import CRITIC_SYSTEM_PROMPT, CRITIC_USER_PROMPT
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)


class CriticAgent:
    def __init__(self):
        self.llm_client = get_llm_client()

    async def filter_candidates(
        self,
        user_preferences: str,
        candidates: List[Dict[str, Any]],
        hard_constraints: List[HardConstraint] = None,
    ) -> Dict[str, Any]:
        if not candidates:
            return {"movies": [], "reasoning": "No candidates to filter."}

        logger.info(f"[Critic Agent] Filtering {len(candidates)} candidates (cross-stream verification)...")

        candidates_str = ""
        for m in candidates:
            year = m.get("year", "N/A")
            rating = m.get("imdbRating", "N/A")
            plot = str(m.get("plot", ""))[:150]
            candidates_str += (
                f"- ID: {m.get('movieId')} | Title: {m.get('title')} "
                f"| Year: {year} | IMDb: {rating} | Plot: {plot}...\n"
            )

        if hard_constraints:
            constraints_str = "; ".join(
                f"{c.constraint} [priority={c.priority}]" if hasattr(c, "constraint") else str(c)
                for c in hard_constraints
            )
        else:
            constraints_str = "None"

        prompt = CRITIC_USER_PROMPT.format(
            user_preferences=user_preferences,
            hard_constraints=constraints_str,
            candidates_str=candidates_str,
        )

        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=CRITIC_SYSTEM_PROMPT,
            )
            cleaned = response.replace("```json", "").replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start == -1 or end <= 0:
                logger.error("[Critic Agent] Invalid LLM output format. Passing through all candidates.")
                return {"movies": candidates, "reasoning": "Could not parse Critic Agent response."}

            data = json.loads(cleaned[start:end])
            approved_ids = data.get("approved_movie_ids", [])
            reasoning = data.get("critic_reasoning", "No specific reasoning provided.")
            logger.info(f"[Critic Agent] Thought: {reasoning}")

            approved_ids_set = {str(i) for i in approved_ids}
            filtered = [m for m in candidates if str(m.get("movieId")) in approved_ids_set]

            logger.info(f"[Critic Agent] Approved {len(filtered)} / {len(candidates)} candidates.")
            return {
                "movies": filtered,
                "reasoning": reasoning,
                "requires_relaxation": data.get("requires_relaxation", False) or len(filtered) < 3,
            }
        except Exception as e:
            logger.error(f"[Critic Agent] Error during reflection loop: {e}")
            return {"movies": candidates, "reasoning": f"Critic Agent error: {str(e)}"}
