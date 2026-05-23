from typing import List, Dict, Any
from infra.llm import get_llm_client
from domain.agent.models import HardConstraint, CriticResponse
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
            genres = m.get("genres") or m.get("genre", "")
            if isinstance(genres, list):
                genres = ", ".join(genres)
            genre_part = f" | Genres: {genres}" if genres else ""
            candidates_str += (
                f"- ID: {m.get('movieId')} | Title: {m.get('title')} "
                f"| Year: {year} | IMDb: {rating}{genre_part} | Plot: {plot}...\n"
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
                response_schema=CriticResponse,
            )
            parsed = CriticResponse.model_validate_json(response)
            logger.info(f"[Critic Agent] Thought: {parsed.critic_reasoning}")

            approved_ids_set = {str(i) for i in parsed.approved_movie_ids}
            filtered = [m for m in candidates if str(m.get("movieId")) in approved_ids_set]

            logger.info(f"[Critic Agent] Approved {len(filtered)} / {len(candidates)} candidates.")
            return {
                "movies": filtered,
                "reasoning": parsed.critic_reasoning,
                "requires_relaxation": parsed.requires_relaxation or len(filtered) < 3,
            }
        except Exception as e:
            logger.error(f"[Critic Agent] Error during reflection loop: {e}")
            return {"movies": candidates, "reasoning": f"Critic Agent error: {str(e)}"}
