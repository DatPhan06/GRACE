import json
from typing import List, Dict, Any
from infra.llm import get_llm_client
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

CRITIC_SYSTEM_PROMPT = r"""
You are the Critic Agent in a conversational recommender system. Your job is a cross-stream verification: candidates come from multiple retrieval sources (semantic vector search, content filter, graph traversal), and only the graph stream filters by hard constraints. The other streams may contain movies that violate the user's explicit requirements.

Your task:
1. Check every candidate against the provided Hard Constraints (year range, director, rating threshold, content restrictions, etc.).
2. Aggressively remove candidates that CLEARLY violate a hard constraint or are completely irrelevant to the core intent.
3. If a movie is borderline or uncertain, keep it — only remove obvious violations.
4. Set `requires_relaxation: true` if fewer than 3 valid candidates remain after filtering.

ALL candidate information (title, plot, year, rating) is sourced directly from the knowledge graph — judge solely based on this provided metadata, not your own parametric memory of the films.

Your response MUST be a valid JSON object:
{
    "approved_movie_ids": ["id1", "id2", "id3"],
    "requires_relaxation": true/false,
    "critic_reasoning": "Removed Movie X: user requested post-2015 release but it was from 2008. Only 2 candidates remain — recommend relaxing the year constraint."
}
DO NOT output markdown (`\```json`) or extra text outside the JSON block.
"""

CRITIC_USER_PROMPT = """
User Preferences: {user_preferences}

Hard Constraints: {hard_constraints}

Candidate Movies:
{candidates_str}

Please evaluate each candidate against the hard constraints and output the JSON of approved IDs.
"""


class CriticAgent:
    def __init__(self):
        self.llm_client = get_llm_client()

    async def filter_candidates(
        self,
        user_preferences: str,
        candidates: List[Dict[str, Any]],
        hard_constraints: List[str] = None,
    ) -> Dict[str, Any]:
        if not candidates:
            return {"movies": [], "reasoning": "No candidates to filter."}

        logger.info(f"[Critic Agent] Filtering {len(candidates)} candidates...")

        review_batch = candidates[:50]
        candidates_str = ""
        for m in review_batch:
            year = m.get("year", "N/A")
            rating = m.get("imdbRating", "N/A")
            plot = str(m.get("plot", ""))[:150]
            candidates_str += (
                f"- ID: {m.get('movieId')} | Title: {m.get('title')} "
                f"| Year: {year} | IMDb: {rating} | Plot: {plot}...\n"
            )

        if hard_constraints:
            constraints_str = "; ".join(
                c.constraint if hasattr(c, "constraint") else str(c)
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

            reviewed_ids = {str(m.get("movieId")) for m in review_batch}
            approved_ids_set = {str(i) for i in approved_ids}

            filtered = []
            for m in candidates:
                mid = str(m.get("movieId"))
                if mid in reviewed_ids:
                    if mid in approved_ids_set:
                        filtered.append(m)
                else:
                    filtered.append(m)

            logger.info(f"[Critic Agent] Filtered out {len(candidates) - len(filtered)} bad candidates.")
            return {
                "movies": filtered,
                "reasoning": reasoning,
                "requires_relaxation": data.get("requires_relaxation", False) or len(filtered) < 3,
            }
        except Exception as e:
            logger.error(f"[Critic Agent] Error during reflection loop: {e}")
            return {"movies": candidates, "reasoning": f"Critic Agent error: {str(e)}"}
