import json
from typing import List, Dict, Any
from shared.utils.logger import setup_logger
from infra.llm import get_llm_client

logger = setup_logger(__name__)

CRITIC_SYSTEM_PROMPT = """
You are the Reranking Critic Agent in a conversational recommender system. Your job is to perform a fast, harsh filter on candidate movies before they are sent to the final scoring model.
You will be provided with the user's intent/preferences and a list of candidate movies retrieved from various sources (vector, text, graph).
Analyze the candidates and aggressively filter out any movies that clearly hallucinatory, flagrantly violate a hard constraint (like asking for a comedy and getting a horror movie), or are simply irrelevant to the core intent.

If a movie is borderline or potentially interesting, keep it. 
Only filter out bad matches.

Your response MUST be a valid JSON object containing the list of approved `movieId`s like this:
{
    "approved_movie_ids": ["id1", "id2", "id3"],
    "requires_relaxation": true/false,
    "critic_reasoning": "I removed Movie X because it was a horror but the user asked for children's comedy. Very few movies left, consider relaxing genre constraints."
}
DO NOT output markdown (`\```json`) or extra text outside the JSON block.
"""

CRITIC_USER_PROMPT = """
User Preferences: {user_preferences}

Candidate Movies:
{candidates_str}

Please evaluate and output the JSON of approved IDs.
"""

class CriticAgent:
    def __init__(self):
        self.llm_client = get_llm_client()

    async def filter_candidates(self, user_preferences: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            return {"movies": [], "reasoning": "No candidates to filter."}
            
        logger.info(f"[Critic Agent] Filtering {len(candidates)} candidates...")
        
        # Batching for LLM context limit - limit to top 50 to avoid massive prompt
        review_batch = candidates[:50]
        
        candidates_str = ""
        for m in review_batch:
            candidates_str += f"- ID: {m.get('movieId')} | Title: {m.get('title')} | Plot: {str(m.get('plot'))[:100]}...\n"

        prompt = CRITIC_USER_PROMPT.format(
            user_preferences=user_preferences,
            candidates_str=candidates_str
        )

        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=CRITIC_SYSTEM_PROMPT
            )
            
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            start = cleaned_response.find("{")
            end = cleaned_response.rfind("}") + 1
            if start == -1 or end <= 0:
                logger.error(f"[Critic Agent] Invalid LLM output format. Passing through all candidates.")
                return {"movies": candidates, "reasoning": "Could not parse Critic Agent response."}

            data = json.loads(cleaned_response[start:end])
            approved_ids = data.get("approved_movie_ids", [])
            reasoning = data.get("critic_reasoning", "No specific reasoning provided.")
            
            logger.info(f"[Critic Agent] Thought: {reasoning}")
            
            reviewed_ids = {str(m.get('movieId')) for m in review_batch}
            approved_ids_set = {str(i) for i in approved_ids}
            
            filtered_candidates = []
            for m in candidates:
                mid = str(m.get('movieId'))
                if mid in reviewed_ids:
                    if mid in approved_ids_set:
                        filtered_candidates.append(m)
                else:
                    # Not reviewed, keep it
                    filtered_candidates.append(m)
                    
            logger.info(f"[Critic Agent] Filtered out {len(candidates) - len(filtered_candidates)} bad candidates.")
            return {
                "movies": filtered_candidates, 
                "reasoning": reasoning, 
                "requires_relaxation": data.get("requires_relaxation", False) or len(filtered_candidates) < 3
            }

        except Exception as e:
            logger.error(f"[Critic Agent] Error during reflection loop: {e}")
            return {"movies": candidates, "reasoning": f"Critic Agent error: {str(e)}"}
