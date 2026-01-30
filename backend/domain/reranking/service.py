from infra.llm import get_llm_client
from typing import List, Dict, Any
import logging
import json

class RerankingService:
    def __init__(self):
        self.llm_client = get_llm_client()

    async def rerank_movies(self, user_preferences: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate movies based on user preferences using LLM.
        """
        if not candidates:
            return []
            
        candidate_titles = [f"{m['title']} (Year: {m.get('year')})" for m in candidates]
        candidates_str = "\n".join(candidate_titles)
        
        prompt = f"""
        You are a movie recommendation expert.
        User Preferences: {user_preferences}
        
        Candidate Movies:
        {candidates_str}
        
        Please rank these movies from most suitable to least suitable for the user.
        Return the top {top_k} movies as a JSON list of titles.
        
        Example output:
        ["Movie A", "Movie B", "Movie C"]
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            start = cleaned_response.find("[")
            end = cleaned_response.rfind("]") + 1
            if start != -1 and end != -1:
                json_str = cleaned_response[start:end]
                ranked_titles = json.loads(json_str)
                
                # Map back to full movie objects
                ranked_movies = []
                for title in ranked_titles:
                    # Simple fuzzy matching or exact match logic
                    for m in candidates:
                        if m['title'].lower() in title.lower() or title.lower() in m['title'].lower():
                            if m not in ranked_movies:
                                ranked_movies.append(m)
                                break
                return ranked_movies[:top_k]
            else:
                 # Fallback: return original order
                 return candidates[:top_k]
        except Exception as e:
            logging.error(f"Error during reranking: {e}")
            return candidates[:top_k]
