from infra.llm import get_llm_client
from infra.llm.cohere_client import get_cohere_client
from domain.reranking.prompts import RERANK_MOVIES_SYSTEM_PROMPT, RERANK_MOVIES_USER_PROMPT
from typing import List, Dict, Any, Optional, Literal
import logging
import json

class RerankingService:
    def __init__(self):
        self.llm_client = get_llm_client()
        # Initialize lazily or check availability via singleton
        self.cohere_client = get_cohere_client()

    async def rerank_movies(
        self, 
        user_preferences: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 5,
        model: Literal["llm", "cohere"] = "cohere"
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate movies based on user preferences.
        """
        if not candidates:
            return []
            
        if model == "cohere" and self.cohere_client:
            return await self._rerank_with_cohere(user_preferences, candidates, top_k)
            
        # Default to LLM
        return await self._rerank_with_llm(user_preferences, candidates, top_k)

    async def _rerank_with_cohere(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:

        # Construct document stings for reranking
        documents = [
            f"{m['title']} (Year: {m.get('year')}) - {m.get('plot', '')}" 
            for m in candidates
        ]
        
        try:
            # BedrockClientV2 usage:
            # Ensure parameters align with the V2 client. 
            # Usually: self.cohere_client.rerank(model="...", documents=..., query=...)
            # Check if top_n is the arg name (common in cohere) or if it changed.
            
            response = self.cohere_client.rerank(
                model="cohere.rerank-v3-5:0", # Bedrock model ID often looks like this
                query=query,
                documents=documents,
                top_n=top_k
            )
            
            ranked_movies = []
            # response.results is typically list of RerankResult
            if hasattr(response, 'results'):
                for result in response.results:
                    ranked_movies.append(candidates[result.index])
            
            return ranked_movies
        except Exception as e:
            logging.error(f"Error during Cohere reranking: {e}")
            # Fallback to original order or LLM? Let's fallback to returning top_k of original
            return candidates[:top_k]

    async def _rerank_with_llm(self, user_preferences: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        candidate_titles = [f"{m['title']} (Year: {m.get('year')})" for m in candidates]
        candidates_str = "\n".join(candidate_titles)
        
        prompt = RERANK_MOVIES_USER_PROMPT.format(
            user_preferences=user_preferences, 
            candidates_str=candidates_str, 
            top_k=top_k
        )
        
        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=RERANK_MOVIES_SYSTEM_PROMPT
            )
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
