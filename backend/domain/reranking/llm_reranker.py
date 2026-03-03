from infra.llm import get_llm_client
from domain.reranking.base import BaseReranker
from domain.reranking.prompts import RERANK_MOVIES_SYSTEM_PROMPT, RERANK_MOVIES_USER_PROMPT
from typing import List, Dict, Any
import json
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMReranker(BaseReranker):
    def __init__(self):
        self.llm_client = get_llm_client()

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using an LLM."""
        conversation: str = kwargs.get("conversation", "")

        candidate_titles = [
            f"- {m['title']} (Year: {m.get('year')}): {m.get('plot', 'No plot available')}"
            for m in candidates
        ]
        candidates_str = "\n".join(candidate_titles)

        prompt = RERANK_MOVIES_USER_PROMPT.format(
            conversation=conversation,
            user_preferences=query,
            candidates_str=candidates_str,
        )

        try:
            formatted_system_prompt = RERANK_MOVIES_SYSTEM_PROMPT.format(top_k=top_k)

            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=formatted_system_prompt,
            )
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            start = cleaned_response.find("[")
            end = cleaned_response.rfind("]") + 1
            if start != -1 and end != -1:
                json_str = cleaned_response[start:end]
                ranked_titles = json.loads(json_str)

                ranked_movies = []
                for title in ranked_titles:
                    for m in candidates:
                        if m["title"].lower() == title.lower() or title.lower() in m["title"].lower():
                            if m not in ranked_movies:
                                ranked_movies.append(m)
                                break

                # Fill up to top_k if not enough matches
                for m in candidates:
                    if m not in ranked_movies and len(ranked_movies) < top_k:
                        ranked_movies.append(m)

                return ranked_movies[:top_k]
            else:
                return candidates[:top_k]
        except Exception as e:
            logger.error(f"Error during LLM reranking: {e}")
            return candidates[:top_k]
