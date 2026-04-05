from domain.retrieval.service import RetrievalService
from domain.generation.service import GenerationService
from domain.reranking.service import RerankingService
from typing import Dict, Any

class ChatService:
    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.generation_service = GenerationService()
        self.reranking_service = RerankingService()

    async def chat(self, conversation_history: str) -> Dict[str, Any]:
        # 1. Summarize conversation/intent (Profiler Agent)
        preferences_data = await self.generation_service.summarize_conversation(conversation_history)
        user_preferences = preferences_data.user_preferences
        liked_movies = preferences_data.liked_movies
        dynamic_weights = preferences_data.dynamic_weights.model_dump()  # {"w_sem": ..., "w_con": ..., "w_col": ...}
        
        # 2. Retrieve candidates (Orchestrator agent uses Profiler weights)
        retrieval_results = await self.retrieval_service.retrieve_movies(
            user_preferences, liked_movies, n=20, dynamic_weights=dynamic_weights
        )
        candidates = retrieval_results.get("combined", [])
        
        # 3. Rerank candidates
        final_movies = await self.reranking_service.rerank_movies(user_preferences, candidates, top_k=5)
        
        # 4. Generate response
        response_text = await self.generation_service.generate_response(user_preferences, final_movies)
        
        return {
            "response": response_text,
            "recommendations": final_movies,
            "debug_info": {
                "preferences": user_preferences,
                "liked_movies": liked_movies,
                "candidate_count": len(candidates)
            }
        }
