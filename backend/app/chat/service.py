from domain.retrieval.service import RetrievalService
from domain.generation.service import GenerationService
from domain.reranking.service import RerankingService
from typing import Dict, Any, AsyncGenerator
import json


class ChatService:
    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.generation_service = GenerationService()
        self.reranking_service = RerankingService()

    def _event(self, event_type: str, payload: Dict[str, Any]) -> str:
        """Serialize a single NDJSON event line."""
        return json.dumps({"event": event_type, **payload}) + "\n"

    async def chat_stream(self, conversation_history: str) -> AsyncGenerator[str, None]:
        """
        Stream agent status events followed by the final result.
        Each line is a JSON object with an 'event' key.

        Events:
          - {"event": "node", "node": "profiler", "status": "running", "message": "..."}
          - {"event": "node", "node": "retrieval", "status": "running", ...}
          - {"event": "node", "node": "reranking", "status": "running", ...}
          - {"event": "node", "node": "generation", "status": "running", ...}
          - {"event": "result", "response": "...", "recommendations": [...], "agent_trace": [...]}
        """
        agent_trace = []

        # ── 1. Profiler Agent ──────────────────────────────────────────────
        yield self._event("node", {
            "node": "profiler",
            "status": "running",
            "message": "Profiler Agent: Analyzing your movie preferences..."
        })

        preferences_data = await self.generation_service.summarize_conversation(conversation_history)
        user_preferences = preferences_data.user_preferences
        liked_movies = preferences_data.liked_movies
        dynamic_weights = preferences_data.dynamic_weights.model_dump()
        profiler_reasoning = preferences_data.profiler_reasoning

        if profiler_reasoning:
            agent_trace.append(f"Profiler Agent: {profiler_reasoning}")

        yield self._event("node", {
            "node": "profiler",
            "status": "done",
            "message": f"Profiler Agent: Done — extracted {len(liked_movies)} liked movies."
        })

        # ── 2. Retrieval / Orchestrator ────────────────────────────────────
        yield self._event("node", {
            "node": "retrieval",
            "status": "running",
            "message": "Orchestrator: Launching Semantic, Content & Graph Agents..."
        })

        retrieval_results = await self.retrieval_service.retrieve_movies(
            user_preferences, liked_movies, n=20, dynamic_weights=dynamic_weights
        )
        candidates = retrieval_results.get("combined", [])
        retrieval_trace = retrieval_results.get("agent_trace", [])
        agent_trace.extend(retrieval_trace)

        yield self._event("node", {
            "node": "retrieval",
            "status": "done",
            "message": f"Orchestrator: Done — fused {len(candidates)} candidates via WRRF."
        })

        # ── 3. Critic + Reranking ──────────────────────────────────────────
        yield self._event("node", {
            "node": "reranking",
            "status": "running",
            "message": "Critic Agent: Filtering & validating candidates..."
        })

        reranking_results = await self.reranking_service.rerank_movies(
            user_preferences, candidates, top_k=5
        )
        final_movies = reranking_results.get("movies", [])
        reranking_trace = reranking_results.get("agent_trace", [])
        agent_trace.extend(reranking_trace)

        yield self._event("node", {
            "node": "reranking",
            "status": "done",
            "message": f"Ranker: Done — selected top {len(final_movies)} recommendations."
        })

        # ── 4. Generation ──────────────────────────────────────────────────
        yield self._event("node", {
            "node": "generation",
            "status": "running",
            "message": "GRACE: Composing your personalized response..."
        })

        response_text = await self.generation_service.generate_response(user_preferences, final_movies)

        yield self._event("node", {
            "node": "generation",
            "status": "done",
            "message": "GRACE: Done!"
        })

        # ── Final result event ─────────────────────────────────────────────
        yield self._event("result", {
            "response": response_text,
            "recommendations": final_movies,
            "agent_trace": agent_trace,
            "debug_info": {
                "preferences": user_preferences,
                "liked_movies": liked_movies,
                "candidate_count": len(candidates),
            }
        })

    async def chat(self, conversation_history: str) -> Dict[str, Any]:
        """Non-streaming version (kept for backward compatibility with evaluation service)."""
        agent_trace = []

        preferences_data = await self.generation_service.summarize_conversation(conversation_history)
        user_preferences = preferences_data.user_preferences
        liked_movies = preferences_data.liked_movies
        dynamic_weights = preferences_data.dynamic_weights.model_dump()
        profiler_reasoning = preferences_data.profiler_reasoning

        if profiler_reasoning:
            agent_trace.append(f"Profiler Agent: {profiler_reasoning}")

        retrieval_results = await self.retrieval_service.retrieve_movies(
            user_preferences, liked_movies, n=20, dynamic_weights=dynamic_weights
        )
        candidates = retrieval_results.get("combined", [])
        retrieval_trace = retrieval_results.get("agent_trace", [])
        agent_trace.extend(retrieval_trace)

        reranking_results = await self.reranking_service.rerank_movies(
            user_preferences, candidates, top_k=5
        )
        final_movies = reranking_results.get("movies", [])
        reranking_trace = reranking_results.get("agent_trace", [])
        agent_trace.extend(reranking_trace)

        response_text = await self.generation_service.generate_response(user_preferences, final_movies)

        return {
            "response": response_text,
            "recommendations": final_movies,
            "agent_trace": agent_trace,
            "debug_info": {
                "preferences": user_preferences,
                "liked_movies": liked_movies,
                "candidate_count": len(candidates),
                "agent_trace": agent_trace
            }
        }
