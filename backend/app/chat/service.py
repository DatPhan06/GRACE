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
        
        yield self._event("node", {
            "node": "profiler",
            "status": "done",
            "message": f"Profiler Agent: Done — extracted {len(preferences_data.liked_movies)} liked movies."
        })

        max_attempts = 2
        attempt = 0
        candidates = []
        final_movies = []
        
        while attempt < max_attempts:
            attempt += 1
            user_preferences = preferences_data.user_preferences
            liked_movies = preferences_data.liked_movies
            hard_constraints = preferences_data.hard_constraints
            semantic_queries = preferences_data.semantic_queries
            dynamic_weights = preferences_data.dynamic_weights.model_dump()
            
            if preferences_data.profiler_reasoning:
                agent_trace.append(f"Profiler (Attempt {attempt}): {preferences_data.profiler_reasoning}")

            # ── 2. Retrieval / Orchestrator ────────────────────────────────────
            yield self._event("node", {
                "node": "retrieval",
                "status": "running",
                "message": f"Orchestrator (Attempt {attempt}): Launching Semantic, Content & Graph Agents..."
            })

            retrieval_results = await self.retrieval_service.retrieve_movies(
                user_preferences,
                liked_movies,
                n=20,
                dynamic_weights=dynamic_weights,
                semantic_queries=semantic_queries,
                hard_constraints=hard_constraints,
                genres=preferences_data.genres
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
                "message": f"Critic Agent (Attempt {attempt}): Filtering & validating candidates..."
            })

            reranking_results = await self.reranking_service.rerank_movies(
                user_preferences, candidates, top_k=5, hard_constraints=hard_constraints
            )
            final_movies = reranking_results.get("movies", [])
            reranking_trace = reranking_results.get("agent_trace", [])
            requires_relaxation = reranking_results.get("requires_relaxation", False)
            critic_reasoning = reranking_results.get("reasoning", "")

            agent_trace.extend(reranking_trace)

            if requires_relaxation and attempt < max_attempts:
                yield self._event("node", {
                    "node": "reranking",
                    "status": "running",
                    "message": "Orchestrator: Constraints too tight. Triggering relaxation loop..."
                })
                preferences_data = await self.generation_service.relax_constraints(preferences_data, critic_reasoning)
                continue # Retry with relaxed preferences
            else:
                yield self._event("node", {
                    "node": "reranking",
                    "status": "done",
                    "message": f"Ranker: Done — selected top {len(final_movies)} recommendations."
                })
                break # Success or max attempts reached

        # ── 4. Generation ──────────────────────────────────────────────────
        yield self._event("node", {
            "node": "generation",
            "status": "running",
            "message": "GRACE: Composing your personalized response..."
        })

        response_text = await self.generation_service.generate_response(preferences_data.user_preferences, final_movies)

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
                "preferences": preferences_data.user_preferences,
                "liked_movies": preferences_data.liked_movies,
                "candidate_count": len(candidates),
                "attempts": attempt
            }
        })

    async def chat(self, conversation_history: str) -> Dict[str, Any]:
        """Non-streaming version (kept for backward compatibility with evaluation service)."""
        agent_trace = []

        preferences_data = await self.generation_service.summarize_conversation(conversation_history)
        
        max_attempts = 2
        attempt = 0
        candidates = []
        final_movies = []
        
        while attempt < max_attempts:
            attempt += 1
            user_preferences = preferences_data.user_preferences
            liked_movies = preferences_data.liked_movies
            hard_constraints = preferences_data.hard_constraints
            semantic_queries = preferences_data.semantic_queries
            dynamic_weights = preferences_data.dynamic_weights.model_dump()

            retrieval_results = await self.retrieval_service.retrieve_movies(
                user_preferences,
                liked_movies,
                n=20,
                dynamic_weights=dynamic_weights,
                semantic_queries=semantic_queries,
                hard_constraints=hard_constraints,
                genres=preferences_data.genres
            )
            candidates = retrieval_results.get("combined", [])
            retrieval_trace = retrieval_results.get("agent_trace", [])
            agent_trace.extend(retrieval_trace)

            reranking_results = await self.reranking_service.rerank_movies(
                user_preferences, candidates, top_k=5, hard_constraints=hard_constraints
            )
            final_movies = reranking_results.get("movies", [])
            reranking_trace = reranking_results.get("agent_trace", [])
            requires_relaxation = reranking_results.get("requires_relaxation", False)
            critic_reasoning = reranking_results.get("reasoning", "")

            agent_trace.extend(reranking_trace)

            if requires_relaxation and attempt < max_attempts:
                preferences_data = await self.generation_service.relax_constraints(preferences_data, critic_reasoning)
                continue
            else:
                break

        response_text = await self.generation_service.generate_response(preferences_data.user_preferences, final_movies)

        return {
            "response": response_text,
            "recommendations": final_movies,
            "agent_trace": agent_trace,
            "debug_info": {
                "preferences": preferences_data.user_preferences,
                "liked_movies": preferences_data.liked_movies,
                "candidate_count": len(candidates),
                "agent_trace": agent_trace,
                "attempts": attempt
            }
        }
