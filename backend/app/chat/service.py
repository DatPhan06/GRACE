from typing import Dict, Any, AsyncGenerator
import json

from app.chat.graph import graph
from app.chat.state import ARGOSState

# Maps graph node names → NDJSON display names + messages
_NODE_CFG: Dict[str, Dict[str, str]] = {
    "profiler": {
        "display": "profiler",
        "start_msg": "Profiler Agent: Analyzing your movie preferences...",
        "end_msg":   "Profiler Agent: Done.",
    },
    "orchestrator": {
        "display": "orchestrator",
        "start_msg": "Orchestrator: Planning retrieval strategy...",
        "end_msg":   "Orchestrator: Done — streams activated.",
    },
    "retrieval": {
        "display": "retrieval",
        "start_msg": "Orchestrator: Launching Semantic, Content & Graph Agents in parallel...",
        "end_msg":   "Orchestrator: Done — candidates fused via WRRF.",
    },
    "critic": {
        "display": "reranking",
        "start_msg": "Critic Agent: Filtering & validating candidates...",
        "end_msg":   "Critic Agent: Done.",
    },
    "relaxation": {
        "display": "reranking",
        "start_msg": "Orchestrator: Constraints too tight. Triggering relaxation loop...",
        "end_msg":   "Relaxation Agent: Done.",
    },
    "reranker": {
        "display": "reranking",
        "start_msg": "Ranker: Finalizing top-K recommendations...",
        "end_msg":   "Ranker: Done.",
    },
    "generator": {
        "display": "generation",
        "start_msg": "GRACE: Composing your personalized response...",
        "end_msg":   "GRACE: Done!",
    },
}

_GRAPH_NODES = set(_NODE_CFG.keys())


def _make_initial_state(conversation_history: str) -> ARGOSState:
    return {
        "conversation_history": conversation_history,
        "preferences": None,
        "candidates": [],
        "critic_reasoning": "",
        "requires_relaxation": False,
        "final_movies": [],
        "response": "",
        "attempt": 0,
        "agent_trace": [],
    }


class ChatService:
    def _event(self, event_type: str, payload: Dict[str, Any]) -> str:
        return json.dumps({"event": event_type, **payload}) + "\n"

    async def chat_stream(self, conversation_history: str) -> AsyncGenerator[str, None]:
        initial = _make_initial_state(conversation_history)

        # Accumulated state across all node outputs
        acc: Dict[str, Any] = {"agent_trace": [], "attempt": 0}

        try:
            async for event in graph.astream_events(initial, version="v2"):
                kind = event["event"]
                name = event.get("name", "")
                meta = event.get("metadata", {})

                # Only handle top-level graph node events
                if meta.get("langgraph_node") != name or name not in _GRAPH_NODES:
                    continue

                cfg = _NODE_CFG[name]

                if kind == "on_chain_start":
                    yield self._event("node", {
                        "node": cfg["display"],
                        "status": "running",
                        "message": cfg["start_msg"],
                    })

                elif kind == "on_chain_end":
                    output = event.get("data", {}).get("output") or {}

                    # Accumulate agent_trace separately (LangGraph reducer appends)
                    acc["agent_trace"].extend(output.pop("agent_trace", []))
                    acc.update(output)

                    # Build dynamic end message
                    msg = cfg["end_msg"]
                    if name == "profiler" and acc.get("preferences"):
                        n = len(acc["preferences"].liked_movies)
                        msg = f"Profiler Agent: Done — extracted {n} liked movies."
                    elif name == "retrieval":
                        n = len(acc.get("candidates", []))
                        msg = f"Orchestrator: Done — fused {n} candidates via WRRF."
                    elif name == "reranker":
                        n = len(acc.get("final_movies", []))
                        msg = f"Ranker: Done — selected top-{n} recommendations."

                    yield self._event("node", {
                        "node": cfg["display"],
                        "status": "done",
                        "message": msg,
                    })

        except Exception as exc:
            yield self._event("error", {"detail": str(exc)})
            return

        prefs = acc.get("preferences")
        yield self._event("result", {
            "response": acc.get("response", ""),
            "recommendations": acc.get("final_movies", []),
            "agent_trace": acc["agent_trace"],
            "debug_info": {
                "preferences": prefs.user_preferences if prefs else "",
                "liked_movies": prefs.liked_movies if prefs else [],
                "candidate_count": len(acc.get("candidates", [])),
                "attempts": acc.get("attempt", 0),
            },
        })

    async def chat(self, conversation_history: str) -> Dict[str, Any]:
        """Non-streaming version (kept for backward compatibility with evaluation service)."""
        final = await graph.ainvoke(_make_initial_state(conversation_history))

        prefs = final.get("preferences")
        return {
            "response": final.get("response", ""),
            "recommendations": final.get("final_movies", []),
            "agent_trace": final.get("agent_trace", []),
            "debug_info": {
                "preferences": prefs.user_preferences if prefs else "",
                "liked_movies": prefs.liked_movies if prefs else [],
                "candidate_count": len(final.get("candidates", [])),
                "attempts": final.get("attempt", 0),
            },
        }
