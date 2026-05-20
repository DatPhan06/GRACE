from langgraph.graph import StateGraph, START, END

from app.chat.state import ARGOSState
from domain.generation.service import GenerationService
from domain.retrieval.service import RetrievalService
from domain.reranking.service import RerankingService

_generation = GenerationService()
_retrieval = RetrievalService()
_reranking = RerankingService()


async def profiler_node(state: ARGOSState) -> dict:
    preferences = await _generation.summarize_conversation(state["conversation_history"])
    trace = []
    if preferences.profiler_reasoning:
        trace.append(f"Profiler: {preferences.profiler_reasoning}")
    return {"preferences": preferences, "agent_trace": trace}


async def retrieval_node(state: ARGOSState) -> dict:
    prefs = state["preferences"]
    results = await _retrieval.retrieve_movies(
        user_preferences=prefs.user_preferences,
        liked_movies=prefs.liked_movies,
        n=20,
        dynamic_weights=prefs.dynamic_weights.model_dump(),
        semantic_queries=prefs.semantic_queries,
        hard_constraints=prefs.hard_constraints,
        genres=prefs.genres,
    )
    return {
        "candidates": results.get("combined", []),
        "attempt": state["attempt"] + 1,
        "agent_trace": results.get("agent_trace", []),
    }


async def critic_node(state: ARGOSState) -> dict:
    prefs = state["preferences"]
    result = await _reranking.run_critic(
        user_preferences=prefs.user_preferences,
        candidates=state["candidates"],
        hard_constraints=prefs.hard_constraints,
    )
    trace = []
    if result.get("reasoning"):
        trace.append(f"Critic Agent: {result['reasoning']}")
    return {
        "candidates": result.get("movies", state["candidates"]),
        "critic_reasoning": result.get("reasoning", ""),
        "requires_relaxation": result.get("requires_relaxation", False),
        "agent_trace": trace,
    }


def _route_after_critic(state: ARGOSState) -> str:
    if state["requires_relaxation"] and state["attempt"] < 2:
        return "relaxation"
    return "reranker"


async def relaxation_node(state: ARGOSState) -> dict:
    new_prefs = await _generation.relax_constraints(
        state["preferences"], state["critic_reasoning"]
    )
    return {
        "preferences": new_prefs,
        "agent_trace": ["Relaxation Agent: constraints relaxed based on critic feedback."],
    }


async def reranker_node(state: ARGOSState) -> dict:
    prefs = state["preferences"]
    movies = await _reranking.run_reranker(
        user_preferences=prefs.user_preferences,
        candidates=state["candidates"],
        top_k=5,
    )
    return {
        "final_movies": movies,
        "agent_trace": [f"Ranker: selected top-{len(movies)} recommendations."],
    }


async def generator_node(state: ARGOSState) -> dict:
    response = await _generation.generate_response(
        state["preferences"].user_preferences, state["final_movies"]
    )
    return {"response": response}


def build_graph():
    workflow = StateGraph(ARGOSState)

    workflow.add_node("profiler", profiler_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("relaxation", relaxation_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("generator", generator_node)

    workflow.add_edge(START, "profiler")
    workflow.add_edge("profiler", "retrieval")
    workflow.add_edge("retrieval", "critic")
    workflow.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"relaxation": "relaxation", "reranker": "reranker"},
    )
    workflow.add_edge("relaxation", "retrieval")
    workflow.add_edge("reranker", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()


graph = build_graph()
