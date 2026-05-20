from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

from domain.generation.service import UserPreference


class ARGOSState(TypedDict):
    conversation_history: str
    preferences: Optional[UserPreference]
    candidates: List[Dict[str, Any]]
    critic_reasoning: str
    requires_relaxation: bool
    final_movies: List[Dict[str, Any]]
    response: str
    attempt: int
    agent_trace: Annotated[List[str], operator.add]
