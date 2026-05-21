from typing import Literal
from pydantic import BaseModel, Field


class RetrievalWeights(BaseModel):
    w_sem: float = Field(default=0.4, ge=0, le=1)
    w_con: float = Field(default=0.3, ge=0, le=1)
    w_col: float = Field(default=0.3, ge=0, le=1)


class HardConstraint(BaseModel):
    constraint: str = Field(description="Natural language predicate expressing the constraint")
    priority: Literal["core", "soft", "optional"] = Field(
        default="soft",
        description="Priority tier for Sacrifice Hierarchy: core=never relax, soft=relax first, optional=can remove"
    )


class UserPreference(BaseModel):
    profiler_reasoning: str = Field(
        default="", description="Chain of thought reasoning from the Profiler Agent")
    user_preferences: str = Field(
        description="Summarized seeker's preferences")
    hard_constraints: list[HardConstraint] = Field(
        default=[],
        description="Explicit hard constraints with priority labels (core/soft/optional) for Sacrifice Hierarchy")
    genres: list[str] = Field(
        default=[], description="Explicit genre names extracted from user preferences")
    semantic_queries: list[str] = Field(
        default=[], description="List of sub-queries for semantic search")
    liked_movies: list[str] = Field(
        default=[], description="List of movies the user liked")
    dynamic_weights: RetrievalWeights = Field(
        default_factory=RetrievalWeights, description="Predicted weights for WRRF")
