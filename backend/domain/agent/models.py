from typing import Literal
from pydantic import BaseModel, Field


class CriticResponse(BaseModel):
    approved_movie_ids: list[str] = Field(description="IDs of candidates that pass constraint verification")
    requires_relaxation: bool = Field(description="True if fewer than 3 valid candidates remain")
    critic_reasoning: str = Field(description="Explanation of which candidates were removed and why")


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


class RelaxationResponse(BaseModel):
    hard_constraints: list[HardConstraint] = Field(description="Updated constraints after minimal relaxation")
    semantic_queries: list[str] = Field(description="Updated semantic queries reflecting the widened search space")


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
