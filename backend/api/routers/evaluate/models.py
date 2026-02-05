from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any

class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""
    dataset: Literal["inspired", "redial"] = Field(
        default="redial",
        description="Dataset to evaluate (inspired or redial)"
    )
    sample_size: int = Field(
        default=40,
        ge=1,
        le=1000,
        description="Number of samples to evaluate (1-1000)"
    )
    start_index: int = Field(
        default=0,
        ge=0,
        description="Starting index for sampling"
    )
    n_sample: int = Field(
        default=100,
        ge=10,
        le=600,
        description="Number of candidate movies to retrieve (10-600)"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Top-k movies to re-rank (1-50)"
    )
    model: Literal["llm", "cohere"] = Field(
        default="cohere",
        description="Reranking model to use (llm or cohere)"
    )


class EvaluateResponse(BaseModel):
    """Response model for evaluation endpoint."""
    dataset: str
    sample_size: int
    start_index: int
    n_sample: int
    top_k: int
    model: str
    avg_recall: float
    results: List[Dict[str, Any]]
    output_dir: str
    message: str


class EvaluationResultResponse(BaseModel):
    """Detailed result for a single conversation in a run."""
    conv_id: str
    recall: float
    ground_truth: List[str] | Any | None = None
    recommendations: List[str] | Any | None = None
    candidate_count: int | None = None
    
    # Granular metrics
    recall_retrieval: float | None = None
    recall_semantic: float | None = None
    recall_content: float | None = None
    recall_collab: float | None = None
    
    semantic_count: int | None = None
    content_count: int | None = None
    collab_count: int | None = None
    
    error: str | None = None

    class Config:
        orm_mode = True


class EvaluationRunResponse(BaseModel):
    """Summary of an evaluation run."""
    id: int
    dataset: str
    sample_size: int | None = None
    n_sample: int | None = None
    top_k: int | None = None
    model: str | None = None
    avg_recall: float | None = None
    
    # Granular aggregates
    avg_recall_retrieval: float | None = None
    avg_recall_semantic: float | None = None
    avg_recall_content: float | None = None
    avg_recall_collab: float | None = None
    
    timestamp: Any = None

    class Config:
        orm_mode = True


class EvaluationRunDetailResponse(EvaluationRunResponse):
    """Detailed evaluation run with all results."""
    results: List[EvaluationResultResponse] = []
