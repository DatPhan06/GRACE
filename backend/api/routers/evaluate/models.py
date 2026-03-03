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






class InitRunRequest(BaseModel):
    name: str | None = None
    dataset: Literal["inspired", "redial"] = Field(default="redial")
    sample_size: int = Field(default=50, ge=1, le=1000)
    start_index: int = Field(default=0, ge=0)


class SummarizationStepRequest(BaseModel):
    run_id: int
    name: str | None = None

class RetrievalStepRequest(BaseModel):
    run_id: int
    name: str | None = None
    n_sample: int = Field(default=100, ge=10, le=600)
    input_batch_id: int | None = None

class RerankingStepRequest(BaseModel):
    run_id: int
    name: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    model: Literal["llm", "cohere"] = Field(default="cohere")
    input_batch_id: int | None = None


