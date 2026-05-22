from pydantic import BaseModel
from typing import List, Any

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
    name: str | None = None
    dataset: str
    sample_size: int | None = None
    n_sample: int | None = None
    top_k: int | None = None
    model: str | None = None
    llm_model: str | None = None
    avg_recall: float | None = None
    
    # Granular aggregates
    avg_recall_retrieval: float | None = None
    avg_recall_semantic: float | None = None
    avg_recall_content: float | None = None
    avg_recall_collab: float | None = None
    
    status: str | None = None
    timestamp: Any = None

    class Config:
        orm_mode = True



class EvaluationRunDetailResponse(EvaluationRunResponse):
    """Detailed evaluation run with all results."""
    results: List[EvaluationResultResponse] = []


from typing import Dict, Optional
class BatchStepExecutionResponse(BaseModel):
    id: int
    run_id: int
    name: Optional[str] = None
    step_type: str
    version: int
    config: Dict[str, Any]
    status: str
    created_at: Any
    input_batch_id: Optional[int] = None

    class Config:
        orm_mode = True


# --- Batch Detail Models ---

class SummarizationDetailItem(BaseModel):
    conv_id: str
    user_preferences: Optional[str] = None

    class Config:
        orm_mode = True


class RetrievalDetailItem(BaseModel):
    conv_id: str
    candidates: Optional[List[Any]] = None
    candidate_count: int = 0
    semantic_count: Optional[int] = None
    content_count: Optional[int] = None
    collab_count: Optional[int] = None

    class Config:
        orm_mode = True


class RerankingDetailItem(BaseModel):
    conv_id: str
    model_used: Optional[str] = None
    reranked_candidates: Optional[List[Any]] = None
    reranked_count: int = 0
    recall: Optional[float] = None

    class Config:
        orm_mode = True


class BatchDetailResponse(BaseModel):
    id: int
    run_id: int
    step_type: str
    version: int
    config: Dict[str, Any]
    status: str
    created_at: Any
    items: List[Any] = []

    class Config:
        orm_mode = True
