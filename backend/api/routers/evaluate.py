"""Evaluation API router."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
from app.evaluation import EvaluationService

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


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


class ConversationResult(BaseModel):
    """Result for a single conversation."""
    conv_id: str | int
    recall: float | None = None
    num_candidates: int | None = None
    num_re_ranked: int | None = None
    ground_truth_count: int | None = None
    error: str | None = None


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


@router.post("/", response_model=EvaluateResponse)
async def evaluate_endpoint(request: EvaluateRequest):
    """
    Run evaluation on a sample of the dataset.
    
    This endpoint processes a sample of conversations through the full evaluation pipeline:
    1. Summarize conversation to extract user preferences
    2. Retrieve candidate movies from graph database
    3. Re-rank movies using LLM
    4. Evaluate results against ground truth
    
    The results are saved to output files and returned with aggregate metrics.
    
    Args:
        request: Evaluation request parameters
        
    Returns:
        Evaluation results with recall metrics and processing details
        
    Raises:
        HTTPException: If evaluation fails
    """
    service = EvaluationService()
    
    try:
        result = await service.evaluate(
            dataset=request.dataset,
            sample_size=request.sample_size,
            start_index=request.start_index,
            n_sample=request.n_sample,
            top_k=request.top_k,
            model=request.model
        )
        
        # Add success message
        result["message"] = (
            f"Successfully evaluated {result['sample_size']} samples from {result['dataset']} dataset. "
            f"Average Recall@{request.top_k}: {result['avg_recall']:.4f}"
        )
        
        return EvaluateResponse(**result)
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/info")
async def get_evaluation_info():
    """
    Get information about available datasets and evaluation parameters.
    
    Returns:
        Information about supported datasets and parameter ranges
    """
    return {
        "datasets": ["inspired", "redial"],
        "parameters": {
            "sample_size": {
                "description": "Number of samples to evaluate",
                "range": "1-1000",
                "default": 40
            },
            "start_index": {
                "description": "Starting index for sampling",
                "range": "0+",
                "default": 0
            },
            "n_sample": {
                "description": "Number of candidate movies to retrieve",
                "range": "10-600",
                "default": 100,
                "options": [100, 200, 300, 400, 500, 600]
            },
            "top_k": {
                "description": "Top-k movies to re-rank",
                "range": "1-50",
                "default": 10,
                "options": [1, 5, 10, 50]
            },
            "model": {
                "description": "Reranking model to use",
                "default": "cohere",
                "options": ["llm", "cohere"]
            }
        },
        "pipeline_steps": [
            "1. Summarize conversation to extract user preferences",
            "2. Retrieve candidate movies from graph database",
            "3. Re-rank movies using LLM",
            "4. Evaluate results against ground truth"
        ],
        "metrics": ["recall@k"]
    }


# History API Models

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
    sample_size: int
    n_sample: int
    top_k: int
    model: str
    avg_recall: float
    
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


@router.get("/runs", response_model=List[EvaluationRunResponse])
async def get_evaluation_runs(skip: int = 0, limit: int = 100):
    """
    Get history of evaluation runs.
    """
    service = EvaluationService()
    return service.evaluation_storage_service.get_runs(skip=skip, limit=limit)


@router.get("/runs/{run_id}", response_model=EvaluationRunDetailResponse)
async def get_evaluation_run(run_id: int):
    """
    Get details of a specific evaluation run.
    """
    service = EvaluationService()
    run = service.evaluation_storage_service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run
