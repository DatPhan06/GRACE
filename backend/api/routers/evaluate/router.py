from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from app.evaluation import EvaluationService
from .models import (
    EvaluateRequest,
    EvaluateResponse,
    InitRunRequest,
    SummarizationStepRequest,
    RetrievalStepRequest,
    RerankingStepRequest
)

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


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
            f"Average Recall@{request.top_k}: {result['avg_recall']:.3f}"
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


@router.post("/init")
async def initialize_run(request: InitRunRequest, background_tasks: BackgroundTasks):
    """Initialize a new evaluation run."""
    service = EvaluationService()
    try:
        return await service.initialize_run(
            dataset=request.dataset,
            sample_size=request.sample_size,
            start_index=request.start_index,
            name=request.name,
            background_tasks=background_tasks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step/profiler")
async def run_profiler_step(request: SummarizationStepRequest, background_tasks: BackgroundTasks):
    """Run Profiler Agent step — extract user preferences from conversation history."""
    service = EvaluationService()
    try:
        return await service.run_profiler_step(
            run_id=request.run_id,
            name=request.name,
            background_tasks=background_tasks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step/retrieve")
async def run_retrieval_step(request: RetrievalStepRequest, background_tasks: BackgroundTasks):
    """Run Retrieval step — semantic, content, and graph retrieval with WRRF fusion."""
    service = EvaluationService()
    try:
        return await service.run_retrieval_step(
            run_id=request.run_id,
            n_sample=request.n_sample,
            summarization_batch_id=request.input_batch_id,
            name=request.name,
            background_tasks=background_tasks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step/rerank")
async def run_reranking_step(request: RerankingStepRequest, background_tasks: BackgroundTasks):
    """Run Critic Agent + Reranker step — filter candidates then select top-K."""
    service = EvaluationService()
    try:
        return await service.run_reranking_step(
            run_id=request.run_id,
            top_k=request.top_k,
            model=request.model,
            retrieval_batch_id=request.input_batch_id,
            name=request.name,
            background_tasks=background_tasks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            "1. Profiler Agent — extract user preferences and retrieval weights",
            "2. Retrieval — semantic, content, and graph streams fused via WRRF",
            "3. Critic Agent — cross-stream verification and constraint filtering",
            "4. Reranker — select top-K from filtered candidates",
            "5. Evaluate — compute Recall@K against ground truth"
        ],
        "metrics": ["recall@k"]
    }


