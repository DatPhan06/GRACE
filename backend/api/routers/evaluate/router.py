from fastapi import APIRouter, HTTPException
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
async def initialize_run(request: InitRunRequest):
    """Initialize a new evaluation run."""
    service = EvaluationService()
    try:
        return await service.initialize_run(
            dataset=request.dataset,
            sample_size=request.sample_size,
            start_index=request.start_index,
            name=request.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step/summarize")
async def run_summarization_step(request: SummarizationStepRequest):
    """Run summarization step for a specific run."""
    service = EvaluationService()
    try:
        return await service.run_summarization_step(
            run_id=request.run_id,
            name=request.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step/retrieve")
async def run_retrieval_step(request: RetrievalStepRequest):
    """Run retrieval step for a specific run."""
    service = EvaluationService()
    try:
        return await service.run_retrieval_step(
            run_id=request.run_id,
            n_sample=request.n_sample,
            summarization_batch_id=request.input_batch_id,
            name=request.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step/rerank")
async def run_reranking_step(request: RerankingStepRequest):
    """Run reranking step for a specific run."""
    service = EvaluationService()
    try:
        return await service.run_reranking_step(
            run_id=request.run_id,
            top_k=request.top_k,
            model=request.model,
            retrieval_batch_id=request.input_batch_id,
            name=request.name
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
            "1. Summarize conversation to extract user preferences",
            "2. Retrieve candidate movies from graph database",
            "3. Re-rank movies using LLM",
            "4. Evaluate results against ground truth"
        ],
        "metrics": ["recall@k"]
    }


