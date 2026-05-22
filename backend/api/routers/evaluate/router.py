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
async def evaluate_endpoint(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """Run evaluation on a dataset sample. Returns immediately; processing runs in background."""
    service = EvaluationService()
    try:
        result = await service.evaluate(
            dataset=request.dataset,
            sample_size=request.sample_size,
            start_index=request.start_index,
            n_sample=request.n_sample,
            top_k=request.top_k,
            model=request.model,
            name=request.name,
            sample_percent=request.sample_percent,
            background_tasks=background_tasks,
        )
        return EvaluateResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/init")
async def initialize_run(request: InitRunRequest, background_tasks: BackgroundTasks):
    """Initialize a new evaluation run."""
    service = EvaluationService()
    try:
        return await service.initialize_run(
            dataset=request.dataset,
            sample_size=request.sample_size,
            sample_percent=request.sample_percent,
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
    """Run Critic → Relaxation → Reranker step — filter, relax if needed, then rank top-K."""
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
    """Get available datasets, parameter ranges, and dataset sizes."""
    import json
    from pathlib import Path
    from shared.settings.config import settings

    def _count(path_str: str) -> int:
        try:
            p = Path(path_str)
            if not p.is_absolute():
                p = Path(__file__).resolve().parents[4] / path_str
            with open(p, encoding="utf-8") as f:
                return len(json.load(f))
        except Exception:
            return 0

    dataset_sizes = {
        "redial": _count(settings.data.REDIAL_DIALOG_DATA),
        "inspired": _count(settings.data.INSPIRED_DIALOG_DATA),
    }

    provider = settings.llm.LLM_PROVIDER.lower()
    if provider == "gemini":
        llm_model = "gemini-2.0-flash"
    elif provider == "azure":
        llm_model = settings.llm.AZURE_LLM_MODEL
    elif provider == "openai":
        llm_model = "gpt-4o-mini"
    else:
        llm_model = provider

    return {
        "datasets": ["inspired", "redial"],
        "dataset_sizes": dataset_sizes,
        "llm_model": llm_model,
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
            "4. Relaxation Agent — if <3 candidates remain, widen constraints and re-retrieve",
            "5. Reranker — select top-K from filtered candidates",
            "6. Evaluate — compute Recall@K against ground truth"
        ],
        "metrics": ["recall@k"]
    }


