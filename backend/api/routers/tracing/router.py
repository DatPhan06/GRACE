from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.evaluation import EvaluationService
from .models import (
    EvaluationRunResponse,
    EvaluationRunDetailResponse,
    BatchStepExecutionResponse,
    BatchDetailResponse
)

router = APIRouter(prefix="/tracing", tags=["tracing"])


@router.get("/steps", response_model=List[BatchStepExecutionResponse])
async def get_steps_by_type(step_type: Optional[str] = Query(None), skip: int = 0, limit: int = 100):
    """
    Get all batch step executions, optionally filtered by step_type.
    """
    service = EvaluationService()
    return await service.get_steps_by_type(step_type, skip, limit)

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


@router.get("/runs/{run_id}/history", response_model=List[BatchStepExecutionResponse])
async def get_step_history(run_id: int):
    """
    Get execution history (batch steps) for a run.
    """
    service = EvaluationService()
    return await service.get_step_history(run_id)


@router.get("/runs/{run_id}/conversations")
async def get_run_conversations(run_id: int):
    """
    Get all conversation logs for a specific initialized run.
    """
    from infra.db.database import get_db
    from infra.db.models import ConversationLogModel

    db = next(get_db())
    try:
        logs = db.query(ConversationLogModel).filter(
            ConversationLogModel.run_id == run_id
        ).order_by(ConversationLogModel.index).all()
        if not logs:
            raise HTTPException(status_code=404, detail="No conversations found for this run")
        return [
            {
                "id": log.id,
                "conv_id": log.conv_id,
                "index": log.index,
                "status": log.status,
                "target": log.target,
                "liked_movies": log.liked_movies,
                "dialog_preview": (log.dialog or "")[:200],
            }
            for log in logs
        ]
    finally:
        db.close()


@router.get("/batches/{batch_id}/details", response_model=BatchDetailResponse)
async def get_batch_detail(batch_id: int, step_type: str = Query(..., description="Type of step (summarization, retrieval, reranking)")):
    """
    Get per-conversation detail data for a specific batch execution.
    """
    service = EvaluationService()
    result = await service.get_batch_detail(batch_id, step_type)
    if not result:
        raise HTTPException(status_code=404, detail="Batch not found")
    return result
