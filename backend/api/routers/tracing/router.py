from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional
import io
import csv
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


@router.delete("/runs/{run_id}", status_code=204)
async def delete_evaluation_run(run_id: int):
    """Delete an evaluation run and all its results."""
    from infra.db.database import get_db
    from infra.db.models import EvaluationRunModel, EvaluationResultModel

    db = next(get_db())
    try:
        run = db.query(EvaluationRunModel).filter_by(id=run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        db.query(EvaluationResultModel).filter_by(run_id=run_id).delete()
        db.delete(run)
        db.commit()
    finally:
        db.close()


@router.get("/runs/{run_id}/export")
async def export_evaluation_run(run_id: int):
    """Export evaluation run results as a TSV file matching the ARGOS output format."""
    from infra.db.database import get_db
    from infra.db.models import (
        EvaluationRunModel, EvaluationResultModel, ConversationLogModel,
        StepRetrievalModel, RetrievalRunModel, SummarizationRunModel
    )

    db = next(get_db())
    try:
        run = db.query(EvaluationRunModel).filter_by(id=run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        results = (
            db.query(EvaluationResultModel, ConversationLogModel)
            .outerjoin(ConversationLogModel, EvaluationResultModel.conversation_id == ConversationLogModel.id)
            .filter(EvaluationResultModel.run_id == run_id)
            .order_by(EvaluationResultModel.id)
            .all()
        )

        # Build a mapping: conversation_id → full candidate list (from step-based retrieval if available)
        candidate_map: dict = {}
        conv_ids = [log.id for _, log in results if log is not None]
        if conv_ids:
            latest_retrieval_rows = (
                db.query(StepRetrievalModel)
                .filter(StepRetrievalModel.conversation_id.in_(conv_ids))
                .order_by(StepRetrievalModel.retrieval_batch_id.desc())
                .all()
            )
            # Keep only the latest retrieval per conversation
            seen = set()
            for row in latest_retrieval_rows:
                if row.conversation_id not in seen:
                    seen.add(row.conversation_id)
                    candidates = row.candidates or []
                    titles = [c["title"] for c in candidates if isinstance(c, dict) and "title" in c]
                    candidate_map[row.conversation_id] = titles

        # Generate TSV
        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "recommend_item", "recommend_movie_list", "movie_candidate_list", "recall"])

        for result, log in results:
            # recommend_item = ground truth target
            gt = result.ground_truth
            if isinstance(gt, list):
                recommend_item = "|".join(str(g) for g in gt)
            else:
                recommend_item = str(gt) if gt is not None else ""

            # recommend_movie_list = final top-K recommendations
            recs = result.recommendations
            if isinstance(recs, list):
                recommend_movie_list = "|".join(str(r) for r in recs)
            else:
                recommend_movie_list = str(recs) if recs is not None else ""

            # movie_candidate_list = full retrieval pool (step-based) or fall back to recommendations
            conv_id_int = log.id if log else None
            if conv_id_int and conv_id_int in candidate_map:
                movie_candidate_list = "|".join(candidate_map[conv_id_int])
            else:
                movie_candidate_list = recommend_movie_list  # fallback

            recall_val = round(result.recall, 4) if result.recall is not None else 0.0

            writer.writerow([result.conv_id, recommend_item, recommend_movie_list, movie_candidate_list, recall_val])

        tsv_bytes = output.getvalue().encode("utf-8")

        model_name = run.llm_model or run.model or "unknown"
        dataset = (run.dataset or "unknown").upper()
        top_k = run.top_k or 10
        n_sample = run.n_sample or 0
        filename = f"{model_name}_{dataset}_recall@{top_k}_{n_sample}sample.tsv"

        return StreamingResponse(
            io.BytesIO(tsv_bytes),
            media_type="text/tab-separated-values",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
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
