"""Evaluation application service."""
import json
import json
import pandas as pd
from pathlib import Path
from typing import Literal, Dict, Any, List, Optional
import random
from fastapi import BackgroundTasks

# Domain services
from domain.generation.service import GenerationService
from domain.retrieval.service import RetrievalService
from domain.reranking.service import RerankingService
from domain.evaluation import EvaluationDomainService
from domain.evaluation_storage.service import EvaluationStorageService
from shared.settings.config import settings
from shared.utils.logger import setup_logger
from infra.db.models import EvaluationRunModel, ConversationLogModel, StepSummarizationModel, StepRetrievalModel, StepRerankingModel, EvaluationResultModel, SummarizationRunModel, RetrievalRunModel, RerankingRunModel
from infra.db.database import get_db

logger = setup_logger(__name__)

class EvaluationService:
    """
    Orchestrates the evaluation process by coordinating domain services.
    Supported modes: 
    - Full evaluation (Legacy)
    - Step-by-step execution for optimization
    """

    def __init__(self):
        # Initialize domain services
        self.generation_service = GenerationService()
        self.retrieval_service = RetrievalService()
        self.reranking_service = RerankingService()
        self.evaluation_domain_service = EvaluationDomainService()
        self.evaluation_storage_service = EvaluationStorageService()

        # Config is now handled via settings
        self.project_root = Path(__file__).parent.parent.parent.parent

    async def initialize_run(
        self,
        dataset: Literal["inspired", "redial"] = "redial",
        sample_size: int = 50,
        start_index: int = 0,
        name: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> Dict[str, Any]:
        """
        Initialize a new evaluation run.
        Loads the dataset sample and creates ConversationLog entries in 'pending' status.
        """
        conversations = self.load_dialogs_sample(dataset, sample_size)
        
        db = next(get_db())
        try:
            # Create Run
            run = EvaluationRunModel(
                name=name,
                dataset=dataset,
                sample_size=len(conversations),
                start_index=start_index,
                status="running"
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            run_id = run.id
        except Exception as e:
            db.rollback()
            db.close()
            logger.error(f"Failed to initialize run: {e}")
            raise e
        finally:
            db.close()

        async def _do_init():
            local_db = next(get_db())
            try:
                logs = []
                for idx, conv in enumerate(conversations, start=start_index):
                    if dataset == "inspired":
                        conv_id = f"{idx} {conv.get('conv_id', idx)}"
                        dialog = conv["processed_dialog"]
                        target = conv["target"]
                        import re
                        movie_mentions = re.findall(r'[A-Z][a-zA-Z\s]+(?:\([0-9]{4}\))?', dialog)
                        common_words = {'RECOMMENDER', 'SEEKER', 'Hi', 'There', 'What', 'types', 'movies', 'like', 'watch', 'Yes', 'No', 'Thanks', 'Thank', 'you'}
                        liked_movies = [m.strip() for m in movie_mentions if m.strip() not in common_words and len(m.strip()) > 3][:5]
                    else:  # redial
                        conv_id = str(idx)
                        dialog = conv["dialog"]
                        target = conv["target"]
                        liked_movies = conv.get("liked_movies", [])

                    log = ConversationLogModel(
                        run_id=run_id,
                        conv_id=conv_id,
                        index=idx,
                        status="pending",
                        dialog=dialog,
                        target=target,
                        liked_movies=liked_movies
                    )
                    logs.append(log)
                
                local_db.add_all(logs)
                
                run_obj = local_db.query(EvaluationRunModel).filter_by(id=run_id).first()
                if run_obj:
                    run_obj.status = "initialized"
                local_db.commit()
            except Exception as e:
                local_db.rollback()
                run_obj = local_db.query(EvaluationRunModel).filter_by(id=run_id).first()
                if run_obj:
                    run_obj.status = "failed"
                    local_db.commit()
                logger.error(f"Failed to initialize logs: {e}")
            finally:
                local_db.close()

        if background_tasks:
            background_tasks.add_task(_do_init)
            return {
                "run_id": run_id,
                "message": f"Initializing run {run_id} with {len(conversations)} conversations in background.",
                "status": "running"
            }
            
        await _do_init()
        return {
            "run_id": run_id,
            "message": f"Initialized run {run_id} with {len(conversations)} conversations.",
            "status": "initialized"
        }

    async def _create_step_run(self, db, model_cls, filters: dict, kwargs: dict):
        last_batch = db.query(model_cls).filter_by(**filters).order_by(model_cls.version.desc()).first()
        new_version = (last_batch.version + 1) if last_batch else 1
        
        batch = model_cls(
            version=new_version,
            status="running",
            **kwargs
        )
        db.add(batch)
        db.commit()
        db.refresh(batch)
        return batch

    async def get_step_history(self, run_id: int) -> List[Dict[str, Any]]:
        """Get execution history for a run. Retrieves from all 3 tables."""
        db = next(get_db())
        try:
            summs = db.query(SummarizationRunModel).filter_by(run_id=run_id).all()
            summ_ids = [s.id for s in summs]
            
            retrs = db.query(RetrievalRunModel).filter(RetrievalRunModel.summarization_batch_id.in_(summ_ids)).all() if summ_ids else []
            retr_ids = [r.id for r in retrs]
            
            reranks = db.query(RerankingRunModel).filter(RerankingRunModel.retrieval_batch_id.in_(retr_ids)).all() if retr_ids else []
            
            results = []
            for s in summs:
                results.append({"id": s.id, "run_id": s.run_id, "name": s.name, "step_type": "summarization", "version": s.version, "config": s.config, "status": s.status, "created_at": s.created_at})
            for r in retrs:
                # Resolve run_id
                summ = next((x for x in summs if x.id == r.summarization_batch_id), None)
                run_id = summ.run_id if summ else -1
                results.append({"id": r.id, "run_id": run_id, "name": r.name, "step_type": "retrieval", "version": r.version, "config": r.config, "status": r.status, "created_at": r.created_at, "input_batch_id": r.summarization_batch_id})
            for rr in reranks:
                retr = next((x for x in retrs if x.id == rr.retrieval_batch_id), None)
                summ = next((x for x in summs if x.id == retr.summarization_batch_id), None) if retr else None
                run_id = summ.run_id if summ else -1
                results.append({"id": rr.id, "run_id": run_id, "name": rr.name, "step_type": "reranking", "version": rr.version, "config": rr.config, "status": rr.status, "created_at": rr.created_at, "input_batch_id": rr.retrieval_batch_id})
                
            results.sort(key=lambda x: x["created_at"], reverse=True)
            return results
        finally:
            db.close()

    async def get_steps_by_type(self, step_type: str = None, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all batch step executions, optionally filtered by step_type."""
        db = next(get_db())
        try:
            results = []
            if step_type == "summarization":
                items = db.query(SummarizationRunModel).order_by(SummarizationRunModel.created_at.desc()).offset(skip).limit(limit).all()
                for i in items:
                    results.append({"id": i.id, "run_id": i.run_id, "name": i.name, "step_type": "summarization", "version": i.version, "config": i.config, "status": i.status, "created_at": i.created_at})
            elif step_type == "retrieval":
                items = db.query(RetrievalRunModel).order_by(RetrievalRunModel.created_at.desc()).offset(skip).limit(limit).all()
                for i in items:
                    summ = db.query(SummarizationRunModel).filter_by(id=i.summarization_batch_id).first()
                    run_id = summ.run_id if summ else -1
                    results.append({"id": i.id, "run_id": run_id, "name": i.name, "step_type": "retrieval", "version": i.version, "config": i.config, "status": i.status, "created_at": i.created_at, "input_batch_id": i.summarization_batch_id})
            elif step_type == "reranking":
                items = db.query(RerankingRunModel).order_by(RerankingRunModel.created_at.desc()).offset(skip).limit(limit).all()
                for i in items:
                    retr = db.query(RetrievalRunModel).filter_by(id=i.retrieval_batch_id).first()
                    summ = db.query(SummarizationRunModel).filter_by(id=retr.summarization_batch_id).first() if retr else None
                    run_id = summ.run_id if summ else -1
                    results.append({"id": i.id, "run_id": run_id, "name": i.name, "step_type": "reranking", "version": i.version, "config": i.config, "status": i.status, "created_at": i.created_at, "input_batch_id": i.retrieval_batch_id})
            return results
        finally:
            db.close()

    async def get_batch_detail(self, batch_id: int, step_type: str) -> dict:
        """Get per-conversation detail data for a specific batch execution."""
        db = next(get_db())
        try:
            items = []
            if step_type == "summarization":
                batch = db.query(SummarizationRunModel).filter_by(id=batch_id).first()
                if not batch: return None
                run_id = batch.run_id
                rows = (
                    db.query(StepSummarizationModel, ConversationLogModel.conv_id)
                    .join(ConversationLogModel, StepSummarizationModel.conversation_id == ConversationLogModel.id)
                    .filter(StepSummarizationModel.summarization_batch_id == batch_id)
                    .all()
                )
                for row, conv_id in rows:
                    items.append({
                        "conv_id": conv_id,
                        "user_preferences": row.user_preferences,
                    })

            elif step_type == "retrieval":
                batch = db.query(RetrievalRunModel).filter_by(id=batch_id).first()
                if not batch: return None
                summ = db.query(SummarizationRunModel).filter_by(id=batch.summarization_batch_id).first()
                run_id = summ.run_id if summ else -1
                rows = (
                    db.query(StepRetrievalModel, ConversationLogModel.conv_id)
                    .join(ConversationLogModel, StepRetrievalModel.conversation_id == ConversationLogModel.id)
                    .filter(StepRetrievalModel.retrieval_batch_id == batch_id)
                    .all()
                )
                for row, conv_id in rows:
                    candidates = row.candidates or []
                    items.append({
                        "conv_id": conv_id,
                        "candidates": candidates,
                        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
                        "semantic_count": row.semantic_count,
                        "content_count": row.content_count,
                        "collab_count": row.collab_count,
                    })

            elif step_type == "reranking":
                batch = db.query(RerankingRunModel).filter_by(id=batch_id).first()
                if not batch: return None
                retr = db.query(RetrievalRunModel).filter_by(id=batch.retrieval_batch_id).first()
                summ = db.query(SummarizationRunModel).filter_by(id=retr.summarization_batch_id).first() if retr else None
                run_id = summ.run_id if summ else -1
                rows = (
                    db.query(StepRerankingModel, ConversationLogModel.conv_id)
                    .join(ConversationLogModel, StepRerankingModel.conversation_id == ConversationLogModel.id)
                    .filter(StepRerankingModel.reranking_batch_id == batch_id)
                    .all()
                )
                for row, conv_id in rows:
                    reranked = row.reranked_candidates or []
                    items.append({
                        "conv_id": conv_id,
                        "model_used": row.model_used,
                        "reranked_candidates": reranked,
                        "reranked_count": len(reranked) if isinstance(reranked, list) else 0,
                        "recall": row.recall,
                    })
            else:
                return None

            return {
                "id": batch.id,
                "run_id": run_id,
                "name": batch.name,
                "step_type": step_type,
                "version": batch.version,
                "config": batch.config,
                "status": batch.status,
                "created_at": batch.created_at,
                "items": items,
            }
        finally:
            db.close()

    async def run_summarization_step(self, run_id: int, name: Optional[str] = None, background_tasks: Optional[BackgroundTasks] = None) -> Dict[str, Any]:
        """
        Run summarization step for a specific run (parallel execution).
        """
        import asyncio

        db = next(get_db())
        try:
            logs = db.query(ConversationLogModel).filter(
                ConversationLogModel.run_id == run_id
            ).all()

            if not logs:
                db.close()
                return {"message": "No logs found for this run.", "count": 0}

            # Create Batch
            batch = await self._create_step_run(
                db,
                SummarizationRunModel,
                {"run_id": run_id},
                {"run_id": run_id, "name": name, "config": {}}
            )
            batch_id = batch.id
            batch_version = batch.version
        except Exception as e:
            db.rollback()
            db.close()
            raise e
        finally:
            db.close()

        async def _do_summ():
            local_db = next(get_db())
            try:
                b = local_db.query(SummarizationRunModel).filter_by(id=batch_id).first()
                if not b: return
                
                local_logs = local_db.query(ConversationLogModel).filter(
                    ConversationLogModel.run_id == run_id
                ).all()

                CONCURRENCY = 10
                semaphore = asyncio.Semaphore(CONCURRENCY)

                async def _summarize_one(log: ConversationLogModel):
                    async with semaphore:
                        user_pref_obj = await self.generation_service.summarize_conversation(
                            str(log.dialog)
                        )
                        return log.id, user_pref_obj.user_preferences

                tasks = [_summarize_one(log) for log in local_logs]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                processed_count = 0
                for outcome in results:
                    if isinstance(outcome, Exception):
                        logger.error(f"Summarization task failed: {outcome}")
                        continue
                    log_id, user_preferences = outcome
                    step = StepSummarizationModel(
                        conversation_id=log_id,
                        summarization_batch_id=batch_id,
                        user_preferences=user_preferences
                    )
                    local_db.add(step)
                    log = local_db.query(ConversationLogModel).filter_by(id=log_id).first()
                    if log:
                        log.status = "summarized"
                    processed_count += 1

                b.status = "completed"
                run = local_db.query(EvaluationRunModel).filter_by(id=run_id).first()
                if run:
                    run.status = "summarized"

                local_db.commit()
            except Exception as e:
                local_db.rollback()
                b = local_db.query(SummarizationRunModel).filter_by(id=batch_id).first()
                if b:
                    b.status = "failed"
                    local_db.commit()
                logger.error(f"Summarization step bg task failed: {e}")
            finally:
                local_db.close()

        if background_tasks:
            background_tasks.add_task(_do_summ)
            return {
                "run_id": run_id,
                "summarization_batch_id": batch_id,
                "message": f"Summarization queued (v{batch_version}).",
                "status": "running"
            }
            
        await _do_summ()
        return {
            "run_id": run_id,
            "summarization_batch_id": batch_id,
            "message": f"Summarized conversations (v{batch_version}).",
            "status": "summarized"
        }


    async def run_retrieval_step(self, run_id: int, n_sample: int = 100, summarization_batch_id: Optional[int] = None, name: Optional[str] = None, background_tasks: Optional[BackgroundTasks] = None) -> Dict[str, Any]:
        """
        Run retrieval step. Can optionally specify which summarization batch to use as input.
        """
        db = next(get_db())
        try:
            logs = db.query(ConversationLogModel).filter(ConversationLogModel.run_id == run_id).all()
            if not logs:
                db.close()
                return {"message": "No logs found.", "count": 0}

            if not summarization_batch_id:
                latest_summ = db.query(SummarizationRunModel).filter_by(run_id=run_id, status="completed").order_by(SummarizationRunModel.version.desc()).first()
                if not latest_summ:
                    db.close()
                    return {"message": "No completed summarization step found to base retrieval on.", "count": 0}
                summarization_batch_id = latest_summ.id

            # Create Batch
            batch = await self._create_step_run(
                db, 
                RetrievalRunModel, 
                {"summarization_batch_id": summarization_batch_id}, 
                {"summarization_batch_id": summarization_batch_id, "name": name, "config": {"n_sample": n_sample}}
            )
            batch_id = batch.id
            batch_version = batch.version
        except Exception as e:
            db.rollback()
            db.close()
            raise e
        finally:
            db.close()

        async def _do_retr():
            local_db = next(get_db())
            try:
                b = local_db.query(RetrievalRunModel).filter_by(id=batch_id).first()
                if not b: return
                
                local_logs = local_db.query(ConversationLogModel).filter(
                    ConversationLogModel.run_id == run_id
                ).all()

                processed_count = 0
                for log in local_logs:
                    # 1. Get User Preferences
                    query = local_db.query(StepSummarizationModel).filter_by(conversation_id=log.id, summarization_batch_id=summarization_batch_id)
                    step_summ = query.first()
                    
                    if not step_summ:
                        continue # Skip if no summarization found for this conv

                    user_preferences = step_summ.user_preferences
                    liked_movies = log.liked_movies

                    # 2. Retrieve
                    retrieval_result = await self.retrieval_service.retrieve_movies(
                        user_preferences=user_preferences,
                        liked_movies=liked_movies,
                        n=n_sample
                    )
                    
                    candidates = retrieval_result["combined"]
                    counts = {
                        "semantic": len(retrieval_result.get("semantic", [])),
                        "content": len(retrieval_result.get("content", [])),
                        "collab": len(retrieval_result.get("collaborative", []))
                    }

                    # Upsert StepRetrieval linked to Batch
                    step = StepRetrievalModel(
                        conversation_id=log.id,
                        retrieval_batch_id=batch_id,
                        candidates=candidates, 
                        semantic_count=counts["semantic"],
                        content_count=counts["content"],
                        collab_count=counts["collab"]
                    )
                    local_db.add(step)
                    log.status = "retrieved"
                    processed_count += 1
                    
                b.status = "completed"
                run = local_db.query(EvaluationRunModel).filter_by(id=run_id).first()
                if run:
                    run.status = "retrieved"
                
                local_db.commit()
            except Exception as e:
                local_db.rollback()
                b = local_db.query(RetrievalRunModel).filter_by(id=batch_id).first()
                if b:
                    b.status = "failed"
                    local_db.commit()
                logger.error(f"Retrieval step failed: {e}")
            finally:
                local_db.close()

        if background_tasks:
            background_tasks.add_task(_do_retr)
            return {"run_id": run_id, "retrieval_batch_id": batch_id, "message": f"Retrieval queued (v{batch_version}).", "status": "running"}

        await _do_retr()
        return {"run_id": run_id, "retrieval_batch_id": batch_id, "message": f"Retrieved candidates (v{batch_version}).", "status": "retrieved"}

    async def run_reranking_step(self, run_id: int, top_k: int = 10, model: Literal["llm", "cohere"] = "llm", retrieval_batch_id: Optional[int] = None, name: Optional[str] = None, background_tasks: Optional[BackgroundTasks] = None) -> Dict[str, Any]:
        """
        Run reranking step for a specific run.
        Requires retrieval to be completed.
        """
        db = next(get_db())
        try:
            logs = db.query(ConversationLogModel).filter(
                ConversationLogModel.run_id == run_id
            ).all()

            if not logs:
                db.close()
                return {"message": "No logs found for this run.", "count": 0}

            if not retrieval_batch_id:
                latest_retr = db.query(RetrievalRunModel).join(SummarizationRunModel).filter(SummarizationRunModel.run_id == run_id, RetrievalRunModel.status == "completed").order_by(RetrievalRunModel.version.desc()).first()
                if not latest_retr:
                    db.close()
                    return {"message": "No completed retrieval step found.", "count": 0}
                retrieval_batch_id = latest_retr.id

            # Create Batch
            batch = await self._create_step_run(
                db, 
                RerankingRunModel, 
                {"retrieval_batch_id": retrieval_batch_id}, 
                {"retrieval_batch_id": retrieval_batch_id, "name": name, "config": {"top_k": top_k, "model": model}}
            )
            batch_id = batch.id
            batch_version = batch.version
        except Exception as e:
            db.rollback()
            db.close()
            raise e
        finally:
            db.close()

        async def _do_rerank():
            local_db = next(get_db())
            try:
                b = local_db.query(RerankingRunModel).filter_by(id=batch_id).first()
                if not b: return
                
                local_logs = local_db.query(ConversationLogModel).filter(
                    ConversationLogModel.run_id == run_id
                ).all()

                processed_count = 0
                for log in local_logs:
                    # 1. Get Retrieval Candidates
                    query = local_db.query(StepRetrievalModel).filter_by(conversation_id=log.id, retrieval_batch_id=retrieval_batch_id)
                    step_retrieval = query.first()

                    if not step_retrieval:
                        continue

                    candidates = step_retrieval.candidates

                    # 2. Get User Preferences
                    retrieval_run = local_db.query(RetrievalRunModel).filter_by(id=retrieval_batch_id).first()
                    summ_batch_id = retrieval_run.summarization_batch_id
                    
                    step_summ = local_db.query(StepSummarizationModel).filter_by(conversation_id=log.id, summarization_batch_id=summ_batch_id).first()
                    
                    if not step_summ:
                        continue

                    user_preferences = step_summ.user_preferences
                    context = log.dialog

                    # 3. Reranking
                    reranked = await self.reranking_service.rerank_movies(
                        user_preferences=user_preferences,
                        candidates=candidates,
                        conversation=str(context),
                        top_k=top_k,
                        model=model
                    )

                    get_titles = lambda movies: [m['title'] for m in movies]
                    final_recs = get_titles(reranked)
                    ground_truth = log.target if isinstance(log.target, list) else [log.target]
                    
                    recall_val = self.evaluation_domain_service.calculate_recall(final_recs, ground_truth, top_k)

                    # Upsert StepReranking
                    step = StepRerankingModel(
                        conversation_id=log.id,
                        reranking_batch_id=batch_id,
                        model_used=model,
                        reranked_candidates=reranked,
                        recall=recall_val
                    )
                    local_db.add(step)
                    
                    # Also compute and save immediate result for this conversation
                    result = local_db.query(EvaluationResultModel).filter_by(conversation_id=log.id).first()
                    if not result:
                        result = EvaluationResultModel(
                            run_id=run_id,
                            conversation_id=log.id,
                            conv_id=log.conv_id
                        )

                    result.recall = recall_val
                    result.ground_truth = ground_truth
                    result.recommendations = final_recs
                    
                    retrieved_candidates = step_retrieval.candidates
                    result.candidate_count = len(retrieved_candidates)
                    result.semantic_count = step_retrieval.semantic_count
                    result.content_count = step_retrieval.content_count
                    result.collab_count = step_retrieval.collab_count
                    
                    result.recall_retrieval = self.evaluation_domain_service.calculate_recall(get_titles(retrieved_candidates), ground_truth, 100)
                    
                    local_db.add(result)
                    processed_count += 1
                    log.status = "completed"
                
                b.status = "completed"
                run = local_db.query(EvaluationRunModel).filter_by(id=run_id).first()
                if run:
                    run.status = "completed"
                    results = local_db.query(EvaluationResultModel).filter_by(run_id=run.id).all()
                    if results:
                        run.avg_recall = sum(r.recall for r in results) / len(results)
                        run.avg_recall_retrieval = sum(r.recall_retrieval for r in results if r.recall_retrieval is not None) / len(results)

                local_db.commit()
            except Exception as e:
                local_db.rollback()
                b = local_db.query(RerankingRunModel).filter_by(id=batch_id).first()
                if b:
                    b.status = "failed"
                    local_db.commit()
                logger.error(f"Error in reranking step: {e}")
            finally:
                local_db.close()

        if background_tasks:
            background_tasks.add_task(_do_rerank)
            return {"run_id": run_id, "batch_id": batch_id, "message": f"Reranking queued (v{batch_version}).", "status": "running"}

        await _do_rerank()
        return {"run_id": run_id, "batch_id": batch_id, "message": f"Reranked evaluated (v{batch_version}).", "status": "completed"}

    def _load_all_dialogs(self, dataset: Literal["inspired", "redial"]) -> List[Dict]:
        """Load only the dialog data from disk (no movie metadata)."""
        if dataset == "inspired":
            dialog_path = Path(settings.data.INSPIRED_DIALOG_DATA)
        elif dataset == "redial":
            dialog_path = Path(settings.data.REDIAL_DIALOG_DATA)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        with open(dialog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_dialogs_sample(
        self,
        dataset: Literal["inspired", "redial"],
        sample_size: int,
    ) -> List[Dict]:
        """
        Load only dialog conversations (skips movie metadata).
        Used by initialize_run to avoid the slow movie-file I/O.
        """
        all_conversations = self._load_all_dialogs(dataset)
        if sample_size >= len(all_conversations):
            logger.warning(
                f"sample_size={sample_size} >= dataset size {len(all_conversations)}. Returning full dataset."
            )
            return all_conversations
        return random.sample(all_conversations, sample_size)

    def load_dataset_sample(
        self,
        dataset: Literal["inspired", "redial"],
        sample_size: int,
        start_index: int,
    ) -> tuple[List[Dict], pd.DataFrame]:
        """Load dialog samples together with the movie metadata DataFrame."""
        all_conversations = self._load_all_dialogs(dataset)

        if dataset == "inspired":
            movie_path = Path(settings.data.INSPIRED_MOVIE_DATA)
            with open(movie_path, "r", encoding="utf-8") as f:
                movie = [json.loads(line) for line in f if line.strip()]
            df_movie = pd.DataFrame(movie)
        else:  # redial
            movie_path = Path(settings.data.REDIAL_MOVIE_DATA)
            df_movie = pd.DataFrame(pd.read_csv(movie_path, encoding="utf-8"))

        if sample_size >= len(all_conversations):
            logger.warning(
                f"sample_size={sample_size} >= dataset size {len(all_conversations)}. Returning full dataset."
            )
            return all_conversations, df_movie

        return random.sample(all_conversations, sample_size), df_movie

    async def evaluate(
        self,
        dataset: Literal["inspired", "redial"] = "redial",
        sample_size: int = 40,
        start_index: int = 0,
        n_sample: int = 100,
        top_k: int = 10,
        model: Literal["llm", "cohere"] = "cohere"
    ) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline on a dataset sample.
        """
        conversations, df_movie = self.load_dataset_sample(
            dataset, sample_size, start_index)

    async def process_single_conversation(
        self,
        conv: Dict,
        index: int,
        dataset: str,
        n_sample: int,
        top_k: int,
        model: str
    ) -> Dict[str, Any]:
        """Process a single conversation for evaluation."""
        try:
            # 0. Preprocessing / Extraction
            if dataset == "inspired":
                conv_id = f"{index} {conv.get('conv_id', index)}"
                context = conv["processed_dialog"]
                target = conv["target"]

                # Extract liked movies using regex (specific to INSPIRED)
                import re
                movie_mentions = re.findall(
                    r'[A-Z][a-zA-Z\s]+(?:\([0-9]{4}\))?', context)
                common_words = {'RECOMMENDER', 'SEEKER', 'Hi', 'There', 'What', 'types',
                                'movies', 'like', 'watch', 'Yes', 'No', 'Thanks', 'Thank', 'you'}
                liked_movies = [m.strip() for m in movie_mentions if m.strip(
                ) not in common_words and len(m.strip()) > 3][:5]
            else:  # redial
                conv_id = index
                context = conv["dialog"]
                target = conv["target"]
                liked_movies = conv.get("liked_movies", [])
            
            logger.info(f"Processing conversation {conv_id} | Found {len(liked_movies)} liked movies: {liked_movies}")


            # 1. Summarization (Domain: Generation)
            # Note: We rely on the generic summarization domain service
            user_pref_obj = await self.generation_service.summarize_conversation(context)
            user_preferences = user_pref_obj.user_preferences

            # 2. Retrieval (Domain: Retrieval)
            # Returns dict with keys: 'combined', 'semantic', 'content', 'collaborative'
            retrieval_results = await self.retrieval_service.retrieve_movies(
                user_preferences=user_preferences,
                liked_movies=liked_movies,
                n=n_sample
            )
            
            candidates = retrieval_results['combined']
            semantic_cands = retrieval_results['semantic']
            content_cands = retrieval_results['content']
            collab_cands = retrieval_results['collaborative']

            # 3. Reranking (Domain: Reranking)
            reranked = await self.reranking_service.rerank_movies(
                user_preferences=user_preferences,
                candidates=candidates,
                conversation=context,
                top_k=top_k,
                model=model
            )

            # 4. Evaluation (Domain: Evaluation)
            ground_truth = [target] if isinstance(target, str) else target
            
            # Helper to extract titles
            get_titles = lambda movies: [m['title'] for m in movies]
            
            # Post-Rerank Recall
            final_recs = get_titles(reranked)
            recall_final = self.evaluation_domain_service.calculate_recall(
                recommendations=final_recs,
                ground_truth=ground_truth,
                top_k=top_k
            )

            # Pre-Rerank Metrics (Recall@N)
            recall_combined_retrieval = self.evaluation_domain_service.calculate_recall(
                recommendations=get_titles(candidates),
                ground_truth=ground_truth,
                top_k=n_sample
            )
            
            recall_semantic = self.evaluation_domain_service.calculate_recall(
                recommendations=get_titles(semantic_cands),
                ground_truth=ground_truth,
                top_k=n_sample
            )
            
            recall_content = self.evaluation_domain_service.calculate_recall(
                recommendations=get_titles(content_cands),
                ground_truth=ground_truth,
                top_k=n_sample
            )
            
            recall_collab = self.evaluation_domain_service.calculate_recall(
                recommendations=get_titles(collab_cands),
                ground_truth=ground_truth,
                top_k=n_sample
            )

            return {
                "conv_id": conv_id,
                "ground_truth": ground_truth,
                # Metrics
                "recall_final": recall_final, 
                "recall_retrieval": recall_combined_retrieval, 
                "recall_semantic": recall_semantic,
                "recall_content": recall_content,
                "recall_collab": recall_collab,
                # Data
                "recommendations": final_recs,
                "candidate_count": len(candidates),
                "semantic_count": len(semantic_cands),
                "content_count": len(content_cands),
                "collab_count": len(collab_cands)
            }

        except Exception as e:
            logger.error(f"Error processing {index}: {e}")
            return {
                "conv_id": index,
                "error": str(e)
            }

    async def evaluate(
        self,
        dataset: Literal["inspired", "redial"] = "redial",
        sample_size: int = 40,
        start_index: int = 0,
        n_sample: int = 100,
        top_k: int = 10,
        model: Literal["llm", "cohere"] = "cohere"
    ) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline on a dataset sample.
        """
        import asyncio
        
        conversations, df_movie = self.load_dataset_sample(
            dataset, sample_size, start_index)

        # Batch processing configuration
        CONCURRENCY_LIMIT = 10
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def sem_task(conv, idx):
            async with semaphore:
                return await self.process_single_conversation(
                    conv, idx, dataset, n_sample, top_k, model
                )

        tasks = [
            sem_task(conv, index) 
            for index, conv in enumerate(conversations, start=start_index)
        ]
        
        # Run all tasks
        results = await asyncio.gather(*tasks)

        # Collect Recalls excluding errors
        recalls = [
            res["recall_final"] 
            for res in results 
            if "recall_final" in res and "error" not in res
        ]

        # Calculate averages
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        
        # Calculate averages for detailed metrics
        def safe_avg(key):
            valid_entries = [r[key] for r in results if key in r]
            return sum(valid_entries) / len(valid_entries) if valid_entries else 0.0

        avg_recall_retrieval = safe_avg("recall_retrieval")
        avg_recall_semantic = safe_avg("recall_semantic")
        avg_recall_content = safe_avg("recall_content")
        avg_recall_collab = safe_avg("recall_collab")

        result_data = {
            "dataset": dataset,
            "sample_size": len(conversations),
            "start_index": start_index,
            "n_sample": n_sample,
            "top_k": top_k,
            "model": model,
            # Placeholder or actual output dir
            "output_dir": str(self.project_root / "output"),
            "avg_recall": avg_recall,
            "avg_recall_retrieval": avg_recall_retrieval,
            "avg_recall_semantic": avg_recall_semantic,
            "avg_recall_content": avg_recall_content,
            "avg_recall_collab": avg_recall_collab,
            "results": results,
            "message": f"Evaluated {len(conversations)} samples. Avg Recall@{top_k}: {avg_recall:.4f}"
        }
        
        # Save to database
        try:
            self.evaluation_storage_service.save_run(result_data)
        except Exception as e:
            print(f"Failed to save evaluation results to DB: {e}")
            
        return result_data
