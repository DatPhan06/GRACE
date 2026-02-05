from typing import Dict, Any, List, Optional
from infra.db.database import SessionLocal
from infra.db.crud import CRUDBase
from infra.db.models import EvaluationRunModel, EvaluationResultModel
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

class EvaluationStorageService:
    """
    Domain service for persisting evaluation data using Generic CRUD.
    """
    
    def __init__(self):
        self.run_crud = CRUDBase(EvaluationRunModel)
        
    def get_runs(self, skip: int = 0, limit: int = 100) -> List[EvaluationRunModel]:
        """Get list of evaluation runs."""
        db = SessionLocal()
        try:
            return self.run_crud.get_multi(db, skip=skip, limit=limit)
        finally:
            db.close()

    def get_run(self, run_id: int) -> Optional[EvaluationRunModel]:
        """Get a single evaluation run by ID with details."""
        from sqlalchemy.orm import joinedload
        db = SessionLocal()
        try:
            return db.query(EvaluationRunModel).options(joinedload(EvaluationRunModel.results)).filter(EvaluationRunModel.id == run_id).first()
        finally:
            db.close()
        
    def save_run(self, run_data: Dict[str, Any]) -> None:
        """
        Save evaluation run and its results.
        
        Args:
            run_data: Dictionary containing run metadata and list of results.
        """
        db = SessionLocal()
        try:
            # Prepare data implementation details
                # Create run model instance
            run_model = EvaluationRunModel(
                dataset=run_data["dataset"],
                sample_size=run_data["sample_size"],
                start_index=run_data["start_index"],
                n_sample=run_data["n_sample"],
                top_k=run_data["top_k"],
                model=run_data["model"],
                avg_recall=round(run_data["avg_recall"], 3),
                avg_recall_retrieval=round(run_data["avg_recall_retrieval"], 3) if run_data.get("avg_recall_retrieval") is not None else None,
                avg_recall_semantic=round(run_data["avg_recall_semantic"], 3) if run_data.get("avg_recall_semantic") is not None else None,
                avg_recall_content=round(run_data["avg_recall_content"], 3) if run_data.get("avg_recall_content") is not None else None,
                avg_recall_collab=round(run_data["avg_recall_collab"], 3) if run_data.get("avg_recall_collab") is not None else None
            )
            
            # Add results as children
            for res in run_data.get("results", []):
                recall = res.get("recall_final") if res.get("recall_final") is not None else res.get("recall", 0.0)
                
                result_model = EvaluationResultModel(
                    conv_id=str(res.get("conv_id")),
                    recall=round(recall, 3), 
                    ground_truth=res.get("ground_truth"),
                    recommendations=res.get("recommendations"),
                    candidate_count=res.get("candidate_count"),
                    
                    # Detailed Metrics
                    recall_retrieval=round(res["recall_retrieval"], 3) if res.get("recall_retrieval") is not None else None,
                    recall_semantic=round(res["recall_semantic"], 3) if res.get("recall_semantic") is not None else None,
                    recall_content=round(res["recall_content"], 3) if res.get("recall_content") is not None else None,
                    recall_collab=round(res["recall_collab"], 3) if res.get("recall_collab") is not None else None,
                    
                    # Counts
                    semantic_count=res.get("semantic_count"),
                    content_count=res.get("content_count"),
                    collab_count=res.get("collab_count"),
                    
                    error=res.get("error")
                )
                run_model.results.append(result_model)
            
            # Use Generic CRUD to Create
            self.run_crud.create(db, run_model)
            logger.info(f"Successfully saved evaluation run {run_model.id} with dataset {run_data.get('dataset')}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation run: {e}")
            db.rollback()
            raise e
        finally:
            db.close()
