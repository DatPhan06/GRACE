from typing import Dict, Any
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
                avg_recall=run_data["avg_recall"]
            )
            
            # Add results as children
            for res in run_data.get("results", []):
                result_model = EvaluationResultModel(
                    conv_id=str(res.get("conv_id")),
                    recall=res.get("recall", 0.0),
                    ground_truth=res.get("ground_truth"),
                    recommendations=res.get("recommendations"),
                    candidate_count=res.get("candidate_count"),
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
