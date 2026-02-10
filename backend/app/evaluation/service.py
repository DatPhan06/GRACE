"""Evaluation application service."""
import json
import json
import pandas as pd
from pathlib import Path
from typing import Literal, Dict, Any, List
import random

# Domain services
from domain.generation.service import GenerationService
from domain.retrieval.service import RetrievalService
from domain.reranking.service import RerankingService
from domain.evaluation import EvaluationDomainService
from domain.evaluation_storage.service import EvaluationStorageService
from shared.settings.config import settings
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)


class EvaluationService:
    """
    Orchestrates the evaluation process by coordinating domain services.
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

    def load_dataset_sample(
        self,
        dataset: Literal["inspired", "redial"],
        sample_size: int,
        start_index: int
    ) -> tuple[List[Dict], pd.DataFrame]:
        """Load dataset samples."""
        if dataset == "inspired":
            dialog_path = Path(settings.data.INSPIRED_DIALOG_DATA)
            movie_path = Path(settings.data.INSPIRED_MOVIE_DATA)

            with open(dialog_path, "r", encoding="utf-8") as file:
                all_conversations = json.load(file)

            with open(movie_path, "r", encoding="utf-8") as file:
                movie = [json.loads(line) for line in file if line.strip()]
            df_movie = pd.DataFrame(movie)

        elif dataset == "redial":
            dialog_path = Path(settings.data.REDIAL_DIALOG_DATA)
            movie_path = Path(settings.data.REDIAL_MOVIE_DATA)

            with open(dialog_path, "r", encoding="utf-8") as file:
                all_conversations = json.load(file)

            movie = pd.read_csv(movie_path, encoding="utf-8")
            df_movie = pd.DataFrame(movie)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Sampling
        # Random Sampling
        if sample_size > len(all_conversations):
            logger.warning(f"Sample size {sample_size} is larger than dataset size {len(all_conversations)}. Returning full dataset.")
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
