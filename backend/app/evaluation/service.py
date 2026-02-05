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
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline on a dataset sample.
        """
        conversations, df_movie = self.load_dataset_sample(
            dataset, sample_size, start_index)

        results = []
        recalls = []

        for index, conv in enumerate(conversations, start=start_index):
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

                # 1. Summarization (Domain: Generation)
                # Note: We rely on the generic summarization domain service
                user_pref_obj = await self.generation_service.summarize_conversation(context)
                user_preferences = user_pref_obj.user_preferences

                # 2. Retrieval (Domain: Retrieval)
                candidates = await self.retrieval_service.retrieve_movies(
                    user_preferences=user_preferences,
                    liked_movies=liked_movies,
                    n=n_sample
                )

                # 3. Reranking (Domain: Reranking)
                # Explicitly use 'llm' model as requested for OpenAI usage
                reranked = await self.reranking_service.rerank_movies(
                    user_preferences=user_preferences,
                    candidates=candidates,
                    conversation=context,
                    top_k=top_k,
                    model="cohere"
                )

                # 4. Evaluation (Domain: Evaluation)
                ground_truth = [target] if isinstance(target, str) else target
                recommendations = [m['title'] for m in reranked]

                recall = self.evaluation_domain_service.calculate_recall(
                    recommendations=recommendations,
                    ground_truth=ground_truth,
                    top_k=top_k
                )

                recalls.append(recall)
                results.append({
                    "conv_id": conv_id,
                    "recall": recall,
                    "ground_truth": ground_truth,
                    "recommendations": recommendations,
                    "candidate_count": len(candidates)
                })

            except Exception as e:
                # print(f"Error processing {index}: {e}")
                results.append({
                    "conv_id": index,
                    "error": str(e)
                })

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

        result_data = {
            "dataset": dataset,
            "sample_size": len(conversations),
            "start_index": start_index,
            "n_sample": n_sample,
            "top_k": top_k,
            # Hardcoded for now as we force "llm" model which implies OpenAI in default setup
            "model": "openai",
            # Placeholder or actual output dir
            "output_dir": str(self.project_root / "output"),
            "avg_recall": avg_recall,
            "results": results,
            "message": f"Evaluated {len(conversations)} samples. Avg Recall@{top_k}: {avg_recall:.4f}"
        }
        
        # Save to database
        try:
            self.evaluation_storage_service.save_run(result_data)
        except Exception as e:
            print(f"Failed to save evaluation results to DB: {e}")
            
        return result_data
