"""Evaluation domain service."""
import re
from typing import List, Dict, Any, Union

class EvaluationDomainService:
    """
    Domain service for calculating evaluation metrics.
    """
    
    def standardize_movie_title(self, movie_title: str) -> str:
        """Normalizes movie titles for consistent comparison."""
        # Remove year and special characters, convert to lowercase
        title = re.sub(r"\s*\(\d{4}\)", "", movie_title)  # Remove year anywhere in title
        title = re.sub(r"[^\w\s]", "", title)  # Remove punctuation
        return title.strip().lower()  # Normalize case and whitespace

    def calculate_recall(
        self, 
        recommendations: List[str], 
        ground_truth: List[str],
        top_k: int = 50
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            recommendations: List of recommended movie titles.
            ground_truth: List of relevant movie titles (target).
            top_k: Cutoff for recommendations.
            
        Returns:
            Recall score (0.0 to 1.0).
        """
        if not ground_truth:
            return 0.0
            
        # Standardize titles
        ranked_movies = [self.standardize_movie_title(m) for m in recommendations[:top_k] if m.strip()]
        normalized_truth = [self.standardize_movie_title(m) for m in ground_truth if m.strip()]
        
        if not normalized_truth:
            return 0.0

        matches = sum(1 for m in normalized_truth if m in ranked_movies)
        recall = matches / len(normalized_truth)
        
        return recall
