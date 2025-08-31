# Hybrid Scoring System for Movie Reranking
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional
from scipy.stats import rankdata
import json
import re

class HybridScorer:
    """
    Combines cosine similarity scores with LLM-based scoring for optimal reranking.
    """
    
    def __init__(
        self,
        cosine_weight: float = 0.3,
        llm_weight: float = 0.7,
        diversity_weight: float = 0.0
    ):
        """
        Initialize hybrid scorer with configurable weights.
        
        Args:
            cosine_weight: Weight for cosine similarity scores (0-1)
            llm_weight: Weight for LLM scores (0-1) 
            diversity_weight: Weight for diversity bonus (0-1)
        """
        # Normalize weights
        total_weight = cosine_weight + llm_weight + diversity_weight
        self.cosine_weight = cosine_weight / total_weight
        self.llm_weight = llm_weight / total_weight
        self.diversity_weight = diversity_weight / total_weight
        
        logging.info(f"Hybrid scorer initialized with weights - Cosine: {self.cosine_weight:.2f}, "
                    f"LLM: {self.llm_weight:.2f}, Diversity: {self.diversity_weight:.2f}")
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range using min-max normalization.
        """
        if not scores or all(s == scores[0] for s in scores):
            return [0.5] * len(scores)  # All equal scores
        
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score
        
        return [(s - min_score) / range_score for s in scores]
    
    def rank_to_score(self, rankings: List[int], reverse: bool = True) -> List[float]:
        """
        Convert rankings to scores. Lower rank = higher score if reverse=True.
        
        Args:
            rankings: List of rank positions (1-based)
            reverse: If True, rank 1 gets highest score
        """
        if not rankings:
            return []
        
        max_rank = max(rankings)
        if reverse:
            scores = [(max_rank - rank + 1) / max_rank for rank in rankings]
        else:
            scores = [rank / max_rank for rank in rankings]
        
        return scores
    
    def compute_diversity_scores(
        self,
        movie_titles: List[str],
        movie_data_path: str
    ) -> List[float]:
        """
        Compute diversity scores based on genre and other features.
        Promotes variety in recommendations.
        """
        
        # Load movie database
        movie_db = {}
        try:
            with open(movie_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        movie = json.loads(line)
                        title = movie.get("title")
                        if title:
                            movie_db[title.lower().strip()] = movie
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            logging.warning(f"Movie data file not found: {movie_data_path}")
            return [0.5] * len(movie_titles)  # Neutral diversity
        
        # Extract features for diversity calculation
        genres_seen = set()
        decades_seen = set()
        directors_seen = set()
        
        diversity_scores = []
        
        for title in movie_titles:
            lookup_title = re.sub(r'\s*\(\d{4}\)$', '', title).strip()
            movie = movie_db.get(lookup_title.lower())
            
            score = 0.0
            
            if movie:
                # Genre diversity
                genre = movie.get("genre", "")
                if genre:
                    primary_genre = genre.split(",")[0].strip().lower()
                    if primary_genre not in genres_seen:
                        score += 0.4
                        genres_seen.add(primary_genre)
                
                # Decade diversity
                year = movie.get("year")
                if year:
                    try:
                        decade = (int(year) // 10) * 10
                        if decade not in decades_seen:
                            score += 0.3
                            decades_seen.add(decade)
                    except (ValueError, TypeError):
                        pass
                
                # Director diversity
                director = movie.get("director", "")
                if director and director not in directors_seen:
                    score += 0.3
                    directors_seen.add(director)
            
            diversity_scores.append(score)
        
        # Normalize diversity scores
        return self.normalize_scores(diversity_scores)
    
    def combine_scores(
        self,
        movie_titles: List[str],
        cosine_scores: List[float],
        llm_rankings: List[str],
        movie_data_path: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Combine different scoring methods into final hybrid scores.
        
        Args:
            movie_titles: Original list of movie titles
            cosine_scores: Cosine similarity scores
            llm_rankings: LLM-ranked movie titles (in order of preference)
            movie_data_path: Path to movie metadata for diversity scoring
        
        Returns:
            List of (movie_title, hybrid_score) tuples, sorted by score descending
        """
        
        if len(movie_titles) != len(cosine_scores):
            logging.error("Length mismatch between movie titles and cosine scores")
            return [(title, 0.0) for title in movie_titles]
        
        # Normalize cosine scores
        normalized_cosine = self.normalize_scores(cosine_scores)
        
        # Convert LLM rankings to scores
        llm_scores = [0.0] * len(movie_titles)
        for i, ranked_movie in enumerate(llm_rankings):
            if ranked_movie in movie_titles:
                movie_idx = movie_titles.index(ranked_movie)
                # Higher rank (lower index) gets higher score
                llm_scores[movie_idx] = (len(llm_rankings) - i) / len(llm_rankings)
        
        # Compute diversity scores if movie data available
        diversity_scores = [0.0] * len(movie_titles)
        if self.diversity_weight > 0 and movie_data_path:
            diversity_scores = self.compute_diversity_scores(movie_titles, movie_data_path)
        
        # Combine scores
        hybrid_scores = []
        for i, title in enumerate(movie_titles):
            combined_score = (
                self.cosine_weight * normalized_cosine[i] +
                self.llm_weight * llm_scores[i] +
                self.diversity_weight * diversity_scores[i]
            )
            hybrid_scores.append((title, combined_score))
        
        # Sort by combined score descending
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        logging.info(f"Hybrid scoring complete. Top 5 scores: "
                    f"{[(title, f'{score:.3f}') for title, score in hybrid_scores[:5]]}")
        
        return hybrid_scores
    
    def adaptive_weight_adjustment(
        self,
        cosine_scores: List[float],
        llm_confidence: float = 0.5
    ) -> None:
        """
        Adaptively adjust weights based on score distributions and LLM confidence.
        
        Args:
            cosine_scores: Distribution of cosine similarity scores
            llm_confidence: Estimated confidence of LLM rankings (0-1)
        """
        
        # Analyze cosine score distribution
        if cosine_scores:
            cosine_std = np.std(cosine_scores)
            cosine_range = max(cosine_scores) - min(cosine_scores)
            
            # If cosine scores have low variance, rely more on LLM
            if cosine_std < 0.1 or cosine_range < 0.2:
                self.cosine_weight *= 0.7
                self.llm_weight *= 1.3
                logging.info("Low cosine variance detected, increasing LLM weight")
            
            # If LLM confidence is low, rely more on cosine
            if llm_confidence < 0.3:
                self.cosine_weight *= 1.2
                self.llm_weight *= 0.8
                logging.info("Low LLM confidence, increasing cosine weight")
        
        # Renormalize weights
        total = self.cosine_weight + self.llm_weight + self.diversity_weight
        self.cosine_weight /= total
        self.llm_weight /= total
        self.diversity_weight /= total

class SmartBatchSelector:
    """
    Intelligently selects which movies to send to LLM for reranking.
    """
    
    def __init__(self, score_threshold: float = 0.6, min_batch_size: int = 20):
        self.score_threshold = score_threshold
        self.min_batch_size = min_batch_size
    
    def select_candidates_for_llm(
        self,
        movie_titles: List[str],
        cosine_scores: List[float],
        target_size: int = 100
    ) -> Tuple[List[str], List[float]]:
        """
        Intelligently select movies for LLM reranking based on cosine scores.
        
        Args:
            movie_titles: All movie titles
            cosine_scores: Corresponding cosine similarity scores
            target_size: Target number of movies for LLM reranking
        
        Returns:
            Tuple of (selected_movies, selected_scores)
        """
        
        if len(movie_titles) <= target_size:
            return movie_titles, cosine_scores
        
        # Create movie-score pairs
        movie_score_pairs = list(zip(movie_titles, cosine_scores))
        
        # Sort by cosine score descending
        movie_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Strategy 1: Take top performers by cosine similarity
        top_candidates = movie_score_pairs[:target_size]
        
        # Strategy 2: Ensure minimum score threshold is met
        if cosine_scores:
            mean_score = np.mean(cosine_scores)
            threshold = max(self.score_threshold, mean_score * 0.8)
            
            filtered_candidates = [
                (movie, score) for movie, score in movie_score_pairs 
                if score >= threshold
            ]
            
            if len(filtered_candidates) >= self.min_batch_size:
                # Use filtered candidates up to target size
                top_candidates = filtered_candidates[:target_size]
        
        selected_movies, selected_scores = zip(*top_candidates)
        
        logging.info(f"Selected {len(selected_movies)} movies for LLM reranking "
                    f"from {len(movie_titles)} total. Score range: "
                    f"{min(selected_scores):.3f} - {max(selected_scores):.3f}")
        
        return list(selected_movies), list(selected_scores)

def compute_recommendation_confidence(
    cosine_scores: List[float],
    llm_rankings: List[str],
    movie_titles: List[str]
) -> float:
    """
    Compute confidence score for the recommendation set.
    
    Returns:
        Confidence score between 0 and 1
    """
    
    confidence_factors = []
    
    # Factor 1: Cosine score distribution
    if cosine_scores:
        # Higher top scores indicate better matches
        top_scores = sorted(cosine_scores, reverse=True)[:10]
        avg_top_score = np.mean(top_scores)
        confidence_factors.append(min(avg_top_score * 2, 1.0))
        
        # Lower variance in top scores indicates consistent quality
        if len(top_scores) > 1:
            score_consistency = 1 - (np.std(top_scores) / np.mean(top_scores))
            confidence_factors.append(max(score_consistency, 0.0))
    
    # Factor 2: LLM ranking completeness
    if llm_rankings and movie_titles:
        coverage = len(set(llm_rankings) & set(movie_titles)) / len(movie_titles)
        confidence_factors.append(coverage)
    
    # Average all factors
    overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
    
    return min(max(overall_confidence, 0.0), 1.0)
