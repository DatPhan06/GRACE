# Optimized Reranking System - Integration of All Solutions
import logging
import time
from typing import List, Dict, Any, Tuple, Optional

# Import our optimization modules
from .hierarchical_reranking import enhanced_hierarchical_reranking
from .enhanced_movie_context import prepare_enhanced_movie_batches, get_movie_embedding_features
from .hybrid_scoring import HybridScorer, SmartBatchSelector, compute_recommendation_confidence
from .semantic_filtering import apply_smart_filtering

def optimized_movie_reranking(
    movie_list: List[str],
    user_preferences: str,
    context: str,
    movie_data_path: str,
    model: str,
    api_keys: List[str],
    cosine_scores: Optional[List[float]] = None,
    final_k: int = 50,
    use_semantic_filtering: bool = True,
    use_hierarchical_reranking: bool = True,
    use_hybrid_scoring: bool = True,
    max_llm_candidates: int = 150,
    batch_size: int = 25,
    candidates_per_batch: int = 5
) -> Dict[str, Any]:
    """
    Optimized movie reranking pipeline that addresses the issues with naive LLM reranking.
    
    Pipeline:
    1. Semantic Filtering: Remove obviously mismatched movies
    2. Smart Candidate Selection: Choose best candidates for LLM processing  
    3. Hierarchical Reranking: Process in manageable batches
    4. Hybrid Scoring: Combine LLM results with cosine similarity
    5. Final Ranking: Return optimized recommendations
    
    Args:
        movie_list: List of candidate movie titles (from vector search)
        user_preferences: Summarized user preferences
        context: Original conversation context
        movie_data_path: Path to movie metadata JSONL file
        model: LLM model name
        api_keys: List of API keys for LLM calls
        cosine_scores: Optional cosine similarity scores corresponding to movie_list
        final_k: Number of final recommendations to return
        use_semantic_filtering: Whether to apply semantic filtering
        use_hierarchical_reranking: Whether to use hierarchical instead of flat reranking
        use_hybrid_scoring: Whether to combine LLM + cosine scores
        max_llm_candidates: Maximum movies to send to LLM for reranking
        batch_size: Size of each batch for hierarchical reranking
        candidates_per_batch: Top movies to select from each batch
        
    Returns:
        Dictionary containing:
        - movie_list: Final ranked movie recommendations
        - confidence: Confidence score for the recommendations
        - method_used: Description of methods applied
        - statistics: Processing statistics
    """
    
    start_time = time.time()
    statistics = {
        "input_movies": len(movie_list),
        "after_semantic_filtering": 0,
        "sent_to_llm": 0,
        "final_output": 0,
        "processing_time": 0,
        "methods_used": []
    }
    
    logging.info(f"Starting optimized reranking for {len(movie_list)} movies")
    
    # Validate inputs
    if not movie_list:
        return {
            "movie_list": [],
            "confidence": 0.0,
            "method_used": "No movies to rank",
            "statistics": statistics
        }
    
    current_movies = movie_list.copy()
    current_scores = cosine_scores.copy() if cosine_scores else [0.5] * len(movie_list)
    
    # Step 1: Semantic Filtering
    if use_semantic_filtering and len(current_movies) > max_llm_candidates:
        logging.info("Step 1: Applying semantic filtering")
        current_movies, current_scores = apply_smart_filtering(
            current_movies,
            user_preferences,
            movie_data_path,
            current_scores,
            max_candidates=max_llm_candidates
        )
        statistics["after_semantic_filtering"] = len(current_movies)
        statistics["methods_used"].append("semantic_filtering")
        logging.info(f"After semantic filtering: {len(current_movies)} movies")
    else:
        statistics["after_semantic_filtering"] = len(current_movies)
    
    # Step 2: Smart Candidate Selection (if still too many)
    if len(current_movies) > max_llm_candidates:
        logging.info("Step 2: Applying smart candidate selection")
        batch_selector = SmartBatchSelector()
        current_movies, current_scores = batch_selector.select_candidates_for_llm(
            current_movies, current_scores, max_llm_candidates
        )
        statistics["methods_used"].append("smart_candidate_selection")
        logging.info(f"After candidate selection: {len(current_movies)} movies")
    
    statistics["sent_to_llm"] = len(current_movies)
    
    # Step 3: Reranking Strategy
    llm_rankings = []
    
    if len(current_movies) <= 50:
        # Small set: Use direct reranking
        logging.info("Step 3: Using direct LLM reranking (small set)")
        try:
            from .GenerativeAI_redial import callLangChainLLMReranking_redial
            result = callLangChainLLMReranking_redial(
                context=context,
                user_preferences=user_preferences,
                movie_str="|".join(current_movies),
                model=model,
                api_key=api_keys,
                k=min(final_k, len(current_movies))
            )
            if isinstance(result, dict) and "movie_list" in result:
                llm_rankings = result["movie_list"]
            statistics["methods_used"].append("direct_llm_reranking")
        except Exception as e:
            logging.error(f"Direct LLM reranking failed: {e}")
            llm_rankings = current_movies[:final_k]
    
    elif use_hierarchical_reranking:
        # Large set: Use hierarchical reranking
        logging.info("Step 3: Using hierarchical LLM reranking")
        try:
            result = enhanced_hierarchical_reranking(
                movie_list=current_movies,
                user_preferences=user_preferences,
                context=context,
                model=model,
                api_keys=api_keys,
                cosine_scores=current_scores,
                final_k=final_k,
                batch_size=batch_size,
                candidates_per_batch=candidates_per_batch
            )
            if isinstance(result, dict) and "movie_list" in result:
                llm_rankings = result["movie_list"]
            statistics["methods_used"].append("hierarchical_reranking")
        except Exception as e:
            logging.error(f"Hierarchical reranking failed: {e}")
            # Fallback to score-based ranking
            movie_score_pairs = list(zip(current_movies, current_scores))
            movie_score_pairs.sort(key=lambda x: x[1], reverse=True)
            llm_rankings = [movie for movie, _ in movie_score_pairs[:final_k]]
            statistics["methods_used"].append("fallback_cosine_ranking")
    else:
        # Fallback: Use cosine similarity ranking
        logging.info("Step 3: Using cosine similarity ranking (no LLM)")
        movie_score_pairs = list(zip(current_movies, current_scores))
        movie_score_pairs.sort(key=lambda x: x[1], reverse=True)
        llm_rankings = [movie for movie, _ in movie_score_pairs[:final_k]]
        statistics["methods_used"].append("cosine_only_ranking")
    
    # Step 4: Hybrid Scoring (if enabled and we have both LLM and cosine scores)
    final_movies = llm_rankings
    
    if use_hybrid_scoring and cosine_scores and llm_rankings and len(llm_rankings) > 5:
        logging.info("Step 4: Applying hybrid scoring")
        try:
            # Initialize hybrid scorer with balanced weights
            hybrid_scorer = HybridScorer(
                cosine_weight=0.4,  # Cosine similarity 
                llm_weight=0.6,     # LLM preference
                diversity_weight=0.0  # Diversity bonus (disabled for now)
            )
            
            # Compute confidence and adjust weights
            confidence = compute_recommendation_confidence(
                current_scores, llm_rankings, current_movies
            )
            hybrid_scorer.adaptive_weight_adjustment(current_scores, confidence)
            
            # Get hybrid scores
            hybrid_results = hybrid_scorer.combine_scores(
                current_movies, current_scores, llm_rankings, movie_data_path
            )
            
            # Extract final ranking
            final_movies = [movie for movie, _ in hybrid_results[:final_k]]
            statistics["methods_used"].append("hybrid_scoring")
            
        except Exception as e:
            logging.error(f"Hybrid scoring failed: {e}")
            # Keep LLM rankings as fallback
            pass
    
    # Ensure we don't exceed requested number
    final_movies = final_movies[:final_k]
    statistics["final_output"] = len(final_movies)
    statistics["processing_time"] = time.time() - start_time
    
    # Compute final confidence score
    confidence = compute_recommendation_confidence(
        current_scores, final_movies, current_movies
    )
    
    method_description = " + ".join(statistics["methods_used"])
    
    logging.info(f"Optimized reranking complete: {len(movie_list)} -> {len(final_movies)} movies "
                f"in {statistics['processing_time']:.2f}s using {method_description}")
    
    return {
        "movie_list": final_movies,
        "confidence": confidence,
        "method_used": method_description,
        "statistics": statistics
    }

def quick_reranking_for_small_sets(
    movie_list: List[str],
    user_preferences: str,
    context: str,
    model: str,
    api_keys: List[str],
    final_k: int = 50
) -> Dict[str, Any]:
    """
    Optimized reranking for small movie sets (< 100 movies).
    Uses direct LLM reranking with enhanced context.
    """
    
    if len(movie_list) <= final_k:
        # Very small set, just return as-is with basic ordering
        return {
            "movie_list": movie_list,
            "confidence": 0.8,
            "method_used": "direct_return_small_set",
            "statistics": {"input_movies": len(movie_list), "final_output": len(movie_list)}
        }
    
    try:
        from .GenerativeAI_redial import callLangChainLLMReranking_redial
        result = callLangChainLLMReranking_redial(
            context=context,
            user_preferences=user_preferences,
            movie_str="|".join(movie_list),
            model=model,
            api_key=api_keys,
            k=final_k
        )
        
        if isinstance(result, dict) and "movie_list" in result:
            return {
                "movie_list": result["movie_list"][:final_k],
                "confidence": 0.7,
                "method_used": "direct_llm_small_set",
                "statistics": {"input_movies": len(movie_list), "final_output": len(result["movie_list"])}
            }
    except Exception as e:
        logging.error(f"Small set reranking failed: {e}")
    
    # Fallback
    return {
        "movie_list": movie_list[:final_k],
        "confidence": 0.3,
        "method_used": "fallback_truncation",
        "statistics": {"input_movies": len(movie_list), "final_output": min(final_k, len(movie_list))}
    }

def adaptive_reranking_strategy(
    movie_list: List[str],
    user_preferences: str,
    context: str,
    movie_data_path: str,
    model: str,
    api_keys: List[str],
    cosine_scores: Optional[List[float]] = None,
    final_k: int = 50
) -> Dict[str, Any]:
    """
    Automatically selects the best reranking strategy based on input size and characteristics.
    
    Strategy selection:
    - Very small (≤ 50): Direct return or minimal reranking
    - Small (51-100): Direct LLM reranking
    - Medium (101-300): Hierarchical reranking with filtering
    - Large (301+): Full optimization pipeline
    """
    
    input_size = len(movie_list)
    
    if input_size <= 50:
        # Very small set
        logging.info(f"Using direct strategy for very small set ({input_size} movies)")
        return quick_reranking_for_small_sets(movie_list, user_preferences, context, model, api_keys, final_k)
    
    elif input_size <= 100:
        # Small set - direct LLM reranking
        logging.info(f"Using direct LLM reranking for small set ({input_size} movies)")
        return optimized_movie_reranking(
            movie_list=movie_list,
            user_preferences=user_preferences,
            context=context,
            movie_data_path=movie_data_path,
            model=model,
            api_keys=api_keys,
            cosine_scores=cosine_scores,
            final_k=final_k,
            use_semantic_filtering=False,
            use_hierarchical_reranking=False,
            use_hybrid_scoring=True,
            max_llm_candidates=100
        )
    
    elif input_size <= 300:
        # Medium set - hierarchical with light filtering
        logging.info(f"Using hierarchical strategy for medium set ({input_size} movies)")
        return optimized_movie_reranking(
            movie_list=movie_list,
            user_preferences=user_preferences,
            context=context,
            movie_data_path=movie_data_path,
            model=model,
            api_keys=api_keys,
            cosine_scores=cosine_scores,
            final_k=final_k,
            use_semantic_filtering=True,
            use_hierarchical_reranking=True,
            use_hybrid_scoring=True,
            max_llm_candidates=150,
            batch_size=20,
            candidates_per_batch=4
        )
    
    else:
        # Large set - full optimization pipeline
        logging.info(f"Using full optimization for large set ({input_size} movies)")
        return optimized_movie_reranking(
            movie_list=movie_list,
            user_preferences=user_preferences,
            context=context,
            movie_data_path=movie_data_path,
            model=model,
            api_keys=api_keys,
            cosine_scores=cosine_scores,
            final_k=final_k,
            use_semantic_filtering=True,
            use_hierarchical_reranking=True,
            use_hybrid_scoring=True,
            max_llm_candidates=200,
            batch_size=25,
            candidates_per_batch=5
        )
