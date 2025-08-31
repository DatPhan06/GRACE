# Hierarchical Reranking Implementation
import logging
import time
import random
from typing import List, Dict, Any, Tuple
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from infra.llm import create_gemini_langchain_llm
from langchain_together import ChatTogether
from pydantic import SecretStr

class MovieBatch(BaseModel):
    """Pydantic model for batch reranking output"""
    top_movies: List[str] = Field(description="Top ranked movies from this batch")
    reasoning: str = Field(description="Brief reasoning for the ranking")

class FinalRanking(BaseModel):
    """Pydantic model for final ranking output"""
    movie_list: List[str] = Field(description="Final ranked movie list")

def create_batch_reranking_chain(model: str, api_key: str, batch_size: int = 20):
    """
    Creates a chain for reranking small batches of movies
    """
    parser = JsonOutputParser(pydantic_object=MovieBatch)
    
    system_prompt = """<role>
You are an expert movie recommendation system specializing in batch ranking.
</role>

<instruction>
Your task is to rank a small batch of {batch_size} movies based on user preferences.
Select the top {top_k} movies that best match the user's taste and provide brief reasoning.
Focus on the most relevant movies that align with the user's stated preferences.
</instruction>

<constraint>
- Return ONLY valid JSON matching the specified schema
- Be decisive and clear in your selections
- Consider genre preferences, themes, and user context
</constraint>"""

    user_prompt = """<input>
User Preferences: {user_preferences}
Movie Batch: {movie_batch}
Conversation Context: {context}
</input>

Select the top {top_k} movies from this batch that best match the user's preferences.

<output_format>
{format_instructions}
</output_format>"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    
    if model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"]:
        llm = create_gemini_langchain_llm(
            api_key=api_key,
            model_name=model,
            max_output_tokens=5000,
        )
    else:
        llm = ChatTogether(model=model, api_key=SecretStr(api_key))
    
    return prompt | llm | parser

def create_final_reranking_chain(model: str, api_key: str):
    """
    Creates a chain for final reranking of top candidates
    """
    parser = JsonOutputParser(pydantic_object=FinalRanking)
    
    system_prompt = """<role>
You are an expert movie recommendation system for final ranking.
</role>

<instruction>
You are given a curated list of top movie candidates that have already been pre-filtered.
Your task is to provide the final ranking of these movies based on user preferences.
These movies are already highly relevant - focus on subtle preference matching.
</instruction>

<constraint>
- Return ONLY valid JSON matching the specified schema
- Rank all provided movies in order of relevance
- Consider nuanced preferences and conversation context
</constraint>"""

    user_prompt = """<input>
User Preferences: {user_preferences}
Top Candidates: {top_candidates}
Conversation Context: {context}
</input>

Provide the final ranking of these pre-selected top movies.

<output_format>
{format_instructions}
</output_format>"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    
    if model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"]:
        llm = create_gemini_langchain_llm(
            api_key=api_key,
            model_name=model,
            max_output_tokens=5000,
        )
    else:
        llm = ChatTogether(model=model, api_key=SecretStr(api_key))
    
    return prompt | llm | parser

def hierarchical_reranking(
    movie_list: List[str],
    user_preferences: str,
    context: str,
    model: str,
    api_keys: List[str],
    final_k: int = 50,
    batch_size: int = 25,
    candidates_per_batch: int = 5
) -> Dict[str, Any]:
    """
    Performs hierarchical reranking in multiple stages:
    
    1. Stage 1: Divide movies into batches, rank each batch
    2. Stage 2: Collect top candidates from all batches  
    3. Stage 3: Final reranking of top candidates
    
    Args:
        movie_list: List of movie titles to rerank
        user_preferences: Summarized user preferences
        context: Conversation context
        model: LLM model name
        api_keys: List of API keys
        final_k: Final number of movies to return
        batch_size: Size of each batch for stage 1
        candidates_per_batch: Number of movies to select from each batch
    
    Returns:
        Dictionary with final ranked movie list
    """
    
    # Stage 1: Batch processing
    logging.info(f"Stage 1: Processing {len(movie_list)} movies in batches of {batch_size}")
    
    batches = [movie_list[i:i + batch_size] for i in range(0, len(movie_list), batch_size)]
    top_candidates = []
    
    batch_chain = create_batch_reranking_chain(model, random.choice(api_keys), batch_size)
    
    for i, batch in enumerate(batches):
        try:
            logging.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} movies")
            
            result = batch_chain.invoke({
                "user_preferences": user_preferences,
                "movie_batch": "|".join(batch),
                "context": context,
                "batch_size": len(batch),
                "top_k": min(candidates_per_batch, len(batch))
            })
            
            if "top_movies" in result:
                top_candidates.extend(result["top_movies"])
                logging.info(f"Selected {len(result['top_movies'])} movies from batch {i+1}")
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logging.error(f"Error processing batch {i+1}: {e}")
            # Fallback: add first few movies from batch
            top_candidates.extend(batch[:candidates_per_batch])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for movie in top_candidates:
        if movie not in seen:
            seen.add(movie)
            unique_candidates.append(movie)
    
    logging.info(f"Stage 1 complete: {len(unique_candidates)} top candidates selected")
    
    # Stage 2: Final reranking
    if len(unique_candidates) <= final_k:
        # If we have fewer candidates than needed, return as is
        return {"movie_list": unique_candidates}
    
    logging.info(f"Stage 2: Final reranking of {len(unique_candidates)} candidates")
    
    try:
        final_chain = create_final_reranking_chain(model, random.choice(api_keys))
        
        final_result = final_chain.invoke({
            "user_preferences": user_preferences,
            "top_candidates": "|".join(unique_candidates),
            "context": context
        })
        
        if "movie_list" in final_result:
            # Return top k movies
            final_movies = final_result["movie_list"][:final_k]
            logging.info(f"Final ranking complete: {len(final_movies)} movies returned")
            return {"movie_list": final_movies}
        
    except Exception as e:
        logging.error(f"Error in final reranking: {e}")
    
    # Fallback: return top candidates truncated to final_k
    return {"movie_list": unique_candidates[:final_k]}

def enhanced_hierarchical_reranking(
    movie_list: List[str],
    user_preferences: str, 
    context: str,
    model: str,
    api_keys: List[str],
    cosine_scores: List[float] = None,
    final_k: int = 50,
    batch_size: int = 20,
    candidates_per_batch: int = 4
) -> Dict[str, Any]:
    """
    Enhanced version with cosine similarity integration
    
    Args:
        cosine_scores: Cosine similarity scores corresponding to movie_list
        Other args same as hierarchical_reranking
    """
    
    # If cosine scores provided, use them to pre-filter very low relevance movies
    if cosine_scores and len(cosine_scores) == len(movie_list):
        # Create movie-score pairs and sort by cosine similarity
        movie_score_pairs = list(zip(movie_list, cosine_scores))
        movie_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 60% based on cosine similarity for LLM reranking
        cutoff = max(final_k * 2, int(len(movie_list) * 0.6))
        filtered_movies = [movie for movie, _ in movie_score_pairs[:cutoff]]
        
        logging.info(f"Pre-filtered to top {len(filtered_movies)} movies based on cosine similarity")
        
        return hierarchical_reranking(
            movie_list=filtered_movies,
            user_preferences=user_preferences,
            context=context,
            model=model,
            api_keys=api_keys,
            final_k=final_k,
            batch_size=batch_size,
            candidates_per_batch=candidates_per_batch
        )
    
    # Fallback to standard hierarchical reranking
    return hierarchical_reranking(
        movie_list=movie_list,
        user_preferences=user_preferences,
        context=context,
        model=model,
        api_keys=api_keys,
        final_k=final_k,
        batch_size=batch_size,
        candidates_per_batch=candidates_per_batch
    )
