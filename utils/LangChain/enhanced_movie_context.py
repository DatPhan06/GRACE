# Enhanced Movie Context for Reranking
import json
import logging
from typing import List, Dict, Optional
import re

def create_compact_movie_context(
    movie_titles: List[str], 
    movie_data_path: str,
    max_context_length: int = 150
) -> List[str]:
    """
    Creates compact but informative context for each movie for reranking.
    
    Args:
        movie_titles: List of movie titles
        movie_data_path: Path to movie data JSONL file
        max_context_length: Maximum characters per movie context
    
    Returns:
        List of compact movie contexts for reranking
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
        logging.error(f"Movie data file not found: {movie_data_path}")
        return movie_titles  # Fallback to just titles
    
    enhanced_contexts = []
    
    for title in movie_titles:
        # Clean title for lookup
        lookup_title = re.sub(r'\s*\(\d{4}\)$', '', title).strip()
        movie = movie_db.get(lookup_title.lower())
        
        if movie:
            # Create compact context with most important info
            context_parts = [f"'{title}'"]
            
            # Add year if available
            if movie.get("year"):
                context_parts.append(f"({movie['year']})")
            
            # Add primary genre
            genre = movie.get("genre", "")
            if genre:
                # Take first genre if multiple
                primary_genre = genre.split(",")[0].strip() if "," in genre else genre
                context_parts.append(f"[{primary_genre}]")
            
            # Add key cast member or director
            director = movie.get("director", "")
            actors = movie.get("actors", "")
            
            if director and len(director) < 30:
                context_parts.append(f"Dir: {director}")
            elif actors:
                # Take first actor
                main_actor = actors.split(",")[0].strip() if "," in actors else actors
                if len(main_actor) < 30:
                    context_parts.append(f"Star: {main_actor}")
            
            # Create context string
            context = " ".join(context_parts)
            
            # Add brief plot if space allows
            plot = movie.get("plot", "")
            if plot and len(context) < max_context_length - 50:
                # Get first sentence of plot
                plot_sentences = plot.split(". ")
                if plot_sentences:
                    brief_plot = plot_sentences[0]
                    if len(brief_plot) < 80:
                        context += f" - {brief_plot}"
            
            enhanced_contexts.append(context)
        else:
            # Fallback to just title
            enhanced_contexts.append(title)
    
    return enhanced_contexts

def create_themed_movie_batches(
    movie_titles: List[str],
    movie_data_path: str,
    batch_size: int = 25
) -> List[List[str]]:
    """
    Groups movies into thematically similar batches for better reranking.
    
    Args:
        movie_titles: List of movie titles
        movie_data_path: Path to movie data
        batch_size: Target size for each batch
    
    Returns:
        List of movie batches grouped by similarity
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
        # Fallback to simple batching
        return [movie_titles[i:i + batch_size] for i in range(0, len(movie_titles), batch_size)]
    
    # Group movies by primary characteristics
    genre_groups = {}
    decade_groups = {}
    ungrouped = []
    
    for title in movie_titles:
        lookup_title = re.sub(r'\s*\(\d{4}\)$', '', title).strip()
        movie = movie_db.get(lookup_title.lower())
        
        if movie:
            # Group by primary genre
            genre = movie.get("genre", "")
            if genre:
                primary_genre = genre.split(",")[0].strip().lower()
                if primary_genre not in genre_groups:
                    genre_groups[primary_genre] = []
                genre_groups[primary_genre].append(title)
            
            # Also group by decade for additional grouping
            year = movie.get("year")
            if year:
                try:
                    decade = (int(year) // 10) * 10
                    if decade not in decade_groups:
                        decade_groups[decade] = []
                    decade_groups[decade].append(title)
                except (ValueError, TypeError):
                    ungrouped.append(title)
            else:
                ungrouped.append(title)
        else:
            ungrouped.append(title)
    
    # Create balanced batches
    batches = []
    
    # Process genre groups first
    for genre, movies in genre_groups.items():
        if len(movies) >= batch_size:
            # Split large groups into multiple batches
            for i in range(0, len(movies), batch_size):
                batches.append(movies[i:i + batch_size])
        elif len(movies) >= batch_size // 2:
            # Medium groups become their own batch
            batches.append(movies)
        # Small groups will be mixed later
    
    # Mix smaller groups and ungrouped movies
    remaining_movies = []
    for genre, movies in genre_groups.items():
        if len(movies) < batch_size // 2:
            remaining_movies.extend(movies)
    remaining_movies.extend(ungrouped)
    
    # Create batches from remaining movies
    for i in range(0, len(remaining_movies), batch_size):
        batches.append(remaining_movies[i:i + batch_size])
    
    return batches

def get_movie_embedding_features(
    movie_titles: List[str],
    movie_data_path: str
) -> Dict[str, Dict[str, str]]:
    """
    Extracts key features for each movie that are useful for semantic matching.
    
    Args:
        movie_titles: List of movie titles
        movie_data_path: Path to movie data
    
    Returns:
        Dictionary mapping movie titles to their key features
    """
    
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
        logging.error(f"Movie data file not found: {movie_data_path}")
        return {}
    
    features = {}
    
    for title in movie_titles:
        lookup_title = re.sub(r'\s*\(\d{4}\)$', '', title).strip()
        movie = movie_db.get(lookup_title.lower())
        
        if movie:
            features[title] = {
                "genre": movie.get("genre", ""),
                "year": str(movie.get("year", "")),
                "director": movie.get("director", ""),
                "main_actor": movie.get("actors", "").split(",")[0].strip() if movie.get("actors") else "",
                "plot_keywords": extract_plot_keywords(movie.get("plot", "")),
                "rating": movie.get("rated", ""),
                "runtime": str(movie.get("runtime", ""))
            }
        else:
            features[title] = {
                "genre": "", "year": "", "director": "", "main_actor": "",
                "plot_keywords": "", "rating": "", "runtime": ""
            }
    
    return features

def extract_plot_keywords(plot: str, max_keywords: int = 5) -> str:
    """
    Extracts key thematic words from movie plot for better matching.
    """
    if not plot:
        return ""
    
    # Simple keyword extraction - in practice, you might use NLP libraries
    # Remove common words and focus on nouns/themes
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'have', 'has',
        'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'this', 'that', 'these', 'those', 'his', 'her', 'their', 'our', 'my'
    }
    
    # Clean and split text
    words = re.findall(r'\b[a-zA-Z]{4,}\b', plot.lower())
    keywords = [word for word in words if word not in common_words]
    
    # Take most frequent words (simple approach)
    from collections import Counter
    word_counts = Counter(keywords)
    top_keywords = [word for word, _ in word_counts.most_common(max_keywords)]
    
    return ", ".join(top_keywords)

# Integration function for the main reranking pipeline
def prepare_enhanced_movie_batches(
    movie_titles: List[str],
    movie_data_path: str,
    batch_size: int = 25,
    include_context: bool = True
) -> List[List[str]]:
    """
    Prepares movie batches with enhanced context for hierarchical reranking.
    
    Args:
        movie_titles: List of movie titles to batch
        movie_data_path: Path to movie metadata
        batch_size: Target batch size
        include_context: Whether to include enhanced context
    
    Returns:
        List of batches, where each batch contains either titles or enhanced contexts
    """
    
    # Create thematically similar batches
    batches = create_themed_movie_batches(movie_titles, movie_data_path, batch_size)
    
    if include_context:
        # Enhance each batch with compact contexts
        enhanced_batches = []
        for batch in batches:
            enhanced_batch = create_compact_movie_context(batch, movie_data_path)
            enhanced_batches.append(enhanced_batch)
        return enhanced_batches
    
    return batches
