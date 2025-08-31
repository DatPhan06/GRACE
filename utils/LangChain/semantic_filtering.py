# Semantic Filtering for Movie Recommendations
import json
import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import numpy as np

class SemanticFilter:
    """
    Filters movies based on semantic relevance to user preferences.
    Removes obviously mismatched movies before expensive LLM reranking.
    """
    
    def __init__(self):
        # Define negative indicators for different content types
        self.genre_conflicts = {
            'family': ['horror', 'thriller', 'adult', 'erotic'],
            'children': ['horror', 'thriller', 'adult', 'erotic', 'mature'],
            'romance': ['horror', 'war', 'documentary'],
            'comedy': ['horror', 'war', 'drama'],
            'horror': ['family', 'children', 'comedy', 'musical'],
            'documentary': ['fantasy', 'sci-fi', 'animation']
        }
        
        # Age-inappropriate content indicators
        self.family_unsafe_keywords = {
            'violence', 'blood', 'murder', 'kill', 'death', 'drug', 'sex', 
            'adult', 'mature', 'graphic', 'brutal', 'gore', 'torture'
        }
        
        # Decade preferences mapping
        self.decade_preferences = {
            'classic': [1940, 1950, 1960, 1970],
            'retro': [1970, 1980, 1990],
            'modern': [2000, 2010, 2020],
            'recent': [2015, 2020],
            'old': [1940, 1950, 1960, 1970, 1980]
        }
    
    def extract_user_constraints(self, user_preferences: str) -> Dict[str, any]:
        """
        Extract filtering constraints from user preferences.
        
        Args:
            user_preferences: Summarized user preferences string
            
        Returns:
            Dictionary of constraints for filtering
        """
        preferences_lower = user_preferences.lower()
        
        constraints = {
            'required_genres': set(),
            'excluded_genres': set(),
            'family_safe': False,
            'children_safe': False,
            'min_year': None,
            'max_year': None,
            'preferred_decades': set(),
            'excluded_keywords': set(),
            'required_keywords': set()
        }
        
        # Genre preferences
        genre_keywords = {
            'action': ['action', 'adventure', 'fight', 'battle'],
            'comedy': ['comedy', 'funny', 'humor', 'laugh'],
            'drama': ['drama', 'emotional', 'serious'],
            'horror': ['horror', 'scary', 'fear', 'frightening'],
            'romance': ['romance', 'romantic', 'love'],
            'sci-fi': ['sci-fi', 'science fiction', 'futuristic', 'space'],
            'fantasy': ['fantasy', 'magic', 'magical', 'mystical'],
            'thriller': ['thriller', 'suspense', 'tense'],
            'family': ['family', 'kids', 'children'],
            'animation': ['animation', 'animated', 'cartoon']
        }
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in preferences_lower for keyword in keywords):
                constraints['required_genres'].add(genre)
        
        # Family safety
        family_indicators = ['family', 'kids', 'children', 'child', 'niece', 'nephew', 'daughter', 'son']
        if any(indicator in preferences_lower for indicator in family_indicators):
            constraints['family_safe'] = True
            constraints['children_safe'] = True
            constraints['excluded_genres'].update(['horror', 'thriller', 'adult'])
        
        # Age ratings
        if any(word in preferences_lower for word in ['pg', 'g-rated', 'family-friendly']):
            constraints['family_safe'] = True
        
        # Explicit dislikes
        dislike_patterns = [
            r'not interested in (\w+)',
            r'dislike (\w+)',
            r'hate (\w+)',
            r'avoid (\w+)',
            r'no (\w+)'
        ]
        
        for pattern in dislike_patterns:
            matches = re.findall(pattern, preferences_lower)
            for match in matches:
                if match in genre_keywords:
                    constraints['excluded_genres'].add(match)
        
        # Time period preferences
        year_patterns = [
            r'from (\d{4})', r'after (\d{4})', r'since (\d{4})',
            r'before (\d{4})', r'until (\d{4})', r'(\d{4})s'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, preferences_lower)
            for match in matches:
                year = int(match)
                if 'from' in pattern or 'after' in pattern or 'since' in pattern:
                    constraints['min_year'] = year
                elif 'before' in pattern or 'until' in pattern:
                    constraints['max_year'] = year
                elif 's' in pattern:  # decade like "1990s"
                    constraints['preferred_decades'].add(year)
        
        # Decade keywords
        for decade_term, years in self.decade_preferences.items():
            if decade_term in preferences_lower:
                constraints['preferred_decades'].update(years)
        
        return constraints
    
    def load_movie_metadata(self, movie_data_path: str) -> Dict[str, Dict]:
        """Load movie metadata for filtering."""
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
        
        return movie_db
    
    def check_genre_compatibility(
        self,
        movie_genres: str,
        required_genres: Set[str],
        excluded_genres: Set[str]
    ) -> bool:
        """
        Check if movie genres are compatible with user preferences.
        """
        if not movie_genres:
            return True  # No genre info, can't filter
        
        movie_genre_list = [g.strip().lower() for g in movie_genres.split(',')]
        
        # Check exclusions first
        if excluded_genres:
            for excluded in excluded_genres:
                if excluded in movie_genre_list:
                    return False
                
                # Check for conflict patterns
                if excluded in self.genre_conflicts:
                    conflicts = self.genre_conflicts[excluded]
                    if any(conflict in movie_genre_list for conflict in conflicts):
                        return False
        
        # Check requirements
        if required_genres:
            # At least one required genre should match
            genre_match = any(req in movie_genre_list for req in required_genres)
            if not genre_match:
                return False
        
        return True
    
    def check_family_safety(self, movie: Dict, family_safe: bool, children_safe: bool) -> bool:
        """
        Check if movie is appropriate for family viewing.
        """
        if not (family_safe or children_safe):
            return True
        
        # Check rating
        rating = movie.get('rated', '').upper()
        if children_safe and rating in ['R', 'NC-17', 'X']:
            return False
        if family_safe and rating in ['R', 'NC-17', 'X']:
            return False
        
        # Check plot for inappropriate content
        plot = movie.get('plot', '').lower()
        genre = movie.get('genre', '').lower()
        
        if family_safe or children_safe:
            # Check for unsafe keywords in plot
            if any(keyword in plot for keyword in self.family_unsafe_keywords):
                return False
            
            # Check for unsafe genres
            unsafe_genres = ['horror', 'thriller', 'adult', 'erotic']
            if any(unsafe in genre for unsafe in unsafe_genres):
                return False
        
        return True
    
    def check_temporal_constraints(
        self,
        movie_year: any,
        min_year: Optional[int],
        max_year: Optional[int],
        preferred_decades: Set[int]
    ) -> bool:
        """
        Check if movie meets temporal constraints.
        """
        if not movie_year:
            return True  # No year info, can't filter
        
        try:
            year = int(movie_year)
        except (ValueError, TypeError):
            return True  # Invalid year format
        
        # Check min/max year constraints
        if min_year and year < min_year:
            return False
        if max_year and year > max_year:
            return False
        
        # Check decade preferences (if specified)
        if preferred_decades:
            movie_decade = (year // 10) * 10
            if movie_decade not in preferred_decades:
                # Allow some flexibility - within 10 years of preferred decades
                decade_ranges = [(d, d+10) for d in preferred_decades]
                in_range = any(start <= year <= end for start, end in decade_ranges)
                if not in_range:
                    return False
        
        return True
    
    def filter_movies(
        self,
        movie_titles: List[str],
        user_preferences: str,
        movie_data_path: str,
        cosine_scores: Optional[List[float]] = None
    ) -> Tuple[List[str], Optional[List[float]]]:
        """
        Filter movies based on semantic relevance to user preferences.
        
        Args:
            movie_titles: List of movie titles to filter
            user_preferences: User preference summary
            movie_data_path: Path to movie metadata
            cosine_scores: Optional cosine similarity scores
            
        Returns:
            Tuple of (filtered_movies, filtered_scores)
        """
        
        # Extract filtering constraints
        constraints = self.extract_user_constraints(user_preferences)
        logging.info(f"Extracted filtering constraints: {constraints}")
        
        # Load movie metadata
        movie_db = self.load_movie_metadata(movie_data_path)
        
        filtered_movies = []
        filtered_scores = []
        
        for i, title in enumerate(movie_titles):
            # Clean title for lookup
            lookup_title = re.sub(r'\s*\(\d{4}\)$', '', title).strip()
            movie = movie_db.get(lookup_title.lower())
            
            if not movie:
                # If no metadata, include movie (can't filter)
                filtered_movies.append(title)
                if cosine_scores:
                    filtered_scores.append(cosine_scores[i])
                continue
            
            # Apply filters
            passes_filters = True
            
            # Genre compatibility
            if not self.check_genre_compatibility(
                movie.get('genre', ''),
                constraints['required_genres'],
                constraints['excluded_genres']
            ):
                passes_filters = False
                logging.debug(f"Filtered out {title}: genre incompatibility")
            
            # Family safety
            if passes_filters and not self.check_family_safety(
                movie,
                constraints['family_safe'],
                constraints['children_safe']
            ):
                passes_filters = False
                logging.debug(f"Filtered out {title}: family safety")
            
            # Temporal constraints
            if passes_filters and not self.check_temporal_constraints(
                movie.get('year'),
                constraints['min_year'],
                constraints['max_year'],
                constraints['preferred_decades']
            ):
                passes_filters = False
                logging.debug(f"Filtered out {title}: temporal constraints")
            
            if passes_filters:
                filtered_movies.append(title)
                if cosine_scores:
                    filtered_scores.append(cosine_scores[i])
        
        logging.info(f"Semantic filtering: {len(movie_titles)} -> {len(filtered_movies)} movies "
                    f"({len(movie_titles) - len(filtered_movies)} filtered out)")
        
        return filtered_movies, filtered_scores if cosine_scores else None

def apply_smart_filtering(
    movie_titles: List[str],
    user_preferences: str,
    movie_data_path: str,
    cosine_scores: List[float],
    max_candidates: int = 200
) -> Tuple[List[str], List[float]]:
    """
    Apply intelligent filtering to reduce candidate set for LLM reranking.
    
    Combines semantic filtering with score-based filtering.
    
    Args:
        movie_titles: All candidate movies
        user_preferences: User preference summary
        movie_data_path: Path to movie metadata
        cosine_scores: Cosine similarity scores
        max_candidates: Maximum number of candidates to pass to LLM
        
    Returns:
        Tuple of (filtered_movies, filtered_scores)
    """
    
    # Step 1: Apply semantic filtering
    semantic_filter = SemanticFilter()
    filtered_movies, filtered_scores = semantic_filter.filter_movies(
        movie_titles, user_preferences, movie_data_path, cosine_scores
    )
    
    # Step 2: If still too many candidates, apply score-based filtering
    if len(filtered_movies) > max_candidates:
        # Create movie-score pairs and sort by score
        movie_score_pairs = list(zip(filtered_movies, filtered_scores))
        movie_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top candidates by score
        top_pairs = movie_score_pairs[:max_candidates]
        filtered_movies, filtered_scores = zip(*top_pairs)
        filtered_movies, filtered_scores = list(filtered_movies), list(filtered_scores)
        
        logging.info(f"Score-based filtering: reduced to top {max_candidates} by cosine similarity")
    
    return filtered_movies, filtered_scores
