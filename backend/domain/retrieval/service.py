import logging
from typing import List, Dict, Any, Optional
from shared.settings.config import settings
from infra.neo4j import get_neo4j_client
from infra.llm import get_llm_client
import random

class RetrievalService:
    """
    Service for retrieving movies from Neo4j graph database using semantic search,
    content-based filtering, and collaborative filtering.
    Refactored from utils/GraphDB/graph_retriever.py
    """
    
    def __init__(self):
        self.neo4j_client = get_neo4j_client()
        self.llm_client = get_llm_client()
        self._cached_genres = None
        
    async def extract_genres_with_llm(self, preferences: str) -> List[str]:
        """
        Use LLM to extract movie genres from user preferences.
        """
        if not preferences:
            return []
            
        try:
            prompt = f"""
            Based on the following user conversation about movie preferences, extract the movie genres they are interested in.
            
            User conversation:
            {preferences}
            
            Available genres: action, comedy, drama, horror, romance, thriller, sci-fi, science fiction, fantasy, animation, documentary, mystery, adventure, crime, family, war, western, musical, biography, historical, psychological, supernatural, martial arts, sports, dance, music
            
            Instructions:
            - Return only the genre names that are clearly mentioned or strongly implied
            - Use lowercase
            - Separate multiple genres with commas
            - If sci-fi or science fiction is mentioned, return "sci-fi"
            - If no clear genres are found, return "none"
            
            Extracted genres:"""
            
            # Use generate from infra.llm (synchronous or async depending on implementation)
            # Assuming generate is synchronous for now based on base.py, but used in async context
            # We will wrap it or use async if available. BaseLLM has agenerate.
            response = await self.llm_client.agenerate(prompt)
            extracted_text = response.strip().lower()
            
            if extracted_text == "none" or not extracted_text:
                return self._extract_genres_keyword_matching(preferences)
            
            genres = [genre.strip() for genre in extracted_text.split(',')]
            genres = [genre for genre in genres if genre and len(genre) > 0]
            
            return await self._match_genres_case_insensitive(genres)
                
        except Exception as e:
            logging.error(f"Error extracting genres with LLM: {e}")
            return self._extract_genres_keyword_matching(preferences)

    def _extract_genres_keyword_matching(self, preferences: str) -> List[str]:
        """Fallback method: Extract genres using keyword matching."""
        preferences_lower = preferences.lower()
        # Simple keyword mapping (could be expanded)
        genre_keywords = {
            'action': ['action', 'fight', 'battle'],
            'comedy': ['comedy', 'funny', 'humor'],
            'drama': ['drama', 'emotional'],
            'horror': ['horror', 'scary', 'ghost'],
            'romance': ['romance', 'love'],
            'sci-fi': ['sci-fi', 'science fiction', 'space'],
            'fantasy': ['fantasy', 'magic'],
            'animation': ['animation', 'cartoon'],
        }
        
        found_genres = []
        for genre, keywords in genre_keywords.items():
            for keyword in keywords:
                if keyword in preferences_lower:
                    found_genres.append(genre)
                    break
        return found_genres

    async def _get_available_genres_from_db(self) -> List[str]:
        """Get all available genres from the Neo4j database."""
        if self._cached_genres:
            return self._cached_genres
            
        try:
            async with self.neo4j_client.get_async_session() as session:
                result = await session.run("""
                    MATCH (g:Genre)
                    RETURN DISTINCT g.name AS genre
                    ORDER BY g.name
                """)
                genres = [record['genre'] async for record in result]
                self._cached_genres = genres
                return genres
        except Exception as e:
            logging.error(f"Error fetching genres from database: {e}")
            return []

    async def _match_genres_case_insensitive(self, llm_genres: List[str]) -> List[str]:
        """Match LLM-extracted genres with actual database genres."""
        db_genres = await self._get_available_genres_from_db()
        if not db_genres:
            return llm_genres
            
        matched_genres = []
        for llm_genre in llm_genres:
            for db_genre in db_genres:
                if llm_genre.lower() == db_genre.lower():
                    matched_genres.append(db_genre.lower())
                    break
        return list(set(matched_genres))

    async def retrieve_movies(self, user_preferences: str, liked_movies: List[str] = [], n: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve movies using content-based and collaborative filtering.
        """
        all_movies = []
        
        # 1. Content-based filtering (via Genres)
        content_movies = await self.get_movies_by_content_filter(user_preferences, n)
        all_movies.extend(content_movies)
        
        # 2. Collaborative filtering (via Liked Movies)
        if liked_movies:
            collab_movies = await self.get_movies_by_collaborative_filtering(n, liked_movies)
            all_movies.extend(collab_movies)
            
        # Deduplicate and limit
        unique_movies = {m['movieId']: m for m in all_movies}.values()
        return list(unique_movies)[:n]

    async def get_movies_by_content_filter(self, preferences: str, n: int = 20) -> List[Dict[str, Any]]:
        found_genres = await self.extract_genres_with_llm(preferences)
        if not found_genres:
            return []
            
        try:
            async with self.neo4j_client.get_async_session() as session:
                result = await session.run("""
                    MATCH (f:Film)-[:IN_GENRE]->(g:Genre)
                    WHERE toLower(g.name) IN $genres
                    OPTIONAL MATCH (f)-[:HAS_RATING]->(r:ImdbRating)
                    RETURN f.movieId AS movieId, f.title AS title, f.plot AS plot, f.year AS year, r.value AS imdbRating
                    ORDER BY r.value DESC
                    LIMIT $limit
                """, genres=found_genres, limit=n)
                
                movies = []
                async for record in result:
                    movies.append({
                        'movieId': record['movieId'],
                        'title': record['title'],
                        'plot': record['plot'],
                        'year': record['year'],
                        'imdbRating': record['imdbRating']
                    })
                return movies
        except Exception as e:
            logging.error(f"Error in content filtering: {e}")
            return []

    async def get_movies_by_collaborative_filtering(self, n: int, liked_movies: List[str]) -> List[Dict[str, Any]]:
        if not liked_movies:
            return []
            
        try:
            async with self.neo4j_client.get_async_session() as session:
                # Simple collaborative filtering: Movies connected to liked movies via actors/directors
                result = await session.run("""
                    MATCH (liked:Film)
                    WHERE toLower(liked.title) IN [t IN $titles | toLower(t)]
                    MATCH (liked)<-[:ACTED_IN|DIRECTED]-(p)-[:ACTED_IN|DIRECTED]->(rec:Film)
                    WHERE NOT toLower(rec.title) IN [t IN $titles | toLower(t)]
                    RETURN rec.movieId AS movieId, rec.title AS title, rec.plot AS plot, rec.year AS year, count(p) as score
                    ORDER BY score DESC
                    LIMIT $limit
                """, titles=liked_movies, limit=n)
                
                movies = []
                async for record in result:
                     movies.append({
                        'movieId': record['movieId'],
                        'title': record['title'],
                        'plot': record['plot'],
                        'year': record['year'],
                        'score': record['score']
                    })
                return movies
        except Exception as e:
            logging.error(f"Error in collaborative filtering: {e}")
            return []
