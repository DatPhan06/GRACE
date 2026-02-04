import logging
import os
from typing import List, Dict, Any, Optional
from domain.retrieval.prompts import EXTRACT_GENRES_SYSTEM_PROMPT, EXTRACT_GENRES_USER_PROMPT
from shared.settings.config import settings
from infra.neo4j import get_neo4j_client
from infra.llm import get_llm_client
from infra.embedding import get_embedding_service
import random


class RetrievalService:
    """
    Service for retrieving movies from Neo4j graph database using semantic search,
    content-based filtering, and collaborative filtering.
    """

    def __init__(self):
        self.neo4j_client = get_neo4j_client()
        self.llm_client = get_llm_client()
        self.embedding_service = get_embedding_service()
        self._cached_genres = None

    async def get_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """Get embedding for the query text using OpenAI."""
        if not query_text or not self.embedding_service:
            return None

        return await self.embedding_service.get_embedding(query_text)

    async def extract_genres_with_llm(self, preferences: str) -> List[str]:
        if not preferences:
            return []
        try:
            prompt = EXTRACT_GENRES_USER_PROMPT.format(preferences=preferences)
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=EXTRACT_GENRES_SYSTEM_PROMPT
            )
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
        preferences_lower = preferences.lower()
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
        if self._cached_genres:
            return self._cached_genres
        try:
            async with self.neo4j_client.get_async_session() as session:
                result = await session.run("MATCH (g:Genre) RETURN DISTINCT g.name AS genre ORDER BY g.name")
                genres = [record['genre'] async for record in result]
                self._cached_genres = genres
                return genres
        except Exception as e:
            logging.error(f"Error fetching genres from database: {e}")
            return []

    async def _match_genres_case_insensitive(self, llm_genres: List[str]) -> List[str]:
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

    async def retrieve_movies(self, user_preferences: str, liked_movies: List[str] = [], n: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve movies using hybrid approach (Semantic + Content + Collab).
        """
        all_movies = []

        # 1. Semantic Similarity (OpenAI)
        # Use roughly n/2 for semantic if available
        if self.embedding_service:
            embedding = await self.get_query_embedding(user_preferences)
            if embedding:
                semantic_movies = await self.get_movies_by_similarity(embedding, n // 2)
                all_movies.extend(semantic_movies)

        # 2. Content-based filtering (via Genres)
        content_n = n // 2
        content_movies = await self.get_movies_by_content_filter(user_preferences, content_n)
        all_movies.extend(content_movies)

        # 3. Collaborative filtering (via Liked Movies)
        if liked_movies:
            collab_n = n // 2
            collab_movies = await self.get_movies_by_collaborative_filtering(collab_n, liked_movies)
            all_movies.extend(collab_movies)

        # Deduplicate
        unique_movies = {}
        for m in all_movies:
            unique_movies[m['movieId']] = m

        final_list = list(unique_movies.values())
        return final_list[:n]

    async def get_movies_by_similarity(self, query_embedding: List[float], n: int = 50) -> List[Dict[str, Any]]:
        if not query_embedding:
            return []
        try:
            async with self.neo4j_client.get_async_session() as session:
                # Vector similarity search using manual dot product
                result = await session.run("""
                    MATCH (f:Film)
                    WHERE f.plot_embedding IS NOT NULL
                    WITH f,
                         reduce(dot = 0.0, i IN range(0, size(f.plot_embedding)-1) | 
                          dot + f.plot_embedding[i] * $queryEmbedding[i]) AS similarity
                    ORDER BY similarity DESC
                    LIMIT $limit
                    OPTIONAL MATCH (f)-[:HAS_RATING]->(r:ImdbRating)
                    RETURN f.movieId AS movieId, 
                           f.title AS title,
                           f.plot AS plot,
                           f.year AS year,
                           r.value AS imdbRating,
                           similarity
                """, queryEmbedding=query_embedding, limit=n)

                movies = []
                async for record in result:
                    movies.append({
                        'movieId': record['movieId'],
                        'title': record['title'],
                        'plot': record['plot'],
                        'year': record['year'],
                        'imdbRating': record['imdbRating'],
                        'similarity': record['similarity']
                    })
                return movies
        except Exception as e:
            logging.error(f"Error in similarity search: {e}")
            return []

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
