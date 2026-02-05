from typing import List, Dict, Any
from shared.utils.logger import setup_logger
from domain.retrieval.components.genre_extractor import GenreExtractor
from domain.retrieval.components.semantic import SemanticRetriever
from domain.retrieval.components.content import ContentRetriever
from domain.retrieval.components.collaborative import CollaborativeRetriever

logger = setup_logger(__name__)


class RetrievalService:
    """
    Service for retrieving movies from Neo4j graph database using semantic search,
    content-based filtering, and collaborative filtering.
    """

    def __init__(self):
        self.genre_extractor = GenreExtractor()
        self.semantic_retriever = SemanticRetriever()
        self.content_retriever = ContentRetriever()
        self.collab_retriever = CollaborativeRetriever()

    async def retrieve_movies(self, user_preferences: str, liked_movies: List[str] = [], n: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve movies using hybrid approach (Semantic + Content + Collab) in parallel.
        """
        import asyncio

        # Define internal flows
        async def _semantic_flow():
            # Use roughly n/2 for semantic if available
            embedding = await self.semantic_retriever.get_query_embedding(user_preferences)
            if embedding:
                return await self.semantic_retriever.retrieve(embedding, n // 2)
            return []

        async def _content_flow():
            content_n = n // 2
            found_genres = await self.genre_extractor.extract_genres(user_preferences)
            if found_genres:
                return await self.content_retriever.retrieve(found_genres, content_n)
            return []

        async def _collab_flow():
            if liked_movies:
                collab_n = n // 2
                return await self.collab_retriever.retrieve(liked_movies, collab_n)
            return []

        # Execute in parallel
        results = await asyncio.gather(
            _semantic_flow(),
            _content_flow(),
            _collab_flow()
        )

        semantic_movies, content_movies, collab_movies = results
        
        all_movies = []
        all_movies.extend(semantic_movies)
        all_movies.extend(content_movies)
        all_movies.extend(collab_movies)

        # Deduplicate
        unique_movies = {}
        for m in all_movies:
            unique_movies[m['movieId']] = m

        final_list = list(unique_movies.values())
        logger.info(f"Retrieved {len(final_list)} unique candidates (Requested n={n})")
        
        return {
            "combined": final_list[:n],
            "semantic": semantic_movies,
            "content": content_movies,
            "collaborative": collab_movies
        }
