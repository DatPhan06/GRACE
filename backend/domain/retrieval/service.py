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

    async def retrieve_movies(self, user_preferences: str, liked_movies: List[str] = None, n: int = 100, dynamic_weights: Dict[str, float] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve movies using hybrid approach (Semantic + Content + Collab) in parallel and fuse with WRRF.
        """
        import asyncio
        if liked_movies is None:
            liked_movies = []
        if dynamic_weights is None:
            dynamic_weights = {"w_sem": 0.4, "w_con": 0.3, "w_col": 0.3}

        # Define internal flows. Retrieve full 'n' for each branch to overlap.
        async def _semantic_flow():
            embedding = await self.semantic_retriever.get_query_embedding(user_preferences)
            if embedding:
                return await self.semantic_retriever.retrieve(embedding, n)
            return []

        async def _content_flow():
            found_genres = await self.genre_extractor.extract_genres(user_preferences)
            if found_genres:
                return await self.content_retriever.retrieve(found_genres, n)
            return []

        async def _collab_flow():
            if liked_movies:
                return await self.collab_retriever.retrieve(liked_movies, n)
            return []

        # Execute in parallel
        results = await asyncio.gather(
            _semantic_flow(),
            _content_flow(),
            _collab_flow()
        )

        semantic_movies, content_movies, collab_movies = results
        
        # Weighted Reciprocal Rank Fusion (WRRF)
        # Score_m = w_sem * (1 / (k + rank_sem_m)) + w_con * (1 / (k + rank_con_m)) + w_col * (1 / (k + rank_col_m))
        k = 60
        movie_scores = {}
        unique_movies = {}

        w_sem = dynamic_weights.get("w_sem", 0.4)
        w_con = dynamic_weights.get("w_con", 0.3)
        w_col = dynamic_weights.get("w_col", 0.3)

        def add_to_fusion(movie_list, weight):
            for rank, m in enumerate(movie_list):
                m_id = m['movieId']
                unique_movies[m_id] = m
                score = weight * (1.0 / (k + rank + 1))
                movie_scores[m_id] = movie_scores.get(m_id, 0.0) + score

        add_to_fusion(semantic_movies, w_sem)
        add_to_fusion(content_movies, w_con)
        add_to_fusion(collab_movies, w_col)

        # Sort by WRRF score
        sorted_m_ids = sorted(movie_scores.keys(), key=lambda x: movie_scores[x], reverse=True)
        final_list = [unique_movies[m_id] for m_id in sorted_m_ids]
        
        logger.info(f"Retrieved {len(final_list)} unique candidates, top n={n} selected using WRRF (w_sem={w_sem}, w_con={w_con}, w_col={w_col})")
        
        return {
            "combined": final_list[:n],
            "semantic": semantic_movies,
            "content": content_movies,
            "collaborative": collab_movies
        }
