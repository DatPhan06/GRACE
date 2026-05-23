from typing import List, Dict, Any, Optional
from shared.utils.logger import setup_logger
from infra.neo4j import get_neo4j_client
from infra.embedding import get_embedding_service

logger = setup_logger(__name__)

class SemanticRetriever:
    """
    Component for semantic search using vector embeddings.
    """
    def __init__(self):
        self.neo4j_client = get_neo4j_client()
        self.embedding_service = get_embedding_service()

    async def get_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """Get embedding for the query text using OpenAI."""
        if not query_text or not self.embedding_service:
            return None
        return await self.embedding_service.get_embedding(query_text)

    async def retrieve(self, query_embeddings: List[List[float]], n: int = 50) -> List[Dict[str, Any]]:
        """Retrieve movies by vector similarity for multiple query embeddings."""
        if not query_embeddings:
            return []
            
        import asyncio
        
        async def _single_retrieve(embedding):
            try:
                async with self.neo4j_client.get_async_session() as session:
                    result = await session.run("""
                        MATCH (f:Film)
                        WHERE f.plot_embedding IS NOT NULL
                        WITH f,
                             reduce(dot = 0.0, i IN range(0, size(f.plot_embedding)-1) |
                              dot + f.plot_embedding[i] * $queryEmbedding[i]) AS similarity
                        ORDER BY similarity DESC
                        LIMIT $limit
                        OPTIONAL MATCH (f)-[:HAS_RATING]->(r:ImdbRating)
                        OPTIONAL MATCH (f)-[:IN_GENRE]->(g:Genre)
                        WITH f, r, similarity, collect(g.name) AS genres
                        RETURN f.movieId AS movieId,
                               f.title AS title,
                               f.plot AS plot,
                               f.year AS year,
                               r.value AS imdbRating,
                               similarity,
                               genres
                    """, queryEmbedding=embedding, limit=n)

                    movies = []
                    async for record in result:
                        movies.append({
                            'movieId': record['movieId'],
                            'title': record['title'],
                            'plot': record['plot'],
                            'year': record['year'],
                            'imdbRating': record['imdbRating'],
                            'similarity': record['similarity'],
                            'genres': record['genres'],
                        })
                    return movies
            except Exception as e:
                logger.error(f"Error in similarity search: {e}")
                return []

        # Run all queries in parallel
        tasks = [_single_retrieve(emb) for emb in query_embeddings]
        all_results = await asyncio.gather(*tasks)
        
        # Deduplicate and sort by max similarity
        unique_movies = {}
        for movie_list in all_results:
            for m in movie_list:
                m_id = m['movieId']
                if m_id not in unique_movies or m['similarity'] > unique_movies[m_id]['similarity']:
                    unique_movies[m_id] = m
        
        # Sort by similarity and return top n
        sorted_movies = sorted(unique_movies.values(), key=lambda x: x['similarity'], reverse=True)
        return sorted_movies[:n]
