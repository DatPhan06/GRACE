from typing import List, Dict, Any
from shared.utils.logger import setup_logger
from infra.neo4j import get_neo4j_client

logger = setup_logger(__name__)

class ContentRetriever:
    """
    Component for content-based filtering (via Genres).
    """
    def __init__(self):
        self.neo4j_client = get_neo4j_client()

    async def retrieve(self, genres: List[str], n: int = 20) -> List[Dict[str, Any]]:
        """Retrieve movies by genres."""
        if not genres:
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
                """, genres=genres, limit=n)

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
            logger.error(f"Error in content filtering: {e}")
            return []
