from typing import List, Dict, Any
import re
from shared.utils.logger import setup_logger
from infra.neo4j import get_neo4j_client

logger = setup_logger(__name__)

class CollaborativeRetriever:
    """
    Component for cooperative filtering (via Liked Movies).
    """
    def __init__(self):
        self.neo4j_client = get_neo4j_client()

    def _normalize_title(self, title: str) -> str:
        """Strip year (YYYY) from title."""
        return re.sub(r'\s*\(\d{4}\)$', '', title).strip()

    async def retrieve(self, liked_movies: List[str], n: int) -> List[Dict[str, Any]]:
        """Retrieve movies based on shared actors/directors with liked movies."""
        if not liked_movies:
            return []
            
        # Normalize titles (remove year) to match Neo4j format
        normalized_movies = [self._normalize_title(t) for t in liked_movies]
        # Use simple deduplication while preserving order
        search_titles = list(dict.fromkeys(normalized_movies))
        
        logger.info(f"Collaborative filtering: Searching for {len(search_titles)} titles (from {len(liked_movies)} inputs): {search_titles}")

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
                """, titles=search_titles, limit=n)

                movies = []
                async for record in result:
                    movies.append({
                        'movieId': record['movieId'],
                        'title': record['title'],
                        'plot': record['plot'],
                        'year': record['year'],
                        'score': record['score']
                    })
                
                logger.info(f"Collaborative filtering: Found {len(movies)} recommendations.")
                
                return movies
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {e}")
            return []
