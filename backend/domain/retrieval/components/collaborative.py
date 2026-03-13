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
        """Retrieve movies based on shared actors/directors with liked movies. Iteratively expands to meet n."""
        if not liked_movies:
            return []
            
        # Normalize titles (remove year) to match Neo4j format
        normalized_movies = [self._normalize_title(t) for t in liked_movies]
        current_titles = list(dict.fromkeys(normalized_movies))
        original_titles = [t.lower() for t in current_titles]
        
        logger.info(f"Collaborative filtering: Starting propagation for {n} candidates from {len(current_titles)} liked movies.")

        movies = []
        exclude_ids = []
        depth = 0
        max_depth = 4  # Prevent infinite loops/excessive queries

        try:
            async with self.neo4j_client.get_async_session() as session:
                while len(movies) < n and depth < max_depth and current_titles:
                    remaining = n - len(movies)
                    logger.info(f"Collaborative depth {depth}: expanding {len(current_titles)} titles to find {remaining} more movies.")
                    
                    result = await session.run(
                        """
                        MATCH (liked:Film)
                        WHERE toLower(liked.title) IN [t IN $titles | toLower(t)]
                        MATCH (liked)<-[:ACTED_IN|DIRECTED]-(p)-[:ACTED_IN|DIRECTED]->(rec:Film)
                        WHERE NOT toLower(rec.title) IN $original_titles
                          AND NOT rec.movieId IN $exclude_ids
                        RETURN rec.movieId AS movieId, rec.title AS title, rec.plot AS plot, rec.year AS year, count(p) as score
                        ORDER BY score DESC
                        LIMIT $limit
                        """, 
                        titles=current_titles, 
                        original_titles=original_titles, 
                        exclude_ids=exclude_ids, 
                        limit=remaining
                    )

                    level_movies = []
                    async for record in result:
                        # Apply depth penalty so direct connections have strictly higher scores
                        depth_penalty = 1.0 / (2 ** depth)
                        level_movies.append({
                            'movieId': record['movieId'],
                            'title': record['title'],
                            'plot': record['plot'],
                            'year': record['year'],
                            'score': float(record['score']) * depth_penalty
                        })
                    
                    if not level_movies:
                        logger.info(f"Collaborative depth {depth} yielded no new movies. Halting propagation.")
                        break
                        
                    movies.extend(level_movies)
                    
                    # Setup for next iteration
                    current_titles = []
                    for m in level_movies:
                        exclude_ids.append(m['movieId'])
                        current_titles.append(self._normalize_title(m['title']))
                        
                    depth += 1
                
                logger.info(f"Collaborative filtering: Finished at depth {depth}. Found {len(movies)} recommendations.")
                return movies
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {e}")
            return movies
