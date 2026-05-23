from typing import List, Dict, Any
from shared.utils.logger import setup_logger
from domain.retrieval.components.semantic import SemanticRetriever
from domain.retrieval.components.content import ContentRetriever
from domain.retrieval.components.graph_agent import GraphReasoningAgent
from domain.retrieval.components.genre_extractor import GENRE_SYNONYMS

logger = setup_logger(__name__)


def _normalize_genres(genres: List[str]) -> List[str]:
    """Map non-standard genre terms to canonical DB genre names."""
    result = []
    for g in genres:
        key = g.lower()
        if key in GENRE_SYNONYMS:
            result.extend(GENRE_SYNONYMS[key])
        else:
            result.append(key)
    return list(set(result))


class RetrievalService:
    """
    Service for retrieving movies from Neo4j graph database using semantic search,
    content-based filtering, and multi-agent graph reasoning.
    """

    def __init__(self):
        self.semantic_retriever = SemanticRetriever()
        self.content_retriever = ContentRetriever()
        self.collab_retriever = GraphReasoningAgent()

    async def retrieve_movies(self, user_preferences: str, liked_movies: List[str] = None, n: int = 100, dynamic_weights: Dict[str, float] = None, semantic_queries: List[str] = None, hard_constraints: List[str] = None, genres: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve movies using hybrid approach (Semantic + Content + Collab) in parallel and fuse with WRRF.
        """
        import asyncio
        if liked_movies is None:
            liked_movies = []
        if dynamic_weights is None:
            dynamic_weights = {"w_sem": 0.4, "w_con": 0.3, "w_col": 0.3}
        if semantic_queries is None:
            semantic_queries = [user_preferences]
        if hard_constraints is None:
            hard_constraints = []
        if genres is None:
            genres = []

        agent_trace = []

        # Normalize genres from profiler before content filtering
        genres = _normalize_genres(genres)

        # Weighted Reciprocal Rank Fusion (WRRF) Setup early
        w_sem = dynamic_weights.get("w_sem", 0.4)
        w_con = dynamic_weights.get("w_con", 0.3)
        w_col = dynamic_weights.get("w_col", 0.3)
        
        # Define internal flows. Retrieve full 'n' for each branch to overlap.
        async def _semantic_flow():
            if w_sem < 0.05:
                agent_trace.append(f"Orchestrator: Bypassing Semantic Agent (weight {w_sem} below threshold)")
                return []
            agent_trace.append(f"Orchestrator: Activating Semantic Agent (weight {w_sem}) with {len(semantic_queries)} sub-queries")
            # Get embeddings for all semantic queries
            emb_tasks = [self.semantic_retriever.get_query_embedding(q) for q in semantic_queries]
            embeddings = await asyncio.gather(*emb_tasks)
            valid_embeddings = [e for e in embeddings if e is not None]
            
            if valid_embeddings:
                return await self.semantic_retriever.retrieve(valid_embeddings, n)
            return []

        async def _content_flow():
            if w_con < 0.05:
                agent_trace.append(f"Orchestrator: Bypassing Content Filter (weight {w_con} below threshold)")
                return []
            if not genres:
                agent_trace.append(f"Orchestrator: Bypassing Content Filter (no genres extracted by Profiler)")
                return []
            agent_trace.append(f"Orchestrator: Activating Content Filter (weight {w_con}) with genres: {genres}")
            return await self.content_retriever.retrieve(genres, n)

        async def _collab_flow():
            if w_col < 0.05:
                agent_trace.append(f"Orchestrator: Bypassing Graph Agent (weight {w_col} below threshold)")
                return {"movies": [], "thoughts": []}
            if liked_movies or hard_constraints:
                agent_trace.append(f"Orchestrator: Activating Graph Agent (weight {w_col})")
                try:
                    return await asyncio.wait_for(
                        self.collab_retriever.retrieve(user_preferences, liked_movies, n, hard_constraints=hard_constraints),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    agent_trace.append("Orchestrator: Graph Agent timed out after 10s — proceeding with Semantic + Content results.")
                    return {"movies": [], "thoughts": []}
            return {"movies": [], "thoughts": []}

        # Execute in parallel
        results = await asyncio.gather(
            _semantic_flow(),
            _content_flow(),
            _collab_flow()
        )

        semantic_movies, content_movies, collab_data = results
        collab_movies = collab_data.get("movies", [])
        graph_thoughts = collab_data.get("thoughts", [])
        
        for t in graph_thoughts:
            agent_trace.append(f"Graph Agent: {t}")

        # Weighted Reciprocal Rank Fusion (WRRF)
        # Score_m = w_sem * (1 / (k + rank_sem_m)) + w_con * (1 / (k + rank_con_m)) + w_col * (1 / (k + rank_col_m))
        k = 60
        movie_scores = {}
        unique_movies = {}

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
            "collaborative": collab_movies,
            "agent_trace": agent_trace
        }
