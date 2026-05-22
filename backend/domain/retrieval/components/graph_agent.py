import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from infra.llm import get_llm_client
from infra.neo4j import get_neo4j_client
from domain.retrieval.prompts.graph import (
    NEO4J_SCHEMA,
    GRAPH_REASONING_SYSTEM_PROMPT,
    GRAPH_CONSTRAINT_USER_PROMPT,
    GRAPH_COLLABORATIVE_USER_PROMPT,
)
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)


class GraphReActStep(BaseModel):
    thought: str = Field(description="Reasoning about what Cypher query to write")
    cypher: str = Field(description="The Cypher query to execute against Neo4j")


class GraphReasoningAgent:
    """
    Two-strategy parallel Graph Agent (ARGOS design):
      - Constraint strategy: translates hard_constraints into Cypher WHERE clauses.
      - Collaborative strategy: traverses ACTED_IN/DIRECTED from liked_movies anchor nodes.
    Both strategies run their own ReAct loop concurrently; results are merged before returning.
    """

    def __init__(self):
        self.llm_client = get_llm_client()
        self.neo4j_client = get_neo4j_client()
        self.max_iterations = 3
        self._system_prompt = GRAPH_REASONING_SYSTEM_PROMPT.format(schema=NEO4J_SCHEMA)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        user_preferences: str,
        liked_movies: List[str],
        n: int = 100,
        hard_constraints: List = None,
    ) -> Dict[str, Any]:
        tasks = []

        if hard_constraints:
            tasks.append(self._constraint_strategy(user_preferences, hard_constraints, n))

        if liked_movies:
            tasks.append(self._collaborative_strategy(user_preferences, liked_movies, n))

        if not tasks:
            return {"movies": [], "thoughts": []}

        results = await asyncio.gather(*tasks)

        # Merge, preserving order of constraint results first, deduplicate by movieId
        merged_movies: List[Dict[str, Any]] = []
        seen_ids: set = set()
        all_thoughts: List[str] = []
        for r in results:
            all_thoughts.extend(r.get("thoughts", []))
            for m in r.get("movies", []):
                mid = m.get("movieId")
                if mid and mid not in seen_ids:
                    merged_movies.append(m)
                    seen_ids.add(mid)

        logger.info(
            f"[Graph Agent] Merged {len(merged_movies)} unique candidates "
            f"from {len(results)} parallel strategies."
        )
        return {"movies": merged_movies[:n], "thoughts": all_thoughts}

    # ------------------------------------------------------------------
    # Private strategies
    # ------------------------------------------------------------------

    async def _constraint_strategy(
        self,
        user_preferences: str,
        hard_constraints: List,
        n: int,
    ) -> Dict[str, Any]:
        constraints_str = ", ".join(
            c.constraint if hasattr(c, "constraint") else str(c)
            for c in hard_constraints
        )
        logger.info(f"[Graph Agent] Constraint strategy with: {constraints_str[:80]}...")
        return await self._react_loop(
            label="Constraint",
            user_preferences=user_preferences,
            user_prompt_template=GRAPH_CONSTRAINT_USER_PROMPT,
            template_kwargs={"hard_constraints": constraints_str},
            n=n,
        )

    async def _collaborative_strategy(
        self,
        user_preferences: str,
        liked_movies: List[str],
        n: int,
    ) -> Dict[str, Any]:
        liked_str = ", ".join(liked_movies)
        logger.info(f"[Graph Agent] Collaborative strategy from: {liked_str[:80]}...")
        return await self._react_loop(
            label="Collaborative",
            user_preferences=user_preferences,
            user_prompt_template=GRAPH_COLLABORATIVE_USER_PROMPT,
            template_kwargs={"liked_movies": liked_str},
            n=n,
        )

    # ------------------------------------------------------------------
    # Core ReAct loop (shared by both strategies)
    # ------------------------------------------------------------------

    async def _react_loop(
        self,
        label: str,
        user_preferences: str,
        user_prompt_template: str,
        template_kwargs: Dict[str, str],
        n: int,
    ) -> Dict[str, Any]:
        history = ""
        thoughts: List[str] = []

        for iteration in range(self.max_iterations):
            logger.info(f"[Graph Agent/{label}] Turn {iteration + 1}")

            prompt = user_prompt_template.format(
                user_preferences=user_preferences,
                history=history or "No previous attempts.",
                **template_kwargs,
            )

            try:
                response = await self.llm_client.agenerate(
                    prompt=prompt,
                    system_instruction=self._system_prompt,
                    response_schema=GraphReActStep,
                )
                step = GraphReActStep.model_validate_json(response)

                if not step.cypher:
                    logger.error(f"[Graph Agent/{label}] No Cypher generated.")
                    break

                logger.info(f"[Graph Agent/{label}] Thought: {step.thought}")
                thoughts.append(f"[{label}] {step.thought}")
                logger.info(f"[Graph Agent/{label}] Executing Cypher: {step.cypher}")

                movies = await self._execute_cypher(step.cypher, n)

                if movies:
                    logger.info(f"[Graph Agent/{label}] Retrieved {len(movies)} movies.")
                    return {"movies": movies, "thoughts": thoughts}

                logger.info(f"[Graph Agent/{label}] 0 results — will reflect and retry.")
                history += (
                    f"Attempt {iteration + 1}:\nThought: {step.thought}\n"
                    f"Cypher: {step.cypher}\nResult: 0 records. The constraint was too strict.\n\n"
                )

            except Exception as e:
                logger.error(f"[Graph Agent/{label}] Error: {e}")
                history += f"Attempt {iteration + 1} Error: {e}\n\n"

        logger.warning(f"[Graph Agent/{label}] Failed after {self.max_iterations} iterations.")
        return {"movies": [], "thoughts": thoughts}

    async def _execute_cypher(self, cypher: str, n: int) -> List[Dict[str, Any]]:
        movies: List[Dict[str, Any]] = []
        async with self.neo4j_client.get_async_session() as session:
            if "LIMIT" not in cypher.upper():
                cypher += f" LIMIT {n}"
            result = await session.run(cypher)
            async for record in result:
                m_data = {
                    "movieId": record.get("movieId"),
                    "title": str(record.get("title")),
                    "plot": record.get("plot"),
                    "year": record.get("year"),
                    "imdbRating": record.get("imdbRating"),
                    "score": 1.0,
                }
                if m_data["movieId"]:
                    movies.append(m_data)
        return movies
