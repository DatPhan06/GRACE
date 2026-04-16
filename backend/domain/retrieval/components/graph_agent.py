import json
import logging
from typing import List, Dict, Any, Optional
from shared.utils.logger import setup_logger
from infra.llm import get_llm_client
from infra.neo4j import get_neo4j_client
from domain.retrieval.prompts.graph import (
    NEO4J_SCHEMA,
    GRAPH_REASONING_SYSTEM_PROMPT,
    GRAPH_REASONING_USER_PROMPT
)

logger = setup_logger(__name__)

class GraphReasoningAgent:
    """
    Multi-Agent style Graph Reasoner. 
    It translates intents to Cypher, executes it, and reflects if results are missing.
    """
    def __init__(self):
        self.llm_client = get_llm_client()
        self.neo4j_client = get_neo4j_client()
        self.max_iterations = 3

    async def retrieve(self, user_preferences: str, liked_movies: List[str], n: int = 100, hard_constraints: List[str] = None) -> Dict[str, Any]:
        """
        Returns:
            Dict containing 'movies' (List) and 'thoughts' (List of strings)
        """
        history = ""
        thoughts_trace = []
        system_prompt = GRAPH_REASONING_SYSTEM_PROMPT.format(schema=NEO4J_SCHEMA)
        
        for iteration in range(self.max_iterations):
            logger.info(f"[Graph Agent] Turn {iteration + 1} for preferences: {user_preferences[:50]}...")
            
            prompt = GRAPH_REASONING_USER_PROMPT.format(
                user_preferences=user_preferences,
                hard_constraints=", ".join(hard_constraints) if hard_constraints else "None",
                liked_movies=", ".join(liked_movies) if liked_movies else "None",
                history=history if history else "No previous attempts."
            )

            try:
                # 1. Thought & Action (Cypher Generation)
                response = await self.llm_client.agenerate(
                    prompt=prompt,
                    system_instruction=system_prompt
                )
                
                # Parse JSON safely
                cleaned_response = response.replace("```json", "").replace("```", "").strip()
                start = cleaned_response.find("{")
                end = cleaned_response.rfind("}") + 1
                if start == -1 or end <= 0:
                    logger.error(f"[Graph Agent] Invalid LLM output format: {cleaned_response}")
                    break
                    
                data = json.loads(cleaned_response[start:end])
                thought = data.get("thought", "")
                cypher = data.get("cypher", "")
                
                if not cypher:
                    logger.error(f"[Graph Agent] No Cypher generated.")
                    break
                    
                logger.info(f"[Graph Agent] Thought: {thought}")
                thoughts_trace.append(thought)
                logger.info(f"[Graph Agent] Executing Cypher: {cypher}")

                # 2. Observation (Execute Cypher)
                movies = []
                async with self.neo4j_client.get_async_session() as session:
                    # Enforce LIMIT to n if not constrained
                    if "LIMIT" not in cypher.upper():
                        cypher += f" LIMIT {n}"
                        
                    result = await session.run(cypher)
                    async for record in result:
                        m_data = {
                            'movieId': record.get('movieId'),
                            'title': str(record.get('title')),
                            'plot': record.get('plot'),
                            'year': record.get('year')
                        }
                        if m_data['movieId']:
                            # Give max score since it perfectly matched graph constraints
                            m_data['score'] = 1.0 
                            movies.append(m_data)

                if len(movies) > 0:
                    logger.info(f"[Graph Agent] Success! Retrieved {len(movies)} movies.")
                    return {"movies": movies[:n], "thoughts": thoughts_trace}
                else:
                    logger.info(f"[Graph Agent] Observation: Cypher returned 0 results. Will reflect and retry.")
                    history += f"Attempt {iteration + 1}:\nThought: {thought}\nCypher: {cypher}\nResult: 0 records found. The constraint was too strict.\n\n"
                    
            except Exception as e:
                logger.error(f"[Graph Agent] Error during reasoning loop: {e}")
                history += f"Attempt {iteration + 1} Error: {str(e)}\n\n"

        logger.warning(f"[Graph Agent] Failed to retrieve movies after {self.max_iterations} iterations.")
        return {"movies": [], "thoughts": thoughts_trace}
