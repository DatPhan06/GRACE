from infra.llm import get_llm_client
from domain.generation.prompts import (
    SUMMARIZE_CONVERSATION_SYSTEM_PROMPT,
    SUMMARIZE_CONVERSATION_USER_PROMPT,
    RECOMMENDATION_RESPONSE_SYSTEM_PROMPT,
    RECOMMENDATION_RESPONSE_USER_PROMPT,
    RELAX_CONSTRAINTS_SYSTEM_PROMPT,
    RELAX_CONSTRAINTS_USER_PROMPT
)
from pydantic import BaseModel, Field
import json
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)


class RetrievalWeights(BaseModel):
    w_sem: float = Field(default=0.4, ge=0, le=1)
    w_con: float = Field(default=0.3, ge=0, le=1)
    w_col: float = Field(default=0.3, ge=0, le=1)


class UserPreference(BaseModel):
    profiler_reasoning: str = Field(
        default="", description="Chain of thought reasoning from the Profiler Agent")
    user_preferences: str = Field(
        description="Summarized seeker's preferences")
    hard_constraints: list[str] = Field(
        default=[], description="Explicit hard constraints like 'after 2010', 'horror genre', etc.")
    genres: list[str] = Field(
        default=[], description="Explicit genre names extracted from user preferences")
    semantic_queries: list[str] = Field(
        default=[], description="List of sub-queries for semantic search")
    liked_movies: list[str] = Field(
        default=[], description="List of movies the user liked")
    dynamic_weights: RetrievalWeights = Field(
        default_factory=RetrievalWeights, description="Predicted weights for WRRF")


class GenerationService:
    def __init__(self):
        self.llm_client = get_llm_client()

    async def summarize_conversation(self, conversation: str) -> UserPreference:
        """
        Summarize the conversation to extract user preferences and liked movies.
        """
        logger.info(
            f"Summarizing conversation with length: {len(conversation)}")
        prompt = SUMMARIZE_CONVERSATION_USER_PROMPT.format(
            conversation=conversation)

        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=SUMMARIZE_CONVERSATION_SYSTEM_PROMPT
            )
            # Basic cleanup if markdown checks block it
            cleaned_response = response.replace(
                "```json", "").replace("```", "").strip()
            # Find the JSON object
            start = cleaned_response.find("{")
            end = cleaned_response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = cleaned_response[start:end]
                logger.info(f"[Profiler RAW] Parsed JSON before loading: {json_str[:300]}")
                data = json.loads(json_str)
                result = UserPreference(**data)
                # Normalize weights so they always sum to 1.0 (graceful degradation)
                w = result.dynamic_weights
                total = w.w_sem + w.w_con + w.w_col
                if total > 0 and abs(total - 1.0) > 1e-6:
                    result.dynamic_weights = RetrievalWeights(
                        w_sem=round(w.w_sem / total, 6),
                        w_con=round(w.w_con / total, 6),
                        w_col=round(w.w_col / total, 6),
                    )
                logger.info(f"[Profiler PARSED] weights={result.dynamic_weights.model_dump()}")
                return result
            else:
                logger.error(
                    f"Could not find JSON in response: {cleaned_response}")
                return UserPreference(user_preferences=response, liked_movies=[], dynamic_weights=RetrievalWeights())
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return UserPreference(user_preferences="", liked_movies=[], dynamic_weights=RetrievalWeights())

    async def generate_response(
        self,
        user_preferences: str,
        recommendations: list,
        relaxation_note: str = "",
    ) -> str:
        movies_str = ", ".join(
            [f"{m['title']} ({m.get('year', 'N/A')})" for m in recommendations]
        )
        note_block = f"Relaxation Note: {relaxation_note}" if relaxation_note else ""
        prompt = RECOMMENDATION_RESPONSE_USER_PROMPT.format(
            user_preferences=user_preferences,
            movies_str=movies_str,
            relaxation_note=note_block,
        )
        return await self.llm_client.agenerate(
            prompt=prompt,
            system_instruction=RECOMMENDATION_RESPONSE_SYSTEM_PROMPT,
        )

    async def relax_constraints(self, preferences: UserPreference, critic_reasoning: str) -> UserPreference:
        """
        Relax strict constraints if no candidates were found.
        """
        logger.info(f"Relaxing constraints based on critic feedback: {critic_reasoning}")
        prompt = RELAX_CONSTRAINTS_USER_PROMPT.format(
            user_preferences=preferences.user_preferences,
            hard_constraints=", ".join(preferences.hard_constraints),
            semantic_queries="; ".join(preferences.semantic_queries),
            critic_reasoning=critic_reasoning,
        )
        
        try:
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=RELAX_CONSTRAINTS_SYSTEM_PROMPT
            )
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            start = cleaned_response.find("{")
            end = cleaned_response.rfind("}") + 1
            if start != -1 and end != -1:
                data = json.loads(cleaned_response[start:end])
                return UserPreference(**data)
            return preferences
        except Exception as e:
            logger.error(f"Error during relaxation: {e}")
            return preferences
