from infra.llm import get_llm_client
from pydantic import BaseModel, Field
import json
import logging

class UserPreference(BaseModel):
    user_preferences: str = Field(description="Summarized seeker's preferences")
    liked_movies: list[str] = Field(default=[], description="List of movies the user liked")

class GenerationService:
    def __init__(self):
        self.llm_client = get_llm_client()

    async def summarize_conversation(self, conversation: str) -> UserPreference:
        """
        Summarize the conversation to extract user preferences and liked movies.
        """
        prompt = f"""
        This conversation is a discussion between a seeker and a recommender.
        Read this conversation, find all the information about the seeker's preferences in movie, actor, genres, etc.
        Also extract any specific movies the seeker has mentioned liking.
        
        The conversation: {conversation}
        
        Return the result as a JSON object with the following format:
        {{
            "user_preferences": "Detailed summary of preferences...",
            "liked_movies": ["Movie 1", "Movie 2"]
        }}
        
        Ensure valid JSON output only.
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            # Basic cleanup if markdown checks block it
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            # Find the JSON object
            start = cleaned_response.find("{")
            end = cleaned_response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = cleaned_response[start:end]
                data = json.loads(json_str)
                return UserPreference(**data)
            else:
                logging.error("Could not find JSON in response")
                return UserPreference(user_preferences=response, liked_movies=[])
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return UserPreference(user_preferences="", liked_movies=[])

    async def generate_response(self, user_preferences: str, recommendations: list) -> str:
        """
        Generate a final response to the user based on recommendations.
        """
        movies_str = ", ".join([f"{m['title']} ({m.get('year', 'N/A')})" for m in recommendations])
        prompt = f"""
        User Preferences: {user_preferences}
        Recommended Movies: {movies_str}
        
        Generate a friendly message recommending these movies to the user, explaining why they fit their preferences.
        """
        return await self.llm_client.agenerate(prompt)
