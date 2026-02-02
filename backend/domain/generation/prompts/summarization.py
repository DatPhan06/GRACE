SUMMARIZE_CONVERSATION_SYSTEM_PROMPT = """
This conversation is a discussion between a seeker and a recommender.
Read this conversation, find all the information about the seeker's preferences in movie, actor, genres, etc.
Also extract any specific movies the seeker has mentioned liking.

Return the result as a JSON object with the following format:
{
    "user_preferences": "Detailed summary of preferences...",
    "liked_movies": ["Movie 1", "Movie 2"]
}

Ensure valid JSON output only.
"""

SUMMARIZE_CONVERSATION_USER_PROMPT = """
The conversation: {conversation}
"""
