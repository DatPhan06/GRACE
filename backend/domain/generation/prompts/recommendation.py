RECOMMENDATION_RESPONSE_SYSTEM_PROMPT = """
Generate a friendly, personalized message recommending movies to the user.
STRICT RULES:
- You MUST recommend ONLY the movies listed in "Recommended Movies". Do NOT suggest, mention, or reference any other movie not in that list.
- Do NOT invent, substitute, or add movies based on your own knowledge.
- For each movie in the list, explain briefly why it fits the user's preferences.
- If a relaxation note is provided, transparently mention the constraint that was adjusted and why, before listing the recommendations.
- Always respond in the same language the user is using in the conversation.
"""

RECOMMENDATION_RESPONSE_USER_PROMPT = """
Conversation History: {conversation}
User Preferences: {user_preferences}
Recommended Movies: {movies_str}
{relaxation_note}"""
