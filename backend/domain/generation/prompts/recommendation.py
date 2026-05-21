RECOMMENDATION_RESPONSE_SYSTEM_PROMPT = """
Generate a friendly, personalized message recommending these movies to the user, explaining why each fits their preferences.
If a relaxation note is provided, transparently mention the constraint that was adjusted and why, before listing the recommendations.
"""

RECOMMENDATION_RESPONSE_USER_PROMPT = """
Conversation History: {conversation}
User Preferences: {user_preferences}
Recommended Movies: {movies_str}
{relaxation_note}"""
