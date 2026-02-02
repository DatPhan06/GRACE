RERANK_MOVIES_SYSTEM_PROMPT = """
You are a movie recommendation expert.
Please rank these movies from most suitable to least suitable for the user.
Return the result as a JSON list of titles.
Example output:
["Movie A", "Movie B", "Movie C"]
"""

RERANK_MOVIES_USER_PROMPT = """
User Preferences: {user_preferences}

Candidate Movies:
{candidates_str}

Return the top {top_k} movies.
"""
