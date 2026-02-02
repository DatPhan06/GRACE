EXTRACT_GENRES_SYSTEM_PROMPT = """
Based on the following user conversation about movie preferences, extract the movie genres they are interested in.
Available genres: action, comedy, drama, horror, romance, thriller, sci-fi, science fiction, fantasy, animation, documentary, mystery, adventure, crime, family, war, western, musical, biography, historical, psychological, supernatural, martial arts, sports, dance, music

Instructions:
- Return only the genre names that are clearly mentioned or strongly implied
- Use lowercase
- Separate multiple genres with commas
- If sci-fi or science fiction is mentioned, return "sci-fi"
- If no clear genres are found, return "none"
"""

EXTRACT_GENRES_USER_PROMPT = """
User conversation:
{preferences}

Extracted genres:
"""
