from typing import List, Optional
from domain.retrieval.prompts import EXTRACT_GENRES_SYSTEM_PROMPT, EXTRACT_GENRES_USER_PROMPT
from shared.utils.logger import setup_logger
from infra.neo4j import get_neo4j_client
from infra.llm import get_llm_client

logger = setup_logger(__name__)

class GenreExtractor:
    """
    Component for extracting and matching genres from user preferences.
    """
    def __init__(self):
        self.neo4j_client = get_neo4j_client()
        self.llm_client = get_llm_client()
        self._cached_genres = None

    async def extract_genres(self, preferences: str) -> List[str]:
        """Extract genres using LLM and clean/match them against DB genres."""
        if not preferences:
            return []
        try:
            prompt = EXTRACT_GENRES_USER_PROMPT.format(preferences=preferences)
            response = await self.llm_client.agenerate(
                prompt=prompt,
                system_instruction=EXTRACT_GENRES_SYSTEM_PROMPT
            )
            extracted_text = response.strip().lower()
            if extracted_text == "none" or not extracted_text:
                return self._extract_genres_keyword_matching(preferences)
            genres = [genre.strip() for genre in extracted_text.split(',')]
            genres = [genre for genre in genres if genre and len(genre) > 0]
            return await self._match_genres_case_insensitive(genres)
        except Exception as e:
            logger.error(f"Error extracting genres with LLM: {e}")
            return self._extract_genres_keyword_matching(preferences)

    def _extract_genres_keyword_matching(self, preferences: str) -> List[str]:
        preferences_lower = preferences.lower()
        genre_keywords = {
            'action': ['action', 'fight', 'battle'],
            'comedy': ['comedy', 'funny', 'humor'],
            'drama': ['drama', 'emotional'],
            'horror': ['horror', 'scary', 'ghost'],
            'romance': ['romance', 'love'],
            'sci-fi': ['sci-fi', 'science fiction', 'space'],
            'fantasy': ['fantasy', 'magic'],
            'animation': ['animation', 'cartoon'],
        }
        found_genres = []
        for genre, keywords in genre_keywords.items():
            for keyword in keywords:
                if keyword in preferences_lower:
                    found_genres.append(genre)
                    break
        return found_genres

    async def _get_available_genres_from_db(self) -> List[str]:
        if self._cached_genres:
            return self._cached_genres
        try:
            async with self.neo4j_client.get_async_session() as session:
                result = await session.run("MATCH (g:Genre) RETURN DISTINCT g.name AS genre ORDER BY g.name")
                genres = [record['genre'] async for record in result]
                self._cached_genres = genres
                return genres
        except Exception as e:
            logger.error(f"Error fetching genres from database: {e}")
            return []

    async def _match_genres_case_insensitive(self, llm_genres: List[str]) -> List[str]:
        db_genres = await self._get_available_genres_from_db()
        if not db_genres:
            return llm_genres
        matched_genres = []
        for llm_genre in llm_genres:
            for db_genre in db_genres:
                if llm_genre.lower() == db_genre.lower():
                    matched_genres.append(db_genre.lower())
                    break
        return list(set(matched_genres))
