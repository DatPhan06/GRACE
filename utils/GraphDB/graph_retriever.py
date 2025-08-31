import os
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import time
import random
from openai import AzureOpenAI
from infra.llm import create_gemini_langchain_llm

load_dotenv()

class GraphRetriever:
    """
    A class to retrieve movies from Neo4j graph database using semantic search,
    content-based filtering, and collaborative filtering.
    """
    
    def __init__(self, uri: str, port: str, user: str, password: str, 
                embedding_client: Optional[AzureOpenAI] = None,
                embedding_deployment: Optional[str] = None,
                llm_model: Optional[str] = None,
                llm_api_keys: Optional[List[str]] = None):
        """
        Initialize graph retriever with Neo4j connection and LLM clients.
        
        Args:
            uri: Neo4j URI
            port: Neo4j port
            user: Neo4j username
            password: Neo4j password
            embedding_client: Azure OpenAI client for embeddings
            embedding_deployment: Embedding model deployment name
            llm_model: LLM model name (e.g., "gemini-2.0-flash-exp")
            llm_api_keys: List of API keys for LLM
        """
        self.driver = GraphDatabase.driver(f"{uri}:{port}", auth=(user, password))
        self.embedding_client = embedding_client
        self.embedding_deployment = embedding_deployment
        self.llm_model = llm_model
        self.llm_api_keys = llm_api_keys or []
        self._current_api_key_index = 0
        self._cached_genres = None  # Cache for available genres
        
        try:
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j for retrieval.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def get_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """
        Get embedding for the query text using same method as graph_builder.py.
        
        Args:
            query_text: Text to embed (summarized conversation)
            
        Returns:
            List of embedding values or None if failed
        """
        if not query_text or not self.embedding_client:
            return None
            
        try:
            response = self.embedding_client.embeddings.create(
                input=query_text, 
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Could not get embedding for query: {e}")
            return None

    def _get_next_api_key(self) -> str:
        """Get next API key using round-robin strategy."""
        if not self.llm_api_keys:
            return ""
        
        api_key = self.llm_api_keys[self._current_api_key_index]
        self._current_api_key_index = (self._current_api_key_index + 1) % len(self.llm_api_keys)
        return api_key

    def extract_genres_with_llm(self, preferences: str) -> List[str]:
        """
        Use Gemini LLM to extract movie genres from user preferences/conversation.
        
        Args:
            preferences: User preferences text from conversation
            
        Returns:
            List of extracted genres
        """
        if not preferences or not self.llm_model or not self.llm_api_keys:
            print("LLM not configured, falling back to keyword matching")
            return self._extract_genres_keyword_matching(preferences)
            
        try:
            # Create a prompt to extract genres
            prompt = f"""
            Based on the following user conversation about movie preferences, extract the movie genres they are interested in.
            
            User conversation:
            {preferences}
            
            Available genres: action, comedy, drama, horror, romance, thriller, sci-fi, science fiction, fantasy, animation, documentary, mystery, adventure, crime, family, war, western, musical, biography, historical, psychological, supernatural, martial arts, sports, dance, music
            
            Instructions:
            - Return only the genre names that are clearly mentioned or strongly implied
            - Use lowercase
            - Separate multiple genres with commas
            - If sci-fi or science fiction is mentioned, return "sci-fi"
            - If no clear genres are found, return "none"
            
            Extracted genres:"""
            
            # Use Gemini LLM via LangChain
            api_key = self._get_next_api_key()
            llm = create_gemini_langchain_llm(
                api_key=api_key,
                model_name=self.llm_model,
                temperature=0.1,
                max_tokens=100
            )
            
            response = llm.invoke(prompt)
            extracted_text = response.content.strip().lower()
            
            if extracted_text == "none" or not extracted_text:
                print("LLM returned no genres, falling back to keyword matching")
                return self._extract_genres_keyword_matching(preferences)
            
            # Parse the extracted genres
            genres = [genre.strip() for genre in extracted_text.split(',')]
            genres = [genre for genre in genres if genre and len(genre) > 0]
            
            # Match with actual database genres (case-insensitive)
            matched_genres = self._match_genres_case_insensitive(genres)
            
            print(f"LLM extracted genres: {genres} -> Final matched: {matched_genres}")
            return matched_genres
                
        except Exception as e:
            print(f"Error extracting genres with LLM: {e}")
            
        # Fallback to keyword matching
        return self._extract_genres_keyword_matching(preferences)
    
    def _get_available_genres_from_db(self) -> List[str]:
        """
        Get all available genres from the Neo4j database (with caching).
        
        Returns:
            List of all genre names (both original case and lowercase)
        """
        # Return cached result if available
        if self._cached_genres is not None:
            return self._cached_genres
            
        if not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (g:Genre)
                    RETURN DISTINCT g.name AS genre
                    ORDER BY g.name
                """)
                
                genres = [record['genre'] for record in result]
                self._cached_genres = genres  # Cache the result
                print(f"Available genres from database: {genres[:10]}...")  # Show first 10
                return genres
                
        except Exception as e:
            print(f"Error fetching genres from database: {e}")
            return []

    def _match_genres_case_insensitive(self, llm_genres: List[str]) -> List[str]:
        """
        Match LLM-extracted genres with actual database genres (case-insensitive).
        
        Args:
            llm_genres: List of genres from LLM (lowercase)
            
        Returns:
            List of matched genres for database query
        """
        if not llm_genres:
            return []
            
        # Get available genres from database
        db_genres = self._get_available_genres_from_db()
        
        if not db_genres:
            # Fallback to lowercase if database query fails
            return llm_genres
        
        matched_genres = []
        
        for llm_genre in llm_genres:
            # Try exact match first (case-insensitive)
            for db_genre in db_genres:
                if llm_genre.lower() == db_genre.lower():
                    matched_genres.append(db_genre.lower())  # Use lowercase for query
                    break
            else:
                # Try partial matching for compound genres
                llm_words = llm_genre.lower().split()
                for db_genre in db_genres:
                    db_words = db_genre.lower().split()
                    if any(word in db_words for word in llm_words) or any(word in llm_words for word in db_words):
                        matched_genres.append(db_genre.lower())
                        break
        
        # Remove duplicates while preserving order
        unique_matched = []
        for genre in matched_genres:
            if genre not in unique_matched:
                unique_matched.append(genre)
        
        print(f"LLM genres: {llm_genres} -> Matched DB genres: {unique_matched}")
        return unique_matched

    def _extract_genres_keyword_matching(self, preferences: str) -> List[str]:
        """
        Fallback method: Extract genres using keyword matching.
        
        Args:
            preferences: User preferences text
            
        Returns:
            List of genres found via keyword matching
        """
        preferences_lower = preferences.lower()
        
        # Enhanced genre keywords mapping
        genre_keywords = {
            'action': ['action', 'fight', 'battle', 'explosion', 'chase', 'martial arts'],
            'comedy': ['comedy', 'funny', 'humor', 'laugh', 'hilarious', 'comic'],
            'drama': ['drama', 'dramatic', 'emotional', 'serious', 'touching'],
            'horror': ['horror', 'scary', 'fear', 'ghost', 'zombie', 'supernatural'],
            'romance': ['romance', 'romantic', 'love', 'relationship', 'dating'],
            'thriller': ['thriller', 'suspense', 'tension', 'mystery', 'psychological'],
            'sci-fi': ['sci-fi', 'science fiction', 'futuristic', 'space', 'alien', 'technology'],
            'fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'mythical', 'enchanted'],
            'animation': ['animation', 'animated', 'cartoon', 'anime'],
            'documentary': ['documentary', 'real life', 'factual', 'educational'],
            'mystery': ['mystery', 'detective', 'investigation', 'puzzle', 'clue'],
            'adventure': ['adventure', 'explore', 'journey', 'quest', 'expedition'],
            'crime': ['crime', 'criminal', 'heist', 'gangster', 'police', 'detective'],
            'family': ['family', 'kids', 'children', 'wholesome', 'all ages'],
            'war': ['war', 'military', 'battle', 'soldier', 'combat', 'battlefield'],
            'western': ['western', 'cowboy', 'frontier', 'wild west'],
            'musical': ['musical', 'music', 'song', 'dance', 'singing'],
            'biography': ['biography', 'biographical', 'true story', 'based on'],
            'sports': ['sports', 'football', 'basketball', 'baseball', 'athlete', 'competition']
        }
        
        found_genres = []
        for genre, keywords in genre_keywords.items():
            for keyword in keywords:
                if keyword in preferences_lower:
                    if genre not in found_genres:
                        found_genres.append(genre)
                    break
        
        print(f"Keyword matching found genres: {found_genres}")
        
        # Match with actual database genres (case-insensitive)
        matched_genres = self._match_genres_case_insensitive(found_genres)
        
        return matched_genres

    def get_movies_by_similarity(self, query_embedding: List[float], n: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve movies using vector similarity search on plot embeddings.
        
        Args:
            query_embedding: Query vector embedding
            n: Number of movies to retrieve
            
        Returns:
            List of movie dictionaries with similarity scores
        """
        if not self.driver or not query_embedding:
            return []
            
        try:
            with self.driver.session() as session:
                # Use Neo4j vector similarity search - using simple dot product
                result = session.run("""
                    MATCH (f:Film)
                    WHERE f.plot_embedding IS NOT NULL
                    OPTIONAL MATCH (f)-[:HAS_RATING]->(r:ImdbRating)
                    WITH f, r,
                            reduce(dot = 0.0, i IN range(0, size(f.plot_embedding)-1) | 
                             dot + f.plot_embedding[i] * $queryEmbedding[i]) AS similarity
                    RETURN f.movieId AS movieId, 
                            f.title AS title,
                            f.plot AS plot,
                            f.year AS year,
                            r.value AS imdbRating,
                            similarity
                    ORDER BY similarity DESC
                    LIMIT $limit
                """, queryEmbedding=query_embedding, limit=n)
                
                movies = []
                for record in result:
                    movies.append({
                        'movieId': record['movieId'],
                        'title': record['title'],
                        'plot': record['plot'],
                        'year': record['year'],
                        'imdbRating': record['imdbRating'],
                        'similarity': record['similarity']
                    })
                    
                return movies
                
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def get_movies_by_content_filter(self, preferences: str, n: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve movies using content-based filtering with LLM-enhanced genre extraction.
        
        Args:
            preferences: User preferences text from conversation
            n: Number of movies to retrieve
            
        Returns:
            List of movie dictionaries
        """
        if not self.driver:
            return []
            
        try:
            # Extract genres using LLM (with fallback to keyword matching)
            found_genres = self.extract_genres_with_llm(preferences)
            
            with self.driver.session() as session:
                if found_genres:
                    print(f"Searching for movies with genres: {found_genres}")
                    # Search by LLM-extracted genres with case-insensitive matching
                    result = session.run("""
                        MATCH (f:Film)-[:IN_GENRE]->(g:Genre)
                        WHERE toLower(g.name) IN $genres
                        OPTIONAL MATCH (f)-[:HAS_RATING]->(r:ImdbRating)
                        WITH f, r, 
                            collect(DISTINCT g.name) AS movieGenres,
                            collect(DISTINCT toLower(g.name)) AS movieGenresLower,
                            size([genre IN $genres WHERE toLower(g.name) = genre]) AS genreMatches
                        RETURN f.movieId AS movieId,
                            f.title AS title,
                            f.plot AS plot,
                            f.year AS year,
                            r.value AS imdbRating,
                            genreMatches,
                            movieGenres,
                            movieGenresLower
                        ORDER BY genreMatches DESC, 
                                CASE WHEN r.value IS NOT NULL THEN r.value ELSE 5.0 END DESC,
                                f.year DESC
                        LIMIT $limit
                    """, genres=found_genres, limit=n)
                else:
                    # Enhanced fallback: get diverse highly rated movies
                    print("No specific genres found, using enhanced fallback strategy")
                    result = session.run("""
                        MATCH (f:Film)-[:HAS_RATING]->(r:ImdbRating)
                        WHERE r.value >= 6.5
                        OPTIONAL MATCH (f)-[:IN_GENRE]->(g:Genre)
                        WITH f, r, count(DISTINCT g) AS genreCount
                        RETURN f.movieId AS movieId,
                            f.title AS title,
                            f.plot AS plot,
                            f.year AS year,
                            r.value AS imdbRating,
                            genreCount
                        ORDER BY r.value DESC, genreCount DESC, f.year DESC
                        LIMIT $limit
                    """, limit=n)
                
                movies = []
                for record in result:
                    movie_dict = {
                        'movieId': record['movieId'],
                        'title': record['title'],
                        'plot': record['plot'],
                        'year': record['year'],
                        'imdbRating': record['imdbRating']
                    }
                    
                    # Add genre match info if available
                    if 'genreMatches' in record:
                        movie_dict['genreMatches'] = record['genreMatches']
                    if 'movieGenres' in record:
                        movie_dict['movieGenres'] = record['movieGenres']
                    
                    movies.append(movie_dict)
                    
                return movies
                
        except Exception as e:
            print(f"Error in enhanced content filtering: {e}")
            return []

    def get_movies_by_collaborative_filtering(self, n: int = 100, liked_movies: List[str] = [], 
                                            max_depth: int = 3, visited_movies: set = None) -> List[Dict[str, Any]]:
        """
        Retrieve movies using collaborative filtering based on liked movies and graph relationships.
        Uses recursive approach to find related movies if not enough found.
        
        Args:
            n: Number of movies to retrieve
            liked_movies: List of liked movie titles to find related movies
            max_depth: Maximum recursion depth to prevent infinite loops
            visited_movies: Set of movie IDs already processed (for recursion tracking)
            
        Returns:
            List of movie dictionaries
        """
        # Initialize visited_movies set if not provided (first call)
        if visited_movies is None:
            visited_movies = set()
        
        # Base case: if max_depth reached or no driver available
        if not self.driver or max_depth <= 0:
            return []
            
        print(f"[Depth {3-max_depth+1}] Finding movies related to: {liked_movies}")
        
        try:
            with self.driver.session() as session:
                all_related_movies = []
                current_liked_film_ids = []
                
                if liked_movies:
                    # Step 1: Find liked movies in the database by title
                    for movie_title in liked_movies:
                        result = session.run("""
                            MATCH (f:Film)
                            WHERE toLower(f.title) CONTAINS toLower($title) 
                                OR toLower($title) CONTAINS toLower(f.title)
                                AND NOT f.movieId IN $visitedIds
                            RETURN f.movieId AS movieId, f.title AS title
                            LIMIT 1
                        """, title=movie_title.strip(), visitedIds=list(visited_movies))
                        
                        for record in result:
                            if record['movieId'] not in visited_movies:
                                current_liked_film_ids.append(record['movieId'])
                                visited_movies.add(record['movieId'])
                                print(f"[Depth {3-max_depth+1}] Found seed movie: {record['title']} (ID: {record['movieId']})")
                    
                    if current_liked_film_ids:
                        # Step 2: Find movies connected through intermediate nodes (Actor, Director, Writer only)
                        # Limit the query by the remaining number of movies needed at this depth
                        limit_value = max(1, n - len(all_related_movies))
                        result = session.run("""
                            MATCH (liked:Film)
                            WHERE liked.movieId IN $likedIds
                            // Pre-collect candidate related films via allowed intermediates
                            OPTIONAL MATCH (liked)<-[:ACTED_IN]-(:Actor)-[:ACTED_IN]->(relA:Film)
                            OPTIONAL MATCH (liked)<-[:DIRECTED]-(:Director)-[:DIRECTED]->(relD:Film)
                            OPTIONAL MATCH (liked)<-[:WROTE]-(:Writer)-[:WROTE]->(relW:Film)
                            WITH liked, [x IN collect(DISTINCT relA) + collect(DISTINCT relD) + collect(DISTINCT relW) WHERE x IS NOT NULL] AS rels
                            UNWIND rels AS related
                            WITH liked, related
                            WHERE related IS NOT NULL
                                AND related.movieId IS NOT NULL
                                AND NOT related.movieId IN $visitedIds
                                AND NOT related.movieId IN $likedIds
                            // Count shared people per type between liked and related
                            OPTIONAL MATCH (liked)<-[:ACTED_IN]-(a:Actor)-[:ACTED_IN]->(related)
                            WITH liked, related, count(DISTINCT a) AS actorCount
                            OPTIONAL MATCH (liked)<-[:DIRECTED]-(d:Director)-[:DIRECTED]->(related)
                            WITH liked, related, actorCount, count(DISTINCT d) AS directorCount
                            OPTIONAL MATCH (liked)<-[:WROTE]-(w:Writer)-[:WROTE]->(related)
                            WITH liked, related, actorCount, directorCount, count(DISTINCT w) AS writerCount
                            WITH related, actorCount, directorCount, writerCount
                            OPTIONAL MATCH (related)-[:HAS_RATING]->(rating:ImdbRating)
                            WITH related, coalesce(rating.value, 5.0) AS imdbRating, actorCount, directorCount, writerCount
                            WHERE (actorCount + directorCount + writerCount) > 0
                            RETURN related.movieId AS movieId,
                                related.title AS title,
                                related.plot AS plot,
                                related.year AS year,
                                max(imdbRating) AS imdbRating,
                                sum(actorCount) AS actorCount,
                                sum(directorCount) AS directorCount,
                                sum(writerCount) AS writerCount,
                                (sum(actorCount) + sum(directorCount) + sum(writerCount)) AS connectionCount
                            ORDER BY connectionCount DESC, imdbRating DESC
                            LIMIT $limit
                        """, likedIds=current_liked_film_ids, visitedIds=list(visited_movies), limit=limit_value)
                        
                        new_related_movies = []
                        for record in result:
                            if record['movieId'] not in visited_movies:
                                connection_types = []
                                if record['actorCount'] and record['actorCount'] > 0:
                                    connection_types.append('ACTED_IN')
                                if record['directorCount'] and record['directorCount'] > 0:
                                    connection_types.append('DIRECTED')
                                if record['writerCount'] and record['writerCount'] > 0:
                                    connection_types.append('WROTE')
                                movie_data = {
                                    'movieId': record['movieId'],
                                    'title': record['title'],
                                    'plot': record['plot'],
                                    'year': record['year'],
                                    'imdbRating': record['imdbRating'],
                                    'connectionCount': record['connectionCount'],
                                    'actorCount': record['actorCount'],
                                    'directorCount': record['directorCount'],
                                    'writerCount': record['writerCount'],
                                    'connectionTypes': connection_types,
                                    'depth': 3 - max_depth + 1
                                }
                                all_related_movies.append(movie_data)
                                new_related_movies.append(movie_data)
                                visited_movies.add(record['movieId'])
                        
                        print(f"[Depth {3-max_depth+1}] Found {len(new_related_movies)} related movies")
                        
                        # Step 3: Recursive call if not enough movies found
                        if len(all_related_movies) < n and max_depth > 1 and new_related_movies:
                            remaining_needed = max(0, n - len(all_related_movies))
                            
                            # Use titles of newly found movies as seeds for next recursion (cap by remaining_needed)
                            new_seed_titles = [movie['title'] for movie in new_related_movies[:min(5, len(new_related_movies), remaining_needed)]]
                            
                            print(f"[Depth {3-max_depth+1}] Not enough movies ({len(all_related_movies)}/{n}), recursing with {len(new_seed_titles)} new seeds")
                            
                            # Recursive call with new seeds
                            recursive_movies = self.get_movies_by_collaborative_filtering(
                                n=remaining_needed,
                                liked_movies=new_seed_titles,
                                max_depth=max_depth - 1,
                                visited_movies=visited_movies.copy()
                            )
                            
                            # Add recursive results
                            for movie in recursive_movies:
                                if movie['movieId'] not in visited_movies:
                                    all_related_movies.append(movie)
                                    visited_movies.add(movie['movieId'])
                            
                            print(f"[Depth {3-max_depth+1}] Added {len(recursive_movies)} movies from recursion")
                
                # Fallback if still not enough movies (only at first depth level)
                if len(all_related_movies) < n and max_depth == 3:
                    additional_needed = n - len(all_related_movies)
                    print(f"[Depth {3-max_depth+1}] Using fallback for {additional_needed} additional movies")
                    
                    fallback_result = session.run("""
                        MATCH (f:Film)
                        WHERE NOT f.movieId IN $visitedIds
                        OPTIONAL MATCH (f)-[:HAS_RATING]->(r:ImdbRating)
                        OPTIONAL MATCH (f)-[:IN_GENRE]->(g:Genre)
                        OPTIONAL MATCH (a:Actor)-[:ACTED_IN]->(f)
                        OPTIONAL MATCH (d:Director)-[:DIRECTED]->(f)
                        WITH f, r.value AS imdbRating, 
                            count(DISTINCT g) AS genreCount,
                            count(DISTINCT a) AS actorCount,
                            count(DISTINCT d) AS directorCount
                        WHERE imdbRating IS NOT NULL
                        RETURN f.movieId AS movieId,
                            f.title AS title,
                            f.plot AS plot,
                            f.year AS year,
                            imdbRating,
                            (genreCount + actorCount + directorCount) AS connectionScore
                        ORDER BY connectionScore DESC, imdbRating DESC
                        LIMIT $limit
                    """, visitedIds=list(visited_movies), limit=additional_needed)
                    
                    for record in fallback_result:
                        if record['movieId'] not in visited_movies:
                            all_related_movies.append({
                                'movieId': record['movieId'],
                                'title': record['title'],
                                'plot': record['plot'],
                                'year': record['year'],
                                'imdbRating': record['imdbRating'],
                                'connectionScore': record['connectionScore'],
                                'depth': 0  # Fallback movies
                            })
                            visited_movies.add(record['movieId'])
                
                # Sort by depth (closer connections first) and rating
                all_related_movies.sort(key=lambda x: (x.get('depth', 0), 
                                                    -(x.get('imdbRating') or 5.0), 
                                                    -(x.get('connectionCount', 0))))
                
                print(f"[Depth {3-max_depth+1}] Returning {len(all_related_movies)} movies")
                return all_related_movies[:n]
                
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
            return []

    def _normalize_title(self, title: Optional[str]) -> str:
        """
        Normalize a movie title for duplicate detection.
        Lowercase and remove non-alphanumeric characters.
        """
        if not title:
            return ""
        normalized_chars = []
        for ch in title.lower():
            if ch.isalnum():
                normalized_chars.append(ch)
        return "".join(normalized_chars)

    def _get_fallback_movies(self, n: int, seen_ids: set, seen_titles: set) -> List[str]:
        """
        Get fallback movies without genre filtering, excluding already seen IDs.
        
        Args:
            n: Number of movies to retrieve
            seen_ids: Set of movie IDs to exclude
            seen_titles: Set of normalized movie titles to exclude
            
        Returns:
            List of movie titles
        """
        if not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (f:Film)-[:HAS_RATING]->(r:ImdbRating)
                    WHERE f.movieId IS NOT NULL 
                        AND f.title IS NOT NULL
                        AND NOT f.movieId IN $seenIds
                        AND r.value >= 6.0
                    RETURN f.movieId AS movieId, f.title AS title
                    ORDER BY r.value DESC
                    LIMIT $limit
                """, seenIds=list(seen_ids), limit=n)
                
                movies = []
                for record in result:
                    title = record['title']
                    normalized = self._normalize_title(title)
                    if normalized in seen_titles:
                        continue
                    movies.append(title)
                    seen_ids.add(record['movieId'])  # Update seen_ids
                    seen_titles.add(normalized)  # Track title duplicates as well
                    
                return movies
                
        except Exception as e:
            print(f"Error in fallback movie retrieval: {e}")
            return []

    def _get_random_popular_movies(self, n: int, seen_ids: set, seen_titles: set) -> List[str]:
        """
        Get random popular movies, excluding already seen IDs.
        
        Args:
            n: Number of movies to retrieve
            seen_ids: Set of movie IDs to exclude
            seen_titles: Set of normalized movie titles to exclude
            
        Returns:
            List of movie titles
        """
        if not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                # Get a larger pool first, then randomize
                result = session.run("""
                    MATCH (f:Film)
                    WHERE f.movieId IS NOT NULL 
                        AND f.title IS NOT NULL
                        AND NOT f.movieId IN $seenIds
                    OPTIONAL MATCH (f)-[:HAS_RATING]->(r:ImdbRating)
                    RETURN f.movieId AS movieId, f.title AS title, 
                            coalesce(r.value, 5.0) AS rating
                    ORDER BY rating DESC
                    LIMIT $poolSize
                """, seenIds=list(seen_ids), poolSize=n*3)
                
                movie_pool = []
                for record in result:
                    movie_pool.append({
                        'movieId': record['movieId'],
                        'title': record['title']
                    })
                
                # Randomly sample from the pool
                random.shuffle(movie_pool)
                
                movies = []
                for movie in movie_pool[:n]:
                    title = movie['title']
                    normalized = self._normalize_title(title)
                    if normalized in seen_titles:
                        continue
                    movies.append(title)
                    seen_ids.add(movie['movieId'])  # Update seen_ids
                    seen_titles.add(normalized)  # Track title duplicates as well
                    
                return movies
                
        except Exception as e:
            print(f"Error in random popular movie retrieval: {e}")
            return []

    def retrieve_movies_hybrid(self, user_preferences: str, n: int = 100, liked_movies: List[str] = []) -> List[str]:
        """
        Enhanced hybrid retrieval with LLM-based content filtering and fallback strategies.
        Ensures exactly n movies are returned.
        
        Args:
            user_preferences: User preference text from conversation
            n: Number of movies to retrieve
            liked_movies: List of liked movie titles
        Returns:
            List of movie titles (exactly n movies or as many as available)
        """
        all_movies = []
        
        # Method 1: Semantic similarity (if embeddings available)
        if self.embedding_client:
            query_embedding = self.get_query_embedding(user_preferences)
            if query_embedding:
                semantic_movies = self.get_movies_by_similarity(query_embedding, n//2)
                all_movies.extend(semantic_movies)
                print(f"Retrieved {len(semantic_movies)} movies via semantic similarity")
        
        # Method 2: Enhanced content-based filtering with LLM
        content_movies = self.get_movies_by_content_filter(user_preferences, n//2)
        all_movies.extend(content_movies)
        print(f"Retrieved {len(content_movies)} movies via LLM-enhanced content filtering")
        
        # Method 3: Recursive collaborative filtering based on liked movies
        collab_movies = self.get_movies_by_collaborative_filtering(n//2, liked_movies, max_depth=3)
        all_movies.extend(collab_movies)
        print(f"Retrieved {len(collab_movies)} movies via recursive collaborative filtering")
        
        # Remove duplicates across methods using both movieId and normalized title
        def sort_key(movie):
            rating = movie.get('imdbRating') or 5.0
            genre_bonus = movie.get('genreMatches', 0) * 0.5  # Bonus for genre matches
            return rating + genre_bonus

        unique_map = {}
        for movie in all_movies:
            movie_id = movie.get('movieId')
            movie_title = movie.get('title') or ""
            if movie_id:
                key = f"id:{movie_id}"
            else:
                key = f"title:{self._normalize_title(movie_title)}"
            if key not in unique_map or sort_key(movie) > sort_key(unique_map[key]):
                unique_map[key] = movie
        unique_movies = list(unique_map.values())
        
        # Sort by rating and genre matches (prioritize LLM-matched movies)
        unique_movies.sort(key=sort_key, reverse=True)
        
        movie_titles = [movie['title'] for movie in unique_movies if movie.get('title')]
        
        # If we don't have enough movies, try fallback strategies
        if len(movie_titles) < n:
            print(f"Not enough unique movies ({len(movie_titles)}/{n}), trying fallback strategies...")
            additional_needed = n - len(movie_titles)
            seen_ids = {m.get('movieId') for m in unique_movies if m.get('movieId')}
            seen_titles = {self._normalize_title(t) for t in movie_titles}
            
            # Fallback 1: Get more movies without genre filtering
            fallback_movies = self._get_fallback_movies(additional_needed, seen_ids, seen_titles)
            movie_titles.extend(fallback_movies)
            print(f"Added {len(fallback_movies)} fallback movies")
            
            # Fallback 2: If still not enough, get random popular movies
            if len(movie_titles) < n:
                remaining_needed = n - len(movie_titles)
                random_movies = self._get_random_popular_movies(remaining_needed, seen_ids, seen_titles)
                movie_titles.extend(random_movies)
                print(f"Added {len(random_movies)} random popular movies")
        
        # Ensure we return exactly n movies (or as many as available)
        final_movies = movie_titles[:n]
        print(f"Final unique movies retrieved: {len(final_movies)}/{n}")
        return final_movies


def query_parse_output_graph(
    df_movie: pd.DataFrame,
    summarized_preferences: str,
    data: str,
    liked_movies: List[str],
    n: int = 100,
    config: Dict[str, Any] = None
) -> List[str]:
    """
    Query the graph database with user preferences and parse movie recommendations.
    
    Args:
        df_movie: DataFrame containing movie metadata (for fallback)
        summarized_preferences: String containing user's movie preferences
        data: Dataset name ("inspired" or "redial")
        liked_movies: List of liked movie titles
        n: Number of movies to retrieve
        config: Configuration dictionary containing LLM and API settings
        
    Returns:
        List of recommended movie titles
    """
    
    # Initialize Neo4j connection
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost")
    NEO4J_PORT = os.getenv("NEO4J_PORT", "7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # Initialize embedding client
    embedding_client = None
    embedding_deployment = None
    
    try:
        embedding_api_key = os.getenv("EMBEDDING__KEY")
        embedding_api_version = os.getenv("EMBEDDING__API_VERSION")
        embedding_azure_endpoint = os.getenv("EMBEDDING__ENDPOINT")
        embedding_deployment = os.getenv("EMBEDDING__DEPLOYMENT_NAME")
        
        if all([embedding_api_key, embedding_api_version, embedding_azure_endpoint, embedding_deployment]):
            embedding_client = AzureOpenAI(
                api_key=embedding_api_key,
                api_version=embedding_api_version,
                azure_endpoint=embedding_azure_endpoint,
            )
    except Exception as e:
        print(f"Could not initialize embedding client: {e}")
    
    # Initialize LLM parameters from config
    llm_model = None
    llm_api_keys = []
    
    if config:
        try:
            # Get LLM model from config (same as main_graph.py)
            llm_model = config.get("GeminiModel", {}).get("2.0_flash")
            
            # Get API keys from config (same as main_graph.py)
            llm_api_keys = [
                config.get("APIKey", {}).get(f"GOOGLE_API_KEY_{i}") 
                for i in range(26)
                if config.get("APIKey", {}).get(f"GOOGLE_API_KEY_{i}")
            ]
            
            print(f"Initialized LLM: model={llm_model}, api_keys_count={len(llm_api_keys)}")
            
        except Exception as e:
            print(f"Could not initialize LLM from config: {e}")
    
    # Create graph retriever with LLM support
    retriever = GraphRetriever(
        NEO4J_URI, NEO4J_PORT, NEO4J_USER, NEO4J_PASSWORD,
        embedding_client, embedding_deployment,
        llm_model, llm_api_keys
    )
    
    try:
        # Retrieve movies using hybrid approach
        movie_titles = retriever.retrieve_movies_hybrid(summarized_preferences, n, liked_movies)
        
        if not movie_titles:
            print("No movies retrieved from graph database, using fallback...")
            # Fallback: return some popular movies from DataFrame
            if data == "inspired":
                fallback_movies = df_movie.head(n)['title'].tolist()
            elif data == "redial":
                fallback_movies = df_movie.head(n)['movieName'].tolist()
            else:
                fallback_movies = []
            
            return fallback_movies[:n]
        
        print("Done retrieving candidates from graph database")
        return movie_titles
        
    except Exception as e:
        print(f"Error in graph retrieval: {e}")
        return []
        
    finally:
        retriever.close()
