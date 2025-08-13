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

load_dotenv()

class GraphRetriever:
    """
    A class to retrieve movies from Neo4j graph database using semantic search,
    content-based filtering, and collaborative filtering.
    """
    
    def __init__(self, uri: str, port: str, user: str, password: str, 
                 embedding_client: Optional[AzureOpenAI] = None,
                 embedding_deployment: Optional[str] = None):
        """
        Initialize graph retriever with Neo4j connection and embedding client.
        
        Args:
            uri: Neo4j URI
            port: Neo4j port
            user: Neo4j username
            password: Neo4j password
            embedding_client: Azure OpenAI client for embeddings
            embedding_deployment: Embedding model deployment name
        """
        self.driver = GraphDatabase.driver(f"{uri}:{port}", auth=(user, password))
        self.embedding_client = embedding_client
        self.embedding_deployment = embedding_deployment
        
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
        Retrieve movies using content-based filtering from graph relationships.
        
        Args:
            preferences: User preferences text
            n: Number of movies to retrieve
            
        Returns:
            List of movie dictionaries
        """
        if not self.driver:
            return []
            
        try:
            # Extract potential genres, actors, directors from preferences
            # This is a simple approach - you could make it more sophisticated
            preferences_lower = preferences.lower()
            
            # Common genres to search for
            genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'thriller', 
                     'sci-fi', 'fantasy', 'animation', 'documentary', 'mystery',
                     'adventure', 'crime', 'family', 'war', 'western', 'musical']
            
            found_genres = [genre for genre in genres if genre in preferences_lower]
            
            with self.driver.session() as session:
                if found_genres:
                    # Search by genre
                    result = session.run("""
                        MATCH (f:Film)-[:IN_GENRE]->(g:Genre)
                        WHERE toLower(g.name) IN $genres
                        OPTIONAL MATCH (f)-[:HAS_RATING]->(r:ImdbRating)
                        RETURN f.movieId AS movieId,
                               f.title AS title,
                               f.plot AS plot,
                               f.year AS year,
                               r.value AS imdbRating,
                               count(g) AS genreMatches
                        ORDER BY genreMatches DESC, r.value DESC
                        LIMIT $limit
                    """, genres=found_genres, limit=n)
                else:
                    # Fallback: get highly rated movies
                    result = session.run("""
                        MATCH (f:Film)-[:HAS_RATING]->(r:ImdbRating)
                        WHERE r.value >= 7.0
                        RETURN f.movieId AS movieId,
                               f.title AS title,
                               f.plot AS plot,
                               f.year AS year,
                               r.value AS imdbRating
                        ORDER BY r.value DESC
                        LIMIT $limit
                    """, limit=n)
                
                movies = []
                for record in result:
                    movies.append({
                        'movieId': record['movieId'],
                        'title': record['title'],
                        'plot': record['plot'],
                        'year': record['year'],
                        'imdbRating': record['imdbRating']
                    })
                    
                return movies
                
        except Exception as e:
            print(f"Error in content filtering: {e}")
            return []

    def get_movies_by_collaborative_filtering(self, n: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve movies using collaborative filtering based on graph relationships.
        
        Args:
            n: Number of movies to retrieve
            
        Returns:
            List of movie dictionaries
        """
        if not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                # Get movies that share many relationships (actors, directors, genres)
                result = session.run("""
                    MATCH (f:Film)
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
                """, limit=n)
                
                movies = []
                for record in result:
                    movies.append({
                        'movieId': record['movieId'],
                        'title': record['title'],
                        'plot': record['plot'],
                        'year': record['year'],
                        'imdbRating': record['imdbRating'],
                        'connectionScore': record['connectionScore']
                    })
                    
                return movies
                
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
            return []

    def retrieve_movies_hybrid(self, user_preferences: str, n: int = 100) -> List[str]:
        """
        Hybrid retrieval combining semantic search, content-based and collaborative filtering.
        
        Args:
            user_preferences: User preference text
            n: Number of movies to retrieve
            
        Returns:
            List of movie titles
        """
        all_movies = []
        
        # Method 1: Semantic similarity (if embeddings available)
        if self.embedding_client:
            query_embedding = self.get_query_embedding(user_preferences)
            if query_embedding:
                semantic_movies = self.get_movies_by_similarity(query_embedding, n//3)
                all_movies.extend(semantic_movies)
                print(f"Retrieved {len(semantic_movies)} movies via semantic similarity")
        
        # Method 2: Content-based filtering
        content_movies = self.get_movies_by_content_filter(user_preferences, n//3)
        all_movies.extend(content_movies)
        print(f"Retrieved {len(content_movies)} movies via content filtering")
        
        # Method 3: Collaborative filtering
        collab_movies = self.get_movies_by_collaborative_filtering(n//3)
        all_movies.extend(collab_movies)
        print(f"Retrieved {len(collab_movies)} movies via collaborative filtering")
        
        # Remove duplicates and extract titles
        seen_ids = set()
        unique_movies = []
        
        for movie in all_movies:
            movie_id = movie.get('movieId')
            if movie_id and movie_id not in seen_ids:
                seen_ids.add(movie_id)
                unique_movies.append(movie)
        
        # Sort by rating and return titles (handle None values)
        unique_movies.sort(key=lambda x: x.get('imdbRating') or 0, reverse=True)
        
        movie_titles = [movie['title'] for movie in unique_movies[:n] if movie.get('title')]
        
        print(f"Final unique movies retrieved: {len(movie_titles)}")
        return movie_titles


def query_parse_output_graph(
    df_movie: pd.DataFrame,
    summarized_preferences: str,
    data: str,
    n: int = 100
) -> List[str]:
    """
    Query the graph database with user preferences and parse movie recommendations.
    
    Args:
        df_movie: DataFrame containing movie metadata (for fallback)
        summarized_preferences: String containing user's movie preferences
        data: Dataset name ("inspired" or "redial")
        n: Number of movies to retrieve
        
    Returns:
        List of recommended movie titles
    """
    
    # Initialize Neo4j connection
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost")
    NEO4J_PORT = os.getenv("NEO4J_PORT", "7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # Initialize embedding client (same as graph_builder.py)
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
    
    # Create graph retriever
    retriever = GraphRetriever(NEO4J_URI, NEO4J_PORT, NEO4J_USER, NEO4J_PASSWORD,
                              embedding_client, embedding_deployment)
    
    try:
        # Retrieve movies using hybrid approach
        movie_titles = retriever.retrieve_movies_hybrid(summarized_preferences, n)
        
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
