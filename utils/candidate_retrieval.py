import pandas as pd

# Error log
import logging

# Request limit
import time

# Retriever Engine
from llama_index.core.query_engine import RetrieverQueryEngine


def query_parse_output(retriever_engine: RetrieverQueryEngine, 
                       df_movie: pd.DataFrame, 
                       summarized_preferences: str) -> str:
    """
    Query the retriever engine with user preferences and parse movie recommendations.
    
    Args:
        retriever_engine: LlamaIndex retriever query engine for semantic search
        df_movie: DataFrame containing movie metadata
        summarized_preferences: String containing user's movie preferences
        
    Returns:
        String containing pipe-separated list of recommended movie names
    """
    
    # Set maximum number of retry attempts
    max_retries = 100
    
    # Attempt to query the retriever with automatic retries on failure
    for attempt in range(max_retries):
        try:
            # Query the retriever engine with the user's preferences
            streaming_response = retriever_engine.query(summarized_preferences)
            
            # Print source nodes for debugging
            # /
            
            # Print the number of retrieved nodes
            print("Nodes:", len(streaming_response.source_nodes))
            
            # Exit the retry loop if successful
            break

        except Exception as e:
            # Log the failed attempt details
            print(f"Attempt {attempt+1} failed: {str(e)}")
            
            # Implement backoff strategy if not the last attempt
            if attempt < max_retries - 1:
                # Log retry information
                logging.error(f"Retrying in 5 seconds...")
                
                # Wait before retrying
                time.sleep(5)
            else:
                # Return empty dict if all retries fail
                print("All retries failed, returning fallback response")
                return {}

    # Initialize list to store retrieved movie names
    movie_name = []

    # Process each node in the retrieval results
    for idx in range(len(streaming_response.source_nodes)):
        try:
            # Extract movie ID from the source node
            movie_idx = streaming_response.source_nodes[idx].node.source_node.node_id
            
            # Look up the movie name in the DataFrame and clean it
            movie_name_idx = (
                df_movie.loc[df_movie["imdb_id"] == movie_idx]["title"]
                .iloc[0]
                .replace("  ", " ")
            )
            
            # Add movie name to the list
            movie_name.append(movie_name_idx)

        except Exception as e:
            # Log errors for movies that can't be found
            logging.error(f"{e}, movie {movie_idx} does not exist")
            
            # Continue to the next movie
            continue

    # Join all movie names with pipe separator
    movie_str = "|".join(movie_name)

    print("Done retrieving candidates")
    
    return movie_str