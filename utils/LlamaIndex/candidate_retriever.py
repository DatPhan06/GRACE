import pandas as pd

# Error log
import logging

# Request limit
import time

# Retriever Engine
from llama_index.core.query_engine import RetrieverQueryEngine

import os
from typing import Literal
import random

# Chorma Databasse for vector data storage
import chromadb

# Import LlamaIndex components for document indexing and retrieval
# Core indexing and response generation tools
from llama_index.core import VectorStoreIndex, get_response_synthesizer 
from llama_index.core.indices.vector_store import VectorIndexRetriever

# For retrieving relevant documents from vector store
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.gemini import GeminiEmbedding 
from llama_index.llms.gemini import Gemini
from google.api_core.exceptions import TooManyRequests
from llama_index.llms.together import TogetherLLM
from llama_index.vector_stores.chroma import ChromaVectorStore

current_index_key = 0

def load_retriever(
    chromadb_path: os.PathLike,
    collection_name: str,
    embedding_model: str,
    model: str,
    api_key: str,
    n: Literal[100, 200, 300, 400, 500, 600] = 100
) -> RetrieverQueryEngine:
    """
    Creates an advanced retriever-based query engine with configurable retrieval parameters.

    Args:
        chromadb_path: File path to ChromaDB database
        collection_name: Name of the collection in ChromaDB
        embedding_model: Model name for generating embeddings
        infer_model: Model name for inference/generation
        n: Number of results to retrieve for each query
        api_key: Google API key for authentication

    Returns:
        Advanced query engine with custom retrieval parameters
    """

    # Initialize the persistent ChromaDB client pointing to the specified path
    client = chromadb.PersistentClient(path=chromadb_path)

    # Get existing collection or create a new one if it doesn't exist
    chroma_collection = client.get_collection(name=collection_name)

    # Create a vector store wrapper around the ChromaDB collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Initialize Gemini embedding model with API key
    gemini_embedding_model = GeminiEmbedding(
        api_key="AIzaSyA3ssFZiquFJYz4Mi29rIFUE5SsdGzrwLA", 
        model_name=f"models/{embedding_model}"
    )

    # Create a vector store index using the vector store and embedding model
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=gemini_embedding_model)

    # Create a customized retriever that fetches n most similar documents
    retriever = VectorIndexRetriever(index=index, similarity_top_k=int(n))

    # Configure response synthesizer with the specified LLM
    if model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"]:
        # Initialize Google's generative AI with the specified model and API key
        llm_llama_index = Gemini(
            api_key=api_key, 
            model_name=f"models/{model}",
            max_output_tokens=10000,  # Tăng lên để xử lý 50 bộ phim
            temperature=0.7,
            top_p=0.95,
            top_k=50
        )

    elif model in [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    ]:
        llm_llama_index = TogetherLLM(api_key=api_key, model=model)

    response_synthesizer = get_response_synthesizer(llm=llm_llama_index)

    # Assemble query engine by combining retriever and response synthesizer
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    return query_engine



# def query_parse_output(
#     retriever_engine: RetrieverQueryEngine, df_movie: pd.DataFrame, summarized_preferences: str, data: str
# ) -> str:
#     """
#     Query the retriever engine with user preferences and parse movie recommendations.

#     Args:
#         retriever_engine: LlamaIndex retriever query engine for semantic search
#         df_movie: DataFrame containing movie metadata
#         summarized_preferences: String containing user's movie preferences

#     Returns:
#         String containing pipe-separated list of recommended movie names
#     """

#     # Set maximum number of retry attempts
#     max_retries = 100

#     # Attempt to query the retriever with automatic retries on failure
#     for attempt in range(max_retries):
#         try:
#             # Query the retriever engine with the user's preferences
#             streaming_response = retriever_engine.query(summarized_preferences)

#             # Print source nodes for debugging
#             # /

#             # Print the number of retrieved nodes
#             print("Nodes:", len(streaming_response.source_nodes))

#             # Exit the retry loop if successful
#             break

#         except Exception as e:
#             # Log the failed attempt details
#             print(f"Attempt {attempt+1} failed: {str(e)}")

#             # Implement backoff strategy if not the last attempt
#             if attempt < max_retries - 1:
#                 # Log retry information
#                 logging.error(f"Retrying in 5 seconds...")

#                 # Wait before retrying
#                 time.sleep(5)
#             else:
#                 # Return empty dict if all retries fail
#                 print("All retries failed, returning fallback response")
#                 return {}

#     # Initialize list to store retrieved movie names
#     movie_name = []

#     # Process each node in the retrieval results
    
#     # data: inspired/redial
#     for idx in range(len(streaming_response.source_nodes)):
#         if data == "inspired":
#             try:
#                 # Extract movie ID from the source node
#                 movie_idx = streaming_response.source_nodes[idx].node.source_node.node_id

#                 # Look up the movie name in the DataFrame and clean it
#                 movie_name_idx = df_movie.loc[df_movie["imdb_id"] == movie_idx]["title"].iloc[0].replace("  ", " ")

#                 # Add movie name to the list
#                 movie_name.append(movie_name_idx)

#             except Exception as e:
#                 # Log errors for movies that can't be found
#                 logging.error(f"{e}, movie {movie_idx} does not exist")

#                 # Continue to the next movie
#                 continue
        
#         elif data == "redial":
#             try:
#                 # Extract movie ID from the source node
#                 movie_idx = int(streaming_response.source_nodes[idx].node.source_node.node_id)

#                 # Look up the movie name in the DataFrame and clean it
#                 movie_name_idx = df_movie.loc[df_movie["movieId"] == movie_idx]["movieName"].iloc[0].replace("  ", " ")

#                 # Add movie name to the list
#                 movie_name.append(movie_name_idx)

#             except Exception as e:
#                 # Log errors for movies that can't be found
#                 logging.error(f"{e}, movie {movie_idx} does not exist")

#                 # Continue to the next movie
#                 continue

#     # Join all movie names with pipe separator
#     movie_str = "|".join(movie_name)

#     print("Done retrieving candidates")

#     return movie_str


def query_parse_output(
    df_movie: pd.DataFrame, 
    summarized_preferences: str, 
    data: str,
    chromadb_path: os.PathLike,
    collection_name: str,
    embedding_model: str,
    model: str,
    api_key: str,
    n: Literal[100, 200, 300, 400, 500, 600] = 100
) -> str:
    """
    Query the retriever engine with user preferences and parse movie recommendations.

    Args:
        retriever_engine: LlamaIndex retriever query engine for semantic search
        df_movie: DataFrame containing movie metadata
        summarized_preferences: String containing user's movie preferences

    Returns:
        String containing pipe-separated list of recommended movie names
    """

    # Initialize with random key index for load balancing
    global current_index_key
    current_index_key = random.randint(0, len(api_key) - 1)
    key_len = len(api_key)
    

    retriever_engine = load_retriever(chromadb_path, collection_name, embedding_model, model, api_key[current_index_key], n)
    # retriever_engine = load_retriever(chromadb_path, collection_name, embedding_model, model, api_key, n)
    
    max_retries = 100

    # Attempt to query the retriever with automatic retries on failure
    for attempt in range(max_retries):
        try:
            # Check token count before querying
            print("\nChecking token count...")
            try:
                # Estimate token count (rough estimate: 1 token ≈ 4 characters)
                estimated_tokens = len(summarized_preferences) // 4
                print(f"Estimated input tokens: {estimated_tokens}")
                
                if estimated_tokens > 6000:  # Giới hạn an toàn cho Gemini
                    print("Warning: Input text might be too long. Truncating...")
                    summarized_preferences = summarized_preferences[:24000]  # Giới hạn khoảng 6000 tokens
            except Exception as e:
                print("Error estimating tokens:", str(e))

            # Query the retriever engine with the user's preferences
            streaming_response = retriever_engine.query(summarized_preferences)

            # Print token usage information
            try:
                usage = streaming_response.metadata.get("usage", {})
                prompt_tokens = usage.get("promptTokenCount", 0)
                candidates_tokens = usage.get("candidatesTokenCount", 0)
                total_tokens = usage.get("totalTokenCount", 0)
                
                print("\nToken Usage:")
                print(f"Prompt tokens: {prompt_tokens}")
                print(f"Candidates tokens: {candidates_tokens}")
                print(f"Total tokens: {total_tokens}")
                
                # Kiểm tra nếu tổng token vượt quá giới hạn
                if total_tokens > 8000:  # Giới hạn an toàn cho Gemini
                    print("Warning: Total tokens exceeded safe limit!")
            except Exception as e:
                print("Could not get token usage information:", str(e))

            # Print the number of retrieved nodes
            print("Nodes:", len(streaming_response.source_nodes))

            # Exit the retry loop if successful
            break

        except TooManyRequests as e:
            logging.error(f"Attempt {attempt+1} failed. HTTP error occurred: {str(e)}")
            current_index_key = (current_index_key + 1) % len(api_key)
            key_len -= 1
            print(f"Switching to next API key : #{current_index_key} ({api_key[current_index_key]})")
            retriever_engine = load_retriever(chromadb_path, collection_name, embedding_model, model, api_key[current_index_key], n)
            
            if key_len == 0 and attempt < max_retries - 1:
                # We've cycled through all keys, wait longer before retrying
                print("Exhausted all API keys.")
                exit
            else:
                # Wait briefly before retrying with the new key
                print("Retrying with new API key in 5 seconds...")
                time.sleep(5)

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
        if data == "inspired":
            try:
                # Extract movie ID from the source node
                movie_idx = streaming_response.source_nodes[idx].node.source_node.node_id

                # Look up the movie name in the DataFrame and clean it
                movie_name_idx = df_movie.loc[df_movie["imdb_id"] == movie_idx]["title"].iloc[0].replace("  ", " ")

                # Add movie name to the list
                movie_name.append(movie_name_idx)

            except Exception as e:
                # Log errors for movies that can't be found
                logging.error(f"{e}, movie {movie_idx} does not exist")

                # Continue to the next movie
                continue
        
        elif data == "redial":
            try:
                # Extract movie ID from the source node
                movie_idx = int(streaming_response.source_nodes[idx].node.source_node.node_id)

                # Look up the movie name in the DataFrame and clean it
                movie_name_idx = df_movie.loc[df_movie["movieId"] == movie_idx]["movieName"].iloc[0].replace("  ", " ")

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