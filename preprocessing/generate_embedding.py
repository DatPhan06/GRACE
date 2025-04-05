# Parsing JSON data
import json

# For file and path operations
import os

# Vector database for storing embeddings
import chromadb

# Import LlamaIndex components for document processing and embedding
from llama_index.core import (
    # Document container for text data
    Document,
    # Context for managing storage backends
    StorageContext,
    # Index for vector retrieval operations
    VectorStoreIndex,
)

# Google's Gemini embedding model
from llama_index.embeddings.gemini import GeminiEmbedding

# ChromaDB integration
from llama_index.vector_stores.chroma import ChromaVectorStore

# Type hints
from typing import Dict, List


def get_document_string(df_current_movie: Dict[str, str]) -> str:
    """
    Creates a structured text representation of a movie for embedding.

    Args:
        df_current_movie: Dictionary containing movie metadata

    Returns:
        String containing formatted movie information for embedding
    """

    # Format all movie attributes into a structured string template
    # This template organizes movie information in a clear, consistent format
    embedding_string = f"""Embedding this movie with the information:
    - Title: {df_current_movie["title"]}.
    - Release Year: {df_current_movie["year"]}.
    - Country: {df_current_movie["country"]}.
    - Genre: {df_current_movie["genre"]}.
    - Duration: {df_current_movie["runtime"]}.
    - Writer: {df_current_movie["writer"]}.
    - Director: {df_current_movie["director"]}.
    - Cast: {df_current_movie["actors"]}.
    - Description: {df_current_movie["plot"]}.
    """

    return embedding_string


def get_document_list(movie_data_path: os.PathLike) -> List[Document]:
    """
    Processes movie data from a file into a list of Document objects.

    Args:
        movie_data_path: Path to the movie data file (JSONL format)

    Returns:
        List of LlamaIndex Document objects, each containing a movie's formatted data
    """
    # Initialize empty list to hold all movie data
    movie_data = []
    for line in open(movie_data_path, "r"):
        # Parse each line as JSON and add to list
        movie_data.append(json.loads(line))

    # Initialize empty list to hold Document objects
    documents = []
    for i in range(len(movie_data)):
        # Get current movie data
        df_current_movie = movie_data[i]

        # Skip entries with insufficient data
        # Check if movie has enough metadata
        if len(df_current_movie) > 2:
            # Create formatted text for the current movie
            embedding_string = get_document_string(df_current_movie)

            # Create Document with text and ID
            documents.append(
                Document(text=embedding_string, doc_id=str(df_current_movie["imdb_id"]))
            )

    return documents


def create_embedding_db(
    db_path: str,
    collection_name: str,
    embedding_model: str,
    api_key: str,
    movie_data_path: os.PathLike,
) -> None:
    """
    Creates vector embeddings for movies and stores them in a ChromaDB database.

    Args:
        db_path: Path where ChromaDB will be stored
        collection_name: Name for the ChromaDB collection
        embedding_model: Model name/identifier for generating embeddings
        api_key: Google API key for authentication
        movie_data_path: Path to the movie data file
    """

    # Get list of Document objects from movie data
    # Process movie data into Documents
    documents = get_document_list(movie_data_path=movie_data_path)

    # Initialize Gemini embedding model with API key
    gemini_embedding_model = GeminiEmbedding(
        api_key=api_key, model_name=f"models/{embedding_model}"
    )

    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path=db_path)

    # Create or get existing collection
    chroma_collection = client.get_or_create_collection(collection_name)

    # Create vector store using ChromaDB collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create storage context using the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create vector index from documents using the embedding model and storage context
    VectorStoreIndex.from_documents(
        # List of Document objects
        documents,
        # Storage context for persistence
        storage_context=storage_context,
        # Model to generate embeddings
        embed_model=gemini_embedding_model,
    )

