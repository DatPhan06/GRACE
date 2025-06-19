import os
from typing import Literal, Union

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
from llama_index.llms.together import TogetherLLM
from llama_index.vector_stores.chroma import ChromaVectorStore


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
        llm_llama_index = Gemini(api_key=api_key, model_name=f"models/{model}", max_output_tokens=10000)

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
