import os
from typing import Any, Optional, Union

# Chorma Databasse for vector data storage  
import chromadb                

# Import LlamaIndex components for document indexing and retrieval
from llama_index.core import VectorStoreIndex, get_response_synthesizer     # Core indexing and response generation tools
from llama_index.core.indices.vector_store import VectorIndexRetriever      # For retrieving relevant documents from vector store
from llama_index.core.query_engine import RetrieverQueryEngine              # Engine to process queries using retriever
from llama_index.embeddings.gemini import GeminiEmbedding                   # Google's Gemini embedding model
from llama_index.llms.gemini import Gemini                                  # Google's Gemini language model
from llama_index.vector_stores.chroma import ChromaVectorStore              # ChromaDB integration for vector storage
from llama_index.core import PromptTemplate as PromptTemplateLlamaIndex     # For creating structured prompts


# def add_summary_to_prompt(query_engine: RetrieverQueryEngine, summary_str: str) -> RetrieverQueryEngine:
#     """
#     Enhances the query engine by incorporating user preference summaries into the prompt.
    
#     Args:
#         query_engine: LlamaIndex query engine to be modified
#         summary_str: String containing user preference summary
        
#     Returns:
#         Modified query engine with updated prompt template
#     """
#     # Define a template that includes the user preference summary
#     template = (
#         """You are an assistant for conversational recommender system.
#   You have a contextual conversation that is an exchange between the user and the assistant about the user's movie preferences.
#   The dialogue is formatted as exchanges that begin "User/Assistant" defines the turn of the user or assistant.
#   Conversation: {query_str} \n
#   Summary of user preference: {summary_str}
#   Read all the contextual conversation, extract user preferences and thereby recommend top 50 movie this user might like.
#   You should return only recommendation list separated by '|'.
#   """
#     )
    
#     # Create a LlamaIndex prompt template from the string template
#     llm_prompt = PromptTemplateLlamaIndex(template)
    
#     # Fill in the summary_str parameter while keeping query_str as a variable
#     llm_prompt = llm_prompt.partial_format(summary_str=summary_str)
    
#     # Update the query engine to use the new prompt template for response synthesis
#     query_engine.update_prompts(
#         {"response_synthesizer:text_qa_template": llm_prompt}
#     )

#     return query_engine


# def load_embedding_db(chromadb_path: os.PathLike, 
#                      collection_name: str, 
#                      embedding_model: str, 
#                      infer_model: str,
#                      api_key: str) -> RetrieverQueryEngine:
#     """
#     Loads or creates a vector database and configures a query engine.
    
#     Args:
#         chromadb_path: File path to ChromaDB database
#         collection_name: Name of the collection in ChromaDB
#         embedding_model: Model name for generating embeddings
#         infer_model: Model name for inference/generation
#         api_key: Google API key for authentication
        
#     Returns:
#         Configured query engine for searching and retrieving information
#     """
#     # Initialize the persistent ChromaDB client pointing to the specified path
#     client = chromadb.PersistentClient(path=chromadb_path)
    
#     # Get existing collection or create a new one if it doesn't exist
#     chroma_collection = client.get_or_create_collection(collection_name)
    
#     # Create a vector store wrapper around the ChromaDB collection
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
#     # Initialize Gemini embedding model with API key
#     gemini_embedding_model = GeminiEmbedding(api_key=api_key, model_name=embedding_model)
    
#     # Create a vector store index using the vector store and embedding model
#     index = VectorStoreIndex.from_vector_store(
#         vector_store,
#         embed_model=gemini_embedding_model
#     )
    
#     # Initialize Gemini language model for generating responses
#     llm_llama_index = Gemini(api_key=api_key, model_name=infer_model)
    
#     # Create a query engine from the index using the configured LLM
#     query_engine = index.as_query_engine(llm = llm_llama_index)
    
#     return query_engine


def load_retriever(chromadb_path: os.PathLike, 
                   collection_name: str, 
                   embedding_model: str, 
                   infer_model: str,
                   top_k: Union[int, str],
                   api_key: str) -> RetrieverQueryEngine:
    """
    Creates an advanced retriever-based query engine with configurable retrieval parameters.
    
    Args:
        chromadb_path: File path to ChromaDB database
        collection_name: Name of the collection in ChromaDB
        embedding_model: Model name for generating embeddings
        infer_model: Model name for inference/generation
        top_k: Number of top results to retrieve for each query
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
    gemini_embedding_model = GeminiEmbedding(api_key=api_key, model_name=embedding_model)
    
    # Create a vector store index using the vector store and embedding model
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=gemini_embedding_model
    )

    # Create a customized retriever that fetches top_k most similar documents
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=int(top_k)
    )

    # Configure response synthesizer with the specified LLM
    llm_llama_index = Gemini(api_key=api_key, 
                             model_name=infer_model)
    response_synthesizer = get_response_synthesizer(llm=llm_llama_index)

    # Assemble query engine by combining retriever and response synthesizer
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer)

    return query_engine