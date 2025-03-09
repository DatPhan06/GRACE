import argparse
import pandas as pd
from IPython.display import Markdown, display
import json
from typing import List, Dict, Any, Literal, Optional, Union

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate as PromptTemplateLangchain

from utils.LangchainUtils import *
from utils.LlamaindexUtils import *

# Import preprocessing modules
from data_preprocessing.redial_dialog_data_preprocess import get_train_data as get_redial_data
from data_preprocessing.dialog_data_transform import transform_tsv_to_dialog_structure
from data_preprocessing.generate_embedding import create_embedding_db
from data_preprocessing.dialog_merge import insp_dialog_merge
from data_preprocessing.redial_dialog_merge import redial_dialog_merge

from utils.read_config import read_config

import os
import configparser

# Config values
config_value = read_config()


# pipeline: input data -> preprocess -> embedding -> converstion merging -> retrieval -> re-ranking

# New data processing pipeline functions
def process_dataset(dataset_type: Literal["redial", "inspired"], 
                    config_values: Dict[str, Any],
                    flag: bool = True) -> List[Any]:
    """
    Process the selected dataset (ReDial or Inspired)
    
    Args:
        dataset_type: Type of dataset to process ("redial" or "inspired")
        config_values: Dictionary of configuration values
        
    Returns:
        List of processed dialog data
    """
    if dataset_type == "redial":
        print(f"Processing ReDial dataset from: {config_values['redial_dialog_train_data_path']}")
        processed_data = get_redial_data(config_values['redial_dialog_train_data_path'])
        
    elif dataset_type == "inspired":
        print(f"Processing Inspired dataset from: {config_values['insp_dialog_train_data_path']}")
        processed_data = transform_tsv_to_dialog_structure(config_values['insp_dialog_train_data_path'])
        
    print(f"Processed {len(processed_data)} conversations")
    return processed_data


def create_embeddings(dataset_type: Literal["redial", "inspired"], 
                      config_values: Dict[str, Any],
                      flag: bool = True) -> None:
    """
    Create embeddings for movie data and store them in ChromaDB
    
    Args:
        dataset_type: Type of dataset for which to create embeddings
        config_values: Dictionary of configuration values
    """
    if dataset_type == "redial":
        print(f"Creating embeddings for ReDial movie data")
        db_path = config_values['redial_chroma_db_path']
        collection_name = config_values['redial_collection_name']
        movie_data_path = config_values['redial_movie_data_path']
        
    elif dataset_type == "inspired":
        print(f"Creating embeddings for Inspired movie data")
        db_path = config_values['insp_chroma_db_path']
        collection_name = config_values['insp_collection_name']
        movie_data_path = config_values['insp_movie_data_path']
    
    # Create embeddings using the appropriate movie data
    create_embedding_db(
        db_path=db_path,
        collection_name=collection_name,
        embedding_model=config_values['model_embedding'],
        api_key=config_values['google_api_key'],
        movie_data_path=movie_data_path
    )
    
    print(f"Embeddings created and stored at: {db_path}")


# def merge_conversations(processed_dialogs: List[Any], 
#                         dataset_type: Literal["redial", "inspired"]) -> List[str]:
#     """
#     Merge processed dialog turns into complete conversations
    
#     Args:
#         processed_dialogs: List of processed dialog data
#         dataset_type: Type of dataset being processed
        
#     Returns:
#         List of merged conversations as strings
#     """
#     merged_conversation = ""
#     rec_item = ""
    
#     if dataset_type == "redial":
#         merge_conversation, rec_item = redial_dialog_merge()
            
#     elif dataset_type == "inspired":
#         merge_conversation, rec_item = insp_dialog_merge()
        
#     return merge_conversation, rec_item


# def save_processed_data(merged_conversations: List[str], output_path: str) -> None:
#     """
#     Save the processed and merged conversations to a file
    
#     Args:
#         merged_conversations: List of conversation strings
#         output_path: Path where to save the output
#     """
#     # Create directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     # Save as JSON for flexibility
#     with open(output_path, 'w') as f:
#         json.dump(merged_conversations, f)
    
#     print(f"Saved {len(merged_conversations)} conversations to {output_path}")


# def run_pipeline(dataset_type: Literal["redial", "inspired"]) -> List[str]:
#     """
#     Run the full pipeline: input data -> preprocess -> embedding -> conversation merging
    
#     Args:
#         dataset_type: Type of dataset to process
        
#     Returns:
#         List of processed and merged conversations
#     """
#     # Get configuration
#     config_values = read_config()
    
#     # Step 1: Process dataset
#     processed_dialogs = process_dataset(dataset_type, config_values)
    
#     # Step 2: Create embeddings for movie data
#     create_embeddings(dataset_type, config_values)
    
#     # Step 3: Merge conversations
#     merged_conversations = merge_conversations(processed_dialogs, dataset_type)
    
#     # Save the processed data
#     output_path = config_values[f'{dataset_type}_output_path'] + f"/processed_conversations.json"
#     save_processed_data(merged_conversations, output_path)
    
#     return merged_conversations


# Original query parsing function - kept as is
def query_parse_output(retriever_engine, df_movie, output):
  streaming_response = retriever_engine.query(output)
  movie_name = []
  print(len(streaming_response.source_nodes))
  for idx in range(len(streaming_response.source_nodes)):
      movie_idx = int(streaming_response.source_nodes[idx].node.source_node.node_id)
      movie_name_idx = df_movie.loc[df_movie['movieId'] == movie_idx][
          'movieName'].iloc[0].replace("  ", " ")
      movie_name.append(movie_name_idx)

      print(f"{movie_name_idx} {movie_idx}")

  movie_str = "|".join(movie_name)

  return movie_str


# Original argument parsing function
def input_parse():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--end_row", type=int, default=5, help="End process row")
    parser.add_argument("--begin_row", type=int, default=0, help="Begin process row")
    parser.add_argument("--file_name", type=str, help="Begin process row")
    args = parser.parse_args()

    return args.begin_row, args.end_row, args.file_name


# New enhanced argument parsing function
def enhanced_input_parse():
    parser = argparse.ArgumentParser(description="Process conversation data and create embeddings.")
    parser.add_argument("--dataset", type=str, choices=["redial", "inspired"], required=True, 
                        help="Dataset to process: 'redial' or 'inspired'")
    parser.add_argument("--end_row", type=int, default=5, help="End process row")
    parser.add_argument("--begin_row", type=int, default=0, help="Begin process row")
    parser.add_argument("--file_name", type=str, help="Output file name")
    args = parser.parse_args()

    return args.dataset, args.begin_row, args.end_row, args.file_name


if __name__ == "__main__":
    # Original main execution code - commented out but preserved
    """
    # Define the input:
    begin_row, end_row, file_name = input_parse()

    # Load config:
    folder_path_output_model = config_value['file_path_output_model']
    file_path_output_model = f"{config_value['file_path_output_model']}\\output_{file_name}.csv"
    train_data_path = config_value['file_path_train']
    file_path_movie = config_value['file_path_input_movie']
    chromadb_path = config_value['chroma_db_path']
    collection_name = config_value['collection_name']
    embedding_model = config_value['model_embedding']
    infer_model = config_value['model_infer']
    gen_model = config_value['model_generate_data']
    api_key = config_value['google_api_key']
    top_k = config_value['top_k']
    
    # Create folder path if not exist:
    if not os.path.exists(folder_path_output_model):
        os.makedirs(folder_path_output_model)

    # Create Llama Index utils
    query_engine = load_embedding_db(chromadb_path=chromadb_path,
                                     collection_name=collection_name,
                                     embedding_model=embedding_model,
                                     infer_model=infer_model,
                                     api_key=api_key,)
    retriever_engine = load_retriever(chromadb_path=chromadb_path,
                                      collection_name=collection_name,
                                      embedding_model=embedding_model,
                                      infer_model=infer_model,
                                      api_key=api_key,
                                      top_k=top_k)
    """
    
    # New main execution code using the data processing pipeline
    # Parse command-line arguments with enhanced parser
    dataset_type, begin_row, end_row, file_name = enhanced_input_parse()
    
    # Run the data processing pipeline
    processed_conversations = run_pipeline(dataset_type)
    
    print(f"Pipeline completed. Processed {len(processed_conversations)} conversations.")
    
    # Load config - similar to original code but updated for dataset-specific paths
    folder_path_output_model = config_value[f'{dataset_type}_output_path']
    file_path_output_model = f"{folder_path_output_model}\\output_{file_name}.csv"
    
    # Set database paths based on the selected dataset
    if dataset_type == "redial":
        chromadb_path = config_value['redial_chroma_db_path']
        collection_name = config_value['redial_collection_name']
        movie_data_path = config_value['redial_movie_data_path']
    else:
        chromadb_path = config_value['insp_chroma_db_path']
        collection_name = config_value['insp_collection_name']
        movie_data_path = config_value['insp_movie_data_path']
    
    # Common settings
    embedding_model = config_value['model_embedding']
    infer_model = config_value['model_infer']
    gen_model = config_value['model_generate_data']
    api_key = config_value['google_api_key']
    top_k = config_value['top_k']
    
    # Create folder path if not exist:
    if not os.path.exists(folder_path_output_model):
        os.makedirs(folder_path_output_model)

    # Create Llama Index utils - same as original
    query_engine = load_embedding_db(chromadb_path=chromadb_path,
                                    collection_name=collection_name,
                                    embedding_model=embedding_model,
                                    infer_model=infer_model,
                                    api_key=api_key,)
    retriever_engine = load_retriever(chromadb_path=chromadb_path,
                                    collection_name=collection_name,
                                    embedding_model=embedding_model,
                                    infer_model=infer_model,
                                    api_key=api_key,
                                    top_k=top_k)
    
    # Now you can continue with retrieval and re-ranking using processed_conversations