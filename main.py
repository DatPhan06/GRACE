from typing import Literal
from openai import api_key
import pandas as pd

from utils.LangchainUtils import *
from utils.LlamaindexUtils import *
from utils.read_config import read_config

from data_preprocessing.generate_embedding import create_embedding_db
from data_preprocessing.dialog_data_transform import *
from data_preprocessing.dialog_merge import *

from evaluating.calculate_recall import calculate_recall


def transform_dialog_data(data: Literal['redial', 'inspired'],
                          config_value: Dict[str, Any]):
    """
    """    
    if data == "redial":
        print(f"Transform ReDial dataset from: {config_value['redial_dialog_train_data_path']}")
        transformed_dialog_data = redial_dialog_data_transform(config_value['redial_dialog_train_data_path'])
        
    elif data == "inspired":
        print(f"Transform Inspired dataset from: {config_value['insp_dialog_train_data_path']}")
        transformed_dialog_data = insp_dialog_data_transform(config_value['insp_dialog_train_data_path'])
        
    print(f"Processed {len(transformed_dialog_data)} conversations")
    
    return transformed_dialog_data


def preprocess_movie_data(data: Literal['redial', 'inspired'],
                          config_value: Dict[str, Any]):
    
    return
    
    
def create_embeddings(data: Literal["redial", "inspired"], 
                      config_value: Dict[str, Any]) -> None:
    """
    Create embeddings for movie data and store them in ChromaDB
    
    Args:
        dataset_type: Type of dataset for which to create embeddings
        config_value: Dictionary of configuration values
    """
    
    if data == "redial":
        print(f"Creating embeddings for ReDial movie data")
        db_path = config_value['redial_chroma_db_path']
        collection_name = config_value['redial_collection_name']
        movie_data_path = config_value['redial_movie_data_path']
        
    elif data == "inspired":
        print(f"Creating embeddings for Inspired movie data")
        db_path = config_value['insp_chroma_db_path']
        collection_name = config_value['insp_collection_name']
        movie_data_path = config_value['insp_movie_data_path']
    
    # Create embeddings using the appropriate movie data
    create_embedding_db(
        db_path=db_path,
        collection_name=collection_name,
        embedding_model=config_value['model_embedding'],
        api_key=config_value['google_api_key'],
        movie_data_path=movie_data_path
    )
    
    print(f"Embeddings created and stored at: {db_path}")
    

def retriever_engine(data: Literal["redial", "inspired"],
                     config_value: Dict[str, Any],
                     embedding_model: str,
                     infer_model: str,
                     top_k: Union[int, str],
                     api_key: str) -> RetrieverQueryEngine:
    """
    """
    if data == 'redial':
        chromadb_path = config_value['redial_chroma_db_path']
        collection_name = config_value['redial_collection_name']
        return load_retriever(chromadb_path=chromadb_path,
                              collection_name=collection_name,
                              embedding_model=embedding_model,
                              infer_model=infer_model,
                              top_k=top_k,
                              api_key=api_key)
        
    elif data == 'inspired':
        chromadb_path = config_value['insp_chroma_db_path']
        collection_name = config_value['insp_collection_name']
        return load_retriever(chromadb_path=chromadb_path,
                              collection_name=collection_name,
                              embedding_model=embedding_model,
                              infer_model=infer_model,
                              top_k=top_k,
                              api_key=api_key)


if __name__ == '__main__':
    # Load config
    config_value = read_config()
    
    embedding_model = config_value['model_embedding']
    infer_model = config_value['model_infer']
    gen_model = config_value['model_generate_data']
    api_key = config_value['google_api_key']
    top_k = config_value['top_k']
    
    # Define the dataset used
    data = 'inspired'
    
    transformed_data = transform_dialog_data(data=data,
                                             config_value=config_value)
    
    preprocess_movie_data()
    
    create_embeddings(data=data,
                      config_value=config_value)
    
    
    retriever = retriever_engine(data=data,
                                 config_value=config_value,
                                 embedding_model=embedding_model,
                                 infer_model=infer_model,
                                 top_k=top_k,
                                 api_key=api_key)
    
    for index, dialog in enumerate(transformed_data):
        try:
            conv_id, context_str, recommend_item = insp_dialog_merge(dialog_data=dialog)
            
            # Call llm to summaries conversation:
            summarized_conversation = call_llm_summarize(document=context_str,
                                                         gen_model=gen_model,
                                                         api_key=api_key)['user_preferences']
            print('Done summarizing')

            # Retrieval similar movie and concat to str list
            movie_candidate_list = query_parse_output(retriever_engine, df_movie, summarized_conversation)
            # print('Done retrieving similar movies')

            # Re-ranking:
            re_ranking_output = call_llm_reranking(context_str=context_str, 
                                                   user_preferences=summarized_conversation, 
                                                   movie_str=movie_candidate_list,
                                                   gen_model=gen_model,
                                                   api_key=api_key)
            # print('Done re-ranking')

            # Calculate Recall
            output_dict = calculate_recall(response=re_ranking_output, 
                                           recommend_item=recommend_item, 
                                           conv_id=conv_id, 
                                           summarized_conversation=summarized_conversation, 
                                           movie_candidate_list=movie_candidate_list)

            # print(output_dict.get('recall'))
            # print(output_dict) 
            
        except:
            print(f"Error on row {conv_id}, passing...")
            continue