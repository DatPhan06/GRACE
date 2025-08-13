import json
import time
import logging

import pandas as pd

import yaml

import argparse

from evaluating.output_eval import evaluate
from utils.LangChain import *
from utils.LangChain.GenerativeAI_redial_test import callLangChainLLMReranking_redial, callLangChainLLMSummarization_redial
from utils.LangChain.GenerativeAI import callLangChainLLMReranking, callLangChainLLMSummarization
from utils.LlamaIndex import *

from evaluating import *

from preprocessing import *

from tqdm import tqdm

from utils.LlamaIndex.candidate_retriever_hybrid import query_parse_output

# Import for embedding-based similarity
import numpy as np
from infra.llm import create_gemini_embedding

def embedding_cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    """
    try:
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    except Exception as e:
        logging.error(f"Error calculating embedding similarity: {str(e)}")
        return 0

def vector_rerank_candidates(movie_candidate_list, user_preferences, df_movie_info, embedding_model, api_key, top_k=100):
    """
    Use embedding-based vector similarity to rerank movie candidates and return top K movies.
    
    Args:
        movie_candidate_list: List of movie names from initial retrieval
        user_preferences: User preferences text
        df_movie_info: DataFrame containing detailed movie information with plots
        embedding_model: Embedding model name
        api_key: API key for embedding model
        top_k: Number of top movies to return (default 100)
        
    Returns:
        List of top K movie names based on embedding similarity
    """
    try:
        # Initialize Gemini embedding model using the helper function
        gemini_embedding = create_gemini_embedding(
            api_key=api_key if isinstance(api_key, str) else api_key[0],
            model_name=embedding_model
        )
        
        # Get embedding for user preferences
        print("Getting embedding for user preferences...")
        user_embedding = gemini_embedding.get_text_embedding(user_preferences)
        
        # Prepare movie information and calculate similarities
        movie_similarities = []
        
        for i, movie_name in enumerate(movie_candidate_list):
            try:
                # Clean movie name for matching
                clean_movie_name = movie_name.replace("  ", " ").strip()
                
                # Find matching movie in the detailed info DataFrame
                movie_match = df_movie_info[df_movie_info['title'].str.strip() == clean_movie_name]
                
                if not movie_match.empty:
                    movie_info = movie_match.iloc[0]
                    # Combine plot with other movie metadata for better similarity calculation
                    movie_text = f"{movie_info['plot']} {movie_info.get('genre', '')} {movie_info.get('director', '')} {movie_info.get('actors', '')}"
                else:
                    # If no detailed info found, use just the movie name
                    movie_text = clean_movie_name
                
                # Get embedding for movie text
                movie_embedding = gemini_embedding.get_text_embedding(movie_text)
                
                # Calculate similarity with user preferences
                similarity = embedding_cosine_similarity(user_embedding, movie_embedding)
                movie_similarities.append((clean_movie_name, similarity))
                
                # Print progress for every 50 movies
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(movie_candidate_list)} movies for similarity calculation...")
                    
            except Exception as e:
                logging.error(f"Error processing movie {movie_name}: {str(e)}")
                # Include movie with 0 similarity as fallback
                movie_similarities.append((movie_name, 0))
        
        if not movie_similarities:
            return movie_candidate_list[:top_k]  # Fallback to original list
        
        # Sort by similarity score (descending)
        movie_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K movies
        top_movies = [movie for movie, similarity in movie_similarities[:top_k]]
        
        print(f"Vector reranking: Selected top {len(top_movies)} movies from {len(movie_candidate_list)} candidates")
        print(f"Top 5 similarity scores: {[f'{movie}: {sim:.4f}' for movie, sim in movie_similarities[:5]]}")
        
        return top_movies
        
    except Exception as e:
        logging.error(f"Error in vector reranking: {str(e)}")
        # Fallback to returning first top_k movies
        return movie_candidate_list[:top_k]

def get_movie_plots_from_candidates(movie_candidate_list, df_movie_info):
    """
    Get plot information for candidate movies from movie_fix_year.json data
    
    Args:
        movie_candidate_list: List of movie names from candidates
        df_movie_info: DataFrame containing detailed movie information with plots
        
    Returns:
        List of strings containing "movie_name: plot" for each movie
    """
    movie_plots = []
    
    for movie_name in movie_candidate_list:
        try:
            # Clean movie name for matching
            clean_movie_name = movie_name.replace("  ", " ").strip()
            
            # Find matching movie in the detailed info DataFrame
            movie_match = df_movie_info[df_movie_info['title'].str.strip() == clean_movie_name]
            
            if not movie_match.empty:
                plot = movie_match.iloc[0]['plot']
                movie_plots.append(f"{clean_movie_name}: {plot}")
            else:
                # If no plot found, just add the movie name
                movie_plots.append(f"{clean_movie_name}: No plot available")
                
        except Exception as e:
            logging.error(f"Error getting plot for movie {movie_name}: {str(e)}")
            movie_plots.append(f"{movie_name}: No plot available")
            
    return movie_plots

def input_parse():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--data", type=str, help="Dataset name")
    parser.add_argument("--k", type=int, help="Top k movie")
    parser.add_argument("--n", type=int, help="Number of samples")
    parser.add_argument("--begin_row", type=int, help="Begin row")
    args = parser.parse_args()

    return args.data, args.k, args.n, args.begin_row

# Config values
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


EMBEDDING_MODEL = config['EmbeddingModel']['gecko']
GENERATIVE_MODEL = config["GeminiModel"]["2.0_flash"]
# GENERATIVE_MODEL = config["LlamaModel"]["llama_3.3_70b"]



GG_API_KEY = [
    config["APIKey"][f"GOOGLE_API_KEY_{i}"] 
    for i in range(26)
]


TOGETHER_API_KEY = [
    config["APIKey"][f"TOGETHER_AI_API_KEY_{i}"]
    for i in range(3)
]


if __name__ == "__main__":

    # Ex: python main.py --data inspired --k 10 --n 300 --begin_row 0
    data, k, n_sample, start = input_parse()
    
    # data = "inspired"  # "inspired" or "redial"

    if data == "inspired":
        # Load config:
        # For Insired
        insp_chroma = config["VectorDB"]["insp_chroma_db_path"]
        insp_collection = config["VectorDB"]["insp_collection_name"]
        insp_movie = config['InspiredDataPath']['processed']['movie']
        insp_train_dialog = config['InspiredDataPath']['processed']['dialog']['train']
        insp_output = config['OutputPath']['inspired']

        # n_sample: [100, 200, 300, 400, 500, 600]
        # k: [1, 5, 10, 50]
        # n_sample = 600
        # k = 5

        # retriever_engine = load_retriever(
        #     chromadb_path=insp_chroma,
        #     collection_name=insp_collection,
        #     embedding_model=EMBEDDING_MODEL,
        #     model=GENERATIVE_MODEL,
        #     api_key=API_KEY,
        #     n=n_sample,
        # )

        # For Inspried
        with open(insp_train_dialog, "r", encoding="utf-8") as file:
            input_data = json.load(file)

        with open(insp_movie, "r", encoding="utf-8") as file:
            movie = [json.loads(line) for line in file if line.strip()]
        df_movie = pd.DataFrame(movie)
        
        for index, conv in tqdm(enumerate(input_data[start:], start=start)):

            conv_id = f"{index} {conv['conv_id']}"
            context = conv["processed_dialog"]
            recommend_item = conv["target"]
            print(f"Conversation {conv_id}")

            summarized_conversation = callLangChainLLMSummarization(
                document=context, 
                model=GENERATIVE_MODEL, 
                api_key=GG_API_KEY)["user_preferences"]

            print("Done summarizing")
            

            # Retrieve similar movie and concat to str list
            movie_candidate_list = query_parse_output(df_movie, 
                                                      summarized_conversation, 
                                                      data, 
                                                      chromadb_path=insp_chroma, 
                                                      collection_name=insp_collection, 
                                                      embedding_model=EMBEDDING_MODEL, 
                                                      model=GENERATIVE_MODEL, 
                                                      api_key=GG_API_KEY, 
                                                      n=n_sample)


            # Re-ranking:
            re_ranking_output = callLangChainLLMReranking(
                context=context,
                user_preferences=summarized_conversation,
                movie_str="|".join(movie_candidate_list),
                model=GENERATIVE_MODEL,
                api_key=GG_API_KEY,
                k=k
            )
            print(re_ranking_output)
            print("Done re-ranking")
            
            # Ensure re_ranking_output is a dict for evaluate
            if not isinstance(re_ranking_output, dict):
                re_ranking_output = {}

            # Output evaluation
            # For Inspired
            evaluate(
                model_name=GENERATIVE_MODEL,
                re_ranked_list=re_ranking_output,
                recommend_item=recommend_item,
                conv_id=conv_id,
                summarized_preferences=summarized_conversation,
                movie_candidate_list=movie_candidate_list,
                output_dir=insp_output,
                n=n_sample,
                top_k=k
            )

        time.sleep(5)

    elif data == "redial":
        
        redial_chroma = config["VectorDB"]["redial_chroma_db_path"]
        redial_collection = config["VectorDB"]["redial_collection_name"]
        redial_train_dialog = config["RedialDataPath"]["processed"]["dialog"]["test"]
        redial_movie = config["RedialDataPath"]["raw"]["movie"]
        redial_movie_processed = config["RedialDataPath"]["processed"]["movie"]
        redial_output = config["OutputPath"]["redial_test"]

        # n_sample: [100, 200, 300, 400, 500, 600]
        # k: [1, 5, 10, 50]
        # n_sample = 600
        # k = 50

        with open(redial_train_dialog, "r", encoding="utf-8") as file:
            input_data = json.load(file)

        movie = pd.read_csv(redial_movie, encoding="utf-8")
        df_movie = pd.DataFrame(movie)
        
        # Load detailed metadata from processed JSON (movie_fix_year)
        with open(redial_movie_processed, "r", encoding="utf-8") as file:
            movie_info = [json.loads(line) for line in file if line.strip()]
        df_movie_info = pd.DataFrame(movie_info)

        
        for index, conv in tqdm(enumerate(input_data[start:], start=start)):

            conv_id = index
            context = conv["dialog"]
            recommend_item = conv["target"]
            print(f"Conversation {conv_id}")


            summarized_conversation = callLangChainLLMSummarization(
                document=context, 
                model=GENERATIVE_MODEL, 
                api_key=GG_API_KEY)["user_preferences"]

            print("Done summarizing")
            

            # Retrieve similar movie and concat to str list
            movie_candidate_list = query_parse_output(df_movie, 
                                                      summarized_conversation, 
                                                      data, 
                                                      chromadb_path=redial_chroma, 
                                                      collection_name=redial_collection, 
                                                      embedding_model=EMBEDDING_MODEL, 
                                                      model=GENERATIVE_MODEL, 
                                                      api_key=GG_API_KEY, 
                                                      n=n_sample)

            # Vector-based reranking to get top 100 movies
            print("Starting vector-based pre-ranking...")
            top_100_movies = vector_rerank_candidates(
                movie_candidate_list=movie_candidate_list,
                user_preferences=summarized_conversation,
                df_movie_info=df_movie_info,
                embedding_model=EMBEDDING_MODEL,
                api_key=GG_API_KEY,
                top_k=100
            )
            print("Done vector-based pre-ranking")

            # Get movie plots for the top 100 movies
            movie_plots_list = get_movie_plots_from_candidates(top_100_movies, df_movie_info)

            # Re-ranking:
            re_ranking_output = callLangChainLLMReranking_redial(
                context=context,
                user_preferences=summarized_conversation,
                movie_str="|".join(movie_plots_list),
                model=GENERATIVE_MODEL,
                api_key=GG_API_KEY,
                k=k
            )
            print(re_ranking_output)
            print("Done re-ranking")
            
            # Ensure re_ranking_output is a dict for evaluate
            if not isinstance(re_ranking_output, dict):
                re_ranking_output = {}

            # Output evaluation
            # For REDIAL - using top_100_movies for evaluation
            evaluate(
                model_name=GENERATIVE_MODEL,
                re_ranked_list=re_ranking_output,
                recommend_item=[recommend_item],
                conv_id=conv_id,
                summarized_preferences=summarized_conversation,
                movie_candidate_list=top_100_movies,  # Use the vector pre-ranked list
                output_dir=redial_output,
                n=n_sample,
                top_k=k
            )

        time.sleep(5)