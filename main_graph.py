import json
import time

import pandas as pd

import yaml

import argparse

from evaluating.output_eval import evaluate
from utils.LangChain import *
from utils.LangChain.GenerativeAI_redial import callLangChainLLMReranking_redial as callLangChainLLMReranking
from utils.LangChain.GenerativeAI import callLangChainLLMSummarization
from utils.LlamaIndex import *

from evaluating import *

from preprocessing import *

from tqdm import tqdm

from utils.GraphDB.graph_retriever import query_parse_output_graph
from utils.config_loader import get_api_key_config

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

# Load API keys from environment variables
api_keys = get_api_key_config()

EMBEDDING_MODEL = config['EmbeddingModel']['gecko']
GENERATIVE_MODEL = config["GeminiModel"]["2.0_flash"]
# GENERATIVE_MODEL = config["LlamaModel"]["llama_3.3_70b"]



GG_API_KEY = [
    api_keys[f"GOOGLE_API_KEY_{i}"] 
    for i in range(26)
    if f"GOOGLE_API_KEY_{i}" in api_keys
]


TOGETHER_API_KEY = [
    api_keys[f"TOGETHER_AI_API_KEY_{i}"]
    for i in range(3)
    if f"TOGETHER_AI_API_KEY_{i}" in api_keys
]


if __name__ == "__main__":

    # Ex: python main.py --data inspired --k 10 --n 300 --begin_row 0
    data, k, n_sample, start = input_parse()
    
    # data = "inspired"  # "inspired" or "redial"

    if data == "inspired":
        # Load config:
        # For Inspired
        insp_chroma = config["VectorDB"]["insp_chroma_db_path"]
        insp_collection = config["VectorDB"]["insp_collection_name"]
        insp_movie = config['InspiredDataPath']['processed']['movie']
        insp_train_dialog = config['InspiredDataPath']['processed']['dialog']['train']
        insp_output = config['OutputPath']['inspired']

        # n_sample: [100, 200, 300, 400, 500, 600]
        # k: [1, 5, 10, 50]
        # n_sample = 600
        # k = 5

        # For Inspired
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
            
            # Extract liked movies from conversation context for INSPIRED
            # Since INSPIRED doesn't have explicit liked_movies field, we'll extract them from the conversation
            liked_movies = []
            try:
                # Try to extract movie mentions from the conversation
                # This is a simple approach - in practice, you might want to use LLM to extract movie titles
                import re
                # Look for movie titles mentioned in the conversation (basic pattern matching)
                movie_mentions = re.findall(r'[A-Z][a-zA-Z\s]+(?:\([0-9]{4}\))?', context)
                # Filter out common words and keep potential movie titles
                common_words = {'RECOMMENDER', 'SEEKER', 'Hi', 'There', 'What', 'types', 'movies', 'like', 'watch', 'Yes', 'No', 'Thanks', 'Thank', 'you'}
                liked_movies = [mention.strip() for mention in movie_mentions if mention.strip() not in common_words and len(mention.strip()) > 3][:5]  # Limit to 5 movies
            except Exception as e:
                print(f"Could not extract liked movies: {e}")
                liked_movies = []

            print(f"Extracted liked movies: {liked_movies}")

            # Retrieve similar movie using graph database (now with liked_movies like REDIAL)
            movie_candidate_list = query_parse_output_graph(df_movie, 
                                                        summarized_conversation, 
                                                        data, 
                                                        liked_movies=liked_movies,
                                                        n=n_sample,
                                                        config=config)


            # Re-ranking:
            re_ranking_output = callLangChainLLMReranking(
                context=context,
                user_preferences=summarized_conversation,
                movie_list=movie_candidate_list,
                movie_data_path="",
                model=GENERATIVE_MODEL,
                api_key=GG_API_KEY,
                k=k
            )
            print(re_ranking_output)
            print("Done re-ranking")
            
            # Output evaluation
            # For Inspired (now using same format as REDIAL)
            evaluate(
                model_name=GENERATIVE_MODEL,
                re_ranked_list=re_ranking_output,
                recommend_item=[recommend_item],
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
        redial_test_dialog = config["RedialDataPath"]["processed"]["dialog"]["test_with_liked_movies"]
        redial_movie = config["RedialDataPath"]["raw"]["movie"]
        redial_movie_processed = config["RedialDataPath"]["processed"]["movie"]
        redial_output = config["OutputPath"]["redial"]

        # n_sample: [100, 200, 300, 400, 500, 600]
        # k: [1, 5, 10, 50]
        # n_sample = 600
        # k = 50

        with open(redial_test_dialog, "r", encoding="utf-8") as file:
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
            liked_movies = conv["liked_movies"]
            print(f"Conversation {conv_id} - {liked_movies}")


            summarized_conversation = callLangChainLLMSummarization(
                document=context, 
                model=GENERATIVE_MODEL, 
                api_key=GG_API_KEY)["user_preferences"]

            print("Done summarizing")
            

            # Retrieve similar movie using graph database
            movie_candidate_list = query_parse_output_graph(df_movie, 
                                                            summarized_conversation, 
                                                            data, 
                                                            liked_movies=liked_movies,
                                                            n=n_sample,
                                                            config=config
                                                            )


            # Re-ranking:
            re_ranking_output = callLangChainLLMReranking(
                context=context,
                user_preferences=summarized_conversation,
                movie_list=movie_candidate_list,
                movie_data_path="",
                model=GENERATIVE_MODEL,
                api_key=GG_API_KEY,
                k=k
            )
            # print(re_ranking_output)
            print("Done re-ranking")
            

            # Output evaluation
            # For Redial
            evaluate(
                model_name=GENERATIVE_MODEL,
                re_ranked_list=re_ranking_output,
                recommend_item=[recommend_item],
                conv_id=conv_id,
                summarized_preferences=summarized_conversation,
                movie_candidate_list=movie_candidate_list,
                output_dir=redial_output,
                n=n_sample,
                top_k=k
            )

        time.sleep(5)