import json
import time

import pandas as pd

import yaml

import argparse

from evaluating.output_eval import evaluate
from utils.LangChain import *

from utils.LangChain.GenerativeAI import callLangChainLLMReranking, callLangChainLLMSummarization
from utils.LlamaIndex import *

from evaluating import *

from preprocessing import *

from tqdm import tqdm

from utils.LlamaIndex.candidate_retriever import query_parse_output

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



# API_KEY = config["APIKey"]["GOOGLE_API_KEY_2"]
GG_API_KEY = [
    config["APIKey"]["GOOGLE_API_KEY_0"],
    config["APIKey"]["GOOGLE_API_KEY_1"],
    config["APIKey"]["GOOGLE_API_KEY_2"],
    config["APIKey"]["GOOGLE_API_KEY_3"],
    config["APIKey"]["GOOGLE_API_KEY_4"],
    config["APIKey"]["GOOGLE_API_KEY_5"],
    config["APIKey"]["GOOGLE_API_KEY_6"],
    config["APIKey"]["GOOGLE_API_KEY_7"],
    config["APIKey"]["GOOGLE_API_KEY_8"],
    config["APIKey"]["GOOGLE_API_KEY_9"],
    config["APIKey"]["GOOGLE_API_KEY_10"],
    config["APIKey"]["GOOGLE_API_KEY_11"],
    config["APIKey"]["GOOGLE_API_KEY_12"],
    config["APIKey"]["GOOGLE_API_KEY_13"],
    config["APIKey"]["GOOGLE_API_KEY_14"],
    config["APIKey"]["GOOGLE_API_KEY_15"],
    config["APIKey"]["GOOGLE_API_KEY_16"],
    config["APIKey"]["GOOGLE_API_KEY_17"],
    config["APIKey"]["GOOGLE_API_KEY_18"],
    config["APIKey"]["GOOGLE_API_KEY_19"],
    config["APIKey"]["GOOGLE_API_KEY_20"],
    config["APIKey"]["GOOGLE_API_KEY_21"],
    config["APIKey"]["GOOGLE_API_KEY_22"],
    config["APIKey"]["GOOGLE_API_KEY_23"],
    config["APIKey"]["GOOGLE_API_KEY_24"],
    config["APIKey"]["GOOGLE_API_KEY_25"],
]

TOGETHR_API_KEY = [
    config["APIKey"]["TOGETHER_AI_API_KEY_0"],
    config["APIKey"]["TOGETHER_AI_API_KEY_1"],
]


if __name__ == "__main__":

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
        # n_sample = 100
        # k = 10
        
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
                movie_str=movie_candidate_list,
                model=GENERATIVE_MODEL,
                api_key=GG_API_KEY,
                k=k
            )
            print(re_ranking_output)
            print("Done re-ranking")
            

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
        redial_train_dialog = config["RedialDataPath"]["processed"]["dialog"]["train"]
        redial_movie = config["RedialDataPath"]["raw"]["movie"]
        redial_output = config["OutputPath"]["redial"]

        # n_sample: [100, 200, 300, 400, 500, 600]
        # k: [1, 5, 10, 50]
        # n_sample = 600
        # k = 50

        with open(redial_train_dialog, "r", encoding="utf-8") as file:
            input_data = json.load(file)

        movie = pd.read_csv(redial_movie, encoding="utf-8")
        df_movie = pd.DataFrame(movie)
        
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


            # Re-ranking:
            re_ranking_output = callLangChainLLMReranking(
                context=context,
                user_preferences=summarized_conversation,
                movie_str=movie_candidate_list,
                model=GENERATIVE_MODEL,
                api_key=GG_API_KEY,
                k=k
            )
            print(re_ranking_output)
            print("Done re-ranking")
            

            # Output evaluation
            # For Inspired
            evaluate(
                model_name=GENERATIVE_MODEL,
                re_ranked_list=re_ranking_output,
                recommend_item=recommend_item,
                conv_id=conv_id,
                summarized_preferences=summarized_conversation,
                movie_candidate_list=movie_candidate_list,
                output_dir=redial_output,
                n=n_sample,
                top_k=k
            )

        time.sleep(5)