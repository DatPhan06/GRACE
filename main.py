import logging

import json
import time

import pandas as pd

import yaml

from evaluating.calculate_recall import calculate_recall
from evaluating.output_eval import evaluate
from utils.LangChain import *

from utils.LangChain.GenerativeAI import callLangChainLLMReranking, callLangChainLLMSummarization
from utils.LlamaIndex import *

from evaluating import *

from preprocessing import *

from tqdm import tqdm

from utils.LlamaIndex.LlamaIndexUtils import load_retriever
from utils.LlamaIndex.candidate_retriever import query_parse_output

from preprocessing.redial import read_process_data_class


# Config values
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


EMBEDDING_MODEL = config['EmbeddingModel']['gecko']
GENERATIVE_MODEL = config["LlamaModel"]["llama_3.3_70b"]
API_KEY = config["APIKey"]["TOGETHER_AI_API_KEY_1"]


if __name__ == "__main__":
    # Load config:
    # For Insired
    insp_chroma = config["VectorDB"]["insp_chroma_db_path"]
    insp_collection = config["VectorDB"]["insp_collection_name"]
    insp_movie = config['InspiredDataPath']['processed']['movie']
    insp_train_dialog = config['InspiredDataPath']['processed']['dialog']['train']
    insp_output = config['OutputPath']['inspired']

    # For Redial
    # redial_chroma = config["VectorDB"]["redial_chroma_db_path"]
    # redial_collection = config["VectorDB"]["redial_collection_name"]
    # redial_movie = config["RedialDataPath"]["processed"]["movie"]
    # redial_output = config["OutputPath"]["redial"]

    # n_sample: [100, 200, 300, 400, 500, 600]
    # k: [1, 5, 10, 50]
    n_sample = 100
    k = 50    

    retriever_engine = load_retriever(
        chromadb_path=insp_chroma,
        collection_name=insp_collection,
        embedding_model=EMBEDDING_MODEL,
        model=GENERATIVE_MODEL,
        api_key=API_KEY,
        n=n_sample,
    )

    # For Inspried
    with open(insp_train_dialog, "r", encoding="utf-8") as file:
        input_data = json.load(file)

    with open(insp_movie, "r", encoding="utf-8") as file:
        movie = [json.loads(line) for line in file if line.strip()]
    df_movie = pd.DataFrame(movie)

    # For Redial
    # input_data = read_process_data_class().train_data
    # df_movie = pd.read_csv(redial_movie)

    for index, conv in tqdm(enumerate(input_data[256:])):
        
        # For Redial
        # context = ""
        # recommend_item = ""
        # turn_recommend = -1
        # for turn_id in reversed(range(len(input_data[conv_id]))):
        #     if input_data[conv_id][turn_id]['is_recommend'] == 1:
        #         turn_recommend = turn_id
        #         recommend_item = input_data[conv_id][turn_id]['item_recommend_name']
        #         print(f"Recommend of this row is: {input_data[conv_id][turn_id]['item_recommend_id']}, name: {recommend_item}")
        #         break

        # for turn_id in range(turn_recommend):
        #     turn = input_data[conv_id][turn_id]
        #     if turn['senderWorkerId'] == turn['initiatorWorkerId']:
        #         context += "SEEKER: " + turn['convert_text'] + "\n"
        #     if turn['senderWorkerId'] != turn['initiatorWorkerId']:
        #         context += "RECOMMENDER: " + turn['convert_text'] + "\n"
        
        # For Inspired
        conv_id = conv["conv_id"]
        context = conv["masked_dialog"]
        recommend_item = conv["target"]
        print(f"Conversation {conv_id}")

        summarized_conversation = callLangChainLLMSummarization(
            document=context, 
            model=GENERATIVE_MODEL, 
            api_key=API_KEY)["user_preferences"]

        print("Done summarizing")

        # Retrieve similar movie and concat to str list
        movie_candidate_list = query_parse_output(retriever_engine, df_movie, summarized_conversation)

        # Re-ranking:
        re_ranking_output = callLangChainLLMReranking(
            context=context,
            user_preferences=summarized_conversation,
            movie_str=movie_candidate_list,
            model=GENERATIVE_MODEL,
            api_key=API_KEY,
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
        
        # For Redial
        # calculate_recall(
        #     model_name=GENERATIVE_MODEL,
        #     response=re_ranking_output,
        #     conv_id=index,
        #     summarized_conversation=summarized_conversation,
        #     movie_candidate_list=movie_candidate_list,
        #     output_dir=redial_output,
        #     n=n_sample,
        #     k=k
        # )

        time.sleep(5)
