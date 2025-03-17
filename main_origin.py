import argparse
import logging
import pandas as pd

from data_preprocessing.dialog_data_transform import *
from data_preprocessing.dialog_merge import *
from utils.LangchainUtils import *
from utils.LlamaindexUtils import *
from utils.read_config import read_config
from utils.candidate_retrieval import query_parse_output
from evaluating.ouput_eval import output_eval

from data_preprocessing import *


# Config values
config_value = read_config()


# def input_parse():
#     parser = argparse.ArgumentParser(description="Process some integers.")
#     parser.add_argument("--end_row", type=int, default=5, help="End process row")
#     parser.add_argument("--begin_row", type=int, default=0, help="Begin process row")
#     parser.add_argument("--file_name", type=str, help="Begin process row")
#     args = parser.parse_args()

#     return args.begin_row, args.end_row, args.file_name


if __name__ == "__main__":
    # Define the input:
    # begin_row, end_row, file_name = input_parse()

    # Load config:
    file_path_output_model = config_value["insp_output_path"]
    train_data_path = config_value["processed_insp_dialog_train_data_path"]
    file_path_movie = config_value["processed_insp_movie_data_path"]
    chromadb_path = config_value["insp_chroma_db_path"]
    collection_name = config_value["insp_collection_name"]

    embedding_model = config_value["model_embedding"]
    infer_model = config_value["model_infer"]
    gen_model = config_value["model_generate_data"]
    api_key = config_value["google_api_key"]
    top_k = config_value["top_k"]
    
    recall_k = 10

    # Create Llama Index utils
    # query_engine = load_embedding_db(chromadb_path=chromadb_path,
    #                                  collection_name=collection_name,
    #                                  embedding_model=embedding_model,
    #                                  infer_model=infer_model,
    #                                  api_key=api_key,)

    retriever_engine = load_retriever(
        chromadb_path=chromadb_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        infer_model=infer_model,
        api_key=api_key,
        top_k=top_k,
    )

    # retriever_engine = load_embedding_db(chromadb_path=chromadb_path,
    #                                      collection_name=collection_name,
    #                                      embedding_model=embedding_model,
    #                                      infer_model=infer_model,
    #                                      api_key=api_key)

    # Load data:
    # train_data = redial_dialog_data_transform(train_data_path=train_data_path)
    # train_data = train_data[:300]

    # df_movie = pd.read_csv(filepath_or_buffer=file_path_movie, delimiter='\t')
    # with open('dataset\preprocessed_data\INSPIRED\dialog_data\dialog_train_data_original_preprocessed.json', 'r') as f:
    #     train_data = json.load(f)

    file = open('dataset/preprocessed_data/INSPIRED/dialog_data/dialog_train_data.json', "r")
    train_data = json.load(file)
    train_data = train_data[651:]

    file = open(file_path_movie, "r", encoding="utf-8")
    movie = data = [json.loads(line) for line in file if line.strip()]
    df_movie = pd.DataFrame(data)

    # Infer
    # for conv_id in range(len(train_data)):
    for conv in train_data:

        conv_id, context_str, recommend_item = insp_dialog_merge(dialog_data=conv)
        print(f"Done merging conversation {conv_id}")
        # print(f'{context_str}\n')
        # continue

        summarized_conversation = call_llm_summarize(
            document=context_str, 
            gen_model=gen_model, 
            api_key=api_key,
            recommended_item=recommend_item)["user_preferences"]
        
        print("Done summarizing")

        # Retrieval similar movie and concat to str list
        movie_candidate_list = query_parse_output(retriever_engine, df_movie, summarized_conversation)

        # Re-ranking:
        re_ranking_output = call_llm_reranking(
            # context_str=context_str,
            user_preferences=summarized_conversation,
            movie_str=movie_candidate_list,
            k=recall_k,
            gen_model=gen_model,
            api_key=api_key,
        )
        print("Done re-ranking")

        # Output evaluation
        output_eval(
            re_ranked_output=re_ranking_output,
            recommend_item=recommend_item,
            conv_id=conv_id,
            summarized_preferences=summarized_conversation,
            movie_candidate_list=movie_candidate_list,
            top_k=top_k,
            output_dir=file_path_output_model,
            movieset="inspMovie",
            top_rank=recall_k
        )

        # nghiên cứu mà clear history đi, kiểu như dưới
        # ok chị
        # Ừ, chị thấy nó đang lưu hết, thì chir được vài đoạn, mà sau đó còn bị nhiễu bởi các đoạn trước,
        # chat_session = model.start_chat(history=[])

        # except:
        #     print(f"Error on row {conv_id}, passing...")
        #     continue
