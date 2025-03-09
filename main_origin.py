import argparse
import pandas as pd

from data_preprocessing.dialog_data_transform import *
from data_preprocessing.dialog_merge import * 
from utils.LangchainUtils import *
from utils.LlamaindexUtils import *
from utils.read_config import read_config

from data_preprocessing import *


# Config values
config_value = read_config()

def calculate_recall(response, recommend_item, conv_id, summarized_conversation, movie_candidate_list, top_k):
    output_dict = {}

    output = response['movie_list'].strip().replace("  "," ")
    count_match_movie = 0
    recommend_movie_list = recommend_item.replace("  "," ").split("|")
    for movie in recommend_movie_list:
        if movie in output:
            count_match_movie += 1
        elif movie == output:
            count_match_movie += 1
    recall = count_match_movie / len(recommend_movie_list)

    output_dict['recall'] = recall
    output_dict['row'] = conv_id
    output_dict['recommend_item'] = recommend_item
    output_dict['summarized_conversation'] = summarized_conversation
    output_dict['recommend_movie_list'] = output
    output_dict['movie_candidate_list'] = f'[{movie_candidate_list}]'
    # print(output_dict)
    pd.DataFrame.from_dict([output_dict]).to_csv(os.path.join(file_path_output_model, f'output_{top_k}.tsv'), index=False, header=False, mode="a", sep='\t')

    return output_dict

def query_parse_output(retriever_engine, df_movie, output):
    max_retries = 100
    for attempt in range(max_retries):
        try:
            streaming_response = retriever_engine.query(output)
            print(streaming_response.source_nodes)
            
            print("Nodes:", len(streaming_response.source_nodes))
            # exit()
            
            movie_name = []
            
            # print(len(streaming_response.source_nodes))
            for idx in range(len(streaming_response.source_nodes)):
                movie_idx = streaming_response.source_nodes[idx].node.source_node.node_id
                movie_name_idx = df_movie.loc[df_movie['imdb_id'] == movie_idx]['title'].iloc[0].replace("  ", " ")
                movie_name.append(movie_name_idx)

                # print(f"{movie_name_idx} {movie_idx}")

            movie_str = "|".join(movie_name)

            print("Done retrieving candidates")
            return movie_str

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)  
            else:
                print("All retries failed, returning fallback response")
                return {}
            

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
    file_path_output_model = config_value['insp_output_path']
    train_data_path = config_value['processed_insp_dialog_train_data_path']
    file_path_movie = config_value['processed_insp_movie_data_path']
    chromadb_path = config_value['insp_chroma_db_path']
    collection_name = config_value['insp_collection_name']
    
    embedding_model = config_value['model_embedding']
    infer_model = config_value['model_infer']
    gen_model = config_value['model_generate_data']
    api_key = config_value['google_api_key']
    top_k = config_value['top_k']


    # Create Llama Index utils
    # query_engine = load_embedding_db(chromadb_path=chromadb_path,
    #                                  collection_name=collection_name,
    #                                  embedding_model=embedding_model,
    #                                  infer_model=infer_model,
    #                                  api_key=api_key,)
    
    retriever_engine = load_retriever(chromadb_path=chromadb_path,
                                      collection_name=collection_name,
                                      embedding_model=embedding_model,
                                      infer_model=infer_model,
                                      api_key=api_key,
                                      top_k=top_k)
    
    # retriever_engine = load_embedding_db(chromadb_path=chromadb_path,
    #                                      collection_name=collection_name,
    #                                      embedding_model=embedding_model,
    #                                      infer_model=infer_model,
    #                                      api_key=api_key)

    # Load data:
    # train_data = redial_dialog_data_transform(train_data_path=train_data_path)
    # df_movie = pd.read_csv(filepath_or_buffer=file_path_movie, delimiter='\t')
    file = open(train_data_path, 'r')
    train_data = json.load(file)
    train_data = train_data[447:]
    
    file = open(file_path_movie, 'r', encoding='utf-8')
    movie = data = [json.loads(line) for line in file if line.strip()]
    df_movie = pd.DataFrame(data)
    
    # Infer
    # for conv_id in range(len(train_data)):
    for conv in train_data:
        
        
        # if conv_id >= begin_row and conv_id <= end_row:
            
            # print(f"Start process recommend for row {conv_id}. Recall of this row: ", end=" ")

            # context_str = ""
            # recommend_item = ""
            # turn_recommend = -1
            # for turn_id in reversed(range(len(train_data[conv_id]))):
            #     if train_data[conv_id][turn_id]['is_recommend'] == 1:
            #         turn_recommend = turn_id
            #         recommend_item = train_data[conv_id][turn_id]['item_recommend_name']
            #         print(f"Recommend of this row is: {train_data[conv_id][turn_id]['item_recommend_id']}, name: {recommend_item}")
            #         break

            # for turn_id in range(turn_recommend):
            #     turn = train_data[conv_id][turn_id]
            #     if turn['senderWorkerId'] == turn['initiatorWorkerId']:
            #         context_str += "User: " + turn['convert_text'] + "\n"
            #         # print("User: " + turn['convert_text'] + "\n")
            #     if turn['senderWorkerId'] != turn['initiatorWorkerId']:
            #         context_str += "Assistant: " + turn['convert_text'] + "\n"
            #         # print("Assistant: " + turn['convert_text'] + "\n")
            # # Call llm to summaries conversation:
            

            ### kết thúc hội thoại chỗ nào?


            conv_id, context_str, recommend_item = insp_dialog_merge(dialog_data=conv)
            print(f'Done merging conversation {conv_id}')
            
            summarized_conversation = call_llm_summarize(document=context_str,
                                                            gen_model=gen_model,
                                                            api_key=api_key)['user_preferences']
            print("Done summarizing")

            # Retrieval similar movie and concat to str list
            movie_candidate_list = query_parse_output(retriever_engine, df_movie, summarized_conversation)
            
            
            # Re-ranking:
            re_ranking_output = call_llm_reranking(context_str=context_str, 
                                                    user_preferences=summarized_conversation, 
                                                    movie_str=movie_candidate_list,
                                                    gen_model=gen_model,
                                                    api_key=api_key)
            print("Done re-ranking")

            # Calculate Recall
            output_dict = calculate_recall(re_ranking_output, 
                                            recommend_item, 
                                            conv_id, 
                                            summarized_conversation, 
                                            movie_candidate_list,
                                            top_k=top_k)

            # nghiên cứu mà clear history đi, kiểu như dưới
            # ok chị 
            # Ừ, chị thấy nó đang lưu hết, thì chir được vài đoạn, mà sau đó còn bị nhiễu bởi các đoạn trước, 
            # chat_session = model.start_chat(history=[])

            # except:
            #     print(f"Error on row {conv_id}, passing...")
            #     continue