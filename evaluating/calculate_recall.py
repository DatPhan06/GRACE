import pandas as pd
import os

def calculate_recall(
    model_name,
    response, 
    recommend_item, 
    conv_id, 
    summarized_conversation, 
    movie_candidate_list,
    output_dir,
    n,
    top_k
):
    output_dict = {}

    output = response["movie_list"].strip().replace("  ", " ")
    count_match_movie = 0
    recommend_movie_list = recommend_item.replace("  ", " ").split("|")
    for movie in recommend_movie_list:
        if movie in output:
            count_match_movie += 1
        elif movie == output:
            count_match_movie += 1
    recall = count_match_movie / len(recommend_movie_list)

    output_dict["recall"] = recall
    output_dict["row"] = conv_id
    output_dict["recommend_item"] = recommend_item
    output_dict["summarized_conversation"] = summarized_conversation
    output_dict["recommend_movie_list"] = output
    output_dict["movie_candidate_list"] = "[[[[" + movie_candidate_list
    # print(output_dict)
    pd.DataFrame.from_dict([output_dict]).to_csv(
        os.path.join(output_dir, f"{model_name.replace('/', '_')}_recall@{top_k}_{n}sample.tsv"),
        index=False,
        header=False,
        mode="a",
        sep="\t",
    )

    return output_dict
