import pandas as pd
import os


def standardize_movie_title(movie_title: str) -> str:
    """Normalizes movie titles for consistent comparison."""
    # Remove year and special characters, convert to lowercase
    title = re.sub(r"\s*\(\d{4}\)", "", movie_title)  # Remove year anywhere in title
    title = re.sub(r"[^\w\s]", "", title)  # Remove punctuation
    return title.strip().lower()  # Normalize case and whitespace


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

    # output = response["movie_list"].strip().replace("  ", " ")
    output = [standardize_movie_title(m) for m in response["movie_list"] if m.strip()]
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
