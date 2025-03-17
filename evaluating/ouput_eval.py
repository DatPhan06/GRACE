import pandas as pd

# Path and directory handle
import os

# Type hints
from typing import Literal, Union


def output_eval(
    re_ranked_output: str,
    recommend_item: str,
    conv_id: Union[str, int],
    summarized_preferences: str,
    movie_candidate_list: str,
    top_k: int,
    output_dir: os.PathLike,
    top_rank: int,
    movieset: str = Literal["redialMovie", "inspMovie"],
) -> None:

    output_dict = {}

    output = re_ranked_output["movie_list"].strip().replace("  ", " ")
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
    output_dict["summarized_conversation"] = summarized_preferences
    output_dict["recommend_movie_list"] = output
    output_dict["movie_candidate_list"] = f"[{movie_candidate_list}]"

    pd.DataFrame.from_dict([output_dict]).to_csv(
        os.path.join(output_dir, f"output_{top_k}_{movieset}_remove_target_recall@{top_rank}.tsv"),
        index=False,
        header=False,
        mode="a",
        sep="\t",
    )
