import pandas as pd

# Path and directory handle
import os

# Type hints
from typing import List, Literal, Union

import re


# def standardize_movie_title(movie_title):
#     """
#     Removes the year from a movie title formatted as "Movie (year)".

#     Args:
#       movie_title: The movie title string.

#     Returns:
#       The movie title without the year, or the original string if no year is found.
#     """
#     pattern = r"\s*\(\d{4}\)$"  # Matches " (year)" at the end of the string
#     return re.sub(pattern, "", movie_title)


# def evaluate(
#     model_name: str,
#     re_ranked_list: str,
#     recommend_item: List[str],
#     conv_id: Union[str, int],
#     movie_candidate_list: str,
#     summarized_preferences: str,
#     output_dir: os.PathLike,
#     n: Literal[100, 200, 300, 400, 500, 600] = 100,
#     top_k: Literal[1, 5, 10, 50] = 50,
# ):

#     output = re_ranked_list["movie_list"].strip().replace("  ", " ")

#     output_dict = {}
#     if len(recommend_item) == 0:
#         output_dict["id"] = conv_id
#         output_dict["recall"] = 0
#         output_dict["recommend_item"] = recommend_item
#         output_dict["summarized_conversation"] = summarized_preferences
#         output_dict["recommend_movie_list"] = output
#         output_dict["movie_candidate_list"] = movie_candidate_list

#         pd.DataFrame.from_dict([output_dict]).to_csv(
#             os.path.join(output_dir, f"{model_name}_recall@{top_k}_{n}sample.tsv"),
#             index=False,
#             header=False,
#             mode="a",
#             sep="\t",
#         )
#         return

#     count_match_movie = 0
#     # recommend_movie_list = recommend_item.replace("  ", " ").split("|")
#     for movie in recommend_item:
#         movie = standardize_movie_title(movie)
#         if movie in output:
#             count_match_movie += 1

#     recall = count_match_movie / len(recommend_item)
#     output_dict["recall"] = recall
#     output_dict["row"] = conv_id
#     output_dict["recommend_item"] = recommend_item
#     output_dict["summarized_conversation"] = summarized_preferences
#     output_dict["recommend_movie_list"] = output
#     output_dict["movie_candidate_list"] = f"[{movie_candidate_list}]"

#     pd.DataFrame.from_dict([output_dict]).to_csv(
#         os.path.join(output_dir, f"{model_name}_recall@{top_k}_{n}sample.tsv"),
#         index=False,
#         header=False,
#         mode="a",
#         sep="\t",
#     )

#     return


def standardize_movie_title(movie_title: str) -> str:
    """Normalizes movie titles for consistent comparison."""
    # Remove year and special characters, convert to lowercase
    title = re.sub(r"\s*\(\d{4}\)", "", movie_title)  # Remove year anywhere in title
    title = re.sub(r"[^\w\s]", "", title)  # Remove punctuation
    return title.strip().lower()  # Normalize case and whitespace


def evaluate(
    model_name: str,
    re_ranked_list: dict,
    recommend_item: List[str],
    conv_id: Union[str, int],
    movie_candidate_list: List[str],
    summarized_preferences: str,
    output_dir: os.PathLike,
    n: Literal[100, 200, 300, 400, 500, 600] = 100,
    top_k: Literal[1, 5, 10, 50] = 50,
):
    # Process and normalize the re-ranked list
    # ranked_movies = [standardize_movie_title(m) for m in re_ranked_list["movie_list"].strip().split("|") if m.strip()]
    ranked_movies = [standardize_movie_title(m) for m in re_ranked_list["movie_list"] if m.strip()]

    # Normalize recommended items
    normalized_recommendations = [standardize_movie_title(m) for m in recommend_item if m.strip()]

    # Prepare output dictionary
    output_dict = {
        "id": conv_id,
        "recommend_item": "|".join(recommend_item),
        "summarized_conversation": summarized_preferences,
        # "recommend_movie_list": "|".join(re_ranked_list["movie_list"].split("|")),
        "recommend_movie_list": "|".join(re_ranked_list["movie_list"]),
        "movie_candidate_list": "|".join(movie_candidate_list),
    }

    # Calculate recall if recommendations exist
    if not normalized_recommendations:
        output_dict["recall"] = 0.0
    else:
        matches = sum(1 for m in normalized_recommendations if m in ranked_movies)
        output_dict["recall"] = matches / len(normalized_recommendations)

    # Save results
    result_df = pd.DataFrame([output_dict])
    result_df.to_csv(
        os.path.join(output_dir, f"{model_name.replace('/', '_')}_recall@{top_k}_{n}sample.tsv"),
        sep="\t",
        index=False,
        header=not os.path.exists(
            os.path.join(output_dir, f"{model_name.replace('/', '_')}_recall@{top_k}_{n}sample.tsv")
        ),  # Only write header if file doesn't exist
        mode="a",
    )

    return output_dict["recall"]
