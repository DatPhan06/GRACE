import pandas as pd
import json
import yaml
import tqdm

with open("config.yaml", "r") as cf:
    config = yaml.safe_load(cf)


class read_process_data_class:
    # Replace movie id with movie name in conversation
    train_data = None

    def add_movie_name_conv(self, turn, movie_mention_list):
        turn_message = turn["text"]
        for item in movie_mention_list:
            if f"@{item}" in turn_message:
                turn_message = turn_message.replace(f"@{item}", movie_mention_list[item])
        return turn_message

    # Add is_recommend, is_liked, is_watched to conversation
    def add_is_recommender_to_conv(self, turn, movie_question_list, movie_mention_list, recommendWorkerID):
        is_recommend = 0
        is_liked = 0
        is_watched = 0
        item_recommend_id_list = []
        item_recommend_name_list = []
        item_recommend_id = ""
        item_recommend_name = ""
        for item in movie_question_list:
            if f"@{item}" in turn["text"]:
                # Check if it is a recommend turn
                if movie_question_list[item]["suggested"] == 1 and recommendWorkerID == turn["senderWorkerId"]:
                    is_recommend = 1
                    item_recommend_id_list.append(item)
                    item_recommend_name_list.append(movie_mention_list[item])
                item_recommend_id = "|".join(item_recommend_id_list)
                item_recommend_name = "|".join(item_recommend_name_list)
                is_liked = movie_question_list[item]["liked"]
                is_watched = movie_question_list[item]["seen"]
        return is_recommend, is_liked, is_watched, item_recommend_id, item_recommend_name

    # General function
    def processing_data(self, data_line):
        movie_mention_list = data_line["movieMentions"]
        conv_list = data_line["messages"]
        movie_question_list = data_line["respondentQuestions"]
        recommender_id = data_line["respondentWorkerId"]

        convert_conv_list = []
        for turn in conv_list:
            turn["convert_text"] = self.add_movie_name_conv(turn, movie_mention_list)
            (
                turn["is_recommend"],
                turn["is_liked"],
                turn["is_watched"],
                turn["item_recommend_id"],
                turn["item_recommend_name"],
            ) = self.add_is_recommender_to_conv(turn, movie_question_list, movie_mention_list, recommender_id)
            convert_conv_list.append(turn)

        data_line["messages"] = convert_conv_list
        return data_line

    # Process train data function
    def get_train_data(self, train_data):
        train_split_conv = []
        for conv in train_data:
            turn_list = []
            for turn in conv["messages"]:
                turn["respondentWorkerId"] = conv["respondentWorkerId"]
                turn["initiatorWorkerId"] = conv["initiatorWorkerId"]
                turn_list.append(turn)
                if turn["is_recommend"] == 1:
                    current_turn_list = turn_list.copy()
                    train_split_conv.append(current_turn_list)
        return train_split_conv

    def __init__(self):
        file_path_train = config["RedialDataPath"]["raw"]["dialog"]["train"]

        # Load train data
        train_data = []
        for line in open(file_path_train, "r", encoding="utf-8"):
            data_line = json.loads(line)
            data_line_processed = self.processing_data(data_line)
            train_data.append(data_line_processed)

        df_train = pd.DataFrame.from_records(train_data)

        # Process train_data:
        self.train_data = self.get_train_data(train_data)
