import pandas as pd
import json
import yaml

with open("config.yaml", "r") as cf:
    config = yaml.safe_load(cf)

class read_process_data_class:
    # Class variable to store processed training data
    train_data = None

    def add_movie_name_conv(self, turn, movie_mention_list):
        """
        Replaces movie IDs with actual movie names in conversation text
        Args:
            turn: Dictionary containing message turn info
            movie_mention_list: Dictionary mapping movie IDs to movie names
        Returns:
            Processed message text with movie names instead of IDs
        """
        turn_message = turn['text']
        for item in movie_mention_list:
            if f"@{item}" in turn_message:
                turn_message = turn_message.replace(f"@{item}", movie_mention_list[item])
        return turn_message

    def add_is_recommender_to_conv(self, turn, movie_question_list, movie_mention_list, recommendWorkerID):
        """
        Adds recommendation metadata to conversation turns
        Args:
            turn: Dictionary containing message turn info
            movie_question_list: Dictionary containing movie interaction data
            movie_mention_list: Dictionary mapping movie IDs to names
            recommendWorkerID: ID of the recommender worker
        Returns:
            Tuple containing:
            - is_recommend: Whether this turn contains a recommendation (0/1)
            - is_liked: Whether recommended movie was liked (0/1)
            - is_watched: Whether movie was watched (0/1)
            - item_recommend_id: String of recommended movie IDs
            - item_recommend_name: String of recommended movie names
        """
        is_recommend = 0
        is_liked = 0
        is_watched = 0
        item_recommend_id_list = []
        item_recommend_name_list = []
        item_recommend_id = ""
        item_recommend_name = ""

        # Process each movie mentioned in the turn
        for item in movie_question_list:
            if f"@{item}" in turn['text']:
                # Check if this is a recommendation turn from the recommender
                if movie_question_list[item]['suggested'] == 1 and recommendWorkerID == turn['senderWorkerId']:
                    is_recommend = 1
                    item_recommend_id_list.append(item)
                    item_recommend_name_list.append(movie_mention_list[item])
                # Join multiple recommendations with '|'
                item_recommend_id = "|".join(item_recommend_id_list)
                item_recommend_name = "|".join(item_recommend_name_list)
                # Get user interaction flags
                is_liked = movie_question_list[item]['liked']
                is_watched = movie_question_list[item]['seen']
        return is_recommend, is_liked, is_watched, item_recommend_id, item_recommend_name

    def processing_data(self, data_line):
        """
        Process a single conversation data line
        Args:
            data_line: Dictionary containing raw conversation data
        Returns:
            Processed conversation data with added metadata
        """
        # Extract relevant data from input
        movie_mention_list = data_line['movieMentions']
        conv_list = data_line['messages']
        movie_question_list = data_line['respondentQuestions']
        recommender_id = data_line['respondentWorkerId']

        # Process each turn in conversation
        convert_conv_list = []
        for turn in conv_list:
            # Replace movie IDs with names
            turn['convert_text'] = self.add_movie_name_conv(turn, movie_mention_list)
            # Add recommendation metadata
            turn['is_recommend'], turn['is_liked'], turn['is_watched'], turn['item_recommend_id'], turn[
                'item_recommend_name'] = self.add_is_recommender_to_conv(turn, movie_question_list, movie_mention_list,
                                                                    recommender_id)
            convert_conv_list.append(turn)

        data_line['messages'] = convert_conv_list
        return data_line

    def get_train_data(self, train_data):
        """
        Process full training dataset and split into recommendation-focused conversations
        Args:
            train_data: List of raw conversation data
        Returns:
            List of conversation segments leading up to recommendations
        """
        train_split_conv = []
        for conv in train_data:
            turn_list = []
            for turn in conv['messages']:
                # Add worker IDs to each turn
                turn['respondentWorkerId'] = conv['respondentWorkerId']
                turn['initiatorWorkerId'] = conv['initiatorWorkerId']
                turn_list.append(turn)
                # When recommendation found, save conversation up to this point
                if turn['is_recommend'] == 1:
                    current_turn_list = turn_list.copy()
                    train_split_conv.append(current_turn_list)
        return train_split_conv
    
    # def merge(self, train_data):
    #     processed = []
    #     for conv_id in range(len(train_data)):
    #         item = {}
    #         context_str = ""
    #         recommend_item = ""
    #         turn_recommend = -1
    #         for turn_id in reversed(range(len(train_data[conv_id]))):
    #             if train_data[conv_id][turn_id]['is_recommend'] == 1:
    #                 turn_recommend = turn_id
    #                 recommend_item = train_data[conv_id][turn_id]['item_recommend_name']
    #                 # print(f"Recommend of this row is: {train_data[conv_id][turn_id]['item_recommend_id']}, name: {recommend_item}")
    #                 break

    #         for turn_id in range(turn_recommend):
    #             turn = train_data[conv_id][turn_id]
    #             if turn['senderWorkerId'] == turn['initiatorWorkerId']:
    #                 context_str += "User: " + turn['convert_text'] + "\n"
    #                 # print("User: " + turn['convert_text'] + "\n")
    #             if turn['senderWorkerId'] != turn['initiatorWorkerId']:
    #                 context_str += "Assistant: " + turn['convert_text'] + "\n"
                    
    #         item['dialog'] = context_str
    #         item['target'] = recommend_item
            
    #         processed.append(item)
        
    #     return processed

    def save_data(self, file_path, data):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def __init__(self):
        """
        Initialize class and process training data
        """
        # Load configuration
        self.train_dialog = config["RedialDataPath"]["raw"]["dialog"]["train"]
        self.processed_train = config['RedialDataPath']['processed']["dialog"]['train']

        # Process training data line by line
        train_data = []
        for line in open(self.train_dialog, "r", encoding="utf-8"):
            data_line = json.loads(line)
            data_line_processed = self.processing_data(data_line)
            train_data.append(data_line_processed)

        # Process and split conversations
        data = self.get_train_data(train_data)
        
        self.save_data(self.processed_train, data)
