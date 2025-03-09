import pandas as pd
import os
import json
from typing import List, Dict, Any  # Import typing classes

# For reading configuration values from config.ini
from utils.read_config import read_config    

# Load configuration values from config file
config_value = read_config()

# Transform .tsv data file into .json data file
def insp_dialog_data_transform(tsv_file_path: os.PathLike) -> List[Dict[str, List[Dict[str, Any]]]]:
    """
    Transform TSV data into a dialog structure with dialog_ids as keys
    and lists of utterance dictionaries as values, preserving all columns.
    
    Args:
        tsv_file_path: Path to the TSV file containing dialog data
        
    Returns:
        List of dictionaries, each containing a dialog_id mapping to a list of utterances
    """
    
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(tsv_file_path, sep='\t')
    
    # Group the data by dialog_id to organize utterances by conversation
    dialog_groups = df.groupby('dialog_id')
    
    # Initialize the result list to store processed dialogs
    result = []
    
    # Process each dialog group separately
    for dialog_id, group in dialog_groups:
        # Create a dictionary for this specific dialog
        dialog_dict = {dialog_id: []}
        
        # Sort group by utt_id to ensure utterances are in correct chronological order
        group = group.sort_values('utt_id')
        
        # Process each utterance in the current dialog group
        for _, row in group.iterrows():
            
            # Create utterance dict with all columns from the row
            utterance = {}
            for column in row.index:
                
                # Skip dialog_id as it's already the dictionary key
                if column == 'dialog_id':
                    continue   
                
                # Handle special dictionary fields - try to parse JSON strings
                if column in ['movie_dict', 'genre_dict', 'actor_dict', 'director_dict', 'others_dict']:
                    try:
                        if isinstance(row[column], str) and row[column].strip():
                            # Using eval to parse Python dictionary literals from strings
                            utterance[column] = eval(row[column])  
                        else:
                            # Handle empty or non-string values
                            utterance[column] = row[column]
                            utterance[column] = {} if pd.isna(row[column]) else row[column]
                    except:
                        # If parsing fails, keep the original value
                        utterance[column] = row[column]
                else:
                    # Handle regular fields, replacing NaN values with empty strings
                    utterance[column] = "" if pd.isna(row[column]) else row[column]
            
            # Add the processed utterance to this dialog's list
            dialog_dict[dialog_id].append(utterance)
        
        # Add the complete dialog dictionary to the result list
        result.append(dialog_dict)

    return result


def add_movie_name_conv(turn, movie_mention_list: list):
    """
    Replaces movie IDs with actual movie names in conversation turns.
    
    Args:
        turn (dict): A dictionary containing conversation turn data with 'text' field
        movie_mention_list (list): Dictionary mapping movie IDs to movie names
        
    Returns:
        str: Modified conversation text with movie IDs replaced by movie names
    """
    turn_message = turn['text']
    # Iterate through each movie ID and replace it with the movie name
    for item in movie_mention_list:
        if f"@{item}" in turn_message:
            turn_message = turn_message.replace(f"@{item}", movie_mention_list[item])
    return turn_message


def add_is_recommender_to_conv(turn, movie_question_list, movie_mention_list, recommendWorkerID):
    """
    Identifies recommendation status and metadata for movies mentioned in a conversation turn.
    
    Args:
        turn (dict): Dictionary containing conversation turn data
        movie_question_list (dict): Dictionary containing movie question data
        movie_mention_list (dict): Dictionary mapping movie IDs to movie names
        recommendWorkerID (str): ID of the worker who is the recommender
        
    Returns:
        tuple: (is_recommend, is_liked, is_watched, item_recommend_id, item_recommend_name)
            - is_recommend (int): 1 if this turn contains a recommendation, 0 otherwise
            - is_liked (int): 1 if the recommended movie was liked, 0 otherwise
            - is_watched (int): 1 if the recommended movie was watched, 0 otherwise
            - item_recommend_id (str): Pipe-separated list of recommended movie IDs
            - item_recommend_name (str): Pipe-separated list of recommended movie names
    """
    is_recommend = 0
    is_liked = 0
    is_watched = 0
    item_recommend_id_list = []
    item_recommend_name_list = []
    item_recommend_id = ""
    item_recommend_name = ""
    # Check each movie in the movie question list
    for item in movie_question_list:
        if f"@{item}" in turn['text']:
            # Check if this turn contains a recommendation by the recommender
            if movie_question_list[item]['suggested'] == 1 and recommendWorkerID == turn['senderWorkerId']:
                is_recommend = 1
                item_recommend_id_list.append(item)
                item_recommend_name_list.append(movie_mention_list[item])
            # Join multiple recommended movie IDs and names with pipe separator
            item_recommend_id = "|".join(item_recommend_id_list)
            item_recommend_name = "|".join(item_recommend_name_list)
            # Get whether the movie was liked and watched
            is_liked = movie_question_list[item]['liked']
            is_watched = movie_question_list[item]['seen']
    return is_recommend, is_liked, is_watched, item_recommend_id, item_recommend_name


def processing_data(data_line):
    """
    Process a single conversation data line by adding movie names and recommendation metadata.
    
    Args:
        data_line (dict): Dictionary containing conversation data
        
    Returns:
        dict: Processed conversation data with enhanced information
    """
    # Extract necessary components from the data line
    movie_mention_list = data_line['movieMentions']
    conv_list = data_line['messages']
    movie_question_list = data_line['respondentQuestions']
    recommender_id = data_line['respondentWorkerId']

    # Process each turn in the conversation
    convert_conv_list = []
    for turn in conv_list:
        # Replace movie IDs with movie names
        turn['convert_text'] = add_movie_name_conv(turn, movie_mention_list)
        # Add recommendation metadata to the turn
        turn['is_recommend'], turn['is_liked'], turn['is_watched'], turn['item_recommend_id'], turn[
            'item_recommend_name'] = add_is_recommender_to_conv(turn, movie_question_list, movie_mention_list,
                                                                recommender_id)
        convert_conv_list.append(turn)

    # Update the messages with processed conversation turns
    data_line['messages'] = convert_conv_list
    return data_line


def redial_dialog_data_transform(train_data_path: os.PathLike):
    """
    Load and process training data from a file, then split conversations at recommendation points.
    
    This function reads each line from the training data file, processes it to add movie names
    and recommendation information, and then splits conversations into segments that end
    with a recommendation.
    
    Args:
        train_data_path (os.PathLike): Path to the training data file
        
    Returns:
        list: List of conversation segments, each ending with a recommendation
    """
    
    # Load and process each line in the training data file
    train_data = []
    for line in open(train_data_path, "r", encoding='utf-8'):
        data_line = json.loads(line)  # Parse JSON line
        data_line_processed = processing_data(data_line)  # Process the conversation data
        train_data.append(data_line_processed)
    
    # Split conversations at recommendation points
    train_split_conv = []
    for conv in train_data:
        turn_list = []
        for turn in conv['messages']:
            # Add conversation metadata to each turn
            turn['respondentWorkerId'] = conv['respondentWorkerId']
            turn['initiatorWorkerId'] = conv['initiatorWorkerId']
            turn_list.append(turn)
            # If this turn contains a recommendation, save the conversation up to this point
            if turn['is_recommend'] == 1:
                current_turn_list = turn_list.copy()
                train_split_conv.append(current_turn_list)
                
    return train_split_conv