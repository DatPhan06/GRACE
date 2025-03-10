import pandas as pd

# Erorr log
import logging

# Type hints for better code documentation
from typing import Dict, List, Tuple


# Load reference movie data from TSV file
ref_movie_data = pd.read_csv(
    filepath_or_buffer="dataset\INSPIRED\movie_data\movie_database.tsv", delimiter="\t"
)


def insp_dialog_merge(dialog_data) -> Tuple[str, str, str]:
    """
    Merge INSPIRED dialog data into a structured format for recommendation.

    Args:
        dialog_data: Dictionary containing dialog data keyed by conversation ID

    Returns:
        Tuple of (conversation_id, context_string, recommended_item_name)
    """

    # Extract dialogue id from the input data
    # Get the first (and only) key as conversation ID
    conv_id = list(dialog_data.keys())[0]
    try:
        print(f"Start making recommendation for conversation '{conv_id}'")

        # Initialize list to store formatted dialogue
        context = []

        # Extract the recommended movie ID from the last message item in the dictionary
        recommend_item_id = dialog_data[conv_id][-1]["movie_id"]

        # Look up the movie name from the reference data
        recommend_item_name = ref_movie_data.loc[
            ref_movie_data["video_id"] == recommend_item_id, "title"
        ].values[0]

        # Track the previous speaker to handle consecutive messages
        last_role = ""

        # Process each message in the dialogue
        for message in dialog_data[conv_id]:
            # Extract speaker role
            speaker = message["speaker"]

            # Extract message text
            text = message["text"]

            # Format sociable strategy tag
            # Get strategy label
            sociable_strategy = message["expert_label"]
            if sociable_strategy == "transparency":
                sociable_strategy = "<offer_help> "
            else:
                # Wrap other strategies in angle brackets
                sociable_strategy = "<" + sociable_strategy + ">"

            if sociable_strategy == "<>":
                sociable_strategy = ""

            # Check if this message is from the same speaker as the previous one
            if last_role == speaker:
                # If same speaker as previous message, append to the last message
                context[-1] += f" {sociable_strategy} {text}"

            else:
                # If new speaker, create a new message entry
                message_str = f"{speaker}: {sociable_strategy} {text}"
                context.append(message_str)  # Add to context list

            # Update last speaker for next iteration
            # Remember current speaker for next message
            last_role = speaker

        # Convert list to string with line breaks
        context_str = "\n".join(context)

        # Return conversation ID, formatted context, and recommended movie name
        return conv_id, context_str, recommend_item_name

    except Exception as e:
        # Handle errors
        logging.error(f"Error on conversation {conv_id}: {str(e)}. Passing ...")
        return "", "", ""


def redial_dialog_merge(dialog_data: List[Dict], conv_id: str) -> Tuple[str, str]:
    """
    Merge ReDial dialog data into a structured context and extract recommendation.

    Args:
        dialog_data: List of dictionaries containing conversation turns
        conv_id: Conversation identifier for logging purposes

    Returns:
        Tuple of (context_string, recommended_item_name)
    """
    try:
        print(f"Start process recommend for row {conv_id}.")

        # Will hold formatted conversation
        context_str = ""

        # Will hold recommended item name
        recommend_item = ""

        # Will hold index of recommendation turn
        turn_recommend = -1

        for turn_id in reversed(range(len(dialog_data))):
            # Check if turn contains recommendation
            if dialog_data[turn_id]["is_recommend"] == 1:
                # Store index of recommendation turn
                turn_recommend = turn_id

                # Get recommended item name
                recommend_item = dialog_data[turn_id]["item_recommend_name"]

                # Stop after finding the first recommendation
                break

        # Format conversation up to the recommendation
        for turn_id in range(turn_recommend):
            # Get current turn data
            turn = dialog_data[turn_id]

            # Format message based on speaker role
            # Check if sender is initiator
            if turn["senderWorkerId"] == turn["initiatorWorkerId"]:
                # Format as user message
                context_str += "User: " + turn["convert_text"] + "\n"

            # Check if sender is responder
            if turn["senderWorkerId"] != turn["initiatorWorkerId"]:
                # Format as assistant message
                context_str += "Assistant: " + turn["convert_text"] + "\n"

        # Return formatted context and recommendation
        return context_str, recommend_item

    except Exception as e:
        # Handle errors
        logging.error(f"Error on conversation {conv_id}: {str(e)}. Passing ...")
        return "", ""
