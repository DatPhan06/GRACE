from typing import Dict
import pandas as pd


def insp_dialog_merge(dialog_data) -> list[str]:
    
    ref_movie_data = pd.read_csv(filepath_or_buffer='dataset\INSPIRED\movie_data\movie_database.tsv',
                                 delimiter='\t')
    # Dialouge id
    conv_id = list(dialog_data.keys())[0]
    # if index >= begin_row and index <= end_row:
    try:
        print(f"Start making recommendation for conversation '{conv_id}'")

        # Complete dialogue
        context = []
        recommend_item_id = dialog_data[conv_id][-1]['movie_id']
        recommend_item_name = ref_movie_data.loc[ref_movie_data['video_id'] == recommend_item_id, 'title'].values[0]
        
        last_role = ''
        for message in dialog_data[conv_id]:
            # Speaker role
            speaker = message['speaker']
            # Message
            text = message['text']
            # Sociable strategy
            sociable_strategy = message['expert_label']
            if sociable_strategy == "transparency":
                sociable_strategy = "<offer_help> "
            else:
                sociable_strategy = "<" + sociable_strategy +">"
            if sociable_strategy =="<>":
                sociable_strategy = ""
            
            if last_role == speaker:
                # If same speaker as previous message, append to the last message
                context[-1] += f' {sociable_strategy} {text}'
            
            else:
                # If new speaker, create a new message entry
                message_str = f'{speaker}: {sociable_strategy} {text}'
                context.append(message_str)

            last_role = speaker
        
        context_str = "\n".join(context)
    
        return conv_id, context_str, recommend_item_name
    
    except:
        print(f'Error on conversation {conv_id}. Passing ...')
        

def redial_dialog_merge(dialog_data: dict, conv_id) -> list[str]:
    try:
        print(f"Start process recommend for row {conv_id}.")

        context_str = ""
        recommend_item = ""
        turn_recommend = -1
        for turn_id in reversed(range(len(dialog_data))):
            if dialog_data[turn_id]['is_recommend'] == 1:
                turn_recommend = turn_id
                recommend_item = dialog_data[turn_id]['item_recommend_name']
                break

        for turn_id in range(turn_recommend):
            turn = dialog_data[turn_id]
            if turn['senderWorkerId'] == turn['initiatorWorkerId']:
                context_str += "User: " + turn['convert_text'] + "\n"

            if turn['senderWorkerId'] != turn['initiatorWorkerId']:
                context_str += "Assistant: " + turn['convert_text'] + "\n"

        return context_str, recommend_item
    
    except:
        print(f'Error on conversation {conv_id}. Passing ...')