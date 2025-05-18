import json
import re
import copy
import os

def process_single_conversation(json_line_str):
    conversation = json.loads(json_line_str)
    
    messages = conversation.get("messages", [])
    initiator_worker_id = conversation.get("initiatorWorkerId")
    respondent_worker_id = conversation.get("respondentWorkerId")
    
    # Lấy và kiểm tra kiểu của initiatorQuestions và respondentQuestions
    initiator_questions = conversation.get("initiatorQuestions", {})
    if not isinstance(initiator_questions, dict):
        # Nếu là list (ví dụ: []) hoặc kiểu khác, coi như dictionary rỗng
        initiator_questions = {}
        
    respondent_questions = conversation.get("respondentQuestions", {})
    if not isinstance(respondent_questions, dict):
        # Nếu là list (ví dụ: []) hoặc kiểu khác, coi như dictionary rỗng
        respondent_questions = {}
        
    movie_mentions = conversation.get("movieMentions", {})
    
    sub_dialogs = []
    processed_target_movie_ids = set()
    
    for i, msg in enumerate(messages):
        sender_id = msg.get("senderWorkerId")
        text = msg.get("text", "")
        
        mentioned_movie_ids_in_text = re.findall(r"@(\d+)", text)
        unique_movie_ids_to_check = sorted(list(set(mentioned_movie_ids_in_text)))

        for movie_id_str in unique_movie_ids_to_check:
            is_suggested_by_sender = False
            if sender_id == initiator_worker_id:
                if initiator_questions.get(movie_id_str, {}).get("suggested") == 1:
                    is_suggested_by_sender = True
            elif sender_id == respondent_worker_id:
                if respondent_questions.get(movie_id_str, {}).get("suggested") == 1:
                    is_suggested_by_sender = True
            
            if is_suggested_by_sender and movie_id_str not in processed_target_movie_ids:
                current_dialog_messages = copy.deepcopy(messages[:i+1])
                
                formatted_dialog_entries = []
                for idx, dialog_msg in enumerate(current_dialog_messages):
                    msg_sender_id = dialog_msg.get("senderWorkerId")
                    msg_text_to_format = dialog_msg.get("text")
                    
                    def replace_non_target_id_with_name(match):
                        _id = match.group(1)
                        if _id == movie_id_str and idx == len(current_dialog_messages) - 1:
                            return "@SUGGESTED_MOVIE"
                        return movie_mentions.get(_id, f"@UNKNOWN_MOVIE[{_id}]")
                    
                    formatted_text = re.sub(r"@(\d+)", replace_non_target_id_with_name, msg_text_to_format)
                                        
                    sender_role = "Unknown"
                    if msg_sender_id == initiator_worker_id:
                        sender_role = "Initiator" 
                    elif msg_sender_id == respondent_worker_id:
                        sender_role = "Respondent"
                    
                    formatted_dialog_entries.append(f"{sender_role}: {formatted_text}")

                target_movie_name_for_output = movie_mentions.get(movie_id_str, f"@UNKNOWN_MOVIE[{movie_id_str}]")

                sub_dialogs.append({
                    "dialog": formatted_dialog_entries,
                    "target": target_movie_name_for_output
                })
                processed_target_movie_ids.add(movie_id_str)
                
    return sub_dialogs

def process_jsonl_file(input_filepath, output_filepath):
    all_processed_dialogs = []
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if line:
                    try:
                        # Xử lý từng dòng (từng cuộc hội thoại)
                        sub_dialogs_from_line = process_single_conversation(line)
                        all_processed_dialogs.extend(sub_dialogs_from_line)
                    except json.JSONDecodeError:
                        print(f"Cảnh báo: Bỏ qua dòng không phải JSON hợp lệ: {line}")
                    except Exception as e:
                        print(f"Cảnh báo: Lỗi khi xử lý dòng: {line}. Lỗi: {e}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp đầu vào '{input_filepath}'")
        return
    except Exception as e:
        print(f"Lỗi khi đọc tệp đầu vào: {e}")
        return

    try:
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(all_processed_dialogs, outfile, indent=2, ensure_ascii=False)
        print(f"Xử lý hoàn tất. Kết quả đã được lưu vào '{output_filepath}'")
    except Exception as e:
        print(f"Lỗi khi ghi tệp đầu ra: {e}")

if __name__ == "__main__":
    # Thay đổi tên tệp đầu vào và đầu ra nếu cần
    input_file = "dataset/REDIAL/raw/dialog_data/train_data.jsonl" 
    output_file = "dataset/REDIAL/processed/dialog_data/train_data.json"

    # Tạo một tệp input.jsonl mẫu nếu nó không tồn tại để kiểm thử
    sample_jsonl_content = ''
    # Kiểm tra và tạo tệp mẫu nếu nó không tồn tại
    if not os.path.exists(input_file):
        print(f"Tạo tệp đầu vào mẫu '{input_file}'...")
        with open(input_file, 'w', encoding='utf-8') as f_sample:
            f_sample.write(sample_jsonl_content)
    
    process_jsonl_file(input_file, output_file)

# result = process_single_conversation(json_conversation_line)
# print(json.dumps(result, indent=2))