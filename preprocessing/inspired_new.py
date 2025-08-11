import json
import os
import copy
from typing import Dict, List
import ast
import re

import pandas as pd
from tqdm import tqdm

def _safe_eval(s: str) -> dict:
    """
    Safely evaluate a string that looks like a dictionary literal.
    Uses ast.literal_eval to prevent arbitrary code execution.
    Returns an empty dictionary if the string is invalid or empty.
    """
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}

def _format_sociable_strategy(label: str) -> str:
    """
    Formats the expert_label from the dataset into a standardized strategy tag.
    For instance, 'personal_opinion' becomes '<personal_opinion> '.
    Handles special cases like 'no_strategy' and 'transparency'.
    """
    if not label or label in ["no_strategy", "<>"]:
        return ""
    if label == "transparency":
        return "<offer_help> "
    return f"<{label}> "

def infer_and_mask_movie(text: str, text_with_placeholder: str, target_idx: int) -> str:
    """
    Suy luận tên phim tương ứng với target_idx và thay thế chính xác vị trí của
    tên phim đó trong text gốc bằng '[SUGGESTED_MOVIE]'.
    Hàm này xử lý các trường hợp có dấu ngoặc kép và định dạng khác nhau.
    """
    # 1. Tìm tất cả placeholder và index của chúng
    placeholder_matches = list(re.finditer(r'\[MOVIE_TITLE_(\d+)\]', text_with_placeholder))

    if not placeholder_matches:
        return text

    # 2. Phân tách chuỗi thành các phần tĩnh (nằm giữa các placeholder)
    static_parts = re.split(r'\[MOVIE_TITLE_\d+\]', text_with_placeholder)

    # 3. Suy luận tên phim cho từng placeholder bằng cách alignment linh hoạt
    extracted_movies: Dict[int, str] = {}
    
    # Tạo một mapping từ placeholder index đến vị trí trong danh sách matches
    placeholder_positions = {}
    for i, match in enumerate(placeholder_matches):
        placeholder_idx = int(match.group(1))
        placeholder_positions[placeholder_idx] = i

    # Xử lý từng placeholder theo thứ tự xuất hiện
    text_cursor = 0
    
    for i, match in enumerate(placeholder_matches):
        placeholder_idx = int(match.group(1))
        
        # Lấy phần tĩnh trước placeholder này
        preceding_static = static_parts[i]
        
        # Tìm phần tĩnh này trong text, bỏ qua dấu ngoặc kép nếu cần
        if preceding_static:
            # Thử tìm exact match trước
            pos = text.find(preceding_static, text_cursor)
            
            # Nếu không tìm thấy và có dấu ngoặc kép, thử bỏ dấu ngoặc kép
            if pos == -1 and '"' in preceding_static:
                # Thử bỏ dấu ngoặc kép ở cuối
                if preceding_static.endswith('"'):
                    pos = text.find(preceding_static[:-1], text_cursor)
                # Thử bỏ dấu ngoặc kép ở đầu
                elif preceding_static.startswith('"'):
                    pos = text.find(preceding_static[1:], text_cursor)
            
            if pos == -1:
                # Fallback: tìm bằng cách bỏ tất cả dấu ngoặc kép
                clean_static = preceding_static.replace('"', '')
                if clean_static:
                    pos = text.find(clean_static, text_cursor)
            
            if pos == -1:
                return text  # Không tìm thấy, trả về text gốc
            
            text_cursor = pos + len(preceding_static.replace('"', ''))
        
        # Tìm phần tĩnh sau placeholder để xác định ranh giới
        following_static = static_parts[i + 1] if i + 1 < len(static_parts) else ""
        
        if following_static:
            # Tương tự, xử lý dấu ngoặc kép cho phần sau
            end_pos = text.find(following_static, text_cursor)
            
            if end_pos == -1 and '"' in following_static:
                # Thử bỏ dấu ngoặc kép ở đầu
                if following_static.startswith('"'):
                    end_pos = text.find(following_static[1:], text_cursor)
                    if end_pos != -1:
                        end_pos -= 1  # Điều chỉnh vị trí vì bỏ ký tự đầu
                # Thử bỏ dấu ngoặc kép ở cuối  
                elif following_static.endswith('"'):
                    end_pos = text.find(following_static[:-1], text_cursor)
            
            if end_pos == -1:
                # Fallback: tìm bằng cách bỏ tất cả dấu ngoặc kép
                clean_static = following_static.replace('"', '')
                if clean_static:
                    end_pos = text.find(clean_static, text_cursor)
            
            if end_pos == -1:
                return text  # Không tìm thấy, trả về text gốc
        else:
            end_pos = len(text)
        
        # Trích xuất tên phim
        movie_name = text[text_cursor:end_pos].strip()
        extracted_movies[placeholder_idx] = movie_name
        
        text_cursor = end_pos

    # 4. Tái tạo lại chuỗi, sử dụng text_with_placeholder làm template
    result = text_with_placeholder
    
    # Thay thế từng placeholder
    for placeholder_idx, movie_name in extracted_movies.items():
        placeholder = f"[MOVIE_TITLE_{placeholder_idx}]"
        
        if placeholder_idx == target_idx:
            # Thay placeholder target bằng [SUGGESTED_MOVIE]
            result = result.replace(placeholder, "[SUGGESTED_MOVIE]")
        else:
            # Thay placeholder khác bằng tên phim thực tế
            result = result.replace(placeholder, movie_name)
    
    return result

def generate_samples_from_dialog(dialog_df: pd.DataFrame) -> List[Dict]:
    """
    Generates training samples from a single conversation/dialog dataframe.
    """
    samples = []
    mentioned_movies_as_target = set()

    dialog_df = dialog_df.sort_values("utt_id")
    utterances = dialog_df.to_dict('records')

    for i, utt in enumerate(utterances):
        if utt['speaker'] != 'RECOMMENDER':
            continue

        # Condition 1: Check if the utterance is a true movie recommendation
        # by ensuring it contains a movie title placeholder. This avoids
        # creating samples from plot summaries or indirect mentions.
        if "[MOVIE_TITLE_" not in utt['text_with_placeholder']:
            continue
            
        movies_str = utt.get("movies")
        if not isinstance(movies_str, str) or not movies_str.strip():
            continue

        current_utt_movies = [m.strip() for m in movies_str.split(';') if m.strip()]
        movie_dict = _safe_eval(utt.get('movie_dict', '{}'))
        if not movie_dict:
            continue

        # A single utterance can recommend multiple movies. We create a separate sample for each.
        for target_movie in current_utt_movies:
            if target_movie in mentioned_movies_as_target:
                continue

            context_utterances = utterances[:i+1]
            
            dialog_context = []
            # Format all previous utterances for the context history.
            for context_utt in context_utterances[:-1]:
                strategy = _format_sociable_strategy(context_utt.get('expert_label', ''))
                dialog_context.append(f"{context_utt['speaker']}: {strategy}{context_utt['text']}")

            final_utterance = copy.deepcopy(context_utterances[-1])
            final_text = final_utterance['text']
            final_text_with_placeholders = final_utterance['text_with_placeholder']
            
            target_movie_idx = movie_dict.get(target_movie)

            # Use the new masking function that handles quotes and other formatting differences
            masked_text = final_text_with_placeholders  # fallback to placeholder text
            if target_movie_idx is not None:
                masked_text = infer_and_mask_movie(final_text, final_text_with_placeholders, target_movie_idx)
            
            # Format the final utterance with its strategy tag.
            strategy = _format_sociable_strategy(final_utterance.get('expert_label', ''))
            dialog_context.append(f"{final_utterance['speaker']}: {strategy}{masked_text}")

            sample = {
                "dialog": dialog_context,
                "target": target_movie
            }
            samples.append(sample)
            
            mentioned_movies_as_target.add(target_movie)
    
    return samples

def process_inspired_file(input_filepath: str, output_filepath: str):
    """
    Reads an INSPIRED .tsv file, processes all conversations, and writes the
    resulting samples to a JSON file.
    """
    try:
        print(f"Reading from: {input_filepath}")
        df = pd.read_csv(input_filepath, sep="\t", keep_default_na=False, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_filepath}'")
        return

    # Group the dataframe by dialog_id to process each conversation separately.
    dialog_groups = df.groupby("dialog_id", sort=False)
    
    all_samples = []
    print(f"Processing {dialog_groups.ngroups} conversations...")
    for _, group in tqdm(dialog_groups):
        samples_from_dialog = generate_samples_from_dialog(group)
        all_samples.extend(samples_from_dialog)

    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generated {len(all_samples)} samples.")
    print(f"Saving processed data to: {output_filepath}")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    RAW_DATA_BASE_DIR = "dataset/INSPIRED/raw/dialog_data"
    PROCESSED_DATA_BASE_DIR = "dataset/INSPIRED/processed/dialog_data"

    # --- Process training data ---
    train_input_file = os.path.join(RAW_DATA_BASE_DIR, "train.tsv")
    train_output_file = os.path.join(PROCESSED_DATA_BASE_DIR, "train_new_processed.json")
    process_inspired_file(train_input_file, train_output_file)

    # --- Process test data ---
    test_input_file = os.path.join(RAW_DATA_BASE_DIR, "test.tsv")
    test_output_file = os.path.join(PROCESSED_DATA_BASE_DIR, "test_new_processed.json")
    process_inspired_file(test_input_file, test_output_file)

    print("\nPreprocessing finished successfully!") 