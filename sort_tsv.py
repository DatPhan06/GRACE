import pandas as pd

def sort_tsv_by_id(input_file, output_file):
    # Đọc file TSV
    df = pd.read_csv(input_file, sep='\t')
    
    # Loại bỏ các dòng trùng lặp và sắp xếp theo id tăng dần
    df_unique = df.drop_duplicates(subset=['id']).sort_values(by='id')
    
    # Lưu kết quả vào file mới
    df_unique.to_csv(output_file, sep='\t', index=False)
    print(f"Đã loại bỏ trùng lặp, sắp xếp và lưu kết quả vào file: {output_file}")

if __name__ == "__main__":
    input_file = "output/REDIAL/gemini-2.0-flash_recall@50_500sample.tsv"  # Tên file input
    output_file = "output/REDIAL/gemini-2.0-flash_recall@50_500sample.tsv"  # Tên file output
    sort_tsv_by_id(input_file, output_file) 