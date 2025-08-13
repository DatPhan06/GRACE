#!/usr/bin/env python3
"""
Phân tích vị trí ranking của recommend_item trong movie_candidate_list
"""

import csv
import statistics
from collections import defaultdict

def analyze_ranking_position(tsv_file):
    """
    Phân tích vị trí của recommend_item trong movie_candidate_list
    
    Args:
        tsv_file (str): Đường dẫn đến file TSV
    """
    
    positions = []  # Lưu vị trí của recommend_item trong candidate_list
    found_count = 0  # Số lượng recommend_item được tìm thấy trong candidate_list
    total_count = 0  # Tổng số records
    not_found_items = []  # Danh sách các item không tìm thấy
    
    print("Đang phân tích file TSV...")
    
    with open(tsv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        for row in reader:
            total_count += 1
            
            recommend_item = row['recommend_item'].strip()
            candidate_list_str = row['movie_candidate_list'].strip()
            
            # Tách danh sách candidate movies
            candidate_list = [movie.strip() for movie in candidate_list_str.split('|')]
            
            # Chuẩn hóa recommend_item để so sánh
            # Loại bỏ năm trong ngoặc và normalize spaces
            recommend_item_clean = recommend_item
            if '(' in recommend_item and ')' in recommend_item:
                recommend_item_clean = recommend_item.split('(')[0].strip()
            
            # Normalize multiple spaces to single space
            recommend_item_clean = ' '.join(recommend_item_clean.split())
            
            # Chuẩn hóa candidate list
            candidate_list_clean = []
            for movie in candidate_list:
                movie_clean = movie
                if '(' in movie and ')' in movie:
                    movie_clean = movie.split('(')[0].strip()
                movie_clean = ' '.join(movie_clean.split())
                candidate_list_clean.append(movie_clean)
            
            # Tìm vị trí của recommend_item trong candidate_list
            try:
                position = candidate_list_clean.index(recommend_item_clean) + 1  # +1 vì index bắt đầu từ 0
                positions.append(position)
                found_count += 1
                print(f"✓ Found '{recommend_item}' -> '{recommend_item_clean}' at position {position}")
            except ValueError:
                # recommend_item không có trong candidate_list
                not_found_items.append(f"{recommend_item} -> {recommend_item_clean}")
                print(f"✗ Not found '{recommend_item}' -> '{recommend_item_clean}'")
    
    print("\n" + "="*60)
    print("KẾT QUẢ PHÂN TÍCH VỊ TRÍ RANKING")
    print("="*60)
    
    print(f"Tổng số records: {total_count:,}")
    print(f"Số items được tìm thấy trong candidate list: {found_count:,}")
    print(f"Số items KHÔNG được tìm thấy: {len(not_found_items):,}")
    print(f"Tỷ lệ tìm thấy: {(found_count/total_count)*100:.2f}%")
    
    if positions:
        print(f"\nTHỐNG KÊ VỊ TRÍ RANKING:")
        print(f"Vị trí trung bình: {statistics.mean(positions):.2f}")
        print(f"Vị trí median: {statistics.median(positions):.0f}")
        print(f"Vị trí tốt nhất (min): {min(positions)}")
        print(f"Vị trí tệ nhất (max): {max(positions)}")
        print(f"Độ lệch chuẩn: {statistics.stdev(positions):.2f}")
        
        # Phân tích top-k
        print(f"\nPHÂN TÍCH TOP-K:")
        top_ranges = [1, 5, 10, 20, 50, 100]
        for k in top_ranges:
            count_in_topk = sum(1 for pos in positions if pos <= k)
            percentage = (count_in_topk / found_count) * 100
            print(f"Top-{k:3d}: {count_in_topk:4d} items ({percentage:5.1f}%)")
        
        # Phân tích phân phối vị trí
        print(f"\nPHÂN PHỐI VỊ TRÍ:")
        ranges = [(1, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, float('inf'))]
        for start, end in ranges:
            if end == float('inf'):
                count = sum(1 for pos in positions if pos >= start)
                print(f"Vị trí {start}+: {count} items ({(count/found_count)*100:.1f}%)")
            else:
                count = sum(1 for pos in positions if start <= pos <= end)
                print(f"Vị trí {start}-{end}: {count} items ({(count/found_count)*100:.1f}%)")
    
    # Phân tích các item không tìm thấy
    if not_found_items:
        print(f"\nTOP 10 ITEMS KHÔNG TÌM THẤY TRONG CANDIDATE LIST:")
        item_counts = defaultdict(int)
        for item in not_found_items:
            item_counts[item] += 1
        
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (item, count) in enumerate(sorted_items[:10], 1):
            print(f"  {i:2d}. {item}: {count} lần")
    
    return {
        'total_count': total_count,
        'found_count': found_count,
        'not_found_count': len(not_found_items),
        'found_rate': (found_count/total_count)*100,
        'positions': positions,
        'avg_position': statistics.mean(positions) if positions else None,
        'median_position': statistics.median(positions) if positions else None,
        'not_found_items': not_found_items
    }

def analyze_recall_vs_ranking(tsv_file):
    """
    Phân tích mối quan hệ giữa recall và vị trí ranking
    """
    print("\n" + "="*60)
    print("PHÂN TÍCH RECALL VS RANKING POSITION")
    print("="*60)
    
    recall_by_position = defaultdict(list)
    
    with open(tsv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        for row in reader:
            recommend_item = row['recommend_item'].strip()
            candidate_list_str = row['movie_candidate_list'].strip()
            recall = float(row['recall'])
            
            candidate_list = [movie.strip() for movie in candidate_list_str.split('|')]
            
            # Chuẩn hóa recommend_item để so sánh (giống logic trên)
            recommend_item_clean = recommend_item
            if '(' in recommend_item and ')' in recommend_item:
                recommend_item_clean = recommend_item.split('(')[0].strip()
            recommend_item_clean = ' '.join(recommend_item_clean.split())
            
            # Chuẩn hóa candidate list
            candidate_list_clean = []
            for movie in candidate_list:
                movie_clean = movie
                if '(' in movie and ')' in movie:
                    movie_clean = movie.split('(')[0].strip()
                movie_clean = ' '.join(movie_clean.split())
                candidate_list_clean.append(movie_clean)
            
            try:
                position = candidate_list_clean.index(recommend_item_clean) + 1
                recall_by_position[position].append(recall)
            except ValueError:
                # Item không tìm thấy trong candidate list
                recall_by_position['not_found'].append(recall)
    
    # Tính recall trung bình cho từng nhóm vị trí
    position_ranges = [
        (1, 1, "Vị trí 1"),
        (2, 5, "Vị trí 2-5"), 
        (6, 10, "Vị trí 6-10"),
        (11, 20, "Vị trí 11-20"),
        (21, 50, "Vị trí 21-50"),
        (51, 100, "Vị trí 51-100"),
        (101, float('inf'), "Vị trí 100+")
    ]
    
    print("RECALL TRUNG BÌNH THEO NHÓM VỊ TRÍ:")
    for start, end, label in position_ranges:
        recalls = []
        for pos, recall_list in recall_by_position.items():
            if pos != 'not_found' and start <= pos <= end:
                recalls.extend(recall_list)
        
        if recalls:
            avg_recall = statistics.mean(recalls)
            print(f"{label:15}: {avg_recall:.4f} (n={len(recalls)})")
    
    # Recall cho items không tìm thấy
    if 'not_found' in recall_by_position:
        not_found_recalls = recall_by_position['not_found']
        avg_recall = statistics.mean(not_found_recalls)
        print(f"{'Không tìm thấy':15}: {avg_recall:.4f} (n={len(not_found_recalls)})")

def main():
    """Hàm main để chạy phân tích"""
    tsv_file = "output/REDIAL/test/gemini-2.0-flash_recall@10_500sample.tsv"
    
    try:
        print("Bắt đầu phân tích vị trí ranking của recommend_item...")
        results = analyze_ranking_position(tsv_file)
        analyze_recall_vs_ranking(tsv_file)
        
        print("\n" + "="*60)
        print("TÓM TẮT KẾT QUẢ")
        print("="*60)
        print(f"📊 Tổng số recommendations: {results['total_count']:,}")
        print(f"✅ Tìm thấy trong candidate list: {results['found_count']:,} ({results['found_rate']:.1f}%)")
        print(f"❌ Không tìm thấy: {results['not_found_count']:,}")
        
        if results['avg_position']:
            print(f"📈 Vị trí trung bình: {results['avg_position']:.2f}")
            print(f"📊 Vị trí median: {results['median_position']:.0f}")
        
        print("\nPhân tích hoàn thành!")
        
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {tsv_file}")
        print("Vui lòng kiểm tra đường dẫn file.")
    except Exception as e:
        print(f"❌ Lỗi trong quá trình phân tích: {e}")
        raise

if __name__ == "__main__":
    main() 