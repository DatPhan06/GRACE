# Graph Database Integration

Hệ thống đã được cập nhật để sử dụng Neo4j Graph Database thay vì Vector Database (ChromaDB).

## Các thay đổi chính

### 1. Module mới: `utils/GraphDB/graph_retriever.py`
- **GraphRetriever class**: Kết nối và truy vấn Neo4j database
- **Hybrid retrieval**: Kết hợp 3 phương pháp:
  - Semantic similarity (sử dụng embeddings)
  - Content-based filtering (dựa trên thể loại, đạo diễn, diễn viên)
  - Collaborative filtering (dựa trên mối quan hệ trong graph)

### 2. Cập nhật `main_graph.py`
- Thay thế `query_parse_output` bằng `query_parse_output_graph`
- Thêm async/await support
- Sử dụng config để kết nối Neo4j

### 3. Cập nhật `config.yaml`
```yaml
GraphDB:
  neo4j_uri: "bolt://localhost"
  neo4j_port: "7687"
  neo4j_user: "neo4j"
  neo4j_password: "password"
```

## Yêu cầu hệ thống

### Dependencies mới
```bash
pip install -r requirements_graph.txt
```

### Neo4j Database
1. **Cài đặt Neo4j**:
   - Download từ https://neo4j.com/download/
   - Hoặc sử dụng Docker: `docker run -p 7474:7474 -p 7687:7687 neo4j`

2. **Cấu hình Neo4j**:
   - Username: `neo4j`
   - Password: `password` (hoặc cập nhật trong config.yaml)
   - URI: `bolt://localhost:7687`

## Hướng dẫn sử dụng

### 1. Khởi tạo dữ liệu
```bash
# Tạo graph database từ movie data
python preprocessing/graph_builder.py
```

### 2. Test integration
```bash
# Kiểm tra kết nối và basic functionality
python test_graph_integration.py
```

### 3. Chạy hệ thống chính
```bash
# Với dataset inspired
python main_graph.py --data inspired --k 10 --n 300 --begin_row 0

# Với dataset redial
python main_graph.py --data redial --k 10 --n 300 --begin_row 0
```

## Ưu điểm của Graph Database

### 1. **Rich Relationships**
- Kết nối trực tiếp giữa movies, actors, directors, genres
- Khám phá mối quan hệ phức tạp không có trong vector search

### 2. **Hybrid Retrieval**
- **Semantic**: Tìm kiếm dựa trên ý nghĩa của plot
- **Content-based**: Lọc theo thể loại, đạo diễn từ user preferences
- **Collaborative**: Tìm movies có nhiều mối quan hệ chung

### 3. **Flexible Queries**
- Có thể query phức tạp: "Action movies directed by Christopher Nolan with rating > 8.0"
- Graph traversal cho recommendations tương tự

### 4. **Scalability**
- Neo4j tối ưu cho graph queries
- Index trên các thuộc tính quan trọng

## Cấu trúc Graph

### Nodes
- **Film**: Thông tin chính về phim
- **Actor, Director, Writer**: Người tham gia
- **Genre**: Thể loại
- **Country, Language**: Metadata
- **ReleaseYear, ImdbRating**: Thông tin đánh giá

### Relationships
- **ACTED_IN, DIRECTED, WROTE**: Quan hệ người - phim
- **IN_GENRE, FROM_COUNTRY, IN_LANGUAGE**: Quan hệ metadata
- **RELEASED_IN, HAS_RATING**: Quan hệ thông tin

## Troubleshooting

### Neo4j Connection Issues
```bash
# Check Neo4j status
sudo systemctl status neo4j

# Restart Neo4j
sudo systemctl restart neo4j
```

### Memory Issues
- Tăng heap size trong neo4j.conf:
```
dbms.memory.heap.initial_size=1G
dbms.memory.heap.max_size=2G
```

### Embedding Issues
- Đảm bảo có Azure OpenAI credentials trong .env
- Hoặc comment out semantic search trong GraphRetriever

## So sánh với Vector Database

| Aspect | Vector DB | Graph DB |
|--------|-----------|----------|
| **Retrieval** | Semantic similarity only | Hybrid (semantic + content + collaborative) |
| **Relationships** | Limited | Rich, explicit |
| **Query Flexibility** | Vector similarity | Complex graph traversal |
| **Explainability** | Low | High (relationship paths) |
| **Setup Complexity** | Medium | Medium-High |
| **Performance** | Fast for similarity | Fast for relationships |

## Kết luận

Graph database cung cấp khả năng retrieval phong phú hơn bằng cách kết hợp multiple retrieval strategies và tận dụng explicit relationships giữa entities. Điều này có thể cải thiện chất lượng recommendations đáng kể.
