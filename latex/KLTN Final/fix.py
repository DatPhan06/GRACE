import codecs

with codecs.open('Chap5/chap5.tex', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

new_lines = lines[:96]
new_text = """Cơ chế "kiểm định" (Critic) và "thỏa hiệp" (Relaxation) cung cấp giải pháp cho các truy vấn chứa yêu cầu mâu thuẫn. Khi không có ứng viên nào đáp ứng toàn bộ bộ tiêu chí, hệ thống sẽ tự động nới lỏng các ràng buộc thứ cấp theo mức ưu tiên để duy trì tính liên tục thay vì phản hồi lỗi. 

Bên cạnh đó, quy trình Xếp hạng hai giai đoạn (Decoupled Reranking) tích hợp với mô hình Cross-Encoder đã giảm tải khối lượng tính toán trực tiếp cho mô hình ngôn ngữ lớn sinh văn bản. Kiến trúc đa tác tử đảm bảo đạt được tính chính xác cao đồng thời giữ vững độ trễ (latency) ở mức cho phép cho một hệ thống tương tác thực tế.

\\begin{table}[H]
    \\caption{Đánh giá nội bộ: So sánh hiệu năng giữa GRACE (Dòng chảy tĩnh) và ARGOS (Hệ thống Đa tác tử) trên tập mẫu 20\\% có tính ý nghĩa thống kê ($p < 0.05$).}
    \\label{tab:argos_internal}
    \\centering
    \\begin{tabular}{lcccccccc}
        \\toprule
        \\multirow{2}{*}{\\textbf{Kiến trúc}} & \\multicolumn{4}{c}{\\textbf{ReDial (20\\% Sample)}} & \\multicolumn{4}{c}{\\textbf{INSPIRED (20\\% Sample)}}                                                                   \\\\
        \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}
                                            & \\textbf{K=1}                                      & \\textbf{K=5}                                        & \\textbf{K=10}  & \\textbf{K=50}
                                            & \\textbf{K=1}                                      & \\textbf{K=5}                                        & \\textbf{K=10}  & \\textbf{K=50}                                  \\\\
        \\midrule
        GRACE (Baseline)                    & 0.063                                             & 0.149                                               & 0.300          & 0.525          & 0.115 & 0.181 & 0.305 & 0.444 \\\\
        \\textbf{ARGOS (Đa tác tử)}          & \\textbf{0.105}                                    & \\textbf{0.225}                                      & \\textbf{0.385} & \\textbf{0.625}
                                            & \\textbf{0.145}                                    & \\textbf{0.235}                                      & \\textbf{0.365} & \\textbf{0.515}                                 \\\\
        \\bottomrule
    \\end{tabular}
\\end{table}

\\section{Nghiên cứu tình huống (Case Studies)}

Bên cạnh các đánh giá định lượng, các nghiên cứu tình huống sau đây sẽ làm rõ cơ chế tương tác giữa các tác tử trong hệ thống ARGOS nhằm giải quyết các trường hợp mà mô hình tĩnh (GRACE) gặp hạn chế.

\\subsection{Đa truy vấn: Khám phá các mối liên hệ ẩn (Hidden Relational Gems)}
\\textbf{Truy vấn người dùng:} \\textit{"Tôi muốn xem phim gì đó giống John Wick nhưng bối cảnh phải ở Châu Á và có màu sắc kiếm hiệp hoài cổ."}
\\begin{itemize}
    \\item \\textbf{Hệ thống tĩnh:} Thực hiện một truy vấn duy nhất với cụm từ khóa "John Wick Asia martial arts". Hệ quả là không gian vector bị dịch chuyển về hướng các bộ phim hành động hiện đại của Châu Á (ví dụ: \\textit{The Raid}), bỏ qua yếu tố "kiếm hiệp hoài cổ".
    \\item \\textbf{Hệ thống ARGOS (Tác tử Hồ sơ):} Tác tử tự động phân rã ý định thành 3 truy vấn con độc lập: (1) \\textit{"Fast-paced gun-fu like John Wick"}, (2) \\textit{"Wuxia aesthetic martial arts movie"}, và (3) \\textit{"Asian setting action choreography"}. Quá trình tìm kiếm song song cho phép hệ thống xác định điểm giao thoa và đề xuất các tác phẩm phù hợp như \\textit{Shadow (2018)} của Trương Nghệ Mưu hoặc \\textit{Blade of the Immortal}. Cơ chế này đóng vai trò quan trọng trong việc gia tăng chỉ số Recall.
\\end{itemize}

\\subsection{Tác tử Đồ thị (Graph Agent): Nội suy thông tin từ dữ liệu khuyết thiếu}
\\textbf{Truy vấn người dùng:} \\textit{"Tìm phim của hãng A24 sản xuất nhé."}
\\begin{itemize}
    \\item \\textbf{Hệ thống tĩnh:} Trình tạo mã Cypher sinh lệnh \\texttt{MATCH (f:Film \\{studio: "A24"\\})}. Tuy nhiên, schema cơ sở dữ liệu không lưu trữ thuộc tính \\texttt{studio}. Lệnh truy vấn trả về rỗng và hệ thống không thể xử lý tiếp.
    \\item \\textbf{Hệ thống ARGOS (Tác tử Đồ thị):} Tác tử thực thi mô hình ReAct. Lượt 1: Truy vấn \\texttt{studio} không thành công. Lượt 2 (Tư duy logic): Tác tử nhận diện sự thiếu hụt của trường dữ liệu và thiết lập chiến lược nội suy: \\textit{"A24 thường hợp tác với các đạo diễn đặc trưng như Ari Aster hoặc Robert Eggers"}. Lượt 3 (Hành động): Tác tử sinh lệnh Cypher mới tìm kiếm phim thông qua danh sách đạo diễn này. Kết quả trả về \\textit{Hereditary} và \\textit{The Lighthouse}. Hệ thống chứng minh khả năng nội suy logic để vượt qua sự thiếu hụt của cơ sở dữ liệu, qua đó nâng cao tính bền vững (Robustness).
\\end{itemize}

\\subsection{Tác tử Kiểm định (Critic Agent): Cơ chế triệt tiêu ảo giác}
\\textbf{Truy vấn người dùng:} \\textit{"Phim hoạt hình nhẹ nhàng cho trẻ em, có robot giống Wall-E nhé."}
\\begin{itemize}
    \\item \\textbf{Hệ thống tĩnh:} Từ khóa "robot" có thể khiến cơ chế truy xuất ngữ nghĩa vô tình đưa \\textit{The Terminator} vào danh sách ứng viên. Do thiếu sự kiểm soát độc lập, mô hình xếp hạng có thể ưu tiên ứng viên này do điểm tương đồng vector cao.
    \\item \\textbf{Hệ thống ARGOS (Tác tử Kiểm định):} Tác tử nhận diện mâu thuẫn logic: \\textit{"Yêu cầu nhấn mạnh đối tượng 'trẻ em', trong khi The Terminator được dán nhãn R. Vi phạm Ràng buộc cứng (Hard Constraint) về đối tượng khán giả"}. Tác tử ngay lập tức thanh lọc kết quả này, chỉ giữ lại các lựa chọn an toàn như \\textit{Big Hero 6} hoặc \\textit{Next Gen}, đảm bảo tính chuẩn xác (Precision) của hệ thống.
\\end{itemize}

\\subsection{Tác tử Thỏa hiệp (Relaxation Agent): Điều chỉnh ý định truy vấn}
\\textbf{Truy vấn người dùng:} \\textit{"Tìm phim kinh dị thuần túy của đạo diễn Christopher Nolan."}
\\begin{itemize}
    \\item \\textbf{Hệ thống tĩnh:} Do Christopher Nolan chưa từng đạo diễn phim kinh dị (Horror) thuần túy, hệ thống trả về kết quả rỗng hoặc đề xuất một phim hành động ngẫu nhiên mà không kèm theo giải thích hợp lý.
    \\item \\textbf{Hệ thống ARGOS (Tác tử Thỏa hiệp):} Tác tử Kiểm định báo cáo kết quả rỗng. Tác tử Thỏa hiệp kích hoạt cơ chế nới lỏng: \\textit{"Đặc trưng 'Christopher Nolan' là sở thích cốt lõi (Core intent) không thể thay thế. Có thể nới lỏng giới hạn 'Horror' sang 'Thriller' (giật gân), vốn là thế mạnh của vị đạo diễn này"}. Hệ thống tái cấu trúc truy vấn và đề xuất các bộ phim như \\textit{Memento} hay \\textit{The Prestige}, kèm theo lời diễn giải: \\textit{"Mặc dù Nolan không đạo diễn phim kinh dị, nhưng đây là những tác phẩm giật gân có yếu tố u tối tiêu biểu của ông..."}. Cơ chế này đảm bảo duy trì mạch hội thoại liên tục và cung cấp trải nghiệm tối ưu cho người dùng.
\\end{itemize}

\\section*{Kết luận chương}

Chương 5 trình bày các kết quả phân tích định lượng và định tính về hiệu suất của hệ thống ARGOS. Bằng việc thực nghiệm trên hai bộ dữ liệu ReDial và INSPIRED, hệ thống thể hiện sự cải thiện rõ rệt so với cả mô hình tiền đề GRACE và các mô hình yêu cầu tinh chỉnh tham số phức tạp (training-based models). Các nghiên cứu tình huống thực tiễn đã minh họa cụ thể luồng xử lý và khả năng tự động phân giải, suy luận đồ thị, và quản lý ràng buộc thông minh. Các kết quả thực chứng này đóng vai trò cơ sở để tiến tới tổng kết khóa luận và đề xuất những định hướng nghiên cứu sâu hơn trong chương kế tiếp.
"""
new_lines.append(new_text)

with codecs.open('Chap5/chap5.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
