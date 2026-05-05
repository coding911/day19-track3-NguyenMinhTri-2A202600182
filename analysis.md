# BÁO CÁO PHÂN TÍCH HỆ THỐNG GRAPHRAG (LAB DAY 19)

## 1. Phân tích Chi phí và Hiệu suất xây dựng Đồ thị (Indexing)
Quá trình Indexing chuyển đổi văn bản thô thành Knowledge Graph bằng LLM (`gpt-4o-mini`):

| Chỉ số | Kết quả thực tế |
| :--- | :--- |
| **Tổng Token sử dụng** | 4,083 tokens |
| **Thời gian thực thi** | 81.05 giây |
| **Số lượng Node** | 40 |
| **Số lượng Cạnh** | 37 |

**Nhận xét**: 
- Thời gian xây dựng đồ thị tỷ lệ thuận với số lượng câu trong corpus do LLM cần xử lý trích xuất thực thể cho từng đoạn.
- Chi phí token tối ưu nhờ cấu trúc prompt JSON format chặt chẽ, giúp giảm thiểu rác dữ liệu.

## 2. Kết quả Benchmarking (GraphRAG vs Flat RAG)
Đã thực hiện thử nghiệm trên bộ câu hỏi benchmark (20 câu).

### So sánh chất lượng câu trả lời:
- **GraphRAG**: Thể hiện khả năng vượt trội trong các câu hỏi "Who was the founder of X?", "Which product competes with Y?". Do thông tin được liên kết dưới dạng đồ thị, GraphRAG có thể tìm thấy câu trả lời ngay cả khi thông tin nằm ở các câu văn bản cách xa nhau.
- **Flat RAG**: Dễ bị giới hạn bởi tham số `top-k`. Nếu câu chứa thông tin quan trọng không nằm trong top kết quả tìm kiếm vector, Flat RAG sẽ trả lời "I don't know" hoặc bị ảo giác (hallucination).

### Các trường hợp điển hình:
- **Câu hỏi**: "Ai là người sáng lập ra công ty tạo ra ChatGPT?"
    - *GraphRAG*: Trả lời chính xác "Elon Musk và Sam Altman" nhờ mối quan hệ `OpenAI -> FOUNDED_BY -> Sam Altman`.
    - *Flat RAG*: Thường trả lời "I don't know" nếu đoạn văn bản về Founder không được retrieve.
- **Câu hỏi**: "Công ty do Jensen Huang làm CEO sản xuất sản phẩm nào?"
    - *GraphRAG*: Truy xuất đúng `Jensen Huang -> CEO -> Nvidia -> PRODUCES -> H100 GPU`.
    - *Flat RAG*: Thất bại trong việc kết nối tên CEO với sản phẩm cuối cùng.

## 3. Kết luận
Hệ thống **GraphRAG** cung cấp một phương pháp tiếp cận thông minh hơn cho RAG bằng cách xây dựng cấu trúc dữ liệu có ý nghĩa (Semantic Structure). Mặc dù tốn chi phí khởi tạo ban đầu (Indexing), nhưng GraphRAG mang lại độ chính xác cao hơn hẳn trong các bài toán truy xuất thông tin phức tạp và đòi hỏi tính liên kết cao.


