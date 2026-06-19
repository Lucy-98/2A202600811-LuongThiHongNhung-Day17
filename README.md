# Chào mừng các bạn đến với Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

Repo này được chia thành ba phần rõ ràng:

- `src/`: bản scaffold dành cho sinh viên, chứa pseudocode và TODO để hoàn thiện
- `data/`: dữ liệu benchmark ở root để dùng cho cả benchmark chuẩn và stress benchmark

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Cách dùng repo này

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Sau đó làm việc trực tiếp với `src/` và `data/` ở root repo.

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.

---

# Báo cáo Phân tích Kết quả & Tính năng Nâng cao (Bonus)

## 1. Kết quả Benchmark thực tế

### Standard Benchmark (10 sessions thông thường)
| Agent          | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline Agent | 1631              | 13792                   | 0.07                 | 0.07             | 0                     | 0           |
| Advanced Agent | 3096              | 23545                   | 0.89                 | 0.89             | 189                   | 0           |

### Long-Context Stress Benchmark (Stress test hội thoại dài)
| Agent          | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline Agent | 472               | 21594                   | 0.00                 | 0.00             | 0                     | 0           |
| Advanced Agent | 632               | 10113                   | 0.83                 | 0.83             | 95                    | 3           |

---

## 2. Phân tích Trade-off & Hiểu biết Kỹ thuật

### A. Vì sao Advanced Agent có recall tốt hơn Baseline Agent?
- **Baseline Agent** chỉ có bộ nhớ ngắn hạn lưu trong luồng (within-session). Khi truy vấn ở thread mới (cross-session), agent hoàn toàn không có thông tin từ thread cũ, dẫn tới tỷ lệ recall gần như bằng 0 (0.00 - 0.07).
- **Advanced Agent** liên tục trích xuất thông tin người dùng từ hội thoại và bền vững hóa vào file `User.md`. Khi bắt đầu lượt chat mới, agent đọc `User.md` để lấy ngữ cảnh dài hạn, giúp tăng vọt tỷ lệ recall lên **0.83 - 0.89**.

### B. Vì sao Advanced Agent tốn token hơn ở hội thoại ngắn?
- Do phải kéo theo file `User.md` trong prompt context ở mỗi lượt giao tiếp.
- Do phản hồi của Advanced Agent được cá nhân hóa sâu và tuân thủ định dạng cấu trúc phức tạp (như bullet points theo preference của người dùng), dẫn tới lượng output tokens sinh ra nhiều hơn (3096 so với 1631 trong Standard Benchmark).
- *Trade-off*: Tăng độ chính xác (Recall) đổi lại chi phí vận hành ban đầu cao hơn một chút.

### C. Lợi ích vượt trội của Compact Memory ở hội thoại dài
- Khi hội thoại kéo dài (Stress Benchmark), Baseline Agent giữ toàn bộ lịch sử thô làm prompt context phình to theo hàm số mũ (21,594 prompt tokens).
- Advanced Agent kích hoạt **Compact Memory**, tự động nén (compact) các message cũ vượt ngưỡng thành một dòng tóm tắt chủ đề cô đọng, chỉ giữ lại các lượt chat mới nhất trong bộ nhớ ngắn hạn. Nhờ đó, lượng prompt tokens cần xử lý giảm hơn một nửa xuống còn **10,113 tokens (tiết kiệm ~53% chi phí)** mà vẫn giữ nguyên khả năng trả lời chính xác.

### D. Tốc độ phình của Memory File & Rủi ro
- File memory (`User.md`) tăng trưởng trung bình từ **95 - 189 bytes** sau mỗi chuỗi hội thoại dài.
- *Rủi ro*: Nếu lưu trữ dạng text thô vô hạn, file profile sẽ phình to và tăng chi phí đọc ở mọi lượt chat mới. Trong thực tế, cần thiết lập cơ chế **Memory Decay** (giảm độ ưu tiên/xóa bớt facts ít dùng) hoặc **Vector DB** để tìm kiếm ngữ nghĩa thay vì đọc toàn bộ file.

---

## 3. Tính năng Nâng cao đã Triển khai (Bonus)

Chúng tôi đã triển khai hai cơ chế quan trọng giúp nâng chất lượng agent lên mức tối đa:

1. **Conflict Handling (Xử lý xung đột thông tin)**:
   - *Mô tả*: Khi người dùng cập nhật thông tin mới (ví dụ: chuyển nơi ở từ Huế sang Đà Nẵng, chuyển nghề từ Backend sang MLOps), agent tự động phân tích cấu trúc profile cũ, ghi đè giá trị mới và loại bỏ fact cũ mâu thuẫn khỏi `User.md`.
   - *Lợi ích*: Tránh việc lưu trữ đồng thời hai thông tin trái ngược làm nhiễu mô hình hoặc sinh câu trả lời sai.

2. **Confidence Filter (Bộ lọc nhiễu hội thoại)**:
   - *Mô tả*: Agent bỏ qua các câu nói đùa hay thông tin tạm thời (như địa danh đi họp tạm, câu đùa về việc chuyển nghề làm PM) để tránh ghi sai vào bộ nhớ dài hạn.
   - *Lợi ích*: Giữ cho file `User.md` cực kỳ sạch và chỉ chứa các fact bền vững có độ tin cậy cao.

