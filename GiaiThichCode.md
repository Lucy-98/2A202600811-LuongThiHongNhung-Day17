# Tài liệu Giải thích Chi tiết Code - Memory Systems for AI Agent

Tài liệu này giải thích chi tiết cấu trúc code, thuật toán và cách hoạt động của hệ thống bộ nhớ (memory system) mà chúng ta đã xây dựng trong thư mục `src/`.

---

## 1. Cấu trúc Tổng quan của Dự án

Dự án được thiết kế tách lớp rõ ràng giữa:
1. **Cấu hình & Provider** (`config.py`, `model_provider.py`): Quản lý biến môi trường, ngưỡng nén bộ nhớ và khởi tạo LLM.
2. **Lớp Lưu trữ Bộ nhớ** (`memory_store.py`): Phần quan trọng nhất chứa ước lượng token, lưu trữ hồ sơ người dùng dài hạn dạng markdown (`User.md`), trích xuất thông tin thực thể (fact extraction) và nén hội thoại.
3. **Lớp Agent** (`agent_baseline.py`, `agent_advanced.py`):
   - **Baseline Agent**: Bộ nhớ ngắn hạn trong phiên chat (thread), quên sạch khi sang thread mới.
   - **Advanced Agent**: Kết hợp bộ nhớ ngắn hạn, hồ sơ dài hạn (`User.md`), và cơ chế nén hội thoại (`CompactMemoryManager`).
4. **Kiểm thử & Đánh giá** (`benchmark.py`, `test_agents.py`): Đo lường các chỉ số recall, chất lượng câu trả lời, sự phình to của file nhớ và số lần nén bộ nhớ.

---

## 2. Giải thích Chi tiết từng File Code

### 2.1. `src/config.py`
Quản lý cấu hình chung của ứng dụng:
- **`LabConfig`**: Class chứa thông tin về thư mục gốc, thư mục dữ liệu, thư mục lưu trữ trạng thái (`state/`), ngưỡng kích hoạt nén bộ nhớ (`compact_threshold_tokens`), số message tối thiểu được giữ lại sau khi nén (`compact_keep_messages`), cấu hình model chính và model judge.
- **`load_config()`**: Sử dụng thư viện `dotenv` để tải các file `.env` chứa các API Key. Tự động khởi tạo thư mục lưu trữ `state/` và thiết lập các tham số mặc định: ngưỡng nén mặc định là **1000 tokens** và giữ lại tối thiểu **4 tin nhắn** gần nhất.

---

### 2.2. `src/model_provider.py`
Quản lý việc ánh xạ và khởi tạo các LLM kết nối tới các nhà cung cấp (Providers) khác nhau:
- **`normalize_provider()`**: Chuẩn hóa tên nhà cung cấp (ví dụ: chuyển từ sai chính tả `anthorpic` sang đúng là `anthropic`).
- **`build_chat_model()`**: Trả về instance của chat model tương ứng thông qua thư viện LangChain nếu chạy live (ví dụ: `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatAnthropic`, `ChatOllama`...).

---

### 2.3. `src/memory_store.py` (Lớp xử lý Bộ nhớ)
Đây là trái tim của hệ thống bộ nhớ.

#### A. Ước lượng Token (`estimate_tokens`)
- Sử dụng heuristic (quy tắc ngón tay cái): Cứ **4 ký tự** (đã cắt khoảng trắng thừa) được tính tương đương **1 token**. Phương pháp này nhanh, ổn định và không phụ thuộc vào thư viện bên ngoài khi chạy offline.

#### B. Lưu trữ Hồ sơ Người dùng (`UserProfileStore`)
- Quản lý việc đọc, ghi, và chỉnh sửa file `User.md` lưu trữ dài hạn trên ổ đĩa.
- **`path_for()`**: Chuẩn hóa `user_id` để tạo tên file hợp lệ dạng `{user_id}.md` trong thư mục `state/profiles/`.
- **`edit_text()`**: Hỗ trợ tìm kiếm (`search_text`) và thay thế (`replacement`) một đoạn text trong file hồ sơ của người dùng.
- **`parse_profile_to_dict()` & `dict_to_profile_markdown()`**: Chuyển đổi qua lại giữa nội dung Markdown (dạng danh sách `- key: value`) và kiểu dữ liệu `dict` trong Python để dễ dàng truy vấn và cập nhật.

#### C. Trích xuất Fact (`extract_profile_updates`)
- Thực hiện trích xuất các thông tin ổn định (tên, nơi ở, nghề nghiệp, đồ uống yêu thích, món ăn yêu thích, thú cưng, style trả lời) từ tin nhắn của người dùng bằng tiếng Việt.
- **Bonus - Confidence Filter & Conflict Handling**:
  - Loại bỏ các thông tin rác hoặc câu đùa (ví dụ: nếu người dùng đùa "muốn làm product manager" hay nói "mới đi Hà Nội họp", agent sẽ không lưu các thông tin này làm nơi ở hay nghề nghiệp thực tế).
  - Khi người dùng cung cấp thông tin mới đính chính cho thông tin cũ (ví dụ: "chuyển từ Huế sang Đà Nẵng"), dữ liệu cũ sẽ bị ghi đè thay vì lưu trùng lặp.

#### D. Nén bộ nhớ (`CompactMemoryManager`)
- Quản lý danh sách tin nhắn hiện tại của một thread.
- Khi tổng số tokens (tin nhắn hiện tại + tóm tắt cũ) vượt quá `threshold_tokens`:
  1. Giữ lại số lượng tin nhắn gần nhất bằng `compact_keep_messages`.
  2. Gom tất cả các tin nhắn cũ hơn lại.
  3. Sử dụng `summarize_messages()` để trích xuất các chủ đề chính (ví dụ: "Artemis III", "X-59", "WMO"...) thành một câu tóm tắt duy nhất dạng chủ đề ngắn gọn.
  4. Cập nhật đè vào trường `summary` của thread để duy trì kích thước tóm tắt siêu nhỏ (không tăng vô hạn).
  5. Tăng biến đếm `compactions`.

---

### 2.4. `src/agent_baseline.py` (Baseline Agent)
- Hoạt động dựa trên class `SessionState`.
- Chỉ lưu trữ lịch sử tin nhắn của cuộc trò chuyện trong bộ nhớ RAM hiện tại (`self.sessions`).
- Khi tính toán Prompt Token Processed, agent sẽ cộng dồn tổng số token của toàn bộ lịch sử hội thoại thô ở mỗi lượt. Vì thế, chi phí prompt tăng theo **hàm số mũ** khi cuộc hội thoại kéo dài.
- Agent này không đọc/ghi file `User.md`. Khi bắt đầu một thread mới (`thread_id` mới), agent hoàn toàn không nhớ bất kỳ thông tin nào từ trước đó.

---

### 2.5. `src/agent_advanced.py` (Advanced Agent)
Tích hợp đủ 3 tầng bộ nhớ:
1. **Bộ nhớ ngắn hạn (Short-term memory)**: Lưu tin nhắn gần nhất trong thread.
2. **Bộ nhớ hồ sơ bền vững (Persistent User.md)**:
   - Ở mỗi tin nhắn người dùng gửi đến, agent trích xuất facts bằng `extract_profile_updates()`.
   - Cập nhật facts vào file `User.md` của người dùng.
   - Khi tạo câu trả lời, agent luôn đọc `User.md` để lấy thông tin cá nhân của người dùng và điều chỉnh định dạng phản hồi (ví dụ: trả lời dạng 3 gạch đầu dòng nếu người dùng yêu cầu).
3. **Bộ nhớ nén (Compact memory)**: Sử dụng `CompactMemoryManager` để nén các tin nhắn cũ khi cuộc trò chuyện quá dài.

**Cách ước lượng Prompt context token**:
Tổng số prompt token đi vào mô hình ở mỗi turn của Advanced Agent là:
$$\text{Tokens}_{\text{User.md}} + \text{Tokens}_{\text{Summary}} + \text{Tokens}_{\text{Recent Messages}}$$
Nhờ cơ chế nén, `Tokens_Summary` luôn được giữ ở mức rất nhỏ, giúp Advanced Agent tiết kiệm đáng kể token so với Baseline Agent khi hội thoại dài.

---

### 2.6. `src/benchmark.py` (Bộ Đánh giá Benchmark)
Tự động chạy đánh giá trên 2 bộ dữ liệu:
1. **Standard Benchmark (`conversations.json`)**: 10 cuộc hội thoại ngắn, kiểm tra xem agent có nhớ thông tin người dùng ở thread mới hay không.
2. **Long-Context Stress Benchmark (`advanced_long_context.json`)**: 1 cuộc hội thoại siêu dài (nhiều lượt chat) để kích hoạt cơ chế nén.

**Cách tính điểm**:
- **`recall_points`**: Trả về `1.0` nếu câu trả lời chứa tất cả các facts mong muốn, `0.5` nếu chỉ chứa một phần, và `0.0` nếu không chứa fact nào.
- **Memory growth**: Đo lượng dung lượng file `User.md` tăng lên (bằng bytes).

---

### 2.7. `src/test_agents.py` (Bộ Test tự động)
Gồm 4 test case quan trọng để kiểm chứng hành vi:
1. **`test_user_markdown_read_write_edit`**: Kiểm tra việc đọc, ghi và sửa đổi file `User.md`.
2. **`test_compact_trigger`**: Xác minh rằng khi độ dài hội thoại vượt ngưỡng cấu hình (50 tokens), cơ chế nén được kích hoạt thành công, giảm số tin nhắn thô về đúng số lượng cấu hình (`keep_messages`).
3. **`test_cross_session_recall`**: Chứng minh Advanced Agent nhớ được tên người dùng trong thread mới (lấy từ `User.md`) còn Baseline Agent thì quên sạch.
4. **`test_compact_reduces_prompt_load_on_long_thread`**: So sánh tích lũy Prompt Tokens. Kết quả kiểm chứng lượng prompt token của Advanced Agent nhỏ hơn đáng kể so với Baseline Agent do các tin nhắn cũ đã được nén.
